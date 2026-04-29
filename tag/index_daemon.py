"""
tag/index_daemon.py
===================
AXIOM index daemon.

The daemon subscribes to canonical bus events, updates lightweight phase and
recipe state, and queues offline work. It does not update structural weights
directly; offline/ owns weight artifacts.
"""

from __future__ import annotations

import asyncio
import contextlib
import heapq
import json
import logging
import mmap
import os
import struct
import sys
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signal_kernel.contracts import (
    FetchAnomalyEvent,
    NewTopologyHintEvent,
    PhaseTransitionEvent,
    RecipeHealthEvent,
    RecipeStaleEvent,
    SignalExtractedEvent,
    StoreHealthEvent,
    SurpriseEvent,
    WeightsUpdatedEvent,
    new_run_id,
)
from tag.config import load_config
from tag.runtime_paths import RuntimePathResolver


logger = logging.getLogger(__name__)


# Shared with daemons/daemon_common.h:
# <BBHfIfQff> = phase, flags, generation, confidence, observation_count,
# surprise_rate, last_updated_unix, npi, recipe_yield.
PHASE_SLOT = struct.Struct("<BBHfIfQff")
PHASE_HEADER = struct.Struct("<4sBBH")
PHASE_MAGIC = b"AXPS"
PHASE_HEADER_BYTES = PHASE_HEADER.size
PHASE_SLOT_BYTES = 32
PHASE_COLD = 1
PHASE_LEARNING = 2
PHASE_KNOWN = 3
GRADIENT_PRIORITY_HIGH = 0
GRADIENT_PRIORITY_MEDIUM = 1
GRADIENT_PRIORITY_LOW = 2
RECIPE_DRAFT_CONFIDENCE_MIN = 0.60
DEFAULT_GRADIENT_BATCH_SIZE = 32
DEFAULT_GRADIENT_FLUSH_INTERVAL_S = 60.0
DEFAULT_GRADIENT_ITEM_TTL_S = 600.0
DEFAULT_PHASE_SCAN_INTERVAL_S = 30.0
DEFAULT_HEALTH_LOG_INTERVAL_S = 120.0
DEFAULT_RECIPE_SAVE_INTERVAL_S = 30.0
DEFAULT_GRADIENT_PURGE_INTERVAL_S = 300.0
DEFAULT_GRADIENT_QUEUE_MAX_SIZE = 1000
OFFLINE_QUEUE_DIR = "offline_queue"
CIRCUIT_BREAKER_BACKOFF_S = 30.0
CIRCUIT_BREAKER_FAILURES = 3


@dataclass
class PhaseState:
    phase: int = PHASE_COLD
    flags: int = 0
    generation: int = 1
    confidence: float = 0.0
    observation_count: int = 0
    surprise_rate: float = 0.0
    updated_unix: int = field(default_factory=lambda: int(time.time()))
    npi: float = 0.0
    recipe_yield: float = 0.0

    def pack(self) -> bytes:
        return PHASE_SLOT.pack(
            self.phase,
            self.flags,
            self.generation,
            self.confidence,
            self.observation_count,
            self.surprise_rate,
            self.updated_unix,
            self.npi,
            self.recipe_yield,
        )


class PhaseStore:
    """Fixed-slot mmap store for topology phase state."""

    def __init__(self, path: Path, slots: int = 4096) -> None:
        self.path = path
        self.slots = slots
        self._active_classes: Set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.has_header = self._detect_header()
        self.header_bytes = PHASE_HEADER_BYTES if self.has_header else 0
        required = self.header_bytes + slots * PHASE_SLOT_BYTES
        if not self.path.exists():
            self.has_header = True
            self.header_bytes = PHASE_HEADER_BYTES
            required = self.header_bytes + slots * PHASE_SLOT_BYTES
            with self.path.open("wb") as f:
                f.write(PHASE_HEADER.pack(PHASE_MAGIC, 1, min(255, slots), 0))
                f.truncate(required)
        elif self.path.stat().st_size < required:
            with self.path.open("r+b") as f:
                if self.has_header and self.path.stat().st_size < PHASE_HEADER_BYTES:
                    f.write(PHASE_HEADER.pack(PHASE_MAGIC, 1, min(255, slots), 0))
                f.truncate(required)
        self._file = self.path.open("r+b")
        self._mmap = mmap.mmap(self._file.fileno(), required)

    def close(self) -> None:
        self._mmap.flush()
        self._mmap.close()
        self._file.close()

    def read(self, topology_class: str) -> PhaseState:
        slot = self._slot(topology_class)
        offset = self._offset(slot)
        raw = self._mmap[offset : offset + PHASE_SLOT.size]
        vals = PHASE_SLOT.unpack(raw)
        phase = vals[0]
        if phase == 0:
            return PhaseState()
        return PhaseState(
            phase=phase,
            flags=vals[1],
            generation=vals[2],
            confidence=vals[3],
            observation_count=vals[4],
            surprise_rate=vals[5],
            updated_unix=vals[6],
            npi=vals[7],
            recipe_yield=vals[8],
        )

    def write(self, topology_class: str, state: PhaseState) -> None:
        slot = self._slot(topology_class)
        offset = self._offset(slot)
        state.generation = (state.generation + 1) & 0xFFFF
        state.updated_unix = int(time.time())
        self._mmap[offset : offset + PHASE_SLOT_BYTES] = state.pack()
        self._mmap.flush()
        self._active_classes.add(topology_class)

    def read_all_active(self) -> Dict[str, PhaseState]:
        return {topology_class: self.read(topology_class) for topology_class in sorted(self._active_classes)}

    def phase_summary(self) -> Dict[str, int]:
        counts = {"cold": 0, "learning": 0, "known": 0, "active_classes": len(self._active_classes)}
        for state in self.read_all_active().values():
            if state.phase == PHASE_KNOWN:
                counts["known"] += 1
            elif state.phase == PHASE_LEARNING:
                counts["learning"] += 1
            else:
                counts["cold"] += 1
        return counts

    def _offset(self, slot: int) -> int:
        return self.header_bytes + slot * PHASE_SLOT_BYTES

    def _slot(self, topology_class: str) -> int:
        return self._hash(topology_class) % self.slots

    def _detect_header(self) -> bool:
        try:
            with self.path.open("rb") as handle:
                return handle.read(4) == PHASE_MAGIC
        except OSError:
            return False

    @staticmethod
    def _hash(topology_class: str) -> int:
        import zlib

        return zlib.crc32(topology_class.encode("utf-8")) & 0xFFFFFFFF


@dataclass
class DaemonStats:
    signal_events: int = 0
    surprise_events: int = 0
    fetch_anomalies: int = 0
    recipe_stale: int = 0
    recipe_health: int = 0
    store_health: int = 0
    weights_updated: int = 0
    phase_transitions: int = 0
    queued_gradients: int = 0
    dropped_gradients: int = 0
    drafted_recipes: int = 0
    dispatched_events: int = 0
    dispatch_errors: int = 0
    expired_gradients: int = 0
    gradient_batches_dispatched: int = 0
    circuit_breaker_trips: int = 0
    phase_promotions: int = 0
    uptime_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass(order=True)
class GradientWorkItem:
    priority: int
    created_unix: int
    sequence: int
    payload: Dict[str, Any] = field(compare=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "priority": self.priority,
            "created_unix": self.created_unix,
            "sequence": self.sequence,
            "payload": self.payload,
        }

    def is_expired(self, *, now: Optional[float] = None, ttl_seconds: float = DEFAULT_GRADIENT_ITEM_TTL_S) -> bool:
        return (now or time.time()) - float(self.created_unix) > ttl_seconds


@dataclass
class RecipeAction:
    action: str
    topology_class: str
    reason: str
    confidence: float = 0.0
    created_unix: int = field(default_factory=lambda: int(time.time()))

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ZoneDraft:
    topology_class: str
    zone_type: str
    selector: str
    confidence: float
    sample_count: int
    source: str

    def to_recipe_step(self) -> Dict[str, Any]:
        return {
            "kind": "extract_zone",
            "topology_class": self.topology_class,
            "zone_type": self.zone_type,
            "selector": self.selector,
            "confidence": self.confidence,
            "sample_count": self.sample_count,
            "source": self.source,
        }


@dataclass
class RecipeRegistry:
    path: Path
    recipes: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    dirty: bool = False

    def load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.recipes = {}
            return
        try:
            raw = self.path.read_bytes().rstrip(b"\x00")
            if raw.startswith(b"AXRR"):
                self.recipes = {}
            else:
                self.recipes = json.loads(raw.decode("utf-8")) if raw else {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            self.recipes = {}
        self.dirty = False

    def add_drafts(self, topology_class: str, drafts: Iterable[ZoneDraft]) -> int:
        bucket = self.recipes.setdefault(topology_class, [])
        before = len(bucket)
        seen = {(item.get("selector"), item.get("zone_type")) for item in bucket}
        for draft in drafts:
            step = draft.to_recipe_step()
            key = (step["selector"], step["zone_type"])
            if key in seen:
                continue
            bucket.append(step)
            seen.add(key)
        added = len(bucket) - before
        if added:
            self.dirty = True
        return added

    def atomic_save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.recipes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp = self.path.with_suffix(self.path.suffix + f".{uuid.uuid4().hex}.tmp")
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)
        self.dirty = False

    def save_if_dirty(self) -> bool:
        if not self.dirty:
            return False
        self.atomic_save()
        return True

    def digest(self) -> str:
        payload = json.dumps(self.recipes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"

    def recipe_count(self) -> int:
        return sum(len(items) for items in self.recipes.values())

    def get_all_classes(self) -> List[str]:
        return sorted(self.recipes)


class IndexDaemon:
    """
    Event-driven coordinator for phase state and offline work queues.
    """

    def __init__(self, *, store_dir: Path = Path("store")) -> None:
        self.config = load_config()
        self.store_dir = (
            RuntimePathResolver(config=self.config).resolve().store_dir
            if Path(store_dir) == Path("store")
            else Path(store_dir)
        )
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.phase_store = PhaseStore(self.store_dir / "phase_states.mmap")
        self.recipe_registry = RecipeRegistry(self.store_dir / "recipe_registry.mmap")
        self.recipe_registry.load()
        self.stats = DaemonStats()
        self.gradient_capacity = self.config.int("index_daemon.queue_max_size", DEFAULT_GRADIENT_QUEUE_MAX_SIZE, low=1, high=1_000_000)
        self.gradient_batch_size = self.config.int("index_daemon.gradient_batch_size", DEFAULT_GRADIENT_BATCH_SIZE, low=1, high=10_000)
        self.gradient_flush_interval_s = self.config.float("index_daemon.gradient_flush_interval_seconds", DEFAULT_GRADIENT_FLUSH_INTERVAL_S, low=0.01)
        self.gradient_item_ttl_s = self.config.float("index_daemon.gradient_item_ttl_seconds", DEFAULT_GRADIENT_ITEM_TTL_S, low=1.0)
        self.phase_scan_interval_s = self.config.float("index_daemon.phase_scan_interval_seconds", DEFAULT_PHASE_SCAN_INTERVAL_S, low=0.01)
        self.health_log_interval_s = self.config.float("index_daemon.health_log_interval_seconds", DEFAULT_HEALTH_LOG_INTERVAL_S, low=0.01)
        self.recipe_save_interval_s = self.config.float("index_daemon.recipe_save_interval_seconds", DEFAULT_RECIPE_SAVE_INTERVAL_S, low=0.01)
        self.gradient_purge_interval_s = self.config.float("index_daemon.gradient_purge_interval_seconds", DEFAULT_GRADIENT_PURGE_INTERVAL_S, low=0.01)
        self.gradient_heap: List[GradientWorkItem] = []
        self._gradient_sequence = 0
        self.recipe_actions: List[RecipeAction] = []
        self.health_events: List[StoreHealthEvent] = []
        self.zone_drafts: List[ZoneDraft] = []
        self._running = False
        self._tasks: List[asyncio.Task[Any]] = []
        self._start_time = time.monotonic()
        self._offline_queue_path = self.store_dir / OFFLINE_QUEUE_DIR
        self._offline_queue_path.mkdir(parents=True, exist_ok=True)
        self._cb_failures = 0
        self._cb_open_until = 0.0

    async def dispatch(self, event: Any) -> None:
        self.stats.dispatched_events += 1
        try:
            if isinstance(event, SignalExtractedEvent):
                await self.handle_signal_extracted(event)
            elif isinstance(event, SurpriseEvent):
                await self.handle_surprise(event)
            elif isinstance(event, FetchAnomalyEvent):
                await self.handle_fetch_anomaly(event)
            elif isinstance(event, RecipeStaleEvent):
                await self.handle_recipe_stale(event)
            elif isinstance(event, RecipeHealthEvent):
                await self.handle_recipe_health(event)
            elif isinstance(event, StoreHealthEvent):
                await self.handle_store_health(event)
            elif isinstance(event, WeightsUpdatedEvent):
                await self.handle_weights_updated(event)
            elif isinstance(event, PhaseTransitionEvent):
                await self.handle_phase_transition(event)
            elif isinstance(event, NewTopologyHintEvent):
                await self.handle_new_topology_hint(event)
            else:
                raise TypeError(f"unsupported event type: {type(event).__name__}")
        except Exception:
            self.stats.dispatch_errors += 1
            raise

    async def handle_signal_extracted(self, event: SignalExtractedEvent) -> None:
        self.stats.signal_events += 1
        state = self.phase_store.read(event.topology_class)
        old_phase = state.phase
        state.observation_count += 1
        state.confidence = min(1.0, max(state.confidence, event.signal_density))
        state.npi = min(1.0, max(state.npi, event.signal_density))
        if event.byte_count > 0:
            state.recipe_yield = min(1.0, max(state.recipe_yield, event.byte_count / max(event.byte_count * 4, 1)))
        state.updated_unix = int(time.time())
        drafts = self.discover_signal_zones(event)
        if drafts:
            self.zone_drafts.extend(drafts)
            added = self.recipe_registry.add_drafts(event.topology_class, drafts)
            if added:
                self.stats.drafted_recipes += added
                self.recipe_registry.atomic_save()
        if state.phase == PHASE_COLD and state.observation_count >= 50 and state.confidence >= 0.70 and state.surprise_rate < 0.20:
            state.phase = PHASE_LEARNING
        if (
            state.phase == PHASE_LEARNING
            and state.observation_count >= 200
            and state.confidence >= 0.85
            and state.surprise_rate < 0.05
            and state.npi >= 0.70
            and state.recipe_yield >= 0.005
        ):
            state.phase = PHASE_KNOWN
        if state.phase != old_phase:
            self.stats.phase_promotions += 1
        self.phase_store.write(event.topology_class, state)

    async def handle_surprise(self, event: SurpriseEvent) -> None:
        self.stats.surprise_events += 1
        state = self.phase_store.read(event.topology_class)
        state.observation_count += 1
        surprise_events = round(state.surprise_rate * max(state.observation_count - 1, 0)) + 1
        state.surprise_rate = min(1.0, surprise_events / max(state.observation_count, 1))
        state.confidence = max(0.0, state.confidence - event.surprise_score)
        if state.phase == PHASE_KNOWN and event.dissolve_triggered:
            state.phase = PHASE_LEARNING
        self.phase_store.write(event.topology_class, state)
        priority = GRADIENT_PRIORITY_HIGH if event.dissolve_triggered or event.surprise_score >= 0.7 else GRADIENT_PRIORITY_MEDIUM
        await self._queue_gradient({"type": "surprise", "topology_class": event.topology_class, "score": event.surprise_score}, priority=priority)

    async def handle_fetch_anomaly(self, event: FetchAnomalyEvent) -> None:
        self.stats.fetch_anomalies += 1
        priority = GRADIENT_PRIORITY_HIGH if event.status_code in {403, 429, 500, 502, 503} else GRADIENT_PRIORITY_MEDIUM
        await self._queue_gradient({"type": "fetch_anomaly", "url": event.url, "anomaly_type": event.anomaly_type}, priority=priority)

    async def handle_recipe_stale(self, event: RecipeStaleEvent) -> None:
        self.stats.recipe_stale += 1
        self.recipe_actions.append(RecipeAction("recompile", event.topology_class, event.reason, event.confidence))

    async def handle_recipe_health(self, event: RecipeHealthEvent) -> None:
        self.stats.recipe_health += 1
        if event.stale:
            self.recipe_actions.append(RecipeAction("inspect", event.topology_class, f"empty_rate={event.empty_rate:.3f}", event.empty_rate))

    async def handle_store_health(self, event: StoreHealthEvent) -> None:
        self.stats.store_health += 1
        self.health_events.append(event)
        if event.critical:
            await self._queue_gradient(
                {
                    "type": "store_health",
                    "store_file": event.store_file,
                    "status": event.status,
                    "detail": event.detail,
                },
                priority=GRADIENT_PRIORITY_HIGH,
            )

    async def handle_weights_updated(self, event: WeightsUpdatedEvent) -> None:
        self.stats.weights_updated += 1

    async def handle_phase_transition(self, event: PhaseTransitionEvent) -> None:
        self.stats.phase_transitions += 1
        state = self.phase_store.read(event.topology_class)
        state.phase = event.to_phase
        state.confidence = event.confidence
        state.updated_unix = int(time.time())
        self.phase_store.write(event.topology_class, state)

    async def handle_new_topology_hint(self, event: NewTopologyHintEvent) -> None:
        topology_class = f"{event.suggested_parent_class}_{event.domain}".upper().replace(".", "_").replace("-", "_")[:96]
        draft = ZoneDraft(
            topology_class=topology_class,
            zone_type="prose",
            selector="article, main, [data-content]",
            confidence=0.65 if event.mdl_supports_split else 0.50,
            sample_count=max(1, event.evidence_count),
            source="topology_hint",
        )
        self.zone_drafts.append(draft)
        added = self.recipe_registry.add_drafts(topology_class, [draft])
        if added:
            self.stats.drafted_recipes += added
            self.recipe_registry.save_if_dirty()

    async def _queue_gradient(self, item: Dict[str, Any], *, priority: int = GRADIENT_PRIORITY_LOW) -> None:
        self._gradient_sequence += 1
        work = GradientWorkItem(priority=priority, created_unix=int(time.time()), sequence=self._gradient_sequence, payload=item)
        if len(self.gradient_heap) >= self.gradient_capacity:
            worst_idx = max(range(len(self.gradient_heap)), key=lambda i: (self.gradient_heap[i].priority, -self.gradient_heap[i].sequence))
            worst = self.gradient_heap[worst_idx]
            if worst.priority <= work.priority:
                self.stats.dropped_gradients += 1
                return
            self.gradient_heap.pop(worst_idx)
            heapq.heapify(self.gradient_heap)
            self.stats.dropped_gradients += 1
        heapq.heappush(self.gradient_heap, work)
        self.stats.queued_gradients = len(self.gradient_heap)

    def pop_gradient_batch(self, limit: int = 32) -> List[Dict[str, Any]]:
        batch = [item.to_dict() for item in self._pop_gradient_items(limit)]
        self.stats.queued_gradients = len(self.gradient_heap)
        return batch

    def _pop_gradient_items(self, limit: int) -> List[GradientWorkItem]:
        batch: List[GradientWorkItem] = []
        now = time.time()
        for _ in range(max(0, limit)):
            while self.gradient_heap:
                item = heapq.heappop(self.gradient_heap)
                if item.is_expired(now=now, ttl_seconds=self.gradient_item_ttl_s):
                    self.stats.expired_gradients += 1
                    continue
                batch.append(item)
                break
            if not self.gradient_heap and len(batch) < limit:
                break
        self.stats.queued_gradients = len(self.gradient_heap)
        return batch

    def _requeue_gradient_items(self, items: Iterable[GradientWorkItem]) -> None:
        for item in items:
            if len(self.gradient_heap) < self.gradient_capacity:
                heapq.heappush(self.gradient_heap, item)
        self.stats.queued_gradients = len(self.gradient_heap)

    async def _dispatch_gradient_batch(self) -> int:
        if time.monotonic() < self._cb_open_until:
            return 0
        items = self._pop_gradient_items(self.gradient_batch_size)
        if not items:
            return 0
        batch = [item.to_dict() for item in items]
        self._offline_queue_path.mkdir(parents=True, exist_ok=True)
        target = self._offline_queue_path / f"gradient-{int(time.time() * 1000)}-{uuid.uuid4().hex}.jsonl"
        tmp = target.with_suffix(target.suffix + ".tmp")
        try:
            with tmp.open("wb") as handle:
                for item in batch:
                    handle.write(json.dumps(item, sort_keys=True, separators=(",", ":")).encode("utf-8"))
                    handle.write(b"\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, target)
            self._cb_failures = 0
            self.stats.gradient_batches_dispatched += 1
            return len(batch)
        except Exception:
            self._requeue_gradient_items(items)
            with contextlib.suppress(OSError):
                tmp.unlink()
            self._cb_failures += 1
            if self._cb_failures >= CIRCUIT_BREAKER_FAILURES:
                self._cb_open_until = time.monotonic() + CIRCUIT_BREAKER_BACKOFF_S
                self.stats.circuit_breaker_trips += 1
            raise

    def _purge_expired_gradients(self) -> int:
        now = time.time()
        before = len(self.gradient_heap)
        self.gradient_heap = [item for item in self.gradient_heap if not item.is_expired(now=now, ttl_seconds=self.gradient_item_ttl_s)]
        removed = before - len(self.gradient_heap)
        if removed:
            heapq.heapify(self.gradient_heap)
            self.stats.expired_gradients += removed
        self.stats.queued_gradients = len(self.gradient_heap)
        return removed

    def discover_signal_zones(self, event: SignalExtractedEvent) -> List[ZoneDraft]:
        if event.zone_count <= 0 or event.signal_density < RECIPE_DRAFT_CONFIDENCE_MIN:
            return []
        zone_type = event.signal_type or "prose"
        selector = self._selector_for_event(event)
        confidence = min(1.0, max(RECIPE_DRAFT_CONFIDENCE_MIN, event.signal_density))
        return [
            ZoneDraft(
                topology_class=event.topology_class,
                zone_type=zone_type,
                selector=selector,
                confidence=confidence,
                sample_count=max(1, event.zone_count),
                source=event.source_component,
            )
        ]

    def _selector_for_event(self, event: SignalExtractedEvent) -> str:
        if event.signal_type == "code":
            return "pre code, code"
        if event.signal_type == "table":
            return "table, [role=table]"
        if event.signal_type == "heading":
            return "h1, h2, h3, [data-title]"
        if event.topology_class.endswith("DOCS"):
            return "main article, article, main"
        return "article, main, [data-content]"

    async def run(self) -> None:
        """Run the async daemon forever until cancelled or stopped."""
        await self.start_background_tasks()
        try:
            while self._running:
                await asyncio.sleep(3600.0)
        except asyncio.CancelledError:
            raise
        finally:
            await self.stop()

    async def start_background_tasks(self) -> None:
        if self._running and self._tasks:
            return
        self._running = True
        self._start_time = time.monotonic()
        self._tasks = [
            asyncio.create_task(self._gradient_dispatch_loop(), name="axiom.index.gradient_dispatch"),
            asyncio.create_task(self._phase_scan_loop(), name="axiom.index.phase_scan"),
            asyncio.create_task(self._health_log_loop(), name="axiom.index.health_log"),
            asyncio.create_task(self._recipe_save_loop(), name="axiom.index.recipe_save"),
            asyncio.create_task(self._gradient_purge_loop(), name="axiom.index.gradient_purge"),
        ]

    async def stop(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.recipe_registry.save_if_dirty()

    async def aclose(self) -> None:
        await self.stop()
        self.close()

    async def _gradient_dispatch_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.gradient_flush_interval_s)
            try:
                await self._dispatch_gradient_batch()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("gradient dispatch failed: %s", exc)

    async def _phase_scan_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.phase_scan_interval_s)
            try:
                for topology_class, state in self.phase_store.read_all_active().items():
                    old_phase = state.phase
                    if state.phase == PHASE_COLD and state.observation_count >= 50 and state.confidence >= 0.70:
                        state.phase = PHASE_LEARNING
                    elif state.phase == PHASE_LEARNING and state.observation_count >= 200 and state.confidence >= 0.85:
                        state.phase = PHASE_KNOWN
                    if state.phase != old_phase:
                        self.stats.phase_promotions += 1
                        self.phase_store.write(topology_class, state)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("phase scan failed: %s", exc)

    async def _health_log_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.health_log_interval_s)
            self.stats.uptime_seconds = time.monotonic() - self._start_time
            logger.info(
                "index daemon health events=%s queued=%s batches=%s uptime=%.1fs",
                self.stats.dispatched_events,
                len(self.gradient_heap),
                self.stats.gradient_batches_dispatched,
                self.stats.uptime_seconds,
            )

    async def _recipe_save_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.recipe_save_interval_s)
            try:
                self.recipe_registry.save_if_dirty()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("recipe save failed: %s", exc)

    async def _gradient_purge_loop(self) -> None:
        while self._running:
            await asyncio.sleep(self.gradient_purge_interval_s)
            self._purge_expired_gradients()

    def status(self) -> Dict[str, Any]:
        self.stats.uptime_seconds = time.monotonic() - self._start_time
        return {
            "running": self._running,
            "stats": self.stats.to_dict(),
            "gradient_queue": len(self.gradient_heap),
            "gradient_capacity": self.gradient_capacity,
            "recipe_actions": len(self.recipe_actions),
            "zone_drafts": len(self.zone_drafts),
            "recipe_registry_crc": self.recipe_registry.digest(),
            "recipe_count": self.recipe_registry.recipe_count(),
            "offline_queue": str(self._offline_queue_path),
            "phase_summary": self.phase_store.phase_summary(),
            "circuit_breaker": {
                "failures": self._cb_failures,
                "open": time.monotonic() < self._cb_open_until,
                "open_until": round(self._cb_open_until, 3),
            },
        }

    def close(self) -> None:
        self._running = False
        for task in self._tasks:
            task.cancel()
        self._tasks.clear()
        self.recipe_registry.save_if_dirty()
        self.phase_store.close()


async def run_once_for_test(store_dir: Path) -> Dict[str, Any]:
    daemon = IndexDaemon(store_dir=store_dir)
    try:
        await daemon.handle_signal_extracted(
            SignalExtractedEvent(
                url="https://example.com",
                topology_class="NEWS_ARTICLE",
                signal_type="prose",
                byte_count=100,
                token_count=20,
                signal_density=0.8,
                zone_count=2,
                source_component="test",
                run_id=str(new_run_id()),
            )
        )
        return daemon.status()
    finally:
        daemon.close()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="AXIOM index daemon")
    parser.add_argument("--store", default=os.environ.get("AXIOM_STORE_DIR", "store"))
    parser.add_argument("--run", action="store_true", help="start the async fire-and-forget daemon loop")
    parser.add_argument("--once", action="store_true", help="run one smoke dispatch and print status")
    args = parser.parse_args()

    if args.once:
        status = asyncio.run(run_once_for_test(Path(args.store)))
        print(json.dumps({"ok": True, "daemon": "index", "status": status}, sort_keys=True))
        return 0

    store_dir = Path(args.store)
    daemon = IndexDaemon(store_dir=store_dir)
    try:
        if args.run:
            print(json.dumps({"ok": True, "daemon": "index", "mode": "run"}), flush=True)
            asyncio.run(daemon.run())
            return 0
        print(json.dumps({"ok": True, "daemon": "index", "status": daemon.status()}, sort_keys=True))
        return 0
    finally:
        daemon.close()


if __name__ == "__main__":
    raise SystemExit(main())
