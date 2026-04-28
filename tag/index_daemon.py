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
import heapq
import json
import mmap
import os
import struct
import time
import uuid
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from signal_kernel.contracts import (
    FetchAnomalyEvent,
    PhaseTransitionEvent,
    RecipeHealthEvent,
    RecipeStaleEvent,
    SignalExtractedEvent,
    StoreHealthEvent,
    SurpriseEvent,
    WeightsUpdatedEvent,
    new_run_id,
)


PHASE_SLOT = struct.Struct("<I I f I I Q I")
PHASE_SLOT_BYTES = 32
PHASE_COLD = 1
PHASE_LEARNING = 2
PHASE_KNOWN = 3
GRADIENT_PRIORITY_HIGH = 0
GRADIENT_PRIORITY_MEDIUM = 1
GRADIENT_PRIORITY_LOW = 2
RECIPE_DRAFT_CONFIDENCE_MIN = 0.60


@dataclass
class PhaseState:
    topology_hash: int
    phase: int = PHASE_COLD
    confidence: float = 0.0
    observations: int = 0
    surprises: int = 0
    updated_unix: int = field(default_factory=lambda: int(time.time()))
    crc32: int = 0

    def pack(self) -> bytes:
        self.crc32 = self.compute_crc()
        return PHASE_SLOT.pack(
            self.topology_hash,
            self.phase,
            self.confidence,
            self.observations,
            self.surprises,
            self.updated_unix,
            self.crc32,
        ) + b"\x00" * (PHASE_SLOT_BYTES - PHASE_SLOT.size)

    def compute_crc(self) -> int:
        import zlib

        raw = PHASE_SLOT.pack(
            self.topology_hash,
            self.phase,
            self.confidence,
            self.observations,
            self.surprises,
            self.updated_unix,
            0,
        )
        return zlib.crc32(raw) & 0xFFFFFFFF


class PhaseStore:
    """Fixed-slot mmap store for topology phase state."""

    def __init__(self, path: Path, slots: int = 4096) -> None:
        self.path = path
        self.slots = slots
        self.path.parent.mkdir(parents=True, exist_ok=True)
        required = slots * PHASE_SLOT_BYTES
        if not self.path.exists() or self.path.stat().st_size < required:
            with self.path.open("wb") as f:
                f.truncate(required)
        self._file = self.path.open("r+b")
        self._mmap = mmap.mmap(self._file.fileno(), required)

    def close(self) -> None:
        self._mmap.flush()
        self._mmap.close()
        self._file.close()

    def read(self, topology_class: str) -> PhaseState:
        slot = self._slot(topology_class)
        offset = slot * PHASE_SLOT_BYTES
        raw = self._mmap[offset : offset + PHASE_SLOT.size]
        vals = PHASE_SLOT.unpack(raw)
        if vals[0] == 0:
            return PhaseState(topology_hash=self._hash(topology_class))
        return PhaseState(*vals)

    def write(self, topology_class: str, state: PhaseState) -> None:
        slot = self._slot(topology_class)
        offset = slot * PHASE_SLOT_BYTES
        self._mmap[offset : offset + PHASE_SLOT_BYTES] = state.pack()
        self._mmap.flush(offset, PHASE_SLOT_BYTES)

    def _slot(self, topology_class: str) -> int:
        return self._hash(topology_class) % self.slots

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
    queued_gradients: int = 0
    dropped_gradients: int = 0
    drafted_recipes: int = 0
    dispatched_events: int = 0
    dispatch_errors: int = 0


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

    def load(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self.recipes = {}
            return
        try:
            raw = self.path.read_bytes().rstrip(b"\x00")
            self.recipes = json.loads(raw.decode("utf-8")) if raw else {}
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            self.recipes = {}

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
        return len(bucket) - before

    def atomic_save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.recipes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tmp = self.path.with_suffix(self.path.suffix + f".{uuid.uuid4().hex}.tmp")
        with tmp.open("wb") as f:
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.path)

    def digest(self) -> str:
        payload = json.dumps(self.recipes, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return f"{zlib.crc32(payload) & 0xFFFFFFFF:08x}"


class IndexDaemon:
    """
    Event-driven coordinator for phase state and offline work queues.
    """

    def __init__(self, *, store_dir: Path = Path("store")) -> None:
        self.store_dir = store_dir
        self.phase_store = PhaseStore(store_dir / "phase_states.mmap")
        self.recipe_registry = RecipeRegistry(store_dir / "recipe_registry.mmap")
        self.recipe_registry.load()
        self.stats = DaemonStats()
        self.gradient_capacity = 1000
        self.gradient_heap: List[GradientWorkItem] = []
        self._gradient_sequence = 0
        self.recipe_actions: List[RecipeAction] = []
        self.health_events: List[StoreHealthEvent] = []
        self.zone_drafts: List[ZoneDraft] = []

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
            else:
                raise TypeError(f"unsupported event type: {type(event).__name__}")
        except Exception:
            self.stats.dispatch_errors += 1
            raise

    async def handle_signal_extracted(self, event: SignalExtractedEvent) -> None:
        self.stats.signal_events += 1
        state = self.phase_store.read(event.topology_class)
        state.observations += 1
        state.confidence = min(1.0, max(state.confidence, event.signal_density))
        state.updated_unix = int(time.time())
        drafts = self.discover_signal_zones(event)
        if drafts:
            self.zone_drafts.extend(drafts)
            added = self.recipe_registry.add_drafts(event.topology_class, drafts)
            if added:
                self.stats.drafted_recipes += added
                self.recipe_registry.atomic_save()
        if state.phase == PHASE_COLD and state.observations >= 10 and state.confidence >= 0.70:
            state.phase = PHASE_LEARNING
        if state.phase == PHASE_LEARNING and state.observations >= 50 and state.confidence >= 0.90 and state.surprises == 0:
            state.phase = PHASE_KNOWN
        self.phase_store.write(event.topology_class, state)

    async def handle_surprise(self, event: SurpriseEvent) -> None:
        self.stats.surprise_events += 1
        state = self.phase_store.read(event.topology_class)
        state.surprises += 1
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

    async def handle_weights_updated(self, event: WeightsUpdatedEvent) -> None:
        self.stats.weights_updated += 1

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
        batch: List[Dict[str, Any]] = []
        for _ in range(max(0, limit)):
            if not self.gradient_heap:
                break
            batch.append(heapq.heappop(self.gradient_heap).to_dict())
        self.stats.queued_gradients = len(self.gradient_heap)
        return batch

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

    def status(self) -> Dict[str, Any]:
        return {
            "stats": self.stats.__dict__,
            "gradient_queue": len(self.gradient_heap),
            "recipe_actions": len(self.recipe_actions),
            "zone_drafts": len(self.zone_drafts),
            "recipe_registry_crc": self.recipe_registry.digest(),
        }

    def close(self) -> None:
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
    store_dir = Path(os.environ.get("AXIOM_STORE_DIR", "store"))
    daemon = IndexDaemon(store_dir=store_dir)
    try:
        print(json.dumps({"ok": True, "daemon": "index", "status": daemon.status()}, sort_keys=True))
        return 0
    finally:
        daemon.close()


if __name__ == "__main__":
    raise SystemExit(main())
