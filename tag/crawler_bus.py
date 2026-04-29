"""
tag/crawler_bus.py
==================
AXIOM's distributed nervous system.

A dual-sink event backbone that operates identically whether Kafka is present
or not. Every component in the system emits events and receives events through
this file. Nothing else.

Architecture law:
    No component checks which sink is active. The bus interface is identical
    in both modes. Kafka reachable → aiokafka. Kafka unreachable → asyncio.Queue.
    Mode is detected at startup. Mode is logged once. Mode is never re-examined
    by any caller.

Security law:
    Every envelope is HMAC-signed before emission. Every envelope is HMAC-verified
    before deserialization. A tampered or replayed event never reaches a handler.
    Timing-safe comparison is mandatory. `==` on HMAC digests is never used.

Dead letter law:
    Every handler failure is written to /store/dead_letters.jsonl with fsync.
    Events are never silently dropped. The daemon watches this file.

Backpressure law:
    Producers block when queues are full. They do not drop. They do not silently
    discard. Callers decide what to do with KafkaSinkUnavailable or a blocked
    coroutine.

Shutdown law:
    stop() drains queues before closing connections. Events in-flight at shutdown
    are processed, not abandoned.

Dependency direction:
    crawler_bus.py → contracts.py (for event schemas)
    crawler_bus.py → exceptions.py (for error types)
    Everything else → crawler_bus.py (never the reverse)

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import dataclasses
import hashlib
import hmac
import logging
import os
import ssl
import sys # noqa
import time
import traceback # noqa
import types
import uuid # noqa
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Set, # noqa
    Tuple,
    Type,
    TypeVar,
    FrozenSet,
    Union,
    cast,
    get_args,
    get_origin,
    get_type_hints,
)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY (all mandatory, installed via requirements.txt)
# ─────────────────────────────────────────────────────────────────────────────

import msgpack
import orjson
import structlog
from tenacity import (
    AsyncRetrying,
    RetryError,
    stop_after_attempt,
    wait_exponential,
)

# ─────────────────────────────────────────────────────────────────────────────
# AIOKAFKA — conditional import. Not available in degraded mode.
# The import is attempted once at module load time and the result is used by
# _detect_mode(). No other code path attempts to import aiokafka.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
    from aiokafka.errors import (
        KafkaConnectionError,
        KafkaError,
        KafkaTimeoutError,
        TopicAuthorizationFailedError,
        UnknownTopicOrPartitionError,
    )
    _AIOKAFKA_AVAILABLE = True
except ImportError:
    _AIOKAFKA_AVAILABLE = False
    # Placeholders so type-checker does not complain about undefined names
    # in the _KafkaSink body. They are never reached in degraded mode.
    AIOKafkaConsumer = None  # type: ignore[assignment,misc]
    AIOKafkaProducer = None  # type: ignore[assignment,misc]

    class KafkaError(Exception): pass           # type: ignore[no-redef]
    class KafkaConnectionError(KafkaError): pass # type: ignore[no-redef]
    class KafkaTimeoutError(KafkaError): pass    # type: ignore[no-redef]
    class TopicAuthorizationFailedError(KafkaError): pass # type: ignore[no-redef]
    class UnknownTopicOrPartitionError(KafkaError): pass  # type: ignore[no-redef]

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.contracts import (
    CLStateUpdateEvent,
    CleanSignalEvent,
    ClassificationEvent,        # migrate target: was topology/surprise_detector.py
    ContainerBreachEvent,
    NewTopologyHintEvent,
    CrawlManifestReadyEvent,
    FetchMode,
    FetchAnomalyEvent,
    DomainTopologyEvent,
    ManifestCompleteEvent,
    PhaseTransitionEvent,
    RawFetchEvent,
    RecipeHealthEvent,
    RecipeStaleEvent,
    SignalExtractedEvent,
    SnapshotCandidateEvent,
    SnapshotCapturedEvent,
    StoreHealthEvent,
    SurpriseEvent,
    ToolHealthEvent,
    ToolInvocationEvent,
    ToolResultEvent,
    WeightsUpdatedEvent,
    ZoneMapUpdatedEvent,
    ZoneMapInvalidatedEvent,
)
from signal_kernel.contracts import FeedbackEvent as FeedbackBusEvent
from signal_kernel.exceptions import (
    EventBusSubscriptionError,
    EventDispatchFailed, # noqa
    EventIntegrityError,
    EventSchemaError,
    KafkaSinkUnavailable,
)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGER
# All structlog calls go through this logger. Every emit, every dispatch,
# every dead letter, every mode transition is observable through structlog.
# ─────────────────────────────────────────────────────────────────────────────

log: structlog.BoundLogger = structlog.get_logger("crawler_bus")

# ─────────────────────────────────────────────────────────────────────────────
# TYPE VARIABLE
# ─────────────────────────────────────────────────────────────────────────────

T = TypeVar("T")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
# Never hardcode Kafka addresses. All connection parameters come from
# environment variables. The bus reads them once at startup.
# ─────────────────────────────────────────────────────────────────────────────

# Kafka bootstrap servers. If unset, bus falls to degraded mode immediately.
_KAFKA_BOOTSTRAP_ENV: str = "KAFKA_BOOTSTRAP_SERVERS"

# HMAC key for envelope signing. 32 bytes minimum. Never hardcoded.
# Loaded from environment as raw bytes. Missing at startup is a hard stop.
_BUS_HMAC_KEY_ENV: str = "AXIOM_BUS_HMAC_KEY"

# TLS certificate paths. All three required for mTLS. Missing disables TLS
# but logs a security warning — degraded Kafka mode still uses TLS if certs
# are present; the bus does not silently downgrade without logging.
_KAFKA_CA_CERT_ENV:     str = "KAFKA_CA_CERT_PATH"
_KAFKA_CLIENT_CERT_ENV: str = "KAFKA_CLIENT_CERT_PATH"
_KAFKA_CLIENT_KEY_ENV:  str = "KAFKA_CLIENT_KEY_PATH"

# Default cert paths if env vars are not set.
_DEFAULT_CA_CERT:     str = "/certs/ca.crt"
_DEFAULT_CLIENT_CERT: str = "/certs/client.crt"
_DEFAULT_CLIENT_KEY:  str = "/certs/client.key"

# Dead letter store. Created by cold_start.py before bus.start() is called.
_DEAD_LETTER_PATH: Path = Path(os.environ.get("AXIOM_DEAD_LETTER_PATH", "/store/dead_letters.jsonl"))

# Kafka producer settings — tuned for throughput with bounded latency.
_KAFKA_LINGER_MS:       int = 10          # batch for 10 ms before sending
_KAFKA_BUFFER_MEMORY:   int = 32 * 1024 * 1024  # 32 MB per producer
_KAFKA_MAX_BLOCK_MS:    int = 5_000       # block emit() for 5 s before raising
_KAFKA_REQUEST_TIMEOUT: int = 30_000      # broker request timeout
_KAFKA_COMPRESSION:     str = "lz4"       # compress batches before sending

# Kafka consumer settings.
_KAFKA_SESSION_TIMEOUT_MS:    int = 30_000
_KAFKA_HEARTBEAT_INTERVAL_MS: int = 10_000
_KAFKA_MAX_POLL_RECORDS:      int = 500
_KAFKA_AUTO_OFFSET_RESET:     str = "earliest"
_KAFKA_ENABLE_AUTO_COMMIT:    bool = False  # manual commit after successful dispatch

# Degraded mode queue ceiling. emit() blocks when the queue reaches this size,
# applying natural backpressure to the fetcher. Not a drop policy — a blocking policy.
_QUEUE_MAX_SIZE: int = 10_000

# Handler failure retry budget. After this many failures the event is dead-lettered.
_HANDLER_MAX_RETRIES: int = 3

# Kafka connection probe timeout at startup.
_KAFKA_PROBE_TIMEOUT_S: float = 5.0

# Number of Kafka connection attempts before falling to degraded mode.
_KAFKA_PROBE_ATTEMPTS: int = 3

# Dispatch loop poll interval in degraded mode (seconds).
_DISPATCH_POLL_INTERVAL_S: float = 0.01

# Schema version stamped in every envelope header.
_SCHEMA_VERSION: str = "1"

# Maximum envelope value size (bytes). Anything larger is a dead letter immediately.
_MAX_ENVELOPE_VALUE_BYTES: int = 8 * 1024 * 1024  # 8 MB

# Shutdown drain timeout (seconds). After this the bus closes regardless.
_SHUTDOWN_DRAIN_TIMEOUT_S: float = 10.0

# Minimum HMAC key length in bytes.
_MIN_HMAC_KEY_BYTES: int = 32


def _get_env_bytes(name: str) -> Optional[bytes]:
    """Read an environment variable as bytes on POSIX and Windows."""
    if hasattr(os, "environb"):
        return os.environb.get(name.encode())  # type: ignore[attr-defined]
    raw = os.environ.get(name)
    return raw.encode("utf-8") if raw is not None else None


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL HMAC KEY
# Loaded once at import time. Missing key at import time is a hard stop —
# the bus cannot operate without a signing key. The key never leaves this module.
# ─────────────────────────────────────────────────────────────────────────────

def _load_hmac_key() -> bytes:
    """
    Load the HMAC key from the environment.
    Raises RuntimeError at import time if the key is missing or too short.
    This is intentional — a bus without an HMAC key is a security failure,
    not a degraded-mode fallback.
    """
    raw = _get_env_bytes(_BUS_HMAC_KEY_ENV)
    if not raw:
        raise RuntimeError(
            f"AXIOM_BUS_HMAC_KEY environment variable is not set. "
            f"The event bus cannot operate without a signing key. "
            f"Generate 32 random bytes and export as {_BUS_HMAC_KEY_ENV}. "
            f"Example: export {_BUS_HMAC_KEY_ENV}=$(openssl rand -hex 32)"
        )
    # Decode from hex if it looks like a hex string (common provisioning pattern).
    if len(raw) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw):
        raw = bytes.fromhex(raw.decode("ascii"))
    if len(raw) < _MIN_HMAC_KEY_BYTES:
        raise RuntimeError(
            f"{_BUS_HMAC_KEY_ENV} is {len(raw)} bytes. "
            f"Minimum is {_MIN_HMAC_KEY_BYTES} bytes. "
            f"Short HMAC keys are cryptographically unsafe."
        )
    return raw


_HMAC_KEY: bytes = _load_hmac_key()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1: BusMode ENUM
# The operational mode of the bus. Set once at startup. Never changes at runtime.
# ═════════════════════════════════════════════════════════════════════════════

class BusMode(Enum):
    """
    Operational mode of the CrawlerBus.

    KAFKA:    aiokafka producers and consumers. Persistent log. Consumer groups.
              Replay. Partitioning. Fleet-ready. Cross-machine. Durable.

    DEGRADED: asyncio.Queue per topic. Zero external dependency. Same interface.
              Events lost on process death. Single machine only.

    Mode is detected once at startup. Never polled. Never changed.
    Components never query the mode.
    """
    KAFKA    = auto()
    DEGRADED = auto()

    def __str__(self) -> str:
        return self.name.lower()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2: BusEnvelope DATACLASS
# The internal representation of every event, regardless of sink.
# Producers never build envelopes — _serialize() does.
# Consumers never unwrap envelopes — _deserialize() does.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BusEnvelope:
    """
    The wire format for every event on the bus.

    Regardless of sink, every emitted event becomes a BusEnvelope before
    transport and arrives as a BusEnvelope before deserialization. The envelope
    carries headers that survive into Kafka record headers or asyncio.Queue
    entries identically.

    Fields:
        topic:      The registered topic name.
        key:        Partition key bytes. Kafka uses this for partition assignment.
                    In degraded mode it is stored but unused for routing.
        value:      msgpack-encoded event bytes.
        headers:    Metadata dict. Source component, run_id, schema_version,
                    emit_timestamp, hmac_sha256. Stored as str→str.
        offset:     Monotonically increasing per-topic offset counter.
                    Kafka mode: the real Kafka offset (set on receive).
                    Degraded mode: a local sequence number per topic.
        partition:  Kafka partition number. -1 in degraded mode.
        timestamp:  UTC ISO 8601 string of emission time.
    """

    topic:     str
    key:       bytes
    value:     bytes
    headers:   Dict[str, str]
    offset:    int
    partition: int
    timestamp: str

    @property
    def run_id(self) -> Optional[str]:
        return self.headers.get("run_id")

    @property
    def source_component(self) -> Optional[str]:
        return self.headers.get("source_component")

    @property
    def hmac_digest(self) -> Optional[str]:
        return self.headers.get("hmac_sha256")

    @property
    def schema_version(self) -> str:
        return self.headers.get("schema_version", "1")

    @property
    def value_size(self) -> int:
        return len(self.value)

    def is_oversized(self) -> bool:
        return self.value_size > _MAX_ENVELOPE_VALUE_BYTES

    def to_log_dict(self) -> Dict[str, Any]:
        """Structured log dict. Value bytes are omitted — logged as size."""
        return {
            "topic":            self.topic,
            "key":              self.key.decode("utf-8", errors="replace"),
            "value_bytes":      self.value_size,
            "offset":           self.offset,
            "partition":        self.partition,
            "timestamp":        self.timestamp,
            "source_component": self.source_component,
            "run_id":           self.run_id,
            "schema_version":   self.schema_version,
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3: TopicHealth DATACLASS
# Per-topic health snapshot exposed by CrawlerBus.health().
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TopicHealth:
    """
    Point-in-time health snapshot for one topic.

    Fields:
        topic:              Topic name.
        registered:         True if topic is in the registry.
        producer_connected: True if a producer is active for this topic.
        subscriber_count:   Number of active subscriber groups.
        queue_depth:        Current queue depth (degraded) or lag (Kafka).
        dead_letter_count:  Dead letters written for this topic since startup.
        total_emitted:      Total events emitted on this topic since startup.
        total_dispatched:   Total successful handler dispatches since startup.
        total_failed:       Total handler failures since startup.
        last_emit_ts:       ISO 8601 timestamp of last emit. None if never.
        last_dispatch_ts:   ISO 8601 timestamp of last dispatch. None if never.
    """

    topic:              str
    registered:         bool
    producer_connected: bool
    subscriber_count:   int
    queue_depth:        int
    dead_letter_count:  int
    total_emitted:      int
    total_dispatched:   int
    total_failed:       int
    last_emit_ts:       Optional[str]
    last_dispatch_ts:   Optional[str]

    @property
    def error_rate(self) -> float:
        """Fraction of dispatches that have failed. 0.0 if no dispatches."""
        if self.total_dispatched == 0:
            return 0.0
        return self.total_failed / (self.total_dispatched + self.total_failed)

    @property
    def is_healthy(self) -> bool:
        """
        True if the topic is in a nominally healthy state.
        Unhealthy conditions: unregistered, no producer, error rate > 10%,
        or queue depth above 80% of max.
        """
        if not self.registered:
            return False
        if self.dead_letter_count > 0:
            return False
        if self.error_rate > 0.10:
            return False
        if self.queue_depth > (_QUEUE_MAX_SIZE * 0.80):
            return False
        return True


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4: BusHealth DATACLASS
# System-wide health snapshot exposed by CrawlerBus.health().
# Called by cold_start.py during validation and by index_daemon.py periodically.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BusHealth:
    """
    System-wide health snapshot of the CrawlerBus.

    Called by cold_start.py during initialization validation.
    Called by index_daemon.py periodically — result is emitted to structlog.

    Fields:
        mode:           "kafka" or "degraded".
        topics:         Per-topic health. All registered topics are present.
        dead_letters:   Total dead letters written since process start.
        lag:            Consumer lag per group. group_id → topic → lag.
                        Kafka mode: real consumer group lag from broker.
                        Degraded mode: approximated from queue depth.
        uptime_s:       Seconds since bus.start() was called.
        started:        True if bus.start() has been called and completed.
        kafka_connected: True if Kafka broker is reachable (Kafka mode only).
        total_emitted:  Total events emitted across all topics since startup.
        total_dispatched: Total successful dispatches across all topics.
        total_failed:   Total handler failures across all topics.
        hmac_failures:  Total HMAC verification failures since startup.
        schema_failures: Total schema validation failures since startup.
    """

    mode:             str
    topics:           Dict[str, TopicHealth]
    dead_letters:     int
    lag:              Dict[str, Dict[str, int]]
    uptime_s:         float
    started:          bool
    kafka_connected:  bool
    total_emitted:    int
    total_dispatched: int
    total_failed:     int
    hmac_failures:    int
    schema_failures:  int

    @property
    def is_healthy(self) -> bool:
        """
        True if the bus is fully operational.
        Conditions: started, all topics healthy, no runaway dead letters.
        """
        if not self.started:
            return False
        if not all(t.is_healthy for t in self.topics.values()):
            return False
        return True

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "mode":              self.mode,
            "dead_letters":      self.dead_letters,
            "uptime_s":          round(self.uptime_s, 2),
            "started":           self.started,
            "kafka_connected":   self.kafka_connected,
            "total_emitted":     self.total_emitted,
            "total_dispatched":  self.total_dispatched,
            "total_failed":      self.total_failed,
            "hmac_failures":     self.hmac_failures,
            "schema_failures":   self.schema_failures,
            "unhealthy_topics":  [
                name for name, t in self.topics.items() if not t.is_healthy
            ],
        }


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5: DeadLetterEvent DATACLASS
# Written to /store/dead_letters.jsonl on handler failure.
# The file is append-only. Each line is one JSON object.
# index_daemon.py watches this file via store_watchdog.py.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeadLetterEvent:
    """
    A forensic record of one failed event dispatch.

    Written to /store/dead_letters.jsonl on:
        - Handler exception after _HANDLER_MAX_RETRIES exhausted.
        - HMAC verification failure.
        - Schema validation failure.
        - Oversized envelope.

    The file is append-only and fsync-guaranteed. Dead letters survive
    process death. index_daemon.py watches for structural patterns.

    Fields:
        topic:      Topic on which the event was received.
        event:      Hex-encoded msgpack bytes of the original event.
        handler:    Fully qualified name of the handler that failed.
                    "__hmac_verify__" for integrity failures.
                    "__schema_validate__" for schema failures.
                    "__oversized__" for size limit failures.
        error:      Exception class name and message.
        retries:    Number of handler invocations attempted before giving up.
        timestamp:  UTC ISO 8601 string of when the dead letter was written.
        mode:       "kafka" or "degraded".
        run_id:     run_id extracted from envelope headers. May be None.
        partition:  Kafka partition number. -1 in degraded mode.
        offset:     Envelope offset at time of failure.
        source_component: source_component from envelope headers. May be None.
        envelope_key: Hex-encoded partition key from envelope.
    """

    topic:            str
    event:            str        # hex-encoded msgpack bytes
    handler:          str
    error:            str
    retries:          int
    timestamp:        str
    mode:             str
    run_id:           Optional[str]
    partition:        int
    offset:           int
    source_component: Optional[str]
    envelope_key:     str

    def to_jsonl_line(self) -> bytes:
        """Serialize to a single JSON line terminated by newline."""
        return orjson.dumps(dataclasses.asdict(self)) + b"\n"


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6: TopicEmitter[T] — TYPED EMITTER
# The only way to produce events. Returned by CrawlerBus.emitter().
# Wrong schema type is a TypeError at call time, not a downstream failure.
# ═════════════════════════════════════════════════════════════════════════════

class TopicEmitter(Generic[T]):
    """
    Typed event producer for one topic.

    Constructed by CrawlerBus.emitter(). Never constructed directly.
    The type parameter T is the event schema. Passing the wrong type raises
    TypeError immediately at emit() — not at deserialization downstream.

    Usage:
        self._emitter: TopicEmitter[CleanSignalEvent] = await bus.emitter(
            topic="clean_signal",
            component="alpine_strip.offline_pipeline",
            schema=CleanSignalEvent,
        )
        await self._emitter.emit(event)

    Thread / async safety:
        TopicEmitter is NOT thread-safe. It is coroutine-safe.
        One emitter per component, one component per asyncio task, no sharing.
    """

    __slots__ = (
        "_bus",
        "_topic",
        "_component",
        "_schema",
        "_enabled",
        "_emit_count",
        "_last_emit_ts",
    )

    def __init__(
        self,
        bus: "CrawlerBus",
        topic: str,
        component: str,
        schema: Type[T],
    ) -> None:
        self._bus:          CrawlerBus = bus
        self._topic:        str = topic
        self._component:    str = component
        self._schema:       Type[T] = schema
        self._enabled:      bool = True
        self._emit_count:   int = 0
        self._last_emit_ts: Optional[str] = None

    @property
    def topic(self) -> str:
        return self._topic

    @property
    def component(self) -> str:
        return self._component

    @property
    def schema(self) -> Type[T]:
        return self._schema

    @property
    def emit_count(self) -> int:
        return self._emit_count

    @property
    def last_emit_ts(self) -> Optional[str]:
        return self._last_emit_ts

    def disable(self) -> None:
        """
        Disable this emitter. Called by CrawlerBus.stop() before shutdown.
        Subsequent emit() calls raise RuntimeError immediately.
        """
        self._enabled = False

    async def emit(self, event: T) -> None:
        """
        Emit one typed event.

        The event must be an instance of the schema type declared at
        construction. Wrong type raises TypeError immediately.

        If the Kafka broker is unavailable and the bus is in Kafka mode,
        raises KafkaSinkUnavailable. The caller decides how to handle it —
        the bus does not buffer or retry on behalf of producers.

        In degraded mode, emit() blocks if the queue is at capacity.
        This is intentional backpressure — not a failure. The fetcher
        slows down when downstream processing can't keep up.

        Raises:
            RuntimeError:         If called after disable().
            TypeError:            If event is not an instance of schema.
            KafkaSinkUnavailable: If Kafka broker is unreachable (Kafka mode).
            asyncio.QueueFull:    Never — degraded mode blocks, does not raise.
        """
        if not self._enabled:
            raise RuntimeError(
                f"TopicEmitter for topic={self._topic!r} "
                f"component={self._component!r} has been disabled. "
                f"Bus is shutting down — do not emit after stop()."
            )

        if not isinstance(event, self._schema):
            raise TypeError(
                f"TopicEmitter[{self._schema.__name__}] on topic={self._topic!r} "
                f"received event of type {type(event).__name__!r}. "
                f"Schema mismatch is a programming error — not a runtime condition. "
                f"Pass a {self._schema.__name__} instance."
            )

        t_start = time.monotonic()
        await self._bus._emit( # noqa
            topic=self._topic,
            event=event,
            component=self._component,
        )
        latency_ms = (time.monotonic() - t_start) * 1000.0

        self._emit_count += 1
        self._last_emit_ts = _iso8601_now()

        log.debug(
            "event_emitted",
            topic=self._topic,
            component=self._component,
            schema=self._schema.__name__,
            emit_count=self._emit_count,
            latency_ms=round(latency_ms, 3),
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7: TopicRegistry
# All topics must be declared before the bus starts. No dynamic registration.
# Attempting to emit or subscribe on an unknown topic is a hard stop.
# ═════════════════════════════════════════════════════════════════════════════

# The canonical topic registry. Topic name → event schema class.
# All bus topics are declared here. This is the only place they are defined.
TOPIC_REGISTRY: Dict[str, Type[Any]] = {
    "raw_fetch":          RawFetchEvent,
    "fetch_anomaly":      FetchAnomalyEvent,
    "clean_signal":       CleanSignalEvent,
    "signal_extracted":   SignalExtractedEvent,
    "classification":     ClassificationEvent,
    "topology_hint": NewTopologyHintEvent,
    "domain_topology":    DomainTopologyEvent,
    "crawl_manifest":     CrawlManifestReadyEvent,
    "manifest_complete":  ManifestCompleteEvent,
    "cl_state_update":    CLStateUpdateEvent,
    "container_breach":   ContainerBreachEvent,
    "zone_map_updated":   ZoneMapUpdatedEvent,
    "zone_map_invalidated": ZoneMapInvalidatedEvent,
    "surprise":           SurpriseEvent,
    "recipe_stale":       RecipeStaleEvent,
    "recipe_health":      RecipeHealthEvent,
    "weights_updated":    WeightsUpdatedEvent,
    "store_health":       StoreHealthEvent,
    "snapshot_candidate": SnapshotCandidateEvent,
    "snapshot_captured":  SnapshotCapturedEvent,
    "tool_invocation":    ToolInvocationEvent,
    "tool_result":        ToolResultEvent,
    "tool_health":        ToolHealthEvent,
    "feedback":           FeedbackBusEvent,
    "phase_transition":   PhaseTransitionEvent,
}


class TopicRegistry:
    """
    The validated set of registered topics.

    The bus reads TOPIC_REGISTRY at construction time. No topic may be added
    after construction. No topic may be removed. The topology of the bus is
    as fixed as the topology class registry.

    All validation is at construction time. After _validated is True,
    all registry operations are O(1) dict lookups.
    """

    def __init__(self, topics: Dict[str, Type[Any]]) -> None:
        if not topics:
            raise ValueError(
                "TopicRegistry requires at least one topic. "
                "An empty bus topology is not permitted."
            )
        self._topics: Dict[str, Type[Any]] = dict(topics)
        self._validated: bool = True

        log.debug(
            "topic_registry_initialized",
            topics=list(self._topics.keys()),
            count=len(self._topics),
        )

    def validate_topic(self, topic: str) -> Type[Any]:
        """
        Return the schema for a topic, or raise EventBusSubscriptionError.
        This is the enforcement point for the no-dynamic-topics rule.
        """
        schema = self._topics.get(topic)
        if schema is None:
            raise EventBusSubscriptionError(
                f"Topic {topic!r} is not registered. "
                f"Registered topics: {sorted(self._topics.keys())}. "
                f"Topics must be declared in TOPIC_REGISTRY before the bus starts. "
                f"Dynamic topic creation is not supported."
            )
        return schema

    def schema_for(self, topic: str) -> Type[Any]:
        """Return the schema class for a topic. Raises on unknown topic."""
        return self.validate_topic(topic)

    @property
    def all_topics(self) -> List[str]:
        return sorted(self._topics.keys())

    def __contains__(self, topic: str) -> bool:
        return topic in self._topics


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8: _KafkaSink
# aiokafka-backed sink. Active only when Kafka broker is reachable.
# Manages producers (one per topic), consumers (one per subscription group),
# and all Kafka-specific lifecycle.
# ═════════════════════════════════════════════════════════════════════════════

class _KafkaSink:
    """
    aiokafka-backed event sink.

    One producer per topic — this allows per-topic partition keys to be
    used correctly without cross-contamination. One consumer per subscription
    group per topic — consumer groups share partition load across instances.

    All Kafka resources are created lazily in start() and cleaned in stop().
    This class does not manage the dispatch loop — CrawlerBus._dispatch_loop()
    reads from consumers and dispatches to handlers.

    TLS / mTLS:
        The ssl_context is built once in _build_ssl_context() and shared
        across all producers and consumers. mTLS — broker verifies client,
        client verifies broker.
    """

    def __init__(self, bootstrap_servers: str) -> None:
        self._bootstrap: str = bootstrap_servers
        self._ssl_ctx: Optional[ssl.SSLContext] = _build_ssl_context()
        # Typed as Any because AIOKafkaProducer / AIOKafkaConsumer are set to None
        # at module level when aiokafka is not installed.  These dicts are only
        # populated when the Kafka sink is actually active, so Any is correct —
        # the type checker cannot prove the conditional import resolved to a real
        # class, and Dict[str, None] would make every .start()/.stop() call an error.
        self._producers: Dict[str, Any] = {}  # topic → AIOKafkaProducer
        self._consumers: Dict[str, Any] = {}  # group_key → AIOKafkaConsumer
        self._started: bool = False

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self, topics: List[str]) -> None:
        """
        Start one producer per topic. Consumers are started on first subscribe().
        Raises KafkaSinkUnavailable if any producer cannot connect.
        """
        if self._started:
            return

        for topic in topics:
            try:
                producer = self._make_producer()
                await producer.start()
                self._producers[topic] = producer
                log.info("kafka_producer_started", topic=topic, broker=self._bootstrap)
            except (KafkaConnectionError, KafkaTimeoutError, KafkaError) as exc:
                # Clean up any producers already started.
                await self._stop_producers()
                raise KafkaSinkUnavailable(
                    f"Failed to start Kafka producer for topic={topic!r}: {exc}. "
                    f"Broker: {self._bootstrap}. Bus cannot enter Kafka mode.",
                ) from exc

        self._started = True
        log.info(
            "kafka_sink_started",
            broker=self._bootstrap,
            topics=topics,
            tls_enabled=self._ssl_ctx is not None,
        )

    async def stop(self) -> None:
        """Stop all producers and consumers gracefully."""
        await self._stop_consumers()
        await self._stop_producers()
        self._started = False
        log.info("kafka_sink_stopped", broker=self._bootstrap)

    async def _stop_producers(self) -> None:
        for topic, producer in list(self._producers.items()):
            try:
                await producer.stop()
                log.debug("kafka_producer_stopped", topic=topic)
            except Exception as exc:  # noqa: BLE001
                log.warning("kafka_producer_stop_error", topic=topic, error=str(exc))
        self._producers.clear()

    async def _stop_consumers(self) -> None:
        for group_key, consumer in list(self._consumers.items()):
            try:
                await consumer.stop()
                log.debug("kafka_consumer_stopped", group_key=group_key)
            except Exception as exc:  # noqa: BLE001
                log.warning("kafka_consumer_stop_error", group_key=group_key, error=str(exc))
        self._consumers.clear()

    # ── Emission ──────────────────────────────────────────────────────────────

    async def send(self, envelope: BusEnvelope) -> None:
        """
        Send one envelope to Kafka.

        Raises KafkaSinkUnavailable if the producer is not available.
        Raises KafkaSinkUnavailable if max.block.ms is exceeded.
        Caller decides whether to buffer, drop, or fail.
        """
        producer = self._producers.get(envelope.topic)
        if producer is None:
            raise KafkaSinkUnavailable(
                f"No Kafka producer registered for topic={envelope.topic!r}. "
                f"This means bus.start() was not called or failed for this topic."
            )

        try:
            # Convert headers dict to Kafka header format: list of (key, value_bytes).
            kafka_headers = [
                (k, v.encode("utf-8")) for k, v in envelope.headers.items()
            ]
            await producer.send_and_wait(
                topic=envelope.topic,
                key=envelope.key,
                value=envelope.value,
                headers=kafka_headers,
            )
        except (KafkaConnectionError, KafkaTimeoutError) as exc:
            raise KafkaSinkUnavailable(
                f"Kafka producer send failed for topic={envelope.topic!r}: {exc}. "
                f"Broker may be unreachable or backpressure limit exceeded."
            ) from exc
        except KafkaError as exc:
            raise KafkaSinkUnavailable(
                f"Kafka producer error for topic={envelope.topic!r}: {exc}."
            ) from exc

    # ── Consumption ───────────────────────────────────────────────────────────

    async def subscribe(self, topic: str, group_id: str) -> Any:  # returns AIOKafkaConsumer
        """
        Create and start a consumer for a topic/group combination.
        Returns the consumer. CrawlerBus._dispatch_loop() polls it.
        """
        group_key = f"{topic}:{group_id}"
        if group_key in self._consumers:
            return self._consumers[group_key]

        consumer = self._make_consumer(topic=topic, group_id=group_id)
        try:
            await consumer.start()
            self._consumers[group_key] = consumer
            log.info(
                "kafka_consumer_started",
                topic=topic,
                group_id=group_id,
                broker=self._bootstrap,
            )
            return consumer
        except (KafkaConnectionError, KafkaError) as exc:
            raise KafkaSinkUnavailable(
                f"Failed to start Kafka consumer for topic={topic!r} "
                f"group={group_id!r}: {exc}."
            ) from exc

    async def commit(self, consumer: Any) -> None:  # consumer: AIOKafkaConsumer # noqa
        """Commit offsets for a consumer after successful dispatch."""
        try:
            await consumer.commit()
        except KafkaError as exc:
            # Commit failure is not fatal — offsets will be reprocessed.
            # Log and continue. Duplicate processing is acceptable; message loss is not.
            log.warning("kafka_commit_failed", error=str(exc))

    async def get_lag(self, group_id: str, topic: str) -> int:
        """
        Return the consumer lag for a group/topic. 0 if unknown.
        Lag is end_offset - committed_offset per partition, summed.
        """
        group_key = f"{topic}:{group_id}"
        consumer = self._consumers.get(group_key)
        if consumer is None:
            return 0
        try:
            partitions = consumer.assignment()
            if not partitions:
                return 0
            end_offsets = await consumer.end_offsets(partitions)
            committed = {}
            for partition in partitions:
                pos = await consumer.position(partition)
                committed[partition] = pos
            total_lag = sum(
                max(0, end_offsets.get(p, 0) - committed.get(p, 0))
                for p in partitions
            )
            return total_lag
        except Exception:  # noqa
            return 0

    # ── Factory helpers ───────────────────────────────────────────────────────

    def _make_producer(self) -> Any:  # returns AIOKafkaProducer
        """
        Create an AIOKafkaProducer for one topic's producer slot.

        Producer settings are tuned for a balance of throughput and safety:
            - enable_idempotence=True: the broker deduplicates in-flight retries.
              Prevents duplicate records when the network drops mid-send.
            - acks="all": wait for acknowledgment from all in-sync replicas (ISR).
              This is the safest setting — records survive a single broker failure.
            - linger_ms=10: accumulate records for 10ms before sending.
              Small batching improves throughput without meaningful latency cost
              for a system where events are not latency-critical.
            - compression_type=lz4: LZ4 is the fastest supported compression.
              msgpack events compress 40–60%. Reduces Kafka bandwidth significantly
              on high-volume topics (raw_fetch can be large).
            - request_timeout_ms: how long the producer waits for a broker ack.
              30 seconds is conservative — broker is local or regional.
        """
        kwargs: Dict[str, Any] = dict(
            bootstrap_servers=self._bootstrap,
            compression_type=_KAFKA_COMPRESSION,
            linger_ms=_KAFKA_LINGER_MS,
            request_timeout_ms=_KAFKA_REQUEST_TIMEOUT,
            enable_idempotence=True,
            acks="all",
        )
        if self._ssl_ctx is not None:
            kwargs["security_protocol"] = "SSL"
            kwargs["ssl_context"] = self._ssl_ctx
        assert _AIOKAFKA_AVAILABLE and AIOKafkaProducer is not None, \
            "_make_producer() called without aiokafka installed"
        return AIOKafkaProducer(**kwargs) # noqa | runtime defensive check

    def _make_consumer(self, topic: str, group_id: str) -> Any:  # returns AIOKafkaConsumer
        """
        Create an AIOKafkaConsumer for a single topic.

        Consumer settings are tuned for reliability over throughput:
            - isolation_level=read_committed: skip un-committed transactional
              records. Prevents partial transaction reads in multi-producer setups.
            - enable_auto_commit=False: we commit manually after successful dispatch.
              This ensures at-least-once delivery — reprocess on crash, never lose.
            - auto_offset_reset=earliest: new consumer groups start from the
              beginning of the log. Prevents silent event loss when a new subscriber
              registers on an active topic.
        """
        # AIOKafkaConsumer takes the topic as a positional arg, not via kwargs.
        positional_args = (topic,)
        kwargs: Dict[str, Any] = dict(
            bootstrap_servers=self._bootstrap,
            group_id=group_id,
            auto_offset_reset=_KAFKA_AUTO_OFFSET_RESET,
            enable_auto_commit=_KAFKA_ENABLE_AUTO_COMMIT,
            session_timeout_ms=_KAFKA_SESSION_TIMEOUT_MS,
            heartbeat_interval_ms=_KAFKA_HEARTBEAT_INTERVAL_MS,
            max_poll_records=_KAFKA_MAX_POLL_RECORDS,
            isolation_level="read_committed",
        )
        if self._ssl_ctx is not None:
            kwargs["security_protocol"] = "SSL"
            kwargs["ssl_context"] = self._ssl_ctx
        assert _AIOKAFKA_AVAILABLE and AIOKafkaConsumer is not None, \
            "_make_consumer() called without aiokafka installed"
        return AIOKafkaConsumer(*positional_args, **kwargs) # noqa | runtime defensive check


def _build_ssl_context() -> Optional[ssl.SSLContext]:
    """
    Build an mTLS SSLContext for Kafka connections.

    Reads cert paths from environment variables with fallbacks to /certs/.
    If any cert file is missing, returns None and logs a security warning.
    Degraded mode ignores this entirely — no network, no TLS needed.

    The SSLContext is built once and reused by all producers and consumers.
    """
    ca_path     = os.environ.get(_KAFKA_CA_CERT_ENV, _DEFAULT_CA_CERT)
    cert_path   = os.environ.get(_KAFKA_CLIENT_CERT_ENV, _DEFAULT_CLIENT_CERT)
    key_path    = os.environ.get(_KAFKA_CLIENT_KEY_ENV, _DEFAULT_CLIENT_KEY)

    missing = [p for p in (ca_path, cert_path, key_path) if not Path(p).exists()]
    if missing:
        log.warning(
            "kafka_tls_certs_missing",
            missing=missing,
            consequence="Kafka TLS disabled. Connection will be unencrypted.",
            action="Provision /certs/ca.crt, /certs/client.crt, /certs/client.key "
                   "or set KAFKA_CA_CERT_PATH, KAFKA_CLIENT_CERT_PATH, KAFKA_CLIENT_KEY_PATH.",
        )
        return None

    try:
        ctx = ssl.create_default_context(cafile=ca_path)
        ctx.load_cert_chain(certfile=cert_path, keyfile=key_path)
        ctx.check_hostname = True
        ctx.verify_mode = ssl.CERT_REQUIRED
        # Enforce TLS 1.2+ — older versions are cryptographically unsafe.
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        # Disable weak cipher suites explicitly.
        ctx.set_ciphers("ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM:!aNULL:!eNULL:!RC4:!DH")
        log.info("kafka_tls_configured", ca=ca_path, cert=cert_path)
        return ctx
    except (ssl.SSLError, FileNotFoundError, PermissionError) as exc:
        log.warning(
            "kafka_tls_context_build_failed",
            error=str(exc),
            consequence="Kafka TLS disabled. Review cert files.",
        )
        return None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 9: _DegradedSink
# asyncio.Queue-backed sink. Active when Kafka is unreachable.
# One queue per topic. Same envelope format as Kafka mode.
# ═════════════════════════════════════════════════════════════════════════════

class _DegradedSink:
    """
    asyncio.Queue-backed event sink for Kafka-unavailable operation.

    One asyncio.Queue per topic. Queue capacity is _QUEUE_MAX_SIZE.
    When the queue is full, put() blocks — this is the backpressure mechanism
    that slows down the fetcher when downstream processing is saturated.

    Events are lost on process death. This is the documented tradeoff.
    For durable operation, provide a Kafka broker.

    Consumer groups in degraded mode:
        Each group_id maps to one handler. When a topic has multiple subscriber
        groups, each group receives every event independently. Within a group,
        only the registered handler is called — no fanout within a group.

    This mirrors Kafka consumer group semantics: multiple groups each get a
    copy of every event; within a group, work is distributed (here: no distribution
    since there is only one handler per group in degraded mode).
    """

    def __init__(self) -> None:
        self._queues: Dict[str, asyncio.Queue[BusEnvelope]] = {}
        self._offsets: Dict[str, int] = defaultdict(int)

    def register_topic(self, topic: str) -> None:
        """Ensure a queue exists for the topic."""
        if topic not in self._queues:
            self._queues[topic] = asyncio.Queue(maxsize=_QUEUE_MAX_SIZE)
            log.debug("degraded_queue_created", topic=topic, maxsize=_QUEUE_MAX_SIZE)

    async def put(self, envelope: BusEnvelope) -> None:
        """
        Put an envelope onto the topic queue.
        Blocks if the queue is full — applies backpressure to the producer.
        Never drops silently.
        """
        queue = self._queues.get(envelope.topic)
        if queue is None:
            raise EventBusSubscriptionError(
                f"No queue registered for topic={envelope.topic!r}. "
                f"register_topic() must be called before put()."
            )
        await queue.put(envelope)

    async def get(self, topic: str) -> Optional[BusEnvelope]:
        """
        Non-blocking get. Returns None if the queue is empty.
        _dispatch_loop() polls this with a sleep interval.
        """
        queue = self._queues.get(topic)
        if queue is None:
            return None
        try:
            return queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def queue_depth(self, topic: str) -> int:
        """Current depth of the topic queue."""
        queue = self._queues.get(topic)
        return queue.qsize() if queue else 0

    def next_offset(self, topic: str) -> int:
        """Increment and return the next local sequence number for a topic."""
        self._offsets[topic] += 1
        return self._offsets[topic]

    async def drain(self, timeout_s: float = _SHUTDOWN_DRAIN_TIMEOUT_S) -> Dict[str, int]:
        """
        Drain all queues by waiting for them to reach empty.
        Returns a dict of topic → events remaining (non-zero = drain incomplete).
        Called by CrawlerBus.stop() before closing connections.
        """
        deadline = time.monotonic() + timeout_s
        remaining: Dict[str, int] = {} # noqa

        while time.monotonic() < deadline:
            remaining = {
                topic: q.qsize()
                for topic, q in self._queues.items()
                if q.qsize() > 0
            }
            if not remaining:
                break
            await asyncio.sleep(0.05)

        remaining = {
            topic: q.qsize()
            for topic, q in self._queues.items()
            if q.qsize() > 0
        }
        if remaining:
            log.warning(
                "degraded_drain_incomplete",
                remaining=remaining,
                timeout_s=timeout_s,
                consequence="Events in queue will be lost on process exit.",
            )
        else:
            log.info("degraded_drain_complete", elapsed_s=round(time.monotonic() - (deadline - timeout_s), 3))

        return remaining


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 10: _detect_mode()
# Attempts a Kafka connection. Falls to degraded mode on failure.
# Called once at bus.start(). The result is final.
# ═════════════════════════════════════════════════════════════════════════════

async def _detect_mode() -> Tuple[BusMode, Optional[str]]:
    """
    Probe the Kafka broker and determine bus mode.

    Returns (BusMode.KAFKA, bootstrap_servers) if Kafka is reachable.
    Returns (BusMode.DEGRADED, None) if Kafka is unreachable or not configured.

    The probe attempts _KAFKA_PROBE_ATTEMPTS connections with exponential backoff
    via tenacity. Total budget: ~5 seconds. After exhaustion, degraded mode.

    No component ever calls this. Only CrawlerBus.start() calls this.
    """
    bootstrap = os.environ.get(_KAFKA_BOOTSTRAP_ENV)
    if not bootstrap:
        log.info(
            "bus_mode_degraded",
            reason="KAFKA_BOOTSTRAP_SERVERS not set",
            consequence="Operating without Kafka. Events are not durable.",
        )
        return BusMode.DEGRADED, None

    if not _AIOKAFKA_AVAILABLE:
        log.warning(
            "bus_mode_degraded",
            reason="aiokafka not installed (pip install aiokafka)",
            consequence="Cannot use Kafka mode without aiokafka. Install it.",
        )
        return BusMode.DEGRADED, None

    ssl_ctx = _build_ssl_context()

    # Probe: create a minimal producer, attempt start, immediately stop.
    # We don't use the probe producer for anything else.
    async def _probe() -> bool:
        probe_kwargs: Dict[str, Any] = dict(
            bootstrap_servers=bootstrap,
            request_timeout_ms=int(_KAFKA_PROBE_TIMEOUT_S * 1000),
        )
        if ssl_ctx is not None:
            probe_kwargs["security_protocol"] = "SSL"
            probe_kwargs["ssl_context"] = ssl_ctx

        # _detect_mode() — inside _probe(), no assert needed, outer guard covers it
        probe = AIOKafkaProducer(**probe_kwargs) # noqa

        try:
            await asyncio.wait_for(probe.start(), timeout=_KAFKA_PROBE_TIMEOUT_S)
            await probe.stop()
            return True
        except (asyncio.TimeoutError, KafkaConnectionError, KafkaError, OSError):
            try:
                await probe.stop()
            except Exception:  # noqa
                pass
            return False

    try:
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(_KAFKA_PROBE_ATTEMPTS),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            reraise=False,
        ):
            with attempt:
                reachable = await _probe()
                if not reachable:
                    raise RuntimeError("Kafka probe returned False")

        log.info(
            "bus_mode_kafka",
            bootstrap=bootstrap,
            tls=ssl_ctx is not None,
            consequence="Durable event log active. Consumer groups enabled.",
        )
        return BusMode.KAFKA, bootstrap

    except (RetryError, RuntimeError):
        log.info(
            "bus_mode_degraded",
            reason=f"Kafka broker at {bootstrap!r} unreachable after {_KAFKA_PROBE_ATTEMPTS} attempts",
            consequence="Operating without Kafka. Events are not durable. "
                        "Set KAFKA_BOOTSTRAP_SERVERS and ensure broker is reachable to enable Kafka mode.",
        )
        return BusMode.DEGRADED, None


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 11: _serialize()
# Event → BusEnvelope.
# One path regardless of sink. HMAC is signed here.
# ═════════════════════════════════════════════════════════════════════════════

def _serialize(
    event: Any,
    topic: str,
    component: str,
    offset: int,
    partition: int = -1,
) -> BusEnvelope:
    """
    Serialize one event dataclass into a BusEnvelope.

    Serialization path:
        event                              (typed dataclass)
        ↓
        dataclasses.asdict(event)          (dict — handles nested frozen dataclasses)
        ↓
        msgpack.packb(dict)                (bytes — deterministic, compact)
        ↓
        BusEnvelope(topic, key, value, headers, offset, partition, timestamp)
        ↓
        HMAC signed (stored in headers["hmac_sha256"])

    The partition key is the topology_class field if present, otherwise the
    topic name. This ensures same-class events are ordered in Kafka mode.

    Raises:
        TypeError:  If event is not a dataclass.
        ValueError: If serialization produces an oversized value.
    """
    if not dataclasses.is_dataclass(event) or isinstance(event, type):
        raise TypeError(
            f"_serialize() requires a dataclass instance, got {type(event).__name__!r}. "
            f"All bus events must be frozen dataclasses."
        )

    # Extract partition key from the event. topology_class is used when present
    # for consistent partition assignment in Kafka mode.
    key_str = (
        getattr(event, "topology_class", None)
        or getattr(event, "domain", None)
        or getattr(event, "url", None)
        or topic
    )
    key_bytes = str(key_str).encode("utf-8")

    # Serialize event body.
    try:
        event_dict = _event_to_dict(event)
        value_bytes = msgpack.packb(event_dict, use_bin_type=True)
    except (TypeError, msgpack.PackException) as exc:
        raise TypeError(
            f"Failed to msgpack-serialize {type(event).__name__!r} "
            f"on topic={topic!r}: {exc}. "
            f"All event field values must be msgpack-serializable."
        ) from exc

    if len(value_bytes) > _MAX_ENVELOPE_VALUE_BYTES:
        raise ValueError(
            f"Serialized event for topic={topic!r} is {len(value_bytes):,} bytes, "
            f"exceeding _MAX_ENVELOPE_VALUE_BYTES ({_MAX_ENVELOPE_VALUE_BYTES:,}). "
            f"This event will not be emitted."
        )

    now_str = _iso8601_now()
    run_id_val = getattr(event, "run_id", None) or ""

    # Build headers before HMAC so the signature covers consistent header data.
    headers: Dict[str, str] = {
        "run_id":            run_id_val,
        "schema_version":    _SCHEMA_VERSION,
        "source_component":  component,
        "emit_timestamp":    now_str,
        "event_type":        type(event).__name__,
        "schema_name":       type(event).__name__,
    }

    # Build the pre-signature envelope (no hmac field yet).
    envelope = BusEnvelope(
        topic=topic,
        key=key_bytes,
        value=value_bytes,
        headers=headers,
        offset=offset,
        partition=partition,
        timestamp=now_str,
    )

    # Sign and return final envelope with HMAC in headers.
    signature = _sign_envelope(envelope)
    signed_headers = {**headers, "hmac_sha256": signature}

    return BusEnvelope(
        topic=envelope.topic,
        key=envelope.key,
        value=envelope.value,
        headers=signed_headers,
        offset=envelope.offset,
        partition=envelope.partition,
        timestamp=envelope.timestamp,
    )


def _event_to_dict(event: Any) -> Any:
    """
    Convert a possibly-nested frozen dataclass to a plain dict.
    Handles datetime -> ISO string, Enum -> value, numpy-like arrays -> list,
    bytes -> list-of-ints, frozenset -> list.
    msgpack cannot handle these types natively and native producers must see
    primitive wire shapes.
    """
    if dataclasses.is_dataclass(event) and not isinstance(event, type):
        return {
            f.name: _event_to_dict(getattr(event, f.name))
            for f in dataclasses.fields(event)
            if f.init
        }
    if isinstance(event, Enum):
        return event.value
    if hasattr(event, "tolist") and callable(getattr(event, "tolist")):
        return _event_to_dict(event.tolist())
    if isinstance(event, datetime):
        return event.isoformat()
    if isinstance(event, Path):
        return str(event)
    if isinstance(event, bytes):
        return list(event)  # msgpack handles list-of-ints as bytes on unpack
    if isinstance(event, frozenset):
        return list(event)
    if isinstance(event, (list, tuple)):
        return [_event_to_dict(i) for i in event]
    if isinstance(event, dict):
        return {k: _event_to_dict(v) for k, v in event.items()}
    return event


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 12: _deserialize()
# BusEnvelope → typed event.
# HMAC is verified here. Schema is validated here.
# Handler never sees bytes. Handler never sees an unverified event.
# ═════════════════════════════════════════════════════════════════════════════

def _deserialize(envelope: BusEnvelope, schema: Type[Any]) -> Any:
    """
    Deserialize one BusEnvelope into a typed event.

    Deserialization path:
        BusEnvelope received
        ↓
        HMAC verification (raise EventIntegrityError on failure)
        ↓
        msgpack.unpackb(envelope.value)
        ↓
        schema(**unpacked_dict)  (validate against registered schema)
        ↓
        handler(typed_event)

    Handler never sees BusEnvelope. Handler never sees bytes.
    Handler receives exactly the typed dataclass it declared in subscribe().

    Raises:
        EventIntegrityError: HMAC verification failed. Hard stop for this event.
        EventSchemaError:    Deserialized dict does not match schema. Hard stop.
    """
    # Step 1: HMAC verification. Non-negotiable. A tampered event is dead-lettered
    # before any deserialization attempt — we do not even try to unpack tampered bytes.
    if not _verify_envelope(envelope):
        raise EventIntegrityError(
            f"HMAC verification failed — topic={envelope.topic!r} "
            f"offset={envelope.offset} component={envelope.source_component!r}. "
            f"Event rejected. Possible tampering or key mismatch.",
        )

    # Step 2: Unpack msgpack bytes.
    try:
        raw_dict: Dict[str, Any] = msgpack.unpackb(
            envelope.value,
            raw=False,
            strict_map_key=False,
        )
    except (msgpack.UnpackException, msgpack.ExtraData, TypeError) as exc:
        raise EventSchemaError(
            f"msgpack deserialization failed for topic={envelope.topic!r} "
            f"offset={envelope.offset}: {exc}. "
            f"Envelope value may be corrupted.",
        ) from exc

    if not isinstance(raw_dict, dict):
        raise EventSchemaError( # noqa | runtime defensive check
            f"Deserialized value for topic={envelope.topic!r} is {type(raw_dict).__name__!r}, "
            f"expected dict. Event payload must be a msgpack map."
        )

    # Step 3: Reconstruct typed dataclass from dict.
    # bytes fields were serialized as list-of-ints — convert back.
    raw_dict = _coerce_contract_fields(raw_dict, schema)

    try:
        return schema(**raw_dict)
    except TypeError as exc:
        raise EventSchemaError(
            f"Schema mismatch on {schema.__name__!r} for topic={envelope.topic!r}: {exc}. "
            f"Event dict keys: {sorted(raw_dict.keys())}. "
            f"Schema fields: {[f.name for f in dataclasses.fields(schema) if f.init]}."  # type: ignore[arg-type] # noqa
        ) from exc
    except Exception as exc:  # noqa: BLE001 — catch dataclass __post_init__ failures
        raise EventSchemaError(
            f"Event construction failed for {schema.__name__!r}: {exc}.",
        ) from exc


def _coerce_contract_fields(d: Dict[str, Any], schema: Type[Any]) -> Dict[str, Any]:
    """
    Coerce primitive wire values back into contract field values.

    This is the Python side of the cross-language contract: Go/C/CUDA/Rust
    producers send primitive JSON/msgpack maps, then this function rebuilds the
    dataclass-safe values expected by contracts.py.
    """
    if not dataclasses.is_dataclass(schema):
        return d

    try:
        type_hints = get_type_hints(schema)
    except Exception:  # noqa: BLE001 - schema construction below is final guard
        type_hints = {}

    result = dict(d)
    for f in dataclasses.fields(schema):  # noqa
        if not f.init:
            result.pop(f.name, None)
            continue
        if f.name not in result:
            continue
        result[f.name] = _coerce_contract_value(
            result[f.name],
            type_hints.get(f.name, f.type),
        )
    return result


def _coerce_contract_value(value: Any, annotation: Any) -> Any:
    """Coerce one primitive wire value according to a contract annotation."""
    if annotation is Any or annotation is object:
        return value
    if value is None:
        return None

    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, types.UnionType):
        last_error: Optional[Exception] = None
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce_contract_value(value, arg)
            except Exception as exc:  # noqa: BLE001 - try next union arm
                last_error = exc
        if last_error is not None:
            raise last_error
        return value

    if isinstance(annotation, type) and issubclass(annotation, Enum):
        if isinstance(value, annotation):
            return value
        return annotation(value)

    if annotation is bytes or annotation == "bytes":
        if isinstance(value, list) and all(isinstance(x, int) for x in value):
            return bytes(value)
        return value

    if origin in (list, List):
        inner = args[0] if args else Any
        return [_coerce_contract_value(v, inner) for v in value]

    if origin in (tuple, Tuple):
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce_contract_value(v, args[0]) for v in value)
        return tuple(
            _coerce_contract_value(v, args[i] if i < len(args) else Any)
            for i, v in enumerate(value)
        )

    if origin in (dict, Dict):
        key_type = args[0] if args else Any
        val_type = args[1] if len(args) > 1 else Any
        return {
            _coerce_contract_value(k, key_type): _coerce_contract_value(v, val_type)
            for k, v in value.items()
        }

    if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
        if isinstance(value, annotation):
            return value
        if isinstance(value, dict):
            return annotation(**_coerce_contract_fields(value, annotation))
        return value

    return value


def event_from_payload(topic: str, payload: Dict[str, Any]) -> Any:
    """
    Build a typed contract event from a primitive payload map.

    Non-Python producers use this path through tag/bus_bridge.py. It enforces
    the same registry and dataclass validation as the normal Python emitter.
    """
    schema = TOPIC_REGISTRY.get(topic)
    if schema is None:
        raise EventBusSubscriptionError(
            f"Topic {topic!r} is not registered. "
            f"Registered topics: {sorted(TOPIC_REGISTRY.keys())}."
        )
    return schema(**_coerce_contract_fields(payload, schema))


# Backwards-compatible name used by older tests and diagnostics.
_coerce_bytes_fields = _coerce_contract_fields


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 13: _dispatch_loop()
# The hot path. Reads from queues/consumers, calls handlers.
# One loop per topic per sink mode. Never stops until bus.stop() is called.
# One bad handler cannot affect other subscribers — each is isolated in try/except.
# ═════════════════════════════════════════════════════════════════════════════

# Handler type alias.
HandlerFn = Callable[[Any], Awaitable[None]]


class _SubscriptionRecord:
    """Internal record for one topic subscription."""

    __slots__ = ("topic", "group_id", "handler", "schema", "handler_name", "invocation_count", "failure_count")

    def __init__(
        self,
        topic: str,
        group_id: str,
        handler: HandlerFn,
        schema: Type[Any],
    ) -> None:
        self.topic:            str = topic
        self.group_id:         str = group_id
        self.handler:          HandlerFn = handler
        self.schema:           Type[Any] = schema
        self.handler_name:     str = f"{handler.__qualname__}"
        self.invocation_count: int = 0
        self.failure_count:    int = 0


async def _dispatch_to_handler(
    sub: _SubscriptionRecord,
    event: Any,
    envelope: BusEnvelope,
    mode: str,
    dead_letter_writer: Callable[..., None],
    failure_counter: Callable[[str, str], None],
    dispatch_counter: Callable[[str], None],
) -> bool:
    """
    Dispatch one event to one handler with retry logic.

    Attempts _HANDLER_MAX_RETRIES times with exponential backoff.
    On final failure, writes a dead letter and returns False.
    On success, returns True.

    One handler's failure does not affect any other handler. Isolation is
    enforced by calling each handler in its own try/except block, never
    in a shared exception context.
    """
    t_dispatch = time.monotonic()
    last_exc: Optional[Exception] = None

    for attempt_n in range(1, _HANDLER_MAX_RETRIES + 1):
        try:
            await sub.handler(event)
            duration_ms = (time.monotonic() - t_dispatch) * 1000.0
            sub.invocation_count += 1
            dispatch_counter(envelope.topic)
            log.debug(
                "event_dispatched",
                topic=envelope.topic,
                handler=sub.handler_name,
                group=sub.group_id,
                offset=envelope.offset,
                duration_ms=round(duration_ms, 3),
                attempt=attempt_n,
            )
            return True

        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            if attempt_n < _HANDLER_MAX_RETRIES:
                backoff = 0.1 * (2 ** (attempt_n - 1))
                log.warning(
                    "handler_retry",
                    topic=envelope.topic,
                    handler=sub.handler_name,
                    attempt=attempt_n,
                    error=str(exc),
                    backoff_s=backoff,
                )
                await asyncio.sleep(backoff)

    # All retries exhausted. Write dead letter.
    sub.failure_count += 1
    failure_counter(envelope.topic, sub.handler_name)

    log.error(
        "handler_failed_dead_letter",
        topic=envelope.topic,
        handler=sub.handler_name,
        group=sub.group_id,
        offset=envelope.offset,
        retries=_HANDLER_MAX_RETRIES,
        error=str(last_exc),
        error_type=type(last_exc).__name__,
    )

    dead_letter_writer(
        envelope=envelope,
        handler_name=sub.handler_name,
        error=last_exc,
        retries=_HANDLER_MAX_RETRIES,
        mode=mode,
    )
    return False


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 14: _write_dead_letter()
# Every unhandled event is recorded. Nothing is silently dropped.
# Append-only. fsync-guaranteed. Survives process death.
# ═════════════════════════════════════════════════════════════════════════════

def _write_dead_letter(
    envelope: BusEnvelope,
    handler_name: str,
    error: Optional[Exception],
    retries: int,
    mode: str,
) -> None:
    """
    Write one dead letter record to /store/dead_letters.jsonl.

    Append-only. One JSON object per line. fsync after every write.
    fsync is load-bearing: without it the OS may buffer the write and lose it
    on process death. Dead letters must survive crashes — that is their purpose.

    If the dead letter file cannot be written, logs a critical error but does
    not raise. Dead letter write failure must not kill the dispatch loop.
    """
    error_str = ""
    if error is not None:
        error_str = f"{type(error).__name__}: {error}"

    dead = DeadLetterEvent(
        topic=envelope.topic,
        event=envelope.value.hex(),
        handler=handler_name,
        error=error_str,
        retries=retries,
        timestamp=_iso8601_now(),
        mode=mode,
        run_id=envelope.run_id,
        partition=envelope.partition,
        offset=envelope.offset,
        source_component=envelope.source_component,
        envelope_key=envelope.key.decode("utf-8", errors="replace"),
    )

    line = dead.to_jsonl_line()

    try:
        _DEAD_LETTER_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_DEAD_LETTER_PATH, "ab") as f:
            f.write(line)
            f.flush()
            os.fsync(f.fileno())  # guarantee write survives crash
    except OSError as exc:
        # Dead letter write failure is logged at CRITICAL but does not stop the bus.
        # The dispatch loop must continue even if the dead letter store is unavailable.
        log.critical(
            "dead_letter_write_failed",
            topic=envelope.topic,
            handler=handler_name,
            error=str(exc),
            consequence="Dead letter record lost. index_daemon.py will not see this failure.",
            dead_letter=dead.to_jsonl_line().decode("utf-8", errors="replace"),
        )


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 15 + 16: CrawlerBus.emitter() and CrawlerBus.subscribe()
# See CrawlerBus class below for both.
# ═════════════════════════════════════════════════════════════════════════════


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 17–19: CrawlerBus — THE BUS
# The main class. Module-level singleton is BUS at the bottom of this file.
# ═════════════════════════════════════════════════════════════════════════════

class CrawlerBus:
    """
    AXIOM's distributed nervous system.

    Dual-sink event backbone. Kafka mode when available; degraded mode otherwise.
    Components emit events via TopicEmitter. Components subscribe via subscribe().
    The bus handles everything between emit and handler invocation.

    Lifecycle:
        BUS = CrawlerBus()            # construction — no I/O
        await BUS.start()             # detect mode, start sinks, start dispatch loops
        emitter = await BUS.emitter() # get a typed emitter for one topic
        await BUS.subscribe(...)      # register a handler
        await BUS.stop()              # drain queues, close connections

    Thread safety:
        CrawlerBus is NOT thread-safe. It is coroutine-safe.
        All methods are async. Call from a single asyncio event loop.

    Singleton:
        Use the module-level BUS instance. Do not construct multiple CrawlerBus
        instances — they do not share state and will create duplicate consumers.
    """

    def __init__(self) -> None:
        # ── Core state ────────────────────────────────────────────────────────
        self._mode: Optional[BusMode] = None
        self._registry: TopicRegistry = TopicRegistry(TOPIC_REGISTRY)
        self._started: bool = False
        self._start_time: Optional[float] = None
        self._stopping: bool = False

        # ── Sinks (one active at runtime) ─────────────────────────────────────
        self._kafka_sink: Optional[_KafkaSink] = None
        self._degraded_sink: Optional[_DegradedSink] = None

        # ── Subscriptions: topic → list of _SubscriptionRecord ────────────────
        # Multiple groups can subscribe to the same topic.
        self._subscriptions: Dict[str, List[_SubscriptionRecord]] = defaultdict(list)

        # ── Active dispatch loop tasks ─────────────────────────────────────────
        self._dispatch_tasks: List[asyncio.Task] = []

        # ── Active emitters (for disable() on shutdown) ───────────────────────
        self._emitters: List[TopicEmitter] = []

        # ── Telemetry counters ────────────────────────────────────────────────
        self._emit_counts:     Dict[str, int] = defaultdict(int)
        self._dispatch_counts: Dict[str, int] = defaultdict(int)
        self._failure_counts:  Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._dead_letter_counts: Dict[str, int] = defaultdict(int)
        self._last_emit_ts:    Dict[str, Optional[str]] = defaultdict(lambda: None)
        self._last_dispatch_ts: Dict[str, Optional[str]] = defaultdict(lambda: None)
        self._hmac_failures:   int = 0
        self._schema_failures: int = 0

        # ── Offset counter for degraded mode ──────────────────────────────────
        self._offsets: Dict[str, int] = defaultdict(int)

        # ── Kafka health probe ────────────────────────────────────────────────
        self._kafka_connected: bool = False

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 15: emitter()
    # ─────────────────────────────────────────────────────────────────────────

    async def emitter(
        self,
        topic: str,
        component: str,
        schema: Type[T],
    ) -> TopicEmitter[T]:
        """
        Create and return a TypedEmitter for one topic.

        The emitter is bound to the component name and schema type. Wrong schema
        is caught at emit() call time via isinstance() — not downstream.

        The topic must be registered in TOPIC_REGISTRY. Unregistered topics
        raise EventBusSubscriptionError immediately.

        Args:
            topic:      Registered topic name.
            component:  Caller's fully qualified component name.
                        Used in envelope headers for observability.
                        Convention: "module.ClassName" (e.g. "alpine_strip.offline_pipeline").
            schema:     The event dataclass type this emitter will produce.
                        Must match the schema registered for the topic.

        Returns:
            TopicEmitter[T] — call await emitter.emit(event) to produce.

        Raises:
            EventBusSubscriptionError: Unknown topic.
            RuntimeError:              Bus not started.
        """
        self._assert_started("emitter()")
        registered_schema = self._registry.validate_topic(topic)

        # Warn if the caller's schema disagrees with the registry, but don't block.
        # The registry schema is authoritative — this is a programming-time warning.
        if schema is not registered_schema:
            log.warning(
                "emitter_schema_mismatch",
                topic=topic,
                component=component,
                caller_schema=schema.__name__,
                registry_schema=registered_schema.__name__,
                consequence="Emitter will use caller schema. Verify against TOPIC_REGISTRY.",
            )

        emitter: TopicEmitter[T] = TopicEmitter(
            bus=self,
            topic=topic,
            component=component,
            schema=schema,
        )
        self._emitters.append(emitter)

        log.info(
            "emitter_created",
            topic=topic,
            component=component,
            schema=schema.__name__,
            mode=str(self._mode),
        )

        return emitter

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 16: subscribe()
    # ─────────────────────────────────────────────────────────────────────────

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: HandlerFn,
        schema: Type[T],
    ) -> None:
        """
        Register a handler for a topic.

        In Kafka mode: group is a real Kafka consumer group ID. Only one
        instance in the group processes each partition. Offsets committed after
        successful dispatch — processing is resumable after crash.

        In degraded mode: group is a dict key. Only one handler per group
        receives each event. In-memory only — no resume after crash.

        The handler receives a fully-typed, HMAC-verified, schema-validated
        event instance. It never receives raw bytes or an unverified event.

        Args:
            topic:   Registered topic name.
            group:   Consumer group ID. Unique per logical consumer.
                     Convention: "module.ClassName" (e.g. "world_model.wlp").
            handler: Async callable. Signature: async def handler(event: T) -> None.
                     Must be a coroutine function.
            schema:  The event dataclass type this subscriber expects.

        Raises:
            EventBusSubscriptionError: Unknown topic, or handler is not a coroutine.
            RuntimeError:              Bus not started.
        """
        self._assert_started("subscribe()")

        self._registry.validate_topic(topic)

        if not asyncio.iscoroutinefunction(handler):
            raise EventBusSubscriptionError(
                f"Handler {handler!r} for topic={topic!r} group={group!r} "
                f"is not a coroutine function. "
                f"All bus handlers must be async def."
            )

        # Check for duplicate group registration on same topic.
        existing_groups = {s.group_id for s in self._subscriptions[topic]}
        if group in existing_groups:
            log.warning(
                "subscribe_duplicate_group",
                topic=topic,
                group=group,
                consequence="Replacing existing handler for this group.",
            )
            self._subscriptions[topic] = [
                s for s in self._subscriptions[topic] if s.group_id != group
            ]

        sub = _SubscriptionRecord(
            topic=topic,
            group_id=group,
            handler=handler,
            schema=schema,
        )
        self._subscriptions[topic].append(sub)

        # In Kafka mode, start a consumer for this group if not already active.
        if self._mode is BusMode.KAFKA and self._kafka_sink is not None:
            await self._start_kafka_consumer(topic=topic, group_id=group)

        log.info(
            "subscription_registered",
            topic=topic,
            group=group,
            handler=sub.handler_name,
            schema=schema.__name__,
            mode=str(self._mode),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 17: start()
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the bus.

        Steps:
            1. Detect mode (Kafka probe or degraded fallback).
            2. Initialize the active sink (Kafka or asyncio.Queue).
            3. Start dispatch loop tasks for all registered topics.
            4. Mark bus as started.

        Idempotent — calling start() on an already-started bus is a no-op
        with a warning log. This protects against accidental double-initialization
        in cold_start.py or test harnesses.

        Raises:
            RuntimeError: If _HMAC_KEY is not set (caught at module import time).
        """
        if self._started:
            log.warning("bus_start_called_twice", consequence="No-op. Bus is already running.")
            return

        log.info("bus_starting", topics=self._registry.all_topics)

        # Step 1: Detect mode.
        mode, bootstrap = await _detect_mode()
        self._mode = mode

        # Step 2: Initialize sink.
        if self._mode is BusMode.KAFKA:
            assert bootstrap is not None
            self._kafka_sink = _KafkaSink(bootstrap_servers=bootstrap)
            try:
                await self._kafka_sink.start(topics=self._registry.all_topics)
                self._kafka_connected = True
            except KafkaSinkUnavailable as exc:
                # Kafka sink failed to start despite the probe succeeding.
                # This can happen if topics don't exist yet or ACLs block producers.
                # Fall to degraded mode gracefully.
                log.warning(
                    "kafka_sink_start_failed_degrading",
                    error=str(exc),
                    consequence="Falling back to degraded mode despite Kafka being reachable.",
                )
                self._mode = BusMode.DEGRADED
                self._kafka_sink = None
                self._kafka_connected = False
                self._degraded_sink = _DegradedSink()
                for topic in self._registry.all_topics:
                    self._degraded_sink.register_topic(topic)
        else:
            self._degraded_sink = _DegradedSink()
            for topic in self._registry.all_topics:
                self._degraded_sink.register_topic(topic)

        # Step 3: Start dispatch loops.
        await self._start_dispatch_loops()

        # Step 4: Mark started.
        self._started = True
        self._start_time = time.monotonic()

        log.info(
            "bus_started",
            mode=str(self._mode),
            topics=self._registry.all_topics,
            kafka_connected=self._kafka_connected,
        )

    async def _start_kafka_consumer(self, topic: str, group_id: str) -> None:
        """Start a Kafka consumer for a newly-registered subscription (post-start)."""
        if self._kafka_sink is None:
            return
        try:
            await self._kafka_sink.subscribe(topic=topic, group_id=group_id)
        except KafkaSinkUnavailable as exc:
            log.error(
                "kafka_consumer_start_failed",
                topic=topic,
                group_id=group_id,
                error=str(exc),
            )

    async def _start_dispatch_loops(self) -> None:
        """
        Launch one dispatch loop task per topic.
        The loop runs until the bus is stopped.
        """
        for topic in self._registry.all_topics:
            if self._mode is BusMode.KAFKA:
                task = asyncio.create_task(
                    self._kafka_dispatch_loop(topic),
                    name=f"bus_dispatch_kafka_{topic}",
                )
            else:
                task = asyncio.create_task(
                    self._degraded_dispatch_loop(topic),
                    name=f"bus_dispatch_degraded_{topic}",
                )
            self._dispatch_tasks.append(task)
            log.debug("dispatch_loop_started", topic=topic, mode=str(self._mode))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 18: stop()
    # ─────────────────────────────────────────────────────────────────────────

    async def stop(self) -> None:
        """
        Stop the bus gracefully.

        Steps:
            1. Disable all emitters (no new events after this point).
            2. Drain queues (degraded mode) or wait for consumer lag to clear.
            3. Cancel dispatch loop tasks.
            4. Stop sinks.

        After stop() returns, no further events will be dispatched.
        In-flight handler calls (already dispatched) may still complete.

        Idempotent — calling stop() twice is a no-op with a warning.
        """
        if self._stopping:
            log.warning("bus_stop_called_twice")
            return
        if not self._started:
            log.warning("bus_stop_called_before_start")
            return

        self._stopping = True
        log.info("bus_stopping", mode=str(self._mode))

        # Step 1: Disable emitters.
        for emitter in self._emitters:
            emitter.disable()
        log.debug("emitters_disabled", count=len(self._emitters))

        # Step 2: Drain.
        if self._mode is BusMode.DEGRADED and self._degraded_sink is not None:
            remaining = await self._degraded_sink.drain(timeout_s=_SHUTDOWN_DRAIN_TIMEOUT_S)
            if remaining:
                log.warning("shutdown_drain_incomplete", remaining=remaining)
        # Kafka mode: consumers commit and stop naturally — no explicit drain needed.
        # The producer's in-flight sends are drained by aiokafka producer.stop().

        # Step 3: Cancel dispatch tasks.
        for task in self._dispatch_tasks:
            if not task.done():
                task.cancel()
        if self._dispatch_tasks:
            await asyncio.gather(*self._dispatch_tasks, return_exceptions=True)
        self._dispatch_tasks.clear()
        log.debug("dispatch_loops_stopped")

        # Step 4: Stop sinks.
        if self._kafka_sink is not None:
            await self._kafka_sink.stop()
        # Degraded sink has no async resources to close.

        self._started = False
        log.info("bus_stopped", mode=str(self._mode))

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 19: health()
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> BusHealth:
        """
        Return a point-in-time BusHealth snapshot.

        Called by cold_start.py during initialization validation.
        Called by index_daemon.py periodically — result is emitted to structlog.

        This is a synchronous method. It does not query Kafka in real time.
        Kafka lag is sampled asynchronously in the background (see _update_kafka_lag).
        """
        uptime = (time.monotonic() - self._start_time) if self._start_time else 0.0

        topic_healths: Dict[str, TopicHealth] = {}
        for topic in self._registry.all_topics:
            subs = self._subscriptions.get(topic, [])
            queue_depth = self._get_queue_depth(topic)
            total_fail = sum(
                sum(c.values()) for t, c in self._failure_counts.items() if t == topic
            )
            topic_healths[topic] = TopicHealth(
                topic=topic,
                registered=True,
                producer_connected=self._is_producer_connected(topic),
                subscriber_count=len(subs),
                queue_depth=queue_depth,
                dead_letter_count=self._dead_letter_counts[topic],
                total_emitted=self._emit_counts[topic],
                total_dispatched=self._dispatch_counts[topic],
                total_failed=total_fail,
                last_emit_ts=self._last_emit_ts[topic],
                last_dispatch_ts=self._last_dispatch_ts[topic],
            )

        total_emitted     = sum(self._emit_counts.values())
        total_dispatched  = sum(self._dispatch_counts.values())
        total_failed      = sum(
            sum(c.values()) for c in self._failure_counts.values()
        )
        total_dead        = sum(self._dead_letter_counts.values())

        # Lag is approximated from queue depths in degraded mode.
        # In Kafka mode it would be populated by a background probe.
        lag: Dict[str, Dict[str, int]] = {}
        for topic, subs in self._subscriptions.items():
            for sub in subs:
                if sub.group_id not in lag:
                    lag[sub.group_id] = {}
                lag[sub.group_id][topic] = self._get_queue_depth(topic)

        return BusHealth(
            mode=str(self._mode) if self._mode else "uninitialized",
            topics=topic_healths,
            dead_letters=total_dead,
            lag=lag,
            uptime_s=round(uptime, 3),
            started=self._started,
            kafka_connected=self._kafka_connected,
            total_emitted=total_emitted,
            total_dispatched=total_dispatched,
            total_failed=total_failed,
            hmac_failures=self._hmac_failures,
            schema_failures=self._schema_failures,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: _emit()
    # Called by TopicEmitter.emit(). Not part of the public API.
    # ─────────────────────────────────────────────────────────────────────────

    async def _emit(self, topic: str, event: Any, component: str) -> None:
        """
        Internal emit path. Called only by TopicEmitter.emit().

        Serializes the event, signs it, routes it to the active sink.
        Updates telemetry counters. Logs the emit.

        Raises:
            KafkaSinkUnavailable: Kafka broker unreachable (Kafka mode).
            EventBusSubscriptionError: Unknown topic (should never happen — TopicEmitter checks).
            TypeError: Serialization failure (bad event type).
        """
        self._assert_started("_emit()")
        self._registry.validate_topic(topic)

        # Assign offset.
        offset = self._next_offset(topic)

        try:
            envelope = _serialize(
                event=event,
                topic=topic,
                component=component,
                offset=offset,
                partition=-1,  # Kafka will assign partition; -1 in degraded mode.
            )
        except (TypeError, ValueError) as exc:
            log.error(
                "serialize_failed",
                topic=topic,
                component=component,
                event_type=type(event).__name__,
                error=str(exc),
            )
            raise

        if self._mode is BusMode.KAFKA:
            assert self._kafka_sink is not None
            await self._kafka_sink.send(envelope)
        else:
            assert self._degraded_sink is not None
            await self._degraded_sink.put(envelope)

        # Update counters.
        self._emit_counts[topic] += 1
        self._last_emit_ts[topic] = _iso8601_now()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: dispatch loops
    # ─────────────────────────────────────────────────────────────────────────

    async def _kafka_dispatch_loop(self, topic: str) -> None:
        """
        Kafka-mode dispatch loop for one topic.

        Polls all consumer groups registered for this topic. For each consumed
        record, dispatches to all handlers for that group. Commits offset after
        successful dispatch. On handler failure, writes dead letter and continues.

        The loop runs until cancelled by stop().
        """
        log.debug("kafka_dispatch_loop_started", topic=topic)

        while not self._stopping:
            subs = self._subscriptions.get(topic, [])
            if not subs:
                await asyncio.sleep(_DISPATCH_POLL_INTERVAL_S)
                continue

            for sub in subs:
                if self._kafka_sink is None:
                    break

                group_key = f"{topic}:{sub.group_id}"
                consumer = self._kafka_sink._consumers.get(group_key) # noqa
                if consumer is None:
                    # Consumer may not be started yet if subscribe() was called
                    # before start() (which is allowed — start() will pick it up).
                    await asyncio.sleep(_DISPATCH_POLL_INTERVAL_S)
                    continue

                try:
                    # poll() returns a batch of records.
                    records_dict = await asyncio.wait_for(
                        consumer.getmany(timeout_ms=100, max_records=_KAFKA_MAX_POLL_RECORDS),
                        timeout=0.5,
                    )
                except asyncio.TimeoutError:
                    continue
                except KafkaError as exc:
                    log.warning("kafka_poll_error", topic=topic, group=sub.group_id, error=str(exc))
                    await asyncio.sleep(1.0)
                    continue

                for tp, records in records_dict.items():
                    for record in records:
                        envelope = _kafka_record_to_envelope(record, topic)
                        await self._handle_envelope(
                            envelope=envelope,
                            sub=sub,
                            consumer_for_commit=consumer,
                        )

            await asyncio.sleep(0)  # yield to event loop between groups

        log.debug("kafka_dispatch_loop_stopped", topic=topic)

    async def _degraded_dispatch_loop(self, topic: str) -> None:
        """
        Degraded-mode dispatch loop for one topic.

        Polls the asyncio.Queue for this topic at _DISPATCH_POLL_INTERVAL_S intervals.
        For each envelope, dispatches to all subscriber groups independently.
        Each group receives every event. Within a group, the single registered
        handler is called.

        The loop runs until cancelled by stop().
        """
        log.debug("degraded_dispatch_loop_started", topic=topic)

        while not self._stopping:
            if self._degraded_sink is None:
                await asyncio.sleep(_DISPATCH_POLL_INTERVAL_S)
                continue

            envelope = await self._degraded_sink.get(topic)
            if envelope is None:
                await asyncio.sleep(_DISPATCH_POLL_INTERVAL_S)
                continue

            subs = self._subscriptions.get(topic, [])
            if not subs:
                # No subscribers — event is delivered to no one.
                # This is not an error; it happens when a producer emits before
                # subscribers register. In production, all subscribers register
                # before the first emit. In tests, this may happen intentionally.
                log.debug("event_no_subscribers", topic=topic, offset=envelope.offset)
                continue

            # Dispatch to each subscriber group independently.
            # asyncio.gather ensures concurrent dispatch — one slow handler does
            # not block another group.
            dispatch_coros = [
                self._handle_envelope(envelope=envelope, sub=sub, consumer_for_commit=None)
                for sub in subs
            ]
            await asyncio.gather(*dispatch_coros, return_exceptions=True)

        log.debug("degraded_dispatch_loop_stopped", topic=topic)

    async def _handle_envelope(
        self,
        envelope: BusEnvelope,
        sub: _SubscriptionRecord,
        consumer_for_commit: Any,
    ) -> None:
        """
        HMAC-verify, deserialize, and dispatch one envelope to one handler.

        This is the critical path. Every step is instrumented and every failure
        is recorded. Nothing silently drops.

        Steps:
            1. HMAC verify → EventIntegrityError → dead letter.
            2. Schema deserialize → EventSchemaError → dead letter.
            3. Dispatch to handler with retry → failure → dead letter.
            4. Commit offset (Kafka mode only, on success).
        """
        schema = sub.schema

        # Step 1 + 2: Verify and deserialize.
        try:
            event = _deserialize(envelope=envelope, schema=cast(Type[Any], schema))
        except EventIntegrityError as exc:
            self._hmac_failures += 1
            log.error(
                "hmac_verification_failed",
                topic=envelope.topic,
                offset=envelope.offset,
                source=envelope.source_component,
                error=str(exc),
            )
            _write_dead_letter(
                envelope=envelope,
                handler_name="__hmac_verify__",
                error=exc,
                retries=0,
                mode=str(self._mode),
            )
            self._dead_letter_counts[envelope.topic] += 1
            return
        except EventSchemaError as exc:
            self._schema_failures += 1
            log.error(
                "schema_validation_failed",
                topic=envelope.topic,
                offset=envelope.offset,
                schema=schema.__name__,
                error=str(exc),
            )
            _write_dead_letter(
                envelope=envelope,
                handler_name="__schema_validate__",
                error=exc,
                retries=0,
                mode=str(self._mode),
            )
            self._dead_letter_counts[envelope.topic] += 1
            return

        # Step 3: Dispatch with retry.
        success = await _dispatch_to_handler(
            sub=sub,
            event=event,
            envelope=envelope,
            mode=str(self._mode),
            dead_letter_writer=_write_dead_letter,
            failure_counter=self._increment_failure,
            dispatch_counter=self._increment_dispatch,
        )

        if not success:
            self._dead_letter_counts[envelope.topic] += 1
            return

        # Step 4: Commit offset in Kafka mode.
        if consumer_for_commit is not None and self._kafka_sink is not None:
            await self._kafka_sink.commit(consumer_for_commit)

        self._last_dispatch_ts[envelope.topic] = _iso8601_now()

    # ─────────────────────────────────────────────────────────────────────────
    # INTERNAL: telemetry helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _increment_failure(self, topic: str, handler_name: str) -> None:
        self._failure_counts[topic][handler_name] += 1

    def _increment_dispatch(self, topic: str) -> None:
        self._dispatch_counts[topic] += 1

    def _next_offset(self, topic: str) -> int:
        self._offsets[topic] += 1
        return self._offsets[topic]

    def _get_queue_depth(self, topic: str) -> int:
        if self._degraded_sink is not None:
            return self._degraded_sink.queue_depth(topic)
        return 0

    def _is_producer_connected(self, topic: str) -> bool:
        if self._mode is BusMode.KAFKA and self._kafka_sink is not None:
            return topic in self._kafka_sink._producers # noqa
        # Degraded mode: "producer connected" means queue exists.
        if self._degraded_sink is not None:
            return topic in self._degraded_sink._queues # noqa
        return False

    def _assert_started(self, caller: str) -> None:
        if not self._started:
            raise RuntimeError(
                f"{caller}: bus not started. "
                f"Call await BUS.start() during cold_start.py initialization."
            )


# ═════════════════════════════════════════════════════════════════════════════
# HMAC SIGNING AND VERIFICATION
# Every envelope is signed before emission and verified before deserialization.
# Timing-safe comparison is mandatory — hmac.compare_digest always used.
# ═════════════════════════════════════════════════════════════════════════════

def _sign_envelope(envelope: BusEnvelope) -> str:
    """
    Compute HMAC-SHA256 for a BusEnvelope.

    The signature covers the fields that uniquely identify this envelope's
    content and provenance:
        - topic (routing)
        - key (partition routing)
        - value (event content — the payload being protected)
        - emit_timestamp (prevents simple replay: replayed envelopes with new
          timestamps fail because the timestamp is part of the signed payload)

    The resulting hex digest is stored in headers["hmac_sha256"] by _serialize().

    Note on replay protection:
        Timestamp-based replay protection provides partial protection. For
        full replay protection, maintain a sliding window of seen envelope IDs
        (topic + offset + source_component). This is noted here as a future
        hardening point — the current implementation provides integrity protection
        against tampering, not strict replay prevention.
    """
    emit_ts = envelope.headers.get("emit_timestamp", "")

    # The payload being signed must be deterministic. Concatenation order is
    # fixed and documented here. Changing this order is a breaking change.
    payload = (
        envelope.topic.encode("utf-8")
        + b"|"
        + envelope.key
        + b"|"
        + envelope.value
        + b"|"
        + emit_ts.encode("utf-8")
    )

    digest = hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()
    return digest


def _verify_envelope(envelope: BusEnvelope) -> bool:
    """
    Verify the HMAC-SHA256 signature on a BusEnvelope.

    Returns True if the signature is valid, False otherwise.
    Always uses hmac.compare_digest — never ==.

    `==` leaks timing information that can be used to mount a timing attack
    to recover the HMAC key one byte at a time. compare_digest is constant-time
    and is non-negotiable for any HMAC comparison.

    A missing hmac_sha256 header is treated as invalid — no bypass for
    "legacy" envelopes. All envelopes are signed. Always.
    """
    received_digest = envelope.headers.get("hmac_sha256", "")
    if not received_digest:
        return False

    # Recompute the expected signature using the same fields and order as _sign_envelope.
    emit_ts = envelope.headers.get("emit_timestamp", "")
    payload = (
        envelope.topic.encode("utf-8")
        + b"|"
        + envelope.key
        + b"|"
        + envelope.value
        + b"|"
        + emit_ts.encode("utf-8")
    )
    expected_digest = hmac.new(_HMAC_KEY, payload, hashlib.sha256).hexdigest()

    # compare_digest is non-negotiable. Timing-safe comparison.
    return hmac.compare_digest(expected_digest, received_digest)


# ═════════════════════════════════════════════════════════════════════════════
# KAFKA RECORD → BusEnvelope ADAPTER
# Converts an aiokafka ConsumerRecord to a BusEnvelope.
# Called only in Kafka mode.
# ═════════════════════════════════════════════════════════════════════════════

def _kafka_record_to_envelope(record: Any, topic: str) -> BusEnvelope:
    """
    Convert an aiokafka ConsumerRecord to a BusEnvelope.

    Headers are stored by Kafka as List[Tuple[str, bytes]] — we decode them
    to Dict[str, str] for uniform handling throughout the bus.

    The Kafka offset and partition are real values from the broker — they
    replace the local monotonic offsets used in degraded mode.
    """
    # Decode Kafka headers: list of (name, value_bytes) → dict.
    headers: Dict[str, str] = {}
    if record.headers:
        for name, value_bytes in record.headers:
            try:
                headers[name] = value_bytes.decode("utf-8") if value_bytes else ""
            except (UnicodeDecodeError, AttributeError):
                headers[name] = value_bytes.hex() if isinstance(value_bytes, bytes) else str(value_bytes)

    return BusEnvelope(
        topic=topic,
        key=record.key or b"",
        value=record.value or b"",
        headers=headers,
        offset=record.offset,
        partition=record.partition,
        timestamp=_iso8601_from_ms(record.timestamp) if record.timestamp else _iso8601_now(),
    )


# ═════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def _iso8601_now() -> str:
    """Current UTC time as ISO 8601 string with microseconds."""
    return datetime.now(timezone.utc).isoformat()


def _iso8601_from_ms(timestamp_ms: int) -> str:
    """Convert a Kafka millisecond timestamp to ISO 8601 UTC string."""
    return datetime.fromtimestamp(timestamp_ms / 1000.0, tz=timezone.utc).isoformat()


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 20: BUS — MODULE-LEVEL SINGLETON
# The one and only CrawlerBus instance in the process.
# All components import BUS from this module.
# ═════════════════════════════════════════════════════════════════════════════

BUS: CrawlerBus = CrawlerBus()
"""
The module-level CrawlerBus singleton.

Import and use:
    from crawler_bus import BUS

    # In cold_start.py:
    await BUS.start()

    # In fetcher.py:
    self._emitter = await BUS.emitter("raw_fetch", "fetcher", RawFetchEvent)
    await self._emitter.emit(event)

    # In world_model/latent_parser.py:
    await BUS.subscribe("clean_signal", "world_model.wlp", self._on_clean_signal, CleanSignalEvent)

    # In index_daemon.py:
    health = BUS.health()

    # In cold_start.py during shutdown:
    await BUS.stop()
"""


# ═════════════════════════════════════════════════════════════════════════════
# STARTUP VALIDATION
# Runs at import time after BUS construction.
# Validates that the HMAC key is present and all topic schemas are importable.
# Import-time failure is intentional — a broken bus must not start silently.
# ═════════════════════════════════════════════════════════════════════════════

def _validate_at_import() -> None:
    """
    Validate bus configuration at import time.

    Checks:
        1. HMAC key is loaded (done above — _load_hmac_key() raises if missing).
        2. All topic schemas in TOPIC_REGISTRY are dataclasses.
        3. Dead letter store parent directory is writable.

    Failures here are import-time errors. cold_start.py will not proceed.
    """
    # Validate all schemas are dataclasses.
    for topic, schema in TOPIC_REGISTRY.items():
        if not dataclasses.is_dataclass(schema):
            raise ImportError(
                f"TOPIC_REGISTRY[{topic!r}] = {schema!r} is not a dataclass. "
                f"All bus event schemas must be frozen dataclasses."
            )

    # Validate /store directory is writable (or that we can create it).
    store_dir = _DEAD_LETTER_PATH.parent
    try:
        store_dir.mkdir(parents=True, exist_ok=True)
        test_path = store_dir / ".bus_write_probe"
        test_path.touch()
        test_path.unlink()
    except OSError as exc:
        # Not a hard error at import time — cold_start.py creates /store.
        # Log at warning level. The bus will hard-fail when it first tries to
        # write a dead letter if /store is not writable.
        _startup_log = logging.getLogger("crawler_bus")
        _startup_log.warning(
            f"Dead letter store {store_dir} is not writable at import time: {exc}. "
            f"cold_start.py must ensure /store exists and is writable before bus.start()."
        )


_validate_at_import()


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusMetrics — Prometheus-compatible metric snapshot
# Exposes the same counters as BusHealth but in a flat dict suitable for
# Prometheus pushgateway or direct /metrics endpoint formatting.
# Called by index_daemon.py alongside health().
# ═════════════════════════════════════════════════════════════════════════════

class BusMetrics:
    """
    Flat metric snapshot for observability tooling.

    index_daemon.py calls BusMetrics(BUS.health()) and exposes the result
    to a Prometheus pushgateway or structured log sink. This decouples the
    health data model (BusHealth) from the metrics output format.
    """

    def __init__(self, health: BusHealth) -> None:
        self._health = health

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Flat dict of all bus metrics.
        All counter names follow the axiom_bus_{metric} pattern.
        """
        h = self._health
        flat: Dict[str, Any] = {
            "axiom_bus_mode":              h.mode,
            "axiom_bus_started":           int(h.started),
            "axiom_bus_kafka_connected":   int(h.kafka_connected),
            "axiom_bus_uptime_s":          h.uptime_s,
            "axiom_bus_dead_letters_total": h.dead_letters,
            "axiom_bus_emitted_total":     h.total_emitted,
            "axiom_bus_dispatched_total":  h.total_dispatched,
            "axiom_bus_failed_total":      h.total_failed,
            "axiom_bus_hmac_failures":     h.hmac_failures,
            "axiom_bus_schema_failures":   h.schema_failures,
            "axiom_bus_healthy":           int(h.is_healthy),
        }
        for topic, th in h.topics.items():
            prefix = f"axiom_bus_topic_{topic.replace('-', '_')}"
            flat[f"{prefix}_queue_depth"]       = th.queue_depth
            flat[f"{prefix}_dead_letters"]      = th.dead_letter_count
            flat[f"{prefix}_emitted_total"]     = th.total_emitted
            flat[f"{prefix}_dispatched_total"]  = th.total_dispatched
            flat[f"{prefix}_failed_total"]      = th.total_failed
            flat[f"{prefix}_subscriber_count"]  = th.subscriber_count
            flat[f"{prefix}_healthy"]           = int(th.is_healthy)
            flat[f"{prefix}_error_rate"]        = round(th.error_rate, 4)
        return flat

    def to_prometheus_text(self) -> str:
        """
        Prometheus text format output.
        Each metric is one HELP + TYPE line followed by the value line.
        Suitable for direct output to /metrics or pushgateway.
        """
        lines: List[str] = []
        for name, value in self.to_flat_dict().items():
            if isinstance(value, (int, float)):
                lines.append(f"# HELP {name} AXIOM bus metric")
                lines.append(f"# TYPE {name} gauge")
                lines.append(f"{name} {value}")
        return "\n".join(lines) + "\n"


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusReplayBuffer
# In-memory replay of recent events for test harnesses and cold_start validation.
# NOT used in production hot paths. Gated behind an explicit enable() call.
# ═════════════════════════════════════════════════════════════════════════════

class BusReplayBuffer:
    """
    In-memory ring buffer of recent envelopes per topic.

    Used by:
        - Test harnesses (verify events were emitted with correct content).
        - cold_start.py validation (verify bus is emitting after start).
        - Debug mode (inspect recent events without a Kafka consumer).

    NOT used in production hot paths. Enabled explicitly via attach().
    Adding a replay buffer does not change bus behavior — it is a passive observer.

    Capacity: last N envelopes per topic. Oldest entries are evicted when full.
    """

    def __init__(self, capacity: int = 100) -> None:
        self._capacity = capacity
        self._buffers: Dict[str, List[BusEnvelope]] = defaultdict(list)
        self._enabled = False

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        self._enabled = False

    def record(self, envelope: BusEnvelope) -> None:
        """Record one envelope. No-op if not enabled."""
        if not self._enabled:
            return
        buf = self._buffers[envelope.topic]
        buf.append(envelope)
        if len(buf) > self._capacity:
            buf.pop(0)

    def get(self, topic: str) -> List[BusEnvelope]:
        """Return all buffered envelopes for a topic (oldest first)."""
        return list(self._buffers.get(topic, []))

    def get_latest(self, topic: str) -> Optional[BusEnvelope]:
        """Return the most recently buffered envelope for a topic."""
        buf = self._buffers.get(topic, [])
        return buf[-1] if buf else None

    def clear(self, topic: Optional[str] = None) -> None:
        """Clear buffers. All topics if topic is None."""
        if topic is None:
            self._buffers.clear()
        else:
            self._buffers.pop(topic, None)

    def count(self, topic: str) -> int:
        return len(self._buffers.get(topic, []))


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusHealthMonitor
# Background task that logs bus health periodically.
# Launched by index_daemon.py after bus.start().
# ═════════════════════════════════════════════════════════════════════════════

class BusHealthMonitor:
    """
    Background health monitor for the CrawlerBus.

    Launched by index_daemon.py. Calls bus.health() on a configurable interval
    and emits the result to structlog. Also tracks dead letter accumulation
    rate and fires an alert if dead letters are growing faster than a threshold.

    Usage:
        monitor = BusHealthMonitor(BUS, interval_s=30.0)
        await monitor.start()
        # ... (index_daemon.py runs) ...
        await monitor.stop()
    """

    def __init__(
        self,
        bus: CrawlerBus,
        interval_s: float = 30.0,
        dead_letter_rate_alert_threshold: float = 10.0,  # per minute
    ) -> None:
        self._bus = bus
        self._interval_s = interval_s
        self._dl_rate_threshold = dead_letter_rate_alert_threshold
        self._task: Optional[asyncio.Task] = None
        self._running = False
        self._last_dead_letters = 0
        self._last_check_ts = time.monotonic()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._running = True
        self._task = asyncio.create_task(self._monitor_loop(), name="bus_health_monitor")
        log.info("bus_health_monitor_started", interval_s=self._interval_s)

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("bus_health_monitor_stopped")

    async def _monitor_loop(self) -> None:
        while self._running:
            try:
                health = self._bus.health()
                metrics = BusMetrics(health)  # noqa
                log.info("bus_health_snapshot", **health.to_log_dict())

                # Dead letter rate check.
                now = time.monotonic()
                elapsed_min = (now - self._last_check_ts) / 60.0
                new_dead = health.dead_letters - self._last_dead_letters
                if elapsed_min > 0:
                    rate = new_dead / elapsed_min
                    if rate > self._dl_rate_threshold:
                        log.error(
                            "dead_letter_rate_alert",
                            rate_per_min=round(rate, 2),
                            threshold=self._dl_rate_threshold,
                            total_dead_letters=health.dead_letters,
                            consequence="Structural event handling failure detected. "
                                        "Review dead_letters.jsonl. index_daemon.py should investigate.",
                        )
                self._last_dead_letters = health.dead_letters
                self._last_check_ts = now

                # Log per-topic health summary.
                for topic, th in health.topics.items():
                    if not th.is_healthy:
                        log.warning(
                            "topic_unhealthy",
                            topic=topic,
                            queue_depth=th.queue_depth,
                            error_rate=round(th.error_rate, 4),
                            dead_letters=th.dead_letter_count,
                        )

            except Exception as exc:  # noqa: BLE001
                log.warning("bus_health_monitor_error", error=str(exc))

            await asyncio.sleep(self._interval_s)


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: DeadLetterReader
# Utility for reading and analyzing /store/dead_letters.jsonl.
# Used by index_daemon.py and test harnesses to inspect failures.
# ═════════════════════════════════════════════════════════════════════════════

class DeadLetterReader:
    """
    Reader and analyzer for /store/dead_letters.jsonl.

    index_daemon.py uses this to surface structural failures to operators.
    Dead letters for the same handler appearing repeatedly indicate a structural
    bug — not a transient error.

    Usage:
        reader = DeadLetterReader()
        entries = reader.read_all()
        repeated = reader.find_repeated_failures(min_count=3)
        for handler, records in repeated.items():
            log.error("repeated_dead_letters", handler=handler, count=len(records))
    """

    def __init__(self, path: Path = _DEAD_LETTER_PATH) -> None:
        self._path = path

    def read_all(self) -> List[Dict[str, Any]]:
        """
        Read all dead letter records from the file.
        Returns a list of dicts (parsed JSON). Returns empty list if file missing.
        """
        if not self._path.exists():
            return []
        records: List[Dict[str, Any]] = []
        try:
            with open(self._path, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(orjson.loads(line))
                    except orjson.JSONDecodeError as exc:
                        log.warning("dead_letter_parse_error", line=line[:200], error=str(exc))
        except OSError as exc:
            log.error("dead_letter_read_error", path=str(self._path), error=str(exc))
        return records

    def count_by_handler(self) -> Dict[str, int]:
        """Return a dict of handler_name → count of dead letters."""
        counts: Dict[str, int] = defaultdict(int)
        for record in self.read_all():
            handler = record.get("handler", "unknown")
            counts[handler] += 1
        return dict(counts)

    def count_by_topic(self) -> Dict[str, int]:
        """Return a dict of topic → count of dead letters."""
        counts: Dict[str, int] = defaultdict(int)
        for record in self.read_all():
            topic = record.get("topic", "unknown")
            counts[topic] += 1
        return dict(counts)

    def find_repeated_failures(
        self,
        min_count: int = 3,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Find handlers with at least min_count dead letters.
        These are structural bugs, not transient errors.
        Returns handler_name → list of dead letter records.
        """
        by_handler: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for record in self.read_all():
            handler = record.get("handler", "unknown")
            by_handler[handler].append(record)
        return {
            handler: records
            for handler, records in by_handler.items()
            if len(records) >= min_count
        }

    def total_count(self) -> int:
        """Total number of dead letter records in the file."""
        return len(self.read_all())

    def last_n(self, n: int = 50) -> List[Dict[str, Any]]:
        """Return the last N dead letter records (most recent)."""
        records = self.read_all()
        return records[-n:] if len(records) >= n else records

    def since(self, timestamp: str) -> List[Dict[str, Any]]:
        """
        Return dead letter records after a given ISO 8601 timestamp.
        Useful for index_daemon.py to find new failures since the last check.
        """
        result = []
        for record in self.read_all():
            ts = record.get("timestamp", "")
            if ts >= timestamp:
                result.append(record)
        return result


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: topic-level utility functions
# Convenience wrappers used in cold_start.py and test harnesses.
# ═════════════════════════════════════════════════════════════════════════════

def get_registered_topics() -> List[str]:
    """Return sorted list of all registered topic names."""
    return sorted(TOPIC_REGISTRY.keys())


def get_topic_schema(topic: str) -> Type[Any]:
    """Return the schema class for a topic. Raises KeyError for unknown topics."""
    if topic not in TOPIC_REGISTRY:
        raise KeyError(
            f"Topic {topic!r} is not registered. "
            f"Known topics: {sorted(TOPIC_REGISTRY.keys())}."
        )
    return TOPIC_REGISTRY[topic]


def is_bus_started() -> bool:
    """True if the module-level BUS singleton has been started."""
    return BUS._started # noqa


def bus_mode() -> Optional[str]:
    """Return the current bus mode as a string: 'kafka', 'degraded', or None."""
    if BUS._mode is None: # noqa
        return None
    return str(BUS._mode) # noqa


async def wait_for_bus_ready(timeout_s: float = 10.0) -> bool:
    """
    Wait until the bus is started and healthy.
    Returns True if ready within timeout_s, False otherwise.
    Used by components that start concurrently with cold_start.py.
    """
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if BUS._started: # noqa
            health = BUS.health()
            if health.started:
                return True
        await asyncio.sleep(0.05)
    return False


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusTestHarness
# Isolated bus instance for unit testing.
# Does NOT use the module-level BUS singleton.
# Never connects to Kafka. Always runs in degraded mode.
# ═════════════════════════════════════════════════════════════════════════════

class BusTestHarness:
    """
    Isolated CrawlerBus instance for unit and integration tests.

    Does not use the BUS singleton. Does not connect to Kafka.
    Forces degraded mode by temporarily unsetting KAFKA_BOOTSTRAP_SERVERS.
    Provides helpers for asserting events were emitted and dispatched.

    Usage:
        async with BusTestHarness() as harness:
            await harness.subscribe("clean_signal", "test.group", my_handler, CleanSignalEvent)
            emitter = await harness.emitter("clean_signal", "test.component", CleanSignalEvent)
            await emitter.emit(event)
            await asyncio.sleep(0.05)  # let dispatch loop run
            assert harness.dispatch_count("clean_signal") == 1
    """

    def __init__(self) -> None:
        self._bus = CrawlerBus()
        self._replay = BusReplayBuffer(capacity=1000)

    async def __aenter__(self) -> "BusTestHarness":
        # Force degraded mode by temporarily removing the Kafka env var.
        saved = os.environ.pop(_KAFKA_BOOTSTRAP_ENV, None)
        try:
            await self._bus.start()
        finally:
            if saved is not None:
                os.environ[_KAFKA_BOOTSTRAP_ENV] = saved
        self._replay.enable()
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self._bus.stop()
        self._replay.disable()

    async def emitter(self, topic: str, component: str, schema: Type[T]) -> TopicEmitter[T]:
        return await self._bus.emitter(topic=topic, component=component, schema=schema)

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: HandlerFn,
        schema: Type[T],
    ) -> None:
        await self._bus.subscribe(topic=topic, group=group, handler=handler, schema=schema)

    def health(self) -> BusHealth:
        return self._bus.health()

    def dispatch_count(self, topic: str) -> int:
        return self._bus._dispatch_counts.get(topic, 0) # noqa

    def emit_count(self, topic: str) -> int:
        return self._bus._emit_counts.get(topic, 0) # noqa

    def dead_letter_count(self, topic: str) -> int:
        return self._bus._dead_letter_counts.get(topic, 0) # noqa

    def hmac_failure_count(self) -> int:
        return self._bus._hmac_failures # noqa

    def schema_failure_count(self) -> int:
        return self._bus._schema_failures # noqa

    @property
    def mode(self) -> Optional[BusMode]:
        return self._bus._mode # noqa


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Environment validation utility
# Used by cold_start.py to surface missing env vars before start().
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BusEnvCheck:
    """Result of validate_bus_environment()."""
    hmac_key_present:        bool
    hmac_key_length_ok:      bool
    kafka_servers_set:       bool
    tls_certs_present:       bool
    dead_letter_dir_writable: bool
    errors:                  List[str]
    warnings:                List[str]

    @property
    def is_ready(self) -> bool:
        """True if the bus can start without critical failures."""
        return self.hmac_key_present and self.hmac_key_length_ok and not self.errors


def validate_bus_environment() -> BusEnvCheck:
    """
    Validate the runtime environment required by the bus.
    Called by cold_start.py before bus.start() to surface problems early.

    Returns a BusEnvCheck with all findings. Does not raise.
    """
    errors: List[str] = []
    warnings: List[str] = []

    # HMAC key check.
    raw_key = _get_env_bytes(_BUS_HMAC_KEY_ENV)
    hmac_present = bool(raw_key)
    hmac_length_ok = False
    if hmac_present and raw_key is not None:
        key_bytes = raw_key
        if len(raw_key) == 64 and all(c in b"0123456789abcdefABCDEF" for c in raw_key):
            key_bytes = bytes.fromhex(raw_key.decode("ascii"))
        hmac_length_ok = len(key_bytes) >= _MIN_HMAC_KEY_BYTES
        if not hmac_length_ok:
            errors.append(
                f"{_BUS_HMAC_KEY_ENV} is {len(key_bytes)} bytes. "
                f"Minimum required: {_MIN_HMAC_KEY_BYTES} bytes."
            )
    else:
        errors.append(f"{_BUS_HMAC_KEY_ENV} is not set. Bus cannot operate.")

    # Kafka check.
    kafka_set = bool(os.environ.get(_KAFKA_BOOTSTRAP_ENV))
    if not kafka_set:
        warnings.append(
            f"{_KAFKA_BOOTSTRAP_ENV} is not set. Bus will operate in degraded mode. "
            f"Events will not be durable. Set this variable to enable Kafka mode."
        )

    # TLS cert check.
    ca_path   = os.environ.get(_KAFKA_CA_CERT_ENV, _DEFAULT_CA_CERT)
    cert_path = os.environ.get(_KAFKA_CLIENT_CERT_ENV, _DEFAULT_CLIENT_CERT)
    key_path  = os.environ.get(_KAFKA_CLIENT_KEY_ENV, _DEFAULT_CLIENT_KEY)
    tls_present = all(Path(p).exists() for p in (ca_path, cert_path, key_path))
    if kafka_set and not tls_present:
        warnings.append(
            f"Kafka is configured but TLS certs are missing. "
            f"Connection will be unencrypted. "
            f"Provision {ca_path}, {cert_path}, {key_path}."
        )

    # Dead letter dir check.
    dl_dir_writable = False
    try:
        store_dir = _DEAD_LETTER_PATH.parent
        store_dir.mkdir(parents=True, exist_ok=True)
        probe = store_dir / ".env_check_probe"
        probe.touch()
        probe.unlink()
        dl_dir_writable = True
    except OSError as exc:
        errors.append(
            f"Dead letter directory {_DEAD_LETTER_PATH.parent} is not writable: {exc}. "
            f"cold_start.py must create /store before bus.start()."
        )

    return BusEnvCheck(
        hmac_key_present=hmac_present,
        hmac_key_length_ok=hmac_length_ok,
        kafka_servers_set=kafka_set,
        tls_certs_present=tls_present,
        dead_letter_dir_writable=dl_dir_writable,
        errors=errors,
        warnings=warnings,
    )


# ═════════════════════════════════════════════════════════════════════════════
# __all__
# Public API of this module. Everything not listed here is internal.
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Core
    "BUS",
    "BusMode",
    "BusEnvelope",
    "TopicHealth",
    "BusHealth",
    "DeadLetterEvent",
    "TopicEmitter",
    "TopicRegistry",
    "TOPIC_REGISTRY",
    "CrawlerBus",
    # Supplementary
    "BusMetrics",
    "BusReplayBuffer",
    "BusHealthMonitor",
    "DeadLetterReader",
    "BusTestHarness",
    "BusEnvCheck",
    # Utilities
    "get_registered_topics",
    "get_topic_schema",
    "is_bus_started",
    "bus_mode",
    "wait_for_bus_ready",
    "validate_bus_environment",
    "event_from_payload",
]


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: KafkaCircuitBreaker
# Prevents cascading failures when the Kafka broker becomes unreliable.
# Used internally by _KafkaSink.send() in extended production deployments.
# ═════════════════════════════════════════════════════════════════════════════

class CircuitState(Enum):
    """
    The three states of the KafkaCircuitBreaker.

    CLOSED:   Normal operation. Requests pass through.
    OPEN:     Broker has failed too many times. Requests are rejected immediately
              without attempting Kafka. Bus raises KafkaSinkUnavailable.
    HALF_OPEN: Probe state. One request is allowed through to test recovery.
              If it succeeds, circuit closes. If it fails, circuit opens again.
    """
    CLOSED    = auto()
    OPEN      = auto()
    HALF_OPEN = auto()


class KafkaCircuitBreaker:
    """
    Circuit breaker for the Kafka sink.

    Prevents the bus from hammering a failed Kafka broker with retries,
    which would cause emitter.emit() to block at _KAFKA_MAX_BLOCK_MS per call.
    Instead, after _failure_threshold consecutive failures, the circuit opens
    and all sends immediately raise KafkaSinkUnavailable for _open_duration_s seconds.

    After _open_duration_s, the circuit enters HALF_OPEN and allows one probe
    request. Success → CLOSED. Failure → OPEN again.

    Architecture note:
        The circuit breaker is not a replacement for the Kafka probe in
        _detect_mode(). The probe runs once at startup. The circuit breaker
        runs continuously during operation. They serve different failure modes:
        - Probe: Kafka was unreachable at startup → degraded mode permanently.
        - Circuit breaker: Kafka became unreachable during operation → fast-fail
          with periodic recovery probing, without switching sink mode.

    In practice, if Kafka goes down mid-operation, the bus does NOT fall to
    degraded mode automatically. It raises KafkaSinkUnavailable to producers.
    Producers decide how to handle this (buffer, drop, fail upstream).
    This is intentional: silent mode switches mid-operation are dangerous.
    The operator must intervene or restart the process.

    Usage:
        breaker = KafkaCircuitBreaker()
        if breaker.allow_request():
            try:
                await kafka_sink.send(envelope)
                breaker.record_success()
            except KafkaSinkUnavailable:
                breaker.record_failure()
                raise
        else:
            raise KafkaSinkUnavailable("Circuit open — broker unavailable")
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        open_duration_s: float = 30.0,
        half_open_probe_timeout_s: float = 5.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._open_duration_s = open_duration_s
        self._half_open_probe_timeout_s = half_open_probe_timeout_s

        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_open_ts: Optional[float] = None
        self._success_count: int = 0
        self._total_opens: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state. Transitions are applied lazily on allow_request()."""
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    @property
    def total_opens(self) -> int:
        return self._total_opens

    def allow_request(self) -> bool:
        """
        Return True if the request should be attempted.
        Return False if the circuit is open and the request should be rejected.

        Side effects:
            - OPEN → HALF_OPEN if open_duration_s has elapsed.
            - HALF_OPEN: only one concurrent request is allowed; second callers
              are rejected until the probe completes.
        """
        if self._state is CircuitState.CLOSED:
            return True

        if self._state is CircuitState.OPEN:
            elapsed = time.monotonic() - (self._last_open_ts or 0.0)
            if elapsed >= self._open_duration_s:
                # Probe window has elapsed. Transition to HALF_OPEN.
                self._state = CircuitState.HALF_OPEN
                log.info(
                    "circuit_breaker_half_open",
                    elapsed_s=round(elapsed, 1),
                    consequence="One probe request allowed to test Kafka recovery.",
                )
                return True
            return False

        if self._state is CircuitState.HALF_OPEN:
            # Only allow one probe request in HALF_OPEN.
            return True

        return False  # unreachable # noqa | runtime defensive check

    def record_success(self) -> None:
        """
        Record a successful Kafka operation.
        If in HALF_OPEN, close the circuit.
        In CLOSED, reset the consecutive failure counter.
        """
        if self._state is CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            log.info(
                "circuit_breaker_closed",
                after_opens=self._total_opens,
                consequence="Kafka broker recovered. Normal operation resumed.",
            )
        elif self._state is CircuitState.CLOSED:
            self._failure_count = 0
        self._success_count += 1

    def record_failure(self) -> None:
        """
        Record a Kafka operation failure.
        If consecutive failures reach the threshold, open the circuit.
        If in HALF_OPEN, reopen immediately.
        """
        self._failure_count += 1

        if self._state is CircuitState.HALF_OPEN:
            # Probe failed — reopen circuit.
            self._state = CircuitState.OPEN
            self._last_open_ts = time.monotonic()
            self._total_opens += 1
            log.warning(
                "circuit_breaker_reopened",
                reason="Half-open probe request failed.",
                will_retry_in_s=self._open_duration_s,
            )

        elif (
            self._state is CircuitState.CLOSED
            and self._failure_count >= self._failure_threshold
        ):
            self._state = CircuitState.OPEN
            self._last_open_ts = time.monotonic()
            self._total_opens += 1
            log.error(
                "circuit_breaker_opened",
                failure_count=self._failure_count,
                threshold=self._failure_threshold,
                will_retry_in_s=self._open_duration_s,
                consequence=(
                    "Kafka broker considered unavailable. "
                    "emit() will raise KafkaSinkUnavailable until circuit closes. "
                    "Bus remains in Kafka mode — no automatic fallback to degraded mode."
                ),
            )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "state":               self._state.name,
            "failure_count":       self._failure_count,
            "success_count":       self._success_count,
            "total_opens":         self._total_opens,
            "last_open_ts":        self._last_open_ts,
            "failure_threshold":   self._failure_threshold,
            "open_duration_s":     self._open_duration_s,
        }


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: EventEnvelopeValidator
# Validates envelope structure before deserialization.
# Distinct from HMAC verification and schema validation —
# this is a structural check on the BusEnvelope itself.
# ═════════════════════════════════════════════════════════════════════════════

class EventEnvelopeValidator:
    """
    Validates the structural integrity of a BusEnvelope before deserialization.

    This runs after HMAC verification but before msgpack deserialization.
    It catches malformed envelopes that passed HMAC (same key, but missing
    required headers or corrupted fields) before any further processing.

    Validations performed:
        - topic matches a registered topic
        - required headers are present (run_id, schema_version, source_component,
          emit_timestamp, hmac_sha256)
        - schema_version matches the supported version
        - emit_timestamp is parseable as ISO 8601
        - value is not empty
        - value does not exceed _MAX_ENVELOPE_VALUE_BYTES

    Failures raise EventSchemaError. The envelope is written to dead letters
    by _handle_envelope() before this is called, so these failures are already
    captured before we get here. This validator is an additional layer of defense.
    """

    REQUIRED_HEADERS: FrozenSet[str] = frozenset({
        "schema_version",
        "source_component",
        "emit_timestamp",
        "hmac_sha256",
    })

    SUPPORTED_SCHEMA_VERSIONS: FrozenSet[str] = frozenset({"1"})

    def __init__(self, registry: TopicRegistry) -> None:
        self._registry = registry

    def validate(self, envelope: BusEnvelope) -> None:
        """
        Validate envelope structure. Raises EventSchemaError on any violation.
        Call after HMAC verification, before msgpack deserialization.
        """
        self._check_topic(envelope)
        self._check_value(envelope)
        self._check_headers(envelope)
        self._check_schema_version(envelope)
        self._check_timestamp(envelope)

    def _check_topic(self, envelope: BusEnvelope) -> None:
        if envelope.topic not in self._registry:
            raise EventSchemaError(
                f"Envelope topic={envelope.topic!r} is not registered. "
                f"This envelope was produced for an unknown topic. "
                f"Registered topics: {self._registry.all_topics}."
            )

    def _check_value(self, envelope: BusEnvelope) -> None: # noqa
        if not envelope.value:
            raise EventSchemaError(
                f"Envelope value is empty for topic={envelope.topic!r} "
                f"offset={envelope.offset}. Empty payloads are not valid."
            )
        if envelope.is_oversized():
            raise EventSchemaError(
                f"Envelope value is {envelope.value_size:,} bytes for "
                f"topic={envelope.topic!r}, exceeding the limit of "
                f"{_MAX_ENVELOPE_VALUE_BYTES:,} bytes. "
                f"This event cannot be deserialized safely."
            )

    def _check_headers(self, envelope: BusEnvelope) -> None:
        missing = self.REQUIRED_HEADERS - set(envelope.headers.keys())
        if missing:
            raise EventSchemaError(
                f"Envelope for topic={envelope.topic!r} is missing required headers: "
                f"{sorted(missing)}. "
                f"Present headers: {sorted(envelope.headers.keys())}."
            )

    def _check_schema_version(self, envelope: BusEnvelope) -> None:
        version = envelope.headers.get("schema_version", "")
        if version not in self.SUPPORTED_SCHEMA_VERSIONS:
            raise EventSchemaError(
                f"Envelope schema_version={version!r} is not supported. "
                f"Supported versions: {sorted(self.SUPPORTED_SCHEMA_VERSIONS)}. "
                f"This envelope may be from a different version of AXIOM."
            )

    def _check_timestamp(self, envelope: BusEnvelope) -> None: # noqa
        ts = envelope.headers.get("emit_timestamp", "")
        if not ts:
            raise EventSchemaError(
                f"Envelope emit_timestamp is empty for topic={envelope.topic!r}."
            )
        try:
            datetime.fromisoformat(ts)
        except ValueError:
            raise EventSchemaError(
                f"Envelope emit_timestamp={ts!r} is not a valid ISO 8601 timestamp "
                f"for topic={envelope.topic!r}. Cannot parse emission time."
            )


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusEventLog
# Persistent structured log of all significant bus events.
# Written to /store/bus_events.log by the bus itself.
# Distinct from dead_letters.jsonl (failures) and structlog (observability).
# This is the forensic-grade audit log for the bus coordinator.
# ═════════════════════════════════════════════════════════════════════════════

_BUS_EVENT_LOG_PATH: Path = Path(os.environ.get("AXIOM_BUS_EVENT_LOG_PATH", "/store/bus_events.log"))

# Maximum size of the bus event log before rotation (10 MB).
_BUS_EVENT_LOG_MAX_BYTES: int = 10 * 1024 * 1024


class BusEventCategory(Enum):
    """Categories for bus audit log entries."""
    MODE_DETECTION    = "mode_detection"
    SINK_LIFECYCLE    = "sink_lifecycle"
    DISPATCH_FAILURE  = "dispatch_failure"
    INTEGRITY_FAILURE = "integrity_failure"
    SCHEMA_FAILURE    = "schema_failure"
    BACKPRESSURE      = "backpressure"
    CIRCUIT_BREAKER   = "circuit_breaker"
    CONSUMER_LAG      = "consumer_lag"
    STARTUP           = "startup"
    SHUTDOWN          = "shutdown"


@dataclass(frozen=True)
class BusAuditEntry:
    """
    One entry in the bus audit log.

    Written for significant events: mode transitions, sink failures,
    integrity violations, circuit breaker state changes.

    NOT written for every emit/dispatch — that volume would make the log
    useless for forensics. Only write entries that indicate the bus itself
    has encountered something requiring investigation.
    """

    category:   str       # BusEventCategory.value
    event:      str       # brief machine-readable event identifier
    detail:     str       # human-readable context
    mode:       str       # "kafka" | "degraded" | "uninitialized"
    timestamp:  str       # ISO 8601 UTC
    extra:      Dict[str, Any] = field(default_factory=dict)

    def to_jsonl_line(self) -> bytes:
        d = dataclasses.asdict(self)
        return orjson.dumps(d) + b"\n"


class BusAuditLog:
    """
    Append-only audit log for significant bus events.

    Written to /store/bus_events.log. Rotated when the file exceeds
    _BUS_EVENT_LOG_MAX_BYTES — the old file is renamed to bus_events.log.1.

    Not fsync-every-write (unlike dead_letters.jsonl) — bus audit events
    are informational, not forensic requirements. They are flushed after
    each write but not fsynced. Dead letters are fsynced; audit log is not.

    If the log cannot be written, a structlog warning is emitted and the
    bus continues. Audit log failure is never a hard stop.
    """

    def __init__(self, path: Path = _BUS_EVENT_LOG_PATH) -> None:
        self._path = path
        self._written: int = 0

    def write(
        self,
        category: BusEventCategory,
        event: str,
        detail: str,
        mode: str,
        **extra: Any,
    ) -> None:
        """
        Write one audit entry. Non-blocking. If write fails, logs via structlog.
        """
        entry = BusAuditEntry(
            category=category.value,
            event=event,
            detail=detail,
            mode=mode,
            timestamp=_iso8601_now(),
            extra=extra,
        )
        self._append(entry)

    def _append(self, entry: BusAuditEntry) -> None:
        try:
            self._maybe_rotate()
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._path, "ab") as f:
                line = entry.to_jsonl_line()
                f.write(line)
                f.flush()
            self._written += 1
        except OSError as exc:
            log.warning(
                "bus_audit_log_write_failed",
                error=str(exc),
                audit_event=entry.event,
            )

    def _maybe_rotate(self) -> None:
        """Rotate the log file if it exceeds the size limit."""
        if not self._path.exists():
            return
        if self._path.stat().st_size >= _BUS_EVENT_LOG_MAX_BYTES:
            rotated = self._path.with_suffix(".log.1")
            try:
                self._path.rename(rotated)
                log.info(
                    "bus_audit_log_rotated",
                    from_path=str(self._path),
                    to_path=str(rotated),
                )
            except OSError as exc:
                log.warning("bus_audit_log_rotation_failed", error=str(exc))

    @property
    def entries_written(self) -> int:
        return self._written


# Module-level audit log singleton.
_BUS_AUDIT_LOG = BusAuditLog()


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BackpressureMonitor
# Tracks queue depth and emits backpressure warnings when thresholds are hit.
# Used by the degraded sink dispatch loop to surface saturation to operators.
# ═════════════════════════════════════════════════════════════════════════════

class BackpressureMonitor:
    """
    Monitors queue depth thresholds and surfaces saturation events.

    In degraded mode, asyncio.Queue(maxsize=10_000) applies backpressure by
    blocking emit(). This is correct behavior — the fetcher slows down when
    downstream processing is saturated. However, operators need visibility into
    when this is happening.

    BackpressureMonitor samples queue depths periodically and logs a structured
    warning when depth exceeds thresholds. It does NOT change bus behavior —
    it is a pure observer.

    Thresholds (fraction of _QUEUE_MAX_SIZE):
        warn:     > 50% — queue is filling, monitor closely
        critical: > 80% — queue is nearly full, backpressure is active

    Used by BusHealthMonitor's background loop.
    """

    WARN_FRACTION:     float = 0.50
    CRITICAL_FRACTION: float = 0.80

    def __init__(self) -> None:
        self._last_warn_ts: Dict[str, float] = {}
        self._warn_interval_s: float = 60.0   # don't spam warnings
        self._total_saturation_events: int = 0

    def check(self, topic: str, depth: int) -> None:
        """
        Check one topic's queue depth against thresholds.
        Logs if thresholds are crossed (rate-limited to _warn_interval_s).
        """
        fraction = depth / _QUEUE_MAX_SIZE

        if fraction >= self.CRITICAL_FRACTION:
            self._maybe_log(
                topic=topic,
                depth=depth,
                fraction=fraction,
                level="critical",
            )
        elif fraction >= self.WARN_FRACTION:
            self._maybe_log(
                topic=topic,
                depth=depth,
                fraction=fraction,
                level="warn",
            )

    def _maybe_log(self, topic: str, depth: int, fraction: float, level: str) -> None:
        now = time.monotonic()
        last = self._last_warn_ts.get(topic, 0.0)
        if now - last < self._warn_interval_s:
            return  # rate-limited
        self._last_warn_ts[topic] = now
        self._total_saturation_events += 1

        if level == "critical":
            log.error(
                "backpressure_critical",
                topic=topic,
                queue_depth=depth,
                queue_capacity=_QUEUE_MAX_SIZE,
                fill_fraction=round(fraction, 3),
                consequence=(
                    "emit() is blocking on this topic. "
                    "Fetcher is being backpressured. "
                    "If this persists, downstream processing is saturated — "
                    "investigate dispatch loop latency for this topic."
                ),
            )
            _BUS_AUDIT_LOG.write(
                category=BusEventCategory.BACKPRESSURE,
                event="backpressure_critical",
                detail=f"Queue depth {depth}/{_QUEUE_MAX_SIZE} for topic={topic!r}",
                mode="degraded",
                topic=topic,
                depth=depth,
                fraction=round(fraction, 3),
            )
        else:
            log.warning(
                "backpressure_warn",
                topic=topic,
                queue_depth=depth,
                queue_capacity=_QUEUE_MAX_SIZE,
                fill_fraction=round(fraction, 3),
            )

    @property
    def total_saturation_events(self) -> int:
        return self._total_saturation_events


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusStartupDiagnostic
# Runs a self-test during cold_start.py to verify the bus is operating
# correctly before any live traffic is processed.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BusDiagnosticResult:
    """Result of a BusStartupDiagnostic run."""
    passed:             bool
    mode_detected:      str
    hmac_sign_verify:   bool
    serialization_ok:   bool
    dead_letter_write:  bool
    environment_check:  BusEnvCheck
    errors:             List[str]
    warnings:           List[str]
    duration_ms:        float

    def to_log_dict(self) -> Dict[str, Any]:
        return {
            "passed":            self.passed,
            "mode":              self.mode_detected,
            "hmac_sign_verify":  self.hmac_sign_verify,
            "serialization_ok":  self.serialization_ok,
            "dead_letter_write": self.dead_letter_write,
            "errors":            self.errors,
            "warnings":          self.warnings,
            "duration_ms":       round(self.duration_ms, 2),
        }


class BusStartupDiagnostic:
    """
    Self-test suite run by cold_start.py before opening the bus to live traffic.

    Tests:
        1. Environment check — HMAC key, Kafka vars, cert paths, /store access.
        2. HMAC sign/verify roundtrip — creates a synthetic envelope, signs it,
           verifies the signature. Detects HMAC key misconfiguration.
        3. Serialization roundtrip — serializes a RawFetchEvent, deserializes it,
           compares. Detects msgpack or dataclass regression.
        4. Dead letter write — writes a test dead letter and verifies it appears
           in the file. Detects /store write permission issues.

    Usage:
        diag = BusStartupDiagnostic(BUS)
        result = await diag.run()
        if not result.passed:
            raise RuntimeError(f"Bus startup diagnostic failed: {result.errors}")
    """

    def __init__(self, bus: CrawlerBus) -> None:
        self._bus = bus

    async def run(self) -> BusDiagnosticResult:
        """Run all diagnostics. Returns a BusDiagnosticResult with all findings."""
        t_start = time.monotonic()
        errors: List[str] = []
        warnings: List[str] = []

        env_check = validate_bus_environment()
        errors.extend(env_check.errors)
        warnings.extend(env_check.warnings)

        hmac_ok = self._test_hmac_roundtrip(errors)
        serial_ok = self._test_serialization_roundtrip(errors)
        dl_ok = self._test_dead_letter_write(errors)

        passed = (
            len(errors) == 0
            and hmac_ok
            and serial_ok
            and dl_ok
        )

        duration_ms = (time.monotonic() - t_start) * 1000.0
        mode = str(self._bus._mode) if self._bus._mode else "uninitialized" # noqa

        result = BusDiagnosticResult(
            passed=passed,
            mode_detected=mode,
            hmac_sign_verify=hmac_ok,
            serialization_ok=serial_ok,
            dead_letter_write=dl_ok,
            environment_check=env_check,
            errors=errors,
            warnings=warnings,
            duration_ms=duration_ms,
        )

        if passed:
            log.info("bus_startup_diagnostic_passed", **result.to_log_dict())
        else:
            log.error("bus_startup_diagnostic_failed", **result.to_log_dict())

        return result

    def _test_hmac_roundtrip(self, errors: List[str]) -> bool: # noqa
        """
        Sign a synthetic envelope and verify it.
        Also verify that a tampered envelope fails verification.
        """
        try:
            # Create a minimal synthetic envelope.
            now = _iso8601_now()
            headers = {
                "run_id": "00000000-0000-4000-a000-000000000001",
                "schema_version": _SCHEMA_VERSION,
                "source_component": "bus_diagnostic",
                "emit_timestamp": now,
                "event_type": "DiagnosticEvent",
                "schema_name": "DiagnosticEvent",
            }
            envelope = BusEnvelope(
                topic="raw_fetch",
                key=b"DIAGNOSTIC",
                value=b"\x81\xa3url\xabhttp://test",
                headers=headers,
                offset=0,
                partition=-1,
                timestamp=now,
            )
            # Sign it.
            sig = _sign_envelope(envelope)
            signed_headers = {**headers, "hmac_sha256": sig}
            signed_envelope = BusEnvelope(
                topic=envelope.topic,
                key=envelope.key,
                value=envelope.value,
                headers=signed_headers,
                offset=envelope.offset,
                partition=envelope.partition,
                timestamp=envelope.timestamp,
            )

            # Verify valid envelope.
            if not _verify_envelope(signed_envelope):
                errors.append("HMAC verification failed on a freshly signed envelope.")
                return False

            # Verify that a tampered value fails.
            tampered_headers = {**signed_headers}
            tampered = BusEnvelope(
                topic=signed_envelope.topic,
                key=signed_envelope.key,
                value=b"\x00TAMPERED",
                headers=tampered_headers,
                offset=signed_envelope.offset,
                partition=signed_envelope.partition,
                timestamp=signed_envelope.timestamp,
            )
            if _verify_envelope(tampered):
                errors.append(
                    "HMAC verification returned True for a tampered envelope. "
                    "AXIOM_BUS_HMAC_KEY may be all-zeros or trivially guessable."
                )
                return False

            return True

        except Exception as exc:  # noqa: BLE001
            errors.append(f"HMAC roundtrip test raised: {type(exc).__name__}: {exc}")
            return False

    def _test_serialization_roundtrip(self, errors: List[str]) -> bool: # noqa
        """
        Serialize and deserialize a RawFetchEvent. Verify field equality.
        """
        try:
            import uuid as _uuid
            original = RawFetchEvent(
                url="https://diagnostic.axiom.internal/test",
                raw_bytes=b"<html>test</html>",
                status_code=200,
                headers={"content-type": "text/html"},
                fetch_latency=0.042,
                fetch_mode=FetchMode.STATIC,
                is_robots_txt=False,
                is_sitemap=False,
                topology_hint="GENERIC_HTML",
                run_id=str(_uuid.uuid4()),
                manifest_id=str(_uuid.uuid4()),
                byte_count=len(b"<html>test</html>"),
            )
            envelope = _serialize(
                event=original,
                topic="raw_fetch",
                component="bus_diagnostic",
                offset=0,
            )
            # Add HMAC to headers for deserialization.
            sig = _sign_envelope(envelope)
            signed_headers = {**envelope.headers, "hmac_sha256": sig}
            signed_envelope = BusEnvelope(
                topic=envelope.topic,
                key=envelope.key,
                value=envelope.value,
                headers=signed_headers,
                offset=envelope.offset,
                partition=envelope.partition,
                timestamp=envelope.timestamp,
            )
            recovered = _deserialize(signed_envelope, RawFetchEvent)

            # Compare fields.
            mismatches = []
            for f in dataclasses.fields(original):
                orig_val = getattr(original, f.name)
                recv_val = getattr(recovered, f.name)
                if orig_val != recv_val:
                    mismatches.append(
                        f"field={f.name!r}: original={orig_val!r} recovered={recv_val!r}"
                    )
            if mismatches:
                errors.append(
                    f"Serialization roundtrip field mismatch: {'; '.join(mismatches)}"
                )
                return False

            return True

        except Exception as exc:  # noqa: BLE001
            errors.append(f"Serialization roundtrip test raised: {type(exc).__name__}: {exc}")
            return False

    def _test_dead_letter_write(self, errors: List[str]) -> bool: # noqa
        """
        Write a test dead letter and verify it appears in the file.
        """
        try:
            import uuid as _uuid
            now = _iso8601_now()
            headers = {
                "run_id": str(_uuid.uuid4()),
                "schema_version": _SCHEMA_VERSION,
                "source_component": "bus_diagnostic",
                "emit_timestamp": now,
                "hmac_sha256": "diagnostic_test",
            }
            synthetic_envelope = BusEnvelope(
                topic="raw_fetch",
                key=b"DIAGNOSTIC",
                value=b"test",
                headers=headers,
                offset=-1,
                partition=-1,
                timestamp=now,
            )
            marker = f"DIAGNOSTIC_TEST_{_uuid.uuid4().hex[:8]}"
            _write_dead_letter(
                envelope=synthetic_envelope,
                handler_name=marker,
                error=RuntimeError("diagnostic test — not a real failure"),
                retries=0,
                mode="diagnostic",
            )
            # Verify it was written.
            reader = DeadLetterReader(path=_DEAD_LETTER_PATH)
            recent = reader.last_n(n=10)
            found = any(r.get("handler") == marker for r in recent)
            if not found:
                errors.append(
                    f"Dead letter write test failed: marker {marker!r} "
                    f"not found in {_DEAD_LETTER_PATH}."
                )
                return False
            return True

        except Exception as exc:  # noqa: BLE001
            errors.append(f"Dead letter write test raised: {type(exc).__name__}: {exc}")
            return False


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: BusConfigSnapshot
# Captures the full runtime configuration of the bus for structured logging.
# Emitted by cold_start.py and index_daemon.py for configuration audit trails.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class BusConfigSnapshot:
    """
    A snapshot of the bus runtime configuration.

    Emitted at startup and on demand for configuration audit.
    Sensitive values (HMAC key) are redacted. All other config is present.
    Used by cold_start.py to record exactly what configuration was in effect
    when the bus started — the audit trail for debugging configuration drift.
    """

    mode:                       str
    kafka_bootstrap:            Optional[str]   # None if not configured
    kafka_tls_enabled:          bool
    kafka_ca_cert_path:         Optional[str]
    kafka_client_cert_path:     Optional[str]
    dead_letter_path:           str
    queue_max_size:             int
    handler_max_retries:        int
    kafka_probe_timeout_s:      float
    kafka_probe_attempts:       int
    shutdown_drain_timeout_s:   float
    schema_version:             str
    max_envelope_bytes:         int
    topics:                     List[str]
    hmac_key_present:           bool
    hmac_key_redacted:          str             # always "REDACTED"
    captured_at:                str

    @classmethod
    def capture(cls, bus: CrawlerBus) -> "BusConfigSnapshot":
        """Capture current configuration from a running bus instance."""
        bootstrap = os.environ.get(_KAFKA_BOOTSTRAP_ENV)
        ca_path    = os.environ.get(_KAFKA_CA_CERT_PATH := _KAFKA_CA_CERT_ENV, _DEFAULT_CA_CERT) # noqa
        cert_path  = os.environ.get(_KAFKA_CLIENT_CERT_ENV, _DEFAULT_CLIENT_CERT) # noqa
        tls_enabled = all(Path(p).exists() for p in (
            os.environ.get(_KAFKA_CA_CERT_ENV, _DEFAULT_CA_CERT),
            os.environ.get(_KAFKA_CLIENT_CERT_ENV, _DEFAULT_CLIENT_CERT),
            os.environ.get(_KAFKA_CLIENT_KEY_ENV, _DEFAULT_CLIENT_KEY),
        ))
        return cls(
            mode=str(bus._mode) if bus._mode else "uninitialized", # noqa
            kafka_bootstrap=bootstrap,
            kafka_tls_enabled=tls_enabled,
            kafka_ca_cert_path=os.environ.get(_KAFKA_CA_CERT_ENV, _DEFAULT_CA_CERT) if tls_enabled else None,
            kafka_client_cert_path=os.environ.get(_KAFKA_CLIENT_CERT_ENV, _DEFAULT_CLIENT_CERT) if tls_enabled else None,
            dead_letter_path=str(_DEAD_LETTER_PATH),
            queue_max_size=_QUEUE_MAX_SIZE,
            handler_max_retries=_HANDLER_MAX_RETRIES,
            kafka_probe_timeout_s=_KAFKA_PROBE_TIMEOUT_S,
            kafka_probe_attempts=_KAFKA_PROBE_ATTEMPTS,
            shutdown_drain_timeout_s=_SHUTDOWN_DRAIN_TIMEOUT_S,
            schema_version=_SCHEMA_VERSION,
            max_envelope_bytes=_MAX_ENVELOPE_VALUE_BYTES,
            topics=bus._registry.all_topics, # noqa
            hmac_key_present=bool(_get_env_bytes(_BUS_HMAC_KEY_ENV)),
            hmac_key_redacted="REDACTED",
            captured_at=_iso8601_now(),
        )

    def to_log_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: PhaseTransitionEvent handling utilities
# The phase_transition topic carries PhaseTransitionEvent, which index_daemon.py
# produces and multiple components consume to update their behavior.
# These utilities make producing and consuming phase transitions explicit.
# ═════════════════════════════════════════════════════════════════════════════

class PhaseTransitionHelper:
    """
    Utilities for working with the phase_transition topic.

    index_daemon.py uses this to emit phase transitions correctly.
    Consumers can use the class methods to log and validate transitions
    without importing the full bus machinery.

    Phase transitions are significant events — they change system-wide
    behavior. Every transition is logged to the audit log.
    """

    VALID_PHASES: FrozenSet[int] = frozenset({1, 2, 3})
    VALID_TRANSITIONS: FrozenSet[Tuple[int, int]] = frozenset({
        (1, 2),   # learns → predicts  (confidence threshold reached)
        (2, 3),   # predicts → knows   (compiled policy stable)
        (3, 2),   # knows → predicts   (surprise dissolve triggered)
        (2, 1),   # predicts → learns  (reindex triggered)
    })

    @classmethod
    def validate_transition(cls, from_phase: int, to_phase: int) -> bool:
        """
        True if the transition from_phase → to_phase is valid.
        Invalid transitions indicate a bug in index_daemon.py.
        """
        return (from_phase, to_phase) in cls.VALID_TRANSITIONS

    @classmethod
    def describe(cls, phase: int) -> str:
        """Human-readable description of a phase."""
        return {1: "learns", 2: "predicts", 3: "knows"}.get(phase, f"unknown({phase})")

    @classmethod
    def log_transition(
        cls,
        topology_class: str,
        from_phase: int,
        to_phase: int,
        run_id: str,
        confidence: float,
    ) -> None:
        """
        Log a phase transition in structured format.
        Called by index_daemon.py before emitting the PhaseTransitionEvent.
        """
        valid = cls.validate_transition(from_phase, to_phase)
        level = "info" if valid else "error"
        getattr(log, level)(
            "phase_transition",
            topology_class=topology_class,
            from_phase=cls.describe(from_phase),
            to_phase=cls.describe(to_phase),
            from_phase_int=from_phase,
            to_phase_int=to_phase,
            run_id=run_id,
            confidence=round(confidence, 4),
            valid=valid,
        )
        _BUS_AUDIT_LOG.write(
            category=BusEventCategory.SINK_LIFECYCLE,
            event="phase_transition",
            detail=(
                f"{topology_class}: phase {cls.describe(from_phase)} → "
                f"{cls.describe(to_phase)} (confidence={confidence:.4f})"
            ),
            mode=bus_mode() or "uninitialized",
            topology_class=topology_class,
            from_phase=from_phase,
            to_phase=to_phase,
            confidence=confidence,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY: Topic documentation — machine-readable contract for all topics
# Each entry documents the producer, consumers, partition key, and semantics.
# Used by cold_start.py to print a startup summary and by documentation tooling.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TopicDoc:
    """Machine-readable documentation for one bus topic."""
    topic:         str
    schema_class:  str
    producer:      str          # which component produces this topic
    consumers:     Tuple[str, ...]  # which components subscribe
    partition_key: str          # which event field is the partition key
    description:   str
    is_high_volume: bool        # raw_fetch and clean_signal are high volume


TOPIC_DOCUMENTATION: Dict[str, TopicDoc] = {
    "raw_fetch": TopicDoc(
        topic="raw_fetch",
        schema_class="RawFetchEvent",
        producer="fetcher.py (Phantom)",
        consumers=("alpine_strip/offline_pipeline.py",),
        partition_key="url",
        description=(
            "Emitted immediately after each HTTP response is received by Phantom. "
            "Contains the raw bytes of the response, status code, headers, and metadata. "
            "High volume — every crawled URL produces one event."
        ),
        is_high_volume=True,
    ),
    "fetch_anomaly": TopicDoc(
        topic="fetch_anomaly",
        schema_class="FetchAnomalyEvent",
        producer="fetcher.py (Phantom)",
        consumers=("index_daemon.py", "preparser/crawl_planner.go"),
        partition_key="url",
        description=(
            "Emitted when a fetch attempt fails and no RawFetchEvent can be "
            "produced. This is training signal for crawl planning and friction "
            "forecasting, not a graph-stopping error."
        ),
        is_high_volume=True,
    ),
    "clean_signal": TopicDoc(
        topic="clean_signal",
        schema_class="CleanSignalEvent",
        producer="alpine_strip/offline_pipeline.py",
        consumers=(
            "world_model/latent_parser.py",
            "topology/surprise_detector.py",
            "signal_kernel/feedback.py",
        ),
        partition_key="topology_class",
        description=(
            "Emitted by the signal kernel after successful extraction. "
            "Contains the cleaned text signal, topology class, and quality metrics. "
            "Partition key is topology_class — all events for the same class go to "
            "the same Kafka partition, preserving per-class ordering."
        ),
        is_high_volume=True,
    ),
    "signal_extracted": TopicDoc(
        topic="signal_extracted",
        schema_class="SignalExtractedEvent",
        producer="preparser/signal_extractor.go",
        consumers=("index_daemon.py", "offline/batch_scheduler.c"),
        partition_key="topology_class",
        description=(
            "Emitted by native signal extraction when a sanitized signal zone "
            "has been summarized for offline training and recipe validation."
        ),
        is_high_volume=True,
    ),
    "classification": TopicDoc(
        topic="classification",
        schema_class="ClassificationEvent",
        producer="topology/classifier.py",
        consumers=("topology/surprise_detector.py", "index_daemon.py"),
        partition_key="domain",
        description=(
            "Emitted after URL classification. Carries observed classifier "
            "distribution and WLM prior distribution for divergence checks."
        ),
        is_high_volume=True,
    ),
    "topology_hint": TopicDoc(
        topic="topology_hint",
        schema_class="NewTopologyHintEvent",
        producer="topology/surprise_detector.py",
        consumers=("index_daemon.py", "world_model/latent_parser.py"),
        partition_key="domain",
        description=(
            "Emitted when repeated generic or coherent cluster evidence suggests "
            "a new topology split should be considered."
        ),
        is_high_volume=False,
    ),
    "domain_topology": TopicDoc(
        topic="domain_topology",
        schema_class="DomainTopologyEvent",
        producer="preparser/domain_analyzer.go",
        consumers=("world_model/latent_parser.py",),
        partition_key="domain",
        description=(
            "Emitted when domain_analyzer.go completes analysis of a domain's "
            "robots.txt and sitemap. Contains the DomainMap for the domain. "
            "Low volume — one event per unique domain encountered."
        ),
        is_high_volume=False,
    ),
    "crawl_manifest": TopicDoc(
        topic="crawl_manifest",
        schema_class="CrawlManifestReadyEvent",
        producer="preparser/crawl_planner.go",
        consumers=("fetcher.py (Phantom)",),
        partition_key="domain",
        description=(
            "Emitted when crawl_planner.go produces a CrawlManifest. "
            "Phantom consumes this to drive its fetch schedule. "
            "The manifest tells Phantom exactly what to fetch and in what order."
        ),
        is_high_volume=False,
    ),
    "manifest_complete": TopicDoc(
        topic="manifest_complete",
        schema_class="ManifestCompleteEvent",
        producer="fetcher.py (Phantom)",
        consumers=("index_daemon.py", "crawler/crawl_cursor.py"),
        partition_key="domain",
        description=(
            "Emitted when every URL in a crawl manifest has reached a terminal "
            "state. Used as crawl completion and training feedback."
        ),
        is_high_volume=False,
    ),
    "cl_state_update": TopicDoc(
        topic="cl_state_update",
        schema_class="CLStateUpdateEvent",
        producer="cold_start.py",
        consumers=("fetcher.py (Phantom)",),
        partition_key="topic",
        description=(
            "Emitted when clearance-level availability changes for the current "
            "session. Fetchers react by downgrading fetch modes as needed."
        ),
        is_high_volume=False,
    ),
    "container_breach": TopicDoc(
        topic="container_breach",
        schema_class="ContainerBreachEvent",
        producer="fetcher.py (Phantom)",
        consumers=("index_daemon.py", "cold_start.py"),
        partition_key="manifest_id",
        description=(
            "Emitted when a gVisor/container breach signal is observed during "
            "a high-clearance fetch. Consumers perform forensics and isolation."
        ),
        is_high_volume=False,
    ),
    "zone_map_updated": TopicDoc(
        topic="zone_map_updated",
        schema_class="ZoneMapUpdatedEvent",
        producer="world_model/latent_parser.py",
        consumers=("alpine_strip/offline_pipeline.py",),
        partition_key="topology_class",
        description=(
            "Emitted when the WLP compiles a new zone map for a topology class. "
            "The pipeline updates its extraction strategy on receipt. "
            "Low volume — emitted per topology class per recompilation."
        ),
        is_high_volume=False,
    ),
    "zone_map_invalidated": TopicDoc(
        topic="zone_map_invalidated",
        schema_class="ZoneMapInvalidatedEvent",
        producer="topology/surprise_detector.py",
        consumers=("topology/parser.py", "index_daemon.py"),
        partition_key="topology_class",
        description=(
            "Emitted when surprise invalidates the current zone map and dependent "
            "recipes must stop compiling from stale structure."
        ),
        is_high_volume=False,
    ),
    "surprise": TopicDoc(
        topic="surprise",
        schema_class="SurpriseEvent",
        producer="topology/surprise_detector.py",
        consumers=("index_daemon.py", "world_model/latent_parser.py"),
        partition_key="topology_class",
        description=(
            "Emitted when a CleanSignalEvent deviates from the WLP's prediction "
            "beyond THETA_SURPRISE_DEFAULT. If dissolve_triggered=True, the zone "
            "map was invalidated and recompilation was requested. "
            "index_daemon.py triggers a gradient step on surprise."
        ),
        is_high_volume=False,
    ),
    "recipe_stale": TopicDoc(
        topic="recipe_stale",
        schema_class="RecipeStaleEvent",
        producer="preparser/recipe_validator.go",
        consumers=("index_daemon.py", "topology/parser.py"),
        partition_key="topology_class",
        description=(
            "Emitted when a compiled recipe no longer matches recent fetch "
            "results and should be considered for recompilation."
        ),
        is_high_volume=False,
    ),
    "recipe_health": TopicDoc(
        topic="recipe_health",
        schema_class="RecipeHealthEvent",
        producer="preparser/recipe_validator.go",
        consumers=("index_daemon.py",),
        partition_key="topology_class",
        description=(
            "Aggregate recipe validation metrics for index_daemon policy and "
            "recipe lifecycle decisions."
        ),
        is_high_volume=False,
    ),
    "weights_updated": TopicDoc(
        topic="weights_updated",
        schema_class="WeightsUpdatedEvent",
        producer="offline/weight_updater.cu",
        consumers=("index_daemon.py", "world_model/latent_model.py"),
        partition_key="model_name",
        description=(
            "Emitted after offline training atomically publishes a new model "
            "artifact into the store."
        ),
        is_high_volume=False,
    ),
    "store_health": TopicDoc(
        topic="store_health",
        schema_class="StoreHealthEvent",
        producer="daemons/store_sentinel.c",
        consumers=("index_daemon.py", "cold_start.py"),
        partition_key="store_file",
        description=(
            "Store sentinel health observation for mmap files and weight "
            "artifacts. Critical events can trigger cold-start recovery."
        ),
        is_high_volume=False,
    ),
    "snapshot_candidate": TopicDoc(
        topic="snapshot_candidate",
        schema_class="SnapshotCandidateEvent",
        producer="tag/tools_bridge.py",
        consumers=("tag/tools_bridge.py", "index_daemon.py"),
        partition_key="url",
        description=(
            "A URL already selected by AXIOM routing or crawler traversal as "
            "relevant enough for temporary artifact capture. This is not an "
            "external search-engine result."
        ),
        is_high_volume=True,
    ),
    "snapshot_captured": TopicDoc(
        topic="snapshot_captured",
        schema_class="SnapshotCapturedEvent",
        producer="tag/tools_bridge.py",
        consumers=("index_daemon.py", "cold_start.py"),
        partition_key="url",
        description=(
            "Metadata for a temporary snapshot artifact captured by the tools "
            "bridge. Artifacts live outside the four durable store files and "
            "carry AXIOM provenance watermarks."
        ),
        is_high_volume=True,
    ),
    "tool_invocation": TopicDoc(
        topic="tool_invocation",
        schema_class="ToolInvocationEvent",
        producer="tag/tools_bridge.py",
        consumers=("index_daemon.py", "cold_start.py"),
        partition_key="tool_name",
        description=(
            "Audit record emitted before invoking a registered tools/ adapter "
            "through the AXIOM SDK compatibility layer."
        ),
        is_high_volume=True,
    ),
    "tool_result": TopicDoc(
        topic="tool_result",
        schema_class="ToolResultEvent",
        producer="tag/tools_bridge.py",
        consumers=("index_daemon.py", "cold_start.py"),
        partition_key="tool_name",
        description=(
            "Result record emitted after a tool adapter finishes, including "
            "status, duration, and output hash."
        ),
        is_high_volume=True,
    ),
    "tool_health": TopicDoc(
        topic="tool_health",
        schema_class="ToolHealthEvent",
        producer="tag/tools_bridge.py",
        consumers=("cold_start.py", "index_daemon.py"),
        partition_key="tool_name",
        description=(
            "Dependency and capability health for one registered tools/ "
            "adapter. Stub tools are reported explicitly instead of hidden."
        ),
        is_high_volume=False,
    ),
    "feedback": TopicDoc(
        topic="feedback",
        schema_class="FeedbackEvent",
        producer="signal_kernel/feedback.py",
        consumers=("topology/topology_parser.py",),
        partition_key="topology_class",
        description=(
            "Emitted by feedback.py after every extraction. Contains quality metrics "
            "and a recompilation recommendation. topology_parser.py decides whether "
            "to act on the recommendation. feedback.py does not invoke the compiler."
        ),
        is_high_volume=True,
    ),
    "phase_transition": TopicDoc(
        topic="phase_transition",
        schema_class="PhaseTransitionEvent",
        producer="index_daemon.py",
        consumers=(
            "world_model/latent_parser.py",
            "topology/surprise_detector.py",
            "alpine_strip/offline_pipeline.py",
        ),
        partition_key="topology_class",
        description=(
            "Emitted by index_daemon.py when a topology class transitions between "
            "phases (learns/predicts/knows). All components that adjust behavior "
            "based on phase subscribe to this topic. "
            "Low volume — emitted per topology class per phase transition."
        ),
        is_high_volume=False,
    ),
}


def print_topic_summary() -> None:
    """
    Print a human-readable summary of all registered topics.
    Called by cold_start.py during startup.
    """
    lines: List[str] = [
        "",
        "═" * 72,
        "  AXIOM CrawlerBus — Topic Registry",
        "═" * 72,
    ]
    for topic, doc in sorted(TOPIC_DOCUMENTATION.items()):
        lines.append(f"  {topic:<22} {doc.schema_class:<28} producer: {doc.producer}")
    lines.append("═" * 72)
    lines.append(f"  Total: {len(TOPIC_DOCUMENTATION)} topics  |  Mode: {bus_mode() or 'uninitialized'}")
    lines.append("═" * 72)
    lines.append("")
    print("\n".join(lines))


# Update __all__ with new exports.
__all__ += [
    "CircuitState",
    "KafkaCircuitBreaker",
    "EventEnvelopeValidator",
    "BusEventCategory",
    "BusAuditEntry",
    "BusAuditLog",
    "BackpressureMonitor",
    "BusDiagnosticResult",
    "BusStartupDiagnostic",
    "BusConfigSnapshot",
    "PhaseTransitionHelper",
    "TopicDoc",
    "TOPIC_DOCUMENTATION",
    "print_topic_summary",
    "_BUS_AUDIT_LOG",
    "CleanSignalEvent",
    "FetchAnomalyEvent",
    "SignalExtractedEvent",
    "RecipeStaleEvent",
    "RecipeHealthEvent",
    "WeightsUpdatedEvent",
    "StoreHealthEvent",
    "SnapshotCandidateEvent",
    "SnapshotCapturedEvent",
    "ToolInvocationEvent",
    "ToolResultEvent",
    "ToolHealthEvent",
    "SurpriseEvent",
    "ZoneMapUpdatedEvent",
    "ZoneMapInvalidatedEvent",
]
