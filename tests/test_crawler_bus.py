"""
test_crawler_bus.py
===================
Smoke test suite for tag/crawler_bus.py.

130+ tests covering every public class, function, and behaviour path:

    Unit tests       — pure logic, no async, no bus lifecycle
    Serialization    — _serialize / _deserialize / _event_to_dict roundtrips
    HMAC             — sign, verify, tamper detection
    DegradedSink     — queue mechanics and backpressure
    CircuitBreaker   — state machine transitions
    EnvelopeValidator— structural checks
    DeadLetterReader — JSONL read/filter utilities
    Integration      — full emit → dispatch via BusTestHarness

Setup
-----
The suite stubs ``signal_kernel.contracts`` and ``signal_kernel.exceptions``
via sys.modules injection in conftest-style fixtures so it runs without the
full AXIOM package tree.

Requirements (all in requirements.txt):
    pytest>=7
    pytest-asyncio>=0.23
    msgpack
    orjson
    structlog
    tenacity
"""

from __future__ import annotations

import asyncio
import dataclasses # noqa
import enum
import os
import sys
import types
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest

# ─────────────────────────────────────────────────────────────────────────────
# 0. BOOTSTRAP: stub signal_kernel before crawler_bus is imported
#
#    crawler_bus imports from signal_kernel.contracts and signal_kernel.exceptions
#    at module level. We inject minimal stub modules so the file can be
#    imported without the full AXIOM package tree.
#
#    The stubs mirror exactly the names imported in crawler_bus.py:
#        from signal_kernel.contracts import (
#            CleanSignalEvent, CrawlManifestReadyEvent, DomainTopologyEvent,
#            PhaseTransitionEvent, RawFetchEvent, SurpriseEvent,
#            ZoneMapUpdatedEvent,
#        )
#        from signal_kernel.contracts import FeedbackEvent as FeedbackBusEvent
#        from signal_kernel.exceptions import (
#            EventBusSubscriptionError, EventDispatchFailed, EventIntegrityError,
#            EventSchemaError, KafkaSinkUnavailable,
#        )
# ─────────────────────────────────────────────────────────────────────────────

# ── Minimal event dataclasses ────────────────────────────────────────────────

class FetchMode(str, enum.Enum):
    STATIC = "static"
    HEADLESS = "headless"
    TOR = "tor"
    TOR_FULL = "tor_full"


@dataclass(frozen=True)
class RawFetchEvent:
    url:           str
    raw_bytes:     bytes
    status_code:   int
    headers:       Dict[str, str]
    fetch_latency: float
    is_robots_txt: bool
    is_sitemap:    bool
    run_id:        str
    fetch_mode:    FetchMode = FetchMode.STATIC
    topology_hint: str = "GENERIC_HTML"
    manifest_id:   str = ""
    byte_count:    int = 0


@dataclass(frozen=True)
class CleanSignalEvent:
    url:              str
    clean_signal:     str
    topology_class:   str
    token_reduction:  float
    signal_density:   float
    extraction_empty: bool
    run_id:           str


@dataclass(frozen=True)
class DomainTopologyEvent:
    domain:     str
    domain_map: Dict[str, Any]


@dataclass(frozen=True)
class ClassificationEvent:
    url: str
    domain: str
    observed_class: str
    observed_confidence: float
    classifier_distribution: Any
    wlm_predicted_class: str
    wlm_predicted_confidence: float
    wlm_prior_distribution: Any
    run_id: str
    manifest_id: str


@dataclass(frozen=True)
class NewTopologyHintEvent:
    domain: str
    trigger: str
    evidence_count: int
    centroid_vector: List[float]
    cluster_variance: Optional[float]
    suggested_parent_class: str
    mdl_supports_split: bool
    betti0_modes: int
    oja_pc1_variance_ratio: float
    phase_at_trigger: int
    run_id: str


@dataclass(frozen=True)
class ZoneMapUpdatedEvent:
    topology_class: str
    new_zone_map:   Dict[str, Any]


@dataclass(frozen=True)
class ZoneMapInvalidatedEvent:
    topology_class: str


@dataclass(frozen=True)
class CrawlManifestReadyEvent:
    domain:   str
    manifest: Dict[str, Any]


@dataclass(frozen=True)
class ManifestCompleteEvent:
    domain: str
    manifest_id: str
    stats: Dict[str, Any]


@dataclass(frozen=True)
class CLStateUpdateEvent:
    cl2_available: bool
    cl3_available: bool
    cl4_available: bool
    reason: str


@dataclass(frozen=True)
class ContainerBreachEvent:
    manifest_id: str
    run_id: str
    fetch_mode: FetchMode
    breach_signal: str
    url: str
    detected_at: datetime


@dataclass(frozen=True)
class SurpriseEvent:
    topology_class:       str
    surprise_score:       float
    theta_surprise:       float
    dissolve_triggered:   bool
    contributing_signals: Dict[str, float]
    current_phase:        int
    run_id:               str
    timestamp:            str


@dataclass(frozen=True)
class FeedbackEvent:
    topology_class: str
    signal_hash:    str
    feedback_type:  str
    run_id:         str


@dataclass(frozen=True)
class PhaseTransitionEvent:
    topology_class: str
    from_phase:     int
    to_phase:       int
    confidence:     float
    reason:         str
    run_id:         str
    timestamp:      str


@dataclass(frozen=True)
class FetchAnomalyEvent:
    url: str
    fetch_mode: FetchMode
    status_code: Optional[int]
    anomaly_type: str
    run_id: str
    manifest_id: str
    timestamp: float = 0.0
    detail: str = ""


@dataclass(frozen=True)
class SignalExtractedEvent:
    url: str
    topology_class: str
    signal_type: str
    byte_count: int
    token_count: int
    signal_density: float
    zone_count: int
    source_component: str
    run_id: str


@dataclass(frozen=True)
class RecipeStaleEvent:
    topology_class: str
    recipe_hash: str
    reason: str
    confidence: float
    run_id: str


@dataclass(frozen=True)
class RecipeHealthEvent:
    topology_class: str
    recipe_hash: str
    sample_count: int
    success_count: int
    failure_count: int
    empty_rate: float
    median_latency_ms: float
    stale: bool
    run_id: str


@dataclass(frozen=True)
class WeightsUpdatedEvent:
    model_name: str
    store_path: str
    staging_path: str
    checksum_sha256: str
    version: int
    batch_count: int
    gradient_steps: int
    run_id: str


@dataclass(frozen=True)
class StoreHealthEvent:
    store_file: str
    status: str
    size_bytes: int
    checksum_sha256: Optional[str]
    critical: bool
    detail: str
    run_id: str


# ── Minimal exception hierarchy ───────────────────────────────────────────────

class TopologyException(Exception):
    exception_code: str = ""
    is_hard_stop: bool = False


class EventBusSubscriptionError(TopologyException):
    exception_code = "TOPOLOGY_BUS_SUBSCRIPTION"
    is_hard_stop   = True


class EventDispatchFailed(TopologyException):
    exception_code = "TOPOLOGY_EVENT_DISPATCH"
    is_hard_stop   = False


class KafkaSinkUnavailable(TopologyException):
    exception_code = "TOPOLOGY_KAFKA_UNAVAILABLE"
    is_hard_stop   = False


class EventIntegrityError(TopologyException):
    exception_code = "TOPOLOGY_EVENT_INTEGRITY"
    is_hard_stop   = False


class EventSchemaError(TopologyException):
    exception_code = "TOPOLOGY_EVENT_SCHEMA"
    is_hard_stop   = False


# ── Build stub modules and inject into sys.modules ────────────────────────────

def _build_stubs() -> None:
    contracts_mod = types.ModuleType("signal_kernel.contracts")
    contracts_mod.FetchMode                = FetchMode               # type: ignore[attr-defined]
    contracts_mod.RawFetchEvent            = RawFetchEvent           # type: ignore[attr-defined]
    contracts_mod.FetchAnomalyEvent        = FetchAnomalyEvent       # type: ignore[attr-defined]
    contracts_mod.CleanSignalEvent         = CleanSignalEvent        # type: ignore[attr-defined]
    contracts_mod.SignalExtractedEvent     = SignalExtractedEvent    # type: ignore[attr-defined]
    contracts_mod.ClassificationEvent      = ClassificationEvent     # type: ignore[attr-defined]
    contracts_mod.NewTopologyHintEvent     = NewTopologyHintEvent    # type: ignore[attr-defined]
    contracts_mod.DomainTopologyEvent      = DomainTopologyEvent     # type: ignore[attr-defined]
    contracts_mod.ZoneMapUpdatedEvent      = ZoneMapUpdatedEvent     # type: ignore[attr-defined]
    contracts_mod.ZoneMapInvalidatedEvent  = ZoneMapInvalidatedEvent # type: ignore[attr-defined]
    contracts_mod.CrawlManifestReadyEvent  = CrawlManifestReadyEvent # type: ignore[attr-defined]
    contracts_mod.ManifestCompleteEvent    = ManifestCompleteEvent   # type: ignore[attr-defined]
    contracts_mod.CLStateUpdateEvent       = CLStateUpdateEvent      # type: ignore[attr-defined]
    contracts_mod.ContainerBreachEvent     = ContainerBreachEvent    # type: ignore[attr-defined]
    contracts_mod.SurpriseEvent            = SurpriseEvent           # type: ignore[attr-defined]
    contracts_mod.RecipeStaleEvent         = RecipeStaleEvent        # type: ignore[attr-defined]
    contracts_mod.RecipeHealthEvent        = RecipeHealthEvent       # type: ignore[attr-defined]
    contracts_mod.WeightsUpdatedEvent      = WeightsUpdatedEvent     # type: ignore[attr-defined]
    contracts_mod.StoreHealthEvent         = StoreHealthEvent        # type: ignore[attr-defined]
    contracts_mod.FeedbackEvent            = FeedbackEvent           # type: ignore[attr-defined]
    contracts_mod.PhaseTransitionEvent     = PhaseTransitionEvent    # type: ignore[attr-defined]

    exceptions_mod = types.ModuleType("signal_kernel.exceptions")
    exceptions_mod.EventBusSubscriptionError = EventBusSubscriptionError # type: ignore[attr-defined]
    exceptions_mod.EventDispatchFailed       = EventDispatchFailed       # type: ignore[attr-defined]
    exceptions_mod.KafkaSinkUnavailable      = KafkaSinkUnavailable      # type: ignore[attr-defined]
    exceptions_mod.EventIntegrityError       = EventIntegrityError       # type: ignore[attr-defined]
    exceptions_mod.EventSchemaError          = EventSchemaError          # type: ignore[attr-defined]

    sk_mod = types.ModuleType("signal_kernel")

    sys.modules.setdefault("signal_kernel",            sk_mod)
    sys.modules.setdefault("signal_kernel.contracts",  contracts_mod)
    sys.modules.setdefault("signal_kernel.exceptions", exceptions_mod)


_build_stubs()

# ── Set HMAC key before crawler_bus imports ────────────────────────────────────
_TEST_HMAC_KEY = os.urandom(32).hex()
os.environ["AXIOM_BUS_HMAC_KEY"] = _TEST_HMAC_KEY
os.environ.pop("KAFKA_BOOTSTRAP_SERVERS", None)   # force degraded mode

# ── Now import crawler_bus ────────────────────────────────────────────────────
import tag.crawler_bus as cb                               # noqa: E402
from tag.crawler_bus import (                              # noqa: E402
    BackpressureMonitor,
    BusAuditEntry,
    BusAuditLog,
    BusDiagnosticResult,
    BusEnvelope,
    BusEnvCheck,
    BusEventCategory,
    BusHealth,
    BusMode,
    BusStartupDiagnostic,
    BusTestHarness,
    CircuitState,
    CrawlerBus,
    DeadLetterEvent,
    DeadLetterReader,
    EventEnvelopeValidator,
    KafkaCircuitBreaker,
    PhaseTransitionHelper,
    TopicEmitter,
    TopicHealth,
    TopicRegistry,
    _DegradedSink,
    _QUEUE_MAX_SIZE,
    _MAX_ENVELOPE_VALUE_BYTES,
    _SCHEMA_VERSION,
    _coerce_bytes_fields,
    _deserialize,
    _event_to_dict,
    _serialize,
    _sign_envelope,
    _verify_envelope,
    get_registered_topics,
    get_topic_schema,
    TOPIC_REGISTRY,
    validate_bus_environment,
)

# Align test aliases with whichever contract module crawler_bus actually bound
# during collection. This keeps the suite stable whether the standalone stubs
# or the real signal_kernel contracts were imported first.
FetchMode = cb.FetchMode
RawFetchEvent = cb.RawFetchEvent
CleanSignalEvent = cb.CleanSignalEvent
EventBusSubscriptionError = cb.EventBusSubscriptionError
EventIntegrityError = cb.EventIntegrityError
EventSchemaError = cb.EventSchemaError
KafkaSinkUnavailable = cb.KafkaSinkUnavailable


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _raw_fetch(
    url: str = "https://example.com/page",
    raw_bytes: bytes = b"<html/>",
    status_code: int = 200,
    run_id: Optional[str] = None,
) -> RawFetchEvent:
    return RawFetchEvent(
        url=url,
        raw_bytes=raw_bytes,
        status_code=status_code,
        headers={"content-type": "text/html"},
        fetch_latency=0.01,
        fetch_mode=FetchMode.STATIC,
        is_robots_txt=False,
        is_sitemap=False,
        topology_hint="GENERIC_HTML",
        run_id=run_id or str(uuid.uuid4()),
        manifest_id=str(uuid.uuid4()),
        byte_count=len(raw_bytes),
    )


def _clean_signal(topology_class: str = "NEWS_ARTICLE") -> CleanSignalEvent:
    return CleanSignalEvent(
        url="https://example.com/page",
        clean_signal="Some extracted signal text",
        topology_class=topology_class,
        token_reduction=0.42,
        signal_density=0.7,
        extraction_empty=False,
        run_id=str(uuid.uuid4()),
    )


def _signed_envelope(
    topic: str = "raw_fetch",
    value: bytes = b"\x80",
    extra_headers: Optional[Dict[str, str]] = None,
) -> BusEnvelope:
    """Build a minimal HMAC-signed BusEnvelope for deserialization tests."""
    now = datetime.now(timezone.utc).isoformat()
    headers: Dict[str, str] = {
        "run_id":           str(uuid.uuid4()),
        "schema_version":   _SCHEMA_VERSION,
        "source_component": "test",
        "emit_timestamp":   now,
        "event_type":       "Test",
        "schema_name":      "Test",
    }
    if extra_headers:
        headers.update(extra_headers)

    env = BusEnvelope(
        topic=topic,
        key=b"test-key",
        value=value,
        headers=headers,
        offset=0,
        partition=-1,
        timestamp=now,
    )
    sig = _sign_envelope(env)
    return BusEnvelope(
        topic=env.topic,
        key=env.key,
        value=env.value,
        headers={**headers, "hmac_sha256": sig},
        offset=env.offset,
        partition=env.partition,
        timestamp=env.timestamp,
    )


def _topic_health(
    *,
    total_dispatched: int = 10,
    total_failed: int = 0,
    dead_letter_count: int = 0,
    queue_depth: int = 0,
) -> TopicHealth:
    return TopicHealth(
        topic="raw_fetch",
        registered=True,
        producer_connected=True,
        subscriber_count=1,
        queue_depth=queue_depth,
        dead_letter_count=dead_letter_count,
        total_emitted=10,
        total_dispatched=total_dispatched,
        total_failed=total_failed,
        last_emit_ts=None,
        last_dispatch_ts=None,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. BusMode
# ═════════════════════════════════════════════════════════════════════════════

class TestBusMode:
    def test_kafka_str(self):
        assert str(BusMode.KAFKA) == "kafka"

    def test_degraded_str(self):
        assert str(BusMode.DEGRADED) == "degraded"

    def test_enum_distinct(self):
        assert BusMode.KAFKA is not BusMode.DEGRADED


# ═════════════════════════════════════════════════════════════════════════════
# 2. BusEnvelope
# ═════════════════════════════════════════════════════════════════════════════

class TestBusEnvelope:
    def _make(self, **overrides) -> BusEnvelope:
        base: Dict[str, Any] = dict(
            topic="raw_fetch",
            key=b"mykey",
            value=b"\x81\xa3url\xabhello",
            headers={
                "run_id":           "run-123",
                "source_component": "test.component",
                "hmac_sha256":      "aabbcc",
                "schema_version":   "1",
            },
            offset=42,
            partition=3,
            timestamp="2026-03-05T10:00:00+00:00",
        )
        base.update(overrides)
        return BusEnvelope(**base)

    def test_run_id_property(self):
        env = self._make()
        assert env.run_id == "run-123"

    def test_run_id_missing(self):
        env = self._make(headers={})
        assert env.run_id is None

    def test_source_component_property(self):
        env = self._make()
        assert env.source_component == "test.component"

    def test_hmac_digest_property(self):
        env = self._make()
        assert env.hmac_digest == "aabbcc"

    def test_schema_version_property(self):
        env = self._make()
        assert env.schema_version == "1"

    def test_schema_version_default(self):
        env = self._make(headers={})
        assert env.schema_version == "1"

    def test_value_size(self):
        env = self._make(value=b"hello")
        assert env.value_size == 5

    def test_is_oversized_false(self):
        env = self._make(value=b"x")
        assert env.is_oversized() is False

    def test_is_oversized_true(self):
        big = b"x" * (_MAX_ENVELOPE_VALUE_BYTES + 1)
        env = self._make(value=big)
        assert env.is_oversized() is True

    def test_to_log_dict_keys(self):
        env = self._make()
        d = env.to_log_dict()
        assert "topic" in d
        assert "key" in d
        assert "value_bytes" in d
        assert "offset" in d
        assert "source_component" in d

    def test_to_log_dict_value_is_size_not_bytes(self):
        env = self._make(value=b"hello")
        assert env.to_log_dict()["value_bytes"] == 5

    def test_frozen(self):
        env = self._make()
        with pytest.raises(AttributeError):
            env.topic = "other"  # type: ignore[misc] # noqa meant to exist


# ═════════════════════════════════════════════════════════════════════════════
# 3. TopicHealth
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicHealth:
    def test_error_rate_zero_dispatches(self):
        th = _topic_health(total_dispatched=0, total_failed=0)
        assert th.error_rate == 0.0

    def test_error_rate_no_failures(self):
        th = _topic_health(total_dispatched=100, total_failed=0)
        assert th.error_rate == 0.0

    def test_error_rate_with_failures(self):
        th = _topic_health(total_dispatched=8, total_failed=2)
        assert abs(th.error_rate - 0.2) < 1e-9

    def test_is_healthy_true(self):
        assert _topic_health().is_healthy is True

    def test_is_healthy_false_high_error_rate(self):
        # error_rate > 0.1 should be unhealthy
        th = _topic_health(total_dispatched=1, total_failed=9)
        assert th.is_healthy is False

    def test_is_healthy_false_dead_letters(self):
        th = _topic_health(dead_letter_count=100)
        assert th.is_healthy is False

    def test_is_healthy_false_high_queue_depth(self):
        th = _topic_health(queue_depth=int(_QUEUE_MAX_SIZE * 0.9))
        assert th.is_healthy is False


# ═════════════════════════════════════════════════════════════════════════════
# 4. BusHealth
# ═════════════════════════════════════════════════════════════════════════════

class TestBusHealth:
    def _make_health(self, started: bool = True, topics_healthy: bool = True) -> BusHealth:
        th = _topic_health() if topics_healthy else _topic_health(dead_letter_count=999)
        return BusHealth(
            mode="degraded",
            topics={"raw_fetch": th},
            dead_letters=0,
            lag={},
            uptime_s=1.0,
            started=started,
            kafka_connected=False,
            total_emitted=5,
            total_dispatched=5,
            total_failed=0,
            hmac_failures=0,
            schema_failures=0,
        )

    def test_is_healthy_not_started(self):
        h = self._make_health(started=False)
        assert h.is_healthy is False

    def test_is_healthy_unhealthy_topic(self):
        h = self._make_health(topics_healthy=False)
        assert h.is_healthy is False

    def test_is_healthy_true(self):
        h = self._make_health()
        assert h.is_healthy is True

    def test_to_log_dict_has_mode(self):
        h = self._make_health()
        d = h.to_log_dict()
        assert d["mode"] == "degraded"
        assert "total_emitted" in d
        assert "uptime_s" in d


# ═════════════════════════════════════════════════════════════════════════════
# 5. DeadLetterEvent
# ═════════════════════════════════════════════════════════════════════════════

class TestDeadLetterEvent:
    def _make(self) -> DeadLetterEvent:
        return DeadLetterEvent(
            topic="raw_fetch",
            event="deadbeef",
            handler="my.Handler._on_event",
            error="ValueError: bad value",
            retries=3,
            timestamp="2026-03-05T10:00:00+00:00",
            mode="degraded",
            run_id="run-abc",
            partition=-1,
            offset=7,
            source_component="test.component",
            envelope_key="mykey",
        )

    def test_to_jsonl_line_is_bytes(self):
        assert isinstance(self._make().to_jsonl_line(), bytes)

    def test_to_jsonl_line_ends_with_newline(self):
        assert self._make().to_jsonl_line().endswith(b"\n")

    def test_to_jsonl_line_parseable(self):
        import orjson
        line = self._make().to_jsonl_line()
        obj = orjson.loads(line)
        assert obj["topic"] == "raw_fetch"
        assert obj["handler"] == "my.Handler._on_event"
        assert obj["retries"] == 3


# ═════════════════════════════════════════════════════════════════════════════
# 6. TopicRegistry
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicRegistry:
    def test_construction(self):
        r = TopicRegistry({"raw_fetch": RawFetchEvent})
        assert "raw_fetch" in r

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            TopicRegistry({})

    def test_validate_topic_valid(self):
        r = TopicRegistry({"raw_fetch": RawFetchEvent})
        schema = r.validate_topic("raw_fetch")
        assert schema is RawFetchEvent

    def test_validate_topic_unknown_raises(self):
        r = TopicRegistry({"raw_fetch": RawFetchEvent})
        with pytest.raises(EventBusSubscriptionError):
            r.validate_topic("nonexistent_topic")

    def test_contains_true(self):
        r = TopicRegistry({"raw_fetch": RawFetchEvent})
        assert "raw_fetch" in r

    def test_contains_false(self):
        r = TopicRegistry({"raw_fetch": RawFetchEvent})
        assert "nope" not in r

    def test_all_topics_sorted(self):
        r = TopicRegistry({"z_topic": RawFetchEvent, "a_topic": CleanSignalEvent})
        assert r.all_topics == ["a_topic", "z_topic"]

    def test_schema_for_alias(self):
        r = TopicRegistry({"raw_fetch": RawFetchEvent})
        assert r.schema_for("raw_fetch") is RawFetchEvent


# ═════════════════════════════════════════════════════════════════════════════
# 7. TOPIC_REGISTRY and helpers
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicRegistryGlobal:
    def test_all_contract_topics_registered(self):
        assert len(TOPIC_REGISTRY) == 20

    def test_expected_topics_present(self):
        expected = {
            "raw_fetch", "fetch_anomaly", "clean_signal", "signal_extracted",
            "classification", "topology_hint", "domain_topology", "crawl_manifest",
            "manifest_complete", "cl_state_update", "container_breach",
            "zone_map_updated", "zone_map_invalidated", "surprise",
            "recipe_stale", "recipe_health", "weights_updated", "store_health",
            "feedback", "phase_transition",
        }
        assert set(TOPIC_REGISTRY.keys()) == expected

    def test_get_registered_topics_length(self):
        assert len(get_registered_topics()) == len(TOPIC_REGISTRY)

    def test_topic_documentation_matches_registry(self):
        assert set(cb.TOPIC_DOCUMENTATION.keys()) == set(TOPIC_REGISTRY.keys())

    def test_get_registered_topics_sorted(self):
        topics = get_registered_topics()
        assert topics == sorted(topics)

    def test_get_topic_schema_valid(self):
        schema = get_topic_schema("raw_fetch")
        assert schema is RawFetchEvent

    def test_get_topic_schema_invalid(self):
        with pytest.raises(KeyError):
            get_topic_schema("not_a_real_topic")


# ═════════════════════════════════════════════════════════════════════════════
# 8. HMAC — sign, verify, tamper detection
# ═════════════════════════════════════════════════════════════════════════════

class TestHMAC:
    def _envelope(self, value: bytes = b"\x80") -> BusEnvelope:
        now = datetime.now(timezone.utc).isoformat()
        return BusEnvelope(
            topic="raw_fetch",
            key=b"k",
            value=value,
            headers={
                "run_id": "r1",
                "schema_version": "1",
                "source_component": "test",
                "emit_timestamp": now,
            },
            offset=0,
            partition=-1,
            timestamp=now,
        )

    def test_sign_returns_str(self):
        env = self._envelope()
        assert isinstance(_sign_envelope(env), str)

    def test_sign_non_empty(self):
        env = self._envelope()
        assert len(_sign_envelope(env)) > 0

    def test_sign_deterministic(self):
        env = self._envelope()
        assert _sign_envelope(env) == _sign_envelope(env)

    def test_verify_valid(self):
        env = self._envelope()
        sig = _sign_envelope(env)
        signed = BusEnvelope(
            topic=env.topic, key=env.key, value=env.value,
            headers={**env.headers, "hmac_sha256": sig},
            offset=env.offset, partition=env.partition, timestamp=env.timestamp,
        )
        assert _verify_envelope(signed) is True

    def test_verify_tampered_value(self):
        env = self._envelope()
        sig = _sign_envelope(env)
        tampered = BusEnvelope(
            topic=env.topic, key=env.key, value=b"\x00TAMPERED",
            headers={**env.headers, "hmac_sha256": sig},
            offset=env.offset, partition=env.partition, timestamp=env.timestamp,
        )
        assert _verify_envelope(tampered) is False

    def test_verify_tampered_key(self):
        env = self._envelope()
        sig = _sign_envelope(env)
        tampered = BusEnvelope(
            topic=env.topic, key=b"WRONG", value=env.value,
            headers={**env.headers, "hmac_sha256": sig},
            offset=env.offset, partition=env.partition, timestamp=env.timestamp,
        )
        assert _verify_envelope(tampered) is False

    def test_verify_missing_header(self):
        env = self._envelope()
        # No hmac_sha256 in headers
        assert _verify_envelope(env) is False

    def test_verify_wrong_signature(self):
        env = self._envelope()
        bad = BusEnvelope(
            topic=env.topic, key=env.key, value=env.value,
            headers={**env.headers, "hmac_sha256": "deadbeef" * 8},
            offset=env.offset, partition=env.partition, timestamp=env.timestamp,
        )
        assert _verify_envelope(bad) is False

    def test_different_values_different_signatures(self):
        env_a = self._envelope(value=b"AAA")
        env_b = self._envelope(value=b"BBB")
        assert _sign_envelope(env_a) != _sign_envelope(env_b)


# ═════════════════════════════════════════════════════════════════════════════
# 9. _event_to_dict
# ═════════════════════════════════════════════════════════════════════════════

class TestEventToDict:
    def test_passthrough_int(self):
        assert _event_to_dict(42) == 42

    def test_passthrough_str(self):
        assert _event_to_dict("hello") == "hello"

    def test_passthrough_none(self):
        assert _event_to_dict(None) is None

    def test_datetime_to_isoformat(self):
        dt = datetime(2026, 3, 5, 10, 0, 0, tzinfo=timezone.utc)
        result = _event_to_dict(dt)
        assert isinstance(result, str)
        assert "2026" in result

    def test_bytes_to_list_of_ints(self):
        result = _event_to_dict(b"\x01\x02\x03")
        assert result == [1, 2, 3]

    def test_frozenset_to_list(self):
        result = _event_to_dict(frozenset({1, 2, 3}))
        assert sorted(result) == [1, 2, 3]

    def test_list_recursive(self):
        result = _event_to_dict([b"\x01", b"\x02"])
        assert result == [[1], [2]]

    def test_tuple_recursive(self):
        result = _event_to_dict((b"\xff",))
        assert result == [[255]]

    def test_dict_recursive(self):
        result = _event_to_dict({"key": b"\xab"})
        assert result == {"key": [171]}

    def test_dataclass_to_dict(self):
        event = _raw_fetch()
        result = _event_to_dict(event)
        assert isinstance(result, dict)
        assert result["url"] == event.url
        assert isinstance(result["raw_bytes"], list)


# ═════════════════════════════════════════════════════════════════════════════
# 10. _serialize
# ═════════════════════════════════════════════════════════════════════════════

class TestSerialize:
    def test_non_dataclass_raises_type_error(self):
        with pytest.raises(TypeError):
            _serialize("not a dataclass", "raw_fetch", "test", offset=0)

    def test_dataclass_class_not_instance_raises(self):
        with pytest.raises(TypeError):
            _serialize(RawFetchEvent, "raw_fetch", "test", offset=0)

    def test_returns_bus_envelope(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "test", offset=1)
        assert isinstance(env, BusEnvelope)

    def test_topic_preserved(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "test", offset=0)
        assert env.topic == "raw_fetch"

    def test_offset_preserved(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "test", offset=99)
        assert env.offset == 99

    def test_key_from_url_when_no_topology_class(self):
        event = _raw_fetch(url="https://example.com/test")
        env = _serialize(event, "raw_fetch", "test", offset=0)
        assert b"example.com" in env.key

    def test_key_from_topology_class(self):
        event = _clean_signal(topology_class="NEWS_ARTICLE")
        env = _serialize(event, "clean_signal", "test", offset=0)
        assert env.key == b"NEWS_ARTICLE"

    def test_headers_have_required_fields(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "test.comp", offset=0)
        assert "schema_version" in env.headers
        assert "source_component" in env.headers
        assert "emit_timestamp" in env.headers
        assert "event_type" in env.headers

    def test_source_component_in_headers(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "my.component", offset=0)
        assert env.headers["source_component"] == "my.component"

    def test_value_is_bytes(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "test", offset=0)
        assert isinstance(env.value, bytes)
        assert len(env.value) > 0

    def test_oversized_raises_value_error(self):
        big_event = _raw_fetch(raw_bytes=b"x" * (_MAX_ENVELOPE_VALUE_BYTES + 1))
        with pytest.raises(ValueError, match="exceeding"):
            _serialize(big_event, "raw_fetch", "test", offset=0)


# ═════════════════════════════════════════════════════════════════════════════
# 11. _coerce_bytes_fields
# ═════════════════════════════════════════════════════════════════════════════

class TestCoerceBytesFields:
    def test_coerces_list_of_ints_to_bytes(self):
        result = _coerce_bytes_fields(
            {"raw_bytes": [72, 101, 108, 108, 111]},
            RawFetchEvent,
        )
        assert result["raw_bytes"] == b"Hello"

    def test_no_coerce_if_already_bytes(self):
        result = _coerce_bytes_fields({"raw_bytes": b"Hello"}, RawFetchEvent)
        assert result["raw_bytes"] == b"Hello"

    def test_non_dataclass_passthrough(self):
        d = {"x": [1, 2, 3]}
        assert _coerce_bytes_fields(d, dict) == d  # type: ignore[arg-type]

    def test_non_bytes_field_untouched(self):
        result = _coerce_bytes_fields({"url": "https://x.com"}, RawFetchEvent)
        assert result["url"] == "https://x.com"


# ═════════════════════════════════════════════════════════════════════════════
# 12. _serialize / _deserialize roundtrip
# ═════════════════════════════════════════════════════════════════════════════

class TestSerializeDeserializeRoundtrip:
    def _signed(self, event: Any, topic: str) -> BusEnvelope:
        env = _serialize(event, topic, "test.component", offset=1)
        sig = _sign_envelope(env)
        return BusEnvelope(
            topic=env.topic, key=env.key, value=env.value,
            headers={**env.headers, "hmac_sha256": sig},
            offset=env.offset, partition=env.partition, timestamp=env.timestamp,
        )

    def test_raw_fetch_roundtrip(self):
        original = _raw_fetch(url="https://test.com/page", raw_bytes=b"<html>hi</html>")
        signed_env = self._signed(original, "raw_fetch")
        recovered = _deserialize(signed_env, RawFetchEvent)
        assert recovered.url == original.url
        assert recovered.raw_bytes == original.raw_bytes
        assert recovered.status_code == original.status_code
        assert recovered.run_id == original.run_id

    def test_clean_signal_roundtrip(self):
        original = _clean_signal("ECOMMERCE_PRODUCT")
        signed_env = self._signed(original, "clean_signal")
        recovered = _deserialize(signed_env, CleanSignalEvent)
        assert recovered.topology_class == "ECOMMERCE_PRODUCT"
        assert recovered.clean_signal == original.clean_signal

    def test_bad_hmac_raises_integrity_error(self):
        env = _serialize(_raw_fetch(), "raw_fetch", "test", offset=0)
        tampered = BusEnvelope(
            topic=env.topic, key=env.key, value=env.value,
            headers={**env.headers, "hmac_sha256": "badhmacsignature"},
            offset=env.offset, partition=env.partition, timestamp=env.timestamp,
        )
        with pytest.raises(EventIntegrityError):
            _deserialize(tampered, RawFetchEvent)

    def test_corrupt_msgpack_raises_schema_error(self):
        signed = _signed_envelope(value=b"\xff\xfe NOT VALID MSGPACK ///")
        with pytest.raises(EventSchemaError):
            _deserialize(signed, RawFetchEvent)

    def test_schema_mismatch_raises_schema_error(self):
        # Serialize a CleanSignalEvent but try to deserialize as RawFetchEvent
        original = _clean_signal()
        signed_env = self._signed(original, "clean_signal")
        with pytest.raises(EventSchemaError):
            _deserialize(signed_env, RawFetchEvent)

    def test_non_dict_msgpack_raises_schema_error(self):
        import msgpack
        # Pack a list, not a dict
        list_value = msgpack.packb(["a", "b", "c"], use_bin_type=True)
        signed = _signed_envelope(value=list_value)
        with pytest.raises(EventSchemaError):
            _deserialize(signed, RawFetchEvent)


# ═════════════════════════════════════════════════════════════════════════════
# 13. TopicEmitter
# ═════════════════════════════════════════════════════════════════════════════

class TestTopicEmitter:
    """Tests that don't need a live bus — only test the emitter's guard logic."""

    @pytest.mark.asyncio
    async def test_wrong_schema_raises_type_error(self):
        async with BusTestHarness() as harness:
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            wrong_event = _clean_signal()  # CleanSignalEvent, not RawFetchEvent
            with pytest.raises(TypeError):
                await emitter.emit(wrong_event)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_disabled_raises_runtime_error(self):
        async with BusTestHarness() as harness:
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            emitter.disable()
            with pytest.raises(RuntimeError):
                await emitter.emit(_raw_fetch())

    @pytest.mark.asyncio
    async def test_emit_count_increments(self):
        async with BusTestHarness() as harness:
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            assert emitter.emit_count == 0
            await emitter.emit(_raw_fetch())
            assert emitter.emit_count == 1
            await emitter.emit(_raw_fetch())
            assert emitter.emit_count == 2

    @pytest.mark.asyncio
    async def test_last_emit_ts_set_after_emit(self):
        async with BusTestHarness() as harness:
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            assert emitter.last_emit_ts is None
            await emitter.emit(_raw_fetch())
            assert emitter.last_emit_ts is not None

    def test_topic_property(self):
        # Construct a dummy emitter without a live bus using direct instantiation
        bus = CrawlerBus()
        emitter: TopicEmitter = TopicEmitter(
            bus=bus, topic="raw_fetch", component="x", schema=RawFetchEvent
        )
        assert emitter.topic == "raw_fetch"

    def test_component_property(self):
        bus = CrawlerBus()
        emitter: TopicEmitter = TopicEmitter(
            bus=bus, topic="raw_fetch", component="my.comp", schema=RawFetchEvent
        )
        assert emitter.component == "my.comp"

    def test_schema_property(self):
        bus = CrawlerBus()
        emitter: TopicEmitter = TopicEmitter(
            bus=bus, topic="raw_fetch", component="x", schema=RawFetchEvent
        )
        assert emitter.schema is RawFetchEvent


# ═════════════════════════════════════════════════════════════════════════════
# 14. _DegradedSink
# ═════════════════════════════════════════════════════════════════════════════

class TestDegradedSink:
    @pytest.mark.asyncio
    async def test_register_topic_and_put_get(self):
        sink = _DegradedSink()
        sink.register_topic("raw_fetch")
        now = datetime.now(timezone.utc).isoformat()
        env = BusEnvelope(
            topic="raw_fetch", key=b"k", value=b"\x80",
            headers={"run_id": "r"}, offset=1, partition=-1, timestamp=now,
        )
        await sink.put(env)
        result = await sink.get("raw_fetch")
        assert result is env

    @pytest.mark.asyncio
    async def test_get_empty_returns_none(self):
        sink = _DegradedSink()
        sink.register_topic("raw_fetch")
        assert await sink.get("raw_fetch") is None

    @pytest.mark.asyncio
    async def test_get_unregistered_returns_none(self):
        sink = _DegradedSink()
        assert await sink.get("not_registered") is None

    @pytest.mark.asyncio
    async def test_put_unregistered_raises(self):
        sink = _DegradedSink()
        now = datetime.now(timezone.utc).isoformat()
        env = BusEnvelope(
            topic="ghost_topic", key=b"k", value=b"\x80",
            headers={}, offset=0, partition=-1, timestamp=now,
        )
        with pytest.raises(EventBusSubscriptionError):
            await sink.put(env)

    def test_queue_depth_zero_initially(self):
        sink = _DegradedSink()
        sink.register_topic("raw_fetch")
        assert sink.queue_depth("raw_fetch") == 0

    def test_queue_depth_unregistered_is_zero(self):
        sink = _DegradedSink()
        assert sink.queue_depth("nope") == 0

    @pytest.mark.asyncio
    async def test_queue_depth_after_put(self):
        sink = _DegradedSink()
        sink.register_topic("raw_fetch")
        now = datetime.now(timezone.utc).isoformat()
        env = BusEnvelope(
            topic="raw_fetch", key=b"k", value=b"\x80",
            headers={}, offset=1, partition=-1, timestamp=now,
        )
        await sink.put(env)
        assert sink.queue_depth("raw_fetch") == 1

    def test_next_offset_increments(self):
        sink = _DegradedSink()
        sink.register_topic("raw_fetch")
        assert sink.next_offset("raw_fetch") == 1
        assert sink.next_offset("raw_fetch") == 2
        assert sink.next_offset("raw_fetch") == 3

    def test_next_offset_independent_per_topic(self):
        sink = _DegradedSink()
        sink.register_topic("raw_fetch")
        sink.register_topic("clean_signal")
        sink.next_offset("raw_fetch")
        sink.next_offset("raw_fetch")
        assert sink.next_offset("clean_signal") == 1


# ═════════════════════════════════════════════════════════════════════════════
# 15. KafkaCircuitBreaker
# ═════════════════════════════════════════════════════════════════════════════

class TestKafkaCircuitBreaker:
    def test_initial_state_closed(self):
        cb = KafkaCircuitBreaker()
        assert cb.state is CircuitState.CLOSED

    def test_allow_request_in_closed(self):
        cb = KafkaCircuitBreaker()
        assert cb.allow_request() is True

    def test_failure_count_increments(self):
        cb = KafkaCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        assert cb.failure_count == 1

    def test_circuit_opens_at_threshold(self):
        cb = KafkaCircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state is CircuitState.CLOSED
        cb.record_failure()
        assert cb.state is CircuitState.OPEN

    def test_allow_request_false_when_open(self):
        cb = KafkaCircuitBreaker(failure_threshold=1, open_duration_s=9999.0)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        assert cb.allow_request() is False

    def test_transitions_to_half_open_after_duration(self):
        cb = KafkaCircuitBreaker(failure_threshold=1, open_duration_s=0.0)
        cb.record_failure()
        assert cb.state is CircuitState.OPEN
        # open_duration_s=0.0 means immediately eligible
        result = cb.allow_request()
        assert result is True
        assert cb.state is CircuitState.HALF_OPEN

    def test_success_in_half_open_closes_circuit(self):
        cb = KafkaCircuitBreaker(failure_threshold=1, open_duration_s=0.0)
        cb.record_failure()          # → OPEN
        cb.allow_request()           # → HALF_OPEN
        cb.record_success()          # → CLOSED
        assert cb.state is CircuitState.CLOSED

    def test_failure_in_half_open_reopens(self):
        cb = KafkaCircuitBreaker(failure_threshold=1, open_duration_s=0.0)
        cb.record_failure()          # → OPEN
        cb.allow_request()           # → HALF_OPEN
        cb.record_failure()          # → OPEN again
        assert cb.state is CircuitState.OPEN

    def test_success_resets_failure_count_in_closed(self):
        cb = KafkaCircuitBreaker(failure_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.failure_count == 0

    def test_total_opens_increments(self):
        cb = KafkaCircuitBreaker(failure_threshold=1, open_duration_s=0.0)
        assert cb.total_opens == 0
        cb.record_failure()
        assert cb.total_opens == 1

    def test_to_dict_keys(self):
        cb = KafkaCircuitBreaker()
        d = cb.to_dict()
        assert "state" in d
        assert "failure_count" in d
        assert "total_opens" in d


# ═════════════════════════════════════════════════════════════════════════════
# 16. EventEnvelopeValidator
# ═════════════════════════════════════════════════════════════════════════════

class TestEventEnvelopeValidator:
    def _registry(self) -> TopicRegistry:
        return TopicRegistry(TOPIC_REGISTRY)

    def _valid_envelope(self) -> BusEnvelope:
        now = datetime.now(timezone.utc).isoformat()
        return BusEnvelope(
            topic="raw_fetch",
            key=b"k",
            value=b"\x80",
            headers={
                "schema_version":   "1",
                "source_component": "test",
                "emit_timestamp":   now,
                "hmac_sha256":      "aabbcc",
            },
            offset=0,
            partition=-1,
            timestamp=now,
        )

    def test_valid_envelope_passes(self):
        v = EventEnvelopeValidator(self._registry())
        v.validate(self._valid_envelope())  # no exception

    def test_unknown_topic_raises(self):
        v = EventEnvelopeValidator(self._registry())
        now = datetime.now(timezone.utc).isoformat()
        env = BusEnvelope(
            topic="ghost_topic", key=b"k", value=b"\x80",
            headers={"schema_version": "1", "source_component": "t",
                     "emit_timestamp": now, "hmac_sha256": "x"},
            offset=0, partition=-1, timestamp=now,
        )
        with pytest.raises(EventSchemaError):
            v.validate(env)

    def test_empty_value_raises(self):
        v = EventEnvelopeValidator(self._registry())
        env = self._valid_envelope()
        now = datetime.now(timezone.utc).isoformat()
        empty = BusEnvelope(
            topic=env.topic, key=env.key, value=b"",
            headers=env.headers, offset=env.offset,
            partition=env.partition, timestamp=env.timestamp,
        )
        with pytest.raises(EventSchemaError):
            v.validate(empty)

    def test_oversized_value_raises(self):
        v = EventEnvelopeValidator(self._registry())
        now = datetime.now(timezone.utc).isoformat()
        huge = BusEnvelope(
            topic="raw_fetch", key=b"k",
            value=b"x" * (_MAX_ENVELOPE_VALUE_BYTES + 1),
            headers={"schema_version": "1", "source_component": "t",
                     "emit_timestamp": now, "hmac_sha256": "x"},
            offset=0, partition=-1, timestamp=now,
        )
        with pytest.raises(EventSchemaError):
            v.validate(huge)

    def test_missing_required_header_raises(self):
        v = EventEnvelopeValidator(self._registry())
        env = self._valid_envelope()
        now = datetime.now(timezone.utc).isoformat()
        # Remove 'hmac_sha256'
        partial_headers = {
            "schema_version": "1",
            "source_component": "t",
            "emit_timestamp": now,
            # hmac_sha256 missing
        }
        broken = BusEnvelope(
            topic=env.topic, key=env.key, value=env.value,
            headers=partial_headers, offset=env.offset,
            partition=env.partition, timestamp=env.timestamp,
        )
        with pytest.raises(EventSchemaError):
            v.validate(broken)

    def test_unsupported_schema_version_raises(self):
        v = EventEnvelopeValidator(self._registry())
        now = datetime.now(timezone.utc).isoformat()
        env = BusEnvelope(
            topic="raw_fetch", key=b"k", value=b"\x80",
            headers={"schema_version": "99", "source_component": "t",
                     "emit_timestamp": now, "hmac_sha256": "x"},
            offset=0, partition=-1, timestamp=now,
        )
        with pytest.raises(EventSchemaError):
            v.validate(env)

    def test_invalid_timestamp_raises(self):
        v = EventEnvelopeValidator(self._registry())
        env = BusEnvelope(
            topic="raw_fetch", key=b"k", value=b"\x80",
            headers={"schema_version": "1", "source_component": "t",
                     "emit_timestamp": "NOT-A-DATE", "hmac_sha256": "x"},
            offset=0, partition=-1, timestamp="NOT-A-DATE",
        )
        with pytest.raises(EventSchemaError):
            v.validate(env)


# ═════════════════════════════════════════════════════════════════════════════
# 17. DeadLetterReader
# ═════════════════════════════════════════════════════════════════════════════

class TestDeadLetterReader:
    def _write_records(self, path: Path, records: List[Dict[str, Any]]) -> None:
        import orjson
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for r in records:
                f.write(orjson.dumps(r) + b"\n")

    def test_read_all_missing_file(self, tmp_path):
        reader = DeadLetterReader(tmp_path / "dead_letters.jsonl")
        assert reader.read_all() == []

    def test_read_all_valid_jsonl(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [
            {"topic": "raw_fetch",   "handler": "h1", "retries": 3},
            {"topic": "clean_signal","handler": "h2", "retries": 3},
        ])
        reader = DeadLetterReader(p)
        records = reader.read_all()
        assert len(records) == 2
        assert records[0]["topic"] == "raw_fetch"

    def test_count_by_handler(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [
            {"handler": "h1"}, {"handler": "h1"}, {"handler": "h2"},
        ])
        reader = DeadLetterReader(p)
        counts = reader.count_by_handler()
        assert counts["h1"] == 2
        assert counts["h2"] == 1

    def test_count_by_topic(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [
            {"topic": "raw_fetch"}, {"topic": "raw_fetch"}, {"topic": "clean_signal"},
        ])
        reader = DeadLetterReader(p)
        counts = reader.count_by_topic()
        assert counts["raw_fetch"] == 2
        assert counts["clean_signal"] == 1

    def test_find_repeated_failures_above_threshold(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [
            {"handler": "bad.Handler"}, {"handler": "bad.Handler"}, {"handler": "bad.Handler"},
            {"handler": "ok.Handler"},
        ])
        reader = DeadLetterReader(p)
        repeated = reader.find_repeated_failures(min_count=3)
        assert "bad.Handler" in repeated
        assert "ok.Handler" not in repeated

    def test_find_repeated_failures_below_threshold(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [
            {"handler": "h1"}, {"handler": "h1"},
        ])
        reader = DeadLetterReader(p)
        assert reader.find_repeated_failures(min_count=3) == {}

    def test_total_count(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [{"handler": "h"}] * 5)
        reader = DeadLetterReader(p)
        assert reader.total_count() == 5

    def test_last_n(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        records = [{"handler": f"h{i}"} for i in range(10)]
        self._write_records(p, records)
        reader = DeadLetterReader(p)
        last3 = reader.last_n(3)
        assert len(last3) == 3
        assert last3[-1]["handler"] == "h9"

    def test_since_timestamp_filter(self, tmp_path):
        p = tmp_path / "dead_letters.jsonl"
        self._write_records(p, [
            {"handler": "old", "timestamp": "2026-01-01T00:00:00"},
            {"handler": "new", "timestamp": "2026-06-01T00:00:00"},
        ])
        reader = DeadLetterReader(p)
        results = reader.since("2026-03-01T00:00:00")
        assert len(results) == 1
        assert results[0]["handler"] == "new"


# ═════════════════════════════════════════════════════════════════════════════
# 18. BackpressureMonitor
# ═════════════════════════════════════════════════════════════════════════════

class TestBackpressureMonitor:
    def test_below_warn_no_saturation_event(self):
        m = BackpressureMonitor()
        m.check("raw_fetch", depth=int(_QUEUE_MAX_SIZE * 0.1))
        assert m.total_saturation_events == 0

    def test_above_warn_threshold_logs(self):
        m = BackpressureMonitor()
        m.check("raw_fetch", depth=int(_QUEUE_MAX_SIZE * 0.6))
        assert m.total_saturation_events == 1

    def test_above_critical_threshold_logs(self):
        m = BackpressureMonitor()
        m.check("raw_fetch", depth=int(_QUEUE_MAX_SIZE * 0.9))
        assert m.total_saturation_events == 1

    def test_rate_limited_does_not_double_log(self):
        m = BackpressureMonitor()
        # First call logs
        m.check("raw_fetch", depth=int(_QUEUE_MAX_SIZE * 0.6))
        # Second call immediately after — rate limited
        m.check("raw_fetch", depth=int(_QUEUE_MAX_SIZE * 0.6))
        assert m.total_saturation_events == 1

    def test_different_topics_independent(self):
        m = BackpressureMonitor()
        m.check("raw_fetch",    depth=int(_QUEUE_MAX_SIZE * 0.6))
        m.check("clean_signal", depth=int(_QUEUE_MAX_SIZE * 0.6))
        assert m.total_saturation_events == 2


# ═════════════════════════════════════════════════════════════════════════════
# 19. BusEnvCheck / validate_bus_environment
# ═════════════════════════════════════════════════════════════════════════════

class TestBusEnvCheck:
    def test_is_ready_true_with_key(self):
        check = BusEnvCheck(
            hmac_key_present=True,
            hmac_key_length_ok=True,
            kafka_servers_set=False,
            tls_certs_present=False,
            dead_letter_dir_writable=True,
            errors=[],
            warnings=[],
        )
        assert check.is_ready is True

    def test_is_ready_false_key_missing(self):
        check = BusEnvCheck(
            hmac_key_present=False,
            hmac_key_length_ok=False,
            kafka_servers_set=False,
            tls_certs_present=False,
            dead_letter_dir_writable=True,
            errors=["HMAC key missing"],
            warnings=[],
        )
        assert check.is_ready is False

    def test_is_ready_false_with_errors(self):
        check = BusEnvCheck(
            hmac_key_present=True,
            hmac_key_length_ok=True,
            kafka_servers_set=False,
            tls_certs_present=False,
            dead_letter_dir_writable=False,
            errors=["Cannot write to /store"],
            warnings=[],
        )
        assert check.is_ready is False

    def test_validate_environment_with_key_set(self):
        # AXIOM_BUS_HMAC_KEY is set in the module-level setup
        result = validate_bus_environment()
        assert result.hmac_key_present is True
        assert result.hmac_key_length_ok is True

    def test_validate_environment_no_kafka(self):
        result = validate_bus_environment()
        assert result.kafka_servers_set is False


# ═════════════════════════════════════════════════════════════════════════════
# 20. PhaseTransitionHelper
# ═════════════════════════════════════════════════════════════════════════════

class TestPhaseTransitionHelper:
    def test_valid_transitions(self):
        assert PhaseTransitionHelper.validate_transition(1, 2) is True
        assert PhaseTransitionHelper.validate_transition(2, 3) is True
        assert PhaseTransitionHelper.validate_transition(3, 2) is True
        assert PhaseTransitionHelper.validate_transition(2, 1) is True

    def test_invalid_skip_transition(self):
        assert PhaseTransitionHelper.validate_transition(1, 3) is False

    def test_invalid_self_transition(self):
        assert PhaseTransitionHelper.validate_transition(1, 1) is False
        assert PhaseTransitionHelper.validate_transition(2, 2) is False

    def test_invalid_reverse_skip(self):
        assert PhaseTransitionHelper.validate_transition(3, 1) is False

    def test_describe_phases(self):
        assert PhaseTransitionHelper.describe(1) == "learns"
        assert PhaseTransitionHelper.describe(2) == "predicts"
        assert PhaseTransitionHelper.describe(3) == "knows"

    def test_describe_unknown(self):
        result = PhaseTransitionHelper.describe(99)
        assert "unknown" in result or "99" in result


# ═════════════════════════════════════════════════════════════════════════════
# 21. BusAuditEntry / BusAuditLog
# ═════════════════════════════════════════════════════════════════════════════

class TestBusAuditLog:
    def test_audit_entry_to_jsonl_line(self):
        import orjson
        entry = BusAuditEntry(
            category=BusEventCategory.STARTUP.value,
            event="bus_started",
            detail="Bus entered degraded mode",
            mode="degraded",
            timestamp="2026-03-05T10:00:00+00:00",
        )
        line = entry.to_jsonl_line()
        obj = orjson.loads(line)
        assert obj["event"] == "bus_started"
        assert obj["mode"] == "degraded"

    def test_audit_log_write_creates_file(self, tmp_path):
        log_path = tmp_path / "bus_events.log"
        audit = BusAuditLog(path=log_path)
        audit.write(
            category=BusEventCategory.STARTUP,
            event="test_event",
            detail="unit test",
            mode="degraded",
        )
        assert log_path.exists()
        assert audit.entries_written == 1

    def test_audit_log_write_multiple(self, tmp_path):
        log_path = tmp_path / "bus_events.log"
        audit = BusAuditLog(path=log_path)
        for i in range(5):
            audit.write(BusEventCategory.STARTUP, f"event_{i}", "detail", "degraded")
        assert audit.entries_written == 5

    def test_audit_log_rotation(self, tmp_path):
        log_path = tmp_path / "bus_events.log"
        audit = BusAuditLog(path=log_path)
        # Manually write beyond max size
        with open(log_path, "wb") as f:
            f.write(b"x" * (10 * 1024 * 1024 + 1))
        # Next write should rotate
        audit.write(BusEventCategory.SHUTDOWN, "shutdown", "detail", "degraded")
        rotated = log_path.with_suffix(".log.1")
        assert rotated.exists()


# ═════════════════════════════════════════════════════════════════════════════
# 22. Integration tests via BusTestHarness
# ═════════════════════════════════════════════════════════════════════════════

class TestBusTestHarness:
    @pytest.mark.asyncio
    async def test_harness_starts_in_degraded_mode(self):
        async with BusTestHarness() as harness:
            assert harness.mode is BusMode.DEGRADED

    @pytest.mark.asyncio
    async def test_health_started_after_start(self):
        async with BusTestHarness() as harness:
            h = harness.health()
            assert h.started is True

    @pytest.mark.asyncio
    async def test_health_mode_is_degraded(self):
        async with BusTestHarness() as harness:
            h = harness.health()
            assert h.mode == "degraded"

    @pytest.mark.asyncio
    async def test_health_has_all_topics(self):
        async with BusTestHarness() as harness:
            h = harness.health()
            assert len(h.topics) == len(TOPIC_REGISTRY)

    @pytest.mark.asyncio
    async def test_emitter_unknown_topic_raises(self):
        async with BusTestHarness() as harness:
            with pytest.raises(EventBusSubscriptionError):
                await harness.emitter("not_real", "test", RawFetchEvent)

    @pytest.mark.asyncio
    async def test_subscribe_unknown_topic_raises(self):
        async with BusTestHarness() as harness:
            async def handler(event): pass
            with pytest.raises(EventBusSubscriptionError):
                await harness.subscribe("not_real", "grp", handler, RawFetchEvent)

    @pytest.mark.asyncio
    async def test_subscribe_sync_handler_raises(self):
        async with BusTestHarness() as harness:
            def sync_handler(event): pass  # not async
            with pytest.raises(EventBusSubscriptionError):
                await harness.subscribe("raw_fetch", "grp", sync_handler, RawFetchEvent)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_emit_increments_emit_count(self):
        async with BusTestHarness() as harness:
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            await emitter.emit(_raw_fetch())
            await emitter.emit(_raw_fetch())
            assert harness.emit_count("raw_fetch") == 2

    @pytest.mark.asyncio
    async def test_emit_and_dispatch_single_event(self):
        received: List[RawFetchEvent] = []

        async def handler(event: RawFetchEvent) -> None:
            received.append(event)

        async with BusTestHarness() as harness:
            await harness.subscribe("raw_fetch", "test.group", handler, RawFetchEvent)
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)

            event = _raw_fetch(url="https://dispatched.example.com")
            await emitter.emit(event)

            await asyncio.sleep(0.15)  # allow dispatch loop to run

        assert len(received) == 1
        assert received[0].url == "https://dispatched.example.com"

    @pytest.mark.asyncio
    async def test_dispatch_count_increments(self):
        async def handler(event: RawFetchEvent) -> None:
            pass

        async with BusTestHarness() as harness:
            await harness.subscribe("raw_fetch", "test.group", handler, RawFetchEvent)
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            await emitter.emit(_raw_fetch())
            await asyncio.sleep(0.15)
            assert harness.dispatch_count("raw_fetch") >= 1

    @pytest.mark.asyncio
    async def test_multiple_subscribers_both_receive(self):
        counts_a: List[int] = []
        counts_b: List[int] = []

        async def handler_a(event: RawFetchEvent) -> None:
            counts_a.append(1)

        async def handler_b(event: RawFetchEvent) -> None:
            counts_b.append(1)

        async with BusTestHarness() as harness:
            await harness.subscribe("raw_fetch", "group.a", handler_a, RawFetchEvent)
            await harness.subscribe("raw_fetch", "group.b", handler_b, RawFetchEvent)
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            await emitter.emit(_raw_fetch())
            await asyncio.sleep(0.2)

        assert len(counts_a) == 1
        assert len(counts_b) == 1

    @pytest.mark.asyncio
    async def test_handler_failure_increments_dead_letter_count(self, tmp_path):
        dead_letter_path = tmp_path / "dead_letters.jsonl"

        async def bad_handler(event: RawFetchEvent) -> None:
            raise ValueError("intentional failure for test")

        with patch.object(cb, "_DEAD_LETTER_PATH", dead_letter_path):
            async with BusTestHarness() as harness:
                await harness.subscribe("raw_fetch", "fail.group", bad_handler, RawFetchEvent)
                emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
                await emitter.emit(_raw_fetch())
                await asyncio.sleep(0.5)  # allow all retries + dead letter write

            assert harness.dead_letter_count("raw_fetch") >= 1

    @pytest.mark.asyncio
    async def test_handler_isolation_one_bad_handler_doesnt_affect_other(self):
        good_received: List[Any] = []

        async def bad_handler(event: RawFetchEvent) -> None:
            raise RuntimeError("this handler always fails")

        async def good_handler(event: RawFetchEvent) -> None:
            good_received.append(event)

        with patch.object(cb, "_DEAD_LETTER_PATH", Path("/tmp/test_isolation_dl.jsonl")):
            async with BusTestHarness() as harness:
                await harness.subscribe("raw_fetch", "bad.group",  bad_handler,  RawFetchEvent)
                await harness.subscribe("raw_fetch", "good.group", good_handler, RawFetchEvent)
                emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
                await emitter.emit(_raw_fetch())
                await asyncio.sleep(0.5)

        assert len(good_received) == 1, "good handler must receive despite bad handler failing"

    @pytest.mark.asyncio
    async def test_duplicate_group_replaces_handler(self):
        calls_v1: List[Any] = []
        calls_v2: List[Any] = []

        async def handler_v1(event: RawFetchEvent) -> None:
            calls_v1.append(event)

        async def handler_v2(event: RawFetchEvent) -> None:
            calls_v2.append(event)

        async with BusTestHarness() as harness:
            await harness.subscribe("raw_fetch", "same.group", handler_v1, RawFetchEvent)
            await harness.subscribe("raw_fetch", "same.group", handler_v2, RawFetchEvent)
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            await emitter.emit(_raw_fetch())
            await asyncio.sleep(0.15)

        # v2 replaced v1 — only v2 should have received
        assert len(calls_v2) == 1
        assert len(calls_v1) == 0

    @pytest.mark.asyncio
    async def test_emitter_disabled_after_stop(self):
        async with BusTestHarness() as harness:
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)

        # After __aexit__, emitter should be disabled
        with pytest.raises(RuntimeError):
            await emitter.emit(_raw_fetch())

    @pytest.mark.asyncio
    async def test_multiple_events_all_dispatched(self):
        received: List[Any] = []

        async def handler(event: RawFetchEvent) -> None:
            received.append(event)

        async with BusTestHarness() as harness:
            await harness.subscribe("raw_fetch", "test.group", handler, RawFetchEvent)
            emitter = await harness.emitter("raw_fetch", "test", RawFetchEvent)
            for i in range(5):
                await emitter.emit(_raw_fetch(url=f"https://example.com/{i}"))
            await asyncio.sleep(0.3)

        assert len(received) == 5
        urls = {e.url for e in received}
        assert len(urls) == 5

    @pytest.mark.asyncio
    async def test_correct_event_type_received_by_handler(self):
        received: List[Any] = []

        async def handler(event: CleanSignalEvent) -> None:
            received.append(event)

        async with BusTestHarness() as harness:
            await harness.subscribe("clean_signal", "grp", handler, CleanSignalEvent)
            emitter = await harness.emitter("clean_signal", "test", CleanSignalEvent)
            original = _clean_signal("TEST_CLASS")
            await emitter.emit(original)
            await asyncio.sleep(0.15)

        assert len(received) == 1
        assert isinstance(received[0], CleanSignalEvent)
        assert received[0].topology_class == "TEST_CLASS"

    @pytest.mark.asyncio
    async def test_bus_health_uptime_positive(self):
        async with BusTestHarness() as harness:
            await asyncio.sleep(0.05)
            h = harness.health()
            assert h.uptime_s >= 0.0


# ═════════════════════════════════════════════════════════════════════════════
# 23. CrawlerBus — pre-start guards
# ═════════════════════════════════════════════════════════════════════════════

class TestCrawlerBusPreStartGuards:
    @pytest.mark.asyncio
    async def test_emitter_before_start_raises(self):
        bus = CrawlerBus()
        with pytest.raises(RuntimeError, match="not started"):
            await bus.emitter("raw_fetch", "test", RawFetchEvent)

    @pytest.mark.asyncio
    async def test_subscribe_before_start_raises(self):
        bus = CrawlerBus()
        async def h(e): pass
        with pytest.raises(RuntimeError, match="not started"):
            await bus.subscribe("raw_fetch", "grp", h, RawFetchEvent)


# ═════════════════════════════════════════════════════════════════════════════
# 24. BusStartupDiagnostic
# ═════════════════════════════════════════════════════════════════════════════

class TestBusStartupDiagnostic:
    @pytest.mark.asyncio
    async def test_diagnostic_passes_in_degraded_mode(self, tmp_path):
        dead_letter_path = tmp_path / "dead_letters.jsonl"
        with patch.object(cb, "_DEAD_LETTER_PATH", dead_letter_path):
            async with BusTestHarness() as harness:
                diag = BusStartupDiagnostic(harness._bus)
                result = await diag.run()

        assert result.hmac_sign_verify is True
        assert result.serialization_ok is True
        assert result.mode_detected == "degraded"

    @pytest.mark.asyncio
    async def test_diagnostic_result_is_dataclass(self, tmp_path):
        dead_letter_path = tmp_path / "dead_letters.jsonl"
        with patch.object(cb, "_DEAD_LETTER_PATH", dead_letter_path):
            async with BusTestHarness() as harness:
                diag = BusStartupDiagnostic(harness._bus)
                result = await diag.run()

        assert isinstance(result, BusDiagnosticResult)
        assert isinstance(result.errors, list)
        assert isinstance(result.duration_ms, float)
        assert result.duration_ms >= 0.0

    @pytest.mark.asyncio
    async def test_diagnostic_to_log_dict(self, tmp_path):
        dead_letter_path = tmp_path / "dead_letters.jsonl"
        with patch.object(cb, "_DEAD_LETTER_PATH", dead_letter_path):
            async with BusTestHarness() as harness:
                diag = BusStartupDiagnostic(harness._bus)
                result = await diag.run()

        d = result.to_log_dict()
        assert "passed" in d
        assert "mode" in d
        assert "hmac_sign_verify" in d


# ═════════════════════════════════════════════════════════════════════════════
# 25. CircuitState enum
# ═════════════════════════════════════════════════════════════════════════════

class TestCircuitState:
    def test_three_states_exist(self):
        assert CircuitState.CLOSED
        assert CircuitState.OPEN
        assert CircuitState.HALF_OPEN

    def test_states_distinct(self):
        states = {CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN}
        assert len(states) == 3
