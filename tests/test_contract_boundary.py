from __future__ import annotations

import os
import sys
import unittest
import uuid

# test_crawler_bus.py intentionally installs lightweight signal_kernel stubs for
# its standalone bus tests. If this module is collected after that file, restore
# the real package before importing contracts/exceptions.
for _module_name in ("signal_kernel.contracts", "signal_kernel.exceptions", "signal_kernel"):
    _module = sys.modules.get(_module_name)
    if _module is not None and getattr(_module, "__file__", None) is None:
        sys.modules.pop(_module_name, None)

from signal_kernel.contracts import (
    FetchAnomalyEvent,
    FetchMode,
    InterfaceRequest,
    RecipeHealthEvent,
    SignalExtractedEvent,
    StoreHealthEvent,
    WeightsUpdatedEvent,
)
from signal_kernel.exceptions import (
    ColdStartSecurityFailed,
    DomainAnalysisFailed,
    EC_COLD_START_SECURITY,
    EC_PREPARSER_DOMAIN_ANALYSIS,
    exception_from_code,
)


SHA = "0" * 64


def run_id() -> str:
    return str(uuid.uuid4())


class ContractBoundaryTests(unittest.TestCase):
    def test_fetch_anomaly_accepts_string_fetch_mode(self) -> None:
        event = FetchAnomalyEvent(
            url="https://example.com",
            fetch_mode="tor",
            status_code=None,
            anomaly_type="tor_unavailable",
            run_id=run_id(),
            manifest_id=run_id(),
        )
        self.assertIs(event.fetch_mode, FetchMode.TOR)

    def test_signal_extracted_contract_validates_topology_and_density(self) -> None:
        event = SignalExtractedEvent(
            url="https://example.com/a",
            topology_class="NEWS_ARTICLE",
            signal_type="prose",
            byte_count=120,
            token_count=20,
            signal_density=0.75,
            zone_count=2,
            source_component="preparser.signal_extractor",
            run_id=run_id(),
        )
        self.assertEqual(event.topology_class, "NEWS_ARTICLE")

    def test_recipe_health_contract(self) -> None:
        event = RecipeHealthEvent(
            topology_class="SAAS_DOCS",
            recipe_hash=SHA,
            sample_count=10,
            success_count=8,
            failure_count=2,
            empty_rate=0.1,
            median_latency_ms=12.5,
            stale=False,
            run_id=run_id(),
        )
        self.assertFalse(event.stale)

    def test_weight_and_store_events_validate_sha(self) -> None:
        weight_event = WeightsUpdatedEvent(
            model_name="structural_layer",
            store_path="/store/structural_layer.pt",
            staging_path="/store/staging/structural_layer.pt.staging",
            checksum_sha256=SHA,
            version=1,
            batch_count=4,
            gradient_steps=16,
            run_id=run_id(),
        )
        store_event = StoreHealthEvent(
            store_file="/store/structural_layer.pt",
            status="ok",
            size_bytes=1024,
            checksum_sha256=SHA,
            critical=False,
            detail="crc ok",
            run_id=run_id(),
        )
        self.assertEqual(weight_event.checksum_sha256, store_event.checksum_sha256)

    def test_interface_request_rejects_empty_search(self) -> None:
        with self.assertRaises(ValueError):
            InterfaceRequest(query_type="SEARCH", payload="", run_id=run_id())

    def test_cross_language_exception_registry(self) -> None:
        self.assertIs(exception_from_code(EC_PREPARSER_DOMAIN_ANALYSIS), DomainAnalysisFailed)
        self.assertIs(exception_from_code(EC_COLD_START_SECURITY), ColdStartSecurityFailed)


class BusCoercionTests(unittest.TestCase):
    def setUp(self) -> None:
        os.environ.setdefault("AXIOM_BUS_HMAC_KEY", "1" * 64)

    def test_bus_payload_builder_coerces_enum(self) -> None:
        try:
            from tag.crawler_bus import event_from_payload
        except Exception as exc:  # pragma: no cover - dependency-gated
            raise unittest.SkipTest(f"crawler_bus dependencies unavailable: {exc}") from exc

        event = event_from_payload(
            "fetch_anomaly",
            {
                "url": "https://example.com",
                "fetch_mode": "headless",
                "status_code": 500,
                "anomaly_type": "server_error",
                "run_id": run_id(),
                "manifest_id": run_id(),
                "detail": "unit test",
            },
        )
        self.assertEqual(event.fetch_mode.value, FetchMode.HEADLESS.value)


if __name__ == "__main__":
    unittest.main()
