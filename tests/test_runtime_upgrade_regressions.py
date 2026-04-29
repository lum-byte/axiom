from __future__ import annotations

import asyncio
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from signal_kernel.contracts import FetchMode, RawFetchEvent, new_run_id
from tag.cold_start import ColdStart
from tag.crawler.swarm import AxiomCrawlSwarmConfig
from tag.index_daemon import GRADIENT_PRIORITY_HIGH, IndexDaemon
from tag.interface import AxiomRuntimeContext, QueryOrchestrator, SearchDocument


class RuntimeUpgradeRegressionTests(unittest.TestCase):
    def test_cold_start_dev_generates_hmac_and_validates_binary_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"AXIOM_ENV": "dev", "PATH": os.environ.get("PATH", "")}, clear=True):
                result = ColdStart(store_dir=Path(td)).run()
                self.assertIn("AXIOM_BUS_HMAC_KEY", os.environ)

            self.assertTrue(result.ok)
            by_name = {item["name"]: item for item in result.store_status}
            self.assertTrue(by_name["phase_states.mmap"]["binary_format_valid"])
            self.assertIn("AXPS", by_name["phase_states.mmap"]["format_detail"])
            self.assertTrue(by_name["recipe_registry.mmap"]["binary_format_valid"])
            self.assertIn("AXRR", by_name["recipe_registry.mmap"]["format_detail"])

    def test_cold_start_production_requires_real_hmac(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.dict(os.environ, {"AXIOM_ENV": "production", "PATH": os.environ.get("PATH", "")}, clear=True):
                result = ColdStart(store_dir=Path(td)).run()
        self.assertFalse(result.ok)
        self.assertTrue(any("AXIOM_BUS_HMAC_KEY" in error for error in result.errors))

    def test_index_daemon_dispatches_gradient_batch_to_offline_queue(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                daemon = IndexDaemon(store_dir=Path(td))
                try:
                    await daemon._queue_gradient(
                        {"type": "unit", "topology_class": "NEWS_ARTICLE"},
                        priority=GRADIENT_PRIORITY_HIGH,
                    )
                    dispatched = await daemon._dispatch_gradient_batch()
                    self.assertEqual(dispatched, 1)
                    files = list((Path(td) / "offline_queue").glob("gradient-*.jsonl"))
                    self.assertEqual(len(files), 1)
                    self.assertIn('"type":"unit"', files[0].read_text(encoding="utf-8"))
                finally:
                    daemon.close()

        asyncio.run(run())

    def test_index_daemon_background_tasks_start_and_stop(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                daemon = IndexDaemon(store_dir=Path(td))
                try:
                    await daemon.start_background_tasks()
                    self.assertTrue(daemon.status()["running"])
                    self.assertTrue(daemon._tasks)
                    await daemon.stop()
                    self.assertFalse(daemon.status()["running"])
                    self.assertEqual(daemon._tasks, [])
                finally:
                    daemon.close()

        asyncio.run(run())

    def test_interface_collect_documents_uses_parallel_gather_batch(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                orchestrator = QueryOrchestrator(runtime)
                active = 0
                max_active = 0

                async def fake_fetch_document(self: QueryOrchestrator, url: str, run_id: str, *, reason: str) -> SearchDocument:
                    nonlocal active, max_active
                    active += 1
                    max_active = max(max_active, active)
                    await asyncio.sleep(0.05)
                    active -= 1
                    return SearchDocument(
                        url=url,
                        domain=url.split("//", 1)[1].split("/", 1)[0],
                        title=url,
                        topology_class="GENERIC_HTML",
                        classification_confidence=0.5,
                        fetch_mode=str(FetchMode.STATIC.value),
                        status_code=200,
                        clean_text="parallel fetch document body",
                        kernel_signal="parallel fetch document body",
                        blocks=["parallel fetch document body"],
                        fetched_unix=int(time.time()),
                    )

                candidates = [
                    {"url": f"https://site{idx}.example/page", "domain": f"site{idx}.example", "reason": "test"}
                    for idx in range(4)
                ]
                config = AxiomCrawlSwarmConfig(worker_count=4, target_documents=4, max_waves=1)
                with mock.patch.object(QueryOrchestrator, "_fetch_document", new=fake_fetch_document):
                    documents = await orchestrator._collect_documents("parallel", str(new_run_id()), candidates, swarm_config=config)
                self.assertEqual(len(documents), 4)
                self.assertGreater(max_active, 1)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
