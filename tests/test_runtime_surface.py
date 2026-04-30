from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

os.environ.setdefault("AXIOM_BUS_HMAC_KEY", "1" * 64)

from tag.cold_start import ColdStart
from signal_kernel.contracts import FetchAnomalyEvent, FetchMode, RawFetchEvent, SignalExtractedEvent, SurpriseEvent, new_run_id
from tag.index_daemon import GRADIENT_PRIORITY_HIGH, IndexDaemon, run_once_for_test
from tag.interface import AxiomInterface, AxiomRuntimeContext, InterfaceSocketServer, JsonLineCodec, QueryOrchestrator, SearchDocument, parse_command


class RuntimeSurfaceTests(unittest.TestCase):
    def test_parse_command(self) -> None:
        parsed = parse_command("search | hello")
        self.assertEqual(parsed.command, "SEARCH")
        self.assertEqual(parsed.payload, "hello")
        prompt = parse_command("axiom")
        self.assertEqual(prompt.command, "STATUS")
        natural = parse_command("latest AI news")
        self.assertEqual(natural.command, "SEARCH")
        self.assertEqual(natural.payload, "latest AI news")

    def test_cold_start_creates_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = ColdStart(store_dir=Path(td)).run()
            self.assertTrue(result.ok)
            self.assertTrue((Path(td) / "phase_states.mmap").exists())
            self.assertEqual(len(result.store_status), 4)
            self.assertTrue(all(item["ok"] for item in result.store_status))

    def test_index_daemon_run_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            status = asyncio.run(run_once_for_test(Path(td)))
            self.assertEqual(status["stats"]["signal_events"], 1)

    def test_index_daemon_dispatch_drafts_recipe_and_prioritizes_gradients(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                daemon = IndexDaemon(store_dir=Path(td))
                try:
                    await daemon.dispatch(
                        SignalExtractedEvent(
                            url="https://example.com/docs",
                            topology_class="SAAS_DOCS",
                            signal_type="code",
                            byte_count=1000,
                            token_count=100,
                            signal_density=0.92,
                            zone_count=3,
                            source_component="test",
                            run_id=str(new_run_id()),
                        )
                    )
                    self.assertEqual(daemon.stats.drafted_recipes, 1)
                    self.assertTrue((Path(td) / "recipe_registry.mmap").exists())
                    await daemon.dispatch(
                        SurpriseEvent(
                            topology_class="SAAS_DOCS",
                            surprise_score=0.9,
                            theta_surprise=0.5,
                            dissolve_triggered=True,
                            contributing_signals={"density": 0.9},
                            current_phase=2,
                            run_id=str(new_run_id()),
                            timestamp="2026-04-28T00:00:00Z",
                        )
                    )
                    batch = daemon.pop_gradient_batch()
                    self.assertEqual(batch[0]["priority"], GRADIENT_PRIORITY_HIGH)
                finally:
                    daemon.close()

        asyncio.run(run())

    def test_index_daemon_fetch_anomaly_dispatch(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                daemon = IndexDaemon(store_dir=Path(td))
                try:
                    await daemon.dispatch(
                        FetchAnomalyEvent(
                            url="https://example.com",
                            fetch_mode=FetchMode.STATIC,
                            status_code=429,
                            anomaly_type="rate_limited",
                            run_id=str(new_run_id()),
                            manifest_id="manifest",
                        )
                    )
                    self.assertEqual(daemon.stats.fetch_anomalies, 1)
                    self.assertEqual(daemon.pop_gradient_batch()[0]["priority"], GRADIENT_PRIORITY_HIGH)
                finally:
                    daemon.close()

        asyncio.run(run())

    def test_interface_flow(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                interface = AxiomInterface(store_dir=Path(td))
                try:
                    learn = await interface.handle_line("learn | example.com")
                    self.assertEqual(learn.status, "accepted")
                    search = await interface.handle_line("search | docs")
                    self.assertEqual(search.status, "ok")
                    status = await interface.handle_line("status |")
                    self.assertEqual(status.status, "ok")
                    self.assertGreaterEqual(status.data["metrics"]["handled"], 2)
                    self.assertTrue(status.data["recent"])
                finally:
                    await interface.runtime.close()

        asyncio.run(run())

    def test_interface_json_request_and_validation(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                interface = AxiomInterface(store_dir=Path(td))
                try:
                    bad = await interface.handle_line("fetch | ftp://example.com/file")
                    self.assertEqual(bad.status, "error")
                    learn = await interface.handle_json({"query_type": "LEARN", "payload": "https://docs.example.com/path"})
                    self.assertEqual(learn.status, "accepted")
                    search = await interface.handle_json({"command": "search", "payload": "docs"})
                    self.assertEqual(search.status, "ok")
                finally:
                    await interface.runtime.close()

        asyncio.run(run())

    def test_interface_search_returns_ranked_blocks(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                runtime.learned_domains.add("wikipedia.org")

                class FakeFetcher:
                    async def fetch_single(self, url: str, cl_level: int = 1, topology_hint: str = "GENERIC_HTML") -> RawFetchEvent:
                        self.last_url = url
                        return RawFetchEvent(
                            url=url,
                            raw_bytes=(
                                b"<html><head><title>Car - Wikipedia</title></head>"
                                b"<body><p>A car is a wheeled motor vehicle used for transportation.</p>"
                                b"<p>Cars usually have four wheels and primarily transport people.</p></body></html>"
                            ),
                            status_code=200,
                            headers={"content-type": "text/html; charset=utf-8"},
                            fetch_latency=0.05,
                            fetch_mode=FetchMode.STATIC,
                            is_robots_txt=False,
                            is_sitemap=False,
                            topology_hint=topology_hint,
                            run_id=str(new_run_id()),
                            manifest_id=str(new_run_id()),
                            byte_count=196,
                        )

                    async def shutdown(self) -> None:
                        return None

                class FakeSanitizer:
                    def process(self, raw: bytes) -> SimpleNamespace:
                        return SimpleNamespace(
                            ok=True,
                            text=(
                                "A car is a wheeled motor vehicle used for transportation.\n\n"
                                "Cars usually have four wheels and primarily transport people."
                            ),
                            events=[],
                            metrics=SimpleNamespace(),
                        )

                async def fake_ensure_crawl_stack() -> None:
                    return None

                runtime.fetcher = FakeFetcher()
                runtime.sanitizer = FakeSanitizer()
                runtime.classifier = False
                runtime.ensure_crawl_stack = fake_ensure_crawl_stack  # type: ignore[assignment]

                interface = AxiomInterface(store_dir=Path(td), runtime=runtime)
                with mock.patch.object(
                    QueryOrchestrator,
                    "_run_kernel",
                    new=mock.AsyncMock(
                        return_value=(
                            "A car is a wheeled motor vehicle used for transportation. "
                            "Cars usually have four wheels and primarily transport people."
                        )
                    ),
                ):
                    search = await interface.handle_line("search | what is a car")

                self.assertEqual(search.status, "ok")
                self.assertFalse(search.data["search_engine"])
                self.assertTrue(search.data["blocks"])
                self.assertIn("wheeled motor vehicle", search.data["blocks"][0]["text"].lower())
                self.assertTrue(any("/wiki/Car" in source["url"] for source in search.data["sources"]))
                source_domains = {source["domain"] for source in search.data["sources"]}
                self.assertIn("wikipedia.org", source_domains)
                self.assertGreater(len(source_domains), 1)
                self.assertTrue(any(source["seeded"] for source in search.data["sources"]))

        asyncio.run(run())

    def test_candidate_sources_are_open_web_not_wikipedia_limited(self) -> None:
        runtime = AxiomRuntimeContext(store_dir=Path("/tmp/axiom-open-web-test"))
        runtime.learned_domains.update({"docs.python.org", "example.com", "ietf.org"})
        sources = QueryOrchestrator(runtime)._candidate_sources("async http standard")
        domains = {source["domain"] for source in sources}
        self.assertIn("docs.python.org", domains)
        self.assertIn("example.com", domains)
        self.assertIn("ietf.org", domains)
        self.assertNotEqual(domains, {"wikipedia.org"})
        self.assertGreater(len(domains), 8)
        self.assertLess(
            sum(1 for source in sources if source["domain"] == "wikipedia.org"),
            len(sources),
        )

    def test_ranker_prefers_subject_definition_over_search_listing(self) -> None:
        runtime = AxiomRuntimeContext(store_dir=Path("/tmp/axiom-definition-rank-test"))
        orchestrator = QueryOrchestrator(runtime)
        docs = [
            SearchDocument(
                url="https://search.example/?q=what+is+google",
                domain="search.example",
                title="Search results for what is google",
                topology_class="GENERIC_HTML",
                classification_confidence=0.0,
                fetch_mode="static",
                status_code=200,
                clean_text="Google Docs is a word processor offered by Google.",
                kernel_signal="Google Docs is a word processor offered by Google.",
                blocks=["Google Docs is a word processor offered by Google."],
                fetched_unix=1,
            ),
            SearchDocument(
                url="https://source.example/google",
                domain="source.example",
                title="Google",
                topology_class="GENERIC_HTML",
                classification_confidence=0.0,
                fetch_mode="static",
                status_code=200,
                clean_text="Google is an American multinational technology company focused on online services.",
                kernel_signal="Google is an American multinational technology company focused on online services.",
                blocks=["Google is an American multinational technology company focused on online services."],
                fetched_unix=2,
            ),
        ]
        ranked = orchestrator._rank_documents("what is google", docs)
        self.assertEqual(ranked[0]["url"], "https://source.example/google")
        answer = orchestrator._build_answer("what is google", ranked)
        self.assertIsNotNone(answer)
        assert answer is not None
        self.assertIn("Google is an American", answer["text"])

    def test_interactive_fetch_bypasses_persistent_dedupe(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                dedupe_values: list[bool] = []

                class FakeFetcher:
                    cl_state = SimpleNamespace(cl1_available=True, cl2_available=False, cl3_available=False, cl4_available=False)

                    async def fetch_single(
                        self,
                        url: str,
                        cl_level: int = 1,
                        topology_hint: str = "GENERIC_HTML",
                        *,
                        dedupe: bool = True,
                    ) -> RawFetchEvent:
                        dedupe_values.append(dedupe)
                        return RawFetchEvent(
                            url=url,
                            raw_bytes=b"<html><head><title>Google</title></head><body><p>Google is a search company.</p></body></html>",
                            status_code=200,
                            headers={"content-type": "text/html; charset=utf-8"},
                            fetch_latency=0.01,
                            fetch_mode=FetchMode.STATIC,
                            is_robots_txt=False,
                            is_sitemap=False,
                            topology_hint=topology_hint,
                            run_id=str(new_run_id()),
                            manifest_id=str(new_run_id()),
                            byte_count=94,
                        )

                class FakeSanitizer:
                    def process(self, raw: bytes) -> SimpleNamespace:
                        return SimpleNamespace(ok=True, text="Google is a search company.", events=[], metrics=SimpleNamespace())

                async def fake_ensure_crawl_stack() -> None:
                    return None

                runtime.fetcher = FakeFetcher()
                runtime.sanitizer = FakeSanitizer()
                runtime.classifier = False
                runtime.ensure_crawl_stack = fake_ensure_crawl_stack  # type: ignore[assignment]

                document = await QueryOrchestrator(runtime)._fetch_document("https://example.com/google", str(new_run_id()), reason="test")
                self.assertIsNotNone(document)
                self.assertEqual(dedupe_values, [False])

        asyncio.run(run())

    def test_interface_dev_mode_clearance_escalation(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                runtime.learned_domains.add("wikipedia.org")
                attempted_levels: list[int] = []

                class FakeFetcher:
                    cl_state = SimpleNamespace(
                        cl1_available=True,
                        cl2_available=True,
                        cl3_available=False,
                        cl4_available=False,
                    )

                    async def fetch_single(self, url: str, cl_level: int = 1, topology_hint: str = "GENERIC_HTML") -> RawFetchEvent | None:
                        attempted_levels.append(cl_level)
                        if cl_level == 1:
                            return None
                        return RawFetchEvent(
                            url=url,
                            raw_bytes=(
                                b"<html><head><title>Car - Wikipedia</title></head>"
                                b"<body><p>A car is a wheeled motor vehicle used for transportation.</p></body></html>"
                            ),
                            status_code=200,
                            headers={"content-type": "text/html; charset=utf-8"},
                            fetch_latency=0.05,
                            fetch_mode=FetchMode.HEADLESS,
                            is_robots_txt=False,
                            is_sitemap=False,
                            topology_hint=topology_hint,
                            run_id=str(new_run_id()),
                            manifest_id=str(new_run_id()),
                            byte_count=128,
                        )

                    async def shutdown(self) -> None:
                        return None

                class FakeSanitizer:
                    def process(self, raw: bytes) -> SimpleNamespace:
                        return SimpleNamespace(
                            ok=True,
                            text="A car is a wheeled motor vehicle used for transportation.",
                            events=[],
                            metrics=SimpleNamespace(),
                        )

                async def fake_ensure_crawl_stack() -> None:
                    return None

                runtime.fetcher = FakeFetcher()
                runtime.sanitizer = FakeSanitizer()
                runtime.classifier = False
                runtime.ensure_crawl_stack = fake_ensure_crawl_stack  # type: ignore[assignment]

                interface = AxiomInterface(store_dir=Path(td), runtime=runtime)
                with (
                    mock.patch.dict(os.environ, {"AXIOM_ENV": "dev", "AXIOM_CLEARANCE_POLICY": "dev"}, clear=False),
                    mock.patch.object(
                        QueryOrchestrator,
                        "_run_kernel",
                        new=mock.AsyncMock(return_value="A car is a wheeled motor vehicle used for transportation."),
                    ),
                ):
                    search = await interface.handle_line("search | what is a car")

                self.assertEqual(search.status, "ok")
                self.assertEqual(attempted_levels[:2], [1, 2])
                self.assertGreaterEqual(len(attempted_levels), 2)
                self.assertEqual(search.data["blocks"][0]["fetch_mode"], "headless")

        asyncio.run(run())

    def test_interface_deep_clearance_policy_without_dev_mode(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                attempted_levels: list[int] = []

                class FakeFetcher:
                    cl_state = SimpleNamespace(
                        cl1_available=True,
                        cl2_available=True,
                        cl3_available=False,
                        cl4_available=False,
                    )

                    async def fetch_single(self, url: str, cl_level: int = 1, topology_hint: str = "GENERIC_HTML") -> RawFetchEvent | None:
                        attempted_levels.append(cl_level)
                        if cl_level == 1:
                            return None
                        return RawFetchEvent(
                            url=url,
                            raw_bytes=b"<html><body><p>Deep clearance fetch succeeded.</p></body></html>",
                            status_code=200,
                            headers={"content-type": "text/html; charset=utf-8"},
                            fetch_latency=0.05,
                            fetch_mode=FetchMode.HEADLESS,
                            is_robots_txt=False,
                            is_sitemap=False,
                            topology_hint=topology_hint,
                            run_id=str(new_run_id()),
                            manifest_id=str(new_run_id()),
                            byte_count=64,
                        )

                    async def shutdown(self) -> None:
                        return None

                class FakeSanitizer:
                    def process(self, raw: bytes) -> SimpleNamespace:
                        return SimpleNamespace(
                            ok=True,
                            text="Deep clearance fetch succeeded.",
                            events=[],
                            metrics=SimpleNamespace(),
                        )

                async def fake_ensure_crawl_stack() -> None:
                    return None

                runtime.fetcher = FakeFetcher()
                runtime.sanitizer = FakeSanitizer()
                runtime.classifier = False
                runtime.ensure_crawl_stack = fake_ensure_crawl_stack  # type: ignore[assignment]

                interface = AxiomInterface(store_dir=Path(td), runtime=runtime)
                with (
                    mock.patch.dict(os.environ, {"AXIOM_ENV": "", "AXIOM_CLEARANCE_POLICY": "deep"}, clear=False),
                    mock.patch.object(
                        QueryOrchestrator,
                        "_run_kernel",
                        new=mock.AsyncMock(return_value="Deep clearance fetch succeeded."),
                    ),
                ):
                    search = await interface.handle_line("search | swarm -2 | depth -1 | deep clearance")

                self.assertEqual(search.status, "ok")
                self.assertEqual(attempted_levels[:2], [1, 2])

        asyncio.run(run())

    def test_json_line_codec_plain_and_json(self) -> None:
        self.assertEqual(JsonLineCodec.decode_line(b"status |\n"), "status |")
        self.assertEqual(JsonLineCodec.decode_line(b'{"command":"learn","payload":"example.com"}\n'), "learn | example.com")

    def test_interface_tcp_socket_server(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                server = InterfaceSocketServer(interface=AxiomInterface(store_dir=Path(td)), port=8876)
                await server.start()
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", 8876)
                    writer.write(b"status |\n")
                    await writer.drain()
                    line = await reader.readline()
                    self.assertIn(b'"status":"ok"', line)
                    writer.close()
                    await writer.wait_closed()
                finally:
                    await server.stop()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
