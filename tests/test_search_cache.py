from __future__ import annotations

import asyncio
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from signal_kernel.contracts import FetchMode, RawFetchEvent, new_run_id
from tag.interface import AxiomInterface, AxiomRuntimeContext, QueryOrchestrator


def test_expanded_search_second_hit_uses_completed_answer_cache() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = AxiomRuntimeContext(store_dir=Path(td))
            runtime.config = runtime.config
            runtime.learned_domains.add("wikipedia.org")
            fetch_count = 0

            class FakeFetcher:
                async def fetch_single(self, url: str, cl_level: int = 1, topology_hint: str = "GENERIC_HTML") -> RawFetchEvent:
                    nonlocal fetch_count
                    fetch_count += 1
                    return RawFetchEvent(
                        url=url,
                        raw_bytes=(
                            b"<html><head><title>GitHub - Wikipedia</title></head>"
                            b"<body><p>GitHub is a developer platform for storing, managing, and sharing code.</p></body></html>"
                        ),
                        status_code=200,
                        headers={"content-type": "text/html; charset=utf-8"},
                        fetch_latency=0.01,
                        fetch_mode=FetchMode.STATIC,
                        is_robots_txt=False,
                        is_sitemap=False,
                        topology_hint=topology_hint,
                        run_id=str(new_run_id()),
                        manifest_id=str(new_run_id()),
                        byte_count=160,
                    )

                async def shutdown(self) -> None:
                    return None

            class FakeSanitizer:
                def process(self, raw: bytes) -> SimpleNamespace:
                    return SimpleNamespace(
                        ok=True,
                        text="GitHub is a developer platform for storing, managing, and sharing code.",
                        events=[],
                        metrics=SimpleNamespace(),
                    )

            async def fake_ensure_crawl_stack() -> None:
                return None

            async def fake_mcp(self, query: str, run_id: str, expansion: object):  # noqa: ANN001
                return []

            runtime.fetcher = FakeFetcher()
            runtime.sanitizer = FakeSanitizer()
            runtime.classifier = False
            runtime.ensure_crawl_stack = fake_ensure_crawl_stack  # type: ignore[assignment]
            interface = AxiomInterface(store_dir=Path(td), runtime=runtime)

            with (
                mock.patch.object(QueryOrchestrator, "_run_kernel", new=mock.AsyncMock(return_value="GitHub is a developer platform for storing, managing, and sharing code.")),
                mock.patch.object(QueryOrchestrator, "_collect_mcp_anchor_documents", new=fake_mcp),
            ):
                first_start = time.perf_counter()
                first = await interface.handle_line("search | fanout -3 | depth -1 | exp -2 | what is github")
                first_elapsed = time.perf_counter() - first_start
                second_start = time.perf_counter()
                second = await interface.handle_line("search | fanout -3 | depth -1 | exp -2 | what is github")
                second_elapsed = time.perf_counter() - second_start

            assert first.status == "ok"
            assert second.status == "ok"
            assert first.data["answer"]["text"] == second.data["answer"]["text"]
            assert first.data["answer"]["structured"]["summary"] == second.data["answer"]["structured"]["summary"]
            assert first.data["cache"]["hit"] is False
            assert second.data["cache"]["hit"] is True
            assert fetch_count > 0
            assert second_elapsed < first_elapsed

    asyncio.run(run())


def test_recheck_bypasses_completed_answer_cache() -> None:
    async def run() -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = AxiomRuntimeContext(store_dir=Path(td))
            runtime.learned_domains.add("wikipedia.org")
            fetch_count = 0

            class FakeFetcher:
                async def fetch_single(self, url: str, cl_level: int = 1, topology_hint: str = "GENERIC_HTML") -> RawFetchEvent:
                    nonlocal fetch_count
                    fetch_count += 1
                    return RawFetchEvent(
                        url=url,
                        raw_bytes=b"<html><head><title>GitHub</title></head><body><p>GitHub is a developer platform for code.</p></body></html>",
                        status_code=200,
                        headers={"content-type": "text/html; charset=utf-8"},
                        fetch_latency=0.01,
                        fetch_mode=FetchMode.STATIC,
                        is_robots_txt=False,
                        is_sitemap=False,
                        topology_hint=topology_hint,
                        run_id=str(new_run_id()),
                        manifest_id=str(new_run_id()),
                        byte_count=120,
                    )

                async def shutdown(self) -> None:
                    return None

            class FakeSanitizer:
                def process(self, raw: bytes) -> SimpleNamespace:
                    return SimpleNamespace(ok=True, text="GitHub is a developer platform for code.", events=[], metrics=SimpleNamespace())

            async def fake_ensure_crawl_stack() -> None:
                return None

            async def fake_mcp(self, query: str, run_id: str, expansion: object):  # noqa: ANN001
                return []

            runtime.fetcher = FakeFetcher()
            runtime.sanitizer = FakeSanitizer()
            runtime.classifier = False
            runtime.ensure_crawl_stack = fake_ensure_crawl_stack  # type: ignore[assignment]
            interface = AxiomInterface(store_dir=Path(td), runtime=runtime)

            with (
                mock.patch.object(QueryOrchestrator, "_run_kernel", new=mock.AsyncMock(return_value="GitHub is a developer platform for code.")),
                mock.patch.object(QueryOrchestrator, "_collect_mcp_anchor_documents", new=fake_mcp),
            ):
                first = await interface.handle_line("search | fanout -3 | depth -1 | exp -2 | what is github")
                second = await interface.handle_line("search | fanout -3 | depth -1 | exp -2 | recheck | what is github")

            assert first.data["cache"]["hit"] is False
            assert second.data["cache"]["hit"] is False
            assert fetch_count > 1

    asyncio.run(run())
