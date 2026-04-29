from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from tag.crawler.swarm import AxiomCrawlSwarm, AxiomCrawlSwarmConfig
from tag.crawler.swarm_bridge import crawl_config_from_plan, parse_swarm_search_payload, plan_from_generic_talk
from tag.interface import AxiomRuntimeContext, QueryOrchestrator, SearchDocument


class CrawlSwarmTests(unittest.TestCase):
    def test_swarm_fetches_distinct_sites_concurrently(self) -> None:
        async def run() -> None:
            active_domains: set[str] = set()
            duplicate_active = False
            max_active = 0

            async def fetch(candidate: dict[str, object]) -> object:
                nonlocal duplicate_active, max_active
                domain = str(candidate["domain"])
                if domain in active_domains:
                    duplicate_active = True
                active_domains.add(domain)
                max_active = max(max_active, len(active_domains))
                await asyncio.sleep(0.02)
                active_domains.remove(domain)
                return candidate

            swarm = AxiomCrawlSwarm(
                fetch_document=fetch,
                rank_documents=lambda docs: [{"score": 0.0}] if docs else [],
                config=AxiomCrawlSwarmConfig(worker_count=8, target_documents=20, max_waves=1),
            )
            candidates = [
                {"url": "https://a.example/one", "domain": "a.example"},
                {"url": "https://a.example/two", "domain": "a.example"},
                {"url": "https://b.example/", "domain": "b.example"},
                {"url": "https://c.example/", "domain": "c.example"},
                {"url": "https://d.example/", "domain": "d.example"},
            ]
            result = await swarm.collect(candidates)
            self.assertFalse(duplicate_active)
            self.assertGreater(max_active, 1)
            self.assertEqual(set(result.attempted_domains), {"a.example", "b.example", "c.example", "d.example"})
            self.assertEqual(result.skipped_duplicate_sites, 1)

        asyncio.run(run())

    def test_query_orchestrator_collects_documents_in_parallel(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                orchestrator = QueryOrchestrator(runtime)
                active = 0
                max_active = 0
                attempted: list[str] = []

                async def fake_fetch_document(url: str, run_id: str, *, reason: str) -> SearchDocument:
                    nonlocal active, max_active
                    active += 1
                    max_active = max(max_active, active)
                    attempted.append(url)
                    await asyncio.sleep(0.02)
                    active -= 1
                    domain = runtime_domain(url)
                    return SearchDocument(
                        url=url,
                        domain=domain,
                        title=domain,
                        topology_class="GENERIC_HTML",
                        classification_confidence=0.2,
                        fetch_mode="static",
                        status_code=200,
                        clean_text="async http standard text",
                        kernel_signal="async http standard text",
                        blocks=["async http standard text"],
                        fetched_unix=int(time.time()),
                    )

                orchestrator._fetch_document = fake_fetch_document  # type: ignore[method-assign]
                candidates = [
                    {"url": f"https://site{i}.example/search?q=async", "domain": f"site{i}.example", "reason": "test"}
                    for i in range(12)
                ]
                documents = await orchestrator._collect_documents("async http standard", "run-id", candidates)
                self.assertGreater(max_active, 1)
                self.assertEqual(len({doc.domain for doc in documents}), len(documents))
                self.assertGreaterEqual(len(attempted), 8)
                self.assertTrue(any(item["type"] == "crawl_swarm_complete" for item in runtime.queued_work))

        asyncio.run(run())

    def test_query_orchestrator_expands_discovered_external_links(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                runtime = AxiomRuntimeContext(store_dir=Path(td))
                orchestrator = QueryOrchestrator(runtime)

                async def fake_fetch_document(url: str, run_id: str, *, reason: str) -> SearchDocument:
                    domain = runtime_domain(url)
                    links = ["https://second.example/async-http-standard"] if domain == "first.example" else []
                    return SearchDocument(
                        url=url,
                        domain=domain,
                        title=domain,
                        topology_class="GENERIC_HTML",
                        classification_confidence=0.0,
                        fetch_mode="static",
                        status_code=200,
                        clean_text="seed text",
                        kernel_signal="seed text",
                        blocks=["seed text"],
                        fetched_unix=int(time.time()),
                        links=links,
                    )

                orchestrator._fetch_document = fake_fetch_document  # type: ignore[method-assign]
                candidates = [{"url": "https://first.example/", "domain": "first.example", "reason": "test"}]
                documents = await orchestrator._collect_documents("async http standard", "run-id", candidates)
                self.assertIn("first.example", {doc.domain for doc in documents})
                self.assertIn("second.example", {doc.domain for doc in documents})

        asyncio.run(run())

    def test_swarm_command_builds_plan_and_runtime_clamps_workers(self) -> None:
        query, plan = parse_swarm_search_payload("swarm -100 | depth -2 | who were the last couple presidents of usa")
        self.assertEqual(query, "who were the last couple presidents of usa")
        self.assertIsNotNone(plan)
        assert plan is not None
        self.assertEqual(plan["requested_worker_count"], 100)
        self.assertEqual(plan["max_waves"], 2)
        self.assertEqual(plan["depth"], 2)
        self.assertIn("whitehouse.gov", plan["seed_domains"])
        self.assertIn("archives.gov", plan["seed_domains"])
        with mock.patch.dict("os.environ", {}, clear=True):
            config = crawl_config_from_plan(plan)
        self.assertEqual(config.requested_worker_count, 100)
        self.assertEqual(config.max_worker_count, 10)
        self.assertEqual(config.worker_count, 10)
        self.assertEqual(config.max_waves, 2)

    def test_swarm_worker_ceiling_is_configurable(self) -> None:
        plan = plan_from_generic_talk("search | swarm -100 | cuda mamba ssm", requested_workers=100)
        with mock.patch.dict("os.environ", {"AXIOM_CRAWL_MAX_WORKERS": "32"}, clear=True):
            config = crawl_config_from_plan(plan)
        self.assertEqual(config.requested_worker_count, 100)
        self.assertEqual(config.max_worker_count, 32)
        self.assertEqual(config.worker_count, 32)

    def test_swarm_plan_prioritizes_plan_domains_in_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            runtime = AxiomRuntimeContext(store_dir=Path(td))
            orchestrator = QueryOrchestrator(runtime)
            plan = plan_from_generic_talk(
                {
                    "kind": "queued_message",
                    "task": {"type": "in_process_teammate", "prompt": "last couple presidents of usa"},
                    "messages": [{"content": "search the whole web for the last couple presidents of usa"}],
                    "requested_workers": 100,
                }
            )
            sources = orchestrator._candidate_sources("last couple presidents of usa", crawl_plan=plan)
        self.assertEqual(sources[0]["domain"], "whitehouse.gov")
        self.assertTrue(sources[0]["reason"].startswith("swarm_bridge"))
        self.assertIn("archives.gov", {source["domain"] for source in sources[:20]})

    def test_ranked_blocks_keep_domain_diversity_when_available(self) -> None:
        runtime = AxiomRuntimeContext(store_dir=Path("/tmp/axiom-rank-diversity-test"))
        orchestrator = QueryOrchestrator(runtime)
        docs = [
            SearchDocument(
                url=f"https://a.example/{index}",
                domain="a.example",
                title="presidents",
                topology_class="GENERIC_HTML",
                classification_confidence=1.0,
                fetch_mode="static",
                status_code=200,
                clean_text="presidents usa history",
                kernel_signal="presidents usa history",
                blocks=[f"presidents usa history repeated block {index} " * 5],
                fetched_unix=int(time.time()),
            )
            for index in range(8)
        ]
        docs.append(
            SearchDocument(
                url="https://b.example/1",
                domain="b.example",
                title="presidents",
                topology_class="GENERIC_HTML",
                classification_confidence=1.0,
                fetch_mode="static",
                status_code=200,
                clean_text="presidents usa history",
                kernel_signal="presidents usa history",
                blocks=["presidents usa history from another source " * 5],
                fetched_unix=int(time.time()),
            )
        )
        ranked = orchestrator._rank_documents("presidents usa history", docs)
        self.assertIn("b.example", {block["domain"] for block in ranked})
        self.assertLessEqual(sum(1 for block in ranked[:4] if block["domain"] == "a.example"), 3)


def runtime_domain(url: str) -> str:
    return url.split("//", 1)[1].split("/", 1)[0]
