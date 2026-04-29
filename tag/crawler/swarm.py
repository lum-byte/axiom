"""
AXIOM crawl swarm coordinator.

This module adapts the imported swarm/coordinator idea to TAG crawling:
one coordinator fans out independent crawler workers, keeps site ownership
exclusive, stops early when enough strong evidence is found, and expands from
discovered links only when the first wave is not good enough.

The workers are not a second crawler implementation.  Callers provide the
existing fetch/document function, normally backed by tag.crawler.fetcher.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse

from tag.crawler.source_config import bounded_float_value, bounded_int_value, crawler_limits

_LIMITS = crawler_limits()
DEFAULT_SWARM_WORKERS = int(_LIMITS.get("default_workers", 10) or 10)
DEFAULT_MAX_SWARM_WORKERS = int(_LIMITS.get("default_max_workers", DEFAULT_SWARM_WORKERS) or DEFAULT_SWARM_WORKERS)
ABSOLUTE_SWARM_WORKER_LIMIT = int(_LIMITS.get("absolute_worker_limit", 100) or 100)
DEFAULT_TARGET_DOCUMENTS = int(_LIMITS.get("target_documents", 12) or 12)
DEFAULT_MAX_WAVES = int(_LIMITS.get("default_waves", 3) or 3)
MAX_SWARM_WAVES = int(_LIMITS.get("max_waves", 16) or 16)
DEFAULT_EARLY_STOP_SCORE = float(_LIMITS.get("early_stop_score", 12.0) or 12.0)


FetchDocument = Callable[[Dict[str, Any]], Awaitable[Optional[Any]]]
FetchBatch = Callable[[Sequence[Dict[str, Any]]], Awaitable[List[Optional[Any]]]]
RankDocuments = Callable[[List[Any]], List[Dict[str, Any]]]
ExpandDocument = Callable[[Any], Iterable[Dict[str, Any]]]


@dataclass(frozen=True)
class AxiomCrawlSwarmConfig:
    worker_count: int = DEFAULT_SWARM_WORKERS
    target_documents: int = DEFAULT_TARGET_DOCUMENTS
    max_waves: int = DEFAULT_MAX_WAVES
    early_stop_score: float = DEFAULT_EARLY_STOP_SCORE
    requested_worker_count: Optional[int] = None
    max_worker_count: int = DEFAULT_MAX_SWARM_WORKERS

    @classmethod
    def from_env(cls, requested_worker_count: Optional[int] = None) -> "AxiomCrawlSwarmConfig":
        max_workers = _bounded_int(
            "AXIOM_CRAWL_MAX_WORKERS",
            DEFAULT_MAX_SWARM_WORKERS,
            1,
            ABSOLUTE_SWARM_WORKER_LIMIT,
        )
        if requested_worker_count is None:
            worker_count = _bounded_int("AXIOM_CRAWL_WORKERS", DEFAULT_SWARM_WORKERS, 1, max_workers)
        else:
            worker_count = max(1, min(max_workers, requested_worker_count))
        return cls(
            worker_count=worker_count,
            target_documents=_bounded_int("AXIOM_CRAWL_TARGET_DOCS", DEFAULT_TARGET_DOCUMENTS, 1, 64),
            max_waves=_bounded_int("AXIOM_CRAWL_WAVES", DEFAULT_MAX_WAVES, 1, MAX_SWARM_WAVES),
            early_stop_score=_bounded_float("AXIOM_CRAWL_EARLY_STOP_SCORE", DEFAULT_EARLY_STOP_SCORE, 1.0, 1000.0),
            requested_worker_count=requested_worker_count,
            max_worker_count=max_workers,
        )


@dataclass
class AxiomCrawlSwarmResult:
    documents: List[Any] = field(default_factory=list)
    attempted_candidates: List[Dict[str, Any]] = field(default_factory=list)
    attempted_domains: List[str] = field(default_factory=list)
    skipped_duplicate_sites: int = 0
    waves_completed: int = 0
    worker_count: int = 0
    requested_worker_count: Optional[int] = None
    max_worker_count: int = DEFAULT_MAX_SWARM_WORKERS
    early_stopped: bool = False

    @property
    def telemetry(self) -> Dict[str, Any]:
        return {
            "documents": len(self.documents),
            "attempted": len(self.attempted_candidates),
            "domains_attempted": len(self.attempted_domains),
            "skipped_duplicate_sites": self.skipped_duplicate_sites,
            "waves_completed": self.waves_completed,
            "worker_count": self.worker_count,
            "requested_worker_count": self.requested_worker_count,
            "max_worker_count": self.max_worker_count,
            "early_stopped": self.early_stopped,
        }


class AxiomCrawlSwarm:
    def __init__(
        self,
        *,
        fetch_document: FetchDocument,
        rank_documents: RankDocuments,
        expand_document: Optional[ExpandDocument] = None,
        fetch_batch: Optional[FetchBatch] = None,
        config: Optional[AxiomCrawlSwarmConfig] = None,
    ) -> None:
        self.fetch_document = fetch_document
        self.fetch_batch = fetch_batch
        self.rank_documents = rank_documents
        self.expand_document = expand_document or (lambda document: ())
        self.config = config or AxiomCrawlSwarmConfig.from_env()

    async def collect(self, candidates: Sequence[Dict[str, Any]]) -> AxiomCrawlSwarmResult:
        result = AxiomCrawlSwarmResult(
            worker_count=self.config.worker_count,
            requested_worker_count=self.config.requested_worker_count,
            max_worker_count=self.config.max_worker_count,
        )
        pending = self._unique_by_url(candidates)
        seen_urls = {str(candidate.get("url", "")) for candidate in pending}
        used_domains: set[str] = set()

        for wave_index in range(self.config.max_waves):
            batch, pending, skipped = self._next_distinct_domain_batch(pending, used_domains)
            result.skipped_duplicate_sites += skipped
            if not batch:
                break

            result.waves_completed = wave_index + 1
            documents = await self._run_batch(batch)
            result.attempted_candidates.extend(batch)
            for candidate in batch:
                domain = candidate_site(candidate)
                if domain:
                    result.attempted_domains.append(domain)

            result.documents.extend(document for document in documents if document is not None)
            if self._should_stop(result.documents):
                result.early_stopped = True
                break

            expanded = self._expand_documents(result.documents, seen_urls)
            pending.extend(expanded)
            seen_urls.update(str(candidate.get("url", "")) for candidate in expanded)

        return result

    async def _run_batch(self, batch: Sequence[Dict[str, Any]]) -> List[Optional[Any]]:
        limited_batch = list(batch[: self.config.worker_count])
        if self.fetch_batch is not None:
            return await self.fetch_batch(limited_batch)
        semaphore = asyncio.Semaphore(max(1, self.config.worker_count))

        async def run_one(candidate: Dict[str, Any]) -> Optional[Any]:
            async with semaphore:
                return await self.fetch_document(candidate)

        tasks = [asyncio.create_task(run_one(candidate)) for candidate in limited_batch]
        if not tasks:
            return []
        return list(await asyncio.gather(*tasks))

    def _next_distinct_domain_batch(
        self,
        pending: Sequence[Dict[str, Any]],
        used_domains: set[str],
    ) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], int]:
        batch: List[Dict[str, Any]] = []
        deferred: List[Dict[str, Any]] = []
        batch_domains: set[str] = set()
        skipped = 0

        for index, candidate in enumerate(pending):
            domain = candidate_site(candidate)
            if not domain:
                skipped += 1
                continue
            if domain in used_domains or domain in batch_domains:
                skipped += 1
                continue
            batch.append(candidate)
            batch_domains.add(domain)
            used_domains.add(domain)
            if len(batch) >= self.config.worker_count:
                deferred.extend(pending[index + 1 :])
                break
        return batch, deferred, skipped

    def _should_stop(self, documents: List[Any]) -> bool:
        if len(documents) >= self.config.target_documents:
            return True
        ranked = self.rank_documents(documents)
        if not ranked:
            return False
        return float(ranked[0].get("score", 0.0)) >= self.config.early_stop_score

    def _expand_documents(self, documents: List[Any], seen_urls: set[str]) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        for document in documents:
            for candidate in self.expand_document(document):
                url = str(candidate.get("url", ""))
                if not url or url in seen_urls:
                    continue
                expanded.append(candidate)
        return self._unique_by_url(expanded)

    @staticmethod
    def _unique_by_url(candidates: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen: set[str] = set()
        unique: List[Dict[str, Any]] = []
        for candidate in candidates:
            url = str(candidate.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            unique.append(candidate)
        return unique


def candidate_site(candidate: Dict[str, Any]) -> str:
    domain = str(candidate.get("domain", "")).strip().lower()
    if domain:
        return domain
    url = str(candidate.get("url", ""))
    parsed = urlparse(url)
    return parsed.netloc.lower().strip(".")


def _bounded_int(name: str, default: int, low: int, high: int) -> int:
    raw = os.environ.get(name, "").strip()
    return bounded_int_value(raw, default, low, high)


def _bounded_float(name: str, default: float, low: float, high: float) -> float:
    raw = os.environ.get(name, "").strip()
    return bounded_float_value(raw, default, low, high)
