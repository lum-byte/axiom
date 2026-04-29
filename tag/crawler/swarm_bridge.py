"""
AXIOM swarm bridge.

The imported TypeScript swarm talks in generic tasks, prompts, messages, and
queues.  This module turns that generic shape into the concrete crawl plan that
TAG can execute: query text, seed domains, source URLs, worker request, and
compute controls.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlparse

from tag.crawler.swarm import AxiomCrawlSwarmConfig, DEFAULT_SWARM_WORKERS


AXIOM_SWARM_WATERMARK = "axiom.swarm.webwide.v1"
SWARM_HEAD_RE = re.compile(r"^\s*swarm(?:\s+-(?P<workers>\d{1,4}))?\s*$", re.IGNORECASE)
DEPTH_SEGMENT_RE = re.compile(r"^\s*depth\s+-?(?P<depth>\d{1,2})\s*$", re.IGNORECASE)
WORKER_TEXT_RE = re.compile(r"\bswarm\s*-(?P<workers>\d{1,4})\b", re.IGNORECASE)
DEPTH_TEXT_RE = re.compile(r"\bdepth\s*-?(?P<depth>\d{1,2})\b", re.IGNORECASE)
DOMAIN_RE = re.compile(
    r"(?<!@)\b(?:https?://)?"
    r"(?P<domain>[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]{2,})+)"
    r"\b",
    re.IGNORECASE,
)

BROAD_SEED_DOMAINS = (
    "archive.org",
    "britannica.com",
    "reuters.com",
    "bbc.com",
    "wikipedia.org",
    "wikidata.org",
    "loc.gov",
    "usa.gov",
)

US_GOVERNMENT_SEED_DOMAINS = (
    "whitehouse.gov",
    "archives.gov",
    "usa.gov",
    "loc.gov",
    "congress.gov",
    "senate.gov",
    "house.gov",
    "supremecourt.gov",
    "state.gov",
    "britannica.com",
    "wikipedia.org",
)

TECH_SEED_DOMAINS = (
    "docs.python.org",
    "developer.mozilla.org",
    "github.com",
    "nist.gov",
    "ietf.org",
    "w3.org",
)

SCIENCE_SEED_DOMAINS = (
    "nasa.gov",
    "nih.gov",
    "noaa.gov",
    "usgs.gov",
    "energy.gov",
    "who.int",
)

SOURCE_SITE_SEARCH_URLS = {
    "archive.org": "https://archive.org/search?query={query}",
    "archives.gov": "https://www.archives.gov/search?search={query}",
    "britannica.com": "https://www.britannica.com/search?query={query}",
    "developer.mozilla.org": "https://developer.mozilla.org/en-US/search?q={query}",
    "docs.python.org": "https://docs.python.org/3/search.html?q={query}",
    "github.com": "https://github.com/search?q={query}",
    "loc.gov": "https://www.loc.gov/search/?fo=json&q={query}",
    "reuters.com": "https://www.reuters.com/site-search/?query={query}",
    "usa.gov": "https://search.usa.gov/search?query={query}&affiliate=usagov",
    "wikidata.org": "https://www.wikidata.org/wiki/Special:Search?search={query}",
    "wikipedia.org": "https://en.wikipedia.org/w/index.php?search={query}",
}


def parse_swarm_search_payload(payload: str) -> tuple[str, Optional[Dict[str, Any]]]:
    parsed = _parse_swarm_command(payload)
    if parsed is None:
        return payload.strip(), None
    query = parsed["query"]
    requested_workers = parsed.get("workers")
    depth = parsed.get("depth")
    plan = plan_from_generic_talk(
        {
            "kind": "axiom_cli_swarm",
            "content": query,
            "requested_workers": requested_workers,
            "depth": depth,
            "hints": {"command_syntax": "search | swarm -N | depth -D | query"},
        },
        default_query=query,
        requested_workers=requested_workers,
        depth=depth,
    )
    return query, plan


def plan_from_generic_talk(
    payload: Any,
    *,
    default_query: str = "",
    requested_workers: Optional[int] = None,
    depth: Optional[int] = None,
) -> Dict[str, Any]:
    text = _normalize_query_text(_extract_text(payload).strip() or default_query.strip())
    requested = requested_workers or _extract_requested_workers(payload) or DEFAULT_SWARM_WORKERS
    max_waves = depth or _extract_requested_depth(payload) or 3
    seed_domains = _unique_domains([*extract_domains(text), *pick_seed_domains(text)])
    source_urls = _source_urls_for_domains(text, seed_domains, reason="swarm_bridge_seed")
    return {
        "watermark": AXIOM_SWARM_WATERMARK,
        "intent": infer_intent(text),
        "query": text,
        "worker_count": requested,
        "requested_worker_count": requested,
        "target_documents": 12,
        "max_waves": _bounded_int_value(max_waves, 3, 1, 8),
        "depth": _bounded_int_value(max_waves, 3, 1, 8),
        "early_stop_score": 12.0,
        "seed_domains": seed_domains,
        "source_urls": source_urls,
        "constraints": {
            "one_worker_per_site": True,
            "no_duplicate_site_fetch": True,
            "no_external_search_engine": True,
            "default_worker_ceiling": 10,
            "absolute_worker_limit": 100,
            "lower_compute": [
                "dedupe_urls",
                "dedupe_sites",
                "quality_early_stop",
                "link_expansion_after_wave",
            ],
        },
        "origin": _origin_context(payload),
    }


def normalize_crawl_plan(payload: Any, *, default_query: str = "") -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return plan_from_generic_talk(payload, default_query=default_query)

    query = _normalize_query_text(str(payload.get("query") or _extract_text(payload) or default_query).strip())
    requested = _extract_requested_workers(payload) or DEFAULT_SWARM_WORKERS
    max_waves = _extract_requested_depth(payload) or _bounded_int_value(payload.get("max_waves"), 3, 1, 8)
    seed_domains = _unique_domains(
        [
            *_normalize_domains(payload.get("seed_domains")),
            *extract_domains(query),
            *pick_seed_domains(query),
        ]
    )
    source_urls = _normalize_source_urls(payload.get("source_urls"))
    source_urls.extend(_source_urls_for_domains(query, seed_domains, reason="swarm_bridge_seed"))
    source_urls = _unique_source_urls(source_urls)
    constraints = dict(payload.get("constraints") or {})
    constraints.setdefault("one_worker_per_site", True)
    constraints.setdefault("no_duplicate_site_fetch", True)
    constraints.setdefault("no_external_search_engine", True)
    constraints.setdefault("default_worker_ceiling", 10)
    constraints.setdefault("absolute_worker_limit", 100)
    constraints.setdefault(
        "lower_compute",
        ["dedupe_urls", "dedupe_sites", "quality_early_stop", "link_expansion_after_wave"],
    )
    return {
        "watermark": AXIOM_SWARM_WATERMARK,
        "intent": str(payload.get("intent") or infer_intent(query)),
        "query": query,
        "worker_count": requested,
        "requested_worker_count": requested,
        "target_documents": _bounded_int_value(payload.get("target_documents"), 12, 1, 64),
        "max_waves": _bounded_int_value(max_waves, 3, 1, 8),
        "depth": _bounded_int_value(max_waves, 3, 1, 8),
        "early_stop_score": _bounded_float_value(payload.get("early_stop_score"), 12.0, 1.0, 1000.0),
        "seed_domains": seed_domains,
        "source_urls": source_urls,
        "constraints": constraints,
        "origin": dict(payload.get("origin") or _origin_context(payload)),
    }


def crawl_config_from_plan(plan: Optional[Mapping[str, Any]]) -> AxiomCrawlSwarmConfig:
    if not plan:
        return AxiomCrawlSwarmConfig.from_env()
    requested = _extract_requested_workers(plan)
    config = AxiomCrawlSwarmConfig.from_env(requested_worker_count=requested)
    return replace(
        config,
        target_documents=_bounded_int_value(plan.get("target_documents"), config.target_documents, 1, 64),
        max_waves=_bounded_int_value(plan.get("max_waves"), config.max_waves, 1, 8),
        early_stop_score=_bounded_float_value(plan.get("early_stop_score"), config.early_stop_score, 1.0, 1000.0),
    )


def infer_intent(text: str) -> str:
    lowered = text.lower()
    if lowered.startswith(("fetch ", "open ", "load ")):
        return "fetch"
    if lowered.startswith(("learn ", "crawl ", "index ")):
        return "learn"
    return "web_search"


def extract_domains(text: str) -> List[str]:
    return _unique_domains(match.group("domain") for match in DOMAIN_RE.finditer(text))


def pick_seed_domains(text: str) -> List[str]:
    lowered = text.lower()
    domains: List[str] = []
    if any(term in lowered for term in ("president", "white house", "usa", "united states", "congress")):
        domains.extend(US_GOVERNMENT_SEED_DOMAINS)
    if any(term in lowered for term in ("python", "javascript", "typescript", "api", "cuda", "mamba", "software", "code")):
        domains.extend(TECH_SEED_DOMAINS)
    if any(term in lowered for term in ("science", "space", "health", "climate", "earthquake", "medicine")):
        domains.extend(SCIENCE_SEED_DOMAINS)
    domains.extend(BROAD_SEED_DOMAINS)
    return _unique_domains(domains)


def _source_urls_for_domains(query: str, domains: Iterable[str], *, reason: str) -> List[Dict[str, Any]]:
    quoted_query = quote(query)
    sources: List[Dict[str, Any]] = []
    for domain in domains:
        search_url = SOURCE_SITE_SEARCH_URLS.get(domain, f"https://{domain}/search?q={{query}}")
        sources.append(
            {
                "url": search_url.format(query=quoted_query),
                "domain": domain,
                "reason": reason,
                "seeded": True,
                "cached": False,
            }
        )
        sources.append(
            {
                "url": f"https://{domain}/",
                "domain": domain,
                "reason": f"{reason}_root",
                "seeded": True,
                "cached": False,
            }
        )
    return sources


def _extract_text(value: Any) -> str:
    parts: List[str] = []
    _extract_text_parts(value, parts, depth=0)
    return "\n".join(part for part in parts if part).strip()


def _normalize_query_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    for line in lines:
        parsed = _parse_swarm_command(line)
        if parsed is not None:
            return parsed["query"]
    for line in lines:
        lowered = line.lower()
        for prefix in (
            "search the whole internet for ",
            "search the internet for ",
            "search the web for ",
            "search for ",
        ):
            if lowered.startswith(prefix):
                return line[len(prefix) :].strip()
    return lines[0]


def _parse_swarm_command(text: str) -> Optional[Dict[str, Any]]:
    segments = [segment.strip() for segment in text.split("|") if segment.strip()]
    if not segments:
        return None
    if segments[0].lower() == "search":
        segments = segments[1:]
    if not segments:
        return None
    swarm_match = SWARM_HEAD_RE.match(segments[0])
    if not swarm_match:
        return None
    requested_workers = _coerce_int(swarm_match.group("workers"))
    depth: Optional[int] = None
    query_segments: List[str] = []
    for segment in segments[1:]:
        depth_match = DEPTH_SEGMENT_RE.match(segment)
        if depth_match:
            depth = _bounded_int_value(depth_match.group("depth"), 3, 1, 8)
            continue
        query_segments.append(segment)
    return {
        "query": " | ".join(query_segments).strip(),
        "workers": requested_workers,
        "depth": depth,
    }


def _extract_text_parts(value: Any, parts: List[str], *, depth: int) -> None:
    if depth > 5 or value is None:
        return
    if isinstance(value, str):
        parts.append(value)
        return
    if isinstance(value, bytes):
        parts.append(value.decode("utf-8", errors="replace"))
        return
    if isinstance(value, Mapping):
        for key in ("query", "content", "text", "prompt", "payload", "value", "command", "title", "description"):
            if key in value:
                _extract_text_parts(value.get(key), parts, depth=depth + 1)
        for key in ("task", "message", "messages", "queue", "hints", "input"):
            if key in value:
                _extract_text_parts(value.get(key), parts, depth=depth + 1)
        return
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        for item in list(value)[:32]:
            _extract_text_parts(item, parts, depth=depth + 1)


def _extract_requested_workers(value: Any) -> Optional[int]:
    if isinstance(value, Mapping):
        for key in ("requested_worker_count", "requested_workers", "worker_count", "workers", "parallelism", "concurrency"):
            worker_count = _coerce_int(value.get(key))
            if worker_count is not None:
                return worker_count
        for key in ("constraints", "hints", "config", "options", "crawl"):
            nested = value.get(key)
            worker_count = _extract_requested_workers(nested)
            if worker_count is not None:
                return worker_count
    text = _extract_text(value)
    match = WORKER_TEXT_RE.search(text)
    return _coerce_int(match.group("workers")) if match else None


def _extract_requested_depth(value: Any) -> Optional[int]:
    if isinstance(value, Mapping):
        for key in ("depth", "crawl_depth", "max_waves", "waves"):
            depth = _coerce_int(value.get(key))
            if depth is not None:
                return _bounded_int_value(depth, 3, 1, 8)
        for key in ("constraints", "hints", "config", "options", "crawl"):
            nested = value.get(key)
            depth = _extract_requested_depth(nested)
            if depth is not None:
                return depth
    text = _extract_text(value)
    parsed = _parse_swarm_command(text)
    if parsed is not None and parsed.get("depth") is not None:
        return _bounded_int_value(parsed.get("depth"), 3, 1, 8)
    match = DEPTH_TEXT_RE.search(text)
    return _bounded_int_value(match.group("depth"), 3, 1, 8) if match else None


def _origin_context(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"kind": "text", "message_count": 1 if str(payload).strip() else 0}
    messages = payload.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0
    task = payload.get("task") if isinstance(payload.get("task"), Mapping) else {}
    identity = payload.get("identity") if isinstance(payload.get("identity"), Mapping) else {}
    return {
        "kind": str(payload.get("kind") or payload.get("type") or "generic_swarm_talk"),
        "task_type": str(task.get("type") or payload.get("task_type") or ""),
        "agent_name": str(identity.get("agentName") or payload.get("agent_name") or ""),
        "team_name": str(identity.get("teamName") or payload.get("team_name") or ""),
        "message_count": message_count,
    }


def _normalize_domains(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = re.split(r"[\s,;]+", value)
    elif isinstance(value, Iterable):
        raw_values = [str(item) for item in value]
    else:
        return []
    return _unique_domains(raw_values)


def _normalize_domain(raw: str) -> str:
    raw = raw.strip().lower()
    if not raw:
        return ""
    if "://" in raw:
        raw = urlparse(raw).netloc
    raw = raw.split("/")[0].strip(".")
    if not raw or "." not in raw or any(ch.isspace() for ch in raw):
        return ""
    return raw


def _normalize_source_urls(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes)):
        return []
    sources: List[Dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url") or "").strip()
        domain = _normalize_domain(str(item.get("domain") or url))
        if not url or not domain:
            continue
        sources.append(
            {
                "url": url,
                "domain": domain,
                "reason": str(item.get("reason") or "swarm_bridge_source"),
                "seeded": bool(item.get("seeded", True)),
                "cached": bool(item.get("cached", False)),
            }
        )
    return sources


def _unique_domains(domains: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for raw in domains:
        domain = _normalize_domain(str(raw))
        if not domain or domain in seen:
            continue
        seen.add(domain)
        unique.append(domain)
    return unique


def _unique_source_urls(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for source in sources:
        url = str(source.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(source)
    return unique[:128]


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bounded_int_value(value: Any, default: int, low: int, high: int) -> int:
    parsed = _coerce_int(value)
    if parsed is None:
        return default
    return max(low, min(high, parsed))


def _bounded_float_value(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, parsed))
