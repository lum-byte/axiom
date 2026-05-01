"""
AXIOM crawl fanout bridge.

The imported TypeScript coordinator talks in generic tasks, prompts, messages, and
queues.  This module turns that generic shape into the concrete crawl plan that
TAG can execute: query text, seed domains, source URLs, worker request, and
compute controls.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from tag.crawler.source_config import (
    bounded_float_value,
    bounded_int_value,
    crawler_limits,
    seed_domains_for_query,
    source_urls_for_domains,
    unique_domains,
)
from tag.crawler.swarm import AxiomCrawlSwarmConfig, ABSOLUTE_SWARM_WORKER_LIMIT, DEFAULT_SWARM_WORKERS
from tag.dic.gbnf_dsl import parse_expansion_directive


AXIOM_FANOUT_WATERMARK = "axiom.fanout.webwide.v1"
AXIOM_SWARM_WATERMARK = AXIOM_FANOUT_WATERMARK
WORKER_HEAD_RE = re.compile(r"^\s*(?:fanout|parallel|workers?|crawlers?|swarm)(?:\s+-(?P<workers>\d{1,4}))?\s*$", re.IGNORECASE)
DEPTH_SEGMENT_RE = re.compile(r"^\s*depth\s+-?(?P<depth>\d{1,2})\s*$", re.IGNORECASE)
EXP_SEGMENT_RE = re.compile(r"^\s*(?:exp|expand|expansion)\s+-?(?P<expansion>\d{1,3})\s*$", re.IGNORECASE)
RECHECK_SEGMENT_RE = re.compile(r"^\s*(?:recheck|refresh|nocache|no-cache)\s*$", re.IGNORECASE)
WORKER_TEXT_RE = re.compile(r"\b(?:fanout|parallel|workers?|crawlers?|swarm)\s*-(?P<workers>\d{1,4})\b", re.IGNORECASE)
DEPTH_TEXT_RE = re.compile(r"\bdepth\s*-?(?P<depth>\d{1,2})\b", re.IGNORECASE)
RECHECK_TEXT_RE = re.compile(r"\b(?:recheck|refresh|nocache|no-cache)\b", re.IGNORECASE)
DOMAIN_RE = re.compile(
    r"(?<!@)\b(?:https?://)?"
    r"(?P<domain>[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?(?:\.[a-z0-9-]{2,})+)"
    r"\b",
    re.IGNORECASE,
)

_LIMITS = crawler_limits()


def parse_swarm_search_payload(payload: str) -> tuple[str, Optional[Dict[str, Any]]]:
    parsed = _parse_swarm_command(payload)
    if parsed is None:
        return payload.strip(), None
    query = parsed["query"]
    requested_workers = parsed.get("workers")
    depth = parsed.get("depth")
    expansion_count = parsed.get("expansion_count")
    recheck = bool(parsed.get("recheck", False))
    plan = plan_from_generic_talk(
        {
            "kind": "axiom_cli_fanout",
            "content": query,
            "requested_workers": requested_workers,
            "depth": depth,
            "expansion_count": expansion_count,
            "recheck": recheck,
            "hints": {"command_syntax": "search | fanout -N | depth -D | query"},
        },
        default_query=query,
        requested_workers=requested_workers,
        depth=depth,
        expansion_count=expansion_count,
        recheck=recheck,
    )
    return query, plan


def plan_from_generic_talk(
    payload: Any,
    *,
    default_query: str = "",
    requested_workers: Optional[int] = None,
    depth: Optional[int] = None,
    expansion_count: Optional[int] = None,
    recheck: bool = False,
) -> Dict[str, Any]:
    text, inline_expansion = parse_expansion_directive(_normalize_query_text(_extract_text(payload).strip() or default_query.strip()))
    requested = requested_workers or _extract_requested_workers(payload) or DEFAULT_SWARM_WORKERS
    max_waves = depth or _extract_requested_depth(payload) or 3
    expansion = expansion_count or _extract_requested_expansion(payload) or inline_expansion or 0
    force_recheck = recheck or _extract_recheck(payload)
    seed_domains = unique_domains([*extract_domains(text), *pick_seed_domains(text)])
    source_urls = source_urls_for_domains(text, seed_domains, reason="fanout_bridge_seed")
    return {
        "watermark": AXIOM_SWARM_WATERMARK,
        "intent": infer_intent(text),
        "query": text,
        "worker_count": requested,
        "requested_worker_count": requested,
        "target_documents": int(_LIMITS.get("target_documents", 12) or 12),
        "max_waves": _bounded_int_value(max_waves, 3, 1, int(_LIMITS.get("max_waves", 16) or 16)),
        "depth": _bounded_int_value(max_waves, 3, 1, int(_LIMITS.get("max_waves", 16) or 16)),
        "expansion_count": _bounded_int_value(expansion, 0, 0, int(_LIMITS.get("max_expansion_limit", 100) or 100)),
        "recheck": bool(force_recheck),
        "early_stop_score": float(_LIMITS.get("early_stop_score", 12.0) or 12.0),
        "seed_domains": seed_domains,
        "source_urls": source_urls,
        "constraints": {
            "one_worker_per_site": True,
            "no_duplicate_site_fetch": True,
            "no_external_search_engine": True,
            "default_worker_ceiling": int(_LIMITS.get("default_max_workers", 10) or 10),
            "absolute_worker_limit": ABSOLUTE_SWARM_WORKER_LIMIT,
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

    query, inline_expansion = parse_expansion_directive(_normalize_query_text(str(payload.get("query") or _extract_text(payload) or default_query).strip()))
    requested = _extract_requested_workers(payload) or DEFAULT_SWARM_WORKERS
    max_waves = _extract_requested_depth(payload) or _bounded_int_value(payload.get("max_waves"), 3, 1, int(_LIMITS.get("max_waves", 16) or 16))
    expansion = _extract_requested_expansion(payload) or inline_expansion or 0
    force_recheck = _extract_recheck(payload)
    seed_domains = unique_domains(
        [
            *_normalize_domains(payload.get("seed_domains")),
            *extract_domains(query),
            *pick_seed_domains(query),
        ]
    )
    source_urls = _normalize_source_urls(payload.get("source_urls"))
    source_urls.extend(source_urls_for_domains(query, seed_domains, reason="fanout_bridge_seed"))
    source_urls = _unique_source_urls(source_urls)
    constraints = dict(payload.get("constraints") or {})
    constraints.setdefault("one_worker_per_site", True)
    constraints.setdefault("no_duplicate_site_fetch", True)
    constraints.setdefault("no_external_search_engine", True)
    constraints.setdefault("default_worker_ceiling", int(_LIMITS.get("default_max_workers", 10) or 10))
    constraints.setdefault("absolute_worker_limit", ABSOLUTE_SWARM_WORKER_LIMIT)
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
        "target_documents": _bounded_int_value(payload.get("target_documents"), int(_LIMITS.get("target_documents", 12) or 12), 1, 64),
        "max_waves": _bounded_int_value(max_waves, 3, 1, int(_LIMITS.get("max_waves", 16) or 16)),
        "depth": _bounded_int_value(max_waves, 3, 1, int(_LIMITS.get("max_waves", 16) or 16)),
        "expansion_count": _bounded_int_value(expansion, 0, 0, int(_LIMITS.get("max_expansion_limit", 100) or 100)),
        "recheck": bool(force_recheck),
        "early_stop_score": _bounded_float_value(payload.get("early_stop_score"), float(_LIMITS.get("early_stop_score", 12.0) or 12.0), 1.0, 1000.0),
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
        max_waves=_bounded_int_value(plan.get("max_waves"), config.max_waves, 1, int(_LIMITS.get("max_waves", 16) or 16)),
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
    return unique_domains(match.group("domain") for match in DOMAIN_RE.finditer(text))


def pick_seed_domains(text: str) -> List[str]:
    return seed_domains_for_query(text)


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
    requested_workers: Optional[int] = None
    depth: Optional[int] = None
    expansion_count: Optional[int] = None
    recheck = False
    saw_directive = False
    query_segments: List[str] = []
    for segment in segments:
        worker_match = WORKER_HEAD_RE.match(segment)
        if worker_match:
            requested_workers = _coerce_int(worker_match.group("workers"))
            saw_directive = True
            continue
        depth_match = DEPTH_SEGMENT_RE.match(segment)
        if depth_match:
            depth = _bounded_int_value(depth_match.group("depth"), 3, 1, _max_waves())
            saw_directive = True
            continue
        expansion_match = EXP_SEGMENT_RE.match(segment)
        if expansion_match:
            expansion_count = _bounded_int_value(expansion_match.group("expansion"), 0, 0, int(_LIMITS.get("max_expansion_limit", 100) or 100))
            saw_directive = True
            continue
        if RECHECK_SEGMENT_RE.match(segment):
            recheck = True
            saw_directive = True
            continue
        query_segments.append(segment)
    if not saw_directive:
        return None
    return {
        "query": " | ".join(query_segments).strip(),
        "workers": requested_workers,
        "depth": depth,
        "expansion_count": expansion_count,
        "recheck": recheck,
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
                return _bounded_int_value(depth, 3, 1, _max_waves())
        for key in ("constraints", "hints", "config", "options", "crawl"):
            nested = value.get(key)
            depth = _extract_requested_depth(nested)
            if depth is not None:
                return depth
    text = _extract_text(value)
    parsed = _parse_swarm_command(text)
    if parsed is not None and parsed.get("depth") is not None:
        return _bounded_int_value(parsed.get("depth"), 3, 1, _max_waves())
    match = DEPTH_TEXT_RE.search(text)
    return _bounded_int_value(match.group("depth"), 3, 1, _max_waves()) if match else None


def _extract_requested_expansion(value: Any) -> Optional[int]:
    if isinstance(value, Mapping):
        for key in ("expansion_count", "expansions", "expand", "exp"):
            expansion = _coerce_int(value.get(key))
            if expansion is not None:
                return _bounded_int_value(expansion, 0, 0, int(_LIMITS.get("max_expansion_limit", 100) or 100))
        for key in ("constraints", "hints", "config", "options", "crawl"):
            nested = value.get(key)
            expansion = _extract_requested_expansion(nested)
            if expansion is not None:
                return expansion
    text = _extract_text(value)
    parsed = _parse_swarm_command(text)
    if parsed is not None and parsed.get("expansion_count") is not None:
        return _bounded_int_value(parsed.get("expansion_count"), 0, 0, int(_LIMITS.get("max_expansion_limit", 100) or 100))
    _, inline = parse_expansion_directive(text)
    return _bounded_int_value(inline, 0, 0, int(_LIMITS.get("max_expansion_limit", 100) or 100)) if inline is not None else None


def _extract_recheck(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key in ("recheck", "refresh", "nocache", "no_cache", "force_refresh"):
            if _truthy(value.get(key)):
                return True
        for key in ("constraints", "hints", "config", "options", "crawl"):
            if _extract_recheck(value.get(key)):
                return True
    text = _extract_text(value)
    parsed = _parse_swarm_command(text)
    if parsed is not None and parsed.get("recheck"):
        return True
    return bool(RECHECK_TEXT_RE.search(text))


def _origin_context(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {"kind": "text", "message_count": 1 if str(payload).strip() else 0}
    messages = payload.get("messages")
    message_count = len(messages) if isinstance(messages, list) else 0
    task = payload.get("task") if isinstance(payload.get("task"), Mapping) else {}
    identity = payload.get("identity") if isinstance(payload.get("identity"), Mapping) else {}
    return {
        "kind": str(payload.get("kind") or payload.get("type") or "generic_fanout_talk"),
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
    return unique_domains(raw_values)


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
                "reason": str(item.get("reason") or "fanout_bridge_source"),
                "seeded": bool(item.get("seeded", True)),
                "cached": bool(item.get("cached", False)),
            }
        )
    return sources


def _unique_source_urls(sources: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for source in sources:
        url = str(source.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        unique.append(source)
    return unique[: _source_url_cap()]


def _coerce_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "recheck", "refresh", "nocache", "no-cache"}


def _bounded_int_value(value: Any, default: int, low: int, high: int) -> int:
    return bounded_int_value(value, default, low, high)


def _bounded_float_value(value: Any, default: float, low: float, high: float) -> float:
    return bounded_float_value(value, default, low, high)


def _max_waves() -> int:
    return int(_LIMITS.get("max_waves", 16) or 16)


def _source_url_cap() -> int:
    return int(_LIMITS.get("max_search_sources", 128) or 128)
