"""
Dynamic crawler source and clearance configuration.

The crawler should learn and adapt at runtime, so source profiles, search URL
templates, crawl limits, and clearance escalation live in JSON/env config
instead of being baked into the interface or swarm bridge.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "crawler_sources.json"
CONFIG_ENV = "AXIOM_CRAWLER_SOURCE_CONFIG"
SOURCE_DOMAIN_ENV = "AXIOM_SOURCE_DOMAINS"
MAX_SEARCH_SOURCES_ENV = "AXIOM_MAX_SEARCH_SOURCES"
LINK_EXPANSION_ENV = "AXIOM_LINK_EXPANSION_PER_DOC"
CLEARANCE_POLICY_ENV = "AXIOM_CLEARANCE_POLICY"
CLEARANCE_LEVELS_ENV = "AXIOM_CLEARANCE_LEVELS"
CLEARANCE_REQUIRE_DEV_ENV = "AXIOM_CLEARANCE_REQUIRE_DEV"

FALLBACK_CONFIG: Dict[str, Any] = {
    "version": "fallback",
    "limits": {
        "default_workers": 10,
        "default_max_workers": 10,
        "absolute_worker_limit": 100,
        "target_documents": 12,
        "default_waves": 3,
        "max_waves": 16,
        "early_stop_score": 12.0,
        "max_search_sources": 96,
        "link_expansion_per_doc": 6,
    },
    "default_clearance_policy": "standard",
    "clearance_policies": {
        "standard": {"levels": [1], "require_dev": False, "respect_availability": True},
        "dev": {"levels": [1, 2, 3, 4], "require_dev": True, "respect_availability": True},
        "deep": {"levels": [1, 2, 3, 4], "require_dev": False, "respect_availability": True},
        "max": {"levels": [1, 2, 3, 4], "require_dev": False, "respect_availability": False},
    },
    "profiles": [{"name": "empty", "always": True, "domains": []}],
    "search_templates": {},
    "article_templates": {},
}


def load_source_config() -> Dict[str, Any]:
    path = Path(os.environ.get(CONFIG_ENV, "") or DEFAULT_CONFIG_PATH)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        payload = dict(FALLBACK_CONFIG)
    return _normalize_config(payload)


def crawler_limits() -> Dict[str, Any]:
    return dict(load_source_config().get("limits", {}))


def configured_source_domains(query: str = "") -> List[str]:
    configured = os.environ.get(SOURCE_DOMAIN_ENV, "").strip()
    if configured:
        raw_domains = re.split(r"[\s,;]+", configured)
        return unique_domains(raw_domains)
    return seed_domains_for_query(query)


def seed_domains_for_query(query: str) -> List[str]:
    config = load_source_config()
    lowered = query.lower()
    matched_domains: List[str] = []
    fallback_domains: List[str] = []
    for profile in config.get("profiles", []):
        if not isinstance(profile, Mapping):
            continue
        terms = [str(term).lower() for term in profile.get("terms", []) if str(term).strip()]
        always = bool(profile.get("always", False))
        if always:
            fallback_domains.extend(str(domain) for domain in profile.get("domains", []))
        elif any(term in lowered for term in terms):
            matched_domains.extend(str(domain) for domain in profile.get("domains", []))
    return unique_domains([*matched_domains, *fallback_domains])


def source_urls_for_domains(query: str, domains: Iterable[str], *, reason: str) -> List[Dict[str, Any]]:
    encoded = quote(query)
    urls: List[Dict[str, Any]] = []
    templates = search_templates()
    for domain in unique_domains(domains):
        template = templates.get(domain, f"https://{domain}/search?q={{query}}")
        urls.append(
            {
                "url": template.format(query=encoded),
                "domain": domain,
                "reason": reason,
                "seeded": True,
                "cached": False,
            }
        )
        urls.append(
            {
                "url": f"https://{domain}/",
                "domain": domain,
                "reason": f"{reason}_root",
                "seeded": True,
                "cached": False,
            }
        )
    return urls


def search_templates() -> Dict[str, str]:
    raw = load_source_config().get("search_templates", {})
    if not isinstance(raw, Mapping):
        return {}
    return {normalize_domain(str(domain)): str(template) for domain, template in raw.items() if normalize_domain(str(domain))}


def article_templates() -> Dict[str, str]:
    raw = load_source_config().get("article_templates", {})
    if not isinstance(raw, Mapping):
        return {}
    return {normalize_domain(str(domain)): str(template) for domain, template in raw.items() if normalize_domain(str(domain))}


def domain_query_url(domain: str) -> str:
    normalized = normalize_domain(domain)
    if not normalized:
        return ""
    return search_templates().get(normalized, f"https://{normalized}/search?q={{query}}")


def domain_article_url(domain: str, slug: str) -> str:
    normalized = normalize_domain(domain)
    clean_slug = str(slug or "").strip()
    if not normalized or not clean_slug:
        return ""
    template = article_templates().get(normalized)
    if not template:
        return ""
    encoded_slug = quote(clean_slug.replace(" ", "_"))
    return template.format(slug=encoded_slug)


def max_search_sources() -> int:
    configured = os.environ.get(MAX_SEARCH_SOURCES_ENV, "").strip()
    default = int(crawler_limits().get("max_search_sources", 96) or 96)
    return bounded_int_value(configured, default, 1, 512)


def link_expansion_limit() -> int:
    configured = os.environ.get(LINK_EXPANSION_ENV, "").strip()
    default = int(crawler_limits().get("link_expansion_per_doc", 6) or 6)
    return bounded_int_value(configured, default, 0, 64)


def clearance_levels(fetcher: Any, *, env_mode: str) -> List[int]:
    explicit = parse_level_list(os.environ.get(CLEARANCE_LEVELS_ENV, ""))
    config = load_source_config()
    policy_name = os.environ.get(CLEARANCE_POLICY_ENV, "").strip().lower()
    if not policy_name:
        policy_name = "dev" if env_mode == "dev" else str(config.get("default_clearance_policy") or "standard")
    policy = config.get("clearance_policies", {}).get(policy_name, {})
    if not isinstance(policy, Mapping):
        policy = config.get("clearance_policies", {}).get("standard", {})
    levels = explicit or parse_level_list(policy.get("levels", [])) or [1]
    require_dev = bool(policy.get("require_dev", False))
    override_require = os.environ.get(CLEARANCE_REQUIRE_DEV_ENV, "").strip().lower()
    if override_require in {"0", "false", "no", "off"}:
        require_dev = False
    elif override_require in {"1", "true", "yes", "on"}:
        require_dev = True
    if require_dev and env_mode != "dev":
        levels = [level for level in levels if level == 1]
    respect_availability = bool(policy.get("respect_availability", True))
    if respect_availability:
        levels = [level for level in levels if level == 1 or _cl_available(fetcher, level)]
    return sorted(set(max(1, min(4, level)) for level in levels)) or [1]


def parse_level_list(raw: Any) -> List[int]:
    if isinstance(raw, list):
        values = raw
    else:
        values = re.split(r"[\s,;]+", str(raw or ""))
    levels: List[int] = []
    for item in values:
        try:
            level = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= level <= 4 and level not in levels:
            levels.append(level)
    return levels


def normalize_domain(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        return ""
    if "://" in value:
        value = urlparse(value).netloc
    value = value.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].strip(".")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-.")
    if not value or "." not in value or any(ch not in allowed for ch in value):
        return ""
    return value


def unique_domains(domains: Iterable[str]) -> List[str]:
    seen: set[str] = set()
    unique: List[str] = []
    for raw in domains:
        domain = normalize_domain(str(raw))
        if not domain or domain in seen:
            continue
        seen.add(domain)
        unique.append(domain)
    return unique


def bounded_int_value(value: Any, default: int, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def bounded_float_value(value: Any, default: float, low: float, high: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(low, min(high, parsed))


def _normalize_config(payload: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(FALLBACK_CONFIG)
    merged.update(payload if isinstance(payload, dict) else {})
    limits = dict(FALLBACK_CONFIG["limits"])
    limits.update(merged.get("limits", {}) if isinstance(merged.get("limits"), Mapping) else {})
    merged["limits"] = limits
    if not isinstance(merged.get("profiles"), list):
        merged["profiles"] = []
    if not isinstance(merged.get("search_templates"), Mapping):
        merged["search_templates"] = {}
    if not isinstance(merged.get("article_templates"), Mapping):
        merged["article_templates"] = {}
    if not isinstance(merged.get("clearance_policies"), Mapping):
        merged["clearance_policies"] = FALLBACK_CONFIG["clearance_policies"]
    return merged


def _cl_available(fetcher: Any, level: int) -> bool:
    cl_state = getattr(fetcher, "cl_state", None)
    if cl_state is None:
        return level == 1
    return bool(getattr(cl_state, f"cl{level}_available", level == 1))
