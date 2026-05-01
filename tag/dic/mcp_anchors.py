"""MCP-style anchor wrappers for DIC.

The module exposes typed anchor requests/results without forcing TAG crawlers to
know about a concrete MCP transport.  A future MCP server can feed these same
types directly; today the wrappers normalize TAG crawler results into the same
schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from tag.config import AxiomConfig, load_config


@dataclass(frozen=True)
class AnchorRequest:
    anchor: str
    query: str
    tool: str
    priority: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "anchor": self.anchor,
            "query": self.query,
            "tool": self.tool,
            "priority": self.priority,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class AnchorContextBlock:
    source: str
    url: str
    title: str
    text: str
    trust_tier: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "url": self.url,
            "title": self.title,
            "text": self.text,
            "trust_tier": self.trust_tier,
            "score": round(self.score, 4),
            "metadata": dict(self.metadata),
        }


class AnchorRegistry:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.anchor_domains = [
            normalize_domain(domain)
            for domain in self.config.get("dic.anchor_domains", [])
            if normalize_domain(str(domain))
        ]
        self.news_domains = {
            normalize_domain(domain)
            for domain in self.config.get("dic.news_anchor_domains", [])
            if normalize_domain(str(domain))
        }
        self.wikipedia_domains = {
            normalize_domain(domain)
            for domain in self.config.get("dic.wikipedia_anchor_domains", [])
            if normalize_domain(str(domain))
        }

    def build_requests(self, query: str) -> List[AnchorRequest]:
        requests: List[AnchorRequest] = []
        for index, domain in enumerate(self.anchor_domains):
            tool = self._tool_for_domain(domain)
            requests.append(
                AnchorRequest(
                    anchor=domain,
                    query=query,
                    tool=tool,
                    priority=100 - index,
                    metadata={"transport": "mcp-compatible", "domain": domain},
                )
            )
        return requests

    def normalize_blocks(self, blocks: Iterable[Dict[str, Any]]) -> List[AnchorContextBlock]:
        normalized: List[AnchorContextBlock] = []
        for block in blocks:
            domain = normalize_domain(str(block.get("domain") or block.get("url") or ""))
            explicit_tier = str(block.get("trust_tier") or "").strip().lower()
            is_typed_mcp_anchor = explicit_tier in {"wikipedia", "news", "scholar", "wayback", "web", "anchor"}
            if not self.is_anchor(domain) and not is_typed_mcp_anchor:
                continue
            normalized.append(
                AnchorContextBlock(
                    source=domain or str(block.get("source") or explicit_tier or "mcp-anchor"),
                    url=str(block.get("url") or ""),
                    title=str(block.get("title") or domain),
                    text=str(block.get("text") or ""),
                    trust_tier=explicit_tier or self.trust_tier(domain),
                    score=float(block.get("score", 0.0) or 0.0),
                    metadata={
                        "rank": block.get("rank"),
                        "fetch_mode": block.get("fetch_mode"),
                        "topology_class": block.get("topology_class"),
                        "mcp_tool": block.get("mcp_tool"),
                    },
                )
            )
        normalized.sort(key=lambda item: (-item.score, item.source, item.url))
        return normalized

    def is_anchor(self, domain: str) -> bool:
        normalized = normalize_domain(domain)
        return any(normalized == anchor or normalized.endswith(f".{anchor}") for anchor in self.anchor_domains)

    def trust_tier(self, domain: str) -> str:
        normalized = normalize_domain(domain)
        if any(normalized == item or normalized.endswith(f".{item}") for item in self.wikipedia_domains):
            return "wikipedia"
        if any(normalized == item or normalized.endswith(f".{item}") for item in self.news_domains):
            return "news"
        return "anchor"

    def _tool_for_domain(self, domain: str) -> str:
        if domain in self.wikipedia_domains:
            return "mcp-wikipedia"
        if domain in self.news_domains:
            return "mcp-news"
        if "scholar" in domain or "arxiv" in domain or "crossref" in domain:
            return "mcp-scholar"
        if "archive" in domain or "wayback" in domain:
            return "mcp-wayback"
        return "mcp-anchor"


def normalize_domain(domain: str) -> str:
    value = str(domain or "").strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0].strip(".")
