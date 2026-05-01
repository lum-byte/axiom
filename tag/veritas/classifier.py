"""VERITAS legitimacy classifier."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from typing import Any, Dict, Iterable, List, Optional

from tag.config import AxiomConfig, load_config
from tag.dic.mcp_anchors import AnchorRegistry
from tag.veritas.adversarial import AdversarialCrawlerPair
from tag.veritas.temporal import temporal_delta_label
from tag.veritas.threshold import ConfidenceSplitter


class VeritasLabel(StrEnum):
    CONFIRMED = "CONFIRMED"
    RUMOR = "RUMOR"
    LEGACY = "LEGACY"
    CONTESTED = "CONTESTED"


class VeritasEngine:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.enabled = self.config.bool("veritas.enabled", True)
        self.splitter = ConfidenceSplitter(config=self.config)
        self.anchors = AnchorRegistry(config=self.config)
        self.pair = AdversarialCrawlerPair()
        self.max_low = self.config.int("veritas.max_low_confidence_items", 24, low=1, high=256)

    async def classify(self, query: str, ranked_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not self.enabled or not ranked_blocks:
            return {"enabled": self.enabled, "classifications": [], "counts": {}}
        split = self.splitter.split(ranked_blocks)
        anchor_blocks = [
            block for block in ranked_blocks
            if self.anchors.is_anchor(str(block.get("domain") or ""))
        ]
        low = split.low[: self.max_low]
        classifications = await asyncio.gather(
            *(self._classify_one(block, anchor_blocks) for block in low),
            return_exceptions=False,
        )
        counts: Dict[str, int] = {}
        for item in classifications:
            counts[item["label"]] = counts.get(item["label"], 0) + 1
        return {
            "enabled": True,
            "query": query,
            "threshold": split.threshold,
            "high_confidence": len(split.high),
            "low_confidence": len(split.low),
            "counts": counts,
            "classifications": classifications,
        }

    async def _classify_one(self, block: Dict[str, Any], anchor_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        await asyncio.sleep(0)
        argument = self.pair.argue(block, anchor_blocks)
        temporal = temporal_delta_label(
            str(block.get("text") or ""),
            [str(anchor.get("text") or "") for anchor in anchor_blocks],
        )
        label = self._label(argument.confirm_score, argument.deny_score, temporal)
        block["veritas"] = {
            "label": label.value,
            "argument": argument.to_dict(),
            "temporal": temporal,
        }
        return {
            "identity": block_identity(block),
            "label": label.value,
            "url": block.get("url"),
            "domain": block.get("domain"),
            "rank": block.get("rank"),
            "score": block.get("score"),
            "argument": argument.to_dict(),
            "temporal": temporal,
        }

    def _label(self, confirm_score: float, deny_score: float, temporal: str) -> VeritasLabel:
        if confirm_score <= 0.05 and deny_score <= 0.05:
            return VeritasLabel.RUMOR
        if deny_score > confirm_score * 1.25:
            return VeritasLabel.LEGACY if temporal == "anchor_newer" else VeritasLabel.CONTESTED
        if confirm_score > deny_score * 1.15:
            return VeritasLabel.CONFIRMED
        return VeritasLabel.CONTESTED


def block_identity(block: Dict[str, Any]) -> str:
    return f"{block.get('url')}#{block.get('rank')}#{str(block.get('text') or '')[:80]}"
