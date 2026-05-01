"""Confidence split logic for VERITAS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tag.config import AxiomConfig, load_config


@dataclass(frozen=True)
class ThresholdSplit:
    high: List[Dict[str, Any]]
    low: List[Dict[str, Any]]
    threshold: float


class ConfidenceSplitter:
    def __init__(self, *, config: AxiomConfig | None = None) -> None:
        self.config = config or load_config()
        self.threshold = self.config.float("veritas.low_confidence_threshold", 7.5, low=0.0, high=1000.0)

    def split(self, blocks: List[Dict[str, Any]]) -> ThresholdSplit:
        high: List[Dict[str, Any]] = []
        low: List[Dict[str, Any]] = []
        for block in blocks:
            score = float(block.get("score", 0.0) or 0.0)
            if score >= self.threshold:
                high.append(block)
            else:
                low.append(block)
        return ThresholdSplit(high=high, low=low, threshold=self.threshold)
