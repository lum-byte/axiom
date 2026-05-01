"""Adversarial confirm/deny scoring for VERITAS."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List

from tag.dic.hybrid_search import semantic_char_score, tokens


NEGATION_TERMS = {
    "false",
    "fake",
    "hoax",
    "rumor",
    "rumour",
    "debunk",
    "denied",
    "incorrect",
    "misleading",
    "outdated",
    "no longer",
    "not true",
    "contradicts",
}


@dataclass(frozen=True)
class AdversarialResult:
    confirm_score: float
    deny_score: float
    confirming_sources: tuple[str, ...]
    denying_sources: tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "confirm_score": round(self.confirm_score, 4),
            "deny_score": round(self.deny_score, 4),
            "confirming_sources": list(self.confirming_sources),
            "denying_sources": list(self.denying_sources),
        }


class AdversarialCrawlerPair:
    def argue(self, claim: Dict[str, Any], anchors: Iterable[Dict[str, Any]]) -> AdversarialResult:
        claim_text = str(claim.get("text") or "")
        confirming: List[str] = []
        denying: List[str] = []
        confirm_score = 0.0
        deny_score = 0.0
        claim_terms = set(tokens(claim_text))
        for anchor in anchors:
            anchor_text = str(anchor.get("text") or "")
            source = str(anchor.get("domain") or anchor.get("source") or anchor.get("url") or "")
            overlap = len(claim_terms & set(tokens(anchor_text))) / max(1, len(claim_terms))
            semantic = semantic_char_score(claim_text[:500], anchor_text[:2000]) / 12.0
            score = overlap * 0.65 + semantic * 0.35
            if score <= 0:
                continue
            if contains_denial(anchor_text):
                deny_score += score
                denying.append(source)
            else:
                confirm_score += score
                confirming.append(source)
        return AdversarialResult(
            confirm_score=confirm_score,
            deny_score=deny_score,
            confirming_sources=tuple(sorted(set(confirming))),
            denying_sources=tuple(sorted(set(denying))),
        )


def contains_denial(text: str) -> bool:
    lowered = re.sub(r"\s+", " ", str(text or "").lower())
    return any(term in lowered for term in NEGATION_TERMS)
