"""Hybrid lexical + semantic fusion for DIC ranked blocks."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

from tag.config import AxiomConfig, load_config
from tag.dic.gbnf_dsl import query_tokens


WORD_RE = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.IGNORECASE)


@dataclass(frozen=True)
class FusionWeights:
    lexical: float
    semantic: float
    topology: float
    anchor: float


DEFAULT_TOPOLOGY_WEIGHTS: Dict[str, FusionWeights] = {
    "NEWS_ARTICLE": FusionWeights(lexical=0.56, semantic=0.28, topology=0.10, anchor=0.06),
    "ACADEMIC": FusionWeights(lexical=0.32, semantic=0.52, topology=0.10, anchor=0.06),
    "FORUM_POST": FusionWeights(lexical=0.42, semantic=0.42, topology=0.08, anchor=0.08),
    "GENERIC_HTML": FusionWeights(lexical=0.48, semantic=0.38, topology=0.06, anchor=0.08),
}


class HybridFusionRanker:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.anchor_domains = {
            normalize_domain(domain)
            for domain in self.config.get("dic.anchor_domains", [])
            if normalize_domain(str(domain))
        }

    def rank(self, query: str, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not blocks:
            return []
        query_terms = query_tokens(query)
        documents = [str(block.get("text") or "") for block in blocks]
        idf = self._idf(documents)
        enriched: List[Dict[str, Any]] = []
        for block in blocks:
            item = dict(block)
            text = str(item.get("text") or "")
            topology = str(item.get("topology_class") or "GENERIC_HTML")
            weights = self._weights_for(topology)
            lexical = bm25_score(query_terms, text, idf)
            semantic = semantic_char_score(query, text)
            topology_score = min(float(item.get("classification_confidence", 0.0) or 0.0), 1.0)
            anchor_score = 1.0 if self._is_anchor_domain(str(item.get("domain") or "")) else 0.0
            base = float(item.get("score", 0.0) or 0.0)
            fusion_score = (
                base
                + weights.lexical * lexical
                + weights.semantic * semantic
                + weights.topology * topology_score
                + weights.anchor * anchor_score
            )
            item["fusion"] = {
                "lexical": round(lexical, 4),
                "semantic": round(semantic, 4),
                "topology": round(topology_score, 4),
                "anchor": round(anchor_score, 4),
                "weights": weights.__dict__.copy(),
            }
            item["score"] = round(fusion_score, 4)
            enriched.append(item)
        enriched.sort(key=lambda item: (-float(item.get("score", 0.0)), str(item.get("url", ""))))
        for index, item in enumerate(enriched, start=1):
            item["rank"] = index
        return enriched

    def _weights_for(self, topology: str) -> FusionWeights:
        key = topology.upper()
        if key in DEFAULT_TOPOLOGY_WEIGHTS:
            return DEFAULT_TOPOLOGY_WEIGHTS[key]
        if "NEWS" in key or "ARTICLE" in key:
            return DEFAULT_TOPOLOGY_WEIGHTS["NEWS_ARTICLE"]
        if "ACADEMIC" in key or "PAPER" in key or "SCHOLAR" in key:
            return DEFAULT_TOPOLOGY_WEIGHTS["ACADEMIC"]
        if "FORUM" in key or "SOCIAL" in key:
            return DEFAULT_TOPOLOGY_WEIGHTS["FORUM_POST"]
        return DEFAULT_TOPOLOGY_WEIGHTS["GENERIC_HTML"]

    def _is_anchor_domain(self, domain: str) -> bool:
        normalized = normalize_domain(domain)
        return any(normalized == anchor or normalized.endswith(f".{anchor}") for anchor in self.anchor_domains)

    @staticmethod
    def _idf(documents: Iterable[str]) -> Dict[str, float]:
        docs = list(documents)
        doc_count = max(1, len(docs))
        df: Counter[str] = Counter()
        for document in docs:
            df.update(set(tokens(document)))
        return {
            term: math.log(1.0 + (doc_count - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }


def bm25_score(query_terms: List[str], document: str, idf: Mapping[str, float]) -> float:
    doc_terms = tokens(document)
    if not query_terms or not doc_terms:
        return 0.0
    counts = Counter(doc_terms)
    doc_len = len(doc_terms)
    avgdl = max(1.0, min(800.0, doc_len / 1.25))
    k1 = 1.4
    b = 0.72
    score = 0.0
    for term in query_terms:
        freq = counts.get(term, 0)
        if freq <= 0:
            continue
        denom = freq + k1 * (1.0 - b + b * doc_len / avgdl)
        score += idf.get(term, 0.1) * (freq * (k1 + 1.0)) / denom
    return min(score, 12.0)


def semantic_char_score(query: str, document: str) -> float:
    q_grams = char_grams(query)
    d_grams = char_grams(document[:3000])
    if not q_grams or not d_grams:
        return 0.0
    overlap = len(q_grams & d_grams)
    union = len(q_grams | d_grams)
    jaccard = overlap / max(1, union)
    phrase_bonus = 1.0 if query.lower() in document.lower() else 0.0
    return min(12.0, jaccard * 18.0 + phrase_bonus)


def tokens(text: str) -> List[str]:
    return [token.lower() for token in WORD_RE.findall(text)]


def char_grams(text: str, n: int = 4) -> set[str]:
    compact = re.sub(r"\s+", " ", text.lower()).strip()
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[index : index + n] for index in range(0, len(compact) - n + 1)}


def normalize_domain(domain: str) -> str:
    value = str(domain or "").strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0].strip(".")
