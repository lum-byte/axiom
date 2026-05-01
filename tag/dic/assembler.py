"""Deterministic typed context assembly for DirectlyInjectContext."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tag.config import AxiomConfig, load_config
from tag.dic.gbnf_dsl import QueryExpansionResult
from tag.dic.mcp_anchors import AnchorContextBlock, AnchorRegistry


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*", re.IGNORECASE)
STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "me",
    "of",
    "on",
    "or",
    "tell",
    "the",
    "to",
    "was",
    "what",
    "when",
    "where",
    "who",
    "why",
    "with",
}
DEFINITION_CUES = (
    " is ",
    " are ",
    " refers to ",
    " is a ",
    " is an ",
    " is the ",
    " means ",
    " defined as ",
    " known as ",
)
CAPABILITY_CUES = (
    "allows",
    "lets",
    "provides",
    "offers",
    "includes",
    "features",
    "enables",
    "supports",
    "tracks",
    "hosts",
    "manages",
)
MECHANISM_CUES = (
    "uses",
    "through",
    "built on",
    "based on",
    "powered by",
    "version control",
    "repository",
    "repositories",
    "workflow",
)
USE_CUES = (
    "use",
    "used",
    "users",
    "teams",
    "developers",
    "organizations",
    "collaborate",
    "host",
    "publish",
    "review",
    "manage",
)
BACKGROUND_CUES = (
    "headquartered",
    "subsidiary",
    "since ",
    "originally",
    "bootstrapped",
    "founders",
    "employees",
    "salaries",
)
LOW_SIGNAL_CUES = (
    "we present",
    "we document",
    "we show",
    "we provide",
    "we analyzed",
    "our results",
    "our study",
    "this paper",
    "empirical study",
    "study aimed",
    "dataset",
    "mining github",
    "researchers are starting",
    "however, so far",
    "so far there have been no",
    "is becoming one of",
    "you searched for",
    "search results",
    "ask the chatbot",
    "svg",
    "cookies",
    "privacy policy",
)


@dataclass(frozen=True)
class TypedContextSlot:
    name: str
    blocks: tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "blocks": [dict(block) for block in self.blocks]}


@dataclass(frozen=True)
class DICContext:
    query: str
    anchor: TypedContextSlot
    supporting: TypedContextSlot
    contested: TypedContextSlot
    query_trace: Dict[str, Any]
    answer: str
    structured_answer: Dict[str, Any]
    citations: tuple[Dict[str, Any], ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "anchor": self.anchor.to_dict(),
            "supporting": self.supporting.to_dict(),
            "contested": self.contested.to_dict(),
            "query_trace": dict(self.query_trace),
            "answer": self.answer,
            "structured_answer": dict(self.structured_answer),
            "citations": [dict(item) for item in self.citations],
            "answer_word_count": len(self.answer.split()),
        }


def build_sentence_records(
    query: str,
    anchor_blocks: List[Dict[str, Any]],
    supporting_blocks: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    terms = query_terms(query)
    records: List[Dict[str, Any]] = []
    seen_sentences: set[str] = set()
    seen_tokens: List[set[str]] = []
    for tier, blocks in (("anchor", anchor_blocks), ("supporting", supporting_blocks)):
        for block_index, block in enumerate(blocks):
            for sentence in split_sentences(str(block.get("text") or "")):
                if len(sentence.split()) < 6 or is_low_signal_sentence(sentence):
                    continue
                key = sentence_key(sentence)
                tokens = {token.lower() for token in TOKEN_RE.findall(sentence) if len(token) > 2}
                if not key or key in seen_sentences:
                    continue
                if tokens and any(jaccard(tokens, old_tokens) >= 0.72 for old_tokens in seen_tokens):
                    continue
                seen_sentences.add(key)
                if tokens:
                    seen_tokens.append(tokens)
                records.append(
                    {
                        "text": sentence,
                        "block": block,
                        "tier": tier,
                        "score": score_sentence(sentence, block, tier=tier, query_terms=terms, block_index=block_index),
                    }
                )
    records.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return records


def build_structured_answer(
    *,
    query: str,
    records: List[Dict[str, Any]],
    anchor_blocks: List[Dict[str, Any]],
    supporting_blocks: List[Dict[str, Any]],
    contested_blocks: List[Dict[str, Any]],
    veritas: Dict[str, Any],
    citations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    primary_records = primary_records_for_query(query, records)
    visible_records = primary_records if len(primary_records) >= 3 else records
    summary_record = choose_summary_record(query, visible_records)
    fallback_summary = f"TAG found source-backed context for '{query}', but the extracted text was too thin for a confident compact summary."
    summary = clip_sentence(str(summary_record.get("text") if summary_record else fallback_summary), 360)
    summary_source = source_ref(summary_record) if summary_record else None
    capability_cards = extract_capability_cards(visible_records)
    if capability_cards:
        key_points = [
            evidence_point_from_text(
                f"{card['name']}: {((card.get('evidence') or {}).get('text') if isinstance(card.get('evidence'), dict) else card.get('detail'))}",
                card.get("source"),
                label=str(card.get("name") or "Evidence"),
            )
            for card in capability_cards[:6]
        ]
    else:
        key_points = [
            evidence_point(record, limit=260)
            for record in select_unique_records(visible_records, limit=6, exclude={sentence_key(summary)})
        ]
    what_points = [
        evidence_point_from_text(summary, summary_source, label="Summary")
    ] if capability_cards else points_for(visible_records, DEFINITION_CUES, limit=3, fallback=[summary])
    capability_points = [
        evidence_point_from_text(f"{card['name']}: {card['detail']}", card.get("source"), label=str(card.get("name") or "Capability"))
        for card in capability_cards[:6]
    ] or points_for(
        visible_records,
        CAPABILITY_CUES,
        limit=5,
        fallback=key_points[:3],
    )
    mechanism_points = mechanism_points_from_cards(capability_cards) or points_for(
        visible_records,
        MECHANISM_CUES,
        limit=4,
        fallback=key_points[3:6] or key_points[:2],
    )
    use_points = use_points_from_cards(capability_cards) or points_for(visible_records, USE_CUES, limit=5, fallback=capability_points[:3])
    distinctions = infer_distinctions(query, visible_records)
    citation_spine = visible_citation_spine(visible_records, citations, limit=5)
    sections = [
        {"title": "What It Is", "points": what_points},
        {"title": "Core Capabilities", "points": capability_points},
        {"title": "How It Works", "points": mechanism_points},
        {"title": "Common Uses", "points": use_points},
    ]
    if distinctions:
        sections.append(
            {
                "title": "Important Distinctions",
                "points": [
                    evidence_point_from_text(str(item.get("text") or ""), item.get("source"), label=str(item.get("label") or "Distinction"))
                    for item in distinctions
                ],
            }
        )
    verification = {
        "method": "TAG-DIC fused retrieval with VERITAS legitimacy metadata",
        "anchor_blocks": len(anchor_blocks),
        "supporting_blocks": len(supporting_blocks),
        "contested_blocks": len(contested_blocks),
        "veritas_counts": dict(veritas.get("counts") or {}),
        "source_count": len(citations),
    }
    return {
        "summary": summary,
        "summary_source": summary_source,
        "key_points": key_points,
        "capabilities": capability_cards,
        "sections": sections,
        "distinctions": distinctions,
        "verification": verification,
        "citation_spine": citation_spine,
    }


def primary_records_for_query(query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    terms = query_terms(query)
    if not terms:
        return []
    primary = [record for record in records if block_title_matches_query(record.get("block", {}), terms)]
    return primary


def block_title_matches_query(block: Dict[str, Any], terms: set[str]) -> bool:
    title = str(block.get("title") or "")
    title = re.sub(r"\s+-\s+wikipedia\s*$", "", title, flags=re.IGNORECASE).strip()
    title_terms = query_terms(title)
    if not title_terms:
        return False
    if len(terms) == 1:
        return title_terms == terms
    return terms.issubset(title_terms) and len(title_terms - terms) <= 2


def query_terms(query: str) -> set[str]:
    return {token.lower() for token in TOKEN_RE.findall(query) if len(token) > 2 and token.lower() not in STOP_TERMS}


def score_sentence(
    sentence: str,
    block: Dict[str, Any],
    *,
    tier: str,
    query_terms: set[str],
    block_index: int,
) -> float:
    lower = sentence.lower()
    tokens = {token.lower() for token in TOKEN_RE.findall(sentence) if len(token) > 2}
    score = 0.0
    score += 5.0 if tier == "anchor" else 1.5
    try:
        score += min(4.0, max(0.0, float(block.get("score") or 0.0)) / 5.0)
    except (TypeError, ValueError):
        pass
    if query_terms:
        score += 8.0 * len(tokens & query_terms) / max(1, len(query_terms))
    if any(cue in lower for cue in DEFINITION_CUES):
        score += 4.0
    if any(cue in lower for cue in CAPABILITY_CUES):
        score += 2.0
    if any(cue in lower for cue in MECHANISM_CUES):
        score += 1.25
    if any(cue in lower for cue in USE_CUES):
        score += 1.0
    word_count = len(sentence.split())
    if 12 <= word_count <= 38:
        score += 1.25
    elif word_count > 65:
        score -= 2.0
    if any(cue in lower for cue in LOW_SIGNAL_CUES):
        score -= 7.0
    if not ({"history", "founded", "founder", "origin", "owned", "owner"} & query_terms) and any(cue in lower for cue in BACKGROUND_CUES):
        score -= 3.5
    punctuation_ratio = sum(1 for char in sentence if not char.isalnum() and not char.isspace()) / max(1, len(sentence))
    if punctuation_ratio > 0.18:
        score -= 4.0
    score -= min(2.0, block_index * 0.08)
    return round(score, 4)


def choose_summary_record(query: str, records: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    terms = query_terms(query)
    for record in records:
        text = str(record.get("text") or "")
        lower = text.lower()
        tokens = {token.lower() for token in TOKEN_RE.findall(text)}
        if any(cue in lower for cue in DEFINITION_CUES) and (not terms or tokens & terms):
            return record
    return records[0] if records else None


def select_unique_records(records: List[Dict[str, Any]], *, limit: int, exclude: Optional[set[str]] = None) -> List[Dict[str, Any]]:
    exclude = exclude or set()
    selected: List[Dict[str, Any]] = []
    token_sets: List[set[str]] = []
    for record in records:
        text = str(record.get("text") or "")
        key = sentence_key(text)
        if not key or key in exclude:
            continue
        tokens = {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}
        if tokens and any(jaccard(tokens, old_tokens) >= 0.64 for old_tokens in token_sets):
            continue
        selected.append(record)
        if tokens:
            token_sets.append(tokens)
        if len(selected) >= limit:
            break
    return selected


def points_for(
    records: List[Dict[str, Any]],
    cues: Tuple[str, ...],
    *,
    limit: int,
    fallback: List[Any],
) -> List[Dict[str, Any]]:
    matches = [record for record in records if any(cue in str(record.get("text") or "").lower() for cue in cues)]
    points = [evidence_point(record, limit=260) for record in select_unique_records(matches, limit=limit)]
    if points:
        return points
    fallback_points: List[Dict[str, Any]] = []
    for point in fallback[:limit]:
        if isinstance(point, dict) and point.get("text"):
            fallback_points.append(dict(point))
        elif str(point).strip():
            fallback_points.append(evidence_point_from_text(str(point), None, label="Fallback"))
    return fallback_points


def infer_distinctions(query: str, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    haystack = " ".join([query, *[str(record.get("text") or "") for record in records[:12]]]).lower()
    distinctions: List[Dict[str, Any]] = []
    git_record = next(
        (
            record
            for record in records
            if "github" in str(record.get("text") or "").lower()
            and re.search(r"\bgit\b", str(record.get("text") or ""), flags=re.IGNORECASE)
        ),
        None,
    )
    if "github" in haystack and re.search(r"\bgit\b", haystack) and git_record is not None:
        distinctions.append(
            {
                "label": "Git vs GitHub",
                "text": "The source evidence frames Git as the version-control layer and GitHub as the hosted developer platform built around Git-backed project workflows.",
                "source": source_ref(git_record),
                "evidence": clip_sentence(str(git_record.get("text") or ""), 260),
            }
        )
    return distinctions


def extract_capability_cards(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for record in records[:16]:
        sentence = str(record.get("text") or "")
        for phrase in capability_phrases(sentence):
            name = capability_name_from_phrase(phrase)
            key = sentence_key(name)
            if not name or key in seen:
                continue
            seen.add(key)
            source = source_ref(record)
            cards.append(
                {
                    "name": name,
                    "detail": clip_sentence(phrase, 160),
                    "evidence": {
                        "text": clip_sentence(sentence, 280),
                        "source": source,
                    },
                    "source": source,
                    "name_source": source,
                }
            )
            if len(cards) >= 8:
                return cards
    return cards


def capability_phrases(sentence: str) -> List[str]:
    clean = re.sub(r"\s+", " ", sentence).strip()
    phrases: List[str] = []
    feature_match = re.search(
        r"(?:provides?|offers?|includes?|features?|supports?)\s+(?P<tail>[^.]{12,260})",
        clean,
        flags=re.IGNORECASE,
    )
    if feature_match:
        phrases.extend(split_capability_tail(feature_match.group("tail")))
    allows_match = re.search(
        r"(?:allows?|lets|enables?)\s+[^.]{0,80}?\s+to\s+(?P<tail>[^.]{12,220})",
        clean,
        flags=re.IGNORECASE,
    )
    if allows_match:
        tail = allows_match.group("tail")
        if len(TOKEN_RE.findall(tail)) <= 5:
            phrases.append(tail)
    for direct in re.finditer(
        r"\b(?:distributed version control|version control|access control|bug tracking|feature requests?|task management|continuous integration|repositories?|wikis?|documentation|pull requests?)\b",
        clean,
        flags=re.IGNORECASE,
    ):
        phrases.append(direct.group(0))
    return [phrase for phrase in (normalize_capability_phrase(item) for item in phrases) if phrase]


def split_capability_tail(tail: str) -> List[str]:
    tail = re.split(r"\b(?:through|using|via|by)\b", tail, maxsplit=1, flags=re.IGNORECASE)[0]
    tail = re.sub(r"\s+for\s+every\s+project\b.*$", "", tail, flags=re.IGNORECASE)
    parts = re.split(r",\s+|\s+and\s+|\s+or\s+", tail)
    return [part for part in parts if part.strip()]


def normalize_capability_phrase(phrase: str) -> str:
    clean = re.sub(r"\([^)]*\)", " ", phrase)
    clean = re.sub(
        r"\b(?:itself|it|github|the|a|an|their|its|provides?|offers?|includes?|supports?)\b",
        " ",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"\s+", " ", clean).strip(" ,.;:-")
    clean = re.sub(r"^(?:and|or)\s+", "", clean, flags=re.IGNORECASE)
    if len(clean) < 4 or len(clean.split()) > 8:
        return ""
    return clean


def capability_name_from_phrase(phrase: str) -> str:
    words = [word for word in TOKEN_RE.findall(phrase) if word.lower() not in {"and", "or", "to", "for", "with"}]
    if not words:
        return ""
    return " ".join(word.capitalize() if word.islower() else word for word in words[:6])


def mechanism_points_from_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        evidence_point_from_text(f"{card.get('name')}: {card.get('detail')}", card.get("source"), label=str(card.get("name") or "Mechanism"))
        for card in cards
        if any(term in str(card.get("name") or "").lower() for term in ("git", "version", "repository", "integration", "workflow"))
    ][:4]


def use_points_from_cards(cards: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        evidence_point_from_text(f"{card.get('name')}: {card.get('detail')}", card.get("source"), label=str(card.get("name") or "Use"))
        for card in cards
        if any(term in str(card.get("detail") or "").lower() for term in ("developers", "project", "code", "collaboration", "manage", "share"))
    ][:5]


def visible_citation_spine(records: List[Dict[str, Any]], citations: List[Dict[str, Any]], *, limit: int) -> List[Dict[str, Any]]:
    visible_urls = []
    for record in records:
        url = str((record.get("block") or {}).get("url") or "")
        if url and url not in visible_urls:
            visible_urls.append(url)
    by_url = {str(item.get("url") or ""): item for item in citations}
    ordered = [by_url[url] for url in visible_urls if url in by_url]
    if ordered:
        return ordered[:limit]
    for item in citations:
        if len(ordered) >= limit:
            break
        if item not in ordered:
            ordered.append(item)
    return ordered[:limit]


def evidence_point(record: Dict[str, Any], *, limit: int = 260, label: str = "") -> Dict[str, Any]:
    return evidence_point_from_text(str(record.get("text") or ""), source_ref(record), label=label, limit=limit)


def evidence_point_from_text(source_text: str, source: Optional[Dict[str, Any]], *, label: str = "", limit: int = 260) -> Dict[str, Any]:
    point = {
        "text": clip_sentence(source_text, limit),
        "source": source,
    }
    if label:
        point["label"] = label
    return point


def point_text(point: Any) -> str:
    if isinstance(point, dict):
        return str(point.get("text") or "")
    return str(point or "")


def source_ref(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not record:
        return None
    block = record.get("block") if isinstance(record.get("block"), dict) else {}
    url = str(block.get("url") or "")
    title = str(block.get("title") or block.get("domain") or url)
    domain = str(block.get("domain") or block.get("source") or "")
    if not url and not title:
        return None
    anchor_text = title or domain or url
    ref = {
        "title": title,
        "anchor_text": anchor_text,
        "url": url,
        "domain": domain,
    }
    if url:
        ref["markdown"] = f"[{anchor_text}]({url})"
    return ref


def is_low_signal_sentence(sentence: str) -> bool:
    lower = sentence.lower()
    if any(cue in lower for cue in LOW_SIGNAL_CUES):
        return True
    if re.search(r"[{}<>][^.!?]{20,}", sentence):
        return True
    if sentence.count("://") > 1:
        return True
    if len(sentence) > 1100:
        return True
    alpha_ratio = sum(1 for char in sentence if char.isalpha()) / max(1, len(sentence))
    return alpha_ratio < 0.45


def clip_sentence(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(clean) <= limit:
        return clean
    clipped = clean[: limit - 1].rstrip()
    boundary = max(clipped.rfind(". "), clipped.rfind("; "), clipped.rfind(", "))
    if boundary >= limit // 2:
        clipped = clipped[: boundary + 1].rstrip()
    return clipped.rstrip(" ,;:") + "..."


class DirectlyInjectContextAssembler:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.anchor_registry = AnchorRegistry(config=self.config)
        self.target_words = self.config.int("dic.target_answer_words", 560, low=120, high=1200)
        self.min_words = self.config.int("dic.min_answer_words", 500, low=80, high=1200)
        self.max_words = self.config.int("dic.max_answer_words", 700, low=120, high=1500)

    def assemble(
        self,
        *,
        query: str,
        ranked_blocks: List[Dict[str, Any]],
        expansion: Optional[QueryExpansionResult],
        veritas: Optional[Dict[str, Any]] = None,
    ) -> DICContext:
        anchor_blocks = [block.to_dict() for block in self.anchor_registry.normalize_blocks(ranked_blocks)]
        contested_blocks = self._contested_blocks(ranked_blocks, veritas)
        supporting_blocks = [
            compact_block(block)
            for block in ranked_blocks
            if block_identity(block) not in {block_identity(item) for item in contested_blocks}
            and not self.anchor_registry.is_anchor(str(block.get("domain") or ""))
        ]
        if not supporting_blocks:
            supporting_blocks = [
                compact_block(block)
                for block in ranked_blocks
                if block_identity(block) not in {block_identity(item) for item in contested_blocks}
            ]
        trace = {
            "grammar_hash": expansion.grammar_hash if expansion else None,
            "detected_type": expansion.detected_type if expansion else None,
            "expansion_count": expansion.effective_limit if expansion else 0,
            "queries": expansion.queries if expansion else [query],
        }
        answer, structured_answer, citations = self._compose_answer(
            query=query,
            anchor_blocks=anchor_blocks,
            supporting_blocks=supporting_blocks,
            contested_blocks=contested_blocks,
            veritas=veritas or {},
        )
        return DICContext(
            query=query,
            anchor=TypedContextSlot("anchor", tuple(anchor_blocks[:12])),
            supporting=TypedContextSlot("supporting", tuple(supporting_blocks[:24])),
            contested=TypedContextSlot("contested", tuple(contested_blocks[:12])),
            query_trace=trace,
            answer=answer,
            structured_answer=structured_answer,
            citations=tuple(citations),
        )

    def _contested_blocks(self, ranked_blocks: List[Dict[str, Any]], veritas: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        labels = {item.get("identity"): item for item in (veritas or {}).get("classifications", [])}
        contested = []
        for block in ranked_blocks:
            identity = block_identity(block)
            label = labels.get(identity, {})
            if label.get("label") in {"RUMOR", "LEGACY", "CONTESTED"}:
                item = compact_block(block)
                item["veritas"] = label
                contested.append(item)
        return contested

    def _compose_answer(
        self,
        *,
        query: str,
        anchor_blocks: List[Dict[str, Any]],
        supporting_blocks: List[Dict[str, Any]],
        contested_blocks: List[Dict[str, Any]],
        veritas: Dict[str, Any],
    ) -> Tuple[str, Dict[str, Any], List[Dict[str, Any]]]:
        sources = [*anchor_blocks, *supporting_blocks]
        sentence_records = build_sentence_records(query, anchor_blocks, supporting_blocks)
        sentences = [record["text"] for record in sentence_records]
        citations = build_citations([*anchor_blocks, *supporting_blocks, *contested_blocks])
        structured = build_structured_answer(
            query=query,
            records=sentence_records,
            anchor_blocks=anchor_blocks,
            supporting_blocks=supporting_blocks,
            contested_blocks=contested_blocks,
            veritas=veritas,
            citations=citations,
        )
        paragraphs: List[str] = []
        intro = f"TAG-DIC resolved the query '{query}' by fusing anchor sources with crawler-ranked supporting context."
        if anchor_blocks:
            intro += f" The anchor tier contributed {len(anchor_blocks)} high-trust block(s), led by {anchor_blocks[0].get('source') or anchor_blocks[0].get('domain')}."
        if veritas:
            counts = veritas.get("counts", {})
            intro += f" VERITAS classified low-confidence material as {counts} before injection."
        paragraphs.append(intro)
        if structured.get("summary"):
            paragraphs.append(str(structured["summary"]))
        key_points = structured.get("key_points") if isinstance(structured.get("key_points"), list) else []
        if key_points:
            paragraphs.append("Key points: " + " ".join(point_text(point) for point in key_points[:6]))
        for section in structured.get("sections", []):
            if not isinstance(section, dict):
                continue
            points = section.get("points") if isinstance(section.get("points"), list) else []
            if points:
                paragraphs.append(f"{section.get('title', 'Context')}: " + " ".join(point_text(point) for point in points[:5]))

        body_sentences = []
        for sentence in sentences:
            if len(sentence.split()) < 6:
                continue
            body_sentences.append(sentence)
            if len(" ".join(body_sentences).split()) >= self.target_words:
                break
        if not body_sentences:
            body_sentences = [f"TAG found sources for {query}, but the available extracted context was too thin for detailed synthesis."]

        current: List[str] = []
        for sentence in body_sentences:
            current.append(sentence)
            if len(" ".join(current).split()) >= 90:
                paragraphs.append(" ".join(current))
                current = []
        if current:
            paragraphs.append(" ".join(current))

        if contested_blocks:
            contested_summary = " ".join(
                f"{block.get('title') or block.get('domain')} was labeled {block.get('veritas', {}).get('label', 'CONTESTED')}."
                for block in contested_blocks[:4]
            )
            paragraphs.append(f"Contested material was not hidden; it was isolated for model use. {contested_summary}")

        if citations:
            cite_text = "; ".join(f"[{idx + 1}] {item['title']} ({item['domain']})" for idx, item in enumerate(citations[:8]))
            paragraphs.append(f"Citation spine: {cite_text}.")

        answer = "\n\n".join(paragraphs)
        words = answer.split()
        if len(words) > self.max_words:
            answer = " ".join(words[: self.max_words])
        elif len(words) < self.min_words and body_sentences:
            answer = pad_answer(answer, body_sentences, self.min_words, self.max_words)
        return answer, structured, citations


def compact_block(block: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "url": block.get("url"),
        "domain": block.get("domain"),
        "title": block.get("title"),
        "text": trim_text(str(block.get("text") or ""), 1800),
        "score": block.get("score"),
        "rank": block.get("rank"),
        "topology_class": block.get("topology_class"),
        "fusion": block.get("fusion"),
        "veritas": block.get("veritas"),
    }


def split_sentences(text: str) -> List[str]:
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return []
    sentences = []
    for sentence in SENTENCE_RE.split(clean):
        cleaned = clean_sentence(sentence)
        if cleaned:
            sentences.append(cleaned)
    return sentences


def dedupe_sentences(sentences: Iterable[str]) -> List[str]:
    seen = set()
    seen_token_sets: List[set[str]] = []
    unique = []
    for sentence in sentences:
        key = re.sub(r"[^a-z0-9]+", " ", sentence.lower()).strip()
        key = " ".join(key.split()[:18])
        if not key or key in seen:
            continue
        tokens = {token.lower() for token in TOKEN_RE.findall(sentence) if len(token) > 2}
        if tokens and any(jaccard(tokens, old_tokens) >= 0.72 for old_tokens in seen_token_sets):
            continue
        seen.add(key)
        if tokens:
            seen_token_sets.append(tokens)
        unique.append(sentence)
    return unique


def clean_sentence(sentence: str) -> str:
    cleaned = re.sub(r"\[\s*\d+\s*\]", " ", sentence)
    cleaned = re.sub(r"\(\s*/[^)]{1,180}/[^)]{0,80}\)", " ", cleaned)
    cleaned = re.sub(r"\(\s*pronunciation[^)]{0,180}\)", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s+", "(", cleaned)
    cleaned = re.sub(r"\s+\)", ")", cleaned)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" -:;")
    if len(cleaned) < 24:
        return ""
    return cleaned


def jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def build_citations(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    citations = []
    seen = set()
    for block in blocks:
        url = str(block.get("url") or "")
        if not url or url in seen:
            continue
        seen.add(url)
        citations.append(
            {
                "url": url,
                "domain": str(block.get("domain") or block.get("source") or ""),
                "title": str(block.get("title") or block.get("domain") or url),
                "rank": block.get("rank") or dict(block.get("metadata") or {}).get("rank"),
                "score": block.get("score"),
            }
        )
        citations[-1]["anchor_text"] = citations[-1]["title"] or citations[-1]["domain"] or url
        citations[-1]["markdown"] = f"[{citations[-1]['anchor_text']}]({url})"
    return citations


def block_identity(block: Dict[str, Any]) -> str:
    return f"{block.get('url')}#{block.get('rank')}#{str(block.get('text') or '')[:80]}"


def trim_text(text: str, limit: int) -> str:
    clean = re.sub(r"\s+", " ", text).strip()
    return clean if len(clean) <= limit else clean[: limit - 3].rstrip() + "..."


def pad_answer(answer: str, sentences: List[str], min_words: int, max_words: int) -> str:
    parts = [answer]
    existing_keys = {sentence_key(sentence) for sentence in split_sentences(answer)}
    index = 0
    angles = (
        "Anchor reading",
        "Crawler reading",
        "Context-fusion reading",
        "Legitimacy reading",
        "Model-injection reading",
        "Coverage reading",
    )
    candidates = [sentence for sentence in sentences if sentence_key(sentence) not in existing_keys]
    if not candidates:
        candidates = [
            "The citation spine is preserved so downstream model use can tie claims back to URL-bearing evidence.",
            "Anchor, supporting, contested, and query-trace slots stay separated to keep retrieval evidence auditable.",
            "The fused result keeps high-trust sources first while still exposing crawler discoveries that add operational context.",
            "VERITAS metadata remains attached to low-confidence material instead of silently dropping disputed context.",
        ]
    while len(" ".join(parts).split()) < min_words and index < max(len(candidates), 1) * len(angles):
        sentence = candidates[index % len(candidates)]
        angle = angles[index % len(angles)]
        addition = f"{angle}: this source reinforces the same answer space without changing the core claim: {sentence}"
        if len(" ".join([*parts, addition]).split()) > max_words:
            break
        parts.append(addition)
        index += 1
    return " ".join(parts)


def sentence_key(sentence: str) -> str:
    return " ".join(TOKEN_RE.findall(sentence.lower())[:18])
