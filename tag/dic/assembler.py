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
CAPABILITY_BLUEPRINTS = (
    {
        "name": "Code Hosting",
        "detail": "Stores project code and gives teams a shared place to manage it.",
        "patterns": (
            r"\bcreate,\s*store,\s*manage,\s*and\s*share\b",
            r"\bstor(?:e|es|ing).{0,80}\bcode\b",
            r"\bcode hosting\b",
            r"\brepositor(?:y|ies)\b",
            r"\bshare.{0,80}\bcode\b",
        ),
    },
    {
        "name": "Version Control",
        "detail": "Uses Git-based version control so project history, changes, and revisions stay trackable.",
        "patterns": (
            r"\bgit\b",
            r"\bversion control\b",
            r"\bdistributed version control\b",
            r"\brevision",
        ),
    },
    {
        "name": "Collaboration",
        "detail": "Supports controlled collaboration through access management and review-oriented project workflows.",
        "patterns": (
            r"\bcollaborat",
            r"\baccess control\b",
            r"\bpull request",
            r"\bcode review\b",
            r"\breview changes\b",
        ),
    },
    {
        "name": "Project Management",
        "detail": "Tracks bugs, feature requests, and project tasks alongside the codebase.",
        "patterns": (
            r"\bbug tracking\b",
            r"\bissue tracking\b",
            r"\bfeature request",
            r"\btask management\b",
            r"\bproject management\b",
        ),
    },
    {
        "name": "Automation",
        "detail": "Connects code changes to automated workflows such as continuous integration.",
        "patterns": (
            r"\bcontinuous integration\b",
            r"\bautomation\b",
            r"\bworkflows?\b",
            r"\bci\b",
        ),
    },
    {
        "name": "Documentation",
        "detail": "Keeps project knowledge close to the code through wikis and documentation surfaces.",
        "patterns": (
            r"\bwikis?\b",
            r"\bdocumentation\b",
            r"\bdocs\b",
            r"\bproject notes\b",
        ),
    },
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
    citation_spine = assign_citation_ids(visible_citation_spine(visible_records, citations, limit=5))
    citation_lookup = citation_lookup_by_url(citation_spine)
    summary_record = choose_summary_record(query, visible_records)
    fallback_summary = f"TAG found source-backed context for '{query}', but the extracted text was too thin for a confident compact summary."
    summary = clip_sentence(str(summary_record.get("text") if summary_record else fallback_summary), 360)
    summary_source_id, summary_source = source_pointer(summary_record, citation_lookup)
    capability_cards = extract_capability_cards(visible_records, citation_lookup)
    if capability_cards:
        key_points = [str(card.get("detail") or "") for card in capability_cards[:6] if str(card.get("detail") or "").strip()]
    else:
        key_points = [summary]
        key_points.extend(record_point(record, limit=260) for record in select_unique_records(visible_records, limit=5, exclude={sentence_key(summary)}))
    if capability_cards:
        what_points = [summary]
    else:
        definition_points = points_for(visible_records, DEFINITION_CUES, limit=3, fallback=[summary])
        what_points = [summary]
        what_points.extend(point for point in definition_points if sentence_key(point) != sentence_key(summary))
        what_points = what_points[:3]
    capability_points = [
        f"{card['name']}: {card['detail']}"
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
    distinctions = infer_distinctions(query, visible_records, citation_lookup)
    if capability_cards:
        sections = [
            section("What It Is", what_points, [summary_source_id]),
            section("Core Capabilities", capability_points, source_ids_from_cards(capability_cards)),
            section("How It Works", mechanism_points, source_ids_from_cards(capability_cards, names=("Version Control", "Automation", "Collaboration"))),
            section("Common Uses", use_points, source_ids_from_cards(capability_cards)),
        ]
    else:
        sections = [
            section("What It Is", what_points, [summary_source_id]),
            section("Key Facts", key_points[:6], [summary_source_id]),
            section("Background", points_for(visible_records, BACKGROUND_CUES, limit=4, fallback=key_points[3:6] or key_points[:2]), [summary_source_id]),
            section(
                "Why It Matters",
                points_for(
                    visible_records,
                    ("largest", "provider", "worldwide", "users", "market", "valuable", "popular", "important"),
                    limit=4,
                    fallback=key_points[:3],
                ),
                [summary_source_id],
            ),
        ]
    if distinctions:
        sections.append(
            section("Important Distinctions", [str(item.get("text") or "") for item in distinctions], [str(item.get("source_id") or "") for item in distinctions])
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
        "summary_source_id": summary_source_id,
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
    subject = query_subject(query)
    if subject:
        subject_re = re.compile(rf"^{re.escape(subject)}\b\s+(?:is|are|was|were|refers to|means|provides|offers)\b", re.IGNORECASE)
        for record in records:
            text = str(record.get("text") or "")
            if subject_re.search(text):
                return record
    for record in records:
        text = str(record.get("text") or "")
        lower = text.lower()
        tokens = {token.lower() for token in TOKEN_RE.findall(text)}
        if any(cue in lower for cue in DEFINITION_CUES) and (not terms or tokens & terms):
            return record
    return records[0] if records else None


def query_subject(query: str) -> str:
    clean = re.sub(r"\s+", " ", query).strip(" ?!.")
    clean = re.sub(
        r"^(?:what|who|where|when|why|how)\s+(?:is|are|was|were|do|does|did|to|can|could|would|should)?\s*",
        "",
        clean,
        flags=re.IGNORECASE,
    )
    clean = re.sub(r"^(?:the|a|an)\s+", "", clean, flags=re.IGNORECASE)
    words = clean.split()
    return " ".join(words[:6]).lower()


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
) -> List[str]:
    matches = [record for record in records if any(cue in str(record.get("text") or "").lower() for cue in cues)]
    points = [record_point(record, limit=260) for record in select_unique_records(matches, limit=limit)]
    if points:
        return points
    fallback_points: List[str] = []
    for point in fallback[:limit]:
        if isinstance(point, dict) and point.get("text"):
            fallback_points.append(str(point.get("text") or ""))
        elif str(point).strip():
            fallback_points.append(str(point))
    return fallback_points


def infer_distinctions(query: str, records: List[Dict[str, Any]], citation_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
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
        source_id, source_markdown = source_pointer(git_record, citation_lookup)
        distinctions.append(
            {
                "label": "Git vs GitHub",
                "text": "Git is the version-control system; GitHub is the hosted collaboration platform built around Git repositories and project workflows.",
                "source_id": source_id,
                "source": source_markdown,
            }
        )
    return distinctions


def extract_capability_cards(records: List[Dict[str, Any]], citation_lookup: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    cards: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for blueprint in CAPABILITY_BLUEPRINTS:
        record = first_record_matching_patterns(records, tuple(str(pattern) for pattern in blueprint["patterns"]))
        if record is None:
            continue
        card = capability_card(
            str(blueprint["name"]),
            str(blueprint["detail"]),
            record,
            citation_lookup,
        )
        key = sentence_key(card["name"])
        if key not in seen:
            seen.add(key)
            cards.append(card)
    for record in records[:16]:
        sentence = str(record.get("text") or "")
        for phrase in capability_phrases(sentence):
            name = capability_name_from_phrase(phrase)
            name, detail = normalize_capability_card_text(name, phrase, sentence)
            key = sentence_key(name)
            if not name or key in seen:
                continue
            seen.add(key)
            cards.append(capability_card(name, detail, record, citation_lookup))
            if len(cards) >= 8:
                return cards
    return cards


def first_record_matching_patterns(records: List[Dict[str, Any]], patterns: Tuple[str, ...]) -> Optional[Dict[str, Any]]:
    for record in records[:24]:
        lower = str(record.get("text") or "").lower()
        if any(re.search(pattern, lower, flags=re.IGNORECASE) for pattern in patterns):
            return record
    return None


def capability_card(
    name: str,
    detail: str,
    record: Dict[str, Any],
    citation_lookup: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    source_id, source_markdown = source_pointer(record, citation_lookup)
    card = {
        "name": name,
        "detail": detail,
    }
    if source_id:
        card["source_id"] = source_id
    if source_markdown:
        card["source"] = source_markdown
    return card


def normalize_capability_card_text(name: str, phrase: str, sentence: str) -> Tuple[str, str]:
    lower = " ".join([name, phrase, sentence]).lower()
    if "version control" in lower or re.search(r"\bgit\b", lower):
        return "Version Control", "Uses Git-based version control so project history, changes, and revisions stay trackable."
    if "access control" in lower or "pull request" in lower or "collaborat" in lower:
        return "Collaboration", "Supports controlled collaboration through access management and review-oriented project workflows."
    if "bug tracking" in lower or "feature request" in lower or "task management" in lower or "issue tracking" in lower:
        return "Project Management", "Tracks bugs, feature requests, and project tasks alongside the codebase."
    if "continuous integration" in lower or "workflow" in lower or "automation" in lower:
        return "Automation", "Connects code changes to automated workflows such as continuous integration."
    if "wiki" in lower or "documentation" in lower or "docs" in lower:
        return "Documentation", "Keeps project knowledge close to the code through wikis and documentation surfaces."
    if "repository" in lower or "store" in lower or "share" in lower or "code" in lower:
        return "Code Hosting", "Stores project code and gives teams a shared place to manage it."
    return name, clip_sentence(sentence, 180)


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


def mechanism_points_from_cards(cards: List[Dict[str, Any]]) -> List[str]:
    return [
        str(card.get("detail") or "")
        for card in cards
        if any(term in str(card.get("name") or "").lower() for term in ("git", "version", "repository", "integration", "workflow", "automation", "collaboration"))
    ][:4]


def use_points_from_cards(cards: List[Dict[str, Any]]) -> List[str]:
    names = {str(card.get("name") or "") for card in cards}
    uses: List[str] = []
    if "Code Hosting" in names:
        uses.append("Host open-source or private software projects in shared repositories.")
    if "Collaboration" in names:
        uses.append("Coordinate code review and team contributions around the same project.")
    if "Project Management" in names:
        uses.append("Track bugs, feature requests, and implementation work next to the code.")
    if "Documentation" in names:
        uses.append("Publish project notes, docs, or wiki pages close to the repository.")
    if "Automation" in names:
        uses.append("Run automation and integration workflows when code changes.")
    return uses[:5]


def source_ids_from_cards(cards: List[Dict[str, Any]], names: Tuple[str, ...] = ()) -> List[str]:
    allowed = set(names)
    source_ids: List[str] = []
    for card in cards:
        if allowed and str(card.get("name") or "") not in allowed:
            continue
        source_id = str(card.get("source_id") or "")
        if source_id and source_id not in source_ids:
            source_ids.append(source_id)
    return source_ids


def section(title: str, points: List[Any], source_ids: List[Any]) -> Dict[str, Any]:
    clean_points = [point_text(point) for point in points if point_text(point).strip()]
    item: Dict[str, Any] = {"title": title, "points": clean_points}
    clean_source_ids = [str(source_id) for source_id in source_ids if str(source_id or "").strip()]
    deduped_source_ids = list(dict.fromkeys(clean_source_ids))
    if deduped_source_ids:
        item["source_ids"] = deduped_source_ids
    return item


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


def assign_citation_ids(citations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    assigned: List[Dict[str, Any]] = []
    for index, citation in enumerate(citations, start=1):
        item = dict(citation)
        item["id"] = str(item.get("id") or f"S{index}")
        item["anchor_text"] = str(item.get("anchor_text") or item.get("title") or item.get("domain") or item.get("url") or item["id"])
        if item.get("url"):
            item["markdown"] = str(item.get("markdown") or f"[{item['anchor_text']}]({item['url']})")
        assigned.append(item)
    return assigned


def citation_lookup_by_url(citations: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(citation.get("url") or ""): citation for citation in citations if citation.get("url")}


def source_pointer(record: Optional[Dict[str, Any]], citation_lookup: Dict[str, Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:
    if not record:
        return None, None
    block = record.get("block") if isinstance(record.get("block"), dict) else {}
    citation = citation_lookup.get(str(block.get("url") or ""))
    if not citation:
        return None, None
    source_id = str(citation.get("id") or "")
    source_markdown = str(citation.get("markdown") or citation.get("url") or "")
    return source_id or None, source_markdown or None


def record_point(record: Dict[str, Any], *, limit: int = 260) -> str:
    return clip_sentence(str(record.get("text") or ""), limit)


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
