"""
GBNF-backed query DSL and expansion engine for TAG-DIC.

The engine does three jobs:
1. Parse CLI expansion directives such as `exp -10`.
2. Classify the natural-language question into one of 100+ query programs.
3. Produce typed crawler directives that TAG can execute deterministically.

The taxonomy is code, but the fanout, caps, and enabled state are config driven.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from tag.config import AxiomConfig, load_config


EXP_SEGMENT_RE = re.compile(r"^\s*(?:exp|expand|expansion)\s+-?(?P<count>\d{1,3})\s*$", re.IGNORECASE)
EXP_TEXT_RE = re.compile(r"\b(?:exp|expand|expansion)\s*-(?P<count>\d{1,3})\b", re.IGNORECASE)
TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_+'-]*", re.IGNORECASE)


@dataclass(frozen=True)
class QueryTypeSpec:
    name: str
    priority: int
    patterns: tuple[str, ...]
    templates: tuple[str, ...]
    anchor_bias: tuple[str, ...] = ()
    topology_bias: str = "GENERIC_HTML"
    temporal: bool = False
    adversarial: bool = False

    def production(self) -> str:
        alternatives = " | ".join(f'"{pattern}"' for pattern in self.patterns[:4])
        if not alternatives:
            alternatives = '"<freeform>"'
        return f'{self.name.lower()} ::= ({alternatives}) ws subject'


@dataclass(frozen=True)
class QueryDirective:
    query: str
    query_type: str
    directive: str
    priority: int
    trace: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "query_type": self.query_type,
            "directive": self.directive,
            "priority": self.priority,
            "trace": dict(self.trace),
        }


@dataclass(frozen=True)
class QueryExpansionResult:
    original_query: str
    requested_limit: int
    effective_limit: int
    detected_type: str
    directives: tuple[QueryDirective, ...]
    grammar_hash: str

    @property
    def queries(self) -> List[str]:
        return [directive.query for directive in self.directives]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_query": self.original_query,
            "requested_limit": self.requested_limit,
            "effective_limit": self.effective_limit,
            "detected_type": self.detected_type,
            "grammar_hash": self.grammar_hash,
            "directives": [directive.to_dict() for directive in self.directives],
        }


QUESTION_TYPE_SPECS: tuple[QueryTypeSpec, ...] = (
    QueryTypeSpec("FACTUAL_DIRECT", 100, ("what is", "who is", "where is"), ("{subject}", "definition of {subject}", "{subject} overview"), ("wikipedia", "news")),
    QueryTypeSpec("FACTUAL_TEMPORAL", 99, ("what was", "status in", "during"), ("{subject} history", "{subject} timeline", "{subject} in {year}"), ("wikipedia", "news"), temporal=True),
    QueryTypeSpec("CAUSAL", 98, ("why did", "why does", "cause of"), ("why {subject}", "{subject} causes", "{subject} explanation"), ("news", "scholar")),
    QueryTypeSpec("COMPARATIVE", 97, (" vs ", "versus", "compare"), ("{subject} comparison", "{subject} differences", "{subject} tradeoffs"), ("wikipedia", "scholar")),
    QueryTypeSpec("PROCEDURAL", 96, ("how to", "steps to", "guide"), ("how to {subject}", "{subject} steps", "{subject} guide"), ("docs", "wikipedia")),
    QueryTypeSpec("DEFINITIONAL", 95, ("define", "meaning of", "in context"), ("define {subject}", "{subject} meaning", "{subject} definition context"), ("wikipedia",)),
    QueryTypeSpec("EXISTENCE_CHECK", 94, ("does", "is real", "exists"), ("does {subject} exist", "{subject} real evidence", "{subject} official"), ("wikipedia", "news"), adversarial=True),
    QueryTypeSpec("RECENCY_CHECK", 93, ("latest", "recent", "today"), ("latest {subject}", "{subject} news", "{subject} recent update"), ("news",), temporal=True),
    QueryTypeSpec("ATTRIBUTION", 92, ("who said", "who did", "attributed to"), ("who said {subject}", "{subject} attribution", "{subject} source"), ("news", "wikipedia")),
    QueryTypeSpec("CONTRADICTION_PROBE", 91, ("evidence against", "contradict", "debunk"), ("evidence against {subject}", "{subject} contradiction", "{subject} debunked"), ("news", "scholar"), adversarial=True),
    QueryTypeSpec("LEGACY_PROBE", 90, ("was true", "used to", "before"), ("was {subject} true", "{subject} old status", "{subject} changed"), ("wayback", "news"), temporal=True, adversarial=True),
    QueryTypeSpec("RUMOR_PROBE", 89, ("rumor", "unverified", "claim"), ("{subject} rumor", "{subject} unverified claim", "{subject} fact check"), ("news",), adversarial=True),
    QueryTypeSpec("NUMERIC_VALUE", 88, ("how many", "how much", "count of"), ("{subject} number", "{subject} count", "{subject} statistics"), ("news", "scholar")),
    QueryTypeSpec("DATE_LOOKUP", 87, ("when did", "date of", "year of"), ("when did {subject}", "{subject} date", "{subject} timeline"), ("wikipedia", "news"), temporal=True),
    QueryTypeSpec("LOCATION_LOOKUP", 86, ("where did", "located in", "location of"), ("where is {subject}", "{subject} location", "{subject} geography"), ("wikipedia", "news")),
    QueryTypeSpec("PERSON_PROFILE", 85, ("who is", "biography", "profile"), ("{subject} biography", "{subject} profile", "{subject} official"), ("wikipedia", "news")),
    QueryTypeSpec("ORGANIZATION_PROFILE", 84, ("company", "organization", "founded"), ("{subject} company", "{subject} organization", "{subject} founded"), ("wikipedia", "news")),
    QueryTypeSpec("PRODUCT_PROFILE", 83, ("product", "tool", "platform"), ("{subject} product", "{subject} platform", "{subject} features"), ("wikipedia", "news", "docs")),
    QueryTypeSpec("TECHNICAL_API", 82, ("api", "sdk", "library"), ("{subject} API", "{subject} documentation", "{subject} SDK"), ("docs", "github")),
    QueryTypeSpec("ERROR_DEBUG", 81, ("error", "traceback", "failed"), ("{subject} error", "{subject} fix", "{subject} issue"), ("docs", "github")),
    QueryTypeSpec("SECURITY_ADVISORY", 80, ("cve", "vulnerability", "exploit"), ("{subject} CVE", "{subject} vulnerability", "{subject} advisory"), ("news", "nist")),
    QueryTypeSpec("LEGAL_STATUS", 79, ("legal", "law", "regulation"), ("{subject} law", "{subject} legal status", "{subject} regulation"), ("news", "government")),
    QueryTypeSpec("MEDICAL_SUMMARY", 78, ("symptom", "treatment", "disease"), ("{subject} medical overview", "{subject} treatment", "{subject} clinical"), ("scholar", "government")),
    QueryTypeSpec("FINANCIAL_STATUS", 77, ("stock", "earnings", "market"), ("{subject} earnings", "{subject} market", "{subject} financials"), ("news",)),
    QueryTypeSpec("SPORTS_RESULT", 76, ("score", "standings", "match"), ("{subject} score", "{subject} standings", "{subject} result"), ("news",)),
    QueryTypeSpec("EVENT_RECAP", 75, ("what happened", "recap", "incident"), ("{subject} recap", "{subject} what happened", "{subject} timeline"), ("news", "wikipedia"), temporal=True),
    QueryTypeSpec("POLICY_CHANGE", 74, ("policy", "rule change", "terms"), ("{subject} policy change", "{subject} rule update", "{subject} terms"), ("news", "docs"), temporal=True),
    QueryTypeSpec("VERSION_CHANGE", 73, ("version", "release", "changelog"), ("{subject} release notes", "{subject} changelog", "{subject} version"), ("docs", "github"), temporal=True),
    QueryTypeSpec("BENCHMARK", 72, ("benchmark", "performance", "speed"), ("{subject} benchmark", "{subject} performance", "{subject} speed test"), ("scholar", "github")),
    QueryTypeSpec("ARCHITECTURE", 71, ("architecture", "design", "system"), ("{subject} architecture", "{subject} design", "{subject} internals"), ("docs", "scholar")),
    QueryTypeSpec("INSTALLATION", 70, ("install", "setup", "dependency"), ("install {subject}", "{subject} setup", "{subject} dependency"), ("docs", "github")),
    QueryTypeSpec("COMPATIBILITY", 69, ("compatible", "support", "works with"), ("{subject} compatibility", "{subject} supported versions", "{subject} works with"), ("docs", "github")),
    QueryTypeSpec("MIGRATION", 68, ("migrate", "upgrade", "port"), ("{subject} migration", "{subject} upgrade guide", "{subject} porting"), ("docs",)),
    QueryTypeSpec("CONFIGURATION", 67, ("config", "configure", "settings"), ("{subject} configuration", "{subject} config file", "{subject} settings"), ("docs", "github")),
    QueryTypeSpec("TROUBLESHOOTING", 66, ("troubleshoot", "not working", "broken"), ("{subject} troubleshooting", "{subject} not working", "{subject} known issue"), ("docs", "github")),
    QueryTypeSpec("ROOT_CAUSE", 65, ("root cause", "why failed", "postmortem"), ("{subject} root cause", "{subject} postmortem", "{subject} failure analysis"), ("news", "github")),
    QueryTypeSpec("CLAIM_CHECK", 64, ("is it true", "claim that", "fact check"), ("is it true {subject}", "{subject} fact check", "{subject} evidence"), ("news", "wikipedia"), adversarial=True),
    QueryTypeSpec("SOURCE_DISCOVERY", 63, ("source for", "citation", "reference"), ("source for {subject}", "{subject} citation", "{subject} reference"), ("wikipedia", "scholar")),
    QueryTypeSpec("PRIMARY_SOURCE", 62, ("official", "primary source", "original"), ("{subject} official", "{subject} primary source", "{subject} original source"), ("government", "docs")),
    QueryTypeSpec("SECONDARY_ANALYSIS", 61, ("analysis", "explained", "commentary"), ("{subject} analysis", "{subject} explained", "{subject} commentary"), ("news", "scholar")),
    QueryTypeSpec("TIMELINE", 60, ("timeline", "chronology", "history"), ("{subject} timeline", "{subject} chronology", "{subject} history"), ("wikipedia", "news"), temporal=True),
    QueryTypeSpec("CAUSE_EFFECT", 59, ("impact of", "effect of", "consequence"), ("impact of {subject}", "{subject} effects", "{subject} consequences"), ("news", "scholar")),
    QueryTypeSpec("PRO_CON", 58, ("pros and cons", "benefits", "risks"), ("{subject} pros cons", "{subject} benefits risks", "{subject} advantages disadvantages"), ("scholar", "news")),
    QueryTypeSpec("DECISION_SUPPORT", 57, ("should i", "best option", "choose"), ("{subject} decision", "{subject} best option", "{subject} comparison"), ("docs", "news")),
    QueryTypeSpec("ENTITY_RELATION", 56, ("relation between", "connected to", "linked to"), ("{subject} relationship", "{subject} connection", "{subject} linked to"), ("wikipedia", "scholar")),
    QueryTypeSpec("OWNERSHIP", 55, ("owned by", "parent company", "subsidiary"), ("{subject} ownership", "{subject} parent company", "{subject} subsidiary"), ("wikipedia", "news")),
    QueryTypeSpec("LEADERSHIP", 54, ("ceo", "leader", "president of"), ("{subject} leadership", "{subject} CEO", "{subject} current leader"), ("news", "wikipedia"), temporal=True),
    QueryTypeSpec("ELECTION", 53, ("election", "vote", "candidate"), ("{subject} election", "{subject} vote results", "{subject} candidate"), ("news", "government"), temporal=True),
    QueryTypeSpec("GEOPOLITICAL", 52, ("country", "war", "treaty"), ("{subject} geopolitics", "{subject} conflict", "{subject} treaty"), ("news", "wikipedia"), temporal=True),
    QueryTypeSpec("SCIENTIFIC_CONSENSUS", 51, ("consensus", "study", "evidence"), ("{subject} scientific consensus", "{subject} studies", "{subject} evidence"), ("scholar", "government")),
    QueryTypeSpec("ACADEMIC_PAPER", 50, ("paper", "research", "arxiv"), ("{subject} paper", "{subject} research", "{subject} arxiv"), ("scholar",)),
    QueryTypeSpec("PATENT", 49, ("patent", "invented", "inventor"), ("{subject} patent", "{subject} inventor", "{subject} invention"), ("government", "scholar")),
    QueryTypeSpec("STANDARD", 48, ("standard", "rfc", "specification"), ("{subject} standard", "{subject} RFC", "{subject} specification"), ("docs", "government")),
    QueryTypeSpec("DATASET", 47, ("dataset", "data source", "statistics"), ("{subject} dataset", "{subject} data source", "{subject} statistics"), ("government", "scholar")),
    QueryTypeSpec("PRICE", 46, ("price", "cost", "pricing"), ("{subject} price", "{subject} cost", "{subject} pricing"), ("news", "docs"), temporal=True),
    QueryTypeSpec("AVAILABILITY", 45, ("available", "release date", "shipping"), ("{subject} availability", "{subject} release date", "{subject} shipping"), ("news", "docs"), temporal=True),
    QueryTypeSpec("OUTAGE", 44, ("outage", "down", "status"), ("{subject} outage", "{subject} status", "{subject} down"), ("news", "docs"), temporal=True),
    QueryTypeSpec("ACQUISITION", 43, ("acquired", "merger", "bought"), ("{subject} acquisition", "{subject} merger", "{subject} bought by"), ("news", "wikipedia"), temporal=True),
    QueryTypeSpec("LAWSUIT", 42, ("lawsuit", "court", "sued"), ("{subject} lawsuit", "{subject} court", "{subject} sued"), ("news", "government"), temporal=True),
    QueryTypeSpec("SCANDAL", 41, ("scandal", "controversy", "backlash"), ("{subject} controversy", "{subject} scandal", "{subject} backlash"), ("news",), adversarial=True),
    QueryTypeSpec("MYTH", 40, ("myth", "misconception", "false"), ("{subject} myth", "{subject} misconception", "{subject} false claim"), ("news", "wikipedia"), adversarial=True),
    QueryTypeSpec("ETYMOLOGY", 39, ("origin of word", "etymology", "name meaning"), ("{subject} etymology", "{subject} name origin", "{subject} word origin"), ("wikipedia",)),
    QueryTypeSpec("LOCAL_CONTEXT", 38, ("near me", "local", "in city"), ("{subject} local", "{subject} near", "{subject} city"), ("news",)),
    QueryTypeSpec("CULTURAL_CONTEXT", 37, ("culture", "meaning in", "social"), ("{subject} culture", "{subject} social meaning", "{subject} context"), ("wikipedia", "news")),
    QueryTypeSpec("HISTORICAL_CONTEXT", 36, ("history of", "historical", "origin"), ("{subject} history", "{subject} origin", "{subject} historical context"), ("wikipedia", "scholar")),
    QueryTypeSpec("BIOGRAPHY_TIMELINE", 35, ("life of", "career", "born"), ("{subject} biography timeline", "{subject} career", "{subject} born"), ("wikipedia", "news"), temporal=True),
    QueryTypeSpec("WORKS_LIST", 34, ("works by", "books by", "movies by"), ("{subject} works", "{subject} bibliography", "{subject} filmography"), ("wikipedia",)),
    QueryTypeSpec("FEATURE_LIST", 33, ("features", "capabilities", "what can"), ("{subject} features", "{subject} capabilities", "{subject} what can it do"), ("docs", "wikipedia")),
    QueryTypeSpec("LIMITATIONS", 32, ("limitations", "can't", "drawbacks"), ("{subject} limitations", "{subject} drawbacks", "{subject} cannot"), ("docs", "news")),
    QueryTypeSpec("SAFETY", 31, ("safe", "risk", "danger"), ("{subject} safety", "{subject} risks", "{subject} danger"), ("government", "news"), adversarial=True),
    QueryTypeSpec("ETHICS", 30, ("ethical", "ethics", "moral"), ("{subject} ethics", "{subject} ethical concerns", "{subject} moral"), ("scholar", "news")),
    QueryTypeSpec("ENVIRONMENT", 29, ("environment", "climate", "emissions"), ("{subject} environmental impact", "{subject} emissions", "{subject} climate"), ("government", "scholar")),
    QueryTypeSpec("ENERGY", 28, ("energy", "power", "electricity"), ("{subject} energy use", "{subject} power", "{subject} electricity"), ("government", "news")),
    QueryTypeSpec("HARDWARE_SPEC", 27, ("specs", "hardware", "chip"), ("{subject} specs", "{subject} hardware", "{subject} chip"), ("docs", "news")),
    QueryTypeSpec("SOFTWARE_LICENSE", 26, ("license", "open source", "proprietary"), ("{subject} license", "{subject} open source", "{subject} proprietary"), ("github", "docs")),
    QueryTypeSpec("REPOSITORY_HEALTH", 25, ("github", "repo", "commits"), ("{subject} GitHub", "{subject} repository", "{subject} commits"), ("github",)),
    QueryTypeSpec("COMMUNITY_SIGNAL", 24, ("reddit", "forum", "community"), ("{subject} community", "{subject} forum", "{subject} reddit"), ("forum", "news")),
    QueryTypeSpec("SENTIMENT", 23, ("sentiment", "reaction", "people think"), ("{subject} reaction", "{subject} sentiment", "{subject} public response"), ("news", "forum")),
    QueryTypeSpec("TREND", 22, ("trend", "popular", "growth"), ("{subject} trend", "{subject} growth", "{subject} popularity"), ("news", "scholar"), temporal=True),
    QueryTypeSpec("MARKET_SHARE", 21, ("market share", "dominates", "usage share"), ("{subject} market share", "{subject} usage share", "{subject} adoption"), ("news", "scholar")),
    QueryTypeSpec("ADOPTION", 20, ("adoption", "users", "used by"), ("{subject} adoption", "{subject} users", "{subject} used by"), ("news", "docs")),
    QueryTypeSpec("ROADMAP", 19, ("roadmap", "future", "planned"), ("{subject} roadmap", "{subject} future plans", "{subject} planned features"), ("docs", "news"), temporal=True),
    QueryTypeSpec("DEPRECATION", 18, ("deprecated", "removed", "end of life"), ("{subject} deprecated", "{subject} end of life", "{subject} removed"), ("docs", "news"), temporal=True),
    QueryTypeSpec("BREAKING_CHANGE", 17, ("breaking change", "incompatible", "major version"), ("{subject} breaking change", "{subject} incompatible", "{subject} major version"), ("docs", "github"), temporal=True),
    QueryTypeSpec("PERMISSION", 16, ("allowed", "permission", "can i"), ("{subject} allowed", "{subject} permission", "{subject} policy"), ("docs", "government")),
    QueryTypeSpec("AUTHENTICATION", 15, ("login", "auth", "oauth"), ("{subject} authentication", "{subject} OAuth", "{subject} login"), ("docs",)),
    QueryTypeSpec("DATA_PRIVACY", 14, ("privacy", "data collection", "tracking"), ("{subject} privacy", "{subject} data collection", "{subject} tracking"), ("news", "docs"), adversarial=True),
    QueryTypeSpec("INTEROPERABILITY", 13, ("integrate", "bridge", "interop"), ("{subject} integration", "{subject} interoperability", "{subject} bridge"), ("docs", "github")),
    QueryTypeSpec("PERFORMANCE_REGRESSION", 12, ("slow", "regression", "latency"), ("{subject} latency", "{subject} performance regression", "{subject} slow"), ("github", "docs")),
    QueryTypeSpec("MEMORY_USAGE", 11, ("memory", "ram", "oom"), ("{subject} memory usage", "{subject} RAM", "{subject} OOM"), ("github", "docs")),
    QueryTypeSpec("GPU_SUPPORT", 10, ("cuda", "gpu", "nvidia"), ("{subject} CUDA", "{subject} GPU support", "{subject} NVIDIA"), ("docs", "github")),
    QueryTypeSpec("CPU_SUPPORT", 9, ("cpu", "no gpu", "processor"), ("{subject} CPU support", "{subject} no GPU", "{subject} processor"), ("docs", "github")),
    QueryTypeSpec("OS_SUPPORT", 8, ("windows", "linux", "macos"), ("{subject} Windows Linux macOS", "{subject} OS support", "{subject} platform support"), ("docs", "github")),
    QueryTypeSpec("LANGUAGE_BINDING", 7, ("python", "typescript", "rust"), ("{subject} language bindings", "{subject} Python", "{subject} TypeScript Rust"), ("docs", "github")),
    QueryTypeSpec("FILE_FORMAT", 6, ("format", "schema", "binary"), ("{subject} file format", "{subject} schema", "{subject} binary format"), ("docs", "github")),
    QueryTypeSpec("PROTOCOL", 5, ("protocol", "wire", "transport"), ("{subject} protocol", "{subject} wire format", "{subject} transport"), ("docs", "standard")),
    QueryTypeSpec("OBSERVABILITY", 4, ("logs", "metrics", "tracing"), ("{subject} observability", "{subject} logs metrics", "{subject} tracing"), ("docs", "github")),
    QueryTypeSpec("TESTING", 3, ("test", "verify", "validation"), ("{subject} tests", "{subject} validation", "{subject} verification"), ("docs", "github")),
    QueryTypeSpec("BUILD_SYSTEM", 2, ("compile", "build", "make"), ("{subject} build", "{subject} compile", "{subject} Makefile"), ("docs", "github")),
    QueryTypeSpec("GENERAL_RESEARCH", 1, ("",), ("{subject}", "{subject} overview", "{subject} sources"), ("wikipedia", "news")),
)


class QueryExpansionEngine:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.enabled = self.config.bool("dic.enabled", True)
        self.default_limit = self.config.int("dic.default_expansion_limit", 0, low=0, high=100)
        self.max_limit = self.config.int("dic.max_expansion_limit", 100, low=1, high=250)
        self.specs = QUESTION_TYPE_SPECS
        self.grammar = build_gbnf_grammar(self.specs)
        self.grammar_hash = hashlib.sha256(self.grammar.encode("utf-8")).hexdigest()[:16]

    def expand(self, query: str, *, requested_limit: Optional[int] = None) -> QueryExpansionResult:
        clean_query = normalize_query(query)
        requested = requested_limit if requested_limit is not None else self.default_limit
        if not self.enabled:
            requested = 0
        effective = max(0, min(self.max_limit, int(requested or 0)))
        detected = classify_query(clean_query, self.specs)
        directives: List[QueryDirective] = []
        if clean_query:
            directives.append(
                QueryDirective(
                    query=clean_query,
                    query_type=detected.name,
                    directive="PRIMARY",
                    priority=detected.priority + 1000,
                    trace={"source": "original"},
                )
            )
        if effective > 0:
            directives.extend(self._expand_from_spec(clean_query, detected, effective))
        unique = dedupe_directives(directives)
        return QueryExpansionResult(
            original_query=clean_query,
            requested_limit=max(0, int(requested or 0)),
            effective_limit=min(effective, max(0, len(unique) - 1)),
            detected_type=detected.name,
            directives=tuple(unique[: max(1, effective + 1)]),
            grammar_hash=self.grammar_hash,
        )

    def _expand_from_spec(self, query: str, detected: QueryTypeSpec, limit: int) -> List[QueryDirective]:
        subject = subject_from_query(query)
        candidates: List[QueryDirective] = []
        ordered_specs = [detected, *sorted((spec for spec in self.specs if spec is not detected), key=lambda item: -item.priority)]
        for spec in ordered_specs:
            for template in spec.templates:
                expanded = normalize_query(template.format(subject=subject, year="latest"))
                if not expanded:
                    continue
                directive = "ADVERSARIAL" if spec.adversarial else ("TEMPORAL" if spec.temporal else "EXPANDED")
                candidates.append(
                    QueryDirective(
                        query=expanded,
                        query_type=spec.name,
                        directive=directive,
                        priority=spec.priority,
                        trace={
                            "subject": subject,
                            "anchor_bias": list(spec.anchor_bias),
                            "topology_bias": spec.topology_bias,
                        },
                    )
                )
                if len(candidates) >= limit * 3:
                    return candidates
        return candidates


def parse_expansion_directive(payload: str) -> tuple[str, Optional[int]]:
    segments = [segment.strip() for segment in payload.split("|")]
    kept: List[str] = []
    count: Optional[int] = None
    for segment in segments:
        match = EXP_SEGMENT_RE.match(segment)
        if match:
            count = int(match.group("count"))
            continue
        kept.append(segment)
    cleaned = " | ".join(segment for segment in kept if segment)
    if count is None:
        match = EXP_TEXT_RE.search(cleaned)
        if match:
            count = int(match.group("count"))
            cleaned = EXP_TEXT_RE.sub("", cleaned).strip(" |")
    return cleaned.strip(), count


def classify_query(query: str, specs: Sequence[QueryTypeSpec] = QUESTION_TYPE_SPECS) -> QueryTypeSpec:
    lowered = f" {query.lower()} "
    best = specs[-1]
    best_score = -1
    for spec in specs:
        score = 0
        for pattern in spec.patterns:
            if not pattern:
                continue
            if pattern.strip() and pattern.lower() in lowered:
                score += len(pattern) + spec.priority
        if score > best_score:
            best = spec
            best_score = score
    return best


def build_gbnf_grammar(specs: Sequence[QueryTypeSpec] = QUESTION_TYPE_SPECS) -> str:
    lines = [
        'root ::= query_program',
        'query_program ::= query_type ws subject modifiers?',
        'query_type ::= ' + " | ".join(spec.name.lower() for spec in specs),
        'subject ::= text',
        'modifiers ::= (ws modifier)*',
        'modifier ::= "depth" ws signed_int | "fanout" ws signed_int | "swarm" ws signed_int | "exp" ws signed_int',
        'signed_int ::= "-"? [0-9]+',
        'text ::= ([A-Za-z0-9_:/?.,"\'()+-] | ws)+',
        'ws ::= [ \\t]+',
        "",
    ]
    lines.extend(spec.production() for spec in specs)
    return "\n".join(lines) + "\n"


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", str(query or "")).strip()


def subject_from_query(query: str) -> str:
    lowered = normalize_query(query)
    prefixes = (
        "what is ",
        "what are ",
        "who is ",
        "who was ",
        "where is ",
        "when did ",
        "why did ",
        "why does ",
        "how to ",
        "define ",
        "latest ",
        "recent ",
    )
    for prefix in prefixes:
        if lowered.lower().startswith(prefix):
            return lowered[len(prefix) :].strip(" ?.")
    return lowered.strip(" ?.")


def dedupe_directives(directives: Iterable[QueryDirective]) -> List[QueryDirective]:
    seen = set()
    unique: List[QueryDirective] = []
    for directive in sorted(directives, key=lambda item: -item.priority):
        key = directive.query.lower()
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(directive)
    return unique


def query_tokens(text: str) -> List[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]
