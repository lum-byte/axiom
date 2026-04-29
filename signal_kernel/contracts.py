"""
signal_kernel/contracts.py
==========================
Typed boundary contracts for every crossing in the signal kernel.

Written first. Every other file in the kernel imports from here.
No runtime logic lives here — only structure, validation, and constants.

Architectural law: if you find yourself wanting to mutate a contract after
construction, you need a new contract. All dataclasses are frozen.
Mutation is not a missing feature — it is an architectural error.

Contract construction validates immediately. A contract that exists has
already passed its invariants. Callers do not re-validate what they receive.

Dependency direction: everything → contracts.py. contracts.py → nothing.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Literal,
    NewType,
    Optional,
    Tuple,
)
import enum

# ─────────────────────────────────────────────────────────────────────────────
# NOMINAL TYPE ALIASES
# These are str at runtime. At the type-checker level they are distinct types
# that cannot be cross-assigned without explicit cast. This prevents passing
# an intent_vector_hash where a recipe_hash is expected — both are str, but
# they carry different semantic meaning and must not be interchangeable.
# ─────────────────────────────────────────────────────────────────────────────

TopologyClassStr = NewType("TopologyClassStr", str)
RecipeHash       = NewType("RecipeHash",       str)
RunID            = NewType("RunID",            str)
IntentVectorHash = NewType("IntentVectorHash", str)
SourceURL        = NewType("SourceURL",        str)

# Topology-layer primitive types — distinct from plain int/float at the type-checker level.
PhaseInt        = NewType("PhaseInt",        int)
ConfidenceFloat = NewType("ConfidenceFloat", float)
SurpriseFloat   = NewType("SurpriseFloat",   float)

# ─────────────────────────────────────────────────────────────────────────────
# LITERAL TYPE ALIASES
# Declared here so every file that imports a contract also gets its
# associated type vocabulary without a separate import.
# ─────────────────────────────────────────────────────────────────────────────

ContentType       = Literal["html", "json"]
TopologyPhase     = Literal["learns", "predicts", "knows"]
RenderMode        = Literal["static", "headless"]
InterfaceQueryType = Literal["SEARCH", "FETCH", "LEARN", "STATUS", "QUIT"]
InterfaceStatus    = Literal["ok", "error", "accepted", "empty"]

class FetchMode(str, enum.Enum):
    """
    Clearance-level fetch mode for a CrawlURL.
    Set by crawl_planner.py. The fetcher executes — it never decides.

    STATIC   = CL1 — httpx clearnet. No unlock required.
    HEADLESS = CL2 — Playwright + Chromium. JS rendering.
    TOR      = CL3 — Tor + Chromium. Anonymized clearnet.
    TOR_FULL = CL4 — Tor + Chromium + new circuit per fetch. Zero fingerprint.

    Fallback chain is always silent and always drops exactly one level:
    CL4 → CL3, CL3 → CL2, CL2 → CL1. Never CL4 → CL1. Never announced.
    """
    STATIC   = "static"
    HEADLESS = "headless"
    TOR      = "tor"
    TOR_FULL = "tor_full"

AuditSeverity     = Literal["critical", "warn", "info"]
ValidationOutcome = Literal[
    "passed",
    "failed_injection",
    "failed_structural",
    "failed_dryrun",
    "failed_hash",
]
AuditEventType = Literal[
    "injection_blocked",
    "hash_mismatch",
    "stderr_non_empty",
    "timeout",
    "spawn_failure",
]

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN TOPOLOGY CLASSES
# The five hardcoded classes are permanent fixtures. topology_parser.py may
# register additional classes. KNOWN_TOPOLOGY_CLASSES is documentation, not
# an allowlist — registry.py accepts any valid topology class string.
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_TOPOLOGY_CLASSES: FrozenSet[str] = frozenset({
    "NEWS_ARTICLE",
    "NEWS_ARTICLE_PAYWALLED",
    "SAAS_DOCS",
    "REST_API_JSON",
    "JSON_LD_STRUCTURED",
    "ECOMMERCE_PRODUCT",
    "GENERIC_HTML",
})

# The five classes for which hardcoded recipes exist in recipes/hardcoded/.
# These are verified by hash manifest on every load. They are never overwritten
# by the recipe compiler at runtime.
HARDCODED_TOPOLOGY_CLASSES: FrozenSet[str] = frozenset({
    "NEWS_ARTICLE",
    "SAAS_DOCS",
    "REST_API_JSON",
    "JSON_LD_STRUCTURED",
    "ECOMMERCE_PRODUCT",
})

# registry.py fallback chain. When no recipe is registered for a class,
# check for a parent class entry before falling back to GENERIC_HTML.
PARENT_CLASS_MAP: Dict[str, str] = {
    "NEWS_ARTICLE_PAYWALLED":    "NEWS_ARTICLE",
    "SAAS_DOCS_VERSIONED":       "SAAS_DOCS",
    "SAAS_DOCS_WITH_CODE":       "SAAS_DOCS",
    "REST_API_JSON_PAGINATED":   "REST_API_JSON",
    "ECOMMERCE_PRODUCT_VARIANT": "ECOMMERCE_PRODUCT",
    "FORUM_THREAD":              "BLOG_POST",
    "BLOG_POST":                 "NEWS_ARTICLE",
    "WIKIPEDIA_ARTICLE":         "NEWS_ARTICLE",
    "LANDING_PAGE":              "GENERIC_HTML",
    "AUTH_REDIRECT":             "GENERIC_HTML",
    "CLOUDFLARE_CHALLENGE":      "GENERIC_HTML",
    "RATE_LIMITED":              "GENERIC_HTML",
}

# Full ordered topology class registry. Superset of KNOWN_TOPOLOGY_CLASSES.
# topology_parser.py may register additional classes at runtime; this list is
# the documented set the system ships with. Not an allowlist — registry.py
# accepts any valid topology class string.
TOPOLOGY_CLASSES: List[str] = [
    "NEWS_ARTICLE",
    "NEWS_ARTICLE_PAYWALLED",
    "SAAS_DOCS",
    "SAAS_DOCS_VERSIONED",
    "SAAS_DOCS_WITH_CODE",
    "REST_API_JSON",
    "REST_API_JSON_PAGINATED",
    "JSON_LD_STRUCTURED",
    "ECOMMERCE_PRODUCT",
    "ECOMMERCE_PRODUCT_VARIANT",
    "FORUM_THREAD",
    "BLOG_POST",
    "WIKIPEDIA_ARTICLE",
    "LANDING_PAGE",
    "AUTH_REDIRECT",
    "CLOUDFLARE_CHALLENGE",
    "RATE_LIMITED",
    "GENERIC_HTML",
]

FALLBACK_TOPOLOGY_CLASS: str = "GENERIC_HTML"

# ─────────────────────────────────────────────────────────────────────────────
# RECIPE SECURITY CONSTANTS
# Defined here so contracts.py is the single source of truth for the
# security membrane. validator.py imports these — it does not define them.
#
# INJECTION_PATTERNS is the ordered list validator.py tests against every
# compiler-generated recipe. Match on any pattern → RecipeInjectionAttempt.
# On detection: log full recipe content with InjectionAuditRecord, raise,
# and never pass the recipe to the kernel under any circumstances.
#
# ALLOWED_RECIPE_COMMANDS is the positive allowlist. Any command found in the
# recipe not in this set → validation failure (failed_structural).
# ─────────────────────────────────────────────────────────────────────────────

INJECTION_PATTERNS: Tuple[str, ...] = (
    r"\$\(",           # command substitution $(...)
    r"`[^`]+`",        # backtick execution
    r";\s*rm\s",       # rm after semicolon
    r";\s*curl\s",     # curl after semicolon
    r";\s*wget\s",     # wget after semicolon
    r">\s*/etc",       # redirect writes to /etc
    r">\s*/usr",       # redirect writes to /usr
    r"\|\s*bash",      # pipe to bash
    r"\|\s*sh\s",      # pipe to sh
    r"\beval\s",       # eval
    r"\bexec\s",       # exec inside recipe body
    r"/dev/tcp",       # network via /dev/tcp pseudo-device
    r"/proc/self",     # proc filesystem access
    r";\s*nc\s",       # netcat after semicolon
    r";\s*python",     # python execution after semicolon
    r";\s*perl",       # perl execution after semicolon
    r"base64\s*-d",    # base64 decode pipe — common obfuscation vector
    r"\$\{IFS\}",      # IFS substitution to bypass space restriction
)

ALLOWED_RECIPE_COMMANDS: FrozenSet[str] = frozenset({
    "grep", "sed", "awk", "cat", "cut", "tr", "head", "tail", "sort", "uniq",
})

MAX_RECIPE_LINE_COUNT: int = 200

# ─────────────────────────────────────────────────────────────────────────────
# STORE FILE CONSTANTS
# The four files that constitute everything TAG knows. Defined once here.
# checkpoint contracts, CheckpointRecord, and mft_checkpoint.sh all reference
# the same set without magic strings scattered across the codebase.
# ─────────────────────────────────────────────────────────────────────────────

STORE_FILE_NAMES: FrozenSet[str] = frozenset({
    "topology_router.pt",
    "recipe_registry.mmap",
    "phase_states.mmap",
    "structural_layer.pt",
})

CHECKPOINT_RETAIN_COUNT: int = 48          # 12 hours at 15-minute intervals
CHECKPOINT_INTERVAL_MINUTES: int = 15
CHECKPOINT_STALE_THRESHOLD_MINUTES: int = 20  # crond considered dead above this

# ─────────────────────────────────────────────────────────────────────────────
# WATCHDOG CONFIGURATION CONSTANTS
# Single source of truth for store_watchdog.py tuning parameters.
# cold_start.py, checkpoint_monitor.py, and any Witness health consumer that
# needs to reason about watchdog behaviour imports these — not from
# store_watchdog.py itself (which would invert the dependency direction).
#
# Rationale for each value is documented alongside the constant so it survives
# the inevitable "why is this 30 and not 10?" review.
# ─────────────────────────────────────────────────────────────────────────────

WATCHDOG_HANDLER_TIMEOUT_S: float = 30.0
# A reload handler that has not returned after 30 s is hung or deadlocked.
# PyTorch model reload from NVMe typically completes in <5 s even for
# structural_layer.pt.  30 s is the outer bound before we cancel and count
# the failure against the circuit breaker.
# Do not lower below the slowest expected cold-storage model load time.

WATCHDOG_CIRCUIT_OPEN_THRESHOLD: int = 5
# After 5 consecutive failures for the same handler, the circuit opens and
# the handler is quarantined.  5 failures means a transient fault has become
# a persistent failure — continued calls would only flood the log.
# cold_start.py calls reset_circuit() on the next cold start.
# Witness alerts on is_circuit_open=True in the WatchdogHealth snapshot.

WATCHDOG_SHUTDOWN_DRAIN_TIMEOUT_S: float = 15.0
# On stop(), we wait up to 15 s for in-flight handler tasks to complete
# before force-cancelling.  15 s is enough for one full model reload to land
# (structural_layer.pt reload is the slowest expected handler).
# In-flight queries that depend on the old weights complete cleanly;
# the new weights are available to the next process after restart.

# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE EXECUTION LIMITS
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_SUBPROCESS_TIMEOUT_MS: int = 5_000
DEFAULT_SPAWN_TIMEOUT_MS: int = 3_000

# Hard ceiling on raw content sent to the kernel. Pages exceeding this limit
# are a signal that something upstream (Phantom, the fetcher) has gone wrong.
# pipeline.py rejects KernelInput construction if content exceeds this.
MAX_RAW_CONTENT_BYTES: int = 4 * 1024 * 1024  # 4 MB

# ─────────────────────────────────────────────────────────────────────────────
# EXTRACTION QUALITY THRESHOLDS
# feedback.py uses these to decide when to flag a recipe for recompilation.
# Defined here so topology_parser.py and feedback.py share the same numbers
# without a separate configuration layer.
# ─────────────────────────────────────────────────────────────────────────────

SIGNAL_DENSITY_FLOOR: float = 0.35
# Below this: recipe is over-stripping or the page had no signal in the
# expected zone. Review candidate.

SIGNAL_DENSITY_CEILING: float = 0.95
# Above this: suspicious — likely retaining markup, not content.

MIN_MEANINGFUL_TOKEN_REDUCTION: float = 0.40
# Below 40% reduction: recipe is under-stripping. Review candidate.
# Target production range: 60-80%.

MAX_MEANINGFUL_TOKEN_REDUCTION: float = 0.95
# Above 95% reduction: recipe is likely over-stripping and discarding signal.

EMPTY_EXTRACTION_RATE_THRESHOLD: float = 0.25
# Rolling empty extraction rate above 25% per topology class
# triggers recompilation recommendation in FeedbackEvent.

QUALITY_WINDOW_SIZE: int = 100
# Rolling window size per topology class in feedback.py.
# Not persisted. Intentionally ephemeral — stale quality history from a
# previous session is worse than no history.

# ─────────────────────────────────────────────────────────────────────────────
# PHASE CONSTANTS
# Topology learning phases — each class progresses through these in order.
# PHASE_I:   system is traversing live and building its world model.
# PHASE_II:  world model is active; system predicts before crawling.
# PHASE_III: compiled policy; direct routing without live traversal.
# ─────────────────────────────────────────────────────────────────────────────

PHASE_I   = PhaseInt(1)   # learns   — live traversal only
PHASE_II  = PhaseInt(2)   # predicts — world model active
PHASE_III = PhaseInt(3)   # knows    — compiled policy, direct routing

# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY CONFIDENCE AND SURPRISE THRESHOLDS
# Defined once here. Classifier, WLP, surprise detector, and feedback.py all
# import these — no file defines its own local copy.
# ─────────────────────────────────────────────────────────────────────────────

THETA_SURPRISE_DEFAULT   = SurpriseFloat(0.35)
THETA_CONFIDENCE_II      = ConfidenceFloat(0.70)
THETA_CONFIDENCE_III     = ConfidenceFloat(0.90)
THETA_CLASSIFY_CONFIDENT = ConfidenceFloat(0.75)
THETA_CLASSIFY_FALLBACK  = ConfidenceFloat(0.40)
THETA_WLP_MIN            = ConfidenceFloat(0.50)
SIGNAL_DENSITY_THRESHOLD = 0.15
SURPRISE_WINDOW_SIZE     = 50


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL VALIDATION HELPERS
# Used only within this module by __post_init__ methods.
# Not part of the public API — do not import these from other files.
# ─────────────────────────────────────────────────────────────────────────────

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOPOLOGY_CLASS_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def _validate_sha256(value: str, field_name: str) -> None:
    if not _SHA256_RE.match(value):
        raise ValueError(
            f"{field_name} must be a lowercase hex SHA-256 digest (64 chars). "
            f"Got {len(value)}-char string starting with {value[:16]!r}."
        )


def _validate_positive(value: float | int, field_name: str) -> None:
    if value < 0:
        raise ValueError(
            f"{field_name} must be ≥ 0, got {value}. "
            "Negative values indicate a measurement error upstream."
        )


def _validate_fraction(value: float, field_name: str) -> None:
    """Validate a value in [0.0, 1.0]."""
    if not 0.0 <= value <= 1.0:
        raise ValueError(
            f"{field_name} must be in [0.0, 1.0], got {value}. "
            "This field represents a fraction or rate."
        )


def _validate_topology_class(value: str) -> None:
    if not _TOPOLOGY_CLASS_RE.match(value):
        raise ValueError(
            f"topology_class must match [A-Z][A-Z0-9_]{{1,63}}, got {value!r}. "
            "topology_parser.py is responsible for generating valid class names."
        )


def _validate_run_id(value: str) -> None:
    try:
        parsed = uuid.UUID(value, version=4)
        if str(parsed) != value:
            raise ValueError
    except (ValueError, AttributeError):
        raise ValueError(
            f"run_id must be a canonical lowercase UUID4 string, got {value!r}. "
            "Use new_run_id() to generate compliant identifiers."
        )


def _validate_http_url(value: str, field_name: str) -> None:
    if not value.startswith(("http://", "https://")): # noqa
        raise ValueError(
            f"{field_name} must begin with http:// or https://, got {value!r}." # noqa
        )


# ═════════════════════════════════════════════════════════════════════════════
# KERNEL BOUNDARY CONTRACTS
#
# These four are the core contracts. Every boundary the kernel enforces is
# expressed through one of them. They are described in the readme.
# All other contracts in this file support or extend these four.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class KernelInput:
    """
    Everything the kernel needs to execute a single extraction.
    Constructed by pipeline.py from the caller's arguments.

    A KernelInput that exists has already validated its own invariants.
    pipeline.py builds one and passes it directly to the subprocess lifecycle.
    No downstream file re-validates these fields.

    Topology class change between invocations requires a new KernelInput —
    live recipe remount is not supported by design. Spawn fresh.
    """

    raw_content:        str
    topology_class:     TopologyClassStr
    intent_vector_hash: IntentVectorHash
    content_type:       ContentType
    source_url:         SourceURL
    run_id:             RunID
    received_at:        datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.raw_content:
            raise ValueError(
                "raw_content must not be empty. "
                "An empty page should not reach the kernel — filter upstream."
            )
        raw_bytes = len(self.raw_content.encode("utf-8"))
        if raw_bytes > MAX_RAW_CONTENT_BYTES:
            raise ValueError(
                f"raw_content is {raw_bytes:,} bytes, exceeding MAX_RAW_CONTENT_BYTES "
                f"({MAX_RAW_CONTENT_BYTES:,}). pipeline.py must truncate before "
                "constructing KernelInput."
            )
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.intent_vector_hash, "intent_vector_hash")
        if self.content_type not in ("html", "json"):
            raise ValueError( # noqa | runtime
                f"content_type must be 'html' or 'json', got {self.content_type!r}."
            )
        _validate_http_url(self.source_url, "source_url")
        _validate_run_id(self.run_id)

    @property
    def raw_byte_count(self) -> int:
        """UTF-8 encoded byte count of raw_content."""
        return len(self.raw_content.encode("utf-8"))

    @property
    def is_html(self) -> bool:
        return self.content_type == "html"

    @property
    def is_json(self) -> bool:
        return self.content_type == "json"

    @property
    def is_known_topology(self) -> bool:
        """True if topology_class is in the documented set. Not a security check."""
        return self.topology_class in KNOWN_TOPOLOGY_CLASSES

    @property
    def is_hardcoded_topology(self) -> bool:
        """True if a hardcoded recipe exists for this topology class."""
        return self.topology_class in HARDCODED_TOPOLOGY_CLASSES


@dataclass(frozen=True)
class KernelOutput:
    """
    The result of one grep pipeline execution. Returned by pipeline.py.

    extraction_empty=True is not always a failure. A paywalled page that
    rendered a login wall will produce empty extraction on a NEWS_ARTICLE
    recipe — this is correct behavior. feedback.py distinguishes empty
    extraction as quality signal vs. as error signal using rolling rates.

    If the kernel timed out or failed, pipeline.py constructs this via
    make_empty_kernel_output() with extraction_empty=True. The AXIOM graph
    continues. Kernel failure is graceful degradation, not a hard stop.
    """

    clean_signal:         str
    raw_byte_count:       int
    clean_byte_count:     int
    token_delta_estimate: int
    recipe_used:          str
    topology_class:       TopologyClassStr
    extraction_empty:     bool
    latency_ms:           float
    run_id:               RunID
    produced_at:          datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_positive(self.raw_byte_count, "raw_byte_count")
        _validate_positive(self.clean_byte_count, "clean_byte_count")
        if self.clean_byte_count > self.raw_byte_count:
            raise ValueError(
                f"clean_byte_count ({self.clean_byte_count:,}) exceeds "
                f"raw_byte_count ({self.raw_byte_count:,}). "
                "Output cannot be larger than input — measurement error."
            )
        _validate_positive(self.token_delta_estimate, "token_delta_estimate")
        _validate_positive(self.latency_ms, "latency_ms")
        if not self.recipe_used:
            raise ValueError("recipe_used must be a non-empty path string.")
        _validate_topology_class(self.topology_class)
        _validate_run_id(self.run_id)
        # Logical consistency between extraction_empty and the byte/signal fields.
        if self.extraction_empty and self.clean_byte_count > 0:
            raise ValueError(
                "extraction_empty=True but clean_byte_count > 0. "
                "These are mutually exclusive — empty extraction produces zero bytes."
            )
        if not self.extraction_empty and not self.clean_signal:
            raise ValueError(
                "extraction_empty=False but clean_signal is empty string. "
                "Either set extraction_empty=True or provide non-empty clean_signal."
            )

    @property
    def compression_ratio(self) -> float:
        """Fraction of raw bytes retained. Lower means more aggressive stripping."""
        if self.raw_byte_count == 0:
            return 0.0
        return self.clean_byte_count / self.raw_byte_count

    @property
    def token_reduction_pct(self) -> float:
        """Estimated fraction of tokens removed. In the expected 60-80% range for known topologies."""
        return 1.0 - self.compression_ratio

    @property
    def signal_density(self) -> float:
        """Fraction of non-whitespace characters in clean_signal."""
        if not self.clean_signal:
            return 0.0
        non_ws = sum(1 for c in self.clean_signal if not c.isspace())
        return non_ws / len(self.clean_signal)

    @property
    def is_under_stripping(self) -> bool:
        """Token reduction below minimum meaningful threshold — recipe not aggressive enough."""
        return self.token_reduction_pct < MIN_MEANINGFUL_TOKEN_REDUCTION

    @property
    def is_over_stripping(self) -> bool:
        """Token reduction above ceiling — recipe may be discarding signal content."""
        return self.token_reduction_pct > MAX_MEANINGFUL_TOKEN_REDUCTION

    @property
    def is_in_target_range(self) -> bool:
        """True if token reduction is in the production target range (60–80%)."""
        return 0.60 <= self.token_reduction_pct <= 0.80


@dataclass(frozen=True)
class RecipeMount:
    """
    The validated recipe that has been approved for kernel execution.
    Constructed by pipeline.py only after validator.py passes the recipe.

    Invariant: if a RecipeMount object exists, the recipe is safe to execute.
    No downstream code re-validates. The existence of this object is the proof.
    """

    recipe_path:   str
    topology_class: TopologyClassStr
    recipe_hash:   RecipeHash
    is_hardcoded:  bool
    line_count:    int
    mounted_at:    datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.recipe_path:
            raise ValueError("recipe_path must be non-empty.")
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_topology_class(self.topology_class)
        if self.line_count < 1:
            raise ValueError(
                f"line_count must be ≥ 1, got {self.line_count}. "
                "A recipe with no lines is not executable."
            )
        if self.line_count > MAX_RECIPE_LINE_COUNT:
            raise ValueError(
                f"line_count {self.line_count} exceeds MAX_RECIPE_LINE_COUNT "
                f"({MAX_RECIPE_LINE_COUNT}). validator.py must have rejected this — "
                "a RecipeMount with excessive line count should not be constructable."
            )

    @property
    def is_compiler_generated(self) -> bool:
        return not self.is_hardcoded

    def audit_key(self) -> str:
        """Stable key for audit log correlation. topology_class:hash_prefix."""
        return f"{self.topology_class}:{self.recipe_hash[:8]}"


@dataclass(frozen=True)
class ExtractionQuality:
    """
    Deterministic quality signal emitted by feedback.py.
    Consumed by topology_parser.py as the training signal for recipe decisions.

    All fields are computed via arithmetic on KernelOutput.
    No LLM. No external service. No inference. If you find yourself wanting
    to call a model to compute these, stop — that would introduce circular
    evaluation where an LLM evaluates LLM input quality.
    """

    topology_class:      TopologyClassStr
    token_reduction_pct: float
    signal_density:      float
    empty_extraction:    bool
    structured_field_count: int
    recipe_hash:         RecipeHash
    run_id:              RunID
    evaluated_at:        datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        _validate_fraction(self.token_reduction_pct, "token_reduction_pct")
        _validate_fraction(self.signal_density, "signal_density")
        if self.structured_field_count < 0:
            raise ValueError(
                f"structured_field_count must be ≥ 0, got {self.structured_field_count}. "
                "Only JSON topology classes have structured fields. HTML classes use 0."
            )
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_run_id(self.run_id)

    @property
    def is_flagged_for_recompilation(self) -> bool:
        """
        True if the quality metrics indicate recipe review is warranted.
        feedback.py emits FeedbackEvent.recompilation_recommended=True when this is True.
        """
        return (
            self.signal_density < SIGNAL_DENSITY_FLOOR
            or self.token_reduction_pct < MIN_MEANINGFUL_TOKEN_REDUCTION
            or self.token_reduction_pct > MAX_MEANINGFUL_TOKEN_REDUCTION
        )

    @property
    def composite_score(self) -> float:
        """
        Human-readable quality score in [0.0, 1.0]. Higher is better.
        Empty extraction unconditionally scores 0.0.

        This is a diagnostic metric — not a training loss. topology_parser.py
        uses token_reduction_pct and signal_density directly for decisions.
        """
        if self.empty_extraction:
            return 0.0
        density_score = min(self.signal_density / SIGNAL_DENSITY_FLOOR, 1.0)
        r = self.token_reduction_pct
        if r < MIN_MEANINGFUL_TOKEN_REDUCTION:
            reduction_score = r / MIN_MEANINGFUL_TOKEN_REDUCTION
        elif r > MAX_MEANINGFUL_TOKEN_REDUCTION:
            over = r - MAX_MEANINGFUL_TOKEN_REDUCTION
            reduction_score = max(0.0, 1.0 - over * 10.0)
        else:
            reduction_score = 1.0
        return round((density_score + reduction_score) / 2.0, 4)


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE CONTRACTS
# Describe the subprocess lifecycle internal to pipeline.py.
# The AXIOM graph never sees these — they are pipeline.py's working state.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ContainerLifecycle:
    """
    Measured timing and outcome data for one Alpine subprocess invocation.
    Captured by pipeline.py immediately after communicate() returns.
    Combined with KernelOutput to produce PipelineTelemetry.
    """

    run_id:           RunID
    topology_class:   TopologyClassStr
    recipe_hash:      RecipeHash
    spawn_latency_ms: float   # wall time from Popen() call to process ready
    total_latency_ms: float   # wall time from invocation to output available
    exit_code:        Optional[int]   # None if killed by timeout
    timed_out:        bool
    stderr_bytes:     int     # non-zero stderr is diagnostic signal, not failure

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_positive(self.spawn_latency_ms, "spawn_latency_ms")
        _validate_positive(self.total_latency_ms, "total_latency_ms")
        if self.spawn_latency_ms > self.total_latency_ms:
            raise ValueError(
                f"spawn_latency_ms ({self.spawn_latency_ms}) exceeds "
                f"total_latency_ms ({self.total_latency_ms}). "
                "Spawn cannot take longer than the total invocation."
            )
        _validate_positive(self.stderr_bytes, "stderr_bytes")
        if self.timed_out and self.exit_code is not None:
            raise ValueError(
                f"timed_out=True but exit_code={self.exit_code} is set. "
                "Processes killed by timeout do not produce an exit code."
            )

    @property
    def clean_exit(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def grep_execution_ms(self) -> float:
        """Execution time excluding container spawn overhead."""
        return self.total_latency_ms - self.spawn_latency_ms

    @property
    def stderr_non_empty(self) -> bool:
        return self.stderr_bytes > 0


@dataclass(frozen=True)
class PipelineTelemetry:
    """
    Structured telemetry event emitted by pipeline.py after every invocation.
    Feeds the Witness observability system. Required instrumentation.

    Every field is either directly measured or deterministically derived.
    No inference. No estimation beyond token_delta_estimate (char/4 approximation).
    This is the data product that validates whether kernel integration performs
    as designed — it is not optional.
    """

    run_id:               RunID
    topology_class:       TopologyClassStr
    recipe_hash:          RecipeHash
    is_hardcoded_recipe:  bool
    raw_byte_count:       int
    clean_byte_count:     int
    token_delta_estimate: int
    signal_density:       float
    extraction_empty:     bool
    spawn_latency_ms:     float
    total_latency_ms:     float
    timed_out:            bool
    stderr_non_empty:     bool
    emitted_at:           datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_positive(self.raw_byte_count, "raw_byte_count")
        _validate_positive(self.clean_byte_count, "clean_byte_count")
        _validate_positive(self.token_delta_estimate, "token_delta_estimate")
        _validate_fraction(self.signal_density, "signal_density")
        _validate_positive(self.spawn_latency_ms, "spawn_latency_ms")
        _validate_positive(self.total_latency_ms, "total_latency_ms")

    def to_log_dict(self) -> Dict[str, object]:
        """
        Flat dict for structured logging. Suitable for json.dumps().
        recipe_hash is abbreviated to 8 chars — full hash is in the audit trail.
        """
        return {
            "run_id":               self.run_id,
            "topology_class":       self.topology_class,
            "recipe_hash_prefix":   self.recipe_hash[:8],
            "is_hardcoded_recipe":  self.is_hardcoded_recipe,
            "raw_byte_count":       self.raw_byte_count,
            "clean_byte_count":     self.clean_byte_count,
            "token_delta_estimate": self.token_delta_estimate,
            "signal_density":       round(self.signal_density, 4),
            "extraction_empty":     self.extraction_empty,
            "spawn_latency_ms":     round(self.spawn_latency_ms, 2),
            "total_latency_ms":     round(self.total_latency_ms, 2),
            "timed_out":            self.timed_out,
            "stderr_non_empty":     self.stderr_non_empty,
            "emitted_at":           self.emitted_at.isoformat(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE CONTRACTS
# validator.py and registry.py import and produce these.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RecipeValidationResult:
    """
    The output of validator.py for one recipe.

    outcome == "passed" is the only state that permits recipe execution.
    All other outcomes are hard stops. pipeline.py raises before mounting.

    For "failed_injection": injection_pattern identifies which INJECTION_PATTERNS
    entry matched. Full recipe content is captured separately in InjectionAuditRecord.
    For "failed_hash":    expected_hash and actual_hash are both set for forensics.
    For "failed_structural": failure_detail names the structural problem.
    For "failed_dryrun":  failure_detail describes which fixtures produced no output.
    """

    recipe_path:       str
    recipe_hash:       RecipeHash
    topology_class:    TopologyClassStr
    is_hardcoded:      bool
    outcome:           ValidationOutcome
    failure_detail:    Optional[str]
    injection_pattern: Optional[str]   # set iff outcome == "failed_injection"
    expected_hash:     Optional[RecipeHash]  # set iff outcome == "failed_hash"
    actual_hash:       Optional[RecipeHash]  # set iff outcome == "failed_hash"
    validated_at:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _VALID_OUTCOMES: FrozenSet[str] = field(
        default=frozenset({
            "passed", "failed_injection", "failed_structural",
            "failed_dryrun", "failed_hash"
        }),
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not self.recipe_path:
            raise ValueError("recipe_path must be non-empty.")
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_topology_class(self.topology_class)
        valid = {"passed", "failed_injection", "failed_structural", "failed_dryrun", "failed_hash"}
        if self.outcome not in valid:
            raise ValueError(
                f"outcome must be one of {valid}, got {self.outcome!r}."
            )
        if self.outcome == "failed_injection" and not self.injection_pattern:
            raise ValueError(
                "injection_pattern must be set when outcome is 'failed_injection'. "
                "This is required for the forensic audit record."
            )
        if self.outcome == "failed_hash":
            if not self.expected_hash or not self.actual_hash:
                raise ValueError(
                    "expected_hash and actual_hash must both be set when outcome is 'failed_hash'."
                )

    @property
    def passed(self) -> bool:
        return self.outcome == "passed"

    @property
    def is_security_failure(self) -> bool:
        """Injection and hash failures require forensic review, not just retry."""
        return self.outcome in ("failed_injection", "failed_hash")

    @property
    def is_quality_failure(self) -> bool:
        """Structural and dry-run failures indicate recipe compiler quality issues."""
        return self.outcome in ("failed_structural", "failed_dryrun")


@dataclass(frozen=True)
class HardcodedRecipeManifestEntry:
    """
    One entry in the hardcoded recipe hash manifest.
    validator.py loads the full manifest on startup and verifies every
    hardcoded recipe against it before permitting any execution.

    Hash mismatch → RecipeMountError. The hardcoded recipes are immutable
    at runtime. This entry is the enforceable guarantee. No human-audited
    recipe may silently change between deploys.
    """

    topology_class:  TopologyClassStr
    recipe_filename: str         # basename only — e.g. "news_article.sh"
    recipe_hash:     RecipeHash
    line_count:      int
    committed_at:    datetime    # when this manifest entry was originally committed

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        if not self.recipe_filename.endswith(".sh"):
            raise ValueError(
                f"recipe_filename must end in .sh, got {self.recipe_filename!r}. "
                "Only shell recipe files are valid kernel execution targets."
            )
        if "/" in self.recipe_filename or self.recipe_filename.startswith("."):
            raise ValueError(
                f"recipe_filename must be a plain basename with no path components, "
                f"got {self.recipe_filename!r}."
            )
        _validate_sha256(self.recipe_hash, "recipe_hash")
        if self.line_count < 1:
            raise ValueError(
                f"line_count must be ≥ 1, got {self.line_count}. "
                "An empty hardcoded recipe should not exist."
            )

    @property
    def recipe_path_relative(self) -> str:
        return f"recipes/hardcoded/{self.recipe_filename}"


@dataclass(frozen=True)
class RecipeRegistryEntry:
    """
    One record in the live recipe registry maintained by registry.py.

    Hardcoded entries are permanent — registry.py never overwrites them.
    Compiler-generated entries can be replaced when topology_parser.py
    produces a new validated recipe for that topology class, triggering a
    new RecipeRegistryEntry with the updated hash and path.
    """

    topology_class:      TopologyClassStr
    recipe_path:         str
    recipe_hash:         RecipeHash
    is_hardcoded:        bool
    registered_at:       datetime
    registration_source: str   # "hardcoded_loader" | "topology_parser" | "fallback"

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        if not self.recipe_path:
            raise ValueError("recipe_path must be non-empty.")
        _validate_sha256(self.recipe_hash, "recipe_hash")
        valid_sources = {"hardcoded_loader", "topology_parser", "fallback"}
        if self.registration_source not in valid_sources:
            raise ValueError(
                f"registration_source must be one of {valid_sources}, "
                f"got {self.registration_source!r}."
            )
        if self.is_hardcoded and self.registration_source not in ("hardcoded_loader", "fallback"):
            raise ValueError(
                "is_hardcoded=True but registration_source is not 'hardcoded_loader' or 'fallback'. "
                "Only the hardcoded loader may register immutable recipes."
            )


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK CONTRACTS
# feedback.py produces these. topology_parser.py consumes them.
# Decoupled by design — topology_parser.py does not call feedback.py directly.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class FeedbackEvent:
    """
    Decoupled quality event emitted by feedback.py.
    topology_parser.py subscribes without calling feedback.py directly.
    This is how the kernel's learning loop is closed without tight coupling.

    recompilation_recommended=True means the recipe for this topology class
    has degraded below acceptable quality thresholds on a rolling basis.
    topology_parser.py decides whether and when to act on the recommendation.
    feedback.py does not invoke the compiler — it only signals.
    """

    run_id:                      RunID
    topology_class:              TopologyClassStr
    recipe_hash:                 RecipeHash
    quality:                     ExtractionQuality
    recompilation_recommended:   bool
    recompilation_reason:        Optional[str]   # required when recommended=True
    window_empty_extraction_rate: float
    window_sample_count:         int
    emitted_at:                  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_fraction(self.window_empty_extraction_rate, "window_empty_extraction_rate")
        if self.window_sample_count < 0:
            raise ValueError(
                f"window_sample_count must be ≥ 0, got {self.window_sample_count}."
            )
        if self.recompilation_recommended and not self.recompilation_reason:
            raise ValueError(
                "recompilation_reason must be set when recompilation_recommended=True. "
                "topology_parser.py needs the reason to log and act correctly."
            )
        if not self.recompilation_recommended and self.recompilation_reason is not None:
            raise ValueError(
                "recompilation_reason must be None when recompilation_recommended=False."
            )

    @property
    def is_empty_rate_above_threshold(self) -> bool:
        return self.window_empty_extraction_rate > EMPTY_EXTRACTION_RATE_THRESHOLD


@dataclass(frozen=True)
class QualityWindowEntry:
    """
    A single sample in feedback.py's in-memory rolling quality window.
    The window is a deque of these, bounded by QUALITY_WINDOW_SIZE.

    Not persisted. Intentionally ephemeral. Quality history from a previous
    process lifetime is not meaningful — the stale signal is worse than no signal.
    On TAG restart, feedback.py starts fresh.
    """

    run_id:              RunID
    recipe_hash:         RecipeHash
    token_reduction_pct: float
    signal_density:      float
    empty_extraction:    bool
    latency_ms:          float
    sampled_at:          datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_fraction(self.token_reduction_pct, "token_reduction_pct")
        _validate_fraction(self.signal_density, "signal_density")
        _validate_positive(self.latency_ms, "latency_ms")


@dataclass(frozen=True)
class RecipeQualityAggregate:
    """
    Aggregated statistics over a quality window for one topology class.
    Computed by feedback.py when topology_parser.py queries recipe performance.
    topology_parser.py uses this to decide whether to trigger recompilation.
    """

    topology_class:          TopologyClassStr
    recipe_hash:             RecipeHash
    sample_count:            int
    mean_token_reduction_pct: float
    mean_signal_density:     float
    empty_extraction_rate:   float
    mean_latency_ms:         float
    flagged_for_review:      bool
    computed_at:             datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be ≥ 0, got {self.sample_count}.")
        _validate_fraction(self.mean_token_reduction_pct, "mean_token_reduction_pct")
        _validate_fraction(self.mean_signal_density, "mean_signal_density")
        _validate_fraction(self.empty_extraction_rate, "empty_extraction_rate")
        _validate_positive(self.mean_latency_ms, "mean_latency_ms")

    @property
    def has_sufficient_sample(self) -> bool:
        """True when the window has enough samples to trust the aggregate."""
        return self.sample_count >= max(10, QUALITY_WINDOW_SIZE // 10)


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT CONTRACTS
# checkpoint_monitor.py produces and consumes these.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class CheckpointRecord:
    """
    Metadata for a single checkpoint archive in /store/checkpoints/.
    checkpoint_monitor.py builds a list of these by scanning the directory.

    Existence of a CheckpointRecord does not mean the archive is valid.
    integrity_verified=True means mft_checkpoint.sh ran `tar -tzf` and passed.
    is_complete=True means the archive contains all four store files.
    """

    archive_path:        str
    timestamp:           datetime
    archive_size_bytes:  int
    integrity_verified:  bool
    files_archived:      FrozenSet[str]

    def __post_init__(self) -> None:
        if not self.archive_path.endswith(".tar.gz"):
            raise ValueError(
                f"archive_path must end in .tar.gz, got {self.archive_path!r}. "
                "mft_checkpoint.sh produces only .tar.gz archives."
            )
        _validate_positive(self.archive_size_bytes, "archive_size_bytes")
        unexpected = self.files_archived - STORE_FILE_NAMES
        if unexpected:
            raise ValueError(
                f"files_archived contains unexpected files: {unexpected}. "
                f"Only STORE_FILE_NAMES should appear in a checkpoint archive."
            )
        missing = STORE_FILE_NAMES - self.files_archived
        if missing and self.integrity_verified:
            raise ValueError(
                f"integrity_verified=True but files_archived is missing: {missing}. "
                "An incomplete archive cannot be considered verified."
            )

    @property
    def is_complete(self) -> bool:
        return self.files_archived == STORE_FILE_NAMES and self.integrity_verified

    @property
    def archive_filename(self) -> str:
        return Path(self.archive_path).name


@dataclass(frozen=True)
class CheckpointHealth:
    """
    Point-in-time health status of the checkpoint system.
    Exposed by checkpoint_monitor.py and consumed by TAG's telemetry layer.
    An unhealthy checkpoint system does not stop the kernel — it triggers
    a Witness alert. Process 1 continues executing regardless of Process 2's state.
    """

    last_checkpoint_time:         Optional[datetime]
    checkpoint_count:             int
    latest_archive_size_bytes:    Optional[int]
    restore_invoked_at_startup:   bool
    restore_succeeded:            Optional[bool]  # None if restore was not invoked
    crond_alive:                  bool
    minutes_since_last_checkpoint: Optional[float]
    assessed_at:                  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if self.checkpoint_count < 0:
            raise ValueError(
                f"checkpoint_count must be ≥ 0, got {self.checkpoint_count}."
            )
        if self.restore_invoked_at_startup and self.restore_succeeded is None:
            raise ValueError(
                "restore_succeeded must be set when restore_invoked_at_startup=True. "
                "A restore attempt must record whether it succeeded."
            )
        if not self.restore_invoked_at_startup and self.restore_succeeded is not None:
            raise ValueError(
                "restore_succeeded must be None when restore_invoked_at_startup=False."
            )
        if self.latest_archive_size_bytes is not None:
            _validate_positive(self.latest_archive_size_bytes, "latest_archive_size_bytes")
        if self.minutes_since_last_checkpoint is not None:
            _validate_positive(self.minutes_since_last_checkpoint, "minutes_since_last_checkpoint")

    @property
    def is_healthy(self) -> bool:
        """
        True iff the checkpoint system is operating normally.
        Three conditions must all hold: crond running, checkpoint recent,
        and restore (if attempted) succeeded.
        """
        if not self.crond_alive:
            return False
        if self.minutes_since_last_checkpoint is not None:
            if self.minutes_since_last_checkpoint > CHECKPOINT_STALE_THRESHOLD_MINUTES:
                return False
        if self.restore_invoked_at_startup and self.restore_succeeded is False:
            return False
        return True

    @property
    def checkpoint_count_in_range(self) -> bool:
        """True if the archive count is at or below the rotation limit."""
        return self.checkpoint_count <= CHECKPOINT_RETAIN_COUNT


@dataclass(frozen=True)
class RestoreAttempt:
    """
    A record of one restore.sh invocation by checkpoint_monitor.py.
    If restore_succeeded=False and TAG cannot proceed, RestoreFailure is raised.

    archives_skipped documents corrupt archives encountered before finding
    a valid one. This is the expected path when the most recent checkpoint is
    corrupt — restore.sh iterates newest-to-oldest until it finds a clean archive.
    """

    attempted_at:     datetime
    restore_succeeded: bool
    archive_used:     Optional[str]       # path of the successful archive
    archives_skipped: int                 # count of corrupt archives skipped
    restored_files:   FrozenSet[str]
    exit_code:        int
    failure_reason:   Optional[str]       # None iff restore_succeeded=True

    def __post_init__(self) -> None:
        if self.archives_skipped < 0:
            raise ValueError(
                f"archives_skipped must be ≥ 0, got {self.archives_skipped}."
            )
        if self.restore_succeeded:
            if not self.archive_used:
                raise ValueError(
                    "archive_used must be set when restore_succeeded=True."
                )
            if self.failure_reason is not None:
                raise ValueError(
                    "failure_reason must be None when restore_succeeded=True."
                )
        else:
            if self.failure_reason is None:
                raise ValueError(
                    "failure_reason must be set when restore_succeeded=False. "
                    "TAG startup catches this to raise RestoreFailure."
                )

    @property
    def recovered_completely(self) -> bool:
        return self.restore_succeeded and self.restored_files == STORE_FILE_NAMES


# ═════════════════════════════════════════════════════════════════════════════
# WATCHDOG HEALTH CONTRACTS
#
# Frozen dataclasses returned by StoreWatchdog.health().
# Witness polls this endpoint; /health exposes it.  Defined here so
# checkpoint_monitor.py, cold_start.py, and any other consumer can import
# the types without pulling in store_watchdog.py's runtime dependencies
# (inotify_simple, asyncio task state, etc.).
#
# Dependency direction: store_watchdog.py imports these from contracts.py.
# Nothing else imports from store_watchdog.py for type purposes.
#
# All timestamps are ISO 8601 UTC strings.  All durations are floats in the
# unit documented on the field.  All counts are non-negative integers.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class WatchdogHandlerHealth:
    """
    Per-handler health snapshot inside a WatchdogHealth report.

    is_circuit_open=True means this handler has failed
    WATCHDOG_CIRCUIT_OPEN_THRESHOLD consecutive times and is no longer being
    called.  Witness must alert on this condition — the corresponding
    component is not receiving file-change notifications and may be serving
    stale weights indefinitely.

    Recovery: call StoreWatchdog.reset_circuit(path, handler_qualname) after
    the underlying failure is resolved, or allow cold_start.py to reset on
    the next process restart.

    Latency fields are None until the handler has been called at least once.
    avg_latency_ms is None until total_calls > 0.
    """

    qualified_name:       str            # handler.__qualname__ — e.g. "Classifier._reload"
    path:                 str            # registered store path — e.g. "topology_router.pt"
    is_circuit_open:      bool
    total_calls:          int
    total_failures:       int
    total_timeouts:       int
    consecutive_failures: int
    last_call_at_iso:     Optional[str]  # None until first call
    last_latency_ms:      Optional[float]
    min_latency_ms:       Optional[float]
    max_latency_ms:       Optional[float]
    avg_latency_ms:       Optional[float]

    def __post_init__(self) -> None:
        if not self.qualified_name:
            raise ValueError("qualified_name must be non-empty.")
        if not self.path:
            raise ValueError("path must be non-empty.")
        _validate_positive(self.total_calls, "total_calls")
        _validate_positive(self.total_failures, "total_failures")
        _validate_positive(self.total_timeouts, "total_timeouts")
        _validate_positive(self.consecutive_failures, "consecutive_failures")
        if self.total_failures > self.total_calls:
            raise ValueError(
                f"total_failures ({self.total_failures}) exceeds "
                f"total_calls ({self.total_calls}). "
                "Failure count cannot exceed call count."
            )
        if self.total_timeouts > self.total_failures:
            raise ValueError(
                f"total_timeouts ({self.total_timeouts}) exceeds "
                f"total_failures ({self.total_failures}). "
                "Timeouts are a subset of failures."
            )
        if self.consecutive_failures > self.total_calls:
            raise ValueError(
                f"consecutive_failures ({self.consecutive_failures}) exceeds "
                f"total_calls ({self.total_calls}). Invariant violated."
            )
        for fname, val in (
            ("last_latency_ms", self.last_latency_ms),
            ("min_latency_ms",  self.min_latency_ms),
            ("max_latency_ms",  self.max_latency_ms),
            ("avg_latency_ms",  self.avg_latency_ms),
        ):
            if val is not None:
                _validate_positive(val, fname)
        if (
            self.min_latency_ms is not None
            and self.max_latency_ms is not None
            and self.min_latency_ms > self.max_latency_ms
        ):
            raise ValueError(
                f"min_latency_ms ({self.min_latency_ms}) > "
                f"max_latency_ms ({self.max_latency_ms}). Invariant violated."
            )

    @property
    def is_degraded(self) -> bool:
        """
        True if the handler has any failures but the circuit has not yet opened.
        Represents a handler under stress — worth monitoring but still active.
        """
        return self.total_failures > 0 and not self.is_circuit_open

    @property
    def failure_rate(self) -> Optional[float]:
        """Fraction of calls that resulted in failure. None if never called."""
        if self.total_calls == 0:
            return None
        return self.total_failures / self.total_calls


@dataclass(frozen=True)
class WatchdogPathHealth:
    """
    Per-watched-path health snapshot inside a WatchdogHealth report.

    active_handlers is the count of handlers with is_circuit_open=False.
    A path with active_handlers=0 means every registered handler has tripped
    its circuit breaker — file changes for this path are being silently dropped.
    Witness must alert on this condition.

    event_count is the cumulative count of debounced dispatches for this path
    since watchdog.start().  It resets to zero on process restart.
    """

    path:              str
    handler_count:     int              # total registered handlers
    active_handlers:   int              # handlers with circuit_open=False
    event_count:       int              # cumulative dispatches since start()
    last_event_at_iso: Optional[str]    # None until first event
    handlers:          Tuple[WatchdogHandlerHealth, ...]

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must be non-empty.")
        _validate_positive(self.handler_count, "handler_count")
        _validate_positive(self.active_handlers, "active_handlers")
        _validate_positive(self.event_count, "event_count")
        if self.active_handlers > self.handler_count:
            raise ValueError(
                f"active_handlers ({self.active_handlers}) exceeds "
                f"handler_count ({self.handler_count}). Invariant violated."
            )
        if len(self.handlers) != self.handler_count:
            raise ValueError(
                f"len(handlers) ({len(self.handlers)}) != "
                f"handler_count ({self.handler_count}). "
                "handlers tuple must contain exactly handler_count entries."
            )

    @property
    def all_circuits_open(self) -> bool:
        """True if every handler for this path has an open circuit breaker."""
        return self.handler_count > 0 and self.active_handlers == 0

    @property
    def has_open_circuits(self) -> bool:
        """True if any handler for this path has an open circuit breaker."""
        return any(h.is_circuit_open for h in self.handlers)


@dataclass(frozen=True)
class WatchdogHealth:
    """
    Complete watchdog health snapshot returned by StoreWatchdog.health().

    is_healthy is False when:
      - is_running is False (watchdog has not been started or has been stopped)
      - open_circuit_count > 0 (one or more handlers are quarantined)

    Witness polls this on the component /health endpoint.  A transition to
    is_healthy=False triggers a Witness alert.  The alert resolves when
    is_healthy returns to True (i.e., when the circuit is reset and the
    watchdog is running again).

    uptime_s is None before the first call to watchdog.start().
    generated_at_iso is the wall-clock time the snapshot was assembled —
    not the time of the most recent file change event.

    total_events_fired is the aggregate count of debounced dispatches across
    all watched paths since start().  It is useful for verifying that the
    watchdog is observing activity during load tests and preparse cycles.
    """

    is_running:          bool
    is_healthy:          bool
    uptime_s:            Optional[float]
    total_events_fired:  int
    open_circuit_count:  int
    watched_paths:       Tuple[WatchdogPathHealth, ...]
    generated_at_iso:    str

    def __post_init__(self) -> None:
        _validate_positive(self.total_events_fired, "total_events_fired")
        _validate_positive(self.open_circuit_count, "open_circuit_count")
        if self.uptime_s is not None:
            _validate_positive(self.uptime_s, "uptime_s")
        if not self.generated_at_iso:
            raise ValueError("generated_at_iso must be a non-empty ISO 8601 string.")
        # Consistency: is_healthy must be False if not running or circuits are open.
        if self.is_healthy and not self.is_running:
            raise ValueError(
                "is_healthy=True but is_running=False. "
                "A stopped watchdog cannot be healthy."
            )
        if self.is_healthy and self.open_circuit_count > 0:
            raise ValueError(
                f"is_healthy=True but open_circuit_count={self.open_circuit_count}. "
                "A watchdog with open circuits cannot be healthy."
            )

    @property
    def path_count(self) -> int:
        """Number of watched paths registered."""
        return len(self.watched_paths)

    @property
    def total_handler_count(self) -> int:
        """Total registered handlers across all paths."""
        return sum(p.handler_count for p in self.watched_paths)

    @property
    def paths_with_open_circuits(self) -> Tuple[str, ...]:
        """Paths that have at least one circuit-open handler. Empty tuple if healthy."""
        return tuple(p.path for p in self.watched_paths if p.has_open_circuits)


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT CONTRACTS
# Security-critical events requiring forensic-grade logging.
# validator.py and pipeline.py produce these. Witness consumes them.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class InjectionAuditRecord:
    """
    Forensic record produced by validator.py on RecipeInjectionAttempt detection.

    full_recipe_content is stored in its entirety. This is intentional —
    the pattern match is evidence, but the full content is the forensic record
    needed to determine how the compiler produced a hostile recipe.
    Log this. Raise RecipeInjectionAttempt. Never pass to kernel.
    """

    detected_at:       datetime
    topology_class:    TopologyClassStr
    recipe_path:       str
    recipe_hash:       RecipeHash
    full_recipe_content: str     # complete shell body — forensic record
    matched_pattern:   str       # which INJECTION_PATTERNS entry fired
    is_hardcoded:      bool
    compiler_metadata: Optional[str]  # topology_parser.py version/metadata if compiler-generated

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        if not self.recipe_path:
            raise ValueError("recipe_path must be non-empty.")
        _validate_sha256(self.recipe_hash, "recipe_hash")
        if not self.full_recipe_content:
            raise ValueError(
                "full_recipe_content must be non-empty. "
                "An injection record with no content cannot be reviewed forensically."
            )
        if not self.matched_pattern:
            raise ValueError(
                "matched_pattern must identify the INJECTION_PATTERNS entry that fired."
            )

    def summary_line(self) -> str:
        """Single-line summary for immediate alerting. Full record goes to audit log."""
        return (
            f"INJECTION_DETECTED topology={self.topology_class} "
            f"hash={self.recipe_hash[:8]} "
            f"pattern={self.matched_pattern!r} "
            f"hardcoded={self.is_hardcoded} "
            f"at={self.detected_at.isoformat()}"
        )


@dataclass(frozen=True)
class KernelAuditEvent:
    """
    High-level audit record for security-relevant or diagnostic-level outcomes.
    Distinct from PipelineTelemetry — PipelineTelemetry is for performance metrics,
    KernelAuditEvent is for events that warrant review.

    severity="critical" events (injection_blocked, hash_mismatch) block execution.
    severity="warn" events (stderr_non_empty, timeout) degrade gracefully.
    severity="info" events (spawn_failure with recovery) are logged but not alerted.
    """

    run_id:       RunID
    event_type:   AuditEventType
    topology_class: TopologyClassStr
    recipe_hash:  RecipeHash
    detail:       str
    severity:     AuditSeverity
    occurred_at:  datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        valid_event_types = {
            "injection_blocked", "hash_mismatch", "stderr_non_empty",
            "timeout", "spawn_failure"
        }
        if self.event_type not in valid_event_types:
            raise ValueError(
                f"event_type must be one of {valid_event_types}, got {self.event_type!r}."
            )
        if not self.detail:
            raise ValueError("detail must be non-empty — audit events require explanation.")
        if self.severity not in ("critical", "warn", "info"):
            raise ValueError( # noqa | runtime
                f"severity must be 'critical', 'warn', or 'info', got {self.severity!r}."
            )
        # Injection and hash events are always critical.
        if self.event_type in ("injection_blocked", "hash_mismatch") and self.severity != "critical":
            raise ValueError(
                f"event_type {self.event_type!r} must have severity='critical'. "
                "These are hard security failures — not warnings."
            )

    @property
    def requires_immediate_alert(self) -> bool:
        return self.severity == "critical"


# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE CONTRACTS
# The boundary between the signal_kernel and the TAG Python layer.
# The AXIOM graph never touches these directly — interface.py is the only entry.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TraversalConfig:
    """
    Traversal parameters derived by the daemon's parameter policy.
    The world model produces these — the controller never sets them directly.
    High-confidence topology gets shallow fast traversal.
    Low-confidence topology gets deep careful traversal.
    Surprise-flagged topology: zero traversal until reindex completes.
    """

    max_depth:           int
    render_mode:         RenderMode
    request_pacing_ms:   int    # minimum ms between requests for this topology class
    retry_budget:        int    # maximum retries before abandoning a URL
    timeout_budget_ms:   int    # per-request timeout

    def __post_init__(self) -> None:
        if self.max_depth < 1:
            raise ValueError(
                f"max_depth must be ≥ 1, got {self.max_depth}. "
                "A depth of zero means no traversal — use surprise_flagged path instead."
            )
        if self.render_mode not in ("static", "headless"):
            raise ValueError( # noqa | runtime
                f"render_mode must be 'static' or 'headless', got {self.render_mode!r}."
            )
        _validate_positive(self.request_pacing_ms, "request_pacing_ms")
        if self.retry_budget < 0:
            raise ValueError(f"retry_budget must be ≥ 0, got {self.retry_budget}.")
        _validate_positive(self.timeout_budget_ms, "timeout_budget_ms")

    @property
    def uses_headless(self) -> bool:
        """True if JS rendering is required for this topology class."""
        return self.render_mode == "headless"


@dataclass(frozen=True)
class DaemonQuery:
    """
    What the AXIOM controller delivers to the daemon via interface.py.
    The kernel sees the topology class that results from processing this —
    it does not see the raw query. DaemonQuery is the interface layer's concern.
    """

    intent_vector_hash: IntentVectorHash
    target_urls:        Tuple[str, ...]
    requested_at:       datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        _validate_sha256(self.intent_vector_hash, "intent_vector_hash")
        if not self.target_urls:
            raise ValueError(
                "target_urls must not be empty. "
                "A query with no target URLs cannot produce a traversal."
            )
        for url in self.target_urls:
            _validate_http_url(url, f"target_urls entry {url!r}")

    @property
    def url_count(self) -> int:
        return len(self.target_urls)


@dataclass(frozen=True)
class DaemonResponse:
    """
    The complete daemon output returned to the AXIOM controller.
    The controller asks nothing else. Everything TAG needs to execute a run
    is in this object.

    recipe is the path pipeline.py mounts to the kernel container.
    It has already passed validator.py — the kernel executes it without
    re-validating. The RecipeMount contract enforces this guarantee.

    From the AXIOM graph's perspective: query in, clean signal out.
    The kernel does not exist at the graph level. This response is the
    last thing the graph sees before pipeline.py takes over.
    """

    traversal_config:  TraversalConfig
    source_priority:   Tuple[str, ...]      # reordered by predicted signal density
    friction_forecast: Dict[str, float]     # url → predicted friction [0.0, 1.0]
    recipe:            str                  # validated recipe path for pipeline.py
    recipe_hash:       RecipeHash
    topology_class:    TopologyClassStr
    mft_hit:           bool
    phase:             TopologyPhase
    responded_at:      datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.source_priority:
            raise ValueError("source_priority must not be empty.")
        for url, score in self.friction_forecast.items():
            _validate_fraction(score, f"friction_forecast[{url!r}]")
        if not self.recipe:
            raise ValueError("recipe must be a non-empty path string.")
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_topology_class(self.topology_class)
        if self.phase not in ("learns", "predicts", "knows"):
            raise ValueError( # noqa | runtime
                f"phase must be 'learns', 'predicts', or 'knows', got {self.phase!r}."
            )

    @property
    def is_phase_three(self) -> bool:
        """True if the topology class has compiled knowledge (direct policy)."""
        return self.phase == "knows"

    @property
    def highest_friction_url(self) -> Optional[str]:
        """URL with the highest predicted friction score. None if forecast is empty."""
        if not self.friction_forecast:
            return None
        return max(self.friction_forecast, key=self.friction_forecast.__getitem__)


# ═════════════════════════════════════════════════════════════════════════════
# TOPOLOGY LAYER CONTRACTS
#
# Bus events, classifier contracts, world model, sanitizer, and daemon
# interface contracts for the topology layer.  These are structurally
# separate from the kernel boundary contracts above — they cross the topology
# subsystem boundary, not the kernel container boundary.
# All dataclasses are frozen per the module-level architectural law.
# ═════════════════════════════════════════════════════════════════════════════

# ── Preparser output contracts ────────────────────────────────────────────────
# Defined before the bus events that reference them.

@dataclass(frozen=True)
class PathPattern:
    """
    A single path rule derived from robots.txt or sitemap analysis.
    pattern is a prefix or glob (e.g. "/blog/*").
    topology_class is the class inferred for all matching paths.
    """
    pattern:        str
    topology_class: str


@dataclass(frozen=True)
class RateLimitProfile:
    """
    Per-domain rate limit encoding derived from Crawl-delay and empirical
    signals.  Phantom reads this before issuing any request for the domain.
    Never mutated after construction — Phantom does not discover limits
    reactively; it reads them from this contract.
    """
    domain:              str
    requests_per_second: float
    crawl_delay_seconds: float   # from Crawl-delay directive; 0.0 if absent
    burst_capacity:      int     # max tokens before throttling begins


@dataclass(frozen=True)
class CrawlURL:
    """
    A single URL in a CrawlManifest with all fetch decisions pre-made.
    Produced by crawl_planner.py. The fetcher reads and executes — it makes
    no decisions based on the fields here.

    fetch_mode and render_mode are set by the preparser from robots.txt,
    sitemap analysis, and FrictionForecast. The fetcher never changes them.
    priority is the execution order — lower integer = higher priority.
    """
    url:                   str
    topology_hint:         str             # preparser's best-guess topology class
    fetch_mode:            FetchMode       # CL level — STATIC | HEADLESS | TOR | TOR_FULL
    render_mode:           RenderMode      # Playwright wait strategy when headless
    priority:              int             # execution order (lower = sooner)
    rate_limit_profile:    RateLimitProfile
    expected_content_type: str             # "text/html" | "application/json" | ...
    crawl_delay_seconds:   float           # raw Crawl-delay value (0.0 if absent)
    max_response_bytes:    int             # truncation ceiling; default MAX_RAW_CONTENT_BYTES
    is_robots:             bool            # True → emit RawFetchEvent with is_robots_txt=True
    is_sitemap:            bool            # True → emit with is_sitemap=True
    run_id:                str             # UUID4; propagated to RawFetchEvent


@dataclass(frozen=True)
class CrawlManifest:
    """
    The complete, ordered fetch plan for one domain.  Produced by
    crawl_planner.py from a DomainMap.  The fetcher executes this blindly —
    it makes no routing decisions of its own.

    urls is the ordered fetch sequence — priority is encoded in each CrawlURL.
    clearance_required is the highest CL level needed anywhere in the manifest.
    The fetcher checks CL availability against this on manifest receipt.
    manifest_id is a UUID4 used for dedup, restart, and cursor tracking.
    """
    domain:                     str
    urls:                       List[CrawlURL]
    total_urls:                 int
    estimated_duration_seconds: float
    clearance_required:         int     # 1 | 2 | 3 | 4
    manifest_id:                str     # UUID4

    def __post_init__(self) -> None:
        if not self.domain:
            raise ValueError("domain must be non-empty.")
        if self.clearance_required not in (1, 2, 3, 4):
            raise ValueError(
                f"clearance_required must be 1, 2, 3, or 4, got {self.clearance_required}."
            )
        if self.total_urls != len(self.urls):
            raise ValueError(
                f"total_urls ({self.total_urls}) does not match len(urls) ({len(self.urls)})."
            )
        _validate_run_id(self.manifest_id)


@dataclass(frozen=True)
class FrontierStats:
    """
    Completion statistics for one manifest's frontier state.
    Returned by frontier.stats(). Consumed by monitoring and fetcher logging.
    """
    manifest_id: str
    pending:     int
    done:        int
    failed:      int
    skipped:     int

    @property
    def total(self) -> int:
        return self.pending + self.done + self.failed + self.skipped

    @property
    def completion_rate(self) -> float:
        if self.total == 0:
            return 0.0
        return (self.done + self.skipped) / self.total


@dataclass(frozen=True)
class DomainMap:
    """
    Everything AXIOM knows about a domain before fetching a single content
    page.  Built by domain_analyzer.py from robots.txt and sitemap.
    The WLM listens for DomainTopologyEvent and uses this to pre-populate
    FrictionForecast before any crawl begins.

    crawl_manifest is handed directly to Phantom via CrawlManifestReadyEvent.
    Phantom does not generate its own manifest.
    """
    domain:                      str
    disallowed_topology_classes: Dict[str, str]   # path → topology class
    allowed_signal_paths:        List[PathPattern]
    crawl_delay_seconds:         float
    sitemap_urls:                List[str]
    path_topology_map:           Dict[str, str]   # /blog/* → NEWS_ARTICLE
    friction_zones:              List[PathPattern]
    signal_zones:                List[PathPattern]
    bot_mitigation:              str              # "cloudflare" | "none" | "custom"
    render_requirements:         Dict[str, str]   # path → "static" | "headless"
    rate_limit_profile:          RateLimitProfile
    crawl_manifest:              CrawlManifest


# ── Bus events ────────────────────────────────────────────────────────────────

@dataclass(frozen=True) # noqa
class RawFetchEvent:
    """Emitted by Phantom immediately after each HTTP response is received."""
    url:           str
    raw_bytes:     bytes
    status_code:   int
    headers:       Dict[str, str]
    fetch_latency: float
    fetch_mode:    FetchMode
    is_robots_txt: bool
    is_sitemap:    bool
    topology_hint: str
    run_id:        str
    manifest_id:   str
    byte_count:    int


@dataclass(frozen=True)
class CleanSignalEvent:
    """Emitted by the signal kernel after a successful extraction."""
    url:              str
    clean_signal:     str
    topology_class:   str
    token_reduction:  float
    signal_density:   float
    extraction_empty: bool
    run_id:           str


@dataclass(frozen=True)
class ClassificationEvent:
    """
    Emitted after every URL classification.  Carries both the classifier's
    output (probability distribution over 18 classes) and the WLM's prior
    prediction so the surprise detector can compute divergence.

    classifier_distribution  — 18-element softmax output from topology_router.pt.
                                Sums to 1.0.  Argmax is observed_class.
    wlm_prior_distribution   — 18-element vector encoding what the WLM expected
                                the topology class distribution to look like for
                                this URL given the domain's phase and history.
                                When the WLM has no prior (domain cold start),
                                set to uniform distribution (1/18 each).

    manifest_id              — UUID4 of the CrawlManifest this URL belongs to.
                                Propagated to SurpriseEvent for audit correlation.
    """
    url:                      str
    domain:                   str
    observed_class:           str        # TopologyClass — argmax of classifier_distribution
    observed_confidence:      float      # ConfidenceFloat — max(classifier_distribution)
    classifier_distribution:  Any        # np.ndarray shape (NUM_TOPOLOGY_CLASSES,)
    wlm_predicted_class:      str        # TopologyClass — argmax of wlm_prior_distribution
    wlm_predicted_confidence: float      # ConfidenceFloat — max(wlm_prior_distribution)
    wlm_prior_distribution:   Any        # np.ndarray shape (NUM_TOPOLOGY_CLASSES,)
    run_id:                   str        # UUID4
    manifest_id:              str        # UUID4

@dataclass(frozen=True)
class NewTopologyHintEvent:
    domain:                  str
    trigger:                 str   # "repeated_generic" | "coherent_cluster"
    evidence_count:          int
    centroid_vector:         List[float]
    cluster_variance:        Optional[float]
    suggested_parent_class:  str
    mdl_supports_split:      bool
    betti0_modes:            int
    oja_pc1_variance_ratio:  float
    phase_at_trigger:        int   # PhaseState int value — avoids importing internal enum
    run_id:                  str

@dataclass(frozen=True)
class DomainTopologyEvent:
    """Emitted when the domain map for a domain has been updated."""
    domain:     str
    domain_map: DomainMap


@dataclass(frozen=True)
class ZoneMapUpdatedEvent:
    """Emitted when the WLP compiles a new zone map for a topology class."""
    topology_class: str
    new_zone_map:   "ZoneMap"


@dataclass(frozen=True)
class ZoneMapInvalidatedEvent:
    """
    Emitted when a SurpriseEvent with dissolve_triggered=True invalidates
    all ZoneMaps for a topology class.

    topology/parser.py waits for a subsequent ZoneMapUpdatedEvent before
    recompiling recipes. This event signals that the current recipes are
    no longer valid — do not compile from stale ZoneMaps.
    """
    topology_class: str


@dataclass(frozen=True)
class CrawlManifestReadyEvent:
    """Emitted when a CrawlManifest is ready for Phantom to consume."""
    domain:   str
    manifest: CrawlManifest

@dataclass(frozen=True)
class ManifestCompleteEvent:
    """
    Emitted by fetcher.py when every URL in a CrawlManifest has been
    processed (done | failed | skipped). The frontier for this manifest_id
    is cleared on receipt by the cursor and frontier components.
    index_daemon subscribes to use completion as a training signal.
    """
    domain:      str
    manifest_id: str
    stats:       FrontierStats


@dataclass(frozen=True)
class CLStateUpdateEvent:
    """
    Emitted by RLCPC when clearance level availability changes for a session.
    fetcher.py subscribes and updates its internal CLState on receipt.
    The fetcher never queries CL availability proactively — it reacts to this event.

    cl2_available through cl4_available reflect current session capability.
    A False value means the corresponding fetch mode will silently fall back
    one level for the remainder of the session.
    """
    cl2_available: bool
    cl3_available: bool
    cl4_available: bool
    reason:        str    # human-readable reason for the state change


@dataclass(frozen=True)
class ContainerBreachEvent:
    """
    Emitted by fetcher.py when gVisor breach signals are detected during
    a CL3 or CL4 fetch. Triggers honeypot activation in the container
    orchestration layer. index_daemon subscribes for forensic logging.

    breach_signal identifies what triggered detection (unexpected syscall,
    outbound on wrong port, write outside /tmp/fetch_staging/, etc.).
    The original container is silently terminated on emission — the fetcher
    does not continue executing URLs from the breached context.
    """
    manifest_id:   str
    run_id:        str
    fetch_mode:    FetchMode       # always TOR or TOR_FULL
    breach_signal: str             # what triggered detection
    url:           str             # URL being fetched when breach was detected
    detected_at:   datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(frozen=True)
class SurpriseEvent:
    """
    Emitted by the surprise detector when a CleanSignalEvent deviates
    from the world model's prediction beyond THETA_SURPRISE_DEFAULT.
    dissolve_triggered=True means the zone map was invalidated and
    recompilation was requested.
    """
    topology_class:       str
    surprise_score:       float
    theta_surprise:       float
    dissolve_triggered:   bool
    contributing_signals: Dict[str, float]
    current_phase:        int
    run_id:               str
    timestamp:            str
    surprise_zone_selector: Optional[str] = None
    """
    CSS selector of the signal zone that caused the surprise, if known.
    Populated by surprise_detector.py when extraction feedback identifies
    a specific zone as the source of the noise (wrong bytes out).
    None when the surprise is topology-wide or the zone cannot be isolated.

    When present: latent_parser._on_surprise() uses _merkle_surgical_decay()
    — only L1 entries containing this selector are decayed.
    When absent: fallback topology-wide decay applies to _zone_knowledge.

    This field is the bridge between surprise_detector.py's zone-level
    diagnosis and the Merkle DAG's surgical eviction capability.
    Do not populate it speculatively — a wrong selector causes under-decay
    (entries that should be penalised are not). Omit it when unsure.
    """

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        _validate_fraction(self.surprise_score,  "surprise_score")
        _validate_fraction(self.theta_surprise,  "theta_surprise")
        if self.current_phase not in (1, 2, 3):
            raise ValueError(
                f"current_phase must be 1, 2, or 3, got {self.current_phase}. "
                f"Valid phases: 1=learns, 2=predicts, 3=knows."
            )
        _validate_run_id(self.run_id)
        if not self.timestamp:
            raise ValueError("timestamp must be a non-empty ISO 8601 string.")


@dataclass(frozen=True)
class PhaseTransitionEvent:
    """
    Emitted by index_daemon.py when a topology class transitions between
    learning phases.

    Phase semantics:
        1 (learns)   — live traversal only; world model building.
        2 (predicts) — world model active; system predicts before crawling.
        3 (knows)    — compiled policy; direct routing without live traversal.

    Valid transitions:
        1 → 2  (learns → predicts)   confidence threshold THETA_CONFIDENCE_II reached
        2 → 3  (predicts → knows)    confidence threshold THETA_CONFIDENCE_III reached
        3 → 2  (knows → predicts)    surprise dissolve triggered; revalidating
        2 → 1  (predicts → learns)   forced reindex; world model invalidated

    Transition 1 → 3 (skipping predicts) is never valid — confidence must be
    earned incrementally. index_daemon.py enforces this at emission time.

    Consumers: world_model/latent_parser.py, topology/surprise_detector.py,
               alpine_strip/offline_pipeline.py.

    All three consumers update their per-class behaviour on receipt.
    No consumer stores the event — they act on it and discard it.

    Fields:
        topology_class:   The class whose phase changed.
        from_phase:       Previous phase integer (1, 2, or 3).
        to_phase:         New phase integer (1, 2, or 3).
        confidence:       Classifier confidence that triggered the transition.
                          Always in [0.0, 1.0]. Stored for audit trail.
        reason:           Human-readable reason string for the transition.
                          Structured convention:
                            "confidence_threshold_reached"  (1→2, 2→3)
                            "surprise_dissolve_triggered"   (3→2)
                            "reindex_forced"                (2→1)
        run_id:           run_id of the index_daemon.py cycle that triggered this.
        timestamp:        UTC ISO 8601 string of when the transition was decided.
                          Set by index_daemon.py at the moment it emits, not at
                          reception — the ordering of causality must be preserved.
    """

    topology_class: str
    from_phase:     int
    to_phase:       int
    confidence:     float
    reason:         str
    run_id:         str
    timestamp:      str

    # Valid transitions as a frozenset of (from, to) tuples.
    # Stored as a class-level constant to avoid re-creating the set on every
    # __post_init__ call.  Not part of the dataclass fields — excluded from
    # equality, hashing, and repr via field(init=False, compare=False, repr=False).
    _VALID_TRANSITIONS: FrozenSet[Tuple[int, int]] = field(
        default=frozenset({(1, 2), (2, 3), (3, 2), (2, 1)}),
        init=False,
        compare=False,
        repr=False,
    )

    _VALID_REASONS: FrozenSet[str] = field(
        default=frozenset({
            "confidence_threshold_reached",
            "surprise_dissolve_triggered",
            "reindex_forced",
        }),
        init=False,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)

        # Phase range validation.
        for attr, val in (("from_phase", self.from_phase), ("to_phase", self.to_phase)):
            if val not in (1, 2, 3):
                raise ValueError(
                    f"{attr} must be 1, 2, or 3, got {val}. "
                    f"Valid phases: 1=learns, 2=predicts, 3=knows."
                )

        # No self-transition. index_daemon.py must never emit from_phase == to_phase.
        if self.from_phase == self.to_phase:
            raise ValueError(
                f"from_phase == to_phase == {self.from_phase}. "
                f"A phase transition event must represent an actual change. "
                f"index_daemon.py must only emit when the phase genuinely changes."
            )

        # Valid transition check.
        if (self.from_phase, self.to_phase) not in self._VALID_TRANSITIONS:
            raise ValueError(
                f"Phase transition {self.from_phase} → {self.to_phase} is not valid "
                f"for topology_class={self.topology_class!r}. "
                f"Valid transitions: {sorted(self._VALID_TRANSITIONS)}. "
                f"Skipping phases (e.g. 1→3) is never permitted."
            )

        # Confidence range validation.
        _validate_fraction(self.confidence, "confidence")

        # Reason validation — must be a known structured reason string.
        if self.reason not in self._VALID_REASONS:
            raise ValueError(
                f"reason={self.reason!r} is not a recognised transition reason. "
                f"Valid reasons: {sorted(self._VALID_REASONS)}. "
                f"Use a structured reason — free-text reasons break log filtering."
            )

        # Confidence floor by target phase. Prevents index_daemon.py from emitting
        # a phase upgrade that bypasses the confidence thresholds defined in contracts.
        if self.to_phase == 2 and self.confidence < THETA_CONFIDENCE_II:
            raise ValueError(
                f"Transition to phase 2 (predicts) requires confidence ≥ "
                f"THETA_CONFIDENCE_II ({THETA_CONFIDENCE_II}), got {self.confidence}. "
                f"index_daemon.py must not emit phase upgrades below threshold."
            )
        if self.to_phase == 3 and self.confidence < THETA_CONFIDENCE_III:
            raise ValueError(
                f"Transition to phase 3 (knows) requires confidence ≥ "
                f"THETA_CONFIDENCE_III ({THETA_CONFIDENCE_III}), got {self.confidence}. "
                f"index_daemon.py must not emit phase upgrades below threshold."
            )

        _validate_run_id(self.run_id)

        if not self.timestamp:
            raise ValueError("timestamp must be a non-empty ISO 8601 string.")

    @property
    def is_upgrade(self) -> bool:
        """True if the transition is a phase upgrade (higher phase number)."""
        return self.to_phase > self.from_phase

    @property
    def is_downgrade(self) -> bool:
        """True if the transition is a phase downgrade (lower phase number)."""
        return self.to_phase < self.from_phase

    @property
    def from_phase_name(self) -> str:
        """Human-readable name for from_phase."""
        return {1: "learns", 2: "predicts", 3: "knows"}[self.from_phase]

    @property
    def to_phase_name(self) -> str:
        """Human-readable name for to_phase."""
        return {1: "learns", 2: "predicts", 3: "knows"}[self.to_phase]

    @property
    def transition_label(self) -> str:
        """
        Compact audit label for structured logs.
        Example: "NEWS_ARTICLE:learns→predicts"
        """
        return f"{self.topology_class}:{self.from_phase_name}→{self.to_phase_name}"


# ── Cross-language bus events ─────────────────────────────────────────────────

@dataclass(frozen=True)
class FetchAnomalyEvent:
    """
    Canonical fetch failure signal.

    Fetcher implementations in Python, Go, or any future native component emit
    this shape when a URL cannot produce RawFetchEvent. It is a training signal,
    not a graph-stopping exception.
    """
    url:          str
    fetch_mode:   FetchMode
    status_code:  Optional[int]
    anomaly_type: str
    run_id:       str
    manifest_id:  str
    timestamp:    float = field(default_factory=lambda: datetime.now(timezone.utc).timestamp())
    detail:       str = ""

    def __post_init__(self) -> None:
        if isinstance(self.fetch_mode, str):
            object.__setattr__(self, "fetch_mode", FetchMode(self.fetch_mode))
        if not self.url:
            raise ValueError("url must be non-empty.")
        if not self.anomaly_type:
            raise ValueError("anomaly_type must be non-empty.")
        if self.status_code is not None and self.status_code < 0:
            raise ValueError("status_code must be non-negative when present.")
        if not self.manifest_id:
            raise ValueError("manifest_id must be non-empty.")


@dataclass(frozen=True)
class SignalExtractedEvent:
    """
    Signal-zone summary produced by native preparser/signal-extractor stages.

    This is deliberately lighter than CleanSignalEvent: it describes what a
    native extractor observed so index_daemon.py can prioritize offline work.
    """
    url:              str
    topology_class:   str
    signal_type:      str
    byte_count:       int
    token_count:      int
    signal_density:   float
    zone_count:       int
    source_component: str
    run_id:           str
    observed_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.url:
            raise ValueError("url must be non-empty.")
        _validate_topology_class(self.topology_class)
        if not self.signal_type:
            raise ValueError("signal_type must be non-empty.")
        _validate_positive(self.byte_count, "byte_count")
        _validate_positive(self.token_count, "token_count")
        _validate_positive(self.zone_count, "zone_count")
        _validate_fraction(self.signal_density, "signal_density")
        if not self.source_component:
            raise ValueError("source_component must be non-empty.")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class RecipeStaleEvent:
    """
    Emitted when recipe validation shows that a compiled recipe no longer
    matches recent observations.
    """
    topology_class: str
    recipe_hash:    str
    reason:         str
    confidence:     float
    run_id:         str
    observed_at:    str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        if not self.reason:
            raise ValueError("reason must be non-empty.")
        _validate_fraction(self.confidence, "confidence")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class RecipeHealthEvent:
    """
    Aggregate recipe health signal emitted by the Go recipe validator.
    """
    topology_class:    str
    recipe_hash:       str
    sample_count:      int
    success_count:     int
    failure_count:     int
    empty_rate:        float
    median_latency_ms: float
    stale:             bool
    run_id:            str
    observed_at:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _validate_topology_class(self.topology_class)
        _validate_sha256(self.recipe_hash, "recipe_hash")
        _validate_positive(self.sample_count, "sample_count")
        _validate_positive(self.success_count, "success_count")
        _validate_positive(self.failure_count, "failure_count")
        if self.success_count + self.failure_count > self.sample_count:
            raise ValueError("success_count + failure_count cannot exceed sample_count.")
        _validate_fraction(self.empty_rate, "empty_rate")
        _validate_positive(self.median_latency_ms, "median_latency_ms")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class WeightsUpdatedEvent:
    """
    Emitted after offline training atomically publishes a new weight artifact.
    """
    model_name:     str
    store_path:     str
    staging_path:   str
    checksum_sha256: str
    version:        int
    batch_count:    int
    gradient_steps: int
    run_id:         str
    updated_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name must be non-empty.")
        if not self.store_path:
            raise ValueError("store_path must be non-empty.")
        _validate_sha256(self.checksum_sha256, "checksum_sha256")
        _validate_positive(self.version, "version")
        _validate_positive(self.batch_count, "batch_count")
        _validate_positive(self.gradient_steps, "gradient_steps")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class StoreHealthEvent:
    """
    Store sentinel health observation for mmap/weight artifacts.
    """
    store_file:      str
    status:          str
    size_bytes:      int
    checksum_sha256: Optional[str]
    critical:        bool
    detail:          str
    run_id:          str
    checked_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.store_file:
            raise ValueError("store_file must be non-empty.")
        if not self.status:
            raise ValueError("status must be non-empty.")
        _validate_positive(self.size_bytes, "size_bytes")
        if self.checksum_sha256 is not None:
            _validate_sha256(self.checksum_sha256, "checksum_sha256")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class SnapshotCandidateEvent:
    """
    A URL selected by AXIOM routing as worth preserving as a temporary artifact.

    This is not an external search result. It is a snapshot request for a URL
    already surfaced by TAG routing, fetch/frontier traversal, or learned source
    priority. The tools bridge decides whether capture is possible.
    """
    url:              str
    reason:           str
    relevance_score:  float
    source_component: str
    run_id:           str
    query:            str = ""
    topology_class:   str = FALLBACK_TOPOLOGY_CLASS
    rank:             int = 0
    ttl_seconds:      int = 3600
    candidate_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if not self.reason:
            raise ValueError("reason must be non-empty.")
        _validate_fraction(self.relevance_score, "relevance_score")
        if not self.source_component:
            raise ValueError("source_component must be non-empty.")
        _validate_topology_class(self.topology_class)
        _validate_positive(self.rank, "rank")
        _validate_positive(self.ttl_seconds, "ttl_seconds")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class SnapshotCapturedEvent:
    """
    Metadata for a temporary snapshot artifact produced by tools_bridge.py.

    The artifact is intentionally outside the four durable store files. It may
    contain raw HTML, rendered HTML, markdown, screenshots, or sidecar metadata.
    The watermark applies to the artifact/provenance record only; clean signal
    sent to the model remains unmodified.
    """
    url:              str
    artifact_path:    str
    artifact_kind:    str
    sha256:           str
    byte_count:       int
    watermark:        str
    source_tool:      str
    run_id:           str
    metadata:         Dict[str, Any] = field(default_factory=dict)
    captured_at:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at:       str = ""

    def __post_init__(self) -> None:
        _validate_http_url(self.url, "url")
        if not self.artifact_path:
            raise ValueError("artifact_path must be non-empty.")
        valid_kinds = {"raw_html", "rendered_html", "markdown", "screenshot", "metadata", "bundle"}
        if self.artifact_kind not in valid_kinds:
            raise ValueError(f"artifact_kind must be one of {sorted(valid_kinds)}, got {self.artifact_kind!r}.")
        _validate_sha256(self.sha256, "sha256")
        _validate_positive(self.byte_count, "byte_count")
        if not self.watermark:
            raise ValueError("watermark must be non-empty.")
        if not self.source_tool:
            raise ValueError("source_tool must be non-empty.")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class ToolInvocationEvent:
    """Audit record for a tool call issued through the AXIOM tools bridge."""
    tool_name:        str
    invocation_id:    str
    input_hash:       str
    run_id:           str
    source_component: str
    mode:             str = "selective"
    permission_class: str = "read_only"
    started_at:       str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty.")
        if not self.invocation_id:
            raise ValueError("invocation_id must be non-empty.")
        _validate_sha256(self.input_hash, "input_hash")
        _validate_run_id(self.run_id)
        if not self.source_component:
            raise ValueError("source_component must be non-empty.")
        if self.mode not in {"selective", "manual", "diagnostic", "snapshot"}:
            raise ValueError("mode must be selective, manual, diagnostic, or snapshot.")
        if self.permission_class not in {"read_only", "write_temp", "write_repo", "network", "orchestration"}:
            raise ValueError("permission_class is not recognized.")


@dataclass(frozen=True)
class ToolResultEvent:
    """Result record for a tool call issued through the AXIOM tools bridge."""
    tool_name:        str
    invocation_id:    str
    status:           str
    output_hash:      str
    duration_ms:      float
    run_id:           str
    output_summary:   str = ""
    error_type:       Optional[str] = None
    completed_at:     str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty.")
        if not self.invocation_id:
            raise ValueError("invocation_id must be non-empty.")
        if self.status not in {"ok", "error", "skipped"}:
            raise ValueError("status must be ok, error, or skipped.")
        _validate_sha256(self.output_hash, "output_hash")
        _validate_positive(self.duration_ms, "duration_ms")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class ToolHealthEvent:
    """Dependency and capability health for one registered tool adapter."""
    tool_name:         str
    status:            str
    dependency_status: Dict[str, bool]
    permission_class:  str
    adapter_kind:      str
    run_id:            str
    detail:            str = ""
    checked_at:        str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        if not self.tool_name:
            raise ValueError("tool_name must be non-empty.")
        if self.status not in {"ready", "missing_deps", "disabled", "error"}:
            raise ValueError("status must be ready, missing_deps, disabled, or error.")
        if not isinstance(self.dependency_status, dict):
            raise ValueError("dependency_status must be a dict.")
        if self.permission_class not in {"read_only", "write_temp", "write_repo", "network", "orchestration"}:
            raise ValueError("permission_class is not recognized.")
        if not self.adapter_kind:
            raise ValueError("adapter_kind must be non-empty.")
        _validate_run_id(self.run_id)


# ── Public interface contracts ────────────────────────────────────────────────

@dataclass(frozen=True)
class InterfaceRequest:
    """One command submitted to tag/interface.py."""
    query_type: InterfaceQueryType
    payload:    str
    run_id:     str
    received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        valid = {"SEARCH", "FETCH", "LEARN", "STATUS", "QUIT"}
        if self.query_type not in valid:
            raise ValueError(f"query_type must be one of {sorted(valid)}, got {self.query_type!r}.")
        if self.query_type in {"SEARCH", "FETCH", "LEARN"} and not self.payload:
            raise ValueError(f"payload must be non-empty for {self.query_type}.")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class InterfaceResponse:
    """One response returned by tag/interface.py to the TUI."""
    run_id:      str
    status:      InterfaceStatus
    message:     str
    data:        Dict[str, Any]
    completed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        valid = {"ok", "error", "accepted", "empty"}
        if self.status not in valid:
            raise ValueError(f"status must be one of {sorted(valid)}, got {self.status!r}.")
        _validate_run_id(self.run_id)


@dataclass(frozen=True)
class SystemStatus:
    """Compact runtime status exposed by the STATUS interface command."""
    run_id:              str
    bus_started:         bool
    bus_mode:            str
    store_ready:         bool
    index_daemon_ready:  bool
    cold_start_complete: bool
    learned_domains:     int
    queued_work_items:   int
    reported_at:         str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self) -> None:
        _validate_run_id(self.run_id)
        _validate_positive(self.learned_domains, "learned_domains")
        _validate_positive(self.queued_work_items, "queued_work_items")


# ── Classifier ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ClassificationWindow:
    """
    The minimum signal set passed to topology_router.pt for window
    classification. content_prefix is capped at 4 KB — never the full page.
    The model must never see more content than necessary to classify.
    """
    url:            str
    headers:        Dict[str, str]
    content_prefix: str       # first 4 KB only — never full page
    content_type:   str       # "html" | "json" | "unknown"


@dataclass(frozen=True)
class TopologyClassification:
    """
    Result produced by topology_router.  classification_path records
    which signal layer resolved the class so callers can assess reliability
    without re-running the classifier.
    """
    topology_class:      str
    confidence:          float
    classification_path: str   # "domain"|"url"|"header"|"window"|"model"
    signals_used:        Dict[str, str]
    latency_ms:          float
    run_id:              str


# ── World model ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneMap:
    """
    Compiled extraction strategy for a topology class.  signal_zones are
    CSS selectors / XPath patterns the recipe targets; noise_zones are
    actively stripped.  Incremented version on every recompile.
    """
    topology_class: str
    signal_zones:   List[str]
    noise_zones:    List[str]
    strategy:       str        # "zone_extract"|"attribute_extract"|"envelope_extract"
    confidence:     float
    version:        int


@dataclass(frozen=True)
class TopologyTraversalPolicy:
    """
    WLP-issued crawl behaviour for one topology class.  Phantom reads this
    before dispatching any request for the class.  Distinct from the
    kernel-layer TraversalConfig — this is a higher-level topology policy.
    """
    topology_class:      str
    depth:               int
    render_mode:         str    # "static" | "headless"
    requests_per_second: float
    retry_budget:        int
    timeout_ms:          int
    confidence:          float


@dataclass(frozen=True)
class FrictionForecast:
    """
    Per-topology-class friction probability vector produced by the WLP.
    mitigation_strategy is the recommended Phantom approach for this class.
    """
    topology_class:            str
    cloudflare_probability:    float
    paywall_probability:       float
    rate_limit_probability:    float
    auth_redirect_probability: float
    mitigation_strategy:       str


@dataclass(frozen=True)
class WLMResponse:
    """Complete world-model response for one topology class query."""
    traversal_policy:  TopologyTraversalPolicy
    friction_forecast: FrictionForecast
    source_priority:   List[str]
    world_confidence:  float


@dataclass(frozen=True)
class SurprisePrediction:
    """
    WLP prediction for what a CleanSignalEvent should look like before
    Phantom fetches the page.  Surprise detector diffs the actual event
    against this prediction.
    """
    topology_class:       str
    predicted_density:    float
    predicted_reduction:  float
    predicted_empty_rate: float
    phase:                int


# ── Sanitizer / SE separator ──────────────────────────────────────────────────

@dataclass(frozen=True)
class SanitizedSignal:
    """Output of the sanitizer stage.  sanitized_empty=True is a hard signal
    that upstream extraction produced nothing usable."""
    text:                 str
    raw_byte_count:       int
    sanitized_byte_count: int
    sanitized_empty:      bool
    operations_applied:   List[str]
    latency_ms:           float
    run_id:               str


@dataclass(frozen=True)
class SeparatedSignal:
    """
    Output of the signal/code separator (SE stage).  A single extraction
    may contain both prose and code blocks; callers that only need one
    stream read the relevant field.
    """
    prose_signal:   str
    code_signal:    str
    has_prose:      bool
    has_code:       bool
    topology_class: str
    run_id:         str


# ── Daemon interface boundary ─────────────────────────────────────────────────

@dataclass(frozen=True)
class DaemonRequest:
    """
    What the AXIOM controller delivers to the topology daemon.
    axiom_state_hash is the controller's view of its own state at dispatch
    time — used for deduplication and replay detection.
    """
    query:            str
    intent_vector:    List[float]
    candidate_urls:   List[str]
    run_id:           str
    axiom_state_hash: str


# ═════════════════════════════════════════════════════════════════════════════
# FACTORY HELPERS
# Canonical construction paths for contracts with standard defaults or
# computed fields. Single creation point for things that must be consistent.
# ═════════════════════════════════════════════════════════════════════════════

def new_run_id() -> RunID:
    """
    Generate a canonical UUID4 run_id.
    Never inline uuid.uuid4() elsewhere — always use this.
    UUID4 guarantees cryptographic randomness. No sequential IDs in audit trails.
    """
    return RunID(str(uuid.uuid4()))


def compute_recipe_hash(recipe_content: str) -> RecipeHash:
    """
    SHA-256 of recipe content, encoded as UTF-8.
    Single computation point. Never inline hashlib.sha256() for recipe hashing.
    Recipe content changes by one byte → completely different hash → audit trail is clean.
    """
    return RecipeHash(hashlib.sha256(recipe_content.encode("utf-8")).hexdigest())


def make_empty_kernel_output(
    *,
    run_id: RunID,
    topology_class: TopologyClassStr,
    recipe_used: str,
    raw_byte_count: int,
    latency_ms: float,
) -> KernelOutput:
    """
    Construct a KernelOutput representing a failed or empty extraction.
    pipeline.py uses this as the graceful degradation path for timeouts,
    spawn failures, and unrecoverable errors.

    The caller returns this to the AXIOM graph — it never raises except for
    RecipeMountError and RecipeInjectionAttempt. Everything else lands here.
    """
    return KernelOutput(
        clean_signal="",
        raw_byte_count=raw_byte_count,
        clean_byte_count=0,
        token_delta_estimate=0,
        recipe_used=recipe_used,
        topology_class=topology_class,
        extraction_empty=True,
        latency_ms=latency_ms,
        run_id=run_id,
    )


def make_quality_from_output(
    output: KernelOutput,
    structured_field_count: int = 0,
) -> ExtractionQuality:
    """
    Derive ExtractionQuality from KernelOutput.
    feedback.py calls this after every extraction. It is the standard
    and only path for computing quality — not an ad-hoc implementation.

    structured_field_count must be computed separately by feedback.py
    for JSON topology classes (REST_API_JSON, JSON_LD_STRUCTURED).
    HTML topology classes always use 0.
    """
    return ExtractionQuality(
        topology_class=output.topology_class,
        token_reduction_pct=output.token_reduction_pct,
        signal_density=output.signal_density,
        empty_extraction=output.extraction_empty,
        structured_field_count=structured_field_count,
        recipe_hash=compute_recipe_hash(output.recipe_used),
        run_id=output.run_id,
    )


def make_pipeline_telemetry(
    output: KernelOutput,
    lifecycle: ContainerLifecycle,
    is_hardcoded_recipe: bool,
) -> PipelineTelemetry:
    """
    Build PipelineTelemetry from the two contracts pipeline.py holds
    after a complete invocation. This is the single construction point —
    pipeline.py does not assemble PipelineTelemetry by hand.
    """
    return PipelineTelemetry(
        run_id=output.run_id,
        topology_class=output.topology_class,
        recipe_hash=lifecycle.recipe_hash,
        is_hardcoded_recipe=is_hardcoded_recipe,
        raw_byte_count=output.raw_byte_count,
        clean_byte_count=output.clean_byte_count,
        token_delta_estimate=output.token_delta_estimate,
        signal_density=output.signal_density,
        extraction_empty=output.extraction_empty,
        spawn_latency_ms=lifecycle.spawn_latency_ms,
        total_latency_ms=lifecycle.total_latency_ms,
        timed_out=lifecycle.timed_out,
        stderr_non_empty=lifecycle.stderr_non_empty,
    )
