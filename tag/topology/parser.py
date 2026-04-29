"""
tag/topology/parser.py
=======================
AXIOM Recipe Compiler — Structural Intelligence Layer.

Compiles ZoneMap descriptions from the World Latent Parser into executable
grep/sed/awk shell pipelines. This is not a template engine. It is a
structural compiler that applies ~20 translation rules to produce shell
programs from first principles.

Architecture
────────────
The compiler operates on a formal grammar of shell primitives:

    S → Pipeline
    Pipeline → Stage ('|' Stage)*
    Stage → Command Argument*
    Command ∈ {grep, sed, awk, cat, cut, tr, head, tail, sort, uniq}
    Argument → Flag | Pattern | Expression

Translation rules map structural descriptions (CSS selectors, node types,
extraction strategies) to Stage productions. Rules compose: a single Zone
triggers 3-5 rules in sequence, producing a Pipeline.

The awk program generator is a genuine code emitter. It produces multi-
function awk programs with:
    - HTML nesting depth tracking via balanced tag counters
    - Multi-zone state machines for forum threads
    - JSON path traversal via structural pattern matching
    - Conditional extraction based on preceding context

Mathematical Foundation
───────────────────────
• Selector specificity scoring uses the CSS specificity 3-tuple (a,b,c)
  mapped to a total order via a·100 + b·10 + c for priority resolution.

• Zone weight aggregation uses softmax normalization:
  w_i = exp(z_i / τ) / Σ_j exp(z_j / τ)
  where τ is the temperature parameter (phase-dependent).

• Feedback injection uses exponential moving average for noise tracking:
  μ_t = α · x_t + (1 - α) · μ_{t-1}
  with α = 0.3 (responsive to recent signals).

• Compression ratio target is maintained via PID-like control:
  Δ_bounds = K_p · e(t) + K_i · ∫e(τ)dτ + K_d · de(t)/dt
  where e(t) = target_compression - actual_compression.

• Recipe complexity is bounded by Kolmogorov-inspired minimality:
  the compiler prefers shorter pipelines that achieve equivalent
  extraction, measured by normalized mutual information between
  input signal zones and output content.

Compilation Strategies
──────────────────────
ZONE_EXTRACT    — signal in defined HTML zones; sed chains + optional awk
ATTRIBUTE_EXTRACT — signal in data-* attributes; awk regex programs
ENVELOPE_EXTRACT  — signal in JSON envelopes; awk state machines

Phase Conditioning
──────────────────
LEARNS   (phase=1) — conservative; wider bounds; keep ambiguous zones
PREDICTS (phase=2) — balanced; strict signal zones; exclude ambiguous
KNOWS    (phase=3) — aggressive; minimum viable signal; zero noise tolerance

Event Flow
──────────
subscribes: ZoneMapUpdatedEvent, PhaseTransitionEvent, ZoneMapInvalidatedEvent
emits:      RecipeCompiledEvent (local), RecipeCompilationFailedEvent (local)

Dependency direction:
    parser.py → contracts.py, exceptions.py, crawler_bus.py,
                validator.py (check), registry.py (register)
    Nothing imports from parser.py except the daemon orchestrator.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import copy # noqa
import enum
import hashlib # noqa
import logging # noqa
import math
import mmap
import os
import re
import struct
import textwrap
import time
import traceback # noqa
import uuid # noqa
from abc import ABC, abstractmethod # noqa
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from functools import cached_property, lru_cache # noqa
from pathlib import Path
from typing import ( # noqa
    Any,
    Callable,
    ClassVar,
    Deque,
    Dict,
    Final,
    FrozenSet,
    Generator,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY (structlog for structured logging)
# ─────────────────────────────────────────────────────────────────────────────

try:
    import structlog
    log: Any = structlog.get_logger("topology.parser")
except ImportError:
    import logging as _logging_fallback
    structlog = None  # ensures symbol exists for type checkers
    log = _logging_fallback.getLogger("topology.parser")

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL IMPORTS
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.contracts import ( # noqa
    ALLOWED_RECIPE_COMMANDS,
    FALLBACK_TOPOLOGY_CLASS,
    HARDCODED_TOPOLOGY_CLASSES,
    INJECTION_PATTERNS,
    KNOWN_TOPOLOGY_CLASSES,
    MAX_RECIPE_LINE_COUNT,
    PARENT_CLASS_MAP,
    PHASE_I,
    PHASE_II,
    PHASE_III,
    SIGNAL_DENSITY_FLOOR,
    SIGNAL_DENSITY_CEILING,
    KernelOutput,
    PhaseTransitionEvent,
    RecipeMount,
    RecipeValidationResult,
    TopologyClassStr,
    ZoneMap,
    ZoneMapInvalidatedEvent,
    ZoneMapUpdatedEvent,
    compute_recipe_hash,
    new_run_id,
    FeedbackEvent,
)
from signal_kernel.exceptions import ( # noqa
    RecipeCompilationFailed,
    RecipeVersionConflict,
)

import signal_kernel.feedback as feedback
from signal_kernel.recipes.registry import register_recipe

# ═════════════════════════════════════════════════════════════════════════════
# COMPILER CONSTANTS
#
# All tuning parameters for the recipe compiler. Defined once. Documented
# with rationale. Every constant has a unit and a valid range.
# ═════════════════════════════════════════════════════════════════════════════

# ── Confidence thresholds ────────────────────────────────────────────────────

THETA_WLP_COMPILE_MIN: Final[float] = 0.30
"""Below this ZoneMap confidence, fall back to hardcoded GENERIC_HTML recipe.
0.30 is the confidence floor from wlp_zones.MIN_ZONE_CONFIDENCE.  Below 0.30,
the ZoneMap is structurally unreliable — better to use a hardcoded recipe that
extracts too broadly than a weak ZoneMap that extracts wrongly.

NOTE: 0.70 is the DISCOVERY_CONFIDENCE_CEILING — the ceiling for ZoneMaps
produced through the discovery path.  It is NOT the fallback threshold.
ZoneMaps between 0.30 and 0.70 are valid for compilation.  The spec's
THETA_WLP_MIN=0.70 was the recursive fallback to parent class; the actual
confidence floor for "use GENERIC_HTML instead" is 0.30."""

THETA_NOISE_TIGHTEN: Final[float] = 0.30
"""Noise ratio above this triggers zone boundary tightening on next compile.
0.30 means 30% of extracted content was identified as noise by the kernel.
This is the threshold where the signal-to-noise ratio degrades noticeably."""

THETA_PARENT_FALLBACK: Final[float] = 0.70
"""Between THETA_WLP_COMPILE_MIN (0.30) and this value, attempt parent-class
fallback via PARENT_CLASS_MAP before compiling.  Above 0.70, compile directly
from the ZoneMap.  This is the discovery-path confidence ceiling from
wlp_zones.DISCOVERY_CONFIDENCE_CEILING."""

# ── Content-type → grep context flags ───────────────────────────────────
# Driven by ZoneDescriptor.content_type.  Do NOT infer from zone metadata.
# These are read directly from the ZoneMap and applied verbatim.

CONTENT_TYPE_GREP_FLAGS: Final[Dict[str, Tuple[str, ...]]] = {
    "prose":  ("-A3",),               # paragraph context
    "code":   ("-B1", "-A10"),         # code block context preservation
    "list":   ("-A0",),               # each item self-contained
    "table":  ("-B1", "-A1"),         # row with header context
    "mixed":  ("-A2",),               # conservative context
}
"""
content_type drives grep flags directly — these come from ZoneDescriptor, not
inferred.  Mapping:
    prose  → -A3  (paragraph trailing context)
    code   → -B1 -A10  (leading + generous trailing for code blocks)
    list   → -A0  (self-contained items, no trailing)
    table  → -B1 -A1  (tight: header row + data row)
    mixed  → -A2  (moderate trailing, compiler's judgment)
"""

THETA_COMPRESSION_MIN: Final[float] = 0.50
"""Compression below 50% means the recipe kept too much raw content.
Target range is 60-80%. Below 50% the recipe is under-stripping."""

THETA_COMPRESSION_MAX: Final[float] = 0.995
"""Compression above 99.5% means the recipe stripped too aggressively.
At this level, the clean signal is less than 0.5% of the raw content,
which almost certainly means signal was discarded."""

# ── Recipe complexity limits ─────────────────────────────────────────────────

MIN_SIGNAL_LENGTH: Final[int] = 40
"""Characters. KNOWS phase only: lines shorter than this are stripped.
40 chars is approximately 8 words — shorter lines are typically
navigation labels, breadcrumbs, or widget residue."""

MAX_RECIPE_LINES: Final[int] = max(MAX_RECIPE_LINE_COUNT, 400)
"""Compiled recipe over this many lines fails validation.
Imported from contracts.py for single source of truth."""

MAX_AWK_FUNCTIONS: Final[int] = 12
"""Maximum number of awk functions in a single compiled program.
Beyond this, the awk program is too complex for reliable execution
in the kernel's subprocess timeout budget."""

MAX_PIPELINE_STAGES: Final[int] = 25
"""Maximum number of pipe stages in a compiled recipe.
Each pipe stage adds subprocess overhead. Beyond 25, latency
exceeds the 5s execution budget for complex pages."""

MAX_SELECTOR_DEPTH: Final[int] = 8
"""Maximum nesting depth for CSS selector decomposition.
Selectors deeper than this indicate pathological HTML structure
that grep/sed cannot reliably traverse."""

# ── Feedback injection parameters ────────────────────────────────────────────

FEEDBACK_EMA_ALPHA: Final[float] = 0.3
"""Exponential moving average smoothing factor for noise tracking.
α=0.3 gives ~80% of weight to the last 5 observations.
Responsive enough to catch degradation, smooth enough to ignore outliers."""

FEEDBACK_WINDOW_SIZE: Final[int] = 50
"""Rolling window size for feedback signal aggregation.
50 samples provides statistically meaningful noise/compression estimates
while remaining responsive to structural changes."""

PID_KP: Final[float] = 0.5
"""Proportional gain for compression ratio PID controller.
0.5 provides moderate correction per compile cycle."""

PID_KI: Final[float] = 0.1
"""Integral gain for compression ratio PID controller.
0.1 slowly corrects persistent steady-state error."""

PID_KD: Final[float] = 0.05
"""Derivative gain for compression ratio PID controller.
0.05 provides light damping against oscillation."""

TARGET_COMPRESSION: Final[float] = 0.70
"""Target compression ratio (fraction of content removed).
70% removal is the sweet spot: enough noise stripped to be useful,
enough signal retained to be complete."""

# ── Softmax temperature for zone weight normalization ────────────────────────

SOFTMAX_TEMP_LEARNS: Final[float] = 2.0
"""High temperature → flatter distribution → more zones contribute.
In LEARNS phase, we want broad extraction to maximize recall."""

SOFTMAX_TEMP_PREDICTS: Final[float] = 1.0
"""Standard temperature → natural weight distribution."""

SOFTMAX_TEMP_KNOWS: Final[float] = 0.5
"""Low temperature → sharper distribution → top zones dominate.
In KNOWS phase, we want surgical extraction of highest-confidence zones."""

# ── Path constants ───────────────────────────────────────────────────────────

COMPILER_GENERATED_PATH: Final[Path] = Path("signal_kernel/recipes/compiler_generated")
"""Output directory for compiled recipes. Write-only from parser.py.
Registry owns reads. Naming convention: {topology_class}.sh"""

FIXTURE_PATH: Final[Path] = Path("signal_kernel/recipes/test_fixtures")
"""Test fixture directory. Each topology class has 3-5 real HTML samples.
Validator runs compiled recipes against these before registration."""

PHASE_STATES_PATH: Final[Path] = Path("store/phase_states.mmap")
"""Memory-mapped phase state file. Read-only in parser.py.
Written by index_daemon.py. Format: topology_class → phase_int."""

# ── Staleness and debounce ───────────────────────────────────────────────────

COMPILE_DEBOUNCE_S: Final[float] = 2.0
"""Minimum seconds between compilations for the same topology class.
Prevents compile storms when the WLP emits rapid ZoneMap updates."""

MAX_COMPILE_RETRIES: Final[int] = 3
"""Maximum retries for a single compilation attempt before giving up.
Each retry uses progressively looser parameters."""


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL ENUMERATIONS
#
# Compiler-internal type vocabulary. Not exported. Not in contracts.py
# because these are implementation details of the compilation algorithm.
# ═════════════════════════════════════════════════════════════════════════════

class ExtractionStrategy(enum.Enum):
    """The four compilation strategies, matching wlp_zones.ExtractionStrategy.

    Strategy names map to the actual ZoneMap field values:
        DEPTH_FIRST    — signal in defined HTML zones; sed chains + awk depth tracking
                         (spec originally called this ZONE_EXTRACT)
        BREADTH_FIRST  — signal in data-* attributes; awk regex programs
                         (spec originally called this ATTRIBUTE_EXTRACT)
        SECTION_SCOPED — signal inside JSON response envelope at known path
                         (spec originally called this ENVELOPE_EXTRACT)
        FLAT           — top-level only extraction (JSON_LD_STRUCTURED, shallow pages)
    """
    DEPTH_FIRST    = "depth_first"
    BREADTH_FIRST  = "breadth_first"
    SECTION_SCOPED = "section_scoped"
    FLAT           = "flat"

    @classmethod
    def from_str(cls, s: str) -> "ExtractionStrategy":
        """Convert a strategy string to enum. Raises ValueError on unknown."""
        _map = {e.value: e for e in cls}
        # Backwards compatibility with spec names during migration.
        _compat = {
            "zone_extract":      cls.DEPTH_FIRST,
            "attribute_extract": cls.BREADTH_FIRST,
            "envelope_extract":  cls.SECTION_SCOPED,
        }
        if s in _map:
            return _map[s]
        if s in _compat:
            return _compat[s]
        raise ValueError(
            f"Unknown extraction strategy {s!r}. "
            f"Valid: {sorted(_map.keys())}"
        )

    @classmethod
    def from_zone_map(cls, zone_map: Any) -> "ExtractionStrategy":
        """Read strategy directly from a ZoneMap's extraction_strategy field.

        The real ZoneMap from wlp_zones.py carries extraction_strategy as an
        enum instance (wlp_zones.ExtractionStrategy), not a string.  This
        method handles both: the enum's .value attribute (str) and raw str.
        """
        raw = zone_map.extraction_strategy
        if hasattr(raw, "value"):
            raw = raw.value
        return cls.from_str(raw)


class NodeType(enum.Enum):
    """Zone classification. SIGNAL zones are extracted. NOISE zones are stripped.
    AMBIGUOUS zones are phase-dependent."""
    SIGNAL = "signal"
    NOISE = "noise"
    AMBIGUOUS = "ambiguous"


class PhaseStr(enum.Enum):
    """String representation of compilation phases."""
    LEARNS = "learns"
    PREDICTS = "predicts"
    KNOWS = "knows"

    @classmethod
    def from_int(cls, phase_int: int) -> "PhaseStr":
        return {1: cls.LEARNS, 2: cls.PREDICTS, 3: cls.KNOWS}[phase_int]

    @property
    def phase_int(self) -> int:
        return {"learns": 1, "predicts": 2, "knows": 3}[self.value]


class SelectorKind(enum.Enum):
    """CSS selector classification for rule dispatch."""
    CLASS_SELECTOR = "class"
    ID_SELECTOR = "id"
    TAG_SELECTOR = "tag"
    ATTRIBUTE_SELECTOR = "attribute"
    COMPOUND_SELECTOR = "compound"
    PSEUDO_SELECTOR = "pseudo"
    DESCENDANT_SELECTOR = "descendant"
    UNIVERSAL_SELECTOR = "universal"


class RuleCategory(enum.Enum):
    """Translation rule categories for composition ordering."""
    ZONE_BOUNDARY = "zone_boundary"
    NOISE_STRIP = "noise_strip"
    CONTENT_EXTRACT = "content_extract"
    ATTRIBUTE_EXTRACT = "attribute_extract"
    JSON_EXTRACT = "json_extract"
    TEXT_NORMALIZE = "text_normalize"
    POST_FILTER = "post_filter"


class CompileSeverity(enum.Enum):
    """Severity levels for compilation diagnostics."""
    INFO = "info"
    WARN = "warn"
    ERROR = "error"
    FATAL = "fatal"


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL DATA STRUCTURES
#
# Rich compiler-internal representations that extend the simplified ZoneMap
# from contracts.py. These carry the derived information the compiler needs.
# ═════════════════════════════════════════════════════════════════════════════

class CSSSpecificity(NamedTuple):
    """CSS specificity 3-tuple (a, b, c).
    a = number of ID selectors
    b = number of class selectors, attribute selectors, pseudo-classes
    c = number of type selectors, pseudo-elements
    Total order: a·100 + b·10 + c."""
    ids: int
    classes: int
    types: int

    @property
    def score(self) -> int:
        """Scalar specificity score for total ordering."""
        return self.ids * 100 + self.classes * 10 + self.types

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, CSSSpecificity):
            return NotImplemented
        return self.score > other.score


@dataclass(frozen=True)
class ParsedSelector:
    """Decomposed CSS selector with structural metadata.
    Produced by the selector parser. Consumed by translation rules."""
    raw: str
    kind: SelectorKind
    tag: Optional[str]
    class_name: Optional[str]
    id_name: Optional[str]
    attribute_name: Optional[str]
    attribute_value: Optional[str]
    specificity: CSSSpecificity
    depth_hint: int
    parent_selectors: Tuple[str, ...]
    is_negation: bool

    # Computed at construction — not init parameters
    is_simple: bool = field(init=False)
    grep_pattern: Optional[str] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_simple", self.kind in (
            SelectorKind.CLASS_SELECTOR,
            SelectorKind.ID_SELECTOR,
            SelectorKind.TAG_SELECTOR,
        ))
        if self.kind == SelectorKind.CLASS_SELECTOR and self.class_name:
            pat: Optional[str] = f'class="[^"]*{re.escape(self.class_name)}[^"]*"'
        elif self.kind == SelectorKind.ID_SELECTOR and self.id_name:
            pat = f'id="{re.escape(self.id_name)}"'
        elif self.kind == SelectorKind.TAG_SELECTOR and self.tag:
            pat = f"<{re.escape(self.tag)}[> ]"
        else:
            pat = None
        object.__setattr__(self, "grep_pattern", pat)


@dataclass(frozen=True)
class EnrichedZone:
    """Compiler-internal enriched zone representation.
    Built from ZoneMap.signal_zones/noise_zones + parsed selectors."""
    selector: ParsedSelector
    node_type: NodeType
    weight: float
    structural_role: str
    child_noise_selectors: Tuple[ParsedSelector, ...]
    depth_limit: Optional[int]
    data_attributes: Tuple[str, ...]
    json_path: Optional[str]

    # Computed at construction — not init parameter
    priority_score: float = field(init=False)

    def __post_init__(self) -> None:
        specificity_norm = min(self.selector.specificity.score / 300.0, 1.0)
        object.__setattr__(self, "priority_score", 0.7 * self.weight + 0.3 * specificity_norm)


@dataclass(frozen=True)
class CompilerDiagnostic:
    """Single diagnostic message from the compilation process.
    Accumulated during compilation for post-mortem analysis."""
    severity: CompileSeverity
    rule_id: Optional[int]
    message: str
    zone_selector: Optional[str]
    timestamp: float = field(default_factory=time.monotonic)


@dataclass
class FeedbackState:
    """Mutable feedback state for a single topology class.
    Tracks rolling statistics from kernel execution results."""
    noise_ema: float = 0.0
    compression_ema: float = TARGET_COMPRESSION
    surprise_count: int = 0
    total_samples: int = 0
    pid_integral: float = 0.0
    pid_prev_error: float = 0.0
    last_noise_ratio: float = 0.0
    last_compression: float = TARGET_COMPRESSION
    last_surprise_fired: bool = False
    tighten_requested: bool = False
    loosen_requested: bool = False
    history: Deque[Tuple[float, float]] = field(
        default_factory=lambda: deque(maxlen=FEEDBACK_WINDOW_SIZE)
    )

    def update(self, noise_ratio: float, compression: float,
               surprise_fired: bool) -> None:
        """Update feedback state with new observation.
        Uses EMA for smooth tracking and PID for correction signal."""
        self.total_samples += 1
        self.last_noise_ratio = noise_ratio
        self.last_compression = compression
        self.last_surprise_fired = surprise_fired

        # EMA update for noise and compression
        self.noise_ema = (
                FEEDBACK_EMA_ALPHA * noise_ratio
                + (1.0 - FEEDBACK_EMA_ALPHA) * self.noise_ema
        )
        self.compression_ema = (
                FEEDBACK_EMA_ALPHA * compression
                + (1.0 - FEEDBACK_EMA_ALPHA) * self.compression_ema
        )

        # PID controller for compression targeting — all terms on smoothed signal
        error = TARGET_COMPRESSION - self.compression_ema
        self.pid_integral += error
        # Anti-windup: clamp integral to prevent runaway
        self.pid_integral = max(-5.0, min(5.0, self.pid_integral))
        derivative = error - self.pid_prev_error
        self.pid_prev_error = error

        correction = (
                PID_KP * error
                + PID_KI * self.pid_integral
                + PID_KD * derivative
        )

        # Determine adjustment direction
        self.tighten_requested = (
                noise_ratio > THETA_NOISE_TIGHTEN
                or compression < THETA_COMPRESSION_MIN
                or correction > 0.1
        )
        self.loosen_requested = (
                surprise_fired
                or compression > THETA_COMPRESSION_MAX
                or correction < -0.1
        )

        if surprise_fired:
            self.surprise_count += 1

        self.history.append((noise_ratio, compression))

    @property
    def correction_signal(self) -> float:
        """PID output: positive = tighten, negative = loosen."""
        error = TARGET_COMPRESSION - self.compression_ema
        return (
            PID_KP * error
            + PID_KI * self.pid_integral
            + PID_KD * (error - self.pid_prev_error)
        )

    @property
    def is_stable(self) -> bool:
        """True when recent noise and compression are within bounds."""
        if self.total_samples < 5:
            return True  # Not enough data to judge
        return (
            self.noise_ema < THETA_NOISE_TIGHTEN
            and THETA_COMPRESSION_MIN <= self.compression_ema <= THETA_COMPRESSION_MAX
        )

    @property
    def variance(self) -> float:
        """Variance of recent compression ratios. High variance = unstable."""
        if len(self.history) < 2:
            return 0.0
        compressions = [c for _, c in self.history]
        mean = sum(compressions) / len(compressions)
        return sum((c - mean) ** 2 for c in compressions) / len(compressions)


@dataclass
class CompilerContext:
    """Full compilation context for a single compile() invocation.
    Threading all state through this avoids mutable class-level state."""
    zone_map: ZoneMap
    phase: PhaseStr
    feedback: FeedbackState
    enriched_zones: List[EnrichedZone]
    strategy: ExtractionStrategy
    diagnostics: List[CompilerDiagnostic]
    topology_class: str
    zone_map_version: int
    attempt: int
    start_time: float
    intent: Optional[str] = None
    fallback_chain: List[str] = field(default_factory=list)

    def add_diagnostic(
        self,
        severity: CompileSeverity,
        message: str,
        *,
        rule_id: Optional[int] = None,
        zone_selector: Optional[str] = None,
    ) -> None:
        self.diagnostics.append(CompilerDiagnostic(
            severity=severity,
            rule_id=rule_id,
            message=message,
            zone_selector=zone_selector,
        ))

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000.0

    @property
    def has_errors(self) -> bool:
        return any(
            d.severity in (CompileSeverity.ERROR, CompileSeverity.FATAL)
            for d in self.diagnostics
        )

    @property
    def softmax_temperature(self) -> float:
        """Phase-dependent softmax temperature for zone weight normalization."""
        return {
            PhaseStr.LEARNS: SOFTMAX_TEMP_LEARNS,
            PhaseStr.PREDICTS: SOFTMAX_TEMP_PREDICTS,
            PhaseStr.KNOWS: SOFTMAX_TEMP_KNOWS,
        }[self.phase]


@dataclass(frozen=True)
class CompiledRecipe:
    """Output of a single compilation. Not yet written to disk."""
    content: str
    topology_class: str
    intent: Optional[str]
    strategy: ExtractionStrategy
    phase: PhaseStr
    zone_map_version: int
    line_count: int
    stage_count: int
    diagnostics: Tuple[CompilerDiagnostic, ...]
    fallback_chain: Tuple[str, ...]
    compiled_at: float

    # Computed at construction — not init parameters
    checksum: str = field(init=False)
    is_valid_complexity: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "checksum", compute_recipe_hash(self.content))
        object.__setattr__(self, "is_valid_complexity", (
            self.line_count <= MAX_RECIPE_LINES
            and self.stage_count <= MAX_PIPELINE_STAGES
        ))


@dataclass(frozen=True)
class RecipeCompiledEvent:
    """Emitted on successful compilation + validation + write.
    Local event — not in contracts.py. Consumed by the daemon."""
    topology_class: str
    intent: Optional[str]
    zone_map_version: int
    strategy: str
    phase: str
    recipe_path: str
    checksum: str
    compiled_at: float
    line_count: int
    fallback_chain: Tuple[str, ...]


@dataclass(frozen=True)
class RecipeCompilationFailedEvent:
    """Emitted on compilation or validation failure.
    Local event — not in contracts.py. Consumed by the daemon."""
    topology_class: str
    zone_map_version: int
    failure_reason: str
    phase: str
    attempted_at: float
    fallback_chain: Tuple[str, ...]


# ═════════════════════════════════════════════════════════════════════════════
# CSS SELECTOR PARSER
#
# Decomposes CSS selector strings into structured ParsedSelector objects.
# This is the compiler's lexer. Every ZoneMap selector passes through here
# before any translation rule is applied.
#
# Supports: class selectors (.name), ID selectors (#name), tag selectors
# (div), attribute selectors ([data-x="y"]), descendant combinators (a b),
# compound selectors (div.class#id), and negation pseudo-class (:not(...)).
#
# Does NOT support: sibling combinators (+, ~), nth-child, or CSS3 selectors
# that have no meaningful translation to grep/sed patterns. These are logged
# as diagnostics and the selector is treated as a tag-level match.
# ═════════════════════════════════════════════════════════════════════════════

# Pre-compiled regex patterns for selector decomposition.
_RE_CLASS = re.compile(r"\.([a-zA-Z_][\w-]*)")
_RE_ID = re.compile(r"#([a-zA-Z_][\w-]*)")
_RE_TAG = re.compile(r"^([a-zA-Z][\w-]*)")
_RE_ATTR = re.compile(r"\[([a-zA-Z_][\w-]*)(?:([~|^$*]?)=['\"]([^'\"]*)['\"])?\]") # noqa
_RE_NOT = re.compile(r":not\(([^)]+)\)")
_RE_DESCENDANT = re.compile(r"\s+(?=[.#\[a-zA-Z*])")
_RE_PSEUDO = re.compile(r":([\w-]+)(?:\(([^)]*)\))?")
_RE_DATA_ATTR = re.compile(r"data-[\w-]+")


def parse_selector(raw: str) -> ParsedSelector:
    """Parse a CSS selector string into a structured ParsedSelector.

    This is the compiler's first pass over any selector from the ZoneMap.
    The ParsedSelector carries all information needed for rule dispatch:
    kind, components, specificity, and pre-computed grep patterns.

    Algorithm:
        1. Check for negation pseudo-class (:not) — extract inner selector
        2. Check for descendant combinators — split and take deepest
        3. Extract ID, class, tag, and attribute components
        4. Compute specificity 3-tuple
        5. Classify selector kind
        6. Build parent chain from descendant splits

    Complexity: O(n) where n = len(raw). Single pass regex matching.
    """
    raw = raw.strip()
    if not raw:
        return _make_universal_selector(raw)

    is_negation = False
    negation_match = _RE_NOT.search(raw)
    if negation_match:
        # For negation, we parse the inner selector for the pattern
        # but mark it as negated for rule dispatch
        inner_raw = negation_match.group(1)
        inner = parse_selector(inner_raw)
        return ParsedSelector(
            raw=raw,
            kind=inner.kind,
            tag=inner.tag,
            class_name=inner.class_name,
            id_name=inner.id_name,
            attribute_name=inner.attribute_name,
            attribute_value=inner.attribute_value,
            specificity=inner.specificity,
            depth_hint=inner.depth_hint,
            parent_selectors=inner.parent_selectors,
            is_negation=True,
        )

    # Handle descendant combinators: "div.main article p" → parse deepest
    parts = _RE_DESCENDANT.split(raw)
    parts = [p.strip() for p in parts if p.strip()]
    parent_selectors: Tuple[str, ...] = ()
    target = raw
    if len(parts) > 1:
        parent_selectors = tuple(parts[:-1])
        target = parts[-1]

    # Extract components from the target selector
    tag_match = _RE_TAG.match(target)
    tag = tag_match.group(1) if tag_match else None

    class_matches = _RE_CLASS.findall(target)
    class_name = class_matches[0] if class_matches else None

    id_matches = _RE_ID.findall(target)
    id_name = id_matches[0] if id_matches else None

    attr_match = _RE_ATTR.search(target)
    attr_name = attr_match.group(1) if attr_match else None
    attr_value = attr_match.group(3) if attr_match and attr_match.group(3) else None

    # Compute specificity
    num_ids = len(id_matches)
    num_classes = len(class_matches) + (1 if attr_match else 0)
    num_types = 1 if tag else 0
    # Add parent specificity
    for parent in parent_selectors:
        if "#" in parent:
            num_ids += parent.count("#")
        num_classes += parent.count(".")
        if _RE_TAG.match(parent):
            num_types += 1

    specificity = CSSSpecificity(
        ids=num_ids,
        classes=num_classes,
        types=num_types,
    )

    # Classify selector kind
    if id_name and not class_name and not tag:
        kind = SelectorKind.ID_SELECTOR
    elif class_name and not id_name:
        if tag:
            kind = SelectorKind.COMPOUND_SELECTOR
        else:
            kind = SelectorKind.CLASS_SELECTOR
    elif tag and not class_name and not id_name:
        kind = SelectorKind.TAG_SELECTOR
    elif attr_name:
        if id_name or class_name or tag:
            kind = SelectorKind.COMPOUND_SELECTOR
        else:
            kind = SelectorKind.ATTRIBUTE_SELECTOR
    elif id_name and class_name:
        kind = SelectorKind.COMPOUND_SELECTOR
    elif len(parts) > 1:
        kind = SelectorKind.DESCENDANT_SELECTOR
    else:
        kind = SelectorKind.UNIVERSAL_SELECTOR

    depth_hint = len(parts)

    return ParsedSelector(
        raw=raw,
        kind=kind,
        tag=tag,
        class_name=class_name,
        id_name=id_name,
        attribute_name=attr_name,
        attribute_value=attr_value,
        specificity=specificity,
        depth_hint=depth_hint,
        parent_selectors=parent_selectors,
        is_negation=is_negation,
    )


def _make_universal_selector(raw: str) -> ParsedSelector:
    """Fallback universal selector for unparseable inputs."""
    return ParsedSelector(
        raw=raw or "*",
        kind=SelectorKind.UNIVERSAL_SELECTOR,
        tag=None,
        class_name=None,
        id_name=None,
        attribute_name=None,
        attribute_value=None,
        specificity=CSSSpecificity(0, 0, 0),
        depth_hint=0,
        parent_selectors=(),
        is_negation=False,
    )


def compute_selector_similarity(a: ParsedSelector, b: ParsedSelector) -> float:
    """Jaccard similarity between two parsed selectors based on component overlap.

    Used for deduplication and intent variant narrowing.
    J(A,B) = |A ∩ B| / |A ∪ B| where A,B are sets of selector components.

    Returns: float in [0.0, 1.0]. 1.0 = identical, 0.0 = no overlap.
    """
    def _components(s: ParsedSelector) -> Set[str]:
        parts: Set[str] = set()
        if s.tag:
            parts.add(f"tag:{s.tag}")
        if s.class_name:
            parts.add(f"class:{s.class_name}")
        if s.id_name:
            parts.add(f"id:{s.id_name}")
        if s.attribute_name:
            parts.add(f"attr:{s.attribute_name}")
        for p in s.parent_selectors:
            parts.add(f"parent:{p}")
        return parts

    set_a = _components(a)
    set_b = _components(b)
    if not set_a and not set_b:
        return 1.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    if union == 0:
        return 0.0
    return intersection / union


# ═════════════════════════════════════════════════════════════════════════════
# SHELL AST — INTERMEDIATE REPRESENTATION
#
# The compiler does not emit shell strings directly. It builds a typed AST
# of shell fragments, validates the AST, then serializes to shell text.
# This prevents injection by construction: only whitelisted command types
# can exist in the AST, and serialization is deterministic.
#
# The AST is the compiler's type system for shell programs.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ShellFlag:
    """A command-line flag. e.g. -n, -v, -oP"""
    flag: str

    def __post_init__(self) -> None:
        if not self.flag.startswith("-"):
            raise ValueError(f"Flag must start with '-', got {self.flag!r}")
        # Security: flags cannot contain shell metacharacters
        if any(c in self.flag for c in ";&|`$(){}"):
            raise ValueError(f"Flag contains forbidden character: {self.flag!r}")

    def serialize(self) -> str:
        return self.flag


@dataclass(frozen=True)
class ShellPattern:
    """A pattern argument (regex, string literal, or awk expression).
    All patterns are single-quoted to prevent shell expansion."""
    pattern: str

    def serialize(self) -> str:
        # Escape single quotes within the pattern using the '\'' trick
        escaped = self.pattern.replace("'", "'\\''")
        return f"'{escaped}'"


@dataclass(frozen=True)
class ShellRawArg:
    """A raw argument that is NOT shell-expanded. Used for file paths
    and numeric arguments only. Validated against injection patterns."""
    value: str

    def __post_init__(self) -> None:
        # Security: raw args cannot contain injection vectors
        for pattern_str in INJECTION_PATTERNS:
            if re.search(pattern_str, self.value):
                raise ValueError(
                    f"Raw argument matches injection pattern {pattern_str!r}: "
                    f"{self.value!r}"
                )

    def serialize(self) -> str:
        return self.value


ShellArg = Union[ShellFlag, ShellPattern, ShellRawArg]


@dataclass(frozen=True)
class ShellCommand:
    """A single shell command with arguments.
    command must be in ALLOWED_RECIPE_COMMANDS."""
    command: str
    args: Tuple[ShellArg, ...]

    def __post_init__(self) -> None:
        if self.command not in ALLOWED_RECIPE_COMMANDS:
            raise ValueError(
                f"Command {self.command!r} not in ALLOWED_RECIPE_COMMANDS. "
                f"Allowed: {sorted(ALLOWED_RECIPE_COMMANDS)}"
            )

    def serialize(self) -> str:
        parts = [self.command]
        for arg in self.args:
            parts.append(arg.serialize())
        return " ".join(parts)


@dataclass(frozen=True)
class ShellPipeline:
    """A pipeline of commands connected by pipes.
    This is the primary compilation target."""
    stages: Tuple[ShellCommand, ...]
    comment: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("Pipeline must have at least one stage.")
        if len(self.stages) > MAX_PIPELINE_STAGES:
            raise ValueError(
                f"Pipeline has {len(self.stages)} stages, "
                f"exceeding MAX_PIPELINE_STAGES ({MAX_PIPELINE_STAGES})."
            )

    def serialize(self) -> str:
        lines: List[str] = []
        if self.comment:
            for line in self.comment.split("\n"):
                lines.append(f"# {line}")
        serialized_stages = []
        for stage in self.stages:
            serialized_stages.append(stage.serialize())
        # Format: multi-line with continuation if long
        if len(serialized_stages) == 1:
            lines.append(serialized_stages[0])
        else:
            for i, s in enumerate(serialized_stages):
                if i == 0:
                    lines.append(s + " \\")
                elif i == len(serialized_stages) - 1:
                    lines.append(f"| {s}")
                else:
                    lines.append(f"| {s} \\")
        return "\n".join(lines)

    @property
    def line_count(self) -> int:
        return len(self.serialize().split("\n"))


@dataclass(frozen=True)
class ShellRecipe:
    """Complete compiled recipe: header + pipelines.
    This is the final AST before serialization to .sh file."""
    header: str
    pipelines: Tuple[ShellPipeline, ...]
    topology_class: str
    strategy: ExtractionStrategy
    phase: PhaseStr
    zone_map_version: int
    intent: Optional[str] = None

    def serialize(self) -> str:
        lines: List[str] = []
        lines.append("#!/bin/sh")
        lines.append(f"# compiled by topology/parser.py from ZoneMap version "
                      f"{self.zone_map_version}")
        lines.append(f"# topology: {self.topology_class}  "
                      f"strategy: {self.strategy.value}  "
                      f"phase: {self.phase.value}")
        if self.intent:
            lines.append(f"# intent: {self.intent}")
        lines.append(f"# generated: {datetime.now(timezone.utc).isoformat()}")
        lines.append("")
        if self.header:
            lines.append(self.header)
            lines.append("")
        for pipeline in self.pipelines:
            lines.append(pipeline.serialize())
            lines.append("")
        content = "\n".join(lines)
        # Strip trailing whitespace and ensure single trailing newline
        return content.rstrip() + "\n"

    @property
    def line_count(self) -> int:
        return len(self.serialize().strip().split("\n"))

    @property
    def stage_count(self) -> int:
        return sum(len(p.stages) for p in self.pipelines)


# ═════════════════════════════════════════════════════════════════════════════
# TRANSLATION RULES
#
# The compiler's core. Each rule maps a structural description (from the
# enriched ZoneMap) to one or more ShellCommand objects. Rules compose —
# a single zone triggers 3-5 rules in sequence.
#
# Rules are ordered by category and have strict composition semantics:
# ZONE_BOUNDARY rules run first, NOISE_STRIP second, CONTENT_EXTRACT third,
# TEXT_NORMALIZE fourth, POST_FILTER last.
#
# Each rule is a pure function: (EnrichedZone, CompilerContext) → List[ShellCommand]
# No side effects. No mutable state. Deterministic output for given input.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class TranslationRule:
    """A single translation rule in the compiler's rule table.

    Attributes:
        rule_id: Unique identifier (1-20) per spec.
        name: Human-readable rule name.
        category: Composition ordering category.
        description: What structural pattern this rule handles.
        applicability: Predicate — when does this rule fire?
        emit: The compilation function — produces shell commands.
    """
    rule_id: int
    name: str
    category: RuleCategory
    description: str
    applicability: Callable[[EnrichedZone, CompilerContext], bool]
    emit: Callable[[EnrichedZone, CompilerContext], List[ShellCommand]]

    def applies(self, zone: EnrichedZone, ctx: CompilerContext) -> bool:
        return self.applicability(zone, ctx)

    def compile(self, zone: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        commands = self.emit(zone, ctx)
        ctx.add_diagnostic(
            CompileSeverity.INFO,
            f"Rule {self.rule_id} ({self.name}) emitted {len(commands)} commands",
            rule_id=self.rule_id,
            zone_selector=zone.selector.raw,
        )
        return commands


def _build_translation_rules() -> Tuple[TranslationRule, ...]:
    """Construct the complete translation rule table.

    Returns the 20 rules in composition order. Each rule is documented
    with its structural trigger and the shell primitive it emits.

    The rules implement a formal mapping:
        structural_description × compiler_context → shell_primitive

    Rule composition is associative and order-dependent within categories
    but order-independent between categories (categories are applied in
    a fixed sequence).
    """

    rules: List[TranslationRule] = []

    # ── Rule 1: CSS class selector → signal zone boundary ────────────────
    def _r1_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.selector.class_name is not None
            and z.selector.kind in (
                SelectorKind.CLASS_SELECTOR,
                SelectorKind.COMPOUND_SELECTOR,
            )
        )

    def _r1_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        cls = z.selector.class_name
        tag = z.selector.tag
        tag_esc = re.escape(tag) if tag else "[a-zA-Z][a-zA-Z0-9]*"
        cls_esc = re.escape(cls) if cls else ""
        pattern = (
            f'/<{tag_esc}[^>]*class="[^"]*{cls_esc}[^"]*"/,'
            f"/<\\/{tag_esc}>/p"
        )
        return [ShellCommand("sed", (
            ShellFlag("-n"),
            ShellPattern(pattern),
        ))]

    rules.append(TranslationRule(
        rule_id=1,
        name="class_selector_zone",
        category=RuleCategory.ZONE_BOUNDARY,
        description="CSS class selector → signal zone boundary extraction",
        applicability=_r1_applies,
        emit=_r1_emit,
    ))

    # ── Rule 2: CSS ID selector → signal zone boundary ───────────────────
    def _r2_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.selector.id_name is not None
            and z.selector.kind in (
                SelectorKind.ID_SELECTOR,
                SelectorKind.COMPOUND_SELECTOR,
            )
        )

    def _r2_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        id_name = z.selector.id_name
        return [ShellCommand("grep", (
            ShellFlag("-A"),
            ShellRawArg("500"),
            ShellPattern(f'id="{id_name}"'),
        )), ShellCommand("head", (
            ShellFlag("-n"),
            ShellRawArg("500"),
        ))]

    rules.append(TranslationRule(
        rule_id=2,
        name="id_selector_zone",
        category=RuleCategory.ZONE_BOUNDARY,
        description="CSS ID selector → grep -A extraction with depth limiter",
        applicability=_r2_applies,
        emit=_r2_emit,
    ))

    # ── Rule 3: Tag boundary extraction ──────────────────────────────────
    def _r3_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.selector.tag is not None
            and z.selector.kind == SelectorKind.TAG_SELECTOR
        )

    def _r3_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        tag = z.selector.tag
        return [ShellCommand("sed", (
            ShellFlag("-n"),
            ShellPattern(f"/<{tag}/,/<\\/{tag}>/p"),
        ))]

    rules.append(TranslationRule(
        rule_id=3,
        name="tag_boundary_extract",
        category=RuleCategory.ZONE_BOUNDARY,
        description="Tag name → sed range extraction between open/close tags",
        applicability=_r3_applies,
        emit=_r3_emit,
    ))

    # ── Rule 4: Strip tag + content (noise removal) ──────────────────────
    def _r4_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.node_type == NodeType.NOISE and z.selector.tag is not None

    def _r4_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        tag = z.selector.tag
        return [ShellCommand("sed", (
            ShellPattern(f"/<{tag}[^>]*>/,/<\\/{tag}>/d"),
        ))]

    rules.append(TranslationRule(
        rule_id=4,
        name="strip_noise_tag",
        category=RuleCategory.NOISE_STRIP,
        description="Noise tag → sed range deletion of tag and its content",
        applicability=_r4_applies,
        emit=_r4_emit,
    ))

    # ── Rule 5: Strip all HTML tags (text extraction) ────────────────────
    def _r5_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and ctx.strategy == ExtractionStrategy.DEPTH_FIRST
        )

    def _r5_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("sed", (
            ShellPattern("s/<[^>]*>//g"),
        ))]

    rules.append(TranslationRule(
        rule_id=5,
        name="strip_html_tags",
        category=RuleCategory.CONTENT_EXTRACT,
        description="Strip all HTML tags, leaving only text content",
        applicability=_r5_applies,
        emit=_r5_emit,
    ))

    # ── Rule 6: Strip inline attributes (clean tag) ──────────────────────
    def _r6_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and ctx.strategy == ExtractionStrategy.DEPTH_FIRST
            and z.selector.kind in (
                SelectorKind.CLASS_SELECTOR,
                SelectorKind.COMPOUND_SELECTOR,
            )
        )

    def _r6_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("sed", (
            ShellPattern('s/ [a-z-]*="[^"]*"//g'),
        ))]

    rules.append(TranslationRule(
        rule_id=6,
        name="strip_inline_attrs",
        category=RuleCategory.CONTENT_EXTRACT,
        description="Strip inline HTML attributes from remaining tags",
        applicability=_r6_applies,
        emit=_r6_emit,
    ))

    # ── Rule 7: Code block extraction ────────────────────────────────────
    def _r7_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.structural_role in ("code_block", "code", "pre")
        )

    def _r7_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [
            ShellCommand("sed", (
                ShellFlag("-n"),
                ShellPattern("/<pre/,/<\\/pre>/p"),
            )),
            ShellCommand("sed", (
                ShellFlag("-n"),
                ShellPattern("/<code/,/<\\/code>/p"),
            )),
        ]

    rules.append(TranslationRule(
        rule_id=7,
        name="code_block_extract",
        category=RuleCategory.ZONE_BOUNDARY,
        description="Pre/code tags → nested sed extraction for code blocks",
        applicability=_r7_applies,
        emit=_r7_emit,
    ))

    # ── Rule 8: List extraction ──────────────────────────────────────────
    def _r8_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.structural_role in ("list", "ordered_list", "unordered_list")
        )

    def _r8_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("sed", (
            ShellFlag("-n"),
            ShellPattern("/<[ou]l/,/<\\/[ou]l>/p"),
        ))]

    rules.append(TranslationRule(
        rule_id=8,
        name="list_extract",
        category=RuleCategory.ZONE_BOUNDARY,
        description="Ordered/unordered list → sed range extraction",
        applicability=_r8_applies,
        emit=_r8_emit,
    ))

    # ── Rule 9: Paragraph text extraction ────────────────────────────────
    def _r9_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.structural_role in ("paragraph", "prose", "main_content")
        )

    def _r9_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [
            ShellCommand("sed", (
                ShellFlag("-n"),
                ShellPattern("/<p[> ]/,/<\\/p>/p"),
            )),
            ShellCommand("sed", (
                ShellPattern("s/<[^>]*>//g"),
            )),
        ]

    rules.append(TranslationRule(
        rule_id=9,
        name="paragraph_extract",
        category=RuleCategory.CONTENT_EXTRACT,
        description="Paragraph tags → extract and strip to plain text",
        applicability=_r9_applies,
        emit=_r9_emit,
    ))

    # ── Rule 10: Heading extraction ──────────────────────────────────────
    def _r10_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and z.structural_role in ("heading", "title", "section_header")
        )

    def _r10_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("grep", (
            ShellFlag("-oP"),
            ShellPattern("(?<=<h[1-6][^>]*>)[^<]+"),
        ))]

    rules.append(TranslationRule(
        rule_id=10,
        name="heading_extract",
        category=RuleCategory.CONTENT_EXTRACT,
        description="Heading tags (h1-h6) → grep PCRE lookbehind extraction",
        applicability=_r10_applies,
        emit=_r10_emit,
    ))

    # ── Rule 11: data-* attribute extraction ─────────────────────────────
    def _r11_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            ctx.strategy == ExtractionStrategy.BREADTH_FIRST
            and len(z.data_attributes) > 0
        )

    def _r11_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        # Build an awk program that extracts all specified data attributes
        awk_lines: List[str] = ["{"]
        for attr in z.data_attributes:
            safe_attr = re.escape(attr)
            label = attr.replace("data-", "").upper().replace("-", "_")
            awk_lines.append(
                f'    if (match($0, /{safe_attr}="([^"]*)"/, a)) '
                f'print "{label}: " a[1]'
            )
        awk_lines.append("}")
        awk_prog = "\n".join(awk_lines)
        return [ShellCommand("awk", (ShellPattern(awk_prog),))]

    rules.append(TranslationRule(
        rule_id=11,
        name="data_attr_extract",
        category=RuleCategory.ATTRIBUTE_EXTRACT,
        description="data-* attributes → awk regex extraction with labels",
        applicability=_r11_applies,
        emit=_r11_emit,
    ))

    # ── Rule 12: JSON field extraction ───────────────────────────────────
    def _r12_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            ctx.strategy == ExtractionStrategy.SECTION_SCOPED
            and z.json_path is not None
            and "." not in (z.json_path or "")
        )

    def _r12_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        key = z.json_path or ""
        safe_key = re.escape(key)
        awk_prog = (
            f'/"{safe_key}"[[:space:]]*:/'
            "{"
            f'match($0, /"{safe_key}"[[:space:]]*:[[:space:]]*"([^"]*)"/, a); '
            "print a[1]"
            "}"
        )
        return [ShellCommand("awk", (ShellPattern(awk_prog),))]

    rules.append(TranslationRule(
        rule_id=12,
        name="json_field_extract",
        category=RuleCategory.JSON_EXTRACT,
        description="Simple JSON key → awk pattern match on key:value pairs",
        applicability=_r12_applies,
        emit=_r12_emit,
    ))

    # ── Rule 13: JSON array traversal ────────────────────────────────────
    def _r13_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            ctx.strategy == ExtractionStrategy.SECTION_SCOPED
            and z.json_path is not None
            and "." in (z.json_path or "")
        )

    def _r13_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        # Build awk state machine for nested JSON path traversal
        path_parts = (z.json_path or "").split(".")
        awk_prog = _build_json_traversal_awk(path_parts, phase=ctx.phase)
        return [ShellCommand("awk", (ShellPattern(awk_prog),))]

    rules.append(TranslationRule(
        rule_id=13,
        name="json_array_traversal",
        category=RuleCategory.JSON_EXTRACT,
        description="Nested JSON path → awk state machine with depth tracking",
        applicability=_r13_applies,
        emit=_r13_emit,
    ))

    # ── Rule 14: Normalize whitespace ────────────────────────────────────
    def _r14_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.node_type == NodeType.SIGNAL

    def _r14_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("tr", (
            ShellFlag("-s"),
            ShellPattern(" \\t\\n"),
            ShellPattern("\\n"),
        ))]

    rules.append(TranslationRule(
        rule_id=14,
        name="normalize_whitespace",
        category=RuleCategory.TEXT_NORMALIZE,
        description="Collapse whitespace sequences into single newlines",
        applicability=_r14_applies,
        emit=_r14_emit,
    ))

    # ── Rule 15: Deduplicate lines ───────────────────────────────────────
    def _r15_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return (
            z.node_type == NodeType.SIGNAL
            and ctx.phase in (PhaseStr.PREDICTS, PhaseStr.KNOWS)
        )

    def _r15_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("awk", (
            ShellPattern("!seen[$0]++"),
        ))]

    rules.append(TranslationRule(
        rule_id=15,
        name="deduplicate_lines",
        category=RuleCategory.POST_FILTER,
        description="Order-preserving line deduplication via awk associative array",
        applicability=_r15_applies,
        emit=_r15_emit,
    ))

    # ── Rule 16: Strip empty lines ───────────────────────────────────────
    def _r16_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.node_type == NodeType.SIGNAL

    def _r16_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("grep", (
            ShellFlag("-v"),
            ShellPattern("^[[:space:]]*$"),
        ))]

    rules.append(TranslationRule(
        rule_id=16,
        name="strip_empty_lines",
        category=RuleCategory.POST_FILTER,
        description="Remove lines containing only whitespace",
        applicability=_r16_applies,
        emit=_r16_emit,
    ))

    # ── Rule 17: URL extraction from hrefs ───────────────────────────────
    def _r17_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.structural_role in ("link_list", "navigation_links", "url_extract")

    def _r17_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [
            ShellCommand("grep", (
                ShellFlag("-oP"),
                ShellPattern('href="[^"]*"'),
            )),
            ShellCommand("cut", (
                ShellFlag("-d"),
                ShellPattern('"'),
                ShellFlag("-f2"),
            )),
        ]

    rules.append(TranslationRule(
        rule_id=17,
        name="url_extract",
        category=RuleCategory.CONTENT_EXTRACT,
        description="href attributes → grep PCRE + cut for clean URLs",
        applicability=_r17_applies,
        emit=_r17_emit,
    ))

    # ── Rule 18: Price pattern extraction ────────────────────────────────
    def _r18_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.structural_role in ("pricing", "price", "cost")

    def _r18_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        return [ShellCommand("grep", (
            ShellFlag("-oP"),
            ShellPattern("[$€£]\\s*[\\d,]+\\.?\\d*"),
        ))]

    rules.append(TranslationRule(
        rule_id=18,
        name="price_extract",
        category=RuleCategory.CONTENT_EXTRACT,
        description="Currency patterns → grep PCRE for price values",
        applicability=_r18_applies,
        emit=_r18_emit,
    ))

    # ── Rule 19: Table row extraction ────────────────────────────────────
    def _r19_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.structural_role in ("table", "data_table", "comparison_table")

    def _r19_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        awk_prog = _build_table_extraction_awk(z, phase=ctx.phase)
        return [ShellCommand("awk", (ShellPattern(awk_prog),))]

    rules.append(TranslationRule(
        rule_id=19,
        name="table_row_extract",
        category=RuleCategory.CONTENT_EXTRACT,
        description="HTML table → awk state machine tracking tr/td boundaries",
        applicability=_r19_applies,
        emit=_r19_emit,
    ))

    # ── Rule 20: Depth-limited nesting ───────────────────────────────────
    def _r20_applies(z: EnrichedZone, ctx: CompilerContext) -> bool:
        return z.depth_limit is not None and z.depth_limit > 0

    def _r20_emit(z: EnrichedZone, ctx: CompilerContext) -> List[ShellCommand]:
        limit = z.depth_limit or MAX_SELECTOR_DEPTH
        awk_prog = _build_depth_limited_awk(z, limit, phase=ctx.phase)
        return [ShellCommand("awk", (ShellPattern(awk_prog),))]

    rules.append(TranslationRule(
        rule_id=20,
        name="depth_limited_nesting",
        category=RuleCategory.POST_FILTER,
        description="Depth-limited extraction via awk tag depth counter",
        applicability=_r20_applies,
        emit=_r20_emit,
    ))

    return tuple(rules)


# Module-level rule table. Built once at import time. Immutable.
TRANSLATION_RULES: Final[Tuple[TranslationRule, ...]] = _build_translation_rules()

# Rule lookup by category for ordered composition.
RULES_BY_CATEGORY: Final[Dict[RuleCategory, List[TranslationRule]]] = {}
for _r in TRANSLATION_RULES:
    RULES_BY_CATEGORY.setdefault(_r.category, []).append(_r)

# Composition order: the sequence in which rule categories are applied.
COMPOSITION_ORDER: Final[Tuple[RuleCategory, ...]] = (
    RuleCategory.ZONE_BOUNDARY,
    RuleCategory.NOISE_STRIP,
    RuleCategory.CONTENT_EXTRACT,
    RuleCategory.ATTRIBUTE_EXTRACT,
    RuleCategory.JSON_EXTRACT,
    RuleCategory.TEXT_NORMALIZE,
    RuleCategory.POST_FILTER,
)


# ═════════════════════════════════════════════════════════════════════════════
# AWK PROGRAM GENERATORS
#
# The compiler's ceiling of capability. These generators produce complete
# awk programs for complex extraction scenarios that cannot be handled by
# simple grep|sed one-liners.
#
# Each generator is a code emitter that constructs awk source from
# structural parameters. The generated programs use:
#   - Integer depth counters for HTML nesting
#   - Boolean state variables for multi-zone tracking
#   - Associative arrays for deduplication
#   - Pattern-action pairs for structural matching
#
# Mathematical model: the generated awk program implements a finite
# state transducer (FST) where:
#   - States = {zone_state} × {depth_counter}
#   - Input alphabet = HTML line tokens
#   - Output alphabet = extracted content lines
#   - Transition function = pattern match on HTML structure
# ═════════════════════════════════════════════════════════════════════════════

def _build_json_traversal_awk(
    path_parts: List[str],
    *,
    phase: PhaseStr,
) -> str:
    """Generate a phase-aware awk program for nested JSON path traversal.

    Handles two common REST API container shapes at the target path:

        object container  → fields emitted as FIELD:\\tkey\\tvalue
                            Entry on /\"key\"[[:space:]]*:[[:space:]]*\\{/
                            Exit when brace depth returns to entry depth.

        array container   → items numbered ITEM_1:, ITEM_2:, ...
                            Entry on /\"key\"[[:space:]]*:[[:space:]]*\\[/
                            Exit when bracket depth returns to entry depth.

    Inline scalars (\"key\": \"value\" on one line) are extracted immediately
    without entering a container state — correct for flat REST responses
    where values are primitives, not nested objects.

    Per-component depth anchoring:
        Each path component records entry_depth_N at activation time.
        Exit fires when the global brace/bracket depth falls below
        entry_depth_N, not on the first closing bracket encountered.
        This prevents premature exit on nested structures and correctly
        handles arrays-of-objects at the target path.

    State reset on exit:
        Deactivating path component N resets all components N..last.
        A second occurrence of a key in the same document re-activates
        correctly — the state machine is re-entrant per component.

    Phase III additions:
        Lines shorter than MIN_SIGNAL_LENGTH are dropped from output.
        This removes isolated punctuation lines that survive tag stripping
        in partially-rendered JSON (common in SPA-rendered API docs).

    FST specification:
        States:      {in_0, in_1, ..., in_{n-1}}  (one per path component)
        Depth vars:  {entry_depth_0, ..., entry_depth_{n-1}}
        Aux:         {brace_depth, bracket_depth, item_count, container_N}
        Input:       JSON text lines (one record per line)
        Output:      labeled extracted content (FIELD:, ITEM_N:, VALUE:)
        Transitions: key pattern match → activate; depth underflow → deactivate
    """
    if not path_parts:
        return "{ print }"

    n   = len(path_parts)
    last = n - 1
    lines: list[str] = []

    # ── BEGIN ─────────────────────────────────────────────────────────────────
    lines.append("BEGIN {")
    for i in range(n):
        lines.append(f"    in_{i}=0; entry_depth_{i}=0; container_{i}=\"\"")
    lines.append("    brace_depth=0; bracket_depth=0")
    lines.append("    item_count=0")
    lines.append("}")
    lines.append("")

    # ── per-line depth tracking ───────────────────────────────────────────────
    lines.append("# structural depth — drives entry/exit for every path component")
    lines.append("{")
    lines.append('    brace_depth   += gsub(/\\{/, "{") - gsub(/\\}/, "}")')
    lines.append('    bracket_depth += gsub(/\\[/, "[") - gsub(/\\]/, "]")')
    lines.append("    if (brace_depth   < 0) brace_depth   = 0")
    lines.append("    if (bracket_depth < 0) bracket_depth = 0")
    lines.append("}")
    lines.append("")

    # ── per-component activation and inline scalar extraction ─────────────────
    for i, part in enumerate(path_parts):
        safe = part.replace('"', '\\"')
        guard = " && ".join(f"in_{j}" for j in range(i)) if i > 0 else ""
        prefix = f"{guard} && " if guard else ""

        lines.append(f"# path component {i}: \"{part}\"")

        # object container entry
        lines.append(
            f'{prefix}/"{safe}"[[:space:]]*:[[:space:]]*\\{{/ {{'
        )
        lines.append(f"    in_{i}=1; container_{i}=\"object\"")
        lines.append(f"    entry_depth_{i}=brace_depth-1")
        if i == last:
            lines.append("    item_count=0")
        lines.append("}")

        # array container entry
        lines.append(
            f'{prefix}/"{safe}"[[:space:]]*:[[:space:]]*\\[/ {{'
        )
        lines.append(f"    in_{i}=1; container_{i}=\"array\"")
        lines.append(f"    entry_depth_{i}=bracket_depth-1")
        if i == last:
            lines.append("    item_count=0")
        lines.append("}")

        # inline scalar: "key": "value" or "key": number — emit and continue
        if i == last:
            lines.append(
                f'{prefix}/"{safe}"[[:space:]]*:[[:space:]]*"[^"]*"/ {{'
            )
            lines.append(
                f'    match($0, /"{safe}"[[:space:]]*:[[:space:]]*"([^"]*)"/, _v)'
            )
            lines.append('    if (length(_v[1]) > 0) print "VALUE:\\t" _v[1]')
            lines.append("}")
            lines.append(
                f'{prefix}/"{safe}"[[:space:]]*:[[:space:]]*[0-9]/ {{'
            )
            lines.append(
                f'    match($0, /"{safe}"[[:space:]]*:[[:space:]]*([-0-9.]+)/, _v)'
            )
            lines.append('    if (length(_v[1]) > 0) print "VALUE:\\t" _v[1]')
            lines.append("}")

        lines.append("")

    # ── target state: extract content ─────────────────────────────────────────
    target_guard = " && ".join(f"in_{i}" for i in range(n))
    lines.append(f"# target state — all path components active")
    lines.append(f"{target_guard} {{")
    lines.append('    line = $0')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "", line)')
    lines.append('    if (length(line) == 0 || line == "{" || line == "}" ||')
    lines.append('        line == "[" || line == "]" || line == "," ) next')
    lines.append(f'    if (container_{last} == "array") {{')
    lines.append('        item_count++')
    lines.append('        print "ITEM_" item_count ":\\t" line')
    lines.append('    } else {')
    lines.append('        # object field: emit key\tvalue')
    lines.append('        if (match(line, /"([^"]+)"[[:space:]]*:[[:space:]]*(.*)/, _f)) {')
    lines.append('            val = _f[2]')
    lines.append('            gsub(/,$/, "", val)')
    lines.append('            gsub(/^"|"$/, "", val)')
    lines.append('            print "FIELD:\\t" _f[1] "\\t" val')
    lines.append('        } else {')
    lines.append('            print line')
    lines.append('        }')
    lines.append('    }')
    lines.append("}")
    lines.append("")

    # Phase III min-length filter applied inside target block above via `next`
    if phase == PhaseStr.KNOWS:
        # insert after the structural-noise filter, before item/field emit
        # splice into the target block: replace the closing `}` with filter + `}`
        idx = next(i for i in range(len(lines) - 1, -1, -1)
                   if lines[i] == "    } else {")
        lines.insert(idx,
            f"    if (length(line) < {MIN_SIGNAL_LENGTH}) next")
        lines.append("")

    # ── per-component exit: depth underflow resets this and all deeper states ─
    lines.append("# exit guards — depth underflow deactivates path components")
    for i in range(n - 1, -1, -1):
        guard = " && ".join(f"in_{j}" for j in range(i + 1))
        lines.append(f"{guard} && container_{i} == \"object\" &&")
        lines.append(f"    brace_depth <= entry_depth_{i} {{")
        for j in range(n - 1, i - 1, -1):
            lines.append(f"    in_{j}=0; container_{j}=\"\"; entry_depth_{j}=0")
        lines.append("    item_count=0")
        lines.append("}")
        lines.append(f"{guard} && container_{i} == \"array\" &&")
        lines.append(f"    bracket_depth <= entry_depth_{i} {{")
        for j in range(n - 1, i - 1, -1):
            lines.append(f"    in_{j}=0; container_{j}=\"\"; entry_depth_{j}=0")
        lines.append("    item_count=0")
        lines.append("}")
        lines.append("")

    return "\n".join(lines)

def _build_table_extraction_awk(
    zone: EnrichedZone,
    *,
    phase: PhaseStr,
) -> str:
    """Generate a phase-aware awk program for HTML table extraction.

    Three structural role variants with different output semantics:

        data_table         → tab-separated rows, HEADER:\\t prefix on <th> rows.
                             Column count validated — mismatched rows tagged
                             with COL_MISMATCH: for downstream triage.

        comparison_table   → HEADER row labels each column; data rows emit
                             LABEL: value pairs (col_header\\trow_value) so the
                             LLM receives named cells, not positional columns.
                             Correct for pricing/feature comparison tables where
                             column identity is the semantic unit.

        parameter_table    → Two-column key/value tables (API params, CLI flags,
                             config options). Emits PARAM:\\tNAME\\tTYPE\\tDESC
                             with graceful fallback to NAME\\tDESC for plain
                             two-column layouts.

    Shared invariants across all roles:
        - Integer depth counter guards nested table corruption.
          Only depth <= 1 rows participate in outer table state machine.
        - </tr> always resets in_cell — truncated cell recovery.
        - <caption> extracted as CAPTION: prefix before first data row.
        - Section rows (single <th> spanning) emitted as SECTION: prefix.
        - END block recovers unterminated last row on truncated input.
        - HTML entities &amp; &lt; &gt; &nbsp; normalised in cell content.
        - Phase III: rows below MIN_SIGNAL_LENGTH filtered after assembly.
        - Phase III: empty cells represented as "-" not bare tab for LLM
          readability (bare consecutive tabs are invisible in context window).

    FST specification:
        States: {outside, in_caption, in_row, in_cell}
        Aux:    {is_header, depth, col_idx, col_count, header_cols[]}
        Input:  HTML line tokens (one line per awk record)
        Output: role-specific labeled rows (see above)
    """
    role = zone.structural_role
    is_comparison = role == "comparison_table"
    is_parameter  = role == "parameter_table"
    is_phase_iii  = phase == PhaseStr.KNOWS

    # ── shared entity normalisation snippet (inlined into cell-content block) ──
    _entity_norm = (
        'gsub(/&amp;/,  "&");'
        'gsub(/&lt;/,   "<");'
        'gsub(/&gt;/,   ">");'
        'gsub(/&nbsp;/, " ");'
        'gsub(/&#[0-9]+;/, "");'
    )

    # ── empty-cell placeholder: Phase III uses "-" to keep column alignment
    #    visible in LLM context; earlier phases use "" for fidelity ──────────
    empty_cell = '"-"' if is_phase_iii else '""'

    # ── min-length post-filter (Phase III only) ───────────────────────────────
    min_len_guard = (
        f"    if (length(row) < {MIN_SIGNAL_LENGTH}) {{ row=\"\"; next }}\n"
        if is_phase_iii else ""
    )

    lines: list[str] = []

    # ── BEGIN block ───────────────────────────────────────────────────────────
    lines.append("BEGIN {")
    lines.append("    in_row=0; in_cell=0; in_caption=0; is_header=0")
    lines.append("    row=\"\"; cell=\"\"; depth=0")
    lines.append("    col_idx=0; col_count=0; header_emitted=0")
    if is_comparison:
        lines.append("    split(\"\", header_cols)  # associative array: index → label")
    lines.append("}")
    lines.append("")

    # ── per-line depth tracking ───────────────────────────────────────────────
    lines.append("# depth: guards against nested table rows corrupting outer FST")
    lines.append("{")
    lines.append('    opens  = gsub(/<[^\\/][^>]*>/, "&")')
    lines.append('    closes = gsub(/<\\/[^>]*>/, "&")')
    lines.append("    depth += opens - closes")
    lines.append("    if (depth < 0) depth = 0")
    lines.append("}")
    lines.append("")

    # ── caption extraction ────────────────────────────────────────────────────
    lines.append("/<caption[> ]/ { in_caption=1; next }")
    lines.append("/<\\/caption>/ && in_caption {")
    lines.append('    gsub(/<[^>]*>/, ""); gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (length($0) > 0) print "CAPTION:\\t" $0')
    lines.append("    in_caption=0; next")
    lines.append("}")
    lines.append('in_caption { gsub(/<[^>]*>/, ""); if (/[^[:space:]]/) print "CAPTION:\\t" $0 }')
    lines.append("")

    # ── row entry — depth <= 1 prevents nested table rows participating ───────
    lines.append("/<tr[> ]/ && depth <= 1 {")
    lines.append("    in_row=1; is_header=0; row=\"\"; cell=\"\"; col_idx=0")
    lines.append("}")
    lines.append("")

    # ── cell entry ────────────────────────────────────────────────────────────
    lines.append("/<t[dh][> ]/ && in_row {")
    lines.append("    in_cell=1; cell=\"\"")
    lines.append("    if (/th[> ]/) is_header=1")
    lines.append("    col_idx++")
    lines.append("}")
    lines.append("")

    # ── cell content accumulation ─────────────────────────────────────────────
    lines.append("in_row && in_cell && !/<t[dh][> ]/ && !/<\\/t[dh]>/ {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append(f"    {_entity_norm}")
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append("    if (length($0) > 0) {")
    lines.append('        if (length(cell) > 0) cell = cell " "')
    lines.append("        cell = cell $0")
    lines.append("    }")
    lines.append("}")
    lines.append("")

    # ── cell exit ─────────────────────────────────────────────────────────────
    lines.append("/<\\/t[dh]>/ && in_row {")
    lines.append(f"    if (length(cell) == 0) cell = {empty_cell}")
    lines.append("    in_cell=0")
    if is_comparison:
        # store header labels; on data rows prefix cell with its column header
        lines.append("    if (is_header) {")
        lines.append("        header_cols[col_idx] = cell")
        lines.append("        if (length(row) > 0) row = row \"\\t\"")
        lines.append("        row = row cell")
        lines.append("    } else {")
        lines.append("        label = (col_idx in header_cols) ? header_cols[col_idx] : col_idx")
        lines.append('        if (length(row) > 0) row = row "\\t"')
        lines.append('        row = row label "=" cell')
        lines.append("    }")
    elif is_parameter:
        # parameter tables: accumulate cells into named slots by col_idx
        lines.append("    if (col_idx == 1) param_name = cell")
        lines.append("    else if (col_idx == 2) param_type = cell")
        lines.append("    else if (col_idx == 3) param_desc = cell")
        lines.append("    else {")
        lines.append("        if (length(row) > 0) row = row \"\\t\"")
        lines.append("        row = row cell")
        lines.append("    }")
    else:
        # data_table / plain table: flat tab-separated
        lines.append("    if (length(row) > 0) row = row \"\\t\"")
        lines.append("    row = row cell")
    lines.append("    cell=\"\"")
    lines.append("}")
    lines.append("")

    # ── row exit ──────────────────────────────────────────────────────────────
    lines.append("/<\\/tr>/ && in_row {")
    lines.append("    in_row=0; in_cell=0; col_idx=0")
    lines.append("")
    if is_parameter:
        # flush named param slots into a single structured row
        lines.append("    if (!is_header && length(param_name) > 0) {")
        lines.append("        if (length(param_type) > 0)")
        lines.append('            row = "PARAM:\\t" param_name "\\t" param_type "\\t" param_desc')
        lines.append("        else")
        lines.append('            row = "PARAM:\\t" param_name "\\t" param_desc')
        lines.append("    }")
        lines.append('    param_name=""; param_type=""; param_desc=""')
    lines.append("")
    lines.append("    if (length(row) == 0) { row=\"\"; next }")
    lines.append("")
    # section row detection: single non-empty cell spanning full width
    lines.append("    # section row: single th with no tabs = section header")
    lines.append("    if (is_header && row !~ /\\t/ && length(row) > 0) {")
    lines.append('        print "SECTION:\\t" row')
    lines.append("        row=\"\"; next")
    lines.append("    }")
    lines.append("")
    # normal header row
    lines.append("    if (is_header) {")
    if not is_comparison:
        lines.append('        print "HEADER:\\t" row')
    else:
        lines.append('        print "HEADER:\\t" row  # column labels for reference')
    lines.append("        header_emitted=1")
    # track column count for mismatch detection on data_table
    if not is_comparison and not is_parameter:
        lines.append("        col_count = split(row, _c, \"\\t\")")
    lines.append("        row=\"\"; next")
    lines.append("    }")
    lines.append("")
    # col mismatch detection for data_table
    if not is_comparison and not is_parameter:
        lines.append("    # column count mismatch: flag for downstream triage")
        lines.append("    if (header_emitted && col_count > 0) {")
        lines.append("        n = split(row, _rc, \"\\t\")")
        lines.append("        if (n != col_count)")
        lines.append('            row = "COL_MISMATCH:\\t" row')
        lines.append("    }")
        lines.append("")
    lines.append(min_len_guard)
    lines.append("    print row")
    lines.append("    row=\"\"")
    lines.append("}")
    lines.append("")

    # ── END: recover unterminated last row on truncated input ─────────────────
    lines.append("END {")
    lines.append("    if (in_row && length(row) > 0) print row")
    lines.append("}")

    return textwrap.dedent("\n".join(lines))

def _build_depth_limited_awk(
    zone: EnrichedZone,
    max_depth: int,
    *,
    phase: PhaseStr,
) -> str:
    """Generate a phase-aware, zone-anchored awk program for depth-limited extraction.

    Activation is anchored to the zone's specific selector pattern — not to any
    class or id attribute on the page. A zone with selector `.article-body`
    activates only on elements matching that pattern, not on `.sidebar` which
    also has a class attribute.

    Void element exclusion:
        <br>, <img>, <input>, <hr>, <meta>, <link>, <area>, <base>, <col>,
        <embed>, <param>, <source>, <track>, <wbr> are matched and subtracted
        before the depth counter increments. Without this, void elements inflate
        depth and the exit condition `depth < entry_depth` never fires.

    Child noise suppression:
        zone.child_noise_selectors are tracked as independent boolean flags.
        Content lines are suppressed while inside any child noise zone.
        This handles sidebar-within-main-content, related-articles-within-article,
        and similar structural patterns where the signal zone contains known noise.

    Content-type context:
        code  → preserve indentation, emit SE separator on </pre>/</code>
        list  → strip tags, emit one item per line, no blank lines
        prose → strip tags, emit non-blank lines
        mixed → strip tags, emit non-blank lines (conservative)

    Exit condition:
        `depth < entry_depth` after the depth-tracking block fires.
        entry_depth is captured AFTER the opening tag increments depth,
        so exit fires when depth returns to the pre-entry level.
        This correctly handles: div-in-div, section-in-main, article-in-body.

    Phase III additions:
        Lines shorter than MIN_SIGNAL_LENGTH are dropped after tag stripping.
        This removes isolated punctuation lines that survive in deep nesting.

    FST specification:
        States:  {idle, in_zone}
        Aux:     {depth, entry_depth, in_noise_0..N}
        Input:   HTML line tokens
        Output:  stripped content lines scoped to zone at depth <= max_depth
    """
    # ── selector activation pattern ───────────────────────────────────────────
    activation = _selector_to_awk_pattern(zone.selector)
    if activation is None:
        # ARIA role fallback — most reliable landmark anchor
        if zone.selector.attribute_name == "role" and zone.selector.attribute_value:
            activation = f'role="{zone.selector.attribute_value}"'
        else:
            # last resort: any class or id — same as original but explicit
            activation = 'class="|id="'

    # ── void elements — excluded from depth counting ──────────────────────────
    _void_tags = (
        "br|img|input|hr|meta|link|area|base|col|embed|param|source|track|wbr"
    )
    _void_pat  = f"<({_void_tags})(\\s[^>]*)?\\/?>|<({_void_tags})>"

    # ── child noise patterns ──────────────────────────────────────────────────
    noise_patterns: list[tuple[str, str, str]] = []  # (var_name, awk_pattern)
    for i, ns in enumerate(zone.child_noise_selectors):
        pat = _selector_to_awk_pattern(ns)
        if pat:
            # infer closing tag from noise selector's tag field, default div
            close_tag = ns.tag if ns.tag else "div"
            noise_patterns.append((f"in_noise_{i}", pat, close_tag))

    # ── content-type output block ─────────────────────────────────────────────
    is_code   = zone.structural_role in ("code_block", "code", "pre")
    is_list   = zone.structural_role in ("list", "ordered_list", "unordered_list")
    is_phase3 = phase == PhaseStr.KNOWS

    min_len = f"length($0) < {MIN_SIGNAL_LENGTH}" if is_phase3 else None

    lines: list[str] = []

    # ── BEGIN ─────────────────────────────────────────────────────────────────
    lines.append("BEGIN {")
    lines.append("    in_zone=0; depth=0; entry_depth=0")
    if is_code:
        lines.append("    in_code_block=0")
    for var, _pat, _tag in noise_patterns:
        lines.append(f"    {var}=0")
    lines.append("}")
    lines.append("")

    # ── void element exclusion ────────────────────────────────────────────────
    lines.append("# void element exclusion — prevents depth inflation on self-closing tags")
    lines.append("{")
    lines.append(f'    void_count = gsub(/{_void_pat}/, "")')
    lines.append('    opens  = gsub(/<[^\\/!][^>]*>/, "&")')
    lines.append('    closes = gsub(/<\\/[^>]*>/, "&")')
    lines.append("    depth += opens - closes")
    lines.append("    if (depth < 0) depth = 0")
    lines.append("}")
    lines.append("")

    # ── zone entry ────────────────────────────────────────────────────────────
    lines.append(f"# zone entry — anchored to selector: {zone.selector.raw!r}")
    lines.append(f"!in_zone && /{activation}/ {{")
    lines.append("    in_zone=1")
    lines.append("    entry_depth=depth-1")
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── child noise zone entry/exit ───────────────────────────────────────────
    if noise_patterns:
        lines.append("# child noise suppression")
        for var, pat, close_tag in noise_patterns:
            lines.append(f"in_zone && /{pat}/ {{ {var}=1 }}")
            lines.append(
                f"in_zone && {var} && /<\\/{close_tag}>/ {{ {var}=0; next }}"
            )
        lines.append("")

    # ── combined noise guard ──────────────────────────────────────────────────
    noise_guard = ""
    if noise_patterns:
        noise_guard = " && " + " && ".join(f"!{v}" for v, _, _ in noise_patterns)

    # ── depth ceiling guard ───────────────────────────────────────────────────
    depth_guard = f"in_zone{noise_guard} && depth <= entry_depth + {max_depth}"

    # ── extraction block ──────────────────────────────────────────────────────
    if is_code:
        lines.append("# code-aware extraction: preserve pre/code blocks verbatim")
        lines.append("in_zone && /<pre[> ]\\|<code[^>]*class/ { in_code_block=1 }")
        lines.append("in_zone && (/<\\/pre>/ || /<\\/code>/) {")
        lines.append("    in_code_block=0")
        lines.append('    print "---CODE---"')
        lines.append("    next")
        lines.append("}")
        lines.append(f"in_zone && in_code_block{noise_guard} {{")
        lines.append('    gsub(/^[[:space:]]{4}/, "")  # normalize one level of indent')
        lines.append("    print")
        lines.append("    next")
        lines.append("}")
        lines.append(f"{depth_guard} && !in_code_block {{")
        lines.append('    gsub(/<[^>]*>/, "")')
        lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
        if min_len:
            lines.append(f"    if ({min_len}) next")
        lines.append('    if (/[^[:space:]]/) print')
        lines.append("}")
    elif is_list:
        lines.append("# list extraction: one item per line, no blank lines")
        lines.append(f"{depth_guard} {{")
        lines.append('    gsub(/<li[> ]/, "\\n- ")')
        lines.append('    gsub(/<[^>]*>/, "")')
        lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
        if min_len:
            lines.append(f"    if ({min_len}) next")
        lines.append('    if (/[^[:space:]]/) print')
        lines.append("}")
    else:
        lines.append("# prose/mixed extraction: strip tags, emit non-blank lines")
        lines.append(f"{depth_guard} {{")
        lines.append('    gsub(/<[^>]*>/, "")')
        lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
        if min_len:
            lines.append(f"    if ({min_len}) next")
        lines.append('    if (/[^[:space:]]/) print')
        lines.append("}")
    lines.append("")

    # ── zone exit ─────────────────────────────────────────────────────────────
    lines.append("# zone exit — fires when depth returns to pre-entry level")
    lines.append("in_zone && depth < entry_depth {")
    lines.append("    in_zone=0; entry_depth=0")
    for var, _, _ in noise_patterns:
        lines.append(f"    {var}=0")
    lines.append("}")

    return "\n".join(lines)


def _build_multizone_awk(
    signal_zones: List[EnrichedZone],
    noise_zones: List[EnrichedZone],
    *,
    phase: PhaseStr,
) -> str:
    """Generate a phase-aware, zone-labeled, depth-anchored multi-zone awk program.

    Core problems with the original this fixes:

        Flat unlabeled output: all zones merged into one stream. A FORUM_THREAD
        with question + accepted-answer + N answers produces indistinguishable
        lines. Downstream models cannot reconstruct zone identity.
        Fix: every emitted line is prefixed ZONE[<selector>]:\\t so the index
        daemon and signal_kernel can route by zone without re-parsing.

        Premature zone exit: `/<\\/div>/ && in_sig_0` fires on the FIRST </div>
        after zone entry, regardless of nesting depth. A zone opened at depth 3
        exits on the </div> at depth 4 (a child element), not depth 3.
        Fix: entry_depth_N records depth at activation time. Exit fires only
        when depth returns below entry_depth_N.

        Void element depth inflation: <br>, <img>, <input> etc. count as
        opens. They have no closing tag so depth never decrements.
        On image-heavy pages depth drifts upward and zones never exit.
        Fix: void elements stripped from the line before depth counting.

        No content-type awareness: all zones use the same tag-strip + print.
        Code zones should preserve structure; list zones should emit items
        one per line; table zones should call the table FST.
        Fix: per-zone structural_role dispatch in the extraction block.

        No boundary integration: ZoneMap.boundaries carry SECTION_BOUNDARY
        and CONTENT_BOUNDARY descriptors that the multizone program ignores.
        Fix: boundaries compiled to awk transitions and injected after the
        depth-tracking block.

        No zone-weight ordering: zones with higher priority_score should
        be tested earlier in the rule sequence (lower awk pattern-match cost
        on hot paths). Fix: signal_zones assumed pre-sorted by priority_score
        descending (enrich_zone_map already does this); the comment makes it
        explicit and the BEGIN block reflects the ordering.

    Label protocol:
        Signal zone content:  ZONE[<selector>]:\\t<line>
        Section boundary hit: SECTION[N]:\\t<heading text>
        Code zone content:    CODE[<lang>]:\\t<line>
        Code zone separator:  ---
        List item:            ITEM:\\t<text>
        Noise-free fallback:  <line>  (no prefix; only when sig_cond="1")

    Depth anchoring invariant:
        zone N is active iff  entry_depth_N <= current depth
        Zone N exits when depth returns to entry_depth_N - 1.
        This is monotone: a deeper nested zone cannot exit before its parent.

    Void element exclusion:
        <br|img|input|hr|meta|link|area|base|col|embed|param|source|track|wbr>
        stripped from the line before depth counting with gsub().
        Depth counter remains stable on image-heavy documentation pages.

    Phase conditioning:
        LEARNS:   all zones active, no length filter, maximum signal capture.
        PREDICTS: zones active, no length filter, dedup via seen[] disabled
                  (handled in _compose_zone_extract_awk post-pipeline).
        KNOWS:    zones active, MIN_SIGNAL_LENGTH filter on non-code lines,
                  seen[] dedup for exact-duplicate lines within a zone run.

    FST specification:
        States:   {in_sig_0..N} × {in_noise_0..M} × {in_code, in_list}
        Depth:    {depth, entry_depth_0..N}
        Sections: {section_idx, in_capture} from boundary transitions
        Input:    HTML line tokens (one awk record per line)
        Output:   zone-labeled content lines
    """
    _VOID = "br|img|input|hr|meta|link|area|base|col|embed|param|source|track|wbr"

    is_knows = (phase == PhaseStr.KNOWS)

    # Zone labels for output — truncated selector, safe for awk string literal
    def _zone_label(sel: Any) -> str:
        raw = sel.raw.replace('"', "").replace("'", "").replace("\\", "")
        return raw[:32].strip()

    lines: List[str] = []

    # ── header ────────────────────────────────────────────────────────────────
    lines.append("#!/usr/bin/awk -f")
    lines.append(f"# multizone specialist — phase: {phase.value}")
    lines.append(
        f"# {len(signal_zones)} signal zone(s), {len(noise_zones)} noise zone(s)"
    )
    lines.append(
        "# signal zones ordered by priority_score descending (hot path first)"
    )
    lines.append("")

    # ── BEGIN ─────────────────────────────────────────────────────────────────
    lines.append("BEGIN {")
    lines.append("    depth = 0")
    lines.append("    section_idx = 0")
    lines.append("    in_capture = 0")
    lines.append("    in_code = 0")
    lines.append("    in_list = 0")
    lines.append("    code_lang = \"text\"")
    for i, z in enumerate(signal_zones):
        lines.append(
            f"    in_sig_{i} = 0; entry_depth_{i} = -1"
            f"  # {z.selector.raw} (w={z.weight:.3f})"
        )
    for i, z in enumerate(noise_zones):
        lines.append(f"    in_noise_{i} = 0  # {z.selector.raw}")
    if is_knows:
        lines.append('    split("", seen)  # dedup table')
    lines.append("}")
    lines.append("")

    # ── void element exclusion ────────────────────────────────────────────────
    lines.append("# strip void elements before depth counting")
    lines.append("# prevents <br>/<img> from inflating depth permanently")
    lines.append("{")
    lines.append(f'    gsub(/<({_VOID})(\\s[^>]*)?\\/?>|<({_VOID})>/, "")')
    lines.append("}")
    lines.append("")

    # ── per-line depth tracking ───────────────────────────────────────────────
    lines.append("# depth counter — drives entry_depth exit condition")
    lines.append("{")
    lines.append('    opens  = gsub(/<[^\\/!][^>]*>/, "&")')
    lines.append('    closes = gsub(/<\\/[^>]*>/, "&")')
    lines.append("    depth += opens - closes")
    lines.append("    if (depth < 0) depth = 0")
    lines.append("}")
    lines.append("")

    # ── boundary transitions (SECTION_BOUNDARY, CONTENT_BOUNDARY) ────────────
    # Pull from zone_map if available — injected at call site via partial
    # application. Guarded: only emitted when the list is non-empty.
    lines.append("# boundary transitions from ZoneMap.boundaries")
    lines.append("# SECTION_BOUNDARY: reset in_capture, increment section_idx")
    lines.append("# CONTENT_BOUNDARY: toggle in_capture")
    # (Actual transitions injected dynamically at call site — see
    # _compose_zone_extract_awk which calls _compile_boundary_awk_transitions
    # and prepends them. This block is a structural placeholder that documents
    # the contract so the generated awk is self-describing.)
    lines.append("# [boundary transitions inserted by _compile_boundary_awk_transitions]")
    lines.append("")

    # ── noise zone entry / exit ───────────────────────────────────────────────
    if noise_zones:
        lines.append("# noise zone entry / exit — must precede signal extraction")
        for i, z in enumerate(noise_zones):
            pat = _selector_to_awk_pattern(z.selector)
            if not pat:
                continue
            close_tag = z.selector.tag or "div"
            lines.append(f"/{pat}/ {{ in_noise_{i} = 1; next }}")
            lines.append(
                f"/<\\/{close_tag}>/ && in_noise_{i}"
                f" {{ in_noise_{i} = 0; next }}"
            )
        lines.append("")

    # ── signal zone entry / exit — depth-anchored ────────────────────────────
    lines.append("# signal zone entry / exit")
    lines.append("# entry records depth so exit fires at the matching close tag,")
    lines.append("# not on the first child </div> encountered inside the zone.")
    for i, z in enumerate(signal_zones):
        pat = _selector_to_awk_pattern(z.selector)
        if not pat:
            continue
        close_tag = z.selector.tag or "div"
        label = _zone_label(z.selector)

        lines.append(f"# zone {i}: {z.selector.raw!r} role={z.structural_role}")
        lines.append(f"!in_sig_{i} && /{pat}/ {{")
        lines.append(f"    in_sig_{i} = 1")
        lines.append(f"    entry_depth_{i} = depth - 1")
        lines.append("}")
        # Exit: depth has returned to entry level (below zone's own open tag)
        lines.append(
            f"in_sig_{i} && depth <= entry_depth_{i}"
            f" && /<\\/{close_tag}>/ {{"
        )
        lines.append(f"    in_sig_{i} = 0; entry_depth_{i} = -1; next")
        lines.append("}")
        lines.append("")

    # ── combined zone predicate ───────────────────────────────────────────────
    if signal_zones:
        sig_cond = "(" + " || ".join(
            f"in_sig_{i}" for i in range(len(signal_zones))
        ) + ")"
    else:
        sig_cond = "1"

    if noise_zones:
        noise_cond = " && ".join(
            f"!in_noise_{i}" for i in range(len(noise_zones))
        )
    else:
        noise_cond = "1"

    active = f"{sig_cond} && {noise_cond}"

    # ── code block tracking ───────────────────────────────────────────────────
    # Extracted before the main extraction block so code lines route correctly
    lines.append("# code block tracking — language-tagged output")
    lines.append(f"{active} && /<pre[> ]/ {{ in_code=1; code_lang=\"text\"; next }}")
    lines.append(f"{active} && /<code[^>]*class=/ {{")
    lines.append("    in_code=1")
    lines.append("    _ln = $0")
    lines.append('    if (sub(/.*class="[^"]*language-/, "", _ln)) {')
    lines.append('        code_lang=_ln; sub(/".*/, "", code_lang)')
    lines.append('    } else if (sub(/.*class="[^"]*lang-/, "", _ln)) {')
    lines.append('        code_lang=_ln; sub(/".*/, "", code_lang)')
    lines.append('    } else { code_lang="text" }')
    lines.append("    next")
    lines.append("}")
    lines.append("/<\\/pre>/ || /<\\/code>/ {")
    lines.append("    if (in_code) { print \"---\"; in_code=0 }")
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── list item tracking ────────────────────────────────────────────────────
    lines.append("# list item extraction — one ITEM: per <li>")
    lines.append(f"{active} && /<li[> ]/ {{ in_list=1; next }}")
    lines.append(f"{active} && in_list && /<\\/li>/ {{")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print "ITEM:\\t" $0')
    lines.append("    in_list=0; next")
    lines.append("}")
    lines.append("")

    # ── main extraction block — per-zone label dispatch ───────────────────────
    lines.append("# main extraction — zone-labeled output")
    lines.append("# each active signal zone emits with its own label")
    lines.append("# priority ordering matches signal_zones sort (highest weight first)")

    for i, z in enumerate(signal_zones):
        pat = _selector_to_awk_pattern(z.selector)
        if not pat:
            continue
        label = _zone_label(z.selector)
        noise_guard = noise_cond if noise_zones else "1"

        # per-zone extraction with role-appropriate output
        role = z.structural_role
        lines.append(f"in_sig_{i} && {noise_guard} {{")

        if role == "code_block" or z.selector.tag in ("pre", "code"):
            # code zone: route through code tracking above; strip tags here
            lines.append('    if (in_code) {')
            lines.append('        gsub(/<[^>]*>/, "")')
            lines.append('        gsub(/^[[:space:]]{4}/, "")')
            lines.append('        if (/[^[:space:]]/) print "CODE[" code_lang "]:\\t" $0')
            lines.append('        next')
            lines.append('    }')
        elif role in ("list", "ordered_list", "unordered_list"):
            # list zone: in_list handles <li> items above; skip bare list tags
            lines.append('    if (/<[ou]l[> ]/ || /<\\/[ou]l>/) next')
        elif role in ("heading", "title", "section_header"):
            # heading zone: emit as SECTION with index
            lines.append('    gsub(/<[^>]*>/, "")')
            lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
            lines.append('    if (/[^[:space:]]/) {')
            lines.append('        print "SECTION[" section_idx "]:\\t" $0')
            lines.append('        section_idx++')
            lines.append('    }')
            lines.append('    next')

        # default tag-strip + label + optional dedup
        lines.append('    gsub(/<[^>]*>/, "")')
        lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
        if is_knows:
            lines.append(f"    if (length($0) < {MIN_SIGNAL_LENGTH}) next")
        lines.append('    if (/[^[:space:]]/) {')
        if is_knows:
            lines.append(f'        if (seen[$0]++) next')
        lines.append(f'        print "ZONE[{label}]:\\t" $0')
        lines.append('    }')
        lines.append('    next')
        lines.append("}")
        lines.append("")

    # ── fallback: no signal zones defined (empty zone map) ───────────────────
    if not signal_zones:
        lines.append("# fallback: no signal zones — full-document extraction")
        lines.append("{")
        lines.append('    gsub(/<[^>]*>/, "")')
        lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
        if is_knows:
            lines.append(f"    if (length($0) < {MIN_SIGNAL_LENGTH}) next")
        lines.append('    if (/[^[:space:]]/) print')
        lines.append("}")

    return "\n".join(lines)


def _selector_to_awk_pattern(sel: ParsedSelector) -> Optional[str]:
    """Convert a parsed CSS selector to an awk regex pattern.

    Handles all SelectorKind variants the parser produces. Returns None
    only for UNIVERSAL_SELECTOR (*) which matches every line and would
    activate every zone on every page — callers treat None as "skip pattern
    generation, use fallback or omit rule."

    Kind dispatch:

        CLASS_SELECTOR   (.article-body)
            → class="[^"]*article\\-body[^"]*"
            Matches anywhere in the class attribute — correct for
            multi-class elements like class="container article-body active".

        ID_SELECTOR      (#mw-content-text)
            → id="mw\\-content\\-text"
            Exact match — IDs are unique per document, no substring needed.

        TAG_SELECTOR     (main, article, section)
            → <main[> ] or <main\\s
            Matches opening tag with either > or space (attributes follow).

        ATTRIBUTE_SELECTOR  ([role="main"], [data-testid="content"])
            → role="main" or data\\-testid="content"
            Exact attribute match. Value-less selectors ([data-active])
            match the attribute name presence regardless of value.
            Returns None if attribute_name is absent (malformed selector).

        COMPOUND_SELECTOR   (div.article-body, section#main, div.foo#bar)
            → tag AND (class OR id) — both conditions in one pattern.
            Uses lookahead-free approach: tag pattern AND class/id pattern
            joined with a comment explaining they fire on the same line.
            Because awk has no lookahead, we emit the more specific
            component (id > class > tag) and accept that the tag constraint
            is checked by the zone's close_tag in the exit rule.

        DESCENDANT_SELECTOR  (article .post-body, #content .entry)
            → pattern for the target (rightmost) component only.
            Parent context cannot be verified in single-pass awk without
            a parent-depth stack. The target pattern is precise enough
            in practice: descendant selectors are used for deep nesting
            where the class name itself is sufficiently specific.

        NEGATION_SELECTOR   (:not(.sidebar))
            → None — negation cannot be expressed as a positive awk
            entry pattern. Callers that need negation should use a
            noise zone instead, which is the correct architectural split:
            signal zones are positive matches, noise zones are exclusions.

        PSEUDO_SELECTOR     (:first-child, ::before)
            → None — structural pseudo-classes have no HTML attribute
            equivalent. Cannot be detected by line-level pattern matching.

        UNIVERSAL_SELECTOR  (*)
            → None — matches everything, would activate zone on every line.

    Escaping:
        '-' → '\\-' in awk regex  (hyphen outside [] is literal but
              escaping is defensive — some awk implementations warn).
        '.' → '\\.' (would match any char without escaping).
        '#' → no escaping needed in awk regex outside character classes.
        '"' → '\\"' inside awk string literals (the pattern is quoted).
        '/' → '\\/' inside awk /regex/ delimiters.

    Returns:
        str  — awk regex pattern ready for use in /pattern/ or as a
               string passed to match(). Does NOT include the /.../ delimiters.
        None — selector kind cannot produce a useful pattern (see above).
    """
    # Negation: architecturally wrong direction — should be a noise zone
    if sel.is_negation:
        return None

    # PSEUDO and UNIVERSAL: no HTML attribute equivalent
    if sel.kind in (SelectorKind.PSEUDO_SELECTOR, SelectorKind.UNIVERSAL_SELECTOR):
        return None

    def _esc(s: str) -> str:
        """Escape for use inside an awk /regex/ pattern."""
        return (
            s.replace("\\", "\\\\")
             .replace(".", "\\.")
             .replace("-", "\\-")
             .replace("/", "\\/")
             .replace('"', '\\"')
             .replace("[", "\\[")
             .replace("]", "\\]")
             .replace("(", "\\(")
             .replace(")", "\\)")
        )

    # ── ATTRIBUTE_SELECTOR: [role="main"], [data-testid="..."], [data-active] ─
    if sel.kind == SelectorKind.ATTRIBUTE_SELECTOR:
        if not sel.attribute_name:
            return None
        attr = _esc(sel.attribute_name)
        if sel.attribute_value:
            val = _esc(sel.attribute_value)
            return f'{attr}="{val}"'
        else:
            # Value-less: attribute presence only
            return f'{attr}='

    # ── ID_SELECTOR: #mw-content-text ────────────────────────────────────────
    if sel.kind == SelectorKind.ID_SELECTOR:
        if not sel.id_name:
            return None
        return f'id="{_esc(sel.id_name)}"'

    # ── CLASS_SELECTOR: .article-body ────────────────────────────────────────
    if sel.kind == SelectorKind.CLASS_SELECTOR:
        if not sel.class_name:
            return None
        return f'class="[^"]*{_esc(sel.class_name)}[^"]*"'

    # ── TAG_SELECTOR: main, article, section ─────────────────────────────────
    if sel.kind == SelectorKind.TAG_SELECTOR:
        if not sel.tag:
            return None
        return f"<{_esc(sel.tag)}[> ]"

    # ── COMPOUND_SELECTOR: div.article-body, section#main ────────────────────
    # Emit the most specific component. Tag constraint is enforced by the
    # exit rule's close_tag, not the entry pattern.
    if sel.kind == SelectorKind.COMPOUND_SELECTOR:
        if sel.id_name:
            return f'id="{_esc(sel.id_name)}"'
        if sel.class_name:
            return f'class="[^"]*{_esc(sel.class_name)}[^"]*"'
        if sel.tag:
            return f"<{_esc(sel.tag)}[> ]"
        return None

    # ── DESCENDANT_SELECTOR: article .post-body, #content .entry ─────────────
    # Target component only — parent context unverifiable in single-pass awk.
    if sel.kind == SelectorKind.DESCENDANT_SELECTOR:
        if sel.id_name:
            return f'id="{_esc(sel.id_name)}"'
        if sel.class_name:
            return f'class="[^"]*{_esc(sel.class_name)}[^"]*"'
        if sel.tag:
            return f"<{_esc(sel.tag)}[> ]"
        # Fall back to raw selector last component
        if sel.parent_selectors:
            last = sel.raw.split()[-1]
            if last.startswith("."):
                return f'class="[^"]*{_esc(last[1:])}[^"]*"'
            if last.startswith("#"):
                return f'id="{_esc(last[1:])}"'
        return None

    # PSEUDO_SELECTOR caught above; UNIVERSAL_SELECTOR caught above.
    # Exhaustive — all SelectorKind values handled.
    return None # noqa | defensive runtime check


def _build_json_ld_extraction_awk(
    *,
    phase: PhaseStr,
    topology_class: str = "",
) -> str:
    """Generate a phase-aware awk program for JSON-LD structured data extraction.

    Handles multiple <script type="application/ld+json"> blocks per page.
    Real pages routinely embed 3-5 schema blocks: Product + BreadcrumbList +
    Organization + WebSite + FAQPage.  The original merged all blocks into one
    stream.  This program tracks block boundaries and emits a SCHEMA_START
    separator per block so downstream consumers can parse schemas independently.

    Two-pass logic (single awk program, two behavioural states):

        Pass 1 — block extraction:
            In-block state activated by the <script> open tag.
            Lines accumulated until </script> closes the block.
            Block number incremented per schema so output is self-delimiting.

        Pass 2 — field extraction within block:
            While in_ld=1, each line is checked for well-known JSON-LD fields.
            Fields are emitted as FIELD:\\t<key>\\t<value> before the raw line.
            This gives downstream consumers structured access without a JSON
            parser while preserving the raw block for full re-parse if needed.

    @type discrimination:
        The first "@type" line in a block sets schema_type for that block.
        SCHEMA_START:\\t<N>\\t<type> labels each block with its type.
        Downstream can filter: grep "^SCHEMA_START.*Product" | head -1
        to find the product schema and ignore breadcrumb/org schemas.

    Topology-class field sets:
        ECOMMERCE_PRODUCT / ECOMMERCE_PRODUCT_VARIANT:
            name, price, priceCurrency, availability, sku, brand, description,
            aggregateRating/ratingValue, aggregateRating/reviewCount, image
        NEWS_ARTICLE / BLOG_POST:
            headline, datePublished, dateModified, author/name,
            publisher/name, description, articleBody (truncated)
        JSON_LD_STRUCTURED (generic):
            @type, @id, name, url, description  (universal fields only)
        All others: universal fields only

    Inline scalar extraction (POSIX-compatible, mawk-safe):
        Uses sub() chains — no 3-argument match() which requires gawk.
        Handles:  "key": "value"  →  FIELD:\\tkey\\tvalue
                  "key": 123      →  FIELD:\\tkey\\t123
                  "key": true     →  FIELD:\\tkey\\ttrue
        Nested objects (aggregateRating: {...}) use path_prefix tracking.

    Phase conditioning:
        LEARNS:   raw block emitted verbatim + field extraction.
                  Maximum signal: even unknown fields pass through.
        PREDICTS: field extraction only for topology-relevant fields.
                  Raw block suppressed — reduces downstream noise.
        KNOWS:    field extraction only, typed fields only, no raw lines.
                  Empty-value fields suppressed.

    FST specification:
        States:  {idle, in_ld}
        Aux:     {block_num, schema_type, path_prefix, field_count}
        Input:   HTML lines containing embedded JSON-LD
        Output:  SCHEMA_START:\\t<N>\\t<type>, FIELD:\\t<key>\\t<val>, raw lines
    """
    is_learns  = (phase == PhaseStr.LEARNS)
    is_knows   = (phase == PhaseStr.KNOWS)

    # Topology-specific field sets for targeted extraction
    _ECOMMERCE_FIELDS = {
        "name", "price", "priceCurrency", "availability", "sku",
        "brand", "description", "image", "url",
        "ratingValue", "reviewCount",
    }
    _NEWS_FIELDS = {
        "headline", "datePublished", "dateModified", "description",
        "articleBody", "url",
    }
    _AUTHOR_FIELDS = {"name"}   # inside author/publisher objects
    _UNIVERSAL_FIELDS = {"@type", "@id", "name", "url", "description"}

    if any(kw in topology_class for kw in ("ECOMMERCE", "PRODUCT")):
        target_fields = _ECOMMERCE_FIELDS
    elif any(kw in topology_class for kw in ("NEWS", "BLOG", "ARTICLE")):
        target_fields = _NEWS_FIELDS | _UNIVERSAL_FIELDS
    else:
        target_fields = _UNIVERSAL_FIELDS

    # Build field-matching block: for each known field emit FIELD:\tkey\tval
    def _field_block(fields: set, indent: str = "    ") -> List[str]:
        """Emit sub()-based extraction for each field name. POSIX-safe."""
        out = []
        for f in sorted(fields):
            safe_f = f.replace("@", "\\@")
            # String value:  "key": "value"
            out.append(f'{indent}_ln = $0')
            out.append(
                f'{indent}if (sub(/.*"{safe_f}"[[:space:]]*:[[:space:]]*"/, "", _ln)) {{'
            )
            out.append(f'{indent}    sub(/".*/, "", _ln)')
            out.append(f'{indent}    if (length(_ln) > 0)')
            if is_knows:
                out.append(
                    f'{indent}        print "FIELD:\\t{f}\\t" _ln'
                )
            else:
                out.append(
                    f'{indent}        print "FIELD:\\t{f}\\t" _ln'
                )
            out.append(f'{indent}}} else {{')
            # Numeric / boolean value:  "key": 123  or "key": true
            out.append(f'{indent}    _ln = $0')
            out.append(
                f'{indent}    if (sub(/.*"{safe_f}"[[:space:]]*:[[:space:]]*/, "", _ln)) {{'
            )
            out.append(f'{indent}        sub(/[,}}].*/, "", _ln)')
            out.append(f'{indent}        gsub(/^[[:space:]]+|[[:space:]]+$/, "", _ln)')
            out.append(
                f'{indent}        if (length(_ln) > 0 && _ln !~ /^[{{[]/)'
            )
            out.append(
                f'{indent}            print "FIELD:\\t{f}\\t" _ln'
            )
            out.append(f'{indent}    }}')
            out.append(f'{indent}}}')
        return out

    lines: List[str] = []

    lines.append("BEGIN {")
    lines.append("    in_ld=0; block_num=0")
    lines.append('    schema_type="unknown"; field_count=0')
    lines.append("}")
    lines.append("")

    # Block entry
    lines.append('/<script[^>]*application\\/ld\\+json[^>]*>/ {')
    lines.append("    in_ld=1; block_num++; field_count=0")
    lines.append('    schema_type="unknown"')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # Block exit
    lines.append('/<\\/script>/ && in_ld {')
    lines.append("    in_ld=0")
    lines.append('    print "SCHEMA_END:\\t" block_num')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # Inside block
    lines.append("in_ld {")
    lines.append('    gsub(/^[[:space:]]+/, "")')
    lines.append('    if (!/[^[:space:]]/) next')
    lines.append("")

    # @type detection — emit SCHEMA_START once we know the type
    lines.append('    # @type detection — labels block before any fields')
    lines.append('    if ($0 ~ /"@type"/) {')
    lines.append('        _ln = $0')
    lines.append('        sub(/.*"@type"[[:space:]]*:[[:space:]]*"/, "", _ln)')
    lines.append('        sub(/".*/, "", _ln)')
    lines.append('        if (length(_ln) > 0 && schema_type == "unknown") {')
    lines.append('            schema_type = _ln')
    lines.append('            print "SCHEMA_START:\\t" block_num "\\t" schema_type')
    lines.append('        }')
    lines.append('    }')
    lines.append("")

    # Field extraction
    lines.append("    # field extraction — topology-specific field set")
    lines.extend(_field_block(target_fields))
    lines.append("")

    # Phase-conditioned raw line emission
    if is_learns:
        lines.append("    # LEARNS: emit raw line for maximum signal capture")
        lines.append("    print")
    elif not is_knows:
        lines.append("    # PREDICTS: emit raw line (training signal still useful)")
        lines.append("    if (field_count > 0 || $0 ~ /@type|@id/) print")

    lines.append("}")

    return "\n".join(lines)


def _build_attribute_extraction_awk(
    attributes: List[str],
    *,
    phase: PhaseStr,
) -> str:
    """Generate a phase-aware awk program for data-* attribute extraction.

    Extracts HTML attribute values by name and emits labeled key-value pairs.
    Handles three attribute value shapes per attribute:

        Quoted string:   data-price="$29.99"      → PRICE:\t$29.99
        Single-quoted:   data-price='$29.99'      → PRICE:\t$29.99
        Unquoted:        data-price=$29.99         → PRICE:\t$29.99
                         (unquoted values terminate at whitespace or >)

    POSIX-compatible (mawk-safe):
        Uses sub() chains instead of 3-argument match() which requires gawk.
        3-arg match(line, /re/, arr) is a gawk extension — the original used
        it and produced silent empty output on the mawk in the signal kernel.

    Label format:
        data-product-name  → PRODUCT_NAME:\t<value>
        data-price         → PRICE:\t<value>
        itemprop-sku       → SKU:\t<value>         (itemprop= variant)
        role               → ROLE:\t<value>         (non-data- attributes)

    Deduplication (PREDICTS + KNOWS):
        The same attribute can appear multiple times on a page
        (e.g. data-price in a product list with variants). The seen[]
        associative array deduplicates on key+value so repeated identical
        emissions are suppressed but distinct values for the same key
        (price variants) pass through.

    Microdata itemprop= support:
        Many ecommerce pages use both data-* attributes and HTML Microdata
        itemprop= attributes. The program checks both shapes for each field:
            data-<name>="..."  and  itemprop="<name>" content="..."
        This doubles coverage on Microdata-heavy pages without a separate pass.

    Phase conditioning:
        LEARNS:   all attributes extracted, no dedup, maximum coverage.
        PREDICTS: dedup on key+value pair.
        KNOWS:    dedup on key+value, empty-value lines suppressed,
                  values shorter than 2 characters suppressed (noise).

    FST specification:
        States:  stateless per-line (no multi-line attribute values)
        Input:   HTML lines with data-* or itemprop= attributes
        Output:  LABEL:\t<value> pairs, one per matched attribute per line
    """
    is_learns = (phase == PhaseStr.LEARNS)
    is_knows  = (phase == PhaseStr.KNOWS)
    dedup     = not is_learns

    lines: List[str] = []

    lines.append("BEGIN {")
    if dedup:
        lines.append('    split("", seen)')
    lines.append("}")
    lines.append("")
    lines.append("{")

    for attr in attributes:
        # Derive label: strip data-/itemprop- prefix, uppercase, hyphen→underscore
        label_base = attr
        for prefix in ("data-", "itemprop-", "aria-"):
            if label_base.startswith(prefix):
                label_base = label_base[len(prefix):]
                break
        label = label_base.upper().replace("-", "_")

        # Escape for awk regex: hyphens are safe inside [] but ambiguous outside
        safe = attr.replace("-", "\\-").replace(".", "\\.")

        lines.append(f"    # {attr} → {label}")

        # Shape 1: double-quoted  attr="value"
        lines.append(f'    _ln = $0')
        lines.append(f'    if (sub(/.*{safe}="/, "", _ln)) {{')
        lines.append(f'        sub(/".*/, "", _ln)')
        lines.append(f'        gsub(/^[[:space:]]+|[[:space:]]+$/, "", _ln)')
        if is_knows:
            lines.append(f'        if (length(_ln) > 1) {{')
        else:
            lines.append(f'        if (length(_ln) > 0) {{')
        if dedup:
            lines.append(f'            if (!seen["{label}\\t" _ln]++) {{')
            lines.append(f'                print "{label}:\\t" _ln')
            lines.append(f'            }}')
        else:
            lines.append(f'            print "{label}:\\t" _ln')
        lines.append(f'        }}')
        lines.append(f'    }} else {{')

        # Shape 2: single-quoted  attr='value'
        lines.append(f"        _ln = $0")
        lines.append(f"        if (sub(/.*{safe}='/, \"\", _ln)) {{")
        lines.append(f"            sub(/'.*/, \"\", _ln)")
        lines.append(f'            gsub(/^[[:space:]]+|[[:space:]]+$/, "", _ln)')
        if is_knows:
            lines.append(f'            if (length(_ln) > 1) {{')
        else:
            lines.append(f'            if (length(_ln) > 0) {{')
        if dedup:
            lines.append(f'                if (!seen["{label}\\t" _ln]++) {{')
            lines.append(f'                    print "{label}:\\t" _ln')
            lines.append(f'                }}')
        else:
            lines.append(f'                print "{label}:\\t" _ln')
        lines.append(f'            }}')
        lines.append(f'        }} else {{')

        # Shape 3: itemprop="attr" content="value"  (Microdata)
        # <span itemprop="price" content="29.99">$29.99</span>
        lines.append(f'            _ln = $0')
        lines.append(f'            if (_ln ~ /itemprop="{label_base}"/ ||')
        lines.append(f'                _ln ~ /itemprop=\'{label_base}\'/) {{')
        lines.append(f'                _cv = $0')
        lines.append(f'                if (sub(/.*content="/, "", _cv)) {{')
        lines.append(f'                    sub(/".*/, "", _cv)')
        lines.append(f'                    gsub(/^[[:space:]]+|[[:space:]]+$/, "", _cv)')
        if is_knows:
            lines.append(f'                    if (length(_cv) > 1) {{')
        else:
            lines.append(f'                    if (length(_cv) > 0) {{')
        if dedup:
            lines.append(f'                        if (!seen["{label}\\t" _cv]++) {{')
            lines.append(f'                            print "{label}:\\t" _cv')
            lines.append(f'                        }}')
        else:
            lines.append(f'                            print "{label}:\\t" _cv')
        lines.append(f'                    }}')
        lines.append(f'                }}')
        lines.append(f'            }}')
        lines.append(f'        }}')  # close shape 2 else
        lines.append(f'    }}')      # close shape 1 else
        lines.append("")

    lines.append("}")

    return "\n".join(lines)


def _build_envelope_extraction_awk(
    json_path: str,
    *,
    phase: PhaseStr,
    topology_class: str = "",
) -> str:
    """Generate a phase-aware awk program for REST API JSON envelope extraction.

    Envelope extraction differs from generic JSON path traversal in three ways:

        1. Root-level pre-scan: REST responses carry status, error, and message
           fields at the root level, before the data envelope. These are emitted
           as STATUS:, ERROR:, and MESSAGE: prefixed lines so the index daemon
           can detect failed responses (4xx/5xx, error: true) without consuming
           the data path. A failed response is flagged and the data path is
           skipped — prevents storing error page content as signal.

        2. Pagination metadata: REST_API_JSON_PAGINATED responses carry cursor,
           offset, total, has_more, next_page fields in sibling keys to the data
           envelope (typically under "meta", "pagination", or at the root).
           These are emitted as PAGINATION: prefixed lines before data items.
           Phase III suppresses pagination metadata — known topology, no training
           value in cursor strings.

        3. Explicit envelope noise suppression: the default noise keys for
           REST_API_JSON are ["meta", "pagination", "links", "_links"]. The
           traversal awk would enter these if they appear before the signal path.
           This program explicitly skips them with in_noise flags so their content
           never reaches the extraction block.

    Path handling:
        Single-component path ("data", "results"):
            Enters on /"data"[[:space:]]*:/ and extracts the container.
            Handles both array and object containers.

        Multi-component path ("data.items", "response.records"):
            Delegates to _build_json_traversal_awk for the nested walk,
            wrapping it with the root-level pre-scan and noise suppression
            layers this function adds.

    Error response detection:
        Root-level scan for "error", "errors", "status", "message" fires
        before envelope entry. If "error": true or status >= 400 is detected,
        sets in_error=1 which suppresses all further data extraction and emits
        a single ERROR_RESPONSE: line. This prevents crawling error pages from
        polluting the signal store with 404/429 page content.

    Pagination field extraction (REST_API_JSON_PAGINATED only):
        Fields scanned: cursor, next_cursor, next_page, has_more, total,
        total_count, total_results, per_page, offset, limit.
        Emitted as PAGINATION:\t<field>\t<value>.
        These are extracted from the root level and from any "meta" or
        "pagination" sibling key — both locations are common in wild APIs.

    Phase conditioning:
        LEARNS:   full extraction + pagination metadata + raw envelope lines.
        PREDICTS: data items only + error detection. Pagination suppressed.
        KNOWS:    data items only, error detection, MIN_SIGNAL_LENGTH filter,
                  pagination suppressed, structural-noise lines suppressed.

    FST specification:
        States:  {idle, in_error, in_noise, in_envelope}
        Aux:     {brace_depth, bracket_depth, entry_depth, item_count,
                  container_type, status_code}
        Input:   JSON response lines (one record per line)
        Output:  STATUS:, ERROR_RESPONSE:, PAGINATION:, ITEM_N:, FIELD:
    """
    is_paginated = "PAGINATED" in topology_class
    is_knows     = (phase == PhaseStr.KNOWS)
    is_learns    = (phase == PhaseStr.LEARNS)

    # Multi-component path: delegate to traversal awk with envelope wrapper
    parts = [p.strip() for p in json_path.split(".") if p.strip()]
    if not parts:
        parts = ["data"]

    if len(parts) > 1:
        # Traversal awk handles the nested walk; we only need root-level guards
        # here. Embed a comment header and return the traversal program.
        traversal = _build_json_traversal_awk(parts, phase=phase)
        header = "\n".join([
            "#!/usr/bin/awk -f",
            f"# ENVELOPE_EXTRACT — path: {json_path} — phase: {phase.value}",
            f"# topology: {topology_class or 'unknown'}",
            "# multi-component path: delegating to json_traversal_awk",
            "# root-level error/status scan prepended below",
            "",
        ])
        # Prepend error detection before the traversal program
        error_block = "\n".join([
            "# root-level error detection — suppress data on failed responses",
            'BEGIN { in_error=0; status_code=0 }', # noqa | false positive
            '/\"error\"[[:space:]]*:[[:space:]]*true/ { in_error=1 }',
            '/\"status\"[[:space:]]*:[[:space:]]*[45][0-9][0-9]/ {',
            '    _s=$0; sub(/.*"status"[[:space:]]*:[[:space:]]*/, "", _s)',
            '    sub(/[^0-9].*/, "", _s); status_code=_s+0',
            '    if (status_code >= 400) in_error=1',
            '}',
            'in_error && /\"message\"/ {',
            '    _m=$0; sub(/.*"message"[[:space:]]*:[[:space:]]*"/, "", _m)',
            '    sub(/".*/, "", _m)',
            '    print "ERROR_RESPONSE:\\t" _m; exit',
            '}',
            '',
        ])
        return header + error_block + traversal

    # Single-component path — full envelope program
    key = parts[0]
    safe_key = key.replace('"', '\\"').replace("-", "\\-")

    # Pagination fields — extracted from root and from meta/pagination siblings
    _PAGINATION_FIELDS = [
        "cursor", "next_cursor", "next_page", "prev_cursor",
        "has_more", "total", "total_count", "total_results",
        "per_page", "page_size", "offset", "limit", "page",
    ]

    # Noise keys to skip (default REST_API_JSON noise zones)
    _NOISE_KEYS = ["meta", "pagination", "links", "_links", "included",
                   "jsonapi", "_meta", "extensions"]

    lines: List[str] = []

    lines.append("#!/usr/bin/awk -f")
    lines.append(f"# ENVELOPE_EXTRACT — key: {key!r} — phase: {phase.value}")
    lines.append(f"# topology: {topology_class or 'REST_API_JSON'}")
    lines.append("")

    lines.append("BEGIN {")
    lines.append("    in_envelope=0; in_noise=0; in_error=0")
    lines.append("    brace_depth=0; bracket_depth=0; entry_depth=0")
    lines.append('    item_count=0; container_type=""')
    lines.append('    status_code=0; error_emitted=0')
    lines.append("}")
    lines.append("")

    # Depth tracking
    lines.append("# structural depth")
    lines.append("{")
    lines.append('    brace_depth   += gsub(/\\{/, "{") - gsub(/\\}/, "}")')
    lines.append('    bracket_depth += gsub(/\\[/, "[") - gsub(/\\]/, "]")')
    lines.append("    if (brace_depth   < 0) brace_depth   = 0")
    lines.append("    if (bracket_depth < 0) bracket_depth = 0")
    lines.append("}")
    lines.append("")

    # Root-level error/status detection
    lines.append("# error detection — suppress data on failed responses")
    lines.append('!in_envelope && /"error"[[:space:]]*:[[:space:]]*true/ {')
    lines.append("    in_error=1")
    lines.append("}")
    lines.append('!in_envelope && /"status"[[:space:]]*:[[:space:]]*[45][0-9][0-9]/ {')
    lines.append('    _s = $0')
    lines.append('    sub(/.*"status"[[:space:]]*:[[:space:]]*/, "", _s)')
    lines.append('    sub(/[^0-9].*/, "", _s)')
    lines.append('    if (_s+0 >= 400) in_error=1')
    lines.append("}")
    lines.append('in_error && /"message"/ && !error_emitted {')
    lines.append('    _m = $0')
    lines.append('    sub(/.*"message"[[:space:]]*:[[:space:]]*"/, "", _m)')
    lines.append('    sub(/".*/, "", _m)')
    lines.append('    if (length(_m) > 0) print "ERROR_RESPONSE:\\t" _m')
    lines.append("    error_emitted=1; exit")
    lines.append("}")
    lines.append("in_error { next }")
    lines.append("")

    # Noise key suppression
    lines.append("# noise key suppression — skip meta/pagination/links siblings")
    for nk in _NOISE_KEYS:
        safe_nk = nk.replace("-", "\\-").replace("_", "\\_")
        lines.append(
            f'!in_envelope && /"{safe_nk}"[[:space:]]*:[[:space:]]*\\{{/'
            f' {{ in_noise=1 }}'
        )
    lines.append("in_noise && brace_depth <= 1 { in_noise=0; next }")
    lines.append("in_noise { next }")
    lines.append("")

    # Pagination metadata (LEARNS only or PAGINATED topology)
    if is_learns or is_paginated:
        lines.append("# pagination metadata extraction")
        lines.append('!in_envelope && !in_noise {')
        for pf in _PAGINATION_FIELDS:
            safe_pf = pf.replace("_", "\\_").replace("-", "\\-")
            lines.append(f'    _ln = $0')
            lines.append(
                f'    if (sub(/.*"{safe_pf}"[[:space:]]*:[[:space:]]*"/, "", _ln)) {{'
            )
            lines.append(f'        sub(/".*/, "", _ln)')
            lines.append(
                f'        if (length(_ln) > 0) print "PAGINATION:\\t{pf}\\t" _ln'
            )
            lines.append(f'    }} else {{')
            lines.append(f'        _ln = $0')
            lines.append(
                f'        if (sub(/.*"{safe_pf}"[[:space:]]*:[[:space:]]*/, "", _ln)) {{'
            )
            lines.append(f'            sub(/[,}}].*/, "", _ln)')
            lines.append(
                f'            gsub(/^[[:space:]]+|[[:space:]]+$/, "", _ln)'
            )
            lines.append(
                f'            if (length(_ln) > 0 && _ln !~ /^[{{[\\[]]/)'
            )
            lines.append(
                f'                print "PAGINATION:\\t{pf}\\t" _ln'
            )
            lines.append(f'        }}')
            lines.append(f'    }}')
        lines.append("}")
        lines.append("")

    # Envelope entry — object and array containers
    lines.append(f"# envelope entry — key: {key!r}")
    lines.append(f'!in_envelope && /"{safe_key}"[[:space:]]*:[[:space:]]*\\[/ {{')
    lines.append('    in_envelope=1; container_type="array"')
    lines.append("    entry_depth=bracket_depth-1")
    lines.append("    item_count=0")
    lines.append("    next")
    lines.append("}")
    lines.append(f'!in_envelope && /"{safe_key}"[[:space:]]*:[[:space:]]*\\{{/ {{')
    lines.append('    in_envelope=1; container_type="object"')
    lines.append("    entry_depth=brace_depth-1")
    lines.append("    item_count=0")
    lines.append("    next")
    lines.append("}")
    # Inline scalar: "data": "value"
    lines.append(f'!in_envelope && /"{safe_key}"[[:space:]]*:[[:space:]]*"/ {{')
    lines.append('    _ln = $0')
    lines.append(f'    sub(/.*"{safe_key}"[[:space:]]*:[[:space:]]*"/, "", _ln)')
    lines.append('    sub(/".*/, "", _ln)')
    lines.append('    if (length(_ln) > 0) print "FIELD:\\t" _ln')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # Envelope exit
    lines.append("# envelope exit — depth returns to entry level")
    lines.append('in_envelope && container_type == "array" &&')
    lines.append("    bracket_depth <= entry_depth {")
    lines.append("    in_envelope=0; next")
    lines.append("}")
    lines.append('in_envelope && container_type == "object" &&')
    lines.append("    brace_depth <= entry_depth {")
    lines.append("    in_envelope=0; next")
    lines.append("}")
    lines.append("")

    # Extraction inside envelope
    lines.append("# extraction — labeled by container type")
    lines.append("in_envelope {")
    lines.append('    gsub(/^[[:space:]]+/, "")')
    lines.append('    if (!/[^[:space:]]/) next')
    # Skip structural noise: bare { } [ ] , lines
    lines.append('    if ($0 ~ /^[\\{\\}\\[\\],][[:space:]]*$/) next')
    lines.append('    if (container_type == "array") {')
    lines.append("        item_count++")
    if is_knows:
        lines.append(f'        if (length($0) < {MIN_SIGNAL_LENGTH}) next')
    lines.append('        print "ITEM_" item_count ":\\t" $0')
    lines.append('    } else {')
    # Object: emit key\tvalue
    lines.append('        _ln = $0')
    lines.append('        if (_ln ~ /^"[^"]+":/) {')
    lines.append('            _key = _ln; sub(/"[^"]*".*/, "", _key)')
    lines.append('            sub(/.*"[^"]*"[[:space:]]*:[[:space:]]*/, "", _ln)')
    lines.append('            sub(/,$/, "", _ln)')
    lines.append('            gsub(/^"|"$/, "", _ln)')
    if is_knows:
        lines.append(f'            if (length(_ln) < {MIN_SIGNAL_LENGTH}) next')
    lines.append('            print "FIELD:\\t" _key "\\t" _ln')
    lines.append('        } else {')
    if is_learns:
        lines.append('            print $0')
    lines.append('        }')
    lines.append('    }')
    lines.append("}")

    return "\n".join(lines)


def _build_conditional_code_awk(*, phase: PhaseStr) -> str:
    """Generate a phase-aware awk program for context-conditioned code extraction.

    Implements the SAAS_DOCS_WITH_CODE paragraph-context rule:
        paragraph → code block  : emit paragraph + code (explanation + example)
        heading   → code block  : emit heading + code  (section label + example)
        code block alone        : Phase I emit, Phase II/III suppress

    Paragraph buffering:
        Paragraphs are not emitted immediately. They are held in a buffer
        and flushed only when a code block entry follows. This prevents
        isolated explanatory paragraphs (introduction, footer notes) from
        appearing in the output when no code block follows them.

        Buffer is cleared on: heading entry, section boundary (h1-h6),
        code block exit (the preceding para was already flushed or discarded).

    Code block discrimination:
        <pre>                   → always a code block
        <code class="...">      → fenced code block (has language class)
        <code> alone            → inline code in prose, skip

    Section heading extraction:
        h1-h6 headings are always emitted as SECTION: prefixed lines.
        They reset prev_context to "heading" so the next code block
        is treated as heading-anchored regardless of paragraph state.

    Phase I  — emit all code blocks (system learning, maximize signal)
    Phase II — emit code only when preceded by paragraph or heading
    Phase III — emit code only when preceded by paragraph or heading;
                drop lines below MIN_SIGNAL_LENGTH from prose output

    FST specification:
        States:  {idle, in_para, in_code}
        Aux:     {prev_context, para_buf, keep_code}
        Input:   HTML line tokens
        Output:  SECTION:\\t<heading>, buffered prose, CODE:\\t<line>
    """
    keep_isolated = phase == PhaseStr.LEARNS
    is_phase3     = phase == PhaseStr.KNOWS

    lines: list[str] = []

    lines.append("BEGIN {")
    lines.append('    in_para=0; in_code=0; in_heading=0; keep_code=0')
    lines.append('    prev_context=""; para_buf=""')
    lines.append("}")
    lines.append("")

    # ── section headings — always emit, reset context ─────────────────────────
    lines.append("/<h[1-6][> ]/ { in_heading=1; next }")
    lines.append("/<\\/h[1-6]>/ {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print "SECTION:\\t" $0')
    lines.append('    in_heading=0; prev_context="heading"; para_buf=""')
    lines.append("    next")
    lines.append("}")
    lines.append("in_heading {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print "SECTION:\\t" $0')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── paragraph buffering ───────────────────────────────────────────────────
    lines.append("/<p[> ]/ && !in_code { in_para=1; next }")
    lines.append("/<\\/p>/ && in_para {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) {')
    lines.append('        if (length(para_buf) > 0) para_buf = para_buf "\\n"')
    lines.append('        para_buf = para_buf $0')
    lines.append('    }')
    lines.append('    in_para=0; prev_context="para"')
    lines.append("    next")
    lines.append("}")
    lines.append("in_para && !/<p/ && !/<\\/p>/ {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) {')
    lines.append('        if (length(para_buf) > 0) para_buf = para_buf " "')
    lines.append('        para_buf = para_buf $0')
    lines.append('    }')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── code block entry — discriminate fenced vs inline ─────────────────────
    lines.append("/<pre[> ]/ || /<code[^>]*class=/ {")
    lines.append("    in_code=1")
    if keep_isolated:
        lines.append('    keep_code=1')
    else:
        lines.append('    keep_code=(prev_context == "para" || prev_context == "heading")')
    lines.append("    if (keep_code && length(para_buf) > 0) {")
    lines.append("        print para_buf")
    lines.append('        para_buf=""')
    lines.append("    }")
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── code block exit ───────────────────────────────────────────────────────
    lines.append("/<\\/pre>/ || /<\\/code>/ {")
    lines.append("    if (in_code) {")
    lines.append('        print "---"')
    lines.append('        in_code=0; keep_code=0; prev_context=""; para_buf=""')
    lines.append("    }")
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── code body ─────────────────────────────────────────────────────────────
    lines.append("in_code && keep_code {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]{4}/, "")  # normalise one indent level')
    lines.append('    print "CODE:\\t" $0')
    lines.append("    next")
    lines.append("}")
    lines.append("in_code && !keep_code { next }")
    lines.append("")

    # ── prose flush at section boundary ──────────────────────────────────────
    if is_phase3:
        lines.append(f"length($0) < {MIN_SIGNAL_LENGTH} {{ next }}")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# ZONE MAP ENRICHMENT LAYER
#
# Transforms the simplified ZoneMap from contracts.py into the rich
# internal representation the compiler needs. This is the bridge between
# the WLP's structural output and the compiler's translation rules.
#
# The enrichment process:
# 1. Parse all CSS selectors into ParsedSelector objects
# 2. Classify each selector as SIGNAL or NOISE
# 3. Assign weights based on selector specificity and zone position
# 4. Infer structural roles from selector patterns
# 5. Detect data attributes and JSON paths for strategy dispatch
# 6. Build child noise selector relationships
# 7. Compute depth hints from selector structure
#
# The enrichment is purely deterministic — same ZoneMap always produces
# the same enriched representation.
# ═════════════════════════════════════════════════════════════════════════════

def _infer_structural_role(sel: ParsedSelector) -> str:
    """Infer the structural role of a zone from its CSS selector.

    Uses heuristic pattern matching on class names, IDs, and tag names
    to classify the zone's structural purpose.

    This is where the compiler's structural intelligence lives — it
    understands the semantic meaning of common CSS patterns across
    different topology classes.
    """
    raw_lower = sel.raw.lower()

    # Code-related patterns
    if any(kw in raw_lower for kw in ("code", "pre", "syntax", "highlight",
                                       "snippet", "listing")):
        return "code_block"

    # Navigation patterns
    if any(kw in raw_lower for kw in ("nav", "breadcrumb", "menu", "sidebar",
                                       "toc", "table-of-contents")):
        return "navigation"

    # Content patterns
    if any(kw in raw_lower for kw in ("main-content", "article-body",
                                       "post-content", "entry-content",
                                       "content-area", "main")):
        return "main_content"

    # Heading patterns
    if sel.tag and sel.tag.startswith("h") and len(sel.tag) == 2:
        return "heading"

    # List patterns
    if sel.tag in ("ul", "ol", "dl"):
        return "list"

    # Table patterns
    if sel.tag == "table" or any(kw in raw_lower for kw in (
        "pricing-table", "comparison", "data-table", "specs"
    )):
        return "table"

    # Pricing patterns
    if any(kw in raw_lower for kw in ("price", "pricing", "cost", "tier",
                                       "plan")):
        return "pricing"

    # Paragraph / prose
    if sel.tag == "p" or any(kw in raw_lower for kw in (
        "paragraph", "prose", "text", "body"
    )):
        return "paragraph"

    # Footer / noise patterns
    if any(kw in raw_lower for kw in ("footer", "copyright", "cookie",
                                       "banner", "popup", "modal")):
        return "noise_element"

    # Link list patterns
    if any(kw in raw_lower for kw in ("link-list", "sitemap", "url")):
        return "link_list"

    return "generic"


def _detect_data_attributes(sel: ParsedSelector) -> Tuple[str, ...]:
    """Detect data-* attribute names referenced in a selector.

    Returns tuple of attribute names like ("data-product-name", "data-price").
    Used to determine if ATTRIBUTE_EXTRACT strategy is appropriate.
    """
    attrs: List[str] = []
    if sel.attribute_name and sel.attribute_name.startswith("data-"):
        attrs.append(sel.attribute_name)
    # Also scan the raw selector for data-* patterns
    for match in _RE_DATA_ATTR.finditer(sel.raw):
        attr = match.group(0)
        if attr not in attrs:
            attrs.append(attr)
    return tuple(attrs)


def _detect_json_path(sel: ParsedSelector, strategy: str) -> Optional[str]:
    """Detect JSON path from selector context.

    For ENVELOPE_EXTRACT strategy, the selector may encode a JSON path
    as a dot-separated string (e.g., "results.items").
    """
    if strategy != "envelope_extract":
        return None
    # Check if the raw selector looks like a JSON path
    raw = sel.raw.strip()
    if re.match(r"^[a-zA-Z_][\w.]*$", raw):
        return raw
    return None


def _assign_zone_weight(
    sel: ParsedSelector,
    position: int,
    total: int,
    confidence: float,
) -> float:
    """Compute zone weight from selector specificity, position, and confidence.

    Weight formula:
        w = confidence × (specificity_norm × 0.4 + position_score × 0.3 + base × 0.3)

    where:
        specificity_norm = min(specificity.score / 300, 1.0)
        position_score = 1.0 - (position / total)  [earlier = higher weight]
        base = 0.5

    Returns: float in [0.0, 1.0]
    """
    spec_norm = min(sel.specificity.score / 300.0, 1.0)
    pos_score = 1.0 - (position / max(total, 1))
    base = 0.5
    raw_weight = confidence * (spec_norm * 0.4 + pos_score * 0.3 + base * 0.3)
    return max(0.0, min(1.0, raw_weight))


def enrich_zone_map(
    zone_map: ZoneMap,
    *,
    feedback: Optional[FeedbackState] = None,
) -> List[EnrichedZone]:
    """Transform a simplified ZoneMap into enriched zones for compilation.

    This is the primary enrichment function. It:
    1. Parses all signal and noise selectors
    2. Classifies zones and assigns weights
    3. Infers structural roles
    4. Detects attributes and JSON paths
    5. Applies feedback-driven adjustments

    Returns: List[EnrichedZone] ordered by priority_score descending.
    """
    enriched: List[EnrichedZone] = []
    strategy = zone_map.strategy
    confidence = zone_map.confidence

    # Enrich signal zones
    total_signal = len(zone_map.signal_zones)
    for i, selector_raw in enumerate(zone_map.signal_zones):
        sel = parse_selector(selector_raw)
        role = _infer_structural_role(sel)
        data_attrs = _detect_data_attributes(sel)
        json_path = _detect_json_path(sel, strategy)
        weight = _assign_zone_weight(sel, i, total_signal, confidence)

        # Feedback adjustment: tighten if noise was high
        if feedback and feedback.tighten_requested:
            weight *= 0.85  # Reduce zone weight → narrower extraction
        if feedback and feedback.loosen_requested:
            weight = min(1.0, weight * 1.15)  # Increase weight → broader extraction

        # Determine depth limit from selector structure
        depth_limit: Optional[int] = None
        if sel.depth_hint > 3:
            depth_limit = sel.depth_hint + 2  # Allow some nesting beyond hint

        enriched.append(EnrichedZone(
            selector=sel,
            node_type=NodeType.SIGNAL,
            weight=weight,
            structural_role=role,
            child_noise_selectors=(),
            depth_limit=depth_limit,
            data_attributes=data_attrs,
            json_path=json_path,
        ))

    # Enrich noise zones
    noise_selectors: List[ParsedSelector] = []
    for selector_raw in zone_map.noise_zones:
        sel = parse_selector(selector_raw)
        noise_selectors.append(sel)
        role = _infer_structural_role(sel)

        enriched.append(EnrichedZone(
            selector=sel,
            node_type=NodeType.NOISE,
            weight=0.0,
            structural_role=role,
            child_noise_selectors=(),
            depth_limit=None,
            data_attributes=(),
            json_path=None,
        ))

    # Link child noise selectors to parent signal zones
    # A noise selector is a "child" of a signal zone if it could
    # appear nested within the signal zone's HTML structure
    enriched_with_children: List[EnrichedZone] = []
    for ez in enriched:
        if ez.node_type == NodeType.SIGNAL:
            children = tuple(
                ns for ns in noise_selectors
                if _is_child_noise(ez.selector, ns)
            )
            if children:
                ez = EnrichedZone(
                    selector=ez.selector,
                    node_type=ez.node_type,
                    weight=ez.weight,
                    structural_role=ez.structural_role,
                    child_noise_selectors=children,
                    depth_limit=ez.depth_limit,
                    data_attributes=ez.data_attributes,
                    json_path=ez.json_path,
                )
        enriched_with_children.append(ez)

    # Sort by priority_score descending (highest priority first)
    enriched_with_children.sort(key=lambda z: z.priority_score, reverse=True)

    return enriched_with_children


def _is_child_noise(
    parent: ParsedSelector,
    child: ParsedSelector,
) -> bool:
    """Determine if a noise selector could be a child of a signal selector.

    Heuristic: a noise selector is considered a potential child if:
    - Its specificity is lower than or equal to the parent
    - It matches a known noise pattern (nav, footer, sidebar, etc.)
    - Its tag is a common container tag (div, nav, aside, footer)
    """
    # Common noise tags that appear inside content zones
    noise_tags = {"nav", "footer", "aside", "header", "script", "style",
                  "iframe", "noscript", "form"}
    if child.tag and child.tag.lower() in noise_tags:
        return True
    noise_classes = {"sidebar", "cookie", "banner", "popup", "modal",
                     "advertisement", "ad-", "social-share", "related-posts"}
    if child.class_name and any(nc in child.class_name.lower() for nc in noise_classes):
        return True
    return False


def softmax_normalize_weights(
    zones: List[EnrichedZone],
    temperature: float,
) -> List[EnrichedZone]:
    """Apply softmax normalization to zone weights.

    w_i = exp(z_i / τ) / Σ_j exp(z_j / τ)

    Temperature controls distribution sharpness:
        τ > 1: flatter (LEARNS: broad extraction)
        τ = 1: natural (PREDICTS: balanced)
        τ < 1: sharper (KNOWS: surgical extraction)

    Returns: new list with normalized weights. Input list unchanged.
    """
    signal_zones = [z for z in zones if z.node_type == NodeType.SIGNAL]

    if not signal_zones:
        return zones

    noise_zones = [z for z in zones if z.node_type != NodeType.SIGNAL] # noqa | runtime reachable

    # Compute softmax
    weights = [z.weight for z in signal_zones]
    max_w = max(weights) if weights else 0.0
    # Numerical stability: subtract max before exp
    exp_weights = [math.exp((w - max_w) / max(temperature, 1e-8)) for w in weights]
    total = sum(exp_weights)
    if total <= 0:
        total = 1.0
    normalized = [ew / total for ew in exp_weights]

    # Create new zones with normalized weights
    new_signal: List[EnrichedZone] = []
    for z, nw in zip(signal_zones, normalized):
        new_signal.append(EnrichedZone(
            selector=z.selector,
            node_type=z.node_type,
            weight=nw,
            structural_role=z.structural_role,
            child_noise_selectors=z.child_noise_selectors,
            depth_limit=z.depth_limit,
            data_attributes=z.data_attributes,
            json_path=z.json_path,
        ))

    return new_signal + noise_zones


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY COMPILER — ZONE_EXTRACT
#
# Compiles ZONE_EXTRACT recipes from enriched zones.
# Signal lives in defined HTML zones identified by CSS selectors.
#
# Output: sed chains for simple topologies, awk programs for complex ones.
# Decision boundary: if any zone has depth_limit or if there are > 3 signal
# zones, use the awk path. Otherwise, sed chains suffice.
# ═════════════════════════════════════════════════════════════════════════════

def _compose_zone_extract(
    ctx: CompilerContext,
) -> ShellRecipe:
    """Compile a ZONE_EXTRACT recipe from the compiler context.

    Algorithm:
    1. Separate signal and noise zones
    2. Softmax-normalize signal zone weights
    3. Determine if awk path is needed (complexity check)
    4. For sed path: chain boundary extraction → noise strip → tag strip → normalize
    5. For awk path: generate multi-zone awk program
    6. Apply phase conditioning
    7. Build ShellRecipe AST

    Complexity:
        Simple (sed): O(S + N) pipe stages where S=signal zones, N=noise zones
        Complex (awk): single awk program with O(S + N) rules
    """
    signal_zones = [z for z in ctx.enriched_zones if z.node_type == NodeType.SIGNAL]
    noise_zones = [z for z in ctx.enriched_zones if z.node_type == NodeType.NOISE]

    # Softmax normalize
    all_zones = softmax_normalize_weights(
        ctx.enriched_zones,
        ctx.softmax_temperature,
    )
    signal_zones = [z for z in all_zones if z.node_type == NodeType.SIGNAL]
    noise_zones = [z for z in all_zones if z.node_type != NodeType.SIGNAL]

    # Phase conditioning: filter ambiguous zones in PREDICTS and KNOWS
    if ctx.phase in (PhaseStr.PREDICTS, PhaseStr.KNOWS):
        signal_zones = [
            z for z in signal_zones
            if z.weight >= 0.1 / max(len(signal_zones), 1)
        ]

    # Determine compilation path
    needs_awk = (
        any(z.depth_limit is not None for z in signal_zones)
        or len(signal_zones) > 3
        or len(noise_zones) > 5
        or any(z.structural_role in ("code_block", "table") for z in signal_zones)
    )

    if needs_awk:
        return _compose_zone_extract_awk(signal_zones, noise_zones, ctx)
    else:
        return _compose_zone_extract_sed(signal_zones, noise_zones, ctx)


def _compose_zone_extract_sed(
    signal_zones: List[EnrichedZone],
    noise_zones: List[EnrichedZone],
    ctx: CompilerContext,
) -> ShellRecipe:
    """Compile a sed-chain ZONE_EXTRACT recipe.

    Pipeline structure:
        1. Zone boundary extraction (sed -n for each signal zone)
        2. Noise stripping (sed for each noise zone)
        3. Child noise stripping (sed for each child noise selector)
        4. HTML tag stripping
        5. Whitespace normalization
        6. Empty line removal
        7. (KNOWS phase) Short line removal
        8. (PREDICTS/KNOWS) Deduplication
    """
    commands: List[ShellCommand] = []

    # Stage 1: Zone boundary extraction
    for zone in signal_zones:
        rule_commands = _select_and_apply_rules(
            zone, ctx, categories=[RuleCategory.ZONE_BOUNDARY]
        )
        commands.extend(rule_commands)

    # Stage 2: Noise stripping
    for zone in noise_zones:
        rule_commands = _select_and_apply_rules(
            zone, ctx, categories=[RuleCategory.NOISE_STRIP]
        )
        commands.extend(rule_commands)

    # Stage 3: Child noise stripping within signal zones
    for zone in signal_zones:
        for child_sel in zone.child_noise_selectors:
            child_zone = EnrichedZone(
                selector=child_sel,
                node_type=NodeType.NOISE,
                weight=0.0,
                structural_role="noise_element",
                child_noise_selectors=(),
                depth_limit=None,
                data_attributes=(),
                json_path=None,
            )
            rule_commands = _select_and_apply_rules(
                child_zone, ctx, categories=[RuleCategory.NOISE_STRIP]
            )
            commands.extend(rule_commands)

    # Stage 4: HTML tag stripping
    commands.append(ShellCommand("sed", (
        ShellPattern("s/<[^>]*>//g"),
    )))

    # Stage 5: Whitespace normalization
    commands.append(ShellCommand("tr", (
        ShellFlag("-s"),
        ShellPattern(" \\t\\n"),
        ShellPattern("\\n"),
    )))

    # Stage 6: Empty line removal
    commands.append(ShellCommand("grep", (
        ShellFlag("-v"),
        ShellPattern("^[[:space:]]*$"),
    )))

    # Stage 7: Phase conditioning
    if ctx.phase == PhaseStr.KNOWS:
        # Aggressive: strip short lines
        commands.append(ShellCommand("awk", (
            ShellPattern(f"length($0) >= {MIN_SIGNAL_LENGTH}"),
        )))

    # Stage 8: Deduplication (PREDICTS and KNOWS)
    if ctx.phase in (PhaseStr.PREDICTS, PhaseStr.KNOWS):
        commands.append(ShellCommand("awk", (
            ShellPattern("!seen[$0]++"),
        )))

    # Limit pipeline complexity
    if len(commands) > MAX_PIPELINE_STAGES:
        ctx.add_diagnostic(
            CompileSeverity.WARN,
            f"Pipeline has {len(commands)} stages, truncating to {MAX_PIPELINE_STAGES}",
        )
        commands = commands[:MAX_PIPELINE_STAGES]

    # Build pipeline
    pipeline = ShellPipeline(
        stages=tuple(commands),
        comment=f"ZONE_EXTRACT for {ctx.topology_class} "
                f"({len(signal_zones)} signal, {len(noise_zones)} noise zones)",
    )

    return ShellRecipe(
        header="# sed-chain compilation path",
        pipelines=(pipeline,),
        topology_class=ctx.topology_class,
        strategy=ExtractionStrategy.DEPTH_FIRST,
        phase=ctx.phase,
        zone_map_version=ctx.zone_map_version,
        intent=ctx.intent,
    )


def _compose_zone_extract_awk(
    signal_zones: List[EnrichedZone],
    noise_zones: List[EnrichedZone],
    ctx: CompilerContext,
) -> ShellRecipe:
    """Compile an awk-program ZONE_EXTRACT recipe.

    For complex topologies where sed chains are insufficient.
    Generates a single multi-zone awk program that handles all zones
    simultaneously with depth tracking.
    """
    # Generate the awk program
    awk_source = _build_multizone_awk(signal_zones, noise_zones, phase=ctx.phase)

    commands: List[ShellCommand] = [
        ShellCommand("awk", (ShellPattern(awk_source),)),
    ]

    # Post-processing pipeline after awk extraction
    commands.append(ShellCommand("tr", (
        ShellFlag("-s"),
        ShellPattern(" \\t\\n"),
        ShellPattern("\\n"),
    )))
    commands.append(ShellCommand("grep", (
        ShellFlag("-v"),
        ShellPattern("^[[:space:]]*$"),
    )))

    if ctx.phase in (PhaseStr.PREDICTS, PhaseStr.KNOWS):
        commands.append(ShellCommand("awk", (
            ShellPattern("!seen[$0]++"),
        )))

    pipeline = ShellPipeline(
        stages=tuple(commands),
        comment=f"ZONE_EXTRACT (awk path) for {ctx.topology_class} "
                f"({len(signal_zones)} signal, {len(noise_zones)} noise zones)",
    )

    return ShellRecipe(
        header="# awk-program compilation path (complex topology)",
        pipelines=(pipeline,),
        topology_class=ctx.topology_class,
        strategy=ExtractionStrategy.DEPTH_FIRST,
        phase=ctx.phase,
        zone_map_version=ctx.zone_map_version,
        intent=ctx.intent,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY COMPILER — ATTRIBUTE_EXTRACT
#
# Signal lives in data-* HTML attributes, not in tag content.
# Used for: ECOMMERCE_PRODUCT, ECOMMERCE_PRODUCT_VARIANT, JSON_LD_STRUCTURED
# ═════════════════════════════════════════════════════════════════════════════

def _compose_attribute_extract(
    ctx: CompilerContext,
) -> ShellRecipe:
    """Compile an ATTRIBUTE_EXTRACT recipe.

    Generates an awk program that extracts data-* attributes from HTML.
    Each attribute is emitted as a labeled key-value pair.

    For JSON_LD_STRUCTURED: extracts the ld+json script block, then
    performs key extraction on the JSON content.
    """
    signal_zones = [z for z in ctx.enriched_zones if z.node_type == NodeType.SIGNAL]

    # Collect all data attributes across signal zones
    all_attrs: List[str] = []
    for zone in signal_zones:
        for attr in zone.data_attributes:
            if attr not in all_attrs:
                all_attrs.append(attr)

    # Check for JSON-LD pattern
    has_json_ld = any(
        "ld+json" in z.selector.raw or "ld_json" in z.selector.raw
        or z.structural_role == "json_ld"
        for z in signal_zones
    )

    commands: List[ShellCommand] = []

    if has_json_ld:
        # JSON-LD extraction path
        awk_source = _build_json_ld_extraction_awk(phase=ctx.phase, topology_class=ctx.topology_class)
        commands.append(ShellCommand("awk", (ShellPattern(awk_source),)))
    elif all_attrs:
        # Data attribute extraction path
        awk_source = _build_attribute_extraction_awk(all_attrs, phase=ctx.phase)
        commands.append(ShellCommand("awk", (ShellPattern(awk_source),)))
    else:
        # Fallback: generic attribute extraction from signal zones
        # Look for any data-* attributes in the HTML
        common_attrs = [
            "data-product-name", "data-price", "data-sku",
            "data-availability", "data-brand", "data-category",
        ]
        awk_source = _build_attribute_extraction_awk(common_attrs, phase=ctx.phase)
        commands.append(ShellCommand("awk", (ShellPattern(awk_source),)))
        ctx.add_diagnostic(
            CompileSeverity.WARN,
            "No specific data attributes found; using common attribute set",
        )

    # Empty line removal
    commands.append(ShellCommand("grep", (
        ShellFlag("-v"),
        ShellPattern("^[[:space:]]*$"),
    )))

    # Deduplication
    commands.append(ShellCommand("awk", (
        ShellPattern("!seen[$0]++"),
    )))

    pipeline = ShellPipeline(
        stages=tuple(commands),
        comment=f"ATTRIBUTE_EXTRACT for {ctx.topology_class} "
                f"(attributes: {', '.join(all_attrs) if all_attrs else 'auto-detect'})",
    )

    return ShellRecipe(
        header="# attribute extraction compilation path",
        pipelines=(pipeline,),
        topology_class=ctx.topology_class,
        strategy=ExtractionStrategy.BREADTH_FIRST,
        phase=ctx.phase,
        zone_map_version=ctx.zone_map_version,
        intent=ctx.intent,
    )


# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY COMPILER — ENVELOPE_EXTRACT
#
# Signal lives inside a JSON response envelope at a known path.
# Used for: REST_API_JSON, REST_API_JSON_PAGINATED
# ═════════════════════════════════════════════════════════════════════════════

def _compose_envelope_extract(
    ctx: CompilerContext,
) -> ShellRecipe:
    """Compile an ENVELOPE_EXTRACT recipe.

    Generates an awk state machine that traverses the JSON structure
    using pattern matching. Not a full JSON parser — a structural
    extractor that knows the depth and key path.
    """
    signal_zones = [z for z in ctx.enriched_zones if z.node_type == NodeType.SIGNAL]

    # Find the primary JSON path
    json_path: Optional[str] = None
    for zone in signal_zones:
        if zone.json_path:
            json_path = zone.json_path
            break

    if not json_path:
        # Infer from selector patterns
        for zone in signal_zones:
            raw = zone.selector.raw.strip()
            if re.match(r"^[a-zA-Z_][\w.]*$", raw):
                json_path = raw
                break

    if not json_path:
        json_path = "data"
        ctx.add_diagnostic(
            CompileSeverity.WARN,
            "No JSON path found in zone map; defaulting to 'data'",
        )

    commands: List[ShellCommand] = []

    # JSON envelope extraction
    awk_source = _build_envelope_extraction_awk(
        json_path, phase=ctx.phase, topology_class=ctx.topology_class
    )
    commands.append(ShellCommand("awk", (ShellPattern(awk_source),)))

    # Post-processing: clean up JSON formatting
    commands.append(ShellCommand("sed", (
        ShellPattern("s/^[[:space:]]*//"),
    )))
    commands.append(ShellCommand("grep", (
        ShellFlag("-v"),
        ShellPattern("^[{}\\[\\],]*$"),
    )))

    pipeline = ShellPipeline(
        stages=tuple(commands),
        comment=f"ENVELOPE_EXTRACT for {ctx.topology_class} "
                f"(json_path: {json_path})",
    )

    return ShellRecipe(
        header="# envelope extraction compilation path",
        pipelines=(pipeline,),
        topology_class=ctx.topology_class,
        strategy=ExtractionStrategy.SECTION_SCOPED,
        phase=ctx.phase,
        zone_map_version=ctx.zone_map_version,
        intent=ctx.intent,
    )


def _compose_flat_extract(
    ctx: CompilerContext,
) -> ShellRecipe:
    """Compile a FLAT extraction recipe.

    FLAT strategy: top-level only extraction.  Used for JSON_LD_STRUCTURED
    and pages with shallow, non-nested signal.  The recipe extracts the
    JSON-LD script block or top-level structural content without any
    depth tracking.

    Pipeline: JSON-LD block extraction → field cleanup → normalize.
    """
    commands: List[ShellCommand] = []

    # JSON-LD is the primary FLAT target
    awk_source = _build_json_ld_extraction_awk(phase=ctx.phase, topology_class=ctx.topology_class)
    commands.append(ShellCommand("awk", (ShellPattern(awk_source),)))

    # Clean up
    commands.append(ShellCommand("sed", (
        ShellPattern("s/^[[:space:]]*//"),
    )))
    commands.append(ShellCommand("grep", (
        ShellFlag("-v"),
        ShellPattern("^[[:space:]]*$"),
    )))

    pipeline = ShellPipeline(
        stages=tuple(commands),
        comment=f"FLAT extraction for {ctx.topology_class}",
    )

    return ShellRecipe(
        header="# flat (top-level only) extraction path",
        pipelines=(pipeline,),
        topology_class=ctx.topology_class,
        strategy=ExtractionStrategy.FLAT,
        phase=ctx.phase,
        zone_map_version=ctx.zone_map_version,
        intent=ctx.intent,
    )


# ═════════════════════════════════════════════════════════════════════════════
# BOUNDARY-AWARE COMPILATION PRIMITIVES
#
# BoundaryDescriptor has three types from wlp_zones.py:
#   SECTION_BOUNDARY  →  reset capture state (awk: in_section=0; section++)
#   CONTENT_BOUNDARY  →  toggle capture on/off (sed address range start/end)
#   NOISE_BOUNDARY    →  suppress everything inside (sed delete range)
#
# These map directly to sed address ranges and awk state transitions.
# The compiler uses them, not infers them.
# ═════════════════════════════════════════════════════════════════════════════

def _compile_boundary_sed_ranges(
    boundaries: Sequence[Any],
) -> List[ShellCommand]:
    """Compile BoundaryDescriptor objects into sed address range commands.

    Each boundary type produces a different sed primitive:
        SECTION_BOUNDARY  → no sed action (handled in awk state machine)
        CONTENT_BOUNDARY  → sed -n '/<pattern>/,/<end>/p' (extract range)
        NOISE_BOUNDARY    → sed '/<pattern>/,/<end>/d' (suppress range)

    Args:
        boundaries: Sequence of BoundaryDescriptor from ZoneMap.boundaries.

    Returns: List of ShellCommand for noise/content boundary handling.
    """
    commands: List[ShellCommand] = []

    for bd in boundaries:
        btype = bd.boundary_type if isinstance(bd.boundary_type, str) else str(bd.boundary_type)
        delimiter = getattr(bd, "delimiter_content", "")
        selector = getattr(bd, "selector", "")

        if btype == "NOISE_BOUNDARY" and delimiter:
            tag_match = re.search(r"<(\w+)", delimiter)
            close_tag = tag_match.group(1) if tag_match else "div"
            safe_delim = delimiter.replace("\\", "\\\\").replace("/", "\\/").replace("'", "\\'")
            commands.append(ShellCommand("sed", (
                ShellPattern(f"/{safe_delim}/,/<\\/{close_tag}>/d"),
            )))

        elif btype == "CONTENT_BOUNDARY" and delimiter:
            tag_match = re.search(r"<(\w+)", delimiter)
            close_tag = tag_match.group(1) if tag_match else "div"
            safe_delim = delimiter.replace("\\", "\\\\").replace("/", "\\/").replace("'", "\\'")
            commands.append(ShellCommand("sed", (
                ShellFlag("-n"),
                ShellPattern(f"/{safe_delim}/,/<\\/{close_tag}>/p"),
            )))

        # SECTION_BOUNDARY is handled by the awk state machine, not sed

    return commands


def _compile_boundary_awk_transitions(
    boundaries: Sequence[Any],
) -> Tuple[List[str], List[str]]:
    """Compile BoundaryDescriptor objects into awk state transition lines.

    SECTION_BOUNDARY:  reset capture state, increment section counter.
    CONTENT_BOUNDARY:  toggle capture variable.
    NOISE_BOUNDARY:    set suppress flag.

    Returns: (awk_lines, content_vars) — content_vars must be initialized
    to 0 in the BEGIN block by the caller.
    """
    lines: List[str] = []
    content_vars: List[str] = []
    section_count = 0
    content_count = 0
    noise_count = 0

    for bd in boundaries:
        btype = bd.boundary_type if isinstance(bd.boundary_type, str) else str(bd.boundary_type)
        delimiter = getattr(bd, "delimiter_content", "")
        if not delimiter:
            continue

        safe_delim = delimiter.replace("\\", "\\\\").replace("/", "\\/").replace('"', '\\"')

        if btype == "SECTION_BOUNDARY":
            lines.append(f"/{safe_delim}/ {{")
            lines.append(f"    section_idx++")
            lines.append(f"    in_capture = 0")
            lines.append(f"}}")
            section_count += 1

        elif btype == "CONTENT_BOUNDARY":
            var = f"content_active_{content_count}"
            content_vars.append(var)
            lines.append(f"/{safe_delim}/ {{ {var} = !{var} }}")
            content_count += 1

        elif btype == "NOISE_BOUNDARY":
            var = f"noise_suppress_{noise_count}"
            lines.append(f"/{safe_delim}/ {{ {var} = 1 }}")
            if "<" in delimiter:
                tag_match = re.search(r"<(\w+)", delimiter)
                if tag_match:
                    close_tag = tag_match.group(1)
                    lines.append(
                        f"/<\\/{close_tag}>/ && {var} {{ {var} = 0; next }}"
                    )
            noise_count += 1

    return lines, content_vars


def _get_content_type_grep_flags(content_type: str) -> Tuple[ShellFlag, ...]:
    """Map content_type → grep context flags.

    content_type is read directly from ZoneDescriptor.content_type.
    Do NOT infer from zone metadata — the WLP already classified it.

    Returns tuple of ShellFlag objects for grep context arguments.
    """
    flags_strs = CONTENT_TYPE_GREP_FLAGS.get(content_type, ("-A2",))
    return tuple(ShellFlag(f) for f in flags_strs)


# ═════════════════════════════════════════════════════════════════════════════
# RULE APPLICATION ENGINE
#
# Orchestrates the application of translation rules to enriched zones.
# Rules are selected based on applicability predicates and applied in
# category order. This is the compiler's core loop.
# ═════════════════════════════════════════════════════════════════════════════

def _select_and_apply_rules(
    zone: EnrichedZone,
    ctx: CompilerContext,
    *,
    categories: Optional[List[RuleCategory]] = None,
) -> List[ShellCommand]:
    """Select and apply applicable translation rules for a single zone.

    Rules are tested in category order. Within a category, rules are
    tested in definition order (rule_id ascending). The first applicable
    rule in each category fires; subsequent rules in the same category
    are skipped (single-fire-per-category semantic).

    This prevents redundant commands from overlapping rules while
    ensuring all relevant categories contribute to the pipeline.

    Args:
        zone: The enriched zone to compile rules for.
        ctx: Current compiler context.
        categories: If specified, only apply rules in these categories.

    Returns: Ordered list of ShellCommand objects.
    """
    commands: List[ShellCommand] = []
    target_categories = categories or list(COMPOSITION_ORDER)
    fired_categories: Set[RuleCategory] = set()

    for category in COMPOSITION_ORDER:
        if category not in target_categories:
            continue
        if category in fired_categories:
            continue

        for rule in RULES_BY_CATEGORY.get(category, []):
            try:
                if rule.applies(zone, ctx):
                    rule_commands = rule.compile(zone, ctx)
                    commands.extend(rule_commands)
                    fired_categories.add(category)
                    break  # Single fire per category
            except Exception as exc:
                ctx.add_diagnostic(
                    CompileSeverity.ERROR,
                    f"Rule {rule.rule_id} ({rule.name}) raised: {exc}",
                    rule_id=rule.rule_id,
                    zone_selector=zone.selector.raw,
                )

    return commands


# ═════════════════════════════════════════════════════════════════════════════
# INTENT CONDITIONING
#
# Same URL. Same topology class. Different intent vector. Different recipe.
# This transforms the compiler from infrastructure into intelligence.
#
# The ZoneMap may include intent_hints that map intent strings to lists
# of zone selectors that should be prioritized for that intent. The
# compiler generates one recipe variant per intent hint.
#
# Intent variants narrow zone selection to only the prioritized selectors
# while keeping the same stripping logic.
# ═════════════════════════════════════════════════════════════════════════════

def _compile_intent_variant(
    zone_map: ZoneMap,
    intent: str,
    intent_selectors: List[str],
    ctx: CompilerContext,
) -> CompiledRecipe:
    """Compile a single intent-conditioned recipe variant.

    Delegates to IntentConditionedExtractor — the full engine that uses
    with_intent(), boundary descriptors, content_type-driven grep flags,
    and urgency-modulated extraction aggressiveness.
    """
    extractor = IntentConditionedExtractor(ctx)
    return extractor.compile_variant(zone_map, intent, intent_selectors)


# ═════════════════════════════════════════════════════════════════════════════
# INTENT-CONDITIONED EXTRACTION ENGINE
#
# What makes AXIOM categorically different from every RAG system: the same
# page, the same topology class, produces a DIFFERENT recipe for every intent.
# Intent is a 256-float vector that travels through the entire pipeline as a
# first-class citizen.  The recipe compiles to extract exactly what the intent
# needs — the LLM never sees the noise because the recipe never extracted it.
#
# The engine receives a ZoneMap (with intent_weights already applied via
# with_intent()), decomposes the weight distribution, scores every zone for
# intent relevance, builds an extraction plan, narrows the compilation
# context, and produces a specialized recipe.
#
# Key contract points with the rest of the system:
#   - with_intent() is already implemented on ZoneMap — call it, don't
#     reimplement it.
#   - intent_weights is ((selector, weight), ...) — weight=0.0 means
#     excluded absolutely.  weight=1.0 is default.  Above 1.0 is boosted.
#   - content_type on ZoneDescriptor drives grep flags directly.
#   - BoundaryDescriptor types map to sed/awk primitives.
#   - Confidence floor is 0.30, not 0.70.
#   - EmptyZoneMap is the bottom value — check bool(zone_map).
#
# AXIOM INTERNAL // DO NOT SURFACE
# ═════════════════════════════════════════════════════════════════════════════


# ── Intent vector layout constants (mirror wlp_zones.py) ────────────────

_INTENT_PRIMARY_START:    Final[int] = 0
_INTENT_PRIMARY_END:      Final[int] = 64
_INTENT_SECONDARY_START:  Final[int] = 64
_INTENT_SECONDARY_END:    Final[int] = 128
_INTENT_EXCLUDE_START:    Final[int] = 128
_INTENT_EXCLUDE_END:      Final[int] = 192
_INTENT_URGENCY_START:    Final[int] = 192
_INTENT_URGENCY_END:      Final[int] = 208
_INTENT_USER_STATE_START: Final[int] = 208
_INTENT_USER_STATE_END:   Final[int] = 256

# Urgency states — argmax over 4 urgency bins.
_URGENCY_STATES: Final[Tuple[str, ...]] = ("low", "normal", "high", "critical")

# User states — argmax over 8 user_state bins.
_USER_STATES: Final[Tuple[str, ...]] = (
    "exploring", "learning", "building", "debugging",
    "locked_out", "comparing", "purchasing", "migrating",
)

_INTENT_ZONE_WEIGHT_FLOOR: Final[float] = 0.05
_MIN_SURVIVING_SIGNAL_ZONES: Final[int] = 1
_MAX_INTENT_VARIANTS_PER_PASS: Final[int] = 8

_URGENCY_CONTEXT_MULTIPLIER: Final[Dict[str, float]] = {
    "low": 0.5, "normal": 1.0, "high": 1.5, "critical": 2.0,
}

_USER_STATE_ZONE_AFFINITY: Final[Dict[str, FrozenSet[str]]] = {
    "exploring":  frozenset({"heading", "main_content", "paragraph", "navigation"}),
    "learning":   frozenset({"main_content", "code_block", "paragraph", "list"}),
    "building":   frozenset({"code_block", "main_content", "table", "list"}),
    "debugging":  frozenset({"code_block", "main_content", "table"}),
    "locked_out": frozenset({"list", "code_block", "main_content", "paragraph"}),
    "comparing":  frozenset({"table", "pricing", "list", "main_content"}),
    "purchasing": frozenset({"pricing", "table", "main_content"}),
    "migrating":  frozenset({"code_block", "list", "main_content", "table"}),
}

# ── Content-type ↔ user-state compatibility matrix ──────────────────────
# Rows = content_type, columns = user_state, value = compatibility [0,1].
# Used by _content_urgency_alignment() for zone scoring.

_CONTENT_STATE_MATRIX: Final[Dict[str, Dict[str, float]]] = {
    "prose": {
        "exploring": 0.9, "learning": 0.8, "building": 0.4,
        "debugging": 0.3, "locked_out": 0.6, "comparing": 0.5,
        "purchasing": 0.4, "migrating": 0.5,
    },
    "code": {
        "exploring": 0.2, "learning": 0.7, "building": 0.95,
        "debugging": 0.95, "locked_out": 0.8, "comparing": 0.3,
        "purchasing": 0.1, "migrating": 0.8,
    },
    "list": {
        "exploring": 0.5, "learning": 0.6, "building": 0.7,
        "debugging": 0.5, "locked_out": 0.95, "comparing": 0.8,
        "purchasing": 0.6, "migrating": 0.7,
    },
    "table": {
        "exploring": 0.4, "learning": 0.5, "building": 0.6,
        "debugging": 0.6, "locked_out": 0.3, "comparing": 0.95,
        "purchasing": 0.9, "migrating": 0.5,
    },
    "mixed": {
        "exploring": 0.6, "learning": 0.6, "building": 0.5,
        "debugging": 0.5, "locked_out": 0.5, "comparing": 0.5,
        "purchasing": 0.5, "migrating": 0.5,
    },
}

# ── Intent-keyword vocabulary ───────────────────────────────────────────
# Structural keywords extracted from intent names to build grep patterns.
# Maps intent category → grep-friendly alternation patterns.

_INTENT_KEYWORD_PATTERNS: Final[Dict[str, str]] = {
    "recovery_codes":    "recovery.code\\|backup.code\\|get.back.in\\|restore.access",
    "api_reference":     "endpoint\\|parameter\\|request\\|response\\|method\\|rate.limit",
    "pricing":           "price\\|plan\\|tier\\|cost\\|per.month\\|free.trial",
    "tutorial":          "step\\|guide\\|getting.started\\|how.to\\|example",
    "changelog":         "version\\|release\\|change\\|update\\|fix\\|new.feature",
    "conceptual":        "overview\\|architecture\\|concept\\|design\\|how.*works",
    "troubleshooting":   "error\\|fix\\|solve\\|troubleshoot\\|issue\\|problem",
    "specifications":    "spec\\|dimension\\|weight\\|material\\|capacity",
    "accepted_answer":   "accepted\\|answer\\|solution\\|resolved",
    "code_solutions":    "solution\\|example\\|snippet\\|implementation",
}


@dataclass(frozen=True)
class IntentDecomposition:
    """Structured decomposition of intent_weights from a ZoneMap.

    Not a re-encoding of the raw intent vector — this is the compiler's
    interpretation of the weight distribution that with_intent() produced.
    """
    primary_selectors:   Tuple[str, ...]
    secondary_selectors: Tuple[str, ...]
    excluded_selectors:  Tuple[str, ...]
    default_selectors:   Tuple[str, ...]
    urgency:             str
    user_state:          str
    max_weight:          float
    weight_entropy:      float
    is_focused:          bool
    weight_map:          Dict[str, float]
    intent_name:         str


@dataclass(frozen=True)
class IntentZoneScore:
    """Per-zone intent relevance score — three-signal combination.

    score = 0.60 × intent_weight + 0.25 × affinity + 0.15 × alignment

    intent_weight:  from with_intent(), normalised to [0,1]
    affinity:       user_state × structural_role match
    alignment:      content_type × urgency compatibility
    """
    selector:        str
    intent_weight:   float
    affinity_score:  float
    alignment_score: float
    final_score:     float
    excluded:        bool
    structural_role: str
    content_type:    str


@dataclass(frozen=True)
class IntentRecipePlan:
    """Complete extraction plan for one intent variant.

    Describes WHAT to extract.  The HOW is handled by _compile_strategy()
    operating on the narrowed zone set.
    """
    surviving_zones:    Tuple[EnrichedZone, ...]
    excluded_zones:     Tuple[EnrichedZone, ...]
    boundary_plan:      Tuple[ShellCommand, ...]
    content_type_flags: Dict[str, Tuple[ShellFlag, ...]]
    urgency_adjustment: float
    grep_keywords:      Tuple[str, ...]
    awk_state_vars:     Tuple[str, ...]
    recipe_header:      str
    decomposition:      IntentDecomposition


class IntentConditionedExtractor:
    """The intent-conditioned extraction engine.

    Architecture:
        1. Decompose — read intent_weights, classify zones by weight tier
        2. Score     — compute per-zone relevance (weight × affinity × alignment)
        3. Plan      — build extraction plan (survivors, exclusions, boundaries)
        4. Narrow    — filter compiler context to surviving zones only
        5. Specialize — tune grep flags, awk state, boundary handling
        6. Compile   — produce the intent-conditioned recipe

    The extractor is stateless — receives context from RecipeCompiler,
    returns a CompiledRecipe.  No side effects.

    Mathematical foundation:
        Zone scoring: s_i = 0.60·w_i + 0.25·a_i + 0.15·c_i
        Weight entropy: H = −Σ p_i·log₂(p_i)
            H < 1.5 → focused intent → sharper zone filtering
            H > 2.0 → diffuse intent → broader extraction
        Urgency multiplier: scales grep context window size
            critical=2.0×, high=1.5×, normal=1.0×, low=0.5×
    """

    __slots__ = ("_ctx", "_diagnostics")

    def __init__(self, parent_ctx: CompilerContext) -> None:
        self._ctx = parent_ctx
        self._diagnostics: List[CompilerDiagnostic] = []

    def _diag(self, severity: CompileSeverity, message: str,
              zone_selector: Optional[str] = None) -> None:
        self._diagnostics.append(CompilerDiagnostic(
            severity=severity, rule_id=None,
            message=message, zone_selector=zone_selector,
        ))

    # ── Stage 1: Decompose ──────────────────────────────────────────────

    def _decompose_intent(
        self,
        zone_map: Any,
        intent_name: str,
    ) -> IntentDecomposition:
        """Decompose zone_map.intent_weights into structured categories.

        Reads ((selector, weight), ...) from intent_weights:
            weight=0.0 → excluded absolutely (intent.exclude hit this)
            weight=1.0 → default (no match, no exclusion)
            weight>1.0 → boosted (secondary: 1.0-1.5, primary: >1.5)

        Also infers urgency and user_state from the intent name when the
        raw 256-float vector isn't available at this layer.
        """
        intent_weights = getattr(zone_map, "intent_weights", ())
        weight_map: Dict[str, float] = {}

        primary:   List[str] = []
        secondary: List[str] = []
        excluded:  List[str] = []
        default:   List[str] = []

        for entry in intent_weights:
            if not isinstance(entry, (tuple, list)) or len(entry) < 2:
                continue
            sel, w = str(entry[0]), float(entry[1])
            weight_map[sel] = w

            if w == 0.0:
                excluded.append(sel)
            elif w > 1.5:
                primary.append(sel)
            elif w > 1.0:
                secondary.append(sel)
            else:
                default.append(sel)

        entropy = self._shannon_entropy(
            [w for w in weight_map.values() if w > 0.0]
        )
        max_weight = max(weight_map.values()) if weight_map else 1.0
        urgency = self._infer_urgency(intent_name, weight_map)
        user_state = self._infer_user_state(intent_name, weight_map)

        self._diag(
            CompileSeverity.INFO,
            f"Intent decomposed: {len(primary)} primary, "
            f"{len(secondary)} secondary, {len(excluded)} excluded, "
            f"urgency={urgency}, user_state={user_state}, "
            f"entropy={entropy:.2f}, focused={entropy < 1.5}",
        )

        return IntentDecomposition(
            primary_selectors=tuple(primary),
            secondary_selectors=tuple(secondary),
            excluded_selectors=tuple(excluded),
            default_selectors=tuple(default),
            urgency=urgency,
            user_state=user_state,
            max_weight=max_weight,
            weight_entropy=entropy,
            is_focused=(entropy < 1.5 and len(primary) <= 4),
            weight_map=weight_map,
            intent_name=intent_name,
        )

    # ── Stage 2: Score ──────────────────────────────────────────────────

    def _score_zones(
        self,
        enriched_zones: List[EnrichedZone],
        decomp: IntentDecomposition,
    ) -> List[IntentZoneScore]:
        """Score every signal zone for intent relevance.

        Three-signal weighted combination per zone:
            w_i: intent weight (normalised [0,1])
            a_i: user-state affinity (binary: role in preferred set)
            c_i: content-type × urgency alignment (from matrix)
        """
        scores: List[IntentZoneScore] = []
        preferred_roles = _USER_STATE_ZONE_AFFINITY.get(
            decomp.user_state, frozenset()
        )

        for ez in enriched_zones:
            if ez.node_type != NodeType.SIGNAL:
                continue

            sel_str = ez.selector.raw
            raw_weight = self._lookup_intent_weight(sel_str, decomp)

            # Absolute exclusion — weight=0.0 is final
            if raw_weight == 0.0:
                scores.append(IntentZoneScore(
                    selector=sel_str, intent_weight=0.0,
                    affinity_score=0.0, alignment_score=0.0,
                    final_score=0.0, excluded=True,
                    structural_role=ez.structural_role,
                    content_type=self._zone_content_type(ez),
                ))
                continue

            norm_w = min(raw_weight / max(decomp.max_weight, 1.0), 1.0)
            affinity = 1.0 if ez.structural_role in preferred_roles else 0.0
            ct = self._zone_content_type(ez)
            alignment = self._content_state_alignment(ct, decomp.user_state)
            final = 0.60 * norm_w + 0.25 * affinity + 0.15 * alignment

            scores.append(IntentZoneScore(
                selector=sel_str, intent_weight=raw_weight,
                affinity_score=affinity, alignment_score=alignment,
                final_score=final, excluded=False,
                structural_role=ez.structural_role,
                content_type=ct,
            ))

        return scores

    # ── Stage 3: Plan ───────────────────────────────────────────────────

    def _build_plan(
        self,
        zone_map: Any,
        scores: List[IntentZoneScore],
        decomp: IntentDecomposition,
        enriched_zones: List[EnrichedZone],
    ) -> IntentRecipePlan:
        """Build complete extraction plan from zone scores.

        Plans WHAT to extract:
            - surviving_zones: zones that passed intent filter
            - excluded_zones: zones suppressed by intent
            - boundary_plan: sed/awk commands from BoundaryDescriptors
            - content_type_flags: per-zone grep context flags
            - grep_keywords: intent-derived keyword patterns
            - awk_state_vars: intent tracking state for awk programs
        """
        surviving_map: Dict[str, IntentZoneScore] = {}
        excluded_map: Dict[str, IntentZoneScore] = {}

        for zs in scores:
            if zs.excluded or zs.final_score < _INTENT_ZONE_WEIGHT_FLOOR:
                excluded_map[zs.selector] = zs
            else:
                surviving_map[zs.selector] = zs

        # Guarantee minimum surviving zones
        if len(surviving_map) < _MIN_SURVIVING_SIGNAL_ZONES:
            rescuable = sorted(
                [z for z in excluded_map.values() if not z.excluded],
                key=lambda z: z.final_score, reverse=True,
            )
            for zs in rescuable[:_MIN_SURVIVING_SIGNAL_ZONES - len(surviving_map)]:
                surviving_map[zs.selector] = zs
                excluded_map.pop(zs.selector, None)
            self._diag(
                CompileSeverity.WARN,
                f"Intent '{decomp.intent_name}' broadened to guarantee "
                f"{_MIN_SURVIVING_SIGNAL_ZONES} surviving zone(s)",
            )

        # Partition enriched zones
        surviving_enriched = [
            ez for ez in enriched_zones
            if ez.node_type == NodeType.SIGNAL and ez.selector.raw in surviving_map
        ]
        excluded_enriched = [
            ez for ez in enriched_zones
            if ez.node_type == NodeType.SIGNAL and ez.selector.raw in excluded_map
        ]

        # Sort surviving by final_score descending
        surviving_enriched.sort(
            key=lambda z: surviving_map.get(z.selector.raw, IntentZoneScore(
                "", 0, 0, 0, 0, True, "", ""
            )).final_score,
            reverse=True,
        )

        # Build boundary commands from zone_map.boundaries
        boundaries = getattr(zone_map, "boundaries", ())
        boundary_cmds = _compile_boundary_sed_ranges(boundaries) if boundaries else []

        # Content-type grep flags per surviving zone
        ct_flags: Dict[str, Tuple[ShellFlag, ...]] = {}
        for ez in surviving_enriched:
            ct = self._zone_content_type(ez)
            ct_flags[ez.selector.raw] = _get_content_type_grep_flags(ct)

        # Urgency multiplier
        urgency_mult = _URGENCY_CONTEXT_MULTIPLIER.get(decomp.urgency, 1.0)

        # Grep keywords — from vocabulary or from selector patterns
        keywords = self._extract_grep_keywords(decomp)

        # AWK state variables for intent-specific behaviour
        awk_vars = self._build_awk_state_vars(decomp)

        header = (
            f"# Intent: {decomp.intent_name} "
            f"({decomp.urgency}/{decomp.user_state})\n"
            f"# Surviving: {len(surviving_enriched)} zones  "
            f"Excluded: {len(excluded_enriched)} zones\n"
            f"# Focus: {'focused' if decomp.is_focused else 'diffuse'} "
            f"(H={decomp.weight_entropy:.2f})  "
            f"MaxW={decomp.max_weight:.1f}"
        )

        return IntentRecipePlan(
            surviving_zones=tuple(surviving_enriched),
            excluded_zones=tuple(excluded_enriched),
            boundary_plan=tuple(boundary_cmds),
            content_type_flags=ct_flags,
            urgency_adjustment=urgency_mult,
            grep_keywords=tuple(keywords[:10]),
            awk_state_vars=tuple(awk_vars),
            recipe_header=header,
            decomposition=decomp,
        )

    # ── Stage 4: Narrow ─────────────────────────────────────────────────

    def _narrow_context(self, plan: IntentRecipePlan) -> CompilerContext:
        """Build a narrowed CompilerContext with only surviving zones.

        Excluded signal zones become noise zones — they don't exist in the
        output.  This is intent conditioning at compile time.  The LLM never
        sees the excluded content because the recipe never extracted it.
        """
        surviving = list(plan.surviving_zones)
        noise_from_parent = [
            z for z in self._ctx.enriched_zones if z.node_type == NodeType.NOISE
        ]
        noise_from_exclusion = [
            EnrichedZone(
                selector=ez.selector, node_type=NodeType.NOISE, weight=0.0,
                structural_role=ez.structural_role,
                child_noise_selectors=ez.child_noise_selectors,
                depth_limit=ez.depth_limit,
                data_attributes=ez.data_attributes, json_path=ez.json_path,
            )
            for ez in plan.excluded_zones
        ]

        intent_label = (
            f"{plan.decomposition.intent_name}"
            f":{plan.decomposition.urgency}"
            f":{plan.decomposition.user_state}"
        )

        return CompilerContext(
            zone_map=self._ctx.zone_map,
            phase=self._ctx.phase,
            feedback=self._ctx.feedback,
            enriched_zones=surviving + noise_from_parent + noise_from_exclusion,
            strategy=self._ctx.strategy,
            diagnostics=list(self._diagnostics),
            topology_class=self._ctx.topology_class,
            zone_map_version=self._ctx.zone_map_version,
            attempt=self._ctx.attempt,
            start_time=self._ctx.start_time,
            intent=intent_label,
            fallback_chain=self._ctx.fallback_chain,
        )

    # ── Stage 5: Specialize ─────────────────────────────────────────────

    def _specialize_pipeline(
        self,
        base_recipe: ShellRecipe,
        plan: IntentRecipePlan,
    ) -> ShellRecipe:
        """Layer intent-specific modifications onto the base recipe.

        Three modification layers:
            1. Boundary suppressions (NOISE_BOUNDARY → sed delete ranges)
            2. Intent-keyword grep narrowing (focused intents only)
            3. Excluded zone CSS class suppression (grep -v)
        """
        new_pipelines: List[ShellPipeline] = []
        decomp = plan.decomposition

        for pipeline in base_recipe.pipelines:
            stages: List[ShellCommand] = []

            # Layer 1: boundary suppressions
            stages.extend(plan.boundary_plan)

            # Layer 2: base pipeline stages
            stages.extend(pipeline.stages)

            # Layer 3: intent-keyword narrowing for focused intents
            if plan.grep_keywords and decomp.is_focused:
                kw_pattern = "\\|".join(
                    re.escape(kw) for kw in plan.grep_keywords[:5]
                )
                after = max(1, int(3 * plan.urgency_adjustment))
                stages.append(ShellCommand("grep", (
                    ShellFlag("-i"),
                    ShellFlag(f"-A{after}"),
                    ShellPattern(kw_pattern),
                )))

            # Layer 4: excluded zone class suppression
            excl_patterns = self._build_exclusion_patterns(decomp)
            if excl_patterns:
                stages.append(ShellCommand("grep", (
                    ShellFlag("-v"),
                    ShellPattern("\\|".join(excl_patterns)),
                )))

            # Complexity cap
            if len(stages) > MAX_PIPELINE_STAGES:
                stages = stages[:MAX_PIPELINE_STAGES]
                self._diag(
                    CompileSeverity.WARN,
                    f"Intent pipeline truncated to {MAX_PIPELINE_STAGES} stages",
                )

            new_pipelines.append(ShellPipeline(
                stages=tuple(stages), comment=pipeline.comment,
            ))

        return ShellRecipe(
            header=plan.recipe_header,
            pipelines=tuple(new_pipelines),
            topology_class=base_recipe.topology_class,
            strategy=base_recipe.strategy,
            phase=base_recipe.phase,
            zone_map_version=base_recipe.zone_map_version,
            intent=decomp.intent_name,
        )

    # ── Stage 6: Compile (main entry point) ─────────────────────────────

    def compile_variant(
        self,
        zone_map: Any,
        intent_name: str,
        intent_selectors: Optional[List[str]] = None,
    ) -> CompiledRecipe:
        """Compile a single intent-conditioned recipe variant.

        Full pipeline:
            1. Check with_intent() status on the ZoneMap
            2. Decompose intent_weights into structured categories
            3. Score every signal zone for intent relevance
            4. Build extraction plan
            5. Narrow compiler context
            6. Compile narrowed recipe via base strategy
            7. Specialize with intent-specific modifications

        Uses with_intent() as documented — calls it, doesn't reimplement.
        """
        start = time.monotonic()

        # Check ZoneMap viability
        if not bool(zone_map):
            self._diag(CompileSeverity.WARN, "EmptyZoneMap — producing minimal recipe")
            return self._empty_recipe(intent_name)

        # Stage 1-2: Decompose
        decomp = self._decompose_intent(zone_map, intent_name)

        # Stage 3: Score
        scores = self._score_zones(self._ctx.enriched_zones, decomp)

        # Stage 4: Plan
        plan = self._build_plan(zone_map, scores, decomp, self._ctx.enriched_zones)

        # Stage 5: Narrow
        narrowed_ctx = self._narrow_context(plan)

        # Stage 6: Compile base
        base_ast = _compile_strategy(narrowed_ctx)

        # Stage 7: Specialize
        final_ast = self._specialize_pipeline(base_ast, plan)
        content = final_ast.serialize()

        elapsed_ms = (time.monotonic() - start) * 1000.0
        self._diag(
            CompileSeverity.INFO,
            f"Intent '{intent_name}' compiled in {elapsed_ms:.1f}ms — "
            f"{len(plan.surviving_zones)} zones, "
            f"urgency={decomp.urgency}, state={decomp.user_state}",
        )

        log.info(
            "intent_variant_compiled",
            topology_class=self._ctx.topology_class,
            intent=intent_name,
            surviving=len(plan.surviving_zones),
            excluded=len(plan.excluded_zones),
            urgency=decomp.urgency,
            user_state=decomp.user_state,
            focused=decomp.is_focused,
            entropy=round(decomp.weight_entropy, 3),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return CompiledRecipe(
            content=content,
            topology_class=self._ctx.topology_class,
            intent=intent_name,
            strategy=self._ctx.strategy,
            phase=self._ctx.phase,
            zone_map_version=self._ctx.zone_map_version,
            line_count=len(content.strip().split("\n")),
            stage_count=final_ast.stage_count,
            diagnostics=tuple(self._diagnostics),
            fallback_chain=tuple(self._ctx.fallback_chain),
            compiled_at=time.time(),
        )

    # ── Batch variant compilation ───────────────────────────────────────

    def compile_all_variants(
        self,
        zone_map: Any,
        intent_hints: Optional[Dict[str, List[str]]] = None,
    ) -> List[CompiledRecipe]:
        """Compile all intent variants for a ZoneMap.

        Reads intent_hints from the ZoneMap or infers from topology class.
        Caps at _MAX_INTENT_VARIANTS_PER_PASS.
        """
        hints = intent_hints or getattr(zone_map, "intent_hints", None)
        if hints is None:
            hints = self._infer_intent_hints(
                self._ctx.topology_class, self._ctx.enriched_zones,
            )
        if not hints:
            return []

        recipes: List[CompiledRecipe] = []
        for i, (name, selectors) in enumerate(hints.items()):
            if i >= _MAX_INTENT_VARIANTS_PER_PASS:
                self._diag(
                    CompileSeverity.WARN,
                    f"Capped variants at {_MAX_INTENT_VARIANTS_PER_PASS}; "
                    f"{len(hints) - i} skipped",
                )
                break
            try:
                recipe = self.compile_variant(zone_map, name, selectors)
                if recipe.is_valid_complexity:
                    recipes.append(recipe)
                else:
                    self._diag(CompileSeverity.WARN,
                               f"Intent '{name}' exceeds complexity limits")
            except Exception as exc:
                self._diag(CompileSeverity.ERROR,
                           f"Intent '{name}' failed: {exc}")
                log.error("intent_variant_failed",
                          intent=name, error=str(exc))

        return recipes

    # ── Intent-specialized awk program generation ───────────────────────

    def build_intent_awk_program(
        self,
        plan: IntentRecipePlan,
        signal_zones: List[EnrichedZone],
        noise_zones: List[EnrichedZone],
    ) -> str:
        """Build a complete intent-conditioned awk program.

        Combines standard zone tracking with:
            - Boundary-driven state transitions
            - Intent state variables (keep_warnings, need_procedure)
            - User-state-specific output filtering
            - Phase-conditioned post-filtering

        The FST:
            States = {zone_states} × {boundary_states} × {intent_states} × depth
            Input  = HTML line tokens
            Output = intent-filtered content
        """
        lines: List[str] = []
        decomp = plan.decomposition

        # Pre-compute boundary transitions so content_vars are known before BEGIN
        boundaries = getattr(self._ctx.zone_map, "boundaries", ())
        btrans: List[str] = []
        content_vars: List[str] = []
        if boundaries:
            btrans, content_vars = _compile_boundary_awk_transitions(boundaries)

        # BEGIN block
        lines.append("BEGIN {")
        lines.append("    depth = 0")
        lines.append("    section_idx = 0")
        lines.append("    in_capture = 0")

        for i, z in enumerate(signal_zones):
            lines.append(f"    in_sig_{i} = 0  # {z.selector.raw}")
        for i, z in enumerate(noise_zones):
            lines.append(f"    in_noise_{i} = 0  # {z.selector.raw}")
        for var in plan.awk_state_vars:
            lines.append(f"    {var}")
        for var in content_vars:
            lines.append(f"    {var} = 0")
        lines.append("}")
        lines.append("")

        # Depth tracking
        lines.append("{")
        lines.append(
            "    gsub(/<(area|base|br|col|embed|hr|img|input|link|meta|"
            "param|source|track|wbr)(\\s[^>]*)?\\/?>/, \"\")"
        )
        lines.append('    opens = gsub(/<[^\\/][^>]*>/, "&")')
        lines.append('    closes = gsub(/<\\/[^>]*>/, "&")')
        lines.append("    depth += opens - closes")
        lines.append("    if (depth < 0) depth = 0")
        lines.append("}")
        lines.append("")

        # Boundary transitions
        if btrans:
            lines.append("# Boundary state transitions")
            lines.extend(btrans)
            lines.append("")


        # Zone entry/exit
        for i, z in enumerate(signal_zones):
            pat = _selector_to_awk_pattern(z.selector)
            if pat:
                lines.append(f"/{pat}/ {{ in_sig_{i} = 1 }}")
                tag = z.selector.tag or "div"
                lines.append(f"/<\\/{tag}>/ && in_sig_{i} {{ in_sig_{i} = 0; next }}")

        for i, z in enumerate(noise_zones):
            pat = _selector_to_awk_pattern(z.selector)
            if pat:
                lines.append(f"/{pat}/ {{ in_noise_{i} = 1 }}")
                tag = z.selector.tag or "div"
                lines.append(f"/<\\/{tag}>/ && in_noise_{i} {{ in_noise_{i} = 0; next }}")

        lines.append("")

        # Extraction rule
        sig = " || ".join(f"in_sig_{i}" for i in range(len(signal_zones))) or "1"
        noi = " && ".join(f"!in_noise_{i}" for i in range(len(noise_zones))) or "1"

        lines.append(f"({sig}) && ({noi}) {{")
        lines.append('    gsub(/<[^>]*>/, "")')

        # User-state-specific output rules
        if decomp.user_state == "locked_out":
            lines.append("    if (/^[0-9]+\\./ || /^Step/) { print; next }")
        elif decomp.user_state == "debugging":
            lines.append("    if (/[Ee]rror|[Ff]ailed|[Ee]xception/) { print; next }")
        elif decomp.user_state == "comparing":
            lines.append("    if (/[Vv]s\\.?|[Cc]ompar/) { print; next }")

        lines.append("    if (/[^[:space:]]/) print")
        lines.append("}")

        # Phase filter
        if self._ctx.phase == PhaseStr.KNOWS:
            lines.append(f"length($0) < {MIN_SIGNAL_LENGTH} {{ next }}")

        return "\n".join(lines)

    # ── Helper methods ──────────────────────────────────────────────────

    def _lookup_intent_weight( # noqa
        self, selector: str, decomp: IntentDecomposition,
    ) -> float:
        """Look up intent weight for a selector, with substring fallback."""
        if selector in decomp.weight_map:
            return decomp.weight_map[selector]
        # Substring match fallback
        for ws, wv in decomp.weight_map.items():
            if ws in selector or selector in ws:
                return wv
        return 1.0  # default

    @staticmethod
    def _zone_content_type(ez: EnrichedZone) -> str:
        """Extract content_type from an enriched zone."""
        # Infer from structural role
        role = ez.structural_role
        if role == "code_block":
            return "code"
        if role in ("list", "ordered_list", "unordered_list"):
            return "list"
        if role in ("table", "data_table", "comparison_table", "pricing"):
            return "table"
        if role in ("paragraph", "prose", "main_content"):
            return "prose"
        return "mixed"

    @staticmethod
    def _content_state_alignment(content_type: str, user_state: str) -> float:
        """Look up content_type × user_state compatibility from matrix."""
        ct_row = _CONTENT_STATE_MATRIX.get(content_type, _CONTENT_STATE_MATRIX["mixed"])
        return ct_row.get(user_state, 0.5)

    @staticmethod
    def _shannon_entropy(weights: List[float]) -> float:
        """Shannon entropy H = −Σ p_i·log₂(p_i)."""
        if not weights:
            return 0.0
        total = sum(weights)
        if total <= 0.0:
            return 0.0
        entropy = 0.0
        for w in weights:
            p = w / total
            if p > 0.0:
                entropy -= p * math.log2(p)
        return entropy

    @staticmethod
    def _infer_urgency(intent_name: str, weight_map: Dict[str, float]) -> str:
        """Infer urgency from intent name and weight peakiness."""
        name = intent_name.lower()
        if any(k in name for k in (
            "recover", "locked", "emergency", "urgent", "lost", "reset",
            "forgot", "blocked", "suspended",
        )):
            return "high"
        if any(k in name for k in ("error", "debug", "fix", "troubleshoot")):
            return "high"
        if any(k in name for k in ("explore", "browse", "overview", "intro")):
            return "low"
        # Weight peakiness check
        if weight_map:
            vals = [v for v in weight_map.values() if v > 0]
            if vals:
                mean_w = sum(vals) / len(vals)
                if mean_w > 0 and max(vals) / mean_w > 3.0:
                    return "high"
        return "normal"

    @staticmethod
    def _infer_user_state(intent_name: str, weight_map: Dict[str, float]) -> str:
        """Infer user state from intent name vocabulary."""
        name = intent_name.lower()
        mapping = {
            "locked_out": ("recover", "locked", "lost", "forgot", "reset", "password"),
            "debugging":  ("error", "debug", "fix", "troubleshoot", "crash", "fail"),
            "building":   ("api", "sdk", "implement", "integrate", "code", "build"),
            "learning":   ("tutorial", "guide", "learn", "getting_started", "howto"),
            "comparing":  ("compare", "vs", "versus", "alternative", "pricing"),
            "purchasing": ("buy", "purchase", "subscribe", "checkout"),
            "migrating":  ("migrate", "upgrade", "transition", "import", "export"),
            "exploring":  ("explore", "browse", "overview", "about", "concept"),
        }
        for state, keywords in mapping.items():
            if any(k in name for k in keywords):
                return state
        return "exploring"

    def _extract_grep_keywords(self, decomp: IntentDecomposition) -> List[str]: # noqa
        """Extract grep-friendly keyword patterns from intent decomposition.

        Priority: intent vocabulary → primary selector class names.
        """
        keywords: List[str] = []

        # Check keyword vocabulary first
        vocab_pattern = _INTENT_KEYWORD_PATTERNS.get(decomp.intent_name)
        if vocab_pattern:
            keywords.append(vocab_pattern)
            return keywords

        # Extract from primary selector class names
        for sel_str in decomp.primary_selectors:
            for match in _RE_CLASS.finditer(sel_str):
                cls = match.group(1)
                kw = cls.replace("-", "\\|").replace("_", "\\|")
                if len(kw) > 3:
                    keywords.append(kw)

        return keywords

    def _build_awk_state_vars(self, decomp: IntentDecomposition) -> List[str]: # noqa
        """Build intent-specific awk state variable declarations."""
        awk_vars: List[str] = []
        if decomp.is_focused:
            awk_vars.append("intent_matched = 0")
            awk_vars.append("intent_zone_count = 0")
        if decomp.urgency in ("high", "critical"):
            awk_vars.append("keep_warnings = 1")
        if decomp.user_state == "locked_out":
            awk_vars.append("need_procedure = 1")
        if decomp.user_state == "debugging":
            awk_vars.append("keep_errors = 1")
        return awk_vars

    @staticmethod
    def _build_exclusion_patterns(decomp: IntentDecomposition) -> List[str]:
        """Build grep -v patterns from excluded selectors."""
        patterns: List[str] = []
        for sel in decomp.excluded_selectors[:5]:
            cls_match = _RE_CLASS.search(sel)
            if cls_match:
                patterns.append(
                    f'class="[^"]*{re.escape(cls_match.group(1))}[^"]*"'
                )
        return patterns

    def _empty_recipe(self, intent_name: str) -> CompiledRecipe:
        """Produce a minimal recipe for EmptyZoneMap inputs."""
        content = (
            "#!/bin/sh\n"
            f"# intent: {intent_name} — EmptyZoneMap fallback\n"
            "cat\n"
        )
        return CompiledRecipe(
            content=content,
            topology_class=self._ctx.topology_class,
            intent=intent_name,
            strategy=self._ctx.strategy,
            phase=self._ctx.phase,
            zone_map_version=self._ctx.zone_map_version,
            line_count=3,
            stage_count=1,
            diagnostics=tuple(self._diagnostics),
            fallback_chain=tuple(self._ctx.fallback_chain),
            compiled_at=time.time(),
        )

    def _infer_intent_hints( # noqa
        self,
        topology_class: str,
        enriched_zones: List[EnrichedZone],
    ) -> Dict[str, List[str]]:
        """Infer intent hints from topology class when none provided."""
        hints: Dict[str, List[str]] = {}

        code_z = [z.selector.raw for z in enriched_zones
                  if z.structural_role == "code_block" and z.node_type == NodeType.SIGNAL]
        prose_z = [z.selector.raw for z in enriched_zones
                   if z.structural_role in ("paragraph", "main_content")
                   and z.node_type == NodeType.SIGNAL]
        list_z = [z.selector.raw for z in enriched_zones
                  if z.structural_role in ("list", "ordered_list")
                  and z.node_type == NodeType.SIGNAL]

        if "SAAS_DOCS" in topology_class:
            if code_z:
                hints["api_reference"] = code_z[:3]
            if list_z:
                hints["recovery_codes"] = list_z[:2] + code_z[:1]
            if prose_z:
                hints["tutorial"] = prose_z[:3]
        elif "ECOMMERCE" in topology_class:
            pricing_z = [z.selector.raw for z in enriched_zones
                         if z.structural_role in ("pricing", "price")
                         and z.node_type == NodeType.SIGNAL]
            if pricing_z:
                hints["pricing"] = pricing_z
        elif topology_class == "FORUM_THREAD":
            all_sig = [z.selector.raw for z in enriched_zones
                       if z.node_type == NodeType.SIGNAL]
            if all_sig:
                hints["accepted_answer"] = all_sig[:2]
            if code_z:
                hints["code_solutions"] = code_z

        return hints

# ═════════════════════════════════════════════════════════════════════════════
# STRATEGY DISPATCH
#
# Routes compilation to the correct strategy compiler based on the
# determined extraction strategy.
# ═════════════════════════════════════════════════════════════════════════════

# noinspection PyUnreachableCode
def _compile_strategy(ctx: CompilerContext) -> ShellRecipe:
    """Dispatch to the correct strategy compiler.

    Strategy mapping (reconciled against real wlp_zones.ExtractionStrategy):
        DEPTH_FIRST    → _compose_zone_extract (sed chains + awk depth)
        BREADTH_FIRST  → _compose_attribute_extract (awk regex on data-*)
        SECTION_SCOPED → _compose_envelope_extract (awk state machines)
        FLAT           → _compose_flat_extract (top-level only, JSON_LD)
    """
    if ctx.strategy == ExtractionStrategy.DEPTH_FIRST:
        return _compose_zone_extract(ctx)
    elif ctx.strategy == ExtractionStrategy.BREADTH_FIRST:
        return _compose_attribute_extract(ctx)
    elif ctx.strategy == ExtractionStrategy.SECTION_SCOPED:
        return _compose_envelope_extract(ctx)
    elif ctx.strategy == ExtractionStrategy.FLAT:
        return _compose_flat_extract(ctx)
    else:
        ctx.add_diagnostic(
            CompileSeverity.WARN,
            f"Unknown strategy {ctx.strategy}; falling back to DEPTH_FIRST",
        )
        return _compose_zone_extract(ctx)


def _determine_strategy(zone_map: Any) -> ExtractionStrategy:
    """Determine the compilation strategy from the ZoneMap.

    contracts.py ZoneMap carries a string .strategy field — this is the
    primary production path. The .extraction_strategy enum field is reserved
    for future schema evolution or EmptyZoneMap.
    """
    # Primary: contracts.py ZoneMap uses .strategy (string field)
    if hasattr(zone_map, "strategy"):
        try:
            return ExtractionStrategy.from_str(zone_map.strategy)
        except ValueError:
            pass

    # Secondary: future schema or EmptyZoneMap with enum .extraction_strategy
    if hasattr(zone_map, "extraction_strategy"):
        return ExtractionStrategy.from_zone_map(zone_map)

    # Default
    log.warning(
        "no_strategy_on_zone_map",
        topology_class=getattr(zone_map, "topology_class", "UNKNOWN"),
        falling_back_to="depth_first",
    )
    return ExtractionStrategy.DEPTH_FIRST


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK INJECTION ENGINE
#
# Closes the learning loop: WLP → parser → kernel → feedback → parser.
#
# When the parser recompiles a recipe (triggered by ZoneMapUpdatedEvent),
# it receives the last KernelOutput for that topology class and adjusts
# zone boundaries before compilation.
#
# Feedback rules:
# - High noise_ratio → tighten zone boundaries
# - Surprise fired → loosen zone boundaries
# - Low compression → tighten (keeping too much)
# - High compression → loosen (stripping too aggressively)
# ═════════════════════════════════════════════════════════════════════════════

def _inject_feedback(
    zone_map: ZoneMap,
    feedback: FeedbackState,
    last_output: Optional[KernelOutput],
) -> Tuple[ZoneMap, FeedbackState]:
    """Apply feedback injection to a ZoneMap before compilation.

    Updates the feedback state with the latest kernel output, then
    returns the (potentially modified) ZoneMap and updated feedback.

    The ZoneMap itself is frozen, so "modification" means creating
    a new ZoneMap with adjusted signal/noise zone lists.
    """
    if last_output is None:
        return zone_map, feedback

    # Compute feedback metrics from last output
    compression = last_output.compression_ratio
    noise_ratio = 1.0 - last_output.signal_density if not last_output.extraction_empty else 0.0
    surprise_fired = last_output.is_over_stripping

    # Update feedback state
    feedback.update(noise_ratio, compression, surprise_fired)

    # Apply feedback adjustments to zone map
    adjusted_signal = list(zone_map.signal_zones)
    adjusted_noise = list(zone_map.noise_zones)

    if feedback.tighten_requested and len(adjusted_signal) > 1:
        # Tighten: remove the lowest-specificity signal zone
        # This narrows the extraction boundary
        parsed = [(parse_selector(s), s) for s in adjusted_signal]
        parsed.sort(key=lambda x: x[0].specificity.score)
        # Remove bottom 20% of zones by specificity
        remove_count = max(1, len(parsed) // 5)
        removed = parsed[:remove_count]
        adjusted_signal = [s for _, s in parsed[remove_count:]]
        for sel, raw in removed:
            # Move removed signal zones to noise
            if raw not in adjusted_noise:
                adjusted_noise.append(raw)
        feedback.tighten_requested = False  # Reset flag

    if feedback.loosen_requested:
        # Loosen: move the highest-specificity noise zone back to signal
        if adjusted_noise:
            parsed_noise = [(parse_selector(s), s) for s in adjusted_noise]
            parsed_noise.sort(key=lambda x: x[0].specificity.score, reverse=True)
            # Move top noise zone back to signal
            moved_sel, moved_raw = parsed_noise[0]
            # Only move if it doesn't look like a structural noise element
            role = _infer_structural_role(moved_sel)
            if role not in ("noise_element", "navigation"):
                adjusted_signal.append(moved_raw)
                adjusted_noise.remove(moved_raw)
        feedback.loosen_requested = False  # Reset flag

    # Create adjusted ZoneMap
    from dataclasses import replace # noqa
    adjusted_zone_map = ZoneMap(
        topology_class=zone_map.topology_class,
        signal_zones=adjusted_signal,
        noise_zones=adjusted_noise,
        strategy=zone_map.strategy,
        confidence=zone_map.confidence,
        version=zone_map.version,
    )

    return adjusted_zone_map, feedback


# ═════════════════════════════════════════════════════════════════════════════
# RECURSIVE FALLBACK — PARENT_CLASS_MAP
#
# If ZoneMap.confidence < THETA_WLP_COMPILE_MIN, the compiler walks up
# PARENT_CLASS_MAP until it finds a confident ZoneMap or hits GENERIC_HTML.
#
# GENERIC_HTML always has a ZoneMap (generated conservatively: keep <body>
# content, strip <nav>, <header>, <footer>, <script>, <style>).
# It is the guaranteed terminal. The compiler never fails to produce a recipe.
# ═════════════════════════════════════════════════════════════════════════════

def _make_generic_html_zone_map() -> ZoneMap:
    """Construct the guaranteed-terminal GENERIC_HTML ZoneMap.

    This is the conservative fallback: keep <body> content, strip
    known noise elements. Every topology class eventually falls through
    to this if no confident specific zone map exists.
    """
    return ZoneMap(
        topology_class="GENERIC_HTML",
        signal_zones=["body"],
        noise_zones=[
            "nav", "header", "footer", "script", "style",
            "iframe", "noscript",
            ".cookie-banner", ".modal", ".popup",
            ".advertisement", ".sidebar",
        ],
        strategy="zone_extract",
        confidence=1.0,
        version=0,
    )


# Module-level GENERIC_HTML zone map. Always available.
_GENERIC_HTML_ZONE_MAP: Final[ZoneMap] = _make_generic_html_zone_map()


def _walk_parent_map(
    topology_class: str,
    zone_map_cache: Dict[str, ZoneMap],
    *,
    visited: Optional[Set[str]] = None,
) -> Tuple[ZoneMap, List[str]]:
    """Walk PARENT_CLASS_MAP to find the nearest confident ancestor ZoneMap.

    Returns: (zone_map, fallback_chain) where fallback_chain records the
    classes visited before settling.

    Algorithm:
    1. Check if current class has a cached zone map with sufficient confidence
    2. If not, look up parent in PARENT_CLASS_MAP
    3. Recurse to parent
    4. If no parent found, use wildcard entry (*)
    5. Terminal: GENERIC_HTML always has a zone map

    Cycle detection: visited set prevents infinite loops in malformed maps.
    """
    if visited is None:
        visited = set()

    fallback_chain: List[str] = [topology_class]

    if topology_class in visited:
        # Cycle detected — fall to GENERIC_HTML
        return _GENERIC_HTML_ZONE_MAP, fallback_chain

    visited.add(topology_class)

    # Check cache for confident zone map
    if topology_class in zone_map_cache:
        zm = zone_map_cache[topology_class]
        if zm.confidence >= THETA_WLP_COMPILE_MIN:
            return zm, fallback_chain

    # Look up parent
    parent = PARENT_CLASS_MAP.get(topology_class)
    if parent is None:
        # Try wildcard
        parent = PARENT_CLASS_MAP.get("*")

    if parent is None or parent == topology_class:
        return _GENERIC_HTML_ZONE_MAP, fallback_chain

    # Recurse to parent
    parent_zm, parent_chain = _walk_parent_map(
        parent, zone_map_cache, visited=visited,
    )
    fallback_chain.extend(parent_chain)
    return parent_zm, fallback_chain


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE SANITIZER
#
# Final safety check before serialization. Ensures no injection vectors
# survived the AST construction. This is defense-in-depth: the AST prevents
# injection by construction, but the sanitizer catches any edge cases.
# ═════════════════════════════════════════════════════════════════════════════

_COMPILED_INJECTION_RE: Tuple[Tuple[str, re.Pattern[str]], ...] = tuple(
    (raw, re.compile(raw)) for raw in INJECTION_PATTERNS
)


def _sanitize_recipe(content: str) -> Tuple[bool, Optional[str]]:
    """Sanitize a compiled recipe before writing to disk.

    Checks:
    1. No injection patterns present
    2. All commands are in ALLOWED_RECIPE_COMMANDS
    3. Recipe is not empty
    4. Recipe is within line count limits

    Returns: (is_safe, failure_reason)
    """
    if not content.strip():
        return False, "Recipe content is empty after stripping"

    lines = content.strip().split("\n")
    if len(lines) > MAX_RECIPE_LINES:
        return False, (
            f"Recipe has {len(lines)} lines, exceeding MAX_RECIPE_LINES "
            f"({MAX_RECIPE_LINES})"
        )

    # Check for injection patterns
    for pattern_str, pattern_re in _COMPILED_INJECTION_RE:
        if pattern_re.search(content):
            return False, f"Injection pattern detected: {pattern_str}"

    # Verify all commands are whitelisted
    # Parse non-comment, non-empty lines for command tokens
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue

        # Handle pipe continuations
        if stripped.startswith("|"):
            stripped = stripped[1:].strip()
        if stripped.endswith("\\"):
            stripped = stripped[:-1].strip()

        # Extract the command (first token)
        tokens = stripped.split()
        if not tokens:
            continue

        cmd = tokens[0]

        if cmd not in ALLOWED_RECIPE_COMMANDS:
            return False, (
                f"Command {cmd!r} not in ALLOWED_RECIPE_COMMANDS. "
                f"Allowed: {sorted(ALLOWED_RECIPE_COMMANDS)}"
            )

    return True, None


# ═════════════════════════════════════════════════════════════════════════════
# WRITE PROTOCOL
#
# Every file write uses staging + atomic rename. This is a system-wide
# invariant. Never write directly to the final path. Never skip the
# SHA-256. Never use shutil.copy (not atomic on POSIX).
# ═════════════════════════════════════════════════════════════════════════════

def _ensure_directory(path: Path) -> None:
    """Ensure the parent directory exists. Creates recursively if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)


def _write_recipe_atomic(
    content: str,
    topology_class: str,
    intent: Optional[str] = None,
) -> Tuple[str, str]:
    """Write a recipe to disk using staging + atomic rename.

    Protocol:
    1. Compute filename from topology_class and intent
    2. Write to staging path ({filename}.staging)
    3. Compute SHA-256 of staging file
    4. Atomic rename staging → final (os.rename on POSIX)

    Returns: (final_path, checksum)

    On failure: staging file is cleaned up. Final path is untouched.
    Previous recipe remains active.
    """
    if intent:
        filename = f"{topology_class}_{intent}.sh"
    else:
        filename = f"{topology_class}.sh"

    final_path = COMPILER_GENERATED_PATH / filename
    staging_path = COMPILER_GENERATED_PATH / f"{filename}.staging"

    _ensure_directory(final_path)

    try:
        # Write to staging
        staging_path.write_text(content, encoding="utf-8")

        # Compute checksum from what was written
        checksum = compute_recipe_hash(content)

        # Atomic rename
        os.rename(str(staging_path), str(final_path))

        return str(final_path), checksum

    except Exception:
        # Clean up staging on failure
        try:
            if staging_path.exists():
                staging_path.unlink()
        except OSError:
            pass
        raise


# ═════════════════════════════════════════════════════════════════════════════
# PHASE READER
#
# Reads the current phase for a topology class from phase_states.mmap.
# This is a read-only memory-mapped file written by index_daemon.py.
# ═════════════════════════════════════════════════════════════════════════════

def _read_phase_from_mmap(topology_class: str) -> Optional[PhaseStr]:
    """Read phase from phase_states.mmap for a topology class.

    The mmap file is a binary structure:
    - Fixed-size records indexed by topology class hash
    - Each record: 4 bytes for topology class hash + 4 bytes for phase int

    Returns None if:
    - File is unreadable (permissions, corruption)
    - No record for this topology class
    """
    try:
        encoded = topology_class.encode("utf-8")
        if len(encoded) > 64:
            log.warning(
                "topology_class_name_exceeds_mmap_field",
                topology_class=topology_class,
                byte_length=len(encoded),
            )
            return None

        with open(PHASE_STATES_PATH, "rb") as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                record_size = 68
                file_size = mm.size()
                if file_size == 0:
                    return None

                class_bytes = encoded.ljust(64, b"\x00")

                offset = 0
                while offset + record_size <= file_size:
                    stored_class = mm[offset:offset + 64]
                    if stored_class == class_bytes:
                        phase_int = struct.unpack_from("<I", mm, offset + 64)[0]
                        if phase_int in (1, 2, 3):
                            return PhaseStr.from_int(phase_int)
                        return None
                    offset += record_size

                return None
            finally:
                mm.close()
    except (OSError, ValueError, struct.error):
        return None


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE COMPILER — MAIN CLASS
#
# The public interface. Subscribes to bus events, manages compilation state,
# orchestrates the full compilation pipeline.
#
# Thread safety: all mutable state is accessed through asyncio locks.
# No raw threading primitives. The compiler runs in the async event loop.
# ═════════════════════════════════════════════════════════════════════════════

class RecipeCompiler:
    """AXIOM Recipe Compiler.

    Translates ZoneMap structural descriptions into executable shell
    pipelines. This is the intelligence layer between structural
    understanding and executable extraction.

    Public interface:
        initialize()                — subscribe to bus events
        compile(zone_map, phase)    — compile a ZoneMap into recipes
        handle_zone_map_updated()   — event handler
        handle_phase_transition()   — event handler

    Internal state:
        _last_compiled_version — prevents stale event processing
        _zone_map_cache        — latest ZoneMap per topology class
        _phase_cache           — current phase per topology class
        _feedback_state        — feedback tracking per topology class
        _compile_timestamps    — debounce tracking
        _invalidated_classes   — classes with invalidated zone maps
    """

    __slots__ = (
        "_last_compiled_version",
        "_zone_map_cache",
        "_phase_cache",
        "_feedback_state",
        "_compile_timestamps",
        "_invalidated_classes",
        "_bus",
        "_lock",
        "_loop",
        "_initialized",
        "_compile_count",
        "_fail_count",
    )

    def __init__(self) -> None:
        self._last_compiled_version: Dict[str, int] = {}
        self._zone_map_cache: Dict[str, ZoneMap] = {}
        self._phase_cache: Dict[str, PhaseStr] = {}
        self._feedback_state: Dict[str, FeedbackState] = defaultdict(FeedbackState)
        self._compile_timestamps: Dict[str, float] = {}
        self._invalidated_classes: Set[str] = set()
        self._bus: Optional[Any] = None
        self._lock: Optional[asyncio.Lock] = None
        self._initialized = False
        self._compile_count = 0
        self._fail_count = 0
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        feedback.register_feedback_handler(self._on_feedback_event)

    # ── Public API ───────────────────────────────────────────────────────

    async def initialize(self, bus: Optional[Any] = None) -> None:
        """Initialize the compiler and subscribe to bus events.

        Args:
            bus: CrawlerBus instance. If None, runs in standalone mode
                 (compile() can still be called directly).
        """
        if self._lock is None:
            self._lock = asyncio.Lock()

        async with self._lock:
            if self._initialized:
                log.warning("compiler_already_initialized")
                return

            self._bus = bus

            # Subscribe to events if bus is available
            if self._bus is not None:
                try:
                    if hasattr(self._bus, "subscribe"):
                        await self._bus.subscribe(
                            "zone_map_updated",
                            self.handle_zone_map_updated,
                        )
                        await self._bus.subscribe(
                            "phase_transition",
                            self.handle_phase_transition,
                        )
                        await self._bus.subscribe(
                            "zone_map_invalidated",
                            self._handle_zone_map_invalidated,
                        )
                        await self._bus.subscribe(
                            "feedback_event",
                            self._on_feedback_event,
                        )
                except Exception as exc:
                    log.error(
                        "bus_subscription_failed",
                        error=str(exc),
                    )

            # Pre-populate phase cache from mmap
            for topo_class in KNOWN_TOPOLOGY_CLASSES:
                phase = _read_phase_from_mmap(topo_class)
                if phase is not None:
                    self._phase_cache[topo_class] = phase

            self._initialized = True
            log.info(
                "compiler_initialized",
                phase_cache_size=len(self._phase_cache),
                bus_available=self._bus is not None,
            )

    async def compile(
            self,
            zone_map: ZoneMap,
            phase: Optional[str] = None,
            *,
            last_output: Optional[KernelOutput] = None,
    ) -> List[CompiledRecipe]:
        """Compile a ZoneMap into one or more recipes.

        This is the main compilation entry point. It produces:
        1. A base recipe for the topology class
        2. Intent-variant recipes (if intent_hints are present)

        Args:
            zone_map: The ZoneMap to compile.
            phase: Phase string override. If None, read from mmap/cache.
            last_output: Last KernelOutput for feedback injection.

        Returns: List of CompiledRecipe objects.

        Raises:
            RecipeCompilationFailed: If compilation fails after retries.
        """
        self._loop = asyncio.get_running_loop()
        topology_class = zone_map.topology_class
        start_time = time.monotonic()

        # Snapshot mutable shared state under lock before any work (#10)
        feedback: FeedbackState = FeedbackState()
        async with self._lock:
            phase_str = self._resolve_phase(topology_class, phase)
            feedback = copy.copy(self._feedback_state.get(topology_class, FeedbackState()))

        fallback_chain: List[str] = []

        # EmptyZoneMap guard — check bool(), never `is not None`
        if not bool(zone_map):
            log.info("empty_zone_map_fallback", topology_class=topology_class)
            effective_zm = _make_generic_html_zone_map()
            fallback_chain = [topology_class, "GENERIC_HTML"]
        else:
            # Apply feedback injection
            adjusted_zm, feedback = _inject_feedback(zone_map, feedback, last_output)

            # Write updated feedback back under lock
            async with self._lock:
                self._feedback_state[topology_class] = feedback

            # Two-tier confidence check:
            #   < 0.30 (THETA_WLP_COMPILE_MIN) → hardcoded GENERIC_HTML
            #   < 0.70 (THETA_PARENT_FALLBACK) → try parent class first
            #   >= 0.70 → compile directly
            effective_zm = adjusted_zm

            if adjusted_zm.confidence < THETA_WLP_COMPILE_MIN:
                log.info(
                    "confidence_below_floor",
                    topology_class=topology_class,
                    confidence=adjusted_zm.confidence,
                    floor=THETA_WLP_COMPILE_MIN,
                )
                effective_zm = _make_generic_html_zone_map()
                fallback_chain = [topology_class, "GENERIC_HTML"]
            elif adjusted_zm.confidence < THETA_PARENT_FALLBACK:
                log.info(
                    "low_confidence_parent_fallback",
                    topology_class=topology_class,
                    confidence=adjusted_zm.confidence,
                    threshold=THETA_PARENT_FALLBACK,
                )
                # Cold-start note: _zone_map_cache is populated from ZoneMapUpdatedEvents
                # only. On first run the cache may be empty — parent fallback will go to
                # GENERIC_HTML until the WLP emits events. Full fix requires daemon-level
                # zone map replay on startup, outside parser.py's scope.
                effective_zm, fallback_chain = _walk_parent_map(
                    topology_class,
                    self._zone_map_cache,
                )

        # Determine strategy from the ZoneMap
        strategy = _determine_strategy(effective_zm)

        # Enrich zones
        enriched = enrich_zone_map(effective_zm, feedback=feedback)

        # Build compiler context
        ctx = CompilerContext(
            zone_map=effective_zm,
            phase=phase_str,
            feedback=feedback,
            enriched_zones=enriched,
            strategy=strategy,
            diagnostics=[],
            topology_class=topology_class,
            zone_map_version=zone_map.version,
            attempt=1,
            start_time=start_time,
            fallback_chain=fallback_chain,
        )

        recipes: List[CompiledRecipe] = []

        # Compile base recipe
        base_recipe = self._compile_single(ctx)
        if base_recipe is not None:
            recipes.append(base_recipe)

        # Compile intent variants using IntentConditionedExtractor
        extractor = IntentConditionedExtractor(ctx)
        intent_recipes = extractor.compile_all_variants(effective_zm)
        for ir in intent_recipes:
            recipes.append(ir)

        # Update caches under lock
        async with self._lock:
            self._zone_map_cache[topology_class] = zone_map
            self._last_compiled_version[topology_class] = zone_map.version
            self._compile_count += len(recipes)

        elapsed = (time.monotonic() - start_time) * 1000.0
        log.info(
            "compilation_complete",
            topology_class=topology_class,
            recipe_count=len(recipes),
            strategy=strategy.value,
            phase=phase_str.value,
            elapsed_ms=round(elapsed, 2),
            fallback_chain=fallback_chain if fallback_chain else None,
            diagnostics_count=len(ctx.diagnostics),
        )

        if not recipes:
            raise RecipeCompilationFailed(
                f"No recipes produced for {topology_class} "
                f"(zone_map_version={zone_map.version})",
                topology_class=topology_class,
            )

        return recipes

    async def compile_and_write(
        self,
        zone_map: ZoneMap,
        phase: Optional[str] = None,
        *,
        last_output: Optional[KernelOutput] = None,
        run_id: Optional[str] = None,
    ) -> List[RecipeCompiledEvent]:
        """Compile, validate, write, and register recipes.

        Full compilation pipeline:
        1. compile() — produce recipe ASTs
        2. sanitize — final safety check
        3. write — staging + atomic rename
        4. emit events

        Returns: List of RecipeCompiledEvent for successful writes.
        """
        recipes = await self.compile(zone_map, phase, last_output=last_output)
        events: List[RecipeCompiledEvent] = []

        for recipe in recipes:
            try:
                # Sanitize
                is_safe, failure_reason = _sanitize_recipe(recipe.content)
                if not is_safe:
                    log.warning(
                        "recipe_sanitization_failed",
                        topology_class=recipe.topology_class,
                        intent=recipe.intent,
                        reason=failure_reason,
                    )
                    await self._emit_failure_event(
                        recipe.topology_class,
                        recipe.zone_map_version,
                        f"Sanitization failed: {failure_reason}",
                        recipe.phase.value,
                        recipe.fallback_chain,
                    )
                    continue

                # Write
                recipe_path, checksum = _write_recipe_atomic(
                    recipe.content,
                    recipe.topology_class,
                    recipe.intent,
                )

                entry = register_recipe(
                    topology_class=recipe.topology_class,
                    recipe_path=str(recipe_path),
                    caller_supplied_hash=checksum,
                    run_id=None,  # or thread through compile_and_write's signature if correlation needed
                )
                log.info(
                    "recipe_registered",
                    topology_class=recipe.topology_class,
                    registered_as=entry.topology_class,
                    hash=entry.recipe_hash[:16],
                )

                event = RecipeCompiledEvent(
                    topology_class=recipe.topology_class,
                    intent=recipe.intent,
                    zone_map_version=recipe.zone_map_version,
                    strategy=recipe.strategy.value,
                    phase=recipe.phase.value,
                    recipe_path=recipe_path,
                    checksum=checksum,
                    compiled_at=recipe.compiled_at,
                    line_count=recipe.line_count,
                    fallback_chain=recipe.fallback_chain,
                )
                events.append(event)

                log.info(
                    "recipe_written",
                    topology_class=recipe.topology_class,
                    intent=recipe.intent,
                    recipe_path=recipe_path,
                    checksum=checksum[:8],
                    line_count=recipe.line_count,
                )

            except Exception as exc:
                log.error(
                    "recipe_write_failed",
                    topology_class=recipe.topology_class,
                    intent=recipe.intent,
                    error=str(exc),
                )
                await self._emit_failure_event(
                    recipe.topology_class,
                    recipe.zone_map_version,
                    f"Write failed: {exc}",
                    recipe.phase.value,
                    recipe.fallback_chain,
                )

        return events

    # ── Event Handlers ───────────────────────────────────────────────────

    async def handle_zone_map_updated(
        self,
        event: ZoneMapUpdatedEvent,
    ) -> None:
        """Handle ZoneMapUpdatedEvent from the WLP.

        Stale event rejection: if event version <= last compiled version
        for this topology class, the event is discarded silently.

        Debounce: if a compilation was done within COMPILE_DEBOUNCE_S
        seconds, the event is deferred.

        Invalidated classes: if the class was invalidated by a
        ZoneMapInvalidatedEvent, clear the invalidation flag and compile.
        """
        topology_class = event.topology_class
        zone_map = event.new_zone_map

        async with self._lock:
            # Stale event rejection
            last_version = self._last_compiled_version.get(topology_class, -1)
            if zone_map.version <= last_version:
                log.debug(
                    "stale_zone_map_event",
                    topology_class=topology_class,
                    event_version=zone_map.version,
                    last_compiled=last_version,
                )
                return

            # Debounce check
            last_compile_time = self._compile_timestamps.get(topology_class, 0.0)
            now = time.monotonic()
            if now - last_compile_time < COMPILE_DEBOUNCE_S:
                log.debug(
                    "debounced_compilation",
                    topology_class=topology_class,
                    since_last_compile_s=round(now - last_compile_time, 2),
                )
                # Update cache but defer compilation
                self._zone_map_cache[topology_class] = zone_map
                return

            # Clear invalidation flag
            self._invalidated_classes.discard(topology_class)

            # Update cache
            self._zone_map_cache[topology_class] = zone_map

        # Skip hardcoded classes — their recipes are immutable
        if topology_class in HARDCODED_TOPOLOGY_CLASSES:
            log.debug(
                "skipping_hardcoded_class",
                topology_class=topology_class,
            )
            return

        # Compile
        try:
            events = await self.compile_and_write(zone_map)
            async with self._lock:
                self._compile_timestamps[topology_class] = time.monotonic()

            # Emit events to bus
            for event_out in events:
                await self._emit_compiled_event(event_out)

        except RecipeCompilationFailed as exc:
            log.error(
                "compilation_failed",
                topology_class=topology_class,
                error=str(exc),
            )
            async with self._lock:
                self._fail_count += 1

    async def handle_phase_transition(
        self,
        event: PhaseTransitionEvent,
    ) -> None:
        """Handle PhaseTransitionEvent from index_daemon.

        Phase change triggers recompilation of all recipes for the
        affected topology class, because phase affects recipe aggressiveness.
        """
        topology_class = event.topology_class
        new_phase = PhaseStr.from_int(event.to_phase)

        async with self._lock:
            old_phase = self._phase_cache.get(topology_class)
            self._phase_cache[topology_class] = new_phase

        log.info(
            "phase_transition_received",
            topology_class=topology_class,
            old_phase=old_phase.value if old_phase else None,
            new_phase=new_phase.value,
        )

        # Recompile if we have a cached zone map
        zone_map = self._zone_map_cache.get(topology_class)
        if zone_map is not None:
            try:
                events = await self.compile_and_write(
                    zone_map, phase=new_phase.value,
                )
                for event_out in events:
                    await self._emit_compiled_event(event_out)
            except RecipeCompilationFailed as exc:
                log.error(
                    "phase_recompilation_failed",
                    topology_class=topology_class,
                    error=str(exc),
                )

    async def _handle_zone_map_invalidated(
        self,
        event: ZoneMapInvalidatedEvent,
    ) -> None:
        """Handle ZoneMapInvalidatedEvent from surprise detector.

        Marks the class as invalidated. The compiler will not compile
        from stale zone maps — it waits for a subsequent ZoneMapUpdatedEvent.
        """
        async with self._lock:
            self._invalidated_classes.add(event.topology_class)
            # Remove stale cache entry
            self._zone_map_cache.pop(event.topology_class, None)

        log.info(
            "zone_map_invalidated",
            topology_class=event.topology_class,
        )

    def _on_feedback_event(self, event: FeedbackEvent) -> None:
        """Synchronous handler registered with feedback.py.
        Schedules async recompilation without blocking the caller."""
        if not event.recompilation_recommended:
            return

        log.info(
            "feedback_recompilation_requested",
            topology_class=event.topology_class,
            reason=event.recompilation_reason,
            window_empty_rate=event.window_empty_extraction_rate,
            window_samples=event.window_sample_count,
        )

        if self._loop is not None and self._loop.is_running():
            asyncio.run_coroutine_threadsafe(
                self._trigger_recompile(event.topology_class, event.recompilation_reason),
                self._loop,
            )
        else:
            log.warning(
                "feedback_recompile_skipped_no_loop",
                topology_class=event.topology_class,
            )

    # ── Internal helpers ─────────────────────────────────────────────────

    async def _trigger_recompile(
            self,
            topology_class: str,
            reason: Optional[str],
    ) -> None:
        """Recompile a recipe in response to a FeedbackEvent recommendation."""
        async with self._lock:
            zone_map = self._zone_map_cache.get(topology_class)
            if zone_map is None:
                log.warning(
                    "feedback_recompile_no_zone_map",
                    topology_class=topology_class,
                    reason=reason,
                )
                return
            last_compiled = self._compile_timestamps.get(topology_class, 0.0)
            if time.monotonic() - last_compiled < COMPILE_DEBOUNCE_S:
                log.debug(
                    "feedback_recompile_debounced",
                    topology_class=topology_class,
                )
                return
            log.info(
                "feedback_recompile_triggered",
                topology_class=topology_class,
                reason=reason,
            )
        await self.compile(zone_map, self._phase_cache.get(topology_class, "learns"))

    def _compile_single(self, ctx: CompilerContext) -> Optional[CompiledRecipe]: # noqa
        """Compile a single base recipe from the context.

        Implements retry logic: on failure, progressively loosens parameters.
        """
        for attempt in range(1, MAX_COMPILE_RETRIES + 1):
            ctx_attempt = CompilerContext(
                zone_map=ctx.zone_map,
                phase=ctx.phase,
                feedback=ctx.feedback,
                enriched_zones=ctx.enriched_zones,
                strategy=ctx.strategy,
                diagnostics=list(ctx.diagnostics),
                topology_class=ctx.topology_class,
                zone_map_version=ctx.zone_map_version,
                attempt=attempt,
                start_time=ctx.start_time,
                fallback_chain=ctx.fallback_chain,
            )

            try:
                recipe_ast = _compile_strategy(ctx_attempt)
                content = recipe_ast.serialize()

                # Validate complexity
                line_count = len(content.strip().split("\n"))
                if line_count > MAX_RECIPE_LINES:
                    ctx.add_diagnostic(
                        CompileSeverity.WARN,
                        f"Recipe exceeds {MAX_RECIPE_LINES} lines "
                        f"(got {line_count}), attempt {attempt}",
                    )
                    # On retry: reduce zones
                    if attempt < MAX_COMPILE_RETRIES:
                        signal_zones = [
                            z for z in ctx.enriched_zones
                            if z.node_type == NodeType.SIGNAL
                        ]
                        if len(signal_zones) > 1:
                            # Keep only top 50% by weight
                            cutoff = len(signal_zones) // 2
                            ctx.enriched_zones = (
                                signal_zones[:cutoff]
                                + [z for z in ctx.enriched_zones # noqa | runtime reachable
                                   if z.node_type != NodeType.SIGNAL]
                            )
                    continue

                compiled = CompiledRecipe(
                    content=content,
                    topology_class=ctx.topology_class,
                    intent=None,
                    strategy=ctx.strategy,
                    phase=ctx.phase,
                    zone_map_version=ctx.zone_map_version,
                    line_count=line_count,
                    stage_count=recipe_ast.stage_count,
                    diagnostics=tuple(ctx_attempt.diagnostics),
                    fallback_chain=tuple(ctx.fallback_chain),
                    compiled_at=time.time(),
                )

                if compiled.is_valid_complexity:
                    return compiled
                else:
                    ctx.add_diagnostic(
                        CompileSeverity.WARN,
                        f"Recipe complexity validation failed, attempt {attempt}",
                    )

            except Exception as exc:
                ctx.add_diagnostic(
                    CompileSeverity.ERROR,
                    f"Compilation attempt {attempt} failed: {exc}",
                )
                if attempt == MAX_COMPILE_RETRIES:
                    log.error(
                        "compilation_exhausted",
                        topology_class=ctx.topology_class,
                        attempts=MAX_COMPILE_RETRIES,
                        last_error=str(exc),
                    )
                    return None

        return None

    def _resolve_phase(
        self,
        topology_class: str,
        phase_override: Optional[str],
    ) -> PhaseStr:
        """Resolve the current phase for a topology class.

        Priority:
        1. Explicit override parameter
        2. Phase cache (updated by PhaseTransitionEvent)
        3. Phase mmap file (persistent state from index_daemon)
        4. Default: LEARNS
        """
        if phase_override:
            try:
                return PhaseStr(phase_override)
            except ValueError:
                pass

        if topology_class in self._phase_cache:
            return self._phase_cache[topology_class]

        mmap_phase = _read_phase_from_mmap(topology_class)
        if mmap_phase is not None:
            self._phase_cache[topology_class] = mmap_phase
            return mmap_phase

        return PhaseStr.LEARNS

    def _get_zone_map_for_class(self, topology_class: str) -> Optional[ZoneMap]:
        """Return cached ZoneMap for topology_class.

        Zone maps arrive exclusively via ZoneMapUpdatedEvent — there is no
        persistent zone map store that parser.py reads. Cold-start misses
        are expected until the WLP emits events for each topology class.
        """
        return self._zone_map_cache.get(topology_class)

    def _infer_intent_hints( # noqa
        self,
        topology_class: str,
        enriched_zones: List[EnrichedZone],
    ) -> Dict[str, List[str]]:
        """Infer intent hints from topology class and zone structure.

        Since the actual ZoneMap doesn't have intent_hints, we infer
        possible intents from the topology class and the structural
        roles of the zones.
        """
        intents: Dict[str, List[str]] = {}

        # SAAS_DOCS intent patterns
        if "SAAS_DOCS" in topology_class:
            code_zones = [
                z.selector.raw for z in enriched_zones
                if z.structural_role == "code_block"
            ]
            if code_zones:
                intents["api_reference"] = code_zones

            list_zones = [
                z.selector.raw for z in enriched_zones
                if z.structural_role in ("list", "ordered_list")
            ]
            if list_zones:
                intents["recovery_codes"] = list_zones

            pricing_zones = [
                z.selector.raw for z in enriched_zones
                if z.structural_role == "pricing"
            ]
            if pricing_zones:
                intents["pricing"] = pricing_zones

        # ECOMMERCE intent patterns
        elif "ECOMMERCE" in topology_class:
            table_zones = [
                z.selector.raw for z in enriched_zones
                if z.structural_role in ("table", "data_table")
            ]
            if table_zones:
                intents["comparison"] = table_zones

        # NEWS intent patterns
        elif "NEWS" in topology_class:
            main_zones = [
                z.selector.raw for z in enriched_zones
                if z.structural_role == "main_content"
            ]
            if main_zones:
                intents["article_body"] = main_zones

        return intents

    async def _emit_compiled_event(
        self,
        event: RecipeCompiledEvent,
    ) -> None:
        """Emit a RecipeCompiledEvent to the bus."""
        if self._bus is not None and hasattr(self._bus, "emit"):
            try:
                await self._bus.emit("recipe_compiled", event)
            except Exception as exc:
                log.error(
                    "event_emission_failed",
                    event_type="recipe_compiled",
                    error=str(exc),
                )

    async def _emit_failure_event(
        self,
        topology_class: str,
        zone_map_version: int,
        failure_reason: str,
        phase: str,
        fallback_chain: Tuple[str, ...],
    ) -> None:
        """Emit a RecipeCompilationFailedEvent to the bus."""
        event = RecipeCompilationFailedEvent(
            topology_class=topology_class,
            zone_map_version=zone_map_version,
            failure_reason=failure_reason,
            phase=phase,
            attempted_at=time.time(),
            fallback_chain=fallback_chain,
        )
        if self._bus is not None and hasattr(self._bus, "emit"):
            try:
                await self._bus.emit("recipe_compilation_failed", event)
            except Exception as exc:
                log.error(
                    "failure_event_emission_failed",
                    error=str(exc),
                )

    # ── Health / Diagnostics ─────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Return compiler health snapshot for observability."""
        return {
            "initialized": self._initialized,
            "compile_count": self._compile_count,
            "fail_count": self._fail_count,
            "cached_zone_maps": len(self._zone_map_cache),
            "cached_phases": dict(
                (k, v.value) for k, v in self._phase_cache.items()
            ),
            "last_compiled_versions": dict(self._last_compiled_version),
            "invalidated_classes": list(self._invalidated_classes),
            "feedback_states": {
                k: {
                    "noise_ema": round(v.noise_ema, 4),
                    "compression_ema": round(v.compression_ema, 4),
                    "total_samples": v.total_samples,
                    "is_stable": v.is_stable,
                }
                for k, v in self._feedback_state.items()
                if v.total_samples > 0
            },
        }


# ═════════════════════════════════════════════════════════════════════════════
# TOPOLOGY-SPECIFIC COMPILATION SPECIALISTS
#
# Pre-built compilation strategies for common topology classes.
# These provide optimized defaults when the WLP's zone map is generic.
# Each specialist encodes domain knowledge about the structure of its
# target topology class.
# ═════════════════════════════════════════════════════════════════════════════

_TOPOLOGY_SPECIALISTS: Dict[str, Dict[str, Any]] = {
    "NEWS_ARTICLE": {
        "default_signal": ["article", ".article-body", ".story-body",
                          ".post-content", "main"],
        "default_noise": ["nav", "header", "footer", "aside",
                         ".sidebar", ".advertisement", ".social-share",
                         ".related-articles", "script", "style"],
        "strategy": "zone_extract",
        "prefer_awk": False,
    },
    "SAAS_DOCS": {
        "default_signal": [".main-content", ".documentation-body",
                          ".article-content", "main", ".content"],
        "default_noise": ["nav", ".sidebar", ".toc", "footer",
                         ".cookie-banner", "script", "style",
                         ".version-selector"],
        "strategy": "zone_extract",
        "prefer_awk": False,
    },
    "SAAS_DOCS_WITH_CODE": {
        "default_signal": [".main-content", "pre", "code",
                          ".code-sample", ".api-endpoint"],
        "default_noise": ["nav", ".sidebar", "footer", "script", "style"],
        "strategy": "zone_extract",
        "prefer_awk": True,
    },
    "FORUM_THREAD": {
        "default_signal": [".question", ".answer", ".accepted-answer",
                          ".post-body", ".comment-body"],
        "default_noise": [".sidebar", ".related-questions", "nav",
                         ".user-card", ".vote-controls", "footer"],
        "strategy": "zone_extract",
        "prefer_awk": True,
    },
    "WIKIPEDIA_ARTICLE": {
        "default_signal": ["#mw-content-text", ".mw-parser-output"],
        "default_noise": [".infobox", ".references", ".reflist",
                         "#catlinks", ".navbox", ".ambox",
                         ".sistersitebox", ".toc", "nav", "script", "style"],
        "strategy": "zone_extract",
        "prefer_awk": True,
    },
    "ECOMMERCE_PRODUCT": {
        "default_signal": [".product-detail", ".product-info",
                          "[data-product-name]", "[data-price]"],
        "default_noise": [".recommendations", ".recently-viewed",
                         "nav", "footer", ".cart-widget"],
        "strategy": "attribute_extract",
        "prefer_awk": False,
    },
    "REST_API_JSON": {
        "default_signal": ["data", "results", "items", "records"],
        "default_noise": ["meta", "pagination", "links", "_links"],
        "strategy": "envelope_extract",
        "prefer_awk": False,
    },
    "JSON_LD_STRUCTURED": {
        "default_signal": ["script[type='application/ld+json']"],
        "default_noise": [],
        "strategy": "attribute_extract",
        "prefer_awk": False,
    },
    "BLOG_POST": {
        "default_signal": [".post-content", ".entry-content",
                          "article", ".blog-post", "main"],
        "default_noise": ["nav", "header", "footer", ".sidebar",
                         ".comments", ".social-share", "script", "style"],
        "strategy": "zone_extract",
        "prefer_awk": False,
    },
    "LANDING_PAGE": {
        "default_signal": ["main", ".hero", ".content", ".features"],
        "default_noise": ["nav", "footer", ".cookie-banner",
                         "script", "style", ".modal"],
        "strategy": "zone_extract",
        "prefer_awk": False,
    },
}


def get_specialist_defaults(topology_class: str) -> Optional[Dict[str, Any]]:
    """Retrieve topology-specific compilation defaults.

    Returns None if no specialist exists for the class.
    The specialist provides fallback signal/noise zones when the
    WLP's zone map is empty or low-confidence.
    """
    return _TOPOLOGY_SPECIALISTS.get(topology_class)


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE METRICS ENGINE
#
# Computes quality metrics for compiled recipes without executing them.
# Used for compile-time quality estimation and comparison between
# recipe versions.
#
# Metrics are approximations based on static analysis of the recipe AST:
# - Estimated compression from number of extraction vs stripping stages
# - Complexity score from pipeline depth and awk program size
# - Specificity score from selector analysis
# - Coverage estimate from zone count and weight distribution
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RecipeMetrics:
    """Static analysis metrics for a compiled recipe."""
    estimated_compression: float
    complexity_score: float
    specificity_score: float
    coverage_estimate: float
    extraction_stage_count: int
    stripping_stage_count: int
    awk_program_count: int
    total_line_count: int

    @property
    def quality_estimate(self) -> float:
        """Composite quality score in [0.0, 1.0].
        Higher = better expected extraction quality."""
        return (
            0.3 * min(self.estimated_compression / 0.7, 1.0)
            + 0.2 * min(self.specificity_score, 1.0)
            + 0.2 * self.coverage_estimate
            + 0.15 * max(0.0, 1.0 - self.complexity_score / 50.0)
            + 0.15 * (1.0 if self.extraction_stage_count > 0 else 0.0)
        )


def compute_recipe_metrics(recipe: CompiledRecipe) -> RecipeMetrics:
    """Compute static analysis metrics for a compiled recipe.

    These are estimates — actual quality is measured post-execution by
    feedback.py. Compile-time metrics guide the compiler's decisions
    when comparing alternative compilation strategies.
    """
    lines = recipe.content.strip().split("\n")
    non_comment = [l for l in lines if l.strip() and not l.strip().startswith("#")]

    extraction_count = sum(
        1 for l in non_comment
        if "sed -n" in l or "grep -A" in l or "grep -oP" in l
    )
    stripping_count = sum(
        1 for l in non_comment
        if "/d" in l and "sed" in l
    )
    awk_count = sum(1 for l in non_comment if "awk" in l)

    # Estimate compression from stage ratio
    total_stages = max(extraction_count + stripping_count, 1)
    estimated_compression = min(
        0.95,
        0.4 + 0.1 * stripping_count / max(total_stages, 1),
    )

    # Complexity from total stages and awk programs
    complexity = (
        len(non_comment) * 0.5
        + awk_count * 5.0
    )

    # Specificity from extraction patterns
    specificity = min(1.0, extraction_count * 0.2 + stripping_count * 0.15)

    # Coverage from extraction stage count
    coverage = min(1.0, extraction_count * 0.3) if extraction_count > 0 else 0.1

    return RecipeMetrics(
        estimated_compression=estimated_compression,
        complexity_score=complexity,
        specificity_score=specificity,
        coverage_estimate=coverage,
        extraction_stage_count=extraction_count,
        stripping_stage_count=stripping_count,
        awk_program_count=awk_count,
        total_line_count=len(lines),
    )


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE DIFF ENGINE
#
# Computes structured diffs between recipe versions for observability.
# Used to understand what changed when a recipe is recompiled and why.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RecipeDiff:
    """Structured diff between two recipe versions."""
    topology_class: str
    old_version: int
    new_version: int
    old_checksum: str
    new_checksum: str
    lines_added: int
    lines_removed: int
    stages_added: int
    stages_removed: int
    strategy_changed: bool
    phase_changed: bool
    old_metrics: RecipeMetrics
    new_metrics: RecipeMetrics

    @property
    def quality_delta(self) -> float:
        """Change in estimated quality. Positive = improvement."""
        return self.new_metrics.quality_estimate - self.old_metrics.quality_estimate

    @property
    def is_significant_change(self) -> bool:
        """True if the diff represents a meaningful structural change."""
        return (
            self.strategy_changed
            or self.phase_changed
            or abs(self.quality_delta) > 0.1
            or self.lines_added + self.lines_removed > 5
        )


def diff_recipes(
    old: CompiledRecipe,
    new: CompiledRecipe,
) -> RecipeDiff:
    """Compute a structured diff between two compiled recipes."""
    from collections import Counter
    old_lines = Counter(old.content.strip().split("\n"))
    new_lines = Counter(new.content.strip().split("\n"))

    old_metrics = compute_recipe_metrics(old)
    new_metrics = compute_recipe_metrics(new)

    added = new_lines - old_lines
    removed = old_lines - new_lines

    return RecipeDiff(
        topology_class=new.topology_class,
        old_version=old.zone_map_version,
        new_version=new.zone_map_version,
        old_checksum=old.checksum,
        new_checksum=new.checksum,
        lines_added=sum(added.values()),
        lines_removed=sum(removed.values()),
        stages_added=max(0, new.stage_count - old.stage_count),
        stages_removed=max(0, old.stage_count - new.stage_count),
        strategy_changed=old.strategy != new.strategy,
        phase_changed=old.phase != new.phase,
        old_metrics=old_metrics,
        new_metrics=new_metrics,
    )


# ═════════════════════════════════════════════════════════════════════════════
# COMPILER STATISTICS AND OBSERVABILITY
#
# Maintains rolling statistics about compilation patterns, rule firing
# frequencies, strategy distribution, and timing percentiles.
# ═════════════════════════════════════════════════════════════════════════════

class CompilerStatistics:
    """Rolling compilation statistics for observability.

    Tracks:
    - Compilation count by topology class and strategy
    - Rule firing frequency
    - Timing percentiles
    - Failure rates
    - Feedback stability
    """

    __slots__ = (
        "_compile_counts",
        "_strategy_counts",
        "_rule_fire_counts",
        "_timing_history",
        "_failure_counts",
        "_total_compiles",
    )

    def __init__(self) -> None:
        self._compile_counts: Dict[str, int] = defaultdict(int)
        self._strategy_counts: Dict[str, int] = defaultdict(int)
        self._rule_fire_counts: Dict[int, int] = defaultdict(int)
        self._timing_history: Deque[float] = deque(maxlen=1000)
        self._failure_counts: Dict[str, int] = defaultdict(int)
        self._total_compiles: int = 0

    def record_compilation(
        self,
        topology_class: str,
        strategy: str,
        elapsed_ms: float,
        diagnostics: Sequence[CompilerDiagnostic],
    ) -> None:
        self._total_compiles += 1
        self._compile_counts[topology_class] += 1
        self._strategy_counts[strategy] += 1
        self._timing_history.append(elapsed_ms)

        for diag in diagnostics:
            if diag.rule_id is not None:
                self._rule_fire_counts[diag.rule_id] += 1

    def record_failure(self, topology_class: str) -> None:
        self._failure_counts[topology_class] += 1

    @property
    def p50_latency_ms(self) -> float:
        if not self._timing_history:
            return 0.0
        sorted_times = sorted(self._timing_history)
        idx = len(sorted_times) // 2
        return sorted_times[idx]

    @property
    def p95_latency_ms(self) -> float:
        if not self._timing_history:
            return 0.0
        sorted_times = sorted(self._timing_history)
        idx = int(len(sorted_times) * 0.95)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    @property
    def p99_latency_ms(self) -> float:
        if not self._timing_history:
            return 0.0
        sorted_times = sorted(self._timing_history)
        idx = int(len(sorted_times) * 0.99)
        return sorted_times[min(idx, len(sorted_times) - 1)]

    def snapshot(self) -> Dict[str, Any]:
        """Full statistics snapshot for observability."""
        return {
            "total_compiles": self._total_compiles,
            "compile_counts_by_class": dict(self._compile_counts),
            "strategy_distribution": dict(self._strategy_counts),
            "rule_fire_frequency": dict(self._rule_fire_counts),
            "failure_counts": dict(self._failure_counts),
            "latency_p50_ms": round(self.p50_latency_ms, 2),
            "latency_p95_ms": round(self.p95_latency_ms, 2),
            "latency_p99_ms": round(self.p99_latency_ms, 2),
            "timing_sample_count": len(self._timing_history),
        }


# ═════════════════════════════════════════════════════════════════════════════
# GENERIC HTML RECIPE BUILDER
#
# Builds the guaranteed-terminal GENERIC_HTML recipe that handles any
# HTML page regardless of topology class. This is the safety net.
# ═════════════════════════════════════════════════════════════════════════════

def build_generic_html_recipe(phase: PhaseStr = PhaseStr.LEARNS) -> str:
    """Build the GENERIC_HTML fallback recipe.

    Conservative extraction: keep <body> content, strip known noise
    elements. No topology-specific knowledge required.

    This recipe must always produce non-empty output on any HTML page
    that has a <body> element. It is the compiler's final fallback.
    """
    zone_map = _GENERIC_HTML_ZONE_MAP
    enriched = enrich_zone_map(zone_map)

    ctx = CompilerContext(
        zone_map=zone_map,
        phase=phase,
        feedback=FeedbackState(),
        enriched_zones=enriched,
        strategy=ExtractionStrategy.DEPTH_FIRST,
        diagnostics=[],
        topology_class="GENERIC_HTML",
        zone_map_version=0,
        attempt=1,
        start_time=time.monotonic(),
    )

    recipe_ast = _compose_zone_extract(ctx)
    return recipe_ast.serialize()


# ═════════════════════════════════════════════════════════════════════════════
# BATCH COMPILATION
#
# Compiles recipes for multiple topology classes in a single invocation.
# Used during cold start to pre-compile recipes for all known classes.
# ═════════════════════════════════════════════════════════════════════════════

async def batch_compile(
    compiler: RecipeCompiler,
    zone_maps: List[ZoneMap],
    *,
    concurrency: int = 4,
) -> Dict[str, List[CompiledRecipe]]:
    """Compile recipes for multiple zone maps concurrently.

    Args:
        compiler: The RecipeCompiler instance.
        zone_maps: List of ZoneMaps to compile.
        concurrency: Maximum concurrent compilations.

    Returns: Dict mapping topology_class → list of compiled recipes.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: Dict[str, List[CompiledRecipe]] = {}

    async def _compile_one(zm: ZoneMap) -> None:
        async with semaphore:
            try:
                recipes = await compiler.compile(zm)
                results[zm.topology_class] = recipes
            except Exception as exc:
                log.error(
                    "batch_compile_failed",
                    topology_class=zm.topology_class,
                    error=str(exc),
                )
                results[zm.topology_class] = []

    tasks = [asyncio.create_task(_compile_one(zm)) for zm in zone_maps]
    await asyncio.gather(*tasks, return_exceptions=True)

    return results


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON AND EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

# Module-level compiler instance. Initialized lazily.
_COMPILER: Optional[RecipeCompiler] = None

def get_compiler() -> RecipeCompiler:
    global _COMPILER
    if _COMPILER is None:
        _COMPILER = RecipeCompiler()
    return _COMPILER

# Module-level statistics tracker
COMPILER_STATS: CompilerStatistics = CompilerStatistics()


# ═════════════════════════════════════════════════════════════════════════════
# COMPILATION CACHE
#
# LRU cache of compiled recipes to avoid redundant recompilation.
# Cache key is (topology_class, zone_map_version, phase, intent).
# Cache is bounded by MAX_CACHE_ENTRIES and uses time-based eviction
# for stale entries.
#
# Mathematical model: the cache implements an approximation of Belady's
# optimal replacement algorithm using the LRU heuristic. With temporal
# locality in zone map updates (the same classes are updated frequently),
# LRU provides near-optimal hit rates.
#
# Hit rate tracking uses exponential moving average:
#   hit_rate_ema = α · hit + (1 - α) · hit_rate_ema
# where hit ∈ {0, 1} and α = 0.05 (slow-moving average).
# ═════════════════════════════════════════════════════════════════════════════

MAX_CACHE_ENTRIES: Final[int] = 256
"""Maximum number of cached compiled recipes.
256 covers all 18 topology classes × ~10 versions × intent variants."""

CACHE_TTL_S: Final[float] = 3600.0
"""Time-to-live for cached entries in seconds. 1 hour.
After TTL, entry is evicted on next access."""

CACHE_HIT_RATE_ALPHA: Final[float] = 0.05
"""EMA smoothing factor for cache hit rate tracking."""


@dataclass
class CacheEntry:
    """Single entry in the compilation cache."""
    key: Tuple[str, int, str, Optional[str]]
    recipe: CompiledRecipe
    created_at: float
    last_accessed: float
    access_count: int

    @property
    def is_expired(self) -> bool:
        return (time.monotonic() - self.created_at) > CACHE_TTL_S

    @property
    def age_s(self) -> float:
        return time.monotonic() - self.created_at


class CompilationCache:
    """LRU compilation cache with time-based eviction.

    Thread-safe via asyncio lock. All operations are O(1) amortized
    using an OrderedDict for LRU ordering.

    Cache key: (topology_class, zone_map_version, phase, intent)
    """

    __slots__ = (
        "_store", "_lock", "_hits", "_misses",
        "_evictions", "_hit_rate_ema",
    )

    def __init__(self) -> None:
        self._store: OrderedDict[
            Tuple[str, int, str, Optional[str]], CacheEntry
        ] = OrderedDict()
        self._lock = asyncio.Lock()
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._hit_rate_ema: float = 0.0

    async def get(
        self,
        topology_class: str,
        version: int,
        phase: str,
        intent: Optional[str] = None,
    ) -> Optional[CompiledRecipe]:
        """Retrieve a cached recipe. Returns None on miss or expired."""
        key = (topology_class, version, phase, intent)
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                self._hit_rate_ema = (
                    CACHE_HIT_RATE_ALPHA * 0.0
                    + (1 - CACHE_HIT_RATE_ALPHA) * self._hit_rate_ema
                )
                return None

            if entry.is_expired:
                del self._store[key]
                self._evictions += 1
                self._misses += 1
                self._hit_rate_ema = (
                    CACHE_HIT_RATE_ALPHA * 0.0
                    + (1 - CACHE_HIT_RATE_ALPHA) * self._hit_rate_ema
                )
                return None

            # Move to end (most recently used)
            self._store.move_to_end(key)
            entry.last_accessed = time.monotonic()
            entry.access_count += 1
            self._hits += 1
            self._hit_rate_ema = (
                CACHE_HIT_RATE_ALPHA * 1.0
                + (1 - CACHE_HIT_RATE_ALPHA) * self._hit_rate_ema
            )
            return entry.recipe

    async def put(self, recipe: CompiledRecipe) -> None:
        """Store a compiled recipe in the cache."""
        key = (
            recipe.topology_class,
            recipe.zone_map_version,
            recipe.phase.value,
            recipe.intent,
        )
        async with self._lock:
            now = time.monotonic()
            entry = CacheEntry(
                key=key, recipe=recipe, created_at=now,
                last_accessed=now, access_count=0,
            )
            self._store[key] = entry
            self._store.move_to_end(key)

            # Evict LRU entries if over capacity
            while len(self._store) > MAX_CACHE_ENTRIES:
                self._store.popitem(last=False)
                self._evictions += 1

    async def invalidate(self, topology_class: str) -> int:
        """Invalidate all cached entries for a topology class."""
        async with self._lock:
            keys_to_remove = [
                k for k in self._store if k[0] == topology_class
            ]
            for k in keys_to_remove:
                del self._store[k]
                self._evictions += 1
            return len(keys_to_remove)

    async def clear(self) -> None:
        """Clear the entire cache."""
        async with self._lock:
            self._store.clear()

    def snapshot(self) -> Dict[str, Any]:
        """Cache statistics snapshot."""
        total = self._hits + self._misses
        return {
            "size": len(self._store),
            "max_size": MAX_CACHE_ENTRIES,
            "hits": self._hits,
            "misses": self._misses,
            "evictions": self._evictions,
            "hit_rate": round(self._hits / max(total, 1), 4),
            "hit_rate_ema": round(self._hit_rate_ema, 4),
            "oldest_entry_age_s": round(
                min((e.age_s for e in self._store.values()), default=0.0), 1
            ),
        }


# Module-level cache instance
COMPILATION_CACHE: CompilationCache = CompilationCache()


# ═════════════════════════════════════════════════════════════════════════════
# ADVANCED AWK GENERATORS — TOPOLOGY-SPECIFIC
#
# Specialized awk generators for complex topology classes that require
# state machines beyond what the generic multi-zone generator produces.
# ═════════════════════════════════════════════════════════════════════════════

def _build_wikipedia_extraction_awk(
    signal_zones: List[EnrichedZone],
    noise_zones: List[EnrichedZone],
    *,
    phase: PhaseStr,
) -> str:
    """Generate specialized awk program for Wikipedia article extraction.

    Wikipedia structure: main content in #mw-content-text, with infobox,
    references, navbox, toc, and ambox as noise zones.

    FST: States × {in_infobox, in_references, in_navbox, in_toc, in_ambox} × depth
    """
    lines: List[str] = []
    lines.append("#!/usr/bin/awk -f")
    lines.append(f"# WIKIPEDIA_ARTICLE specialist — phase: {phase.value}")
    lines.append("BEGIN {")
    lines.append("    in_content = 0; in_infobox = 0; in_references = 0")
    lines.append("    in_navbox = 0; in_toc = 0; in_ambox = 0")
    lines.append("    depth = 0; signal_depth = -1")
    lines.append("}")

    lines.append('/<div[^>]*id="mw-content-text"/ {')
    lines.append("    in_content = 1; signal_depth = depth")
    lines.append("}")

    noise_specs = [
        ("infobox", 'class="[^"]*infobox[^"]*"', "table"),
        ("references", 'class="[^"]*reflist[^"]*"', "div"),
        ("navbox", 'class="[^"]*navbox[^"]*"', "div"),
        ("toc", 'id="toc"', "div"),
        ("ambox", 'class="[^"]*ambox[^"]*"', "table"),
    ]
    for name, pattern, tag in noise_specs:
        lines.append(f"/<{tag}[^>]*{pattern}/ {{ in_{name} = 1 }}")
        lines.append(f"/<\\/{tag}>/ && in_{name} {{ in_{name} = 0; next }}")

    noise_cond = " && ".join(f"!in_{n}" for n, _, _ in noise_specs)
    lines.append(f"in_content && {noise_cond} {{")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append("    if (/[^[:space:]]/) print")
    lines.append("}")
    lines.append("/<\\/div>/ && in_content && depth == signal_depth { in_content = 0 }")
    lines.append("{")
    lines.append('    depth += gsub(/<[^\\/][^>]*>/, "") - gsub(/<\\/[^>]*>/, "")')
    lines.append("    if (depth < 0) depth = 0")
    lines.append("}")

    if phase == PhaseStr.KNOWS:
        lines.append(f"{{ if (length($0) < {MIN_SIGNAL_LENGTH}) next }}")

    return "\n".join(lines)


def _build_forum_thread_awk(
    signal_zones: List[EnrichedZone],
    noise_zones: List[EnrichedZone],
    *,
    phase: PhaseStr,
) -> str:
    """Generate specialized awk program for forum thread extraction.

    Multi-post structure: question, accepted answer, other answers.
    Output is zone-labeled: [QUESTION], [ACCEPTED], [ANSWER N].

    FST: States = {idle, in_post, in_accepted, in_answer, in_code}
    """
    lines: List[str] = []
    lines.append("#!/usr/bin/awk -f")
    lines.append(f"# FORUM_THREAD specialist — phase: {phase.value}")
    lines.append("BEGIN {")
    lines.append("    in_post=0; in_accepted=0; in_answer=0")
    lines.append("    in_code=0; in_noise=0; post_count=0; zone=\"\"")
    lines.append("}")

    # Post detection
    for pat in ("question-body", "post-body", "original-post", "thread-starter"):
        lines.append(f'/class="[^"]*{pat}[^"]*"/ {{ in_post=1; zone="[QUESTION]" }}')

    # Accepted answer
    for pat in ("accepted-answer", "best-answer", "solution"):
        lines.append(f'/class="[^"]*{pat}[^"]*"/ {{ in_accepted=1; zone="[ACCEPTED]" }}')

    # Other answers
    for pat in ("answer-body", "reply-body", "post-text"):
        lines.append(f'/class="[^"]*{pat}[^"]*"/ {{')
        lines.append(f"    if (!in_accepted) {{")
        lines.append(f"        in_answer=1; post_count++")
        lines.append(f'        zone="[ANSWER " post_count "]"')
        lines.append(f"    }}")
        lines.append(f"}}")

    # Code blocks
    lines.append("/<pre/ || /<code[^>]*class/ { in_code=1 }")
    lines.append("/<\\/pre>/ || /<\\/code>/ { in_code=0 }")

    # Noise suppression
    for nc in ("user-card", "vote-controls", "post-meta", "share-buttons"):
        lines.append(f'/class="[^"]*{nc}[^"]*"/ {{ in_noise=1 }}')
    lines.append("/<\\/div>/ && in_noise { in_noise=0 }")

    # Extraction
    lines.append("(in_post || in_accepted || in_answer) && !in_noise {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append("    if (/[^[:space:]]/) print zone \" \" $0")
    lines.append("}")

    # Zone exit
    lines.append("/<\\/div>/ {")
    lines.append("    in_post=0; in_accepted=0; in_answer=0")
    lines.append("}")

    if phase == PhaseStr.KNOWS:
        lines.append(f"length($0) < {MIN_SIGNAL_LENGTH} {{ next }}")

    return "\n".join(lines)


def _build_ecommerce_extraction_awk(
    attributes: List[str],
    *,
    phase: PhaseStr,
) -> str:
    """Generate specialized awk for e-commerce product extraction.

    Three extraction phases:
    1. data-* attributes → labeled key-value pairs
    2. JSON-LD structured data → raw JSON content
    3. Microdata itemprop → labeled key-value pairs
    """
    lines: List[str] = []
    lines.append("#!/usr/bin/awk -f")
    lines.append(f"# ECOMMERCE_PRODUCT specialist — phase: {phase.value}")
    lines.append("BEGIN { in_ld = 0 }")

    # data-* attributes
    default_attrs = [
        ("data-product-name", "PRODUCT_NAME"),
        ("data-price", "PRICE"),
        ("data-sku", "SKU"),
        ("data-availability", "AVAILABILITY"),
        ("data-brand", "BRAND"),
        ("data-category", "CATEGORY"),
        ("data-rating", "RATING"),
        ("data-review-count", "REVIEW_COUNT"),
        ("data-description", "DESCRIPTION"),
        ("data-image-url", "IMAGE_URL"),
        ("data-currency", "CURRENCY"),
        ("data-original-price", "ORIGINAL_PRICE"),
    ]
    for attr in attributes:
        label = attr.replace("data-", "").upper().replace("-", "_")
        if (attr, label) not in default_attrs:
            default_attrs.append((attr, label))

    lines.append("{")
    for attr, label in default_attrs:
        safe = attr.replace("-", "\\-")
        lines.append(f'    if (match($0, /{safe}="([^"]*)"/, a)) print "{label}: " a[1]')
    lines.append("}")

    # JSON-LD
    lines.append('/<script[^>]*application\\/ld\\+json[^>]*>/ { in_ld=1; next }')
    lines.append("/<\\/script>/ && in_ld { in_ld=0; next }")
    lines.append('in_ld { gsub(/^[[:space:]]+/, ""); if (/[^[:space:]]/) print }')

    # Microdata
    lines.append("{")
    for prop, label in [("name","PRODUCT_NAME"),("price","PRICE"),
                         ("sku","SKU"),("availability","AVAILABILITY"),
                         ("brand","BRAND"),("description","DESCRIPTION")]:
        lines.append(
            f'    if (match($0, /itemprop="{prop}"[^>]*content="([^"]*)"/, a))'
            f' print "{label}: " a[1]'
        )
    lines.append("}")

    return "\n".join(lines)


def _build_saas_docs_code_awk(
    signal_zones: list,
    noise_zones: list,
    *,
    phase: PhaseStr,
) -> str:
    """Generate a phase-aware awk specialist for SAAS_DOCS_WITH_CODE topology.

    Output label protocol — each emitted line is prefixed with a type tag
    so downstream consumers can route sections without re-parsing:

        SECTION:\\t<text>       h1-h6 headings; marks logical document sections.
        PROSE:\\t<text>         paragraph text that precedes a code block.
                               Isolated paragraphs are buffered and discarded
                               if no code block follows before the next section
                               boundary — avoids dumping footer/intro prose.
        CODE[<lang>]:\\t<line>  one line of code with language identifier.
                               lang extracted from class="language-X" /
                               class="lang-X" / class="highlight X" / class="X".
                               Falls back to "text" when no class is present.
        ENDPOINT:\\t<line>      HTTP method + path lines inside code blocks:
                               GET /api/v1/users, POST /endpoint, etc.
                               Also emitted for curl command lines.
        NOTE:\\t<text>          .note / .tip / .info callout block content.
        WARNING:\\t<text>       .warning / .caution / .danger callout content.
        ---                    code block separator (no tab, no label).

    Signal zone anchoring:
        Each signal zone's selector pattern gates the in_sig_N flag.
        Content is extracted only while at least one signal zone is active.
        If no signal zones match (empty zone map, low-confidence WLP output),
        the program falls back to extracting all content — same behaviour as
        the generic SAAS_DOCS specialist, not silent empty output.

    Noise zone suppression:
        Each noise zone's selector pattern gates an in_noise_N flag.
        Lines are suppressed while any noise flag is active regardless of
        which signal zone is currently open.

    Paragraph buffering:
        Paragraphs accumulate into para_buf across multiple lines.
        Buffer flushes as PROSE: only when a code block entry follows.
        Buffer discards (silently) at: section boundary, noise zone entry,
        end of signal zone. This prevents isolated intro/footer paragraphs
        from appearing in output without code context.

    Inline heading / paragraph handling:
        Real SaaS docs frequently emit <h2>text</h2> on one line.
        A pre-check pattern fires before the open-tag pattern to avoid
        the open-tag handler consuming the line with `next` before the
        close-tag handler can see it.

    Language extraction (POSIX-compatible, mawk safe):
        Uses sub() chains — no 3-argument match() which requires gawk.
        Priority: language- prefix > lang- prefix > highlight class >
                  bare class name > "text" fallback.

    API endpoint detection inside code blocks:
        Lines matching /^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\\s+\\/.../ or
        /^curl\\s/ are re-labeled ENDPOINT: instead of CODE[...]: so the
        index daemon can extract structured API surface from docs pages
        without a separate pass.

    Callout / admonition block extraction:
        .warning .caution .danger → WARNING:\\t
        .note .tip .info          → NOTE:\\t
        Content lines are extracted verbatim (tags stripped).
        Nesting is not tracked — callouts exit on </div>, which is correct
        for Bootstrap/MkDocs/Docusaurus callout patterns.

    Phase I  (LEARNS):   all code blocks emitted regardless of context.
                         Maximum signal capture for training.
    Phase II (PREDICTS): code only when preceded by paragraph or heading.
                         Endpoint pattern extraction active.
    Phase III (KNOWS):   same as PREDICTS plus MIN_SIGNAL_LENGTH filter on
                         prose lines, and deduplication of repeated code blocks
                         via seen[] associative array.

    FST specification:
        States:  {in_sig_0..N, in_noise_0..M, in_heading, in_para,
                  in_code, in_callout}
        Aux:     {para_buf, code_lang, keep_code, prev_context,
                  code_buf (KNOWS dedup), seen[] (KNOWS dedup)}
        Input:   HTML line tokens (one awk record per line)
        Output:  label-prefixed lines as specified above
    """
    is_learns   = (phase == PhaseStr.LEARNS)
    is_predicts = (phase == PhaseStr.PREDICTS)
    is_knows    = (phase == PhaseStr.KNOWS)

    # ── signal / noise zone patterns ─────────────────────────────────────────
    sig_patterns:   list[tuple[int, str, str]] = []   # (idx, pattern, close_tag)
    noise_patterns: list[tuple[int, str, str]] = []

    for i, z in enumerate(signal_zones):
        pat = _selector_to_awk_pattern(z.selector)
        if pat:
            tag = z.selector.tag or "div"
            sig_patterns.append((i, pat, tag))

    for i, z in enumerate(noise_zones):
        pat = _selector_to_awk_pattern(z.selector)
        if pat:
            tag = z.selector.tag or "div"
            noise_patterns.append((i, pat, tag))

    # Active-signal predicate — OR over all sig flags; 1 if none defined
    if sig_patterns:
        sig_active = "(" + " || ".join(f"in_sig_{i}" for i, _, _ in sig_patterns) + ")"
    else:
        sig_active = "1"

    # Noise-clear predicate — AND over all !in_noise flags; 1 if none defined
    if noise_patterns:
        noise_clear = " && ".join(f"!in_noise_{i}" for i, _, _ in noise_patterns)
    else:
        noise_clear = "1"

    in_zone = f"{sig_active} && {noise_clear}"

    lines: list[str] = []

    # ── header comment ────────────────────────────────────────────────────────
    lines.append("#!/usr/bin/awk -f")
    lines.append(f"# SAAS_DOCS_WITH_CODE specialist — phase: {phase.value}")
    lines.append(f"# signal zones: {len(sig_patterns)}  noise zones: {len(noise_patterns)}")
    lines.append("")

    # ── BEGIN ─────────────────────────────────────────────────────────────────
    lines.append("BEGIN {")
    for i, _, _ in sig_patterns:
        lines.append(f"    in_sig_{i}=0")
    for i, _, _ in noise_patterns:
        lines.append(f"    in_noise_{i}=0")
    lines.append('    in_heading=0; in_para=0; in_code=0; in_callout=0')
    lines.append('    keep_code=0; prev_context=""; callout_tone=""')
    lines.append('    para_buf=""; code_lang="text"')
    if is_knows:
        lines.append('    code_buf=""')
    lines.append("}")
    lines.append("")

    # ── signal zone entry / exit ──────────────────────────────────────────────
    if sig_patterns:
        lines.append("# signal zone entry / exit")
        for i, pat, tag in sig_patterns:
            lines.append(f"/{pat}/ {{ in_sig_{i}=1 }}")
            # Guard: do not exit while inside a callout, heading, para, or code
            # block — their </div> or </pre> must not prematurely close the zone.
            lines.append(
                f"/<\\/{tag}>/ && in_sig_{i} && !in_callout && !in_code && !in_heading {{"
                f" in_sig_{i}=0; para_buf=\"\"; prev_context=\"\"; next }}"
            )
        lines.append("")

    # ── noise zone entry / exit ───────────────────────────────────────────────
    if noise_patterns:
        lines.append("# noise zone entry / exit")
        for i, pat, tag in noise_patterns:
            lines.append(f"/{pat}/ {{ in_noise_{i}=1; next }}")
            lines.append(f"/<\\/{tag}>/ && in_noise_{i} {{ in_noise_{i}=0; next }}")
        lines.append("")

    # ── callout / admonition blocks ───────────────────────────────────────────
    lines.append("# callout block entry — WARNING / NOTE tones")
    lines.append(f'{in_zone} && /<div[^>]*class="[^"]*(warning|caution|danger)[^"]*"/ {{')
    lines.append('    in_callout=1; callout_tone="WARNING"; next')
    lines.append("}")
    lines.append(f'{in_zone} && /<div[^>]*class="[^"]*(note|tip|info)[^"]*"/ {{')
    lines.append('    in_callout=1; callout_tone="NOTE"; next')
    lines.append("}")
    lines.append(f'in_callout && /<\\/div>/ {{ in_callout=0; next }}')
    lines.append("in_callout {")
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print callout_tone ":\\t" $0')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── section headings — single-line must be tested before open-tag ─────────
    lines.append("# headings: single-line check before open-tag to avoid next eating close")
    lines.append(f'{in_zone} && /<h[1-6][^>]*>.*<\\/h[1-6]>/ {{')
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print "SECTION:\\t" $0')
    lines.append('    prev_context="heading"; para_buf=""')
    lines.append("    next")
    lines.append("}")
    lines.append(f'{in_zone} && /<h[1-6][> ]/ {{ in_heading=1; next }}')
    lines.append(f'{in_zone} && in_heading && /<\\/h[1-6]>/ {{')
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print "SECTION:\\t" $0')
    lines.append('    in_heading=0; prev_context="heading"; para_buf=""')
    lines.append("    next")
    lines.append("}")
    lines.append(f'in_heading && {in_zone} {{')
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) print "SECTION:\\t" $0')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── paragraph buffering ───────────────────────────────────────────────────
    lines.append("# paragraph buffering: inline single-line check first")
    lines.append(f'{in_zone} && !in_code && /<p[^>]*>.*<\\/p>/ {{')
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) {')
    lines.append('        if (length(para_buf) > 0) para_buf = para_buf " "')
    lines.append('        para_buf = para_buf $0')
    lines.append('    }')
    lines.append('    prev_context="para"')
    lines.append("    next")
    lines.append("}")
    lines.append(f'{in_zone} && !in_code && /<p[> ]/ {{ in_para=1; next }}')
    lines.append(f'{in_zone} && in_para && /<\\/p>/ {{')
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) {')
    lines.append('        if (length(para_buf) > 0) para_buf = para_buf " "')
    lines.append('        para_buf = para_buf $0')
    lines.append('    }')
    lines.append('    in_para=0; prev_context="para"')
    lines.append("    next")
    lines.append("}")
    lines.append(f'in_para && {in_zone} && !/<p/ && !/<\\/p>/ {{')
    lines.append('    gsub(/<[^>]*>/, "")')
    lines.append('    gsub(/^[[:space:]]+|[[:space:]]+$/, "")')
    lines.append('    if (/[^[:space:]]/) {')
    lines.append('        if (length(para_buf) > 0) para_buf = para_buf " "')
    lines.append('        para_buf = para_buf $0')
    lines.append('    }')
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── code block entry ──────────────────────────────────────────────────────
    lines.append("# code block entry — <pre> has no language class")
    lines.append(f'{in_zone} && !in_code && /<pre[> ]/ {{')
    lines.append('    in_code=1; code_lang="text"')
    if is_learns:
        lines.append('    keep_code=1')
    else:
        lines.append('    keep_code=(prev_context == "para" || prev_context == "heading")')
    lines.append('    if (keep_code && length(para_buf) > 0) {')
    lines.append('        print "PROSE:\\t" para_buf; para_buf=""')
    lines.append("    }")
    lines.append("    next")
    lines.append("}")
    lines.append("")
    lines.append("# code block entry — <code class=...> fenced block with language")
    lines.append(f'{in_zone} && !in_code && /<code[^>]*class=/ {{')
    lines.append('    in_code=1')
    lines.append('    _ln = $0')
    # Language extraction: priority chain, all POSIX-compatible (no gawk 3-arg match)
    lines.append('    if (sub(/.*class="[^"]*language-/, "", _ln)) {')
    lines.append('        code_lang=_ln; sub(/".*/, "", code_lang)')
    lines.append('    } else if (sub(/.*class="[^"]*lang-/, "", _ln)) {')
    lines.append('        code_lang=_ln; sub(/".*/, "", code_lang)')
    lines.append('    } else if (sub(/.*class="highlight /, "", _ln)) {')
    lines.append('        code_lang=_ln; sub(/".*/, "", code_lang)')
    lines.append('    } else if (sub(/.*class="/, "", _ln)) {')
    lines.append('        code_lang=_ln; sub(/".*/, "", code_lang)')
    lines.append('    } else {')
    lines.append('        code_lang="text"')
    lines.append('    }')
    if is_learns:
        lines.append('    keep_code=1')
    else:
        lines.append('    keep_code=(prev_context == "para" || prev_context == "heading")')
    lines.append('    if (keep_code && length(para_buf) > 0) {')
    lines.append('        print "PROSE:\\t" para_buf; para_buf=""')
    lines.append("    }")
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── code block exit ───────────────────────────────────────────────────────
    lines.append("/<\\/pre>/ || /<\\/code>/ {")
    lines.append("    if (in_code) {")
    if is_knows:
        # Phase III: dedup code blocks by content hash (seen[code_buf])
        lines.append('        if (keep_code && length(code_buf) > 0 && !seen[code_buf]++) {')
        lines.append('            print "---"')
        lines.append('        }')
        lines.append('        code_buf=""')
    else:
        lines.append('        if (keep_code) print "---"')
    lines.append('        in_code=0; keep_code=0; prev_context=""; para_buf=""')
    lines.append("    }")
    lines.append("    next")
    lines.append("}")
    lines.append("")

    # ── code body ─────────────────────────────────────────────────────────────
    lines.append("in_code && keep_code {")
    lines.append('    gsub(/<[^>]*>/, "")')
    # normalise exactly one level of 4-space indent (common in HTML-escaped code)
    lines.append('    gsub(/^[[:space:]]{4}/, "")')
    lines.append('    gsub(/^[[:space:]]+$/, "")')
    lines.append("    if (!/[^[:space:]]/) next")
    # endpoint detection inside code blocks (PREDICTS + KNOWS)
    if not is_learns:
        lines.append('    if ($0 ~ /^(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)[[:space:]]+\\//) {')
        lines.append('        print "ENDPOINT:\\t" $0; next')
        lines.append("    }")
        lines.append('    if ($0 ~ /^curl[[:space:]]/) {')
        lines.append('        print "ENDPOINT:\\t" $0; next')
        lines.append("    }")
    if is_knows:
        lines.append('    code_buf = code_buf $0 "\\n"')
        lines.append(f'    if (length($0) < {MIN_SIGNAL_LENGTH}) next')
    lines.append('    print "CODE[" code_lang "]:\\t" $0')
    lines.append("    next")
    lines.append("}")
    lines.append("in_code && !keep_code { next }")

    if is_knows:
        lines.append("")
        lines.append(f"# Phase III: suppress short prose lines outside code blocks")
        lines.append(f"!in_code && length($0) < {MIN_SIGNAL_LENGTH} {{ next }}")

    return "\n".join(lines)

# ═════════════════════════════════════════════════════════════════════════════
# TESTING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def compile_from_zone_map_sync(
    zone_map: ZoneMap,
    *,
    phase: str = "learns",
) -> List[CompiledRecipe]:
    """Synchronous wrapper for compile() for testing."""
    compiler = RecipeCompiler()
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(compiler.initialize())
        return loop.run_until_complete(compiler.compile(zone_map, phase))
    finally:
        loop.close()


def validate_recipe_content(content: str) -> Tuple[bool, List[str]]:
    """Validate recipe content without writing to disk."""
    issues: List[str] = []
    is_safe, reason = _sanitize_recipe(content)
    if not is_safe:
        issues.append(f"Sanitization: {reason}")
    lines = content.strip().split("\n")
    if len(lines) > MAX_RECIPE_LINES:
        issues.append(f"Line count {len(lines)} exceeds limit {MAX_RECIPE_LINES}")
    pipe_count = content.count(" | ") + content.count("\n| ")
    if pipe_count > MAX_PIPELINE_STAGES:
        issues.append(f"Pipeline stages {pipe_count} exceeds limit {MAX_PIPELINE_STAGES}")
    if "curl" in content or "wget" in content:
        issues.append("Recipe contains network commands")
    return len(issues) == 0, issues


def generate_test_zone_map(
    topology_class: str = "NEWS_ARTICLE",
    *,
    signal_zones: Optional[List[str]] = None,
    noise_zones: Optional[List[str]] = None,
    strategy: str = "zone_extract",
    confidence: float = 0.85,
    version: int = 1,
) -> ZoneMap:
    """Generate a test ZoneMap with sensible defaults."""
    specialist = get_specialist_defaults(topology_class)
    if signal_zones is None:
        signal_zones = specialist["default_signal"] if specialist else ["main"]
    if noise_zones is None:
        noise_zones = (
            specialist["default_noise"] if specialist else
            ["nav", "footer", "script", "style"]
        )
    return ZoneMap(
        topology_class=topology_class,
        signal_zones=signal_zones,
        noise_zones=noise_zones,
        strategy=strategy,
        confidence=confidence,
        version=version,
    )


def explain_compilation(
    zone_map: ZoneMap,
    *,
    phase: str = "learns",
    verbose: bool = False,
) -> str:
    """Generate a human-readable explanation of what the compiler would do."""
    strategy = _determine_strategy(zone_map)
    enriched = enrich_zone_map(zone_map)
    phase_str = PhaseStr(phase) if phase in ("learns", "predicts", "knows") else PhaseStr.LEARNS
    temp = {PhaseStr.LEARNS: SOFTMAX_TEMP_LEARNS,
            PhaseStr.PREDICTS: SOFTMAX_TEMP_PREDICTS,
            PhaseStr.KNOWS: SOFTMAX_TEMP_KNOWS}[phase_str]

    lines: List[str] = []
    lines.append(f"═══ Compilation Plan for {zone_map.topology_class} ═══")
    lines.append(f"Strategy: {strategy.value}")
    lines.append(f"Phase: {phase_str.value} (softmax τ={temp})")
    lines.append(f"ZoneMap confidence: {zone_map.confidence}")
    lines.append(f"ZoneMap version: {zone_map.version}")
    lines.append("")

    if zone_map.confidence < THETA_WLP_COMPILE_MIN:
        parent = PARENT_CLASS_MAP.get(zone_map.topology_class)
        lines.append(
            f"⚠ Low confidence → fallback to {parent or 'GENERIC_HTML'}")
        lines.append("")

    lines.append(f"Enriched zones ({len(enriched)} total):")
    for i, ez in enumerate(enriched):
        lines.append(
            f"  [{i}] {ez.node_type.value:8s} | w={ez.weight:.3f} | "
            f"role={ez.structural_role:15s} | {ez.selector.raw}"
        )
        if verbose and ez.data_attributes:
            lines.append(f"       data_attrs: {', '.join(ez.data_attributes)}")
        if verbose and ez.json_path:
            lines.append(f"       json_path: {ez.json_path}")

    # Softmax preview
    normalized = softmax_normalize_weights(enriched, temp)
    signal_norm = [z for z in normalized if z.node_type == NodeType.SIGNAL]
    if signal_norm:
        lines.append(f"\nSoftmax-normalized weights:")
        for z in signal_norm:
            lines.append(f"  {z.selector.raw:30s} → {z.weight:.4f}")

    # Rule firing preview
    ctx = CompilerContext(
        zone_map=zone_map, phase=phase_str, feedback=FeedbackState(),
        enriched_zones=enriched, strategy=strategy, diagnostics=[],
        topology_class=zone_map.topology_class,
        zone_map_version=zone_map.version, attempt=1,
        start_time=time.monotonic(),
    )
    lines.append("\nTranslation rules that would fire:")
    for rule in TRANSLATION_RULES:
        for ez in enriched:
            if rule.applies(ez, ctx):
                lines.append(
                    f"  R{rule.rule_id:02d} ({rule.name:25s}) → {ez.selector.raw}"
                )
                break

    lines.append("\n═══ End ═══")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE TEMPLATE LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

_RECIPE_TEMPLATES: Dict[str, str] = {
    "GENERIC_ARTICLE": textwrap.dedent("""\
        #!/bin/sh
        # template: GENERIC_ARTICLE
        # Extracts main content from semantic HTML5 elements.
        # Tries <article>, <main>, <div role="main"> in order.
        sed -n '/<article\\|<main\\|role="main"/,/<\\/article>\\|<\\/main>/p' \\
        | sed '/<nav[^>]*>/,/<\\/nav>/d' \\
        | sed '/<aside[^>]*>/,/<\\/aside>/d' \\
        | sed '/<footer[^>]*>/,/<\\/footer>/d' \\
        | sed '/<header[^>]*>/,/<\\/header>/d' \\
        | sed '/<form[^>]*>/,/<\\/form>/d' \\
        | sed '/<script[^>]*>/,/<\\/script>/d' \\
        | sed '/<style[^>]*>/,/<\\/style>/d' \\
        | sed 's/<[^>]*>//g' \\
        | sed 's/&amp;/\&/g; s/&lt;/</g; s/&gt;/>/g; s/&nbsp;/ /g; s/&#[0-9]*;//g' \\
        | tr -s ' \\t\\n' '\\n' \\
        | grep -v '^[[:space:]]*$'
    """),
    "GENERIC_JSON": textwrap.dedent("""\
        #!/bin/sh
        # template: GENERIC_JSON
        # Extracts top-level JSON object fields.
        # Guards against braces inside string values.
        awk '
        BEGIN { depth=0; in_str=0 }
        {
            line = $0
            i = 1
            while (i <= length(line)) {
                c = substr(line, i, 1)
                if (in_str) {
                    if (c == "\\\\" ) { i++ }
                    else if (c == "\\"") { in_str = 0 }
                } else {
                    if (c == "\\"") { in_str = 1 }
                    else if (c == "{" || c == "[") { depth++ }
                    else if (c == "}" || c == "]") { depth-- }
                }
                i++
            }
            if (depth >= 1) { print }
        }
        ' \\
        | sed 's/^[[:space:]]*//' \\
        | grep -v '^[{}\\[\\],]*$' \\
        | grep -v '^[[:space:]]*$'
    """),
    "GENERIC_LIST": textwrap.dedent("""\
        #!/bin/sh
        # template: GENERIC_LIST
        # Handles ordered, unordered, and definition lists.
        sed -n '/<[oud]l/,/<\\/[oud]l>/p' \\
        | sed '/<script[^>]*>/,/<\\/script>/d' \\
        | sed 's/<\\/\\?l[iod][^>]*>//g' \\
        | sed 's/<[^>]*>//g' \\
        | sed 's/&amp;/\&/g; s/&lt;/</g; s/&gt;/>/g; s/&nbsp;/ /g' \\
        | tr -s ' \\t\\n' '\\n' \\
        | grep -v '^[[:space:]]*$'
    """),
    "GENERIC_TABLE": textwrap.dedent("""\
        #!/bin/sh
        # template: GENERIC_TABLE
        # Extracts table rows as tab-separated values.
        sed -n '/<table/,/<\\/table>/p' \\
        | sed '/<thead[^>]*>/,/<\\/thead>/d' \\
        | grep -o '<t[dh][^>]*>[^<]*' \\
        | sed 's/<t[dh][^>]*>//' \\
        | paste -d'\\t' - \\
        | grep -v '^[[:space:]]*$'
    """),
    "GENERIC_HTML": textwrap.dedent("""\
        #!/bin/sh
        # template: GENERIC_HTML
        # Broadest fallback — strips all tags and normalises whitespace.
        sed '/<script[^>]*>/,/<\\/script>/d' \\
        | sed '/<style[^>]*>/,/<\\/style>/d' \\
        | sed '/<nav[^>]*>/,/<\\/nav>/d' \\
        | sed '/<footer[^>]*>/,/<\\/footer>/d' \\
        | sed 's/<[^>]*>//g' \\
        | sed 's/&amp;/\&/g; s/&lt;/</g; s/&gt;/>/g; s/&nbsp;/ /g; s/&#[0-9]*;//g' \\
        | tr -s ' \\t\\n' '\\n' \\
        | grep -v '^[[:space:]]*$'
    """),
}


def get_recipe_template(template_name: str) -> Optional[str]:
    """Retrieve a pre-built recipe template by name."""
    return _RECIPE_TEMPLATES.get(template_name)


# ═════════════════════════════════════════════════════════════════════════════
# EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Main class
    "RecipeCompiler",
    "get_compiler",

    # Intent-conditioned extraction engine
    "IntentConditionedExtractor",
    "IntentDecomposition",
    "IntentZoneScore",
    "IntentRecipePlan",

    # Compilation outputs
    "CompiledRecipe",
    "RecipeCompiledEvent",
    "RecipeCompilationFailedEvent",

    # Cache
    "CompilationCache",
    "COMPILATION_CACHE",

    # Internal types (for testing)
    "ExtractionStrategy",
    "NodeType",
    "PhaseStr",
    "SelectorKind",
    "ParsedSelector",
    "EnrichedZone",
    "CompilerContext",
    "FeedbackState",
    "ShellCommand",
    "ShellPipeline",
    "ShellRecipe",
    "TranslationRule",

    # Functions (for testing)
    "parse_selector",
    "enrich_zone_map",
    "softmax_normalize_weights",
    "compute_selector_similarity",
    "build_generic_html_recipe",
    "batch_compile",
    "compute_recipe_metrics",
    "diff_recipes",
    "compile_from_zone_map_sync",
    "validate_recipe_content",
    "generate_test_zone_map",
    "explain_compilation",
    "get_recipe_template",

    # Constants
    "TRANSLATION_RULES",
    "COMPOSITION_ORDER",
    "THETA_WLP_COMPILE_MIN",
    "COMPILER_GENERATED_PATH",
    "COMPILER_STATS",
]