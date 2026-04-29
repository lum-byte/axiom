"""
signal_kernel/feedback.py
==========================
Computes ExtractionQuality from KernelOutput and emits it as a training
signal back to topology_parser.py via FeedbackEvent. This is the file
where the learning loop either compounds correctly or silently drifts.

feedback.py sits downstream of execution and upstream of learning. It
touches nothing in the execution path and nothing in the storage layer.

Dependency position:

    contracts.py
        ↓
    pipeline.py → KernelOutput → feedback.py → FeedbackEvent → topology_parser.py

feedback.py imports only contracts.py (for types and constants) and
exceptions.py (for FeedbackEmissionError). Everything else is computed
from the KernelOutput it receives.

What feedback.py does:
    1. Receives KernelOutput from pipeline.py's caller
    2. Computes ExtractionQuality deterministically
    3. Records quality in a rolling per-topology-class window
    4. Evaluates whether recompilation is recommended
    5. Emits FeedbackEvent through the registered handler

What feedback.py does NOT do:
    - Does not call an LLM
    - Does not call any external service
    - Does not persist quality history to disk
    - Does not call pipeline.py, registry.py, or validator.py
    - Does not make routing decisions
    - Does not touch the store or the network

All computation is deterministic arithmetic. If you find yourself
wanting to call a model to evaluate extraction quality, stop — that
would introduce circular evaluation where an LLM evaluates LLM input
quality. The kernel's quality loop is intentionally model-free.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import json
import logging
import math # noqa
import re
import statistics
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence, # noqa
    Set, # noqa
    Tuple,
)

from signal_kernel.contracts import (
    EMPTY_EXTRACTION_RATE_THRESHOLD,
    HARDCODED_TOPOLOGY_CLASSES, # noqa
    KNOWN_TOPOLOGY_CLASSES, # noqa
    MAX_MEANINGFUL_TOKEN_REDUCTION,
    MIN_MEANINGFUL_TOKEN_REDUCTION,
    QUALITY_WINDOW_SIZE,
    SIGNAL_DENSITY_CEILING, # noqa
    SIGNAL_DENSITY_FLOOR,
    ExtractionQuality,
    FeedbackEvent,
    KernelOutput,
    QualityWindowEntry,
    RecipeHash,
    RecipeQualityAggregate,
    RunID, # noqa
    TopologyClassStr,
    compute_recipe_hash,
    make_quality_from_output,
)
from signal_kernel.exceptions import ( # noqa
    FeedbackEmissionError,
)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
#
# Feedback-specific parameters. Security constants and quality thresholds
# are imported from contracts.py. What is defined here are operational
# parameters that are feedback.py's internal concern.
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("signal_kernel.feedback")

# ── Token estimation ──────────────────────────────────────────────────────

# Character-to-token ratio for the char/4 approximation.
# ~4 characters per token holds for English text across most tokenizers.
# feedback.py does NOT call a real tokenizer — deterministic arithmetic only.
_CHARS_PER_TOKEN: int = 4

# ── Recompilation thresholds ──────────────────────────────────────────────

# Minimum number of samples in the rolling window before recompilation
# recommendations are considered meaningful. With fewer samples, the
# statistics are too noisy to trust. Default: max(10, QUALITY_WINDOW_SIZE // 10).
_MIN_SAMPLES_FOR_RECOMMENDATION: int = max(10, QUALITY_WINDOW_SIZE // 10)

# When the rolling empty extraction rate exceeds this threshold for a
# topology class, recompilation is recommended. A sustained empty rate
# means the recipe consistently fails to extract content from pages that
# should have signal in the expected zone.
_EMPTY_RATE_RECOMPILATION_THRESHOLD: float = EMPTY_EXTRACTION_RATE_THRESHOLD

# When the rolling mean signal density falls below this threshold,
# recompilation is recommended. Sustained low density means the recipe
# is producing mostly whitespace — it is stripping signal along with noise.
_DENSITY_RECOMPILATION_THRESHOLD: float = SIGNAL_DENSITY_FLOOR

# When the rolling mean token reduction percentage falls below this
# threshold, the recipe is under-stripping — it is not aggressive enough
# and the downstream LLM is still burning tokens on noise.
_REDUCTION_FLOOR_THRESHOLD: float = MIN_MEANINGFUL_TOKEN_REDUCTION

# When the rolling mean token reduction exceeds this ceiling, the recipe
# is over-stripping — it is too aggressive and may be discarding signal.
_REDUCTION_CEILING_THRESHOLD: float = MAX_MEANINGFUL_TOKEN_REDUCTION

# ── Trend analysis ────────────────────────────────────────────────────────

# Number of recent samples to compare against the full window for trend
# detection. If the recent subset has notably worse metrics than the full
# window, the recipe is degrading and recompilation is recommended even
# if the full-window averages are still above threshold.
_TREND_RECENT_WINDOW: int = max(5, QUALITY_WINDOW_SIZE // 5)

# How much worse the recent subset must be compared to the full window
# (as a fraction of the full-window average) to trigger a degradation flag.
# 0.15 means the recent average must be 15% worse than the full-window
# average. This catches recipes that are slowly drifting toward failure
# before they cross the hard thresholds.
_TREND_DEGRADATION_FACTOR: float = 0.15

# ── JSON structured field configuration ───────────────────────────────────

# Topology classes that are JSON-based and have structured fields to count.
# HTML topology classes always get structured_field_count=0.
_JSON_TOPOLOGY_CLASSES: FrozenSet[str] = frozenset({
    "REST_API_JSON",
    "JSON_LD_STRUCTURED",
})

# Expected signal keys per JSON topology class. These are the top-level
# keys the recipe is designed to extract. Missing keys mean the recipe
# is under-extracting — the JSON structure has changed or the recipe's
# key matching patterns are too narrow.
#
# These key sets are defined here (not in contracts.py) because they are
# a feedback concern — they describe what feedback.py looks for in the
# output. The recipes themselves know which keys to extract (that is their
# job). feedback.py independently verifies whether those keys appeared in
# the output. If contracts.py defined these, feedback.py would need to
# import them and the two would be coupled on a detail that only feedback.py
# uses.
_EXPECTED_SIGNAL_KEYS: Dict[str, FrozenSet[str]] = {
    "REST_API_JSON": frozenset({
        "data", "results", "items", "records", "entries",
        "content", "payload", "response", "body", "value",
    }),
    "JSON_LD_STRUCTURED": frozenset({
        "@context", "@type", "name", "description", "url",
        "author", "datePublished", "headline", "image",
        "publisher", "mainEntityOfPage", "articleBody",
    }),
}

# Expected noise keys that should NOT appear in the output. If these
# keys are present, the recipe is under-stripping — it failed to discard
# envelope metadata. Their presence in clean_signal is a quality penalty.
_EXPECTED_NOISE_KEYS: Dict[str, FrozenSet[str]] = {
    "REST_API_JSON": frozenset({
        "pagination", "meta", "links", "_links", "cursor",
        "paging", "page_info", "rate_limit", "headers",
        "included", "relationships",
    }),
    "JSON_LD_STRUCTURED": frozenset({
        "WebSite", "SearchAction", "BreadcrumbList",
        "SiteNavigationElement",
    }),
}

# Minimum number of signal keys that must appear for the extraction to be
# considered structurally successful. If fewer than this threshold appear,
# the recipe may have missed signal sections entirely.
_MIN_SIGNAL_KEYS_THRESHOLD: int = 2

# ── Per-recipe tracking ───────────────────────────────────────────────────

# Maximum number of distinct recipe hashes tracked per topology class.
# When the compiler produces new recipes, old recipe hashes are evicted
# from history. This prevents unbounded growth of per-class state.
_MAX_RECIPE_HASHES_PER_CLASS: int = 10

# ── Emission limits ───────────────────────────────────────────────────────

# Maximum consecutive emission failures before feedback.py stops attempting
# to call the handler. After this many failures, the handler is presumed
# dead and emission becomes a no-op until a new handler is registered.
# This prevents log spam from a persistently broken handler.
_MAX_CONSECUTIVE_EMISSION_FAILURES: int = 50

# ── Recompilation cooldown ────────────────────────────────────────────────

# After a recompilation recommendation is emitted for a topology class,
# feedback.py will NOT emit another recommendation for this many samples.
# This prevents recommendation spam when the recipe is consistently poor
# — topology_parser.py needs time to recompile and deploy a new recipe.
# During cooldown, quality is still tracked and the window still updates;
# only the recompilation_recommended flag is suppressed.
_RECOMPILATION_COOLDOWN_SAMPLES: int = 25

# ── Anomaly detection ─────────────────────────────────────────────────────

# An individual extraction is flagged as an anomaly if its quality metrics
# deviate from the rolling window mean by more than this many standard
# deviations. Anomalies are logged for diagnostic purposes but do NOT
# trigger recompilation recommendations on their own — a single bad page
# should not cause a recompile. Only sustained quality degradation
# (captured by the rolling window averages) triggers recommendations.
_ANOMALY_STDEV_THRESHOLD: float = 2.5

# Minimum samples before anomaly detection activates. With fewer samples,
# the standard deviation is too noisy for meaningful anomaly detection.
_ANOMALY_MIN_SAMPLES: int = 15


# ═════════════════════════════════════════════════════════════════════════════
# PER-TOPOLOGY CONFIGURABLE THRESHOLDS
#
# Different topology classes have different expected quality ranges.
# REST_API_JSON typically produces very high token reduction (90%+) because
# the noise envelope (pagination, meta, links) is large relative to signal.
# NEWS_ARTICLE typically produces 60-80% reduction. A single set of
# thresholds would either false-positive on JSON classes (flagging normal
# high reduction as over-stripping) or miss problems on HTML classes
# (not flagging genuinely low reduction).
#
# _TopologyThresholds overrides the global thresholds for specific classes.
# Classes without a custom override use the global defaults.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _TopologyThresholds:
    """
    Quality thresholds customized for a specific topology class.

    Each field overrides the corresponding global threshold. None means
    "use the global default."
    """
    signal_density_floor:    Optional[float] = None
    signal_density_ceiling:  Optional[float] = None
    min_token_reduction:     Optional[float] = None
    max_token_reduction:     Optional[float] = None
    empty_rate_threshold:    Optional[float] = None
    min_signal_keys:         Optional[int]   = None


# Per-topology threshold overrides. Add entries here as the system learns
# what "normal" looks like for each class.
_TOPOLOGY_THRESHOLDS: Dict[str, _TopologyThresholds] = {
    # JSON APIs typically have very high reduction because the noise envelope
    # (pagination, meta, links, headers, rate_limit) is massive. A 90%
    # reduction is normal. The global ceiling of 0.95 would flag these.
    "REST_API_JSON": _TopologyThresholds(
        max_token_reduction=0.98,
        min_token_reduction=0.50,
        signal_density_floor=0.40,
        min_signal_keys=_MIN_SIGNAL_KEYS_THRESHOLD,
    ),
    # JSON-LD extraction from <head> strips the entire body — reduction
    # can be 95%+ on content-heavy pages. This is correct behavior.
    "JSON_LD_STRUCTURED": _TopologyThresholds(
        max_token_reduction=0.99,
        min_token_reduction=0.40,
        signal_density_floor=0.45,
        min_signal_keys=_MIN_SIGNAL_KEYS_THRESHOLD,
    ),
    # Ecommerce pages are typically large (images, tracking, analytics).
    # Higher reduction is expected.
    "ECOMMERCE_PRODUCT": _TopologyThresholds(
        max_token_reduction=0.95,
        min_token_reduction=0.50,
    ),
    # News articles are the canonical topology class. Global defaults
    # are calibrated for this class. No overrides needed.
    # "NEWS_ARTICLE": _TopologyThresholds(),

    # GENERIC_HTML is the fallback — maximally conservative extraction.
    # Lower density is expected because the recipe retains more noise.
    "GENERIC_HTML": _TopologyThresholds(
        signal_density_floor=0.25,
        min_token_reduction=0.20,
        # Higher empty rate is expected — GENERIC_HTML is a catch-all
        # that runs on unknown page structures, many of which produce
        # nothing useful.
        empty_rate_threshold=0.40,
    ),
}


def _get_threshold(
    topology_class: str,
    field: str,
    global_default: float,
) -> float:
    """
    Get the effective threshold for a topology class and field.

    Returns the per-topology override if one exists, otherwise the global
    default.
    """
    overrides = _TOPOLOGY_THRESHOLDS.get(topology_class)
    if overrides is None:
        return global_default

    value = getattr(overrides, field, None)
    if value is None:
        return global_default

    return value


def _get_int_threshold(
    topology_class: str,
    field: str,
    global_default: int,
) -> int:
    """Get the effective integer threshold for a topology class."""
    overrides = _TOPOLOGY_THRESHOLDS.get(topology_class)
    if overrides is None:
        return global_default
    value = getattr(overrides, field, None)
    if value is None:
        return global_default
    return value


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE TRANSITION TRACKING
#
# When topology_parser.py deploys a new recipe for a topology class, the
# quality window contains samples from both the old and new recipes. This
# is intentional — the old samples provide a baseline against which the
# new recipe's performance is evaluated.
#
# _RecipeTransition tracks each recipe hash change for a topology class,
# recording the window aggregate at the transition point. This allows
# topology_parser.py to query: "did the new recipe improve over the old?"
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _RecipeTransitionRecord:
    """
    A record of a recipe hash change for a topology class.

    Captures the quality aggregate at the time the transition was detected.
    The 'old' aggregate is the window state before the new recipe's first
    sample. The 'new' aggregate will be computed by the caller when enough
    post-transition samples have accumulated.
    """
    topology_class:     str
    old_recipe_hash:    str
    new_recipe_hash:    str
    detected_at:        datetime
    window_sample_count: int
    old_mean_reduction: float
    old_mean_density:   float
    old_empty_rate:     float


class _RecipeTransitionTracker:
    """
    Tracks recipe hash transitions per topology class.

    On every process() call, compares the current recipe hash against the
    last seen hash for this class. If they differ, a transition is recorded.

    topology_parser.py can query transitions to evaluate whether a new
    recipe improved quality.
    """

    __slots__ = ("_last_hash", "_transitions", "_max_transitions")

    def __init__(self, max_transitions_per_class: int = 20) -> None:
        self._last_hash: Dict[str, str] = {}
        self._transitions: Dict[str, Deque[_RecipeTransitionRecord]] = {}
        self._max_transitions: int = max_transitions_per_class

    def check_transition(
        self,
        topology_class: str,
        recipe_hash: str,
        window: _QualityWindow,
    ) -> Optional[_RecipeTransitionRecord]:
        """
        Check if the recipe hash has changed for this topology class.

        Returns a _RecipeTransitionRecord if a transition was detected,
        None otherwise. The first observation for a class is not a
        transition — it is an initialization.
        """
        previous_hash = self._last_hash.get(topology_class)
        self._last_hash[topology_class] = recipe_hash

        if previous_hash is None:
            # First observation — not a transition.
            return None

        if previous_hash == recipe_hash:
            # Same recipe — no transition.
            return None

        # Transition detected.
        record = _RecipeTransitionRecord(
            topology_class=topology_class,
            old_recipe_hash=previous_hash,
            new_recipe_hash=recipe_hash,
            detected_at=datetime.now(timezone.utc),
            window_sample_count=window.sample_count,
            old_mean_reduction=window.mean_token_reduction_pct(),
            old_mean_density=window.mean_signal_density(),
            old_empty_rate=window.empty_extraction_rate(),
        )

        if topology_class not in self._transitions:
            self._transitions[topology_class] = deque(
                maxlen=self._max_transitions
            )
        self._transitions[topology_class].append(record)

        return record

    def get_transitions(
        self, topology_class: str
    ) -> List[_RecipeTransitionRecord]:
        """Get the transition history for a topology class."""
        if topology_class not in self._transitions:
            return []
        return list(self._transitions[topology_class])

    def latest_transition(
        self, topology_class: str
    ) -> Optional[_RecipeTransitionRecord]:
        """Get the most recent transition for a topology class."""
        transitions = self._transitions.get(topology_class)
        if not transitions:
            return None
        return transitions[-1]


# ═════════════════════════════════════════════════════════════════════════════
# ANOMALY DETECTION
#
# Individual extraction anomaly detection. An extraction is anomalous if
# its quality metrics deviate significantly from the rolling window mean.
#
# Anomalies are NOT used for recompilation decisions — a single bad page
# should not cause a recipe recompile. They are used for:
#   - Diagnostic logging (helps identify problematic page structures)
#   - Source URL tagging (the URL that produced the anomaly is useful for
#     topology_parser.py to understand what kind of pages the recipe
#     struggles with)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _AnomalyRecord:
    """Record of an anomalous individual extraction."""
    run_id:            str
    topology_class:    str
    recipe_hash:       str
    metric:            str      # which metric was anomalous
    value:             float    # the anomalous value
    window_mean:       float    # the window mean at detection time
    window_stdev:      float    # the window stdev at detection time
    deviation_stdevs:  float    # how many stdevs from the mean
    detected_at:       datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


def _detect_anomalies(
    output: KernelOutput,
    window: _QualityWindow,
) -> List[_AnomalyRecord]:
    """
    Check if this extraction is anomalous relative to the rolling window.

    Returns a list of _AnomalyRecord for each metric that exceeds the
    anomaly threshold. Returns an empty list if no anomalies are detected
    or if there are insufficient samples for detection.

    Empty extractions are not checked for anomalies — an empty extraction
    is a known outcome type, not an anomaly in the statistical sense.
    """
    if output.extraction_empty:
        return []

    if window.sample_count < _ANOMALY_MIN_SAMPLES:
        return []

    recipe_hash = compute_recipe_hash(output.recipe_used)
    anomalies: List[_AnomalyRecord] = []

    # Check token reduction.
    mean_reduction = window.mean_token_reduction_pct()
    stdev_reduction = window.stdev_token_reduction()

    if stdev_reduction > 0.01:  # avoid division by near-zero
        deviation = abs(output.token_reduction_pct - mean_reduction) / stdev_reduction
        if deviation > _ANOMALY_STDEV_THRESHOLD:
            anomalies.append(_AnomalyRecord(
                run_id=output.run_id,
                topology_class=output.topology_class,
                recipe_hash=recipe_hash,
                metric="token_reduction_pct",
                value=output.token_reduction_pct,
                window_mean=mean_reduction,
                window_stdev=stdev_reduction,
                deviation_stdevs=round(deviation, 2),
            ))

    # Check signal density.
    mean_density = window.mean_signal_density()
    stdev_density = window.stdev_signal_density()

    if stdev_density > 0.01:
        deviation = abs(output.signal_density - mean_density) / stdev_density
        if deviation > _ANOMALY_STDEV_THRESHOLD:
            anomalies.append(_AnomalyRecord(
                run_id=output.run_id,
                topology_class=output.topology_class,
                recipe_hash=recipe_hash,
                metric="signal_density",
                value=output.signal_density,
                window_mean=mean_density,
                window_stdev=stdev_density,
                deviation_stdevs=round(deviation, 2),
            ))

    return anomalies


# ═════════════════════════════════════════════════════════════════════════════
# RECOMPILATION COOLDOWN
#
# After emitting a recompilation recommendation, feedback.py enters a
# cooldown period for that topology class. During cooldown, quality is
# still tracked and the window still updates, but the
# recompilation_recommended flag is suppressed in FeedbackEvent.
#
# This prevents recommendation spam when a recipe is consistently poor.
# topology_parser.py needs time to recompile and deploy — there is no
# value in sending 100 identical "please recompile" messages.
# ═════════════════════════════════════════════════════════════════════════════

class _CooldownTracker:
    """
    Tracks recompilation cooldown per topology class.

    After a recommendation is emitted, the class enters cooldown for
    _RECOMPILATION_COOLDOWN_SAMPLES samples. During cooldown,
    should_suppress() returns True.

    The cooldown counter decrements on every process() call for the class.
    When it reaches zero, the class exits cooldown and recommendations
    are permitted again.

    Recipe hash changes reset the cooldown — a new recipe should be
    evaluated fresh without inheriting the old recipe's cooldown.
    """

    __slots__ = ("_remaining", "_active_hash")

    def __init__(self) -> None:
        self._remaining: Dict[str, int] = {}
        self._active_hash: Dict[str, str] = {}

    def enter_cooldown(self, topology_class: str, recipe_hash: str) -> None:
        """Enter cooldown for a topology class."""
        self._remaining[topology_class] = _RECOMPILATION_COOLDOWN_SAMPLES
        self._active_hash[topology_class] = recipe_hash

    def tick(self, topology_class: str, recipe_hash: str) -> None:
        """
        Decrement the cooldown counter for a topology class.

        If the recipe hash has changed, reset the cooldown — the new
        recipe deserves fresh evaluation.
        """
        if topology_class not in self._remaining:
            return

        # Recipe hash change → reset cooldown.
        if self._active_hash.get(topology_class) != recipe_hash:
            del self._remaining[topology_class]
            if topology_class in self._active_hash:
                del self._active_hash[topology_class]
            return

        self._remaining[topology_class] -= 1
        if self._remaining[topology_class] <= 0:
            del self._remaining[topology_class]
            if topology_class in self._active_hash:
                del self._active_hash[topology_class]

    def should_suppress(self, topology_class: str) -> bool:
        """True if the class is in cooldown and recommendations should be suppressed."""
        return self._remaining.get(topology_class, 0) > 0

    def remaining_samples(self, topology_class: str) -> int:
        """Number of samples remaining in cooldown. 0 if not in cooldown."""
        return self._remaining.get(topology_class, 0)


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY WINDOW — PER-TOPOLOGY-CLASS ROLLING HISTORY
#
# The core in-memory state of feedback.py. One _QualityWindow per topology
# class. Each window is a bounded deque of QualityWindowEntry objects.
#
# Not persisted. Intentionally ephemeral. On TAG restart, all windows are
# empty. Quality history from a previous session is stale signal — worse
# than no signal. This is correct by design.
# ═════════════════════════════════════════════════════════════════════════════

class _QualityWindow:
    """
    Rolling quality window for one topology class.

    Maintains a bounded deque of QualityWindowEntry objects, bounded by
    QUALITY_WINDOW_SIZE. When the deque is full, the oldest entry is
    evicted automatically by deque.append(). No manual eviction needed.

    Provides aggregate statistics (mean reduction, mean density, empty
    rate, etc.) over the current window contents. These statistics drive
    the recompilation recommendation in FeedbackEvent.

    Also tracks recipe hash transitions — when the compiler produces a
    new recipe for this class, the window may contain samples from both
    the old and new recipes. The window does NOT flush on recipe change
    (the old samples are still valid quality history). But it does track
    which recipe hashes have contributed, and the aggregate includes a
    per-recipe breakdown when multiple hashes are present.
    """

    __slots__ = (
        "_topology_class",
        "_window",
        "_max_size",
        "_recipe_hashes_seen",
        "_total_processed",
        "_total_empty",
        "_created_at",
    )

    def __init__(
        self,
        topology_class: str,
        max_size: int = QUALITY_WINDOW_SIZE,
    ) -> None:
        self._topology_class: str = topology_class
        self._window: Deque[QualityWindowEntry] = deque(maxlen=max_size)
        self._max_size: int = max_size
        self._recipe_hashes_seen: Deque[str] = deque(
            maxlen=_MAX_RECIPE_HASHES_PER_CLASS
        )
        self._total_processed: int = 0
        self._total_empty: int = 0
        self._created_at: datetime = datetime.now(timezone.utc)

    # ── Properties ────────────────────────────────────────────────────

    @property
    def topology_class(self) -> str:
        return self._topology_class

    @property
    def sample_count(self) -> int:
        """Number of samples currently in the window."""
        return len(self._window)

    @property
    def is_empty(self) -> bool:
        return len(self._window) == 0

    @property
    def is_full(self) -> bool:
        return len(self._window) >= self._max_size

    @property
    def total_processed(self) -> int:
        """Lifetime count of samples processed (not just in window)."""
        return self._total_processed

    @property
    def total_empty(self) -> int:
        """Lifetime count of empty extractions processed."""
        return self._total_empty

    @property
    def has_sufficient_samples(self) -> bool:
        """True when the window has enough samples for meaningful statistics."""
        return self.sample_count >= _MIN_SAMPLES_FOR_RECOMMENDATION

    @property
    def most_recent_recipe_hash(self) -> Optional[str]:
        """Recipe hash of the most recently appended sample."""
        if not self._window:
            return None
        return self._window[-1].recipe_hash

    @property
    def distinct_recipe_hashes(self) -> int:
        """Number of distinct recipe hashes in the current window."""
        return len(set(e.recipe_hash for e in self._window))

    # ── Append ────────────────────────────────────────────────────────

    def append(self, entry: QualityWindowEntry) -> None:
        """
        Append a quality sample to the window.

        If the window is full, the oldest entry is silently evicted.
        This is the only mutation point for the window's contents.
        """
        self._window.append(entry)
        self._total_processed += 1

        if entry.empty_extraction:
            self._total_empty += 1

        # Track recipe hash.
        rh = entry.recipe_hash
        if rh not in self._recipe_hashes_seen:
            self._recipe_hashes_seen.append(rh)

    # ── Aggregate statistics ──────────────────────────────────────────

    def empty_extraction_rate(self) -> float:
        """
        Fraction of samples in the current window that are empty extractions.

        Returns 0.0 if the window is empty. This is a rolling rate — as
        old samples are evicted by the bounded deque, the rate naturally
        adjusts.
        """
        if not self._window:
            return 0.0
        empty_count = sum(1 for e in self._window if e.empty_extraction)
        return empty_count / len(self._window)

    def mean_token_reduction_pct(self) -> float:
        """
        Mean token reduction percentage across non-empty samples in the window.

        Empty extractions are excluded from the mean because they have
        token_reduction_pct=1.0 (all tokens are "reduced" because there
        is no output), which would skew the mean upward and mask
        under-stripping problems on pages that do produce output.

        Returns 0.0 if there are no non-empty samples.
        """
        non_empty = [e.token_reduction_pct for e in self._window if not e.empty_extraction]
        if not non_empty:
            return 0.0
        return statistics.mean(non_empty)

    def mean_signal_density(self) -> float:
        """
        Mean signal density across non-empty samples in the window.

        Empty extractions are excluded (density=0.0 would skew the mean).
        Returns 0.0 if there are no non-empty samples.
        """
        non_empty = [e.signal_density for e in self._window if not e.empty_extraction]
        if not non_empty:
            return 0.0
        return statistics.mean(non_empty)

    def mean_latency_ms(self) -> float:
        """Mean latency across all samples in the window."""
        if not self._window:
            return 0.0
        return statistics.mean(e.latency_ms for e in self._window)

    def stdev_token_reduction(self) -> float:
        """
        Standard deviation of token reduction among non-empty samples.

        High standard deviation indicates inconsistent recipe performance
        across different pages in the same topology class. This can happen
        when the topology class is too broad (lumping different page
        structures under one class) or when the recipe has edge-case bugs.

        Returns 0.0 if fewer than 2 non-empty samples.
        """
        non_empty = [e.token_reduction_pct for e in self._window if not e.empty_extraction]
        if len(non_empty) < 2:
            return 0.0
        return statistics.stdev(non_empty)

    def stdev_signal_density(self) -> float:
        """Standard deviation of signal density among non-empty samples."""
        non_empty = [e.signal_density for e in self._window if not e.empty_extraction]
        if len(non_empty) < 2:
            return 0.0
        return statistics.stdev(non_empty)

    def recent_mean_token_reduction(self, n: int = _TREND_RECENT_WINDOW) -> float:
        """Mean token reduction of the N most recent non-empty samples."""
        recent = list(self._window)[-n:]
        non_empty = [e.token_reduction_pct for e in recent if not e.empty_extraction]
        if not non_empty:
            return 0.0
        return statistics.mean(non_empty)

    def recent_mean_signal_density(self, n: int = _TREND_RECENT_WINDOW) -> float:
        """Mean signal density of the N most recent non-empty samples."""
        recent = list(self._window)[-n:]
        non_empty = [e.signal_density for e in recent if not e.empty_extraction]
        if not non_empty:
            return 0.0
        return statistics.mean(non_empty)

    def recent_empty_rate(self, n: int = _TREND_RECENT_WINDOW) -> float:
        """Empty extraction rate over the N most recent samples."""
        recent = list(self._window)[-n:]
        if not recent:
            return 0.0
        empty_count = sum(1 for e in recent if e.empty_extraction)
        return empty_count / len(recent)

    def per_recipe_breakdown(self) -> Dict[str, Dict[str, object]]:
        """
        Quality breakdown per recipe hash in the current window.

        Used for diagnostic logging when multiple recipe hashes are present
        in the same window (during recipe transitions). Shows whether the
        new recipe is performing better or worse than the old one.
        """
        by_hash: Dict[str, List[QualityWindowEntry]] = {}
        for entry in self._window:
            rh = entry.recipe_hash
            if rh not in by_hash:
                by_hash[rh] = []
            by_hash[rh].append(entry)

        result: Dict[str, Dict[str, object]] = {}
        for rh, entries in by_hash.items():
            non_empty = [e for e in entries if not e.empty_extraction]
            result[rh[:8]] = {
                "sample_count": len(entries),
                "empty_count": len(entries) - len(non_empty),
                "mean_reduction": (
                    round(statistics.mean(e.token_reduction_pct for e in non_empty), 4)
                    if non_empty else 0.0
                ),
                "mean_density": (
                    round(statistics.mean(e.signal_density for e in non_empty), 4)
                    if non_empty else 0.0
                ),
            }

        return result

    # ── Snapshot ──────────────────────────────────────────────────────

    def snapshot(self) -> List[QualityWindowEntry]:
        """
        Return a copy of the window contents as a list.

        The list is ordered oldest → newest. Callers should not hold
        references to the list across process() calls — the window
        mutates on every append.
        """
        return list(self._window)

    def to_diagnostic_dict(self) -> Dict[str, object]:
        """Full diagnostic snapshot for structured logging."""
        return {
            "topology_class":        self._topology_class,
            "sample_count":          self.sample_count,
            "max_size":              self._max_size,
            "total_processed":       self._total_processed,
            "total_empty":           self._total_empty,
            "empty_rate":            round(self.empty_extraction_rate(), 4),
            "mean_reduction":        round(self.mean_token_reduction_pct(), 4),
            "mean_density":          round(self.mean_signal_density(), 4),
            "mean_latency_ms":       round(self.mean_latency_ms(), 2),
            "stdev_reduction":       round(self.stdev_token_reduction(), 4),
            "stdev_density":         round(self.stdev_signal_density(), 4),
            "distinct_hashes":       self.distinct_recipe_hashes,
            "has_sufficient":        self.has_sufficient_samples,
            "created_at":            self._created_at.isoformat(),
        }


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK STATE — MODULE-LEVEL SINGLETONS
#
# These persist across invocations within the same Python process.
# On process restart, they are empty. This is correct — stale quality
# history from a previous session is worse than no history.
# ═════════════════════════════════════════════════════════════════════════════

# Per-topology-class quality windows. Lazily created on first observation.
_windows: Dict[str, _QualityWindow] = {}

# Recipe transition tracker. Detects recipe hash changes per class.
_transition_tracker: _RecipeTransitionTracker = _RecipeTransitionTracker()

# Recompilation cooldown tracker. Prevents recommendation spam.
_cooldown: _CooldownTracker = _CooldownTracker()

# The registered feedback handler. topology_parser.py calls
# register_feedback_handler() once on startup to provide this callable.
#
# _feedback_handler is NEVER None. When no real handler is registered it
# holds _null_handler — a no-op sentinel that logs and discards the event.
# This eliminates the entire class of 'None is not callable' errors:
# the slot is always a valid callable, so handler(event) is unconditionally
# safe without a preceding None guard.

def _null_handler(event: FeedbackEvent) -> None:
    """No-op sentinel used before register_feedback_handler() is called."""
    logger.debug(
        "No feedback handler registered. FeedbackEvent for "
        "topology=%s run_id=%s discarded.",
        event.topology_class,
        event.run_id,
    )


_feedback_handler: Callable[[FeedbackEvent], None] = _null_handler

# Consecutive emission failure counter. When this exceeds
# _MAX_CONSECUTIVE_EMISSION_FAILURES, the handler is presumed dead.
_consecutive_emission_failures: int = 0

# Lifetime counters for diagnostic logging.
_total_events_emitted: int = 0
_total_events_failed: int = 0
_total_events_suppressed: int = 0


def _get_or_create_window(topology_class: str) -> _QualityWindow:
    """
    Get the quality window for a topology class, creating it lazily.

    New windows start empty. They fill as process() is called with
    KernelOutput for that class.
    """
    if topology_class not in _windows:
        _windows[topology_class] = _QualityWindow(topology_class)
        logger.debug(
            "Created quality window for topology class: %s",
            topology_class,
        )
    return _windows[topology_class]


# ═════════════════════════════════════════════════════════════════════════════
# HANDLER REGISTRATION
# ═════════════════════════════════════════════════════════════════════════════

def register_feedback_handler(
    handler: Callable[[FeedbackEvent], None],
) -> None:
    """
    Register the callback that receives FeedbackEvent emissions.

    Called once by topology_parser.py on startup. After registration,
    every process() invocation that computes a FeedbackEvent will deliver
    it through this handler.

    The handler must be a callable that accepts a single FeedbackEvent
    argument and returns None. It should not raise — if it does, the
    exception is caught, logged, and counted. After too many consecutive
    failures, the handler is presumed dead and emission becomes a no-op.

    Calling register_feedback_handler() again replaces the previous
    handler and resets the consecutive failure counter. This allows
    topology_parser.py to re-register after a restart without requiring
    feedback.py to restart.

    Parameters
    ----------
    handler : Callable[[FeedbackEvent], None]
        The callback function that receives FeedbackEvents.
    """
    global _feedback_handler, _consecutive_emission_failures

    if not callable(handler):
        raise TypeError( # noqa | defensive runtime check
            f"register_feedback_handler: handler must be callable, "
            f"got {type(handler).__name__!r}.  "
            "topology_parser.py must pass a function or callable object."
        )

    _feedback_handler = handler
    _consecutive_emission_failures = 0

    logger.info(
        "Feedback handler registered: %s.%s",
        handler.__module__ if hasattr(handler, "__module__") else "?",
        handler.__qualname__ if hasattr(handler, "__qualname__") else "?",
    )


def unregister_feedback_handler() -> None:
    """
    Remove the registered feedback handler.

    After this call, FeedbackEvent emission is a no-op. Quality
    computation and window updates continue — only emission is disabled.
    """
    global _feedback_handler
    _feedback_handler = _null_handler  # never None — always a valid callable
    logger.info("Feedback handler unregistered. Emission is now a no-op.")


# ═════════════════════════════════════════════════════════════════════════════
# JSON STRUCTURED FIELD ANALYSIS
#
# For JSON topology classes only. HTML topology classes always get
# structured_field_count=0.
#
# feedback.py scans the clean_signal for expected JSON keys and counts
# how many appeared. Missing keys indicate the recipe is under-extracting.
# Present noise keys indicate the recipe is under-stripping.
# ═════════════════════════════════════════════════════════════════════════════

def _count_signal_keys(clean_signal: str, topology_class: str) -> int:
    """
    Count the number of expected signal keys present in clean_signal.

    Scans for key patterns that match the expected signal keys for this
    topology class. The scan is string-based — it does not parse JSON,
    because the clean_signal may contain multiple JSON fragments separated
    by blank lines (e.g. JSON-LD blocks) and may not be valid JSON as a
    whole.

    The scan looks for patterns like:
      "keyname"       (JSON key in double quotes)
      keyname:        (awk-extracted label: value pairs)
      @type           (JSON-LD @-prefixed keys)

    Returns the count of distinct expected signal keys found. This count
    feeds ExtractionQuality.structured_field_count.
    """
    expected = _EXPECTED_SIGNAL_KEYS.get(topology_class)
    if expected is None:
        return 0

    found_count = 0
    signal_lower = clean_signal.lower()

    for key in expected:
        key_lower = key.lower()

        # Pattern 1: JSON key in double quotes — "keyname" :
        if f'"{key_lower}"' in signal_lower:
            found_count += 1
            continue

        # Pattern 2: bare key followed by colon (awk output format)
        # e.g. "name: Product Name"
        if f"{key_lower}:" in signal_lower:
            found_count += 1
            continue

        # Pattern 3: JSON-LD @-prefixed keys without quotes
        if key.startswith("@") and key_lower in signal_lower:
            found_count += 1
            continue

    return found_count


def _count_noise_keys(clean_signal: str, topology_class: str) -> int:
    """
    Count the number of noise keys still present in clean_signal.

    These keys should have been stripped by the recipe. Their presence
    indicates the recipe is under-stripping — it failed to discard
    envelope metadata, pagination, tracking, or CMS-injected noise.

    Returns the count of noise keys found. Used as a quality penalty
    in the composite score and in the recompilation recommendation.
    """
    noise_keys = _EXPECTED_NOISE_KEYS.get(topology_class)
    if noise_keys is None:
        return 0

    found_count = 0
    signal_lower = clean_signal.lower()

    for key in noise_keys:
        key_lower = key.lower()
        if f'"{key_lower}"' in signal_lower or f"{key_lower}:" in signal_lower:
            found_count += 1

    return found_count


def _compute_structured_field_count(
    output: KernelOutput,
) -> int:
    """
    Compute the structured_field_count for ExtractionQuality.

    For JSON topology classes: counts expected signal keys present in
    clean_signal. This is the positive signal — how many expected fields
    did the recipe successfully extract?

    For HTML topology classes: always returns 0. HTML extraction quality
    is measured by token_reduction_pct and signal_density, not by field
    presence.

    If the extraction is empty, returns 0 regardless of topology class.
    """
    if output.extraction_empty:
        return 0

    if output.topology_class not in _JSON_TOPOLOGY_CLASSES:
        return 0

    return _count_signal_keys(output.clean_signal, output.topology_class)


# ═════════════════════════════════════════════════════════════════════════════
# JSON QUALITY ANALYSIS — DEEPER STRUCTURAL CHECKS
#
# Beyond simple key counting, feedback.py performs structural analysis of
# JSON output quality. These checks catch subtle quality issues:
#   - Truncated JSON (unbalanced braces)
#   - Excessively nested noise (depth > expected for this topology)
#   - Duplicate key sections (recipe extracted the same section twice)
#   - Empty value fields ("key": "" or "key": null)
# ═════════════════════════════════════════════════════════════════════════════

def _check_json_structural_integrity(
    clean_signal: str,
    topology_class: str,
) -> Dict[str, object]:
    """
    Structural integrity analysis of JSON clean_signal.

    Returns a dict with diagnostic fields:
      - balanced_braces:    True if { and } counts match
      - balanced_brackets:  True if [ and ] counts match
      - noise_keys_present: count of noise keys still in output
      - empty_values:       count of empty string or null values
      - signal_key_ratio:   signal_keys_found / total_expected_keys
      - structural_healthy: True if all checks pass

    These diagnostics feed the recompilation recommendation for JSON
    topology classes. A recipe that produces structurally broken JSON
    output needs immediate attention even if the byte-level metrics
    (reduction, density) look normal.
    """
    open_braces = clean_signal.count("{")
    close_braces = clean_signal.count("}")
    open_brackets = clean_signal.count("[")
    close_brackets = clean_signal.count("]")

    balanced_braces = open_braces == close_braces
    balanced_brackets = open_brackets == close_brackets

    noise_count = _count_noise_keys(clean_signal, topology_class)
    signal_count = _count_signal_keys(clean_signal, topology_class)

    expected_keys = _EXPECTED_SIGNAL_KEYS.get(topology_class, frozenset())
    total_expected = len(expected_keys) if expected_keys else 1

    # Count empty values: "key": "" or "key": null
    empty_string_pattern = re.compile(r'"\w+":\s*""')
    null_pattern = re.compile(r'"\w+":\s*null\b')
    empty_values = (
        len(empty_string_pattern.findall(clean_signal))
        + len(null_pattern.findall(clean_signal))
    )

    signal_key_ratio = signal_count / total_expected if total_expected > 0 else 0.0

    structural_healthy = (
        balanced_braces
        and balanced_brackets
        and noise_count == 0
        and signal_key_ratio >= 0.3
    )

    return {
        "balanced_braces":    balanced_braces,
        "balanced_brackets":  balanced_brackets,
        "noise_keys_present": noise_count,
        "empty_values":       empty_values,
        "signal_count":       signal_count,
        "signal_key_ratio":   round(signal_key_ratio, 4),
        "structural_healthy": structural_healthy,
    }


# ═════════════════════════════════════════════════════════════════════════════
# RECOMPILATION RECOMMENDATION
#
# The decision engine that determines whether to recommend recipe
# recompilation for a given topology class. Uses the rolling window
# statistics and the current extraction's quality metrics.
#
# The recommendation is advisory — feedback.py does not invoke the
# compiler. It emits FeedbackEvent.recompilation_recommended=True with
# a human-readable reason string. topology_parser.py decides whether
# and when to act on the recommendation.
# ═════════════════════════════════════════════════════════════════════════════

def _evaluate_recompilation(
    quality: ExtractionQuality, # noqa
    window: _QualityWindow,
    json_diagnostics: Optional[Dict[str, object]],
) -> Tuple[bool, Optional[str]]:
    """
    Evaluate whether recompilation should be recommended.

    Decision rules (in priority order — first match wins):

    1. INSUFFICIENT SAMPLES — if the window has fewer than
       _MIN_SAMPLES_FOR_RECOMMENDATION, do NOT recommend. The statistics
       are too noisy to trust. Return (False, None).

    2. EMPTY RATE EXCEEDED — if the rolling empty extraction rate exceeds
       _EMPTY_RATE_RECOMPILATION_THRESHOLD, recommend. The recipe is
       consistently failing to extract content.

    3. DENSITY FLOOR BREACHED — if the rolling mean signal density is below
       _DENSITY_RECOMPILATION_THRESHOLD, recommend. The recipe is producing
       mostly whitespace.

    4. REDUCTION FLOOR BREACHED — if the rolling mean token reduction is
       below _REDUCTION_FLOOR_THRESHOLD, recommend. The recipe is not
       stripping enough noise.

    5. REDUCTION CEILING BREACHED — if the rolling mean token reduction
       exceeds _REDUCTION_CEILING_THRESHOLD, recommend. The recipe is
       over-stripping and may be discarding signal.

    6. JSON STRUCTURAL FAILURE — for JSON topology classes, if the
       structural integrity check failed, recommend.

    7. TREND DEGRADATION — if the recent subset of the window shows
       notably worse metrics than the full window, recommend. The recipe
       is degrading over time.

    8. HIGH VARIANCE — if the standard deviation of token reduction
       exceeds 0.2, the recipe is inconsistent. Recommend review.

    9. OTHERWISE — do not recommend. Return (False, None).

    Returns (recommended: bool, reason: Optional[str]).
    When recommended=True, reason is a human-readable string explaining
    which rule triggered the recommendation. When False, reason is None.
    """
    # ── Rule 1: Insufficient samples ──────────────────────────────────
    if not window.has_sufficient_samples:
        return False, None

    reasons: List[str] = []

    # ── Rule 2: Empty rate ────────────────────────────────────────────
    empty_rate = window.empty_extraction_rate()
    eff_empty_threshold = _get_threshold(
        window.topology_class, "empty_rate_threshold",
        _EMPTY_RATE_RECOMPILATION_THRESHOLD,
    )
    if empty_rate > eff_empty_threshold:
        reasons.append(
            f"empty_rate={empty_rate:.2%} > threshold={eff_empty_threshold:.2%} "
            f"over {window.sample_count} samples"
        )

    # ── Rule 3: Density floor ─────────────────────────────────────────
    mean_density = window.mean_signal_density()
    eff_density_floor = _get_threshold(
        window.topology_class, "signal_density_floor",
        _DENSITY_RECOMPILATION_THRESHOLD,
    )
    if 0 < mean_density < eff_density_floor:
        reasons.append(
            f"mean_density={mean_density:.4f} < floor={eff_density_floor}"
        )

    # ── Rule 4: Reduction floor ───────────────────────────────────────
    mean_reduction = window.mean_token_reduction_pct()
    eff_reduction_floor = _get_threshold(
        window.topology_class, "min_token_reduction",
        _REDUCTION_FLOOR_THRESHOLD,
    )
    if 0 < mean_reduction < eff_reduction_floor:
        reasons.append(
            f"mean_reduction={mean_reduction:.4f} < floor={eff_reduction_floor}"
        )

    # ── Rule 5: Reduction ceiling ─────────────────────────────────────
    eff_reduction_ceiling = _get_threshold(
        window.topology_class, "max_token_reduction",
        _REDUCTION_CEILING_THRESHOLD,
    )
    if mean_reduction > eff_reduction_ceiling:
        reasons.append(
            f"mean_reduction={mean_reduction:.4f} > ceiling={eff_reduction_ceiling}"
        )

    # ── Rule 6: JSON structural failure ───────────────────────────────
    if json_diagnostics is not None and not json_diagnostics.get("structural_healthy", True):
        noise_present = json_diagnostics.get("noise_keys_present", 0)
        signal_ratio = json_diagnostics.get("signal_key_ratio", 0)
        balanced = json_diagnostics.get("balanced_braces", True)
        reasons.append(
            f"JSON structural failure: noise_keys={noise_present} "
            f"signal_ratio={signal_ratio} balanced_braces={balanced}"
        )

    # ── Rule 7: Trend degradation ─────────────────────────────────────
    if window.sample_count >= _TREND_RECENT_WINDOW * 2:
        recent_reduction = window.recent_mean_token_reduction()
        full_reduction = window.mean_token_reduction_pct()

        if full_reduction > 0 and recent_reduction > 0:
            reduction_degradation = (full_reduction - recent_reduction) / full_reduction
            if reduction_degradation > _TREND_DEGRADATION_FACTOR:
                reasons.append(
                    f"reduction degrading: recent={recent_reduction:.4f} "
                    f"vs full={full_reduction:.4f} "
                    f"({reduction_degradation:.1%} worse)"
                )

        recent_density = window.recent_mean_signal_density()
        full_density = window.mean_signal_density()

        if full_density > 0 and recent_density > 0:
            density_degradation = (full_density - recent_density) / full_density
            if density_degradation > _TREND_DEGRADATION_FACTOR:
                reasons.append(
                    f"density degrading: recent={recent_density:.4f} "
                    f"vs full={full_density:.4f} "
                    f"({density_degradation:.1%} worse)"
                )

        recent_empty = window.recent_empty_rate()
        full_empty = window.empty_extraction_rate()

        if recent_empty > full_empty + 0.10:
            reasons.append(
                f"empty rate rising: recent={recent_empty:.2%} "
                f"vs full={full_empty:.2%}"
            )

    # ── Rule 8: High variance ─────────────────────────────────────────
    stdev_reduction = window.stdev_token_reduction()
    if stdev_reduction > 0.20:
        reasons.append(
            f"high variance: stdev_reduction={stdev_reduction:.4f} > 0.20. "
            "Inconsistent extraction quality across pages."
        )

    # ── Assemble recommendation ───────────────────────────────────────
    if reasons:
        combined_reason = "; ".join(reasons)
        return True, combined_reason

    return False, None


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK EVENT EMISSION
# ═════════════════════════════════════════════════════════════════════════════

def _emit_feedback_event(event: FeedbackEvent) -> None:
    """
    Deliver a FeedbackEvent through the registered handler.

    If no handler is registered, the event is logged at DEBUG level and
    discarded. feedback.py never fails because topology_parser.py is
    unavailable.

    If the handler raises, the exception is caught, logged, and counted.
    After _MAX_CONSECUTIVE_EMISSION_FAILURES consecutive failures, the
    handler is presumed dead and emission becomes a no-op with a periodic
    warning log.
    """
    global _consecutive_emission_failures, _total_events_emitted
    global _total_events_failed, _total_events_suppressed

    # Snapshot the handler into a local. _feedback_handler is never None
    # (it holds _null_handler when unregistered) so this call is always
    # safe. The local copy is still good practice for thread safety: a
    # concurrent re-registration cannot swap the callable mid-call.
    handler = _feedback_handler

    # Check if we have exceeded the consecutive failure limit.
    if _consecutive_emission_failures >= _MAX_CONSECUTIVE_EMISSION_FAILURES:
        _total_events_suppressed += 1
        # Log periodically — every 100th suppressed event.
        if _total_events_suppressed % 100 == 1:
            logger.warning(
                "Feedback handler presumed dead after %d consecutive failures. "
                "Emission suppressed. total_suppressed=%d. "
                "Call register_feedback_handler() to reset.",
                _MAX_CONSECUTIVE_EMISSION_FAILURES,
                _total_events_suppressed,
            )
        return

    try:
        handler(event)
        _consecutive_emission_failures = 0
        _total_events_emitted += 1
    except Exception as exc:
        _consecutive_emission_failures += 1
        _total_events_failed += 1

        logger.warning(
            "FeedbackEvent emission failed (%d/%d consecutive): %s. "
            "topology=%s run_id=%s quality_score=%.4f",
            _consecutive_emission_failures,
            _MAX_CONSECUTIVE_EMISSION_FAILURES,
            exc,
            event.topology_class,
            event.run_id,
            event.quality.composite_score,
        )


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY WINDOW ENTRY CONSTRUCTION
# ═════════════════════════════════════════════════════════════════════════════

def _build_window_entry(output: KernelOutput) -> QualityWindowEntry:
    """
    Construct a QualityWindowEntry from a KernelOutput.

    This is the single construction path for window entries. It extracts
    the fields feedback.py tracks in the rolling window from the
    KernelOutput contract.
    """
    return QualityWindowEntry(
        run_id=output.run_id,
        recipe_hash=compute_recipe_hash(output.recipe_used),
        token_reduction_pct=output.token_reduction_pct,
        signal_density=output.signal_density,
        empty_extraction=output.extraction_empty,
        latency_ms=output.latency_ms,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — process()
#
# The only public function that processes KernelOutput. Called once per
# extraction by TAG's Python layer after pipeline.py returns.
# ═════════════════════════════════════════════════════════════════════════════

def process(output: KernelOutput) -> Optional[FeedbackEvent]:
    """
    Process one KernelOutput and emit a FeedbackEvent.

    This is the complete feedback pipeline:
      1. Compute structured field count (JSON topologies only)
      2. Build ExtractionQuality via make_quality_from_output()
      3. Build QualityWindowEntry and append to rolling window
      4. Evaluate recompilation recommendation
      5. Construct FeedbackEvent
      6. Emit FeedbackEvent through registered handler
      7. Log the quality observation

    Returns the FeedbackEvent for callers that want to inspect it
    (e.g. test harnesses). In production, the handler receives the
    event — the return value is informational.

    Returns None if FeedbackEvent construction fails. This should not
    happen in a correct system — it indicates a contract invariant
    violation in contracts.py. Logged at ERROR level.

    Never raises. feedback.py never interrupts the extraction cycle.
    If any step fails, it is logged and the function returns None.
    The extraction result is not affected.

    Parameters
    ----------
    output : KernelOutput
        The extraction result from pipeline.py. Must be a fully
        constructed KernelOutput with all fields valid.
    """
    try:
        return _process_impl(output)
    except Exception as exc:
        logger.error(
            "Unexpected error in feedback.process(): %s: %s. "
            "run_id=%s topology=%s. "
            "Feedback for this extraction is lost. "
            "The extraction result is not affected.",
            type(exc).__name__,
            exc,
            output.run_id,
            output.topology_class,
            exc_info=True,
        )
        return None


def _process_impl(output: KernelOutput) -> Optional[FeedbackEvent]:
    """
    Internal implementation of process(). Separated so that the outer
    process() can provide the universal try/except wrapper without
    nesting inside the implementation logic.
    """
    # ── Step 1: Compute structured field count ────────────────────────
    structured_field_count = _compute_structured_field_count(output)

    # ── Step 2: Build ExtractionQuality ───────────────────────────────
    try:
        quality = make_quality_from_output(
            output,
            structured_field_count=structured_field_count,
        )
    except (ValueError, TypeError) as exc:
        logger.error(
            "Failed to construct ExtractionQuality: %s. "
            "run_id=%s topology=%s. "
            "Using fallback quality with zero scores.",
            exc,
            output.run_id,
            output.topology_class,
        )
        return None

    # ── Step 3: Update rolling window ─────────────────────────────────
    window = _get_or_create_window(output.topology_class)
    entry = _build_window_entry(output)
    window.append(entry)

    recipe_hash = compute_recipe_hash(output.recipe_used)

    # ── Step 3a: Recipe transition detection ──────────────────────────
    transition = _transition_tracker.check_transition(
        output.topology_class,
        recipe_hash,
        window,
    )

    if transition is not None:
        logger.info(
            "RECIPE TRANSITION: topology=%s old=%s → new=%s "
            "window_samples=%d old_reduction=%.4f old_density=%.4f old_empty_rate=%.4f",
            transition.topology_class,
            transition.old_recipe_hash[:8],
            transition.new_recipe_hash[:8],
            transition.window_sample_count,
            transition.old_mean_reduction,
            transition.old_mean_density,
            transition.old_empty_rate,
        )

    # ── Step 3b: Cooldown tick ────────────────────────────────────────
    _cooldown.tick(output.topology_class, recipe_hash)

    # ── Step 4: JSON structural analysis (JSON topologies only) ───────
    json_diagnostics: Optional[Dict[str, object]] = None
    if (
        output.topology_class in _JSON_TOPOLOGY_CLASSES
        and not output.extraction_empty
    ):
        json_diagnostics = _check_json_structural_integrity(
            output.clean_signal,
            output.topology_class,
        )

        if not json_diagnostics.get("structural_healthy", True):
            logger.info(
                "JSON structural issue: topology=%s run_id=%s "
                "diagnostics=%s",
                output.topology_class,
                output.run_id,
                json.dumps(json_diagnostics, default=str),
            )

    # ── Step 4a: Anomaly detection ────────────────────────────────────
    anomalies = _detect_anomalies(output, window)
    if anomalies:
        for anomaly in anomalies:
            logger.info(
                "ANOMALY detected: topology=%s run=%s metric=%s "
                "value=%.4f mean=%.4f stdev=%.4f deviation=%.1f σ",
                anomaly.topology_class,
                anomaly.run_id[:8],
                anomaly.metric,
                anomaly.value,
                anomaly.window_mean,
                anomaly.window_stdev,
                anomaly.deviation_stdevs,
            )

    # ── Step 5: Evaluate recompilation recommendation ─────────────────
    recommended, reason = _evaluate_recompilation(
        quality, window, json_diagnostics,
    )

    # ── Step 5a: Apply cooldown suppression ───────────────────────────
    if recommended and _cooldown.should_suppress(output.topology_class):
        remaining = _cooldown.remaining_samples(output.topology_class)
        logger.debug(
            "Recompilation recommendation suppressed by cooldown: "
            "topology=%s remaining=%d",
            output.topology_class,
            remaining,
        )
        recommended = False
        reason = None

    # If recommendation passes cooldown, enter cooldown for next period.
    if recommended:
        _cooldown.enter_cooldown(output.topology_class, recipe_hash)

    # ── Step 6: Construct FeedbackEvent ───────────────────────────────
    try:
        event = FeedbackEvent(
            run_id=output.run_id,
            topology_class=output.topology_class,
            recipe_hash=compute_recipe_hash(output.recipe_used),
            quality=quality,
            recompilation_recommended=recommended,
            recompilation_reason=reason,
            window_empty_extraction_rate=window.empty_extraction_rate(),
            window_sample_count=window.sample_count,
        )
    except (ValueError, TypeError) as exc:
        logger.error(
            "Failed to construct FeedbackEvent: %s. "
            "run_id=%s topology=%s. "
            "Quality was computed but event emission is lost.",
            exc,
            output.run_id,
            output.topology_class,
        )
        return None

    # ── Step 7: Emit ──────────────────────────────────────────────────
    _emit_feedback_event(event)

    # ── Step 8: Log ───────────────────────────────────────────────────
    _log_quality_observation(output, quality, window, event, json_diagnostics)

    return event


# ═════════════════════════════════════════════════════════════════════════════
# QUALITY OBSERVATION LOGGING
#
# Structured logging of every quality observation. Feeds Witness and is
# the primary diagnostic tool for recipe quality analysis.
# ═════════════════════════════════════════════════════════════════════════════

def _log_quality_observation(
    output: KernelOutput,
    quality: ExtractionQuality,
    window: _QualityWindow,
    event: FeedbackEvent,
    json_diagnostics: Optional[Dict[str, object]],
) -> None:
    """
    Log a structured quality observation.

    Logging levels:
      - Recompilation recommended → WARNING (requires attention)
      - Empty extraction → INFO (informational, not alarming individually)
      - Normal quality → DEBUG (routine observation)
    """
    log_data = {
        "run_id":           output.run_id,
        "topology_class":   output.topology_class,
        "recipe_hash":      compute_recipe_hash(output.recipe_used)[:8],
        "extraction_empty": output.extraction_empty,
        "reduction_pct":    round(quality.token_reduction_pct, 4),
        "signal_density":   round(quality.signal_density, 4),
        "composite_score":  quality.composite_score,
        "structured_fields": quality.structured_field_count,
        "window_samples":   window.sample_count,
        "window_empty_rate": round(window.empty_extraction_rate(), 4),
        "recompilation":    event.recompilation_recommended,
    }

    if json_diagnostics is not None:
        log_data["json_diagnostics"] = json_diagnostics

    if event.recompilation_recommended:
        logger.warning(
            "FEEDBACK RECOMPILATION_RECOMMENDED: topology=%s reason=%s "
            "quality=%s",
            output.topology_class,
            event.recompilation_reason,
            json.dumps(log_data, default=str),
        )
    elif output.extraction_empty:
        logger.info(
            "FEEDBACK empty: topology=%s run=%s window_empty_rate=%.2f%%",
            output.topology_class,
            output.run_id[:8],
            window.empty_extraction_rate() * 100,
        )
    else:
        logger.debug(
            "FEEDBACK quality: %s",
            json.dumps(log_data, default=str),
        )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC QUERY API
#
# topology_parser.py queries these to read aggregate quality stats for
# a topology class. These are read-only views — they do not modify the
# rolling windows.
# ═════════════════════════════════════════════════════════════════════════════

def get_quality_aggregate(
    topology_class: str,
) -> Optional[RecipeQualityAggregate]:
    """
    Compute a RecipeQualityAggregate for the given topology class.

    Returns None if no quality window exists for this class (no samples
    have been observed yet).

    The aggregate is computed fresh on every call from the current window
    contents. It is not cached. The window may change between calls.

    topology_parser.py uses this to decide whether to trigger recompilation.
    """
    if topology_class not in _windows:
        return None

    window = _windows[topology_class]

    if window.is_empty:
        return None

    most_recent_hash = window.most_recent_recipe_hash
    if most_recent_hash is None:
        return None

    mean_reduction = window.mean_token_reduction_pct()
    mean_density = window.mean_signal_density()
    empty_rate = window.empty_extraction_rate()
    mean_latency = window.mean_latency_ms()

    # Determine if the aggregate warrants a review flag.
    flagged = (
        empty_rate > _EMPTY_RATE_RECOMPILATION_THRESHOLD
        or (0 < mean_density < _DENSITY_RECOMPILATION_THRESHOLD)
        or (0 < mean_reduction < _REDUCTION_FLOOR_THRESHOLD)
        or mean_reduction > _REDUCTION_CEILING_THRESHOLD
    )

    try:
        return RecipeQualityAggregate(
            topology_class=TopologyClassStr(topology_class),
            recipe_hash=RecipeHash(most_recent_hash),
            sample_count=window.sample_count,
            mean_token_reduction_pct=mean_reduction,
            mean_signal_density=mean_density,
            empty_extraction_rate=empty_rate,
            mean_latency_ms=mean_latency,
            flagged_for_review=flagged,
        )
    except (ValueError, TypeError) as exc:
        logger.error(
            "Failed to construct RecipeQualityAggregate: %s. "
            "topology=%s",
            exc,
            topology_class,
        )
        return None


def get_window_snapshot(
    topology_class: str,
) -> List[QualityWindowEntry]:
    """
    Return a copy of the quality window contents for a topology class.

    Returns an empty list if no window exists. The list is ordered
    oldest → newest.
    """
    if topology_class not in _windows:
        return []
    return _windows[topology_class].snapshot()


def get_all_topology_classes() -> List[str]:
    """
    Return a list of all topology classes that have quality windows.

    Useful for topology_parser.py to enumerate which classes have
    active quality tracking.
    """
    return sorted(_windows.keys())


def get_window_diagnostics(
    topology_class: str,
) -> Optional[Dict[str, object]]:
    """
    Return the full diagnostic dict for a topology class window.

    Returns None if no window exists.
    """
    if topology_class not in _windows:
        return None
    return _windows[topology_class].to_diagnostic_dict()


def get_all_diagnostics() -> Dict[str, object]:
    """
    Return the complete diagnostic state of the feedback system.

    Includes per-class window diagnostics, emission statistics, handler
    status, cooldown state, and transition history. Consumed by TAG's
    telemetry layer for Witness.
    """
    return {
        "handler_registered": _feedback_handler is not _null_handler,
        "consecutive_failures": _consecutive_emission_failures,
        "total_events_emitted": _total_events_emitted,
        "total_events_failed": _total_events_failed,
        "total_events_suppressed": _total_events_suppressed,
        "tracked_classes": len(_windows),
        "per_class": {
            tc: {
                **window.to_diagnostic_dict(),
                "cooldown_remaining": _cooldown.remaining_samples(tc),
                "transitions": len(_transition_tracker.get_transitions(tc)),
            }
            for tc, window in sorted(_windows.items())
        },
    }


def get_recipe_transitions(
    topology_class: str,
) -> List[Dict[str, object]]:
    """
    Return the recipe transition history for a topology class.

    topology_parser.py uses this to evaluate whether new recipes are
    improving over old ones.
    """
    transitions = _transition_tracker.get_transitions(topology_class)
    return [
        { # noqa
            "old_hash":          t.old_recipe_hash[:8],
            "new_hash":          t.new_recipe_hash[:8],
            "detected_at":       t.detected_at.isoformat(),
            "window_samples":    t.window_sample_count,
            "old_mean_reduction": round(t.old_mean_reduction, 4),
            "old_mean_density":  round(t.old_mean_density, 4),
            "old_empty_rate":    round(t.old_empty_rate, 4),
        }
        for t in transitions
    ]


def get_latest_transition(
    topology_class: str,
) -> Optional[Dict[str, object]]:
    """
    Return the most recent recipe transition for a topology class.

    Returns None if no transitions have been recorded.
    """
    t = _transition_tracker.latest_transition(topology_class)
    if t is None:
        return None
    return { # noqa
        "old_hash":          t.old_recipe_hash[:8],
        "new_hash":          t.new_recipe_hash[:8],
        "detected_at":       t.detected_at.isoformat(),
        "window_samples":    t.window_sample_count,
        "old_mean_reduction": round(t.old_mean_reduction, 4),
        "old_mean_density":  round(t.old_mean_density, 4),
        "old_empty_rate":    round(t.old_empty_rate, 4),
    }


# ═════════════════════════════════════════════════════════════════════════════
# RESET — TEST HARNESS ONLY
# ═════════════════════════════════════════════════════════════════════════════

def reset() -> None:
    """
    Reset all feedback state to initial empty values.

    Clears all quality windows, unregisters the handler, and resets
    all counters. Used by test harnesses to ensure clean state between
    tests. Not for production use.
    """
    global _windows, _feedback_handler, _consecutive_emission_failures
    global _total_events_emitted, _total_events_failed, _total_events_suppressed
    global _transition_tracker, _cooldown

    _windows = {}
    _feedback_handler = _null_handler  # never None — always a valid callable
    _consecutive_emission_failures = 0
    _total_events_emitted = 0
    _total_events_failed = 0
    _total_events_suppressed = 0
    _transition_tracker = _RecipeTransitionTracker()
    _cooldown = _CooldownTracker()

    logger.info("Feedback system reset to initial state.")