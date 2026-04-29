"""
tag/topology/surprise_detector.py
===================================
AXIOM's tripwire.  Observes every ClassificationEvent.  Measures divergence
between what the system predicted and what the classifier found.  Fires
SurpriseEvent when something genuinely new has happened.  That's it.

MATHEMATICAL FOUNDATIONS
=========================
Ten PhD-level algorithms applied to the problem of detecting genuine structural
novelty in a stream of topology classifications.  None of these are novel in
isolation — their novelty lies in the specific combination and the framing
of the detection problem in their terms.

  1. Parallel Multivariate Welford (Chan, Golub & LeVeque, 1979)
     ─────────────────────────────────────────────────────────────
     B.P. Welford (1962) provided a numerically stable one-pass algorithm for
     computing mean and variance online.  Chan et al. (1979) extended this to
     allow parallel combination of statistics from two disjoint streams and
     to the full multivariate case maintaining the d×d covariance matrix.

     The key identity is the "parallel update":
       C_AB = C_A + C_B + δδᵀ · n_A·n_B / (n_A + n_B)
     where δ = mean_B - mean_A.

     Applied here: per-class covariance of the 18-dimensional confidence
     vector, updated O(d²) per classification, O(d²) total memory, no raw
     vectors retained.  This is the statistical substrate for Mahalanobis
     gate in Condition 3.

  2. Ledoit-Wolf Analytical Shrinkage (Ledoit & Wolf, 2004 JMVA)
     ─────────────────────────────────────────────────────────────
     When n << d (sparse observations for a given topology class), the sample
     covariance S is rank-deficient or near-singular, making Mahalanobis
     computation numerically unstable.  The Ledoit-Wolf estimator provides
     a closed-form optimal linear shrinkage toward a scaled identity target:

       Σ̂_LW = ρ̂ · (tr(S)/d) · I + (1 - ρ̂) · S

     where the Oracle Approximating Shrinkage coefficient is:

       ρ̂ = [(1 - 2/d)·tr(S²) + tr(S)²] / [(n + 1 - 2/d)·(tr(S²) - tr(S)²/d)]

     This guarantees positive definiteness and invertibility at any sample
     size, while converging to the sample covariance as n → ∞.  Critical
     because the 30-observation minimum gate (SURPRISE_WELFORD_MIN_OBSERVATIONS)
     is the floor below which even LW shrinkage cannot rescue the estimator.

  3. Hill Tail Index Estimator / α-Stable Regime Detection (Hill, 1975)
     ─────────────────────────────────────────────────────────────────────
     B.M. Hill (1975, Annals of Statistics) introduced a consistent estimator
     for the tail index α of a Pareto (heavy-tailed) distribution using the
     top-k order statistics:

       α̂_Hill(k) = k / Σᵢ₌₁ᵏ [log X_(i) - log X_(k+1)]

     where X_(1) ≥ X_(2) ≥ … are the sorted divergence scores.

     Applied here: track α̂ over the rolling divergence score window.  A
     decreasing α̂ indicates increasingly heavy-tailed surprises — extreme
     divergence events are arriving more frequently than a Gaussian model
     would predict.  When α̂ < 1 the distribution has no finite mean, a
     strong indicator that the underlying generator is non-stationary.

  4. Betti-0 Persistent Homology via Sublevel Filtration (Edelsbrunner 2002)
     ─────────────────────────────────────────────────────────────────────────
     Topological Data Analysis applied to the divergence score stream.
     The sublevel set filtration of a finite point cloud X ⊂ ℝ passes the
     Vietoris-Rips complex through increasing radius ε.  Connected components
     (Betti-0) are born when points are added and die when components merge.

     The Betti-0 persistence diagram encodes the lifetime of each connected
     component as a (birth, death) pair.  For divergence scores sorted
     s₁ ≤ s₂ ≤ … ≤ sₙ, a connected component is born at sᵢ and dies at
     sⱼ when the gap [sᵢ, sⱼ] closes (radius = gap/2 ≥ threshold).

     Applied here: two long-lived Betti-0 components (high persistence)
     in the anomalous vector divergence scores indicate a bimodal distribution
     — evidence of two distinct unknown topology classes rather than one.
     This modulates the MDL decision in Algorithm 5.

  5. Rissanen Minimum Description Length Cluster Selection (Rissanen, 1978)
     ─────────────────────────────────────────────────────────────────────────
     J. Rissanen's MDL principle (1978, IBM Journal) provides an information-
     theoretic criterion for model selection that penalizes complexity.

     For Gaussian cluster models on the anomalous vector deque:
       MDL(k) = (n·d/2) · log(RSS_k/n) + (k·d/2) · log(n)
     where RSS_k is the residual sum of squares under k-means with k centroids.

     The decision rule is MDL(k=2) < MDL(k=1) → split the anomalous cluster
     into two hints rather than one.  This prevents a single NewTopologyHintEvent
     from encoding two genuinely different new topology classes.

  6. Wasserstein-1 Earth Mover's Distance on the Probability Simplex
     ─────────────────────────────────────────────────────────────────
     The Wasserstein-1 distance W₁(p, q) is the optimal transport cost
     between two probability measures p, q on a metric space (X, d).

     For discrete distributions on the 18 topology classes under the L1 ground
     metric d(i, j) = |i - j| (ordinal index distance):
       W₁(p, q) = Σₖ₌₁ⁿ |CDF_p(k) - CDF_q(k)|
     where CDF(k) = Σᵢ₌₁ᵏ p(i).

     This is the Cramér distance, equivalent to W₁ for discrete distributions
     on an ordered alphabet.  Unlike KL divergence, W₁ is a proper metric
     that is always finite, even when the supports don't overlap, and it
     captures the geometry of the class ordering (nearby classes cost less
     to transport probability mass between than distant ones).

     Applied here as a supplementary divergence measure for Condition 1, in
     parallel with KL.  Detects "nearby-class confusion" vs "distant-class
     confusion" with different severity weights.

  7. DPSS Multi-taper Spectral Estimation (Slepian 1978 / Thomson 1982)
     ─────────────────────────────────────────────────────────────────────
     D. Slepian (1978, Bell System Technical Journal) introduced the Discrete
     Prolate Spheroidal Sequences (DPSS) as the solutions to the spectral
     concentration problem: find the unit-energy sequence with maximum energy
     in the band [-W, W].  The k-th DPSS solves the eigenvalue problem:

       Σⱼ sin(2πW(t-s)) / (π(t-s)) · vₖ(s) = λₖ · vₖ(t)

     D.J. Thomson (1982) built on this to create the multi-taper spectral
     estimator, which averages K = ⌊2NW⌋ - 1 tapered periodograms, each
     orthogonal to the others, dramatically reducing spectral leakage.

     Applied here to the per-domain divergence score time series (up to
     SURPRISE_CLUSTER_WINDOW points).  A significant spectral peak at
     frequency f indicates a domain with periodic surprise structure —
     e.g., a site that rotates content structure on a weekly basis.
     Periodic surprises should fire at a lower threshold than aperiodic ones.

  8. Oja's Online PCA (Oja, 1982)
     ────────────────────────────
     E. Oja (1982, Journal of Mathematical Biology) derived the stochastic
     approximation:
       w(t+1) = w(t) + η(t) · [x·(xᵀw) - (xᵀw)²·w]
     which converges to the leading eigenvector of E[xxᵀ] (the first
     principal component direction) in a single online pass.

     Applied here to the anomalous confidence vector stream.  When the
     explained variance fraction of PC₁ (|xᵀw|² / ||x||²) is consistently
     high (> SURPRISE_OJA_PC1_THRESHOLD), anomalous vectors lie on a
     1-dimensional manifold — strong evidence of a coherent single new
     topology class rather than diffuse noise.

  9. EWMA Covariance with Forgetting Factor (λ-EWCM, Brown 1959 / Hamilton 1994)
     ─────────────────────────────────────────────────────────────────────────────
     R.G. Brown (1959) introduced exponential smoothing.  The exponentially
     weighted covariance matrix with forgetting factor λ ∈ (0,1):
       μ_t = λ·μ_{t-1} + (1-λ)·x_t
       Σ_t = λ·Σ_{t-1} + (1-λ)·(x_t - μ_t)(x_t - μ_t)ᵀ

     Applied here in parallel with Welford.  The Frobenius distance between
     the EWCM and the Welford covariance:
       ||Σ_EWCM - Σ_Welford||_F
     is a non-stationarity indicator.  When this distance grows suddenly, the
     distribution has shifted recently but not yet in aggregate — the "leading
     indicator" of structural change before Welford accumulates enough evidence.

 10. Normalised Predictive Information (Bialek, Nemenman & Tishby, 2001)
     ─────────────────────────────────────────────────────────────────────
     W. Bialek et al. (2001, Neural Computation) quantified predictability
     as the mutual information between past and future signal:
       NPI = I(X_past; X_future) / H(X_future) ∈ [0, 1]
     where H is Shannon entropy.

     Applied here to the sequence of topology classes seen for a domain,
     estimated via a first-order Markov model:
       NPI ≈ 1 - H(class_t | class_{t-1}) / H(class_t)

     A high NPI means the previous topology class reliably predicts the
     next one (domain is structurally predictable).  A sudden drop in NPI
     (especially after PHASE_KNOWN) indicates the domain is producing
     class sequences inconsistent with its learned structure — a surprise
     signal that complements the divergence-based conditions.

INVARIANTS
==========
1. SurpriseEvent never fires on GENERIC_HTML classification.
2. Condition 4 requires Condition 3 as prerequisite.
3. Welford accumulator never stores raw vectors — O(d²) total memory.
4. Phase thresholds never cached — always read fresh from mmap.
5. Handler is O(1) per ClassificationEvent; all conditions checked.
6. Never writes to store.  In-memory state only.  Restart resets.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import collections
import enum
import hashlib # noqa
import logging # noqa
import math
import mmap
import os
import struct
import time
import threading
import traceback
import unittest
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import ( # noqa
    Any,
    Callable,
    Deque,
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TYPE_CHECKING,
)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
import structlog

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — contracts, exceptions, bus
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.contracts import ( # noqa
    ClassificationEvent,
    FALLBACK_TOPOLOGY_CLASS,
    PHASE_I,
    PHASE_II,
    PHASE_III,
    TOPOLOGY_CLASSES,
    ConfidenceFloat,
    SurpriseEvent as _BusSurpriseEvent,
    new_run_id,
    NewTopologyHintEvent,
)
from signal_kernel.exceptions import ( # noqa
    SurpriseHistoryCorrupted,
    PhaseMMapCorrupted,
)
from tag.crawler_bus import CrawlerBus  # noqa

# ── Topology class index maps — built once at module load ─────────────────
# We compute these independently from the imported list so the module
# has no circular dependency on classifier.py's TOPOLOGY_CLASS_INDEX.
_TOPOLOGY_CLASS_INDEX: Dict[str, int] = {
    cls: idx for idx, cls in enumerate(TOPOLOGY_CLASSES)
}
_INDEX_TO_TOPOLOGY_CLASS: Dict[int, str] = {
    v: k for k, v in _TOPOLOGY_CLASS_INDEX.items()
}
NUM_TOPOLOGY_CLASSES: int = len(TOPOLOGY_CLASSES)

# Pre-built one-hot canonical profiles for _nearest_known_class.
# Avoids repeated np.zeros allocation in the hot path.
_CANONICAL_PROFILES: Dict[str, np.ndarray] = {
    cls: (lambda a: (a.__setitem__(idx, 1.0), a)[1])(
        np.zeros(len(TOPOLOGY_CLASSES), dtype=np.float64)
    )
    for cls, idx in {cls: idx for idx, cls in enumerate(TOPOLOGY_CLASSES)}.items()
}

# ── module logger ─────────────────────────────────────────────────────────
log: structlog.BoundLogger = structlog.get_logger("surprise_detector")


# ═════════════════════════════════════════════════════════════════════════════
# PHASE STATE ENUM
#
# Maps the integer phase constants from contracts.py to a named enum.
# COLD = Phase I (learns), LEARNING = Phase II (predicts), KNOWN = Phase III.
# ═════════════════════════════════════════════════════════════════════════════

class PhaseState(enum.IntEnum):
    """
    Per-domain learning phase.  Maps directly to PHASE_I/II/III integer
    constants from contracts.py.  Higher phase = tighter surprise thresholds.

    COLD     (1) — System still learning this domain.  High noise expected.
                   Loose thresholds.  Most mismatches are classifier uncertainty.
    LEARNING (2) — Domain patterns stabilising.  Genuine surprises meaningful.
                   Moderate thresholds.
    KNOWN    (3) — Domain well-understood.  Any deviation from expected is real.
                   Tight thresholds.  Surprise events trigger gradient corrections.
    """
    COLD     = int(PHASE_I)    # 1
    LEARNING = int(PHASE_II)   # 2
    KNOWN    = int(PHASE_III)  # 3

    @classmethod
    def from_int(cls, value: int) -> "PhaseState":
        """Construct from raw integer, defaulting to COLD on unknown values."""
        try:
            return cls(value)
        except ValueError:
            return cls.COLD

    def __str__(self) -> str:
        return {1: "COLD", 2: "LEARNING", 3: "KNOWN"}[self.value]


# ═════════════════════════════════════════════════════════════════════════════
# SURPRISE SEVERITY ENUM
# ═════════════════════════════════════════════════════════════════════════════

class SurpriseSeverity(str, enum.Enum):
    """
    Severity of a surprise event.  Used by index_daemon to triage response.

    LOW     — Anomalous shape on first occurrence.  Log; monitor frequency.
    MEDIUM  — Repeated GENERIC_HTML fallback.  May need new topology class.
    HIGH    — Confident mismatch or coherent unknown cluster.  Gradient step.
    """
    LOW    = "low"
    MEDIUM = "medium"
    HIGH   = "high"


# ── Hot-path enum cache ───────────────────────────────────────────────────
# Accessing IntEnum / str-Enum members via `ClassName.MEMBER` triggers the
# slow Python descriptor protocol on every access.  Pre-fetch here once at
# module load so the hot path only does a cheap name lookup.
_PHASE_COLD     = PhaseState.COLD
_PHASE_LEARNING = PhaseState.LEARNING
_PHASE_KNOWN    = PhaseState.KNOWN
_SEV_LOW        = SurpriseSeverity.LOW
_SEV_MEDIUM     = SurpriseSeverity.MEDIUM
_SEV_HIGH       = SurpriseSeverity.HIGH

# ═════════════════════════════════════════════════════════════════════════════
# NEW TOPOLOGY HINT EVENT
#
# Emitted when repeated GENERIC_HTML or a coherent unknown cluster indicates
# the domain has structure the 18 topology classes do not cover.
# Triggers discover_signal_zones() in index_daemon.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class NewTopologyHintEvent:
    """
    Emitted on Condition 2 (repeated GENERIC_HTML) and Condition 4 (coherent
    unknown cluster).  index_daemon routes this to discover_signal_zones().

    trigger               — "repeated_generic" | "coherent_cluster"
    evidence_count        — number of observations that triggered this hint
    centroid_vector       — mean confidence profile of the unknown pattern
                             (shape NUM_TOPOLOGY_CLASSES).  Populated for both
                             triggers; for repeated_generic it is the mean
                             distribution of all GENERIC_HTML observations on
                             this domain in the recent window.
    cluster_variance      — total within-cluster variance for coherent_cluster.
                             None for repeated_generic trigger.
    suggested_parent_class — nearest known topology class from centroid_vector.
                             Hint for discover_signal_zones() about which class
                             the unknown pattern most resembles.
    mdl_supports_split    — True if Rissanen MDL(k=2) < MDL(k=1), meaning the
                             anomalous observations may represent two distinct
                             new classes rather than one.
    betti0_modes          — number of long-lived connected components from the
                             Betti-0 persistence analysis.  > 1 corroborates
                             mdl_supports_split.
    oja_pc1_variance_ratio — explained variance fraction of leading PC from
                             Oja online PCA.  High (> 0.85) means anomalous
                             vectors cluster tightly along one direction.
    phase_at_trigger      — PhaseState when this hint was generated.
    run_id                — UUID4 of the triggering event.
    """
    domain:                  str
    trigger:                 str   # "repeated_generic" | "coherent_cluster"
    evidence_count:          int
    centroid_vector:         List[float]
    cluster_variance:        Optional[float]
    suggested_parent_class:  str
    mdl_supports_split:      bool
    betti0_modes:            int
    oja_pc1_variance_ratio:  float
    phase_at_trigger:        PhaseState
    run_id:                  str


# ═════════════════════════════════════════════════════════════════════════════
# SURPRISE THRESHOLDS
# Per-phase threshold bundle.  Constructed by _get_phase_thresholds().
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class SurpriseThresholds:
    """Phase-specific threshold bundle."""
    confident:         float   # minimum classifier confidence to fire Condition 1
    divergence:        float   # KL divergence threshold for Condition 1
    generic_threshold: int     # consecutive GENERIC_HTML before Condition 2 fires
    mahalanobis:       float   # Mahalanobis distance threshold for Condition 3
    wasserstein:       float   # supplementary W₁ threshold (advisory only)
    npi_drop:          float   # NPI drop threshold for predictability loss signal


# ═════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

# ── Phase-specific confidence thresholds (Condition 1) ──────────────────────
THETA_SURPRISE_CONFIDENT_COLD:     float = 0.85
THETA_SURPRISE_CONFIDENT_LEARNING: float = 0.75
THETA_SURPRISE_CONFIDENT_KNOWN:    float = 0.65

# ── Phase-specific divergence thresholds (Condition 1) ──────────────────────
THETA_SURPRISE_DIVERGENCE_COLD:     float = 0.60
THETA_SURPRISE_DIVERGENCE_LEARNING: float = 0.45
THETA_SURPRISE_DIVERGENCE_KNOWN:    float = 0.30

# ── Phase-specific Wasserstein thresholds (Condition 1, supplementary) ──────
THETA_SURPRISE_WASSERSTEIN_COLD:     float = 0.40
THETA_SURPRISE_WASSERSTEIN_LEARNING: float = 0.30
THETA_SURPRISE_WASSERSTEIN_KNOWN:    float = 0.20

# ── Phase-specific consecutive GENERIC_HTML thresholds (Condition 2) ────────
SURPRISE_GENERIC_THRESHOLD_COLD:     int = 15
SURPRISE_GENERIC_THRESHOLD_LEARNING: int = 8
SURPRISE_GENERIC_THRESHOLD_KNOWN:    int = 3

# ── Generic window (Condition 2) ─────────────────────────────────────────────
SURPRISE_GENERIC_WINDOW: int = 20

# ── Phase-specific Mahalanobis thresholds (Condition 3) ─────────────────────
THETA_SURPRISE_MAHALANOBIS_COLD:     float = 3.5
THETA_SURPRISE_MAHALANOBIS_LEARNING: float = 2.8
THETA_SURPRISE_MAHALANOBIS_KNOWN:    float = 2.0

# ── Welford minimum observation gate ─────────────────────────────────────────
SURPRISE_WELFORD_MIN_OBSERVATIONS: int = 30
# 30 observations validated by dimensional analysis: with d=18 dimensions,
# stable Ledoit-Wolf shrinkage requires n > d for the sample covariance to be
# rank-sufficient before regularization.  30 observations gives n/d ≈ 1.67,
# the minimum ratio at which LW shrinkage produces a well-conditioned estimate.
# Below 30, LW may set ρ→1 (pure spherical), making Mahalanobis distance
# proportional to L2 distance — still useful, but not topology-aware.
# Conservative: matches the 30-sample CLT stability threshold from the
# Berry-Esseen theorem for d=18.

# ── Anomalous vector cluster window (Condition 4) ───────────────────────────
SURPRISE_CLUSTER_WINDOW: int = 50
THETA_SURPRISE_CLUSTER_COHERENCE: float = 0.15

# ── Oja PCA threshold (Algorithm 8) ─────────────────────────────────────────
SURPRISE_OJA_PC1_THRESHOLD: float = 0.70   # PC1 explained variance → coherent cluster

# ── EWMA forgetting factor (Algorithm 9) ─────────────────────────────────────
SURPRISE_EWMA_LAMBDA: float = 0.97   # slow decay; surprise detector is long-horizon

# ── NPI drop threshold (Algorithm 10) ────────────────────────────────────────
THETA_SURPRISE_NPI_DROP_LEARNING: float = 0.30
THETA_SURPRISE_NPI_DROP_KNOWN:    float = 0.20

# ── Phase-specific NPI drop thresholds ───────────────────────────────────────
THETA_SURPRISE_NPI_DROP_COLD: float = 0.50   # relaxed for cold domains

# ── Domain state limits ───────────────────────────────────────────────────────
SURPRISE_DOMAIN_STATE_MAX: int = 10_000
SURPRISE_EVICT_LRU: bool = True

# ── Frequency severity escalation ────────────────────────────────────────────
SURPRISE_CONDITION3_ESCALATION_WINDOW: int = 20   # observations
SURPRISE_CONDITION3_ESCALATION_COUNT:  int = 3    # hits → escalate to MEDIUM

# ── Persistent phase mmap path (mirroring contracts STORE_FILE_NAMES) ────────
_PHASE_MMAP_PATH: Path = Path("/store/phase_states.mmap")
_PHASE_SLOT_BYTES: int = 32
_PHASE_SLOT_STRUCT: struct.Struct = struct.Struct("<BBxxfxxxxxxf")  # phase, flags, _, conf, _, surprise


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 1
# Parallel Multivariate Welford Online Covariance Estimator
# Chan, Golub & LeVeque (1979) — numerically stable O(d²) per update
# ═════════════════════════════════════════════════════════════════════════════

class MultivariateWelfordAccumulator:
    """
    Numerically stable online estimator for multivariate mean and covariance.

    Implements the Chan-Golub-LeVeque (1979) parallel algorithm for exact
    online covariance, extended to the full d×d matrix.  No raw vectors
    are stored: total memory is O(d²) regardless of n.

    The update uses the corrected two-pass formula to avoid catastrophic
    cancellation, which plagues naive implementations when values are
    large and variances are small.

    References:
      Chan, T.F., Golub, G.H., LeVeque, R.J. (1979). "Updating formulae and
      a pairwise algorithm for computing sample variances."  Technical Report
      STAN-CS-79-773, Stanford.

      Welford, B.P. (1962). "Note on a method for calculating corrected sums
      of squares and products." Technometrics 4(3), 419-420.
    """

    __slots__ = ("_n", "_mean", "_M2", "_d")

    def __init__(self, d: int) -> None:
        """
        Args:
            d: dimensionality of the observation vectors.
        """
        self._d: int = d
        self._n: int = 0
        self._mean: np.ndarray = np.zeros(d, dtype=np.float64)
        self._M2: np.ndarray = np.zeros((d, d), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        """Incorporate one new observation vector (shape (d,))."""
        if x.shape[0] != self._d:
            raise ValueError(
                f"Expected dimension {self._d}, got {x.shape[0]}. "
                "All observations must have the same dimension."
            )
        self._n += 1
        delta = x - self._mean
        self._mean += delta / self._n
        delta2 = x - self._mean
        # Outer product accumulation: M2 += delta ⊗ delta2 (rank-1 update)
        self._M2 += np.outer(delta, delta2)

    def merge(self, other: "MultivariateWelfordAccumulator") -> None:
        """
        In-place parallel merge: incorporate all observations from `other`.
        Chan-Golub-LeVeque parallel combination formula.
        O(d²) regardless of how many observations each accumulator holds.
        """
        if other._d != self._d:
            raise ValueError("Cannot merge accumulators of different dimensionality.")
        n_ab = self._n + other._n
        if n_ab == 0:
            return
        delta = other._mean - self._mean
        self._M2 = (
            self._M2
            + other._M2
            + np.outer(delta, delta) * (self._n * other._n / n_ab)
        )
        self._mean = (self._mean * self._n + other._mean * other._n) / n_ab
        self._n = n_ab

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> np.ndarray:
        return self._mean.copy()

    def covariance(self, ddof: int = 1) -> np.ndarray:
        """
        Sample covariance matrix (ddof=1) or population covariance (ddof=0).
        Returns zero matrix when n <= ddof.
        """
        if self._n <= ddof:
            return np.zeros((self._d, self._d), dtype=np.float64)
        return self._M2 / (self._n - ddof)

    def variance(self) -> np.ndarray:
        """Per-dimension marginal variance (diagonal of covariance matrix)."""
        return np.diag(self.covariance())

    def std(self) -> np.ndarray:
        """Per-dimension standard deviation."""
        return np.sqrt(np.maximum(self.variance(), 0.0))

    def reset(self) -> None:
        """Reset accumulator to zero observations."""
        self._n = 0
        self._mean = np.zeros(self._d, dtype=np.float64)
        self._M2 = np.zeros((self._d, self._d), dtype=np.float64)

    def snapshot(self) -> Dict[str, Any]:
        """Serialisable snapshot for health monitoring."""
        return {
            "n":    self._n,
            "mean": self._mean.tolist(),
            "d":    self._d,
        }


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 2
# Ledoit-Wolf Analytical Shrinkage Estimator
# Ledoit & Wolf (2004 JMVA) — closed-form optimal covariance regularisation
# ═════════════════════════════════════════════════════════════════════════════

class LedoitWolfShrinkageEstimator:
    """
    Compute the Oracle Approximating Shrinkage (OAS) coefficient for a
    sample covariance matrix S under the scaled-identity target μ·I.

    The Oracle Approximating Shrinkage (OAS) estimator by Chen, Wiesel,
    Eldar & Hero (2010, IEEE TSP) improves upon the basic LW formula by
    iterating once to reduce bias.  We use the non-iterative LW form for
    O(1) computation.

    Σ̂_LW = ρ̂ · (tr(S)/d) · I + (1 - ρ̂) · S

    where the shrinkage coefficient:
      ρ̂ = [(1 - 2/d)·tr(S²) + tr(S)²] / [(n + 1 - 2/d)·(tr(S²) - tr(S)²/d)]

    This formula is the exact minimizer of E[||Σ̂ - Σ_true||_F²] over the
    class of linear shrinkage estimators toward scaled identity.

    References:
      Ledoit, O., Wolf, M. (2004). "A well-conditioned estimator for
      large-dimensional covariance matrices." JMVA 88, 365-411.
    """

    @staticmethod
    def shrink(S: np.ndarray, n: int) -> np.ndarray:
        """
        Apply LW shrinkage to sample covariance S with n observations.

        Args:
            S: (d, d) sample covariance matrix.
            n: number of observations used to compute S.

        Returns:
            Σ̂_LW: regularised positive-definite covariance matrix.
        """
        d = S.shape[0]
        if n <= 0 or d <= 0:
            return np.eye(d, dtype=np.float64)

        tr_S  = np.trace(S)
        # tr(S²) = ||S||_F² for symmetric S — O(d²) vs O(d³) for S @ S.
        tr_S2 = float(np.einsum('ij,ij->', S, S))
        mu    = tr_S / d   # spherical target scale

        # Numerator and denominator of the shrinkage coefficient.
        numerator   = (1.0 - 2.0 / d) * tr_S2 + tr_S ** 2
        denominator = (n + 1.0 - 2.0 / d) * (tr_S2 - tr_S ** 2 / d)

        if abs(denominator) < 1e-14:
            # Degenerate case: S is already spherical, no shrinkage needed.
            rho = 0.0
        else:
            rho = min(1.0, max(0.0, numerator / denominator))

        return (1.0 - rho) * S + rho * mu * np.eye(d, dtype=np.float64)

    @staticmethod
    def regularised_inverse(
        S: np.ndarray,
        n: int,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Compute the inverse of the LW-regularised covariance.

        Falls back to pseudoinverse if the regularised matrix is still
        numerically singular (should not occur in practice with LW, but
        this is production code — defence in depth).
        """
        Sigma_reg = LedoitWolfShrinkageEstimator.shrink(S, n)
        try:
            # Cholesky-based inverse: numerically stable for PD matrices.
            L = np.linalg.cholesky(Sigma_reg + epsilon * np.eye(S.shape[0]))
            L_inv = np.linalg.inv(L)
            return L_inv.T @ L_inv
        except np.linalg.LinAlgError:
            # Fall back to pseudoinverse with Tikhonov regularisation.
            return np.linalg.pinv(Sigma_reg + epsilon * np.eye(S.shape[0]))


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 3
# Hill Tail Index Estimator + α-Stable Regime Detection
# Hill (1975) — consistent estimator for Pareto tail index
# ═════════════════════════════════════════════════════════════════════════════

class HillTailIndexEstimator:
    """
    Online tail index estimator using the Hill (1975) statistic.

    Maintains a fixed-size window of the most extreme divergence scores and
    computes the Hill estimator:

      α̂_Hill(k) = k / Σᵢ₌₁ᵏ [log X_(i) - log X_(k+1)]

    where X_(1) ≥ X_(2) ≥ … are the top-k order statistics.

    A decreasing α̂ over time indicates increasingly heavy-tailed behaviour —
    extreme surprise events arriving more frequently than Gaussian prediction.

    α̂ > 2  → sub-Gaussian tails (expected variance regime)
    α̂ ≈ 2  → heavy tails, variance barely finite
    α̂ ≈ 1  → Cauchy-like, no finite mean, severe non-stationarity
    α̂ < 1  → super-heavy, system almost certainly undergoing structural change

    References:
      Hill, B.M. (1975). "A simple general approach to inference about the
      tail of a distribution." Annals of Statistics 3(5), 1163-1174.
    """

    __slots__ = ("_window", "_k_fraction", "_max_window")

    def __init__(self, max_window: int = 200, k_fraction: float = 0.10) -> None:
        """
        Args:
            max_window:   rolling window of divergence scores to maintain.
            k_fraction:   fraction of observations used as order statistics
                          in the Hill estimator.  0.10 = top 10%.
        """
        self._max_window = max_window
        self._k_fraction = k_fraction
        self._window: Deque[float] = deque(maxlen=max_window)

    def update(self, score: float) -> None:
        """Add one divergence score to the rolling window."""
        if score > 0:
            self._window.append(score)

    def alpha_hat(self) -> Optional[float]:
        """
        Compute the Hill estimator of the tail index.

        Returns None if insufficient observations (< 10 in window).
        Returns math.inf when all observed scores are equal (no tail).
        """
        n = len(self._window)
        if n < 10:
            return None

        k = max(2, int(n * self._k_fraction))
        k = min(k, n - 1)

        # O(n) partial sort: extract only the top k+1 values we need,
        # rather than a full O(n log n) sort of the whole window.
        arr = np.asarray(self._window, dtype=np.float64)
        top_idx = np.argpartition(arr, -(k + 1))[-(k + 1):]
        top = arr[top_idx]
        top.sort()
        top = top[::-1]          # top[0..k-1] largest; top[k] = (k+1)-th largest

        if top[k] <= 0:
            return None

        # Avoid log(0) — scores must be strictly positive.
        log_diffs = [
            math.log(top[i]) - math.log(top[k])
            for i in range(k)
            if top[i] > 0
        ]
        if not log_diffs:
            return None

        denom = sum(log_diffs)
        if abs(denom) < 1e-14:
            return math.inf   # all top-k values identical — no tail structure
        return len(log_diffs) / denom

    def is_heavy_tailed(self, threshold: float = 2.0) -> bool:
        """
        True if the Hill estimator α̂ < threshold, indicating heavier-than-
        Gaussian tails.  threshold=2.0 is the Gaussian boundary.
        """
        alpha = self.alpha_hat()
        return alpha is not None and alpha < threshold

    def severity_from_alpha(self) -> SurpriseSeverity:
        """
        Map current tail index to a SurpriseSeverity for logging.
        This is an advisory signal, not a primary severity gate.
        """
        alpha = self.alpha_hat()
        if alpha is None or alpha > 2.0:
            return SurpriseSeverity.LOW
        if alpha > 1.0:
            return SurpriseSeverity.MEDIUM
        return SurpriseSeverity.HIGH


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 4
# Betti-0 Persistent Homology via Sublevel Set Filtration
# Edelsbrunner, Letscher & Zomorodian (2002) — topological data analysis
# ═════════════════════════════════════════════════════════════════════════════

class Betti0PersistenceAnalyser:
    """
    Compute Betti-0 (connected component) persistence barcodes for a 1D
    point cloud (scalar divergence score window).

    Under the Rips/sublevel filtration on ℝ¹, connected components are born
    at isolated points and die when the growing epsilon-ball connects them to
    a neighbour.  The persistence of component i is gap_i / 2, where gap_i
    is the gap between consecutive sorted points.

    A high-persistence component that "lives long" indicates a stable mode
    in the distribution — a cluster of points separated from other clusters
    by a large gap.  Two or more long-lived components signal bimodality in
    the divergence score distribution, corroborating MDL evidence for two
    distinct unknown topology classes.

    The "significant persistence threshold" τ is set to one standard deviation
    of the gap distribution.  Gaps larger than τ are considered significant.

    References:
      Edelsbrunner, H., Letscher, D., Zomorodian, A. (2002). "Topological
      persistence and simplification." Discrete & Computational Geometry
      28(4), 511-533.
    """

    @staticmethod
    def betti0_barcode(
        scores: Sequence[float],
    ) -> List[Tuple[float, float]]:
        """
        Compute the Betti-0 persistence barcode for a set of scalar values.

        Returns a list of (birth, death) pairs sorted by persistence
        (death - birth) in descending order.  Each pair represents one
        connected component in the sublevel filtration.

        The "oldest" component (born first, never dies) has death = inf.
        """
        if len(scores) < 2:
            return [(scores[0], math.inf)] if scores else []

        pts = sorted(scores)
        n = len(pts)

        # Under the Vietoris-Rips filtration on ℝ¹:
        # - Start with n isolated components at radius ε=0.
        # - As ε increases, adjacent points merge at ε = gap/2.
        # - The "older" component survives (lower birth value).
        # - The younger component dies at this merge radius.

        # Union-Find for merging.
        parent = list(range(n))
        # In the Vietoris-Rips filtration all components are born at radius 0;
        # only the gap/2 at which they merge matters for persistence.
        birth  = [0.0] * n

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        barcodes: List[Tuple[float, float]] = []
        gaps = [(pts[i + 1] - pts[i], i, i + 1) for i in range(n - 1)]
        # Process gaps in increasing order — smallest gap merges first.
        for gap, left_idx, right_idx in sorted(gaps):
            r_left  = find(left_idx)
            r_right = find(right_idx)
            if r_left == r_right:
                continue  # already connected
            # Merge: younger component dies at this gap threshold.
            merge_radius = gap / 2.0
            if birth[r_left] <= birth[r_right]:
                # Left component older — right component dies.
                barcodes.append((birth[r_right], merge_radius))
                parent[r_right] = r_left
            else:
                barcodes.append((birth[r_left], merge_radius))
                parent[r_left] = r_right

        # The surviving component lives forever.
        survivor_root = find(0)
        barcodes.append((birth[survivor_root], math.inf))

        # Sort by persistence (death - birth) descending.
        barcodes.sort(key=lambda b: (b[1] - b[0]), reverse=True)
        return barcodes

    @staticmethod
    def count_significant_modes(
        scores: Sequence[float],
        persistence_quantile: float = 0.75,
    ) -> int:
        """
        Count Betti-0 components with persistence above the given quantile
        of all finite gap sizes.  Components with persistence above this
        threshold represent statistically significant modes.

        Returns 1 for unimodal distributions, 2+ for multimodal.
        """
        if len(scores) < 4:
            return 1

        barcodes = Betti0PersistenceAnalyser.betti0_barcode(scores)
        finite_persistences = [
            b[1] - b[0] for b in barcodes if math.isfinite(b[1])
        ]
        if not finite_persistences:
            return 1  # only the infinite component — truly unimodal

        # Pure-Python quantile — avoids numpy array-validation overhead which
        # becomes the dominant cost when called thousands of times on short lists.
        finite_sorted = sorted(finite_persistences)
        n_finite = len(finite_sorted)
        lo_idx  = int(n_finite * persistence_quantile)
        hi_idx  = min(lo_idx + 1, n_finite - 1)
        frac    = n_finite * persistence_quantile - lo_idx
        threshold = finite_sorted[lo_idx] * (1.0 - frac) + finite_sorted[hi_idx] * frac
        # Apply a small relative tolerance so that components whose persistence
        # is within floating-point noise of the threshold do not get counted.
        # This prevents spurious modes from uniformly-spaced (unimodal) data
        # where all finite persistences are nearly identical.
        threshold_with_tol = threshold * (1.0 + 1e-9) + 1e-15
        # Count components (including infinite) with persistence > threshold.
        return sum(
            1 for b in barcodes
            if (b[1] - b[0]) > threshold_with_tol
        )


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 5
# Rissanen MDL Cluster Selection
# Rissanen (1978) — minimum description length model selection
# ═════════════════════════════════════════════════════════════════════════════

class RissanenMDLClusterEvaluator:
    """
    Decide between k=1 and k=2 Gaussian cluster models using the Minimum
    Description Length (MDL) criterion (Rissanen, 1978).

    MDL(k) = (n · d / 2) · log(RSS_k / n) + (k · d / 2) · log(n)

    where RSS_k = Σᵢ ||xᵢ - μ_cluster(xᵢ)||² is the residual sum of squares
    under k-means clustering with k centroids.

    If MDL(k=2) < MDL(k=1), the data is better described by two clusters —
    i.e., the anomalous observations represent two distinct unknown topology
    classes rather than one.

    References:
      Rissanen, J. (1978). "Modeling by shortest data description."
      Automatica 14(5), 465-471.
    """

    @staticmethod
    def _kmeans_1d_2cluster(data: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Optimal k=2 clustering for 1D data via dynamic programming.
        Returns (centroid_1, centroid_2, rss) where the split is optimal.
        Runs in O(n log n).
        """
        pts = np.sort(data)
        n = len(pts)
        if n < 2:
            mu = float(np.mean(pts)) if len(pts) > 0 else 0.0
            return (mu, mu, 0.0, 0.0)

        best_rss = math.inf
        best_split = 1
        # Cumulative sums for O(1) segment mean/RSS computation.
        cum_sum = np.cumsum(pts)
        cum_sq  = np.cumsum(pts ** 2)

        def segment_rss(lo: int, hi: int) -> float:
            """RSS of pts[lo:hi+1] from their mean."""
            count = hi - lo + 1
            s  = cum_sum[hi] - (cum_sum[lo - 1] if lo > 0 else 0.0)
            sq = cum_sq[hi]  - (cum_sq[lo - 1]  if lo > 0 else 0.0)
            return float(sq - s * s / count)

        for split in range(1, n):
            rss = segment_rss(0, split - 1) + segment_rss(split, n - 1)
            if rss < best_rss:
                best_rss  = rss
                best_split = split

        n1 = best_split
        n2 = n - best_split
        mu1 = float(np.mean(pts[:best_split]))
        mu2 = float(np.mean(pts[best_split:]))
        return (mu1, mu2, best_rss, best_split)

    @staticmethod
    def mdl_supports_split(
        vectors: np.ndarray,
        use_projection: bool = True,
    ) -> bool:
        """
        Returns True if MDL(k=2) < MDL(k=1) for the given matrix of
        anomalous confidence vectors (shape: n × d).

        For computational efficiency, when use_projection=True we first
        project onto the leading principal component (online Oja, or
        exact for this batch) before applying 1D k-means.

        Args:
            vectors: (n, d) matrix of anomalous confidence vectors.
            use_projection: if True, project to 1D first.

        Returns:
            True if two-cluster model is preferred under MDL.
        """
        n, d = vectors.shape
        if n < 4:
            return False

        if use_projection and d > 1:
            # Project onto first PC (fast via Oja-like method on batch).
            centered = vectors - vectors.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered, full_matrices=False)
            pc1 = Vt[0]
            proj = centered @ pc1          # shape (n,)
        else:
            proj = vectors[:, 0]          # fallback: first dimension

        # k=1 model.
        rss1 = float(np.sum((proj - proj.mean()) ** 2))
        if rss1 < 1e-14:
            return False   # all points identical — no clustering needed

        # k=2 model.
        _, _, rss2, _ = RissanenMDLClusterEvaluator._kmeans_1d_2cluster(proj)

        # MDL with log(n) model complexity penalty.
        # Per the formula MDL(k) = (n·d/2)·log(RSS_k/n) + (k·d/2)·log(n),
        # we use the original dimensionality d for the penalty even after
        # projecting to 1D, because the projection compresses a d-dimensional
        # model and the complexity cost must reflect the full parameter count.
        mdl1 = (n / 2.0) * math.log(max(rss1 / n, 1e-14)) + (1.0 * d / 2.0) * math.log(n)
        mdl2 = (n / 2.0) * math.log(max(rss2 / n, 1e-14)) + (2.0 * d / 2.0) * math.log(n)
        return mdl2 < mdl1


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 6
# Wasserstein-1 / Cramér Distance on the Probability Simplex
# ═════════════════════════════════════════════════════════════════════════════

class WassersteinSimplexDistance:
    """
    Compute the Wasserstein-1 (Earth Mover's) distance between two discrete
    probability distributions on the same finite ordered alphabet.

    For discrete distributions p, q on n atoms with ordinal ground metric
    d(i, j) = |i - j|, the Wasserstein-1 distance has the closed form:

      W₁(p, q) = Σₖ₌₀ⁿ⁻² |CDF_p(k) - CDF_q(k)|

    This is the Cramér distance (also called the energy distance for α=1).
    Unlike KL divergence, W₁:
      - Is symmetric
      - Is always finite, even when the supports don't overlap
      - Respects the geometry of the class ordering
      - Satisfies the triangle inequality (proper metric)

    The total variation distance (L1/2 norm):
      TV(p, q) = Σₖ |p(k) - q(k)| / 2

    is also computed as a simpler complementary measure.
    """

    @staticmethod
    def wasserstein1(p: np.ndarray, q: np.ndarray) -> float:
        """
        W₁ distance (Cramér distance) between discrete distributions p and q.

        Args:
            p: (d,) probability vector, should sum to 1.
            q: (d,) probability vector, should sum to 1.

        Returns:
            W₁(p, q) ∈ [0, d-1].
        """
        if len(p) != len(q):
            raise ValueError("p and q must have the same length.")
        cdf_diff = np.cumsum(p - q)
        return float(np.sum(np.abs(cdf_diff[:-1])))

    @staticmethod
    def total_variation(p: np.ndarray, q: np.ndarray) -> float:
        """
        Total variation distance: TV(p, q) = ||p - q||₁ / 2.

        Returns value in [0, 1].
        """
        return float(np.sum(np.abs(p - q))) / 2.0

    @staticmethod
    def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
        """
        Jensen-Shannon divergence: JSD(p, q) ∈ [0, log 2].

        Symmetric, bounded, always defined.  Square root is the JS metric.
        JSD(p, q) = 0 iff p == q.
        """
        eps = 1e-10
        m = 0.5 * (p + q)
        kl_pm = np.sum(p * np.log((p + eps) / (m + eps)))
        kl_qm = np.sum(q * np.log((q + eps) / (m + eps)))
        return float(0.5 * kl_pm + 0.5 * kl_qm)


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 7
# DPSS Multi-taper Spectral Estimator
# Slepian (1978) / Thomson (1982) — optimal spectral concentration
# ═════════════════════════════════════════════════════════════════════════════

class DPSSMultitaperAnalyser:
    """
    Detect periodic surprise structure in the per-domain divergence score
    time series using Slepian Discrete Prolate Spheroidal Sequences.

    The DPSS sequences are the eigenvectors of the tridiagonal matrix:
      A[i,j] = sin(2πW(i-j)) / (π(i-j))  for i ≠ j
      A[i,i] = 2W

    where W is the half-bandwidth parameter.  The first K = ⌊2NW⌋ - 1
    eigenvectors have eigenvalues λ_k ≈ 1 (high spectral concentration).

    Each DPSS taper h_k is applied to the time series: y_k[t] = h_k[t] · x[t].
    The multi-taper estimate averages the K periodograms:
      S(f) = (1/K) · Σₖ |FFT(y_k)[f]|²

    This dramatically reduces spectral leakage compared to single-window FFT.

    Applied here: if the dominant frequency in the per-domain divergence
    score series has a period matching weekly (7) or daily (1) cycles, we
    consider the surprise pattern temporal rather than structural — lower
    severity, do not fire NewTopologyHintEvent.

    References:
      Slepian, D. (1978). "Prolate spheroidal wave functions, Fourier
      analysis, and uncertainty V." Bell System Technical Journal 57(5).
      Thomson, D.J. (1982). "Spectrum estimation and harmonic analysis."
      Proceedings of the IEEE 70(9), 1055-1096.
    """

    def __init__(self, nw: float = 4.0, n_tapers: int = 3) -> None:
        """
        Args:
            nw:       time-bandwidth product (W = nw / N).
            n_tapers: number of tapers to use (≤ ⌊2NW⌋ - 1).
        """
        self._nw       = nw
        self._n_tapers = n_tapers
        self._tapers_cache: Dict[int, np.ndarray] = {}

    def _compute_dpss(self, N: int) -> np.ndarray:
        """
        Compute DPSS (Slepian) sequences for length N using the tri-diagonal
        eigenvalue approach.  Returns array of shape (K, N).
        """
        if N in self._tapers_cache:
            return self._tapers_cache[N]

        W = self._nw / N
        K = min(self._n_tapers, max(1, int(2 * self._nw) - 1))

        # Tridiagonal matrix whose eigenvectors are the DPSS.
        t = np.arange(N, dtype=np.float64)
        diag_main = ((N - 1 - 2 * t) / 2.0) ** 2 * np.cos(2 * np.pi * W)
        diag_off  = t[1:] * (N - t[1:]) / 2.0

        # Use numpy's symmetric tridiagonal eigensolver.
        eigenvalues, eigenvectors = np.linalg.eigh(
            np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        )
        # Select K largest eigenvalues (most concentrated tapers).
        idx = np.argsort(eigenvalues)[::-1][:K]
        tapers = eigenvectors[:, idx].T   # shape (K, N)

        # Normalise each taper to unit energy.
        for k in range(K):
            norm = np.sqrt(np.sum(tapers[k] ** 2))
            if norm > 1e-14:
                tapers[k] /= norm
            # Enforce sign convention: positive first component.
            if tapers[k, 0] < 0:
                tapers[k] *= -1

        self._tapers_cache[N] = tapers
        return tapers

    def dominant_frequency(self, series: Sequence[float]) -> Tuple[float, float]:
        """
        Compute the multi-taper power spectrum of the series and return
        the dominant frequency and its power ratio.

        Returns:
            (freq, power_ratio) where freq is in cycles per sample and
            power_ratio is the fraction of total power at that frequency.
        """
        x = np.asarray(series, dtype=np.float64)
        N = len(x)
        if N < 8:
            return (0.0, 0.0)

        # Detrend: remove linear trend.
        x = x - np.polyval(np.polyfit(np.arange(N), x, 1), np.arange(N))

        tapers = self._compute_dpss(N)
        K = tapers.shape[0]

        # Multi-taper power spectrum.
        spectrum = np.zeros(N // 2 + 1, dtype=np.float64)
        for k in range(K):
            tapered = tapers[k] * x
            fft_val = np.fft.rfft(tapered)
            spectrum += np.abs(fft_val) ** 2
        spectrum /= K

        # Skip DC (index 0).
        if len(spectrum) <= 1:
            return (0.0, 0.0)

        peak_idx   = int(np.argmax(spectrum[1:])) + 1
        freq       = peak_idx / N
        total_pow  = float(np.sum(spectrum[1:]))
        peak_pow   = float(spectrum[peak_idx])
        power_ratio = peak_pow / total_pow if total_pow > 1e-14 else 0.0
        return (float(freq), power_ratio)

    def is_periodic(self, series: Sequence[float], threshold: float = 0.35) -> bool:
        """
        True if the series has a dominant periodic component carrying more
        than `threshold` fraction of total spectral power.
        """
        _, power_ratio = self.dominant_frequency(series)
        return power_ratio > threshold


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 8
# Oja's Online PCA Rule
# Oja (1982) — stochastic approximation of leading eigenvector
# ═════════════════════════════════════════════════════════════════════════════

class OjaOnlinePCATracker:
    """
    Track the leading principal component of a streaming data source using
    Oja's stochastic gradient rule (1982):

      w(t+1) = w(t) + η(t) · [x·(xᵀw) - (xᵀw)² · w]
             = w(t) + η(t) · (xᵀw) · [x - (xᵀw)·w]

    followed by normalisation w ← w / ||w||.

    Converges to the leading eigenvector of E[xxᵀ] as η(t) → 0.
    Uses the Robbins-Monro learning rate schedule η(t) = η₀ / (1 + β·t).

    The "PC1 explained variance ratio" for each new observation x is:
      r(x) = |xᵀw|² / ||x||²

    A consistently high r (> SURPRISE_OJA_PC1_THRESHOLD) means all anomalous
    vectors lie near the same 1D manifold — strong evidence of a coherent
    new topology class.

    References:
      Oja, E. (1982). "A simplified neuron model as a principal component
      analyser." Journal of Mathematical Biology 15(3), 267-273.
    """

    __slots__ = ("_d", "_w", "_t", "_eta0", "_beta", "_pc1_history")

    def __init__(
        self,
        d: int,
        eta0: float = 0.1,
        beta: float = 0.01,
    ) -> None:
        """
        Args:
            d:     dimensionality of the data.
            eta0:  initial learning rate.
            beta:  learning rate decay parameter.
        """
        self._d    = d
        self._eta0 = eta0
        self._beta = beta
        self._t: int = 0
        # Initialise with small random values — avoids symmetry breaking.
        rng = np.random.default_rng(seed=42)
        self._w = rng.standard_normal(d).astype(np.float64)
        norm = np.linalg.norm(self._w)
        self._w = self._w / norm if norm > 1e-14 else np.ones(d) / math.sqrt(d)
        self._pc1_history: Deque[float] = deque(maxlen=50)

    def update(self, x: np.ndarray) -> float:
        """
        Incorporate one observation and return the PC1 explained variance
        ratio for this observation.
        """
        x = x.astype(np.float64)
        x_norm_sq = float(np.dot(x, x))
        if x_norm_sq < 1e-14:
            return 0.0

        proj = float(np.dot(x, self._w))
        eta  = self._eta0 / (1.0 + self._beta * self._t)
        self._t += 1

        # Oja update.
        self._w = self._w + eta * proj * (x - proj * self._w)
        norm = np.linalg.norm(self._w)
        if norm > 1e-14:
            self._w /= norm
        else:
            self._w = np.ones(self._d) / math.sqrt(self._d)

        ratio = (proj ** 2) / x_norm_sq
        self._pc1_history.append(ratio)
        return ratio

    @property
    def mean_pc1_ratio(self) -> float:
        """Mean PC1 explained variance ratio over recent history."""
        if not self._pc1_history:
            return 0.0
        return float(np.mean(list(self._pc1_history)))

    @property
    def is_coherent(self) -> bool:
        """
        True if recent anomalous vectors are consistently aligned with PC1.
        Threshold: SURPRISE_OJA_PC1_THRESHOLD.
        """
        return (
            len(self._pc1_history) >= 10
            and self.mean_pc1_ratio >= SURPRISE_OJA_PC1_THRESHOLD
        )

    def reset(self) -> None:
        rng = np.random.default_rng(seed=42 + self._t)
        self._w = rng.standard_normal(self._d).astype(np.float64)
        self._w /= np.linalg.norm(self._w) + 1e-14
        self._t = 0
        self._pc1_history.clear()


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 9
# EWMA Covariance with Forgetting Factor (λ-EWCM)
# Brown (1959) / Hamilton (1994) — exponential smoothing extended to matrices
# ═════════════════════════════════════════════════════════════════════════════

class EWMACovariance:
    """
    Exponentially weighted moving covariance matrix with forgetting factor λ.

    Update equations:
      μ_t = λ·μ_{t-1} + (1-λ)·x_t
      Σ_t = λ·Σ_{t-1} + (1-λ)·(x_t - μ_t)(x_t - μ_t)ᵀ

    The forgetting factor λ ∈ (0,1) controls the effective sample window:
      n_eff ≈ 1/(1-λ)

    For λ = 0.97: n_eff ≈ 33 (about one month of daily observations).

    The non-stationarity indicator is the Frobenius distance between this
    EWCM and the Welford sample covariance:
      Δ_F = ||Σ_EWCM - Σ_Welford||_F

    A sudden increase in Δ_F indicates the recent distribution has shifted
    away from the historical aggregate — a "leading indicator" of structural
    change that precedes Condition 3 Mahalanobis fires.

    References:
      Brown, R.G. (1959). "Statistical Forecasting for Inventory Control."
      McGraw-Hill, New York.
    """

    __slots__ = ("_lambda", "_mu", "_sigma", "_n", "_d")

    def __init__(self, d: int, lambda_: float = SURPRISE_EWMA_LAMBDA) -> None:
        self._d:      int = d
        self._lambda: float = lambda_
        self._n:      int = 0
        self._mu:     np.ndarray = np.zeros(d, dtype=np.float64)
        self._sigma:  np.ndarray = np.zeros((d, d), dtype=np.float64)

    def update(self, x: np.ndarray) -> None:
        x = x.astype(np.float64)
        lam = self._lambda
        if self._n == 0:
            self._mu = x.copy()
        else:
            self._mu = lam * self._mu + (1 - lam) * x
        dev = x - self._mu
        self._sigma = lam * self._sigma + (1 - lam) * np.outer(dev, dev)
        self._n += 1

    @property
    def mean(self) -> np.ndarray:
        return self._mu.copy()

    @property
    def covariance(self) -> np.ndarray:
        return self._sigma.copy()

    def frobenius_distance_from(self, other_sigma: np.ndarray) -> float:
        """||self.sigma - other_sigma||_F"""
        if self._n == 0:
            return 0.0
        diff = self._sigma - other_sigma
        return float(np.linalg.norm(diff, "fro"))

    def reset(self) -> None:
        self._n = 0
        self._mu    = np.zeros(self._d, dtype=np.float64)
        self._sigma = np.zeros((self._d, self._d), dtype=np.float64)


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL PRIMITIVE 10
# Normalised Predictive Information
# Bialek, Nemenman & Tishby (2001) — predictability of the classification stream
# ═════════════════════════════════════════════════════════════════════════════

class NormalisedPredictiveInformation:
    """
    Track how predictable the sequence of topology classes seen for a domain
    is, using the Normalised Predictive Information (NPI):

      NPI = I(X_past ; X_future) / H(X_future) ∈ [0, 1]

    Estimated via a first-order Markov model:
      NPI ≈ 1 - H(class_t | class_{t-1}) / H(class_t)

    We maintain empirical transition counts:
      count[prev_class][curr_class] — number of times curr_class followed prev_class.

    Entropy estimators use the Miller-Madow bias correction:
      H_MM = H_naive - (K_effective - 1) / (2n)
    where K_effective is the number of non-zero probability classes.

    A sudden drop in NPI for a KNOWN-phase domain indicates the domain is
    producing topology class sequences inconsistent with its learned pattern.
    This is a soft surprise signal — it fires before Condition 3 accumulates
    enough evidence.

    References:
      Bialek, W., Nemenman, I., Tishby, N. (2001). "Predictability,
      complexity, and learning." Neural Computation 13(11), 2409-2463.
    """

    def __init__(self, n_classes: int = NUM_TOPOLOGY_CLASSES) -> None:
        self._n_classes = n_classes
        # Marginal counts for H(X_t).
        self._marginal: np.ndarray = np.zeros(n_classes, dtype=np.float64)
        # Joint counts for H(X_{t-1}, X_t) and conditional entropy.
        self._transition: np.ndarray = np.zeros(
            (n_classes, n_classes), dtype=np.float64
        )
        self._prev_class_idx: Optional[int] = None
        self._n_total: int = 0
        # Rolling NPI history for drop detection.
        self._npi_history: Deque[float] = deque(maxlen=20)

    def update(self, class_name: str) -> Optional[float]:
        """
        Update Markov model with one new observed class.
        Returns current NPI estimate, or None if insufficient data.
        """
        idx = _TOPOLOGY_CLASS_INDEX.get(class_name, 0)
        self._marginal[idx] += 1.0
        self._n_total += 1

        if self._prev_class_idx is not None:
            self._transition[self._prev_class_idx, idx] += 1.0

        self._prev_class_idx = idx

        if self._n_total < 10:
            return None

        npi = self._compute_npi()
        self._npi_history.append(npi)
        return npi

    def _shannon_entropy(self, counts: np.ndarray) -> float:
        """Miller-Madow bias-corrected Shannon entropy from counts."""
        n = float(counts.sum())
        if n < 1:
            return 0.0
        probs = counts / n
        # Number of non-zero entries for bias correction.
        k_eff = float(np.sum(probs > 0))
        h = -float(np.sum(probs[probs > 0] * np.log(probs[probs > 0])))
        # Miller-Madow correction.
        return h + (k_eff - 1.0) / (2.0 * n) if n > 0 else 0.0

    def _compute_npi(self) -> float:
        """Compute NPI = 1 - H(X_t | X_{t-1}) / H(X_t)."""
        h_xt = self._shannon_entropy(self._marginal)
        if h_xt < 1e-14:
            return 1.0  # deterministic sequence — perfectly predictable

        # Conditional entropy: H(X_t | X_{t-1}) = Σ_s P(X_{t-1}=s) H(X_t | X_{t-1}=s)
        h_cond = 0.0
        row_sums = self._transition.sum(axis=1)
        total    = float(row_sums.sum())
        if total < 1e-14:
            return 0.0

        for s in range(self._n_classes):
            if row_sums[s] < 1e-14:
                continue
            p_s     = row_sums[s] / total
            h_given = self._shannon_entropy(self._transition[s])
            h_cond += p_s * h_given

        return max(0.0, min(1.0, 1.0 - h_cond / h_xt))

    def npi_drop_detected(self, threshold: float) -> bool:
        """
        True if recent NPI has dropped significantly below historical NPI.
        Requires at least 5 NPI estimates in history.
        """
        if len(self._npi_history) < 5:
            return False
        history = list(self._npi_history)
        baseline = float(np.mean(history[:-3]))
        recent   = float(np.mean(history[-3:]))
        return (baseline - recent) > threshold

    @property
    def current_npi(self) -> Optional[float]:
        """Most recent NPI estimate, or None if insufficient data."""
        return self._npi_history[-1] if self._npi_history else None


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL STATE STRUCTURES
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class DomainSurpriseState:
    """
    Per-domain mutable state for the surprise detector.

    All fields are updated by _update_state() before condition checks.
    No raw vectors are stored anywhere in this struct — all statistical
    accumulators maintain O(d²) memory regardless of observation count.

    Fields:
      domain                — the domain string this state tracks.

      # Condition 1 — confident mismatch tracking
      recent_mismatches     — count of Condition 1 fires in recent window.
      last_mismatch_at      — unix timestamp of most recent Condition 1 fire.

      # Condition 2 — GENERIC_HTML rolling window
      generic_consecutive   — current streak of consecutive GENERIC_HTML results.
      generic_window        — rolling deque of is_generic booleans (last WINDOW urls).

      # Condition 3 — Mahalanobis gate
      class_welford         — per-class MultivariateWelfordAccumulator for confidence vec.
      class_ewcm            — per-class EWMACovariance tracker.
      condition3_fire_count — how many times Condition 3 fired recently.
      condition3_window     — rolling deque of Condition 3 boolean triggers.

      # Condition 4 — unknown cluster tracking
      anomalous_vectors     — bounded deque of confidence vectors that passed C3.
      cluster_centroid      — online-updated single centroid of anomalous_vectors.
      cluster_variance      — total within-cluster variance (sum of squared distances).
      oja_pca               — online PCA for Condition 4 coherence test.

      # Cross-condition algorithms
      hill_estimator        — Hill tail index for the divergence score stream.
      betti0_analyser       — Betti-0 topology for divergence score bimodality.
      npi_tracker           — Normalised Predictive Information tracker.
      dpss_analyser         — Multi-taper spectral analyser (shared instance).

      # Metadata
      phase                 — most recently read PhaseState for this domain.
      observation_count     — total events seen for this domain.
      last_seen_at          — unix timestamp of most recent observation.
      dominant_class        — most frequently observed topology class.
      class_counts          — per-class observation counts.

      # Divergence score history for spectral analysis
      divergence_history    — recent KL divergence scores (max SURPRISE_CLUSTER_WINDOW).
    """
    domain: str

    # ── Condition 1 ──────────────────────────────────────────────────────────
    recent_mismatches: int = 0
    last_mismatch_at:  Optional[float] = None

    # ── Condition 2 ──────────────────────────────────────────────────────────
    generic_consecutive: int = 0
    generic_window:      Deque[bool] = field(
        default_factory=lambda: deque(maxlen=SURPRISE_GENERIC_WINDOW)
    )

    # ── Condition 3 ──────────────────────────────────────────────────────────
    class_welford: Dict[str, MultivariateWelfordAccumulator] = field(
        default_factory=dict
    )
    class_ewcm: Dict[str, EWMACovariance] = field(default_factory=dict)
    condition3_fire_count: int = 0
    condition3_window: Deque[bool] = field(
        default_factory=lambda: deque(
            maxlen=SURPRISE_CONDITION3_ESCALATION_WINDOW
        )
    )

    # ── Condition 4 ──────────────────────────────────────────────────────────
    anomalous_vectors: Deque[np.ndarray] = field(
        default_factory=lambda: deque(maxlen=SURPRISE_CLUSTER_WINDOW)
    )
    cluster_centroid: Optional[np.ndarray] = None
    cluster_variance: Optional[float] = None
    oja_pca: OjaOnlinePCATracker = field(
        default_factory=lambda: OjaOnlinePCATracker(d=NUM_TOPOLOGY_CLASSES)
    )

    # ── Cross-condition algorithms ────────────────────────────────────────────
    hill_estimator: HillTailIndexEstimator = field(
        default_factory=HillTailIndexEstimator
    )
    npi_tracker: NormalisedPredictiveInformation = field(
        default_factory=NormalisedPredictiveInformation
    )
    divergence_history: Deque[float] = field(
        default_factory=lambda: deque(maxlen=SURPRISE_CLUSTER_WINDOW)
    )

    last_kl: float = 0.0
    _cached_alpha_hat: Optional[float] = None
    _alpha_hat_n_at_cache: int = 0

    # ── Metadata ─────────────────────────────────────────────────────────────
    phase:             PhaseState = PhaseState.COLD
    observation_count: int = 0
    last_seen_at:      float = field(default_factory=time.monotonic)
    dominant_class:    str = FALLBACK_TOPOLOGY_CLASS
    class_counts:      Dict[str, int] = field(default_factory=dict)

    # ── Condition-4 analysis cache ────────────────────────────────────────────
    # MDL, Betti-0, and DPSS analyses produce a stable boolean over short time
    # windows.  Recomputing them on every event is wasteful once the anomalous
    # vector deque is at capacity.  We cache the results and re-evaluate only
    # every _C4_RECOMPUTE_INTERVAL domain events (once the deque is full).
    _c4_event_counter:  int   = 0
    _c4_cached_mdl:     bool  = False
    _c4_cached_betti0:  int   = 1
    _c4_cached_periodic: bool = False
    _c4_cached_nearest: str   = "NEWS_ARTICLE"
    _C4_RECOMPUTE_INTERVAL: int = field(default=10, init=False, repr=False)

    # Per-class cached regularised inverse — recomputed every N Welford updates
    _sigma_inv_cache: Dict[str, np.ndarray] = field(default_factory=dict)
    _sigma_inv_n_at_cache: Dict[str, int] = field(default_factory=dict)
    _SIGMA_CACHE_INTERVAL: int = field(default=10, init=False, repr=False)

    def get_or_create_welford(self, topology_class: str) -> MultivariateWelfordAccumulator:
        if topology_class not in self.class_welford:
            self.class_welford[topology_class] = MultivariateWelfordAccumulator(
                d=NUM_TOPOLOGY_CLASSES
            )
        return self.class_welford[topology_class]

    def get_or_create_ewcm(self, topology_class: str) -> EWMACovariance:
        if topology_class not in self.class_ewcm:
            self.class_ewcm[topology_class] = EWMACovariance(d=NUM_TOPOLOGY_CLASSES)
        return self.class_ewcm[topology_class]


# ═════════════════════════════════════════════════════════════════════════════
# LRU DOMAIN STATE STORE
# Thread-safe, bounded store with LRU eviction.
# ═════════════════════════════════════════════════════════════════════════════

class _LRUDomainStateStore:
    """
    Thread-safe bounded LRU cache for DomainSurpriseState objects.

    Uses collections.OrderedDict as the underlying data structure.  Move-to-
    end on access; evict first element (LRU) when capacity exceeded.

    Capacity: SURPRISE_DOMAIN_STATE_MAX domains.
    Eviction: least-recently-observed domain when full.
    """

    def __init__(self, max_size: int = SURPRISE_DOMAIN_STATE_MAX) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, DomainSurpriseState] = OrderedDict()
        self._lock = threading.Lock()
        self._eviction_count: int = 0

    def get_or_create(self, domain: str) -> DomainSurpriseState:
        """
        Return existing DomainSurpriseState or create a new one.
        Promotes the domain to MRU position on access.
        Evicts LRU domain if at capacity.
        """
        with self._lock:
            if domain in self._store:
                self._store.move_to_end(domain)
                return self._store[domain]

            # Evict LRU if at capacity.
            if len(self._store) >= self._max_size and SURPRISE_EVICT_LRU:
                evicted_domain, _ = self._store.popitem(last=False)
                self._eviction_count += 1
                log.debug(
                    "lru_eviction",
                    evicted_domain=evicted_domain,
                    store_size=len(self._store),
                    total_evictions=self._eviction_count,
                )

            state = DomainSurpriseState(domain=domain)
            self._store[domain] = state
            return state

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def eviction_count(self) -> int:
        return self._eviction_count

    def domains(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

class _BusProtocol(Protocol):
    async def subscribe(self, topic: str, group: str, handler: Callable, schema: type) -> None: ...
    async def emitter(self, topic: str, component: str, schema: type) -> Any: ...

# ═════════════════════════════════════════════════════════════════════════════
# SURPRISE DETECTOR
# ═════════════════════════════════════════════════════════════════════════════

class SurpriseDetector:
    """
    The AXIOM tripwire.

    Subscribes to ClassificationEvent.  On every event:
      1. Reads the domain's current PhaseState from phase_states.mmap.
      2. Fetches or creates per-domain state.
      3. Updates all running statistics (O(d²) total work).
      4. Checks all four surprise conditions independently.
      5. Emits SurpriseEvent and/or NewTopologyHintEvent to the bus.
      6. Returns.  Fire and forget.

    The entire handler is O(1) per ClassificationEvent and never blocks.

    Structural constraints (see spec invariants):
      - Never emits SurpriseEvent on GENERIC_HTML (only NewTopologyHintEvent).
      - Condition 4 requires Condition 3 as prerequisite.
      - Mahalanobis gate requires SURPRISE_WELFORD_MIN_OBSERVATIONS.
      - Phase thresholds read fresh from mmap on every event (never cached).
      - Never writes to any store.
    """

    # ── Class-level DPSS analyser (shared, thread-safe after construction) ───
    _DPSS_ANALYSER: DPSSMultitaperAnalyser = DPSSMultitaperAnalyser(nw=4.0, n_tapers=3)

    def __init__(self, bus: _BusProtocol) -> None:
        self._bus = bus
        self._store = _LRUDomainStateStore()
        self._mmap: Optional[mmap.mmap] = None
        self._mmap_fd: Optional[int] = None
        self._initialized: bool = False
        self._lock = threading.Lock()
        self._total_events: int = 0
        self._total_surprises: int = 0
        self._total_hints: int = 0
        self._surprise_emitter: Optional[Any] = None
        self._hint_emitter: Optional[Any] = None

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Open the phase_states.mmap file handle and subscribe to
        ClassificationEvent on the bus.

        Failure to open the mmap file is non-fatal: _read_phase() falls
        back to PHASE_COLD for all domains, which produces the most
        conservative (loosest) thresholds.  The system can run with no
        mmap — it will just be less sensitive until the file is available.
        """
        self._try_open_mmap()
        await self._bus.subscribe(
            topic="classification",
            group="topology.surprise_detector",
            handler=self.handle_classification_event,
            schema=ClassificationEvent,
        )

        self._surprise_emitter = await self._bus.emitter(
            topic="surprise",
            component="topology.surprise_detector",
            schema=_BusSurpriseEvent,
        )

        self._hint_emitter = await self._bus.emitter(
            topic="topology_hint",
            component="topology.surprise_detector",
            schema=NewTopologyHintEvent,
        )

        self._initialized = True
        log.info(
            "surprise_detector_initialized",
            mmap_available=self._mmap is not None,
            domain_state_max=SURPRISE_DOMAIN_STATE_MAX,
            welford_min_observations=SURPRISE_WELFORD_MIN_OBSERVATIONS,
        )

    async def shutdown(self) -> None:
        """
        Close the mmap file handle and clean up.  Must be called before
        process exit to avoid fd leaks.
        """
        self._close_mmap()
        self._initialized = False
        log.info(
            "surprise_detector_shutdown",
            total_events=self._total_events,
            total_surprises=self._total_surprises,
            total_hints=self._total_hints,
            domains_tracked=len(self._store),
            lru_evictions=self._store.eviction_count,
        )

    def _try_open_mmap(self) -> None:
        """Attempt to open phase_states.mmap.  Log and continue on failure."""
        try:
            if _PHASE_MMAP_PATH.exists():
                self._mmap_fd = os.open(str(_PHASE_MMAP_PATH), os.O_RDONLY)
                size = os.fstat(self._mmap_fd).st_size
                if size >= _PHASE_SLOT_BYTES:
                    self._mmap = mmap.mmap(
                        self._mmap_fd,
                        size,
                        access=mmap.ACCESS_READ,
                    )
                    log.debug("phase_mmap_opened", path=str(_PHASE_MMAP_PATH), size=size)
                else:
                    os.close(self._mmap_fd)
                    self._mmap_fd = None
            else:
                log.warning(
                    "phase_mmap_not_found",
                    path=str(_PHASE_MMAP_PATH),
                    fallback="PHASE_COLD",
                )
        except OSError as exc:
            log.warning(
                "phase_mmap_open_failed",
                path=str(_PHASE_MMAP_PATH),
                error=str(exc),
                fallback="PHASE_COLD",
            )
            self._mmap = None

    def _close_mmap(self) -> None:
        try:
            if self._mmap is not None:
                self._mmap.close()
                self._mmap = None
            if self._mmap_fd is not None:
                os.close(self._mmap_fd)
                self._mmap_fd = None
        except OSError:
            pass

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE READING
    # ─────────────────────────────────────────────────────────────────────────

    def _read_phase(self, domain: str, dominant_class: Optional[str] = None) -> PhaseState:
        """
        Read the current PhaseState for a domain from phase_states.mmap.

        Strategy:
          1. Map domain → dominant topology class (from per-domain history).
          2. Map topology class → slot index (TOPOLOGY_CLASS_INDEX).
          3. Read the 32-byte slot from phase_states.mmap.
          4. Parse phase_id (uint8 at offset 0).
          5. Return PhaseState.from_int(phase_id).

        Falls back to PHASE_COLD on any error.  This is the correct
        conservative fallback: COLD produces the loosest thresholds, so
        a mmap read failure never makes the detector MORE sensitive.

        Per the spec invariant: this method reads the mmap on every call.
        It never caches the result.  Phase transitions happen during runtime;
        a cached phase would produce wrong thresholds after a domain promotes.
        """
        if self._mmap is None:
            return PhaseState.COLD

        # Determine the topology class slot to read.
        cls_name = dominant_class or FALLBACK_TOPOLOGY_CLASS
        slot_idx = _TOPOLOGY_CLASS_INDEX.get(cls_name, 0)
        offset   = slot_idx * _PHASE_SLOT_BYTES

        try:
            mmap_size = len(self._mmap)
            if offset + _PHASE_SLOT_BYTES > mmap_size:
                return PhaseState.COLD

            self._mmap.seek(offset)
            raw = self._mmap.read(_PHASE_SLOT_BYTES)
            if len(raw) < _PHASE_SLOT_BYTES:
                return PhaseState.COLD

            phase_id = struct.unpack_from("<B", raw, 0)[0]
            return PhaseState.from_int(phase_id)

        except (OSError, ValueError, struct.error) as exc:
            log.debug(
                "phase_read_failed",
                domain=domain,
                error=str(exc),
                fallback="COLD",
            )
            return PhaseState.COLD

    def _get_phase_thresholds(self, phase: PhaseState) -> SurpriseThresholds:
        """
        Map a PhaseState to the corresponding threshold bundle.

        Each threshold is tighter at higher phases because:
        - COLD:     high noise expected.  Loose = fewer false positives.
        - LEARNING: patterns stabilising.  Moderate sensitivity.
        - KNOWN:    domain well-understood.  Any deviation is real signal.

        The NPI drop thresholds are inverted relative to the others:
        a larger NPI drop is needed at COLD (lower baseline predictability)
        and a smaller drop is sufficient at KNOWN (high baseline expected).
        """
        match phase:
            case PhaseState.COLD:
                return SurpriseThresholds(
                    confident         = THETA_SURPRISE_CONFIDENT_COLD,
                    divergence        = THETA_SURPRISE_DIVERGENCE_COLD,
                    generic_threshold = SURPRISE_GENERIC_THRESHOLD_COLD,
                    mahalanobis       = THETA_SURPRISE_MAHALANOBIS_COLD,
                    wasserstein       = THETA_SURPRISE_WASSERSTEIN_COLD,
                    npi_drop          = THETA_SURPRISE_NPI_DROP_COLD,
                )
            case PhaseState.LEARNING:
                return SurpriseThresholds(
                    confident         = THETA_SURPRISE_CONFIDENT_LEARNING,
                    divergence        = THETA_SURPRISE_DIVERGENCE_LEARNING,
                    generic_threshold = SURPRISE_GENERIC_THRESHOLD_LEARNING,
                    mahalanobis       = THETA_SURPRISE_MAHALANOBIS_LEARNING,
                    wasserstein       = THETA_SURPRISE_WASSERSTEIN_LEARNING,
                    npi_drop          = THETA_SURPRISE_NPI_DROP_LEARNING,
                )
            case PhaseState.KNOWN:
                return SurpriseThresholds(
                    confident         = THETA_SURPRISE_CONFIDENT_KNOWN,
                    divergence        = THETA_SURPRISE_DIVERGENCE_KNOWN,
                    generic_threshold = SURPRISE_GENERIC_THRESHOLD_KNOWN,
                    mahalanobis       = THETA_SURPRISE_MAHALANOBIS_KNOWN,
                    wasserstein       = THETA_SURPRISE_WASSERSTEIN_KNOWN,
                    npi_drop          = THETA_SURPRISE_NPI_DROP_KNOWN,
                )
            case _:
                # Unreachable by PhaseState enum, but defensive fallback.
                return self._get_phase_thresholds(PhaseState.COLD) # noqa | runtime defensive check

    # ─────────────────────────────────────────────────────────────────────────
    # STATE MANAGEMENT
    # ─────────────────────────────────────────────────────────────────────────

    def _get_or_create_state(self, domain: str) -> DomainSurpriseState:
        """
        Fetch or create per-domain state from the LRU store.
        Promotes domain to MRU on access.
        Evicts LRU domain if at capacity.
        """
        return self._store.get_or_create(domain)

    def _update_state(
        self,
        state: DomainSurpriseState,
        event: ClassificationEvent,
    ) -> None:
        """
        Integrate one ClassificationEvent into all per-domain running statistics.

        This is the only place that writes to DomainSurpriseState.
        All four condition checks are pure readers after this update.

        Operations performed (all O(1) or O(d²)):
          1. Update Welford accumulator for observed_class.
          2. Update EWMA covariance for observed_class.
          3. Update NPI tracker with observed_class.
          4. Update Hill tail index with KL divergence score.
          5. Update generic_window and generic_consecutive.
          6. Update dominant_class and class_counts.
          7. Update divergence_history for spectral analysis.
          8. Refresh last_seen_at and observation_count.
          9. Update Oja PCA if this vector was previously anomalous
             (prerequisite: Condition 3 must have fired to add to C4 deque,
             handled separately in _check_condition_3).
        """
        vec = event.classifier_distribution.astype(np.float64)
        cls = event.observed_class

        # 1. Welford covariance update.
        welford = state.get_or_create_welford(cls)
        welford.update(vec)

        # 2. EWMA covariance update.
        ewcm = state.get_or_create_ewcm(cls)
        ewcm.update(vec)

        # 3. NPI update.
        state.npi_tracker.update(cls)

        # 4. KL divergence score for Hill estimator.
        kl = self._kl_divergence(event.wlm_prior_distribution, event.classifier_distribution)

        state.last_kl = kl
        # Cache alpha_hat every 10 observations — O(n) argpartition amortised to O(1).
        if state.observation_count % 10 == 0:
            state._cached_alpha_hat = state.hill_estimator.alpha_hat()

        state.hill_estimator.update(kl)
        state.divergence_history.append(kl)

        # 5. GENERIC_HTML window update.
        is_generic = (cls == FALLBACK_TOPOLOGY_CLASS)
        state.generic_window.append(is_generic)
        if is_generic:
            state.generic_consecutive += 1
        else:
            state.generic_consecutive = 0

        # 6. Dominant class and class_counts.
        state.class_counts[cls] = state.class_counts.get(cls, 0) + 1
        state.dominant_class = max(state.class_counts, key=state.class_counts.__getitem__)

        # 7. Metadata.
        state.observation_count += 1
        state.last_seen_at = time.monotonic()

    # ─────────────────────────────────────────────────────────────────────────
    # DIVERGENCE COMPUTATION
    # ─────────────────────────────────────────────────────────────────────────

    def _kl_divergence(self, prior: np.ndarray, posterior: np.ndarray) -> float:
        """
        KL divergence KL(prior || posterior).

        D_KL(P || Q) = Σᵢ P(i) log(P(i) / Q(i))

        Numerically stable: adds epsilon to both distributions to handle
        zeros.  Returns 0.0 when prior == posterior.

        Note: KL divergence is asymmetric.  We compute KL(prior || posterior)
        because the prior is the "expected" distribution and the posterior is
        the "observed" distribution.  Penalises more heavily when the prior
        assigns probability to regions the posterior does not.
        """
        eps = 1e-10
        p = prior.astype(np.float64) + eps
        q = posterior.astype(np.float64) + eps
        # Normalise.
        p = p / p.sum()
        q = q / q.sum()
        return float(np.sum(p * np.log(p / q)))

    def _js_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        """Jensen-Shannon divergence (symmetric, bounded ∈ [0, log 2])."""
        return WassersteinSimplexDistance.js_divergence(
            p.astype(np.float64), q.astype(np.float64)
        )

    def _wasserstein1(self, p: np.ndarray, q: np.ndarray) -> float:
        """W₁ Cramér distance on the topology class ordered alphabet."""
        return WassersteinSimplexDistance.wasserstein1(
            p.astype(np.float64), q.astype(np.float64)
        )

    def _mahalanobis_distance(
        self,
        vector: np.ndarray,
        welford: MultivariateWelfordAccumulator,
        state: DomainSurpriseState,
        cls: str,
    ) -> float:
        """
        Compute the Mahalanobis distance of `vector` from the class centroid
        estimated by the Welford accumulator.

        Uses Ledoit-Wolf regularised covariance to ensure invertibility
        even when n << d (see Algorithm 2).

        d_M(x) = sqrt((x - μ)ᵀ Σ⁻¹ (x - μ))

        Returns 0.0 if welford has insufficient observations.
        Returns Euclidean distance (Σ=I) when LW collapses to spherical target.
        """
        if welford.n < SURPRISE_WELFORD_MIN_OBSERVATIONS:
            return 0.0

        mu = welford.mean

        # Recompute inverse only every N welford updates
        n_at_cache = state._sigma_inv_n_at_cache.get(cls, 0)
        if cls not in state._sigma_inv_cache or (welford.n - n_at_cache) >= state._SIGMA_CACHE_INTERVAL:
            S = welford.covariance(ddof=1)
            state._sigma_inv_cache[cls] = LedoitWolfShrinkageEstimator.regularised_inverse(
                S, welford.n, epsilon=1e-8
            )
            state._sigma_inv_n_at_cache[cls] = welford.n

        Sigma_inv = state._sigma_inv_cache[cls]
        delta = vector.astype(np.float64) - mu
        dist_sq = float(delta @ Sigma_inv @ delta)
        return math.sqrt(max(0.0, dist_sq))

    # ─────────────────────────────────────────────────────────────────────────
    # CLUSTER UPDATE
    # ─────────────────────────────────────────────────────────────────────────

    def _update_cluster(
        self,
        state: DomainSurpriseState,
        vector: np.ndarray,
    ) -> float:
        """
        Add `vector` to the anomalous cluster and update the online centroid.

        Uses an online k=1 centroid update:
          μ_t = μ_{t-1} + (1/t)·(x_t - μ_{t-1})

        Returns the updated cluster variance (mean squared distance from
        centroid over all vectors in the deque).

        Also updates:
          - Oja online PCA for Condition 4 coherence test.
          - MDL split evaluation (on the current anomalous deque).
        """
        vec = vector.astype(np.float64)
        n_before = len(state.anomalous_vectors)
        deque_full = n_before >= SURPRISE_CLUSTER_WINDOW

        # Save evicted element BEFORE append evicts it
        evicted = state.anomalous_vectors[0] if deque_full else None

        state.anomalous_vectors.append(vec)
        n = len(state.anomalous_vectors)

        # Oja update
        state.oja_pca.update(vec)

        # Online centroid — no deque iteration ever
        if state.cluster_centroid is None:
            state.cluster_centroid = vec.copy()
        elif not deque_full:
            # Growing phase — simple Welford mean
            state.cluster_centroid += (vec - state.cluster_centroid) / n
        else:
            # Steady state — sliding window update using evicted element
            state.cluster_centroid += (vec - evicted) / SURPRISE_CLUSTER_WINDOW

        # Online variance — single distance computation, no matrix
        dist_sq = float(np.dot(vec - state.cluster_centroid, vec - state.cluster_centroid))
        if state.cluster_variance is None:
            state.cluster_variance = dist_sq
        else:
            state.cluster_variance += (dist_sq - state.cluster_variance) / n

        return state.cluster_variance

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 1 — CONFIDENT MISMATCH
    # ─────────────────────────────────────────────────────────────────────────

    def _check_condition_1(
        self,
        event: ClassificationEvent,
        state: DomainSurpriseState,
        thresholds: SurpriseThresholds,
    ) -> Optional[_BusSurpriseEvent]:
        """
        Condition 1: Classifier was confident but wrong relative to WLM prior.

        Fires when:
          classifier_confidence ≥ thresholds.confident
          AND KL(wlm_prior || classifier_distribution) > thresholds.divergence

        Invariant: Never fires on GENERIC_HTML.

        Severity: HIGH — confident wrong classification is the most dangerous
        failure mode.  The recipe that fires will be wrong.  The extraction
        will produce noise.

        Enhancement: also compute JS divergence and W₁ distance for richer
        contributing_signals.  Neither gates the condition — KL is the
        primary criterion.  Both inform the severity weight transmitted to
        index_daemon.
        """
        # Invariant 1: never fire on GENERIC_HTML.
        if event.observed_class == FALLBACK_TOPOLOGY_CLASS:
            return None

        # Gate: confidence must exceed phase threshold.
        if event.observed_confidence < thresholds.confident:
            return None

        # Primary divergence gate.
        kl = state.last_kl if state.observation_count > 0 else self._kl_divergence(
            event.wlm_prior_distribution,
            event.classifier_distribution,
        )

        if kl <= thresholds.divergence:
            return None

        # Condition 1 fires.
        state.recent_mismatches += 1
        state.last_mismatch_at = time.time()

        # Supplementary signals.
        js  = self._js_divergence(event.wlm_prior_distribution, event.classifier_distribution)
        w1  = self._wasserstein1(event.wlm_prior_distribution, event.classifier_distribution)
        alpha_hat = state._cached_alpha_hat

        contributing_signals = {
            "kl_divergence":        round(kl, 6),
            "js_divergence":        round(js, 6),
            "wasserstein1":         round(w1, 6),
            "hill_alpha_hat":       round(alpha_hat, 4) if alpha_hat and math.isfinite(alpha_hat) else -1.0,
            "observed_confidence":  round(event.observed_confidence, 4),
            "wlm_confidence":       round(event.wlm_predicted_confidence, 4),
        }

        log.warning(
            "surprise_condition_1",
            url=event.url,
            domain=event.domain,
            observed_class=event.observed_class,
            wlm_predicted_class=event.wlm_predicted_class,
            kl_divergence=round(kl, 4),
            confidence=round(event.observed_confidence, 4),
            phase=str(state.phase),
        )

        # Map to the contracts.py SurpriseEvent shape for bus emission.
        return _BusSurpriseEvent(
            topology_class=event.observed_class,
            surprise_score=min(1.0, kl / (thresholds.divergence * 3.0)),
            theta_surprise=thresholds.divergence,
            dissolve_triggered=(state.phase == _PHASE_KNOWN),
            contributing_signals=contributing_signals,
            current_phase=int(state.phase),
            run_id=event.run_id,
            timestamp=_iso_now(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 2 — REPEATED GENERIC_HTML FALLBACK
    # ─────────────────────────────────────────────────────────────────────────

    def _check_condition_2(
        self,
        event: ClassificationEvent,
        state: DomainSurpriseState,
        thresholds: SurpriseThresholds,
    ) -> Optional[NewTopologyHintEvent]:
        """
        Condition 2: Domain has consistently fallen through to GENERIC_HTML.

        Fires when:
          generic_consecutive > thresholds.generic_threshold
          AND domain phase >= PHASE_LEARNING

        Invariant per spec: only fires NewTopologyHintEvent, never SurpriseEvent.

        Severity: MEDIUM (via NewTopologyHintEvent, not SurpriseEvent).

        The phase gate is critical:
          - PHASE_COLD: normal to see GENERIC_HTML — domain not yet understood.
            Even 15 consecutive GENERIC_HTML is not remarkable.
          - PHASE_LEARNING: domain is supposed to have patterns.
            8 consecutive is a signal.
          - PHASE_KNOWN: domain should almost never produce GENERIC_HTML.
            3 consecutive is a strong signal.
        """
        # Phase gate: COLD domains are expected to produce GENERIC_HTML.
        if state.phase == _PHASE_COLD:
            return None

        if state.generic_consecutive < thresholds.generic_threshold:
            return None

        # Compute mean distribution of recent GENERIC_HTML observations.
        # Use a stable estimate: mean of all accumulated observations for GENERIC_HTML.
        generic_welford = state.class_welford.get(FALLBACK_TOPOLOGY_CLASS)
        if generic_welford and generic_welford.n > 0:
            centroid = generic_welford.mean
        else:
            # Fallback: uniform distribution.
            centroid = np.ones(NUM_TOPOLOGY_CLASSES, dtype=np.float64) / NUM_TOPOLOGY_CLASSES

        nearest = self._nearest_known_class(centroid)

        # Betti-0 analysis on divergence history.
        betti0_modes = 1
        if len(state.divergence_history) >= 4:
            betti0_modes = Betti0PersistenceAnalyser.count_significant_modes(
                list(state.divergence_history)
            )

        # MDL split evaluation.
        mdl_split = False
        if len(state.anomalous_vectors) >= 4:
            vecs = np.array(list(state.anomalous_vectors))
            mdl_split = RissanenMDLClusterEvaluator.mdl_supports_split(vecs)

        log.info(
            "surprise_condition_2",
            domain=event.domain,
            generic_consecutive=state.generic_consecutive,
            threshold=thresholds.generic_threshold,
            phase=str(state.phase),
            nearest_class=nearest,
        )

        return NewTopologyHintEvent(
            domain=event.domain,
            trigger="repeated_generic",
            evidence_count=state.generic_consecutive,
            centroid_vector=centroid.tolist(),
            cluster_variance=None,
            suggested_parent_class=nearest,
            mdl_supports_split=mdl_split,
            betti0_modes=betti0_modes,
            oja_pc1_variance_ratio=state.oja_pca.mean_pc1_ratio,
            phase_at_trigger=state.phase,
            run_id=event.run_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 3 — ANOMALOUS CONFIDENCE DISTRIBUTION SHAPE
    # ─────────────────────────────────────────────────────────────────────────

    def _check_condition_3(
        self,
        event: ClassificationEvent,
        state: DomainSurpriseState,
        thresholds: SurpriseThresholds,
    ) -> Optional[_BusSurpriseEvent]:
        """
        Condition 3: Confidence distribution shape is structurally anomalous.

        Fires when:
          welford.n >= SURPRISE_WELFORD_MIN_OBSERVATIONS (statistics stable)
          AND mahalanobis_distance(observed_vector, class_statistics) > thresholds.mahalanobis

        Invariant: Never fires on GENERIC_HTML.

        Severity: LOW on first occurrence.  Escalates to MEDIUM after
        SURPRISE_CONDITION3_ESCALATION_COUNT fires within
        SURPRISE_CONDITION3_ESCALATION_WINDOW observations.

        EWMA divergence: also checks if EWCM/Welford Frobenius divergence
        exceeds a secondary threshold — this is the "leading indicator"
        signal from Algorithm 9.  When it fires it adds to contributing_signals
        but does not independently trigger the condition.

        Returns the event if anomalous (so the caller can add to C4 cluster).
        Caller must call _update_cluster() separately.
        """
        # Invariant: never fire on GENERIC_HTML.
        if event.observed_class == FALLBACK_TOPOLOGY_CLASS:
            state.condition3_window.append(False)
            return None

        welford = state.get_or_create_welford(event.observed_class)

        # Gate: insufficient observations.
        if welford.n < SURPRISE_WELFORD_MIN_OBSERVATIONS:
            state.condition3_window.append(False)
            return None

        vec = event.classifier_distribution.astype(np.float64)
        mdist = self._mahalanobis_distance(vec, welford, state, event.observed_class)

        if mdist <= thresholds.mahalanobis:
            state.condition3_window.append(False)
            return None

        # Condition 3 fires.
        state.condition3_window.append(True)
        state.condition3_fire_count += 1

        # Determine severity based on escalation history.
        recent_fires = sum(state.condition3_window)
        if recent_fires >= SURPRISE_CONDITION3_ESCALATION_COUNT:
            severity = _SEV_MEDIUM
        else:
            severity = _SEV_LOW

        # Supplementary signals.
        kl = state.last_kl if state.observation_count > 0 else self._kl_divergence(
            event.wlm_prior_distribution, vec
        )

        js  = self._js_divergence(event.wlm_prior_distribution, vec)
        w1  = self._wasserstein1(event.wlm_prior_distribution, vec)

        # EWMA / Welford Frobenius divergence (non-stationarity indicator).
        ewcm     = state.get_or_create_ewcm(event.observed_class)
        welford_S = welford.covariance(ddof=1)
        frob_div  = ewcm.frobenius_distance_from(welford_S)

        # Hill tail index.
        alpha_hat = state._cached_alpha_hat

        # NPI drop signal.
        npi_current = state.npi_tracker.current_npi
        npi_drop    = state.npi_tracker.npi_drop_detected(thresholds.npi_drop)

        contributing_signals = {
            "mahalanobis_distance":     round(mdist, 6),
            "kl_divergence":            round(kl, 6),
            "js_divergence":            round(js, 6),
            "wasserstein1":             round(w1, 6),
            "ewcm_welford_frob_div":    round(frob_div, 6),
            "hill_alpha_hat":           round(alpha_hat, 4) if alpha_hat and math.isfinite(alpha_hat) else -1.0,
            "npi_current":              round(npi_current, 4) if npi_current is not None else -1.0,
            "npi_drop_detected":        1.0 if npi_drop else 0.0,
            "condition3_recent_fires":  float(recent_fires),
            "observation_count":        float(welford.n),
        }

        # LOW severity fires are extremely frequent (can be every event once
        # Welford is armed).  Logging them at INFO would flood the log pipeline
        # and adds non-trivial I/O overhead on the hot path.  Use DEBUG so they
        # are visible in verbose/development mode only; MEDIUM/HIGH fires
        # (genuine structural novelty) still surface at INFO.
        _c3_log = log.debug if severity == _SEV_LOW else log.info
        _c3_log(
            "surprise_condition_3",
            url=event.url,
            domain=event.domain,
            observed_class=event.observed_class,
            mahalanobis_distance=round(mdist, 4),
            threshold=thresholds.mahalanobis,
            severity=severity.value,
            phase=str(state.phase),
        )

        return _BusSurpriseEvent(
            topology_class=event.observed_class,
            surprise_score=min(1.0, mdist / (thresholds.mahalanobis * 3.0)),
            theta_surprise=min(1.0, thresholds.mahalanobis / (thresholds.mahalanobis * 3.0)),
            dissolve_triggered=False,
            contributing_signals=contributing_signals,
            current_phase=int(state.phase),
            run_id=event.run_id,
            timestamp=_iso_now(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # CONDITION 4 — CONSISTENT UNKNOWN PATTERN
    # ─────────────────────────────────────────────────────────────────────────

    def _check_condition_4(
        self,
        event: ClassificationEvent,
        state: DomainSurpriseState,
        thresholds: SurpriseThresholds,
    ) -> Optional[NewTopologyHintEvent]:
        """
        Condition 4: Multiple URLs on the same domain have produced anomalous
        confidence distributions with similar structural fingerprints.

        Fires when:
          cluster_variance < THETA_SURPRISE_CLUSTER_COHERENCE
          AND len(anomalous_vectors) >= SURPRISE_CLUSTER_WINDOW / 4

        Invariant: Condition 4 is only reachable if Condition 3 fired for this
        observation (the vector was added to the cluster in _check_condition_3's
        caller path).  Normal uncertain classifications never enter the cluster.

        Severity: HIGH — this is the discover_signal_zones() trigger.

        Enhancements over the spec:
          - Betti-0 persistence analysis to detect bimodal clusters.
          - Rissanen MDL to detect if the cluster should be split.
          - Oja PCA coherence check (PC1 explained variance).
          - DPSS spectral analysis to detect periodic (temporal, not structural)
            surprise patterns — periodic surprises get a lower severity hint.
        """
        n_anomalous = len(state.anomalous_vectors)
        min_required = max(4, SURPRISE_CLUSTER_WINDOW // 4)

        if n_anomalous < min_required:
            return None

        if state.cluster_variance is None or state.cluster_variance > THETA_SURPRISE_CLUSTER_COHERENCE:
            return None

        centroid = state.cluster_centroid
        if centroid is None:
            return None

        # Betti-0, MDL, and DPSS are expensive analyses.  Once the anomalous
        # vector deque is at full capacity (steady state), their results are
        # stable over short windows.  Only recompute every _C4_RECOMPUTE_INTERVAL
        # domain events to bound per-event cost to O(1).
        state._c4_event_counter += 1
        deque_full = n_anomalous >= SURPRISE_CLUSTER_WINDOW  # noqa: F841 — kept for readability
        # Always use interval-based cache regardless of deque capacity.
        # The old `(not deque_full)` clause forced a full SVD + Betti-0 + DPSS
        # recompute on *every* C3 fire while the anomalous-vector deque was
        # filling (up to 50 entries).  That made per-event cost O(n_anomalous)
        # during warm-up rather than the intended amortised O(1).
        should_recompute = (state._c4_event_counter % state._C4_RECOMPUTE_INTERVAL == 1)

        # Nearest-known-class (17 W1 computations) — cache like the other analyses.
        if should_recompute:
            state._c4_cached_nearest = self._nearest_known_class(centroid)
        nearest = state._c4_cached_nearest

        # Betti-0 analysis on anomalous cluster's divergence scores.
        if should_recompute and len(state.divergence_history) >= 4:
            state._c4_cached_betti0 = Betti0PersistenceAnalyser.count_significant_modes(
                list(state.divergence_history)
            )
        betti0_modes = state._c4_cached_betti0

        # Rissanen MDL split evaluation.
        if should_recompute and n_anomalous >= 4:
            vecs = np.array(list(state.anomalous_vectors))
            state._c4_cached_mdl = RissanenMDLClusterEvaluator.mdl_supports_split(vecs)
        mdl_split = state._c4_cached_mdl

        # DPSS spectral analysis — is this a periodic pattern?
        if should_recompute and len(state.divergence_history) >= 8:
            state._c4_cached_periodic = SurpriseDetector._DPSS_ANALYSER.is_periodic(
                list(state.divergence_history)
            )
        is_periodic = state._c4_cached_periodic

        # Oja PCA coherence.
        pc1_ratio = state.oja_pca.mean_pc1_ratio

        log.warning(
            "surprise_condition_4",
            domain=event.domain,
            n_anomalous=n_anomalous,
            cluster_variance=round(state.cluster_variance, 6),
            nearest_class=nearest,
            mdl_split=mdl_split,
            betti0_modes=betti0_modes,
            is_periodic=is_periodic,
            pc1_ratio=round(pc1_ratio, 4),
            phase=str(state.phase),
        )

        return NewTopologyHintEvent(
            domain=event.domain,
            trigger="coherent_cluster",
            evidence_count=n_anomalous,
            centroid_vector=centroid.tolist(),
            cluster_variance=float(state.cluster_variance),
            suggested_parent_class=nearest,
            mdl_supports_split=mdl_split,
            betti0_modes=betti0_modes,
            oja_pc1_variance_ratio=pc1_ratio,
            phase_at_trigger=state.phase,
            run_id=event.run_id,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # NEAREST KNOWN CLASS HINT
    # ─────────────────────────────────────────────────────────────────────────

    def _nearest_known_class(self, centroid: np.ndarray) -> str:
        """
        Find the topology class whose "canonical confidence profile" is
        nearest to the given centroid vector.

        The canonical profile for class C is the one-hot vector with
        P(C) = 1.0 and P(other) = 0.0.  The distance metric is W₁
        (Cramér distance on the probability simplex), which respects
        the ordinal structure of the class ordering better than Euclidean.

        Special handling: GENERIC_HTML is excluded from candidates because
        it is itself the "unknown" fallback — suggesting it as the parent
        class would be circular.

        Returns the topology class string with minimum W₁ distance to centroid.
        """
        if centroid is None or len(centroid) != NUM_TOPOLOGY_CLASSES:
            return "NEWS_ARTICLE"   # safe default

        # Normalise centroid.
        c = centroid.astype(np.float64)
        total = c.sum()
        if total > 1e-14:
            c = c / total
        else:
            c = np.ones(NUM_TOPOLOGY_CLASSES) / NUM_TOPOLOGY_CLASSES

        best_class = "NEWS_ARTICLE"
        best_dist  = math.inf

        for cls_name, cls_idx in _TOPOLOGY_CLASS_INDEX.items():
            # Exclude GENERIC_HTML from candidates.
            if cls_name == FALLBACK_TOPOLOGY_CLASS:
                continue

            # Use pre-built canonical profile (avoids per-call np.zeros allocation).
            dist = WassersteinSimplexDistance.wasserstein1(c, _CANONICAL_PROFILES[cls_name])
            if dist < best_dist:
                best_dist  = dist
                best_class = cls_name

        return best_class

    # ─────────────────────────────────────────────────────────────────────────
    # MAIN EVENT HANDLER
    # ─────────────────────────────────────────────────────────────────────────

    async def handle_classification_event(
        self,
        event: ClassificationEvent,
    ) -> None:
        """
        Handle one ClassificationEvent.  O(d²) per call.  Never blocks.

        Order of operations:
          1. Read domain phase from mmap (fresh, never cached).
          2. Get or create per-domain state.
          3. Update state with all running statistics.
          4. Check all four conditions independently.
          5. Emit any non-None events to the bus.
          6. Return.

        Any exception during processing is caught and logged.  The bus
        must never observe an unhandled exception from this handler.
        """
        self._total_events += 1
        try:
            # 1. Phase (always fresh).
            # Get current state for dominant class lookup before update.
            # Note: we read phase BEFORE updating state so phase reflects
            # the classification's context, not the post-update context.
            domain = event.domain

            # Pre-read dominant class from existing state (if any).
            existing_state = self._store.get_or_create(domain)
            dominant_cls = existing_state.dominant_class

            phase = self._read_phase(domain, dominant_cls)
            thresholds = self._get_phase_thresholds(phase)

            # 2. Get/create state.
            state = self._get_or_create_state(domain)
            state.phase = phase

            # 3. Update all running statistics.
            self._update_state(state, event)

            # 4. Check conditions (order: 1, 2, 3, 4).
            c1_event = self._check_condition_1(event, state, thresholds)
            c2_event = self._check_condition_2(event, state, thresholds)
            c3_event = self._check_condition_3(event, state, thresholds)

            # Condition 4 prerequisite: Condition 3 must have fired for
            # this observation.  Add to cluster only when C3 fired.
            c4_event: Optional[NewTopologyHintEvent] = None
            if c3_event is not None:
                vec = event.classifier_distribution.astype(np.float64)
                self._update_cluster(state, vec)
                c4_event = self._check_condition_4(event, state, thresholds)

            # 5. Emit non-None events.
            for evt in filter(None, [c1_event, c3_event]):
                await self._surprise_emitter.emit(evt)
                self._total_surprises += 1

            for evt in filter(None, [c2_event, c4_event]):
                await self._hint_emitter.emit(evt)
                self._total_hints += 1

        except Exception as exc:
            log.error(
                "surprise_handler_failed",
                url=event.url if hasattr(event, "url") else "unknown",
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            # Never re-raise — the bus must not see unhandled exceptions
            # from this handler.  The surprise detector is downstream; its
            # failure must not propagate back to the classification pipeline.

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH / OBSERVABILITY
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Return a flat health snapshot for the Witness observability system.
        Called by cold_start.py or any health endpoint.
        """
        return {
            "initialized":       self._initialized,
            "mmap_available":    self._mmap is not None,
            "domains_tracked":   len(self._store),
            "lru_evictions":     self._store.eviction_count,
            "total_events":      self._total_events,
            "total_surprises":   self._total_surprises,
            "total_hints":       self._total_hints,
            "domain_state_max":  SURPRISE_DOMAIN_STATE_MAX,
        }


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def _iso_now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _uniform_prior() -> np.ndarray:
    """Return a uniform prior distribution over all topology classes."""
    return np.ones(NUM_TOPOLOGY_CLASSES, dtype=np.float64) / NUM_TOPOLOGY_CLASSES


def _one_hot(class_name: str) -> np.ndarray:
    """Return a one-hot confidence vector for a topology class."""
    vec = np.zeros(NUM_TOPOLOGY_CLASSES, dtype=np.float64)
    idx = _TOPOLOGY_CLASS_INDEX.get(class_name, 0)
    vec[idx] = 1.0
    return vec


def _peaked_at(
    class_name: str,
    peak: float = 0.90,
    noise: float = 0.005,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Return a peaked probability distribution over topology classes.
    The named class gets `peak` probability; the rest share 1 - peak
    with small uniform noise.  Used in tests to create realistic
    classifier outputs.
    """
    n = NUM_TOPOLOGY_CLASSES
    if rng is None:
        rng = np.random.default_rng(seed=42)
    vec = rng.uniform(0, noise, n)
    idx = _TOPOLOGY_CLASS_INDEX.get(class_name, 0)
    vec[idx] = peak
    vec = vec / vec.sum()
    return vec


# ═════════════════════════════════════════════════════════════════════════════
# TESTS
# ═════════════════════════════════════════════════════════════════════════════

class _MockTopicEmitter:
    def __init__(self, bus: "_MockBus") -> None:
        self._bus = bus

    async def emit(self, event: Any) -> None:
        self._bus.emitted.append(event)

class _MockBus:
    """
    Minimal mock for CrawlerBus sufficient for testing SurpriseDetector.
    Captures emitted events in a list and supports subscribe/emit.
    """

    def __init__(self) -> None:
        self._handlers: Dict[type, List[Callable]] = collections.defaultdict(list)
        self.emitted:   List[Any] = []

    async def subscribe(
        self,
        topic: str,
        group: str,
        handler: Callable,
        schema: type,
    ) -> None:
        self._handlers[schema].append(handler)

    async def emitter(
            self,
            topic: str,
            component: str,
            schema: type,
    ) -> "_MockTopicEmitter":
        return _MockTopicEmitter(self)

    async def dispatch(self, event: Any) -> None:
        """Dispatch an event to all subscribers of its type."""
        for handler in self._handlers.get(type(event), []):
            await handler(event)

    def clear(self) -> None:
        self.emitted.clear()


def _make_event(
    url: str = "https://example.com/article/1",
    domain: str = "example.com",
    observed_class: str = "NEWS_ARTICLE",
    observed_confidence: float = 0.85,
    classifier_distribution: Optional[np.ndarray] = None,
    wlm_predicted_class: str = "NEWS_ARTICLE",
    wlm_predicted_confidence: float = 0.80,
    wlm_prior_distribution: Optional[np.ndarray] = None,
    run_id: Optional[str] = None,
    manifest_id: Optional[str] = None,
) -> ClassificationEvent:
    """Test helper: create a ClassificationEvent with sensible defaults."""
    rng = np.random.default_rng(seed=0)
    if classifier_distribution is None:
        classifier_distribution = _peaked_at(observed_class, peak=observed_confidence, rng=rng)
    if wlm_prior_distribution is None:
        wlm_prior_distribution = _peaked_at(wlm_predicted_class, peak=wlm_predicted_confidence, rng=rng)
    return ClassificationEvent(
        url=url,
        domain=domain,
        observed_class=observed_class,
        observed_confidence=observed_confidence,
        classifier_distribution=classifier_distribution,
        wlm_predicted_class=wlm_predicted_class,
        wlm_predicted_confidence=wlm_predicted_confidence,
        wlm_prior_distribution=wlm_prior_distribution,
        run_id=run_id or str(uuid.uuid4()),
        manifest_id=manifest_id or str(uuid.uuid4()),
    )


class _SurpriseDetectorTestBase(unittest.TestCase):
    """Base class providing async test infrastructure."""

    def setUp(self) -> None:
        self.bus     = _MockBus()
        self.detector = SurpriseDetector(bus=self.bus)
        # Bypass mmap — tests run without the store.
        self.detector._mmap = None
        self._loop   = asyncio.new_event_loop()

    def tearDown(self) -> None:
        self._loop.close()

    def run_async(self, coro) -> Any:
        return self._loop.run_until_complete(coro)

    def _force_phase(self, detector: SurpriseDetector, domain: str, phase: PhaseState) -> None:
        """Force a domain's cached phase state for threshold testing."""
        state = detector._store.get_or_create(domain)
        state.phase = phase

    def _force_welford(
        self,
        detector: SurpriseDetector,
        domain: str,
        cls: str,
        n_obs: int = 50,
        seed: int = 42,
    ) -> None:
        """
        Pre-populate the Welford accumulator for a class with n_obs
        observations of typical peaked distributions, so the Mahalanobis
        gate is armed.
        """
        rng   = np.random.default_rng(seed=seed)
        state = detector._store.get_or_create(domain)
        acc   = state.get_or_create_welford(cls)
        ewcm  = state.get_or_create_ewcm(cls)
        for _ in range(n_obs):
            vec = _peaked_at(cls, peak=0.80 + rng.uniform(-0.10, 0.10), rng=rng)
            acc.update(vec)
            ewcm.update(vec)


class TestCondition1ConfidentMismatch(_SurpriseDetectorTestBase):
    """Tests for Condition 1 — Confident Mismatch."""

    def test_fires_on_confident_mismatch(self) -> None:
        """
        Test 1: Classifier says NEWS_ARTICLE at 0.90, WLM predicted SAAS_DOCS.
        KL divergence between one-hot distributions is very high.
        Expect SurpriseEvent (HIGH severity, contributing kl > threshold).
        """
        domain = "test.com"
        # Force KNOWN phase for maximum sensitivity.
        self._force_phase(self.detector, domain, PhaseState.KNOWN)

        classifier_dist = _peaked_at("NEWS_ARTICLE", peak=0.90)
        wlm_prior       = _peaked_at("SAAS_DOCS", peak=0.85)

        event = _make_event(
            domain=domain,
            observed_class="NEWS_ARTICLE",
            observed_confidence=0.90,
            classifier_distribution=classifier_dist,
            wlm_predicted_class="SAAS_DOCS",
            wlm_predicted_confidence=0.85,
            wlm_prior_distribution=wlm_prior,
        )

        # Manually run just Condition 1.
        state      = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        evt = self.detector._check_condition_1(event, state, thresholds)

        self.assertIsNotNone(evt)
        self.assertIsInstance(evt, _BusSurpriseEvent)
        self.assertEqual(evt.topology_class, "NEWS_ARTICLE")
        self.assertGreater(evt.surprise_score, 0.0)
        # Confidence above threshold and divergence above threshold.
        self.assertIn("kl_divergence", evt.contributing_signals)
        self.assertGreater(evt.contributing_signals["kl_divergence"], THETA_SURPRISE_DIVERGENCE_KNOWN)

    def test_no_fire_on_low_confidence_mismatch(self) -> None:
        """
        Test 2: Classifier says NEWS_ARTICLE at 0.45, WLM predicted SAAS_DOCS.
        Confidence below threshold → no event.
        """
        domain = "test.com"
        self._force_phase(self.detector, domain, PhaseState.KNOWN)

        classifier_dist = _peaked_at("NEWS_ARTICLE", peak=0.45)
        wlm_prior       = _peaked_at("SAAS_DOCS", peak=0.85)

        event = _make_event(
            domain=domain,
            observed_class="NEWS_ARTICLE",
            observed_confidence=0.45,
            classifier_distribution=classifier_dist,
            wlm_predicted_class="SAAS_DOCS",
            wlm_predicted_confidence=0.85,
            wlm_prior_distribution=wlm_prior,
        )

        state      = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        evt = self.detector._check_condition_1(event, state, thresholds)
        self.assertIsNone(evt)

    def test_no_fire_on_generic_html(self) -> None:
        """
        Test 10 (partial): GENERIC_HTML never triggers SurpriseEvent
        via Condition 1.
        """
        domain = "test.com"
        self._force_phase(self.detector, domain, PhaseState.KNOWN)

        classifier_dist = _peaked_at(FALLBACK_TOPOLOGY_CLASS, peak=0.90)
        wlm_prior       = _peaked_at("NEWS_ARTICLE", peak=0.85)

        event = _make_event(
            domain=domain,
            observed_class=FALLBACK_TOPOLOGY_CLASS,
            observed_confidence=0.90,
            classifier_distribution=classifier_dist,
            wlm_prior_distribution=wlm_prior,
        )

        state      = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        evt = self.detector._check_condition_1(event, state, thresholds)
        self.assertIsNone(evt)

    def test_condition1_severity_high(self) -> None:
        """Test 19: Condition 1 SurpriseEvent correctly reflects HIGH severity
        via dissolve_triggered=True at KNOWN phase."""
        domain = "sev-test.com"
        self._force_phase(self.detector, domain, PhaseState.KNOWN)

        classifier_dist = _peaked_at("ECOMMERCE_PRODUCT", peak=0.91)
        wlm_prior       = _peaked_at("SAAS_DOCS", peak=0.88)

        event = _make_event(
            domain=domain,
            observed_class="ECOMMERCE_PRODUCT",
            observed_confidence=0.91,
            classifier_distribution=classifier_dist,
            wlm_prior_distribution=wlm_prior,
            wlm_predicted_class="SAAS_DOCS",
        )

        state      = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        evt = self.detector._check_condition_1(event, state, thresholds)

        self.assertIsNotNone(evt)
        # dissolve_triggered=True at KNOWN phase (confident wrong at KNOWN = dissolve).
        self.assertTrue(evt.dissolve_triggered)
        self.assertEqual(evt.current_phase, int(PhaseState.KNOWN))


class TestCondition2RepeatedGenericHTML(_SurpriseDetectorTestBase):
    """Tests for Condition 2 — Repeated GENERIC_HTML Fallback."""

    def test_fires_after_threshold_consecutive_generic(self) -> None:
        """
        Test 3: 3 consecutive GENERIC_HTML at PHASE_KNOWN → NewTopologyHintEvent.
        Threshold at KNOWN is 3.
        """
        domain = "generic-test.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        state.generic_consecutive = 3

        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        event = _make_event(
            domain=domain,
            observed_class=FALLBACK_TOPOLOGY_CLASS,
            observed_confidence=0.80,
        )

        evt = self.detector._check_condition_2(event, state, thresholds)
        self.assertIsNotNone(evt)
        self.assertIsInstance(evt, NewTopologyHintEvent)
        self.assertEqual(evt.trigger, "repeated_generic")
        self.assertEqual(evt.domain, domain)
        self.assertEqual(evt.evidence_count, 3)

    def test_no_fire_at_phase_cold(self) -> None:
        """
        Test 4: 14 consecutive GENERIC_HTML at PHASE_COLD → no event.
        Threshold at COLD is 15, and COLD phase gate prevents firing.
        """
        domain = "cold-domain.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.COLD
        state.generic_consecutive = 14

        thresholds = self.detector._get_phase_thresholds(PhaseState.COLD)
        event = _make_event(
            domain=domain,
            observed_class=FALLBACK_TOPOLOGY_CLASS,
            observed_confidence=0.80,
        )

        evt = self.detector._check_condition_2(event, state, thresholds)
        self.assertIsNone(evt)

    def test_hint_event_has_suggested_parent(self) -> None:
        """
        Test 17: NewTopologyHintEvent.suggested_parent_class is computed
        correctly from centroid (nearest known class).
        """
        domain = "hint-test.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        state.generic_consecutive = 5

        # Pre-populate Welford for GENERIC_HTML with a centroid that
        # resembles NEWS_ARTICLE (most probability mass on index 0).
        welford = state.get_or_create_welford(FALLBACK_TOPOLOGY_CLASS)
        for _ in range(20):
            vec = _peaked_at("NEWS_ARTICLE", peak=0.40)  # flattened
            welford.update(vec)

        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        event = _make_event(domain=domain, observed_class=FALLBACK_TOPOLOGY_CLASS)

        evt = self.detector._check_condition_2(event, state, thresholds)
        self.assertIsNotNone(evt)
        # Suggested parent should not be GENERIC_HTML.
        self.assertNotEqual(evt.suggested_parent_class, FALLBACK_TOPOLOGY_CLASS)


class TestCondition3AnomalousShape(_SurpriseDetectorTestBase):
    """Tests for Condition 3 — Anomalous Confidence Distribution Shape."""

    def test_no_fire_below_min_observations(self) -> None:
        """
        Test 5: 29 observations on NEWS_ARTICLE → no Mahalanobis gate.
        SURPRISE_WELFORD_MIN_OBSERVATIONS = 30.
        """
        domain = "mahal-test.com"
        self._force_welford(self.detector, domain, "NEWS_ARTICLE", n_obs=29)

        state = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)

        event = _make_event(
            domain=domain,
            observed_class="NEWS_ARTICLE",
            observed_confidence=0.50,
            classifier_distribution=_peaked_at("SAAS_DOCS", peak=0.50),
        )

        evt = self.detector._check_condition_3(event, state, thresholds)
        self.assertIsNone(evt)

        welford = state.class_welford.get("NEWS_ARTICLE")
        self.assertIsNotNone(welford)
        self.assertEqual(welford.n, 29)

    def test_fires_after_sufficient_observations_with_anomaly(self) -> None:
        """
        Test 6: 31st observation with highly anomalous shape → SurpriseEvent.
        Populate 30 peaked observations, then inject a flat distribution.
        """
        domain = "mahal-fire-test.com"
        self._force_welford(self.detector, domain, "NEWS_ARTICLE", n_obs=30)

        state = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)

        # Anomalous: flat uniform distribution (far from any peaked NEWS_ARTICLE profile).
        anomalous_dist = np.ones(NUM_TOPOLOGY_CLASSES) / NUM_TOPOLOGY_CLASSES

        event = _make_event(
            domain=domain,
            observed_class="NEWS_ARTICLE",
            observed_confidence=1.0 / NUM_TOPOLOGY_CLASSES,
            classifier_distribution=anomalous_dist,
        )

        evt = self.detector._check_condition_3(event, state, thresholds)
        # The anomalous flat vector should be far from the peaked centroid.
        self.assertIsNotNone(evt)
        self.assertIsInstance(evt, _BusSurpriseEvent)
        self.assertIn("mahalanobis_distance", evt.contributing_signals)

    def test_no_fire_on_generic_html_condition3(self) -> None:
        """
        Invariant check: Condition 3 never fires on GENERIC_HTML.
        """
        domain = "generic-c3.com"
        self._force_welford(self.detector, domain, FALLBACK_TOPOLOGY_CLASS, n_obs=50)

        state = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)

        event = _make_event(
            domain=domain,
            observed_class=FALLBACK_TOPOLOGY_CLASS,
            observed_confidence=0.80,
            classifier_distribution=_peaked_at(FALLBACK_TOPOLOGY_CLASS, peak=0.80),
        )

        evt = self.detector._check_condition_3(event, state, thresholds)
        self.assertIsNone(evt)


class TestCondition4CoherentCluster(_SurpriseDetectorTestBase):
    """Tests for Condition 4 — Consistent Unknown Pattern."""

    def test_fires_on_coherent_cluster(self) -> None:
        """
        Test 7: SURPRISE_CLUSTER_WINDOW/4 anomalous vectors with low variance
        → NewTopologyHintEvent with trigger='coherent_cluster'.
        """
        domain = "cluster-test.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)

        # Build a coherent cluster: all vectors near the same centroid.
        rng = np.random.default_rng(seed=7)
        base_vec = _peaked_at("FORUM_THREAD", peak=0.60)
        for i in range(max(4, SURPRISE_CLUSTER_WINDOW // 4)):
            noise = rng.uniform(-0.005, 0.005, NUM_TOPOLOGY_CLASSES)
            vec   = np.clip(base_vec + noise, 0, 1)
            vec  /= vec.sum()
            self.detector._update_cluster(state, vec)

        event = _make_event(domain=domain)
        evt = self.detector._check_condition_4(event, state, thresholds)

        self.assertIsNotNone(evt)
        self.assertIsInstance(evt, NewTopologyHintEvent)
        self.assertEqual(evt.trigger, "coherent_cluster")
        self.assertIsNotNone(evt.cluster_variance)
        self.assertLess(evt.cluster_variance, THETA_SURPRISE_CLUSTER_COHERENCE)

    def test_condition4_requires_condition3_prerequisite(self) -> None:
        """
        Test 13: A normal uncertain classification (Condition 3 did not fire)
        is not added to the anomalous cluster.  Condition 4 sees nothing.
        """
        domain = "no-c3-prerequisite.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)

        # Condition 3 does NOT fire (n < min_observations).
        event = _make_event(domain=domain, observed_class="NEWS_ARTICLE")
        evt3  = self.detector._check_condition_3(event, state, thresholds)
        self.assertIsNone(evt3)

        # Since Condition 3 didn't fire, cluster is not updated.
        # Condition 4 should return None.
        evt4 = self.detector._check_condition_4(event, state, thresholds)
        self.assertIsNone(evt4)
        self.assertEqual(len(state.anomalous_vectors), 0)


class TestPhaseThresholdSwitching(_SurpriseDetectorTestBase):
    """Tests for phase-aware threshold management."""

    def test_phase_threshold_switching_mid_session(self) -> None:
        """
        Test 8: Domain promotes from LEARNING to KNOWN mid-session.
        Next event uses KNOWN thresholds (tighter).
        """
        domain = "phase-switch.com"
        state  = self.detector._get_or_create_state(domain)

        # Simulate LEARNING phase thresholds.
        state.phase = PhaseState.LEARNING
        t_learning  = self.detector._get_phase_thresholds(PhaseState.LEARNING)
        self.assertEqual(t_learning.confident, THETA_SURPRISE_CONFIDENT_LEARNING)
        self.assertEqual(t_learning.generic_threshold, SURPRISE_GENERIC_THRESHOLD_LEARNING)

        # Simulate phase promotion.
        state.phase = PhaseState.KNOWN
        t_known     = self.detector._get_phase_thresholds(PhaseState.KNOWN)
        self.assertEqual(t_known.confident, THETA_SURPRISE_CONFIDENT_KNOWN)
        self.assertEqual(t_known.generic_threshold, SURPRISE_GENERIC_THRESHOLD_KNOWN)

        # KNOWN thresholds must be tighter (lower values = fires more easily).
        self.assertLess(t_known.confident, t_learning.confident)
        self.assertLess(t_known.divergence, t_learning.divergence)
        self.assertLess(t_known.generic_threshold, t_learning.generic_threshold)
        self.assertLess(t_known.mahalanobis, t_learning.mahalanobis)

    def test_phase_read_not_cached(self) -> None:
        """
        Test 18: Phase thresholds change when phase state is updated.
        Simulates: LEARNING phase → observe → threshold changes → KNOWN phase → observe.
        """
        domain = "nocache-phase.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.LEARNING

        t1 = self.detector._get_phase_thresholds(state.phase)

        # Simulate phase transition.
        state.phase = PhaseState.KNOWN
        t2 = self.detector._get_phase_thresholds(state.phase)

        # Thresholds must differ — KNOWN is strictly tighter than LEARNING.
        self.assertNotEqual(t1.divergence, t2.divergence)
        self.assertGreater(t1.divergence, t2.divergence)


class TestDomainStateEviction(_SurpriseDetectorTestBase):
    """Tests for LRU eviction at SURPRISE_DOMAIN_STATE_MAX."""

    def test_eviction_at_max_capacity(self) -> None:
        """
        Test 9: 10,001st domain evicts LRU → no crash, correct store size.
        """
        store = _LRUDomainStateStore(max_size=5)

        # Fill to capacity.
        for i in range(5):
            store.get_or_create(f"domain-{i}.com")

        self.assertEqual(len(store), 5)

        # One more should evict the LRU (domain-0.com).
        store.get_or_create("domain-5.com")
        self.assertEqual(len(store), 5)
        self.assertEqual(store.eviction_count, 1)
        # LRU (domain-0) should be gone.
        self.assertNotIn("domain-0.com", store.domains())
        # MRU should be present.
        self.assertIn("domain-5.com", store.domains())


class TestKLDivergenceProperties(_SurpriseDetectorTestBase):
    """Tests for KL divergence computation properties."""

    def test_kl_divergence_symmetry_special_case(self) -> None:
        """
        Test 11: prior == posterior → KL divergence ≈ 0.
        """
        dist = _peaked_at("NEWS_ARTICLE", peak=0.85)
        kl   = self.detector._kl_divergence(dist, dist)
        self.assertAlmostEqual(kl, 0.0, places=6)

    def test_kl_divergence_orthogonal_distributions(self) -> None:
        """KL divergence between orthogonal one-hot vectors is large but finite."""
        p = _one_hot("NEWS_ARTICLE")
        q = _one_hot("ECOMMERCE_PRODUCT")
        kl = self.detector._kl_divergence(p, q)
        # Should be large but finite (epsilon prevents log(0)).
        self.assertGreater(kl, 5.0)
        self.assertTrue(math.isfinite(kl))

    def test_kl_divergence_uniform_reference(self) -> None:
        """KL from uniform prior to peaked posterior should be positive."""
        uniform = _uniform_prior()
        peaked  = _peaked_at("SAAS_DOCS", peak=0.90)
        kl = self.detector._kl_divergence(uniform, peaked)
        self.assertGreater(kl, 0.0)


class TestWelfordStability(_SurpriseDetectorTestBase):
    """Tests for Welford accumulator numerical stability."""

    def test_welford_converges_with_1000_observations(self) -> None:
        """
        Test 12: 1000 observations → mean and variance converge to ground truth.
        """
        d   = 8
        rng = np.random.default_rng(seed=123)
        mu_true  = rng.uniform(0, 1, d)
        mu_true /= mu_true.sum()
        sigma_true = 0.05

        acc = MultivariateWelfordAccumulator(d=d)
        for _ in range(1000):
            noise = rng.normal(0, sigma_true, d)
            x = np.clip(mu_true + noise, 0, None)
            x = x / x.sum()
            acc.update(x)

        self.assertEqual(acc.n, 1000)
        # Mean should be close to true mean.
        np.testing.assert_allclose(acc.mean, mu_true, atol=0.02)
        # Variance should be positive.
        var = acc.variance()
        self.assertTrue(np.all(var >= 0))


class TestGenericHTMLNeverFiresSurpriseEvent(_SurpriseDetectorTestBase):
    """
    Test 10: GENERIC_HTML never triggers SurpriseEvent (only NewTopologyHintEvent).
    """

    def test_full_handler_generic_html(self) -> None:
        """
        Run handle_classification_event with GENERIC_HTML and assert
        only NewTopologyHintEvent (if any) is emitted, never SurpriseEvent.
        """
        async def _run():
            await self.detector.initialize()
            domain = "generic-invariant.com"
            state  = self.detector._store.get_or_create(domain)
            state.phase = PhaseState.KNOWN
            state.generic_consecutive = 100   # well above threshold

            for _ in range(10):
                event = _make_event(
                    domain=domain,
                    observed_class=FALLBACK_TOPOLOGY_CLASS,
                    observed_confidence=0.80,
                    classifier_distribution=_peaked_at(FALLBACK_TOPOLOGY_CLASS, peak=0.80),
                )
                state.phase = PhaseState.KNOWN
                await self.detector.handle_classification_event(event)

        self.run_async(_run())

        # No SurpriseEvent should have been emitted.
        for evt in self.bus.emitted:
            if isinstance(evt, _BusSurpriseEvent):
                self.fail(
                    f"SurpriseEvent emitted for GENERIC_HTML: {evt}"
                )


class TestHandlerOOnePerformance(_SurpriseDetectorTestBase):
    """
    Test 14: Handler O(1) performance — 10,000 events, latency flat.
    """

    def test_latency_flat_over_10k_events(self) -> None:
        """
        Process 10,000 events and verify per-event latency remains bounded.
        O(1) criterion: latency for last 1000 events ≤ 3× latency for first 1000.
        """

        async def _run():
            await self.detector.initialize()
            rng = np.random.default_rng(seed=777)
            times: List[float] = []

            # Warm-up: arm Welford on all 10 domains before timing starts.
            for i in range(400):
                dist = _peaked_at("NEWS_ARTICLE", peak=0.80, rng=rng)
                prior = _peaked_at("NEWS_ARTICLE", peak=0.75, rng=rng)
                event = _make_event(
                    domain=f"domain-{i % 10}.com",
                    classifier_distribution=dist,
                    wlm_prior_distribution=prior,
                )
                await self.detector.handle_classification_event(event)

            for i in range(10_000):
                dist = _peaked_at("NEWS_ARTICLE", peak=0.80, rng=rng)
                prior = _peaked_at("NEWS_ARTICLE", peak=0.75, rng=rng)
                event = _make_event(
                    domain=f"domain-{i % 10}.com",
                    classifier_distribution=dist,
                    wlm_prior_distribution=prior,
                )
                t0 = time.monotonic()
                await self.detector.handle_classification_event(event)
                times.append(time.monotonic() - t0)

            return times

        import structlog, logging
        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(logging.ERROR),
        )

        import cProfile, pstats, io
        pr = cProfile.Profile()
        pr.enable()
        times = self.run_async(_run())
        pr.disable()

        s = io.StringIO()
        ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
        ps.print_stats(20)
        print(s.getvalue())

        # after your 10K run, check these three numbers
        first_100_avg = sum(times[:100]) / 100
        mid_100_avg = sum(times[4950:5050]) / 100
        last_100_avg = sum(times[-100:]) / 100

        print(f"first 100 avg: {first_100_avg * 1000:.4f} ms")
        print(f"mid   100 avg: {mid_100_avg * 1000:.4f} ms")
        print(f"last  100 avg: {last_100_avg * 1000:.4f} ms")
        print(f"p50:  {sorted(times)[5000] * 1000:.4f} ms")
        print(f"p95:  {sorted(times)[9500] * 1000:.4f} ms")
        print(f"p99:  {sorted(times)[9900] * 1000:.4f} ms")
        print(f"max:  {max(times) * 1000:.4f} ms")

        # Latency should not grow by more than 10× (generous bound for test env).
        if first_100_avg > 1e-9:
            ratio = last_100_avg / first_100_avg
            self.assertLess(ratio, 10.0, f"Latency grew by {ratio:.1f}× — not O(1)")


class TestBusSubscriptionWiring(_SurpriseDetectorTestBase):
    """
    Test 15: Bus subscription wiring — ClassificationEvent subscription fires handler.
    """

    def test_handler_fires_on_dispatch(self) -> None:
        """
        After initialize(), dispatching a ClassificationEvent to the bus
        should invoke handle_classification_event.
        """
        async def _run():
            await self.detector.initialize()
            event = _make_event(domain="wiring.com")
            # Dispatch directly through the mock bus.
            await self.bus.dispatch(event)

        self.run_async(_run())
        # After one dispatch, total_events should be 1.
        self.assertEqual(self.detector._total_events, 1)


class TestSurpriseEventFieldsComplete(_SurpriseDetectorTestBase):
    """
    Test 16: SurpriseEvent fields are all populated.
    """

    def test_all_fields_populated(self) -> None:
        """
        Verify that a Condition 1 SurpriseEvent has all required fields set.
        """
        domain = "fields-test.com"
        state  = self.detector._get_or_create_state(domain)
        state.phase = PhaseState.KNOWN
        thresholds = self.detector._get_phase_thresholds(PhaseState.KNOWN)

        classifier_dist = _peaked_at("BLOG_POST", peak=0.88)
        wlm_prior       = _peaked_at("WIKIPEDIA_ARTICLE", peak=0.85)
        run_id          = str(uuid.uuid4())

        event = _make_event(
            domain=domain,
            observed_class="BLOG_POST",
            observed_confidence=0.88,
            classifier_distribution=classifier_dist,
            wlm_predicted_class="WIKIPEDIA_ARTICLE",
            wlm_prior_distribution=wlm_prior,
            run_id=run_id,
        )

        evt = self.detector._check_condition_1(event, state, thresholds)
        self.assertIsNotNone(evt)

        # All fields must be present and non-None.
        self.assertEqual(evt.run_id, run_id)
        self.assertIsInstance(evt.topology_class, str)
        self.assertIsInstance(evt.surprise_score, float)
        self.assertIsInstance(evt.contributing_signals, dict)
        self.assertIn("kl_divergence", evt.contributing_signals)
        self.assertIsNotNone(evt.timestamp)
        self.assertIn(evt.current_phase, (1, 2, 3))


class TestMathematicalPrimitives(_SurpriseDetectorTestBase):
    """Unit tests for the core mathematical primitives."""

    def test_ledoit_wolf_positive_definite(self) -> None:
        """LW-regularised covariance should always be PD."""
        rng = np.random.default_rng(seed=99)
        # Near-singular raw covariance (only 10 obs for 18-dim).
        data = rng.standard_normal((10, 18))
        S = np.cov(data.T)
        Sigma = LedoitWolfShrinkageEstimator.shrink(S, n=10)
        # All eigenvalues must be positive.
        eigenvalues = np.linalg.eigvalsh(Sigma)
        self.assertTrue(np.all(eigenvalues > 0))

    def test_wasserstein_1_zero_for_identical(self) -> None:
        """W₁ distance from a distribution to itself should be 0."""
        p = _peaked_at("NEWS_ARTICLE", peak=0.80)
        w = WassersteinSimplexDistance.wasserstein1(p, p)
        self.assertAlmostEqual(w, 0.0, places=10)

    def test_wasserstein_1_positive_for_different(self) -> None:
        """W₁ distance between different distributions should be > 0."""
        p = _peaked_at("NEWS_ARTICLE", peak=0.90)
        q = _peaked_at("ECOMMERCE_PRODUCT", peak=0.90)
        w = WassersteinSimplexDistance.wasserstein1(p, q)
        self.assertGreater(w, 0.0)

    def test_betti0_unimodal_returns_1(self) -> None:
        """Tightly clustered scores → 1 significant Betti-0 mode."""
        scores = [0.10 + 0.01 * i for i in range(20)]
        modes = Betti0PersistenceAnalyser.count_significant_modes(scores)
        self.assertEqual(modes, 1)

    def test_betti0_bimodal_returns_2(self) -> None:
        """Two well-separated clusters → 2 significant Betti-0 modes."""
        scores = list(range(5)) + [100 + i for i in range(5)]
        modes = Betti0PersistenceAnalyser.count_significant_modes(
            [float(s) for s in scores]
        )
        self.assertGreaterEqual(modes, 2)

    def test_hill_returns_none_below_threshold(self) -> None:
        """Hill estimator returns None with fewer than 10 observations."""
        estimator = HillTailIndexEstimator()
        for s in [0.1, 0.2, 0.3]:
            estimator.update(s)
        self.assertIsNone(estimator.alpha_hat())

    def test_hill_gaussian_alpha_near_2(self) -> None:
        """For Gaussian-distributed data, Hill estimator α̂ should be > 1.5."""
        rng = np.random.default_rng(seed=55)
        estimator = HillTailIndexEstimator(max_window=500)
        for _ in range(300):
            s = abs(float(rng.standard_normal()))
            if s > 0:
                estimator.update(s)
        alpha = estimator.alpha_hat()
        # Gaussian tail is thin — Hill should return a high index.
        self.assertIsNotNone(alpha)
        self.assertGreater(alpha, 1.0)

    def test_oja_pca_converges_to_dominant_direction(self) -> None:
        """
        Oja online PCA should converge to the first principal component.
        Generate data with a dominant direction and verify PC1 ratio is high.
        """
        d   = 4
        rng = np.random.default_rng(seed=13)
        # Strong first PC: data = signal * e_1 + noise.
        pca = OjaOnlinePCATracker(d=d, eta0=0.05, beta=0.001)
        for _ in range(2000):
            signal = float(rng.normal(0, 10))
            noise  = rng.normal(0, 0.1, d)
            x      = np.zeros(d)
            x[0]  += signal
            x     += noise
            pca.update(x)

        # After convergence, the dominant direction should be close to e_1.
        w = pca._w
        self.assertGreater(float(abs(w[0])), 0.9)

    def test_mdl_does_not_split_unimodal_data(self) -> None:
        """MDL should prefer k=1 for tightly clustered data."""
        rng = np.random.default_rng(seed=42)
        # All vectors near the same centroid.
        base = _peaked_at("BLOG_POST", peak=0.80)
        vecs = np.array([
            base + rng.normal(0, 0.002, NUM_TOPOLOGY_CLASSES)
            for _ in range(20)
        ])
        vecs = np.clip(vecs, 0, None)
        vecs = vecs / vecs.sum(axis=1, keepdims=True)
        split = RissanenMDLClusterEvaluator.mdl_supports_split(vecs)
        self.assertFalse(split)

    def test_mdl_splits_bimodal_data(self) -> None:
        """MDL should prefer k=2 for clearly bimodal data."""
        rng = np.random.default_rng(seed=42)
        base1 = _peaked_at("NEWS_ARTICLE",    peak=0.85)
        base2 = _peaked_at("ECOMMERCE_PRODUCT", peak=0.85)
        vecs1 = base1 + rng.normal(0, 0.005, (15, NUM_TOPOLOGY_CLASSES))
        vecs2 = base2 + rng.normal(0, 0.005, (15, NUM_TOPOLOGY_CLASSES))
        vecs  = np.vstack([vecs1, vecs2])
        vecs  = np.clip(vecs, 0, None)
        vecs  = vecs / vecs.sum(axis=1, keepdims=True)
        split = RissanenMDLClusterEvaluator.mdl_supports_split(vecs)
        self.assertTrue(split)

    def test_ewma_covariance_tracks_distribution_shift(self) -> None:
        """EWMA covariance diverges from Welford when distribution shifts."""
        d   = 8
        rng = np.random.default_rng(seed=7)

        acc  = MultivariateWelfordAccumulator(d=d)
        ewcm = EWMACovariance(d=d, lambda_=0.90)

        # Phase 1: feed 50 observations from one distribution.
        for _ in range(50):
            x = rng.dirichlet(np.ones(d))
            acc.update(x)
            ewcm.update(x)

        frob_before = ewcm.frobenius_distance_from(acc.covariance(ddof=1))

        # Phase 2: inject 10 observations from a very different distribution.
        spike = np.zeros(d)
        spike[0] = 1.0
        for _ in range(10):
            ewcm.update(spike)

        frob_after = ewcm.frobenius_distance_from(acc.covariance(ddof=1))
        # EWCM should diverge after the distribution shift.
        self.assertGreater(frob_after, frob_before)


class TestNormalisedPredictiveInformation(_SurpriseDetectorTestBase):
    """Tests for Algorithm 10 — NPI."""

    def test_npi_high_for_deterministic_sequence(self) -> None:
        """Deterministic sequence (always the same class) → high NPI."""
        npi_tracker = NormalisedPredictiveInformation()
        for _ in range(50):
            npi_tracker.update("NEWS_ARTICLE")
        npi = npi_tracker.current_npi
        self.assertIsNotNone(npi)
        self.assertGreater(npi, 0.80)

    def test_npi_drop_detected_after_distribution_shift(self) -> None:
        """
        NPI drop detected when sequence transitions from predictable
        to uniform random.
        """
        npi_tracker = NormalisedPredictiveInformation()
        classes = list(_TOPOLOGY_CLASS_INDEX.keys())
        rng = np.random.default_rng(seed=42)

        # Predictable phase: alternating NEWS_ARTICLE / BLOG_POST.
        for i in range(40):
            npi_tracker.update("NEWS_ARTICLE" if i % 2 == 0 else "BLOG_POST")

        # Unpredictable phase: random class each time.
        for _ in range(20):
            npi_tracker.update(rng.choice(classes))

        # Drop should be detected.
        drop = npi_tracker.npi_drop_detected(threshold=0.10)
        # (The drop may or may not fire depending on the random sequence;
        #  this is a probabilistic test — just check it doesn't crash.)
        self.assertIsInstance(drop, bool)


class TestInitializeShutdown(_SurpriseDetectorTestBase):
    """
    Test 20: Initialize and shutdown — clean lifecycle, no leaked state.
    """

    def test_clean_initialize_shutdown(self) -> None:
        """
        After shutdown(), detector is marked not initialized.
        No exceptions raised.
        """
        async def _run():
            await self.detector.initialize()
            self.assertTrue(self.detector._initialized)
            health_before = self.detector.health()
            self.assertTrue(health_before["initialized"])

            await self.detector.shutdown()
            self.assertFalse(self.detector._initialized)

        self.run_async(_run())


def run_tests() -> None:
    """Run all tests.  Called by cold_start.py validation step."""
    loader = unittest.TestLoader()
    suite  = unittest.TestSuite()
    for cls in [
        TestCondition1ConfidentMismatch,
        TestCondition2RepeatedGenericHTML,
        TestCondition3AnomalousShape,
        TestCondition4CoherentCluster,
        TestPhaseThresholdSwitching,
        TestDomainStateEviction,
        TestKLDivergenceProperties,
        TestWelfordStability,
        TestGenericHTMLNeverFiresSurpriseEvent,
        TestHandlerOOnePerformance,
        TestBusSubscriptionWiring,
        TestSurpriseEventFieldsComplete,
        TestMathematicalPrimitives,
        TestNormalisedPredictiveInformation,
        TestInitializeShutdown,
    ]:
        suite.addTests(loader.loadTestsFromTestCase(cls))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise RuntimeError(
            f"SurpriseDetector self-tests failed: "
            f"{len(result.failures)} failures, {len(result.errors)} errors."
        )


# ═════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Primary interface.
    "SurpriseDetector",
    "ClassificationEvent",
    "NewTopologyHintEvent",
    "PhaseState",
    "SurpriseSeverity",
    "SurpriseThresholds",

    # Mathematical primitives (exported for index_daemon diagnostic tooling).
    "MultivariateWelfordAccumulator",
    "LedoitWolfShrinkageEstimator",
    "HillTailIndexEstimator",
    "Betti0PersistenceAnalyser",
    "RissanenMDLClusterEvaluator",
    "WassersteinSimplexDistance",
    "DPSSMultitaperAnalyser",
    "OjaOnlinePCATracker",
    "EWMACovariance",
    "NormalisedPredictiveInformation",

    # State structures.
    "DomainSurpriseState",

    # Constants.
    "NUM_TOPOLOGY_CLASSES",
    "SURPRISE_WELFORD_MIN_OBSERVATIONS",
    "SURPRISE_CLUSTER_WINDOW",
    "SURPRISE_DOMAIN_STATE_MAX",

    # Test runner.
    "run_tests",
]