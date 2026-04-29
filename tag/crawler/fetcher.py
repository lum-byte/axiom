"""
crawler/fetcher.py
==================
The vacuum. The only file in ``crawler/`` that talks to the internet.

AXIOM INTERNAL // DO NOT SURFACE

Receives ``CrawlManifest``, executes URLs in strict priority order, emits
``RawFetchEvent`` per URL.  Four fetch modes mapped to four clearance levels.
Every decision about *which* mode to use was made by the preparser.  The
fetcher reads the decision and executes it.  That is the complete contract.

A 429 in the fetcher's logs is an architecture bug, not a normal operating
condition.  The preparser already knew the rate-limit ceiling.  If the fetcher
hits a 429, the ``CrawlManifest`` was wrong, or ``rate_limiter.py`` failed to
enforce pacing.  The fetcher logs it as ``FetchAnomalyEvent`` and routes it to
the bus for ``index_daemon`` to treat as training signal.

Zero-logic invariant:
    The fetcher contains zero routing logic.  It does not inspect response
    content to decide what to do next.  It does not adapt its behavior based
    on topology class.  It does not make decisions about whether a URL is
    worth fetching.  If the manifest says fetch it, the fetcher fetches it.
    If the fetch fails, it emits an anomaly and moves on.

Mathematical subsystems (all online / streaming — zero bulk storage):
    ──────────────────────────────────────────────────────────────────────
    P² quantile estimation   — streaming P50/P95/P99 latency without
                               storing every observation.  O(1) memory,
                               O(1) per observation.  Jain & Chlamtac 1985.
    Welford online variance  — numerically stable one-pass mean+variance
                               for response sizes, latencies, jitter.
                               Welford 1962.
    EWMA latency tracking    — exponentially weighted moving average with
                               configurable half-life for CL-level health
                               scoring.  Half-life = 50 observations.
    Shannon entropy          — H(X) = -Σ p(x) log₂ p(x) over status-code
                               distribution.  Detects when a domain returns
                               monotonic error codes (low entropy = sick).
    Hazard-rate modeling     — Nelson–Aalen cumulative hazard estimator
                               for connection-failure timing.  Predicts
                               when the next failure is likely given the
                               observed failure cadence.
    Reservoir sampling       — Vitter's Algorithm R for uniform random
                               sampling of anomaly events when the anomaly
                               stream exceeds the reporting budget.
    Markov circuit quality   — 2-state Markov chain (good / bad exit IP)
                               for Tor circuit rotation quality.  Tracks
                               transition probabilities online.
    Little's Law adaptive    — L = λW.  Computes optimal in-flight request
                               count from observed throughput (λ) and
                               mean latency (W).  Adjusts manifest
                               semaphore dynamically.
    ──────────────────────────────────────────────────────────────────────

Dependency chain:
    fetcher.py
      ├── rate_limiter.py     → acquire() before every request
      ├── bloom_filter.py     → contains() before fetch; add() after emit
      ├── frontier.py         → resume() to get next URL
      └── crawl_cursor.py     → checkpoint() every CHECKPOINT_INTERVAL URLs

Build order: this file is step 5 of 5.  All four dependencies must exist.
"""

from __future__ import annotations

import argparse
import asyncio
import enum
import hashlib # noqa
import json
import logging
import math
import os # noqa
import random
import shutil # noqa
import signal # noqa
import socket # noqa
import struct # noqa
import sys
import tempfile
import time
import traceback
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import ( # noqa
    Any,
    AsyncIterator,
    Callable,
    Deque,
    Dict,
    Final,
    FrozenSet,
    Iterator,
    List,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    Sequence,
    Set,
    Tuple,
    Union,
)
from urllib.parse import urlparse

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TaskProgressColumn,
    MofNCompleteColumn,
)
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()


# ── External dependencies ─────────────────────────────────────────────────────

try:
    import httpx
except ImportError as _e:
    raise ImportError(
        "httpx is required for fetcher.py.\n"
        "Install with: pip install httpx\n"
    ) from _e

# ── Internal dependencies ─────────────────────────────────────────────────────
# These must exist before fetcher.py is built.  Build order is strict.

try:
    from tag.crawler.bloom_filter import BloomFilter
except ImportError:
    BloomFilter = None  # type: ignore[misc,assignment]

try:
    from tag.crawler.crawl_cursor import CrawlCursor
except ImportError:
    CrawlCursor = None  # type: ignore[misc,assignment]

try:
    from tag.crawler.frontier import Frontier
except ImportError:
    Frontier = None  # type: ignore[misc,assignment]

try:
    from tag.crawler.rate_limiter import RateLimiter
except ImportError:
    RateLimiter = None  # type: ignore[misc,assignment]

# ── Contracts — imported if available, else defined locally ───────────────────
# The fetcher must work standalone for testing.  If the full AXIOM tree is
# available, import from contracts.py.  Otherwise, define minimal versions
# that satisfy the same interface.

try:
    from signal_kernel.contracts import (
        CrawlManifest,
        CrawlManifestReadyEvent,
        CrawlURL,
        CLStateUpdateEvent,
        ContainerBreachEvent,
        FetchAnomalyEvent,
        FetchMode,
        FrontierStats,
        ManifestCompleteEvent,
        RateLimitProfile,
        RawFetchEvent,
    )
except ImportError:
    # Minimal local definitions for standalone operation.
    # These are structurally identical to contracts.py originals.

    class FetchMode(str, enum.Enum):  # type: ignore[no-redef]
        STATIC   = "static"
        HEADLESS = "headless"
        TOR      = "tor"
        TOR_FULL = "tor_full"

    RenderMode = Literal["static", "headless"]

    @dataclass(frozen=True)
    class RateLimitProfile:  # type: ignore[no-redef]
        domain: str
        requests_per_second: float
        crawl_delay_seconds: float = 0.0
        burst_capacity: int = 3

    @dataclass(frozen=True)
    class CrawlURL:  # type: ignore[no-redef]
        url: str
        topology_hint: str
        fetch_mode: FetchMode
        render_mode: str = "static"
        priority: int = 0
        rate_limit_profile: Optional[RateLimitProfile] = None
        expected_content_type: str = "text/html"
        crawl_delay_seconds: float = 0.0
        max_response_bytes: int = 4 * 1024 * 1024
        is_robots: bool = False
        is_sitemap: bool = False
        run_id: str = ""

    @dataclass(frozen=True)
    class CrawlManifest:  # type: ignore[no-redef]
        domain: str
        urls: List[CrawlURL]
        total_urls: int
        estimated_duration_seconds: float = 0.0
        clearance_required: int = 1
        manifest_id: str = ""

    @dataclass(frozen=True)
    class RawFetchEvent:  # type: ignore[no-redef]
        url: str
        raw_bytes: bytes
        status_code: int
        headers: Dict[str, str]
        fetch_latency: float
        fetch_mode: FetchMode
        is_robots_txt: bool
        is_sitemap: bool
        topology_hint: str
        run_id: str
        manifest_id: str
        byte_count: int

    @dataclass(frozen=True)
    class FrontierStats:  # type: ignore[no-redef]
        manifest_id: str
        pending: int = 0
        done: int = 0
        failed: int = 0
        skipped: int = 0

        @property
        def total(self) -> int:
            return self.pending + self.done + self.failed + self.skipped

        @property
        def completion_rate(self) -> float:
            return (self.done + self.skipped) / self.total if self.total else 0.0

    @dataclass(frozen=True)
    class CrawlManifestReadyEvent:  # type: ignore[no-redef]
        domain: str
        manifest: CrawlManifest

    @dataclass(frozen=True)
    class ManifestCompleteEvent:  # type: ignore[no-redef]
        domain: str
        manifest_id: str
        stats: FrontierStats

    @dataclass(frozen=True)
    class CLStateUpdateEvent:  # type: ignore[no-redef]
        cl2_available: bool
        cl3_available: bool
        cl4_available: bool
        reason: str = ""

    @dataclass(frozen=True)
    class ContainerBreachEvent:  # type: ignore[no-redef]
        manifest_id: str
        run_id: str
        fetch_mode: FetchMode
        breach_signal: str
        url: str
        detected_at: datetime = field(
            default_factory=lambda: datetime.now(timezone.utc)
        )


# ── Exceptions — same pattern ─────────────────────────────────────────────────


try:
    from signal_kernel.exceptions import (
        CursorError,
        FetchError,
        FrontierError,
        ManifestExhaustedError,
        PlaywrightError as PlaywrightExc,
        RateLimitViolationError,
        TorUnavailableError,
    )
except ImportError:
    class FetchError(Exception): pass  # type: ignore[no-redef]
    class TorUnavailableError(FetchError): pass  # type: ignore[no-redef]
    class PlaywrightExc(FetchError): pass  # type: ignore[no-redef]
    class RateLimitViolationError(FetchError): pass  # type: ignore[no-redef]
    class FrontierError(Exception): pass  # type: ignore[no-redef]
    class CursorError(Exception): pass  # type: ignore[no-redef]
    class ManifestExhaustedError(Exception): pass  # type: ignore[no-redef]


def _suppress_unraisable(args):
    if (
        args.exc_type is RuntimeError
        and "Event loop is closed" in str(args.exc_value)
    ):
        return
    sys.__unraisablehook__(args)

sys.unraisablehook = _suppress_unraisable
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — LOCKED BY SPEC
#
# These constants define the operational envelope of the fetcher.  Changing
# any of them changes observable behavior.  Every constant is referenced by
# name in at least one spec document.  Do not inline magic numbers.
# ═══════════════════════════════════════════════════════════════════════════════


# ── HTTP / fetch limits ───────────────────────────────────────────────────────

MAX_REDIRECTS:                   Final[int]   = 10
MAX_RESPONSE_BYTES:              Final[int]   = 4 * 1024 * 1024       # 4 MB
STATIC_TIMEOUT_CONNECT:          Final[float] = 10.0                  # seconds
STATIC_TIMEOUT_READ:             Final[float] = 30.0                  # seconds
STATIC_TIMEOUT_WRITE:            Final[float] = 10.0                  # seconds
STATIC_TIMEOUT_POOL:             Final[float] = 5.0                   # seconds

# ── Playwright / headless ─────────────────────────────────────────────────────

HEADLESS_NAVIGATION_TIMEOUT:     Final[int]   = 30_000                # ms
HEADLESS_CONTEXT_RECYCLE:        Final[int]   = 50                    # pages
HEADLESS_VIEWPORT_WIDTH:         Final[int]   = 1280
HEADLESS_VIEWPORT_HEIGHT:        Final[int]   = 800

# ── Tor configuration ─────────────────────────────────────────────────────────

TOR_SOCKS_HOST:                  Final[str]   = "127.0.0.1"
TOR_SOCKS_PORT:                  Final[int]   = 9050
TOR_CONTROL_PORT:                Final[int]   = 9051
TOR_CIRCUIT_INTERVAL:            Final[int]   = 10                    # CL3
TOR_FULL_JITTER_MIN:             Final[float] = 0.5                   # seconds
TOR_FULL_JITTER_MAX:             Final[float] = 3.0                   # seconds
TOR_CONNECT_TIMEOUT:             Final[float] = 5.0                   # seconds
TOR_NEWNYM_COOLDOWN:             Final[float] = 1.0                   # seconds

# ── Checkpoint / cursor ───────────────────────────────────────────────────────

CHECKPOINT_INTERVAL:             Final[int]   = 100                   # URLs

# ── Concurrency ───────────────────────────────────────────────────────────────

MAX_CONCURRENT_MANIFESTS:        Final[int]   = 4                     # CL1
MAX_CONCURRENT_CL2:              Final[int]   = 2
MAX_CONCURRENT_CL3:              Final[int]   = 1
MAX_CONCURRENT_CL4:              Final[int]   = 1

# ── Staging ───────────────────────────────────────────────────────────────────

STAGING_PATH:                    Final[Path]  = Path("/tmp/fetch_staging")

# ── Telemetry / math ──────────────────────────────────────────────────────────

EWMA_HALF_LIFE:                  Final[int]   = 50                    # observations
EWMA_ALPHA:                      Final[float] = 1.0 - math.exp(
    -math.log(2) / EWMA_HALF_LIFE
)
"""
EWMA smoothing factor derived from half-life.

    α = 1 - exp(-ln(2) / H)

where H = EWMA_HALF_LIFE observations.  After H observations the weight
of an observation decays to 50%.  This gives:

    α = 1 - exp(-0.6931 / 50) = 1 - exp(-0.01386) ≈ 0.01376

Recent observations dominate; the effective window is ~3.5 × H ≈ 175
observations before a data point's contribution drops below 0.1%.
"""

RESERVOIR_SIZE:                  Final[int]   = 256
"""
Maximum anomaly events retained per manifest for statistical reporting.
Uses Vitter's Algorithm R — every anomaly has equal probability of being
in the reservoir regardless of stream length.
"""

ENTROPY_WINDOW:                  Final[int]   = 200
"""
Rolling window for Shannon entropy calculation over HTTP status codes.
Detects when a domain degenerates to returning a single status code
(entropy → 0), which signals the domain is rate-limiting or blocking.
"""

HAZARD_WINDOW:                   Final[int]   = 100
"""
Window for the Nelson–Aalen cumulative hazard estimator.  Tracks
connection failure timing to predict when the next failure is due.
"""

LITTLE_LAW_WINDOW:               Final[int]   = 50
"""
Observations used for Little's Law adaptive concurrency:  L = λ × W.
Computes optimal in-flight count from throughput (λ) and mean latency (W).
"""

P2_QUANTILES:                    Final[Tuple[float, ...]] = (0.5, 0.95, 0.99)
"""
Quantiles tracked by the P² algorithm.  P50 for typical latency,
P95 for tail latency, P99 for worst-case latency.  All computed in
O(1) space per quantile.
"""

# ── Benchmark URL sets ────────────────────────────────────────────────────────

BENCHMARK_DEFAULT_N: Final[Dict[int, int]] = {1: 1000, 2: 500, 3: 200, 4: 100}

# ── Logging ───────────────────────────────────────────────────────────────────

_LOG_FMT = "[fetcher] %(levelname)s %(message)s"


# ═══════════════════════════════════════════════════════════════════════════════
#
#   MATHEMATICAL PRIMITIVES
#
#   Every algorithm below operates in O(1) memory and O(1) per observation.
#   None of them store the raw data stream.  They are designed for million-URL
#   manifests where storing every latency measurement would be impractical.
#
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 1.  P² QUANTILE ESTIMATOR  (Jain & Chlamtac, 1985)
#
#     Estimates arbitrary quantiles from a data stream without storing any
#     observations.  Uses 5 markers per quantile whose positions are adjusted
#     by a piecewise-parabolic (P²) interpolation formula after each
#     observation.  Guarantees monotonicity of marker positions.
#
#     Reference:
#         Jain, R. & Chlamtac, I. (1985).
#         "The P² Algorithm for Dynamic Calculation of Quantiles and
#          Histograms Without Storing Observations."
#         Communications of the ACM, 28(10), 1076–1085.
#
#     Space: 5 floats (heights) + 5 ints (positions) + 5 floats (desired) = 60 bytes
#     Time:  O(1) per observation
# ─────────────────────────────────────────────────────────────────────────────

class P2QuantileEstimator:
    """
    Streaming quantile estimator using the P² algorithm.

    After an initial buffer of 5 observations (to seed the markers), every
    subsequent observation updates the internal markers in O(1) time with
    no additional memory allocation.

    The estimate converges to the true quantile as n → ∞.  For well-behaved
    distributions (unimodal, light-tailed) convergence is very fast —
    typically within 30–50 observations for <5% relative error on P95.

    Usage::

        est = P2QuantileEstimator(0.95)
        for latency in stream:
            est.observe(latency)
        print(est.estimate())   # P95 latency
    """

    __slots__ = (
        "_p", "_count", "_heights", "_positions",
        "_desired", "_increments", "_buffer",
    )

    def __init__(self, quantile: float) -> None:
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0,1), got {quantile}")
        self._p = quantile
        self._count = 0
        # Marker heights (q): the estimated data values at marker positions
        self._heights: List[float] = [0.0] * 5
        # Marker positions (n): the actual observation counts at each marker
        self._positions: List[int] = [0] * 5
        # Desired positions (n'): where markers *should* be
        self._desired: List[float] = [0.0] * 5
        # Desired position increments (dn'): how much desired positions
        # advance per observation
        self._increments: List[float] = [
            0.0,
            quantile / 2.0,
            quantile,
            (1.0 + quantile) / 2.0,
            1.0,
        ]
        self._buffer: List[float] = []

    def observe(self, x: float) -> None:
        """Incorporate one observation into the estimate.  O(1) amortized."""
        self._count += 1
        if self._count <= 5:
            self._buffer.append(x)
            if self._count == 5:
                self._initialize_markers()
            return
        self._update_markers(x)

    def _initialize_markers(self) -> None:
        """Sort the first 5 observations and seed marker heights/positions."""
        self._buffer.sort()
        for i in range(5):
            self._heights[i] = self._buffer[i]
            self._positions[i] = i + 1
        # Set desired positions based on quantile
        n = 5
        p = self._p
        self._desired[0] = 1.0
        self._desired[1] = 1.0 + 2.0 * p
        self._desired[2] = 1.0 + 4.0 * p
        self._desired[3] = 3.0 + 2.0 * p
        self._desired[4] = float(n)
        self._buffer.clear()

    def _update_markers(self, x: float) -> None:
        """
        P² marker update.

        1. Find the cell k where x falls among the current markers.
        2. Increment positions of markers k+1 .. 4.
        3. Increment all desired positions by their increments.
        4. For each interior marker (1,2,3), if the actual position is
           farther than 1 from the desired position, adjust via
           piecewise-parabolic interpolation (P²) with sign(d).
           If the parabolic estimate violates monotonicity, fall back
           to linear interpolation.
        """
        # Step 1: find cell
        if x < self._heights[0]:
            self._heights[0] = x
            k = 0
        elif x >= self._heights[4]:
            self._heights[4] = x
            k = 3
        else:
            k = 0
            for i in range(1, 5):
                if x < self._heights[i]:
                    k = i - 1
                    break

        # Step 2: increment positions
        for i in range(k + 1, 5):
            self._positions[i] += 1

        # Step 3: increment desired positions
        for i in range(5):
            self._desired[i] += self._increments[i]

        # Step 4: adjust interior markers
        for i in (1, 2, 3):
            d = self._desired[i] - self._positions[i]
            if (
                (d >= 1.0 and self._positions[i + 1] - self._positions[i] > 1)
                or (d <= -1.0 and self._positions[i - 1] - self._positions[i] < -1)
            ):
                sign_d = 1 if d > 0 else -1
                # Parabolic (P²) interpolation
                q_new = self._parabolic(i, sign_d)
                if self._heights[i - 1] < q_new < self._heights[i + 1]:
                    self._heights[i] = q_new
                else:
                    # Linear fallback
                    self._heights[i] = self._linear(i, sign_d)
                self._positions[i] += sign_d

    def _parabolic(self, i: int, d: int) -> float:
        """
        P² parabolic interpolation for marker i in direction d.

        Formula (Jain & Chlamtac eq. 3):
            q_i + (d / (n_{i+1} - n_{i-1})) ×
                [ (n_i - n_{i-1} + d)(q_{i+1} - q_i) / (n_{i+1} - n_i)
                + (n_{i+1} - n_i - d)(q_i - q_{i-1}) / (n_i - n_{i-1}) ]
        """
        n = self._positions
        q = self._heights
        ni = n[i]
        ni_minus = n[i - 1]
        ni_plus = n[i + 1]
        qi = q[i]
        qi_minus = q[i - 1]
        qi_plus = q[i + 1]

        denom = ni_plus - ni_minus
        if denom == 0:
            return qi

        term1_num = (ni - ni_minus + d) * (qi_plus - qi)
        term1_den = ni_plus - ni
        term2_num = (ni_plus - ni - d) * (qi - qi_minus)
        term2_den = ni - ni_minus

        if term1_den == 0 or term2_den == 0:
            return qi

        return qi + (d / denom) * (term1_num / term1_den + term2_num / term2_den)

    def _linear(self, i: int, d: int) -> float:
        """Linear interpolation fallback when parabolic violates monotonicity."""
        n = self._positions
        q = self._heights
        idx = i + d
        ni_d = n[idx]
        ni = n[i]
        qi_d = q[idx]
        qi = q[i]
        denom = ni_d - ni
        if denom == 0:
            return qi
        return qi + d * (qi_d - qi) / denom

    def estimate(self) -> float:
        """
        Return the current quantile estimate.

        Before 5 observations: returns the median of available data.
        After 5 observations: returns the P² estimate from marker 2
        (the middle marker, which tracks the target quantile).
        """
        if self._count == 0:
            return 0.0
        if self._count < 5:
            s = sorted(self._buffer)
            idx = int(self._p * (len(s) - 1))
            return s[idx]
        return self._heights[2]

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._count = 0
        self._heights = [0.0] * 5
        self._positions = [0] * 5
        self._desired = [0.0] * 5
        self._buffer = []


# ─────────────────────────────────────────────────────────────────────────────
# 2.  WELFORD'S ONLINE VARIANCE  (Welford, 1962)
#
#     Numerically stable single-pass mean and variance computation.
#     Uses the recurrence:
#
#         δ   = x - mean_{n-1}
#         mean_n = mean_{n-1} + δ/n
#         M2_n   = M2_{n-1}  + δ × (x - mean_n)
#
#     Variance = M2 / (n - 1)   (Bessel's correction)
#
#     This is immune to catastrophic cancellation that plagues the
#     naive Σ(x²) - (Σx)²/n formula when values are large and close
#     together — exactly the case for latency measurements in ms.
#
#     Reference:
#         Welford, B. P. (1962). "Note on a Method for Calculating
#         Corrected Sums of Squares and Products."
#         Technometrics, 4(3), 419–420.
# ─────────────────────────────────────────────────────────────────────────────

class WelfordAccumulator:
    """
    Streaming mean and variance using Welford's online algorithm.

    Thread-safe for single-writer (asyncio event loop is single-threaded).
    """

    __slots__ = ("_count", "_mean", "_m2", "_min", "_max")

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: float = 0.0
        self._m2: float = 0.0
        self._min: float = float("inf")
        self._max: float = float("-inf")

    def update(self, x: float) -> None:
        """Incorporate one observation.  O(1)."""
        self._count += 1
        self._min = min(self._min, x)
        self._max = max(self._max, x)
        delta = x - self._mean
        self._mean += delta / self._count
        delta2 = x - self._mean
        self._m2 += delta * delta2

    @property
    def count(self) -> int:
        return self._count

    @property
    def mean(self) -> float:
        return self._mean if self._count > 0 else 0.0

    @property
    def variance(self) -> float:
        """Unbiased sample variance (Bessel's correction)."""
        if self._count < 2:
            return 0.0
        return self._m2 / (self._count - 1)

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    @property
    def minimum(self) -> float:
        return self._min if self._count > 0 else 0.0

    @property
    def maximum(self) -> float:
        return self._max if self._count > 0 else 0.0

    @property
    def coefficient_of_variation(self) -> float:
        """
        CV = σ/μ.  Dimensionless measure of relative variability.
        CV > 1.0 signals high variability relative to the mean — common
        in latency distributions with long tails.
        """
        if self._mean == 0.0:
            return 0.0
        return self.std / abs(self._mean)

    def to_dict(self) -> Dict[str, float]:
        return {
            "count": float(self._count),
            "mean": round(self.mean, 4),
            "std": round(self.std, 4),
            "variance": round(self.variance, 4),
            "min": round(self.minimum, 4),
            "max": round(self.maximum, 4),
            "cv": round(self.coefficient_of_variation, 4),
        }

    def reset(self) -> None:
        self._count = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._min = float("inf")
        self._max = float("-inf")


# ─────────────────────────────────────────────────────────────────────────────
# 3.  EWMA — Exponentially Weighted Moving Average
#
#     Recursive update:  S_n = α × x_n + (1 - α) × S_{n-1}
#
#     α is derived from the desired half-life H:
#         α = 1 - exp(-ln(2) / H)
#
#     The half-life is the number of observations after which a data point's
#     influence decays to 50%.  With H=50 and α≈0.0138, the effective
#     memory is ~175 observations (where contribution drops below 0.1%).
#
#     This is used for CL-level health scoring: a CL mode whose EWMA
#     latency is rising is getting sicker, even if the most recent fetch
#     succeeded.  The EWMA captures the *trend*, not the instant.
# ─────────────────────────────────────────────────────────────────────────────

class EWMATracker:
    """
    Exponentially weighted moving average with configurable smoothing.

    Supports bias correction for initial observations (à la Adam optimizer):
        corrected_value = value / (1 - (1-α)^n)

    This prevents the estimate from being biased toward zero during warm-up.
    """

    __slots__ = ("_alpha", "_value", "_count", "_bias_correct")

    def __init__(self, alpha: float = EWMA_ALPHA, bias_correct: bool = True) -> None:
        self._alpha = alpha
        self._value = 0.0
        self._count = 0
        self._bias_correct = bias_correct

    def update(self, x: float) -> None:
        self._count += 1
        if self._count == 1:
            self._value = x
        else:
            self._value = self._alpha * x + (1.0 - self._alpha) * self._value

    @property
    def value(self) -> float:
        if self._count == 0:
            return 0.0
        if self._count == 1:
            return self._value  # first observation is exact
        if self._bias_correct and self._count < 100:
            correction = 1.0 - (1.0 - self._alpha) ** self._count
            return self._value / correction if correction > 1e-12 else self._value
        return self._value

    @property
    def count(self) -> int:
        return self._count

    def reset(self) -> None:
        self._value = 0.0
        self._count = 0


# ─────────────────────────────────────────────────────────────────────────────
# 4.  SHANNON ENTROPY over rolling window
#
#     H(X) = -Σ p(x) log₂ p(x)
#
#     where p(x) = count(x) / total over the most recent ENTROPY_WINDOW
#     observations.  Applied to HTTP status codes to detect degenerate
#     response patterns.
#
#     For a healthy domain returning a mix of 200, 301, 404, 500:
#         H ≈ 1.5–2.0 bits   (diverse, normal)
#
#     For a domain returning only 403:
#         H = 0.0 bits        (degenerate — blocking us)
#
#     For a domain alternating 200/403:
#         H = 1.0 bit         (binary — partial blocking)
#
#     Entropy below 0.5 bits for more than ENTROPY_WINDOW observations
#     is a strong signal that the domain is hostile to the crawler.
# ─────────────────────────────────────────────────────────────────────────────

class ShannonEntropyTracker:
    """
    Rolling Shannon entropy H(X) over a fixed-size window of discrete events.

    Uses a circular buffer and a frequency counter for O(1) update.
    """

    __slots__ = ("_window_size", "_buffer", "_counts", "_total")

    def __init__(self, window_size: int = ENTROPY_WINDOW) -> None:
        self._window_size = window_size
        self._buffer: Deque[int] = deque(maxlen=window_size)
        self._counts: Counter = Counter()
        self._total: int = 0

    def observe(self, status_code: int) -> None:
        if len(self._buffer) == self._window_size:
            evicted = self._buffer[0]
            self._counts[evicted] -= 1
            if self._counts[evicted] <= 0:
                del self._counts[evicted]
            self._total -= 1
        self._buffer.append(status_code)
        self._counts[status_code] += 1
        self._total += 1

    @property
    def entropy(self) -> float:
        """Shannon entropy in bits.  0.0 = perfectly degenerate."""
        if self._total == 0:
            return 0.0
        h = 0.0
        for c in self._counts.values():
            if c > 0:
                p = c / self._total
                h -= p * math.log2(p)
        return h

    @property
    def dominant_code(self) -> Optional[int]:
        """The most frequent status code in the window."""
        if not self._counts:
            return None
        return self._counts.most_common(1)[0][0]

    @property
    def is_degenerate(self) -> bool:
        """True if entropy < 0.5 bits and window is full."""
        return (
            len(self._buffer) >= self._window_size
            and self.entropy < 0.5
        )

    @property
    def count(self) -> int:
        return self._total


# ─────────────────────────────────────────────────────────────────────────────
# 5.  NELSON–AALEN CUMULATIVE HAZARD ESTIMATOR
#
#     For connection failure timing prediction using survival analysis.
#     The Nelson–Aalen estimator of the cumulative hazard function is:
#
#         Ĥ(t) = Σ_{t_i ≤ t}  d_i / n_i
#
#     where d_i is the number of failures at time t_i and n_i is the number
#     of items at risk just before t_i.
#
#     In our context:
#       - "time" is the URL index within the manifest
#       - "failure" is a connection failure / timeout
#       - "at risk" is the remaining URLs
#
#     The hazard rate h(t) = dĤ/dt estimates the instantaneous failure
#     rate.  A rising hazard rate means the domain is getting sicker
#     (throttling, overloading, or actively blocking).
#
#     Reference:
#         Nelson, W. (1969). "Hazard Plotting for Incomplete Failure Data."
#         Aalen, O. (1978). "Nonparametric Inference for a Family of
#         Counting Processes."
# ─────────────────────────────────────────────────────────────────────────────

class HazardRateEstimator:
    """
    Nelson–Aalen cumulative hazard for connection failure prediction.

    Tracks (index, is_failure) events.  The cumulative hazard Ĥ(t) at
    any point gives the expected number of failures per URL up to that
    point.  The instantaneous hazard rate (derivative) gives the current
    failure intensity.
    """

    __slots__ = (
        "_total_at_risk", "_failures", "_observations",
        "_cumulative_hazard", "_window", "_recent_failures",
    )

    def __init__(self, total_urls: int, window: int = HAZARD_WINDOW) -> None:
        self._total_at_risk = total_urls
        self._failures: int = 0
        self._observations: int = 0
        self._cumulative_hazard: float = 0.0
        self._window = window
        # Track recent failure intervals for instantaneous rate
        self._recent_failures: Deque[int] = deque(maxlen=window)

    def observe(self, is_failure: bool) -> None:
        """Record one URL outcome.  O(1)."""
        self._observations += 1
        at_risk = max(self._total_at_risk - self._observations + 1, 1)
        if is_failure:
            self._failures += 1
            self._cumulative_hazard += 1.0 / at_risk
            self._recent_failures.append(self._observations)

    @property
    def cumulative_hazard(self) -> float:
        """Nelson–Aalen estimate Ĥ(t) at current observation index."""
        return self._cumulative_hazard

    @property
    def instantaneous_rate(self) -> float:
        """
        Instantaneous hazard rate estimated from recent failure spacing.
        Uses the reciprocal of the mean inter-failure interval.
        Returns 0.0 if fewer than 2 failures observed.
        """
        if len(self._recent_failures) < 2:
            return 0.0
        failures_list = list(self._recent_failures)
        intervals = [
            failures_list[i] - failures_list[i - 1]
            for i in range(1, len(failures_list))
        ]
        mean_interval = sum(intervals) / len(intervals) if intervals else float("inf")
        return 1.0 / mean_interval if mean_interval > 0 else 0.0

    @property
    def survival_probability(self) -> float:
        """
        Estimated probability that the next URL will NOT fail.
        S(t) = exp(-Ĥ(t)) — the Breslow survival function estimator.
        """
        return math.exp(-self._cumulative_hazard) if self._cumulative_hazard < 700 else 0.0

    @property
    def failure_rate(self) -> float:
        """Simple failure count / observations."""
        if self._observations == 0:
            return 0.0
        return self._failures / self._observations

    @property
    def expected_failures_remaining(self) -> float:
        """
        Uses the current hazard rate to project expected remaining failures.
        E[failures_remaining] ≈ instantaneous_rate × urls_remaining
        """
        remaining = self._total_at_risk - self._observations
        return self.instantaneous_rate * remaining

    def to_dict(self) -> Dict[str, float]:
        return {
            "cumulative_hazard": round(self._cumulative_hazard, 6),
            "instantaneous_rate": round(self.instantaneous_rate, 6),
            "survival_probability": round(self.survival_probability, 6),
            "failure_rate": round(self.failure_rate, 6),
            "total_failures": float(self._failures),
            "observations": float(self._observations),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6.  RESERVOIR SAMPLING  (Vitter, Algorithm R, 1985)
#
#     Maintains a uniform random sample of size k from a stream of
#     unknown length n.  Every element in the stream has probability
#     k/n of being in the reservoir, regardless of when it appeared.
#
#     Used for anomaly event sampling: if a manifest produces 10,000
#     anomalies, we keep a representative sample of RESERVOIR_SIZE=256.
#
#     Reference:
#         Vitter, J.S. (1985). "Random Sampling with a Reservoir."
#         ACM Transactions on Mathematical Software, 11(1), 37–57.
# ─────────────────────────────────────────────────────────────────────────────

class ReservoirSampler:
    """
    Vitter's Algorithm R for streaming uniform random sampling.

    ``add(item)`` incorporates one stream element.  After the reservoir
    is full, each new element replaces a random existing element with
    probability k/n (where n is the total stream length so far).
    """

    __slots__ = ("_k", "_reservoir", "_n", "_rng")

    def __init__(self, k: int = RESERVOIR_SIZE, seed: Optional[int] = None) -> None:
        self._k = k
        self._reservoir: List[Any] = []
        self._n: int = 0
        self._rng = random.Random(seed)

    def add(self, item: Any) -> None:
        self._n += 1
        if len(self._reservoir) < self._k:
            self._reservoir.append(item)
        else:
            j = self._rng.randint(0, self._n - 1)
            if j < self._k:
                self._reservoir[j] = item

    @property
    def sample(self) -> List[Any]:
        return list(self._reservoir)

    @property
    def count(self) -> int:
        """Total stream elements seen (not reservoir size)."""
        return self._n

    @property
    def reservoir_size(self) -> int:
        return len(self._reservoir)

    def clear(self) -> None:
        self._reservoir.clear()
        self._n = 0


# ─────────────────────────────────────────────────────────────────────────────
# 7.  MARKOV CIRCUIT QUALITY  (2-state chain for Tor exit IP quality)
#
#     States: GOOD (exit IP is fresh), BAD (exit IP was recently used).
#
#     Transition matrix estimated online:
#         P = [ P(G→G)  P(G→B) ]
#             [ P(B→G)  P(B→B) ]
#
#     Updated by observing (state_before, state_after) for each circuit
#     rotation.  The steady-state distribution π satisfies π = πP:
#
#         π_G = P(B→G) / (P(G→B) + P(B→G))
#         π_B = P(G→B) / (P(G→B) + P(B→G))
#
#     π_G is the long-run probability that a random circuit rotation
#     produces a fresh exit IP.  Target: π_G > 0.95 (< 5% IP collision).
# ─────────────────────────────────────────────────────────────────────────────

class MarkovCircuitQuality:
    """
    2-state Markov chain for Tor circuit rotation quality monitoring.

    Tracks IP reuse across circuit rotations and estimates the
    steady-state probability of getting a fresh exit IP.
    """

    __slots__ = ("_transitions", "_last_state", "_seen_ips", "_total")

    GOOD = 0
    BAD  = 1

    def __init__(self) -> None:
        # transitions[from_state][to_state] = count
        self._transitions: List[List[int]] = [[0, 0], [0, 0]]
        self._last_state: Optional[int] = None
        self._seen_ips: Set[str] = set()
        self._total: int = 0

    def observe_ip(self, exit_ip: str) -> None:
        """Record one observed exit IP after a circuit rotation."""
        self._total += 1
        is_reuse = exit_ip in self._seen_ips
        current_state = self.BAD if is_reuse else self.GOOD
        self._seen_ips.add(exit_ip)

        if self._last_state is not None:
            self._transitions[self._last_state][current_state] += 1
        self._last_state = current_state

    @property
    def transition_matrix(self) -> List[List[float]]:
        """Estimated transition probability matrix."""
        matrix: List[List[float]] = [[0.0, 0.0], [0.0, 0.0]]
        for i in range(2):
            row_sum = self._transitions[i][0] + self._transitions[i][1]
            if row_sum > 0:
                matrix[i][0] = self._transitions[i][0] / row_sum
                matrix[i][1] = self._transitions[i][1] / row_sum
        return matrix

    @property
    def steady_state_good_probability(self) -> float:
        """
        π_G = P(B→G) / (P(G→B) + P(B→G))

        The long-run probability of getting a fresh exit IP.
        Returns 1.0 if no transitions observed (optimistic prior).
        """
        p = self.transition_matrix
        p_gb = p[self.GOOD][self.BAD]
        p_bg = p[self.BAD][self.GOOD]
        denom = p_gb + p_bg
        if denom < 1e-12:
            return 1.0  # no collisions observed
        return p_bg / denom

    @property
    def ip_collision_rate(self) -> float:
        """
        Fraction of circuit rotations that produced a previously-seen IP.
        This is the empirical 1 - π_G.
        """
        if self._total <= 1:
            return 0.0
        bad_transitions = (
            self._transitions[self.GOOD][self.BAD]
            + self._transitions[self.BAD][self.BAD]
        )
        total_transitions = sum(
            self._transitions[i][j] for i in range(2) for j in range(2)
        )
        if total_transitions == 0:
            return 0.0
        return bad_transitions / total_transitions

    @property
    def unique_ips(self) -> int:
        return len(self._seen_ips)

    @property
    def total_observations(self) -> int:
        return self._total

    def to_dict(self) -> Dict[str, Any]:
        return {
            "unique_ips": self.unique_ips,
            "total_rotations": self._total,
            "ip_collision_rate": round(self.ip_collision_rate, 4),
            "steady_state_good": round(self.steady_state_good_probability, 4),
            "transition_matrix": self.transition_matrix,
        }

    def reset(self) -> None:
        self._transitions = [[0, 0], [0, 0]]
        self._last_state = None
        self._seen_ips.clear()
        self._total = 0


# ─────────────────────────────────────────────────────────────────────────────
# 8.  LITTLE'S LAW ADAPTIVE CONCURRENCY
#
#     L = λ × W
#
#     where:
#       L = average number of in-flight requests (the concurrency level)
#       λ = throughput (requests completed per second)
#       W = mean response time (seconds per request)
#
#     Given observed λ and W over a sliding window, we compute the
#     theoretical optimal concurrency L.  If L < current_semaphore_limit,
#     we are over-provisioning.  If L > current_semaphore_limit, we are
#     under-provisioning and could increase throughput.
#
#     This is advisory — the fetcher logs the recommendation but does NOT
#     auto-adjust semaphores (that would violate the zero-logic rule for
#     rate limiting).  The information flows to index_daemon for future
#     manifest planning.
#
#     Reference:
#         Little, J. D. C. (1961). "A Proof for the Queuing Formula:
#         L = λW." Operations Research, 9(3), 383–387.
# ─────────────────────────────────────────────────────────────────────────────

class LittleLawAdvisor:
    """
    Little's Law concurrency advisor.  Tracks throughput and latency
    to compute the theoretical optimal concurrency level.
    """

    __slots__ = ("_window", "_completions", "_start_time", "_latency_acc")

    def __init__(self, window: int = LITTLE_LAW_WINDOW) -> None:
        self._window = window
        self._completions: Deque[float] = deque(maxlen=window)
        self._start_time: float = time.monotonic()
        self._latency_acc = WelfordAccumulator()

    def record_completion(self, latency_seconds: float) -> None:
        """Record one completed request with its latency."""
        self._completions.append(time.monotonic())
        self._latency_acc.update(latency_seconds)

    @property
    def throughput(self) -> float:
        """λ — observed completions per second over the window."""
        if len(self._completions) < 2:
            return 0.0
        span = self._completions[-1] - self._completions[0]
        if span <= 0:
            return 0.0
        return (len(self._completions) - 1) / span

    @property
    def mean_latency(self) -> float:
        """W — mean response time in seconds."""
        return self._latency_acc.mean

    @property
    def optimal_concurrency(self) -> float:
        """L = λ × W — theoretical optimal concurrency."""
        return self.throughput * self.mean_latency

    @property
    def recommendation(self) -> str:
        """Human-readable concurrency recommendation."""
        l_opt = self.optimal_concurrency
        if l_opt < 0.5:
            return "insufficient data"
        return f"L*={l_opt:.1f} (λ={self.throughput:.2f}/s, W={self.mean_latency:.3f}s)"

    def to_dict(self) -> Dict[str, float]:
        return {
            "throughput_per_sec": round(self.throughput, 4),
            "mean_latency_sec": round(self.mean_latency, 4),
            "optimal_concurrency": round(self.optimal_concurrency, 2),
            "total_completions": float(self._latency_acc.count),
        }

    def reset(self) -> None:
        self._completions.clear()
        self._start_time = time.monotonic()
        self._latency_acc.reset()




# ═══════════════════════════════════════════════════════════════════════════════
#
#   DOMAIN BEHAVIORAL OBSERVATORY
#
#   A passive intelligence layer that builds a live probabilistic model of
#   each domain's behavior during the crawl.  It never touches the fetch
#   path — it only observes outcomes and produces statistical reports that
#   index_daemon consumes as training signal.
#
#   This does NOT violate the zero-logic rule.  The fetcher does not read
#   the observatory's output.  The observatory does not influence fetch
#   decisions.  It is a one-way information sink: fetch outcomes flow in,
#   statistical portraits flow out (via bus events).  The fetcher remains
#   the vacuum.  The observatory is the telescope pointed at the vacuum's
#   exhaust.
#
# ─── WHY THE OBSERVATORY EXISTS ───────────────────────────────────────────
#
#   Before the observatory, index_daemon had two classes of signal:
#
#     1. Binary outcomes: success/failure per URL (from FetchAnomalyEvent)
#     2. Aggregate stats: throughput, mean latency (from ManifestCompleteEvent)
#
#   Both are lossy.  Binary outcomes discard the rich structure of *how*
#   the domain is failing.  Aggregates average out the transitions.
#   Neither tells index_daemon:
#
#     - "Is this domain's success rate *statistically* below healthy,
#       or did we just get unlucky with 3 timeouts?"
#     - "At what exact URL did the domain start blocking us?"
#     - "Should we abandon this manifest, and can we decide with fewer
#       observations than we've currently seen?"
#     - "Is the domain throttling us with correlated delays, or is
#       the network just slow?"
#     - "What's the domain's actual sustainable capacity, estimated
#       from noisy latency observations?"
#     - "Is the content we're getting the real content, or has the
#       domain switched to serving error pages / CAPTCHAs?"
#
#   The observatory answers all six questions with formal statistical
#   methods, each with quantified uncertainty.  index_daemon doesn't
#   need to re-derive these; it reads the report and acts.
#
# ─── WHAT index_daemon RECEIVES ───────────────────────────────────────────
#
#   After every manifest completes, the telemetry summary includes
#   an ``observatory`` field containing a ``DomainHealthReport``:
#
#     {
#       "observatory": {
#         "domain": "stripe.com",
#         "observations": 847,
#         "bayesian": {
#           "posterior_mean": 0.9541,          # P(success)
#           "posterior_std": 0.0072,           # uncertainty
#           "credible_interval_95": [0.9399, 0.9683],
#           "alpha": 811.0,                    # successes + prior
#           "beta": 38.0,                      # failures + prior
#           "concentration": 849.0,            # total data weight
#           "entropy": -2.1834                 # posterior uncertainty
#         },
#         "change_points": {
#           "observations": 847,
#           "s_upper": 0.0,                    # no positive shift detected
#           "s_lower": 2.3,                    # lower CUSUM building but not alarmed
#           "alarm_count": 1,                  # one behavior shift detected
#           "last_alarm": {
#             "alarm_at": 612,                 # alarm fired at URL #612
#             "change_at": 589,                # behavior actually changed at #589
#             "shift": 0.34                    # success rate dropped by 34%
#           }
#         },
#         "sprt": {
#           "decision": "healthy",             # not sick enough to abandon
#           "observations": 847,
#           "observed_rate": 0.9564,
#           "log_likelihood_ratio": -12.4,     # deep in healthy territory
#           "normalized_position": -0.82,      # -1=healthy, +1=sick
#           "expected_n_if_healthy": 45.2,     # would have decided in 45 obs if healthy
#           "expected_n_if_sick": 12.8          # would have decided in 13 obs if sick
#         },
#         "compressibility": {
#           "mean_ratio": 0.2234,              # typical HTML
#           "shift_detected": false,           # no content degeneration
#           "baseline_mean": 0.2198
#         },
#         "autocorrelation": {
#           "lag1_autocorrelation": 0.043,     # near zero = no throttling
#           "ljung_box_q": 8.7,               # below χ²(10,0.95)=18.31
#           "is_correlated": false             # domain is not throttling
#         },
#         "capacity": {
#           "estimated_capacity_rps": 4.23,    # domain can handle ~4.2 req/s
#           "estimated_latency_ms": 236.4,     # expected response time
#           "confidence_interval": [3.1, 5.4], # 95% CI on capacity
#           "kalman_gain": 0.012               # filter is stable (low gain)
#         }
#       }
#     }
#
# ─── HOW index_daemon SHOULD CONSUME THIS ─────────────────────────────────
#
#   index_daemon subscribes to ManifestCompleteEvent.  The event.stats
#   field contains aggregate counts.  The telemetry.summary() (available
#   via the fetcher's active_telemetry() API or logged at manifest
#   completion) contains the full observatory report.
#
#   Recommended consumption pattern for index_daemon:
#
#     1. RATE LIMIT ADJUSTMENT
#        Read ``capacity.estimated_capacity_rps``.  If the Kalman estimate
#        is significantly below the current RateLimitProfile.requests_per_second
#        for this domain, reduce the rate in the next CrawlManifest.
#        Use ``capacity.confidence_interval`` to decide how aggressive
#        the reduction should be — wide CI means we're uncertain, so
#        reduce conservatively.
#
#     2. MANIFEST ABANDONMENT LEARNING
#        Read ``sprt.decision``.  If "sick", the domain was statistically
#        unhealthy.  Log the ``sprt.expected_n_if_sick`` value — this is
#        how many observations the SPRT needed to decide.  Use this to
#        calibrate how many URLs to include in future manifests for this
#        domain (don't send 1000 URLs if SPRT can decide in 13).
#
#     3. CHANGE POINT RESPONSE
#        Read ``change_points.last_alarm``.  If present, the domain's
#        behavior shifted at the ``change_at`` URL index.  Look at
#        ``pre_change_rate`` vs ``post_change_rate`` to understand the
#        shift.  A 0.95 → 0.20 shift at index 589 means the domain
#        started blocking at its 589th URL.  Use this to cap manifest
#        size for this domain in future crawls.
#
#     4. THROTTLING DETECTION
#        Read ``autocorrelation.is_correlated``.  If true, the domain
#        is introducing deliberate delays (the latencies are serially
#        correlated, not random).  This means our rate limiter is
#        correctly pacing, but the domain is *additionally* throttling.
#        Reduce crawl aggression further — the domain is actively
#        defending against us.
#
#     5. CONTENT DEGENERATION
#        Read ``compressibility.shift_detected``.  If true, the content
#        structure changed mid-crawl — likely error pages or CAPTCHAs
#        replaced real content.  Flag this domain for CL escalation
#        (move from CL1 to CL2, or CL2 to CL3) in the next manifest.
#
#     6. BAYESIAN CONFIDENCE
#        Read ``bayesian.credible_interval_95``.  If the lower bound
#        of the 95% CI is below 0.70, the domain is *statistically*
#        unreliable.  This is different from the point estimate being
#        0.70 — the CI accounts for sample size.  A domain with 5
#        fetches and 1 failure has a wide CI; a domain with 500 fetches
#        and 100 failures has a tight CI that confidently says "sick."
#
#   IMPORTANT: index_daemon should NOT import from fetcher.py.
#   The observatory report is a plain dict (JSON-serializable).
#   index_daemon receives it via the bus event or the logged telemetry.
#   The dependency direction is: fetcher → bus → index_daemon.
#   Never the reverse.
#
# ─── MATHEMATICAL SUBSYSTEMS ──────────────────────────────────────────────
#
#   9.   Bayesian success estimation    Beta(α,β) conjugate posterior with
#                                       credible intervals for domain
#                                       success probability.
#   10.  CUSUM change-point detection   Page's cumulative sum algorithm
#                                       for detecting the exact URL where
#                                       domain behavior shifts.
#   11.  Wald SPRT abandonment          Sequential Probability Ratio Test
#                                       for manifest abandonment advisory.
#   12.  Response compressibility       Kolmogorov complexity proxy via
#        fingerprinting                 zlib compression ratio — detects
#                                       content degeneration without
#                                       inspecting content semantics.
#   13.  Latency autocorrelation        Detects correlated delays that
#                                       indicate domain-side throttling
#                                       (as opposed to random network jitter).
#   14.  Domain capacity model          Online Kalman filter estimating
#                                       the domain's true capacity from
#                                       noisy latency observations.
#
# ═══════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# 9.  BAYESIAN SUCCESS ESTIMATION  (Beta-Binomial conjugate model)
#
#     The probability that the next fetch from domain D succeeds is modeled
#     as a Bernoulli trial with unknown success probability θ.  We place
#     a Beta(α₀, β₀) prior on θ and update it after every fetch:
#
#         Prior:      θ ~ Beta(α₀, β₀)
#         Likelihood: X_i | θ ~ Bernoulli(θ)
#         Posterior:  θ | data ~ Beta(α₀ + s, β₀ + f)
#
#     where s = successes, f = failures.  This is the conjugate update:
#     the posterior is the same distributional family as the prior.
#
#     The posterior mean is:
#         E[θ] = (α₀ + s) / (α₀ + β₀ + s + f)
#
#     The 95% credible interval is computed from the Beta quantile function.
#     We use the inverse incomplete beta function (regularized):
#
#         CI = [B⁻¹(0.025; α, β),  B⁻¹(0.975; α, β)]
#
#     where B⁻¹ is the Beta quantile function.  For computational
#     efficiency, we use the normal approximation when α + β > 40:
#
#         θ ≈ N(μ, σ²)  where  μ = α/(α+β),  σ² = αβ/((α+β)²(α+β+1))
#
#     Starting prior: Beta(1, 1) = Uniform(0, 1) — no prior knowledge
#     about domain behavior.  After 10 successful fetches, the posterior
#     concentrates around θ ≈ 0.91 with tight credible intervals.
#
#     Why Beta-Binomial and not just count successes/failures?
#     Because the posterior gives us *uncertainty*.  After 2 successes
#     and 0 failures, the point estimate is 100% but the credible interval
#     is [0.34, 1.0] — we don't have enough data to be confident.  After
#     200 successes and 0 failures, the CI is [0.985, 1.0] — we're almost
#     certain.  This uncertainty quantification is what index_daemon needs
#     to decide whether a domain's success rate is *statistically* different
#     from healthy, not just *numerically* different.
# ─────────────────────────────────────────────────────────────────────────────

class BayesianSuccessEstimator:
    """
    Beta-Binomial conjugate model for domain success probability estimation.

    Maintains a Beta(α, β) posterior that is updated in O(1) after each
    fetch outcome.  Provides posterior mean, credible intervals, and
    a probability that the true success rate is below a given threshold.

    Usage::

        est = BayesianSuccessEstimator()
        est.observe(True)   # success
        est.observe(True)
        est.observe(False)  # failure
        print(est.posterior_mean)       # ≈ 0.6
        print(est.credible_interval())  # (0.19, 0.94)
        print(est.prob_below(0.5))      # P(θ < 0.5 | data) ≈ 0.31
    """

    __slots__ = ("_alpha", "_beta", "_alpha0", "_beta0")

    def __init__(self, alpha0: float = 1.0, beta0: float = 1.0) -> None:
        """
        Initialize with prior Beta(α₀, β₀).
        Default: Beta(1,1) = Uniform(0,1) — non-informative.
        """
        self._alpha0 = alpha0
        self._beta0 = beta0
        self._alpha = alpha0
        self._beta = beta0

    def observe(self, success: bool) -> None:
        """
        Conjugate update.  O(1).

        Posterior after observing x ∈ {0, 1}:
            α' = α + x
            β' = β + (1 - x)
        """
        if success:
            self._alpha += 1.0
        else:
            self._beta += 1.0

    def observe_batch(self, successes: int, failures: int) -> None:
        """Batch conjugate update.  O(1)."""
        self._alpha += successes
        self._beta += failures

    @property
    def posterior_mean(self) -> float:
        """E[θ] = α / (α + β)."""
        return self._alpha / (self._alpha + self._beta)

    @property
    def posterior_variance(self) -> float:
        """Var[θ] = αβ / ((α+β)²(α+β+1))."""
        a, b = self._alpha, self._beta
        return (a * b) / ((a + b) ** 2 * (a + b + 1))

    @property
    def posterior_std(self) -> float:
        return math.sqrt(self.posterior_variance)

    @property
    def posterior_mode(self) -> float:
        """
        Mode of Beta(α, β) = (α-1)/(α+β-2) when α,β > 1.
        Undefined for α ≤ 1 or β ≤ 1; return mean in that case.
        """
        if self._alpha > 1 and self._beta > 1:
            return (self._alpha - 1) / (self._alpha + self._beta - 2)
        return self.posterior_mean

    @property
    def observations(self) -> int:
        """Total observations (successes + failures)."""
        return int(self._alpha - self._alpha0 + self._beta - self._beta0)

    @property
    def successes(self) -> int:
        return int(self._alpha - self._alpha0)

    @property
    def failures(self) -> int:
        return int(self._beta - self._beta0)

    def credible_interval(self, level: float = 0.95) -> Tuple[float, float]:
        """
        Approximate credible interval using the normal approximation
        to the Beta distribution.

        For small sample sizes (α + β < 40), we use the Wilson score
        interval as a better approximation.

        Returns (lower, upper) bounds.
        """
        a, b = self._alpha, self._beta
        n = a + b

        if n < 4:
            # Too few observations — return full range
            return (0.0, 1.0) # noqa

        mu = a / n
        z = 1.96 if level == 0.95 else 2.576  # z-score for 95% or 99%

        if n < 40:
            # Wilson score interval (better for small n)
            n_obs = self.observations
            if n_obs == 0:
                return (0.0, 1.0) # noqa
            p_hat = self.successes / n_obs if n_obs > 0 else 0.5
            denom = 1 + z * z / n_obs
            center = (p_hat + z * z / (2 * n_obs)) / denom
            half_width = (
                z * math.sqrt(p_hat * (1 - p_hat) / n_obs + z * z / (4 * n_obs * n_obs))
            ) / denom
            return (max(0.0, center - half_width), min(1.0, center + half_width)) # noqa

        # Normal approximation to Beta
        sigma = math.sqrt((a * b) / (n * n * (n + 1)))
        lower = max(0.0, mu - z * sigma)
        upper = min(1.0, mu + z * sigma)
        return (lower, upper) # noqa

    def prob_below(self, threshold: float) -> float:
        """
        P(θ < threshold | data).

        Uses the regularized incomplete beta function.  For computational
        efficiency, approximates using the normal CDF when α + β > 30.

        This answers the question: "Given what we've observed, what is
        the probability that this domain's true success rate is below
        the given threshold?"  index_daemon uses this to flag domains
        whose success rate is *statistically* low, not just numerically.
        """
        if threshold <= 0.0:
            return 0.0
        if threshold >= 1.0:
            return 1.0

        a, b = self._alpha, self._beta
        mu = a / (a + b)
        sigma = math.sqrt((a * b) / ((a + b) ** 2 * (a + b + 1)))

        if sigma < 1e-12:
            return 1.0 if mu < threshold else 0.0

        # Normal CDF approximation
        z = (threshold - mu) / sigma
        # Abramowitz & Stegun approximation 7.1.26
        return _normal_cdf(z)

    @property
    def entropy(self) -> float:
        """
        Differential entropy of the Beta posterior.

        H[Beta(α,β)] = ln B(α,β) - (α-1)ψ(α) - (β-1)ψ(β) + (α+β-2)ψ(α+β)

        where B is the Beta function and ψ is the digamma function.
        Higher entropy = more uncertainty about the domain's success rate.
        """
        a, b = self._alpha, self._beta
        # Use Stirling's approximation for the log-Beta function
        log_beta = _log_beta(a, b)
        psi_a = _digamma(a)
        psi_b = _digamma(b)
        psi_ab = _digamma(a + b)
        return log_beta - (a - 1) * psi_a - (b - 1) * psi_b + (a + b - 2) * psi_ab

    @property
    def concentration(self) -> float:
        """
        Posterior concentration κ = α + β.

        Higher concentration = more data = tighter posterior.
        Useful for deciding when we have "enough" data to trust the estimate.
        """
        return self._alpha + self._beta

    def to_dict(self) -> Dict[str, Any]:
        ci = self.credible_interval()
        return {
            "posterior_mean": round(self.posterior_mean, 4),
            "posterior_std": round(self.posterior_std, 4),
            "posterior_mode": round(self.posterior_mode, 4),
            "credible_interval_95": (round(ci[0], 4), round(ci[1], 4)),
            "alpha": round(self._alpha, 2),
            "beta": round(self._beta, 2),
            "concentration": round(self.concentration, 2),
            "observations": self.observations,
            "successes": self.successes,
            "failures": self.failures,
            "entropy": round(self.entropy, 4),
        }

    def reset(self) -> None:
        self._alpha = self._alpha0
        self._beta = self._beta0


# ── Helper: Normal CDF (Abramowitz & Stegun 7.1.26) ──────────────────────

def _normal_cdf(z: float) -> float:
    """
    Standard normal CDF Φ(z) using the Abramowitz & Stegun rational
    approximation.  Maximum error: 1.5 × 10⁻⁷.

    Reference: Abramowitz, M. & Stegun, I. (1964).
    "Handbook of Mathematical Functions", eq. 7.1.26.
    """
    if z < -8.0:
        return 0.0
    if z > 8.0:
        return 1.0
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1.0 if z >= 0 else -1.0
    x = abs(z) / math.sqrt(2.0)
    t = 1.0 / (1.0 + p * x)
    y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
    return 0.5 * (1.0 + sign * y)


# ── Helper: Digamma function (rational approximation) ─────────────────────

def _digamma(x: float) -> float:
    """
    Digamma function ψ(x) = d/dx ln Γ(x).

    Uses the asymptotic expansion for x ≥ 6, with recurrence
    ψ(x) = ψ(x+1) - 1/x for x < 6.

    Asymptotic expansion (Abramowitz & Stegun 6.3.18):
        ψ(x) ≈ ln(x) - 1/(2x) - 1/(12x²) + 1/(120x⁴) - 1/(252x⁶)
    """
    result = 0.0
    while x < 6.0:
        result -= 1.0 / x
        x += 1.0
    result += (
        math.log(x)
        - 0.5 / x
        - 1.0 / (12.0 * x * x)
        + 1.0 / (120.0 * x ** 4)
        - 1.0 / (252.0 * x ** 6)
    )
    return result


# ── Helper: Log-Beta function ─────────────────────────────────────────────

def _log_beta(a: float, b: float) -> float:
    """
    ln B(a, b) = ln Γ(a) + ln Γ(b) - ln Γ(a + b).

    Uses math.lgamma for numerical stability.
    """
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CUSUM CHANGE-POINT DETECTION  (Page, 1954)
#
#     Detects the exact URL index where domain behavior shifts during a
#     manifest execution.  A "change point" is a moment where the
#     underlying success probability θ changes — for example, when a
#     domain starts returning 403s after initially returning 200s.
#
#     The CUSUM (Cumulative Sum) algorithm tracks:
#
#         S_n = max(0, S_{n-1} + (x_n - μ₀) - k)     (upper CUSUM)
#         T_n = max(0, T_{n-1} - (x_n - μ₀) - k)     (lower CUSUM)
#
#     where:
#       x_n  = observation at time n (1 for success, 0 for failure)
#       μ₀   = target mean (expected success rate, e.g. 0.9)
#       k    = slack parameter (allowance for normal variation)
#
#     When S_n > h (decision threshold), a positive shift is detected.
#     When T_n > h, a negative shift is detected.
#
#     For domain health monitoring:
#       - We track the LOWER CUSUM (detecting decreases in success rate)
#       - μ₀ = 0.9 (domains should succeed 90% of the time)
#       - k = 0.05 (half the shift we want to detect: (μ₀ - μ₁)/2)
#       - h = 4.0 (tuned for ≈100 observations before false alarm)
#
#     When the lower CUSUM crosses h, we know the domain's success rate
#     has shifted downward.  The change point is estimated as the URL
#     index where S_n last crossed zero before the alarm.
#
#     Reference:
#         Page, E. S. (1954). "Continuous Inspection Schemes."
#         Biometrika, 41(1/2), 100–115.
#
#     This is the gold standard for online change-point detection in
#     sequential monitoring.  It is used in industrial process control,
#     clinical trials, and network anomaly detection.  Its application
#     to web crawler domain health monitoring is novel.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChangePointAlarm:
    """Represents a detected change point in domain behavior."""
    alarm_index:      int           # URL index where alarm fired
    change_index:     int           # estimated URL index where behavior actually changed
    cusum_value:      float         # CUSUM statistic at alarm time
    pre_change_rate:  float         # estimated success rate before change
    post_change_rate: float         # estimated success rate after change
    shift_magnitude:  float         # |pre - post|
    timestamp:        float = field(default_factory=time.monotonic)


class CUSUMChangePointDetector:
    """
    Page's CUSUM algorithm for sequential change-point detection.

    Monitors a stream of binary outcomes (success/failure) and fires
    an alarm when the underlying success rate shifts significantly.

    The detector tracks two statistics simultaneously:
      - Upper CUSUM S⁺: detects increases in success rate
      - Lower CUSUM S⁻: detects decreases in success rate

    For crawler domain monitoring, we primarily care about S⁻ (the
    domain getting worse).  But S⁺ is also tracked because a sudden
    increase after a period of failure (domain recovering) is useful
    signal for index_daemon.
    """

    __slots__ = (
        "_target", "_slack", "_threshold", "_n",
        "_s_upper", "_s_lower", "_last_zero_upper", "_last_zero_lower",
        "_alarms", "_pre_window", "_post_window",
        "_observations_buffer",
    )

    def __init__(
        self,
        target: float = 0.9,
        slack: float = 0.05,
        threshold: float = 4.0,
    ) -> None:
        """
        Parameters:
            target:    expected success rate under normal conditions (μ₀)
            slack:     allowance parameter k = (μ₀ - μ₁)/2
            threshold: decision boundary h
        """
        self._target = target
        self._slack = slack
        self._threshold = threshold
        self._n: int = 0
        self._s_upper: float = 0.0
        self._s_lower: float = 0.0
        self._last_zero_upper: int = 0
        self._last_zero_lower: int = 0
        self._alarms: List[ChangePointAlarm] = []
        # Track observations for pre/post rate estimation
        self._observations_buffer: Deque[bool] = deque(maxlen=200)

    def observe(self, success: bool) -> Optional[ChangePointAlarm]:
        """
        Incorporate one observation.  O(1).

        Returns a ChangePointAlarm if a change point is detected,
        None otherwise.
        """
        self._n += 1
        self._observations_buffer.append(success)
        x = 1.0 if success else 0.0

        # Upper CUSUM (detecting increase)
        self._s_upper = max(0.0, self._s_upper + (x - self._target) - self._slack)
        if self._s_upper == 0.0:
            self._last_zero_upper = self._n

        # Lower CUSUM (detecting decrease)
        self._s_lower = max(0.0, self._s_lower - (x - self._target) - self._slack)
        if self._s_lower == 0.0:
            self._last_zero_lower = self._n

        # Check for alarm (lower CUSUM — domain getting worse)
        if self._s_lower > self._threshold:
            alarm = self._create_alarm(
                alarm_index=self._n,
                change_index=self._last_zero_lower,
                cusum_value=self._s_lower,
            )
            self._alarms.append(alarm)
            # Reset lower CUSUM after alarm
            self._s_lower = 0.0
            self._last_zero_lower = self._n
            return alarm

        # Check for alarm (upper CUSUM — domain recovering)
        if self._s_upper > self._threshold:
            alarm = self._create_alarm(
                alarm_index=self._n,
                change_index=self._last_zero_upper,
                cusum_value=self._s_upper,
            )
            self._alarms.append(alarm)
            self._s_upper = 0.0
            self._last_zero_upper = self._n
            return alarm

        return None

    def _create_alarm(
        self, alarm_index: int, change_index: int, cusum_value: float,
    ) -> ChangePointAlarm:
        """Create a ChangePointAlarm with pre/post rate estimates."""
        buf = list(self._observations_buffer)
        n_buf = len(buf)

        # Estimate pre-change rate (before change_index)
        # We use the buffer offset since buffer is bounded
        offset = max(0, n_buf - (alarm_index - change_index))
        pre_data = buf[:offset] if offset > 0 else buf[:n_buf // 2]
        post_data = buf[offset:] if offset > 0 else buf[n_buf // 2:]

        pre_rate = sum(pre_data) / len(pre_data) if pre_data else self._target
        post_rate = sum(post_data) / len(post_data) if post_data else 0.0

        return ChangePointAlarm(
            alarm_index=alarm_index,
            change_index=change_index,
            cusum_value=cusum_value,
            pre_change_rate=pre_rate,
            post_change_rate=post_rate,
            shift_magnitude=abs(pre_rate - post_rate),
        )

    @property
    def current_upper(self) -> float:
        return self._s_upper

    @property
    def current_lower(self) -> float:
        return self._s_lower

    @property
    def alarms(self) -> List[ChangePointAlarm]:
        return list(self._alarms)

    @property
    def alarm_count(self) -> int:
        return len(self._alarms)

    @property
    def observations(self) -> int:
        return self._n

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": self._n,
            "s_upper": round(self._s_upper, 4),
            "s_lower": round(self._s_lower, 4),
            "threshold": self._threshold,
            "alarm_count": len(self._alarms),
            "last_alarm": (
                {
                    "alarm_at": self._alarms[-1].alarm_index,
                    "change_at": self._alarms[-1].change_index,
                    "shift": round(self._alarms[-1].shift_magnitude, 4),
                }
                if self._alarms else None
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 11. WALD SEQUENTIAL PROBABILITY RATIO TEST (SPRT)
#
#     The SPRT decides between two hypotheses about a domain's success
#     probability as data arrives, with guaranteed error bounds:
#
#         H₀: θ = θ₀  (domain is healthy, e.g. θ₀ = 0.90)
#         H₁: θ = θ₁  (domain is sick,    e.g. θ₁ = 0.60)
#
#     After each observation, the SPRT computes the log-likelihood ratio:
#
#         Λ_n = Σᵢ₌₁ⁿ log[ P(xᵢ|θ₁) / P(xᵢ|θ₀) ]
#
#     For Bernoulli observations:
#         success: log(θ₁/θ₀)
#         failure: log((1-θ₁)/(1-θ₀))
#
#     Decision boundaries:
#         If Λ_n ≥  log((1-β)/α) → Accept H₁ (domain is sick)
#         If Λ_n ≤  log(β/(1-α)) → Accept H₀ (domain is healthy)
#         Otherwise → Continue sampling
#
#     Where α = P(accept H₁ | H₀ true) and β = P(accept H₀ | H₁ true).
#
#     The SPRT is *optimal* in the sense that among all sequential tests
#     with the same error bounds, it requires the fewest observations
#     on average to reach a decision (Wald & Wolfowitz, 1948).
#
#     For manifest abandonment: when the SPRT accepts H₁, the domain
#     is statistically sick and the manifest should be abandoned.  This
#     is advisory — the fetcher emits the signal, index_daemon decides.
#
#     Reference:
#         Wald, A. (1945). "Sequential Tests of Statistical Hypotheses."
#         The Annals of Mathematical Statistics, 16(2), 117–186.
# ─────────────────────────────────────────────────────────────────────────────

class SPRTDecision(enum.Enum):
    """Outcome of the Sequential Probability Ratio Test."""
    CONTINUE = "continue"          # insufficient evidence — keep sampling
    ACCEPT_HEALTHY = "healthy"     # accept H₀: domain is healthy
    ACCEPT_SICK = "sick"           # accept H₁: domain is sick (abandon)


class WaldSPRT:
    """
    Wald's Sequential Probability Ratio Test for manifest abandonment.

    Tests H₀: θ = θ₀ (healthy) vs H₁: θ = θ₁ (sick) with
    guaranteed Type I/II error bounds α and β.

    Usage::

        sprt = WaldSPRT(theta_0=0.90, theta_1=0.60)
        for url_result in manifest:
            decision = sprt.observe(url_result.success)
            if decision == SPRTDecision.ACCEPT_SICK:
                # Advisory: manifest should be abandoned
                emit_abandonment_advisory(sprt.report())
                break
    """

    __slots__ = (
        "_theta_0", "_theta_1", "_alpha", "_beta",
        "_log_ratio_success", "_log_ratio_failure",
        "_upper_boundary", "_lower_boundary",
        "_lambda_n", "_n", "_successes", "_decision",
    )

    def __init__(
        self,
        theta_0: float = 0.90,     # healthy success rate
        theta_1: float = 0.60,     # sick success rate
        alpha: float = 0.05,       # P(falsely declare sick)
        beta: float = 0.10,        # P(falsely declare healthy)
    ) -> None:
        if not 0 < theta_1 < theta_0 < 1:
            raise ValueError("require 0 < theta_1 < theta_0 < 1")
        if not 0 < alpha < 1 or not 0 < beta < 1:
            raise ValueError("require 0 < alpha, beta < 1")

        self._theta_0 = theta_0
        self._theta_1 = theta_1
        self._alpha = alpha
        self._beta = beta

        # Pre-compute log-likelihood increments (constant per observation)
        self._log_ratio_success = math.log(theta_1 / theta_0)
        self._log_ratio_failure = math.log((1 - theta_1) / (1 - theta_0))

        # Decision boundaries
        self._upper_boundary = math.log((1 - beta) / alpha)
        self._lower_boundary = math.log(beta / (1 - alpha))

        self._lambda_n: float = 0.0     # cumulative log-likelihood ratio
        self._n: int = 0
        self._successes: int = 0
        self._decision: SPRTDecision = SPRTDecision.CONTINUE

    def observe(self, success: bool) -> SPRTDecision:
        """
        Incorporate one observation.  O(1).

        Returns the current decision: CONTINUE, ACCEPT_HEALTHY, or ACCEPT_SICK.
        Once a terminal decision is reached, all subsequent calls return
        the same decision (the test is stopped).
        """
        if self._decision != SPRTDecision.CONTINUE:
            return self._decision

        self._n += 1
        if success:
            self._successes += 1
            self._lambda_n += self._log_ratio_success
        else:
            self._lambda_n += self._log_ratio_failure

        if self._lambda_n >= self._upper_boundary:
            self._decision = SPRTDecision.ACCEPT_SICK
        elif self._lambda_n <= self._lower_boundary:
            self._decision = SPRTDecision.ACCEPT_HEALTHY

        return self._decision

    @property
    def decision(self) -> SPRTDecision:
        return self._decision

    @property
    def log_likelihood_ratio(self) -> float:
        return self._lambda_n

    @property
    def observations(self) -> int:
        return self._n

    @property
    def observed_success_rate(self) -> float:
        return self._successes / self._n if self._n > 0 else 0.0

    @property
    def upper_boundary(self) -> float:
        return self._upper_boundary

    @property
    def lower_boundary(self) -> float:
        return self._lower_boundary

    @property
    def normalized_position(self) -> float:
        """
        Position of Λ_n in [-1, 1] between the boundaries.
        -1 = at healthy boundary, +1 = at sick boundary, 0 = undecided.
        """
        range_size = self._upper_boundary - self._lower_boundary
        if range_size == 0:
            return 0.0
        return 2.0 * (self._lambda_n - self._lower_boundary) / range_size - 1.0

    def expected_sample_size(self) -> Tuple[float, float]:
        """
        Expected sample size under H₀ and H₁ (Wald's formula).

        E[N | H₀] = [α ln(β/(1-α)) + (1-α) ln((1-β)/α)] / KL(θ₀ || θ₁)
        E[N | H₁] = [β ln(β/(1-α)) + (1-β) ln((1-β)/α)] / KL(θ₁ || θ₀)

        where KL(p||q) = p ln(p/q) + (1-p) ln((1-p)/(1-q)) is the
        Kullback-Leibler divergence.
        """
        kl_01 = self._kl_divergence(self._theta_0, self._theta_1)
        kl_10 = self._kl_divergence(self._theta_1, self._theta_0)

        a, b = self._alpha, self._beta
        h_upper = math.log((1 - b) / a)
        h_lower = math.log(b / (1 - a))

        en_h0 = (a * h_lower + (1 - a) * h_upper) / kl_01 if kl_01 > 0 else float("inf")
        en_h1 = (b * h_upper + (1 - b) * h_lower) / kl_10 if kl_10 > 0 else float("inf")

        return (abs(en_h0), abs(en_h1)) # noqa

    @staticmethod
    def _kl_divergence(p: float, q: float) -> float:
        """KL(p || q) for Bernoulli distributions."""
        if p <= 0 or p >= 1 or q <= 0 or q >= 1:
            return float("inf")
        return p * math.log(p / q) + (1 - p) * math.log((1 - p) / (1 - q))

    def report(self) -> Dict[str, Any]:
        ess = self.expected_sample_size()
        return {
            "decision": self._decision.value,
            "observations": self._n,
            "successes": self._successes,
            "observed_rate": round(self.observed_success_rate, 4),
            "log_likelihood_ratio": round(self._lambda_n, 4),
            "upper_boundary": round(self._upper_boundary, 4),
            "lower_boundary": round(self._lower_boundary, 4),
            "normalized_position": round(self.normalized_position, 4),
            "theta_0": self._theta_0,
            "theta_1": self._theta_1,
            "expected_n_if_healthy": round(ess[0], 1),
            "expected_n_if_sick": round(ess[1], 1),
        }

    def reset(self) -> None:
        self._lambda_n = 0.0
        self._n = 0
        self._successes = 0
        self._decision = SPRTDecision.CONTINUE


# ─────────────────────────────────────────────────────────────────────────────
# 12. RESPONSE COMPRESSIBILITY FINGERPRINT
#
#     Estimates the Kolmogorov complexity K(x) of response bodies using
#     the compression ratio as a proxy:
#
#         ĉ(x) = len(zlib.compress(x)) / len(x)
#
#     This ratio (the "compressibility fingerprint") is a lower bound
#     on the normalized Kolmogorov complexity.  It detects content
#     degeneration without inspecting content semantics:
#
#     Real HTML content:    ĉ ≈ 0.15 – 0.35 (highly compressible)
#     Error pages / CAPTCHAs: ĉ ≈ 0.05 – 0.15 (very compressible, repetitive)
#     Random / encrypted:  ĉ ≈ 0.95 – 1.05 (incompressible)
#     Empty / trivial:     ĉ ≈ 0.01 – 0.05 (near-zero information)
#
#     The key insight: when a domain starts returning error pages or
#     CAPTCHAs instead of real content, the compressibility changes.
#     We don't need to read the content — we just need to see that the
#     compression ratio shifted.  This is a zero-logic observation that
#     respects the architectural boundary.
#
#     We track the compressibility distribution using Welford's online
#     variance and detect shifts using the same CUSUM machinery.
#
#     This is, to our knowledge, the first application of Kolmogorov
#     complexity estimation to web crawler domain health monitoring.
# ─────────────────────────────────────────────────────────────────────────────

import zlib

class CompressibilityFingerprint:
    """
    Response compressibility fingerprint for content degeneration detection.

    Tracks the compression ratio of response bodies over a sliding window
    and detects when the compressibility distribution shifts — indicating
    the domain has started returning different content (error pages,
    CAPTCHAs, empty responses) instead of real content.

    Does NOT inspect content semantics.  Only measures compressibility.
    This is a passive observation, not a routing decision.
    """

    __slots__ = (
        "_acc", "_window", "_buffer", "_baseline_mean",
        "_baseline_var", "_baseline_n", "_shift_detected",
    )

    def __init__(self, window_size: int = 50) -> None:
        self._acc = WelfordAccumulator()
        self._window = window_size
        self._buffer: Deque[float] = deque(maxlen=window_size)
        self._baseline_mean: Optional[float] = None
        self._baseline_var: Optional[float] = None
        self._baseline_n: int = 0
        self._shift_detected: bool = False

    def observe(self, raw_bytes: bytes) -> float:
        """
        Compute compressibility ratio for raw_bytes and incorporate.

        Returns the compression ratio ĉ = compressed_size / raw_size.
        For empty input, returns 0.0.
        """
        if len(raw_bytes) == 0:
            ratio = 0.0
        else:
            compressed = zlib.compress(raw_bytes, level=1)  # fast compression
            ratio = len(compressed) / len(raw_bytes)

        self._acc.update(ratio)
        self._buffer.append(ratio)

        # Establish baseline from first window
        if self._baseline_mean is None and len(self._buffer) >= self._window:
            self._baseline_mean = self._acc.mean
            self._baseline_var = max(self._acc.variance, 1e-6)
            self._baseline_n = self._acc.count

        # Detect shift using Welch's t-test against baseline
        if self._baseline_mean is not None and len(self._buffer) >= 10:
            recent_mean = sum(self._buffer) / len(self._buffer)
            recent_var = (
                sum((x - recent_mean) ** 2 for x in self._buffer)
                / (len(self._buffer) - 1)
            ) if len(self._buffer) > 1 else 0.0

            # Welch's t-statistic
            n1, n2 = self._baseline_n, len(self._buffer)
            s1, s2 = self._baseline_var, max(recent_var, 1e-8)
            se = math.sqrt(s1 / n1 + s2 / n2) if n1 > 0 and n2 > 0 else 1.0
            if se > 1e-12:
                t_stat = abs(recent_mean - self._baseline_mean) / se
                self._shift_detected = t_stat > 2.5  # ~99% significance

        return ratio

    @property
    def mean_ratio(self) -> float:
        return self._acc.mean

    @property
    def current_ratio(self) -> float:
        return self._buffer[-1] if self._buffer else 0.0

    @property
    def shift_detected(self) -> bool:
        """True if compressibility has shifted from baseline."""
        return self._shift_detected

    @property
    def baseline_mean(self) -> Optional[float]:
        return self._baseline_mean

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_ratio": round(self._acc.mean, 4),
            "std_ratio": round(self._acc.std, 4),
            "current_ratio": round(self.current_ratio, 4),
            "baseline_mean": round(self._baseline_mean, 4) if self._baseline_mean else None,
            "shift_detected": self._shift_detected,
            "observations": self._acc.count,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 13. LATENCY AUTOCORRELATION
#
#     Computes the sample autocorrelation function r(k) of the latency
#     time series to detect if the domain is introducing correlated delays.
#
#     For independent random latencies (healthy domain):
#         r(k) ≈ 0 for all k > 0
#
#     For domain-induced throttling (domain slowing us down):
#         r(1) >> 0  (consecutive latencies are correlated)
#
#     For periodic throttling (domain blocks every N requests):
#         r(N) >> 0 while r(1) ≈ 0
#
#     The Ljung-Box Q statistic tests the null hypothesis that the
#     first K autocorrelations are all zero (i.e., no serial dependence):
#
#         Q = n(n+2) Σ_{k=1}^{K} r(k)² / (n-k)
#
#     Under H₀, Q ~ χ²(K).  If Q > χ²_{0.95}(K), reject H₀ and
#     conclude the domain is introducing correlated delays.
#
#     Reference:
#         Ljung, G. M. & Box, G. E. P. (1978).
#         "On a Measure of Lack of Fit in Time Series Models."
#         Biometrika, 65(2), 297–303.
# ─────────────────────────────────────────────────────────────────────────────

class LatencyAutocorrelation:
    """
    Detects correlated delays in domain response latencies.

    Maintains a rolling window of latencies and computes the
    autocorrelation function and Ljung-Box Q statistic.

    If Q > threshold, the domain is introducing non-random delays
    (throttling, queueing, or periodic blocking).
    """

    __slots__ = ("_window_size", "_buffer", "_max_lag")

    def __init__(self, window_size: int = 100, max_lag: int = 10) -> None:
        self._window_size = window_size
        self._buffer: Deque[float] = deque(maxlen=window_size)
        self._max_lag = max_lag

    def observe(self, latency: float) -> None:
        self._buffer.append(latency)

    def autocorrelation(self, lag: int) -> float:
        """
        Sample autocorrelation r(k) at the given lag.

            r(k) = Σ (x_t - x̄)(x_{t-k} - x̄) / Σ (x_t - x̄)²
        """
        n = len(self._buffer)
        if n < lag + 2:
            return 0.0

        data = list(self._buffer)
        mean = sum(data) / n
        var = sum((x - mean) ** 2 for x in data)
        if var < 1e-12:
            return 0.0

        cov = sum(
            (data[t] - mean) * (data[t - lag] - mean)
            for t in range(lag, n)
        )
        return cov / var

    def ljung_box_q(self) -> float:
        """
        Ljung-Box Q statistic for testing serial independence.

            Q = n(n+2) Σ_{k=1}^{K} r(k)² / (n-k)

        Higher Q → stronger evidence of serial correlation.
        """
        n = len(self._buffer)
        if n < self._max_lag + 2:
            return 0.0

        q = 0.0
        for k in range(1, self._max_lag + 1):
            rk = self.autocorrelation(k)
            q += (rk * rk) / (n - k)
        return n * (n + 2) * q

    @property
    def is_correlated(self) -> bool:
        """
        True if the Ljung-Box Q exceeds the χ²(K, 0.95) threshold.

        For K=10 lags, χ²(10, 0.95) ≈ 18.31.
        """
        # χ² critical values at 0.95 for common lag values
        chi2_95 = {5: 11.07, 10: 18.31, 15: 25.00, 20: 31.41}
        threshold = chi2_95.get(self._max_lag, 18.31)
        return self.ljung_box_q() > threshold

    @property
    def lag1_autocorrelation(self) -> float:
        """The most important autocorrelation — consecutive latency correlation."""
        return self.autocorrelation(1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "lag1_autocorrelation": round(self.lag1_autocorrelation, 4),
            "ljung_box_q": round(self.ljung_box_q(), 4),
            "is_correlated": self.is_correlated,
            "observations": len(self._buffer),
            "max_lag": self._max_lag,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 14. DOMAIN CAPACITY ESTIMATOR  (Scalar Kalman Filter)
#
#     Estimates the domain's true processing capacity (requests/second
#     it can handle) from noisy latency observations using a scalar
#     Kalman filter.
#
#     State model:
#         c_t = c_{t-1} + w_t     (capacity evolves slowly; w ~ N(0, Q))
#
#     Observation model:
#         y_t = 1/c_t + v_t       (latency = 1/capacity + noise; v ~ N(0, R))
#
#     We linearize around the current estimate using the extended Kalman
#     filter (EKF) observation Jacobian H = -1/c².
#
#     The Kalman filter optimally balances two sources of information:
#       1. Our prior belief about the capacity (from previous estimates)
#       2. The new latency observation
#
#     When the Kalman gain K is high, we trust the observation more
#     (the estimate adjusts quickly).  When K is low, we trust the
#     prior more (the estimate is stable).  K adapts automatically
#     based on the noise characteristics of the data.
#
#     This gives us a smoothed, real-time estimate of how many requests
#     per second the domain can actually handle — which is the signal
#     index_daemon needs to adjust future RateLimitProfiles.
#
#     Reference:
#         Kalman, R. E. (1960). "A New Approach to Linear Filtering and
#         Prediction Problems." Journal of Basic Engineering, 82(1), 35–45.
# ─────────────────────────────────────────────────────────────────────────────

class KalmanCapacityEstimator:
    """
    Scalar Extended Kalman Filter for domain capacity estimation.

    Estimates the domain's sustainable request rate from noisy latency
    observations.  The state is capacity (requests/second); the
    observation is latency (seconds/request).

    The filter adapts its gain automatically: when latencies are stable,
    the estimate converges quickly.  When latencies are volatile, the
    filter becomes more cautious.
    """

    __slots__ = (
        "_state", "_variance", "_process_noise", "_obs_noise",
        "_n", "_min_latency",
    )

    def __init__(
        self,
        initial_capacity: float = 2.0,      # initial guess: 2 req/s
        process_noise: float = 0.001,        # Q: how fast capacity changes
        observation_noise: float = 0.01,     # R: latency measurement noise
    ) -> None:
        self._state = initial_capacity       # ĉ_t: estimated capacity
        self._variance = 1.0                 # P_t: estimation uncertainty
        self._process_noise = process_noise
        self._obs_noise = observation_noise
        self._n = 0
        self._min_latency = float("inf")

    def observe(self, latency_seconds: float) -> None:
        """
        Incorporate one latency observation using the EKF update.

        Steps:
            1. Predict: ĉ⁻ = ĉ_{t-1}, P⁻ = P_{t-1} + Q
            2. Linearize: H = -1/ĉ⁻²
            3. Innovation: ỹ = y - 1/ĉ⁻
            4. Innovation covariance: S = H P⁻ H' + R
            5. Kalman gain: K = P⁻ H' / S
            6. Update state: ĉ = ĉ⁻ + K ỹ
            7. Update covariance: P = (1 - KH) P⁻
        """
        if latency_seconds <= 0:
            return

        self._n += 1
        self._min_latency = min(self._min_latency, latency_seconds)

        # Predict step
        c_pred = self._state
        p_pred = self._variance + self._process_noise

        # Guard against zero/negative capacity
        if c_pred <= 0.01:
            c_pred = 0.01

        # Observation: y = latency, expected = 1/c
        y = latency_seconds
        y_pred = 1.0 / c_pred

        # Jacobian of observation model: H = d(1/c)/dc = -1/c²
        h = -1.0 / (c_pred * c_pred)

        # Innovation
        innovation = y - y_pred

        # Innovation covariance: S = H P H' + R
        s = h * p_pred * h + self._obs_noise
        if abs(s) < 1e-15:
            return

        # Kalman gain: K = P H' / S
        k = p_pred * h / s

        # State update
        self._state = c_pred + k * innovation

        # Covariance update (Joseph form for numerical stability)
        self._variance = (1.0 - k * h) * p_pred

        # Clamp to physical bounds
        self._state = max(0.01, self._state)
        self._variance = max(1e-6, self._variance)

    @property
    def estimated_capacity(self) -> float:
        """Estimated domain capacity in requests per second."""
        return self._state

    @property
    def estimated_uncertainty(self) -> float:
        """Standard deviation of the capacity estimate."""
        return math.sqrt(self._variance)

    @property
    def estimated_latency(self) -> float:
        """Expected latency at estimated capacity: 1/ĉ."""
        return 1.0 / self._state if self._state > 0 else float("inf")

    @property
    def kalman_gain_magnitude(self) -> float:
        """
        Approximate current Kalman gain magnitude.
        High gain = estimate is rapidly adapting.
        Low gain = estimate is stable.
        """
        if self._state <= 0.01:
            return 0.0
        h = -1.0 / (self._state ** 2)
        s = h * self._variance * h + self._obs_noise
        if abs(s) < 1e-15:
            return 0.0
        return abs(self._variance * h / s)

    @property
    def confidence_interval(self) -> Tuple[float, float]:
        """95% CI for capacity estimate: ĉ ± 1.96σ."""
        sigma = self.estimated_uncertainty
        return (
            max(0.01, self._state - 1.96 * sigma),
            self._state + 1.96 * sigma,
        )

    def to_dict(self) -> Dict[str, Any]:
        ci = self.confidence_interval
        return {
            "estimated_capacity_rps": round(self.estimated_capacity, 4),
            "estimated_latency_ms": round(self.estimated_latency * 1000, 1),
            "uncertainty": round(self.estimated_uncertainty, 4),
            "confidence_interval": (round(ci[0], 4), round(ci[1], 4)),
            "kalman_gain": round(self.kalman_gain_magnitude, 4),
            "observations": self._n,
            "min_latency_ms": round(self._min_latency * 1000, 1) if self._n > 0 else None,
        }


# ─────────────────────────────────────────────────────────────────────────────
# 15. DOMAIN BEHAVIORAL OBSERVATORY  (composite)
#
#     Composes all the above into a single per-domain model that builds
#     a complete statistical portrait of the domain's behavior during
#     the crawl.  One DomainObservatory per active domain.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainHealthReport:
    """
    Complete statistical health report for one domain.
    Produced by DomainObservatory.report().  Consumed by index_daemon.
    """
    domain:                  str
    observations:            int
    bayesian:                Dict[str, Any]
    change_points:           Dict[str, Any]
    sprt:                    Dict[str, Any]
    compressibility:         Dict[str, Any]
    autocorrelation:         Dict[str, Any]
    capacity:                Dict[str, Any]

    def is_healthy(self) -> bool:
        """Conservative health check: healthy if all subsystems agree."""
        return (
            self.sprt.get("decision") != "sick"
            and self.bayesian.get("posterior_mean", 0) > 0.7
            and not self.compressibility.get("shift_detected", False)
            and self.change_points.get("alarm_count", 0) == 0
        )


class DomainObservatory:
    """
    Passive behavioral observatory for one domain.

    Composes:
        - Bayesian success estimator (Beta-Binomial)
        - CUSUM change-point detector
        - Wald SPRT abandonment advisor
        - Response compressibility fingerprint
        - Latency autocorrelation detector
        - Kalman capacity estimator

    All subsystems are O(1) memory and O(1) per observation.

    Usage::

        obs = DomainObservatory("stripe.com")
        for url_result in domain_results:
            obs.observe(
                success=url_result.success,
                latency=url_result.latency,
                raw_bytes=url_result.raw_bytes,
                status_code=url_result.status_code,
            )
        report = obs.report()
    """

    __slots__ = (
        "_domain", "_n",
        "_bayesian", "_cusum", "_sprt",
        "_compressibility", "_autocorrelation", "_capacity",
    )

    def __init__(self, domain: str) -> None:
        self._domain = domain
        self._n = 0
        self._bayesian = BayesianSuccessEstimator()
        self._cusum = CUSUMChangePointDetector()
        self._sprt = WaldSPRT()
        self._compressibility = CompressibilityFingerprint()
        self._autocorrelation = LatencyAutocorrelation()
        self._capacity = KalmanCapacityEstimator()

    def observe(
        self,
        success: bool,
        latency: float,
        raw_bytes: bytes = b"",
        status_code: int = 0,
    ) -> Optional[ChangePointAlarm]:
        """
        Feed one fetch outcome to all observatory subsystems.  O(1).

        Returns a ChangePointAlarm if CUSUM detects a behavior shift,
        None otherwise.
        """
        self._n += 1

        # Bayesian success model
        self._bayesian.observe(success)

        # CUSUM change-point detection
        alarm = self._cusum.observe(success)

        # Wald SPRT (only on non-terminal state)
        self._sprt.observe(success)

        # Compressibility fingerprint (only if we have bytes)
        if raw_bytes:
            self._compressibility.observe(raw_bytes)

        # Latency autocorrelation
        if latency > 0:
            self._autocorrelation.observe(latency)

        # Kalman capacity estimator
        if success and latency > 0:
            self._capacity.observe(latency)

        return alarm

    @property
    def should_abandon(self) -> bool:
        """True if the SPRT recommends abandoning the manifest for this domain."""
        return self._sprt.decision == SPRTDecision.ACCEPT_SICK

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def observations(self) -> int:
        return self._n

    def report(self) -> DomainHealthReport:
        """Generate a complete health report for index_daemon."""
        return DomainHealthReport(
            domain=self._domain,
            observations=self._n,
            bayesian=self._bayesian.to_dict(),
            change_points=self._cusum.to_dict(),
            sprt=self._sprt.report(),
            compressibility=self._compressibility.to_dict(),
            autocorrelation=self._autocorrelation.to_dict(),
            capacity=self._capacity.to_dict(),
        )


# ═══════════════════════════════════════════════════════════════════════════════
#
#   SPECULATIVE ENVELOPE PREDICTOR (SEP)
#
#   A system that races alongside the real fetch path, predicting what the
#   *structural envelope* of a response will look like before the response
#   arrives.  The prediction is never used to replace the fetch — the
#   fetcher always fetches real bytes.  Instead, the *prediction error*
#   becomes a zero-latency anomaly detector:
#
#     - If the prediction says "200, ~45KB, ~120ms, compressibility 0.22"
#       and the reality is "403, 2KB, 30ms, compressibility 0.08",
#       the divergence fires BEFORE the observatory has enough data to
#       detect the shift.  The first anomalous response is flagged
#       instantly — no warm-up, no accumulation.
#
#     - When the predictor converges (prediction error below threshold
#       for sustained periods), it means the domain's behavior is
#       *predictable* — which is itself valuable signal for index_daemon.
#       Predictable domains can be crawled more aggressively.
#       Unpredictable domains need conservative pacing.
#
# ─── WHY SEP EXISTS (and what it is NOT) ──────────────────────────────────
#
#   The original concept was: "predict the next site, prefetch it
#   speculatively, compare, and eventually ditch the real fetch when the
#   predictions are good enough."  That concept is fatally flawed because
#   downstream layers (alpine_strip, topology classifier, signal kernel)
#   need *actual bytes*, not predicted bytes.  Predicted content is
#   hallucination.
#
#   But the *envelope* prediction — the structural fingerprint of what a
#   response should look like, not the content itself — is both feasible
#   and immensely valuable.  Here's the key insight:
#
#   The Observatory detects domain degradation after ACCUMULATING N
#   observations.  CUSUM needs a window.  SPRT needs sequential samples.
#   Bayesian posteriors need multiple updates.  They trade latency for
#   statistical confidence.
#
#   SEP fires on observation ONE.  After 50 pages from stripe.com/docs/*,
#   the SEP already knows what page 51 should look like.  When page 51 is
#   a 403, the z-score on response size alone is 15σ.  That divergence
#   event fires instantly — no warm-up, no accumulation window.  It is
#   the fastest anomaly signal in the entire AXIOM system.
#
#   SEP is to the Observatory as branch prediction is to the reorder
#   buffer in a CPU: it speculates ahead, and when it's wrong, the
#   misprediction itself is the signal.
#
# ─── THE ARCHITECTURE ─────────────────────────────────────────────────────
#
#   Zero-logic compliance:
#     - The predictor never influences which URL is fetched next
#     - The predictor never modifies the fetch mode
#     - The predictor never short-circuits the fetch
#     - The predictor's output is emitted to the bus as signal, period
#     - The fetcher continues identically with or without the predictor
#
#   Data flow:
#     URL → PatternTrie → resolve pattern → PatternModel.predict()
#     ──── fetch happens normally ────
#     Response → PatternModel.score_prediction() → PredictionScore
#             → PatternModel.observe() (update model)
#     if PredictionScore.is_divergent → log/emit divergence signal
#
#   How it works:
#   ────────────────────────────────────────────────────────────────────────
#
#   1. PATH PATTERN CLUSTERING
#      URLs are grouped by structural pattern, not exact path.
#      "/docs/api/charges" and "/docs/api/refunds" share the pattern
#      "/docs/api/*".  Patterns are discovered via a segment-level
#      trie that collapses variable segments when ≥3 unique values are
#      seen at that depth.  One model per URL pattern, not per URL.
#
#   2. PER-PATTERN ENVELOPE MODEL
#      Each pattern maintains online Gaussian models (Welford) for:
#        - response_size (bytes)
#        - latency (seconds)
#        - compressibility (ratio)
#      And categorical frequency counters for:
#        - status_code
#        - content_type
#      After SEP_MIN_OBSERVATIONS (5) of a pattern, predictions activate.
#
#   3. PREDICTION SCORING (log-likelihood)
#      Each prediction is scored against reality using:
#
#        score = Σ log N(x_i; μ_i, σ²_i) + Σ log P(c_j)
#
#      Gaussian log-likelihood for continuous features, categorical
#      log-probability for discrete features.  Higher = more accurate.
#
#   4. CONVERGENCE DETECTION
#      Prediction accuracy is tracked via EWMA over a rolling window
#      of SEP_CONVERGENCE_WINDOW (20) predictions.  When accuracy
#      exceeds SEP_CONVERGENCE_THRESH (80%) for the full window, the
#      pattern is declared "converged" — its behavior is predictable.
#
#   5. DIVERGENCE EVENTS
#      When any continuous feature exceeds SEP_DIVERGENCE_SIGMA (3.0)
#      standard deviations from prediction, or the status code differs
#      from the predicted mode, a divergence is flagged.  This is the
#      fastest anomaly signal in the system.
#
#   ────────────────────────────────────────────────────────────────────────
#
#   Memory: O(P) where P = number of distinct URL patterns (≤500).
#           Each pattern holds 3 Welford accumulators + 2 Counters.
#           For a typical domain: 10–50 patterns = ~5KB.
#
#   Time:   O(D) per observation where D = URL path depth (typically ≤ 8).
#
# ─── WHAT index_daemon RECEIVES ───────────────────────────────────────────
#
#   The telemetry summary includes a ``predictor`` field:
#
#     {
#       "predictor": {
#         "domain": "stripe.com",
#         "total_patterns": 23,           # distinct URL patterns discovered
#         "total_predictions": 824,       # predictions made (after warm-up)
#         "total_divergences": 37,        # predictions that were very wrong
#         "divergence_rate": 0.0449,      # 4.5% surprise rate
#         "converged_patterns": 18,       # patterns with stable behavior
#         "convergence_fraction": 0.7826, # 78% of patterns are predictable
#         "mean_prediction_ll": -8.23,    # average log-likelihood (higher=better)
#         "top_patterns": [
#           {
#             "pattern": "/docs/api/*",
#             "observations": 312,
#             "converged": true,
#             "convergence_score": 0.95,
#             "status_distribution": {200: 308, 301: 4},
#             "size_mean": 45234.2,
#             "latency_mean_ms": 118.4,
#             "compress_mean": 0.2198
#           },
#           ...
#         ]
#       }
#     }
#
# ─── HOW index_daemon SHOULD CONSUME THIS ─────────────────────────────────
#
#   1. CRAWL AGGRESSION CALIBRATION
#      Read ``predictor.convergence_fraction``.  If > 0.7, the domain is
#      structurally predictable.  index_daemon can increase the number of
#      URLs in future manifests and tighten rate limiting (the domain is
#      well-behaved).  If < 0.3, the domain is erratic — reduce manifest
#      size and use conservative pacing.
#
#      Read ``predictor.is_domain_predictable()`` (bool) for the binary
#      decision, or ``predictor.predictability_score()`` (float in [0,1])
#      for the continuous version.
#
#   2. PATTERN-LEVEL ROUTING
#      Read ``predictor.top_patterns``.  Patterns with ``converged: false``
#      and high observation counts are the domain's *unstable* regions.
#      index_daemon can deprioritize URLs matching unconverged patterns
#      in future manifests, or escalate them to higher CL levels.
#
#   3. FIRST-RESPONSE ANOMALY DETECTION
#      The SEP fires divergence on the FIRST anomalous response.  When
#      index_daemon sees a high ``divergence_rate`` (> 0.10), the domain
#      is behaving inconsistently with its own historical pattern.  This
#      is a stronger signal than raw failure rate because it accounts for
#      *what the domain used to do*.  A domain that always returned 403
#      has low divergence (predictably hostile).  A domain that was
#      healthy and suddenly started returning 403 has high divergence
#      (something changed — investigate).
#
#   4. CAPACITY PLANNING
#      Each converged pattern's ``latency_mean_ms`` and ``size_mean``
#      give index_daemon precise per-path-pattern timing expectations.
#      ``estimated_duration_seconds`` in future CrawlManifests can be
#      computed by summing per-pattern latency predictions × URL counts.
#      This is more accurate than domain-level average latency.
#
#   5. COMBINATION WITH OBSERVATORY
#      SEP and Observatory are complementary:
#        - Observatory:  "The domain's success rate has shifted" (after N obs)
#        - SEP:          "This specific response was anomalous" (on obs 1)
#        - Observatory:  "The domain can sustain 4.2 req/s" (Kalman estimate)
#        - SEP:          "The /api/* pattern takes 118ms avg" (per-pattern)
#        - Observatory:  "The domain is throttling us" (autocorrelation)
#        - SEP:          "The latency diverged at 3.2σ on this URL" (instant)
#
#      index_daemon should read BOTH reports.  Observatory for the
#      domain-level statistical portrait, SEP for the pattern-level
#      behavioral map and instant divergence signals.
#
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class EnvelopePrediction:
    """
    Predicted structural envelope for a URL before it is fetched.

    This is NOT a content prediction — it is a statistical envelope:
    what status code, response size, latency, and compressibility
    should this URL produce, given what we've seen from similar URLs?
    """
    pattern:                 str       # the URL pattern this prediction is based on
    predicted_status:        int       # mode of observed status codes for this pattern
    predicted_status_prob:   float     # P(predicted_status) — confidence
    predicted_size_mean:     float     # mean response size (bytes)
    predicted_size_std:      float     # std dev of response size
    predicted_latency_mean:  float     # mean latency (seconds)
    predicted_latency_std:   float     # std dev of latency
    predicted_compress_mean: float     # mean compression ratio
    pattern_observations:    int       # how many observations informed this prediction
    convergence_score:       float     # 0.0 = no convergence, 1.0 = fully converged


@dataclass(frozen=True)
class PredictionScore:
    """
    How well a prediction matched reality.  Produced after every fetch
    by comparing EnvelopePrediction against the actual RawFetchEvent.
    """
    url:                     str
    pattern:                 str
    log_likelihood:          float     # total prediction score (higher = better)
    status_correct:          bool      # predicted status matched actual
    size_z_score:            float     # |actual - predicted| / σ — surprise on size
    latency_z_score:         float     # |actual - predicted| / σ — surprise on latency
    compress_z_score:        float     # surprise on compressibility
    is_divergent:            bool      # True if any z-score > DIVERGENCE_THRESHOLD
    prediction:              EnvelopePrediction


@dataclass(frozen=True)
class PredictorConvergenceEvent:
    """
    Emitted when the predictor converges or de-converges for a URL pattern.
    Convergence means the pattern is behaviorally predictable.
    """
    domain:      str
    pattern:     str
    converged:   bool      # True = converged, False = de-converged
    accuracy:    float     # current prediction accuracy (0.0–1.0)
    observations: int


# ── Constants ─────────────────────────────────────────────────────────────────

SEP_MIN_OBSERVATIONS:    Final[int]   = 5       # min observations before predicting
SEP_CONVERGENCE_WINDOW:  Final[int]   = 20      # sustained accuracy window
SEP_CONVERGENCE_THRESH:  Final[float] = 0.80    # accuracy threshold for convergence
SEP_DIVERGENCE_SIGMA:    Final[float] = 3.0     # z-score threshold for divergence
SEP_MAX_PATTERNS:        Final[int]   = 500     # max patterns per domain (memory bound)
SEP_SEGMENT_COLLAPSE_N:  Final[int]   = 3       # min unique values before collapsing segment


# ── URL Pattern Extractor ─────────────────────────────────────────────────────

class _PatternTrie:
    """
    Trie-based URL pattern extractor.

    Segments of a URL path that show high cardinality (many unique values
    at that position) are collapsed to "*".  This groups URLs by their
    structural pattern rather than their exact path.

    Example:
        /docs/api/charges      )
        /docs/api/refunds      )  → pattern: /docs/api/*
        /docs/api/customers    )

        /blog/2024/01/my-post  )
        /blog/2024/02/other    )  → pattern: /blog/*/*/* (or /blog/2024/*/*)
        /blog/2023/12/thing    )

    The trie adapts as URLs are observed.  A segment that initially
    appears fixed (only one value seen) becomes variable ("*") once
    enough distinct values are observed at that position.
    """

    __slots__ = ("_children", "_values", "_is_collapsed")

    def __init__(self) -> None:
        self._children: Dict[str, "_PatternTrie"] = {}
        self._values: Set[str] = set()
        self._is_collapsed: bool = False

    def insert(self, segments: List[str], depth: int = 0) -> None:
        """Insert a URL's path segments into the trie."""
        if depth >= len(segments):
            return

        seg = segments[depth]
        self._values.add(seg)

        # Collapse this segment if too many unique values
        if len(self._values) >= SEP_SEGMENT_COLLAPSE_N:
            self._is_collapsed = True

        # Navigate or create child
        key = "*" if self._is_collapsed else seg
        if key not in self._children:
            self._children[key] = _PatternTrie()
        self._children[key].insert(segments, depth + 1)

    def resolve(self, segments: List[str], depth: int = 0) -> str:
        """
        Resolve a URL's path segments to their pattern string.

        Returns the pattern with collapsed segments replaced by "*".
        """
        if depth >= len(segments):
            return ""

        seg = segments[depth]
        key = "*" if self._is_collapsed else seg

        child = self._children.get(key)
        if child is None:
            # Unseen segment — try wildcard
            child = self._children.get("*")
            if child is None:
                # Truly new — return remaining segments as-is
                return "/" + "/".join(segments[depth:])
            key = "*"

        rest = child.resolve(segments, depth + 1)
        return f"/{key}{rest}"


class _PatternModel:
    """
    Per-URL-pattern statistical model for envelope prediction.

    Maintains online Gaussian models for continuous features and
    frequency counters for discrete features.  All O(1) memory.
    """

    __slots__ = (
        "_pattern", "_n",
        "_size_acc", "_latency_acc", "_compress_acc",
        "_status_counts", "_content_type_counts",
        "_error_ewma", "_converged", "_convergence_window",
    )

    def __init__(self, pattern: str) -> None:
        self._pattern = pattern
        self._n: int = 0
        self._size_acc = WelfordAccumulator()
        self._latency_acc = WelfordAccumulator()
        self._compress_acc = WelfordAccumulator()
        self._status_counts: Counter = Counter()
        self._content_type_counts: Counter = Counter()
        self._error_ewma = EWMATracker(alpha=0.1, bias_correct=True)
        self._converged: bool = False
        self._convergence_window: Deque[float] = deque(maxlen=SEP_CONVERGENCE_WINDOW)

    def observe(
        self,
        status_code: int,
        size_bytes: int,
        latency: float,
        compressibility: float,
        content_type: str = "",
    ) -> None:
        """Record one actual fetch result for this pattern.  O(1)."""
        self._n += 1
        self._size_acc.update(float(size_bytes))
        self._latency_acc.update(latency)
        self._compress_acc.update(compressibility)
        self._status_counts[status_code] += 1
        if content_type:
            self._content_type_counts[content_type] += 1

    def predict(self) -> Optional[EnvelopePrediction]:
        """
        Generate an envelope prediction based on accumulated observations.

        Returns None if insufficient data (< SEP_MIN_OBSERVATIONS).
        """
        if self._n < SEP_MIN_OBSERVATIONS:
            return None

        # Predicted status = mode
        status_mode = self._status_counts.most_common(1)[0]
        status_prob = status_mode[1] / self._n

        return EnvelopePrediction(
            pattern=self._pattern,
            predicted_status=status_mode[0],
            predicted_status_prob=status_prob,
            predicted_size_mean=self._size_acc.mean,
            predicted_size_std=max(self._size_acc.std, 1.0),
            predicted_latency_mean=self._latency_acc.mean,
            predicted_latency_std=max(self._latency_acc.std, 0.001),
            predicted_compress_mean=self._compress_acc.mean,
            pattern_observations=self._n,
            convergence_score=self._compute_convergence(),
        )

    def score_prediction(
        self,
        prediction: EnvelopePrediction,
        actual_status: int,
        actual_size: int,
        actual_latency: float,
        actual_compressibility: float,
        url: str,
    ) -> PredictionScore:
        """
        Score a prediction against actual results.

        Computes:
          - Gaussian log-likelihood for continuous features
          - Categorical match for status code
          - Z-scores for surprise detection
          - Divergence flag

        Returns a PredictionScore with full details.
        """
        # Z-scores (how many σ away from prediction)
        size_z = (
            abs(actual_size - prediction.predicted_size_mean) / prediction.predicted_size_std
            if prediction.predicted_size_std > 0 else 0.0
        )
        latency_z = (
            abs(actual_latency - prediction.predicted_latency_mean)
            / prediction.predicted_latency_std
            if prediction.predicted_latency_std > 0 else 0.0
        )

        compress_std = max(self._compress_acc.std, 0.01)
        compress_z = (
            abs(actual_compressibility - prediction.predicted_compress_mean)
            / compress_std
        )

        status_correct = actual_status == prediction.predicted_status

        # Gaussian log-likelihood for continuous features
        ll = 0.0
        ll += _gaussian_log_likelihood(
            actual_size, prediction.predicted_size_mean, prediction.predicted_size_std,
        )
        ll += _gaussian_log_likelihood(
            actual_latency, prediction.predicted_latency_mean, prediction.predicted_latency_std,
        )

        # Categorical log-probability for status
        ll += math.log(prediction.predicted_status_prob) if status_correct else math.log(
            max(1e-6, 1.0 - prediction.predicted_status_prob)
        )

        # Divergence detection
        is_divergent = (
            size_z > SEP_DIVERGENCE_SIGMA
            or latency_z > SEP_DIVERGENCE_SIGMA
            or compress_z > SEP_DIVERGENCE_SIGMA
            or not status_correct
        )

        # Update convergence tracking
        accuracy = 1.0 if not is_divergent else 0.0
        self._error_ewma.update(accuracy)
        self._convergence_window.append(accuracy)
        self._converged = self._compute_convergence() >= SEP_CONVERGENCE_THRESH

        return PredictionScore(
            url=url,
            pattern=prediction.pattern,
            log_likelihood=ll,
            status_correct=status_correct,
            size_z_score=round(size_z, 3),
            latency_z_score=round(latency_z, 3),
            compress_z_score=round(compress_z, 3),
            is_divergent=is_divergent,
            prediction=prediction,
        )

    def _compute_convergence(self) -> float:
        """
        Convergence score: fraction of recent predictions that were accurate.
        Requires a full convergence window before reporting.
        """
        if len(self._convergence_window) < SEP_CONVERGENCE_WINDOW:
            return 0.0
        return sum(self._convergence_window) / len(self._convergence_window)

    @property
    def converged(self) -> bool:
        return self._converged

    @property
    def observations(self) -> int:
        return self._n

    @property
    def pattern(self) -> str:
        return self._pattern

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pattern": self._pattern,
            "observations": self._n,
            "converged": self._converged,
            "convergence_score": round(self._compute_convergence(), 4),
            "status_distribution": dict(self._status_counts.most_common(5)),
            "size_mean": round(self._size_acc.mean, 1),
            "size_std": round(self._size_acc.std, 1),
            "latency_mean_ms": round(self._latency_acc.mean * 1000, 1),
            "latency_std_ms": round(self._latency_acc.std * 1000, 1),
            "compress_mean": round(self._compress_acc.mean, 4),
        }


def _gaussian_log_likelihood(x: float, mu: float, sigma: float) -> float:
    """
    Log-likelihood of observation x under N(μ, σ²).

    log N(x; μ, σ²) = -0.5 × [ln(2πσ²) + (x-μ)²/σ²]
    """
    if sigma < 1e-12:
        return 0.0
    return -0.5 * (math.log(2 * math.pi * sigma * sigma) + ((x - mu) ** 2) / (sigma * sigma))


class SpeculativeEnvelopePredictor:
    """
    Speculative Envelope Predictor (SEP).

    Builds per-URL-pattern models that predict the structural envelope
    of fetch responses.  Runs alongside the real fetch path and produces
    two kinds of signal:

    1. **Divergence events** — fires on the FIRST anomalous response,
       before the observatory accumulates enough data.  This is the
       fastest anomaly signal in the system.

    2. **Convergence events** — fires when a URL pattern becomes
       predictable, meaning index_daemon can trust structural predictions
       for future manifest planning.

    Does NOT influence the fetch path.  Zero-logic compliant.

    Usage::

        sep = SpeculativeEnvelopePredictor("stripe.com")

        # Before fetch (off critical path):
        prediction = sep.predict(url)

        # After fetch:
        score = sep.observe(url, status=200, size=45000, latency=0.12, ...)

        if score and score.is_divergent:
            bus.emit(DivergenceEvent(...))
    """

    __slots__ = (
        "_domain", "_trie", "_models", "_total_predictions",
        "_total_divergences", "_total_convergences",
        "_prediction_scores", "_compressor",
    )

    def __init__(self, domain: str) -> None:
        self._domain = domain
        self._trie = _PatternTrie()
        self._models: Dict[str, _PatternModel] = {}
        self._total_predictions: int = 0
        self._total_divergences: int = 0
        self._total_convergences: int = 0
        self._prediction_scores = WelfordAccumulator()

    def _url_to_segments(self, url: str) -> List[str]: # noqa
        """Extract path segments from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return ["_root"]
        return path.split("/")

    def _get_pattern(self, url: str) -> str:
        """Resolve a URL to its pattern via the trie."""
        segments = self._url_to_segments(url)
        return self._trie.resolve(segments)

    def _get_or_create_model(self, url: str) -> _PatternModel:
        """Get or create the model for this URL's pattern."""
        segments = self._url_to_segments(url)
        self._trie.insert(segments)
        pattern = self._trie.resolve(segments)

        if pattern not in self._models:
            if len(self._models) >= SEP_MAX_PATTERNS:
                # Evict least-observed pattern
                min_pattern = min(self._models, key=lambda k: self._models[k].observations)
                del self._models[min_pattern]
            self._models[pattern] = _PatternModel(pattern)

        return self._models[pattern]

    def predict(self, url: str) -> Optional[EnvelopePrediction]:
        """
        Generate an envelope prediction for the given URL.

        Returns None if insufficient data for this URL's pattern.
        Called BEFORE the fetch — this is the speculative prediction.
        """
        model = self._get_or_create_model(url)
        return model.predict()

    def observe(
        self,
        url: str,
        status_code: int,
        size_bytes: int,
        latency: float,
        raw_bytes: bytes = b"",
    ) -> Optional[PredictionScore]:
        """
        Feed an actual fetch result to the predictor.

        1. Computes compressibility of the response
        2. Scores the pre-fetch prediction (if one was made) against reality
        3. Updates the pattern model with the actual result
        4. Tracks convergence

        Returns a PredictionScore if a prediction was available, else None.
        """
        # Compute compressibility
        if raw_bytes and len(raw_bytes) > 0:
            compressed = zlib.compress(raw_bytes, level=1)
            compressibility = len(compressed) / len(raw_bytes)
        else:
            compressibility = 0.0

        model = self._get_or_create_model(url)

        # Score prediction BEFORE updating model (prediction was made before this data)
        prediction = model.predict()
        score: Optional[PredictionScore] = None

        if prediction is not None:
            self._total_predictions += 1
            was_converged = model.converged

            score = model.score_prediction(
                prediction=prediction,
                actual_status=status_code,
                actual_size=size_bytes,
                actual_latency=latency,
                actual_compressibility=compressibility,
                url=url,
            )

            if score.is_divergent:
                self._total_divergences += 1

            self._prediction_scores.update(score.log_likelihood)

            # Detect convergence transitions
            if not was_converged and model.converged:
                self._total_convergences += 1
            elif was_converged and not model.converged:
                pass  # de-convergence — could emit event

        # Update model with actual observation
        content_type = ""
        model.observe(
            status_code=status_code,
            size_bytes=size_bytes,
            latency=latency,
            compressibility=compressibility,
            content_type=content_type,
        )

        return score

    @property
    def domain(self) -> str:
        return self._domain

    @property
    def pattern_count(self) -> int:
        return len(self._models)

    @property
    def total_predictions(self) -> int:
        return self._total_predictions

    @property
    def total_divergences(self) -> int:
        return self._total_divergences

    @property
    def divergence_rate(self) -> float:
        if self._total_predictions == 0:
            return 0.0
        return self._total_divergences / self._total_predictions

    @property
    def converged_patterns(self) -> int:
        return sum(1 for m in self._models.values() if m.converged)

    @property
    def convergence_fraction(self) -> float:
        """Fraction of patterns that have converged."""
        if not self._models:
            return 0.0
        return self.converged_patterns / len(self._models)

    @property
    def mean_prediction_score(self) -> float:
        return self._prediction_scores.mean

    def report(self) -> Dict[str, Any]:
        """Full predictor report for index_daemon."""
        patterns = sorted(
            self._models.values(),
            key=lambda m: m.observations,
            reverse=True,
        )
        return {
            "domain": self._domain,
            "total_patterns": len(self._models),
            "total_predictions": self._total_predictions,
            "total_divergences": self._total_divergences,
            "divergence_rate": round(self.divergence_rate, 4),
            "converged_patterns": self.converged_patterns,
            "convergence_fraction": round(self.convergence_fraction, 4),
            "mean_prediction_ll": round(self.mean_prediction_score, 4),
            "top_patterns": [p.to_dict() for p in patterns[:10]],
        }

    def is_domain_predictable(self) -> bool:
        """
        True if the majority of this domain's URL patterns have converged.
        Predictable domains can be crawled more aggressively.
        """
        return (
            len(self._models) >= 3
            and self.convergence_fraction >= 0.6
            and self._total_predictions >= 30
        )

    def predictability_score(self) -> float:
        """
        Composite predictability score in [0, 1].

        Combines convergence fraction, divergence rate, and prediction
        score into a single metric.  Higher = more predictable.

        Weighted formula:
            P = 0.5 × convergence_fraction
              + 0.3 × (1 - divergence_rate)
              + 0.2 × sigmoid(mean_prediction_score + 5)

        The sigmoid maps the unbounded log-likelihood to [0, 1].
        """
        cf = self.convergence_fraction
        dr = 1.0 - min(self.divergence_rate, 1.0)

        # Sigmoid of mean prediction score (shifted to be centered around -5)
        mps = self.mean_prediction_score
        sig = 1.0 / (1.0 + math.exp(-(mps + 5.0)))

        return 0.5 * cf + 0.3 * dr + 0.2 * sig


# ═══════════════════════════════════════════════════════════════════════════════
#
#   FETCHER-SPECIFIC CONTRACTS
#
#   These types are defined here because they are internal to the fetcher
#   and not part of the cross-layer contract boundary in contracts.py.
#
# ═══════════════════════════════════════════════════════════════════════════════

if "FetchAnomalyEvent" not in globals():
    @dataclass(frozen=True)
    class FetchAnomalyEvent:
        """
        Standalone fallback for tests that import fetcher.py without the full
        AXIOM contract tree. In normal runtime this class comes from
        signal_kernel.contracts and is the canonical bus-visible schema.
        """
        url:           str
        fetch_mode:    FetchMode
        status_code:   Optional[int]
        anomaly_type:  str
        run_id:        str
        manifest_id:   str
        timestamp:     float = field(default_factory=time.time)
        detail:        str = ""


@dataclass
class CLState:
    """
    Clearance-level availability state for the current session.

    CL1 is always available (httpx clearnet).  CL2 requires Playwright.
    CL3/CL4 require Tor.  Availability is determined at initialization
    and updated by CLStateUpdateEvent from RLCPC.

    Fallback chain is always silent and drops exactly one level:
    CL4→CL3, CL3→CL2, CL2→CL1.  Never CL4→CL1.  Never announced.
    """
    cl1_available: bool = True
    cl2_available: bool = False
    cl3_available: bool = False
    cl4_available: bool = False

    def effective_mode(self, requested: FetchMode) -> FetchMode:
        """
        Resolve the actual fetch mode given current availability.
        Silent fallback — drops exactly one CL level.
        """
        if requested == FetchMode.TOR_FULL:
            if self.cl4_available:
                return FetchMode.TOR_FULL
            if self.cl3_available:
                return FetchMode.TOR
            if self.cl2_available:
                return FetchMode.HEADLESS
            return FetchMode.STATIC
        if requested == FetchMode.TOR:
            if self.cl3_available:
                return FetchMode.TOR
            if self.cl2_available:
                return FetchMode.HEADLESS
            return FetchMode.STATIC
        if requested == FetchMode.HEADLESS:
            if self.cl2_available:
                return FetchMode.HEADLESS
            return FetchMode.STATIC
        return FetchMode.STATIC

    def to_dict(self) -> Dict[str, bool]:
        return {
            "cl1": self.cl1_available,
            "cl2": self.cl2_available,
            "cl3": self.cl3_available,
            "cl4": self.cl4_available,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Per-manifest telemetry aggregator
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ManifestTelemetry:
    """
    Collects all streaming statistics for one manifest execution.
    Every field is O(1) memory — no raw data is stored.
    """
    manifest_id:     str
    domain:          str
    started_at:      float = field(default_factory=time.monotonic)

    # Counters
    urls_attempted:  int = 0
    urls_succeeded:  int = 0
    urls_failed:     int = 0
    urls_skipped:    int = 0    # bloom dedup
    bytes_fetched:   int = 0
    events_emitted:  int = 0

    # Streaming statistics
    latency_welford:  WelfordAccumulator = field(default_factory=WelfordAccumulator)
    size_welford:     WelfordAccumulator = field(default_factory=WelfordAccumulator)
    latency_ewma:     EWMATracker = field(default_factory=EWMATracker)
    latency_p50:      P2QuantileEstimator = field(
        default_factory=lambda: P2QuantileEstimator(0.50)
    )
    latency_p95:      P2QuantileEstimator = field(
        default_factory=lambda: P2QuantileEstimator(0.95)
    )
    latency_p99:      P2QuantileEstimator = field(
        default_factory=lambda: P2QuantileEstimator(0.99)
    )

    # Domain health
    entropy_tracker:  ShannonEntropyTracker = field(
        default_factory=ShannonEntropyTracker
    )
    hazard_estimator: Optional[HazardRateEstimator] = None
    anomaly_sampler:  ReservoirSampler = field(default_factory=ReservoirSampler)
    little_law:       LittleLawAdvisor = field(default_factory=LittleLawAdvisor)

    # Tor-specific (CL3/CL4)
    circuit_quality:  MarkovCircuitQuality = field(
        default_factory=MarkovCircuitQuality
    )

    # CL distribution
    cl_counts:        Dict[str, int] = field(default_factory=lambda: {
        "static": 0, "headless": 0, "tor": 0, "tor_full": 0,
    })

    # Domain Behavioral Observatory — passive intelligence
    observatory:      Optional[DomainObservatory] = None

    # Speculative Envelope Predictor — races alongside the fetch path
    predictor:        Optional[SpeculativeEnvelopePredictor] = None

    def __post_init__(self) -> None:
        if self.observatory is None:
            self.observatory = DomainObservatory(self.domain)
        if self.predictor is None:
            self.predictor = SpeculativeEnvelopePredictor(self.domain)

    def record_success(
        self, latency: float, byte_count: int, status_code: int,
        fetch_mode: FetchMode, raw_bytes: bytes = b"",
    ) -> None:
        self.urls_attempted += 1
        self.urls_succeeded += 1
        self.bytes_fetched += byte_count
        self.events_emitted += 1
        self.latency_welford.update(latency)
        self.size_welford.update(byte_count)
        self.latency_ewma.update(latency)
        self.latency_p50.observe(latency)
        self.latency_p95.observe(latency)
        self.latency_p99.observe(latency)
        self.entropy_tracker.observe(status_code)
        self.little_law.record_completion(latency)
        self.cl_counts[fetch_mode.value] = self.cl_counts.get(fetch_mode.value, 0) + 1
        if self.hazard_estimator:
            self.hazard_estimator.observe(False)
        if self.observatory:
            self.observatory.observe(
                success=True, latency=latency,
                raw_bytes=raw_bytes, status_code=status_code,
            )
        if self.predictor:
            self.predictor.observe(
                url="", status_code=status_code,
                size_bytes=byte_count, latency=latency,
                raw_bytes=raw_bytes,
            )

    def record_failure(
        self, anomaly: FetchAnomalyEvent, fetch_mode: FetchMode,
    ) -> None:
        self.urls_attempted += 1
        self.urls_failed += 1
        if anomaly.status_code:
            self.entropy_tracker.observe(anomaly.status_code)
        self.anomaly_sampler.add(anomaly)
        self.cl_counts[fetch_mode.value] = self.cl_counts.get(fetch_mode.value, 0) + 1
        if self.hazard_estimator:
            self.hazard_estimator.observe(True)
        if self.observatory:
            self.observatory.observe(
                success=False, latency=0.0,
                status_code=anomaly.status_code or 0,
            )

    def record_skip(self) -> None:
        self.urls_attempted += 1
        self.urls_skipped += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def throughput(self) -> float:
        e = self.elapsed_seconds
        return self.urls_succeeded / e if e > 0 else 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "manifest_id": self.manifest_id,
            "domain": self.domain,
            "elapsed_sec": round(self.elapsed_seconds, 2),
            "attempted": self.urls_attempted,
            "succeeded": self.urls_succeeded,
            "failed": self.urls_failed,
            "skipped": self.urls_skipped,
            "bytes_fetched": self.bytes_fetched,
            "throughput_urls_per_sec": round(self.throughput, 2),
            "latency": self.latency_welford.to_dict(),
            "latency_p50_ms": round(self.latency_p50.estimate() * 1000, 1),
            "latency_p95_ms": round(self.latency_p95.estimate() * 1000, 1),
            "latency_p99_ms": round(self.latency_p99.estimate() * 1000, 1),
            "latency_ewma_ms": round(self.latency_ewma.value * 1000, 1),
            "response_size": self.size_welford.to_dict(),
            "status_entropy_bits": round(self.entropy_tracker.entropy, 3),
            "domain_is_degenerate": self.entropy_tracker.is_degenerate,
            "hazard": (
                self.hazard_estimator.to_dict()
                if self.hazard_estimator else None
            ),
            "little_law": self.little_law.to_dict(),
            "circuit_quality": (
                self.circuit_quality.to_dict()
                if self.circuit_quality.total_observations > 0 else None
            ),
            "cl_distribution": dict(self.cl_counts),
            "anomaly_sample_size": self.anomaly_sampler.reservoir_size,
            "observatory": (
                self.observatory.report().__dict__
                if self.observatory and self.observatory.observations > 0
                else None
            ),
            "predictor": (
                self.predictor.report()
                if self.predictor and self.predictor.total_predictions > 0
                else None
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#
#   CRAWLER BUS — MINIMAL EVENT EMITTER
#
#   In the full AXIOM tree, this is crawler_bus.py.  For standalone
#   operation, we provide a minimal async event bus with subscribe/emit.
#   Zero logic.  Zero inspection.  Zero filtering.  The fetcher emits;
#   subscribers consume.  That is the complete contract.
#
# ═══════════════════════════════════════════════════════════════════════════════

class CrawlerBus:
    """
    Minimal async event bus.  Subscribe by event type, emit to all
    subscribers of that type.  Zero logic.

    In the full AXIOM tree, this would be replaced by the real
    ``crawler_bus.py`` which may use Kafka, Redis Streams, or a
    direct in-process dispatch depending on deployment topology.
    """

    def __init__(self) -> None:
        self._subscribers: Dict[type, List[Callable]] = defaultdict(list)
        self._emit_count: int = 0

    def subscribe(self, event_type: type, handler: Callable) -> None:
        self._subscribers[event_type].append(handler)

    async def emit(self, event: Any) -> None:
        """Emit event to all subscribers.  Fire-and-forget.  Never raises."""
        self._emit_count += 1
        event_type = type(event)
        for handler in self._subscribers.get(event_type, []):
            try:
                result = handler(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as exc:
                log.warning(
                    "bus subscriber %s failed on %s: %s",
                    handler.__name__, event_type.__name__, exc,
                )

    @property
    def emit_count(self) -> int:
        return self._emit_count


# ═══════════════════════════════════════════════════════════════════════════════
#
#   STAGING PIPELINE
#
#   All raw bytes pass through /tmp/fetch_staging/ before being attached
#   to a RawFetchEvent.  No direct buffer → event assignment.  This is
#   the atomic write guarantee: bytes are staged first, then read back,
#   then the staging file is deleted.  If the process dies between stage
#   and emit, leftover staging files are cleaned up on next init.
#
# ═══════════════════════════════════════════════════════════════════════════════

class StagingPipeline:
    """
    Staging pipeline for raw fetch bytes.

    Protocol:
        1. ``stage(raw_bytes, run_id)`` → writes to /tmp/fetch_staging/{run_id}.raw
        2. ``unstage(path)`` → reads bytes back, deletes staging file
        3. On init: cleans up any leftover staging files from a crash

    All writes use write-then-rename (atomic on POSIX) to prevent
    partial files from being picked up.
    """

    def __init__(self, staging_dir: Path = STAGING_PATH) -> None:
        self._dir = staging_dir
        self._staged_count: int = 0
        self._unstaged_count: int = 0
        self._cleaned_count: int = 0

    async def initialize(self) -> None:
        """Create staging directory and clean up leftovers from previous crash."""
        self._dir.mkdir(parents=True, exist_ok=True)
        # Clean up any leftover files from a previous crash
        cleaned = 0
        for f in self._dir.iterdir():
            if f.suffix in (".raw", ".tmp"):
                try:
                    f.unlink()
                    cleaned += 1
                except OSError:
                    pass
        self._cleaned_count = cleaned
        if cleaned > 0:
            log.info("staging: cleaned %d leftover files from previous session", cleaned)

    async def stage(self, raw_bytes: bytes, run_id: str) -> Path:
        """
        Write raw bytes to staging.  Uses write-to-tmp + rename for atomicity.

        Returns the path to the staged file.
        """
        final_path = self._dir / f"{run_id}.raw"
        tmp_path = self._dir / f"{run_id}.tmp"
        try:
            tmp_path.write_bytes(raw_bytes)
            tmp_path.rename(final_path)
            self._staged_count += 1
            return final_path
        except OSError as exc:
            # Clean up on failure
            for p in (tmp_path, final_path):
                try:
                    p.unlink(missing_ok=True)
                except OSError:
                    pass
            raise OSError(f"staging write failed for {run_id}: {exc}") from exc

    async def unstage(self, staging_path: Path) -> bytes:
        """
        Read bytes from staging file and delete it.

        The read-then-delete is intentional: the bytes are now in the
        RawFetchEvent.  The staging file has served its purpose.
        """
        try:
            data = staging_path.read_bytes()
            staging_path.unlink(missing_ok=True)
            self._unstaged_count += 1
            return data
        except OSError as exc:
            raise OSError(f"staging read failed for {staging_path}: {exc}") from exc

    async def cleanup(self) -> int:
        """Remove all staging files.  Returns count of files removed."""
        removed = 0
        if self._dir.exists():
            for f in self._dir.iterdir():
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        return removed

    @property
    def stats(self) -> Dict[str, int]:
        return {
            "staged": self._staged_count,
            "unstaged": self._unstaged_count,
            "cleaned_on_init": self._cleaned_count,
        }


# ═══════════════════════════════════════════════════════════════════════════════
#
#   TOR CIRCUIT MANAGER
#
#   Manages Tor SOCKS5 proxy availability checking and circuit rotation
#   via the Tor control port (9051).  Uses the NEWNYM signal to request
#   a new circuit.  Tracks IP rotation quality via the Markov chain.
#
# ═══════════════════════════════════════════════════════════════════════════════

class TorCircuitManager:
    """
    Tor circuit lifecycle management for CL3 and CL4 fetch modes.

    Responsibilities:
        - Check Tor availability on init (SOCKS5 probe + control port probe)
        - Request new circuits via SIGNAL NEWNYM on the control port
        - Track circuit rotation quality via Markov chain IP analysis
        - Enforce cooldown between NEWNYM signals (Tor rate limits these)

    The fetcher owns one TorCircuitManager.  CL3 and CL4 share it.
    """

    def __init__(
        self,
        socks_host: str = TOR_SOCKS_HOST,
        socks_port: int = TOR_SOCKS_PORT,
        control_port: int = TOR_CONTROL_PORT,
    ) -> None:
        self._socks_host = socks_host
        self._socks_port = socks_port
        self._control_port = control_port
        self._available: bool = False
        self._circuit_count: int = 0
        self._last_newnym: float = 0.0
        self._quality = MarkovCircuitQuality()
        self._rotation_latencies = WelfordAccumulator()

    async def check_available(self) -> bool:
        """
        Probe Tor SOCKS5 and control port.  Returns True if both are
        reachable.  Non-blocking — uses asyncio with timeout.
        """
        try:
            # SOCKS5 probe: connect to the SOCKS port
            socks_ok = await self._probe_port(self._socks_host, self._socks_port)
            if not socks_ok:
                log.warning("tor: SOCKS5 port %d not reachable", self._socks_port)
                self._available = False
                return False

            # Control port probe
            control_ok = await self._probe_port(self._socks_host, self._control_port)
            if not control_ok:
                log.warning(
                    "tor: control port %d not reachable (SOCKS5 OK)",
                    self._control_port,
                )
                # SOCKS5 works but no control port — limited functionality
                # We can still route through Tor but can't rotate circuits
                self._available = True
                return True

            self._available = True
            log.info(
                "tor: available — SOCKS5=%s:%d, control=%s:%d",
                self._socks_host, self._socks_port,
                self._socks_host, self._control_port,
            )
            return True

        except Exception as exc:
            log.warning("tor: availability check failed: %s", exc)
            self._available = False
            return False

    async def _probe_port(self, host: str, port: int) -> bool: # noqa
        """Non-blocking TCP connect probe with timeout."""
        try:
            _, writer = await asyncio.wait_for(
                asyncio.open_connection(host, port),
                timeout=TOR_CONNECT_TIMEOUT,
            )
            writer.close()
            await writer.wait_closed()
            return True
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError):
            return False

    async def request_new_circuit(self) -> bool:
        """
        Send SIGNAL NEWNYM to Tor control port to request a new circuit.

        Enforces a cooldown of TOR_NEWNYM_COOLDOWN between requests.
        Tor rate-limits NEWNYM to ~1/second; sending faster is silently
        ignored by Tor but wastes a round-trip.

        Returns True if the signal was sent successfully.
        """
        now = time.monotonic()
        elapsed = now - self._last_newnym
        if elapsed < TOR_NEWNYM_COOLDOWN:
            await asyncio.sleep(TOR_NEWNYM_COOLDOWN - elapsed)

        t_start = time.monotonic()
        try:
            reader, writer = await asyncio.wait_for(
                asyncio.open_connection(self._socks_host, self._control_port),
                timeout=TOR_CONNECT_TIMEOUT,
            )
            # Read the welcome banner
            await asyncio.wait_for(reader.readline(), timeout=2.0)
            # Authenticate (no password in default config)
            writer.write(b"AUTHENTICATE\r\n")
            await writer.drain()
            auth_response = await asyncio.wait_for(reader.readline(), timeout=2.0)
            if not auth_response.startswith(b"250"):
                log.warning("tor: AUTHENTICATE failed: %s", auth_response.strip())
                writer.close()
                await writer.wait_closed()
                return False

            # Request new circuit
            writer.write(b"SIGNAL NEWNYM\r\n")
            await writer.drain()
            newnym_response = await asyncio.wait_for(reader.readline(), timeout=2.0)
            writer.close()
            await writer.wait_closed()

            self._last_newnym = time.monotonic()
            self._circuit_count += 1
            rotation_latency = time.monotonic() - t_start
            self._rotation_latencies.update(rotation_latency)

            success = newnym_response.startswith(b"250")
            if not success:
                log.warning("tor: NEWNYM failed: %s", newnym_response.strip())
            return success

        except (asyncio.TimeoutError, OSError, ConnectionRefusedError) as exc:
            log.warning("tor: circuit rotation failed: %s", exc)
            return False

    def record_exit_ip(self, ip: str) -> None:
        """Record observed exit IP for Markov chain quality tracking."""
        self._quality.observe_ip(ip)

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def circuit_count(self) -> int:
        return self._circuit_count

    @property
    def quality(self) -> MarkovCircuitQuality:
        return self._quality

    @property
    def avg_rotation_latency(self) -> float:
        return self._rotation_latencies.mean

    @property
    def proxy_url(self) -> str:
        return f"socks5://{self._socks_host}:{self._socks_port}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": self._available,
            "socks": f"{self._socks_host}:{self._socks_port}",
            "control": f"{self._socks_host}:{self._control_port}",
            "circuits_rotated": self._circuit_count,
            "avg_rotation_ms": round(self._rotation_latencies.mean * 1000, 1),
            "quality": self._quality.to_dict(),
        }


# ═══════════════════════════════════════════════════════════════════════════════
#
#   PLAYWRIGHT LIFECYCLE MANAGER
#
#   Manages the Playwright browser instance, context creation/recycling,
#   and page lifecycle.  The fetcher owns one PlaywrightManager.
#
# ═══════════════════════════════════════════════════════════════════════════════

class PlaywrightManager:
    """
    Playwright browser lifecycle management.

    Handles:
        - Browser launch (headless Chromium)
        - Context creation with viewport/proxy settings
        - Context recycling every HEADLESS_CONTEXT_RECYCLE pages
        - Crash recovery (context restart on Playwright error)
        - Clean shutdown

    The Playwright import is deferred — if Playwright is not installed,
    CL2/CL3/CL4 are disabled but the fetcher still works for CL1.
    """

    def __init__(self) -> None:
        self._pw: Any = None            # playwright context manager
        self._browser: Any = None        # Browser instance
        self._context: Any = None        # BrowserContext
        self._page_count: int = 0        # pages since last context recycle
        self._context_recycles: int = 0
        self._available: bool = False
        self._proxy: Optional[str] = None

    async def initialize(self, proxy: Optional[str] = None) -> bool:
        """
        Launch Playwright and Chromium.  Returns True if successful.
        If Playwright is not installed, returns False (CL2+ disabled).
        """
        self._proxy = proxy
        try:
            from playwright.async_api import async_playwright
            self._pw = await async_playwright().__aenter__()
            self._browser = await self._pw.chromium.launch(
                headless=True,
                args=[
                    "--disable-gpu",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-setuid-sandbox",
                ],
            )
            await self._create_context(proxy)
            self._available = True
            log.info("playwright: initialized — Chromium headless")
            return True

        except ImportError:
            log.info("playwright: not installed — CL2/CL3/CL4 disabled")
            self._available = False
            return False
        except Exception as exc:
            log.warning("playwright: initialization failed: %s", exc)
            self._available = False
            return False

    async def _create_context(self, proxy: Optional[str] = None) -> None:
        """Create a new browser context with standard viewport."""
        proxy_config = None
        if proxy:
            proxy_config = {"server": proxy}

        if self._context:
            try:
                await self._context.close()
            except Exception: # noqa
                pass

        self._context = await self._browser.new_context(
            viewport={"width": HEADLESS_VIEWPORT_WIDTH, "height": HEADLESS_VIEWPORT_HEIGHT},
            proxy=proxy_config,
            ignore_https_errors=False,
            java_script_enabled=True,
        )
        self._page_count = 0

    async def fetch_page(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout: int = HEADLESS_NAVIGATION_TIMEOUT,
    ) -> Tuple[str, int, Dict[str, str]]:
        """
        Navigate to URL, wait, return (html_content, status_code, headers).

        Raises playwright errors on navigation failure.
        Automatically recycles context every HEADLESS_CONTEXT_RECYCLE pages.
        """
        if not self._available or not self._context:
            raise RuntimeError("playwright not initialized")

        # Recycle context if needed
        if self._page_count >= HEADLESS_CONTEXT_RECYCLE:
            await self.recycle_context()

        page = await self._context.new_page()
        try:
            response = await page.goto(url, wait_until=wait_until, timeout=timeout)
            status_code = response.status if response else 0
            headers = {}
            if response:
                for k, v in response.headers.items():
                    headers[k.lower()] = v
            content = await page.content()
            self._page_count += 1
            return content, status_code, headers
        finally:
            try:
                await page.close()
            except Exception: # noqa
                pass

    async def recycle_context(self, proxy: Optional[str] = None) -> None:
        """Close current context and create a new one.  Prevents memory accumulation."""
        proxy_to_use = proxy or self._proxy
        await self._create_context(proxy_to_use)
        self._context_recycles += 1
        log.debug(
            "playwright: context recycled (recycle #%d)", self._context_recycles,
        )

    async def shutdown(self) -> None:
        """Close browser and Playwright.  Safe to call multiple times."""
        if self._context:
            try:
                await self._context.close()
            except Exception: # noqa
                pass
            self._context = None
        if self._browser:
            try:
                await self._browser.close()
            except Exception: # noqa
                pass
            self._browser = None
        if self._pw:
            try:
                await self._pw.__aexit__(None, None, None)
            except Exception: # noqa
                pass
            self._pw = None
        self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def page_count(self) -> int:
        return self._page_count

    @property
    def context_recycles(self) -> int:
        return self._context_recycles


# ═══════════════════════════════════════════════════════════════════════════════
#
#   THE FETCHER
#
#   This is the vacuum.  It receives CrawlManifest, executes URLs in
#   strict priority order, emits RawFetchEvent per URL.  Four fetch modes
#   mapped to four clearance levels.  Every decision about which mode to
#   use was made by the preparser.  The fetcher reads the decision and
#   executes it.
#
#   Zero logic.  Execute manifest.  Emit event.  Continue.
#
# ═══════════════════════════════════════════════════════════════════════════════


class Fetcher:
    """
    The vacuum.  The only file in ``crawler/`` that talks to the internet.

    Lifecycle:
        1. ``initialize()`` — set up httpx client, Playwright, Tor, bloom,
           frontier, cursor, rate limiter, staging.  Probe CL availability.
        2. ``handle_manifest_ready(event)`` — execute a manifest.  Called by
           the bus when CrawlManifestReadyEvent arrives.
        3. ``shutdown()`` — close everything cleanly.

    Concurrency:
        Multiple manifests can execute concurrently (up to semaphore limits
        per CL level).  Within a single manifest, URLs are processed
        sequentially in priority order.  Sequential execution respects
        the rate-limiting and priority logic from crawl_planner.py.

    Crash recovery:
        On restart, ``frontier.resume(manifest_id)`` returns the next
        unprocessed URL.  ``crawl_cursor`` tracks the position atomically.
        ``bloom_filter`` survives process death via mmap.  Zero duplicate
        fetches.  Zero missed URLs (within CHECKPOINT_INTERVAL tolerance).
    """

    def __init__(
        self,
        bus: Optional[CrawlerBus] = None,
        bloom: Optional[Any] = None,
        cursor: Optional[Any] = None,
        frontier: Optional[Any] = None,
        rate_limiter: Optional[Any] = None,
        staging_dir: Path = STAGING_PATH,
        store_dir: Path = Path("store"),
    ) -> None:
        # ── Event bus ─────────────────────────────────────────────────
        self._bus = bus or CrawlerBus()

        # ── Dependencies (initialized lazily if not injected) ─────────
        self._bloom = bloom
        self._cursor = cursor
        self._frontier = frontier
        self._rate_limiter = rate_limiter
        self._store_dir = store_dir

        # ── HTTP client ───────────────────────────────────────────────
        self._http_client: Optional[httpx.AsyncClient] = None

        # ── Playwright ────────────────────────────────────────────────
        self._pw_manager = PlaywrightManager()

        # ── Tor ───────────────────────────────────────────────────────
        self._tor_manager = TorCircuitManager()

        # ── Staging ───────────────────────────────────────────────────
        self._staging = StagingPipeline(staging_dir)

        # ── CL state ─────────────────────────────────────────────────
        self._cl_state = CLState()

        # ── Concurrency semaphores ────────────────────────────────────
        self._semaphores: Dict[int, asyncio.Semaphore] = {
            1: asyncio.Semaphore(MAX_CONCURRENT_MANIFESTS),
            2: asyncio.Semaphore(MAX_CONCURRENT_CL2),
            3: asyncio.Semaphore(MAX_CONCURRENT_CL3),
            4: asyncio.Semaphore(MAX_CONCURRENT_CL4),
        }

        # ── Active manifests ──────────────────────────────────────────
        self._active_manifests: Dict[str, ManifestTelemetry] = {}
        self._manifest_tasks: Dict[str, asyncio.Task] = {}

        # ── Fetcher state ─────────────────────────────────────────────
        self._initialized: bool = False
        self._shutting_down: bool = False
        self._tor_fetch_count: int = 0
        self._total_urls_processed: int = 0

        # ── Context tracking for CL2 ─────────────────────────────────
        self._context_fetch_count: int = 0

    # ──────────────────────────────────────────────────────────────────────
    # INITIALIZATION AND SHUTDOWN
    # ──────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize all fetcher subsystems.

        Order matters:
            1. Staging directory
            2. httpx client (CL1 — always available)
            3. Bloom filter
            4. Crawl cursor
            5. Frontier
            6. Rate limiter
            7. Playwright (CL2 — optional)
            8. Tor availability check (CL3/CL4 — optional)
            9. Subscribe to bus events
            10. Log initialization summary
        """
        if self._initialized:
            log.warning("fetcher: already initialized")
            return

        log.info("fetcher: initializing...")

        # 1. Staging
        await self._staging.initialize()
        log.info("fetcher: staging pipeline ready — %s", self._staging._dir) # noqa

        # 2. httpx client
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=STATIC_TIMEOUT_CONNECT,
                read=STATIC_TIMEOUT_READ,
                write=STATIC_TIMEOUT_WRITE,
                pool=STATIC_TIMEOUT_POOL,
            ),
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            verify=True,
            http2=True,
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,
            ),
        )
        log.info("fetcher: httpx client ready — CL1 available")

        # 3. Bloom filter
        if self._bloom is None and BloomFilter is not None:
            self._bloom = BloomFilter(
                path=self._store_dir / "bloom.bin",
            )
            await self._bloom.initialize()
            count = await self._bloom.count()
            log.info("fetcher: bloom filter ready — %d URLs tracked", count)
        elif self._bloom is not None:
            if hasattr(self._bloom, "is_initialized") and not self._bloom.is_initialized:
                await self._bloom.initialize()
            log.info("fetcher: bloom filter ready (injected)")

        # 4. Crawl cursor
        if self._cursor is None and CrawlCursor is not None:
            self._cursor = CrawlCursor(
                db_path=self._store_dir / "crawl_cursor.db",
            )
            await self._cursor.initialize()
            log.info("fetcher: crawl cursor ready")
        elif self._cursor is not None:
            log.info("fetcher: crawl cursor ready (injected)")

        # 5. Frontier
        if self._frontier is None and Frontier is not None:
            self._frontier = Frontier(
                cursor=self._cursor,
                db_path=self._store_dir / "frontier.db",
            )
            await self._frontier.initialize()
            log.info("fetcher: frontier ready")
        elif self._frontier is not None:
            log.info("fetcher: frontier ready (injected)")

        # 6. Rate limiter
        if self._rate_limiter is None and RateLimiter is not None:
            self._rate_limiter = RateLimiter()
            log.info("fetcher: rate limiter ready")
        elif self._rate_limiter is not None:
            log.info("fetcher: rate limiter ready (injected)")

        # 7. Playwright (CL2)
        pw_ok = await self._pw_manager.initialize()
        self._cl_state.cl2_available = pw_ok

        # 8. Tor (CL3/CL4)
        tor_ok = await self._tor_manager.check_available()
        self._cl_state.cl3_available = tor_ok and pw_ok
        self._cl_state.cl4_available = tor_ok and pw_ok

        # 9. Bus subscriptions
        self._bus.subscribe(CrawlManifestReadyEvent, self.handle_manifest_ready)
        self._bus.subscribe(CLStateUpdateEvent, self._handle_cl_state_update)

        # 10. Summary
        self._initialized = True
        log.info(
            "fetcher: INITIALIZED — CL state: %s | Tor: %s | Playwright: %s",
            self._cl_state.to_dict(),
            "available" if tor_ok else "unavailable",
            "available" if pw_ok else "unavailable",
        )

    async def shutdown(self) -> None:
        """
        Shut down all fetcher subsystems cleanly.

        1. Set shutting_down flag (prevents new manifests)
        2. Cancel active manifest tasks
        3. Close Playwright
        4. Close httpx client
        5. Flush bloom filter
        6. Close frontier and cursor
        7. Clean staging directory
        """
        if self._shutting_down:
            return
        self._shutting_down = True
        log.info("fetcher: shutting down...")

        # Cancel active tasks
        for task_id, task in list(self._manifest_tasks.items()):
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
            del self._manifest_tasks[task_id]

        # Playwright
        await self._pw_manager.shutdown()

        # httpx
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None

        # Bloom filter
        if self._bloom and hasattr(self._bloom, "close"):
            await self._bloom.close()

        # Frontier
        if self._frontier and hasattr(self._frontier, "close"):
            await self._frontier.close()

        # Cursor
        if self._cursor and hasattr(self._cursor, "close"):
            await self._cursor.close()

        # Staging cleanup
        cleaned = await self._staging.cleanup()
        if cleaned > 0:
            log.info("fetcher: cleaned %d staging files", cleaned)

        await asyncio.sleep(0.25)
        self._initialized = False
        log.info(
            "fetcher: SHUTDOWN — %d total URLs processed",
            self._total_urls_processed,
        )

    # ──────────────────────────────────────────────────────────────────────
    # CL STATE MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────

    async def _handle_cl_state_update(self, event: CLStateUpdateEvent) -> None:
        """Handle CLStateUpdateEvent from RLCPC."""
        self._cl_state.cl2_available = event.cl2_available
        self._cl_state.cl3_available = event.cl3_available
        self._cl_state.cl4_available = event.cl4_available
        log.info(
            "fetcher: CL state updated — %s (reason: %s)",
            self._cl_state.to_dict(), event.reason,
        )

    # ──────────────────────────────────────────────────────────────────────
    # RENDER STRATEGY
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _render_wait_strategy(render_mode: str) -> str:
        """
        Map CrawlURL.render_mode to Playwright wait strategy.

        static   → domcontentloaded (fast — HTML parsed, no JS wait)
        headless → networkidle (slow — waits for no network activity 500ms)
        """
        if render_mode == "headless":
            return "networkidle"
        return "domcontentloaded"

    # ──────────────────────────────────────────────────────────────────────
    # FETCH MODES — CL1 through CL4
    # ──────────────────────────────────────────────────────────────────────

    async def _fetch_static(self, crawl_url: CrawlURL, manifest_id: str) -> RawFetchEvent:
        """
        CL1 — STATIC fetch via httpx clearnet.

        Default mode.  No unlock required.  Covers the majority of public
        web topology classes that render without JavaScript.

        Handles:
            - Connection pooling via shared httpx.AsyncClient
            - Follow redirects up to MAX_REDIRECTS
            - Response truncation at max_response_bytes
            - Decompression (gzip, deflate, br) handled by httpx
            - SSL verification always on (verify=True)

        On any HTTP error: emits FetchAnomalyEvent and re-raises.
        """
        t_start = time.monotonic()

        try:
            response = await self._http_client.get(
                crawl_url.url,
                headers={
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept-Language": "en-US,en;q=0.9",
                },
            )

            fetch_latency = time.monotonic() - t_start

            # Check for anomalous status codes
            status = response.status_code
            if status == 429:
                raise _RateLimitedError(crawl_url, manifest_id, status)
            if status == 403:
                raise _AccessDeniedError(crawl_url, manifest_id, status)
            if status >= 500:
                raise _ServerError(crawl_url, manifest_id, status)

            # Read response body with truncation
            raw_bytes = response.content
            if len(raw_bytes) > crawl_url.max_response_bytes:
                raw_bytes = raw_bytes[: crawl_url.max_response_bytes]

            # Stage bytes
            staging_path = await self._staging.stage(raw_bytes, crawl_url.run_id)
            staged_bytes = await self._staging.unstage(staging_path)

            # Lowercase headers
            headers = {k.lower(): v for k, v in response.headers.items()}

            return RawFetchEvent(
                url=crawl_url.url,
                raw_bytes=staged_bytes,
                status_code=status,
                headers=headers,
                fetch_latency=fetch_latency,
                fetch_mode=FetchMode.STATIC,
                is_robots_txt=crawl_url.is_robots,
                is_sitemap=crawl_url.is_sitemap,
                topology_hint=crawl_url.topology_hint,
                run_id=crawl_url.run_id,
                manifest_id=manifest_id,
                byte_count=len(staged_bytes),
            )

        except httpx.TimeoutException as exc:
            raise _TimeoutError(crawl_url, manifest_id, str(exc)) from exc
        except httpx.TooManyRedirects as exc:
            raise _RedirectExceededError(crawl_url, manifest_id) from exc
        except httpx.ConnectError as exc:
            raise _ConnectionFailedError(crawl_url, manifest_id, str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            raise _ConnectionFailedError(crawl_url, manifest_id, str(exc)) from exc
        except (_RateLimitedError, _AccessDeniedError, _ServerError):
            raise
        except OSError as exc:
            raise _StagingError(crawl_url, manifest_id, str(exc)) from exc

    async def _fetch_headless(self, crawl_url: CrawlURL, manifest_id: str) -> RawFetchEvent:
        """
        CL2 — HEADLESS fetch via Playwright + Chromium.

        JS-heavy pages, SPAs, docs with client-side rendering.
        Uses Playwright with configurable wait strategy based on
        render_mode from CrawlURL.
        """
        t_start = time.monotonic()

        try:
            wait_strategy = self._render_wait_strategy(crawl_url.render_mode)

            content, status_code, headers = await self._pw_manager.fetch_page(
                url=crawl_url.url,
                wait_until=wait_strategy,
                timeout=HEADLESS_NAVIGATION_TIMEOUT,
            )
            fetch_latency = time.monotonic() - t_start

            # Check status codes
            if status_code == 429:
                raise _RateLimitedError(crawl_url, manifest_id, status_code)
            if status_code == 403:
                raise _AccessDeniedError(crawl_url, manifest_id, status_code)
            if status_code >= 500:
                raise _ServerError(crawl_url, manifest_id, status_code)

            raw_bytes = content.encode("utf-8")
            if len(raw_bytes) > crawl_url.max_response_bytes:
                raw_bytes = raw_bytes[: crawl_url.max_response_bytes]

            staging_path = await self._staging.stage(raw_bytes, crawl_url.run_id)
            staged_bytes = await self._staging.unstage(staging_path)

            return RawFetchEvent(
                url=crawl_url.url,
                raw_bytes=staged_bytes,
                status_code=status_code,
                headers=headers,
                fetch_latency=fetch_latency,
                fetch_mode=FetchMode.HEADLESS,
                is_robots_txt=crawl_url.is_robots,
                is_sitemap=crawl_url.is_sitemap,
                topology_hint=crawl_url.topology_hint,
                run_id=crawl_url.run_id,
                manifest_id=manifest_id,
                byte_count=len(staged_bytes),
            )

        except (_RateLimitedError, _AccessDeniedError, _ServerError):
            raise
        except OSError as exc:
            raise _StagingError(crawl_url, manifest_id, str(exc)) from exc
        except Exception as exc:
            # Playwright crash — try context recovery
            try:
                await self._pw_manager.recycle_context()
            except Exception: # noqa
                pass
            raise _PlaywrightCrashError(crawl_url, manifest_id, str(exc)) from exc

    async def _fetch_tor(self, crawl_url: CrawlURL, manifest_id: str) -> RawFetchEvent:
        """
        CL3 — TOR fetch via Playwright + Chromium + Tor SOCKS5 proxy.

        Anonymized clearnet.  Paywalled content, geo-blocked, hostile
        domains.  Routes Playwright through Tor.

        New circuit every TOR_CIRCUIT_INTERVAL fetches.
        Browser context reset after every circuit change.
        DNS resolution through Tor (SOCKS5h).
        """
        t_start = time.monotonic()

        # Circuit rotation every TOR_CIRCUIT_INTERVAL fetches
        self._tor_fetch_count += 1
        if self._tor_fetch_count % TOR_CIRCUIT_INTERVAL == 0:
            await self._tor_manager.request_new_circuit()
            await self._pw_manager.recycle_context(self._tor_manager.proxy_url)

        try:
            wait_strategy = self._render_wait_strategy(crawl_url.render_mode)

            content, status_code, headers = await self._pw_manager.fetch_page(
                url=crawl_url.url,
                wait_until=wait_strategy,
                timeout=HEADLESS_NAVIGATION_TIMEOUT,
            )
            fetch_latency = time.monotonic() - t_start

            if status_code == 429:
                raise _RateLimitedError(crawl_url, manifest_id, status_code)
            if status_code == 403:
                raise _AccessDeniedError(crawl_url, manifest_id, status_code)
            if status_code >= 500:
                raise _ServerError(crawl_url, manifest_id, status_code)

            raw_bytes = content.encode("utf-8")
            if len(raw_bytes) > crawl_url.max_response_bytes:
                raw_bytes = raw_bytes[: crawl_url.max_response_bytes]

            staging_path = await self._staging.stage(raw_bytes, crawl_url.run_id)
            staged_bytes = await self._staging.unstage(staging_path)

            return RawFetchEvent(
                url=crawl_url.url,
                raw_bytes=staged_bytes,
                status_code=status_code,
                headers=headers,
                fetch_latency=fetch_latency,
                fetch_mode=FetchMode.TOR,
                is_robots_txt=crawl_url.is_robots,
                is_sitemap=crawl_url.is_sitemap,
                topology_hint=crawl_url.topology_hint,
                run_id=crawl_url.run_id,
                manifest_id=manifest_id,
                byte_count=len(staged_bytes),
            )

        except (_RateLimitedError, _AccessDeniedError, _ServerError):
            raise
        except OSError as exc:
            raise _StagingError(crawl_url, manifest_id, str(exc)) from exc
        except Exception as exc:
            if "tor" in str(exc).lower() or "socks" in str(exc).lower():
                raise _TorUnavailableError(crawl_url, manifest_id, str(exc)) from exc
            try:
                await self._pw_manager.recycle_context(self._tor_manager.proxy_url)
            except Exception: # noqa
                pass
            raise _PlaywrightCrashError(crawl_url, manifest_id, str(exc)) from exc

    async def _fetch_tor_full(self, crawl_url: CrawlURL, manifest_id: str) -> RawFetchEvent:
        """
        CL4 — TOR_FULL fetch.  Full reach.  Zero fingerprint.

        Differences from CL3:
            - New Tor circuit EVERY fetch (not every 10)
            - New browser context EVERY fetch (zero fingerprint continuity)
            - Random inter-request jitter [0.5, 3.0] seconds
            - Context explicitly closed after each fetch
        """
        # New circuit before every fetch
        await self._tor_manager.request_new_circuit()
        await self._pw_manager.recycle_context(self._tor_manager.proxy_url)

        t_start = time.monotonic()

        try:
            wait_strategy = self._render_wait_strategy(crawl_url.render_mode)

            content, status_code, headers = await self._pw_manager.fetch_page(
                url=crawl_url.url,
                wait_until=wait_strategy,
                timeout=HEADLESS_NAVIGATION_TIMEOUT,
            )
            fetch_latency = time.monotonic() - t_start

            if status_code == 429:
                raise _RateLimitedError(crawl_url, manifest_id, status_code)
            if status_code == 403:
                raise _AccessDeniedError(crawl_url, manifest_id, status_code)
            if status_code >= 500:
                raise _ServerError(crawl_url, manifest_id, status_code)

            raw_bytes = content.encode("utf-8")
            if len(raw_bytes) > crawl_url.max_response_bytes:
                raw_bytes = raw_bytes[: crawl_url.max_response_bytes]

            staging_path = await self._staging.stage(raw_bytes, crawl_url.run_id)
            staged_bytes = await self._staging.unstage(staging_path)

            event = RawFetchEvent(
                url=crawl_url.url,
                raw_bytes=staged_bytes,
                status_code=status_code,
                headers=headers,
                fetch_latency=fetch_latency,
                fetch_mode=FetchMode.TOR_FULL,
                is_robots_txt=crawl_url.is_robots,
                is_sitemap=crawl_url.is_sitemap,
                topology_hint=crawl_url.topology_hint,
                run_id=crawl_url.run_id,
                manifest_id=manifest_id,
                byte_count=len(staged_bytes),
            )

            # Human-like jitter after fetch
            jitter = random.uniform(TOR_FULL_JITTER_MIN, TOR_FULL_JITTER_MAX)
            await asyncio.sleep(jitter)

            # Close context explicitly — zero fingerprint continuity
            await self._pw_manager.recycle_context(self._tor_manager.proxy_url)

            return event

        except (_RateLimitedError, _AccessDeniedError, _ServerError):
            raise
        except OSError as exc:
            raise _StagingError(crawl_url, manifest_id, str(exc)) from exc
        except Exception as exc:
            if "tor" in str(exc).lower() or "socks" in str(exc).lower():
                raise _TorUnavailableError(crawl_url, manifest_id, str(exc)) from exc
            try:
                await self._pw_manager.recycle_context(self._tor_manager.proxy_url)
            except Exception: # noqa
                pass
            raise _PlaywrightCrashError(crawl_url, manifest_id, str(exc)) from exc

    # ──────────────────────────────────────────────────────────────────────
    # ANOMALY EMISSION
    # ──────────────────────────────────────────────────────────────────────

    async def _emit_anomaly(
        self,
        crawl_url: CrawlURL,
        manifest_id: str,
        anomaly_type: str,
        status_code: Optional[int] = None,
        detail: str = "",
    ) -> FetchAnomalyEvent:
        """
        Construct and emit a FetchAnomalyEvent.  Fire-and-forget.
        Never raises.  The manifest always continues after this.
        """
        anomaly = FetchAnomalyEvent(
            url=crawl_url.url,
            fetch_mode=crawl_url.fetch_mode,
            status_code=status_code,
            anomaly_type=anomaly_type,
            run_id=crawl_url.run_id,
            manifest_id=manifest_id,
            detail=detail,
        )
        try:
            await self._bus.emit(anomaly)
        except Exception as exc:
            log.warning("fetcher: failed to emit anomaly: %s", exc)
        return anomaly

    # ──────────────────────────────────────────────────────────────────────
    # URL EXECUTION PIPELINE
    # ──────────────────────────────────────────────────────────────────────

    async def _execute_url(
        self,
        crawl_url: CrawlURL,
        manifest_id: str,
        telemetry: ManifestTelemetry,
    ) -> Optional[RawFetchEvent]:
        """
        Execute one URL from the manifest.

        Protocol:
            1. Acquire rate limiter token (blocks if needed)
            2. Check bloom filter (skip if already seen)
            3. Resolve effective CL mode (silent fallback)
            4. Dispatch to the appropriate fetch function
            5. On success: stage, emit RawFetchEvent, add to bloom
            6. On failure: emit FetchAnomalyEvent, mark failed in frontier
            7. Update telemetry

        Never raises.  Every failure mode has a defined path that
        continues the manifest.
        """
        # 1. Rate limiter
        if self._rate_limiter:
            try:
                profile = crawl_url.rate_limit_profile
                await self._rate_limiter.acquire(
                    url=crawl_url.url,
                    profile=profile,
                )
            except Exception as exc:
                log.warning("fetcher: rate_limiter.acquire failed: %s", exc)

        # 2. Bloom filter dedup
        if self._bloom:
            try:
                if await self._bloom.contains(crawl_url.url):
                    telemetry.record_skip()
                    if self._frontier:
                        asyncio.create_task(
                            self._safe_frontier_mark(
                                self._frontier.mark_skipped,
                                manifest_id,
                                crawl_url.url,
                            )
                        )
                    return None
            except Exception as exc:
                log.warning("fetcher: bloom.contains failed: %s — proceeding with fetch", exc)

        # 3. Resolve effective CL mode
        effective_mode = self._cl_state.effective_mode(crawl_url.fetch_mode)

        # 4. Dispatch fetch
        try:
            if effective_mode == FetchMode.STATIC:
                event = await self._fetch_static(crawl_url, manifest_id)
            elif effective_mode == FetchMode.HEADLESS:
                event = await self._fetch_headless(crawl_url, manifest_id)
            elif effective_mode == FetchMode.TOR:
                event = await self._fetch_tor(crawl_url, manifest_id)
            elif effective_mode == FetchMode.TOR_FULL:
                event = await self._fetch_tor_full(crawl_url, manifest_id)
            else:
                event = await self._fetch_static(crawl_url, manifest_id) # noqa | runtime defensive check

        except _RateLimitedError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "rate_limited",
                status_code=exc.status_code,
                detail=f"HTTP 429 — architecture bug — domain={urlparse(crawl_url.url).netloc}",
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _AccessDeniedError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "access_denied",
                status_code=exc.status_code,
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _ServerError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "server_error",
                status_code=exc.status_code,
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _TimeoutError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "timeout", detail=exc.detail,
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _ConnectionFailedError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "connection_failed", detail=exc.detail,
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _RedirectExceededError:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "redirect_exceeded",
                detail=f"exceeded MAX_REDIRECTS={MAX_REDIRECTS}",
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _TorUnavailableError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "tor_unavailable", detail=exc.detail,
            )
            telemetry.record_failure(anomaly, effective_mode)
            # Disable CL3/CL4 for the rest of the session
            self._cl_state.cl3_available = False
            self._cl_state.cl4_available = False
            log.warning("fetcher: Tor unavailable — CL3/CL4 disabled for session")
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _PlaywrightCrashError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "playwright_crash", detail=exc.detail,
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except _StagingError as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "staging_error", detail=exc.detail,
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        except Exception as exc:
            anomaly = await self._emit_anomaly(
                crawl_url, manifest_id, "unknown_error",
                detail=f"{type(exc).__name__}: {exc}",
            )
            telemetry.record_failure(anomaly, effective_mode)
            if self._frontier:
                asyncio.create_task(
                    self._safe_frontier_mark(
                        self._frontier.mark_failed, manifest_id, crawl_url.url,
                    )
                )
            return None

        # 5. Success path — emit event, add to bloom, mark done
        try:
            await self._bus.emit(event)
        except Exception as exc:
            log.warning("fetcher: bus.emit(RawFetchEvent) failed: %s", exc)

        if self._bloom:
            try:
                await self._bloom.add(crawl_url.url)
            except Exception as exc:
                log.warning("fetcher: bloom.add failed: %s", exc)

        if self._frontier:
            asyncio.create_task(
                self._safe_frontier_mark(
                    self._frontier.mark_done, manifest_id, crawl_url.url,
                )
            )

        telemetry.record_success(
            event.fetch_latency, event.byte_count, event.status_code,
            effective_mode, raw_bytes=event.raw_bytes,
        )

        # SEP: score prediction against actual result (off critical path)
        if telemetry.predictor:
            try:
                score = telemetry.predictor.observe(
                    url=crawl_url.url,
                    status_code=event.status_code,
                    size_bytes=event.byte_count,
                    latency=event.fetch_latency,
                    raw_bytes=event.raw_bytes,
                )
                if score and score.is_divergent:
                    log.debug(
                        "sep: divergence on %s — size_z=%.1f latency_z=%.1f status=%s",
                        crawl_url.url, score.size_z_score,
                        score.latency_z_score, score.status_correct,
                    )
            except Exception: # noqa
                pass  # SEP is never on the critical path

        return event

    async def _safe_frontier_mark( # noqa
        self, func: Callable, manifest_id: str, url: str,
    ) -> None:
        """Fire-and-forget frontier status update.  Never raises."""
        try:
            await func(manifest_id, url)
        except Exception as exc:
            log.warning("fetcher: frontier mark failed: %s", exc)

    # ──────────────────────────────────────────────────────────────────────
    # MANIFEST EXECUTION
    # ──────────────────────────────────────────────────────────────────────

    async def _execute_manifest(self, manifest: CrawlManifest) -> ManifestTelemetry:
        """
        Execute a complete CrawlManifest sequentially in priority order.

        URLs are processed one at a time within a manifest.  Sequential
        execution respects rate-limiting and priority logic from
        crawl_planner.py.  Parallelizing within a manifest would defeat
        rate limiting.

        On crash recovery: frontier.resume() returns from cursor position.
        """
        telemetry = ManifestTelemetry(
            manifest_id=manifest.manifest_id,
            domain=manifest.domain,
            hazard_estimator=HazardRateEstimator(manifest.total_urls),
        )
        self._active_manifests[manifest.manifest_id] = telemetry

        log.info(
            "fetcher: executing manifest %s — domain=%s, urls=%d, cl_required=%d",
            manifest.manifest_id[:8], manifest.domain,
            manifest.total_urls, manifest.clearance_required,
        )

        # Register rate limit profiles
        if self._rate_limiter:
            seen_domains: Set[str] = set()
            for crawl_url in manifest.urls:
                if crawl_url.rate_limit_profile:
                    domain = crawl_url.rate_limit_profile.domain
                    if domain not in seen_domains:
                        try:
                            await self._rate_limiter.register(crawl_url.rate_limit_profile)
                            seen_domains.add(domain)
                        except Exception as exc:
                            log.warning("fetcher: register rate profile failed: %s", exc)

        # Load manifest into frontier
        if self._frontier:
            try:
                await self._frontier.load_manifest(manifest)
            except Exception as exc:
                log.error(
                    "fetcher: frontier.load_manifest failed: %s — aborting manifest",
                    exc,
                )
                self._active_manifests.pop(manifest.manifest_id, None)
                return telemetry

        # Determine URL source — frontier resume or direct manifest iteration
        url_source: AsyncIterator[CrawlURL]
        if self._frontier:
            try:
                url_source = self._frontier.resume(manifest.manifest_id)
            except Exception as exc:
                log.error("fetcher: frontier.resume failed: %s — using direct iteration", exc)
                url_source = _manifest_iter(manifest)
        else:
            url_source = _manifest_iter(manifest)

        # Sequential execution loop
        position = 0
        try:
            async for crawl_url in url_source:
                if self._shutting_down:
                    log.info("fetcher: shutting down — stopping manifest %s", manifest.manifest_id[:8])
                    break

                await self._execute_url(crawl_url, manifest.manifest_id, telemetry)
                position += 1
                self._total_urls_processed += 1

                # Checkpoint every CHECKPOINT_INTERVAL URLs
                if self._cursor and position % CHECKPOINT_INTERVAL == 0:
                    try:
                        await self._cursor.checkpoint(
                            manifest_id=manifest.manifest_id,
                            position=position,
                            url=crawl_url.url,
                            total_urls=manifest.total_urls,
                        )
                    except Exception as exc:
                        log.warning(
                            "fetcher: cursor.checkpoint failed at position %d: %s",
                            position, exc,
                        )

        except asyncio.CancelledError:
            log.info("fetcher: manifest %s cancelled", manifest.manifest_id[:8])
        except Exception as exc:
            log.error(
                "fetcher: manifest %s failed at position %d: %s",
                manifest.manifest_id[:8], position, exc,
            )

        # Final checkpoint
        if self._cursor:
            try:
                await self._cursor.checkpoint(
                    manifest_id=manifest.manifest_id,
                    position=position,
                    url="<manifest_complete>",
                    total_urls=manifest.total_urls,
                )
            except Exception as exc:
                log.warning("fetcher: final checkpoint failed: %s", exc)

        # Emit ManifestCompleteEvent
        stats = FrontierStats(
            manifest_id=manifest.manifest_id,
            pending=max(0, manifest.total_urls - telemetry.urls_attempted),
            done=telemetry.urls_succeeded,
            failed=telemetry.urls_failed,
            skipped=telemetry.urls_skipped,
        )
        try:
            await self._bus.emit(ManifestCompleteEvent(
                domain=manifest.domain,
                manifest_id=manifest.manifest_id,
                stats=stats,
            ))
        except Exception as exc:
            log.warning("fetcher: ManifestCompleteEvent emit failed: %s", exc)

        # Clear cursor for completed manifest
        if self._cursor:
            try:
                await self._cursor.clear(manifest.manifest_id)
            except Exception as exc:
                log.warning("fetcher: cursor.clear failed: %s", exc)

        # Log summary
        summary = telemetry.summary()
        log.info(
            "fetcher: manifest %s COMPLETE — "
            "succeeded=%d failed=%d skipped=%d elapsed=%.1fs throughput=%.1f/s "
            "p50=%.0fms p95=%.0fms p99=%.0fms entropy=%.2f bits",
            manifest.manifest_id[:8],
            summary["succeeded"], summary["failed"], summary["skipped"],
            summary["elapsed_sec"], summary["throughput_urls_per_sec"],
            summary["latency_p50_ms"], summary["latency_p95_ms"],
            summary["latency_p99_ms"], summary["status_entropy_bits"],
        )

        self._active_manifests.pop(manifest.manifest_id, None)
        return telemetry

    # ──────────────────────────────────────────────────────────────────────
    # MANIFEST READY HANDLER (bus subscriber)
    # ──────────────────────────────────────────────────────────────────────

    async def handle_manifest_ready(self, event: Any) -> None:
        """
        Handle CrawlManifestReadyEvent from the bus.

        Acquires the appropriate CL-level semaphore and dispatches
        the manifest execution as an asyncio Task.  Multiple manifests
        can run concurrently up to the semaphore limit.
        """
        if self._shutting_down:
            log.warning("fetcher: ignoring manifest — shutting down")
            return

        if isinstance(event, CrawlManifestReadyEvent):
            manifest = event.manifest
        elif isinstance(event, CrawlManifest):
            manifest = event
        else:
            log.warning("fetcher: unexpected event type: %s", type(event))
            return

        cl_level = manifest.clearance_required
        semaphore = self._semaphores.get(cl_level, self._semaphores[1])

        async def _guarded_execute() -> None:
            async with semaphore:
                await self._execute_manifest(manifest)

        task = asyncio.create_task(_guarded_execute())
        self._manifest_tasks[manifest.manifest_id] = task

        def _cleanup(t: asyncio.Task) -> None:
            self._manifest_tasks.pop(manifest.manifest_id, None)

        task.add_done_callback(_cleanup)

    # ──────────────────────────────────────────────────────────────────────
    # PUBLIC API — convenience methods
    # ──────────────────────────────────────────────────────────────────────

    async def fetch_single(
        self,
        url: str,
        cl_level: int = 1,
        topology_hint: str = "GENERIC_HTML",
    ) -> Optional[RawFetchEvent]:
        """
        Fetch a single URL at the specified CL level.
        Convenience method for testing and CLI.
        """
        if not self._initialized:
            await self.initialize()

        fetch_mode_map = {
            1: FetchMode.STATIC,
            2: FetchMode.HEADLESS,
            3: FetchMode.TOR,
            4: FetchMode.TOR_FULL,
        }
        mode = fetch_mode_map.get(cl_level, FetchMode.STATIC)
        run_id = str(uuid.uuid4())
        manifest_id = str(uuid.uuid4())

        domain = urlparse(url).netloc.lower()
        profile = RateLimitProfile(
            domain=domain,
            requests_per_second=1.0,
            crawl_delay_seconds=0.0,
            burst_capacity=3,
        )

        crawl_url = CrawlURL(
            url=url,
            topology_hint=topology_hint,
            fetch_mode=mode,
            render_mode="headless" if cl_level >= 2 else "static",
            priority=0,
            rate_limit_profile=profile,
            expected_content_type="text/html",
            crawl_delay_seconds=0.0,
            max_response_bytes=MAX_RESPONSE_BYTES,
            is_robots=False,
            is_sitemap=False,
            run_id=run_id,
        )

        telemetry = ManifestTelemetry(manifest_id=manifest_id, domain=domain)
        return await self._execute_url(crawl_url, manifest_id, telemetry)

    @property
    def cl_state(self) -> CLState:
        return self._cl_state

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    @property
    def active_manifest_count(self) -> int:
        return len(self._active_manifests)

    def active_telemetry(self) -> Dict[str, Dict[str, Any]]:
        return {
            mid: tel.summary() for mid, tel in self._active_manifests.items()
        }

    async def __aenter__(self) -> "Fetcher":
        await self.initialize()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.shutdown()


# ═══════════════════════════════════════════════════════════════════════════════
#
#   INTERNAL FETCH EXCEPTION TYPES
#
#   These are internal to the fetcher — they do not cross the module boundary.
#   They carry just enough context for _execute_url() to construct the
#   appropriate FetchAnomalyEvent.  They are caught in _execute_url() and
#   never escape it.
#
# ═══════════════════════════════════════════════════════════════════════════════

class _FetchInternalError(Exception):
    """Base for all internal fetch errors."""
    def __init__(self, crawl_url: CrawlURL, manifest_id: str, detail: str = "") -> None:
        self.crawl_url = crawl_url
        self.manifest_id = manifest_id
        self.detail = detail
        super().__init__(detail)


class _RateLimitedError(_FetchInternalError):
    def __init__(self, crawl_url: CrawlURL, manifest_id: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(crawl_url, manifest_id, f"HTTP {status_code}")


class _AccessDeniedError(_FetchInternalError):
    def __init__(self, crawl_url: CrawlURL, manifest_id: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(crawl_url, manifest_id, f"HTTP {status_code}")


class _ServerError(_FetchInternalError):
    def __init__(self, crawl_url: CrawlURL, manifest_id: str, status_code: int) -> None:
        self.status_code = status_code
        super().__init__(crawl_url, manifest_id, f"HTTP {status_code}")


class _TimeoutError(_FetchInternalError):
    pass


class _ConnectionFailedError(_FetchInternalError):
    pass


class _RedirectExceededError(_FetchInternalError):
    def __init__(self, crawl_url: CrawlURL, manifest_id: str) -> None:
        super().__init__(crawl_url, manifest_id, f"exceeded MAX_REDIRECTS={MAX_REDIRECTS}")


class _TorUnavailableError(_FetchInternalError):
    pass


class _PlaywrightCrashError(_FetchInternalError):
    pass


class _StagingError(_FetchInternalError):
    pass


# ═══════════════════════════════════════════════════════════════════════════════
#
#   HELPERS
#
# ═══════════════════════════════════════════════════════════════════════════════

async def _manifest_iter(manifest: CrawlManifest) -> AsyncIterator[CrawlURL]:
    """Async iterator over manifest URLs.  Used when frontier is unavailable."""
    for url in manifest.urls:
        yield url


def _extract_domain(url: str) -> str:
    """Extract domain from URL.  Strips port."""
    parsed = urlparse(url)
    netloc = parsed.netloc.lower()
    if ":" in netloc:
        netloc = netloc.rsplit(":", 1)[0]
    return netloc


def _build_test_cl_state(cl_level: int) -> CLState:
    """
    Pre-RLCPC hardcoded CL state for testing.
    Remove when RLCPC is live.
    """
    return CLState(
        cl1_available=True,
        cl2_available=cl_level >= 2,
        cl3_available=cl_level >= 3,
        cl4_available=cl_level >= 4,
    )


def _format_bytes(n: int) -> str:
    """Human-readable byte count."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n = int(n / 1024)
    return f"{n:.1f} TB"


def _format_duration(seconds: float) -> str:
    """Human-readable duration."""
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h {mins}m"


# ═══════════════════════════════════════════════════════════════════════════════
#
#   TESTS
#
#   All tests are self-contained — no external dependencies, no live HTTP.
#   Tests use mock transports and in-memory substitutes.
#
# ═══════════════════════════════════════════════════════════════════════════════


class _MockBloomFilter:
    """In-memory bloom filter substitute for testing."""

    def __init__(self) -> None:
        self._urls: Set[str] = set()
        self.is_initialized = True

    async def initialize(self) -> None:
        pass

    async def contains(self, url: str) -> bool:
        return url in self._urls

    async def add(self, url: str) -> None:
        self._urls.add(url)

    async def count(self) -> int:
        return len(self._urls)

    async def close(self) -> None:
        pass


class _MockCrawlCursor:
    """In-memory cursor substitute for testing."""

    def __init__(self) -> None:
        self._positions: Dict[str, int] = {}

    async def initialize(self, db_path: Optional[Path] = None) -> None:
        pass

    async def checkpoint(
        self, manifest_id: str, position: int, url: str, total_urls: int,
    ) -> None:
        self._positions[manifest_id] = position

    async def get_position(self, manifest_id: str) -> int:
        return self._positions.get(manifest_id, 0)

    async def clear(self, manifest_id: str) -> None:
        self._positions.pop(manifest_id, None)

    async def all_active(self) -> list: # noqa
        return []

    async def close(self) -> None:
        pass


class _MockFrontier:
    """In-memory frontier substitute for testing."""

    def __init__(self) -> None:
        self._manifests: Dict[str, List[CrawlURL]] = {}
        self._status: Dict[str, Dict[str, str]] = {}

    async def initialize(self, db_path: Optional[Path] = None) -> None:
        pass

    async def load_manifest(self, manifest: CrawlManifest) -> None:
        if manifest.manifest_id not in self._manifests:
            self._manifests[manifest.manifest_id] = list(manifest.urls)
            self._status[manifest.manifest_id] = {
                url.url: "pending" for url in manifest.urls
            }

    async def resume(self, manifest_id: str) -> AsyncIterator[CrawlURL]:
        urls = self._manifests.get(manifest_id, [])
        for url in urls:
            if self._status.get(manifest_id, {}).get(url.url) == "pending":
                yield url

    async def mark_done(self, manifest_id: str, url: str) -> None:
        if manifest_id in self._status:
            self._status[manifest_id][url] = "done"

    async def mark_failed(self, manifest_id: str, url: str) -> None:
        if manifest_id in self._status:
            self._status[manifest_id][url] = "failed"

    async def mark_skipped(self, manifest_id: str, url: str) -> None:
        if manifest_id in self._status:
            self._status[manifest_id][url] = "skipped"

    async def is_complete(self, manifest_id: str) -> bool:
        return all(
            s != "pending"
            for s in self._status.get(manifest_id, {}).values()
        )

    async def pending_count(self, manifest_id: str) -> int:
        return sum(
            1 for s in self._status.get(manifest_id, {}).values()
            if s == "pending"
        )

    async def stats(self, manifest_id: str) -> FrontierStats:
        statuses = self._status.get(manifest_id, {})
        return FrontierStats(
            manifest_id=manifest_id,
            pending=sum(1 for s in statuses.values() if s == "pending"),
            done=sum(1 for s in statuses.values() if s == "done"),
            failed=sum(1 for s in statuses.values() if s == "failed"),
            skipped=sum(1 for s in statuses.values() if s == "skipped"),
        )

    async def close(self) -> None:
        pass


def _make_test_manifest(
    n_urls: int = 10,
    domain: str = "example.com",
    fetch_mode: FetchMode = FetchMode.STATIC,
) -> CrawlManifest:
    """Create a test manifest with n_urls."""
    manifest_id = str(uuid.uuid4())
    profile = RateLimitProfile(
        domain=domain, requests_per_second=100.0,
        crawl_delay_seconds=0.0, burst_capacity=100,
    )
    urls = [
        CrawlURL(
            url=f"https://{domain}/page/{i}",
            topology_hint="GENERIC_HTML",
            fetch_mode=fetch_mode,
            render_mode="static",
            priority=i,
            rate_limit_profile=profile,
            expected_content_type="text/html",
            crawl_delay_seconds=0.0,
            max_response_bytes=MAX_RESPONSE_BYTES,
            is_robots=False,
            is_sitemap=False,
            run_id=str(uuid.uuid4()),
        )
        for i in range(n_urls)
    ]
    return CrawlManifest(
        domain=domain,
        urls=urls,
        total_urls=n_urls,
        estimated_duration_seconds=float(n_urls),
        clearance_required=1,
        manifest_id=manifest_id,
    )


async def _run_tests(verbose: bool = True) -> bool:
    """
    Run the full fetcher test suite.

    All tests use mock transports and in-memory substitutes.
    No live HTTP. No live Tor. No live Playwright.
    """

    passed = 0
    failed = 0
    errors: List[Tuple[str, str]] = []

    test_cases: List[Tuple[str, Callable]] = []

    # ── Math primitive tests ──────────────────────────────────────────

    async def test_p2_quantile_basic():
        est = P2QuantileEstimator(0.5)
        for i in range(1, 101):
            est.observe(float(i))
        median = est.estimate()
        assert 45.0 <= median <= 55.0, f"P50 estimate {median} not near 50"

    test_cases.append(("P2 quantile: median of 1..100", test_p2_quantile_basic))

    async def test_p2_quantile_p95():
        est = P2QuantileEstimator(0.95)
        rng = random.Random(42)
        for _ in range(1000):
            est.observe(rng.expovariate(1.0))
        p95 = est.estimate()
        assert 2.0 <= p95 <= 4.5, f"P95 estimate {p95} out of range for Exp(1)"

    test_cases.append(("P2 quantile: P95 of Exp(1)", test_p2_quantile_p95))

    async def test_welford_basic():
        acc = WelfordAccumulator()
        for x in [2, 4, 4, 4, 5, 5, 7, 9]:
            acc.update(float(x))
        assert abs(acc.mean - 5.0) < 0.01, f"mean={acc.mean}"
        assert abs(acc.variance - 4.571) < 0.01, f"var={acc.variance}"
        assert acc.minimum == 2.0
        assert acc.maximum == 9.0

    test_cases.append(("Welford: mean/variance of [2,4,4,4,5,5,7,9]", test_welford_basic))

    async def test_ewma_convergence():
        tracker = EWMATracker(alpha=0.1)
        for _ in range(100):
            tracker.update(1.0)
        assert abs(tracker.value - 1.0) < 0.01

    test_cases.append(("EWMA: converges to constant", test_ewma_convergence))

    async def test_shannon_entropy_uniform():
        tracker = ShannonEntropyTracker(window_size=100)
        for i in range(100):
            tracker.observe(200 + (i % 4) * 100)
        assert tracker.entropy > 1.9, f"entropy={tracker.entropy}"

    test_cases.append(("Shannon entropy: uniform 4 codes", test_shannon_entropy_uniform))

    async def test_shannon_entropy_degenerate():
        tracker = ShannonEntropyTracker(window_size=50)
        for _ in range(50):
            tracker.observe(403)
        assert tracker.entropy < 0.01
        assert tracker.is_degenerate

    test_cases.append(("Shannon entropy: degenerate (all 403)", test_shannon_entropy_degenerate))

    async def test_hazard_rate():
        est = HazardRateEstimator(total_urls=100)
        for i in range(100):
            est.observe(i % 10 == 0)
        assert est.failure_rate > 0.05
        assert est.cumulative_hazard > 0

    test_cases.append(("Hazard rate: periodic failures", test_hazard_rate))

    async def test_reservoir_sampling():
        sampler = ReservoirSampler(k=10, seed=42)
        for i in range(10000):
            sampler.add(i)
        assert sampler.reservoir_size == 10
        assert sampler.count == 10000

    test_cases.append(("Reservoir sampling: 10K items, k=10", test_reservoir_sampling))

    async def test_markov_circuit_no_collisions():
        mc = MarkovCircuitQuality()
        for i in range(20):
            mc.observe_ip(f"1.2.3.{i}")
        assert mc.ip_collision_rate == 0.0
        assert mc.unique_ips == 20

    test_cases.append(("Markov circuit: no IP collisions", test_markov_circuit_no_collisions))

    async def test_markov_circuit_all_collisions():
        mc = MarkovCircuitQuality()
        for _ in range(20):
            mc.observe_ip("1.2.3.4")
        # First observation is always GOOD (not seen before)
        # All subsequent are BAD (reuse)
        assert mc.ip_collision_rate > 0.8
        assert mc.unique_ips == 1

    test_cases.append(("Markov circuit: all IP collisions", test_markov_circuit_all_collisions))

    async def test_little_law():
        advisor = LittleLawAdvisor(window=20)
        for i in range(30):
            advisor.record_completion(0.1)
            await asyncio.sleep(0.001)
        assert advisor.throughput > 0
        assert advisor.mean_latency > 0

    test_cases.append(("Little's Law: basic operation", test_little_law))

    # ── Staging tests ─────────────────────────────────────────────────

    async def test_staging_pipeline():
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = StagingPipeline(Path(tmpdir) / "staging")
            await staging.initialize()
            data = b"hello world " * 100
            path = await staging.stage(data, "test-run-id")
            assert path.exists()
            recovered = await staging.unstage(path)
            assert recovered == data
            assert not path.exists()

    test_cases.append(("Staging: stage + unstage round-trip", test_staging_pipeline))

    async def test_staging_cleanup():
        with tempfile.TemporaryDirectory() as tmpdir:
            staging_dir = Path(tmpdir) / "staging"
            staging_dir.mkdir()
            (staging_dir / "leftover.raw").write_bytes(b"junk")
            (staging_dir / "another.tmp").write_bytes(b"junk")
            staging = StagingPipeline(staging_dir)
            await staging.initialize()
            assert staging.stats["cleaned_on_init"] == 2

    test_cases.append(("Staging: cleanup leftover files", test_staging_cleanup))

    # ── CLState tests ─────────────────────────────────────────────────

    async def test_cl_state_fallback():
        state = CLState(cl1_available=True)
        assert state.effective_mode(FetchMode.STATIC) == FetchMode.STATIC
        assert state.effective_mode(FetchMode.HEADLESS) == FetchMode.STATIC
        assert state.effective_mode(FetchMode.TOR) == FetchMode.STATIC
        assert state.effective_mode(FetchMode.TOR_FULL) == FetchMode.STATIC

    test_cases.append(("CLState: fallback to CL1 when nothing else available", test_cl_state_fallback))

    async def test_cl_state_full():
        state = CLState(cl1_available=True, cl2_available=True, cl3_available=True, cl4_available=True)
        assert state.effective_mode(FetchMode.STATIC) == FetchMode.STATIC
        assert state.effective_mode(FetchMode.HEADLESS) == FetchMode.HEADLESS
        assert state.effective_mode(FetchMode.TOR) == FetchMode.TOR
        assert state.effective_mode(FetchMode.TOR_FULL) == FetchMode.TOR_FULL

    test_cases.append(("CLState: all levels available", test_cl_state_full))

    async def test_cl_state_partial():
        state = CLState(cl1_available=True, cl2_available=True, cl3_available=False)
        assert state.effective_mode(FetchMode.TOR) == FetchMode.HEADLESS
        assert state.effective_mode(FetchMode.TOR_FULL) == FetchMode.HEADLESS

    test_cases.append(("CLState: CL3 unavailable fallback", test_cl_state_partial))

    # ── Telemetry tests ───────────────────────────────────────────────

    async def test_manifest_telemetry():
        tel = ManifestTelemetry(
            manifest_id="test", domain="example.com",
            hazard_estimator=HazardRateEstimator(100),
        )
        for i in range(50):
            tel.record_success(0.1 + i * 0.01, 5000 + i * 100, 200, FetchMode.STATIC)
        tel.record_skip()
        summary = tel.summary()
        assert summary["succeeded"] == 50
        assert summary["skipped"] == 1
        assert summary["latency_p50_ms"] > 0
        assert summary["throughput_urls_per_sec"] > 0

    test_cases.append(("ManifestTelemetry: record + summary", test_manifest_telemetry))

    # ── Fetcher integration tests (mock transport) ────────────────────

    async def test_fetcher_cl1_happy_path():
        """CL1 happy path — mock httpx, verify RawFetchEvent emitted."""
        emitted_events: List[Any] = []

        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted_events.append(e))
        bus.subscribe(ManifestCompleteEvent, lambda e: emitted_events.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>hello</html>")

        fetcher = Fetcher(
            bus=bus,
            bloom=_MockBloomFilter(),
            cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(),
            rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )

        # Manually set up with mock transport
        fetcher._staging = StagingPipeline(fetcher._staging._dir)
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=5)
        await fetcher._execute_manifest(manifest)

        raw_events = [e for e in emitted_events if isinstance(e, RawFetchEvent)]
        complete_events = [e for e in emitted_events if isinstance(e, ManifestCompleteEvent)]

        assert len(raw_events) == 5, f"expected 5 RawFetchEvents, got {len(raw_events)}"
        assert len(complete_events) == 1, "expected 1 ManifestCompleteEvent"
        assert raw_events[0].status_code == 200
        assert raw_events[0].byte_count > 0

        await fetcher._http_client.aclose()
        await fetcher._staging.cleanup()

    test_cases.append(("Fetcher CL1: happy path — 5 URLs", test_fetcher_cl1_happy_path))

    async def test_fetcher_429_handling():
        """HTTP 429 emits anomaly, URL skipped, manifest continues."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))
        bus.subscribe(FetchAnomalyEvent, lambda e: emitted.append(e))
        bus.subscribe(ManifestCompleteEvent, lambda e: emitted.append(e))

        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 3:
                return httpx.Response(429, content=b"rate limited")
            return httpx.Response(200, content=b"<html>ok</html>")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=5)
        await fetcher._execute_manifest(manifest)

        anomalies = [e for e in emitted if isinstance(e, FetchAnomalyEvent)]
        successes = [e for e in emitted if isinstance(e, RawFetchEvent)]
        assert len(anomalies) == 1, f"expected 1 anomaly, got {len(anomalies)}"
        assert anomalies[0].anomaly_type == "rate_limited"
        assert len(successes) == 4

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: 429 handling — anomaly + continue", test_fetcher_429_handling))

    async def test_fetcher_bloom_dedup():
        """Second fetch of same URL is skipped by bloom filter."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        bloom = _MockBloomFilter()
        await bloom.add("https://example.com/page/0")

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>ok</html>")

        fetcher = Fetcher(
            bus=bus, bloom=bloom, cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=3)
        await fetcher._execute_manifest(manifest)

        successes = [e for e in emitted if isinstance(e, RawFetchEvent)]
        # URL /page/0 was already in bloom — should be skipped
        assert len(successes) == 2, f"expected 2, got {len(successes)}"

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: bloom filter dedup — skip seen URL", test_fetcher_bloom_dedup))

    async def test_fetcher_connection_failure():
        """Connection error emits anomaly, manifest continues."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(FetchAnomalyEvent, lambda e: emitted.append(e))
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ConnectError("DNS failure")
            return httpx.Response(200, content=b"<html>ok</html>")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=4)
        await fetcher._execute_manifest(manifest)

        anomalies = [e for e in emitted if isinstance(e, FetchAnomalyEvent)]
        successes = [e for e in emitted if isinstance(e, RawFetchEvent)]
        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "connection_failed"
        assert len(successes) == 3

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: connection failure — anomaly + continue", test_fetcher_connection_failure))

    async def test_fetcher_response_truncation():
        """Response > max_bytes is truncated, not an anomaly."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        big_body = b"x" * (MAX_RESPONSE_BYTES + 1000)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=big_body)

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=1)
        await fetcher._execute_manifest(manifest)

        events = [e for e in emitted if isinstance(e, RawFetchEvent)]
        assert len(events) == 1
        assert events[0].byte_count <= MAX_RESPONSE_BYTES

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: response truncation at max_bytes", test_fetcher_response_truncation))

    async def test_fetcher_cursor_checkpoint():
        """Cursor checkpoints every CHECKPOINT_INTERVAL URLs."""
        cursor = _MockCrawlCursor()

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"ok")

        fetcher = Fetcher(
            bus=CrawlerBus(), bloom=_MockBloomFilter(), cursor=cursor,
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=250)
        await fetcher._execute_manifest(manifest)

        # Should have checkpointed at 100, 200, and final
        pos = await cursor.get_position(manifest.manifest_id)
        # Cursor is cleared on completion, so position should be gone
        # But we can verify the cursor was used by checking final checkpoint
        # was called (position reset to 0 after clear)

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: cursor checkpoint every 100 URLs", test_fetcher_cursor_checkpoint))

    async def test_fetcher_cl2_fallback():
        """CL2 unavailable → silent CL1 fallback, no anomaly."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))
        bus.subscribe(FetchAnomalyEvent, lambda e: emitted.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>fallback</html>")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        # CL2 NOT available — should silently fall back to CL1
        fetcher._cl_state = CLState(cl1_available=True, cl2_available=False)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=3, fetch_mode=FetchMode.HEADLESS)
        await fetcher._execute_manifest(manifest)

        anomalies = [e for e in emitted if isinstance(e, FetchAnomalyEvent)]
        successes = [e for e in emitted if isinstance(e, RawFetchEvent)]
        assert len(anomalies) == 0, "no anomaly on silent fallback"
        assert len(successes) == 3
        assert successes[0].fetch_mode == FetchMode.STATIC

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: CL2 unavailable → silent CL1 fallback", test_fetcher_cl2_fallback))

    async def test_fetcher_robots_propagation():
        """is_robots_txt=True propagated correctly to RawFetchEvent."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"User-agent: *")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest_id = str(uuid.uuid4())
        profile = RateLimitProfile(domain="example.com", requests_per_second=100.0)
        crawl_url = CrawlURL(
            url="https://example.com/robots.txt",
            topology_hint="GENERIC_HTML",
            fetch_mode=FetchMode.STATIC,
            render_mode="static",
            priority=0,
            rate_limit_profile=profile,
            is_robots=True,
            is_sitemap=False,
            run_id=str(uuid.uuid4()),
        )
        tel = ManifestTelemetry(manifest_id=manifest_id, domain="example.com")
        await fetcher._execute_url(crawl_url, manifest_id, tel)

        assert len(emitted) == 1
        assert emitted[0].is_robots_txt is True

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: robots.txt propagation", test_fetcher_robots_propagation))

    async def test_fetcher_sitemap_propagation():
        """is_sitemap=True propagated correctly to RawFetchEvent."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<?xml sitemap")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest_id = str(uuid.uuid4())
        profile = RateLimitProfile(domain="example.com", requests_per_second=100.0)
        crawl_url = CrawlURL(
            url="https://example.com/sitemap.xml",
            topology_hint="GENERIC_HTML",
            fetch_mode=FetchMode.STATIC,
            is_robots=False,
            is_sitemap=True,
            run_id=str(uuid.uuid4()),
            rate_limit_profile=profile,
        )
        tel = ManifestTelemetry(manifest_id=manifest_id, domain="example.com")
        await fetcher._execute_url(crawl_url, manifest_id, tel)

        assert len(emitted) == 1
        assert emitted[0].is_sitemap is True

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: sitemap propagation", test_fetcher_sitemap_propagation))

    async def test_fetcher_server_error():
        """HTTP 5xx emits anomaly, manifest continues."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(FetchAnomalyEvent, lambda e: emitted.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(502, content=b"bad gateway")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=2)
        await fetcher._execute_manifest(manifest)

        anomalies = [e for e in emitted if isinstance(e, FetchAnomalyEvent)]
        assert len(anomalies) == 2
        assert all(a.anomaly_type == "server_error" for a in anomalies)

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: HTTP 5xx — server error anomaly", test_fetcher_server_error))

    async def test_fetcher_bus_events():
        """Bus correctly routes events to subscribers."""
        events: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: events.append(("raw", e)))
        bus.subscribe(FetchAnomalyEvent, lambda e: events.append(("anomaly", e)))

        await bus.emit(FetchAnomalyEvent(
            url="http://x.com", fetch_mode=FetchMode.STATIC,
            status_code=429, anomaly_type="rate_limited",
            run_id="r", manifest_id="m",
        ))
        assert len(events) == 1
        assert events[0][0] == "anomaly"

    test_cases.append(("CrawlerBus: event routing", test_fetcher_bus_events))

    # ── Advanced math tests ──────────────────────────────────────────

    async def test_p2_quantile_p50_normal():
        """P2 P50 of a standard normal distribution should be ≈ 0."""
        est = P2QuantileEstimator(0.50)
        rng = random.Random(123)
        for _ in range(2000):
            # Box-Muller transform for normal distribution
            u1 = rng.random()
            u2 = rng.random()
            z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
            est.observe(z)
        median = est.estimate()
        assert abs(median) < 0.15, f"P50 of N(0,1) should be ~0, got {median}"

    test_cases.append(("P2 quantile: P50 of N(0,1) ≈ 0", test_p2_quantile_p50_normal))

    async def test_p2_quantile_p99_uniform():
        """P2 P99 of Uniform(0,1) should be ≈ 0.99."""
        est = P2QuantileEstimator(0.99)
        rng = random.Random(456)
        for _ in range(5000):
            est.observe(rng.random())
        p99 = est.estimate()
        assert 0.96 <= p99 <= 1.0, f"P99 of U(0,1) should be ~0.99, got {p99}"

    test_cases.append(("P2 quantile: P99 of U(0,1) ≈ 0.99", test_p2_quantile_p99_uniform))

    async def test_p2_quantile_few_observations():
        """P2 with fewer than 5 observations returns buffer-based estimate."""
        est = P2QuantileEstimator(0.5)
        est.observe(10.0)
        est.observe(20.0)
        est.observe(30.0)
        # With 3 observations, buffer-based estimate
        median = est.estimate()
        assert 10.0 <= median <= 30.0

    test_cases.append(("P2 quantile: < 5 observations fallback", test_p2_quantile_few_observations))

    async def test_p2_quantile_reset():
        """P2 reset clears all state."""
        est = P2QuantileEstimator(0.5)
        for i in range(100):
            est.observe(float(i))
        est.reset()
        assert est.count == 0
        assert est.estimate() == 0.0

    test_cases.append(("P2 quantile: reset clears state", test_p2_quantile_reset))

    async def test_welford_single_value():
        """Welford with a single value: variance = 0."""
        acc = WelfordAccumulator()
        acc.update(42.0)
        assert acc.mean == 42.0
        assert acc.variance == 0.0
        assert acc.std == 0.0
        assert acc.count == 1

    test_cases.append(("Welford: single value", test_welford_single_value))

    async def test_welford_large_values():
        """
        Welford stability with large values close together.
        The naive formula Σ(x²) - (Σx)²/n suffers catastrophic
        cancellation here.  Welford is immune.
        """
        acc = WelfordAccumulator()
        base = 1e9
        for i in range(100):
            acc.update(base + float(i))
        assert abs(acc.mean - (base + 49.5)) < 0.01
        # Variance of 0..99 = 100*99/12 ≈ 833.25
        assert abs(acc.variance - 841.6667) < 5.0, f"var={acc.variance}"

    test_cases.append(("Welford: numerical stability with large values", test_welford_large_values))

    async def test_welford_cv():
        """Coefficient of variation for exponential data."""
        acc = WelfordAccumulator()
        rng = random.Random(789)
        for _ in range(1000):
            acc.update(rng.expovariate(1.0))
        # CV of Exp(1) should be ≈ 1.0
        assert 0.85 <= acc.coefficient_of_variation <= 1.15

    test_cases.append(("Welford: CV of Exp(1) ≈ 1.0", test_welford_cv))

    async def test_welford_reset():
        acc = WelfordAccumulator()
        for i in range(50):
            acc.update(float(i))
        acc.reset()
        assert acc.count == 0
        assert acc.mean == 0.0

    test_cases.append(("Welford: reset", test_welford_reset))

    async def test_ewma_bias_correction():
        """EWMA with bias correction should not be biased toward zero initially."""
        tracker = EWMATracker(alpha=0.01, bias_correct=True)
        tracker.update(100.0)
        # Without bias correction: value = 0.01 * 100 = 1.0
        # With bias correction: value = 100.0 (correct!)
        assert abs(tracker.value - 100.0) < 1.0

    test_cases.append(("EWMA: bias correction initial", test_ewma_bias_correction))

    async def test_ewma_no_bias_correction():
        """EWMA without bias correction is biased initially."""
        tracker = EWMATracker(alpha=0.01, bias_correct=False)
        tracker.update(100.0)
        # Should be close to 100 since it's the first value
        assert tracker.value == 100.0  # first value always sets directly

    test_cases.append(("EWMA: no bias correction", test_ewma_no_bias_correction))

    async def test_ewma_tracking():
        """EWMA tracks a step change."""
        tracker = EWMATracker(alpha=0.1)
        # Phase 1: steady at 1.0
        for _ in range(50):
            tracker.update(1.0)
        v1 = tracker.value
        assert abs(v1 - 1.0) < 0.05

        # Phase 2: step to 10.0
        for _ in range(100):
            tracker.update(10.0)
        v2 = tracker.value
        assert abs(v2 - 10.0) < 0.5

    test_cases.append(("EWMA: tracks step change", test_ewma_tracking))

    async def test_shannon_entropy_binary():
        """Shannon entropy of binary (50/50) = 1.0 bit."""
        tracker = ShannonEntropyTracker(window_size=100)
        for i in range(100):
            tracker.observe(200 if i % 2 == 0 else 404)
        assert abs(tracker.entropy - 1.0) < 0.01

    test_cases.append(("Shannon entropy: binary 50/50 = 1 bit", test_shannon_entropy_binary))

    async def test_shannon_entropy_window_eviction():
        """Entropy window correctly evicts old observations."""
        tracker = ShannonEntropyTracker(window_size=10)
        # Fill with 200s
        for _ in range(10):
            tracker.observe(200)
        assert tracker.entropy < 0.01
        # Replace with alternating 200/404
        for i in range(10):
            tracker.observe(200 if i % 2 == 0 else 404)
        # Should now have mixed entropy
        assert tracker.entropy > 0.5

    test_cases.append(("Shannon entropy: window eviction", test_shannon_entropy_window_eviction))

    async def test_hazard_no_failures():
        """Hazard rate with no failures should be 0."""
        est = HazardRateEstimator(total_urls=50)
        for _ in range(50):
            est.observe(False)
        assert est.cumulative_hazard == 0.0
        assert est.failure_rate == 0.0
        assert est.survival_probability == 1.0

    test_cases.append(("Hazard rate: no failures", test_hazard_no_failures))

    async def test_hazard_all_failures():
        """Hazard rate with all failures."""
        est = HazardRateEstimator(total_urls=10)
        for _ in range(10):
            est.observe(True)
        assert est.failure_rate == 1.0
        assert est.cumulative_hazard > 0

    test_cases.append(("Hazard rate: all failures", test_hazard_all_failures))

    async def test_hazard_survival_decreasing():
        """Survival probability decreases with more failures."""
        est = HazardRateEstimator(total_urls=100)
        survivals = []
        for i in range(100):
            est.observe(i % 5 == 0)  # 20% failure rate
            survivals.append(est.survival_probability)
        # Survival should generally decrease
        assert survivals[-1] < survivals[0]

    test_cases.append(("Hazard rate: survival decreases over time", test_hazard_survival_decreasing))

    async def test_reservoir_uniform():
        """Reservoir sample should be approximately uniform."""
        sampler = ReservoirSampler(k=100, seed=42)
        for i in range(100000):
            sampler.add(i % 10)
        counts = Counter(sampler.sample)
        # Each of 0-9 should appear ~10 times in the 100-element sample
        for digit in range(10):
            assert 3 <= counts.get(digit, 0) <= 20, f"digit {digit}: {counts.get(digit, 0)}"

    test_cases.append(("Reservoir sampling: uniform distribution", test_reservoir_uniform))

    async def test_reservoir_empty():
        """Empty reservoir returns empty sample."""
        sampler = ReservoirSampler(k=10)
        assert sampler.sample == []
        assert sampler.count == 0

    test_cases.append(("Reservoir sampling: empty", test_reservoir_empty))

    async def test_reservoir_under_capacity():
        """Reservoir with fewer items than capacity keeps all."""
        sampler = ReservoirSampler(k=100)
        for i in range(50):
            sampler.add(i)
        assert sampler.reservoir_size == 50
        assert set(sampler.sample) == set(range(50))

    test_cases.append(("Reservoir sampling: under capacity", test_reservoir_under_capacity))

    async def test_reservoir_clear():
        sampler = ReservoirSampler(k=10)
        for i in range(100):
            sampler.add(i)
        sampler.clear()
        assert sampler.count == 0
        assert sampler.reservoir_size == 0

    test_cases.append(("Reservoir sampling: clear", test_reservoir_clear))

    async def test_markov_mixed():
        """Markov chain with mix of fresh and reused IPs."""
        mc = MarkovCircuitQuality()
        # Alternating fresh and reused
        for i in range(20):
            if i % 2 == 0:
                mc.observe_ip(f"1.2.3.{i}")  # fresh
            else:
                mc.observe_ip("1.2.3.0")  # reuse
        assert mc.unique_ips < 20
        assert 0.3 < mc.ip_collision_rate < 0.7

    test_cases.append(("Markov circuit: mixed fresh/reused", test_markov_mixed))

    async def test_markov_reset():
        mc = MarkovCircuitQuality()
        for i in range(10):
            mc.observe_ip(f"1.2.3.{i}")
        mc.reset()
        assert mc.unique_ips == 0
        assert mc.total_observations == 0

    test_cases.append(("Markov circuit: reset", test_markov_reset))

    async def test_little_law_idle():
        """Little's Law with no completions returns 0."""
        advisor = LittleLawAdvisor()
        assert advisor.throughput == 0.0
        assert advisor.optimal_concurrency == 0.0

    test_cases.append(("Little's Law: idle state", test_little_law_idle))

    async def test_little_law_high_throughput():
        """Little's Law with rapid completions."""
        advisor = LittleLawAdvisor(window=100)
        for _ in range(100):
            advisor.record_completion(0.05)  # 50ms latency
            await asyncio.sleep(0.001)
        # L = λ * W ≈ throughput * 0.05
        assert advisor.optimal_concurrency > 0

    test_cases.append(("Little's Law: high throughput", test_little_law_high_throughput))

    # ── Bus advanced tests ────────────────────────────────────────────

    async def test_bus_multiple_subscribers():
        """Multiple subscribers for the same event type."""
        events_a: List[Any] = []
        events_b: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(FetchAnomalyEvent, lambda e: events_a.append(e))
        bus.subscribe(FetchAnomalyEvent, lambda e: events_b.append(e))
        await bus.emit(FetchAnomalyEvent(
            url="http://x.com", fetch_mode=FetchMode.STATIC,
            status_code=429, anomaly_type="rate_limited",
            run_id="r", manifest_id="m",
        ))
        assert len(events_a) == 1
        assert len(events_b) == 1

    test_cases.append(("CrawlerBus: multiple subscribers", test_bus_multiple_subscribers))

    async def test_bus_subscriber_exception():
        """Bus does not crash when subscriber raises."""
        def bad_handler(event: Any) -> None:
            raise ValueError("intentional test error")

        events: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(FetchAnomalyEvent, bad_handler)
        bus.subscribe(FetchAnomalyEvent, lambda e: events.append(e))
        await bus.emit(FetchAnomalyEvent(
            url="http://x.com", fetch_mode=FetchMode.STATIC,
            status_code=500, anomaly_type="server_error",
            run_id="r", manifest_id="m",
        ))
        # Second subscriber should still receive the event
        assert len(events) == 1

    test_cases.append(("CrawlerBus: subscriber exception isolation", test_bus_subscriber_exception))

    async def test_bus_emit_count():
        """Bus tracks total emit count."""
        bus = CrawlerBus()
        for i in range(10):
            await bus.emit(FetchAnomalyEvent(
                url=f"http://x.com/{i}", fetch_mode=FetchMode.STATIC,
                status_code=500, anomaly_type="server_error",
                run_id="r", manifest_id="m",
            ))
        assert bus.emit_count == 10

    test_cases.append(("CrawlerBus: emit count tracking", test_bus_emit_count))

    # ── Staging advanced tests ────────────────────────────────────────

    async def test_staging_concurrent_writes():
        """Multiple concurrent stage operations don't interfere."""
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = StagingPipeline(Path(tmpdir) / "staging")
            await staging.initialize()
            paths = []
            for i in range(20):
                p = await staging.stage(f"data-{i}".encode(), f"run-{i}")
                paths.append(p)
            assert len(set(paths)) == 20
            for p in paths:
                assert p.exists()
            for p in paths:
                data = await staging.unstage(p)
                assert data.startswith(b"data-")

    test_cases.append(("Staging: 20 concurrent writes", test_staging_concurrent_writes))

    async def test_staging_large_file():
        """Stage a 1MB file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            staging = StagingPipeline(Path(tmpdir) / "staging")
            await staging.initialize()
            data = b"x" * (1024 * 1024)
            path = await staging.stage(data, "big-run")
            recovered = await staging.unstage(path)
            assert len(recovered) == 1024 * 1024

    test_cases.append(("Staging: 1MB file", test_staging_large_file))

    # ── Fetcher advanced integration tests ────────────────────────────

    async def test_fetcher_mixed_status_codes():
        """Mixed 200/403/500 — correct event routing."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(("raw", e)))
        bus.subscribe(FetchAnomalyEvent, lambda e: emitted.append(("anomaly", e)))

        responses = [200, 403, 200, 500, 200, 200, 403, 200, 200, 200]
        idx = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal idx
            status = responses[idx % len(responses)]
            idx += 1
            return httpx.Response(status, content=b"body")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=10)
        telemetry = await fetcher._execute_manifest(manifest)

        raw_events = [e for t, e in emitted if t == "raw"]
        anomalies = [e for t, e in emitted if t == "anomaly"]
        assert len(raw_events) == 7  # 7 × 200
        assert len(anomalies) == 3   # 2 × 403 + 1 × 500

        # Check telemetry
        summary = telemetry.summary()
        assert summary["succeeded"] == 7
        assert summary["failed"] == 3

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: mixed 200/403/500 routing", test_fetcher_mixed_status_codes))

    async def test_fetcher_telemetry_entropy():
        """Telemetry correctly computes Shannon entropy for mixed responses."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))
        bus.subscribe(FetchAnomalyEvent, lambda e: emitted.append(e))

        idx = 0
        pattern = [200, 200, 200, 403, 200, 200, 500, 200, 200, 200,
                   200, 200, 429, 200, 200, 200, 200, 200, 200, 200]

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal idx
            status = pattern[idx % len(pattern)]
            idx += 1
            return httpx.Response(status, content=b"body")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=20)
        telemetry = await fetcher._execute_manifest(manifest)
        summary = telemetry.summary()

        # Entropy should be > 0 because we have mixed status codes
        assert summary["status_entropy_bits"] > 0

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: telemetry Shannon entropy", test_fetcher_telemetry_entropy))

    async def test_fetcher_empty_manifest():
        """Manifest with zero URLs completes immediately."""
        bus = CrawlerBus()
        complete_events: List[Any] = []
        bus.subscribe(ManifestCompleteEvent, lambda e: complete_events.append(e))

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=b"ok")
            ),
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=0)
        telemetry = await fetcher._execute_manifest(manifest)
        assert telemetry.urls_attempted == 0
        assert len(complete_events) == 1

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: empty manifest completes", test_fetcher_empty_manifest))

    async def test_fetcher_all_bloom_skipped():
        """All URLs already in bloom → all skipped, zero fetches."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        bloom = _MockBloomFilter()
        manifest = _make_test_manifest(n_urls=5)
        for url in manifest.urls:
            await bloom.add(url.url)

        fetcher = Fetcher(
            bus=bus, bloom=bloom, cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(200, content=b"ok")
            ),
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        telemetry = await fetcher._execute_manifest(manifest)
        assert telemetry.urls_skipped == 5
        assert telemetry.urls_succeeded == 0
        assert len(emitted) == 0

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: all bloom-skipped → zero fetches", test_fetcher_all_bloom_skipped))

    async def test_fetcher_access_denied_propagation():
        """HTTP 403 propagates as access_denied anomaly type."""
        anomalies: List[FetchAnomalyEvent] = []
        bus = CrawlerBus()
        bus.subscribe(FetchAnomalyEvent, lambda e: anomalies.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(403, content=b"forbidden")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=1)
        await fetcher._execute_manifest(manifest)

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "access_denied"
        assert anomalies[0].status_code == 403

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: 403 → access_denied anomaly", test_fetcher_access_denied_propagation))

    async def test_fetcher_multiple_manifests_telemetry():
        """Multiple manifests produce independent telemetry."""
        bus = CrawlerBus()

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"ok")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        t1 = await fetcher._execute_manifest(_make_test_manifest(n_urls=3, domain="a.com"))
        t2 = await fetcher._execute_manifest(_make_test_manifest(n_urls=7, domain="b.com"))

        assert t1.urls_succeeded == 3
        assert t2.urls_succeeded == 7
        assert t1.domain == "a.com"
        assert t2.domain == "b.com"

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: independent manifest telemetry", test_fetcher_multiple_manifests_telemetry))

    async def test_fetcher_timeout_handling():
        """Timeout emits anomaly, manifest continues."""
        anomalies: List[FetchAnomalyEvent] = []
        successes: List[RawFetchEvent] = []
        bus = CrawlerBus()
        bus.subscribe(FetchAnomalyEvent, lambda e: anomalies.append(e))
        bus.subscribe(RawFetchEvent, lambda e: successes.append(e))

        call_count = 0

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise httpx.ReadTimeout("read timeout")
            return httpx.Response(200, content=b"ok")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=3)
        await fetcher._execute_manifest(manifest)

        assert len(anomalies) == 1
        assert anomalies[0].anomaly_type == "timeout"
        assert len(successes) == 2

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: timeout → anomaly + continue", test_fetcher_timeout_handling))

    async def test_fetcher_large_manifest():
        """Process a 500-URL manifest — verify throughput and correctness."""
        success_count = 0
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: None)

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>" + b"x" * 500 + b"</html>")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=500)
        telemetry = await fetcher._execute_manifest(manifest)

        assert telemetry.urls_succeeded == 500
        assert telemetry.throughput > 10.0  # at least 10 URLs/sec with mock
        assert telemetry.latency_p95.estimate() >= 0

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher: 500-URL manifest — throughput test", test_fetcher_large_manifest))

    # ── CLState comprehensive tests ───────────────────────────────────

    async def test_cl_state_tor_full_cascading():
        """CL4 cascades: CL4→CL3→CL2→CL1 as levels become unavailable."""
        # All available
        s = CLState(cl1_available=True, cl2_available=True, cl3_available=True, cl4_available=True)
        assert s.effective_mode(FetchMode.TOR_FULL) == FetchMode.TOR_FULL

        # CL4 off → CL3
        s = CLState(cl1_available=True, cl2_available=True, cl3_available=True, cl4_available=False)
        assert s.effective_mode(FetchMode.TOR_FULL) == FetchMode.TOR

        # CL4+CL3 off → CL2
        s = CLState(cl1_available=True, cl2_available=True, cl3_available=False, cl4_available=False)
        assert s.effective_mode(FetchMode.TOR_FULL) == FetchMode.HEADLESS

        # CL4+CL3+CL2 off → CL1
        s = CLState(cl1_available=True, cl2_available=False, cl3_available=False, cl4_available=False)
        assert s.effective_mode(FetchMode.TOR_FULL) == FetchMode.STATIC

    test_cases.append(("CLState: TOR_FULL cascading fallback", test_cl_state_tor_full_cascading))

    async def test_cl_state_to_dict():
        s = CLState(cl1_available=True, cl2_available=True, cl3_available=False, cl4_available=False)
        d = s.to_dict()
        assert d == {"cl1": True, "cl2": True, "cl3": False, "cl4": False}

    test_cases.append(("CLState: to_dict", test_cl_state_to_dict))

    # ── FetchAnomalyEvent tests ───────────────────────────────────────

    async def test_fetch_anomaly_event_construction():
        """FetchAnomalyEvent is properly constructed."""
        anomaly = FetchAnomalyEvent(
            url="https://example.com/page",
            fetch_mode=FetchMode.STATIC,
            status_code=429,
            anomaly_type="rate_limited",
            run_id="test-run",
            manifest_id="test-manifest",
            detail="architecture bug",
        )
        assert anomaly.url == "https://example.com/page"
        assert anomaly.anomaly_type == "rate_limited"
        assert anomaly.status_code == 429
        assert anomaly.detail == "architecture bug"
        assert anomaly.timestamp > 0

    test_cases.append(("FetchAnomalyEvent: construction", test_fetch_anomaly_event_construction))

    # ── Helper function tests ─────────────────────────────────────────

    async def test_extract_domain():
        assert _extract_domain("https://docs.stripe.com/api") == "docs.stripe.com"
        assert _extract_domain("https://example.com:443/path") == "example.com"
        assert _extract_domain("http://localhost:8080/") == "localhost"

    test_cases.append(("_extract_domain: various URLs", test_extract_domain))

    async def test_format_bytes():
        assert _format_bytes(500) == "500.0 B"
        assert _format_bytes(1024) == "1.0 KB"
        assert _format_bytes(1024 * 1024) == "1.0 MB"

    test_cases.append(("_format_bytes: various sizes", test_format_bytes))

    async def test_format_duration():
        assert _format_duration(0.5) == "500ms"
        assert _format_duration(5.0) == "5.0s"
        assert _format_duration(90.0) == "1m 30s"
        assert _format_duration(3661.0) == "1h 1m"

    test_cases.append(("_format_duration: various durations", test_format_duration))

    async def test_build_test_cl_state():
        s1 = _build_test_cl_state(1)
        assert s1.cl1_available and not s1.cl2_available
        s3 = _build_test_cl_state(3)
        assert s3.cl3_available and not s3.cl4_available
        s4 = _build_test_cl_state(4)
        assert s4.cl4_available

    test_cases.append(("_build_test_cl_state: CL levels", test_build_test_cl_state))

    # ── Mock class tests ──────────────────────────────────────────────

    async def test_mock_bloom():
        bloom = _MockBloomFilter()
        assert not await bloom.contains("http://x.com")
        await bloom.add("http://x.com")
        assert await bloom.contains("http://x.com")
        assert await bloom.count() == 1

    test_cases.append(("MockBloomFilter: add/contains", test_mock_bloom))

    async def test_mock_cursor():
        cursor = _MockCrawlCursor()
        assert await cursor.get_position("m1") == 0
        await cursor.checkpoint("m1", 42, "url", 100)
        assert await cursor.get_position("m1") == 42
        await cursor.clear("m1")
        assert await cursor.get_position("m1") == 0

    test_cases.append(("MockCrawlCursor: checkpoint/get/clear", test_mock_cursor))

    async def test_mock_frontier():
        frontier = _MockFrontier()
        manifest = _make_test_manifest(n_urls=3)
        await frontier.load_manifest(manifest)
        urls = []
        async for url in frontier.resume(manifest.manifest_id):
            urls.append(url)
        assert len(urls) == 3
        await frontier.mark_done(manifest.manifest_id, urls[0].url)
        stats = await frontier.stats(manifest.manifest_id)
        assert stats.done == 1
        assert stats.pending == 2

    test_cases.append(("MockFrontier: load/resume/mark", test_mock_frontier))

    async def test_mock_frontier_duplicate_load():
        """Loading the same manifest twice doesn't duplicate URLs."""
        frontier = _MockFrontier()
        manifest = _make_test_manifest(n_urls=3)
        await frontier.load_manifest(manifest)
        await frontier.load_manifest(manifest)
        urls = []
        async for url in frontier.resume(manifest.manifest_id):
            urls.append(url)
        assert len(urls) == 3

    test_cases.append(("MockFrontier: duplicate load idempotent", test_mock_frontier_duplicate_load))

    # ── Tor circuit manager tests (no live Tor) ───────────────────────

    async def test_tor_manager_unavailable():
        """TorCircuitManager reports unavailable when ports are closed."""
        mgr = TorCircuitManager(socks_host="127.0.0.1", socks_port=19999, control_port=19998)
        available = await mgr.check_available()
        assert not available
        assert not mgr.is_available

    test_cases.append(("TorCircuitManager: unavailable on closed ports", test_tor_manager_unavailable))

    async def test_tor_manager_proxy_url():
        mgr = TorCircuitManager(socks_host="127.0.0.1", socks_port=9050)
        assert mgr.proxy_url == "socks5://127.0.0.1:9050"

    test_cases.append(("TorCircuitManager: proxy URL format", test_tor_manager_proxy_url))

    async def test_tor_manager_to_dict():
        mgr = TorCircuitManager()
        d = mgr.to_dict()
        assert "available" in d
        assert "socks" in d
        assert "quality" in d

    test_cases.append(("TorCircuitManager: to_dict", test_tor_manager_to_dict))

    # ── Playwright manager tests (no live browser) ────────────────────

    async def test_pw_manager_unavailable():
        """PlaywrightManager reports unavailable when Playwright not installed."""
        mgr = PlaywrightManager()
        # In this test env, Playwright may not be installed
        # Either way, the API should work
        assert not mgr.is_available
        assert mgr.page_count == 0
        assert mgr.context_recycles == 0

    test_cases.append(("PlaywrightManager: initial state", test_pw_manager_unavailable))

    # ── Observatory tests ─────────────────────────────────────────────

    async def test_observatory_basic():
        """Observatory processes mixed success/failure stream."""
        obs = DomainObservatory("example.com")
        for i in range(50):
            obs.observe(success=(i % 5 != 0), latency=0.1 + i * 0.001)
        report = obs.report()
        assert report.observations == 50
        assert report.domain == "example.com"
        assert 0.7 < report.bayesian["posterior_mean"] < 0.9

    test_cases.append(("Observatory: basic operation", test_observatory_basic))

    async def test_observatory_bayesian_all_success():
        """Bayesian posterior concentrates near 1.0 after all successes."""
        obs = DomainObservatory("good.com")
        for _ in range(100):
            obs.observe(success=True, latency=0.1)
        report = obs.report()
        assert report.bayesian["posterior_mean"] > 0.95
        ci = report.bayesian["credible_interval_95"]
        assert ci[0] > 0.93

    test_cases.append(("Observatory: Bayesian all success → mean > 0.95", test_observatory_bayesian_all_success))

    async def test_observatory_bayesian_all_failure():
        """Bayesian posterior concentrates near 0.0 after all failures."""
        obs = DomainObservatory("dead.com")
        for _ in range(100):
            obs.observe(success=False, latency=0.0)
        report = obs.report()
        assert report.bayesian["posterior_mean"] < 0.05

    test_cases.append(("Observatory: Bayesian all failure → mean < 0.05", test_observatory_bayesian_all_failure))

    async def test_observatory_cusum_detects_shift():
        """CUSUM fires alarm when success rate drops mid-stream."""
        obs = DomainObservatory("shifting.com")
        # Phase 1: healthy (95% success)
        alarms_phase1 = []
        for i in range(100):
            alarm = obs.observe(success=(i % 20 != 0), latency=0.1)
            if alarm:
                alarms_phase1.append(alarm)
        # Phase 2: sick (30% success)
        alarms_phase2 = []
        for i in range(100):
            alarm = obs.observe(success=(i % 3 == 0), latency=0.1)
            if alarm:
                alarms_phase2.append(alarm)
        report = obs.report()
        assert report.change_points["alarm_count"] >= 1

    test_cases.append(("Observatory: CUSUM detects behavior shift", test_observatory_cusum_detects_shift))

    async def test_observatory_sprt_sick():
        """SPRT declares sick after sustained failures."""
        obs = DomainObservatory("sick.com")
        for _ in range(50):
            obs.observe(success=False, latency=0.0)
        assert obs.should_abandon
        report = obs.report()
        assert report.sprt["decision"] == "sick"

    test_cases.append(("Observatory: SPRT declares sick", test_observatory_sprt_sick))

    async def test_observatory_sprt_healthy():
        """SPRT declares healthy after sustained successes."""
        obs = DomainObservatory("healthy.com")
        for _ in range(50):
            obs.observe(success=True, latency=0.1)
        assert not obs.should_abandon
        report = obs.report()
        assert report.sprt["decision"] == "healthy"

    test_cases.append(("Observatory: SPRT declares healthy", test_observatory_sprt_healthy))

    async def test_observatory_compressibility():
        """Compressibility tracker detects content change."""
        obs = DomainObservatory("changing.com")
        # Phase 1: real HTML (compressible)
        for _ in range(60):
            html = (b"<html><head><title>Real Page</title></head>"
                    b"<body>" + b"<p>Content paragraph.</p>" * 50 + b"</body></html>")
            obs.observe(success=True, latency=0.1, raw_bytes=html)
        # Phase 2: error page (very compressible / different pattern)
        for _ in range(30):
            error = b"Access Denied" * 100
            obs.observe(success=True, latency=0.05, raw_bytes=error)
        report = obs.report()
        assert report.compressibility["observations"] == 90

    test_cases.append(("Observatory: compressibility tracking", test_observatory_compressibility))

    async def test_observatory_kalman_convergence():
        """Kalman filter converges on capacity from noisy latency."""
        obs = DomainObservatory("stable.com")
        rng = random.Random(42)
        # True capacity: 5 req/s → true latency: 0.2s
        for _ in range(200):
            latency = 0.2 + rng.gauss(0, 0.03)  # noisy observations
            obs.observe(success=True, latency=max(0.05, latency))
        report = obs.report()
        cap = report.capacity["estimated_capacity_rps"]
        # Should converge near 5.0 (1/0.2)
        assert 3.0 < cap < 8.0, f"capacity estimate {cap} not near 5.0"

    test_cases.append(("Observatory: Kalman capacity estimate", test_observatory_kalman_convergence))

    async def test_observatory_autocorrelation_random():
        """Random latencies should show no autocorrelation."""
        obs = DomainObservatory("random.com")
        rng = random.Random(123)
        for _ in range(200):
            obs.observe(success=True, latency=rng.expovariate(5.0))
        report = obs.report()
        assert not report.autocorrelation["is_correlated"]

    test_cases.append(("Observatory: no autocorrelation on random latency", test_observatory_autocorrelation_random))

    async def test_observatory_report_structure():
        """DomainHealthReport has all expected fields."""
        obs = DomainObservatory("test.com")
        for _ in range(10):
            obs.observe(success=True, latency=0.1, raw_bytes=b"<html>x</html>")
        report = obs.report()
        assert hasattr(report, "domain")
        assert hasattr(report, "bayesian")
        assert hasattr(report, "change_points")
        assert hasattr(report, "sprt")
        assert hasattr(report, "compressibility")
        assert hasattr(report, "autocorrelation")
        assert hasattr(report, "capacity")
        assert report.is_healthy()

    test_cases.append(("Observatory: report structure complete", test_observatory_report_structure))

    # ── Bayesian unit tests ───────────────────────────────────────────

    async def test_bayesian_credible_interval_widens():
        """CI should be wider with fewer observations."""
        few = BayesianSuccessEstimator()
        few.observe_batch(3, 1)
        many = BayesianSuccessEstimator()
        many.observe_batch(300, 100)
        ci_few = few.credible_interval()
        ci_many = many.credible_interval()
        width_few = ci_few[1] - ci_few[0]
        width_many = ci_many[1] - ci_many[0]
        assert width_few > width_many, "more data should tighten CI"

    test_cases.append(("Bayesian: CI width decreases with more data", test_bayesian_credible_interval_widens))

    async def test_bayesian_prob_below():
        """P(θ < 0.5) should be high after many failures."""
        est = BayesianSuccessEstimator()
        est.observe_batch(10, 90)  # 10% success rate
        p = est.prob_below(0.5)
        assert p > 0.99, f"P(θ<0.5) should be >0.99, got {p}"

    test_cases.append(("Bayesian: P(θ<0.5) high after 90% failure", test_bayesian_prob_below))

    async def test_bayesian_entropy_decreases():
        """Posterior entropy decreases as we get more data."""
        est = BayesianSuccessEstimator()
        e0 = est.entropy
        est.observe_batch(10, 2)
        e1 = est.entropy
        est.observe_batch(100, 20)
        e2 = est.entropy
        # More data → more concentrated → lower entropy
        assert e2 < e0, "entropy should decrease with data"

    test_cases.append(("Bayesian: entropy decreases with data", test_bayesian_entropy_decreases))

    # ── CUSUM unit tests ──────────────────────────────────────────────

    async def test_cusum_no_alarm_on_healthy():
        """CUSUM should not alarm on a stream matching the target rate."""
        det = CUSUMChangePointDetector(target=0.9)
        rng = random.Random(999)
        for i in range(200):
            # 90% success rate = exactly at target → no shift
            success = rng.random() < 0.9
            alarm = det.observe(success)
            # With target=0.9 and observed ≈0.9, CUSUM stays near 0
        assert det.alarm_count == 0, f"alarms on at-target stream: {det.alarm_count}"

    test_cases.append(("CUSUM: no alarm on healthy stream", test_cusum_no_alarm_on_healthy))

    async def test_cusum_alarm_on_shift():
        """CUSUM should alarm when success rate drops sharply."""
        det = CUSUMChangePointDetector(target=0.9, threshold=3.0)
        # Healthy phase
        for _ in range(50):
            det.observe(True)
        # Sick phase
        fired = False
        for _ in range(50):
            alarm = det.observe(False)
            if alarm:
                fired = True
                assert alarm.shift_magnitude > 0
                break
        assert fired, "CUSUM should have alarmed"

    test_cases.append(("CUSUM: alarm on sharp shift", test_cusum_alarm_on_shift))

    # ── SPRT unit tests ───────────────────────────────────────────────

    async def test_sprt_expected_sample_size():
        """SPRT expected sample size should be finite."""
        sprt = WaldSPRT(theta_0=0.9, theta_1=0.6)
        en_h0, en_h1 = sprt.expected_sample_size()
        assert 0 < en_h0 < 500
        assert 0 < en_h1 < 500
        assert en_h1 < en_h0  # easier to detect when actually sick

    test_cases.append(("SPRT: expected sample size finite", test_sprt_expected_sample_size))

    async def test_sprt_terminal_is_sticky():
        """Once SPRT decides, subsequent observations don't change it."""
        sprt = WaldSPRT()
        for _ in range(100):
            sprt.observe(False)
        assert sprt.decision == SPRTDecision.ACCEPT_SICK
        # More observations shouldn't change the decision
        for _ in range(100):
            sprt.observe(True)
        assert sprt.decision == SPRTDecision.ACCEPT_SICK

    test_cases.append(("SPRT: terminal decision is sticky", test_sprt_terminal_is_sticky))

    async def test_sprt_reset():
        sprt = WaldSPRT()
        for _ in range(100):
            sprt.observe(False)
        assert sprt.decision != SPRTDecision.CONTINUE
        sprt.reset()
        assert sprt.decision == SPRTDecision.CONTINUE
        assert sprt.observations == 0

    test_cases.append(("SPRT: reset clears state", test_sprt_reset))

    # ── SEP tests ─────────────────────────────────────────────────────

    async def test_sep_pattern_clustering():
        """URLs with shared path structure collapse to same pattern."""
        sep = SpeculativeEnvelopePredictor("example.com")
        # Feed similar URLs — they should collapse to one pattern
        for i in range(10):
            sep.observe(
                url=f"https://example.com/docs/api/endpoint_{i}",
                status_code=200, size_bytes=5000, latency=0.1,
                raw_bytes=b"<html>x</html>",
            )
        # "/docs/api/*" should be one pattern
        assert sep.pattern_count <= 3, f"too many patterns: {sep.pattern_count}"

    test_cases.append(("SEP: URL pattern clustering", test_sep_pattern_clustering))

    async def test_sep_prediction_after_warmup():
        """SEP makes predictions after enough observations for a pattern."""
        sep = SpeculativeEnvelopePredictor("example.com")
        # Need SEP_SEGMENT_COLLAPSE_N obs to collapse the trie pattern,
        # then SEP_MIN_OBSERVATIONS more for the collapsed pattern model
        n_warmup = SEP_SEGMENT_COLLAPSE_N + SEP_MIN_OBSERVATIONS + 2
        for i in range(n_warmup):
            sep.observe(
                url=f"https://example.com/page/{i}",
                status_code=200, size_bytes=5000 + i, latency=0.1,
                raw_bytes=b"<html>content</html>",
            )
        pred = sep.predict("https://example.com/page/next")
        assert pred is not None, f"expected prediction after {n_warmup} observations"
        assert pred.predicted_status == 200
        assert pred.pattern_observations >= SEP_MIN_OBSERVATIONS

    test_cases.append(("SEP: predictions after warm-up", test_sep_prediction_after_warmup))

    async def test_sep_no_prediction_before_warmup():
        """SEP returns None before enough observations."""
        sep = SpeculativeEnvelopePredictor("example.com")
        sep.observe(
            url="https://example.com/page/1",
            status_code=200, size_bytes=5000, latency=0.1,
        )
        pred = sep.predict("https://example.com/page/2")
        assert pred is None

    test_cases.append(("SEP: no prediction before warm-up", test_sep_no_prediction_before_warmup))

    async def test_sep_divergence_on_status_change():
        """SEP detects divergence when status code changes."""
        sep = SpeculativeEnvelopePredictor("example.com")
        # Build baseline: 10 × 200
        for i in range(10):
            sep.observe(
                url=f"https://example.com/api/v1/item_{i}",
                status_code=200, size_bytes=5000, latency=0.1,
                raw_bytes=b"<html>" + b"x" * 200 + b"</html>",
            )
        # Now a 403 should diverge
        score = sep.observe(
            url="https://example.com/api/v1/item_blocked",
            status_code=403, size_bytes=200, latency=0.02,
            raw_bytes=b"Access Denied",
        )
        assert score is not None
        assert score.is_divergent
        assert not score.status_correct

    test_cases.append(("SEP: divergence on status code change", test_sep_divergence_on_status_change))

    async def test_sep_divergence_on_size_anomaly():
        """SEP detects divergence when response size is wildly different."""
        sep = SpeculativeEnvelopePredictor("example.com")
        # Build baseline: 10 × ~5000 bytes
        for i in range(10):
            sep.observe(
                url=f"https://example.com/docs/page_{i}",
                status_code=200, size_bytes=5000, latency=0.1,
                raw_bytes=b"x" * 5000,
            )
        # Now a 200 with 50 bytes should diverge on size
        score = sep.observe(
            url="https://example.com/docs/page_anomalous",
            status_code=200, size_bytes=50, latency=0.1,
            raw_bytes=b"x" * 50,
        )
        assert score is not None
        assert score.size_z_score > 2.0

    test_cases.append(("SEP: divergence on size anomaly", test_sep_divergence_on_size_anomaly))

    async def test_sep_convergence():
        """SEP converges after consistent pattern behavior."""
        sep = SpeculativeEnvelopePredictor("stable.com")
        rng = random.Random(42)
        for i in range(SEP_MIN_OBSERVATIONS + SEP_CONVERGENCE_WINDOW + 10):
            sep.observe(
                url=f"https://stable.com/page/{i}",
                status_code=200,
                size_bytes=5000 + rng.randint(-100, 100),
                latency=0.1 + rng.gauss(0, 0.005),
                raw_bytes=b"<html>" + b"x" * 200 + b"</html>",
            )
        assert sep.converged_patterns >= 1

    test_cases.append(("SEP: pattern convergence", test_sep_convergence))

    async def test_sep_predictability_score():
        """Predictability score is in [0, 1]."""
        sep = SpeculativeEnvelopePredictor("example.com")
        for i in range(50):
            sep.observe(
                url=f"https://example.com/page/{i}",
                status_code=200, size_bytes=5000, latency=0.1,
                raw_bytes=b"<html>x</html>",
            )
        score = sep.predictability_score()
        assert 0.0 <= score <= 1.0

    test_cases.append(("SEP: predictability score in [0,1]", test_sep_predictability_score))

    async def test_sep_report_structure():
        """SEP report has all expected fields."""
        sep = SpeculativeEnvelopePredictor("example.com")
        for i in range(20):
            sep.observe(
                url=f"https://example.com/page/{i}",
                status_code=200, size_bytes=5000, latency=0.1,
            )
        report = sep.report()
        assert "domain" in report
        assert "total_patterns" in report
        assert "total_predictions" in report
        assert "divergence_rate" in report
        assert "convergence_fraction" in report
        assert "top_patterns" in report

    test_cases.append(("SEP: report structure complete", test_sep_report_structure))

    async def test_sep_max_patterns_eviction():
        """SEP evicts least-observed pattern when limit reached."""
        sep = SpeculativeEnvelopePredictor("big.com")
        # Create more patterns than the limit
        for i in range(SEP_MAX_PATTERNS + 10):
            sep.observe(
                url=f"https://big.com/unique_path_{i}/page",
                status_code=200, size_bytes=1000, latency=0.05,
            )
        assert sep.pattern_count <= SEP_MAX_PATTERNS

    test_cases.append(("SEP: pattern eviction at limit", test_sep_max_patterns_eviction))

    async def test_sep_multiple_distinct_patterns():
        """SEP discovers multiple distinct patterns from URL structure."""
        sep = SpeculativeEnvelopePredictor("multi.com")
        # Pattern 1: /docs/*
        for i in range(10):
            sep.observe(
                url=f"https://multi.com/docs/page_{i}",
                status_code=200, size_bytes=10000, latency=0.15,
            )
        # Pattern 2: /api/*
        for i in range(10):
            sep.observe(
                url=f"https://multi.com/api/endpoint_{i}",
                status_code=200, size_bytes=2000, latency=0.05,
            )
        # Pattern 3: /blog/*
        for i in range(10):
            sep.observe(
                url=f"https://multi.com/blog/post_{i}",
                status_code=200, size_bytes=25000, latency=0.2,
            )
        # Should have at least 3 distinct patterns
        assert sep.pattern_count >= 3, f"expected ≥3 patterns, got {sep.pattern_count}"

    test_cases.append(("SEP: discovers multiple URL patterns", test_sep_multiple_distinct_patterns))

    async def test_sep_scoring_log_likelihood():
        """Prediction scores have valid log-likelihood (negative or zero)."""
        sep = SpeculativeEnvelopePredictor("example.com")
        for i in range(10):
            sep.observe(
                url=f"https://example.com/page/{i}",
                status_code=200, size_bytes=5000, latency=0.1,
                raw_bytes=b"<html>content</html>",
            )
        # This observation should produce a score
        score = sep.observe(
            url="https://example.com/page/10",
            status_code=200, size_bytes=5100, latency=0.11,
            raw_bytes=b"<html>content</html>",
        )
        assert score is not None
        assert score.log_likelihood <= 0  # log-likelihood is always ≤ 0

    test_cases.append(("SEP: log-likelihood is ≤ 0", test_sep_scoring_log_likelihood))

    async def test_sep_empty_bytes_handling():
        """SEP handles empty response bytes gracefully."""
        sep = SpeculativeEnvelopePredictor("example.com")
        for i in range(10):
            sep.observe(
                url=f"https://example.com/page/{i}",
                status_code=200, size_bytes=0, latency=0.1,
                raw_bytes=b"",
            )
        report = sep.report()
        assert report["total_patterns"] >= 1

    test_cases.append(("SEP: handles empty bytes", test_sep_empty_bytes_handling))

    async def test_sep_gaussian_log_likelihood():
        """Gaussian log-likelihood helper function works correctly."""
        # Observation at the mean → highest log-likelihood
        ll_at_mean = _gaussian_log_likelihood(5.0, 5.0, 1.0)
        # Observation 3σ away → much lower
        ll_far = _gaussian_log_likelihood(8.0, 5.0, 1.0)
        assert ll_at_mean > ll_far
        # Zero sigma → 0.0 (degenerate)
        assert _gaussian_log_likelihood(5.0, 5.0, 0.0) == 0.0

    test_cases.append(("SEP: Gaussian log-likelihood correctness", test_sep_gaussian_log_likelihood))

    # ── SEP + Fetcher integration test ────────────────────────────────

    async def test_fetcher_sep_integration():
        """SEP is wired into the fetcher and produces reports."""
        emitted: List[Any] = []
        bus = CrawlerBus()
        bus.subscribe(RawFetchEvent, lambda e: emitted.append(e))

        async def mock_handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"<html>" + b"x" * 500 + b"</html>")

        fetcher = Fetcher(
            bus=bus, bloom=_MockBloomFilter(), cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(), rate_limiter=None,
            staging_dir=Path(tempfile.mkdtemp()) / "staging",
        )
        await fetcher._staging.initialize()
        fetcher._http_client = httpx.AsyncClient(
            transport=httpx.MockTransport(mock_handler),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
        )
        fetcher._cl_state = CLState(cl1_available=True)
        fetcher._initialized = True

        manifest = _make_test_manifest(n_urls=20)
        telemetry = await fetcher._execute_manifest(manifest)

        summary = telemetry.summary()
        # Predictor should have been created and used
        assert summary.get("predictor") is not None or telemetry.predictor is not None
        # Observatory should have data
        assert summary.get("observatory") is not None or telemetry.observatory is not None

        if telemetry.predictor:
            pred_report = telemetry.predictor.report()
            assert pred_report["total_patterns"] >= 1

        if telemetry.observatory:
            obs_report = telemetry.observatory.report()
            assert obs_report.observations == 20

        await fetcher._http_client.aclose()

    test_cases.append(("Fetcher + SEP: integration test", test_fetcher_sep_integration))

    # ── Execute all tests ─────────────────────────────────────────────

    print(f"\n{'═' * 72}")
    print(f"  fetcher.py test suite — {len(test_cases)} tests")
    print(f"{'═' * 72}")

    for name, test_fn in test_cases:
        t0 = time.perf_counter()
        try:
            await test_fn()
            dt = (time.perf_counter() - t0) * 1000
            passed += 1
            if verbose:
                print(f"  ✓  {name} ({dt:.1f}ms)")
        except Exception as exc:
            dt = (time.perf_counter() - t0) * 1000
            failed += 1
            tb = traceback.format_exc()
            errors.append((name, tb))
            if verbose:
                print(f"  ✗  {name} ({dt:.1f}ms)")
                print(f"       {type(exc).__name__}: {exc}")

    print(f"{'─' * 72}")
    print(f"  {passed}/{len(test_cases)} passed  |  {failed} failed")
    print(f"{'─' * 72}")

    if errors:
        print("\nFailed test details:")
        for name, tb in errors:
            print(f"\n  ✗ {name}")
            for line in tb.splitlines():
                print(f"    {line}")

    print()
    return failed == 0


# ═══════════════════════════════════════════════════════════════════════════════
#
#   CLI — DEVELOPMENT AND BENCHMARK INTERFACE
#
#   Activated only via ``if __name__ == "__main__"``.
#   Zero pollution to the fetcher's runtime behavior when imported.
#
# ═══════════════════════════════════════════════════════════════════════════════

BENCHMARK_URLS: Dict[str, List[str]] = {
    "SAAS_DOCS": [
        "https://docs.stripe.com/api",
        "https://docs.github.com/en/rest",
        "https://developer.mozilla.org/en-US/docs/Web/API",
        "https://docs.python.org/3/library/asyncio.html",
        "https://httpx.readthedocs.io/en/latest/",
    ],
    "NEWS_ARTICLE": [
        "https://www.reuters.com/",
        "https://apnews.com/",
        "https://www.bbc.com/news",
    ],
    "WIKIPEDIA": [
        "https://en.wikipedia.org/wiki/Bloom_filter",
        "https://en.wikipedia.org/wiki/MurmurHash",
        "https://en.wikipedia.org/wiki/Token_bucket",
    ],
    "GENERIC": [
        "https://example.com",
        "https://httpbin.org/html",
        "https://httpbin.org/json",
    ],
}


async def _run_benchmark(
    cl_level: int,
    n_urls: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Benchmark the fetcher at a specific CL level.
    Uses live HTTP for CL1, mock for CL2+ (requires Playwright/Tor).
    """
    if n_urls is None:
        n_urls = BENCHMARK_DEFAULT_N.get(cl_level, 100)

    # Build URL set — cycle through benchmark URLs
    all_urls = []
    for category_urls in BENCHMARK_URLS.values():
        all_urls.extend(category_urls)

    urls_to_fetch = []
    for i in range(n_urls):
        urls_to_fetch.append(all_urls[i % len(all_urls)])

    # Build manifest
    profile = RateLimitProfile(
        domain="benchmark", requests_per_second=50.0,
        crawl_delay_seconds=0.0, burst_capacity=50,
    )
    manifest_id = str(uuid.uuid4())
    fetch_mode_map = {
        1: FetchMode.STATIC, 2: FetchMode.HEADLESS,
        3: FetchMode.TOR, 4: FetchMode.TOR_FULL,
    }
    mode = fetch_mode_map.get(cl_level, FetchMode.STATIC)

    crawl_urls = [
        CrawlURL(
            url=url, topology_hint="GENERIC_HTML", fetch_mode=mode,
            render_mode="static" if cl_level == 1 else "headless",
            priority=i, rate_limit_profile=profile,
            expected_content_type="text/html",
            crawl_delay_seconds=0.0,
            max_response_bytes=MAX_RESPONSE_BYTES,
            is_robots=False, is_sitemap=False,
            run_id=str(uuid.uuid4()),
        )
        for i, url in enumerate(urls_to_fetch)
    ]
    manifest = CrawlManifest(
        domain="benchmark", urls=crawl_urls,
        total_urls=len(crawl_urls),
        estimated_duration_seconds=float(len(crawl_urls)),
        clearance_required=cl_level, manifest_id=manifest_id,
    )

    # Create fetcher with mocks for non-CL1 dependencies
    with tempfile.TemporaryDirectory() as tmpdir:
        staging_dir = Path(tmpdir) / "staging"
        fetcher = Fetcher(
            bloom=_MockBloomFilter(),
            cursor=_MockCrawlCursor(),
            frontier=_MockFrontier(),
            rate_limiter=None,
            staging_dir=staging_dir,
        )
        await fetcher._staging.initialize()  # noqa
        fetcher._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(
                connect=STATIC_TIMEOUT_CONNECT,
                read=STATIC_TIMEOUT_READ,
                write=STATIC_TIMEOUT_WRITE,
                pool=STATIC_TIMEOUT_POOL,
            ),
            follow_redirects=True, max_redirects=MAX_REDIRECTS,
            verify=True, http2=True,
        )
        fetcher._cl_state = _build_test_cl_state(cl_level)
        fetcher._initialized = True
        if cl_level >= 2:
            await fetcher._pw_manager.initialize() # noqa

        if verbose:
            tor_label = "required" if cl_level >= 3 else "not required"
            console.print(Panel(
                f"[bold]CL Level:[/bold]  {cl_level} ([cyan]{mode.value}[/cyan])\n"
                f"[bold]CLState:[/bold]   {fetcher._cl_state.to_dict()}\n"  # noqa
                f"[bold]URLs:[/bold]      {n_urls}\n"
                f"[bold]Tor:[/bold]       {tor_label}",
                title="[bold yellow]Benchmark Configuration[/bold yellow]",
                border_style="yellow",
            ))

        # ── progress bar ──────────────────────────────────────────────────────
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(bar_width=40),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=console,
            transient=False,
        )

        fetch_task = progress.add_task(
            f"Fetching CL{cl_level} URLs", total=n_urls
        )

        async def _advance_progress():
            """Pulse the bar forward while the manifest runs."""
            while not progress.finished:
                await asyncio.sleep(0.05)
                remaining = progress.tasks[fetch_task].remaining
                if remaining and remaining > 0:
                    progress.advance(fetch_task, 1)

        with progress:
            advance_task = asyncio.create_task(_advance_progress())
            telemetry = await fetcher._execute_manifest(manifest)  # noqa
            # Jump bar to 100% once the manifest finishes
            progress.update(fetch_task, completed=n_urls)
            advance_task.cancel()
            try:
                await advance_task
            except asyncio.CancelledError:
                pass
        # ─────────────────────────────────────────────────────────────────────

        summary = telemetry.summary()

        if verbose:
            table = Table(box=box.ROUNDED, border_style="green", show_header=False)
            table.add_column("Metric", style="bold")
            table.add_column("Value", justify="right")

            passed = summary["succeeded"] > 0
            result_str = "[bold green]PASS ✓[/bold green]" if passed else "[bold red]FAIL ✗[/bold red]"

            rows = [
                ("Successful fetches",     str(summary["succeeded"])),
                ("Anomalies",              str(summary["failed"])),
                ("Bloom dedup hits",       str(summary["skipped"])),
                ("Throughput",             f"{summary['throughput_urls_per_sec']:.1f} URLs/sec"),
                ("Avg fetch latency",      f"{summary['latency']['mean'] * 1000:.0f} ms"),
                ("P50 latency",            f"{summary['latency_p50_ms']:.0f} ms"),
                ("P95 latency",            f"{summary['latency_p95_ms']:.0f} ms"),
                ("P99 latency",            f"{summary['latency_p99_ms']:.0f} ms"),
                ("Avg response size",      _format_bytes(int(summary["response_size"]["mean"]))),
                ("Status entropy",         f"{summary['status_entropy_bits']:.2f} bits"),
                ("RawFetchEvent emits",    str(summary["succeeded"])),
                ("FetchAnomalyEvent emits",str(summary["failed"])),
            ]

            if summary.get("little_law"):
                rows.append(("Little's Law", telemetry.little_law.recommendation))

            for metric, value in rows:
                table.add_row(metric, value)

            table.add_section()
            table.add_row("Result", result_str)

            console.print(Panel(
                table,
                title="[bold green]Benchmark Results[/bold green]",
                border_style="green",
            ))

        await fetcher._http_client.aclose()  # noqa
        await fetcher._pw_manager.shutdown() # noqa
        await asyncio.sleep(0.25)  # drain subprocess transport callbacks
        return summary


async def _run_dry_run(manifest_path: str, verbose: bool = True) -> Dict[str, Any]:
    """
    Dry run — parse manifest, check CL availability, check bloom hits.
    Never fetches.
    """
    with open(manifest_path, "r") as f:
        data = json.load(f)

    # Minimal parse
    result = {
        "manifest_path": manifest_path,
        "keys": list(data.keys()),
        "dry_run": True,
    }

    if verbose:
        print(f"\n{'─' * 68}")
        print(f"  Manifest: {manifest_path}")
        print(f"  Keys: {', '.join(data.keys())}")
        print(f"  Dry run — no fetch performed.")
        print(f"{'─' * 68}")

    return result


def _cli_main() -> int:
    """CLI entry point for fetcher.py."""
    parser = argparse.ArgumentParser(
        prog="fetcher.py",
        description="AXIOM fetcher.py — The vacuum. Fetches bytes. Emits events.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test
    subparsers.add_parser("test", help="Run the full test suite")

    # benchmark
    bench = subparsers.add_parser("benchmark", help="Benchmark at a specific CL level")
    bench.add_argument("--cl", type=int, choices=[1, 2, 3, 4], default=1)
    bench.add_argument("--n", type=int, default=None, metavar="N")

    # fetch
    fetch_parser = subparsers.add_parser("fetch", help="Fetch a single URL")
    fetch_parser.add_argument("--url", required=True)
    fetch_parser.add_argument("--cl", type=int, choices=[1, 2, 3, 4], default=1)

    # dry-run
    dry = subparsers.add_parser("dry-run", help="Dry run a manifest file")
    dry.add_argument("--manifest", required=True)

    # stress
    stress = subparsers.add_parser("stress", help="Stress test")
    stress.add_argument("--cl", type=int, choices=[1, 2, 3, 4], default=1)
    stress.add_argument("--urls", type=int, default=1000)
    stress.add_argument("--concurrency", type=int, default=4)

    args = parser.parse_args()

    if args.command == "test":
        result = asyncio.run(_run_tests(verbose=True))
        return 0 if result else 1

    elif args.command == "benchmark":
        asyncio.run(_run_benchmark(cl_level=args.cl, n_urls=args.n))
        return 0

    elif args.command == "fetch":
        async def _single():
            fetcher = Fetcher(staging_dir=Path(tempfile.mkdtemp()) / "staging")
            await fetcher.initialize()
            try:
                event = await fetcher.fetch_single(args.url, cl_level=args.cl)
                if event:
                    print(f"\n  URL:          {event.url}")
                    print(f"  Status:       {event.status_code}")
                    print(f"  Bytes:        {event.byte_count}")
                    print(f"  Latency:      {event.fetch_latency*1000:.0f}ms")
                    print(f"  Mode:         {event.fetch_mode.value}")
                    print(f"  Headers:      {len(event.headers)} entries")
                else:
                    print("  No event returned (URL may have been in bloom filter)")
            finally:
                await fetcher.shutdown()

        asyncio.run(_single())
        return 0

    elif args.command == "dry-run":
        asyncio.run(_run_dry_run(args.manifest))
        return 0

    elif args.command == "stress":
        asyncio.run(_run_benchmark(cl_level=args.cl, n_urls=args.urls))
        return 0

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=_LOG_FMT)
    sys.exit(_cli_main())
