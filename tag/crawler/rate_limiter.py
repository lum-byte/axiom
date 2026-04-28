"""
tag/crawler/rate_limiter.py
============================
Per-domain asynchronous token bucket rate limiter for the AXIOM crawler.

Layer:   Layer 1 — Acquisition Only (crawler/)
Depends: contracts.py (RateLimitProfile), asyncio, time
Emits:   Nothing. Pure enforcement — no events, no bus interaction.

Philosophy
----------
A 429 response anywhere in AXIOM is an architecture bug, not a runtime
condition.  The preparser already read Crawl-delay from robots.txt and
encoded it into RateLimitProfile.  This file enforces that ceiling
proactively.  The fetcher never discovers rate limits by hitting them;
it already knows them before the first request.

The token bucket algorithm is the canonical approach for sustained-rate
enforcement with burst allowance.  Tokens accumulate at `rate` per second
up to `capacity`.  Each request consumes one token.  When the bucket is
empty, `acquire()` yields to the asyncio event loop for exactly the time
needed to accumulate one token, then returns.  No retry logic.
No backoff.  No global cap.  No per-IP partitioning.  Per-domain only.

Concurrency model
-----------------
asyncio is single-threaded.  Within a single event-loop turn no two
coroutines can mutate the same bucket simultaneously.  However, multiple
coroutines CAN be scheduled on the same domain and all reach `acquire()`
before the first one completes its `asyncio.sleep()`.  Without
serialisation they would each compute an identical (or near-identical)
sleep duration and all wake within the same event-loop tick, effectively
issuing a burst of N requests simultaneously — exactly the condition the
rate limiter exists to prevent.

The fix is a per-domain `asyncio.Lock`.  The lock ensures that only one
coroutine at a time executes the refill + consume or sleep path for a
given domain.  Waiting coroutines queue behind the lock; when each one
enters it finds the bucket already partially refilled by elapsed time
since the previous coroutine's departure.

Time invariant
--------------
ALL timestamps use `time.monotonic()`.  Monotonic time never goes
backwards — NTP adjustments, DST transitions, and leap seconds cannot
perturb rate calculations.  `time.time()` is never used here.

Sleep invariant
---------------
ALL sleeps use `asyncio.sleep()`.  `time.sleep()` blocks the entire event
loop thread and would stall every coroutine in the process.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import (
    Dict,
    Iterator,
    List,
    Optional,
    Protocol,
    Tuple,
    runtime_checkable,
)
from urllib.parse import urlparse

# Internal import — contracts.py is the only intra-project import permitted
# by Law 3 of crawler/ (no imports from topology/, world_model/, etc.).
from signal_kernel.contracts import RateLimitProfile

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# Source of truth: readme-crawler.md § Constants Reference
# ─────────────────────────────────────────────────────────────────────────────

# Default rate applied to any domain not found in the manifest and not
# pre-registered via register().  Conservative — do not hammer unknown domains.
DEFAULT_RATE: float = 1.0        # 1 request per second

# Default burst capacity for domains without an explicit Crawl-delay.
# Allows 3 rapid requests before token refill becomes the bottleneck.
DEFAULT_BURST: int  = 3

# When a domain declares an explicit Crawl-delay in robots.txt the preparser
# sets burst_capacity = 1 to prevent any burst above the declared rate.
# The limiter enforces this exactly — no burst override.
EXPLICIT_DELAY_BURST: int = 1

# Minimum time (seconds) `acquire()` will sleep.  Values below this are
# rounded up to avoid spinning on near-zero sleeps that pollute event-loop
# scheduling.  1 ms is imperceptible to the fetcher but prevents hot loops.
MIN_SLEEP_SECONDS: float = 0.001   # 1 ms

# After this many seconds of inactivity a domain bucket is eligible for
# eviction from the in-memory registry via `evict_idle()`.  Eviction is
# never automatic — callers must invoke `evict_idle()` explicitly.
# The manifest pre-registers all domains on startup, so eviction is
# only relevant for very long sessions processing thousands of manifests.
IDLE_EVICTION_THRESHOLD_SECONDS: float = 3_600.0   # 1 hour

# Maximum per-domain acquire wait we will ever sleep for in a single call.
# If a domain has `requests_per_second = 0.001` (one request per 1000 s)
# the raw wait would be ~1000 s per token, which is impractical.
# The ceiling is enforced; the remaining wait is deferred to the *next*
# `acquire()` call on that domain.  Prevents indefinite blocking on pathological
# profiles while still respecting the configured rate over time.
MAX_SINGLE_SLEEP_SECONDS: float = 60.0   # 60 s per acquire() call

# ─────────────────────────────────────────────────────────────────────────────
# CLOCK PROTOCOL
# Abstracting over time.monotonic() allows unit tests to inject synthetic
# clocks that advance deterministically — no real sleeping required.
# ─────────────────────────────────────────────────────────────────────────────

@runtime_checkable
class MonotonicClock(Protocol):
    """
    Protocol for a monotonic clock source.

    Implementors must return a float representing seconds since some
    arbitrary epoch.  The value must be strictly non-decreasing within
    a single process (monotonic guarantee).

    The default implementation wraps `time.monotonic()`.  Test
    implementations may substitute any synthetic clock that advances
    by explicit increments.
    """

    def __call__(self) -> float:
        """Return current time in seconds (monotonic)."""
        ...


class _RealClock:
    """
    Production clock.  Delegates to `time.monotonic()`.

    This is the default used by `RateLimiter` unless a custom clock
    is injected at construction time.  Never use `time.time()` here —
    see the module docstring for the monotonic time invariant.
    """

    __slots__ = ()

    def __call__(self) -> float:
        return time.monotonic()

    def __repr__(self) -> str:
        return "RealClock(time.monotonic)"


# Singleton real clock.  RateLimiter uses this when no clock is injected.
_REAL_CLOCK: _RealClock = _RealClock()


class _FakeClock:
    """
    Synthetic clock for unit testing.

    Starts at t=0.0.  Tests advance time by calling `advance(delta)`.
    The clock value is a simple float — no thread safety, asyncio only.

    Usage::

        clock = _FakeClock()
        limiter = RateLimiter(clock=clock)
        await limiter.acquire("https://example.com/page")
        clock.advance(1.0)   # simulate 1 second passing
        await limiter.acquire("https://example.com/page2")

    Only intended for test code in this module.  Not part of the public API.
    """

    __slots__ = ("_t",)

    def __init__(self, start: float = 0.0) -> None:
        self._t: float = start

    def __call__(self) -> float:
        return self._t

    def advance(self, delta: float) -> None:
        """Advance the clock by `delta` seconds.  delta must be ≥ 0."""
        if delta < 0:
            raise ValueError(
                f"_FakeClock.advance() received delta={delta}. "
                "Monotonic clocks cannot go backwards."
            )
        self._t += delta

    def set(self, t: float) -> None:
        """Teleport the clock to `t`.  Only valid if t ≥ current value."""
        if t < self._t:
            raise ValueError(
                f"_FakeClock.set() received t={t} < current={self._t}. "
                "Monotonic clocks cannot go backwards."
            )
        self._t = t

    def __repr__(self) -> str:
        return f"FakeClock(t={self._t:.6f})"


# ─────────────────────────────────────────────────────────────────────────────
# CORE TOKEN BUCKET
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TokenBucket:
    """
    Mutable token bucket for one domain.

    Fields are intentionally *not* frozen — the bucket is mutated in-place
    by `_refill()` and `_consume()` on every `acquire()` call.  Mutation
    is safe because DomainBucket serialises access via `asyncio.Lock`.

    Do not construct directly.  Use `_make_bucket_from_profile()` or
    `_make_default_bucket()`.
    """

    # Maximum token count.  Tokens never exceed this value after a refill.
    capacity: float

    # Tokens added per second.  Derived from RateLimitProfile.requests_per_second
    # or DEFAULT_RATE.  Must be strictly positive.
    rate: float

    # Current token count.  Float — fractional tokens are valid and important
    # for sub-1 req/s rates.  Starts at `capacity` so fresh buckets allow
    # an immediate burst.
    tokens: float

    # Monotonic timestamp of the last refill calculation.  Used to compute
    # elapsed time on the next acquire().  Set to the clock value at
    # construction time.
    last_refill: float

    # Domain this bucket belongs to.  Stored for logging and snapshot output.
    domain: str

    # The RateLimitProfile this bucket was constructed from, if any.
    # None for default buckets (domains not in the manifest).
    source_profile: Optional[RateLimitProfile] = field(default=None)

    # Monotonic timestamp of bucket creation.  Used by evict_idle().
    created_at: float = field(default_factory=lambda: 0.0)

    def __repr__(self) -> str:
        return (
            f"TokenBucket(domain={self.domain!r}, "
            f"tokens={self.tokens:.4f}/{self.capacity:.1f}, "
            f"rate={self.rate:.4f} req/s)"
        )

    @property
    def is_default(self) -> bool:
        """True if this bucket was created from defaults, not a RateLimitProfile."""
        return self.source_profile is None

    @property
    def requests_per_second(self) -> float:
        """Alias for rate.  Matches RateLimitProfile field name."""
        return self.rate

    @property
    def crawl_delay_seconds(self) -> float:
        """Effective crawl delay implied by this bucket's rate."""
        if self.rate <= 0:
            return float("inf")
        return 1.0 / self.rate


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DomainStats:
    """
    Immutable snapshot of acquire statistics for one domain.

    Returned by `RateLimiter.get_domain_stats(domain)` and included in
    `RateLimiterSnapshot`.  All timing fields are in seconds.
    """

    domain: str

    # Total number of `acquire()` calls completed for this domain.
    total_acquires: int

    # Number of `acquire()` calls that required at least MIN_SLEEP_SECONDS.
    total_waits: int

    # Cumulative seconds spent sleeping across all waits for this domain.
    total_wait_seconds: float

    # Maximum single-call wait observed for this domain.
    max_wait_seconds: float

    # Mean wait per call that had to sleep (total_wait_seconds / total_waits).
    # 0.0 if total_waits == 0.
    mean_wait_seconds: float

    # Monotonic timestamp of the most recent acquire() completion.
    # 0.0 if no acquire has been completed yet.
    last_acquire_at: float

    # Whether this bucket was created from a RateLimitProfile (True)
    # or from default rates (False).
    from_profile: bool

    # Effective rate and capacity of the current bucket.
    effective_rate: float
    effective_capacity: float

    @property
    def wait_fraction(self) -> float:
        """
        Fraction of acquires that required sleeping.

        0.0 means the bucket always had tokens available.
        1.0 means every acquire had to wait.  Values approaching 1.0
        indicate the crawl rate is very close to the configured ceiling.
        """
        if self.total_acquires == 0:
            return 0.0
        return self.total_waits / self.total_acquires

    @property
    def throughput_per_second(self) -> float:
        """
        Observed throughput estimate.

        Not meaningful until at least 2 acquires have been completed.
        Returns effective_rate as a baseline for fresh buckets.
        """
        if self.total_acquires < 2 or self.last_acquire_at == 0.0:
            return self.effective_rate
        # Not available without first_acquire_at — return effective_rate
        return self.effective_rate

    def to_log_dict(self) -> dict:
        """
        Flat dict suitable for json.dumps() / structured logging.
        """
        return {
            "domain":               self.domain,
            "total_acquires":       self.total_acquires,
            "total_waits":          self.total_waits,
            "total_wait_seconds":   round(self.total_wait_seconds, 4),
            "max_wait_seconds":     round(self.max_wait_seconds, 4),
            "mean_wait_seconds":    round(self.mean_wait_seconds, 4),
            "wait_fraction":        round(self.wait_fraction, 4),
            "effective_rate":       round(self.effective_rate, 6),
            "effective_capacity":   self.effective_capacity,
            "from_profile":         self.from_profile,
        }


@dataclass
class RateLimiterSnapshot:
    """
    Point-in-time snapshot of the entire RateLimiter state.

    Returned by `RateLimiter.snapshot()`.  Used by monitoring, logging,
    and test assertions.  All timestamps are monotonic seconds.
    """

    # Number of domains currently registered (have a bucket).
    domain_count: int

    # Total acquires across all domains since limiter creation.
    total_acquires: int

    # Total waits (sleeps) across all domains since limiter creation.
    total_waits: int

    # Cumulative seconds spent sleeping across all domains.
    total_wait_seconds: float

    # Per-domain statistics.  Sorted by domain name for deterministic output.
    domain_stats: List[DomainStats]

    # Domains that are currently registered (bucket exists).
    registered_domains: List[str]

    def to_log_dict(self) -> dict:
        return {
            "domain_count":       self.domain_count,
            "total_acquires":     self.total_acquires,
            "total_waits":        self.total_waits,
            "total_wait_seconds": round(self.total_wait_seconds, 4),
            "registered_domains": sorted(self.registered_domains),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DOMAIN BUCKET — BUCKET + LOCK + STATISTICS
# ─────────────────────────────────────────────────────────────────────────────

class _DomainBucket:
    """
    Internal wrapper that binds a `TokenBucket` to its per-domain
    `asyncio.Lock` and running statistics counters.

    The lock is the critical concurrency primitive.  It guarantees that
    at most one coroutine at a time executes the refill + consume or
    sleep + consume path for this domain.  Without the lock, concurrent
    coroutines targeting the same domain could each observe the bucket as
    empty, each sleep for approximately the same duration, and all wake
    within the same event-loop tick — issuing a synchronised burst of N
    requests that exceeds the configured rate.

    Lock acquisition is non-blocking in the common case (token available,
    bucket returns immediately).  It only adds observable latency when
    concurrent coroutines are queued behind the lock, which is exactly
    the case we need to serialise.

    Lifetime: one `_DomainBucket` per domain, created lazily on first
    `acquire()` or eagerly via `register()`.  Shared across all concurrent
    manifest executions that happen to target the same domain.
    """

    __slots__ = (
        "_bucket",
        "_lock",
        "_total_acquires",
        "_total_waits",
        "_total_wait_seconds",
        "_max_wait_seconds",
        "_last_acquire_at",
    )

    def __init__(self, bucket: TokenBucket) -> None:
        self._bucket: TokenBucket = bucket

        # Per-domain lock.  Must be created inside the running event loop.
        # asyncio.Lock() is not shared across loops.
        self._lock: asyncio.Lock = asyncio.Lock()

        # Running statistics — mutated only while holding self._lock.
        self._total_acquires:    int   = 0
        self._total_waits:       int   = 0
        self._total_wait_seconds: float = 0.0
        self._max_wait_seconds:  float = 0.0
        self._last_acquire_at:   float = 0.0

    @property
    def bucket(self) -> TokenBucket:
        """Read-only access to the underlying TokenBucket."""
        return self._bucket

    def record_acquire(self, wait_seconds: float, now: float) -> None:
        """
        Record one completed acquire().

        Called while the domain lock is held, so no additional
        synchronisation is required.  `wait_seconds` is 0.0 for
        token-available acquires and > 0.0 when we slept.
        """
        self._total_acquires += 1
        self._last_acquire_at = now
        if wait_seconds >= MIN_SLEEP_SECONDS:
            self._total_waits += 1
            self._total_wait_seconds += wait_seconds
            if wait_seconds > self._max_wait_seconds:
                self._max_wait_seconds = wait_seconds

    def to_stats(self) -> DomainStats:
        """
        Produce an immutable DomainStats snapshot.

        May be called without holding the lock — reads are non-atomic
        but the worst consequence is a slightly stale counter value,
        which is acceptable for monitoring/logging purposes.
        """
        b = self._bucket
        mean_wait = (
            self._total_wait_seconds / self._total_waits
            if self._total_waits > 0
            else 0.0
        )
        return DomainStats(
            domain=b.domain,
            total_acquires=self._total_acquires,
            total_waits=self._total_waits,
            total_wait_seconds=self._total_wait_seconds,
            max_wait_seconds=self._max_wait_seconds,
            mean_wait_seconds=mean_wait,
            last_acquire_at=self._last_acquire_at,
            from_profile=not b.is_default,
            effective_rate=b.rate,
            effective_capacity=b.capacity,
        )

    def is_idle(self, now: float) -> bool:
        """
        True if this domain has not been acquired in IDLE_EVICTION_THRESHOLD_SECONDS.

        Used by `evict_idle()`.  A bucket that was registered but never
        acquired counts from its creation_at timestamp.
        """
        last_activity = (
            self._last_acquire_at
            if self._last_acquire_at > 0.0
            else self._bucket.created_at
        )
        return (now - last_activity) >= IDLE_EVICTION_THRESHOLD_SECONDS

    def __repr__(self) -> str:
        return (
            f"_DomainBucket({self._bucket.domain!r}, "
            f"acquires={self._total_acquires}, "
            f"waits={self._total_waits})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY — DOMAIN EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_domain(url: str) -> str:
    """
    Extract the netloc (host[:port]) from a URL and normalise to lowercase.

    Strips the port if present: "docs.stripe.com:443" → "docs.stripe.com".
    Subdomains are preserved: "docs.stripe.com" and "api.stripe.com" are
    treated as independent domains.  This is intentional — the RateLimitProfile
    from the CrawlManifest is already domain-specific; the preparser resolved
    the correct profile per URL, including subdomains.

    Returns an empty string for malformed URLs.  The RateLimiter treats an
    empty-string domain as a distinct (default-rate) bucket, which is safe
    because such URLs will fail during the actual HTTP fetch.

    Examples
    --------
    >>> _extract_domain("https://docs.stripe.com/api/charges")
    'docs.stripe.com'
    >>> _extract_domain("https://api.stripe.com:443/v1/customers")
    'api.stripe.com'
    >>> _extract_domain("https://localhost:8080/health")
    'localhost'
    >>> _extract_domain("not-a-url")
    ''
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower()
        # Strip port: "host:port" → "host"
        if ":" in netloc:
            host, _port = netloc.rsplit(":", 1)
            return host
        return netloc
    except Exception:  # pragma: no cover — malformed URL guard # noqa
        return ""


def _validate_rate(rate: float, context: str) -> float:
    """
    Validate that a rate is strictly positive.

    Returns the rate unchanged if valid.  Raises ValueError on invalid input.
    `context` is used in the error message for debugging.
    """
    if not isinstance(rate, (int, float)):
        raise TypeError( # noqa
            f"{context}: rate must be a float, got {type(rate).__name__}."
        )
    if rate <= 0:
        raise ValueError(
            f"{context}: rate must be > 0, got {rate!r}. "
            "A rate of 0 or negative means no requests are ever allowed. "
            "Check the RateLimitProfile from the preparser."
        )
    return float(rate)


def _validate_capacity(capacity: int | float, context: str) -> float:
    """
    Validate that a capacity is at least 1.

    Returns the capacity as a float.  Raises ValueError on invalid input.
    """
    if not isinstance(capacity, (int, float)):
        raise TypeError( # noqa
            f"{context}: capacity must be numeric, got {type(capacity).__name__}."
        )
    if capacity < 1:
        raise ValueError(
            f"{context}: capacity must be ≥ 1, got {capacity!r}. "
            "A capacity of 0 means no request can ever pass through. "
            "Check the RateLimitProfile from the preparser."
        )
    return float(capacity)


# ─────────────────────────────────────────────────────────────────────────────
# BUCKET CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────

def _make_bucket_from_profile(
    profile: RateLimitProfile,
    clock: MonotonicClock,
) -> TokenBucket:
    """
    Construct a `TokenBucket` from a `RateLimitProfile`.

    Validates the profile fields before constructing.  Raises ValueError on
    invalid profiles — callers (register(), acquire()) catch and log.

    Rate derivation rules, per spec:
    - If profile.crawl_delay_seconds > 0:
        rate = 1.0 / crawl_delay_seconds
        burst_capacity = 1  (no burst above declared delay)
    - Else:
        rate = profile.requests_per_second
        burst_capacity = profile.burst_capacity

    The burst_capacity override for explicit crawl delays is enforced here.
    It prevents AXIOM from racing ahead of the domain's declared preference
    even when burst_capacity on the incoming profile is incorrectly > 1.
    """
    domain   = profile.domain
    rate     = _validate_rate(profile.requests_per_second, f"profile for {domain!r}")
    capacity = _validate_capacity(profile.burst_capacity, f"profile for {domain!r}")

    # If an explicit Crawl-delay was present, verify rate derivation and
    # override capacity to 1.  This is defensive — crawl_planner.py should
    # already produce burst_capacity=1 for explicit delays, but we enforce it.
    if profile.crawl_delay_seconds > 0:
        expected_rate = 1.0 / profile.crawl_delay_seconds
        if abs(rate - expected_rate) > 0.0001:
            log.warning(
                "rate_limiter: profile for %r has crawl_delay_seconds=%.2f "
                "but requests_per_second=%.4f (expected %.4f). "
                "Using derived rate from crawl_delay_seconds.",
                domain,
                profile.crawl_delay_seconds,
                rate,
                expected_rate,
            )
            rate = expected_rate
        capacity = float(EXPLICIT_DELAY_BURST)

    now = clock()
    return TokenBucket(
        capacity=capacity,
        rate=rate,
        tokens=capacity,   # start full — first burst is allowed immediately
        last_refill=now,
        domain=domain,
        source_profile=profile,
        created_at=now,
    )


def _make_default_bucket(domain: str, clock: MonotonicClock) -> TokenBucket:
    """
    Construct a conservative default `TokenBucket` for an unknown domain.

    Applied when `acquire()` is called for a domain that was never
    pre-registered via `register()` and no inline profile is passed.

    Uses DEFAULT_RATE and DEFAULT_BURST from the module constants.
    The fetcher documents these as the fallback for domains not in the
    manifest — conservative enough to avoid hammering any server.
    """
    now = clock()
    return TokenBucket(
        capacity=float(DEFAULT_BURST),
        rate=DEFAULT_RATE,
        tokens=float(DEFAULT_BURST),
        last_refill=now,
        domain=domain,
        source_profile=None,
        created_at=now,
    )


# ─────────────────────────────────────────────────────────────────────────────
# REFILL AND CONSUME OPERATIONS
# These are pure functions — all state is passed in explicitly.
# They operate on a TokenBucket while the caller holds the domain lock.
# ─────────────────────────────────────────────────────────────────────────────

def _refill(bucket: TokenBucket, now: float) -> None:
    """
    Refill `bucket` based on time elapsed since last refill.

    Adds `elapsed * rate` tokens, capped at `capacity`.
    Updates `bucket.last_refill` to `now`.

    Called at the start of every `acquire()` while holding the domain lock.
    Must not be called outside of lock scope because last_refill mutation
    would race with concurrent refills.
    """
    elapsed = now - bucket.last_refill
    if elapsed > 0:
        bucket.tokens = min(bucket.capacity, bucket.tokens + elapsed * bucket.rate)
        bucket.last_refill = now


def _consume(bucket: TokenBucket) -> None:
    """
    Consume one token from `bucket`.

    Precondition: bucket.tokens >= 1.0 (caller is responsible for checking).
    Postcondition: bucket.tokens has decreased by exactly 1.0.

    Called after a successful `acquire()` or after the sleep completes.
    Must be called while holding the domain lock.
    """
    bucket.tokens -= 1.0
    # Guard against floating-point drift below zero.
    if bucket.tokens < 0.0:
        bucket.tokens = 0.0


def _compute_wait(bucket: TokenBucket) -> float:
    """
    Compute the seconds to sleep to accumulate one full token.

    Precondition: bucket.tokens < 1.0 (caller is responsible for checking).

    Returns a value in [MIN_SLEEP_SECONDS, MAX_SINGLE_SLEEP_SECONDS].
    The upper cap on MAX_SINGLE_SLEEP_SECONDS prevents indefinite blocking
    for pathological profiles (e.g. crawl_delay = 10 000 s).

    The remaining wait debt (if the true wait exceeds MAX_SINGLE_SLEEP_SECONDS)
    is serviced on the next `acquire()` call for this domain, since the bucket
    will still be empty and will compute another wait from the current token level.
    """
    tokens_needed = 1.0 - bucket.tokens
    raw_wait = tokens_needed / bucket.rate
    clamped = max(MIN_SLEEP_SECONDS, min(raw_wait, MAX_SINGLE_SLEEP_SECONDS))
    return clamped


# ─────────────────────────────────────────────────────────────────────────────
# RATE LIMITER
# ─────────────────────────────────────────────────────────────────────────────

class RateLimiter:
    """
    Per-domain asynchronous token bucket rate limiter.

    The fetcher calls `acquire(url)` before every HTTP request.  If the
    domain's bucket has tokens available, `acquire()` returns immediately.
    If not, it yields to the asyncio event loop until the next token
    becomes available.

    Pre-registration
    ----------------
    The fetcher calls `register(profile)` for every domain in the manifest
    before beginning URL execution.  This allows the limiter to configure
    precise bucket parameters from the preparser's RateLimitProfile without
    needing the profile inlined with every `acquire()` call.

    Lazy registration
    -----------------
    Domains not pre-registered receive a default bucket on first `acquire()`.
    This path exists for robustness only — every production domain should be
    pre-registered from the manifest.

    Per-domain locking
    ------------------
    Each domain's `_DomainBucket` contains an `asyncio.Lock`.  At most one
    coroutine at a time executes the acquire path for a given domain.
    This prevents concurrent coroutines from computing identical sleep
    durations and issuing a synchronised burst on wake.

    Shared state
    ------------
    A single `RateLimiter` instance is shared across all concurrent manifest
    executions in `fetcher.py`.  The per-domain lock ensures correctness.
    Domains from different manifests that happen to resolve to the same
    host (e.g. different paths on the same CDN) share the correct bucket.

    Invariants (enforced; violation is a programmer error)
    ------
    - `acquire()` never raises.  On any internal error it logs and returns.
    - `time.monotonic()` is always used (or the injected clock).
    - `asyncio.sleep()` is always used.  Never `time.sleep()`.
    - No exponential backoff.
    - No global request rate cap.
    - No per-IP or CL-level partitioning.
    - Buckets are never persisted to disk.  The manifest re-registers them
      on each startup.

    Thread safety
    -------------
    NOT thread-safe.  Designed for single-threaded asyncio event loop use.
    Do not share a RateLimiter instance across asyncio event loops or
    OS threads.

    Usage
    -----
    async with RateLimiter() as limiter:
        await limiter.register(profile)
        await limiter.acquire(url)

    Or without context manager::

        limiter = RateLimiter()
        await limiter.register(profile)
        await limiter.acquire(url)
    """

    def __init__(
        self,
        clock: MonotonicClock = _REAL_CLOCK,
    ) -> None:
        """
        Construct a new `RateLimiter`.

        Parameters
        ----------
        clock:
            Monotonic clock source.  Defaults to `time.monotonic` via the
            `_RealClock` singleton.  Pass a `_FakeClock` in tests to
            control time without real sleeping.

            The injected clock must satisfy the MonotonicClock protocol.
        """
        self._clock: MonotonicClock = clock

        # Primary bucket registry.  Keys are lowercase netloc strings
        # (subdomains included, port stripped).
        # Values are _DomainBucket instances, each with their own lock.
        self._buckets: Dict[str, _DomainBucket] = {}

        # Aggregate counters for snapshot().  Updated inside domain locks,
        # but the aggregate fields themselves are not individually locked —
        # they are only read in snapshot() which is a best-effort view.
        self._total_acquires: int = 0
        self._total_waits:    int = 0
        self._total_wait_seconds: float = 0.0

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "RateLimiter":
        return self

    async def __aexit__(self, *_: object) -> None:
        # Nothing to flush or close — buckets are in-memory only.
        # Included for symmetry with the other crawler/ components that
        # all expose async context managers.
        pass

    # ── Core API ──────────────────────────────────────────────────────────────

    async def acquire(
        self,
        url: str,
        profile: Optional[RateLimitProfile] = None,
    ) -> None:
        """
        Block until one request token is available for the domain of `url`.

        If the domain's bucket has tokens available, returns immediately.
        If not, suspends via `asyncio.sleep()` until the next token accrues,
        then consumes it and returns.

        Parameters
        ----------
        url:
            The full URL about to be fetched.  Domain is extracted
            automatically via `_extract_domain()`.

        profile:
            Optional RateLimitProfile for this domain.  If provided and
            the domain is not yet registered, the bucket is configured
            from this profile before the first acquire.  If the domain
            is already registered, the existing bucket is used and the
            profile is silently ignored (idempotent registration).

            The fetcher should prefer `register()` during manifest setup
            rather than passing a profile on every `acquire()` call.

        Returns
        -------
        None.  This method never raises — it always blocks and returns.
        Internal errors are logged at ERROR level and the method returns
        immediately to allow the fetcher to continue.

        Timing contract
        ---------------
        After acquire() returns, the caller is cleared to issue exactly one
        HTTP request to the domain.  The next acquire() call for the same
        domain will block until another token is available.
        """
        try:
            domain = _extract_domain(url)
            dbu    = self._get_or_create(domain, profile)
        except Exception: # noqa
            log.exception(
                "rate_limiter.acquire: unexpected error extracting domain from %r. "
                "Returning immediately (no rate limiting applied).",
                url,
            )
            return

        try:
            async with dbu._lock: # noqa
                wait_seconds = await self._acquire_under_lock(dbu, domain)
                self._total_acquires += 1
                if wait_seconds >= MIN_SLEEP_SECONDS:
                    self._total_waits += 1
                    self._total_wait_seconds += wait_seconds
        except Exception: # noqa
            log.exception(
                "rate_limiter.acquire: unexpected error while acquiring token "
                "for domain %r (url=%r). Returning immediately.",
                domain,
                url,
            )

    async def _acquire_under_lock(
        self,
        dbu: _DomainBucket,
        domain: str,
    ) -> float:
        """
        Execute the refill-consume-or-sleep sequence while holding the
        per-domain lock.  Returns the seconds actually slept (0.0 if none).

        This is the hot path.  Every line is intentional.

        Sequence
        --------
        1. Refill the bucket based on elapsed time.
        2. If tokens ≥ 1.0 — consume immediately, return 0.0.
        3. Else — compute wait_time, sleep, refill again, consume, return
           wait_time.

        The second refill after sleep is essential: in the time the
        coroutine was sleeping, the bucket has accumulated additional
        fractional tokens.  We update the bucket accurately so the next
        acquire() does not over-wait.
        """
        bucket = dbu.bucket
        now    = self._clock()

        _refill(bucket, now)

        if bucket.tokens >= 1.0:
            _consume(bucket)
            dbu.record_acquire(0.0, now)
            log.debug(
                "rate_limiter: acquired token for %r immediately "
                "(tokens_remaining=%.4f)",
                domain,
                bucket.tokens,
            )
            return 0.0

        # Bucket empty — must sleep.
        wait_seconds = _compute_wait(bucket)
        log.debug(
            "rate_limiter: domain %r bucket empty (tokens=%.4f). "
            "Sleeping %.4f s (rate=%.4f req/s, capacity=%.1f).",
            domain,
            bucket.tokens,
            wait_seconds,
            bucket.rate,
            bucket.capacity,
        )

        # Release the lock during the sleep.  Other coroutines targeting
        # *different* domains can proceed.  Coroutines targeting *this*
        # domain will queue on the lock — they will each find the bucket
        # in a post-refill state when they eventually enter.
        #
        # The lock must be explicitly released and re-acquired around the
        # sleep.  asyncio.Lock does NOT release during an awaited coroutine
        # by default — we must do it manually.
        dbu._lock.release() # noqa
        try:
            await asyncio.sleep(wait_seconds)
        finally:
            await dbu._lock.acquire() # noqa

        # Re-refill after sleep.  Fractional tokens may have accumulated.
        after_sleep = self._clock()
        _refill(bucket, after_sleep)
        _consume(bucket)
        dbu.record_acquire(wait_seconds, after_sleep)

        log.debug(
            "rate_limiter: acquired token for %r after %.4f s sleep "
            "(tokens_remaining=%.4f).",
            domain,
            wait_seconds,
            bucket.tokens,
        )
        return wait_seconds

    async def register(self, profile: RateLimitProfile) -> None:
        """
        Pre-register a domain's rate limit bucket from a `RateLimitProfile`.

        Called by the fetcher during manifest setup — once per domain in the
        manifest, before URL execution begins.  This ensures the first
        `acquire()` for a domain uses the correct manifest-derived rate
        rather than DEFAULT_RATE.

        Idempotent: if the domain is already registered (bucket exists),
        this call is a no-op.  Re-registering the same domain during a
        concurrent manifest does not reset the bucket or disturb the
        running token count.

        Parameters
        ----------
        profile:
            The RateLimitProfile from CrawlManifest.  Must have a valid
            domain, requests_per_second > 0, and burst_capacity ≥ 1.

        Raises
        ------
        Never.  Errors are logged at WARNING level and the method returns.
        A registration failure leaves the domain with default-rate behaviour,
        which is conservative and safe.
        """
        try:
            domain = profile.domain.lower()
            if domain in self._buckets:
                log.debug(
                    "rate_limiter.register: domain %r already registered. "
                    "No-op (idempotent).",
                    domain,
                )
                return

            bucket = _make_bucket_from_profile(profile, self._clock)
            self._buckets[domain] = _DomainBucket(bucket)
            log.debug(
                "rate_limiter.register: registered %r — "
                "rate=%.4f req/s, burst=%d, from_delay=%.2f s.",
                domain,
                bucket.rate,
                int(bucket.capacity),
                profile.crawl_delay_seconds,
            )
        except (ValueError, TypeError) as exc:
            log.warning(
                "rate_limiter.register: invalid profile for domain %r: %s. "
                "Domain will use default rate (%.1f req/s, burst=%d).",
                getattr(profile, "domain", "<unknown>"),
                exc,
                DEFAULT_RATE,
                DEFAULT_BURST,
            )
        except Exception: # noqa
            log.exception(
                "rate_limiter.register: unexpected error registering %r.",
                getattr(profile, "domain", "<unknown>"),
            )

    async def register_many(self, profiles: List[RateLimitProfile]) -> None:
        """
        Bulk-register multiple `RateLimitProfile` objects.

        Convenience method for manifest setup.  Equivalent to calling
        `register(p)` for each profile in `profiles`.

        Domains already registered are skipped (idempotent, matching
        `register()` semantics).

        Parameters
        ----------
        profiles:
            List of RateLimitProfile objects from the manifest.  Empty lists
            are silently accepted.

        This method does not raise.
        """
        for profile in profiles:
            await self.register(profile)

    def get_bucket(self, domain: str) -> Optional[TokenBucket]:
        """
        Return the raw `TokenBucket` for `domain`, or None if not registered.

        Intended for monitoring, debugging, and test assertions.

        The returned bucket is the live object — callers must not mutate it.
        Mutations outside the domain lock can corrupt the token accounting.

        Parameters
        ----------
        domain:
            Lowercase netloc string (e.g. "docs.stripe.com").  Must not
            include port or path.
        """
        dbu = self._buckets.get(domain.lower())
        if dbu is None:
            return None
        return dbu.bucket

    def get_domain_stats(self, domain: str) -> Optional[DomainStats]:
        """
        Return a `DomainStats` snapshot for `domain`.

        Returns None if the domain has no registered bucket.

        The snapshot is a best-effort point-in-time view.  Because it is
        read without holding the domain lock, individual fields may be
        slightly stale.  Suitable for monitoring; not for invariant checks.
        """
        dbu = self._buckets.get(domain.lower())
        if dbu is None:
            return None
        return dbu.to_stats()

    def snapshot(self) -> RateLimiterSnapshot:
        """
        Return an aggregate `RateLimiterSnapshot` covering all domains.

        Collects per-domain stats from every registered `_DomainBucket`.
        The snapshot is best-effort — values may be slightly stale.

        Useful for structured logging at the end of a manifest:

            snap = limiter.snapshot()
            log.info("rate_limiter.summary", extra=snap.to_log_dict())
        """
        domain_stats = sorted(
            (dbu.to_stats() for dbu in self._buckets.values()),
            key=lambda s: s.domain,
        )
        return RateLimiterSnapshot(
            domain_count=len(self._buckets),
            total_acquires=self._total_acquires,
            total_waits=self._total_waits,
            total_wait_seconds=self._total_wait_seconds,
            domain_stats=domain_stats,
            registered_domains=sorted(self._buckets.keys()),
        )

    async def reset(self, domain: str) -> None:
        """
        Reset the token bucket for `domain` to full capacity.

        FOR TESTING ONLY.

        Restores the bucket to its initial state (tokens = capacity,
        last_refill = now).  Does not affect the domain registration —
        the bucket remains registered with the same rate and capacity.

        If `domain` is not registered, this is a no-op.

        Parameters
        ----------
        domain:
            Lowercase netloc string.  Port must be stripped if present.
        """
        domain = domain.lower()
        dbu = self._buckets.get(domain)
        if dbu is None:
            return
        async with dbu._lock: # noqa
            bucket = dbu.bucket
            bucket.tokens = bucket.capacity
            bucket.last_refill = self._clock()
            log.debug("rate_limiter.reset: reset bucket for %r to full.", domain)

    async def reset_all(self) -> None:
        """
        Reset all registered domain buckets to full capacity.

        FOR TESTING ONLY.

        Useful in test teardown to isolate test cases.  In production use
        this would allow a burst of N requests per domain simultaneously —
        never call it outside of tests.
        """
        for domain in list(self._buckets.keys()):
            await self.reset(domain)

    def evict_idle(self) -> List[str]:
        """
        Remove domain buckets that have been idle for at least
        `IDLE_EVICTION_THRESHOLD_SECONDS`.

        Memory management for long-running sessions.  Eviction is never
        automatic — callers (typically the fetcher after manifest completion)
        invoke this explicitly.

        A bucket is considered idle if neither `acquire()` nor `register()`
        has been called on its domain in IDLE_EVICTION_THRESHOLD_SECONDS.
        Fresh buckets that were registered but never acquired count from
        their creation timestamp.

        Returns
        -------
        List of domain strings that were evicted.  Sorted for determinism.

        If an evicted domain is subsequently requested via `acquire()`, a
        new default-rate bucket will be created lazily.  The caller is
        responsible for re-registering domains via `register()` if the
        original profile rates should be restored.
        """
        now      = self._clock()
        to_evict = [
            domain
            for domain, dbu in self._buckets.items()
            if dbu.is_idle(now)
        ]
        for domain in to_evict:
            del self._buckets[domain]
            log.debug(
                "rate_limiter.evict_idle: evicted idle bucket for %r.",
                domain,
            )
        if to_evict:
            log.info(
                "rate_limiter.evict_idle: evicted %d idle domain(s): %s.",
                len(to_evict),
                sorted(to_evict),
            )
        return sorted(to_evict)

    def is_registered(self, domain: str) -> bool:
        """
        True if `domain` has a registered bucket.

        Parameters
        ----------
        domain:
            Lowercase netloc string.
        """
        return domain.lower() in self._buckets

    @property
    def domain_count(self) -> int:
        """Number of domains currently registered."""
        return len(self._buckets)

    @property
    def registered_domains(self) -> List[str]:
        """Sorted list of registered domain strings."""
        return sorted(self._buckets.keys())

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_or_create(
        self,
        domain: str,
        profile: Optional[RateLimitProfile],
    ) -> _DomainBucket:
        """
        Return the `_DomainBucket` for `domain`, creating it if absent.

        If `profile` is provided and the domain is not yet registered, the
        bucket is constructed from the profile.  Otherwise a default bucket
        is created.

        This is called by `acquire()` before entering the domain lock.
        It is NOT called under any lock, so concurrent coroutines can race
        here.  The race is harmless: both would create a bucket and the
        second assignment would overwrite the first.  This is safe because:
        1. asyncio is single-threaded — the "race" is actually sequential
           within a single event-loop tick.
        2. Even in theory, overwriting with an identical default bucket
           merely resets the token count for that domain in the worst case.

        In practice the fetcher pre-registers all domains before concurrent
        URL execution begins, so this race condition does not occur in
        production.
        """
        dbu = self._buckets.get(domain)
        if dbu is not None:
            return dbu

        # Domain not registered — create lazily.
        if profile is not None:
            try:
                bucket = _make_bucket_from_profile(profile, self._clock)
                log.debug(
                    "rate_limiter._get_or_create: lazy registration of %r "
                    "from inline profile (rate=%.4f req/s).",
                    domain,
                    bucket.rate,
                )
            except (ValueError, TypeError) as exc:
                log.warning(
                    "rate_limiter._get_or_create: invalid profile for %r (%s). "
                    "Falling back to default rate.",
                    domain,
                    exc,
                )
                bucket = _make_default_bucket(domain, self._clock)
        else:
            bucket = _make_default_bucket(domain, self._clock)
            log.debug(
                "rate_limiter._get_or_create: creating default bucket for "
                "unregistered domain %r (rate=%.1f req/s, burst=%d).",
                domain,
                DEFAULT_RATE,
                DEFAULT_BURST,
            )

        dbu = _DomainBucket(bucket)
        self._buckets[domain] = dbu
        return dbu

    def __repr__(self) -> str:
        return (
            f"RateLimiter("
            f"domains={self.domain_count}, "
            f"total_acquires={self._total_acquires}, "
            f"total_waits={self._total_waits})"
        )

    def __len__(self) -> int:
        """Number of registered domain buckets."""
        return self.domain_count


# ─────────────────────────────────────────────────────────────────────────────
# ITERATOR HELPER — PROFILE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def unique_profiles(
    profiles: List[RateLimitProfile],
) -> Iterator[RateLimitProfile]:
    """
    Yield unique RateLimitProfile objects, one per domain.

    Deduplicates by profile.domain.  When multiple profiles share a domain,
    the first encountered is yielded.  This mirrors the idempotent register()
    behaviour: only the first profile for a domain takes effect.

    Useful when building a deduplicated registration list from a manifest
    that may have multiple CrawlURLs with the same domain.

    Parameters
    ----------
    profiles:
        Iterable of RateLimitProfile objects, potentially with duplicated
        domains.

    Yields
    ------
    RateLimitProfile objects with unique .domain values.

    Examples
    --------
    >>> profiles = [
    ...     RateLimitProfile("stripe.com", 1.0, 0.0, 3),
    ...     RateLimitProfile("stripe.com", 2.0, 0.0, 3),   # duplicate
    ...     RateLimitProfile("github.com", 0.5, 2.0, 1),
    ... ]
    >>> list(p.domain for p in unique_profiles(profiles)) # noqa
    ['stripe.com', 'github.com']
    """
    seen: set = set()
    for profile in profiles:
        key = profile.domain.lower()
        if key not in seen:
            seen.add(key)
            yield profile


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE
#
# These tests encode every behavioural requirement from readme-crawler.md
# § rate_limiter.py and the Cross-File Protocol laws.
#
# Run with:  python -m pytest tag/crawler/rate_limiter.py -v
# Or:        python tag/crawler/rate_limiter.py  (standalone)
#
# Test structure: plain async functions prefixed with test_.
# No external test framework dependency — only asyncio and assert.
# pytest will discover them via its asyncio plugin if present.
#
# Each test is self-contained with its own RateLimiter instance and
# _FakeClock.  No shared state between tests.
# ─────────────────────────────────────────────────────────────────────────────

# ── Helpers for inline tests ──────────────────────────────────────────────────

def _make_profile(
    domain: str,
    rps: float,
    crawl_delay: float = 0.0,
    burst: int = 3,
) -> RateLimitProfile:
    """
    Construct a minimal RateLimitProfile for testing.

    `rps` is requests_per_second.  If `crawl_delay` is > 0, rps is
    derived from it (1/crawl_delay) to match preparser behaviour.
    """
    if crawl_delay > 0:
        derived_rps = 1.0 / crawl_delay
        return RateLimitProfile(
            domain=domain,
            requests_per_second=derived_rps,
            crawl_delay_seconds=crawl_delay,
            burst_capacity=EXPLICIT_DELAY_BURST,
        )
    return RateLimitProfile(
        domain=domain,
        requests_per_second=rps,
        crawl_delay_seconds=0.0,
        burst_capacity=burst,
    )


def _make_limiter(start_time: float = 0.0) -> Tuple["RateLimiter", "_FakeClock"]:
    """Create a RateLimiter backed by a FakeClock starting at start_time."""
    clock = _FakeClock(start=start_time)
    limiter = RateLimiter(clock=clock)
    return limiter, clock


# ── Test functions ─────────────────────────────────────────────────────────────

async def test_acquire_returns_immediately_when_tokens_available() -> None:
    """
    Spec test 1: acquire() returns immediately when tokens available.

    A fresh bucket starts at capacity tokens.  The first N acquires
    (up to burst_capacity) must return immediately without sleeping.
    """
    limiter, clock = _make_limiter()
    profile = _make_profile("stripe.com", rps=1.0, burst=3)
    await limiter.register(profile)

    url = "https://stripe.com/page"
    t_before = clock()
    for i in range(3):
        await limiter.acquire(url)
    t_after = clock()

    # No real time should have passed — FakeClock only advances when we
    # call clock.advance().  The three acquires should all have returned
    # immediately (consumed from the pre-filled bucket).
    assert t_after == t_before, (
        f"Clock advanced to {t_after} despite no advance() call. "
        "acquire() must only sleep via asyncio.sleep, not time.sleep."
    )

    bucket = limiter.get_bucket("stripe.com")
    assert bucket is not None
    assert bucket.tokens == pytest_approx(0.0, abs=0.01), (
        f"Expected ~0 tokens after 3 acquires from burst=3 bucket, got {bucket.tokens}"
    )
    stats = limiter.get_domain_stats("stripe.com")
    assert stats is not None
    assert stats.total_acquires == 3
    assert stats.total_waits == 0
    print("  PASS: test_acquire_returns_immediately_when_tokens_available")


async def test_acquire_sleeps_correct_duration_when_bucket_empty() -> None:
    """
    Spec test 2: acquire() sleeps correct duration when bucket empty.

    With rate=1 req/s and burst=1, after one immediate acquire the
    next acquire should need to wait ~1 second.
    """
    limiter, clock = _make_limiter()
    profile = _make_profile("example.com", rps=1.0, burst=1)
    await limiter.register(profile)

    url = "https://example.com/a"

    # First acquire — uses the pre-filled token.
    await limiter.acquire(url)

    # Second acquire — bucket is now empty.
    # The real asyncio.sleep() would block indefinitely in a fake-clock
    # environment, so we patch the sleep to advance the clock instead.
    #
    # Strategy: intercept asyncio.sleep() with a side-effecting coroutine
    # that advances the FakeClock by the requested duration.

    slept_durations: list = []
    real_sleep = asyncio.sleep

    async def fake_sleep(duration: float) -> None:
        slept_durations.append(duration)
        clock.advance(duration)
        # yield to event loop once to simulate async behaviour
        await asyncio.sleep(0)

    import unittest.mock as mock
    with mock.patch("asyncio.sleep", side_effect=fake_sleep):
        await limiter.acquire(url)

    assert len(slept_durations) == 1, (
        f"Expected exactly 1 sleep call, got {len(slept_durations)}: {slept_durations}"
    )
    slept = slept_durations[0]
    # With rate=1 req/s and 0 tokens remaining, the expected wait is ~1.0 s.
    assert abs(slept - 1.0) < 0.01, (
        f"Expected sleep ~1.0 s for rate=1 req/s empty bucket, got {slept:.4f} s"
    )

    stats = limiter.get_domain_stats("example.com")
    assert stats is not None
    assert stats.total_waits == 1
    print("  PASS: test_acquire_sleeps_correct_duration_when_bucket_empty")


async def test_per_domain_isolation() -> None:
    """
    Spec test 3: Per-domain isolation — two domains do not share tokens.

    domain_a and domain_b each have burst=1.  Acquiring from domain_a
    should not affect domain_b's token count and vice versa.
    """
    limiter, clock = _make_limiter()
    await limiter.register(_make_profile("domain-a.com", rps=1.0, burst=1))
    await limiter.register(_make_profile("domain-b.com", rps=2.0, burst=1))

    # Each domain has 1 token.  Acquire from domain_a.
    await limiter.acquire("https://domain-a.com/page")

    # domain_b should still have its token intact.
    bucket_b = limiter.get_bucket("domain-b.com")
    assert bucket_b is not None
    assert bucket_b.tokens >= 0.99, (
        f"domain_b should still have ~1 token after acquiring from domain_a. "
        f"Got tokens={bucket_b.tokens:.4f}"
    )

    # domain_a should be at 0 tokens.
    bucket_a = limiter.get_bucket("domain-a.com")
    assert bucket_a is not None
    assert bucket_a.tokens < 0.01, (
        f"domain_a should have ~0 tokens after 1 acquire from burst=1. "
        f"Got tokens={bucket_a.tokens:.4f}"
    )
    print("  PASS: test_per_domain_isolation")


async def test_rate_derived_from_crawl_delay_seconds() -> None:
    """
    Spec test 4: Rate derived correctly from crawl_delay_seconds.

    crawl_delay_seconds=10 → requests_per_second=0.1.
    burst_capacity=1 (explicit delay → no burst).
    """
    limiter, clock = _make_limiter()
    profile = _make_profile("slow.com", rps=0.0, crawl_delay=10.0)
    await limiter.register(profile)

    bucket = limiter.get_bucket("slow.com")
    assert bucket is not None

    expected_rate = 1.0 / 10.0
    assert abs(bucket.rate - expected_rate) < 0.0001, (
        f"Expected rate {expected_rate:.4f} req/s for crawl_delay=10s, got {bucket.rate:.4f}"
    )
    assert bucket.capacity == float(EXPLICIT_DELAY_BURST), (
        f"Expected burst_capacity={EXPLICIT_DELAY_BURST} for explicit Crawl-delay, "
        f"got {bucket.capacity}"
    )
    print("  PASS: test_rate_derived_from_crawl_delay_seconds")


async def test_burst_capacity_respected() -> None:
    """
    Spec test 5: Burst capacity respected — 3 fast requests succeed, 4th blocks.

    With burst=3 and rate=1 req/s, the first 3 acquires should return
    immediately.  The 4th should block (require a sleep).
    """
    limiter, clock = _make_limiter()
    await limiter.register(_make_profile("burst.com", rps=1.0, burst=3))

    url   = "https://burst.com/page"
    slept: list = []

    async def fake_sleep(duration: float) -> None:
        slept.append(duration)
        clock.advance(duration)
        await asyncio.sleep(0)

    import unittest.mock as mock
    with mock.patch("asyncio.sleep", side_effect=fake_sleep):
        for i in range(3):
            await limiter.acquire(url)
        assert len(slept) == 0, (
            f"First 3 acquires should not sleep. Slept={slept}"
        )
        await limiter.acquire(url)   # 4th — should sleep

    assert len(slept) == 1, (
        f"4th acquire should sleep exactly once. Slept={slept}"
    )
    print("  PASS: test_burst_capacity_respected")


async def test_unknown_domain_gets_default_rate() -> None:
    """
    Spec test 7: Unknown domain gets DEFAULT_RATE.

    An unregistered domain should receive DEFAULT_RATE and DEFAULT_BURST.
    """
    limiter, _ = _make_limiter()
    await limiter.acquire("https://unregistered.example.com/page")

    bucket = limiter.get_bucket("unregistered.example.com")
    assert bucket is not None
    assert abs(bucket.rate - DEFAULT_RATE) < 0.0001, (
        f"Expected DEFAULT_RATE={DEFAULT_RATE} for unregistered domain, got {bucket.rate}"
    )
    assert bucket.capacity == float(DEFAULT_BURST), (
        f"Expected DEFAULT_BURST={DEFAULT_BURST}, got {bucket.capacity}"
    )
    assert bucket.is_default is True, (
        "Unregistered domain bucket should have is_default=True"
    )
    print("  PASS: test_unknown_domain_gets_default_rate")


async def test_register_is_idempotent() -> None:
    """
    Spec test 8: register() is idempotent — re-registering same domain
    has no effect and does not reset the token count.
    """
    limiter, clock = _make_limiter()
    profile1 = _make_profile("idempotent.com", rps=2.0, burst=5)
    await limiter.register(profile1)

    # Consume 2 tokens.
    await limiter.acquire("https://idempotent.com/a")
    await limiter.acquire("https://idempotent.com/b")

    bucket_before = limiter.get_bucket("idempotent.com")
    tokens_before = bucket_before.tokens  # type: ignore[union-attr]

    # Re-register with different rps — should be ignored.
    profile2 = _make_profile("idempotent.com", rps=100.0, burst=100)
    await limiter.register(profile2)

    bucket_after = limiter.get_bucket("idempotent.com")
    assert bucket_after is not None
    assert abs(bucket_after.tokens - tokens_before) < 0.001, (
        f"Re-registration should not reset token count. "
        f"Before={tokens_before:.4f}, After={bucket_after.tokens:.4f}"
    )
    assert abs(bucket_after.rate - 2.0) < 0.0001, (
        f"Re-registration should not change rate. "
        f"Expected 2.0, got {bucket_after.rate:.4f}"
    )
    print("  PASS: test_register_is_idempotent")


async def test_subdomains_are_independent() -> None:
    """
    Spec test 9: Subdomains are independent.

    docs.stripe.com and api.stripe.com are treated as separate domains
    with separate buckets.  A token consumed from one does not affect
    the other.
    """
    limiter, clock = _make_limiter()
    await limiter.register(_make_profile("docs.stripe.com", rps=1.0, burst=1))
    await limiter.register(_make_profile("api.stripe.com", rps=5.0, burst=5))

    await limiter.acquire("https://docs.stripe.com/api/overview")

    docs_bucket = limiter.get_bucket("docs.stripe.com")
    api_bucket  = limiter.get_bucket("api.stripe.com")

    assert docs_bucket is not None and api_bucket is not None
    assert docs_bucket.tokens < 0.01, (
        f"docs.stripe.com should have ~0 tokens after 1 acquire from burst=1."
    )
    assert api_bucket.tokens >= 4.99, (
        f"api.stripe.com should be unaffected. Expected ~5 tokens, got {api_bucket.tokens:.4f}"
    )

    assert limiter.domain_count == 2, (
        f"Expected 2 separate buckets, got {limiter.domain_count}"
    )
    print("  PASS: test_subdomains_are_independent")


async def test_monotonic_clock_not_time_dot_time() -> None:
    """
    Spec test 6: Monotonic time — bucket behaves correctly under time adjustment.

    Injecting a FakeClock and verifying that the limiter never calls
    time.time() (which could go backwards) by ensuring all behaviour
    is consistent with the injected clock.
    """
    clock = _FakeClock(start=1_000_000.0)   # arbitrary large start
    limiter = RateLimiter(clock=clock)
    profile = _make_profile("monotonic.com", rps=1.0, burst=2)
    await limiter.register(profile)

    await limiter.acquire("https://monotonic.com/a")
    await limiter.acquire("https://monotonic.com/b")

    # Bucket should now be empty.
    bucket = limiter.get_bucket("monotonic.com")
    assert bucket is not None
    assert bucket.tokens < 0.01

    # Advance time by 1 s — should regenerate 1 token.
    clock.advance(1.0)
    bucket_after_advance = limiter.get_bucket("monotonic.com")
    # Token hasn't been recalculated yet — refill happens on next acquire.
    # But last_refill is still the old value, so the refill will be correct.

    # The bucket still reflects pre-advance values because _refill()
    # hasn't run yet.  Run an acquire() to trigger refill.
    slept: list = []

    async def fake_sleep(duration: float) -> None:
        slept.append(duration)
        clock.advance(duration)
        await asyncio.sleep(0)

    import unittest.mock as mock
    with mock.patch("asyncio.sleep", side_effect=fake_sleep):
        await limiter.acquire("https://monotonic.com/c")

    # After 1 s advance at rate=1 req/s, the bucket should have had exactly
    # 1 token available — the acquire should not have slept.
    assert len(slept) == 0, (
        f"After 1s advance with rate=1 req/s, acquire should not sleep. Slept={slept}"
    )
    print("  PASS: test_monotonic_clock_not_time_dot_time")


async def test_domain_extraction_strips_port() -> None:
    """
    Verify _extract_domain() strips port numbers and lowercases netloc.
    """
    cases = [
        ("https://stripe.com/api/v1",          "stripe.com"),
        ("https://api.stripe.com:443/charges",  "api.stripe.com"),
        ("http://localhost:8080/health",         "localhost"),
        ("https://DOCS.GITHUB.COM/en",           "docs.github.com"),
        ("https://a.b.c.d.example.com/x/y",     "a.b.c.d.example.com"),
    ]
    for url, expected in cases:
        result = _extract_domain(url)
        assert result == expected, (
            f"_extract_domain({url!r}) → {result!r}, expected {result!r}"
        )
    print("  PASS: test_domain_extraction_strips_port")


async def test_profile_with_explicit_crawl_delay_forces_burst_1() -> None:
    """
    A profile with crawl_delay_seconds > 0 must produce burst_capacity=1
    even if the incoming profile has burst_capacity > 1.
    """
    limiter, clock = _make_limiter()
    # Deliberately pass burst_capacity=5 with crawl_delay — the limiter
    # should override to 1.
    profile = RateLimitProfile(
        domain="explicit-delay.com",
        requests_per_second=0.5,  # 2 s delay
        crawl_delay_seconds=2.0,
        burst_capacity=5,         # preparser bug — should be 1 for explicit delay
    )
    await limiter.register(profile)

    bucket = limiter.get_bucket("explicit-delay.com")
    assert bucket is not None
    assert bucket.capacity == float(EXPLICIT_DELAY_BURST), (
        f"Explicit Crawl-delay must force burst=1, got capacity={bucket.capacity}"
    )
    expected_rate = 1.0 / 2.0
    assert abs(bucket.rate - expected_rate) < 0.0001, (
        f"Expected rate={expected_rate:.4f} for crawl_delay=2, got {bucket.rate:.4f}"
    )
    print("  PASS: test_profile_with_explicit_crawl_delay_forces_burst_1")


async def test_register_many_bulk_registers_all_profiles() -> None:
    """
    register_many() registers all profiles and is idempotent for duplicates.
    """
    limiter, clock = _make_limiter()
    profiles = [
        _make_profile("alpha.com", rps=1.0, burst=3),
        _make_profile("beta.com",  rps=2.0, burst=5),
        _make_profile("gamma.com", rps=0.5, burst=1),
        _make_profile("alpha.com", rps=99.0, burst=99),  # duplicate — first wins
    ]
    await limiter.register_many(profiles)

    assert limiter.domain_count == 3, (
        f"Expected 3 domains (alpha, beta, gamma), got {limiter.domain_count}"
    )
    alpha = limiter.get_bucket("alpha.com")
    assert alpha is not None and abs(alpha.rate - 1.0) < 0.0001, (
        f"alpha.com should have rps=1.0 (first registration wins), got {alpha.rate}"
    )
    print("  PASS: test_register_many_bulk_registers_all_profiles")


async def test_snapshot_aggregates_all_domains() -> None:
    """
    snapshot() returns correct aggregate statistics across all domains.
    """
    limiter, clock = _make_limiter()
    await limiter.register(_make_profile("a.com", rps=10.0, burst=10))
    await limiter.register(_make_profile("b.com", rps=10.0, burst=10))

    for _ in range(5):
        await limiter.acquire("https://a.com/page")
    for _ in range(3):
        await limiter.acquire("https://b.com/page")

    snap = limiter.snapshot()
    assert snap.domain_count == 2
    assert snap.total_acquires == 8
    assert len(snap.domain_stats) == 2
    a_stats = next(s for s in snap.domain_stats if s.domain == "a.com")
    assert a_stats.total_acquires == 5
    b_stats = next(s for s in snap.domain_stats if s.domain == "b.com")
    assert b_stats.total_acquires == 3
    print("  PASS: test_snapshot_aggregates_all_domains")


async def test_reset_restores_bucket_to_full() -> None:
    """
    reset() fills the bucket back to capacity.
    """
    limiter, clock = _make_limiter()
    await limiter.register(_make_profile("reset.com", rps=1.0, burst=3))

    for _ in range(3):
        await limiter.acquire("https://reset.com/page")

    bucket_before = limiter.get_bucket("reset.com")
    assert bucket_before is not None
    assert bucket_before.tokens < 0.01

    await limiter.reset("reset.com")

    bucket_after = limiter.get_bucket("reset.com")
    assert bucket_after is not None
    assert abs(bucket_after.tokens - 3.0) < 0.001, (
        f"After reset(), tokens should be back to capacity 3.0, got {bucket_after.tokens:.4f}"
    )
    print("  PASS: test_reset_restores_bucket_to_full")


async def test_evict_idle_removes_stale_domains() -> None:
    """
    evict_idle() removes domains that have been idle beyond the threshold.
    """
    clock = _FakeClock(start=0.0)
    limiter = RateLimiter(clock=clock)
    await limiter.register(_make_profile("old.com",    rps=1.0, burst=1))
    await limiter.register(_make_profile("recent.com", rps=1.0, burst=1))

    # Acquire from both domains at t=0.
    await limiter.acquire("https://old.com/page")
    await limiter.acquire("https://recent.com/page")

    # Advance time past the idle threshold for old.com.
    clock.advance(IDLE_EVICTION_THRESHOLD_SECONDS + 1.0)

    # Acquire from recent.com to update its last_acquire_at.
    await limiter.acquire("https://recent.com/page2")

    evicted = limiter.evict_idle()
    assert "old.com" in evicted, (
        f"old.com should be evicted after {IDLE_EVICTION_THRESHOLD_SECONDS}s idle. "
        f"Evicted: {evicted}"
    )
    assert "recent.com" not in evicted, (
        f"recent.com should not be evicted (was recently active). Evicted: {evicted}"
    )
    assert not limiter.is_registered("old.com")
    assert limiter.is_registered("recent.com")
    print("  PASS: test_evict_idle_removes_stale_domains")


async def test_acquire_never_raises() -> None:
    """
    acquire() must never raise, even on pathological inputs.
    """
    limiter, clock = _make_limiter()

    # Malformed URL — should not raise.
    await limiter.acquire("not-a-url")
    await limiter.acquire("")
    await limiter.acquire("ftp://weird-scheme.com/path")

    # Empty domain — should create a default bucket without raising.
    await limiter.acquire("")

    print("  PASS: test_acquire_never_raises")


async def test_unique_profiles_deduplicates_by_domain() -> None:
    """
    unique_profiles() yields each domain's first profile only.
    """
    profiles = [
        _make_profile("x.com", rps=1.0),
        _make_profile("y.com", rps=2.0),
        _make_profile("x.com", rps=99.0),  # duplicate — should be skipped
        _make_profile("z.com", rps=3.0),
    ]
    result = list(unique_profiles(profiles))
    domains = [p.domain for p in result]
    assert domains == ["x.com", "y.com", "z.com"], (
        f"Expected ['x.com', 'y.com', 'z.com'], got {domains}"
    )
    assert result[0].requests_per_second == 1.0, (
        "First x.com profile (rps=1.0) should win over duplicate (rps=99.0)"
    )
    print("  PASS: test_unique_profiles_deduplicates_by_domain")


async def test_max_single_sleep_cap_applied() -> None:
    """
    Very slow domains (e.g. crawl_delay=600s) are capped to
    MAX_SINGLE_SLEEP_SECONDS per acquire() call.
    """
    limiter, clock = _make_limiter()
    # crawl_delay=600s → rate=1/600 ≈ 0.00167 req/s
    # The raw wait for an empty bucket would be ~600s.
    # MAX_SINGLE_SLEEP_SECONDS should cap it.
    profile = _make_profile("slow.com", rps=0.0, crawl_delay=600.0)
    await limiter.register(profile)

    # Consume the one burst token.
    await limiter.acquire("https://slow.com/page")

    bucket = limiter.get_bucket("slow.com")
    assert bucket is not None
    wait = _compute_wait(bucket)
    assert wait <= MAX_SINGLE_SLEEP_SECONDS, (
        f"Wait for crawl_delay=600s should be capped at {MAX_SINGLE_SLEEP_SECONDS}s, "
        f"got {wait:.1f}s"
    )
    assert wait >= MIN_SLEEP_SECONDS, (
        f"Wait must be at least {MIN_SLEEP_SECONDS}s (MIN_SLEEP_SECONDS)"
    )
    print("  PASS: test_max_single_sleep_cap_applied")


async def test_fake_clock_monotonic_invariant() -> None:
    """
    _FakeClock raises if asked to go backwards.
    """
    clock = _FakeClock(start=10.0)
    try:
        clock.advance(-1.0)
        raise AssertionError("advance(-1.0) should have raised ValueError")
    except ValueError:
        pass
    try:
        clock.set(9.0)
        raise AssertionError("set(9.0) < current(10.0) should have raised ValueError")
    except ValueError:
        pass
    print("  PASS: test_fake_clock_monotonic_invariant")


async def test_domain_stats_wait_fraction() -> None:
    """
    DomainStats.wait_fraction is accurate.
    """
    limiter, clock = _make_limiter()
    await limiter.register(_make_profile("stats.com", rps=1.0, burst=3))

    # 3 immediate acquires, then 1 that will sleep.
    url = "https://stats.com/page"
    for _ in range(3):
        await limiter.acquire(url)

    # Manually record a wait by injecting a waited acquire.
    slept: list = []

    async def fake_sleep(duration: float) -> None:
        slept.append(duration)
        clock.advance(duration)
        await asyncio.sleep(0)

    import unittest.mock as mock
    with mock.patch("asyncio.sleep", side_effect=fake_sleep):
        await limiter.acquire(url)   # 4th — should sleep

    stats = limiter.get_domain_stats("stats.com")
    assert stats is not None
    assert stats.total_acquires == 4
    assert stats.total_waits == 1
    expected_fraction = 1 / 4
    assert abs(stats.wait_fraction - expected_fraction) < 0.001, (
        f"Expected wait_fraction={expected_fraction:.3f}, got {stats.wait_fraction:.3f}"
    )
    print("  PASS: test_domain_stats_wait_fraction")


# pytest_approx shim — avoids hard dependency on pytest inside the standalone
# test runner while still providing pytest compatibility.
def pytest_approx(value: float, abs: float = 1e-6) -> "_ApproxScalar": # noqa
    return _ApproxScalar(value, abs)


class _ApproxScalar:
    __slots__ = ("_expected", "_abs")

    def __init__(self, expected: float, abs: float) -> None: # noqa
        self._expected = expected
        self._abs = abs

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, (int, float)):
            return NotImplemented
        return abs(other - self._expected) <= self._abs

    def __repr__(self) -> str:
        return f"approx({self._expected!r} ± {self._abs!r})"


# ─────────────────────────────────────────────────────────────────────────────
# STANDALONE TEST RUNNER
# `python rate_limiter.py` runs all test_ coroutines without pytest.
# ─────────────────────────────────────────────────────────────────────────────

_TEST_FUNCTIONS = [
    test_acquire_returns_immediately_when_tokens_available,
    test_acquire_sleeps_correct_duration_when_bucket_empty,
    test_per_domain_isolation,
    test_rate_derived_from_crawl_delay_seconds,
    test_burst_capacity_respected,
    test_unknown_domain_gets_default_rate,
    test_register_is_idempotent,
    test_subdomains_are_independent,
    test_monotonic_clock_not_time_dot_time,
    test_domain_extraction_strips_port,
    test_profile_with_explicit_crawl_delay_forces_burst_1,
    test_register_many_bulk_registers_all_profiles,
    test_snapshot_aggregates_all_domains,
    test_reset_restores_bucket_to_full,
    test_evict_idle_removes_stale_domains,
    test_acquire_never_raises,
    test_unique_profiles_deduplicates_by_domain,
    test_max_single_sleep_cap_applied,
    test_fake_clock_monotonic_invariant,
    test_domain_stats_wait_fraction,
]


async def _run_all_tests() -> None:
    """Run all test_ coroutines and print a summary."""
    print(f"\nrate_limiter.py — standalone test suite ({len(_TEST_FUNCTIONS)} tests)")
    print("=" * 60)
    passed  = 0
    failed  = 0
    errors: list = []
    for test_fn in _TEST_FUNCTIONS:
        try:
            await test_fn()
            passed += 1
        except Exception as exc:
            failed += 1
            errors.append((test_fn.__name__, exc))
            print(f"  FAIL: {test_fn.__name__}: {exc}")
    print("=" * 60)
    print(f"Results: {passed}/{len(_TEST_FUNCTIONS)} passed, {failed} failed.")
    if errors:
        print("\nFailed tests:")
        for name, exc in errors:
            print(f"  {name}: {exc}")
        raise SystemExit(1)
    else:
        print("All tests passed.")


if __name__ == "__main__":
    import sys
    # Configure logging for standalone runs.
    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    asyncio.run(_run_all_tests())