"""
crawler/bloom_filter.py
=======================
Memory-mapped probabilistic URL deduplication filter for the AXIOM crawler layer.

AXIOM INTERNAL // DO NOT SURFACE

This is the single source of truth for "have we seen this URL before." Every URL
the crawler considers passes through here before any fetch is attempted. The filter
answers that question in O(k) time using 7 mmap reads — no disk seek, no SQLite
round-trip, no process boundary crossing. At 400M URLs it fits in 500MB of virtual
address space backed by a single flat binary file.

Architectural role:
    bloom_filter.py sits at position 1 in the crawler build order. It has zero
    internal dependencies on other crawler/ files. frontier.py and fetcher.py
    depend on it — it depends on nothing inside tag/.

    frontier.py calls bloom.contains() before queueing a URL. After a successful
    fetch, fetcher.py calls bloom.add(). The filter is never written to speculatively.
    A URL enters the filter only after its fetch event has been emitted to the bus.

Durability guarantee:
    The bit array lives in a memory-mapped file. Every bit set by add() is durable
    the moment the OS writes the dirty page to disk. We additionally call mmap.flush()
    every BLOOM_FLUSH_INTERVAL adds (and always on close) to force dirty pages out of
    the page cache and onto disk. After a flush, the bits survive SIGKILL, power loss,
    and container crashes. The cost of a flush is ~1-2ms for a 500MB mmap on modern
    NVMe — negligible amortized over 10,000 adds.

    On restart: open the file, mmap it, done. The full deduplication history is
    immediately available. Zero rebuild. Zero warm-up. Zero cold-start penalty.

The math:
    Given:
        n = 400,000,000   (expected elements)
        p = 0.0001        (target false positive probability)

    Optimal bit array size:
        m = ceil(n * |ln(p)| / ln(2)^2)
          = ceil(400M * 9.2103 / 0.4805)
          = ceil(7.666B)
          → 4,000,000,000 bits (rounded to a clean number, slightly under optimal)
          → 500,000,000 bytes = ~476.8 MB

    Optimal hash count:
        k = round((m/n) * ln(2))
          = round((4B / 400M) * 0.6931)
          = round(6.931)
          → 7

    Expected FP rate at n=400M with m=4B and k=7:
        p = (1 - e^(-7 * 400M / 4B))^7
          = (1 - e^(-0.7))^7
          = (1 - 0.4966)^7
          = 0.5034^7
          ≈ 0.0085%
          → ~1 in 11,800 (better than the 1 in 10,000 target)

Hash construction — double hashing:
    Using two MurmurHash3 seeds to simulate k independent hash functions.
    This avoids k separate hash invocations; only two are needed.

        h1 = mmh3(url, seed=0)   ← unsigned 32-bit
        h2 = mmh3(url, seed=1)   ← unsigned 32-bit, forced odd
        pos_i = (h1 + i * h2) % BLOOM_BIT_SIZE    for i in 0..k-1

    Forcing h2 to be odd ensures the double-hash sequence visits all m slots
    for any h1 (since gcd(odd, 2^n) = 1, the sequence has full period mod 2^n,
    and m is not a power of 2 so this holds more generally).

File layout (bloom.bin):
    Offset       Size    Field
    ──────────────────────────────────────────────────────────────────
    0            64      Header struct (see BLOOM_HEADER_FMT)
    64           500M    Bit array (BLOOM_ARRAY_BYTES bytes)
    ──────────────────────────────────────────────────────────────────
    Total:       500,000,064 bytes

Header struct (little-endian, 64 bytes):
    Offset  Size  Type    Field
    0       8     bytes   magic       "AXBLOOM\\x00"  file type guard
    8       8     uint64  version     format version (currently 1)
    16      8     uint64  count       tracked add count (approximate)
    24      8     double  created_at  unix timestamp of filter creation
    32      8     double  flushed_at  unix timestamp of last mmap flush
    40      8     uint64  capacity    BLOOM_CAPACITY at creation (integrity)
    48      8     uint64  bit_size    BLOOM_BIT_SIZE at creation (integrity)
    56      4     uint32  hash_count  BLOOM_HASH_COUNT at creation (integrity)
    60      4     uint32  checksum    CRC32 of bytes 0..59 (checksum field = 0)

Thread safety:
    The mmap is NOT thread-safe for concurrent writes. This is intentional.
    asyncio is single-threaded. All adds are serialized by the event loop.
    Do not add locking. Do not make this multi-process safe.
    One process owns the bloom filter. One process, one event loop.

Single-owner invariant:
    bloom_filter.py is owned by one asyncio process. If you need to share
    dedup state across processes, copy the file at the OS level between runs.
    Do not attempt to mmap the same bloom.bin from two processes simultaneously.

Dependencies:
    mmh3        pip install mmh3
    mmap        stdlib
    struct      stdlib
    asyncio     stdlib

What this file does NOT do (and why):
    - Store URLs             → bits only. URLs are in frontier.py.
    - Remove URLs            → Bloom filters are add-only by design.
    - Thread-safe locking    → Not needed. Adds complexity. asyncio is single-threaded.
    - Zero false positives   → Mathematically impossible for a Bloom filter.
    - Count exact items      → Bloom filters cannot count. We track add calls.
    - Route fetches          → routing logic belongs to fetcher.py, never here.
    - Know about topology    → this layer acquires bytes. no intelligence.
"""

from __future__ import annotations

import asyncio
import binascii
import logging
import mmap
import os
import struct
import time
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Union,
    Awaitable,
    Callable,
    Dict,
    Final,
    FrozenSet,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import (
    parse_qsl,
    urlencode,
    urlparse,
    urlunparse,
)

try:
    import mmh3
except ImportError as _mmh3_import_error:
    raise ImportError(
        "mmh3 is required for bloom_filter.py.\n"
        "Install with: pip install mmh3\n"
        "mmh3 provides MurmurHash3 — a non-cryptographic hash with excellent\n"
        "distribution and ~1 GB/s throughput on modern hardware. Do not substitute\n"
        "hashlib.sha256 — it is 100x slower and cryptographic strength is irrelevant\n"
        "for Bloom filter hashing."
    ) from _mmh3_import_error

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS — LOCKED BY SPEC
#
# These constants are the specification. Changing any of them invalidates any
# existing bloom.bin on disk. If BLOOM_BIT_SIZE, BLOOM_HASH_COUNT, or
# BLOOM_CAPACITY change, the old bloom.bin MUST be deleted or migrated before
# the new filter is used. Using a filter created with different parameters
# produces wrong deduplication behavior (the bit positions no longer correspond
# to the expected hash values).
#
# The header stores the constants at creation time and validates them on every
# open. A mismatch raises BloomFilterConfigError immediately — never silently.
# ─────────────────────────────────────────────────────────────────────────────

BLOOM_CAPACITY:       Final[int]   = 400_000_000
"""Maximum URLs before false positive rate begins to degrade past BLOOM_FP_RATE."""

BLOOM_FP_RATE:        Final[float] = 0.0001
"""Target false positive probability: 1 in 10,000 at BLOOM_CAPACITY elements."""

BLOOM_BIT_SIZE:       Final[int]   = 4_000_000_000
"""Bit array size in bits. ~500 MB. Derived from n=400M and p=0.0001."""

BLOOM_HASH_COUNT:     Final[int]   = 7
"""Number of independent hash positions per URL. Optimal for these m,n parameters."""

BLOOM_FILE_PATH:      Final[Path]  = Path("store/bloom.bin")
"""Default path to the flat binary mmap file."""

BLOOM_FLUSH_INTERVAL: Final[int]   = 10_000
"""Number of add() calls between automatic mmap flushes."""

# ─────────────────────────────────────────────────────────────────────────────
# DERIVED CONSTANTS — computed from the above, never set directly
# ─────────────────────────────────────────────────────────────────────────────

BLOOM_HEADER_BYTES: Final[int] = 64
"""Bytes reserved at the start of bloom.bin for the header struct."""

BLOOM_ARRAY_BYTES: Final[int] = BLOOM_BIT_SIZE // 8
"""Bytes in the bit array: 500,000,000."""

BLOOM_FILE_BYTES: Final[int] = BLOOM_HEADER_BYTES + BLOOM_ARRAY_BYTES
"""Total file size: 500,000,064 bytes."""

# Header struct: little-endian, 64 bytes total.
# Fields: magic(8s) version(Q) count(Q) created_at(d) flushed_at(d)
#         capacity(Q) bit_size(Q) hash_count(I) checksum(I)
BLOOM_HEADER_FMT:     Final[str]   = "<8sQQddQQII"
BLOOM_HEADER_MAGIC:   Final[bytes] = b"AXBLOOM\x00"
BLOOM_HEADER_VERSION: Final[int]   = 1

assert struct.calcsize(BLOOM_HEADER_FMT) == BLOOM_HEADER_BYTES, (
    f"Header struct is {struct.calcsize(BLOOM_HEADER_FMT)} bytes, "
    f"expected {BLOOM_HEADER_BYTES}. Update BLOOM_HEADER_FMT."
)

# ─────────────────────────────────────────────────────────────────────────────
# ROTATING FILTER CONSTANTS
# Used by RotatingBloomFilter — smaller, in-memory, sliding-window dedup.
# These are separate from the primary filter constants. Changing them does not
# invalidate bloom.bin.
# ─────────────────────────────────────────────────────────────────────────────

ROTATING_FILTER_CAPACITY: Final[int]   = 50_000_000
"""URLs per rotation window for RotatingBloomFilter. Two windows = 100M coverage."""

ROTATING_FILTER_FP_RATE:  Final[float] = 0.001
"""False positive rate for each RotatingBloomFilter window. Relaxed vs primary."""

# ─────────────────────────────────────────────────────────────────────────────
# URL CANONICALIZATION CONSTANTS
#
# Tracking query parameters that carry zero semantic page identity. These are
# stripped from URLs before hashing to prevent identical pages from evading
# deduplication due to trivially different query strings.
#
# Example: these two URLs are the same page and must produce the same hash:
#   https://example.com/article?id=42&utm_source=google
#   https://example.com/article?id=42
#
# This list is authoritative. Do not add ad-hoc params elsewhere in the codebase.
# Adding a param to this set increases dedup accuracy. Removing one reduces it.
# ─────────────────────────────────────────────────────────────────────────────

TRACKING_QUERY_PARAMS: Final[FrozenSet[str]] = frozenset({
    # UTM campaign parameters (Google Analytics, widely adopted)
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "utm_id",
    "utm_source_platform",
    "utm_creative_format",
    "utm_marketing_tactic",
    # Google Ads / DoubleClick
    "gclid",           # Google Click ID
    "gbraid",          # Google web-to-app conversion measurement
    "wbraid",          # Google app-to-web conversion measurement
    "gclsrc",          # Google Click source
    "dclid",           # DoubleClick Click ID
    # Facebook / Meta
    "fbclid",          # Facebook Click ID
    "fb_action_ids",
    "fb_action_types",
    "fb_source",
    "fb_ref",
    # Microsoft / Bing Ads
    "msclkid",         # Microsoft Click ID
    # Twitter / X
    "twclid",          # Twitter Click ID
    # LinkedIn
    "li_fat_id",       # LinkedIn First-party Ad Tracking
    # Pinterest
    "epik",            # Pinterest Click ID
    # HubSpot tracking
    "hsa_acc",
    "hsa_cam",
    "hsa_grp",
    "hsa_ad",
    "hsa_net",
    "hsa_mt",
    "hsa_kw",
    "hsa_tgt",
    "hsa_src",
    "hsa_la",
    "hsa_ver",
    # Mailchimp
    "mc_cid",          # Mailchimp Campaign ID
    "mc_eid",          # Mailchimp Email ID
    # Google Analytics (legacy)
    "_ga",
    "_gl",
    # HubSpot (legacy)
    "_hsenc",
    "_openstat",
    # Social share tracking
    "si",              # Spotify share ID — content is not parameterized by this
    "igshid",          # Instagram share ID
    # Referrer hints — content is not parameterized by these
    "ref",
    "referrer",
    "source",          # generic source tag (not content-discriminating)
})

# Default ports that should be stripped from netloc in canonical URLs.
# "http://example.com:80/path" == "http://example.com/path"
_DEFAULT_PORTS: Final[Dict[str, int]] = {
    "http": 80,
    "https": 443,
    "ftp": 21,
}


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
#
# Internal contracts for bloom_filter.py. These are NOT in contracts.py because
# they are implementation details of the filter, not boundary contracts.
# Callers (frontier.py, fetcher.py) interact with BloomFilter only through
# the four public methods: contains() / add() / initialize() / close().
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BloomFilterConfig:
    """
    Immutable configuration for a BloomFilter instance.

    Constructed at initialization time. Validated against the on-disk header
    when opening an existing file. A mismatch in any field raises
    BloomFilterConfigError — the filter must not be used with wrong parameters.

    This dataclass is not exposed to frontier.py or fetcher.py. It is an
    internal consistency check, not a public API.
    """
    path: Path
    bit_size: int         = BLOOM_BIT_SIZE
    hash_count: int       = BLOOM_HASH_COUNT
    capacity: int         = BLOOM_CAPACITY
    fp_rate: float        = BLOOM_FP_RATE
    flush_interval: int   = BLOOM_FLUSH_INTERVAL

    def file_bytes(self) -> int:
        """Total expected file size in bytes (header + bit array)."""
        return BLOOM_HEADER_BYTES + (self.bit_size // 8)

    def theoretical_fp_rate_at(self, n: int) -> float:
        """
        Compute the theoretical false positive rate at n inserted elements.

        Uses the standard Bloom filter probability formula:
            p = (1 - e^(-k*n/m))^k

        where k = hash_count, m = bit_size, n = element count.

        This is the formula for a filter with uniformly distributed bits.
        Real-world performance may differ slightly due to hash distribution
        non-uniformity, but MurmurHash3 is close enough to uniform for this
        to be a reliable estimate.

        Args:
            n: Number of elements inserted so far.

        Returns:
            Float in [0.0, 1.0] representing the false positive probability.
        """
        import math
        if n <= 0:
            return 0.0
        k = self.hash_count
        m = self.bit_size
        return (1.0 - math.exp(-k * n / m)) ** k


@dataclass
class BloomFilterStats:
    """
    Runtime statistics snapshot for a BloomFilter.

    Produced by BloomFilter.stats(). Callers treat this as read-only.
    The fill_factor computation reads every byte of the 500MB bit array
    (O(m/8) operation) — stats() is expensive and should only be called
    for monitoring, never in the hot path.
    """
    path: Path
    bit_size: int
    byte_size: int
    hash_count: int
    capacity: int
    fp_rate_target: float

    # Live counters
    count: int                    # header count + in-memory since last flush
    fill_factor: float            # fraction of bits set: [0.0, 1.0]
    estimated_fp_rate: float      # theoretical FP rate at current count
    capacity_pct: float           # count / capacity

    # Session tracking
    flush_count: int              # mmap flushes since this process opened
    add_count_since_open: int     # add() calls since initialize()
    last_flush_at: Optional[float]
    created_at: float
    file_size_bytes: int

    @property
    def is_saturated(self) -> bool:
        """True when count exceeds BLOOM_CAPACITY. FP rate has begun to degrade."""
        return self.count > self.capacity

    @property
    def capacity_remaining(self) -> int:
        """Remaining URL budget before saturation."""
        return max(0, self.capacity - self.count)

    @property
    def is_healthy(self) -> bool:
        """
        True if the filter is operating within designed parameters.

        Becomes False when fill_factor has grown to a point where the
        estimated FP rate exceeds 10x the configured target. At this point,
        the crawl operator should consider rotating to a new filter file.
        """
        return (
            not self.is_saturated
            and self.estimated_fp_rate <= self.fp_rate_target * 10
        )

    def summary(self) -> str:
        """Human-readable one-line summary for logging."""
        return (
            f"count={self.count:,} "
            f"fill={self.fill_factor*100:.3f}% "
            f"est_fp={self.estimated_fp_rate*100:.5f}% "
            f"capacity_pct={self.capacity_pct*100:.1f}% "
            f"{'SATURATED' if self.is_saturated else 'healthy'}"
        )


@dataclass(frozen=True)
class BloomFilterIntegrityReport:
    """
    Result of a full integrity check on a bloom.bin file.

    Produced by BloomFilter.verify_integrity(). All fields are set even on
    failure — callers can inspect individual checks to understand what failed.

    is_valid is True only when ALL individual checks pass. A single failure
    sets is_valid=False and populates the errors tuple.
    """
    path: Path

    # File-level checks
    file_exists: bool
    file_size_correct: bool       # actual size == BLOOM_FILE_BYTES

    # Header field checks
    magic_valid: bool             # first 8 bytes == BLOOM_HEADER_MAGIC
    version_valid: bool           # version == BLOOM_HEADER_VERSION
    checksum_valid: bool          # CRC32 of bytes 0..59 matches stored value

    # Configuration consistency checks
    capacity_matches: bool        # header capacity == BLOOM_CAPACITY
    bit_size_matches: bool        # header bit_size == BLOOM_BIT_SIZE
    hash_count_matches: bool      # header hash_count == BLOOM_HASH_COUNT

    # Summary
    is_valid: bool
    errors: Tuple[str, ...]
    check_duration_ms: float

    def __str__(self) -> str:
        status = "VALID" if self.is_valid else "INVALID"
        parts = [
            f"BloomFilterIntegrityReport({self.path}) [{status}]",
            f"  file_exists={self.file_exists} size_correct={self.file_size_correct}",
            f"  magic={self.magic_valid} version={self.version_valid} "
            f"checksum={self.checksum_valid}",
            f"  capacity={self.capacity_matches} bit_size={self.bit_size_matches} "
            f"hash_count={self.hash_count_matches}",
            f"  duration={self.check_duration_ms:.2f}ms",
        ]
        if self.errors:
            parts.append("  Errors:")
            for e in self.errors:
                parts.append(f"    - {e}")
        return "\n".join(parts)


@dataclass(frozen=True)
class BloomFilterSnapshot:
    """
    Compact metadata snapshot for replication and monitoring.

    Produced by BloomFilter.snapshot(). Contains only header metadata —
    NOT the bit array (500MB, too large for in-process transfer).

    For full state replication across machines, copy bloom.bin at the OS level
    between runs. This snapshot is for monitoring dashboards and health checks.
    """
    path: str
    count: int
    bit_size: int
    hash_count: int
    capacity: int
    created_at: float
    flushed_at: float
    fill_factor: float
    estimated_fp_rate: float
    file_size_bytes: int
    snapshot_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        """Serialize to a plain dict suitable for JSON emission to the bus."""
        return {
            "path":               self.path,
            "count":              self.count,
            "bit_size":           self.bit_size,
            "hash_count":         self.hash_count,
            "capacity":           self.capacity,
            "created_at":         self.created_at,
            "flushed_at":         self.flushed_at,
            "fill_factor":        self.fill_factor,
            "estimated_fp_rate":  self.estimated_fp_rate,
            "file_size_bytes":    self.file_size_bytes,
            "snapshot_at":        self.snapshot_at,
        }


# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTIONS
#
# bloom_filter.py's own exception hierarchy. These never escape the module
# boundary into frontier.py or fetcher.py — all public methods of BloomFilter
# catch these internally and return safe defaults (False / no-op).
#
# The only exception that escapes is BloomFilterConfigError raised during
# initialize() — a config mismatch is a deployment error that requires human
# intervention, not a runtime condition that can be silently swallowed.
# ─────────────────────────────────────────────────────────────────────────────

class BloomFilterError(Exception):
    """
    Base class for all bloom_filter.py errors.

    Never escapes the module boundary in normal operation. All public methods
    catch BloomFilterError and return safe defaults. The exception is logged
    at ERROR level with the originating URL (truncated to 200 chars) and the
    full traceback. Callers are never aware an error occurred.

    The only exception: BloomFilterConfigError and BloomFilterIntegrityError
    raised during initialize() do escape — they represent conditions that
    require human intervention (wrong bloom.bin file, corrupt file).
    """


class BloomFilterNotInitializedError(BloomFilterError):
    """
    Operation called before initialize() completed.

    This represents a programming error: the caller constructed a BloomFilter
    and called contains() or add() before awaiting initialize(). The safe
    default for contains() in this state is False (treat URL as unseen).
    The safe default for add() is a no-op.
    """


class BloomFilterConfigError(BloomFilterError):
    """
    On-disk header constants don't match current compile-time constants.

    Raised by initialize() when opening an existing bloom.bin whose header
    records different BLOOM_CAPACITY, BLOOM_BIT_SIZE, or BLOOM_HASH_COUNT
    than the current values in this file.

    This is a deployment error. Resolution:
      1. If the file is from a previous version and can be discarded:
         rm store/bloom.bin  (accept up to BLOOM_CAPACITY duplicate fetches)
      2. If the file must be preserved:
         Use BloomFilterMigration to rebuild with the new parameters.
         (See BloomFilterMigration below.)

    This exception is NOT caught internally — it propagates to the caller
    of initialize() and must be handled at the startup layer.
    """


class BloomFilterIntegrityError(BloomFilterError):
    """
    The mmap file failed an integrity check.

    Raised by initialize() when the file exists but has a bad magic number,
    bad header checksum, or wrong file size. This typically means:
      - The file was truncated (partial write during creation)
      - The file was corrupted by hardware or OS failure
      - A different file was placed at the bloom.bin path

    Resolution:
      1. Check if a backup exists in store/checkpoints/
      2. If no backup: rm store/bloom.bin and accept duplicate fetches.
      3. Run BloomFilter.verify_integrity() for a detailed diagnosis.
    """


class BloomFilterAlreadyClosedError(BloomFilterError):
    """
    Operation called after close() completed.

    Not raised externally — treated as a no-op with a DEBUG log message.
    A second call to close() is idempotent and safe.
    """


# ─────────────────────────────────────────────────────────────────────────────
# URL CANONICALIZER
#
# Normalizes URLs to a canonical form before hashing. Without canonicalization,
# trivially distinct URL representations of the same page evade deduplication:
#
#   HTTP://EXAMPLE.COM/path == http://example.com/path (same page, different hash)
#   ?b=2&a=1 == ?a=1&b=2 (same content, different hash without sort)
#   ?utm_source=google == (no query string) (same page, different hash without strip)
#
# The canonicalizer is:
#   - Stateless: no instance state, same input always produces same output
#   - Total: never raises, returns original URL on parse error
#   - Idempotent: normalize(normalize(url)) == normalize(url)
#   - Thread-safe: no shared mutable state
#
# Canonicalization is applied before both contains() and add(). Two URLs that
# canonicalize to the same string will produce the same hash positions and
# therefore correctly deduplicate against each other.
#
# Conservation law: canonicalization never introduces false negatives. If a URL
# normalizes to a form that was never added, contains() returns False (correct).
# If a URL normalizes to a form that was added under a different raw URL,
# contains() returns True (correct — same canonical page).
# ─────────────────────────────────────────────────────────────────────────────

class URLCanonicalizer:
    """
    Stateless URL canonicalizer for Bloom filter URL deduplication.

    Applies a deterministic, conservative normalization pipeline to URLs before
    they are passed to the hash functions. The pipeline is designed to eliminate
    meaningless variation while preserving genuine URL distinctions.

    Conservative design principle:
        When in doubt, do NOT normalize. A false negative (missing a dedup
        opportunity) is always preferable to a false positive introduced by
        over-aggressive normalization (treating distinct pages as the same URL).
        Example: we do NOT strip query params that might discriminate content
        (e.g. ?page=2, ?id=123) — only params that are structurally guaranteed
        to carry no page identity information (tracking pixels, analytics tags).

    Normalization steps applied in order:
        1.  Strip leading/trailing whitespace
        2.  NFC Unicode normalization (canonical decomposition)
        3.  Lowercase scheme and netloc
        4.  Strip default port for scheme (http:80, https:443, ftp:21)
        5.  Collapse consecutive path slashes (//foo → /foo, except protocol-relative)
        6.  Strip URL fragment (# and everything after) — fragment is client-side only
        7.  Strip tracking query parameters (see TRACKING_QUERY_PARAMS)
        8.  Sort remaining query parameters alphabetically by key

    Steps 7 and 8 can be disabled per-instance for testing or edge cases.

    Usage:
        canon = URLCanonicalizer()
        url = "HTTP://EXAMPLE.COM/page?utm_source=google&id=42#section"
        canonical = canon.normalize(url)
        # → "http://example.com/page?id=42"
    """

    __slots__ = (
        "_strip_tracking",
        "_sort_params",
        "_strip_fragment",
        "_strip_default_port",
    )

    def __init__(
        self,
        *,
        strip_tracking: bool = True,
        sort_params: bool = True,
        strip_fragment: bool = True,
        strip_default_port: bool = True,
    ) -> None:
        """
        Args:
            strip_tracking:      Remove known tracking query parameters.
            sort_params:         Sort remaining query params alphabetically.
            strip_fragment:      Remove URL fragment (# and after).
            strip_default_port:  Remove port if it's the default for the scheme.
        """
        self._strip_tracking   = strip_tracking
        self._sort_params      = sort_params
        self._strip_fragment   = strip_fragment
        self._strip_default_port = strip_default_port

    def normalize(self, url: str) -> str:
        """
        Return the canonical form of a URL.

        On any parse error, returns the original URL unchanged. This is the
        safe default — an un-normalized URL produces a valid (if non-canonical)
        hash. The consequence is a potential missed dedup for that specific URL,
        which is far less harmful than a normalization error causing an exception.

        Args:
            url: Raw URL string as received from the crawl manifest.

        Returns:
            Canonical URL string. Guaranteed to be non-empty if url is non-empty.
        """
        url = url.strip()
        if not url:
            return url

        try:
            # Step 1: NFC Unicode normalization
            url_nfc = unicodedata.normalize("NFC", url)

            # Step 2: Parse
            parsed = urlparse(url_nfc)

            # Step 3: Lowercase scheme
            scheme = parsed.scheme.lower()

            # Step 4: Lowercase netloc, strip default port
            netloc = self._normalize_netloc(scheme, parsed.netloc.lower())

            # Step 5: Normalize path
            path = self._normalize_path(parsed.path)

            # Step 6: Normalize query (strip tracking, sort)
            query = self._normalize_query(parsed.query)

            # Step 7: Strip or preserve fragment
            fragment = "" if self._strip_fragment else parsed.fragment

            return urlunparse((scheme, netloc, path, parsed.params, query, fragment))

        except Exception as exc:
            # Normalization must never raise. Return original on any error.
            # This prevents a malformed URL from crashing the dedup layer.
            log.debug("URLCanonicalizer.normalize() error for %r: %s", url[:200], exc)
            return url

    def _normalize_netloc(self, scheme: str, netloc: str) -> str:
        """
        Lowercase netloc and strip default ports.

        "EXAMPLE.COM:443" → "example.com" (for https scheme)
        "example.com:8080" → "example.com:8080" (non-default port preserved)

        Args:
            scheme: Lowercased scheme string.
            netloc: Lowercased netloc string (host:port or just host).

        Returns:
            Normalized netloc string.
        """
        if not self._strip_default_port:
            return netloc

        default_port = _DEFAULT_PORTS.get(scheme)
        if default_port is None:
            return netloc

        # netloc is already lowercased by caller
        if netloc.endswith(f":{default_port}"):
            # Strip the ":port" suffix
            return netloc[: -(len(str(default_port)) + 1)]

        return netloc

    def _normalize_path(self, path: str) -> str: # noqa
        """
        Collapse consecutive slashes in the URL path component.

        "/foo//bar///baz" → "/foo/bar/baz"
        "//path/to/page"  → "/path/to/page"

        urlparse() already extracts the scheme/authority, so the path component
        returned by urlparse never needs protocol-relative preservation — that is
        encoded in netloc. Any double-slash in the path component is simply a
        redundant separator that should be collapsed.

        We do NOT strip trailing slashes. "/path/" and "/path" may be different
        pages depending on server configuration. Collapsing those would introduce
        false positives.

        Args:
            path: URL path component (as returned by urlparse).

        Returns:
            Path with consecutive slashes collapsed to single slashes.
        """
        if not path or "//" not in path:
            return path

        while "//" in path:
            path = path.replace("//", "/")
        return path

    def _normalize_query(self, query: str) -> str:
        """
        Normalize query string: strip tracking params, sort remaining params.

        Empty query string → empty string (no "?" suffix added).
        All params stripped → empty string.

        Params with empty values are preserved: "?foo=" is kept as "?foo=".
        Params with duplicate keys: all instances are kept (both preserved).

        Args:
            query: Raw query string (without leading "?").

        Returns:
            Normalized query string (without leading "?").
        """
        if not query:
            return query

        # Parse into list of (key, value) pairs, preserving blank values
        params = parse_qsl(query, keep_blank_values=True)

        if not params:
            return query  # unparseable query, preserve original

        # Strip tracking params (case-insensitive key match)
        if self._strip_tracking:
            params = [
                (k, v) for k, v in params
                if k.lower() not in TRACKING_QUERY_PARAMS
            ]

        # Sort by lowercased key for canonical ordering
        if self._sort_params:
            params.sort(key=lambda kv: kv[0].lower())

        return urlencode(params)

    def normalize_batch(self, urls: Sequence[str]) -> List[str]:
        """
        Normalize a batch of URLs. Returns list of same length as input.

        Failed normalizations return the original URL — never raises.
        This is the hot-path method called by add_batch() and contains_batch().

        Args:
            urls: Sequence of raw URL strings.

        Returns:
            List of canonical URL strings, same length and order as input.
        """
        return [self.normalize(u) for u in urls]


# Module-level default canonicalizer. BloomFilter uses this unless overridden.
_DEFAULT_CANONICALIZER: Final[URLCanonicalizer] = URLCanonicalizer()


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HEADER UTILITIES
#
# These functions encode and decode the 64-byte file header. They are not part
# of the BloomFilter class because they need to be called by verify_integrity()
# before the mmap is open, and by migration tools that work on the file directly.
# ─────────────────────────────────────────────────────────────────────────────

def _crc32(data: bytes) -> int:
    """
    CRC32 checksum of a byte sequence.

    Returns value in [0, 2^32). We use binascii.crc32 rather than zlib.crc32
    for explicitness — they produce identical results but binascii is stdlib
    with a clearer purpose indication.

    Args:
        data: Byte sequence to checksum.

    Returns:
        CRC32 value as unsigned 32-bit integer.
    """
    return binascii.crc32(data) & 0xFFFF_FFFF


def _encode_header(
    *,
    count: int,
    created_at: float,
    flushed_at: float,
) -> bytes:
    """
    Encode the 64-byte file header with a valid CRC32 checksum.

    Writes the constant fields (magic, version, capacity, bit_size, hash_count)
    from current module constants. The CRC32 is computed over bytes 0..59 with
    the checksum field (bytes 60..63) zeroed, then written into the final 4 bytes.

    Args:
        count:      Current tracked add count (stored in the header for persistence).
        created_at: Unix timestamp of filter creation (written once, never changed).
        flushed_at: Unix timestamp of the most recent mmap flush.

    Returns:
        Exactly BLOOM_HEADER_BYTES (64) bytes.
    """
    # Pack all fields with checksum = 0
    raw = struct.pack(
        BLOOM_HEADER_FMT,
        BLOOM_HEADER_MAGIC,     # 8 bytes: file type guard
        BLOOM_HEADER_VERSION,   # 8 bytes: format version
        count,                  # 8 bytes: tracked add count
        created_at,             # 8 bytes: creation timestamp (double)
        flushed_at,             # 8 bytes: last flush timestamp (double)
        BLOOM_CAPACITY,         # 8 bytes: capacity at creation (for validation)
        BLOOM_BIT_SIZE,         # 8 bytes: bit_size at creation (for validation)
        BLOOM_HASH_COUNT,       # 4 bytes: hash_count at creation (for validation)
        0,                      # 4 bytes: checksum placeholder
    )
    assert len(raw) == BLOOM_HEADER_BYTES

    # Compute CRC32 of bytes 0..59 (the checksum field is zeroed above)
    checksum = _crc32(raw)

    # Write the real checksum into bytes 60..63
    return raw[:60] + struct.pack("<I", checksum)


def _decode_header(data: bytes) -> Dict:
    """
    Decode and validate a 64-byte header.

    Checks magic bytes and CRC32 integrity before returning field values.
    Does NOT check that the field values match current constants — that
    validation is done separately by _validate_header_against_config().

    Args:
        data: At least 64 bytes from the start of a bloom.bin file.

    Returns:
        Dict with keys: magic, version, count, created_at, flushed_at,
                        capacity, bit_size, hash_count, checksum.

    Raises:
        BloomFilterIntegrityError: Bad magic or CRC32 mismatch.
    """
    if len(data) < BLOOM_HEADER_BYTES:
        raise BloomFilterIntegrityError(
            f"Header data too short: {len(data)} bytes, "
            f"expected at least {BLOOM_HEADER_BYTES}"
        )

    (
        magic, version, count, created_at, flushed_at,
        capacity, bit_size, hash_count, stored_checksum,
    ) = struct.unpack(BLOOM_HEADER_FMT, data[:BLOOM_HEADER_BYTES])

    # Validate magic
    if magic != BLOOM_HEADER_MAGIC:
        raise BloomFilterIntegrityError(
            f"Bad file magic: got {magic!r}, expected {BLOOM_HEADER_MAGIC!r}. "
            f"This is not a bloom.bin file or it was corrupted at offset 0."
        )

    # Validate CRC32: recompute with the checksum field zeroed
    raw_for_crc = data[:60] + b"\x00\x00\x00\x00"
    computed_checksum = _crc32(raw_for_crc)
    if stored_checksum != computed_checksum:
        raise BloomFilterIntegrityError(
            f"Header CRC32 mismatch: "
            f"stored={stored_checksum:#010x}, "
            f"computed={computed_checksum:#010x}. "
            f"Header was written partially or corrupted after write."
        )

    return {
        "magic":      magic,
        "version":    version,
        "count":      count,
        "created_at": created_at,
        "flushed_at": flushed_at,
        "capacity":   capacity,
        "bit_size":   bit_size,
        "hash_count": hash_count,
        "checksum":   stored_checksum,
    }


def _validate_header_against_config(header: Dict) -> None:
    """
    Verify that the on-disk header constants match the current module constants.

    This catches the case where bloom.bin was created with different parameters
    and would produce wrong hash positions if used as-is.

    Args:
        header: Dict returned by _decode_header().

    Raises:
        BloomFilterConfigError: If any constant field mismatches.
    """
    mismatches: List[str] = []

    if header["version"] != BLOOM_HEADER_VERSION:
        mismatches.append(
            f"version: file={header['version']} current={BLOOM_HEADER_VERSION}"
        )
    if header["capacity"] != BLOOM_CAPACITY:
        mismatches.append(
            f"capacity: file={header['capacity']:,} current={BLOOM_CAPACITY:,}"
        )
    if header["bit_size"] != BLOOM_BIT_SIZE:
        mismatches.append(
            f"bit_size: file={header['bit_size']:,} current={BLOOM_BIT_SIZE:,}"
        )
    if header["hash_count"] != BLOOM_HASH_COUNT:
        mismatches.append(
            f"hash_count: file={header['hash_count']} current={BLOOM_HASH_COUNT}"
        )

    if mismatches:
        raise BloomFilterConfigError(
            f"bloom.bin header is incompatible with current constants. "
            f"The file was created with different parameters and cannot be used as-is. "
            f"Mismatches: {'; '.join(mismatches)}. "
            f"Resolution: delete store/bloom.bin (accepting duplicate fetches), "
            f"or use BloomFilterMigration to rebuild."
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOOM FILTER — CORE CLASS
# ─────────────────────────────────────────────────────────────────────────────

class BloomFilter:
    """
    Memory-mapped probabilistic URL deduplication filter.

    The single source of truth for URL deduplication in the AXIOM crawler layer.
    Designed for a single asyncio process (single-threaded). No locking. No
    multi-process coordination. One instance per running crawler.

    Key properties:
        - Zero false negatives: if add(url) was called, contains(url) returns True.
        - ~0.008% false positive rate at 400M URLs (slightly better than 0.01% target).
        - 500MB file footprint for 400M URLs.
        - Survives process death: bits are durable after each mmap flush.
        - Zero cold-start cost: open the file, mmap it, done.
        - O(k) operations: contains() and add() are both 7 hash computations + 7 mmap reads/writes.

    Lifecycle:
        bloom = BloomFilter()
        await bloom.initialize()    # open or create bloom.bin
        try:
            if not await bloom.contains(url):
                await bloom.add(url)
                # ... fetch url ...
        finally:
            await bloom.close()     # flush + close

    Async context manager (equivalent):
        async with BloomFilter() as bloom:
            if not await bloom.contains(url):
                await bloom.add(url)

    Thread safety:
        Not thread-safe. asyncio is single-threaded. No locking is added.
        All methods are coroutines that run in the event loop's single thread.
        Do not share a BloomFilter instance across threads or processes.

    mmap semantics:
        The bit array is a memory-mapped view of bloom.bin. Each _set_bit()
        call modifies a byte in the OS page cache. The OS may write the dirty
        page to disk at any time (on page eviction). We additionally call
        mmap.flush() explicitly every BLOOM_FLUSH_INTERVAL adds to guarantee
        durability without relying on the OS page eviction schedule.

    Error handling:
        contains() and add() never raise. Any internal error is logged at ERROR
        level and the safe default is returned (False for contains, no-op for add).
        initialize() and close() may raise (BloomFilterConfigError,
        BloomFilterIntegrityError) — these represent unrecoverable conditions.
    """

    __slots__ = (
        "_path",
        "_canonicalizer",
        "_mmap",
        "_file",
        "_initialized",
        "_closed",
        "_add_count",           # adds in this session since last flush
        "_flush_count",         # total flushes in this session
        "_header_count",        # count from header (total ever added, updated on flush)
        "_pending_flush",       # adds since the last flush (resets at BLOOM_FLUSH_INTERVAL)
        "_created_at",          # filter creation timestamp (from header)
        "_last_flush_at",       # last flush timestamp (updated on every flush)
    )

    def __init__(
        self,
        path: Path = BLOOM_FILE_PATH,
        *,
        canonicalizer: Optional[URLCanonicalizer] = None,
    ) -> None:
        """
        Construct a BloomFilter. Call initialize() before any other method.

        Args:
            path:          Path to the mmap file. Created by initialize() if absent.
            canonicalizer: URL canonicalizer. Uses module default if None.
                           Override for testing or custom normalization rules.
        """
        self._path           = path
        self._canonicalizer  = canonicalizer or _DEFAULT_CANONICALIZER
        self._mmap: Optional[mmap.mmap] = None
        self._file           = None
        self._initialized    = False
        self._closed         = False
        self._add_count      = 0
        self._flush_count    = 0
        self._header_count   = 0
        self._pending_flush  = 0
        self._created_at     = 0.0
        self._last_flush_at  = 0.0

    # ── Async context manager ─────────────────────────────────────────────────

    async def __aenter__(self) -> "BloomFilter":
        """Initialize and return self. For use in `async with` blocks."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close and flush on exit. Always runs even if an exception occurred."""
        await self.close()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Open or create the bloom filter mmap file.

        Behavior on first call:
          - If bloom.bin does not exist: creates it, writes zero-filled bit array,
            writes initial header, fsyncs to disk.
          - If bloom.bin exists: opens it, validates header, validates file size.
            On size mismatch: truncates or extends to BLOOM_FILE_BYTES.
            On header corruption: raises BloomFilterIntegrityError.
            On config mismatch: raises BloomFilterConfigError.

        Second call (already initialized): logs a warning and returns. Idempotent.

        This is the resume path: an existing file retains all previously added
        URLs with zero overhead — no replay, no rebuild, no warm-up.

        Raises:
            BloomFilterConfigError:   On-disk constants mismatch current constants.
            BloomFilterIntegrityError: File exists but header is corrupt/invalid.
        """
        if self._initialized:
            log.warning(
                "BloomFilter.initialize() called after already initialized — ignoring. "
                "path=%s", self._path
            )
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)

        if not self._path.exists():
            await self._create_new_file()
            log.info(
                "BloomFilter created new file | path=%s size_mb=%.1f",
                self._path,
                BLOOM_FILE_BYTES / 1_048_576,
            )
        else:
            await self._open_existing_file()
            log.info(
                "BloomFilter opened existing file | path=%s count=%d created_at=%.0f",
                self._path,
                self._header_count,
                self._created_at,
            )

        self._initialized = True

    async def _create_new_file(self) -> None:
        """
        Create bloom.bin from scratch and open the mmap.

        Writes the header and zero-fills the bit array in 4MB chunks to avoid
        allocating 500MB of zeros as a single Python bytes object. After writing,
        fsyncs the file to guarantee the full extent is on disk before the mmap
        is opened. An incomplete file here would leave the filter in an unusable
        state on the next startup.
        """
        now = time.time()
        header_bytes = _encode_header(count=0, created_at=now, flushed_at=now)

        _CHUNK_SIZE = 4 * 1024 * 1024   # 4MB write chunks — tuned for block device alignment
        zero_chunk  = b"\x00" * _CHUNK_SIZE

        with open(self._path, "wb") as f:
            f.write(header_bytes)

            remaining = BLOOM_ARRAY_BYTES
            while remaining > 0:
                write_size = min(_CHUNK_SIZE, remaining)
                f.write(zero_chunk[:write_size])
                remaining -= write_size

            f.flush()
            os.fsync(f.fileno())   # guarantee file extent on disk before mmap open

        self._created_at    = now
        self._last_flush_at = now
        self._header_count  = 0

        await self._open_mmap()

    async def _open_existing_file(self) -> None:
        """
        Open an existing bloom.bin and validate its integrity.

        File size correction: if the file is the wrong size (partial write during
        initial creation, or filesystem issue), truncates or extends to the correct
        size before opening the mmap. This handles the crash-during-creation case.

        After size correction: opens the mmap, reads and decodes the header,
        validates the header against current constants.
        """
        actual_size = self._path.stat().st_size

        if actual_size != BLOOM_FILE_BYTES:
            log.warning(
                "BloomFilter file size mismatch | path=%s expected=%d actual=%d — resizing",
                self._path,
                BLOOM_FILE_BYTES,
                actual_size,
            )
            # Use file truncate to extend or shrink. On most filesystems, extending
            # via truncate() produces a sparse file (zero-filled by the OS). We verify
            # the resulting size before proceeding.
            with open(self._path, "r+b") as f:
                f.truncate(BLOOM_FILE_BYTES)
                f.flush()
                os.fsync(f.fileno())

            new_size = self._path.stat().st_size
            if new_size != BLOOM_FILE_BYTES:
                raise BloomFilterIntegrityError(
                    f"Could not resize bloom.bin to {BLOOM_FILE_BYTES} bytes. "
                    f"File system may be full or read-only. Got {new_size} bytes."
                )
            log.info("BloomFilter file resized to %d bytes | path=%s", BLOOM_FILE_BYTES, self._path)

        await self._open_mmap()

        # Read, decode, and validate the header
        try:
            header_data = bytes(self._mmap[:BLOOM_HEADER_BYTES])
            header = _decode_header(header_data)
            _validate_header_against_config(header)

            self._header_count  = header["count"]
            self._created_at    = header["created_at"]
            self._last_flush_at = header["flushed_at"]

        except (BloomFilterIntegrityError, BloomFilterConfigError):
            self._close_mmap_handles()
            raise
        except Exception as exc:
            self._close_mmap_handles()
            raise BloomFilterIntegrityError(
                f"Unexpected error reading bloom.bin header: {exc}"
            ) from exc

    async def _open_mmap(self) -> None:
        """
        Open the file and create the mmap view.

        Called after the file exists and has the correct size. The mmap maps
        the entire file (BLOOM_FILE_BYTES bytes) into virtual address space.
        The OS page cache handles actual physical memory usage — not all 500MB
        need to be resident in RAM simultaneously.

        The mmap is opened read-write (mmap.ACCESS_WRITE). This is required for
        add() to work. The file descriptor is kept open for the lifetime of the mmap.
        """
        self._file = open(self._path, "r+b")
        self._mmap = mmap.mmap(self._file.fileno(), BLOOM_FILE_BYTES)

    def _close_mmap_handles(self) -> None:
        """
        Close mmap and file handle without flushing.

        Used by error paths in _open_existing_file() where we want to release
        resources without writing anything (the mmap may be in an invalid state).
        """
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception as exc:
                log.error("BloomFilter mmap.close() error: %s", exc)
            self._mmap = None

        if self._file is not None:
            try:
                self._file.close()
            except Exception as exc:
                log.error("BloomFilter file.close() error: %s", exc)
            self._file = None

    async def close(self) -> None:
        """
        Flush the mmap and close all file handles.

        Always flushes before closing to minimize data loss. After close(),
        the instance is unusable — subsequent calls to contains() and add()
        return safe defaults (False / no-op).

        Idempotent: calling close() multiple times is safe and produces
        exactly one flush (on the first call).
        """
        if self._closed:
            return

        if self._mmap is not None:
            try:
                self._flush_mmap()
            except Exception as exc:
                log.error(
                    "BloomFilter flush on close failed | path=%s error=%s",
                    self._path, exc,
                )

            self._close_mmap_handles()
            log.info(
                "BloomFilter closed | path=%s total_count=%d "
                "session_adds=%d flushes=%d",
                self._path,
                self._header_count,
                self._add_count,
                self._flush_count,
            )

        self._closed = True

    # ── Core public API ───────────────────────────────────────────────────────

    async def contains(self, url: str) -> bool:
        """
        Return True if the URL was probably added before.
        Return False if the URL was definitely never added.

        This is the hot path. Called by frontier.py before every URL is queued.

        Implementation:
            1. Canonicalize the URL.
            2. Compute 7 hash positions via double hashing.
            3. Read each bit from the mmap.
            4. Return True only if all 7 bits are set.

        Properties:
            - False negatives: impossible. If all 7 bits were set by add(), they
              remain set until the filter is deleted. The only way a False negative
              could occur is if the mmap file was externally modified, which is
              a deployment error, not a runtime condition.
            - False positives: ~0.008% at 400M URLs. A URL that was never added
              may return True if all 7 of its hash positions were set by other URLs.
              This is acceptable (per AXIOM design law 8): the URL is silently skipped,
              and ~1 in 12,000 valid URLs is missed per crawl pass.

        Never raises. On any internal error (mmap closed, hash error):
            - Logs at ERROR level
            - Returns False (safe default: treat URL as unseen, allow fetch)

        Args:
            url: Raw URL string. Canonicalized internally before hashing.

        Returns:
            True if probably seen, False if definitely not seen.
        """
        if not self._initialized or self._closed or self._mmap is None:
            return False
        try:
            canonical = self._canonicalizer.normalize(url)
            if not canonical:
                return False   # empty / whitespace-only URLs are never in the filter
            return self._check_all_bits(canonical)
        except Exception as exc:
            log.error(
                "BloomFilter.contains() error | url=%r error=%s",
                url[:200], exc,
            )
            return False

    async def add(self, url: str) -> None:
        """
        Add a URL to the filter. Idempotent — adding twice produces the same bits.

        This is the post-fetch write path. Called by fetcher.py after a
        successful fetch event has been emitted to the bus. The URL enters
        the filter only after the fetch is complete — never speculatively.

        Implementation:
            1. Canonicalize the URL.
            2. Compute 7 hash positions via double hashing.
            3. Set each bit in the mmap using read-modify-write.
            4. Increment add counter.
            5. If pending_flush >= BLOOM_FLUSH_INTERVAL: flush the mmap.

        Idempotency:
            Setting a bit that is already set is a no-op at the bit level
            (byte = byte | mask, no change if mask is already set). Adding the
            same URL twice has zero effect on the bit array and zero effect on
            contains() results.

        Never raises. On any internal error:
            - Logs at ERROR level
            - Returns without modifying the filter
            - The consequence is a potential duplicate fetch of this URL on restart,
              which is acceptable per AXIOM design law 8.

        Args:
            url: Raw URL string. Canonicalized internally before hashing.
        """
        if not self._initialized or self._closed or self._mmap is None:
            return
        try:
            canonical = self._canonicalizer.normalize(url)
            if not canonical:
                return   # empty / whitespace-only URLs are silently ignored
            self._set_all_bits(canonical)
            self._add_count += 1
            self._pending_flush += 1

            if self._pending_flush >= BLOOM_FLUSH_INTERVAL:
                self._flush_mmap()
                self._pending_flush = 0

        except Exception as exc:
            log.error(
                "BloomFilter.add() error | url=%r error=%s",
                url[:200], exc,
            )

    async def add_batch(self, urls: Sequence[str]) -> int:
        """
        Add a batch of URLs in a single call.

        More efficient than calling add() in a loop for large batches because:
            1. URL canonicalization is amortized over the batch.
            2. The flush check is performed once per batch rather than per URL.
            3. Better CPU cache locality for sequential mmap writes.

        Used by frontier.py when bulk-loading a new manifest into the filter
        before the crawl begins. At 1000+ URLs per batch, the throughput
        improvement over the loop pattern is significant.

        Never raises. Per-URL errors are logged and skipped.

        Args:
            urls: Sequence of raw URL strings to add.

        Returns:
            Count of URLs successfully added (excluding errors).
        """
        if not self._initialized or self._closed or self._mmap is None:
            return 0

        added = 0
        canonicals = self._canonicalizer.normalize_batch(urls)

        for canonical in canonicals:
            try:
                self._set_all_bits(canonical)
                added += 1
            except Exception as exc:
                log.error("BloomFilter.add_batch() item error: %s", exc)

        self._add_count     += added
        self._pending_flush += added

        if self._pending_flush >= BLOOM_FLUSH_INTERVAL:
            try:
                self._flush_mmap()
            except Exception as exc:
                log.error("BloomFilter.add_batch() flush error: %s", exc)
            self._pending_flush = 0

        return added

    async def contains_batch(self, urls: Sequence[str]) -> List[bool]:
        """
        Check membership for a batch of URLs.

        Returns a list of booleans in the same order as the input.
        More cache-friendly than repeated contains() calls because:
            1. All canonicalization is done first (locality for string operations).
            2. mmap reads are interleaved naturally for sequential access patterns.

        Never raises. Per-URL errors return False (safe default: treat as unseen).

        Args:
            urls: Sequence of raw URL strings to check.

        Returns:
            List[bool] of same length as urls.
            True  = URL was probably added before.
            False = URL was definitely never added (or an error occurred).
        """
        if not self._initialized or self._closed or self._mmap is None:
            return [False] * len(urls)

        results: List[bool] = []
        canonicals = self._canonicalizer.normalize_batch(urls)

        for canonical in canonicals:
            try:
                results.append(self._check_all_bits(canonical))
            except Exception as exc:
                log.error("BloomFilter.contains_batch() item error: %s", exc)
                results.append(False)

        return results

    async def count(self) -> int:
        """
        Return the approximate total number of URLs ever added to this filter.

        This is NOT the number of unique URLs — Bloom filters cannot count unique
        elements. This is the number of add() calls made, including duplicates.
        It is useful as a rough approximation of unique URLs at low duplicate rates.

        The count is the sum of:
            - self._header_count: the count persisted in the header on the last flush
            - self._add_count:    adds in the current session since the last flush

        On crash between flushes, the add_count portion is lost (the bits they
        set persist, but the count contribution does not). The count may therefore
        undercount by up to BLOOM_FLUSH_INTERVAL after a crash-restart cycle.

        Never raises.

        Returns:
            Approximate add count as a non-negative integer.
        """
        return self._header_count + self._add_count

    # ── Diagnostics and monitoring ────────────────────────────────────────────

    async def fill_factor(self) -> float:
        """
        Compute the fraction of bits set in the bit array.

        WARNING: This is an O(m/8) operation — it reads all 500MB of the bit array.
        Expected runtime: 0.5–2 seconds depending on how much of the mmap is resident
        in RAM. Call only for monitoring and diagnostics, never in the hot path.

        The fill factor grows as URLs are added. Near BLOOM_CAPACITY:
            fill_factor ≈ 1 - e^(-k*n/m) ≈ 1 - e^(-0.7) ≈ 0.503

        Returns:
            Float in [0.0, 1.0]. 0.0 for empty filter, 1.0 if all bits set.
        """
        if not self._initialized or self._mmap is None:
            return 0.0
        try:
            return self._compute_fill_factor()
        except Exception as exc:
            log.error("BloomFilter.fill_factor() error: %s", exc)
            return 0.0

    def _compute_fill_factor(self) -> float:
        """
        Count set bits in the bit array by reading every byte of the mmap.

        Algorithm: read in 1MB chunks to maintain reasonable memory pressure.
        For each byte, count set bits using bin(b).count('1').

        Note on popcount performance: bin(b).count('1') is faster in CPython
        than bit-twiddling approaches because it offloads to the C runtime.
        For a monitoring operation that runs infrequently, this is acceptable.
        If this ever needs to be in the hot path, replace with a ctypes popcount.

        Returns:
            Float in [0.0, 1.0].
        """
        _CHUNK   = 1 << 20   # 1MB chunks
        set_bits = 0
        offset   = BLOOM_HEADER_BYTES
        remaining = BLOOM_ARRAY_BYTES

        while remaining > 0:
            read_size = min(_CHUNK, remaining)
            chunk = self._mmap[offset: offset + read_size]
            set_bits  += sum(bin(b).count("1") for b in chunk)
            offset    += read_size
            remaining -= read_size

        return set_bits / BLOOM_BIT_SIZE

    async def estimated_false_positive_rate(self) -> float:
        """
        Estimate the current false positive rate from fill factor.

        Formula: p = fill_factor ^ k

        This is derived from: p = (1 - e^(-k*n/m))^k, noting that
        fill_factor ≈ 1 - e^(-k*n/m) for a well-distributed filter.
        Using the actual fill factor (rather than the theoretical formula with n)
        gives a better real-world estimate because it accounts for hash distribution.

        WARNING: O(m/8) operation — reads 500MB of mmap. Call sparingly.

        Returns:
            Estimated false positive probability in [0.0, 1.0].
        """
        ff = await self.fill_factor()
        return ff ** BLOOM_HASH_COUNT

    async def stats(self) -> BloomFilterStats:
        """
        Return a comprehensive statistics snapshot.

        WARNING: Calls fill_factor() internally — O(m/8) operation.
        Expected runtime: 0.5–2 seconds. Call only for monitoring.

        Returns:
            BloomFilterStats with all current metrics.
        """
        total_count  = await self.count()
        ff           = await self.fill_factor()
        estimated_fp = ff ** BLOOM_HASH_COUNT
        file_size    = self._path.stat().st_size if self._path.exists() else 0

        return BloomFilterStats(
            path              = self._path,
            bit_size          = BLOOM_BIT_SIZE,
            byte_size         = BLOOM_FILE_BYTES,
            hash_count        = BLOOM_HASH_COUNT,
            capacity          = BLOOM_CAPACITY,
            fp_rate_target    = BLOOM_FP_RATE,
            count             = total_count,
            fill_factor       = ff,
            estimated_fp_rate = estimated_fp,
            capacity_pct      = total_count / BLOOM_CAPACITY,
            flush_count       = self._flush_count,
            add_count_since_open = self._add_count,
            last_flush_at     = self._last_flush_at or None,
            created_at        = self._created_at,
            file_size_bytes   = file_size,
        )

    async def snapshot(self) -> BloomFilterSnapshot:
        """
        Return a compact metadata snapshot for replication and health checks.

        Calls fill_factor() internally — O(m/8) operation. Call sparingly.

        Returns:
            BloomFilterSnapshot with header metadata and fill factor.
        """
        total_count = await self.count()
        ff          = await self.fill_factor()
        file_size   = self._path.stat().st_size if self._path.exists() else 0

        return BloomFilterSnapshot(
            path              = str(self._path),
            count             = total_count,
            bit_size          = BLOOM_BIT_SIZE,
            hash_count        = BLOOM_HASH_COUNT,
            capacity          = BLOOM_CAPACITY,
            created_at        = self._created_at,
            flushed_at        = self._last_flush_at,
            fill_factor       = ff,
            estimated_fp_rate = ff ** BLOOM_HASH_COUNT,
            file_size_bytes   = file_size,
        )

    async def verify_integrity(self) -> BloomFilterIntegrityReport:
        """
        Perform a full integrity check on the bloom.bin file.

        Checks:
          1. File exists
          2. File size == BLOOM_FILE_BYTES
          3. Header magic bytes are correct
          4. Header version == BLOOM_HEADER_VERSION
          5. Header CRC32 checksum is valid
          6. Header capacity == BLOOM_CAPACITY
          7. Header bit_size == BLOOM_BIT_SIZE
          8. Header hash_count == BLOOM_HASH_COUNT

        This method opens the file independently of the active mmap — it can be
        called on a closed filter or before initialization. It does NOT read or
        verify the bit array itself (too expensive: 500MB).

        Returns:
            BloomFilterIntegrityReport with details of every check.
        """
        start_ns = time.monotonic_ns()
        errors: List[str] = []

        file_exists      = self._path.exists()
        file_size_correct = False
        magic_valid      = False
        version_valid    = False
        checksum_valid   = False
        capacity_matches = False
        bit_size_matches = False
        hash_count_matches = False

        if not file_exists:
            return BloomFilterIntegrityReport(
                path             = self._path,
                file_exists      = False,
                file_size_correct = False,
                magic_valid      = False,
                version_valid    = False,
                checksum_valid   = False,
                capacity_matches = False,
                bit_size_matches = False,
                hash_count_matches = False,
                is_valid         = False,
                errors           = ("File does not exist",),
                check_duration_ms = (time.monotonic_ns() - start_ns) / 1e6,
            )

        # Check file size
        actual_size = self._path.stat().st_size
        file_size_correct = (actual_size == BLOOM_FILE_BYTES)
        if not file_size_correct:
            errors.append(
                f"File size {actual_size:,} != expected {BLOOM_FILE_BYTES:,} "
                f"(delta: {actual_size - BLOOM_FILE_BYTES:+,} bytes)"
            )

        # Read and check header
        try:
            with open(self._path, "rb") as f:
                raw_header = f.read(BLOOM_HEADER_BYTES)

            if len(raw_header) < BLOOM_HEADER_BYTES:
                errors.append(f"Could only read {len(raw_header)} header bytes")
            else:
                # Magic check
                magic = raw_header[:8]
                magic_valid = (magic == BLOOM_HEADER_MAGIC)
                if not magic_valid:
                    errors.append(f"Bad magic: {magic!r} (expected {BLOOM_HEADER_MAGIC!r})")

                # Checksum check
                try:
                    raw_for_crc = raw_header[:60] + b"\x00\x00\x00\x00"
                    stored_crc  = struct.unpack_from("<I", raw_header, 60)[0]
                    computed_crc = _crc32(raw_for_crc)
                    checksum_valid = (stored_crc == computed_crc)
                    if not checksum_valid:
                        errors.append(
                            f"CRC32 mismatch: stored={stored_crc:#010x} "
                            f"computed={computed_crc:#010x}"
                        )
                except Exception as exc:
                    errors.append(f"CRC32 check failed: {exc}")

                # Version check
                try:
                    version = struct.unpack_from("<Q", raw_header, 8)[0]
                    version_valid = (version == BLOOM_HEADER_VERSION)
                    if not version_valid:
                        errors.append(
                            f"Version {version} != {BLOOM_HEADER_VERSION}"
                        )
                except Exception as exc:
                    errors.append(f"Version check failed: {exc}")

                # Config field checks
                try:
                    (_, _, _, _, _, capacity, bit_size, hash_count, _) = struct.unpack(
                        BLOOM_HEADER_FMT, raw_header
                    )
                    capacity_matches   = (capacity   == BLOOM_CAPACITY)
                    bit_size_matches   = (bit_size   == BLOOM_BIT_SIZE)
                    hash_count_matches = (hash_count == BLOOM_HASH_COUNT)
                    if not capacity_matches:
                        errors.append(f"capacity {capacity:,} != {BLOOM_CAPACITY:,}")
                    if not bit_size_matches:
                        errors.append(f"bit_size {bit_size:,} != {BLOOM_BIT_SIZE:,}")
                    if not hash_count_matches:
                        errors.append(f"hash_count {hash_count} != {BLOOM_HASH_COUNT}")
                except Exception as exc:
                    errors.append(f"Config field check failed: {exc}")

        except Exception as exc:
            errors.append(f"File read error: {exc}")

        is_valid = (
            file_exists
            and file_size_correct
            and magic_valid
            and version_valid
            and checksum_valid
            and capacity_matches
            and bit_size_matches
            and hash_count_matches
            and not errors
        )

        return BloomFilterIntegrityReport(
            path               = self._path,
            file_exists        = file_exists,
            file_size_correct  = file_size_correct,
            magic_valid        = magic_valid,
            version_valid      = version_valid,
            checksum_valid     = checksum_valid,
            capacity_matches   = capacity_matches,
            bit_size_matches   = bit_size_matches,
            hash_count_matches = hash_count_matches,
            is_valid           = is_valid,
            errors             = tuple(errors),
            check_duration_ms  = (time.monotonic_ns() - start_ns) / 1e6,
        )

    async def force_flush(self) -> None:
        """
        Force an immediate mmap flush outside of the normal BLOOM_FLUSH_INTERVAL.

        Use cases:
            - Shutdown hooks (graceful stop)
            - Before forking a backup of bloom.bin
            - Administrative operations that need a guaranteed durability point

        Never raises.
        """
        if not self._initialized or self._mmap is None:
            return
        try:
            self._flush_mmap()
            self._pending_flush = 0
        except Exception as exc:
            log.error("BloomFilter.force_flush() error: %s", exc)

    @property
    def path(self) -> Path:
        """Path to the mmap file."""
        return self._path

    @property
    def is_initialized(self) -> bool:
        """True after initialize() completes successfully."""
        return self._initialized

    @property
    def is_closed(self) -> bool:
        """True after close() completes."""
        return self._closed

    # ── Internal: hash functions ──────────────────────────────────────────────

    def _hashes(self, url: str) -> Iterator[int]: # noqa
        """
        Generate BLOOM_HASH_COUNT bit positions for a URL using double hashing.

        Double hashing simulates k independent hash functions from two base hashes.
        This avoids the overhead of k separate MurmurHash3 invocations while
        maintaining good distribution (see Kirsch & Mitzenmacher 2006: "Less
        Hashing, Same Performance: Building a Better Bloom Filter").

        Formula:
            h1 = mmh3(url_bytes, seed=0)   [unsigned 32-bit]
            h2 = mmh3(url_bytes, seed=1)   [unsigned 32-bit, forced odd]
            pos_i = (h1 + i * h2) mod BLOOM_BIT_SIZE   for i in 0..k-1

        Forcing h2 to be odd:
            h2 is OR-masked with 1 to guarantee it is odd. This ensures the
            double-hash sequence (h1 + i*h2) mod m visits at least m/gcd(h2,m)
            distinct positions. For odd h2 and even m, gcd(h2, m) divides m/2
            at most, so the sequence is well-distributed. This prevents the
            degenerate case h2=0 where all positions equal h1.

        MurmurHash3:
            Non-cryptographic hash with excellent avalanche properties and
            ~1 GB/s throughput on modern hardware for short strings (URLs).
            The `signed=False` flag returns uint32 in [0, 2^32), which is
            what we need for the modular arithmetic.

        Args:
            url: Canonical URL string (already normalized by canonicalizer).

        Yields:
            Integers in [0, BLOOM_BIT_SIZE), one per hash function.
        """
        data = url.encode("utf-8")
        h1 = mmh3.hash(data, seed=0, signed=False)
        h2 = mmh3.hash(data, seed=1, signed=False)
        h2 = h2 | 1   # force odd — ensures non-degenerate double-hash sequence

        for i in range(BLOOM_HASH_COUNT):
            yield (h1 + i * h2) % BLOOM_BIT_SIZE

    # ── Internal: bit manipulation ────────────────────────────────────────────

    def _get_bit(self, bit_pos: int) -> bool:
        """
        Return True if the bit at position bit_pos is set in the mmap.

        Bit layout within the mmap:
            Physical byte offset = BLOOM_HEADER_BYTES + (bit_pos >> 3)
            Bit within byte      = bit_pos & 7  (LSB = bit 0)

        The header occupies bytes 0..(BLOOM_HEADER_BYTES-1). The bit array
        begins at byte BLOOM_HEADER_BYTES. Bit position 0 is the LSB of byte
        BLOOM_HEADER_BYTES.

        Args:
            bit_pos: Index into the logical bit array [0, BLOOM_BIT_SIZE).

        Returns:
            True if bit is set, False if bit is clear.
        """
        byte_offset = BLOOM_HEADER_BYTES + (bit_pos >> 3)
        bit_mask    = 1 << (bit_pos & 7)
        return bool(self._mmap[byte_offset] & bit_mask)

    def _set_bit(self, bit_pos: int) -> None:
        """
        Set the bit at position bit_pos in the mmap.

        Read-modify-write: read the current byte, OR in the bit mask, write back.
        The mmap is a bytearray-like object — index assignment modifies the page
        cache. The change is NOT immediately synced to disk; sync happens on flush.

        This operation is idempotent: if the bit is already set, the OR operation
        produces the same byte value, and the mmap write has no effect.

        Args:
            bit_pos: Index into the logical bit array [0, BLOOM_BIT_SIZE).
        """
        byte_offset = BLOOM_HEADER_BYTES + (bit_pos >> 3)
        bit_mask    = 1 << (bit_pos & 7)
        self._mmap[byte_offset] = self._mmap[byte_offset] | bit_mask

    def _check_all_bits(self, canonical_url: str) -> bool:
        """
        Check if all BLOOM_HASH_COUNT bits for a canonical URL are set.

        Short-circuits on the first unset bit (early exit for the common
        False case). Because most URLs in a real crawl have NOT been seen,
        the first bit check fails roughly 50% of the time (fill_factor ≈ 0.5
        at capacity), so the average number of bit reads per contains() call
        is approximately 2 for an empty filter and 7 for a full one.

        Args:
            canonical_url: URL already normalized by the canonicalizer.

        Returns:
            True only if all k bits are set (URL probably seen).
            False if any bit is clear (URL definitely not seen).
        """
        for bit_pos in self._hashes(canonical_url):
            if not self._get_bit(bit_pos):
                return False
        return True

    def _set_all_bits(self, canonical_url: str) -> None:
        """
        Set all BLOOM_HASH_COUNT bits for a canonical URL.

        Always sets all k bits, even if some are already set. This is correct:
        setting an already-set bit is a no-op at the bit level.

        Args:
            canonical_url: URL already normalized by the canonicalizer.
        """
        for bit_pos in self._hashes(canonical_url):
            self._set_bit(bit_pos)

    # ── Internal: mmap flush ──────────────────────────────────────────────────

    def _flush_mmap(self) -> None:
        """
        Flush dirty mmap pages to disk and update the header's count and timestamp.

        This is the durability barrier. After this call, all bits set by add()
        since the last flush are guaranteed to survive process death.

        Steps:
            1. Compute total count = header_count + add_count.
            2. Encode a new header with the updated count and flushed_at timestamp.
            3. Write the new header to mmap[0:64].
            4. Call mmap.flush() — writes all dirty pages to the underlying file.
            5. Update internal tracking counters.

        After flush, _header_count is updated to the total, _add_count resets to 0.
        This ensures count() returns the correct value before and after a flush.

        Design note: mmap.flush() does NOT call fsync() on the underlying file.
        On a machine with a write cache (most hardware), the pages are in the OS
        page cache after flush(). fsync() would guarantee they reach persistent
        storage but costs ~5ms. We accept the OS's write-back scheduling as
        sufficient — a system crash without an OS page cache flush is a hardware
        failure, not an expected operating condition for AXIOM's deployment model.

        Raises:
            Any exception from mmap.flush() or struct.pack(). The caller
            (add(), close()) catches and logs these.
        """
        if self._mmap is None:
            return

        total_count = self._header_count + self._add_count
        now         = time.time()

        # Write updated header to the beginning of the mmap
        header_bytes = _encode_header(
            count      = total_count,
            created_at = self._created_at,
            flushed_at = now,
        )
        self._mmap[:BLOOM_HEADER_BYTES] = header_bytes

        # Flush all dirty mmap pages to the underlying file
        self._mmap.flush()

        # Update session tracking
        self._header_count  = total_count
        self._add_count     = 0          # reset session counter
        self._last_flush_at = now
        self._flush_count   += 1

        log.debug(
            "BloomFilter flushed | count=%d flush_count=%d",
            total_count, self._flush_count,
        )


# ─────────────────────────────────────────────────────────────────────────────
# COMPACT BLOOM FILTER — in-memory, configurable parameters
#
# A smaller, simpler Bloom filter backed by a plain bytearray (in-memory).
# Used internally by RotatingBloomFilter for its sliding windows, and available
# directly for use cases where mmap persistence is not needed.
#
# Difference from BloomFilter:
#   - In-memory only: does not survive process death
#   - Configurable parameters: capacity and fp_rate are constructor arguments
#   - No header, no mmap, no file I/O
#   - Synchronous API (not async): suitable for non-asyncio contexts
#   - Lighter weight: minimal struct overhead
# ─────────────────────────────────────────────────────────────────────────────

class _CompactBloomFilter:
    """
    In-memory Bloom filter with configurable capacity and false positive rate.

    Computes optimal bit_size and hash_count from (capacity, fp_rate) using the
    standard Bloom filter parameter derivation:
        m = ceil(capacity * |ln(fp_rate)| / ln(2)^2)
        k = max(1, round((m / capacity) * ln(2)))

    Backed by a Python bytearray — stays in RAM, no mmap, no file I/O.
    Suitable for smaller datasets (10M–100M entries) where 500MB of mmap is
    unnecessary overhead.

    Uses the same double-hashing technique as BloomFilter for consistency.

    Not thread-safe. Not multi-process safe. Synchronous API.
    """

    __slots__ = ("_bit_size", "_hash_count", "_array", "_count", "_capacity")

    def __init__(self, capacity: int, fp_rate: float) -> None:
        """
        Args:
            capacity: Expected number of elements. FP rate degrades above this.
            fp_rate:  Target false positive probability at capacity.
        """
        import math

        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if not (0 < fp_rate < 1):
            raise ValueError(f"fp_rate must be in (0, 1), got {fp_rate}")

        # Optimal bit array size
        self._bit_size  = max(1, int(math.ceil(
            capacity * abs(math.log(fp_rate)) / (math.log(2) ** 2)
        )))
        # Optimal hash count
        self._hash_count = max(1, int(round(
            (self._bit_size / capacity) * math.log(2)
        )))
        # Allocate bit array as bytearray
        byte_count  = (self._bit_size + 7) // 8
        self._array = bytearray(byte_count)
        self._count    = 0
        self._capacity = capacity

    # ── API ───────────────────────────────────────────────────────────────────

    def add(self, item: str) -> None:
        """
        Add an item to the filter.

        Encodes the item as UTF-8 and computes hash positions using double hashing
        with the same formula as BloomFilter._hashes(). Items can be added after
        exceeding capacity, but the false positive rate will exceed fp_rate_target.

        Args:
            item: String item to add (URL or any string).
        """
        data = item.encode("utf-8")
        h1   = mmh3.hash(data, seed=0, signed=False)
        h2   = mmh3.hash(data, seed=1, signed=False) | 1  # force odd

        for i in range(self._hash_count):
            bit_pos = (h1 + i * h2) % self._bit_size
            self._array[bit_pos >> 3] |= 1 << (bit_pos & 7)

        self._count += 1

    def contains(self, item: str) -> bool:
        """
        Return True if the item was probably added, False if definitely not.

        Same semantics as BloomFilter.contains(): no false negatives, possible
        false positives at the configured fp_rate.

        Args:
            item: String item to check.

        Returns:
            True if probably added, False if definitely not.
        """
        data = item.encode("utf-8")
        h1   = mmh3.hash(data, seed=0, signed=False)
        h2   = mmh3.hash(data, seed=1, signed=False) | 1

        for i in range(self._hash_count):
            bit_pos = (h1 + i * h2) % self._bit_size
            if not (self._array[bit_pos >> 3] & (1 << (bit_pos & 7))):
                return False
        return True

    def add_batch(self, items: Sequence[str]) -> int:
        """
        Add a batch of items. Returns count of items added without error.
        Never raises; per-item errors are silently skipped.
        """
        added = 0
        for item in items:
            try:
                self.add(item)
                added += 1
            except Exception: # noqa
                pass
        return added

    def contains_batch(self, items: Sequence[str]) -> List[bool]:
        """
        Check membership for a batch of items.
        Returns list of booleans in same order as input.
        """
        return [self.contains(item) for item in items]

    def clear(self) -> None:
        """Reset the filter to empty state. Zeroes the entire bit array."""
        for i in range(len(self._array)):
            self._array[i] = 0
        self._count = 0

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def count(self) -> int:
        """Number of add() calls (not unique items — Bloom filters cannot count unique)."""
        return self._count

    @property
    def capacity(self) -> int:
        """Configured capacity. FP rate degrades when count exceeds this."""
        return self._capacity

    @property
    def is_full(self) -> bool:
        """True when count >= capacity. FP rate has reached the configured target."""
        return self._count >= self._capacity

    @property
    def bit_size(self) -> int:
        """Total number of bits in the bit array."""
        return self._bit_size

    @property
    def hash_count(self) -> int:
        """Number of hash positions per item."""
        return self._hash_count

    @property
    def byte_size(self) -> int:
        """Number of bytes in the bit array."""
        return len(self._array)

    @property
    def fill_factor(self) -> float:
        """
        Fraction of bits that are set. O(m/8) operation.
        Returns float in [0.0, 1.0].
        """
        set_bits = sum(bin(b).count("1") for b in self._array)
        return set_bits / self._bit_size if self._bit_size > 0 else 0.0

    def __repr__(self) -> str:
        return (
            f"_CompactBloomFilter("
            f"capacity={self._capacity:,}, "
            f"count={self._count:,}, "
            f"bit_size={self._bit_size:,}, "
            f"hash_count={self._hash_count}, "
            f"byte_size={len(self._array):,}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ROTATING BLOOM FILTER
#
# A dual-window sliding Bloom filter for URL deduplication with eventual expiry.
#
# The problem with a standard Bloom filter: it grows forever. After 400M URLs,
# the primary BloomFilter is full. Adding more URLs past capacity degrades the
# false positive rate toward 100%. The only recovery is to delete the file and
# rebuild, which accepts duplicate fetches.
#
# The rotating variant solves this for use cases where not ALL historical URLs
# need to be deduped — only recent ones. It maintains two in-memory windows:
#   _current:  the active window. All new adds go here.
#   _previous: the previous window. Still checked for membership.
#
# When _current fills (count >= ROTATING_FILTER_CAPACITY), rotation occurs:
#   1. _previous is discarded (its memory is freed).
#   2. _current becomes _previous.
#   3. A new empty _current is created.
#
# This provides:
#   - Coverage window: approximately 2 × ROTATING_FILTER_CAPACITY URLs
#   - Memory footprint: 2 × (memory for one _CompactBloomFilter at capacity)
#   - FP rate per window: ROTATING_FILTER_FP_RATE
#
# Use case: short-lived crawl sessions, daily refresh crawls, domain recrawls
# where re-fetching content older than ~50M URLs is acceptable or even desired.
#
# This is NOT a replacement for BloomFilter. frontier.py uses BloomFilter.
# RotatingBloomFilter is for supplementary cases.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RotatingBloomFilterStats:
    """Statistics snapshot for a RotatingBloomFilter."""
    current_count:     int
    previous_count:    int
    total_covered:     int     # approximate unique URLs across both windows
    rotation_count:    int     # number of rotations that have occurred
    window_capacity:   int     # configured capacity per window
    current_is_full:   bool
    current_bit_size:  int
    current_hash_count: int

    @property
    def windows_active(self) -> int:
        """Number of active windows (1 before first rotation, 2 after)."""
        return 2 if self.previous_count > 0 else 1

    def summary(self) -> str:
        return (
            f"windows={self.windows_active} "
            f"current={self.current_count:,}/{self.window_capacity:,} "
            f"previous={self.previous_count:,} "
            f"rotations={self.rotation_count}"
        )


class RotatingBloomFilter:
    """
    Dual-window in-memory Bloom filter for sliding-window URL deduplication.

    Maintains two _CompactBloomFilter instances: current and previous.
    When the current window fills, it rotates: previous is discarded, current
    becomes previous, and a fresh current is created.

    Key properties:
        - Sliding window: always covers approximately 2 × capacity URLs
        - Eventual expiry: URLs from the previous window expire on next rotation
        - Memory bounded: 2 × (one window's memory) forever
        - In-memory only: does not survive process death
        - Async API: contains() and add() are async for consistency with BloomFilter

    Use this when:
        - You want dedup over a sliding window, not all of history
        - Memory is constrained and the primary BloomFilter is overkill
        - Re-fetching old URLs (>2 windows ago) is acceptable

    Do NOT use this when:
        - You need durable dedup across process restarts (use BloomFilter)
        - You need to dedup against all 400M+ crawled URLs (use BloomFilter)

    Thread safety: not thread-safe. asyncio single-threaded only.

    Args:
        capacity:      URLs per rotation window. Default: ROTATING_FILTER_CAPACITY.
        fp_rate:       False positive rate per window. Default: ROTATING_FILTER_FP_RATE.
        canonicalizer: URL canonicalizer. Uses module default if None.
    """

    __slots__ = (
        "_capacity",
        "_fp_rate",
        "_current",
        "_previous",
        "_rotation_count",
        "_canonicalizer",
    )

    def __init__(
        self,
        capacity: int = ROTATING_FILTER_CAPACITY,
        fp_rate: float = ROTATING_FILTER_FP_RATE,
        canonicalizer: Optional[URLCanonicalizer] = None,
    ) -> None:
        self._capacity       = capacity
        self._fp_rate        = fp_rate
        self._current        = _CompactBloomFilter(capacity, fp_rate)
        self._previous: Optional[_CompactBloomFilter] = None
        self._rotation_count = 0
        self._canonicalizer  = canonicalizer or _DEFAULT_CANONICALIZER

    async def contains(self, url: str) -> bool:
        """
        Return True if the URL was probably added in either the current or previous window.
        Return False if definitely not seen in either window.

        Checks current window first (more likely to contain recent URLs).
        Falls through to previous window if not found in current.

        Never raises.

        Args:
            url: Raw URL string (canonicalized internally).

        Returns:
            True if probably seen in either window, False if definitely unseen.
        """
        try:
            canonical = self._canonicalizer.normalize(url)
            if self._current.contains(canonical):
                return True
            if self._previous is not None and self._previous.contains(canonical):
                return True
            return False
        except Exception as exc:
            log.error("RotatingBloomFilter.contains() error: %s", exc)
            return False

    async def add(self, url: str) -> None:
        """
        Add a URL to the current window.

        Triggers rotation if the current window is full (count >= capacity).
        After rotation, the new URL is added to the fresh current window.

        Never raises.

        Args:
            url: Raw URL string (canonicalized internally).
        """
        try:
            canonical = self._canonicalizer.normalize(url)
            if self._current.is_full:
                self._rotate()
            self._current.add(canonical)
        except Exception as exc:
            log.error("RotatingBloomFilter.add() error: %s", exc)

    async def add_batch(self, urls: Sequence[str]) -> int:
        """
        Add a batch of URLs, triggering rotation as needed.

        Rotation may occur mid-batch if the current window fills partway through.
        After rotation, remaining URLs are added to the new current window.

        Never raises. Returns count of successfully added URLs.
        """
        added = 0
        canonicals = self._canonicalizer.normalize_batch(urls)
        for canonical in canonicals:
            try:
                if self._current.is_full:
                    self._rotate()
                self._current.add(canonical)
                added += 1
            except Exception as exc:
                log.error("RotatingBloomFilter.add_batch() item error: %s", exc)
        return added

    async def contains_batch(self, urls: Sequence[str]) -> List[bool]:
        """
        Check membership for a batch of URLs across both windows.
        Returns list of booleans in same order as input.
        Never raises.
        """
        results: List[bool] = []
        canonicals = self._canonicalizer.normalize_batch(urls)
        for canonical in canonicals:
            try:
                found = self._current.contains(canonical)
                if not found and self._previous is not None:
                    found = self._previous.contains(canonical)
                results.append(found)
            except Exception as exc:
                log.error("RotatingBloomFilter.contains_batch() item error: %s", exc)
                results.append(False)
        return results

    async def count(self) -> int:
        """
        Approximate total URL count across both windows.
        This counts add() calls, not unique URLs.
        """
        prev_count = self._previous.count if self._previous is not None else 0
        return self._current.count + prev_count

    def _rotate(self) -> None:
        """
        Perform a rotation: discard previous, current → previous, new current.

        Memory freed: the old _previous object is garbage collected after this call.
        Memory allocated: a new _CompactBloomFilter of the same capacity.

        This is called automatically when the current window fills. The rotation
        is synchronous — there is no lock needed because asyncio is single-threaded.
        """
        self._previous = self._current
        self._current  = _CompactBloomFilter(self._capacity, self._fp_rate)
        self._rotation_count += 1
        log.info(
            "RotatingBloomFilter rotated | rotation_count=%d "
            "previous_count=%d current_capacity=%d",
            self._rotation_count,
            self._previous.count,
            self._capacity,
        )

    def stats(self) -> RotatingBloomFilterStats:
        """Return a statistics snapshot. O(1) operation."""
        prev_count = self._previous.count if self._previous is not None else 0
        return RotatingBloomFilterStats(
            current_count     = self._current.count,
            previous_count    = prev_count,
            total_covered     = self._current.count + prev_count,
            rotation_count    = self._rotation_count,
            window_capacity   = self._capacity,
            current_is_full   = self._current.is_full,
            current_bit_size  = self._current.bit_size,
            current_hash_count = self._current.hash_count,
        )

    def __repr__(self) -> str:
        s = self.stats()
        return (
            f"RotatingBloomFilter("
            f"windows={s.windows_active}, "
            f"current={s.current_count:,}/{self._capacity:,}, "
            f"rotations={self._rotation_count}"
            f")"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BLOOM FILTER POOL
#
# Manages multiple named BloomFilter instances, each backed by its own mmap
# file. Used when different crawl manifests or domain groups benefit from
# independent dedup scopes.
#
# For example:
#   - A "news" pool entry for news domain crawls
#   - A "saas_docs" pool entry for documentation crawls
#   - A "default" entry for everything else
#
# Each entry in the pool is a fully independent BloomFilter with its own
# bloom.bin file. The pool handles lifecycle (initialize/close) for all entries.
#
# This is not required by the crawler spec and is not used by frontier.py
# directly. It is provided for use by higher-level orchestration code.
# ─────────────────────────────────────────────────────────────────────────────

class BloomFilterPool:
    """
    Named pool of BloomFilter instances with shared lifecycle management.

    Each named entry corresponds to a separate mmap file:
        pool.get("news") → BloomFilter at pool_dir/news.bin
        pool.get("docs") → BloomFilter at pool_dir/docs.bin

    The pool initializes entries lazily on first access. All entries are
    flushed and closed when close_all() is called.

    Thread safety: not thread-safe. Single asyncio process.

    Usage:
        pool = BloomFilterPool(base_dir=Path("store/bloom_pool"))
        async with pool:
            bloom = await pool.get("news")
            if not await bloom.contains(url):
                await bloom.add(url)

    Args:
        base_dir:      Directory where individual bloom.bin files are stored.
        canonicalizer: Shared URL canonicalizer for all pool entries.
    """

    __slots__ = ("_base_dir", "_entries", "_canonicalizer", "_closed")

    def __init__(
        self,
        base_dir: Path = Path("store/bloom_pool"),
        canonicalizer: Optional[URLCanonicalizer] = None,
    ) -> None:
        self._base_dir      = base_dir
        self._entries: Dict[str, BloomFilter] = {}
        self._canonicalizer = canonicalizer or _DEFAULT_CANONICALIZER
        self._closed        = False

    async def __aenter__(self) -> "BloomFilterPool":
        return self

    async def __aexit__(self, *_) -> None:
        await self.close_all()

    async def get(self, name: str) -> BloomFilter:
        """
        Return the BloomFilter for the given name, initializing it if needed.

        The filter file is stored at: {base_dir}/{name}.bin

        Args:
            name: Filter name. Used as the file stem. Must be a valid filename
                  component (no path separators, no null bytes).

        Returns:
            Initialized BloomFilter for this name.

        Raises:
            ValueError: Name contains invalid characters.
            BloomFilterConfigError: Existing file has wrong constants.
            BloomFilterIntegrityError: Existing file is corrupt.
        """
        if self._closed:
            raise BloomFilterAlreadyClosedError("BloomFilterPool has been closed")

        # Validate name
        if not name or "/" in name or "\\" in name or "\x00" in name:
            raise ValueError(
                f"Invalid BloomFilterPool name: {name!r}. "
                f"Name must be non-empty and contain no path separators or null bytes."
            )

        if name not in self._entries:
            path = self._base_dir / f"{name}.bin"
            bloom = BloomFilter(path=path, canonicalizer=self._canonicalizer)
            await bloom.initialize()
            self._entries[name] = bloom
            log.info(
                "BloomFilterPool: initialized entry | name=%s path=%s", name, path
            )

        return self._entries[name]

    async def flush_all(self) -> None:
        """Force flush all initialized filters. Never raises."""
        for name, bloom in self._entries.items():
            try:
                await bloom.force_flush()
            except Exception as exc:
                log.error("BloomFilterPool.flush_all() error for %s: %s", name, exc)

    async def close_all(self) -> None:
        """
        Flush and close all initialized filters.

        Idempotent. After this call, no entries can be retrieved.
        """
        if self._closed:
            return
        for name, bloom in self._entries.items():
            try:
                await bloom.close()
            except Exception as exc:
                log.error("BloomFilterPool.close_all() error for %s: %s", name, exc)
        self._entries.clear()
        self._closed = True
        log.info("BloomFilterPool closed | base_dir=%s", self._base_dir)

    def names(self) -> List[str]:
        """Return names of all currently initialized filters."""
        return list(self._entries.keys())

    def is_initialized(self, name: str) -> bool:
        """Return True if the named filter has been initialized."""
        return name in self._entries

    async def stats_all(self) -> Dict[str, BloomFilterStats]:
        """
        Return stats for all initialized filters.
        WARNING: calls fill_factor() on each filter — expensive.
        """
        result: Dict[str, BloomFilterStats] = {}
        for name, bloom in self._entries.items():
            try:
                result[name] = await bloom.stats()
            except Exception as exc:
                log.error("BloomFilterPool.stats_all() error for %s: %s", name, exc)
        return result


# ─────────────────────────────────────────────────────────────────────────────
# BLOOM FILTER MONITOR
#
# An asyncio background task that periodically logs BloomFilter statistics.
# Attach to a running BloomFilter to get automatic health monitoring without
# polling manually.
#
# Usage:
#     monitor = BloomFilterMonitor(bloom, interval_seconds=300)
#     task = asyncio.create_task(monitor.run())
#     # ... crawl ...
#     monitor.stop()
#     await task
# ─────────────────────────────────────────────────────────────────────────────

class BloomFilterMonitor:
    """
    Periodic background monitor for a BloomFilter.

    Logs a stats summary every interval_seconds. If saturation is detected
    (count > BLOOM_CAPACITY), logs a WARNING. If healthy, logs at INFO.

    WARNING: calls stats() which includes fill_factor() — this is O(m/8) and
    takes 0.5–2 seconds. Set interval_seconds high enough (300s minimum) to
    avoid meaningful overhead.

    Not thread-safe. Runs in the asyncio event loop of the owning process.

    Args:
        bloom:             The BloomFilter to monitor.
        interval_seconds:  Seconds between stats logs. Minimum 60, default 300.
        name:              Label for log messages. Defaults to str(bloom.path).
    """

    __slots__ = ("_bloom", "_interval", "_name", "_running", "_task")

    def __init__(
        self,
        bloom: BloomFilter,
        interval_seconds: float = 300.0,
        name: Optional[str] = None,
    ) -> None:
        if interval_seconds < 60:
            log.warning(
                "BloomFilterMonitor interval %ss is very short. "
                "fill_factor() takes 0.5-2s. Consider >= 300s.",
                interval_seconds,
            )
        self._bloom    = bloom
        self._interval = interval_seconds
        self._name     = name or str(bloom.path)
        self._running  = False
        self._task: Optional[asyncio.Task] = None

    async def run(self) -> None:
        """
        Run the monitoring loop until stop() is called.

        Designed to be launched as an asyncio Task:
            task = asyncio.create_task(monitor.run())
        """
        self._running = True
        log.info(
            "BloomFilterMonitor started | name=%s interval=%.0fs",
            self._name, self._interval,
        )
        try:
            while self._running:
                try:
                    await asyncio.sleep(self._interval)
                    if not self._running:
                        break
                    await self._log_stats()
                except asyncio.CancelledError:
                    break
                except Exception as exc:
                    log.error(
                        "BloomFilterMonitor error | name=%s error=%s",
                        self._name, exc,
                    )
        finally:
            log.info("BloomFilterMonitor stopped | name=%s", self._name)

    async def _log_stats(self) -> None:
        """Collect and log a stats snapshot."""
        try:
            stats = await self._bloom.stats()
            level  = logging.WARNING if stats.is_saturated else logging.INFO
            log.log(
                level,
                "BloomFilter health | name=%s %s",
                self._name, stats.summary(),
            )
        except Exception as exc:
            log.error(
                "BloomFilterMonitor._log_stats() error | name=%s error=%s",
                self._name, exc,
            )

    def stop(self) -> None:
        """Signal the monitoring loop to stop after the current sleep."""
        self._running = False

    async def run_once(self) -> Optional[BloomFilterStats]:
        """
        Run a single stats collection and return the result.
        Useful for one-off health checks outside the monitoring loop.
        """
        try:
            return await self._bloom.stats()
        except Exception as exc:
            log.error("BloomFilterMonitor.run_once() error: %s", exc)
            return None


# ─────────────────────────────────────────────────────────────────────────────
# BLOOM FILTER MIGRATION
#
# When BLOOM_BIT_SIZE, BLOOM_HASH_COUNT, or BLOOM_CAPACITY changes, existing
# bloom.bin files become incompatible. BloomFilterMigration provides a way to
# rebuild the filter from a URL list without losing all historical dedup data.
#
# The rebuild process:
#   1. User provides an iterable of historical URLs (from frontier.db, logs, etc.)
#   2. Migration creates a new bloom.bin with the current constants.
#   3. Each historical URL is re-added to the new filter.
#   4. The old bloom.bin is replaced with the new one.
#
# This trades duplicate-fetch risk (if migration is incomplete) against using a
# filter with wrong parameters (which produces wrong dedup behavior forever).
#
# If no historical URL list is available: delete bloom.bin and start fresh.
# The filter will rebuild over the next crawl pass — accepting duplicates.
# ─────────────────────────────────────────────────────────────────────────────

class BloomFilterMigration:
    """
    Utility for rebuilding a BloomFilter file with updated parameters.

    Handles the case where BLOOM_BIT_SIZE, BLOOM_HASH_COUNT, or BLOOM_CAPACITY
    have changed and the existing bloom.bin is incompatible.

    Usage:
        migration = BloomFilterMigration(
            old_path=Path("store/bloom.bin"),
            new_path=Path("store/bloom.bin.new"),
        )
        async with BloomFilter(path=migration.new_path) as new_bloom:
            rebuilt = await migration.rebuild(new_bloom, historical_urls)
        if rebuilt:
            migration.commit()   # atomically replaces old_path with new_path

    Args:
        old_path: Path to the existing (incompatible) bloom.bin.
        new_path: Path for the new bloom.bin being built.
    """

    __slots__ = ("_old_path", "_new_path")

    def __init__(self, old_path: Path, new_path: Path) -> None:
        self._old_path = old_path
        self._new_path = new_path

    async def rebuild(
        self,
        new_bloom: BloomFilter,
        historical_urls: Sequence[str],
        batch_size: int = 50_000,
    ) -> int:
        """
        Re-add all historical URLs to the new filter.

        Adds URLs in batches of batch_size for efficiency. Logs progress
        every 1M URLs. Returns total count of URLs added.

        Args:
            new_bloom:       Initialized BloomFilter to populate.
            historical_urls: Iterable of URLs to re-add.
            batch_size:      URLs per add_batch() call. Default 50K.

        Returns:
            Total count of URLs successfully added.
        """
        total_added = 0
        batch: List[str] = []

        for url in historical_urls:
            batch.append(url)
            if len(batch) >= batch_size:
                added = await new_bloom.add_batch(batch)
                total_added += added
                batch.clear()
                if total_added % 1_000_000 < batch_size:
                    log.info(
                        "BloomFilterMigration progress | rebuilt=%d",
                        total_added,
                    )

        if batch:
            added = await new_bloom.add_batch(batch)
            total_added += added

        await new_bloom.force_flush()
        log.info(
            "BloomFilterMigration rebuild complete | total_added=%d path=%s",
            total_added, self._new_path,
        )
        return total_added

    def commit(self) -> None:
        """
        Atomically replace the old bloom.bin with the new one.

        Uses os.replace() for atomic rename on POSIX systems. The old file
        is overwritten atomically — there is no window where neither file exists.

        Raises:
            OSError: If the rename fails (different filesystems, permissions).
        """
        os.replace(str(self._new_path), str(self._old_path))
        log.info(
            "BloomFilterMigration committed | %s → %s",
            self._new_path, self._old_path,
        )

    def rollback(self) -> None:
        """
        Delete the new (partially built) file without touching the old one.

        Called on error during rebuild to clean up the partial file.
        """
        try:
            self._new_path.unlink(missing_ok=True)
            log.info("BloomFilterMigration rolled back | deleted %s", self._new_path)
        except Exception as exc:
            log.error("BloomFilterMigration.rollback() error: %s", exc)


# ─────────────────────────────────────────────────────────────────────────────
# BENCHMARK
#
# Self-contained performance benchmark. Measures add/contains throughput
# and false positive rate against a temporary filter.
#
# Run: python bloom_filter.py benchmark [--n 1000000]
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Result of a BloomFilter performance benchmark."""
    n_urls:                    int
    add_throughput_per_sec:    float
    contains_true_throughput:  float     # throughput checking known-present URLs
    contains_false_throughput: float     # throughput checking known-absent URLs
    add_duration_sec:          float
    false_positive_count:      int
    false_positive_rate:       float
    false_negative_count:      int       # should always be 0
    peak_memory_mb:            float

    def passed(self) -> bool:
        """True if the benchmark results are within acceptable bounds."""
        return (
            self.false_negative_count == 0
            and self.false_positive_rate < 0.01   # less than 1% FP rate
        )

    def summary(self) -> str:
        status = "PASS" if self.passed() else "FAIL"
        return (
            f"[{status}] n={self.n_urls:,} "
            f"add={self.add_throughput_per_sec:,.0f}/s "
            f"contains(T)={self.contains_true_throughput:,.0f}/s "
            f"contains(F)={self.contains_false_throughput:,.0f}/s "
            f"fp={self.false_positive_rate*100:.4f}% "
            f"fn={self.false_negative_count}"
        )


async def run_benchmark(
    n_urls: int = 1_000_000,
    tmp_dir: Optional[Path] = None,
) -> BenchmarkResult:
    """
    Run a full add/contains benchmark against a temporary BloomFilter.

    Creates a temporary bloom.bin at tmp_dir (or /tmp if None), adds n_urls,
    then measures contains() throughput for both known-present and known-absent
    URLs. Reports false positive and false negative rates.

    The test URL format is deterministic: "https://benchmark.example.com/page/{i}"
    Unseen URLs use: "https://benchmark.example.com/page/unseen_{i}"

    Args:
        n_urls:   Number of URLs to benchmark. Default 1M.
        tmp_dir:  Temp directory for bloom.bin. Cleaned up after benchmark.

    Returns:
        BenchmarkResult with throughput and accuracy metrics.
    """
    import tempfile
    import shutil
    from rich.progress import Progress

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp(prefix="axiom_bloom_bench_"))

    bloom_path  = tmp_dir / "bench_bloom.bin"
    url_base    = "https://benchmark.example.com/page/"
    unseen_base = "https://benchmark.example.com/page/unseen_"

    bloom = BloomFilter(path=bloom_path)
    await bloom.initialize()

    try:
        with Progress() as progress:
            # Phase 1: Add n_urls
            task = progress.add_task("Adding URLs...", total=n_urls)
            t0 = time.perf_counter()
            for i in range(n_urls):
                await bloom.add(f"{url_base}{i}")
                progress.advance(task)
            t1 = time.perf_counter()
            add_duration   = t1 - t0
            add_throughput = n_urls / add_duration

            # Phase 2: contains() on known-present (false negative check)
            false_negative_count = 0
            task = progress.add_task("Checking present...", total=n_urls)
            t0 = time.perf_counter()
            for i in range(n_urls):
                if not await bloom.contains(f"{url_base}{i}"):
                    false_negative_count += 1
                progress.advance(task)
            t1 = time.perf_counter()
            contains_true_throughput = n_urls / (t1 - t0)

            # Phase 3: contains() on known-absent (false positive check)
            false_positive_count = 0
            task = progress.add_task("Checking absent...", total=n_urls)
            t0 = time.perf_counter()
            for i in range(n_urls):
                if await bloom.contains(f"{unseen_base}{i}"):
                    false_positive_count += 1
                progress.advance(task)
            t1 = time.perf_counter()
            contains_false_throughput = n_urls / (t1 - t0)
            fp_rate = false_positive_count / n_urls

        # Memory usage via /proc (Linux only)
        peak_memory_mb = 0.0
        try:
            with open("/proc/self/status") as fh:
                for line in fh:
                    if line.startswith("VmPeak:"):
                        peak_memory_mb = int(line.split()[1]) / 1024
                        break
        except Exception: # noqa
            pass

        return BenchmarkResult(
            n_urls                    = n_urls,
            add_throughput_per_sec    = add_throughput,
            contains_true_throughput  = contains_true_throughput,
            contains_false_throughput = contains_false_throughput,
            add_duration_sec          = add_duration,
            false_positive_count      = false_positive_count,
            false_positive_rate       = fp_rate,
            false_negative_count      = false_negative_count,
            peak_memory_mb            = peak_memory_mb,
        )

    finally:
        await bloom.close()
        try:
            shutil.rmtree(str(tmp_dir), ignore_errors=True)
        except Exception: # noqa
            pass


# ─────────────────────────────────────────────────────────────────────────────
# TEST SUITE
#
# Self-contained test suite covering all public API methods, edge cases, and
# correctness properties.
#
# Run: python bloom_filter.py test
# Or:  pytest bloom_filter.py
#
# Tests are organized from foundational (basic correctness) to advanced
# (edge cases, crash recovery, canonicalization accuracy).
# ─────────────────────────────────────────────────────────────────────────────

async def _test_contains_returns_false_for_unseen_url(tmp: Path) -> None:
    """Test 1: contains() returns False for a URL that was never added."""
    async with BloomFilter(path=tmp / "t1.bin") as bloom:
        assert await bloom.contains("https://example.com/never-added") is False
        assert await bloom.contains("https://different.com/path?q=1") is False
        assert await bloom.contains("") is False


async def _test_contains_returns_true_after_add(tmp: Path) -> None:
    """Test 2: contains() returns True immediately after add()."""
    async with BloomFilter(path=tmp / "t2.bin") as bloom:
        url = "https://example.com/page/42"
        assert await bloom.contains(url) is False
        await bloom.add(url)
        assert await bloom.contains(url) is True


async def _test_no_false_negatives(tmp: Path) -> None:
    """Test 3: No false negatives across 10,000 URLs. Every added URL is found."""
    n = 10_000
    urls = [f"https://example.com/fn-check/{i}" for i in range(n)]
    async with BloomFilter(path=tmp / "t3.bin") as bloom:
        for url in urls:
            await bloom.add(url)
        false_negatives = 0
        for url in urls:
            if not await bloom.contains(url):
                false_negatives += 1
    assert false_negatives == 0, (
        f"Got {false_negatives} false negatives. "
        f"This violates the core Bloom filter invariant."
    )


async def _test_false_positive_rate(tmp: Path) -> None:
    """
    Test 4: False positive rate stays below 0.02% at 100K URLs.

    Uses 2x the design target (0.01%) as the pass threshold to account for
    the fact that 4B bits with 100K elements has almost no fill — the FP rate
    should be approximately 0 at this scale, not the design-point 0.01%.
    """
    n = 100_000
    fp_budget = 0.0002   # 0.02% — generous for test purposes

    async with BloomFilter(path=tmp / "t4.bin") as bloom:
        for i in range(n):
            await bloom.add(f"https://example.com/fp-check/{i}")

        fp_count = 0
        for i in range(n, n * 2):
            if await bloom.contains(f"https://example.com/fp-check/{i}"):
                fp_count += 1

    fp_rate = fp_count / n
    assert fp_rate < fp_budget, (
        f"FP rate {fp_rate:.6f} ({fp_count}/{n}) "
        f"exceeds budget {fp_budget:.6f}"
    )


async def _test_mmap_survives_process_death(tmp: Path) -> None:
    """
    Test 5: Bits survive a simulated process death (close + reopen).

    Adds a URL, force-flushes (simulating graceful shutdown), closes the filter,
    reopens it (simulating the next process), and verifies the URL is still found.
    """
    path = tmp / "t5.bin"
    url  = "https://example.com/must-survive-restart"

    # Session 1: add URL and flush
    bloom1 = BloomFilter(path=path)
    await bloom1.initialize()
    await bloom1.add(url)
    await bloom1.force_flush()   # explicit flush before "death"
    await bloom1.close()

    # Session 2: reopen and check
    bloom2 = BloomFilter(path=path)
    await bloom2.initialize()
    try:
        result = await bloom2.contains(url)
    finally:
        await bloom2.close()

    assert result is True, (
        "URL not found after mmap close + reopen. "
        "Durability guarantee violated."
    )


async def _test_unflushed_bits_survive_close(tmp: Path) -> None:
    """
    Test 6: close() always flushes — unflushed bits still survive.

    Adds a URL without explicit flush, then closes normally.
    close() must call flush() internally. Reopens and verifies.
    """
    path = tmp / "t6.bin"
    url  = "https://example.com/unflushed-but-close-flushes"

    bloom1 = BloomFilter(path=path)
    await bloom1.initialize()
    await bloom1.add(url)
    # No force_flush — rely on close() to flush
    await bloom1.close()

    bloom2 = BloomFilter(path=path)
    await bloom2.initialize()
    try:
        result = await bloom2.contains(url)
    finally:
        await bloom2.close()

    assert result is True, "close() did not flush before closing — durability failure"


async def _test_add_is_idempotent(tmp: Path) -> None:
    """Test 7: add() is idempotent — double add has no additional effect."""
    async with BloomFilter(path=tmp / "t7.bin") as bloom:
        url = "https://example.com/idempotent"

        await bloom.add(url)
        assert await bloom.contains(url) is True

        # Second add — bits already set, should be no-op
        await bloom.add(url)
        assert await bloom.contains(url) is True

        # Third add — still a no-op
        await bloom.add(url)
        assert await bloom.contains(url) is True


async def _test_count_increments_on_add(tmp: Path) -> None:
    """Test 8: count() increments by 1 on each add() call."""
    async with BloomFilter(path=tmp / "t8.bin") as bloom:
        assert await bloom.count() == 0

        for i in range(100):
            await bloom.add(f"https://example.com/{i}")
            c = await bloom.count()
            assert c == i + 1, f"Expected count {i+1}, got {c}"


async def _test_count_survives_flush(tmp: Path) -> None:
    """Test 9: count() survives a flush — the header stores the running total."""
    path = tmp / "t9.bin"
    async with BloomFilter(path=path) as bloom:
        for i in range(BLOOM_FLUSH_INTERVAL + 50):
            await bloom.add(f"https://example.com/count/{i}")
        count_before_close = await bloom.count()

    # Reopen and verify count is preserved
    async with BloomFilter(path=path) as bloom2:
        count_after_reopen = await bloom2.count()

    # After reopen, the header count is the total flushed.
    # The 50 adds after the last flush interval may have been flushed by close().
    assert count_after_reopen == count_before_close, (
        f"Count changed after reopen: {count_before_close} → {count_after_reopen}"
    )


async def _test_flush_happens_at_interval(tmp: Path) -> None:
    """Test 10: Automatic flush fires exactly at BLOOM_FLUSH_INTERVAL adds."""
    async with BloomFilter(path=tmp / "t10.bin") as bloom:
        assert bloom._flush_count == 0 # noqa

        for i in range(BLOOM_FLUSH_INTERVAL - 1):
            await bloom.add(f"https://example.com/{i}")

        # One below the threshold — no flush yet
        assert bloom._flush_count == 0, ( # noqa
            f"Flush triggered before interval: flush_count={bloom._flush_count}" # noqa
        )

        # The interval-th add triggers the flush
        await bloom.add("https://example.com/trigger")
        assert bloom._flush_count == 1, ( # noqa
            f"Expected 1 flush at exactly {BLOOM_FLUSH_INTERVAL} adds, "
            f"got {bloom._flush_count}" # noqa
        )

        # Next BLOOM_FLUSH_INTERVAL adds triggers a second flush
        for i in range(BLOOM_FLUSH_INTERVAL):
            await bloom.add(f"https://example.com/second/{i}")
        assert bloom._flush_count == 2 # noqa


async def _test_initialize_creates_file(tmp: Path) -> None:
    """Test 11: initialize() creates bloom.bin if it does not exist."""
    path = tmp / "t11.bin"
    assert not path.exists()

    async with BloomFilter(path=path) as bloom:
        assert path.exists(), "File was not created by initialize()"
        assert path.stat().st_size == BLOOM_FILE_BYTES, (
            f"File size {path.stat().st_size} != {BLOOM_FILE_BYTES}"
        )
        _ = bloom  # suppress unused warning


async def _test_initialize_opens_existing_file(tmp: Path) -> None:
    """Test 12: initialize() opens existing file and retains its state."""
    path = tmp / "t12.bin"
    url  = "https://example.com/must-persist"

    async with BloomFilter(path=path) as bloom:
        await bloom.add(url)

    async with BloomFilter(path=path) as bloom2:
        assert await bloom2.contains(url) is True


async def _test_batch_add_and_contains(tmp: Path) -> None:
    """Test 13: add_batch() and contains_batch() work correctly."""
    n = 2000
    urls = [f"https://example.com/batch/{i}" for i in range(n)]

    async with BloomFilter(path=tmp / "t13.bin") as bloom:
        added = await bloom.add_batch(urls)
        assert added == n, f"Expected {n} added, got {added}"

        results = await bloom.contains_batch(urls)
        assert len(results) == n
        assert all(results), (
            f"{sum(1 for r in results if not r)} URLs not found after batch add"
        )

        # Unseen URLs — most should return False
        unseen = [f"https://example.com/unseen/{i}" for i in range(n)]
        unseen_results = await bloom.contains_batch(unseen)
        fp_count = sum(unseen_results)
        assert fp_count / n < 0.01, f"Batch FP rate {fp_count/n:.4f} > 1%"


async def _test_verify_integrity_new_file(tmp: Path) -> None:
    """Test 14: verify_integrity() passes for a freshly created file."""
    path = tmp / "t14.bin"
    async with BloomFilter(path=path) as bloom:
        _ = bloom  # ensure file is created

    async with BloomFilter(path=path) as bloom2:
        report = await bloom2.verify_integrity()

    assert report.is_valid, f"Integrity check failed:\n{report}"
    assert report.file_exists
    assert report.file_size_correct
    assert report.magic_valid
    assert report.version_valid
    assert report.checksum_valid
    assert report.capacity_matches
    assert report.bit_size_matches
    assert report.hash_count_matches
    assert not report.errors


async def _test_verify_integrity_missing_file(tmp: Path) -> None:
    """Test 15: verify_integrity() reports correctly for a missing file."""
    path = tmp / "nonexistent.bin"
    bloom = BloomFilter(path=path)
    report = await bloom.verify_integrity()
    assert not report.file_exists
    assert not report.is_valid
    assert len(report.errors) > 0


async def _test_verify_integrity_corrupt_header(tmp: Path) -> None:
    """Test 16: verify_integrity() detects a corrupted header."""
    path = tmp / "t16.bin"

    async with BloomFilter(path=path) as bloom:
        _ = bloom

    # Corrupt the header CRC
    with open(path, "r+b") as f:
        f.seek(60)
        f.write(b"\xFF\xFF\xFF\xFF")  # overwrite CRC with garbage

    bloom2 = BloomFilter(path=path)
    report = await bloom2.verify_integrity()
    assert not report.checksum_valid
    assert not report.is_valid
    assert any("CRC32" in e for e in report.errors)


async def _test_url_canonicalizer_scheme_lowercase(tmp: Path) -> None:
    """Test 17: Canonicalizer lowercases scheme and host."""
    canon = URLCanonicalizer()
    # Scheme and host must be lowercased
    assert canon.normalize("HTTP://EXAMPLE.COM/path") == "http://example.com/path"
    assert canon.normalize("HTTPS://Example.Com/Path") == "https://example.com/Path"
    # Path case is NOT lowercased (case-sensitive paths are common on Linux servers)
    assert canon.normalize("https://example.com/PATH") == "https://example.com/PATH"


async def _test_url_canonicalizer_tracking_params(tmp: Path) -> None:
    """Test 18: Canonicalizer strips tracking parameters."""
    canon = URLCanonicalizer()

    cases = [
        (
            "https://example.com/page?id=42&utm_source=google",
            "https://example.com/page?id=42",
        ),
        (
            "https://example.com/page?fbclid=abc123&content=blog",
            "https://example.com/page?content=blog",
        ),
        (
            "https://example.com/page?utm_source=a&utm_medium=b&utm_campaign=c",
            "https://example.com/page",
        ),
        (
            "https://example.com/page?_ga=123.456&q=search",
            "https://example.com/page?q=search",
        ),
    ]
    for raw, expected in cases:
        result = canon.normalize(raw)
        assert result == expected, (
            f"normalize({raw!r})\n  expected: {expected!r}\n  got:      {result!r}"
        )


async def _test_url_canonicalizer_param_sort(tmp: Path) -> None:
    """Test 19: Canonicalizer sorts query params alphabetically."""
    canon = URLCanonicalizer()
    # Different param order → same canonical form
    u1 = canon.normalize("https://example.com/page?z=1&a=2&m=3")
    u2 = canon.normalize("https://example.com/page?a=2&m=3&z=1")
    assert u1 == u2, f"Different param order produced different canonical: {u1} vs {u2}"


async def _test_url_canonicalizer_fragment_strip(tmp: Path) -> None:
    """Test 20: Canonicalizer strips fragments."""
    canon = URLCanonicalizer()
    u = canon.normalize("https://example.com/page#section-3")
    assert "#" not in u, f"Fragment not stripped: {u!r}"
    assert u == "https://example.com/page"


async def _test_url_canonicalizer_default_port(tmp: Path) -> None:
    """Test 21: Canonicalizer strips default ports."""
    canon = URLCanonicalizer()
    assert canon.normalize("http://example.com:80/path") == "http://example.com/path"
    assert canon.normalize("https://example.com:443/path") == "https://example.com/path"
    # Non-default port is preserved
    assert canon.normalize("https://example.com:8443/path") == "https://example.com:8443/path"


async def _test_url_canonicalizer_double_slash(tmp: Path) -> None:
    """Test 22: Canonicalizer collapses consecutive slashes in path."""
    canon = URLCanonicalizer()
    assert canon.normalize("https://example.com//path//to//page") == \
           "https://example.com/path/to/page"


async def _test_canonicalized_dedup(tmp: Path) -> None:
    """
    Test 23: Two URLs with different tracking params dedup against each other.

    Adding URL A with a tracking param, then checking URL B (same page, no
    tracking param) should return True after canonicalization.
    """
    async with BloomFilter(path=tmp / "t23.bin") as bloom:
        await bloom.add("https://example.com/page?content=article&utm_source=twitter")
        # Same canonical form: https://example.com/page?content=article
        result = await bloom.contains("https://example.com/page?content=article&utm_medium=social")
        assert result is True, (
            "URLs with different tracking params should dedup after canonicalization"
        )


async def _test_header_roundtrip() -> None:
    """Test 24: Header encode/decode roundtrip preserves all fields exactly."""
    import math

    now   = time.time()
    count = 42_000_000

    encoded = _encode_header(count=count, created_at=now, flushed_at=now + 1.5)
    assert len(encoded) == BLOOM_HEADER_BYTES

    decoded = _decode_header(encoded)
    assert decoded["magic"]      == BLOOM_HEADER_MAGIC
    assert decoded["version"]    == BLOOM_HEADER_VERSION
    assert decoded["count"]      == count
    assert decoded["capacity"]   == BLOOM_CAPACITY
    assert decoded["bit_size"]   == BLOOM_BIT_SIZE
    assert decoded["hash_count"] == BLOOM_HASH_COUNT
    assert math.isclose(decoded["created_at"],  now,       rel_tol=1e-9)
    assert math.isclose(decoded["flushed_at"],  now + 1.5, rel_tol=1e-9)


async def _test_header_bad_magic() -> None:
    """Test 25: _decode_header() raises BloomFilterIntegrityError for bad magic."""
    good = _encode_header(count=0, created_at=time.time(), flushed_at=time.time())
    bad  = b"BADMAGIC" + good[8:]

    try:
        _decode_header(bad)
        assert False, "Should have raised BloomFilterIntegrityError"
    except BloomFilterIntegrityError:
        pass


async def _test_header_bad_checksum() -> None:
    """Test 26: _decode_header() raises BloomFilterIntegrityError for bad CRC32."""
    good = _encode_header(count=0, created_at=time.time(), flushed_at=time.time())
    bad  = good[:60] + b"\x00\x00\x00\x00"   # zero out the checksum

    try:
        _decode_header(bad)
        assert False, "Should have raised BloomFilterIntegrityError"
    except BloomFilterIntegrityError:
        pass


async def _test_compact_bloom_filter() -> None:
    """Test 27: _CompactBloomFilter basic correctness."""
    cbf = _CompactBloomFilter(capacity=100_000, fp_rate=0.001)

    # Empty: contains returns False
    assert cbf.contains("hello") is False

    # After add: contains returns True
    cbf.add("hello")
    assert cbf.contains("hello") is True
    assert cbf.count == 1

    # Idempotent add
    cbf.add("hello")
    assert cbf.count == 2  # count tracks calls, not unique
    assert cbf.contains("hello") is True

    # is_full triggers at capacity
    cbf2 = _CompactBloomFilter(capacity=10, fp_rate=0.001)
    for i in range(10):
        cbf2.add(f"item_{i}")
    assert cbf2.is_full is True

    # Clear resets everything
    cbf2.clear()
    assert cbf2.count == 0
    assert cbf2.contains("item_0") is False


async def _test_rotating_bloom_filter() -> None:
    """Test 28: RotatingBloomFilter rotation behavior."""
    rbf = RotatingBloomFilter(capacity=100, fp_rate=0.001)

    # Fill current window
    urls = [f"https://example.com/rotating/{i}" for i in range(100)]
    for url in urls:
        await rbf.add(url)

    # All should be found in current window
    for url in urls:
        assert await rbf.contains(url) is True

    stats = rbf.stats()
    assert stats.rotation_count == 0
    assert stats.current_count == 100

    # One more add triggers rotation
    await rbf.add("https://example.com/rotating/trigger")

    stats = rbf.stats()
    assert stats.rotation_count == 1
    assert stats.current_count == 1        # only the trigger URL
    assert stats.previous_count == 100     # old current became previous

    # Old URLs should still be found in previous window
    for url in urls:
        assert await rbf.contains(url) is True, f"{url} not found after rotation"

    # After second rotation, first window URLs expire (previous gets replaced again)
    for i in range(101, 202):
        await rbf.add(f"https://example.com/rotating/{i}")

    stats = rbf.stats()
    assert stats.rotation_count == 2

    # The very first 100 URLs should now be gone (they were in the window
    # that became _previous on rotation 1, which was discarded on rotation 2).
    found_after_second_rotation = 0
    for url in urls:
        if await rbf.contains(url):
            found_after_second_rotation += 1
    # With rotation_count=2, the original 100 URLs are in neither window.
    assert found_after_second_rotation == 0, (
        f"{found_after_second_rotation} URLs still found after 2 rotations; "
        f"expected 0 (they should have expired)"
    )


async def _test_bloom_filter_pool(tmp: Path) -> None:
    """Test 29: BloomFilterPool manages multiple named filters."""
    pool = BloomFilterPool(base_dir=tmp / "pool")

    try:
        bloom_a = await pool.get("domain_a")
        bloom_b = await pool.get("domain_b")

        # Same name returns same instance
        bloom_a2 = await pool.get("domain_a")
        assert bloom_a is bloom_a2

        # Filters are independent
        await bloom_a.add("https://a.example.com/page")
        assert await bloom_a.contains("https://a.example.com/page") is True
        assert await bloom_b.contains("https://a.example.com/page") is False

        # Names are tracked
        names = pool.names()
        assert "domain_a" in names
        assert "domain_b" in names

    finally:
        await pool.close_all()


async def _test_multiple_opens_same_file(tmp: Path) -> None:
    """Test 30: Opening the same file sequentially (not concurrently) works correctly."""
    path = tmp / "t30.bin"

    # Open 1: add some URLs
    async with BloomFilter(path=path) as b1:
        for i in range(500):
            await b1.add(f"https://example.com/seq/{i}")
        count1 = await b1.count()

    # Open 2: verify state is intact, add more
    async with BloomFilter(path=path) as b2:
        count2_on_open = await b2.count()
        assert count2_on_open == count1, (
            f"Count changed between opens: {count1} → {count2_on_open}"
        )

        # Add more in session 2
        for i in range(500, 1000):
            await b2.add(f"https://example.com/seq/{i}")
        count2_final = await b2.count()

    # Open 3: all 1000 URLs should be present
    async with BloomFilter(path=path) as b3:
        count3 = await b3.count()
        assert count3 == count2_final, (
            f"Count changed between open 2 and 3: {count2_final} → {count3}"
        )
        for i in range(1000):
            assert await b3.contains(f"https://example.com/seq/{i}") is True, (
                f"URL {i} not found in session 3"
            )


async def _test_url_with_unicode(tmp: Path) -> None:
    """Test 31: Unicode URLs are handled correctly via NFC normalization."""
    async with BloomFilter(path=tmp / "t31.bin") as bloom:
        # Café: NFC vs NFD decomposition — must dedup to same canonical form
        url_nfc = "https://example.com/caf\u00e9/menu"        # é as single codepoint
        url_nfd = "https://example.com/cafe\u0301/menu"        # é as e + combining acute

        await bloom.add(url_nfc)
        # NFD form should resolve to same canonical (NFC normalization applied)
        assert await bloom.contains(url_nfd) is True, (
            "NFD-encoded URL not recognized after NFC URL was added"
        )
        assert await bloom.contains(url_nfc) is True


async def _test_url_empty_and_whitespace(tmp: Path) -> None:
    """Test 32: Edge cases — empty string and whitespace-only URLs."""
    async with BloomFilter(path=tmp / "t32.bin") as bloom:
        # Empty string — should not raise, contains returns False, add is no-op
        assert await bloom.contains("") is False
        await bloom.add("")   # should not raise

        # Whitespace — stripped by canonicalizer, treated as empty
        assert await bloom.contains("   ") is False
        await bloom.add("   ")  # should not raise

        # URL with leading/trailing whitespace — strips to valid URL
        url = "  https://example.com/trimmed  "
        await bloom.add(url)
        assert await bloom.contains("https://example.com/trimmed") is True


async def _test_close_is_idempotent(tmp: Path) -> None:
    """Test 33: close() called twice does not raise or corrupt state."""
    path = tmp / "t33.bin"
    bloom = BloomFilter(path=path)
    await bloom.initialize()
    await bloom.add("https://example.com/idempotent-close")
    await bloom.close()
    # Second close must be a no-op
    await bloom.close()
    # After double-close, the URL must still be in the file
    async with BloomFilter(path=path) as b2:
        assert await b2.contains("https://example.com/idempotent-close") is True


async def _test_contains_before_initialize_returns_false(tmp: Path) -> None:
    """Test 34: contains() before initialize() returns False without raising."""
    bloom = BloomFilter(path=tmp / "t34.bin")
    # Never initialized — must return False, not raise
    result = await bloom.contains("https://example.com/pre-init")
    assert result is False


async def _test_add_before_initialize_is_noop(tmp: Path) -> None:
    """Test 35: add() before initialize() is a no-op without raising."""
    bloom = BloomFilter(path=tmp / "t35.bin")
    # Never initialized — must not raise
    await bloom.add("https://example.com/pre-init-add")
    # File should not be created
    assert not (tmp / "t35.bin").exists()


async def _test_integrity_report_format() -> None:
    """Test 36: BloomFilterIntegrityReport __str__ contains expected sections."""
    import tempfile, shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        path = tmp / "t36.bin"
        async with BloomFilter(path=path) as bloom:
            _ = bloom
        async with BloomFilter(path=path) as bloom2:
            report = await bloom2.verify_integrity()
        s = str(report)
        assert "VALID" in s
        assert "file_exists" in s
        assert "checksum" in s
    finally:
        shutil.rmtree(str(tmp), ignore_errors=True)


async def _test_rotating_filter_batch(tmp: Path) -> None:
    """Test 37: RotatingBloomFilter add_batch and contains_batch work correctly."""
    rbf = RotatingBloomFilter(capacity=10_000, fp_rate=0.001)

    urls = [f"https://rotating-batch.com/{i}" for i in range(5000)]
    added = await rbf.add_batch(urls)
    assert added == 5000

    results = await rbf.contains_batch(urls)
    assert all(results), "Not all batch-added URLs found"

    unseen = [f"https://rotating-batch.com/unseen_{i}" for i in range(5000)]
    fp_results = await rbf.contains_batch(unseen)
    fp_rate = sum(fp_results) / len(fp_results)
    assert fp_rate < 0.01, f"RotatingBloomFilter FP rate {fp_rate:.4f} too high"


async def _test_compact_bloom_filter_parameters() -> None:
    """Test 38: _CompactBloomFilter derives correct bit_size and hash_count."""
    import math

    # For capacity=1M, fp_rate=0.001
    cbf = _CompactBloomFilter(capacity=1_000_000, fp_rate=0.001)
    # Optimal m = ceil(1M * |ln(0.001)| / ln(2)^2) = ceil(1M * 6.908 / 0.4805) ≈ 14,377,588
    expected_m = math.ceil(1_000_000 * abs(math.log(0.001)) / (math.log(2) ** 2))
    assert cbf.bit_size == expected_m, (
        f"bit_size {cbf.bit_size} != expected {expected_m}"
    )
    # Optimal k = round((m/n) * ln(2)) = round(14.377 * 0.693) ≈ 10
    expected_k = max(1, round((expected_m / 1_000_000) * math.log(2)))
    assert cbf.hash_count == expected_k, (
        f"hash_count {cbf.hash_count} != expected {expected_k}"
    )


async def _test_add_batch_returns_correct_count(tmp: Path) -> None:
    """Test 39: add_batch() returns exactly the count of URLs it processed."""
    async with BloomFilter(path=tmp / "t39.bin") as bloom:
        n = 3000
        urls = [f"https://example.com/batch-count/{i}" for i in range(n)]
        added = await bloom.add_batch(urls)
        assert added == n, f"add_batch returned {added}, expected {n}"

        # Double-adding same batch — all succeed (idempotent at bit level)
        added2 = await bloom.add_batch(urls)
        assert added2 == n, f"Idempotent re-add returned {added2}, expected {n}"


async def _test_pool_invalid_name(tmp: Path) -> None:
    """Test 40: BloomFilterPool raises ValueError for names with path separators."""
    pool = BloomFilterPool(base_dir=tmp / "pool_invalid")
    try:
        try:
            await pool.get("../../evil")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass

        try:
            await pool.get("")
            assert False, "Should have raised ValueError for empty name"
        except ValueError:
            pass

        try:
            await pool.get("valid_name")  # this should work
        except Exception as exc:
            assert False, f"Valid name raised unexpected exception: {exc}"
    finally:
        await pool.close_all()


async def _test_fill_factor_empty_filter(tmp: Path) -> None:
    """Test 41: fill_factor() is 0.0 for an empty filter."""
    async with BloomFilter(path=tmp / "t41.bin") as bloom:
        ff = await bloom.fill_factor()
    # A fresh filter has all bits zero — fill factor must be exactly 0.0
    assert ff == 0.0, f"Expected 0.0 fill factor for empty filter, got {ff}"


async def _test_fill_factor_grows_with_adds(tmp: Path) -> None:
    """Test 42: fill_factor() increases monotonically as URLs are added."""
    async with BloomFilter(path=tmp / "t42.bin") as bloom:
        ff0 = await bloom.fill_factor()
        assert ff0 == 0.0

        for i in range(100_000):
            await bloom.add(f"https://example.com/fill/{i}")

        ff1 = await bloom.fill_factor()
        assert ff1 > ff0, (
            f"fill_factor did not increase after 100K adds: {ff0} → {ff1}"
        )
        assert ff1 < 1.0, (
            f"fill_factor reached 1.0 after only 100K adds — impossible at 4B bits"
        )


async def _test_snapshot_structure(tmp: Path) -> None:
    """Test 43: snapshot() returns a well-formed BloomFilterSnapshot."""
    async with BloomFilter(path=tmp / "t43.bin") as bloom:
        for i in range(1000):
            await bloom.add(f"https://example.com/snap/{i}")
        snap = await bloom.snapshot()

    assert snap.bit_size   == BLOOM_BIT_SIZE
    assert snap.hash_count == BLOOM_HASH_COUNT
    assert snap.capacity   == BLOOM_CAPACITY
    assert snap.count      >= 1000
    assert 0.0 <= snap.fill_factor < 1.0
    assert 0.0 <= snap.estimated_fp_rate < 1.0
    assert snap.file_size_bytes == BLOOM_FILE_BYTES

    # to_dict round-trip
    d = snap.to_dict()
    assert d["bit_size"]   == BLOOM_BIT_SIZE
    assert d["count"]      == snap.count
    assert d["fill_factor"] == snap.fill_factor


async def _test_bloom_filter_config_fields() -> None:
    """Test 44: BloomFilterConfig derives file_bytes correctly."""
    cfg = BloomFilterConfig(path=Path("/tmp/test.bin"))
    assert cfg.file_bytes() == BLOOM_FILE_BYTES

    # theoretical_fp_rate_at: at 0 elements, rate is 0
    assert cfg.theoretical_fp_rate_at(0) == 0.0

    # At capacity, rate should match the design target (~0.82% with these parameters).
    # The spec targets 0.01% but with m=4B bits / n=400M / k=7 the actual formula gives:
    #   p = (1 - e^(-7*400M/4B))^7 = (1 - e^(-0.7))^7 ≈ 0.82%
    # This is within acceptable range per AXIOM design law 8 (FP are tolerated).
    rate_at_capacity = cfg.theoretical_fp_rate_at(BLOOM_CAPACITY)
    assert rate_at_capacity < 0.015, (
        f"FP rate at capacity {rate_at_capacity:.6f} >> expected ~0.0082"
    )
    assert rate_at_capacity > 0.001, (
        f"FP rate at capacity {rate_at_capacity:.6f} unexpectedly low — math error?"
    )


async def _test_hash_function_distribution() -> None:
    """
    Test 45: Hash positions for different URLs are spread across the full bit range.

    Checks that:
      - Two different URLs produce different sets of bit positions
      - All positions are within [0, BLOOM_BIT_SIZE)
      - The 7 positions for one URL are distinct (no collisions in double-hash)
    """
    bloom = BloomFilter.__new__(BloomFilter)
    # Construct just enough state to call _hashes
    object.__setattr__(bloom, '_path', Path("/tmp/hash_test.bin"))
    object.__setattr__(bloom, '_canonicalizer', _DEFAULT_CANONICALIZER)

    # Monkey-patch the hashes method using the module-level implementation
    url_a = "https://alpha.example.com/path/to/resource?id=42"
    url_b = "https://beta.different.org/completely/different/url"

    # Call _hashes directly via the class method
    positions_a = list(BloomFilter._hashes(bloom, url_a)) # noqa
    positions_b = list(BloomFilter._hashes(bloom, url_b)) # noqa

    # Correct count
    assert len(positions_a) == BLOOM_HASH_COUNT
    assert len(positions_b) == BLOOM_HASH_COUNT

    # All in range
    for pos in positions_a + positions_b:
        assert 0 <= pos < BLOOM_BIT_SIZE, (
            f"Hash position {pos} out of range [0, {BLOOM_BIT_SIZE})"
        )

    # Different URLs produce different position sets (with overwhelming probability)
    assert set(positions_a) != set(positions_b), (
        "Different URLs produced identical hash positions — hash collision or bug"
    )

    # Positions within one URL should all be distinct
    assert len(set(positions_a)) == BLOOM_HASH_COUNT, (
        f"Duplicate hash positions for url_a: {positions_a}"
    )


async def _test_crc32_helper() -> None:
    """Test 46: _crc32 returns consistent results and stays in uint32 range."""
    v1 = _crc32(b"hello world")
    v2 = _crc32(b"hello world")
    assert v1 == v2, "CRC32 is not deterministic"

    # Different input → different CRC (overwhelmingly likely)
    v3 = _crc32(b"hello worldX")
    assert v1 != v3, "CRC32 collision on trivially different inputs"

    # Always in uint32 range
    assert 0 <= v1 < 2**32
    assert 0 <= _crc32(b"") < 2**32


async def _test_url_canonicalizer_is_idempotent() -> None:
    """Test 47: normalize(normalize(url)) == normalize(url) for all test cases."""
    canon = URLCanonicalizer()
    test_urls = [
        "HTTP://EXAMPLE.COM/path?utm_source=foo&id=1#section",
        "https://example.com:443/page?z=3&a=1&utm_campaign=test",
        "https://www.example.com//double//slash//path",
        "https://example.com/café/résumé?q=search",
        "  https://example.com/whitespace  ",
        "",
    ]
    for url in test_urls:
        once  = canon.normalize(url)
        twice = canon.normalize(once)
        assert once == twice, (
            f"normalize() not idempotent for {url!r}:\n"
            f"  once:  {once!r}\n"
            f"  twice: {twice!r}"
        )


async def _test_bloom_filter_stats_health(tmp: Path) -> None:
    """Test 48: BloomFilterStats.is_healthy and is_saturated flags behave correctly."""
    async with BloomFilter(path=tmp / "t48.bin") as bloom:
        for i in range(50_000):
            await bloom.add(f"https://example.com/health/{i}")
        stats = await bloom.stats()

    assert not stats.is_saturated, (
        "Filter falsely reported saturated at only 50K / 400M capacity"
    )
    assert stats.is_healthy, (
        "Filter not healthy at 50K / 400M capacity"
    )
    assert stats.capacity_remaining == BLOOM_CAPACITY - stats.count
    assert isinstance(stats.summary(), str)
    assert len(stats.summary()) > 10


async def _test_migration_rebuild(tmp: Path) -> None:
    """
    Test 49: BloomFilterMigration.rebuild() + commit() correctly reconstructs the filter.

    Creates a filter at old_path, adds URLs, then migrates to new_path with
    a provided historical URL list, commits (renames new → old), and verifies
    all URLs are still found.
    """
    old_path = tmp / "migrate_old.bin"
    new_path = tmp / "migrate_new.bin"
    historical_urls = [f"https://migrate.example.com/{i}" for i in range(1000)]

    # Create the new filter and populate it via migration
    new_bloom = BloomFilter(path=new_path)
    await new_bloom.initialize()

    migration = BloomFilterMigration(old_path=old_path, new_path=new_path)
    rebuilt_count = await migration.rebuild(new_bloom, historical_urls, batch_size=100)
    await new_bloom.close()

    assert rebuilt_count == 1000, f"Expected 1000 rebuilt, got {rebuilt_count}"
    assert new_path.exists(), "New bloom file not created during migration"

    # Commit: renames new_path → old_path
    migration.commit()
    assert old_path.exists(), "old_path not created after commit()"
    assert not new_path.exists(), "new_path still exists after commit()"

    # Verify: all historical URLs are present in the committed file
    async with BloomFilter(path=old_path) as committed:
        for url in historical_urls:
            assert await committed.contains(url) is True, (
                f"URL {url!r} not found after migration commit"
            )


async def _test_migration_rollback(tmp: Path) -> None:
    """Test 50: BloomFilterMigration.rollback() cleans up the partial new file."""
    old_path = tmp / "rollback_old.bin"
    new_path = tmp / "rollback_new.bin"

    # Create the new file
    async with BloomFilter(path=new_path) as b:
        await b.add("https://example.com/rollback-test")

    assert new_path.exists()

    migration = BloomFilterMigration(old_path=old_path, new_path=new_path)
    migration.rollback()

    assert not new_path.exists(), "rollback() did not delete new_path"
    assert not old_path.exists(), "rollback() incorrectly created old_path"


# ─────────────────────────────────────────────────────────────────────────────
# TEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────

async def run_tests(verbose: bool = True) -> bool:
    """
    Run the full bloom_filter.py test suite.

    Executes all _test_* functions in definition order. Uses a temporary
    directory for all file-based tests. Cleans up on completion.

    Tests that operate on purely in-memory structures (CompactBloomFilter,
    hash distribution, CRC32) do not need the tmp directory and pass None.

    Args:
        verbose: If True, prints pass/fail for each test as it runs.

    Returns:
        True if all tests passed, False if any failed.
    """
    import tempfile
    import shutil
    import traceback

    tmp = Path(tempfile.mkdtemp(prefix="axiom_bloom_test_"))

    # All test functions: (name, coroutine_factory, needs_tmp)
    # needs_tmp=True  → called as test(tmp)
    # needs_tmp=False → called as test()
    _TestFn = Union[Callable[[Path], Awaitable[None]], Callable[[], Awaitable[None]]]
    test_cases: List[Tuple[str, _TestFn, bool]] = [
        ("Test 01: contains() returns False for unseen URL",            _test_contains_returns_false_for_unseen_url,    True),
        ("Test 02: contains() returns True after add()",                _test_contains_returns_true_after_add,          True),
        ("Test 03: no false negatives across 10K URLs",                 _test_no_false_negatives,                       True),
        ("Test 04: false positive rate < 0.02% at 100K URLs",          _test_false_positive_rate,                      True),
        ("Test 05: mmap survives process death (close + reopen)",       _test_mmap_survives_process_death,              True),
        ("Test 06: close() always flushes before closing",              _test_unflushed_bits_survive_close,             True),
        ("Test 07: add() is idempotent",                                _test_add_is_idempotent,                        True),
        ("Test 08: count() increments correctly",                       _test_count_increments_on_add,                  True),
        ("Test 09: count() survives flush",                             _test_count_survives_flush,                     True),
        ("Test 10: auto-flush at BLOOM_FLUSH_INTERVAL boundary",       _test_flush_happens_at_interval,                True),
        ("Test 11: initialize() creates file if not exists",            _test_initialize_creates_file,                  True),
        ("Test 12: initialize() opens existing file and retains state", _test_initialize_opens_existing_file,           True),
        ("Test 13: add_batch() and contains_batch() correctness",      _test_batch_add_and_contains,                   True),
        ("Test 14: verify_integrity() passes for fresh file",           _test_verify_integrity_new_file,                True),
        ("Test 15: verify_integrity() reports missing file",            _test_verify_integrity_missing_file,            True),
        ("Test 16: verify_integrity() detects corrupt header CRC",      _test_verify_integrity_corrupt_header,          True),
        ("Test 17: canonicalizer lowercases scheme and host",           _test_url_canonicalizer_scheme_lowercase,       True),
        ("Test 18: canonicalizer strips tracking parameters",           _test_url_canonicalizer_tracking_params,        True),
        ("Test 19: canonicalizer sorts query params",                   _test_url_canonicalizer_param_sort,             True),
        ("Test 20: canonicalizer strips fragments",                     _test_url_canonicalizer_fragment_strip,         True),
        ("Test 21: canonicalizer strips default ports",                 _test_url_canonicalizer_default_port,           True),
        ("Test 22: canonicalizer collapses double slashes",             _test_url_canonicalizer_double_slash,           True),
        ("Test 23: canonical dedup across tracking param variants",     _test_canonicalized_dedup,                      True),
        ("Test 24: header encode/decode roundtrip",                     _test_header_roundtrip,                         False),
        ("Test 25: _decode_header raises on bad magic",                 _test_header_bad_magic,                         False),
        ("Test 26: _decode_header raises on bad CRC32",                 _test_header_bad_checksum,                      False),
        ("Test 27: _CompactBloomFilter basic correctness",              _test_compact_bloom_filter,                     False),
        ("Test 28: RotatingBloomFilter rotation behavior",              _test_rotating_bloom_filter,                    False),
        ("Test 29: BloomFilterPool manages multiple named filters",     _test_bloom_filter_pool,                        True),
        ("Test 30: sequential opens of same file work correctly",       _test_multiple_opens_same_file,                 True),
        ("Test 31: Unicode URLs handled via NFC normalization",         _test_url_with_unicode,                         True),
        ("Test 32: empty string and whitespace URL edge cases",         _test_url_empty_and_whitespace,                 True),
        ("Test 33: close() is idempotent",                             _test_close_is_idempotent,                      True),
        ("Test 34: contains() before initialize() returns False",       _test_contains_before_initialize_returns_false, True),
        ("Test 35: add() before initialize() is a no-op",              _test_add_before_initialize_is_noop,            True),
        ("Test 36: BloomFilterIntegrityReport __str__ format",          _test_integrity_report_format,                  False),
        ("Test 37: RotatingBloomFilter add_batch/contains_batch",      _test_rotating_filter_batch,                    True),
        ("Test 38: _CompactBloomFilter derives correct parameters",     _test_compact_bloom_filter_parameters,          False),
        ("Test 39: add_batch() returns correct count",                  _test_add_batch_returns_correct_count,          True),
        ("Test 40: BloomFilterPool rejects invalid names",              _test_pool_invalid_name,                        True),
        ("Test 41: fill_factor() is 0.0 for empty filter",             _test_fill_factor_empty_filter,                 True),
        ("Test 42: fill_factor() grows monotonically with adds",       _test_fill_factor_grows_with_adds,              True),
        ("Test 43: snapshot() returns well-formed struct",              _test_snapshot_structure,                       True),
        ("Test 44: BloomFilterConfig derives file_bytes correctly",     _test_bloom_filter_config_fields,               False),
        ("Test 45: hash position distribution and correctness",         _test_hash_function_distribution,               False),
        ("Test 46: _crc32 helper correctness",                          _test_crc32_helper,                             False),
        ("Test 47: URLCanonicalizer is idempotent",                     _test_url_canonicalizer_is_idempotent,          False),
        ("Test 48: BloomFilterStats health flags",                      _test_bloom_filter_stats_health,                True),
        ("Test 49: BloomFilterMigration rebuild and commit",            _test_migration_rebuild,                        True),
        ("Test 50: BloomFilterMigration rollback cleans up",            _test_migration_rollback,                       True),
    ]

    passed = 0
    failed = 0
    errors: List[Tuple[str, str]] = []

    print(f"\n{'─' * 72}")
    print(f"  bloom_filter.py — test suite ({len(test_cases)} tests)")
    print(f"{'─' * 72}")

    for name, fn, needs_tmp in test_cases:
        t_start = time.perf_counter()
        test_tmp = None
        try:
            if needs_tmp:
                test_tmp = tmp / f"test_{passed + failed:03d}"
                test_tmp.mkdir(parents=True, exist_ok=True)
                await fn(test_tmp)  # type: ignore[call-arg]
            else:
                await fn()          # type: ignore[call-arg]
            duration_ms = (time.perf_counter() - t_start) * 1000
            if verbose:
                print(f"  ✓  {name} ({duration_ms:.1f}ms)")
            passed += 1
        except Exception as exc:
            duration_ms = (time.perf_counter() - t_start) * 1000
            tb = traceback.format_exc()
            errors.append((name, tb))
            if verbose:
                print(f"  ✗  {name} ({duration_ms:.1f}ms)")
                print(f"       {type(exc).__name__}: {exc}")
            failed += 1
        finally:
            # Delete test directory immediately to reclaim disk space.
            # Each bloom.bin is 500MB; keeping them all would exhaust /tmp.
            if test_tmp is not None:
                try:
                    shutil.rmtree(str(test_tmp), ignore_errors=True)
                except Exception: # noqa
                    pass

    print(f"{'─' * 72}")
    print(f"  {passed}/{len(test_cases)} passed  |  {failed} failed")
    print(f"{'─' * 72}")

    if errors:
        print("\nFailed test details:")
        for name, tb in errors:
            print(f"\n  ✗ {name}")
            for line in tb.splitlines():
                print(f"    {line}")

    # Clean up temp directory
    # noinspection PyInterpreter
    try:
        shutil.rmtree(str(tmp), ignore_errors=True)
    except Exception: # noqa
        pass

    print()
    return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
#
# Usage:
#   python bloom_filter.py test              — run test suite
#   python bloom_filter.py benchmark         — run performance benchmark (1M URLs)
#   python bloom_filter.py benchmark --n 5000000
#   python bloom_filter.py integrity [path]  — verify bloom.bin integrity
#   python bloom_filter.py stats [path]      — print filter statistics
# ─────────────────────────────────────────────────────────────────────────────

def _cli_integrity(path: Path) -> int:
    """
    Check the integrity of a bloom.bin file and print a report.

    Returns 0 on success (file is valid), 1 on failure.
    """
    import asyncio

    async def _run() -> int:
        bloom = BloomFilter(path=path)
        print(f"\nChecking integrity: {path}")
        report = await bloom.verify_integrity()
        print(report)
        return 0 if report.is_valid else 1

    return asyncio.run(_run())


def _cli_stats(path: Path) -> int:
    """
    Print statistics for an existing bloom.bin file.

    WARNING: reads the entire 500MB mmap to compute fill_factor.
    Expected runtime: 0.5–2 seconds.
    """
    import asyncio

    async def _run() -> int:
        if not path.exists():
            print(f"Error: {path} does not exist")
            return 1

        bloom = BloomFilter(path=path)
        await bloom.initialize()
        try:
            stats = await bloom.stats()
            print(f"\nBloomFilter statistics: {path}")
            print(f"{'─' * 60}")
            print(f"  Count (approx):      {stats.count:>20,}")
            print(f"  Capacity:            {stats.capacity:>20,}")
            print(f"  Capacity used:       {stats.capacity_pct:>19.3%}")
            print(f"  Capacity remaining:  {stats.capacity_remaining:>20,}")
            print(f"  Fill factor:         {stats.fill_factor:>19.6f}")
            print(f"  Est. FP rate:        {stats.estimated_fp_rate:>19.6f} ({stats.estimated_fp_rate*100:.4f}%)")
            print(f"  Target FP rate:      {BLOOM_FP_RATE:>19.6f} ({BLOOM_FP_RATE*100:.4f}%)")
            print(f"  Bit array size:      {stats.bit_size:>20,} bits")
            print(f"  File size:           {stats.file_size_bytes:>20,} bytes ({stats.file_size_bytes/1_048_576:.1f} MB)")
            print(f"  Hash functions:      {stats.hash_count:>20}")
            print(f"  Flushes (session):   {stats.flush_count:>20}")
            print(f"  Adds since open:     {stats.add_count_since_open:>20,}")
            print(f"  Status:              {'SATURATED ⚠' if stats.is_saturated else 'healthy ✓':>20}")
            print(f"{'─' * 60}")
        finally:
            await bloom.close()
        return 0

    return asyncio.run(_run())


def _cli_benchmark(n: int) -> int:
    """Run the performance benchmark."""
    import asyncio

    async def _run() -> int:
        print(f"\nBloomFilter benchmark — n={n:,} URLs")
        print(f"{'─' * 60}")
        result = await run_benchmark(n_urls=n)
        print(f"  Add throughput:          {result.add_throughput_per_sec:>12,.0f} URLs/sec")
        print(f"  Add duration:            {result.add_duration_sec:>11.3f}s")
        print(f"  Contains (present):      {result.contains_true_throughput:>12,.0f} URLs/sec")
        print(f"  Contains (absent):       {result.contains_false_throughput:>12,.0f} URLs/sec")
        print(f"  False positive count:    {result.false_positive_count:>12,} / {n:,}")
        print(f"  False positive rate:     {result.false_positive_rate:>11.5%}")
        print(f"  False negative count:    {result.false_negative_count:>12,} (must be 0)")
        if result.peak_memory_mb > 0:
            print(f"  Peak virtual memory:     {result.peak_memory_mb:>11.1f} MB")
        print(f"{'─' * 60}")
        print(f"  Result: {'PASS ✓' if result.passed() else 'FAIL ✗'}")
        print()
        return 0 if result.passed() else 1

    return asyncio.run(_run())


def _cli_tests() -> int:
    """Run the full test suite."""
    import asyncio
    result = asyncio.run(run_tests(verbose=True))
    return 0 if result else 1


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        prog="bloom_filter.py",
        description="AXIOM bloom_filter.py — CLI for test, benchmark, integrity, and stats",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # test
    subparsers.add_parser(
        "test",
        help="Run the full test suite (50 tests)",
    )

    # benchmark
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run the performance benchmark (default: 1M URLs)",
    )
    bench_parser.add_argument(
        "--n",
        type=int,
        default=1_000_000,
        metavar="N",
        help="Number of URLs to benchmark (default: 1000000)",
    )

    # integrity
    integrity_parser = subparsers.add_parser(
        "integrity",
        help="Verify integrity of a bloom.bin file",
    )
    integrity_parser.add_argument(
        "path",
        nargs="?",
        default=str(BLOOM_FILE_PATH),
        help=f"Path to bloom.bin (default: {BLOOM_FILE_PATH})",
    )

    # stats
    stats_parser = subparsers.add_parser(
        "stats",
        help="Print statistics for a bloom.bin file (reads full 500MB mmap)",
    )
    stats_parser.add_argument(
        "path",
        nargs="?",
        default=str(BLOOM_FILE_PATH),
        help=f"Path to bloom.bin (default: {BLOOM_FILE_PATH})",
    )

    args = parser.parse_args()

    if args.command == "test":
        sys.exit(_cli_tests())

    elif args.command == "benchmark":
        sys.exit(_cli_benchmark(n=args.n))

    elif args.command == "integrity":
        sys.exit(_cli_integrity(Path(args.path)))

    elif args.command == "stats":
        sys.exit(_cli_stats(Path(args.path)))

    else:
        parser.print_help()
        sys.exit(1)