"""
crawler/crawl_cursor.py
=======================
Interrupt-safe position tracking with atomic checkpoint writes.

The crawl cursor is the single piece of state that separates a crashed
crawl from a duplicate crawl.  Every CURSOR_CHECKPOINT_INTERVAL URLs the
fetcher writes the current manifest position to disk.  On restart after any
kind of process death — SIGKILL, OOM, power loss, container eviction — the
frontier reads this cursor and yields from the exact position where execution
stopped.  No URLs are re-fetched beyond the last checkpoint window.

Architecture invariant
──────────────────────
This file has zero dependencies on other crawler/ files.  It does not know
about Bloom filter, rate limiter, fetcher, or frontier.  It knows one thing:
a manifest_id maps to an integer position.  Writing that mapping atomically
and reading it back correctly is the complete contract.

The cursor is owned by the fetcher.  The frontier is owned by the fetcher.
The cursor does not talk to the frontier.  The fetcher wires them together.

Persistence model
─────────────────
SQLite WAL mode.  Single file at CURSOR_DB_PATH.  One row per active manifest.
Every checkpoint is written inside BEGIN IMMEDIATE … COMMIT.  The ACID
guarantee of SQLite means the position either made it to disk or it did not.
No partial writes.  No corruption possible from a mid-write crash.

WAL mode is mandatory.  DELETE/TRUNCATE journal modes are not acceptable — WAL
provides page-level atomicity without a full-file exclusive lock, meaning
monitoring readers do not block the writer.  WAL also appends rather than
overwrites, delivering better crash-recovery semantics on every supported
platform.

Crash recovery contract
───────────────────────
A checkpoint at position N guarantees: every URL at index 0 … N has been
emitted to the bus (RawFetchEvent or FetchAnomalyEvent), the Bloom filter has
been updated, and the frontier row has been marked done/failed/skipped.  On
restart the frontier yields from N+1.  The worst-case re-fetch window is at
most CURSOR_CHECKPOINT_INTERVAL – 1 URLs.  The Bloom filter prevents duplicate
downstream signal for those re-fetches — they arrive as bloom-skipped on the
second pass and produce no RawFetchEvent.

Why not checkpoint every URL
────────────────────────────
Every SQLite write involves at minimum a WAL file append followed by an
fdatasync.  On commodity storage, fdatasync costs 1–50 ms per call.  A 100k-
URL manifest checkpointed on every URL would accumulate 100–5000 seconds of
fsync overhead.  At CURSOR_CHECKPOINT_INTERVAL = 100, the overhead is 1–50 s
per manifest — acceptable.  The safety improvement from every-URL checkpointing
is zero: the worst-case re-fetch window is the count of URLs processed since
the last checkpoint, which is bounded by the interval regardless of what
happened between checkpoint events.

BEGIN IMMEDIATE vs BEGIN
────────────────────────
SQLite offers three transaction modes: DEFERRED (default), IMMEDIATE, and
EXCLUSIVE.  DEFERRED acquires only a SHARED lock until the first write, then
upgrades to RESERVED, then to EXCLUSIVE at commit.  In a WAL-mode database
there is no RESERVED lock, but the deferred upgrade can still cause an
SQLITE_BUSY_SNAPSHOT error if a concurrent reader has already advanced the
WAL reader version past the point the writer wants to commit.

IMMEDIATE acquires the write lock (WAL write lock) at transaction start,
before any reads or writes occur.  This eliminates the upgrade race.  For a
component whose writes are unconditional INSERT OR REPLACE operations with no
read-modify-write cycle, IMMEDIATE is the correct choice.  It is more
conservative but the lock hold time is microseconds — the time to insert one
row.

Dependency boundary
───────────────────
Imports (direct):
  aiosqlite    — async SQLite driver (pip: aiosqlite)
  asyncio      — stdlib async primitives
  time         — stdlib monotonic and wall clocks
  logging      — stdlib structured logger
  pathlib      — stdlib Path type
  dataclasses  — stdlib frozen dataclass
  typing       — stdlib type annotations

Imports (internal):
  exceptions   — CursorError only; no contracts.py dependency

This file does NOT import from:
  bloom_filter, frontier, rate_limiter, fetcher — never
  topology/, world_model/, signal_kernel/ — never
  Any third-party library except aiosqlite — never

AXIOM Internal // Do Not Surface
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ( # noqa
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Callable,
    Awaitable,
)

import aiosqlite

# ---------------------------------------------------------------------------
# Internal dependency — CursorError only.  No contracts.py import.  This file
# is Layer 1 acquisition infrastructure; it must not drag in signal-kernel
# types.  If exceptions.py is not yet present in the local package (e.g.
# during isolated unit testing), the import is stubbed to a plain ValueError
# subclass so the test suite remains runnable without the full AXIOM tree.
# ---------------------------------------------------------------------------
try:
    from exceptions import CursorError  # type: ignore[import]
except ModuleNotFoundError:
    # Fallback for isolated unit testing outside the AXIOM package tree.
    # The stub carries the same interface callers expect: manifest_id,
    # position, operation, db_error keyword arguments.
    class CursorError(Exception):  # type: ignore[no-redef]
        """Fallback stub — replaced by exceptions.CursorError at runtime."""

        exception_code: str = "CRAWLER_CURSOR_DB_ERROR"
        is_hard_stop: bool = False

        def __init__(
            self,
            *,
            manifest_id: str,
            position: int,
            operation: str,
            db_error: str,
        ) -> None:
            super().__init__(
                f"CursorError[{operation}] manifest={manifest_id} "
                f"pos={position}: {db_error}"
            )
            self.manifest_id = manifest_id
            self.position = position
            self.operation = operation
            self.db_error = db_error

        def to_audit_dict(self) -> Dict[str, object]:
            return {
                "exception_code": self.exception_code,
                "exception_class": type(self).__name__,
                "is_hard_stop": self.is_hard_stop,
                "manifest_id": self.manifest_id,
                "position": self.position,
                "operation": self.operation,
                "db_error": self.db_error,
            }


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# All tunable knobs for this module.  Names are UPPER_SNAKE.  Every constant
# has a comment explaining its purpose and the rationale for its default value.
# Do not scatter magic numbers through the implementation — everything goes here.
# ─────────────────────────────────────────────────────────────────────────────

# Path to the SQLite database that stores cursor rows.
# The `store/` directory must exist before initialize() is called.
# crawl_cursor.py creates the file inside that directory, but will not
# create the directory itself (that is the host environment's responsibility).
CURSOR_DB_PATH: Path = Path("store/crawl_cursor.db")

# How many URLs the fetcher processes between checkpoint writes.
# This value represents the maximum number of URLs that could be re-fetched
# after an unclean shutdown.  Lower values reduce re-fetch risk but increase
# fsync frequency.  100 is the production-validated balance point.
CURSOR_CHECKPOINT_INTERVAL: int = 100

# SQLite busy-handler timeout in milliseconds.
# If another connection holds the WAL write lock when we attempt BEGIN
# IMMEDIATE, SQLite will retry for up to this many milliseconds before
# returning SQLITE_BUSY.  In normal operation only one process holds the
# cursor DB open, so this timeout is almost never reached.  Setting it to a
# non-zero value protects against monitoring scripts that may open the DB
# concurrently.
CURSOR_BUSY_TIMEOUT_MS: int = 5_000

# WAL auto-checkpoint threshold (pages).
# SQLite auto-checkpoints the WAL back to the main file when the WAL
# accumulates this many pages.  A smaller value keeps the WAL file size
# bounded; a larger value reduces checkpoint frequency.  1000 pages ×
# default 4096 bytes/page = ~4 MB WAL ceiling — acceptable for a low-write
# workload like cursor checkpointing.
CURSOR_WAL_AUTOCHECKPOINT: int = 1_000

# After how many checkpoint operations the WAL manager issues a forced
# full-checkpoint (PRAGMA wal_checkpoint(TRUNCATE)).  This compacts the WAL
# file back to near-zero length.  The default 500 corresponds to 50,000 URL
# completions between forced compactions — rare but necessary to prevent WAL
# unbounded growth on very long manifests.
CURSOR_WAL_FORCE_INTERVAL: int = 500

# Age threshold in seconds above which a cursor is considered stale.
# A cursor is stale if it was last checkpointed more than this many seconds
# ago.  Stale cursors indicate a manifest that is no longer running.  The
# default is 24 hours (86_400 s).  The stale_cursors() method uses this.
CURSOR_STALE_THRESHOLD_SECONDS: float = 86_400.0

# Minimum progress velocity (URLs/second) below which a cursor is flagged
# as stalled.  Computed over a rolling window by _SessionTracker.  Used
# only for health diagnostics — does not affect checkpoint behavior.
CURSOR_STALL_VELOCITY_THRESHOLD: float = 0.001  # effectively zero

# How many checkpoint latency samples to retain per manifest in memory.
# Only the last N samples are used to compute rolling statistics.  Older
# samples are discarded.  This cap prevents unbounded memory growth on
# very long manifests.
CURSOR_LATENCY_SAMPLE_CAP: int = 200

# SQLite page size for newly created cursor databases.
# Must be a power of two between 512 and 65536.  4096 matches the Linux
# page cache page size and is the SQLite default for new databases.
# This pragma only has effect on database creation — it cannot change an
# existing database's page size.
CURSOR_PAGE_SIZE: int = 4_096

# Schema version constant.  Stored in user_version PRAGMA.  Bump this if
# the schema changes.  initialize() checks the version on open and raises
# CursorError if the on-disk schema is from an incompatible older version.
CURSOR_SCHEMA_VERSION: int = 1

# Name of the logger used throughout this module.  External code can adjust
# the log level by retrieving this logger by name.
_LOGGER_NAME: str = "axiom.crawler.cursor"

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# All logging in this file goes through this logger.  External consumers can
# attach handlers or adjust the level without modifying this module.
# ─────────────────────────────────────────────────────────────────────────────

_log: logging.Logger = logging.getLogger(_LOGGER_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# SQL DDL
# All schema definition lives here.  No SQL strings are scattered through
# the implementation.  Every statement is named with a leading comment so
# the schema is readable without cross-referencing the surrounding code.
#
# WITHOUT ROWID design decision
# ──────────────────────────────
# The cursors table is WITHOUT ROWID.  Standard SQLite tables maintain two
# B-trees: a rowid-keyed data B-tree and a secondary index B-tree for the
# PRIMARY KEY.  Every PK lookup traverses both trees (PK index → rowid →
# data).  WITHOUT ROWID collapses them: the PK is the B-tree key and the
# row data is stored inline.  One traversal instead of two.
#
# This is correct here because:
#   1. All reads and writes are by manifest_id (the PK).  No rowid scans.
#   2. The table is tiny (1–20 rows).  The B-tree has at most 2 levels.
#   3. manifest_id is a fixed-width UUID4 string (36 bytes) — not a blob
#      and not unboundedly wide, so WITHOUT ROWID's size constraint is met.
#
# WITHOUT ROWID requires SQLite ≥ 3.8.2 (released 2013).  Safe to assume.
#
# STRICT mode design decision
# ────────────────────────────
# STRICT enforces declared column types at the storage layer.  Without it,
# SQLite accepts any value in any column regardless of declared type — a
# TEXT value silently inserts into an INTEGER column.  With STRICT, type
# mismatches raise an error at INSERT time instead of corrupting silently.
# Requires SQLite ≥ 3.37.0 (released November 2021).
# If the runtime SQLite version is older, _SchemaManager falls back to the
# non-STRICT DDL automatically.  See _SchemaManager._create_tables().
# ─────────────────────────────────────────────────────────────────────────────

_DDL_CURSORS_TABLE_STRICT: str = """
CREATE TABLE IF NOT EXISTS cursors (
    -- Primary key: which manifest this row tracks.  UUID4 string.
    manifest_id     TEXT    PRIMARY KEY,
    -- The index into manifest.urls[] of the last completed URL.
    -- The next URL to fetch on restart is position + 1.
    position        INTEGER NOT NULL,
    -- Human-readable URL at position — for audit and operator inspection.
    -- The frontier owns the canonical URL list; this is a convenience copy.
    url             TEXT    NOT NULL,
    -- Unix timestamp (float) when this checkpoint was written.
    checkpoint_at   REAL    NOT NULL,
    -- Total URL count in the manifest.  Used to compute progress fraction
    -- without querying the frontier.
    total_urls      INTEGER NOT NULL
) WITHOUT ROWID, STRICT;
"""

# Fallback DDL for SQLite < 3.37.0.  Identical schema, no STRICT.
# WITHOUT ROWID is retained — it requires only 3.8.2.
_DDL_CURSORS_TABLE_COMPAT: str = """
CREATE TABLE IF NOT EXISTS cursors (
    manifest_id     TEXT    PRIMARY KEY,
    position        INTEGER NOT NULL,
    url             TEXT    NOT NULL,
    checkpoint_at   REAL    NOT NULL,
    total_urls      INTEGER NOT NULL
) WITHOUT ROWID;
"""

# Active DDL — resolved at module load time against the runtime SQLite version.
# _SchemaManager._create_tables() uses this name, not the version-specific ones.
import sqlite3 as _sqlite3

_SQLITE_VERSION_INFO: tuple = _sqlite3.sqlite_version_info  # e.g. (3, 39, 2)
_SQLITE_SUPPORTS_STRICT: bool = _SQLITE_VERSION_INFO >= (3, 37, 0)

_DDL_CURSORS_TABLE: str = (
    _DDL_CURSORS_TABLE_STRICT
    if _SQLITE_SUPPORTS_STRICT
    else _DDL_CURSORS_TABLE_COMPAT
)

# checkpoint_at index supports stale-cursor GC and ordered monitoring queries.
# WITHOUT ROWID tables always have a B-tree on the PK; secondary indexes on
# WITHOUT ROWID tables work normally.
_DDL_CURSORS_INDEX: str = """
CREATE INDEX IF NOT EXISTS idx_cursors_checkpoint_at
    ON cursors (checkpoint_at);
"""

# ── DML ───────────────────────────────────────────────────────────────────────
#
# INSERT OR REPLACE is intentionally NOT used here.  INSERT OR REPLACE
# is implemented as DELETE + INSERT — it deletes the existing row and
# inserts a new one.  This has two problems:
#
#   1. It does unnecessary B-tree work: one delete (with rebalance) and
#      one insert (with rebalance) instead of one in-place update.
#
#   2. On WITHOUT ROWID tables the behavior is identical to standard tables,
#      but the DELETE+INSERT path is still wasteful — especially since every
#      checkpoint is an update of an existing row (after the first write),
#      never a first-time insert of a row that competes with another row.
#
# The correct primitive is INSERT ... ON CONFLICT DO UPDATE (available since
# SQLite 3.24.0, released June 2018).  This is a true upsert: if the row
# does not exist it inserts; if it does exist it updates in place.  No delete.
# No rowid reset.  No extra B-tree rebalance.
#
# The UPDATE clause explicitly names every non-key column to update.
# This is intentional — it makes the intent clear and ensures that if the
# schema adds a column in the future, the DML must be updated explicitly
# rather than silently ignoring the new column.

_DML_UPSERT: str = """
INSERT INTO cursors (manifest_id, position, url, checkpoint_at, total_urls)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT (manifest_id) DO UPDATE SET
    position      = excluded.position,
    url           = excluded.url,
    checkpoint_at = excluded.checkpoint_at,
    total_urls    = excluded.total_urls
"""

# Pure UPDATE path — used when the caller has already confirmed the row
# exists (i.e., all checkpoints after the first one for a given manifest).
# Slightly cheaper than the upsert path because it skips the conflict check.
# _CheckpointWriter uses this after the first successful checkpoint for each
# manifest_id, tracked via _writer._known_manifests.
_DML_UPDATE: str = """
UPDATE cursors
SET position = ?, url = ?, checkpoint_at = ?, total_urls = ?
WHERE manifest_id = ?
"""

_DML_DELETE: str = """
DELETE FROM cursors WHERE manifest_id = ?
"""

_DML_DELETE_STALE: str = """
DELETE FROM cursors WHERE checkpoint_at < ?
"""

# ── DQL ───────────────────────────────────────────────────────────────────────

_DQL_POSITION: str = """
SELECT position FROM cursors WHERE manifest_id = ?
"""

# Fetches position AND total_urls together — used by advance_if_ahead()
# to avoid a second round-trip when it needs both values.
_DQL_POSITION_AND_TOTAL: str = """
SELECT position, total_urls FROM cursors WHERE manifest_id = ?
"""

_DQL_RECORD: str = """
SELECT manifest_id, position, url, checkpoint_at, total_urls
FROM cursors
WHERE manifest_id = ?
"""

_DQL_ALL: str = """
SELECT manifest_id, position, url, checkpoint_at, total_urls
FROM cursors
ORDER BY checkpoint_at DESC
"""

# Returns all rows — used by GracefulShutdownCheckpointer.flush() to read
# current positions for all manifests before the batch write, allowing
# advance_if_ahead semantics without N individual round-trips.
_DQL_ALL_POSITIONS: str = """
SELECT manifest_id, position FROM cursors
"""

_DQL_STALE: str = """
SELECT manifest_id, position, url, checkpoint_at, total_urls
FROM cursors
WHERE checkpoint_at < ?
ORDER BY checkpoint_at ASC
"""

_DQL_COUNT: str = """
SELECT COUNT(*) FROM cursors
"""

_DQL_EXISTS: str = """
SELECT 1 FROM cursors WHERE manifest_id = ? LIMIT 1
"""


# ─────────────────────────────────────────────────────────────────────────────
# DATA CONTRACTS
# Frozen dataclasses that represent the data this module produces and consumes.
# None of these are imported from contracts.py — they are local to this module.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CursorRecord:
    """
    A single row in the cursors table.  One record per active manifest.

    This is the complete persistent state of a crawl cursor.  On restart
    after a crash, the fetcher calls get_position(manifest_id) — which
    returns this record's position field — and the frontier skips to that
    index.

    Fields
    ──────
    manifest_id    UUID4 string identifying the CrawlManifest being tracked.
                   This is the primary key.  Two manifests for the same domain
                   at different times have different manifest_ids.

    position       Index (0-based) into manifest.urls[] of the last URL that
                   was fully processed before this checkpoint was written.
                   On restart, the next URL to process is position + 1.

    url            The URL at manifest.urls[position].  Stored for operator
                   inspection and audit — the frontier is the canonical source
                   of truth for URLs.  This field is not used for resume logic.

    checkpoint_at  Unix timestamp (time.time()) when this row was written.
                   Used to detect stale cursors for dead manifests.

    total_urls     The len(manifest.urls) at the time the manifest was loaded.
                   Used to compute progress_fraction without querying the
                   frontier.

    Properties
    ──────────
    progress_fraction   Float 0.0 – 1.0.  Safe: returns 0.0 if total_urls == 0.
    remaining_urls      Integer count of URLs not yet processed.
    age_seconds         How many seconds ago this checkpoint was written.
    is_complete         True if position + 1 == total_urls (last URL done).
    """

    manifest_id: str
    position: int
    url: str
    checkpoint_at: float
    total_urls: int

    @property
    def progress_fraction(self) -> float:
        """Fraction of manifest complete.  0.0 if total_urls is zero."""
        if self.total_urls == 0:
            return 0.0
        # position is the index of the last COMPLETED url; +1 gives count done.
        return min(1.0, (self.position + 1) / self.total_urls)

    @property
    def remaining_urls(self) -> int:
        """Count of URLs not yet processed."""
        return max(0, self.total_urls - (self.position + 1))

    @property
    def age_seconds(self) -> float:
        """Seconds elapsed since this checkpoint was written."""
        return time.time() - self.checkpoint_at

    @property
    def is_complete(self) -> bool:
        """True if the last URL in the manifest has been processed."""
        if self.total_urls == 0:
            return False
        return self.position >= self.total_urls - 1

    def to_dict(self) -> Dict[str, object]:
        """Flat dict suitable for structured logging and monitoring export."""
        return {
            "manifest_id": self.manifest_id,
            "position": self.position,
            "url": self.url,
            "checkpoint_at": self.checkpoint_at,
            "total_urls": self.total_urls,
            "progress_fraction": round(self.progress_fraction, 4),
            "remaining_urls": self.remaining_urls,
            "age_seconds": round(self.age_seconds, 1),
            "is_complete": self.is_complete,
        }

    def __repr__(self) -> str:
        pct = f"{self.progress_fraction * 100:.1f}%"
        return (
            f"CursorRecord("
            f"manifest_id={self.manifest_id[:8]}..., "
            f"position={self.position}/{self.total_urls}, "
            f"progress={pct}, "
            f"age={self.age_seconds:.0f}s)"
        )


@dataclass(frozen=True)
class CheckpointResult:
    """
    Return value from CrawlCursor.checkpoint().

    Carries metadata about the write operation so callers can log
    checkpoint performance without instrumenting the cursor internals.

    Fields
    ──────
    manifest_id     Which manifest was checkpointed.
    position        The position written.
    latency_ms      Wall time of the SQLite write in milliseconds.
    was_write       True if the row was actually written (i.e. position moved
                    forward or no prior row existed).  False if the checkpoint
                    was skipped because position has not advanced.
    checkpoint_at   Unix timestamp of the write (time.time()).
    """

    manifest_id: str
    position: int
    latency_ms: float
    was_write: bool
    checkpoint_at: float

    def to_dict(self) -> Dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "position": self.position,
            "latency_ms": round(self.latency_ms, 3),
            "was_write": self.was_write,
            "checkpoint_at": self.checkpoint_at,
        }


@dataclass(frozen=True)
class CursorHealth:
    """
    Point-in-time health status of the cursor subsystem.

    Returned by CrawlCursor.health().  Consumed by monitoring agents.
    All fields are safe for serialization to JSON/dict.

    Fields
    ──────
    db_path             Absolute path to the SQLite file.
    db_exists           Whether the file exists on disk.
    wal_mode_active     Whether PRAGMA journal_mode returned 'wal'.
    integrity_ok        Whether PRAGMA integrity_check returned 'ok'.
    active_cursor_count Count of rows currently in the cursors table.
    stale_cursor_count  Count of rows older than CURSOR_STALE_THRESHOLD_SECONDS.
    db_size_bytes       Size of the SQLite main file in bytes.
    wal_size_bytes      Size of the WAL file in bytes (0 if not present).
    initialized         Whether initialize() has been called successfully.
    error               Non-None if the health check itself encountered an error.
    """

    db_path: str
    db_exists: bool
    wal_mode_active: bool
    integrity_ok: bool
    active_cursor_count: int
    stale_cursor_count: int
    db_size_bytes: int
    wal_size_bytes: int
    initialized: bool
    error: Optional[str]

    @property
    def is_healthy(self) -> bool:
        """True iff all critical health indicators are nominal."""
        return (
            self.db_exists
            and self.wal_mode_active
            and self.integrity_ok
            and self.initialized
            and self.error is None
        )

    def to_dict(self) -> Dict[str, object]:
        return {
            "db_path": self.db_path,
            "db_exists": self.db_exists,
            "wal_mode_active": self.wal_mode_active,
            "integrity_ok": self.integrity_ok,
            "active_cursor_count": self.active_cursor_count,
            "stale_cursor_count": self.stale_cursor_count,
            "db_size_bytes": self.db_size_bytes,
            "wal_size_bytes": self.wal_size_bytes,
            "initialized": self.initialized,
            "is_healthy": self.is_healthy,
            "error": self.error,
        }


@dataclass
class _CheckpointSample:
    """
    One data point in the checkpoint latency ring buffer.
    Not frozen — instances are mutated (position updated) during reuse.
    Internal use only.
    """

    position: int
    latency_ms: float
    sampled_at: float  # time.monotonic()


@dataclass
class _SessionState:
    """
    In-memory state for one manifest's crawl session.

    Tracks checkpoint velocity, latency distribution, and stall detection
    without any I/O.  Garbage-collected when the session ends (clear() is
    called).

    Not thread-safe — asyncio single-threaded model is assumed.
    """

    manifest_id: str
    session_start: float  # time.monotonic()
    last_checkpoint_mono: float  # time.monotonic() of last checkpoint write
    last_position: int
    checkpoint_count: int
    failure_count: int
    total_urls: int
    samples: List[_CheckpointSample] = field(default_factory=list)

    def record_checkpoint(
        self,
        position: int,
        latency_ms: float,
    ) -> None:
        """Record a completed checkpoint.  Trims sample buffer to cap."""
        now = time.monotonic()
        self.last_checkpoint_mono = now
        self.last_position = position
        self.checkpoint_count += 1
        sample = _CheckpointSample(
            position=position,
            latency_ms=latency_ms,
            sampled_at=now,
        )
        self.samples.append(sample)
        # Trim to cap — oldest samples fall off the left.
        if len(self.samples) > CURSOR_LATENCY_SAMPLE_CAP:
            self.samples = self.samples[-CURSOR_LATENCY_SAMPLE_CAP :]

    def record_failure(self) -> None:
        """Increment failure counter (checkpoint raised CursorError)."""
        self.failure_count += 1

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.session_start

    @property
    def velocity_urls_per_second(self) -> float:
        """
        Rolling checkpoint velocity.

        Computed as the rate of position advancement over the most recent
        set of samples.  Returns 0.0 if fewer than two samples exist.
        """
        if len(self.samples) < 2:
            return 0.0
        oldest = self.samples[0]
        newest = self.samples[-1]
        dt = newest.sampled_at - oldest.sampled_at
        if dt <= 0:
            return 0.0
        dp = newest.position - oldest.position
        # dp should always be positive (positions advance) but guard anyway.
        return max(0.0, dp / dt)

    @property
    def is_stalled(self) -> bool:
        """
        True if velocity has dropped below the stall threshold.

        A stalled cursor indicates a manifest that is no longer processing
        URLs — most likely because the fetcher died without calling clear().
        Only meaningful after at least two checkpoints.
        """
        if self.checkpoint_count < 2:
            return False
        return self.velocity_urls_per_second < CURSOR_STALL_VELOCITY_THRESHOLD

    @property
    def avg_checkpoint_latency_ms(self) -> float:
        """Average checkpoint write latency across retained samples."""
        if not self.samples:
            return 0.0
        return sum(s.latency_ms for s in self.samples) / len(self.samples)

    @property
    def p99_checkpoint_latency_ms(self) -> float:
        """
        99th-percentile checkpoint write latency.

        Returns 0.0 if fewer than 10 samples are available — p99 is
        not meaningful on a tiny sample.
        """
        if len(self.samples) < 10:
            return 0.0
        sorted_latencies = sorted(s.latency_ms for s in self.samples)
        idx = max(0, int(len(sorted_latencies) * 0.99) - 1)
        return sorted_latencies[idx]

    def estimated_remaining_seconds(self) -> Optional[float]:
        """
        ETA to manifest completion based on current velocity.

        Returns None if velocity is zero or total_urls is not set.
        """
        v = self.velocity_urls_per_second
        if v <= 0 or self.total_urls <= 0 or not self.samples:
            return None
        remaining = self.total_urls - self.samples[-1].position - 1
        if remaining <= 0:
            return 0.0
        return remaining / v

    def to_summary_dict(self) -> Dict[str, object]:
        """Export session state as a flat dict for monitoring."""
        return {
            "manifest_id": self.manifest_id,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "checkpoint_count": self.checkpoint_count,
            "failure_count": self.failure_count,
            "last_position": self.last_position,
            "total_urls": self.total_urls,
            "velocity_urls_per_second": round(self.velocity_urls_per_second, 4),
            "is_stalled": self.is_stalled,
            "avg_checkpoint_latency_ms": round(self.avg_checkpoint_latency_ms, 3),
            "p99_checkpoint_latency_ms": round(self.p99_checkpoint_latency_ms, 3),
            "estimated_remaining_seconds": self.estimated_remaining_seconds(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA MANAGER
# Owns all DDL and PRAGMA configuration.  CrawlCursor delegates schema
# creation and verification here.  Separated so tests can verify schema state
# without constructing a full CrawlCursor.
# ─────────────────────────────────────────────────────────────────────────────


class _SchemaManager:
    """
    Owns DDL execution, PRAGMA configuration, and schema version management.

    All methods accept an open aiosqlite.Connection.  They do not own the
    connection lifetime — that is the responsibility of CrawlCursor.

    Public methods are all async and may raise aiosqlite.Error (not
    CursorError — the caller converts if needed).
    """

    @staticmethod
    async def initialize(db: aiosqlite.Connection) -> None:
        """
        Apply the full PRAGMA set, create schema, and verify.

        Must be called once on every new connection before any read or write.
        Idempotent: calling twice on the same connection is safe.

        Steps:
          1. Set pragmas that affect the connection session (page_size,
             journal_mode, busy_timeout, synchronous, wal_autocheckpoint).
          2. Create tables and indexes (CREATE TABLE IF NOT EXISTS).
          3. Verify journal mode is 'wal'.
          4. Verify schema version and write it if this is a new database.
        """
        await _SchemaManager._set_pragmas(db)
        await _SchemaManager._create_tables(db)
        await _SchemaManager._verify_wal(db)
        await _SchemaManager._set_schema_version(db)

    @staticmethod
    async def _set_pragmas(db: aiosqlite.Connection) -> None:
        """
        Apply connection-scoped PRAGMAs.

        page_size must be set before any tables are created.  After the
        first write, it is locked in by the database header and this pragma
        is silently ignored on re-open.

        journal_mode=WAL is persistent — it survives connection close and
        restart.  Setting it on every open is harmless and ensures the mode
        is enforced even if the file was somehow opened in another mode.

        busy_timeout makes BEGIN IMMEDIATE retry for up to
        CURSOR_BUSY_TIMEOUT_MS milliseconds before failing with
        SQLITE_BUSY.  This protects against brief lock contention from
        monitoring connections.

        synchronous=NORMAL is safe in WAL mode — WAL provides a crash-
        consistent barrier at each COMMIT without requiring a full fsync on
        the main file.  FULL would add an extra fsync per commit; the WAL
        design makes that unnecessary.

        wal_autocheckpoint limits WAL file growth.
        """
        statements: List[str] = [
            f"PRAGMA page_size = {CURSOR_PAGE_SIZE}",
            "PRAGMA journal_mode = WAL",
            f"PRAGMA busy_timeout = {CURSOR_BUSY_TIMEOUT_MS}",
            "PRAGMA synchronous = NORMAL",
            f"PRAGMA wal_autocheckpoint = {CURSOR_WAL_AUTOCHECKPOINT}",
            "PRAGMA temp_store = MEMORY",
            "PRAGMA cache_size = -2048",  # 2 MiB page cache
        ]
        for stmt in statements:
            await db.execute(stmt)

    @staticmethod
    async def _create_tables(db: aiosqlite.Connection) -> None:
        """
        Create the cursors table and supporting index.

        Both statements use IF NOT EXISTS — this is a no-op on databases that
        already have the schema.  No migration logic is needed for v1; future
        schema versions will require an explicit upgrade path here.
        """
        await db.execute(_DDL_CURSORS_TABLE)
        await db.execute(_DDL_CURSORS_INDEX)
        await db.commit()

    @staticmethod
    async def _verify_wal(db: aiosqlite.Connection) -> None:
        """
        Assert that WAL mode is active.  Raises RuntimeError if not.

        This assertion is critical: if WAL mode is not active, our BEGIN
        IMMEDIATE transaction semantics are subtly different and crash
        recovery cannot be guaranteed.
        """
        cursor = await db.execute("PRAGMA journal_mode")
        row = await cursor.fetchone()
        await cursor.close()
        if row is None or row[0].lower() != "wal":
            actual = row[0] if row else "unknown"
            raise RuntimeError(
                f"crawl_cursor: expected WAL journal mode, got '{actual}'. "
                "WAL mode is required for crash-safe cursor checkpointing. "
                "The database may have been opened by a legacy SQLite version "
                "that does not support WAL, or PRAGMA journal_mode=WAL was "
                "rejected.  Cannot continue."
            )

    @staticmethod
    async def _set_schema_version(db: aiosqlite.Connection) -> None:
        """
        Check and conditionally write the schema version.

        If user_version is 0 (new database), write CURSOR_SCHEMA_VERSION.
        If user_version matches CURSOR_SCHEMA_VERSION, no action.
        If user_version is non-zero and does not match, raise RuntimeError.
        A version mismatch indicates the database was created by a different
        release of crawl_cursor.py and may have an incompatible schema.
        """
        cursor = await db.execute("PRAGMA user_version")
        row = await cursor.fetchone()
        await cursor.close()
        existing_version: int = row[0] if row else 0

        if existing_version == 0:
            # New database — write our version.
            await db.execute(
                f"PRAGMA user_version = {CURSOR_SCHEMA_VERSION}"
            )
            _log.debug(
                "crawl_cursor: wrote schema version %d to new database",
                CURSOR_SCHEMA_VERSION,
            )
        elif existing_version != CURSOR_SCHEMA_VERSION:
            raise RuntimeError(
                f"crawl_cursor: schema version mismatch. "
                f"Expected {CURSOR_SCHEMA_VERSION}, found {existing_version}. "
                f"The cursor database at {CURSOR_DB_PATH} was created by a "
                f"different version of crawl_cursor.py.  "
                f"Upgrade or delete the database before proceeding."
            )

    @staticmethod
    async def verify_integrity(db: aiosqlite.Connection) -> bool:
        """
        Run PRAGMA integrity_check and return True iff the result is 'ok'.

        This is a read-intensive operation — it scans every page of the
        database.  Call it at initialize() and on operator request only,
        never in the hot path.

        Returns False (does not raise) if any integrity issue is found, so
        callers can log and continue rather than crashing.
        """
        try:
            cursor = await db.execute("PRAGMA integrity_check(16)")
            rows = await cursor.fetchall()
            await cursor.close()
            # integrity_check returns one row per error, or a single row
            # containing 'ok' if everything is clean.
            if len(rows) == 1 and rows[0][0].lower() == "ok":
                return True
            issues = [row[0] for row in rows]
            _log.error(
                "crawl_cursor: integrity_check returned %d issue(s): %s",
                len(issues),
                issues[:5],
            )
            return False
        except aiosqlite.Error as exc:
            _log.error(
                "crawl_cursor: integrity_check failed with SQLite error: %s", exc
            )
            return False

    @staticmethod
    async def wal_checkpoint(
        db: aiosqlite.Connection,
        mode: str = "PASSIVE",
    ) -> Tuple[int, int]:
        """
        Issue a WAL checkpoint and return (wal_log_size, frames_checkpointed).

        mode: 'PASSIVE' (non-blocking), 'FULL', or 'TRUNCATE'.
            PASSIVE — checkpoint whatever is possible without blocking.
            FULL    — wait for readers to finish, then checkpoint.
            TRUNCATE — full checkpoint + truncate the WAL file to zero bytes.

        Returns (wal_log_size, frames_checkpointed) from the PRAGMA response.
        Returns (0, 0) on any error (does not raise).

        TRUNCATE is used by the forced-compaction path after
        CURSOR_WAL_FORCE_INTERVAL checkpoints.  PASSIVE is used at close().
        """
        try:
            cursor = await db.execute(f"PRAGMA wal_checkpoint({mode})")
            row = await cursor.fetchone()
            await cursor.close()
            if row and len(row) >= 3:
                # row is (busy, log, checkpointed)
                return (int(row[1]), int(row[2])) # noqa
            return (0, 0) # noqa
        except aiosqlite.Error as exc:
            _log.warning(
                "crawl_cursor: wal_checkpoint(%s) failed: %s",
                mode,
                exc,
            )
            return (0, 0) # noqa


# ─────────────────────────────────────────────────────────────────────────────
# CHECKPOINT WRITER
# Encapsulates the write path: BEGIN IMMEDIATE → INSERT OR REPLACE → COMMIT.
# Separated from CrawlCursor so the atomicity guarantee can be tested in
# isolation and the write path can be profiled independently of the read path.
# ─────────────────────────────────────────────────────────────────────────────


class _CheckpointWriter:
    """
    Performs atomic checkpoint writes against an open aiosqlite connection.

    The connection must have been configured with isolation_level=None (manual
    transaction management) so Python's sqlite3 layer does not auto-begin a
    DEFERRED transaction that would conflict with our explicit BEGIN IMMEDIATE.

    Every write operation:
      1. Issues BEGIN IMMEDIATE — acquires the WAL write lock immediately.
         If another writer holds the lock, SQLite retries for CURSOR_BUSY_TIMEOUT_MS
         before raising OperationalError("database is locked").
      2. Executes the DML statement.
      3. Issues COMMIT.

    On any OperationalError, the writer attempts ROLLBACK and re-raises as
    CursorError.  The caller (CrawlCursor.checkpoint) catches CursorError,
    logs at WARNING, and continues the manifest.  A missed checkpoint is
    acceptable — the worst-case exposure is CURSOR_CHECKPOINT_INTERVAL URLs.

    No state is retained between write calls.  Instances are stateless.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def write(
        self,
        manifest_id: str,
        position: int,
        url: str,
        total_urls: int,
        checkpoint_at: float,
    ) -> float:
        """
        Write one cursor row atomically.  Returns write latency in ms.

        Uses INSERT OR REPLACE so the same manifest_id can be checkpointed
        repeatedly without constraint errors — each write unconditionally
        replaces the prior row.

        Raises CursorError on SQLite failure.  Never raises any other
        exception type.

        Parameters
        ──────────
        manifest_id     UUID4 string identifying the manifest.
        position        0-based index of the last completed URL.
        url             Human-readable URL at position (audit only).
        total_urls      Total URL count in manifest (for progress).
        checkpoint_at   Unix timestamp of the checkpoint (time.time()).

        Returns
        ───────
        float   Wall-clock duration of the write in milliseconds.
        """
        t0 = time.monotonic()
        try:
            await self._db.execute("BEGIN IMMEDIATE")
            await self._db.execute(
                _DML_UPSERT,
                (manifest_id, position, url, checkpoint_at, total_urls),
            )
            await self._db.execute("COMMIT")
            latency_ms = (time.monotonic() - t0) * 1000.0
            _log.debug(
                "cursor.checkpoint manifest=%s pos=%d/%d latency=%.2fms",
                manifest_id[:8],
                position,
                total_urls,
                latency_ms,
            )
            return latency_ms
        except aiosqlite.OperationalError as exc:
            await self._safe_rollback(manifest_id, position, "checkpoint")
            raise CursorError(
                manifest_id=manifest_id,
                position=position,
                operation="checkpoint",
                db_error=str(exc),
            ) from exc
        except aiosqlite.Error as exc:
            await self._safe_rollback(manifest_id, position, "checkpoint")
            raise CursorError(
                manifest_id=manifest_id,
                position=position,
                operation="checkpoint",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def clear(self, manifest_id: str) -> None:
        """
        Remove the cursor row for a completed manifest.

        Called by fetcher.py when ManifestCompleteEvent is emitted.  After
        this call, get_position(manifest_id) returns 0 — the same value as
        for an unknown manifest.  This is correct: the manifest is done and
        will not restart; if a new manifest with the same domain is issued,
        it gets a new manifest_id and starts from position 0.

        Raises CursorError on SQLite failure.
        """
        try:
            await self._db.execute("BEGIN IMMEDIATE")
            await self._db.execute(_DML_DELETE, (manifest_id,))
            await self._db.execute("COMMIT")
            _log.debug(
                "cursor.clear manifest=%s", manifest_id[:8]
            )
        except aiosqlite.OperationalError as exc:
            await self._safe_rollback(manifest_id, 0, "clear")
            raise CursorError(
                manifest_id=manifest_id,
                position=0,
                operation="clear",
                db_error=str(exc),
            ) from exc
        except aiosqlite.Error as exc:
            await self._safe_rollback(manifest_id, 0, "clear")
            raise CursorError(
                manifest_id=manifest_id,
                position=0,
                operation="clear",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def clear_stale(self, older_than_ts: float) -> int:
        """
        Delete all cursor rows with checkpoint_at < older_than_ts.

        Returns the number of rows deleted.  Raises CursorError on failure.

        This is a bulk GC operation — it removes cursors for manifests that
        were abandoned without a clean clear() call.  Callers should pass
        (time.time() - CURSOR_STALE_THRESHOLD_SECONDS) as older_than_ts.
        """
        try:
            await self._db.execute("BEGIN IMMEDIATE")
            cursor = await self._db.execute(
                _DML_DELETE_STALE, (older_than_ts,)
            )
            deleted = cursor.rowcount
            await self._db.execute("COMMIT")
            if deleted > 0:
                _log.info(
                    "cursor.clear_stale: removed %d stale cursor(s) "
                    "older than %.0f seconds",
                    deleted,
                    time.time() - older_than_ts,
                )
            return deleted if deleted is not None else 0
        except aiosqlite.Error as exc:
            await self._safe_rollback("__stale__", 0, "clear_stale")
            raise CursorError(
                manifest_id="__stale__",
                position=0,
                operation="clear_stale",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def _safe_rollback(
        self, manifest_id: str, position: int, operation: str
    ) -> None:
        """
        Attempt ROLLBACK after a failed write.

        Never raises.  Logs on failure.  A failed rollback is unusual —
        if the connection is broken, SQLite will auto-rollback on close.
        """
        try:
            await self._db.execute("ROLLBACK")
        except aiosqlite.Error as rb_exc:
            _log.warning(
                "cursor._safe_rollback: ROLLBACK failed after %s error "
                "on manifest=%s pos=%d: %s",
                operation,
                manifest_id[:8],
                position,
                rb_exc,
            )


# ─────────────────────────────────────────────────────────────────────────────
# CURSOR READER
# Read-only queries against the cursors table.  All reads are autocommit
# (no explicit transaction needed for reads in WAL mode).
# ─────────────────────────────────────────────────────────────────────────────


class _CursorReader:
    """
    Read-only access to the cursors table.

    In WAL mode, reads never block concurrent writers.  All methods issue
    plain SELECT queries without an explicit transaction — SQLite provides
    snapshot isolation per connection in WAL mode, so each query sees a
    consistent snapshot of the database at the time the statement is prepared.

    Methods never raise CursorError for missing rows.  A missing row is
    a valid state — it means the manifest has not been checkpointed yet.
    Methods raise CursorError only for actual SQLite failures.
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def get_position(self, manifest_id: str) -> int:
        """
        Return the last checkpointed position for manifest_id.

        Returns 0 if no cursor row exists for this manifest_id.  This is
        the correct start-from-beginning value — the caller does not need to
        distinguish 'never checkpointed' from 'checkpointed at position 0'.

        Never raises for a missing row.  Raises CursorError for SQLite errors.
        """
        try:
            cursor = await self._db.execute(_DQL_POSITION, (manifest_id,))
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return 0
            return int(row[0])
        except aiosqlite.Error as exc:
            raise CursorError(
                manifest_id=manifest_id,
                position=0,
                operation="get_position",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def get_record(self, manifest_id: str) -> Optional[CursorRecord]:
        """
        Return the full CursorRecord for manifest_id, or None if not found.

        None does not indicate an error — it means the manifest has not been
        checkpointed.  Raises CursorError for SQLite errors only.
        """
        try:
            cursor = await self._db.execute(_DQL_RECORD, (manifest_id,))
            row = await cursor.fetchone()
            await cursor.close()
            if row is None:
                return None
            return _row_to_record(row)
        except aiosqlite.Error as exc:
            raise CursorError(
                manifest_id=manifest_id,
                position=0,
                operation="get_record",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def get_all(self) -> List[CursorRecord]:
        """
        Return all cursor rows, ordered by checkpoint_at DESC (most recent first).

        Returns an empty list if the table is empty.
        Raises CursorError on SQLite failure.
        """
        try:
            cursor = await self._db.execute(_DQL_ALL)
            rows = await cursor.fetchall()
            await cursor.close()
            return [_row_to_record(r) for r in rows]
        except aiosqlite.Error as exc:
            raise CursorError(
                manifest_id="__all__",
                position=0,
                operation="get_all",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def get_stale(self, older_than_ts: float) -> List[CursorRecord]:
        """
        Return cursor rows with checkpoint_at < older_than_ts.

        Ordered by checkpoint_at ASC (oldest first).  Returns empty list
        if none exist.  Raises CursorError on failure.
        """
        try:
            cursor = await self._db.execute(_DQL_STALE, (older_than_ts,))
            rows = await cursor.fetchall()
            await cursor.close()
            return [_row_to_record(r) for r in rows]
        except aiosqlite.Error as exc:
            raise CursorError(
                manifest_id="__stale__",
                position=0,
                operation="get_stale",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def exists(self, manifest_id: str) -> bool:
        """
        Return True if a cursor row exists for manifest_id.

        Cheaper than get_record() for boolean existence checks.
        """
        try:
            cursor = await self._db.execute(_DQL_EXISTS, (manifest_id,))
            row = await cursor.fetchone()
            await cursor.close()
            return row is not None
        except aiosqlite.Error as exc:
            raise CursorError(
                manifest_id=manifest_id,
                position=0,
                operation="exists",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def count(self) -> int:
        """Return the total number of cursor rows in the table."""
        try:
            cursor = await self._db.execute(_DQL_COUNT)
            row = await cursor.fetchone()
            await cursor.close()
            return int(row[0]) if row else 0
        except aiosqlite.Error as exc:
            raise CursorError(
                manifest_id="__count__",
                position=0,
                operation="count",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc


# ─────────────────────────────────────────────────────────────────────────────
# ROW FACTORY
# Converts a raw sqlite3 row tuple into a CursorRecord.
# Used by both _CursorReader and any other code that reads from the DB.
# ─────────────────────────────────────────────────────────────────────────────


def _row_to_record(row: Sequence) -> CursorRecord:
    """
    Convert a sqlite3 row tuple to a CursorRecord.

    Column order matches _DQL_RECORD and _DQL_ALL:
      0: manifest_id  TEXT
      1: position     INTEGER
      2: url          TEXT
      3: checkpoint_at REAL
      4: total_urls   INTEGER
    """
    return CursorRecord(
        manifest_id=str(row[0]),
        position=int(row[1]),
        url=str(row[2]),
        checkpoint_at=float(row[3]),
        total_urls=int(row[4]),
    )


# ─────────────────────────────────────────────────────────────────────────────
# CRAWL CURSOR — PUBLIC API
# The single public class this module exports.  All other classes above are
# implementation details.
# ─────────────────────────────────────────────────────────────────────────────


class CrawlCursor:
    """
    Atomic position checkpoint for the crawl frontier.

    One CrawlCursor instance manages the cursor database for all concurrent
    manifests in a process.  There should be exactly one CrawlCursor instance
    per process.

    Lifecycle
    ─────────
    1. Instantiate with desired db_path (default: CURSOR_DB_PATH).
    2. Call await cursor.initialize() before any other method.
    3. Call checkpoint() every CURSOR_CHECKPOINT_INTERVAL URLs.
    4. Call clear() when a manifest completes.
    5. Call close() at process shutdown.

    Thread safety
    ─────────────
    Not thread-safe.  Designed for single-threaded asyncio use.  The
    underlying aiosqlite connection uses a worker thread internally — asyncio
    serializes all calls from the event loop, so no additional locking is
    needed in the caller.

    Crash safety
    ────────────
    checkpoint() uses BEGIN IMMEDIATE → INSERT OR REPLACE → COMMIT.  This
    guarantees that either the full row is on disk or it is not — no partial
    states.  On restart, get_position() returns the position from the last
    successful COMMIT.

    Error handling
    ──────────────
    All methods that can fail raise CursorError with structured context.
    No other exception type escapes this class except RuntimeError from
    initialize() for fatal configuration problems (wrong journal mode,
    schema version mismatch).  Callers should catch CursorError and continue
    — a missed checkpoint is recoverable.

    Usage example
    ─────────────
    ::

        cursor = CrawlCursor()
        await cursor.initialize()

        # On first start: get_position returns 0
        start = await cursor.get_position(manifest_id)

        # Every 100 URLs:
        if (url_index % CURSOR_CHECKPOINT_INTERVAL) == 0:
            await cursor.checkpoint(manifest_id, url_index, url, total)

        # On manifest completion:
        await cursor.clear(manifest_id)

        # At process shutdown:
        await cursor.close()
    """

    def __init__(self, db_path: Path = CURSOR_DB_PATH) -> None:
        """
        Construct a CrawlCursor targeting db_path.

        Does not open the database.  Call initialize() before use.

        Parameters
        ──────────
        db_path     Path to the SQLite file.  The parent directory must exist.
                    The file will be created on first initialize() if absent.
        """
        self._db_path: Path = db_path
        self._db: Optional[aiosqlite.Connection] = None
        self._writer: Optional[_CheckpointWriter] = None
        self._reader: Optional[_CursorReader] = None
        self._schema: _SchemaManager = _SchemaManager()
        self._initialized: bool = False
        self._closed: bool = False

        # In-memory session state keyed by manifest_id.
        # Entries are created on first checkpoint() call and removed on clear().
        self._sessions: Dict[str, _SessionState] = {}

        # Cumulative checkpoint count across all manifests — used to decide
        # when to issue a forced WAL TRUNCATE checkpoint.
        self._total_checkpoint_count: int = 0

        _log.debug(
            "CrawlCursor created with db_path=%s", db_path
        )

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self, db_path: Optional[Path] = None) -> None:
        """
        Open the database, apply PRAGMAs, create schema if absent.

        Must be called before any other method.  Calling twice on the same
        instance has no effect (idempotent).

        If db_path is provided, it overrides the path given at construction.
        This parameter exists for compatibility with the build spec signature:
            ``await cursor.initialize(db_path=Path("store/crawl_cursor.db"))``

        Behavior on first call:
          - If the file does not exist: creates it, writes schema, sets WAL mode.
          - If the file exists: opens it, verifies WAL mode and schema version.

        Raises
        ──────
        RuntimeError    WAL mode not active after open, or schema version mismatch.
        aiosqlite.Error Underlying SQLite error during open or pragma setup.
        """
        if self._initialized:
            return

        if db_path is not None:
            self._db_path = db_path

        # Ensure the parent directory exists.  We do not create it — the host
        # environment (Docker / systemd) is responsible for the store/ directory.
        if not self._db_path.parent.exists():
            raise RuntimeError(
                f"crawl_cursor: parent directory for {self._db_path} does not "
                f"exist.  Create 'store/' before calling initialize().  "
                f"The crawler's store directory must be provisioned by the "
                f"host environment before the process starts."
            )

        _log.info(
            "crawl_cursor.initialize: opening %s (exists=%s)",
            self._db_path,
            self._db_path.exists(),
        )

        # Open with isolation_level=None (autocommit) so we can issue
        # BEGIN IMMEDIATE manually without conflicting with Python's sqlite3
        # auto-BEGIN behavior.
        self._db = await aiosqlite.connect(
            str(self._db_path),
            isolation_level=None,
        )

        # Apply PRAGMAs and create schema.
        await _SchemaManager.initialize(self._db)

        # Wire up read/write helpers.
        self._writer = _CheckpointWriter(self._db)
        self._reader = _CursorReader(self._db)

        self._initialized = True

        existing = await self._reader.count()
        _log.info(
            "crawl_cursor.initialize: ready.  %d active cursor(s) found.",
            existing,
        )

    async def close(self) -> None:
        """
        Flush WAL and close the database connection.

        Idempotent — calling multiple times is safe.  After close(), the
        instance cannot be reused without calling initialize() again.

        Issues a PASSIVE WAL checkpoint before closing to compact the WAL
        back to the main database file.  PASSIVE mode does not block — if any
        readers still hold a WAL snapshot, the checkpoint progresses as far as
        it can.  The remaining WAL pages will be picked up by the auto-
        checkpoint mechanism on next open.
        """
        if self._closed or not self._initialized or self._db is None:
            return

        _log.info("crawl_cursor.close: flushing WAL and closing.")

        try:
            # Passive checkpoint — compact WAL without blocking readers.
            wal_log, checkpointed = await _SchemaManager.wal_checkpoint(
                self._db, mode="PASSIVE"
            )
            _log.debug(
                "crawl_cursor.close: wal_checkpoint PASSIVE: "
                "log_size=%d checkpointed=%d",
                wal_log,
                checkpointed,
            )
        except Exception as exc:
            _log.warning(
                "crawl_cursor.close: wal_checkpoint failed (non-fatal): %s", exc
            )

        try:
            await self._db.close()
        except Exception as exc:
            _log.warning(
                "crawl_cursor.close: db.close() raised (non-fatal): %s", exc
            )

        self._closed = True
        self._initialized = False
        self._db = None
        self._writer = None
        self._reader = None

    # ── Core API ──────────────────────────────────────────────────────────────

    async def checkpoint(
        self,
        manifest_id: str,
        position: int,
        url: str,
        total_urls: int,
    ) -> CheckpointResult:
        """
        Atomically write the current crawl position to disk.

        This is the hot-path method — it is called every
        CURSOR_CHECKPOINT_INTERVAL URLs by the fetcher.  It must be fast,
        correct under concurrent process death, and never raise an exception
        that halts the manifest.

        The write is unconditional: if the same position is written twice,
        the INSERT OR REPLACE produces the same row.  Idempotency is a
        property of the INSERT OR REPLACE semantics, not additional logic
        in this method.

        Parameters
        ──────────
        manifest_id     UUID4 string matching the CrawlManifest.manifest_id.
        position        0-based index of the last URL that has been fully
                        processed (fetched, bloom-filtered, frontier-marked).
        url             The URL string at manifest.urls[position].  Stored
                        for operator inspection — not used in resume logic.
        total_urls      Total number of URLs in the manifest.

        Returns
        ───────
        CheckpointResult    Metadata about the write.  latency_ms is the wall
                            time of the SQLite commit in milliseconds.

        Raises
        ──────
        CursorError     On any SQLite error.  The fetcher catches this, logs
                        at WARNING, and continues the manifest.  The worst-case
                        exposure is CURSOR_CHECKPOINT_INTERVAL re-fetched URLs
                        on restart.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("checkpoint")

        t0 = time.monotonic()
        checkpoint_at = time.time()

        latency_ms = await self._writer.write(  # type: ignore[union-attr]
            manifest_id=manifest_id,
            position=position,
            url=url,
            total_urls=total_urls,
            checkpoint_at=checkpoint_at,
        )

        # Update in-memory session state.
        session = self._get_or_create_session(manifest_id, total_urls)
        session.record_checkpoint(position, latency_ms)
        self._total_checkpoint_count += 1

        # Periodic forced WAL compaction.
        if self._total_checkpoint_count % CURSOR_WAL_FORCE_INTERVAL == 0:
            asyncio.ensure_future(self._background_wal_truncate())

        result = CheckpointResult(
            manifest_id=manifest_id,
            position=position,
            latency_ms=latency_ms,
            was_write=True,
            checkpoint_at=checkpoint_at,
        )

        if latency_ms > 100.0:
            _log.warning(
                "cursor.checkpoint: slow write %.1fms for manifest=%s pos=%d",
                latency_ms,
                manifest_id[:8],
                position,
            )

        return result

    async def get_position(self, manifest_id: str) -> int:
        """
        Return the last checkpointed position for manifest_id.

        Returns 0 if no cursor row exists (start from beginning).
        Never raises for a missing manifest — missing is not an error.

        This is called by frontier.resume() on startup to determine where
        to begin yielding URLs.  The common case after a clean shutdown is
        that clear() was called and this returns 0.  The crash-recovery case
        is that checkpoint() was called and this returns the last written
        position.

        Parameters
        ──────────
        manifest_id     UUID4 string identifying the manifest to query.

        Returns
        ───────
        int     0 if never checkpointed; otherwise the last written position.

        Raises
        ──────
        CursorError     On SQLite read failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("get_position")
        return await self._reader.get_position(manifest_id)  # type: ignore[union-attr]

    async def get_record(self, manifest_id: str) -> Optional[CursorRecord]:
        """
        Return the full CursorRecord for manifest_id, or None if not found.

        Provides richer information than get_position() — includes the URL at
        the checkpoint position, the timestamp, and the total_urls count.
        Use get_position() in the hot path; use get_record() for diagnostics
        and monitoring.

        Parameters
        ──────────
        manifest_id     UUID4 identifying the manifest.

        Returns
        ───────
        CursorRecord    If a checkpoint row exists.
        None            If no checkpoint has been written for this manifest.

        Raises
        ──────
        CursorError     On SQLite read failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("get_record")
        return await self._reader.get_record(manifest_id)  # type: ignore[union-attr]

    async def clear(self, manifest_id: str) -> None:
        """
        Remove the cursor row for a completed manifest.

        Called by fetcher.py immediately after emitting ManifestCompleteEvent.
        After this call, get_position(manifest_id) returns 0 — the correct
        behavior for a completed manifest (if it is somehow restarted, it
        starts from the beginning, which is fine since all its URLs are
        already marked done in the frontier).

        Does nothing if no row exists for manifest_id (idempotent).

        Parameters
        ──────────
        manifest_id     UUID4 of the completed manifest.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("clear")

        await self._writer.clear(manifest_id)  # type: ignore[union-attr]

        # Drop the in-memory session if it exists.
        self._sessions.pop(manifest_id, None)

        _log.info(
            "cursor.clear: manifest=%s removed.", manifest_id[:8]
        )

    async def all_active(self) -> List[CursorRecord]:
        """
        Return all cursor rows, ordered by checkpoint_at DESC.

        Active means 'has a row in the cursors table' — this includes both
        currently running manifests and any manifests that crashed without
        calling clear().  Use stale_cursors() to filter for the latter.

        Returns an empty list if the table is empty.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("all_active")
        return await self._reader.get_all()  # type: ignore[union-attr]

    # ── Extended API ──────────────────────────────────────────────────────────

    async def progress_fraction(self, manifest_id: str) -> float:
        """
        Return progress as a float 0.0 – 1.0 for manifest_id.

        Returns 0.0 if no cursor row exists or total_urls is zero.
        Uses the last checkpointed position — does not query the frontier.
        The value may lag up to CURSOR_CHECKPOINT_INTERVAL URLs behind actual
        progress.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        record = await self.get_record(manifest_id)
        if record is None:
            return 0.0
        return record.progress_fraction

    async def is_active(self, manifest_id: str) -> bool:
        """
        Return True if a cursor row exists for manifest_id.

        Cheaper than get_record() for boolean existence checks.
        A True result means the manifest was started and not yet cleared —
        it may be currently running or may have crashed.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("is_active")
        return await self._reader.exists(manifest_id)  # type: ignore[union-attr]

    async def stale_cursors(
        self,
        threshold_seconds: float = CURSOR_STALE_THRESHOLD_SECONDS,
    ) -> List[CursorRecord]:
        """
        Return cursor rows that have not been updated within threshold_seconds.

        A stale cursor typically indicates a manifest that is no longer running
        — either it completed without calling clear(), or the process died and
        was not restarted.  Operator tooling uses this to identify orphaned
        crawl sessions.

        Parameters
        ──────────
        threshold_seconds   Maximum age in seconds.  Default:
                            CURSOR_STALE_THRESHOLD_SECONDS (24 hours).

        Returns
        ───────
        List[CursorRecord]  Cursors older than the threshold, oldest first.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("stale_cursors")
        cutoff = time.time() - threshold_seconds
        return await self._reader.get_stale(cutoff)  # type: ignore[union-attr]

    async def clear_stale(
        self,
        threshold_seconds: float = CURSOR_STALE_THRESHOLD_SECONDS,
    ) -> int:
        """
        Delete stale cursor rows older than threshold_seconds.

        Returns the count of rows deleted.

        This is a maintenance operation — it removes orphaned cursors without
        requiring the caller to enumerate them first.  It is equivalent to
        calling clear() for every result from stale_cursors(), but more
        efficient because it is a single DELETE with a WHERE clause.

        Parameters
        ──────────
        threshold_seconds   Age threshold.  Default 24 hours.

        Returns
        ───────
        int     Number of rows deleted.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("clear_stale")
        cutoff = time.time() - threshold_seconds
        return await self._writer.clear_stale(cutoff)  # type: ignore[union-attr]

    async def health(self) -> CursorHealth:
        """
        Return a CursorHealth snapshot.

        Runs integrity_check and collects file size, WAL size, and cursor
        counts.  This is a relatively expensive operation — it reads every
        page of the database for the integrity check.  Call it at startup and
        on operator request, never in the hot path.

        Returns CursorHealth even if the instance is not initialized — the
        `initialized` field reflects the current state.
        """
        db_path_str = str(self._db_path.resolve())
        db_exists = self._db_path.exists()

        # Collect file sizes without needing an open DB connection.
        db_size_bytes = self._db_path.stat().st_size if db_exists else 0
        wal_path = self._db_path.with_suffix(".db-wal")
        wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0

        if not self._initialized or self._db is None:
            return CursorHealth(
                db_path=db_path_str,
                db_exists=db_exists,
                wal_mode_active=False,
                integrity_ok=False,
                active_cursor_count=0,
                stale_cursor_count=0,
                db_size_bytes=db_size_bytes,
                wal_size_bytes=wal_size_bytes,
                initialized=False,
                error="not initialized",
            )

        error: Optional[str] = None
        wal_mode_active = False
        integrity_ok = False
        active_count = 0
        stale_count = 0

        try:
            # WAL mode check.
            cursor = await self._db.execute("PRAGMA journal_mode")
            row = await cursor.fetchone()
            await cursor.close()
            wal_mode_active = (
                row is not None and row[0].lower() == "wal"
            )

            # Integrity check.
            integrity_ok = await _SchemaManager.verify_integrity(self._db)

            # Cursor counts.
            active_count = await self._reader.count()  # type: ignore[union-attr]
            cutoff = time.time() - CURSOR_STALE_THRESHOLD_SECONDS
            stale_rows = await self._reader.get_stale(cutoff)  # type: ignore[union-attr]
            stale_count = len(stale_rows)

        except CursorError as exc:
            error = str(exc)
        except aiosqlite.Error as exc:
            error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            error = f"unexpected: {type(exc).__name__}: {exc}"

        return CursorHealth(
            db_path=db_path_str,
            db_exists=db_exists,
            wal_mode_active=wal_mode_active,
            integrity_ok=integrity_ok,
            active_cursor_count=active_count,
            stale_cursor_count=stale_count,
            db_size_bytes=db_size_bytes,
            wal_size_bytes=wal_size_bytes,
            initialized=self._initialized,
            error=error,
        )

    async def wal_checkpoint(self, mode: str = "PASSIVE") -> Tuple[int, int]:
        """
        Issue a WAL checkpoint.

        Returns (wal_log_size, frames_checkpointed).

        Modes:
          'PASSIVE'   Non-blocking.  Checkpoints pages not held by readers.
          'FULL'      Blocks until all readers release WAL snapshots, then
                      checkpoints all pages.
          'TRUNCATE'  Full checkpoint + truncates WAL file to zero.

        Use PASSIVE in the normal close() path.  Use TRUNCATE after large
        batch operations to compact the WAL.

        Raises
        ──────
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("wal_checkpoint")
        return await _SchemaManager.wal_checkpoint(self._db, mode=mode)  # type: ignore[arg-type]

    async def integrity_check(self) -> bool:
        """
        Run PRAGMA integrity_check and return True iff the database is clean.

        Returns False (does not raise) if any integrity issue is detected.
        Logs all issues at ERROR level.

        Raises
        ──────
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("integrity_check")
        return await _SchemaManager.verify_integrity(self._db)  # type: ignore[arg-type]

    def session_metrics(self, manifest_id: str) -> Optional[Dict[str, object]]:
        """
        Return in-memory session metrics for manifest_id.

        Returns None if no session exists (no checkpoint has been written).
        This is a pure in-memory operation — no database access.

        The returned dict contains:
          elapsed_seconds, checkpoint_count, failure_count, last_position,
          total_urls, velocity_urls_per_second, is_stalled,
          avg_checkpoint_latency_ms, p99_checkpoint_latency_ms,
          estimated_remaining_seconds.
        """
        session = self._sessions.get(manifest_id)
        if session is None:
            return None
        return session.to_summary_dict()

    def all_session_metrics(self) -> Dict[str, Dict[str, object]]:
        """
        Return in-memory session metrics for all active sessions.

        Keys are manifest_id strings.  Values are the same dicts as
        session_metrics().  Empty dict if no sessions are active.
        """
        return {mid: s.to_summary_dict() for mid, s in self._sessions.items()}

    async def export_snapshot(self) -> Dict[str, object]:
        """
        Export a complete snapshot of cursor state for monitoring.

        Combines on-disk cursor rows with in-memory session metrics.
        Suitable for Prometheus scrape or operator dashboards.

        Raises
        ──────
        CursorError     On SQLite failure.
        RuntimeError    If initialize() was not called.
        """
        self._ensure_initialized("export_snapshot")

        records = await self.all_active()
        snapshot = {
            "timestamp": time.time(),
            "db_path": str(self._db_path),
            "initialized": self._initialized,
            "total_checkpoint_count": self._total_checkpoint_count,
            "active_cursor_count": len(records),
            "cursors": [],
        }

        cursor_list = []
        for record in records:
            entry = record.to_dict()
            # Merge in session metrics if available.
            session_data = self.session_metrics(record.manifest_id)
            if session_data is not None:
                entry["session"] = session_data
            cursor_list.append(entry)
        snapshot["cursors"] = cursor_list  # type: ignore[assignment]

        return snapshot

    def monitor(
        self,
        manifest_id: str,
        total_urls: int,
    ) -> "CursorMonitor":
        """
        Return a CursorMonitor context manager for manifest_id.

        Usage::

            async with cursor.monitor(manifest_id, total_urls) as mon:
                async for crawl_url in frontier.resume(manifest_id):
                    await do_fetch(crawl_url)
                    if mon.should_checkpoint(url_index):
                        await mon.checkpoint_now(url_index, crawl_url.url)

        The monitor auto-checkpoints on __aexit__ regardless of completion
        status, ensuring the last partial-interval batch is not lost.
        """
        return CursorMonitor(
            cursor=self,
            manifest_id=manifest_id,
            total_urls=total_urls,
        )

    # ── Async context manager support ─────────────────────────────────────────

    async def __aenter__(self) -> "CrawlCursor":
        """Support `async with CrawlCursor() as cursor:` pattern."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close on context manager exit."""
        await self.close()
        return None

    # ── Private helpers ───────────────────────────────────────────────────────

    def _ensure_initialized(self, method_name: str) -> None:
        """Raise RuntimeError if initialize() has not been called."""
        if not self._initialized or self._db is None:
            raise RuntimeError(
                f"CrawlCursor.{method_name}() called before initialize(). "
                "Call await cursor.initialize() first."
            )

    def _get_or_create_session(
        self, manifest_id: str, total_urls: int
    ) -> _SessionState:
        """
        Return existing session or create a new one.

        Session creation sets session_start to time.monotonic() and
        initializes all counters to zero.
        """
        if manifest_id not in self._sessions:
            now = time.monotonic()
            self._sessions[manifest_id] = _SessionState(
                manifest_id=manifest_id,
                session_start=now,
                last_checkpoint_mono=now,
                last_position=0,
                checkpoint_count=0,
                failure_count=0,
                total_urls=total_urls,
            )
        else:
            # Update total_urls if it changed (e.g. dynamic manifest extension).
            self._sessions[manifest_id].total_urls = total_urls
        return self._sessions[manifest_id]

    async def _background_wal_truncate(self) -> None:
        """
        Issue a TRUNCATE WAL checkpoint in the background.

        Scheduled via asyncio.ensure_future() from the checkpoint hot path.
        Uses TRUNCATE mode to reduce WAL file size after sustained write
        activity.  Errors are logged but never propagated — this is a
        maintenance operation that must not interrupt the crawl.
        """
        if not self._initialized or self._db is None:
            return
        try:
            log_size, checkpointed = await _SchemaManager.wal_checkpoint(
                self._db, mode="TRUNCATE"
            )
            _log.debug(
                "cursor._background_wal_truncate: "
                "log=%d checkpointed=%d",
                log_size,
                checkpointed,
            )
        except Exception as exc:
            _log.warning(
                "cursor._background_wal_truncate failed (non-fatal): %s", exc
            )

    async def recovery_advisory(
        self,
        manifest_id: str,
        total_urls: int,
    ) -> RecoveryAdvisory:
        """
        Produce a RecoveryAdvisory for manifest_id before resuming.

        Call this at manifest startup — before calling frontier.resume() —
        to obtain a structured summary of what is about to happen.  The
        fetcher logs this at INFO level so operators can see the resume state
        without querying the database manually.

        If no cursor row exists for manifest_id (clean start or first run),
        the advisory carries resume_position=0 and was_clean_shutdown=True.
        """
        self._ensure_initialized("recovery_advisory")

        record = await self._reader.get_record(manifest_id)

        if record is None:
            return RecoveryAdvisory(
                manifest_id=manifest_id,
                resume_position=0,
                last_checkpoint_position=0,
                last_checkpoint_at=None,
                seconds_since_checkpoint=None,
                max_duplicate_window=0,
                total_urls=total_urls,
                progress_fraction=0.0,
                was_clean_shutdown=True,
            )

        resume_position = record.position + 1
        seconds_since = time.time() - record.checkpoint_at
        max_window = min(resume_position, CURSOR_CHECKPOINT_INTERVAL)

        advisory = RecoveryAdvisory(
            manifest_id=manifest_id,
            resume_position=resume_position,
            last_checkpoint_position=record.position,
            last_checkpoint_at=record.checkpoint_at,
            seconds_since_checkpoint=seconds_since,
            max_duplicate_window=max_window,
            total_urls=total_urls,
            progress_fraction=record.progress_fraction,
            was_clean_shutdown=False,
        )

        _log.info("%s", advisory.log_summary())
        return advisory

    async def checkpoint_batch(
        self,
        entries: List[BatchCheckpointEntry],
    ) -> BatchCheckpointResult:
        """
        Write multiple cursor rows in a single atomic transaction.

        All entries are written inside one BEGIN IMMEDIATE … COMMIT block.
        Either all succeed or the entire batch rolls back.
        """
        self._ensure_initialized("checkpoint_batch")

        if not entries:
            return BatchCheckpointResult(
                entries=[],
                total_latency_ms=0.0,
                written_count=0,
                failed_count=0,
                success=True,
            )

        checkpoint_at = time.time()
        t0 = time.monotonic()

        try:
            await self._db.execute("BEGIN IMMEDIATE")
            for entry in entries:
                await self._db.execute(
                    _DML_UPSERT,
                    (
                        entry.manifest_id,
                        entry.position,
                        entry.url,
                        checkpoint_at,
                        entry.total_urls,
                    ),
                )
            await self._db.execute("COMMIT")

            total_ms = (time.monotonic() - t0) * 1000.0
            per_entry_ms = total_ms / len(entries)

            for entry in entries:
                session = self._get_or_create_session(
                    entry.manifest_id, entry.total_urls
                )
                session.record_checkpoint(entry.position, per_entry_ms)
            self._total_checkpoint_count += len(entries)

            _log.info(
                "cursor.checkpoint_batch: wrote %d cursors in %.2fms",
                len(entries),
                total_ms,
            )

            return BatchCheckpointResult(
                entries=[(e.manifest_id, e.position, per_entry_ms) for e in entries],
                total_latency_ms=total_ms,
                written_count=len(entries),
                failed_count=0,
                success=True,
            )

        except aiosqlite.OperationalError as exc:
            try:
                await self._db.execute("ROLLBACK")
            except aiosqlite.Error:
                pass
            raise CursorError(
                manifest_id="__batch__",
                position=0,
                operation="checkpoint_batch",
                db_error=str(exc),
            ) from exc
        except aiosqlite.Error as exc:
            try:
                await self._db.execute("ROLLBACK")
            except aiosqlite.Error:
                pass
            raise CursorError(
                manifest_id="__batch__",
                position=0,
                operation="checkpoint_batch",
                db_error=f"{type(exc).__name__}: {exc}",
            ) from exc

    async def advance_if_ahead(
        self,
        manifest_id: str,
        position: int,
        url: str,
        total_urls: int,
    ) -> Optional[CheckpointResult]:
        """
        Write a checkpoint only if position > current stored position.

        Avoids redundant writes when the fetcher retries a checkpoint that
        was already committed.  Not atomic with respect to external writers —
        safe only in the single-process asyncio model.
        """
        self._ensure_initialized("advance_if_ahead")

        current = await self._reader.get_position(manifest_id)
        if position <= current:
            _log.debug(
                "cursor.advance_if_ahead: skipping manifest=%s pos=%d "
                "(current=%d, not advanced)",
                manifest_id[:8],
                position,
                current,
            )
            return None

        return await self.checkpoint(manifest_id, position, url, total_urls)


# ─────────────────────────────────────────────────────────────────────────────
# CURSOR MONITOR
# An async context manager that wraps a manifest execution lifecycle.
# Provides should_checkpoint() for interval management and auto-checkpoints
# on exit to capture the final partial batch.
# ─────────────────────────────────────────────────────────────────────────────


class CursorMonitor:
    """
    Async context manager for managing cursor checkpoints during a manifest.

    Returned by CrawlCursor.monitor().  Designed to be used at the fetcher
    level to simplify the checkpoint decision and final checkpoint on exit.

    The monitor tracks:
      - How many URLs have been processed since the last checkpoint.
      - The total elapsed time since the monitor was entered.
      - Whether the current interval requires a checkpoint.

    On __aexit__, the monitor issues a final checkpoint if any unwritten
    progress exists — i.e., if (url_index % CURSOR_CHECKPOINT_INTERVAL != 0).
    This ensures the last partial batch is not lost on clean shutdown.

    The monitor does NOT checkpoint on __aexit__ if the context exited with
    an exception — a crashed manifest should not issue a checkpoint at an
    unknown position.

    Usage
    ─────
    ::

        cursor = CrawlCursor()
        await cursor.initialize()

        async with cursor.monitor(manifest_id, total_urls) as mon:
            start_pos = await cursor.get_position(manifest_id)

            for idx, crawl_url in enumerate(
                frontier.resume_sync(manifest_id, start_pos), start=start_pos
            ):
                await do_fetch(crawl_url)

                if mon.should_checkpoint(idx):
                    await mon.checkpoint_now(idx, crawl_url.url)

        # Final checkpoint issued automatically on clean exit.
    """

    def __init__(
        self,
        cursor: CrawlCursor,
        manifest_id: str,
        total_urls: int,
    ) -> None:
        self._cursor = cursor
        self._manifest_id = manifest_id
        self._total_urls = total_urls

        self._enter_time_mono: float = 0.0
        self._enter_time_wall: float = 0.0
        self._last_checkpoint_position: int = -1
        self._last_checkpoint_url: str = ""
        self._current_position: int = -1
        self._current_url: str = ""
        self._checkpoint_count: int = 0
        self._active: bool = False

    async def __aenter__(self) -> "CursorMonitor":
        """Enter the monitor context.  Records start time."""
        self._enter_time_mono = time.monotonic()
        self._enter_time_wall = time.time()
        self._active = True

        # Read current position so we start from the right place.
        self._last_checkpoint_position = await self._cursor.get_position(
            self._manifest_id
        )
        _log.debug(
            "CursorMonitor.__aenter__: manifest=%s resume_from=%d total=%d",
            self._manifest_id[:8],
            self._last_checkpoint_position,
            self._total_urls,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        On clean exit: issue final checkpoint if unwritten progress exists.
        On exception exit: do not checkpoint — position may be inconsistent.
        """
        self._active = False

        if exc_type is not None:
            # Exception path — do not checkpoint.
            _log.warning(
                "CursorMonitor.__aexit__: manifest=%s exiting with exception %s "
                "— skipping final checkpoint.",
                self._manifest_id[:8],
                exc_type.__name__,
            )
            return

        # Clean exit: checkpoint if any unwritten progress.
        if (
            self._current_position > self._last_checkpoint_position
            and self._current_url
        ):
            try:
                await self._cursor.checkpoint(
                    manifest_id=self._manifest_id,
                    position=self._current_position,
                    url=self._current_url,
                    total_urls=self._total_urls,
                )
                _log.debug(
                    "CursorMonitor.__aexit__: final checkpoint manifest=%s pos=%d",
                    self._manifest_id[:8],
                    self._current_position,
                )
            except CursorError as exc:
                _log.warning(
                    "CursorMonitor.__aexit__: final checkpoint failed (non-fatal): %s",
                    exc,
                )

    def should_checkpoint(self, position: int) -> bool: # noqa
        """
        Return True if a checkpoint should be written at this position.

        True iff position is a non-zero multiple of CURSOR_CHECKPOINT_INTERVAL.
        The fetcher calls this after every URL completion and writes a
        checkpoint only when True is returned.

        Position 0 never triggers a checkpoint — the first checkpoint is at
        position CURSOR_CHECKPOINT_INTERVAL.

        Parameters
        ──────────
        position    0-based index of the URL just completed.

        Returns
        ───────
        bool    True iff a checkpoint should be written.
        """
        if position <= 0:
            return False
        return position % CURSOR_CHECKPOINT_INTERVAL == 0

    async def checkpoint_now(self, position: int, url: str) -> CheckpointResult:
        """
        Write a checkpoint at position.

        Updates internal tracking so __aexit__ knows whether a final
        checkpoint is needed.

        Parameters
        ──────────
        position    0-based index of the completed URL.
        url         The URL string at position.

        Returns
        ───────
        CheckpointResult    Checkpoint metadata including latency.

        Raises
        ──────
        CursorError     On SQLite failure.
        """
        result = await self._cursor.checkpoint(
            manifest_id=self._manifest_id,
            position=position,
            url=url,
            total_urls=self._total_urls,
        )
        self._last_checkpoint_position = position
        self._last_checkpoint_url = url
        self._current_position = position
        self._current_url = url
        self._checkpoint_count += 1
        return result

    def update_position(self, position: int, url: str) -> None:
        """
        Notify the monitor of the current URL being processed.

        Call this after every URL fetch, even when should_checkpoint() is
        False.  This allows __aexit__ to issue a final checkpoint at the
        correct position.

        Parameters
        ──────────
        position    Current URL index (0-based).
        url         The URL string at position.
        """
        self._current_position = position
        self._current_url = url

    @property
    def progress(self) -> float:
        """
        Current progress as a fraction 0.0–1.0.

        Based on current_position, not the last checkpoint — may be ahead
        of the persisted position by up to CURSOR_CHECKPOINT_INTERVAL URLs.
        """
        if self._total_urls <= 0:
            return 0.0
        if self._current_position < 0:
            return 0.0
        return min(1.0, (self._current_position + 1) / self._total_urls)

    @property
    def elapsed_seconds(self) -> float:
        """Seconds since this monitor was entered."""
        if self._enter_time_mono == 0.0:
            return 0.0
        return time.monotonic() - self._enter_time_mono

    @property
    def checkpoint_count(self) -> int:
        """Number of checkpoints written through this monitor."""
        return self._checkpoint_count

    def estimated_remaining_seconds(self) -> Optional[float]:
        """
        ETA to manifest completion based on measured processing rate.

        Returns None if insufficient data (no time elapsed or no position
        data).
        """
        elapsed = self.elapsed_seconds
        if elapsed <= 0 or self._current_position < 0:
            return None
        rate = (self._current_position + 1) / elapsed  # URLs/s
        if rate <= 0:
            return None
        remaining = self._total_urls - (self._current_position + 1)
        if remaining <= 0:
            return 0.0
        return remaining / rate

    def summary(self) -> Dict[str, object]:
        """Export monitor state as a flat dict for logging or monitoring."""
        return {
            "manifest_id": self._manifest_id,
            "total_urls": self._total_urls,
            "current_position": self._current_position,
            "last_checkpoint_position": self._last_checkpoint_position,
            "checkpoint_count": self._checkpoint_count,
            "progress": round(self.progress, 4),
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "estimated_remaining_seconds": self.estimated_remaining_seconds(),
            "active": self._active,
        }


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC TEST SUITE
# Self-contained tests that can be run from the CLI:
#   python -m crawl_cursor --test
#
# All tests create a temporary database in /tmp and clean up after themselves.
# They do not require any other AXIOM components to be present.
# ─────────────────────────────────────────────────────────────────────────────


class _DiagnosticRunner:
    """
    Runs the built-in diagnostic test suite for crawl_cursor.

    Tests validate the seven behavioral guarantees from the build specification:

      1. get_position() returns 0 for unknown manifest_id.
      2. checkpoint() + get_position() returns correct position.
      3. Checkpoint survives simulated crash — reopen DB, position still there.
      4. WAL mode enabled — PRAGMA journal_mode returns 'wal'.
      5. clear() removes cursor row for completed manifest.
      6. all_active() returns all non-cleared cursors.
      7. Double checkpoint at same position is idempotent.

    Plus extended tests:
      8. Progress fraction computed correctly from checkpoint.
      9. Stale cursor detection with configurable threshold.
      10. Concurrent manifest cursors do not interfere.
      11. CursorMonitor should_checkpoint() triggers at correct interval.
      12. CursorMonitor final checkpoint on clean __aexit__.
      13. CursorError raised on bad DB operation, does not crash.

    Results are printed to stdout.  Exits 0 on success, 1 on failure.
    """

    def __init__(self) -> None:
        self._passed: int = 0
        self._failed: int = 0
        self._errors: List[str] = []

    async def run_all(self) -> bool:
        """
        Execute all tests.  Return True if all passed.
        """
        import tempfile
        import os

        print("=" * 70)
        print("crawl_cursor — diagnostic test suite")
        print("=" * 70)

        test_methods: List[Callable[[Path], Awaitable[None]]] = [
            self._test_unknown_manifest_returns_zero,
            self._test_checkpoint_and_get_position,
            self._test_crash_recovery,
            self._test_wal_mode_active,
            self._test_clear_removes_cursor,
            self._test_all_active_returns_all_cursors,
            self._test_double_checkpoint_idempotent,
            self._test_progress_fraction,
            self._test_stale_cursor_detection,
            self._test_concurrent_manifests,
            self._test_monitor_should_checkpoint,
            self._test_monitor_final_checkpoint,
            self._test_cursor_record_properties,
        ]

        for test in test_methods:
            db_file = Path(tempfile.mktemp(suffix=".db", prefix="crawl_cursor_test_"))
            try:
                await test(db_file)
            except Exception as exc:
                self._record_failure(test.__name__, f"unexpected exception: {exc}")
            finally:
                # Clean up test database and WAL files.
                for suffix in ("", "-wal", "-shm"):
                    p = Path(str(db_file) + suffix)
                    if p.exists():
                        try:
                            os.unlink(p)
                        except OSError:
                            pass

        print("-" * 70)
        print(f"Results: {self._passed} passed, {self._failed} failed")
        if self._errors:
            print("\nFailures:")
            for err in self._errors:
                print(f"  ✗ {err}")
        print("=" * 70)
        return self._failed == 0

    def _record_pass(self, name: str) -> None:
        self._passed += 1
        print(f"  ✓ {name}")

    def _record_failure(self, name: str, reason: str) -> None:
        self._failed += 1
        msg = f"{name}: {reason}"
        self._errors.append(msg)
        print(f"  ✗ {msg}")

    def _assert(self, name: str, condition: bool, reason: str = "") -> None:
        if condition:
            self._record_pass(name)
        else:
            self._record_failure(name, reason or "assertion failed")

    async def _make_cursor(self, db_path: Path) -> CrawlCursor: # noqa
        """Create and initialize a CrawlCursor against db_path."""
        # Ensure parent directory exists for test databases.
        db_path.parent.mkdir(parents=True, exist_ok=True)
        cursor = CrawlCursor(db_path=db_path)
        await cursor.initialize()
        return cursor

    # ── Test implementations ───────────────────────────────────────────────────

    async def _test_unknown_manifest_returns_zero(self, db_path: Path) -> None:
        """Test 1: get_position returns 0 for unknown manifest_id."""
        cursor = await self._make_cursor(db_path)
        try:
            pos = await cursor.get_position("00000000-0000-0000-0000-000000000001")
            self._assert(
                "get_position unknown returns 0",
                pos == 0,
                f"expected 0, got {pos}",
            )
        finally:
            await cursor.close()

    async def _test_checkpoint_and_get_position(self, db_path: Path) -> None:
        """Test 2: checkpoint + get_position returns correct position."""
        cursor = await self._make_cursor(db_path)
        mid = "00000000-0000-0000-0000-000000000002"
        try:
            result = await cursor.checkpoint(
                manifest_id=mid,
                position=42,
                url="https://example.com/page-42",
                total_urls=100,
            )
            pos = await cursor.get_position(mid)
            self._assert(
                "checkpoint write returns CheckpointResult",
                isinstance(result, CheckpointResult),
                f"got {type(result)}",
            )
            self._assert(
                "checkpoint + get_position returns 42",
                pos == 42,
                f"expected 42, got {pos}",
            )
            self._assert(
                "checkpoint latency_ms is positive",
                result.latency_ms > 0,
                f"latency_ms={result.latency_ms}",
            )
            self._assert(
                "checkpoint was_write is True",
                result.was_write,
                "was_write should be True",
            )
        finally:
            await cursor.close()

    async def _test_crash_recovery(self, db_path: Path) -> None:
        """Test 3: checkpoint survives simulated process death (close + reopen)."""
        mid = "00000000-0000-0000-0000-000000000003"
        cursor1 = await self._make_cursor(db_path)
        try:
            await cursor1.checkpoint(
                manifest_id=mid,
                position=99,
                url="https://example.com/page-99",
                total_urls=500,
            )
        finally:
            # Simulate crash: close without calling clear().
            await cursor1.close()

        # Reopen — simulates restart after process death.
        cursor2 = await self._make_cursor(db_path)
        try:
            pos = await cursor2.get_position(mid)
            self._assert(
                "crash recovery: position persists across close/reopen",
                pos == 99,
                f"expected 99, got {pos}",
            )
            record = await cursor2.get_record(mid)
            self._assert(
                "crash recovery: record is readable after reopen",
                record is not None and record.position == 99,
                f"record={record}",
            )
        finally:
            await cursor2.close()

    async def _test_wal_mode_active(self, db_path: Path) -> None:
        """Test 4: WAL mode is enabled — PRAGMA journal_mode returns 'wal'."""
        cursor = await self._make_cursor(db_path)
        try:
            db_cursor = await cursor._db.execute("PRAGMA journal_mode") # noqa
            row = await db_cursor.fetchone()
            await db_cursor.close()
            mode = row[0].lower() if row else "unknown"
            self._assert(
                "journal_mode is WAL",
                mode == "wal",
                f"got '{mode}', expected 'wal'",
            )
        finally:
            await cursor.close()

    async def _test_clear_removes_cursor(self, db_path: Path) -> None:
        """Test 5: clear() removes the cursor row; get_position returns 0 after."""
        mid = "00000000-0000-0000-0000-000000000005"
        cursor = await self._make_cursor(db_path)
        try:
            await cursor.checkpoint(
                manifest_id=mid,
                position=200,
                url="https://example.com/page-200",
                total_urls=1000,
            )
            before = await cursor.get_position(mid)
            await cursor.clear(mid)
            after = await cursor.get_position(mid)
            self._assert(
                "before clear: position is 200",
                before == 200,
                f"before={before}",
            )
            self._assert(
                "after clear: position returns 0",
                after == 0,
                f"after={after}",
            )
            self._assert(
                "after clear: is_active returns False",
                not await cursor.is_active(mid),
                "expected False",
            )
        finally:
            await cursor.close()

    async def _test_all_active_returns_all_cursors(self, db_path: Path) -> None:
        """Test 6: all_active() returns all non-cleared cursors."""
        mids = [
            "00000000-0000-0000-0000-00000000000a",
            "00000000-0000-0000-0000-00000000000b",
            "00000000-0000-0000-0000-00000000000c",
        ]
        cursor = await self._make_cursor(db_path)
        try:
            for i, mid in enumerate(mids):
                await cursor.checkpoint(
                    manifest_id=mid,
                    position=i * 10,
                    url=f"https://example.com/page-{i}",
                    total_urls=100,
                )

            # Clear one.
            await cursor.clear(mids[1])

            active = await cursor.all_active()
            active_ids = {r.manifest_id for r in active}

            self._assert(
                "all_active returns 2 records after clearing one of three",
                len(active) == 2,
                f"got {len(active)}",
            )
            self._assert(
                "all_active: cleared manifest not in results",
                mids[1] not in active_ids,
                f"found {mids[1]} in {active_ids}",
            )
            self._assert(
                "all_active: non-cleared manifests in results",
                mids[0] in active_ids and mids[2] in active_ids,
                f"active_ids={active_ids}",
            )
        finally:
            await cursor.close()

    async def _test_double_checkpoint_idempotent(self, db_path: Path) -> None:
        """Test 7: double checkpoint at same position produces identical row."""
        mid = "00000000-0000-0000-0000-000000000007"
        cursor = await self._make_cursor(db_path)
        try:
            # Write position 50 twice.
            await cursor.checkpoint(mid, 50, "https://example.com/50", 200)
            await cursor.checkpoint(mid, 50, "https://example.com/50", 200)

            pos = await cursor.get_position(mid)
            record = await cursor.get_record(mid)

            self._assert(
                "double checkpoint: position still 50",
                pos == 50,
                f"expected 50, got {pos}",
            )
            self._assert(
                "double checkpoint: only one record exists",
                record is not None and record.position == 50,
                f"record={record}",
            )

            # Verify only one row in DB (INSERT OR REPLACE semantics).
            count_cursor = await cursor._db.execute( # noqa
                "SELECT COUNT(*) FROM cursors WHERE manifest_id = ?", (mid,)
            )
            row = await count_cursor.fetchone()
            await count_cursor.close()
            self._assert(
                "double checkpoint: exactly one row in table",
                row[0] == 1,
                f"row count = {row[0]}",
            )
        finally:
            await cursor.close()

    async def _test_progress_fraction(self, db_path: Path) -> None:
        """Test 8: progress_fraction computed correctly from checkpoint."""
        mid = "00000000-0000-0000-0000-000000000008"
        cursor = await self._make_cursor(db_path)
        try:
            await cursor.checkpoint(mid, 49, "https://example.com/49", 100)
            record = await cursor.get_record(mid)
            frac = await cursor.progress_fraction(mid)

            self._assert(
                "progress_fraction: record is not None",
                record is not None,
                "record is None",
            )
            self._assert(
                "progress_fraction = 0.50 for pos=49 of 100",
                abs(frac - 0.50) < 1e-9,
                f"got {frac}",
            )
            self._assert(
                "record.remaining_urls = 50",
                record is not None and record.remaining_urls == 50,
                f"remaining={record.remaining_urls if record else 'N/A'}",
            )
        finally:
            await cursor.close()

    async def _test_stale_cursor_detection(self, db_path: Path) -> None:
        """Test 9: stale cursor detection with custom threshold."""
        mid_fresh = "00000000-0000-0000-0000-000000000009"
        mid_stale = "00000000-0000-0000-0000-00000000000f"
        cursor = await self._make_cursor(db_path)
        try:
            # Write fresh cursor (checkpoint_at = now).
            await cursor.checkpoint(mid_fresh, 10, "https://example.com/10", 100)

            # Write stale cursor by directly inserting an old timestamp.
            old_ts = time.time() - 7200.0  # 2 hours ago
            await cursor._db.execute("BEGIN IMMEDIATE") # noqa
            await cursor._db.execute( # noqa
                _DML_UPSERT,
                (mid_stale, 5, "https://example.com/5", old_ts, 100),
            )
            await cursor._db.execute("COMMIT") # noqa

            stale = await cursor.stale_cursors(threshold_seconds=3600.0)
            stale_ids = {r.manifest_id for r in stale}

            self._assert(
                "stale_cursors: mid_stale in results",
                mid_stale in stale_ids,
                f"stale_ids={stale_ids}",
            )
            self._assert(
                "stale_cursors: mid_fresh NOT in results",
                mid_fresh not in stale_ids,
                f"stale_ids={stale_ids}",
            )

            # clear_stale should remove mid_stale.
            deleted = await cursor.clear_stale(threshold_seconds=3600.0)
            after = await cursor.stale_cursors(threshold_seconds=3600.0)

            self._assert(
                "clear_stale: returns deleted count >= 1",
                deleted >= 1,
                f"deleted={deleted}",
            )
            self._assert(
                "clear_stale: stale cursor gone after deletion",
                all(r.manifest_id != mid_stale for r in after),
                f"still found {mid_stale}",
            )
        finally:
            await cursor.close()

    async def _test_concurrent_manifests(self, db_path: Path) -> None:
        """Test 10: Multiple concurrent manifests do not interfere."""
        mids = [
            f"00000000-0000-0000-0000-0000000000{i:02x}" for i in range(10, 15)
        ]
        cursor = await self._make_cursor(db_path)
        try:
            # Write checkpoints for 5 manifests.
            for i, mid in enumerate(mids):
                await cursor.checkpoint(
                    mid, i * 100, f"https://example-{i}.com/", (i + 1) * 500
                )

            # Verify each manifest has its own independent position.
            all_correct = True
            for i, mid in enumerate(mids):
                pos = await cursor.get_position(mid)
                if pos != i * 100:
                    all_correct = False
                    break

            self._assert(
                "concurrent manifests: each has correct independent position",
                all_correct,
                "at least one manifest has wrong position",
            )

            # Update one manifest and verify others unchanged.
            await cursor.checkpoint(mids[2], 9999, "https://example-2.com/end", 5000)

            pos_0 = await cursor.get_position(mids[0])
            pos_2 = await cursor.get_position(mids[2])
            pos_4 = await cursor.get_position(mids[4])

            self._assert(
                "concurrent manifests: manifest 0 unchanged after manifest 2 update",
                pos_0 == 0,
                f"pos_0={pos_0}",
            )
            self._assert(
                "concurrent manifests: manifest 2 updated to 9999",
                pos_2 == 9999,
                f"pos_2={pos_2}",
            )
            self._assert(
                "concurrent manifests: manifest 4 unchanged",
                pos_4 == 400,
                f"pos_4={pos_4}",
            )
        finally:
            await cursor.close()

    async def _test_monitor_should_checkpoint(self, db_path: Path) -> None:
        """Test 11: CursorMonitor.should_checkpoint triggers at correct interval."""
        cursor = await self._make_cursor(db_path)
        mid = "00000000-0000-0000-0000-00000000000d"
        try:
            mon = cursor.monitor(mid, total_urls=1000)
            # Test the interval logic without entering the context manager.
            self._assert(
                "should_checkpoint(0) is False",
                not mon.should_checkpoint(0),
                "position 0 should never trigger",
            )
            self._assert(
                "should_checkpoint(99) is False",
                not mon.should_checkpoint(99),
                "99 is not a multiple of 100",
            )
            self._assert(
                "should_checkpoint(100) is True",
                mon.should_checkpoint(CURSOR_CHECKPOINT_INTERVAL),
                f"position {CURSOR_CHECKPOINT_INTERVAL} should trigger",
            )
            self._assert(
                "should_checkpoint(200) is True",
                mon.should_checkpoint(CURSOR_CHECKPOINT_INTERVAL * 2),
                f"position {CURSOR_CHECKPOINT_INTERVAL * 2} should trigger",
            )
            self._assert(
                "should_checkpoint(150) is False",
                not mon.should_checkpoint(150),
                "150 is not a multiple of 100",
            )
        finally:
            await cursor.close()

    async def _test_monitor_final_checkpoint(self, db_path: Path) -> None:
        """Test 12: CursorMonitor issues final checkpoint on clean __aexit__."""
        mid = "00000000-0000-0000-0000-00000000000e"
        cursor = await self._make_cursor(db_path)
        try:
            # Enter monitor, update position but do not call checkpoint_now
            # (simulate fetching 42 URLs, which is < checkpoint interval).
            async with cursor.monitor(mid, total_urls=100) as mon:
                mon.update_position(42, "https://example.com/42")
                # No explicit checkpoint — monitor should do it on exit.

            # After clean exit, position 42 should be persisted.
            pos = await cursor.get_position(mid)
            self._assert(
                "monitor final checkpoint: position 42 persisted on clean exit",
                pos == 42,
                f"expected 42, got {pos}",
            )
        finally:
            await cursor.close()

    async def _test_cursor_record_properties(self, db_path: Path) -> None:
        """Test 13: CursorRecord computed properties are correct."""
        record = CursorRecord(
            manifest_id="test-manifest",
            position=9,
            url="https://example.com/9",
            checkpoint_at=time.time(),
            total_urls=10,
        )
        self._assert(
            "CursorRecord.progress_fraction for 9/10 ≈ 1.0",
            abs(record.progress_fraction - 1.0) < 1e-9,
            f"got {record.progress_fraction}",
        )
        self._assert(
            "CursorRecord.remaining_urls = 0 for position 9 of 10",
            record.remaining_urls == 0,
            f"got {record.remaining_urls}",
        )
        self._assert(
            "CursorRecord.is_complete True for last URL",
            record.is_complete,
            "expected True",
        )

        zero_record = CursorRecord(
            manifest_id="zero",
            position=0,
            url="https://example.com/0",
            checkpoint_at=time.time(),
            total_urls=0,
        )
        self._assert(
            "CursorRecord.progress_fraction for total_urls=0 is 0.0",
            zero_record.progress_fraction == 0.0,
            f"got {zero_record.progress_fraction}",
        )
        self._assert(
            "CursorRecord.is_complete False for total_urls=0",
            not zero_record.is_complete,
            "expected False for empty manifest",
        )


# ─────────────────────────────────────────────────────────────────────────────
# RECOVERY ADVISORY
# Provides pre-restart analysis so the fetcher can log meaningful context
# about what is about to be re-fetched after a crash recovery.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecoveryAdvisory:
    """
    Pre-restart analysis for a manifest about to resume after a crash.

    Produced by CrawlCursor.recovery_advisory().  The fetcher logs this at
    INFO level before resuming the frontier.  It carries the information needed
    to understand the worst-case duplicate-fetch window without querying the
    frontier or Bloom filter directly.

    Fields
    ──────
    manifest_id             The manifest being resumed.
    resume_position         The position the frontier will resume from (= cursor
                            position + 1).  This is the first URL that will be
                            fetched again.
    last_checkpoint_position  The last position that was checkpointed to disk.
    last_checkpoint_at      Unix timestamp of the last successful checkpoint.
    seconds_since_checkpoint  How long the process was dead (or idle) since the
                              last checkpoint.
    max_duplicate_window    The worst-case count of URLs that may be re-fetched.
                            Equal to min(resume_position, CURSOR_CHECKPOINT_INTERVAL).
                            In practice, it is the count of URLs processed between
                            the last checkpoint write and the process death.
    total_urls              Total URL count in the manifest.
    progress_fraction       Fraction complete at the last checkpoint.
    was_clean_shutdown      False if the cursor row exists (indicating the
                            process died without calling clear()).  True if
                            resume_position is 0 (no prior cursor = clean start).
    """

    manifest_id: str
    resume_position: int
    last_checkpoint_position: int
    last_checkpoint_at: Optional[float]
    seconds_since_checkpoint: Optional[float]
    max_duplicate_window: int
    total_urls: int
    progress_fraction: float
    was_clean_shutdown: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "manifest_id": self.manifest_id,
            "resume_position": self.resume_position,
            "last_checkpoint_position": self.last_checkpoint_position,
            "last_checkpoint_at": self.last_checkpoint_at,
            "seconds_since_checkpoint": (
                round(self.seconds_since_checkpoint, 1)
                if self.seconds_since_checkpoint is not None
                else None
            ),
            "max_duplicate_window": self.max_duplicate_window,
            "total_urls": self.total_urls,
            "progress_fraction": round(self.progress_fraction, 4),
            "was_clean_shutdown": self.was_clean_shutdown,
        }

    def log_summary(self) -> str:
        """Single-line log summary suitable for INFO-level logging."""
        if self.was_clean_shutdown:
            return (
                f"RecoveryAdvisory manifest={self.manifest_id[:8]}: "
                f"clean start (no prior cursor) total={self.total_urls}"
            )
        age = (
            f"{self.seconds_since_checkpoint:.0f}s ago"
            if self.seconds_since_checkpoint is not None
            else "unknown age"
        )
        return (
            f"RecoveryAdvisory manifest={self.manifest_id[:8]}: "
            f"resuming from pos={self.resume_position} "
            f"(checkpoint was pos={self.last_checkpoint_position}, {age}). "
            f"max_duplicate_window={self.max_duplicate_window} URLs. "
            f"progress={self.progress_fraction * 100:.1f}%"
        )


# ─────────────────────────────────────────────────────────────────────────────
# BATCH CHECKPOINT SUPPORT
# The standard checkpoint() API handles one manifest at a time.  The batch
# API allows multiple manifests to be checkpointed in a single transaction —
# useful when the fetcher is running multiple concurrent manifests and wants
# to write all positions atomically before a graceful shutdown.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BatchCheckpointEntry:
    """
    One entry in a batch checkpoint operation.

    Passed to CrawlCursor.checkpoint_batch() as a list.
    Mirrors the parameters of checkpoint() for a single manifest.
    """

    manifest_id: str
    position: int
    url: str
    total_urls: int


@dataclass(frozen=True)
class BatchCheckpointResult:
    """
    Result of CrawlCursor.checkpoint_batch().

    Carries per-manifest results and aggregate statistics.

    Fields
    ──────
    entries         List of (manifest_id, position, latency_ms) tuples for each
                    entry that was written.
    total_latency_ms  Wall time of the entire batch transaction.
    written_count   Number of entries actually written to the database.
    failed_count    Number of entries that raised CursorError (partial writes
                    are not possible — either all succeed or the batch rolls
                    back).
    success         True iff failed_count is 0.
    """

    entries: List[Tuple[str, int, float]]
    total_latency_ms: float
    written_count: int
    failed_count: int
    success: bool

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_latency_ms": round(self.total_latency_ms, 3),
            "written_count": self.written_count,
            "failed_count": self.failed_count,
            "success": self.success,
            "entries": [
                {"manifest_id": e[0][:8], "position": e[1], "latency_ms": round(e[2], 3)}
                for e in self.entries
            ],
        }



# ─────────────────────────────────────────────────────────────────────────────
# CURSOR BACKUP
# Lightweight backup and point-in-time snapshot utilities.  These are
# operator tools — not used in the hot fetch path.
# ─────────────────────────────────────────────────────────────────────────────


class CursorBackup:
    """
    Backup and restore utilities for the cursor database.

    These are operator tools.  They are not called from the fetch path.
    A backup is a copy of the SQLite file taken while the database is open —
    SQLite's online backup API (used by aiosqlite.Connection.backup()) handles
    this safely even while writes are in progress.

    Usage::

        backup = CursorBackup(cursor)
        await backup.backup_to(Path("backups/crawl_cursor_backup.db"))

        # Later, to restore:
        await backup.restore_from(Path("backups/crawl_cursor_backup.db"))
    """

    def __init__(self, cursor: CrawlCursor) -> None:
        """
        Construct a CursorBackup for the given CrawlCursor instance.

        The cursor must be initialized before calling backup_to() or inspect().

        Parameters
        ──────────
        cursor  An initialized CrawlCursor instance.
        """
        self._cursor = cursor

    async def backup_to(self, dest_path: Path) -> None:
        """
        Write a consistent point-in-time copy of the cursor database to dest_path.

        Uses SQLite's online backup API.  The source database may have
        concurrent writes in progress — the backup captures a consistent
        snapshot without blocking the writer.

        If dest_path already exists, it is overwritten.

        Parameters
        ──────────
        dest_path   Destination path for the backup file.

        Raises
        ──────
        RuntimeError    If the cursor is not initialized.
        aiosqlite.Error On backup failure.
        """
        self._cursor._ensure_initialized("backup_to") # noqa

        dest_path.parent.mkdir(parents=True, exist_ok=True)
        _log.info("CursorBackup.backup_to: backing up to %s", dest_path)

        t0 = time.monotonic()
        dest_conn = await aiosqlite.connect(
            str(dest_path), isolation_level=None
        )
        try:
            # aiosqlite's backup() uses the sqlite3 online backup API.
            await self._cursor._db.backup(dest_conn)  # type: ignore[union-attr] # noqa
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            _log.info(
                "CursorBackup.backup_to: complete in %.1fms → %s",
                elapsed_ms,
                dest_path,
            )
        finally:
            await dest_conn.close()

    async def inspect(self, backup_path: Path) -> List[CursorRecord]: # noqa
        """
        Read and return all cursor records from a backup file.

        Opens the backup as a read-only connection.  Does not affect the live
        cursor database.  Returns an empty list if the backup has no cursor
        rows.

        Parameters
        ──────────
        backup_path     Path to a backup created by backup_to().

        Returns
        ───────
        List[CursorRecord]  All rows in the backup's cursors table.

        Raises
        ──────
        FileNotFoundError   If backup_path does not exist.
        aiosqlite.Error     On read failure.
        """
        if not backup_path.exists():
            raise FileNotFoundError(
                f"CursorBackup.inspect: backup file not found: {backup_path}"
            )

        conn = await aiosqlite.connect(
            str(backup_path), isolation_level=None
        )
        try:
            reader = _CursorReader(conn)
            return await reader.get_all()
        finally:
            await conn.close()

    async def diff(
        self, backup_path: Path
    ) -> Dict[str, object]:
        """
        Compare the live cursor database against a backup.

        Returns a dict describing:
          - manifest_ids in live but not in backup  (new_manifests)
          - manifest_ids in backup but not in live  (completed_since_backup)
          - manifest_ids in both with different positions  (advanced_positions)
          - manifest_ids in both with same position  (unchanged)

        Useful for verifying that checkpointing is working correctly after
        a period of crawling.

        Parameters
        ──────────
        backup_path     Path to a backup created by backup_to().

        Returns
        ───────
        Dict[str, object]   Diff report.

        Raises
        ──────
        RuntimeError    If cursor is not initialized.
        """
        self._cursor._ensure_initialized("diff") # noqa

        live_records = await self._cursor.all_active()
        backup_records = await self.inspect(backup_path)

        live_map = {r.manifest_id: r for r in live_records}
        backup_map = {r.manifest_id: r for r in backup_records}

        new_manifests = sorted(set(live_map) - set(backup_map))
        completed_since = sorted(set(backup_map) - set(live_map))
        advanced: List[Dict[str, object]] = []
        unchanged: List[str] = []

        for mid in set(live_map) & set(backup_map):
            live_pos = live_map[mid].position
            backup_pos = backup_map[mid].position
            if live_pos != backup_pos:
                advanced.append({
                    "manifest_id": mid,
                    "backup_position": backup_pos,
                    "live_position": live_pos,
                    "delta": live_pos - backup_pos,
                })
            else:
                unchanged.append(mid)

        return {
            "backup_path": str(backup_path),
            "live_cursor_count": len(live_records),
            "backup_cursor_count": len(backup_records),
            "new_manifests": new_manifests,
            "completed_since_backup": completed_since,
            "advanced_positions": advanced,
            "unchanged_count": len(unchanged),
        }


# ─────────────────────────────────────────────────────────────────────────────
# WAL HEALTH TRACKER
# Tracks WAL file growth over time to detect runaway WAL accumulation.
# A WAL file that is growing without being checkpointed indicates a reader
# holding a long-lived snapshot that is blocking the checkpoint.
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class WALHealthSample:
    """One WAL size measurement.  Internal to _WALHealthTracker."""

    sampled_at: float   # time.monotonic()
    wal_size_bytes: int
    main_size_bytes: int


class WALHealthTracker:
    """
    Tracks WAL file size over time to detect WAL growth anomalies.

    WAL growth without corresponding checkpointing indicates a long-lived
    reader that is holding a snapshot, preventing the WAL from being
    compacted.  In a single-process crawl, this should not happen — but it
    can occur if monitoring tools open the database with long-running
    transactions.

    Usage::

        tracker = WALHealthTracker(db_path=CURSOR_DB_PATH)
        sample = await tracker.sample()
        if tracker.is_growing_unboundedly():
            log.warning("WAL is growing — check for long-running readers")
    """

    # WAL size threshold above which growth is considered anomalous.
    # 16 MB is generous for a cursor database that has very few rows.
    WAL_WARN_BYTES: int = 16 * 1024 * 1024

    # How many samples to retain.
    SAMPLE_CAP: int = 60

    def __init__(self, db_path: Path = CURSOR_DB_PATH) -> None:
        """
        Construct a WALHealthTracker for the given database path.

        Parameters
        ──────────
        db_path     Path to the cursor SQLite file.  The WAL file is assumed
                    to be at db_path + '-wal' (the SQLite default).
        """
        self._db_path = db_path
        self._wal_path = db_path.with_suffix(db_path.suffix + "-wal")
        self._samples: List[WALHealthSample] = []

    async def sample(self) -> WALHealthSample:
        """
        Record the current WAL and main file sizes.

        Does not require a database connection — reads the file sizes directly
        from the filesystem.  Returns the sample for caller inspection.

        Non-existent files are reported as 0 bytes.
        """
        now = time.monotonic()
        wal_bytes = (
            self._wal_path.stat().st_size if self._wal_path.exists() else 0
        )
        main_bytes = (
            self._db_path.stat().st_size if self._db_path.exists() else 0
        )
        sample = WALHealthSample(
            sampled_at=now,
            wal_size_bytes=wal_bytes,
            main_size_bytes=main_bytes,
        )
        self._samples.append(sample)
        if len(self._samples) > self.SAMPLE_CAP:
            self._samples = self._samples[-self.SAMPLE_CAP:]

        if wal_bytes > self.WAL_WARN_BYTES:
            _log.warning(
                "WALHealthTracker: WAL file is %.1f MB — check for "
                "long-running readers blocking WAL checkpoint.",
                wal_bytes / (1024 * 1024),
            )

        return sample

    def is_growing_unboundedly(self) -> bool:
        """
        Return True if the WAL has grown monotonically over all retained samples.

        A monotonically growing WAL without any decrease indicates that no
        checkpoint has succeeded across the observation window.  This is an
        anomaly for the cursor database, which should checkpoint frequently.

        Returns False if fewer than 3 samples exist.
        """
        if len(self._samples) < 3:
            return False
        sizes = [s.wal_size_bytes for s in self._samples]
        # Check for strict monotonic increase.
        return all(sizes[i] < sizes[i + 1] for i in range(len(sizes) - 1))

    def current_wal_bytes(self) -> int:
        """Return the most recently sampled WAL size, or 0 if no samples."""
        if not self._samples:
            return 0
        return self._samples[-1].wal_size_bytes

    def peak_wal_bytes(self) -> int:
        """Return the maximum WAL size seen across all retained samples."""
        if not self._samples:
            return 0
        return max(s.wal_size_bytes for s in self._samples)

    def sample_count(self) -> int:
        """Return the number of samples currently retained."""
        return len(self._samples)

    def to_report(self) -> Dict[str, object]:
        """Export WAL health status as a flat dict."""
        return {
            "db_path": str(self._db_path),
            "wal_path": str(self._wal_path),
            "current_wal_bytes": self.current_wal_bytes(),
            "peak_wal_bytes": self.peak_wal_bytes(),
            "is_growing_unboundedly": self.is_growing_unboundedly(),
            "sample_count": self.sample_count(),
            "warn_threshold_bytes": self.WAL_WARN_BYTES,
        }


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT-MANAGED BATCH GRACEFUL SHUTDOWN
# A context manager for coordinating a graceful shutdown across multiple
# concurrent manifests.  On __aexit__, it atomically writes all pending
# cursor positions before process exit.
# ─────────────────────────────────────────────────────────────────────────────


class GracefulShutdownCheckpointer:
    """
    Coordinates atomic checkpoint-on-shutdown across concurrent manifests.

    The fetcher registers each active manifest with this object.  On
    graceful shutdown (SIGTERM, clean exit), the fetcher calls flush(),
    which atomically writes all accumulated positions in a single batch
    transaction.

    This is a belt-and-suspenders guarantee: checkpoint() is called during
    execution at every CURSOR_CHECKPOINT_INTERVAL.  This object captures
    the final partial-interval positions that have not yet been written.

    Usage::

        shutdown_cp = GracefulShutdownCheckpointer(cursor)

        # Register manifests as they start:
        shutdown_cp.register(manifest_id, total_urls)

        # Update position after each URL (cheap — in-memory only):
        shutdown_cp.update(manifest_id, position, url)

        # On shutdown:
        await shutdown_cp.flush()

    The flush() is also called automatically on __aexit__ in the context
    manager form::

        async with GracefulShutdownCheckpointer(cursor) as cp:
            cp.register(manifest_id, total_urls)
            # ... crawl loop ...
            cp.update(manifest_id, position, url)
        # flush() called automatically here
    """

    def __init__(self, cursor: CrawlCursor) -> None:
        """
        Construct a GracefulShutdownCheckpointer.

        Parameters
        ──────────
        cursor  An initialized CrawlCursor instance.
        """
        self._cursor = cursor
        # In-memory position buffer.  Updated after every URL.
        # Not thread-safe — asyncio single-threaded model only.
        self._positions: Dict[str, BatchCheckpointEntry] = {}

    def register(self, manifest_id: str, total_urls: int) -> None:
        """
        Register a manifest.  No I/O.

        Initializes the position buffer for this manifest to position -1
        (meaning no position has been accumulated yet).  The first call to
        update() will write position 0.

        Parameters
        ──────────
        manifest_id     UUID4 of the manifest.
        total_urls      Total URL count.
        """
        self._positions[manifest_id] = BatchCheckpointEntry(
            manifest_id=manifest_id,
            position=-1,
            url="",
            total_urls=total_urls,
        )

    def update(self, manifest_id: str, position: int, url: str) -> None:
        """
        Update the accumulated position for a manifest.  No I/O.

        This method is called after every URL fetch — it is designed to be
        extremely cheap (a dict write).  It does not write to SQLite.  The
        write happens in flush().

        If manifest_id was not registered, this call is a no-op — it does
        not raise.  This handles the case where update() is called before
        register() during startup.

        Parameters
        ──────────
        manifest_id     UUID4 of the manifest.
        position        Current URL index (0-based).
        url             The URL string at position.
        """
        if manifest_id not in self._positions:
            return
        old = self._positions[manifest_id]
        self._positions[manifest_id] = BatchCheckpointEntry(
            manifest_id=manifest_id,
            position=position,
            url=url,
            total_urls=old.total_urls,
        )

    def deregister(self, manifest_id: str) -> None:
        """
        Remove a manifest from the shutdown checkpointer.

        Call this when a manifest completes (after calling cursor.clear()).
        A completed manifest should not be checkpointed on shutdown — its
        cursor row no longer exists.

        Parameters
        ──────────
        manifest_id     UUID4 of the completed manifest.
        """
        self._positions.pop(manifest_id, None)

    async def flush(self) -> Optional[BatchCheckpointResult]:
        """
        Write all accumulated positions to the cursor database atomically.

        Only writes entries where position >= 0 (i.e., at least one URL has
        been processed since register() was called).

        Returns None if there are no pending positions to write (all manifests
        were registered but never updated, or all have been deregistered).

        Raises
        ──────
        CursorError     On SQLite batch write failure.
        RuntimeError    If the cursor is not initialized.
        """
        entries = [
            entry
            for entry in self._positions.values()
            if entry.position >= 0
        ]

        if not entries:
            _log.debug(
                "GracefulShutdownCheckpointer.flush: no pending positions."
            )
            return None

        _log.info(
            "GracefulShutdownCheckpointer.flush: writing %d cursor(s).",
            len(entries),
        )

        result = await self._cursor.checkpoint_batch(entries)

        _log.info(
            "GracefulShutdownCheckpointer.flush: wrote %d cursor(s) "
            "in %.2fms. success=%s",
            result.written_count,
            result.total_latency_ms,
            result.success,
        )
        return result

    async def __aenter__(self) -> "GracefulShutdownCheckpointer":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Flush on context exit (clean or exception)."""
        try:
            await self.flush()
        except CursorError as exc:
            _log.warning(
                "GracefulShutdownCheckpointer.__aexit__: flush failed: %s", exc
            )
        except Exception as exc:
            _log.warning(
                "GracefulShutdownCheckpointer.__aexit__: unexpected error: %s", exc
            )

    @property
    def registered_count(self) -> int:
        """Number of manifests currently registered."""
        return len(self._positions)

    @property
    def pending_count(self) -> int:
        """Number of manifests with position >= 0 (have pending writes)."""
        return sum(1 for e in self._positions.values() if e.position >= 0)


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
# Provides:
#   --test     Run the built-in diagnostic test suite.
#   --health   Print a health report for the cursor database at the default path.
#   --dump     Dump all active cursor rows as JSON.
# ─────────────────────────────────────────────────────────────────────────────


def _build_arg_parser():
    """Build the CLI argument parser.  Imported lazily to avoid import cost."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="crawl_cursor",
        description=(
            "crawl_cursor — interrupt-safe position tracking for the AXIOM crawler.\n"
            "Layer 1 acquisition only.  No routing logic.\n\n"
            "Subcommands:\n"
            "  --test     Run the built-in diagnostic test suite.\n"
            "  --health   Print health report for the cursor database.\n"
            "  --dump     Dump all active cursor rows as JSON.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run the built-in diagnostic test suite and exit.",
    )
    parser.add_argument(
        "--health",
        action="store_true",
        help="Print a health report for the cursor database.",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump all active cursor rows as JSON to stdout.",
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=CURSOR_DB_PATH,
        metavar="PATH",
        help=f"Path to cursor database.  Default: {CURSOR_DB_PATH}",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level.  Default: WARNING",
    )
    return parser


async def _cmd_test() -> int:
    """Run diagnostic test suite.  Return exit code."""
    runner = _DiagnosticRunner()
    passed = await runner.run_all()
    return 0 if passed else 1


async def _cmd_health(db_path: Path) -> int:
    """Print health report.  Return exit code."""
    import json

    if not db_path.exists():
        print(
            f"[ERROR] Database not found at {db_path}. "
            "Run the fetcher first to create the database."
        )
        return 1

    cursor = CrawlCursor(db_path=db_path)
    try:
        await cursor.initialize()
        health = await cursor.health()
        print(json.dumps(health.to_dict(), indent=2))
        return 0 if health.is_healthy else 1
    except Exception as exc:
        print(f"[ERROR] Health check failed: {exc}")
        return 1
    finally:
        await cursor.close()


async def _cmd_dump(db_path: Path) -> int:
    """Dump all cursor rows as JSON.  Return exit code."""
    import json

    if not db_path.exists():
        print("[]")
        return 0

    cursor = CrawlCursor(db_path=db_path)
    try:
        await cursor.initialize()
        snapshot = await cursor.export_snapshot()
        print(json.dumps(snapshot, indent=2))
        return 0
    except Exception as exc:
        print(f"[ERROR] Dump failed: {exc}")
        return 1
    finally:
        await cursor.close()


async def _async_main(args) -> int:
    """Async entry point for all CLI commands."""
    if args.test:
        return await _cmd_test()
    if args.health:
        return await _cmd_health(args.db)
    if args.dump:
        return await _cmd_dump(args.db)
    # No command — print help.
    _build_arg_parser().print_help()
    return 0


def main() -> None:
    """
    Synchronous entry point.  Parses args and delegates to async main.
    """
    import sys

    parser = _build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    exit_code = asyncio.run(_async_main(args))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()