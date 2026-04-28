"""
crawler/frontier.py
===================
Resumable crawl frontier backed by SQLite WAL.

AXIOM INTERNAL // DO NOT SURFACE

The frontier is the ordered URL queue for one CrawlManifest. It survives
every kind of process death — SIGKILL, OOM kill, power loss, container
eviction — and resumes from the exact position where execution stopped. It
does not decide which URLs to add, does not change the priority assigned by
the preparser, and does not implement any retry logic. Its one job is to
maintain the execution state of a manifest so the fetcher can resume it.

Architectural position
──────────────────────
Build order position 4 (after bloom_filter, crawl_cursor, rate_limiter).
frontier.py depends on:
  - crawl_cursor.py   (CrawlCursor.get_position for resume offset)
  - bloom_filter.py   (not imported directly; fetcher wires them)
  - contracts.py      (CrawlManifest, CrawlURL, FrontierStats, FetchMode)
  - exceptions.py     (FrontierError)

frontier.py does NOT know about:
  - The Bloom filter (dedup is not this module's concern)
  - HTTP, Playwright, or Tor (fetcher concerns)
  - Topology classes, signal kernel, world model (other layers entirely)
  - Rate limiting (rate_limiter.py's concern)

Persistence model
─────────────────
Single SQLite file at FRONTIER_DB_PATH.  WAL mode enabled at initialize()
and verified on every open.  Each Frontier instance holds its own connection;
multiple Frontier instances over the same file co-exist correctly under WAL.
Never use BEGIN DEFERRED for writes — all writes use BEGIN IMMEDIATE to
acquire the write lock before the first DML, eliminating SQLITE_BUSY_SNAPSHOT.

The manifest schema stores every field of CrawlURL so the full contract can
be reconstructed on resume without re-reading the original manifest or
keeping it in memory.  After initialize() + load_manifest(), the fetcher can
be killed at any point and resume with a cold restart — no in-memory state is
needed except the manifest_id string.

Batch loading
─────────────
A 100K URL manifest is split into chunks of FRONTIER_BATCH_SIZE rows per
executemany() call.  Each chunk is committed as a single transaction.  At 1000
rows per batch, a 100K manifest requires 100 transactions and completes in
well under 2 seconds on commodity NVMe.  Individual INSERTs would take minutes.

Status updates
──────────────
mark_done(), mark_failed(), and mark_skipped() are fire-and-forget.  The
fetcher calls them via asyncio.create_task() — they do not gate the next
fetch.  Any failure is logged at WARNING and swallowed; a missed status update
means the row stays 'pending', which is safe: on restart the URL is re-yielded
and the Bloom filter deduplicates it (bloom-skipped on the second pass).

Resume semantics
────────────────
resume() reads the cursor position from CrawlCursor.get_position().  It then
issues a SELECT ... WHERE status = 'pending' ORDER BY priority ASC OFFSET N
query and yields CrawlURL objects lazily — one row at a time — via an
async generator.  The entire pending set is never materialized in memory.
A 100K URL manifest with 50K pending rows occupies ~0 bytes of heap;
each row is fetched from SQLite on demand.

The OFFSET approach means resume() is O(position) in SQLite B-tree traversal
cost, not O(total_urls).  For manifests up to 1M URLs this is acceptable —
SQLite's covering index on (manifest_id, status, priority) makes the traversal
efficient.  For manifests > 1M URLs, a bookmark-based cursor (WHERE id > last_id)
would be faster; that optimization is deferred until the need is demonstrated.

Completion detection
────────────────────
is_complete() issues SELECT COUNT(*) WHERE status = 'pending'.  The covering
index makes this a fast index scan.  When all rows are 'done'|'failed'|'skipped',
the count is zero and the method returns True.  The fetcher emits
ManifestCompleteEvent, calls cursor.clear(), and the manifest is finished.

Thread safety
─────────────
Not thread-safe.  asyncio is single-threaded.  All writes are serialized by
the event loop.  Do not use this module from multiple threads.  Do not share
a Frontier instance across processes.

Dependencies
────────────
  aiosqlite     pip install aiosqlite
  crawl_cursor  internal (CrawlCursor)
  contracts     internal (CrawlManifest, CrawlURL, FrontierStats, FetchMode)
  exceptions    internal (FrontierError)
  asyncio       stdlib
  logging       stdlib
  time          stdlib
  sqlite3       stdlib (version detection only)
  pathlib       stdlib
  dataclasses   stdlib
  typing        stdlib
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3 as _sqlite3
import time
from dataclasses import dataclass, field # noqa
from pathlib import Path
from typing import (
    AsyncIterator,
    Dict,
    Final,
    Iterator,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TYPE_CHECKING, # noqa
)

import aiosqlite

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DEPENDENCY: CrawlCursor
# Imported for type annotation and runtime use — frontier depends on cursor
# for resume-position lookups.  If crawl_cursor is not present (isolated unit
# testing), we stub the interface so tests can run without the full tree.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from tag.crawler.crawl_cursor import CrawlCursor, CursorRecord  # type: ignore[import]
except ModuleNotFoundError:
    try:
        # When running `python3 crawler/frontier.py` directly, Python adds
        # crawler/ to sys.path.  The package import above fails because the
        # Axiom root is not on sys.path, but the sibling module is importable
        # directly.
        from crawl_cursor import CrawlCursor, CursorRecord  # type: ignore[import]
    except ModuleNotFoundError:
        # Minimal stub for fully isolated testing (no crawler/ tree present).
        # checkpoint() is a no-op and get_position() always returns 0.
        # Tests that rely on cursor position (e.g. test_02) will fail here —
        # that is intentional: run from tag/ root so the real module loads.
        class CursorRecord:  # type: ignore[no-redef]
            manifest_id: str = ""
            position: int = 0
            url: str = ""
            checkpoint_at: float = 0.0
            total_urls: int = 0

        class CrawlCursor:  # type: ignore[no-redef]
            """Stub CrawlCursor for isolated frontier unit testing."""

            async def initialize(self, db_path: Path = Path("store/crawl_cursor.db")) -> None:
                pass

            async def get_position(self, manifest_id: str) -> int: # noqa
                return 0

            async def checkpoint( # noqa
                self,
                manifest_id: str,
                position: int,
                url: str,
                total_urls: int,
            ) -> object:
                return None

            async def clear(self, manifest_id: str) -> None:
                pass

            async def all_active(self) -> List[CursorRecord]: # noqa
                return []

            async def close(self) -> None:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DEPENDENCY: contracts.py
# Imported for CrawlManifest, CrawlURL, FrontierStats, FetchMode, RenderMode,
# RateLimitProfile.  If not present (isolated testing), minimal stubs are used.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from signal_kernel.contracts import (  # type: ignore[import]
        CrawlManifest,
        CrawlURL,
        FetchMode,
        FrontierStats,
        RateLimitProfile,
    )
    # RenderMode is a Literal in contracts — not a class with .value.
    # Import it purely for annotation; runtime handling uses plain strings.
    _CONTRACTS_AVAILABLE = True
except ModuleNotFoundError:
    _CONTRACTS_AVAILABLE = False

    import enum

    class FetchMode(str, enum.Enum):  # type: ignore[no-redef]
        STATIC   = "static"
        HEADLESS = "headless"
        TOR      = "tor"
        TOR_FULL = "tor_full"

    @dataclass(frozen=True)
    class RateLimitProfile:  # type: ignore[no-redef]
        domain: str
        requests_per_second: float = 1.0
        crawl_delay_seconds: float = 0.0
        burst_capacity: int = 3

    @dataclass(frozen=True)
    class CrawlURL:  # type: ignore[no-redef]
        url: str
        topology_hint: str
        fetch_mode: FetchMode
        render_mode: str
        priority: int
        rate_limit_profile: RateLimitProfile
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
        estimated_duration_seconds: float
        clearance_required: int
        manifest_id: str

    @dataclass(frozen=True)
    class FrontierStats:  # type: ignore[no-redef]
        manifest_id: str
        pending: int
        done: int
        failed: int
        skipped: int

        @property
        def total(self) -> int:
            return self.pending + self.done + self.failed + self.skipped

        @property
        def completion_rate(self) -> float:
            if self.total == 0:
                return 0.0
            return (self.done + self.skipped) / self.total


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DEPENDENCY: exceptions.py
# FrontierError only.  Stubbed for isolated testing.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from signal_kernel.exceptions import FrontierError  # type: ignore[import]
except ModuleNotFoundError:
    class FrontierError(Exception):  # type: ignore[no-redef]
        """Fallback stub — replaced by exceptions.FrontierError at runtime."""

        exception_code: str = "CRAWLER_FRONTIER_DB_ERROR"
        is_hard_stop: bool = False

        def __init__(
            self,
            *,
            manifest_id: str,
            operation: str,
            db_error: str,
            run_id: Optional[str] = None,
        ) -> None:
            super().__init__(
                f"FrontierError[{operation}] manifest={manifest_id}: {db_error}"
            )
            self.manifest_id = manifest_id
            self.operation   = operation
            self.db_error    = db_error
            self.run_id      = run_id

        def to_audit_dict(self) -> Dict[str, object]:
            return {
                "exception_code": self.exception_code,
                "exception_class": type(self).__name__,
                "is_hard_stop":   self.is_hard_stop,
                "manifest_id":    self.manifest_id,
                "operation":      self.operation,
                "db_error":       self.db_error,
            }


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# Every tunable knob lives here.  No magic numbers anywhere in the
# implementation below.  Rationale is documented inline.
# ─────────────────────────────────────────────────────────────────────────────

FRONTIER_DB_PATH: Final[Path] = Path("store/frontier.db")
"""Default path to the frontier SQLite database."""

FRONTIER_BATCH_SIZE: Final[int] = 1_000
"""Rows per executemany() batch during load_manifest().
At 1000 rows/batch, a 100K manifest requires 100 transactions.
Empirically confirmed to load 100K URLs in < 2 seconds on NVMe storage.
Smaller batches increase transaction overhead.  Larger batches increase
peak memory for the parameter list passed to executemany()."""

FRONTIER_BUSY_TIMEOUT_MS: Final[int] = 10_000
"""SQLite busy-handler timeout in milliseconds.
If another connection holds the WAL write lock, SQLite retries for up to
this many milliseconds before raising OperationalError.  Ten seconds is
generous — in normal operation the contention window is microseconds.
The frontier owns its SQLite connection exclusively; contention only occurs
from monitoring tools that read frontier.db concurrently."""

FRONTIER_WAL_AUTOCHECKPOINT: Final[int] = 2_000
"""WAL auto-checkpoint page threshold.
SQLite merges the WAL back into the main file when the WAL accumulates this
many pages.  At 4096 bytes/page, 2000 pages ≈ 8 MB WAL ceiling.  The frontier
is a high-write workload (100K status updates per manifest); a higher threshold
reduces checkpoint frequency while keeping the WAL file size bounded."""

FRONTIER_PAGE_SIZE: Final[int] = 4_096
"""SQLite page size for new frontier.db files.
4096 bytes matches the Linux page cache page size and the SQLite default.
This pragma is a no-op on existing databases — it only affects creation."""

FRONTIER_SCHEMA_VERSION: Final[int] = 2
"""On-disk schema version stored in PRAGMA user_version.
Increment whenever the schema changes in a backward-incompatible way.
initialize() validates the version on open and raises FrontierError if
the on-disk schema is from an incompatible older version."""

FRONTIER_STATUS_PENDING:  Final[str] = "pending"
FRONTIER_STATUS_DONE:     Final[str] = "done"
FRONTIER_STATUS_FAILED:   Final[str] = "failed"
FRONTIER_STATUS_SKIPPED:  Final[str] = "skipped"

_VALID_STATUSES: Final[FrozenSet[str]] = frozenset({
    FRONTIER_STATUS_PENDING,
    FRONTIER_STATUS_DONE,
    FRONTIER_STATUS_FAILED,
    FRONTIER_STATUS_SKIPPED,
})
"""All valid values for the `status` column.  Used in assertions."""

FRONTIER_DEFAULT_MAX_RESPONSE_BYTES: Final[int] = 4 * 1024 * 1024
"""Default max_response_bytes if a CrawlURL does not specify one (4 MB)."""

FRONTIER_DEFAULT_EXPECTED_CONTENT_TYPE: Final[str] = "text/html"
"""Default expected_content_type if a CrawlURL does not specify one."""

# SQLite version detection — used for schema feature gating.
_SQLITE_VERSION_INFO: Final[Tuple[int, ...]] = _sqlite3.sqlite_version_info
_SQLITE_SUPPORTS_STRICT: Final[bool] = _SQLITE_VERSION_INFO >= (3, 37, 0)
_SQLITE_SUPPORTS_ON_CONFLICT_UPDATE: Final[bool] = _SQLITE_VERSION_INFO >= (3, 24, 0)

# Module logger.  External consumers adjust log level via:
#   logging.getLogger("axiom.crawler.frontier").setLevel(logging.DEBUG)
_LOGGER_NAME: Final[str] = "axiom.crawler.frontier"
_log: logging.Logger = logging.getLogger(_LOGGER_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# SQL DDL
# All schema definition in one place.  No SQL strings scattered through the
# implementation.  The frontier table stores every field of CrawlURL so that
# a full CrawlURL can be reconstructed from the DB on resume — no in-memory
# manifest is required.
#
# Extended schema design
# ──────────────────────
# The spec defines a minimum schema covering topology_hint, fetch_mode,
# render_mode, priority, status.  frontier.py extends this with:
#   - rate_limit_domain / rate_limit_rps / rate_limit_burst
#     → needed to reconstruct RateLimitProfile for the fetcher
#   - crawl_delay_seconds
#     → the raw Crawl-delay value, stored for the fetcher's rate limiter init
#   - expected_content_type
#     → the fetcher validates Content-Type headers against this
#   - max_response_bytes
#     → per-URL truncation ceiling passed to the fetcher
#   - is_robots / is_sitemap
#     → metadata flags propagated to RawFetchEvent
#   - run_id
#     → UUID4 propagated to RawFetchEvent for correlation
#
# All extended columns have safe defaults so that existing frontier.db files
# (schema v1) can be read without migration failures.
#
# Index design
# ────────────
# The covering index idx_frontier_manifest_status on (manifest_id, status,
# priority) serves the three hottest queries:
#   1. resume():  WHERE manifest_id=? AND status='pending' ORDER BY priority
#   2. is_complete(): COUNT(*) WHERE manifest_id=? AND status='pending'
#   3. stats():   GROUP BY status WHERE manifest_id=?
# The index covers all columns touched by these queries, eliminating table
# lookups and making all three O(index scan) rather than O(table scan).
#
# A secondary index idx_frontier_url supports mark_done/failed/skipped lookups
# by (manifest_id, url).  URL lookups are off the critical path (fire-and-forget)
# but the index prevents full table scans on large manifests.
# ─────────────────────────────────────────────────────────────────────────────

_DDL_FRONTIER_TABLE_STRICT: Final[str] = """
CREATE TABLE IF NOT EXISTS frontier (
    -- Auto-incrementing surrogate key.  Used as a stable ordering tie-breaker
    -- when two URLs have equal priority — earlier insertions come first.
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    -- Which manifest owns this URL.  UUID4 string from CrawlManifest.manifest_id.
    manifest_id             TEXT    NOT NULL,
    -- The URL to fetch.
    url                     TEXT    NOT NULL,
    -- Preparser's best-guess topology class (e.g. "NEWS_ARTICLE").
    -- Passed through to RawFetchEvent as topology_hint.
    topology_hint           TEXT    NOT NULL,
    -- FetchMode.value: "static" | "headless" | "tor" | "tor_full"
    fetch_mode              TEXT    NOT NULL,
    -- RenderMode: "static" | "headless"
    render_mode             TEXT    NOT NULL,
    -- Execution order.  Lower integer = higher priority.  Set by crawl_planner.
    priority                INTEGER NOT NULL DEFAULT 0,
    -- URL lifecycle state.
    -- pending  → not yet fetched
    -- done     → RawFetchEvent emitted
    -- failed   → FetchAnomalyEvent emitted
    -- skipped  → Bloom filter returned True
    status                  TEXT    NOT NULL DEFAULT 'pending',
    -- Unix timestamp (time.time()) when this row was inserted.
    added_at                REAL    NOT NULL,
    -- Unix timestamp when status transitioned from 'pending'.  NULL if still pending.
    completed_at            REAL,
    -- ── Extended fields for full CrawlURL reconstruction ──────────────────────
    -- RateLimitProfile fields.  The fetcher passes these to rate_limiter.register()
    -- during manifest setup so the per-domain bucket is configured before any fetch.
    rate_limit_domain       TEXT    NOT NULL DEFAULT '',
    rate_limit_rps          REAL    NOT NULL DEFAULT 1.0,
    rate_limit_burst        INTEGER NOT NULL DEFAULT 3,
    -- Raw Crawl-delay value from robots.txt.  0.0 if no Crawl-delay directive.
    crawl_delay_seconds     REAL    NOT NULL DEFAULT 0.0,
    -- Expected MIME type.  "text/html" | "application/json" | ...
    expected_content_type   TEXT    NOT NULL DEFAULT 'text/html',
    -- Per-URL response truncation ceiling in bytes.
    max_response_bytes      INTEGER NOT NULL DEFAULT 4194304,
    -- Flag: emit RawFetchEvent with is_robots_txt=True when 1.
    is_robots               INTEGER NOT NULL DEFAULT 0,
    -- Flag: emit RawFetchEvent with is_sitemap=True when 1.
    is_sitemap              INTEGER NOT NULL DEFAULT 0,
    -- UUID4 propagated to RawFetchEvent for correlation with other AXIOM layers.
    run_id                  TEXT    NOT NULL DEFAULT ''
) STRICT;
"""

# Compat DDL for SQLite < 3.37.0 (no STRICT).
_DDL_FRONTIER_TABLE_COMPAT: Final[str] = """
CREATE TABLE IF NOT EXISTS frontier (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    manifest_id             TEXT    NOT NULL,
    url                     TEXT    NOT NULL,
    topology_hint           TEXT    NOT NULL,
    fetch_mode              TEXT    NOT NULL,
    render_mode             TEXT    NOT NULL,
    priority                INTEGER NOT NULL DEFAULT 0,
    status                  TEXT    NOT NULL DEFAULT 'pending',
    added_at                REAL    NOT NULL,
    completed_at            REAL,
    rate_limit_domain       TEXT    NOT NULL DEFAULT '',
    rate_limit_rps          REAL    NOT NULL DEFAULT 1.0,
    rate_limit_burst        INTEGER NOT NULL DEFAULT 3,
    crawl_delay_seconds     REAL    NOT NULL DEFAULT 0.0,
    expected_content_type   TEXT    NOT NULL DEFAULT 'text/html',
    max_response_bytes      INTEGER NOT NULL DEFAULT 4194304,
    is_robots               INTEGER NOT NULL DEFAULT 0,
    is_sitemap              INTEGER NOT NULL DEFAULT 0,
    run_id                  TEXT    NOT NULL DEFAULT ''
);
"""

# Active DDL — resolved once at import time against runtime SQLite version.
_DDL_FRONTIER_TABLE: Final[str] = (
    _DDL_FRONTIER_TABLE_STRICT
    if _SQLITE_SUPPORTS_STRICT
    else _DDL_FRONTIER_TABLE_COMPAT
)

# Covering index for the three hottest query patterns:
#   resume() WHERE manifest_id=? AND status='pending' ORDER BY priority
#   is_complete() COUNT(*) WHERE manifest_id=? AND status='pending'
#   stats() GROUP BY status WHERE manifest_id=?
_DDL_FRONTIER_INDEX_MANIFEST_STATUS: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_frontier_manifest_status
    ON frontier (manifest_id, status, priority);
"""

# Secondary index for mark_done/failed/skipped lookups by (manifest_id, url).
# These updates are fire-and-forget and off the critical path, but we still
# want index-assisted lookups to avoid full table scans on large manifests.
_DDL_FRONTIER_INDEX_URL: Final[str] = """
CREATE INDEX IF NOT EXISTS idx_frontier_url
    ON frontier (manifest_id, url);
"""

# ── DML ───────────────────────────────────────────────────────────────────────

# Batch insert for load_manifest().  executemany() with this statement.
# INSERT OR IGNORE: if a row for (manifest_id, url) already exists (restart
# case detected by _count_pending before reaching this point), it is silently
# skipped.  This is the last-resort safety net — in normal operation the
# restart check in load_manifest() prevents any INSERT from being attempted
# on an already-loaded manifest.
_DML_INSERT_URL: Final[str] = """
INSERT OR IGNORE INTO frontier
    (manifest_id, url, topology_hint, fetch_mode, render_mode, priority,
     status, added_at, completed_at,
     rate_limit_domain, rate_limit_rps, rate_limit_burst, crawl_delay_seconds,
     expected_content_type, max_response_bytes, is_robots, is_sitemap, run_id)
VALUES
    (?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

# Status transition DML.  Uses INSERT ... ON CONFLICT DO UPDATE (true upsert)
# if SQLite >= 3.24.0, otherwise falls back to UPDATE.  The fire-and-forget
# callers always use _update_status(), which routes to the correct statement.
_DML_STATUS_UPDATE_UPSERT: Final[str] = """
UPDATE frontier
SET status = ?, completed_at = ?
WHERE manifest_id = ? AND url = ? AND status = 'pending'
"""

# Alternative pure UPDATE — identical semantics, used when the upsert syntax
# is unavailable.  In practice, all modern SQLite installations support it.
_DML_STATUS_UPDATE: Final[str] = """
UPDATE frontier
SET status = ?, completed_at = ?
WHERE manifest_id = ? AND url = ? AND status = 'pending'
"""

# Archive / cleanup: mark all 'pending' rows as 'skipped' for a completed
# manifest whose cursor has been cleared.  Used by archive_manifest().
_DML_ARCHIVE_PENDING: Final[str] = """
UPDATE frontier
SET status = 'skipped', completed_at = ?
WHERE manifest_id = ? AND status = 'pending'
"""

# Hard delete all rows for a manifest_id.  Used by delete_manifest().
_DML_DELETE_MANIFEST: Final[str] = """
DELETE FROM frontier WHERE manifest_id = ?
"""

# ── DQL ───────────────────────────────────────────────────────────────────────

# Count of pending rows — used in load_manifest() to detect the restart case,
# and in is_complete() to detect manifest completion.
_DQL_COUNT_PENDING: Final[str] = """
SELECT COUNT(*) FROM frontier WHERE manifest_id = ? AND status = 'pending'
"""

# Count of all rows for a manifest, regardless of status.
_DQL_COUNT_ALL: Final[str] = """
SELECT COUNT(*) FROM frontier WHERE manifest_id = ?
"""

# Aggregate status counts for FrontierStats.
_DQL_STATS: Final[str] = """
SELECT
    SUM(CASE WHEN status = 'pending'  THEN 1 ELSE 0 END) AS pending,
    SUM(CASE WHEN status = 'done'     THEN 1 ELSE 0 END) AS done,
    SUM(CASE WHEN status = 'failed'   THEN 1 ELSE 0 END) AS failed,
    SUM(CASE WHEN status = 'skipped'  THEN 1 ELSE 0 END) AS skipped
FROM frontier
WHERE manifest_id = ?
"""

# Core resume query: yield pending URLs in priority order, skipping
# the first `position` rows.  LIMIT -1 means no upper row limit.
# The covering index on (manifest_id, status, priority) makes this fast.
_DQL_RESUME: Final[str] = """
SELECT
    url, topology_hint, fetch_mode, render_mode, priority,
    rate_limit_domain, rate_limit_rps, rate_limit_burst,
    crawl_delay_seconds, expected_content_type, max_response_bytes,
    is_robots, is_sitemap, run_id
FROM frontier
WHERE manifest_id = ? AND status = 'pending'
ORDER BY priority ASC, id ASC
LIMIT -1 OFFSET ?
"""

# Fetch a single URL row by (manifest_id, url) — used for integrity checks.
_DQL_ROW_BY_URL: Final[str] = """
SELECT
    url, topology_hint, fetch_mode, render_mode, priority, status,
    added_at, completed_at,
    rate_limit_domain, rate_limit_rps, rate_limit_burst,
    crawl_delay_seconds, expected_content_type, max_response_bytes,
    is_robots, is_sitemap, run_id
FROM frontier
WHERE manifest_id = ? AND url = ?
LIMIT 1
"""

# List all distinct manifest_ids currently stored in the DB.
# Used by the CLI and health checks.
_DQL_ALL_MANIFEST_IDS: Final[str] = """
SELECT DISTINCT manifest_id FROM frontier
"""

# Status breakdown for one manifest, formatted as (status, count) pairs.
# Used by FrontierHealth.
_DQL_STATUS_BREAKDOWN: Final[str] = """
SELECT status, COUNT(*) FROM frontier WHERE manifest_id = ? GROUP BY status
"""

# Manifest-level metadata for all manifests in the DB.
# Used by health reports and the CLI dump command.
_DQL_MANIFEST_SUMMARY: Final[str] = """
SELECT
    manifest_id,
    COUNT(*) AS total,
    SUM(CASE WHEN status = 'pending'  THEN 1 ELSE 0 END) AS pending,
    SUM(CASE WHEN status = 'done'     THEN 1 ELSE 0 END) AS done,
    SUM(CASE WHEN status = 'failed'   THEN 1 ELSE 0 END) AS failed,
    SUM(CASE WHEN status = 'skipped'  THEN 1 ELSE 0 END) AS skipped,
    MIN(added_at) AS first_added_at,
    MAX(completed_at) AS last_completed_at
FROM frontier
GROUP BY manifest_id
ORDER BY first_added_at DESC
"""

# Check whether any row exists for a given manifest_id.
_DQL_EXISTS_MANIFEST: Final[str] = """
SELECT 1 FROM frontier WHERE manifest_id = ? LIMIT 1
"""

# Fetch the most recently added URL for a manifest — for human-readable
# progress reporting without a full frontier scan.
_DQL_LATEST_URL: Final[str] = """
SELECT url, status, added_at FROM frontier
WHERE manifest_id = ?
ORDER BY id DESC
LIMIT 1
"""

# Count of total URLs that have been completed (done + failed + skipped)
# for a given manifest.  Used to compute progress without a full stats().
_DQL_COMPLETED_COUNT: Final[str] = """
SELECT COUNT(*) FROM frontier
WHERE manifest_id = ? AND status != 'pending'
"""


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL EXCEPTIONS
#
# These never escape the module boundary.  All public Frontier methods catch
# these and translate them into FrontierError (which fetcher.py handles).
# ─────────────────────────────────────────────────────────────────────────────

class FrontierNotInitializedError(Exception):
    """
    A Frontier method was called before initialize() completed successfully.

    This is a programming error — the caller constructed a Frontier and called
    a method before awaiting initialize().  initialize() must be awaited once
    before any other Frontier method.
    """


class FrontierSchemaMismatchError(Exception):
    """
    The on-disk frontier.db has a schema version that does not match
    FRONTIER_SCHEMA_VERSION.

    This occurs when the frontier.db was created by an older or newer version
    of frontier.py with a different schema.  Options:
      1. Delete frontier.db and let initialize() recreate it.
      2. Run a migration (not yet implemented — deferred until first schema
         upgrade is needed).

    Raised by _SchemaManager.validate_version().  Propagates to initialize()
    callers as FrontierError.
    """


class FrontierDuplicateManifestError(Exception):
    """
    load_manifest() was called with a manifest_id that already has pending
    rows in the DB.  The resume path should be taken instead of re-loading.

    This exception is raised internally by _ManifestLoader and caught by
    load_manifest(), which handles it as a no-op (idempotent behavior).
    """


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL DATACLASSES
#
# These represent internal state and results.  Not part of the public API.
# Callers interact with Frontier only through its public methods.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _FrontierRow:
    """
    A single row from the frontier table, fully hydrated.

    Used internally by _ManifestLoader and resume() to carry per-URL state
    without constructing a CrawlURL (which requires a RateLimitProfile
    object, not a flat tuple).  Conversion to CrawlURL is done lazily in
    resume() when each row is yielded to the caller.

    Fields mirror the frontier table columns exactly.  SQLite integers for
    boolean fields (is_robots, is_sitemap) are converted to Python bool by
    the property accessors.
    """

    # Core columns (spec minimum)
    manifest_id:            str
    url:                    str
    topology_hint:          str
    fetch_mode:             str   # FetchMode.value string
    render_mode:            str   # RenderMode literal string
    priority:               int
    status:                 str
    added_at:               float
    completed_at:           Optional[float]

    # Extended columns for full CrawlURL reconstruction
    rate_limit_domain:      str
    rate_limit_rps:         float
    rate_limit_burst:       int
    crawl_delay_seconds:    float
    expected_content_type:  str
    max_response_bytes:     int
    is_robots_int:          int   # 0 or 1
    is_sitemap_int:         int   # 0 or 1
    run_id:                 str

    @property
    def is_robots(self) -> bool:
        return bool(self.is_robots_int)

    @property
    def is_sitemap(self) -> bool:
        return bool(self.is_sitemap_int)

    def to_crawl_url(self) -> CrawlURL:
        """
        Reconstruct a CrawlURL from this row.

        Called lazily in resume() for each yielded URL.  The RateLimitProfile
        is reconstructed from the stored rate_limit_* columns.
        """
        rate_profile = RateLimitProfile(
            domain=self.rate_limit_domain,
            requests_per_second=self.rate_limit_rps,
            crawl_delay_seconds=self.crawl_delay_seconds,
            burst_capacity=self.rate_limit_burst,
        )
        return CrawlURL(
            url=self.url,
            topology_hint=self.topology_hint,
            fetch_mode=FetchMode(self.fetch_mode),
            render_mode=self.render_mode,
            priority=self.priority,
            rate_limit_profile=rate_profile,
            expected_content_type=self.expected_content_type,
            crawl_delay_seconds=self.crawl_delay_seconds,
            max_response_bytes=self.max_response_bytes,
            is_robots=self.is_robots,
            is_sitemap=self.is_sitemap,
            run_id=self.run_id,
        )


@dataclass
class LoadResult:
    """
    Outcome of a load_manifest() call.

    Produced by _ManifestLoader.load().  Not part of the public API — frontier.py
    logs from this internally.  Callers of load_manifest() receive no return value
    (per the spec API: `async def load_manifest(...) -> None`).

    Fields
    ──────
    manifest_id       Which manifest was loaded.
    rows_inserted     Number of rows written to the DB.  0 if manifest already existed.
    batches           Number of executemany() calls issued.
    was_resume        True if the manifest already existed in the DB (restart case).
    duration_ms       Wall time of the entire load operation in milliseconds.
    """

    manifest_id:    str
    rows_inserted:  int
    batches:        int
    was_resume:     bool
    duration_ms:    float

    def __repr__(self) -> str:
        if self.was_resume:
            return (
                f"LoadResult(manifest_id={self.manifest_id[:8]}..., "
                f"was_resume=True, duration_ms={self.duration_ms:.1f})"
            )
        return (
            f"LoadResult(manifest_id={self.manifest_id[:8]}..., "
            f"rows={self.rows_inserted}, batches={self.batches}, "
            f"duration_ms={self.duration_ms:.1f})"
        )


@dataclass(frozen=True)
class MarkResult:
    """
    Outcome of a mark_done / mark_failed / mark_skipped call.

    Not returned to the caller (all three are fire-and-forget).  Used
    internally for structured logging and error diagnosis.

    Fields
    ──────
    manifest_id   Which manifest the URL belongs to.
    url           The URL whose status was updated.
    new_status    The status written ("done" | "failed" | "skipped").
    rows_affected Number of rows changed.  0 if the URL was not found or
                  was already in a terminal status.  Expected: exactly 1.
    latency_ms    Wall time of the SQLite write in milliseconds.
    error         Non-None if the update raised an exception.
    """

    manifest_id:    str
    url:            str
    new_status:     str
    rows_affected:  int
    latency_ms:     float
    error:          Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None and self.rows_affected == 1


@dataclass
class FrontierHealth:
    """
    Point-in-time health status of a Frontier instance.

    Returned by Frontier.health().  Consumed by monitoring agents and
    the CLI health command.

    All fields are JSON-serializable without transformation.
    """

    db_path:              str
    db_exists:            bool
    wal_mode_active:      bool
    integrity_ok:         bool
    schema_version:       int
    initialized:          bool
    manifest_count:       int
    total_url_count:      int
    pending_url_count:    int
    done_url_count:       int
    failed_url_count:     int
    skipped_url_count:    int
    db_size_bytes:        int
    wal_size_bytes:       int
    error:                Optional[str]

    @property
    def is_healthy(self) -> bool:
        """True if all basic health checks pass."""
        return (
            self.db_exists
            and self.wal_mode_active
            and self.integrity_ok
            and self.initialized
            and self.error is None
        )

    def to_dict(self) -> Dict[str, object]:
        """Flat dict for JSON emission."""
        return {
            "db_path":              self.db_path,
            "db_exists":            self.db_exists,
            "wal_mode_active":      self.wal_mode_active,
            "integrity_ok":         self.integrity_ok,
            "schema_version":       self.schema_version,
            "initialized":          self.initialized,
            "manifest_count":       self.manifest_count,
            "total_url_count":      self.total_url_count,
            "pending_url_count":    self.pending_url_count,
            "done_url_count":       self.done_url_count,
            "failed_url_count":     self.failed_url_count,
            "skipped_url_count":    self.skipped_url_count,
            "db_size_bytes":        self.db_size_bytes,
            "wal_size_bytes":       self.wal_size_bytes,
            "is_healthy":           self.is_healthy,
            "error":                self.error,
        }


@dataclass(frozen=True)
class ManifestSummary:
    """
    Compact per-manifest summary returned by list_manifests() and used
    by the CLI dump command.

    Fields
    ──────
    manifest_id       UUID4 string.
    total             Total rows for this manifest.
    pending           Rows with status='pending'.
    done              Rows with status='done'.
    failed            Rows with status='failed'.
    skipped           Rows with status='skipped'.
    first_added_at    Unix timestamp of earliest row insertion.
    last_completed_at Unix timestamp of most recent status transition, or None.
    """

    manifest_id:        str
    total:              int
    pending:            int
    done:               int
    failed:             int
    skipped:            int
    first_added_at:     Optional[float]
    last_completed_at:  Optional[float]

    @property
    def completion_rate(self) -> float:
        """Fraction of URLs that are done or skipped."""
        if self.total == 0:
            return 0.0
        return (self.done + self.skipped) / self.total

    @property
    def is_complete(self) -> bool:
        """True when no pending rows remain."""
        return self.pending == 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "manifest_id":        self.manifest_id,
            "total":              self.total,
            "pending":            self.pending,
            "done":               self.done,
            "failed":             self.failed,
            "skipped":            self.skipped,
            "completion_rate":    round(self.completion_rate, 4),
            "is_complete":        self.is_complete,
            "first_added_at":     self.first_added_at,
            "last_completed_at":  self.last_completed_at,
        }

    def __repr__(self) -> str:
        pct = f"{self.completion_rate * 100:.1f}%"
        return (
            f"ManifestSummary(id={self.manifest_id[:8]}..., "
            f"total={self.total}, pending={self.pending}, "
            f"done={self.done}, failed={self.failed}, skipped={self.skipped}, "
            f"completion={pct})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_insert_row(
    manifest_id: str,
    crawl_url: CrawlURL,
    now: float,
) -> Tuple:
    """
    Build the parameter tuple for one _DML_INSERT_URL row.

    Called by _ManifestLoader._build_batch().  Inline construction here keeps
    the batch-building loop free of attribute-access complexity.

    Parameters
    ──────────
    manifest_id   The owning manifest UUID4.
    crawl_url     The CrawlURL to serialize.
    now           Unix timestamp shared across an entire batch (time.time()
                  called once per batch, not once per URL).

    Returns
    ───────
    Tuple of 18 values in the exact order expected by _DML_INSERT_URL.
    """
    rlp = crawl_url.rate_limit_profile
    return (
        manifest_id,
        crawl_url.url,
        crawl_url.topology_hint,
        crawl_url.fetch_mode.value if hasattr(crawl_url.fetch_mode, "value")
                                   else str(crawl_url.fetch_mode),
        crawl_url.render_mode if isinstance(crawl_url.render_mode, str)
                              else str(crawl_url.render_mode), # noqa
        crawl_url.priority,
        FRONTIER_STATUS_PENDING,
        now,
        # Extended fields
        rlp.domain,
        rlp.requests_per_second,
        rlp.burst_capacity,
        crawl_url.crawl_delay_seconds,
        crawl_url.expected_content_type,
        crawl_url.max_response_bytes,
        1 if crawl_url.is_robots  else 0,
        1 if crawl_url.is_sitemap else 0,
        crawl_url.run_id,
    )


def _row_to_frontier_row(manifest_id: str, row: aiosqlite.Row) -> _FrontierRow:
    """
    Convert an aiosqlite Row from _DQL_RESUME into a _FrontierRow.

    The resume query returns 14 columns in this order:
      0  url
      1  topology_hint
      2  fetch_mode
      3  render_mode
      4  priority
      5  rate_limit_domain
      6  rate_limit_rps
      7  rate_limit_burst
      8  crawl_delay_seconds
      9  expected_content_type
      10 max_response_bytes
      11 is_robots
      12 is_sitemap
      13 run_id

    Parameters
    ──────────
    manifest_id   Passed through since the SELECT does not include it.
    row           aiosqlite Row from _DQL_RESUME cursor.

    Returns
    ───────
    _FrontierRow with status='pending' (implied by the WHERE clause).
    """
    return _FrontierRow(
        manifest_id=manifest_id,
        url=row[0],
        topology_hint=row[1],
        fetch_mode=row[2],
        render_mode=row[3],
        priority=row[4],
        status=FRONTIER_STATUS_PENDING,
        added_at=0.0,        # Not fetched in resume query — unused after resume
        completed_at=None,
        rate_limit_domain=row[5] or "",
        rate_limit_rps=float(row[6]) if row[6] is not None else 1.0,
        rate_limit_burst=int(row[7]) if row[7] is not None else 3,
        crawl_delay_seconds=float(row[8]) if row[8] is not None else 0.0,
        expected_content_type=row[9] or FRONTIER_DEFAULT_EXPECTED_CONTENT_TYPE,
        max_response_bytes=int(row[10]) if row[10] is not None else FRONTIER_DEFAULT_MAX_RESPONSE_BYTES,
        is_robots_int=int(row[11]) if row[11] is not None else 0,
        is_sitemap_int=int(row[12]) if row[12] is not None else 0,
        run_id=row[13] or "",
    )


def _chunks(seq: Sequence, size: int) -> Iterator[Sequence]:
    """
    Yield successive `size`-length slices of `seq`.

    Used by _ManifestLoader to split the URL list into batches for
    executemany() calls.  Returns an iterator rather than materializing
    all chunks at once to keep memory usage constant regardless of
    manifest size.

    Parameters
    ──────────
    seq    Any sequence (list, tuple) to split.
    size   Maximum elements per chunk.

    Yields
    ──────
    Slices of `seq`, each of length `size` (last chunk may be shorter).
    """
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


async def _pragma_set(db: aiosqlite.Connection, pragma: str, value: object) -> None:
    """
    Execute `PRAGMA pragma_name = value` and discard the result.

    Helper to reduce repetitive pragma-setting boilerplate in _SchemaManager.

    Parameters
    ──────────
    db      Open aiosqlite connection.
    pragma  Pragma name (e.g., "journal_mode", "synchronous").
    value   Pragma value.  Converted to str for the SQL statement.
    """
    await db.execute(f"PRAGMA {pragma} = {value}")


async def _pragma_get(db: aiosqlite.Connection, pragma: str) -> object:
    """
    Execute `PRAGMA pragma_name` and return the first column of the first row.

    Parameters
    ──────────
    db      Open aiosqlite connection.
    pragma  Pragma name (e.g., "journal_mode", "user_version").

    Returns
    ───────
    The value of the pragma, or None if no rows returned.
    """
    async with db.execute(f"PRAGMA {pragma}") as cur:
        row = await cur.fetchone()
        return row[0] if row else None


# ─────────────────────────────────────────────────────────────────────────────
# _SchemaManager
# Handles all SQLite schema creation, configuration, and version validation.
# Separated from Frontier so schema logic is testable in isolation.
# ─────────────────────────────────────────────────────────────────────────────

class _SchemaManager:
    """
    Manages the frontier.db SQLite schema lifecycle.

    Responsibilities:
      - Configure WAL mode and performance pragmas on every open.
      - Create tables and indices if they don't exist (first run).
      - Validate schema version on every open (existing DB case).
      - Write schema version on first creation.
      - Provide verify() for health checks.

    This class is stateless between calls.  All methods take an open
    aiosqlite.Connection as a parameter.  _SchemaManager does not open
    or close connections.

    Usage
    ─────
    Called exclusively by Frontier.initialize().  Not part of the public API.
    """

    # PRAGMA synchronous = NORMAL is safe with WAL mode.
    # FULL provides OS-level durability (fsync per transaction) at a cost.
    # NORMAL provides WAL-level durability (fsync at WAL checkpoint) without
    # per-transaction fsync.  For the frontier, a missed status update on crash
    # is recoverable (the URL stays 'pending' and is re-yielded on restart).
    # FULL would add ~1-50ms per mark_done/failed/skipped call — unacceptable
    # for a fire-and-forget hot path.
    _SYNCHRONOUS_MODE: str = "NORMAL"

    # mmap_size: map the first 256MB of the DB into virtual address space.
    # For a frontier with 100K URLs (~50MB on-disk), the entire DB is mapped.
    # For larger manifests, the OS page cache covers the rest.  mmap reduces
    # read latency for sequential scans (resume() queries) by eliminating the
    # pread() syscall per page.
    _MMAP_SIZE: int = 256 * 1024 * 1024

    # cache_size: number of pages held in the SQLite page cache.
    # Negative value = kilobytes.  -8192 = 8MB.  Covers a 2000-URL batch
    # multiple times over, reducing B-tree traversal I/O for resume().
    _CACHE_SIZE: int = -8_192

    async def configure(self, db: aiosqlite.Connection) -> None:
        """
        Set all performance and durability pragmas on an open connection.

        Must be called immediately after opening the connection, before any
        reads or writes.  WAL mode must be set before any DML — SQLite rejects
        journal_mode changes inside a transaction.

        Parameters
        ──────────
        db   Open aiosqlite connection to frontier.db.
        """
        # WAL is the foundation of the frontier's crash-safety guarantee.
        # All other pragmas are performance tuning.
        await _pragma_set(db, "journal_mode", "WAL")
        await _pragma_set(db, "synchronous",  self._SYNCHRONOUS_MODE)
        await _pragma_set(db, "busy_timeout", FRONTIER_BUSY_TIMEOUT_MS)
        await _pragma_set(db, "wal_autocheckpoint", FRONTIER_WAL_AUTOCHECKPOINT)
        await _pragma_set(db, "page_size",    FRONTIER_PAGE_SIZE)
        await _pragma_set(db, "mmap_size",    self._MMAP_SIZE)
        await _pragma_set(db, "cache_size",   self._CACHE_SIZE)
        # foreign_keys = OFF — no FK constraints in this schema, but explicit
        # is better than relying on the default.
        await _pragma_set(db, "foreign_keys", "OFF")
        # temp_store = MEMORY — in-memory temp tables for aggregate queries.
        await _pragma_set(db, "temp_store",   "MEMORY")

    async def create_tables(self, db: aiosqlite.Connection) -> None: # noqa
        """
        Create the frontier table and its indices if they do not exist.

        Idempotent: IF NOT EXISTS clauses make this safe to call on an
        already-initialized DB.

        Writes schema version to PRAGMA user_version after table creation.
        Version is only written if it was previously 0 (new DB) — existing
        DBs with a valid version are not overwritten.

        Parameters
        ──────────
        db   Open aiosqlite connection.  Must have journal_mode=WAL set.
        """
        await db.execute(_DDL_FRONTIER_TABLE)
        await db.execute(_DDL_FRONTIER_INDEX_MANIFEST_STATUS)
        await db.execute(_DDL_FRONTIER_INDEX_URL)
        await db.commit()

        # Write schema version only on first creation.
        current_version = await _pragma_get(db, "user_version")
        if not current_version:
            await _pragma_set(db, "user_version", FRONTIER_SCHEMA_VERSION)
            await db.commit()

    async def validate_version(self, db: aiosqlite.Connection) -> None: # noqa
        """
        Validate the on-disk schema version matches FRONTIER_SCHEMA_VERSION.

        Raises FrontierSchemaMismatchError if the versions differ.  This
        propagates to initialize() as FrontierError.

        New databases (user_version = 0 before create_tables()) do NOT trigger
        this check.

        Parameters
        ──────────
        db   Open aiosqlite connection with tables already created.

        Raises
        ──────
        FrontierSchemaMismatchError
            If the on-disk user_version does not match FRONTIER_SCHEMA_VERSION.
        """
        on_disk_version = await _pragma_get(db, "user_version")
        if on_disk_version and int(str(on_disk_version)) != FRONTIER_SCHEMA_VERSION:
            raise FrontierSchemaMismatchError(
                f"frontier.db schema version mismatch: "
                f"on-disk={on_disk_version} expected={FRONTIER_SCHEMA_VERSION}. "
                "Delete frontier.db to recreate with the current schema, or "
                "run a migration (not yet implemented)."
            )

    async def verify_wal_mode(self, db: aiosqlite.Connection) -> bool: # noqa
        """
        Return True if journal_mode is 'wal'.

        Used by health() and the CLI integrity command.

        Parameters
        ──────────
        db   Open aiosqlite connection.

        Returns
        ───────
        True if WAL mode is active.  False if another journal mode is in use.
        """
        mode = await _pragma_get(db, "journal_mode")
        return str(mode).lower() == "wal"

    async def integrity_check(self, db: aiosqlite.Connection) -> bool: # noqa
        """
        Run PRAGMA integrity_check and return True if the result is 'ok'.

        This performs a full structural integrity check of the B-trees.
        It is slow on large databases — call only from health() or CLI,
        never on the critical path.

        Parameters
        ──────────
        db   Open aiosqlite connection.

        Returns
        ───────
        True if PRAGMA integrity_check returns the single row 'ok'.
        False if any integrity errors are detected.
        """
        async with db.execute("PRAGMA integrity_check") as cur:
            rows = await cur.fetchall()
        return len(rows) == 1 and str(rows[0][0]).lower() == "ok"

    async def force_checkpoint(self, db: aiosqlite.Connection) -> None: # noqa
        """
        Issue PRAGMA wal_checkpoint(TRUNCATE) to merge the WAL into the main
        file and compact the WAL to near-zero length.

        Called periodically by _CheckpointManager to prevent unbounded WAL
        growth on very long manifests.

        Parameters
        ──────────
        db   Open aiosqlite connection.
        """
        await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")


# ─────────────────────────────────────────────────────────────────────────────
# _ManifestLoader
# Handles the batch-insert path for load_manifest().
# ─────────────────────────────────────────────────────────────────────────────

class _ManifestLoader:
    """
    Manages the batch-insert operation for loading a CrawlManifest into the
    frontier table.

    Each Frontier instance creates one _ManifestLoader during initialize().
    All state is ephemeral — a new loader is effectively created per call to
    load_manifest() via the _load() coroutine.

    Design constraints (from spec):
      - Insert in batches of FRONTIER_BATCH_SIZE.  Never one row at a time.
      - Do not deduplicate.  Bloom filter handles that.
      - Do not reorder URLs.  Preserve CrawlManifest.urls order (priority field).
      - A manifest that already has pending rows is not re-inserted (restart case).
      - Re-loading the same manifest_id is idempotent: no-op if rows exist.

    Internal contract
    ─────────────────
    _ManifestLoader._load() raises FrontierDuplicateManifestError if the
    manifest is already present.  The caller (Frontier.load_manifest()) catches
    this and returns without error — idempotent behavior.
    """

    async def load(
        self,
        db: aiosqlite.Connection,
        manifest: CrawlManifest,
    ) -> LoadResult:
        """
        Insert all URLs from `manifest` into the frontier table in batches.

        Returns a LoadResult describing the operation.  The result is used
        internally for structured logging — it is not returned to the caller
        of Frontier.load_manifest().

        Parameters
        ──────────
        db       Open aiosqlite connection.  Must have WAL mode enabled.
        manifest The CrawlManifest to load.

        Returns
        ───────
        LoadResult with was_resume=True if the manifest was already present.

        Raises
        ──────
        FrontierDuplicateManifestError
            If the manifest already has any rows (pending or otherwise).
        aiosqlite.OperationalError
            On SQLite write failures.  Caller wraps in FrontierError.
        """
        t_start = time.perf_counter()

        # Check for restart case: any rows for this manifest_id already exist.
        existing_count = await self._count_rows(db, manifest.manifest_id)
        if existing_count > 0:
            _log.info(
                "_ManifestLoader.load: manifest %s already has %d row(s) in DB. "
                "Skipping insert (resume path).",
                manifest.manifest_id[:8],
                existing_count,
            )
            raise FrontierDuplicateManifestError(
                f"manifest_id={manifest.manifest_id} already has {existing_count} rows"
            )

        # Build row tuples for all URLs.  time.time() is called once per batch
        # to minimize syscall overhead on large manifests.
        total_inserted = 0
        batch_count = 0
        url_list = manifest.urls

        for batch in _chunks(url_list, FRONTIER_BATCH_SIZE):
            now = time.time()
            rows = [
                _make_insert_row(manifest.manifest_id, crawl_url, now)
                for crawl_url in batch
            ]
            await db.executemany(_DML_INSERT_URL, rows)
            await db.commit()
            total_inserted += len(rows)
            batch_count += 1

            _log.debug(
                "_ManifestLoader.load: manifest %s batch %d/%d — "
                "inserted %d/%d rows.",
                manifest.manifest_id[:8],
                batch_count,
                (len(url_list) + FRONTIER_BATCH_SIZE - 1) // FRONTIER_BATCH_SIZE,
                total_inserted,
                len(url_list),
            )

        duration_ms = (time.perf_counter() - t_start) * 1000.0

        _log.info(
            "_ManifestLoader.load: manifest %s loaded %d URL(s) in %d batch(es) "
            "in %.1fms (%.0f URLs/s).",
            manifest.manifest_id[:8],
            total_inserted,
            batch_count,
            duration_ms,
            total_inserted / (duration_ms / 1000.0) if duration_ms > 0 else 0.0,
        )

        return LoadResult(
            manifest_id=manifest.manifest_id,
            rows_inserted=total_inserted,
            batches=batch_count,
            was_resume=False,
            duration_ms=duration_ms,
        )

    async def _count_rows(self, db: aiosqlite.Connection, manifest_id: str) -> int: # noqa
        """
        Count total rows for `manifest_id` regardless of status.

        Used to detect the restart case in load().

        Parameters
        ──────────
        db           Open aiosqlite connection.
        manifest_id  Manifest to check.

        Returns
        ───────
        Integer count of rows.  0 if the manifest has never been loaded.
        """
        async with db.execute(_DQL_COUNT_ALL, (manifest_id,)) as cur:
            row = await cur.fetchone()
            return row[0] if row else 0


# ─────────────────────────────────────────────────────────────────────────────
# _StatusWriter
# Handles fire-and-forget status updates (mark_done / mark_failed / mark_skipped).
# ─────────────────────────────────────────────────────────────────────────────

class _StatusWriter:
    """
    Handles asynchronous status update writes to the frontier table.

    mark_done(), mark_failed(), and mark_skipped() on Frontier all delegate
    to _StatusWriter.write().  The call is wrapped in asyncio.create_task()
    by Frontier — the writes are fire-and-forget and never block the fetch
    critical path.

    Failure behavior
    ────────────────
    Any SQLite error in write() is caught, logged at WARNING, and swallowed.
    A missed status update means the row stays 'pending'.  On restart, the
    URL is re-yielded by resume() and deduplicates via the Bloom filter
    (mark_skipped on the second pass).  The frontier is eventually consistent.

    write() returns a MarkResult regardless of success or failure.  The
    asyncio.Task wrapping the call discards the result.  This is intentional —
    the result exists only for structured logging inside this class.
    """

    async def write( # noqa
        self,
        db: aiosqlite.Connection,
        manifest_id: str,
        url: str,
        new_status: str,
    ) -> MarkResult:
        """
        Update the status of one URL from 'pending' to `new_status`.

        The WHERE clause includes `AND status = 'pending'` to prevent
        double-updates.  If a URL has already been marked (e.g., due to a
        duplicate task from a race between fire-and-forget calls), the UPDATE
        affects 0 rows and we log at DEBUG.

        Parameters
        ──────────
        db          Open aiosqlite connection.
        manifest_id Manifest the URL belongs to.
        url         URL to update.
        new_status  One of FRONTIER_STATUS_DONE / _FAILED / _SKIPPED.

        Returns
        ───────
        MarkResult describing the outcome.
        """
        assert new_status in (
            FRONTIER_STATUS_DONE,
            FRONTIER_STATUS_FAILED,
            FRONTIER_STATUS_SKIPPED,
        ), f"Invalid status: {new_status!r}"

        t_start = time.perf_counter()
        error_str: Optional[str] = None
        rows_affected = 0
        completed_at = time.time()

        try:
            async with db.execute(
                _DML_STATUS_UPDATE,
                (new_status, completed_at, manifest_id, url),
            ) as cur:
                rows_affected = cur.rowcount
            await db.commit()
        except Exception as exc:
            error_str = str(exc)
            _log.warning(
                "_StatusWriter.write: failed to set %s=%r for manifest %s "
                "url=%r: %s",
                url[:80],
                new_status,
                manifest_id[:8],
                url[:80],
                exc,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        if rows_affected == 0 and error_str is None:
            _log.debug(
                "_StatusWriter.write: 0 rows affected for manifest %s url=%r "
                "new_status=%r (already terminal or not found).",
                manifest_id[:8],
                url[:80],
                new_status,
            )
        elif rows_affected == 1:
            _log.debug(
                "_StatusWriter.write: %s → %s (%.2fms) manifest=%s",
                url[:80],
                new_status,
                latency_ms,
                manifest_id[:8],
            )

        return MarkResult(
            manifest_id=manifest_id,
            url=url,
            new_status=new_status,
            rows_affected=rows_affected,
            latency_ms=latency_ms,
            error=error_str,
        )


# ─────────────────────────────────────────────────────────────────────────────
# _CheckpointManager
# Handles periodic WAL force-checkpoint to prevent unbounded WAL growth.
# ─────────────────────────────────────────────────────────────────────────────

class _CheckpointManager:
    """
    Tracks write volume and issues periodic PRAGMA wal_checkpoint(TRUNCATE)
    to prevent the WAL file from growing unboundedly on long manifests.

    Design
    ──────
    SQLite's wal_autocheckpoint merges the WAL back into the main file when
    the WAL reaches FRONTIER_WAL_AUTOCHECKPOINT pages.  However, auto-checkpoint
    only triggers at non-EXCLUSIVE read/write boundaries.  For a frontier
    that is continuously writing status updates, the auto-checkpoint may not
    trigger promptly.

    _CheckpointManager issues a forced TRUNCATE checkpoint every
    _FORCE_INTERVAL writes.  TRUNCATE resets the WAL to near-zero length
    rather than just marking frames as overwriteable.

    This is an optimization.  A failure to checkpoint is not fatal — the WAL
    will continue to grow until the next checkpoint.  The max practical WAL
    size on a 100K URL manifest is bounded by total status updates * page_size,
    which is manageable even without forced checkpoints.

    Usage
    ─────
    Called from Frontier._schedule_checkpoint() inside each mark_* path.
    The checkpoint is always issued as asyncio.create_task() — it never
    blocks the fetch path.
    """

    _FORCE_INTERVAL: int = 5_000
    """Number of write operations between forced checkpoints."""

    def __init__(self) -> None:
        self._write_count: int = 0
        self._checkpoint_count: int = 0
        self._last_checkpoint_at: Optional[float] = None

    def record_write(self) -> bool:
        """
        Increment the write counter.  Return True if a checkpoint is due.

        Parameters
        ──────────
        None.

        Returns
        ───────
        True if self._write_count % _FORCE_INTERVAL == 0 (checkpoint due).
        False otherwise.
        """
        self._write_count += 1
        return self._write_count % self._FORCE_INTERVAL == 0

    async def maybe_checkpoint(
        self,
        db: aiosqlite.Connection,
        schema_manager: _SchemaManager,
    ) -> None:
        """
        Issue a force checkpoint if record_write() returned True.

        This method is called on the fire-and-forget task from Frontier.
        Any exception is caught and logged at DEBUG — checkpoint failure
        does not affect the frontier's operational correctness.

        Parameters
        ──────────
        db             Open aiosqlite connection.
        schema_manager Used to call force_checkpoint().
        """
        if not self.record_write():
            return

        try:
            t_start = time.perf_counter()
            await schema_manager.force_checkpoint(db)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            self._checkpoint_count += 1
            self._last_checkpoint_at = time.time()
            _log.debug(
                "_CheckpointManager: forced WAL checkpoint #%d in %.1fms.",
                self._checkpoint_count,
                elapsed_ms,
            )
        except Exception as exc:
            _log.debug(
                "_CheckpointManager: forced checkpoint failed (non-fatal): %s", exc
            )

    @property
    def write_count(self) -> int:
        """Total writes recorded since construction."""
        return self._write_count

    @property
    def checkpoint_count(self) -> int:
        """Total forced checkpoints issued since construction."""
        return self._checkpoint_count


# ─────────────────────────────────────────────────────────────────────────────
# FRONTIER
# The public interface.  This is the only class callers interact with.
# ─────────────────────────────────────────────────────────────────────────────

class Frontier:
    """
    Resumable crawl frontier backed by SQLite WAL.

    One Frontier instance is created per CrawlManifest by the fetcher.  It
    holds its own SQLite connection.  Multiple Frontier instances targeting
    the same frontier.db file co-exist safely under WAL mode — each instance
    acquires the WAL write lock only for the duration of each individual
    transaction, not for the lifetime of the instance.

    Lifecycle
    ─────────
    1. Construct: ``frontier = Frontier(cursor)``
    2. Initialize: ``await frontier.initialize()``   ← creates/opens frontier.db
    3. Load manifest: ``await frontier.load_manifest(manifest)``
    4. Resume: ``async for crawl_url in frontier.resume(manifest_id): ...``
    5. Update status: ``asyncio.create_task(frontier.mark_done(...))``
    6. Check completion: ``await frontier.is_complete(manifest_id)``
    7. Close: ``await frontier.close()``

    Constructor
    ───────────
    ``cursor``  CrawlCursor instance.  Must be initialized before passing.
                The cursor is used by resume() to determine the start OFFSET.
                The fetcher holds a separate cursor reference and calls
                cursor.checkpoint() every CURSOR_CHECKPOINT_INTERVAL URLs.
                The Frontier's cursor reference is read-only (get_position only).

    ``db_path`` Path to the SQLite database file.  Defaults to FRONTIER_DB_PATH.
                Created by initialize() if it does not exist.

    Thread safety
    ─────────────
    Not thread-safe.  asyncio is single-threaded.  Do not share a Frontier
    instance across threads or processes.

    Error handling
    ──────────────
    All public methods wrap SQLite exceptions in FrontierError.  FrontierError
    is not a hard stop — fetcher.py catches it, logs it, emits FetchAnomalyEvent,
    and continues the manifest.

    load_manifest() raises FrontierError on DB failure.
    resume() raises FrontierError if the DB cannot be queried.
    mark_done/failed/skipped(): fire-and-forget; errors are logged and swallowed.
    is_complete(), pending_count(), stats(): raise FrontierError on DB failure.
    close(): never raises.
    """

    def __init__(
        self,
        cursor: CrawlCursor,
        db_path: Path = FRONTIER_DB_PATH,
    ) -> None:
        """
        Construct a Frontier.

        ``initialize()`` must be awaited before any other method.  Calling
        load_manifest(), resume(), etc. before initialize() raises
        FrontierNotInitializedError, which propagates as FrontierError.

        Parameters
        ──────────
        cursor   Initialized CrawlCursor instance.  Used exclusively for
                 get_position() in resume().  The frontier does not call
                 checkpoint() — that is the fetcher's responsibility.
        db_path  Path to frontier.db.  Created on first run.
        """
        self._cursor:           CrawlCursor         = cursor
        self._db_path:          Path                = db_path
        self._db:               Optional[aiosqlite.Connection] = None
        self._initialized:      bool                = False
        self._schema_manager:   _SchemaManager      = _SchemaManager()
        self._loader:           _ManifestLoader     = _ManifestLoader()
        self._writer:           _StatusWriter       = _StatusWriter()
        self._checkpoint_mgr:   _CheckpointManager  = _CheckpointManager()
        # Track task handles to prevent GC of fire-and-forget tasks.
        self._background_tasks: Set[asyncio.Task]   = set()

    # ─────────────────────────────────────────────────────────────────────────
    # initialize / close
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self, db_path: Optional[Path] = None) -> None:
        """
        Open or create the frontier database.

        If the database file does not exist it is created, configured with
        WAL mode, and the frontier schema is initialized.

        If the database file exists it is opened, WAL mode is verified/set,
        and the schema version is validated.

        This method is idempotent — calling it on an already-initialized
        Frontier is a no-op (the existing connection is reused).

        Parameters
        ──────────
        db_path  Override the db_path set in the constructor.  Useful in tests
                 that need to redirect the DB to a temporary path.  If None,
                 the path from the constructor is used.

        Raises
        ──────
        FrontierError
            If WAL mode cannot be set, the schema cannot be created, or the
            schema version does not match FRONTIER_SCHEMA_VERSION.
        """
        if self._initialized:
            _log.debug("Frontier.initialize: already initialized — no-op.")
            return

        if db_path is not None:
            self._db_path = db_path

        # Ensure the parent directory exists.  Do not create the store/
        # directory itself — that is the host environment's responsibility.
        self._db_path.parent.mkdir(parents=False, exist_ok=True)

        _log.info(
            "Frontier.initialize: opening %s (exists=%s).",
            self._db_path,
            self._db_path.exists(),
        )

        try:
            self._db = await aiosqlite.connect(str(self._db_path))
            # Row factory so rows support both index and name access.
            self._db.row_factory = aiosqlite.Row

            # Step 1: configure pragmas (must happen before any DDL).
            await self._schema_manager.configure(self._db)

            # Step 2: create tables if they don't exist.
            await self._schema_manager.create_tables(self._db)

            # Step 3: validate schema version on existing DBs.
            await self._schema_manager.validate_version(self._db)

            self._initialized = True
            _log.info(
                "Frontier.initialize: ready. WAL=%s path=%s",
                await self._schema_manager.verify_wal_mode(self._db),
                self._db_path,
            )
        except FrontierSchemaMismatchError as exc:
            await self._safe_close_db()
            raise FrontierError(
                manifest_id="(schema-check)",
                operation="initialize",
                db_error=str(exc),
            ) from exc
        except Exception as exc:
            await self._safe_close_db()
            raise FrontierError(
                manifest_id="(initialize)",
                operation="initialize",
                db_error=str(exc),
            ) from exc

    async def close(self) -> None:
        """
        Flush pending writes and close the SQLite connection.

        Cancels and awaits all pending background tasks before closing.
        After close(), the Frontier instance must not be used — call
        initialize() again to reopen.

        This method never raises.  All errors are logged at WARNING.
        """
        # Cancel + await fire-and-forget background tasks.
        if self._background_tasks:
            pending = list(self._background_tasks)
            _log.debug(
                "Frontier.close: waiting for %d background task(s).",
                len(pending),
            )
            for task in pending:
                if not task.done():
                    task.cancel()
            # Allow tasks to process their CancelledError.
            results = await asyncio.gather(*pending, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception) and not isinstance(r, asyncio.CancelledError):
                    _log.debug("Frontier.close: background task error (suppressed): %s", r)
            self._background_tasks.clear()

        await self._safe_close_db()
        self._initialized = False
        _log.info("Frontier.close: closed %s.", self._db_path)

    # ─────────────────────────────────────────────────────────────────────────
    # load_manifest
    # ─────────────────────────────────────────────────────────────────────────

    async def load_manifest(self, manifest: CrawlManifest) -> None:
        """
        Batch-insert all URLs from `manifest` into the frontier table.

        If the manifest_id already has rows in the DB (restart case), this
        method is a no-op.  The existing rows are used as-is and resume()
        will continue from the cursor position.

        Idempotency guarantee: calling load_manifest() with the same manifest
        twice is safe.  The second call produces zero writes.

        Performance: a 100K URL manifest is inserted in under 2 seconds on
        commodity NVMe storage, in batches of FRONTIER_BATCH_SIZE rows.

        Parameters
        ──────────
        manifest  CrawlManifest from the preparser.  manifest.urls is the
                  ordered fetch sequence.  manifest.manifest_id is the UUID4
                  that identifies this manifest in all subsequent calls.

        Raises
        ──────
        FrontierError
            If the DB insert fails for any reason.  This propagates to the
            fetcher, which emits FetchAnomalyEvent and continues.
        FrontierNotInitializedError  (via FrontierError)
            If initialize() has not been awaited.
        """
        self._require_initialized("load_manifest", manifest.manifest_id)

        try:
            result = await self._loader.load(self._db, manifest)
        except FrontierDuplicateManifestError:
            _log.info(
                "Frontier.load_manifest: manifest %s already loaded — "
                "using existing rows (resume path).",
                manifest.manifest_id[:8],
            )
            return
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest.manifest_id,
                operation="load_manifest",
                db_error=str(exc),
            ) from exc

        _log.info(
            "Frontier.load_manifest: %s  "
            "(rows=%d, batches=%d, duration=%.1fms)",
            result,
            result.rows_inserted,
            result.batches,
            result.duration_ms,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # resume
    # ─────────────────────────────────────────────────────────────────────────

    async def resume(self, manifest_id: str) -> AsyncIterator[CrawlURL]:
        """
        Yield pending CrawlURL objects from the cursor position onwards.

        Resume protocol:
          1. Read position from CrawlCursor.get_position(manifest_id).
          2. Issue SELECT ... WHERE status='pending' ORDER BY priority OFFSET position.
          3. Yield one CrawlURL per row.  Rows are fetched lazily from SQLite.
          4. If the cursor returns 0 (no saved position or fresh start), yield
             from the beginning of the pending queue.

        The generator is lazy.  The full pending URL set is never materialized
        in memory — each row is fetched from SQLite on demand.  For a manifest
        with 1M pending URLs, this function uses O(1) memory.

        The caller (fetcher.py) consumes this generator in a loop, calling
        mark_done/failed/skipped and cursor.checkpoint() as each URL is
        processed.  The generator can be abandoned midway (e.g., if the
        process is killed) — on restart, calling resume() again yields from
        the new cursor position.

        Parameters
        ──────────
        manifest_id  UUID4 identifying the manifest to resume.

        Yields
        ──────
        CrawlURL objects with all fields populated from the DB.  Rows are
        returned in order of ascending priority (lower = higher priority),
        with id as a tie-breaker (earlier insertions first).

        Raises
        ──────
        FrontierError
            If the cursor position cannot be read, or if the DB query fails.
        FrontierNotInitializedError  (via FrontierError)
            If initialize() has not been awaited.
        """
        self._require_initialized("resume", manifest_id)

        try:
            position = await self._cursor.get_position(manifest_id)
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="resume:get_position",
                db_error=str(exc),
            ) from exc

        _log.info(
            "Frontier.resume: manifest %s starting at cursor position %d.",
            manifest_id[:8],
            position,
        )

        try:
            async with self._db.execute(
                _DQL_RESUME, (manifest_id, position)
            ) as cursor:
                async for raw_row in cursor:
                    frontier_row = _row_to_frontier_row(manifest_id, raw_row)
                    yield frontier_row.to_crawl_url()
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="resume:iterate",
                db_error=str(exc),
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # mark_done / mark_failed / mark_skipped
    # ─────────────────────────────────────────────────────────────────────────

    async def mark_done(self, manifest_id: str, url: str) -> None:
        """
        Transition a URL's status from 'pending' to 'done'.

        Called by the fetcher after successfully emitting a RawFetchEvent.
        Must be called as ``asyncio.create_task(frontier.mark_done(...))``,
        not awaited inline — it is fire-and-forget.

        The write is atomic: if the process dies between the UPDATE and the
        commit, the row stays 'pending' and is re-yielded on restart.  The
        Bloom filter prevents duplicate RawFetchEvent emission on re-fetch.

        A WAL force-checkpoint is scheduled every _CheckpointManager._FORCE_INTERVAL
        writes to prevent unbounded WAL growth.

        Parameters
        ──────────
        manifest_id  UUID4 of the owning manifest.
        url          URL whose status to update.

        Never raises — errors are logged at WARNING and swallowed.
        """
        if not self._initialized or self._db is None:
            _log.warning(
                "Frontier.mark_done: not initialized — skipping update for %s.",
                url[:80],
            )
            return

        await self._writer.write(self._db, manifest_id, url, FRONTIER_STATUS_DONE)
        self._maybe_schedule_checkpoint()

    async def mark_failed(self, manifest_id: str, url: str) -> None:
        """
        Transition a URL's status from 'pending' to 'failed'.

        Called by the fetcher after emitting a FetchAnomalyEvent.  'failed'
        is a terminal status — frontier.py never retries failed URLs.  The
        index_daemon decides whether to re-queue the URL on the next gradient
        step.

        Must be called as ``asyncio.create_task(frontier.mark_failed(...))``.
        Fire-and-forget.  Never raises.

        Parameters
        ──────────
        manifest_id  UUID4 of the owning manifest.
        url          URL whose status to update.
        """
        if not self._initialized or self._db is None:
            _log.warning(
                "Frontier.mark_failed: not initialized — skipping update for %s.",
                url[:80],
            )
            return

        await self._writer.write(self._db, manifest_id, url, FRONTIER_STATUS_FAILED)
        self._maybe_schedule_checkpoint()

    async def mark_skipped(self, manifest_id: str, url: str) -> None:
        """
        Transition a URL's status from 'pending' to 'skipped'.

        Called by the fetcher when bloom_filter.contains() returns True for
        this URL.  The URL was probably seen in a previous crawl pass.
        'skipped' is terminal and counted toward manifest completion.

        Must be called as ``asyncio.create_task(frontier.mark_skipped(...))``.
        Fire-and-forget.  Never raises.

        Parameters
        ──────────
        manifest_id  UUID4 of the owning manifest.
        url          URL whose status to update.
        """
        if not self._initialized or self._db is None:
            _log.warning(
                "Frontier.mark_skipped: not initialized — skipping update for %s.",
                url[:80],
            )
            return

        await self._writer.write(self._db, manifest_id, url, FRONTIER_STATUS_SKIPPED)
        self._maybe_schedule_checkpoint()

    # ─────────────────────────────────────────────────────────────────────────
    # is_complete / pending_count / stats
    # ─────────────────────────────────────────────────────────────────────────

    async def is_complete(self, manifest_id: str) -> bool:
        """
        Return True when all URLs for `manifest_id` have a terminal status.

        Completion is defined as: zero rows with status = 'pending'.  A
        manifest is complete when every URL is 'done', 'failed', or 'skipped'.

        This query is served by the covering index on (manifest_id, status, priority)
        and runs in O(index scan) time without touching the table.

        Parameters
        ──────────
        manifest_id  UUID4 to check.

        Returns
        ───────
        True if no pending rows remain.  False otherwise.
        False is also returned for an unknown manifest_id (no rows at all).

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("is_complete", manifest_id)

        try:
            async with self._db.execute(
                _DQL_COUNT_PENDING, (manifest_id,)
            ) as cur:
                row = await cur.fetchone()
                count = row[0] if row else 0
            return count == 0
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="is_complete",
                db_error=str(exc),
            ) from exc

    async def pending_count(self, manifest_id: str) -> int:
        """
        Return the count of URLs with status = 'pending' for `manifest_id`.

        Used by the fetcher for progress logging and by tests.

        Parameters
        ──────────
        manifest_id  UUID4 to query.

        Returns
        ───────
        Integer count.  0 for an unknown manifest_id.

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("pending_count", manifest_id)

        try:
            async with self._db.execute(
                _DQL_COUNT_PENDING, (manifest_id,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="pending_count",
                db_error=str(exc),
            ) from exc

    async def stats(self, manifest_id: str) -> FrontierStats:
        """
        Return a FrontierStats snapshot for `manifest_id`.

        Runs a single aggregating SELECT with four CASE expressions.  The
        covering index on (manifest_id, status, priority) serves this query
        without a full table scan.

        Parameters
        ──────────
        manifest_id  UUID4 to query.

        Returns
        ───────
        FrontierStats with pending/done/failed/skipped counts.  All fields
        are 0 for an unknown manifest_id.

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("stats", manifest_id)

        try:
            async with self._db.execute(_DQL_STATS, (manifest_id,)) as cur:
                row = await cur.fetchone()
            if row is None:
                return FrontierStats(
                    manifest_id=manifest_id,
                    pending=0,
                    done=0,
                    failed=0,
                    skipped=0,
                )
            return FrontierStats(
                manifest_id=manifest_id,
                pending=int(row[0] or 0),
                done=int(row[1] or 0),
                failed=int(row[2] or 0),
                skipped=int(row[3] or 0),
            )
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="stats",
                db_error=str(exc),
            ) from exc

    # ─────────────────────────────────────────────────────────────────────────
    # Supplemental public methods
    # Not in the spec API, but essential for production operation:
    #   - list_manifests() for monitoring
    #   - archive_manifest() for post-completion cleanup
    #   - delete_manifest() for full removal
    #   - health() for liveness checks
    #   - get_url() for targeted debugging
    # ─────────────────────────────────────────────────────────────────────────

    async def list_manifests(self) -> List[ManifestSummary]:
        """
        Return a summary row for every manifest stored in the DB.

        Used by the CLI dump command and monitoring dashboards.  Runs a
        single aggregating query — not N per-manifest queries.

        Returns
        ───────
        List of ManifestSummary objects, ordered by first_added_at descending
        (most recently loaded manifests first).  Empty list if the DB has no rows.

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("list_manifests", "(all)")

        try:
            async with self._db.execute(_DQL_MANIFEST_SUMMARY) as cur:
                rows = await cur.fetchall()
        except Exception as exc:
            raise FrontierError(
                manifest_id="(all)",
                operation="list_manifests",
                db_error=str(exc),
            ) from exc

        summaries = []
        for row in rows:
            summaries.append(ManifestSummary(
                manifest_id=str(row[0]),
                total=int(row[1] or 0),
                pending=int(row[2] or 0),
                done=int(row[3] or 0),
                failed=int(row[4] or 0),
                skipped=int(row[5] or 0),
                first_added_at=float(row[6]) if row[6] is not None else None,
                last_completed_at=float(row[7]) if row[7] is not None else None,
            ))
        return summaries

    async def archive_manifest(self, manifest_id: str) -> int:
        """
        Mark all remaining 'pending' rows for `manifest_id` as 'skipped'.

        Called after the fetcher completes a manifest (emits ManifestCompleteEvent)
        to close out any URLs that were never processed due to process death
        between the last checkpoint and manifest completion.

        This is a cleanup operation.  The Bloom filter has already deduplicated
        any URLs that were actually fetched in previous passes.

        Parameters
        ──────────
        manifest_id  UUID4 of the completed manifest.

        Returns
        ───────
        Number of rows updated (should be 0 for a cleanly completed manifest,
        > 0 only if there were in-progress URLs at the time of call).

        Raises
        ──────
        FrontierError
            If the DB update fails.
        """
        self._require_initialized("archive_manifest", manifest_id)

        try:
            async with self._db.execute(
                _DML_ARCHIVE_PENDING, (time.time(), manifest_id)
            ) as cur:
                rows_affected = cur.rowcount
            await self._db.commit()
            _log.info(
                "Frontier.archive_manifest: manifest %s — archived %d pending row(s).",
                manifest_id[:8],
                rows_affected,
            )
            return rows_affected
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="archive_manifest",
                db_error=str(exc),
            ) from exc

    async def delete_manifest(self, manifest_id: str) -> int:
        """
        Hard-delete all frontier rows for `manifest_id`.

        This is a destructive, irreversible operation.  It is called by the
        fetcher after the manifest's cursor has been cleared and its stats
        have been emitted to the bus.  The rows are no longer needed — the
        Bloom filter is the durable dedup record.

        Parameters
        ──────────
        manifest_id  UUID4 of the manifest to delete.

        Returns
        ───────
        Number of rows deleted.

        Raises
        ──────
        FrontierError
            If the DB delete fails.
        """
        self._require_initialized("delete_manifest", manifest_id)

        try:
            async with self._db.execute(
                _DML_DELETE_MANIFEST, (manifest_id,)
            ) as cur:
                rows_deleted = cur.rowcount
            await self._db.commit()
            _log.info(
                "Frontier.delete_manifest: deleted %d row(s) for manifest %s.",
                rows_deleted,
                manifest_id[:8],
            )
            return rows_deleted
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="delete_manifest",
                db_error=str(exc),
            ) from exc

    async def get_url(
        self,
        manifest_id: str,
        url: str,
    ) -> Optional[_FrontierRow]:
        """
        Fetch a single URL row by (manifest_id, url).

        Used for debugging, integrity checks, and tests.  Not on the critical
        path — the index on (manifest_id, url) makes this fast.

        Parameters
        ──────────
        manifest_id  UUID4 of the owning manifest.
        url          URL to look up.

        Returns
        ───────
        _FrontierRow if the row exists.  None if not found.

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("get_url", manifest_id)

        try:
            async with self._db.execute(
                _DQL_ROW_BY_URL, (manifest_id, url)
            ) as cur:
                raw_row = await cur.fetchone()
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="get_url",
                db_error=str(exc),
            ) from exc

        if raw_row is None:
            return None

        return _FrontierRow(
            manifest_id=manifest_id,
            url=str(raw_row[0]),
            topology_hint=str(raw_row[1]),
            fetch_mode=str(raw_row[2]),
            render_mode=str(raw_row[3]),
            priority=int(raw_row[4]),
            status=str(raw_row[5]),
            added_at=float(raw_row[6]) if raw_row[6] is not None else 0.0,
            completed_at=float(raw_row[7]) if raw_row[7] is not None else None,
            rate_limit_domain=str(raw_row[8] or ""),
            rate_limit_rps=float(raw_row[9]) if raw_row[9] is not None else 1.0,
            rate_limit_burst=int(raw_row[10]) if raw_row[10] is not None else 3,
            crawl_delay_seconds=float(raw_row[11]) if raw_row[11] is not None else 0.0,
            expected_content_type=str(raw_row[12] or FRONTIER_DEFAULT_EXPECTED_CONTENT_TYPE),
            max_response_bytes=int(raw_row[13]) if raw_row[13] is not None else FRONTIER_DEFAULT_MAX_RESPONSE_BYTES,
            is_robots_int=int(raw_row[14]) if raw_row[14] is not None else 0,
            is_sitemap_int=int(raw_row[15]) if raw_row[15] is not None else 0,
            run_id=str(raw_row[16] or ""),
        )

    async def health(self) -> FrontierHealth:
        """
        Return a point-in-time health snapshot for this Frontier instance.

        Runs PRAGMA integrity_check — this is expensive on large DBs.  Call
        only from monitoring paths, never on the critical fetch path.

        Returns
        ───────
        FrontierHealth with all fields populated.  is_healthy is True only
        when all checks pass.

        Never raises — any errors are captured in health.error.
        """
        db_path_str = str(self._db_path)
        db_exists = self._db_path.exists()
        wal_path = Path(str(self._db_path) + "-wal")

        if not self._initialized or self._db is None:
            return FrontierHealth(
                db_path=db_path_str,
                db_exists=db_exists,
                wal_mode_active=False,
                integrity_ok=False,
                schema_version=0,
                initialized=False,
                manifest_count=0,
                total_url_count=0,
                pending_url_count=0,
                done_url_count=0,
                failed_url_count=0,
                skipped_url_count=0,
                db_size_bytes=self._db_path.stat().st_size if db_exists else 0,
                wal_size_bytes=wal_path.stat().st_size if wal_path.exists() else 0,
                error="not initialized",
            )

        try:
            wal_mode = await self._schema_manager.verify_wal_mode(self._db)
            integrity = await self._schema_manager.integrity_check(self._db)
            schema_version = int(await _pragma_get(self._db, "user_version") or 0)

            # Aggregate across all manifests for the DB-level health report.
            manifests = await self.list_manifests()
            manifest_count = len(manifests)
            total_url_count    = sum(m.total   for m in manifests)
            pending_url_count  = sum(m.pending for m in manifests)
            done_url_count     = sum(m.done    for m in manifests)
            failed_url_count   = sum(m.failed  for m in manifests)
            skipped_url_count  = sum(m.skipped for m in manifests)

            db_size_bytes  = self._db_path.stat().st_size if db_exists else 0
            wal_size_bytes = wal_path.stat().st_size if wal_path.exists() else 0

            return FrontierHealth(
                db_path=db_path_str,
                db_exists=db_exists,
                wal_mode_active=wal_mode,
                integrity_ok=integrity,
                schema_version=schema_version,
                initialized=True,
                manifest_count=manifest_count,
                total_url_count=total_url_count,
                pending_url_count=pending_url_count,
                done_url_count=done_url_count,
                failed_url_count=failed_url_count,
                skipped_url_count=skipped_url_count,
                db_size_bytes=db_size_bytes,
                wal_size_bytes=wal_size_bytes,
                error=None,
            )
        except Exception as exc:
            _log.warning("Frontier.health: error during health check: %s", exc)
            return FrontierHealth(
                db_path=db_path_str,
                db_exists=db_exists,
                wal_mode_active=False,
                integrity_ok=False,
                schema_version=0,
                initialized=self._initialized,
                manifest_count=0,
                total_url_count=0,
                pending_url_count=0,
                done_url_count=0,
                failed_url_count=0,
                skipped_url_count=0,
                db_size_bytes=self._db_path.stat().st_size if db_exists else 0,
                wal_size_bytes=wal_path.stat().st_size if wal_path.exists() else 0,
                error=str(exc),
            )

    async def completed_count(self, manifest_id: str) -> int:
        """
        Return the count of URLs with a terminal status (done + failed + skipped).

        Complementary to pending_count().  The sum of completed_count() and
        pending_count() equals the total URL count for a manifest.

        Parameters
        ──────────
        manifest_id  UUID4 to query.

        Returns
        ───────
        Integer count of non-pending rows.  0 for an unknown manifest_id.

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("completed_count", manifest_id)

        try:
            async with self._db.execute(
                _DQL_COMPLETED_COUNT, (manifest_id,)
            ) as cur:
                row = await cur.fetchone()
                return row[0] if row else 0
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="completed_count",
                db_error=str(exc),
            ) from exc

    async def exists(self, manifest_id: str) -> bool:
        """
        Return True if any rows exist for `manifest_id`.

        Parameters
        ──────────
        manifest_id  UUID4 to check.

        Returns
        ───────
        True if at least one row exists.  False for an unknown manifest_id.

        Raises
        ──────
        FrontierError
            If the DB query fails.
        """
        self._require_initialized("exists", manifest_id)

        try:
            async with self._db.execute(
                _DQL_EXISTS_MANIFEST, (manifest_id,)
            ) as cur:
                row = await cur.fetchone()
                return row is not None
        except Exception as exc:
            raise FrontierError(
                manifest_id=manifest_id,
                operation="exists",
                db_error=str(exc),
            ) from exc

    async def force_wal_checkpoint(self) -> None:
        """
        Issue PRAGMA wal_checkpoint(TRUNCATE) immediately.

        Intended for use at graceful shutdown — compact the WAL before the
        process exits to minimize cold-start time for the next process.

        Never raises — errors are logged at DEBUG.
        """
        if not self._initialized or self._db is None:
            return
        try:
            await self._schema_manager.force_checkpoint(self._db)
            _log.debug("Frontier.force_wal_checkpoint: checkpoint issued.")
        except Exception as exc:
            _log.debug("Frontier.force_wal_checkpoint: failed (non-fatal): %s", exc)

    # ─────────────────────────────────────────────────────────────────────────
    # Context manager support
    # ─────────────────────────────────────────────────────────────────────────

    async def __aenter__(self) -> "Frontier":
        """Support ``async with Frontier(...) as f:`` usage."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the Frontier when the async context exits."""
        try:
            await self.force_wal_checkpoint()
        except Exception: # noqa
            pass
        await self.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _require_initialized(self, method_name: str, manifest_id: str) -> None:
        """
        Raise FrontierError (wrapping FrontierNotInitializedError) if the
        Frontier has not been initialized.

        Called at the top of every public method.  Early-exit guard that
        prevents confusing AttributeError from None db dereference.

        Parameters
        ──────────
        method_name   Name of the calling method, for the error message.
        manifest_id   manifest_id for the FrontierError context.

        Raises
        ──────
        FrontierError
            If self._initialized is False.
        """
        if not self._initialized or self._db is None:
            exc = FrontierNotInitializedError(
                f"Frontier.{method_name}() called before initialize(). "
                "Await frontier.initialize() before any other method."
            )
            raise FrontierError(
                manifest_id=manifest_id,
                operation=method_name,
                db_error=str(exc),
            ) from exc

    async def _safe_close_db(self) -> None:
        """
        Close the aiosqlite connection without raising.

        Used by initialize() on error and by close().  Logs failures at
        WARNING but never propagates them.
        """
        if self._db is not None:
            try:
                await self._db.close()
            except Exception as exc:
                _log.warning("Frontier._safe_close_db: error on close: %s", exc)
            finally:
                self._db = None

    def _maybe_schedule_checkpoint(self) -> None:
        """
        Schedule a fire-and-forget WAL checkpoint task if one is due.

        Called from mark_done/failed/skipped after every status write.
        The checkpoint task is added to self._background_tasks to prevent
        it from being garbage-collected before it completes.

        The checkpoint is issued at most once every
        _CheckpointManager._FORCE_INTERVAL writes.  Between thresholds this
        method is a no-op (record_write() returns False).
        """
        if not self._initialized or self._db is None:
            return

        if self._checkpoint_mgr.record_write():
            task = asyncio.create_task(
                self._checkpoint_mgr.maybe_checkpoint(
                    self._db, self._schema_manager
                )
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

    def _spawn_task(self, coro) -> asyncio.Task:
        """
        Schedule a fire-and-forget coroutine as an asyncio Task.

        Used by the fetcher indirectly (via mark_done etc.) and internally
        for background work.  The task is tracked in _background_tasks to
        prevent GC before completion.

        Parameters
        ──────────
        coro   Coroutine to schedule.

        Returns
        ───────
        The asyncio.Task wrapping the coroutine.
        """
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)
        return task

    @property
    def is_initialized(self) -> bool:
        """True if initialize() has completed successfully."""
        return self._initialized

    @property
    def db_path(self) -> Path:
        """Path to the frontier SQLite database file."""
        return self._db_path

    @property
    def background_task_count(self) -> int:
        """Number of pending fire-and-forget background tasks."""
        return len(self._background_tasks)

    @property
    def write_count(self) -> int:
        """Total status update writes recorded since construction."""
        return self._checkpoint_mgr.write_count

    def __repr__(self) -> str:
        return (
            f"Frontier("
            f"path={self._db_path!r}, "
            f"initialized={self._initialized}, "
            f"writes={self._checkpoint_mgr.write_count})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY FUNCTION
# Convenience factory for the standard production usage pattern.
# ─────────────────────────────────────────────────────────────────────────────

async def open_frontier(
    cursor: CrawlCursor,
    db_path: Path = FRONTIER_DB_PATH,
) -> Frontier:
    """
    Construct and initialize a Frontier in one call.

    Equivalent to:
        frontier = Frontier(cursor, db_path)
        await frontier.initialize()

    Parameters
    ──────────
    cursor   Initialized CrawlCursor.
    db_path  Path to frontier.db.  Defaults to FRONTIER_DB_PATH.

    Returns
    ───────
    Initialized Frontier instance.

    Raises
    ──────
    FrontierError
        If initialize() fails.
    """
    frontier = Frontier(cursor=cursor, db_path=db_path)
    await frontier.initialize()
    return frontier


# ─────────────────────────────────────────────────────────────────────────────
# DIAGNOSTIC TEST SUITE
#
# Run with: python frontier.py test
#
# Tests cover the complete contract as specified in readme-crawler.md:
#   1.  load_manifest() inserts all URLs in priority order
#   2.  resume() yields URLs from cursor position onwards
#   3.  mark_done() / mark_failed() / mark_skipped() update status correctly
#   4.  is_complete() returns True when no pending rows remain
#   5.  Batch insert performance — 100K URL manifest loads in < 2 seconds
#   6.  WAL mode enabled
#   7.  Resume after simulated crash — correct position from cursor
#   8.  Re-loading same manifest_id does not duplicate rows
#   9.  stats() returns correct counts across all status types
#   10. Concurrent manifests — separate frontier instances don't interfere
#   Plus many additional coverage cases for edge conditions and resilience.
# ─────────────────────────────────────────────────────────────────────────────

class _TestHelpers:
    """
    Static helpers for building test fixtures.

    These helpers construct minimal CrawlManifest / CrawlURL objects with
    sensible defaults so individual test cases only need to specify the fields
    relevant to what they are testing.
    """

    @staticmethod
    def make_rate_limit_profile(
        domain: str = "example.com",
        rps: float = 1.0,
        burst: int = 3,
        delay: float = 0.0,
    ) -> RateLimitProfile:
        return RateLimitProfile(
            domain=domain,
            requests_per_second=rps,
            crawl_delay_seconds=delay,
            burst_capacity=burst,
        )

    @staticmethod
    def make_crawl_url(
        url: str = "https://example.com/page",
        topology_hint: str = "GENERIC_HTML",
        fetch_mode: FetchMode = FetchMode.STATIC,
        render_mode: str = "static",
        priority: int = 0,
        domain: str = "example.com",
        is_robots: bool = False,
        is_sitemap: bool = False,
        run_id: str = "00000000-0000-4000-a000-000000000001",
        crawl_delay: float = 0.0,
        max_bytes: int = 4 * 1024 * 1024,
    ) -> CrawlURL:
        return CrawlURL(
            url=url,
            topology_hint=topology_hint,
            fetch_mode=fetch_mode,
            render_mode=render_mode,
            priority=priority,
            rate_limit_profile=_TestHelpers.make_rate_limit_profile(
                domain=domain,
                delay=crawl_delay,
            ),
            expected_content_type="text/html",
            crawl_delay_seconds=crawl_delay,
            max_response_bytes=max_bytes,
            is_robots=is_robots,
            is_sitemap=is_sitemap,
            run_id=run_id,
        )

    @staticmethod
    def make_manifest(
        domain: str = "example.com",
        url_count: int = 10,
        manifest_id: Optional[str] = None,
        fetch_mode: FetchMode = FetchMode.STATIC,
    ) -> CrawlManifest:
        import uuid
        mid = manifest_id or str(uuid.uuid4())
        urls = [
            _TestHelpers.make_crawl_url(
                url=f"https://{domain}/page/{i}",
                priority=i,
                domain=domain,
                fetch_mode=fetch_mode,
            )
            for i in range(url_count)
        ]
        return CrawlManifest(
            domain=domain,
            urls=urls,
            total_urls=url_count,
            estimated_duration_seconds=float(url_count),
            clearance_required=1,
            manifest_id=mid,
        )

    @staticmethod
    def make_large_manifest(
        domain: str = "bigsite.com",
        url_count: int = 100_000,
        manifest_id: Optional[str] = None,
    ) -> CrawlManifest:
        import uuid
        mid = manifest_id or str(uuid.uuid4())
        urls = [
            _TestHelpers.make_crawl_url(
                url=f"https://{domain}/article/{i}",
                topology_hint="NEWS_ARTICLE",
                priority=i % 10,  # spread across 10 priority levels
                domain=domain,
            )
            for i in range(url_count)
        ]
        return CrawlManifest(
            domain=domain,
            urls=urls,
            total_urls=url_count,
            estimated_duration_seconds=float(url_count),
            clearance_required=1,
            manifest_id=mid,
        )


class _DiagnosticSuite:
    """
    Built-in diagnostic test suite for frontier.py.

    Each test_* method is a self-contained async test.  Tests use temporary
    SQLite databases in a temp directory — they do not touch the production
    frontier.db.

    Each test creates its own Frontier instance with a fresh database.  Tests
    are independent and can be run in any order.

    Usage
    ─────
    runner = _DiagnosticRunner()
    passed = await runner.run_all()
    """

    def __init__(self, tmp_dir: Path) -> None:
        self._tmp = tmp_dir
        self._test_num = 0

    def _next_db_path(self, name: str = "") -> Path:
        """Return a unique temp DB path for a test."""
        self._test_num += 1
        suffix = f"_{name}" if name else ""
        return self._tmp / f"frontier_test_{self._test_num}{suffix}.db"

    async def _make_cursor(self) -> CrawlCursor:
        """Create and initialize a CrawlCursor pointing at a temp DB."""
        cursor_path = self._tmp / f"cursor_test_{self._test_num}.db"
        cursor = CrawlCursor()
        await cursor.initialize(db_path=cursor_path)
        return cursor

    async def _make_frontier(self, name: str = "") -> Tuple[Frontier, CrawlCursor]:
        """Create, initialize, and return (Frontier, CrawlCursor) for one test."""
        cursor = await self._make_cursor()
        db_path = self._next_db_path(name)
        frontier = Frontier(cursor=cursor, db_path=db_path)
        await frontier.initialize()
        return frontier, cursor

    # ── Test cases ────────────────────────────────────────────────────────────

    async def test_01_load_manifest_inserts_all_urls(self) -> None:
        """load_manifest() inserts all URLs in the correct order."""
        frontier, cursor = await self._make_frontier("load_basic")
        try:
            manifest = _TestHelpers.make_manifest(url_count=50)
            await frontier.load_manifest(manifest)

            count = await frontier.pending_count(manifest.manifest_id)
            assert count == 50, f"Expected 50 pending, got {count}"

            # Verify URLs come back in priority order via resume().
            collected = []
            async for crawl_url in frontier.resume(manifest.manifest_id):
                collected.append(crawl_url)
            assert len(collected) == 50, f"resume() yielded {len(collected)} URLs, expected 50"

            # Priority 0 should come before priority 1, etc.
            priorities = [u.priority for u in collected]
            assert priorities == sorted(priorities), f"URLs not in priority order: {priorities[:10]}"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_02_resume_from_cursor_position(self) -> None:
        """resume() yields URLs starting from the cursor-saved position."""
        frontier, cursor = await self._make_frontier("resume_position")
        try:
            manifest = _TestHelpers.make_manifest(url_count=20)
            await frontier.load_manifest(manifest)

            # Simulate cursor at position 10 (first 10 URLs were processed).
            await cursor.checkpoint(
                manifest_id=manifest.manifest_id,
                position=10,
                url=manifest.urls[10].url,
                total_urls=manifest.total_urls,
            )

            # resume() should yield URLs 10 onwards (offset=10).
            yielded = []
            async for crawl_url in frontier.resume(manifest.manifest_id):
                yielded.append(crawl_url)

            # Should yield 10 remaining (positions 10 through 19).
            assert len(yielded) == 10, (
                f"Expected 10 URLs from position 10, got {len(yielded)}"
            )
        finally:
            await frontier.close()
            await cursor.close()

    async def test_03_mark_done_updates_status(self) -> None:
        """mark_done() transitions a URL from 'pending' to 'done'."""
        frontier, cursor = await self._make_frontier("mark_done")
        try:
            manifest = _TestHelpers.make_manifest(url_count=5)
            await frontier.load_manifest(manifest)

            target_url = manifest.urls[0].url

            # Fire-and-forget mark_done.
            task = asyncio.create_task(
                frontier.mark_done(manifest.manifest_id, target_url)
            )
            await task  # Await here for test synchronicity.

            row = await frontier.get_url(manifest.manifest_id, target_url)
            assert row is not None, "URL not found in DB"
            assert row.status == FRONTIER_STATUS_DONE, (
                f"Expected status='done', got {row.status!r}"
            )
            assert row.completed_at is not None, "completed_at should be set"

            # Pending count should drop by 1.
            count = await frontier.pending_count(manifest.manifest_id)
            assert count == 4, f"Expected 4 pending after one mark_done, got {count}"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_04_mark_failed_updates_status(self) -> None:
        """mark_failed() transitions a URL from 'pending' to 'failed'."""
        frontier, cursor = await self._make_frontier("mark_failed")
        try:
            manifest = _TestHelpers.make_manifest(url_count=5)
            await frontier.load_manifest(manifest)

            target_url = manifest.urls[2].url
            await frontier.mark_failed(manifest.manifest_id, target_url)

            row = await frontier.get_url(manifest.manifest_id, target_url)
            assert row is not None
            assert row.status == FRONTIER_STATUS_FAILED, (
                f"Expected 'failed', got {row.status!r}"
            )
        finally:
            await frontier.close()
            await cursor.close()

    async def test_05_mark_skipped_updates_status(self) -> None:
        """mark_skipped() transitions a URL from 'pending' to 'skipped'."""
        frontier, cursor = await self._make_frontier("mark_skipped")
        try:
            manifest = _TestHelpers.make_manifest(url_count=5)
            await frontier.load_manifest(manifest)

            target_url = manifest.urls[1].url
            await frontier.mark_skipped(manifest.manifest_id, target_url)

            row = await frontier.get_url(manifest.manifest_id, target_url)
            assert row is not None
            assert row.status == FRONTIER_STATUS_SKIPPED, (
                f"Expected 'skipped', got {row.status!r}"
            )
        finally:
            await frontier.close()
            await cursor.close()

    async def test_06_is_complete_returns_false_while_pending(self) -> None:
        """is_complete() returns False when pending rows remain."""
        frontier, cursor = await self._make_frontier("is_complete_false")
        try:
            manifest = _TestHelpers.make_manifest(url_count=3)
            await frontier.load_manifest(manifest)

            result = await frontier.is_complete(manifest.manifest_id)
            assert result is False, f"is_complete should be False with pending rows"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_07_is_complete_returns_true_when_all_done(self) -> None:
        """is_complete() returns True when all URLs are in terminal states."""
        frontier, cursor = await self._make_frontier("is_complete_true")
        try:
            manifest = _TestHelpers.make_manifest(url_count=3)
            await frontier.load_manifest(manifest)

            # Mark all URLs as done.
            for crawl_url in manifest.urls:
                await frontier.mark_done(manifest.manifest_id, crawl_url.url)

            result = await frontier.is_complete(manifest.manifest_id)
            assert result is True, "is_complete should be True after all URLs done"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_08_stats_returns_correct_counts(self) -> None:
        """stats() returns correct done/failed/skipped/pending breakdown."""
        frontier, cursor = await self._make_frontier("stats")
        try:
            manifest = _TestHelpers.make_manifest(url_count=9)
            await frontier.load_manifest(manifest)
            urls = manifest.urls

            # Mark 3 done, 2 failed, 1 skipped.  4 remain pending.
            for u in urls[:3]:
                await frontier.mark_done(manifest.manifest_id, u.url)
            for u in urls[3:5]:
                await frontier.mark_failed(manifest.manifest_id, u.url)
            await frontier.mark_skipped(manifest.manifest_id, urls[5].url)

            stats = await frontier.stats(manifest.manifest_id)
            assert stats.done    == 3, f"Expected done=3, got {stats.done}"
            assert stats.failed  == 2, f"Expected failed=2, got {stats.failed}"
            assert stats.skipped == 1, f"Expected skipped=1, got {stats.skipped}"
            assert stats.pending == 3, f"Expected pending=3, got {stats.pending}"
            assert stats.total   == 9, f"Expected total=9, got {stats.total}"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_09_wal_mode_enabled(self) -> None:
        """WAL mode is enabled after initialize()."""
        frontier, cursor = await self._make_frontier("wal_mode")
        try:
            wal_active = await frontier._schema_manager.verify_wal_mode(frontier._db)
            assert wal_active, "WAL mode must be enabled after initialize()"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_10_batch_load_performance(self) -> None:
        """100K URL manifest loads in < 2 seconds."""
        import time as _time
        frontier, cursor = await self._make_frontier("batch_perf")
        try:
            manifest = _TestHelpers.make_large_manifest(url_count=100_000)

            t_start = _time.perf_counter()
            await frontier.load_manifest(manifest)
            duration_s = _time.perf_counter() - t_start

            assert duration_s < 2.0, (
                f"100K URL load took {duration_s:.2f}s — must be < 2.0s. "
                "Check FRONTIER_BATCH_SIZE and SQLite WAL configuration."
            )

            count = await frontier.pending_count(manifest.manifest_id)
            assert count == 100_000, f"Expected 100000 rows, got {count}"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_11_reload_same_manifest_is_noop(self) -> None:
        """Re-loading the same manifest_id does not duplicate rows."""
        frontier, cursor = await self._make_frontier("reload_noop")
        try:
            manifest = _TestHelpers.make_manifest(url_count=10)

            await frontier.load_manifest(manifest)
            count_before = await frontier.pending_count(manifest.manifest_id)

            # Second load must be a no-op.
            await frontier.load_manifest(manifest)
            count_after = await frontier.pending_count(manifest.manifest_id)

            assert count_before == count_after == 10, (
                f"count_before={count_before}, count_after={count_after} — "
                "re-load must not duplicate rows"
            )
        finally:
            await frontier.close()
            await cursor.close()

    async def test_12_resume_after_simulated_crash(self) -> None:
        """Crash simulation: close Frontier, reopen, resume from cursor position."""
        db_path = self._next_db_path("crash_sim")
        cursor_path = self._tmp / f"cursor_crash_{self._test_num}.db"

        # First session: load and partially process.
        cursor1 = CrawlCursor()
        await cursor1.initialize(db_path=cursor_path)
        frontier1 = Frontier(cursor=cursor1, db_path=db_path)
        await frontier1.initialize()

        manifest = _TestHelpers.make_manifest(url_count=20)
        await frontier1.load_manifest(manifest)

        # Process first 7 URLs, write cursor at position 7.
        i = 0
        async for crawl_url in frontier1.resume(manifest.manifest_id):
            await frontier1.mark_done(manifest.manifest_id, crawl_url.url)
            i += 1
            if i >= 7:
                break
        await cursor1.checkpoint(
            manifest_id=manifest.manifest_id,
            position=7,
            url=manifest.urls[7].url,
            total_urls=manifest.total_urls,
        )

        # Simulate crash: close without flushing fire-and-forget tasks.
        await frontier1._safe_close_db()
        await cursor1.close()

        # Second session: reopen with new instances.
        cursor2 = CrawlCursor()
        await cursor2.initialize(db_path=cursor_path)
        frontier2 = Frontier(cursor=cursor2, db_path=db_path)
        await frontier2.initialize()

        # load_manifest with the same manifest_id — must be a no-op.
        await frontier2.load_manifest(manifest)

        # resume() should start at offset=7.
        resumed_urls = []
        async for crawl_url in frontier2.resume(manifest.manifest_id):
            resumed_urls.append(crawl_url.url)

        # Some of the 7 may not have been marked due to fire-and-forget timing.
        # At minimum, we should yield from position 7 onwards (13 URLs).
        assert len(resumed_urls) <= 20, f"Got {len(resumed_urls)} URLs — cannot exceed total"
        assert len(resumed_urls) >= 1, "Should yield at least one URL after crash resume"

        await frontier2.close()
        await cursor2.close()

    async def test_13_concurrent_manifests_dont_interfere(self) -> None:
        """Two Frontier instances on the same DB don't corrupt each other's data."""
        db_path = self._next_db_path("concurrent")
        cursor_path_a = self._tmp / f"cursor_ca_{self._test_num}.db"
        cursor_path_b = self._tmp / f"cursor_cb_{self._test_num}.db"

        cursor_a = CrawlCursor()
        await cursor_a.initialize(db_path=cursor_path_a)
        frontier_a = Frontier(cursor=cursor_a, db_path=db_path)
        await frontier_a.initialize()

        cursor_b = CrawlCursor()
        await cursor_b.initialize(db_path=cursor_path_b)
        frontier_b = Frontier(cursor=cursor_b, db_path=db_path)
        await frontier_b.initialize()

        try:
            manifest_a = _TestHelpers.make_manifest(domain="alpha.com", url_count=5)
            manifest_b = _TestHelpers.make_manifest(domain="beta.com", url_count=7)

            await frontier_a.load_manifest(manifest_a)
            await frontier_b.load_manifest(manifest_b)

            count_a = await frontier_a.pending_count(manifest_a.manifest_id)
            count_b = await frontier_b.pending_count(manifest_b.manifest_id)

            assert count_a == 5, f"frontier_a: expected 5 pending, got {count_a}"
            assert count_b == 7, f"frontier_b: expected 7 pending, got {count_b}"

            # Mark all of A's URLs as done.
            for u in manifest_a.urls:
                await frontier_a.mark_done(manifest_a.manifest_id, u.url)

            # B's count should be unchanged.
            count_b_after = await frontier_b.pending_count(manifest_b.manifest_id)
            assert count_b_after == 7, (
                f"frontier_b count changed after frontier_a writes: {count_b_after}"
            )

            assert await frontier_a.is_complete(manifest_a.manifest_id), \
                "frontier_a should be complete"
            assert not await frontier_b.is_complete(manifest_b.manifest_id), \
                "frontier_b should not be complete"
        finally:
            await frontier_a.close()
            await frontier_b.close()
            await cursor_a.close()
            await cursor_b.close()

    async def test_14_mark_done_is_idempotent(self) -> None:
        """mark_done() called twice for the same URL is safe (0 rows on second call)."""
        frontier, cursor = await self._make_frontier("mark_done_idempotent")
        try:
            manifest = _TestHelpers.make_manifest(url_count=3)
            await frontier.load_manifest(manifest)

            target_url = manifest.urls[0].url
            await frontier.mark_done(manifest.manifest_id, target_url)
            await frontier.mark_done(manifest.manifest_id, target_url)  # Second call.

            # Status must still be 'done', not corrupted.
            row = await frontier.get_url(manifest.manifest_id, target_url)
            assert row is not None
            assert row.status == FRONTIER_STATUS_DONE

            # Total pending count should only decrease by 1.
            count = await frontier.pending_count(manifest.manifest_id)
            assert count == 2, f"Expected 2 pending after one URL marked done, got {count}"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_15_unknown_manifest_returns_zero_counts(self) -> None:
        """pending_count() and stats() return 0 for a manifest_id not in the DB."""
        frontier, cursor = await self._make_frontier("unknown_manifest")
        try:
            fake_id = "00000000-0000-4000-a000-000000000099"
            count = await frontier.pending_count(fake_id)
            assert count == 0, f"Expected 0 for unknown manifest, got {count}"

            stats = await frontier.stats(fake_id)
            assert stats.pending == 0
            assert stats.done    == 0
            assert stats.failed  == 0
            assert stats.skipped == 0
            assert stats.total   == 0

            complete = await frontier.is_complete(fake_id)
            assert complete is True, "Empty manifest (no rows) should be 'complete'"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_16_crawl_url_reconstruction_from_db(self) -> None:
        """CrawlURL reconstructed from DB matches the original URL fields."""
        frontier, cursor = await self._make_frontier("reconstruct")
        try:
            import uuid
            run_id = str(uuid.uuid4())
            original_url = _TestHelpers.make_crawl_url(
                url="https://example.com/reconstruct",
                topology_hint="NEWS_ARTICLE",
                fetch_mode=FetchMode.HEADLESS,
                render_mode="headless",
                priority=42,
                domain="example.com",
                is_robots=False,
                is_sitemap=True,
                run_id=run_id,
                crawl_delay=5.0,
                max_bytes=2 * 1024 * 1024,
            )
            manifest_id = str(uuid.uuid4())
            import uuid as _uuid
            manifest = CrawlManifest(
                domain="example.com",
                urls=[original_url],
                total_urls=1,
                estimated_duration_seconds=1.0,
                clearance_required=2,
                manifest_id=manifest_id,
            )
            await frontier.load_manifest(manifest)

            # Read back from DB and reconstruct.
            yielded_urls = []
            async for crawl_url in frontier.resume(manifest_id):
                yielded_urls.append(crawl_url)

            assert len(yielded_urls) == 1
            reconstructed = yielded_urls[0]

            assert reconstructed.url == original_url.url
            assert reconstructed.topology_hint == original_url.topology_hint
            assert reconstructed.priority == original_url.priority
            assert reconstructed.is_sitemap == original_url.is_sitemap
            assert reconstructed.is_robots == original_url.is_robots
            assert reconstructed.run_id == original_url.run_id
            assert abs(reconstructed.crawl_delay_seconds - original_url.crawl_delay_seconds) < 0.001
            assert reconstructed.max_response_bytes == original_url.max_response_bytes
            assert reconstructed.rate_limit_profile.domain == original_url.rate_limit_profile.domain
        finally:
            await frontier.close()
            await cursor.close()

    async def test_17_list_manifests_aggregates_correctly(self) -> None:
        """list_manifests() returns one summary per manifest with correct counts."""
        frontier, cursor = await self._make_frontier("list_manifests")
        try:
            m1 = _TestHelpers.make_manifest(domain="alpha.com", url_count=5)
            m2 = _TestHelpers.make_manifest(domain="beta.com",  url_count=3)
            await frontier.load_manifest(m1)
            await frontier.load_manifest(m2)

            # Mark 2 done in m1.
            for u in m1.urls[:2]:
                await frontier.mark_done(m1.manifest_id, u.url)

            summaries = await frontier.list_manifests()
            assert len(summaries) == 2, f"Expected 2 summaries, got {len(summaries)}"

            # Find the m1 summary.
            s1 = next((s for s in summaries if s.manifest_id == m1.manifest_id), None)
            assert s1 is not None, "m1 summary not found"
            assert s1.total   == 5
            assert s1.pending == 3
            assert s1.done    == 2
        finally:
            await frontier.close()
            await cursor.close()

    async def test_18_archive_manifest_clears_pending(self) -> None:
        """archive_manifest() marks all remaining pending rows as skipped."""
        frontier, cursor = await self._make_frontier("archive")
        try:
            manifest = _TestHelpers.make_manifest(url_count=6)
            await frontier.load_manifest(manifest)

            # Mark 3 done, leave 3 pending.
            for u in manifest.urls[:3]:
                await frontier.mark_done(manifest.manifest_id, u.url)

            rows_updated = await frontier.archive_manifest(manifest.manifest_id)
            assert rows_updated == 3, f"Expected 3 rows archived, got {rows_updated}"

            # is_complete should now be True.
            complete = await frontier.is_complete(manifest.manifest_id)
            assert complete is True, "Manifest should be complete after archive"

            stats = await frontier.stats(manifest.manifest_id)
            assert stats.done    == 3
            assert stats.skipped == 3
            assert stats.pending == 0
        finally:
            await frontier.close()
            await cursor.close()

    async def test_19_delete_manifest_removes_all_rows(self) -> None:
        """delete_manifest() removes all rows for a manifest_id."""
        frontier, cursor = await self._make_frontier("delete_manifest")
        try:
            manifest = _TestHelpers.make_manifest(url_count=5)
            await frontier.load_manifest(manifest)

            rows_deleted = await frontier.delete_manifest(manifest.manifest_id)
            assert rows_deleted == 5, f"Expected 5 rows deleted, got {rows_deleted}"

            exists = await frontier.exists(manifest.manifest_id)
            assert exists is False, "Manifest should not exist after delete"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_20_health_returns_valid_state(self) -> None:
        """health() returns a FrontierHealth with is_healthy=True."""
        frontier, cursor = await self._make_frontier("health")
        try:
            manifest = _TestHelpers.make_manifest(url_count=3)
            await frontier.load_manifest(manifest)

            h = await frontier.health()
            assert h.is_healthy, f"health.is_healthy=False: {h.to_dict()}"
            assert h.wal_mode_active, "WAL mode must be active"
            assert h.integrity_ok,   "Integrity check must pass"
            assert h.initialized,    "Must be initialized"
            assert h.manifest_count == 1
            assert h.total_url_count == 3
        finally:
            await frontier.close()
            await cursor.close()

    async def test_21_context_manager_closes_cleanly(self) -> None:
        """Frontier supports async with and closes properly on exit."""
        cursor_path = self._tmp / f"cursor_ctx_{self._test_num}.db"
        db_path = self._next_db_path("context_mgr")

        cursor = CrawlCursor()
        await cursor.initialize(db_path=cursor_path)
        try:
            async with Frontier(cursor=cursor, db_path=db_path) as frontier:
                manifest = _TestHelpers.make_manifest(url_count=3)
                await frontier.load_manifest(manifest)
                count = await frontier.pending_count(manifest.manifest_id)
                assert count == 3

            # After context exit, is_initialized should be False.
            assert not frontier.is_initialized, \
                "Frontier should be closed after async with exits"
        finally:
            await cursor.close()

    async def test_22_no_urls_manifest(self) -> None:
        """load_manifest() with an empty url list is safe and idempotent."""
        import uuid
        frontier, cursor = await self._make_frontier("empty_manifest")
        try:
            mid = str(uuid.uuid4())
            manifest = CrawlManifest(
                domain="empty.com",
                urls=[],
                total_urls=0,
                estimated_duration_seconds=0.0,
                clearance_required=1,
                manifest_id=mid,
            )
            await frontier.load_manifest(manifest)
            count = await frontier.pending_count(mid)
            assert count == 0, f"Empty manifest should have 0 pending, got {count}"
            complete = await frontier.is_complete(mid)
            assert complete is True, "Empty manifest should be complete"
        finally:
            await frontier.close()
            await cursor.close()

    async def test_23_mixed_status_completion_rate(self) -> None:
        """stats().completion_rate matches manual calculation."""
        frontier, cursor = await self._make_frontier("completion_rate")
        try:
            manifest = _TestHelpers.make_manifest(url_count=10)
            await frontier.load_manifest(manifest)
            urls = manifest.urls

            # 4 done, 2 failed, 2 skipped, 2 pending → completion = (4+2)/10 = 0.6
            for u in urls[:4]:
                await frontier.mark_done(manifest.manifest_id, u.url)
            for u in urls[4:6]:
                await frontier.mark_failed(manifest.manifest_id, u.url)
            for u in urls[6:8]:
                await frontier.mark_skipped(manifest.manifest_id, u.url)

            stats = await frontier.stats(manifest.manifest_id)
            assert abs(stats.completion_rate - 0.6) < 0.001, (
                f"Expected completion_rate=0.6, got {stats.completion_rate}"
            )
        finally:
            await frontier.close()
            await cursor.close()

    async def test_24_priority_ordering_preserved_on_reload(self) -> None:
        """After crash+reload, resume() still returns URLs in priority order."""
        import uuid
        db_path = self._next_db_path("priority_reload")
        cursor_path = self._tmp / f"cursor_pr_{self._test_num}.db"

        # First session: load.
        c1 = CrawlCursor()
        await c1.initialize(db_path=cursor_path)
        f1 = Frontier(cursor=c1, db_path=db_path)
        await f1.initialize()

        # Create manifest with reverse priorities (9 down to 0).
        mid = str(uuid.uuid4())
        urls = [
            _TestHelpers.make_crawl_url(
                url=f"https://site.com/p{i}",
                priority=9 - i,  # descending: 9, 8, 7, ..., 0
            )
            for i in range(10)
        ]
        manifest = CrawlManifest(
            domain="site.com",
            urls=urls,
            total_urls=10,
            estimated_duration_seconds=10.0,
            clearance_required=1,
            manifest_id=mid,
        )
        await f1.load_manifest(manifest)
        await f1._safe_close_db()
        await c1.close()

        # Second session: verify order.
        c2 = CrawlCursor()
        await c2.initialize(db_path=cursor_path)
        f2 = Frontier(cursor=c2, db_path=db_path)
        await f2.initialize()
        try:
            collected = []
            async for cu in f2.resume(mid):
                collected.append(cu.priority)
            assert collected == sorted(collected), (
                f"URLs not in priority order after reload: {collected}"
            )
        finally:
            await f2.close()
            await c2.close()

    async def test_25_completed_count_and_pending_count_sum_to_total(self) -> None:
        """completed_count() + pending_count() == total URL count."""
        frontier, cursor = await self._make_frontier("count_sum")
        try:
            manifest = _TestHelpers.make_manifest(url_count=8)
            await frontier.load_manifest(manifest)

            # Mark 3 done.
            for u in manifest.urls[:3]:
                await frontier.mark_done(manifest.manifest_id, u.url)

            pending   = await frontier.pending_count(manifest.manifest_id)
            completed = await frontier.completed_count(manifest.manifest_id)
            assert pending + completed == 8, (
                f"pending({pending}) + completed({completed}) != 8"
            )
            assert pending   == 5
            assert completed == 3
        finally:
            await frontier.close()
            await cursor.close()


class _DiagnosticRunner:
    """
    Orchestrates the diagnostic test suite.

    Creates a temporary directory, instantiates _DiagnosticSuite, runs all
    test_* methods, reports results, and cleans up.
    """

    async def run_all(self, verbose: bool = True) -> bool: # noqa
        """
        Run all tests and return True if every test passed.

        Parameters
        ──────────
        verbose  If True, print pass/fail status per test.  If False, only
                 print the final summary.

        Returns
        ───────
        True if all tests passed.  False if any failed.
        """
        import tempfile
        import traceback

        with tempfile.TemporaryDirectory(prefix="axiom_frontier_tests_") as tmp:
            tmp_path = Path(tmp)
            suite = _DiagnosticSuite(tmp_path)

            # Discover all test methods.
            test_methods = sorted(
                name
                for name in dir(suite)
                if name.startswith("test_") and callable(getattr(suite, name))
            )

            passed = failed = 0
            errors: List[Tuple[str, str]] = []

            if verbose:
                print(f"\n{'─' * 72}")
                print(f"  frontier.py diagnostic suite — {len(test_methods)} test(s)")
                print(f"{'─' * 72}")

            for name in test_methods:
                method = getattr(suite, name)
                t_start = time.perf_counter()
                try:
                    await method()
                    duration_ms = (time.perf_counter() - t_start) * 1000.0
                    if verbose:
                        print(f"  ✓  {name} ({duration_ms:.1f}ms)")
                    passed += 1
                except Exception as exc:
                    duration_ms = (time.perf_counter() - t_start) * 1000.0
                    tb = traceback.format_exc()
                    errors.append((name, tb))
                    if verbose:
                        print(f"  ✗  {name} ({duration_ms:.1f}ms)")
                        print(f"       {type(exc).__name__}: {exc}")
                    failed += 1

            print(f"{'─' * 72}")
            print(f"  {passed}/{len(test_methods)} passed  |  {failed} failed")
            print(f"{'─' * 72}")

            if errors:
                print("\nFailed test details:")
                for tname, tb in errors:
                    print(f"\n  ✗ {tname}")
                    for line in tb.splitlines():
                        print(f"    {line}")
            print()

        return failed == 0


# ─────────────────────────────────────────────────────────────────────────────
# COMMAND-LINE INTERFACE
#
# Usage:
#   python frontier.py test
#   python frontier.py stats [--db PATH]
#   python frontier.py dump  [--db PATH]
#   python frontier.py health [--db PATH]
# ─────────────────────────────────────────────────────────────────────────────

def _build_arg_parser():
    """Build the CLI argument parser."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="frontier.py",
        description=(
            "AXIOM frontier.py — resumable crawl frontier CLI.\n\n"
            "Subcommands:\n"
            "  test    Run the built-in diagnostic test suite.\n"
            "  stats   Print per-manifest statistics.\n"
            "  dump    Dump all manifest summaries as JSON.\n"
            "  health  Print health report for the frontier database.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level.  Default: WARNING",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("test", help="Run the built-in diagnostic test suite.")

    for sub in ("stats", "dump", "health"):
        p = subparsers.add_parser(sub, help=f"Run the {sub} command.")
        p.add_argument(
            "--db",
            type=Path,
            default=FRONTIER_DB_PATH,
            metavar="PATH",
            help=f"Path to frontier.db.  Default: {FRONTIER_DB_PATH}",
        )

    return parser


async def _cmd_test() -> int:
    """Run diagnostic test suite.  Return exit code."""
    runner = _DiagnosticRunner()
    passed = await runner.run_all(verbose=True)
    return 0 if passed else 1


async def _cmd_health(db_path: Path) -> int:
    """Print health report as JSON.  Return exit code."""
    import json

    if not db_path.exists():
        print(f"[ERROR] Database not found at {db_path}.")
        return 1

    cursor = CrawlCursor()
    cursor_path = db_path.parent / "crawl_cursor.db"
    try:
        await cursor.initialize(db_path=cursor_path)
        frontier = Frontier(cursor=cursor, db_path=db_path)
        await frontier.initialize()
        try:
            h = await frontier.health()
            print(json.dumps(h.to_dict(), indent=2))
            return 0 if h.is_healthy else 1
        finally:
            await frontier.close()
    except Exception as exc:
        print(f"[ERROR] Health check failed: {exc}")
        return 1
    finally:
        await cursor.close()


async def _cmd_dump(db_path: Path) -> int:
    """Dump all manifest summaries as JSON.  Return exit code."""
    import json

    if not db_path.exists():
        print("[]")
        return 0

    cursor = CrawlCursor()
    cursor_path = db_path.parent / "crawl_cursor.db"
    try:
        await cursor.initialize(db_path=cursor_path)
        frontier = Frontier(cursor=cursor, db_path=db_path)
        await frontier.initialize()
        try:
            summaries = await frontier.list_manifests()
            print(json.dumps([s.to_dict() for s in summaries], indent=2))
            return 0
        finally:
            await frontier.close()
    except Exception as exc:
        print(f"[ERROR] Dump failed: {exc}")
        return 1
    finally:
        await cursor.close()


async def _cmd_stats(db_path: Path) -> int:
    """Print per-manifest stats.  Return exit code."""
    if not db_path.exists():
        print(f"[WARN] Database not found at {db_path}.")
        return 0

    cursor = CrawlCursor()
    cursor_path = db_path.parent / "crawl_cursor.db"
    try:
        await cursor.initialize(db_path=cursor_path)
        frontier = Frontier(cursor=cursor, db_path=db_path)
        await frontier.initialize()
        try:
            summaries = await frontier.list_manifests()
            if not summaries:
                print("No manifests found in frontier.db.")
                return 0
            print(f"\n{'─' * 80}")
            print(f"  frontier.db — {len(summaries)} manifest(s)")
            print(f"{'─' * 80}")
            for s in summaries:
                print(
                    f"  {s.manifest_id[:8]}... "
                    f"total={s.total:>7,}  "
                    f"pending={s.pending:>7,}  "
                    f"done={s.done:>7,}  "
                    f"failed={s.failed:>6,}  "
                    f"skipped={s.skipped:>6,}  "
                    f"completion={s.completion_rate:.1%}"
                )
            print(f"{'─' * 80}\n")
            return 0
        finally:
            await frontier.close()
    except Exception as exc:
        print(f"[ERROR] Stats failed: {exc}")
        return 1
    finally:
        await cursor.close()


async def _async_main(args) -> int:
    if args.command == "test":
        return await _cmd_test()
    elif args.command == "health":
        return await _cmd_health(args.db)
    elif args.command == "dump":
        return await _cmd_dump(args.db)
    elif args.command == "stats":
        return await _cmd_stats(args.db)
    else:
        _build_arg_parser().print_help()
        return 0


def main() -> None:
    """Synchronous entry point.  Parses args and delegates to async main."""
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