"""
signal_kernel/checkpoint_monitor.py
=====================================
Read-only health monitor for the checkpoint system managed by crond and
mft_checkpoint.sh. This module is the observability and verification layer
for the checkpoint subsystem — it never writes, never modifies archives,
never starts or restarts crond, and never holds checkpoint data between calls.

═══════════════════════════════════════════════════════════════════════════════
OWNERSHIP MODEL
═══════════════════════════════════════════════════════════════════════════════

  mft_checkpoint.sh    owns: writing archives to CHECKPOINT_DIR, rotation
  restore.sh           owns: extracting archives to STORE_DIR
  checkpoint_monitor   owns: finding archives, verifying integrity, health

The three processes are strictly separated. checkpoint_monitor.py reads
archive files and sidecar hash files. It never opens a file for writing.
It never forks or execs. It never modifies STORE_DIR directly.

═══════════════════════════════════════════════════════════════════════════════
PUBLIC API — EXACTLY TWO FUNCTIONS
═══════════════════════════════════════════════════════════════════════════════

  restore(checkpoint_dir) → bytes
    Find the most recent valid checkpoint archive in checkpoint_dir,
    verify it through all six integrity layers, and return its raw bytes.
    Iterates archives newest-to-oldest if the latest fails verification.
    Raises CheckpointCorruptionError when all candidates are exhausted.
    The caller is responsible for extraction and deserialization — this
    function returns raw .tar.gz bytes and nothing else.

  health(checkpoint_dir, *, restore_invoked_at_startup, restore_succeeded)
    → CheckpointHealth
    Return a point-in-time health snapshot of the checkpoint system.
    Never raises. All problems are encoded in the returned dataclass.
    The caller (pipeline.py, TAG startup, Witness integration) decides
    how to respond. Emits CheckpointStalenessWarning via warnings.warn()
    when the most recent archive exceeds the staleness threshold.

═══════════════════════════════════════════════════════════════════════════════
INTEGRITY VERIFICATION MODEL — SIX LAYERS
═══════════════════════════════════════════════════════════════════════════════

  Layer 1 — File existence and type
    The archive path must exist, be a regular file (not a symlink to a
    directory, not a device, not a FIFO), and be readable by the current
    process effective UID.

  Layer 2 — File size sanity bounds
    Size must be ≥ _MIN_ARCHIVE_SIZE_BYTES (512 B) — a valid gzip-compressed
    tar of four non-empty files cannot be smaller.
    Size must be ≤ _MAX_ARCHIVE_SIZE_BYTES (1 GiB) — a sanity cap that
    prevents checkpoint_monitor from being used to exhaust memory.

  Layer 3 — SHA-256 vs sidecar hash
    If a .sha256 sidecar file exists adjacent to the archive (written by
    mft_checkpoint.sh), compute the SHA-256 of the archive in streaming
    64 KB chunks and compare against the sidecar. A mismatch means the
    archive was partially written or tampered with — reject immediately.
    If no sidecar exists, fall back to tar-stream-only verification
    (layers 4-6) and log at DEBUG. Absence of a sidecar is not a failure;
    many deployments may not produce sidecar files.

  Layer 4 — gzip stream integrity
    Open the archive with Python's tarfile module in streaming mode
    (mode="r:gz"). A truncated or corrupt gzip stream raises tarfile
    exceptions before any member data is read — these are caught and
    translated to failure outcomes.

  Layer 5 — Manifest completeness
    Verify that the archive contains exactly the four files named in
    contracts.STORE_FILE_NAMES: topology_router.pt, recipe_registry.mmap,
    phase_states.mmap, structural_layer.pt. Missing files, extra files,
    and duplicate member names are all rejection conditions.

  Layer 6 — Member safety
    Each member must be a regular file (not a directory, symlink, block
    device, or character device). No member may have an absolute path or
    a path traversal component (../../). Each member's recorded size must
    be ≥ 0 and ≤ _MAX_MEMBER_SIZE_BYTES (4 GiB).

  A single layer failure is sufficient to reject an archive. All layers
  are tested in order; the first failure short-circuits. The failure reason
  is captured verbatim in _IntegrityOutcome.failure_reason and propagated
  to log records and CheckpointCorruptionError.

═══════════════════════════════════════════════════════════════════════════════
STALENESS MODEL
═══════════════════════════════════════════════════════════════════════════════

  An archive is stale when its mtime exceeds CHECKPOINT_STALE_THRESHOLD_SECONDS
  (1800 s = 30 min = 2× the 15-minute crond interval). Staleness implies
  crond has silently exited — it has missed at least two consecutive
  scheduled runs.

  Staleness is observable, not fatal. health() reports it via:
    - CheckpointHealth.crond_alive = False
    - CheckpointHealth.minutes_since_last_checkpoint > threshold
    - warnings.warn(CheckpointStalenessWarning, ...)

  Process 1 (grep pipeline) never checks or responds to staleness. It
  continues executing regardless of the checkpoint system's health.

  Clock skew detection: if an archive's mtime is more than
  _FUTURE_MTIME_TOLERANCE_SECONDS (300 s) in the future, a filesystem
  clock skew is logged at WARNING and the archive is treated as non-stale
  (we cannot determine its true age from a future timestamp).

═══════════════════════════════════════════════════════════════════════════════
CONCURRENT-WRITE SAFETY
═══════════════════════════════════════════════════════════════════════════════

  mft_checkpoint.sh writes archives atomically: it writes to a .tmp file
  and renames it to the canonical path only after integrity verification
  passes. A file at the canonical path is therefore always complete — it
  was never partially visible to readers.

  Despite this guarantee, checkpoint_monitor implements three additional
  concurrent-write defenses:

  1. Re-stat after SHA-256: after computing the archive's SHA-256 in
     _verify_integrity(), the file is re-stated. If the size changed between
     the initial stat (captured in _ArchiveStat) and the re-stat, a
     concurrent write is detected and the archive is rejected.

  2. Re-stat before read: in restore(), the file is re-stated immediately
     before path.read_bytes() to detect any replacement that occurred
     between _verify_integrity() and the read.

  3. Post-read SHA-256 comparison: after path.read_bytes(), the SHA-256 of
     the returned bytes is recomputed and compared against the SHA-256
     captured during _verify_integrity(). A mismatch indicates the file was
     replaced between verification and read. The archive is rejected and the
     next candidate is tried.

═══════════════════════════════════════════════════════════════════════════════
THREADING AND STATEFULNESS
═══════════════════════════════════════════════════════════════════════════════

  checkpoint_monitor is synchronous and module-stateless. Every public call
  re-reads from disk. No in-memory caching. No background thread. No global
  mutable variables. All state lives in the filesystem and in local variables
  within each call.

  Thread safety: the two public functions are re-entrant. They do not share
  mutable state. Concurrent callers from different threads will each
  independently perform a fresh disk scan, which is the correct behavior.

═══════════════════════════════════════════════════════════════════════════════
DEPENDENCY DIRECTION
═══════════════════════════════════════════════════════════════════════════════

  checkpoint_monitor → contracts  (imports CheckpointHealth, CheckpointRecord,
                                   STORE_FILE_NAMES, CHECKPOINT_* constants)
  checkpoint_monitor → exceptions (imports CheckpointCorruptionError,
                                   CheckpointStalenessWarning)
  checkpoint_monitor ← pipeline.py, TAG startup sequence, Witness integration

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import errno
import hashlib
import logging
import os
import re
import stat
import tarfile
import time # noqa
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

from signal_kernel.contracts import (
    CHECKPOINT_INTERVAL_MINUTES,
    CHECKPOINT_RETAIN_COUNT,
    CheckpointHealth,
    CheckpointRecord,
    STORE_FILE_NAMES,
)
from signal_kernel.exceptions import (
    CheckpointCorruptionError,
    CheckpointStalenessWarning,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# ─────────────────────────────────────────────────────────────────────────────

log: logging.Logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Primary staleness threshold: 2× the 15-minute crond interval.
# An archive whose mtime is older than this implies crond has silently
# exited — it missed at least two consecutive scheduled write cycles.
# Unit: seconds. Named at module level for external callsite reference.
CHECKPOINT_STALE_THRESHOLD_SECONDS: int = 1800

# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Archive filename pattern produced by mft_checkpoint.sh.
# Format: mft_YYYYMMDD_HHMMSS.tar.gz
# Six capture groups: year, month, day, hour, minute, second.
# Anchored at both ends — no prefix or suffix allowed.
_ARCHIVE_FILENAME_RE: re.Pattern[str] = re.compile(
    r"^mft_(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})\.tar\.gz$"
)

# Sidecar hash file suffix. Written by mft_checkpoint.sh alongside each
# archive as <archive_path>.sha256. Content: 64-char lowercase SHA-256 hex,
# optionally in sha256sum format "<hash>  <filename>", optionally trailed
# by a single newline.
_SIDECAR_SUFFIX: str = ".sha256"

# SHA-256 hex digest pattern: exactly 64 lowercase hexadecimal characters.
_SHA256_RE: re.Pattern[str] = re.compile(r"^[0-9a-f]{64}$")

# Read chunk size for streaming SHA-256 computation. 64 KiB balances
# syscall overhead against memory pressure on multi-hundred-MB archives.
_HASH_READ_CHUNK_BYTES: int = 65_536

# Minimum plausible archive size. A valid tar.gz of four non-empty index
# files (minimum 64 bytes each per mft_checkpoint.sh's MIN_FILE_BYTES)
# cannot produce a compressed archive smaller than this. Anything smaller
# is evidence of an empty file, partial write, or deliberate placeholder.
_MIN_ARCHIVE_SIZE_BYTES: int = 512

# Maximum archive size we will load into memory in restore(). Chosen to be
# safely above any realistic checkpoint (four index files, even at their
# largest, are well under 1 GiB compressed) while preventing restore()
# from being weaponized to exhaust the process address space.
_MAX_ARCHIVE_SIZE_BYTES: int = 1_073_741_824  # 1 GiB

# Maximum size of a .sha256 sidecar file. The content is 64 hex chars +
# optional two-space + filename + newline. Nothing valid exceeds 512 bytes.
# Files larger than this are rejected without reading their content.
_MAX_SIDECAR_SIZE_BYTES: int = 512

# Expected member count inside a valid archive.
# Must equal len(STORE_FILE_NAMES) — verified by module-load assertion.
_EXPECTED_MEMBER_COUNT: int = 4

# Maximum size (bytes) for any single member inside the archive.
# Individual index files should be well under 4 GiB each. A member
# reporting a size above this is either a corrupt header or a rogue archive.
_MAX_MEMBER_SIZE_BYTES: int = 4_294_967_296  # 4 GiB

# Future mtime tolerance. Filesystem clocks can drift slightly between
# the node that writes archives and the node that reads them. We tolerate
# up to 5 minutes of forward skew before logging a clock-skew warning.
# Anything beyond this threshold triggers a WARNING log but does NOT
# cause the archive to be treated as stale (true age is indeterminate).
_FUTURE_MTIME_TOLERANCE_SECONDS: int = 300  # 5 minutes

# Maximum number of archive candidates considered in a single call.
# Bounded to guarantee O(1) worst-case scan time regardless of how many
# spurious files accumulate in the checkpoint directory. Set to retain
# count + a small margin to tolerate one rotation cycle's worth of extra files.
_ARCHIVE_SCAN_LIMIT: int = CHECKPOINT_RETAIN_COUNT + 8

# crond process name patterns searched in /proc/<pid>/comm and cmdline.
# Lowercase; matched against lowercased process names.
_CROND_COMM_NAMES: FrozenSet[str] = frozenset({"crond", "cron", "fcron"})
_CROND_CMDLINE_SUBSTRINGS: Tuple[str, ...] = ("crond", "fcron", "/usr/sbin/cron")

# Filesystem root paths to guard against — operating on these would be
# a severe misconfiguration. Checked against the resolved checkpoint_dir.
_FILESYSTEM_ROOTS: FrozenSet[str] = frozenset({"/", "//"})

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LOAD INVARIANT ASSERTIONS
#
# These assertions catch contract drift between checkpoint_monitor.py and
# contracts.py at import time — not during a live restore attempt when
# the damage is hardest to diagnose.
# ─────────────────────────────────────────────────────────────────────────────

assert len(STORE_FILE_NAMES) == _EXPECTED_MEMBER_COUNT, (
    f"_EXPECTED_MEMBER_COUNT ({_EXPECTED_MEMBER_COUNT}) must equal "
    f"len(contracts.STORE_FILE_NAMES) ({len(STORE_FILE_NAMES)}). "
    f"STORE_FILE_NAMES = {sorted(STORE_FILE_NAMES)}. "
    "Update _EXPECTED_MEMBER_COUNT in checkpoint_monitor.py to match."
)

assert CHECKPOINT_STALE_THRESHOLD_SECONDS == CHECKPOINT_INTERVAL_MINUTES * 60 * 2, (
    f"CHECKPOINT_STALE_THRESHOLD_SECONDS ({CHECKPOINT_STALE_THRESHOLD_SECONDS}) "
    f"must equal 2 × CHECKPOINT_INTERVAL_MINUTES ({CHECKPOINT_INTERVAL_MINUTES}) × 60 "
    f"= {CHECKPOINT_INTERVAL_MINUTES * 60 * 2}. "
    "Update one of them — they must stay in sync."
)

assert _MIN_ARCHIVE_SIZE_BYTES < _MAX_ARCHIVE_SIZE_BYTES, (
    "Internal constant error: _MIN_ARCHIVE_SIZE_BYTES must be "
    "strictly less than _MAX_ARCHIVE_SIZE_BYTES."
)

assert _HASH_READ_CHUNK_BYTES > 0 and (_HASH_READ_CHUNK_BYTES & (_HASH_READ_CHUNK_BYTES - 1)) == 0, (
    "_HASH_READ_CHUNK_BYTES must be a positive power of two."
)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL RESULT TYPES
#
# Named tuples used as structured return values for private functions.
# Using NamedTuple instead of dicts makes field access explicit and allows
# type checkers to validate field names at call sites.
# ─────────────────────────────────────────────────────────────────────────────

class _ArchiveStat(NamedTuple):
    """
    Aggregated stat information for one candidate archive file.
    Produced by _stat_archive(); consumed by _verify_integrity() and
    _check_staleness(). Carries both mtime and the timestamp parsed from
    the filename for two-key sorting (mtime primary, filename secondary).
    """
    path:               Path
    size_bytes:         int
    mtime_utc:          datetime
    filename_timestamp: Optional[datetime]  # None if filename parse fails


class _IntegrityOutcome(NamedTuple):
    """
    Full result of a six-layer integrity verification pass on one archive.

    passed=True   → archive is structurally valid and safe to return from restore()
    passed=False  → archive must be rejected; failure_reason explains why

    sha256_verified and sha256_fallback are mutually exclusive:
      sha256_verified=True  → sidecar was present and matched computed digest
      sha256_fallback=True  → no sidecar; only tar-stream verification was done
      both=False            → verification failed before reaching SHA-256 layer

    computed_sha256 is set whenever the SHA-256 computation succeeded, even
    if the overall outcome is failed (e.g., sidecar mismatch). It is used
    by restore() for the post-read re-verification step.
    """
    passed:               bool
    sha256_verified:      bool
    sha256_fallback:      bool
    tar_structure_valid:  bool
    manifest_complete:    bool
    member_sanity_passed: bool
    failure_reason:       Optional[str]
    computed_sha256:      Optional[str]


class _StalenessOutcome(NamedTuple):
    """
    Result of a staleness check on one archive file.

    is_stale=True         → mtime exceeds CHECKPOINT_STALE_THRESHOLD_SECONDS
    future_skew_detected  → mtime is ahead of current clock beyond tolerance
    age_seconds           → 0.0 when future_skew_detected (true age unknown)
    """
    is_stale:             bool
    age_seconds:          float
    mtime_utc:            datetime
    future_skew_detected: bool


class _TarVerifyResult(NamedTuple):
    """
    Result of _verify_tar_structure(). Returned as a plain tuple to avoid
    creating a new object at every call site — the caller always immediately
    destructures it.
    """
    passed:         bool
    failure_reason: str           # empty string when passed=True
    found_names:    FrozenSet[str]


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: DIRECTORY VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_checkpoint_dir(checkpoint_dir: object) -> Path:
    """
    Validate and resolve checkpoint_dir to an absolute Path.

    Enforces, in order:
      1. Type: checkpoint_dir must be str or Path (not None, int, bytes, etc.)
      2. Conversion: Path(checkpoint_dir).resolve(strict=False) must succeed
      3. Not root: resolved path must not be the filesystem root
      4. Existence: resolved path must exist on disk
      5. Is directory: resolved path must be a directory
      6. Readable: current process must have read + execute permission

    All six checks are explicit and produce distinct error messages so that
    misconfigured deployments can be diagnosed from logs alone.

    Returns the resolved absolute Path on success.

    Raises:
      TypeError:            checkpoint_dir is an unsupported type
      ValueError:           path resolves to filesystem root or is malformed
      FileNotFoundError:    directory does not exist
      NotADirectoryError:   path exists but is not a directory
      PermissionError:      directory is not readable by current process
    """
    if checkpoint_dir is None:
        raise TypeError(
            "checkpoint_dir must not be None. "
            "Pass an explicit str or Path to the checkpoint directory. "
            "None is never a valid checkpoint directory path."
        )
    if isinstance(checkpoint_dir, bytes):
        raise TypeError(
            "checkpoint_dir must be str or Path, not bytes. "
            f"Got: {checkpoint_dir!r}. "
            "Decode the bytes to a str before passing."
        )
    if not isinstance(checkpoint_dir, (str, os.PathLike)):
        raise TypeError(
            f"checkpoint_dir must be str or os.PathLike, got {type(checkpoint_dir).__name__!r}. "
            f"Value: {checkpoint_dir!r}."
        )

    # Resolve to absolute. strict=False so we can give a better FileNotFoundError below.
    try:
        resolved = Path(checkpoint_dir).resolve(strict=False)
    except (TypeError, ValueError, OSError) as exc:
        raise ValueError(
            f"checkpoint_dir {str(checkpoint_dir)!r} cannot be resolved to an "
            f"absolute path: {exc}"
        ) from exc

    # Guard against filesystem root. An operator passing "/" as checkpoint_dir
    # would cause _glob_archives to scan the entire filesystem.
    if str(resolved) in _FILESYSTEM_ROOTS or resolved.parent == resolved:
        raise ValueError(
            f"checkpoint_dir resolves to {str(resolved)!r}, which is the filesystem root. "
            "This is a misconfiguration — checkpoint directories must be subdirectories, "
            "not the root."
        )

    # Existence check before the is_dir() check so the error message is accurate.
    if not resolved.exists():
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            str(resolved),
        )

    # Directory check. is_dir() follows symlinks — a symlink to a directory
    # is acceptable; a symlink to a regular file is not.
    if not resolved.is_dir():
        raise NotADirectoryError(
            errno.ENOTDIR,
            os.strerror(errno.ENOTDIR),
            str(resolved),
        )

    # Permission check using effective UID/GID. os.access(os.R_OK | os.X_OK)
    # tests both read (list directory entries) and execute (traverse into it).
    if not os.access(resolved, os.R_OK | os.X_OK):
        raise PermissionError(
            errno.EACCES,
            os.strerror(errno.EACCES),
            str(resolved),
        )

    return resolved


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: ARCHIVE FILENAME PARSING
# ─────────────────────────────────────────────────────────────────────────────

def _parse_filename_timestamp(filename: str) -> Optional[datetime]:
    """
    Parse the timestamp embedded in an archive filename.

    mft_checkpoint.sh names archives as: mft_YYYYMMDD_HHMMSS.tar.gz
    This timestamp is used as a secondary sort key when two archives have
    identical mtimes (which can occur if mft_checkpoint.sh is invoked twice
    within the same second, which should not happen under normal crond
    scheduling but is defended against).

    Returns a UTC datetime if the filename matches _ARCHIVE_FILENAME_RE
    and the embedded calendar values are valid.
    Returns None (and logs at DEBUG) if the name does not match, or if the
    embedded values form an invalid date (e.g., month=13, day=32).

    Never raises.
    """
    m = _ARCHIVE_FILENAME_RE.match(filename)
    if not m:
        return None

    year, month, day, hour, minute, second = (int(g) for g in m.groups())
    try:
        return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
    except ValueError:
        log.debug(
            "Archive filename %r embeds an invalid calendar value "
            "(year=%d month=%d day=%d hour=%d minute=%d second=%d) — "
            "using mtime as sole sort key for this archive.",
            filename, year, month, day, hour, minute, second,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: ARCHIVE STAT
# ─────────────────────────────────────────────────────────────────────────────

def _stat_archive(path: Path) -> Optional[_ArchiveStat]:
    """
    Stat one archive path and return an _ArchiveStat.

    Returns None if:
      - The file cannot be stat-ed (race: deleted between glob and stat)
      - The path is not a regular file (directory, symlink to dir, device, FIFO)

    Intentionally returns None rather than raising so the caller (_glob_archives)
    can skip individual unreadable archives and continue scanning. The reason
    is logged at WARNING level so the skip is visible in logs.

    Does NOT read or open the file — stat(2) only.
    """
    try:
        st = path.stat()
    except OSError as exc:
        log.warning(
            "Cannot stat archive %r (errno=%d: %s) — "
            "skipping this candidate.",
            str(path), exc.errno, exc.strerror,
        )
        return None

    if not stat.S_ISREG(st.st_mode):
        file_type = _describe_file_type(stat.S_IFMT(st.st_mode))
        log.warning(
            "Archive path %r is not a regular file (type=%s mode=0o%o) — "
            "skipping this candidate.",
            str(path), file_type, stat.S_IFMT(st.st_mode),
        )
        return None

    mtime_utc = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
    filename_timestamp = _parse_filename_timestamp(path.name)

    return _ArchiveStat(
        path=path,
        size_bytes=st.st_size,
        mtime_utc=mtime_utc,
        filename_timestamp=filename_timestamp,
    )


def _describe_file_type(ifmt: int) -> str:
    """
    Return a human-readable string for a stat mode file-type bits value.
    Used in warning messages to make unusual file types diagnosable.
    """
    _TABLE = {
        stat.S_IFDIR:  "directory",
        stat.S_IFLNK:  "symlink",
        stat.S_IFIFO:  "named_pipe",
        stat.S_IFSOCK: "socket",
        stat.S_IFBLK:  "block_device",
        stat.S_IFCHR:  "char_device",
    }
    return _TABLE.get(ifmt, f"unknown(0o{ifmt:o})")


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: ARCHIVE DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────

def _glob_archives(checkpoint_dir: Path) -> List[_ArchiveStat]:
    """
    Discover all valid-named archive files in checkpoint_dir and return them
    sorted newest-first.

    Sort order (three keys, descending):
      Primary:   mtime_utc   — most recently modified archive first
      Secondary: filename_timestamp — timestamp in the filename, for tie-breaking
      Tertiary:  path.name   — deterministic alphabetical tie-break of last resort

    Filtering applied to every entry:
      - Name must start with "mft_" and end with ".tar.gz" (pre-filter)
      - Name must match _ARCHIVE_FILENAME_RE in full (strict filter)
      - Entry must be stat-able by the current process
      - Entry must be a regular file

    Archives that fail filtering are logged and skipped — the caller receives
    the valid subset. An empty list is returned (not raised) when no valid
    archives exist.

    Bounded by _ARCHIVE_SCAN_LIMIT to guarantee O(constant) worst-case
    runtime regardless of directory size.

    Never raises — all OSError from iterdir() are logged and produce an
    empty list.
    """
    try:
        raw_entries: List[os.DirEntry] = []
        with os.scandir(checkpoint_dir) as it:
            for entry in it:
                raw_entries.append(entry)
    except OSError as exc:
        log.warning(
            "Cannot scan checkpoint directory %r (errno=%d: %s). "
            "Returning empty archive list.",
            str(checkpoint_dir), exc.errno, exc.strerror,
        )
        return []

    candidates: List[_ArchiveStat] = []
    scanned = 0
    skipped_pre_filter = 0
    skipped_strict = 0

    _MIN_NAME = len("mft_") + 1
    for entry in raw_entries:
        name: str = entry.name

        # Pre-filter: fast rejection before regex (avoids regex overhead for
        # the common case of non-archive files in the directory).
        if not (name.startswith("mft_") and name.endswith(".tar.gz") and len(name) > _MIN_NAME):
            skipped_pre_filter += 1
            continue

        # Strict regex filter: the full filename must match exactly.
        # This rejects temp files like "mft_20240101_120000.tar.gz.tmp"
        # and partial names left by interrupted writes.
        if not _ARCHIVE_FILENAME_RE.match(name):
            log.debug(
                "Skipping %r — pre-filter passed but full regex did not match. "
                "This is likely a .tmp or in-progress write.",
                name,
            )
            skipped_strict += 1
            continue

        scanned += 1
        archive_stat = _stat_archive(Path(entry.path))
        if archive_stat is None:
            continue  # Warning already emitted by _stat_archive.

        candidates.append(archive_stat)

        if len(candidates) >= _ARCHIVE_SCAN_LIMIT:
            log.warning(
                "Archive scan limit reached (%d candidates) in %r. "
                "The directory contains an unusually large number of archives — "
                "verify that checkpoint rotation (CHECKPOINT_RETAIN_COUNT=%d) "
                "is running correctly.",
                _ARCHIVE_SCAN_LIMIT, str(checkpoint_dir), CHECKPOINT_RETAIN_COUNT,
            )
            break

    log.debug(
        "Archive scan: dir=%r total_entries=%d scanned=%d candidates=%d "
        "skipped_pre=%d skipped_strict=%d",
        str(checkpoint_dir),
        len(raw_entries),
        scanned,
        len(candidates),
        skipped_pre_filter,
        skipped_strict,
    )

    if not candidates:
        return []

    # Sort newest-first using three keys.
    _EPOCH_UTC = datetime.fromtimestamp(0, tz=timezone.utc)

    candidates.sort(
        key=lambda a: (
            a.mtime_utc,
            a.filename_timestamp if a.filename_timestamp is not None else _EPOCH_UTC,
            a.path.name,
        ),
        reverse=True,
    )

    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: SHA-256 COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_file_sha256(path: Path, expected_size: int) -> str:
    """
    Compute the SHA-256 hex digest of the file at path, reading in
    _HASH_READ_CHUNK_BYTES chunks (streaming — never loads the whole file).

    expected_size is the size observed by the caller's prior stat call.
    If total bytes read do not equal expected_size, the file changed during
    the read (concurrent write or deletion). A ValueError is raised to
    signal this to the caller so it can reject this archive.

    Returns the 64-character lowercase hex digest.

    Raises:
      OSError:    if the file cannot be opened or read
      ValueError: if bytes_read != expected_size (concurrent modification)
    """
    h = hashlib.sha256()
    bytes_read = 0

    try:
        with open(path, "rb") as fh:
            while True:
                chunk = fh.read(_HASH_READ_CHUNK_BYTES)
                if not chunk:
                    break
                h.update(chunk)
                bytes_read += len(chunk)
    except OSError as exc:
        raise OSError(
            exc.errno,
            f"SHA-256 read of {path.name!r} failed at byte {bytes_read:,}: "
            f"{exc.strerror}",
            str(path),
        ) from exc

    if bytes_read != expected_size:
        raise ValueError(
            f"SHA-256 of {path.name!r}: expected to read {expected_size:,} bytes "
            f"but read {bytes_read:,}. File was modified during hashing."
        )

    digest = h.hexdigest()
    log.debug(
        "SHA-256 %r: bytes=%d digest=%s...",
        path.name, bytes_read, digest[:16],
    )
    return digest


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: SIDECAR HASH
# ─────────────────────────────────────────────────────────────────────────────

def _read_sidecar_hash(archive_path: Path) -> Optional[str]:
    """
    Read the SHA-256 hex digest from the .sha256 sidecar file adjacent to
    the archive, if one exists.

    Sidecar path: <archive_path>.sha256
      e.g. /store/checkpoints/mft_20240101_120000.tar.gz.sha256

    Accepted content formats:
      A) Plain hex:           "a3f7...b21c\n"           (64 hex chars + optional newline)
      B) sha256sum format:    "a3f7...b21c  filename\n"  (hash, two spaces, basename)

    For format B, the embedded filename is compared against archive_path.name.
    A mismatch is treated as a sidecar/archive pairing error and the sidecar
    is ignored (returns None with a WARNING log).

    Returns the 64-character hex digest if the sidecar exists and is valid.
    Returns None in all other cases — absence of a sidecar is NOT a failure;
    the caller falls back to tar-stream-only verification.

    Security posture: the sidecar file is a trust anchor only if it exists
    and passes all validation. A missing sidecar degrades verification
    quality (no SHA-256 pre-check) but does not block the restore — layers
    4-6 still apply. A present-but-invalid sidecar is logged at WARNING and
    ignored (not treated as a corrupt archive by itself).

    Never raises.
    """
    sidecar_path = Path(str(archive_path) + _SIDECAR_SUFFIX)

    # --- Existence ---
    if not sidecar_path.exists():
        log.debug(
            "No sidecar file for %r (checked %r) — "
            "falling back to tar-stream-only integrity check.",
            archive_path.name, sidecar_path.name,
        )
        return None

    # --- Type check ---
    try:
        sidecar_stat = sidecar_path.stat()
    except OSError as exc:
        log.warning(
            "Cannot stat sidecar %r (errno=%d: %s) — ignoring sidecar.",
            sidecar_path.name, exc.errno, exc.strerror,
        )
        return None

    if not stat.S_ISREG(sidecar_stat.st_mode):
        log.warning(
            "Sidecar %r is not a regular file (type=%s) — ignoring.",
            sidecar_path.name,
            _describe_file_type(stat.S_IFMT(sidecar_stat.st_mode)),
        )
        return None

    # --- Size bounds ---
    if sidecar_stat.st_size == 0:
        log.warning(
            "Sidecar %r is empty (zero bytes) — ignoring.",
            sidecar_path.name,
        )
        return None

    if sidecar_stat.st_size > _MAX_SIDECAR_SIZE_BYTES:
        log.warning(
            "Sidecar %r is %d bytes, exceeding the %d-byte limit. "
            "A valid .sha256 sidecar cannot be this large — ignoring.",
            sidecar_path.name, sidecar_stat.st_size, _MAX_SIDECAR_SIZE_BYTES,
        )
        return None

    # --- Read ---
    try:
        raw_content = sidecar_path.read_bytes()
    except OSError as exc:
        log.warning(
            "Cannot read sidecar %r (errno=%d: %s) — ignoring.",
            sidecar_path.name, exc.errno, exc.strerror,
        )
        return None

    # --- Decode as ASCII (SHA-256 hex is always ASCII) ---
    try:
        text = raw_content.decode("ascii").strip()
    except (UnicodeDecodeError, ValueError) as exc:
        log.warning(
            "Sidecar %r contains non-ASCII bytes (%s) — "
            "not a valid hex digest file. Ignoring.",
            sidecar_path.name, exc,
        )
        return None

    # --- Parse formats A and B ---
    if not text:
        log.warning(
            "Sidecar %r is whitespace-only after decode — ignoring.",
            sidecar_path.name,
        )
        return None

    candidate_hash: str
    if "  " in text:
        # sha256sum format: "<hash>  <filename>"
        parts = text.split("  ", 1)
        candidate_hash = parts[0].strip()
        embedded_filename = parts[1].strip() if len(parts) > 1 else ""
        if embedded_filename and embedded_filename != archive_path.name:
            log.warning(
                "Sidecar %r embeds filename %r but the archive is %r — "
                "sidecar/archive filename mismatch. "
                "This sidecar may belong to a different archive. Ignoring.",
                sidecar_path.name, embedded_filename, archive_path.name,
            )
            return None
    elif " " in text and len(text.split()) == 2:
        # Single-space variant (some tools use one space).
        parts = text.split(None, 1)
        candidate_hash = parts[0]
        embedded_filename = parts[1] if len(parts) > 1 else ""
        if embedded_filename and embedded_filename != archive_path.name:
            log.warning(
                "Sidecar %r (single-space format) embeds filename %r but archive is %r — "
                "ignoring.",
                sidecar_path.name, embedded_filename, archive_path.name,
            )
            return None
    else:
        # Plain hash only.
        candidate_hash = text

    # --- Validate format ---
    if not _SHA256_RE.match(candidate_hash):
        # Provide a safe preview: at most 80 chars, no control characters.
        safe_preview = "".join(
            c if c.isprintable() else "?" for c in candidate_hash[:80]
        )
        log.warning(
            "Sidecar %r contains %r which is not a valid 64-char lowercase "
            "hex SHA-256 digest — ignoring.",
            sidecar_path.name, safe_preview,
        )
        return None

    log.debug(
        "Sidecar hash for %r: %s...",
        archive_path.name, candidate_hash[:16],
    )
    return candidate_hash


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: TAR STRUCTURE VERIFICATION (Layers 4, 5, 6)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_tar_structure(archive_path: Path) -> _TarVerifyResult:
    """
    Open the archive in streaming mode and verify layers 4, 5, and 6.

    Layer 4 — gzip stream integrity:
      tarfile.open(mode="r:gz") opens the stream without decompressing it
      entirely. If the gzip stream is corrupt or truncated, tarfile raises
      one of: TruncatedHeaderError, ReadError, CompressionError, HeaderError.
      All are caught and mapped to failure outcomes.

    Layer 5 — manifest completeness:
      Iterates all member headers (no data extraction). Collects basenames.
      After iteration, verifies:
        - All four STORE_FILE_NAMES are present
        - No extra files outside STORE_FILE_NAMES
        - No duplicate member basenames

    Layer 6 — member safety:
      For each member:
        - Must be a regular file (tarfile.REGTYPE or compatible)
        - Name must not be absolute (no leading /)
        - Normalized name must not begin with ".." (path traversal)
        - Recorded size must be ≥ 0 and ≤ _MAX_MEMBER_SIZE_BYTES
        - Basename after normalization must be non-empty

    SECURITY NOTE:
      No member data is read or extracted. We only iterate member *headers*.
      This means this function consumes minimal memory regardless of member
      sizes. It is safe to call on archives of any size up to
      _MAX_ARCHIVE_SIZE_BYTES.

    Returns _TarVerifyResult(passed, failure_reason, found_names).
    Never raises — all tarfile exceptions are caught and translated.
    """
    found_basenames: List[str] = []

    try:
        # mode="r:gz" — streaming, no random access, no full decompression upfront.
        # This means we discover gzip corruption as we iterate, not all at once.
        with tarfile.open(str(archive_path), mode="r:gz") as tf:
            for member in tf:
                member_name: str = member.name

                # ── Layer 6a: Absolute path guard ────────────────────────────
                if os.path.isabs(member_name):
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive member {member_name!r} has an absolute path. "
                            "Archives produced by mft_checkpoint.sh should contain "
                            "only relative paths. This archive is suspect."
                        ),
                        found_names=frozenset(),
                    )

                # ── Layer 6b: Path traversal guard ───────────────────────────
                # os.path.normpath resolves .. and . components.
                normalized = os.path.normpath(member_name)
                if normalized.startswith("..") or normalized.startswith("/"):
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive member {member_name!r} normalizes to "
                            f"{normalized!r} which escapes the archive root — "
                            "path traversal rejected."
                        ),
                        found_names=frozenset(),
                    )

                # ── Layer 6c: Regular file only ──────────────────────────────
                # tarfile.REGTYPE = b'0'; older tar also uses b'\0' (AREGTYPE).
                # member.isfile() checks both.
                if not member.isfile():
                    type_name = _describe_tar_member_type(member)
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive member {member_name!r} is not a regular file "
                            f"(type={type_name!r}). "
                            "Only regular files are expected in checkpoint archives."
                        ),
                        found_names=frozenset(),
                    )

                # ── Layer 6d: Size bounds ─────────────────────────────────────
                if member.size < 0:
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive member {member_name!r} has a negative recorded "
                            f"size ({member.size}). The tar header is internally "
                            "inconsistent."
                        ),
                        found_names=frozenset(),
                    )

                if member.size > _MAX_MEMBER_SIZE_BYTES:
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive member {member_name!r} reports size "
                            f"{member.size:,} bytes, exceeding the "
                            f"{_MAX_MEMBER_SIZE_BYTES:,}-byte per-member limit. "
                            "This is implausible for an index file — "
                            "archive is suspect."
                        ),
                        found_names=frozenset(),
                    )

                # ── Layer 6e: Non-empty basename ─────────────────────────────
                basename = os.path.basename(normalized)
                if not basename:
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive member {member_name!r} has an empty basename "
                            f"after normalization (normalized={normalized!r}). "
                            "Cannot match against STORE_FILE_NAMES."
                        ),
                        found_names=frozenset(),
                    )

                found_basenames.append(basename)

                # ── Early exit: too many members ─────────────────────────────
                # We check > _EXPECTED_MEMBER_COUNT + 1 to allow reporting the
                # extra member name rather than a generic "too many" message.
                if len(found_basenames) > _EXPECTED_MEMBER_COUNT + 1:
                    return _TarVerifyResult(
                        passed=False,
                        failure_reason=(
                            f"Archive contains more than {_EXPECTED_MEMBER_COUNT} "
                            f"members (found at least {len(found_basenames)}). "
                            "Only STORE_FILE_NAMES should be present. "
                            f"First extra member: {basename!r}."
                        ),
                        found_names=frozenset(found_basenames),
                    )

    # ── Layer 4: gzip / tar stream integrity ─────────────────────────────────
    except tarfile.ReadError as exc:
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Archive {archive_path.name!r} cannot be read as a tar stream: {exc}. "
                "Possible gzip corruption, truncation, or non-tar content."
            ),
            found_names=frozenset(),
        )
    except tarfile.CompressionError as exc:
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Archive {archive_path.name!r} has a gzip decompression error: {exc}. "
                "The gzip stream is corrupt."
            ),
            found_names=frozenset(),
        )
    except tarfile.HeaderError as exc:
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Archive {archive_path.name!r} has an invalid tar header: {exc}."
            ),
            found_names=frozenset(),
        )
    except OSError as exc:
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"I/O error reading {archive_path.name!r} "
                f"(errno={exc.errno}: {exc.strerror})."
            ),
            found_names=frozenset(),
        )
    except Exception as exc:  # noqa: BLE001
        # Broad catch: severely malformed archives can produce unexpected
        # exceptions from tarfile internals. Log the type for diagnostics.
        log.warning(
            "Unexpected %s reading archive %r: %s",
            type(exc).__name__, archive_path.name, exc,
        )
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Unexpected {type(exc).__name__} reading "
                f"{archive_path.name!r}: {exc}."
            ),
            found_names=frozenset(),
        )

    # ── Layer 5: Manifest completeness ───────────────────────────────────────
    found_set = frozenset(found_basenames)
    missing = STORE_FILE_NAMES - found_set
    extra = found_set - STORE_FILE_NAMES

    if missing:
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Archive {archive_path.name!r} is missing required store files: "
                f"{sorted(missing)}. "
                f"Files present: {sorted(found_set)}."
            ),
            found_names=found_set,
        )

    if extra:
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Archive {archive_path.name!r} contains unexpected files: "
                f"{sorted(extra)}. "
                "Only STORE_FILE_NAMES are permitted in checkpoint archives."
            ),
            found_names=found_set,
        )

    # Duplicate detection: found_basenames list length vs set size.
    if len(found_basenames) != len(found_set):
        duplicates = [
            name for name in found_set
            if found_basenames.count(name) > 1
        ]
        return _TarVerifyResult(
            passed=False,
            failure_reason=(
                f"Archive {archive_path.name!r} contains duplicate member names: "
                f"{sorted(duplicates)}. "
                f"Total members={len(found_basenames)} unique names={len(found_set)}."
            ),
            found_names=found_set,
        )

    return _TarVerifyResult(passed=True, failure_reason="", found_names=found_set)


def _describe_tar_member_type(member: tarfile.TarInfo) -> str:
    """Return a human-readable string for a tarfile member's type byte."""
    _TABLE: Dict[bytes, str] = {
        tarfile.REGTYPE:   "regular_file",
        tarfile.AREGTYPE:  "regular_file_alt",
        tarfile.LNKTYPE:   "hard_link",
        tarfile.SYMTYPE:   "symbolic_link",
        tarfile.CHRTYPE:   "char_device",
        tarfile.BLKTYPE:   "block_device",
        tarfile.DIRTYPE:   "directory",
        tarfile.FIFOTYPE:  "fifo",
        tarfile.CONTTYPE:  "contiguous_file",
    }
    return _TABLE.get(member.type, f"unknown(0x{member.type.hex() if isinstance(member.type, bytes) else member.type!r})") # noqa | defensive runtime check


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: FULL INTEGRITY VERIFICATION (orchestrator)
# ─────────────────────────────────────────────────────────────────────────────

def _verify_integrity(archive_stat: _ArchiveStat) -> _IntegrityOutcome:
    """
    Run all six integrity layers on one archive and return an _IntegrityOutcome.

    Layers:
      1 — File existence and type  (already asserted by _stat_archive, re-checked here
                                    defensively to catch race conditions)
      2 — Size sanity bounds
      3 — SHA-256 vs sidecar (or fallback if no sidecar)
      4 — gzip stream validity (via _verify_tar_structure → tarfile.open)
      5 — Manifest completeness
      6 — Member safety

    Never raises. All failures are encoded in _IntegrityOutcome.
    Failure at any layer short-circuits subsequent layers (except that
    the SHA-256 computation result is always captured for use in restore()'s
    post-read re-verification, even when the overall outcome is failed).

    CONCURRENT-WRITE DETECTION:
      After computing the SHA-256, the file is re-stated. If the size changed
      since the original stat captured in archive_stat, the file was written
      concurrently. The outcome is failed with a descriptive reason.
      This is defense-in-depth — mft_checkpoint.sh's atomic rename should
      prevent this, but we defend regardless.
    """
    path = archive_stat.path
    pre_stat_size = archive_stat.size_bytes

    # ── Layer 1 (re-check): existence and regular-file ───────────────────────
    # The original stat is in archive_stat. We trust it enough to proceed, but
    # if the file was deleted in the window between _stat_archive and now, the
    # subsequent operations will fail with informative OSErrors.

    # ── Layer 2: Size bounds ─────────────────────────────────────────────────
    if pre_stat_size < _MIN_ARCHIVE_SIZE_BYTES:
        return _IntegrityOutcome(
            passed=False,
            sha256_verified=False,
            sha256_fallback=False,
            tar_structure_valid=False,
            manifest_complete=False,
            member_sanity_passed=False,
            failure_reason=(
                f"Archive {path.name!r} is only {pre_stat_size} bytes — "
                f"below the minimum plausible size of {_MIN_ARCHIVE_SIZE_BYTES} bytes. "
                "Likely an empty or partially-written file."
            ),
            computed_sha256=None,
        )

    if pre_stat_size > _MAX_ARCHIVE_SIZE_BYTES:
        return _IntegrityOutcome(
            passed=False,
            sha256_verified=False,
            sha256_fallback=False,
            tar_structure_valid=False,
            manifest_complete=False,
            member_sanity_passed=False,
            failure_reason=(
                f"Archive {path.name!r} is {pre_stat_size:,} bytes — "
                f"exceeds the {_MAX_ARCHIVE_SIZE_BYTES:,}-byte safety cap. "
                "This is not a plausible checkpoint archive."
            ),
            computed_sha256=None,
        )

    # ── Layer 3: SHA-256 computation ─────────────────────────────────────────
    sidecar_hash = _read_sidecar_hash(path)
    computed_sha256: Optional[str] = None # noqa
    sha256_verified = False
    sha256_fallback = False

    try:
        computed_sha256 = _compute_file_sha256(path, expected_size=pre_stat_size)
    except OSError as exc:
        return _IntegrityOutcome(
            passed=False,
            sha256_verified=False,
            sha256_fallback=False,
            tar_structure_valid=False,
            manifest_complete=False,
            member_sanity_passed=False,
            failure_reason=(
                f"Cannot read {path.name!r} for SHA-256 computation: {exc.strerror}. "
                "File may have been deleted or become inaccessible."
            ),
            computed_sha256=None,
        )
    except ValueError as exc:
        # File changed size during read — concurrent write detected.
        return _IntegrityOutcome(
            passed=False,
            sha256_verified=False,
            sha256_fallback=False,
            tar_structure_valid=False,
            manifest_complete=False,
            member_sanity_passed=False,
            failure_reason=str(exc),
            computed_sha256=None,
        )

    # ── Concurrent-write re-stat ─────────────────────────────────────────────
    try:
        current_stat = path.stat()
        if current_stat.st_size != pre_stat_size:
            return _IntegrityOutcome(
                passed=False,
                sha256_verified=False,
                sha256_fallback=False,
                tar_structure_valid=False,
                manifest_complete=False,
                member_sanity_passed=False,
                failure_reason=(
                    f"Archive {path.name!r} size changed between stat and SHA-256 "
                    f"computation: {pre_stat_size} → {current_stat.st_size} bytes. "
                    "Concurrent write detected — this archive is unsafe to use."
                ),
                computed_sha256=computed_sha256,
            )
    except OSError:
        # Re-stat failed (file deleted?). Proceed optimistically — the SHA-256
        # computation already completed successfully, which means the file was
        # readable. The tar-structure check will confirm or reject.
        pass

    # ── Layer 3b: Sidecar comparison ─────────────────────────────────────────
    if sidecar_hash is not None:
        if computed_sha256 == sidecar_hash:
            sha256_verified = True
            log.debug(
                "SHA-256 verified against sidecar for %r: %s...",
                path.name, computed_sha256[:16],
            )
        else:
            return _IntegrityOutcome(
                passed=False,
                sha256_verified=False,
                sha256_fallback=False,
                tar_structure_valid=False,
                manifest_complete=False,
                member_sanity_passed=False,
                failure_reason=(
                    f"SHA-256 MISMATCH for {path.name!r}: "
                    f"sidecar={sidecar_hash[:16]}... "
                    f"computed={computed_sha256[:16]}... "
                    "Archive content does not match its recorded hash. "
                    "Possible partial write, filesystem corruption, or tampering."
                ),
                computed_sha256=computed_sha256,
            )
    else:
        # No sidecar — layers 4-6 are the only verification.
        sha256_fallback = True

    # ── Layers 4, 5, 6: Tar structure, manifest, member safety ───────────────
    tar_result = _verify_tar_structure(path)

    if not tar_result.passed:
        return _IntegrityOutcome(
            passed=False,
            sha256_verified=sha256_verified,
            sha256_fallback=sha256_fallback,
            tar_structure_valid=False,
            manifest_complete=False,
            member_sanity_passed=False,
            failure_reason=tar_result.failure_reason,
            computed_sha256=computed_sha256,
        )

    # ── All six layers passed ─────────────────────────────────────────────────
    log.debug(
        "Integrity OK for %r: "
        "sha256_verified=%s sha256_fallback=%s members=%s",
        path.name,
        sha256_verified,
        sha256_fallback,
        sorted(tar_result.found_names),
    )
    return _IntegrityOutcome(
        passed=True,
        sha256_verified=sha256_verified,
        sha256_fallback=sha256_fallback,
        tar_structure_valid=True,
        manifest_complete=True,
        member_sanity_passed=True,
        failure_reason=None,
        computed_sha256=computed_sha256,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: STALENESS CHECK
# ─────────────────────────────────────────────────────────────────────────────

def _check_staleness(archive_stat: _ArchiveStat) -> _StalenessOutcome:
    """
    Determine whether the most recent archive is stale and compute its age.

    Staleness: mtime is more than CHECKPOINT_STALE_THRESHOLD_SECONDS in the past.
    This implies crond missed ≥ 2 consecutive scheduled write cycles.

    Clock skew detection: if mtime is more than _FUTURE_MTIME_TOLERANCE_SECONDS
    in the future, a WARNING is logged and the archive is treated as non-stale
    (true age is unknown when mtime is future). The CheckpointStalenessWarning
    is not issued for future-skew cases.

    Returns _StalenessOutcome. Never raises.
    """
    now_utc = datetime.now(timezone.utc)
    mtime_utc = archive_stat.mtime_utc
    delta_seconds = (now_utc - mtime_utc).total_seconds()

    # Forward clock skew: mtime significantly in the future.
    if delta_seconds < -_FUTURE_MTIME_TOLERANCE_SECONDS:
        skew_seconds = abs(delta_seconds)
        log.warning(
            "Archive %r has mtime %s which is %.1f seconds ahead of "
            "current time %s. Filesystem clock skew detected "
            "(tolerance=%ds). Treating as non-stale; true age is unknown.",
            archive_stat.path.name,
            mtime_utc.isoformat(),
            skew_seconds,
            now_utc.isoformat(),
            _FUTURE_MTIME_TOLERANCE_SECONDS,
        )
        return _StalenessOutcome(
            is_stale=False,
            age_seconds=0.0,
            mtime_utc=mtime_utc,
            future_skew_detected=True,
        )

    # Small forward skew within tolerance: treat as age=0 (just written).
    if delta_seconds < 0:
        log.debug(
            "Archive %r mtime is %.1f seconds ahead of current clock "
            "(within tolerance=%ds). Treating as age=0.",
            archive_stat.path.name,
            abs(delta_seconds),
            _FUTURE_MTIME_TOLERANCE_SECONDS,
        )
        return _StalenessOutcome(
            is_stale=False,
            age_seconds=0.0,
            mtime_utc=mtime_utc,
            future_skew_detected=False,
        )

    is_stale = delta_seconds > CHECKPOINT_STALE_THRESHOLD_SECONDS

    if is_stale:
        log.warning(
            "Checkpoint archive %r is STALE: age=%.1f s "
            "(threshold=%d s = %.1f min). crond may have silently exited.",
            archive_stat.path.name,
            delta_seconds,
            CHECKPOINT_STALE_THRESHOLD_SECONDS,
            CHECKPOINT_STALE_THRESHOLD_SECONDS / 60.0,
        )
    else:
        log.debug(
            "Checkpoint archive %r is fresh: age=%.1f s (threshold=%d s).",
            archive_stat.path.name,
            delta_seconds,
            CHECKPOINT_STALE_THRESHOLD_SECONDS,
        )

    return _StalenessOutcome(
        is_stale=is_stale,
        age_seconds=delta_seconds,
        mtime_utc=mtime_utc,
        future_skew_detected=False,
    )


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: CROND LIVENESS DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def _detect_crond_alive(staleness: Optional[_StalenessOutcome]) -> bool:
    """
    Determine whether crond (Process 2) is alive, using two independent signals.

    SIGNAL A — Mtime freshness (from staleness argument):
      If the most recent archive mtime is within CHECKPOINT_STALE_THRESHOLD_SECONDS,
      crond recently completed a write cycle. If stale (or no archives exist),
      crond may have exited.

      Special cases:
        staleness is None           → no archives at all → treat as dead
        future_skew_detected=True   → true age unknown → treat as fresh

    SIGNAL B — /proc process scan:
      Walk /proc/<pid>/comm and /proc/<pid>/cmdline for each numeric PID.
      Search for process names in _CROND_COMM_NAMES and substrings in
      _CROND_CMDLINE_SUBSTRINGS.
      Returns True (found), False (not found), or None (/proc unavailable).

    DECISION TABLE:
      Signal A  Signal B  Result
      fresh     found     crond_alive=True
      fresh     not found crond_alive=False   (process exited but wrote recently)
      fresh     None      crond_alive=True    (/proc unavailable; trust mtime)
      stale     found     crond_alive=False   (process exists but not writing)
      stale     not found crond_alive=False
      stale     None      crond_alive=False   (stale dominates)
      None      *         crond_alive=False   (no archives → never ran or dead)

    Conservative: any unresolvable doubt → crond_alive=False.
    """
    # Signal A
    if staleness is None:
        mtime_fresh = False
        log.debug("crond liveness Signal A: no staleness data → mtime_fresh=False")
    elif staleness.future_skew_detected:
        mtime_fresh = True
        log.debug("crond liveness Signal A: future skew detected → mtime_fresh=True (optimistic)")
    else:
        mtime_fresh = not staleness.is_stale
        log.debug(
            "crond liveness Signal A: age=%.1fs stale=%s → mtime_fresh=%s",
            staleness.age_seconds, staleness.is_stale, mtime_fresh,
        )

    # Signal B
    proc_found: Optional[bool] = _probe_crond_in_proc()

    # Decision table
    if not mtime_fresh:
        crond_alive = False
        log.debug(
            "crond liveness: mtime_fresh=False → crond_alive=False "
            "(stale checkpoint overrides proc_found=%s)",
            proc_found,
        )
    elif proc_found is None:
        # /proc not available — trust mtime alone
        crond_alive = True
        log.debug(
            "crond liveness: mtime_fresh=True proc_signal=unavailable "
            "→ crond_alive=True (trusting mtime, /proc unavailable)"
        )
    elif proc_found:
        crond_alive = True
        log.debug("crond liveness: mtime_fresh=True proc_found=True → crond_alive=True")
    else:
        crond_alive = False
        log.debug(
            "crond liveness: mtime_fresh=True proc_found=False → crond_alive=False "
            "(mtime fresh but no crond process found — may have just exited)"
        )

    return crond_alive


def _probe_crond_in_proc() -> Optional[bool]:
    """
    Scan /proc for a running crond-like process.

    Returns:
      True   — found a process whose name or cmdline matches a crond pattern
      False  — /proc is accessible but no matching process found
      None   — /proc is not available, not readable, or scan failed

    Reads /proc/<pid>/comm (process name, limited to 15 chars by Linux)
    and /proc/<pid>/cmdline (full command line, null-byte separated).

    Race conditions: processes can exit between readdir and file read.
    These races produce OSError (ENOENT, ESRCH) on individual /proc entries,
    which are silently skipped per-entry. The scan continues.

    Permission errors: /proc/<pid>/ directories owned by other users produce
    PermissionError on comm or cmdline reads. These are silently skipped —
    they are expected and do not indicate a problem.

    Numeric PID filter: only /proc/<numeric-pid>/ directories are examined.
    Non-numeric entries (e.g., /proc/self, /proc/cpuinfo) are skipped.

    Scan is bounded by the number of /proc entries. On typical Alpine systems
    with a small container process count, this completes in < 1 ms.
    """
    proc_root = Path("/proc")

    # Quick availability check before iterating.
    if not proc_root.is_dir():
        log.debug("/proc is not a directory — process table scan unavailable.")
        return None

    try:
        proc_entries = list(proc_root.iterdir())
    except OSError as exc:
        log.debug(
            "Cannot list /proc (errno=%d: %s) — process table scan skipped.",
            exc.errno, exc.strerror,
        )
        return None

    for entry in proc_entries:
        pid_str = entry.name
        if not pid_str.isdigit():
            continue

        # ── Check /proc/<pid>/comm ────────────────────────────────────────────
        comm_path = entry / "comm"
        try:
            comm = comm_path.read_text(encoding="ascii", errors="replace").strip().lower()
            if comm in _CROND_COMM_NAMES:
                log.debug(
                    "Found crond-like process in /proc/%s/comm: %r",
                    pid_str, comm,
                )
                return True
        except OSError:
            pass  # Process exited or permission denied — expected.

        # ── Check /proc/<pid>/cmdline ─────────────────────────────────────────
        cmdline_path = entry / "cmdline"
        try:
            raw = cmdline_path.read_bytes()
            if not raw:
                continue
            # Null bytes separate argv elements — replace with spaces.
            cmdline = raw.replace(b"\x00", b" ").decode("ascii", errors="replace").lower().strip()
            for pattern in _CROND_CMDLINE_SUBSTRINGS:
                if pattern in cmdline:
                    log.debug(
                        "Found crond-like process in /proc/%s/cmdline: %r",
                        pid_str, cmdline[:100],
                    )
                    return True
        except OSError:
            pass  # Same as above.

    log.debug("No crond-like process found in /proc scan.")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: CheckpointRecord BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_checkpoint_record(
    archive_stat: _ArchiveStat,
    integrity: _IntegrityOutcome,
) -> Optional[CheckpointRecord]:
    """
    Construct a contracts.CheckpointRecord from an _ArchiveStat and
    _IntegrityOutcome. Used internally to produce structured checkpoint
    metadata for callers that want to inspect individual archive records.

    files_archived:
      - If integrity.passed: STORE_FILE_NAMES (all four files verified present)
      - If not passed: frozenset() (archive is corrupt; no files can be trusted)

    Returns None if CheckpointRecord construction fails (e.g., contracts.py
    validation rejects the inputs). The failure is logged at WARNING.
    Callers must handle None gracefully.
    """
    try:
        return CheckpointRecord(
            archive_path=str(archive_stat.path),
            timestamp=archive_stat.mtime_utc,
            archive_size_bytes=archive_stat.size_bytes,
            integrity_verified=integrity.passed,
            files_archived=frozenset(STORE_FILE_NAMES) if integrity.passed else frozenset(),
        )
    except (ValueError, TypeError) as exc:
        log.warning(
            "Cannot construct CheckpointRecord for %r: %s. "
            "The record was rejected by contracts.py validation.",
            archive_stat.path.name, exc,
        )
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PRIVATE: DEGRADED HEALTH FALLBACK
# ─────────────────────────────────────────────────────────────────────────────

def _make_degraded_health(
    *,
    assessed_at: datetime,
    restore_invoked_at_startup: bool,
    restore_succeeded: Optional[bool],
) -> CheckpointHealth:
    """
    Construct a minimally safe CheckpointHealth representing a fully degraded
    checkpoint system — used when health() cannot assess the real disk state
    (e.g., checkpoint_dir is inaccessible or CheckpointHealth construction fails
    with valid-looking inputs).

    All fields reflect worst-case: crond dead, no archives, no size, no time.
    CheckpointHealth.is_healthy will be False.

    Applies the restore_invoked_at_startup / restore_succeeded contract from
    contracts.py::CheckpointHealth.__post_init__ before construction, to prevent
    this fallback path from itself raising.

    In the extreme case where even the degraded construction fails (contracts.py
    changed incompatibly), we log CRITICAL and re-raise — the caller must handle
    this as an internal error. This represents a checkpoint_monitor / contracts.py
    version mismatch that must be fixed.
    """
    # Enforce contracts.py::CheckpointHealth contract before calling constructor.
    if restore_invoked_at_startup and restore_succeeded is None:
        restore_succeeded = False
    if not restore_invoked_at_startup and restore_succeeded is not None:
        restore_succeeded = None

    try:
        return CheckpointHealth(
            last_checkpoint_time=None,
            checkpoint_count=0,
            latest_archive_size_bytes=None,
            restore_invoked_at_startup=restore_invoked_at_startup,
            restore_succeeded=restore_succeeded,
            crond_alive=False,
            minutes_since_last_checkpoint=None,
            assessed_at=assessed_at,
        )
    except Exception as exc:  # noqa: BLE001
        log.critical(
            "CRITICAL: Cannot construct even a degraded CheckpointHealth: %s. "
            "contracts.py and checkpoint_monitor.py are out of sync. "
            "This must be fixed before the next deployment.",
            exc,
        )
        raise


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: restore()
# ═════════════════════════════════════════════════════════════════════════════

def restore(checkpoint_dir: object) -> bytes:
    """
    Find the most recent valid checkpoint archive, verify its integrity
    through all six layers, and return its raw bytes.

    ─────────────────────────────────────────────────────────────────────────
    WHAT THIS FUNCTION RETURNS
    ─────────────────────────────────────────────────────────────────────────
    The raw bytes of a verified .tar.gz archive. The caller is responsible
    for extraction (e.g., by passing the bytes to tarfile or by writing them
    to disk and invoking restore.sh). checkpoint_monitor does not extract or
    deserialize — it verifies and returns.

    The returned bytes are guaranteed to:
      - Be non-empty
      - Represent a gzip-compressed tar archive
      - Contain exactly the four STORE_FILE_NAMES and nothing else
      - Match the SHA-256 recorded in the .sha256 sidecar, if one existed
      - Have been re-verified by SHA-256 after the final read from disk

    ─────────────────────────────────────────────────────────────────────────
    ITERATION STRATEGY
    ─────────────────────────────────────────────────────────────────────────
    Archives are tried newest-to-oldest (by mtime, then by filename timestamp,
    then by name — same sort order as _glob_archives). If the latest archive
    fails integrity, the next-newest is tried, and so on until either a
    valid archive is found or all candidates are exhausted.

    This mirrors restore.sh's strategy. Under normal operation the newest
    archive passes all checks on the first try. Iteration handles the rare
    case where mft_checkpoint.sh wrote a corrupt archive (e.g., due to disk
    full or power loss during the write before the atomic rename).

    ─────────────────────────────────────────────────────────────────────────
    CONCURRENT-WRITE DEFENSE
    ─────────────────────────────────────────────────────────────────────────
    Three independent defenses against concurrent writes:

    1. _verify_integrity() re-stats the file after SHA-256 computation and
       rejects the archive if the size changed.

    2. Before read: the file is re-stated immediately before path.read_bytes().
       If the size changed since _verify_integrity ran, the archive is rejected.

    3. After read: the SHA-256 of the returned bytes is computed and compared
       against _verify_integrity's computed_sha256. If they differ, the file
       was replaced between verification and read. The archive is rejected and
       the next candidate is tried.

    ─────────────────────────────────────────────────────────────────────────
    ARGS
    ─────────────────────────────────────────────────────────────────────────
    checkpoint_dir:
      str or Path pointing to the directory containing mft_*.tar.gz archives.
      Must exist and be readable by the current process.

    ─────────────────────────────────────────────────────────────────────────
    RETURNS
    ─────────────────────────────────────────────────────────────────────────
    bytes — the complete, verified .tar.gz archive data. Never empty.

    ─────────────────────────────────────────────────────────────────────────
    RAISES
    ─────────────────────────────────────────────────────────────────────────
    CheckpointCorruptionError
      All available archives failed integrity. No restorable checkpoint exists.
      archives_tried is set to the number of candidates that were inspected.
      failure_reason describes why the last candidate was rejected.

    FileNotFoundError
      checkpoint_dir exists and is readable but contains no valid-named archives.
      The errno is ENOENT. This is distinct from CheckpointCorruptionError —
      it means the checkpoint system has never written an archive, not that
      archives are corrupt.

    TypeError / ValueError
      checkpoint_dir argument is of the wrong type or resolves to an invalid path.

    PermissionError / NotADirectoryError / FileNotFoundError
      checkpoint_dir itself is inaccessible. The errno is set appropriately.
    """
    resolved_dir = _resolve_checkpoint_dir(checkpoint_dir)
    archives = _glob_archives(resolved_dir)

    if not archives:
        raise FileNotFoundError(
            errno.ENOENT,
            (
                f"No valid checkpoint archives found in {str(resolved_dir)!r}. "
                f"Expected files matching the pattern mft_YYYYMMDD_HHMMSS.tar.gz. "
                "Either mft_checkpoint.sh has never completed a write cycle, "
                "or all archives were deleted (check rotation config)."
            ),
            str(resolved_dir),
        )

    log.info(
        "restore(): starting — found %d archive candidate(s) in %r.",
        len(archives), str(resolved_dir),
    )

    last_failure_reason: str = "no archives were inspected"
    archives_tried: int = 0

    for archive_stat in archives:
        archives_tried += 1
        path = archive_stat.path

        log.info(
            "restore(): trying candidate %d/%d: %r "
            "(size=%d bytes mtime=%s)",
            archives_tried,
            len(archives),
            path.name,
            archive_stat.size_bytes,
            archive_stat.mtime_utc.isoformat(),
        )

        # ── Full six-layer integrity check ────────────────────────────────────
        integrity = _verify_integrity(archive_stat)

        if not integrity.passed:
            last_failure_reason = integrity.failure_reason or "unknown integrity failure"
            log.warning(
                "restore(): candidate %r rejected: %s — trying next.",
                path.name, last_failure_reason,
            )
            continue

        # ── Pre-read re-stat (defense 2) ──────────────────────────────────────
        try:
            pre_read_stat = path.stat()
        except OSError as exc:
            last_failure_reason = (
                f"Cannot re-stat {path.name!r} before reading "
                f"(errno={exc.errno}: {exc.strerror}). "
                "File may have been deleted between verification and read."
            )
            log.warning("restore(): %s — trying next.", last_failure_reason)
            continue

        if pre_read_stat.st_size != archive_stat.size_bytes:
            last_failure_reason = (
                f"Archive {path.name!r} size changed between verification "
                f"({archive_stat.size_bytes:,} → {pre_read_stat.st_size:,} bytes) "
                "immediately before the read. Concurrent write detected."
            )
            log.warning("restore(): %s — trying next.", last_failure_reason)
            continue

        # Re-check size cap before loading into memory — the pre-stat might have
        # seen a larger file due to a concurrent write that started after our
        # _verify_integrity size check.
        if pre_read_stat.st_size > _MAX_ARCHIVE_SIZE_BYTES:
            last_failure_reason = (
                f"Archive {path.name!r} is now {pre_read_stat.st_size:,} bytes, "
                f"exceeding the {_MAX_ARCHIVE_SIZE_BYTES:,}-byte memory-read cap. "
                "Refusing to load into memory."
            )
            log.warning("restore(): %s — trying next.", last_failure_reason)
            continue

        # ── Read raw bytes ────────────────────────────────────────────────────
        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            last_failure_reason = (
                f"I/O error reading {path.name!r} "
                f"(errno={exc.errno}: {exc.strerror})."
            )
            log.warning("restore(): %s — trying next.", last_failure_reason)
            continue

        if not raw_bytes:
            last_failure_reason = (
                f"Read zero bytes from {path.name!r} despite "
                f"stat reporting {pre_read_stat.st_size:,} bytes. "
                "Possible filesystem inconsistency."
            )
            log.warning("restore(): %s — trying next.", last_failure_reason)
            continue

        if len(raw_bytes) != pre_read_stat.st_size:
            last_failure_reason = (
                f"Read {len(raw_bytes):,} bytes from {path.name!r} but "
                f"pre-read stat reported {pre_read_stat.st_size:,} bytes. "
                "File changed size during read."
            )
            log.warning("restore(): %s — trying next.", last_failure_reason)
            continue

        # ── Post-read SHA-256 re-verification (defense 3) ────────────────────
        if integrity.computed_sha256 is not None:
            post_read_digest = hashlib.sha256(raw_bytes).hexdigest()
            if post_read_digest != integrity.computed_sha256:
                last_failure_reason = (
                    f"Post-read SHA-256 mismatch for {path.name!r}: "
                    f"verification computed {integrity.computed_sha256[:16]}... "
                    f"read bytes compute {post_read_digest[:16]}... "
                    "Archive was replaced between integrity verification and read."
                )
                log.warning("restore(): %s — trying next.", last_failure_reason)
                continue

        # ── All checks passed — return the bytes ──────────────────────────────
        log.info(
            "restore(): SUCCESS — archive=%r size=%d bytes "
            "sha256=%s... sha256_from_sidecar=%s sha256_fallback=%s "
            "candidates_tried=%d/%d",
            path.name,
            len(raw_bytes),
            integrity.computed_sha256[:16] if integrity.computed_sha256 else "n/a",
            integrity.sha256_verified,
            integrity.sha256_fallback,
            archives_tried,
            len(archives),
        )
        return raw_bytes

    # ── All candidates exhausted ──────────────────────────────────────────────
    log.error(
        "restore(): FAILED — exhausted all %d archive candidate(s) in %r. "
        "Last failure: %s",
        archives_tried, str(resolved_dir), last_failure_reason,
    )
    raise CheckpointCorruptionError(
        archive_path=str(resolved_dir),
        archives_tried=archives_tried,
        failure_reason=last_failure_reason,
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC: health()
# ═════════════════════════════════════════════════════════════════════════════

def health(
    checkpoint_dir: object,
    *,
    restore_invoked_at_startup: bool = False,
    restore_succeeded: Optional[bool] = None,
) -> CheckpointHealth:
    """
    Return a point-in-time health snapshot of the checkpoint system.

    ─────────────────────────────────────────────────────────────────────────
    DESIGN CONTRACT
    ─────────────────────────────────────────────────────────────────────────
    health() NEVER raises. Any problem encountered is encoded in the returned
    CheckpointHealth object. The caller (pipeline.py, TAG startup sequence,
    Witness integration) decides how to respond. This contract allows health()
    to be called safely from monitoring code paths, error handlers, and startup
    sequences where raising would compound the problem.

    The sole exception to this contract is the extreme case where even the
    degraded fallback CheckpointHealth construction fails — which indicates a
    contracts.py / checkpoint_monitor.py version mismatch that prevents any
    CheckpointHealth from being constructed. In that case, _make_degraded_health
    logs CRITICAL and re-raises, which is the correct behavior (the caller's
    catch clause will surface the mismatch clearly).

    ─────────────────────────────────────────────────────────────────────────
    FIELDS POPULATED
    ─────────────────────────────────────────────────────────────────────────
    last_checkpoint_time:
      mtime (UTC) of the most recently modified valid-named archive in
      checkpoint_dir. Does NOT require the archive to pass integrity check —
      we report when crond last wrote, not when it last wrote a valid archive.
      None if no archives exist.

    checkpoint_count:
      Count of valid-named archives found (after filename pattern filtering,
      before integrity verification). This is the total number of archives
      present, not the number that passed integrity.

    latest_archive_size_bytes:
      Byte size of the most recently modified archive, from stat.
      None if no archives exist.

    restore_invoked_at_startup:
      Passed through from caller — health() does not invoke restore.

    restore_succeeded:
      Passed through from caller — health() does not invoke restore.

    crond_alive:
      True iff BOTH of the following hold:
        - The latest archive's mtime is within CHECKPOINT_STALE_THRESHOLD_SECONDS
        - Either /proc shows a running crond-like process, or /proc is unavailable

    minutes_since_last_checkpoint:
      Age of the latest archive in minutes. 0.0 if future clock skew detected.
      None if no archives exist.

    assessed_at:
      UTC timestamp of this health() call.

    ─────────────────────────────────────────────────────────────────────────
    STALENESS WARNINGS
    ─────────────────────────────────────────────────────────────────────────
    When the latest archive is stale (age > CHECKPOINT_STALE_THRESHOLD_SECONDS),
    health() issues a CheckpointStalenessWarning via warnings.warn() with
    stacklevel=2 (so the warning points at the caller of health(), not at
    this function). The warning is in addition to the log.warning() call
    inside _check_staleness().

    To suppress: warnings.filterwarnings("ignore", category=CheckpointStalenessWarning)
    To catch in tests: warnings.catch_warnings(record=True)

    ─────────────────────────────────────────────────────────────────────────
    RESTORE CONTRACT ENFORCEMENT
    ─────────────────────────────────────────────────────────────────────────
    contracts.py::CheckpointHealth.__post_init__ enforces:
      restore_invoked_at_startup=True  → restore_succeeded must be True or False
      restore_invoked_at_startup=False → restore_succeeded must be None

    health() validates this contract and adjusts restore_succeeded if the
    caller passed an inconsistent combination, logging the adjustment at
    WARNING level. This prevents CheckpointHealth construction from raising
    due to caller mistakes in the health() argument.

    ─────────────────────────────────────────────────────────────────────────
    ARGS
    ─────────────────────────────────────────────────────────────────────────
    checkpoint_dir:
      str or Path pointing to the checkpoint directory. May not exist or may
      be inaccessible — health() handles this gracefully and returns a
      degraded CheckpointHealth rather than raising.

    restore_invoked_at_startup:
      True if the TAG startup sequence called restore.sh before calling health().

    restore_succeeded:
      True or False if restore_invoked_at_startup is True (result of restore.sh).
      None if restore_invoked_at_startup is False.

    ─────────────────────────────────────────────────────────────────────────
    RETURNS
    ─────────────────────────────────────────────────────────────────────────
    CheckpointHealth — always. Never raises (see design contract above).
    """
    assessed_at = datetime.now(timezone.utc)

    # ── Validate and normalize caller-supplied restore state ──────────────────
    # Apply the contracts.py::CheckpointHealth contract proactively so that
    # CheckpointHealth construction cannot raise due to argument inconsistency.
    if restore_invoked_at_startup and restore_succeeded is None:
        log.warning(
            "health() called with restore_invoked_at_startup=True but "
            "restore_succeeded=None. Defaulting to restore_succeeded=False "
            "to satisfy the CheckpointHealth contract."
        )
        restore_succeeded = False

    if not restore_invoked_at_startup and restore_succeeded is not None:
        log.warning(
            "health() called with restore_invoked_at_startup=False but "
            "restore_succeeded=%s. Setting to None to satisfy the "
            "CheckpointHealth contract.",
            restore_succeeded,
        )
        restore_succeeded = None

    # ── Resolve and validate checkpoint_dir ──────────────────────────────────
    try:
        resolved_dir = _resolve_checkpoint_dir(checkpoint_dir)
    except (TypeError, ValueError, FileNotFoundError, NotADirectoryError, PermissionError) as exc:
        log.warning(
            "health(): checkpoint_dir is inaccessible (%s: %s). "
            "Returning degraded CheckpointHealth (crond_alive=False).",
            type(exc).__name__, exc,
        )
        return _make_degraded_health(
            assessed_at=assessed_at,
            restore_invoked_at_startup=restore_invoked_at_startup,
            restore_succeeded=restore_succeeded,
        )

    # ── Discover archives ─────────────────────────────────────────────────────
    archives = _glob_archives(resolved_dir)
    checkpoint_count = len(archives)

    # ── No archives: system has never checkpointed ────────────────────────────
    if not archives:
        log.info(
            "health(): no checkpoint archives in %r — "
            "crond has never written a checkpoint, or all were deleted.",
            str(resolved_dir),
        )
        try:
            return CheckpointHealth(
                last_checkpoint_time=None,
                checkpoint_count=0,
                latest_archive_size_bytes=None,
                restore_invoked_at_startup=restore_invoked_at_startup,
                restore_succeeded=restore_succeeded,
                crond_alive=False,
                minutes_since_last_checkpoint=None,
                assessed_at=assessed_at,
            )
        except (ValueError, TypeError) as exc:
            log.error(
                "health(): CheckpointHealth construction failed for no-archives case: %s. "
                "Returning degraded health.",
                exc,
            )
            return _make_degraded_health(
                assessed_at=assessed_at,
                restore_invoked_at_startup=restore_invoked_at_startup,
                restore_succeeded=restore_succeeded,
            )

    # ── Assess latest archive ─────────────────────────────────────────────────
    latest = archives[0]

    log.debug(
        "health(): latest archive is %r (size=%d bytes mtime=%s). "
        "%d total archive(s) in directory.",
        latest.path.name,
        latest.size_bytes,
        latest.mtime_utc.isoformat(),
        checkpoint_count,
    )

    # ── Staleness check ───────────────────────────────────────────────────────
    staleness = _check_staleness(latest)

    # Emit Python warning for stale archives so monitoring integrations
    # that filter on Python warnings (e.g., pytest -W, logging.captureWarnings)
    # receive the signal without needing to parse log output.
    if staleness.is_stale:
        warnings.warn(
            (
                f"Checkpoint archive {latest.path.name!r} is stale: "
                f"age={staleness.age_seconds:.1f}s "
                f"(threshold={CHECKPOINT_STALE_THRESHOLD_SECONDS}s = "
                f"{CHECKPOINT_STALE_THRESHOLD_SECONDS / 60.0:.0f} min). "
                "crond may have silently exited."
            ),
            CheckpointStalenessWarning,
            stacklevel=2,
        )

    # ── crond liveness ────────────────────────────────────────────────────────
    crond_alive = _detect_crond_alive(staleness)

    # ── minutes_since_last_checkpoint ────────────────────────────────────────
    minutes_since: Optional[float]
    if staleness.future_skew_detected:
        # Cannot determine true age — report 0.0 (just written).
        minutes_since = 0.0
    else:
        minutes_since = round(staleness.age_seconds / 60.0, 3)

    # ── Construct and return CheckpointHealth ─────────────────────────────────
    try:
        ch = CheckpointHealth(
            last_checkpoint_time=latest.mtime_utc,
            checkpoint_count=checkpoint_count,
            latest_archive_size_bytes=latest.size_bytes,
            restore_invoked_at_startup=restore_invoked_at_startup,
            restore_succeeded=restore_succeeded,
            crond_alive=crond_alive,
            minutes_since_last_checkpoint=minutes_since,
            assessed_at=assessed_at,
        )
    except (ValueError, TypeError) as exc:
        # Should not happen given our pre-validation, but defend against
        # contracts.py changes that add new validation we haven't accounted for.
        log.error(
            "health(): CheckpointHealth construction failed unexpectedly: %s. "
            "This likely indicates a contracts.py change. "
            "Returning degraded health.",
            exc,
        )
        return _make_degraded_health(
            assessed_at=assessed_at,
            restore_invoked_at_startup=restore_invoked_at_startup,
            restore_succeeded=restore_succeeded,
        )

    log.info(
        "health(): assessed — count=%d crond_alive=%s is_healthy=%s "
        "minutes_since=%.1f stale=%s future_skew=%s "
        "restore_invoked=%s restore_ok=%s",
        checkpoint_count,
        crond_alive,
        ch.is_healthy,
        minutes_since if minutes_since is not None else -1.0,
        staleness.is_stale,
        staleness.future_skew_detected,
        restore_invoked_at_startup,
        restore_succeeded,
    )

    return ch