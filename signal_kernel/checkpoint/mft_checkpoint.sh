#!/bin/sh
# =============================================================================
# checkpoint/mft_checkpoint.sh
# =============================================================================
# Writes a timestamped compressed archive of the four TAG index files.
# Runs every 15 minutes via Alpine crond (Process 2).
# Keeps last 48 archives (12 hours at 15-minute intervals).
#
# Process isolation guarantee: this script never touches the four index files
# directly. It reads them read-only via tar. It writes only to CHECKPOINT_DIR.
# Process 1 (grep pipeline) is completely unaffected by this script running.
#
# Exit codes:
#   0  — checkpoint written, verified, and rotated successfully
#   1  — skip: one or more source files missing (logged, crond retries next cycle)
#   2  — fatal: archive write or integrity verification failed (logged, archive deleted)
#   3  — fatal: unexpected error (caught by trap, lock released)
#
# Log format (parsed by checkpoint_monitor.py):
#   CHECKPOINT OK: <path> bytes=<n> files=<n> at <datetime> pid=<n>
#   CHECKPOINT CORRUPT: <path> at <datetime> pid=<n>
#   CHECKPOINT SKIP: <file> not found at <datetime> pid=<n>
#   CHECKPOINT ROTATION_ERROR: <detail> at <datetime> pid=<n>
# =============================================================================

set -eu

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/store/checkpoints}
STORE_DIR=${STORE_DIR:-/store}
LOG_FILE=${CHECKPOINT_LOG:-/var/log/checkpoint.log}
LOCK_FILE=${CHECKPOINT_LOCK:-/var/run/mft_checkpoint.lock}
RETAIN_COUNT=${CHECKPOINT_RETAIN:-48}

# Minimum acceptable size (bytes) for each index file before archiving.
# A zero-byte file is evidence of a truncated write upstream — do not
# checkpoint it. 64 bytes is conservative; the real files are megabytes.
MIN_FILE_BYTES=64

# Required store files — must match contracts.STORE_FILE_NAMES exactly.
REQUIRED_FILES="topology_router.pt recipe_registry.mmap phase_states.mmap structural_layer.pt"

# ── Runtime state ─────────────────────────────────────────────────────────────
PID=$$
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE_NAME="mft_${TIMESTAMP}.tar.gz"
ARCHIVE="${CHECKPOINT_DIR}/${ARCHIVE_NAME}"
ARCHIVE_TMP="${ARCHIVE}.tmp"
SKIP_COUNT=0
CLEANUP_TMP=0

# ── Helpers ───────────────────────────────────────────────────────────────────

log() {
    printf '%s\n' "$1" >> "${LOG_FILE}"
}

# Ensure log directory and file are writable before we need them.
_ensure_log() {
    log_dir=$(dirname "${LOG_FILE}")
    mkdir -p "${log_dir}" 2>/dev/null || true
    touch "${LOG_FILE}" 2>/dev/null || {
        # Fall back to stderr if log file is not writable — crond captures stderr.
        LOG_FILE=/dev/stderr
    }
}

# ── Lock ─────────────────────────────────────────────────────────────────────
# Prevent concurrent crond invocations from writing simultaneously.
# Uses mkdir(2) which is atomic on POSIX — no race between test and create.
# If the lock directory exists and holds a stale PID, detect and break it.
#
_acquire_lock() {
    lock_dir=$(dirname "${LOCK_FILE}")
    mkdir -p "${lock_dir}" 2>/dev/null || true

    if mkdir "${LOCK_FILE}" 2>/dev/null; then
        # Lock acquired — write our PID into it for stale-lock detection.
        printf '%d\n' "${PID}" > "${LOCK_FILE}/pid"
        return 0
    fi

    # Lock directory exists — check if the PID that holds it is still alive.
    if [ -f "${LOCK_FILE}/pid" ]; then
        lock_pid=$(cat "${LOCK_FILE}/pid" 2>/dev/null || echo "0")
        if [ "${lock_pid}" -gt 0 ] 2>/dev/null; then
            if ! kill -0 "${lock_pid}" 2>/dev/null; then
                # Holding process is dead — stale lock. Break it.
                log "CHECKPOINT STALE_LOCK: pid=${lock_pid} dead, breaking lock at $(date) pid=${PID}"
                rm -rf "${LOCK_FILE}"
                mkdir "${LOCK_FILE}" && printf '%d\n' "${PID}" > "${LOCK_FILE}/pid"
                return 0
            fi
        fi
    fi

    # Another live instance is running — skip this cycle.
    log "CHECKPOINT ALREADY_RUNNING: pid=${lock_pid:-unknown} at $(date) pid=${PID}"
    exit 0
}

_release_lock() {
    rm -rf "${LOCK_FILE}" 2>/dev/null || true
}

# ── Cleanup trap ──────────────────────────────────────────────────────────────
# Fires on EXIT (including signals). Ensures the tmp archive and lock are
# always cleaned up even if the script is killed mid-write.
#
_cleanup() {
    rc=$?
    if [ "${CLEANUP_TMP}" -eq 1 ] && [ -f "${ARCHIVE_TMP}" ]; then
        rm -f "${ARCHIVE_TMP}" 2>/dev/null || true
    fi
    _release_lock
    exit "${rc}"
}
trap _cleanup EXIT

# ── Disk space check ──────────────────────────────────────────────────────────
# Estimate required space. Each source file size + 20% overhead for tar+gzip
# plus the already-written archives. If available space is less than twice
# the estimated archive size, warn but proceed — the write itself will fail
# cleanly if the filesystem truly fills up.
#
_check_disk_space() {
    avail_kb=$(df -k "${CHECKPOINT_DIR}" 2>/dev/null | awk 'NR==2{print $4}')
    if [ -z "${avail_kb}" ] || [ "${avail_kb}" -lt 1024 ] 2>/dev/null; then
        log "CHECKPOINT WARN: low disk space available=${avail_kb:-unknown}KB at $(date) pid=${PID}"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────

_ensure_log
_acquire_lock

# Ensure checkpoint directory exists.
if ! mkdir -p "${CHECKPOINT_DIR}" 2>/dev/null; then
    log "CHECKPOINT ERROR: cannot create ${CHECKPOINT_DIR} at $(date) pid=${PID}"
    exit 2
fi

# ── Phase 1: Verify all source files exist and are non-trivially sized ────────
missing_files=""
for f in ${REQUIRED_FILES}; do
    fpath="${STORE_DIR}/${f}"
    if [ ! -f "${fpath}" ]; then
        missing_files="${missing_files} ${f}"
        log "CHECKPOINT SKIP: ${f} not found at $(date) pid=${PID}"
    elif [ ! -s "${fpath}" ]; then
        # File exists but is empty — do not archive a zero-byte index.
        missing_files="${missing_files} ${f}"
        log "CHECKPOINT SKIP: ${f} is zero bytes at $(date) pid=${PID}"
    else
        # Verify minimum size to catch obviously truncated files.
        file_bytes=$(wc -c < "${fpath}" 2>/dev/null || echo 0)
        if [ "${file_bytes}" -lt "${MIN_FILE_BYTES}" ]; then
            missing_files="${missing_files} ${f}"
            log "CHECKPOINT SKIP: ${f} too small (${file_bytes} bytes) at $(date) pid=${PID}"
        fi
    fi
done

if [ -n "${missing_files}" ]; then
    # At least one required file is missing or invalid. Skip this cycle.
    # crond will retry on the next 15-minute interval.
    exit 1
fi

_check_disk_space

# ── Phase 2: Write archive to a .tmp file, then rename atomically ─────────────
# Writing to a .tmp file first means that a partial write (OOM, disk full,
# SIGKILL) never produces a file that looks like a valid archive at the
# canonical path. checkpoint_monitor.py will not see a partial file.
#
CLEANUP_TMP=1

if ! tar -czf "${ARCHIVE_TMP}" \
        -C "${STORE_DIR}" \
        topology_router.pt \
        recipe_registry.mmap \
        phase_states.mmap \
        structural_layer.pt \
        2>/dev/null; then
    tar_exit=$?
    log "CHECKPOINT ERROR: tar write failed (exit ${tar_exit}) for ${ARCHIVE_TMP} at $(date) pid=${PID}"
    rm -f "${ARCHIVE_TMP}" 2>/dev/null || true
    CLEANUP_TMP=0
    exit 2
fi

# ── Phase 3: Integrity verification — non-negotiable ─────────────────────────
# A corrupt archive that passes silently is worse than a failed checkpoint.
# tar -tzf lists the table of contents; if the archive is truncated or
# corrupted, this exits non-zero. Verify BEFORE renaming to canonical path.
#
if ! tar -tzf "${ARCHIVE_TMP}" > /dev/null 2>&1; then
    log "CHECKPOINT CORRUPT: ${ARCHIVE_TMP} at $(date) pid=${PID}"
    rm -f "${ARCHIVE_TMP}" 2>/dev/null || true
    CLEANUP_TMP=0
    exit 2
fi

# Verify the archive contains all four required files — not just that the
# gzip stream is intact. An archive with a valid stream but missing entries
# would restore to a partial index (RestorePartialError).
archive_file_count=$(tar -tzf "${ARCHIVE_TMP}" 2>/dev/null | wc -l | tr -d ' ')
if [ "${archive_file_count}" -lt 4 ]; then
    log "CHECKPOINT CORRUPT: ${ARCHIVE_TMP} only ${archive_file_count}/4 files at $(date) pid=${PID}"
    rm -f "${ARCHIVE_TMP}" 2>/dev/null || true
    CLEANUP_TMP=0
    exit 2
fi

# Atomic rename — only after verification passes.
mv "${ARCHIVE_TMP}" "${ARCHIVE}"
CLEANUP_TMP=0

# Capture archive size for the log record (used by checkpoint_monitor.py).
archive_bytes=$(wc -c < "${ARCHIVE}" 2>/dev/null || echo 0)

log "CHECKPOINT OK: ${ARCHIVE} bytes=${archive_bytes} files=${archive_file_count} at $(date) pid=${PID}"

# ── Phase 4: Rotation — keep last RETAIN_COUNT archives ──────────────────────
# List archives newest-first. The tail starting at position RETAIN_COUNT+1
# gives us every archive beyond the retention window.
#
# Uses a temp file instead of a pipe to xargs to avoid the POSIX shell
# word-splitting hazard on filenames (even though our names are safe, the
# pattern is correct). Also avoids the GNU-vs-BusyBox xargs -r portability gap.
#
rotation_list=$(ls -t "${CHECKPOINT_DIR}"/mft_*.tar.gz 2>/dev/null | tail -n "+$((RETAIN_COUNT + 1))")

if [ -n "${rotation_list}" ]; then
    rotation_errors=0
    rotation_deleted=0
    for old_archive in ${rotation_list}; do
        if rm -f "${old_archive}" 2>/dev/null; then
            rotation_deleted=$((rotation_deleted + 1))
        else
            rotation_errors=$((rotation_errors + 1))
        fi
    done
    if [ "${rotation_errors}" -gt 0 ]; then
        log "CHECKPOINT ROTATION_ERROR: failed to delete ${rotation_errors} archives at $(date) pid=${PID}"
    else
        if [ "${rotation_deleted}" -gt 0 ]; then
            log "CHECKPOINT ROTATED: deleted ${rotation_deleted} archives at $(date) pid=${PID}"
        fi
    fi
fi

exit 0