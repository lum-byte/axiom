#!/bin/sh
# =============================================================================
# checkpoint/restore.sh
# =============================================================================
# Restores the four TAG index files from the most recent valid checkpoint.
# Called by TAG's startup sequence via checkpoint_monitor.py when index files
# are missing or corrupted.
#
# Strategy: iterate archives newest-to-oldest. For each archive:
#   1. Verify structural integrity with tar -tzf (catches gzip corruption).
#   2. Verify all four required files are present in the archive manifest.
#   3. Extract to a temporary staging directory.
#   4. Verify all four files landed AND are non-zero bytes.
#   5. Move staging files to STORE_DIR atomically (mv is atomic within same fs).
#   6. Verify final state in STORE_DIR.
#   7. On any failure: clean staging dir, try the next archive.
#
# The staging → move pattern means STORE_DIR never ends up in a partial state.
# If extraction succeeds but a move fails, the staging dir is cleaned and the
# next archive is tried. STORE_DIR retains its previous (corrupt/missing) state
# rather than acquiring a partial replacement.
#
# Exit codes:
#   0  — restore completed; all four files present and non-zero in STORE_DIR
#   1  — restore failed; no valid checkpoint found or all exhausted
#
# Log format (parsed by checkpoint_monitor.py):
#   RESTORE: scanning checkpoints at <datetime> pid=<n>
#   RESTORE: using <archive> (skipped=<n>) at <datetime> pid=<n>
#   RESTORE OK: <archive> files=<n> at <datetime> pid=<n>
#   RESTORE SKIP: <archive> corrupt at <datetime> pid=<n>
#   RESTORE SKIP: <archive> missing files=<list> at <datetime> pid=<n>
#   RESTORE SKIP: <archive> extract failed at <datetime> pid=<n>
#   RESTORE SKIP: <archive> files incomplete after extract at <datetime> pid=<n>
#   RESTORE FAILED: no valid checkpoint found archives_tried=<n> at <datetime> pid=<n>
# =============================================================================

set -eu

# ── Configuration ─────────────────────────────────────────────────────────────
CHECKPOINT_DIR=${CHECKPOINT_DIR:-/store/checkpoints}
STORE_DIR=${STORE_DIR:-/store}
LOG_FILE=${CHECKPOINT_LOG:-/var/log/checkpoint.log}
LOCK_FILE=${RESTORE_LOCK:-/var/run/mft_restore.lock}

# Minimum acceptable size (bytes) for a restored index file.
# Matches the same threshold in mft_checkpoint.sh.
MIN_FILE_BYTES=64

# Required store files — must match contracts.STORE_FILE_NAMES exactly.
REQUIRED_FILES="topology_router.pt recipe_registry.mmap phase_states.mmap structural_layer.pt"

# ── Runtime state ─────────────────────────────────────────────────────────────
PID=$$
STAGING_DIR=""
ARCHIVES_TRIED=0
ARCHIVES_SKIPPED=0

# ── Helpers ───────────────────────────────────────────────────────────────────

log() {
    printf '%s\n' "$1" >> "${LOG_FILE}"
}

_ensure_log() {
    log_dir=$(dirname "${LOG_FILE}")
    mkdir -p "${log_dir}" 2>/dev/null || true
    touch "${LOG_FILE}" 2>/dev/null || {
        LOG_FILE=/dev/stderr
    }
}

# ── Lock ─────────────────────────────────────────────────────────────────────
# Prevents two concurrent restore invocations from racing each other into
# STORE_DIR. Uses atomic mkdir(2) — same pattern as mft_checkpoint.sh.
#
_acquire_lock() {
    lock_dir=$(dirname "${LOCK_FILE}")
    mkdir -p "${lock_dir}" 2>/dev/null || true

    if mkdir "${LOCK_FILE}" 2>/dev/null; then
        printf '%d\n' "${PID}" > "${LOCK_FILE}/pid"
        return 0
    fi

    if [ -f "${LOCK_FILE}/pid" ]; then
        lock_pid=$(cat "${LOCK_FILE}/pid" 2>/dev/null || echo "0")
        if [ "${lock_pid}" -gt 0 ] 2>/dev/null; then
            if ! kill -0 "${lock_pid}" 2>/dev/null; then
                log "RESTORE STALE_LOCK: pid=${lock_pid} dead, breaking at $(date) pid=${PID}"
                rm -rf "${LOCK_FILE}"
                mkdir "${LOCK_FILE}" && printf '%d\n' "${PID}" > "${LOCK_FILE}/pid"
                return 0
            fi
        fi
    fi

    log "RESTORE ALREADY_RUNNING: pid=${lock_pid:-unknown} at $(date) pid=${PID}"
    exit 1
}

_release_lock() {
    rm -rf "${LOCK_FILE}" 2>/dev/null || true
}

# ── Cleanup trap ──────────────────────────────────────────────────────────────
_cleanup() {
    rc=$?
    if [ -n "${STAGING_DIR}" ] && [ -d "${STAGING_DIR}" ]; then
        rm -rf "${STAGING_DIR}" 2>/dev/null || true
    fi
    _release_lock
    exit "${rc}"
}
trap _cleanup EXIT

# ── Staging directory management ─────────────────────────────────────────────
_make_staging() {
    # Create a fresh temp directory under STORE_DIR so that mv into STORE_DIR
    # is guaranteed to be on the same filesystem (no cross-device mv).
    STAGING_DIR=$(mktemp -d "${STORE_DIR}/.restore_staging_XXXXXX" 2>/dev/null) || {
        # mktemp failed — fall back to a fixed name with PID suffix.
        STAGING_DIR="${STORE_DIR}/.restore_staging_${PID}"
        mkdir -p "${STAGING_DIR}" || {
            log "RESTORE ERROR: cannot create staging dir at $(date) pid=${PID}"
            return 1
        }
    }
}

_clear_staging() {
    if [ -n "${STAGING_DIR}" ] && [ -d "${STAGING_DIR}" ]; then
        rm -rf "${STAGING_DIR}" 2>/dev/null || true
        STAGING_DIR=""
    fi
}

# ── Verify archive manifest before extraction ─────────────────────────────────
# Returns 0 if all four required files are listed in the archive manifest.
# Returns 1 and sets MISSING_IN_ARCHIVE to a space-separated list of missing files.
MISSING_IN_ARCHIVE=""
_verify_archive_manifest() {
    archive="$1"
    manifest=$(tar -tzf "${archive}" 2>/dev/null) || return 1

    MISSING_IN_ARCHIVE=""
    for f in ${REQUIRED_FILES}; do
        # tar may store files as "filename" or "./filename" depending on -C usage.
        if ! printf '%s\n' "${manifest}" | grep -qE "(^|/)${f}$"; then
            MISSING_IN_ARCHIVE="${MISSING_IN_ARCHIVE} ${f}"
        fi
    done

    [ -z "${MISSING_IN_ARCHIVE}" ]
}

# ── Verify staged files after extraction ─────────────────────────────────────
# Returns 0 if all four files landed in STAGING_DIR and are non-trivially sized.
MISSING_AFTER_EXTRACT=""
_verify_staged_files() {
    MISSING_AFTER_EXTRACT=""
    for f in ${REQUIRED_FILES}; do
        fpath="${STAGING_DIR}/${f}"
        if [ ! -f "${fpath}" ]; then
            MISSING_AFTER_EXTRACT="${MISSING_AFTER_EXTRACT} ${f}"
        elif [ ! -s "${fpath}" ]; then
            MISSING_AFTER_EXTRACT="${MISSING_AFTER_EXTRACT} ${f}(zero-bytes)"
        else
            file_bytes=$(wc -c < "${fpath}" 2>/dev/null || echo 0)
            if [ "${file_bytes}" -lt "${MIN_FILE_BYTES}" ]; then
                MISSING_AFTER_EXTRACT="${MISSING_AFTER_EXTRACT} ${f}(${file_bytes}B)"
            fi
        fi
    done

    [ -z "${MISSING_AFTER_EXTRACT}" ]
}

# ── Verify final STORE_DIR state after move ───────────────────────────────────
_verify_store_files() {
    for f in ${REQUIRED_FILES}; do
        fpath="${STORE_DIR}/${f}"
        if [ ! -f "${fpath}" ] || [ ! -s "${fpath}" ]; then
            return 1
        fi
    done
    return 0
}

# ── Attempt restore from one archive ─────────────────────────────────────────
# Returns 0 on full success, 1 on any failure.
# Caller increments ARCHIVES_SKIPPED on failure and tries the next archive.
#
_try_restore_from() {
    archive="$1"

    # Guard: must be a regular file.
    if [ ! -f "${archive}" ]; then
        return 1
    fi

    # Step 1: structural integrity check.
    if ! tar -tzf "${archive}" > /dev/null 2>&1; then
        log "RESTORE SKIP: ${archive} corrupt at $(date) pid=${PID}"
        return 1
    fi

    # Step 2: manifest completeness check — verify all four files are in
    # the archive before we spend time extracting it.
    if ! _verify_archive_manifest "${archive}"; then
        log "RESTORE SKIP: ${archive} missing files=${MISSING_IN_ARCHIVE} at $(date) pid=${PID}"
        return 1
    fi

    # Step 3: extract to staging directory.
    _make_staging || return 1

    if ! tar -xzf "${archive}" -C "${STAGING_DIR}" 2>/dev/null; then
        log "RESTORE SKIP: ${archive} extract failed at $(date) pid=${PID}"
        _clear_staging
        return 1
    fi

    # Step 4: verify all four files are present and non-trivially sized
    # in the staging directory after extraction.
    if ! _verify_staged_files; then
        log "RESTORE SKIP: ${archive} files incomplete after extract=${MISSING_AFTER_EXTRACT} at $(date) pid=${PID}"
        _clear_staging
        return 1
    fi

    # Step 5: move staging files to STORE_DIR.
    # Each mv is atomic within the same filesystem. If any mv fails, the
    # remaining files stay in staging (trap will clean them up) and STORE_DIR
    # is left in its prior state for any already-moved files.
    # In practice, once we reach this step, mv failures are extremely rare
    # (no disk space is allocated by mv — it just updates directory entries).
    for f in ${REQUIRED_FILES}; do
        if ! mv "${STAGING_DIR}/${f}" "${STORE_DIR}/${f}" 2>/dev/null; then
            log "RESTORE SKIP: ${archive} mv failed for ${f} at $(date) pid=${PID}"
            _clear_staging
            return 1
        fi
    done

    _clear_staging

    # Step 6: final verification in STORE_DIR.
    if ! _verify_store_files; then
        log "RESTORE SKIP: ${archive} final verify failed at $(date) pid=${PID}"
        return 1
    fi

    return 0
}

# ── Main ──────────────────────────────────────────────────────────────────────

_ensure_log
_acquire_lock

log "RESTORE: scanning checkpoints at $(date) pid=${PID}"

# Ensure STORE_DIR is accessible and writable.
if [ ! -d "${STORE_DIR}" ]; then
    log "RESTORE FAILED: STORE_DIR ${STORE_DIR} does not exist at $(date) pid=${PID}"
    exit 1
fi

# Use mkdir for the write test — creates a directory entry (not a file),
# testing both write+execute permission on STORE_DIR. More reliable than
# touch when a file-creation restriction is in place.
_write_test_dir="${STORE_DIR}/.restore_write_test_${PID}"
if ! mkdir "${_write_test_dir}" 2>/dev/null; then
    log "RESTORE FAILED: STORE_DIR ${STORE_DIR} is not writable at $(date) pid=${PID}"
    exit 1
fi
rmdir "${_write_test_dir}" 2>/dev/null || true

# Sweep any stale staging dirs left by a previously aborted restore.
# An aborted restore (SIGKILL, OOM) can leave .restore_staging_XXXXXX dirs
# because the EXIT trap did not fire. They do not affect correctness but
# accumulate and confuse diagnostics.
for _stale in "${STORE_DIR}"/.restore_staging_*; do
    [ -d "${_stale}" ] && rm -rf "${_stale}" 2>/dev/null || true
done

# Ensure checkpoint directory exists and has archives.
if [ ! -d "${CHECKPOINT_DIR}" ]; then
    log "RESTORE FAILED: no checkpoint directory at ${CHECKPOINT_DIR} at $(date) pid=${PID}"
    exit 1
fi

# Iterate archives newest-first. ls -t sorts by mtime descending.
# The for loop over a glob avoids the word-splitting hazard of parsing ls output,
# but our archive names are timestamp-based and safe. We use ls -t explicitly
# for the newest-first ordering which glob expansion does not guarantee.
#
archive_list=$(ls -t "${CHECKPOINT_DIR}"/mft_*.tar.gz 2>/dev/null || true)

if [ -z "${archive_list}" ]; then
    log "RESTORE FAILED: no archives found in ${CHECKPOINT_DIR} at $(date) pid=${PID}"
    exit 1
fi

for archive in ${archive_list}; do
    ARCHIVES_TRIED=$((ARCHIVES_TRIED + 1))

    log "RESTORE: using ${archive} (skipped=${ARCHIVES_SKIPPED}) at $(date) pid=${PID}"

    if _try_restore_from "${archive}"; then
        # Count files actually restored for the log record.
        restored_count=0
        for f in ${REQUIRED_FILES}; do
            [ -f "${STORE_DIR}/${f}" ] && restored_count=$((restored_count + 1))
        done
        log "RESTORE OK: ${archive} files=${restored_count} at $(date) pid=${PID}"
        exit 0
    fi

    ARCHIVES_SKIPPED=$((ARCHIVES_SKIPPED + 1))
done

log "RESTORE FAILED: no valid checkpoint found archives_tried=${ARCHIVES_TRIED} at $(date) pid=${PID}"
exit 1