#!/bin/sh
# =============================================================================
# signal_kernel/entrypoint.sh
# =============================================================================
# Two processes, one container, one file.
#
# Process 2 — crond checkpoint daemon.
#   Backgrounded before exec. Fires every 15 minutes independent of Process 1.
#   Writes compressed archives of the four index files to /store/checkpoints/.
#
# Process 1 — grep pipeline.
#   exec replaces this shell with /bin/sh /recipe/run.sh entirely.
#   SIGTERM from Docker reaches the pipeline directly — no shell wrapper
#   between Docker and the process that matters.
#   pipeline.py's stdin/stdout pipe connects directly to the pipeline process.
#
# Signal flow after exec succeeds:
#   Docker stop → SIGTERM → PID 1 (pipeline process, formerly this shell)
#   Pipeline exits → Docker stop_grace_period → SIGKILL to crond if still alive
#
# This file has no business logic. It starts two processes and gets out of
# the way. If you find yourself adding logic here, the wrong architecture
# decision was made upstream.
#
# AXIOM INTERNAL // DO NOT SURFACE
# =============================================================================

RECIPE=/recipe/run.sh

# ── Pre-flight: verify recipe is mounted before starting anything ─────────────
# Detecting absence here — rather than letting exec fail silently — produces
# a stderr line that pipeline.py captures as ContainerSpawnError.os_error.
# Exit 127 (command/file not found) matches pipeline.py's _classify_exit_code
# taxonomy: "command_not_found".
if [ ! -f "$RECIPE" ]; then
    printf 'entrypoint: %s not found — recipe volume not mounted\n' "$RECIPE" >&2
    exit 127
fi

# ── Process 2: crond checkpoint daemon ───────────────────────────────────────
# -f  run in foreground (crond does not fork — we own the PID)
# -l 8  suppress Alpine dcron log output below WARNING level
# &   background — exec must follow
crond -f -l 8 &
CROND_PID=$!

# ── Trap: SIGTERM / SIGINT → kill crond, exit cleanly ────────────────────────
# This trap fires in exactly two cases:
#
#   1. SIGTERM/INT arrives in the window between crond start and exec.
#      Narrow race — containers are rarely stopped in this millisecond window —
#      but the trap covers it correctly.
#
#   2. exec fails entirely (fallthrough to exit 1 below). The shell is still
#      alive. Any subsequent signal would leave crond as an orphan without it.
#
# After exec succeeds, this shell no longer exists and the trap is gone.
# Docker's stop sequence handles crond from that point: SIGTERM to all
# remaining pids, then SIGKILL after stop_grace_period.
#
# wait "$CROND_PID" is non-negotiable: without it, the shell exits before
# crond has processed the SIGTERM — identical to not trapping at all.
trap 'kill "$CROND_PID" 2>/dev/null; wait "$CROND_PID" 2>/dev/null; exit 0' TERM INT

# ── Process 1: grep pipeline ──────────────────────────────────────────────────
# exec replaces this shell process entirely. On success it never returns.
exec /bin/sh "$RECIPE"

# ── exec fallthrough — recipe not executable or /bin/sh unavailable ──────────
# exec never returns on success. Reaching this line means the recipe is
# present on disk but /bin/sh cannot execute it. Exit 1 (general error) so
# Docker and pipeline.py's subprocess monitor both know the container failed
# to start. ContainerSpawnError in pipeline.py catches this.
printf 'entrypoint: exec /bin/sh %s failed\n' "$RECIPE" >&2
kill "$CROND_PID" 2>/dev/null
exit 1