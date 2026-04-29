# tag/store_watchdog.py
"""
store_watchdog.py
=================
inotify-based file change notification for the /store layer.

The connective tissue between learning and serving.

Every gradient step, every preparse cycle, every recipe compilation,
every phase transition — all of it compounds in real time because this
file delivers the signal.  Without it the system learns; nobody benefits.

Architecture:
    Components call register() at initialize() time, before start().
    start() arms inotify watches and launches the watch loop.
    File changes → ghost-filtered → debounced → isolated background tasks.
    No polling. No process restarts. No staleness.

Correctness invariants:
    1. A handler that exists has already passed duplicate-registration checks.
    2. A circuit-open handler is never called.  It stays open until
       reset_circuit() is called (by cold_start.py or an operator).
    3. CancelledError is never caught in _safe_dispatch — it propagates to
       asyncio so shutdown is always clean.
    4. The watchdog never touches the files it watches.  Observer only.
    5. register() after start() raises immediately.  All registrations must
       complete before start().
    6. Ghost-event filtering gates every dispatch.  If stat() says the file
       did not change, the event is suppressed.  Better to fire once than miss.
    7. Handler latency is tracked (min/max/last) per handler instance,
       accessible via health().  Witness polls this.

Dependency direction: components → store_watchdog → nothing in tag/
    Imports signal_kernel exception codes for structured log EC tagging.
    Does not import from classifier, parser, wlm, wlp, or any topology component.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import asyncio
import os # noqa
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Callable,
    Awaitable,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
)
import functools
import structlog

try:
    import inotify_simple
    HAS_INOTIFY = True
except Exception:
    HAS_INOTIFY = False

    @dataclass(frozen=True)
    class _ShimEvent:
        wd: int
        mask: int = 0
        cookie: int = 0
        name: str = ""

    class _ShimFlags:
        CLOSE_WRITE = 0
        MOVED_TO = 0

    class _ShimINotify:
        """
        Windows/dev fallback for modules that import WATCHDOG but do not require
        live Linux inotify behavior.

        The shim keeps the public shape used by StoreWatchdog so imports and
        registrations continue to work on Windows. read() yields no events,
        which is acceptable for local development flows that do not depend on
        hot-reload notifications.
        """

        def __init__(self) -> None:
            self._watches: Dict[str, int] = {}
            self._next_wd = 1

        def add_watch(self, path: str, flags: int) -> int:
            del flags
            if path not in self._watches:
                self._watches[path] = self._next_wd
                self._next_wd += 1
            return self._watches[path]

        def read(self, timeout: int = 0) -> List[_ShimEvent]:
            if timeout > 0:
                time.sleep(timeout / 1000.0)
            return []

        def close(self) -> None:
            self._watches.clear()

    class _ShimINotifyModule:
        INotify = _ShimINotify
        Event = _ShimEvent
        flags = _ShimFlags()

    inotify_simple = _ShimINotifyModule()  # type: ignore[assignment]

from signal_kernel.contracts import WatchdogHealth, WatchdogPathHealth, WatchdogHandlerHealth

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION CODES — WATCHDOG LAYER
#
# Stable short identifiers for programmatic log filtering and Witness routing.
# Format: WATCHDOG_{SPECIFIC}
# These never change for a given failure mode.  Do not rename.
#
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.exceptions import (
    EC_WATCHDOG_STARTUP,
    EC_WATCHDOG_INOTIFY_EXHAUST,
    EC_WATCHDOG_HANDLER_TIMEOUT,
    EC_WATCHDOG_DUPLICATE_REG,
    EC_WATCHDOG_POST_START_REG,
    EC_WATCHDOG_CIRCUIT_OPEN,
    EC_WATCHDOG_GHOST_EVENT,
    EC_WATCHDOG_LOOP_ERROR,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION CONSTANTS
#
# Load-bearing values with documented rationale.
# Do not change without reading the rationale and load-testing the change.
# ─────────────────────────────────────────────────────────────────────────────

STORE_ROOT = Path("/store")

# Per-file debounce in milliseconds.
#
# .pt files: 500ms.  Large files (hundreds of MB).  The OS fires IN_CLOSE_WRITE
# after the final write, but the atomic rename triggers IN_MOVED_TO, which is
# what we actually watch.  500ms absorbs any NFS lag and double-event scenarios.
#
# .mmap files: 100ms.  Written atomically via os.rename().  The window is tight
# but correct — mmap writes complete in microseconds.
#
# trigger files: generous (200–1000ms).  These are signals, not data.  The
# process that creates them is also starting up; we do not want to race it.
DEBOUNCE_MS: Dict[str, int] = {
    "topology_router.pt":          500,
    "structural_layer.pt":         500,
    "recipe_registry.mmap":        100,
    "phase_states.mmap":           100,
    "triggers/cold_start":        1000,
    "triggers/preparse":          1000,
    "triggers/reload_structural":  200,
    "triggers/hivemind_sync":      200,
}

# Handler execution timeout in seconds.
# A reload handler that exceeds this is hung or deadlocked.
# We cancel it, log EC_WATCHDOG_HANDLER_TIMEOUT, and record the failure
# against the circuit breaker.  30s is generous — PyTorch model reload
# from NVMe typically takes <5s.
HANDLER_TIMEOUT_S: float = 30.0

# Circuit breaker threshold.
# After this many consecutive failures for the same handler, the circuit
# opens: the handler is no longer called until reset_circuit() is invoked.
# Prevents a broken handler from flooding logs on every file-change event.
HANDLER_CIRCUIT_OPEN_THRESHOLD: int = 5

# Graceful shutdown drain timeout.
# On stop(), we wait this long for active handler tasks to complete before
# force-cancelling them.  15s is enough for one model reload to land.
SHUTDOWN_DRAIN_TIMEOUT_S: float = 15.0

# Watch loop poll interval in milliseconds.
# inotify.read() blocks for this long before yielding to the event loop.
# 100ms = at most 100ms latency from file change to event read.
# Increasing this reduces syscall pressure during idle; decreasing improves
# response time.  100ms is the right tradeoff for this system.
WATCH_LOOP_POLL_MS: int = 100

# Watch loop error back-off in seconds.
# After an unexpected error in the watch loop, we sleep this long before
# retrying.  Prevents a tight error loop from saturating the log.
WATCH_LOOP_ERROR_BACKOFF_S: float = 0.5

# inotify errno values that indicate fd table exhaustion.
# EMFILE (24): per-process fd limit exceeded.
# ENFILE (23): system-wide fd table full.
_INOTIFY_EXHAUST_ERRNOS: frozenset[int] = frozenset({23, 24})


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL STATE TYPES
#
# Not exported.  External callers interact through the public API only.
# Mutable — these are runtime state containers, not boundary contracts.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _HandlerState:
    """Mutable runtime state for one registered handler."""

    handler:              Callable[[], Awaitable[None]]

    # Circuit breaker
    consecutive_failures: int  = 0
    circuit_open:         bool = False

    # Aggregate counters
    total_calls:          int  = 0
    total_failures:       int  = 0
    total_timeouts:       int  = 0

    # Timing (monotonic offsets — converted to ISO at health() time)
    last_call_at:         Optional[float] = None
    last_latency_ms:      Optional[float] = None
    min_latency_ms:       Optional[float] = None
    max_latency_ms:       Optional[float] = None
    sum_latency_ms:       float           = 0.0


@dataclass
class _WatchedFile:
    """Mutable runtime state for one watched path."""

    path:            str                          # relative to STORE_ROOT
    debounce_ms:     int
    handler_states:  List[_HandlerState]          = field(default_factory=list)
    pending_task:    Optional[asyncio.Task]        = None
    last_event_at:   Optional[float]              = None   # monotonic
    last_stat:       Optional[Tuple[int, float]]  = None   # (size, mtime)
    event_count:     int                          = 0

# ─────────────────────────────────────────────────────────────────────────────
# WATCHDOG
# ─────────────────────────────────────────────────────────────────────────────

class StoreWatchdog:
    """
    inotify-based file change notifier for /store.

    Lifecycle:
        watchdog = StoreWatchdog()                    # or use WATCHDOG singleton
        watchdog.register(path, handler)              # at component initialize()
        await watchdog.start()                        # once, after all register()s
        ...                                           # handlers fire on change
        await watchdog.stop()                         # graceful drain then close

    Thread safety:
        Not thread-safe.  All calls must be made from the same asyncio event
        loop.  register() is synchronous; must be called before start().
        Post-start registration raises immediately.

    Handler contract:
        Handlers are async callables with no arguments.
        They must be idempotent — they may be called more than once.
        They own their own validation logic; the watchdog does not verify
        that a reload succeeded, only that it completed without raising.
        Handlers must not raise CancelledError; if they spin up subtasks,
        they must propagate CancelledError correctly.

    Singleton:
        WATCHDOG at module bottom is the shared instance.  Components import
        WATCHDOG and call WATCHDOG.register() at initialize() time.
    """

    def __init__(self) -> None:
        self._watched:               Dict[str, _WatchedFile]  = {}
        self._inotify:               Optional[inotify_simple.INotify] = None
        self._wd_to_dir:             Dict[int, Path]           = {}
        self._dir_to_wd:             Dict[Path, int]           = {}
        self._running:               bool                      = False
        self._started_at:            Optional[float]           = None
        self._watch_task:            Optional[asyncio.Task]    = None
        self._active_handler_tasks:  Set[asyncio.Task]         = set()
        self._total_events:          int                       = 0
        self._log = structlog.get_logger().bind(component="store_watchdog")

    # ── Public API ────────────────────────────────────────────────────────────

    def register(
        self,
        path: str,
        handler: Callable[[], Awaitable[None]],
        debounce_ms: Optional[int] = None,
    ) -> None:
        """
        Register an async handler for a path relative to STORE_ROOT.

        Called at component initialize() time — always before watchdog.start().
        Multiple handlers per path are allowed; each is dispatched independently
        as an isolated background task.  One handler's failure cannot affect another.

        Args:
            path:        Path relative to STORE_ROOT.
                         e.g. "topology_router.pt", "triggers/preparse".
            handler:     Async callable that takes no arguments.  Must be
                         idempotent.  Must not retain a reference to the watchdog.
            debounce_ms: Override the default debounce for this path.  If None,
                         uses DEBOUNCE_MS[path] → DEBOUNCE_MS[basename] → 500ms.

        Raises:
            RuntimeError: Called after start().  All registrations must complete
                          before start() is called.
            ValueError:   Same handler object registered for the same path twice.
                          Indicates a double-initialize() call upstream.
        """
        if self._running:
            self._log.error(
                "store_watchdog.post_start_registration_rejected",
                path=path,
                handler=handler.__qualname__,
                exception_code=EC_WATCHDOG_POST_START_REG,
            )
            raise RuntimeError(
                f"[{EC_WATCHDOG_POST_START_REG}] "
                f"Cannot register handlers after start(). "
                f"path={path!r} handler={handler.__qualname__!r}. "
                "All register() calls must occur before watchdog.start()."
            )

        if path not in self._watched:
            resolved_debounce = (
                debounce_ms
                if debounce_ms is not None
                else DEBOUNCE_MS.get(path, DEBOUNCE_MS.get(Path(path).name, 500))
            )
            self._watched[path] = _WatchedFile(
                path=path,
                debounce_ms=resolved_debounce,
            )

        watched = self._watched[path]

        # Duplicate registration guard.
        # Identity check (not equality) — same callable object, same path.
        existing = [hs.handler for hs in watched.handler_states]
        if handler in existing:
            self._log.warning(
                "store_watchdog.duplicate_registration_rejected",
                path=path,
                handler=handler.__qualname__,
                exception_code=EC_WATCHDOG_DUPLICATE_REG,
            )
            raise ValueError(
                f"[{EC_WATCHDOG_DUPLICATE_REG}] "
                f"Handler {handler.__qualname__!r} is already registered for "
                f"path {path!r}.  Duplicate registration — check for double "
                "initialize() calls in the registering component."
            )

        watched.handler_states.append(_HandlerState(handler=handler))

        self._log.debug(
            "store_watchdog.handler_registered",
            path=path,
            handler=handler.__qualname__,
            debounce_ms=watched.debounce_ms,
            total_handlers_for_path=len(watched.handler_states),
        )

    async def start(self) -> None:
        """
        Arm inotify watches and launch the watch loop.

        Called once, after all register() calls have completed.
        Idempotent — a second call while already running logs a warning and
        returns immediately.

        Raises:
            OSError: inotify_init() or add_watch() failed.  System fd exhaustion,
                     permission denied, or the parent directory does not exist.
                     The caller (typically cold_start.py) must treat this as fatal
                     and halt startup.
        """
        if self._running:
            self._log.warning(
                "store_watchdog.start_called_while_already_running",
                note="ignoring — watchdog is already running",
            )
            return

        try:
            self._inotify = inotify_simple.INotify()
        except OSError as exc:
            self._log.error(
                "store_watchdog.inotify_init_failed",
                error=str(exc),
                errno=exc.errno,
                exception_code=EC_WATCHDOG_STARTUP,
                hint="check /proc/sys/fs/inotify/max_user_instances",
            )
            raise OSError(
                f"[{EC_WATCHDOG_STARTUP}] inotify_init() failed — watchdog cannot start. "
                f"os error: {exc}.  "
                "Verify /proc/sys/fs/inotify/max_user_instances and max_user_watches."
            ) from exc

        dirs_watched: Set[Path] = set()

        for path, watched in self._watched.items():
            full_path  = STORE_ROOT / path
            parent_dir = full_path.parent

            # Warn if parent dir missing at startup.
            # Trigger directories may not exist until the signalling process creates them.
            # We attempt the watch anyway — add_watch() will raise if dir is truly absent.
            if not parent_dir.exists():
                self._log.warning(
                    "store_watchdog.parent_dir_missing_at_start",
                    path=path,
                    parent_dir=str(parent_dir),
                    note="parent directory must exist before events can be received",
                )

            # Note if the target file doesn't exist yet.
            # Normal for trigger files and freshly deployed stores — the watch
            # will fire when the file is first created via atomic rename.
            if not full_path.exists():
                self._log.info(
                    "store_watchdog.watched_file_not_yet_present",
                    path=path,
                    full_path=str(full_path),
                    note="will fire when the file first appears via atomic rename",
                )
            else:
                # Capture baseline stat for ghost-event filtering.
                try:
                    st = full_path.stat()
                    watched.last_stat = (st.st_size, st.st_mtime)
                except OSError:
                    pass  # non-fatal — ghost filter will assume changed on first event

            # Watch each parent directory exactly once.
            if parent_dir in dirs_watched:
                continue

            try:
                wd = self._inotify.add_watch(
                    str(parent_dir),
                    inotify_simple.flags.CLOSE_WRITE | inotify_simple.flags.MOVED_TO,
                )
            except OSError as exc:
                ec = (
                    EC_WATCHDOG_INOTIFY_EXHAUST
                    if exc.errno in _INOTIFY_EXHAUST_ERRNOS
                    else EC_WATCHDOG_STARTUP
                )
                self._log.error(
                    "store_watchdog.add_watch_failed",
                    parent_dir=str(parent_dir),
                    path=path,
                    error=str(exc),
                    errno=exc.errno,
                    exception_code=ec,
                    hint="check /proc/sys/fs/inotify/max_user_watches",
                )
                raise OSError(
                    f"[{ec}] inotify add_watch failed for {parent_dir!r}. "
                    f"os error: {exc}.  "
                    "Check /proc/sys/fs/inotify/max_user_watches."
                ) from exc

            self._wd_to_dir[wd] = parent_dir
            self._dir_to_wd[parent_dir] = wd
            dirs_watched.add(parent_dir)

        self._running    = True
        self._started_at = time.monotonic()

        self._watch_task = asyncio.create_task(
            self._watch_loop(),
            name="store_watchdog.watch_loop",
        )

        self._log.info(
            "store_watchdog.started",
            watched_paths=list(self._watched.keys()),
            watched_directories=len(dirs_watched),
            total_handlers=sum(len(w.handler_states) for w in self._watched.values()),
        )

    async def stop(self) -> None:
        """
        Graceful shutdown.

        Sequence:
          1. Clear _running flag so the watch loop exits on its next iteration.
          2. Cancel all pending debounce tasks.
          3. Cancel the watch loop task.
          4. Wait SHUTDOWN_DRAIN_TIMEOUT_S for active handler tasks to complete.
          5. Force-cancel any remaining handler tasks.
          6. Close the inotify fd.

        Idempotent — safe to call multiple times.
        """
        if not self._running:
            return

        self._running = False

        # Cancel pending debounce tasks.
        for watched in self._watched.values():
            if watched.pending_task and not watched.pending_task.done():
                watched.pending_task.cancel()

        # Stop the watch loop.
        if self._watch_task and not self._watch_task.done():
            self._watch_task.cancel()
            try:
                await self._watch_task
            except (asyncio.CancelledError, Exception):
                pass

        # Drain active handler tasks.
        if self._active_handler_tasks:
            live = set(self._active_handler_tasks)
            self._log.info(
                "store_watchdog.draining_handler_tasks",
                count=len(live),
                drain_timeout_s=SHUTDOWN_DRAIN_TIMEOUT_S,
            )
            done, pending = await asyncio.wait(
                live,
                timeout=SHUTDOWN_DRAIN_TIMEOUT_S,
            )
            for task in pending:
                task.cancel()
            if pending:
                self._log.warning(
                    "store_watchdog.shutdown_forced_cancel",
                    cancelled_count=len(pending),
                    note="handlers did not drain within shutdown window",
                )

        if self._inotify:
            self._inotify.close()
            self._inotify = None

        uptime = (
            round(time.monotonic() - self._started_at, 2)
            if self._started_at else None
        )
        self._log.info(
            "store_watchdog.stopped",
            uptime_s=uptime,
            total_events_fired=self._total_events,
        )

    def health(self) -> WatchdogHealth:
        """
        Return a frozen health snapshot.  Safe to call at any time.

        O(paths × handlers) — negligible for the expected registration count.
        Timestamp arithmetic converts monotonic offsets to approximate wall-clock
        ISO strings.  Precision is sufficient for Witness dashboards; do not use
        for audit trail timestamps.
        """
        now_mono  = time.monotonic()
        now_wall  = datetime.now(timezone.utc)
        now_iso   = now_wall.isoformat()

        path_snapshots: List[WatchdogPathHealth] = []
        open_circuit_count = 0

        for path, watched in self._watched.items():
            handler_snapshots: List[WatchdogHandlerHealth] = []
            active = 0

            for hs in watched.handler_states:
                if hs.circuit_open:
                    open_circuit_count += 1
                else:
                    active += 1

                last_call_iso: Optional[str] = None
                if hs.last_call_at is not None:
                    elapsed = now_mono - hs.last_call_at
                    approx  = now_wall.timestamp() - elapsed
                    last_call_iso = datetime.fromtimestamp(
                        approx, tz=timezone.utc
                    ).isoformat()

                avg_ms: Optional[float] = None
                if hs.total_calls > 0:
                    avg_ms = round(hs.sum_latency_ms / hs.total_calls, 2)

                handler_snapshots.append(WatchdogHandlerHealth(
                    qualified_name       = hs.handler.__qualname__,
                    path                 = path,
                    is_circuit_open      = hs.circuit_open,
                    total_calls          = hs.total_calls,
                    total_failures       = hs.total_failures,
                    total_timeouts       = hs.total_timeouts,
                    consecutive_failures = hs.consecutive_failures,
                    last_call_at_iso     = last_call_iso,
                    last_latency_ms      = hs.last_latency_ms,
                    min_latency_ms       = hs.min_latency_ms,
                    max_latency_ms       = hs.max_latency_ms,
                    avg_latency_ms       = avg_ms,
                ))

            last_event_iso: Optional[str] = None
            if watched.last_event_at is not None:
                elapsed = now_mono - watched.last_event_at
                approx  = now_wall.timestamp() - elapsed
                last_event_iso = datetime.fromtimestamp(
                    approx, tz=timezone.utc
                ).isoformat()

            path_snapshots.append(WatchdogPathHealth(
                path              = path,
                handler_count     = len(watched.handler_states),
                active_handlers   = active,
                event_count       = watched.event_count,
                last_event_at_iso = last_event_iso,
                handlers          = tuple(handler_snapshots),
            ))

        return WatchdogHealth(
            is_running         = self._running,
            is_healthy         = self._running and open_circuit_count == 0,
            uptime_s           = round(time.monotonic() - self._started_at, 2)
                                 if self._started_at else None,
            total_events_fired = self._total_events,
            open_circuit_count = open_circuit_count,
            watched_paths      = tuple(path_snapshots),
            generated_at_iso   = now_iso,
        )

    # ── Introspection helpers ─────────────────────────────────────────────────

    def registered_paths(self) -> List[str]:
        """All registered paths in insertion order."""
        return list(self._watched.keys())

    def handler_count(self, path: str) -> int:
        """Number of registered handlers for path. 0 if path is not registered."""
        watched = self._watched.get(path)
        return len(watched.handler_states) if watched else 0

    def reset_circuit(self, path: str, handler_qualname: str) -> bool:
        """
        Reset a circuit-open handler back to active.

        Called by cold_start.py (or an operator via admin tooling) after the
        underlying failure condition has been resolved.  The handler will be
        called again on the next file-change event.

        Returns True if the handler was found and reset, False otherwise.
        """
        watched = self._watched.get(path)
        if not watched:
            return False

        for hs in watched.handler_states:
            if hs.handler.__qualname__ == handler_qualname:
                hs.circuit_open         = False
                hs.consecutive_failures = 0
                self._log.info(
                    "store_watchdog.circuit_reset",
                    path=path,
                    handler=handler_qualname,
                )
                return True

        return False

    # ── Internal watch loop ───────────────────────────────────────────────────

    async def _watch_loop(self) -> None:
        """
        Main inotify event loop.  Runs as a background task from start().

        Reads events in WATCH_LOOP_POLL_MS windows, yielding to the event loop
        between each read.  On each event: maps the inotify watch descriptor to
        a directory, checks if the specific registered file changed, applies the
        ghost-event stat filter, then schedules debounced dispatch.

        CancelledError propagates cleanly — the task is cancelled by stop().
        Any other read error backs off WATCH_LOOP_ERROR_BACKOFF_S and retries,
        unless _running is False (shutdown already in progress).
        """
        loop = asyncio.get_event_loop()

        while self._running:
            try:
                events = await loop.run_in_executor(
                    None,
                    functools.partial(self._inotify.read, timeout=WATCH_LOOP_POLL_MS),
                ) # type: ignore[arg-type]
            except asyncio.CancelledError:
                # Shutdown — let it propagate so asyncio cleans up cleanly.
                raise
            except Exception as exc:
                if not self._running:
                    # inotify fd was closed by stop() — expected during shutdown.
                    break
                self._log.error(
                    "store_watchdog.inotify_read_error",
                    error=str(exc),
                    error_type=type(exc).__name__,
                    backoff_s=WATCH_LOOP_ERROR_BACKOFF_S,
                    exception_code=EC_WATCHDOG_LOOP_ERROR,
                )
                await asyncio.sleep(WATCH_LOOP_ERROR_BACKOFF_S)
                continue

            for event in events:
                await self._handle_raw_event(event)

    async def _handle_raw_event(self, event: inotify_simple.Event) -> None:
        """
        Process one raw inotify event.

        Maps watch descriptor → parent directory → registered paths.
        Checks that the changed filename matches the registered path's basename.
        Applies stat-based ghost-event filtering.
        Schedules debounced dispatch for matching paths.
        """
        parent_dir = self._wd_to_dir.get(event.wd)
        if parent_dir is None:
            return  # event from a directory we no longer track

        changed_name = event.name  # basename only; no directory component

        for path, watched in self._watched.items():
            full_path = STORE_ROOT / path

            # Parent directory must match.
            if full_path.parent != parent_dir:
                continue

            # Filename must match exactly.
            if full_path.name != changed_name:
                continue

            # Ghost-event filter: stat() must confirm the file actually changed.
            # inotify can fire on some filesystems (NFS, overlayfs) without a
            # real write.  We suppress those here.
            if not self._stat_changed(full_path, watched):
                self._log.debug(
                    "store_watchdog.ghost_event_suppressed",
                    path=path,
                    exception_code=EC_WATCHDOG_GHOST_EVENT,
                )
                continue

            await self._schedule_handlers(path)

    def _stat_changed(self, full_path: Path, watched: _WatchedFile) -> bool: # noqa
        """
        Return True if the file's size or mtime differs from the last captured stat.
        Updates watched.last_stat on a detected change.

        Returns True (fire the handlers) when stat() fails — the file may have
        been renamed away (trigger files are sometimes removed after dispatch).
        We prefer a spurious handler call over a missed file-change event.
        """
        try:
            st           = full_path.stat()
            current_stat = (st.st_size, st.st_mtime)
        except OSError:
            # File gone — treat as changed.  Handler will observe current state.
            watched.last_stat = None
            return True

        if watched.last_stat is None or current_stat != watched.last_stat:
            watched.last_stat = current_stat
            return True

        return False

    async def _schedule_handlers(self, path: str) -> None:
        """
        (Re-)schedule debounced dispatch for path.

        Cancels any existing debounce task for this path before creating a new
        one.  This absorbs rapid successive events (e.g., multiple CLOSE_WRITE
        events during a multi-part write) into a single dispatch.
        """
        watched = self._watched[path]

        if watched.pending_task and not watched.pending_task.done():
            watched.pending_task.cancel()

        debounce_s = watched.debounce_ms / 1000.0
        watched.pending_task = asyncio.create_task(
            self._debounced_dispatch(path, debounce_s),
            name=f"store_watchdog.debounce.{Path(path).name}",
        )

    async def _debounced_dispatch(self, path: str, debounce_s: float) -> None:
        """
        Sleep the debounce window, then dispatch all active handlers for path.

        CancelledError from _schedule_handlers (re-debounce) propagates cleanly.
        The replacement task will fire after its own debounce window.  We do not
        catch CancelledError here — it is the correct re-debounce mechanism.
        """
        await asyncio.sleep(debounce_s)

        watched = self._watched[path]
        watched.last_event_at  = time.monotonic()
        watched.event_count   += 1
        self._total_events    += 1

        active_states  = [hs for hs in watched.handler_states if not hs.circuit_open]
        skipped_count  = len(watched.handler_states) - len(active_states)

        self._log.info(
            "store_watchdog.file_changed",
            path=path,
            handlers_dispatching=len(active_states),
            handlers_circuit_open=skipped_count,
            cumulative_event_number=watched.event_count,
        )

        if skipped_count:
            self._log.warning(
                "store_watchdog.circuit_open_handlers_skipped",
                path=path,
                skipped_count=skipped_count,
                exception_code=EC_WATCHDOG_CIRCUIT_OPEN,
                recovery="call StoreWatchdog.reset_circuit() or restart the process",
            )

        # Dispatch each active handler as an independent background task.
        # Handlers are tracked in _active_handler_tasks for shutdown draining.
        for hs in active_states:
            task = asyncio.create_task(
                self._safe_dispatch(path, hs),
                name=f"store_watchdog.handler.{hs.handler.__qualname__}",
            )
            self._active_handler_tasks.add(task)
            task.add_done_callback(self._active_handler_tasks.discard)

    async def _safe_dispatch(self, path: str, hs: _HandlerState) -> None:
        """
        Execute one handler with timeout enforcement and circuit-breaker accounting.

        This is the outermost exception boundary for handler execution.

        Success path:
            Resets consecutive_failure counter.  Records latency.  Logs completion.

        Timeout path:
            Cancels the hung handler.  Increments failure counters.
            Logs EC_WATCHDOG_HANDLER_TIMEOUT.  Checks circuit threshold.

        Exception path:
            Increments failure counters.  Logs full error with type.
            Checks circuit threshold.

        CancelledError path (shutdown in progress):
            Does NOT increment failure counters — cancellation is not a failure.
            Re-raises so asyncio can clean up the task properly.

        Never re-raises except for CancelledError.
        """
        hs.total_calls  += 1
        hs.last_call_at  = time.monotonic()
        t0               = time.perf_counter()

        try:
            await asyncio.wait_for(hs.handler(), timeout=HANDLER_TIMEOUT_S)

            # ── Success ──────────────────────────────────────────────────────
            latency_ms               = (time.perf_counter() - t0) * 1000.0
            hs.last_latency_ms       = round(latency_ms, 2)
            hs.sum_latency_ms       += latency_ms
            hs.min_latency_ms        = (
                min(hs.min_latency_ms, latency_ms)
                if hs.min_latency_ms is not None else latency_ms
            )
            hs.max_latency_ms        = (
                max(hs.max_latency_ms, latency_ms)
                if hs.max_latency_ms is not None else latency_ms
            )
            hs.consecutive_failures  = 0

            self._log.info(
                "store_watchdog.handler_complete",
                path=path,
                handler=hs.handler.__qualname__,
                latency_ms=hs.last_latency_ms,
            )

        except asyncio.TimeoutError:
            # ── Timeout ───────────────────────────────────────────────────────
            # wait_for() already cancelled the inner coroutine.
            latency_ms               = (time.perf_counter() - t0) * 1000.0
            hs.last_latency_ms       = round(latency_ms, 2)
            hs.total_failures       += 1
            hs.total_timeouts       += 1
            hs.consecutive_failures += 1

            self._log.error(
                "store_watchdog.handler_timeout",
                path=path,
                handler=hs.handler.__qualname__,
                timeout_s=HANDLER_TIMEOUT_S,
                consecutive_failures=hs.consecutive_failures,
                exception_code=EC_WATCHDOG_HANDLER_TIMEOUT,
            )
            self._maybe_open_circuit(path, hs)

        except asyncio.CancelledError:
            # ── Shutdown cancellation — not a failure ─────────────────────────
            # Do not update failure counters.
            # Re-raise so asyncio can propagate the cancellation properly.
            raise

        except Exception as exc:
            # ── Handler raised ────────────────────────────────────────────────
            latency_ms               = (time.perf_counter() - t0) * 1000.0
            hs.last_latency_ms       = round(latency_ms, 2)
            hs.total_failures       += 1
            hs.consecutive_failures += 1

            self._log.error(
                "store_watchdog.handler_failed",
                path=path,
                handler=hs.handler.__qualname__,
                error=str(exc),
                error_type=type(exc).__name__,
                consecutive_failures=hs.consecutive_failures,
            )
            self._maybe_open_circuit(path, hs)

    def _maybe_open_circuit(self, path: str, hs: _HandlerState) -> None:
        """
        Open the circuit breaker if the consecutive failure threshold is exceeded.

        Once open, hs is skipped in _debounced_dispatch until reset_circuit()
        is called.  The Witness health poller will detect is_circuit_open=True
        in the next health() snapshot.
        """
        if hs.consecutive_failures >= HANDLER_CIRCUIT_OPEN_THRESHOLD:
            hs.circuit_open = True
            self._log.error(
                "store_watchdog.circuit_opened",
                path=path,
                handler=hs.handler.__qualname__,
                consecutive_failures=hs.consecutive_failures,
                threshold=HANDLER_CIRCUIT_OPEN_THRESHOLD,
                exception_code=EC_WATCHDOG_CIRCUIT_OPEN,
                recovery=(
                    "call StoreWatchdog.reset_circuit(path, handler_qualname) after "
                    "resolving the underlying failure, or restart the process.  "
                    "cold_start.py handles automated recovery on process restart."
                ),
            )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
#
# The single shared instance.  Every component that reads from /store imports
# this and calls WATCHDOG.register() at initialize() time.
# cold_start.py (or the orchestrator entrypoint) calls await WATCHDOG.start()
# exactly once, after all component initialize() calls complete.
# ─────────────────────────────────────────────────────────────────────────────

WATCHDOG: StoreWatchdog = StoreWatchdog()
