# tests/test_store_watchdog.py
"""
Comprehensive test suite for store_watchdog.py.

Coverage:
    Registration        — normal, post-start, duplicate, debounce resolution
    Start               — inotify setup, idempotency, missing dirs, baseline stat
    Stop                — graceful drain, force-cancel, fd close, idempotency
    _stat_changed       — all four branches (no baseline, unchanged, size, mtime, OSError)
    _handle_raw_event   — unknown wd, wrong filename, ghost suppression, real dispatch
    _schedule_handlers  — task creation, prior task cancellation
    _debounced_dispatch — sleep fires, circuit-open skip, event counters, CancelledError
    _safe_dispatch      — success, timeout, exception, CancelledError re-raise,
                          latency tracking (min/max/avg), counter accuracy
    Circuit breaker     — threshold boundary, open blocks calls, reset re-arms,
                          re-opens after reset-then-failures, isolation between handlers
    Handler isolation   — one handler raises, sibling is unaffected
    health()            — all fields, before/after start, open circuits, latency stats
    reset_circuit()     — found, not found, unknown path
    Debounce resolution — full path key, basename key, fallback 500ms, explicit override
    Rapid events        — debounce collapses N events into one dispatch
    Full lifecycle      — register → start → event → handler called → stop

Run:
    python3 -m pytest tests/test_store_watchdog.py -v          (if pytest installed)
    python3 -m unittest tests.test_store_watchdog -v           (stdlib only)
    python3 test_store_watchdog.py                             (direct)
"""

from __future__ import annotations

import asyncio
import sys
import types
import unittest
import tempfile
import os # noqa
from collections import namedtuple
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock, patch, call # noqa

# ─────────────────────────────────────────────────────────────────────────────
# MOCK EXTERNAL DEPENDENCIES
# Must be injected into sys.modules BEFORE store_watchdog is imported.
# ─────────────────────────────────────────────────────────────────────────────

# ── inotify_simple ────────────────────────────────────────────────────────────

FakeEvent = namedtuple("FakeEvent", ["wd", "mask", "cookie", "name"])

class FakeINotify:
    """Controllable stand-in for inotify_simple.INotify."""

    def __init__(self):
        self._closed  = False
        self._events  = []          # pre-loaded events to return from read()
        self.add_watch_calls: list  = []
        self.close_called: bool     = False
        self._next_wd: int          = 1

    def add_watch(self, path: str, mask: int) -> int:
        wd = self._next_wd
        self._next_wd += 1
        self.add_watch_calls.append((path, mask))
        return wd

    def read(self, timeout: int = 0):
        if self._events:
            batch = list(self._events)
            self._events.clear()
            return batch
        return []

    def close(self):
        self._closed      = True
        self.close_called = True

    def push_event(self, event: FakeEvent):
        self._events.append(event)

_fake_inotify_module      = types.ModuleType("inotify_simple")
_fake_inotify_module.INotify = FakeINotify
_fake_inotify_module.Event   = FakeEvent

class _FakeFlags:
    CLOSE_WRITE = 8
    MOVED_TO    = 128

_fake_inotify_module.flags = _FakeFlags()
sys.modules["inotify_simple"] = _fake_inotify_module
import inotify_simple

# ── signal_kernel.contracts ───────────────────────────────────────────────────
# The health dataclasses are defined locally in store_watchdog.py and shadow
# the import.  We only need the import to not raise.

_fake_contracts = types.ModuleType("signal_kernel.contracts")

@dataclass(frozen=True)
class _FakeWatchdogHandlerHealth:
    qualified_name: str; path: str; is_circuit_open: bool
    total_calls: int; total_failures: int; total_timeouts: int
    consecutive_failures: int; last_call_at_iso: Optional[str]
    last_latency_ms: Optional[float]; min_latency_ms: Optional[float]
    max_latency_ms: Optional[float]; avg_latency_ms: Optional[float]

@dataclass(frozen=True)
class _FakeWatchdogPathHealth:
    path: str; handler_count: int; active_handlers: int
    event_count: int; last_event_at_iso: Optional[str]
    handlers: Tuple[_FakeWatchdogHandlerHealth, ...]

@dataclass(frozen=True)
class _FakeWatchdogHealth:
    is_running: bool; is_healthy: bool; uptime_s: Optional[float]
    total_events_fired: int; open_circuit_count: int
    watched_paths: Tuple[_FakeWatchdogPathHealth, ...]
    generated_at_iso: str

_fake_contracts.WatchdogHandlerHealth = _FakeWatchdogHandlerHealth
_fake_contracts.WatchdogPathHealth    = _FakeWatchdogPathHealth
_fake_contracts.WatchdogHealth        = _FakeWatchdogHealth

# Use the real signal_kernel package. Earlier versions of this test replaced
# signal_kernel in sys.modules, which leaks across full-suite collection and
# hides the canonical contracts added at the bus seam.

# ── structlog ─────────────────────────────────────────────────────────────────

_fake_structlog = types.ModuleType("structlog")

class _FakeLogger:
    def bind(self, **kw):   return self
    def info(self, *a, **k):    pass
    def warning(self, *a, **k): pass
    def error(self, *a, **k):   pass
    def debug(self, *a, **k):   pass

_fake_structlog.get_logger = lambda: _FakeLogger()
sys.modules["structlog"] = _fake_structlog

# ─────────────────────────────────────────────────────────────────────────────
# NOW import the module under test
# ─────────────────────────────────────────────────────────────────────────────

import tag.store_watchdog as sw
from tag.store_watchdog import (
    StoreWatchdog,
    _HandlerState,
    _WatchedFile,
    HANDLER_CIRCUIT_OPEN_THRESHOLD,
    HANDLER_TIMEOUT_S, # noqa
    DEBOUNCE_MS,
    SHUTDOWN_DRAIN_TIMEOUT_S, # noqa
)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def make_watchdog() -> StoreWatchdog:
    """Fresh watchdog instance with a FakeINotify injected."""
    wd = StoreWatchdog()
    return wd


async def noop_handler():
    """The simplest valid handler: does nothing."""


async def failing_handler():
    raise RuntimeError("deliberate failure")


async def slow_handler():
    """Hangs forever — used to test timeout."""
    await asyncio.sleep(9999)


def run(coro):
    """Run a coroutine synchronously in a new event loop."""
    return asyncio.get_event_loop().run_until_complete(coro)


# ─────────────────────────────────────────────────────────────────────────────
# TEST CLASSES
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistration(unittest.TestCase):
    """register() — all paths."""

    def setUp(self):
        self.wd = make_watchdog()

    def test_single_handler_registered(self):
        self.wd.register("topology_router.pt", noop_handler)
        self.assertEqual(self.wd.handler_count("topology_router.pt"), 1)
        self.assertIn("topology_router.pt", self.wd.registered_paths())

    def test_multiple_handlers_same_path(self):
        async def h2(): pass
        self.wd.register("topology_router.pt", noop_handler)
        self.wd.register("topology_router.pt", h2)
        self.assertEqual(self.wd.handler_count("topology_router.pt"), 2)

    def test_multiple_paths_independent(self):
        self.wd.register("topology_router.pt", noop_handler)
        self.wd.register("structural_layer.pt", noop_handler)
        self.assertEqual(len(self.wd.registered_paths()), 2)
        self.assertEqual(self.wd.handler_count("topology_router.pt"), 1)
        self.assertEqual(self.wd.handler_count("structural_layer.pt"), 1)

    def test_debounce_full_path_key(self):
        """DEBOUNCE_MS has exact path as key → use it."""
        self.wd.register("topology_router.pt", noop_handler)
        self.assertEqual(
            self.wd._watched["topology_router.pt"].debounce_ms,
            DEBOUNCE_MS["topology_router.pt"],
        )

    def test_debounce_basename_key(self):
        """Path not in DEBOUNCE_MS but basename is."""
        # "recipe_registry.mmap" is registered by basename lookup
        self.wd.register("subdir/recipe_registry.mmap", noop_handler)
        self.assertEqual(
            self.wd._watched["subdir/recipe_registry.mmap"].debounce_ms,
            DEBOUNCE_MS["recipe_registry.mmap"],
        )

    def test_debounce_fallback_default(self):
        """Unknown path and basename → fallback 500ms."""
        self.wd.register("unknown_file.bin", noop_handler)
        self.assertEqual(self.wd._watched["unknown_file.bin"].debounce_ms, 500)

    def test_debounce_explicit_override(self):
        """Explicit debounce_ms argument wins over DEBOUNCE_MS table."""
        self.wd.register("topology_router.pt", noop_handler, debounce_ms=42)
        self.assertEqual(self.wd._watched["topology_router.pt"].debounce_ms, 42)

    def test_debounce_not_overridden_on_second_handler(self):
        """Second handler on same path does not change debounce_ms."""
        async def h2(): pass
        self.wd.register("topology_router.pt", noop_handler, debounce_ms=77)
        self.wd.register("topology_router.pt", h2, debounce_ms=999)
        # _WatchedFile already exists; debounce_ms set on first register wins
        self.assertEqual(self.wd._watched["topology_router.pt"].debounce_ms, 77)

    def test_post_start_registration_raises_runtime_error(self):
        self.wd._running = True
        with self.assertRaises(RuntimeError) as ctx:
            self.wd.register("topology_router.pt", noop_handler)
        self.assertIn("WATCHDOG_POST_START_REGISTRATION", str(ctx.exception))

    def test_duplicate_handler_raises_value_error(self):
        self.wd.register("topology_router.pt", noop_handler)
        with self.assertRaises(ValueError) as ctx:
            self.wd.register("topology_router.pt", noop_handler)
        self.assertIn("WATCHDOG_DUPLICATE_REGISTRATION", str(ctx.exception))

    def test_same_handler_different_paths_allowed(self):
        """Same callable on two different paths is legitimate."""
        self.wd.register("topology_router.pt", noop_handler)
        self.wd.register("structural_layer.pt", noop_handler)   # must not raise
        self.assertEqual(self.wd.handler_count("structural_layer.pt"), 1)

    def test_different_handlers_same_path_allowed(self):
        async def h2(): pass
        self.wd.register("topology_router.pt", noop_handler)
        self.wd.register("topology_router.pt", h2)              # must not raise
        self.assertEqual(self.wd.handler_count("topology_router.pt"), 2)


class TestIntrospection(unittest.TestCase):
    """registered_paths() and handler_count()."""

    def setUp(self):
        self.wd = make_watchdog()

    def test_registered_paths_empty_on_new_instance(self):
        self.assertEqual(self.wd.registered_paths(), [])

    def test_registered_paths_preserves_insertion_order(self):
        self.wd.register("phase_states.mmap", noop_handler)
        self.wd.register("topology_router.pt", noop_handler)
        self.wd.register("structural_layer.pt", noop_handler)
        self.assertEqual(
            self.wd.registered_paths(),
            ["phase_states.mmap", "topology_router.pt", "structural_layer.pt"],
        )

    def test_handler_count_unknown_path_returns_zero(self):
        self.assertEqual(self.wd.handler_count("nonexistent.pt"), 0)

    def test_handler_count_increments_per_handler(self):
        async def h2(): pass
        async def h3(): pass
        self.wd.register("topology_router.pt", noop_handler)
        self.wd.register("topology_router.pt", h2)
        self.wd.register("topology_router.pt", h3)
        self.assertEqual(self.wd.handler_count("topology_router.pt"), 3)


class TestStart(unittest.IsolatedAsyncioTestCase):
    """start() — inotify setup, idempotency, baseline stat capture."""

    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp()
        self.store_root_patch = patch.object(sw, "STORE_ROOT", Path(self.tmp))
        self.store_root_patch.start()

    async def asyncTearDown(self):
        self.store_root_patch.stop()

    async def _make_started(self, path="topology_router.pt", handler=None):
        handler = handler or noop_handler
        wd = make_watchdog()
        wd.register(path, handler)
        # Inject a controllable FakeINotify
        fake = FakeINotify()
        wd._inotify = cast(Optional[inotify_simple.INotify], fake)  # pre-inject so start() uses it
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        return wd, fake

    async def test_start_sets_running_true(self):
        wd, _ = await self._make_started()
        self.assertTrue(wd._running)
        await wd.stop()

    async def test_start_sets_started_at(self):
        wd, _ = await self._make_started()
        self.assertIsNotNone(wd._started_at)
        await wd.stop()

    async def test_start_launches_watch_task(self):
        wd, _ = await self._make_started()
        self.assertIsNotNone(wd._watch_task)
        self.assertFalse(wd._watch_task.done())
        await wd.stop()

    async def test_start_idempotent_second_call_does_nothing(self):
        wd, fake = await self._make_started()
        initial_add_watch_count = len(fake.add_watch_calls)
        await wd.start()  # second call — must not re-arm watches
        self.assertEqual(len(fake.add_watch_calls), initial_add_watch_count)
        await wd.stop()

    async def test_start_calls_add_watch_for_parent_dir(self):
        wd, fake = await self._make_started("topology_router.pt")
        self.assertEqual(len(fake.add_watch_calls), 1)
        watched_dir, mask = fake.add_watch_calls[0]
        self.assertEqual(Path(watched_dir), Path(self.tmp))
        await wd.stop()

    async def test_start_watches_each_parent_dir_once_for_sibling_files(self):
        """Two files in the same dir → one add_watch call."""
        wd = make_watchdog()
        async def h2(): pass
        wd.register("topology_router.pt", noop_handler)
        wd.register("structural_layer.pt", h2)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        # Both paths resolve to STORE_ROOT (the tmp dir) as their parent
        self.assertEqual(len(fake.add_watch_calls), 1)
        await wd.stop()

    async def test_start_captures_baseline_stat_for_existing_file(self):
        path = "topology_router.pt"
        full = Path(self.tmp) / path
        full.write_bytes(b"weights")
        wd = make_watchdog()
        wd.register(path, noop_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        self.assertIsNotNone(wd._watched[path].last_stat)
        await wd.stop()

    async def test_start_no_baseline_for_missing_file(self):
        """File doesn't exist at start time → last_stat stays None."""
        path = "not_yet_created.pt"
        wd = make_watchdog()
        wd.register(path, noop_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        self.assertIsNone(wd._watched[path].last_stat)
        await wd.stop()

    async def test_start_inotify_init_failure_raises_os_error(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        with patch("inotify_simple.INotify", side_effect=OSError(24, "Too many open files")):
            with self.assertRaises(OSError) as ctx:
                await wd.start()
        self.assertIn("WATCHDOG_STARTUP_FAILED", str(ctx.exception))
        self.assertFalse(wd._running)

    async def test_start_add_watch_failure_raises_os_error(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        bad_fake = FakeINotify()
        bad_fake.add_watch = MagicMock(side_effect=OSError(28, "No space left"))
        with patch("inotify_simple.INotify", return_value=bad_fake):
            with self.assertRaises(OSError):
                await wd.start()


class TestStop(unittest.IsolatedAsyncioTestCase):
    """stop() — drain, cancel, fd close, idempotency."""

    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patch = patch.object(sw, "STORE_ROOT", Path(self.tmp))
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()

    async def _started_watchdog(self, path="topology_router.pt"):
        wd = make_watchdog()
        wd.register(path, noop_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        return wd, fake

    async def test_stop_before_start_is_noop(self):
        wd = make_watchdog()
        await wd.stop()          # must not raise
        self.assertFalse(wd._running)

    async def test_stop_clears_running_flag(self):
        wd, _ = await self._started_watchdog()
        await wd.stop()
        self.assertFalse(wd._running)

    async def test_stop_closes_inotify_fd(self):
        wd, fake = await self._started_watchdog()
        await wd.stop()
        self.assertTrue(fake.close_called)

    async def test_stop_cancels_pending_debounce_task(self):
        wd, _ = await self._started_watchdog()
        # Manually inject a pending debounce task
        watched = wd._watched["topology_router.pt"]
        long_task = asyncio.create_task(asyncio.sleep(999))
        watched.pending_task = long_task
        await wd.stop()
        self.assertTrue(long_task.cancelled() or long_task.done())

    async def test_stop_idempotent_double_call(self):
        wd, fake = await self._started_watchdog()
        await wd.stop()
        await wd.stop()          # must not raise
        self.assertFalse(wd._running)

    async def test_stop_cancels_watch_loop_task(self):
        wd, _ = await self._started_watchdog()
        task = wd._watch_task
        await wd.stop()
        self.assertTrue(task.done())

    async def test_stop_drains_active_handler_tasks(self):
        """Active handler tasks that complete within drain window are awaited cleanly."""
        completed = []

        async def quick_handler():
            await asyncio.sleep(0.01)
            completed.append(True)

        wd = make_watchdog()
        wd.register("topology_router.pt", quick_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()

        # Manually create a handler task and track it
        hs = wd._watched["topology_router.pt"].handler_states[0]
        task = asyncio.create_task(wd._safe_dispatch("topology_router.pt", hs))
        wd._active_handler_tasks.add(task)
        task.add_done_callback(wd._active_handler_tasks.discard)

        await wd.stop()
        self.assertTrue(completed)

    async def test_stop_force_cancels_handlers_that_exceed_drain_timeout(self):
        """Handlers that don't finish within SHUTDOWN_DRAIN_TIMEOUT_S are cancelled."""
        wd = make_watchdog()
        wd.register("topology_router.pt", slow_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()

        hs = wd._watched["topology_router.pt"].handler_states[0]
        task = asyncio.create_task(wd._safe_dispatch("topology_router.pt", hs))
        wd._active_handler_tasks.add(task)
        task.add_done_callback(wd._active_handler_tasks.discard)

        # Patch drain timeout short; also patch handler timeout large so
        # wait_for does not race the drain window.
        with patch.object(sw, "SHUTDOWN_DRAIN_TIMEOUT_S", 0.05):
            with patch.object(sw, "HANDLER_TIMEOUT_S", 999.0):
                await wd.stop()

        # cancel() is not awaited in stop() — give the loop a tick to process it.
        await asyncio.sleep(0.05)
        self.assertTrue(task.done())


class TestStatChanged(unittest.TestCase):
    """_stat_changed() — all branching paths."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.wd  = make_watchdog()

    def _watched(self, last_stat=None) -> _WatchedFile:
        return _WatchedFile(path="test.pt", debounce_ms=500, last_stat=last_stat)

    def test_no_baseline_returns_true(self):
        full = Path(self.tmp) / "f.pt"
        full.write_bytes(b"data")
        watched = self._watched(last_stat=None)
        self.assertTrue(self.wd._stat_changed(full, watched))

    def test_unchanged_stat_returns_false(self):
        full = Path(self.tmp) / "f.pt"
        full.write_bytes(b"data")
        st = full.stat()
        watched = self._watched(last_stat=(st.st_size, st.st_mtime))
        self.assertFalse(self.wd._stat_changed(full, watched))

    def test_size_changed_returns_true(self):
        full = Path(self.tmp) / "f.pt"
        full.write_bytes(b"original")
        st = full.stat()
        watched = self._watched(last_stat=(st.st_size - 1, st.st_mtime))
        self.assertTrue(self.wd._stat_changed(full, watched))

    def test_mtime_changed_returns_true(self):
        full = Path(self.tmp) / "f.pt"
        full.write_bytes(b"data")
        st = full.stat()
        watched = self._watched(last_stat=(st.st_size, st.st_mtime - 1.0))
        self.assertTrue(self.wd._stat_changed(full, watched))

    def test_os_error_returns_true_and_clears_baseline(self):
        """File vanished (renamed away) — treat as changed, clear baseline."""
        full = Path(self.tmp) / "gone.pt"  # does not exist
        watched = self._watched(last_stat=(100, 1234567.0))
        result  = self.wd._stat_changed(full, watched)
        self.assertTrue(result)
        self.assertIsNone(watched.last_stat)

    def test_updates_baseline_after_detecting_change(self):
        full = Path(self.tmp) / "f.pt"
        full.write_bytes(b"v1")
        st = full.stat()
        # baseline has wrong size
        watched = self._watched(last_stat=(st.st_size - 1, st.st_mtime))
        self.wd._stat_changed(full, watched)
        # baseline must now reflect actual current stat
        self.assertEqual(watched.last_stat, (st.st_size, st.st_mtime))


class TestHandleRawEvent(unittest.IsolatedAsyncioTestCase):
    """_handle_raw_event() — routing and ghost-event filter."""

    async def asyncSetUp(self):
        self.tmp = tempfile.mkdtemp()
        self.patch = patch.object(sw, "STORE_ROOT", Path(self.tmp))
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()

    def _fake_event(self, wd: int, name: str) -> FakeEvent:
        return FakeEvent(wd=wd, mask=128, cookie=0, name=name)

    def _wire_watchdog(self, path: str) -> tuple:
        """Return (watchdog, wd_int) with the path pre-registered and wd_to_dir set."""
        watchdog = make_watchdog()
        watchdog.register(path, noop_handler)
        parent = (Path(self.tmp) / path).parent
        watchdog._wd_to_dir[1] = parent
        return watchdog, 1

    async def test_unknown_wd_is_ignored(self):
        watchdog, _ = self._wire_watchdog("topology_router.pt")
        event = self._fake_event(wd=999, name="topology_router.pt")
        called = []
        with patch.object(watchdog, "_schedule_handlers",
                          side_effect=lambda p: called.append(p)):
            await watchdog._handle_raw_event(event)
        self.assertEqual(called, [])

    async def test_wrong_filename_is_ignored(self):
        watchdog, wd_int = self._wire_watchdog("topology_router.pt")
        event = self._fake_event(wd=wd_int, name="other_file.pt")
        called = []
        with patch.object(watchdog, "_schedule_handlers",
                          AsyncMock(side_effect=lambda p: called.append(p))):
            await watchdog._handle_raw_event(event)
        self.assertEqual(called, [])

    async def test_ghost_event_is_suppressed(self):
        """Same stat as baseline → _schedule_handlers not called."""
        path = "topology_router.pt"
        full = Path(self.tmp) / path
        full.write_bytes(b"data")
        st = full.stat()
        watchdog, wd_int = self._wire_watchdog(path)
        # Pre-load baseline to match current stat → ghost event
        watchdog._watched[path].last_stat = (st.st_size, st.st_mtime)

        event  = self._fake_event(wd=wd_int, name="topology_router.pt")
        called = []
        with patch.object(watchdog, "_schedule_handlers",
                          AsyncMock(side_effect=lambda p: called.append(p))):
            await watchdog._handle_raw_event(event)
        self.assertEqual(called, [])

    async def test_real_change_calls_schedule_handlers(self):
        """File content changed → _schedule_handlers called with correct path."""
        path = "topology_router.pt"
        full = Path(self.tmp) / path
        full.write_bytes(b"new weights")
        # No baseline set → stat_changed returns True
        watchdog, wd_int = self._wire_watchdog(path)
        watchdog._watched[path].last_stat = None

        event  = self._fake_event(wd=wd_int, name="topology_router.pt")
        called = []
        mock_schedule = AsyncMock(side_effect=lambda p: called.append(p))
        with patch.object(watchdog, "_schedule_handlers", mock_schedule):
            await watchdog._handle_raw_event(event)
        self.assertEqual(called, [path])


class TestScheduleHandlers(unittest.IsolatedAsyncioTestCase):
    """_schedule_handlers() — task creation and debounce cancellation."""

    async def test_creates_pending_task(self):
        wd = make_watchdog()
        wd.register("phase_states.mmap", noop_handler)
        await wd._schedule_handlers("phase_states.mmap")
        task = wd._watched["phase_states.mmap"].pending_task
        self.assertIsNotNone(task)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    async def test_cancels_existing_pending_task_before_creating_new(self):
        wd = make_watchdog()
        wd.register("phase_states.mmap", noop_handler)

        # Schedule first time
        await wd._schedule_handlers("phase_states.mmap")
        first_task = wd._watched["phase_states.mmap"].pending_task
        self.assertIsNotNone(first_task)

        # Schedule again — first task must be cancelled
        await wd._schedule_handlers("phase_states.mmap")
        second_task = wd._watched["phase_states.mmap"].pending_task

        self.assertIsNot(first_task, second_task)
        # Give the event loop a tick to propagate the cancellation
        await asyncio.sleep(0)
        self.assertTrue(first_task.cancelled() or first_task.done())
        second_task.cancel()
        try:
            await second_task
        except asyncio.CancelledError:
            pass


class TestDebouncedDispatch(unittest.IsolatedAsyncioTestCase):
    """_debounced_dispatch() — sleep, event counters, circuit-open skip."""

    async def test_fires_after_debounce_sleep(self):
        called = []

        async def tracking_handler():
            called.append(True)

        wd = make_watchdog()
        wd.register("phase_states.mmap", tracking_handler)
        await wd._debounced_dispatch("phase_states.mmap", 0.0)
        await asyncio.sleep(0.02)   # let background task land
        self.assertTrue(called)

    async def test_increments_event_count(self):
        wd = make_watchdog()
        wd.register("phase_states.mmap", noop_handler)
        await wd._debounced_dispatch("phase_states.mmap", 0.0)
        self.assertEqual(wd._watched["phase_states.mmap"].event_count, 1)

    async def test_increments_total_events(self):
        wd = make_watchdog()
        wd.register("phase_states.mmap", noop_handler)
        await wd._debounced_dispatch("phase_states.mmap", 0.0)
        self.assertEqual(wd._total_events, 1)

    async def test_circuit_open_handler_skipped(self):
        called = []

        async def tracking_handler():
            called.append(True)

        wd = make_watchdog()
        wd.register("phase_states.mmap", tracking_handler)
        wd._watched["phase_states.mmap"].handler_states[0].circuit_open = True

        await wd._debounced_dispatch("phase_states.mmap", 0.0)
        await asyncio.sleep(0.02)
        self.assertEqual(called, [])

    async def test_active_handler_dispatched_when_sibling_circuit_open(self):
        active_calls  = []
        circuit_calls = []

        async def active_h():    active_calls.append(True)
        async def circuit_h():   circuit_calls.append(True)

        wd = make_watchdog()
        wd.register("phase_states.mmap", active_h)
        wd.register("phase_states.mmap", circuit_h)
        wd._watched["phase_states.mmap"].handler_states[1].circuit_open = True

        await wd._debounced_dispatch("phase_states.mmap", 0.0)
        await asyncio.sleep(0.05)

        self.assertTrue(active_calls)
        self.assertEqual(circuit_calls, [])

    async def test_cancelled_before_sleep_does_not_dispatch(self):
        called = []

        async def tracking_handler():
            called.append(True)

        wd = make_watchdog()
        wd.register("topology_router.pt", tracking_handler, debounce_ms=500)

        # Schedule but immediately cancel — simulates rapid re-event
        task = asyncio.create_task(
            wd._debounced_dispatch("topology_router.pt", 99.0)
        )
        await asyncio.sleep(0)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        await asyncio.sleep(0.05)
        self.assertEqual(called, [])


class TestSafeDispatch(unittest.IsolatedAsyncioTestCase):
    """_safe_dispatch() — every exit path, counter accuracy, latency tracking."""

    def _hs(self, handler) -> _HandlerState:
        return _HandlerState(handler=handler)

    async def test_success_increments_total_calls(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.total_calls, 1)

    async def test_success_resets_consecutive_failures(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        hs.consecutive_failures = 3
        await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.consecutive_failures, 0)

    async def test_success_does_not_increment_failure_counters(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.total_failures, 0)
        self.assertEqual(hs.total_timeouts, 0)

    async def test_success_records_latency(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertIsNotNone(hs.last_latency_ms)
        self.assertGreaterEqual(hs.last_latency_ms, 0.0)

    async def test_success_sets_min_and_max_latency_on_first_call(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertIsNotNone(hs.min_latency_ms)
        self.assertIsNotNone(hs.max_latency_ms)
        self.assertAlmostEqual(hs.min_latency_ms, hs.max_latency_ms, places=1)

    async def test_success_max_tracks_slowest_call(self):
        delays = [0.0, 0.05, 0.01]

        async def variable_handler():
            await asyncio.sleep(delays.pop(0))

        wd = make_watchdog()
        hs = self._hs(variable_handler)
        for _ in range(3):
            await wd._safe_dispatch("test.pt", hs)

        self.assertGreater(hs.max_latency_ms, hs.min_latency_ms)

    async def test_success_sum_latency_accumulates(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        await wd._safe_dispatch("test.pt", hs)
        await wd._safe_dispatch("test.pt", hs)
        self.assertGreater(hs.sum_latency_ms, 0.0)

    async def test_success_sets_last_call_at(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertIsNotNone(hs.last_call_at)

    async def test_exception_increments_total_failures(self):
        wd = make_watchdog()
        hs = self._hs(failing_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.total_failures, 1)

    async def test_exception_increments_consecutive_failures(self):
        wd = make_watchdog()
        hs = self._hs(failing_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.consecutive_failures, 1)

    async def test_exception_records_latency(self):
        wd = make_watchdog()
        hs = self._hs(failing_handler)
        await wd._safe_dispatch("test.pt", hs)
        self.assertIsNotNone(hs.last_latency_ms)

    async def test_exception_does_not_reraise(self):
        """_safe_dispatch swallows non-CancelledError exceptions."""
        wd = make_watchdog()
        hs = self._hs(failing_handler)
        try:
            await wd._safe_dispatch("test.pt", hs)
        except Exception as e:
            self.fail(f"_safe_dispatch unexpectedly raised: {e}")

    async def test_timeout_increments_total_failures_and_timeouts(self):
        wd = make_watchdog()
        hs = self._hs(slow_handler)
        with patch.object(sw, "HANDLER_TIMEOUT_S", 0.05):
            await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.total_failures, 1)
        self.assertEqual(hs.total_timeouts, 1)

    async def test_timeout_increments_consecutive_failures(self):
        wd = make_watchdog()
        hs = self._hs(slow_handler)
        with patch.object(sw, "HANDLER_TIMEOUT_S", 0.05):
            await wd._safe_dispatch("test.pt", hs)
        self.assertEqual(hs.consecutive_failures, 1)

    async def test_timeout_records_latency(self):
        wd = make_watchdog()
        hs = self._hs(slow_handler)
        with patch.object(sw, "HANDLER_TIMEOUT_S", 0.05):
            await wd._safe_dispatch("test.pt", hs)
        self.assertIsNotNone(hs.last_latency_ms)

    async def test_cancelled_error_reraises(self):
        """CancelledError must propagate — it is not a failure."""
        async def cancel_self():
            raise asyncio.CancelledError()

        wd = make_watchdog()
        hs = self._hs(cancel_self)
        with self.assertRaises(asyncio.CancelledError):
            await wd._safe_dispatch("test.pt", hs)

    async def test_cancelled_error_does_not_increment_failure_counters(self):
        async def cancel_self():
            raise asyncio.CancelledError()

        wd = make_watchdog()
        hs = self._hs(cancel_self)
        try:
            await wd._safe_dispatch("test.pt", hs)
        except asyncio.CancelledError:
            pass
        self.assertEqual(hs.total_failures, 0)
        self.assertEqual(hs.consecutive_failures, 0)


class TestCircuitBreaker(unittest.IsolatedAsyncioTestCase):
    """Circuit breaker — threshold, isolation, reset."""

    def _hs(self, handler) -> _HandlerState:
        return _HandlerState(handler=handler)

    async def test_circuit_stays_closed_below_threshold(self):
        wd = make_watchdog()
        hs = self._hs(failing_handler)
        for _ in range(HANDLER_CIRCUIT_OPEN_THRESHOLD - 1):
            await wd._safe_dispatch("test.pt", hs)
        self.assertFalse(hs.circuit_open)

    async def test_circuit_opens_exactly_at_threshold(self):
        wd = make_watchdog()
        hs = self._hs(failing_handler)
        for _ in range(HANDLER_CIRCUIT_OPEN_THRESHOLD):
            await wd._safe_dispatch("test.pt", hs)
        self.assertTrue(hs.circuit_open)

    async def test_circuit_open_handler_not_dispatched(self):
        """Once circuit opens, _debounced_dispatch skips the handler."""
        called = []

        async def tracking_h():
            called.append(True)

        wd = make_watchdog()
        wd.register("topology_router.pt", tracking_h)
        wd._watched["topology_router.pt"].handler_states[0].circuit_open = True

        await wd._debounced_dispatch("topology_router.pt", 0.0)
        await asyncio.sleep(0.05)
        self.assertEqual(called, [])

    async def test_reset_circuit_re_arms_handler(self):
        called = []

        async def tracking_h():
            called.append(True)

        wd = make_watchdog()
        # Register failing_handler to trip the circuit, then swap in tracking_h.
        wd.register("topology_router.pt", failing_handler)
        hs = wd._watched["topology_router.pt"].handler_states[0]

        # Trip the circuit using the failing handler
        for _ in range(HANDLER_CIRCUIT_OPEN_THRESHOLD):
            await wd._safe_dispatch("topology_router.pt", hs)
        self.assertTrue(hs.circuit_open)

        # Swap handler to tracking_h so we can verify re-arm works
        hs.handler = tracking_h

        # Reset it
        reset = wd.reset_circuit("topology_router.pt", tracking_h.__qualname__)
        self.assertTrue(reset)
        self.assertFalse(hs.circuit_open)
        self.assertEqual(hs.consecutive_failures, 0)

        # Now dispatch should call the handler again
        await wd._debounced_dispatch("topology_router.pt", 0.0)
        await asyncio.sleep(0.05)
        self.assertTrue(called)

    async def test_reset_circuit_unknown_path_returns_false(self):
        wd = make_watchdog()
        result = wd.reset_circuit("no_such.pt", "some.qualname")
        self.assertFalse(result)

    async def test_reset_circuit_unknown_handler_qualname_returns_false(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        result = wd.reset_circuit("topology_router.pt", "NonExistent.handler")
        self.assertFalse(result)

    async def test_circuit_reopens_after_reset_and_failures(self):
        wd = make_watchdog()
        hs = self._hs(failing_handler)

        # Trip
        for _ in range(HANDLER_CIRCUIT_OPEN_THRESHOLD):
            await wd._safe_dispatch("test.pt", hs)
        self.assertTrue(hs.circuit_open)

        # Reset manually
        hs.circuit_open         = False
        hs.consecutive_failures = 0

        # Fail again until threshold
        for _ in range(HANDLER_CIRCUIT_OPEN_THRESHOLD):
            await wd._safe_dispatch("test.pt", hs)
        self.assertTrue(hs.circuit_open)

    async def test_successful_call_after_failures_resets_consecutive_count(self):
        """Alternating fail/pass — consecutive counter must reset on success."""
        n = [0]

        async def flaky():
            n[0] += 1
            if n[0] % 2 == 1:
                raise RuntimeError("odd call fails")

        wd = make_watchdog()
        hs = self._hs(flaky)

        await wd._safe_dispatch("test.pt", hs)  # call 1: fails → consecutive=1
        self.assertEqual(hs.consecutive_failures, 1)
        await wd._safe_dispatch("test.pt", hs)  # call 2: succeeds → consecutive=0
        self.assertEqual(hs.consecutive_failures, 0)
        await wd._safe_dispatch("test.pt", hs)  # call 3: fails → consecutive=1
        self.assertEqual(hs.consecutive_failures, 1)
        self.assertFalse(hs.circuit_open)


class TestHandlerIsolation(unittest.IsolatedAsyncioTestCase):
    """One handler's failure must not affect sibling handlers."""

    async def test_failing_handler_does_not_prevent_sibling_from_running(self):
        success_calls = []

        async def good_handler():
            success_calls.append(True)

        wd = make_watchdog()
        wd.register("structural_layer.pt", failing_handler)
        wd.register("structural_layer.pt", good_handler)

        await wd._debounced_dispatch("structural_layer.pt", 0.0)
        await asyncio.sleep(0.05)

        self.assertTrue(success_calls)

    async def test_timeout_handler_does_not_block_sibling(self):
        success_calls = []

        async def good_handler():
            success_calls.append(True)

        wd = make_watchdog()
        wd.register("structural_layer.pt", slow_handler)
        wd.register("structural_layer.pt", good_handler)

        with patch.object(sw, "HANDLER_TIMEOUT_S", 0.05):
            await wd._debounced_dispatch("structural_layer.pt", 0.0)
            await asyncio.sleep(0.2)

        self.assertTrue(success_calls)

    async def test_circuit_opened_on_one_handler_leaves_other_active(self):
        wd = make_watchdog()
        wd.register("structural_layer.pt", failing_handler)

        async def good_h(): pass
        wd.register("structural_layer.pt", good_h)

        # Trip only the first handler's circuit
        hs = wd._watched["structural_layer.pt"].handler_states[0]
        for _ in range(HANDLER_CIRCUIT_OPEN_THRESHOLD):
            await wd._safe_dispatch("structural_layer.pt", hs)

        self.assertTrue(hs.circuit_open)
        self.assertFalse(
            wd._watched["structural_layer.pt"].handler_states[1].circuit_open
        )


class TestHealth(unittest.IsolatedAsyncioTestCase):
    """health() — all fields, before/after start, circuit states, latency."""

    async def asyncSetUp(self):
        self.tmp   = tempfile.mkdtemp()
        self.patch = patch.object(sw, "STORE_ROOT", Path(self.tmp))
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()

    def test_health_before_start_is_not_running(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        h  = wd.health()
        self.assertFalse(h.is_running)
        self.assertFalse(h.is_healthy)

    def test_health_uptime_none_before_start(self):
        wd = make_watchdog()
        self.assertIsNone(wd.health().uptime_s)

    async def test_health_after_start_is_running_and_healthy(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        h = wd.health()
        self.assertTrue(h.is_running)
        self.assertTrue(h.is_healthy)
        self.assertEqual(h.open_circuit_count, 0)
        await wd.stop()

    async def test_health_uptime_positive_after_start(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        fake = FakeINotify()
        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()
        await asyncio.sleep(0.01)
        h = wd.health()
        self.assertIsNotNone(h.uptime_s)
        self.assertGreater(h.uptime_s, 0.0)
        await wd.stop()

    def test_health_not_healthy_with_open_circuit(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        wd._running = True
        wd._started_at = 0.0
        wd._watched["topology_router.pt"].handler_states[0].circuit_open = True
        h = wd.health()
        self.assertFalse(h.is_healthy)
        self.assertEqual(h.open_circuit_count, 1)

    def test_health_open_circuit_count_across_paths(self):
        async def h2(): pass
        async def h3(): pass

        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        wd.register("structural_layer.pt", h2)
        wd.register("structural_layer.pt", h3)
        wd._running    = True
        wd._started_at = 0.0

        wd._watched["topology_router.pt"].handler_states[0].circuit_open = True
        wd._watched["structural_layer.pt"].handler_states[1].circuit_open = True

        h = wd.health()
        self.assertEqual(h.open_circuit_count, 2)

    def test_health_total_events_fired(self):
        wd = make_watchdog()
        wd._total_events = 17
        self.assertEqual(wd.health().total_events_fired, 17)

    def test_health_path_event_count(self):
        wd = make_watchdog()
        wd.register("phase_states.mmap", noop_handler)
        wd._watched["phase_states.mmap"].event_count = 5
        ph = wd.health().watched_paths[0]
        self.assertEqual(ph.event_count, 5)

    def test_health_active_handlers_count(self):
        async def h2(): pass
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        wd.register("topology_router.pt", h2)
        wd._watched["topology_router.pt"].handler_states[0].circuit_open = True
        ph = wd.health().watched_paths[0]
        self.assertEqual(ph.active_handlers, 1)
        self.assertEqual(ph.handler_count, 2)

    async def test_health_latency_stats_after_calls(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        hs = wd._watched["topology_router.pt"].handler_states[0]

        # Run three dispatches
        for _ in range(3):
            await wd._safe_dispatch("topology_router.pt", hs)

        h  = wd.health()
        hh = h.watched_paths[0].handlers[0]

        self.assertEqual(hh.total_calls, 3)
        self.assertIsNotNone(hh.last_latency_ms)
        self.assertIsNotNone(hh.min_latency_ms)
        self.assertIsNotNone(hh.max_latency_ms)
        self.assertIsNotNone(hh.avg_latency_ms)
        self.assertLessEqual(hh.min_latency_ms, hh.max_latency_ms)
        self.assertGreater(hh.avg_latency_ms, 0.0)

    def test_health_generated_at_iso_is_set(self):
        wd = make_watchdog()
        h  = wd.health()
        self.assertIsNotNone(h.generated_at_iso)
        self.assertIn("T", h.generated_at_iso)  # ISO 8601 contains 'T'

    def test_health_handler_qualified_name(self):
        wd = make_watchdog()
        wd.register("topology_router.pt", noop_handler)
        hh = wd.health().watched_paths[0].handlers[0]
        self.assertEqual(hh.qualified_name, noop_handler.__qualname__)


class TestRapidEvents(unittest.IsolatedAsyncioTestCase):
    """Debounce must collapse rapid successive events into one dispatch."""

    async def test_rapid_events_produce_single_dispatch(self):
        dispatch_count = []

        async def counting_handler():
            dispatch_count.append(True)

        wd = make_watchdog()
        wd.register("topology_router.pt", counting_handler, debounce_ms=50)

        # Simulate 5 rapid events
        for _ in range(5):
            await wd._schedule_handlers("topology_router.pt")
            await asyncio.sleep(0.005)   # 5ms apart — within 50ms debounce window

        # Wait for debounce to fire and handler to run
        await asyncio.sleep(0.2)

        self.assertEqual(len(dispatch_count), 1,
                         f"Expected 1 dispatch, got {len(dispatch_count)}")

    async def test_two_widely_spaced_events_produce_two_dispatches(self):
        dispatch_count = []

        async def counting_handler():
            dispatch_count.append(True)

        wd = make_watchdog()
        wd.register("topology_router.pt", counting_handler, debounce_ms=20)

        # First event
        await wd._schedule_handlers("topology_router.pt")
        await asyncio.sleep(0.1)    # well past debounce window

        # Second event
        await wd._schedule_handlers("topology_router.pt")
        await asyncio.sleep(0.1)

        self.assertEqual(len(dispatch_count), 2)


class TestFullLifecycle(unittest.IsolatedAsyncioTestCase):
    """End-to-end: register → start → inject inotify event → handler called → stop."""

    async def asyncSetUp(self):
        self.tmp   = tempfile.mkdtemp()
        self.patch = patch.object(sw, "STORE_ROOT", Path(self.tmp))
        self.patch.start()

    async def asyncTearDown(self):
        self.patch.stop()

    async def test_full_lifecycle_handler_called_on_file_change(self):
        called = []

        async def reload_handler():
            called.append(True)

        # Create the file so baseline stat is captured
        file_path = Path(self.tmp) / "topology_router.pt"
        file_path.write_bytes(b"v1")

        wd   = make_watchdog()
        fake = FakeINotify()
        wd.register("topology_router.pt", reload_handler)

        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()

        # Simulate atomic rename: update the file, then push an inotify event
        file_path.write_bytes(b"v2 with updated weights")
        wd_int  = list(wd._wd_to_dir.keys())[0]
        event   = FakeEvent(wd=wd_int, mask=128, cookie=0, name="topology_router.pt")
        fake.push_event(event)

        # Manually drive one watch loop iteration
        await wd._handle_raw_event(event)

        # Wait for debounce + handler
        await asyncio.sleep(0.8)

        self.assertTrue(called, "reload_handler was never called")
        await wd.stop()

    async def test_full_lifecycle_multiple_paths_independent(self):
        a_calls = []
        b_calls = []

        async def handler_a(): a_calls.append(True)
        async def handler_b(): b_calls.append(True)

        path_a = Path(self.tmp) / "topology_router.pt"
        path_b = Path(self.tmp) / "structural_layer.pt"
        path_a.write_bytes(b"a"); path_b.write_bytes(b"b")

        wd   = make_watchdog()
        fake = FakeINotify()
        wd.register("topology_router.pt", handler_a)
        wd.register("structural_layer.pt", handler_b)

        with patch("inotify_simple.INotify", return_value=fake):
            await wd.start()

        wd_int = list(wd._wd_to_dir.keys())[0]

        # Only fire event for path_a
        path_a.write_bytes(b"a updated")
        event = FakeEvent(wd=wd_int, mask=128, cookie=0, name="topology_router.pt")
        await wd._handle_raw_event(event)

        await asyncio.sleep(0.8)

        self.assertTrue(a_calls)
        self.assertEqual(b_calls, [], "handler_b should not have been called")
        await wd.stop()


class TestMaybeOpenCircuit(unittest.TestCase):
    """_maybe_open_circuit() — boundary conditions."""

    def _hs(self, handler) -> _HandlerState:
        return _HandlerState(handler=handler)

    def test_below_threshold_does_not_open(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        hs.consecutive_failures = HANDLER_CIRCUIT_OPEN_THRESHOLD - 1
        wd._maybe_open_circuit("test.pt", hs)
        self.assertFalse(hs.circuit_open)

    def test_at_threshold_opens(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        hs.consecutive_failures = HANDLER_CIRCUIT_OPEN_THRESHOLD
        wd._maybe_open_circuit("test.pt", hs)
        self.assertTrue(hs.circuit_open)

    def test_above_threshold_opens(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        hs.consecutive_failures = HANDLER_CIRCUIT_OPEN_THRESHOLD + 10
        wd._maybe_open_circuit("test.pt", hs)
        self.assertTrue(hs.circuit_open)

    def test_already_open_stays_open(self):
        wd = make_watchdog()
        hs = self._hs(noop_handler)
        hs.circuit_open         = True
        hs.consecutive_failures = HANDLER_CIRCUIT_OPEN_THRESHOLD
        wd._maybe_open_circuit("test.pt", hs)
        self.assertTrue(hs.circuit_open)


class TestDebounceResolution(unittest.TestCase):
    """DEBOUNCE_MS lookup order: full path → basename → 500ms fallback."""

    def setUp(self):
        self.wd = make_watchdog()

    def test_trigger_path_full_key(self):
        self.wd.register("triggers/preparse", noop_handler)
        self.assertEqual(
            self.wd._watched["triggers/preparse"].debounce_ms,
            DEBOUNCE_MS["triggers/preparse"],
        )

    def test_mmap_basename_fallthrough(self):
        # Full path "subdir/phase_states.mmap" not in DEBOUNCE_MS,
        # but basename "phase_states.mmap" is.
        self.wd.register("subdir/phase_states.mmap", noop_handler)
        self.assertEqual(
            self.wd._watched["subdir/phase_states.mmap"].debounce_ms,
            DEBOUNCE_MS["phase_states.mmap"],
        )

    def test_unknown_name_defaults_to_500(self):
        self.wd.register("mystery_file.xyz", noop_handler)
        self.assertEqual(self.wd._watched["mystery_file.xyz"].debounce_ms, 500)

    def test_explicit_zero_debounce_respected(self):
        self.wd.register("topology_router.pt", noop_handler, debounce_ms=0)
        self.assertEqual(self.wd._watched["topology_router.pt"].debounce_ms, 0)


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
