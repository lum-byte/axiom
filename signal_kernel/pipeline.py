"""
signal_kernel/pipeline.py
==========================
The most complex file in the kernel.

Manages the full subprocess lifecycle for Process 1 — the stateless grep
pipeline that transforms raw HTML or JSON into clean signal text.

Call order:
    1. resolve   — registry.get_recipe(topology_class)
    2. validate  — validator.check(recipe_mount)
    3. execute   — _run_subprocess(recipe_mount, raw_content)
    4. capture   — _build_kernel_output(recipe_mount, raw_content, output)
    5. return    — KernelOutput to AXIOM graph

Error handling contract:
    pipeline.py NEVER raises to its caller except:
      - RecipeMountError        — hard stop, requires human review
      - RecipeInjectionAttempt  — hard stop, requires forensic review
    Every other exception is caught, logged, and returned as
    KernelOutput(extraction_empty=True). The AXIOM graph continues.
    Kernel failure degrades gracefully — it never stops a run.

Dependency direction:
    pipeline.py → contracts.py, exceptions.py, recipes/registry.py,
                   recipes/validator.py
    Nothing depends on pipeline.py except TAG's Python layer above it.

Process architecture:
    pipeline.py sits in TAG's Python layer. It does NOT run inside the
    Alpine container. It spawns the container (or subprocess) externally,
    pipes raw content to stdin, reads clean signal from stdout, and returns
    a KernelOutput to the caller. The container is an execution boundary
    that pipeline.py manages from outside.

    The warm container optimization keeps a previously spawned container
    alive between invocations of the same topology class and recipe hash.
    Topology class change → spawn fresh. Recipe hash change → spawn fresh.
    Fresh spawn latency on Alpine is milliseconds. The complexity of live
    recipe remount is not worth the engineering cost.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal # noqa
import time
from contextlib import contextmanager # noqa
from dataclasses import dataclass, field # noqa
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Dict,
    Generator, # noqa
    List,
    Callable,
    Optional,
    Tuple,
)

from signal_kernel.contracts import (
    DEFAULT_SPAWN_TIMEOUT_MS,
    DEFAULT_SUBPROCESS_TIMEOUT_MS,
    ContainerLifecycle,
    KernelAuditEvent,
    KernelInput,
    KernelOutput,
    AuditEventType,
    AuditSeverity,
    PipelineTelemetry,
    RecipeHash, # noqa
    RecipeMount,
    RunID, # noqa
    TopologyClassStr, # noqa
    make_empty_kernel_output,
    make_pipeline_telemetry,
    new_run_id,
)
from signal_kernel.exceptions import (
    ContainerSpawnError,
    EmptyExtractionError,
    KernelException,
    OutputMeasurementError, # noqa
    ProcessStateCorruption, # noqa
    RecipeInjectionAttempt,
    RecipeMountError,
    StdinEncodingError,
    SubprocessTimeout, # noqa
    classify,
    is_hard_stop,
    is_security_event, # noqa
)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
#
# Timing, resource limits, and operational parameters that govern the
# subprocess lifecycle. All derived from contracts.py constants where a
# canonical value exists. Pipeline-specific parameters are defined here.
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("signal_kernel.pipeline")

# ── Spawn configuration ───────────────────────────────────────────────────

# Maximum number of spawn attempts before giving up. The first attempt is
# the initial spawn. If it fails, one retry is performed. If the retry
# also fails, ContainerSpawnError is raised and pipeline.py returns empty
# KernelOutput. Two attempts total — never three.
_SPAWN_MAX_ATTEMPTS: int = 2

# Delay between spawn retry attempts in seconds. This gives the Docker
# daemon a moment to release resources from the failed spawn. Without this
# delay, the retry races against resource cleanup and frequently fails for
# the same reason as the first attempt.
_SPAWN_RETRY_DELAY_SECONDS: float = 0.25

# ── Process termination escalation ────────────────────────────────────────

# After sending SIGTERM to a subprocess, wait this long for graceful exit.
# If the process has not exited after this window, send SIGKILL. SIGTERM
# allows the shell pipeline to flush stdout. SIGKILL does not — some
# output may be lost. The escalation is a safety net against hung processes
# that trap SIGTERM (which they should not, but hostile recipes could).
_GRACEFUL_KILL_TIMEOUT_SECONDS: float = 2.0

# After SIGKILL, wait this long for the OS to reap the process. If the
# process is still alive after SIGKILL + this delay, something is deeply
# wrong at the OS level (zombie, D-state I/O, broken cgroup). Log at
# CRITICAL and abandon — do not loop.
_FORCE_KILL_REAP_TIMEOUT_SECONDS: float = 1.0

# ── Stderr handling ───────────────────────────────────────────────────────

# Maximum bytes of stderr to retain in memory for logging. stderr is
# diagnostic signal (sed/awk warnings, shell errors) not execution output.
# Truncation protects against pathological recipes that write megabytes to
# stderr. The first N bytes are diagnostic; the rest is noise.
_STDERR_RETAIN_LIMIT_BYTES: int = 8192

# Maximum bytes of stderr to include in structured log entries. The full
# retained stderr goes to debug logs. The log entry gets a preview for
# quick triage without scrolling through kilobytes of awk warnings.
_STDERR_LOG_PREVIEW_BYTES: int = 512

# ── Stdin encoding ────────────────────────────────────────────────────────

# The encoding used to convert raw_content (str) into bytes for the
# subprocess stdin pipe. Always UTF-8. The recipe shell scripts expect
# UTF-8 input. If the raw content contains characters that cannot be
# encoded as UTF-8, StdinEncodingError is raised.
_STDIN_ENCODING: str = "utf-8"

# Error handling mode for stdin encoding. 'strict' raises on any
# unencodable character. This is the correct mode — a page with
# unencodable characters should fail fast at encoding, not produce
# corrupt output from the grep pipeline.
_STDIN_ENCODING_ERRORS: str = "strict"

# ── Token estimation ──────────────────────────────────────────────────────

# Character-to-token ratio used for the token_delta_estimate field in
# KernelOutput. This is a deliberate approximation — the ratio of ~4
# characters per token holds well for English text across most tokenizers.
# This estimate is sufficient for feedback.py's quality scoring. It does
# not need to be exact. Calling a real tokenizer would be slower and
# would introduce a dependency the kernel should not have.
_CHARS_PER_TOKEN: int = 4

# ── Warm container ────────────────────────────────────────────────────────

# Maximum idle time for a warm container before it is proactively killed.
# A container sitting idle for this long is consuming resources without
# benefit. The next invocation will spawn fresh. This prevents resource
# leaks from topology classes that see infrequent traffic.
_WARM_CONTAINER_MAX_IDLE_SECONDS: float = 300.0


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE CONFIGURATION
#
# Frozen dataclass holding all tunables for one pipeline instance.
# Constructed once at startup. Passed to execute() or used as module
# default. The caller (TAG's Python layer) can override any value.
# Defaults are the production values from contracts.py.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class PipelineConfig:
    """
    Immutable configuration for the pipeline's subprocess lifecycle.

    Default values are production values. Test harnesses override them
    via construction. The config is frozen — mid-run changes are not
    possible. If you need different settings, construct a new config.
    """

    subprocess_timeout_ms: int = DEFAULT_SUBPROCESS_TIMEOUT_MS
    spawn_timeout_ms:      int = DEFAULT_SPAWN_TIMEOUT_MS
    max_spawn_attempts:    int = _SPAWN_MAX_ATTEMPTS
    spawn_retry_delay_s:   float = _SPAWN_RETRY_DELAY_SECONDS
    graceful_kill_timeout_s: float = _GRACEFUL_KILL_TIMEOUT_SECONDS
    force_kill_reap_timeout_s: float = _FORCE_KILL_REAP_TIMEOUT_SECONDS
    stderr_retain_limit:   int = _STDERR_RETAIN_LIMIT_BYTES
    warm_container_max_idle_s: float = _WARM_CONTAINER_MAX_IDLE_SECONDS
    shell_executable:      str = "/bin/sh"

    def __post_init__(self) -> None:
        if self.subprocess_timeout_ms < 100:
            raise ValueError(
                f"subprocess_timeout_ms must be ≥ 100, got {self.subprocess_timeout_ms}. "
                "Timeouts below 100ms are not meaningful for a subprocess lifecycle."
            )
        if self.spawn_timeout_ms < 100:
            raise ValueError(
                f"spawn_timeout_ms must be ≥ 100, got {self.spawn_timeout_ms}. "
                "Spawn timeouts below 100ms will false-positive on healthy systems."
            )
        if self.max_spawn_attempts < 1:
            raise ValueError(
                f"max_spawn_attempts must be ≥ 1, got {self.max_spawn_attempts}."
            )
        if self.max_spawn_attempts > 5:
            raise ValueError(
                f"max_spawn_attempts must be ≤ 5, got {self.max_spawn_attempts}. "
                "More than 5 spawn attempts indicates a systemic failure, not a transient one."
            )
        if self.spawn_retry_delay_s < 0:
            raise ValueError(
                f"spawn_retry_delay_s must be ≥ 0, got {self.spawn_retry_delay_s}."
            )
        if self.graceful_kill_timeout_s < 0:
            raise ValueError(
                f"graceful_kill_timeout_s must be ≥ 0, got {self.graceful_kill_timeout_s}."
            )
        if self.stderr_retain_limit < 0:
            raise ValueError(
                f"stderr_retain_limit must be ≥ 0, got {self.stderr_retain_limit}."
            )

    @property
    def subprocess_timeout_seconds(self) -> float:
        """Timeout in seconds for asyncio.wait_for()."""
        return self.subprocess_timeout_ms / 1000.0

    @property
    def spawn_timeout_seconds(self) -> float:
        """Spawn timeout in seconds."""
        return self.spawn_timeout_ms / 1000.0


# Module-level default configuration. Used when execute() is called without
# an explicit config. Constructed once at import time.
_DEFAULT_CONFIG: PipelineConfig = PipelineConfig()


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL DATA STRUCTURES
#
# These are pipeline.py's working state. They do not cross file boundaries.
# The AXIOM graph never sees them. They exist to organize the pipeline's
# internal lifecycle management.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _SubprocessResult:
    """
    Raw output captured from one subprocess invocation.

    This is the intermediate representation between communicate() returning
    and KernelOutput construction. It carries raw bytes, timing, and the
    exit code — not yet interpreted into clean_signal, extraction_empty,
    or any other domain concept. Interpretation happens in
    _build_kernel_output().
    """

    stdout_bytes:     bytes
    stderr_bytes:     bytes
    exit_code:        Optional[int]    # None if killed by timeout
    timed_out:        bool
    spawn_latency_ms: float            # wall time from Popen to process ready
    total_latency_ms: float            # wall time from invocation start to output available
    pid:              Optional[int]    # subprocess PID for logging

    @property
    def stdout_size(self) -> int:
        return len(self.stdout_bytes)

    @property
    def stderr_size(self) -> int:
        return len(self.stderr_bytes)

    @property
    def has_stdout(self) -> bool:
        """True if stdout contains any non-whitespace bytes."""
        return bool(self.stdout_bytes.strip())

    @property
    def has_stderr(self) -> bool:
        return len(self.stderr_bytes) > 0

    @property
    def clean_exit(self) -> bool:
        return self.exit_code == 0 and not self.timed_out


class _WarmContainer:
    """
    Tracks a previously spawned subprocess that may be reusable.

    Warm reuse is valid when:
      1. The warm container's topology class matches the new invocation
      2. The warm container's recipe hash matches the new invocation
      3. The warm container process is still alive
      4. The warm container has not been idle beyond the max idle threshold

    Topology class change requires recipe remount — spawn fresh.
    Recipe hash change means the compiler produced a new recipe — spawn fresh.
    Process death means the container crashed — spawn fresh.
    Excessive idle means the container has been consuming resources without
    benefit — kill it and spawn fresh on next invocation.

    The warm container is module-level state. It persists across invocations
    within the same Python process. On process restart, it starts empty.

    In the current subprocess model, "warm" means the process is ready to
    accept input immediately without the spawn overhead. In the Docker
    container model, "warm" means the container is running and a new exec
    can be issued without docker-create + docker-start overhead.

    IMPORTANT: The warm container optimization is a performance concern,
    not a correctness concern. If warm reuse is disabled (always spawn fresh),
    the pipeline still works correctly — just slower.
    """

    def __init__(self) -> None:
        self._topology_class: Optional[str] = None
        self._recipe_hash:    Optional[str] = None
        self._process:        Optional[asyncio.subprocess.Process] = None
        self._pid:            Optional[int] = None
        self._last_used_at:   float = 0.0    # monotonic time
        self._spawn_count:    int = 0
        self._reuse_count:    int = 0

    @property
    def is_warm(self) -> bool:
        """True if a warm container exists and its process is alive."""
        if self._process is None:
            return False
        return self._process.returncode is None

    @property
    def topology_class(self) -> Optional[str]:
        return self._topology_class

    @property
    def recipe_hash(self) -> Optional[str]:
        return self._recipe_hash

    @property
    def idle_seconds(self) -> float:
        """Seconds since last use. 0.0 if never used."""
        if self._last_used_at == 0.0:
            return 0.0
        return time.monotonic() - self._last_used_at

    @property
    def spawn_count(self) -> int:
        """Total number of spawns performed by this warm container tracker."""
        return self._spawn_count

    @property
    def reuse_count(self) -> int:
        """Total number of warm reuses (invocations that skipped spawn)."""
        return self._reuse_count

    def is_compatible(
        self,
        topology_class: str,
        recipe_hash: str,
        max_idle_s: float,
    ) -> bool:
        """
        True if the warm container can be reused for this invocation.

        Four conditions must all hold:
          1. Process is alive (returncode is None)
          2. Topology class matches
          3. Recipe hash matches
          4. Idle time is within threshold
        """
        if not self.is_warm:
            return False
        if self._topology_class != topology_class:
            return False
        if self._recipe_hash != recipe_hash:
            return False
        if self.idle_seconds > max_idle_s:
            return False
        return True

    def record_spawn(
        self,
        topology_class: str,
        recipe_hash: str,
        process: asyncio.subprocess.Process,
    ) -> None:
        """Record a new spawn. Replaces any existing warm container state."""
        self._topology_class = topology_class
        self._recipe_hash = recipe_hash
        self._process = process
        self._pid = process.pid
        self._last_used_at = time.monotonic()
        self._spawn_count += 1

    def record_reuse(self) -> None:
        """Record a warm reuse. Updates the last-used timestamp."""
        self._last_used_at = time.monotonic()
        self._reuse_count += 1

    async def kill(self) -> None:
        """
        Kill the warm container process. Best-effort.

        SIGTERM first, then SIGKILL if the process does not exit within
        the graceful window. Does not raise — a failed kill is logged but
        does not affect the pipeline. The next invocation will spawn fresh.
        """
        if self._process is None:
            return
        if self._process.returncode is not None:
            # Already exited.
            self._process = None
            return

        pid = self._pid
        try:
            self._process.terminate()
            try:
                await asyncio.wait_for(
                    self._process.wait(),
                    timeout=_GRACEFUL_KILL_TIMEOUT_SECONDS,
                )
            except asyncio.TimeoutError:
                logger.warning(
                    "Warm container pid=%s did not exit after SIGTERM, sending SIGKILL",
                    pid,
                )
                self._process.kill()
                try:
                    await asyncio.wait_for(
                        self._process.wait(),
                        timeout=_FORCE_KILL_REAP_TIMEOUT_SECONDS,
                    )
                except asyncio.TimeoutError:
                    logger.critical(
                        "Warm container pid=%s did not exit after SIGKILL. "
                        "Possible zombie or D-state process. Abandoning.",
                        pid,
                    )
        except ProcessLookupError:
            # Process already gone. Race condition — benign.
            pass
        except OSError as exc:
            logger.warning(
                "OS error killing warm container pid=%s: %s", pid, exc
            )
        finally:
            self._process = None

    def invalidate(self) -> None:
        """
        Mark the warm container as invalid without killing.

        Used when the process has already exited on its own. We just need
        to clear the state so the next invocation spawns fresh.
        """
        self._process = None
        self._topology_class = None
        self._recipe_hash = None

    def diagnostic_dict(self) -> Dict[str, object]:
        """Diagnostic snapshot for structured logging."""
        return {
            "is_warm":        self.is_warm,
            "topology_class": self._topology_class,
            "recipe_hash":    self._recipe_hash[:8] if self._recipe_hash else None,
            "pid":            self._pid,
            "idle_seconds":   round(self.idle_seconds, 2),
            "spawn_count":    self._spawn_count,
            "reuse_count":    self._reuse_count,
        }


# Module-level warm container. One per Python process.
_warm: _WarmContainer = _WarmContainer()


# ═════════════════════════════════════════════════════════════════════════════
# INVOCATION CONTEXT
#
# A complete tracking object for one pipeline invocation. Accumulates
# lifecycle events, timing marks, and diagnostic data as the invocation
# progresses through phases. Provides a structured audit trail and is the
# single source of truth for what happened during a specific run.
#
# Not a frozen dataclass — it is mutated as the invocation progresses.
# Created at the top of execute(), populated through each phase, and
# consumed at the end for telemetry emission and diagnostic logging.
# ═════════════════════════════════════════════════════════════════════════════

class _InvocationContext:
    """
    Full lifecycle context for one pipeline invocation.

    Created at the beginning of execute(). Populated incrementally as the
    invocation passes through each phase. Consumed at the end for telemetry,
    audit events, and diagnostic logging.

    The context acts as a transaction journal: if the invocation fails at
    any phase, the context contains the full history up to the failure
    point, which is invaluable for post-mortem debugging.
    """

    __slots__ = (
        "run_id",
        "topology_class",
        "source_url",
        "raw_byte_count",
        "config",
        "_timer",
        "_events",
        "_recipe_mount",
        "_subprocess_result",
        "_lifecycle",
        "_kernel_output",
        "_telemetry",
        "_phase",
        "_outcome",
        "_spawn_reused",
        "_attempt_count",
        "_created_at",
    )

    def __init__(
        self,
        kernel_input: KernelInput,
        config: PipelineConfig,
    ) -> None:
        self.run_id: str = kernel_input.run_id
        self.topology_class: str = kernel_input.topology_class
        self.source_url: str = kernel_input.source_url
        self.raw_byte_count: int = kernel_input.raw_byte_count
        self.config: PipelineConfig = config

        self._timer: _MonotonicTimer = _MonotonicTimer()
        self._events: List[Tuple[float, str, str]] = []  # (monotonic, phase, detail)
        self._recipe_mount: Optional[RecipeMount] = None
        self._subprocess_result: Optional[_SubprocessResult] = None
        self._lifecycle: Optional[ContainerLifecycle] = None
        self._kernel_output: Optional[KernelOutput] = None
        self._telemetry: Optional[PipelineTelemetry] = None
        self._phase: str = "init"
        self._outcome: str = "pending"
        self._spawn_reused: bool = False
        self._attempt_count: int = 0
        self._created_at: datetime = datetime.now(timezone.utc)

    @property
    def timer(self) -> _MonotonicTimer:
        return self._timer

    @property
    def recipe_mount(self) -> Optional[RecipeMount]:
        return self._recipe_mount

    @recipe_mount.setter
    def recipe_mount(self, value: RecipeMount) -> None:
        self._recipe_mount = value
        self._record_event("recipe_resolved", value.audit_key())

    @property
    def subprocess_result(self) -> Optional[_SubprocessResult]:
        return self._subprocess_result

    @subprocess_result.setter
    def subprocess_result(self, value: _SubprocessResult) -> None:
        self._subprocess_result = value
        self._record_event(
            "subprocess_complete",
            f"pid={value.pid} exit={value.exit_code} timed_out={value.timed_out} "
            f"stdout={value.stdout_size} stderr={value.stderr_size}",
        )

    @property
    def lifecycle(self) -> Optional[ContainerLifecycle]:
        return self._lifecycle

    @lifecycle.setter
    def lifecycle(self, value: ContainerLifecycle) -> None:
        self._lifecycle = value

    @property
    def kernel_output(self) -> Optional[KernelOutput]:
        return self._kernel_output

    @kernel_output.setter
    def kernel_output(self, value: KernelOutput) -> None:
        self._kernel_output = value

    @property
    def phase(self) -> str:
        return self._phase

    @property
    def outcome(self) -> str:
        return self._outcome

    def enter_phase(self, phase: str) -> None:
        """Mark the start of a named lifecycle phase."""
        self._phase = phase
        self._timer.mark(f"{phase}_start")
        self._record_event("phase_enter", phase)

    def exit_phase(self, phase: str) -> None:
        """Mark the end of a named lifecycle phase."""
        self._timer.mark(f"{phase}_end")
        self._record_event("phase_exit", phase)

    def set_outcome(self, outcome: str) -> None:
        """Set the terminal outcome of this invocation."""
        self._outcome = outcome
        self._record_event("outcome", outcome)

    def record_spawn_reuse(self) -> None:
        """Record that the warm container was reused (no new spawn)."""
        self._spawn_reused = True
        self._record_event("spawn_reused", "warm container compatible")

    def record_spawn_attempt(self, attempt: int) -> None:
        """Record a spawn attempt."""
        self._attempt_count = attempt
        self._record_event("spawn_attempt", f"attempt {attempt}")

    def elapsed_ms(self, start_phase: str, end_phase: str) -> float:
        """Elapsed ms between phase start and phase end."""
        return self._timer.elapsed_ms(
            f"{start_phase}_start",
            f"{end_phase}_end",
        )

    def total_elapsed_ms(self) -> float:
        """Total elapsed ms since context creation."""
        return self._timer.elapsed_since_ms("execute_start")

    def _record_event(self, event_type: str, detail: str) -> None:
        """Append an event to the internal journal."""
        self._events.append((time.monotonic(), event_type, detail))

    def to_diagnostic_dict(self) -> Dict[str, object]:
        """
        Full diagnostic snapshot of this invocation context.

        Suitable for structured logging on failure paths. Contains the
        complete lifecycle journal, timing breakdown, and outcome
        information. Large — do not emit on every successful invocation.
        """
        recipe_key = (
            self._recipe_mount.audit_key() if self._recipe_mount else "none"
        )
        return {
            "run_id":         self.run_id,
            "topology_class": self.topology_class,
            "source_url":     self.source_url,
            "raw_byte_count": self.raw_byte_count,
            "recipe":         recipe_key,
            "phase":          self._phase,
            "outcome":        self._outcome,
            "spawn_reused":   self._spawn_reused,
            "attempt_count":  self._attempt_count,
            "event_count":    len(self._events),
            "created_at":     self._created_at.isoformat(),
        }

    def to_summary_line(self) -> str:
        """One-line summary for log output."""
        recipe_hash = (
            self._recipe_mount.recipe_hash[:8]
            if self._recipe_mount else "none"
        )
        return (
            f"run={self.run_id[:8]} topology={self.topology_class} "
            f"recipe={recipe_hash} phase={self._phase} "
            f"outcome={self._outcome}"
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUBPROCESS ENVIRONMENT
#
# The environment the subprocess runs in must be sanitized. We strip
# potentially dangerous environment variables, set LC_ALL=C for
# deterministic text processing, and constrain PATH to prevent the
# recipes from finding executables outside the allowed set.
# ═════════════════════════════════════════════════════════════════════════════

# Environment variables that must be removed from the subprocess env.
# These can influence shell behavior in ways that break the security
# model. For example:
#   BASH_ENV     — sourced by bash on startup, could inject commands
#   ENV          — sourced by sh on startup in interactive mode
#   IFS          — internal field separator, can corrupt argument parsing
#   CDPATH       — changes cd behavior
#   GLOBIGNORE   — changes glob behavior
#   SHELLOPTS    — changes shell option defaults
#   BASHOPTS     — changes bash option defaults
#   LD_PRELOAD   — injects shared libraries
#   LD_LIBRARY_PATH — changes library search path
_STRIPPED_ENV_VARS: Tuple[str, ...] = (
    "BASH_ENV",
    "ENV",
    "IFS",
    "CDPATH",
    "GLOBIGNORE",
    "SHELLOPTS",
    "BASHOPTS",
    "LD_PRELOAD",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "PERL5LIB",
    "RUBYLIB",
    "NODE_PATH",
    "PROMPT_COMMAND",
    "PS1",
    "PS2",
    "PS4",
    "COMP_WORDBREAKS",
)

# Forced environment variables for the subprocess.
# LC_ALL=C ensures deterministic text processing:
#   - Character class ranges ([a-z], [A-Z]) behave as ASCII
#   - Sort order is byte-order, not locale-order
#   - No locale-dependent behavior in sed, awk, grep
# This is load-bearing. Without LC_ALL=C, a recipe that works in en_US
# may produce different output in de_DE because [a-z] includes ä, ö, ü
# in the de_DE locale.
_FORCED_ENV_VARS: Dict[str, str] = {
    "LC_ALL": "C",
    "LANG": "C",
    "LANGUAGE": "C",
}

# Restricted PATH for the subprocess. Only directories that contain the
# allowed commands. /usr/bin and /bin cover grep, sed, awk, cat, cut, tr,
# head, tail, sort, uniq on Alpine and most Linux distributions.
# The restricted PATH prevents the recipe from finding executables like
# curl, wget, python, perl even if they are installed on the system.
_RESTRICTED_PATH: str = "/usr/bin:/bin"


def _build_subprocess_env() -> Dict[str, str]:
    """
    Construct the sanitized environment for subprocess execution.

    Starts from a minimal base (not os.environ). Only includes:
      - PATH (restricted)
      - HOME (required by some tools for temp file locations)
      - LC_ALL, LANG, LANGUAGE (forced to C)
      - TERM (xterm-256color — prevents tool warnings)

    All dangerous environment variables are excluded. This is defense-in-depth:
    the container's network_mode=none and read-only recipe mount already limit
    the attack surface, but a sanitized environment closes another vector.
    """
    env: Dict[str, str] = {
        "PATH":  _RESTRICTED_PATH,
        "HOME":  "/tmp",
        "TERM":  "xterm-256color",
    }

    # Apply forced variables.
    env.update(_FORCED_ENV_VARS)

    return env


# Module-level cached environment. Built once at import time.
# The subprocess environment never changes at runtime.
_SUBPROCESS_ENV: Dict[str, str] = _build_subprocess_env()


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE STAGING
#
# Before a subprocess can execute a recipe, the recipe file must be:
#   1. Present on disk at recipe_mount.recipe_path
#   2. A regular file (not a symlink, directory, or device)
#   3. Readable by the current process
#   4. Non-empty
#   5. Not too large (sanity bound)
#
# These checks are defense-in-depth. The registry and validator already
# verify most of these. But pipeline.py verifies independently because
# the file could have been deleted, moved, or corrupted between
# validation and execution. The time window is small but non-zero.
#
# In the Docker deployment model, the recipe is mounted at /recipe/run.sh
# via a read-only volume bind. The staging check verifies the mount exists.
# ═════════════════════════════════════════════════════════════════════════════

# Maximum recipe file size in bytes. Defense against a replaced file.
# The validator already checks this, but pipeline.py re-checks because
# the file could have been replaced between validation and execution.
_MAX_RECIPE_FILE_BYTES: int = 256 * 1024  # 256 KB


def _preflight_recipe(recipe_mount: RecipeMount) -> None:
    """
    Pre-flight verification of the recipe file before subprocess spawn.

    Checks:
      1. File exists on disk
      2. Is a regular file (not symlink, directory, device)
      3. Is readable
      4. Is non-empty
      5. Does not exceed size ceiling
      6. Starts with a shebang or valid shell content (not binary)

    On any failure: raises RecipeMountError. This is a hard stop —
    the recipe cannot be executed.

    This function is intentionally paranoid. The registry and validator
    have already verified most of these conditions. pipeline.py re-checks
    because the file system can change between validation and execution.
    The window is small but non-zero, and the cost of re-checking is
    negligible compared to the cost of executing a corrupt recipe.
    """
    path = Path(recipe_mount.recipe_path)

    if not path.exists():
        raise RecipeMountError(
            f"Recipe file missing at execution time: {recipe_mount.recipe_path!r}. "
            "File existed during validation but is gone now. "
            "Check: volume mounts, concurrent file operations, filesystem health.",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        )

    if not path.is_file():
        raise RecipeMountError(
            f"Recipe path is not a regular file at execution time: "
            f"{recipe_mount.recipe_path!r}. "
            f"Type: {oct(path.stat().st_mode) if path.exists() else 'gone'}. "
            "Symlinks, directories, and device nodes are not valid recipes.",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        )

    try:
        stat = path.stat()
    except OSError as exc:
        raise RecipeMountError(
            f"Cannot stat recipe file: {recipe_mount.recipe_path!r}: {exc}",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        ) from exc

    if stat.st_size == 0:
        raise RecipeMountError(
            f"Recipe file is empty (0 bytes) at execution time: "
            f"{recipe_mount.recipe_path!r}. "
            "File may have been truncated between validation and execution.",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        )

    if stat.st_size > _MAX_RECIPE_FILE_BYTES:
        raise RecipeMountError(
            f"Recipe file is {stat.st_size:,} bytes at execution time, "
            f"exceeding {_MAX_RECIPE_FILE_BYTES:,}-byte ceiling: "
            f"{recipe_mount.recipe_path!r}. "
            "File may have been replaced between validation and execution.",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        )

    # Verify the file is readable. This catches permission issues that
    # appeared between validation (which reads the file) and execution.
    if not os.access(recipe_mount.recipe_path, os.R_OK):
        raise RecipeMountError(
            f"Recipe file is not readable: {recipe_mount.recipe_path!r}. "
            "Permissions may have changed between validation and execution.",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        )

    # Quick binary detection: read the first 512 bytes and check for
    # null bytes. A shell script should never contain null bytes. If the
    # file starts with binary content, it is not a valid recipe.
    try:
        with open(recipe_mount.recipe_path, "rb") as f:
            header = f.read(512)
    except OSError as exc:
        raise RecipeMountError(
            f"Cannot read recipe file header: {recipe_mount.recipe_path!r}: {exc}",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        ) from exc

    if b"\x00" in header:
        raise RecipeMountError(
            f"Recipe file contains null bytes (binary content): "
            f"{recipe_mount.recipe_path!r}. "
            "A shell recipe must be valid text. The file may have been "
            "corrupted or replaced with a binary.",
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
        )

    logger.debug(
        "Recipe pre-flight passed: path=%s size=%d bytes",
        recipe_mount.recipe_path,
        stat.st_size,
    )


def _preflight_shell(config: PipelineConfig) -> None:
    """
    Verify the shell executable exists and is executable.

    Called once at pipeline startup, not on every invocation. If the shell
    is missing, every invocation will fail with ContainerSpawnError. Better
    to detect this once at startup with a clear error message.

    Raises RecipeMountError if the shell is not available.
    """
    shell = Path(config.shell_executable)

    if not shell.exists():
        raise RecipeMountError(
            f"Shell executable not found: {config.shell_executable!r}. "
            "pipeline.py requires a POSIX-compatible shell to execute recipes.",
            run_id=new_run_id(),
        )

    if not os.access(config.shell_executable, os.X_OK):
        raise RecipeMountError(
            f"Shell executable is not executable: {config.shell_executable!r}. "
            f"Mode: {oct(shell.stat().st_mode)}. "
            "Check file permissions.",
            run_id=new_run_id(),
        )


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE HEALTH
#
# Module-level rolling statistics for the pipeline's operational health.
# Consumed by TAG's telemetry layer and Witness. Not persisted — reset on
# process restart. Ephemeral by design: stale health stats from a previous
# session are misleading.
# ═════════════════════════════════════════════════════════════════════════════

class _PipelineHealth:
    """
    Rolling health statistics for the pipeline.

    Tracks:
      - Total invocation count
      - Success count (extraction_empty=False)
      - Empty extraction count
      - Timeout count
      - Spawn failure count
      - Hard stop count (RecipeMountError, RecipeInjectionAttempt)
      - Cumulative latency for mean calculation

    Thread-safe: the GIL protects integer increments. No explicit locking
    needed for the statistics — they are approximate by design.
    Approximate rolling stats are more useful than precise delayed stats.
    """

    __slots__ = (
        "_total",
        "_success",
        "_empty",
        "_timeout",
        "_spawn_fail",
        "_hard_stop",
        "_encoding_fail",
        "_cumulative_latency_ms",
        "_started_at",
    )

    def __init__(self) -> None:
        self._total:               int = 0
        self._success:             int = 0
        self._empty:               int = 0
        self._timeout:             int = 0
        self._spawn_fail:          int = 0
        self._hard_stop:           int = 0
        self._encoding_fail:       int = 0
        self._cumulative_latency_ms: float = 0.0
        self._started_at:          datetime = datetime.now(timezone.utc)

    def record_success(self, latency_ms: float) -> None:
        self._total += 1
        self._success += 1
        self._cumulative_latency_ms += latency_ms

    def record_empty(self, latency_ms: float) -> None:
        self._total += 1
        self._empty += 1
        self._cumulative_latency_ms += latency_ms

    def record_timeout(self, latency_ms: float) -> None:
        self._total += 1
        self._timeout += 1
        self._cumulative_latency_ms += latency_ms

    def record_spawn_failure(self) -> None:
        self._total += 1
        self._spawn_fail += 1

    def record_hard_stop(self) -> None:
        self._total += 1
        self._hard_stop += 1

    def record_encoding_failure(self) -> None:
        self._total += 1
        self._encoding_fail += 1

    @property
    def total(self) -> int:
        return self._total

    @property
    def success_rate(self) -> float:
        """Fraction of invocations that produced non-empty output."""
        if self._total == 0:
            return 0.0
        return self._success / self._total

    @property
    def empty_rate(self) -> float:
        """Fraction of invocations with empty extraction."""
        if self._total == 0:
            return 0.0
        return self._empty / self._total

    @property
    def timeout_rate(self) -> float:
        if self._total == 0:
            return 0.0
        return self._timeout / self._total

    @property
    def mean_latency_ms(self) -> float:
        if self._total == 0:
            return 0.0
        return self._cumulative_latency_ms / self._total

    def to_health_dict(self) -> Dict[str, object]:
        """Health snapshot for Witness telemetry."""
        return {
            "total_invocations":  self._total,
            "success_count":     self._success,
            "empty_count":       self._empty,
            "timeout_count":     self._timeout,
            "spawn_fail_count":  self._spawn_fail,
            "hard_stop_count":   self._hard_stop,
            "encoding_fail_count": self._encoding_fail,
            "success_rate":      round(self.success_rate, 4),
            "empty_rate":        round(self.empty_rate, 4),
            "timeout_rate":      round(self.timeout_rate, 4),
            "mean_latency_ms":   round(self.mean_latency_ms, 2),
            "uptime_since":      self._started_at.isoformat(),
        }


class _TopologyHealthTracker:
    """
    Per-topology-class health tracking.

    Tracks the same metrics as _PipelineHealth but broken down by topology
    class. This allows Witness to identify topology classes that are
    degrading — a high timeout rate on REST_API_JSON but not on NEWS_ARTICLE
    is a recipe-specific problem, not a pipeline-wide problem.

    Uses a dict of topology_class → counters. New topology classes are
    added lazily on first invocation.
    """

    __slots__ = ("_per_class",)

    def __init__(self) -> None:
        self._per_class: Dict[str, Dict[str, object]] = {}

    def _ensure_class(self, topology_class: str) -> Dict[str, object]:
        """Lazily create counters for a topology class."""
        if topology_class not in self._per_class:
            self._per_class[topology_class] = {
                "total":              0,
                "success":            0,
                "empty":              0,
                "timeout":            0,
                "cumulative_ms":      0.0,
                "last_success_at":    None,
                "last_empty_at":      None,
                "consecutive_empty":  0,
            }
        return self._per_class[topology_class]

    def record_success(self, topology_class: str, latency_ms: float) -> None:
        c = self._ensure_class(topology_class)
        c["total"] += 1  # type: ignore[operator]
        c["success"] += 1  # type: ignore[operator]
        c["cumulative_ms"] += latency_ms  # type: ignore[operator]
        c["last_success_at"] = datetime.now(timezone.utc).isoformat()
        c["consecutive_empty"] = 0

    def record_empty(self, topology_class: str, latency_ms: float) -> None:
        c = self._ensure_class(topology_class)
        c["total"] += 1  # type: ignore[operator]
        c["empty"] += 1  # type: ignore[operator]
        c["cumulative_ms"] += latency_ms  # type: ignore[operator]
        c["last_empty_at"] = datetime.now(timezone.utc).isoformat()
        c["consecutive_empty"] += 1  # type: ignore[operator]

    def record_timeout(self, topology_class: str, latency_ms: float) -> None:
        c = self._ensure_class(topology_class)
        c["total"] += 1  # type: ignore[operator]
        c["timeout"] += 1  # type: ignore[operator]
        c["cumulative_ms"] += latency_ms  # type: ignore[operator]

    def consecutive_empty_count(self, topology_class: str) -> int:
        """Number of consecutive empty extractions for this class."""
        if topology_class not in self._per_class:
            return 0
        return self._per_class[topology_class]["consecutive_empty"]  # type: ignore[return-value]

    def empty_rate(self, topology_class: str) -> float:
        """Empty extraction rate for this specific topology class."""
        if topology_class not in self._per_class:
            return 0.0
        c = self._per_class[topology_class]
        total = c["total"]
        if total == 0:  # type: ignore[operator]
            return 0.0
        return c["empty"] / total  # type: ignore[operator]

    def to_health_dict(self) -> Dict[str, object]:
        """Per-class health snapshot."""
        return dict(self._per_class)


# Module-level health trackers. One of each per Python process.
_health: _PipelineHealth = _PipelineHealth()
_topology_health: _TopologyHealthTracker = _TopologyHealthTracker()


def get_pipeline_health() -> Dict[str, object]:
    """
    Return the pipeline's current health statistics.

    Called by TAG's telemetry layer to expose pipeline health to Witness.
    Returns a snapshot — the values may change between the call and the
    caller's use of the result.
    """
    return {
        "aggregate": _health.to_health_dict(),
        "per_topology": _topology_health.to_health_dict(),
        "warm_container": _warm.diagnostic_dict(),
    }


# ═════════════════════════════════════════════════════════════════════════════
# TIMING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

class _MonotonicTimer:
    """
    Monotonic wall-clock timer for measuring subprocess lifecycle phases.

    Uses time.monotonic() which is immune to NTP adjustments, system clock
    changes, and daylight saving transitions. This is the only correct
    clock source for measuring latency.

    Usage:
        timer = _MonotonicTimer()
        timer.mark("spawn_start")
        ... spawn ...
        timer.mark("spawn_end")
        timer.mark("communicate_start")
        ... communicate ...
        timer.mark("communicate_end")

        spawn_ms = timer.elapsed_ms("spawn_start", "spawn_end")
        total_ms = timer.elapsed_ms("spawn_start", "communicate_end")
    """

    __slots__ = ("_marks",)

    def __init__(self) -> None:
        self._marks: Dict[str, float] = {}

    def mark(self, name: str) -> None:
        """Record a named timestamp."""
        self._marks[name] = time.monotonic()

    def elapsed_ms(self, start: str, end: str) -> float:
        """
        Elapsed time in milliseconds between two marks.

        Raises KeyError if either mark does not exist — this is a
        programming error in pipeline.py, not a runtime failure.
        """
        return (self._marks[end] - self._marks[start]) * 1000.0

    def elapsed_since_ms(self, start: str) -> float:
        """Elapsed time in milliseconds from a mark to now."""
        return (time.monotonic() - self._marks[start]) * 1000.0

    def has_mark(self, name: str) -> bool:
        return name in self._marks


# ═════════════════════════════════════════════════════════════════════════════
# STDIN ENCODING
#
# Converting raw_content (str) to bytes for the subprocess stdin pipe.
# Not trivial: malformed HTML from hostile pages can contain characters
# that fail UTF-8 encoding despite being valid Python str objects (e.g.
# bare surrogates from broken JavaScript string handling).
# ═════════════════════════════════════════════════════════════════════════════

def _encode_stdin(
    kernel_input: KernelInput,
) -> bytes:
    """
    Encode raw_content as UTF-8 bytes for the subprocess stdin pipe.

    On encoding failure, raises StdinEncodingError with full diagnostic
    context. The caller (execute()) catches this and returns empty
    KernelOutput. The recipe never sees the content.

    This is a deliberate chokepoint: every byte that enters the subprocess
    passes through here. No raw content bypasses this function.
    """
    try:
        encoded = kernel_input.raw_content.encode(
            _STDIN_ENCODING,
            errors=_STDIN_ENCODING_ERRORS,
        )
    except UnicodeEncodeError as exc:
        raise StdinEncodingError(
            run_id=kernel_input.run_id,
            topology_class=kernel_input.topology_class,
            source_url=kernel_input.source_url,
            encoding_error=str(exc),
            raw_byte_count=kernel_input.raw_byte_count,
        ) from exc

    return encoded


# ═════════════════════════════════════════════════════════════════════════════
# TOKEN ESTIMATION
# ═════════════════════════════════════════════════════════════════════════════

def _estimate_token_delta(raw_byte_count: int, clean_byte_count: int) -> int:
    """
    Estimate the number of LLM tokens saved by noise stripping.

    Uses the char/4 approximation: approximately 4 characters per token
    for English text across most tokenizers. The byte count is a proxy for
    character count in UTF-8 where most HTML content is ASCII.

    This estimate is sufficient for feedback.py's quality scoring and
    for Witness telemetry. It does not need to be exact. Calling a real
    tokenizer would introduce a dependency the kernel should not have.

    Returns a non-negative integer. If clean exceeds raw (which should
    never happen — validated by KernelOutput.__post_init__), returns 0.
    """
    if clean_byte_count >= raw_byte_count:
        return 0
    delta_bytes = raw_byte_count - clean_byte_count
    return max(0, delta_bytes // _CHARS_PER_TOKEN)


# ═════════════════════════════════════════════════════════════════════════════
# SUBPROCESS LIFECYCLE — SPAWN
#
# Creating the asyncio subprocess. This is where the container is born.
# The subprocess executes `sh recipe_path` with stdin piped. stdout and
# stderr are captured via PIPE. No shell wrapping — the recipe IS the
# shell script.
# ═════════════════════════════════════════════════════════════════════════════

async def _spawn_process(
    recipe_mount: RecipeMount,
    config: PipelineConfig,
) -> Tuple[asyncio.subprocess.Process, float]:
    """
    Spawn the subprocess that will execute the recipe.

    Returns (process, spawn_latency_ms).

    The subprocess is created via asyncio.create_subprocess_exec() with:
      - stdin=PIPE   : we write raw content to it
      - stdout=PIPE  : we read clean signal from it
      - stderr=PIPE  : we capture diagnostic output
      - No shell wrapping — the recipe IS the shell argument to sh

    Spawn timeout is enforced via asyncio.wait_for(). If the process
    does not become ready within spawn_timeout_ms, the spawn attempt
    is considered failed.

    Raises ContainerSpawnError on failure. The caller retries once.
    """
    timer = _MonotonicTimer()
    timer.mark("spawn_start")

    try:
        process = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                config.shell_executable,
                recipe_mount.recipe_path,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                # Sanitized environment: LC_ALL=C, restricted PATH,
                # stripped dangerous variables. See _build_subprocess_env().
                env=_SUBPROCESS_ENV,
                # Prevent the subprocess from inheriting parent signal
                # handlers. The subprocess should receive signals directly
                # from the OS, not through the Python process's signal
                # infrastructure. This ensures SIGTERM reaches the shell
                # pipeline rather than being caught by Python's handler.
                preexec_fn=os.setpgrp if hasattr(os, "setpgrp") else None,
            ),
            timeout=config.spawn_timeout_seconds,
        )
    except asyncio.TimeoutError:
        spawn_ms = timer.elapsed_since_ms("spawn_start")
        raise ContainerSpawnError(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
            spawn_timeout_ms=config.spawn_timeout_ms,
            attempt_number=0,  # caller sets the real attempt number
            os_error=f"Spawn timed out after {spawn_ms:.1f}ms",
        )
    except OSError as exc:
        spawn_ms = timer.elapsed_since_ms("spawn_start") # noqa
        raise ContainerSpawnError(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_hash=recipe_mount.recipe_hash,
            spawn_timeout_ms=config.spawn_timeout_ms,
            attempt_number=0,
            os_error=str(exc),
        )

    spawn_ms = timer.elapsed_since_ms("spawn_start")

    logger.debug(
        "Subprocess spawned: pid=%s topology=%s recipe=%s spawn_ms=%.1f",
        process.pid,
        recipe_mount.topology_class,
        recipe_mount.recipe_hash[:8],
        spawn_ms,
    )

    return process, spawn_ms


# ═════════════════════════════════════════════════════════════════════════════
# SUBPROCESS LIFECYCLE — COMMUNICATE
#
# Writing stdin, reading stdout/stderr, enforcing timeout. This is the
# most critical phase — large pages (500KB+) can cause pipe deadlock if
# stdin writing blocks before the grep pipeline consumes. communicate()
# handles this correctly by writing and reading concurrently.
# ═════════════════════════════════════════════════════════════════════════════

async def _communicate_with_timeout(
    process: asyncio.subprocess.Process,
    stdin_bytes: bytes,
    timeout_seconds: float,
    config: PipelineConfig,
) -> Tuple[bytes, bytes, Optional[int], bool]:
    """
    Write stdin, read stdout/stderr, enforce timeout.

    Returns (stdout_bytes, stderr_bytes, exit_code, timed_out).

    communicate() is the correct primitive for subprocess I/O:
      - It writes stdin in a coroutine that closes the pipe on completion
      - It reads stdout and stderr concurrently
      - It handles the pipe buffer dance correctly for large payloads
      - It waits for the process to exit after I/O completes

    asyncio.wait_for() wraps the communicate() call with a hard timeout.
    On timeout:
      1. The communicate() coroutine is cancelled
      2. We kill the process (SIGTERM → SIGKILL escalation)
      3. We drain any remaining output from the pipes
      4. We return (partial_stdout, partial_stderr, None, True)

    The timed_out=True flag tells the caller that the output is incomplete.
    exit_code is None because a killed process does not produce a meaningful
    exit code.
    """
    timed_out = False
    stdout_bytes = b""
    stderr_bytes = b""
    exit_code: Optional[int] = None # noqa

    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            process.communicate(input=stdin_bytes),
            timeout=timeout_seconds,
        )
        exit_code = process.returncode

    except asyncio.TimeoutError:
        # ── Timeout path ──────────────────────────────────────────────
        # The grep pipeline did not complete within the allowed window.
        # Kill the process. Drain any buffered output.
        timed_out = True

        logger.warning(
            "Subprocess pid=%s timed out after %.1fs — killing",
            process.pid,
            timeout_seconds,
        )

        await _kill_process(process)

        # After killing, try to drain any output that was buffered before
        # the kill. The pipes may have partial data. This is best-effort —
        # if the drain fails, we return empty bytes.
        try:
            if process.stdout is not None:
                stdout_bytes = await asyncio.wait_for(
                    process.stdout.read(),
                    timeout=1.0,
                )
        except (asyncio.TimeoutError, Exception):
            stdout_bytes = b""

        try:
            if process.stderr is not None:
                stderr_bytes = await asyncio.wait_for(
                    process.stderr.read(),
                    timeout=1.0,
                )
        except (asyncio.TimeoutError, Exception):
            stderr_bytes = b""

        exit_code = None  # killed process — no meaningful exit code

    except Exception:
        # ── Unexpected failure in communicate() ───────────────────────
        # BrokenPipeError, ConnectionResetError, or other transport-level
        # failure. Kill the process and return empty.
        await _kill_process(process)
        raise

    # Truncate stderr to retain limit.
    if len(stderr_bytes) > config.stderr_retain_limit:
        stderr_bytes = stderr_bytes[: config.stderr_retain_limit]

    return stdout_bytes, stderr_bytes, exit_code, timed_out


# ═════════════════════════════════════════════════════════════════════════════
# SUBPROCESS LIFECYCLE — KILL
#
# Graceful termination with escalation. SIGTERM first, then SIGKILL if
# the process does not exit. SIGTERM allows the shell pipeline to flush
# stdout. SIGKILL does not — some output may be lost.
# ═════════════════════════════════════════════════════════════════════════════

async def _kill_process(
    process: asyncio.subprocess.Process,
) -> None:
    """
    Kill a subprocess with SIGTERM → SIGKILL escalation.

    Does not raise. A failed kill is logged but does not affect the caller.
    The caller will return empty KernelOutput regardless.
    """
    if process.returncode is not None:
        # Already exited.
        return

    pid = process.pid

    try:
        process.terminate()  # SIGTERM
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("Failed to SIGTERM pid=%s: %s", pid, exc)
        return

    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_GRACEFUL_KILL_TIMEOUT_SECONDS,
        )
        logger.debug("Process pid=%s exited gracefully after SIGTERM", pid)
        return
    except asyncio.TimeoutError:
        pass

    # Process did not exit after SIGTERM. Escalate to SIGKILL.
    logger.warning(
        "Process pid=%s did not exit after SIGTERM (%.1fs). Sending SIGKILL.",
        pid,
        _GRACEFUL_KILL_TIMEOUT_SECONDS,
    )

    try:
        process.kill()  # SIGKILL
    except ProcessLookupError:
        return
    except OSError as exc:
        logger.warning("Failed to SIGKILL pid=%s: %s", pid, exc)
        return

    try:
        await asyncio.wait_for(
            process.wait(),
            timeout=_FORCE_KILL_REAP_TIMEOUT_SECONDS,
        )
        logger.debug("Process pid=%s reaped after SIGKILL", pid)
    except asyncio.TimeoutError:
        logger.critical(
            "Process pid=%s not reaped after SIGKILL + %.1fs. "
            "Possible zombie or D-state I/O. Abandoning process.",
            pid,
            _FORCE_KILL_REAP_TIMEOUT_SECONDS,
        )


# ═════════════════════════════════════════════════════════════════════════════
# SUBPROCESS LIFECYCLE — SINGLE ATTEMPT
#
# One complete spawn → communicate → capture cycle. The caller may retry
# on ContainerSpawnError. Other failures are not retried.
# ═════════════════════════════════════════════════════════════════════════════

async def _run_single_attempt(
    recipe_mount: RecipeMount,
    stdin_bytes: bytes,
    config: PipelineConfig,
    attempt_number: int,
    timer: _MonotonicTimer,
) -> _SubprocessResult:
    """
    Execute one complete subprocess lifecycle attempt.

    Phases:
      1. Spawn the subprocess
      2. communicate() with timeout
      3. Capture results into _SubprocessResult

    On spawn failure: raises ContainerSpawnError with the attempt_number.
    On communicate failure: raises the underlying exception.
    On success: returns _SubprocessResult with all captured data.

    The process is guaranteed to be dead or exited when this function
    returns — either normally or via exception. No leaked processes.
    """
    process: Optional[asyncio.subprocess.Process] = None

    try:
        # ── Phase 1: Spawn ────────────────────────────────────────────
        timer.mark(f"spawn_start_{attempt_number}")

        try:
            process, spawn_ms = await _spawn_process(recipe_mount, config)
        except ContainerSpawnError as exc:
            # Re-raise with correct attempt number.
            raise ContainerSpawnError(
                run_id=exc.run_id or new_run_id(),
                topology_class=recipe_mount.topology_class,
                recipe_hash=recipe_mount.recipe_hash,
                spawn_timeout_ms=config.spawn_timeout_ms,
                attempt_number=attempt_number,
                os_error=str(exc),
            ) from exc

        timer.mark(f"spawn_end_{attempt_number}")

        # Record the new spawn so warm container reuse works on next invocation.
        _warm.record_spawn(
            recipe_mount.topology_class,
            recipe_mount.recipe_hash,
            process,
        )

        # ── Phase 2: Communicate ──────────────────────────────────────
        timer.mark(f"communicate_start_{attempt_number}")

        stdout_bytes, stderr_bytes, exit_code, timed_out = (
            await _communicate_with_timeout(
                process,
                stdin_bytes,
                config.subprocess_timeout_seconds,
                config,
            )
        )

        timer.mark(f"communicate_end_{attempt_number}")

        # ── Phase 3: Capture ──────────────────────────────────────────
        total_ms = timer.elapsed_ms(
            f"spawn_start_{attempt_number}",
            f"communicate_end_{attempt_number}",
        )

        return _SubprocessResult(
            stdout_bytes=stdout_bytes,
            stderr_bytes=stderr_bytes,
            exit_code=exit_code,
            timed_out=timed_out,
            spawn_latency_ms=spawn_ms,
            total_latency_ms=total_ms,
            pid=process.pid if process else None,
        )

    except Exception:
        # Ensure the process is dead on any exception path.
        if process is not None and process.returncode is None:
            await _kill_process(process)
        raise


async def _run_with_retry(
    recipe_mount: RecipeMount,
    stdin_bytes: bytes,
    config: PipelineConfig,
    kernel_input: KernelInput, # noqa
) -> _SubprocessResult:
    """
    Execute the subprocess with retry on spawn failure.

    Retry policy: one retry on ContainerSpawnError. Other exceptions are
    not retried — they indicate content-level or recipe-level problems
    that a retry will not fix.

    On spawn failure after all attempts: raises the final
    ContainerSpawnError. The caller catches this and returns empty
    KernelOutput.
    """
    timer = _MonotonicTimer()
    timer.mark("run_start")

    last_spawn_error: Optional[ContainerSpawnError] = None

    for attempt in range(1, config.max_spawn_attempts + 1):
        try:
            result = await _run_single_attempt(
                recipe_mount,
                stdin_bytes,
                config,
                attempt_number=attempt,
                timer=timer,
            )

            if attempt > 1:
                logger.info(
                    "Subprocess succeeded on attempt %d (previous spawn failures): "
                    "topology=%s pid=%s",
                    attempt,
                    recipe_mount.topology_class,
                    result.pid,
                )

            return result

        except ContainerSpawnError as exc:
            last_spawn_error = exc

            if attempt < config.max_spawn_attempts:
                logger.warning(
                    "Spawn attempt %d/%d failed for topology=%s: %s. "
                    "Retrying in %.2fs...",
                    attempt,
                    config.max_spawn_attempts,
                    recipe_mount.topology_class,
                    exc,
                    config.spawn_retry_delay_s,
                )
                await asyncio.sleep(config.spawn_retry_delay_s)
            else:
                logger.error(
                    "All %d spawn attempts exhausted for topology=%s. "
                    "Final error: %s",
                    config.max_spawn_attempts,
                    recipe_mount.topology_class,
                    exc,
                )

    # All attempts exhausted. Re-raise the last spawn error.
    # This is always set because max_spawn_attempts ≥ 1 and we only
    # reach here if every attempt raised ContainerSpawnError.
    assert last_spawn_error is not None  # noqa: S101
    raise last_spawn_error


# ═════════════════════════════════════════════════════════════════════════════
# OUTPUT CONSTRUCTION
#
# Converting raw subprocess results into domain contracts.
# Every field in KernelOutput and ContainerLifecycle is computed here.
# No downstream file re-computes these — the values are final.
# ═════════════════════════════════════════════════════════════════════════════

def _decode_stdout(stdout_bytes: bytes) -> str:
    """
    Decode subprocess stdout as UTF-8 text.

    The recipe's grep/sed pipeline outputs UTF-8 text. If the output
    contains invalid UTF-8 sequences (which should not happen with a
    correct recipe processing valid UTF-8 input), we use 'replace' mode
    to substitute the replacement character. This is a defensive measure —
    losing one character is better than crashing the pipeline.
    """
    return stdout_bytes.decode("utf-8", errors="replace")


def _build_container_lifecycle(
    kernel_input: KernelInput,
    recipe_mount: RecipeMount,
    result: _SubprocessResult,
) -> ContainerLifecycle:
    """
    Construct the ContainerLifecycle contract from subprocess measurements.

    This is pipeline.py's internal record of what happened during the
    subprocess invocation. Combined with KernelOutput to produce
    PipelineTelemetry via make_pipeline_telemetry().

    On ProcessStateCorruption (timed_out=True with exit_code set):
    This should never happen — a process killed by timeout does not produce
    a meaningful exit code. If it occurs, the subprocess lifecycle tracking
    has a bug. We correct it defensively (set exit_code to None) and log
    at CRITICAL level.
    """
    exit_code = result.exit_code
    timed_out = result.timed_out

    # Defensive correction for the mutually exclusive invariant.
    # If the process timed out AND has an exit code, something is wrong
    # in our lifecycle tracking. Clear the exit code and log the corruption.
    if timed_out and exit_code is not None:
        logger.critical(
            "Process state corruption: timed_out=True but exit_code=%d. "
            "Clearing exit_code defensively. "
            "run_id=%s topology=%s recipe=%s pid=%s",
            exit_code,
            kernel_input.run_id,
            kernel_input.topology_class,
            recipe_mount.recipe_hash[:8],
            result.pid,
        )
        exit_code = None

    return ContainerLifecycle(
        run_id=kernel_input.run_id,
        topology_class=kernel_input.topology_class,
        recipe_hash=recipe_mount.recipe_hash,
        spawn_latency_ms=result.spawn_latency_ms,
        total_latency_ms=result.total_latency_ms,
        exit_code=exit_code,
        timed_out=timed_out,
        stderr_bytes=result.stderr_size,
    )


def _build_kernel_output(
    kernel_input: KernelInput,
    recipe_mount: RecipeMount,
    result: _SubprocessResult,
) -> KernelOutput:
    """
    Construct the KernelOutput contract from subprocess results.

    This is the principal output of pipeline.py. Everything the AXIOM graph
    receives about this extraction is in this object.

    Decision tree:
      1. timed_out → extraction_empty=True, clean_signal=""
      2. no stdout → extraction_empty=True, clean_signal=""
      3. stdout present → extraction_empty=False, clean_signal=decoded stdout

    Byte counting:
      - raw_byte_count: from KernelInput (measured at input, not re-measured)
      - clean_byte_count: len(clean_signal.encode("utf-8"))
      - These must satisfy: clean_byte_count ≤ raw_byte_count

    Token delta:
      - Estimated via char/4 approximation
      - 0 if extraction is empty

    If clean_byte_count > raw_byte_count (which should be impossible):
    this is a measurement bug in pipeline.py. We log at CRITICAL and return
    empty KernelOutput rather than construct a KernelOutput with an invariant
    violation that would crash in __post_init__.
    """
    raw_byte_count = kernel_input.raw_byte_count

    # ── Timeout or empty stdout → empty extraction ────────────────────
    if result.timed_out or not result.has_stdout:
        return make_empty_kernel_output(
            run_id=kernel_input.run_id,
            topology_class=kernel_input.topology_class,
            recipe_used=recipe_mount.recipe_path,
            raw_byte_count=raw_byte_count,
            latency_ms=result.total_latency_ms,
        )

    # ── Decode stdout ─────────────────────────────────────────────────
    clean_signal = _decode_stdout(result.stdout_bytes)

    # Strip trailing whitespace/newlines that the recipe's final sed may
    # have left. This ensures clean_signal is clean.
    clean_signal = clean_signal.rstrip()

    if not clean_signal:
        # Stdout contained only whitespace — treat as empty extraction.
        return make_empty_kernel_output(
            run_id=kernel_input.run_id,
            topology_class=kernel_input.topology_class,
            recipe_used=recipe_mount.recipe_path,
            raw_byte_count=raw_byte_count,
            latency_ms=result.total_latency_ms,
        )

    # ── Measure clean output ──────────────────────────────────────────
    clean_byte_count = len(clean_signal.encode("utf-8"))

    # Invariant check: clean ≤ raw.
    # The grep pipeline strips content. Its output cannot be larger than
    # its input. If this check fails, our byte counting is wrong.
    if clean_byte_count > raw_byte_count:
        logger.critical(
            "MEASUREMENT BUG: clean_byte_count (%d) > raw_byte_count (%d). "
            "run_id=%s topology=%s recipe=%s. "
            "Returning empty output instead of constructing invalid KernelOutput.",
            clean_byte_count,
            raw_byte_count,
            kernel_input.run_id,
            kernel_input.topology_class,
            recipe_mount.recipe_hash[:8],
        )
        return make_empty_kernel_output(
            run_id=kernel_input.run_id,
            topology_class=kernel_input.topology_class,
            recipe_used=recipe_mount.recipe_path,
            raw_byte_count=raw_byte_count,
            latency_ms=result.total_latency_ms,
        )

    token_delta = _estimate_token_delta(raw_byte_count, clean_byte_count)

    return KernelOutput(
        clean_signal=clean_signal,
        raw_byte_count=raw_byte_count,
        clean_byte_count=clean_byte_count,
        token_delta_estimate=token_delta,
        recipe_used=recipe_mount.recipe_path,
        topology_class=kernel_input.topology_class,
        extraction_empty=False,
        latency_ms=result.total_latency_ms,
        run_id=kernel_input.run_id,
    )


# ═════════════════════════════════════════════════════════════════════════════
# TELEMETRY EMISSION
#
# Every invocation produces a PipelineTelemetry event. Every security-relevant
# or diagnostic-level outcome produces a KernelAuditEvent. These feed Witness.
# Telemetry emission must never crash the pipeline — if emission fails, log
# the failure and continue.
# ═════════════════════════════════════════════════════════════════════════════

def _emit_telemetry(
    output: KernelOutput,
    lifecycle: ContainerLifecycle,
    recipe_mount: RecipeMount,
) -> Optional[PipelineTelemetry]:
    """
    Construct and log PipelineTelemetry for one invocation.

    Returns the telemetry object for callers that need it (e.g. test
    harnesses). In production, the structured log is the primary output.

    Never raises. If telemetry construction fails (e.g. due to a contract
    invariant violation), the failure is logged and None is returned.
    The pipeline does not fail because telemetry failed.
    """
    try:
        telemetry = make_pipeline_telemetry(
            output=output,
            lifecycle=lifecycle,
            is_hardcoded_recipe=recipe_mount.is_hardcoded,
        )
    except (ValueError, TypeError) as exc:
        logger.error(
            "Failed to construct PipelineTelemetry: %s. "
            "run_id=%s topology=%s. "
            "Telemetry for this invocation is lost.",
            exc,
            output.run_id,
            output.topology_class,
        )
        return None

    # Emit as structured JSON log line. Witness consumes these.
    logger.info(
        "TELEMETRY %s",
        json.dumps(telemetry.to_log_dict(), default=str),
    )

    return telemetry


def _emit_audit_event(
    run_id: RunID,
    event_type: AuditEventType,
    topology_class: TopologyClassStr,
    recipe_hash: RecipeHash,
    detail: str,
    severity: AuditSeverity,
) -> Optional[KernelAuditEvent]:
    """
    Construct and log a KernelAuditEvent.

    Audit events are emitted for security-relevant outcomes (injection
    blocked, hash mismatch) and diagnostic outcomes (stderr non-empty,
    timeout, spawn failure). They are distinct from PipelineTelemetry —
    telemetry is for performance metrics, audit events are for events
    that warrant review.

    Never raises. Audit emission failure is logged but does not affect
    the pipeline.
    """
    try:
        event = KernelAuditEvent(
            run_id=run_id,
            event_type=event_type,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
            detail=detail,
            severity=severity,
        )
    except (ValueError, TypeError) as exc:
        logger.error(
            "Failed to construct KernelAuditEvent: %s. "
            "event_type=%s topology=%s. "
            "Audit event for this invocation is lost.",
            exc,
            event_type,
            topology_class,
        )
        return None

    log_method = {
        "critical": logger.critical,
        "warn":     logger.warning,
        "info":     logger.info,
    }.get(severity, logger.info)

    log_method(
        "AUDIT [%s] %s topology=%s recipe=%s: %s",
        severity.upper(),
        event_type,
        topology_class,
        recipe_hash[:8],
        detail,
    )

    return event


# ═════════════════════════════════════════════════════════════════════════════
# STDERR DIAGNOSTICS
#
# stderr from the subprocess is diagnostic signal, not execution output.
# sed warnings ("unterminated s command"), awk errors, and shell errors
# appear here. Non-empty stderr with non-empty stdout is normal — the
# recipe produced output and also had warnings. Non-empty stderr with
# empty stdout may indicate a broken recipe.
# ═════════════════════════════════════════════════════════════════════════════

def _log_stderr(
    kernel_input: KernelInput,
    recipe_mount: RecipeMount,
    result: _SubprocessResult,
) -> None:
    """
    Log stderr content from the subprocess.

    Logging levels:
      - stderr non-empty + stdout non-empty → DEBUG (warnings with output)
      - stderr non-empty + stdout empty → WARNING (recipe may be broken)
      - stderr empty → no log

    The full retained stderr goes to the log. Additionally, a
    KernelAuditEvent with severity="warn" is emitted for non-empty stderr
    so Witness can track stderr rates per topology class.
    """
    if not result.has_stderr:
        return

    stderr_text = result.stderr_bytes.decode("utf-8", errors="replace")
    preview = stderr_text[:_STDERR_LOG_PREVIEW_BYTES]

    if result.has_stdout:
        # Warnings with output — informational, not alarming.
        logger.debug(
            "Subprocess stderr (with stdout): run_id=%s topology=%s "
            "recipe=%s stderr_bytes=%d preview=%r",
            kernel_input.run_id,
            kernel_input.topology_class,
            recipe_mount.recipe_hash[:8],
            result.stderr_size,
            preview,
        )
    else:
        # Warnings without output — recipe may have a problem.
        logger.warning(
            "Subprocess stderr (NO stdout): run_id=%s topology=%s "
            "recipe=%s exit_code=%s stderr_bytes=%d preview=%r",
            kernel_input.run_id,
            kernel_input.topology_class,
            recipe_mount.recipe_hash[:8],
            result.exit_code,
            result.stderr_size,
            preview,
        )

    # Emit audit event for non-empty stderr.
    _emit_audit_event(
        run_id=kernel_input.run_id,
        event_type="stderr_non_empty",
        topology_class=kernel_input.topology_class,
        recipe_hash=recipe_mount.recipe_hash,
        detail=(
            f"stderr_bytes={result.stderr_size} "
            f"has_stdout={result.has_stdout} "
            f"exit_code={result.exit_code} "
            f"preview={preview!r}"
        ),
        severity="warn" if not result.has_stdout else "info",
    )


def _classify_exit_code(exit_code: Optional[int]) -> str:
    """
    Human-readable classification of a subprocess exit code.

    Shell exit code conventions:
      0     — success
      1     — general error (grep: no match found)
      2     — shell/usage error
      126   — command not executable (permissions)
      127   — command not found
      128+N — killed by signal N (137=SIGKILL, 143=SIGTERM)
      None  — killed by timeout (no exit code)
    """
    if exit_code is None:
        return "killed_by_timeout"
    if exit_code == 0:
        return "success"
    if exit_code == 1:
        return "no_match_or_general_error"
    if exit_code == 2:
        return "shell_usage_error"
    if exit_code == 126:
        return "not_executable"
    if exit_code == 127:
        return "command_not_found"
    if exit_code > 128:
        signal_num = exit_code - 128
        signal_name = {
            9:  "SIGKILL",
            15: "SIGTERM",
            11: "SIGSEGV",
            6:  "SIGABRT",
            13: "SIGPIPE",
        }.get(signal_num, f"signal_{signal_num}")
        return f"killed_by_{signal_name}"
    return f"exit_{exit_code}"


# ═════════════════════════════════════════════════════════════════════════════
# RECIPE RESOLUTION AND VALIDATION
#
# These are thin wrappers around registry.get_recipe() and validator.check().
# They exist to provide structured logging and error context at the
# pipeline.py level. They do not add business logic — that lives in
# registry.py and validator.py respectively.
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_recipe(
    topology_class: str,
    run_id: str,
    registry: object,
) -> RecipeMount:
    """
    Resolve a recipe for the given topology class via the registry.

    The registry owns fallback logic:
      1. Exact match for topology_class
      2. Parent class fallback (e.g. NEWS_ARTICLE_PAYWALLED → NEWS_ARTICLE)
      3. GENERIC_HTML fallback
      4. Never returns None — always resolves to something

    On failure: the registry raises RecipeMountError (RecipeNotFound,
    RecipeHashMismatch, etc.). These are hard stops. pipeline.py re-raises
    them to the caller.

    The registry parameter is passed as object to avoid importing
    recipes.registry at module level. pipeline.py receives the registry
    from the caller or from a module-level reference.
    """
    logger.debug(
        "Resolving recipe: topology=%s run_id=%s",
        topology_class,
        run_id,
    )

    # The registry's get_recipe() method returns a RecipeMount.
    # Type: ignore because we accept the registry as object to avoid
    # circular import issues at module level.
    recipe_mount: RecipeMount = registry.get_recipe(topology_class)  # type: ignore[attr-defined]

    logger.debug(
        "Recipe resolved: topology=%s → path=%s hash=%s hardcoded=%s",
        topology_class,
        recipe_mount.recipe_path,
        recipe_mount.recipe_hash[:8],
        recipe_mount.is_hardcoded,
    )

    return recipe_mount


def _validate_recipe(
    recipe_mount: RecipeMount,
    validator_check: Callable[[RecipeMount], None],
) -> None:
    """
    Validate the recipe for safe kernel execution via validator.check().

    Hardcoded recipes: validator returns immediately (skip all gates).
    Compiler-generated recipes: pass through injection, allowlist, line
    count, and dry-run gates.

    On failure: validator raises RecipeInjectionAttempt or
    RecipeStructuralViolation or RecipeDryRunFailure. All are subclasses
    of RecipeMountError. All are hard stops. pipeline.py re-raises them.
    """
    logger.debug(
        "Validating recipe: topology=%s path=%s hardcoded=%s",
        recipe_mount.topology_class,
        recipe_mount.recipe_path,
        recipe_mount.is_hardcoded,
    )

    # validator.check() returns None on success, raises on failure.
    validator_check(recipe_mount)  # type: ignore[operator]

    logger.debug(
        "Recipe validated: topology=%s hash=%s",
        recipe_mount.topology_class,
        recipe_mount.recipe_hash[:8],
    )


# ═════════════════════════════════════════════════════════════════════════════
# WARM CONTAINER MANAGEMENT
#
# Decides whether to reuse the warm container or spawn fresh.
# The warm container is a module-level singleton that persists across
# invocations within the same Python process.
# ═════════════════════════════════════════════════════════════════════════════

async def _prepare_warm_container(
    recipe_mount: RecipeMount,
    config: PipelineConfig,
) -> None:
    """
    Prepare the warm container for the upcoming invocation.

    Decision logic:
      1. If warm container is compatible (same topology, same recipe,
         alive, not idle too long) → reuse. No action needed.
      2. If warm container exists but is NOT compatible → kill it.
         The next invocation will spawn fresh.
      3. If no warm container → nothing to do.

    This function only handles the teardown side. Spawning is done by
    _run_with_retry() during the actual subprocess lifecycle.
    """
    if not _warm.is_warm:
        return

    if _warm.is_compatible(
        recipe_mount.topology_class,
        recipe_mount.recipe_hash,
        config.warm_container_max_idle_s,
    ):
        _warm.record_reuse()
        logger.debug(
            "Warm container reuse: topology=%s hash=%s reuse_count=%d",
            recipe_mount.topology_class,
            recipe_mount.recipe_hash[:8],
            _warm.reuse_count,
        )
        return

    # Incompatible — kill and clear.
    reason: str
    if _warm.topology_class != recipe_mount.topology_class:
        reason = (
            f"topology change: {_warm.topology_class} → "
            f"{recipe_mount.topology_class}"
        )
    elif _warm.recipe_hash != recipe_mount.recipe_hash:
        reason = (
            f"recipe hash change: "
            f"{_warm.recipe_hash[:8] if _warm.recipe_hash else '?'} → "
            f"{recipe_mount.recipe_hash[:8]}"
        )
    elif _warm.idle_seconds > config.warm_container_max_idle_s:
        reason = f"idle {_warm.idle_seconds:.1f}s > {config.warm_container_max_idle_s}s"
    else:
        reason = "process died"

    logger.info(
        "Killing warm container: %s. warm_state=%s",
        reason,
        json.dumps(_warm.diagnostic_dict(), default=str),
    )
    await _warm.kill()


# ═════════════════════════════════════════════════════════════════════════════
# GRACEFUL DEGRADATION
#
# The unified path for returning empty KernelOutput on any non-hard-stop
# failure. Every catch clause that decides to degrade rather than re-raise
# goes through this function.
# ═════════════════════════════════════════════════════════════════════════════

def _degrade_to_empty(
    kernel_input: KernelInput,
    recipe_mount: Optional[RecipeMount],
    elapsed_ms: float,
    reason: str,
) -> KernelOutput:
    """
    Construct the empty KernelOutput that represents graceful degradation.

    The AXIOM graph receives this as a signal that the kernel could not
    extract useful content from this page. The graph continues — it does
    not stop the run. The extraction is simply marked as empty.

    The reason string is logged but NOT included in the KernelOutput
    contract — the graph does not need to know why the extraction was
    empty, only that it was.
    """
    recipe_path = recipe_mount.recipe_path if recipe_mount else "unknown"

    logger.warning(
        "Degrading to empty output: topology=%s run_id=%s "
        "recipe=%s reason=%s latency_ms=%.1f",
        kernel_input.topology_class,
        kernel_input.run_id,
        recipe_path,
        reason,
        elapsed_ms,
    )

    return make_empty_kernel_output(
        run_id=kernel_input.run_id,
        topology_class=kernel_input.topology_class,
        recipe_used=recipe_path,
        raw_byte_count=kernel_input.raw_byte_count,
        latency_ms=elapsed_ms,
    )


# ═════════════════════════════════════════════════════════════════════════════
# EXCEPTION HANDLING
#
# The central error handler for execute(). Classifies every exception caught
# during the pipeline lifecycle and decides: re-raise (hard stop) or degrade
# (return empty KernelOutput).
#
# The contract is absolute:
#   - RecipeMountError family → re-raise. AXIOM graph is informed.
#   - RecipeInjectionAttempt  → re-raise. Forensic review required.
#   - Everything else         → degrade. Graph continues.
#
# No exception type is ambiguous. is_hard_stop() from exceptions.py is the
# authoritative classifier.
# ═════════════════════════════════════════════════════════════════════════════

def _handle_exception(
    exc: BaseException,
    kernel_input: KernelInput,
    recipe_mount: Optional[RecipeMount],
    elapsed_ms: float,
) -> KernelOutput:
    """
    Classify an exception and either re-raise or degrade to empty output.

    Hard stops: RecipeMountError, RecipeInjectionAttempt, and any
    non-KernelException BaseException. These propagate to the caller.

    Soft failures: SubprocessTimeout, ContainerSpawnError,
    StdinEncodingError, EmptyExtractionError, OutputMeasurementError,
    and all other KernelException subclasses. These degrade to empty
    KernelOutput.

    This function ALWAYS either raises or returns KernelOutput. It never
    returns None. It never swallows exceptions silently — every exception
    is logged with full classification context before the decision is made.
    """
    classification = classify(exc)

    # Log the full classification for every exception.
    log_method = logger.critical if classification["is_hard_stop"] else logger.warning

    log_method(
        "Pipeline exception: %s [%s] hard_stop=%s security=%s "
        "run_id=%s topology=%s — %s",
        classification["exception_class"],
        classification["exception_code"],
        classification["is_hard_stop"],
        classification["is_security_event"],
        kernel_input.run_id,
        kernel_input.topology_class,
        exc,
    )

    # Emit audit event for security exceptions.
    if classification["is_security_event"]:
        _emit_audit_event(
            run_id=kernel_input.run_id,
            event_type="injection_blocked" if isinstance(exc, RecipeInjectionAttempt) else "hash_mismatch",
            topology_class=kernel_input.topology_class,
            recipe_hash=recipe_mount.recipe_hash if recipe_mount else RecipeHash("0" * 64),
            detail=str(exc),
            severity="critical",
        )

    # ── Hard stop → re-raise ──────────────────────────────────────────
    if is_hard_stop(exc):
        raise exc

    # ── Soft failure → degrade ────────────────────────────────────────
    return _degrade_to_empty(
        kernel_input,
        recipe_mount,
        elapsed_ms,
        reason=f"{classification['exception_class']}: {exc}",
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — execute()
#
# The only public function in pipeline.py that the AXIOM graph's TAG layer
# calls. Everything above this point is internal implementation.
#
# execute() is async because the subprocess lifecycle uses asyncio.
# execute_sync() provides a synchronous wrapper for callers without an
# existing event loop.
# ═════════════════════════════════════════════════════════════════════════════

async def execute(
    kernel_input: KernelInput,
    *,
    registry: object,
    validator_check: Callable[[RecipeMount], None],
    config: Optional[PipelineConfig] = None,
) -> KernelOutput:
    """
    Execute the complete extraction pipeline for one page.

    This is the principal entry point for the signal kernel. TAG's Python
    layer calls this with a KernelInput and receives a KernelOutput. The
    entire subprocess lifecycle — recipe resolution, validation, spawn,
    stdin piping, timeout enforcement, stdout capture, telemetry emission —
    is managed internally.

    Parameters
    ----------
    kernel_input : KernelInput
        The page to extract. Must be a fully constructed KernelInput with
        valid raw_content, topology_class, and all other fields.

    registry : object
        The recipe registry instance. Must have a get_recipe(topology_class)
        method that returns a RecipeMount. Passed as object to avoid
        circular import at module level.

    validator_check : callable
        The validator.check function. Must accept a RecipeMount and return
        None on success or raise RecipeMountError on failure. Passed as
        callable to avoid circular import at module level.

    config : PipelineConfig, optional
        Pipeline configuration. If None, uses the module default.

    Returns
    -------
    KernelOutput
        The extraction result. extraction_empty=True on any soft failure.
        clean_signal contains the extracted text on success.

    Raises
    ------
    RecipeMountError
        The recipe could not be resolved, validated, or mounted. Hard stop.
        Requires human review. Includes: RecipeNotFound, RecipeHashMismatch,
        RecipeStructuralViolation, RecipeDryRunFailure.

    RecipeInjectionAttempt
        The validator detected a shell injection pattern in the recipe.
        Hard stop. Requires forensic review. Full recipe content is in
        the InjectionAuditRecord.

    No other exception is raised. All other failure modes degrade to
    KernelOutput(extraction_empty=True). The AXIOM graph continues.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    timer = _MonotonicTimer()
    timer.mark("execute_start")

    # ── Create invocation context ─────────────────────────────────────
    ctx = _InvocationContext(kernel_input, config)
    ctx.timer.mark("execute_start")

    recipe_mount: Optional[RecipeMount] = None
    lifecycle: Optional[ContainerLifecycle] = None

    try:
        # ══════════════════════════════════════════════════════════════
        # PHASE 1 — RESOLVE
        #
        # Call registry.get_recipe() to find the recipe for this topology
        # class. The registry handles fallback logic (parent class,
        # GENERIC_HTML). On failure: RecipeMountError propagates.
        # ══════════════════════════════════════════════════════════════

        ctx.enter_phase("resolve")
        recipe_mount = _resolve_recipe(
            kernel_input.topology_class,
            kernel_input.run_id,
            registry,
        )
        ctx.recipe_mount = recipe_mount
        ctx.exit_phase("resolve")

        logger.debug(
            "Phase 1 RESOLVE complete: topology=%s → recipe=%s "
            "hardcoded=%s resolve_ms=%.1f",
            kernel_input.topology_class,
            recipe_mount.recipe_hash[:8],
            recipe_mount.is_hardcoded,
            ctx.elapsed_ms("resolve", "resolve"),
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 2 — VALIDATE
        #
        # Call validator.check() to verify the recipe is safe to execute.
        # Hardcoded recipes: validator returns immediately (skip all gates).
        # Compiler recipes: injection, allowlist, line count, dry-run.
        # On failure: RecipeMountError or RecipeInjectionAttempt propagates.
        # ══════════════════════════════════════════════════════════════

        ctx.enter_phase("validate")
        _validate_recipe(recipe_mount, validator_check)
        ctx.exit_phase("validate")

        logger.debug(
            "Phase 2 VALIDATE complete: topology=%s validate_ms=%.1f",
            kernel_input.topology_class,
            ctx.elapsed_ms("validate", "validate"),
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 3 — PREPARE
        #
        # Pre-flight the recipe file. Encode raw_content for stdin.
        # Prepare the warm container.
        # On preflight failure: RecipeMountError → hard stop.
        # On encoding failure: StdinEncodingError → degrade to empty.
        # ══════════════════════════════════════════════════════════════

        ctx.enter_phase("prepare")

        # Pre-flight: verify the recipe file is still present and valid
        # between validation and execution. Small window, real risk.
        _preflight_recipe(recipe_mount)

        stdin_bytes = _encode_stdin(kernel_input)
        await _prepare_warm_container(recipe_mount, config)

        ctx.exit_phase("prepare")

        logger.debug(
            "Phase 3 PREPARE complete: stdin_bytes=%d prepare_ms=%.1f",
            len(stdin_bytes),
            ctx.elapsed_ms("prepare", "prepare"),
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 4 — EXECUTE
        #
        # Run the subprocess. Write stdin. Read stdout/stderr. Enforce
        # timeout. Handle spawn failure with retry.
        # On timeout: SubprocessTimeout → degrade to empty.
        # On spawn failure: ContainerSpawnError → degrade to empty.
        # ══════════════════════════════════════════════════════════════

        ctx.enter_phase("subprocess")

        result = await _run_with_retry(
            recipe_mount,
            stdin_bytes,
            config,
            kernel_input,
        )

        ctx.subprocess_result = result
        ctx.exit_phase("subprocess")

        logger.debug(
            "Phase 4 EXECUTE complete: pid=%s exit_code=%s timed_out=%s "
            "stdout_bytes=%d stderr_bytes=%d subprocess_ms=%.1f",
            result.pid,
            result.exit_code,
            result.timed_out,
            result.stdout_size,
            result.stderr_size,
            ctx.elapsed_ms("subprocess", "subprocess"),
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 5 — CAPTURE
        #
        # Build KernelOutput from subprocess results. Build
        # ContainerLifecycle for telemetry. Log stderr diagnostics.
        # ══════════════════════════════════════════════════════════════

        ctx.enter_phase("capture")

        # Log stderr before building output — diagnostic context.
        _log_stderr(kernel_input, recipe_mount, result)

        # Build ContainerLifecycle for telemetry.
        lifecycle = _build_container_lifecycle(
            kernel_input, recipe_mount, result,
        )
        ctx.lifecycle = lifecycle

        # Build the principal output.
        kernel_output = _build_kernel_output(
            kernel_input, recipe_mount, result,
        )
        ctx.kernel_output = kernel_output

        ctx.exit_phase("capture")

        # Handle specific subprocess outcomes that warrant audit events.
        if result.timed_out:
            _emit_audit_event(
                run_id=kernel_input.run_id,
                event_type="timeout",
                topology_class=kernel_input.topology_class,
                recipe_hash=recipe_mount.recipe_hash,
                detail=(
                    f"Subprocess timed out after {config.subprocess_timeout_ms}ms. "
                    f"raw_bytes={kernel_input.raw_byte_count} "
                    f"source_url={kernel_input.source_url}"
                ),
                severity="warn",
            )

        # Handle non-zero exit code (when not timed out).
        if not result.timed_out and result.exit_code is not None and result.exit_code != 0:
            exit_classification = _classify_exit_code(result.exit_code)
            logger.debug(
                "Non-zero exit code: %d (%s) topology=%s recipe=%s",
                result.exit_code,
                exit_classification,
                kernel_input.topology_class,
                recipe_mount.recipe_hash[:8],
            )

            # Exit code 1 from grep means "no match" — this is normal for
            # pages that have no content in the expected zone. Not an error.
            # Exit codes ≥ 2 indicate a recipe or shell problem.
            if result.exit_code >= 2:
                _emit_audit_event(
                    run_id=kernel_input.run_id,
                    event_type="stderr_non_empty",
                    topology_class=kernel_input.topology_class,
                    recipe_hash=recipe_mount.recipe_hash,
                    detail=(
                        f"Recipe exited with code {result.exit_code} "
                        f"({exit_classification}). "
                        f"has_stdout={result.has_stdout} "
                        f"has_stderr={result.has_stderr}"
                    ),
                    severity="warn",
                )

        logger.debug(
            "Phase 5 CAPTURE complete: empty=%s clean_bytes=%d "
            "capture_ms=%.1f",
            kernel_output.extraction_empty,
            kernel_output.clean_byte_count,
            ctx.elapsed_ms("capture", "capture"),
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 6 — TELEMETRY
        #
        # Emit PipelineTelemetry. This is not optional — it is the data
        # product that validates whether kernel integration performs as
        # designed. But it must never crash the pipeline.
        # ══════════════════════════════════════════════════════════════

        _emit_telemetry(kernel_output, lifecycle, recipe_mount)

        total_ms = ctx.total_elapsed_ms()

        # ── Record health statistics ──────────────────────────────────
        if result.timed_out:
            _health.record_timeout(total_ms)
            _topology_health.record_timeout(kernel_input.topology_class, total_ms)
            ctx.set_outcome("timeout")
        elif kernel_output.extraction_empty:
            _health.record_empty(total_ms)
            _topology_health.record_empty(kernel_input.topology_class, total_ms)
            ctx.set_outcome("empty")

            # ── EmptyExtractionError handling ─────────────────────────
            # The subprocess completed but produced zero bytes of output.
            # This is logged as an EmptyExtractionError for the caller's
            # structured error handling, but it is NOT a hard stop —
            # pipeline.py catches it internally and returns the empty
            # KernelOutput. The AXIOM graph continues.
            #
            # The error is raised to trigger the standard exception
            # logging path, then caught immediately. This ensures
            # EmptyExtractionError appears in the structured log stream
            # alongside SubprocessTimeout and other lifecycle events.
            stderr_text = (
                result.stderr_bytes.decode("utf-8", errors="replace")
                if result.has_stderr else None
            )
            try:
                raise EmptyExtractionError(
                    run_id=kernel_input.run_id,
                    topology_class=kernel_input.topology_class,
                    recipe_hash=recipe_mount.recipe_hash,
                    source_url=kernel_input.source_url,
                    stderr_content=stderr_text,
                    latency_ms=total_ms,
                )
            except EmptyExtractionError as empty_exc:
                # Caught immediately — this is expected. Log it and
                # continue with the empty KernelOutput already built.
                logger.info(
                    "Empty extraction: topology=%s source=%s "
                    "has_stderr=%s latency_ms=%.1f consecutive_empty=%d",
                    kernel_input.topology_class,
                    kernel_input.source_url,
                    empty_exc.has_stderr,
                    total_ms,
                    _topology_health.consecutive_empty_count(
                        kernel_input.topology_class
                    ),
                )
        else:
            _health.record_success(total_ms)
            _topology_health.record_success(kernel_input.topology_class, total_ms)
            ctx.set_outcome("success")

        logger.info(
            "Pipeline complete: topology=%s empty=%s "
            "raw=%d clean=%d reduction=%.1f%% "
            "total_ms=%.1f run_id=%s",
            kernel_input.topology_class,
            kernel_output.extraction_empty,
            kernel_output.raw_byte_count,
            kernel_output.clean_byte_count,
            kernel_output.token_reduction_pct * 100,
            total_ms,
            kernel_input.run_id,
        )

        # ══════════════════════════════════════════════════════════════
        # PHASE 7 — RETURN
        # ══════════════════════════════════════════════════════════════

        return kernel_output

    except (RecipeMountError, RecipeInjectionAttempt):
        # ── Hard stops → re-raise directly ────────────────────────────
        # These two exception families are the ONLY things pipeline.py
        # allows to propagate to its caller. Everything else is caught
        # below. This catch clause is first to ensure these are never
        # accidentally caught by the broad except below.
        _health.record_hard_stop()
        ctx.set_outcome("hard_stop")

        logger.error(
            "Hard stop in pipeline: %s",
            ctx.to_summary_line(),
        )
        raise

    except StdinEncodingError as exc:
        # ── Encoding failure → degrade ────────────────────────────────
        _health.record_encoding_failure()
        ctx.set_outcome("encoding_failure")
        elapsed = ctx.total_elapsed_ms()

        logger.warning(
            "Stdin encoding failure: %s — %s",
            ctx.to_summary_line(),
            exc,
        )

        return _degrade_to_empty(
            kernel_input, recipe_mount, elapsed,
            reason=f"StdinEncodingError: {exc}",
        )

    except ContainerSpawnError as exc:
        # ── Spawn exhaustion → degrade ────────────────────────────────
        _health.record_spawn_failure()
        ctx.set_outcome("spawn_failure")
        elapsed = ctx.total_elapsed_ms()

        _emit_audit_event(
            run_id=kernel_input.run_id,
            event_type="spawn_failure",
            topology_class=kernel_input.topology_class,
            recipe_hash=recipe_mount.recipe_hash if recipe_mount else RecipeHash("0" * 64),
            detail=str(exc),
            severity="warn",
        )

        return _degrade_to_empty(
            kernel_input, recipe_mount, elapsed,
            reason=f"ContainerSpawnError: {exc}",
        )

    except KernelException as exc:
        # ── Known kernel exceptions → degrade gracefully ──────────────
        elapsed = ctx.total_elapsed_ms()
        ctx.set_outcome("kernel_exception")

        # Emit telemetry for the failure if we have enough context.
        if recipe_mount is not None and lifecycle is not None:
            empty_output = _degrade_to_empty(
                kernel_input, recipe_mount, elapsed,
                reason=f"{type(exc).__name__}: {exc}",
            )
            _emit_telemetry(empty_output, lifecycle, recipe_mount)
            _health.record_empty(elapsed)
            return empty_output

        _health.record_empty(elapsed)
        return _handle_exception(exc, kernel_input, recipe_mount, elapsed)

    except Exception as exc:
        # ── Unexpected exceptions → classify and handle ───────────────
        # This catches anything we did not anticipate: BrokenPipeError,
        # ConnectionResetError, RuntimeError from asyncio, etc.
        # The classification helpers in exceptions.py determine whether
        # to re-raise or degrade.
        elapsed = ctx.total_elapsed_ms()
        ctx.set_outcome("unexpected_exception")

        logger.critical(
            "Unexpected exception in pipeline: %s: %s. "
            "run_id=%s topology=%s invocation_ctx=%s",
            type(exc).__name__,
            exc,
            kernel_input.run_id,
            kernel_input.topology_class,
            json.dumps(ctx.to_diagnostic_dict(), default=str),
            exc_info=True,
        )

        _health.record_empty(elapsed)
        return _handle_exception(exc, kernel_input, recipe_mount, elapsed)


def execute_sync(
    kernel_input: KernelInput,
    *,
    registry: object,
    validator_check: Callable[[RecipeMount], None],
    config: Optional[PipelineConfig] = None,
) -> KernelOutput:
    """
    Synchronous wrapper around execute() for callers without an event loop.

    Creates a new event loop, runs execute() to completion, and returns
    the result. The event loop is closed after the call.

    This is the entry point for simple command-line testing and for callers
    in TAG's Python layer that do not manage an asyncio event loop.

    For production use with high throughput, prefer calling execute()
    directly within an existing event loop.

    Error handling contract is identical to execute():
      - RecipeMountError → raises
      - RecipeInjectionAttempt → raises
      - Everything else → returns KernelOutput(extraction_empty=True)
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(
            execute(
                kernel_input,
                registry=registry,
                validator_check=validator_check,
                config=config,
            )
        )
    finally:
        # Clean shutdown: cancel any remaining tasks, close the loop.
        try:
            pending = asyncio.all_tasks(loop)
            for task in pending:
                task.cancel()
            if pending:
                loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
        except Exception: # noqa
            pass
        finally:
            loop.close()


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE SHUTDOWN
#
# Cleanup function called when TAG's Python layer is shutting down.
# Kills the warm container if one exists. Called from a shutdown hook
# or explicitly by the caller.
# ═════════════════════════════════════════════════════════════════════════════

async def shutdown() -> None:
    """
    Gracefully shut down the pipeline.

    Kills the warm container if one exists. Logs the final warm container
    and health statistics.

    Called by TAG's shutdown sequence. Safe to call multiple times.
    """
    if _warm.is_warm:
        logger.info(
            "Pipeline shutdown: killing warm container. final_stats=%s",
            json.dumps(_warm.diagnostic_dict(), default=str),
        )
        await _warm.kill()
    else:
        logger.debug("Pipeline shutdown: no warm container to kill.")

    logger.info(
        "Pipeline shutdown complete. total_spawns=%d total_reuses=%d "
        "health=%s",
        _warm.spawn_count,
        _warm.reuse_count,
        json.dumps(_health.to_health_dict(), default=str),
    )


def shutdown_sync() -> None:
    """Synchronous wrapper for shutdown()."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(shutdown())
    finally:
        loop.close()


# ═════════════════════════════════════════════════════════════════════════════
# PIPELINE INITIALIZATION
#
# Called once at TAG startup before any invocation of execute(). Performs
# one-time pre-flight checks that validate the execution environment:
# shell exists, filesystem is healthy, and the pipeline is ready to
# accept invocations.
#
# If initialization fails, the pipeline is not ready to serve traffic.
# TAG must handle this at the startup level — either retry, alert, or
# fall back to conventional (non-kernel) extraction.
# ═════════════════════════════════════════════════════════════════════════════

_initialized: bool = False


def initialize(config: Optional[PipelineConfig] = None) -> None:
    """
    One-time pipeline initialization. Must be called before execute().

    Performs pre-flight checks:
      1. Verify the shell executable exists and is executable
      2. Verify the subprocess environment is sane
      3. Log the pipeline configuration for audit trail

    On failure: raises RecipeMountError. TAG startup catches this and
    decides how to proceed.

    Safe to call multiple times — subsequent calls are no-ops.
    """
    global _initialized

    if _initialized:
        logger.debug("Pipeline already initialized. Skipping.")
        return

    if config is None:
        config = _DEFAULT_CONFIG

    logger.info(
        "Initializing pipeline: timeout_ms=%d spawn_timeout_ms=%d "
        "max_attempts=%d shell=%s",
        config.subprocess_timeout_ms,
        config.spawn_timeout_ms,
        config.max_spawn_attempts,
        config.shell_executable,
    )

    # ── Verify shell executable ───────────────────────────────────────
    _preflight_shell(config)
    logger.info("Shell executable verified: %s", config.shell_executable)

    # ── Verify subprocess environment ─────────────────────────────────
    env = _build_subprocess_env()
    logger.debug(
        "Subprocess environment: %s",
        json.dumps(env, default=str),
    )

    # Verify PATH contains at least one directory that exists.
    path_dirs = env.get("PATH", "").split(":")
    existing_dirs = [d for d in path_dirs if Path(d).is_dir()]
    if not existing_dirs:
        logger.warning(
            "No PATH directories exist in subprocess environment: %s. "
            "Recipe commands (grep, sed, awk) may not be found.",
            path_dirs,
        )
    else:
        logger.debug(
            "Subprocess PATH verified: %d/%d directories exist: %s",
            len(existing_dirs),
            len(path_dirs),
            existing_dirs,
        )

    # ── Verify required commands are available ─────────────────────────
    # Check that at least grep, sed, and awk are findable in the PATH.
    # These are the minimum commands every recipe requires.
    critical_commands = ("grep", "sed", "awk")
    for cmd in critical_commands:
        found = False
        for d in existing_dirs:
            cmd_path = Path(d) / cmd
            if cmd_path.exists() and os.access(str(cmd_path), os.X_OK):
                found = True
                break
        if not found:
            logger.warning(
                "Critical command %r not found in subprocess PATH. "
                "Recipes requiring this command will fail.",
                cmd,
            )
        else:
            logger.debug("Command %r found in PATH.", cmd)

    _initialized = True
    logger.info("Pipeline initialization complete.")


def is_initialized() -> bool:
    """True if initialize() has been called successfully."""
    return _initialized


def reset() -> None:
    """
    Reset the pipeline to pre-initialization state.

    Kills the warm container and resets all health statistics.
    Used in test harnesses to ensure clean state between tests.
    Not for production use.
    """
    global _initialized, _health, _topology_health

    _warm.invalidate()
    _health = _PipelineHealth()
    _topology_health = _TopologyHealthTracker()
    _initialized = False

    logger.info("Pipeline reset to pre-initialization state.")