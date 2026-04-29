"""
signal_kernel/exceptions.py
============================
Exception taxonomy for every failure mode the kernel can produce.

This file is the consequence map of contracts.py.

The correspondence is exact and structural: for every contract defined in
contracts.py, exceptions.py defines the set of exceptions that represent
violations of that contract. The hierarchy of exceptions mirrors the
grouping of contracts. Reading exceptions.py alone, you should be able to
infer the shape of contracts.py from the exception names, their constructor
arguments, and their docstrings. Reading contracts.py alone, you should be
able to infer every failure mode from the field names and the validation
logic. They are two projections of the same architecture.

Exception construction law:
  Every exception carries structured context — run_id, topology_class,
  recipe_hash — sufficient to correlate with PipelineTelemetry in Witness.
  No exception is a bare message string. Callers do not reconstruct context
  after the fact — context is captured at the raise site, where it exists.

Raise law:
  No bare `raise Exception()` anywhere in the codebase.
  No bare `raise KernelException()` anywhere in the codebase.
  Every raise site names a concrete leaf exception.
  `from original` chaining is mandatory when wrapping lower-level errors.

Handling law:
  pipeline.py never raises to the AXIOM graph except:
    RecipeMountError  — hard stop, requires human review
    RecipeInjectionAttempt — hard stop, requires forensic review
  Every other exception is caught by pipeline.py, logged, and returned as
  KernelOutput(extraction_empty=True). The graph continues.
  `is_hard_stop` on every exception declares which side of this line it sits on.

Dependency direction: exceptions.py → contracts.py (for type imports only).
Nothing in exceptions.py has runtime logic that could fail.
If importing contracts fails, exceptions.py is also broken — they are a unit.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Imported only for type annotations — never at runtime in exception bodies.
    # Exception construction must never import contracts at the raise site in a
    # way that could itself raise, masking the original error.
    from signal_kernel.contracts import ( # noqa | type checking only
        RecipeHash,
        RunID,
        TopologyClassStr,
        ValidationOutcome,
    )

# ─────────────────────────────────────────────────────────────────────────────
# EXCEPTION CODES
# Stable short identifiers for programmatic handling and log filtering.
# Use these in structured log queries rather than matching on exception class
# names (which can be refactored) or message strings (which are not stable).
#
# Format: KERNEL_{GROUP}_{SPECIFIC}
# These are constants — they never change for a given exception class.
# ─────────────────────────────────────────────────────────────────────────────

# KernelInput violations
EC_INPUT_EMPTY            = "KERNEL_INPUT_EMPTY"
EC_INPUT_OVERFLOW         = "KERNEL_INPUT_OVERFLOW"
EC_INPUT_ENCODING         = "KERNEL_INPUT_ENCODING"
EC_INPUT_CONTENT_TYPE     = "KERNEL_INPUT_CONTENT_TYPE"
EC_INPUT_URL_INVALID      = "KERNEL_INPUT_URL_INVALID"
EC_INPUT_TOPOLOGY_INVALID = "KERNEL_INPUT_TOPOLOGY_INVALID"
EC_INPUT_RUN_ID_INVALID   = "KERNEL_INPUT_RUN_ID_INVALID"

# KernelOutput violations
EC_OUTPUT_EMPTY           = "KERNEL_OUTPUT_EMPTY"
EC_OUTPUT_TIMEOUT         = "KERNEL_OUTPUT_TIMEOUT"
EC_OUTPUT_MEASUREMENT     = "KERNEL_OUTPUT_MEASUREMENT"

# Container violations
EC_CONTAINER_SPAWN        = "KERNEL_CONTAINER_SPAWN"
EC_PROCESS_STATE          = "KERNEL_PROCESS_STATE"

# Recipe violations — hard stops
EC_RECIPE_NOT_FOUND       = "KERNEL_RECIPE_NOT_FOUND"
EC_RECIPE_HASH_MISMATCH   = "KERNEL_RECIPE_HASH_MISMATCH"
EC_RECIPE_INJECTION       = "KERNEL_RECIPE_INJECTION"
EC_RECIPE_STRUCTURAL      = "KERNEL_RECIPE_STRUCTURAL"
EC_RECIPE_DRY_RUN         = "KERNEL_RECIPE_DRY_RUN"

# Registry violations
EC_REGISTRY_CORRUPT       = "KERNEL_REGISTRY_CORRUPT"
EC_REGISTRY_OVERWRITE     = "KERNEL_REGISTRY_OVERWRITE"
EC_MANIFEST_CORRUPT       = "KERNEL_MANIFEST_CORRUPT"

# Checkpoint violations
EC_CHECKPOINT_WRITE       = "KERNEL_CHECKPOINT_WRITE"
EC_CHECKPOINT_INTEGRITY   = "KERNEL_CHECKPOINT_INTEGRITY"
EC_CHECKPOINT_ROTATION    = "KERNEL_CHECKPOINT_ROTATION"
EC_CROND_DEAD             = "KERNEL_CROND_DEAD"
EC_CHECKPOINT_CORRUPT     = "KERNEL_CHECKPOINT_CORRUPT"  # monitor-layer hard stop

# Restore violations
EC_RESTORE_FAILURE        = "KERNEL_RESTORE_FAILURE"
EC_RESTORE_EXHAUSTED      = "KERNEL_RESTORE_EXHAUSTED"
EC_RESTORE_PARTIAL        = "KERNEL_RESTORE_PARTIAL"

# Feedback / telemetry / audit violations
EC_FEEDBACK_EMISSION      = "KERNEL_FEEDBACK_EMISSION"
EC_TELEMETRY_EMISSION     = "KERNEL_TELEMETRY_EMISSION"
EC_AUDIT_EMISSION         = "KERNEL_AUDIT_EMISSION"

# Interface / daemon violations
EC_DAEMON_QUERY           = "KERNEL_DAEMON_QUERY"
EC_DAEMON_RESPONSE        = "KERNEL_DAEMON_RESPONSE"

# Internal invariant violations — should never appear in production
EC_CONTRACT_VIOLATION     = "KERNEL_CONTRACT_VIOLATION"

# ── Topology layer exception codes ──────────────────────────────────────────
# Format: TOPOLOGY_{GROUP}_{SPECIFIC}

# Classifier
EC_TOPO_CLASSIFIER_NOT_INIT   = "TOPOLOGY_CLASSIFIER_NOT_INIT"
EC_TOPO_CONFIDENCE_TOO_LOW    = "TOPOLOGY_CONFIDENCE_TOO_LOW"
EC_TOPO_WINDOW_TOO_SMALL      = "TOPOLOGY_WINDOW_TOO_SMALL"

# Parser / recipe
EC_TOPO_RECIPE_COMPILE_FAILED = "TOPOLOGY_RECIPE_COMPILE_FAILED"
EC_TOPO_RECIPE_VERSION_CONFLICT = "TOPOLOGY_RECIPE_VERSION_CONFLICT"
EC_TOPO_WLP_QUERY_FAILED      = "TOPOLOGY_WLP_QUERY_FAILED"

# Sanitizer
EC_TOPO_SANITIZER_INPUT       = "TOPOLOGY_SANITIZER_INPUT"

# Surprise detector
EC_TOPO_SURPRISE_HISTORY      = "TOPOLOGY_SURPRISE_HISTORY"

# Phantom
EC_TOPO_PHANTOM_FETCH_FAILED  = "TOPOLOGY_PHANTOM_FETCH_FAILED"
EC_TOPO_PHANTOM_RENDER_TIMEOUT = "TOPOLOGY_PHANTOM_RENDER_TIMEOUT"
EC_TOPO_PHANTOM_FRICTION      = "TOPOLOGY_PHANTOM_FRICTION"

# Index daemon
EC_TOPO_GRADIENT_STEP_FAILED  = "TOPOLOGY_GRADIENT_STEP_FAILED"
EC_TOPO_PHASE_MMAP_CORRUPTED  = "TOPOLOGY_PHASE_MMAP_CORRUPTED"

# Bus
EC_TOPO_BUS_SUBSCRIPTION      = "TOPOLOGY_BUS_SUBSCRIPTION"
EC_TOPO_EVENT_DISPATCH        = "TOPOLOGY_EVENT_DISPATCH"
EC_TOPO_KAFKA_UNAVAILABLE     = "TOPOLOGY_KAFKA_UNAVAILABLE"
EC_TOPO_EVENT_INTEGRITY       = "TOPOLOGY_EVENT_INTEGRITY"
EC_TOPO_EVENT_SCHEMA          = "TOPOLOGY_EVENT_SCHEMA"

# ── Store watchdog exception codes ───────────────────────────────────────────
# Format: WATCHDOG_{SPECIFIC}
# These are the single source of truth — store_watchdog.py imports them from
# here rather than defining its own local strings.  Log analysis tools filter
# on these codes; do not rename them.

EC_WATCHDOG_STARTUP         = "WATCHDOG_STARTUP_FAILED"
EC_WATCHDOG_INOTIFY_EXHAUST = "WATCHDOG_INOTIFY_FD_EXHAUSTED"
EC_WATCHDOG_HANDLER_TIMEOUT = "WATCHDOG_HANDLER_TIMEOUT"
EC_WATCHDOG_DUPLICATE_REG   = "WATCHDOG_DUPLICATE_REGISTRATION"
EC_WATCHDOG_POST_START_REG  = "WATCHDOG_POST_START_REGISTRATION"
EC_WATCHDOG_CIRCUIT_OPEN    = "WATCHDOG_HANDLER_CIRCUIT_OPEN"
EC_WATCHDOG_GHOST_EVENT     = "WATCHDOG_GHOST_EVENT_SUPPRESSED"
EC_WATCHDOG_LOOP_ERROR      = "WATCHDOG_WATCH_LOOP_ERROR"

# ── Crawler layer exception codes ────────────────────────────────────────────
# Format: CRAWLER_{COMPONENT}_{SPECIFIC}

# Fetcher
EC_CRAWLER_FETCH_FAILED       = "CRAWLER_FETCH_FAILED"
EC_CRAWLER_TOR_UNAVAILABLE    = "CRAWLER_TOR_UNAVAILABLE"
EC_CRAWLER_PLAYWRIGHT_CRASH   = "CRAWLER_PLAYWRIGHT_CRASH"
EC_CRAWLER_RATE_LIMIT_HIT     = "CRAWLER_RATE_LIMIT_VIOLATED"
EC_CRAWLER_STAGING_WRITE      = "CRAWLER_STAGING_WRITE_FAILED"
EC_CRAWLER_MANIFEST_EXHAUSTED = "CRAWLER_MANIFEST_EXHAUSTED"

# Frontier
EC_CRAWLER_FRONTIER_ERROR     = "CRAWLER_FRONTIER_DB_ERROR"

# Cursor
EC_CRAWLER_CURSOR_ERROR       = "CRAWLER_CURSOR_DB_ERROR"

# ── Cross-language runtime exception codes ───────────────────────────────────
# Format: AXIOM_{LAYER}_{SPECIFIC}

# Go preparser
EC_PREPARSER_DOMAIN_ANALYSIS  = "AXIOM_PREPARSER_DOMAIN_ANALYSIS_FAILED"
EC_PREPARSER_CRAWL_PLAN       = "AXIOM_PREPARSER_CRAWL_PLAN_FAILED"
EC_PREPARSER_SIGNAL_EXTRACT   = "AXIOM_PREPARSER_SIGNAL_EXTRACT_FAILED"
EC_PREPARSER_RECIPE_VALIDATE  = "AXIOM_PREPARSER_RECIPE_VALIDATE_FAILED"

# C strip/daemon layers
EC_NATIVE_STRIP_ENGINE        = "AXIOM_NATIVE_STRIP_ENGINE_FAILED"
EC_NATIVE_BATCH_RUNNER        = "AXIOM_NATIVE_BATCH_RUNNER_FAILED"
EC_NATIVE_PHASE_DAEMON        = "AXIOM_NATIVE_PHASE_DAEMON_FAILED"
EC_NATIVE_STORE_SENTINEL      = "AXIOM_NATIVE_STORE_SENTINEL_FAILED"

# CUDA/C offline layer
EC_OFFLINE_GPU_UNAVAILABLE    = "AXIOM_OFFLINE_GPU_UNAVAILABLE"
EC_OFFLINE_GRADIENT_STEP      = "AXIOM_OFFLINE_GRADIENT_STEP_FAILED"
EC_OFFLINE_WEIGHT_PUBLISH     = "AXIOM_OFFLINE_WEIGHT_PUBLISH_FAILED"

# Cold start and interface
EC_COLD_START_VALIDATION      = "AXIOM_COLD_START_VALIDATION_FAILED"
EC_COLD_START_SECURITY        = "AXIOM_COLD_START_SECURITY_FAILED"
EC_INTERFACE_COMMAND_INVALID  = "AXIOM_INTERFACE_COMMAND_INVALID"
EC_INTERFACE_DISPATCH_FAILED  = "AXIOM_INTERFACE_DISPATCH_FAILED"

# ═════════════════════════════════════════════════════════════════════════════
# BASE EXCEPTION
# All kernel exceptions inherit from KernelException.
# KernelException is never raised directly. Concrete leaf classes only.
# ═════════════════════════════════════════════════════════════════════════════

class KernelException(Exception):
    """
    Base class for all signal_kernel exceptions.

    Carries the minimum context required to correlate any exception with
    its execution context in Witness: run_id for the PipelineTelemetry
    stream, topology_class for the recipe/class it was operating on,
    and the exception_code for programmatic log filtering.

    is_hard_stop is the architectural boundary:
      True  → pipeline.py re-raises to its caller (RecipeMountError family,
               RecipeInjectionAttempt). The AXIOM graph is informed.
      False → pipeline.py catches, logs, returns make_empty_kernel_output().
               The graph continues. Kernel failure degrades gracefully.

    Never raise KernelException directly. Always raise a concrete subclass.
    """

    exception_code: str = EC_CONTRACT_VIOLATION  # overridden in every subclass
    is_hard_stop:   bool = False                  # overridden where appropriate

    def __init__(
        self,
        message: str,
        *,
        run_id:         Optional[str] = None,
        topology_class: Optional[str] = None,
        recipe_hash:    Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self._message        = message
        self._run_id         = run_id
        self._topology_class = topology_class
        self._recipe_hash    = recipe_hash
        self._raised_at      = datetime.now(timezone.utc)

    @property
    def run_id(self) -> Optional[str]:
        return self._run_id

    @property
    def topology_class(self) -> Optional[str]:
        return self._topology_class

    @property
    def recipe_hash(self) -> Optional[str]:
        return self._recipe_hash

    @property
    def recipe_hash_prefix(self) -> Optional[str]:
        """8-char prefix for log lines. Full hash is in audit records."""
        return self._recipe_hash[:8] if self._recipe_hash else None

    @property
    def raised_at(self) -> datetime:
        return self._raised_at

    def to_audit_dict(self) -> Dict[str, object]:
        """
        Flat dict for structured logging. Consumed by Witness and the
        kernel's audit trail. Suitable for json.dumps() without further
        transformation. Subclasses extend this — call super().to_audit_dict()
        and update with additional fields.
        """
        return {
            "exception_code":   self.exception_code,
            "exception_class":  type(self).__name__,
            "is_hard_stop":     self.is_hard_stop,
            "message":          self._message,
            "run_id":           self._run_id,
            "topology_class":   self._topology_class,
            "recipe_hash_prefix": self.recipe_hash_prefix,
            "raised_at":        self._raised_at.isoformat(),
        }

    def __str__(self) -> str:
        parts = [f"[{self.exception_code}]", self._message]
        if self._run_id:
            parts.append(f"run={self._run_id}")
        if self._topology_class:
            parts.append(f"topology={self._topology_class}")
        if self.recipe_hash_prefix:
            parts.append(f"recipe={self.recipe_hash_prefix}")
        return " | ".join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# KernelInput VIOLATIONS
#
# Mirrors contracts.py :: KernelInput
#
# KernelInput.__post_init__ enforces:
#   - raw_content is non-empty
#   - raw_content ≤ MAX_RAW_CONTENT_BYTES when UTF-8 encoded
#   - topology_class matches [A-Z][A-Z0-9_]{1,63}
#   - intent_vector_hash is a 64-char lowercase hex SHA-256
#   - content_type is "html" or "json"
#   - source_url begins with http:// or https://
#   - run_id is a canonical UUID4 string
#
# Distinct from these construction failures is the runtime failure that
# occurs when a valid KernelInput's raw_content cannot be encoded for the
# subprocess stdin pipe. StdinEncodingError is the runtime violation of
# the encoding guarantee implicit in content_type="html"|"json".
# ═════════════════════════════════════════════════════════════════════════════

class KernelInputError(KernelException):
    """
    Base class for all violations of the KernelInput contract.

    KernelInput.__post_init__ raises ValueError for these — pipeline.py
    wraps them into typed KernelInputError subclasses with run-time context.
    KernelInputError is never raised directly; use a concrete subclass.

    Handling: not a hard stop. pipeline.py catches, logs, and returns
    make_empty_kernel_output(). A malformed input must not halt the graph.
    """
    exception_code = EC_CONTRACT_VIOLATION  # subclasses override
    is_hard_stop   = False


class RawContentEmpty(KernelInputError):
    """
    Violation of KernelInput.raw_content non-empty invariant.

    raw_content="" means Phantom or the fetcher upstream delivered nothing.
    This should be filtered before reaching pipeline.py. If it arrives here,
    either the fetcher is broken or the empty-content filter is not running.

    Mirrors: KernelInput.__post_init__ → "raw_content must not be empty"
    Handling: pipeline.py returns make_empty_kernel_output(). Not a hard stop.
    """
    exception_code = EC_INPUT_EMPTY
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        source_url:     str,
    ) -> None:
        super().__init__(
            f"raw_content is empty — Phantom delivered no content for {source_url}. "
            "Empty content should be filtered before reaching pipeline.py.",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._source_url = source_url

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d["source_url"] = self._source_url
        return d


class RawContentOverflow(KernelInputError):
    """
    Violation of KernelInput.raw_content ≤ MAX_RAW_CONTENT_BYTES invariant.

    A page exceeding 4 MB indicates something has gone wrong upstream —
    Phantom fetched a binary asset, a CDN delivered a tarball, or the
    fetcher is not respecting content-length limits. This is not a normal
    HTML page. The kernel should not receive it.

    Mirrors: KernelInput.__post_init__ → "raw_content is {n} bytes,
             exceeding MAX_RAW_CONTENT_BYTES"
    Handling: pipeline.py returns make_empty_kernel_output(). Not a hard stop.
    """
    exception_code = EC_INPUT_OVERFLOW
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        source_url:     str,
        actual_bytes:   int,
        limit_bytes:    int,
    ) -> None:
        super().__init__(
            f"raw_content is {actual_bytes:,} bytes, exceeding the {limit_bytes:,}-byte limit. "
            f"source_url={source_url}. "
            "This indicates a non-HTML asset or a fetcher misconfiguration upstream.",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._source_url   = source_url
        self._actual_bytes = actual_bytes
        self._limit_bytes  = limit_bytes

    @property
    def overflow_bytes(self) -> int:
        """How many bytes over the limit the payload is."""
        return self._actual_bytes - self._limit_bytes

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "source_url":    self._source_url,
            "actual_bytes":  self._actual_bytes,
            "limit_bytes":   self._limit_bytes,
            "overflow_bytes": self.overflow_bytes,
        })
        return d


class StdinEncodingError(KernelInputError):
    """
    Runtime failure to encode KernelInput.raw_content for the subprocess stdin pipe.

    KernelInput construction succeeds, but when pipeline.py attempts to encode
    raw_content as UTF-8 bytes for the stdin write, the encode call raises
    UnicodeEncodeError. This is common with malformed HTML from hostile pages —
    pages that declare UTF-8 but embed bytes from another encoding, or that
    contain bare surrogates from broken JavaScript string handling.

    A valid KernelInput is a necessary but not sufficient condition for
    successful stdin encoding. This exception is the gap between them.

    Mirrors: KernelInput.raw_content runtime encoding guarantee
    Handling: pipeline.py catches, logs, returns make_empty_kernel_output().
    The kernel never sees the content. Not a hard stop.
    """
    exception_code = EC_INPUT_ENCODING
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        source_url:     str,
        encoding_error: str,
        raw_byte_count: int,
    ) -> None:
        super().__init__(
            f"raw_content could not be encoded for stdin pipe. "
            f"source_url={source_url} raw_bytes={raw_byte_count:,}. "
            f"encoding_error={encoding_error}. "
            "Common with malformed HTML from hostile pages containing non-UTF-8 bytes.",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._source_url     = source_url
        self._encoding_error = encoding_error
        self._raw_byte_count = raw_byte_count

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "source_url":     self._source_url,
            "encoding_error": self._encoding_error,
            "raw_byte_count": self._raw_byte_count,
        })
        return d


class InvalidTopologyClass(KernelInputError):
    """
    Violation of KernelInput.topology_class format invariant.

    topology_class must match [A-Z][A-Z0-9_]{1,63}. If this arrives malformed,
    the classifier upstream (index_daemon.py or topology_parser.py) has
    produced a class name that does not conform to the naming contract.
    This is an internal consistency error in TAG, not a user input error.

    Mirrors: KernelInput.__post_init__ → _validate_topology_class
    Handling: pipeline.py returns make_empty_kernel_output(). Not a hard stop.
    """
    exception_code = EC_INPUT_TOPOLOGY_INVALID
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:              str,
        invalid_class_value: str,
        source:              str,
    ) -> None:
        super().__init__(
            f"topology_class value {invalid_class_value!r} does not match "
            "[A-Z][A-Z0-9_]{1,63}. "
            f"source={source}. "
            "topology_parser.py or index_daemon.py produced a non-conforming class name.",
            run_id=run_id,
        )
        self._invalid_value = invalid_class_value
        self._source        = source

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "invalid_topology_class": self._invalid_value,
            "source":                 self._source,
        })
        return d


class InvalidRunID(KernelInputError):
    """
    Violation of KernelInput.run_id UUID4 format invariant.

    run_id must be a canonical lowercase UUID4 string. The only correct
    construction path is contracts.new_run_id(). If an invalid run_id
    arrives, a caller is constructing run_ids by hand — a codebase error.

    Mirrors: KernelInput.__post_init__ → _validate_run_id
    Handling: Not a hard stop, but this indicates a programming error that
    must be corrected. Pipeline.py logs at ERROR level, returns empty output.
    """
    exception_code = EC_INPUT_RUN_ID_INVALID
    is_hard_stop   = False

    def __init__(
        self,
        *,
        invalid_run_id: str,
        source:         str,
    ) -> None:
        super().__init__(
            f"run_id {invalid_run_id!r} is not a canonical lowercase UUID4 string. "
            f"source={source}. "
            "Use contracts.new_run_id() — never construct run_ids by hand.",
            run_id=None,
        )
        self._invalid_run_id = invalid_run_id
        self._source         = source

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "invalid_run_id": self._invalid_run_id,
            "source":         self._source,
        })
        return d


class InvalidContentType(KernelInputError):
    """
    Violation of KernelInput.content_type allowlist invariant.

    content_type must be exactly "html" or "json". Any other value means
    the caller is passing an unsupported content type — either a new type
    that has not been added to the ContentType Literal, or a programming
    error in the caller.

    Mirrors: KernelInput.__post_init__ → "content_type must be 'html' or 'json'"
    Handling: pipeline.py returns make_empty_kernel_output(). Not a hard stop.
    """
    exception_code = EC_INPUT_CONTENT_TYPE
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:               str,
        topology_class:       str,
        invalid_content_type: str,
        source_url:           str,
    ) -> None:
        super().__init__(
            f"content_type {invalid_content_type!r} is not a recognised value. "
            "Must be 'html' or 'json'. "
            f"source_url={source_url}. "
            "Add a new ContentType Literal in contracts.py if this type is intentional.",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._invalid_content_type = invalid_content_type
        self._source_url           = source_url

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "invalid_content_type": self._invalid_content_type,
            "source_url":           self._source_url,
        })
        return d


class InvalidSourceURL(KernelInputError):
    """
    Violation of KernelInput.source_url http(s) scheme invariant.

    source_url must begin with http:// or https://. A non-HTTP URL means
    Phantom or the fetcher produced a URL from a non-web source — e.g. a
    file:// path, a data URI, or a bare domain with no scheme. The kernel
    only processes web content.

    Mirrors: KernelInput.__post_init__ → _validate_http_url
    Handling: pipeline.py returns make_empty_kernel_output(). Not a hard stop.
    """
    exception_code = EC_INPUT_URL_INVALID
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        invalid_url:    str,
    ) -> None:
        super().__init__(
            f"source_url {invalid_url!r} does not begin with http:// or https://. "
            "The kernel only processes web content. "
            "Phantom or the fetcher produced a non-web URL — investigate upstream.",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._invalid_url = invalid_url

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d["invalid_url"] = self._invalid_url
        return d


# ═════════════════════════════════════════════════════════════════════════════
# KernelOutput VIOLATIONS
#
# Mirrors contracts.py :: KernelOutput
#
# KernelOutput represents either a successful extraction or a graceful
# degradation (extraction_empty=True). Both are valid states. The exceptions
# here represent the conditions that cause an empty KernelOutput to be
# produced — they are the causes, not the output itself.
#
# SubprocessTimeout    → grep pipeline hit the wall-clock limit
# EmptyExtractionError → subprocess completed but produced zero output
# OutputMeasurementError → internal measurement invariant violated
#                          (clean_byte_count > raw_byte_count)
# ═════════════════════════════════════════════════════════════════════════════

class SubprocessTimeout(KernelException):
    """
    The grep pipeline subprocess exceeded the configured timeout_ms threshold.

    asyncio.wait_for() wraps the communicate() call. When it fires, the
    subprocess is killed, and pipeline.py constructs make_empty_kernel_output()
    with extraction_empty=True. This exception is raised to the pipeline.py
    error handler, which catches it and handles the graceful degradation.

    Large pages (500KB+) are the common trigger — the grep pipeline stalls
    on a page that has pathological HTML structure. Test with 1MB pages.

    Mirrors: KernelOutput.latency_ms exceeding DEFAULT_SUBPROCESS_TIMEOUT_MS
             ContainerLifecycle.timed_out=True
    Handling: Caught by pipeline.py. Returns make_empty_kernel_output().
    Not a hard stop. The AXIOM graph continues.
    """
    exception_code = EC_OUTPUT_TIMEOUT
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        recipe_hash:    str,
        timeout_ms:     int,
        raw_byte_count: int,
        source_url:     str,
    ) -> None:
        super().__init__(
            f"grep pipeline timed out after {timeout_ms}ms. "
            f"raw_byte_count={raw_byte_count:,} source_url={source_url}. "
            "pipeline.py will return empty KernelOutput. The graph continues.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._timeout_ms     = timeout_ms
        self._raw_byte_count = raw_byte_count
        self._source_url     = source_url

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    @property
    def raw_byte_count(self) -> int:
        return self._raw_byte_count

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "timeout_ms":     self._timeout_ms,
            "raw_byte_count": self._raw_byte_count,
            "source_url":     self._source_url,
        })
        return d


class EmptyExtractionError(KernelException):
    """
    The grep pipeline completed successfully but produced zero bytes of output.

    NOT always an error. A paywalled page that rendered a login wall produces
    empty extraction on a NEWS_ARTICLE recipe — the recipe is not broken, the
    page has no signal in the expected structural zone. This is correct behavior.

    feedback.py tracks empty extraction rate per topology class on a rolling
    window. Sustained high rates (> EMPTY_EXTRACTION_RATE_THRESHOLD) trigger
    a FeedbackEvent with recompilation_recommended=True. A single empty
    extraction is informational. A pattern is a signal.

    Mirrors: KernelOutput.extraction_empty=True (when the subprocess exited
             cleanly with zero bytes on stdout)
    Handling: Caught by pipeline.py. Returns make_empty_kernel_output().
    Not a hard stop. Emitted as a quality observation to feedback.py.
    """
    exception_code = EC_OUTPUT_EMPTY
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        recipe_hash:    str,
        source_url:     str,
        stderr_content: Optional[str],
        latency_ms:     float,
    ) -> None:
        has_stderr = bool(stderr_content)
        super().__init__(
            f"grep pipeline produced zero bytes. "
            f"topology={topology_class} latency={latency_ms:.1f}ms "
            f"has_stderr={has_stderr} source_url={source_url}. "
            "May indicate: paywall, login wall, JS-only page, or recipe over-stripping.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._source_url     = source_url
        self._stderr_content = stderr_content
        self._latency_ms     = latency_ms

    @property
    def has_stderr(self) -> bool:
        """True if the subprocess wrote to stderr despite producing no stdout."""
        return bool(self._stderr_content)

    @property
    def stderr_content(self) -> Optional[str]:
        """Stderr output from the grep pipeline. Diagnostic, not failure signal."""
        return self._stderr_content

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "source_url":  self._source_url,
            "has_stderr":  self.has_stderr,
            "latency_ms":  round(self._latency_ms, 2),
            # stderr content is truncated in audit dict — full content goes to debug log
            "stderr_preview": (self._stderr_content[:200] if self._stderr_content else None),
        })
        return d


class OutputMeasurementError(KernelException):
    """
    Violation of KernelOutput's internal byte-count invariant.

    clean_byte_count > raw_byte_count is physically impossible — the output
    of a grep/sed pipeline cannot be larger than its input. If this condition
    is detected when constructing KernelOutput, the measurement system
    (pipeline.py's byte counting logic) has a bug.

    This exception should never appear in a correct system. Its presence
    indicates a programming error in pipeline.py's output capture, not a
    runtime failure of the extraction itself.

    Mirrors: KernelOutput.__post_init__ → "Output cannot be larger than input"
    Handling: pipeline.py catches, logs at CRITICAL level (this is a bug),
    returns make_empty_kernel_output(). Not a hard stop but must be investigated.
    """
    exception_code = EC_OUTPUT_MEASUREMENT
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:           str,
        topology_class:   str,
        recipe_hash:      str,
        raw_byte_count:   int,
        clean_byte_count: int,
    ) -> None:
        super().__init__(
            f"clean_byte_count ({clean_byte_count:,}) > raw_byte_count ({raw_byte_count:,}). "
            "Output cannot exceed input — pipeline.py byte-counting logic has a bug. "
            "This is a programming error, not a runtime failure. Investigate immediately.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._raw_byte_count   = raw_byte_count
        self._clean_byte_count = clean_byte_count

    @property
    def phantom_bytes(self) -> int:
        """How many bytes appeared from nowhere. Should always be zero."""
        return self._clean_byte_count - self._raw_byte_count

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "raw_byte_count":   self._raw_byte_count,
            "clean_byte_count": self._clean_byte_count,
            "phantom_bytes":    self.phantom_bytes,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# CONTAINER / PIPELINE LIFECYCLE VIOLATIONS
#
# Mirrors contracts.py :: ContainerLifecycle
#
# ContainerLifecycle tracks the measured outcome of one subprocess invocation.
# Its violations are the failure modes of the subprocess lifecycle itself —
# not of the content being processed.
# ═════════════════════════════════════════════════════════════════════════════

class ContainerSpawnError(KernelException):
    """
    The Alpine container failed to start within spawn_timeout_ms.

    pipeline.py retries once with a fresh spawn attempt. If the second
    attempt also fails, ContainerSpawnError is raised to pipeline.py's
    error handler, which returns make_empty_kernel_output().

    Common causes: Docker daemon under load, resource exhaustion (memory,
    file descriptors), the recipe mount path does not exist, permission
    issues on the recipe volume bind.

    Mirrors: ContainerLifecycle — subprocess fails before communicate()
    Handling: pipeline.py retries once, then catches and returns empty
    KernelOutput. Not a hard stop — the graph continues.
    """
    exception_code = EC_CONTAINER_SPAWN
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:           str,
        topology_class:   str,
        recipe_hash:      str,
        spawn_timeout_ms: int,
        attempt_number:   int,
        os_error:         Optional[str],
    ) -> None:
        super().__init__(
            f"Alpine container failed to start within {spawn_timeout_ms}ms "
            f"(attempt {attempt_number}/2). "
            f"os_error={os_error or 'none'}. "
            "Check: Docker daemon health, resource limits, recipe mount path, permissions.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._spawn_timeout_ms = spawn_timeout_ms
        self._attempt_number   = attempt_number
        self._os_error         = os_error

    @property
    def is_final_attempt(self) -> bool:
        """True if this failure is after the retry. pipeline.py gives up."""
        return self._attempt_number >= 2

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "spawn_timeout_ms": self._spawn_timeout_ms,
            "attempt_number":   self._attempt_number,
            "is_final_attempt": self.is_final_attempt,
            "os_error":         self._os_error,
        })
        return d


class ProcessStateCorruption(KernelException):
    """
    Violation of ContainerLifecycle's internal subprocess state invariant.

    ContainerLifecycle enforces: timed_out=True is mutually exclusive with
    a non-None exit_code. A process killed by timeout does not produce an
    exit code — the OS does not deliver one through the wait() mechanism when
    the process was externally killed. If both are set, pipeline.py's subprocess
    lifecycle tracking has entered an inconsistent state.

    This exception should never appear in a correct system.

    Mirrors: ContainerLifecycle.__post_init__ →
             "timed_out=True but exit_code is set"
    Handling: pipeline.py catches, logs at CRITICAL level, returns empty output.
    """
    exception_code = EC_PROCESS_STATE
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        recipe_hash:    str,
        timed_out:      bool,
        exit_code:      Optional[int],
        detail:         str,
    ) -> None:
        super().__init__(
            f"Subprocess state is internally inconsistent: "
            f"timed_out={timed_out} exit_code={exit_code}. "
            f"detail={detail}. "
            "This is a programming error in pipeline.py's lifecycle tracking.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._timed_out = timed_out
        self._exit_code = exit_code
        self._detail    = detail

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "timed_out": self._timed_out,
            "exit_code": self._exit_code,
            "detail":    self._detail,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# RecipeMount VIOLATIONS — HARD STOPS
#
# Mirrors contracts.py :: RecipeMount, RecipeValidationResult,
#                         HardcodedRecipeManifestEntry
#
# RecipeMount is the validated recipe approval contract. Its existence
# is the proof that a recipe is safe to execute. The exceptions here
# represent the conditions that prevent RecipeMount from being constructable
# — conditions under which the kernel must never be invoked.
#
# RecipeMountError and RecipeInjectionAttempt are the TWO exceptions that
# pipeline.py re-raises to its caller rather than catching internally.
# They are the only exceptions that reach the AXIOM graph.
# Every other exception is caught and returned as empty KernelOutput.
#
# is_hard_stop=True for this entire family.
# ═════════════════════════════════════════════════════════════════════════════

class RecipeMountError(KernelException):
    """
    Base class for all conditions that prevent a recipe from being mounted.

    RecipeMountError is a hard stop. pipeline.py re-raises it to its caller.
    The AXIOM graph receives it and decides how to proceed — either falling
    back to GENERIC_HTML or escalating to operator review.

    RecipeMountError is never raised directly. Use a concrete subclass.
    The subclass tells you exactly what went wrong with the recipe.
    """
    exception_code = EC_RECIPE_NOT_FOUND  # base default; subclasses override
    is_hard_stop   = True


class RecipeNotFound(RecipeMountError):
    """
    The recipe file for the requested topology class is missing or unreadable.

    This can occur when:
    - topology_parser.py registered a recipe path that no longer exists
    - the Docker volume bind for the recipe mount failed silently
    - the file system is in a degraded state

    Mirrors: RecipeMount.recipe_path validation / registry.py get_recipe()
             returning a path that does not exist on disk.
    Handling: Hard stop. pipeline.py re-raises to caller. No recipe, no kernel.
    The AXIOM graph falls back to GENERIC_HTML if available.
    """
    exception_code = EC_RECIPE_NOT_FOUND
    is_hard_stop   = True

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        recipe_path:    str,
        is_hardcoded:   bool,
    ) -> None:
        super().__init__(
            f"Recipe file missing or unreadable: {recipe_path!r}. "
            f"topology={topology_class} is_hardcoded={is_hardcoded}. "
            f"{'Hardcoded recipe should always exist — check Docker volume bind.' if is_hardcoded else 'Compiler-generated recipe may not have been written yet.'}",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._recipe_path  = recipe_path
        self._is_hardcoded = is_hardcoded

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "recipe_path":  self._recipe_path,
            "is_hardcoded": self._is_hardcoded,
        })
        return d


class RecipeHashMismatch(RecipeMountError):
    """
    The SHA-256 hash of a recipe file does not match its manifest entry.

    For hardcoded recipes: mismatch means the file was modified after commit.
    validator.py verifies hardcoded recipe hashes on every load against the
    manifest. If the hash does not match, the recipe cannot be trusted.
    This is a security event — a hardcoded recipe should be immutable at runtime.
    Only a deliberate code change with a corresponding manifest update is valid.

    For compiler-generated recipes: mismatch means the file was modified after
    topology_parser.py registered it, or the registration recorded the wrong hash.

    Mirrors: HardcodedRecipeManifestEntry.recipe_hash verification
             RecipeRegistryEntry.recipe_hash verification at mount time
    Handling: Hard stop. Raise to caller. Log InjectionAuditRecord.
    """
    exception_code = EC_RECIPE_HASH_MISMATCH
    is_hard_stop   = True

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        recipe_path:    str,
        expected_hash:  str,
        actual_hash:    str,
        is_hardcoded:   bool,
    ) -> None:
        super().__init__(
            f"Recipe hash mismatch for {recipe_path!r}. "
            f"expected={expected_hash[:16]}... actual={actual_hash[:16]}... "
            f"is_hardcoded={is_hardcoded}. "
            f"{'SECURITY: hardcoded recipe modified at runtime — this is a filesystem tamper event.' if is_hardcoded else 'Compiler-generated recipe was modified after registration.'}",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=actual_hash,
        )
        self._recipe_path   = recipe_path
        self._expected_hash = expected_hash
        self._actual_hash   = actual_hash
        self._is_hardcoded  = is_hardcoded

    @property
    def is_tamper_event(self) -> bool:
        """
        True if this mismatch is on a hardcoded recipe.
        Hardcoded recipe modification at runtime is a filesystem tamper event.
        Compiler-generated recipe mismatch is a consistency bug.
        """
        return self._is_hardcoded

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "recipe_path":    self._recipe_path,
            "expected_hash":  self._expected_hash,
            "actual_hash":    self._actual_hash,
            "is_hardcoded":   self._is_hardcoded,
            "is_tamper_event": self.is_tamper_event,
        })
        return d


class RecipeInjectionAttempt(RecipeMountError):
    """
    validator.py detected a shell injection pattern in a recipe.

    This is the most serious exception in the kernel. The recipe compiler
    (topology_parser.py) assembles grep/sed pipelines from patterns learned
    from web page structures. Web pages can be adversarial — a hostile page
    could produce a DOM structure that causes the recipe compiler to emit a
    shell injection payload into a compiler-generated recipe.

    validator.py is the last line of defense. On detection:
    1. Capture full recipe content in InjectionAuditRecord — this is the
       forensic record needed to determine how the compiler produced a hostile
       recipe. Log it in its entirety. Do not truncate.
    2. Raise RecipeInjectionAttempt.
    3. Never pass the recipe to the kernel under any circumstances.
    4. Alert Witness immediately.

    Mirrors: RecipeValidationResult.outcome == "failed_injection"
             InjectionAuditRecord — the forensic companion to this exception
    Handling: Hard stop. pipeline.py re-raises to caller. Requires human review.
    """
    exception_code = EC_RECIPE_INJECTION
    is_hard_stop   = True

    def __init__(
        self,
        *,
        run_id:           str,
        topology_class:   str,
        recipe_path:      str,
        recipe_hash:      str,
        matched_pattern:  str,
        is_hardcoded:     bool,
        compiler_metadata: Optional[str],
    ) -> None:
        super().__init__(
            f"INJECTION DETECTED in recipe {recipe_path!r}. "
            f"matched_pattern={matched_pattern!r} "
            f"is_hardcoded={is_hardcoded}. "
            "Full recipe content logged in InjectionAuditRecord. "
            "Do NOT pass to kernel. Requires forensic review.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._recipe_path       = recipe_path
        self._matched_pattern   = matched_pattern
        self._is_hardcoded      = is_hardcoded
        self._compiler_metadata = compiler_metadata

    @property
    def matched_pattern(self) -> str:
        return self._matched_pattern

    @property
    def is_hardcoded(self) -> bool:
        return self._is_hardcoded

    @property
    def requires_forensic_review(self) -> bool:
        """Always True. RecipeInjectionAttempt always requires human forensic review."""
        return True

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "recipe_path":       self._recipe_path,
            "matched_pattern":   self._matched_pattern,
            "is_hardcoded":      self._is_hardcoded,
            "compiler_metadata": self._compiler_metadata,
            # Note: full recipe content is in InjectionAuditRecord, not here.
            # KernelAuditEvent references the InjectionAuditRecord by recipe_hash.
        })
        return d


class RecipeStructuralViolation(RecipeMountError):
    """
    A recipe passed injection checks but failed structural validation.

    Structural validation enforces:
    - Only ALLOWED_RECIPE_COMMANDS (grep, sed, awk, cat, cut, tr, head, tail, sort, uniq)
    - At least one grep or sed invocation
    - Line count ≤ MAX_RECIPE_LINE_COUNT (50)
    - Valid UTF-8 encoding throughout

    A recipe that fails structural validation is not necessarily hostile —
    it may be the result of a compiler bug that produced a recipe with an
    unsupported command (e.g., `jq` for a JSON topology class) or a recipe
    with no transformation logic. It is a quality failure, not a security failure.

    Mirrors: RecipeValidationResult.outcome == "failed_structural"
    Handling: Hard stop. The recipe must not execute. topology_parser.py
    must produce a corrected recipe before this topology class can be served.
    """
    exception_code = EC_RECIPE_STRUCTURAL
    is_hard_stop   = True

    def __init__(
        self,
        *,
        run_id:           str,
        topology_class:   str,
        recipe_path:      str,
        recipe_hash:      str,
        failure_detail:   str,
        is_hardcoded:     bool,
    ) -> None:
        super().__init__(
            f"Recipe failed structural validation: {failure_detail}. "
            f"recipe={recipe_path!r} is_hardcoded={is_hardcoded}. "
            "Check: command allowlist, line count, grep/sed presence, UTF-8 encoding.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._recipe_path    = recipe_path
        self._failure_detail = failure_detail
        self._is_hardcoded   = is_hardcoded

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "recipe_path":    self._recipe_path,
            "failure_detail": self._failure_detail,
            "is_hardcoded":   self._is_hardcoded,
        })
        return d


class RecipeDryRunFailure(RecipeMountError):
    """
    A compiler-generated recipe passed injection and structural checks but
    produced empty output on all three canonical test fixtures.

    Dry-run validation runs compiler-generated recipes against three real
    test pages per topology class (in recipes/test_fixtures/). A recipe that
    produces no output on any of them is broken — it may have inverted grep
    logic, over-specified patterns that match nothing in the fixtures, or
    incorrect zone extraction logic.

    Hardcoded recipes skip dry-run — they were validated by hand before commit.

    Mirrors: RecipeValidationResult.outcome == "failed_dryrun"
    Handling: Hard stop. The compiler produced a non-functional recipe.
    topology_parser.py must recompile for this class. GENERIC_HTML fallback
    should be used until a valid compiler-generated recipe is available.
    """
    exception_code = EC_RECIPE_DRY_RUN
    is_hard_stop   = True

    def __init__(
        self,
        *,
        run_id:          str,
        topology_class:  str,
        recipe_path:     str,
        recipe_hash:     str,
        fixtures_tested: int,
        failure_detail:  str,
    ) -> None:
        super().__init__(
            f"Compiler-generated recipe produced empty output on all {fixtures_tested} "
            f"test fixtures. recipe={recipe_path!r}. "
            f"detail={failure_detail}. "
            "Recipe logic is non-functional. topology_parser.py must recompile. "
            "GENERIC_HTML fallback will be used until a valid recipe is produced.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._recipe_path    = recipe_path
        self._fixtures_tested = fixtures_tested
        self._failure_detail = failure_detail

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "recipe_path":     self._recipe_path,
            "fixtures_tested": self._fixtures_tested,
            "failure_detail":  self._failure_detail,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# REGISTRY VIOLATIONS
#
# Mirrors contracts.py :: RecipeRegistryEntry, HardcodedRecipeManifestEntry
#
# The registry (registry.py) maintains the live mapping of topology class →
# recipe. It loads hardcoded entries on startup and accepts registrations
# from topology_parser.py. Violations here corrupt that mapping.
# ═════════════════════════════════════════════════════════════════════════════

class RecipeRegistryError(KernelException):
    """
    Base class for registry consistency violations.

    The registry is the authoritative source of recipe paths for every
    topology class. A corrupt or inconsistent registry means the kernel
    cannot reliably resolve recipes for any class. This is a startup-time
    failure — registry.py detects these during initialization.
    """
    exception_code = EC_REGISTRY_CORRUPT
    is_hard_stop   = True


class RecipeRegistryCorrupt(RecipeRegistryError):
    """
    The recipe registry is in an internally inconsistent state.

    This can occur when the mmap file backing the registry is partially
    written, when a registration write completed only partially before a
    crash, or when the registry file was modified outside the normal
    write path. The registry cannot be trusted.

    Mirrors: RecipeRegistryEntry validation failures on load
    Handling: Hard stop at startup. TAG must restore from checkpoint.
    """
    exception_code = EC_REGISTRY_CORRUPT
    is_hard_stop   = True

    def __init__(
        self,
        *,
        detail:            str,
        corrupt_entry:     Optional[str] = None,
    ) -> None:
        super().__init__(
            f"Recipe registry is corrupt and cannot be trusted: {detail}. "
            f"corrupt_entry={corrupt_entry or 'unknown'}. "
            "TAG must restore registry from checkpoint before serving any requests.",
        )
        self._detail        = detail
        self._corrupt_entry = corrupt_entry

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "detail":        self._detail,
            "corrupt_entry": self._corrupt_entry,
        })
        return d


class HardcodedRecipeOverwriteAttempt(RecipeRegistryError):
    """
    A runtime process attempted to overwrite a hardcoded recipe registry entry.

    RecipeRegistryEntry enforces: is_hardcoded=True entries can only have
    registration_source in {"hardcoded_loader", "fallback"}. If topology_parser.py
    or any other runtime process attempts to register a new recipe under a
    topology class that has a hardcoded entry, registry.py must reject it.

    Hardcoded recipes are immutable at runtime. This constraint is the guarantee
    that the five hand-validated recipes cannot be silently replaced by a
    compiler-generated recipe — even one that passes validation.

    Mirrors: RecipeRegistryEntry.__post_init__ →
             "is_hardcoded=True but registration_source is not 'hardcoded_loader'"
    Handling: Hard stop. The attempted registration is rejected. Log to Witness.
    """
    exception_code = EC_REGISTRY_OVERWRITE
    is_hard_stop   = True

    def __init__(
        self,
        *,
        topology_class:      str,
        attempted_source:    str,
        existing_hash:       str,
        attempted_hash:      str,
    ) -> None:
        super().__init__(
            f"Attempt to overwrite hardcoded recipe for topology_class={topology_class!r} "
            f"by source={attempted_source!r}. "
            f"existing_hash={existing_hash[:16]}... attempted_hash={attempted_hash[:16]}... "
            "Hardcoded recipes are immutable at runtime. Registration rejected.",
            topology_class=topology_class,
            recipe_hash=existing_hash,
        )
        self._attempted_source = attempted_source
        self._existing_hash    = existing_hash
        self._attempted_hash   = attempted_hash

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "attempted_source": self._attempted_source,
            "existing_hash":    self._existing_hash,
            "attempted_hash":   self._attempted_hash,
        })
        return d


class RecipeManifestCorruption(RecipeRegistryError):
    """
    The hardcoded recipe hash manifest is missing, unreadable, or structurally invalid.

    validator.py loads the manifest on startup and uses it to verify every
    hardcoded recipe. If the manifest itself cannot be loaded, validator.py
    cannot verify any hardcoded recipe — and therefore cannot permit any
    hardcoded recipe to execute.

    The manifest is a static file committed to the repository. It should
    never be absent or unreadable in a correctly deployed system. Its absence
    indicates a packaging error or filesystem corruption.

    Mirrors: HardcodedRecipeManifestEntry — the manifest contains these entries
    Handling: Hard stop at startup. No recipes can be validated. TAG cannot start.
    """
    exception_code = EC_MANIFEST_CORRUPT
    is_hard_stop   = True

    def __init__(
        self,
        *,
        manifest_path:  str,
        detail:         str,
    ) -> None:
        super().__init__(
            f"Hardcoded recipe manifest at {manifest_path!r} is missing or corrupt: {detail}. "
            "validator.py cannot verify any hardcoded recipe without the manifest. "
            "TAG cannot start. Check packaging and filesystem integrity.",
        )
        self._manifest_path = manifest_path
        self._detail        = detail

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "manifest_path": self._manifest_path,
            "detail":        self._detail,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINT VIOLATIONS
#
# Mirrors contracts.py :: CheckpointRecord, CheckpointHealth, RestoreAttempt
#
# Process 2 (crond + mft_checkpoint.sh) produces checkpoints. Process 1
# (the grep pipeline) does not interact with checkpoints at all. An unhealthy
# checkpoint system does not stop Process 1 — it triggers a Witness alert.
# The exceptions here represent failures of the checkpoint system itself.
#
# Architectural rule: a checkpoint failure must never stop an extraction.
# Process 1 and Process 2 share an OS and nothing else.
# ═════════════════════════════════════════════════════════════════════════════

class CheckpointWriteError(KernelException):
    """
    Base class for failures of the checkpoint write path.

    mft_checkpoint.sh writes archives every 15 minutes. If the write fails
    for any reason, checkpoint_monitor.py detects it via integrity check on
    the next crond cycle and logs it. The write failure itself does not stop
    Process 1 — it triggers a Witness alert and is retried on the next cycle.
    """
    exception_code = EC_CHECKPOINT_WRITE
    is_hard_stop   = False


class CheckpointIntegrityError(KernelException):
    """
    The checkpoint archive was written but failed its own integrity verification.

    mft_checkpoint.sh runs `tar -tzf` immediately after writing the archive.
    A corrupt archive that passes silently is worse than a logged failure.
    If verification fails, the archive is deleted immediately. The next crond
    cycle will attempt a fresh write.

    This is a post-write verification failure, not a write failure — it must
    not inherit from CheckpointWriteError.  A broad catch on CheckpointWriteError
    must not silently absorb integrity errors, which represent a distinct failure
    mode (the write succeeded; the result is corrupt) requiring a separate
    remediation path (delete archive, alert Witness, await next crond cycle).

    Mirrors: CheckpointRecord.integrity_verified=False after a write attempt
             mft_checkpoint.sh → "CHECKPOINT CORRUPT: {archive}"
    Handling: Not a hard stop. Log. Delete the corrupt archive. Alert Witness.
    The next 15-minute cycle will attempt a fresh checkpoint.
    """
    exception_code = EC_CHECKPOINT_INTEGRITY
    is_hard_stop   = False

    def __init__(
        self,
        *,
        archive_path:  str,
        written_bytes: int,
        tar_exit_code: int,
    ) -> None:
        super().__init__(
            f"Checkpoint archive {archive_path!r} failed integrity verification "
            f"(tar -tzf exited {tar_exit_code}). "
            f"written_bytes={written_bytes:,}. "
            "Archive deleted. Next crond cycle will attempt a fresh checkpoint.",
        )
        self._archive_path  = archive_path
        self._written_bytes = written_bytes
        self._tar_exit_code = tar_exit_code

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "archive_path":  self._archive_path,
            "written_bytes": self._written_bytes,
            "tar_exit_code": self._tar_exit_code,
        })
        return d


class CheckpointRotationError(CheckpointWriteError):
    """
    The checkpoint rotation step failed — old archives beyond the 48-archive
    retain limit could not be deleted.

    mft_checkpoint.sh runs `ls -t ... | tail -n +49 | xargs rm -f` after a
    successful write. If this fails, the checkpoint directory will accumulate
    archives beyond the 12-hour retain window. This is not a data loss event —
    it is a storage growth event that checkpoint_monitor.py must alert on.

    Mirrors: CHECKPOINT_RETAIN_COUNT enforcement in mft_checkpoint.sh
    Handling: Not a hard stop. Log. Alert Witness. Manual cleanup may be needed.
    """
    exception_code = EC_CHECKPOINT_ROTATION
    is_hard_stop   = False

    def __init__(
        self,
        *,
        archive_count_before: int,
        archives_to_delete:   int,
        os_error:             str,
    ) -> None:
        super().__init__(
            f"Checkpoint rotation failed: could not delete {archives_to_delete} old archives. "
            f"archive_count_before={archive_count_before} os_error={os_error}. "
            "Storage may accumulate beyond the 48-archive limit. Manual cleanup may be needed.",
        )
        self._archive_count_before = archive_count_before
        self._archives_to_delete   = archives_to_delete
        self._os_error             = os_error

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "archive_count_before": self._archive_count_before,
            "archives_to_delete":   self._archives_to_delete,
            "os_error":             self._os_error,
        })
        return d


class CrondProcessError(KernelException):
    """
    Process 2 (crond) has stopped running or is no longer firing.

    checkpoint_monitor.py detects this when no new checkpoint log entry
    appears within CHECKPOINT_STALE_THRESHOLD_MINUTES (20 minutes) — more
    than one missed cycle. When detected, checkpoint_monitor.py attempts to
    restart crond. If restart fails, Witness is alerted.

    Mirrors: CheckpointHealth.crond_alive=False
             CheckpointHealth.minutes_since_last_checkpoint > 20
    Handling: Not a hard stop for Process 1 — extractions continue.
    checkpoint_monitor.py restarts crond. Witness is alerted.
    Maximum data loss from crond death: one 15-minute checkpoint interval.
    """
    exception_code = EC_CROND_DEAD
    is_hard_stop   = False

    def __init__(
        self,
        *,
        minutes_since_last_checkpoint: float,
        last_checkpoint_time:          Optional[str],
        restart_attempted:             bool,
        restart_succeeded:             Optional[bool],
    ) -> None:
        super().__init__(
            f"crond (Process 2) appears dead — no checkpoint in "
            f"{minutes_since_last_checkpoint:.1f} minutes "
            f"(threshold={20}min). "
            f"last_checkpoint={last_checkpoint_time or 'never'}. "
            f"restart_attempted={restart_attempted} restart_succeeded={restart_succeeded}. "
            "Process 1 continues. checkpoint_monitor.py will attempt restart.",
        )
        self._minutes_since = minutes_since_last_checkpoint
        self._last_time     = last_checkpoint_time
        self._restart_attempted = restart_attempted
        self._restart_succeeded = restart_succeeded

    @property
    def is_restart_failed(self) -> bool:
        return self._restart_attempted and self._restart_succeeded is False

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "minutes_since_last_checkpoint": round(self._minutes_since, 1),
            "last_checkpoint_time":          self._last_time,
            "restart_attempted":             self._restart_attempted,
            "restart_succeeded":             self._restart_succeeded,
            "is_restart_failed":             self.is_restart_failed,
        })
        return d


class CheckpointCorruptionError(KernelException):
    """
    checkpoint_monitor.py attempted a restore() and every available archive
    failed one or more integrity-verification layers.

    This is a monitor-layer hard stop — distinct from CheckpointIntegrityError
    (which is a write-path failure logged by mft_checkpoint.sh and is NOT a
    hard stop) and from CheckpointExhaustionError (which describes restore.sh
    exhausting all archives at the shell level).

    CheckpointCorruptionError is raised by checkpoint_monitor.restore() after
    iterating ALL available candidate archives newest-to-oldest and finding
    none that pass the six-layer integrity check:

      Layer 1 — file existence and type
      Layer 2 — file size sanity bounds
      Layer 3 — SHA-256 vs .sha256 sidecar (when sidecar is present)
      Layer 4 — gzip stream validity (tarfile.open streaming mode)
      Layer 5 — manifest completeness (all four STORE_FILE_NAMES present)
      Layer 6 — member safety (no abs paths, no path traversal, sizes plausible)

    A single layer failure is sufficient to reject an archive. All archives
    must be exhausted before this exception is raised — checkpoint_monitor
    iterates newest-to-oldest exactly as restore.sh does.

    This is a hard stop: the calling system (TAG startup sequence or
    checkpoint_monitor.py's caller) cannot obtain valid index files from the
    checkpoint store and must escalate to operator review.

    Relation to CheckpointIntegrityError:
      CheckpointIntegrityError   — write-path failure; is_hard_stop=False;
                                   raised by checkpoint_monitor when a newly
                                   written archive fails post-write verification.
      CheckpointCorruptionError  — restore-path failure; is_hard_stop=True;
                                   raised when ALL archives fail and no restore
                                   is possible.

    Mirrors: checkpoint_monitor.restore() returning no valid bytes
             after exhausting all candidates.
    Handling: Hard stop. TAG startup cannot continue. Operator must intervene.
    """
    exception_code = EC_CHECKPOINT_CORRUPT
    is_hard_stop   = True

    def __init__(
        self,
        *,
        archive_path:   str,
        archives_tried: int,
        failure_reason: str,
    ) -> None:
        super().__init__(
            f"All {archives_tried} checkpoint archive(s) in {archive_path!r} failed "
            f"integrity verification. "
            f"Last failure: {failure_reason}. "
            "checkpoint_monitor.restore() exhausted every candidate "
            "(newest-to-oldest) with no recoverable archive. "
            "TAG cannot restore index files from the checkpoint store. "
            "Operator must provide a valid store directory or archives.",
        )
        self._archive_path   = archive_path
        self._archives_tried = archives_tried
        self._failure_reason = failure_reason

    @property
    def archives_tried(self) -> int:
        """Number of archives inspected before giving up."""
        return self._archives_tried

    @property
    def failure_reason(self) -> str:
        """Human-readable description of why the last archive was rejected."""
        return self._failure_reason

    @property
    def archive_path(self) -> str:
        """The checkpoint directory (or archive path) that was inspected."""
        return self._archive_path

    @property
    def is_total_loss(self) -> bool:
        """
        True when no archives at all were found (archives_tried == 0).
        Distinguished from the case where archives exist but all are corrupt.
        """
        return self._archives_tried == 0

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "archive_path":   self._archive_path,
            "archives_tried": self._archives_tried,
            "failure_reason": self._failure_reason,
            "is_total_loss":  self.is_total_loss,
        })
        return d


# ─────────────────────────────────────────────────────────────────────────────
# CheckpointStalenessWarning
#
# NOT a KernelException. NOT raised. NOT a hard stop.
#
# A Python UserWarning subclass surfaced by checkpoint_monitor.health() via
# warnings.warn() when the most recent archive mtime exceeds
# CHECKPOINT_STALE_THRESHOLD_SECONDS (1800s = 30 min = 2× the crond interval).
#
# Staleness is observable, not fatal. Process 1 (the grep pipeline) continues
# regardless. The warning is a signal to operators and the Witness monitoring
# layer that crond may have silently exited.
#
# To suppress programmatically: warnings.filterwarnings("ignore",
#   category=CheckpointStalenessWarning)
# To catch: use warnings.catch_warnings() in tests.
# ─────────────────────────────────────────────────────────────────────────────

class CheckpointStalenessWarning(UserWarning):
    """
    Emitted by checkpoint_monitor.health() when the most recent checkpoint
    archive is stale — its mtime exceeds CHECKPOINT_STALE_THRESHOLD_SECONDS
    (1800 seconds = 30 minutes = 2× the 15-minute crond interval).

    Staleness implies crond (Process 2) has silently exited. Extractions
    (Process 1) are unaffected. The warning is informational — it asks the
    monitoring layer to investigate crond liveness and potentially alert.

    This is a Python UserWarning, not a KernelException. It is never raised
    with `raise`. It is always surfaced with `warnings.warn()` from within
    checkpoint_monitor.health().

    Format surfaced via warnings.warn():
      CheckpointStalenessWarning: <message string>

    The message string contains:
      - archive filename
      - age in seconds
      - threshold in seconds

    Callers that want to inspect staleness programmatically should read
    CheckpointHealth.minutes_since_last_checkpoint and check it against
    contracts.CHECKPOINT_STALE_THRESHOLD_MINUTES rather than catching this
    warning — warnings are for monitoring integration, not control flow.
    """


# ═════════════════════════════════════════════════════════════════════════════
# RestoreAttempt VIOLATIONS
#
# Mirrors contracts.py :: RestoreAttempt
#
# restore.sh is called by checkpoint_monitor.py on TAG startup when index
# files are missing or corrupted. It iterates archives newest-to-oldest.
# These exceptions represent conditions where restoration cannot complete.
# ═════════════════════════════════════════════════════════════════════════════

class RestoreFailure(KernelException):
    """
    restore.sh exited with a non-zero exit code — restoration failed.

    TAG startup sequence catches this. If TAG cannot obtain valid index files
    (topology_router.pt, recipe_registry.mmap, phase_states.mmap,
    structural_layer.pt) from restore.sh, it cannot operate. This exception
    propagates to TAG's startup sequence for operator handling.

    Mirrors: RestoreAttempt.restore_succeeded=False
             restore.sh exit code 1 → "RESTORE FAILED: no valid checkpoint found"
    Handling: Hard stop at startup. TAG cannot serve requests. Operator review required.
    """
    exception_code = EC_RESTORE_FAILURE
    is_hard_stop   = True

    def __init__(
        self,
        *,
        exit_code:       int,
        failure_reason:  str,
        archives_tried:  int,
    ) -> None:
        super().__init__(
            f"restore.sh failed with exit code {exit_code}: {failure_reason}. "
            f"archives_tried={archives_tried}. "
            "TAG cannot operate without valid index files. "
            "Operator must provide a valid store directory or restore manually.",
        )
        self._exit_code      = exit_code
        self._failure_reason = failure_reason
        self._archives_tried = archives_tried

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "exit_code":      self._exit_code,
            "failure_reason": self._failure_reason,
            "archives_tried": self._archives_tried,
        })
        return d


class CheckpointExhaustionError(RestoreFailure):
    """
    All archived checkpoints are corrupt. restore.sh found no valid archive.

    restore.sh iterates from newest to oldest, skipping corrupt archives via
    `tar -tzf` verification. If every archive in the 48-archive window is
    corrupt, there is no recoverable state and restore.sh exits with code 1.

    This is the worst-case checkpoint scenario. Worst-case data loss:
    everything since the last successful checkpoint write (≤ 15 minutes of
    RL training in normal operation).

    Mirrors: RestoreAttempt.archives_skipped == all archives,
             restore_succeeded=False, failure_reason="no valid checkpoint found"
    Handling: Hard stop at startup. Operator must provide a clean store.
    """
    exception_code = EC_RESTORE_EXHAUSTED
    is_hard_stop   = True

    def __init__(
        self,
        *,
        total_archives_found: int,
        corrupt_archives:     int,
    ) -> None:
        super().__init__(
            exit_code=1,
            failure_reason=(
                f"All {total_archives_found} checkpoint archives are corrupt. "
                f"corrupt_archives={corrupt_archives}. "
                "12-hour checkpoint window is entirely unrecoverable."
            ),
            archives_tried=total_archives_found,
        )
        self._total_archives   = total_archives_found
        self._corrupt_archives = corrupt_archives

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "total_archives_found": self._total_archives,
            "corrupt_archives":     self._corrupt_archives,
        })
        return d


class RestorePartialError(RestoreFailure):
    """
    restore.sh extracted an archive but not all four store files were recovered.

    The four required files are: topology_router.pt, recipe_registry.mmap,
    phase_states.mmap, structural_layer.pt. An archive that is structurally
    valid (passes `tar -tzf`) but was created before all four files existed
    will produce a partial restore. This can happen if TAG was checkpointed
    during initial index construction before all four files were written.

    Mirrors: RestoreAttempt.recovered_completely=False
    Handling: Hard stop. Partial index state is not safe to operate on.
    checkpoint_monitor.py should try the next archive if available.
    """
    exception_code = EC_RESTORE_PARTIAL
    is_hard_stop   = True

    def __init__(
        self,
        *,
        archive_path:    str,
        restored_files:  set,
        missing_files:   set,
    ) -> None:
        super().__init__(
            exit_code=1,
            failure_reason=(
                f"Archive {archive_path!r} extracted but missing files: {missing_files}. "
                f"restored={sorted(restored_files)} missing={sorted(missing_files)}. "
                "Partial index state is not safe to operate on."
            ),
            archives_tried=1,
        )
        self._archive_path   = archive_path
        self._restored_files = frozenset(restored_files)
        self._missing_files  = frozenset(missing_files)

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "archive_path":   self._archive_path,
            "restored_files": sorted(self._restored_files),
            "missing_files":  sorted(self._missing_files),
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# FEEDBACK VIOLATIONS
#
# Mirrors contracts.py :: FeedbackEvent, QualityWindowEntry,
#                         RecipeQualityAggregate
#
# ExtractionQuality never fails hard — quality scoring is deterministic
# arithmetic and degrades with low scores, not exceptions. This is by design:
# the feedback path must never interrupt an extraction cycle.
#
# FeedbackEmissionError is the one exception in this group — it fires when
# feedback.py cannot deliver a FeedbackEvent to topology_parser.py. The
# quality signal is lost for this invocation, but the extraction continues.
# ═════════════════════════════════════════════════════════════════════════════

class FeedbackError(KernelException):
    """
    Base class for feedback path failures.

    Feedback failures are never hard stops. The extraction has already
    completed by the time feedback.py runs. A failed FeedbackEvent means
    topology_parser.py misses one quality signal — the rolling window
    loses one sample. The system degrades imperceptibly.
    """
    exception_code = EC_FEEDBACK_EMISSION
    is_hard_stop   = False


class FeedbackEmissionError(FeedbackError):
    """
    feedback.py could not deliver a FeedbackEvent to topology_parser.py.

    FeedbackEvent is the decoupled quality event that closes the learning
    loop. topology_parser.py consumes these to decide when to recompile a
    recipe. If the emission fails (topology_parser.py is restarting, the
    channel is full, serialization failed), the event is lost for this run.

    The rolling quality window for this topology class loses one sample.
    This is architecturally acceptable — the window is bounded by
    QUALITY_WINDOW_SIZE and one missed sample does not materially affect
    recompilation decisions. Log and continue.

    Mirrors: FeedbackEvent — the event that could not be delivered
    Handling: Not a hard stop. Log. The extraction result is not affected.
    """
    exception_code = EC_FEEDBACK_EMISSION
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         str,
        topology_class: str,
        recipe_hash:    str,
        emission_error: str,
        quality_score:  float,
    ) -> None:
        super().__init__(
            f"FeedbackEvent emission failed for topology={topology_class}: {emission_error}. "
            f"quality_score={quality_score:.4f}. "
            "One quality sample lost from rolling window. Learning loop slightly degraded.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._emission_error = emission_error
        self._quality_score  = quality_score

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "emission_error": self._emission_error,
            "quality_score":  round(self._quality_score, 4),
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# TELEMETRY VIOLATIONS
#
# Mirrors contracts.py :: PipelineTelemetry
#
# PipelineTelemetry is required instrumentation — not optional telemetry.
# Every invocation must emit a telemetry record to Witness. A failed emission
# is a data product gap. It does not affect the extraction result, but it
# is a gap in the observability stream that validates kernel performance.
# ═════════════════════════════════════════════════════════════════════════════

class TelemetryError(KernelException):
    """Base class for telemetry path failures."""
    exception_code = EC_TELEMETRY_EMISSION
    is_hard_stop   = False


class TelemetryEmissionError(TelemetryError):
    """
    pipeline.py could not emit PipelineTelemetry to Witness.

    PipelineTelemetry is the data product that validates whether kernel
    integration is performing as designed. Every invocation must produce one.
    A failed emission means Witness has a gap in the kernel performance stream
    for this run_id. The extraction result is not affected.

    Common causes: Witness endpoint unreachable (though the kernel has
    network_mode: none — telemetry is written to a file or queue), disk full,
    serialization failure in to_log_dict().

    Mirrors: PipelineTelemetry.to_log_dict() → downstream emission failure
    Handling: Not a hard stop. Log locally. The extraction result is returned.
    The observability gap must be monitored — sustained gaps require investigation.
    """
    exception_code = EC_TELEMETRY_EMISSION
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:          str,
        topology_class:  str,
        recipe_hash:     str,
        emission_error:  str,
        timed_out:       bool,
        extraction_empty: bool,
    ) -> None:
        super().__init__(
            f"PipelineTelemetry emission failed for run_id={run_id}: {emission_error}. "
            f"timed_out={timed_out} extraction_empty={extraction_empty}. "
            "Witness observability gap for this run. "
            "Sustained gaps indicate a telemetry path problem requiring investigation.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._emission_error  = emission_error
        self._timed_out       = timed_out
        self._extraction_empty = extraction_empty

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "emission_error":  self._emission_error,
            "timed_out":       self._timed_out,
            "extraction_empty": self._extraction_empty,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# AUDIT VIOLATIONS
#
# Mirrors contracts.py :: InjectionAuditRecord, KernelAuditEvent
#
# Audit records are forensic-grade. An AuditEmissionError is treated with
# greater severity than a TelemetryEmissionError — losing a performance
# metric is tolerable, losing a security-critical forensic record is not.
# ═════════════════════════════════════════════════════════════════════════════

class AuditError(KernelException):
    """Base class for audit path failures."""
    exception_code = EC_AUDIT_EMISSION
    is_hard_stop   = False


class AuditEmissionError(AuditError):
    """
    An InjectionAuditRecord or KernelAuditEvent could not be written.

    Audit records are forensic-grade. A failed audit write for a
    RecipeInjectionAttempt means the forensic record — full recipe content,
    matched pattern, compiler metadata — may be lost. This is a security
    control failure on top of the injection event itself.

    For critical-severity audit events: AuditEmissionError is treated as
    requiring immediate investigation even though it is not itself a hard stop
    for the extraction. The security event (injection attempt) is still raised
    as RecipeInjectionAttempt — the audit emission failure is additional damage.

    For warn/info-severity audit events: logging gap only. Continue.

    Mirrors: InjectionAuditRecord — the forensic record that could not be written
             KernelAuditEvent — the high-level event that could not be recorded
    Handling: Not a hard stop for extraction. Log the failure itself. Alert Witness
    for critical-severity events. The underlying security exception still propagates.
    """
    exception_code = EC_AUDIT_EMISSION
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:           str,
        topology_class:   str,
        recipe_hash:      str,
        event_type:       str,
        severity:         str,
        emission_error:   str,
        forensic_content_lost: bool,
    ) -> None:
        severity_note = (
            "SECURITY CONTROL FAILURE: forensic recipe content may be lost."
            if forensic_content_lost else
            "Observability gap only — no forensic content was involved."
        )
        super().__init__(
            f"Audit record could not be written: event_type={event_type!r} "
            f"severity={severity!r}. {emission_error}. {severity_note}",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._event_type            = event_type
        self._severity              = severity
        self._emission_error        = emission_error
        self._forensic_content_lost = forensic_content_lost

    @property
    def is_security_critical(self) -> bool:
        """True if a critical-severity forensic record was lost."""
        return self._severity == "critical" and self._forensic_content_lost

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "event_type":             self._event_type,
            "severity":               self._severity,
            "emission_error":         self._emission_error,
            "forensic_content_lost":  self._forensic_content_lost,
            "is_security_critical":   self.is_security_critical,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# INTERFACE / DAEMON VIOLATIONS
#
# Mirrors contracts.py :: DaemonQuery, DaemonResponse, TraversalConfig
#
# The interface layer is the boundary between the AXIOM graph and the TAG
# kernel. DaemonQuery comes from the controller. DaemonResponse goes back.
# Violations here indicate the controller and TAG have lost contract alignment.
# ═════════════════════════════════════════════════════════════════════════════

class DaemonError(KernelException):
    """Base class for interface/daemon contract violations."""
    exception_code = EC_DAEMON_QUERY  # overridden in subclasses
    is_hard_stop   = False


class DaemonQueryError(DaemonError):
    """
    The AXIOM controller delivered a malformed DaemonQuery to interface.py.

    DaemonQuery enforces: intent_vector_hash is SHA-256, target_urls is
    non-empty, every URL begins with http:// or https://. A query that
    violates any of these arrived from the controller in a broken state.

    This indicates the controller and TAG have lost contract alignment.
    Either the controller built a query without using DaemonQuery's
    construction path, or something corrupted the query in transit.

    Mirrors: DaemonQuery.__post_init__ validation failures
    Handling: Not a hard stop for the kernel. interface.py returns an error
    response to the controller. The controller decides how to handle it.
    """
    exception_code = EC_DAEMON_QUERY
    is_hard_stop   = False

    def __init__(
        self,
        *,
        intent_vector_hash: Optional[str],
        detail:             str,
        url_count:          int,
    ) -> None:
        super().__init__(
            f"Malformed DaemonQuery from AXIOM controller: {detail}. "
            f"intent_vector_hash={'set' if intent_vector_hash else 'missing'} "
            f"url_count={url_count}. "
            "Controller and TAG have lost contract alignment. "
            "Check controller query construction path.",
        )
        self._detail   = detail
        self._url_count = url_count

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "detail":    self._detail,
            "url_count": self._url_count,
        })
        return d


class DaemonResponseError(DaemonError):
    """
    interface.py could not construct a valid DaemonResponse.

    DaemonResponse enforces: source_priority is non-empty, every friction
    score is in [0.0, 1.0], recipe is non-empty, recipe_hash is SHA-256,
    topology_class is valid, phase is one of "learns"/"predicts"/"knows".

    A DaemonResponse that fails construction means the daemon's internal
    state (world_model.py output) is inconsistent. This is an internal
    TAG error, not a controller error.

    Mirrors: DaemonResponse.__post_init__, TraversalConfig.__post_init__
    Handling: Not a hard stop for the kernel itself — interface.py returns
    an error response to the AXIOM controller. The controller falls back to
    a default traversal configuration.
    """
    exception_code = EC_DAEMON_RESPONSE
    is_hard_stop   = False

    def __init__(
        self,
        *,
        run_id:         Optional[str],
        topology_class: Optional[str],
        detail:         str,
        phase:          Optional[str],
    ) -> None:
        super().__init__(
            f"DaemonResponse construction failed: {detail}. "
            f"phase={phase or 'unknown'}. "
            "Daemon internal state is inconsistent. "
            "world_model.py output validation should be checked.",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._detail = detail
        self._phase  = phase

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "detail": self._detail,
            "phase":  self._phase,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL INVARIANT VIOLATION
#
# A ContractViolation represents an internal consistency condition that should
# be impossible in a correct system — detected by assertion or explicit check,
# it indicates a programming error in the kernel itself rather than a runtime
# failure of the content being processed or the infrastructure it runs on.
#
# ContractViolation is never a hard stop for the AXIOM graph — pipeline.py
# catches it and returns empty KernelOutput — but it must be logged at
# CRITICAL level and is always an indication that a code path requires
# immediate investigation.
# ═════════════════════════════════════════════════════════════════════════════

class ContractViolation(KernelException):
    """
    An internal kernel invariant that should never be violated was violated.

    This class represents the category of "impossible at runtime if the code
    is correct" — conditions that can only occur due to a programming error
    in the kernel itself. They are distinct from RecipeMountError (bad recipe),
    SubprocessTimeout (slow page), or RestoreFailure (corrupt checkpoint) —
    those are expected failure modes in an adversarial environment.
    ContractViolation means the kernel's own logic is broken.

    Examples:
    - ProcessStateCorruption (timed_out=True with exit_code set)
    - OutputMeasurementError (clean bytes > raw bytes)
    - A make_*() factory function receiving inconsistent inputs from the
      same pipeline.py invocation

    Any ContractViolation in production is a bug report. It must be logged at
    CRITICAL level with full context, filed as a defect, and fixed before the
    next deploy. It is never a hard stop — the extraction degrades, the graph
    continues — but it must not be silently swallowed.

    Never raise ContractViolation directly. The concrete subclasses
    (ProcessStateCorruption, OutputMeasurementError) are the raise sites.
    Use ContractViolation only for catch clauses that need to handle all
    internal invariant failures as a group.
    """
    exception_code = EC_CONTRACT_VIOLATION
    is_hard_stop   = False

    def __init__(
        self,
        *,
        invariant:      str,
        detail:         str,
        run_id:         Optional[str] = None,
        topology_class: Optional[str] = None,
        recipe_hash:    Optional[str] = None,
    ) -> None:
        super().__init__(
            f"Internal kernel invariant violated: {invariant}. "
            f"detail={detail}. "
            "This is a programming error. Log at CRITICAL. File a defect. "
            "Do not silence this exception without investigation.",
            run_id=run_id,
            topology_class=topology_class,
            recipe_hash=recipe_hash,
        )
        self._invariant = invariant
        self._detail    = detail

    @property
    def invariant(self) -> str:
        return self._invariant

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "invariant": self._invariant,
            "detail":    self._detail,
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# TOPOLOGY LAYER EXCEPTIONS
#
# All inherit from TopologyException, which itself inherits KernelException,
# so existing catch blocks in the kernel that handle KernelException will
# also catch topology errors — no existing handlers require modification.
#
# Raise law from the base module applies here unchanged:
#   - No bare raises; always a concrete leaf class.
#   - `from original` chaining is mandatory when wrapping lower-level errors.
#   - is_hard_stop=True means the index daemon or bus must halt; False means
#     degrade gracefully and continue.
# ═════════════════════════════════════════════════════════════════════════════

class TopologyException(KernelException):
    """Base for all topology layer exceptions. Inherits KernelException
    so existing catch blocks in the kernel still catch topology errors."""


# ── Classifier ────────────────────────────────────────────────────────────────

class ClassifierModelNotInitialized(TopologyException): # noqa
    """topology_router.pt not loaded before classify() was called.
    Hard stop — system is not ready to serve queries."""
    exception_code = EC_TOPO_CLASSIFIER_NOT_INIT
    is_hard_stop   = True


class ClassificationConfidenceTooLow(TopologyException):
    """All classification paths returned confidence below THETA_CLASSIFY_FALLBACK.
    Soft — caller falls back to GENERIC_HTML. Log as WARNING not ERROR."""
    exception_code = EC_TOPO_CONFIDENCE_TOO_LOW
    is_hard_stop   = False


class ClassificationWindowTooSmall(TopologyException):
    """content_prefix below minimum viable size for window classification.
    Soft — caller uses URL + header signals only, skips window path."""
    exception_code = EC_TOPO_WINDOW_TOO_SMALL
    is_hard_stop   = False


# ── Parser ────────────────────────────────────────────────────────────────────

class RecipeCompilationFailed(TopologyException):
    """WLP returned empty zone map and no parent class fallback available.
    Soft — registry falls back to GENERIC_HTML recipe."""
    exception_code = EC_TOPO_RECIPE_COMPILE_FAILED
    is_hard_stop   = False


class RecipeVersionConflict(TopologyException):
    """Attempted to register a recipe version that already exists in registry.
    Soft — increment version and retry."""
    exception_code = EC_TOPO_RECIPE_VERSION_CONFLICT
    is_hard_stop   = False


class WLPQueryFailed(TopologyException):
    """latent_parser.py returned None or raised unexpectedly during query().
    Soft — parser uses last known good recipe for this topology class."""
    exception_code = EC_TOPO_WLP_QUERY_FAILED
    is_hard_stop   = False


# ── Sanitizer ─────────────────────────────────────────────────────────────────

class SanitizerInputError(TopologyException):
    """KernelOutput.clean_signal is None or not a string.
    Hard stop — pipeline.py should never produce this state."""
    exception_code = EC_TOPO_SANITIZER_INPUT
    is_hard_stop   = True


# ── Surprise detector ─────────────────────────────────────────────────────────

class SurpriseHistoryCorrupted(TopologyException): # noqa
    """phase_states.mmap surprise history section failed integrity check.
    Soft — reinitialize history for affected class and continue."""
    exception_code = EC_TOPO_SURPRISE_HISTORY
    is_hard_stop   = False


# ── Phantom ───────────────────────────────────────────────────────────────────

class PhantomFetchFailed(TopologyException):
    """httpx or Playwright fetch failed after retry budget exhausted.
    Soft — DaemonResponse returns extraction_empty=True."""
    exception_code = EC_TOPO_PHANTOM_FETCH_FAILED
    is_hard_stop   = False


class PhantomRenderTimeout(TopologyException):
    """Playwright render exceeded TraversalPolicy.timeout_ms.
    Soft — same as PhantomFetchFailed."""
    exception_code = EC_TOPO_PHANTOM_RENDER_TIMEOUT
    is_hard_stop   = False


class PhantomFrictionDetected(TopologyException):
    """Response matches known friction fingerprint: Cloudflare challenge,
    paywall, rate limit, or auth redirect.
    Soft — DaemonResponse returns extraction_empty=True with friction_type."""
    exception_code = EC_TOPO_PHANTOM_FRICTION
    is_hard_stop   = False


# ── Index daemon ──────────────────────────────────────────────────────────────

class GradientStepFailed(TopologyException):
    """topology_router.pt gradient step raised unexpectedly.
    Hard stop — weights may be corrupt. Restore from checkpoint before continuing."""
    exception_code = EC_TOPO_GRADIENT_STEP_FAILED
    is_hard_stop   = True


class PhaseMMapCorrupted(TopologyException):
    """phase_states.mmap failed integrity check on read.
    Hard stop — restore from last checkpoint before continuing."""
    exception_code = EC_TOPO_PHASE_MMAP_CORRUPTED
    is_hard_stop   = True


# ── Bus ───────────────────────────────────────────────────────────────────────

class EventBusSubscriptionError(TopologyException):
    """Component attempted to subscribe to an unrecognized event type.
    Hard stop — programming error, not a runtime condition."""
    exception_code = EC_TOPO_BUS_SUBSCRIPTION
    is_hard_stop   = True


class EventDispatchFailed(TopologyException):
    """A subscriber handler raised during event dispatch.
    Soft — log, continue dispatch to remaining subscribers."""
    exception_code = EC_TOPO_EVENT_DISPATCH
    is_hard_stop   = False


class KafkaSinkUnavailable(TopologyException):
    """
    Kafka broker is unreachable or refused a producer/consumer operation.

    Raised in two distinct scenarios:

    1.  During bus.start() — the Kafka probe succeeded (broker was reachable)
        but the first producer.start() call subsequently failed.  The bus
        falls to degraded mode automatically and this exception is caught
        internally.  Components never see it in this case.

    2.  During emit() at runtime — the broker became unavailable after the bus
        entered Kafka mode.  The bus does NOT automatically fall back to
        degraded mode mid-operation (silent sink switching is dangerous and
        makes event ordering guarantees impossible).  The exception propagates
        to the TopicEmitter.emit() caller.  The caller decides whether to:
            - buffer the event locally and retry
            - drop the event and log the loss
            - propagate upward and stall the fetcher

    is_hard_stop = False: one unavailable broker does not kill the process.
    The circuit breaker in _KafkaSink will open and fast-fail subsequent
    emit() calls until the broker recovers.

    Not raised in degraded mode — asyncio.Queue never becomes "unavailable".
    If the degraded-mode queue is full, emit() blocks (backpressure), never raises.
    """
    exception_code = EC_TOPO_KAFKA_UNAVAILABLE
    is_hard_stop   = False


class EventIntegrityError(TopologyException):
    """
    HMAC-SHA256 verification failed on a received BusEnvelope.

    Raised by _deserialize() before any msgpack deserialization is attempted.
    The handler is never called for an envelope that fails integrity verification.
    The event is written to /store/dead_letters.jsonl with handler="__hmac_verify__".

    Causes:
        - Envelope was tampered with in transit (value, key, or timestamp modified).
        - AXIOM_BUS_HMAC_KEY mismatch between producer and consumer processes
          (common after a key rotation where not all processes were restarted).
        - Missing hmac_sha256 header (envelope produced by a version of the bus
          that predates the HMAC requirement — not possible in production since
          HMAC was present from the first release).
        - Replay of an envelope with a different emit_timestamp than the one
          that was originally signed.

    is_hard_stop = False: one integrity failure is a dead letter, not a process
    death.  Repeated integrity failures for the same source_component in the
    dead letter log ARE a structural problem — they surface via index_daemon.py's
    dead letter monitoring.

    Security note:
        EventIntegrityError must never be caught and silently swallowed by any
        component.  If a handler wraps its subscribe() call in a broad except,
        it must re-raise EventIntegrityError.  Swallowing integrity errors
        defeats the entire HMAC protection layer.
    """
    exception_code = EC_TOPO_EVENT_INTEGRITY
    is_hard_stop   = False


class EventSchemaError(TopologyException):
    """
    A received BusEnvelope could not be deserialized into its registered schema,
    or the envelope's structural format is invalid.

    Raised by _deserialize() after successful HMAC verification but before
    the handler is called.  The handler is never called.  The event is written
    to /store/dead_letters.jsonl with handler="__schema_validate__".

    Raised in three sub-cases (all caught as EventSchemaError by callers):

    1.  msgpack deserialization failure — envelope.value is not valid msgpack,
        or the top-level unpacked value is not a dict.  Indicates corruption
        or a producer serialization bug.

    2.  Schema construction failure — the unpacked dict does not match the
        dataclass constructor for the registered schema (wrong keys, wrong
        types, __post_init__ raises).  Indicates a schema version mismatch
        between producer and consumer, or a producer contract violation.

    3.  Structural envelope validation failure — a required header is missing,
        the schema_version is unsupported, or emit_timestamp is unparseable.
        Raised by EventEnvelopeValidator before msgpack deserialization.

    is_hard_stop = False: one schema failure is a dead letter.  Repeated schema
    failures for the same topic in the dead letter log indicate a schema
    migration was deployed to producers but not consumers (or vice versa) —
    this surfaces via index_daemon.py's dead letter monitoring.
    """
    exception_code = EC_TOPO_EVENT_SCHEMA
    is_hard_stop   = False


# ═════════════════════════════════════════════════════════════════════════════
# STORE WATCHDOG VIOLATIONS
#
# Mirrors store_watchdog.py :: StoreWatchdog
#
# These exceptions are raised by store_watchdog.py itself at the boundary
# points listed below.  They are not raised by component reload handlers —
# handler failures are caught internally by _safe_dispatch() and logged
# without raising.  Only watchdog infrastructure failures surface here.
#
# Raise sites:
#   WatchdogStartupError      — start() if inotify_init() fails
#   WatchdogInotifyExhausted  — start() if add_watch() hits fd limit
#   WatchdogHandlerTimeout    — _safe_dispatch() on asyncio.TimeoutError
#                               (logged internally; raised for Witness only)
#   WatchdogCircuitOpen       — _maybe_open_circuit() on threshold breach
#                               (logged internally; raised for Witness only)
#   WatchdogRegistrationError — register() after start(), or duplicate handler
#
# is_hard_stop semantics:
#   WatchdogStartupError / WatchdogInotifyExhausted: True.
#     The system cannot learn or compound without a functioning watchdog.
#     cold_start.py treats these as fatal startup failures and halts.
#   All others: False.
#     The watchdog is degraded but still running.  Witness alerts via the
#     WatchdogHealth snapshot.  The process continues serving (possibly with
#     stale weights) until cold_start.py can reset the circuit.
# ═════════════════════════════════════════════════════════════════════════════

class WatchdogError(KernelException):
    """
    Base class for all store watchdog exceptions.
    Never raised directly — use a concrete subclass.
    """
    exception_code = EC_CONTRACT_VIOLATION   # concrete subclasses override
    is_hard_stop   = False


class WatchdogStartupError(WatchdogError):
    """
    inotify_init() or add_watch() failed during watchdog.start().

    Causes:
        - /proc/sys/fs/inotify/max_user_instances exceeded (too many
          inotify file descriptors open process-wide)
        - The parent directory of a watched path does not exist and could
          not be created before start() was called
        - Permission denied on /store or a subdirectory

    is_hard_stop=True: without a working watchdog, no component will be
    notified of model or registry changes.  The system would serve stale
    weights silently until the next restart — which is worse than a clean
    halt.  cold_start.py must treat this as a fatal startup error.
    """
    exception_code = EC_WATCHDOG_STARTUP
    is_hard_stop   = True

    def __init__(
        self,
        *,
        message:  str,
        path:     Optional[str] = None,
        os_error: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self._path     = path
        self._os_error = os_error

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":     self._path,
            "os_error": self._os_error,
        })
        return d


class WatchdogInotifyExhausted(WatchdogStartupError):
    """
    /proc/sys/fs/inotify/max_user_watches exhausted.

    The per-process or system-wide inotify watch limit was reached when
    store_watchdog.py called add_watch() during start().  This is a
    distinct sub-case of WatchdogStartupError — the fix is an OS-level
    sysctl change, not a code change.

    Recovery:
        sysctl fs.inotify.max_user_watches=524288
        echo "fs.inotify.max_user_watches=524288" >> /etc/sysctl.conf

    is_hard_stop=True: inherits from WatchdogStartupError.
    """
    exception_code = EC_WATCHDOG_INOTIFY_EXHAUST
    is_hard_stop   = True

    def __init__(
        self,
        *,
        path:            str,
        current_watches: Optional[int] = None,
        os_error:        Optional[str] = None,
    ) -> None:
        super().__init__(
            message=(
                f"inotify fd limit exhausted adding watch for {path!r}.  "
                "Increase /proc/sys/fs/inotify/max_user_watches.  "
                f"os_error={os_error!r}"
            ),
            path=path,
            os_error=os_error,
        )
        self._current_watches = current_watches

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d["current_watches"] = self._current_watches
        return d


class WatchdogHandlerTimeout(WatchdogError):
    """
    A reload handler was cancelled after exceeding WATCHDOG_HANDLER_TIMEOUT_S.

    This exception is constructed and logged by _safe_dispatch() when
    asyncio.wait_for() raises TimeoutError.  The failure is counted against
    the circuit breaker for this handler.  After WATCHDOG_CIRCUIT_OPEN_THRESHOLD
    consecutive timeouts the circuit opens — see WatchdogCircuitOpen.

    This exception is not raised to callers — it is logged internally and
    used to construct the to_audit_dict() payload for Witness.  It is defined
    as a class so log analysis tools can reconstruct the type from audit records
    via exception_from_code(EC_WATCHDOG_HANDLER_TIMEOUT).

    Causes:
        - Component reload handler is deadlocked on a lock held by the hot path
        - Model file is being served from cold storage (NVMe spin-up latency)
        - PyTorch model load is racing with a simultaneous large file write

    is_hard_stop=False: the component continues serving stale weights.
    The watchdog is still watching the file; the next file-change event
    will attempt the reload again.
    """
    exception_code = EC_WATCHDOG_HANDLER_TIMEOUT
    is_hard_stop   = False

    def __init__(
        self,
        *,
        path:             str,
        handler_qualname: str,
        timeout_s:        float,
    ) -> None:
        super().__init__(
            f"Handler {handler_qualname!r} exceeded timeout {timeout_s}s "
            f"on path {path!r} and was cancelled."
        )
        self._path             = path
        self._handler_qualname = handler_qualname
        self._timeout_s        = timeout_s

    @property
    def handler_qualname(self) -> str:
        return self._handler_qualname

    @property
    def timeout_s(self) -> float:
        return self._timeout_s

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":             self._path,
            "handler_qualname": self._handler_qualname,
            "timeout_s":        self._timeout_s,
        })
        return d


class WatchdogCircuitOpen(WatchdogError):
    """
    A handler's circuit breaker tripped after WATCHDOG_CIRCUIT_OPEN_THRESHOLD
    consecutive failures (timeouts or exceptions).

    Once the circuit is open the handler will not be called on subsequent
    file-change events.  File changes for its watched path are silently
    dropped from that handler's perspective until the circuit is reset.

    The component owning the handler is now serving stale weights indefinitely.
    Witness must alert on is_circuit_open=True in the WatchdogHealth snapshot.

    This exception is constructed and logged by _maybe_open_circuit() when
    the threshold is breached.  Like WatchdogHandlerTimeout, it is not raised
    to callers but is defined as a class for audit reconstruction.

    Recovery:
        StoreWatchdog.reset_circuit(path, handler_qualname)
        — called by cold_start.py on process restart, or by an operator
          via admin tooling after the underlying failure is resolved.

    is_hard_stop=False: the process continues running.  Witness alerts and
    cold_start.py handles recovery.
    """
    exception_code = EC_WATCHDOG_CIRCUIT_OPEN
    is_hard_stop   = False

    def __init__(
        self,
        *,
        path:                 str,
        handler_qualname:     str,
        consecutive_failures: int,
        threshold:            int,
    ) -> None:
        super().__init__(
            f"Circuit opened for {handler_qualname!r} on {path!r} after "
            f"{consecutive_failures} consecutive failures "
            f"(threshold={threshold}).  Handler is now quarantined."
        )
        self._path                 = path
        self._handler_qualname     = handler_qualname
        self._consecutive_failures = consecutive_failures
        self._threshold            = threshold

    @property
    def handler_qualname(self) -> str:
        return self._handler_qualname

    @property
    def consecutive_failures(self) -> int:
        return self._consecutive_failures

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":                 self._path,
            "handler_qualname":     self._handler_qualname,
            "consecutive_failures": self._consecutive_failures,
            "threshold":            self._threshold,
        })
        return d


class WatchdogRegistrationError(WatchdogError):
    """
    register() was called after start().

    All register() calls must complete before watchdog.start() is called.
    A post-start call indicates a component initialize() method is being
    called a second time after the system is already running — a sequencing
    bug in the caller, not in the watchdog.

    is_hard_stop=False: the registration was rejected.  The caller's
    initialize() sequencing is broken but the watchdog itself is healthy.
    The component should not proceed — it will operate without its reload
    handler registered.
    """
    exception_code = EC_WATCHDOG_POST_START_REG
    is_hard_stop   = False

    def __init__(
        self,
        *,
        message:          str,
        path:             str,
        handler_qualname: str,
    ) -> None:
        super().__init__(message)
        self._path             = path
        self._handler_qualname = handler_qualname

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":             self._path,
            "handler_qualname": self._handler_qualname,
        })
        return d


class WatchdogDuplicateRegistrationError(WatchdogError):
    """
    The same handler object was registered for the same path twice.

    The same callable identity registered for the same path twice means
    a component's initialize() method has been called twice without an
    intervening stop().  This would cause every file-change event to
    invoke the same reload handler twice, doubling model reload work.

    Previously conflated with WatchdogRegistrationError under
    EC_WATCHDOG_POST_START_REG.  Split into its own class so log analysis
    tools can distinguish a duplicate-registration sequencing bug from a
    post-start registration bug — they have different remediation paths.

    is_hard_stop=False: the duplicate was rejected.  The first registration
    remains active.  The component should not proceed with the second
    initialize() call until stop() has been called.
    """
    exception_code = EC_WATCHDOG_DUPLICATE_REG
    is_hard_stop   = False

    def __init__(
        self,
        *,
        path:             str,
        handler_qualname: str,
    ) -> None:
        super().__init__(
            f"Handler {handler_qualname!r} is already registered for {path!r}. "
            "Duplicate registration rejected — component initialize() was called "
            "twice without an intervening stop()."
        )
        self._path             = path
        self._handler_qualname = handler_qualname

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":             self._path,
            "handler_qualname": self._handler_qualname,
        })
        return d


class WatchdogGhostEventSuppressed(WatchdogError):
    """
    An inotify event arrived for a path that no longer exists.

    Ghost events occur when a file is created and deleted faster than the
    inotify watch loop can dispatch the event.  By the time _safe_dispatch()
    runs, os.path.exists() returns False.  The event is suppressed rather
    than forwarded — invoking a reload handler for a file that is gone would
    produce a FileNotFoundError in every handler.

    This is logged at DEBUG in development and suppressed at INFO in
    production.  A burst of ghost events on structural_layer.pt or
    recipe_registry.mmap may indicate an upstream writer that is atomically
    replacing files via rename() — which is correct — or a writer that is
    deleting and recreating files non-atomically — which is a bug upstream.

    is_hard_stop=False: the watchdog continues operating normally.
    """
    exception_code = EC_WATCHDOG_GHOST_EVENT
    is_hard_stop   = False

    def __init__(
        self,
        *,
        path:       str,
        event_mask: Optional[int] = None,
    ) -> None:
        super().__init__(
            f"Ghost inotify event suppressed for {path!r} — path no longer exists. "
            f"event_mask={event_mask!r}."
        )
        self._path       = path
        self._event_mask = event_mask

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":       self._path,
            "event_mask": self._event_mask,
        })
        return d


class WatchdogWatchLoopError(WatchdogError):
    """
    An unexpected exception escaped the inotify watch loop.

    The watch loop in store_watchdog.py wraps dispatch with a broad
    except clause so that a single bad event cannot kill the loop.
    WatchdogWatchLoopError is constructed and logged when that clause
    fires — the loop continues, but the event that triggered the error
    was not dispatched.

    A single occurrence is informational.  Repeated occurrences on the
    same path indicate a persistent fault in the event pipeline that
    must be investigated.  Witness alerts on watch_loop_error_count > 0
    in the WatchdogHealth snapshot.

    is_hard_stop=False: the watch loop continues.  Stale weights are
    possible if the failing event was a model update.
    """
    exception_code = EC_WATCHDOG_LOOP_ERROR
    is_hard_stop   = False

    def __init__(
        self,
        *,
        path:      str,
        exc_type:  str,
        exc_value: str,
    ) -> None:
        super().__init__(
            f"Watch loop error on {path!r}: {exc_type}: {exc_value}. "
            "Event was not dispatched.  Watch loop continues."
        )
        self._path      = path
        self._exc_type  = exc_type
        self._exc_value = exc_value

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "path":      self._path,
            "exc_type":  self._exc_type,
            "exc_value": self._exc_value,
        })
        return d

# ═════════════════════════════════════════════════════════════════════════════
# CRAWLER LAYER EXCEPTIONS
#
# Mirrors contracts.py :: CrawlManifest, CrawlURL, RawFetchEvent,
#                         FetchAnomalyEvent, FrontierStats
#
# Handling law for all crawler exceptions:
#   fetcher.py catches every exception internally.
#   Fetch failures → FetchAnomalyEvent emitted to bus, manifest continues.
#   Infrastructure failures (Frontier, Cursor) → logged, execution continues.
#   ManifestExhaustedError is not a failure — it is the normal completion signal.
#
# No crawler exception is a hard stop for the AXIOM graph.
# The manifests always complete. They may emit many anomaly events. They complete.
# ═════════════════════════════════════════════════════════════════════════════

class CrawlerException(KernelException):
    """
    Base class for all crawler layer exceptions.
    Never raised directly — use a concrete subclass.

    Inherits KernelException so existing catch blocks that handle
    KernelException also catch crawler errors — no existing handlers
    require modification.
    """
    exception_code = EC_CONTRACT_VIOLATION  # concrete subclasses override
    is_hard_stop   = False


class FetchError(CrawlerException):
    """
    A fetch operation failed and produced a FetchAnomalyEvent instead of
    a RawFetchEvent. This is the base class for all fetch-path failures.

    FetchError subclasses represent the cause of the anomaly. The fetcher
    constructs the appropriate subclass, logs it, emits FetchAnomalyEvent,
    and continues the manifest. Never a hard stop.

    Never raise FetchError directly — use a concrete subclass.
    """
    exception_code = EC_CRAWLER_FETCH_FAILED
    is_hard_stop   = False

    def __init__(
        self,
        message:     str,
        *,
        url:         str,
        fetch_mode:  str,
        manifest_id: str,
        run_id:      str,
    ) -> None:
        super().__init__(message, run_id=run_id)
        self._url         = url
        self._fetch_mode  = fetch_mode
        self._manifest_id = manifest_id

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "url":         self._url,
            "fetch_mode":  self._fetch_mode,
            "manifest_id": self._manifest_id,
        })
        return d


class TorUnavailableError(FetchError):
    """
    The Tor SOCKS5 proxy at localhost:9050 is unreachable or refused the
    connection — either at fetcher initialization or during a CL3/CL4 fetch.

    On detection at initialize(): CL3 and CL4 are disabled for the session.
    All TOR and TOR_FULL mode URLs fall back to HEADLESS silently.
    On detection mid-fetch: emit FetchAnomalyEvent(anomaly_type="tor_unavailable").
    Do not retry with clearnet. Skip URL. Continue manifest.

    Mirrors: CL3/CL4 fetch path → SOCKS5 connection refused
    Handling: Not a hard stop. CL3/CL4 disabled for session.
    """
    exception_code = EC_CRAWLER_TOR_UNAVAILABLE
    is_hard_stop   = False

    def __init__(
        self,
        *,
        url:          str,
        manifest_id:  str,
        run_id:       str,
        socks_host:   str,
        socks_port:   int,
        os_error:     Optional[str],
    ) -> None:
        super().__init__(
            f"Tor SOCKS5 proxy at {socks_host}:{socks_port} is unavailable. "
            f"os_error={os_error or 'none'}. "
            "CL3/CL4 disabled for this session. TOR/TOR_FULL URLs fall back to HEADLESS.",
            url=url,
            fetch_mode="tor",
            manifest_id=manifest_id,
            run_id=run_id,
        )
        self._socks_host = socks_host
        self._socks_port = socks_port
        self._os_error   = os_error

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "socks_host": self._socks_host,
            "socks_port": self._socks_port,
            "os_error":   self._os_error,
        })
        return d


class PlaywrightError(FetchError):
    """
    A Playwright browser crash, navigation failure, or context error during
    a CL2, CL3, or CL4 fetch.

    On detection: emit FetchAnomalyEvent(anomaly_type="playwright_crash").
    Restart the browser context. Continue the manifest.
    A single Playwright crash does not disable CL2 for the session —
    context restart is mechanical recovery, not intelligent retry.

    Mirrors: CL2/CL3/CL4 fetch path → Playwright raises PlaywrightError
    Handling: Not a hard stop. Context restarted. Manifest continues.
    """
    exception_code = EC_CRAWLER_PLAYWRIGHT_CRASH
    is_hard_stop   = False

    def __init__(
        self,
        *,
        url:          str,
        fetch_mode:   str,
        manifest_id:  str,
        run_id:       str,
        pw_error:     str,
        context_recycled: bool,
    ) -> None:
        super().__init__(
            f"Playwright error during {fetch_mode} fetch: {pw_error}. "
            f"context_recycled={context_recycled}. "
            "FetchAnomalyEvent emitted. Manifest continues.",
            url=url,
            fetch_mode=fetch_mode,
            manifest_id=manifest_id,
            run_id=run_id,
        )
        self._pw_error        = pw_error
        self._context_recycled = context_recycled

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "pw_error":         self._pw_error,
            "context_recycled": self._context_recycled,
        })
        return d


class RateLimitViolationError(FetchError):
    """
    The fetcher received HTTP 429 — a rate limit was hit despite proactive
    pacing from rate_limiter.py. This is an architecture bug: the preparser's
    CrawlManifest encoded the wrong crawl_delay, or rate_limiter.py failed
    to enforce it.

    Not a runtime error in the conventional sense — it is a signal that
    crawl_planner.py produced an over-aggressive manifest for this domain.
    index_daemon subscribes to FetchAnomalyEvent("rate_limited") and adjusts
    the domain's RateLimitProfile on the next gradient step.

    Mirrors: HTTP 429 response on any CL fetch
    Handling: Not a hard stop. FetchAnomalyEvent emitted. No retry. No backoff.
    """
    exception_code = EC_CRAWLER_RATE_LIMIT_HIT
    is_hard_stop   = False

    def __init__(
        self,
        *,
        url:          str,
        fetch_mode:   str,
        manifest_id:  str,
        run_id:       str,
        domain:       str,
        current_rate: float,
    ) -> None:
        super().__init__(
            f"HTTP 429 received for {domain} despite rate_limiter pacing at "
            f"{current_rate:.3f} req/s. "
            "This is a CrawlManifest quality issue — crawl_planner.py produced "
            "an over-aggressive manifest. index_daemon will adjust on next gradient step. "
            "FetchAnomalyEvent emitted. No retry. No backoff.",
            url=url,
            fetch_mode=fetch_mode,
            manifest_id=manifest_id,
            run_id=run_id,
        )
        self._domain       = domain
        self._current_rate = current_rate

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "domain":       self._domain,
            "current_rate": round(self._current_rate, 4),
        })
        return d


class FrontierError(CrawlerException):
    """
    A SQLite operation in frontier.py failed — schema creation, batch insert,
    status update, or query failure.

    frontier.py raises this; fetcher.py catches it, logs it, and emits
    FetchAnomalyEvent for the affected URL. The manifest continues.
    A FrontierError for one URL does not stop the frontier — subsequent
    URLs are still pulled and attempted.

    Mirrors: frontier.py SQLite operations
    Handling: Not a hard stop. Fetcher catches, emits anomaly, continues.
    """
    exception_code = EC_CRAWLER_FRONTIER_ERROR
    is_hard_stop   = False

    def __init__(
        self,
        *,
        manifest_id:  str,
        operation:    str,   # "load_manifest" | "mark_done" | "mark_failed" | "resume" | ...
        db_error:     str,
        run_id:       Optional[str] = None,
    ) -> None:
        super().__init__(
            f"Frontier SQLite error during {operation!r}: {db_error}. "
            f"manifest_id={manifest_id}. "
            "Fetcher catches this and emits FetchAnomalyEvent. Manifest continues.",
            run_id=run_id,
        )
        self._manifest_id = manifest_id
        self._operation   = operation
        self._db_error    = db_error

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "manifest_id": self._manifest_id,
            "operation":   self._operation,
            "db_error":    self._db_error,
        })
        return d


class CursorError(CrawlerException):
    """
    A SQLite operation in crawl_cursor.py failed — checkpoint write or
    position read failure.

    crawl_cursor.py raises this; fetcher.py catches it, logs at WARNING,
    and skips the checkpoint for this interval. The manifest continues.
    A missed checkpoint means up to CURSOR_CHECKPOINT_INTERVAL URLs may
    be re-fetched on restart — acceptable.

    Mirrors: crawl_cursor.py SQLite operations
    Handling: Not a hard stop. Fetcher skips checkpoint. Manifest continues.
    Maximum duplicate fetch exposure: CURSOR_CHECKPOINT_INTERVAL (100) URLs.
    """
    exception_code = EC_CRAWLER_CURSOR_ERROR
    is_hard_stop   = False

    def __init__(
        self,
        *,
        manifest_id:  str,
        position:     int,
        operation:    str,   # "checkpoint" | "get_position" | "clear"
        db_error:     str,
    ) -> None:
        super().__init__(
            f"CrawlCursor SQLite error during {operation!r} at position {position}: "
            f"{db_error}. manifest_id={manifest_id}. "
            "Checkpoint skipped. Manifest continues. "
            f"Up to {100} URLs may be re-fetched if process dies before next checkpoint.",
        )
        self._manifest_id = manifest_id
        self._position    = position
        self._operation   = operation
        self._db_error    = db_error

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "manifest_id": self._manifest_id,
            "position":    self._position,
            "operation":   self._operation,
            "db_error":    self._db_error,
        })
        return d


class ManifestExhaustedError(CrawlerException):
    """
    All URLs in a CrawlManifest have been processed (done | failed | skipped).
    This is NOT a failure — it is the normal, expected completion signal.

    fetcher.py raises this internally to signal _execute_manifest() that
    the manifest loop should exit cleanly. It is caught immediately by
    _execute_manifest(), which emits ManifestCompleteEvent and returns.
    It never propagates outside fetcher.py.

    Mirrors: frontier.is_complete() returning True
    Handling: Caught internally by fetcher. ManifestCompleteEvent emitted.
    Not a hard stop. Not an error. The crawl succeeded.
    """
    exception_code = EC_CRAWLER_MANIFEST_EXHAUSTED
    is_hard_stop   = False

    def __init__(
        self,
        *,
        manifest_id: str,
        domain:      str,
        total_urls:  int,
        done:        int,
        failed:      int,
        skipped:     int,
    ) -> None:
        super().__init__(
            f"Manifest {manifest_id} for {domain} exhausted: "
            f"total={total_urls} done={done} failed={failed} skipped={skipped}. "
            "ManifestCompleteEvent will be emitted.",
        )
        self._manifest_id = manifest_id
        self._domain      = domain
        self._total_urls  = total_urls
        self._done        = done
        self._failed      = failed
        self._skipped     = skipped

    @property
    def success_rate(self) -> float:
        if self._total_urls == 0:
            return 0.0
        return (self._done + self._skipped) / self._total_urls

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "manifest_id":  self._manifest_id,
            "domain":       self._domain,
            "total_urls":   self._total_urls,
            "done":         self._done,
            "failed":       self._failed,
            "skipped":      self._skipped,
            "success_rate": round(self.success_rate, 4),
        })
        return d


# ═════════════════════════════════════════════════════════════════════════════
# CROSS-LANGUAGE RUNTIME EXCEPTIONS
#
# These exceptions represent failures reported by Go, C, CUDA/C, and Rust
# components at the Python boundary. Native code never invents ad-hoc error
# strings for bus routing; it maps failures to these stable codes.
# ═════════════════════════════════════════════════════════════════════════════

class CrossLanguageRuntimeError(KernelException):
    """Base class for non-Python component failures crossing into Python."""
    exception_code = EC_CONTRACT_VIOLATION
    is_hard_stop   = False

    def __init__(
        self,
        *,
        component: str,
        operation: str,
        detail: str,
        run_id: Optional[str] = None,
        topology_class: Optional[str] = None,
    ) -> None:
        super().__init__(
            f"{component} failed during {operation}: {detail}",
            run_id=run_id,
            topology_class=topology_class,
        )
        self._component = component
        self._operation = operation
        self._detail    = detail

    def to_audit_dict(self) -> Dict[str, object]:
        d = super().to_audit_dict()
        d.update({
            "component": self._component,
            "operation": self._operation,
            "detail": self._detail,
        })
        return d


class DomainAnalysisFailed(CrossLanguageRuntimeError):
    """Go domain_analyzer failed to produce a domain fingerprint/map."""
    exception_code = EC_PREPARSER_DOMAIN_ANALYSIS


class CrawlPlanFailed(CrossLanguageRuntimeError):
    """Go crawl_planner failed to produce a valid CrawlManifest."""
    exception_code = EC_PREPARSER_CRAWL_PLAN


class SignalExtractionFailed(CrossLanguageRuntimeError):
    """Go signal_extractor failed to summarize a sanitized signal."""
    exception_code = EC_PREPARSER_SIGNAL_EXTRACT


class RecipeValidationServiceFailed(CrossLanguageRuntimeError):
    """Go recipe_validator failed while validating compiled recipes."""
    exception_code = EC_PREPARSER_RECIPE_VALIDATE


class NativeStripEngineFailed(CrossLanguageRuntimeError):
    """C strip_engine failed to execute or validate a recipe."""
    exception_code = EC_NATIVE_STRIP_ENGINE


class NativeBatchRunnerFailed(CrossLanguageRuntimeError):
    """C batch_runner failed while processing an NDJSON batch."""
    exception_code = EC_NATIVE_BATCH_RUNNER


class NativePhaseDaemonFailed(CrossLanguageRuntimeError):
    """C phase_daemon failed while scanning or promoting phase state."""
    exception_code = EC_NATIVE_PHASE_DAEMON


class NativeStoreSentinelFailed(CrossLanguageRuntimeError):
    """C store_sentinel failed while checking mmap/weight artifacts."""
    exception_code = EC_NATIVE_STORE_SENTINEL


class OfflineGPUUnavailable(CrossLanguageRuntimeError):
    """CUDA offline layer could not access a required GPU context."""
    exception_code = EC_OFFLINE_GPU_UNAVAILABLE


class OfflineGradientStepFailed(CrossLanguageRuntimeError):
    """CUDA/C offline layer failed during gradient accumulation or update."""
    exception_code = EC_OFFLINE_GRADIENT_STEP


class OfflineWeightPublishFailed(CrossLanguageRuntimeError):
    """CUDA/C offline layer failed to publish weights via staging/rename."""
    exception_code = EC_OFFLINE_WEIGHT_PUBLISH
    is_hard_stop   = True


class ColdStartValidationFailed(CrossLanguageRuntimeError):
    """cold_start.py or cold_start_c rejected the runtime environment."""
    exception_code = EC_COLD_START_VALIDATION
    is_hard_stop   = True


class ColdStartSecurityFailed(CrossLanguageRuntimeError):
    """cold_start_c detected a security/isolation setup failure."""
    exception_code = EC_COLD_START_SECURITY
    is_hard_stop   = True


class InterfaceCommandInvalid(CrossLanguageRuntimeError):
    """tag/interface.py received an invalid public command."""
    exception_code = EC_INTERFACE_COMMAND_INVALID


class InterfaceDispatchFailed(CrossLanguageRuntimeError):
    """tag/interface.py failed while dispatching a valid command."""
    exception_code = EC_INTERFACE_DISPATCH_FAILED


# ═════════════════════════════════════════════════════════════════════════════
# EXCEPTION REGISTRY
#
# Flat mapping from exception_code → exception class.
# Used by log analysis tools and checkpoint_monitor.py to reconstruct
# exception types from serialized audit records without importing the full
# exception module hierarchy.
#
# Every concrete exception class that can appear in a to_audit_dict() output
# is registered here. Abstract bases (KernelException, RecipeMountError, etc.)
# are not registered — they are never raised directly.
# ═════════════════════════════════════════════════════════════════════════════

EXCEPTION_REGISTRY: Dict[str, type] = {
    EC_INPUT_EMPTY:            RawContentEmpty,
    EC_INPUT_OVERFLOW:         RawContentOverflow,
    EC_INPUT_ENCODING:         StdinEncodingError,
    EC_INPUT_CONTENT_TYPE:     InvalidContentType,
    EC_INPUT_URL_INVALID:      InvalidSourceURL,
    EC_INPUT_TOPOLOGY_INVALID: InvalidTopologyClass,
    EC_INPUT_RUN_ID_INVALID:   InvalidRunID,
    EC_OUTPUT_EMPTY:           EmptyExtractionError,
    EC_OUTPUT_TIMEOUT:         SubprocessTimeout,
    EC_OUTPUT_MEASUREMENT:     OutputMeasurementError,
    EC_CONTAINER_SPAWN:        ContainerSpawnError,
    EC_PROCESS_STATE:          ProcessStateCorruption,
    EC_RECIPE_NOT_FOUND:       RecipeNotFound,
    EC_RECIPE_HASH_MISMATCH:   RecipeHashMismatch,
    EC_RECIPE_INJECTION:       RecipeInjectionAttempt,
    EC_RECIPE_STRUCTURAL:      RecipeStructuralViolation,
    EC_RECIPE_DRY_RUN:         RecipeDryRunFailure,
    EC_REGISTRY_CORRUPT:       RecipeRegistryCorrupt,
    EC_REGISTRY_OVERWRITE:     HardcodedRecipeOverwriteAttempt,
    EC_MANIFEST_CORRUPT:       RecipeManifestCorruption,
    EC_CHECKPOINT_WRITE:       CheckpointWriteError,
    EC_CHECKPOINT_INTEGRITY:   CheckpointIntegrityError,
    EC_CHECKPOINT_ROTATION:    CheckpointRotationError,
    EC_CROND_DEAD:             CrondProcessError,
    EC_CHECKPOINT_CORRUPT:     CheckpointCorruptionError,
    EC_RESTORE_FAILURE:        RestoreFailure,
    EC_RESTORE_EXHAUSTED:      CheckpointExhaustionError,
    EC_RESTORE_PARTIAL:        RestorePartialError,
    EC_FEEDBACK_EMISSION:      FeedbackEmissionError,
    EC_TELEMETRY_EMISSION:     TelemetryEmissionError,
    EC_AUDIT_EMISSION:         AuditEmissionError,
    EC_DAEMON_QUERY:           DaemonQueryError,
    EC_DAEMON_RESPONSE:        DaemonResponseError,
    EC_CONTRACT_VIOLATION:     ContractViolation,
    # Topology layer
    EC_TOPO_CLASSIFIER_NOT_INIT:    ClassifierModelNotInitialized,
    EC_TOPO_CONFIDENCE_TOO_LOW:     ClassificationConfidenceTooLow,
    EC_TOPO_WINDOW_TOO_SMALL:       ClassificationWindowTooSmall,
    EC_TOPO_RECIPE_COMPILE_FAILED:  RecipeCompilationFailed,
    EC_TOPO_RECIPE_VERSION_CONFLICT: RecipeVersionConflict,
    EC_TOPO_WLP_QUERY_FAILED:       WLPQueryFailed,
    EC_TOPO_SANITIZER_INPUT:        SanitizerInputError,
    EC_TOPO_SURPRISE_HISTORY:       SurpriseHistoryCorrupted,
    EC_TOPO_PHANTOM_FETCH_FAILED:   PhantomFetchFailed,
    EC_TOPO_PHANTOM_RENDER_TIMEOUT: PhantomRenderTimeout,
    EC_TOPO_PHANTOM_FRICTION:       PhantomFrictionDetected,
    EC_TOPO_GRADIENT_STEP_FAILED:   GradientStepFailed,
    EC_TOPO_PHASE_MMAP_CORRUPTED:   PhaseMMapCorrupted,
    EC_TOPO_BUS_SUBSCRIPTION:       EventBusSubscriptionError,
    EC_TOPO_EVENT_DISPATCH:         EventDispatchFailed,
    EC_TOPO_KAFKA_UNAVAILABLE:      KafkaSinkUnavailable,
    EC_TOPO_EVENT_INTEGRITY:        EventIntegrityError,
    EC_TOPO_EVENT_SCHEMA:           EventSchemaError,
    # Store watchdog layer
    EC_WATCHDOG_STARTUP:            WatchdogStartupError,
    EC_WATCHDOG_INOTIFY_EXHAUST:    WatchdogInotifyExhausted,
    EC_WATCHDOG_HANDLER_TIMEOUT:    WatchdogHandlerTimeout,
    EC_WATCHDOG_CIRCUIT_OPEN:       WatchdogCircuitOpen,
    EC_WATCHDOG_POST_START_REG:     WatchdogRegistrationError,
    EC_WATCHDOG_DUPLICATE_REG:      WatchdogDuplicateRegistrationError,
    EC_WATCHDOG_GHOST_EVENT:        WatchdogGhostEventSuppressed,
    EC_WATCHDOG_LOOP_ERROR:         WatchdogWatchLoopError,
    # Crawler layer
    EC_CRAWLER_FETCH_FAILED:       FetchError,
    EC_CRAWLER_TOR_UNAVAILABLE:    TorUnavailableError,
    EC_CRAWLER_PLAYWRIGHT_CRASH:   PlaywrightError,
    EC_CRAWLER_RATE_LIMIT_HIT:     RateLimitViolationError,
    EC_CRAWLER_FRONTIER_ERROR:     FrontierError,
    EC_CRAWLER_CURSOR_ERROR:       CursorError,
    EC_CRAWLER_MANIFEST_EXHAUSTED: ManifestExhaustedError,
    # Cross-language runtime layer
    EC_PREPARSER_DOMAIN_ANALYSIS: DomainAnalysisFailed,
    EC_PREPARSER_CRAWL_PLAN: CrawlPlanFailed,
    EC_PREPARSER_SIGNAL_EXTRACT: SignalExtractionFailed,
    EC_PREPARSER_RECIPE_VALIDATE: RecipeValidationServiceFailed,
    EC_NATIVE_STRIP_ENGINE: NativeStripEngineFailed,
    EC_NATIVE_BATCH_RUNNER: NativeBatchRunnerFailed,
    EC_NATIVE_PHASE_DAEMON: NativePhaseDaemonFailed,
    EC_NATIVE_STORE_SENTINEL: NativeStoreSentinelFailed,
    EC_OFFLINE_GPU_UNAVAILABLE: OfflineGPUUnavailable,
    EC_OFFLINE_GRADIENT_STEP: OfflineGradientStepFailed,
    EC_OFFLINE_WEIGHT_PUBLISH: OfflineWeightPublishFailed,
    EC_COLD_START_VALIDATION: ColdStartValidationFailed,
    EC_COLD_START_SECURITY: ColdStartSecurityFailed,
    EC_INTERFACE_COMMAND_INVALID: InterfaceCommandInvalid,
    EC_INTERFACE_DISPATCH_FAILED: InterfaceDispatchFailed,
}


def exception_from_code(code: str) -> Optional[type]:
    """
    Look up an exception class by its exception_code.
    Returns None for unknown codes rather than raising — callers are
    typically log analysis tools that must tolerate unknown codes gracefully.
    """
    return EXCEPTION_REGISTRY.get(code)


# ═════════════════════════════════════════════════════════════════════════════
# HANDLE CLASSIFICATION HELPERS
#
# pipeline.py catch clauses use these to classify exceptions without
# importing every concrete class individually. The two questions pipeline.py
# must always answer about any KernelException it catches:
#   1. Is this a hard stop? (re-raise vs. degrade)
#   2. What exception code does this carry? (for structured log routing)
# ═════════════════════════════════════════════════════════════════════════════

def is_hard_stop(exc: BaseException) -> bool:
    """
    True if this exception must be re-raised to the AXIOM graph.
    False if pipeline.py should catch, log, and return empty KernelOutput.

    Equivalent to checking exc.is_hard_stop, but safe for use with
    non-KernelException BaseException instances (which are always hard stops —
    they represent conditions the kernel did not anticipate at all).
    """
    if isinstance(exc, KernelException):
        return exc.is_hard_stop
    # Non-KernelException BaseExceptions are not handled — always re-raise.
    return True


def is_security_event(exc: BaseException) -> bool:
    """
    True if this exception requires immediate security audit and Witness alert,
    regardless of whether it is a hard stop.

    Security events: RecipeInjectionAttempt, RecipeHashMismatch (is_tamper_event),
    HardcodedRecipeOverwriteAttempt, AuditEmissionError (is_security_critical).
    """
    if isinstance(exc, RecipeInjectionAttempt):
        return True
    if isinstance(exc, RecipeHashMismatch) and exc.is_tamper_event:
        return True
    if isinstance(exc, HardcodedRecipeOverwriteAttempt):
        return True
    if isinstance(exc, AuditEmissionError) and exc.is_security_critical:
        return True
    return False


def classify(exc: BaseException) -> Dict[str, object]:
    """
    Produce a classification dict for any exception caught by pipeline.py.
    Used to route exceptions to the correct handler and log category without
    a cascade of isinstance checks at every call site.

    Returns a dict with:
      exception_code    — for log filtering
      exception_class   — for human-readable logs
      is_hard_stop      — True → re-raise; False → degrade to empty output
      is_security_event — True → alert Witness immediately
      audit_dict        — full to_audit_dict() if available, else basic dict
    """
    if isinstance(exc, KernelException):
        return {
            "exception_code":    exc.exception_code,
            "exception_class":   type(exc).__name__,
            "is_hard_stop":      exc.is_hard_stop,
            "is_security_event": is_security_event(exc),
            "audit_dict":        exc.to_audit_dict(),
        }
    # Bare BaseException — not anticipated by the kernel.
    return {
        "exception_code":    EC_CONTRACT_VIOLATION,
        "exception_class":   type(exc).__name__,
        "is_hard_stop":      True,
        "is_security_event": False,
        "audit_dict": {
            "exception_code":  EC_CONTRACT_VIOLATION,
            "exception_class": type(exc).__name__,
            "message":         str(exc),
            "is_hard_stop":    True,
        },
    }
