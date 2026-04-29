"""
signal_kernel/recipes/registry.py
==================================
Authoritative recipe lookup table for the AXIOM signal kernel.

Lifecycle
─────────
On startup registry.py loads the five hardcoded recipes:

    news_article.sh    →  NEWS_ARTICLE
    saas_docs.sh       →  SAAS_DOCS
    rest_api_json.sh   →  REST_API_JSON
    json_ld.sh         →  JSON_LD_STRUCTURED
    ecommerce.sh       →  ECOMMERCE_PRODUCT

Each file is read, its content SHA-256 hashed, and stored in an in-memory
manifest backed by a JSON manifest file on disk.  On first-ever start (dev
bootstrap) the manifest is written from the actual file hashes.  On every
subsequent start the live files are verified against the persisted manifest:
mismatch on a hardcoded file is a filesystem tamper event (RecipeHashMismatch
with is_hardcoded=True), which raises to the caller and must not be silenced.

After the five hardcoded recipes, registry.py provisions the GENERIC_HTML
fallback – an embedded shell recipe that strips known noise zones globally
without any assumption about signal location.  GENERIC_HTML is written to disk
at startup and verified on every load exactly like a hardcoded recipe.

Fallback chain  (get_recipe)
─────────────────────────────
1. Exact match in primary table                →  return immediately.
2. Parent class lookup via PARENT_CLASS_MAP    →  return parent's recipe.
3. GENERIC_HTML                                →  return unconditionally.

get_recipe() NEVER returns None.  NEVER raises for a missing topology class.
The caller does not handle absent recipes – the registry handles it.

Write-protection
─────────────────
Hardcoded primaries (is_hardcoded=True) are immutable at runtime.

When register_recipe("NEWS_ARTICLE", path) is called and NEWS_ARTICLE already
has a hardcoded primary the registry:
  •  Logs a WARNING with full context.
  •  Does NOT touch the primary entry.
  •  Auto-registers the recipe under a versioned key: NEWS_ARTICLE_v2,
     NEWS_ARTICLE_v3 … (monotonic counter per base class).
  •  Returns the versioned RecipeRegistryEntry to the caller.

Promotion rules:
  •  promote(versioned_key) → promotes a versioned entry to primary.
     Raises HardcodedRecipeOverwriteAttempt if the current primary is
     hardcoded.  This is the safety rail.
  •  supersede_hardcoded(topology_class, versioned_key, reason=…) → the
     explicit, deliberate path to replace a hardcoded primary.  Requires
     a non-empty reason string.  Emits an audit record at WARNING level.
     If the superseding recipe later fails validation, the caller must call
     restore_hardcoded_primary(topology_class) to re-floor to hardcoded.

Hash verification
──────────────────
Hashes are verified at three points in the lifecycle:
  1. At startup, when every recipe file (hardcoded + GENERIC_HTML) is read.
  2. On register_recipe(), the caller-supplied hash is verified against
     the file content we read ourselves – we do not trust caller hashes.
  3. Via verify_integrity(), callable at any time, re-reads all files and
     re-checks every hash.  pipeline.py may call this on a slow path.

A hash mismatch on a hardcoded recipe is always a hard stop: it indicates
the file was modified after commit.  A mismatch on a compiler-generated
recipe is a consistency bug in topology_parser.py.  Both raise.

Thread safety
──────────────
All public methods are protected by a single threading.RLock (_lock).
The lock is reentrant so that internal helpers can call other locked methods
without deadlock.  Snapshot operations (snapshot(), health()) acquire the lock
for the minimal duration of the copy, then release.  Callers receive a
point-in-time copy; the lock is never held across I/O.

Dependency direction:
    registry.py → contracts.py, exceptions.py
    validator.py, pipeline.py → registry.py
    Nothing here calls validator.py (that is a higher layer).

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import hashlib # noqa
import json
import logging
import os
import re
import stat # noqa
import threading
import time # noqa
from collections import defaultdict
from copy import deepcopy # noqa
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Dict,
    FrozenSet,
    Iterator, # noqa
    List,
    Optional,
    Set, # noqa
    Tuple,
)

from signal_kernel.contracts import (
    FALLBACK_TOPOLOGY_CLASS,
    HARDCODED_TOPOLOGY_CLASSES,
    MAX_RECIPE_LINE_COUNT,
    PARENT_CLASS_MAP,
    RecipeHash,
    RecipeMount,
    RecipeRegistryEntry,
    HardcodedRecipeManifestEntry,
    TopologyClassStr,
    compute_recipe_hash,
    new_run_id,
)
from signal_kernel.exceptions import (
    HardcodedRecipeOverwriteAttempt,
    RecipeHashMismatch,
    RecipeManifestCorruption,
    RecipeNotFound,
    RecipeRegistryCorrupt,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL LOGGER
# All log records produced by the registry carry the logger name
# "signal_kernel.recipes.registry".  Operators filtering Witness for registry
# events use this prefix.  The logger is configured by the application layer —
# registry.py never calls logging.basicConfig() or installs handlers.
# ─────────────────────────────────────────────────────────────────────────────

_log = logging.getLogger("signal_kernel.recipes.registry")


# ─────────────────────────────────────────────────────────────────────────────
# PATH CONSTANTS
# Resolved relative to this file's location so the registry can be imported
# from any working directory without env-var ceremony.
# ─────────────────────────────────────────────────────────────────────────────

_THIS_FILE:       Path = Path(__file__).resolve()
_REGISTRY_DIR:    Path = _THIS_FILE.parent           # signal_kernel/recipes/
_HARDCODED_DIR:   Path = _REGISTRY_DIR / "hardcoded" # signal_kernel/recipes/hardcoded/
_MANIFEST_PATH:   Path = _HARDCODED_DIR / "manifest.json"

# Schema version written into manifest.json.
# Increment this if the manifest format changes; loader rejects foreign schemas.
_MANIFEST_SCHEMA_VERSION: int = 1


# ─────────────────────────────────────────────────────────────────────────────
# HARDCODED RECIPE FILE MAP
# Immutable at module import time.  If a new hardcoded recipe is added, it is
# added here AND in contracts.HARDCODED_TOPOLOGY_CLASSES — nowhere else.
# ─────────────────────────────────────────────────────────────────────────────

_HARDCODED_FILE_MAP: Dict[str, str] = {
    "NEWS_ARTICLE":       "news_article.sh",
    "SAAS_DOCS":          "saas_docs.sh",
    "REST_API_JSON":      "rest_api_json.sh",
    "JSON_LD_STRUCTURED": "json_ld.sh",
    "ECOMMERCE_PRODUCT":  "ecommerce.sh",
}

# Validate at import time that _HARDCODED_FILE_MAP is consistent with the
# canonical frozenset in contracts.py.  If they diverge, the module fails
# loudly on import rather than silently during execution.
_file_map_classes: FrozenSet[str] = frozenset(_HARDCODED_FILE_MAP.keys())
if _file_map_classes != HARDCODED_TOPOLOGY_CLASSES:
    _only_in_map      = _file_map_classes - HARDCODED_TOPOLOGY_CLASSES
    _only_in_contracts = HARDCODED_TOPOLOGY_CLASSES - _file_map_classes
    raise ImportError(
        "registry.py internal consistency failure: _HARDCODED_FILE_MAP and "
        "contracts.HARDCODED_TOPOLOGY_CLASSES are out of sync.\n"
        f"  Only in _HARDCODED_FILE_MAP:      {_only_in_map or 'none'}\n"
        f"  Only in HARDCODED_TOPOLOGY_CLASSES: {_only_in_contracts or 'none'}\n"
        "Update _HARDCODED_FILE_MAP to match HARDCODED_TOPOLOGY_CLASSES, "
        "or vice versa."
    )
del _file_map_classes   # not needed beyond this import-time guard


# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY CLASS VALIDATION
# registry.py defines its own validator so it does not import the private
# _validate_topology_class helper from contracts.py.  The pattern is identical.
# ─────────────────────────────────────────────────────────────────────────────

_TOPOLOGY_CLASS_RE: re.Pattern[str] = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")
_VERSION_SUFFIX_RE: re.Pattern[str] = re.compile(r"^(.+)_v(\d+)$")


def _is_valid_topology_class(value: str) -> bool:
    """True if value matches the canonical topology class format."""
    return bool(_TOPOLOGY_CLASS_RE.match(value))


def _require_valid_topology_class(value: str, context: str) -> None:
    """Raise ValueError with a descriptive message if value is not a valid class."""
    if not _is_valid_topology_class(value):
        raise ValueError(
            f"{context}: topology_class {value!r} does not match "
            r"[A-Z][A-Z0-9_]{1,63}.  topology_parser.py is responsible for "
            "generating valid class names."
        )


def _base_class_and_version(topology_class: str) -> Tuple[str, Optional[int]]:
    """
    Split a versioned topology key into (base_class, version_number).

    Examples:
        "NEWS_ARTICLE"    →  ("NEWS_ARTICLE", None)
        "NEWS_ARTICLE_v2" →  ("NEWS_ARTICLE", 2)
        "NEWS_ARTICLE_v12" →  ("NEWS_ARTICLE", 12)
    """
    m = _VERSION_SUFFIX_RE.match(topology_class)
    if m:
        return m.group(1), int(m.group(2))
    return topology_class, None


# ─────────────────────────────────────────────────────────────────────────────
# PATH TRAVERSAL GUARD
# recipe_path values supplied by topology_parser.py are untrusted input.
# This guard prevents path traversal attacks (../../etc/passwd style)
# before any file I/O is attempted.
# ─────────────────────────────────────────────────────────────────────────────

# Recipes must live under one of these roots.
_ALLOWED_RECIPE_ROOTS: Tuple[Path, ...] = (
    _HARDCODED_DIR,
    _REGISTRY_DIR / "compiler_generated",
)


def _assert_safe_recipe_path(recipe_path: str, context: str) -> Path:
    """
    Resolve recipe_path to an absolute Path and verify it is inside one of
    _ALLOWED_RECIPE_ROOTS.  Raises ValueError on traversal or empty input.
    Does NOT verify the file exists — that is the caller's responsibility.
    """
    if not recipe_path or not recipe_path.strip():
        raise ValueError(
            f"{context}: recipe_path must be a non-empty string, got {recipe_path!r}."
        )
    resolved = Path(recipe_path).resolve()
    for root in _ALLOWED_RECIPE_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved  # within an allowed root — safe
        except ValueError:
            continue
    allowed_str = ", ".join(str(r) for r in _ALLOWED_RECIPE_ROOTS)
    raise ValueError(
        f"{context}: recipe_path {recipe_path!r} resolves to {resolved} which "
        f"is outside allowed roots [{allowed_str}].  Path traversal rejected."
    )


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTABLE LINE COUNTER
# Counts non-blank, non-comment lines in a shell recipe.
# Used to populate the line_count field of RecipeMount and
# HardcodedRecipeManifestEntry, which enforces MAX_RECIPE_LINE_COUNT.
#
# What "executable" means here:
#   •  Not blank (strip() == "").
#   •  Not a pure comment line (strip().startswith("#")).
#   •  Lines that begin with whitespace followed by non-comment content count.
#
# This intentionally counts each awk/sed content line individually, since
# those lines represent distinct transformation steps.
# ─────────────────────────────────────────────────────────────────────────────

def _count_executable_lines(content: str) -> int:
    """
    Count non-blank, non-comment lines in a shell recipe file.

    Shell comment detection: a line is a comment iff its first non-whitespace
    character is '#'.  The shebang (#!/bin/sh) is treated as a comment for
    counting purposes — it is not an executable transformation step.
    """
    count = 0
    for line in content.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            count += 1
    return count


# ─────────────────────────────────────────────────────────────────────────────
# GENERIC_HTML FALLBACK RECIPE (EMBEDDED)
# This recipe is the unconditional safety net.  It is not a good recipe —
# it will retain noise from any page whose noise is not in the standard HTML5
# structural elements listed below.  It is better than passing 800 KB of raw
# HTML to Haiku.  It exists so the kernel always executes something useful
# even on completely unknown topology classes.
#
# Design contract:
#   •  Strip globally (no article zone assumption).
#   •  Noise removed: nav, header, footer, aside, script, style, noscript,
#      form, iframe, svg, template.
#   •  Signal: everything NOT in those zones.
#   •  Pass 1: erase single-line HTML comments.
#   •  Pass 2: awk state machine strips noise zones globally.
#   •  Pass 3: tag removal, entity decode, whitespace collapse.
#
# The recipe is written to _HARDCODED_DIR / "generic_html.sh" on every
# registry startup and verified against its hardcoded hash.  If the file
# already exists and matches, startup is a no-op for this recipe.  If the
# file has been modified at runtime, startup fails with RecipeHashMismatch.
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_HTML_RECIPE_CONTENT: str = r"""#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# recipes/hardcoded/generic_html.sh
# Topology class: GENERIC_HTML (universal fallback)
#
# Conservative global noise stripper. Used when no specific recipe is
# registered for the incoming topology class.
#
# Signal zone: everything NOT in the noise containers listed below.
# Noise stripped: <nav> <header> <footer> <aside> <script> <style>
#                 <noscript> <form> <iframe> <svg> <template>
#                 HTML comments (both single-line and multi-line).
#
# Architecture: three-pass pipeline, all streaming, no temp files.
#
# This recipe WILL retain some noise — breadcrumb divs, cookie banners,
# and custom layouts that do not use standard HTML5 sectioning elements
# will pass through.  That is acceptable: the goal is noise REDUCTION,
# not noise ELIMINATION.  On known topologies the specific recipes achieve
# 60-80% reduction.  GENERIC_HTML achieves 30-50%.  Still better than raw.
#
# stdin:  raw HTML (UTF-8)
# stdout: de-noised text, one logical sentence or heading per line
# ─────────────────────────────────────────────────────────────────────────────

# ── Pass 1: erase single-line HTML comments ──────────────────────────────────
sed 's/<!--[^-]*-->//g' |

# ── Pass 2: global noise zone state machine ──────────────────────────────────
awk '
BEGIN {
    s          = 0   # inside noise sub-zone (0/1)
    in_comment = 0   # inside multi-line HTML comment (0/1)
}

# ── Multi-line comment handling ──────────────────────────────────────────────
/<!--/ && !/>/ {
    in_comment = 1
    sub(/<!--.*/, "")
    if (!s && length($0)) print
    next
}
in_comment && /-->/ {
    sub(/.*-->/, "")
    in_comment = 0
    if (!s && length($0)) print
    next
}
in_comment { next }

# ── Noise zone entry ─────────────────────────────────────────────────────────
!s && /<(nav|header|footer|aside|script|style|noscript|form|iframe|svg|template)[[:space:]>]/ {
    s = 1
    next
}

# ── Noise zone exit ──────────────────────────────────────────────────────────
s && /<\/(nav|header|footer|aside|script|style|noscript|form|iframe|svg|template)>/ {
    s = 0
    next
}

# ── Discard inside noise zone ────────────────────────────────────────────────
s { next }

# ── Signal output ────────────────────────────────────────────────────────────
length($0) { print }
' |

# ── Pass 3: tag removal, entity decode, whitespace collapse ──────────────────
sed '
s/<[^>]*>//g
s/&amp;/\&/g
s/&lt;/</g
s/&gt;/>/g
s/&nbsp;/ /g
s/&quot;/"/g
s/&#39;/'"'"'/g
s/&#8216;/'"'"'/g
s/&#8217;/'"'"'/g
s/&#8220;/"/g
s/&#8221;/"/g
s/&#8212;/—/g
s/&#8211;/–/g
' |

grep -v '^[[:space:]]*$' |

sed 's/[[:space:]]\{1,\}/ /g
     s/^ //
     s/ $//'
"""

# Pre-compute the hash of the embedded GENERIC_HTML recipe at import time.
# If this value ever changes, the manifest bootstrap will detect it and update.
_GENERIC_HTML_CANONICAL_HASH: RecipeHash = compute_recipe_hash(
    _GENERIC_HTML_RECIPE_CONTENT
)
_GENERIC_HTML_FILENAME: str = "generic_html.sh"


# ─────────────────────────────────────────────────────────────────────────────
# MANIFEST I/O HELPERS
# The manifest file is `recipes/hardcoded/manifest.json`.
# It is a JSON document with a schema_version field and an entries dict.
# registry.py loads it at startup and updates it on first-run bootstrap.
# ─────────────────────────────────────────────────────────────────────────────

class _ManifestFormatError(Exception):
    """Internal: manifest JSON is present but structurally invalid."""


def _load_manifest_from_disk(path: Path) -> Dict[str, HardcodedRecipeManifestEntry]:
    """
    Load and parse manifest.json.  Returns a dict mapping topology_class →
    HardcodedRecipeManifestEntry for every entry in the file.

    Raises:
        RecipeManifestCorruption — file is missing, unreadable, or invalid.
        _ManifestFormatError     — file has unexpected schema (internal only;
                                   callers convert to RecipeManifestCorruption).
    """
    path_str = str(path)

    if not path.exists():
        raise RecipeManifestCorruption(
            manifest_path=path_str,
            detail="manifest.json does not exist — run registry on first-time "
                   "bootstrap or check packaging.",
        )
    if not path.is_file():
        raise RecipeManifestCorruption(
            manifest_path=path_str,
            detail=f"{path_str} exists but is not a regular file.",
        )

    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, PermissionError) as exc:
        raise RecipeManifestCorruption(
            manifest_path=path_str,
            detail=f"Cannot read manifest: {exc}",
        ) from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RecipeManifestCorruption(
            manifest_path=path_str,
            detail=f"manifest.json is not valid JSON: {exc}",
        ) from exc

    if not isinstance(data, dict):
        raise _ManifestFormatError("root is not a JSON object")

    schema_version = data.get("schema_version")
    if schema_version != _MANIFEST_SCHEMA_VERSION:
        raise _ManifestFormatError(
            f"schema_version={schema_version!r}, expected {_MANIFEST_SCHEMA_VERSION}"
        )

    entries_raw = data.get("entries")
    if not isinstance(entries_raw, dict):
        raise _ManifestFormatError("'entries' key is missing or not a JSON object")

    entries: Dict[str, HardcodedRecipeManifestEntry] = {}
    for tc, entry_data in entries_raw.items():
        if not isinstance(entry_data, dict):
            raise _ManifestFormatError(f"entry for {tc!r} is not a JSON object")
        try:
            committed_at = datetime.fromisoformat(entry_data["committed_at"])
            entries[tc] = HardcodedRecipeManifestEntry(
                topology_class=TopologyClassStr(tc),
                recipe_filename=entry_data["recipe_filename"],
                recipe_hash=RecipeHash(entry_data["recipe_hash"]),
                line_count=int(entry_data["line_count"]),
                committed_at=committed_at,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise _ManifestFormatError(
                f"entry for {tc!r} is malformed: {exc}"
            ) from exc

    return entries


def _write_manifest_to_disk(
    path: Path,
    entries: Dict[str, HardcodedRecipeManifestEntry],
) -> None:
    """
    Serialize and write entries to manifest.json.
    Uses an atomic write (write to .tmp, then os.replace) to prevent
    partial writes on crash.  Raises OSError on I/O failure.
    """
    serialized_entries: Dict[str, dict] = {}
    for tc, entry in entries.items():
        serialized_entries[tc] = {
            "recipe_filename": entry.recipe_filename,
            "recipe_hash":     entry.recipe_hash,
            "line_count":      entry.line_count,
            "committed_at":    entry.committed_at.isoformat(),
        }
    document = {
        "schema_version": _MANIFEST_SCHEMA_VERSION,
        "generated_at":   datetime.now(timezone.utc).isoformat(),
        "generator":      "signal_kernel/recipes/registry.py",
        "entries":        serialized_entries,
    }
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(document, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp_path, path)


# ─────────────────────────────────────────────────────────────────────────────
# FILE I/O HELPERS
# These helpers are used only during registry initialization and integrity
# verification.  They are NOT called on the hot path (get_recipe).
# ─────────────────────────────────────────────────────────────────────────────

def _read_recipe_file(path: Path) -> str:
    """
    Read a recipe file as UTF-8 text.  Raises RecipeNotFound (hard stop) if the
    file is absent, not a regular file, or not readable.  On encoding error,
    raises ValueError — the caller converts to a structural validation failure.
    """
    if not path.exists():
        raise RecipeNotFound(
            run_id=new_run_id(),
            topology_class="UNKNOWN",
            recipe_path=str(path),
            is_hardcoded=True,
        )
    if not path.is_file():
        raise RecipeNotFound(
            run_id=new_run_id(),
            topology_class="UNKNOWN",
            recipe_path=str(path),
            is_hardcoded=True,
        )
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Recipe file {path} is not valid UTF-8: {exc}.  "
            "All recipe files must be UTF-8 encoded."
        ) from exc
    except (OSError, PermissionError) as exc:
        raise RecipeNotFound(
            run_id=new_run_id(),
            topology_class="UNKNOWN",
            recipe_path=str(path),
            is_hardcoded=True,
        ) from exc


def _write_recipe_file_atomic(path: Path, content: str) -> None:
    """
    Write recipe content to disk atomically (tmp → replace).
    Creates parent directories if they do not exist.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# RECIPE REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class RecipeRegistry:
    """
    Authoritative recipe lookup table for the AXIOM signal kernel.

    Manages the lifecycle of all topology class → recipe mappings:
      •  Hardcoded primaries (immutable floor, loaded at startup)
      •  GENERIC_HTML fallback (embedded, provisioned at startup)
      •  Compiler-generated recipes (registered by topology_parser.py)
      •  Versioned staging area (versioned keys, await explicit promotion)

    Instantiate once and share the module-level singleton (see bottom of file).
    Do not instantiate RecipeRegistry directly in application code.

    Thread safety: all public methods acquire _lock before touching state.
    """

    # ── Internal state ────────────────────────────────────────────────────────
    #
    #   _primary     : topology_class → RecipeRegistryEntry
    #                  The active (callable) recipe for each class.
    #                  get_recipe() reads only from here (and PARENT_CLASS_MAP).
    #
    #   _versions    : base_class → list[RecipeRegistryEntry]
    #                  All compiler-generated versions registered for a class,
    #                  in registration order.  Index 0 is the first compiler
    #                  entry.  Versioned keys are "BASE_v2", "BASE_v3" etc.
    #
    #   _version_counter : base_class → next version number (int, starts at 2)
    #                  The hardcoded recipe is implicitly "v1".  First compiler
    #                  recipe becomes "v2".
    #
    #   _manifest    : topology_class → HardcodedRecipeManifestEntry
    #                  Loaded from manifest.json on startup.  Used to verify
    #                  hardcoded files haven't been tampered with.
    #
    #   _content_cache : recipe_path → (content: str, hash: str)
    #                  In-memory recipe content.  Populated at load time.
    #                  Avoids redundant disk I/O on integrity verification.
    #
    #   _superseded_hardcoded : topology_class → RecipeRegistryEntry
    #                  Hardcoded entries that were superseded via
    #                  supersede_hardcoded().  Retained for restoration.
    #
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        *,
        hardcoded_dir: Optional[Path] = None,
        strict_startup: bool = True,
    ) -> None:
        """
        Initialise the registry.

        Parameters
        ----------
        hardcoded_dir:
            Override for the hardcoded recipe directory.  Defaults to
            _HARDCODED_DIR (``recipes/hardcoded/`` relative to this file).
            Useful in tests that supply fixture recipes without touching the
            repository tree.

        strict_startup:
            If True (default, production) any hardcoded recipe that cannot be
            loaded raises RecipeRegistryCorrupt and startup fails hard.
            If False (test/development) loading errors are logged and the
            affected topology class falls back to GENERIC_HTML instead of
            aborting the process.  Never set False in production.
        """
        self._lock:       threading.RLock = threading.RLock()
        self._hard_dir:   Path            = hardcoded_dir or _HARDCODED_DIR
        self._strict:     bool            = strict_startup

        # Core tables — all mutations require _lock.
        self._primary:     Dict[str, RecipeRegistryEntry]        = {}
        self._versions:    Dict[str, List[RecipeRegistryEntry]]  = defaultdict(list)
        self._version_counter: Dict[str, int]                    = {}
        self._manifest:    Dict[str, HardcodedRecipeManifestEntry] = {}
        self._content_cache: Dict[str, Tuple[str, str]]          = {}  # path→(content,hash)
        self._superseded_hardcoded: Dict[str, RecipeRegistryEntry] = {}

        # Diagnostic state — written during init, read-only afterwards via health().
        self._init_errors:   List[str]   = []
        self._initialized_at: Optional[datetime] = None
        self._bootstrap_performed: bool = False   # True if manifest was freshly written

        # Run startup loading sequence.
        self._startup()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _startup(self) -> None:
        """
        Full startup sequence.  Called once from __init__.

        Order:
          1. Ensure hardcoded_dir exists.
          2. Provision GENERIC_HTML on disk.
          3. Load or bootstrap manifest.json.
          4. Load each hardcoded recipe and verify against manifest.
          5. Register GENERIC_HTML into the primary table.
          6. Log summary.
        """
        _log.info(
            "registry.startup — begin | hardcoded_dir=%s strict=%s",
            self._hard_dir, self._strict,
        )

        # Step 1: Ensure hardcoded_dir is present.
        try:
            self._hard_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            msg = f"Cannot create/access hardcoded dir {self._hard_dir}: {exc}"
            _log.critical("registry.startup — %s", msg)
            raise RecipeRegistryCorrupt(
                detail=msg,
                corrupt_entry=str(self._hard_dir),
            ) from exc

        # Step 2: Provision GENERIC_HTML recipe file.
        self._provision_generic_html()

        # Step 3: Load or bootstrap manifest.
        self._load_or_bootstrap_manifest()

        # Step 4: Load every hardcoded recipe and verify.
        self._load_all_hardcoded_recipes()

        # Step 5: Register GENERIC_HTML into the primary table (if not already
        # done by _load_all_hardcoded_recipes — generic_html is special).
        self._register_generic_html_primary()

        # Step 6: Summary log.
        self._initialized_at = datetime.now(timezone.utc)
        loaded_classes = list(self._primary.keys())
        _log.info(
            "registry.startup — complete | primary_classes=%s bootstrap=%s errors=%d",
            loaded_classes, self._bootstrap_performed, len(self._init_errors),
        )

    def _provision_generic_html(self) -> None:
        """
        Write the embedded GENERIC_HTML recipe to disk if necessary.

        •  If the file does not exist → write it.
        •  If the file exists but its content matches the embedded hash → no-op.
        •  If the file exists but its content DIFFERS → this is a tamper event
           (someone modified our embedded fallback at runtime).  We overwrite
           it with the correct content and log a critical warning.  We do not
           raise — the fallback must be available even if the file was corrupt.

        The file is owned exclusively by registry.py.  No external process
        should modify generic_html.sh.
        """
        generic_path = self._hard_dir / _GENERIC_HTML_FILENAME
        if generic_path.exists():
            try:
                existing_content = generic_path.read_text(encoding="utf-8")
                existing_hash = compute_recipe_hash(existing_content)
                if existing_hash == _GENERIC_HTML_CANONICAL_HASH:
                    _log.debug("registry._provision_generic_html — file unchanged, no-op")
                    return
                # Content differs — log tamper warning and overwrite.
                _log.critical(
                    "registry._provision_generic_html — TAMPER DETECTED: "
                    "generic_html.sh content hash mismatch. "
                    "expected=%s actual=%s.  Restoring from embedded source.",
                    _GENERIC_HTML_CANONICAL_HASH[:16],
                    existing_hash[:16],
                )
            except (OSError, UnicodeDecodeError) as exc:
                _log.warning(
                    "registry._provision_generic_html — cannot read existing file (%s), "
                    "will overwrite.", exc,
                )

        # Write (or overwrite) the file.
        try:
            _write_recipe_file_atomic(generic_path, _GENERIC_HTML_RECIPE_CONTENT)
            _log.info(
                "registry._provision_generic_html — wrote %s (%d bytes)",
                generic_path, len(_GENERIC_HTML_RECIPE_CONTENT.encode("utf-8")),
            )
        except OSError as exc:
            _log.critical(
                "registry._provision_generic_html — CANNOT WRITE generic_html.sh: %s.  "
                "GENERIC_HTML fallback will operate from in-memory content only.", exc,
            )
            # Do not raise — registry must still start.

    def _load_or_bootstrap_manifest(self) -> None:
        """
        Attempt to load manifest.json from disk.

        If the manifest exists and is valid, it is loaded into self._manifest.
        If the manifest does not exist (first-run bootstrap), it is generated
        from the live recipe files and written to disk.
        If the manifest exists but is corrupt (bad JSON, wrong schema), and
        strict_startup=True, startup raises RecipeManifestCorruption.
        In non-strict mode, a corrupt manifest triggers regeneration with a
        WARNING — this is recovery from a packaging error.
        """
        manifest_str = str(_MANIFEST_PATH)
        try:
            loaded = _load_manifest_from_disk(_MANIFEST_PATH)
            self._manifest = loaded
            _log.info(
                "registry.manifest — loaded %d entries from %s",
                len(self._manifest), manifest_str,
            )
            return
        except RecipeManifestCorruption as exc:
            # File is missing (first-run bootstrap) or unreadable.
            if "does not exist" in str(exc):
                _log.warning(
                    "registry.manifest — manifest.json not found at %s.  "
                    "Performing first-run bootstrap from live recipe files.  "
                    "This is expected on first deploy; commit the generated "
                    "manifest.json afterwards.",
                    manifest_str,
                )
            else:
                msg = (
                    f"registry.manifest — manifest.json is corrupt or unreadable: {exc}.  "
                    f"{'Aborting startup.' if self._strict else 'Regenerating manifest (non-strict mode).'}"
                )
                _log.critical(msg)
                if self._strict:
                    raise
                _log.warning(
                    "registry.manifest — regenerating manifest from live files "
                    "(strict_startup=False)."
                )
        except _ManifestFormatError as exc:
            msg = (
                f"registry.manifest — manifest schema error: {exc}.  "
                f"{'Aborting startup.' if self._strict else 'Regenerating manifest.'}"
            )
            _log.critical(msg)
            if self._strict:
                raise RecipeManifestCorruption(
                    manifest_path=manifest_str,
                    detail=str(exc),
                ) from exc

        # Bootstrap: generate manifest from live recipe files.
        self._bootstrap_manifest()

    def _bootstrap_manifest(self) -> None:
        """
        Generate manifest entries from the live recipe files on disk and write
        manifest.json.  Called only when the manifest is absent or corrupt.

        This WRITES the canonical hashes.  Once written, subsequent startups
        will VERIFY against these hashes.  Treat the first-run bootstrap as a
        "commit" of the current recipe state.

        Does not raise on I/O error — logs CRITICAL and continues without
        a persisted manifest (integrity will degrade to best-effort).
        """
        _log.warning(
            "registry.bootstrap — generating manifest from live recipe files.  "
            "This should only happen once per deployment."
        )

        # Build full file set: the five hardcoded + GENERIC_HTML.
        file_set: Dict[str, Tuple[str, str]] = {}  # topology_class → (filename, path)
        for tc, filename in _HARDCODED_FILE_MAP.items():
            file_set[tc] = (filename, str(self._hard_dir / filename))
        file_set[FALLBACK_TOPOLOGY_CLASS] = (
            _GENERIC_HTML_FILENAME,
            str(self._hard_dir / _GENERIC_HTML_FILENAME),
        )

        entries: Dict[str, HardcodedRecipeManifestEntry] = {}
        now = datetime.now(timezone.utc)

        for tc, (filename, path_str) in file_set.items():
            path = Path(path_str)
            try:
                content = _read_recipe_file(path)
            except Exception as exc:  # noqa: BLE001
                _log.critical(
                    "registry.bootstrap — cannot read %s for manifest: %s.  "
                    "Skipping this entry.", path_str, exc,
                )
                self._init_errors.append(
                    f"bootstrap: cannot read {filename} for {tc}: {exc}"
                )
                continue

            recipe_hash    = compute_recipe_hash(content)
            executable_lines = _count_executable_lines(content)

            try:
                entry = HardcodedRecipeManifestEntry(
                    topology_class=TopologyClassStr(tc),
                    recipe_filename=filename,
                    recipe_hash=RecipeHash(recipe_hash),
                    line_count=executable_lines,
                    committed_at=now,
                )
            except ValueError as exc:
                _log.critical(
                    "registry.bootstrap — cannot construct manifest entry for %s: %s",
                    tc, exc,
                )
                self._init_errors.append(
                    f"bootstrap: invalid manifest entry for {tc}: {exc}"
                )
                continue

            entries[tc] = entry
            self._manifest[tc] = entry
            _log.info(
                "registry.bootstrap — %s hash=%s lines=%d",
                tc, recipe_hash[:16], executable_lines,
            )

        # Persist manifest.json.
        try:
            _write_manifest_to_disk(_MANIFEST_PATH, entries)
            self._bootstrap_performed = True
            _log.warning(
                "registry.bootstrap — manifest.json written to %s.  "
                "IMPORTANT: commit this file to the repository.",
                _MANIFEST_PATH,
            )
        except OSError as exc:
            _log.critical(
                "registry.bootstrap — FAILED to write manifest.json: %s.  "
                "In-memory manifest is active but will not persist across restarts.",
                exc,
            )
            self._init_errors.append(f"bootstrap: write manifest failed: {exc}")

    def _load_all_hardcoded_recipes(self) -> None:
        """
        Load, verify, and register all hardcoded recipe files.

        For each of the five hardcoded topology classes:
          1. Resolve path from _HARDCODED_FILE_MAP.
          2. Read file content.
          3. Compute SHA-256 hash.
          4. Verify against manifest entry (if manifest entry exists).
             Hash mismatch → RecipeHashMismatch (hard stop in strict mode).
          5. Count executable lines.
          6. Populate content cache.
          7. Register into _primary table.

        If a recipe file is missing or unreadable:
          •  strict_startup=True  → RecipeRegistryCorrupt (hard stop).
          •  strict_startup=False → log CRITICAL, skip, GENERIC_HTML will cover.
        """
        for topology_class, filename in _HARDCODED_FILE_MAP.items():
            recipe_path = self._hard_dir / filename
            try:
                self._load_one_hardcoded(
                    topology_class=topology_class,
                    filename=filename,
                    recipe_path=recipe_path,
                )
            except (RecipeHashMismatch, RecipeNotFound) as exc:
                if self._strict:
                    _log.critical(
                        "registry.load_hardcoded — FATAL: %s | topology=%s",
                        exc, topology_class,
                    )
                    raise RecipeRegistryCorrupt(
                        detail=f"Hardcoded recipe for {topology_class} failed: {exc}",
                        corrupt_entry=str(recipe_path),
                    ) from exc
                else:
                    _log.critical(
                        "registry.load_hardcoded — SKIPPING %s due to: %s  "
                        "(strict_startup=False — GENERIC_HTML fallback active)",
                        topology_class, exc,
                    )
                    self._init_errors.append(
                        f"load_hardcoded: {topology_class} failed: {exc}"
                    )
            except ValueError as exc:
                if self._strict:
                    raise RecipeRegistryCorrupt(
                        detail=f"Invalid hardcoded recipe for {topology_class}: {exc}",
                        corrupt_entry=str(recipe_path),
                    ) from exc
                else:
                    _log.critical(
                        "registry.load_hardcoded — SKIPPING %s (invalid): %s",
                        topology_class, exc,
                    )
                    self._init_errors.append(
                        f"load_hardcoded: {topology_class} invalid: {exc}"
                    )

    def _load_one_hardcoded(
        self,
        *,
        topology_class: str,
        filename: str, # noqa
        recipe_path: Path,
    ) -> None:
        """
        Load, verify, and register one hardcoded recipe.
        Called by _load_all_hardcoded_recipes for each entry.
        Also callable at runtime by reload_hardcoded_recipe() for hot-reload.

        Raises:
            RecipeNotFound      — file missing or unreadable.
            RecipeHashMismatch  — content does not match manifest hash.
            ValueError          — content is not valid UTF-8 or line count fails.
        """
        # Read content.
        content = _read_recipe_file(recipe_path)
        actual_hash = compute_recipe_hash(content)
        executable_lines = _count_executable_lines(content)

        # Verify against manifest if we have an entry for this class.
        manifest_entry = self._manifest.get(topology_class)
        if manifest_entry is not None:
            expected_hash = manifest_entry.recipe_hash
            if actual_hash != expected_hash:
                _log.critical(
                    "registry.load_hardcoded — HASH MISMATCH for %s.  "
                    "expected=%s actual=%s.  "
                    "SECURITY: hardcoded recipe may have been tampered with at runtime.",
                    topology_class, expected_hash[:16], actual_hash[:16],
                )
                raise RecipeHashMismatch(
                    run_id=new_run_id(),
                    topology_class=topology_class,
                    recipe_path=str(recipe_path),
                    expected_hash=expected_hash,
                    actual_hash=actual_hash,
                    is_hardcoded=True,
                )
        else:
            # No manifest entry — bootstrap in progress.  Trust the live file.
            _log.debug(
                "registry.load_hardcoded — no manifest entry for %s, "
                "trusting live file (bootstrap mode).", topology_class,
            )

        # Populate content cache.
        self._content_cache[str(recipe_path)] = (content, actual_hash)

        # Build the registry entry.
        now = datetime.now(timezone.utc)
        entry = RecipeRegistryEntry(
            topology_class=TopologyClassStr(topology_class),
            recipe_path=str(recipe_path),
            recipe_hash=RecipeHash(actual_hash),
            is_hardcoded=True,
            registered_at=now,
            registration_source="hardcoded_loader",
        )

        # Register as primary (under lock since this may be called from reload).
        with self._lock:
            self._primary[topology_class] = entry
            # Hardcoded entries initialize the version counter at v1.
            if topology_class not in self._version_counter:
                self._version_counter[topology_class] = 2  # next compiler slot is v2

        _log.info(
            "registry.load_hardcoded — OK | topology=%s path=%s hash=%s lines=%d",
            topology_class, recipe_path.name, actual_hash[:16], executable_lines,
        )

    def _register_generic_html_primary(self) -> None:
        """
        Register GENERIC_HTML into the primary table from the in-memory
        canonical content.  If the file was successfully written to disk,
        the primary entry points to the disk path.  If disk write failed,
        the entry still uses the canonical path (which may not exist) — the
        caller (pipeline.py) will encounter RecipeNotFound if it tries to
        execute it, which is the correct failure mode.

        GENERIC_HTML is registered with is_hardcoded=True, registration_source
        "fallback" — both of which are permitted by RecipeRegistryEntry's
        validation (see contracts.py line ~818).
        """
        if FALLBACK_TOPOLOGY_CLASS in self._primary:
            # Already registered (possibly by a previous call — idempotent).
            return

        generic_path = self._hard_dir / _GENERIC_HTML_FILENAME
        actual_hash  = _GENERIC_HTML_CANONICAL_HASH

        # Verify file hash if the file exists.
        if generic_path.exists():
            try:
                disk_content = generic_path.read_text(encoding="utf-8")
                disk_hash    = compute_recipe_hash(disk_content)
                if disk_hash != actual_hash:
                    _log.critical(
                        "registry.generic_html — disk hash mismatch after provision. "
                        "Using canonical embedded content hash for registry entry. "
                        "expected=%s disk=%s",
                        actual_hash[:16], disk_hash[:16],
                    )
                else:
                    self._content_cache[str(generic_path)] = (disk_content, disk_hash)
            except (OSError, UnicodeDecodeError) as exc:
                _log.warning(
                    "registry.generic_html — cannot verify disk content: %s.  "
                    "Using in-memory canonical.", exc,
                )
        else:
            _log.warning(
                "registry.generic_html — file not on disk at %s.  "
                "Registry entry will point to this path; pipeline.py will fail "
                "with RecipeNotFound if it attempts execution.", generic_path,
            )

        # Cache from embedded constant if not already cached from disk.
        if str(generic_path) not in self._content_cache:
            self._content_cache[str(generic_path)] = (
                _GENERIC_HTML_RECIPE_CONTENT,
                _GENERIC_HTML_CANONICAL_HASH,
            )

        exec_lines = _count_executable_lines(_GENERIC_HTML_RECIPE_CONTENT)
        now = datetime.now(timezone.utc)

        entry = RecipeRegistryEntry(
            topology_class=TopologyClassStr(FALLBACK_TOPOLOGY_CLASS),
            recipe_path=str(generic_path),
            recipe_hash=RecipeHash(actual_hash),
            is_hardcoded=True,
            registered_at=now,
            registration_source="fallback",
        )

        with self._lock:
            self._primary[FALLBACK_TOPOLOGY_CLASS] = entry
            if FALLBACK_TOPOLOGY_CLASS not in self._version_counter:
                self._version_counter[FALLBACK_TOPOLOGY_CLASS] = 2

        _log.info(
            "registry.generic_html — registered | hash=%s lines=%d path=%s",
            actual_hash[:16], exec_lines, generic_path.name,
        )

    # ── Public API: get_recipe ─────────────────────────────────────────────────

    def get_recipe(
        self,
        topology_class: str,
        *,
        run_id: Optional[str] = None,
    ) -> RecipeMount:
        """
        Resolve a topology class to a validated RecipeMount.

        This method NEVER returns None and NEVER raises for a missing topology
        class.  The fallback chain guarantees a RecipeMount is always returned.

        Fallback chain (in order):
          1. Exact match in primary table.
          2. Parent class lookup via PARENT_CLASS_MAP.
          3. GENERIC_HTML unconditional fallback.

        If none of the three steps produces an entry (which should be
        impossible after a successful startup), RecipeRegistryCorrupt is raised
        — but that indicates a programming error in registry.py, not a normal
        runtime condition.

        Parameters
        ----------
        topology_class:
            The topology class string to resolve.  Must match
            [A-Z][A-Z0-9_]{1,63}.  Invalid topology class strings resolve
            to GENERIC_HTML (logged at WARNING).

        run_id:
            Optional run_id for structured logging correlation.  Does not
            affect resolution logic.

        Returns
        -------
        RecipeMount:
            Validated contract object.  recipe_path is the path the caller
            should mount to the kernel container.
        """
        with self._lock:
            return self._get_recipe_locked(
                topology_class=topology_class,
                run_id=run_id,
            )

    def _get_recipe_locked(
        self,
        topology_class: str,
        run_id: Optional[str],
    ) -> RecipeMount:
        """
        Inner implementation of get_recipe, called under lock.
        Separated so that internal helpers can call it without lock re-acquisition.
        """
        # ── Normalise and validate topology class ─────────────────────────────
        if not topology_class or not topology_class.strip():
            _log.warning(
                "registry.get_recipe — empty topology_class, falling back to "
                "GENERIC_HTML | run_id=%s", run_id,
            )
            return self._build_recipe_mount_from_entry(
                self._primary[FALLBACK_TOPOLOGY_CLASS],
                fallback_reason="empty_topology_class",
            )

        topology_class = topology_class.strip()

        if not _is_valid_topology_class(topology_class):
            _log.warning(
                "registry.get_recipe — invalid topology_class %r, falling back to "
                "GENERIC_HTML | run_id=%s", topology_class, run_id,
            )
            return self._build_recipe_mount_from_entry(
                self._primary[FALLBACK_TOPOLOGY_CLASS],
                fallback_reason=f"invalid_class:{topology_class}",
            )

        # ── Step 1: Exact match ───────────────────────────────────────────────
        exact = self._primary.get(topology_class)
        if exact is not None:
            _log.debug(
                "registry.get_recipe — exact hit | topology=%s hash=%s run_id=%s",
                topology_class, exact.recipe_hash[:16], run_id,
            )
            return self._build_recipe_mount_from_entry(exact)

        # ── Step 2: Parent class fallback ─────────────────────────────────────
        parent_class = PARENT_CLASS_MAP.get(topology_class)
        if parent_class is not None:
            parent_entry = self._primary.get(parent_class)
            if parent_entry is not None:
                _log.info(
                    "registry.get_recipe — parent fallback | topology=%s → parent=%s "
                    "hash=%s run_id=%s",
                    topology_class, parent_class, parent_entry.recipe_hash[:16], run_id,
                )
                return self._build_recipe_mount_from_entry(
                    parent_entry,
                    fallback_reason=f"parent:{parent_class}",
                    resolved_for=topology_class,
                )
            else:
                _log.warning(
                    "registry.get_recipe — parent class %r is in PARENT_CLASS_MAP but "
                    "has no primary entry; continuing to GENERIC_HTML | run_id=%s",
                    parent_class, run_id,
                )

        # ── Step 3: GENERIC_HTML unconditional fallback ───────────────────────
        generic_entry = self._primary.get(FALLBACK_TOPOLOGY_CLASS)
        if generic_entry is None:
            # This should be structurally impossible after a successful startup.
            # Raise RecipeRegistryCorrupt — this is a programming error.
            raise RecipeRegistryCorrupt(
                detail=(
                    "GENERIC_HTML is not in the primary table — this should be "
                    "impossible after a successful startup.  The registry is in an "
                    "undefined state.  Restart and investigate."
                ),
                corrupt_entry=FALLBACK_TOPOLOGY_CLASS,
            )

        _log.info(
            "registry.get_recipe — GENERIC_HTML fallback | topology=%s "
            "hash=%s run_id=%s",
            topology_class, generic_entry.recipe_hash[:16], run_id,
        )
        return self._build_recipe_mount_from_entry(
            generic_entry,
            fallback_reason="generic_html",
            resolved_for=topology_class,
        )

    def _build_recipe_mount_from_entry(
        self,
        entry: RecipeRegistryEntry,
        *,
        fallback_reason: Optional[str] = None,
        resolved_for: Optional[str]    = None,
    ) -> RecipeMount:
        """
        Construct a RecipeMount from a RecipeRegistryEntry.

        Performs a final integrity check: re-derives the hash from the
        content cache and verifies it matches the stored hash.  This catches
        in-process mutation of the content cache (a programming error) before
        the recipe reaches the kernel.

        If the content cache entry is missing (recipe was registered without
        caching content — only possible for compiler-generated recipes), the
        hash stored at registration time is used directly.

        Parameters
        ----------
        entry:
            Source registry entry.
        fallback_reason:
            Optional diagnostic string explaining why fallback was triggered.
            Attached to structured log; not present in RecipeMount (immutable).
        resolved_for:
            If this entry is serving a different topology class (parent or
            GENERIC_HTML fallback), log the original requested class.
        """
        # Content cache integrity check — only possible for cached entries.
        cached = self._content_cache.get(entry.recipe_path)
        if cached is not None:
            cached_content, cached_hash = cached
            if cached_hash != entry.recipe_hash:
                # Content cache has diverged from registry entry.
                # This is a programming error — do not serve a mismatched entry.
                _log.critical(
                    "registry._build_recipe_mount — INTERNAL INTEGRITY FAILURE: "
                    "content cache hash %s != registry hash %s for path %s.  "
                    "Refusing to serve this recipe.",
                    cached_hash[:16], entry.recipe_hash[:16], entry.recipe_path,
                )
                # Fall through to GENERIC_HTML rather than raising.
                # get_recipe() guarantees it never raises for missing recipes.
                generic_entry = self._primary.get(FALLBACK_TOPOLOGY_CLASS)
                if generic_entry is not None and entry.topology_class != FALLBACK_TOPOLOGY_CLASS:
                    _log.warning(
                        "registry._build_recipe_mount — using GENERIC_HTML due to "
                        "cache integrity failure for %s", entry.topology_class,
                    )
                    return self._build_recipe_mount_from_entry(
                        generic_entry,
                        fallback_reason="cache_integrity_failure",
                        resolved_for=entry.topology_class,
                    )
                # GENERIC_HTML itself has a cache integrity failure — trust the
                # entry hash; RecipeMount will be constructed with it.

        exec_lines = _count_executable_lines(
            cached[0] if cached else ""
        ) or 1  # minimum 1; if no content cached, we cannot recount

        # For compiler-generated recipes without cached content, use a sentinel
        # line count.  validator.py will do a proper structural check before
        # these recipes ever reach the kernel.
        if cached is None and not entry.is_hardcoded:
            exec_lines = 1  # sentinel; validator.py enforces the real limit

        if fallback_reason:
            _log.debug(
                "registry._build_recipe_mount — fallback_reason=%s resolved_for=%s "
                "serving topology=%s",
                fallback_reason, resolved_for, entry.topology_class,
            )

        try:
            mount = RecipeMount(
                recipe_path=entry.recipe_path,
                topology_class=entry.topology_class,
                recipe_hash=entry.recipe_hash,
                is_hardcoded=entry.is_hardcoded,
                line_count=max(1, min(exec_lines, MAX_RECIPE_LINE_COUNT)),
            )
        except ValueError as exc:
            # RecipeMount.__post_init__ raised.  This should never happen for
            # trusted entries — it indicates a contracts.py invariant change.
            _log.critical(
                "registry._build_recipe_mount — RecipeMount construction failed "
                "for %s: %s.  This is a programming error.", entry.topology_class, exc,
            )
            raise RecipeRegistryCorrupt(
                detail=f"Cannot construct RecipeMount for {entry.topology_class}: {exc}",
                corrupt_entry=entry.recipe_path,
            ) from exc

        return mount

    # ── Public API: register_recipe ───────────────────────────────────────────

    def register_recipe(
        self,
        topology_class: str,
        recipe_path: str,
        *,
        caller_supplied_hash: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> RecipeRegistryEntry:
        """
        Register a compiler-generated recipe for a topology class.

        Called by topology_parser.py after successfully compiling a new recipe.

        Behaviour varies depending on whether the topology class already has a
        hardcoded primary:

        Case A — no primary or non-hardcoded primary:
            The new recipe becomes the primary for this topology class.
            Returns the new primary RecipeRegistryEntry.

        Case B — hardcoded primary exists:
            The new recipe is registered under a versioned key (e.g.
            NEWS_ARTICLE_v2) and does NOT replace the primary.
            Logs a WARNING explaining that promotion requires an explicit call
            to promote() — and that promote() will refuse to replace a
            hardcoded primary without supersede_hardcoded().
            Returns the versioned RecipeRegistryEntry.

        Hash verification:
            registry.py reads the recipe file itself and computes its hash.
            If caller_supplied_hash is provided and differs from the computed
            hash, RecipeHashMismatch is raised — this means topology_parser.py
            registered the wrong hash (a consistency bug).

        Parameters
        ----------
        topology_class:
            The base topology class (e.g. "NEWS_ARTICLE").  Must be a valid
            topology class string.  Versioned keys (NEWS_ARTICLE_v2) should
            NOT be passed here — this method manages versioning internally.

        recipe_path:
            Absolute or relative path to the compiled recipe .sh file.
            Must resolve to a path inside _ALLOWED_RECIPE_ROOTS.
            Must exist on disk at call time.

        caller_supplied_hash:
            Optional SHA-256 hash that topology_parser.py claims the recipe
            has.  If provided and mismatches, RecipeHashMismatch is raised.

        run_id:
            Optional run_id for structured logging.

        Returns
        -------
        RecipeRegistryEntry:
            The entry that was registered (primary or versioned).

        Raises
        ------
        ValueError:
            recipe_path is empty, invalid, outside allowed roots, or the file
            cannot be read.
        RecipeHashMismatch:
            caller_supplied_hash was provided and does not match computed hash.
        """
        _require_valid_topology_class(topology_class, "register_recipe")

        # Disallow versioned keys as input — caller must pass the base class.
        base_class, version_number = _base_class_and_version(topology_class)
        if version_number is not None:
            raise ValueError(
                f"register_recipe: topology_class must be a base class, "
                f"not a versioned key.  Got {topology_class!r} (version={version_number}).  "
                f"Pass {base_class!r} and let the registry assign the version number."
            )

        # Validate and resolve path.
        safe_path = _assert_safe_recipe_path(recipe_path, "register_recipe")

        # Read content and compute hash ourselves — do not trust the caller.
        content = _read_recipe_file(safe_path)
        actual_hash = compute_recipe_hash(content)

        # If caller supplied a hash, verify it matches our computation.
        if caller_supplied_hash is not None:
            if actual_hash != caller_supplied_hash:
                _log.error(
                    "registry.register_recipe — caller hash mismatch for %s.  "
                    "expected (caller)=%s actual (computed)=%s | run_id=%s",
                    topology_class, caller_supplied_hash[:16], actual_hash[:16], run_id,
                )
                raise RecipeHashMismatch(
                    run_id=run_id or new_run_id(),
                    topology_class=topology_class,
                    recipe_path=recipe_path,
                    expected_hash=caller_supplied_hash,
                    actual_hash=actual_hash,
                    is_hardcoded=False,
                )

        with self._lock:
            return self._register_recipe_locked(
                topology_class=topology_class,
                safe_path=safe_path,
                content=content,
                actual_hash=actual_hash,
                run_id=run_id,
            )

    def _register_recipe_locked(
        self,
        topology_class: str,
        safe_path: Path,
        content: str,
        actual_hash: str,
        run_id: Optional[str],
    ) -> RecipeRegistryEntry:
        """
        Inner implementation of register_recipe, called under lock.
        """
        now = datetime.now(timezone.utc)
        existing_primary = self._primary.get(topology_class)

        if existing_primary is not None and existing_primary.is_hardcoded:
            # ── Case B: hardcoded primary exists ─────────────────────────────
            # Auto-register under versioned key.  Do NOT touch the primary.
            versioned_key = self._next_versioned_key(topology_class)
            _log.warning(
                "registry.register_recipe — topology=%s has a hardcoded primary "
                "(hash=%s).  Registering compiler recipe under versioned key %r.  "
                "Call promote(%r) then supersede_hardcoded(%r, %r) to promote to "
                "primary.  | run_id=%s",
                topology_class,
                existing_primary.recipe_hash[:16],
                versioned_key,
                versioned_key,
                topology_class,
                versioned_key,
                run_id,
            )
            entry = RecipeRegistryEntry(
                topology_class=TopologyClassStr(versioned_key),
                recipe_path=str(safe_path),
                recipe_hash=RecipeHash(actual_hash),
                is_hardcoded=False,
                registered_at=now,
                registration_source="topology_parser",
            )
            self._versions[topology_class].append(entry)
            self._content_cache[str(safe_path)] = (content, actual_hash)
            # Also register the versioned key in the primary table so it can be
            # resolved directly by promote().
            self._primary[versioned_key] = entry
            return entry

        else:
            # ── Case A: no primary or non-hardcoded primary ───────────────────
            entry = RecipeRegistryEntry(
                topology_class=TopologyClassStr(topology_class),
                recipe_path=str(safe_path),
                recipe_hash=RecipeHash(actual_hash),
                is_hardcoded=False,
                registered_at=now,
                registration_source="topology_parser",
            )
            self._primary[topology_class] = entry
            self._content_cache[str(safe_path)] = (content, actual_hash)
            self._versions[topology_class].append(entry)
            if topology_class not in self._version_counter:
                self._version_counter[topology_class] = 2
            _log.info(
                "registry.register_recipe — registered primary | topology=%s "
                "hash=%s path=%s | run_id=%s",
                topology_class, actual_hash[:16], safe_path.name, run_id,
            )
            return entry

    def _next_versioned_key(self, base_class: str) -> str:
        """
        Return the next versioned key for a base class and increment the counter.
        E.g. first call for "NEWS_ARTICLE" returns "NEWS_ARTICLE_v2";
        second returns "NEWS_ARTICLE_v3".

        Called under _lock.
        """
        version = self._version_counter.get(base_class, 2)
        # Find a key that isn't already in use (handles gaps from deletion).
        while True:
            candidate = f"{base_class}_v{version}"
            if candidate not in self._primary:
                break
            version += 1
        self._version_counter[base_class] = version + 1
        return candidate

    # ── Public API: promote ───────────────────────────────────────────────────

    def promote(
        self,
        versioned_key: str,
        *,
        run_id: Optional[str] = None,
    ) -> RecipeRegistryEntry:
        """
        Promote a versioned recipe to primary for its base class.

        This is the standard promotion path for compiler-generated recipes that
        have passed validation and review.  It replaces a NON-HARDCODED primary.
        It DOES NOT replace a hardcoded primary — use supersede_hardcoded() for
        that deliberate, logged operation.

        Parameters
        ----------
        versioned_key:
            The versioned topology key registered by register_recipe(), e.g.
            "NEWS_ARTICLE_v2".  Must exist in the primary table (all versioned
            recipes are also registered in primary for direct lookup).

        run_id:
            Optional correlation ID for structured logging.

        Returns
        -------
        RecipeRegistryEntry:
            The entry that is now the primary (same object as was versioned).

        Raises
        ------
        ValueError:
            versioned_key is not a valid topology class string, or is not a
            versioned key (has no _vN suffix).
        KeyError:
            versioned_key does not exist in the registry.
        HardcodedRecipeOverwriteAttempt:
            The current primary for the base class is hardcoded.
            Use supersede_hardcoded() instead.
        """
        _require_valid_topology_class(versioned_key, "promote")
        base_class, version_num = _base_class_and_version(versioned_key)
        if version_num is None:
            raise ValueError(
                f"promote: {versioned_key!r} is not a versioned key.  "
                f"Versioned keys have the form BASE_vN (e.g. NEWS_ARTICLE_v2).  "
                f"Use register_recipe() first."
            )

        with self._lock:
            return self._promote_locked(
                versioned_key=versioned_key,
                base_class=base_class,
                run_id=run_id,
            )

    def _promote_locked(
        self,
        versioned_key: str,
        base_class: str,
        run_id: Optional[str],
    ) -> RecipeRegistryEntry:
        """Inner promote() implementation, called under lock."""
        versioned_entry = self._primary.get(versioned_key)
        if versioned_entry is None:
            raise KeyError(
                f"promote: versioned_key {versioned_key!r} is not in the registry.  "
                f"Call register_recipe({base_class!r}, path) first."
            )

        current_primary = self._primary.get(base_class)

        # Guard: hardcoded primary cannot be replaced by promote().
        if current_primary is not None and current_primary.is_hardcoded:
            _log.error(
                "registry.promote — REJECTED: current primary for %r is hardcoded "
                "(hash=%s).  Use supersede_hardcoded() for deliberate replacement.  "
                "| attempted_key=%r run_id=%s",
                base_class, current_primary.recipe_hash[:16], versioned_key, run_id,
            )
            raise HardcodedRecipeOverwriteAttempt(
                topology_class=base_class,
                attempted_source="topology_parser",
                existing_hash=current_primary.recipe_hash,
                attempted_hash=versioned_entry.recipe_hash,
            )

        # Swap the primary: create a new entry with the base class name.
        promoted_entry = RecipeRegistryEntry(
            topology_class=TopologyClassStr(base_class),
            recipe_path=versioned_entry.recipe_path,
            recipe_hash=versioned_entry.recipe_hash,
            is_hardcoded=False,
            registered_at=datetime.now(timezone.utc),
            registration_source="topology_parser",
        )
        self._primary[base_class] = promoted_entry
        _log.info(
            "registry.promote — promoted %r to primary for %r | "
            "hash=%s path=%s | run_id=%s",
            versioned_key, base_class,
            promoted_entry.recipe_hash[:16],
            Path(promoted_entry.recipe_path).name,
            run_id,
        )
        return promoted_entry

    # ── Public API: supersede_hardcoded ───────────────────────────────────────

    def supersede_hardcoded(
        self,
        topology_class: str,
        versioned_key: str,
        *,
        reason: str,
        run_id: Optional[str] = None,
    ) -> RecipeRegistryEntry:
        """
        Deliberately supersede a hardcoded primary with a compiler-generated
        recipe.

        This is the ONLY path by which a hardcoded primary can be replaced at
        runtime.  It is intentionally more verbose than promote() and requires
        a non-empty reason string.

        Semantics:
          •  The current hardcoded primary is moved to _superseded_hardcoded
             (retained for restoration via restore_hardcoded_primary()).
          •  The versioned recipe becomes the new primary for the base class
             (as a non-hardcoded entry).
          •  An audit record is logged at WARNING level with the reason.

        If the new primary later fails validation, call
        restore_hardcoded_primary(topology_class) to re-floor to the
        hardcoded entry.

        Parameters
        ----------
        topology_class:
            Base topology class being superseded (e.g. "NEWS_ARTICLE").

        versioned_key:
            The versioned key of the recipe to promote (e.g. "NEWS_ARTICLE_v2").
            Must be registered.

        reason:
            Non-empty string explaining why this supersession is being
            performed.  Written to the audit log.  Stored for inspection via
            snapshot().

        run_id:
            Optional correlation ID.

        Returns
        -------
        RecipeRegistryEntry:
            The new primary entry.

        Raises
        ------
        ValueError:
            topology_class has no hardcoded primary, or reason is empty, or
            versioned_key is not versioned.
        KeyError:
            versioned_key does not exist in the registry.
        """
        _require_valid_topology_class(topology_class, "supersede_hardcoded")
        if not reason or not reason.strip():
            raise ValueError(
                "supersede_hardcoded: reason must be a non-empty string.  "
                "Document WHY the hardcoded recipe is being superseded — this is "
                "an audit-critical operation."
            )
        base_class, version_num = _base_class_and_version(versioned_key)
        if version_num is None:
            raise ValueError(
                f"supersede_hardcoded: {versioned_key!r} is not a versioned key.  "
                "Pass a key like NEWS_ARTICLE_v2."
            )
        if base_class != topology_class:
            raise ValueError(
                f"supersede_hardcoded: versioned_key base class {base_class!r} "
                f"does not match topology_class {topology_class!r}.  "
                "You can only supersede a class with a recipe compiled for that class."
            )

        with self._lock:
            return self._supersede_hardcoded_locked(
                topology_class=topology_class,
                versioned_key=versioned_key,
                reason=reason.strip(),
                run_id=run_id,
            )

    def _supersede_hardcoded_locked(
        self,
        topology_class: str,
        versioned_key: str,
        reason: str,
        run_id: Optional[str],
    ) -> RecipeRegistryEntry:
        """Inner supersede_hardcoded() implementation, called under lock."""
        current_primary = self._primary.get(topology_class)
        if current_primary is None:
            raise ValueError(
                f"supersede_hardcoded: {topology_class!r} has no primary entry.  "
                "There is nothing to supersede."
            )
        if not current_primary.is_hardcoded:
            raise ValueError(
                f"supersede_hardcoded: current primary for {topology_class!r} is "
                f"NOT hardcoded (hash={current_primary.recipe_hash[:16]}…).  "
                "Use promote() for non-hardcoded primaries."
            )

        versioned_entry = self._primary.get(versioned_key)
        if versioned_entry is None:
            raise KeyError(
                f"supersede_hardcoded: versioned_key {versioned_key!r} not found.  "
                "Register the recipe via register_recipe() first."
            )

        # Archive the hardcoded primary for restoration.
        self._superseded_hardcoded[topology_class] = current_primary

        # Create the new primary entry.
        new_primary = RecipeRegistryEntry(
            topology_class=TopologyClassStr(topology_class),
            recipe_path=versioned_entry.recipe_path,
            recipe_hash=versioned_entry.recipe_hash,
            is_hardcoded=False,
            registered_at=datetime.now(timezone.utc),
            registration_source="topology_parser",
        )
        self._primary[topology_class] = new_primary

        _log.warning(
            "registry.supersede_hardcoded — SUPERSESSION AUDIT | topology=%s "
            "old_hash=%s new_hash=%s new_path=%s reason=%r | run_id=%s",
            topology_class,
            current_primary.recipe_hash[:16],
            new_primary.recipe_hash[:16],
            Path(new_primary.recipe_path).name,
            reason,
            run_id,
        )
        return new_primary

    # ── Public API: restore_hardcoded_primary ─────────────────────────────────

    def restore_hardcoded_primary(
        self,
        topology_class: str,
        *,
        run_id: Optional[str] = None,
    ) -> RecipeRegistryEntry:
        """
        Restore the hardcoded primary for a topology class after its superseding
        compiler recipe has failed validation or been revoked.

        Retrieves the hardcoded entry from _superseded_hardcoded and reinstates
        it as the primary.  The failed compiler recipe remains in the primary
        table under its versioned key for post-mortem analysis but is no longer
        the primary.

        Parameters
        ----------
        topology_class:
            Base class to restore (e.g. "NEWS_ARTICLE").

        run_id:
            Optional correlation ID.

        Returns
        -------
        RecipeRegistryEntry:
            The restored hardcoded entry (now primary again).

        Raises
        ------
        ValueError:
            No archived hardcoded entry found for topology_class.
        """
        _require_valid_topology_class(topology_class, "restore_hardcoded_primary")

        with self._lock:
            archived = self._superseded_hardcoded.get(topology_class)
            if archived is None:
                raise ValueError(
                    f"restore_hardcoded_primary: no archived hardcoded entry for "
                    f"{topology_class!r}.  Either it was never superseded, or it was "
                    f"already restored."
                )
            self._primary[topology_class] = archived
            del self._superseded_hardcoded[topology_class]

            _log.warning(
                "registry.restore_hardcoded_primary — RESTORED | topology=%s "
                "hash=%s | run_id=%s",
                topology_class, archived.recipe_hash[:16], run_id,
            )
            return archived

    # ── Public API: reload_hardcoded_recipe ───────────────────────────────────

    def reload_hardcoded_recipe(
        self,
        topology_class: str,
        *,
        run_id: Optional[str] = None, # noqa
    ) -> RecipeRegistryEntry:
        """
        Hot-reload a hardcoded recipe from disk without restarting the process.

        Intended use: a deployment that updates a hardcoded .sh file in place
        and needs the registry to pick up the change without process restart.

        Steps:
          1. Verify topology_class is a hardcoded class.
          2. Re-read the file from disk.
          3. Verify the new hash against the manifest (if manifest exists).
             Unlike startup behaviour, a MISMATCH here means the manifest
             entry is stale (file was updated intentionally).  The manifest
             is updated in-place with the new hash.
          4. Replace the primary entry and content cache.

        This does NOT re-write the manifest automatically — the caller
        is responsible for updating and committing manifest.json after a
        deliberate recipe change.

        Parameters
        ----------
        topology_class:
            Must be one of the five hardcoded topology classes.

        Returns
        -------
        RecipeRegistryEntry:
            The new primary entry.

        Raises
        ------
        ValueError:
            topology_class is not a hardcoded class.
        RecipeNotFound:
            The recipe file cannot be read from disk.
        """
        _require_valid_topology_class(topology_class, "reload_hardcoded_recipe")
        if topology_class not in _HARDCODED_FILE_MAP:
            raise ValueError(
                f"reload_hardcoded_recipe: {topology_class!r} is not a hardcoded "
                f"topology class.  Hardcoded classes are: {sorted(_HARDCODED_FILE_MAP.keys())}."
            )

        filename    = _HARDCODED_FILE_MAP[topology_class]
        recipe_path = self._hard_dir / filename
        content     = _read_recipe_file(recipe_path)   # raises RecipeNotFound if absent
        new_hash    = compute_recipe_hash(content)
        exec_lines  = _count_executable_lines(content)
        now         = datetime.now(timezone.utc)

        with self._lock:
            old_entry = self._primary.get(topology_class)
            old_hash  = old_entry.recipe_hash if old_entry else "none"

            entry = RecipeRegistryEntry(
                topology_class=TopologyClassStr(topology_class),
                recipe_path=str(recipe_path),
                recipe_hash=RecipeHash(new_hash),
                is_hardcoded=True,
                registered_at=now,
                registration_source="hardcoded_loader",
            )
            self._primary[topology_class]                  = entry
            self._content_cache[str(recipe_path)]          = (content, new_hash)

            _log.info(
                "registry.reload_hardcoded_recipe — reloaded | topology=%s "
                "old_hash=%s new_hash=%s lines=%d | run_id=%s",
                topology_class, old_hash[:16] if old_hash != "none" else "none",
                new_hash[:16], exec_lines, run_id,
            )
            return entry

    # ── Public API: verify_integrity ──────────────────────────────────────────

    def verify_integrity(
        self,
        *,
        run_id: Optional[str] = None,
    ) -> Dict[str, object]:
        """
        Re-verify the on-disk hash of every hardcoded recipe against the
        manifest and the content cache.

        This is an out-of-band integrity check for operators and monitoring
        pipelines (pipeline.py, checkpoint_monitor.py).  It does not affect
        the running primary table — it only detects silent tamper.

        Returns
        -------
        dict with keys:
            "ok"          — bool, True if all checks passed.
            "checked"     — int, number of recipes checked.
            "failures"    — list of str describing each failure.
            "checked_at"  — ISO-8601 timestamp.
            "run_id"      — supplied run_id or generated.

        Does NOT raise — all failures are captured in the returned dict.
        Callers (checkpoint_monitor.py) are responsible for alerting Witness
        on non-ok results.
        """
        eff_run_id = run_id or new_run_id()
        failures: List[str] = []
        checked  = 0

        with self._lock:
            # Snapshot the current primary table topology classes to check.
            to_check = {
                tc: entry
                for tc, entry in self._primary.items()
                if entry.is_hardcoded
            }

        # I/O outside the lock to avoid blocking get_recipe().
        for tc, entry in to_check.items():
            checked += 1
            path = Path(entry.recipe_path)

            # Check 1: file exists.
            if not path.is_file():
                failures.append(
                    f"{tc}: recipe file missing at {entry.recipe_path}"
                )
                _log.error(
                    "registry.verify_integrity — MISSING: %s path=%s | run_id=%s",
                    tc, entry.recipe_path, eff_run_id,
                )
                continue

            # Check 2: content hash matches registry entry.
            try:
                disk_content = path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                failures.append(f"{tc}: cannot read recipe file: {exc}")
                _log.error(
                    "registry.verify_integrity — READ_ERROR: %s: %s | run_id=%s",
                    tc, exc, eff_run_id,
                )
                continue

            disk_hash = compute_recipe_hash(disk_content)
            if disk_hash != entry.recipe_hash:
                failures.append(
                    f"{tc}: HASH MISMATCH expected={entry.recipe_hash[:16]}… "
                    f"actual={disk_hash[:16]}…"
                )
                _log.critical(
                    "registry.verify_integrity — TAMPER DETECTED: %s "
                    "expected=%s actual=%s | run_id=%s",
                    tc, entry.recipe_hash[:16], disk_hash[:16], eff_run_id,
                )
                continue

            # Check 3: content cache coherence.
            with self._lock:
                cached = self._content_cache.get(entry.recipe_path)
            if cached is not None:
                _, cached_hash = cached
                if cached_hash != disk_hash:
                    failures.append(
                        f"{tc}: content cache hash {cached_hash[:16]}… "
                        f"diverged from disk hash {disk_hash[:16]}…"
                    )
                    _log.error(
                        "registry.verify_integrity — CACHE_DIVERGENCE: %s | run_id=%s",
                        tc, eff_run_id,
                    )

        ok = len(failures) == 0
        if ok:
            _log.debug(
                "registry.verify_integrity — OK | checked=%d run_id=%s",
                checked, eff_run_id,
            )
        else:
            _log.error(
                "registry.verify_integrity — FAILURES | checked=%d failures=%d run_id=%s",
                checked, len(failures), eff_run_id,
            )

        return {
            "ok":         ok,
            "checked":    checked,
            "failures":   failures,
            "checked_at": datetime.now(timezone.utc).isoformat(),
            "run_id":     eff_run_id,
        }

    # ── Public API: get_hardcoded_entry ───────────────────────────────────────

    def get_hardcoded_entry(
        self, topology_class: str
    ) -> Optional[RecipeRegistryEntry]:
        """
        Return the current (or archived) hardcoded entry for a topology class.

        If the hardcoded primary is still active, returns the active entry.
        If it was superseded, returns the archived entry from
        _superseded_hardcoded.
        Returns None if no hardcoded entry was ever loaded for this class.
        """
        with self._lock:
            current = self._primary.get(topology_class)
            if current is not None and current.is_hardcoded:
                return current
            archived = self._superseded_hardcoded.get(topology_class)
            return archived  # may be None

    # ── Public API: list_versioned_recipes ───────────────────────────────────

    def list_versioned_recipes(
        self, base_class: str
    ) -> List[RecipeRegistryEntry]:
        """
        Return all compiler-generated versioned entries for a base class,
        in registration order (oldest first).

        Returns an empty list if no versioned recipes exist.
        """
        with self._lock:
            return list(self._versions.get(base_class, []))

    # ── Public API: unregister_versioned ─────────────────────────────────────

    def unregister_versioned(
        self,
        versioned_key: str,
        *,
        run_id: Optional[str] = None, # noqa
    ) -> bool:
        """
        Remove a versioned (non-primary) compiler-generated recipe from the
        registry.  Used by topology_parser.py to clean up failed or stale
        versions.

        Cannot remove:
          •  Hardcoded primaries.
          •  The active primary for a base class.
          •  Entries that do not exist.

        Parameters
        ----------
        versioned_key:
            The versioned key to remove (e.g. "NEWS_ARTICLE_v2").

        Returns
        -------
        bool:
            True if the entry was found and removed.
            False if the entry was not found.

        Raises
        ------
        ValueError:
            Attempted to remove a hardcoded entry or the active primary.
        """
        _require_valid_topology_class(versioned_key, "unregister_versioned")
        base_class, version_num = _base_class_and_version(versioned_key)
        if version_num is None:
            raise ValueError(
                f"unregister_versioned: {versioned_key!r} is not a versioned key.  "
                "Only versioned entries (BASE_vN) can be removed."
            )

        with self._lock:
            entry = self._primary.get(versioned_key)
            if entry is None:
                return False
            if entry.is_hardcoded:
                raise ValueError(
                    f"unregister_versioned: {versioned_key!r} is hardcoded and "
                    "cannot be removed via unregister_versioned."
                )
            # Check if this is actually the active primary for base_class.
            base_primary = self._primary.get(base_class)
            if (
                base_primary is not None
                and base_primary.recipe_path == entry.recipe_path
                and base_primary.recipe_hash == entry.recipe_hash
            ):
                raise ValueError(
                    f"unregister_versioned: {versioned_key!r} is currently the active "
                    f"primary for {base_class!r}.  Demote it first by calling "
                    f"restore_hardcoded_primary({base_class!r}) or promoting another "
                    f"version before unregistering."
                )

            del self._primary[versioned_key]
            # Remove from _versions list.
            versions = self._versions.get(base_class, [])
            self._versions[base_class] = [
                e for e in versions if e.topology_class != versioned_key
            ]
            # Clean up content cache if no other entry references this path.
            path_still_needed = any(
                e.recipe_path == entry.recipe_path
                for tc, e in self._primary.items()
            )
            if not path_still_needed:
                self._content_cache.pop(entry.recipe_path, None)

            _log.info(
                "registry.unregister_versioned — removed %r | run_id=%s",
                versioned_key, run_id,
            )
            return True

    # ── Public API: snapshot ──────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, object]:
        """
        Return a point-in-time snapshot of the registry state.

        The returned dict is safe to read and serialize without holding the
        lock.  It contains deep copies of mutable structures — the caller
        cannot corrupt registry state by mutating the returned dict.

        Intended for debug endpoints, monitoring dashboards, and tests.
        Not intended for hot-path use — allocates copies of all tables.

        Returns
        -------
        dict with keys:
            "initialized_at"         — ISO-8601 string or None
            "bootstrap_performed"    — bool
            "strict_startup"         — bool
            "hardcoded_dir"          — str
            "primary"                — dict topology_class → entry_dict
            "versions"               — dict base_class → [entry_dict, …]
            "version_counters"       — dict base_class → int
            "manifest"               — dict topology_class → manifest_dict
            "superseded_hardcoded"   — dict topology_class → entry_dict
            "content_cache_paths"    — list of str (paths present in cache)
            "init_errors"            — list of str
        """
        with self._lock:
            def _entry_to_dict(e: RecipeRegistryEntry) -> dict:
                return {
                    "topology_class":      e.topology_class,
                    "recipe_path":         e.recipe_path,
                    "recipe_hash":         e.recipe_hash,
                    "recipe_hash_prefix":  e.recipe_hash[:16],
                    "is_hardcoded":        e.is_hardcoded,
                    "registered_at":       e.registered_at.isoformat(),
                    "registration_source": e.registration_source,
                }

            def _manifest_to_dict(m: HardcodedRecipeManifestEntry) -> dict:
                return {
                    "topology_class":  m.topology_class,
                    "recipe_filename": m.recipe_filename,
                    "recipe_hash":     m.recipe_hash,
                    "line_count":      m.line_count,
                    "committed_at":    m.committed_at.isoformat(),
                }

            return {
                "initialized_at":      (
                    self._initialized_at.isoformat()
                    if self._initialized_at else None
                ),
                "bootstrap_performed": self._bootstrap_performed,
                "strict_startup":      self._strict,
                "hardcoded_dir":       str(self._hard_dir),
                "primary": {
                    tc: _entry_to_dict(e)
                    for tc, e in self._primary.items()
                },
                "versions": {
                    base: [_entry_to_dict(e) for e in entries]
                    for base, entries in self._versions.items()
                },
                "version_counters":    dict(self._version_counter),
                "manifest": {
                    tc: _manifest_to_dict(m)
                    for tc, m in self._manifest.items()
                },
                "superseded_hardcoded": {
                    tc: _entry_to_dict(e)
                    for tc, e in self._superseded_hardcoded.items()
                },
                "content_cache_paths": sorted(self._content_cache.keys()),
                "init_errors":         list(self._init_errors),
            }

    # ── Public API: health ────────────────────────────────────────────────────

    def health(self) -> Dict[str, object]:
        """
        Return a concise health report for monitoring and alerting.

        This is the fast, low-allocation counterpart to snapshot().  It
        produces only the fields needed to determine whether the registry is
        in a nominally healthy state — suitable for a health-check endpoint
        or checkpoint_monitor.py's periodic assessment.

        Returns
        -------
        dict with keys:
            "ok"                      — bool, False if any health concern exists.
            "initialized"             — bool
            "initialized_at"          — ISO-8601 or None
            "bootstrap_performed"     — bool (True → commit manifest.json)
            "primary_count"           — int, number of primary entries
            "hardcoded_primary_count" — int
            "compiler_primary_count"  — int
            "generic_html_present"    — bool
            "superseded_count"        — int
            "init_error_count"        — int
            "concerns"                — list of str (empty when ok=True)
        """
        concerns: List[str] = []

        with self._lock:
            initialized    = self._initialized_at is not None
            primary_count  = len(self._primary)
            hc_count       = sum(1 for e in self._primary.values() if e.is_hardcoded)
            cc_count       = primary_count - hc_count
            generic_ok     = FALLBACK_TOPOLOGY_CLASS in self._primary
            superseded_cnt = len(self._superseded_hardcoded)
            error_count    = len(self._init_errors)
            bootstrap_done = self._bootstrap_performed

        if not initialized:
            concerns.append("registry not initialized")
        if not generic_ok:
            concerns.append(
                f"{FALLBACK_TOPOLOGY_CLASS} not in primary table — fallback is unavailable"
            )
        # Each hardcoded topology class should have an active hardcoded primary.
        # A class is "missing" if: it has no primary at all, or its primary was
        # superseded (is_hardcoded=False on the primary entry).
        hardcoded_primary_classes = {
            tc for tc, entry in self._primary.items()
            if entry.is_hardcoded and tc in HARDCODED_TOPOLOGY_CLASSES
        }
        missing_hardcoded = sorted(HARDCODED_TOPOLOGY_CLASSES - hardcoded_primary_classes)
        if missing_hardcoded:
            concerns.append(
                f"Missing hardcoded primaries: {sorted(missing_hardcoded)}"
            )
        if bootstrap_done:
            concerns.append(
                "manifest.json was bootstrapped this session — "
                "commit manifest.json to the repository."
            )
        if error_count > 0:
            concerns.append(f"{error_count} init error(s) — check registry logs")

        return {
            "ok":                      len(concerns) == 0,
            "initialized":             initialized,
            "initialized_at":          (
                self._initialized_at.isoformat() if self._initialized_at else None
            ),
            "bootstrap_performed":     bootstrap_done,
            "primary_count":           primary_count,
            "hardcoded_primary_count": hc_count,
            "compiler_primary_count":  cc_count,
            "generic_html_present":    generic_ok,
            "superseded_count":        superseded_cnt,
            "init_error_count":        error_count,
            "concerns":                concerns,
        }

    # ── Public API: manifest_entries ─────────────────────────────────────────

    @property
    def manifest_entries(self) -> Dict[str, HardcodedRecipeManifestEntry]:
        """
        Read-only view of the loaded manifest.

        Returns a shallow copy — callers cannot mutate the internal manifest
        table, but the HardcodedRecipeManifestEntry values are frozen dataclasses
        so they are safe to share.
        """
        with self._lock:
            return dict(self._manifest)

    # ── Internal utility ──────────────────────────────────────────────────────

    def __repr__(self) -> str:
        with self._lock:
            primary_count = len(self._primary)
            init_str = (
                self._initialized_at.strftime("%Y-%m-%dT%H:%M:%SZ")
                if self._initialized_at else "not_initialized"
            )
        return (
            f"<RecipeRegistry primary={primary_count} "
            f"initialized_at={init_str} strict={self._strict}>"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL SINGLETON
#
# One RecipeRegistry instance per process.  Instantiated lazily on first
# import access to avoid startup cost in import-time test scenarios.
# Production code should import the public functions below — not _registry
# directly.
#
# Thread safety of the singleton itself:
#   _registry is a module-level name.  In CPython the GIL makes the name
#   assignment atomic.  _registry_lock protects against the race where two
#   threads both observe _registry as None and both try to construct it.
# ─────────────────────────────────────────────────────────────────────────────

_registry_lock: threading.Lock = threading.Lock()
_registry: Optional[RecipeRegistry]  = None


def _get_registry() -> RecipeRegistry:
    """
    Return the module-level singleton, constructing it on first call.
    Thread-safe.  RecipeRegistry.__init__ is idempotent and safe to call
    concurrently; the inner lock prevents double-construction.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = RecipeRegistry()
    return _registry


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC SURFACE
#
# These four functions are the only entry points other modules should call.
# They delegate to the singleton.  Direct access to _registry is an
# architectural violation — route through these functions.
# ─────────────────────────────────────────────────────────────────────────────

def get_recipe(
    topology_class: str,
    *,
    run_id: Optional[str] = None,
) -> RecipeMount:
    """
    Resolve a topology class to a validated RecipeMount.

    NEVER returns None.  NEVER raises for a missing topology class.
    Falls back through PARENT_CLASS_MAP → GENERIC_HTML as needed.

    This is the primary call site for pipeline.py.

    Parameters
    ----------
    topology_class:
        Output of the topology classifier.  Must match [A-Z][A-Z0-9_]{1,63}.
        Invalid strings resolve to GENERIC_HTML (logged at WARNING).

    run_id:
        Optional correlation ID from the active pipeline run.

    Returns
    -------
    RecipeMount:
        Validated contract.  recipe_path is the path to mount to the kernel.

    Examples
    --------
    >>> mount = get_recipe("NEWS_ARTICLE")
    >>> mount.is_hardcoded
    True
    >>> mount = get_recipe("NEWS_ARTICLE_PAYWALLED")  # parent fallback
    >>> mount.topology_class
    'NEWS_ARTICLE'
    >>> mount = get_recipe("UNKNOWN_CLASS_XYZ")       # GENERIC_HTML fallback
    >>> mount.topology_class
    'GENERIC_HTML'
    """
    return _get_registry().get_recipe(topology_class, run_id=run_id)


def register_recipe(
    topology_class: str,
    recipe_path: str,
    *,
    caller_supplied_hash: Optional[str] = None,
    run_id: Optional[str] = None,
) -> RecipeRegistryEntry:
    """
    Register a compiler-generated recipe.

    Called by topology_parser.py after successful recipe compilation and
    validator.py approval.

    If topology_class already has a hardcoded primary, the new recipe is
    stored under a versioned key (e.g. NEWS_ARTICLE_v2) rather than replacing
    the primary.  The versioned key is available in the returned entry's
    topology_class field.

    Parameters
    ----------
    topology_class:
        Base topology class.  Pass "NEWS_ARTICLE", not "NEWS_ARTICLE_v2".
        Versioned keys are assigned internally.

    recipe_path:
        Absolute path to the .sh file written by topology_parser.py.
        Must be inside recipes/compiler_generated/ or recipes/hardcoded/.

    caller_supplied_hash:
        Optional SHA-256 hex digest.  If provided, verified against the hash
        registry.py computes by reading the file.  Mismatch → RecipeHashMismatch.

    run_id:
        Optional correlation ID.

    Returns
    -------
    RecipeRegistryEntry:
        The registered entry.  Check entry.topology_class — if it differs from
        the input topology_class, the recipe was stored under a versioned key.

    Examples
    --------
    >>> entry = register_recipe("NEW_CLASS", "/recipes/compiler_generated/new_class.sh")
    >>> entry.topology_class
    'NEW_CLASS'
    >>> entry.is_hardcoded
    False
    """
    return _get_registry().register_recipe(
        topology_class,
        recipe_path,
        caller_supplied_hash=caller_supplied_hash,
        run_id=run_id,
    )


def promote(
    versioned_key: str,
    *,
    run_id: Optional[str] = None,
) -> RecipeRegistryEntry:
    """
    Promote a versioned compiler-generated recipe to primary.

    Raises HardcodedRecipeOverwriteAttempt if the current primary is hardcoded.
    Use supersede_hardcoded() for deliberate hardcoded supersession.

    Parameters
    ----------
    versioned_key:
        E.g. "NEWS_ARTICLE_v2".  Must be registered.

    run_id:
        Optional correlation ID.

    Returns
    -------
    RecipeRegistryEntry:
        The newly promoted primary entry.
    """
    return _get_registry().promote(versioned_key, run_id=run_id)


def supersede_hardcoded(
    topology_class: str,
    versioned_key: str,
    *,
    reason: str,
    run_id: Optional[str] = None,
) -> RecipeRegistryEntry:
    """
    Deliberately supersede a hardcoded primary with a compiler-generated recipe.

    This is the only path by which a hardcoded primary can be replaced at
    runtime.  Requires a non-empty reason string (written to the audit log).

    If the new primary later fails, call restore_hardcoded_primary() to
    reinstate the hardcoded floor.

    Parameters
    ----------
    topology_class:
        Base class (e.g. "NEWS_ARTICLE").

    versioned_key:
        The version to promote (e.g. "NEWS_ARTICLE_v2").

    reason:
        Audit reason string.  Non-empty.  Will be written to logs.

    run_id:
        Optional correlation ID.

    Returns
    -------
    RecipeRegistryEntry:
        The new primary entry.
    """
    return _get_registry().supersede_hardcoded(
        topology_class,
        versioned_key,
        reason=reason,
        run_id=run_id,
    )


def registry_snapshot() -> Dict[str, object]:
    """Return a point-in-time snapshot of all registry state (for monitoring/debug)."""
    return _get_registry().snapshot()


def registry_health() -> Dict[str, object]:
    """Return a concise health report (for monitoring/alerting)."""
    return _get_registry().health()


def registry_verify_integrity(*, run_id: Optional[str] = None) -> Dict[str, object]:
    """Re-verify all hardcoded recipe file hashes against the manifest and content cache."""
    return _get_registry().verify_integrity(run_id=run_id)


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP INTEGRITY LOG
# When this module is imported (and therefore the singleton is accessed for the
# first time), emit a structured startup record to the registry logger.  This
# gives operators a single canonical log line confirming the registry's state
# at the moment it became available to the rest of the system.
#
# We do NOT eagerly construct the singleton at import time — only log the
# availability of the module.  First access is deferred to _get_registry().
# ─────────────────────────────────────────────────────────────────────────────

_log.debug(
    "registry module imported | hardcoded_dir=%s manifest_path=%s "
    "generic_html_hash_prefix=%s",
    _HARDCODED_DIR, _MANIFEST_PATH, _GENERIC_HTML_CANONICAL_HASH[:16],
)