"""
signal_kernel/recipes/validator.py
===================================
Security membrane between topology_parser.py's compiled recipes and the
kernel execution environment.

Every compiler-generated recipe passes through this before touching the
kernel. Every recipe. No exceptions.

This file answers one question per recipe: is this safe and structurally
sound enough to execute at all? Quality is a post-execution concern —
that is feedback.py's job. Safety is a pre-execution concern. They live
in different files.

Validator is synchronous, deterministic, and fast. It never calls an LLM,
never touches the network, never reads from the store. It reads the recipe
file and the fixtures. That is all.

Call order in pipeline.py:

    registry.get_recipe(topology_class)
        → validator.check(recipe_mount)     ← this file
            → if passes: mount and execute
            → if fails:  raise, never execute

What validator.py does NOT own:
    - Hash verification — registry owns this at three lifecycle points
    - Hardcoded recipe integrity — registry owns this
    - Recipe quality scoring — feedback.py owns this post-execution
    - Recipe storage or lookup — registry owns this
    - Any decision about which recipe to use — pipeline.py owns this

Imports:
    from contracts  — RecipeMount, constants, InjectionAuditRecord
    from exceptions — RecipeInjectionAttempt, RecipeStructuralViolation,
                      RecipeDryRunFailure

That is the complete import surface. No registry. No feedback. No world
model. Nothing else.

Dependency direction: validator.py → contracts.py, exceptions.py → nothing.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import logging
import os # noqa
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import FrozenSet, List, Optional, Sequence, Tuple # noqa

from signal_kernel.contracts import (
    ALLOWED_RECIPE_COMMANDS,
    INJECTION_PATTERNS,
    InjectionAuditRecord,
    MAX_RECIPE_LINE_COUNT,
    RecipeMount,
    compute_recipe_hash,
    new_run_id,
)
from signal_kernel.exceptions import (
    RecipeDryRunFailure,
    RecipeInjectionAttempt,
    RecipeStructuralViolation,
)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL CONSTANTS
#
# INJECTION_PATTERNS and ALLOWED_RECIPE_COMMANDS are imported from
# contracts.py — contracts.py is the single source of truth for the
# security membrane. This file does not redefine them.
#
# What IS defined here are compile-time derived artefacts and operational
# parameters that are validator-internal concerns.
# ═════════════════════════════════════════════════════════════════════════════

logger = logging.getLogger("signal_kernel.validator")

# Pre-compiled regex patterns from contracts.INJECTION_PATTERNS.
# Compiled once at module load. re.compile() is not free — the cost is
# paid here, not on every check() invocation. The compiled tuple is ordered
# identically to contracts.INJECTION_PATTERNS so that index correlation
# between the source pattern and the compiled pattern is trivial.
_COMPILED_INJECTION_PATTERNS: Tuple[Tuple[str, re.Pattern[str]], ...] = tuple(
    (raw, re.compile(raw)) for raw in INJECTION_PATTERNS
)

# Shell operators that separate commands in a pipeline or compound statement.
# Order matters: || must be tested before | to prevent a double-split on
# the first character. The regex uses alternation in length-descending order.
_SHELL_OPERATOR_SPLIT: re.Pattern[str] = re.compile(
    r"\|\||&&|\||;|\n"
)

# A valid command token in a recipe is strictly lowercase ASCII alpha.
# No digits, no paths, no shell metacharacters, no unicode. The allowed
# command set (grep, sed, awk, cat, cut, tr, head, tail, sort, uniq) is
# exclusively lowercase alpha. Anything that does not match this pattern
# as a command position token is either a shell builtin we do not permit
# or an injection attempt using path-qualified commands, variable expansion,
# or unicode homoglyph substitution.
_VALID_COMMAND_TOKEN: re.Pattern[str] = re.compile(r"^[a-z]+$")

# Patterns that indicate shell features recipes must never use.
# These are not in contracts.INJECTION_PATTERNS because they are structural
# concerns (recipes should not use these features at all), not injection
# signatures (which indicate deliberate hostile payloads). The distinction
# matters for audit classification: injection → forensic review, structural
# → compiler bug review.
_FORBIDDEN_SHELL_FEATURES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    ("heredoc",             re.compile(r"<<[-~]?\s*\w")),
    ("process_substitution", re.compile(r"[<>]\(")),
    ("brace_expansion",     re.compile(r"\{[^}]*,[^}]*\}")), # noqa
    ("arithmetic_expansion", re.compile(r"\$\(\(")),
    ("null_byte",           re.compile(r"\x00")),
    ("carriage_return",     re.compile(r"\r")),
    ("function_definition", re.compile(r"\b\w+\s*\(\s*\)")),
    ("source_builtin",      re.compile(r"(?:^|\s)\.\s+/")),
    ("tilde_expansion",     re.compile(r"(?:^|\s)~/")),
    ("coproc",              re.compile(r"\bcoproc\b")),
    ("select_loop",         re.compile(r"\bselect\b")),
)

# Minimum transformation commands: at least one of these must appear.
# A recipe with no grep and no sed has no transformation logic and is
# nonsensical — the compiler produced a recipe that does not extract anything.
_MINIMUM_TRANSFORM_COMMANDS: FrozenSet[str] = frozenset({"grep", "sed"})

# ── Fixture configuration ─────────────────────────────────────────────────

# Fixture base directory is relative to this file's location.
# validator.py lives at signal_kernel/recipes/validator.py.
# Fixtures live at signal_kernel/recipes/test_fixtures/{topology_class}/.
_FIXTURE_BASE_DIR: Path = Path(__file__).resolve().parent / "test_fixtures"

# Exact number of fixture files required per topology class.
# Not a minimum — exactly three. If the fixture set is incomplete, the
# dry-run cannot produce a meaningful pass/fail result and validation
# fails immediately. Fixtures must exist for every topology class the
# compiler can produce.
_REQUIRED_FIXTURE_COUNT: int = 3

# Dry-run subprocess timeout in seconds. If a recipe hangs on a fixture
# for this long, the recipe is broken. Real recipes complete in milliseconds
# on fixture-sized pages. Ten seconds is generous by three orders of magnitude.
_DRY_RUN_TIMEOUT_SECONDS: int = 10

# Maximum recipe file size in bytes. A recipe file larger than this is
# not a shell pipeline — it is either a binary, a corrupted file, or an
# attempt to exhaust validator memory during content read. The largest
# hardcoded recipe (saas_docs.sh) is ~6KB. A 256KB ceiling is absurdly
# generous while still protecting against pathological inputs.
_MAX_RECIPE_FILE_BYTES: int = 256 * 1024


# ═════════════════════════════════════════════════════════════════════════════
# PRIVATE UTILITIES
#
# These are not validation steps — they are deterministic transformations
# used by the four validation functions. They do not raise domain exceptions
# and they do not log. They convert data from one form to another.
# ═════════════════════════════════════════════════════════════════════════════

def _read_recipe_content(recipe_mount: RecipeMount) -> str:
    """
    Read recipe file content as UTF-8 text.

    Validates:
      - File exists and is a regular file
      - File is within the size ceiling (_MAX_RECIPE_FILE_BYTES)
      - Content is valid UTF-8

    Returns the full recipe content as a string.
    Raises RecipeStructuralViolation on any failure.

    This function is called once per check() invocation. The OS page cache
    ensures repeated reads of the same small file are essentially free,
    but in practice this function is only called once and the result is
    passed to each validation step via check().
    """
    recipe_path = Path(recipe_mount.recipe_path)

    if not recipe_path.exists():
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Recipe file does not exist: {recipe_mount.recipe_path!r}. "
                "The registry resolved a path that is no longer on disk."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    if not recipe_path.is_file():
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Recipe path is not a regular file: {recipe_mount.recipe_path!r}. "
                f"stat mode: {oct(recipe_path.stat().st_mode)}. "
                "Symlinks, directories, and device nodes are not valid recipes."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    # Size check BEFORE read. Prevents memory exhaustion on a 2GB file
    # that someone placed at the recipe path (or a /dev/zero symlink that
    # passed the is_file check on some filesystems).
    file_size = recipe_path.stat().st_size
    if file_size > _MAX_RECIPE_FILE_BYTES:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Recipe file is {file_size:,} bytes, exceeding the "
                f"{_MAX_RECIPE_FILE_BYTES:,}-byte ceiling. "
                "A shell recipe this large is not a pipeline — check file integrity."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    if file_size == 0:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail="Recipe file is empty (0 bytes). An empty file is not a recipe.",
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    try:
        content = recipe_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Recipe file is not valid UTF-8: {exc}. "
                "Binary content or non-UTF-8 encoded files are not valid recipes."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        ) from exc
    except OSError as exc:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Failed to read recipe file: {exc}. "
                "Check file permissions, filesystem health, and Docker volume mounts."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        ) from exc

    return content


def _strip_shell_comments(content: str) -> str:
    """
    Remove shell comments from recipe content for command extraction.

    Handles:
      - Full-line comments (# at start of line after optional whitespace)
      - Shebang lines (#!/...)
      - Inline comments (# after command content, outside quotes)

    Does NOT remove # characters inside quoted strings — those are stripped
    by _strip_quoted_strings() which runs before this in the pipeline.

    The return value is NOT valid shell. It is a reduced form suitable only
    for command token extraction. Do not execute it.
    """
    lines = content.split("\n")
    result: List[str] = []

    for line in lines:
        stripped = line.strip()

        # Shebang — skip entirely.
        if stripped.startswith("#!"):
            continue

        # Full-line comment — skip.
        if stripped.startswith("#"):
            continue

        # Inline comments: find # that is not inside a quoted string.
        # At this point, quoted strings have NOT yet been stripped (caller
        # decides the ordering). We do a conservative approach: only strip
        # # that appears outside any single or double quote context.
        # For the command extraction pipeline, the caller strips quotes
        # first, then calls this function. So inline # characters that
        # were inside quotes are already gone.
        if "#" in line:
            in_single = False
            in_double = False
            for i, ch in enumerate(line):
                if ch == "'" and not in_double:
                    in_single = not in_single
                elif ch == '"' and not in_single:
                    in_double = not in_double
                elif ch == "#" and not in_single and not in_double:
                    line = line[:i]
                    break

        result.append(line)

    return "\n".join(result)


def _strip_quoted_strings(content: str) -> str:
    """
    Replace all single-quoted and double-quoted string bodies with empty
    markers. This prevents command extraction from finding command names
    inside awk programs, sed expressions, and grep patterns.

    In POSIX sh, single-quoted strings have no escape mechanism. Everything
    between ' and ' is literal. This makes single-quote stripping trivial.

    Double-quoted strings allow backslash escaping of " inside them. The
    regex handles this by matching \\\\" (escaped quote) as non-terminating.

    The replacement preserves the quote delimiters as empty strings ("")
    so that the shell operator structure (pipes, semicolons) is not disturbed.

    Returns content with all quoted bodies removed. The result is suitable
    for command token extraction but is NOT valid shell.
    """
    # Single quotes: match ' followed by anything that is not ' followed by '.
    # POSIX sh: no escape inside single quotes. This is exact.
    result = re.sub(r"'[^']*'", "''", content)

    # Double quotes: match " followed by either \\" (escaped quote) or any
    # character that is not an unescaped ". Non-greedy to handle multiple
    # quoted strings on the same line.
    result = re.sub(r'"(?:[^"\\]|\\.)*"', '""', result)

    return result


def _join_continuation_lines(content: str) -> str:
    """
    Join backslash-continuation lines into single logical lines.

    In shell, a line ending in \\ followed by a newline continues on the
    next physical line. The backslash and newline are both removed, and
    the two physical lines become one logical line.

    This must run BEFORE quote stripping, because a backslash inside a
    quoted string is not a continuation character — but at that point the
    quotes are intact and the regex will not match \\<newline> inside quotes
    because the quote stripping handles that context.

    In practice, recipes use continuation lines for readability in multi-flag
    commands (sed -e '...' \\ -e '...'). Joining them produces the complete
    command that the tokenizer can then split on pipes.
    """
    return content.replace("\\\n", " ")


def _extract_command_tokens(content: str) -> List[str]:
    """
    Extract all command-position tokens from pre-processed shell content.

    Precondition: content has already been through:
      1. _join_continuation_lines()
      2. _strip_quoted_strings()
      3. _strip_shell_comments()

    This function splits on shell operators (|, ||, &&, ;, newline) and
    extracts the first token from each resulting segment. That first token
    is the command for that pipeline stage.

    Handles variable assignments in command prefix position: VAR=value cmd
    is parsed as command 'cmd', not 'VAR=value'. Shell allows environment
    variable overrides before the command: LC_ALL=C grep ... — the command
    is grep, not LC_ALL=C.

    Returns a list of command tokens. May contain duplicates (e.g. multiple
    grep invocations). Caller is responsible for set operations.
    """
    segments = _SHELL_OPERATOR_SPLIT.split(content)
    commands: List[str] = []

    for segment in segments:
        segment = segment.strip()
        if not segment:
            continue

        tokens = segment.split()
        for token in tokens:
            # Skip shell variable assignments in command prefix position.
            # Pattern: WORD=VALUE before the actual command.
            # Valid variable names are [A-Za-z_][A-Za-z0-9_]* followed by =.
            if "=" in token and not token.startswith("="):
                prefix = token.split("=", 1)[0]
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", prefix):
                    continue

            # Skip shell redirection targets: >, <, >>, 2>, 2>&1, etc.
            # These are not commands. If a token starts with a redirection
            # operator, skip it and whatever follows.
            if token.startswith(("<", ">")):
                break
            if re.match(r"^\d+[<>]", token):
                break

            # This is the command token.
            commands.append(token)
            break

    return commands


def _load_fixtures(topology_class: str) -> List[Path]:
    """
    Load and validate fixture files for a topology class.

    Returns exactly _REQUIRED_FIXTURE_COUNT fixture file paths, sorted
    lexicographically for deterministic ordering across runs.

    Raises RecipeDryRunFailure if:
      - Fixture base directory does not exist
      - Topology class fixture directory does not exist
      - Fewer than _REQUIRED_FIXTURE_COUNT fixture files exist
      - Any fixture file is empty or unreadable
    """
    if not _FIXTURE_BASE_DIR.is_dir():
        raise RecipeDryRunFailure(
            run_id=new_run_id(),
            topology_class=topology_class,
            recipe_path="",
            recipe_hash="0" * 64,
            fixtures_tested=0,
            failure_detail=(
                f"Fixture base directory does not exist: {_FIXTURE_BASE_DIR}. "
                "Dry-run validation requires test fixtures. "
                "Run the fixture generation script or check the container volume mount."
            ),
        )

    fixture_dir = _FIXTURE_BASE_DIR / topology_class
    if not fixture_dir.is_dir():
        raise RecipeDryRunFailure(
            run_id=new_run_id(),
            topology_class=topology_class,
            recipe_path="",
            recipe_hash="0" * 64,
            fixtures_tested=0,
            failure_detail=(
                f"No fixture directory for topology class {topology_class!r}: "
                f"expected {fixture_dir}. "
                "Every topology class that the compiler targets must have fixtures. "
                "A class without fixtures cannot be validated."
            ),
        )

    # Collect all regular files in the fixture directory. Ignore hidden files
    # and subdirectories. Sort for deterministic ordering.
    fixture_files = sorted(
        f for f in fixture_dir.iterdir()
        if f.is_file() and not f.name.startswith(".")
    )

    if len(fixture_files) < _REQUIRED_FIXTURE_COUNT:
        raise RecipeDryRunFailure(
            run_id=new_run_id(),
            topology_class=topology_class,
            recipe_path="",
            recipe_hash="0" * 64,
            fixtures_tested=0,
            failure_detail=(
                f"Topology class {topology_class!r} has {len(fixture_files)} fixture(s), "
                f"but {_REQUIRED_FIXTURE_COUNT} are required. "
                f"Fixture directory: {fixture_dir}. "
                "Incomplete fixture sets produce unreliable dry-run results."
            ),
        )

    # Validate the first _REQUIRED_FIXTURE_COUNT fixtures are non-empty.
    selected = fixture_files[:_REQUIRED_FIXTURE_COUNT]
    for fixture_path in selected:
        if fixture_path.stat().st_size == 0:
            raise RecipeDryRunFailure(
                run_id=new_run_id(),
                topology_class=topology_class,
                recipe_path="",
                recipe_hash="0" * 64,
                fixtures_tested=0,
                failure_detail=(
                    f"Fixture file is empty: {fixture_path}. "
                    "Empty fixtures cannot validate recipe output. "
                    "Replace with a canonical test page for this topology class."
                ),
            )

    return selected


def _run_recipe_against_fixture(
    recipe_path: str,
    fixture_path: Path,
) -> Tuple[bool, str, str]:
    """
    Execute a recipe against a single fixture file.

    Returns (produced_output, stdout, stderr) where produced_output is True
    if the recipe wrote any non-whitespace content to stdout.

    On subprocess timeout or execution error, returns (False, "", stderr).
    Does not raise — dry-run logic aggregates results across all fixtures
    before deciding pass/fail.
    """
    try:
        fixture_content = fixture_path.read_bytes()
    except OSError as exc:
        logger.warning(
            "Failed to read fixture %s: %s",
            fixture_path, exc,
        )
        return False, "", str(exc)

    try:
        result = subprocess.run(
            ["sh", recipe_path],
            input=fixture_content,
            capture_output=True,
            timeout=_DRY_RUN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        logger.warning(
            "Recipe %s timed out after %ds on fixture %s",
            recipe_path, _DRY_RUN_TIMEOUT_SECONDS, fixture_path.name,
        )
        return False, "", f"timeout after {_DRY_RUN_TIMEOUT_SECONDS}s"
    except OSError as exc:
        logger.warning(
            "Failed to execute recipe %s: %s",
            recipe_path, exc,
        )
        return False, "", str(exc)

    stdout_text = result.stdout.decode("utf-8", errors="replace")
    stderr_text = result.stderr.decode("utf-8", errors="replace")

    # Non-whitespace output means the recipe extracted something.
    produced_output = bool(stdout_text.strip())

    return produced_output, stdout_text, stderr_text


# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION FUNCTIONS — THE FOUR GATES
#
# Every compiler-generated recipe passes through these four in sequence.
# The first that fails raises. The recipe never reaches the kernel.
#
# Gate 1: _check_injection     — security layer (forensic-grade)
# Gate 2: _check_allowlist     — structural integrity
# Gate 3: _check_line_count    — sanity bound
# Gate 4: _dry_run             — functional verification
#
# If all four pass, the recipe is safe and structurally sound enough to
# execute. Whether it produces *good* output is feedback.py's concern.
# ═════════════════════════════════════════════════════════════════════════════


def _check_injection(recipe_mount: RecipeMount, content: str) -> None:
    """
    Gate 1 — Injection detection.

    Scans the full recipe content against every pattern in
    contracts.INJECTION_PATTERNS. These patterns detect:

      - Command substitution: $(...), `...`
      - Destructive chaining: ; rm, ; curl, ; wget, ; nc, ; python, ; perl
      - Write to protected paths: > /etc, > /usr
      - Execution escalation: | bash, | sh, eval, exec
      - Network access: /dev/tcp
      - Proc filesystem: /proc/self
      - Obfuscation vectors: base64 -d, ${IFS}

    On detection:
      1. Build InjectionAuditRecord with full recipe content — this is the
         forensic record. The full content is stored intentionally. Do not
         truncate.
      2. Log the audit record at CRITICAL level with topology class and
         the matched pattern for immediate triage.
      3. Raise RecipeInjectionAttempt. Never continue.

    The injection check runs against the RAW recipe content. No preprocessing.
    No comment stripping. No quote stripping. Injection patterns inside
    comments or string literals are still hostile — a comment containing
    `$(rm -rf /)` might be activated by a shell parsing quirk or a
    multiline quote mismatch earlier in the file. The only safe answer is
    to reject any recipe containing any pattern anywhere.
    """
    for raw_pattern, compiled_pattern in _COMPILED_INJECTION_PATTERNS:
        match = compiled_pattern.search(content)
        if match is None:
            continue

        # ── Injection detected ────────────────────────────────────────
        # Build the forensic record first, then log, then raise.

        recipe_hash = compute_recipe_hash(content)

        audit_record = InjectionAuditRecord(
            detected_at=datetime.now(timezone.utc),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_hash,
            full_recipe_content=content,
            matched_pattern=raw_pattern,
            is_hardcoded=recipe_mount.is_hardcoded,
            compiler_metadata=None,
        )

        # Log the FULL audit record. Do not abbreviate. The forensic value
        # is in the complete recipe content — it tells topology_parser.py
        # maintainers exactly what the compiler produced and why.
        logger.critical(
            "INJECTION DETECTED — %s\n"
            "topology_class=%s recipe_hash=%s pattern=%r\n"
            "match_span=[%d:%d] match_text=%r\n"
            "── full recipe content ──\n%s\n"
            "── end recipe content ──",
            audit_record.summary_line(),
            recipe_mount.topology_class,
            recipe_hash,
            raw_pattern,
            match.start(),
            match.end(),
            match.group(),
            content,
        )

        raise RecipeInjectionAttempt(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_hash,
            matched_pattern=raw_pattern,
            is_hardcoded=recipe_mount.is_hardcoded,
            compiler_metadata=None,
        )


def _check_allowlist(recipe_mount: RecipeMount, content: str) -> None:
    """
    Gate 2 — Command allowlist enforcement.

    Verifies that every command invocation in the recipe is in
    contracts.ALLOWED_RECIPE_COMMANDS:
        {grep, sed, awk, cat, cut, tr, head, tail, sort, uniq}

    Also enforces:
      - At least one grep or sed invocation — a recipe with no
        transformation is nonsensical. The compiler produced a pipeline
        that reads stdin and writes stdout without extracting anything.
      - No forbidden shell features (heredocs, process substitution,
        brace expansion, arithmetic expansion, function definitions,
        null bytes, carriage returns).
      - Command tokens are strictly lowercase ASCII alpha — no paths,
        no variable expansion, no unicode homoglyphs.

    The analysis pipeline:
      1. _join_continuation_lines()  — merge \\ continuations
      2. _strip_quoted_strings()     — eliminate awk/sed/grep bodies
      3. _strip_shell_comments()     — eliminate comment text
      4. _extract_command_tokens()   — isolate command-position tokens
      5. Validate every token against ALLOWED_RECIPE_COMMANDS

    This ordering is deliberate. Continuation joining produces complete
    logical lines. Quote stripping removes awk program bodies that contain
    words like 'print', 'next', 'length' which are not commands but would
    false-positive against the allowlist. Comment stripping removes command
    names mentioned in documentation.
    """

    # ── Phase 1: Forbidden shell feature detection ────────────────────
    # These are structural prohibitions. The recipe must not use these
    # features regardless of what command they contain. A recipe with a
    # heredoc is not a streaming pipeline. A recipe with process
    # substitution has execution paths outside the visible pipeline.
    for feature_name, pattern in _FORBIDDEN_SHELL_FEATURES:
        match = pattern.search(content)
        if match is not None:
            raise RecipeStructuralViolation(
                run_id=new_run_id(),
                topology_class=recipe_mount.topology_class,
                recipe_path=recipe_mount.recipe_path,
                recipe_hash=recipe_mount.recipe_hash,
                failure_detail=(
                    f"Forbidden shell feature detected: {feature_name}. "
                    f"Match at position {match.start()}: {match.group()!r}. "
                    "Recipes must be streaming pipelines using only "
                    "the allowed command set. Heredocs, process substitution, "
                    "brace expansion, arithmetic expansion, function definitions, "
                    "and embedded null/CR bytes are not permitted."
                ),
                is_hardcoded=recipe_mount.is_hardcoded,
            )

    # ── Phase 2: Command token extraction ─────────────────────────────
    preprocessed = _join_continuation_lines(content)
    preprocessed = _strip_quoted_strings(preprocessed)
    preprocessed = _strip_shell_comments(preprocessed)

    command_tokens = _extract_command_tokens(preprocessed)

    if not command_tokens:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                "No command tokens found in recipe. "
                "After stripping comments and quoted strings, the recipe "
                "contains no identifiable command invocations. "
                "A recipe must be a pipeline of allowed commands."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    # ── Phase 3: Token format validation ──────────────────────────────
    # Every command token must be strictly lowercase alpha. This catches:
    #   - Path-qualified commands: /usr/bin/grep → fails
    #   - Variable-expanded commands: $cmd → fails
    #   - Mixed case: Grep, GREP → fails
    #   - Commands with digits: base64 → fails (also caught by injection)
    #   - Unicode homoglyphs: grеp (cyrillic е) → fails
    for token in command_tokens:
        if not _VALID_COMMAND_TOKEN.match(token):
            raise RecipeStructuralViolation(
                run_id=new_run_id(),
                topology_class=recipe_mount.topology_class,
                recipe_path=recipe_mount.recipe_path,
                recipe_hash=recipe_mount.recipe_hash,
                failure_detail=(
                    f"Invalid command token: {token!r}. "
                    "Command tokens must be strictly lowercase ASCII alpha "
                    "(matching ^[a-z]+$). "
                    "Path-qualified commands, variable expansion, and "
                    "non-ASCII characters are not permitted in recipes."
                ),
                is_hardcoded=recipe_mount.is_hardcoded,
            )

    # ── Phase 4: Allowlist check ──────────────────────────────────────
    for token in command_tokens:
        if token not in ALLOWED_RECIPE_COMMANDS:
            raise RecipeStructuralViolation(
                run_id=new_run_id(),
                topology_class=recipe_mount.topology_class,
                recipe_path=recipe_mount.recipe_path,
                recipe_hash=recipe_mount.recipe_hash,
                failure_detail=(
                    f"Command not in allowlist: {token!r}. "
                    f"Allowed commands: {sorted(ALLOWED_RECIPE_COMMANDS)}. "
                    "The recipe compiler produced a pipeline with a command "
                    "that the kernel does not permit. If this command is "
                    "needed, it must be added to ALLOWED_RECIPE_COMMANDS in "
                    "contracts.py and the security implications reviewed."
                ),
                is_hardcoded=recipe_mount.is_hardcoded,
            )

    # ── Phase 5: Minimum transformation check ─────────────────────────
    command_set = frozenset(command_tokens)
    if not command_set & _MINIMUM_TRANSFORM_COMMANDS:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Recipe contains no transformation command. "
                f"Found commands: {sorted(command_set)}. "
                f"At least one of {sorted(_MINIMUM_TRANSFORM_COMMANDS)} is required. "
                "A recipe with no grep or sed has no extraction logic — "
                "it would pass stdin to stdout unmodified or discard everything."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    logger.debug(
        "Allowlist check passed: topology=%s commands=%s",
        recipe_mount.topology_class,
        sorted(command_set),
    )


def _check_line_count(recipe_mount: RecipeMount, content: str) -> None:
    """
    Gate 3 — Line count sanity bound.

    Counts physical lines in the recipe content. If the count exceeds
    contracts.MAX_RECIPE_LINE_COUNT (50), the compiler has gone wrong.

    This is a sanity check, not a security check. A recipe over 50 lines
    is not hostile — it is a compiler that has lost coherence and is
    producing unreasonably complex pipelines. The correct response is to
    investigate the compiler, not to run a 200-line shell script against
    real pages.

    The line count includes all lines: comments, blank lines, continuation
    lines. This is the physical file size, not the logical command count.
    MAX_RECIPE_LINE_COUNT in contracts.py is calibrated against the hardcoded
    recipes (the largest is ~167 lines for saas_docs.sh, but hardcoded recipes
    are validated by hand and skip this gate). Compiler-generated recipes
    should be more concise than the hand-written ones.
    """
    line_count = content.count("\n") + (1 if content and not content.endswith("\n") else 0)

    if line_count > MAX_RECIPE_LINE_COUNT:
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            failure_detail=(
                f"Recipe is {line_count} lines, exceeding the "
                f"{MAX_RECIPE_LINE_COUNT}-line maximum. "
                "Compiler-generated recipes must be concise streaming pipelines. "
                "A recipe this long indicates the compiler has produced an "
                "unreasonably complex pipeline that needs investigation."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    logger.debug(
        "Line count check passed: topology=%s lines=%d max=%d",
        recipe_mount.topology_class,
        line_count,
        MAX_RECIPE_LINE_COUNT,
    )


def _dry_run(recipe_mount: RecipeMount, content: str) -> None: # noqa
    """
    Gate 4 — Dry-run against canonical test fixtures.

    Runs the recipe against three fixture files for its topology class
    from recipes/test_fixtures/{topology_class}/. If ALL three produce
    empty output, the recipe fails validation.

    If at least one fixture produces non-empty output, the recipe passes.
    This is deliberately permissive — some fixtures may represent edge cases
    (paywalled pages, error pages) that legitimately produce no output. The
    bar is: does this recipe work on at least one canonical page?

    Hardcoded recipes skip this gate — they were validated by hand before
    commit. Only compiler-generated recipes are dry-run tested.

    The subprocess invocation mirrors pipeline.py's mechanism: the recipe
    is executed as `sh recipe.sh` with the fixture content piped to stdin.
    This ensures the dry run exercises the same code path as production.

    Fixture files must exist for every topology class the compiler can
    target. A missing fixture directory raises immediately — do not execute
    an unvalidated recipe against a real page.
    """
    fixture_paths = _load_fixtures(recipe_mount.topology_class)

    produced_any_output = False
    fixture_results: List[Tuple[str, bool, str]] = []

    for fixture_path in fixture_paths:
        produced_output, stdout, stderr = _run_recipe_against_fixture(
            recipe_mount.recipe_path,
            fixture_path,
        )

        fixture_results.append((fixture_path.name, produced_output, stderr))

        if produced_output:
            produced_any_output = True

        if stderr.strip():
            logger.debug(
                "Dry-run stderr for topology=%s fixture=%s: %s",
                recipe_mount.topology_class,
                fixture_path.name,
                stderr.strip()[:500],
            )

    if not produced_any_output:
        # Build a diagnostic summary of what each fixture produced.
        fixture_summary = "; ".join(
            f"{name}: {'output' if ok else 'empty'}"
            + (f" (stderr: {err[:100]})" if err.strip() else "")
            for name, ok, err in fixture_results
        )

        raise RecipeDryRunFailure(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=recipe_mount.recipe_hash,
            fixtures_tested=len(fixture_results),
            failure_detail=(
                f"Recipe produced empty output on all {len(fixture_results)} "
                f"test fixtures. Fixture results: [{fixture_summary}]. "
                "The recipe passed injection and structural checks but does "
                "not extract any content from canonical test pages. "
                "Possible causes: inverted grep logic, over-specified patterns, "
                "incorrect zone extraction, or awk state machine errors. "
                "topology_parser.py must recompile for this class. "
                "GENERIC_HTML fallback will be used until a valid recipe is produced."
            ),
        )

    # Log which fixtures produced output for diagnostic tracing.
    passed_count = sum(1 for _, ok, _ in fixture_results if ok)
    logger.debug(
        "Dry-run passed: topology=%s fixtures=%d/%d produced output",
        recipe_mount.topology_class,
        passed_count,
        len(fixture_results),
    )


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC API — ONE FUNCTION
#
# check() is the only public function in this file. It is the complete
# interface that pipeline.py and registry.py call. If you are importing
# anything else from this module, something is architecturally wrong.
# ═════════════════════════════════════════════════════════════════════════════


def check(recipe_mount: RecipeMount) -> None:
    """
    Validate a recipe for safe kernel execution.

    Hardcoded recipes return immediately — they were validated by hand
    before commit and their integrity is verified by registry.py via
    SHA-256 hash manifest on every load. Validator does not re-check
    what registry already guarantees.

    Compiler-generated recipes pass through four sequential gates:

      1. Injection detection  — the security layer
      2. Command allowlist    — structural integrity
      3. Line count bound     — compiler sanity check
      4. Dry-run validation   — functional verification

    The first gate that fails raises. The recipe never reaches the kernel.
    No exception is caught internally — all four exceptions are hard stops
    that propagate to pipeline.py, which propagates them to the AXIOM graph.

    On success: returns None. The recipe is safe and structurally sound
    enough to execute. Whether it produces *good* output is feedback.py's
    concern after execution.

    On failure: raises one of:
      - RecipeInjectionAttempt   — security failure, forensic review required
      - RecipeStructuralViolation — structural failure, compiler bug review
      - RecipeDryRunFailure      — functional failure, recompilation needed

    All three are subclasses of RecipeMountError. All three are hard stops.
    pipeline.py re-raises them to the AXIOM graph.

    Parameters
    ----------
    recipe_mount : RecipeMount
        The recipe to validate. Must be a fully constructed RecipeMount
        with a valid recipe_path pointing to a readable file on disk.
    """
    # ── Hardcoded recipes: skip all validation ────────────────────────
    # Registry owns integrity checking for hardcoded recipes via SHA-256
    # hash manifest verification at three lifecycle points (startup,
    # per-load, and periodic audit). Validator does not duplicate that work.
    if recipe_mount.is_hardcoded:
        logger.debug(
            "Skipping validation for hardcoded recipe: topology=%s path=%s",
            recipe_mount.topology_class,
            recipe_mount.recipe_path,
        )
        return

    # ── Read recipe content once ──────────────────────────────────────
    # All four gates need the content. Read it once. The file is small
    # (bounded by _MAX_RECIPE_FILE_BYTES). _read_recipe_content validates
    # existence, file type, size, and UTF-8 encoding.
    content = _read_recipe_content(recipe_mount)

    # ── Verify recipe hash matches mount ──────────────────────────────
    # The content we just read must hash to what the RecipeMount claims.
    # If it does not, the file was modified between registration and
    # validation — a race condition or a filesystem tamper event.
    # This is defense-in-depth: registry also checks this, but validator
    # verifies independently because it is the last gate before execution.
    actual_hash = compute_recipe_hash(content)
    if actual_hash != recipe_mount.recipe_hash:
        logger.critical(
            "Recipe hash mismatch during validation: topology=%s "
            "expected=%s actual=%s path=%s",
            recipe_mount.topology_class,
            recipe_mount.recipe_hash[:16],
            actual_hash[:16],
            recipe_mount.recipe_path,
        )
        raise RecipeStructuralViolation(
            run_id=new_run_id(),
            topology_class=recipe_mount.topology_class,
            recipe_path=recipe_mount.recipe_path,
            recipe_hash=actual_hash,
            failure_detail=(
                f"Recipe content hash mismatch. "
                f"RecipeMount claims {recipe_mount.recipe_hash[:16]}... "
                f"but file content hashes to {actual_hash[:16]}... "
                "The file was modified between registration and validation. "
                "This is either a race condition in topology_parser.py or "
                "a filesystem tamper event. Investigate immediately."
            ),
            is_hardcoded=recipe_mount.is_hardcoded,
        )

    # ── Gate 1: Injection detection ───────────────────────────────────
    _check_injection(recipe_mount, content)

    # ── Gate 2: Command allowlist ─────────────────────────────────────
    _check_allowlist(recipe_mount, content)

    # ── Gate 3: Line count sanity ─────────────────────────────────────
    _check_line_count(recipe_mount, content)

    # ── Gate 4: Dry-run against fixtures ──────────────────────────────
    _dry_run(recipe_mount, content)

    logger.info(
        "Validation passed: topology=%s hash=%s path=%s",
        recipe_mount.topology_class,
        recipe_mount.recipe_hash[:8],
        recipe_mount.recipe_path,
    )