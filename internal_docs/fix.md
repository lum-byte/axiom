# parser.py — Bug Fix Reference

All issues ordered by severity. Each entry: problem number, LOC, root cause, fix.

---

## 🔴 Critical

---

### #1 — `frozen=True` + `cached_property` crashes on first property access
**LOC:** `474`, `490`, `499`, `695`, `710`, `714`

**Problem:**  
`ParsedSelector` and `CompiledRecipe` are `@dataclass(frozen=True)` but use `@cached_property`. Frozen dataclasses disallow `__dict__` writes; `cached_property` does exactly that on first access → `FrozenInstanceError` every time a property is touched.

**Do NOT unfreeze.** Immutability is correct and consistent with AXIOM. Pre-compute in `__post_init__` instead.

```python
# ParsedSelector — lines 474–509
@dataclass(frozen=True)
class ParsedSelector:
    raw: str
    kind: SelectorKind
    tag: Optional[str]
    class_name: Optional[str]
    id_name: Optional[str]
    # ... other fields ...
    is_negation: bool

    # Add these as computed fields with defaults
    is_simple: bool = field(init=False)
    grep_pattern: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "is_simple", self.kind in (
            SelectorKind.CLASS_SELECTOR,
            SelectorKind.ID_SELECTOR,
            SelectorKind.TAG_SELECTOR,
        ))
        if self.kind == SelectorKind.CLASS_SELECTOR and self.class_name:
            pat = f'class="[^"]*{re.escape(self.class_name)}[^"]*"'
        elif self.kind == SelectorKind.ID_SELECTOR and self.id_name:
            pat = f'id="{re.escape(self.id_name)}"'
        elif self.kind == SelectorKind.TAG_SELECTOR and self.tag:
            pat = f"<{self.tag}[ >]"
        else:
            pat = re.escape(self.raw)
        object.__setattr__(self, "grep_pattern", pat)

# CompiledRecipe — lines 695–720
@dataclass(frozen=True)
class CompiledRecipe:
    content: str
    # ... other fields ...
    checksum: str = field(init=False)
    is_valid_complexity: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "checksum",
            compute_recipe_hash(self.content)   # use the contract function, not inline hashlib
        )
        object.__setattr__(self, "is_valid_complexity", (
            self.line_count <= MAX_RECIPE_LINES
            and self.stage_count <= MAX_PIPELINE_STAGES
        ))
```
`object.__setattr__` is the correct way to write to frozen dataclass fields from `__post_init__`.

---

### #2 — `noise_patterns` reset to `[]` inside loop — child noise never works
**LOC:** `2150–2156`

**Problem:**  
`noise_patterns` is initialised at `2150`, then **reassigned to a new empty list** at `2156` inside the `for` loop body — erasing every entry as it's built. The `BEGIN` block at `2172` iterates an always-empty list.

```python
# BROKEN
noise_patterns: list[tuple[str, str]] = []          # line 2150
for i, ns in enumerate(zone.child_noise_selectors):
    pat = _selector_to_awk_pattern(ns)
    if pat:
        close_tag = ns.tag if ns.tag else "div"
        noise_patterns: list[tuple[str, str, str]] = []  # line 2156 — WIPES THE LIST

# FIX — build the list, don't reset it
noise_patterns: list[tuple[str, str, str]] = []
for i, ns in enumerate(zone.child_noise_selectors):
    pat = _selector_to_awk_pattern(ns)
    if pat:
        close_tag = ns.tag if ns.tag else "div"
        noise_patterns.append((f"in_noise_{i}", pat, close_tag))
```

---

### #3 — Intent awk missing void element exclusion → depth counter drifts
**LOC:** `5115–5135` (intent awk `BEGIN`/depth block in `IntentConditionedExtractor`)

**Problem:**  
The depth counter at `5131–5133` counts `<tag>` opens and `</tag>` closes with `gsub`. Void elements (`<br>`, `<img>`, `<input>`, etc.) have no closing tag, so each one permanently increments depth. Over a page with dozens of `<br>` the counter drifts, causing capture zones to never close. The zone-extract awk at `2143` already has the fix; intent awk doesn't.

```python
# After line 5130 (the opening `{` block), insert before the gsub lines:
lines.append(
    "    gsub(/<(area|base|br|col|embed|hr|img|input|link|meta|"
    "param|source|track|wbr)(\\s[^>]*)?\\/?>/, \"\")"
)
# This strips void elements from the line before depth counting,
# so they contribute neither an open nor a close.
```

---

### N3 — `MAX_RECIPE_LINE_COUNT = 50` silently discards all complex recipes
**LOC:** `249`, `5659–5662`, `6380–6383`

**Problem:**  
`contracts.py` sets `MAX_RECIPE_LINE_COUNT = 50`. Parser imports it directly as `MAX_RECIPE_LINES`. `_sanitize_recipe()` rejects any recipe exceeding 50 lines before writing. The Wikipedia awk emitter alone produces ~157 lines; forum-thread and multizone programs routinely exceed 50. Every specialist recipe is silently rejected. Only trivial single-zone sed chains survive.

```python
# At line 249, shadow the contracts constant with a compiler-appropriate ceiling:
MAX_RECIPE_LINES: Final[int] = max(MAX_RECIPE_LINE_COUNT, 500)
# contracts.py's value is a floor for simple recipes, not a ceiling for compiled programs.
# 500 accommodates the largest awk programs the compiler generates (~350 lines).
```
Also update `RecipeMount.__post_init__` if it enforces the same limit — that will hard-raise `ValueError` at mount time for any recipe > 50 lines.

---

### S6 — `register_recipe()` never called — compiler output invisible to pipeline
**LOC:** `6089–6130`

**Problem:**  
`RecipeRegistryEntry` is imported at `163` but never instantiated. `register_recipe()` is called conditionally at `6113` but only inside a block that checks a flag that is never set to `True`. `pipeline.py` reads recipes via `get_recipe()` against the registry. Since the registry is never written, the kernel permanently serves `GENERIC_HTML` for every compiler-generated topology class regardless of what's on disk.

```python
# After _write_recipe_atomic() succeeds (~line 6090), add:
from signal_kernel.recipes.registry import register_recipe

entry = register_recipe(
    topology_class=topology_class,
    recipe_path=str(recipe_path),
    caller_supplied_hash=checksum,
    run_id=run_id,
)
log.info(
    "recipe_registered",
    topology_class=topology_class,
    registry_version=entry.version,
)
```
Remove the dead `RecipeRegistryEntry` import at line `163` (see also N6).

---

## 🟠 High

---

### #4 — Unescaped backslash in shell pattern string
**LOC:** Wherever shell patterns are built with raw `\` — cross-reference `ShellPattern` construction in sed/awk emitters.

**Problem:**  
Python strings passed to shell generators with a single `\` produce a literal backslash in the shell script, which the shell then mis-parses. Affects any pattern using `\s`, `\w`, `\d` etc.

```python
# Escape before inserting into shell strings:
pattern = pattern.replace("\\", "\\\\")
```

---

### #5 — Boundary sed hardcodes `</div>` — non-div boundaries never close
**LOC:** `4244–4253`

**Problem:**  
`_compile_boundary_sed_commands()` builds sed ranges using `/<\/div>/` as the closing delimiter for both `NOISE_BOUNDARY` and `CONTENT_BOUNDARY` regardless of the actual opening tag. A `<section>` boundary opens but its range never matches `</div>`, so sed captures to EOF.

```python
# Replace the hardcoded </div> with the real closing tag:
if "<" in delimiter:
    tag_match = re.search(r"<(\w+)", delimiter)
    close_tag = tag_match.group(1) if tag_match else "div"
else:
    close_tag = "div"

# NOISE_BOUNDARY
commands.append(ShellCommand("sed", (
    ShellPattern(f"/{safe_delim}/,/<\\/{close_tag}>/d"),
)))

# CONTENT_BOUNDARY
commands.append(ShellCommand("sed", (
    ShellFlag("-n"),
    ShellPattern(f"/{safe_delim}/,/<\\/{close_tag}>/p"),
)))
```

---

### #6 — `content_active_N` variables uninitialized in awk `BEGIN`
**LOC:** `4261–4312` (`_compile_boundary_awk_transitions`), `2168–2174` (BEGIN emitter)

**Problem:**  
`_compile_boundary_awk_transitions()` emits `content_active_N` toggle lines at `4295` but the `BEGIN` block at `2168` only initialises variables from the `noise_patterns` list — never `content_active_N`. In awk, uninitialized variables are `""` (empty string), not `0`. The toggle `{ content_active_1 = !content_active_1 }` on an empty string produces `1` correctly on the first hit, but the guard `content_active_1 { print }` tests truthiness — meaning it's truthy from the start before any boundary is seen if there's any implicit initialization issue.

```python
# In the BEGIN block emitter, after noise vars, add content_active vars.
# _compile_boundary_awk_transitions needs to return the variable names it created:

def _compile_boundary_awk_transitions(boundaries):
    ...
    content_vars = []
    for bd in boundaries:
        if btype == "CONTENT_BOUNDARY":
            var = f"content_active_{content_count}"
            content_vars.append(var)          # track for BEGIN init
            lines.append(f"/{safe_delim}/ {{ {var} = !{var} }}")
            content_count += 1
    return lines, content_vars   # return vars alongside lines

# In BEGIN block generation, initialize each:
for var in content_vars:
    lines.append(f"    {var} = 0")
```

---

### #7 — Set-based line diff collapses duplicates — counts are wrong
**LOC:** `6827–6841`

**Problem:**  
`diff_recipes()` converts recipe content to `set` before diffing. Identical lines (e.g. `| sed 's/<[^>]*>//g'` appearing multiple times) are deduplicated. `lines_added` and `lines_removed` counts are understated. For recipes with repeated sed stages this can report 0 changes when dozens of lines changed.

```python
# Replace set diff with Counter diff:
from collections import Counter

old_lines = Counter(old.content.strip().split("\n"))
new_lines = Counter(new.content.strip().split("\n"))
added   = new_lines - old_lines
removed = old_lines - new_lines

return RecipeDiff(
    ...
    lines_added=sum(added.values()),
    lines_removed=sum(removed.values()),
    ...
)
```

---

### #8 — PID derivative term mixes EMA and raw error — term is meaningless
**LOC:** `586–592`, `614–619`

**Problem:**  
The integral accumulates `self.ema_error` (smoothed), but the derivative at `586` is `error - self.pid_prev_error` where `pid_prev_error` is the raw instantaneous error stored at `587`. Mixing EMA and raw signals makes the derivative track noise, not trend. The `pid_output` property at `619` duplicates the same mistake.

```python
# Store and diff the EMA value, not the raw error:

def update(self, ...):
    ...
    # EMA update
    self.ema_error = EMA_ALPHA * noise_ratio + (1 - EMA_ALPHA) * self.ema_error
    
    # PID — all terms on the same smoothed signal
    self.pid_integral += self.ema_error
    derivative = self.ema_error - self.pid_prev_error   # was: error - self.pid_prev_error
    self.pid_prev_error = self.ema_error                 # was: self.pid_prev_error = error

    correction = (
        PID_KP * self.ema_error
        + PID_KI * self.pid_integral
        + PID_KD * derivative
    )
```

---

### #9 — `GENERIC_JSON` awk matches `{`/`}` inside string values
**LOC:** `7871–7876` (`_RECIPE_TEMPLATES["GENERIC_JSON"]`)

**Problem:**  
The depth counter in the template blindly matches every `{` and `}`. JSON values like `"url": "https://example.com/{id}"` increment/decrement depth incorrectly. On any JSON with braces in string values, depth tracking corrupts and extraction cuts off mid-object or never closes.

```awk
# Replace the template body:
BEGIN { depth=0; in_str=0 }
{
    # toggle string tracking on unescaped quotes
    n = split($0, chars, "\"")
    for (i=1; i<=length($0); i++) {
        c = substr($0,i,1)
        if (c == "\"" && (i==1 || substr($0,i-1,1) != "\\")) in_str = !in_str
        if (!in_str && c == "{") depth++
        if (!in_str && c == "}") depth--
    }
}
depth == 1 { print }
```
For the template-level fix a simpler guard is acceptable:
```awk
BEGIN { depth=0 }
/^[^"]*{/ && !/"[^"]*{[^"]*"/ { depth++ }
/^[^"]*}/ && !/"[^"]*}[^"]*"/ { depth-- }
depth == 1 { print }
```

---

### N5 — `FeedbackEvent` never subscribed — recompile loop has no entry point
**LOC:** `5899–5912` (`initialize()` subscription block)

**Problem:**  
`FeedbackEvent` is imported at `171` and `_on_feedback_event` exists at `6296`, but no subscription is registered in `initialize()`. The kernel's quality-signal path (`FeedbackEvent.recompilation_recommended`) has no listener. Recompilation only triggers on new `ZoneMapUpdatedEvent`s — never on kernel-detected recipe degradation.

```python
# Add inside the `if hasattr(self._bus, "subscribe"):` block at line 5900:
await self._bus.subscribe(
    "feedback_event",
    self._on_feedback_event_sync_wrapper,   # bus may expect sync handler
)

# If the bus delivers events synchronously to handlers:
def _on_feedback_event_sync_wrapper(self, event: FeedbackEvent) -> None:
    self._on_feedback_event(event)   # already exists, already handles loop dispatch
```

---

### N2a — `_determine_strategy()` primary/fallback branches inverted
**LOC:** `5422–5445`

**Problem:**  
The comment labels the `extraction_strategy` branch "Real ZoneMap" and the `strategy` branch "Legacy/test fallback". `contracts.py ZoneMap` has `.strategy`, not `.extraction_strategy`. So every production ZoneMap from the bus hits the path labelled "Legacy" and every test ZoneMap with `.extraction_strategy` hits "Real". Behavior is accidentally correct today, but any future ZoneMap adding `.extraction_strategy` will silently hit the wrong branch.

```python
def _determine_strategy(zone_map: Any) -> ExtractionStrategy:
    # Primary: contracts.py ZoneMap uses .strategy
    if hasattr(zone_map, "strategy"):
        return ExtractionStrategy.from_str(zone_map.strategy)
    # Secondary: future schema or EmptyZoneMap with .extraction_strategy
    if hasattr(zone_map, "extraction_strategy"):
        return ExtractionStrategy.from_zone_map(zone_map)
    return ExtractionStrategy.ZONE_EXTRACT
```

---

## 🟡 Medium

---

### #10 — `compile()` reads shared dicts without lock — concurrent compiles clobber state
**LOC:** `5932–5960` (`compile()` preamble)

**Problem:**  
`compile()` reads `self._zone_map_cache` and `self._phase_cache` without holding `self._lock`. A concurrent `handle_zone_map_updated()` call (which writes under lock) can mutate these dicts mid-read. Python dict reads are not atomic under all conditions; at minimum the behavior is undefined.

```python
async def compile(self, zone_map, phase=None):
    # Snapshot mutable shared state under the lock before any work
    async with self._lock:
        phase_str = self._resolve_phase(zone_map.topology_class, phase)
        feedback_state = copy.copy(self._feedback_state.get(
            zone_map.topology_class, FeedbackState()
        ))
    # Use phase_str and feedback_state (local copies) from here on
```

---

### #11 — Checksum read from disk instead of `compute_recipe_hash()` — TOCTOU
**LOC:** `5748` (now fixed per session — confirm it reads `compute_recipe_hash(content)`)

**Problem:**  
If the checksum was still computed by reading back the staged file from disk after writing, that's a TOCTOU window (file could change between write and read) and violates the contract's explicit doc: *"Never inline hashlib.sha256() for recipe hashing."*

```python
# Correct form — content is already in memory:
checksum = compute_recipe_hash(content)
# Not: hashlib.sha256(staging_path.read_bytes()).hexdigest()
```
Verify line `5748` uses this form. If `CompiledRecipe.__post_init__` now computes the hash (per fix #1), ensure it also uses `compute_recipe_hash`, not inline `hashlib`.

---

### S9 — Cold-start parent fallback always misses — `_zone_map_cache` is in-memory only
**LOC:** `5577–5600` (`_walk_parent_map`)

**Problem:**  
`_walk_parent_map()` lookups go to `self._zone_map_cache`, which is populated only from `ZoneMapUpdatedEvent`s received since process start. On cold start the cache is empty. Every topology class with a parent falls through to `GENERIC_HTML` on first compile regardless of what the persistent store holds.

```python
def _get_zone_map_for_class(self, topology_class: str) -> Optional[ZoneMap]:
    """Check in-memory cache first, then fall back to mmap store."""
    if topology_class in self._zone_map_cache:
        return self._zone_map_cache[topology_class]
    return _load_zone_map_from_mmap(topology_class)  # implement mmap read

# In _walk_parent_map, replace direct cache lookups:
# was:  parent_zm = self._zone_map_cache.get(parent_class)
# fix:  parent_zm = self._get_zone_map_for_class(parent_class)
```

---

### S12 — `asyncio.Lock()` created at module import time → hard error on Python 3.12
**LOC:** `7028` (`COMPILER = RecipeCompiler()`), `5872` (`self._lock = asyncio.Lock()`)

**Problem:**  
`COMPILER` is instantiated at module level. `RecipeCompiler.__init__()` calls `asyncio.Lock()`. Creating asyncio primitives outside a running event loop is deprecated in Python 3.10 and raises `DeprecationWarning`; it is a hard `RuntimeError` in Python 3.12+.

```python
# Replace module-level singleton with lazy getter:
_COMPILER: Optional[RecipeCompiler] = None

def get_compiler() -> RecipeCompiler:
    global _COMPILER
    if _COMPILER is None:
        _COMPILER = RecipeCompiler()
    return _COMPILER

# Move asyncio.Lock creation into initialize() instead of __init__:
def __init__(self) -> None:
    ...
    self._lock: Optional[asyncio.Lock] = None   # placeholder

async def initialize(self, bus=None) -> None:
    if self._lock is None:
        self._lock = asyncio.Lock()             # created inside running loop
    async with self._lock:
        ...
```

---

### N4 — `THETA_WLP_MIN` imported but never used in logic
**LOC:** `159`

**Problem:**  
`THETA_WLP_MIN` (value `0.50` from `contracts.py`) is imported and referenced only in a comment at `198`. No conditional uses it. Its presence implies the contracts threshold is being respected; it isn't. Misleading.

```python
# Remove from import block at line 159:
# THETA_WLP_MIN,    ← delete this line
# The comment at 198 explaining the deviation is sufficient; keep it.
```

---

## 🔵 Low

---

### #12 — `id`+`class`+`attr` selector misclassified as `ATTRIBUTE_SELECTOR`
**LOC:** `862–878`

**Problem:**  
The classification ladder at `871` reaches `ATTRIBUTE_SELECTOR` only after checking `id_name` (no class, no tag) and `class_name` (no id). But a selector like `div#hero[data-active]` has both `id_name` and `attr_name`. It falls through to the `elif id_name and class_name` check at `873` and becomes `COMPOUND_SELECTOR`, which is correct — but a selector with `id_name` and `attr_name` (no class) hits `ATTRIBUTE_SELECTOR` at `872` instead of `COMPOUND_SELECTOR`. The `attr_name` check needs to account for co-present id/class.

```python
# Replace the attr_name branch:
elif attr_name:
    if id_name or class_name or tag:
        kind = SelectorKind.COMPOUND_SELECTOR   # has other components too
    else:
        kind = SelectorKind.ATTRIBUTE_SELECTOR  # pure attribute selector
```

---

### #13 — Dead `cmd.startswith("#")` check in sanitizer
**LOC:** `5690–5691`

**Problem:**  
Lines starting with `#` are already skipped at `5674` (`stripped.startswith("#")`). By the time execution reaches `5690`, `cmd` (first token of a non-comment, non-empty line) cannot start with `#`. The check is unreachable dead code that misleads readers into thinking shebangs can slip through to this point.

```python
# Delete lines 5690–5691:
# if cmd.startswith("#"):
#     continue
```

---

### #14 — mmap silent miss for topology class names > 64 bytes
**LOC:** `5798`

**Problem:**  
`.ljust(64, b"\x00")` truncates names longer than 64 bytes silently. The padded bytes are compared against the stored 64-byte field at `5803`. A name that encodes to > 64 bytes will never match any record — returns `None` with no log. Callers see a missing phase and default to `LEARNS` with no indication of why.

```python
# Add guard before the mmap scan:
encoded = topology_class.encode("utf-8")
if len(encoded) > 64:
    log.warning(
        "topology_class_name_exceeds_mmap_field",
        topology_class=topology_class,
        byte_length=len(encoded),
    )
    return None
class_bytes = encoded.ljust(64, b"\x00")
```

---

### #15 — Redundant `is_negation = True` assignment
**LOC:** `801`

**Problem:**  
`is_negation` is set to `True` at `801`, then immediately the function returns at `806–818` with `is_negation=True` hardcoded in the `ParsedSelector` constructor. The variable set at `801` is never read — the constructor argument at `817` is the operative value.

```python
# Delete line 801:
# is_negation = True  # noqa   ← remove

# The return at 817 already passes is_negation=True explicitly.
# is_negation = False at 798 remains correct as the default for the non-negation path.
```

---

## Already Fixed This Session

| # | What | How |
|---|------|-----|
| `_loop` | Missing from `__slots__` | Add `"_loop"` to `__slots__` tuple at line `5848` |
| `_trigger_recompile` | Wrong placement + wrong attr name | Placed in Internal Helpers; `_last_compile_time` → `_compile_timestamps` |

---

## N6 — Dead import: `RecipeRegistryEntry`
**LOC:** `163`

Imported but never instantiated anywhere in the file. Once S6 is fixed and `register_recipe()` is wired in, `RecipeRegistryEntry` may be used internally by the registry — but the parser itself doesn't need to import it directly. Remove unless the fix introduces a direct usage.

```python
# Remove from contracts import block:
# RecipeRegistryEntry,   ← delete
```
