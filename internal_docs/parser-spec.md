# `topology/parser.py` — Build Specification
**AXIOM Internal // Do Not Surface**

**File:** `tag/topology/parser.py`
**Built entirely by:** Opus  
**Role in AXIOM:** Recipe compiler. The intelligence layer between structural understanding and executable extraction.  
**Written:** After `topology/classifier.py` is complete and tested.

---

## What It Is — The Real Explanation

`parser.py` is not a lookup table. It is not a template engine. It is not a recipe library with a selection mechanism.

It is a **structural compiler**.

The WLP (`world_model/latent_parser.py`) produces a `ZoneMap` — a structural description of where signal lives on a given topology class. CSS selectors, node classifications, structural roles, extraction hints. This is the WLP's street-level map: "on SAAS_DOCS pages, signal is in `.main-content`, code is in `pre code`, navigation is noise, the cookie banner is noise, the sidebar is noise."

`parser.py` receives that ZoneMap and **translates structural descriptions into shell primitives**. It does not have a library of recipes to choose from. It has ~20 translation rules and a compiler that applies them to whatever ZoneMap it receives.

The result: a working `grep`/`sed`/`awk` pipeline written to `signal_kernel/recipes/compiler_generated/` that the signal kernel can execute in milliseconds against any page of that topology class.

**The genuinely important thing:** A topology subclass can appear that has never existed before — `discover_signal_zones()` auto-creates a ZoneMap from structural signals, emits `ZoneMapUpdatedEvent`, and the parser compiles a working recipe for a page structure that did not exist when the compiler was written. This is structural compilation from first principles, not pattern matching against known templates.

The hardcoded recipes in `signal_kernel/recipes/hardcoded/` exist for the five most common classes (bootstrap and performance). The compiler-generated recipes handle everything else, including new subclasses, variants, and intent-conditioned variants. Over time, as the WLP's zone maps improve, the compiler re-emits better recipes that replace older ones — automatically, without human intervention.

---

## What It Is Not

| parser.py does NOT | That belongs to |
|---|---|
| Fetch pages | `phantom.py` / `fetcher.py` |
| Classify pages | `topology/classifier.py` |
| Produce ZoneMaps | `world_model/latent_parser.py` |
| Execute recipes | `signal_kernel/pipeline.py` |
| Validate signal quality post-execution | `topology/surprise_detector.py` |
| Write to `topology_router.pt` | `index_daemon.py` only |
| Use an LLM to strip content | Never. Shell primitives only. |
| Read recipe files it wrote | Write-only. Registry owns reads. |
| Know what a query means | That is intent routing, upstream |

---

## File Dependencies

### Direct imports — must exist before parser.py is written

```
tag/
├── contracts.py                ← ZoneMap, Zone, RecipeMount, TopologyClass,
│                                 ZoneMapUpdatedEvent, RecipeCompiledEvent,
│                                 RecipeCompilationFailedEvent, IntentVector,
│                                 ExtractionStrategy, NodeType, KernelOutput
├── exceptions.py               ← RecipeCompilationError, ZoneMapInvalidError,
│                                 RecipeValidationError, StoreWriteError
├── crawler_bus.py              ← event subscription + emission
│
├── world_model/
│   └── latent_parser.py        ← source of ZoneMap. parser.py is the consumer.
│                                 ZoneMap schema must be stable before parser.py is written.
│
├── topology/
│   └── classifier.py           ← PARENT_CLASS_MAP, TopologyClass enum, class constants
│
└── signal_kernel/
    └── recipes/
        ├── validator.py        ← validator.check() called on every compiled recipe
        │                         before it is written to disk. non-negotiable.
        └── registry.py         ← registration of compiled recipe as RecipeMount
```

### Store files — must be initialized before runtime

```
tag/store/
├── recipe_registry.mmap        ← mmap. compiled recipes stored here.
│                                 parser.py writes via staging + atomic rename.
├── phase_states.mmap           ← read-only in parser.py.
│                                 phase awareness conditions recipe aggressiveness.
└── recipes/compiler_generated/ ← shell files written here.
                                  naming convention: {topology_class}.sh
                                  intent variants: {topology_class}_{intent}.sh
```

### Written output locations

```
signal_kernel/recipes/compiler_generated/
├── NEWS_ARTICLE.sh
├── NEWS_ARTICLE_recovery.sh          ← intent variant
├── SAAS_DOCS.sh
├── SAAS_DOCS_api_reference.sh        ← intent variant
├── SAAS_DOCS_recovery_codes.sh       ← intent variant
├── SAAS_DOCS_pricing.sh              ← intent variant
├── FORUM_THREAD.sh
├── WIKIPEDIA_ARTICLE.sh
├── ...
└── {any_new_subclass}.sh             ← auto-generated when new topology appears
```

---

## Input Contract — ZoneMap

The `ZoneMap` is produced by `latent_parser.py` and emitted via `ZoneMapUpdatedEvent`. This is the compiler's only input data source for structural knowledge.

```python
@dataclass(frozen=True)
class Zone:
    selector: str                            # CSS selector string
    node_type: NodeType                      # SIGNAL | NOISE | AMBIGUOUS
    structural_role: str                     # human-readable: "main content", "code block", "navigation"
    extraction_strategy: ExtractionStrategy  # ZONE_EXTRACT | ATTRIBUTE_EXTRACT | ENVELOPE_EXTRACT
    weight: float                            # [0.0, 1.0]. confidence WLP assigns to this zone.
    attributes: Optional[List[str]]          # for ATTRIBUTE_EXTRACT: which data-* attributes carry signal
    json_path: Optional[str]                 # for ENVELOPE_EXTRACT: dot-path into JSON structure
    child_noise_selectors: List[str]         # within a SIGNAL zone, these sub-selectors are noise
    depth_limit: Optional[int]               # for nested HTML: max nesting depth to traverse

@dataclass(frozen=True)
class ZoneMap:
    topology_class: str                      # one of 18 topology class strings
    zones: List[Zone]                        # ordered by weight descending
    confidence: float                        # [0.0, 1.0]. WLP confidence in this ZoneMap.
    intent_hints: Optional[Dict[str, List[str]]]  # intent → list of zone selectors to prioritize
    version: int                             # monotonically incrementing. parser uses to skip stale events.
    source_domain: Optional[str]             # domain that generated this map, if domain-specific
```

The ZoneMap contains everything the compiler needs. There is no other data source for structural decisions.

---

## Output Contract — RecipeMount

Every compiled recipe is registered as a `RecipeMount` in `recipe_registry.mmap`.

```python
@dataclass(frozen=True)
class RecipeMount:
    topology_class: str          # which class this recipe serves
    intent: Optional[str]        # None = base recipe. str = intent-conditioned variant.
    recipe_path: str             # absolute path to compiled .sh file
    checksum: str                # SHA-256 of recipe file content
    compiled_at: float           # unix timestamp
    zone_map_version: int        # version of the ZoneMap this was compiled from
    strategy: ExtractionStrategy # which compilation strategy was used
    phase_at_compile: str        # LEARNS | PREDICTS | KNOWS. recipe is phase-conditioned.
```

---

## The Three Compilation Strategies

Determined by `Zone.extraction_strategy` on the dominant signal zones.

### ZONE_EXTRACT
Signal lives in defined HTML zones identified by CSS selectors.

Used for: `NEWS_ARTICLE`, `SAAS_DOCS`, `BLOG_POST`, `WIKIPEDIA_ARTICLE`, `FORUM_THREAD`, `LANDING_PAGE`

The compiled recipe:
1. Locates zone boundaries (start/end HTML markers from selector)
2. Extracts content within zone boundaries
3. Strips child noise selectors within the signal zone
4. Strips remaining HTML tags from extracted content
5. Normalizes whitespace

```bash
# Example compiled output for SAAS_DOCS:
sed -n '/<div[^>]*class="[^"]*main-content[^"]*"/,/<\/div>/p' \
| sed '/<nav[^>]*>/,/<\/nav>/d' \
| sed '/<div[^>]*class="[^"]*sidebar[^"]*"/,/<\/div>/d' \
| sed '/<div[^>]*class="[^"]*cookie[^"]*"/,/<\/div>/d' \
| sed 's/<[^>]*>//g' \
| tr -s ' \t\n' '\n' \
| grep -v '^[[:space:]]*$'
```

The awk path handles depth-aware extraction for nested structures (FORUM_THREAD, WIKIPEDIA_ARTICLE subsections). This can be a 20-30 line awk program that tracks HTML nesting depth as a counter, treating depth > N as escape from the signal zone.

### ATTRIBUTE_EXTRACT
Signal lives in `data-*` HTML attributes, not in tag content.

Used for: `ECOMMERCE_PRODUCT`, `ECOMMERCE_PRODUCT_VARIANT`, `JSON_LD_STRUCTURED`

Product name, price, SKU, availability — these live in structured attributes, not visible text. The compiled recipe uses awk regex matching against attribute patterns.

```bash
# Example compiled output for ECOMMERCE_PRODUCT:
awk '{
    if (match($0, /data-product-name="([^"]*)"/, a)) print "NAME: " a[1]
    if (match($0, /data-price="([^"]*)"/, a))        print "PRICE: " a[1]
    if (match($0, /data-sku="([^"]*)"/, a))          print "SKU: " a[1]
    if (match($0, /data-availability="([^"]*)"/, a)) print "AVAIL: " a[1]
}' \
| grep -v '^[[:space:]]*$'
```

For `JSON_LD_STRUCTURED`: signal is in the `<script type="application/ld+json">` block. The compiler emits an awk program that extracts the JSON block verbatim, then a second awk pass that extracts the relevant fields from the JSON structure using pattern matching on key names.

### ENVELOPE_EXTRACT
Signal lives inside a JSON response envelope at a known path.

Used for: `REST_API_JSON`, `REST_API_JSON_PAGINATED`

The entire response is JSON. Signal is at `zone.json_path` (e.g. `results.items`, `data.entries`). The compiler emits an awk program that traverses the JSON structure using pattern matching — not a full JSON parser (no dependencies), but a structural extractor that knows the depth and key path.

```bash
# Example compiled output for REST_API_JSON with json_path="results.items":
awk '
BEGIN { depth=0; in_results=0; in_items=0 }
/"results"[[:space:]]*:/ { in_results=1 }
in_results && /"items"[[:space:]]*:/ { in_items=1 }
in_items { print }
in_items && /\]/ { exit }
' \
| sed 's/^[[:space:]]*//' \
| grep -v '^[{}[\],]*$'
```

---

## The Translation Rules — All ~20

This is the compiler's core. Each rule maps a structural description (from ZoneMap) to one or more shell primitives. Rules compose — a single Zone can trigger 3–5 rules in sequence.

| # | Structural description | Shell primitive emitted |
|---|---|---|
| 1 | CSS class selector → signal zone | `sed -n '/<[^>]*class="[^"]*CLASSNAME[^"]*"/,/<\/[A-Z][A-Z]*>/p'` |
| 2 | CSS id selector → signal zone | `grep -A 500 'id="IDNAME"' \| head -500` with depth counter in awk |
| 3 | Tag boundary extraction | `sed -n '/<TAG/,/<\/TAG>/p'` |
| 4 | Strip tag + content (noise removal) | `sed '/<TAG[^>]*>/,/<\/TAG>/d'` |
| 5 | Strip all HTML tags (text extraction) | `sed 's/<[^>]*>//g'` |
| 6 | Strip inline attributes (clean tag) | `sed 's/ [a-z-]*="[^"]*"//g'` |
| 7 | Code block extraction (`<pre>`, `<code>`) | `sed -n '/<pre/,/<\/pre>/p'` then nested `sed -n '/<code/,/<\/code>/p'` |
| 8 | Ordered/unordered list extraction | `sed -n '/<[ou]l/,/<\/[ou]l>/p'` |
| 9 | Paragraph text extraction | `sed -n '/<p[> ]/,/<\/p>/p'` then strip tags |
| 10 | Heading extraction (h1–h6) | `grep -oP '(?<=<h[1-6][^>]*>)[^<]+'` |
| 11 | data-* attribute extraction | `awk '{match($0, /data-KEY="([^"]*)"/, a); if(a[1]) print a[1]}'` |
| 12 | JSON field extraction | `awk '/"KEY"[[:space:]]*:/{match($0, /"KEY"[[:space:]]*:[[:space:]]*"([^"]*)"/, a); print a[1]}'` |
| 13 | JSON array traversal | awk state machine tracking `[` / `]` depth with key path context |
| 14 | Normalize whitespace | `tr -s ' \t\n' '\n'` |
| 15 | Deduplicate lines | `awk '!seen[$0]++'` (order-preserving, not sort-based) |
| 16 | Strip empty lines | `grep -v '^[[:space:]]*$'` |
| 17 | URL extraction from hrefs | `grep -oP 'href="[^"]*"' \| cut -d'"' -f2` |
| 18 | Price pattern extraction | `grep -oP '[$€£]\s*[\d,]+\.?\d*'` |
| 19 | Table row extraction | awk program tracking `<tr>` / `<td>` boundaries, tab-separated output |
| 20 | Depth-limited nesting | awk with `depth` counter: increments on `<`, decrements on `</`, extracts only when `depth <= N` |

Rules compose in the order the compiler determines from the ZoneMap. Rule 20 (depth-limited nesting) is always the last rule applied when `zone.depth_limit` is set.

---

## awk as a Complete Programming Language

This is not a secondary concern. It is central to the compiler's power.

For simple topologies, the compiled recipe is a `grep | sed | awk` one-liner. For complex topologies (`FORUM_THREAD`, `WIKIPEDIA_ARTICLE`, `SAAS_DOCS_WITH_CODE`), the compiler emits a multi-function awk program. awk is Turing-complete. The compiler treats it as a first-class compilation target, not a cleanup utility.

**What awk programs handle that one-liners cannot:**

1. **HTML nesting depth** — track `depth` as an integer counter. Increment on any opening tag, decrement on any closing tag. Extract only content where `depth == SIGNAL_DEPTH`. This correctly handles nested `<div>` structures that `sed -n '/<div/,/<\/div>/p'` gets wrong (closes on the first `</div>`, not the matching one).

2. **Multi-zone stitching** — FORUM_THREAD has: original post, accepted answer, top-voted answers, code blocks within answers. A single awk program can track which zone it's in (`in_post`, `in_accepted`, `in_answer`) and emit labeled sections.

3. **State machine extraction** — REST_API_JSON_PAGINATED: extract `next_cursor` field from envelope, then extract `items` array. One awk pass, two pieces of state.

4. **Conditional extraction** — SAAS_DOCS_WITH_CODE: if a paragraph is immediately followed by a code block, keep both. If a code block is isolated (not preceded by explanation text), skip it. awk tracks `prev_was_paragraph` as boolean state.

Example of a complex compiled awk program for `WIKIPEDIA_ARTICLE`:

```awk
#!/usr/bin/awk -f
# compiled by topology/parser.py from ZoneMap version 14
# topology: WIKIPEDIA_ARTICLE  strategy: ZONE_EXTRACT
BEGIN {
    in_content = 0
    in_infobox = 0
    in_references = 0
    depth = 0
    signal_depth = -1
}
/<div[^>]*id="mw-content-text"/ {
    in_content = 1
    signal_depth = depth
}
/<table[^>]*class="[^"]*infobox[^"]*"/ { in_infobox = 1 }
/<\/table>/ && in_infobox { in_infobox = 0; next }
/<div[^>]*class="[^"]*references[^"]*"/ { in_references = 1 }
/<\/div>/ && in_references { in_references = 0; next }
in_content && !in_infobox && !in_references {
    gsub(/<[^>]*>/, "")
    if (/[^[:space:]]/) print
}
/<\/div>/ && in_content && depth == signal_depth { in_content = 0 }
{ depth += gsub(/<[^\/][^>]*>/, "") - gsub(/<\/[^>]*>/, "") }
```

The compiler generates this from translation rules 1, 3, 4, 5, 16, and 20. It does not hardcode this program. It composes it.

---

## Intent Conditioning

Same URL. Same topology class. Different intent vector. Different recipe.

This is what makes AXIOM not just fast but genuinely intelligent about what it keeps.

### Intent vectors

```python
class IntentVector(str, Enum):
    API_REFERENCE    = "api_reference"
    RECOVERY_CODES   = "recovery_codes"
    PRICING          = "pricing"
    TUTORIAL         = "tutorial"
    CHANGELOG        = "changelog"
    CONCEPTUAL       = "conceptual"
    TROUBLESHOOTING  = "troubleshooting"
```

### How intent conditioning works

The `ZoneMap.intent_hints` field maps intent strings to lists of zone selectors that should be prioritized for that intent. The compiler generates a base recipe (all SIGNAL zones) and one variant per intent hint.

```python
# ZoneMap.intent_hints example for SAAS_DOCS:
{
    "recovery_codes": [".support-content", ".backup-codes", "ol.recovery-list"],
    "api_reference":  [".api-endpoint", "pre.code-sample", ".parameter-table"],
    "pricing":        [".pricing-table", ".tier-comparison", ".plan-features"],
}
```

The intent-conditioned recipe narrows zone selection to only the prioritized selectors. Same stripping logic, different zone boundaries.

**File naming convention:**
```
SAAS_DOCS.sh                   ← base recipe (no intent)
SAAS_DOCS_recovery_codes.sh    ← intent variant
SAAS_DOCS_api_reference.sh     ← intent variant
SAAS_DOCS_pricing.sh           ← intent variant
```

**Recipe selection at query time** (not in parser.py — this is how downstream uses it):

`phantom.py` receives a query with an optional intent vector. Passes to registry: `registry.get(topology_class, intent)`. Registry returns the intent-conditioned recipe if it exists, falls back to base recipe.

**The Twilio example:** A `SAAS_DOCS` page about recovery codes. Structural signature: heading + ordered list + monospace code blocks. Without intent conditioning, the base recipe keeps the whole main content area — which includes navigation headers, unrelated documentation links, cookie notices that survived the outer strip, and finally the recovery codes. With `SAAS_DOCS_recovery_codes.sh`: the recipe targets specifically the structural pattern of ordered list + monospace blocks, discards everything else. 1.6MB page → 1.5KB of exactly the recovery codes.

---

## Recursive Fallback — PARENT_CLASS_MAP

If `ZoneMap.confidence < THETA_WLP_MIN`, the compiler does not refuse to compile. It walks up `PARENT_CLASS_MAP` until it finds a confident ZoneMap or hits `GENERIC_HTML`.

```python
THETA_WLP_MIN = 0.70  # below this, recurse to parent

PARENT_CLASS_MAP = {
    "NEWS_ARTICLE_PAYWALLED":    "NEWS_ARTICLE",
    "SAAS_DOCS_VERSIONED":       "SAAS_DOCS",
    "SAAS_DOCS_WITH_CODE":       "SAAS_DOCS",
    "REST_API_JSON_PAGINATED":   "REST_API_JSON",
    "ECOMMERCE_PRODUCT_VARIANT": "ECOMMERCE_PRODUCT",
    "FORUM_THREAD":              "BLOG_POST",      # fallback: treat like blog prose
    "BLOG_POST":                 "NEWS_ARTICLE",   # fallback: treat like news prose
    "WIKIPEDIA_ARTICLE":         "NEWS_ARTICLE",   # fallback: structured prose
    "LANDING_PAGE":              "GENERIC_HTML",
    "JSON_LD_STRUCTURED":        "GENERIC_HTML",
    "*":                         "GENERIC_HTML",   # terminal
}
```

`GENERIC_HTML` always has a ZoneMap (generated conservatively: keep `<body>` content, strip `<nav>`, `<header>`, `<footer>`, `<script>`, `<style>`). It is the guaranteed terminal. The compiler never fails to produce a recipe.

The `RecipeMount.fallback_chain` field records which classes were visited before settling.

---

## Phase Conditioning

The compiler reads `phase_states.mmap` before compiling any recipe.

```python
def _read_phase(self, topology_class: str) -> Optional[str]:
    # returns "LEARNS" | "PREDICTS" | "KNOWS" | None
```

Phase changes recipe aggressiveness:

| Phase | Recipe behavior | Why |
|---|---|---|
| `LEARNS` | Conservative. Wider extraction bounds. Keep AMBIGUOUS zones. | Domain is new. Prefer recall over precision. Surprise detector catches overextraction. |
| `PREDICTS` | Balanced. Strict SIGNAL zones only. AMBIGUOUS zones excluded. | Growing confidence. Narrower bounds reduce noise sent to Haiku. |
| `KNOWS` | Aggressive. Minimum viable signal only. Zero tolerance for noise. | High confidence. Extraction is surgical. Index quality is maximized. |

Phase is embedded in the compiled recipe via a comment header and recorded in `RecipeMount.phase_at_compile`. When phase transitions occur (`index_daemon` writes to `phase_states.mmap`), `store_watchdog.py` detects the change and triggers recompilation of all recipes for that topology class.

---

## Feedback Injection

```python
def _inject_feedback(self, zone_map: ZoneMap, last_output: Optional[KernelOutput]) -> ZoneMap:
```

The signal kernel produces `KernelOutput` after executing a recipe. This includes compression ratio, noise fragments detected, and whether `surprise_detector` fired. When the parser recompiles a recipe (triggered by `ZoneMapUpdatedEvent` with incremented version), it receives the last `KernelOutput` for that topology class and adjusts zone boundaries before compilation.

**Feedback rules:**
- If `last_output.noise_ratio > NOISE_THRESHOLD`: tighten zone boundaries. Shrink `child_noise_selectors` matching, exclude AMBIGUOUS zones.
- If `last_output.surprise_fired`: loosen zone boundaries. The recipe missed signal. Expand to include adjacent selectors.
- If `last_output.compression_ratio < MIN_COMPRESSION`: recipe is keeping too much. Tighten.
- If `last_output.compression_ratio > MAX_COMPRESSION`: recipe stripped too aggressively. Loosen.

This closes the loop: WLP → parser → kernel → surprise_detector → index_daemon → WLP → parser.

---

## Compilation Algorithm — Step by Step

```
receive ZoneMapUpdatedEvent
  └── if event.version <= last_compiled_version[topology_class]: discard (stale)

read phase from phase_states.mmap
  └── _read_phase(topology_class) → phase: str | None

load last KernelOutput for topology_class (from registry metadata)
  └── may be None if first compilation

apply feedback injection
  └── _inject_feedback(zone_map, last_output) → adjusted ZoneMap

determine compilation strategy
  └── dominant_strategy = mode(zone.extraction_strategy for zone in zones if zone.node_type == SIGNAL)
  └── if mixed strategies: ZONE_EXTRACT wins over ATTRIBUTE_EXTRACT wins over ENVELOPE_EXTRACT

select translation rules
  └── for each SIGNAL zone in adjusted ZoneMap (ordered by weight descending):
      └── for each structural property of zone:
          └── lookup rule in TRANSLATION_TABLE
          └── emit rule into compilation pipeline

compose pipeline
  └── ZONE_EXTRACT: sed chain + optional awk program
  └── ATTRIBUTE_EXTRACT: awk regex program
  └── ENVELOPE_EXTRACT: awk state machine

apply phase conditioning
  └── KNOWS: append aggressive post-filter (strip lines < MIN_SIGNAL_LENGTH)
  └── LEARNS: omit aggressive post-filter

compile intent variants
  └── for each intent in zone_map.intent_hints:
      └── narrow zone selection to intent-prioritized selectors
      └── compile separate recipe using same rules

validate all compiled recipes
  └── validator.check(recipe_path, test_fixture) for base recipe
  └── validator.check(recipe_path, test_fixture) for each intent variant
  └── if any validation fails: raise RecipeValidationError, do NOT write to disk

write base recipe
  └── staging path: {topology_class}.sh.staging
  └── write recipe content to staging
  └── SHA-256 hash of staging file
  └── os.rename(staging, final)  ← atomic

write intent variants (same staging protocol for each)

register all RecipeMounts in recipe_registry.mmap
  └── staging + atomic rename for mmap write

emit RecipeCompiledEvent to crawler_bus
  └── includes topology_class, intent_variants_compiled, zone_map_version
```

---

## Write Protocol — Non-Negotiable

Every file write uses staging + atomic rename. This is a system-wide invariant.

```python
def _write_recipe(self, recipe_content: str, topology_class: str, intent: Optional[str] = None) -> str:
    filename = f"{topology_class}.sh" if not intent else f"{topology_class}_{intent}.sh"
    staging = COMPILER_GENERATED_PATH / f"{filename}.staging"
    final   = COMPILER_GENERATED_PATH / filename

    staging.write_text(recipe_content, encoding="utf-8")
    checksum = hashlib.sha256(staging.read_bytes()).hexdigest()
    os.rename(staging, final)  # atomic on POSIX

    return checksum
```

Never write directly to the final path. Never skip the SHA-256. Never use `shutil.copy` (not atomic).

---

## Validator Integration

`validator.check()` runs on every compiled recipe before it is registered. Non-negotiable.

```python
# from signal_kernel/recipes/validator.py
def check(recipe_path: str, fixture_dir: str) -> ValidationResult:
    # dry-run: pipe test fixture through recipe
    # check: output is non-empty
    # check: output is < fixture input size (compression happened)
    # check: no shell injection in recipe (grep/sed/awk only, no subshell expansion)
    # check: recipe is syntactically valid shell
    # returns ValidationResult(passed: bool, noise_ratio: float, compression_ratio: float)
```

The validator runs the recipe against test fixtures in `signal_kernel/recipes/test_fixtures/{TOPOLOGY_CLASS}/`. Each topology class has 3–5 real HTML samples from the wild. The recipe must produce non-empty, compressed output against every fixture.

If `ValidationResult.passed == False`: the recipe is discarded. `RecipeCompilationFailedEvent` is emitted to the bus. The previous recipe for that topology class remains active. This is the graceful degradation path.

**Allowed commands** — validator enforces this whitelist:
```python
ALLOWED_RECIPE_COMMANDS = {
    "grep", "sed", "awk", "cat", "cut", "tr",
    "head", "tail", "sort", "uniq"
}
```

Any compiled recipe that contains a command outside this set fails validation immediately. No exceptions. The sanitizer does not use LLM, the compiler does not use curl, wget, or any network command, ever.

---

## Event Subscriptions and Emissions

```python
# subscribes
ZoneMapUpdatedEvent      # from world_model/latent_parser.py via crawler_bus
PhaseTransitionEvent     # from index_daemon via crawler_bus (triggers recompilation)

# emits
RecipeCompiledEvent      # on successful compilation + validation + write
RecipeCompilationFailedEvent  # on validation failure or write error
```

`RecipeCompiledEvent` payload:
```python
@dataclass(frozen=True)
class RecipeCompiledEvent:
    topology_class: str
    intent_variants: List[str]       # list of intent strings compiled
    zone_map_version: int
    strategy: ExtractionStrategy
    phase: str
    recipe_path: str
    checksum: str
    compiled_at: float
```

---

## Class Structure

```python
class RecipeCompiler:
    # public
    async def initialize(self) -> None
    async def handle_zone_map_updated(self, event: ZoneMapUpdatedEvent) -> None
    async def handle_phase_transition(self, event: PhaseTransitionEvent) -> None
    async def compile(self, zone_map: ZoneMap, phase: str) -> List[RecipeMount]

    # compilation
    def _determine_strategy(self, zone_map: ZoneMap) -> ExtractionStrategy
    def _select_rules(self, zone: Zone) -> List[TranslationRule]
    def _compose_zone_extract(self, zones: List[Zone], phase: str) -> str
    def _compose_attribute_extract(self, zones: List[Zone]) -> str
    def _compose_envelope_extract(self, zones: List[Zone]) -> str
    def _compose_awk_program(self, zones: List[Zone], depth_limit: Optional[int]) -> str
    def _compile_intent_variant(self, zone_map: ZoneMap, intent: str, phase: str) -> str

    # feedback
    def _inject_feedback(self, zone_map: ZoneMap, last_output: Optional[KernelOutput]) -> ZoneMap
    def _read_last_output(self, topology_class: str) -> Optional[KernelOutput]

    # phase
    def _read_phase(self, topology_class: str) -> Optional[str]

    # fallback
    def _walk_parent_map(self, topology_class: str) -> str
    def _get_zone_map_for_class(self, topology_class: str) -> Optional[ZoneMap]

    # write
    def _write_recipe(self, content: str, topology_class: str, intent: Optional[str]) -> str
    def _register_mount(self, mount: RecipeMount) -> None

    # internal state
    _last_compiled_version: Dict[str, int]     # topology_class → last ZoneMap version compiled
    _zone_map_cache: Dict[str, ZoneMap]         # topology_class → latest ZoneMap
    _phase_cache: Dict[str, str]                # topology_class → current phase
    _bus: CrawlerBus
    _validator: RecipeValidator
    _registry: RecipeRegistry
```

---

## Constants

```python
THETA_WLP_MIN          = 0.70   # below this, recurse to parent class
THETA_NOISE_TIGHTEN    = 0.30   # noise ratio above this: tighten zone bounds on next compile
THETA_COMPRESSION_MIN  = 0.50   # compression below 50% means recipe kept too much
THETA_COMPRESSION_MAX  = 0.995  # compression above 99.5% means recipe stripped too aggressively
MIN_SIGNAL_LENGTH      = 40     # chars. KNOWS phase only: lines shorter than this stripped.
MAX_RECIPE_LINES       = 200    # compiled recipe over 200 lines fails validation (complexity cap)

COMPILER_GENERATED_PATH = Path("signal_kernel/recipes/compiler_generated")
FIXTURE_PATH            = Path("signal_kernel/recipes/test_fixtures")
```

---

## Error Handling

```python
RecipeCompilationError   # raised when compilation logic fails (not validation failure)
ZoneMapInvalidError      # raised when ZoneMap is malformed (missing required fields)
RecipeValidationError    # raised when validator.check() returns passed=False
StoreWriteError          # raised when os.rename() fails (disk full, permissions)
```

On `RecipeValidationError` or `StoreWriteError`: emit `RecipeCompilationFailedEvent`, log full error, leave existing recipe in place. Do not crash. The system continues serving the previous recipe.

On `ZoneMapInvalidError`: emit `RecipeCompilationFailedEvent`, log malformed ZoneMap version for debugging. Do not recurse (the ZoneMap is the problem, not the topology class).

---

## Tests to Write (Guidance for Opus)

The test suite should cover:

1. **Strategy selection** — mixed ZoneMap with overlapping strategies selects correctly
2. **Translation rules** — each of the 20 rules individually produces valid shell primitives
3. **awk program generation** — depth-limited extraction correctly tracks nesting
4. **Intent conditioning** — base recipe and intent variant produce different outputs on same fixture
5. **Recursive fallback** — low-confidence ZoneMap walks PARENT_CLASS_MAP correctly
6. **Phase conditioning** — KNOWS recipe strips more aggressively than LEARNS recipe
7. **Feedback injection** — high noise_ratio in KernelOutput tightens zone bounds on recompile
8. **Stale event rejection** — ZoneMapUpdatedEvent with version <= last compiled is discarded
9. **Validation gate** — recipe that produces empty output on fixture fails and is not written
10. **Staging protocol** — write failure leaves previous recipe intact
11. **Allowed commands whitelist** — recipe containing `curl` fails validation
12. **Multi-zone stitching** — FORUM_THREAD fixtures produce labeled post + answer output

Each topology class needs at least one real-world HTML fixture in `signal_kernel/recipes/test_fixtures/`. These fixtures are used by both the validator and the test suite.

---

## Build Notes for Opus

This is a pure Opus file. No Sonnet paths. The entire compiler requires reasoning about representations — translating structural descriptions into executable programs — which is Opus-level work throughout.

The most important thing to get right is the awk program generator (`_compose_awk_program`). It is the ceiling of what the compiler can do. Simple topologies work fine with sed chains. Complex topologies (WIKIPEDIA_ARTICLE, FORUM_THREAD, SAAS_DOCS_WITH_CODE) require awk programs that track state. The generator must be a real composition engine, not a set of templates with string substitution.

The feedback injection loop is the second most important thing. Without it, the compiler compiles once and never improves. With it, every `ZoneMapUpdatedEvent` produces a recipe that is conditioned on actual kernel performance, not just structural description. This is what closes the learning loop at the compilation layer.

Intent conditioning is third. It transforms the compiler from an infrastructure component into an intelligence layer. The Twilio recovery codes example is the concrete proof: same topology class, radically different extraction behavior, driven by intent.

Write the compiler in this order:
1. Data structures + constants
2. `_determine_strategy()` + `_select_rules()` — the translation table
3. `_compose_zone_extract()` — ZONE_EXTRACT strategy, sed chains only first
4. `_compose_awk_program()` — the depth-aware awk generator
5. `_compose_attribute_extract()` — ATTRIBUTE_EXTRACT strategy
6. `_compose_envelope_extract()` — ENVELOPE_EXTRACT strategy
7. `_walk_parent_map()` + `_get_zone_map_for_class()` — fallback
8. `_read_phase()` — mmap read
9. `_inject_feedback()` — feedback loop
10. `_compile_intent_variant()` — intent conditioning
11. `_write_recipe()` + `_register_mount()` — write protocol
12. `compile()` — orchestrator (assembles everything above)
13. `handle_zone_map_updated()` + `handle_phase_transition()` — event handlers
14. `initialize()` — bus subscription setup
15. Tests

---

*AXIOM Internal // Do Not Surface*
*TAG not RAG. Structure as compiler input. Shell as compiler output. Intelligence in the translation.*
