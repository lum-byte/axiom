# tag/ — Topology Layer Developer Reference
**AXIOM Core Searching Algorithm — TAG Intelligence Layer**
**Classification: AXIOM INTERNAL // DO NOT SURFACE**
**Supersedes: readme-topologydev.md v1 (pre-bus architecture)**

---

## What Was Built. What Gets Built Next.

`signal_kernel/` is complete and production-validated:

```
Stripe API docs:   1,155,697 bytes  →  8,460 bytes   in 7.4ms   (99.27% reduction)
Twilio API docs:   1,621,723 bytes  →  1,571 bytes   in 9.0ms   (99.90% reduction)
350 tests passing.
pipeline.py warm container bug identified and fixed.
checkpoint_monitor.py written and validated.
Dockerfile + docker-compose.yml complete.
```

The kernel executes. It does not think. It runs whatever recipe it is given at C speed
and returns `KernelOutput`. It has proven the core premise: noise can be stripped at
99%+ reduction in milliseconds before any LLM sees a page.

Everything in `tag/` is what makes the kernel useful at scale. The topology layer is
what decides what topology class a page belongs to, compiles the recipe that extracts
from it, evaluates the quality of what came back, learns from every run, and gets
progressively faster and cheaper as it accumulates structural knowledge of the web.

The kernel was infrastructure. `tag/` is intelligence.

---

## The Architecture Model

AXIOM's `tag/` layer is built around a **crawler bus** — a typed event system where
every intelligent component subscribes to events it cares about and acts independently.
No component calls another component directly. Every component communicates via the bus.

This is not LangGraph. This is not a DAG. The Semantic Intent Graph is custom-built
because AXIOM's intelligence is not linear. Phase-aware nodes, recursive topology
resolution, mid-run reclassification — none of these fit a DAG framework's assumptions.
The bus is the correct abstraction.

```
CRAWLER BUS — typed event flow

fetcher.py
    emits RawFetchEvent
          │
          ▼
alpine_strip/offline_pipeline.py        (subscribes to RawFetchEvent)
    emits CleanSignalEvent
          │
    ┌─────┴──────────────────────────────────────────────────┐
    │                                                         │
    ▼                                                         ▼
world_model/latent_parser.py            world_model/latent_model.py
(subscribes to CleanSignalEvent)        (subscribes to DomainTopologyEvent)
    emits ZoneMapUpdatedEvent               emits TraversalPolicy updates → MFT
          │
          ▼
topology/parser.py
(subscribes to ZoneMapUpdatedEvent)
    recompiles recipe
    writes to signal_kernel/recipes/compiler_generated/
          │
          ▼
topology/surprise_detector.py
    emits SurpriseEvent
          │
          ▼
index_daemon.py
(subscribes to SurpriseEvent, CleanSignalEvent, ZoneMapUpdatedEvent,
 DomainTopologyEvent — subscribes to everything)
    gradient steps on topology_router.pt
    phase state transitions
    hivemind share on new structural primitives
```

---

## How signal_kernel/ Fits Into This

The kernel is not a layer in the bus. It is a **tool** that multiple layers use.

```
ONLINE PATH (query time — critical path):
    interface.py receives DaemonRequest
    → classifier.py classifies URL
    → wlm.query() parallel with wlp.query()
    → phantom.fetch() fetches page
    → registry.get_recipe() resolves recipe
    → pipeline.execute() runs kernel
    → sanitizer.sanitize() cleans output
    → se_separator.split() splits prose/code
    → Haiku x2 parallel extractions
    → surprise_detector.evaluate()
    → index_daemon.process() (async, fire and forget via bus)
    → DaemonResponse returned

OFFLINE PATH (crawl time — background):
    fetcher.py fetches URL batch
    → alpine_strip/offline_pipeline.py
       wraps signal_kernel/pipeline.py
       same warm Alpine container
       same grep recipes
       emits CleanSignalEvent to bus
    → wlp, ingestion, index_daemon subscribe and process
```

The kernel appears in both paths. The container that strips Stripe docs in 7.4ms
at query time is the same container that batch-processes Wikipedia at 80K URLs/sec
during crawl. One image. Two jobs. Zero additional infrastructure.

**What signal_kernel/ provides to tag/ — the exact imports:**

```python
from signal_kernel.contracts import (
    KernelInput,        # what goes into pipeline.execute()
    KernelOutput,       # what comes back
    RecipeMount,        # resolved recipe path + hash + topology class
    ExtractionQuality,  # quality scores from feedback.py
    CheckpointHealth,   # from checkpoint_monitor.py
)

from signal_kernel.exceptions import (
    KernelException,            # base — all topology exceptions inherit this
    SubprocessTimeout,          # grep pipeline timed out
    RecipeMountError,           # recipe missing or invalid
    RecipeInjectionAttempt,     # validator caught shell injection
    ContainerSpawnError,        # Alpine container failed to start
    StdinEncodingError,         # raw_content encoding failed
    CheckpointCorruptionError,  # hash mismatch on checkpoint file
)

from signal_kernel.pipeline import execute_sync, initialize, get_pipeline_health
from signal_kernel.feedback import process as feedback_process
from signal_kernel.recipes.registry import registry
from signal_kernel.recipes.validator import check
from signal_kernel.checkpoint.checkpoint_monitor import restore, health
```

The topology layer does not reimplement anything the kernel already provides.
It extends. It wraps. It consumes `KernelOutput` and adds intelligence on top.

---

## Contracts Additions Required — signal_kernel/contracts.py

Before any topology file is written, add these to the existing contracts.py.
They live there — not in a separate topology contracts file — because every
layer imports from one place. One file. No hunting.

```python
# ── Topology primitive types ───────────────────────────────────────────────
from typing import NewType

TopologyClassStr  = NewType("TopologyClassStr", str)
PhaseInt          = NewType("PhaseInt", int)
ConfidenceFloat   = NewType("ConfidenceFloat", float)
SurpriseFloat     = NewType("SurpriseFloat", float)

# ── Phase constants ────────────────────────────────────────────────────────
PHASE_I   = PhaseInt(1)   # learns   — live traversal only, no compiled policy
PHASE_II  = PhaseInt(2)   # predicts — world model active, MFT building
PHASE_III = PhaseInt(3)   # knows    — compiled policy, direct routing

# ── Confidence thresholds ──────────────────────────────────────────────────
THETA_SURPRISE_DEFAULT     = SurpriseFloat(0.35)   # dissolve threshold
THETA_CONFIDENCE_II        = ConfidenceFloat(0.70)  # Phase I  → II transition
THETA_CONFIDENCE_III       = ConfidenceFloat(0.90)  # Phase II → III transition
THETA_CLASSIFY_CONFIDENT   = ConfidenceFloat(0.75)  # skip ML path if above
THETA_CLASSIFY_FALLBACK    = ConfidenceFloat(0.40)  # GENERIC_HTML if below

# ── Bus event types ────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RawFetchEvent:
    url:           str
    raw_bytes:     bytes
    status_code:   int
    headers:       Dict[str, str]
    fetch_latency: float
    is_robots_txt: bool
    is_sitemap:    bool
    run_id:        str

@dataclass(frozen=True)
class CleanSignalEvent:
    url:              str
    clean_signal:     str
    topology_class:   str
    token_reduction:  float
    signal_density:   float
    extraction_empty: bool
    run_id:           str

@dataclass(frozen=True)
class DomainTopologyEvent:
    domain:     str
    domain_map: "DomainMap"

@dataclass(frozen=True)
class ZoneMapUpdatedEvent:
    topology_class: str
    new_zone_map:   "ZoneMap"

@dataclass(frozen=True)
class SurpriseEvent:
    topology_class:       str
    surprise_score:       float
    theta_surprise:       float
    dissolve_triggered:   bool
    contributing_signals: Dict[str, float]
    current_phase:        int
    run_id:               str
    timestamp:            str

# ── Topology classifier output ─────────────────────────────────────────────
@dataclass(frozen=True)
class TopologyClassification:
    topology_class:      str
    confidence:          float
    classification_path: str    # "domain"|"url"|"header"|"window"|"model"
    signals_used:        Dict[str, str]
    latency_ms:          float
    run_id:              str

@dataclass(frozen=True)
class ClassificationWindow:
    url:            str
    headers:        Dict[str, str]
    content_prefix: str         # first 4KB only — never full page
    content_type:   str         # "html"|"json"|"unknown"

# ── World model outputs ────────────────────────────────────────────────────
@dataclass(frozen=True)
class ZoneMap:
    topology_class: str
    signal_zones:   List[str]   # HTML tag patterns / CSS selectors
    noise_zones:    List[str]   # patterns to exclude
    strategy:       str         # "zone_extract"|"attribute_extract"|"envelope_extract"
    confidence:     float
    version:        int

@dataclass(frozen=True)
class TraversalPolicy:
    topology_class:      str
    depth:               int
    render_mode:         str    # "static"|"headless"
    requests_per_second: float
    retry_budget:        int
    timeout_ms:          int
    confidence:          float

@dataclass(frozen=True)
class FrictionForecast:
    topology_class:            str
    cloudflare_probability:    float
    paywall_probability:       float
    rate_limit_probability:    float
    auth_redirect_probability: float
    mitigation_strategy:       str

@dataclass(frozen=True)
class WLMResponse:
    traversal_policy:  TraversalPolicy
    friction_forecast: FrictionForecast
    source_priority:   List[str]
    world_confidence:  float

# ── Sanitizer + SE separator outputs ──────────────────────────────────────
@dataclass(frozen=True)
class SanitizedSignal:
    text:                 str
    raw_byte_count:       int
    sanitized_byte_count: int
    sanitized_empty:      bool
    operations_applied:   List[str]
    latency_ms:           float
    run_id:               str

@dataclass(frozen=True)
class SeparatedSignal:
    prose_signal:   str
    code_signal:    str
    has_prose:      bool
    has_code:       bool
    topology_class: str
    run_id:         str

# ── Interface boundary ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class DaemonRequest:
    query:            str
    intent_vector:    List[float]
    candidate_urls:   List[str]
    run_id:           str
    axiom_state_hash: str

@dataclass(frozen=True)
class DaemonResponse:
    clean_signal:         str
    topology_class:       str
    topology_confidence:  float
    token_reduction_pct:  float
    extraction_empty:     bool
    phase:                int
    recipe_used:          str
    latency_ms:           float
    run_id:               str
```

---

## Exceptions Additions Required — signal_kernel/exceptions.py

```python
# ── Topology base ──────────────────────────────────────────────────────────
class TopologyException(KernelException): ...
    # all topology layer exceptions inherit from this
    # inheriting KernelException keeps unified catch blocks working

# ── Classifier ────────────────────────────────────────────────────────────
class ClassifierModelNotInitialized(TopologyException): ...
    # topology_router.pt not loaded — hard stop on startup

class ClassificationConfidenceTooLow(TopologyException): ...
    # all paths below THETA_CLASSIFY_FALLBACK
    # soft — falls back to GENERIC_HTML with WARNING log

class ClassificationWindowTooSmall(TopologyException): ...
    # content_prefix below minimum viable size
    # soft — uses URL + header signals only

# ── Parser ────────────────────────────────────────────────────────────────
class RecipeCompilationFailed(TopologyException): ...
    # WLP returned empty zone map, no fallback available
    # soft — registry falls back to parent class recipe

class RecipeVersionConflict(TopologyException): ...
    # attempted to compile version that already exists in registry
    # soft — increment version and retry

class WLPQueryFailed(TopologyException): ...
    # latent_parser.py returned None or raised unexpectedly
    # soft — use last known good recipe for this class

# ── Sanitizer ─────────────────────────────────────────────────────────────
class SanitizerInputError(TopologyException): ...
    # KernelOutput.clean_signal is None or not a string
    # hard stop — should never happen if pipeline.py is correct

# ── Surprise detector ─────────────────────────────────────────────────────
class SurpriseHistoryCorrupted(TopologyException): ...
    # phase_states.mmap surprise history section failed integrity check
    # soft — reinitialize history for affected class and continue

# ── Phantom ───────────────────────────────────────────────────────────────
class PhantomFetchFailed(TopologyException): ...
class PhantomRenderTimeout(TopologyException): ...
class PhantomFrictionDetected(TopologyException): ...
    # soft — DaemonResponse returns extraction_empty=True with friction type

# ── Index daemon ──────────────────────────────────────────────────────────
class GradientStepFailed(TopologyException): ...
    # hard stop — topology_router.pt may be corrupt
    # do not continue training — restore from checkpoint

class PhaseMMapCorrupted(TopologyException): ...
    # hard stop — phase_states.mmap failed integrity check on read
    # restore from last checkpoint before continuing

# ── Bus ───────────────────────────────────────────────────────────────────
class EventBusSubscriptionError(TopologyException): ...
    # component attempted to subscribe to unknown event type
    # hard stop — programming error, not runtime error

class EventDispatchFailed(TopologyException): ...
    # handler raised during dispatch
    # soft — log and continue, other subscribers unaffected
```

---

## Topology Class Registry

Every URL that enters AXIOM resolves to exactly one of these.
The question is never "what is this page about?" It is "how is this page built?"

```python
TOPOLOGY_CLASSES = [
    # News and editorial
    "NEWS_ARTICLE",
    "NEWS_ARTICLE_PAYWALLED",

    # Documentation
    "SAAS_DOCS",
    "SAAS_DOCS_VERSIONED",
    "SAAS_DOCS_WITH_CODE",       # SE separator splits prose + code

    # APIs
    "REST_API_JSON",
    "REST_API_JSON_PAGINATED",
    "JSON_LD_STRUCTURED",

    # Commerce
    "ECOMMERCE_PRODUCT",
    "ECOMMERCE_PRODUCT_VARIANT",

    # Community
    "FORUM_THREAD",
    "BLOG_POST",

    # Knowledge base
    "WIKIPEDIA_ARTICLE",         # pre-parsed into structural_layer.pt

    # Friction / dead ends
    "LANDING_PAGE",
    "AUTH_REDIRECT",
    "CLOUDFLARE_CHALLENGE",
    "RATE_LIMITED",

    # Fallback — always last resort
    "GENERIC_HTML",
]

PARENT_CLASS_MAP = {
    "NEWS_ARTICLE_PAYWALLED":    "NEWS_ARTICLE",
    "SAAS_DOCS_VERSIONED":       "SAAS_DOCS",
    "SAAS_DOCS_WITH_CODE":       "SAAS_DOCS",
    "REST_API_JSON_PAGINATED":   "REST_API_JSON",
    "ECOMMERCE_PRODUCT_VARIANT": "ECOMMERCE_PRODUCT",
    "FORUM_THREAD":              "BLOG_POST",
    "BLOG_POST":                 "NEWS_ARTICLE",
    "WIKIPEDIA_ARTICLE":         "NEWS_ARTICLE",
    "LANDING_PAGE":              "GENERIC_HTML",
    "AUTH_REDIRECT":             "GENERIC_HTML",
    "CLOUDFLARE_CHALLENGE":      "GENERIC_HTML",
    "RATE_LIMITED":              "GENERIC_HTML",
}
# fallback chain always terminates at GENERIC_HTML
# registry.get_recipe() never returns None
# maximum recursion depth = 3 (longest parent chain)
```

---

## File-by-File Specification

---

### `tag/crawler_bus.py`
**Builder: Sonnet | Depends on: contracts.py only**

The event bus. Typed subscribe/emit with zero logic. Every intelligent component
subscribes to events it cares about at startup. Every producing component calls
`BUS.emit(event)`. The bus dispatches to all registered handlers as background
asyncio tasks. No handler waits for another. No handler can block the bus.
If a handler raises, caught, logged, next handler runs unaffected.

```python
class CrawlerBus:
    def subscribe(
        self,
        event_type: Type[T],
        handler: Callable[[T], Awaitable[None]],
    ) -> None

    async def emit(self, event: Any) -> None
    def handler_count(self, event_type: Type) -> int   # health checks
    async def drain(self) -> None                       # testing only

BUS = CrawlerBus()    # singleton — imported everywhere
```

**Rule:** No routing logic. No filtering. No priority. If a component needs to
filter events, it does so in its own handler. The bus dispatches to all subscribers.

---

### `tag/store_watchdog.py`
**Builder: Sonnet | Depends on: contracts.py, exceptions.py**

inotify-based `/store` file watcher. Zero polling. Change detected in microseconds.
Debounced — `os.rename()` triggers one clean event, debounce prevents handling
partial writes as complete. Every store-reading component registers at startup.

```python
class StoreWatchdog:
    def register(
        self,
        path: str,
        handler: Callable[[], Awaitable[None]],
        debounce_ms: int = 500,
    ) -> None

    async def start(self) -> None
    async def stop(self) -> None

WATCHDOG = StoreWatchdog()    # singleton
```

**Debounce by file:**
```
structural_layer.pt     500ms   large file, write takes time
topology_router.pt      500ms   same
recipe_registry.mmap    100ms   fast write, tight window fine
phase_states.mmap       100ms   same
triggers/cold_start    1000ms   generous window on startup
triggers/preparse      1000ms   same
triggers/reload_*       200ms   reload signals need fast response
```

**What happens when `structural_layer.pt` changes (atomic swap by preparse):**
```
store_watchdog fires (debounced 500ms)
    → wlm._reload_structural_layer()   (background task)
    → wlp._reload_structural_layer()   (background task)
    → index_daemon._reload_structural_layer()  (background task)
All three run in parallel. None block the critical path.
Next query uses new weights.
```

---

### `tag/topology/classifier.py`
**Builder: Opus (model path) + Sonnet (signal paths)**
**Depends on: contracts.py, exceptions.py, store/topology_router.pt**

Entry point for every run. Receives `ClassificationWindow`. Returns
`TopologyClassification`. **Never fetches a page.** Classification happens
before fetching because the traversal policy depends on knowing the topology class.

**Signal hierarchy — tried in order, stops when confidence ≥ THETA_CLASSIFY_CONFIDENT:**

**Path 1 — Domain fingerprint** *(Sonnet)*
Direct lookup. `docs.stripe.com` → `SAAS_DOCS`. No model. No HTTP. Microseconds.
`DOMAIN_FINGERPRINT_TABLE` constant. Extended by `structural_layer.pt` over time as
the preparser maps more domains — the table is a starting point, not a ceiling.

```python
DOMAIN_FINGERPRINT_TABLE: Dict[str, str] = {
    "docs.stripe.com":       "SAAS_DOCS",
    "docs.twilio.com":       "SAAS_DOCS",
    "developer.mozilla.org": "SAAS_DOCS",
    "*.wikipedia.org/wiki/*":"WIKIPEDIA_ARTICLE",
    "arxiv.org/abs/*":       "JSON_LD_STRUCTURED",
    # ... grows as preparser maps more domains
}
```

**Path 2 — URL structure** *(Sonnet)*
Regex patterns against URL path. `/api/v*/` → `REST_API_JSON`. No HTTP call.

**Path 3 — Response headers** *(Sonnet)*
`Content-Type: application/json` → `REST_API_JSON`.
`CF-RAY` header → `CLOUDFLARE_CHALLENGE` check.
`X-Robots-Tag: noindex` on known domains → `NEWS_ARTICLE_PAYWALLED`.

**Path 4 — Classification window** *(Sonnet)*
First 4KB of content. grep pass. Bounded, fast.
`<script type="application/ld+json">` → `JSON_LD_STRUCTURED`.
`data-product-id` → `ECOMMERCE_PRODUCT`.
`<article` → `NEWS_ARTICLE`.

**Path 5 — ML classifier** *(Opus)*
Only invoked when paths 1-4 below `THETA_CLASSIFY_CONFIDENT`.
`_embed_signals()` converts all signal outputs into a feature vector.
`topology_router.pt` forward pass → probability distribution over topology classes.
Argmax → predicted class. Max probability → confidence.

```python
def _embed_signals(
    url: str,
    headers: Dict[str, str],
    content_prefix: str,
    domain_hint: Optional[str],
) -> torch.Tensor:
    # Opus builds this — feature engineering for topology classification
    # must produce consistent vectors across restarts
    # feature space must be meaningful to the MLP's learned weights

def _classify_via_model(features: torch.Tensor) -> Tuple[str, float]:
    # Opus builds this — topology_router.pt forward pass
    # returns (topology_class, confidence)
```

**Fallback:** path 5 confidence below `THETA_CLASSIFY_FALLBACK` →
emit `ClassificationConfidenceTooLow` warning, return `GENERIC_HTML`.

**Store interaction:**
```python
# registered at initialize()
WATCHDOG.register("store/topology_router.pt", self._reload_classifier_model, 500)

async def _reload_classifier_model(self) -> None:
    new_model = torch.load("store/topology_router.pt")
    self._model = new_model    # GIL-safe atomic assignment
```

**Write order:**
```
1.  DOMAIN_FINGERPRINT_TABLE constant
2.  URL_STRUCTURE_PATTERNS constant
3.  HEADER_SIGNALS constant
4.  _load_classifier_model()
5.  _classify_by_domain()         Sonnet
6.  _classify_by_url()            Sonnet
7.  _classify_by_headers()        Sonnet
8.  _classify_by_window()         Sonnet
9.  _embed_signals()              Opus
10. _classify_via_model()         Opus
11. classify()                    public
12. initialize()                  registers watchdog, loads model
```

---

### `tag/world_model/latent_parser.py` (WLP)
**Builder: Opus | Depends on: contracts.py, exceptions.py, crawler_bus.py, store/structural_layer.pt**

Street-level view. Knows where signal lives inside pages of each topology class.
Produces `ZoneMap` objects consumed by `parser.py` to compile grep recipes.

**Two distinct knowledge layers encoded in `structural_layer.pt`:**
```
Universal (all topology classes):
    nav     → noise
    footer  → noise
    aside   → noise
    .sidebar → noise
    .breadcrumb → noise
    .cookie-banner → noise

Class-specific:
    SAAS_DOCS:          <main> → signal
    NEWS_ARTICLE:       <article> → signal
    REST_API_JSON:      "data" key → signal, "pagination"/"meta" → noise
    ECOMMERCE_PRODUCT:  data-product-* → signal
    WIKIPEDIA_ARTICLE:  #mw-content-text .mw-parser-output > p → signal
                        .navbox → noise, .infobox → separate structured pass
```

**Bus subscription:**
```python
BUS.subscribe(CleanSignalEvent, self._on_clean_signal)

async def _on_clean_signal(self, event: CleanSignalEvent) -> None:
    # update zone confidence for this topology class
    # high signal_density → zone map confirmed
    # low signal_density → zone map may need revision
    # if confidence delta exceeds threshold → emit ZoneMapUpdatedEvent
    updated = await self._update_zone_confidence(event)
    if updated:
        await BUS.emit(ZoneMapUpdatedEvent(
            topology_class=event.topology_class,
            new_zone_map=self._zone_maps[event.topology_class],
        ))
```

**Core Opus responsibility:**
`_update_zone_confidence()` — the learning rate of the WLP's structural knowledge.
How much a single low-density extraction should affect zone confidence before
triggering a recompile. How to weight new observations against rolling history.
How to distinguish "bad page" from "bad zone map."

**At query time (called directly by interface.py in parallel with WLM):**
```python
async def query(self, topology_class: str) -> ZoneMap:
    # returns cached zone map — must be fast
    # O(1) dict lookup for known classes
    # falls back to parent class zone map if class not yet learned
```

**Write order:**
```
1.  UNIVERSAL_NOISE_ZONES constant
2.  ZONE_STRATEGY_MAP constant         class → "zone_extract"|"attribute_extract"|"envelope_extract"
3.  _load_structural_layer()
4.  _get_cached_zone_map()
5.  _compute_zone_confidence()         Opus
6.  _update_zone_confidence()          Opus
7.  _should_emit_zone_update()         Opus: threshold + hysteresis
8.  _on_clean_signal()                 bus handler
9.  query()                            public, critical path call
10. initialize()                       loads structural_layer.pt, subscribes bus, registers watchdog
```

---

### `tag/world_model/latent_model.py` (WLM)
**Builder: Opus | Depends on: contracts.py, exceptions.py, crawler_bus.py, store/structural_layer.pt**

Satellite view. Knows how to approach a domain before the first byte is fetched.
Outputs `TraversalPolicy` and `FrictionForecast`. `phantom.py` uses these to
decide render mode, pacing, depth, and retry budget.

**Bus subscription:**
```python
BUS.subscribe(DomainTopologyEvent, self._on_domain_topology)

async def _on_domain_topology(self, event: DomainTopologyEvent) -> None:
    # preparser has built a DomainMap for a new or updated domain
    # WLM ingests and updates its structural knowledge
    # known domain? update existing policy
    # unknown domain? create new policy from structural primitives
```

**At query time (called directly by interface.py in parallel with WLP):**
```python
async def query(self, topology_class: str) -> WLMResponse:
    # critical path call — must be fast
    # forward pass on structural_layer.pt
    # O(1) for known topology classes
    # returns TraversalPolicy + FrictionForecast + SourcePriority
```

**Core Opus responsibility:**
The forward pass converting topology class + structural layer weights into
`TraversalPolicy`. How structural primitives encode into traversal parameters.
`docs.stripe.com` → known Cloudflare → render mode headless → crawl delay 10s.
All of that is learned in weights. Opus designs the feature space.

**Write order:**
```
1.  DEFAULT_TRAVERSAL_POLICY constant      safe defaults for unknown topology
2.  CDN_FINGERPRINT_TABLE constant         cloudflare/fastly/akamai patterns
3.  BOT_MITIGATION_SIGNATURES constant
4.  _load_structural_layer()
5.  _forward_pass()                        Opus: structural_layer.pt → policy vectors
6.  _build_traversal_policy()              Opus
7.  _build_friction_forecast()             Opus
8.  _on_domain_topology()                  bus handler
9.  query()                                public, critical path call
10. initialize()                           loads weights, subscribes bus, registers watchdog
```

---

### `tag/topology/parser.py`
**Builder: Opus**
**Depends on: contracts.py, exceptions.py, crawler_bus.py,
              world_model/latent_parser.py,
              signal_kernel/recipes/registry.py,
              signal_kernel/recipes/validator.py**

The recipe compiler. Receives `ZoneMap` from WLP. Compiles into executable
grep recipe. Writes to `signal_kernel/recipes/compiler_generated/`. Registers
in `registry.py`. Returns `RecipeMount` for `pipeline.execute()`.

**Bus subscription:**
```python
BUS.subscribe(ZoneMapUpdatedEvent, self._on_zone_map_updated)

async def _on_zone_map_updated(self, event: ZoneMapUpdatedEvent) -> None:
    # WLP revised its zone knowledge — recompile recipe for that class
    # old recipe stays active until new one passes validator.check()
    new_recipe = await self.compile(event.topology_class)
    # registry atomically swaps to new recipe on successful compile
```

**Three compilation strategies (Opus builds all three):**
```
ZONE_EXTRACT        signal in defined HTML zone
                    grep -A 10000 "<signal_zone" | grep -B 10000 "</signal_zone>"
                    | grep -v noise_zone_1 | grep -v noise_zone_2
                    | sed 's/<[^>]*>//g' | grep -v "^[[:space:]]*$"
                    | tr -s ' '

ATTRIBUTE_EXTRACT   signal in data-* attributes (ECOMMERCE classes)
                    grep -E 'data-(product|price|sku|name)'
                    | grep -v "data-analytics" | grep -v "data-tracking"
                    | sed 's/<[^>]*>//g' | grep -v "^$"

ENVELOPE_EXTRACT    signal in JSON envelope (REST_API classes)
                    grep -o '"data":[^}]*'
                    | grep -v '"pagination"' | grep -v '"meta"' | grep -v '"links"'
```

**Feedback injection (Opus) — what makes recipes smarter over time:**
```python
def _inject_feedback(
    self,
    recipe_lines: List[str],
    last_kernel_output: Optional[KernelOutput],
) -> List[str]:
    if last_kernel_output is None:
        return recipe_lines   # first compile, no feedback yet

    if last_kernel_output.extraction_empty:
        # empty extraction: widen zone selector, add fallback zones
        # e.g. if <main> produced nothing, also try <article>, .content

    if last_kernel_output.signal_density < SIGNAL_DENSITY_THRESHOLD:
        # low density: reduce noise exclusions (over-stripping)

    return adjusted_recipe_lines
```

**Recursive fallback:**
```python
def compile(
    self,
    topology_class: str,
    feedback: Optional[KernelOutput] = None,
) -> RecipeMount:

    zone_map = self._query_wlp(topology_class)

    if zone_map.confidence < THETA_WLP_MIN:
        parent = PARENT_CLASS_MAP.get(topology_class)
        if parent:
            return self.compile(parent, feedback)   # recurse up
        return registry.get_recipe("GENERIC_HTML")  # base case

    recipe = self._assemble_recipe(zone_map, feedback)
    validator.check(recipe)       # raises RecipeInjectionAttempt if hostile
    return self._register_recipe(recipe)
```

**Write order:**
```
1.  RECIPE_TEMPLATE_ZONE constant
2.  RECIPE_TEMPLATE_ATTRIBUTE constant
3.  RECIPE_TEMPLATE_ENVELOPE constant
4.  ALLOWED_COMMANDS constant          mirrors validator.py allowlist exactly
5.  MAX_RECIPE_LINES constant          50, mirrors validator.py ceiling
6.  _query_wlp()                       Opus
7.  _select_strategy()                 Opus
8.  _build_zone_extract_recipe()       Opus
9.  _build_attribute_extract_recipe()  Opus
10. _build_envelope_extract_recipe()   Opus
11. _inject_feedback()                 Opus
12. _assemble_recipe()                 Opus
13. _write_recipe_file()               Sonnet: write to disk, hash
14. _register_recipe()                 Sonnet: call registry.register_recipe()
15. _on_zone_map_updated()             bus handler
16. compile()                          public
```

---

### `tag/topology/sanitizer.py`
**Builder: Sonnet entirely | Depends on: contracts.py, exceptions.py**

Last mile strip. Input: `KernelOutput.clean_signal` (already text — kernel
converted HTML to text). Output: `SanitizedSignal`. Removes residual artifacts
grep cannot see. Deterministic. No model. No LLM. String operations only.

**Operations in strict order (order is load-bearing):**
```
1. HTML entity decode       html.unescape() + custom table for misses
2. Unicode NFC normalize    unicodedata.normalize("NFC", text)
3. Lone punctuation strip   lines containing only |›»·— punctuation chars
4. GDPR fragment strip      "we use cookies", "accept all cookies", "privacy policy"
5. JS artifact strip        lines starting with var, function(, window., document.
6. Code remnant strip       lines with >40% symbol density (escaped zone fragments)
7. Whitespace normalize     collapse >2 blank lines. strip per-line. strip document.
8. Minimum length check     below 64 bytes → sanitized_empty=True
```

Step 6 is new in this version. grep zone stripping sometimes leaves behind
high-symbol-density lines from inline script fragments or encoded content.
These are code-adjacent noise, not signal. Removed here before SE separator sees them.

**Write order:**
```
1. GDPR_FRAGMENT_PATTERNS constant
2. JS_ARTIFACT_PATTERNS constant
3. LONE_PUNCTUATION_PATTERN constant
4. CODE_DENSITY_THRESHOLD constant         0.40
5. MIN_SIGNAL_BYTES constant               64
6-13. one private function per operation   in operation order above
14. sanitize()                             public, applies 6-13 in order
```

---

### `tag/topology/se_separator.py`
**Builder: Sonnet | Depends on: contracts.py**

New file. Sits between sanitizer output and Haiku invocation. Splits
`SanitizedSignal.text` into `SeparatedSignal`. Prose zone and code zone
go to Haiku in parallel with different prompts. Two focused calls beat
one blended call — prose needs comprehension, code needs structural parsing.

**Split markers:**
```
<pre>...</pre>              → code zone
<code>...</code>            → code if >4 words, inline otherwise (stays in prose)
```python ... ```           → code (fenced blocks surviving sanitizer)
lines starting with $       → code (shell commands)
lines starting with >>>     → code (Python REPL)
lines starting with //      → code (comments)
everything else             → prose zone
```

```python
# Interface.py usage
separated = se_separator.split(sanitized_signal)

prose_result, code_result = await asyncio.gather(
    haiku.extract_prose(separated.prose_signal, topology_class),
    haiku.extract_code(separated.code_signal, topology_class),
)
```

---

### `tag/topology/surprise_detector.py`
**Builder: Opus (score + dissolve) + Sonnet (history + mmap writes)**
**Depends on: contracts.py, exceptions.py, crawler_bus.py, store/phase_states.mmap**

Compares actual extraction outcomes against world model predictions. Computes
surprise score. Emits `SurpriseEvent` to bus. `index_daemon.py` subscribes and
decides gradient step vs policy dissolve. The immune system of Phase III.

**What it measures:**
```
signal_density_divergence   |predicted_density - actual_density|
token_reduction_divergence  |predicted_reduction - actual_reduction|
empty_extraction_rate       empty extractions / window size (rolling)
recipe_error_rate           non-zero exit codes / window size (rolling)
```

**Surprise score formula (Opus designs weights):**
```
surprise = w1 * density_divergence
         + w2 * reduction_divergence
         + w3 * empty_rate_above_baseline
         + w4 * recipe_error_rate

Phase II weights: w3, w4 heavier (early warning)
Phase III weights: balanced (defending compiled policy)
```

**Policy dissolve — hysteresis (Opus):**
Single spikes do not trigger dissolve. Opus reasons about appropriate consecutive
evaluation thresholds: Phase III requires N=3 consecutive evaluations above theta
before dissolving. Phase II requires N=1 (faster to respond, less confident policy).

**Bus emission:**
```python
async def evaluate(
    self,
    actual:    KernelOutput,
    predicted: SurprisePrediction,
) -> SurpriseEvent:
    score   = self._compute_surprise_score(actual, predicted)
    dissolve = self._should_dissolve(score, actual.topology_class)
    event   = SurpriseEvent(...)
    await BUS.emit(event)    # index_daemon subscribes
    return event             # interface.py receives but does not wait for downstream
```

**Write order:**
```
1.  THETA_SURPRISE_DEFAULT constant
2.  SURPRISE_WINDOW_SIZE constant          50 runs per class
3.  PHASE_WEIGHTS constant                 w1-w4 per phase
4.  _load_history()                        Sonnet: from phase_states.mmap
5.  _save_history()                        Sonnet: to phase_states.mmap
6.  _compute_density_divergence()
7.  _compute_reduction_divergence()
8.  _compute_empty_rate()
9.  _compute_recipe_error_rate()
10. _compute_surprise_score()              Opus
11. _should_dissolve()                     Opus: threshold + hysteresis
12. evaluate()                             public, emits SurpriseEvent
13. get_prediction()                       public, returns SurprisePrediction for class
```

---

### `tag/phantom/phantom.py`
**Builder: Sonnet | Depends on: contracts.py, exceptions.py, world_model/latent_model.py**

Live traversal. The only component in `tag/` that touches the network.
Receives `TraversalPolicy` from WLM. Returns `PhantomResult`. No decisions inside —
all decisions are made by `render_policy.py` before `phantom.py` is called.

**Two modes:**
```
STATIC   httpx — for render_mode: "static"
                 sub-100ms
                 correct for REST API JSON, most news articles, JSON-LD

HEADLESS Playwright — for render_mode: "headless"
                      required when structural_layer.pt says JS is needed
                      SaaS docs with lazy-loading
                      ecommerce with React rendering
                      expensive — only when WLM says necessary
```

**Friction detection inside fetch:**
```python
def _detect_friction(self, response: httpx.Response) -> Optional[str]:
    # checks response for known friction patterns
    # "cloudflare" | "paywall" | "rate_limited" | "auth_redirect" | None
    # if friction detected → raises PhantomFrictionDetected
    # caller catches and returns extraction_empty=True with friction type
```

**Write order:**
```
1. _fetch_static()     httpx, respect TraversalPolicy.requests_per_second
2. _fetch_headless()   Playwright, respect TraversalPolicy.timeout_ms
3. _detect_friction()  check for known friction fingerprints
4. fetch()             public, dispatches to static or headless
```

---

### `tag/index_daemon.py`
**Builder: Opus (RL logic) + Sonnet (lifecycle + mmap writes + rsync trigger)**
**Depends on: contracts.py, exceptions.py, crawler_bus.py,
              signal_kernel/feedback.py,
              topology/surprise_detector.py,
              store/topology_router.pt,
              store/phase_states.mmap,
              store/structural_layer.pt**

The closed RL loop. Never terminates. Subscribes to everything. Runs gradient
steps on `topology_router.pt` when surprise is low. Triggers policy dissolve
when surprise is high. Manages phase transitions. Shares new structural
primitives to the hivemind fleet via rsync trigger.

**Bus subscriptions:**
```python
BUS.subscribe(SurpriseEvent,       self._on_surprise)
BUS.subscribe(CleanSignalEvent,    self._on_clean_signal)
BUS.subscribe(ZoneMapUpdatedEvent, self._on_zone_map_updated)
BUS.subscribe(DomainTopologyEvent, self._on_domain_topology)
```

**The update decision tree (Opus):**
```
SurpriseEvent arrives:
    if surprise_score < THETA_SURPRISE:
        gradient_step(topology_router.pt, feedback_signal)
        update_phase_confidence(topology_class, +delta)
        if confidence > THETA_CONFIDENCE_III:
            transition_to_phase_iii(topology_class)
        elif confidence > THETA_CONFIDENCE_II:
            transition_to_phase_ii(topology_class)

    else:  # high surprise
        dissolve_policy(topology_class)
        if is_new_structural_primitive(surprise_event):
            append_to_structural_layer(structural_layer.pt)
            Path("/store/triggers/hivemind_sync").touch()
            # rsync daemon picks this up — shares to fleet
```

**Gradient steps are atomic:**
`topology_router.pt` updated via staging + rename — same pattern as preparse.
Never partially written. In-flight forward passes complete against old weights.
Next forward pass uses new weights.

**Store interaction:**
```python
# registered at initialize()
WATCHDOG.register("store/structural_layer.pt", self._reload_structural_layer, 500)
WATCHDOG.register("store/topology_router.pt",  self._reload_topology_router,  500)
```

**Write order:**
```
1.  _load_topology_router()
2.  _load_phase_states()
3.  _load_structural_layer()
4.  _gradient_step()                    Opus: RL update semantics
5.  _dissolve_policy()                  Opus: phase III → II or II → I
6.  _is_new_structural_primitive()      Opus: what qualifies as structural primitive
7.  _append_to_structural_layer()       Opus: how to extend structural weights
8.  _update_phase_confidence()          Sonnet: mmap write
9.  _transition_phase()                 Sonnet: mmap write + log
10. _trigger_hivemind_sync()            Sonnet: touch trigger file
11. _on_surprise()                      bus handler (primary — most work here)
12. _on_clean_signal()                  bus handler (lightweight feedback signal)
13. _on_zone_map_updated()              bus handler (structural knowledge update)
14. _on_domain_topology()              bus handler (new domain from preparser)
15. run()                               public, starts event loop, never returns
```

---

### `tag/interface.py`
**Builder: Sonnet | Depends on: all of the above**

The only public surface of the TAG layer. Written last. The AXIOM graph calls
`execute()`. It calls nothing else in `tag/` directly.

**Full call sequence:**
```python
async def execute(request: DaemonRequest) -> DaemonResponse:

    # 1. classify — before any fetch
    classification = await classifier.classify(ClassificationWindow(
        url=request.candidate_urls[0],
        headers={}, content_prefix="", content_type="unknown",
    ))

    # 2. world model — parallel, different concerns
    wlm_response, zone_map = await asyncio.gather(
        wlm.query(classification.topology_class),
        wlp.query(classification.topology_class),
    )

    # 3. compile recipe from WLP zone map
    recipe = parser.compile(
        classification.topology_class,
        feedback=_get_last_feedback(classification.topology_class),
    )

    # 4. fetch — WLM policy drives render mode + pacing
    phantom_result = await phantom.fetch(
        url=request.candidate_urls[0],
        policy=wlm_response.traversal_policy,
    )

    # 5. execute kernel
    kernel_output = pipeline.execute_sync(KernelInput(
        raw_content=phantom_result.raw_content,
        topology_class=classification.topology_class,
        content_type=phantom_result.content_type,
        source_url=phantom_result.url,
    ))

    # 6. sanitize
    sanitized = sanitizer.sanitize(kernel_output)

    # 7. SE separate — prose and code
    separated = se_separator.split(sanitized)

    # 8. surprise — async, non-blocking
    prediction = surprise_detector.get_prediction(classification.topology_class)
    await surprise_detector.evaluate(kernel_output, prediction)
    # BUS.emit(SurpriseEvent) happens inside evaluate()
    # index_daemon processes it in background — we never await it

    # 9. return
    return DaemonResponse(
        clean_signal=sanitized.text,
        topology_class=classification.topology_class,
        topology_confidence=classification.confidence,
        token_reduction_pct=_compute_reduction(kernel_output, phantom_result),
        extraction_empty=sanitized.sanitized_empty,
        phase=_get_phase(classification.topology_class),
        recipe_used=recipe.recipe_path,
        latency_ms=_elapsed(),
        run_id=request.run_id,
    )
```

---

### `tag/cold_start.py`
**Builder: Sonnet | Depends on: everything**

Runs before `interface.py` accepts queries. Fast path if weights are fresh (<24h).
Slow path triggers preparse via trigger file — crond + preparse_daemon handles it.
"Cold start" is the cover. GPU batch preparse is the reality. Nobody asks why.

```python
async def cold_start() -> None:
    # Phase 1 — infrastructure
    await WATCHDOG.start()
    await registry.initialize()
    await checkpoint_monitor.initialize()

    # Phase 2 — weights
    if store_is_fresh():
        await _load_existing_weights()           # seconds
    else:
        Path("/store/triggers/cold_start").touch()
        await _wait_for_preparse_completion()    # minutes on first boot

    # Phase 3 — topology layer
    await classifier.initialize()
    await wlm.initialize()
    await wlp.initialize()
    asyncio.create_task(index_daemon.run())      # background, never awaited

    # Phase 4 — validation
    await _validate_store_integrity()
    await _validate_classifier_ready()
    await _validate_kernel_ready()

    # Phase 5 — open
    interface.set_ready(True)

def store_is_fresh() -> bool:
    m = manifest.read()
    if not m: return False
    return (now() - m.last_parse).total_seconds() < 86400   # 24 hours
```

---

## Build Sequence — Strict Order

```
════════════════════════════════════════════════════════════════
 BEFORE ANYTHING ELSE — contract + exception additions
════════════════════════════════════════════════════════════════

    signal_kernel/contracts.py    add all dataclasses listed above
    signal_kernel/exceptions.py   add full topology exception hierarchy


════════════════════════════════════════════════════════════════
 PHASE 1 — INFRASTRUCTURE
════════════════════════════════════════════════════════════════

    tag/crawler_bus.py            SONNET    depends: contracts
    tag/store_watchdog.py         SONNET    depends: contracts, exceptions


════════════════════════════════════════════════════════════════
 PHASE 2 — WORLD MODEL
════════════════════════════════════════════════════════════════

    tag/world_model/latent_parser.py    OPUS      depends: contracts, bus, structural_layer.pt
    tag/world_model/latent_model.py     OPUS      depends: contracts, bus, structural_layer.pt


════════════════════════════════════════════════════════════════
 PHASE 3 — CLASSIFICATION
════════════════════════════════════════════════════════════════

    tag/topology/classifier.py    OPUS+SONNET   depends: contracts, exceptions, topology_router.pt


════════════════════════════════════════════════════════════════
 PHASE 4 — COMPILATION + LAST MILE
════════════════════════════════════════════════════════════════

    tag/topology/parser.py        OPUS      depends: contracts, exceptions, bus,
                                                     latent_parser.py,
                                                     signal_kernel/recipes/registry.py,
                                                     signal_kernel/recipes/validator.py

    tag/topology/sanitizer.py     SONNET    depends: contracts, exceptions

    tag/topology/se_separator.py  SONNET    depends: contracts


════════════════════════════════════════════════════════════════
 PHASE 5 — LIVE TRAVERSAL
════════════════════════════════════════════════════════════════

    tag/phantom/render_policy.py  SONNET    depends: contracts
    tag/phantom/phantom.py        SONNET    depends: contracts, exceptions, latent_model.py


════════════════════════════════════════════════════════════════
 PHASE 6 — SURPRISE DETECTION
════════════════════════════════════════════════════════════════

    tag/topology/surprise_detector.py   OPUS+SONNET   depends: contracts, exceptions,
                                                               bus, phase_states.mmap


════════════════════════════════════════════════════════════════
 PHASE 7 — THE LOOP
════════════════════════════════════════════════════════════════

    tag/index_daemon.py           OPUS+SONNET   depends: contracts, exceptions, bus,
                                                          surprise_detector.py,
                                                          signal_kernel/feedback.py,
                                                          topology_router.pt,
                                                          phase_states.mmap,
                                                          structural_layer.pt


════════════════════════════════════════════════════════════════
 PHASE 8 — BOUNDARY + INIT — written absolutely last
════════════════════════════════════════════════════════════════

    tag/interface.py              SONNET    depends: everything above
    tag/cold_start.py             SONNET    depends: everything above
```

---

## Non-Negotiable Rules

**1. The classifier never fetches a page.**
Classification happens before fetching. URL + headers + 4KB window is the budget.

**2. The parser never executes recipes.**
`topology/parser.py` never calls `pipeline.execute()`.
Parser compiles. Kernel executes. These are different processes.

**3. The sanitizer never uses an LLM.**
If it needs a model, the recipe is under-stripping. Fix the recipe.

**4. The surprise detector fires on divergence, not on failure.**
Empty extraction on a predicted-paywalled page is near-zero surprise.
Prediction error is the signal, not absolute outcome quality.

**5. `index_daemon.py` processes events asynchronously.**
`interface.py` does not await gradient steps before returning `DaemonResponse`.
Query latency is never held by the training loop.

**6. Phases are earned, not assigned.**
No component sets a topology class to Phase III by editing config.
Phase III is the result of the RL loop accumulating sufficient confidence.

**7. `interface.py` is the only public surface.**
AXIOM graph calls `interface.py`. Nothing else in `tag/`. No exceptions.

**8. `crawler_bus.py` has no logic.**
It dispatches. Filtering and routing belong in subscriber handlers.

**9. All store writes use staging + atomic rename.**
`topology_router.pt` and `structural_layer.pt` are never partially written.
Same pattern as the kernel's checkpoint system. Same reason.

**10. WLM and WLP are called in parallel, never sequentially.**
```python
wlm_response, zone_map = await asyncio.gather(
    wlm.query(topology_class),
    wlp.query(topology_class),
)
```
One awaits the other in no version of this codebase.

---

## LOC Estimate

```
contracts.py additions            ~400 loc
exceptions.py additions           ~200 loc
crawler_bus.py                    ~200 loc
store_watchdog.py                 ~300 loc
topology/classifier.py          ~1,500 loc
topology/parser.py              ~2,000 loc
topology/sanitizer.py             ~500 loc
topology/se_separator.py          ~300 loc
topology/surprise_detector.py   ~1,200 loc
world_model/latent_model.py     ~2,000 loc
world_model/latent_parser.py    ~1,800 loc
phantom/phantom.py                ~800 loc
phantom/render_policy.py          ~200 loc
index_daemon.py                 ~2,500 loc
interface.py                      ~700 loc
cold_start.py                     ~400 loc
                                ─────────
tag/ production total:          ~15,000 loc
Tests (estimated 1:1):          ~15,000 loc
                                ─────────
tag/ total estimate:            ~30,000 loc

Combined with signal_kernel/ (25K):
AXIOM TAG layer complete:       ~55,000 loc
```

---

## What Success Looks Like

```
Classifier:         16 topology classes, confidence >0.75 for known domains in <5ms
                    <15ms for unknown domains requiring ML path
                    GENERIC_HTML fallback fires correctly on unknown structural patterns

Compiler:           All 16 classes produce valid recipes passing validator.py
                    All recipes achieve >50% token reduction on canonical test pages
                    Recursive fallback correctly resolves to parent class on WLP miss

Sanitizer:          Zero HTML entities, GDPR fragments, JS artifacts reaching SE separator
                    Idempotent on already-clean input (no content lost)

SE separator:       Prose and code correctly identified and separated
                    Parallel Haiku calls produce demonstrably richer output than single call

Surprise detector:  Policy dissolve fires correctly when structural change introduced
                    Hysteresis prevents dissolve on single-run noise
                    Near-zero surprise on correctly-predicted empty extractions

Index daemon:       topology_router.pt weights measurably improving across runs
                    Phase transitions firing at correct confidence thresholds
                    Gradient steps complete without corrupting store files

Full loop:          DaemonRequest → DaemonResponse in <2 seconds
                    including live traversal on SAAS_DOCS page
                    Haiku receives clean signal, not HTML, not noise

Compounding:        Third run on a previously unknown class measurably faster than first
                    Evidence that the system learns, not just executes
```

---

*AXIOM Core Searching Algorithm — Topology Layer Developer Reference v2*
*signal_kernel/ complete and production-validated.*
*tag/ begins with contracts.py additions.*
*Crawler is a bus. Layers subscribe. WLM and WLP run in parallel.*
*Loop never terminates. Phases are earned.*
*AXIOM INTERNAL // DO NOT SURFACE*
