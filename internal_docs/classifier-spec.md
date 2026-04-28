# `topology/classifier.py` — Build Specification

**File:** `tag/topology/classifier.py`  
**Estimated LOC:** ~1,500 (production) / ~2,200 (with tests)  
**Sonnet builds:** Signal paths 1–4, hard overrides, store interaction, `classify()`, `initialize()`  
**Opus builds:** `_embed_signals()`, `_classify_via_model()` (the ML path)  
**Depends on:** `contracts.py`, `exceptions.py`, `store/topology_router.pt`, `store_watchdog.py`  
**Written by:** Sonnet (infrastructure) + Opus (ML path)

---

## What It Is

The classifier is the first thing that runs on every URL AXIOM touches. It takes a URL, HTTP response headers, and the first 4KB of content and produces a `TopologyClassification` — one of 18 known classes with a confidence score. Everything downstream (traversal policy, fetch strategy, recipe selection, WLM routing) depends on this result.

Classification happens **before** full page fetching. The traversal policy is determined by topology class. You cannot fetch intelligently without knowing what you're fetching.

## What It Is Not

| Classifier does NOT | That belongs to |
|---|---|
| Fetch pages | `phantom.py` |
| Compile recipes | `topology/parser.py` |
| Know about ZoneMaps | WLP owns that |
| Run gradient steps | `index_daemon.py` |
| Write to store | Read-only from `topology_router.pt` |
| Communicate with WLP | Parallel and independent |
| Return `None` | Never. Fallback is always `GENERIC_HTML` |

---

## Input Contract

```python
@dataclass(frozen=True)
class ClassifierInput:
    url: str                          # full URL including query string
    headers: Dict[str, str]          # HTTP response headers, lowercased keys
    content_prefix: bytes            # first 4096 bytes of response body
    response_code: int               # HTTP status code
```

## Output Contract

```python
@dataclass(frozen=True)
class TopologyClassification:
    topology_class: TopologyClass    # one of 18 known classes
    confidence: ConfidenceFloat      # [0.0, 1.0]
    path_used: ClassificationPath    # which signal path resolved it
    fallback_chain: List[str]        # classes tried before settling (may be empty)
```

Both are defined in `contracts.py`.

---

## Classification Paths (in order)

```
Path 1 → Domain fingerprint         (microseconds, dict lookup)
Path 2 → URL structure              (microseconds, regex)
Path 3 → Response headers           (microseconds, key lookup)
Path 4 → Classification window      (milliseconds, bounded grep)
                ↓
       Hard overrides checked here  (AUTH, CLOUDFLARE, RATE_LIMITED)
                ↓
Path 5 → ML model                   (<15ms, topology_router.pt forward pass)
                ↓
       GENERIC_HTML fallback        (if Path 5 confidence < THETA_CLASSIFY_FALLBACK)
```

Paths 1–4 are deterministic. No model. No randomness. If any path returns confidence ≥ `THETA_CLASSIFY_CONFIDENT = 0.75`, the ML path is skipped entirely.

---

## SONNET BUILDS

### Constants

**`DOMAIN_FINGERPRINT_TABLE`**

Direct dict lookup. Pattern → TopologyClass string. Extended by preparser over time as domains are mapped. Sonnet seeds this at build time.

```python
DOMAIN_FINGERPRINT_TABLE: Dict[str, str] = {
    "docs.stripe.com":            "SAAS_DOCS",
    "docs.twilio.com":            "SAAS_DOCS",
    "developer.mozilla.org":      "SAAS_DOCS",
    "docs.github.com":            "SAAS_DOCS",
    "docs.aws.amazon.com":        "SAAS_DOCS",
    "*.wikipedia.org/wiki/*":     "WIKIPEDIA_ARTICLE",
    "arxiv.org/abs/*":            "JSON_LD_STRUCTURED",
    "api.github.com/*":           "REST_API_JSON",
    "*.shopify.com/products/*":   "ECOMMERCE_PRODUCT",
    "reddit.com/r/*/comments/*":  "FORUM_THREAD",
    "news.ycombinator.com/item*": "FORUM_THREAD",
    "medium.com/*":               "BLOG_POST",
    # grows as preparser maps more domains
}
```

Wildcard patterns (`*`) are resolved left-to-right with glob matching — not regex. Fast. Predictable.

---

**`URL_STRUCTURE_PATTERNS`**

Ordered list of `(regex_pattern, topology_class)` tuples. Evaluated in order. First match wins.

```python
URL_STRUCTURE_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"/api/v\d+/"),            "REST_API_JSON"),
    (re.compile(r"/api/"),                 "REST_API_JSON"),
    (re.compile(r"\.json$"),               "REST_API_JSON"),
    (re.compile(r"/products?/\d+"),        "ECOMMERCE_PRODUCT"),
    (re.compile(r"/wiki/"),                "WIKIPEDIA_ARTICLE"),
    (re.compile(r"/docs?/"),               "SAAS_DOCS"),
    (re.compile(r"/blog/"),                "BLOG_POST"),
    (re.compile(r"/news/"),                "NEWS_ARTICLE"),
    (re.compile(r"/article/"),             "NEWS_ARTICLE"),
    (re.compile(r"/forum/"),               "FORUM_THREAD"),
    (re.compile(r"/thread/"),              "FORUM_THREAD"),
    (re.compile(r"/login|/signin|/auth"),  "AUTH_REDIRECT"),
]
```

---

**`HEADER_SIGNALS`**

Dict of `header_name → (header_value_pattern, topology_class)`. Checked after domain + URL. Handles content-type, cloudflare, paywalls.

```python
HEADER_SIGNALS: List[Tuple[str, str, str]] = [
    # (header_key, value_pattern, topology_class)
    ("content-type",     "application/json",      "REST_API_JSON"),
    ("content-type",     "application/ld+json",   "JSON_LD_STRUCTURED"),
    ("cf-ray",           "*",                     "CLOUDFLARE_CHALLENGE"),  # presence check
    ("x-robots-tag",     "noindex",               "NEWS_ARTICLE_PAYWALLED"),
    ("x-frame-options",  "deny",                  "AUTH_REDIRECT"),
    ("location",         "*",                     "AUTH_REDIRECT"),          # 3xx with Location
]
```

---

**`HARD_OVERRIDE_CLASSES`**

These three classes bypass everything, including the ML path. Detected in headers or content window. Once set, classification is done.

```python
HARD_OVERRIDE_CLASSES: FrozenSet[str] = frozenset({
    "AUTH_REDIRECT",
    "CLOUDFLARE_CHALLENGE",
    "RATE_LIMITED",
})
```

Rate-limited detection: HTTP 429 or `Retry-After` header present.  
Cloudflare detection: `cf-ray` header or known Cloudflare challenge strings in content window.  
Auth redirect: HTTP 301/302 with `/login` / `/signin` destination, or content window contains login form markers.

---

**`PARENT_CLASS_MAP`**

Recursive fallback chain. Used when a subclass has no recipe. Terminates at `GENERIC_HTML`. Maximum depth = 3.

```python
PARENT_CLASS_MAP: Dict[str, str] = {
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
    "JSON_LD_STRUCTURED":        "GENERIC_HTML",
}
# every path in this map terminates at GENERIC_HTML within 3 hops
```

---

### Signal Path Methods (Sonnet)

**`_classify_by_domain(url: str) → Optional[Tuple[str, float]]`**

Extracts domain + path from URL. Checks `DOMAIN_FINGERPRINT_TABLE` in order: exact domain match → wildcard match → None. Returns `(topology_class, 0.99)` on hit, `None` on miss. Confidence is always 0.99 for fingerprint hits — these are ground truth.

---

**`_classify_by_url(url: str) → Optional[Tuple[str, float]]`**

Runs URL path through `URL_STRUCTURE_PATTERNS` in order. First match wins. Returns `(topology_class, 0.85)` on hit (URL structure is reliable but not infallible), `None` on miss.

---

**`_classify_by_headers(headers: Dict[str, str], response_code: int) → Optional[Tuple[str, float]]`**

Checks `HEADER_SIGNALS`. Also checks response code directly: 429 → `RATE_LIMITED`. 301/302 with auth-pattern destination → `AUTH_REDIRECT`. Returns `(topology_class, 0.95)` on hit, `None` on miss.

Hard overrides checked here first: if result is in `HARD_OVERRIDE_CLASSES`, return immediately without checking content window or ML path.

---

**`_classify_by_window(content_prefix: bytes) → Optional[Tuple[str, float]]`**

Decodes `content_prefix` as UTF-8 (errors=replace). Runs ordered grep passes:

```python
WINDOW_PATTERNS: List[Tuple[str, str, float]] = [
    # (grep_string, topology_class, confidence)
    ('application/ld+json',       "JSON_LD_STRUCTURED",    0.90),
    ('data-product-id',           "ECOMMERCE_PRODUCT",     0.85),
    ('"@type": "Product"',        "ECOMMERCE_PRODUCT",     0.90),
    ('<article',                  "NEWS_ARTICLE",           0.70),
    ('class="post-content"',      "BLOG_POST",             0.75),
    ('id="mw-content-text"',      "WIKIPEDIA_ARTICLE",     0.95),
    ('class="forum-post"',        "FORUM_THREAD",          0.85),
    ('"version":',                "SAAS_DOCS_VERSIONED",   0.70),
    ('cf-browser-verification',   "CLOUDFLARE_CHALLENGE",  0.99),
    ('input[type="password"]',    "AUTH_REDIRECT",         0.80),
]
```

First match above threshold wins. Bounded to 4096 bytes — never reads further.

---

**`_load_classifier_model() → torch.nn.Module`**

Loads `store/topology_router.pt` at startup. Always `weights_only=True`. Raises `StoreCorruptionError` (from `exceptions.py`) if load fails. Moves model to eval mode immediately.

```python
def _load_classifier_model(self) -> torch.nn.Module:
    model = torch.load(
        self._model_path,
        weights_only=True,
        map_location="cuda" if torch.cuda.is_available() else "cpu"
    )
    model.eval()
    return model
```

---

**`_reload_classifier_model() → None`** (WATCHDOG callback)

Called by `store_watchdog.py` when `topology_router.pt` changes on disk. Loads new model then does atomic GIL-safe assignment. Never blocks `classify()`.

```python
async def _reload_classifier_model(self) -> None:
    new_model = self._load_classifier_model()
    self._model = new_model    # GIL-safe atomic assignment
```

---

**`classify(input: ClassifierInput) → TopologyClassification`** (Sonnet orchestrates)

Runs paths 1–4 in order. If any path returns confidence ≥ `THETA_CLASSIFY_CONFIDENT`, returns immediately. If hard override detected, returns immediately. Otherwise invokes Opus ML path. If ML path confidence < `THETA_CLASSIFY_FALLBACK`, emits warning and returns `GENERIC_HTML`.

```python
async def classify(self, input: ClassifierInput) -> TopologyClassification:
    # Path 1
    result = self._classify_by_domain(input.url)
    if result and result[1] >= THETA_CLASSIFY_CONFIDENT:
        return self._build_result(*result, ClassificationPath.DOMAIN_FINGERPRINT)

    # Path 2
    result = self._classify_by_url(input.url)
    if result and result[1] >= THETA_CLASSIFY_CONFIDENT:
        return self._build_result(*result, ClassificationPath.URL_STRUCTURE)

    # Path 3 (includes hard override check)
    result = self._classify_by_headers(input.headers, input.response_code)
    if result:
        if result[0] in HARD_OVERRIDE_CLASSES:
            return self._build_result(*result, ClassificationPath.HEADER_SIGNAL)
        if result[1] >= THETA_CLASSIFY_CONFIDENT:
            return self._build_result(*result, ClassificationPath.HEADER_SIGNAL)

    # Path 4
    result = self._classify_by_window(input.content_prefix)
    if result:
        if result[0] in HARD_OVERRIDE_CLASSES:
            return self._build_result(*result, ClassificationPath.CONTENT_WINDOW)
        if result[1] >= THETA_CLASSIFY_CONFIDENT:
            return self._build_result(*result, ClassificationPath.CONTENT_WINDOW)

    # Path 5 (Opus)
    topology_class, confidence = self._classify_via_model(
        self._embed_signals(input.url, input.headers, input.content_prefix, None)
    )
    if confidence < THETA_CLASSIFY_FALLBACK:
        logger.warning("ClassificationConfidenceTooLow: %.3f for %s", confidence, input.url)
        return self._build_result("GENERIC_HTML", confidence, ClassificationPath.FALLBACK)

    return self._build_result(topology_class, confidence, ClassificationPath.MODEL)
```

---

**`initialize() → None`** (Sonnet)

Called once at system startup. Registers WATCHDOG. Loads model. No bus subscription — classifier is synchronous and called directly.

```python
async def initialize(self) -> None:
    self._model = self._load_classifier_model()
    WATCHDOG.register(
        path=self._model_path,
        callback=self._reload_classifier_model,
        debounce_ms=500,
    )
```

---

## OPUS BUILDS

### `_embed_signals()`

Only invoked when paths 1–4 are all below `THETA_CLASSIFY_CONFIDENT`. Converts the four signal streams into a feature tensor for `topology_router.pt`. Opus designs the full feature engineering. The feature space must be:

- **Consistent across restarts** — same input always produces same vector
- **Meaningful to the MLP** — the learned weights of `topology_router.pt` depend on this feature space remaining stable
- **Bounded** — fixed-width tensor regardless of input length

```python
def _embed_signals(
    self,
    url: str,
    headers: Dict[str, str],
    content_prefix: bytes,
    domain_hint: Optional[str],
) -> torch.Tensor:
    """
    Opus builds this.

    Design requirements:
    - Fixed-width output tensor (model input_dim must match)
    - Deterministic: same input → same output, no randomness
    - Encodes: URL path tokens, header presence/value flags, content window n-grams,
      HTTP status signal, partial domain match signal, response-code one-hot
    - No LLM. No external calls.
    - Must survive model reload without recomputation.
    """
    ...
```

Opus decides: exact feature dimensions, encoding strategy for URL tokens, header bitmask layout, content n-gram hashing scheme, and normalization. The output dimension must match `topology_router.pt`'s `input_dim` exactly.

---

### `_classify_via_model()`

Forward pass through `topology_router.pt`. Produces probability distribution over 18 topology classes. Returns argmax class + max probability as confidence.

```python
def _classify_via_model(
    self,
    features: torch.Tensor,
) -> Tuple[str, float]:
    """
    Opus builds this.

    Design requirements:
    - topology_router.pt forward pass (MLP, weights frozen at inference time)
    - Softmax over 18-class output layer
    - Returns (topology_class_string, confidence_float)
    - confidence is sigmoid-bounded [0.0, 1.0] — never raw logit
    - No gradient computation (torch.no_grad())
    - Thread-safe: model object must not be modified
    - If CUDA available, input tensor must be on same device as model
    """
    ...
```

The model is owned by `index_daemon.py` for training. The classifier only reads it. `weights_only=True` enforced on load. `torch.no_grad()` wrapped around all inference.

---

## Thresholds

Defined in `contracts.py`:

```python
THETA_CLASSIFY_CONFIDENT  = ConfidenceFloat(0.75)  # skip ML path if any deterministic path hits this
THETA_CLASSIFY_FALLBACK   = ConfidenceFloat(0.40)  # return GENERIC_HTML if ML path below this
```

These are not magic numbers. `0.75` was chosen because domain fingerprint hits at `0.99`, URL patterns at `0.85`, headers at `0.95` — all well above. Only ambiguous cases fall through to the model. `0.40` was chosen because below this, the model is effectively guessing and `GENERIC_HTML` is a safer default than a wrong recipe.

---

## 18 Topology Classes

```
NEWS_ARTICLE            NEWS_ARTICLE_PAYWALLED
SAAS_DOCS               SAAS_DOCS_VERSIONED         SAAS_DOCS_WITH_CODE
REST_API_JSON           REST_API_JSON_PAGINATED
JSON_LD_STRUCTURED
ECOMMERCE_PRODUCT       ECOMMERCE_PRODUCT_VARIANT
FORUM_THREAD
BLOG_POST
WIKIPEDIA_ARTICLE
LANDING_PAGE
AUTH_REDIRECT           CLOUDFLARE_CHALLENGE         RATE_LIMITED
GENERIC_HTML
```

All 18 are members of the `TopologyClass` enum in `contracts.py`.

---

## Performance Targets

| Path | Target | Notes |
|---|---|---|
| Path 1 — Domain fingerprint | < 0.1ms | Dict lookup + glob match |
| Path 2 — URL structure | < 1ms | Precompiled regex |
| Path 3 — Headers | < 1ms | Key lookup + value compare |
| Path 4 — Content window | < 5ms | Bounded to 4096 bytes |
| Path 5 — ML model | < 15ms | GPU forward pass |
| **Known domains (warm)** | **> 75% skip ML** | Domain table hit rate target |

The classifier is called once per URL, before any fetching. At 50K links/run, total classification budget is 250ms (5µs average). Paths 1–4 comfortably achieve this. The ML path is rare — only for genuinely ambiguous URLs with no structural signals.

---

## Write Order

Write in this exact sequence. Each function depends only on what's above it.

```
1.   DOMAIN_FINGERPRINT_TABLE constant         Sonnet
2.   URL_STRUCTURE_PATTERNS constant           Sonnet
3.   HEADER_SIGNALS constant                   Sonnet
4.   HARD_OVERRIDE_CLASSES constant            Sonnet
5.   WINDOW_PATTERNS constant                  Sonnet
6.   PARENT_CLASS_MAP constant                 Sonnet
7.   _load_classifier_model()                  Sonnet
8.   _reload_classifier_model()                Sonnet
9.   _classify_by_domain()                     Sonnet
10.  _classify_by_url()                        Sonnet
11.  _classify_by_headers()                    Sonnet
12.  _classify_by_window()                     Sonnet
13.  _embed_signals()                          OPUS
14.  _classify_via_model()                     OPUS
15.  _build_result()                           Sonnet  (helper, wraps dataclass construction)
16.  classify()                                Sonnet  (orchestrates 9–14)
17.  initialize()                              Sonnet  (watchdog + model load)
```

Steps 13–14 are the Opus handoff. Sonnet writes everything else, then hands off a clean stub with the exact method signatures above. Opus fills `_embed_signals()` and `_classify_via_model()` only — nothing else.

---

## Non-Negotiable Rules

1. **Classifier never fetches.** URL + headers + 4KB window is the full budget. No HTTP calls. No disk reads except `topology_router.pt` at startup.
2. **Hard overrides fire before ML.** `AUTH_REDIRECT`, `CLOUDFLARE_CHALLENGE`, `RATE_LIMITED` are checked in paths 3 and 4. They never reach path 5.
3. **`weights_only=True` on all `torch.load()` calls.** No exceptions. This is a security requirement.
4. **Never returns `None`.** Fallback is `GENERIC_HTML` with whatever confidence the model returned. The caller always gets a `TopologyClassification`.
5. **WATCHDOG reload is atomic.** `self._model = new_model` is a single assignment. GIL protects it. No locking needed. No `classify()` call will see a half-loaded model.
6. **Paths 1–4 are fully deterministic.** No model. No randomness. Same input always produces same output. Tests can assert exact return values.
7. **ML path (Opus) is only invoked when all deterministic paths are below `THETA_CLASSIFY_CONFIDENT`.** Paths 1–4 must all be checked before the model fires.
8. **`torch.no_grad()` wraps all inference.** The classifier never triggers gradient computation.
9. **Feature vector must be stable across restarts.** `_embed_signals()` is a pure function. Same input → same output every time. Model weights are trained against this contract.
10. **`PARENT_CLASS_MAP` fallback depth ≤ 3.** If recipe is missing for a class, walk the map. Terminate at `GENERIC_HTML`. Never infinite loop.

---

## Success Criteria

- All 18 topology classes correctly classified from fixture URLs in test suite
- Domain fingerprint path hits > 95% confidence on known domains
- Hard overrides tested: Cloudflare challenge page, 429 response, login redirect all return correct class before ML path is invoked
- WATCHDOG reload test: model file swapped on disk, new classifications reflect new weights without restart
- `GENERIC_HTML` fallback fires correctly when model confidence < 0.40
- PARENT_CLASS_MAP tested: `SAAS_DOCS_WITH_CODE` falls back to `SAAS_DOCS` if no recipe; `FORUM_THREAD` falls to `BLOG_POST` then `NEWS_ARTICLE`
- Zero HTTP calls in classifier (verified via mock in tests)
- `classify()` never raises — all exceptions caught, fallback returned
- Cold path (all 4 deterministic paths miss) covered in tests
- Hot path (domain fingerprint hit) covered in tests
- Performance: 10K sequential `classify()` calls < 5s on CPU (no GPU required for paths 1–4)
