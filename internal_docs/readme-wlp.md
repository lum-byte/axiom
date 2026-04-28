# World Latent Parser (WLP) — Complete Engineering Specification
**AXIOM TAG Intelligence Layer — Street Level**
**Classification: AXIOM INTERNAL // DO NOT SURFACE**
**Builder: Opus for latent_parser.py — entirely. Sonnet for wlp_graph.py and wlp_zones.py.**

---

## Preamble — Read This Before Anything Else

This document is the single source of truth for:

```
tag/world_model/wlp_graph.py       — Sonnet
tag/world_model/wlp_zones.py       — Sonnet
tag/world_model/latent_parser.py   — Opus
```

It is written for both builders. Every design decision is explained with full
reasoning. Every interface is specified with exact types. Every edge case is
addressed. There are no gaps. If something is not in this document it does not
belong in any of these three files.

**Read the WLM readme before reading this.**
Not because WLP depends on WLM — it does not, ever — but because understanding
what WLM does from above makes it clear what WLP must do from within. The WLM
tells AXIOM how to approach a topology class. The WLP tells AXIOM where exactly
to look once it arrives. Two different questions. Two different architectures.
Same moment in time — asyncio.gather() runs them simultaneously.

**The WLP is the precision layer.**
The WLM can be slightly wrong and the system degrades gracefully — it fetches
the right page with slightly wrong timeout settings. The WLP cannot be slightly
wrong — a wrong ZoneMap compiles a wrong recipe, a wrong recipe runs on the
wrong DOM nodes, the extraction produces noise instead of signal, and that
failure is silent. The kernel runs. Bytes come out. They are the wrong bytes.
Nobody upstream knows. The damage propagates quietly.

This is not the place for approximations. Every threshold justified.
Every edge case handled. Every failure path returns a valid value.
Do not rush these files. The ZoneMap is the foundation every recipe
is compiled from. Get it right once.

---

## File Division — Why Three Files

WLM needed four files because its concerns were each large enough and
distinct enough to warrant complete isolation. WLP has the same property
but collapses the model file and orchestration file into one — the
LatentParser nn.Module is standard PyG SAGEConv layers, not a custom SSM
architecture. It does not need its own file.

```
wlp_graph.py    The input processing layer.
                Tree-sitter parses HTML/JSON/JS/CSS → CST.
                CST is converted to PyTorch Geometric graph.
                ANTLR4 fallback for custom DSL topologies.
                Node feature vectors assembled here.
                This file does NOT know about ZoneMaps.
                This file does NOT know about GraphSAGE.
                It knows about parse trees and graph construction.
                Nothing else.

wlp_zones.py    The output processing layer.
                ZoneMap and all related dataclasses defined here.
                Node classification tensors → ZoneMap assembly.
                Intent vector → zone weight conditioning.
                ZoneDescriptor, BoundaryDescriptor, ExtractionStrategy.
                This file does NOT know about Tree-sitter.
                This file does NOT know about GraphSAGE.
                It knows about structural zone representation.
                Nothing else.

latent_parser.py The model and orchestration layer.
                LatentParser nn.Module with SAGEConv layers.
                WorldLatentParser public class.
                query() — the only public method outside this directory.
                Bus subscriptions, watchdog registration.
                Cache management L1/L2/L3.
                discover_signal_zones() for unknown topology.
                This file imports wlp_graph.py and wlp_zones.py.
                Neither of them imports from here.
                Dependency direction is strict and one-way.
```

The mirror to WLM's structure:

```
wlm_tokenizer.py  →  wlp_graph.py      input processing, Sonnet builds
wlm_decoders.py   →  wlp_zones.py      output processing, Sonnet builds
mamba_router.py   ↘
                    latent_parser.py    model + orchestration, Opus builds
latent_model.py   ↗
```

---

## What The WLP Is

The World Latent Parser is a **Tree-sitter + GraphSAGE** system that processes
the structural topology of individual pages. It is the street level of AXIOM's
world model — it sees DOM nodes, structural boundaries, signal zones, and noise
patterns from within a single page, not from above.

It is a **cartographer**. Not a classifier.

Every other system in the world asks: "what is this page about?"
The WLP asks: "what is the shape of this page?" — and draws the map.

The map it draws is a ZoneMap. The ZoneMap tells topology/parser.py exactly
where signal lives — down to the CSS selector level. The recipe compiler
translates that structural description into grep/sed/awk commands that extract
precisely what needs extracting and nothing else.

It answers one question after every fetch:

```
Given the structural topology of this specific page,
where exactly does signal live,
where is the noise,
and what is the structural boundary between them?
```

**The WLP is a cartographer building a structural map of every page it sees.**
The map gets more accurate with every crawl cycle. Not because someone labeled
training data. Because extraction quality feedback tells it where it was wrong.
Every SurpriseEvent is a GPS ping correcting the map. Over time the map is
precise enough that the recipe compiler never has to guess.

---

## What The WLP Is Not

```
Not the WLM.
    The WLM sees domains over time as sequences. Temporal. Stateful.
    The WLP sees one page's structure at a time. Structural. Stateless.
    They share no architecture, no code, no imports.
    WLP never imports from latent_model.py. Not once. Not ever.
    WLM never imports from latent_parser.py. Same rule.
    They coordinate only via asyncio.gather() at the caller level.
    That is the entirety of their relationship.

Not a scraper.
    The WLP never fetches pages. Never. Under any condition.
    It receives CleanSignalEvent from the bus.
    The content was already fetched by phantom.py.
    The content was already stripped by signal_kernel/.
    The WLP sees the clean output, not the raw page.
    It never calls phantom.py. It never makes HTTP requests.
    This boundary is absolute and non-negotiable.

Not a classifier.
    The classifier assigns topology class from URL + headers + 4KB window.
    The WLP receives an already-classified page and finds zones within it.
    It does not reassign topology class.
    It does not override the classifier's output.
    If the page is classified as SAAS_DOCS, it is SAAS_DOCS.
    WLP operates within that classification, not around it.

Not a recipe compiler.
    The WLP produces a ZoneMap — a structural description.
    topology/parser.py compiles that description into shell commands.
    The WLP never generates grep, sed, or awk patterns.
    It never knows what commands will be used.
    Structural description and shell compilation are entirely separate.
    The boundary between wlp_zones.py and topology/parser.py is total.

Not a validator.
    sanitizer.py does last-mile validation and strip.
    The WLP's job ends when it emits ZoneMapUpdatedEvent.
    Whether the recipe that compiled from its ZoneMap produces
    clean output is not the WLP's concern.
    It maps. It does not validate the map's consequences.

Not a database.
    ZoneMaps are written to structural_layer.pt.
    The WLP writes them. It does not own the store.
    store_watchdog.py notifies the WLP when structural_layer.pt changes.
    The WLP reloads when notified. It does not poll. It does not manage TTL.
    The store is shared — WLM reads source_matrix from the same file.
    WLP owns zone_knowledge in that file. Nothing else.
```

---

## What The WLP Should Be

```
A structural cartographer.
    Builds precise ZoneMaps from parse tree analysis.
    Maps accumulate accuracy over time through extraction feedback.
    Every page seen is a data point correcting the map.

Inductive.
    GraphSAGE learns structural patterns that generalize to unseen pages.
    A new page arrives: classify its nodes from learned structure.
    No retraining required. No full graph recomputation required.
    The model generalizes because it learned node classification,
    not graph memorization.

Intent-aware from day one.
    Intent conditioning is not an afterthought.
    Every ZoneMap has intent_weights.
    wlp_zones.py builds with_intent() as a core method.
    The recipe compiler receives intent-weighted zones from the start.
    This is the feature that makes AXIOM's extraction genuinely novel.
    Do not defer it. Wire it in at construction.

Gracefully degrading.
    Tree-sitter fails → ANTLR4 fallback.
    ANTLR4 fails → discover_signal_zones() heuristics.
    Heuristics fail → EmptyZoneMap with confidence 0.0.
    EmptyZoneMap → topology/parser.py uses hardcoded recipe.
    At no point does the system stop. At no point does query() raise.
    Every failure path has a defined, valid output.

Fast on the critical path.
    L1 cache hit: <0.5ms. This is most queries on a warm system.
    L2 cache hit: <1ms.
    L3 fresh parse: <15ms even on complex pages.
    with_intent() on a cached ZoneMap: <0.1ms — O(1) arithmetic.
    A new intent_vector never triggers a new GraphSAGE forward pass.
    Intent conditioning is free at cache tier.

Confirmation-building.
    A ZoneMap seen once has confidence capped at 0.70.
    A ZoneMap confirmed by 10+ high-quality extractions has confidence 0.90+.
    Confidence is earned through repeated correct prediction.
    Not assigned. Not configured. Earned.
```

---

## What The WLP Should Not Be

```
Should not be monolithic.
    Three files for three concerns. Not one file for everything.
    wlp_graph.py does not know about ZoneMaps.
    wlp_zones.py does not know about Tree-sitter.
    latent_parser.py does not inline parsing or zone assembly.
    It calls the right file for each concern. That is all.

Should not be approximate about zone boundaries.
    A zone boundary that is one node off compiles a recipe
    that captures one node of noise in every extraction.
    Multiplied across millions of crawl cycles this is
    significant noise accumulation. Zone boundaries are exact
    or the system degrades silently and permanently.

Should not silently succeed on bad input.
    Malformed HTML is not an error — Tree-sitter handles it.
    A page with 80% error nodes is structurally significant.
    The error recovery pattern IS the structure.
    Log the error rate. Proceed with what was parsed.
    Never pretend malformed HTML is well-formed.

Should not cache PyG graphs.
    PyG Data objects are large and not serializable cleanly.
    Only ZoneMaps are cached and persisted.
    After classification the PyG graph is garbage collected.
    Memory budget is for active inference, not graph storage.

Should not share tree_sitter.Parser instances across coroutines.
    tree_sitter.Parser is not thread-safe.
    One parser instance per async task.
    Parser pool or per-call instantiation.
    This is not optional — shared parser instances corrupt CSTs.

Should not retrain at query time.
    forward() with gradient flow is preparse_daemon.py's domain.
    At query time: frozen weights, torch.no_grad(), readout only.
    Training and inference share the same model file.
    They do not share the same code path. Ever.

Should not claim confidence above 0.70 for discovered zones.
    discover_signal_zones() operates on unknown structure.
    Unknown structure = uncertain classification.
    Confidence ceiling 0.70 for discovery mode.
    Confidence above 0.70 requires confirmed extractions.
    Confidence is not a free parameter to inflate.
```

---

## The Complexity — Why This Is Not Trivial

WLP looks simpler than WLM on the surface. Mamba SSM is a custom
architecture. GraphSAGE is standard PyG. Tree-sitter is a mature library.
This is misleading. The WLP's complexity is not architectural — it is
operational. The hard part is not the model. The hard part is everything
around it.

```
Parsing complexity:
    Four Tree-sitter grammars active simultaneously on SAAS_DOCS_WITH_CODE.
    HTML parser handles malformed input gracefully — error nodes are signal.
    JavaScript parsing inside HTML — embedded grammar handling required.
    CSS parsing for inline styles that define structural zones.
    ANTLR4 fallback requires grammar inference — the input is unknown,
    the grammar that matches it must be discovered from structure.
    Subgraph sampling for pages with >50,000 DOM nodes.
    Every parse must be deterministic — same bytes, same CST, always.

Graph construction complexity:
    Three edge types: PARENT_CHILD, SIBLING, SKIP_SIBLING.
    SKIP_SIBLING edges require two-pass construction.
    Bidirectional edges double the edge count.
    Variable node count per page — batching requires careful padding.
    Node feature assembly: 128 dimensions per node, 5 feature groups.
    Intent bias dimensions (28 of 128) require intent vector projection
    before graph construction — intent is baked into node features.

Zone assembly complexity:
    GraphSAGE produces per-node logits: (n_nodes, 3).
    Converting logits to ZoneDescriptors requires:
        argmax classification per node
        confidence score per node
        grouping adjacent SIGNAL nodes into zones
        computing CSS selectors from node positions in CST
        determining scope (parent selector) for each zone
        computing content_type from node feature patterns
        ordering zones by priority
    CSS selector generation from CST node position is not trivial.
    The selector must be specific enough to isolate the zone,
    general enough to work on similar pages of the same topology class.

Intent conditioning complexity:
    Intent weights applied at zone level, not node level.
    Zone content_type must match intent semantics — not keyword matching,
    structural inference from content_type and node feature patterns.
    exclude zones get weight 0.0 — exclude overrides all other weights.
    with_intent() must be O(1) — no model forward pass, pure arithmetic.
    The same ZoneMap must produce different weights for different intents
    without recomputing the underlying classification.

discover_signal_zones() complexity:
    Called for GENERIC_HTML and dissolved ZoneMaps.
    Three-pass analysis: heuristics + GraphSAGE + confidence-weighted merge.
    Heuristic and model may disagree — merge strategy must be principled.
    Result has confidence ceiling 0.70 — never claimed more certain.
    After 10+ confirmed extractions: may trigger topology subclass creation.
    The subclass creation is a ZoneMapUpdatedEvent with subclass_candidate flag.
    topology/parser.py decides whether to promote. WLP does not promote itself.

Concurrency complexity:
    tree_sitter.Parser is not thread-safe.
    Multiple simultaneous wlp.query() calls from asyncio event loop.
    Parser pool required. Pool management adds lifecycle complexity.
    structural_layer.pt writes must be atomic — staging + rename.
    Multiple async tasks may attempt simultaneous writes.
    Write serialization required. asyncio.Lock on write path.

Wikipedia preparse complexity:
    6.7M articles. 6.7M CST graphs. 6.7M forward passes.
    Estimated 30 minutes on RTX 5080.
    Batch processing with GPU parallelism.
    Ground truth labels from Wikipedia's known structure.
    Not labeled by humans — derived from structural conventions
    that are consistent across all Wikipedia articles.
    Training labels must be correct. A mislabeled Wikipedia
    infobox as SIGNAL instead of NOISE trains wrong patterns
    that propagate to all pages using infobox-like structure.
```

**Precision is not optional here. It is the product.**

The 99.27% compression numbers from signal_kernel/ are downstream
of a correct ZoneMap. A recipe compiled from a correct ZoneMap
extracts 8.4KB from 1.1MB. A recipe compiled from a wrong ZoneMap
extracts 8.4KB of noise from 1.1MB of signal. Same numbers.
Different outcome. The WLP is where that distinction is made.

---

## Why GraphSAGE — Full Justification

### Why Not BeautifulSoup or lxml

```
BeautifulSoup:
    Permissive HTML parser — accepts malformed HTML and guesses structure.
    Parser corrections are invisible to downstream analysis.
    WLP needs the actual structure including errors.
    A malformed page that renders strangely IS structurally significant.
    BeautifulSoup normalizes that away. WLP cannot afford normalization.

lxml:
    Same fundamental problem. Parser corrections invisible.
    No grammar-based CST. Only a corrected DOM tree.
    Faster than BeautifulSoup. Still wrong for this purpose.

Tree-sitter:
    Grammar-based. Produces Concrete Syntax Tree.
    Error nodes are first-class. Malformed structure is exposed, not hidden.
    Deterministic: same bytes, same CST. Always.
    Required for reproducible ZoneMap generation.
    Four grammars in one interface: HTML, JSON, JavaScript, CSS.
```

### Why Not Transformer for Node Classification

```
Transformer:
    Self-attention is O(n²) in sequence length.
    DOM nodes are the sequence. Complex pages have thousands of nodes.
    O(n²) on thousands of nodes, millions of pages = not viable.
    No inductive bias toward graph structure.
    DOM IS a graph. Transformer treats nodes as sequences.
    Wrong inductive bias for structural zone detection.
```

### Why Not GCN

```
GCN (Graph Convolutional Network):
    Transductive — requires all nodes present during training.
    WLP must generalize to pages it has never seen.
    A new page has new nodes not in the training graph.
    GCN cannot classify them without full graph retraining.
    Not viable for a system that crawls new pages continuously.
```

### Why GraphSAGE Is Exactly Right

```
Inductive learning:
    GraphSAGE learns a function that generalizes to unseen nodes.
    New page arrives: classify its nodes correctly.
    No retraining. No full graph recomputation.
    Exactly what WLP needs.

Neighborhood aggregation:
    Each node aggregates from k-hop neighbors.
    A DOM node's classification is informed by parent, children, siblings.
    Exactly the right inductive bias for structural zone detection.
    A nav element surrounded by other nav elements = NOISE with high confidence.
    A p element inside article surrounded by other p elements = SIGNAL with high confidence.

O(k) per node inference where k is neighborhood sample size.
    Scales to pages with thousands of nodes without quadratic cost.

Trained on 6.7M Wikipedia parse trees:
    Learns what SIGNAL, NOISE, BOUNDARY look like structurally
    across the largest consistently-structured web corpus available.
    Generalizes to non-Wikipedia pages because structural patterns
    are universal conventions across the web.
    Header + navigation + main content + footer = every page.
    GraphSAGE learns these patterns from Wikipedia and applies them everywhere.
```

---

## wlp_graph.py — Exact Specification

**Builder: Sonnet**
**LOC target: 1,600–1,900**

### What It Is

The input processing layer. Takes raw HTML/JSON/CSS/JavaScript bytes
and produces a PyTorch Geometric Data object that LatentParser can
classify. Owns all parsing logic. Owns all graph construction logic.
Owns all node feature vector assembly.

### What It Is Not

Not a model. No nn.Module. No parameters. No gradients.
Not a zone assembler. Does not know what SIGNAL, NOISE, BOUNDARY mean.
Does not know about ZoneMaps. Does not know about CSS selectors.
Produces a graph. What the graph means is not its concern.

### Grammar Management

```python
# Four grammars. Four parser instances (not shared across coroutines).
# All initialized at module load — grammar loading is expensive.

HTML_GRAMMAR    = tree_sitter.Language(GRAMMAR_PATH, "html")
JSON_GRAMMAR    = tree_sitter.Language(GRAMMAR_PATH, "json")
JS_GRAMMAR      = tree_sitter.Language(GRAMMAR_PATH, "javascript")
CSS_GRAMMAR     = tree_sitter.Language(GRAMMAR_PATH, "css")

# Parser pool — tree_sitter.Parser is NOT thread-safe
# Pool size = CPU count * 2 (parsers are CPU-bound during parsing)
# Each coroutine acquires a parser from the pool, uses it, releases it

class ParserPool:
    """
    Thread-safe pool of tree_sitter.Parser instances.
    Acquired via async context manager.
    Parser returned to pool after CST production.
    """
```

### Node Extraction Rules — Exact

```
From the CST, extract nodes as follows:

INCLUDE as graph nodes:
    All element nodes (div, p, article, section, h1-h6, etc.)
    Error nodes — malformed structure IS structural signal
    Document node — root always included

EXCLUDE from graph nodes:
    Text nodes — content encoded into parent's feature vector
    Comment nodes — stripped entirely
    Whitespace-only text nodes — discard

Node ordering:
    Depth-first, left-to-right traversal
    Node index in traversal order = node index in feature matrix
    Deterministic: same CST always produces same node ordering
```

### Node Feature Vector — 128 Dimensions Exact

```
[0:18]    topology class one-hot (18 dimensions)
          which of the 18 topology classes this page belongs to
          every node on the page shares this feature
          enables class-specific zone pattern learning

[18:36]   node type one-hot (18 dimensions)
          div, article, section, p, h1, h2, h3, h4, h5, h6,
          ul, ol, li, code, pre, table, a, span
          (nav, header, footer encoded in CSS class presence bits)

[36:52]   CSS class presence bits (16 dimensions)
          sidebar, nav, footer, header, content, main, article,
          code, pre, warning, note, callout, pricing, modal,
          overlay, advertisement

[52:60]   HTML attribute signals (8 dimensions)
          has_id, has_data_attr, has_aria_label, has_role,
          role_is_main, role_is_navigation,
          role_is_complementary, has_itemprop

[60:68]   structural position (8 dimensions)
          depth_normalized [0,1], siblings_normalized [0,1],
          children_normalized [0,1], text_density_normalized [0,1],
          link_density_normalized [0,1],
          is_first_child, is_last_child,
          has_only_text_children

[68:84]   content signals (16 dimensions)
          contains_code_block, contains_numbered_list,
          contains_table, contains_definition_list,
          text_length_bucket_0, text_length_bucket_1,    # 4 bits
          text_length_bucket_2, text_length_bucket_3,
          child_count_bucket_0, child_count_bucket_1,    # 4 bits
          child_count_bucket_2, child_count_bucket_3,
          contains_external_link, contains_anchor_link,
          is_empty_node, contains_only_whitespace

[84:100]  structural pattern signals (16 dimensions)
          matches_nav_pattern, matches_footer_pattern,
          matches_sidebar_pattern, matches_article_pattern,
          matches_api_schema_pattern, matches_code_example_pattern,
          matches_warning_pattern, matches_pricing_pattern,
          matches_breadcrumb_pattern, matches_toc_pattern,
          matches_cookie_banner_pattern, matches_chat_widget_pattern,
          matches_ad_pattern, matches_social_share_pattern,
          matches_modal_pattern, matches_paywall_pattern

[100:128] intent bias (28 dimensions)
          derived from intent_vector projection via nn.Linear(256, 28)
          shifts node features toward intent-relevant structural patterns
          intent.exclude → negative bias on excluded pattern dimensions
          intent.primary → positive bias on relevant pattern dimensions
          all zeros when intent_vector is None (no projection)
```

### Edge Construction — Three Edge Types

```python
def build_edges(nodes: List[CSTNode]) -> Tuple[Tensor, Tensor]:
    """
    Construct edge_index and edge_attr for PyG Data.

    Edge types:
        PARENT_CHILD = [1, 0, 0]  — containment hierarchy
        SIBLING      = [0, 1, 0]  — same-level adjacency
        SKIP_SIBLING = [0, 0, 1]  — two-hop sibling context

    All edges bidirectional.
    For PARENT_CHILD: add parent→child AND child→parent.
    For SIBLING: add prev→next AND next→prev.
    For SKIP_SIBLING: add node_i→node_i+2 AND node_i+2→node_i.

    Returns:
        edge_index: (2, n_edges) — COO sparse format
        edge_attr:  (n_edges, 3) — edge type one-hot
    """
```

### Subgraph Sampling

```
Pages with > 50,000 DOM nodes are subgraph-sampled before
constructing the PyG graph. Reason: memory constraint.
10,000 nodes = 5.1MB feature matrix. 50,000 = 25.6MB. Acceptable.
100,000 nodes = 51.2MB. Exceeds per-inference memory budget on 5080
when multiple concurrent queries are in flight.

Sampling strategy:
    Preserve all nodes within 3 hops of document root
    (these define the page's top-level structure)
    Random sample of remaining nodes, preserving
    SIGNAL-likely nodes (high text density, article-like classes)
    over NOISE-likely nodes (nav, sidebar, footer patterns)
    Result: representative subgraph of the structural zones
    that matter for ZoneMap production
```

### ANTLR4 Fallback

```python
def antlr4_fallback_parse(
    content: bytes,
    topology_class: str,
) -> Optional[Data]:
    """
    Called when Tree-sitter error node rate > 0.50.
    Attempts grammar inference and ANTLR4 parse.

    Grammar library:
        GRAMMAR_RST       — reStructuredText (Sphinx docs)
        GRAMMAR_ASCIIDOC  — AsciiDoc (enterprise docs)
        GRAMMAR_DOCBOOK   — DocBook XML (legacy enterprise)
        GRAMMAR_DITA      — DITA XML (enterprise standard)
        GRAMMAR_OPENAPI   — OpenAPI/Swagger YAML
        GRAMMAR_GRAPHQL   — GraphQL SDL

    Grammar inference:
        Examine first 2KB of content for grammar fingerprints.
        RST:      '.. ' directive markers, '=====' underlines
        AsciiDoc: '= ' title markers, '----' delimiters
        DocBook:  '<?xml' header + 'docbook' namespace
        DITA:     '<?xml' header + 'dita' DOCTYPE
        OpenAPI:  'openapi:' or 'swagger:' root key
        GraphQL:  'type ' + 'Query' or 'Mutation' + '{' pattern

    Returns:
        PyG Data object compatible with LatentParser.forward()
        if grammar resolved and parse succeeded
        None if grammar inference failed or ANTLR4 parse failed
        (caller falls through to discover_signal_zones() heuristics)
    """
```

### Full Function List

```
ParserPool class
    acquire() — async context manager
    release() — return parser to pool

parse_html() → tree_sitter.Tree
parse_json() → tree_sitter.Tree
parse_javascript() → tree_sitter.Tree
parse_css() → tree_sitter.Tree
    All: acquire parser from pool, parse, release, return tree

extract_nodes() → List[CSTNode]
    Filter: include elements + errors, exclude text + comments

assemble_node_features() → Tensor  # (n_nodes, 128)
    topology_class_onehot()
    node_type_onehot()
    css_class_bits()
    attribute_signals()
    structural_position()
    content_signals()
    structural_pattern_signals()
    apply_intent_bias()

build_edges() → Tuple[Tensor, Tensor]  # edge_index, edge_attr
    parent_child_edges()
    sibling_edges()
    skip_sibling_edges()
    concat_bidirectional()

subgraph_sample() → List[CSTNode]
    Reduce to <= 50,000 nodes preserving structural priority

cst_to_pyg_graph() → Data
    Main entry point. parse → extract → features → edges → Data

antlr4_fallback_parse() → Optional[Data]
    Grammar inference + ANTLR4 parse + Data construction

error_rate() → float
    Fraction of CST nodes that are error nodes
    Used to trigger ANTLR4 fallback (threshold: 0.50)
```

### LOC Breakdown

```
ParserPool class:                   ~80 loc
Grammar loading + constants:        ~40 loc
CSTNode dataclass:                  ~30 loc
parse_html/json/js/css():          ~80 loc
extract_nodes():                   ~100 loc
topology_class_onehot():            ~30 loc
node_type_onehot():                 ~40 loc
css_class_bits():                   ~80 loc
attribute_signals():                ~60 loc
structural_position():              ~80 loc
content_signals():                  ~80 loc
structural_pattern_signals():      ~120 loc
apply_intent_bias():                ~60 loc
assemble_node_features():           ~40 loc
parent_child_edges():               ~60 loc
sibling_edges():                    ~60 loc
skip_sibling_edges():               ~40 loc
build_edges():                      ~40 loc
subgraph_sample():                 ~100 loc
cst_to_pyg_graph():                 ~60 loc
antlr4_fallback_parse():           ~200 loc
error_rate():                       ~20 loc
docstrings + comments:             ~300 loc
                                  ────────
wlp_graph.py total:             ~1,680 loc
```

---

## wlp_zones.py — Exact Specification

**Builder: Sonnet**
**LOC target: 1,400–1,700**

### What It Is

The output processing layer. Takes node classification tensors from
LatentParser and produces ZoneMaps. Owns all zone-related dataclasses.
Owns intent conditioning logic. Owns zone assembly from classified nodes.

### What It Is Not

Not a model. No nn.Module. No forward pass. No gradients.
Not a parser. Does not know about Tree-sitter or ANTLR4.
Does not know about PyG graphs.
Receives tensors. Produces ZoneMaps. Nothing else.

### ZoneMap — Exact Specification

```python
@dataclass(frozen=True)
class ZoneMap:
    """
    Structural extraction map for a specific page.
    Immutable once produced. New ZoneMap replaces old — never mutates.
    Consumed by topology/parser.py to compile grep/sed/awk recipes.
    Stored in structural_layer.pt keyed by (domain, topology_class).
    """
    topology_class: str
    domain: str

    signal_zones:  Tuple[ZoneDescriptor, ...]
    noise_zones:   Tuple[ZoneDescriptor, ...]
    boundaries:    Tuple[BoundaryDescriptor, ...]

    extraction_strategy: ExtractionStrategy

    intent_weights: Tuple[Tuple[str, float], ...]
    # ((zone_selector, weight), ...)
    # weight 0.0 = suppress (intent.exclude hit)
    # weight 1.0 = default relevance
    # weight 2.0 = primary intent match

    confidence: float                  # [0.0, 1.0]
    node_count: int
    signal_node_count: int
    noise_node_count: int
    boundary_node_count: int

    version: int                       # monotonic, increments on update
    produced_at: float                 # time.monotonic()
    topology_router_version: int       # WLM version at production time

    def is_stale(self, current_router_version: int) -> bool:
        """
        True if WLM weights updated since this ZoneMap was produced.
        A stale ZoneMap may have incorrect intent projections —
        the intent projection in node features used the old WLM weights.
        Stale ZoneMaps trigger L3 refresh on next query.
        """
        return self.topology_router_version < current_router_version

    def with_intent(
        self,
        intent_vector: List[float],
    ) -> "ZoneMap":
        """
        Return new ZoneMap with intent-conditioned zone weights.
        DOES NOT modify self — ZoneMaps are immutable.
        DOES NOT require a new GraphSAGE forward pass.
        O(1) weight arithmetic on existing zone descriptors.

        Called on every cache hit when intent_vector is not None.
        This is the critical path operation for intent conditioning.
        Must complete in <0.1ms.
        """
```

### ZoneDescriptor — Exact Specification

```python
@dataclass(frozen=True)
class ZoneDescriptor:
    """
    Description of a single structural zone.
    topology/parser.py translates this into shell extraction commands.
    The translation is structural — not hardcoded per-site.
    """
    selector: str          # CSS selector (preferred) or XPath
    selector_type: str     # "css" or "xpath"
    scope: str             # parent selector containing this zone
    content_type: str      # "prose", "code", "list", "table", "mixed"
    average_depth: float   # average DOM depth of nodes in this zone
    density: float         # signal density: signal_chars/total_chars [0.0, 1.0]
    priority: int          # extraction order (lower = extract first)


@dataclass(frozen=True)
class BoundaryDescriptor:
    """
    Structural boundary marker.
    Recipe compiler uses these as sed range delimiters
    and awk capture state reset triggers.
    """
    selector: str
    boundary_type: str     # "SECTION_BOUNDARY", "CONTENT_BOUNDARY", "NOISE_BOUNDARY"
    delimiter_content: str # regex pattern matching this boundary's content


class ExtractionStrategy(enum.Enum):
    DEPTH_FIRST     = "depth_first"     # traverse depth-first within signal zones
    BREADTH_FIRST   = "breadth_first"   # wide pages with parallel signal columns
    SECTION_SCOPED  = "section_scoped"  # each boundary = independent extraction scope
    FLAT            = "flat"            # no nesting, top-level signal zones only


class EmptyZoneMap:
    """
    Returned when all classification paths fail.
    Never None. Never exception. Always this.
    topology/parser.py falls back to hardcoded recipe on empty.
    confidence = 0.0 signals to parser that fallback is needed.
    """
    confidence = 0.0
    signal_zones = ()
    noise_zones = ()
    boundaries = ()
```

### Zone Assembly — From Node Classifications to ZoneMap

```python
def assemble_zone_map(
    node_classifications: Tensor,    # (n_nodes, 3) logits from LatentParser
    node_confidences: Tensor,        # (n_nodes, 1) confidence scores
    cst_nodes: List[CSTNode],        # original nodes in traversal order
    topology_class: str,
    domain: str,
    intent_vector: Optional[List[float]],
    topology_router_version: int,
) -> ZoneMap:
    """
    Convert per-node logits into a structured ZoneMap.

    Steps:
        1. argmax per node → SIGNAL/NOISE/BOUNDARY label
        2. Group adjacent SIGNAL nodes into candidate zones
        3. Compute CSS selector for each zone from CST node position
        4. Determine scope (parent CSS selector) for each zone
        5. Compute content_type from node feature patterns
        6. Compute density from text content analysis
        7. Order zones by priority (DOM depth, signal density)
        8. Identify BOUNDARY nodes → BoundaryDescriptors
        9. Determine ExtractionStrategy from zone topology
        10. Apply intent_weights if intent_vector provided
        11. Compute overall confidence from node confidence scores

    CSS selector generation (step 3) is the hardest step.
    The selector must:
        Be specific enough to isolate this zone on this page
        Be general enough to work on similar pages of same topology class
    Strategy:
        Use element type + most discriminative class (not all classes)
        Scope to nearest landmark ancestor (article, main, [role=main])
        Avoid positional selectors (:nth-child) — too page-specific
        Prefer semantic selectors (article, .content, [role=main])
    """
```

### Intent-Conditioned Zone Weighting — Exact Rules

```python
def apply_intent_weights(
    zone_map: ZoneMap,
    intent_vector: List[float],
    intent_tags: IntentTags,  # parsed from intent_vector
) -> Tuple[Tuple[str, float], ...]:
    """
    Compute per-zone intent weights.

    Rules (applied in order, exclude overrides all):

    EXCLUDE CHECK (applied first, terminates weight computation):
        for each zone in signal_zones:
            for each tag in intent_tags.exclude:
                if zone_matches_intent_semantics(zone, tag):
                    weight = 0.0
                    break  # exclude is absolute

    PRIMARY BOOST (if not excluded):
        for each tag in intent_tags.primary:
            if zone_matches_intent_semantics(zone, tag):
                weight += 1.0  (additive across primary tags)

    SECONDARY BOOST (if not excluded):
        for each tag in intent_tags.secondary:
            if zone_matches_intent_semantics(zone, tag):
                weight += 0.5

    URGENCY MODIFIER:
        if intent_tags.urgency == "high":
            if zone.content_type contains warnings or callouts:
                weight += 0.5

    USER STATE MODIFIER:
        if intent_tags.user_state == "locked_out":
            if zone has numbered list structure and content_type == "list":
                weight += 0.8  (procedure prioritized for locked-out users)

    DEFAULT:
        weight = 1.0 for any zone not modified above

    Zone matching semantics — structural, not keyword:
        "account_recovery" → zones with list+code structure, warning/note classes
        "api_reference"    → zones with code+table structure, schema-like density
        "pricing"          → zones with table structure, currency/number density
        Matching reads zone.content_type, zone.density, zone.selector patterns.
        Never reads page text content. WLP does not read content. Structure only.
    """
```

### Full Function List

```
ZoneMap dataclass (frozen) + methods
ZoneDescriptor dataclass (frozen)
BoundaryDescriptor dataclass (frozen)
ExtractionStrategy enum
EmptyZoneMap class
EmptyZoneKnowledge class

assemble_zone_map()
    classify_nodes()              argmax + threshold
    group_signal_nodes()          adjacent SIGNAL → zones
    generate_css_selector()       CST position → CSS selector
    determine_scope()             nearest landmark ancestor
    infer_content_type()          node features → prose/code/list/table/mixed
    compute_density()             signal_chars / total_chars
    order_zones_by_priority()     depth + density ranking
    identify_boundaries()         BOUNDARY nodes → BoundaryDescriptors
    select_extraction_strategy()  zone topology → strategy enum
    compute_zone_confidence()     mean node confidence within zone

apply_intent_weights()
    parse_intent_tags()           intent_vector → IntentTags struct
    zone_matches_intent_semantics() structural matching, not keyword
    compute_exclude_set()         O(1) lookup on exclude tags

ZoneMap.with_intent()             O(1) weight arithmetic on cached zones
ZoneMap.is_stale()                topology_router_version comparison

validate_zone_map()               post-assembly invariant checks
    signal_zones is not None (may be empty tuple — that is valid)
    all selectors are valid CSS or XPath
    confidence in [0.0, 1.0]
    version is monotonically increasing
```

### LOC Breakdown

```
ZoneMap + ZoneDescriptor + BoundaryDescriptor:   ~160 loc
ExtractionStrategy enum:                          ~20 loc
EmptyZoneMap + EmptyZoneKnowledge:                ~60 loc
IntentTags dataclass:                             ~30 loc
classify_nodes():                                 ~60 loc
group_signal_nodes():                            ~100 loc
generate_css_selector():                         ~150 loc
determine_scope():                                ~60 loc
infer_content_type():                             ~80 loc
compute_density():                                ~40 loc
order_zones_by_priority():                        ~50 loc
identify_boundaries():                            ~80 loc
select_extraction_strategy():                     ~60 loc
compute_zone_confidence():                        ~40 loc
assemble_zone_map() orchestration:                ~80 loc
parse_intent_tags():                              ~60 loc
zone_matches_intent_semantics():                 ~120 loc
compute_exclude_set():                            ~40 loc
apply_intent_weights():                           ~80 loc
ZoneMap.with_intent():                            ~60 loc
ZoneMap.is_stale():                               ~10 loc
validate_zone_map():                              ~80 loc
docstrings + comments:                           ~250 loc
                                                ────────
wlp_zones.py total:                           ~1,570 loc
```

---

## latent_parser.py — Exact Specification

**Builder: Opus**
**LOC target: 1,700–2,100**

### What It Is

The model and orchestration layer. Contains LatentParser (the nn.Module)
and WorldLatentParser (the public class). Imports wlp_graph.py for
graph construction. Imports wlp_zones.py for ZoneMap assembly.
Owns the GraphSAGE architecture. Owns the cache. Owns bus subscriptions.
Owns watchdog registration. Owns discover_signal_zones().

### What It Is Not

Not a parser (wlp_graph.py parses).
Not a zone assembler (wlp_zones.py assembles).
Does not know about Tree-sitter internals.
Does not know about CSS selector generation.
Orchestrates. Does not inline what the other two files already own.

### LatentParser nn.Module — Exact Architecture

```python
class LatentParser(nn.Module):
    """
    GraphSAGE node classification model.

    Three SAGEConv layers:
        Layer 1: local structure (parent/child/sibling)
        Layer 2: section structure (div/article/section level)
        Layer 3: page structure (body-level layout)

    Input:  PyG Data from wlp_graph.cst_to_pyg_graph()
    Output: per-node logits (n_nodes, 3) + confidences (n_nodes, 1)

    Training: preparse_daemon.py calls forward() with gradient flow
    Inference: WorldLatentParser calls readout() with torch.no_grad()
    """

    SIGNAL   = 0
    NOISE    = 1
    BOUNDARY = 2

    def __init__(self, config: WLPConfig) -> None:
        super().__init__()

        self.conv1 = SAGEConv(config.node_feature_dim, config.hidden_dim)
        self.conv2 = SAGEConv(config.hidden_dim, config.hidden_dim)
        self.conv3 = SAGEConv(config.hidden_dim, config.hidden_dim)

        self.dropout = nn.Dropout(config.dropout)

        # Node classification head: hidden_dim → 3 (SIGNAL/NOISE/BOUNDARY)
        self.classifier = nn.Linear(config.hidden_dim, 3)

        # Boundary type head: hidden_dim → 3 (SECTION/CONTENT/NOISE boundary)
        self.boundary_head = nn.Linear(config.hidden_dim, 3)

        # Per-node confidence head: hidden_dim → 1 sigmoid
        self.confidence_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, data: Data) -> Tuple[Tensor, Tensor]:
        """
        Full forward pass with gradient flow.
        Called ONLY by preparse_daemon.py during training.
        Never called from query path.
        Returns (logits (n_nodes, 3), confidences (n_nodes, 1))
        """

    @torch.no_grad()
    def readout(self, data: Data) -> Tuple[Tensor, Tensor]:
        """
        Inference forward pass. No gradient flow. Frozen weights.
        Called from WorldLatentParser._l3_fresh_parse().
        Returns (logits (n_nodes, 3), confidences (n_nodes, 1))
        """
```

### WorldLatentParser — The Public Class

```python
class WorldLatentParser:
    """
    The only public-facing component in tag/world_model/ for WLP.
    
    Called via asyncio.gather() in parallel with WorldLatentModel.
    Never imports from latent_model.py.
    Never sequential with WLM. Never blocking WLM.
    
    Single public method:
    
        async def query(
            topology_class: str,
            domain: str,
            intent_vector: Optional[List[float]] = None,
            phase: int = PHASE_I,
        ) -> ZoneMap
    
    Always returns ZoneMap. Never None. Never raises.
    EmptyZoneMap is the correct failure value.
    """
```

### Three-Tier Cache

```
L1: (domain, topology_class) → ZoneMap
    Key:   exact domain + topology class
    Value: most recent ZoneMap for this domain
    TTL:   mirrors WLM CACHE_TTL_BY_CLASS
    Why:   same domain + same class = same structural layout
           the DOM structure of stripe.com/docs does not change hourly

L2: topology_class → ZoneMap (generalized)
    Key:   topology class only
    Value: ZoneMap from most-confident domain in this class
    TTL:   2× domain TTL (class structure more stable than per-domain)
    Why:   new domain in known class still has predictable zone structure
           SAAS_DOCS always has nav + main content + code blocks
           selectors may differ but zone structure is consistent

L3: fresh parse
    No key. No TTL. Full pipeline:
        wlp_graph.cst_to_pyg_graph() → Data
        LatentParser.readout() → logits + confidences
        wlp_zones.assemble_zone_map() → ZoneMap
    Expected frequency: <15% queries on warm system
    Slowest: <15ms on 5080 for complex pages

Intent conditioning at every tier:
    L1 hit + intent: ZoneMap.with_intent() → O(1) weight update
    L2 hit + intent: ZoneMap.with_intent() → O(1) weight update
    L3 + intent: intent applied during assemble_zone_map() call
    A new intent_vector never forces L3.
    Intent is free at any cache tier.

Stale ZoneMap detection:
    ZoneMap.is_stale(current_topology_router_version)
    If WLM weights updated since ZoneMap production:
        intent projection in node features used old WLM weights
        ZoneMap may have subtly wrong intent-conditioned features
        Force L3 refresh on next query
    This is rare — WLM weights update on index_daemon cycles
    not on every query
```

### discover_signal_zones() — Three-Pass Analysis

```python
async def discover_signal_zones(
    cst_graph: Data,
    domain: str,
    raw_selector_hints: Optional[List[str]] = None,
) -> ZoneMap:
    """
    Auto-discover zones for unknown page structure.
    Called when:
        topology_class == GENERIC_HTML
        ZoneMap dissolved by SurpriseEvent
        ZoneMap not in L1 or L2 cache

    Pass 1 — Structural heuristics (no ML):
        High text density nodes → candidate SIGNAL
        header/footer/nav structural positions → NOISE
        H1-H6 at depth <= 3 → candidate BOUNDARY
        Base confidence: 0.60 for heuristic zones

    Pass 2 — GraphSAGE on GENERIC_HTML class:
        GENERIC_HTML is a trained topology class in LatentParser.
        It learned from all pages that fell through to generic.
        Classifications are coarser but directionally correct.
        Confidence from model's confidence_head output.

    Pass 3 — Confidence-weighted merge:
        If heuristic and model agree on a zone: confidence boost
        If they disagree: keep heuristic (more reliable for unknown)
        Final confidence ceiling: 0.70
        discovery_mode flag set in extraction_strategy metadata

    Subclass promotion trigger:
        After 10+ confirmed high-quality extractions from this domain
        using this discovered ZoneMap:
            emit ZoneMapUpdatedEvent with subclass_candidate=True
            topology/parser.py decides whether to promote
            WLP does not promote. WLP signals candidacy. That is all.

    Returns:
        ZoneMap with confidence <= 0.70
        Never confidence > 0.70 for discovered zones
        Never None
    """
```

### Bus Handlers — Exact Behavior

```python
async def _on_clean_signal(self, event: CleanSignalEvent) -> None:
    """
    Background task. Never blocks query().
    Spawned with asyncio.create_task() — fire and forget.

    Process:
        1. Parse content with wlp_graph.cst_to_pyg_graph()
        2. LatentParser.readout() → node classifications
        3. wlp_zones.assemble_zone_map() → new ZoneMap
        4. Compare new ZoneMap to stored ZoneMap for this domain
        5a. If different:
                Update structural_layer.pt via staging + atomic rename
                Invalidate L1 cache for (domain, topology_class)
                Emit ZoneMapUpdatedEvent → topology/parser.py recompiles
        5b. If identical:
                Increment confirmation counter for this ZoneMap
                If confirmations >= 10 and confidence < 0.90:
                    Increment confidence by 0.02 (confirmation-based growth)
                    Update stored ZoneMap with new confidence
    """

async def _on_surprise(self, event: SurpriseEvent) -> None:
    """
    Background task. Never blocks query().
    Spawned with asyncio.create_task() — fire and forget.

    If event.dissolve_triggered:
        Invalidate ALL L1 cache entries for event.topology_class
        Invalidate L2 cache entry for event.topology_class
        Set confidence = 0.0 for all stored ZoneMaps of this class
        Emit ZoneMapInvalidatedEvent
        Do NOT invalidate other topology classes

    If not dissolve_triggered (partial surprise):
        Reduce confidence for zones that diverged from prediction
        Confidence reduction: -0.05 per partial surprise
        Floor: 0.30 (never reduce to zero on partial surprise)
        Do NOT invalidate cache entries
    """
```

### WATCHDOG Registration

```python
async def initialize(self) -> None:
    # structural_layer.pt: reload zone_knowledge when it changes
    # debounce 500ms: preparse_daemon writes rapidly during batch
    WATCHDOG.register(
        "structural_layer.pt",
        self._reload_zone_knowledge,
        debounce_ms=500,
    )

    # Bus subscriptions
    await BUS.subscribe(TopicName.CLEAN_SIGNAL, self._on_clean_signal)
    await BUS.subscribe(TopicName.SURPRISE, self._on_surprise)

    # Load existing zone knowledge or EmptyZoneKnowledge if file absent
    await self._reload_zone_knowledge()

    log.info(
        "wlp_initialized",
        zone_count=len(self._zone_knowledge),
        device=str(self._device),
    )
```

### cold_start_warmup()

```python
async def cold_start_warmup(self) -> None:
    """
    Pre-populate L2 cache for all 18 topology classes
    before interface.py accepts any queries.

    For each topology class:
        Find the highest-confidence stored ZoneMap for this class
        (from structural_layer.pt zone_knowledge)
        Store in L2 cache

    Ordering: topology classes with most confirmed ZoneMaps first.
    Wikipedia-related classes will have highest confidence (6.7M training articles).
    GENERIC_HTML warmed last (lowest confidence, most heterogeneous).

    Completes before interface.py accepts queries.
    cold_start.py enforces this via await.
    WLP warm-up is not optional.
    """
```

### Full Method List

```
LatentParser (nn.Module)
    __init__()
    forward()      — training only, gradient flow
    readout()      — inference only, torch.no_grad()

WLPConfig dataclass

WorldLatentParser
    __init__()
    initialize()
    shutdown()
    query()                          — single public method
    discover_signal_zones()
    _l1_lookup() / _l1_store()
    _l2_lookup() / _l2_store()
    _l3_fresh_parse()
    _on_clean_signal()               — bus handler
    _on_surprise()                   — bus handler
    _reload_zone_knowledge()         — watchdog handler
    _load_model()
    _validate_model()
    _write_zone_knowledge()          — staging + atomic rename
    cold_start_warmup()
    health() → dict
    
WLP = WorldLatentParser()            — module-level singleton
```

### LOC Breakdown

```
WLPConfig dataclass:                  ~30 loc
LatentParser nn.Module:              ~150 loc
    __init__() SAGEConv layers:       ~40 loc
    forward():                        ~50 loc
    readout():                        ~30 loc
    (docstrings included in above)
WorldLatentParser.__init__:           ~70 loc
WorldLatentParser.initialize():       ~60 loc
WorldLatentParser.query():           ~120 loc
    L1/L2/L3 routing + intent         
_l1_lookup / _l1_store:               ~80 loc
_l2_lookup / _l2_store:               ~60 loc
_l3_fresh_parse():                   ~100 loc
discover_signal_zones():             ~180 loc
_on_clean_signal():                  ~100 loc
_on_surprise():                       ~80 loc
_reload_zone_knowledge():             ~60 loc
_load_model() + _validate_model():    ~80 loc
_write_zone_knowledge():              ~70 loc
cold_start_warmup():                  ~70 loc
health():                             ~40 loc
shutdown():                           ~40 loc
Module singleton + imports:           ~30 loc
Docstrings + comments:               ~400 loc
                                    ────────
latent_parser.py total:           ~1,840 loc
```

---

## Dependency Directions — Strict and One-Way

```
wlp_graph.py:
    imports: contracts.py, exceptions.py, torch, torch_geometric
             tree_sitter, antlr4
    NEVER imports: wlp_zones.py, latent_parser.py,
                   latent_model.py, mamba_router.py,
                   crawler_bus.py, store_watchdog.py

wlp_zones.py:
    imports: contracts.py, exceptions.py, torch
    NEVER imports: wlp_graph.py, latent_parser.py,
                   latent_model.py, mamba_router.py,
                   crawler_bus.py, store_watchdog.py

latent_parser.py:
    imports: contracts.py, exceptions.py, torch, torch_geometric
             crawler_bus.py, store_watchdog.py
             wlp_graph.py, wlp_zones.py
    NEVER imports: latent_model.py, mamba_router.py,
                   wlm_tokenizer.py, wlm_decoders.py
```

---

## Non-Negotiable Implementation Rules

```
1.  WLP never imports from latent_model.py. Not once. Not ever.
    They are peers. asyncio.gather() is the entirety of their relationship.

2.  wlp.query() always returns ZoneMap — never None, never raises.
    EmptyZoneMap is the defined failure value. Return it. Do not raise.

3.  LatentParser.readout() uses torch.no_grad() always.
    Inference never allows gradient flow. No exceptions.

4.  forward() is called only by preparse_daemon.py.
    Never from query path. Never from bus handlers.
    Training and inference share the model. Not the code path.

5.  All structural_layer.pt writes use staging + atomic rename.
    asyncio.Lock on write path — one writer at a time.
    SHA-256 verify staging before rename.
    os.replace() is the only acceptable rename call.

6.  Intent conditioning never triggers a new GraphSAGE forward pass.
    ZoneMap.with_intent() is O(1). It must stay O(1).
    If intent conditioning requires model inference: architecture violation.

7.  discover_signal_zones() confidence ceiling is 0.70.
    Unknown topology = uncertain classification.
    Above 0.70 requires confirmed extractions. Not configuration.

8.  tree_sitter.Parser is not shared across coroutines.
    ParserPool manages per-coroutine parser allocation.
    Shared parser instances corrupt CSTs. This is not recoverable.

9.  PyG Data objects are not persisted.
    ZoneMaps are persisted. Graphs are garbage collected after readout().
    structural_layer.pt stores ZoneDescriptors, not tensor graphs.

10. cold_start_warmup() completes before interface.py accepts queries.
    L2 cache must be populated for all 18 topology classes.
    cold_start.py enforces this via await.
    This is not optional.
```

---

## What The WLP Does Not Do

```
Does not fetch pages                phantom.py fetches
Does not classify URLs              classifier.py classifies
Does not compile recipes            topology/parser.py compiles
Does not strip noise                signal_kernel/ strips
Does not detect surprise            surprise_detector.py detects
Does not train at query time        preparse_daemon.py trains
Does not write source_matrix        preparse_daemon.py owns that region
Does not communicate with WLM       peers — asyncio.gather() only
Does not block on bus handlers      all handlers are background tasks
Does not re-parse on intent change  with_intent() is O(1) arithmetic
Does not emit SurpriseEvent         surprise_detector.py owns that
Does not make network calls         ever, under any condition
Does not return None from query()   EmptyZoneMap is the failure value
Does not cache PyG graphs           ZoneMaps only
Does not promote topology subclasses  signals candidacy, does not promote
Does not share tree_sitter.Parser   ParserPool enforces per-coroutine isolation
```

---

## LOC Summary — Three Files

```
wlp_graph.py:       ~1,680 loc
wlp_zones.py:       ~1,570 loc
latent_parser.py:   ~1,840 loc
                    ──────────
Total production:   ~5,090 loc

Tests (target 1:1): ~5,090 loc
                    ──────────
WLP total:          ~10,180 loc
```

Production code alone: **5,090 LOC — within 4,500–5,500 target.**

---

## What Success Looks Like

```
Functional correctness:
    query() returns valid ZoneMap for all 18 topology classes
    ZoneMap.confidence > 0.80 after 10+ confirmed extractions
    EmptyZoneMap returned on all failure paths — never None, never exception
    with_intent() correctly boosts primary zones, zeroes exclude zones
    discover_signal_zones() returns ZoneMap with confidence <= 0.70
    ANTLR4 fallback activates on Tree-sitter error rate > 0.50
    wlp_graph.py never imports wlp_zones.py
    wlp_zones.py never imports wlp_graph.py
    latent_parser.py never imports latent_model.py

Performance:
    query() L1 cache hit: <0.5ms
    query() L2 cache hit: <1ms
    query() L3 fresh parse (1,000 nodes): <5ms on 5080
    query() L3 fresh parse (10,000 nodes): <15ms on 5080
    ZoneMap.with_intent(): <0.1ms
    cold_start_warmup(): <200ms

Structural correctness:
    SAAS_DOCS ZoneMaps always include code block zones
    REST_API_JSON ZoneMaps always include schema/endpoint zones
    WIKIPEDIA_ARTICLE zones match preparse ground truth
    NEWS_ARTICLE ZoneMaps exclude nav, sidebar, footer reliably

Compounding:
    after Wikipedia preparse (6.7M articles):
        GraphSAGE trained on largest structured web corpus
        L2 cache populated for WIKIPEDIA_ARTICLE with confidence 0.95+
        zone patterns generalize to non-Wikipedia SAAS_DOCS pages

    after 1 week of crawl:
        L1 cache hits >70% for known domains
        discovered subclasses appearing in topology class list
        intent conditioning measurably reduces noise vs non-intent baseline
```

---

## Files Opus Needs To Build latent_parser.py

```
readme-wlp.md           ← this document (primary reference)
contracts.py            ← topology classes, phase constants, store paths
exceptions.py           ← full exception taxonomy
crawler_bus.py          ← bus subscription interface, topic names, event types
store_watchdog.py       ← watchdog registration interface, WATCHDOG singleton
wlp_graph.py            ← already built by Sonnet (import reference)
wlp_zones.py            ← already built by Sonnet (import reference)
```

No WLM files. No mamba_router.py. No latent_model.py.
No wlm_tokenizer.py. No wlm_decoders.py.
Those do not exist at the WLP level.

---

## Build Sequence Within WLP

```
Step 1: Sonnet builds wlp_graph.py
        No dependencies on wlp_zones.py or latent_parser.py
        Can be built immediately

Step 2: Sonnet builds wlp_zones.py
        No dependencies on wlp_graph.py or latent_parser.py
        Can be built immediately (parallel with Step 1 if desired)

Step 3: Opus builds latent_parser.py
        Requires wlp_graph.py and wlp_zones.py complete
        Imports from both — dependency direction enforced
        Full fresh session for Opus — no rushing
        This is the most complex of the three
```

---

*World Latent Parser — Complete Engineering Specification*
*Three files. Three concerns. One peer to the WLM.*
*wlp_graph.py parses. wlp_zones.py assembles. latent_parser.py orchestrates.*
*The map IS the model. Reading the map IS the forward pass.*
*Updating the map IS the gradient step.*
*The WLM sees the web from above. The WLP sees it from within.*
*Together via asyncio.gather() — never sequential, never coupled.*
*AXIOM INTERNAL // DO NOT SURFACE*
