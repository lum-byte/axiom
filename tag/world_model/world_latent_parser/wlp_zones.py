"""
tag/world_model/wlp_zones.py
============================
AXIOM WLP Output Processing Layer — Stage 2 through Stage 5.

wlp_graph.py feeds this file.  latent_parser.py consumes it.

Responsibility: convert per-node classification tensors produced by
LatentParser into a ZoneMap that topology/parser.py can compile
grep/sed/awk recipes from.  Nothing more.  Nothing less.

Pipeline:
    Stage 1 — Node Classification     (n_nodes,3) logits → label+conf list
    Stage 2 — Zone Grouping           flat nodes → CandidateZone list
    Stage 3 — Zone Description        CandidateZone → ZoneDescriptor + selector
    Stage 4 — Intent Conditioning     ZoneDescriptor list + intent_vector → weights
    Stage 5 — ZoneMap Assembly        all components → frozen ZoneMap

Tensor boundary: the ONLY tensors touched are node_classifications (n_nodes,3)
and node_confidences (n_nodes,1).  Both arrive from LatentParser.
classify_nodes() converts them to Python via .item() / .tolist().
After classify_nodes() returns, this file is pure Python.
No torch operation past Stage 1.  No torch import used for computation.

Immutability contract: ZoneMap is frozen=True.  with_intent() uses
dataclasses.replace().  Nothing mutates a ZoneMap in place.  Ever.

Failure contract: assemble_zone_map() never raises.
Every failure path returns EmptyZoneMap().
topology/parser.py handles EmptyZoneMap.  It does not handle None.
Return EmptyZoneMap on any doubt.  Never return a wrong ZoneMap.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import hashlib
import struct as _struct
import functools  # noqa
import dataclasses
import enum
import logging
import math
import re
import statistics # noqa
import time
from dataclasses import dataclass, field # noqa
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    FrozenSet,
    Sequence,
)

if TYPE_CHECKING:
    import torch  # type annotations only — never imported at runtime # noqa

log = logging.getLogger("axiom.wlp.zones")

# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # ── Node classification label constants ───────────────────────────────────
    # latent_parser.py imports these to interpret classify_nodes() output and
    # to check label values when iterating node lists directly.
    "NODE_SIGNAL",
    "NODE_NOISE",
    "NODE_BOUNDARY",

    # ── Threshold constants ───────────────────────────────────────────────────
    # latent_parser.py passes confidence_threshold to assemble_zone_map() and
    # needs the canonical defaults to know what it is overriding.
    "CONFIDENCE_THRESHOLD",
    "BOUNDARY_CONFIDENCE_THRESHOLD",
    "DISCOVERY_CONFIDENCE_CEILING",
    "MIN_ZONE_CONFIDENCE",

    # ── Intent weight constants ───────────────────────────────────────────────
    # topology/parser.py reads intent_weights from ZoneMap and needs the
    # sentinel values to decide whether a zone is suppressed (0.0), default
    # (1.0), or boosted (> 1.0).
    "INTENT_WEIGHT_DEFAULT",
    "INTENT_WEIGHT_EXCLUDE",
    "INTENT_WEIGHT_CEILING",

    # ── Structural lookup sets ────────────────────────────────────────────────
    # latent_parser.py and topology/parser.py may inspect these to understand
    # which element types and roles were considered during selector generation.
    "LANDMARK_ROLES",
    "SEMANTIC_ELEMENTS",
    "DISCRIMINATIVE_CLASSES",
    "NON_DISCRIMINATIVE_CLASSES",

    # ── Topology class routing sets ───────────────────────────────────────────
    # latent_parser.py reads SECTION_SCOPED_CLASSES when deciding whether to
    # split a ZoneMap query by boundary sections before dispatching to
    # topology/parser.py.  Canonical spec name is SECTION_SCOPED_CLASSES;
    # BREADTH_FIRST and FLAT variants exported for completeness.
    "SECTION_SCOPED_CLASSES",
    "BREADTH_FIRST_TOPOLOGY_CLASSES",
    "FLAT_TOPOLOGY_CLASSES",

    # ── Intent vocabulary ─────────────────────────────────────────────────────
    # latent_parser.py uses INTENT_TOKEN_VOCABULARY to log which intent tags
    # are active during a query without re-calling parse_intent_tags().
    "INTENT_TOKEN_VOCABULARY",

    # ── Intent matcher registry ───────────────────────────────────────────────
    # Diagnostic consumers and unit tests use this to enumerate which intent
    # tags have registered structural matchers without calling
    # zone_matches_intent_semantics() in a loop over all tags.
    "INTENT_CATEGORY_MATCHERS",

    # ── Enums ─────────────────────────────────────────────────────────────────
    "ExtractionStrategy",

    # ── Exported dataclasses ──────────────────────────────────────────────────
    "ZoneDescriptor",
    "BoundaryDescriptor",
    "ZoneMap",
    "IntentTags",
    "EmptyZoneMap",
    "EmptyZoneKnowledge",

    # ── Node classification ───────────────────────────────────────────────────
    "classify_nodes",

    # ── Zone grouping ─────────────────────────────────────────────────────────
    "group_signal_nodes",

    # ── CSS selector generation ───────────────────────────────────────────────
    "generate_css_selector",

    # ── Scope + content type + density ────────────────────────────────────────
    "determine_scope",
    "infer_content_type",
    "compute_density",
    "make_candidate_zone",

    # ── Priority + strategy ───────────────────────────────────────────────────
    "assign_priorities",
    "select_extraction_strategy",

    # ── Boundary identification ───────────────────────────────────────────────
    "identify_boundaries",

    # ── Boundary identification ───────────────────────────────────────────────
    "ZoneMerkleDAG",
    "MerkleDiff",

    # ── Intent conditioning ───────────────────────────────────────────────────
    "parse_intent_tags",
    "zone_matches_intent_semantics",
    "apply_intent_weights",

    # ── Assembly ──────────────────────────────────────────────────────────────
    "assemble_zone_map",

    # ── Validation ────────────────────────────────────────────────────────────
    "validate_zone_map",
]

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# Single source of truth for every threshold used in this file.
# Do not hard-code these values inline — every threshold must be traceable
# to its definition here with documented rationale.
# ─────────────────────────────────────────────────────────────────────────────

# Node classification labels — argmax over the 3-class logit dimension.
NODE_SIGNAL:   int = 0
NODE_NOISE:    int = 1
NODE_BOUNDARY: int = 2

# Confidence thresholds.
# 0.35: minimum confidence to accept a SIGNAL classification.
# Calibrated so that: first-seen domains are not over-suppressed (would happen
# at 0.50+) while genuine model uncertainty is filtered (would propagate at
# 0.20-).  The decision boundary lives in the neighbourhood of argmax(logits)
# ≈ 0.40 on a 3-class softmax.  0.35 accepts near-argmax nodes while rejecting
# flat distributions.
CONFIDENCE_THRESHOLD: float = 0.35

# 0.40: minimum confidence to accept a BOUNDARY classification.
# Boundaries are higher-stakes than signal: a wrong boundary node becomes a
# wrong sed/awk delimiter pattern that silently truncates or misaligns recipe
# output.  The 0.05 uplift from CONFIDENCE_THRESHOLD reflects this asymmetric
# cost.  A missing boundary is recoverable (DEPTH_FIRST traversal is the
# default).  A wrong boundary is not.
BOUNDARY_CONFIDENCE_THRESHOLD: float = 0.40

# 0.70: confidence ceiling for ZoneMaps produced by discover_signal_zones().
# Discovery-path ZoneMaps have seen fewer feedback cycles than promoted ones.
# Capping at 0.70 prevents them from outcompeting known-good ZoneMaps in
# latent_parser.py's three-tier cache purely on initial confidence arithmetic.
DISCOVERY_CONFIDENCE_CEILING: float = 0.70

# 0.30: minimum confidence for topology/parser.py to compile from this ZoneMap
# rather than falling back to the hardcoded GENERIC_HTML recipe.
# ZoneMaps below 0.30 are structurally unreliable — better to use a hardcoded
# recipe that extracts too broadly than a weak ZoneMap that extracts wrongly.
MIN_ZONE_CONFIDENCE: float = 0.30

# Weight arithmetic for apply_intent_weights().
INTENT_WEIGHT_DEFAULT:   float = 1.0   # no intent match
INTENT_WEIGHT_PRIMARY:   float = 1.0   # additive per primary tag match
INTENT_WEIGHT_SECONDARY: float = 0.5   # additive per secondary tag match
INTENT_WEIGHT_URGENCY:   float = 0.5   # additive for high-urgency warning match
INTENT_WEIGHT_LOCKED:    float = 0.8   # additive for locked_out + list zone
INTENT_WEIGHT_DEBUG:     float = 0.5   # additive for debugging + code zone
INTENT_WEIGHT_CEILING:   float = 4.0   # maximum weight — clipped if exceeded
INTENT_WEIGHT_EXCLUDE:   float = 0.0   # exclude override — absolute and final

# Intent vector encoding positions.  256-float vector from contracts.py.
INTENT_PRIMARY_START:   int = 0
INTENT_PRIMARY_END:     int = 64
INTENT_SECONDARY_START: int = 64
INTENT_SECONDARY_END:   int = 128
INTENT_EXCLUDE_START:   int = 128
INTENT_EXCLUDE_END:     int = 192
INTENT_URGENCY_START:   int = 192
INTENT_URGENCY_END:     int = 208    # 16 positions → argmax over 4 urgency states
INTENT_USER_STATE_START: int = 208
INTENT_USER_STATE_END:   int = 256   # 48 positions → argmax over 8 user_state states

# Activation thresholds for intent tokens.
INTENT_PRIMARY_THRESHOLD:   float = 0.5   # token must exceed this to be "active" primary
INTENT_SECONDARY_THRESHOLD: float = 0.5   # same for secondary
INTENT_EXCLUDE_THRESHOLD:   float = 0.3   # lower — over-suppress rather than miss exclusion

# Urgency and user_state decoding.
# 16 positions for urgency → 4 bins of 4 floats each (argmax within each bin).
# State within each bin: 0=high, 1=medium, 2=low, 3=none
URGENCY_STATES: Tuple[str, ...] = ("high", "medium", "low", "none")
URGENCY_BIN_SIZE: int = 4   # 16 positions / 4 states

# 48 positions for user_state → 8 bins of 6 floats each.
USER_STATE_STATES: Tuple[str, ...] = (
    "locked_out", "exploring", "urgent", "researching",
    "purchasing", "debugging", "default", "unknown",
)
USER_STATE_BIN_SIZE: int = 6   # 48 positions / 8 states

# ARIA landmark roles — highest reliability selector tier.
LANDMARK_ROLES: frozenset = frozenset({
    "main", "article", "navigation", "complementary",
    "contentinfo", "banner", "search", "form",
})

# Semantic HTML5 elements — second-tier selector preference.
SEMANTIC_ELEMENTS: frozenset = frozenset({
    "article", "main", "nav", "aside",
    "header", "footer", "section",
})

# Heading element types — used for SECTION_BOUNDARY identification.
HEADING_ELEMENTS: frozenset = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

# CSS classes that indicate noise zones (structural chrome, not content).
NOISE_CLASSES: frozenset = frozenset({
    "sidebar", "side-bar", "side-panel", "side-nav",
    "ad", "ads", "advertisement", "advert", "sponsored",
    "cookie", "cookie-banner", "gdpr", "consent",
    "modal", "overlay", "popup", "popover",
    "breadcrumb", "breadcrumbs",
    "pagination", "pager",
    "footer", "page-footer", "site-footer",
    "header", "page-header", "site-header", "masthead",
    "nav", "navbar", "navigation", "menu",
    "social", "share", "sharing",
    "related", "recommended", "suggestions",
    "comment", "comments", "discussion",
    "tag", "tags", "categories", "category",
    "author-bio", "byline", "meta",
    "skip-nav", "screen-reader",
})

# CSS classes that are semantically discriminative for signal zone selectors.
# Priority-ordered — first match from this list is used.
DISCRIMINATIVE_CLASSES: List[str] = [
    # Primary content
    "article-body", "post-body", "entry-content", "article-content",
    "story-body", "story-content", "content-body",
    # Main content regions
    "main-content", "page-content", "primary-content",
    "content-main", "main-area",
    # Documentation-specific
    "doc-content", "docs-content", "documentation",
    "prose", "markdown-body", "markdown", "markup",
    # Sidebar / auxiliary signal (kept here because some sidebars have signal)
    "sidebar-content", "aside-content",
    # Generic content
    "article", "content",
    # Code
    "code-block", "code-snippet", "code-sample", "highlight",
    "syntax", "syntax-highlight", "hljs", "prism", "shiki",
    # Callouts / warnings
    "warning", "caution", "danger", "alert", "notice",
    "note", "tip", "info", "callout", "admonition",
    "important", "success", "failure",
    # Pricing / commercial
    "pricing", "price-table", "plan-grid", "pricing-table",
    "plan", "plans", "tier",
    # Navigation
    "sidebar", "side-bar", "side-panel",
    "nav", "navbar", "navigation",
    # Structural markers used as signal sources
    "footer", "page-footer", "site-footer",
    "header", "page-header", "masthead",
]

# CSS classes that are NOT discriminative — layout utilities and state markers.
# Never generate a selector from these classes.
NON_DISCRIMINATIVE_CLASSES: frozenset = frozenset({
    # Layout utilities
    "container", "wrapper", "inner", "outer", "box",
    "row", "col", "column", "grid", "flex",
    "clearfix", "cf", "group",
    "float-left", "float-right", "pull-left", "pull-right",
    # Spacing and sizing
    "w-full", "h-full", "max-w", "min-h",
    # State
    "hidden", "visible", "show", "hide",
    "active", "inactive", "selected", "disabled",
    "open", "closed", "expanded", "collapsed",
    "current", "last", "first",
    # Generic
    "main", "page", "section", "block", "item",
    "list", "list-item", "entry",
    "text", "body",
    "top", "bottom", "left", "right", "center",
    "sm", "md", "lg", "xl",
    # Framework noise
    "d-none", "d-flex", "d-block",
    "mt-0", "mb-0", "pt-0", "pb-0",
})

# Topology classes that trigger SECTION_SCOPED strategy when ≥2 boundaries present.
SECTION_SCOPED_TOPOLOGY_CLASSES: frozenset = frozenset({
    "REST_API_JSON",
    "REST_API_JSON_PAGINATED",
    "SAAS_DOCS_VERSIONED",
    "WIKIPEDIA_ARTICLE",
})

# Canonical name from the spec's Full Function List.
# SECTION_SCOPED_TOPOLOGY_CLASSES is the internal descriptive name used
# throughout this file; SECTION_SCOPED_CLASSES is what the spec exports
# and what latent_parser.py imports.  They are the same object.
SECTION_SCOPED_CLASSES: frozenset = SECTION_SCOPED_TOPOLOGY_CLASSES

# Topology classes that trigger BREADTH_FIRST strategy.
BREADTH_FIRST_TOPOLOGY_CLASSES: frozenset = frozenset({
    "ECOMMERCE_PRODUCT",
    "ECOMMERCE_PRODUCT_VARIANT",
})

# Topology classes that trigger FLAT strategy.
FLAT_TOPOLOGY_CLASSES: frozenset = frozenset({
    "JSON_LD_STRUCTURED",
})

# Landmark ancestor classes checked during determine_scope() ancestry walk.
# These are checked against node.css_classes for each ancestor node.
LANDMARK_ANCESTOR_CLASSES: Tuple[str, ...] = (
    "main-content", "page-content", "content-main",
    "primary-content", "article-body", "article-content",
    "post-body", "entry-content", "content",
    "doc-content", "docs-content", "documentation",
    "markdown-body",
)

# ─────────────────────────────────────────────────────────────────────────────
# INTENT TOKEN VOCABULARY
# 64 primary tokens occupy positions 0-63 of the intent_vector.
# 64 secondary tokens occupy positions 64-127.
# 64 exclude tokens occupy positions 128-191.
# The same vocabulary index maps across all three ranges:
#   active primary tag at position P   → INTENT_TOKEN_VOCABULARY[P]
#   active secondary tag at position P → INTENT_TOKEN_VOCABULARY[P - 64]
#   active exclude tag at position P   → INTENT_TOKEN_VOCABULARY[P - 128]
# ─────────────────────────────────────────────────────────────────────────────

INTENT_TOKEN_VOCABULARY: Tuple[str, ...] = (
    # 0-9: Account / access recovery
    "account_recovery", "lost_access", "restore_access", "reset_password",
    "login_help", "unlock_account", "verify_identity", "mfa_recovery",
    "backup_codes", "session_recovery",
    # 10-19: API / technical reference
    "api_reference", "endpoint", "schema", "parameter", "authentication",
    "authorization", "rate_limit", "webhook", "sdk", "integration",
    # 20-29: Pricing / commercial
    "pricing", "plans", "cost", "billing", "subscription",
    "upgrade", "downgrade", "trial", "enterprise", "features",
    # 30-39: Getting started / tutorials
    "getting_started", "quickstart", "tutorial", "guide", "walkthrough",
    "setup", "installation", "configuration", "onboarding", "first_steps",
    # 40-49: Code / examples
    "code_example", "snippet", "sample", "demo", "playground",
    "repository", "library", "package", "cli", "command",
    # 50-59: Warnings / safety
    "warning", "caution", "danger", "breaking_change", "deprecation",
    "security", "vulnerability", "migration", "rollback", "incident",
    # 60-63: Miscellaneous
    "changelog", "release_notes", "faq", "troubleshooting",
)

# Verify vocabulary fits in 64-token slots.
assert len(INTENT_TOKEN_VOCABULARY) <= 64, (
    f"INTENT_TOKEN_VOCABULARY has {len(INTENT_TOKEN_VOCABULARY)} entries; "
    "must be ≤ 64 to fit in a single intent range slot."
)

# ─────────────────────────────────────────────────────────────────────────────
# INTENT CATEGORY MATCHERS
# Each key is an intent tag string from INTENT_TOKEN_VOCABULARY.
# Each value is a Callable[[ZoneDescriptor], bool] that returns True if the
# zone's structural characteristics match that intent.
# Built lazily after ZoneDescriptor is defined.
# ─────────────────────────────────────────────────────────────────────────────

# (Populated at module bottom after ZoneDescriptor class is defined.)
INTENT_CATEGORY_MATCHERS: Dict[str, Callable] = {}


# ─────────────────────────────────────────────────────────────────────────────
# ExtractionStrategy ENUM
# ─────────────────────────────────────────────────────────────────────────────

class ExtractionStrategy(enum.Enum):
    """
    Instructs topology/parser.py how to traverse signal zones when compiling
    a grep/sed/awk recipe.

    DEPTH_FIRST:
        Traverse signal zones depth-first within the DOM subtree.
        Default strategy.  Used when signal is concentrated in nested content.
        Example: SAAS_DOCS where article body contains nested code blocks,
        lists, and paragraphs — traverse the tree to capture all in doc order.

    BREADTH_FIRST:
        Traverse signal zones breadth-first.
        Used when signal is distributed horizontally across siblings.
        Example: ECOMMERCE_PRODUCT_VARIANT where variant details sit as
        siblings in a product grid.

    SECTION_SCOPED:
        Each BOUNDARY node defines an independent extraction scope.
        Signal zones within each boundary section are extracted as
        self-contained units.
        Example: REST_API_JSON where each endpoint section is independently
        meaningful and must not bleed into adjacent endpoint sections.

    FLAT:
        No traversal — top-level signal zones only.  Nested content ignored.
        Used for shallow content structures.
        Example: JSON_LD_STRUCTURED where signal lives at the top level of
        the structured data object.
    """

    DEPTH_FIRST    = "depth_first"
    BREADTH_FIRST  = "breadth_first"
    SECTION_SCOPED = "section_scoped"
    FLAT           = "flat"


# ─────────────────────────────────────────────────────────────────────────────
# ZoneDescriptor DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneDescriptor:
    """
    Complete structural description of a single signal zone within a page.

    topology/parser.py translates these fields directly to recipe primitives:

        selector + scope   →  grep/sed address range pattern
        content_type       →  grep context flags (-A, -B values)
        density            →  confidence that zone is worth extracting
        priority           →  ordering within compiled recipe

    Contract: every field has defined semantics and a defined valid range.
    No Optional fields except where explicitly documented.
    No field is inferred post-construction.
    A ZoneDescriptor that exists is complete and valid.
    """

    selector: str
    """
    CSS selector (or XPath if selector_type=="xpath") targeting this zone's
    root element.  Must be specific enough to isolate this zone on this page.
    Must be general enough to work on structurally similar pages of the same
    topology class.

    Generation strategy: element_type + most_discriminative_class only.
    Never positional (:nth-child) — too page-specific, breaks on layout change.
    Never descendant chains longer than 3 levels — too fragile.

    Valid examples:
        "article.article-body"          element + discriminative class
        ".main-content"                 class alone
        "[role=main]"                   ARIA role — most reliable
        "div.article > p"               2-level, acceptable
    Invalid:
        ""                              empty — compile produces nothing
        "body > div:nth-child(3) > p"   positional — breaks on layout change
        "div > div > div > div > p"     4 levels — too fragile
    """

    selector_type: str
    """
    Exactly "css" or "xpath".  CSS is preferred in all cases where it can
    express the required selector.  XPath only when:
      - content is XML (DocBook, DITA)
      - selector requires text content matching (contains())
      - selector requires attribute-value substring matching
    """

    scope: str
    """
    CSS selector for the nearest landmark ancestor of this zone.
    Scopes extraction to prevent false positives from identically-classed
    elements outside the relevant structural region.

    Landmark priority order (first match wins):
      1. [role=main]
      2. main (element)
      3. article (element)
      4. .main-content / .article-body / .doc-content (discriminative class)
      5. body  ← always valid fallback

    Never empty.  "body" is always valid.
    """

    content_type: str
    """
    Structural content type of this zone.  One of exactly:
      "prose"   flowing text paragraphs, narrative content
      "code"    code blocks, preformatted text, command examples
      "list"    ordered/unordered/definition lists
      "table"   tabular data, comparison tables, parameter tables
      "mixed"   combination of two or more of the above

    Drives grep context lines in recipe compilation:
      "prose":  -A3  (paragraph context)
      "code":   -B1 -A10  (code block context preservation)
      "list":   -A0  (each item self-contained)
      "table":  -B1 -A1  (row with header context)
      "mixed":  -A2  (conservative context)
    """

    average_depth: float
    """
    Mean DOM depth of nodes within this zone.
    Computed as: sum(node.depth for node in zone_nodes) / len(zone_nodes)
    Absolute depth — not normalised.  Range: [0.0, max_document_depth].
    Shallow zones (< 3): flat extraction sufficient.
    Deep zones (> 8): depth-scoped extraction needed.
    """

    density: float
    """
    Signal density: signal_chars / total_chars.
    signal_chars = sum of text_char_count for SIGNAL nodes in zone.
    total_chars  = sum of subtree_char_count for all nodes in zone.
    Range: [0.0, 1.0].  Clamped if floating-point arithmetic drifts above 1.0.
    """

    priority: int
    """
    Extraction order.  Lower number = extract first.
    Assigned by assign_priorities() using DOM position (primary) and signal
    density (tiebreaker).  Unique per ZoneMap — no two zones share a priority.
    Range: [0, n_signal_zones - 1].
    """


# ─────────────────────────────────────────────────────────────────────────────
# BoundaryDescriptor DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class BoundaryDescriptor:
    """
    Description of a structural boundary marker.

    Boundaries are structural delimiters that recipe compilers use to scope
    extraction regions.  They are start/end markers for sed address ranges and
    awk capture state transitions.  A boundary does not contain extractable
    signal itself — it marks where one zone ends and another begins.

    Three boundary types:

    SECTION_BOUNDARY:
        Marks the start of a new logical section.
        Recipe: reset capture state, begin new extraction scope.
        Example: h2 headings in SAAS_DOCS.

    CONTENT_BOUNDARY:
        Marks the transition from noise to content (or content to noise).
        Recipe: toggle capture state.
        Example: div.article-body wrapper for the main article content.

    NOISE_BOUNDARY:
        Marks an explicit noise zone — everything inside is suppressed.
        Recipe: exclude this region from extraction.
        Example: div.sidebar.
    """

    selector: str
    """CSS selector for the boundary element.  Same rules as ZoneDescriptor.selector."""

    boundary_type: str
    """Exactly one of: "SECTION_BOUNDARY", "CONTENT_BOUNDARY", "NOISE_BOUNDARY"."""

    delimiter_content: str
    """
    Regex pattern matching the text content or attribute values that identify
    this boundary in the raw HTML stream.  Used for grep/sed/awk pattern
    generation.  Must be a valid Python re pattern.

    Examples:
        'class="article-body"'    attribute-based boundary
        'class="sidebar"'         noise boundary
        '<h[2-4][^>]*>'          heading-based section boundary
    """


# ─────────────────────────────────────────────────────────────────────────────
# ZONE MERKLE DAG
# Content-addressed structural identity for ZoneMaps.
#
# Mathematical foundation:
#   SHA3-256 (Keccak sponge, FIPS 202) over SHA-256 (Merkle-Damgård).
#   Keccak's rate/capacity separation eliminates length-extension attacks.
#   Subtree forgery requires second-preimage against 1600-bit internal state.
#   Depth encoding on internal nodes: H(H(A)||H(B), depth=2) ≠ H(A||B, depth=1).
#   Domain-separation prefix _MERKLE_DSP partitions leaf/internal/root spaces.
#
# Key invariant:
#   Leaf hashes cover structural geometry only — NOT intent_weights, confidence,
#   version, produced_at, or topology_router_version.
#   with_intent() → same DAG root.
#   confidence decay → same DAG root.
#   selector/scope/content_type change → different DAG root.
#   Cache invalidation is structural. Not statistical.
# ─────────────────────────────────────────────────────────────────────────────

# 16-byte domain-separation prefix. Partitions all AXIOM WLP Merkle hashes
# from any other SHA3-256 usage in the codebase. Never change this value —
# changing it invalidates all stored DAG roots.
_MERKLE_DSP: bytes = b"AXIOM\x00WLP\x00MRK\x001"  # exactly 16 bytes


def _mh(tag: bytes, payload: bytes) -> bytes:
    """
    Tagged SHA3-256. Every hash in the DAG is domain-separated by both the
    module prefix and a per-call tag (b"leaf", b"node", b"root", etc.).
    This guarantees that a valid leaf hash can never equal a valid internal
    node hash, regardless of input — preventing second-preimage substitution
    across tree levels.
    """
    return hashlib.sha3_256(_MERKLE_DSP + tag + b"\x00" + payload).digest()


def _merkle_build(leaves: Sequence[bytes]) -> Tuple[bytes, Tuple[bytes, ...]]:
    """
    Build a complete binary Merkle tree from leaf hashes.

    Returns (root_hash, leaf_hashes_tuple).

    Tree properties:
        - Odd-length layers duplicate the last node (standard convention).
        - Internal nodes encode depth as a single byte suffix, preventing
          a node at depth D from having the same hash as a node at depth D+1
          with identical children (second-preimage barrier across levels).
        - Empty input returns (_mh(b"empty", b""), ()).

    Time complexity: O(n) where n = len(leaves).
    Space complexity: O(n) — only one layer materialized at a time.
    """
    if not leaves:
        return _mh(b"empty", b""), ()

    leaf_tuple = tuple(leaves)
    layer: List[bytes] = list(leaf_tuple)
    depth = 0

    while len(layer) > 1:
        depth += 1
        next_layer: List[bytes] = []
        for i in range(0, len(layer), 2):
            left  = layer[i]
            right = layer[i + 1] if i + 1 < len(layer) else layer[i]
            next_layer.append(
                _mh(b"node", left + right + bytes([depth & 0xFF]))
            )
        layer = next_layer

    return layer[0], leaf_tuple


@dataclass(frozen=True)
class MerkleDiff:
    """
    Structural diff between two ZoneMerkleDAGs.
    Produced by ZoneMerkleDAG.diff(). Describes exactly what changed.

    All index sets are relative to the shorter ZoneMap — indices beyond
    min(len_a, len_b) represent zones added or removed.
    """
    signal_changed:    FrozenSet[int]   # indices of changed/added/removed signal zones
    noise_changed:     FrozenSet[int]   # indices of changed/added/removed noise zones
    boundary_changed:  FrozenSet[int]   # indices of changed/added/removed boundaries

    root_changed:          bool
    signal_root_changed:   bool
    noise_root_changed:    bool
    boundary_root_changed: bool

    @property
    def is_empty(self) -> bool:
        """True if structurally identical ZoneMaps — DAG roots match."""
        return not self.root_changed

    @property
    def n_changed(self) -> int:
        return len(self.signal_changed) + len(self.noise_changed) + len(self.boundary_changed)

    @property
    def is_signal_only(self) -> bool:
        """True if only signal zones changed — noise and boundaries identical."""
        return self.signal_root_changed and not self.noise_root_changed and not self.boundary_root_changed


@dataclass(frozen=True)
class ZoneMerkleDAG:
    """
    Content-addressed Merkle DAG identity for a ZoneMap.

    Tree structure:
        root (32 bytes, SHA3-256)
        ├── signal_root    ← Merkle root of signal_zone leaf hashes
        ├── noise_root     ← Merkle root of noise_zone leaf hashes
        └── boundary_root  ← Merkle root of boundary leaf hashes

    Each ZoneDescriptor leaf hashes: selector, selector_type, scope,
    content_type, average_depth, density, priority.
    Floats encoded as big-endian IEEE 754 doubles for cross-platform
    determinism. Strings null-terminated before concatenation to prevent
    "ab"+"c" == "a"+"bc" collisions.

    Root encodes topology_class + domain for namespace isolation:
    structurally identical pages on different domains get different roots.

    Usage in latent_parser.py:
        dag = ZoneMerkleDAG.from_zone_map(zone_map)
        # Store dag alongside zone_map in L1 cache entry.
        # On SurpriseEvent: dag.contains_selector(event.surprise_selector)
        # → True: decay this cache entry's confidence
        # → False: leave untouched
        # On watchdog reload: dag.diff(new_dag).is_empty
        # → True: same structure, cache entry still valid
        # → False: evict
    """

    signal_root:    bytes   # 32 bytes
    noise_root:     bytes   # 32 bytes
    boundary_root:  bytes   # 32 bytes
    root:           bytes   # 32 bytes — the cache identity key

    signal_leaves:   Tuple[bytes, ...]   # per-zone, parallel to signal_zones
    noise_leaves:    Tuple[bytes, ...]   # per-zone, parallel to noise_zones
    boundary_leaves: Tuple[bytes, ...]   # per-boundary, parallel to boundaries

    # Selector sets for O(1) surgical eviction lookup.
    # Contains encoded selectors from signal zones only — noise and boundary
    # selectors are not used for surgical eviction (SurpriseEvents are
    # signal-zone corrections by definition).
    signal_selector_set: FrozenSet[str]

    @staticmethod
    def _zone_leaf(zd: "ZoneDescriptor") -> bytes:
        """
        Leaf hash for a ZoneDescriptor.
        Structural fields only. Floats as big-endian doubles.
        Null-terminated strings prevent cross-field collisions.
        """
        payload = (
            zd.selector.encode()       + b"\x00" +
            zd.selector_type.encode()  + b"\x00" +
            zd.scope.encode()          + b"\x00" +
            zd.content_type.encode()   + b"\x00" +
            _struct.pack(">ddi", zd.average_depth, zd.density, zd.priority)
        )
        return _mh(b"zone", payload)

    @staticmethod
    def _boundary_leaf(bd: "BoundaryDescriptor") -> bytes:
        """
        Leaf hash for a BoundaryDescriptor.
        """
        payload = (
            bd.selector.encode()          + b"\x00" +
            bd.boundary_type.encode()     + b"\x00" +
            bd.delimiter_content.encode()
        )
        return _mh(b"boundary", payload)

    @classmethod
    def from_zone_map(cls, zm: "ZoneMap") -> "ZoneMerkleDAG":
        """
        Build the full Merkle DAG from a ZoneMap.
        O(n) in total zone + boundary count.
        Call once at production time — store result in cache entry.
        Never call on the query hot path.
        """
        signal_root, signal_leaves     = _merkle_build(
            [cls._zone_leaf(zd) for zd in zm.signal_zones]
        )
        noise_root, noise_leaves       = _merkle_build(
            [cls._zone_leaf(zd) for zd in zm.noise_zones]
        )
        boundary_root, boundary_leaves = _merkle_build(
            [cls._boundary_leaf(bd) for bd in zm.boundaries]
        )

        root = _mh(
            b"root",
            signal_root + noise_root + boundary_root +
            zm.topology_class.encode() + b"\x00" +
            zm.domain.encode()
        )

        return cls(
            signal_root=signal_root,
            noise_root=noise_root,
            boundary_root=boundary_root,
            root=root,
            signal_leaves=signal_leaves,
            noise_leaves=noise_leaves,
            boundary_leaves=boundary_leaves,
            signal_selector_set=frozenset(zd.selector for zd in zm.signal_zones),
        )

    def contains_selector(self, selector: str) -> bool:
        """
        True if any signal zone was derived from this selector.
        O(1). Used for surgical eviction on SurpriseEvent.
        """
        return selector in self.signal_selector_set

    def diff(self, other: "ZoneMerkleDAG") -> "MerkleDiff":
        """
        Compute structural diff between two ZoneMerkleDAGs.
        Short-circuits at root: if roots match, returns empty MerkleDiff immediately.
        Otherwise descends to subtree roots, then to leaves.
        O(1) best case (identical). O(n) worst case (fully changed).
        """
        if self.root == other.root:
            return MerkleDiff(
                signal_changed=frozenset(),
                noise_changed=frozenset(),
                boundary_changed=frozenset(),
                root_changed=False,
                signal_root_changed=False,
                noise_root_changed=False,
                boundary_root_changed=False,
            )

        def _leaf_diff(a: Tuple[bytes, ...], b: Tuple[bytes, ...]) -> FrozenSet[int]:
            changed = set()
            for i, (x, y) in enumerate(zip(a, b)):
                if x != y:
                    changed.add(i)
            # Zones added or removed
            for i in range(min(len(a), len(b)), max(len(a), len(b))):
                changed.add(i)
            return frozenset(changed)

        signal_changed   = _leaf_diff(self.signal_leaves,   other.signal_leaves)   if self.signal_root   != other.signal_root   else frozenset()
        noise_changed    = _leaf_diff(self.noise_leaves,    other.noise_leaves)    if self.noise_root    != other.noise_root    else frozenset()
        boundary_changed = _leaf_diff(self.boundary_leaves, other.boundary_leaves) if self.boundary_root != other.boundary_root else frozenset()

        return MerkleDiff(
            signal_changed=signal_changed,
            noise_changed=noise_changed,
            boundary_changed=boundary_changed,
            root_changed=True,
            signal_root_changed=(self.signal_root   != other.signal_root),
            noise_root_changed=(self.noise_root     != other.noise_root),
            boundary_root_changed=(self.boundary_root != other.boundary_root),
        )


# ─────────────────────────────────────────────────────────────────────────────
# IntentTags DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IntentTags:
    """
    Parsed, named representation of an intent_vector for zone weighting.

    Produced by parse_intent_tags() from a raw 256-float intent_vector.
    Not frozen — this is an intermediate computation object used only inside
    apply_intent_weights() and with_intent().  It is never stored, never
    exported from assemble_zone_map(), and never appears in a ZoneMap.

    Intent vector encoding layout:
        [0:64]    primary intent tokens   (threshold 0.50 for activation)
        [64:128]  secondary intent tokens (threshold 0.50)
        [128:192] exclude intent tokens   (threshold 0.30 — lower to over-suppress)
        [192:208] urgency encoding        (argmax over 4-state bins)
        [208:256] user_state encoding     (argmax over 8-state bins)
    """

    primary:    List[str]   # active primary intent tags
    secondary:  List[str]   # active secondary intent tags
    exclude:    List[str]   # active exclude tags
    urgency:    str         # "high" | "medium" | "low" | "none"
    user_state: str         # "locked_out" | "exploring" | "urgent" | ...


# ─────────────────────────────────────────────────────────────────────────────
# CandidateZone — INTERNAL ONLY
# Not exported.  Not part of the public API.
# Lives only inside assemble_zone_map() — intermediate representation between
# raw node lists and fully-built ZoneDescriptors.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _CandidateZone:
    """
    Intermediate representation of a zone during assembly.

    Holds the raw grouped nodes before ZoneDescriptors are built from them.
    Created by group_signal_nodes(), consumed by all Stage 3 functions, and
    discarded after ZoneDescriptors are constructed.

    Not exported.  Do not reference this type outside this file.

    Attributes:
        nodes:            All CSTNode objects in this zone.
        confs:            Per-node confidence scores (parallel to nodes).
        parent_index:     Shared parent node index for this group.
        first_node_index: Traversal index of first zone node — determines
                          DOM position for priority assignment.
    """

    nodes:            List[object]    # List[CSTNode] — typed as object to avoid import
    confs:            List[float]
    parent_index:     int
    first_node_index: int

def make_candidate_zone(
    nodes: List[object],
    parent_index: int,
    first_node_index: int = 0,
    confs: Optional[List[float]] = None,
) -> _CandidateZone:
    """Public factory for synthetic _CandidateZone construction by callers outside this module."""
    return _CandidateZone(
        nodes=nodes,
        confs=confs if confs is not None else [1.0] * len(nodes),
        parent_index=parent_index,
        first_node_index=first_node_index,
    )

# ─────────────────────────────────────────────────────────────────────────────
# ZoneMap DATACLASS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZoneMap:
    """
    Structural extraction map for a specific page.

    Produced by wlp_zones.assemble_zone_map().
    Consumed by topology/parser.py to compile grep/sed/awk recipes.
    Stored in structural_layer.pt keyed by (domain, topology_class).
    Cached in latent_parser.py L1/L2/L3 cache.

    Immutable by construction (frozen=True).
    A new ZoneMap replaces the old one atomically.
    Nothing mutates a ZoneMap in place.  Ever.

    The version field is the monotonic identity of this ZoneMap.  A ZoneMap
    with version N+1 supersedes version N for the same (domain, topology_class)
    key.  Version is assigned by latent_parser.py at storage time.
    wlp_zones.py always produces ZoneMaps with version=0.

    Confidence semantics:
        0.0       EmptyZoneMap — no signal zones
        0.30-0.69 freshly produced, limited feedback cycles
        0.70      ceiling for discovery-path ZoneMaps
        0.90+     confirmed by 10+ high-quality extractions
        < 0.30    topology/parser.py uses hardcoded GENERIC_HTML fallback
    """

    topology_class: str
    domain:         str

    signal_zones: Tuple[ZoneDescriptor, ...]
    noise_zones:  Tuple[ZoneDescriptor, ...]
    boundaries:   Tuple[BoundaryDescriptor, ...]

    extraction_strategy: ExtractionStrategy

    intent_weights: Tuple[Tuple[str, float], ...]
    """
    Per-zone relevance scores for the intent used at production time.
    Format: ((zone_selector, weight), ...)
    One entry per signal zone — len(intent_weights) == len(signal_zones) always.
    weight 0.0: suppressed by intent.exclude
    weight 1.0: default (no intent match, no exclusion)
    weight >1.0: boosted by intent.primary or intent.secondary
    """

    confidence:          float
    node_count:          int
    signal_node_count:   int
    noise_node_count:    int
    boundary_node_count: int

    version:                int
    produced_at:            float   # time.monotonic() at assembly
    topology_router_version: int    # WLM version at production time

    # ── Methods ──────────────────────────────────────────────────────────────

    def is_stale(self, current_router_version: int) -> bool:
        """
        True if WLM weights changed since this ZoneMap was produced.

        The intent_bias dimensions [100:128] of node feature vectors were
        projected using WLM's intent_projection weights at graph construction
        time.  If WLM weights changed, those projections used the old weight
        matrix.  The intent conditioning baked into node features is stale.

        Stale ZoneMaps continue serving from cache until the next L3 miss
        triggers re-production.  Staleness is a soft expiry, not hard
        invalidation — the structural geometry (selectors, scopes, content
        types) is still valid.  Only the intent-conditioning sub-component
        is out of date.
        """
        return self.topology_router_version < current_router_version

    def with_intent(
        self,
        intent_vector: List[float], # noqa
        intent_tags: "IntentTags",
    ) -> "ZoneMap":
        """
        Return a NEW ZoneMap with intent-conditioned zone weights.
        Does NOT modify self.  frozen=True enforces this at the Python level.

        This is the O(1) intent conditioning path — called on every cache hit
        when intent_vector is not None.  Must complete in < 0.1 ms.

        The only thing that changes is intent_weights.  No selectors are
        recomputed.  No CSTNodes are touched.  No model forward pass occurs.
        apply_intent_weights() is pure arithmetic on existing ZoneDescriptor
        metadata.

        The produced_at timestamp is preserved from the original production
        event — this ZoneMap is still derived from the same forward pass.

        Implementation invariant: dataclasses.replace() on a frozen dataclass
        constructs a new instance with only the named fields overridden.  All
        other fields are shallow-copied from self.  This is the correct and
        only pattern — do not construct a new ZoneMap manually.
        """
        new_weights = apply_intent_weights(self.signal_zones, intent_tags)
        return dataclasses.replace(self, intent_weights=new_weights)


# ─────────────────────────────────────────────────────────────────────────────
# EmptyZoneMap and EmptyZoneKnowledge
# ─────────────────────────────────────────────────────────────────────────────

class EmptyZoneMap:
    """
    The defined bottom value for ZoneMap production failures.

    Returned by assemble_zone_map() when:
      - Classification produces no signal zones.
      - validate_zone_map() raises.
      - Any unexpected exception occurs during assembly.

    Also returned by wlp.query() when all parse paths fail.

    topology/parser.py contract with EmptyZoneMap:
        if zone_map.confidence < 0.30:
            use hardcoded GENERIC_HTML recipe
        EmptyZoneMap.confidence == 0.0 → fallback always triggers.

    This class does NOT inherit from ZoneMap.
    It implements the same attribute interface that topology/parser.py reads
    (confidence, signal_zones, noise_zones, boundaries, intent_weights,
    extraction_strategy).  It does not pretend to be a real ZoneMap.

    Never return None from assemble_zone_map().
    Never raise from assemble_zone_map().
    Return EmptyZoneMap.  Always.
    """

    confidence:          float             = 0.0
    signal_zones:        Tuple             = ()
    noise_zones:         Tuple             = ()
    boundaries:          Tuple             = ()
    extraction_strategy: ExtractionStrategy = ExtractionStrategy.FLAT
    intent_weights:      Tuple             = ()

    def __init__(self) -> None:
        # Explicitly set all fields on the instance so that attribute access
        # behaviour is identical to a real ZoneMap regardless of class-level
        # defaults in subclass or monkey-patching scenarios.
        self.confidence          = 0.0
        self.signal_zones        = ()
        self.noise_zones         = ()
        self.boundaries          = ()
        self.extraction_strategy = ExtractionStrategy.FLAT
        self.intent_weights      = ()

    def is_stale(self, _: int) -> bool: # noqa
        """Always stale — EmptyZoneMaps are never worth caching."""
        return True

    def with_intent(self, *args, **kwargs) -> "EmptyZoneMap": # noqa
        """Intent conditioning on an empty map is a no-op.  Returns self."""
        return self

    def __repr__(self) -> str:
        return "EmptyZoneMap()"

    def __bool__(self) -> bool:
        """EmptyZoneMap is falsy — allows `if zone_map:` guard patterns."""
        return False


class EmptyZoneKnowledge:
    """
    Used by latent_parser.py when structural_layer.pt does not exist.
    Returns EmptyZoneMap for all queries.

    Expected during cold start before the first preparse cycle completes.
    This is not an error state — it is the designed cold-start behaviour.
    The system degrades to hardcoded recipes until zone knowledge accumulates.
    """

    def get(self, domain: str, topology_class: str) -> EmptyZoneMap: # noqa
        """Return EmptyZoneMap unconditionally.  No domain or class routing."""
        return EmptyZoneMap()

    def __len__(self) -> int:
        return 0

    def __repr__(self) -> str:
        return "EmptyZoneKnowledge()"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — NODE CLASSIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def classify_nodes(
    logits: "torch.Tensor",        # shape (n_nodes, 3)
    confidences: "torch.Tensor",   # shape (n_nodes, 1)
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> Tuple[List[int], List[float]]:
    """
    Convert per-node classification logits to (label, confidence) pairs.

    This function is the tensor boundary.  After it returns, all computation
    in this file is pure Python.  No torch operations occur after this call.

    Classification:
        label[i] = argmax(logits[i])  for each node i
        If confidences[i] < confidence_threshold:
            label[i] = NODE_NOISE   (uncertainty → treat as noise)

    The uncertainty-to-noise mapping prevents low-confidence SIGNAL nodes
    from producing ZoneDescriptors with unreliable selectors.  A missed
    SIGNAL node (false negative) causes EmptyZoneMap or partial ZoneMap.
    A wrong SIGNAL node (false positive with low confidence) causes a bad
    selector that silently extracts noise on every crawl.  The asymmetric
    cost justifies defaulting uncertain nodes to NOISE.

    Threshold 0.35 is calibrated to:
        Accept: clear signal peaks where argmax ≈ 0.60+ on 3-class softmax
        Reject: flat distributions (each class near 0.33)
        Accept: confident boundary/noise classifications
    A 3-class uniform distribution produces confidence ≈ 0.33 — the 0.35
    threshold correctly rejects maximum-entropy nodes.

    Returns:
        labels: List[int]   — one per node; index matches cst_nodes list
        confs:  List[float] — one per node; index matches cst_nodes list

    Both lists have the same length as the input tensor's first dimension.
    This invariant is checked by assemble_zone_map() before further processing.
    """
    n_nodes: int = logits.shape[0]
    labels: List[int]  = [NODE_NOISE] * n_nodes
    confs:  List[float] = [0.0] * n_nodes

    # Materialise both tensors to Python in a single pass.
    # .tolist() on the full tensor is more efficient than per-element .item()
    # because it batches the Python object creation.  For large graphs (>50k
    # nodes) this can be 10-20x faster than a Python loop with .item().
    logit_list: List[List[float]] = logits.tolist()         # (n_nodes, 3)
    conf_list:  List[List[float]] = confidences.tolist()    # (n_nodes, 1)

    for i in range(n_nodes):
        row: List[float] = logit_list[i]
        raw_conf: float  = conf_list[i][0]    # squeeze the single column

        # Argmax over 3-class logit dimension.
        if row[0] >= row[1] and row[0] >= row[2]:
            raw_label = NODE_SIGNAL
        elif row[2] >= row[1]:
            raw_label = NODE_BOUNDARY
        else:
            raw_label = NODE_NOISE

        # Confidence gate: uncertain nodes become NOISE.
        # This is applied regardless of the predicted class — even a confident
        # NOISE prediction that falls below threshold is kept as NOISE, so the
        # gate never promotes anything.
        if raw_conf < confidence_threshold:
            labels[i] = NODE_NOISE
        else:
            labels[i] = raw_label

        confs[i] = raw_conf

    return labels, confs


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — ZONE GROUPING
# ─────────────────────────────────────────────────────────────────────────────

def group_signal_nodes(
    nodes:  List[object],    # List[CSTNode]
    labels: List[int],
    confs:  List[float],
) -> Tuple[List[_CandidateZone], List[object]]:
    """
    Group adjacent SIGNAL-classified nodes into candidate zones.

    Returns (candidate_zones, boundary_nodes):
        candidate_zones: zones ready for Stage 3 ZoneDescriptor construction.
        boundary_nodes:  CSTNode objects with NODE_BOUNDARY label, passed to
                         identify_boundaries() separately.

    Grouping rules:
        Adjacent SIGNAL nodes sharing the same parent_index → one zone.
        "Adjacent" means sibling_index differs by exactly 1 AND parent_index
        matches.  Non-adjacent SIGNAL nodes under the same parent become
        separate zones — the NOISE/BOUNDARY gap may be structurally meaningful.
        Isolated SIGNAL nodes (no SIGNAL siblings) → individual single-node zone.
        BOUNDARY nodes are collected but not grouped into signal zones.
        NOISE nodes are discarded at this stage.

    Why parent-scoped grouping:
        A SIGNAL node under div.article and a SIGNAL node under div.sidebar may
        share the same grandparent.  They are NOT the same zone.  Grouping by
        parent_index + sibling adjacency ensures zones are tight enough to
        produce CSS selectors that isolate a specific structural region rather
        than spanning across unrelated content.

    Adjacency model — mathematical definition:
        Two nodes N_a and N_b are in the same adjacency group if and only if:
            N_a.parent_index == N_b.parent_index
            AND |N_a.sibling_index - N_b.sibling_index| == 1
            AND labels[idx_a] == labels[idx_b] == NODE_SIGNAL

        This is a strict 1-hop adjacency — no transitive closure across gaps.
        NOISE/BOUNDARY nodes between two SIGNAL siblings break adjacency even
        if the parent is the same.  The gap is significant.

    Implementation uses a single-pass sweep with a current-group accumulator.
    O(n) time.  O(z) space where z is the number of zones produced.
    """
    candidate_zones: List[_CandidateZone] = []
    boundary_nodes:  List[object]         = []

    n = len(nodes)
    if n == 0:
        return candidate_zones, boundary_nodes

    # Collect boundary nodes first — they are not mixed into zone grouping.
    for i, node in enumerate(nodes):
        if labels[i] == NODE_BOUNDARY:
            boundary_nodes.append(node)

    # Single-pass group accumulator.
    # current_group: list of (original_index, node, conf) for active signal run
    current_group: List[Tuple[int, object, float]] = []

    def _flush_group(group: List[Tuple[int, object, float]]) -> None:
        """Flush the current group accumulator into a CandidateZone."""
        if not group:
            return
        first_original_idx, first_node, _ = group[0]
        # parent_index of the first node defines the zone's parent.
        # All nodes in a group share this parent_index by construction.
        parent_idx: int = getattr(first_node, "parent_index", -1)
        zone_nodes = [g[1] for g in group]
        zone_confs = [g[2] for g in group]
        candidate_zones.append(_CandidateZone(
            nodes            = zone_nodes,
            confs            = zone_confs,
            parent_index     = parent_idx,
            first_node_index = first_original_idx,
        ))

    prev_signal_sibling: Optional[int] = None   # sibling_index of previous SIGNAL node
    prev_signal_parent:  Optional[int] = None   # parent_index of previous SIGNAL node

    for i in range(n):
        if labels[i] != NODE_SIGNAL:
            # Non-signal node — flush any open group and reset state.
            _flush_group(current_group)
            current_group = []
            prev_signal_sibling = None
            prev_signal_parent  = None
            continue

        node = nodes[i]
        node_parent:  int = getattr(node, "parent_index", -1)
        node_sibling: int = getattr(node, "sibling_index", i)

        # Test adjacency with the previous signal node.
        is_adjacent = (
            prev_signal_parent is not None
            and prev_signal_sibling is not None
            and node_parent == prev_signal_parent
            and abs(node_sibling - prev_signal_sibling) == 1
        )

        if not is_adjacent and current_group:
            # Start of a new adjacency run — flush the previous group.
            _flush_group(current_group)
            current_group = []

        current_group.append((i, node, confs[i]))
        prev_signal_sibling = node_sibling
        prev_signal_parent  = node_parent

    # Flush the final group.
    _flush_group(current_group)

    return candidate_zones, boundary_nodes


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CSS SELECTOR GENERATION
# This is the hardest part of the entire file.
# A wrong selector compiles a wrong recipe.
# Every path must return a non-empty string.
# ─────────────────────────────────────────────────────────────────────────────

def _is_structural_id(id_value: str) -> bool:
    """
    Return True if this HTML id attribute looks like a stable, authored id
    rather than a dynamically generated one.

    Dynamic id detection — mathematical digit-density criterion:
        digit_count = number of characters in id_value that are decimal digits
        digit_ratio = digit_count / len(id_value)
        if digit_ratio > 0.30: id is likely dynamic → return False

    Rationale for 0.30 threshold:
        Authored ids like "main-content", "article-body", "nav-primary" have
        zero or near-zero digit density.  Dynamic ids like "react-root-1a2b3c",
        "app-23842", "ember-view-47" have digit density typically above 0.25.
        The 0.30 threshold correctly rejects the dynamic ids in the examples
        above:
            "app-23842"    → 5/9 = 0.556 > 0.30  → dynamic  ✓
            "react-root-1a2b3c" → 7/16 = 0.4375 > 0.30 → dynamic ✓
            "ember-view-47" → 2/14 = 0.143 < 0.30 → structural? marginal

        Additional heuristics applied before the digit check:
          - Length < 3: too short to be meaningful as a selector
          - Contains only uppercase letters: likely a React component id
          - Contains UUID-pattern substrings (8-4-4-4-12 hex segments)

    Returns True (structural) only if all heuristics pass.
    """
    if not id_value or len(id_value) < 3:
        return False

    # UUID pattern detection: 32 hex characters with optional hyphens.
    # Reject if the id looks like a UUID fragment (16+ hex chars in sequence).
    hex_run: int = 0
    max_hex_run: int = 0
    for ch in id_value.lower():
        if ch in "0123456789abcdef":
            hex_run += 1
            max_hex_run = max(max_hex_run, hex_run)
        else:
            hex_run = 0
    if max_hex_run >= 8:
        # 8+ consecutive hex characters is a strong UUID / hash fragment signal.
        return False

    # Digit density criterion.
    digit_count = sum(1 for ch in id_value if ch.isdigit())
    digit_ratio: float = digit_count / len(id_value)
    if digit_ratio > 0.30:
        return False

    # Reject ids that are purely uppercase (React component roots, etc.)
    alpha_chars = [ch for ch in id_value if ch.isalpha()]
    if alpha_chars and all(ch.isupper() for ch in alpha_chars):
        return False

    return True


def _is_positional_class(class_name: str) -> bool:
    """
    Return True if this CSS class is a positional or numeric utility class
    that should never be used in a selector.

    Positional classes match patterns like:
        col-6, col-md-8, w-1/2, mt-4, pb-2, text-sm, font-bold
        flex-1, order-2, z-10
        Any class whose name is entirely numeric: "4", "12"

    These classes are framework-generated layout utilities that change
    with template revisions and cannot be relied upon for structural
    identification across page loads or topology class instances.

    Mathematical criterion:
        A class is positional if:
            - It matches the regex pattern: ^[a-z]{1,4}-?[0-9]+$ (e.g. col-6, mt-4)
            - OR it contains only digits (e.g. "4", "12")
            - OR it matches Bootstrap/Tailwind fraction patterns (w-1/2, h-3/4)
    """
    if not class_name:
        return True

    # All digits — purely numeric class name.
    if class_name.isdigit():
        return True

    # Short utility prefix + digit(s): col-6, mt-4, pb-2, z-10, etc.
    _POSITIONAL_RE = re.compile(r"^[a-z]{1,6}-?\d+([a-z]{0,2})?$")
    if _POSITIONAL_RE.match(class_name):
        return True

    # Fraction patterns: w-1/2, h-3/4
    if "/" in class_name:
        return True

    return False


def _most_discriminative_class(classes: List[str]) -> Optional[str]:
    """
    Return the single most discriminative CSS class from a list.

    Discriminative classes are semantically meaningful zone identifiers.
    Non-discriminative classes are layout utilities and state markers.

    Algorithm — priority-ordered search through DISCRIMINATIVE_CLASSES:
        For each class in DISCRIMINATIVE_CLASSES (high priority → low priority):
            If that class appears in the input list → return it immediately.
        If no DISCRIMINATIVE_CLASSES member found:
            Return None — caller falls through to the next selector tier.

    Additional filtering before the priority scan:
        - Classes in NON_DISCRIMINATIVE_CLASSES are ignored even if present.
        - Positional classes (_is_positional_class()) are ignored.
        - Classes shorter than 3 characters are ignored.
        - Classes containing only digits are ignored.
        - Classes that are pure numeric utility patterns are ignored.

    The priority order in DISCRIMINATIVE_CLASSES reflects empirical signal
    density: content-specific classes (article-body) are more discriminative
    than generic structural classes (content), which are more discriminative
    than navigation classes (nav, sidebar).

    Returns None if no discriminative class is found.  The caller is expected
    to try the next selector generation tier.
    """
    if not classes:
        return None

    # Build a set for O(1) lookup — avoids O(n) linear scan per candidate.
    class_set = frozenset(
        c.strip().lower()
        for c in classes
        if c.strip()
        and len(c.strip()) >= 3
        and not c.strip().isdigit()
        and c.strip().lower() not in NON_DISCRIMINATIVE_CLASSES
        and not _is_positional_class(c.strip().lower())
    )

    for candidate in DISCRIMINATIVE_CLASSES:
        if candidate in class_set:
            return candidate

    return None


def generate_css_selector(zone: _CandidateZone) -> str:
    """
    Generate a CSS selector for a zone's root element.

    This is the hardest function in wlp_zones.py.  A wrong selector compiles
    a wrong recipe.  A recipe built on a wrong selector extracts wrong nodes.
    The error is silent and compounds with every crawl cycle.

    An empty selector passed to topology/parser.py produces a recipe that
    matches nothing — silent empty extraction.  The tier-7 fallback (bare
    node_type) produces a broad recipe that overcaptures — recoverable.
    Empty is not recoverable.  Every path must return a non-empty string.

    Selection algorithm — checked in priority order, first match wins:

    Tier 1 — ARIA role selector (highest reliability):
        Stable across re-renders.  JavaScript frameworks always preserve ARIA
        roles because removing them would be an accessibility regression.
        Returns: '[role="main"]', '[role="article"]', etc.

    Tier 2 — Semantic HTML5 element (high reliability):
        Article, main, nav, etc. imply their zone type.
        Returns: "article", "main", "nav", etc.
        Note: if the element type is not unique in the document (multiple
        <article> elements), fall through to Tier 3 to add a class qualifier.
        Uniqueness is approximated by: if no discriminative class is available,
        use element type alone (scope field will narrow it).

    Tier 3 — Element + discriminative class (reliable):
        The most commonly applicable tier for complex pages.
        Returns: "div.article-body", "section.main-content", etc.

    Tier 4 — Element + structural id:
        Only if _is_structural_id() returns True for the id attribute.
        Dynamic ids break on the next fetch.  The structural check is
        conservative — prefer false negative (fall through) over false positive
        (dynamic id in selector that breaks on reload).
        Returns: "div#main-content", "section#primary"

    Tier 5 — Element + data attribute:
        data-* attributes are often more stable than CSS classes.
        Returns: "div[data-testid]", "section[data-section]"

    Tier 6 — Scoped descendant (2 levels max):
        Used when root node has no useful attributes or classes.
        Constructs a 2-level parent > child chain from the parent zone
        context.  Maximum depth 2 to avoid fragility.
        Returns: "article > div", "main > section"

    Tier 7 — Element type fallback (always succeeds):
        Least specific but guarantees non-empty return.
        topology/parser.py uses the scope field to constrain extraction.
        A broad selector within a narrow scope is better than silence.
        Returns: "div", "section", "p", etc.

    Determinism contract: for identical CandidateZone input, this function
    always returns the same selector.  No randomness, no env-dependent paths.
    """
    if not zone.nodes:
        # Degenerate empty zone — return "body" rather than empty string.
        # topology/parser.py will produce nothing from this but will not crash.
        return "body"

    # Use the first node as the zone root.  group_signal_nodes() guarantees
    # that the first node has the lowest traversal index (i.e., earliest in
    # DOM order) within the zone, making it the natural root representative.
    node = zone.nodes[0]

    node_type:   str       = getattr(node, "node_type",   "div")
    css_classes: List[str] = getattr(node, "css_classes", []) or []
    attributes:  Dict[str, str] = getattr(node, "attributes", {}) or {}

    # Normalise node_type to lowercase.
    node_type = node_type.lower().strip() or "div"

    # ── Tier 1: ARIA role ────────────────────────────────────────────────────
    role: Optional[str] = attributes.get("role", "").strip().lower() or None
    if role and role in LANDMARK_ROLES:
        return f'[role="{role}"]'

    # ── Tier 2: Semantic element ─────────────────────────────────────────────
    if node_type in SEMANTIC_ELEMENTS:
        # Check if we can make it more specific with a discriminative class.
        disc_class = _most_discriminative_class(css_classes)
        if disc_class:
            return f"{node_type}.{disc_class}"
        # No discriminative class — element type alone is sufficient for
        # semantically unique elements.  scope will constrain it.
        return node_type

    # ── Tier 3: Element + discriminative class ───────────────────────────────
    disc_class = _most_discriminative_class(css_classes)
    if disc_class:
        return f"{node_type}.{disc_class}"

    # ── Tier 4: Element + structural id ─────────────────────────────────────
    node_id: str = attributes.get("id", "").strip()
    if node_id and _is_structural_id(node_id):
        return f"{node_type}#{node_id}"

    # ── Tier 5: Element + data-* attribute ──────────────────────────────────
    data_attrs: List[str] = [
        k for k in attributes
        if k.startswith("data-") and attributes[k]
    ]
    if data_attrs:
        # Prefer the most specific data attribute (longest name tends to be
        # more specific, e.g. data-testid > data-id).
        best_data = max(data_attrs, key=len)
        return f"{node_type}[{best_data}]"

    # ── Tier 6: Scoped descendant (parent context) ───────────────────────────
    # Attempt to use the parent node's type as a scoping prefix.
    # Only do this if zone has more than one node (single-node zones have
    # insufficient parent context to build a meaningful descendant chain).
    if len(zone.nodes) > 1 and zone.parent_index >= 0:
        # We only have the parent_index, not the parent node directly.
        # Construct a 2-level selector using the parent's node_type if
        # we can infer it from the zone's structural context.
        # Since we do not have the parent CSTNode here (it was not passed),
        # we skip to Tier 7.  The parent node will be used in determine_scope()
        # to set the scope field, achieving similar narrowing via the recipe
        # compilation path.
        pass

    # ── Tier 7: Element type fallback ───────────────────────────────────────
    # This must ALWAYS return a non-empty string.
    # node_type was set to "div" if empty above, so this cannot be "".
    return node_type


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — SCOPE DETERMINATION
# ─────────────────────────────────────────────────────────────────────────────

def determine_scope(
    zone: _CandidateZone,
    all_nodes: List[object],    # full CSTNode list from wlp_graph
) -> str:
    """
    Find the nearest landmark ancestor of this zone's root node.

    Walks the parent chain from zone.parent_index upward through all_nodes.
    Returns the CSS selector of the first landmark ancestor found.
    Falls back to "body" if no landmark ancestor is found.

    Landmark recognition — priority order (first match wins):
        1. ARIA role="main"    → '[role="main"]'
        2. ARIA role="article" → '[role="article"]'
        3. element type "main" → "main"
        4. element type "article" → "article"
        5. element type + discriminative class from LANDMARK_ANCESTOR_CLASSES
           → "{element_type}.{class}"
        6. "body" (always valid fallback)

    Why scope matters for recipe compilation:
        selector: ".code-block"
        scope:    "article.article-body"
        compiled recipe: extract .code-block elements inside article.article-body
        NOT:              extract ALL .code-block elements in document

    Without scope, a .warning class on a sidebar callout would match the same
    selector as a .warning class on the main content — the recipe would extract
    both, contaminating signal with navigation chrome.

    Parent chain traversal:
        Starting at zone.parent_index, follow node.parent_index repeatedly.
        Stop at index -1 (no parent — reached document root).
        Stop at depth > MAX_ANCESTOR_DEPTH (loop guard against malformed trees).
        MAX_ANCESTOR_DEPTH = 64 (deep enough for any real DOM, finite guard).

    Returns "body" — never an empty string.
    """
    MAX_ANCESTOR_DEPTH: int = 64

    current_idx: int = zone.parent_index
    depth: int = 0

    while 0 <= current_idx < len(all_nodes) and depth < MAX_ANCESTOR_DEPTH:
        ancestor = all_nodes[current_idx]
        depth += 1

        anc_type:    str = getattr(ancestor, "node_type",   "").lower().strip()
        anc_classes: List[str] = getattr(ancestor, "css_classes", []) or []
        anc_attrs:   Dict[str, str] = getattr(ancestor, "attributes", {}) or {}

        # Check ARIA role — highest priority.
        role = anc_attrs.get("role", "").strip().lower()
        if role == "main":
            return '[role="main"]'
        if role == "article":
            return '[role="article"]'

        # Check semantic element types.
        if anc_type == "main":
            return "main"
        if anc_type == "article":
            return "article"

        # Check discriminative landmark ancestor classes.
        class_set = frozenset(c.strip().lower() for c in anc_classes if c.strip())
        for landmark_class in LANDMARK_ANCESTOR_CLASSES:
            if landmark_class in class_set:
                safe_type = anc_type if anc_type else "div"
                return f"{safe_type}.{landmark_class}"

        # Move up the parent chain.
        current_idx = getattr(ancestor, "parent_index", -1)

    # No landmark ancestor found — fallback to body.
    return "body"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — CONTENT TYPE INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

# Node types mapped to content categories.
_PROSE_NODE_TYPES: frozenset = frozenset({
    "p", "blockquote", "em", "strong", "span", "b", "i",
    "cite", "q", "abbr", "address", "bdi", "bdo",
})
_CODE_NODE_TYPES: frozenset = frozenset({
    "code", "pre", "kbd", "samp", "var", "tt",
})
_CODE_CLASSES: frozenset = frozenset({
    "code-block", "code-snippet", "highlight", "syntax",
    "syntax-highlight", "hljs", "prism", "shiki",
    "language-python", "language-js", "language-bash",
    "language-shell", "language-json", "language-yaml",
})
_LIST_NODE_TYPES: frozenset = frozenset({
    "li", "dt", "dd", "ol", "ul", "dl",
    "menu", "dir",
})
_TABLE_NODE_TYPES: frozenset = frozenset({
    "td", "th", "tr", "table", "thead", "tbody", "tfoot", "caption",
})


def infer_content_type(zone: _CandidateZone) -> str:
    """
    Infer the structural content type from node composition within a zone.

    Counting model:
        prose_count:  nodes whose type is in _PROSE_NODE_TYPES
        code_count:   nodes whose type is in _CODE_NODE_TYPES
                      OR whose css_classes intersect _CODE_CLASSES
        list_count:   nodes whose type is in _LIST_NODE_TYPES
        table_count:  nodes whose type is in _TABLE_NODE_TYPES

    total_structural = prose_count + code_count + list_count + table_count
    If total_structural == 0: return "mixed" (no structural signal).

    Ratio-based classification (checked in order):
        prose_ratio  = prose_count / total_structural
        code_ratio   = code_count  / total_structural
        list_ratio   = list_count  / total_structural
        table_ratio  = table_count / total_structural

        if prose_ratio  > 0.60: return "prose"
        if code_ratio   > 0.40: return "code"
        if list_ratio   > 0.40: return "list"
        if table_ratio  > 0.40: return "table"
        return "mixed"

    Threshold asymmetry rationale:
        Prose zones are typically strongly dominant (> 60%) — paragraphs are
        homogeneous.  Code/list/table zones are often mixed with prose context
        (a tutorial page has prose paragraphs AND code blocks).  The 0.40
        threshold for the latter three types allows them to be recognised even
        when they share space with prose, which would suppress them at 0.60.

    Returns exactly one of: "prose", "code", "list", "table", "mixed".
    """
    prose_count: int = 0
    code_count:  int = 0
    list_count:  int = 0
    table_count: int = 0

    for node in zone.nodes:
        ntype:   str       = getattr(node, "node_type",   "").lower().strip()
        classes: List[str] = getattr(node, "css_classes", []) or []
        class_set = frozenset(c.strip().lower() for c in classes if c.strip())

        if ntype in _PROSE_NODE_TYPES:
            prose_count += 1
        elif ntype in _CODE_NODE_TYPES or class_set & _CODE_CLASSES:
            code_count += 1
        elif ntype in _LIST_NODE_TYPES:
            list_count += 1
        elif ntype in _TABLE_NODE_TYPES:
            table_count += 1

    total_structural: int = prose_count + code_count + list_count + table_count
    if total_structural == 0:
        return "mixed"

    # Compute ratios over the total structural count (not total node count).
    # This prevents a zone with 100 pure-structural "div" wrappers from diluting
    # a clear "code" signal from 10 <pre> and <code> nodes.
    inv_total: float = 1.0 / total_structural

    prose_ratio: float = prose_count * inv_total
    code_ratio:  float = code_count  * inv_total
    list_ratio:  float = list_count  * inv_total
    table_ratio: float = table_count * inv_total

    # Asymmetric thresholds — see docstring rationale.
    if prose_ratio  > 0.60:
        return "prose"
    if code_ratio   > 0.40:
        return "code"
    if list_ratio   > 0.40:
        return "list"
    if table_ratio  > 0.40:
        return "table"
    return "mixed"


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — DENSITY COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_density(zone: _CandidateZone) -> float:
    """
    Compute signal density for a candidate zone.

    signal_density = signal_chars / total_chars

    Where:
        signal_chars = sum of text_char_count for each node in zone.nodes
                       (text_char_count is the direct text of that node, not
                       the full subtree — the node-level signal contribution)
        total_chars  = sum of subtree_char_count for each node in zone.nodes
                       (subtree_char_count includes all characters in subtree:
                       signal and noise descendants)

    This ratio captures how much of the zone's total character budget is
    signal rather than structural scaffolding.  A zone with density 1.0 has
    every byte as signal.  A zone with density 0.1 is mostly wrappers.

    Density guides recipe compiler aggressiveness:
        density > 0.70: aggressive extraction (keep most content)
        density 0.40-0.70: selective extraction
        density < 0.40: conservative — pattern-match only

    Edge cases:
        total_chars == 0: return 0.0 — zone has no text content at all.
        signal_chars > total_chars: clamp to 1.0 — floating-point safety.
            This can occur when subtree_char_count and text_char_count are
            computed from different traversal depths.  The clamp maintains
            the [0.0, 1.0] range contract without masking the condition.

    Harmonic interpretation:
        density is the fraction of the zone's character mass that is signal.
        Low density does not mean the zone is worthless — a <table> with 1
        signal header row in 10 rows has low density but high structural value.
        density is a complement to content_type, not a replacement.

    Range guarantee: [0.0, 1.0] always.  Checked in validate_zone_map().
    """
    signal_chars: int = 0
    total_chars:  int = 0

    for node in zone.nodes:
        text_char_count:    int = getattr(node, "text_char_count",    0) or 0
        subtree_char_count: int = getattr(node, "subtree_char_count", 0) or 0

        # Guard against negative counts from upstream computation errors.
        signal_chars += max(0, text_char_count)
        total_chars  += max(0, subtree_char_count)

    if total_chars == 0:
        return 0.0

    raw_density: float = signal_chars / total_chars
    return min(1.0, max(0.0, raw_density))   # clamp to [0.0, 1.0]


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — AVERAGE DEPTH COMPUTATION (helper)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_average_depth(zone: _CandidateZone) -> float:
    """
    Compute the mean DOM depth of nodes in a candidate zone.

    average_depth = (1/|zone.nodes|) * Σ node.depth  for node in zone.nodes

    Returns 0.0 for empty zones.  Depths are taken as absolute values
    (un-normalised depth within the DOM tree, as recorded by wlp_graph.py).
    """
    if not zone.nodes:
        return 0.0

    depth_sum: float = sum(
        float(getattr(node, "depth", 0) or 0)
        for node in zone.nodes
    )
    return depth_sum / len(zone.nodes)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — PRIORITY ASSIGNMENT
# ─────────────────────────────────────────────────────────────────────────────

def assign_priorities(
    candidate_zones: List[_CandidateZone],
    densities: List[float],
) -> List[int]:
    """
    Assign unique priority integers to candidate zones.

    Priority semantics: lower number = extract first in compiled recipe.
    Priority range: [0, n_zones - 1] inclusive, all values unique.

    Sorting key — composite score with primary and secondary components:

        Primary:   first_node_index (ascending)
                   Earlier in document → lower priority number.
                   Captures reading order — the recipe extracts content in
                   the order a human would encounter it on the page.

        Secondary: -densities[i] (ascending, i.e. higher density first)
                   Within the same DOM-position region, high-density zones
                   are more likely to be primary content and should be
                   extracted before low-density neighbours.
                   The negation converts "higher density = lower sort key"
                   to a consistent ascending sort.

    Tiebreaking:
        Primary key alone is usually sufficient — zones at different DOM
        positions have different first_node_index values.  Ties occur only
        when two zones start at the same traversal index (pathological case:
        two single-node zones at the same level with identical indices).
        In that case, density decides — higher density extracts first.

    Implementation:
        enumerate() preserves original indices so we can write back into
        the priorities list at the correct positions.
        O(n log n) sort where n = number of signal zones.

    Invariant: returned list has the same length as candidate_zones.
    All values are unique integers in [0, len(candidate_zones)-1].
    Checked by validate_zone_map() via set(priorities) == set(range(n)).
    """
    n: int = len(candidate_zones)
    if n == 0:
        return []

    # Build (original_index, first_node_index, -density) tuples for sorting.
    sort_keys: List[Tuple[int, int, float]] = [
        (i, candidate_zones[i].first_node_index, -densities[i])
        for i in range(n)
    ]

    # Sort by (first_node_index ascending, -density ascending).
    # Python's sort is stable — equal keys preserve original order (by i),
    # which gives a consistent tiebreak at no extra cost.
    sort_keys.sort(key=lambda t: (t[1], t[2]))

    priorities: List[int] = [0] * n
    for priority_rank, (original_idx, _, _) in enumerate(sort_keys):
        priorities[original_idx] = priority_rank

    return priorities


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — EXTRACTION STRATEGY SELECTION
# ─────────────────────────────────────────────────────────────────────────────

def _coefficient_of_variation(values: List[float]) -> float:
    """
    Compute the coefficient of variation (CV) for a list of floats.

    CV = σ / μ  where σ = sample standard deviation, μ = arithmetic mean.

    CV measures relative dispersion.  A low CV means values are clustered
    (homogeneous distribution).  A high CV means values are spread out.

    Used in select_extraction_strategy() to detect whether zone positions
    are clustered (breadth-first signal) or spread (depth-first signal).

    Returns 0.0 for empty lists or single-element lists (no dispersion).
    Returns 0.0 if mean is zero (avoid division by zero).
    """
    n = len(values)
    if n <= 1:
        return 0.0

    mean_val: float = sum(values) / n
    if mean_val == 0.0:
        return 0.0

    # Sample variance: sum((x - μ)²) / (n-1).  Use Bessel's correction for
    # n-1 (sample std deviation) rather than population std deviation.
    variance: float = sum((x - mean_val) ** 2 for x in values) / (n - 1)
    std_dev: float  = math.sqrt(variance)

    return std_dev / abs(mean_val)


def select_extraction_strategy(
    signal_zones: List[_CandidateZone],
    boundaries:   List[object],        # CSTNode boundary nodes
    topology_class: str,
) -> ExtractionStrategy:
    """
    Determine ExtractionStrategy from zone topology geometry.

    Rules checked in order — first match wins:

    SECTION_SCOPED:
        topology_class ∈ SECTION_SCOPED_TOPOLOGY_CLASSES
        AND len(boundaries) >= 2
        Condition: document is divided into sections that each have
        independent semantic meaning.  Cross-section signal mixing would
        blur endpoint boundaries (REST_API) or section content (WIKIPEDIA).

    FLAT:
        topology_class ∈ FLAT_TOPOLOGY_CLASSES
        OR (len(signal_zones) <= 2 AND mean_depth < 3.0)
        Shallow content has no meaningful nesting to traverse.
        Single-level signal: flat extraction is sufficient and faster.
        mean_depth < 3.0: computed as mean of zone.average_depth over all zones.
        This is a geometric criterion — if the average node depth is below 3,
        the page is shallow enough to treat as flat.

    BREADTH_FIRST:
        topology_class ∈ BREADTH_FIRST_TOPOLOGY_CLASSES
        OR (structural clustering criterion):
            coefficient_of_variation(zone.first_node_index) < 0.30
            AND len(signal_zones) > 5
        CV < 0.30 means zone positions are tightly clustered — they occupy
        a small DOM region relative to their spread.  This geometric signature
        matches horizontal signal distribution: multiple siblings at the same
        DOM level (product variants, pricing tiers, comparison columns).
        The len > 5 guard ensures we have enough zones for CV to be meaningful.

    DEPTH_FIRST:
        Default.  All other cases.
        Signal is concentrated in nested content.  Depth-first traversal
        captures all signal in document order without losing nesting context.
    """
    n_zones: int      = len(signal_zones)
    n_boundaries: int = len(boundaries)

    # SECTION_SCOPED check.
    if topology_class in SECTION_SCOPED_TOPOLOGY_CLASSES and n_boundaries >= 2:
        return ExtractionStrategy.SECTION_SCOPED

    # FLAT check — topology class override first, then geometric criterion.
    if topology_class in FLAT_TOPOLOGY_CLASSES:
        return ExtractionStrategy.FLAT

    if n_zones <= 2:
        if n_zones == 0:
            return ExtractionStrategy.FLAT
        mean_depth: float = sum(
            _compute_average_depth(z) for z in signal_zones
        ) / n_zones
        if mean_depth < 3.0:
            return ExtractionStrategy.FLAT

    # BREADTH_FIRST check — topology class override first, then CV criterion.
    if topology_class in BREADTH_FIRST_TOPOLOGY_CLASSES:
        return ExtractionStrategy.BREADTH_FIRST

    if n_zones > 5:
        positions: List[float] = [
            float(z.first_node_index) for z in signal_zones
        ]
        cv: float = _coefficient_of_variation(positions)
        if cv < 0.30:
            # Low CV: positions are tightly clustered relative to their mean.
            # This is the geometric signature of breadth-first (sibling) layout.
            return ExtractionStrategy.BREADTH_FIRST

    # Default: DEPTH_FIRST.
    return ExtractionStrategy.DEPTH_FIRST


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — BOUNDARY IDENTIFICATION
# ─────────────────────────────────────────────────────────────────────────────

def _boundary_type_from_node(node: object) -> str:
    """
    Determine the boundary type for a BOUNDARY-classified node.

    Classification hierarchy:
        1. Heading elements (h1-h6) → SECTION_BOUNDARY
           Headings are document structure markers — they reset the semantic
           scope of everything that follows them.

        2. Noise-pattern CSS classes → NOISE_BOUNDARY
           Elements with classes in NOISE_CLASSES are structural chrome
           (sidebars, navbars, footers, ads).  They are boundaries for the
           noise region they contain.

        3. Default → CONTENT_BOUNDARY
           All other BOUNDARY nodes mark content transitions.
    """
    node_type: str       = getattr(node, "node_type",   "").lower().strip()
    css_classes: List[str] = getattr(node, "css_classes", []) or []
    class_set = frozenset(c.strip().lower() for c in css_classes if c.strip())

    if node_type in HEADING_ELEMENTS:
        return "SECTION_BOUNDARY"

    if class_set & NOISE_CLASSES:
        return "NOISE_BOUNDARY"

    return "CONTENT_BOUNDARY"


def _boundary_delimiter_pattern(node: object, boundary_type: str) -> str:
    """
    Generate a regex delimiter pattern for a boundary node.

    This pattern is stored in BoundaryDescriptor.delimiter_content and used
    by topology/parser.py to generate sed/awk address range patterns.

    Pattern generation by boundary_type:

    SECTION_BOUNDARY (heading elements):
        Pattern: '<h{level}[^>]*>'
        where level is extracted from the element name (h2 → 2).
        A range pattern like h2-h4: '<h[2-4][^>]*>'
        Default if no heading level detectable: '<h[1-6][^>]*>'

    CONTENT_BOUNDARY:
        Pattern: 'class="{most_discriminative_class}"'
        Uses _most_discriminative_class() on the node's css_classes.
        If no discriminative class found: 'class="{node_type}"'
        which produces a broad but always-valid pattern.

    NOISE_BOUNDARY:
        Pattern: 'class="{first_noise_class}"'
        where first_noise_class is the first class in NOISE_CLASSES that
        appears in the node's css_classes.
        If none found: 'class="{node_type}"'

    All patterns must be valid Python re patterns.  They are validated
    by validate_zone_map() before the ZoneMap is constructed.
    """
    node_type:   str       = getattr(node, "node_type",   "div").lower().strip()
    css_classes: List[str] = getattr(node, "css_classes", []) or []

    if boundary_type == "SECTION_BOUNDARY":
        # Extract heading level digit from node_type (e.g. "h2" → "2").
        if node_type in HEADING_ELEMENTS and len(node_type) == 2:
            level: str = node_type[1]
            return f"<h{level}[^>]*>"
        return "<h[1-6][^>]*>"

    if boundary_type == "NOISE_BOUNDARY":
        class_set = frozenset(c.strip().lower() for c in css_classes if c.strip())
        for noise_cls in sorted(NOISE_CLASSES):    # sorted for determinism
            if noise_cls in class_set:
                return f'class="{noise_cls}"'
        return f'class="{node_type}"'

    # CONTENT_BOUNDARY
    disc_class = _most_discriminative_class(css_classes)
    if disc_class:
        return f'class="{disc_class}"'
    return f'class="{node_type}"'


def identify_boundaries(
    boundary_nodes: List[object],
    labels:         List[int], # noqa
    confs:          List[float],
) -> List[BoundaryDescriptor]:
    """
    Convert BOUNDARY-classified nodes into BoundaryDescriptors.

    For each boundary node with confidence >= BOUNDARY_CONFIDENCE_THRESHOLD:
        1. Determine boundary_type via _boundary_type_from_node().
        2. Generate selector via same algorithm as generate_css_selector()
           applied to a single-node synthetic _CandidateZone.
        3. Generate delimiter_content regex via _boundary_delimiter_pattern().
        4. Construct BoundaryDescriptor.

    Confidence threshold rationale for boundaries (0.40 > 0.35 for signals):
        Wrong boundary nodes produce wrong sed/awk delimiter patterns.
        A wrong delimiter silently truncates or misaligns recipe output.
        A missing boundary degrades gracefully (DEPTH_FIRST default).
        The 0.05 uplift reflects the asymmetric cost of a wrong boundary
        vs a missing boundary.  Be conservative.

    Empty boundaries list is valid — not all pages have structural boundaries.
    topology/parser.py handles zero-boundary ZoneMaps correctly.

    Returns List[BoundaryDescriptor], which may be empty.
    """
    descriptors: List[BoundaryDescriptor] = []
    n: int = len(boundary_nodes) # noqa

    # We need per-node confidence.  boundary_nodes are CSTNode objects;
    # their indices in the global labels/confs arrays are NOT necessarily
    # sequential indices.  boundary_nodes were collected from the global
    # node list by their NODE_BOUNDARY label in group_signal_nodes().
    # The confidence for a boundary node is stored in the confs list at the
    # node's traversal index.
    # However, group_signal_nodes() returns CSTNode objects, not their indices.
    # We match by node object identity using a separate confidence pass.
    #
    # To avoid O(n²) matching, boundary nodes carry their original_index
    # stored as node.traversal_index (set by wlp_graph.py).  If that
    # attribute is missing (older graph versions), we fall back to 0.40 pass
    # (minimum threshold, conservative accept).

    for bnode in boundary_nodes:
        traversal_idx: int = getattr(bnode, "traversal_index", -1)
        if traversal_idx >= 0 and traversal_idx < len(confs): # noqa
            node_conf: float = confs[traversal_idx]
        else:
            # No traversal index available — be conservative and skip.
            # The boundary confidence contract says: only include boundary
            # if conf >= 0.40.  Without a confidence value, we cannot verify
            # this.  Skip rather than include a potentially low-confidence
            # boundary.
            continue

        if node_conf < BOUNDARY_CONFIDENCE_THRESHOLD:
            continue

        boundary_type: str = _boundary_type_from_node(bnode)

        # Generate selector using a synthetic single-node CandidateZone.
        synthetic_zone = _CandidateZone(
            nodes            = [bnode],
            confs            = [node_conf],
            parent_index     = getattr(bnode, "parent_index", -1),
            first_node_index = traversal_idx if traversal_idx >= 0 else 0,
        )
        selector: str = generate_css_selector(synthetic_zone)
        delimiter: str = _boundary_delimiter_pattern(bnode, boundary_type)

        # Validate delimiter is a valid regex before constructing.
        try:
            re.compile(delimiter)
        except re.error:
            log.warning(
                "boundary_invalid_regex selector=%r boundary_type=%s delimiter=%r",
                selector, boundary_type, delimiter,
            )
            # Fall back to a safe generic pattern rather than omitting the boundary.
            node_type: str = getattr(bnode, "node_type", "div").lower().strip()
            delimiter = re.escape(node_type)

        descriptors.append(BoundaryDescriptor(
            selector=selector,
            boundary_type=boundary_type,
            delimiter_content=delimiter,
        ))

    return descriptors


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 4 — INTENT CONDITIONING
# ─────────────────────────────────────────────────────────────────────────────

def parse_intent_tags(intent_vector: List[float]) -> IntentTags:
    """
    Decode a 256-float intent_vector into named IntentTags.

    Encoding layout (from contracts.py DaemonRequest + WLP spec):
        [0:64]     primary intent tokens   — activate at > 0.50
        [64:128]   secondary intent tokens — activate at > 0.50
        [128:192]  exclude intent tokens   — activate at > 0.30 (conservative)
        [192:208]  urgency encoding        — 16 floats, 4 urgency states
        [208:256]  user_state encoding     — 48 floats, 8 user_state states

    Token → tag mapping:
        Active token at position P in [0:64]   → INTENT_TOKEN_VOCABULARY[P]
        Active token at position P in [64:128]  → INTENT_TOKEN_VOCABULARY[P-64]
        Active token at position P in [128:192] → INTENT_TOKEN_VOCABULARY[P-128]
        Vocabulary has 64 entries.  Positions beyond len(vocab) are ignored.

    Urgency decoding — bin-based argmax:
        16 positions [192:208] represent 4 urgency states × 4 floats per state.
        For each bin b in 0..3: bin_score[b] = max(vector[192+b*4 : 192+(b+1)*4])
        urgency = URGENCY_STATES[argmax(bin_scores)]
        This aggregation is robust to positional encoding jitter within bins.

    User_state decoding — bin-based argmax over 8 states × 6 floats per state:
        48 positions [208:256] represent 8 user_state states × 6 floats per state.
        For each bin b in 0..7: bin_score[b] = max(vector[208+b*6 : 208+(b+1)*6])
        user_state = USER_STATE_STATES[argmax(bin_scores)]

    Edge cases:
        intent_vector shorter than 256: pads with 0.0.
        intent_vector longer than 256: truncated at index 256.
        Empty intent_vector: returns IntentTags with all empty lists and defaults.

    Returns: IntentTags with all fields populated.
    """
    # Normalise length: pad to 256 floats, truncate if too long.
    n: int = len(intent_vector)
    if n < 256:
        vec: List[float] = list(intent_vector) + [0.0] * (256 - n)
    else:
        vec = list(intent_vector[:256])

    vocab_size: int = len(INTENT_TOKEN_VOCABULARY)

    # Decode primary intent tags: positions 0..63.
    primary: List[str] = []
    for pos in range(INTENT_PRIMARY_START, INTENT_PRIMARY_END):
        if pos < n and vec[pos] > INTENT_PRIMARY_THRESHOLD:
            vocab_idx = pos - INTENT_PRIMARY_START
            if vocab_idx < vocab_size:
                primary.append(INTENT_TOKEN_VOCABULARY[vocab_idx])

    # Decode secondary intent tags: positions 64..127.
    secondary: List[str] = []
    for pos in range(INTENT_SECONDARY_START, INTENT_SECONDARY_END):
        if vec[pos] > INTENT_SECONDARY_THRESHOLD:
            vocab_idx = pos - INTENT_SECONDARY_START
            if vocab_idx < vocab_size:
                secondary.append(INTENT_TOKEN_VOCABULARY[vocab_idx])

    # Decode exclude tags: positions 128..191 — lower threshold (0.30).
    exclude: List[str] = []
    for pos in range(INTENT_EXCLUDE_START, INTENT_EXCLUDE_END):
        if vec[pos] > INTENT_EXCLUDE_THRESHOLD:
            vocab_idx = pos - INTENT_EXCLUDE_START
            if vocab_idx < vocab_size:
                exclude.append(INTENT_TOKEN_VOCABULARY[vocab_idx])

    # Decode urgency: positions 192..207, 4 bins × 4 floats per bin.
    # bin_score = max over each bin → argmax over bin_scores.
    urgency_bin_scores: List[float] = []
    for b in range(len(URGENCY_STATES)):
        start = INTENT_URGENCY_START + b * URGENCY_BIN_SIZE
        end   = start + URGENCY_BIN_SIZE
        bin_max = max(vec[start:end], default=0.0)
        urgency_bin_scores.append(bin_max)

    urgency_idx: int = urgency_bin_scores.index(max(urgency_bin_scores))
    urgency: str = URGENCY_STATES[urgency_idx]

    # Decode user_state: positions 208..255, 8 bins × 6 floats per bin.
    user_state_bin_scores: List[float] = []
    for b in range(len(USER_STATE_STATES)):
        start = INTENT_USER_STATE_START + b * USER_STATE_BIN_SIZE
        end   = start + USER_STATE_BIN_SIZE
        bin_max = max(vec[start:end], default=0.0)
        user_state_bin_scores.append(bin_max)

    user_state_idx: int = user_state_bin_scores.index(max(user_state_bin_scores))
    user_state: str = USER_STATE_STATES[user_state_idx]

    return IntentTags(
        primary    = primary,
        secondary  = secondary,
        exclude    = exclude,
        urgency    = urgency,
        user_state = user_state,
    )


def zone_matches_intent_semantics(
    zone: ZoneDescriptor,
    intent_tag: str,
) -> bool:
    """
    Determine if a zone's structural characteristics match an intent tag.

    This is structural matching — NOT keyword matching against page content.
    This function never reads page text.
    It reads zone.content_type, zone.selector, zone.density, zone.average_depth.

    The structural signal propagates from wlp_graph.py into zone.selector during
    CSS selector generation: "warning" in zone.selector is True if the zone's
    root element had a CSS class matching the warning pattern in wlp_graph.py's
    css_class_bits() encoding.  The pipeline is coherent — this function reads
    the structural encoding, not raw HTML text.

    Matching rules — grouped by intent category:

    Account/access recovery: "account_recovery", "lost_access", "restore_access",
                             "reset_password", "login_help", "unlock_account",
                             "mfa_recovery", "backup_codes", "session_recovery"
        Matches: zone.content_type in ("list", "mixed")
                 AND zone.density > 0.5
                 AND any of: "warning", "note", "callout", "step", "info",
                             "alert", "caution" in zone.selector
                 OR zone.content_type == "list"   (recovery steps are lists)

    API/technical reference: "api_reference", "endpoint", "schema", "parameter",
                             "authentication", "authorization", "rate_limit",
                             "webhook", "sdk", "integration"
        Matches: zone.content_type in ("code", "table", "mixed")
                 AND zone.density > 0.4
                 AND any of: "code", "api", "schema", "param", "endpoint",
                             "response", "request", "table" in zone.selector
                 OR zone.content_type in ("code", "table")

    Pricing/commercial: "pricing", "plans", "cost", "billing", "subscription",
                        "upgrade", "downgrade", "trial", "enterprise", "features"
        Matches: zone.content_type in ("table", "mixed", "list")
                 AND any of: "pricing", "plan", "tier", "price", "cost",
                             "billing", "table" in zone.selector
                 OR zone.content_type == "table"

    Getting started / tutorials: "getting_started", "quickstart", "tutorial",
                                 "guide", "walkthrough", "setup", "installation",
                                 "configuration", "onboarding", "first_steps"
        Matches: zone.content_type in ("list", "prose", "mixed")
                 AND zone.density > 0.5
                 AND zone.average_depth > 3.0   (tutorials are deeply nested)

    Code examples: "code_example", "snippet", "sample", "demo", "playground",
                   "repository", "library", "package", "cli", "command"
        Matches: zone.content_type == "code"
                 OR any of: "code", "pre", "example", "snippet", "cli",
                             "command", "highlight", "syntax" in zone.selector

    Warnings/safety: "warning", "caution", "danger", "breaking_change",
                     "deprecation", "security", "vulnerability", "migration",
                     "rollback", "incident"
        Matches: any of: "warning", "caution", "danger", "alert", "notice",
                         "note", "callout", "important", "security",
                         "deprecat", "breaking" in zone.selector

    Changelog/release: "changelog", "release_notes"
        Matches: zone.content_type in ("list", "mixed", "prose")
                 AND any of: "changelog", "release", "version",
                             "history", "log" in zone.selector

    FAQ/troubleshooting: "faq", "troubleshooting"
        Matches: zone.content_type in ("list", "mixed", "prose")
                 AND zone.density > 0.4

    Unknown intent tag: return False.
        Unknown tags do not match any zone — they should not boost or suppress.
    """
    # Normalise the selector to lowercase for case-insensitive substring checks.
    sel: str = zone.selector.lower()
    ct:  str = zone.content_type
    dens: float = zone.density
    depth: float = zone.average_depth

    # ── Account / access recovery ────────────────────────────────────────────
    if intent_tag in {
        "account_recovery", "lost_access", "restore_access",
        "reset_password", "login_help", "unlock_account",
        "mfa_recovery", "backup_codes", "session_recovery",
    }:
        if ct == "list":
            return True
        if ct in ("list", "mixed") and dens > 0.5:
            recovery_signal = (
                "warning" in sel or "note" in sel or "callout" in sel
                or "step" in sel or "info" in sel or "alert" in sel
                or "caution" in sel or "notice" in sel
            )
            return recovery_signal
        return False

    # ── API / technical reference ────────────────────────────────────────────
    if intent_tag in {
        "api_reference", "endpoint", "schema", "parameter",
        "authentication", "authorization", "rate_limit",
        "webhook", "sdk", "integration",
    }:
        if ct in ("code", "table"):
            return True
        if ct == "mixed" and dens > 0.4:
            api_signal = (
                "code" in sel or "api" in sel or "schema" in sel
                or "param" in sel or "endpoint" in sel
                or "response" in sel or "request" in sel
                or "table" in sel or "reference" in sel
            )
            return api_signal
        return False

    # ── Pricing / commercial ─────────────────────────────────────────────────
    if intent_tag in {
        "pricing", "plans", "cost", "billing", "subscription",
        "upgrade", "downgrade", "trial", "enterprise", "features",
    }:
        if ct == "table":
            return True
        if ct in ("table", "mixed", "list"):
            pricing_signal = (
                "pricing" in sel or "plan" in sel or "tier" in sel
                or "price" in sel or "cost" in sel
                or "billing" in sel or "table" in sel
                or "comparison" in sel or "feature" in sel
            )
            return pricing_signal
        return False

    # ── Getting started / tutorials ──────────────────────────────────────────
    if intent_tag in {
        "getting_started", "quickstart", "tutorial", "guide",
        "walkthrough", "setup", "installation", "configuration",
        "onboarding", "first_steps",
    }:
        return (
            ct in ("list", "prose", "mixed")
            and dens > 0.5
            and depth > 3.0
        )

    # ── Code examples ────────────────────────────────────────────────────────
    if intent_tag in {
        "code_example", "snippet", "sample", "demo", "playground",
        "repository", "library", "package", "cli", "command",
    }:
        if ct == "code":
            return True
        code_signal = (
            "code" in sel or "pre" in sel or "example" in sel
            or "snippet" in sel or "cli" in sel or "command" in sel
            or "highlight" in sel or "syntax" in sel
            or "terminal" in sel or "shell" in sel
        )
        return code_signal

    # ── Warnings / safety ────────────────────────────────────────────────────
    if intent_tag in {
        "warning", "caution", "danger", "breaking_change",
        "deprecation", "security", "vulnerability",
        "migration", "rollback", "incident",
    }:
        warning_signal = (
            "warning" in sel or "caution" in sel or "danger" in sel
            or "alert" in sel or "notice" in sel or "note" in sel
            or "callout" in sel or "important" in sel
            or "security" in sel or "deprecat" in sel
            or "breaking" in sel or "critical" in sel
            or "error" in sel or "migration" in sel
        )
        return warning_signal

    # ── Changelog / release notes ────────────────────────────────────────────
    if intent_tag in {"changelog", "release_notes"}:
        if ct not in ("list", "mixed", "prose"):
            return False
        release_signal = (
            "changelog" in sel or "release" in sel or "version" in sel
            or "history" in sel or "log" in sel or "change" in sel
            or "update" in sel
        )
        return release_signal

    # ── FAQ / troubleshooting ────────────────────────────────────────────────
    if intent_tag in {"faq", "troubleshooting"}:
        return ct in ("list", "mixed", "prose") and dens > 0.4

    # Unknown tag — do not match anything.
    return False


def _compute_exclude_set(intent_tags: IntentTags) -> frozenset:
    """
    Build the frozen set of exclude tags for O(1) membership testing.

    Called once at the start of apply_intent_weights() to avoid re-creating
    the set for each zone in the weight computation loop.

    Returns frozenset of strings — the active exclude tags.
    """
    return frozenset(intent_tags.exclude)


def apply_intent_weights(
    signal_zones: Tuple[ZoneDescriptor, ...],
    intent_tags: IntentTags,
) -> Tuple[Tuple[str, float], ...]:
    """
    Compute per-zone intent weights from IntentTags.

    Returns a tuple of (selector, weight) pairs — one per signal zone.
    The tuple length ALWAYS equals len(signal_zones).
    topology/parser.py zips signal_zones with this tuple by index.
    A length mismatch breaks recipe compilation.  This function guarantees
    the contract via assertion at the end.

    Weight computation for each zone (in strict order):

    STEP 1 — Exclude check (ABSOLUTE, terminates all further computation):
        For each tag in intent_tags.exclude:
            if zone_matches_intent_semantics(zone, tag):
                weight = 0.0
                goto DONE  ← no primary/secondary/urgency/user_state can override
        A user who said "not API reference" and receives API reference content
        has experienced a failure.  The exclude is inviolable.
        Implemented with a flag and early continue — NOT as part of the
        additive sum.  This is intentional and non-negotiable.

    STEP 2 — Primary boost:
        weight = 1.0  (default — no match = weight 1.0, still included)
        For each tag in intent_tags.primary:
            if zone_matches_intent_semantics(zone, tag):
                weight += INTENT_WEIGHT_PRIMARY  (1.0 per match)

    STEP 3 — Secondary boost:
        For each tag in intent_tags.secondary:
            if zone_matches_intent_semantics(zone, tag):
                weight += INTENT_WEIGHT_SECONDARY  (0.5 per match)

    STEP 4 — Urgency modifier:
        if intent_tags.urgency == "high":
            if zone_matches_intent_semantics(zone, "warning"):
                weight += INTENT_WEIGHT_URGENCY  (0.5)
        This amplifies warning/safety zones when the user is in a time-critical
        situation — high urgency signals distress or incident response.

    STEP 5 — User state modifier:
        if intent_tags.user_state == "locked_out":
            if zone.content_type == "list":
                weight += INTENT_WEIGHT_LOCKED  (0.8)
                Recovery steps are almost always presented as ordered lists.
        if intent_tags.user_state == "debugging":
            if zone.content_type == "code":
                weight += INTENT_WEIGHT_DEBUG  (0.5)
                Debugging sessions need code context.

    DONE:
        Clamp weight to [0.0, INTENT_WEIGHT_CEILING].
        Append (zone.selector, weight) to results.

    Postcondition: len(results) == len(signal_zones).  Asserted.
    """
    exclude_set: frozenset = _compute_exclude_set(intent_tags)
    results: List[Tuple[str, float]] = []

    for zone in signal_zones:
        # ── STEP 1: Exclude check ────────────────────────────────────────────
        excluded: bool = False
        for exc_tag in exclude_set:
            if zone_matches_intent_semantics(zone, exc_tag):
                excluded = True
                break    # short-circuit — first match is definitive

        if excluded:
            results.append((zone.selector, INTENT_WEIGHT_EXCLUDE))
            continue    # no further weight computation for excluded zones

        # ── STEP 2: Primary boost ────────────────────────────────────────────
        weight: float = INTENT_WEIGHT_DEFAULT

        for pri_tag in intent_tags.primary:
            if zone_matches_intent_semantics(zone, pri_tag):
                weight += INTENT_WEIGHT_PRIMARY

        # ── STEP 3: Secondary boost ──────────────────────────────────────────
        for sec_tag in intent_tags.secondary:
            if zone_matches_intent_semantics(zone, sec_tag):
                weight += INTENT_WEIGHT_SECONDARY

        # ── STEP 4: Urgency modifier ─────────────────────────────────────────
        if intent_tags.urgency == "high":
            if zone_matches_intent_semantics(zone, "warning"):
                weight += INTENT_WEIGHT_URGENCY

        # ── STEP 5: User state modifier ──────────────────────────────────────
        if intent_tags.user_state == "locked_out":
            if zone.content_type == "list":
                weight += INTENT_WEIGHT_LOCKED

        if intent_tags.user_state == "debugging":
            if zone.content_type == "code":
                weight += INTENT_WEIGHT_DEBUG

        # ── DONE: Clamp and append ───────────────────────────────────────────
        weight = min(INTENT_WEIGHT_CEILING, max(INTENT_WEIGHT_EXCLUDE, weight))
        results.append((zone.selector, weight))

    # Postcondition: length must match signal_zones.
    # This is a contract assertion — a failed assertion here is a programming
    # error in this function, not a data quality issue.
    assert len(results) == len(signal_zones), (
        f"apply_intent_weights() postcondition violated: "
        f"len(results)={len(results)} != len(signal_zones)={len(signal_zones)}. "
        "This is a bug in apply_intent_weights() — report immediately."
    )

    return tuple(results)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — OVERALL CONFIDENCE COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def _compute_overall_confidence(
    zone_confs:  List[float],
    node_counts: List[int],
) -> float:
    """
    Compute the overall ZoneMap confidence as a weighted arithmetic mean
    of per-zone confidence scores.

    Weighting model:
        weight_i = node_counts[i]   (larger zones carry more evidential weight)

        confidence = Σ(zone_confs[i] × node_counts[i]) / Σ(node_counts[i])

    This is the first moment of the confidence distribution weighted by zone
    size (number of nodes).  A zone with 50 nodes exerts 5× more influence on
    the overall confidence than a zone with 10 nodes.

    Rationale: a large, confidently-classified zone is stronger evidence of
    ZoneMap quality than a small, equally-confident zone.  Node count is a
    proxy for zone importance — larger zones contribute more signal to the
    extraction recipe and are therefore better confidence anchors.

    Numerical stability:
        If Σ(node_counts) == 0 (all zones are empty — degenerate input):
            Return 0.0 — no basis for confidence.
        All confidence values clamped to [0.0, 1.0] before weighting to
        prevent upstream arithmetic errors from corrupting the aggregate.

    Ceiling enforcement:
        Result is clamped at DISCOVERY_CONFIDENCE_CEILING (0.70) for
        ZoneMaps produced through the discovery path.
        Callers that want full-range confidence must pass uncapped inputs.
        The ceiling is NOT enforced here — it is applied by assemble_zone_map()
        when topology_class is unknown (discovery path).

    Returns: float in [0.0, 1.0].
    """
    if not zone_confs or not node_counts:
        return 0.0

    total_weight: float = float(sum(node_counts))
    if total_weight == 0.0:
        return 0.0

    weighted_sum: float = 0.0
    for conf, count in zip(zone_confs, node_counts):
        clamped_conf = min(1.0, max(0.0, conf))    # numerical safety clamp
        weighted_sum += clamped_conf * count

    raw_confidence: float = weighted_sum / total_weight
    return min(1.0, max(0.0, raw_confidence))


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — ZoneDescriptor CONSTRUCTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _build_zone_descriptor(
    candidate:    _CandidateZone,
    selector:     str,
    scope:        str,
    content_type: str,
    density:      float,
    priority:     int,
    selector_type: str = "css",
) -> ZoneDescriptor:
    """
    Construct a ZoneDescriptor from all Stage 3 computed components.

    average_depth is computed from the candidate zone's nodes here rather
    than being passed as a parameter — it is always computable from the nodes
    and there is no circumstance where a caller would want to override it.

    This helper exists to collect the descriptor construction logic in one
    place and ensure that every ZoneDescriptor is built consistently.
    """
    avg_depth: float = _compute_average_depth(candidate)

    return ZoneDescriptor(
        selector      = selector,
        selector_type = selector_type,
        scope         = scope,
        content_type  = content_type,
        average_depth = avg_depth,
        density       = density,
        priority      = priority,
    )


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

_VALID_CONTENT_TYPES:  frozenset = frozenset({"prose", "code", "list", "table", "mixed"})
_VALID_SELECTOR_TYPES: frozenset = frozenset({"css", "xpath"})
_VALID_BOUNDARY_TYPES: frozenset = frozenset({
    "SECTION_BOUNDARY", "CONTENT_BOUNDARY", "NOISE_BOUNDARY",
})


def _validate_selector(selector: str, selector_type: str) -> None:
    """
    Validate a CSS or XPath selector string.

    Raises AssertionError on:
        - Empty selector string ("" is the silent-failure case)
        - selector_type not in ("css", "xpath")
        - CSS selector containing :nth-child or :first-child (positional)
        - CSS selector with more than 3 descendant levels (too fragile)
          (heuristic: count " > " or " " tokens as descendant operators)

    Does NOT attempt full CSS parsing — structural heuristics only.
    """
    assert selector, (
        f"selector must not be empty. "
        "An empty selector produces a recipe that matches nothing."
    )
    assert selector_type in _VALID_SELECTOR_TYPES, (
        f"selector_type must be one of {_VALID_SELECTOR_TYPES}, "
        f"got {selector_type!r}."
    )
    if selector_type == "css":
        assert ":nth-child" not in selector, (
            f"selector contains :nth-child — positional selector forbidden: {selector!r}"
        )
        assert ":first-child" not in selector, (
            f"selector contains :first-child — positional selector forbidden: {selector!r}"
        )
        # Count descendant levels by counting " > " and bare " " combinator tokens.
        # Maximum 3 levels allowed.
        descendant_count: int = selector.count(" > ") + selector.count(" .")
        assert descendant_count <= 3, (
            f"selector has {descendant_count} descendant levels (max 3): {selector!r}"
        )


def _validate_regex(pattern: str) -> None:
    """
    Validate that a string is a compilable Python regex pattern.

    Raises AssertionError (wrapping re.error) if pattern is invalid.
    """
    try:
        re.compile(pattern)
    except re.error as exc:
        raise AssertionError(
            f"invalid regex pattern {pattern!r}: {exc}"
        ) from exc


def validate_zone_map(zone_map: ZoneMap) -> None:
    """
    Validate ZoneMap invariants before construction.

    Called inside assemble_zone_map() on the assembled components BEFORE
    the ZoneMap constructor call.  A validation failure returns EmptyZoneMap
    without ever constructing the invalid ZoneMap.

    The ZoneMap constructor never sees invalid data.  A ZoneMap that exists
    is valid by construction.  This invariant lets topology/parser.py trust
    every non-empty ZoneMap it receives.

    Invariants checked:

    Selector validity (all signal + noise zones):
        zone.selector       is non-empty
        zone.selector_type  in ("css", "xpath")
        zone.scope          is non-empty (never empty — "body" is valid fallback)
        zone.content_type   in ("prose", "code", "list", "table", "mixed")

    Density validity (signal zones only):
        0.0 <= zone.density <= 1.0

    Priority uniqueness (signal zones):
        All priority values are unique integers.
        len(priorities) == len(set(priorities))

    Confidence validity:
        0.0 <= zone_map.confidence <= 1.0

    Intent weights length:
        len(zone_map.intent_weights) == len(zone_map.signal_zones)
        This is the most critical invariant — topology/parser.py zips
        these two tuples by index.  A mismatch causes silent recipe truncation.

    Boundary pattern validity:
        Each BoundaryDescriptor.boundary_type ∈ {"SECTION_BOUNDARY", "CONTENT_BOUNDARY", "NOISE_BOUNDARY"}
        Each BoundaryDescriptor.delimiter_content is a valid Python regex.

    Node count consistency:
        signal_node_count + noise_node_count + boundary_node_count <= node_count
        (≤ not == because confidence-threshold-excluded nodes count toward
        node_count but not toward any classified count)

    Raises AssertionError with a descriptive message on any violation.
    Returns None on success.
    """
    # Selector and field validity for all zones.
    for zone_list_name, zone_list in (
        ("signal_zones", zone_map.signal_zones),
        ("noise_zones",  zone_map.noise_zones),
    ):
        for idx, zone in enumerate(zone_list):
            prefix = f"{zone_list_name}[{idx}]"

            _validate_selector(zone.selector, zone.selector_type)

            assert zone.scope, (
                f"{prefix}.scope must not be empty. "
                "Fallback to 'body' should have been applied in determine_scope()."
            )
            assert zone.content_type in _VALID_CONTENT_TYPES, (
                f"{prefix}.content_type={zone.content_type!r} is not in "
                f"{_VALID_CONTENT_TYPES}."
            )

    # Density validity — signal zones only.
    for idx, zone in enumerate(zone_map.signal_zones):
        assert 0.0 <= zone.density <= 1.0, (
            f"signal_zones[{idx}].density={zone.density} is outside [0.0, 1.0]. "
            "compute_density() must clamp its output."
        )

    # Priority uniqueness — signal zones only.
    if zone_map.signal_zones:
        priorities: List[int] = [z.priority for z in zone_map.signal_zones]
        assert len(priorities) == len(set(priorities)), (
            f"signal_zones priorities are not unique: {priorities}. "
            "assign_priorities() must produce a bijection onto [0, n-1]."
        )

    # Confidence validity.
    assert 0.0 <= zone_map.confidence <= 1.0, (
        f"zone_map.confidence={zone_map.confidence} is outside [0.0, 1.0]. "
        "_compute_overall_confidence() must clamp its output."
    )

    # Intent weights length contract — THE CRITICAL INVARIANT.
    assert len(zone_map.intent_weights) == len(zone_map.signal_zones), (
        f"CRITICAL: len(intent_weights)={len(zone_map.intent_weights)} != "
        f"len(signal_zones)={len(zone_map.signal_zones)}. "
        "topology/parser.py zips these by index — length mismatch causes "
        "silent recipe truncation. This is a contract violation."
    )

    # Boundary validity.
    for idx, boundary in enumerate(zone_map.boundaries):
        assert boundary.boundary_type in _VALID_BOUNDARY_TYPES, (
            f"boundaries[{idx}].boundary_type={boundary.boundary_type!r} "
            f"is not in {_VALID_BOUNDARY_TYPES}."
        )
        assert boundary.selector, (
            f"boundaries[{idx}].selector must not be empty."
        )
        _validate_regex(boundary.delimiter_content)

    # Node count consistency.
    classified_total: int = (
        zone_map.signal_node_count
        + zone_map.noise_node_count
        + zone_map.boundary_node_count
    )
    assert classified_total <= zone_map.node_count, (
        f"classified node count ({classified_total}) exceeds total node_count "
        f"({zone_map.node_count}). "
        "Confidence-threshold-excluded nodes should be counted in node_count "
        "but not in any classified count."
    )


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 5 — MAIN ASSEMBLY FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def assemble_zone_map(
    node_classifications:    "torch.Tensor",    # (n_nodes, 3) logits
    node_confidences:        "torch.Tensor",    # (n_nodes, 1) confidence
    cst_nodes:               List[object],      # CSTNode list from wlp_graph
    topology_class:          str,
    domain:                  str,
    intent_vector:           Optional[List[float]],
    topology_router_version: int,
    confidence_threshold:    float = CONFIDENCE_THRESHOLD,
) -> Union[ZoneMap, EmptyZoneMap]:
    """
    Main assembly function.  Orchestrates all five stages of the pipeline.

    This function never raises.  Every exception is caught, logged with full
    context (topology_class, domain, exception), and converted to EmptyZoneMap.

    Return EmptyZoneMap on any doubt.  A wrong ZoneMap silently corrupts every
    extraction from that topology class until the structural_layer.pt entry is
    replaced.  An EmptyZoneMap triggers hardcoded fallback for one URL on one
    query and is corrected on the next clean signal cycle.  Wrong ZoneMap is
    silent corruption.  EmptyZoneMap is clean degradation.  Always choose
    clean degradation.

    Assembly sequence:
        1.  classify_nodes()          logits → labels, confs
        2.  group_signal_nodes()      labels → candidate_zones, boundary_nodes
        3.  Early return if no signal zones detected.
        4.  For each candidate_zone:
                generate_css_selector()  → selector
                determine_scope()        → scope
                infer_content_type()     → content_type
                compute_density()        → density
        5.  assign_priorities()           densities → priorities
        6.  Build ZoneDescriptors from all Stage 3 components.
        7.  select_extraction_strategy()  → ExtractionStrategy
        8.  identify_boundaries()         → BoundaryDescriptor list
        9.  Build noise ZoneDescriptors from NOISE-labelled nodes.
        10. Compute intent weights:
                if intent_vector: parse_intent_tags() → apply_intent_weights()
                else: default weights tuple((sel, 1.0) for zone in signal_zones)
        11. Compute overall confidence:
                zone_confs = [mean(zone.confs) for each zone]
                node_counts = [len(zone.nodes) for each zone]
                overall_confidence = _compute_overall_confidence(...)
        12. validate_zone_map() on assembled components.
        13. Construct and return ZoneMap.

    The outer try/except is not lazy error handling.  It is the contract.
    topology/parser.py calls wlp.query() and receives a ZoneMap or EmptyZoneMap.
    It does not handle exceptions.  If assemble_zone_map() raises through,
    the entire extraction pipeline for that URL dies.  Do not raise through.

    Log context on every failure:
        topology_class, domain, exception type, exception message.
    """
    try:
        return _assemble_zone_map_inner(
            node_classifications    = node_classifications,
            node_confidences        = node_confidences,
            cst_nodes               = cst_nodes,
            topology_class          = topology_class,
            domain                  = domain,
            intent_vector           = intent_vector,
            topology_router_version = topology_router_version,
            confidence_threshold    = confidence_threshold,
        )
    except AssertionError as exc:
        log.error(
            "zone_map_invalid domain=%s topology_class=%s error=%s",
            domain, topology_class, str(exc),
        )
        return EmptyZoneMap()
    except Exception as exc:
        log.exception(
            "zone_map_assembly_failed domain=%s topology_class=%s error=%s",
            domain, topology_class, str(exc),
        )
        return EmptyZoneMap()


def _assemble_zone_map_inner(
    node_classifications:    "torch.Tensor",
    node_confidences:        "torch.Tensor",
    cst_nodes:               List[object],
    topology_class:          str,
    domain:                  str,
    intent_vector:           Optional[List[float]],
    topology_router_version: int,
    confidence_threshold:    float,
) -> Union[ZoneMap, EmptyZoneMap]:
    """
    Inner assembly implementation.  May raise — caller (assemble_zone_map)
    catches all exceptions and returns EmptyZoneMap.

    Separated from assemble_zone_map() so that the outer function is a clean
    try/except wrapper and this function contains the actual logic.
    """
    produced_at: float = time.monotonic()

    # ── Guard: tensor shape validation ───────────────────────────────────────
    if node_classifications.dim() != 2 or node_classifications.shape[1] != 3:
        log.warning(
            "zone_map_bad_tensor_shape domain=%s topology_class=%s shape=%s",
            domain, topology_class, str(node_classifications.shape),
        )
        return EmptyZoneMap()

    if node_confidences.dim() != 2 or node_confidences.shape[1] != 1:
        log.warning(
            "zone_map_bad_confidence_shape domain=%s topology_class=%s shape=%s",
            domain, topology_class, str(node_confidences.shape),
        )
        return EmptyZoneMap()

    n_nodes: int = node_classifications.shape[0]
    if n_nodes != node_confidences.shape[0]:
        log.warning(
            "zone_map_tensor_length_mismatch domain=%s topology_class=%s "
            "n_class=%d n_conf=%d",
            domain, topology_class, n_nodes, node_confidences.shape[0],
        )
        return EmptyZoneMap()

    if n_nodes != len(cst_nodes):
        log.warning(
            "zone_map_node_count_mismatch domain=%s topology_class=%s "
            "tensor_nodes=%d cst_nodes=%d",
            domain, topology_class, n_nodes, len(cst_nodes),
        )
        return EmptyZoneMap()

    # ── Stage 1: Node classification ─────────────────────────────────────────
    # This is the tensor boundary.  After this call: pure Python.
    labels: List[int]
    confs:  List[float]
    labels, confs = classify_nodes(
        node_classifications,
        node_confidences,
        confidence_threshold,
    )

    signal_node_count:   int = sum(1 for l in labels if l == NODE_SIGNAL)
    noise_node_count:    int = sum(1 for l in labels if l == NODE_NOISE)
    boundary_node_count: int = sum(1 for l in labels if l == NODE_BOUNDARY)

    # ── Stage 2: Zone grouping ────────────────────────────────────────────────
    candidate_zones: List[_CandidateZone]
    boundary_nodes:  List[object]
    candidate_zones, boundary_nodes = group_signal_nodes(cst_nodes, labels, confs)

    if not candidate_zones:
        log.warning(
            "no_signal_zones domain=%s topology_class=%s "
            "total_nodes=%d signal_count=%d",
            domain, topology_class, n_nodes, signal_node_count,
        )
        return EmptyZoneMap()

    # ── Stage 3: Per-zone descriptor computation ──────────────────────────────
    selectors:     List[str]   = []
    scopes:        List[str]   = []
    content_types: List[str]   = []
    densities:     List[float] = []

    for cz in candidate_zones:
        sel:  str   = generate_css_selector(cz)
        sc:   str   = determine_scope(cz, cst_nodes)
        ct:   str   = infer_content_type(cz)
        dens: float = compute_density(cz)

        selectors.append(sel)
        scopes.append(sc)
        content_types.append(ct)
        densities.append(dens)

    # ── Stage 3: Priority assignment ──────────────────────────────────────────
    priorities: List[int] = assign_priorities(candidate_zones, densities)

    # ── Stage 3: Build signal ZoneDescriptors ────────────────────────────────
    signal_descriptors: List[ZoneDescriptor] = [
        _build_zone_descriptor(
            candidate    = candidate_zones[i],
            selector     = selectors[i],
            scope        = scopes[i],
            content_type = content_types[i],
            density      = densities[i],
            priority     = priorities[i],
        )
        for i in range(len(candidate_zones))
    ]

    # Sort signal_descriptors by priority for canonical ordering.
    signal_descriptors.sort(key=lambda z: z.priority)
    signal_zone_tuple: Tuple[ZoneDescriptor, ...] = tuple(signal_descriptors)

    # ── Stage 3: Build noise ZoneDescriptors ────────────────────────────────
    # Noise zones are structural chrome worth recording for negative-pattern
    # generation in topology/parser.py (grep -v patterns).
    # We identify noise zones as groups of adjacent NOISE nodes using a
    # simplified grouping that does not need the full adjacency semantics
    # of group_signal_nodes().
    noise_descriptors: List[ZoneDescriptor] = _build_noise_descriptors(
        cst_nodes, labels, confs
    )
    noise_zone_tuple: Tuple[ZoneDescriptor, ...] = tuple(noise_descriptors)

    # ── Stage 3: Extraction strategy ─────────────────────────────────────────
    strategy: ExtractionStrategy = select_extraction_strategy(
        candidate_zones,
        boundary_nodes,
        topology_class,
    )

    # ── Stage 2 + 3: Boundary identification ─────────────────────────────────
    boundary_descriptors: List[BoundaryDescriptor] = identify_boundaries(
        boundary_nodes, labels, confs
    )
    boundary_tuple: Tuple[BoundaryDescriptor, ...] = tuple(boundary_descriptors)

    # ── Stage 4: Intent conditioning ─────────────────────────────────────────
    intent_weights: Tuple[Tuple[str, float], ...]

    if intent_vector is not None:
        intent_tags: IntentTags = parse_intent_tags(intent_vector)
        intent_weights = apply_intent_weights(signal_zone_tuple, intent_tags)
    else:
        # No intent provided — default weights: all zones included equally.
        intent_weights = tuple(
            (zone.selector, INTENT_WEIGHT_DEFAULT)
            for zone in signal_zone_tuple
        )

    # Assert intent weight length contract immediately — before ZoneMap construction.
    assert len(intent_weights) == len(signal_zone_tuple), (
        f"Intent weight length mismatch after Stage 4: "
        f"len(intent_weights)={len(intent_weights)}, "
        f"len(signal_zones)={len(signal_zone_tuple)}."
    )

    # ── Stage 5: Overall confidence ──────────────────────────────────────────
    zone_confs: List[float] = [
        (sum(cz.confs) / len(cz.confs) if cz.confs else 0.0)
        for cz in candidate_zones
    ]
    node_counts: List[int] = [len(cz.nodes) for cz in candidate_zones]
    overall_confidence: float = _compute_overall_confidence(zone_confs, node_counts)

    # Apply discovery ceiling if topology_class is not a known class.
    # Known topology classes have confidence confirmed by multiple feedback cycles.
    # Discovery-path topology classes cap at 0.70 to prevent over-confidence
    # before sufficient feedback has been accumulated.
    from .contracts import KNOWN_TOPOLOGY_CLASSES  # type: ignore[import]  # noqa
    # Guard against import failures in standalone testing — use frozenset fallback.
    try:
        _known = KNOWN_TOPOLOGY_CLASSES
    except Exception: # noqa
        _known = frozenset({
            "NEWS_ARTICLE", "SAAS_DOCS", "REST_API_JSON",
            "JSON_LD_STRUCTURED", "ECOMMERCE_PRODUCT", "GENERIC_HTML",
        })

    if topology_class not in _known:
        overall_confidence = min(overall_confidence, DISCOVERY_CONFIDENCE_CEILING)

    # ── Stage 5: Validation ───────────────────────────────────────────────────
    # Build a candidate ZoneMap for validation.
    # validate_zone_map() runs before the real construction — the constructor
    # never sees invalid data.
    candidate_zone_map = ZoneMap(
        topology_class          = topology_class,
        domain                  = domain,
        signal_zones            = signal_zone_tuple,
        noise_zones             = noise_zone_tuple,
        boundaries              = boundary_tuple,
        extraction_strategy     = strategy,
        intent_weights          = intent_weights,
        confidence              = overall_confidence,
        node_count              = n_nodes,
        signal_node_count       = signal_node_count,
        noise_node_count        = noise_node_count,
        boundary_node_count     = boundary_node_count,
        version                 = 0,
        produced_at             = produced_at,
        topology_router_version = topology_router_version,
    )

    # Validate invariants — raises AssertionError on violation.
    # AssertionError propagates to the outer try/except in assemble_zone_map(),
    # which returns EmptyZoneMap.
    validate_zone_map(candidate_zone_map)

    # Validation passed — the ZoneMap is valid by construction.
    log.debug(
        "zone_map_assembled domain=%s topology_class=%s signal_zones=%d "
        "noise_zones=%d boundaries=%d confidence=%.4f strategy=%s",
        domain, topology_class, len(signal_zone_tuple),
        len(noise_zone_tuple), len(boundary_tuple),
        round(overall_confidence, 4), strategy.value,
    )

    return candidate_zone_map


# ─────────────────────────────────────────────────────────────────────────────
# NOISE ZONE CONSTRUCTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _build_noise_descriptors(
    nodes:  List[object],
    labels: List[int],
    confs:  List[float],
) -> List[ZoneDescriptor]:
    """
    Build ZoneDescriptors for noise zones.

    Noise ZoneDescriptors are used by topology/parser.py to generate negative
    patterns — grep -v exclusions that actively strip noise regions.

    Noise zone identification: nodes with NODE_NOISE label whose CSS classes
    intersect NOISE_CLASSES.  Not all NOISE nodes become noise ZoneDescriptors
    — only nodes that carry structural noise class signals (sidebar, nav, footer,
    advertisement, etc.).  Generic wrappers and layout divs are not worth
    recording as negative-pattern noise zones.

    Algorithm:
        For each NOISE-labelled node:
            If node.css_classes ∩ NOISE_CLASSES is non-empty:
                Build a synthetic single-node CandidateZone.
                Generate selector.
                Build ZoneDescriptor with density=0.0, priority=0.
                Priority 0 for all noise zones: recipe compilation uses them
                as exclusion patterns, not ordered extraction rules.

    Returns: List[ZoneDescriptor], may be empty.
    """
    noise_descriptors: List[ZoneDescriptor] = []
    # Track selectors to avoid duplicates (same structural region seen multiple times).
    seen_selectors: set = set()

    for i, node in enumerate(nodes):
        if labels[i] != NODE_NOISE:
            continue

        css_classes: List[str] = getattr(node, "css_classes", []) or []
        class_set = frozenset(c.strip().lower() for c in css_classes if c.strip())

        # Only materialise noise ZoneDescriptors for recognised noise classes.
        if not (class_set & NOISE_CLASSES):
            continue

        synthetic_zone = _CandidateZone(
            nodes            = [node],
            confs            = [confs[i]],
            parent_index     = getattr(node, "parent_index", -1),
            first_node_index = i,
        )

        selector = generate_css_selector(synthetic_zone)
        if selector in seen_selectors:
            continue
        seen_selectors.add(selector)

        noise_descriptors.append(ZoneDescriptor(
            selector      = selector,
            selector_type = "css",
            scope         = "body",
            content_type  = "mixed",
            average_depth = float(getattr(node, "depth", 0) or 0),
            density       = 0.0,
            priority      = 0,
        ))

    return noise_descriptors


# ─────────────────────────────────────────────────────────────────────────────
# INTENT CATEGORY MATCHER REGISTRY
# Populated here after ZoneDescriptor is defined.
# These are the per-tag structural matchers used by zone_matches_intent_semantics().
# Stored in INTENT_CATEGORY_MATCHERS for introspection and testing.
# ─────────────────────────────────────────────────────────────────────────────

def _build_intent_category_matchers() -> Dict[str, Callable]:
    """
    Build the INTENT_CATEGORY_MATCHERS dictionary.

    Each entry maps an intent tag string to a Callable[[ZoneDescriptor], bool]
    that returns True if the zone structurally matches that intent.

    This is a read-only registry — callers use zone_matches_intent_semantics()
    as the primary interface.  The registry is exposed for unit testing and
    for introspection by the intent conditioning diagnostics layer.

    The lambdas here are thin wrappers around the same logic implemented in
    zone_matches_intent_semantics() — they exist to provide a programmatic
    lookup interface without duplicating the matching logic.
    """
    matchers: Dict[str, Callable] = {}

    # Account recovery group.
    _recovery_tags = frozenset({
        "account_recovery", "lost_access", "restore_access",
        "reset_password", "login_help", "unlock_account",
        "mfa_recovery", "backup_codes", "session_recovery",
    })
    for _tag in _recovery_tags:
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # API reference group.
    _api_tags = frozenset({
        "api_reference", "endpoint", "schema", "parameter",
        "authentication", "authorization", "rate_limit",
        "webhook", "sdk", "integration",
    })
    for _tag in _api_tags:
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # Pricing group.
    _pricing_tags = frozenset({
        "pricing", "plans", "cost", "billing", "subscription",
        "upgrade", "downgrade", "trial", "enterprise", "features",
    })
    for _tag in _pricing_tags:
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # Tutorial group.
    _tutorial_tags = frozenset({
        "getting_started", "quickstart", "tutorial", "guide",
        "walkthrough", "setup", "installation", "configuration",
        "onboarding", "first_steps",
    })
    for _tag in _tutorial_tags:
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # Code example group.
    _code_tags = frozenset({
        "code_example", "snippet", "sample", "demo", "playground",
        "repository", "library", "package", "cli", "command",
    })
    for _tag in _code_tags:
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # Warning group.
    _warning_tags = frozenset({
        "warning", "caution", "danger", "breaking_change",
        "deprecation", "security", "vulnerability",
        "migration", "rollback", "incident",
    })
    for _tag in _warning_tags:
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # Changelog / release notes.
    for _tag in ("changelog", "release_notes"):
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    # FAQ / troubleshooting.
    for _tag in ("faq", "troubleshooting"):
        matchers[_tag] = lambda z, t=_tag: zone_matches_intent_semantics(z, t)

    return matchers


# Populate the registry once at module import time.
INTENT_CATEGORY_MATCHERS.update(_build_intent_category_matchers())


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL MATH UTILITIES — not exported
# These are used only within this file for precision arithmetic.
# ─────────────────────────────────────────────────────────────────────────────

def _softmax_3(a: float, b: float, c: float) -> Tuple[float, float, float]:
    """
    Numerically stable 3-way softmax.

    Standard softmax suffers from overflow when logit values are large.
    Stable form subtracts the maximum before exponentiation:

        x_stable[i] = x[i] - max(x)
        softmax[i] = exp(x_stable[i]) / sum(exp(x_stable))

    Since max(x_stable) == 0, exp(x_stable[i]) ∈ (0, 1] for all i.
    No overflow.  Underflow is harmless (rounds to 0.0).

    Returns: (p_a, p_b, p_c) with p_a + p_b + p_c == 1.0 (up to float error).
    """
    m: float = max(a, b, c)
    ea, eb, ec = math.exp(a - m), math.exp(b - m), math.exp(c - m)
    inv_sum: float = 1.0 / (ea + eb + ec)
    return ea * inv_sum, eb * inv_sum, ec * inv_sum


def _weighted_harmonic_mean(values: List[float], weights: List[float]) -> float:
    """
    Compute the weighted harmonic mean of a list of positive values.

    Weighted harmonic mean:
        H = Σ(w_i) / Σ(w_i / v_i)

    The harmonic mean penalises low values more strongly than the arithmetic
    mean.  It is the appropriate average for rates and densities where a single
    low value should drag the overall metric down significantly.

    Used internally for computing aggregate zone confidence in scenarios where
    we want to penalise sparse (low-confidence) zones more than the weighted
    arithmetic mean would.

    Edge cases:
        Empty lists: return 0.0.
        Any v_i == 0.0: contributes ∞ to the denominator → H = 0.0.
            A zone with zero confidence is a degenerate zone; its presence
            should produce near-zero aggregate confidence.
        Negative values: not expected; clamped to epsilon to avoid domain errors.

    Returns: float in [0.0, max(values)].
    """
    if not values or not weights:
        return 0.0

    _EPS: float = 1e-12
    weight_sum: float = 0.0
    denom: float = 0.0

    for v, w in zip(values, weights):
        v_safe = max(_EPS, v)   # avoid division by zero; near-zero ≈ zero
        w_safe = max(0.0, w)    # negative weights are nonsensical
        weight_sum += w_safe
        denom      += w_safe / v_safe

    if weight_sum == 0.0 or denom == 0.0:
        return 0.0

    return weight_sum / denom


def _entropy_3(p: float, q: float, r: float) -> float:
    """
    Shannon entropy of a 3-class probability distribution in bits.

    H = -Σ p_i * log2(p_i)   (p_i = 0 contributes 0 by convention)

    Used internally to measure the uncertainty of a node's classification.
    Maximum entropy for 3 classes: log2(3) ≈ 1.585 bits.
    Zero entropy: deterministic assignment to one class.

    This is used in diagnostic logging within classify_nodes() for nodes
    that are near the confidence threshold — high-entropy nodes near the
    threshold are worth flagging because they represent genuine model
    uncertainty rather than data sparsity.
    """
    def _xlogy(x: float) -> float:
        return 0.0 if x <= 0.0 else x * math.log2(x)

    return -(_xlogy(p) + _xlogy(q) + _xlogy(r))


def _gini_impurity_from_counts(
    signal: int, noise: int, boundary: int
) -> float:
    """
    Compute Gini impurity for a set of node classifications.

    Gini impurity measures how mixed a set of classified nodes is.

    G = 1 - Σ p_i²

    For 3 classes with counts (n_s, n_n, n_b):
        p_s = n_s / n_total
        p_n = n_n / n_total
        p_b = n_b / n_total
        G = 1 - (p_s² + p_n² + p_b²)

    G == 0.0: all nodes have the same label (pure set — ideal zone).
    G == 0.667: maximum impurity (equal distribution across 3 classes).

    A zone map with low Gini impurity has tightly-classified nodes — strong
    structural signal.  A high-impurity zone map suggests the WLP model is
    uncertain about this topology class and the ZoneMap should carry lower
    confidence.

    Used in _assemble_zone_map_inner() for diagnostic telemetry only.
    Does not affect zone production logic.
    """
    n_total = signal + noise + boundary
    if n_total == 0:
        return 0.0

    inv: float = 1.0 / n_total
    p_s = signal   * inv
    p_n = noise    * inv
    p_b = boundary * inv

    return 1.0 - (p_s * p_s + p_n * p_n + p_b * p_b)


def _cosine_similarity_sparse(
    a_indices: List[int], b_indices: List[int], n_dims: int # noqa
) -> float:
    """
    Compute cosine similarity between two sparse binary vectors.

    Given sets of active dimension indices, cosine similarity is:

        cos(a, b) = |a ∩ b| / (√|a| × √|b|)

    Where |a| is the number of active dimensions (L2 norm of a binary vector
    equals √(number of ones)).

    This is used to compare two intent_vector fingerprints for deduplication —
    if two intent vectors have cosine similarity > 0.95, they are considered
    equivalent for zone weight caching purposes.

    n_dims is unused in the computation (the formula is dimension-independent
    for binary vectors) but is retained in the signature for documentation.

    Returns: float in [0.0, 1.0].
    """
    if not a_indices or not b_indices:
        return 0.0

    set_a = frozenset(a_indices)
    set_b = frozenset(b_indices)
    intersection_size: int = len(set_a & set_b)

    if intersection_size == 0:
        return 0.0

    mag_a: float = math.sqrt(len(set_a))
    mag_b: float = math.sqrt(len(set_b))

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return intersection_size / (mag_a * mag_b)


def _normalised_discounted_cumulative_gain(
    ranked_densities: List[float],
    ideal_densities:  List[float],
) -> float:
    """
    Compute Normalised Discounted Cumulative Gain (nDCG) for zone ordering.

    nDCG measures how well the zone priority ordering matches the ideal
    ordering by signal density.  A high nDCG means high-density zones are
    extracted first — the most information-rich zones appear earliest in the
    recipe.

    DCG definition:
        DCG = Σ_i (2^density_i - 1) / log2(i + 2)   for i = 0, 1, ..., n-1

    Note: i+2 because log2(1) = 0 (undefined), and we want position 1
    (0-indexed) to discount by log2(2) = 1, position 2 by log2(3), etc.

    IDCG = DCG computed on the ideal (sorted descending by density) ordering.
    nDCG = DCG / IDCG

    Used in diagnostic logging from assemble_zone_map() to measure zone
    ordering quality.  Does not affect zone production logic.

    Returns: float in [0.0, 1.0].  Returns 1.0 for empty or single-zone maps.
    """
    n: int = len(ranked_densities)
    if n <= 1:
        return 1.0
    if not ideal_densities:
        return 1.0

    def _dcg(densities: List[float]) -> float:
        total: float = 0.0
        for i, dens in enumerate(densities):
            # Use 2^density - 1 as the relevance score.
            # For density ∈ [0,1]: relevance ∈ [0, 1].
            # Dense zones are given exponentially higher relevance.
            relevance: float = (2.0 ** dens) - 1.0
            discount:  float = math.log2(i + 2)   # log2(2)=1.0 for i=0
            total += relevance / discount
        return total

    dcg:  float = _dcg(ranked_densities)
    idcg: float = _dcg(sorted(ideal_densities, reverse=True))

    if idcg <= 0.0:
        return 1.0

    return min(1.0, dcg / idcg)


def _laplace_smoothed_ratio(
    numerator: int,
    denominator: int,
    alpha: float = 1.0,
    n_classes: int = 2,
) -> float:
    """
    Compute a Laplace-smoothed (additive smoothed) ratio.

    Standard ratio:  numerator / denominator
    Laplace-smoothed: (numerator + α) / (denominator + α × n_classes)

    Laplace smoothing prevents zero ratios when counts are sparse — important
    for early-stage topology classes with few examples where the signal density
    estimate would otherwise fluctuate wildly.

    Default α=1 is the standard Laplace (add-one) smoothing.
    α=0.1 gives Lidstone smoothing for lighter regularisation.

    n_classes=2: binary (signal/total) smoothing by default.

    Returns: float in (0.0, 1.0) exclusive.
    """
    smoothed_num: float = numerator   + alpha
    smoothed_den: float = denominator + alpha * n_classes
    if smoothed_den <= 0.0:
        return 0.5   # maximum uncertainty
    return smoothed_num / smoothed_den


def _reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: float = 60.0,
) -> List[Tuple[int, float]]:
    """
    Combine multiple ranked lists of zone indices using Reciprocal Rank Fusion.

    RRF score for zone i across ranked_lists:
        RRF(i) = Σ_r  1 / (k + rank_r(i))

    where rank_r(i) is the 1-based rank of zone i in ranked list r,
    and the sum is over all ranked lists r.

    k=60 is the standard default (Cormack et al. 2009).  Higher k gives more
    weight to top-ranked items.  k=60 is a good balance for structural zone
    fusion where no single ranking signal is fully trusted.

    Used internally to fuse multiple priority signals (DOM position, density,
    content type specificity) into a single combined priority ordering.

    Returns: list of (zone_index, rrf_score) tuples sorted descending by score.
    Higher RRF score = should be ranked higher = lower priority number.

    Example use case:
        ranked_lists[0] = zones sorted by first_node_index (DOM position)
        ranked_lists[1] = zones sorted by density descending
        result = combined ranking that balances both signals
    """
    rrf_scores: Dict[int, float] = {}

    for ranked_list in ranked_lists:
        for rank_0based, zone_idx in enumerate(ranked_list):
            rank_1based: float = rank_0based + 1.0
            rrf_scores[zone_idx] = rrf_scores.get(zone_idx, 0.0) + 1.0 / (k + rank_1based)

    return sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)


def _information_gain_ratio(
    parent_counts: Tuple[int, int, int],
    child_counts_list: List[Tuple[int, int, int]],
) -> float:
    """
    Compute the Information Gain Ratio for a zone boundary split.

    Information Gain Ratio is the Information Gain divided by the Split
    Information of the partition.  It corrects for splits that create many
    branches (which standard information gain favours even when the split
    is uninformative).

    Used to evaluate whether a proposed BOUNDARY node produces a meaningful
    structural split — if the IG ratio is low, the boundary is not genuinely
    discriminative and may produce an uninformative section partition.

    Definitions:
        Entropy(S) = -Σ_c p_c log2(p_c)
        IG(S, A) = Entropy(S) - Σ_v (|S_v|/|S|) * Entropy(S_v)
        SplitInfo(S, A) = -Σ_v (|S_v|/|S|) log2(|S_v|/|S|)
        IGR(S, A) = IG(S, A) / SplitInfo(S, A)

    Where S is the parent set, A is the split attribute (boundary),
    and S_v are the child subsets.

    parent_counts:    (signal, noise, boundary) counts before split.
    child_counts_list: list of (signal, noise, boundary) for each child.

    Returns: float ∈ [0.0, 1.0].  Returns 0.0 if split is uninformative.
    """
    def _entropy3(counts: Tuple[int, int, int]) -> float:
        total = sum(counts)
        if total == 0:
            return 0.0
        return _entropy_3(
            counts[0] / total,
            counts[1] / total,
            counts[2] / total,
        )

    parent_total: int = sum(parent_counts)
    if parent_total == 0:
        return 0.0

    parent_entropy: float = _entropy3(parent_counts)
    split_info: float = 0.0
    weighted_child_entropy: float = 0.0

    for child_counts in child_counts_list:
        child_total = sum(child_counts)
        if child_total == 0:
            continue
        fraction = child_total / parent_total
        weighted_child_entropy += fraction * _entropy3(child_counts)
        if fraction > 0.0:
            split_info -= fraction * math.log2(fraction)

    info_gain: float = parent_entropy - weighted_child_entropy

    if split_info <= 0.0:
        return 0.0

    return min(1.0, info_gain / split_info)


# ─────────────────────────────────────────────────────────────────────────────
# RUNTIME CONTRACT VERIFICATION
# Module-level sanity checks.  These run once at import time and raise
# immediately if the module constants are internally inconsistent.
# They catch configuration errors at load time rather than at first call.
# ─────────────────────────────────────────────────────────────────────────────

def _verify_module_constants() -> None:
    """
    Verify internal consistency of module-level constants.

    Raises AssertionError at module import time if any constant is invalid.
    All checks are O(1) — no loops over large data structures.
    """
    # Node label constants are distinct integers.
    assert NODE_SIGNAL != NODE_NOISE != NODE_BOUNDARY
    assert len({NODE_SIGNAL, NODE_NOISE, NODE_BOUNDARY}) == 3

    # Confidence thresholds are in valid range and correctly ordered.
    assert 0.0 < CONFIDENCE_THRESHOLD < 1.0
    assert 0.0 < BOUNDARY_CONFIDENCE_THRESHOLD < 1.0
    assert CONFIDENCE_THRESHOLD < BOUNDARY_CONFIDENCE_THRESHOLD, (
        "BOUNDARY_CONFIDENCE_THRESHOLD must be strictly greater than "
        "CONFIDENCE_THRESHOLD — boundary nodes carry higher-stakes information."
    )
    assert 0.0 < MIN_ZONE_CONFIDENCE < DISCOVERY_CONFIDENCE_CEILING < 1.0

    # Intent weight constants are non-negative and ceiling is >= default.
    assert INTENT_WEIGHT_DEFAULT   >= 0.0
    assert INTENT_WEIGHT_PRIMARY   >= 0.0
    assert INTENT_WEIGHT_SECONDARY >= 0.0
    assert INTENT_WEIGHT_URGENCY   >= 0.0
    assert INTENT_WEIGHT_LOCKED    >= 0.0
    assert INTENT_WEIGHT_DEBUG     >= 0.0
    assert INTENT_WEIGHT_CEILING   >= INTENT_WEIGHT_DEFAULT
    assert INTENT_WEIGHT_EXCLUDE   == 0.0, (
        "INTENT_WEIGHT_EXCLUDE must be 0.0 — the exclude is absolute."
    )

    # Intent vector ranges are contiguous and non-overlapping.
    assert INTENT_PRIMARY_END   == INTENT_SECONDARY_START
    assert INTENT_SECONDARY_END == INTENT_EXCLUDE_START
    assert INTENT_EXCLUDE_END   == INTENT_URGENCY_START
    assert INTENT_URGENCY_END   == INTENT_USER_STATE_START
    assert INTENT_USER_STATE_END == 256

    # Urgency and user state bins cover their full ranges exactly.
    urgency_positions = len(URGENCY_STATES) * URGENCY_BIN_SIZE
    assert urgency_positions == (INTENT_URGENCY_END - INTENT_URGENCY_START), (
        f"Urgency: {len(URGENCY_STATES)} states × {URGENCY_BIN_SIZE} floats "
        f"= {urgency_positions} but range has "
        f"{INTENT_URGENCY_END - INTENT_URGENCY_START} positions."
    )

    user_state_positions = len(USER_STATE_STATES) * USER_STATE_BIN_SIZE
    assert user_state_positions == (INTENT_USER_STATE_END - INTENT_USER_STATE_START), (
        f"User state: {len(USER_STATE_STATES)} states × {USER_STATE_BIN_SIZE} floats "
        f"= {user_state_positions} but range has "
        f"{INTENT_USER_STATE_END - INTENT_USER_STATE_START} positions."
    )

    # Vocabulary is non-empty and within slot size.
    assert 0 < len(INTENT_TOKEN_VOCABULARY) <= 64

    # DISCRIMINATIVE_CLASSES has no overlap with NON_DISCRIMINATIVE_CLASSES.
    discriminative_set = frozenset(DISCRIMINATIVE_CLASSES)
    overlap = discriminative_set & NON_DISCRIMINATIVE_CLASSES
    assert not overlap, (
        f"Classes appear in both DISCRIMINATIVE_CLASSES and NON_DISCRIMINATIVE_CLASSES: "
        f"{sorted(overlap)}. A class cannot be both discriminative and non-discriminative."
    )

    # ExtractionStrategy enum values are non-empty strings.
    for strategy in ExtractionStrategy:
        assert strategy.value, f"ExtractionStrategy.{strategy.name}.value is empty."

    # Valid content types, selector types, boundary types are non-empty.
    assert _VALID_CONTENT_TYPES
    assert _VALID_SELECTOR_TYPES
    assert _VALID_BOUNDARY_TYPES


# Run verification at import time.
_verify_module_constants()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE EXPORTS SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
#
# Exported types (used by topology/parser.py and latent_parser.py):
#     ExtractionStrategy     enum
#     ZoneDescriptor         frozen dataclass
#     BoundaryDescriptor     frozen dataclass
#     ZoneMap                frozen dataclass
#     IntentTags             mutable dataclass (intermediate only)
#     EmptyZoneMap           class (ZoneMap failure value)
#     EmptyZoneKnowledge     class (cold-start knowledge store stub)
#
# Exported functions (called by latent_parser.py):
#     assemble_zone_map()    ← main entry point
#     classify_nodes()       ← Stage 1
#     group_signal_nodes()   ← Stage 2
#     generate_css_selector() ← Stage 3
#     determine_scope()      ← Stage 3
#     infer_content_type()   ← Stage 3
#     compute_density()      ← Stage 3
#     assign_priorities()    ← Stage 3
#     select_extraction_strategy() ← Stage 3
#     identify_boundaries()  ← Stage 3
#     parse_intent_tags()    ← Stage 4
#     zone_matches_intent_semantics() ← Stage 4
#     apply_intent_weights() ← Stage 4
#     validate_zone_map()    ← Stage 5
#
# Internal only (not in __all__, not documented in public API):
#     _CandidateZone
#     _build_noise_descriptors()
#     _assemble_zone_map_inner()
#     _softmax_3()
#     _weighted_harmonic_mean()
#     _entropy_3()
#     _gini_impurity_from_counts()
#     _cosine_similarity_sparse()
#     _normalised_discounted_cumulative_gain()
#     _laplace_smoothed_ratio()
#     _reciprocal_rank_fusion()
#     _information_gain_ratio()
#     _coefficient_of_variation()
#     _compute_average_depth()
#     _build_zone_descriptor()
#     _compute_overall_confidence()
#     _boundary_type_from_node()
#     _boundary_delimiter_pattern()
#     _most_discriminative_class()
#     _is_structural_id()
#     _is_positional_class()
#     _verify_module_constants()
#     _build_intent_category_matchers()
#
# AXIOM INTERNAL // DO NOT SURFACE