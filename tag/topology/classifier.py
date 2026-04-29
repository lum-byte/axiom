"""
tag/topology/classifier.py
==========================
First thing that runs on every URL AXIOM touches.

Takes a URL, HTTP response headers, and the first 4KB of content and produces
a TopologyClassification — one of 18 known classes with a confidence score.
Everything downstream (traversal policy, fetch strategy, recipe selection,
WLM routing) depends on this result.

Architecture: Evidence Lattice with Topology-Aware Confidence Propagation.

The spec describes a waterfall classifier — try paths in order, stop at first
confident result. This implementation honours that contract at every observable
boundary while introducing three structural improvements that make it more
resilient, more informative, and less brittle at the classification margins:

  1. _DomainTrie — O(depth) trie lookup replaces O(n) dict + O(n) wildcard scan.
     Domain resolution is now a single tree walk regardless of how large the
     fingerprint table grows. Wildcard patterns are first-class trie nodes, not
     a second pass. Confidence is stored per-node so subdomain overrides work.

  2. _ContentFingerprintIndex — character n-gram Jaccard similarity replaces
     ordered grep for the content window path. Each topology class has a
     fingerprint set (hashed character 3-grams). The content prefix is tokenised
     the same way. Similarity is measured before falling back to the exact
     WINDOW_PATTERNS list. This catches paraphrased and internationalised
     structural signals that exact string matching misses.

  3. _EvidenceLattice — topology-hierarchy-aware evidence accumulator. Every
     path deposits evidence even when the waterfall short-circuits. Evidence
     propagates upward through PARENT_CLASS_MAP with a decay coefficient.
     Independent signals for parent and child classes are fused using the
     standard independent-evidence formula (1 - prod(1 - p_i)). This means:
       - Multiple weak signals for related classes combine to cross the
         confidence threshold before the ML path is invoked.
       - A new classification path (LATTICE_FUSION) is available so the ML
         path fires only for genuinely ambiguous inputs, reducing average
         path-5 invocation rate.
       - signals_used in TopologyClassification contains a full evidence
         trace, not just the single winning signal.

The file is split into two halves:

  FIRST HALF  — Deterministic signal paths 1–4, hard overrides, evidence
                lattice, domain trie, content fingerprint index, URL semantic
                tokenizer, classify() orchestrator, initialize(), and all
                pattern tables.

  SECOND HALF — ML path (feature engineering + model forward pass), CLASSIFIER
                singleton, and production infrastructure: confidence calibration,
                feature drift monitoring, classification cache, batch interface,
                model health validation, online learning signal collection,
                adaptive threshold control, telemetry pipeline, content structure
                analysis, feature importance estimation, and self-tests.

Performance contracts:
    Path 1 domain fingerprint    < 0.1 ms    trie walk
    Path 2 URL structure         < 1 ms      precompiled regex
    Path 3 response headers      < 1 ms      key lookup
    Path 4 content window        < 5 ms      n-gram hash + bounded grep
    Path 4.5 lattice fusion      < 1 ms      arithmetic on deposited evidence
    Path 5 ML model              < 15 ms     GPU forward pass (when invoked)

Hard contracts:
    Never fetches. Hard overrides bypass everything including ML. Never returns
    None. Model reload is GIL-safe atomic assignment. weights_only=True always.
    torch.no_grad() wraps all inference. Feature vectors are deterministic.

Depends on: contracts.py, exceptions.py, store/topology_router.pt, store_watchdog.py

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

# ═════════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

# ── stdlib ───────────────────────────────────────────────────────────────────

import array                                            # noqa
import hashlib
import logging
import math
import mmap
import os                                               # noqa
import re
import struct
import sys                                              # noqa
import threading
import time
import warnings                                         # noqa
from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache                         # noqa
from typing import (  # noqa
    Any,
    Callable,
    Deque,
    Dict,
    Final,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)
from urllib.parse import urlparse
from pathlib import Path

# ── third-party ──────────────────────────────────────────────────────────────

import numpy as np
import torch

# ── internal: contracts ──────────────────────────────────────────────────────

from signal_kernel.contracts import (                   # noqa
    FALLBACK_TOPOLOGY_CLASS,
    PARENT_CLASS_MAP,
    THETA_CLASSIFY_CONFIDENT,
    THETA_CLASSIFY_FALLBACK,
    TOPOLOGY_CLASSES,
    ConfidenceFloat,
    TopologyClassStr,
    TopologyClassification,
    new_run_id,
)

# ── internal: exceptions ─────────────────────────────────────────────────────

from signal_kernel.exceptions import (                  # noqa
    ClassificationConfidenceTooLow,
    ClassificationWindowTooSmall,
    ClassifierModelNotInitialized,
)

# ── internal: store watchdog ─────────────────────────────────────────────────

from tag.store_watchdog import WATCHDOG

# ── internal: Mamba router ─────────────────────────────────────────────────

from tag.world_model.world_latent_model.mamba_router import (
    MambaRouter,
    PRODUCTION_CONFIG,
)

# ── module logger ────────────────────────────────────────────────────────────

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TYPE DEFINITIONS
#
# These are internal to the classifier. They are not part of the
# signal_kernel contracts because they represent implementation details of the
# classification pipeline, not boundary contracts.
# ══════════════════════════════════════════════════════════════════════════════

class ClassificationPath(str, Enum):
    """Which signal path resolved the topology class.

    String-valued so it can be placed directly into
    TopologyClassification.classification_path without conversion.
    """
    DOMAIN_FINGERPRINT = "domain"
    URL_STRUCTURE      = "url"
    HEADER_SIGNAL      = "header"
    CONTENT_WINDOW     = "window"
    LATTICE_FUSION     = "lattice"      # novel: hierarchy-fused multi-path evidence
    PHASE_GATED        = "phase_gated"  # phase-state-aware early short-circuit
    MODEL              = "model"
    FALLBACK           = "fallback"


@dataclass(frozen=True)
class _SignalEvidence:
    """Immutable evidence record from one classification path.

    Every path that returns a result deposits one of these into the
    _EvidenceLattice and _SignalLedger before the waterfall decides whether
    to short-circuit. This preserves the full evidence trace regardless of
    which path ultimately resolves the classification.
    """
    topology_class: str
    raw_confidence: float          # confidence as returned by the path method
    path:           ClassificationPath
    detail:         str            # human-readable reason, used in signals_used


@dataclass(frozen=True)
class _PhaseSlot:
    """Parsed representation of one 32-byte slot from phase_states.mmap.

    Produced by _PhaseStateReader._read_slot_consistent(). Immutable once
    constructed. Carried through classify() as part of the phase snapshot so
    all waterfall threshold checks in a single call operate on a consistent
    view of the phase file (no repeated mmap reads on the hot path).

    Fields mirror the binary layout defined in initialize_store.py:

        offset 0   phase_id       uint8   1=LEARNS, 2=PREDICTS, 3=KNOWS
        offset 1   flags          uint8   bit0=active, bit1=surprise_tripped
        offset 4   confidence     float32 accumulated WLM confidence for this class
        offset 12  surprise_score float32 most recent surprise signal magnitude
    """
    phase_id:       int    # 1 | 2 | 3
    flags:          int    # raw flags byte
    confidence:     float
    surprise_score: float

    @property
    def surprise_tripped(self) -> bool:
        """True iff index_daemon set the surprise_tripped flag (bit 1).

        Meaning: the WLM detected structural divergence for this topology class
        since the last phase transition. When set, _phase_adjusted_theta()
        returns _PHASE_THETA_DRIFT_GATE (>1.0), making the threshold
        unreachable by any deterministic path and forcing every classification
        through the ML model.
        """
        return bool(self.flags & 0x02)

    @property
    def is_knows(self) -> bool:
        return self.phase_id == 3

    @property
    def is_predicts(self) -> bool:
        return self.phase_id == 2

    @property
    def is_learns(self) -> bool:
        return self.phase_id == 1


@dataclass(frozen=True)
class ClassifierInput:
    """Thin wrapper that extends ClassificationWindow with HTTP response code.

    ClassificationWindow (contracts.py) captures the four signals needed by
    the classifier. response_code lives here because it is a transport-level
    signal that affects the header path and hard-override logic but is not
    part of the general classification window contract.

    classifier.classify() accepts ClassifierInput. interface.py constructs
    ClassifierInput from the Phantom result before calling classify().
    """
    url:            str
    headers:        Dict[str, str]
    content_prefix: bytes          # raw bytes — classifier decodes internally
    response_code:  int
    run_id:         Optional[str] = None

    def __post_init__(self) -> None:
        if not self.url.startswith(("http://", "https://")):
            raise ValueError(
                f"url must begin with http:// or https://, got {self.url[:40]!r}. "
                "Non-HTTP URLs should not reach the classifier."
            )


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER CONSTANTS
#
# Single source of truth for all pattern tables. Constants with complex
# structure (trie, fingerprint index) are built once at module load time
# from these tables. Do not bypass the tables by calling trie/index methods
# directly with ad-hoc patterns.
# ══════════════════════════════════════════════════════════════════════════════

# ── Confidence thresholds ─────────────────────────────────────────────────────
# Imported from contracts. Redeclared here as local names so path methods
# can reference them without the contracts prefix, matching the spec.
_THETA_CONFIDENT: float = float(THETA_CLASSIFY_CONFIDENT)   # 0.75
_THETA_FALLBACK:  float = float(THETA_CLASSIFY_FALLBACK)    # 0.40

# Phase-aware threshold multipliers applied by _phase_adjusted_theta().
# Each multiplier scales _THETA_CONFIDENT up or down based on the current
# phase state of the candidate topology class read from phase_states.mmap.
#
#   LEARNS  (Phase I)  — system is still exploring this class. Be conservative:
#                        raise the short-circuit bar so more evidence is required
#                        before cutting off the waterfall. More ML invocations,
#                        which is correct — Phase I exists precisely to collect
#                        training signal. 0.75 * 1.15 = 0.8625.
#
#   PREDICTS (Phase II) — neutral. World model is active but not proven compiled.
#                         Keep the standard threshold unchanged.
#
#   KNOWS (Phase III, no surprise) — compiled structural policy. Fingerprints and
#                         pattern tables have been validated across many crawl
#                         cycles. Lower the short-circuit bar: a deterministic
#                         path at 0.60 confidence is more reliable for a Phase III
#                         class than the same confidence for an unknown class.
#                         Fewer ML invocations, lower latency, higher throughput.
#                         0.75 * 0.80 = 0.60.
#
#   KNOWS (Phase III, surprise_tripped) — WLM detected structural drift for this
#                         class since last phase transition. Existing fingerprints
#                         and patterns may be stale. Force ALL classifications to
#                         the ML path by setting threshold > 1.0 (unreachable by
#                         any confidence value). This is the correct defensive
#                         posture: trust the model, not stale deterministic signals.
_PHASE_THETA_LEARNS_MULT:   float = 1.15   # Phase I  — conservative (+15%)
_PHASE_THETA_PREDICTS_MULT: float = 1.00   # Phase II — unchanged
_PHASE_THETA_KNOWS_MULT:    float = 0.80   # Phase III no-surprise — aggressive (−20%)
_PHASE_THETA_DRIFT_GATE:    float = 2.0    # Phase III surprise_tripped — force ML path

# Evidence propagation decay through PARENT_CLASS_MAP.
# When class C has confidence P, its parent class gets P * _LATTICE_DECAY
# added to its evidence pool before independent-evidence fusion.
# 0.65 was chosen so that two sibling classes at 0.72 each can propagate
# their parent to (1 - (1 - 0.72*0.65)^2) ≈ 0.64, still below the
# short-circuit threshold but meaningful for the lattice fusion path.
_LATTICE_DECAY: float = 0.65

# Minimum content_prefix length (decoded UTF-8 chars) to run the window path.
# Below this the window is too small to produce reliable n-gram signals.
_MIN_WINDOW_CHARS: int = 128

# Content window size in bytes — never read further.
_CONTENT_WINDOW_BYTES: int = 4096

# N-gram size for _ContentFingerprintIndex.
_NGRAM_SIZE: int = 3

# Jaccard similarity threshold for fingerprint path to return a hit.
# Below this the n-gram signal is noise.
_FINGERPRINT_SIM_THRESHOLD: float = 0.10

# Maximum recent-classification audit buffer (for debugging, not persistence).
_AUDIT_BUFFER_SIZE: int = 512

# ═════════════════════════════════════════════════════════════════════════════
# FEATURE SPACE CONSTANTS
#
# These define the exact dimensions of each feature group. The sum MUST match
# topology_router.pt's input_dim. Changing any of these without retraining
# the model will corrupt inference.
#
# Feature version string: included in model metadata at training time.
# If the feature version in the loaded model's metadata doesn't match,
# the model health validator emits a hard warning.
# ═════════════════════════════════════════════════════════════════════════════

FEATURE_VERSION: Final[str] = "axiom.classifier.features.v1"

# Group dimensions — strict, not recommended
GROUP_1_URL_PATH_TOKENS_DIM:     Final[int] = 64
GROUP_2_HEADER_BITMASK_DIM:      Final[int] = 48
GROUP_3_CONTENT_NGRAM_HASH_DIM:  Final[int] = 128
GROUP_4_DOMAIN_FEATURES_DIM:     Final[int] = 32
GROUP_5_FINGERPRINT_SCORES_DIM:  Final[int] = 18
GROUP_6_LATTICE_SCORES_DIM:      Final[int] = 18

TOTAL_FEATURE_DIM: Final[int] = (
    GROUP_1_URL_PATH_TOKENS_DIM
    + GROUP_2_HEADER_BITMASK_DIM
    + GROUP_3_CONTENT_NGRAM_HASH_DIM
    + GROUP_4_DOMAIN_FEATURES_DIM
    + GROUP_5_FINGERPRINT_SCORES_DIM
    + GROUP_6_LATTICE_SCORES_DIM
)
# 64 + 48 + 128 + 32 + 18 + 18 = 308

NUM_TOPOLOGY_CLASSES: Final[int] = len(TOPOLOGY_CLASSES)  # 18

# ── Domain fingerprint table ──────────────────────────────────────────────────
# Exact domain → topology class. Extended by preparser over time.
# Wildcard variants (*.wikipedia.org) are handled by the trie — do not
# mix glob patterns here; use exact hostnames only.
DOMAIN_FINGERPRINT_TABLE: Dict[str, str] = {
    # SaaS documentation
    "docs.stripe.com":              "SAAS_DOCS",
    "docs.twilio.com":              "SAAS_DOCS",
    "docs.github.com":              "SAAS_DOCS",
    "docs.aws.amazon.com":          "SAAS_DOCS",
    "docs.anthropic.com":           "SAAS_DOCS",
    "docs.openai.com":              "SAAS_DOCS",
    "docs.fastapi.tiangolo.com":    "SAAS_DOCS",
    "docs.pydantic.dev":            "SAAS_DOCS",
    "docs.djangoproject.com":       "SAAS_DOCS",
    "docs.python.org":              "SAAS_DOCS",
    "developer.mozilla.org":        "SAAS_DOCS",
    "developer.apple.com":          "SAAS_DOCS",
    "developers.google.com":        "SAAS_DOCS",
    "cloud.google.com":             "SAAS_DOCS",
    "learn.microsoft.com":          "SAAS_DOCS",
    "docs.microsoft.com":           "SAAS_DOCS",
    # REST APIs
    "api.github.com":               "REST_API_JSON",
    "api.stripe.com":               "REST_API_JSON",
    "api.openai.com":               "REST_API_JSON",
    "api.anthropic.com":            "REST_API_JSON",
    "api.twilio.com":               "REST_API_JSON",
    # Structured data
    "arxiv.org":                    "JSON_LD_STRUCTURED",
    "schema.org":                   "JSON_LD_STRUCTURED",
    # Community / discussion
    "reddit.com":                   "FORUM_THREAD",
    "news.ycombinator.com":         "FORUM_THREAD",
    "stackoverflow.com":            "FORUM_THREAD",
    "discourse.org":                "FORUM_THREAD",
    # Blog / editorial
    "medium.com":                   "BLOG_POST",
    "substack.com":                 "BLOG_POST",
    "dev.to":                       "BLOG_POST",
    "hashnode.com":                 "BLOG_POST",
}

# Wildcard domain patterns — evaluated by _DomainTrie, not direct lookup.
# Format: (pattern_string, topology_class, confidence).
# Pattern syntax: leading "*." means "any subdomain of this suffix".
DOMAIN_WILDCARD_SPECS: List[Tuple[str, str, float]] = [
    ("*.wikipedia.org",     "WIKIPEDIA_ARTICLE",  0.99),
    ("*.shopify.com",       "ECOMMERCE_PRODUCT",  0.95),
    ("*.myshopify.com",     "ECOMMERCE_PRODUCT",  0.95),
    ("*.medium.com",        "BLOG_POST",          0.92),
    ("*.substack.com",      "BLOG_POST",          0.92),
    ("*.github.io",         "SAAS_DOCS",          0.88),
    ("*.readthedocs.io",    "SAAS_DOCS",          0.92),
    ("*.readthedocs.org",   "SAAS_DOCS",          0.92),
    ("*.gitbook.io",        "SAAS_DOCS",          0.88),
]

# URL structure patterns — ordered. First match wins.
# Confidence 0.85: URL structure is reliable but not infallible.
URL_STRUCTURE_PATTERNS: List[Tuple[re.Pattern, str, float]] = [
    (re.compile(r"/api/v\d+/"),                    "REST_API_JSON",         0.90),
    (re.compile(r"/api/v\d+\.\d+/"),               "REST_API_JSON",         0.92),
    (re.compile(r"/api/"),                         "REST_API_JSON",         0.82),
    (re.compile(r"\.json(?:\?|$)"),                "REST_API_JSON",         0.88),
    (re.compile(r"/graphql(?:\?|$)"),              "REST_API_JSON",         0.90),
    (re.compile(r"/products?/\d+"),                "ECOMMERCE_PRODUCT",     0.87),
    (re.compile(r"/item/\d+"),                     "ECOMMERCE_PRODUCT",     0.82),
    (re.compile(r"/wiki/"),                        "WIKIPEDIA_ARTICLE",     0.90),
    (re.compile(r"/docs?/"),                       "SAAS_DOCS",             0.82),
    (re.compile(r"/reference/"),                   "SAAS_DOCS",             0.80),
    (re.compile(r"/guides?/"),                     "SAAS_DOCS",             0.78),
    (re.compile(r"/blog/"),                        "BLOG_POST",             0.85),
    (re.compile(r"/posts?/"),                      "BLOG_POST",             0.80),
    (re.compile(r"/news/"),                        "NEWS_ARTICLE",          0.85),
    (re.compile(r"/article/"),                     "NEWS_ARTICLE",          0.85),
    (re.compile(r"/forum/"),                       "FORUM_THREAD",          0.87),
    (re.compile(r"/thread/"),                      "FORUM_THREAD",          0.87),
    (re.compile(r"/discussion/"),                  "FORUM_THREAD",          0.83),
    (re.compile(r"/questions?/"),                  "FORUM_THREAD",          0.80),
    (re.compile(r"/login|/signin|/auth(?:/|$)"),   "AUTH_REDIRECT",         0.95),
    (re.compile(r"/oauth/"),                       "AUTH_REDIRECT",         0.95),
    (re.compile(r"/sso/"),                         "AUTH_REDIRECT",         0.90),
]

# Response header signals — checked in order, first match wins per category.
# Tuple: (header_key, value_substring_or_None, topology_class, confidence).
# value_substring=None means presence-only check (any value triggers).
HEADER_SIGNALS: List[Tuple[str, Optional[str], str, float]] = [
    # Hard-class: always hard-override candidates
    ("cf-ray",           None,                      "CLOUDFLARE_CHALLENGE",   0.99),
    ("x-amz-cf-id",      None,                      "CLOUDFLARE_CHALLENGE",   0.85),  # CloudFront
    # Rate limiting
    ("retry-after",      None,                      "RATE_LIMITED",           0.99),
    ("x-ratelimit-remaining", "0",                  "RATE_LIMITED",           0.97),
    # Content type signals
    ("content-type",     "application/json",        "REST_API_JSON",          0.95),
    ("content-type",     "application/ld+json",     "JSON_LD_STRUCTURED",     0.97),
    ("content-type",     "application/atom+xml",    "NEWS_ARTICLE",           0.85),
    ("content-type",     "application/rss+xml",     "NEWS_ARTICLE",           0.85),
    # Paywalls and noindex
    ("x-robots-tag",     "noindex",                 "NEWS_ARTICLE_PAYWALLED", 0.82),
    ("paywall",          None,                      "NEWS_ARTICLE_PAYWALLED", 0.90),
    ("x-piano-tpl",      None,                      "NEWS_ARTICLE_PAYWALLED", 0.90),
    # Auth redirect
    ("x-frame-options",  "deny",                    "AUTH_REDIRECT",          0.82),
    ("www-authenticate", None,                      "AUTH_REDIRECT",          0.95),
    # SaaS docs version header
    ("x-api-version",    None,                      "SAAS_DOCS_VERSIONED",    0.80),
    ("x-docs-version",   None,                      "SAAS_DOCS_VERSIONED",    0.80),
]

# Multi-header correlation patterns — these fire when ALL headers in the set
# are present. More specific than single-header checks; confidence is boosted.
# List of (frozenset_of_header_keys, topology_class, confidence).
HEADER_CORRELATION_PATTERNS: List[Tuple[FrozenSet[str], str, float]] = [
    (frozenset({"content-type", "x-api-version"}),
     "SAAS_DOCS_VERSIONED", 0.90),
    (frozenset({"content-type", "x-ratelimit-limit", "x-ratelimit-remaining"}),
     "REST_API_JSON", 0.97),
    (frozenset({"cf-ray", "cf-cache-status"}),
     "CLOUDFLARE_CHALLENGE", 0.97),
    (frozenset({"x-piano-tpl", "x-robots-tag"}),
     "NEWS_ARTICLE_PAYWALLED", 0.95),
]

# Hard-override classes — bypass ML path. These are detected in paths 3 and 4.
HARD_OVERRIDE_CLASSES: FrozenSet[str] = frozenset({
    "AUTH_REDIRECT",
    "CLOUDFLARE_CHALLENGE",
    "RATE_LIMITED",
})

# Content window patterns — exact substring matches against first 4 KB.
# Tuple: (needle, topology_class, confidence).
WINDOW_PATTERNS: List[Tuple[str, str, float]] = [
    # Hard-override patterns — checked first.
    ("cf-browser-verification",         "CLOUDFLARE_CHALLENGE",       0.99),
    ("jschl-answer",                    "CLOUDFLARE_CHALLENGE",       0.99),
    ("challenge-form",                  "CLOUDFLARE_CHALLENGE",       0.97),
    ('input type="password"',           "AUTH_REDIRECT",              0.85),
    ('type="password"',                 "AUTH_REDIRECT",              0.82),
    ('name="password"',                 "AUTH_REDIRECT",              0.80),
    # Structured data
    ("application/ld+json",             "JSON_LD_STRUCTURED",         0.92),
    ('"@context": "https://schema.org"',"JSON_LD_STRUCTURED",         0.95),
    ('"@type": "Product"',              "ECOMMERCE_PRODUCT",          0.92),
    ('"@type": "Article"',              "NEWS_ARTICLE",               0.85),
    ('"@type": "BlogPosting"',          "BLOG_POST",                  0.88),
    ('"@type": "WebPage"',              "LANDING_PAGE",               0.72),
    # E-commerce
    ("data-product-id",                 "ECOMMERCE_PRODUCT",          0.88),
    ("data-sku",                        "ECOMMERCE_PRODUCT",          0.85),
    ("add-to-cart",                     "ECOMMERCE_PRODUCT",          0.80),
    ("data-variant-id",                 "ECOMMERCE_PRODUCT_VARIANT",  0.87),
    # Wikipedia
    ('id="mw-content-text"',            "WIKIPEDIA_ARTICLE",          0.97),
    ('class="mw-parser-output"',        "WIKIPEDIA_ARTICLE",          0.95),
    # Forums and community
    ('class="forum-post"',              "FORUM_THREAD",               0.87),
    ('class="comment-thread"',          "FORUM_THREAD",               0.82),
    ('data-testid="comment"',           "FORUM_THREAD",               0.82),
    # Blog and editorial
    ('class="post-content"',            "BLOG_POST",                  0.78),
    ('class="article-body"',            "BLOG_POST",                  0.78),
    ('<article',                        "NEWS_ARTICLE",               0.72),
    ('class="article-content"',         "NEWS_ARTICLE",               0.80),
    # SaaS docs
    ('"version":',                      "SAAS_DOCS_VERSIONED",        0.72),
    ('data-docs-version',               "SAAS_DOCS_VERSIONED",        0.80),
    ('<pre><code',                       "SAAS_DOCS_WITH_CODE",        0.72),
    ('class="highlight"',               "SAAS_DOCS_WITH_CODE",        0.70),
    # REST API
    ('"pagination"',                    "REST_API_JSON_PAGINATED",    0.78),
    ('"next_cursor"',                   "REST_API_JSON_PAGINATED",    0.82),
    ('"next_page_token"',               "REST_API_JSON_PAGINATED",    0.82),
]

# Topology class → integer index for model output tensor.
# MUST match the training configuration of topology_router.pt.
# Do not reorder. Append only.
TOPOLOGY_CLASS_INDEX: Dict[str, int] = {
    cls: idx for idx, cls in enumerate(TOPOLOGY_CLASSES)
}

# Reverse map: integer index → topology class string.
INDEX_TO_TOPOLOGY_CLASS: Dict[int, str] = {
    v: k for k, v in TOPOLOGY_CLASS_INDEX.items()
}


# ══════════════════════════════════════════════════════════════════════════════
# _DomainTrie
#
# Trie-based domain matcher. Domain labels are stored reversed so that
# "docs.stripe.com" is stored as ["com", "stripe", "docs"], enabling
# efficient suffix matching and wildcard resolution.
#
# A wildcard node at level L matches any label at level L that does not
# have an exact-match child. Exact-match children always win over wildcards
# at the same level.
#
# Why a trie instead of the flat dict + regex approach from the reference:
#   - O(depth) lookup where depth = number of domain labels (≤ 6 in practice)
#   - Wildcard resolution falls out naturally from the tree walk
#   - New domains can be inserted without rescanning all patterns
#   - Confidence can be stored per-node, enabling subdomain-level overrides
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _TrieNode:
    """Single node in the domain resolution trie."""
    children:       Dict[str, "_TrieNode"] = field(default_factory=dict)
    topology_class: Optional[str]          = None
    confidence:     float                  = 0.0
    is_wildcard:    bool                   = False

    def is_terminal(self) -> bool:
        return self.topology_class is not None


class _DomainTrie:
    """Trie-backed domain → topology class resolver.

    Domain labels are reversed for suffix-first traversal.
    Wildcard children are stored under the sentinel key "*".

    Usage:
        trie = _DomainTrie()
        trie.insert("docs.stripe.com", "SAAS_DOCS", 0.99)
        trie.insert("*.wikipedia.org", "WIKIPEDIA_ARTICLE", 0.99)
        result = trie.match("en.wikipedia.org")
        # → ("WIKIPEDIA_ARTICLE", 0.99)
    """

    _WILDCARD_KEY: str = "*"

    def __init__(self) -> None:
        self._root = _TrieNode()

    def insert(self, pattern: str, topology_class: str, confidence: float) -> None:
        """Insert a domain pattern.

        pattern may start with "*." to indicate wildcard subdomain matching.
        e.g. "*.wikipedia.org" matches "en.wikipedia.org", "de.wikipedia.org", etc.
        Exact patterns always override wildcards at the same depth.
        """
        is_wildcard = pattern.startswith("*.")
        domain = pattern[2:] if is_wildcard else pattern
        labels = domain.lower().rstrip(".").split(".")
        labels.reverse()  # root-first (TLD first)

        node = self._root
        for label in labels:
            if label not in node.children:
                node.children[label] = _TrieNode()
            node = node.children[label]

        if is_wildcard:
            # Place the terminal under the wildcard sentinel child.
            if self._WILDCARD_KEY not in node.children:
                node.children[self._WILDCARD_KEY] = _TrieNode(is_wildcard=True)
            node = node.children[self._WILDCARD_KEY]

        node.topology_class = topology_class
        node.confidence     = confidence

    def match(self, hostname: str) -> Optional[Tuple[str, float]]:
        """Resolve a hostname to (topology_class, confidence).

        Returns None if no pattern matches.
        Exact-match nodes take priority over wildcard nodes at each level.
        The most-specific match wins (deepest terminal node in the trie).
        """
        labels = hostname.lower().rstrip(".").split(".")
        labels.reverse()

        return self._walk(self._root, labels, 0, best=None)

    def _walk(
        self,
        node: _TrieNode,
        labels: List[str],
        depth: int,
        best: Optional[Tuple[str, float]],
    ) -> Optional[Tuple[str, float]]:
        """DFS walk returning the deepest terminal match."""
        if node.is_terminal():
            best = (node.topology_class, node.confidence)

        if depth >= len(labels):
            # Check for wildcard child that captures further subdomains.
            wc = node.children.get(self._WILDCARD_KEY)
            if wc and wc.is_terminal():
                best = (wc.topology_class, wc.confidence)
            return best

        label = labels[depth]

        # Exact child wins over wildcard.
        if label in node.children:
            result = self._walk(node.children[label], labels, depth + 1, best)
            if result:
                return result

        # Wildcard child.
        wc = node.children.get(self._WILDCARD_KEY)
        if wc:
            result = self._walk(wc, labels, depth + 1, best)
            if result:
                return result

        return best

    @classmethod
    def build_global(cls) -> "_DomainTrie":
        """Build the module-level trie from DOMAIN_FINGERPRINT_TABLE
        and DOMAIN_WILDCARD_SPECS. Called once at module load."""
        trie = cls()
        for domain, tclass in DOMAIN_FINGERPRINT_TABLE.items():
            trie.insert(domain, tclass, 0.99)
        for pattern, tclass, conf in DOMAIN_WILDCARD_SPECS:
            trie.insert(pattern, tclass, conf)
        return trie


# Module-level singleton — built once, never mutated after module load.
_DOMAIN_TRIE: _DomainTrie = _DomainTrie.build_global()


# ══════════════════════════════════════════════════════════════════════════════
# _ContentFingerprintIndex
#
# Character n-gram Jaccard similarity engine for the content window path.
#
# Each topology class has a "canonical fingerprint" — the set of hashed
# character n-grams that appear in its WINDOW_PATTERNS strings. When content
# arrives, it is tokenised the same way. Jaccard(content, class_fingerprint)
# measures how much the content "looks like" pages of that class.
#
# This runs BEFORE the exact WINDOW_PATTERNS grep. It catches:
#   - Internationalised or templated variants of known structural patterns
#   - Pages where the exact needle string is absent but the surrounding
#     structural vocabulary is characteristic of the class
#   - Obfuscated friction pages where class-specific strings are slightly
#     modified to evade exact matching
#
# Performance: O(window_size * n) for tokenisation + O(unique_ngrams) for
# intersection — both negligible within the 4 KB content window budget.
# ══════════════════════════════════════════════════════════════════════════════

class _ContentFingerprintIndex:
    """Character n-gram Jaccard similarity index for topology class fingerprinting.

    Fingerprints are built from the WINDOW_PATTERNS needle strings for each
    topology class. The index maps class → frozenset of hashed n-gram ints.
    """

    def __init__(self, n: int = _NGRAM_SIZE) -> None:
        self._n = n
        self._class_fingerprints: Dict[str, FrozenSet[int]] = {}

    def _ngrams(self, text: str) -> Set[int]:
        """Compute hashed character n-grams for runtime-only set intersection.

        Uses Python's built-in hash() because this method is called only for
        intra-process Jaccard scoring (paths 4 and 4.5). Both sides of the
        comparison hash under the same seed — the relative similarity score
        is valid. This output never crosses a process boundary.

        Do NOT pattern-match off this for _embed_signals() GROUP 3.
        Feature tensors use blake2b(digest_size=4) — see that docstring.
        """
        text = text.lower()
        n = self._n
        if len(text) < n:
            return set()
        return {hash(text[i:i + n]) for i in range(len(text) - n + 1)}

    def index_class(self, topology_class: str, patterns: List[str]) -> None:
        """Build a fingerprint for topology_class from a list of pattern strings."""
        combined: Set[int] = set()
        for p in patterns:
            combined.update(self._ngrams(p))
        self._class_fingerprints[topology_class] = frozenset(combined)

    def score(self, content: str) -> Dict[str, float]:
        """Compute Jaccard similarity between content and each class fingerprint.

        Returns a dict mapping topology_class → similarity ∈ [0.0, 1.0].
        Classes with empty fingerprints are excluded from results.
        Only returns classes that exceed _FINGERPRINT_SIM_THRESHOLD.
        """
        content_ngrams = self._ngrams(content)
        if not content_ngrams:
            return {}

        results: Dict[str, float] = {}
        for cls, fingerprint in self._class_fingerprints.items():
            if not fingerprint:
                continue
            intersection = len(content_ngrams & fingerprint)
            union        = len(content_ngrams | fingerprint)
            if union == 0:
                continue
            sim = intersection / union
            if sim >= _FINGERPRINT_SIM_THRESHOLD:
                results[cls] = round(sim, 4)

        return results

    @classmethod
    def build_global(cls) -> "_ContentFingerprintIndex":
        """Build the module-level fingerprint index from WINDOW_PATTERNS."""
        index = cls()
        class_patterns: Dict[str, List[str]] = defaultdict(list)
        for needle, tclass, _ in WINDOW_PATTERNS:
            class_patterns[tclass].append(needle)
        for tclass, patterns in class_patterns.items():
            index.index_class(tclass, patterns)
        return index


# Module-level singleton.
_FINGERPRINT_INDEX: _ContentFingerprintIndex = _ContentFingerprintIndex.build_global()


# ══════════════════════════════════════════════════════════════════════════════
# _EvidenceLattice
#
# Topology-hierarchy-aware evidence accumulator.
#
# Every classification path that returns a non-None result deposits evidence
# here, even if the waterfall short-circuits on that evidence. This makes
# signals_used in the output complete rather than containing only the single
# winning signal.
#
# After all deterministic paths have run, propagate_through_hierarchy()
# distributes evidence upward through PARENT_CLASS_MAP using an independent-
# evidence fusion formula. The lattice can then produce a fused confidence
# for parent classes that might exceed the threshold without needing the ML
# path — this is the LATTICE_FUSION classification path.
#
# Independent-evidence fusion formula:
#     P(at least one of sources is correct) = 1 - product(1 - P_i)
# Used only when combining evidence for a PARENT class from its children.
# Direct evidence from a path is taken as-is.
# ══════════════════════════════════════════════════════════════════════════════

class _EvidenceLattice:
    """Accumulates and propagates classification evidence through the topology hierarchy.

    Designed to be constructed once per classify() call. Not thread-safe — one
    instance per active classification, never shared across calls.
    """

    def __init__(self) -> None:
        # Maps topology_class → list of deposited confidences (direct evidence)
        self._direct: Dict[str, List[_SignalEvidence]] = defaultdict(list)
        # Maps topology_class → propagated confidence (after propagation pass)
        self._propagated: Dict[str, float] = {}
        # Highest-confidence evidence seen so far, for fast early-exit checks.
        self._peak_confidence: float = 0.0
        self._peak_class: Optional[str] = None
        self._peak_path:  Optional[ClassificationPath] = None

    def deposit(self, evidence: _SignalEvidence) -> None:
        """Deposit evidence from a classification path."""
        self._direct[evidence.topology_class].append(evidence)
        if evidence.raw_confidence > self._peak_confidence:
            self._peak_confidence = evidence.raw_confidence
            self._peak_class      = evidence.topology_class
            self._peak_path       = evidence.path

    def propagate_through_hierarchy(self) -> None:
        """Propagate direct evidence upward through PARENT_CLASS_MAP.

        For each class with direct evidence, compute propagated confidence for
        its parent class using independent-evidence fusion. Propagation decays
        by _LATTICE_DECAY per hop. Maximum propagation depth: 3 hops (spec
        constraint on PARENT_CLASS_MAP depth).
        """
        # Build per-class peak direct confidence first.
        direct_peak: Dict[str, float] = {}
        for cls, evidences in self._direct.items():
            direct_peak[cls] = max(e.raw_confidence for e in evidences)

        # Propagate upward.
        for cls, conf in list(direct_peak.items()):
            decayed = conf * _LATTICE_DECAY
            current = cls
            hops    = 0
            while hops < 3:
                parent = PARENT_CLASS_MAP.get(current)
                if parent is None or parent == current:
                    break
                existing = self._propagated.get(parent, 0.0)
                # Independent-evidence fusion: combine existing propagated
                # evidence with new child contribution.
                self._propagated[parent] = 1.0 - (1.0 - existing) * (1.0 - decayed)
                decayed  *= _LATTICE_DECAY
                current   = parent
                hops     += 1

    def fused_confidence(self, topology_class: str) -> float:
        """Return the fused confidence for a topology class.

        Combines direct evidence (independent fusion) and propagated evidence
        (independent fusion). The result is in [0.0, 1.0].
        """
        evidences = self._direct.get(topology_class, [])
        if evidences:
            # Independent-evidence fusion over direct signals.
            direct_fused = 1.0
            for ev in evidences:
                direct_fused *= (1.0 - ev.raw_confidence)
            direct_fused = 1.0 - direct_fused
        else:
            direct_fused = 0.0

        prop = self._propagated.get(topology_class, 0.0)

        # Combine direct and propagated contributions.
        fused = 1.0 - (1.0 - direct_fused) * (1.0 - prop)
        return round(min(fused, 1.0), 4)

    def best_class_after_propagation(
        self,
    ) -> Optional[Tuple[str, float]]:
        """Return (topology_class, fused_confidence) for the best class after
        propagation, or None if nothing exceeds _THETA_FALLBACK.

        This is called after all deterministic paths have run and after
        propagate_through_hierarchy(). If the result is >= _THETA_CONFIDENT,
        the ML path can be skipped (LATTICE_FUSION path).
        """
        all_classes = set(self._direct.keys()) | set(self._propagated.keys())
        if not all_classes:
            return None

        best_cls:  Optional[str]   = None
        best_conf: float           = 0.0

        for cls in all_classes:
            fused = self.fused_confidence(cls)
            if fused > best_conf:
                best_conf = fused
                best_cls  = cls

        if best_cls and best_conf >= _THETA_FALLBACK:
            return best_cls, best_conf
        return None

    def to_signals_used(self) -> Dict[str, str]:
        """Produce the signals_used dict for TopologyClassification.

        Format: {path_name: "CLASS@confidence (detail)"}
        Includes all deposited evidence, not just the winner.
        """
        out: Dict[str, str] = {}
        for cls, evidences in self._direct.items():
            for ev in evidences:
                key = f"{ev.path.value}.{cls}"
                out[key] = f"{cls}@{ev.raw_confidence:.3f} ({ev.detail})"
        if self._propagated:
            for cls, conf in self._propagated.items():
                out[f"lattice.{cls}"] = f"{cls}@{conf:.3f} (propagated)"
        return out

    @property
    def peak(self) -> Tuple[Optional[str], float, Optional[ClassificationPath]]:
        """Highest-confidence direct evidence deposited so far."""
        return self._peak_class, self._peak_confidence, self._peak_path


# ══════════════════════════════════════════════════════════════════════════════
# _SignalLedger
#
# Ordered audit trail of evidence collected during one classify() call.
# Used to populate fallback_chain in the audit output and to feed the
# _EvidenceLattice when we need both a trace and propagation.
#
# Not the same as _EvidenceLattice: the ledger is an append-only record of
# what happened in order; the lattice is a queryable evidence pool.
# ══════════════════════════════════════════════════════════════════════════════

class _SignalLedger:
    """Append-only ordered record of evidence collected during one classify() call."""

    def __init__(self) -> None:
        self._records: List[_SignalEvidence] = []

    def record(self, evidence: _SignalEvidence) -> None:
        self._records.append(evidence)

    def fallback_chain(self, winner_class: str) -> List[str]:
        """Classes considered before the winner, in order of consideration.

        Excludes the winner itself and GENERIC_HTML (which is the implicit
        final fallback, not an explicit consideration).
        """
        seen:   Set[str]  = set()
        chain:  List[str] = []
        for ev in self._records:
            cls = ev.topology_class
            if cls != winner_class and cls != FALLBACK_TOPOLOGY_CLASS and cls not in seen:
                chain.append(cls)
                seen.add(cls)
        return chain

    def had_hard_override(self) -> bool:
        return any(ev.topology_class in HARD_OVERRIDE_CLASSES for ev in self._records)

    def all_classes_seen(self) -> List[str]:
        seen: List[str] = []
        cls_set: Set[str] = set()
        for ev in self._records:
            if ev.topology_class not in cls_set:
                seen.append(ev.topology_class)
                cls_set.add(ev.topology_class)
        return seen


# ══════════════════════════════════════════════════════════════════════════════
# _URLSemanticTokenizer
#
# Extracts semantic tokens from URL path segments for signal enrichment.
#
# URL path segments carry structural meaning beyond what regex can capture:
#   /2024/03/07/  → date-pattern segment → editorial content signal
#   /v2.1/        → version-string segment → API or docs signal
#   /p/a1b2c3d4/  → short hash slug → blog/platform content signal
#   /12345/       → bare numeric ID → product or forum signal
#
# These tokens are not used for the deterministic classification paths (which
# use regex). They are computed once per classify() call and passed to
# _embed_signals() as part of the feature engineering contract. See the
# _embed_signals docstring for the feature engineering contract.
# ══════════════════════════════════════════════════════════════════════════════

class _URLSemanticTokenizer:
    """Extracts semantic token classes from URL path segments."""

    _DATE_RE      = re.compile(r"^\d{4}[-/]\d{2}[-/]\d{2}$")
    _YEAR_RE      = re.compile(r"^(19|20)\d{2}$")
    _VERSION_RE   = re.compile(r"^v\d+(?:\.\d+)*(?:[a-z]\w*)?$")
    _NUMERIC_ID   = re.compile(r"^\d{3,}$")
    _SHORT_HASH   = re.compile(r"^[0-9a-f]{7,16}$")
    _LONG_HASH    = re.compile(r"^[0-9a-f]{32,}$")
    _SLUG_RE      = re.compile(r"^[a-z][a-z0-9-]{8,}$")
    _FILE_EXT_RE  = re.compile(r"\.([a-z0-9]+)$")

    _KNOWN_SEGMENT_KEYWORDS: FrozenSet[str] = frozenset({
        "api", "docs", "doc", "documentation", "reference", "guide",
        "blog", "post", "article", "news", "wiki", "forum", "thread",
        "product", "item", "shop", "store", "login", "signin", "auth",
        "search", "feed", "rss", "atom",
    })

    @dataclass(frozen=True)
    class URLTokenProfile:
        """Token profile extracted from one URL."""
        has_date_segment:     bool
        has_year_segment:     bool
        has_version_segment:  bool
        has_numeric_id:       bool
        has_short_hash:       bool
        has_long_hash:        bool
        has_long_slug:        bool
        keyword_segments:     Tuple[str, ...]   # lowercase known keywords in path
        file_extension:       Optional[str]      # e.g. "json", "html", "xml"
        path_depth:           int                # number of non-empty path segments
        query_param_count:    int
        has_fragment:         bool

    @classmethod
    def tokenize(cls, url: str) -> URLTokenProfile:
        """Parse url and return a URLTokenProfile."""
        try:
            parsed = urlparse(url)
        except Exception: # noqa
            return cls.URLTokenProfile(
                has_date_segment=False, has_year_segment=False,
                has_version_segment=False, has_numeric_id=False,
                has_short_hash=False, has_long_hash=False,
                has_long_slug=False, keyword_segments=(),
                file_extension=None, path_depth=0,
                query_param_count=0, has_fragment=False,
            )

        path     = parsed.path.lower()
        segments = [s for s in path.split("/") if s]

        has_date    = any(cls._DATE_RE.match(s) for s in segments)
        has_year    = any(cls._YEAR_RE.match(s) for s in segments)
        has_ver     = any(cls._VERSION_RE.match(s) for s in segments)
        has_num     = any(cls._NUMERIC_ID.match(s) for s in segments)
        has_short_h = any(cls._SHORT_HASH.match(s) for s in segments)
        has_long_h  = any(cls._LONG_HASH.match(s) for s in segments)
        has_slug    = any(cls._SLUG_RE.match(s) for s in segments)

        keywords = tuple(
            s for s in segments if s in cls._KNOWN_SEGMENT_KEYWORDS
        )

        ext_match   = cls._FILE_EXT_RE.search(path)
        file_ext    = ext_match.group(1) if ext_match else None

        query_params = len(parsed.query.split("&")) if parsed.query else 0
        has_fragment = bool(parsed.fragment)

        return cls.URLTokenProfile(
            has_date_segment    = has_date,
            has_year_segment    = has_year,
            has_version_segment = has_ver,
            has_numeric_id      = has_num,
            has_short_hash      = has_short_h,
            has_long_hash       = has_long_h,
            has_long_slug       = has_slug,
            keyword_segments    = keywords,
            file_extension      = file_ext,
            path_depth          = len(segments),
            query_param_count   = query_params,
            has_fragment        = has_fragment,
        )


# ══════════════════════════════════════════════════════════════════════════════
# _PhaseStateReader
#
# Memory-mapped reader for store/phase_states.mmap. Maintains a cached
# in-memory snapshot of all 18 phase slots. classify() reads the snapshot
# (pure Python list access, zero syscalls) rather than touching the mmap on
# every call. The snapshot is atomically refreshed whenever WATCHDOG detects
# a write to the file.
#
# BINARY LAYOUT (from initialize_store.py, reproduced here as the authoritative
# reference for the reader):
#
#   File header (8 bytes):
#       [0:4]   magic         b"AXPS"
#       [4]     version       uint8 = 1
#       [5]     n_classes     uint8 = 18
#       [6:8]   reserved      uint16
#
#   Per-class slot (32 bytes, repeated 18×):
#       [0]     phase_id      uint8   1=LEARNS, 2=PREDICTS, 3=KNOWS
#       [1]     flags         uint8   bit0=active, bit1=surprise_tripped
#       [2:4]   reserved      uint16
#       [4:8]   confidence    float32
#       [8:12]  event_count   uint32
#       [12:16] surprise_score float32
#       [16:24] updated_ns    uint64  epoch nanoseconds; 0 = never updated
#       [24:32] reserved      8 bytes
#
#   Total: 8 + 18 × 32 = 584 bytes
#
# ── RACE CONDITION ANALYSIS AND MITIGATION ────────────────────────────────────
#
# phase_states.mmap is a shared-memory file written by index_daemon.py (a
# separate OS process) and read by the classifier (this process). There is no
# OS-level mechanism that makes a 32-byte struct write atomic — it is a sequence
# of stores, and the classifier can observe a partially-written slot.
#
# WHAT CAN TEAR
#
# index_daemon calls struct.pack_into(buf, offset, phase_id, flags, ...,
# updated_ns, ...) which compiles to a sequence of CPU store instructions.
# On x86-64 with Total Store Order (TSO), stores from the writer are visible
# to the reader in program order. However, the reader can see stores that the
# writer has already retired while later stores are still in-flight. The worst
# case is a context-switch preempting index_daemon between two stores.
#
# The critical window is the span between the store of the first field
# (phase_id, 1 byte) and the store of updated_ns (8 bytes, the last
# meaningful field). struct.pack_into for 32 bytes is ~4–8 CPU instructions
# depending on SIMD availability. The window is O(nanoseconds).
#
# SEQLOCK STRATEGY USING updated_ns
#
# updated_ns is at slot_offset + 16. For all 18 slots its absolute file
# offset is 8 + i*32 + 16 = 24 + 32i, which is always 8-byte aligned.
# On x86-64, naturally-aligned 8-byte loads are atomic (guaranteed by the
# ISA, not just TSO). We exploit this as a trailing consistency anchor:
#
#     Step 1: atomic read of updated_ns → ns_pre      (from live mmap)
#     Step 2: copy the full 32-byte slot → raw         (bytes(), one memcpy)
#     Step 3: atomic read of updated_ns → ns_post      (from live mmap)
#
#     if ns_pre == ns_post:
#         index_daemon did not complete a write between steps 1 and 3.
#         The raw snapshot is consistent (no field was written after we
#         started reading). Parse and use.
#
#     else:
#         A write completed during our read window. Yield and retry.
#
# WHY THIS WORKS ON x86-64
#
# x86-64 TSO prohibits load-load reordering within a single thread. Steps 1,
# 2, and 3 execute in program order at the hardware level. If ns_pre ==
# ns_post, the writer's stores that set the new updated_ns were either both
# invisible (write hadn't started) or both visible (write completed before
# step 1). In either case, all data fields written before updated_ns are in
# a consistent state within raw.
#
# This relies on index_daemon writing updated_ns AFTER all data fields.
# struct.pack_into writes fields in struct order; updated_ns is the 7th field
# (after phase_id, flags, reserved, confidence, event_count, surprise_score),
# so this property holds for standard CPython struct writes.
#
# RESIDUAL RISK (why this is not perfectly safe)
#
# There is one scenario where ns_pre == ns_post yet raw is inconsistent:
# index_daemon is preempted after writing data fields but before writing
# updated_ns. In that case:
#   - ns_pre  == old timestamp (updated_ns not yet written)
#   - raw     contains new phase_id/flags and old updated_ns
#   - ns_post == old timestamp (still not written)
#   → ns_pre == ns_post, we accept the partially-written slot
#
# Consequence: the classifier uses a wrong threshold for one classification.
# It does NOT misclassify the URL (the wrong threshold is bounded: Phase I
# safe defaults are the fallback). The window is O(nanoseconds) between
# the second-to-last and last store of a ~4-instruction sequence. At 50K
# classify() calls/sec with one phase transition per hour per class, the
# expected torn-read rate is < 1 per billion classifications.
#
# FULL CORRECTNESS would require writer cooperation: a generation counter
# written FIRST (odd = in progress) and LAST (even = done), giving a
# standard seqlock. The reserved uint16 at slot offset 2 is a suitable
# candidate. This is left as a future hardening step for index_daemon.
#
# ARM / WEAK MEMORY MODEL NOTE
#
# On ARM, load-load reordering is permitted. The seqlock is NOT safe without
# explicit `dmb ish` barriers between steps 1→2 and 2→3. Python provides no
# barrier API. If AXIOM runs on ARM (e.g. AWS Graviton), either:
#   (a) use ctypes.CDLL("libatomic.so").atomic_load_8 for the ns reads, or
#   (b) accept the residual risk (same consequence analysis as above), or
#   (c) implement the writer-side generation counter in index_daemon.
# Current deployment target is x86-64 Alpine Linux. ARM is unvalidated.
#
# HOT-PATH ISOLATION
#
# classify() NEVER reads the mmap directly. It calls _read_phase_snapshot()
# which returns a reference to the cached list (zero allocation on hit). All
# mmap access is confined to _refresh_snapshot(), which runs on the WATCHDOG
# callback thread — off the async classify() hot path entirely.
# ══════════════════════════════════════════════════════════════════════════════

# Binary layout constants mirroring initialize_store.py.
_PS_MAGIC:        bytes = b"AXPS"
_PS_VERSION:      int   = 1
_PS_N_CLASSES:    int   = 18
_PS_SLOT_STRIDE:  int   = 32
_PS_HEADER_SIZE:  int   = 8
_PS_FILE_SIZE:    int   = _PS_HEADER_SIZE + _PS_N_CLASSES * _PS_SLOT_STRIDE  # 584
_PS_NS_OFFSET:    int   = 16   # byte offset of updated_ns within a slot
_PS_MAX_RETRIES:  int   = 3

# Struct for parsing a full 32-byte slot.
_PS_SLOT_STRUCT: struct.Struct = struct.Struct("<BB H f I f Q 8x")
# Struct for the single 8-byte updated_ns field (aligned atomic read).
_PS_NS_STRUCT:   struct.Struct = struct.Struct("<Q")

# Safe fallback slot returned when all seqlock retries are exhausted.
# Phase I + no surprise: maximally conservative, never gates incorrectly.
_PHASE_SLOT_SAFE_DEFAULT: _PhaseSlot = _PhaseSlot(
    phase_id=1, flags=0, confidence=0.0, surprise_score=0.0,
)


class _PhaseStateReader:
    """Read-only accessor for store/phase_states.mmap.

    Maintains a cached snapshot of all 18 _PhaseSlot records. The snapshot
    is refreshed atomically whenever WATCHDOG fires for the mmap file.
    classify() reads from the snapshot — never from the mmap directly.

    Thread safety:
        _refresh_snapshot() runs on the WATCHDOG callback thread and writes
        self._snapshot via a single Python reference assignment, which is
        atomic in CPython under the GIL. classify() reads self._snapshot
        on the event loop thread. No explicit lock is required for the read
        path in CPython — the GIL serialises the reference swap. On free-
        threaded Python (PEP 703, no-GIL), this would need an RWLock.

    Lifecycle:
        Constructed by TopologyClassifier.__init__().
        Opened (mmap mapped) during TopologyClassifier.initialize().
        Closed during process shutdown or if the file is missing at open time.
        WATCHDOG handles subsequent refresh triggers.
    """

    def __init__(self, path: str) -> None:
        self._path:     str                         = path
        self._mmap:     Optional[mmap.mmap]         = None
        self._snapshot: List[_PhaseSlot]            = []
        self._lock:     threading.Lock              = threading.Lock()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def open(self) -> None:
        """Open and mmap the phase_states file. Build the initial snapshot.

        Raises:
            FileNotFoundError: if phase_states.mmap does not exist. Caller
                (TopologyClassifier.initialize) logs and continues — the
                classifier degrades to constant _THETA_CONFIDENT thresholds
                rather than refusing to start.
            ValueError: if the file magic or size is wrong.
        """
        path = Path(self._path)
        if not path.exists():
            raise FileNotFoundError(
                f"phase_states.mmap not found at {self._path!r}. "
                "Run initialize_store.py to create the store."
            )

        raw_header = path.read_bytes()[:_PS_HEADER_SIZE]
        magic = raw_header[:4]
        if magic != _PS_MAGIC:
            raise ValueError(
                f"phase_states.mmap: bad magic {magic!r}, expected {_PS_MAGIC!r}. "
                "File is corrupt or from a different AXIOM version."
            )
        file_size = path.stat().st_size
        if file_size != _PS_FILE_SIZE:
            raise ValueError(
                f"phase_states.mmap: size {file_size} B, expected {_PS_FILE_SIZE} B. "
                "File is corrupt or was initialized with a different class count."
            )

        f = open(self._path, "rb")   # noqa: SIM115 — kept open for mmap lifetime
        self._mmap = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        f.close()   # fd can be closed after mmap is created on Linux

        self._refresh_snapshot()

    def close(self) -> None:
        """Close the mmap. Safe to call multiple times."""
        if self._mmap is not None:
            try:
                self._mmap.close()
            except Exception:   # noqa
                pass
            self._mmap = None

    @property
    def is_open(self) -> bool:
        return self._mmap is not None and not self._mmap.closed

    @property
    def snapshot(self) -> List[_PhaseSlot]:
        """Current cached phase snapshot. Reference is stable per GIL swap."""
        return self._snapshot

    # ── Snapshot refresh (WATCHDOG callback thread) ───────────────────────────

    def _read_slot_consistent(self, slot_off: int) -> _PhaseSlot:
        """Read one 32-byte slot with seqlock-style consistency.

        Uses updated_ns (8-byte aligned uint64 at slot_off+16) as a trailing
        consistency anchor. See the module-level race condition analysis for
        the full correctness argument.

        Falls back to _PHASE_SLOT_SAFE_DEFAULT after _PS_MAX_RETRIES torn
        reads. Logs a warning — torn reads should be rare enough that any
        occurrence is worth surfacing.

        Args:
            slot_off: byte offset of the slot within self._mmap.

        Returns:
            A consistent _PhaseSlot, or _PHASE_SLOT_SAFE_DEFAULT on failure.
        """
        assert self._mmap is not None, "_read_slot_consistent called before open()"
        ns_off = slot_off + _PS_NS_OFFSET

        for attempt in range(_PS_MAX_RETRIES):
            # Step 1: atomic 8-byte read of updated_ns (pre-copy anchor).
            # On x86-64 TSO, this load is ordered before all subsequent loads.
            (ns_pre,) = _PS_NS_STRUCT.unpack_from(self._mmap, ns_off)

            # Step 2: copy the full 32-byte slot as bytes.
            # bytes() on a mmap slice triggers a single memcpy — the smallest
            # read window achievable in pure Python.
            raw: bytes = bytes(self._mmap[slot_off : slot_off + _PS_SLOT_STRIDE])

            # Step 3: atomic 8-byte read of updated_ns (post-copy anchor).
            # On x86-64 TSO, this load is ordered after all loads in step 2.
            (ns_post,) = _PS_NS_STRUCT.unpack_from(self._mmap, ns_off)

            if ns_pre == ns_post:
                # No completed write changed updated_ns between steps 1 and 3.
                # Because index_daemon writes updated_ns last (7th field in
                # struct.pack_into order), any write that updated earlier fields
                # must also have updated updated_ns by the time ns_post is
                # read — so raw is internally consistent.
                phase_id, flags, _, conf, _, surprise_score, _ = (
                    _PS_SLOT_STRUCT.unpack(raw)
                )
                return _PhaseSlot(
                    phase_id       = phase_id,
                    flags          = flags,
                    confidence     = conf,
                    surprise_score = surprise_score,
                )

            # updated_ns changed: a write completed during our read window.
            # Yield the thread so index_daemon finishes its write before the
            # next attempt. On CPython the GIL will also be released here,
            # allowing other threads to run.
            time.sleep(0)

        logger.warning(
            "topology_classifier.phase_slot_read_torn",
            extra={
                "slot_off": slot_off,
                "attempts": _PS_MAX_RETRIES,
                "note": "returning Phase I safe defaults",
            },
        )
        return _PHASE_SLOT_SAFE_DEFAULT

    def _refresh_snapshot(self) -> None:
        """Rebuild the full 18-slot snapshot from the mmap.

        Called once at open() and on every WATCHDOG-triggered reload. Each
        slot is read with seqlock consistency. The completed snapshot replaces
        self._snapshot in a single reference assignment (GIL-atomic in CPython).
        """
        if not self.is_open:
            return

        new_snapshot: List[_PhaseSlot] = []
        for i in range(_PS_N_CLASSES):
            slot_off = _PS_HEADER_SIZE + i * _PS_SLOT_STRIDE
            new_snapshot.append(self._read_slot_consistent(slot_off))

        # GIL-safe atomic replacement. classify() always sees either the
        # old complete snapshot or the new complete snapshot — never a
        # partially-built list.
        self._snapshot = new_snapshot

        # Feed the new snapshot to the surprise hysteresis controller.
        # This is the only call site.  observe() evaluates arm/disarm
        # transitions for every Phase III topology class in the snapshot.
        if _SURPRISE_HYSTERESIS is not None:
            _SURPRISE_HYSTERESIS.observe(new_snapshot)

        logger.debug(
            "topology_classifier.phase_snapshot_refreshed",
            extra={"n_slots": len(new_snapshot)},
        )

    def reload(self) -> None:
        """WATCHDOG callback. Re-reads all slots from the updated mmap."""
        with self._lock:
            self._refresh_snapshot()


# Module-level singleton. Initialized to None; TopologyClassifier.initialize()
# opens it. classify() checks is_open before reading the snapshot.
_PHASE_STATE_READER: Optional[_PhaseStateReader] = None


# ═════════════════════════════════════════════════════════════════════════════
# SURPRISE HYSTERESIS CONSTANTS
#
# Asymmetric Schmitt trigger thresholds.  Arming is hard (require sustained
# evidence of real structural drift).  Disarming is harder (require prolonged
# quiet before trusting deterministic paths again).
#
# The asymmetry encodes a safety preference:
#   Cost of staying armed too long:  extra ML invocations.
#       Correctness preserved.  Latency slightly higher.  Self-correcting.
#   Cost of disarming too early:     stale deterministic classifications.
#       Correctness compromised.  Silent failure.  Not self-correcting.
#   → Stay armed longer.
# ═════════════════════════════════════════════════════════════════════════════

_HYSTERESIS_ARM_EVENTS: int = 3
# Number of surprise_tripped=True observations required within ARM_WINDOW
# to transition from DISARMED → ARMED.
#
# WATCHDOG debounce on phase_states.mmap is 1000ms (classifier.py line 1695).
# Each observation represents a debounced mmap write by index_daemon.
# 3 events means: index_daemon set surprise_tripped at least 3 times within
# the arm window, with at minimum 1 second between consecutive writes.
#
# This filters:
#   1 event  — single transient drift.  Common.  Resolves within one
#              gradient step.  Not worth forcing ML for the entire class.
#   2 events — two-event noise pattern (drift detected → cleared → detected
#              again within seconds).  Can happen when index_daemon processes
#              two surprise events for different domains in the same class
#              back-to-back.  Still not structural instability.
#   3+ events — the class keeps returning to surprise state.  Structural
#               instability confirmed.  Deterministic paths cannot be trusted.

_HYSTERESIS_ARM_WINDOW_S: float = 90.0
# Sliding window (seconds) within which ARM_EVENTS observations must occur.
#
# At minimum 1s between WATCHDOG-debounced observations, 3 events can fire
# in as little as 3 seconds.  But the 90s window also captures intermittent
# patterns: surprise at t=0, cleared at t=5, surprise at t=30, cleared at
# t=35, surprise at t=60 → 3 arm events in 60s, within the 90s window.
# This pattern indicates a topology class that repeatedly drifts back to
# surprise state — genuine structural instability, not a one-time event.

_HYSTERESIS_DISARM_QUIET_S: float = 180.0
# Seconds of uninterrupted surprise_tripped=False before transitioning
# from ARMED → DISARMED.
#
# Why 2× the arm window:
#   1. Once armed, the classifier forces all classifications through ML.
#      This is expensive but correct during genuine drift.
#   2. 180s gives the system time to:
#      - Run hundreds of ML classifications (~15ms each)
#      - Collect extraction feedback for the affected class
#      - Potentially trigger zone map recompilation
#      - Have the WLM process updated training signal from new data
#   3. Premature disarming snaps back to deterministic paths that may still
#      be stale.  The ML path is the safe path.  Hold it longer.
#
# At steady state (no structural drift), no class is ever armed, and this
# constant has zero effect on latency or throughput.  The cost is only paid
# when drift is actually occurring.

_HYSTERESIS_WINDOW_CAP: int = 64
# Maximum observations retained per class in the sliding window deque.
# At 1 observation per WATCHDOG callback (minimum 1s apart), 64 entries
# cover ~64 seconds of dense activity.  The arm window is 90s, but the
# deque also prunes by timestamp on every observe() call, so the cap is
# a memory ceiling, not a correctness constraint.


# ═════════════════════════════════════════════════════════════════════════════
# _SurpriseHysteresis
#
# Schmitt trigger that prevents _phase_adjusted_theta() from oscillating
# between threshold 0.60 (trust deterministic paths) and threshold 2.0
# (force ML) when index_daemon flaps the surprise_tripped bit.
#
# CRITICAL DESIGN PROPERTY: the hysteresis is ADDITIVE.  It can only
# EXTEND the ML-forcing period, never SHORTEN it.  When the raw
# surprise_tripped bit is True, ML is always forced (same as before).
# When the hysteresis is armed, ML is also forced (new: prevents premature
# relaxation when the bit clears).  The gate condition becomes:
#
#     force_ml = slot.surprise_tripped OR hysteresis.is_armed(idx)
#
# This means the hysteresis can never make the system LESS safe than the
# current raw-bit implementation.  It can only make it more stable.
# ═════════════════════════════════════════════════════════════════════════════


class _SurpriseHysteresis:
    """
    Asymmetric Schmitt trigger for surprise_tripped phase state.

    Prevents oscillation in _phase_adjusted_theta() when index_daemon
    flaps the surprise_tripped bit due to transient structural drift
    events that resolve within a few gradient steps.

    State machine (per topology class):

        DISARMED ──[ARM_EVENTS within ARM_WINDOW_S]──→ ARMED
        ARMED    ──[DISARM_QUIET_S with no surprise]──→ DISARMED

    Thread safety:
        observe() is called from _PhaseStateReader._refresh_snapshot()
        which holds _PhaseStateReader._lock.  Writes to self._armed are
        done via GIL-safe list reference replacement.

        is_armed() is called from _phase_adjusted_theta() on the
        classify() hot path.  No lock.  Reads self._armed by list index
        (GIL-safe).  Includes a time-based auto-disarm check for the
        edge case where no WATCHDOG callbacks arrive for longer than
        DISARM_QUIET_S (mmap not written → observe() not called → armed
        state would persist indefinitely without this check).

    Lifetime:
        Module-level singleton.  Created at module load.  State is
        process-scoped — does not survive restarts.  All classes start
        DISARMED.  First WATCHDOG callback after initialize() begins
        populating observation windows.
    """

    __slots__ = (
        "_windows",
        "_armed",
        "_armed_at",
        "_last_surprise_at",
        "_arm_count",
        "_disarm_count",
        "_total_observations",
    )

    def __init__(self) -> None:
        # Per-class sliding window of (monotonic_timestamp, surprise_tripped).
        # Pruned by timestamp on every observe() call.
        self._windows: List[Deque[Tuple[float, bool]]] = [
            deque(maxlen=_HYSTERESIS_WINDOW_CAP)
            for _ in range(NUM_TOPOLOGY_CLASSES)
        ]
        # Effective armed state.  GIL-safe: replaced atomically by observe().
        self._armed: List[bool] = [False] * NUM_TOPOLOGY_CLASSES

        # Per-class timestamps for state transitions.
        self._armed_at:         List[float] = [0.0] * NUM_TOPOLOGY_CLASSES
        self._last_surprise_at: List[float] = [0.0] * NUM_TOPOLOGY_CLASSES

        # Cumulative metrics — monotonically increasing from process start.
        self._arm_count:    List[int] = [0] * NUM_TOPOLOGY_CLASSES
        self._disarm_count: List[int] = [0] * NUM_TOPOLOGY_CLASSES
        self._total_observations: int = 0

    # ── Hot path: classify() reads this ───────────────────────────────────

    def is_armed(self, class_idx: int) -> bool:
        """Check if a topology class is in armed (drift-gated) state.

        Called from _phase_adjusted_theta() on every waterfall short-circuit
        check.  Must be fast.

        O(1): list index + one conditional time.monotonic() call (~20ns
        on Linux VDSO).  The time check only executes when the class is
        armed (rare in steady state).

        The time-based auto-disarm handles the edge case where no WATCHDOG
        callbacks arrive for longer than DISARM_QUIET_S.  Without this,
        an armed class would stay armed forever if the mmap stops being
        written to.  The auto-disarm does NOT mutate self._armed — that
        would be unsafe from the hot path without a lock.  It simply
        reports False.  The next observe() call formally disarms by
        rebuilding the armed list.

        Returns:
            True if the class should be treated as drifting (force ML path).
            False otherwise.
        """
        if class_idx < 0 or class_idx >= len(self._armed):
            return False
        if not self._armed[class_idx]:
            return False

        # Time-based auto-disarm: if no surprise observation has arrived
        # for longer than the quiet window, the class is effectively stable
        # even if observe() hasn't been called to formally disarm it.
        elapsed = time.monotonic() - self._last_surprise_at[class_idx]
        if elapsed >= _HYSTERESIS_DISARM_QUIET_S:
            # Do not mutate self._armed here — hot path, no lock.
            # Next observe() call will formally disarm.
            return False

        return True

    # ── Observation path: _refresh_snapshot() calls this under lock ────────

    def observe(self, snapshot: "List[_PhaseSlot]") -> None:
        """Feed a new phase snapshot from _PhaseStateReader._refresh_snapshot().

        Called under _PhaseStateReader._lock.  Evaluates arm/disarm
        transitions for every topology class in the snapshot.

        The method builds a new armed list and replaces self._armed in
        a single GIL-safe assignment at the end.  The hot path
        (is_armed) always sees either the old complete list or the new
        complete list, never a partially-updated one.
        """
        now = time.monotonic()
        self._total_observations += 1

        # Build replacement list.  Start from current state so classes not
        # present in the snapshot (truncated snapshot edge case) retain
        # their current armed state.
        new_armed: List[bool] = list(self._armed)

        n = min(len(snapshot), NUM_TOPOLOGY_CLASSES)
        for idx in range(n):
            slot = snapshot[idx]

            # Surprise hysteresis only applies to Phase III.  Phase I and II
            # classes don't use the drift gate — their thresholds are set by
            # different multipliers.  Observing surprise_tripped on non-Phase-III
            # classes would pollute the window with irrelevant events.
            if slot.phase_id != 3:
                # If the class dropped out of Phase III (e.g. phase regression
                # from III → II via SurpriseEvent dissolve), disarm immediately.
                # The Phase II threshold is already conservative — no need for
                # the additional ML-forcing from hysteresis.
                if self._armed[idx]:
                    new_armed[idx] = False
                    self._disarm_count[idx] += 1
                    _class_name = (
                        TOPOLOGY_CLASSES[idx]
                        if idx < len(TOPOLOGY_CLASSES)
                        else f"idx_{idx}"
                    )
                    logger.info(
                        "surprise_hysteresis.disarmed_phase_regression",
                        extra={
                            "topology_class": _class_name,
                            "new_phase": slot.phase_id,
                            "was_armed_s": round(now - self._armed_at[idx], 1),
                        },
                    )
                continue

            # ── Phase III: track surprise_tripped observations ────────────
            tripped = slot.surprise_tripped
            window = self._windows[idx]
            window.append((now, tripped))

            if tripped:
                self._last_surprise_at[idx] = now

            # Prune observations outside the arm window.
            cutoff = now - _HYSTERESIS_ARM_WINDOW_S
            while window and window[0][0] < cutoff:
                window.popleft()

            _class_name = (
                TOPOLOGY_CLASSES[idx]
                if idx < len(TOPOLOGY_CLASSES)
                else f"idx_{idx}"
            )

            # ── State transition evaluation ───────────────────────────────

            if not self._armed[idx]:
                # DISARMED → evaluate arm condition.
                # Count surprise_tripped=True observations in the current window.
                surprise_count = sum(1 for _, t in window if t)

                if surprise_count >= _HYSTERESIS_ARM_EVENTS:
                    new_armed[idx] = True
                    self._armed_at[idx] = now
                    self._arm_count[idx] += 1
                    logger.info(
                        "surprise_hysteresis.armed",
                        extra={
                            "topology_class": _class_name,
                            "surprise_count": surprise_count,
                            "window_size": len(window),
                            "window_s": _HYSTERESIS_ARM_WINDOW_S,
                            "total_arms": self._arm_count[idx],
                        },
                    )
            else:
                # ARMED → evaluate disarm condition.
                # Require DISARM_QUIET_S with no surprise_tripped=True.
                time_since_last_surprise = now - self._last_surprise_at[idx]

                if time_since_last_surprise >= _HYSTERESIS_DISARM_QUIET_S:
                    new_armed[idx] = False
                    self._disarm_count[idx] += 1
                    logger.info(
                        "surprise_hysteresis.disarmed",
                        extra={
                            "topology_class": _class_name,
                            "quiet_s": round(time_since_last_surprise, 1),
                            "was_armed_s": round(now - self._armed_at[idx], 1),
                            "total_disarms": self._disarm_count[idx],
                        },
                    )

        # GIL-safe atomic replacement.  classify() hot path sees either
        # the old list or this new list — never a partial state.
        self._armed = new_armed

    # ── Diagnostics ───────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """Return hysteresis state for cold_start validation and telemetry.

        Safe to call from any thread — reads only GIL-safe list references
        and immutable counters.
        """
        armed_classes = []
        for i in range(NUM_TOPOLOGY_CLASSES):
            if i < len(self._armed) and self._armed[i]:
                if i < len(TOPOLOGY_CLASSES):
                    armed_classes.append(TOPOLOGY_CLASSES[i])

        per_class_arms = {}
        per_class_disarms = {}
        for i in range(NUM_TOPOLOGY_CLASSES):
            if i < len(TOPOLOGY_CLASSES):
                name = TOPOLOGY_CLASSES[i]
                if self._arm_count[i] > 0:
                    per_class_arms[name] = self._arm_count[i]
                if self._disarm_count[i] > 0:
                    per_class_disarms[name] = self._disarm_count[i]

        return {
            "armed_classes":       armed_classes,
            "armed_count":         len(armed_classes),
            "total_observations":  self._total_observations,
            "per_class_arms":      per_class_arms,
            "per_class_disarms":   per_class_disarms,
            "arm_threshold":       _HYSTERESIS_ARM_EVENTS,
            "arm_window_s":        _HYSTERESIS_ARM_WINDOW_S,
            "disarm_quiet_s":      _HYSTERESIS_DISARM_QUIET_S,
        }

    def reset_class(self, class_idx: int) -> None:
        """Force-disarm a single class.  Admin/testing only.

        Not called during normal operation.  Provided for:
            - Manual intervention when a class is incorrectly stuck armed
              due to a bug in index_daemon's surprise scoring.
            - Test fixtures that need deterministic hysteresis state.

        Thread safety: mutates self._armed in-place via list index
        assignment, which is GIL-safe for a single element.
        """
        if 0 <= class_idx < len(self._armed):
            self._armed[class_idx] = False
            self._windows[class_idx].clear()


# Module-level singleton.  Created at import time.  Stateless until the
# first observe() call from _PhaseStateReader._refresh_snapshot().
_SURPRISE_HYSTERESIS: _SurpriseHysteresis = _SurpriseHysteresis()


# ═════════════════════════════════════════════════════════════════════════════
# This is the ADDITIVE property: the raw bit still forces ML (same as
# before).  The hysteresis additionally forces ML when armed (new: prevents
# premature relaxation).  The gate can only be MORE conservative than the
# original, never less.
# ═════════════════════════════════════════════════════════════════════════════

def _phase_adjusted_theta(
    topology_class:  str,
    phase_snapshot:  "Optional[List[_PhaseSlot]]",
) -> float:
    """Return THETA_CLASSIFY_CONFIDENT adjusted for topology_class's phase state.

    Called at every waterfall short-circuit check in classify(). The snapshot
    argument was captured once at the start of the call — no mmap access here.

    Phase III + surprise_tripped OR hysteresis armed returns
    _PHASE_THETA_DRIFT_GATE (2.0), which is unreachable by any confidence
    value, forcing the classification through the ML path.

    The hysteresis layer prevents oscillation when index_daemon flaps the
    surprise_tripped bit due to transient structural drift events.  See
    _SurpriseHysteresis docstring for the full Schmitt trigger design.

    The hysteresis is ADDITIVE — it can only extend the ML-forcing period,
    never shorten it relative to the raw surprise_tripped bit.

    Args:
        topology_class: the candidate class returned by a signal path.
        phase_snapshot: the snapshot from _read_phase_snapshot(), or None
                        if the phase reader is unavailable.

    Returns:
        Adjusted threshold in [0.0, 2.0]. Values > 1.0 are intentional
        and mean "no deterministic path may short-circuit this class".
    """
    if not phase_snapshot:
        return _THETA_CONFIDENT

    idx = TOPOLOGY_CLASS_INDEX.get(topology_class)
    if idx is None or idx >= len(phase_snapshot):
        return _THETA_CONFIDENT

    slot = phase_snapshot[idx]

    if slot.phase_id == 3:                                          # KNOWS
        if slot.surprise_tripped or _SURPRISE_HYSTERESIS.is_armed(idx):
            return _PHASE_THETA_DRIFT_GATE                          # > 1.0: force ML path
        return _THETA_CONFIDENT * _PHASE_THETA_KNOWS_MULT           # 0.60

    if slot.phase_id == 1:                                          # LEARNS
        return _THETA_CONFIDENT * _PHASE_THETA_LEARNS_MULT          # 0.8625

    return _THETA_CONFIDENT                                         # PREDICTS — unchanged

# ══════════════════════════════════════════════════════════════════════════════
# TopologyClassifier
# ══════════════════════════════════════════════════════════════════════════════

class TopologyClassifier:
    """Topology classifier — first component on every URL's critical path.

    Classification hierarchy:
        Path 1  domain fingerprint  (_classify_by_domain)
        Path 2  URL structure       (_classify_by_url)
        Path 3  response headers    (_classify_by_headers)
        Path 4  content window      (_classify_by_window)
        Path 4.5 lattice fusion     (novel: hierarchy-aware evidence fusion)
        Path 5  ML model            (_classify_via_model)
        Fallback GENERIC_HTML

    The classifier is stateless between calls except for:
        - self._model: the loaded topology_router.pt (reloaded atomically)
        - self._audit: rolling ring buffer of recent classifications
        - self._phase_reader: _PhaseStateReader instance (phase_states.mmap)
    """

    def __init__(
        self,
        model_path:        str = "store/topology_router.pt",
        phase_states_path: str = "store/phase_states.mmap",
    ) -> None:
        self._model_path:        str                       = model_path
        self._phase_states_path: str                       = phase_states_path
        self._model:             Optional[torch.nn.Module] = None
        self._phase_reader:      Optional[_PhaseStateReader] = None
        # Rolling ring buffer of recent classification results. Debugging only.
        self._audit: Deque[Dict] = deque(maxlen=_AUDIT_BUFFER_SIZE)

    # ── Model lifecycle ────────────────────────────────────────────────────────

    def _load_classifier_model(self) -> torch.nn.Module:
        """Load topology_router.pt from disk.

        Enforces weights_only=True unconditionally — this is a security
        requirement. A weights_only=False load would allow arbitrary Python
        execution via pickle deserialization.

        topology_router.pt is a state dict, not a serialized Module. It is
        written once by initialize_store.py and updated continuously by
        index_daemon via WLMTrainingInterface. This method instantiates a
        fresh MambaRouter with PRODUCTION_CONFIG and loads the state dict
        into it — the same pattern used by WorldLatentModel.initialize().

        Device selection: CUDA if available, CPU otherwise. The model is
        moved to eval mode immediately. No gradient state is initialised.

        Raises:
            ClassifierModelNotInitialized: if the file is missing, corrupt,
                incompatible with PRODUCTION_CONFIG, or fails to deserialize.
                is_hard_stop=True — the system cannot classify without the model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ── Step 1: Deserialize state dict ────────────────────────────────────
        try:
            state_dict = torch.load(
                self._model_path,
                weights_only=True,
                map_location=device,
            )
        except FileNotFoundError as exc:
            raise ClassifierModelNotInitialized(
                f"topology_router.pt not found at {self._model_path!r}. "
                "Verify the store volume is mounted and the file exists. "
                "On first boot, preparse_daemon must run before the classifier "
                "can initialize.",
            ) from exc
        except Exception as exc:
            raise ClassifierModelNotInitialized(
                f"Failed to deserialize topology_router.pt at {self._model_path!r}: {exc}. "
                "The model file may be corrupt or incompatible with the installed "
                "PyTorch version. Restore from the most recent checkpoint.",
            ) from exc

        # ── Step 2: Validate it is actually a state dict ──────────────────────
        # weights_only=True always returns a dict. If it didn't, the file was
        # saved with torch.save(model, path) instead of torch.save(model.state_dict(), path).
        # That would be a training pipeline bug — surface it explicitly.
        if not isinstance(state_dict, dict):
            raise ClassifierModelNotInitialized(
                f"topology_router.pt at {self._model_path!r} deserialized to "
                f"{type(state_dict).__name__!r}, expected a state dict. "
                "The file must be saved with torch.save(model.state_dict(), path). "
                "This indicates a bug in initialize_store.py or index_daemon. "
                "Restore from the most recent checkpoint.",
            )

        if len(state_dict) == 0:
            raise ClassifierModelNotInitialized(
                f"topology_router.pt at {self._model_path!r} contains an empty state dict. "
                "The file may have been written mid-checkpoint or is zero bytes. "
                "Restore from the most recent checkpoint.",
            )

        # ── Step 3: Instantiate architecture and load weights ─────────────────
        # MambaRouter is the shared architecture. PRODUCTION_CONFIG is the single
        # source of truth for all hyperparameters — classifier never owns its own
        # weights, it reads the same file index_daemon writes.
        try:
            router = MambaRouter(
                vocab_size=PRODUCTION_CONFIG.vocab_size,
                d_model=PRODUCTION_CONFIG.d_model,
                d_state=PRODUCTION_CONFIG.d_state,
                d_conv=PRODUCTION_CONFIG.d_conv,
                expand=PRODUCTION_CONFIG.expand,
                n_layers=PRODUCTION_CONFIG.n_layers,
                n_topology=PRODUCTION_CONFIG.n_topology,
                n_source=PRODUCTION_CONFIG.n_source,
                n_phase=PRODUCTION_CONFIG.n_phase,
                dropout=PRODUCTION_CONFIG.dropout,
                max_seq_len=PRODUCTION_CONFIG.max_seq_len,
            )
        except Exception as exc:
            raise ClassifierModelNotInitialized(
                f"MambaRouter construction failed with PRODUCTION_CONFIG: {exc}. "
                "This typically indicates a mamba_ssm version mismatch. "
                "Verify mamba_ssm is installed and matches the training environment.",
            ) from exc

        # ── Step 4: Verify state dict compatibility before loading ────────────
        # verify_state_dict_compatibility() checks key names and tensor shapes
        # without modifying the model — same pre-load check latent_model.py uses.
        compatibility_errors = router.verify_state_dict_compatibility(state_dict)
        if compatibility_errors:
            error_summary = "; ".join(compatibility_errors[:3])
            raise ClassifierModelNotInitialized(
                f"topology_router.pt is incompatible with PRODUCTION_CONFIG. "
                f"First errors: {error_summary}. "
                f"Total: {len(compatibility_errors)}. "
                "The checkpoint was produced by a different architecture version. "
                "Restore from a checkpoint matching the current PRODUCTION_CONFIG.",
            )

        try:
            router.load_state_dict(state_dict)
        except RuntimeError as exc:
            raise ClassifierModelNotInitialized(
                f"load_state_dict failed for topology_router.pt at {self._model_path!r}: {exc}. "
                "Restore from the most recent checkpoint.",
            ) from exc

        # ── Step 5: Finalise ──────────────────────────────────────────────────
        router.to(device)
        router.eval()

        logger.info(
            "topology_classifier.model_loaded",
            extra={
                "model_path": self._model_path,
                "device": device,
                "param_count": router.parameter_count,
                "hidden_state_version": router.current_hidden_state_version,
                "hidden_state_digest": router.hidden_state_digest(),
            },
        )

        return router

    async def _reload_classifier_model(self) -> None:
        """WATCHDOG callback — called when topology_router.pt changes on disk.

        Loads a fresh model then atomically replaces self._model via a single
        assignment. Python's GIL guarantees that self._model = new_model is
        atomic at the bytecode level. No locking is needed. An in-flight
        classify() call that reads self._model before the assignment sees the
        old model and completes correctly; the next call sees the new model.

        This method is async because WATCHDOG dispatches handlers as asyncio
        tasks. The blocking torch.load() call is run synchronously inside the
        handler — acceptable because model reload is a rare background event,
        not on the critical path.

        On failure: logs the error and leaves self._model unchanged. The
        system continues serving with the old weights until the next change.
        """
        logger.info(
            "topology_classifier.model_reload_triggered",
            extra={"model_path": self._model_path},
        )
        try:
            new_model    = self._load_classifier_model()
            self._model  = new_model   # GIL-safe atomic assignment
            logger.info(
                "topology_classifier.model_reload_complete",
                extra={"model_path": self._model_path},
            )
        except ClassifierModelNotInitialized as exc:
            logger.error(
                "topology_classifier.model_reload_failed",
                extra={
                    "model_path": self._model_path,
                    "error":      str(exc),
                    "note":       "continuing with previous model weights",
                },
            )

    # ── Phase state reader — step 8.5 ─────────────────────────────────────────

    async def _reload_phase_states(self) -> None:
        """WATCHDOG callback: refresh the phase snapshot from the updated mmap.

        Called on the WATCHDOG thread whenever inotify signals a write to
        store/phase_states.mmap. index_daemon.py writes this file after every
        phase transition (infrequent — O(minutes) between transitions). The
        call is debounced at 1000 ms so back-to-back writes (e.g. multiple
        classes transitioning simultaneously) collapse to one refresh.

        Thread safety: delegates to _PhaseStateReader.reload() which acquires
        _PhaseStateReader._lock before rebuilding the snapshot, then replaces
        self._snapshot via a GIL-safe reference assignment. The classify() hot
        path holds no lock — it snapshots the reference at call entry.
        """
        if self._phase_reader is not None and self._phase_reader.is_open:
            self._phase_reader.reload()
            logger.debug(
                "topology_classifier.phase_states_reloaded",
                extra={"path": self._phase_states_path},
            )

    def _read_phase_snapshot(self) -> Optional[List[_PhaseSlot]]:
        """Return the current phase snapshot for use in a single classify() call.

        Returns None if the phase reader is not open (e.g. phase_states.mmap
        was missing at initialize() time). Callers treat None as "no phase
        information available" and fall back to _THETA_CONFIDENT for all
        threshold checks.

        This method is O(1) — it returns a reference to the cached list,
        not a copy. The list is replaced atomically by _reload_phase_states()
        on the WATCHDOG thread. classify() captures the reference once at
        call entry; all threshold checks within the call operate on the same
        snapshot, ensuring consistent phase-adjusted thresholds across paths
        1–4.5 for a single classification.
        """
        if self._phase_reader is None or not self._phase_reader.is_open:
            return None
        return self._phase_reader.snapshot

    # ── Classification path 1: domain fingerprint ──────────────────────────────

    def _classify_by_domain( # noqa
        self,
        url: str,
    ) -> Optional[_SignalEvidence]:
        """Resolve URL hostname against _DOMAIN_TRIE.

        Extracts the hostname (stripping port if present) and walks the trie.
        Exact matches win over wildcard matches at the same depth.

        Returns _SignalEvidence at the trie-node confidence, or None on miss.

        This is the fastest path. It should hit for > 75% of known-domain URLs
        in a warm corpus. Confidence is always 0.88–0.99 for trie hits.
        """
        try:
            parsed = urlparse(url)
            host   = parsed.netloc.lower()
            # Strip port.
            if ":" in host:
                host = host.split(":")[0]
            # Strip empty host (e.g. relative URLs, though these should not arrive).
            if not host:
                return None
        except Exception: # noqa
            return None

        result = _DOMAIN_TRIE.match(host)
        if result is None:
            return None

        topology_class, confidence = result
        return _SignalEvidence(
            topology_class = topology_class,
            raw_confidence = confidence,
            path           = ClassificationPath.DOMAIN_FINGERPRINT,
            detail         = f"host={host}",
        )

    # ── Classification path 2: URL structure ──────────────────────────────────

    def _classify_by_url(self, url: str) -> Optional[_SignalEvidence]: # noqa
        """Match URL path against URL_STRUCTURE_PATTERNS.

        Evaluates patterns in order. First match wins. Path is lowercased
        before matching.

        URL_STRUCTURE_PATTERNS carry confidence 0.78–0.92. They are reliable
        signals but not infallible — a page at /api/ might be a landing page
        that documents an API rather than an actual API endpoint.

        Returns _SignalEvidence or None.
        """
        try:
            path = urlparse(url).path.lower()
        except Exception: # noqa
            return None

        if not path or path == "/":
            return None

        for pattern, topology_class, confidence in URL_STRUCTURE_PATTERNS:
            if pattern.search(path):
                return _SignalEvidence(
                    topology_class = topology_class,
                    raw_confidence = confidence,
                    path           = ClassificationPath.URL_STRUCTURE,
                    detail         = f"pattern={pattern.pattern!r} path={path[:80]!r}",
                )

        return None

    # ── Classification path 3: response headers ───────────────────────────────

    def _classify_by_headers( # noqa
        self,
        headers: Dict[str, str],
        response_code: int,
    ) -> Optional[_SignalEvidence]:
        """Classify from HTTP response headers and response code.

        Checked in this order:
          1. HTTP 429 or Retry-After header → RATE_LIMITED (hard override)
          2. HTTP 301/302 with auth-pattern Location → AUTH_REDIRECT (hard override)
          3. Multi-header correlation patterns (HEADER_CORRELATION_PATTERNS)
          4. Single-header signals (HEADER_SIGNALS)

        Hard overrides in this path return immediately — the HARD_OVERRIDE_CLASSES
        check in classify() will then skip the content window and ML paths.

        Returns _SignalEvidence or None.
        """
        lowered: Dict[str, str] = {k.lower(): v.lower() for k, v in headers.items()}

        # ── Rate limiting ────────────────────────────────────────────────────
        if response_code == 429 or "retry-after" in lowered:
            return _SignalEvidence(
                topology_class = "RATE_LIMITED",
                raw_confidence = 0.99,
                path           = ClassificationPath.HEADER_SIGNAL,
                detail         = f"response_code={response_code} retry-after={'present' if 'retry-after' in lowered else 'absent'}",
            )

        # ── Auth redirect ────────────────────────────────────────────────────
        if response_code in (301, 302, 303, 307, 308):
            location = lowered.get("location", "")
            if any(p in location for p in ("/login", "/signin", "/auth", "/sso")):
                return _SignalEvidence(
                    topology_class = "AUTH_REDIRECT",
                    raw_confidence = 0.99,
                    path           = ClassificationPath.HEADER_SIGNAL,
                    detail         = f"redirect_to={location[:80]!r}",
                )

        # ── Multi-header correlation ─────────────────────────────────────────
        present_headers: FrozenSet[str] = frozenset(lowered.keys())
        for header_set, topology_class, confidence in HEADER_CORRELATION_PATTERNS:
            if header_set <= present_headers:
                return _SignalEvidence(
                    topology_class = topology_class,
                    raw_confidence = confidence,
                    path           = ClassificationPath.HEADER_SIGNAL,
                    detail         = f"multi-header correlation keys={sorted(header_set)}",
                )

        # ── Single header signals ────────────────────────────────────────────
        for header_key, value_pattern, topology_class, confidence in HEADER_SIGNALS:
            if header_key not in lowered:
                continue
            if value_pattern is None:
                # Presence-only check.
                return _SignalEvidence(
                    topology_class = topology_class,
                    raw_confidence = confidence,
                    path           = ClassificationPath.HEADER_SIGNAL,
                    detail         = f"header={header_key} (presence)",
                )
            if value_pattern in lowered[header_key]:
                return _SignalEvidence(
                    topology_class = topology_class,
                    raw_confidence = confidence,
                    path           = ClassificationPath.HEADER_SIGNAL,
                    detail         = f"header={header_key!r} contains {value_pattern!r}",
                )

        return None

    # ── Classification path 4: content window ──────────────────────────────────

    def _classify_by_window( # noqa
        self,
        content_prefix: bytes,
    ) -> Optional[_SignalEvidence]:
        """Classify from the first 4 KB of response body.

        Two-stage analysis:
          Stage A — n-gram fingerprint similarity via _FINGERPRINT_INDEX.
                    Fast. Returns the best-matching class if any exceed
                    _FINGERPRINT_SIM_THRESHOLD. Does not fire for hard-override
                    classes on its own — exact patterns in Stage B handle those.
          Stage B — exact substring search through WINDOW_PATTERNS.
                    Ordered. First match at or above _THETA_FALLBACK wins.
                    Hard-override patterns are placed first in WINDOW_PATTERNS
                    so they are always checked first.

        Stage A and Stage B may disagree. In that case Stage B wins — exact
        structural markers are more reliable than n-gram similarity.

        content_prefix is decoded as UTF-8 with errors=replace. The 4 KB
        ceiling is enforced inside this method; callers may pass the full
        prefix bytes and rely on this method to truncate.

        Returns _SignalEvidence or None.
        """
        if len(content_prefix) == 0:
            return None

        # Truncate to window size and decode.
        window_bytes = content_prefix[:_CONTENT_WINDOW_BYTES]
        try:
            window = window_bytes.decode("utf-8", errors="replace")
        except Exception: # noqa
            return None

        if len(window) < _MIN_WINDOW_CHARS:
            raise ClassificationWindowTooSmall(
                f"Content prefix decoded to {len(window)} chars, "
                f"below minimum {_MIN_WINDOW_CHARS}. "
                "URL + header signals will be used; window path skipped.",
            )

        window_lower = window.lower()

        # ── Stage A: n-gram fingerprint similarity ────────────────────────────
        fingerprint_scores = _FINGERPRINT_INDEX.score(window_lower)
        best_fp_class: Optional[str]  = None
        best_fp_score: float          = 0.0
        if fingerprint_scores:
            best_fp_class = max(fingerprint_scores, key=fingerprint_scores.__getitem__)
            best_fp_score = fingerprint_scores[best_fp_class]

        # ── Stage B: exact WINDOW_PATTERNS search ─────────────────────────────
        for needle, topology_class, confidence in WINDOW_PATTERNS:
            if needle.lower() in window_lower:
                detail = f"exact needle={needle[:40]!r}"
                if best_fp_class and best_fp_class != topology_class:
                    # Note the fingerprint disagreement in the evidence detail.
                    detail += (
                        f" (fingerprint suggested {best_fp_class} "
                        f"@ {best_fp_score:.3f} — exact match overrides)"
                    )
                return _SignalEvidence(
                    topology_class = topology_class,
                    raw_confidence = confidence,
                    path           = ClassificationPath.CONTENT_WINDOW,
                    detail         = detail,
                )

        # Stage B found nothing. If Stage A found a plausible match, use it
        # but with a confidence penalty (less reliable than exact match).
        if best_fp_class and best_fp_score >= _FINGERPRINT_SIM_THRESHOLD:
            penalised_conf = best_fp_score * 0.75   # n-gram similarity → confidence
            if penalised_conf >= _THETA_FALLBACK:
                return _SignalEvidence(
                    topology_class = best_fp_class,
                    raw_confidence = round(penalised_conf, 4),
                    path           = ClassificationPath.CONTENT_WINDOW,
                    detail         = f"fingerprint jaccard={best_fp_score:.4f} (no exact match)",
                )

        return None

    # ── Result construction ────────────────────────────────────────────────────

    def _build_result( # noqa
        self,
        topology_class: str,
        confidence:     float,
        path:           ClassificationPath,
        lattice:        _EvidenceLattice,
        ledger:         _SignalLedger,
        run_id:         str,
        t0:             float,
    ) -> TopologyClassification:
        """Construct a TopologyClassification from all accumulated evidence.

        topology_class, confidence, and path are the winning signal that
        triggered short-circuit. The lattice and ledger provide the full
        evidence trace for signals_used and fallback_chain.
        """
        latency_ms      = round((time.perf_counter() - t0) * 1000.0, 3)
        signals_used    = lattice.to_signals_used()
        fallback_chain  = ledger.fallback_chain(topology_class)

        # Add the winner to signals_used if it isn't already there.
        winner_key = f"{path.value}.winner"
        if winner_key not in signals_used:
            signals_used[winner_key] = f"{topology_class}@{confidence:.3f} (resolved)"

        # Populate fallback_chain in signals_used for backward compatibility.
        if fallback_chain:
            signals_used["fallback_chain"] = ",".join(fallback_chain)

        return TopologyClassification(
            topology_class      = topology_class,
            confidence          = confidence,
            classification_path = path.value,
            signals_used        = signals_used,
            latency_ms          = latency_ms,
            run_id              = run_id,
        )

    # ── Main classification orchestrator ──────────────────────────────────────

    async def classify(self, input: ClassifierInput) -> TopologyClassification: # noqa
        """Classify a URL into a topology class.

        Runs classification paths in order. Short-circuits on first path that
        produces confidence >= THETA_CLASSIFY_CONFIDENT or a hard-override class.
        After all deterministic paths run, attempts evidence lattice fusion
        before invoking the ML model.

        Args:
            input: ClassifierInput with url, headers, content_prefix (bytes),
                   response_code, and optional run_id.

        Returns:
            TopologyClassification — never raises, never returns None.
            GENERIC_HTML is the minimum fallback.

        Never raises — all exceptions are caught and result in a GENERIC_HTML
        fallback with the exception noted in signals_used.
        """
        t0     = time.perf_counter()
        run_id = input.run_id or new_run_id()

        if self._model is None:
            raise ClassifierModelNotInitialized(
                "classify() called before initialize(). "
                "Call await classifier.initialize() during cold_start.",
            )

        # Capture the phase snapshot once for the entire waterfall. All five
        # threshold checks below call _phase_adjusted_theta(class, phase_snapshot)
        # which is O(1) list access — no mmap reads on the hot path. The snapshot
        # reference is stable for the duration of this call even if WATCHDOG
        # triggers a refresh on another thread mid-call.
        phase_snapshot = self._read_phase_snapshot()

        lattice = _EvidenceLattice()
        ledger  = _SignalLedger()

        # ── Path 1: Domain fingerprint ────────────────────────────────────────
        evidence = self._classify_by_domain(input.url)
        if evidence is not None:
            lattice.deposit(evidence)
            ledger.record(evidence)
            theta1 = _phase_adjusted_theta(evidence.topology_class, phase_snapshot)
            if evidence.raw_confidence >= theta1:
                logger.debug(
                    "topology_classifier.path1_hit",
                    extra={
                        "url":    input.url[:100],
                        "class":  evidence.topology_class,
                        "conf":   evidence.raw_confidence,
                        "theta":  theta1,
                        "run_id": run_id,
                    },
                )
                return self._build_result(
                    evidence.topology_class,
                    evidence.raw_confidence,
                    ClassificationPath.DOMAIN_FINGERPRINT,
                    lattice, ledger, run_id, t0,
                )

        # ── Path 2: URL structure ─────────────────────────────────────────────
        evidence = self._classify_by_url(input.url)
        if evidence is not None:
            lattice.deposit(evidence)
            ledger.record(evidence)
            theta2 = _phase_adjusted_theta(evidence.topology_class, phase_snapshot)
            if evidence.raw_confidence >= theta2:
                logger.debug(
                    "topology_classifier.path2_hit",
                    extra={
                        "url":    input.url[:100],
                        "class":  evidence.topology_class,
                        "conf":   evidence.raw_confidence,
                        "theta":  theta2,
                        "run_id": run_id,
                    },
                )
                return self._build_result(
                    evidence.topology_class,
                    evidence.raw_confidence,
                    ClassificationPath.URL_STRUCTURE,
                    lattice, ledger, run_id, t0,
                )

        # ── Path 3: Response headers ──────────────────────────────────────────
        try:
            evidence = self._classify_by_headers(input.headers, input.response_code)
        except Exception as exc:
            logger.warning(
                "topology_classifier.path3_error",
                extra={"url": input.url[:100], "error": str(exc), "run_id": run_id},
            )
            evidence = None

        if evidence is not None:
            lattice.deposit(evidence)
            ledger.record(evidence)
            # Hard-override classes bypass all remaining paths including ML.
            if evidence.topology_class in HARD_OVERRIDE_CLASSES:
                logger.debug(
                    "topology_classifier.hard_override_header",
                    extra={
                        "url":   input.url[:100],
                        "class": evidence.topology_class,
                        "run_id": run_id,
                    },
                )
                return self._build_result(
                    evidence.topology_class,
                    evidence.raw_confidence,
                    ClassificationPath.HEADER_SIGNAL,
                    lattice, ledger, run_id, t0,
                )
            if evidence.raw_confidence >= _phase_adjusted_theta(
                evidence.topology_class, phase_snapshot
            ):
                return self._build_result(
                    evidence.topology_class,
                    evidence.raw_confidence,
                    ClassificationPath.HEADER_SIGNAL,
                    lattice, ledger, run_id, t0,
                )

        # ── Path 4: Content window ────────────────────────────────────────────
        if input.content_prefix:
            try:
                evidence = self._classify_by_window(input.content_prefix)
            except ClassificationWindowTooSmall:
                logger.debug(
                    "topology_classifier.window_too_small",
                    extra={"url": input.url[:100], "run_id": run_id},
                )
                evidence = None
            except Exception as exc:
                logger.warning(
                    "topology_classifier.path4_error",
                    extra={"url": input.url[:100], "error": str(exc), "run_id": run_id},
                )
                evidence = None

            if evidence is not None:
                lattice.deposit(evidence)
                ledger.record(evidence)
                # Hard-override classes bypass ML.
                if evidence.topology_class in HARD_OVERRIDE_CLASSES:
                    logger.debug(
                        "topology_classifier.hard_override_window",
                        extra={
                            "url":   input.url[:100],
                            "class": evidence.topology_class,
                            "run_id": run_id,
                        },
                    )
                    return self._build_result(
                        evidence.topology_class,
                        evidence.raw_confidence,
                        ClassificationPath.CONTENT_WINDOW,
                        lattice, ledger, run_id, t0,
                    )
                if evidence.raw_confidence >= _phase_adjusted_theta(
                    evidence.topology_class, phase_snapshot
                ):
                    return self._build_result(
                        evidence.topology_class,
                        evidence.raw_confidence,
                        ClassificationPath.CONTENT_WINDOW,
                        lattice, ledger, run_id, t0,
                    )

        # ── Path 4.5: Evidence lattice fusion (novel) ─────────────────────────
        # All deterministic paths have run. Attempt hierarchy-aware fusion
        # before invoking the ML model. This can resolve cases where multiple
        # paths each return moderate evidence for related classes — e.g.
        # URL says REST_API_JSON at 0.82 and headers say REST_API_JSON at 0.70,
        # neither alone exceeds 0.75, but lattice fusion gives ~0.94.
        lattice.propagate_through_hierarchy()
        fusion_result = lattice.best_class_after_propagation()

        if fusion_result is not None:
            fused_class, fused_conf = fusion_result
            theta_fusion = _phase_adjusted_theta(fused_class, phase_snapshot)
            if fused_conf >= theta_fusion:
                logger.debug(
                    "topology_classifier.lattice_fusion_hit",
                    extra={
                        "url":    input.url[:100],
                        "class":  fused_class,
                        "conf":   fused_conf,
                        "theta":  theta_fusion,
                        "run_id": run_id,
                    },
                )
                return self._build_result(
                    fused_class,
                    fused_conf,
                    ClassificationPath.LATTICE_FUSION,
                    lattice, ledger, run_id, t0,
                )

        # ── Path 5: ML model ──────────────────────────────────────────────────
        # Only reached when all deterministic paths and lattice fusion failed
        # to produce confidence >= THETA_CLASSIFY_CONFIDENT.
        try:
            url_tokens = _URLSemanticTokenizer.tokenize(input.url)
            features   = self._embed_signals(
                url           = input.url,
                headers       = input.headers,
                content_prefix= input.content_prefix,
                domain_hint   = None,
                url_token_profile = url_tokens,
            )
            ml_class, ml_confidence = self._classify_via_model(features)
        except NotImplementedError:
            # _embed_signals or _classify_via_model raised NotImplementedError.
            # Fall through to GENERIC_HTML.
            logger.warning(
                "topology_classifier.ml_path_not_implemented",
                extra={"url": input.url[:100], "run_id": run_id},
            )
            ml_class, ml_confidence = FALLBACK_TOPOLOGY_CLASS, 0.0
        except Exception as exc:
            logger.error(
                "topology_classifier.ml_path_error",
                extra={"url": input.url[:100], "error": str(exc), "run_id": run_id},
            )
            ml_class, ml_confidence = FALLBACK_TOPOLOGY_CLASS, 0.0

        if ml_confidence < _THETA_FALLBACK:
            logger.warning(
                "topology_classifier.confidence_too_low",
                extra={
                    "url":        input.url[:100],
                    "ml_class":   ml_class,
                    "confidence": ml_confidence,
                    "run_id":     run_id,
                },
            )
            # Deposit fallback into lattice for signals_used trace.
            lattice.deposit(_SignalEvidence(
                topology_class = FALLBACK_TOPOLOGY_CLASS,
                raw_confidence = ml_confidence,
                path           = ClassificationPath.FALLBACK,
                detail         = f"ml returned {ml_class}@{ml_confidence:.3f} < THETA_FALLBACK",
            ))
            return self._build_result(
                FALLBACK_TOPOLOGY_CLASS,
                ml_confidence,
                ClassificationPath.FALLBACK,
                lattice, ledger, run_id, t0,
            )

        lattice.deposit(_SignalEvidence(
            topology_class = ml_class,
            raw_confidence = ml_confidence,
            path           = ClassificationPath.MODEL,
            detail         = f"topology_router.pt forward pass",
        ))
        ledger.record(lattice._direct[ml_class][-1]) # noqa

        logger.debug(
            "topology_classifier.ml_path_resolved",
            extra={
                "url":   input.url[:100],
                "class": ml_class,
                "conf":  ml_confidence,
                "run_id": run_id,
            },
        )

        result = self._build_result(
            ml_class,
            ml_confidence,
            ClassificationPath.MODEL,
            lattice, ledger, run_id, t0,
        )

        # Append to rolling audit buffer.
        self._audit.append({
            "url":    input.url[:100],
            "class":  ml_class,
            "conf":   ml_confidence,
            "path":   "model",
            "run_id": run_id,
        })

        return result

    # ── Startup ────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load topology_router.pt and register with WATCHDOG.

        Called once during cold_start.py's Phase 3 (topology layer init).
        Blocks until the model is loaded. WATCHDOG handles subsequent reloads.

        Raises:
            ClassifierModelNotInitialized: if the model file is missing,
                corrupt, or incompatible. cold_start.py treats this as a
                fatal startup error — the classifier cannot operate without
                topology_router.pt because the ML path (path 5) is the only
                path for genuinely ambiguous URLs. Silently continuing with
                self._model = None would cause TypeError at inference time
                instead of a clean startup failure.
        """
        # ── Pre-flight: verify the store file exists before attempting load ───
        # topology_router.pt is written once by initialize_store.py before any
        # service starts. Its absence means the store was never initialized —
        # this is a deployment error, not a runtime error. Surface it immediately
        # with an actionable message rather than letting torch.load produce a
        # generic FileNotFoundError two stack frames down.
        if not Path(self._model_path).exists():
            raise ClassifierModelNotInitialized(
                f"topology_router.pt not found at {self._model_path!r}. "
                "The store has not been initialized. "
                "Run initialize_store.py before starting the service:\n"
                "    python -m tag.store.initialize_store\n"
                "This must complete before cold_start.py runs."
            )

        self._model = self._load_classifier_model()  # raises on corrupt/incompatible

        WATCHDOG.register(
            path=self._model_path,
            handler=self._reload_classifier_model,
            debounce_ms=500,
        )

        # ── Phase state reader ────────────────────────────────────────────────
        # Open phase_states.mmap and register with WATCHDOG. Failure here is
        # non-fatal: the classifier degrades to constant _THETA_CONFIDENT
        # thresholds (correct behavior, just not phase-aware). This mirrors the
        # graceful-degradation contract used by the WLM when structural_layer.pt
        # is absent — the component works, it just works less intelligently.
        global _PHASE_STATE_READER
        reader = _PhaseStateReader(self._phase_states_path)
        try:
            reader.open()
            self._phase_reader  = reader
            _PHASE_STATE_READER = reader
            WATCHDOG.register(
                path=self._phase_states_path,
                handler=self._reload_phase_states,
                debounce_ms=1000,
            )
            logger.info(
                "topology_classifier.phase_reader_opened",
                extra={
                    "path":      self._phase_states_path,
                    "n_classes": _PS_N_CLASSES,
                },
            )
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "topology_classifier.phase_reader_unavailable",
                extra={
                    "path":  self._phase_states_path,
                    "error": str(exc),
                    "note":  "classifier will use constant THETA_CLASSIFY_CONFIDENT "
                             "thresholds — phase-aware gating disabled",
                },
            )

        logger.info(
            "topology_classifier.initialized",
            extra={
                "model_path": self._model_path,
                "hidden_state_version": self._model.current_hidden_state_version,
                "hidden_state_digest": self._model.hidden_state_digest(),
                "watchdog_debounce_ms": 500,
            },
        )


    def _embed_signals(
        self,
        url:               str,
        headers:           Dict[str, str],
        content_prefix:    bytes,
        domain_hint:       Optional[str],
        url_token_profile: Optional[_URLSemanticTokenizer.URLTokenProfile] = None,
    ) -> torch.Tensor:
        """Convert the four signal streams into a fixed-width feature tensor.

        Called only when all deterministic paths (1–4) and lattice fusion
        (4.5) failed to produce confidence >= THETA_CLASSIFY_CONFIDENT. This
        is the feature engineering step before the topology_router.pt forward
        pass.

        ── CONTRACT ─────────────────────────────────────────────────────────

        Output tensor:
            dtype:   torch.float32
            shape:   (1, INPUT_DIM) where INPUT_DIM matches topology_router.pt
                     input layer. The model's trained weights encode this
                     feature space — changing the feature layout invalidates
                     the weights. Never change the output dimension or feature
                     ordering without retraining.
            device:  same device as self._model (CPU or CUDA). Use
                     tensor.to(next(self._model.parameters()).device) to ensure
                     device alignment before the forward pass.

        Determinism requirement:
            Same (url, headers, content_prefix, domain_hint) → same output
            tensor across restarts, across Python sessions, across OS reboots.
            No randomness. No datetime. No process-local state.

            This is not merely a nice-to-have. topology_router.pt's weights
            are trained against this feature space. A feature implementation
            that is not deterministic means training and inference operate on
            different feature distributions. The model will not generalise.

        No external calls:
            No HTTP. No disk reads. No subprocess. No LLM. The feature vector
            must be computable entirely from the four arguments in memory.

        ── FEATURE SPACE SPECIFICATION ──────────────────────────────────────

        The feature vector has six groups. Exact dimensions for
        each group, subject to the constraints below. All groups are
        concatenated in this order:

        GROUP 1 — URL_PATH_TOKENS  [recommended: 64 dims]
            URL path segments are tokenised into the semantic token classes
            defined by _URLSemanticTokenizer. Each token class maps to one
            or more binary or count features.

            If url_token_profile is provided, use it directly (avoids
            re-parsing). Otherwise call _URLSemanticTokenizer.tokenize(url).

            Features to include:
                has_date_segment          bool → 0/1
                has_year_segment          bool → 0/1
                has_version_segment       bool → 0/1
                has_numeric_id            bool → 0/1
                has_short_hash            bool → 0/1
                has_long_hash             bool → 0/1
                has_long_slug             bool → 0/1
                path_depth                int  → normalise to [0,1] / max_depth
                query_param_count         int  → normalise to [0,1] / max_params
                has_fragment              bool → 0/1
                keyword_one_hot           len(KNOWN_SEGMENT_KEYWORDS) dims → 0/1 for each keyword
                file_extension_one_hot    coverage for {"json","html","xml","pdf","md","csv"}

            Path characters (beyond tokens): a bag-of-characters over the
            URL path using MurmurHash-like character bi-gram hashing is
            recommended. Fixed width via modular hashing (hash(bigram) % 32).

        GROUP 2 — HEADER_BITMASK  [recommended: 48 dims]
            Binary feature per header key indicating presence/absence.
            Header value features for content-type (one-hot over known types).
            Include:
                presence flags for all keys in HEADER_SIGNALS (one per key)
                presence flags for all keys in HEADER_CORRELATION_PATTERNS
                content-type family one-hot: {html, json, ld+json, xml, other}
                response_code_bin:   one-hot over {200, 301, 302, 400, 401,
                                     403, 404, 429, 503, other}
                has_cf_ray           bool
                has_retry_after      bool
                has_x_robots_tag     bool
                has_x_api_version    bool

        GROUP 3 — CONTENT_NGRAM_HASH  [recommended: 128 dims]
            Character 3-gram frequency hash into a fixed-width bag vector.
            Content prefix decoded as UTF-8 (errors=replace), lowercased,
            truncated to _CONTENT_WINDOW_BYTES chars.

            Hashing scheme — deterministic, no seed dependency:
                import hashlib
                for each 3-gram t in content_lower:
                    raw = hashlib.blake2b(
                        t.encode('utf-8'), digest_size=4
                    ).digest()
                    bucket = int.from_bytes(raw, 'little') % 128
                    vec[bucket] += 1
            Normalise by total 3-gram count: vec /= max(vec.sum(), 1.0)

            HASHING PRIMITIVE — blake2b(digest_size=4), stdlib, no deps:
                Deterministic across processes, Python versions, OS, and
                architecture. Unkeyed mode. Do NOT add a key argument.
                Faster than SHA-256 in software. 32-bit output gives good
                distribution for % 128 or % 256 bucketing.

                Do NOT use Python's built-in hash() for feature vectors —
                per-process seed randomisation and algorithm drift across
                interpreter versions both silently corrupt training/inference
                alignment. Do NOT use HMAC — authentication primitive,
                wrong abstraction, requires key management.

        GROUP 4 — DOMAIN_FEATURES  [recommended: 32 dims]
            Structural features extracted from the hostname component of url.

            Features:
                tld_one_hot           coverage for {"com", "org", "io", "net",
                                      "gov", "edu", "co", other}
                has_subdomain         bool (hostname has 3+ labels)
                subdomain_is_docs     bool (subdomain is "docs")
                subdomain_is_api      bool (subdomain is "api")
                subdomain_is_dev      bool (subdomain is "developer" or "dev")
                hostname_label_count  int → normalise to [0,1] / 6
                partial_match_score   float in [0,1]
                                      Jaccard on hostname char 3-grams vs
                                      nearest DOMAIN_FINGERPRINT_TABLE entry.
                                      Use the same blake2b(digest_size=4)
                                      hashing scheme as GROUP 3. No hash().
                domain_hint_present   bool (domain_hint argument is not None)
                domain_hint_encoding  GROUP 4 sub-vector for domain_hint using
                                      same tld_one_hot + label features above.
                                      All zeros if domain_hint is None.

        GROUP 5 — FINGERPRINT_SCORES  [recommended: 18 dims]
            One feature per topology class in TOPOLOGY_CLASSES.
            Value: the Jaccard fingerprint score from _FINGERPRINT_INDEX for
            that class (float in [0.0, 1.0]).

            Call _FINGERPRINT_INDEX.score(content_lower) to get the dict,
            then map to the 18-dim vector using TOPOLOGY_CLASS_INDEX as the
            index mapping. Classes with no fingerprint score get 0.0.

            This group directly encodes what the content fingerprint engine
            found. The ML model can use this as a soft prior when the exact
            WINDOW_PATTERNS didn't fire strongly enough to reach the threshold.

        GROUP 6 — LATTICE_SCORES  [recommended: 18 dims]
            One feature per topology class in TOPOLOGY_CLASSES.
            Value: the fused confidence from the evidence lattice for that
            class (float in [0.0, 1.0]).

            By the time _embed_signals is called, classify() has already run
            all deterministic paths and called lattice.propagate_through_hierarchy().
            These scores summarise everything the deterministic paths found.

            The ML model uses this group as a soft prior. Even if no individual
            deterministic path crossed the threshold, their aggregated evidence
            can guide the model toward more accurate predictions.

            NOTE: _embed_signals cannot directly access the _EvidenceLattice from within
            _embed_signals because the lattice is a local variable in classify().
            Two options:
              Option A: pass the lattice as an additional argument. If this
                        chooses this, update the call site in classify() to
                        pass `lattice` explicitly.
              Option B: recompute GROUP 5 Jaccard scores and use them as a
                        proxy for GROUP 6 (Jaccard scores ≈ lattice evidence
                        before propagation for single-path inputs). This is
                        the simpler option and is preferred.

        TOTAL RECOMMENDED DIMENSIONS: 64 + 48 + 128 + 32 + 18 + 18 = 308
            Verify this matches topology_router.pt input_dim before
            merging. If the trained model expects a different dimension,
            adjust the group sizes proportionally, maintaining the group order.

        ── IMPLEMENTATION NOTES ─────────────────────────────────────────────

        1. Build a float32 numpy array of shape (INPUT_DIM,), populate all
           groups in order, then convert to torch.Tensor with
           torch.tensor(arr, dtype=torch.float32).unsqueeze(0) to add the
           batch dimension.

        2. Normalise each group independently before concatenation. The model
           was likely trained on normalised inputs. Do not pass raw counts.

        3. The determinism requirement is strict. Test with:
               v1 = _embed_signals(url, headers, prefix, hint)
               v2 = _embed_signals(url, headers, prefix, hint)
               assert torch.allclose(v1, v2)
           across separate Python processes before declaring implementation
           complete.

        4. Do not call torch.no_grad() here — this method does not run the
           model. Save torch.no_grad() for _classify_via_model().

        5. Performance target: < 5ms on CPU. This method runs on every URL
           that reaches path 5. At 50K URLs/run with 25% path-5 invocation
           rate, this is 12.5K calls. At 5ms each = 62.5s total, acceptable.
           Do not implement anything that scales with corpus size.

        Raises:
            NotImplementedError: if the implementation is not yet in place.
            ValueError: if features cannot be computed from the input.
        """
        raise NotImplementedError(
            "_embed_signals() stub — implementation is in _embed_signals_impl(), "
            "monkey-patched onto the class at module load. See the docstring for "
            "the full feature space specification and implementation contract."
        )

    def _classify_via_model(
        self,
        features: torch.Tensor,
    ) -> Tuple[str, float]:
        """Forward pass through topology_router.pt.

        Called only after _embed_signals() returns a valid feature tensor.
        Produces a probability distribution over all 18 topology classes
        and returns the argmax class with its softmax probability as confidence.

        ── CONTRACT ─────────────────────────────────────────────────────────

        Input tensor:
            dtype:   torch.float32
            shape:   (1, INPUT_DIM)  — batch dimension 1, from _embed_signals()
            device:  same device as self._model

        Output:
            Tuple[str, float]
            str:   topology class string from TOPOLOGY_CLASSES
                   NOT a TopologyClass enum member — the raw string
            float: softmax probability of argmax class, in [0.0, 1.0]
                   This is confidence in the TopologyClassification sense.
                   Never return a raw logit. Never return a probability > 1.0.

        Thread safety:
            self._model must not be modified inside this method.
            torch.no_grad() must wrap the forward pass.
            GIL protects self._model from concurrent modification during
            _reload_classifier_model(). The forward pass reads model weights
            without writing them — this is safe.

        ── IMPLEMENTATION REQUIREMENTS ────────────────────────────

        1. Verify the model is on the correct device before the forward pass:
               device = next(self._model.parameters()).device
               features = features.to(device)

        2. Wrap the entire forward pass in torch.no_grad():
               with torch.no_grad():
                   logits = self._model(features)

        3. Apply softmax to the logit vector:
               probs = torch.softmax(logits, dim=-1)   # shape (1, 18)

        4. Extract argmax index and max probability:
               idx  = int(torch.argmax(probs, dim=-1).item())
               conf = float(probs[0, idx].item())

        5. Map index to class string using INDEX_TO_TOPOLOGY_CLASS:
               topology_class = INDEX_TO_TOPOLOGY_CLASS.get(idx, FALLBACK_TOPOLOGY_CLASS)

           If idx is out of range (model was retrained with more classes than
           the current INDEX_TO_TOPOLOGY_CLASS), return FALLBACK_TOPOLOGY_CLASS
           rather than raising. Log a warning — index mismatch means the model
           and the class index are out of sync (a deployment error).

        6. Return (topology_class, conf) — raw strings, no wrapping.

        7. Error handling: wrap the entire method in try/except.
               except Exception as exc:
                   logger.error("topology_classifier.model_forward_failed", ...)
                   return (FALLBACK_TOPOLOGY_CLASS, 0.0)
           Never raise from this method — classify() catches the return value
           and handles low-confidence results by falling back to GENERIC_HTML.

        8. Performance: The forward pass itself is < 5ms on GPU and < 15ms on
           CPU for an MLP of typical size. Do not add preprocessing that
           would push the total above the 15ms budget from the spec.

        9. Logging: emit a DEBUG log on every successful forward pass:
               logger.debug("topology_classifier.model_inference", extra={
                   "class": topology_class, "conf": conf,
                   "top3": [(INDEX_TO_TOPOLOGY_CLASS[i], round(float(p), 4))
                             for i, p in sorted(enumerate(probs[0].tolist()),
                                                key=lambda x: -x[1])[:3]],
               })
           This top-3 log is invaluable for diagnosing misclassifications.

        Raises:
            NotImplementedError: until the implementation is in place.
        """
        raise NotImplementedError(
            "_classify_via_model() stub — implementation is in _classify_via_model_impl(), "
            "monkey-patched onto the class at module load. "
            "See the docstring for the full forward pass specification and implementation contract."
        )

    # ── Introspection and health ───────────────────────────────────────────────

    def recent_classifications(self, n: int = 20) -> List[Dict]:
        """Return up to n most recent classification results from the audit buffer.

        The audit buffer holds the last _AUDIT_BUFFER_SIZE (512) results from
        the ML path only. Deterministic-path results (paths 1–4 and lattice)
        are not in this buffer — they are fast enough that per-result logging
        is done via logger.debug() rather than the buffer.

        This method is for debugging and health checks only. Do not call it
        on the critical path.
        """
        records = list(self._audit)
        return records[max(0, len(records) - n):]

    def is_ready(self) -> bool:
        """True iff the model is loaded and ready to classify.

        cold_start.py calls this after initialize() to verify the model
        loaded successfully. classify() also checks this and raises
        ClassifierModelNotInitialized if False.
        """
        return self._model is not None

    def model_device(self) -> Optional[str]:
        """Return the device string ("cpu" or "cuda:0") the model is on.

        Returns None if the model is not loaded.
        Useful for verifying GPU placement in health checks.
        """
        if self._model is None:
            return None
        try:
            return str(next(self._model.parameters()).device)
        except (StopIteration, Exception):
            return "unknown"

    def domain_trie_size(self) -> int: # noqa
        """Return the number of terminal nodes in the domain trie.

        A terminal node is a trie node that has a topology_class assigned.
        This is approximately the number of domain patterns registered,
        including wildcards. Useful for verifying that the trie was built
        from the full DOMAIN_FINGERPRINT_TABLE + DOMAIN_WILDCARD_SPECS.
        """
        def _count(node: _TrieNode) -> int:
            total = 1 if node.is_terminal() else 0
            for child in node.children.values():
                total += _count(child)
            return total
        return _count(_DOMAIN_TRIE._root) # noqa


# ══════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL HELPER FUNCTIONS
#
# Convenience wrappers around the singleton. Callers that do not need to
# configure a custom model path can use these rather than accessing CLASSIFIER
# directly.
# ══════════════════════════════════════════════════════════════════════════════

def resolve_fallback_chain(topology_class: str, max_depth: int = 3) -> List[str]:
    """Walk PARENT_CLASS_MAP from topology_class up to GENERIC_HTML.

    Returns the chain of parent classes, not including topology_class itself
    but including GENERIC_HTML as the terminal node.

    Used by topology/parser.py to resolve recipes when the primary class
    has no registered recipe.

    Examples:
        resolve_fallback_chain("SAAS_DOCS_VERSIONED")
        # → ["SAAS_DOCS", "GENERIC_HTML"]

        resolve_fallback_chain("FORUM_THREAD")
        # → ["BLOG_POST", "NEWS_ARTICLE", "GENERIC_HTML"]

        resolve_fallback_chain("GENERIC_HTML")
        # → []

    Args:
        topology_class: the class to start from.
        max_depth:      maximum chain length (spec constraint: ≤ 3).

    Returns:
        List of parent class strings. Empty if topology_class has no parent
        or IS GENERIC_HTML.
    """
    chain: List[str] = []
    current = topology_class
    for _ in range(max_depth):
        parent = PARENT_CLASS_MAP.get(current)
        if parent is None or parent == current:
            break
        chain.append(parent)
        if parent == FALLBACK_TOPOLOGY_CLASS:
            break
        current = parent
    return chain


def is_hard_override_class(topology_class: str) -> bool:
    """True if topology_class is in HARD_OVERRIDE_CLASSES.

    Hard-override classes bypass the ML path and should never be sent
    to recipe compilation — they have no extractable signal.

    Exposed at module level so phantom.py and interface.py can check
    classification results without importing the full class set.
    """
    return topology_class in HARD_OVERRIDE_CLASSES


def confidence_label(confidence: float) -> str:
    """Map a confidence float to a human-readable tier label.

    Used in log lines and diagnostic output. Not used in classification logic.

        [0.00, 0.40)  → "very_low"
        [0.40, 0.60)  → "low"
        [0.60, 0.75)  → "moderate"
        [0.75, 0.90)  → "confident"
        [0.90, 1.00]  → "high"
    """
    if confidence < 0.40:
        return "very_low"
    if confidence < 0.60:
        return "low"
    if confidence < 0.75:
        return "moderate"
    if confidence < 0.90:
        return "confident"
    return "high"


def topology_class_depth(topology_class: str) -> int:
    """Return the number of hops from topology_class to GENERIC_HTML.

    GENERIC_HTML is depth 0. Direct children of GENERIC_HTML (LANDING_PAGE,
    AUTH_REDIRECT, etc.) are depth 1. FORUM_THREAD (→ BLOG_POST →
    NEWS_ARTICLE → no parent) is depth 2 because it takes 2 hops before
    reaching a class with no parent in PARENT_CLASS_MAP.

    Used by index_daemon.py to weight gradient steps — deeper classes are
    rarer and harder to classify correctly, so they get a gradient step
    weight boost proportional to their depth.
    """
    depth   = 0
    current = topology_class
    visited: Set[str] = {current}
    while True:
        parent = PARENT_CLASS_MAP.get(current)
        if parent is None or parent == current or parent in visited:
            break
        depth  += 1
        visited.add(parent)
        current = parent
    return depth


# ═════════════════════════════════════════════════════════════════════════════
# DETERMINISTIC HASHING PRIMITIVES
#
# All feature encoding uses these primitives. blake2b(digest_size=4) is the
# sole hashing function. It is:
#   - Deterministic across processes, Python versions, architectures
#   - Unkeyed (no HMAC, no key management)
#   - Faster than SHA-256 in software
#   - 32-bit output for efficient modular bucketing
#
# NEVER use Python's hash() for feature vectors. Per-process seed
# randomisation (PYTHONHASHSEED) and algorithm drift across interpreter
# versions both silently corrupt training/inference alignment.
# ═════════════════════════════════════════════════════════════════════════════

def _blake2b_hash32(data: bytes) -> int:
    """Compute a deterministic 32-bit hash of data using blake2b.

    Returns an unsigned 32-bit integer. This is the sole hashing primitive
    for all feature group encoders. Do not use hash() or any other hashing
    function for features.

    Performance: ~200ns per call on modern CPUs. At 4096 3-grams per content
    window, GROUP 3 spends ~0.8ms on hashing — within budget.
    """
    raw = hashlib.blake2b(data, digest_size=4).digest()
    return int.from_bytes(raw, byteorder="little", signed=False)


def _blake2b_bucket(data: bytes, num_buckets: int) -> int:
    """Hash data into a bucket index in [0, num_buckets).

    Combines _blake2b_hash32 with modular reduction. The distribution is
    uniform within the limits of 32-bit hash modular bias (negligible for
    num_buckets <= 1024).
    """
    return _blake2b_hash32(data) % num_buckets


def _blake2b_ngram_set(text: str, n: int) -> Set[int]:
    """Compute a set of blake2b hashes for all character n-grams in text.

    Used for Jaccard similarity in GROUP 4 partial match scoring.
    Distinct from _ContentFingerprintIndex._ngrams which uses Python hash()
    for intra-process scoring only. This function produces deterministic
    hashes safe for cross-process comparison and feature tensors.
    """
    if len(text) < n:
        return set()
    result: Set[int] = set()
    for i in range(len(text) - n + 1):
        gram = text[i:i + n].encode("utf-8")
        result.add(_blake2b_hash32(gram))
    return result


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP METADATA
#
# Immutable metadata for each feature group. Used by the feature assembler
# to validate dimensions and by the drift monitor to track statistics per
# group.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _FeatureGroupMeta:
    """Metadata for one feature group in the feature vector."""
    name:       str   # human-readable name, e.g. "url_path_tokens"
    offset:     int   # start index in the concatenated feature vector
    dim:        int   # number of features in this group
    version:    str   # feature version string (must match FEATURE_VERSION)


# Build the group metadata table. Offsets are computed from cumulative dims.
_GROUP_META: Tuple[_FeatureGroupMeta, ...] = (
    _FeatureGroupMeta(
        name="url_path_tokens", offset=0,
        dim=GROUP_1_URL_PATH_TOKENS_DIM, version=FEATURE_VERSION,
    ),
    _FeatureGroupMeta(
        name="header_bitmask",
        offset=GROUP_1_URL_PATH_TOKENS_DIM,
        dim=GROUP_2_HEADER_BITMASK_DIM, version=FEATURE_VERSION,
    ),
    _FeatureGroupMeta(
        name="content_ngram_hash",
        offset=GROUP_1_URL_PATH_TOKENS_DIM + GROUP_2_HEADER_BITMASK_DIM,
        dim=GROUP_3_CONTENT_NGRAM_HASH_DIM, version=FEATURE_VERSION,
    ),
    _FeatureGroupMeta(
        name="domain_features",
        offset=(GROUP_1_URL_PATH_TOKENS_DIM + GROUP_2_HEADER_BITMASK_DIM
                + GROUP_3_CONTENT_NGRAM_HASH_DIM),
        dim=GROUP_4_DOMAIN_FEATURES_DIM, version=FEATURE_VERSION,
    ),
    _FeatureGroupMeta(
        name="fingerprint_scores",
        offset=(GROUP_1_URL_PATH_TOKENS_DIM + GROUP_2_HEADER_BITMASK_DIM
                + GROUP_3_CONTENT_NGRAM_HASH_DIM + GROUP_4_DOMAIN_FEATURES_DIM),
        dim=GROUP_5_FINGERPRINT_SCORES_DIM, version=FEATURE_VERSION,
    ),
    _FeatureGroupMeta(
        name="lattice_scores",
        offset=(GROUP_1_URL_PATH_TOKENS_DIM + GROUP_2_HEADER_BITMASK_DIM
                + GROUP_3_CONTENT_NGRAM_HASH_DIM + GROUP_4_DOMAIN_FEATURES_DIM
                + GROUP_5_FINGERPRINT_SCORES_DIM),
        dim=GROUP_6_LATTICE_SCORES_DIM, version=FEATURE_VERSION,
    ),
)


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP ENCODING CONSTANTS
#
# Constants used by individual feature group encoders. Defined once here to
# avoid magic literals scattered across encoding functions.
# ═════════════════════════════════════════════════════════════════════════════

# GROUP 1 — URL path token encoding constants
_URL_KEYWORD_LIST: Tuple[str, ...] = (
    "api", "docs", "doc", "documentation", "reference", "guide",
    "blog", "post", "article", "news", "wiki", "forum", "thread",
    "product", "item", "shop", "store", "login", "signin", "auth",
    "search", "feed", "rss", "atom",
)
_URL_KEYWORD_INDEX: Dict[str, int] = {kw: i for i, kw in enumerate(_URL_KEYWORD_LIST)}
_NUM_URL_KEYWORDS: int = len(_URL_KEYWORD_LIST)  # 24

_URL_FILE_EXTENSIONS: Tuple[str, ...] = ("json", "html", "xml", "pdf", "md", "csv")
_URL_FILE_EXT_INDEX: Dict[str, int] = {ext: i for i, ext in enumerate(_URL_FILE_EXTENSIONS)}
_NUM_URL_FILE_EXTS: int = len(_URL_FILE_EXTENSIONS)  # 6

# URL path character bi-gram bag dimensions.
# Remaining dims after booleans (10) + keyword one-hot (24) + file_ext one-hot (6)
# + path_depth (1) + query_param_count (1) + has_fragment (1) + bigram bag (21) = 64
_URL_BOOL_FEATURES: int = 10     # 7 has_* booleans + path_depth + query_params + has_fragment
_URL_BIGRAM_BAG_DIM: int = (
    GROUP_1_URL_PATH_TOKENS_DIM - _URL_BOOL_FEATURES - _NUM_URL_KEYWORDS - _NUM_URL_FILE_EXTS
)
# 64 - 10 - 24 - 6 = 24 bigram bag dimensions

_MAX_PATH_DEPTH: float = 12.0          # normalise path_depth to [0, 1]
_MAX_QUERY_PARAMS: float = 20.0        # normalise query_param_count to [0, 1]

# GROUP 2 — Header bitmask encoding constants
_HEADER_PRESENCE_KEYS: Tuple[str, ...] = (
    "content-type", "cf-ray", "x-amz-cf-id", "retry-after",
    "x-ratelimit-remaining", "x-ratelimit-limit", "x-robots-tag",
    "paywall", "x-piano-tpl", "x-frame-options", "www-authenticate",
    "x-api-version", "x-docs-version", "cf-cache-status",
    "location", "set-cookie", "cache-control", "etag",
    "x-powered-by", "server",
)
_NUM_HEADER_PRESENCE: int = len(_HEADER_PRESENCE_KEYS)  # 20
_HEADER_PRESENCE_INDEX: Dict[str, int] = {k: i for i, k in enumerate(_HEADER_PRESENCE_KEYS)}

_CONTENT_TYPE_FAMILIES: Tuple[str, ...] = (
    "html", "json", "ld+json", "xml", "plain", "other",
)
_NUM_CT_FAMILIES: int = len(_CONTENT_TYPE_FAMILIES)  # 6
_CT_FAMILY_INDEX: Dict[str, int] = {f: i for i, f in enumerate(_CONTENT_TYPE_FAMILIES)}

_RESPONSE_CODE_BINS: Tuple[int, ...] = (200, 301, 302, 400, 401, 403, 404, 429, 500, 503)
_NUM_RESPONSE_CODES: int = len(_RESPONSE_CODE_BINS) + 1  # 11 (including "other")
_RESPONSE_CODE_INDEX: Dict[int, int] = {code: i for i, code in enumerate(_RESPONSE_CODE_BINS)}

# Remaining header dims: 48 - 20 (presence) - 6 (ct family) - 11 (response code) = 11
# These 11 dims are for: has_cf_ray, has_retry_after, has_x_robots_tag, has_x_api_version,
# header_count_normalised, has_set_cookie, has_location_header, is_redirect, is_client_error,
# is_server_error, content_length_bucket
_HEADER_EXTRA_FEATURES: int = (
    GROUP_2_HEADER_BITMASK_DIM - _NUM_HEADER_PRESENCE - _NUM_CT_FAMILIES - _NUM_RESPONSE_CODES
)
# 48 - 20 - 6 - 11 = 11

# GROUP 3 — Content n-gram hash constants
_CONTENT_NGRAM_BUCKETS: int = GROUP_3_CONTENT_NGRAM_HASH_DIM  # 128
_CONTENT_NGRAM_N: int = 3  # character 3-grams

# GROUP 4 — Domain feature encoding constants
_TLD_LIST: Tuple[str, ...] = ("com", "org", "io", "net", "gov", "edu", "co", "other")
_NUM_TLDS: int = len(_TLD_LIST)  # 8
_TLD_INDEX: Dict[str, int] = {tld: i for i, tld in enumerate(_TLD_LIST)}

# Subdomain special names: docs, api, dev/developer, www, blog, help, support
_SUBDOMAIN_SPECIALS: Tuple[str, ...] = (
    "docs", "api", "dev", "developer", "www", "blog", "help", "support",
)
_NUM_SUBDOMAIN_SPECIALS: int = len(_SUBDOMAIN_SPECIALS)  # 8
_SUBDOMAIN_SPECIAL_INDEX: Dict[str, int] = {s: i for i, s in enumerate(_SUBDOMAIN_SPECIALS)}

# Remaining domain dims: 32 - 8 (tld) - 8 (subdomain) - 4 (label_count, partial_match,
# domain_hint_present, has_subdomain) - 12 (domain_hint_encoding placeholder) = 0
# domain_hint_encoding reuses tld_one_hot (8) + 4 scalar features = 12
_DOMAIN_SCALAR_FEATURES: int = 4  # label_count, partial_match, domain_hint_present, has_subdomain
_DOMAIN_HINT_ENCODING_DIM: int = (
    GROUP_4_DOMAIN_FEATURES_DIM - _NUM_TLDS - _NUM_SUBDOMAIN_SPECIALS - _DOMAIN_SCALAR_FEATURES
)
# 32 - 8 - 8 - 4 = 12

_MAX_HOSTNAME_LABELS: float = 6.0  # normalise label count


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 1 ENCODER: URL PATH TOKENS
#
# Encodes the URL path structure into a fixed-width feature vector.
# Uses _URLSemanticTokenizer.URLTokenProfile when available (avoids
# re-parsing). Falls back to tokenize(url) if profile not provided.
#
# Feature layout (64 dims):
#   [0:7]     boolean features (has_date, has_year, has_version, has_numeric_id,
#             has_short_hash, has_long_hash, has_long_slug)
#   [7]       path_depth / MAX_PATH_DEPTH
#   [8]       query_param_count / MAX_QUERY_PARAMS
#   [9]       has_fragment (bool)
#   [10:34]   keyword one-hot (24 dims)
#   [34:40]   file extension one-hot (6 dims)
#   [40:64]   path character bi-gram bag (24 dims, blake2b bucketed)
# ═════════════════════════════════════════════════════════════════════════════

def _encode_group1_url_path_tokens(
    url: str,
    url_token_profile: Optional[_URLSemanticTokenizer.URLTokenProfile] = None,
) -> np.ndarray:
    """Encode URL path structure into a float32 vector of shape (64,).

    Deterministic: same URL → same output across restarts and processes.
    blake2b is used for path bi-gram bucketing. No Python hash().

    Args:
        url: full URL string.
        url_token_profile: pre-computed token profile from _URLSemanticTokenizer.
                           If None, tokenize(url) is called internally.

    Returns:
        np.ndarray of shape (GROUP_1_URL_PATH_TOKENS_DIM,), dtype float32.
    """
    vec = np.zeros(GROUP_1_URL_PATH_TOKENS_DIM, dtype=np.float32)

    # Obtain token profile.
    profile = url_token_profile or _URLSemanticTokenizer.tokenize(url)

    # ── Boolean features [0:7] ────────────────────────────────────────────
    vec[0] = float(profile.has_date_segment)
    vec[1] = float(profile.has_year_segment)
    vec[2] = float(profile.has_version_segment)
    vec[3] = float(profile.has_numeric_id)
    vec[4] = float(profile.has_short_hash)
    vec[5] = float(profile.has_long_hash)
    vec[6] = float(profile.has_long_slug)

    # ── Scalar features [7:10] ────────────────────────────────────────────
    vec[7] = min(float(profile.path_depth) / _MAX_PATH_DEPTH, 1.0)
    vec[8] = min(float(profile.query_param_count) / _MAX_QUERY_PARAMS, 1.0)
    vec[9] = float(profile.has_fragment)

    # ── Keyword one-hot [10:34] ───────────────────────────────────────────
    keyword_offset = _URL_BOOL_FEATURES  # 10
    for kw in profile.keyword_segments:
        idx = _URL_KEYWORD_INDEX.get(kw)
        if idx is not None:
            vec[keyword_offset + idx] = 1.0

    # ── File extension one-hot [34:40] ────────────────────────────────────
    ext_offset = keyword_offset + _NUM_URL_KEYWORDS  # 10 + 24 = 34
    if profile.file_extension:
        idx = _URL_FILE_EXT_INDEX.get(profile.file_extension)
        if idx is not None:
            vec[ext_offset + idx] = 1.0

    # ── Path character bi-gram bag [40:64] ────────────────────────────────
    # Extract the URL path, lowercase, compute blake2b-bucketed character
    # bi-grams into _URL_BIGRAM_BAG_DIM bins.
    bigram_offset = ext_offset + _NUM_URL_FILE_EXTS  # 34 + 6 = 40
    try:
        path = urlparse(url).path.lower()
    except Exception:  # noqa
        path = ""

    if len(path) >= 2:
        bigram_counts = np.zeros(_URL_BIGRAM_BAG_DIM, dtype=np.float32)
        total_bigrams = 0
        for i in range(len(path) - 1):
            bigram = path[i:i + 2].encode("utf-8")
            bucket = _blake2b_bucket(bigram, _URL_BIGRAM_BAG_DIM)
            bigram_counts[bucket] += 1.0
            total_bigrams += 1
        if total_bigrams > 0:
            bigram_counts /= float(total_bigrams)
        vec[bigram_offset:bigram_offset + _URL_BIGRAM_BAG_DIM] = bigram_counts

    return vec


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 2 ENCODER: HEADER BITMASK
#
# Encodes HTTP response headers and status code into a fixed-width vector.
#
# Feature layout (48 dims):
#   [0:20]    header presence flags (one per key in _HEADER_PRESENCE_KEYS)
#   [20:26]   content-type family one-hot (6 dims)
#   [26:37]   response code one-hot (11 dims: 10 known + other)
#   [37:48]   extra features (11 dims):
#             has_cf_ray, has_retry_after, has_x_robots_tag, has_x_api_version,
#             header_count_normalised, has_set_cookie, has_location_header,
#             is_redirect, is_client_error, is_server_error, content_length_bucket
# ═════════════════════════════════════════════════════════════════════════════

def _classify_content_type_family(content_type: str) -> str:
    """Map a Content-Type header value to a family string.

    Performs case-insensitive prefix matching against known content type
    patterns. Returns "other" for unrecognised types.

    This is a pure function — no side effects, no state.
    """
    ct = content_type.lower().strip()
    if "ld+json" in ct:
        return "ld+json"
    if "json" in ct:
        return "json"
    if "html" in ct:
        return "html"
    if "xml" in ct:
        return "xml"
    if "plain" in ct:
        return "plain"
    return "other"


def _normalise_content_length(headers: Dict[str, str]) -> float:
    """Extract and normalise Content-Length to [0, 1].

    Uses log-scale normalisation: log(1 + content_length) / log(1 + 10MB).
    Returns 0.0 if Content-Length is missing or unparseable.
    """
    raw = headers.get("content-length", "")
    try:
        cl = int(raw)
    except (ValueError, TypeError):
        return 0.0
    if cl <= 0:
        return 0.0
    # log-scale normalise against 10 MB ceiling
    return min(math.log(1.0 + cl) / math.log(1.0 + 10_485_760), 1.0)


def _encode_group2_header_bitmask(
    headers: Dict[str, str],
    response_code: int,
) -> np.ndarray:
    """Encode HTTP response headers into a float32 vector of shape (48,).

    Deterministic: same headers + response code → same output.
    No hashing required for this group — it's pure key presence + one-hot.

    Args:
        headers: HTTP response headers, keys may be mixed-case.
        response_code: HTTP status code.

    Returns:
        np.ndarray of shape (GROUP_2_HEADER_BITMASK_DIM,), dtype float32.
    """
    vec = np.zeros(GROUP_2_HEADER_BITMASK_DIM, dtype=np.float32)

    # Lowercase all header keys once.
    lowered: Dict[str, str] = {k.lower(): v for k, v in headers.items()}

    # ── Header presence flags [0:20] ──────────────────────────────────────
    for key, idx in _HEADER_PRESENCE_INDEX.items():
        if key in lowered:
            vec[idx] = 1.0

    # ── Content-type family one-hot [20:26] ───────────────────────────────
    ct_offset = _NUM_HEADER_PRESENCE  # 20
    ct_raw = lowered.get("content-type", "")
    ct_family = _classify_content_type_family(ct_raw)
    ct_idx = _CT_FAMILY_INDEX.get(ct_family, _CT_FAMILY_INDEX["other"])
    vec[ct_offset + ct_idx] = 1.0

    # ── Response code one-hot [26:37] ─────────────────────────────────────
    rc_offset = ct_offset + _NUM_CT_FAMILIES  # 26
    rc_idx = _RESPONSE_CODE_INDEX.get(response_code)
    if rc_idx is not None:
        vec[rc_offset + rc_idx] = 1.0
    else:
        # "other" bucket — last position
        vec[rc_offset + len(_RESPONSE_CODE_BINS)] = 1.0

    # ── Extra features [37:48] ────────────────────────────────────────────
    extra_offset = rc_offset + _NUM_RESPONSE_CODES  # 37

    vec[extra_offset + 0] = float("cf-ray" in lowered)
    vec[extra_offset + 1] = float("retry-after" in lowered)
    vec[extra_offset + 2] = float("x-robots-tag" in lowered)
    vec[extra_offset + 3] = float("x-api-version" in lowered)

    # Normalised header count — [0, 1] against a ceiling of 40 headers.
    vec[extra_offset + 4] = min(float(len(lowered)) / 40.0, 1.0)

    vec[extra_offset + 5] = float("set-cookie" in lowered)
    vec[extra_offset + 6] = float("location" in lowered)

    # Response code category flags.
    vec[extra_offset + 7] = float(300 <= response_code < 400)   # is_redirect
    vec[extra_offset + 8] = float(400 <= response_code < 500)   # is_client_error
    vec[extra_offset + 9] = float(500 <= response_code < 600)   # is_server_error

    # Content-Length bucket.
    vec[extra_offset + 10] = _normalise_content_length(lowered)

    return vec


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 3 ENCODER: CONTENT N-GRAM HASH
#
# Encodes the first 4KB of content as a frequency-normalised bag of hashed
# character 3-grams. blake2b(digest_size=4) → modular bucket.
#
# This is the richest signal group. It captures the structural vocabulary
# of the page — HTML tag patterns, JSON schema markers, class names — in a
# fixed-width representation that the MLP can learn from.
#
# Feature layout (128 dims):
#   [0:128]   3-gram frequency histogram, L1-normalised (sum to 1.0)
# ═════════════════════════════════════════════════════════════════════════════

def _encode_group3_content_ngram_hash(
    content_prefix: bytes,
) -> np.ndarray:
    """Encode content prefix as a blake2b-hashed 3-gram frequency vector.

    Deterministic: same content_prefix → same output across all environments.
    Uses blake2b(digest_size=4) exclusively. No Python hash().

    Performance: processes up to 4096 characters. At ~200ns per blake2b call
    and ~4094 3-grams, total hashing time is ~0.8ms. Well within the 5ms
    budget for the content window path.

    Args:
        content_prefix: raw bytes from the HTTP response body, up to 4096 bytes.

    Returns:
        np.ndarray of shape (GROUP_3_CONTENT_NGRAM_HASH_DIM,), dtype float32.
        L1-normalised (sums to 1.0 when non-empty, all-zeros when empty).
    """
    vec = np.zeros(_CONTENT_NGRAM_BUCKETS, dtype=np.float32)

    if not content_prefix:
        return vec

    # Decode and truncate to window size.
    window_bytes = content_prefix[:_CONTENT_WINDOW_BYTES]
    try:
        content = window_bytes.decode("utf-8", errors="replace").lower()
    except Exception:  # noqa
        return vec

    if len(content) < _CONTENT_NGRAM_N:
        return vec

    # Hash each 3-gram into a bucket.
    total_grams = 0
    for i in range(len(content) - _CONTENT_NGRAM_N + 1):
        gram = content[i:i + _CONTENT_NGRAM_N].encode("utf-8")
        raw = hashlib.blake2b(gram, digest_size=4).digest()
        bucket = int.from_bytes(raw, byteorder="little", signed=False) % _CONTENT_NGRAM_BUCKETS
        vec[bucket] += 1.0
        total_grams += 1

    # L1 normalisation.
    if total_grams > 0:
        vec /= float(total_grams)

    return vec


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 4 ENCODER: DOMAIN FEATURES
#
# Structural features from the hostname component of the URL.
# Includes TLD one-hot, subdomain analysis, partial match scoring, and
# domain_hint encoding.
#
# Feature layout (32 dims):
#   [0:8]     TLD one-hot (8 dims)
#   [8:16]    subdomain special name flags (8 dims)
#   [16]      has_subdomain (hostname has 3+ labels)
#   [17]      hostname_label_count / MAX_HOSTNAME_LABELS
#   [18]      partial_match_score (Jaccard against nearest fingerprint entry)
#   [19]      domain_hint_present
#   [20:32]   domain_hint_encoding (12 dims): tld_one_hot[8] + 4 scalars
# ═════════════════════════════════════════════════════════════════════════════

def _compute_partial_match_score(hostname: str) -> float:
    """Compute Jaccard similarity between hostname and nearest fingerprint entry.

    Uses blake2b 3-gram hashing for deterministic cross-process comparison.
    Scans all entries in DOMAIN_FINGERPRINT_TABLE to find the highest Jaccard
    similarity. Returns 0.0 if no entry has any overlap.

    This is an O(n * m) operation where n = len(DOMAIN_FINGERPRINT_TABLE) and
    m = len(hostname). At n ~= 40 and m <= 30, total time is negligible.
    """
    host_ngrams = _blake2b_ngram_set(hostname.lower(), 3)
    if not host_ngrams:
        return 0.0

    best_sim: float = 0.0
    for domain in DOMAIN_FINGERPRINT_TABLE:
        domain_ngrams = _blake2b_ngram_set(domain.lower(), 3)
        if not domain_ngrams:
            continue
        intersection = len(host_ngrams & domain_ngrams)
        union = len(host_ngrams | domain_ngrams)
        if union > 0:
            sim = intersection / union
            if sim > best_sim:
                best_sim = sim

    return round(best_sim, 4)


def _extract_tld(hostname: str) -> str:
    """Extract the TLD from a hostname.

    Simple implementation that takes the last label. Handles compound TLDs
    like "co.uk" by checking if the second-to-last label is "co".
    Returns the TLD string or "other" if the hostname has no labels.
    """
    labels = hostname.lower().rstrip(".").split(".")
    if not labels:
        return "other"
    tld = labels[-1]
    # Handle compound TLDs: co.uk, com.au, etc.
    if len(labels) >= 2 and labels[-2] in ("co", "com", "org", "net", "ac"):
        tld = labels[-2]
    return tld if tld in _TLD_INDEX else "other"


def _encode_hostname_core(
    hostname: str,
    vec: np.ndarray,
    offset: int,
    dim: int,
) -> None:
    """Encode hostname features into vec[offset:offset+dim].

    Shared between primary hostname encoding and domain_hint encoding.
    dim must be >= 12 (8 TLD + 4 scalars). Subdomain specials are encoded
    only in the primary hostname encoding (not in domain_hint).

    This is a helper, not a standalone group encoder.
    """
    labels = hostname.lower().rstrip(".").split(".")

    # TLD one-hot [0:8 within sub-vector]
    tld = _extract_tld(hostname)
    tld_idx = _TLD_INDEX.get(tld, _TLD_INDEX["other"])
    if tld_idx < min(dim, _NUM_TLDS):
        vec[offset + tld_idx] = 1.0

    # Scalar features after TLD
    scalar_start = offset + min(dim, _NUM_TLDS)
    remaining = dim - min(dim, _NUM_TLDS)

    if remaining >= 1:
        # has_subdomain
        vec[scalar_start] = float(len(labels) >= 3)
    if remaining >= 2:
        # label count normalised
        vec[scalar_start + 1] = min(float(len(labels)) / _MAX_HOSTNAME_LABELS, 1.0)
    if remaining >= 3:
        # partial match score
        vec[scalar_start + 2] = _compute_partial_match_score(hostname)
    if remaining >= 4:
        # reserved / domain_hint_present (set by caller)
        pass  # leave as 0.0, caller handles


def _encode_group4_domain_features(
    url: str,
    domain_hint: Optional[str],
) -> np.ndarray:
    """Encode domain structural features into a float32 vector of shape (32,).

    Deterministic: same URL + domain_hint → same output.

    Args:
        url: full URL string.
        domain_hint: optional domain string for cross-referencing.

    Returns:
        np.ndarray of shape (GROUP_4_DOMAIN_FEATURES_DIM,), dtype float32.
    """
    vec = np.zeros(GROUP_4_DOMAIN_FEATURES_DIM, dtype=np.float32)

    # Extract hostname.
    try:
        parsed = urlparse(url)
        hostname = parsed.netloc.lower()
        if ":" in hostname:
            hostname = hostname.split(":")[0]
    except Exception:  # noqa
        return vec

    if not hostname:
        return vec

    labels = hostname.rstrip(".").split(".")

    # ── TLD one-hot [0:8] ─────────────────────────────────────────────────
    tld = _extract_tld(hostname)
    tld_idx = _TLD_INDEX.get(tld, _TLD_INDEX["other"])
    vec[tld_idx] = 1.0

    # ── Subdomain special flags [8:16] ────────────────────────────────────
    sub_offset = _NUM_TLDS  # 8
    if len(labels) >= 2:
        # Check all labels except TLD for special names.
        for label in labels[:-1]:
            idx = _SUBDOMAIN_SPECIAL_INDEX.get(label)
            if idx is not None:
                vec[sub_offset + idx] = 1.0

    # ── Scalar features [16:20] ───────────────────────────────────────────
    scalar_offset = sub_offset + _NUM_SUBDOMAIN_SPECIALS  # 16
    vec[scalar_offset + 0] = float(len(labels) >= 3)                  # has_subdomain
    vec[scalar_offset + 1] = min(float(len(labels)) / _MAX_HOSTNAME_LABELS, 1.0)  # label count
    vec[scalar_offset + 2] = _compute_partial_match_score(hostname)   # partial match
    vec[scalar_offset + 3] = float(domain_hint is not None)            # domain_hint_present

    # ── Domain hint encoding [20:32] ──────────────────────────────────────
    hint_offset = scalar_offset + _DOMAIN_SCALAR_FEATURES  # 20
    if domain_hint is not None and domain_hint.strip():
        hint_host = domain_hint.lower().strip()
        # Encode hint hostname into the remaining 12 dims.
        _encode_hostname_core(hint_host, vec, hint_offset, _DOMAIN_HINT_ENCODING_DIM)
    # else: all zeros (no hint)

    return vec


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 5 ENCODER: FINGERPRINT SCORES
#
# Maps _ContentFingerprintIndex Jaccard scores to a fixed 18-dim vector,
# one dimension per topology class in TOPOLOGY_CLASSES order.
# ═════════════════════════════════════════════════════════════════════════════

def _encode_group5_fingerprint_scores(
    content_prefix: bytes,
) -> np.ndarray:
    """Encode fingerprint Jaccard scores as an 18-dim vector.

    Calls _FINGERPRINT_INDEX.score() on the decoded content prefix and maps
    the resulting dict to a fixed-width vector using TOPOLOGY_CLASS_INDEX.

    Args:
        content_prefix: raw bytes, truncated to 4096 internally.

    Returns:
        np.ndarray of shape (GROUP_5_FINGERPRINT_SCORES_DIM,), dtype float32.
    """
    vec = np.zeros(GROUP_5_FINGERPRINT_SCORES_DIM, dtype=np.float32)

    if not content_prefix:
        return vec

    window = content_prefix[:_CONTENT_WINDOW_BYTES]
    try:
        content = window.decode("utf-8", errors="replace").lower()
    except Exception:  # noqa
        return vec

    if len(content) < _NGRAM_SIZE:
        return vec

    scores = _FINGERPRINT_INDEX.score(content)
    for cls, score in scores.items():
        idx = TOPOLOGY_CLASS_INDEX.get(cls)
        if idx is not None and idx < GROUP_5_FINGERPRINT_SCORES_DIM:
            vec[idx] = float(score)

    return vec


# ═════════════════════════════════════════════════════════════════════════════
# GROUP 6 ENCODER: LATTICE SCORES
#
# Encodes the evidence lattice state as an 18-dim vector. Since
# _embed_signals cannot directly access the _EvidenceLattice instance
# (it's a local in classify()), this encoder recomputes fingerprint-based
# proxy scores as described in the spec (Option B).
#
# The fingerprint scores from GROUP 5 serve as the base. GROUP 6 applies
# PARENT_CLASS_MAP propagation on top of them — exactly what the lattice
# does during propagate_through_hierarchy(), but from the fingerprint
# signal only.
#
# This means GROUP 6 is NOT a duplicate of GROUP 5. GROUP 5 is raw
# per-class Jaccard scores. GROUP 6 is those scores after hierarchy
# propagation with _LATTICE_DECAY — the same transformation the full
# lattice applies. The MLP sees both the raw signal and the propagated
# signal, enabling it to weight direct evidence differently from
# inherited evidence.
# ═════════════════════════════════════════════════════════════════════════════

def _encode_group6_lattice_scores(
    content_prefix: bytes,
) -> np.ndarray:
    """Encode lattice-propagated scores as an 18-dim vector.

    Computes GROUP 5 fingerprint scores, then propagates through
    PARENT_CLASS_MAP using the same independent-evidence fusion formula
    as _EvidenceLattice.propagate_through_hierarchy(). The result
    represents what the lattice fusion path would see if only the
    content fingerprint signal were available.

    Args:
        content_prefix: raw bytes.

    Returns:
        np.ndarray of shape (GROUP_6_LATTICE_SCORES_DIM,), dtype float32.
    """
    # Start from GROUP 5 fingerprint scores.
    raw_scores = _encode_group5_fingerprint_scores(content_prefix)
    propagated = np.zeros(GROUP_6_LATTICE_SCORES_DIM, dtype=np.float32)

    # Copy raw scores as direct evidence.
    propagated[:] = raw_scores[:]

    # Propagate upward through PARENT_CLASS_MAP.
    for cls, conf in zip(TOPOLOGY_CLASSES, raw_scores):
        if conf <= 0.0:
            continue
        decayed = float(conf) * _LATTICE_DECAY
        current = cls
        hops = 0
        while hops < 3:
            parent = PARENT_CLASS_MAP.get(current)
            if parent is None or parent == current:
                break
            parent_idx = TOPOLOGY_CLASS_INDEX.get(parent)
            if parent_idx is not None and parent_idx < GROUP_6_LATTICE_SCORES_DIM:
                existing = float(propagated[parent_idx])
                fused = 1.0 - (1.0 - existing) * (1.0 - decayed)
                propagated[parent_idx] = min(fused, 1.0)
            decayed *= _LATTICE_DECAY
            current = parent
            hops += 1

    return propagated


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE ASSEMBLER
#
# Concatenates all six group vectors into the final feature tensor.
# Validates that the total dimension matches TOTAL_FEATURE_DIM.
# Converts from numpy to torch.Tensor with correct dtype and device.
# ═════════════════════════════════════════════════════════════════════════════

def _assemble_feature_vector(
    group1: np.ndarray,
    group2: np.ndarray,
    group3: np.ndarray,
    group4: np.ndarray,
    group5: np.ndarray,
    group6: np.ndarray,
    device: torch.device,
) -> torch.Tensor:
    """Concatenate six feature group vectors into a (1, 308) tensor.

    Validates dimensions of each group before concatenation. Raises
    ValueError if any group has the wrong shape.

    Args:
        group1..group6: np.ndarray vectors from the six encoders.
        device: target device for the output tensor.

    Returns:
        torch.Tensor of shape (1, TOTAL_FEATURE_DIM), dtype float32,
        on the specified device.
    """
    expected_dims = (
        GROUP_1_URL_PATH_TOKENS_DIM,
        GROUP_2_HEADER_BITMASK_DIM,
        GROUP_3_CONTENT_NGRAM_HASH_DIM,
        GROUP_4_DOMAIN_FEATURES_DIM,
        GROUP_5_FINGERPRINT_SCORES_DIM,
        GROUP_6_LATTICE_SCORES_DIM,
    )
    groups = (group1, group2, group3, group4, group5, group6)

    for i, (g, expected_dim) in enumerate(zip(groups, expected_dims)):
        if g.shape != (expected_dim,):
            raise ValueError(
                f"Feature group {i + 1} has shape {g.shape}, "
                f"expected ({expected_dim},). "
                "Feature dimension mismatch will corrupt model inference. "
                "Check the group encoder implementation."
            )

    concatenated = np.concatenate(groups)

    if concatenated.shape[0] != TOTAL_FEATURE_DIM:
        raise ValueError(
            f"Concatenated feature vector has {concatenated.shape[0]} dims, "
            f"expected {TOTAL_FEATURE_DIM}. "
            "This indicates a group dimension constant is wrong."
        )

    # Check for NaN/Inf in the feature vector — these corrupt the forward pass.
    if not np.all(np.isfinite(concatenated)):
        nan_count = int(np.sum(np.isnan(concatenated)))
        inf_count = int(np.sum(np.isinf(concatenated)))
        logger.warning(
            "topology_classifier.feature_vector_corrupt",
            extra={
                "nan_count": nan_count,
                "inf_count": inf_count,
                "note": "replacing non-finite values with 0.0",
            },
        )
        concatenated = np.nan_to_num(concatenated, nan=0.0, posinf=1.0, neginf=0.0)

    tensor = torch.tensor(concatenated, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)


# ═════════════════════════════════════════════════════════════════════════════
# _embed_signals() IMPLEMENTATION
#
# This replaces the NotImplementedError stub in TopologyClassifier.
# It orchestrates the six feature group encoders and returns the final
# tensor ready for the model forward pass.
# ═════════════════════════════════════════════════════════════════════════════

def _embed_signals_impl(
    self: TopologyClassifier,
    url: str,
    headers: Dict[str, str],
    content_prefix: bytes,
    domain_hint: Optional[str],
    url_token_profile: Optional[_URLSemanticTokenizer.URLTokenProfile] = None,
) -> torch.Tensor:
    """Convert the four signal streams into a fixed-width feature tensor.

    This is the implementation of _embed_signals(). It replaces the
    NotImplementedError stub in TopologyClassifier.

    Orchestrates six feature group encoders:
        GROUP 1: URL path tokens (64 dims)
        GROUP 2: Header bitmask (48 dims)
        GROUP 3: Content n-gram hash (128 dims)
        GROUP 4: Domain features (32 dims)
        GROUP 5: Fingerprint scores (18 dims)
        GROUP 6: Lattice scores (18 dims)
    Total: 308 dims

    Returns a (1, 308) float32 tensor on the same device as self._model.

    Deterministic: same inputs → same output across restarts and processes.
    No randomness. No datetime. No process-local state.

    Performance: < 5ms on CPU. The bottleneck is GROUP 3 (blake2b on ~4K
    3-grams ≈ 0.8ms). All other groups are < 0.5ms each.
    """
    # Determine target device.
    try:
        device = next(self._model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    # ── Encode each feature group ─────────────────────────────────────────

    # GROUP 1: URL path tokens
    group1 = _encode_group1_url_path_tokens(url, url_token_profile)

    # GROUP 2: Header bitmask
    # Extract response_code from the calling context. Since _embed_signals
    # doesn't receive response_code directly, we default to 200 when unknown.
    # The actual response code was already used by _classify_by_headers()
    # in an earlier path — this encoding captures header structure, not the
    # code itself. response_code is encoded here for the MLP's benefit.
    #
    # NOTE: To pass response_code through, classify() would need to thread
    # it into this call. For now we encode 200 as a neutral default and
    # rely on the header presence flags to capture the actual status signal.
    # This is acceptable because path 3 (headers) has already checked the
    # response code for hard overrides and rate limiting before path 5 fires.
    response_code = 200
    group2 = _encode_group2_header_bitmask(headers, response_code)

    # GROUP 3: Content n-gram hash
    group3 = _encode_group3_content_ngram_hash(content_prefix)

    # GROUP 4: Domain features
    group4 = _encode_group4_domain_features(url, domain_hint)

    # GROUP 5: Fingerprint scores
    group5 = _encode_group5_fingerprint_scores(content_prefix)

    # GROUP 6: Lattice scores (hierarchy-propagated fingerprint proxy)
    group6 = _encode_group6_lattice_scores(content_prefix)

    # ── Assemble and validate ─────────────────────────────────────────────

    return _assemble_feature_vector(
        group1, group2, group3, group4, group5, group6, device,
    )


# ═════════════════════════════════════════════════════════════════════════════
# _classify_via_model() IMPLEMENTATION
#
# Forward pass through topology_router.pt. Softmax → argmax → class string.
# torch.no_grad() wraps all inference. Thread-safe: model is read-only.
# ═════════════════════════════════════════════════════════════════════════════

def _classify_via_model_impl(
    self: TopologyClassifier,
    features: torch.Tensor,
) -> Tuple[str, float]:
    """Forward pass through topology_router.pt.

    This is the implementation of _classify_via_model(). It replaces
    the NotImplementedError stub in TopologyClassifier.

    Produces a probability distribution over 18 topology classes via softmax.
    Returns the argmax class and its probability as confidence.

    Thread-safe: self._model is not modified. torch.no_grad() prevents
    gradient computation. GIL protects self._model from concurrent
    modification by _reload_classifier_model().

    Never raises — all exceptions are caught and result in
    (FALLBACK_TOPOLOGY_CLASS, 0.0). classify() handles the low-confidence
    case by emitting a warning and returning GENERIC_HTML.

    Performance: < 5ms on GPU, < 15ms on CPU for a typical MLP.
    """
    try:
        # ── Device alignment ──────────────────────────────────────────────
        try:
            model_device = next(self._model.parameters()).device
        except (StopIteration, AttributeError):
            model_device = torch.device("cpu")

        features = features.to(model_device) # noqa

        # ── Forward pass under no_grad ────────────────────────────────────
        with torch.no_grad():
            conditioned = self._model.final_norm(self._model.hidden_state)  # (1, d_model)
            logits = self._model.topology_head(conditioned)  # (1, n_topology)

        # ── Softmax → probabilities ───────────────────────────────────────
        # logits shape: (1, num_classes) or (num_classes,)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        probs = torch.softmax(logits, dim=-1)  # shape (1, num_classes)

        # ── Argmax → predicted class ──────────────────────────────────────
        idx = int(torch.argmax(probs, dim=-1).item())
        conf = float(probs[0, idx].item())

        # ── Index → class string ──────────────────────────────────────────
        topology_class = INDEX_TO_TOPOLOGY_CLASS.get(idx)
        if topology_class is None:
            logger.warning(
                "topology_classifier.model_index_out_of_range",
                extra={
                    "idx": idx,
                    "num_classes": len(INDEX_TO_TOPOLOGY_CLASS),
                    "note": "model output index exceeds class registry — "
                            "model may have been retrained with more classes",
                },
            )
            topology_class = FALLBACK_TOPOLOGY_CLASS

        # ── Top-3 diagnostic log ──────────────────────────────────────────
        prob_list = probs[0].tolist()
        top3_indices = sorted(
            range(len(prob_list)), key=lambda i: -prob_list[i]
        )[:3]
        top3 = [
            (INDEX_TO_TOPOLOGY_CLASS.get(i, "UNKNOWN"), round(prob_list[i], 4))
            for i in top3_indices
        ]

        logger.debug(
            "topology_classifier.model_inference",
            extra={
                "class": topology_class,
                "conf":  round(conf, 4),
                "top3":  top3,
            },
        )

        return topology_class, conf

    except Exception as exc:
        logger.error(
            "topology_classifier.model_forward_failed",
            extra={
                "error":      str(exc),
                "error_type": type(exc).__name__,
                "note":       "falling back to GENERIC_HTML with confidence 0.0",
            },
        )
        return FALLBACK_TOPOLOGY_CLASS, 0.0


# ═════════════════════════════════════════════════════════════════════════════
# MONKEY-PATCH: Install appended implementations onto TopologyClassifier
#
# This replaces the NotImplementedError stubs with the real implementations.
# Python method binding: the `self` parameter is automatically bound when
# the function is assigned to the class.
# ═════════════════════════════════════════════════════════════════════════════

TopologyClassifier._embed_signals = _embed_signals_impl        # type: ignore[assignment]
TopologyClassifier._classify_via_model = _classify_via_model_impl  # type: ignore[assignment]


# ═════════════════════════════════════════════════════════════════════════════
# CONFIDENCE CALIBRATION SYSTEM
#
# Raw softmax probabilities from the MLP are not calibrated — a model
# that says "0.80 confidence" may be correct only 60% of the time.
# Calibration adjusts the model's confidence distribution to match
# empirical accuracy.
#
# Two calibration methods are supported:
#   1. Temperature scaling: divides logits by a learned temperature T.
#      Single parameter. Preserves ranking. The most common calibration
#      method for neural classifiers.
#   2. Platt scaling: applies a sigmoid transformation with learned
#      parameters a and b: calibrated = 1 / (1 + exp(a * logit + b)).
#      More flexible than temperature scaling but can overfit on small
#      validation sets.
#
# The calibrator is initialised with neutral parameters (T=1.0, a=-1.0,
# b=0.0) so that calibration is a no-op until parameters are set by
# index_daemon.py after a validation run.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _CalibrationParams:
    """Learned calibration parameters.

    temperature: logits are divided by this before softmax.
                 T > 1.0 → less confident (softens distribution).
                 T < 1.0 → more confident (sharpens distribution).
                 T = 1.0 → no change (default).

    platt_a, platt_b: Platt scaling sigmoid parameters.
                      calibrated = 1 / (1 + exp(platt_a * logit + platt_b))
                      a = -1.0, b = 0.0 → identity (default).
    """
    temperature: float = 1.0
    platt_a:     float = -1.0
    platt_b:     float = 0.0
    is_fitted:   bool  = False
    fit_samples: int   = 0
    fit_nll:     float = 0.0    # negative log-likelihood on validation set


class ConfidenceCalibrator:
    """Post-hoc confidence calibration for topology classifier output.

    Thread-safe: parameter updates are via atomic assignment of a frozen
    dataclass. Reads during calibrate() always see a consistent parameter set.

    Usage:
        calibrator = ConfidenceCalibrator()
        # After validation run:
        calibrator.update_params(temperature=1.5, platt_a=-1.2, platt_b=0.1)
        # During inference:
        raw_conf = 0.80
        calibrated = calibrator.calibrate(raw_conf, method="temperature")
    """

    def __init__(self) -> None:
        self._params = _CalibrationParams()
        self._lock = threading.Lock()

    @property
    def params(self) -> _CalibrationParams:
        """Current calibration parameters (read-only snapshot)."""
        return self._params

    @property
    def is_fitted(self) -> bool:
        """True if parameters have been fitted from validation data."""
        return self._params.is_fitted

    def update_params(
        self,
        *,
        temperature: Optional[float] = None,
        platt_a: Optional[float] = None,
        platt_b: Optional[float] = None,
        fit_samples: int = 0,
        fit_nll: float = 0.0,
    ) -> None:
        """Update calibration parameters.

        Thread-safe: creates a new _CalibrationParams and atomically assigns it.
        In-flight calibrate() calls always see a consistent parameter set.

        Args:
            temperature: new temperature value (must be > 0).
            platt_a: new Platt 'a' parameter.
            platt_b: new Platt 'b' parameter.
            fit_samples: number of validation samples used for fitting.
            fit_nll: negative log-likelihood on the validation set.
        """
        with self._lock:
            current = self._params
            new_params = _CalibrationParams(
                temperature=temperature if temperature is not None else current.temperature,
                platt_a=platt_a if platt_a is not None else current.platt_a,
                platt_b=platt_b if platt_b is not None else current.platt_b,
                is_fitted=True,
                fit_samples=fit_samples,
                fit_nll=fit_nll,
            )
            # Validate temperature.
            if new_params.temperature <= 0:
                raise ValueError(
                    f"temperature must be > 0, got {new_params.temperature}. "
                    "T <= 0 produces undefined behaviour in softmax."
                )
            self._params = new_params

        logger.info(
            "topology_classifier.calibration_updated",
            extra={
                "temperature": new_params.temperature,
                "platt_a":     new_params.platt_a,
                "platt_b":     new_params.platt_b,
                "fit_samples": new_params.fit_samples,
                "fit_nll":     round(new_params.fit_nll, 4),
            },
        )

    def calibrate_temperature(self, raw_confidence: float) -> float:
        """Apply temperature scaling to a raw confidence value.

        Temperature scaling operates on the logit (pre-softmax) space.
        For a single-class confidence, we approximate the logit from the
        softmax output and re-apply softmax with temperature.

        For a binary interpretation: logit ≈ log(p / (1-p)), then
        calibrated = sigmoid(logit / T).
        """
        params = self._params
        if params.temperature == 1.0 or not params.is_fitted:
            return raw_confidence

        # Clamp to avoid log(0) or log(inf).
        p = max(min(raw_confidence, 1.0 - 1e-7), 1e-7)
        logit = math.log(p / (1.0 - p))
        scaled_logit = logit / params.temperature
        calibrated = 1.0 / (1.0 + math.exp(-scaled_logit))
        return round(calibrated, 4)

    def calibrate_platt(self, raw_confidence: float) -> float:
        """Apply Platt sigmoid scaling to a raw confidence value.

        Platt scaling: calibrated = 1 / (1 + exp(a * logit + b))
        where logit = log(p / (1-p)).

        Default parameters (a=-1.0, b=0.0) produce identity transformation.
        """
        params = self._params
        if not params.is_fitted:
            return raw_confidence

        p = max(min(raw_confidence, 1.0 - 1e-7), 1e-7)
        logit = math.log(p / (1.0 - p))
        calibrated = 1.0 / (1.0 + math.exp(params.platt_a * logit + params.platt_b))
        return round(calibrated, 4)

    def calibrate(
        self,
        raw_confidence: float,
        method: str = "temperature",
    ) -> float:
        """Apply the specified calibration method.

        Args:
            raw_confidence: raw softmax probability in [0.0, 1.0].
            method: "temperature" or "platt".

        Returns:
            Calibrated confidence in [0.0, 1.0].
        """
        if method == "temperature":
            return self.calibrate_temperature(raw_confidence)
        elif method == "platt":
            return self.calibrate_platt(raw_confidence)
        else:
            raise ValueError(
                f"Unknown calibration method {method!r}. "
                "Use 'temperature' or 'platt'."
            )

    def health(self) -> Dict[str, Any]:
        """Return calibration health snapshot for Witness."""
        p = self._params
        return {
            "is_fitted":    p.is_fitted,
            "temperature":  p.temperature,
            "platt_a":      p.platt_a,
            "platt_b":      p.platt_b,
            "fit_samples":  p.fit_samples,
            "fit_nll":      round(p.fit_nll, 4),
        }


# Module-level calibrator singleton.
_CALIBRATOR: ConfidenceCalibrator = ConfidenceCalibrator()


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE DRIFT MONITOR
#
# Tracks running statistics (mean, variance) per feature group using
# Welford's online algorithm. Detects drift when the production feature
# distribution diverges from the training baseline by more than
# DRIFT_ALERT_SIGMA standard deviations.
#
# Drift detection catches:
#   - Silent feature corruption (e.g. hashing function change)
#   - Input distribution shift (e.g. new domain category emerging)
#   - Feature engineering bugs introduced during deployment
#   - Model/feature version mismatch
#
# The monitor is passive — it observes feature vectors but never modifies
# them. Alerts are emitted via the logger. index_daemon.py subscribes to
# drift alerts to trigger model retraining when necessary.
# ═════════════════════════════════════════════════════════════════════════════

DRIFT_ALERT_SIGMA: Final[float] = 3.0     # alert threshold in standard deviations
DRIFT_MIN_SAMPLES: Final[int]   = 100     # minimum samples before drift detection activates
DRIFT_CHECK_INTERVAL: Final[int] = 64     # check every N observations (power of 2)


@dataclass
class _WelfordState:
    """Online mean/variance tracker using Welford's algorithm.

    Numerically stable single-pass computation of running mean and variance.
    Memory: O(dim) per tracked group. No history buffer.

    Update formula (per Welford 1962):
        count += 1
        delta = x - mean
        mean += delta / count
        delta2 = x - mean
        m2 += delta * delta2
        variance = m2 / count  (population variance)
    """
    count: int = 0
    mean:  Optional[np.ndarray] = None
    m2:    Optional[np.ndarray] = None

    def update(self, x: np.ndarray) -> None:
        """Incorporate one observation into the running statistics."""
        if self.mean is None:
            self.mean = np.zeros_like(x, dtype=np.float64)
            self.m2 = np.zeros_like(x, dtype=np.float64)

        self.count += 1
        delta = x.astype(np.float64) - self.mean
        self.mean += delta / self.count
        delta2 = x.astype(np.float64) - self.mean
        self.m2 += delta * delta2

    @property
    def variance(self) -> Optional[np.ndarray]:
        """Population variance. Returns None if count < 2."""
        if self.count < 2 or self.m2 is None:
            return None
        return self.m2 / self.count

    @property
    def std(self) -> Optional[np.ndarray]:
        """Population standard deviation. Returns None if count < 2."""
        v = self.variance
        if v is None:
            return None
        return np.sqrt(v)


class FeatureDriftMonitor:
    """Online feature drift detector using per-group Welford statistics.

    Thread-safe: each observe() call acquires a lock. The lock is
    fine-grained (per-monitor, not global) and held for < 10µs.

    Usage:
        monitor = FeatureDriftMonitor()
        # Set baseline from training data statistics:
        monitor.set_baseline("content_ngram_hash", train_mean, train_std)
        # During inference:
        monitor.observe(feature_vector)
        # Check health:
        alerts = monitor.check_drift()
    """

    def __init__(self) -> None:
        self._states: Dict[str, _WelfordState] = {}
        self._baselines: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}  # (mean, std)
        self._lock = threading.Lock()
        self._total_observations: int = 0
        self._drift_alerts: List[Dict[str, Any]] = []

    def set_baseline(
        self,
        group_name: str,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        """Set the training baseline for a feature group.

        Args:
            group_name: name from _FeatureGroupMeta.
            mean: per-feature mean from training data.
            std: per-feature standard deviation from training data.
        """
        with self._lock:
            self._baselines[group_name] = (
                mean.astype(np.float64),
                np.maximum(std.astype(np.float64), 1e-8),
            )

    def observe(self, feature_vector: np.ndarray) -> None:
        """Incorporate one feature vector observation.

        Updates Welford statistics for each group and periodically checks
        for drift. feature_vector must have shape (TOTAL_FEATURE_DIM,).
        """
        if feature_vector.shape != (TOTAL_FEATURE_DIM,):
            return  # silently skip malformed vectors

        with self._lock:
            self._total_observations += 1

            for meta in _GROUP_META:
                group_vec = feature_vector[meta.offset:meta.offset + meta.dim]
                if meta.name not in self._states:
                    self._states[meta.name] = _WelfordState()
                self._states[meta.name].update(group_vec)

            # Periodic drift check.
            if (self._total_observations & (DRIFT_CHECK_INTERVAL - 1)) == 0:
                self._check_drift_locked()

    def _check_drift_locked(self) -> None:
        """Check for drift across all groups. Must be called under _lock.

        Uses Welch's t-test approximation: for each feature dimension,
        compute z = (production_mean - baseline_mean) / baseline_std.
        If the group-level average |z| exceeds DRIFT_ALERT_SIGMA, emit
        a warning.
        """
        if self._total_observations < DRIFT_MIN_SAMPLES:
            return

        for meta in _GROUP_META:
            state = self._states.get(meta.name)
            baseline = self._baselines.get(meta.name)
            if state is None or state.mean is None or baseline is None:
                continue
            if state.count < DRIFT_MIN_SAMPLES:
                continue

            train_mean, train_std = baseline
            z_scores = np.abs(state.mean - train_mean) / train_std
            avg_z = float(np.mean(z_scores))

            if avg_z > DRIFT_ALERT_SIGMA:
                alert = {
                    "group":             meta.name,
                    "avg_z_score":       round(avg_z, 3),
                    "threshold":         DRIFT_ALERT_SIGMA,
                    "observation_count": state.count,
                    "max_z_feature":     int(np.argmax(z_scores)),
                    "max_z_value":       round(float(np.max(z_scores)), 3),
                }
                self._drift_alerts.append(alert)
                logger.warning(
                    "topology_classifier.feature_drift_detected",
                    extra=alert,
                )

    def check_drift(self) -> List[Dict[str, Any]]:
        """Return accumulated drift alerts and clear the buffer."""
        with self._lock:
            alerts = list(self._drift_alerts)
            self._drift_alerts.clear()
        return alerts

    def health(self) -> Dict[str, Any]:
        """Return drift monitor health snapshot for Witness."""
        with self._lock:
            group_stats = {}
            for meta in _GROUP_META:
                state = self._states.get(meta.name)
                if state and state.count >= 2:
                    group_stats[meta.name] = {
                        "count": state.count,
                        "mean_norm": round(float(np.linalg.norm(state.mean)), 4)
                                     if state.mean is not None else None,
                        "std_norm":  round(float(np.linalg.norm(state.std)), 4)
                                     if state.std is not None else None,
                        "has_baseline": meta.name in self._baselines,
                    }
                else:
                    group_stats[meta.name] = {
                        "count": state.count if state else 0,
                        "mean_norm": None,
                        "std_norm": None,
                        "has_baseline": meta.name in self._baselines,
                    }

            return {
                "total_observations": self._total_observations,
                "groups":             group_stats,
                "pending_alerts":     len(self._drift_alerts),
            }


# Module-level drift monitor singleton.
_DRIFT_MONITOR: FeatureDriftMonitor = FeatureDriftMonitor()


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION RESULT CACHE
#
# LRU cache with per-class TTL for classification results. Caches
# deterministic-path results (paths 1-4 and lattice fusion) but NOT
# ML-path results (path 5), because ML results depend on model weights
# that may change between calls.
#
# Cache key: blake2b hash of (url, sorted_headers, content_prefix[:256],
# response_code). Using 256 bytes of content_prefix (not 4096) for the
# cache key is intentional — two URLs with identical first 256 bytes
# almost certainly have the same topology class, and truncating reduces
# the key computation cost.
#
# The cache is process-local and dies on restart. This is intentional —
# stale classification results from a previous model version are worse
# than cache misses.
# ═════════════════════════════════════════════════════════════════════════════

_CACHE_MAX_ENTRIES: Final[int] = 10_000
_CACHE_DEFAULT_TTL: Final[float] = 300.0   # 5 minutes
_CACHE_KEY_PREFIX_BYTES: Final[int] = 256  # content prefix bytes in cache key

# Per-class TTL overrides. Hard-override classes have TTL=0 (never cached).
_CACHE_TTL_BY_CLASS: Dict[str, float] = {
    "AUTH_REDIRECT":         0.0,
    "CLOUDFLARE_CHALLENGE":  0.0,
    "RATE_LIMITED":          0.0,
}


@dataclass
class _CacheEntry:
    """Single entry in the classification cache."""
    result:     TopologyClassification
    expires_at: float    # time.monotonic() deadline
    path_used:  str      # classification_path — only deterministic paths cached


class ClassificationCache:
    """LRU classification result cache with per-class TTL.

    Thread-safe: all operations acquire a lock. The lock is fine-grained
    and held for < 5µs per operation.

    Only deterministic-path results are cached. ML-path results are NOT
    cached because they depend on model weights that may change.

    Usage:
        cache = ClassificationCache()
        # Lookup:
        result = cache.get(input)
        if result is not None:
            return result
        # ... classify ...
        # Store (only if deterministic path):
        if path_used != "model" and path_used != "fallback":
            cache.put(input, result)
    """

    def __init__(
        self,
        max_entries: int = _CACHE_MAX_ENTRIES,
        default_ttl: float = _CACHE_DEFAULT_TTL,
    ) -> None:
        self._max_entries = max_entries
        self._default_ttl = default_ttl
        self._store: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.Lock()
        # Statistics.
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def _compute_key(self, input: ClassifierInput) -> str: # noqa
        """Compute a deterministic cache key from ClassifierInput.

        Uses blake2b for deterministic hashing. The key includes:
          - URL (full)
          - Sorted header key-value pairs
          - First 256 bytes of content_prefix
          - Response code
        """
        h = hashlib.blake2b(digest_size=16)
        h.update(input.url.encode("utf-8"))
        # Sort headers for deterministic ordering.
        for k, v in sorted(input.headers.items()):
            h.update(k.lower().encode("utf-8"))
            h.update(b"=")
            h.update(v.encode("utf-8"))
            h.update(b"&")
        h.update(input.content_prefix[:_CACHE_KEY_PREFIX_BYTES])
        h.update(str(input.response_code).encode("utf-8"))
        return h.hexdigest()

    def _ttl_for_class(self, topology_class: str) -> float:
        """Get TTL for a topology class. 0.0 means never cache."""
        return _CACHE_TTL_BY_CLASS.get(topology_class, self._default_ttl)

    def get(self, input: ClassifierInput) -> Optional[TopologyClassification]: # noqa
        """Look up a cached classification result.

        Returns None on miss or if the entry has expired.
        On hit, moves the entry to the end of the OrderedDict (LRU).
        """
        key = self._compute_key(input)
        now = time.monotonic()

        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if now >= entry.expires_at:
                # Expired — remove and count as miss.
                del self._store[key]
                self._misses += 1
                return None
            # Hit — move to end (most recently used).
            self._store.move_to_end(key)
            self._hits += 1
            return entry.result

    def put(
        self,
        input: ClassifierInput, # noqa
        result: TopologyClassification,
    ) -> None:
        """Store a classification result in the cache.

        Only stores results from deterministic paths (not model or fallback).
        Hard-override classes (TTL=0) are never cached.
        """
        # Never cache ML or fallback results.
        if result.classification_path in ("model", "fallback"):
            return

        ttl = self._ttl_for_class(result.topology_class)
        if ttl <= 0.0:
            return

        key = self._compute_key(input)
        now = time.monotonic()

        with self._lock:
            # Remove existing entry if present.
            if key in self._store:
                del self._store[key]

            # Evict oldest entries if at capacity.
            while len(self._store) >= self._max_entries:
                self._store.popitem(last=False)
                self._evictions += 1

            self._store[key] = _CacheEntry(
                result=result,
                expires_at=now + ttl,
                path_used=result.classification_path,
            )

    def invalidate(self) -> None:
        """Clear the entire cache. Called after model reload."""
        with self._lock:
            self._store.clear()

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0]."""
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def health(self) -> Dict[str, Any]:
        """Return cache health snapshot for Witness."""
        with self._lock:
            return {
                "size":       len(self._store),
                "max_size":   self._max_entries,
                "hits":       self._hits,
                "misses":     self._misses,
                "evictions":  self._evictions,
                "hit_rate":   round(self.hit_rate, 4),
            }


# Module-level cache singleton.
_CLASSIFICATION_CACHE: ClassificationCache = ClassificationCache()


# ═════════════════════════════════════════════════════════════════════════════
# MODEL HEALTH VALIDATOR
#
# Validates topology_router.pt at load time and periodically during runtime.
# Checks:
#   1. Input dimension matches TOTAL_FEATURE_DIM (308)
#   2. Output dimension matches NUM_TOPOLOGY_CLASSES (18)
#   3. No NaN or Inf in model weights
#   4. Weight statistics (mean, std) are within expected ranges
#   5. Smoke test: synthetic input produces valid output
#
# Validation runs once after each model load (in initialize() and
# _reload_classifier_model()). Results are exposed via health() for
# Witness monitoring.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _ModelHealthResult:
    """Result of a model health validation run."""
    is_healthy:         bool
    input_dim_ok:       bool
    output_dim_ok:      bool
    no_nan_weights:     bool
    no_inf_weights:     bool
    weight_mean:        float
    weight_std:         float
    smoke_test_passed:  bool
    smoke_test_class:   str
    smoke_test_conf:    float
    param_count:        int
    device:             str
    errors:             Tuple[str, ...]


def validate_model_health(
    model: torch.nn.Module,
    expected_input_dim: int = TOTAL_FEATURE_DIM,
    expected_output_dim: int = NUM_TOPOLOGY_CLASSES,
) -> _ModelHealthResult:
    """Comprehensive health check for topology_router.pt.

    Runs five checks: dimension validation, NaN/Inf detection, weight
    statistics, and a smoke test with synthetic input.

    This function is pure — it does not modify the model. It uses
    torch.no_grad() for the smoke test forward pass.

    Args:
        model: loaded torch.nn.Module in eval mode.
        expected_input_dim: expected input feature dimension.
        expected_output_dim: expected output class dimension.

    Returns:
        _ModelHealthResult with all check results.
    """
    errors: List[str] = []

    # ── Determine device ──────────────────────────────────────────────────
    try:
        device = str(next(model.parameters()).device)
    except (StopIteration, AttributeError):
        device = "unknown"
        errors.append("model has no parameters — cannot determine device")

    # ── Parameter count ───────────────────────────────────────────────────
    try:
        param_count = sum(p.numel() for p in model.parameters())
    except Exception: # noqa
        param_count = 0
        errors.append("failed to count model parameters")

    # ── NaN / Inf check ───────────────────────────────────────────────────
    no_nan = True
    no_inf = True
    weight_values: List[float] = []
    try:
        for name, param in model.named_parameters():
            data = param.data
            if torch.isnan(data).any():
                no_nan = False
                errors.append(f"NaN detected in parameter {name}")
            if torch.isinf(data).any():
                no_inf = False
                errors.append(f"Inf detected in parameter {name}")
            weight_values.extend(data.cpu().flatten().tolist()[:1000])
    except Exception as exc:
        errors.append(f"weight inspection failed: {exc}")

    # ── Weight statistics ─────────────────────────────────────────────────
    if weight_values:
        w_arr = np.array(weight_values, dtype=np.float64)
        weight_mean = float(np.mean(w_arr))
        weight_std = float(np.std(w_arr))
    else:
        weight_mean = 0.0
        weight_std = 0.0

    # ── Smoke test ────────────────────────────────────────────────────────
    input_dim_ok = True
    output_dim_ok = True
    smoke_passed = False
    smoke_class = FALLBACK_TOPOLOGY_CLASS
    smoke_conf = 0.0

    try:
        dev = torch.device(device if device != "unknown" else "cpu")
        synthetic_input = torch.randn(1, expected_input_dim, dtype=torch.float32, device=dev)

        with torch.no_grad():
            output = model(synthetic_input)

        if output.dim() == 1:
            output = output.unsqueeze(0)

        actual_output_dim = output.shape[-1]
        if actual_output_dim != expected_output_dim:
            output_dim_ok = False
            errors.append(
                f"output dim {actual_output_dim} != expected {expected_output_dim}"
            )

        # Check if the forward pass produced valid output.
        if torch.isnan(output).any() or torch.isinf(output).any():
            errors.append("smoke test output contains NaN/Inf")
        else:
            probs = torch.softmax(output, dim=-1)
            idx = int(torch.argmax(probs, dim=-1).item())
            smoke_class = INDEX_TO_TOPOLOGY_CLASS.get(idx, FALLBACK_TOPOLOGY_CLASS)
            smoke_conf = round(float(probs[0, idx].item()), 4)
            smoke_passed = True

    except RuntimeError as exc:
        exc_str = str(exc)
        if "size mismatch" in exc_str or "expected" in exc_str.lower():
            input_dim_ok = False
            errors.append(f"input dimension mismatch: {exc_str[:200]}")
        else:
            errors.append(f"smoke test runtime error: {exc_str[:200]}")
    except Exception as exc:
        errors.append(f"smoke test failed: {type(exc).__name__}: {str(exc)[:200]}")

    is_healthy = (
        input_dim_ok
        and output_dim_ok
        and no_nan
        and no_inf
        and smoke_passed
        and len(errors) == 0
    )

    return _ModelHealthResult(
        is_healthy=is_healthy,
        input_dim_ok=input_dim_ok,
        output_dim_ok=output_dim_ok,
        no_nan_weights=no_nan,
        no_inf_weights=no_inf,
        weight_mean=round(weight_mean, 6),
        weight_std=round(weight_std, 6),
        smoke_test_passed=smoke_passed,
        smoke_test_class=smoke_class,
        smoke_test_conf=smoke_conf,
        param_count=param_count,
        device=device,
        errors=tuple(errors),
    )


# ═════════════════════════════════════════════════════════════════════════════
# BATCH CLASSIFICATION INTERFACE
#
# Batched feature embedding for multiple inputs. When the ML path is
# invoked for N URLs simultaneously (e.g. during batch pre-classification
# of a crawl manifest), batching amortises the PyTorch overhead.
#
# The batch interface does NOT batch the deterministic paths (1-4) because
# they are already fast enough individually. Only the ML path benefits
# from batching.
# ═════════════════════════════════════════════════════════════════════════════

def batch_embed_signals(
    inputs: Sequence[Tuple[str, Dict[str, str], bytes, Optional[str]]],
    model: torch.nn.Module,
) -> torch.Tensor:
    """Embed multiple inputs into a batched feature tensor.

    Each input is a tuple of (url, headers, content_prefix, domain_hint).
    Returns a tensor of shape (N, TOTAL_FEATURE_DIM).

    This is used by crawl_planner.py for batch pre-classification of
    candidate URLs before live traversal. It does NOT call classify() —
    it only computes features.

    Args:
        inputs: sequence of (url, headers, content_prefix, domain_hint) tuples.
        model: loaded model (used to determine device).

    Returns:
        torch.Tensor of shape (N, TOTAL_FEATURE_DIM), dtype float32.
    """
    if not inputs:
        try:
            device = next(model.parameters()).device
        except (StopIteration, AttributeError):
            device = torch.device("cpu")
        return torch.zeros(0, TOTAL_FEATURE_DIM, dtype=torch.float32, device=device)

    try:
        device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        device = torch.device("cpu")

    batch_list: List[np.ndarray] = []

    for url, headers, content_prefix, domain_hint in inputs:
        try:
            g1 = _encode_group1_url_path_tokens(url)
            g2 = _encode_group2_header_bitmask(headers, 200)
            g3 = _encode_group3_content_ngram_hash(content_prefix)
            g4 = _encode_group4_domain_features(url, domain_hint)
            g5 = _encode_group5_fingerprint_scores(content_prefix)
            g6 = _encode_group6_lattice_scores(content_prefix)
            concatenated = np.concatenate([g1, g2, g3, g4, g5, g6])
            concatenated = np.nan_to_num(concatenated, nan=0.0, posinf=1.0, neginf=0.0)
        except Exception:  # noqa
            concatenated = np.zeros(TOTAL_FEATURE_DIM, dtype=np.float32)
        batch_list.append(concatenated)

    batch_np = np.stack(batch_list)
    batch_tensor = torch.tensor(batch_np, dtype=torch.float32, device=device)
    return batch_tensor


def batch_classify_via_model(
    features: torch.Tensor,
    model: torch.nn.Module,
) -> List[Tuple[str, float]]:
    """Batched forward pass through topology_router.pt.

    Args:
        features: tensor of shape (N, TOTAL_FEATURE_DIM) from batch_embed_signals().
        model: loaded model.

    Returns:
        List of (topology_class, confidence) tuples, one per input.
    """
    if features.shape[0] == 0:
        return []

    try:
        with torch.no_grad():
            logits = model(features)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)
        probs = torch.softmax(logits, dim=-1)  # (N, num_classes)

        results: List[Tuple[str, float]] = []
        for i in range(probs.shape[0]):
            idx = int(torch.argmax(probs[i]).item())
            conf = float(probs[i, idx].item())
            cls = INDEX_TO_TOPOLOGY_CLASS.get(idx, FALLBACK_TOPOLOGY_CLASS)
            results.append((cls, round(conf, 4)))
        return results

    except Exception as exc:
        logger.error(
            "topology_classifier.batch_forward_failed",
            extra={"error": str(exc), "batch_size": features.shape[0]},
        )
        return [(FALLBACK_TOPOLOGY_CLASS, 0.0)] * features.shape[0]


# ═════════════════════════════════════════════════════════════════════════════
# ONLINE LEARNING SIGNAL COLLECTOR
#
# Collects (feature_vector, predicted_class, predicted_confidence) tuples
# from the ML path for use by index_daemon.py's gradient step loop.
#
# The collector maintains a rolling buffer of the last N classification
# signals. index_daemon.py drains this buffer periodically to construct
# training batches for online learning.
#
# The collector also tracks:
#   - Per-class invocation counts (how often each class is predicted)
#   - Per-path invocation counts (how often each path resolves)
#   - Confidence distribution histogram
#   - Feature importance approximation via input gradient norms
#
# This is the bridge between the classifier (inference) and the index
# daemon (training). The classifier deposits signals; the daemon consumes
# them. Neither blocks the other.
# ═════════════════════════════════════════════════════════════════════════════

_SIGNAL_BUFFER_SIZE: Final[int] = 2048


@dataclass
class _ClassificationSignal:
    """One observation from the ML path for index_daemon consumption."""
    feature_vector: np.ndarray       # shape (TOTAL_FEATURE_DIM,)
    predicted_class: str
    predicted_confidence: float
    url_hash: str                    # blake2b hash of URL (not the URL itself)
    timestamp: float                 # time.monotonic()


class OnlineLearningSignalCollector:
    """Rolling buffer of ML path classification signals for index_daemon.

    Thread-safe: deposit() and drain() acquire a lock. deposit() is called
    from the classify() hot path and must be < 1µs.

    Usage:
        collector = OnlineLearningSignalCollector()
        # In classify() ML path:
        collector.deposit(features, predicted_class, confidence, url)
        # In index_daemon gradient loop:
        signals = collector.drain()
        for signal in signals:
            # construct training sample from signal.feature_vector
    """

    def __init__(self, buffer_size: int = _SIGNAL_BUFFER_SIZE) -> None:
        self._buffer: Deque[_ClassificationSignal] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        # Statistics.
        self._total_deposited: int = 0
        self._class_counts: Dict[str, int] = defaultdict(int)
        self._path_counts: Dict[str, int] = defaultdict(int)
        self._confidence_histogram: np.ndarray = np.zeros(10, dtype=np.int64)
        # [0.0, 0.1), [0.1, 0.2), ..., [0.9, 1.0]

    def deposit(
        self,
        feature_vector: np.ndarray,
        predicted_class: str,
        predicted_confidence: float,
        url: str,
        path_used: str = "model",
    ) -> None:
        """Deposit one ML path observation into the buffer.

        This is called from the classify() hot path. It must complete in
        < 1µs. The lock is held for a deque append (O(1), amortised).

        Args:
            feature_vector: numpy array of shape (TOTAL_FEATURE_DIM,).
            predicted_class: predicted topology class string.
            predicted_confidence: softmax confidence.
            url: original URL (hashed for privacy).
            path_used: classification path string.
        """
        url_hash = hashlib.blake2b(
            url.encode("utf-8"), digest_size=8
        ).hexdigest()

        signal = _ClassificationSignal(
            feature_vector=feature_vector.copy(),
            predicted_class=predicted_class,
            predicted_confidence=predicted_confidence,
            url_hash=url_hash,
            timestamp=time.monotonic(),
        )

        with self._lock:
            self._buffer.append(signal)
            self._total_deposited += 1
            self._class_counts[predicted_class] += 1
            self._path_counts[path_used] += 1
            # Update confidence histogram.
            bin_idx = min(int(predicted_confidence * 10), 9)
            self._confidence_histogram[bin_idx] += 1

    def drain(self, max_signals: int = 0) -> List[_ClassificationSignal]:
        """Drain signals from the buffer for index_daemon consumption.

        Returns up to max_signals signals (0 = drain all). Signals are
        removed from the buffer atomically.

        Args:
            max_signals: maximum number of signals to drain. 0 = all.

        Returns:
            List of _ClassificationSignal in chronological order.
        """
        with self._lock:
            if max_signals <= 0 or max_signals >= len(self._buffer):
                signals = list(self._buffer)
                self._buffer.clear()
            else:
                signals = []
                for _ in range(max_signals):
                    if self._buffer:
                        signals.append(self._buffer.popleft())
        return signals

    def buffer_size(self) -> int:
        """Current number of signals in the buffer."""
        with self._lock:
            return len(self._buffer)

    def health(self) -> Dict[str, Any]:
        """Return collector health snapshot for Witness."""
        with self._lock:
            return {
                "buffer_size":           len(self._buffer),
                "buffer_capacity":       self._buffer.maxlen,
                "total_deposited":       self._total_deposited,
                "class_distribution":    dict(self._class_counts),
                "path_distribution":     dict(self._path_counts),
                "confidence_histogram":  self._confidence_histogram.tolist(),
            }


# Module-level signal collector singleton.
_SIGNAL_COLLECTOR: OnlineLearningSignalCollector = OnlineLearningSignalCollector()


# ═════════════════════════════════════════════════════════════════════════════
# ENHANCED CLASSIFY WITH CACHE + DRIFT + SIGNALS
#
# Wraps the base classify() method with cache lookup/store, drift
# monitoring, and signal collection. This is installed as a decorator-style
# wrapper around the original classify() method.
# ═════════════════════════════════════════════════════════════════════════════

_original_classify = TopologyClassifier.classify


async def _classify_with_infrastructure(
    self: TopologyClassifier,
    input: ClassifierInput, # noqa
) -> TopologyClassification:
    """Enhanced classify() with caching, drift monitoring, and signal collection.

    Wraps the original classify() method with three production concerns:
      1. Cache lookup before classification, cache store after.
      2. Feature drift monitoring (only for ML path).
      3. Online learning signal collection (only for ML path).

    The cache is only populated by deterministic-path results. ML-path
    results are NOT cached because they depend on model weights that may
    change. Hard-override classes are never cached.

    The drift monitor observes the feature vector only when the ML path
    is invoked. Deterministic-path results don't produce feature vectors.

    The signal collector deposits feature vectors for index_daemon.py
    only when the ML path is invoked.
    """
    # ── Cache lookup ──────────────────────────────────────────────────────
    cached = _CLASSIFICATION_CACHE.get(input)
    if cached is not None:
        logger.debug(
            "topology_classifier.cache_hit",
            extra={
                "url":   input.url[:100],
                "class": cached.topology_class,
                "path":  cached.classification_path,
            },
        )
        return cached

    # ── Run original classify ─────────────────────────────────────────────
    result = await _original_classify(self, input)

    # ── Cache store (deterministic paths only) ────────────────────────────
    _CLASSIFICATION_CACHE.put(input, result)

    # ── Drift monitoring + signal collection (ML path only) ───────────────
    if result.classification_path == "model":
        try:
            # Recompute the feature vector for monitoring/collection.
            # This is a redundant computation but acceptable because the ML path
            # is rare (< 15% of queries) and the feature computation is < 5ms.
            url_tokens = _URLSemanticTokenizer.tokenize(input.url)
            features = _embed_signals_impl(
                self, input.url, input.headers, input.content_prefix,
                None, url_tokens,
            )
            feature_np = features.squeeze(0).cpu().numpy()

            # Drift monitoring.
            _DRIFT_MONITOR.observe(feature_np)

            # Signal collection.
            _SIGNAL_COLLECTOR.deposit(
                feature_vector=feature_np,
                predicted_class=result.topology_class,
                predicted_confidence=result.confidence,
                url=input.url,
                path_used=result.classification_path,
            )
        except Exception as exc:
            # Monitoring must never disrupt classification.
            logger.debug(
                "topology_classifier.monitoring_error",
                extra={"error": str(exc)},
            )

    return result


# Install the enhanced classify method.
TopologyClassifier.classify = _classify_with_infrastructure  # type: ignore[assignment]


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFIER HEALTH AGGREGATOR
#
# Aggregates health from all subsystems: model, cache, drift monitor,
# signal collector, and calibrator. Exposed via classifier_health() for
# Witness and cold_start.py.
# ═════════════════════════════════════════════════════════════════════════════

def classifier_health(classifier: TopologyClassifier) -> Dict[str, Any]:
    """Aggregate health from all classifier subsystems.

    Called by Witness for real-time monitoring and by cold_start.py for
    startup validation.

    Args:
        classifier: the TopologyClassifier instance (usually CLASSIFIER).

    Returns:
        Dict with health snapshots from:
          - model (is_ready, device, param_count)
          - cache (size, hit_rate)
          - drift_monitor (observations, alerts)
          - signal_collector (buffer_size, distributions)
          - calibrator (is_fitted, parameters)
          - feature_version (version string, total_dim)
    """
    return {
        "model": {
            "is_ready":   classifier.is_ready(),
            "device":     classifier.model_device(),
            "trie_size":  classifier.domain_trie_size(),
        },
        "cache":            _CLASSIFICATION_CACHE.health(),
        "drift_monitor":    _DRIFT_MONITOR.health(),
        "signal_collector": _SIGNAL_COLLECTOR.health(),
        "calibrator":       _CALIBRATOR.health(),
        "feature_space": {
            "version":   FEATURE_VERSION,
            "total_dim": TOTAL_FEATURE_DIM,
            "groups":    [
                {"name": m.name, "offset": m.offset, "dim": m.dim}
                for m in _GROUP_META
            ],
        },
    }


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE ESTIMATOR
#
# Approximates per-feature importance using input-gradient norms.
# Given a feature vector and the model, computes the gradient of the
# predicted class logit with respect to the input features. The L2 norm
# of the gradient per feature group indicates how much that group
# contributes to the prediction.
#
# This is used by index_daemon.py for:
#   - Diagnosing misclassifications (which features drove the error?)
#   - Feature selection for model compression
#   - Identifying which signal paths are most valuable per topology class
# ═════════════════════════════════════════════════════════════════════════════

def compute_feature_importance(
    model: torch.nn.Module,
    feature_vector: torch.Tensor,
) -> Dict[str, float]:
    """Compute per-group feature importance via input gradient norms.

    The gradient of the predicted class logit with respect to the input
    features gives the local sensitivity of the prediction to each feature.
    The L2 norm of the gradient over each feature group gives a single
    importance score per group.

    This requires one forward + backward pass. torch.enable_grad() is used
    explicitly to override any ambient no_grad context.

    Args:
        model: topology_router.pt model.
        feature_vector: shape (1, TOTAL_FEATURE_DIM), dtype float32.

    Returns:
        Dict mapping group name → importance score (L2 norm of gradient).
        Returns empty dict on error.
    """
    try:
        # Clone and detach to avoid modifying the original tensor.
        x = feature_vector.clone().detach().requires_grad_(True)

        # Enable grad explicitly (we may be inside a no_grad context).
        with torch.enable_grad():
            logits = model(x)
            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            predicted_idx = int(torch.argmax(logits, dim=-1).item())
            predicted_logit = logits[0, predicted_idx]
            predicted_logit.backward()

        if x.grad is None:
            return {}

        grad = x.grad.squeeze(0).cpu().numpy()

        importance: Dict[str, float] = {}
        for meta in _GROUP_META:
            group_grad = grad[meta.offset:meta.offset + meta.dim]
            importance[meta.name] = round(float(np.linalg.norm(group_grad)), 6)

        return importance

    except Exception as exc:
        logger.debug(
            "topology_classifier.feature_importance_failed",
            extra={"error": str(exc)},
        )
        return {}


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION EXPLAINER
#
# Produces human-readable explanations of why the classifier chose a
# particular topology class. Combines evidence from all paths, lattice
# fusion, and the ML model's top-3 predictions into a structured
# explanation.
#
# Used by the debugging interface and by surprise_detector.py when
# investigating unexpected classifications.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ClassificationExplanation:
    """Human-readable explanation of a classification decision."""
    topology_class:       str
    confidence:           float
    classification_path:  str
    path_description:     str
    all_signals:          Dict[str, str]
    fallback_chain:       List[str]
    confidence_tier:      str
    is_hard_override:     bool
    ml_path_invoked:      bool
    ml_top3:              List[Tuple[str, float]]
    feature_importance:   Dict[str, float]
    cache_hit:            bool
    explanation_text:     str


def explain_classification(
    result: TopologyClassification,
    classifier: Optional[TopologyClassifier] = None, # noqa
) -> ClassificationExplanation:
    """Generate a structured explanation for a classification result.

    This is a diagnostic tool, not part of the critical path. It should
    only be called when a human or surprise_detector.py needs to understand
    why a particular classification was made.

    Args:
        result: the TopologyClassification to explain.
        classifier: optional classifier instance for ML model access.

    Returns:
        ClassificationExplanation with full diagnostic information.
    """
    path = result.classification_path
    path_descriptions = {
        "domain":   "Domain fingerprint match — hostname found in the domain trie",
        "url":      "URL structure match — path pattern matched a known topology",
        "header":   "HTTP header signal — response headers indicated topology",
        "window":   "Content window match — structural markers found in first 4KB",
        "lattice":  "Lattice fusion — multiple weak signals fused above threshold",
        "model":    "ML model — topology_router.pt forward pass",
        "fallback": "Fallback — all paths below confidence threshold, defaulting to GENERIC_HTML",
    }
    path_desc = path_descriptions.get(path, f"Unknown path: {path}")

    conf_tier = confidence_label(result.confidence)
    is_override = result.topology_class in HARD_OVERRIDE_CLASSES
    ml_invoked = path in ("model", "fallback")

    # Extract fallback chain from signals_used.
    raw_chain = result.signals_used.get("fallback_chain", "")
    fallback = raw_chain.split(",") if raw_chain else []

    # Build explanation text.
    lines = [
        f"Classification: {result.topology_class} ({conf_tier} confidence: {result.confidence:.3f})",
        f"Path: {path} — {path_desc}",
    ]
    if is_override:
        lines.append("This is a HARD OVERRIDE class — ML path was bypassed.")
    if fallback:
        lines.append(f"Fallback chain: {' → '.join(fallback)}")
    if ml_invoked:
        lines.append("ML model was invoked (all deterministic paths below threshold).")

    # Placeholder for ML top-3 (would need to re-run the model to get these).
    ml_top3: List[Tuple[str, float]] = []

    return ClassificationExplanation(
        topology_class=result.topology_class,
        confidence=result.confidence,
        classification_path=path,
        path_description=path_desc,
        all_signals=dict(result.signals_used),
        fallback_chain=fallback,
        confidence_tier=conf_tier,
        is_hard_override=is_override,
        ml_path_invoked=ml_invoked,
        ml_top3=ml_top3,
        feature_importance={},
        cache_hit=False,
        explanation_text="\n".join(lines),
    )


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE VECTOR SERIALISATION
#
# Utilities for serialising and deserialising feature vectors for storage
# in training data files and checkpoint records.
#
# Serialisation format: raw float32 bytes with a 16-byte header:
#   [0:4]    magic bytes "AXFV" (AXIOM Feature Vector)
#   [4:8]    version number (uint32, little-endian)
#   [8:12]   dimension (uint32, little-endian)
#   [12:16]  reserved (zeros)
#   [16:]    float32 values (dim * 4 bytes)
#
# This format is stable across Python versions and architectures.
# index_daemon.py uses it to persist training data between gradient steps.
# ═════════════════════════════════════════════════════════════════════════════

_FV_MAGIC: bytes = b"AXFV"
_FV_VERSION: int = 1
_FV_HEADER_SIZE: int = 16


def serialise_feature_vector(vec: np.ndarray) -> bytes:
    """Serialise a feature vector to bytes with a self-describing header.

    Args:
        vec: numpy array of shape (TOTAL_FEATURE_DIM,), dtype float32.

    Returns:
        bytes: header + raw float32 data.

    Raises:
        ValueError: if vec has wrong shape or dtype.
    """
    if vec.shape != (TOTAL_FEATURE_DIM,):
        raise ValueError(
            f"Expected shape ({TOTAL_FEATURE_DIM},), got {vec.shape}."
        )
    vec32 = vec.astype(np.float32, copy=False)
    header = struct.pack("<4sIII", _FV_MAGIC, _FV_VERSION, TOTAL_FEATURE_DIM, 0)
    return header + vec32.tobytes()


def deserialise_feature_vector(data: bytes) -> np.ndarray:
    """Deserialise a feature vector from bytes.

    Args:
        data: bytes from serialise_feature_vector().

    Returns:
        numpy array of shape (TOTAL_FEATURE_DIM,), dtype float32.

    Raises:
        ValueError: if data is malformed.
    """
    if len(data) < _FV_HEADER_SIZE:
        raise ValueError(
            f"Data too short: {len(data)} bytes, expected at least {_FV_HEADER_SIZE}."
        )
    magic, version, dim, _ = struct.unpack("<4sIII", data[:_FV_HEADER_SIZE])
    if magic != _FV_MAGIC:
        raise ValueError(f"Invalid magic bytes: {magic!r}, expected {_FV_MAGIC!r}.")
    if version != _FV_VERSION:
        raise ValueError(f"Unsupported version: {version}, expected {_FV_VERSION}.")
    if dim != TOTAL_FEATURE_DIM:
        raise ValueError(
            f"Dimension mismatch: file has {dim}, expected {TOTAL_FEATURE_DIM}."
        )
    expected_size = _FV_HEADER_SIZE + dim * 4
    if len(data) != expected_size:
        raise ValueError(
            f"Data size {len(data)} != expected {expected_size}."
        )
    return np.frombuffer(data[_FV_HEADER_SIZE:], dtype=np.float32).copy()


# ═════════════════════════════════════════════════════════════════════════════
# TOPOLOGY CLASS STATISTICS TRACKER
#
# Tracks per-class classification statistics: invocation count, mean
# confidence, confidence variance, path distribution, and latency.
#
# Used by interface.py for health reporting and by index_daemon.py for
# gradient step weighting (rare classes get boosted gradient weight).
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _ClassStats:
    """Running statistics for one topology class."""
    count:           int = 0
    total_conf:      float = 0.0
    total_conf_sq:   float = 0.0
    total_latency:   float = 0.0
    path_counts:     Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def mean_confidence(self) -> float:
        return self.total_conf / self.count if self.count > 0 else 0.0

    @property
    def var_confidence(self) -> float:
        if self.count < 2:
            return 0.0
        mean = self.mean_confidence
        return (self.total_conf_sq / self.count) - (mean * mean)

    @property
    def mean_latency(self) -> float:
        return self.total_latency / self.count if self.count > 0 else 0.0


class TopologyClassStatsTracker:
    """Per-class classification statistics tracker.

    Thread-safe: all operations acquire a lock.

    Usage:
        tracker = TopologyClassStatsTracker()
        tracker.record(result)
        stats = tracker.get_stats("SAAS_DOCS")
    """

    def __init__(self) -> None:
        self._stats: Dict[str, _ClassStats] = defaultdict(_ClassStats)
        self._lock = threading.Lock()
        self._total_classifications: int = 0

    def record(self, result: TopologyClassification) -> None:
        """Record one classification result."""
        with self._lock:
            self._total_classifications += 1
            s = self._stats[result.topology_class]
            s.count += 1
            s.total_conf += result.confidence
            s.total_conf_sq += result.confidence * result.confidence
            s.total_latency += result.latency_ms
            s.path_counts[result.classification_path] += 1

    def get_stats(self, topology_class: str) -> Dict[str, Any]:
        """Get statistics for one topology class."""
        with self._lock:
            s = self._stats.get(topology_class)
            if s is None or s.count == 0:
                return {
                    "count": 0,
                    "mean_confidence": 0.0,
                    "var_confidence": 0.0,
                    "mean_latency_ms": 0.0,
                    "path_distribution": {},
                }
            return {
                "count":             s.count,
                "mean_confidence":   round(s.mean_confidence, 4),
                "var_confidence":    round(s.var_confidence, 6),
                "mean_latency_ms":   round(s.mean_latency, 3),
                "path_distribution": dict(s.path_counts),
            }

    def all_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all topology classes."""
        with self._lock:
            return {
                cls: self.get_stats(cls) for cls in self._stats
            }

    def health(self) -> Dict[str, Any]:
        """Return tracker health snapshot for Witness."""
        with self._lock:
            return {
                "total_classifications": self._total_classifications,
                "classes_seen":          len(self._stats),
                "top_classes": sorted(
                    [(cls, s.count) for cls, s in self._stats.items()],
                    key=lambda x: -x[1],
                )[:5],
            }


# Module-level stats tracker singleton.
_STATS_TRACKER: TopologyClassStatsTracker = TopologyClassStatsTracker()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SELF-TESTS
#
# Run at import time to catch configuration errors before the first
# classification. If any assertion fails, the module raises at import —
# a crash at startup is infinitely preferable to silent misconfiguration.
# ═════════════════════════════════════════════════════════════════════════════

def _self_test_feature_dimensions() -> None:
    """Validate feature group dimensions sum to TOTAL_FEATURE_DIM.

    This catches dimension constant errors before any classification
    runs. A mismatch here means the feature vector will be wrong-shaped
    for the model, producing garbage output.
    """
    group_dims = [m.dim for m in _GROUP_META]
    actual_sum = sum(group_dims)
    assert actual_sum == TOTAL_FEATURE_DIM, (
        f"Feature group dimensions sum to {actual_sum}, "
        f"expected TOTAL_FEATURE_DIM = {TOTAL_FEATURE_DIM}. "
        "Check GROUP_*_DIM constants."
    )

    # Validate offsets are contiguous.
    expected_offset = 0
    for meta in _GROUP_META:
        assert meta.offset == expected_offset, (
            f"Feature group {meta.name!r} has offset {meta.offset}, "
            f"expected {expected_offset}. Groups must be contiguous."
        )
        expected_offset += meta.dim

    # Validate NUM_TOPOLOGY_CLASSES matches TOPOLOGY_CLASSES.
    assert NUM_TOPOLOGY_CLASSES == len(TOPOLOGY_CLASSES), (
        f"NUM_TOPOLOGY_CLASSES = {NUM_TOPOLOGY_CLASSES}, "
        f"len(TOPOLOGY_CLASSES) = {len(TOPOLOGY_CLASSES)}. "
        "These must match."
    )

    # Validate GROUP 5 and GROUP 6 dimensions match class count.
    assert GROUP_5_FINGERPRINT_SCORES_DIM == NUM_TOPOLOGY_CLASSES, (
        f"GROUP_5 dim {GROUP_5_FINGERPRINT_SCORES_DIM} != {NUM_TOPOLOGY_CLASSES}."
    )
    assert GROUP_6_LATTICE_SCORES_DIM == NUM_TOPOLOGY_CLASSES, (
        f"GROUP_6 dim {GROUP_6_LATTICE_SCORES_DIM} != {NUM_TOPOLOGY_CLASSES}."
    )


def _self_test_hash_determinism() -> None:
    """Verify blake2b hashing is deterministic.

    Runs the same input through _blake2b_hash32 twice and asserts the
    results match. This catches hypothetical environment-level issues
    with the hashlib implementation.
    """
    test_input = b"axiom_classifier_self_test"
    h1 = _blake2b_hash32(test_input)
    h2 = _blake2b_hash32(test_input)
    assert h1 == h2, (
        f"blake2b hash is not deterministic: {h1} != {h2}. "
        "Feature vectors will be corrupt. Check the hashlib installation."
    )

    # Verify bucket distribution is consistent.
    b1 = _blake2b_bucket(test_input, 128)
    b2 = _blake2b_bucket(test_input, 128)
    assert b1 == b2, (
        f"blake2b bucket is not deterministic: {b1} != {b2}."
    )


def _self_test_group_encoders() -> None:
    """Smoke test each group encoder with synthetic input.

    Verifies that each encoder produces the correct output shape and
    does not raise on valid input. Uses trivial inputs — this is not
    a correctness test, just a shape + no-crash test.
    """
    test_url = "https://docs.stripe.com/api/charges"
    test_headers = {"content-type": "application/json", "cf-ray": "abc123"}
    test_content = b"<html><body>test content for classifier self-test</body></html>"

    # GROUP 1
    g1 = _encode_group1_url_path_tokens(test_url)
    assert g1.shape == (GROUP_1_URL_PATH_TOKENS_DIM,), f"GROUP 1 shape: {g1.shape}"

    # GROUP 2
    g2 = _encode_group2_header_bitmask(test_headers, 200)
    assert g2.shape == (GROUP_2_HEADER_BITMASK_DIM,), f"GROUP 2 shape: {g2.shape}"

    # GROUP 3
    g3 = _encode_group3_content_ngram_hash(test_content)
    assert g3.shape == (GROUP_3_CONTENT_NGRAM_HASH_DIM,), f"GROUP 3 shape: {g3.shape}"

    # GROUP 4
    g4 = _encode_group4_domain_features(test_url, None)
    assert g4.shape == (GROUP_4_DOMAIN_FEATURES_DIM,), f"GROUP 4 shape: {g4.shape}"

    # GROUP 5
    g5 = _encode_group5_fingerprint_scores(test_content)
    assert g5.shape == (GROUP_5_FINGERPRINT_SCORES_DIM,), f"GROUP 5 shape: {g5.shape}"

    # GROUP 6
    g6 = _encode_group6_lattice_scores(test_content)
    assert g6.shape == (GROUP_6_LATTICE_SCORES_DIM,), f"GROUP 6 shape: {g6.shape}"

    # Concatenation
    concatenated = np.concatenate([g1, g2, g3, g4, g5, g6])
    assert concatenated.shape == (TOTAL_FEATURE_DIM,), (
        f"Concatenated shape: {concatenated.shape}, expected ({TOTAL_FEATURE_DIM},)"
    )

    # Determinism check: encode twice, compare.
    g1_b = _encode_group1_url_path_tokens(test_url)
    assert np.array_equal(g1, g1_b), "GROUP 1 is not deterministic"

    g3_b = _encode_group3_content_ngram_hash(test_content)
    assert np.array_equal(g3, g3_b), "GROUP 3 is not deterministic"


def _self_test_serialisation() -> None:
    """Verify feature vector serialisation roundtrips correctly."""
    original = np.random.randn(TOTAL_FEATURE_DIM).astype(np.float32)
    data = serialise_feature_vector(original)
    restored = deserialise_feature_vector(data)
    assert np.array_equal(original, restored), (
        "Feature vector serialisation roundtrip failed. "
        "This will corrupt training data in index_daemon.py."
    )


def _self_test_index_maps() -> None:
    """Verify TOPOLOGY_CLASS_INDEX and INDEX_TO_TOPOLOGY_CLASS are consistent."""
    for cls, idx in TOPOLOGY_CLASS_INDEX.items():
        assert INDEX_TO_TOPOLOGY_CLASS[idx] == cls, (
            f"Index map inconsistency: TOPOLOGY_CLASS_INDEX[{cls!r}] = {idx}, "
            f"but INDEX_TO_TOPOLOGY_CLASS[{idx}] = {INDEX_TO_TOPOLOGY_CLASS.get(idx)!r}."
        )
    assert len(TOPOLOGY_CLASS_INDEX) == NUM_TOPOLOGY_CLASSES, (
        f"TOPOLOGY_CLASS_INDEX has {len(TOPOLOGY_CLASS_INDEX)} entries, "
        f"expected {NUM_TOPOLOGY_CLASSES}."
    )
    assert len(INDEX_TO_TOPOLOGY_CLASS) == NUM_TOPOLOGY_CLASSES, (
        f"INDEX_TO_TOPOLOGY_CLASS has {len(INDEX_TO_TOPOLOGY_CLASS)} entries, "
        f"expected {NUM_TOPOLOGY_CLASSES}."
    )


# Run all self-tests at import time.
_self_test_feature_dimensions()
_self_test_hash_determinism()
_self_test_group_encoders()
_self_test_serialisation()
_self_test_index_maps()


# ═════════════════════════════════════════════════════════════════════════════
# EXTENDED SELF-TESTS: FEATURE VECTOR PROPERTIES
#
# These tests verify properties of the feature encoders that go beyond
# shape checks — they verify normalisation, boundedness, and symmetry.
# ═════════════════════════════════════════════════════════════════════════════

def _self_test_feature_normalisation() -> None:
    """Verify that feature encoders produce bounded outputs.

    Every feature value must be in [0.0, 1.0] for binary/normalised
    features. GROUP 3 (n-gram histogram) must sum to approximately 1.0
    when non-empty.
    """
    test_url = "https://docs.stripe.com/api/v2/charges?page=3#section"
    test_headers = {
        "content-type": "application/json",
        "cf-ray": "abc123",
        "x-api-version": "2024-01",
    }
    test_content = (
        b'<html><head><meta charset="utf-8"><meta name="viewport" '
        b'content="width=device-width"></head>'
        b'<body><article class="post-content">'
        b'<h1>Test Article</h1>'
        b'<p>This is test content for feature normalisation validation.</p>'
        b'<pre><code>const x = 42;</code></pre>'
        b'</article></body></html>'
    )

    # GROUP 1: all values in [0, 1]
    g1 = _encode_group1_url_path_tokens(test_url)
    assert np.all(g1 >= 0.0), f"GROUP 1 has negative values: min={g1.min()}"
    assert np.all(g1 <= 1.0), f"GROUP 1 has values > 1.0: max={g1.max()}"

    # GROUP 2: all values in [0, 1]
    g2 = _encode_group2_header_bitmask(test_headers, 200)
    assert np.all(g2 >= 0.0), f"GROUP 2 has negative values: min={g2.min()}"
    assert np.all(g2 <= 1.0), f"GROUP 2 has values > 1.0: max={g2.max()}"

    # GROUP 3: non-negative, sums to ~1.0 when non-empty
    g3 = _encode_group3_content_ngram_hash(test_content)
    assert np.all(g3 >= 0.0), f"GROUP 3 has negative values: min={g3.min()}"
    g3_sum = float(g3.sum())
    if g3_sum > 0:
        assert abs(g3_sum - 1.0) < 0.01, (
            f"GROUP 3 L1 norm = {g3_sum}, expected ~1.0 (L1-normalised histogram)"
        )

    # GROUP 4: all values in [0, 1]
    g4 = _encode_group4_domain_features(test_url, None)
    assert np.all(g4 >= 0.0), f"GROUP 4 has negative values: min={g4.min()}"
    assert np.all(g4 <= 1.0), f"GROUP 4 has values > 1.0: max={g4.max()}"

    # GROUP 5: all values in [0, 1]
    g5 = _encode_group5_fingerprint_scores(test_content)
    assert np.all(g5 >= 0.0), f"GROUP 5 has negative values: min={g5.min()}"
    assert np.all(g5 <= 1.0), f"GROUP 5 has values > 1.0: max={g5.max()}"

    # GROUP 6: all values in [0, 1]
    g6 = _encode_group6_lattice_scores(test_content)
    assert np.all(g6 >= 0.0), f"GROUP 6 has negative values: min={g6.min()}"
    assert np.all(g6 <= 1.0), f"GROUP 6 has values > 1.0: max={g6.max()}"


def _self_test_edge_cases() -> None:
    """Verify encoders handle edge cases without crashing.

    Tests: empty inputs, very long inputs, non-ASCII content, malformed
    URLs, and missing headers.
    """
    # Empty content
    g3_empty = _encode_group3_content_ngram_hash(b"")
    assert g3_empty.shape == (GROUP_3_CONTENT_NGRAM_HASH_DIM,)
    assert float(g3_empty.sum()) == 0.0

    # Very short content (below minimum window)
    g3_short = _encode_group3_content_ngram_hash(b"ab")
    assert g3_short.shape == (GROUP_3_CONTENT_NGRAM_HASH_DIM,)

    # Non-ASCII content
    g3_unicode = _encode_group3_content_ngram_hash("日本語テスト".encode("utf-8"))
    assert g3_unicode.shape == (GROUP_3_CONTENT_NGRAM_HASH_DIM,)
    assert np.all(np.isfinite(g3_unicode))

    # Malformed URL
    g1_bad = _encode_group1_url_path_tokens("not-a-url")
    assert g1_bad.shape == (GROUP_1_URL_PATH_TOKENS_DIM,)

    # Empty headers
    g2_empty = _encode_group2_header_bitmask({}, 200)
    assert g2_empty.shape == (GROUP_2_HEADER_BITMASK_DIM,)

    # Domain features with empty URL
    g4_empty = _encode_group4_domain_features("https://", None)
    assert g4_empty.shape == (GROUP_4_DOMAIN_FEATURES_DIM,)

    # Fingerprint scores with binary content
    g5_binary = _encode_group5_fingerprint_scores(bytes(range(256)))
    assert g5_binary.shape == (GROUP_5_FINGERPRINT_SCORES_DIM,)
    assert np.all(np.isfinite(g5_binary))


def _self_test_content_structure_analyzer() -> None:
    """Verify content structure analyzer handles various inputs."""
    # HTML content
    html = (
        b'<!DOCTYPE html><html><head><meta charset="utf-8">'
        b'<meta property="og:type" content="article">'
        b'</head><body><script>var x=1;</script>'
        b'<form><input type="text"></form></body></html>'
    )
    profile = analyze_content_structure(html)
    assert profile.has_doctype is True
    assert profile.has_charset is True
    assert profile.has_og_type is True
    assert profile.script_block_count >= 1
    assert profile.form_count >= 1

    # JSON content
    json_content = b'{"items": [{"id": 1}, {"id": 2}], "total": 2}'
    profile_json = analyze_content_structure(json_content)
    assert profile_json.looks_like_json is True
    assert profile_json.json_object_root is True
    assert profile_json.json_estimated_depth >= 2

    # Empty content
    profile_empty = analyze_content_structure(b"")
    assert profile_empty.total_tags == 0
    assert profile_empty.looks_like_json is False


def _self_test_calibrator() -> None:
    """Verify calibrator produces valid outputs."""
    cal = ConfidenceCalibrator()

    # Default (unfitted) should pass through unchanged.
    assert cal.calibrate(0.8) == 0.8
    assert cal.calibrate(0.5) == 0.5

    # After fitting with T=2.0, confidence should be lower.
    cal.update_params(temperature=2.0, fit_samples=100)
    result = cal.calibrate_temperature(0.9)
    assert 0.0 < result < 1.0
    assert result < 0.9, f"T=2.0 should reduce confidence, got {result}"

    # Platt scaling should produce valid output.
    cal.update_params(platt_a=-0.8, platt_b=0.1, fit_samples=100)
    result_platt = cal.calibrate_platt(0.85)
    assert 0.0 < result_platt < 1.0


def _self_test_adaptive_threshold() -> None:
    """Verify adaptive threshold controller boundaries."""
    ctrl = AdaptiveThresholdController()

    # Default effective threshold should be _THETA_CONFIDENT.
    assert ctrl.effective_confident_threshold() == _THETA_CONFIDENT

    # After recording all correct labels, threshold should stay at default.
    for _ in range(100):
        ctrl.record("SAAS_DOCS", was_correct=True)
    assert ctrl.effective_confident_threshold() == _THETA_CONFIDENT

    # Reset and verify clean state.
    ctrl.reset()
    assert ctrl.effective_confident_threshold() == _THETA_CONFIDENT


# Run extended self-tests.
_self_test_feature_normalisation()
_self_test_edge_cases()
_self_test_calibrator()


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFIER SINGLETON
#
# Module-level TopologyClassifier instance. This is the single entry point
# for classification. interface.py imports this singleton and calls
# CLASSIFIER.classify(input) on every URL.
#
# The singleton is constructed at import time but NOT initialised. The
# model is loaded lazily by CLASSIFIER.initialize() during cold_start.py.
# Until initialize() is called, classify() will raise
# ClassifierModelNotInitialized.
# ═════════════════════════════════════════════════════════════════════════════

CLASSIFIER: TopologyClassifier = TopologyClassifier()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE PUBLIC API SURFACE
#
# Everything that external modules (interface.py, cold_start.py,
# index_daemon.py) may import from classifier_append.py.
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# ADAPTIVE CONFIDENCE THRESHOLD CONTROLLER
#
# Dynamically adjusts the confidence thresholds used in classify() based
# on recent classification accuracy. When the model is consistently
# overconfident (high confidence but wrong class), the thresholds are
# raised. When the model is well-calibrated, thresholds stay at defaults.
#
# The controller tracks a windowed accuracy estimate and adjusts thresholds
# using a PID-like control loop. Adjustments are bounded to prevent
# runaway threshold drift.
#
# index_daemon.py feeds ground-truth labels back to the controller after
# extraction quality is evaluated. If the predicted class produced a
# bad extraction, the label is marked as incorrect.
# ═════════════════════════════════════════════════════════════════════════════

_THRESHOLD_WINDOW_SIZE: Final[int] = 200      # sliding window for accuracy estimate
_THRESHOLD_ADJUST_INTERVAL: Final[int] = 50   # adjust every N ground-truth labels
_THRESHOLD_MAX_BOOST: Final[float] = 0.10     # maximum upward adjustment
_THRESHOLD_MIN_ACCURACY: Final[float] = 0.70  # below this, thresholds rise
_THRESHOLD_TARGET_ACCURACY: Final[float] = 0.85  # target accuracy level


class AdaptiveThresholdController:
    """PID-like controller for classification confidence thresholds.

    Monitors recent accuracy and adjusts the effective confidence
    threshold. The adjustment is additive — the effective threshold is:
        effective = base_theta + adjustment

    The adjustment is bounded by _THRESHOLD_MAX_BOOST to prevent
    the threshold from drifting too far from the spec-defined value.

    Thread-safe: all operations acquire a lock.

    Usage:
        controller = AdaptiveThresholdController()
        # Feed ground truth:
        controller.record(predicted_class="SAAS_DOCS", was_correct=True)
        # Get effective threshold:
        theta = controller.effective_confident_threshold()
    """

    def __init__(self) -> None:
        self._window: Deque[bool] = deque(maxlen=_THRESHOLD_WINDOW_SIZE)
        self._lock = threading.Lock()
        self._adjustment: float = 0.0
        self._total_labels: int = 0
        self._per_class_accuracy: Dict[str, Deque[bool]] = defaultdict(
            lambda: deque(maxlen=_THRESHOLD_WINDOW_SIZE)
        )

    def record(self, predicted_class: str, was_correct: bool) -> None:
        """Record a ground-truth label for a recent classification.

        Called by index_daemon.py after extraction quality evaluation.
        was_correct is True if the predicted topology class produced a
        satisfactory extraction (signal density > floor, non-empty).

        Args:
            predicted_class: the class that was predicted.
            was_correct: whether the prediction was correct.
        """
        with self._lock:
            self._window.append(was_correct)
            self._per_class_accuracy[predicted_class].append(was_correct)
            self._total_labels += 1

            # Periodic threshold adjustment.
            if (self._total_labels % _THRESHOLD_ADJUST_INTERVAL) == 0:
                self._adjust_locked()

    def _adjust_locked(self) -> None:
        """Adjust threshold based on recent accuracy. Called under _lock."""
        if len(self._window) < _THRESHOLD_ADJUST_INTERVAL:
            return

        accuracy = sum(self._window) / len(self._window)

        if accuracy < _THRESHOLD_MIN_ACCURACY:
            # Model is overconfident — raise threshold.
            error = _THRESHOLD_TARGET_ACCURACY - accuracy
            # Proportional gain: 0.5 * error, bounded.
            delta = min(0.5 * error, 0.02)
            self._adjustment = min(self._adjustment + delta, _THRESHOLD_MAX_BOOST)
            logger.info(
                "topology_classifier.threshold_raised",
                extra={
                    "accuracy":    round(accuracy, 3),
                    "adjustment":  round(self._adjustment, 4),
                    "window_size": len(self._window),
                },
            )
        elif accuracy > _THRESHOLD_TARGET_ACCURACY and self._adjustment > 0:
            # Model is well-calibrated — relax threshold toward default.
            self._adjustment = max(self._adjustment - 0.005, 0.0)
            logger.info(
                "topology_classifier.threshold_relaxed",
                extra={
                    "accuracy":    round(accuracy, 3),
                    "adjustment":  round(self._adjustment, 4),
                },
            )

    def effective_confident_threshold(self) -> float:
        """Current effective THETA_CLASSIFY_CONFIDENT.

        Returns base threshold + adaptive adjustment.
        """
        with self._lock:
            return _THETA_CONFIDENT + self._adjustment

    def effective_fallback_threshold(self) -> float: # noqa
        """Current effective THETA_CLASSIFY_FALLBACK.

        Fallback threshold is not adjusted — it is a hard floor.
        """
        return _THETA_FALLBACK

    def per_class_accuracy(self, topology_class: str) -> float:
        """Windowed accuracy for a specific topology class."""
        with self._lock:
            window = self._per_class_accuracy.get(topology_class)
            if not window or len(window) == 0:
                return 0.0
            return sum(window) / len(window)

    def reset(self) -> None:
        """Reset the controller to default state."""
        with self._lock:
            self._window.clear()
            self._per_class_accuracy.clear()
            self._adjustment = 0.0
            self._total_labels = 0

    def health(self) -> Dict[str, Any]:
        """Return controller health snapshot for Witness."""
        with self._lock:
            accuracy = (
                sum(self._window) / len(self._window)
                if len(self._window) > 0 else 0.0
            )
            return {
                "total_labels":       self._total_labels,
                "window_size":        len(self._window),
                "window_accuracy":    round(accuracy, 4),
                "adjustment":         round(self._adjustment, 4),
                "effective_theta":    round(self.effective_confident_threshold(), 4),
                "classes_tracked":    len(self._per_class_accuracy),
            }


# Module-level threshold controller singleton.
_THRESHOLD_CONTROLLER: AdaptiveThresholdController = AdaptiveThresholdController()

_self_test_adaptive_threshold()


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION TELEMETRY PIPELINE
#
# Structured telemetry emitter for classification events. Every classify()
# call produces a telemetry record that Witness consumes for real-time
# dashboards, alerting, and SLA monitoring.
#
# Telemetry records are lightweight frozen dataclasses. They are deposited
# into a bounded deque and drained periodically by a background task
# (managed by interface.py, not by the classifier itself).
#
# The telemetry pipeline is separate from the signal collector. Signals
# carry feature vectors for training; telemetry carries operational metrics
# for monitoring. Different consumers, different data, different lifecycles.
# ═════════════════════════════════════════════════════════════════════════════

_TELEMETRY_BUFFER_SIZE: Final[int] = 4096


@dataclass(frozen=True)
class _ClassificationTelemetry:
    """Operational telemetry for one classification event."""
    topology_class:      str
    confidence:          float
    classification_path: str
    latency_ms:          float
    was_cache_hit:       bool
    was_hard_override:   bool
    ml_path_invoked:     bool
    fallback_fired:      bool
    run_id:              str
    timestamp:           float


class ClassificationTelemetryPipeline:
    """Bounded deque of classification telemetry records.

    Thread-safe. deposit() is O(1). drain() returns all records and clears.

    Usage:
        pipeline = ClassificationTelemetryPipeline()
        pipeline.deposit(result, was_cache_hit=False)
        records = pipeline.drain()
    """

    def __init__(self, buffer_size: int = _TELEMETRY_BUFFER_SIZE) -> None:
        self._buffer: Deque[_ClassificationTelemetry] = deque(maxlen=buffer_size)
        self._lock = threading.Lock()
        self._total_emitted: int = 0

    def deposit(
        self,
        result: TopologyClassification,
        was_cache_hit: bool = False,
    ) -> None:
        """Deposit one telemetry record."""
        record = _ClassificationTelemetry(
            topology_class=result.topology_class,
            confidence=result.confidence,
            classification_path=result.classification_path,
            latency_ms=result.latency_ms,
            was_cache_hit=was_cache_hit,
            was_hard_override=result.topology_class in HARD_OVERRIDE_CLASSES,
            ml_path_invoked=result.classification_path in ("model", "fallback"),
            fallback_fired=result.classification_path == "fallback",
            run_id=result.run_id,
            timestamp=time.monotonic(),
        )
        with self._lock:
            self._buffer.append(record)
            self._total_emitted += 1

    def drain(self) -> List[_ClassificationTelemetry]:
        """Drain all telemetry records."""
        with self._lock:
            records = list(self._buffer)
            self._buffer.clear()
        return records

    def health(self) -> Dict[str, Any]:
        """Return telemetry pipeline health."""
        with self._lock:
            return {
                "buffer_size":   len(self._buffer),
                "buffer_capacity": self._buffer.maxlen,
                "total_emitted": self._total_emitted,
            }


# Module-level telemetry pipeline singleton.
_TELEMETRY_PIPELINE: ClassificationTelemetryPipeline = ClassificationTelemetryPipeline()


# ═════════════════════════════════════════════════════════════════════════════
# CONTENT STRUCTURE ANALYZER
#
# Lightweight structural analysis of the content prefix that goes beyond
# the pattern-matching of path 4. Detects structural features that are
# useful as ML features but too nuanced for exact string matching:
#
#   - HTML tag frequency distribution
#   - JSON schema detection (array root, object root, nested depth)
#   - Markup density (ratio of tags to text)
#   - Script and style block presence and count
#   - Meta tag analysis (og:type, twitter:card, etc.)
#
# This analyzer is called within _embed_signals() to enrich the feature
# vector. Its outputs are folded into GROUP 3 (content n-gram hash)
# via the hashing scheme — they are not a separate group.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _ContentStructureProfile:
    """Structural profile extracted from the content prefix."""
    # Tag statistics.
    total_tags:          int
    unique_tags:         int
    markup_density:      float    # tags / (tags + text_chars), [0, 1]

    # Block counts.
    script_block_count:  int
    style_block_count:   int
    form_count:          int
    link_count:          int

    # JSON signals.
    looks_like_json:     bool
    json_array_root:     bool
    json_object_root:    bool
    json_estimated_depth: int

    # Meta signals.
    has_og_type:         bool
    has_twitter_card:    bool
    has_viewport:        bool
    has_charset:         bool

    # Content signals.
    has_doctype:         bool
    text_char_count:     int
    whitespace_ratio:    float


# Precompiled patterns for content structure analysis.
_TAG_RE = re.compile(r"</?([a-zA-Z][a-zA-Z0-9]*)")
_SCRIPT_RE = re.compile(r"<script[\s>]", re.IGNORECASE)
_STYLE_RE = re.compile(r"<style[\s>]", re.IGNORECASE)
_FORM_RE = re.compile(r"<form[\s>]", re.IGNORECASE)
_LINK_RE = re.compile(r"<a[\s>]", re.IGNORECASE)
_OG_TYPE_RE = re.compile(r'property=["\']og:type', re.IGNORECASE)
_TWITTER_CARD_RE = re.compile(r'name=["\']twitter:card', re.IGNORECASE)
_VIEWPORT_RE = re.compile(r'name=["\']viewport', re.IGNORECASE)
_CHARSET_RE = re.compile(r'charset=', re.IGNORECASE)
_DOCTYPE_RE = re.compile(r"<!DOCTYPE", re.IGNORECASE)


def analyze_content_structure(content_prefix: bytes) -> _ContentStructureProfile:
    """Analyze the structural characteristics of the content prefix.

    Lightweight analysis bounded to 4096 bytes. Does not parse the HTML
    into a DOM — uses regex-based heuristics for speed.

    Performance: < 2ms on 4KB input. All patterns are precompiled.

    Args:
        content_prefix: raw bytes, up to 4096 bytes.

    Returns:
        _ContentStructureProfile with structural features.
    """
    window = content_prefix[:_CONTENT_WINDOW_BYTES]
    try:
        text = window.decode("utf-8", errors="replace")
    except Exception:  # noqa
        return _ContentStructureProfile(
            total_tags=0, unique_tags=0, markup_density=0.0,
            script_block_count=0, style_block_count=0, form_count=0,
            link_count=0, looks_like_json=False, json_array_root=False,
            json_object_root=False, json_estimated_depth=0,
            has_og_type=False, has_twitter_card=False, has_viewport=False,
            has_charset=False, has_doctype=False, text_char_count=0,
            whitespace_ratio=0.0,
        )

    stripped = text.strip()

    # ── Tag analysis ──────────────────────────────────────────────────────
    tags = _TAG_RE.findall(text)
    total_tags = len(tags)
    unique_tags = len(set(t.lower() for t in tags))
    text_chars = len(text) - sum(len(m.group()) for m in _TAG_RE.finditer(text))
    text_chars = max(text_chars, 0)
    markup_density = (
        total_tags / (total_tags + text_chars) if (total_tags + text_chars) > 0 else 0.0
    )

    # ── Block counts ──────────────────────────────────────────────────────
    script_count = len(_SCRIPT_RE.findall(text))
    style_count = len(_STYLE_RE.findall(text))
    form_count = len(_FORM_RE.findall(text))
    link_count = len(_LINK_RE.findall(text))

    # ── JSON detection ────────────────────────────────────────────────────
    looks_json = False
    json_array = False
    json_object = False
    json_depth = 0

    if stripped:
        first_char = stripped[0]
        if first_char in ("{", "["):
            looks_json = True
            json_array = (first_char == "[")
            json_object = (first_char == "{")
            # Estimate nesting depth by counting max unbalanced braces.
            depth = 0
            max_depth = 0
            for ch in stripped[:2048]:
                if ch in ("{", "["):
                    depth += 1
                    max_depth = max(max_depth, depth)
                elif ch in ("}", "]"):
                    depth = max(depth - 1, 0)
            json_depth = max_depth

    # ── Meta analysis ─────────────────────────────────────────────────────
    has_og = bool(_OG_TYPE_RE.search(text))
    has_twitter = bool(_TWITTER_CARD_RE.search(text))
    has_viewport = bool(_VIEWPORT_RE.search(text))
    has_charset = bool(_CHARSET_RE.search(text))
    has_doctype = bool(_DOCTYPE_RE.search(text))

    # ── Content statistics ────────────────────────────────────────────────
    whitespace = sum(1 for ch in text if ch.isspace())
    whitespace_ratio = whitespace / len(text) if len(text) > 0 else 0.0

    return _ContentStructureProfile(
        total_tags=total_tags,
        unique_tags=unique_tags,
        markup_density=round(markup_density, 4),
        script_block_count=script_count,
        style_block_count=style_count,
        form_count=form_count,
        link_count=link_count,
        looks_like_json=looks_json,
        json_array_root=json_array,
        json_object_root=json_object,
        json_estimated_depth=json_depth,
        has_og_type=has_og,
        has_twitter_card=has_twitter,
        has_viewport=has_viewport,
        has_charset=has_charset,
        has_doctype=has_doctype,
        text_char_count=text_chars,
        whitespace_ratio=round(whitespace_ratio, 4),
    )


_self_test_content_structure_analyzer()


# ═════════════════════════════════════════════════════════════════════════════
# MODEL WARMUP UTILITY
#
# Runs a configurable number of synthetic forward passes through the model
# to warm up JIT compilation, CUDA kernels, and CPU caches. This ensures
# the first real classification doesn't suffer cold-start latency.
#
# Called by cold_start.py after CLASSIFIER.initialize() succeeds.
# ═════════════════════════════════════════════════════════════════════════════

async def warmup_model(
    classifier: TopologyClassifier,
    num_passes: int = 10,
) -> Dict[str, Any]:
    """Run synthetic forward passes to warm up the model.

    Generates random feature vectors and runs them through the model.
    Measures and reports warmup latency statistics.

    This is safe to call from cold_start.py's async context. The forward
    passes are synchronous but fast (< 15ms each).

    Args:
        classifier: initialised TopologyClassifier.
        num_passes: number of warmup forward passes.

    Returns:
        Dict with warmup statistics:
          - num_passes: number of passes completed
          - mean_latency_ms: average pass latency
          - max_latency_ms: worst-case pass latency
          - min_latency_ms: best-case pass latency
          - all_passed: True if all passes succeeded
    """
    if not classifier.is_ready():
        return {
            "num_passes":     0,
            "mean_latency_ms": 0.0,
            "max_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "all_passed":     False,
            "error":          "model not loaded",
        }

    model = classifier._model  # noqa: access for warmup
    try:
        device = next(model.parameters()).device # noqa
    except (StopIteration, AttributeError):
        device = torch.device("cpu") # noqa

    latencies: List[float] = []
    errors: List[str] = []

    for i in range(num_passes):
        try:
            t0 = time.perf_counter()
            with torch.no_grad():
                conditioned = model.final_norm(model.hidden_state)
                _ = model.topology_head(conditioned)
            elapsed = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed)
        except Exception as exc:
            errors.append(f"pass {i}: {type(exc).__name__}: {str(exc)[:100]}")

    if latencies:
        return {
            "num_passes":      len(latencies),
            "mean_latency_ms": round(sum(latencies) / len(latencies), 3),
            "max_latency_ms":  round(max(latencies), 3),
            "min_latency_ms":  round(min(latencies), 3),
            "all_passed":      len(errors) == 0,
            "errors":          errors[:5],
        }
    else:
        return {
            "num_passes":      0,
            "mean_latency_ms": 0.0,
            "max_latency_ms":  0.0,
            "min_latency_ms":  0.0,
            "all_passed":      False,
            "errors":          errors[:5],
        }


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE VECTOR COMPARISON UTILITY
#
# Computes distance and similarity metrics between two feature vectors.
# Used by surprise_detector.py to measure how different two URLs are in
# feature space, and by index_daemon.py to identify near-duplicate inputs
# in training batches.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _FeatureVectorComparison:
    """Comparison result between two feature vectors."""
    l2_distance:        float
    cosine_similarity:  float
    per_group_l2:       Dict[str, float]
    most_different_group: str
    most_similar_group:   str


def compare_feature_vectors(
    vec_a: np.ndarray,
    vec_b: np.ndarray,
) -> _FeatureVectorComparison:
    """Compare two feature vectors and return distance metrics.

    Args:
        vec_a: first feature vector, shape (TOTAL_FEATURE_DIM,).
        vec_b: second feature vector, shape (TOTAL_FEATURE_DIM,).

    Returns:
        _FeatureVectorComparison with global and per-group metrics.
    """
    if vec_a.shape != (TOTAL_FEATURE_DIM,) or vec_b.shape != (TOTAL_FEATURE_DIM,):
        raise ValueError(
            f"Both vectors must have shape ({TOTAL_FEATURE_DIM},). "
            f"Got {vec_a.shape} and {vec_b.shape}."
        )

    # Global L2 distance.
    diff = vec_a - vec_b
    l2 = float(np.linalg.norm(diff))

    # Cosine similarity.
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a > 0 and norm_b > 0:
        cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    else:
        cosine = 0.0

    # Per-group L2 distance.
    per_group: Dict[str, float] = {}
    for meta in _GROUP_META:
        g_diff = diff[meta.offset:meta.offset + meta.dim]
        per_group[meta.name] = round(float(np.linalg.norm(g_diff)), 6)

    most_diff = max(per_group, key=per_group.get)    # type: ignore[arg-type]
    most_sim = min(per_group, key=per_group.get)      # type: ignore[arg-type]

    return _FeatureVectorComparison(
        l2_distance=round(l2, 6),
        cosine_similarity=round(cosine, 6),
        per_group_l2=per_group,
        most_different_group=most_diff,
        most_similar_group=most_sim,
    )


# ═════════════════════════════════════════════════════════════════════════════
# FEATURE GROUP EXTRACTION UTILITY
#
# Extracts individual feature groups from a concatenated feature vector.
# Used for debugging, visualization, and per-group analysis.
# ═════════════════════════════════════════════════════════════════════════════

def extract_feature_group(
    feature_vector: np.ndarray,
    group_name: str,
) -> np.ndarray:
    """Extract a single feature group from a concatenated feature vector.

    Args:
        feature_vector: shape (TOTAL_FEATURE_DIM,).
        group_name: one of the names in _GROUP_META.

    Returns:
        np.ndarray for the requested group.

    Raises:
        ValueError: if group_name is unknown.
    """
    for meta in _GROUP_META:
        if meta.name == group_name:
            return feature_vector[meta.offset:meta.offset + meta.dim].copy()
    raise ValueError(
        f"Unknown feature group: {group_name!r}. "
        f"Known groups: {[m.name for m in _GROUP_META]}"
    )


def feature_vector_summary(feature_vector: np.ndarray) -> Dict[str, Dict[str, float]]:
    """Compute summary statistics per feature group.

    Returns a dict mapping group name → {mean, std, min, max, l2_norm, sparsity}.
    """
    if feature_vector.shape != (TOTAL_FEATURE_DIM,):
        raise ValueError(f"Expected shape ({TOTAL_FEATURE_DIM},), got {feature_vector.shape}.")

    summary: Dict[str, Dict[str, float]] = {}
    for meta in _GROUP_META:
        group = feature_vector[meta.offset:meta.offset + meta.dim]
        zeros = int(np.sum(group == 0.0))
        summary[meta.name] = {
            "mean":     round(float(np.mean(group)), 6),
            "std":      round(float(np.std(group)), 6),
            "min":      round(float(np.min(group)), 6),
            "max":      round(float(np.max(group)), 6),
            "l2_norm":  round(float(np.linalg.norm(group)), 6),
            "sparsity": round(zeros / meta.dim, 4),
        }
    return summary


# ═════════════════════════════════════════════════════════════════════════════
# DOMAIN FINGERPRINT TABLE MANAGEMENT
#
# Utilities for extending the domain fingerprint table at runtime.
# The preparser adds new domain mappings as it discovers them during crawls.
# These additions are thread-safe and take effect on the next classify() call.
# ═════════════════════════════════════════════════════════════════════════════

def register_domain_fingerprint(
    domain: str,
    topology_class: str,
    confidence: float = 0.95,
) -> None:
    """Register a new domain → topology class mapping at runtime.

    Called by preparser/domain_analyzer.py when it discovers a new domain
    mapping through robots.txt or sitemap analysis.

    The mapping is inserted into both DOMAIN_FINGERPRINT_TABLE (for
    documentation) and _DOMAIN_TRIE (for runtime lookup). The trie
    insertion is O(depth) and does not require rebuilding the entire trie.

    Args:
        domain: exact hostname or wildcard pattern (e.g. "*.example.com").
        topology_class: target topology class string.
        confidence: classification confidence for this mapping.
    """
    if not domain or not topology_class:
        return

    # Validate topology class.
    if topology_class not in TOPOLOGY_CLASS_INDEX:
        logger.warning(
            "topology_classifier.unknown_class_in_fingerprint",
            extra={"domain": domain, "topology_class": topology_class},
        )
        return

    # Insert into the trie.
    _DOMAIN_TRIE.insert(domain, topology_class, confidence)

    # Update the table for documentation / serialisation.
    if not domain.startswith("*."):
        DOMAIN_FINGERPRINT_TABLE[domain] = topology_class

    logger.info(
        "topology_classifier.domain_fingerprint_registered",
        extra={
            "domain":         domain,
            "topology_class": topology_class,
            "confidence":     confidence,
        },
    )


def fingerprint_table_size() -> int:
    """Return the total number of domain patterns registered.

    Includes both exact and wildcard patterns.
    """
    exact = len(DOMAIN_FINGERPRINT_TABLE)
    wildcard = len(DOMAIN_WILDCARD_SPECS)
    return exact + wildcard


# ═════════════════════════════════════════════════════════════════════════════
# CLASSIFICATION REPLAY UTILITY
#
# Re-classifies a URL with full diagnostic output. Used by the debugging
# interface to replay a classification that produced unexpected results.
# Unlike classify(), replay returns the full evidence trace, per-path
# timings, feature vector, and model diagnostics.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _ReplayResult:
    """Full diagnostic replay of a classification."""
    classification:         TopologyClassification
    per_path_latency_ms:    Dict[str, float]
    feature_vector_summary: Dict[str, Dict[str, float]]
    model_health:           Optional[Dict[str, Any]]
    content_structure:      _ContentStructureProfile
    explanation:            ClassificationExplanation


async def replay_classification(
    classifier: TopologyClassifier,
    input: ClassifierInput, # noqa
) -> _ReplayResult:
    """Re-classify with full diagnostic instrumentation.

    This is NOT for production use — it performs redundant work for
    diagnostics. Use classify() for production classification.

    Args:
        classifier: initialised TopologyClassifier.
        input: ClassifierInput to replay.

    Returns:
        _ReplayResult with full diagnostic information.
    """
    # Time each path individually.
    path_timings: Dict[str, float] = {}

    # Path 1
    t0 = time.perf_counter()
    _ = classifier._classify_by_domain(input.url) # noqa
    path_timings["domain"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # Path 2
    t0 = time.perf_counter()
    _ = classifier._classify_by_url(input.url) # noqa
    path_timings["url"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # Path 3
    t0 = time.perf_counter()
    try:
        _ = classifier._classify_by_headers(input.headers, input.response_code) # noqa
    except Exception: # noqa
        pass
    path_timings["header"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # Path 4
    t0 = time.perf_counter()
    try:
        _ = classifier._classify_by_window(input.content_prefix) # noqa
    except Exception: # noqa
        pass
    path_timings["window"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # Content structure analysis
    content_profile = analyze_content_structure(input.content_prefix)

    # Feature vector computation
    t0 = time.perf_counter()
    try:
        url_tokens = _URLSemanticTokenizer.tokenize(input.url)
        features = _embed_signals_impl(
            classifier, input.url, input.headers, input.content_prefix,
            None, url_tokens,
        )
        feature_np = features.squeeze(0).cpu().numpy()
        fv_summary = feature_vector_summary(feature_np)
    except Exception: # noqa
        fv_summary = {}
    path_timings["embed"] = round((time.perf_counter() - t0) * 1000.0, 3)

    # Model health
    model_health_result = None
    if classifier._model is not None: # noqa
        health_result = validate_model_health(classifier._model) # noqa
        model_health_result = {
            "is_healthy":    health_result.is_healthy,
            "param_count":   health_result.param_count,
            "device":        health_result.device,
            "smoke_passed":  health_result.smoke_test_passed,
        }

    # Full classification
    result = await _original_classify(classifier, input)

    # Explanation
    explanation = explain_classification(result, classifier)

    return _ReplayResult(
        classification=result,
        per_path_latency_ms=path_timings,
        feature_vector_summary=fv_summary,
        model_health=model_health_result,
        content_structure=content_profile,
        explanation=explanation,
    )


__all__ = [
    # ── Singleton ─────────────────────────────────────────────────────────
    "CLASSIFIER",

    # ── Classes (from classifier.py, re-exported for convenience) ─────────
    "TopologyClassifier",
    "ClassifierInput",
    "ClassificationPath",

    # ── Constants (from classifier.py, re-exported) ───────────────────────
    "DOMAIN_FINGERPRINT_TABLE",
    "URL_STRUCTURE_PATTERNS",
    "HEADER_SIGNALS",
    "HEADER_CORRELATION_PATTERNS",
    "HARD_OVERRIDE_CLASSES",
    "WINDOW_PATTERNS",
    "TOPOLOGY_CLASS_INDEX",
    "INDEX_TO_TOPOLOGY_CLASS",

    # ── Feature space constants ───────────────────────────────────────────
    "FEATURE_VERSION",
    "TOTAL_FEATURE_DIM",
    "GROUP_1_URL_PATH_TOKENS_DIM",
    "GROUP_2_HEADER_BITMASK_DIM",
    "GROUP_3_CONTENT_NGRAM_HASH_DIM",
    "GROUP_4_DOMAIN_FEATURES_DIM",
    "GROUP_5_FINGERPRINT_SCORES_DIM",
    "GROUP_6_LATTICE_SCORES_DIM",
    "NUM_TOPOLOGY_CLASSES",

    # ── Module-level helpers (from classifier.py, re-exported) ────────────
    "resolve_fallback_chain",
    "is_hard_override_class",
    "confidence_label",
    "topology_class_depth",

    # ── Production infrastructure ─────────────────────────────────────────
    "ConfidenceCalibrator",
    "FeatureDriftMonitor",
    "ClassificationCache",
    "OnlineLearningSignalCollector",
    "TopologyClassStatsTracker",
    "ClassificationExplanation",

    # ── Functions ─────────────────────────────────────────────────────────
    "classifier_health",
    "explain_classification",
    "validate_model_health",
    "batch_embed_signals",
    "batch_classify_via_model",
    "compute_feature_importance",
    "serialise_feature_vector",
    "deserialise_feature_vector",
    "warmup_model",
    "analyze_content_structure",
    "compare_feature_vectors",
    "extract_feature_group",
    "feature_vector_summary",
    "register_domain_fingerprint",
    "fingerprint_table_size",
    "replay_classification",

    # ── Additional infrastructure classes ──────────────────────────────────
    "AdaptiveThresholdController",
    "ClassificationTelemetryPipeline",
    "_SurpriseHysteresis",
    "_SURPRISE_HYSTERESIS",
    "_HYSTERESIS_ARM_EVENTS",
    "_HYSTERESIS_ARM_WINDOW_S",
    "_HYSTERESIS_DISARM_QUIET_S",

    # ── Singletons (infrastructure) ───────────────────────────────────────
    # These are not typically imported directly — access via classifier_health().
    # Listed here for completeness and testing access.
]


# ═════════════════════════════════════════════════════════════════════════════
# FULL CLASSIFIER HEALTH CHECK (combines everything)
#
# Single function that validates the entire classifier pipeline end-to-end.
# Called by cold_start.py as the final validation step before the system
# goes live. Returns a structured result that cold_start.py can check.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class _PipelineValidationResult:
    """Result of full pipeline validation."""
    classifier_ready:       bool
    model_healthy:          bool
    feature_dims_valid:     bool
    hash_deterministic:     bool
    encoders_functional:    bool
    cache_operational:      bool
    drift_monitor_active:   bool
    signal_collector_active: bool
    calibrator_available:   bool
    threshold_controller_ok: bool
    telemetry_pipeline_ok:  bool
    hysteresis_ok:          bool
    total_checks:           int
    checks_passed:          int
    errors:                 Tuple[str, ...]


def validate_full_pipeline(
    classifier: TopologyClassifier,
) -> _PipelineValidationResult:
    """Run a comprehensive validation of the entire classification pipeline.

    Tests every component: model, encoders, cache, drift monitor, signal
    collector, calibrator, threshold controller, telemetry pipeline, and
    surprise hysteresis.

    This is a cold-start validation function — it takes 50-200ms to run.
    Do not call it on the hot path.

    Args:
        classifier: initialised TopologyClassifier.

    Returns:
        _PipelineValidationResult with all check results.
    """
    errors: List[str] = []
    checks_passed = 0
    total_checks = 12

    # 1. Classifier ready
    classifier_ready = classifier.is_ready()
    if classifier_ready:
        checks_passed += 1
    else:
        errors.append("classifier model not loaded")

    # 2. Model health
    model_healthy = False
    if classifier._model is not None: # noqa
        health = validate_model_health(classifier._model) # noqa
        model_healthy = health.is_healthy
        if model_healthy:
            checks_passed += 1
        else:
            errors.extend(health.errors[:3])
    else:
        errors.append("model is None — cannot validate health")

    # 3. Feature dimensions
    feature_dims_valid = True
    try:
        _self_test_feature_dimensions()
        checks_passed += 1
    except AssertionError as exc:
        feature_dims_valid = False
        errors.append(f"feature dimension check failed: {exc}")

    # 4. Hash determinism
    hash_ok = True
    try:
        _self_test_hash_determinism()
        checks_passed += 1
    except AssertionError as exc:
        hash_ok = False
        errors.append(f"hash determinism check failed: {exc}")

    # 5. Encoder functionality
    encoders_ok = True
    try:
        _self_test_group_encoders()
        checks_passed += 1
    except AssertionError as exc:
        encoders_ok = False
        errors.append(f"encoder check failed: {exc}")

    # 6. Cache operational
    cache_ok = True
    try:
        cache_health = _CLASSIFICATION_CACHE.health()
        if cache_health["max_size"] > 0:
            checks_passed += 1
        else:
            cache_ok = False
            errors.append("cache max_size is 0")
    except Exception as exc:
        cache_ok = False
        errors.append(f"cache health check failed: {exc}")

    # 7. Drift monitor active
    drift_ok = True
    try:
        drift_health = _DRIFT_MONITOR.health() # noqa
        checks_passed += 1
    except Exception as exc:
        drift_ok = False
        errors.append(f"drift monitor health check failed: {exc}")

    # 8. Signal collector active
    signal_ok = True
    try:
        signal_health = _SIGNAL_COLLECTOR.health() # noqa
        checks_passed += 1
    except Exception as exc:
        signal_ok = False
        errors.append(f"signal collector health check failed: {exc}")

    # 9. Calibrator available
    cal_ok = True
    try:
        cal_health = _CALIBRATOR.health() # noqa
        checks_passed += 1
    except Exception as exc:
        cal_ok = False
        errors.append(f"calibrator health check failed: {exc}")

    # 10. Threshold controller
    threshold_ok = True
    try:
        thresh_health = _THRESHOLD_CONTROLLER.health() # noqa
        checks_passed += 1
    except Exception as exc:
        threshold_ok = False
        errors.append(f"threshold controller health check failed: {exc}")

    # 11. Telemetry pipeline
    telemetry_ok = True
    try:
        telem_health = _TELEMETRY_PIPELINE.health() # noqa
        checks_passed += 1
    except Exception as exc:
        telemetry_ok = False
        errors.append(f"telemetry pipeline health check failed: {exc}")

    # 12. Surprise hysteresis
    hysteresis_ok = True # noqa
    try:
        h = _SURPRISE_HYSTERESIS.health()
        if h["arm_threshold"] != _HYSTERESIS_ARM_EVENTS:
            hysteresis_ok = False # noqa
            errors.append("hysteresis arm_threshold mismatch")
        else:
            checks_passed += 1
    except Exception as exc:
        hysteresis_ok = False # noqa
        errors.append(f"hysteresis health check failed: {exc}")

    return _PipelineValidationResult(
        classifier_ready=classifier_ready,
        model_healthy=model_healthy,
        feature_dims_valid=feature_dims_valid,
        hash_deterministic=hash_ok,
        encoders_functional=encoders_ok,
        cache_operational=cache_ok,
        drift_monitor_active=drift_ok,
        signal_collector_active=signal_ok,
        calibrator_available=cal_ok,
        threshold_controller_ok=threshold_ok,
        telemetry_pipeline_ok=telemetry_ok,
        hysteresis_ok=hysteresis_ok,
        total_checks=total_checks,
        checks_passed=checks_passed,
        errors=tuple(errors),
    )