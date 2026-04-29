"""
tag/world_model/mamba_router.py
================================
The World Latent Model's neural architecture.

This file defines MambaRouter — the nn.Module that maintains a compressed
latent representation of the structural topology of the web.  MambaRouter
is the satellite view: it sees domains, topology classes, friction patterns,
and structural relationships from above, not from within.

MambaRouter contains two operational modes on the same set of weights:

    readout()  — O(1) inference.  Projects hidden_state through output heads.
                 Never modifies hidden_state.  Enforced via torch.no_grad()
                 AND via zero assignment to self.hidden_state in the method body.
                 This is the critical path — called on every query.

    forward()  — Training only.  Processes token sequences through Mamba blocks.
                 Produces gradient-capable outputs for loss computation.
                 Hidden state update is explicit and gated behind a training lock.
                 Only reachable via WLMTrainingInterface (latent_model.py).
                 Direct calls from the inference path are an architecture violation.

The hidden_state buffer IS the MFT.  Reading the MFT = readout().
Writing the MFT = index_daemon running gradient steps and calling
update_hidden_state() through WLMTrainingInterface.

File scope:
    This file contains ONLY the nn.Module definition and its direct
    supporting infrastructure (constants, config, structural layer view).
    No tokenization (→ wlm_tokenizer.py).
    No output decoding into contracts (→ wlm_decoders.py).
    No bus integration, caching, or query orchestration (→ latent_model.py).

Bootstrap compatibility:
    initialize_store.py instantiates MambaRouter() with default arguments
    and calls .state_dict() to produce topology_router.pt.  The key names
    in the state dict must exactly match what mamba_ssm.Mamba produces for
    the inner SSM blocks.  This means we use mamba_ssm.Mamba directly in
    nn.ModuleList — no wrapper classes that would alter key prefixes.

Dependency direction:
    mamba_router.py → contracts.py (for type imports and phase constants)
    mamba_router.py → mamba_ssm (for Mamba block implementation)
    mamba_router.py → torch (for nn.Module, tensors, functional)
    Nothing else imports from mamba_router.py except latent_model.py and
    wlm_decoders.py.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import hashlib
import math # noqa
import threading
import time # noqa
from dataclasses import dataclass, field # noqa
from typing import ( # noqa
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa
from mamba_ssm import Mamba

from signal_kernel.contracts import ( # noqa
    TOPOLOGY_CLASSES,
    FALLBACK_TOPOLOGY_CLASS,
    PHASE_I,
    PHASE_II,
    PHASE_III,
    THETA_CONFIDENCE_II,
    THETA_CONFIDENCE_III,
    THETA_SURPRISE_DEFAULT,
    STORE_FILE_NAMES,
    TopologyClassStr,
    PhaseInt,
    ConfidenceFloat,
)


# ═════════════════════════════════════════════════════════════════════════════
# RENDER MODE CONSTANTS
#
# These determine when the WLM recommends headless (Playwright/JS) vs static
# (plain HTTP GET) fetching.  The threshold is intentionally biased toward
# static: headless is 10-50× more expensive.  The model must be confident
# that JS rendering is actually needed before recommending it.
# ═════════════════════════════════════════════════════════════════════════════

RENDER_THRESHOLD: float = 0.60
# sigmoid threshold for headless vs static.
# 0.60, not 0.50 — prefer static (faster, cheaper).
# The model's sigmoid output must exceed this before headless is recommended.
# Set above 0.5 to create an asymmetric cost: false-positive headless is
# more expensive than false-positive static.

ALWAYS_HEADLESS_CLASSES: FrozenSet[str] = frozenset({
    "SAAS_DOCS_WITH_CODE",
    # Lazy-loaded code blocks in Docusaurus, GitBook, and similar SSG
    # frameworks.  Static fetch returns placeholder divs with no content.
    # Headless is the only viable extraction strategy.

    "ECOMMERCE_PRODUCT",
    # React/Next.js rendered product pages.  Price, availability, and
    # variant selectors are client-side hydrated.  Static fetch returns
    # the SSR shell but misses dynamic pricing.

    "ECOMMERCE_PRODUCT_VARIANT",
    # Same rendering pipeline as ECOMMERCE_PRODUCT.  Variant selectors
    # (color, size) are JavaScript-driven state machines.
})

ALWAYS_STATIC_CLASSES: FrozenSet[str] = frozenset({
    "REST_API_JSON",
    # JSON API endpoint.  No HTML, no JavaScript, no rendering.
    # Headless fetch would add 5-10s of Playwright overhead for zero benefit.

    "REST_API_JSON_PAGINATED",
    # Same as REST_API_JSON — pagination is URL-parameter based, not JS.

    "JSON_LD_STRUCTURED",
    # Structured data embedded in <script type="application/ld+json">.
    # Present in the initial HTML response.  JS execution adds nothing.

    "WIKIPEDIA_ARTICLE",
    # Wikipedia serves complete HTML on first response.  No lazy loading.
    # The MediaWiki rendering pipeline is server-side.

    "AUTH_REDIRECT",
    # HTTP 302/301 to a login page.  The redirect itself is the signal.
    # Rendering the destination adds latency but no useful content.

    "CLOUDFLARE_CHALLENGE",
    # The challenge page is the signal — we detect it, not solve it.
    # Headless rendering of a Cloudflare challenge triggers JS challenges
    # that are intentionally difficult for automated browsers.

    "RATE_LIMITED",
    # HTTP 429 response.  The status code is the signal.
    # No rendering needed — the body is an error page.
})

HEADLESS_MIN_TIMEOUT_MS: int = 8_000
# Minimum timeout for headless render.
# Playwright needs time to:
#   1. Launch browser context (~500ms warm, ~2s cold)
#   2. Navigate to URL (~500ms-5s depending on page weight)
#   3. Wait for networkidle event (~1s-3s for SPAs)
# 8s gives most SPA frameworks time to render.
# This is a floor — the model can recommend higher timeouts.

TOR_THRESHOLD: float = 0.70
# sigmoid threshold for tor_required recommendation.
# High threshold — prefer clearnet (faster, more reliable).
# Tor adds 3-10s of circuit setup latency.  Only recommend when the
# model is very confident that the target requires Tor-level anonymity.


# ═════════════════════════════════════════════════════════════════════════════
# RATE LIMITING CONSTANTS
#
# Domain categories that require conservative request pacing.
# News sites and forums are operated by small teams that actively monitor
# crawl patterns and block aggressive bots.  Respect their infrastructure.
# ═════════════════════════════════════════════════════════════════════════════

CONSERVATIVE_RPS_CLASSES: FrozenSet[str] = frozenset({
    "NEWS_ARTICLE",
    # News publishers actively detect and block automated crawlers.
    # They sell their content — aggressive crawling is adversarial.

    "NEWS_ARTICLE_PAYWALLED",
    # Same operators as NEWS_ARTICLE, with additional bot detection
    # from paywall vendors (Piano, Pico, Leaky Paywall).

    "BLOG_POST",
    # Often hosted on shared infrastructure (Substack, Medium, Ghost).
    # Rate limits are per-platform, not per-domain.  One blog's aggressive
    # crawling can affect rate limits for every blog on the platform.

    "FORUM_THREAD",
    # Forum software (Discourse, phpBB, vBulletin) has aggressive
    # rate limiting.  Forums are community infrastructure — treat gently.
})

CONSERVATIVE_RPS_CEILING: float = 2.0
# Maximum requests/second for conservative topology classes.
# 2.0 rps = one request every 500ms.  Comfortably under the detection
# threshold for most news site bot-detection systems.
# The model can recommend lower than 2.0 — it cannot exceed 2.0 for
# these classes regardless of its raw output.

ZERO_RETRY_CLASSES: FrozenSet[str] = frozenset({
    "AUTH_REDIRECT",
    # Retrying an auth redirect just hits the same 302.
    # Without credentials, retrying is deterministically futile.

    "CLOUDFLARE_CHALLENGE",
    # Retrying a Cloudflare challenge with the same headers produces
    # the same challenge.  Retry wastes budget and may escalate to
    # a harder challenge or IP block.

    "RATE_LIMITED",
    # Retrying a 429 within the retry-after window is adversarial.
    # The WLM recommends zero retries; the backoff logic lives in
    # phantom.py where it respects Retry-After headers.
})


# ═════════════════════════════════════════════════════════════════════════════
# TRAVERSAL BIAS — PER-TOPOLOGY-CLASS OVERRIDE TABLE
#
# These biases are learned from real crawl data and baked in as constants.
# They override the model's raw output for specific topology classes where
# empirical data strongly favors a particular traversal strategy.
#
# Keys:
#   render_threshold — per-class override for RENDER_THRESHOLD
#   depth_bias       — additive offset to raw depth sigmoid before clamping
# ═════════════════════════════════════════════════════════════════════════════

TOPOLOGY_TRAVERSAL_BIAS: Dict[str, Dict[str, float]] = {
    "SAAS_DOCS": {
        "render_threshold": 0.45,
        # SaaS docs frequently use client-side rendering (Docusaurus, GitBook).
        # Lower threshold = more willing to recommend headless.
        # 0.45 < 0.50 means bias toward headless for SaaS docs.
        "depth_bias": 0.3,
        # SaaS docs have meaningful link structure: getting started → API ref
        # → guides.  Slightly deeper traversal captures the full doc tree.
    },
    "REST_API_JSON": {
        "render_threshold": 0.95,
        # Almost never headless.  API endpoints return raw JSON.
        # 0.95 means the model must be 95% confident to override ALWAYS_STATIC.
        # In practice this never fires because REST_API_JSON is in ALWAYS_STATIC.
        "depth_bias": -0.2,
        # API endpoints are flat.  Shallower traversal — each endpoint is
        # self-contained.  Pagination is URL-based, handled by phantom.
    },
    "WIKIPEDIA_ARTICLE": {
        "render_threshold": 0.99,
        # Wikipedia is static HTML.  Never headless.
        "depth_bias": 0.5,
        # Wikipedia has rich internal link structure.  Deeper traversal
        # captures related articles and category pages.
    },
    "ECOMMERCE_PRODUCT": {
        "render_threshold": 0.40,
        # Ecommerce sites are almost always React/Next.js.
        # Low threshold = strong bias toward headless.
        "depth_bias": -0.3,
        # Product pages are leaf nodes.  Shallow traversal — the product
        # page itself is the signal.  Category pages are separate topology.
    },
    "NEWS_ARTICLE": {
        "render_threshold": 0.55,
        # Many news sites are static but some use client-side paywalls
        # (Piano/Tinypass inject via JS).  Moderate threshold.
        "depth_bias": 0.0,
        # Default depth — news articles are self-contained but may have
        # meaningful "related articles" links.
    },
    "NEWS_ARTICLE_PAYWALLED": {
        "render_threshold": 0.50,
        # Paywalled articles are more likely to use JS paywall injection.
        # Lower threshold than NEWS_ARTICLE.
        "depth_bias": -0.1,
        # Slightly shallower — paywalled content often has truncated
        # article bodies with "subscribe to read more" gates.
    },
    "FORUM_THREAD": {
        "render_threshold": 0.55,
        # Discourse forums use Ember.js (headless required).
        # phpBB and vBulletin are static.  Moderate threshold.
        "depth_bias": 0.2,
        # Forum threads have pagination.  Slightly deeper to capture
        # multi-page threads.
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# SOURCE PRIORITY CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

SOURCE_PRIORITY_TOP_K: int = 10
# Return top 10 domains in source priority list.
# 10 is the sweet spot: enough diversity for robust fallback, few enough
# that phantom.py does not waste requests on low-probability sources.

SOURCE_PRIORITY_FALLBACK: List[str] = ["GENERIC_FALLBACK"]
# Returned when structural_layer.pt is missing or source head output is
# all zeros.  phantom.py treats GENERIC_FALLBACK as "use whatever URLs
# the AXIOM graph provided, in the order provided".  This is the safe
# default — it adds no intelligence but also introduces no risk.


# ═════════════════════════════════════════════════════════════════════════════
# HIDDEN STATE CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

HIDDEN_STATE_DIM: int = 256
# The MFT dimension.
# 256 is chosen because:
#   - Large enough to encode 18 topology classes × their friction/traversal
#     profiles × temporal evolution patterns.
#   - Small enough for O(1) readout: one (1, 256) → (256, 7) matmul = 1792
#     multiply-accumulate ops per head.  Microseconds on any modern CPU/GPU.
#   - Fits in a single 64-byte cache line on x86 when stored as float16,
#     or two cache lines as float32.  Cache-friendly for repeated readout.
#   - 2^8 — clean power of two.  Aligned to SIMD lane widths on AVX2 (8×32)
#     and AVX-512 (16×32).

HIDDEN_STATE_UPDATE_DELAY_MS: int = 500
# After index_daemon runs optimizer.step(), wait 500ms before updating
# hidden_state.  This delay allows in-flight readout() queries to complete
# against the current hidden state before the new state is visible.
#
# 500ms is not arbitrary:
#   - readout() completes in <2ms.  A 500ms window is 250× the worst-case
#     readout latency.  Any in-flight query will finish.
#   - store_watchdog debounce for topology_router.pt is also 500ms.  The
#     hidden state update and the file write are deliberately aligned so
#     that:  gradient step → 500ms delay → hidden state update → file write
#     → 500ms debounce → watchdog fires → all components reload.
#   - The total pipeline from gradient step to all-components-reloaded is
#     ~1100ms.  Acceptable for a system that trains in background.


# ═════════════════════════════════════════════════════════════════════════════
# MITIGATION STRATEGY CONSTANTS
#
# Strategy strings that wlm_decoders.py produces from friction head output.
# Defined here as constants to prevent string typos across files.
# ═════════════════════════════════════════════════════════════════════════════

MITIGATION_NONE:              str = "none"
MITIGATION_CLOUDFLARE_WAIT:   str = "cloudflare_wait"
MITIGATION_CLOUDFLARE_TOR:    str = "cloudflare_tor"
MITIGATION_PAYWALL_CACHED:    str = "paywall_cached"
MITIGATION_RATE_LIMIT_BACKOFF: str = "rate_limit_backoff"
MITIGATION_AUTH_SKIP:         str = "auth_skip"
MITIGATION_NONE_WITH_CAUTION: str = "none_with_caution"

VALID_MITIGATION_STRATEGIES: FrozenSet[str] = frozenset({
    MITIGATION_NONE,
    MITIGATION_CLOUDFLARE_WAIT,
    MITIGATION_CLOUDFLARE_TOR,
    MITIGATION_PAYWALL_CACHED,
    MITIGATION_RATE_LIMIT_BACKOFF,
    MITIGATION_AUTH_SKIP,
    MITIGATION_NONE_WITH_CAUTION,
})

# Friction probability thresholds for mitigation strategy derivation.
# Used by wlm_decoders._derive_mitigation_strategy().
# Defined here so mamba_router.py is the single source of truth for all
# thresholds that affect model output interpretation.

FRICTION_THRESHOLD_AUTH_SKIP:          float = 0.85
FRICTION_THRESHOLD_PAYWALL_CACHED:     float = 0.80
FRICTION_THRESHOLD_CLOUDFLARE_TOR_CF:  float = 0.70
FRICTION_THRESHOLD_CLOUDFLARE_TOR_BOT: float = 0.65
FRICTION_THRESHOLD_CLOUDFLARE_WAIT:    float = 0.60
FRICTION_THRESHOLD_RATE_LIMIT:         float = 0.65
FRICTION_THRESHOLD_CAUTION_MEAN:       float = 0.35


# ═════════════════════════════════════════════════════════════════════════════
# VOCABULARY RANGE CONSTANTS
#
# The WLM vocabulary is partitioned into four contiguous ranges.
# Each range has a fixed start and end index.  wlm_tokenizer.py maps
# discrete tokens into these ranges.  MambaRouter's embedding layer
# covers the full vocabulary; the ranges are semantic, not structural.
# ═════════════════════════════════════════════════════════════════════════════

VOCAB_TOPOLOGY_START:   int = 0
VOCAB_TOPOLOGY_END:     int = 17      # inclusive — 18 topology classes
VOCAB_TOPOLOGY_COUNT:   int = 18

VOCAB_STRUCTURAL_START: int = 18
VOCAB_STRUCTURAL_END:   int = 1023    # inclusive
VOCAB_STRUCTURAL_COUNT: int = 1006

VOCAB_DOMAIN_START:     int = 1024
VOCAB_DOMAIN_END:       int = 4095    # inclusive
VOCAB_DOMAIN_COUNT:     int = 3072

VOCAB_INTENT_START:     int = 4096
VOCAB_INTENT_END:       int = 8191    # inclusive
VOCAB_INTENT_COUNT:     int = 4096

VOCAB_TOTAL_SIZE:       int = 8192


# ═════════════════════════════════════════════════════════════════════════════
# ARCHITECTURE HYPERPARAMETER DEFAULTS
#
# These are the default values for MambaRouter's constructor.  They are
# defined as module-level constants so that:
#   1. initialize_store.py can reference them without importing MambaRouter
#   2. Tests can assert against them
#   3. They are documented in one place
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_VOCAB_SIZE:    int   = VOCAB_TOTAL_SIZE   # 8192
DEFAULT_D_MODEL:       int   = HIDDEN_STATE_DIM   # 256
DEFAULT_D_STATE:       int   = 64                 # SSM state dimension
DEFAULT_D_CONV:        int   = 4                  # local convolution width
DEFAULT_EXPAND:        int   = 2                  # inner expansion factor
DEFAULT_N_LAYERS:      int   = 4                  # four Mamba blocks
DEFAULT_N_TOPOLOGY:    int   = len(TOPOLOGY_CLASSES)  # 18
DEFAULT_N_SOURCE:      int   = 512                # source embedding dimension
DEFAULT_N_PHASE:       int   = 3                  # phases I, II, III
DEFAULT_DROPOUT:       float = 0.1
DEFAULT_MAX_SEQ_LEN:   int   = 512                # maximum tokens per event

# Output head dimensions — derived from architecture spec.
TRAVERSAL_HEAD_DIM:    int   = 7      # depth, render, rps, retry, timeout, tor, confidence
FRICTION_HEAD_DIM:     int   = 5      # cf, pw, rl, auth, bot
PHASE_HEAD_DIM:        int   = DEFAULT_N_PHASE  # 3

# Traversal output indices — used by wlm_decoders.py for tensor slicing.
TRAVERSAL_IDX_DEPTH:            int = 0
TRAVERSAL_IDX_RENDER_MODE:      int = 1
TRAVERSAL_IDX_RPS:              int = 2
TRAVERSAL_IDX_RETRY_BUDGET:     int = 3
TRAVERSAL_IDX_TIMEOUT_MS:       int = 4
TRAVERSAL_IDX_TOR_REQUIRED:     int = 5
TRAVERSAL_IDX_CONFIDENCE:       int = 6

# Friction output indices.
FRICTION_IDX_CLOUDFLARE:        int = 0
FRICTION_IDX_PAYWALL:           int = 1
FRICTION_IDX_RATE_LIMIT:        int = 2
FRICTION_IDX_AUTH_REDIRECT:     int = 3
FRICTION_IDX_BOT_DETECTION:     int = 4

# Traversal output range clamps — applied after activation.
DEPTH_MIN:          int   = 1
DEPTH_MAX:          int   = 5
RPS_MIN:            float = 0.1
RPS_MAX:            float = 100.0
RETRY_MIN:          int   = 0
RETRY_MAX:          int   = 5
TIMEOUT_MIN_MS:     int   = 1_000
TIMEOUT_MAX_MS:     int   = 30_000

# Model validation constants — used by _validate_architecture().
VALIDATION_TEST_TOPOLOGY: str = "SAAS_DOCS"
VALIDATION_TEST_SEQ_LEN:  int = 9
VALIDATION_TEST_BATCH:    int = 1


# ═════════════════════════════════════════════════════════════════════════════
# CONFIGURATION DATACLASSES
#
# Frozen after construction.  MambaRouter reads these at __init__ time and
# never re-reads them.  Changing model configuration requires constructing
# a new MambaRouter instance.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class VocabRange:
    """
    Defines one contiguous segment of the WLM vocabulary.

    Used by _validate_token_input() to ensure tokens fall within their
    expected range.  Construction validates that start <= end and that
    the count is consistent.  A VocabRange that exists is structurally valid.
    """
    name:  str
    start: int
    end:   int     # inclusive

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError(
                f"VocabRange '{self.name}': start must be >= 0, got {self.start}."
            )
        if self.end < self.start:
            raise ValueError(
                f"VocabRange '{self.name}': end ({self.end}) < start ({self.start})."
            )

    @property
    def count(self) -> int:
        """Number of tokens in this range (inclusive bounds)."""
        return self.end - self.start + 1

    def contains(self, token_id: int) -> bool:
        """True if token_id falls within this range."""
        return self.start <= token_id <= self.end

    def __repr__(self) -> str:
        return f"VocabRange({self.name!r}, [{self.start}, {self.end}], count={self.count})"


# Pre-built VocabRange instances for each vocabulary segment.
VOCAB_RANGE_TOPOLOGY   = VocabRange("topology",   VOCAB_TOPOLOGY_START,   VOCAB_TOPOLOGY_END)
VOCAB_RANGE_STRUCTURAL = VocabRange("structural",  VOCAB_STRUCTURAL_START, VOCAB_STRUCTURAL_END)
VOCAB_RANGE_DOMAIN     = VocabRange("domain",      VOCAB_DOMAIN_START,     VOCAB_DOMAIN_END)
VOCAB_RANGE_INTENT     = VocabRange("intent",      VOCAB_INTENT_START,     VOCAB_INTENT_END)

ALL_VOCAB_RANGES: Tuple[VocabRange, ...] = (
    VOCAB_RANGE_TOPOLOGY,
    VOCAB_RANGE_STRUCTURAL,
    VOCAB_RANGE_DOMAIN,
    VOCAB_RANGE_INTENT,
)


@dataclass(frozen=True)
class HeadConfig:
    """
    Configuration for a single output head of MambaRouter.

    Each head is a two-layer MLP: Linear → GELU → Dropout → Linear.
    The inner dimension is controlled by inner_factor × d_model.
    The output dimension is the number of raw outputs the head produces.

    use_dropout=False for the phase head — it has a smaller inner dimension
    (d_model // 2) and dropout on a narrow bottleneck is too aggressive.
    """
    name:         str
    output_dim:   int
    inner_factor: float = 1.0   # inner dim = d_model * inner_factor
    use_dropout:  bool  = True

    def __post_init__(self) -> None:
        if self.output_dim < 1:
            raise ValueError(
                f"HeadConfig '{self.name}': output_dim must be >= 1, got {self.output_dim}."
            )
        if self.inner_factor <= 0:
            raise ValueError(
                f"HeadConfig '{self.name}': inner_factor must be > 0, got {self.inner_factor}."
            )

    def inner_dim(self, d_model: int) -> int:
        """Compute inner dimension from base model dimension."""
        return max(1, int(d_model * self.inner_factor))


# Head configurations — matches the spec exactly.
HEAD_TOPOLOGY  = HeadConfig("topology",  output_dim=DEFAULT_N_TOPOLOGY, inner_factor=1.0)
HEAD_TRAVERSAL = HeadConfig("traversal", output_dim=TRAVERSAL_HEAD_DIM, inner_factor=1.0)
HEAD_FRICTION  = HeadConfig("friction",  output_dim=FRICTION_HEAD_DIM,  inner_factor=1.0)
HEAD_SOURCE    = HeadConfig("source",    output_dim=DEFAULT_N_SOURCE,   inner_factor=1.0)
HEAD_PHASE     = HeadConfig("phase",     output_dim=DEFAULT_N_PHASE,    inner_factor=0.5, use_dropout=False)

ALL_HEAD_CONFIGS: Tuple[HeadConfig, ...] = (
    HEAD_TOPOLOGY,
    HEAD_TRAVERSAL,
    HEAD_FRICTION,
    HEAD_SOURCE,
    HEAD_PHASE,
)


@dataclass(frozen=True)
class ModelConfig:
    """
    Complete hyperparameter configuration for a MambaRouter instance.

    Frozen after construction — hyperparameters do not change at runtime.
    MambaRouter stores a reference to this config for introspection and
    serialization.  Tests assert against it to detect accidental changes.

    All defaults match the spec.  Non-default configurations are used only
    in unit tests (smaller models for faster test cycles).
    """
    vocab_size:   int   = DEFAULT_VOCAB_SIZE
    d_model:      int   = DEFAULT_D_MODEL
    d_state:      int   = DEFAULT_D_STATE
    d_conv:       int   = DEFAULT_D_CONV
    expand:       int   = DEFAULT_EXPAND
    n_layers:     int   = DEFAULT_N_LAYERS
    n_topology:   int   = DEFAULT_N_TOPOLOGY
    n_source:     int   = DEFAULT_N_SOURCE
    n_phase:      int   = DEFAULT_N_PHASE
    dropout:      float = DEFAULT_DROPOUT
    max_seq_len:  int   = DEFAULT_MAX_SEQ_LEN

    def __post_init__(self) -> None:
        if self.vocab_size < 1:  # noqa
            raise ValueError(f"vocab_size must be >= 1, got {self.vocab_size}.")
        if self.d_model < 1:
            raise ValueError(f"d_model must be >= 1, got {self.d_model}.")
        if self.d_state < 1:
            raise ValueError(f"d_state must be >= 1, got {self.d_state}.")
        if self.d_conv < 1:
            raise ValueError(f"d_conv must be >= 1, got {self.d_conv}.")
        if self.expand < 1: # noqa
            raise ValueError(f"expand must be >= 1, got {self.expand}.")
        if self.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {self.n_layers}.")
        if self.n_topology < 1:
            raise ValueError(f"n_topology must be >= 1, got {self.n_topology}.")
        if self.n_source < 1:
            raise ValueError(f"n_source must be >= 1, got {self.n_source}.")
        if self.n_phase < 1:
            raise ValueError(f"n_phase must be >= 1, got {self.n_phase}.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}.")
        if self.max_seq_len < 1:
            raise ValueError(f"max_seq_len must be >= 1, got {self.max_seq_len}.")
        # d_model must be divisible by 2 for the phase head's d_model // 2 bottleneck.
        if self.d_model % 2 != 0:
            raise ValueError(
                f"d_model must be even (phase head uses d_model // 2 bottleneck), "
                f"got {self.d_model}."
            )

    @property
    def d_inner(self) -> int:
        """Inner dimension of Mamba blocks (d_model * expand)."""
        return self.d_model * self.expand

    @property
    def hidden_state_shape(self) -> Tuple[int, int]:
        """Shape of the persistent hidden_state buffer: (1, d_model)."""
        return (1, self.d_model) # noqa

    @property
    def traversal_dim(self) -> int:
        """Output dimension of the traversal head."""
        return TRAVERSAL_HEAD_DIM

    @property
    def friction_dim(self) -> int:
        """Output dimension of the friction head."""
        return FRICTION_HEAD_DIM

    @property
    def phase_bottleneck_dim(self) -> int:
        """Inner dimension of the phase head bottleneck."""
        return self.d_model // 2

    def estimated_parameter_count(self) -> int:
        """
        Rough parameter count estimate for VRAM planning.

        Does not include buffers (hidden_state, hidden_state_version).
        Accurate to within ~5% of the actual count — use model.parameter_count
        for the exact number after construction.
        """
        # Token embedding: vocab_size × d_model
        embed_params = self.vocab_size * self.d_model
        # Position embedding: max_seq_len × d_model
        pos_params = self.max_seq_len * self.d_model
        # Domain projection: d_model × d_model + d_model (bias)
        domain_proj_params = self.d_model * self.d_model + self.d_model
        # Intent projection: 256 × d_model + d_model (bias)
        intent_proj_params = HIDDEN_STATE_DIM * self.d_model + self.d_model
        # Mamba blocks: approximately 4 × d_model × d_inner per block
        # (A, B, C, D matrices plus convolution and gate parameters)
        mamba_per_block = 4 * self.d_model * self.d_inner + self.d_inner * self.d_conv
        mamba_params = self.n_layers * mamba_per_block
        # Layer norm: 2 × d_model (weight + bias)
        norm_params = 2 * self.d_model
        # Output heads: sum of per-head parameter counts
        head_params = 0
        for hc in ALL_HEAD_CONFIGS:
            inner = hc.inner_dim(self.d_model)
            # Linear(d_model, inner) + Linear(inner, output_dim) with biases
            head_params += (self.d_model * inner + inner) + (inner * hc.output_dim + hc.output_dim)

        return (
            embed_params + pos_params + domain_proj_params + intent_proj_params
            + mamba_params + norm_params + head_params
        )

    def estimated_vram_bytes(self, dtype_bytes: int = 4) -> int:
        """
        Estimated VRAM consumption in bytes for the model parameters.
        Default assumes float32 (4 bytes per parameter).
        For float16/bfloat16 inference, pass dtype_bytes=2.
        """
        return self.estimated_parameter_count() * dtype_bytes

    def estimated_vram_mb(self, dtype_bytes: int = 4) -> float:
        """Estimated VRAM in megabytes."""
        return self.estimated_vram_bytes(dtype_bytes) / (1024 * 1024)


# The canonical production configuration.
# MambaRouter() with no arguments uses this.
PRODUCTION_CONFIG = ModelConfig()


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURAL LAYER VIEW
#
# The structural layer is a secondary data artifact (structural_layer.pt)
# that the source head uses for domain ranking.  It is NOT part of the
# MambaRouter nn.Module — it is loaded separately by WorldLatentModel.
#
# Defined in mamba_router.py because the source head's output dimension
# must match the structural layer's embedding dimension, and that
# correspondence is part of the architecture specification.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class StructuralLayerView:
    """
    Read-only view of structural_layer.pt for source priority computation.

    source_matrix:    (n_domains, n_source) float32 tensor.
                      Row i is the embedding of domain_index[i].
                      Dot product with source head output gives relevance scores.

    domain_index:     List of domain name strings, in the same row order as
                      source_matrix.  domain_index[i] is the domain whose
                      embedding is source_matrix[i].

    intent_clusters:  Optional (n_clusters, n_source) tensor of intent cluster
                      centroids.  Used for intent-conditioned source priority.
                      None if the structural layer was produced before intent
                      clustering was added.

    cluster_domains:  Optional list of domain lists, one per cluster.
                      cluster_domains[i] contains the domains in cluster i.
                      Length matches intent_clusters.shape[0].

    Construction validates shape consistency.  A StructuralLayerView that
    exists has already passed its invariants.
    """
    source_matrix:   torch.Tensor
    domain_index:    List[str]
    intent_clusters: Optional[torch.Tensor] = None
    cluster_domains: Optional[List[List[str]]] = None

    def __post_init__(self) -> None:
        # source_matrix must be 2D
        if self.source_matrix.dim() != 2:
            raise ValueError(
                f"source_matrix must be 2D, got {self.source_matrix.dim()}D "
                f"with shape {tuple(self.source_matrix.shape)}."
            )
        # Row count must match domain_index length
        n_domains = self.source_matrix.shape[0]
        if n_domains != len(self.domain_index):
            raise ValueError(
                f"source_matrix has {n_domains} rows but domain_index has "
                f"{len(self.domain_index)} entries.  These must match."
            )
        # Embedding dimension should match DEFAULT_N_SOURCE for production models.
        # This is a warning-level check, not a hard error — test models may
        # use smaller dimensions.
        embed_dim = self.source_matrix.shape[1]
        if embed_dim != DEFAULT_N_SOURCE:
            import warnings
            warnings.warn(
                f"StructuralLayerView: source_matrix embedding dimension is {embed_dim}, "
                f"expected {DEFAULT_N_SOURCE}.  This is valid only for test models.",
                stacklevel=2,
            )
        # intent_clusters / cluster_domains consistency
        if self.intent_clusters is not None:
            if self.intent_clusters.dim() != 2:
                raise ValueError(
                    f"intent_clusters must be 2D, got {self.intent_clusters.dim()}D."
                )
            if self.intent_clusters.shape[1] != embed_dim:
                raise ValueError(
                    f"intent_clusters embedding dim ({self.intent_clusters.shape[1]}) "
                    f"does not match source_matrix embedding dim ({embed_dim})."
                )
            if self.cluster_domains is None:
                raise ValueError(
                    "cluster_domains must be provided when intent_clusters is present."
                )
            if len(self.cluster_domains) != self.intent_clusters.shape[0]:
                raise ValueError(
                    f"cluster_domains length ({len(self.cluster_domains)}) must match "
                    f"intent_clusters row count ({self.intent_clusters.shape[0]})."
                )
        elif self.cluster_domains is not None:
            raise ValueError(
                "cluster_domains provided without intent_clusters.  Both must be "
                "present or both absent."
            )

    @property
    def n_domains(self) -> int:
        """Total number of domains in the structural layer."""
        return self.source_matrix.shape[0]

    @property
    def embed_dim(self) -> int:
        """Embedding dimension (should be DEFAULT_N_SOURCE for production)."""
        return self.source_matrix.shape[1]

    @property
    def n_clusters(self) -> int:
        """Number of intent clusters, or 0 if intent clustering is absent."""
        if self.intent_clusters is None:
            return 0
        return self.intent_clusters.shape[0]

    @property
    def has_intent_clusters(self) -> bool:
        """True if intent clustering data is available."""
        return self.intent_clusters is not None

    def domain_at(self, index: int) -> str:
        """Return the domain name at the given row index."""
        return self.domain_index[index]

    def score_sources(
        self,
        source_embedding: torch.Tensor,
        top_k: int = SOURCE_PRIORITY_TOP_K,
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Compute dot-product similarity between source_embedding and all domain
        embeddings.  Returns (scores_tensor, domain_names) for top-K domains.

        source_embedding: (embed_dim,) tensor from the source head.
        Returns:
            scores:  (min(top_k, n_domains),) float tensor of similarity scores.
            domains: List of domain names in score-descending order.
        """
        # Dot product similarity: (n_domains, embed_dim) @ (embed_dim, 1) → (n_domains, 1)
        scores = torch.matmul(
            self.source_matrix,
            source_embedding.unsqueeze(1),
        ).squeeze(1)  # (n_domains,)

        k = min(top_k, self.n_domains)
        top_k_result = torch.topk(scores, k)
        top_domains = [self.domain_index[i] for i in top_k_result.indices.tolist()]
        return top_k_result.values, top_domains


class EmptyStructuralLayer:
    """
    Stand-in for StructuralLayerView when structural_layer.pt does not exist.

    Returns empty results for all source priority queries.  This is the
    expected state during cold start before the first preparse cycle completes.
    The WLM functions correctly without a structural layer — it just cannot
    rank source domains, so it returns SOURCE_PRIORITY_FALLBACK.
    """

    @property
    def n_domains(self) -> int:
        return 0

    @property
    def embed_dim(self) -> int:
        return DEFAULT_N_SOURCE

    @property
    def n_clusters(self) -> int:
        return 0

    @property
    def has_intent_clusters(self) -> bool:
        return False

    @property
    def source_matrix(self) -> None:
        return None

    @property
    def domain_index(self) -> List[str]:
        return []

    def domain_at(self, index: int) -> str:
        raise IndexError("EmptyStructuralLayer has no domains.")

    def score_sources( # noqa
        self,
        source_embedding: torch.Tensor, # noqa
        top_k: int = SOURCE_PRIORITY_TOP_K, # noqa
    ) -> Tuple[torch.Tensor, List[str]]:
        """Returns empty results — no structural layer loaded."""
        return torch.tensor([], dtype=torch.float32), list(SOURCE_PRIORITY_FALLBACK)


# ═════════════════════════════════════════════════════════════════════════════
# READOUT RESULT
#
# The raw tensor container returned by MambaRouter.readout().
# wlm_decoders.py consumes this and produces WLMResponse contracts.
# Defined here because it is part of MambaRouter's return type contract.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class ReadoutResult:
    """
    Raw tensor outputs from MambaRouter.readout().

    All tensors are detached, on CPU, and have no gradient history.
    wlm_decoders.py applies activation functions and range clamping to
    convert these into semantically meaningful TraversalPolicy and
    FrictionForecast contracts.

    topology_class is passed through for decoder convenience — decoders
    need it for per-class bias lookups and ALWAYS_HEADLESS/ALWAYS_STATIC
    overrides.

    Fields:
        traversal_raw:    (7,)   — raw traversal head output
        friction_raw:     (5,)   — raw friction head output
        source_raw:       (n_source,) — raw source head embedding
        phase_raw:        (3,)   — raw phase head logits
        topology_class:   str    — passthrough for decoder context
        hidden_state_version: int — version of hidden_state used for readout
    """
    traversal_raw:        torch.Tensor
    friction_raw:         torch.Tensor
    source_raw:           torch.Tensor
    phase_raw:            torch.Tensor
    topology_class:       str
    hidden_state_version: int

    def __post_init__(self) -> None:
        # Validate tensor dimensions.
        if self.traversal_raw.dim() != 1 or self.traversal_raw.shape[0] != TRAVERSAL_HEAD_DIM:
            raise ValueError(
                f"traversal_raw must be ({TRAVERSAL_HEAD_DIM},), "
                f"got {tuple(self.traversal_raw.shape)}."
            )
        if self.friction_raw.dim() != 1 or self.friction_raw.shape[0] != FRICTION_HEAD_DIM:
            raise ValueError(
                f"friction_raw must be ({FRICTION_HEAD_DIM},), "
                f"got {tuple(self.friction_raw.shape)}."
            )
        if self.source_raw.dim() != 1:
            raise ValueError(
                f"source_raw must be 1D, got {self.source_raw.dim()}D."
            )
        if self.phase_raw.dim() != 1 or self.phase_raw.shape[0] != PHASE_HEAD_DIM:
            raise ValueError(
                f"phase_raw must be ({PHASE_HEAD_DIM},), "
                f"got {tuple(self.phase_raw.shape)}."
            )
        # All tensors must be detached.
        for name, tensor in [
            ("traversal_raw", self.traversal_raw),
            ("friction_raw", self.friction_raw),
            ("source_raw", self.source_raw),
            ("phase_raw", self.phase_raw),
        ]:
            if tensor.requires_grad:
                raise ValueError(
                    f"ReadoutResult.{name} has requires_grad=True.  "
                    "All readout tensors must be detached from the computation graph."
                )


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD RESULT
#
# The raw tensor container returned by MambaRouter.forward().
# Used by WLMTrainingInterface for loss computation and hidden state update.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ForwardResult:
    """
    Raw tensor outputs from MambaRouter.forward().

    Unlike ReadoutResult, these tensors MAY have gradient history when
    update_hidden=False (the training path where gradients flow).
    When update_hidden=True, new_hidden is detached before being stored
    as the updated hidden state.

    Fields:
        topology_logits:  (B, n_topology) — for topology prediction loss
        traversal_raw:    (B, 7)          — for traversal prediction loss
        friction_raw:     (B, 5)          — for friction prediction loss
        source_embedding: (B, n_source)   — for source ranking loss
        phase_logits:     (B, n_phase)    — for phase prediction loss
        new_hidden:       (B, d_model)    — candidate hidden state from last token
    """
    topology_logits:  torch.Tensor
    traversal_raw:    torch.Tensor
    friction_raw:     torch.Tensor
    source_embedding: torch.Tensor
    phase_logits:     torch.Tensor
    new_hidden:       torch.Tensor


# ═════════════════════════════════════════════════════════════════════════════
# MAMBA ROUTER — THE NN.MODULE
# ═════════════════════════════════════════════════════════════════════════════

class MambaRouter(nn.Module):
    """
    The World Latent Model's neural architecture.

    MambaRouter is a Mamba State Space Model with persistent hidden state
    that encodes compressed structural knowledge of the web.  The hidden
    state IS the MFT (Master File Table) — it is what AXIOM knows about
    how the web is structured.

    Architecture overview:
        Input:  token sequences (topology class + structural primitives +
                domain hash + intent tokens) → embedding → positional encoding

        Body:   4 × Mamba SSM blocks with residual connections and LayerNorm.
                Each block selectively updates the representation through
                input-dependent A (forgetting), B (updating), C (readout)
                matrices.  The selective mechanism learns which domain events
                are informationally valuable.

        Output: 5 projection heads:
                  topology_head  → (n_topology,)   topology class logits
                  traversal_head → (7,)            traversal policy parameters
                  friction_head  → (5,)            friction probability parameters
                  source_head    → (n_source,)     source domain embedding
                  phase_head     → (n_phase,)      phase prediction logits

        State:  hidden_state buffer (1, d_model) — the MFT.
                Registered as a buffer: saved in state_dict, not a gradient
                parameter.  Updated ONLY by WLMTrainingInterface after
                gradient steps.  NEVER updated by readout().

    Operational modes:

        readout(topology_class, intent_vector=None) → ReadoutResult
            O(1) inference.  Projects hidden_state through output heads.
            If intent_vector is provided, it conditions the readout by
            adding a projected intent embedding to the hidden state
            (in a LOCAL VARIABLE — self.hidden_state is never touched).
            Called on every query.  Must be fast.

        forward(token_sequence, update_hidden=False) → ForwardResult
            Training-time forward pass.  Processes token sequences through
            all Mamba blocks.  Produces gradient-capable outputs.
            Only callable when the training lock is held.
            Only WLMTrainingInterface holds the training lock.

    Training lock:
        forward() requires the training lock to be acquired first via
        acquire_training_lock().  This prevents accidental forward() calls
        from the inference path.  The lock is a threading.Lock — it is
        NOT an asyncio lock because gradient computation happens in a
        thread pool, not on the event loop.

    Bootstrap compatibility:
        MambaRouter() with no arguments produces a valid model with default
        hyperparameters.  Calling .state_dict() on the default model
        produces topology_router.pt with key names that exactly match
        what mamba_ssm.Mamba produces.  This is how initialize_store.py
        bootstraps the store — it does not build tensors by hand.
    """

    def __init__(
        self,
        vocab_size:   int   = DEFAULT_VOCAB_SIZE,
        d_model:      int   = DEFAULT_D_MODEL,
        d_state:      int   = DEFAULT_D_STATE,
        d_conv:       int   = DEFAULT_D_CONV,
        expand:       int   = DEFAULT_EXPAND,
        n_layers:     int   = DEFAULT_N_LAYERS,
        n_topology:   int   = DEFAULT_N_TOPOLOGY,
        n_source:     int   = DEFAULT_N_SOURCE,
        n_phase:      int   = DEFAULT_N_PHASE,
        dropout:      float = DEFAULT_DROPOUT,
        max_seq_len:  int   = DEFAULT_MAX_SEQ_LEN,
    ) -> None:
        super().__init__()

        # ── Validate and store configuration ──────────────────────────────
        self.config = ModelConfig(
            vocab_size=vocab_size,
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_layers=n_layers,
            n_topology=n_topology,
            n_source=n_source,
            n_phase=n_phase,
            dropout=dropout,
            max_seq_len=max_seq_len,
        )
        # ModelConfig.__post_init__ validates all hyperparameters.
        # If we reach here, config is structurally valid.

        # ── Training lock ─────────────────────────────────────────────────
        # forward() requires this lock.  readout() does not.
        # WLMTrainingInterface acquires it; nothing else should.
        self._training_lock = threading.Lock()
        self._training_lock_holder: Optional[str] = None

        # ── Build architecture ────────────────────────────────────────────
        self._build_embeddings()
        self._build_mamba_stack()
        self._build_output_heads()
        self._register_state_buffers()

        # ── Post-construction validation ──────────────────────────────────
        # Verify that the architecture produces outputs with expected shapes.
        # This catches dimensional mismatches before the first real query.
        self._validate_architecture()

    # ── Architecture builders ─────────────────────────────────────────────
    # Separated from __init__ for clarity.  Each builder is responsible for
    # one section of the architecture.  They are called exactly once.

    def _build_embeddings(self) -> None:
        """
        Build the input encoding layers.

        The embedding pipeline converts discrete token IDs into dense vectors
        in model space.  Three embedding types handle different input signals:

        token_embedding:     Maps vocabulary tokens to d_model vectors.
                             Vocabulary covers topology classes (0-17),
                             structural primitives (18-1023), domain hashes
                             (1024-4095), and intent tokens (4096-8191).

        position_embedding:  Learned positional encoding for within-sequence
                             position.  Max sequence length is 512 tokens.
                             Domain topology events produce 11-43 tokens;
                             query-time inputs produce 9 tokens.  512 is
                             the upper bound.

        domain_projection:   Projects domain-specific features into model
                             space.  Applied after token + position embedding
                             to add a domain-specific bias.

        intent_projection:   Projects the 256-dimensional intent vector from
                             the AXIOM graph into model space.  Used by
                             readout() for intent-conditioned policy output.
        """
        cfg = self.config

        self.token_embedding = nn.Embedding(
            num_embeddings=cfg.vocab_size,
            embedding_dim=cfg.d_model,
        )
        # Initialization: default nn.Embedding uses N(0, 1).
        # For a vocabulary of 8192, this is fine — Xavier-scale is sqrt(1/8192)
        # ≈ 0.011, but N(0, 1) works because the Mamba blocks normalize
        # internally.  We do not override initialization here.

        self.position_embedding = nn.Embedding(
            num_embeddings=cfg.max_seq_len,
            embedding_dim=cfg.d_model,
        )
        # Learned positional encoding.  Fixed sinusoidal encoding was
        # considered and rejected: the WLM's token sequences are short
        # (11-43 tokens) and their structure is fixed, not free-form.
        # Learned embeddings can capture the specific positional semantics
        # of the WLM vocabulary layout (topology token is always first,
        # structural primitives follow in a fixed order, etc.).

        self.domain_projection = nn.Linear(cfg.d_model, cfg.d_model)
        # Adds a domain-specific bias to the embedded representation.
        # Applied in forward() after token + position embedding sum.
        # This projection allows the model to learn domain-dependent
        # feature interactions that the token embedding alone cannot
        # capture (e.g., a domain hash token followed by a CDN token
        # has different meaning than the same CDN token after a
        # different domain hash).

        self.intent_projection = nn.Linear(HIDDEN_STATE_DIM, cfg.d_model)
        # Projects the AXIOM graph's 256-dim intent vector into model space.
        # Used in readout() for intent-conditioned output:
        #   conditioned_state = hidden_state + intent_projection(intent_vector)
        # The intent vector comes from the AXIOM controller's query embedding.
        # It encodes the semantic direction of the user's information need.
        # Input dimension is HIDDEN_STATE_DIM (256), not d_model, because
        # the intent vector has a fixed dimensionality regardless of model size.

        self.embedding_dropout = nn.Dropout(cfg.dropout)
        # Applied after the full embedding pipeline (token + position +
        # domain projection) before feeding into the Mamba stack.
        # Regularizes the input representation during training.
        # Disabled at eval time via model.eval().

    def _build_mamba_stack(self) -> None:
        """
        Build the Mamba SSM block stack with residual connections and norms.

        The stack consists of n_layers Mamba blocks.  Each block implements
        the selective state space model update:

            H_t = A(x_t) ⊙ H_{t-1} + B(x_t) ⊙ x_t
            y_t = C(x_t) ⊙ H_t

        Where A, B, C are input-dependent matrices learned by each block.
        The selectivity mechanism learns which domain events are informationally
        valuable for updating the structural representation.

        Block specialization (emergent, not enforced):
            Block 0: Low-level structural patterns (CDN, CMS, render reqs)
            Block 1: Topology class relationships (parent/child patterns)
            Block 2: Friction and traversal patterns
            Block 3: Source priority and quality patterns

        This is a soft assignment — the model learns these divisions through
        gradient flow.  We do not constrain which block learns what.

        Residual connections are applied around each Mamba block.
        Pre-norm architecture: LayerNorm before each block, not after.
        This follows the modern convention (GPT-2 style) that stabilizes
        training for deep SSM stacks.

        CRITICAL: We use mamba_ssm.Mamba directly in the ModuleList.
        No wrapper classes.  This ensures that state_dict key names are:
            blocks.0.{mamba_ssm internal keys}
            blocks.1.{mamba_ssm internal keys}
            ...
        which is exactly what initialize_store.py expects.
        """
        cfg = self.config

        # Pre-block layer norms — one per Mamba block.
        # Applied before each block in the residual path.
        self.block_norms = nn.ModuleList([
            nn.LayerNorm(cfg.d_model)
            for _ in range(cfg.n_layers)
        ])

        # Mamba SSM blocks — the core of the architecture.
        # Each block is a standalone mamba_ssm.Mamba instance.
        # d_model:  model dimension (256)
        # d_state:  SSM state dimension (64) — the compression factor.
        #           Higher d_state = richer recurrence at the cost of memory.
        #           64 is the sweet spot for structural pattern encoding.
        # d_conv:   local convolution width (4) — captures local token
        #           dependencies within a 4-token window before the SSM.
        # expand:   inner expansion factor (2) — the Mamba block internally
        #           projects to d_model × expand (512) before the SSM,
        #           then projects back.  Similar to FFN expansion in transformers.
        self.blocks = nn.ModuleList([
            Mamba(
                d_model=cfg.d_model,
                d_state=cfg.d_state,
                d_conv=cfg.d_conv,
                expand=cfg.expand,
            )
            for _ in range(cfg.n_layers)
        ])

        # Post-block dropout — applied to each block's output before
        # the residual addition.  Regularizes inter-block communication.
        self.block_dropouts = nn.ModuleList([
            nn.Dropout(cfg.dropout)
            for _ in range(cfg.n_layers)
        ])

        # Final layer norm — applied after the last block.
        # Normalizes the representation before head projection.
        self.final_norm = nn.LayerNorm(cfg.d_model)

    def _build_output_heads(self) -> None:
        """
        Build the five output projection heads.

        Each head is a two-layer MLP that projects the d_model representation
        into a task-specific output space.  Architecture per head:

            Linear(d_model, inner_dim) → GELU → [Dropout] → Linear(inner_dim, output_dim)

        GELU activation is used throughout (not ReLU) because GELU provides
        smoother gradients that work better with the Mamba block's output
        distribution.  The SSM produces non-sparse activations that benefit
        from GELU's non-zero gradient region near zero.

        Head-specific notes:

        topology_head:   18-class logits.  Used at training time for topology
                         prediction loss (cross-entropy against known class).
                         Not the primary output — TraversalPolicy is.

        traversal_head:  7 raw outputs.  wlm_decoders.py applies per-output
                         activations (sigmoid, softplus) and range clamping.
                         This is the primary output that drives phantom.py.

        friction_head:   5 raw outputs.  All sigmoid-bounded to [0, 1] by
                         wlm_decoders.py.  Represents probability of each
                         friction type for the queried topology class.

        source_head:     n_source (512) dimensional embedding.  Used for
                         dot-product similarity against structural_layer.pt's
                         source_matrix.  This is not a classification head —
                         it is a learned embedding projection.

        phase_head:      3-class logits with a narrower bottleneck (d_model // 2).
                         The phase prediction task is simpler than the others
                         (only 3 classes with clear phase semantics) so a
                         smaller inner dimension is appropriate.  No dropout
                         on this head — the narrow bottleneck is sufficient
                         regularization.
        """
        cfg = self.config

        # ── Topology head ─────────────────────────────────────────────────
        self.topology_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_topology),
        )

        # ── Traversal head ────────────────────────────────────────────────
        self.traversal_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, TRAVERSAL_HEAD_DIM),
        )

        # ── Friction head ─────────────────────────────────────────────────
        self.friction_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, FRICTION_HEAD_DIM),
        )

        # ── Source head ───────────────────────────────────────────────────
        self.source_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.n_source),
        )

        # ── Phase head ────────────────────────────────────────────────────
        # Narrower bottleneck: d_model → d_model//2 → n_phase.
        # No dropout — the bottleneck is sufficient regularization for
        # a 3-class prediction task.
        self.phase_head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model // 2),
            nn.GELU(),
            nn.Linear(cfg.d_model // 2, cfg.n_phase),
        )

    def _register_state_buffers(self) -> None:
        """
        Register persistent buffers that are saved in state_dict but are
        NOT gradient parameters.

        hidden_state:
            The MFT.  Shape (1, d_model).  Accumulates across every domain
            event.  Query 1,000,000 is informed by queries 1 through 999,999.
            Updated ONLY by WLMTrainingInterface.update_hidden_state()
            after gradient steps.  NEVER updated by readout().

            Registered as a buffer (not a parameter) because:
            1. It must be saved in state_dict (persist across checkpoints).
            2. It must NOT receive gradients during backward().
            3. It must NOT appear in model.parameters().
            4. It must be moved to the correct device by model.to(device).

        hidden_state_version:
            Monotonic counter incremented by WLMTrainingInterface on every
            hidden state update.  Used to detect stale reads and to track
            compounding progress in logs.  Long tensor — can count to 2^63.
        """
        cfg = self.config

        self.register_buffer(
            "hidden_state",
            torch.zeros(1, cfg.d_model, dtype=torch.float32),
        )
        # Shape: (1, d_model) — always.
        # The batch dimension of 1 is permanent.  readout() operates on
        # this single state vector.  If multiple hidden states are needed
        # (they are not), that would require a fundamentally different
        # architecture.

        self.register_buffer(
            "hidden_state_version",
            torch.tensor(0, dtype=torch.long),
        )
        # Starts at 0.  Incremented by 1 on every update_hidden_state() call.
        # A production system after 1 week of continuous crawl will have
        # version ~10,000 (one gradient step per ~60 seconds).
        # Long tensor: overflow at 2^63 ≈ 9.2 × 10^18.  Not a concern.

    # ── Input validation ──────────────────────────────────────────────────

    def _validate_token_input(
        self,
        token_sequence: torch.Tensor,
        context: str = "forward",
    ) -> None:
        """
        Validate that a token sequence is within expected ranges.

        Checks:
        1. Tensor is 2D with shape (batch, seq_len).
        2. seq_len <= max_seq_len.
        3. All token IDs are in [0, vocab_size).
        4. Tensor dtype is integer (long or int).

        Raises ValueError with detailed context if any check fails.
        This is a development-time safety check, not a hot-path gate.
        Called in forward() but not in readout() (which has no token input).
        """
        if token_sequence.dim() != 2:
            raise ValueError(
                f"[{context}] token_sequence must be 2D (batch, seq_len), "
                f"got {token_sequence.dim()}D with shape {tuple(token_sequence.shape)}."
            )

        batch_size, seq_len = token_sequence.shape

        if seq_len > self.config.max_seq_len:
            raise ValueError(
                f"[{context}] token_sequence seq_len={seq_len} exceeds "
                f"max_seq_len={self.config.max_seq_len}.  "
                "Truncate or split the sequence before calling forward()."
            )

        if seq_len == 0:
            raise ValueError(
                f"[{context}] token_sequence has seq_len=0.  "
                "A zero-length sequence produces no hidden state candidate."
            )

        if not token_sequence.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            raise ValueError(
                f"[{context}] token_sequence dtype must be integer, "
                f"got {token_sequence.dtype}.  "
                "Token IDs must be discrete indices, not continuous values."
            )

        # Range check — only on CPU tensors (GPU tensors require a sync).
        if token_sequence.device.type == "cpu":
            min_val = token_sequence.min().item()
            max_val = token_sequence.max().item()
            if min_val < 0:
                raise ValueError(
                    f"[{context}] token_sequence contains negative token ID {min_val}.  "
                    "All token IDs must be >= 0."
                )
            if max_val >= self.config.vocab_size:
                raise ValueError(
                    f"[{context}] token_sequence contains token ID {max_val} "
                    f">= vocab_size {self.config.vocab_size}.  "
                    "Token ID is out of vocabulary range."
                )

    def _validate_intent_vector( # noqa
        self,
        intent_vector: torch.Tensor,
    ) -> None:
        """
        Validate that an intent vector has the expected shape and dtype.

        Intent vectors are 256-dimensional float tensors from the AXIOM graph's
        query embedding.  They are projected into model space by intent_projection.
        """
        if intent_vector.dim() != 2:
            raise ValueError(
                f"intent_vector must be 2D (1, {HIDDEN_STATE_DIM}), "
                f"got {intent_vector.dim()}D with shape {tuple(intent_vector.shape)}."
            )
        if intent_vector.shape[0] != 1:
            raise ValueError(
                f"intent_vector batch dimension must be 1, "
                f"got {intent_vector.shape[0]}.  "
                "readout() operates on a single intent vector."
            )
        if intent_vector.shape[1] != HIDDEN_STATE_DIM:
            raise ValueError(
                f"intent_vector feature dimension must be {HIDDEN_STATE_DIM}, "
                f"got {intent_vector.shape[1]}.  "
                "Intent vectors have a fixed dimensionality."
            )

    # ── Architecture validation ───────────────────────────────────────────

    def _validate_architecture(self) -> None:
        """
        Post-construction structural validation.

        Runs a single forward pass with dummy input to verify that all
        layers produce outputs with expected shapes.  Called once at the
        end of __init__().

        This catches dimensional mismatches, misconfigured heads, and
        Mamba block initialization failures before the model is used.

        Does NOT validate semantic correctness — only structural correctness.
        A model that passes this check produces validly shaped outputs.
        Whether the values are useful depends on training.
        """
        cfg = self.config

        # Create a dummy token sequence: (1, VALIDATION_TEST_SEQ_LEN)
        dummy_tokens = torch.zeros(
            VALIDATION_TEST_BATCH,
            VALIDATION_TEST_SEQ_LEN,
            dtype=torch.long,
        )

        # Run forward pass with training lock held.
        self._training_lock.acquire()
        self._training_lock_holder = "_validate_architecture"
        try:
            with torch.no_grad():
                result = self._forward_impl(dummy_tokens, update_hidden=False)
        finally:
            self._training_lock_holder = None
            self._training_lock.release()

        # Verify output shapes.
        expected_shapes = {
            "topology_logits":  (VALIDATION_TEST_BATCH, cfg.n_topology),
            "traversal_raw":    (VALIDATION_TEST_BATCH, TRAVERSAL_HEAD_DIM),
            "friction_raw":     (VALIDATION_TEST_BATCH, FRICTION_HEAD_DIM),
            "source_embedding": (VALIDATION_TEST_BATCH, cfg.n_source),
            "phase_logits":     (VALIDATION_TEST_BATCH, cfg.n_phase),
            "new_hidden":       (VALIDATION_TEST_BATCH, cfg.d_model),
        }

        for field_name, expected_shape in expected_shapes.items():
            actual = getattr(result, field_name)
            if tuple(actual.shape) != expected_shape:
                raise RuntimeError(
                    f"Architecture validation failed: {field_name} has shape "
                    f"{tuple(actual.shape)}, expected {expected_shape}.  "
                    "This indicates a dimensional mismatch in the model construction."
                )

        # Verify hidden_state was NOT modified by the validation pass.
        if self.hidden_state.abs().sum().item() != 0.0:
            raise RuntimeError(
                "Architecture validation failed: hidden_state was modified during "
                "validation forward pass.  This indicates a bug in _forward_impl()."
            )

    # ── Readout — the critical path ───────────────────────────────────────

    def readout(
        self,
        topology_class: str,
        intent_vector: Optional[List[float]] = None,
    ) -> ReadoutResult:
        """
        O(1) inference readout from the persistent hidden state.

        THE CRITICAL PATH CALL.  Called by WorldLatentModel.query() on
        every query that does not hit the cache.  Must complete in <2ms.

        Projects hidden_state through all output heads to produce raw
        prediction tensors.  wlm_decoders.py converts these into
        TraversalPolicy and FrictionForecast contracts.

        If intent_vector is provided, it conditions the readout by
        adding a projected intent embedding to the hidden state.
        This does NOT modify self.hidden_state — the addition is
        performed on a local variable.

        INVARIANT: self.hidden_state is identical before and after
        this call.  This invariant is enforced by:
            1. torch.no_grad() — no gradient graph, no autograd side effects.
            2. Zero assignment to self.hidden_state in this method body.
               The word "self.hidden_state =" never appears in readout().
            3. All head projections operate on conditioned_state, which is
               a local variable, not a buffer reference.

        Violation of this invariant = MFT corruption at query time.
        index_daemon via WLMTrainingInterface is the ONLY writer
        of hidden_state.  Always.

        Parameters:
            topology_class:  The topology class being queried.  Passed through
                             to ReadoutResult for decoder context.  Must be one
                             of TOPOLOGY_CLASSES but this is NOT enforced here —
                             enforcement is in WorldLatentModel.query().

            intent_vector:   Optional 256-dimensional float list from the AXIOM
                             graph's query embedding.  None for Phase III known
                             topology classes where intent does not affect policy.
                             Provided for Phase I/II where intent matters.

        Returns:
            ReadoutResult containing raw, detached tensors for all five heads
            plus the topology_class passthrough and hidden_state_version.
        """
        with torch.no_grad():
            # Capture hidden_state_version BEFORE any computation.
            # This is the version that was used for readout.
            version = int(self.hidden_state_version.item())

            # ── Condition on intent (if provided) ─────────────────────────
            # conditioned_state is a LOCAL VARIABLE.
            # It is NOT self.hidden_state.
            # Adding intent_projected to it does NOT modify the buffer.
            if intent_vector is not None:
                # Convert list to tensor.
                # Device must match hidden_state.
                intent_t = torch.tensor(
                    intent_vector,
                    dtype=torch.float32,
                    device=self.hidden_state.device,
                ).unsqueeze(0)  # (1, 256)

                self._validate_intent_vector(intent_t)

                intent_projected = self.intent_projection(intent_t)  # (1, d_model)

                # LOCAL variable — NOT a buffer mutation.
                conditioned_state = self.hidden_state + intent_projected
                conditioned_state = self.final_norm(conditioned_state)
            else:
                # Pure readout — no intent conditioning.
                # Still a local variable — clone is unnecessary because
                # torch.no_grad() prevents any in-place mutation, and
                # all head projections return new tensors.
                conditioned_state = self.final_norm(self.hidden_state)

            # ── Project through output heads ──────────────────────────────
            # Each head receives the same conditioned_state.
            # Each returns a new tensor — no mutation of conditioned_state.
            traversal_raw = self.traversal_head(conditioned_state)   # (1, 7)
            friction_raw  = self.friction_head(conditioned_state)    # (1, 5)
            source_raw    = self.source_head(conditioned_state)      # (1, n_source)
            phase_raw     = self.phase_head(conditioned_state)       # (1, n_phase)

            # Squeeze batch dimension for ReadoutResult.
            # ReadoutResult expects 1D tensors: (7,), (5,), etc.
            traversal_out = traversal_raw.squeeze(0).cpu()
            friction_out  = friction_raw.squeeze(0).cpu()
            source_out    = source_raw.squeeze(0).cpu()
            phase_out     = phase_raw.squeeze(0).cpu()

        # ── Construct result ──────────────────────────────────────────────
        # All tensors are detached (torch.no_grad context) and on CPU.
        return ReadoutResult(
            traversal_raw=traversal_out,
            friction_raw=friction_out,
            source_raw=source_out,
            phase_raw=phase_out,
            topology_class=topology_class,
            hidden_state_version=version,
        )

    # ── Forward — training only ───────────────────────────────────────────

    def forward(
        self,
        token_sequence: torch.Tensor,
        update_hidden: bool = False,
    ) -> ForwardResult:
        """
        Training-time forward pass through the full Mamba stack.

        TRAINING ONLY.  This method is gated behind the training lock.
        WLMTrainingInterface.get_model() returns this MambaRouter instance;
        index_daemon.py calls model.forward() after acquiring the lock.
        Calling forward() without the training lock raises RuntimeError.

        The training lock exists to prevent accidental forward() calls from
        the inference path (WorldLatentModel.query()).  The inference path
        calls readout(), not forward().  This separation is load-bearing.

        Sequence of operations:
            1. Validate input token sequence.
            2. Embed tokens: token_embedding + position_embedding.
            3. Apply domain projection.
            4. Pass through Mamba blocks with residual connections.
            5. Apply final normalization.
            6. Extract last-token representation as hidden state candidate.
            7. Project through all output heads.
            8. If update_hidden=True: atomically update self.hidden_state.

        Parameters:
            token_sequence:  (B, seq_len) long tensor of token IDs.
                             B is the batch dimension.  seq_len <=  max_seq_len.
                             All token IDs in [0, vocab_size).

            update_hidden:   If True, updates self.hidden_state with the new
                             hidden state candidate (detached from graph, mean
                             over batch dimension).  This is the MFT update.
                             Only set to True by WLMTrainingInterface after
                             optimizer.step() completes.

                             If False, hidden_state is not modified.  This is
                             the normal training mode: forward() produces outputs
                             for loss computation, gradients flow, but the hidden
                             state is not yet committed.

        Returns:
            ForwardResult containing tensors with gradient history (when
            update_hidden=False) for loss computation.
        """
        # ── Training lock guard ───────────────────────────────────────────
        if not self._training_lock.locked():
            raise RuntimeError(
                "MambaRouter.forward() called without the training lock held.  "
                "forward() is only reachable via WLMTrainingInterface.  "
                "If you are seeing this from the inference path, you have an "
                "architecture violation — use readout() instead."
            )

        # ── Delegate to implementation ────────────────────────────────────
        return self._forward_impl(token_sequence, update_hidden)

    def _forward_impl(
        self,
        token_sequence: torch.Tensor,
        update_hidden: bool = False,
    ) -> ForwardResult:
        """
        Internal implementation of the forward pass.

        Separated from forward() so that _validate_architecture() can call
        it directly (with the lock held) without hitting the lock check
        in the public forward() method.

        This method contains the actual neural network computation.
        It is NOT called from any inference path.
        """
        # ── Input validation ──────────────────────────────────────────────
        self._validate_token_input(token_sequence, context="forward")

        batch_size, seq_len = token_sequence.shape
        device = token_sequence.device

        # ── Token embedding + positional encoding ─────────────────────────
        # positions: (seq_len,) — shared across batch
        positions = torch.arange(seq_len, device=device, dtype=torch.long)

        # x: (B, seq_len, d_model)
        x = self.token_embedding(token_sequence)

        # Add positional encoding — broadcast over batch dimension.
        x = x + self.position_embedding(positions).unsqueeze(0)

        # Apply domain projection to the full sequence.
        # This adds a learned linear transformation that captures
        # position-independent feature interactions.
        x = self.domain_projection(x)

        # Embedding dropout — regularizes the input representation.
        x = self.embedding_dropout(x)

        # ── Mamba block stack ─────────────────────────────────────────────
        # Pre-norm residual connections:
        #   x = x + dropout(mamba_block(layer_norm(x)))
        # Each Mamba block selectively updates its portion of the
        # representation through input-dependent A, B, C matrices.
        for i in range(self.config.n_layers):
            residual = x
            x_normed = self.block_norms[i](x)
            x_mamba = self.blocks[i](x_normed)
            x_drop = self.block_dropouts[i](x_mamba)
            x = residual + x_drop

        # ── Final normalization ───────────────────────────────────────────
        x = self.final_norm(x)  # (B, seq_len, d_model)

        # ── Last-token representation = hidden state candidate ────────────
        # The last token in the sequence carries the accumulated SSM state
        # from all preceding tokens.  This is the Mamba equivalent of a
        # transformer's [CLS] token or a GRU's final hidden state.
        new_hidden = x[:, -1, :]  # (B, d_model)

        # ── Project through output heads ──────────────────────────────────
        topology_logits  = self.topology_head(new_hidden)   # (B, n_topology)
        traversal_raw    = self.traversal_head(new_hidden)   # (B, 7)
        friction_raw     = self.friction_head(new_hidden)    # (B, 5)
        source_embedding = self.source_head(new_hidden)      # (B, n_source)
        phase_logits     = self.phase_head(new_hidden)       # (B, n_phase)

        # ── MFT update (if requested) ─────────────────────────────────────
        # Only WLMTrainingInterface sets update_hidden=True, and only after
        # optimizer.step() has already been called.  The gradient step is
        # complete; this is the commit of the new structural knowledge.
        if update_hidden:
            # Detach from computation graph.
            # The stored hidden state must not carry gradient history.
            # If gradients flowed through the stored state, the next
            # forward pass would accumulate a graph reaching back to
            # previous training steps — memory would grow without bound.
            new_h = new_hidden.detach().mean(dim=0, keepdim=True)
            # Mean over batch dimension.
            # Multiple training examples contribute to one consensus state.
            # hidden_state is always (1, d_model) regardless of batch size.

            # Direct buffer assignment.
            # This is the ONLY code path that writes to self.hidden_state.
            # readout() never writes here.
            # _forward_impl() with update_hidden=False never writes here.
            self.hidden_state.copy_(new_h)
            self.hidden_state_version.add_(1)

        # ── Return ────────────────────────────────────────────────────────
        return ForwardResult(
            topology_logits=topology_logits,
            traversal_raw=traversal_raw,
            friction_raw=friction_raw,
            source_embedding=source_embedding,
            phase_logits=phase_logits,
            new_hidden=new_hidden,
        )

    # ── Training lock management ──────────────────────────────────────────

    def acquire_training_lock(self, holder: str = "unknown") -> bool:
        """
        Acquire the training lock.  Must be held before calling forward().

        Parameters:
            holder: Identifier string for the lock holder (for diagnostics).
                    WLMTrainingInterface passes "WLMTrainingInterface".

        Returns:
            True if the lock was acquired.  False if already held.

        Thread safety:
            The training lock is a threading.Lock.  It is NOT an asyncio lock.
            Gradient computation happens in a thread pool executor, not on
            the event loop.  A threading.Lock is the correct synchronization
            primitive.
        """
        acquired = self._training_lock.acquire(blocking=False)
        if acquired:
            self._training_lock_holder = holder
        return acquired

    def release_training_lock(self) -> None:
        """
        Release the training lock.

        Must be called after forward() completes (including after exceptions).
        WLMTrainingInterface uses try/finally to ensure release.

        Raises RuntimeError if the lock is not held.
        """
        if not self._training_lock.locked():
            raise RuntimeError(
                "release_training_lock() called but the training lock is not held.  "
                "This indicates a lock management bug in WLMTrainingInterface."
            )
        self._training_lock_holder = None
        self._training_lock.release()

    @property
    def training_lock_held(self) -> bool:
        """True if the training lock is currently held."""
        return self._training_lock.locked()

    @property
    def training_lock_holder(self) -> Optional[str]:
        """Identifier of the current training lock holder, or None."""
        return self._training_lock_holder

    # ── Hidden state management ───────────────────────────────────────────

    def get_hidden_state_snapshot(self) -> Tuple[torch.Tensor, int]:
        """
        Return a detached copy of the current hidden state and its version.

        The snapshot is independent of the live buffer — modifying the
        snapshot does not affect the model's state.

        Used by:
        - Diagnostics and health checks.
        - WLMTrainingInterface to capture state before gradient steps.
        - Tests to verify readout() does not modify hidden state.
        """
        with torch.no_grad():
            snapshot = self.hidden_state.clone().detach().cpu()
            version = int(self.hidden_state_version.item())
        return snapshot, version

    def reset_hidden_state(self) -> None:
        """
        Reset hidden_state to zeros and version to 0.

        Used by:
        - Tests (clean state between test cases).
        - cold_start.py if topology_router.pt is not found and bootstrap
          creates a fresh model.

        NOT used during normal operation — hidden state is preserved across
        checkpoints.  Resetting it throws away all accumulated structural
        knowledge.
        """
        with torch.no_grad():
            self.hidden_state.zero_()
            self.hidden_state_version.zero_()

    def hidden_state_digest(self) -> str:
        """
        SHA-256 hex digest of the current hidden state tensor bytes.

        Used for integrity verification:
        - Checkpoint writes include this digest.
        - Checkpoint loads verify the digest matches.
        - WLMTrainingInterface verifies that readout() does not change
          the digest (paranoid mode in tests).

        The digest is computed from the raw float32 bytes of the hidden
        state tensor.  It is sensitive to any bit-level change.
        """
        with torch.no_grad():
            state_bytes = self.hidden_state.cpu().numpy().tobytes()
        return hashlib.sha256(state_bytes).hexdigest()

    def set_hidden_state_from_tensor(
        self,
        new_state: torch.Tensor,
        new_version: int,
    ) -> None:
        """
        Replace the hidden state with a provided tensor and version.

        This is the low-level state replacement used by
        WLMTrainingInterface.update_hidden_state() (which adds the 500ms
        delay and logging).

        Validates shape before assignment.  Detaches the tensor.
        Moves to the hidden_state's device if necessary.

        Parameters:
            new_state:   (1, d_model) float tensor — the new MFT.
            new_version: The version number to assign.  Must be > current version.
        """
        expected_shape = self.config.hidden_state_shape
        if tuple(new_state.shape) != expected_shape:
            raise ValueError(
                f"new_state shape {tuple(new_state.shape)} does not match "
                f"expected shape {expected_shape}."
            )
        current_version = int(self.hidden_state_version.item())
        if new_version <= current_version:
            raise ValueError(
                f"new_version ({new_version}) must be > current version "
                f"({current_version}).  Hidden state version is monotonically "
                "increasing."
            )

        with torch.no_grad():
            self.hidden_state.copy_(new_state.detach().to(self.hidden_state.device))
            self.hidden_state_version.fill_(new_version)

    # ── Architecture introspection ────────────────────────────────────────

    @property
    def parameter_count(self) -> int:
        """Exact count of trainable parameters (excludes buffers)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def buffer_count(self) -> int:
        """Exact count of buffer elements (hidden_state + hidden_state_version)."""
        return sum(b.numel() for b in self.buffers())

    @property
    def total_element_count(self) -> int:
        """Total tensor elements in the model (parameters + buffers)."""
        return self.parameter_count + self.buffer_count

    @property
    def d_model(self) -> int:
        """Model dimension (convenience accessor)."""
        return self.config.d_model

    @property
    def n_layers(self) -> int:
        """Number of Mamba blocks (convenience accessor)."""
        return self.config.n_layers

    @property
    def vocab_size(self) -> int:
        """Vocabulary size (convenience accessor)."""
        return self.config.vocab_size

    @property
    def current_hidden_state_version(self) -> int:
        """Current hidden state version number."""
        return int(self.hidden_state_version.item())

    @property
    def device(self) -> torch.device:
        """The device of the hidden_state buffer (and thus the model)."""
        return self.hidden_state.device

    def architecture_summary(self) -> Dict[str, object]:
        """
        Produce a summary dict of the model architecture.

        Used by:
        - cold_start.py to log the model architecture at startup.
        - health() endpoints for diagnostic telemetry.
        - Tests to verify architecture consistency across checkpoints.

        Returns a flat dict suitable for structured logging.
        """
        cfg = self.config
        return {
            "model_class":             "MambaRouter",
            "vocab_size":              cfg.vocab_size,
            "d_model":                 cfg.d_model,
            "d_state":                 cfg.d_state,
            "d_conv":                  cfg.d_conv,
            "expand":                  cfg.expand,
            "d_inner":                 cfg.d_inner,
            "n_layers":                cfg.n_layers,
            "n_topology":              cfg.n_topology,
            "n_source":                cfg.n_source,
            "n_phase":                 cfg.n_phase,
            "dropout":                 cfg.dropout,
            "max_seq_len":             cfg.max_seq_len,
            "traversal_head_dim":      TRAVERSAL_HEAD_DIM,
            "friction_head_dim":       FRICTION_HEAD_DIM,
            "hidden_state_shape":      list(cfg.hidden_state_shape),
            "parameter_count":         self.parameter_count,
            "buffer_count":            self.buffer_count,
            "hidden_state_version":    self.current_hidden_state_version,
            "hidden_state_digest":     self.hidden_state_digest(),
            "training_lock_held":      self.training_lock_held,
            "estimated_vram_mb_fp32":  round(cfg.estimated_vram_mb(4), 2),
            "estimated_vram_mb_fp16":  round(cfg.estimated_vram_mb(2), 2),
        }

    def head_output_shapes(self) -> Dict[str, Tuple[int, ...]]:
        """
        Return the expected output shapes for each head (excluding batch dim).

        Useful for tests and for wlm_decoders.py to validate received tensors.
        """
        cfg = self.config
        return {
            "topology_logits":  (cfg.n_topology,),
            "traversal_raw":    (TRAVERSAL_HEAD_DIM,),
            "friction_raw":     (FRICTION_HEAD_DIM,),
            "source_embedding": (cfg.n_source,),
            "phase_logits":     (cfg.n_phase,),
        }

    def state_dict_key_summary(self) -> Dict[str, Tuple[int, ...]]:
        """
        Return a mapping of state_dict key → tensor shape.

        Used by initialize_store.py to verify that the bootstrapped
        state_dict has the expected structure before saving to disk.
        """
        return {k: tuple(v.shape) for k, v in self.state_dict().items()}

    def verify_state_dict_compatibility(self, state_dict: Dict[str, torch.Tensor]) -> List[str]:
        """
        Verify that a loaded state_dict is compatible with this architecture.

        Returns a list of error messages.  Empty list = compatible.
        Does NOT load the state_dict — only checks key names and shapes.

        Used by WorldLatentModel._load_weights() before calling
        load_state_dict() to provide clear error messages when a checkpoint
        is from an incompatible architecture version.
        """
        errors: List[str] = []
        expected = self.state_dict()

        # Check for missing keys.
        for key in expected:
            if key not in state_dict:
                errors.append(f"Missing key: {key}")

        # Check for unexpected keys.
        for key in state_dict:
            if key not in expected:
                errors.append(f"Unexpected key: {key}")

        # Check shapes of matching keys.
        for key in expected:
            if key in state_dict:
                expected_shape = tuple(expected[key].shape)
                actual_shape = tuple(state_dict[key].shape)
                if expected_shape != actual_shape:
                    errors.append(
                        f"Shape mismatch for {key}: "
                        f"expected {expected_shape}, got {actual_shape}"
                    )

        return errors

    def freeze_all_except(self, head_names: Sequence[str]) -> int:
        """
        Freeze all parameters except the named output heads.

        Used by index_daemon for targeted fine-tuning:
        - Phase 1 training: freeze everything except traversal_head and friction_head.
        - Phase 2 training: unfreeze all heads.

        Parameters:
            head_names: Sequence of head attribute names to keep unfrozen.
                        Valid names: "topology_head", "traversal_head",
                        "friction_head", "source_head", "phase_head".

        Returns:
            Number of parameters that were frozen.
        """
        valid_heads = {"topology_head", "traversal_head", "friction_head",
                       "source_head", "phase_head"}
        for name in head_names:
            if name not in valid_heads:
                raise ValueError(
                    f"Unknown head name: {name!r}.  Valid names: {sorted(valid_heads)}"
                )

        frozen_count = 0
        unfrozen_heads = set(head_names)

        for param_name, param in self.named_parameters():
            # Check if this parameter belongs to an unfrozen head.
            is_unfrozen = any(
                param_name.startswith(head_name + ".")
                for head_name in unfrozen_heads
            )
            if is_unfrozen:
                param.requires_grad = True
            else:
                param.requires_grad = False
                frozen_count += param.numel()

        return frozen_count

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters.  Used after targeted fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def get_head_parameters(self, head_name: str) -> List[nn.Parameter]:
        """
        Return the parameter list for a named output head.

        Used by index_daemon to create per-head optimizers or
        per-head learning rate schedules.
        """
        head = getattr(self, head_name, None)
        if head is None:
            raise ValueError(f"No head named {head_name!r} exists on MambaRouter.")
        if not isinstance(head, nn.Module):
            raise ValueError(f"{head_name!r} is not an nn.Module.")
        return list(head.parameters())

    def get_backbone_parameters(self) -> List[nn.Parameter]:
        """
        Return parameters that are NOT part of any output head.

        This includes:
        - token_embedding
        - position_embedding
        - domain_projection
        - intent_projection
        - embedding_dropout (no parameters, but for completeness)
        - block_norms
        - blocks (Mamba SSM blocks)
        - block_dropouts (no parameters)
        - final_norm

        Used by index_daemon to create separate optimizer groups
        with different learning rates (lower LR for backbone,
        higher LR for heads).
        """
        head_names = {"topology_head", "traversal_head", "friction_head",
                      "source_head", "phase_head"}
        backbone_params: List[nn.Parameter] = []
        for name, param in self.named_parameters():
            is_head = any(name.startswith(h + ".") for h in head_names)
            if not is_head:
                backbone_params.append(param)
        return backbone_params

    def extra_repr(self) -> str:
        """Compact string representation for print(model)."""
        cfg = self.config
        return (
            f"vocab_size={cfg.vocab_size}, d_model={cfg.d_model}, "
            f"d_state={cfg.d_state}, d_conv={cfg.d_conv}, expand={cfg.expand}, "
            f"n_layers={cfg.n_layers}, n_topology={cfg.n_topology}, "
            f"n_source={cfg.n_source}, n_phase={cfg.n_phase}, "
            f"dropout={cfg.dropout}, max_seq_len={cfg.max_seq_len}, "
            f"params={self.parameter_count:,}, "
            f"hidden_state_version={self.current_hidden_state_version}"
        )