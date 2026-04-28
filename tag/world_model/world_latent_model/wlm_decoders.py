"""
tag/world_model/wlm_decoders.py
================================
Complete output decoding layer for the World Latent Model.

This file owns the full transformation pipeline from raw model output tensors
to typed contracts consumed by the rest of AXIOM.  Every activation function,
every output range enforcement, every topology-specific bias adjustment, every
source priority computation, and every output contract validation lives here.

Architectural position
----------------------
wlm_decoders.py sits between the MambaRouter's output heads and the callers
of WorldLatentModel.query().  It receives raw floating-point tensors and
returns frozen, validated contracts.  Nothing outside this file makes decisions
about how raw model outputs become routing policies.

What this file is
-----------------
    The complete output decoding layer for the WLM.
    Three primary decoders, each self-contained:
        _decode_traversal  — (7,) tensor → TopologyTraversalPolicy
        _decode_friction   — (5,) tensor → FrictionForecast
        _decode_source_priority — (512,) tensor + structural_layer → ranked domains
    All activation functions, standalone and testable in isolation.
    All topology-specific bias and floor constants.
    Structural layer loading, validation, and empty-state handling.
    Output contract validation for every decoded type.
    Batch decoding functions for cold_start_warmup() pre-population.

What this file is not
---------------------
    Not a model.  No nn.Module.  No parameters.  No forward pass.  No gradients.
    Not a tokenizer.  Does not process inputs.  Only interprets outputs.
    Not a cache.  Does not store decoded policies between calls.
    Not a classifier.  Receives topology classes already decided.
    Not responsible for what the model learned — only that its outputs are
        correctly transformed into valid contracts.
    Not allowed to make network calls, file reads outside structural layer
        loading, or system calls.  Pure tensor arithmetic only.

Activation order contract
-------------------------
The following order is mandatory and must never be violated:

    For traversal:
        1. Apply per-dimension activations  (sigmoid / softplus)
        2. Apply range clamping             (with warning log before each clamp)
        3. Apply topology-specific bias     (absolute overrides from TOPOLOGY_TRAVERSAL_BIAS)
        4. Construct and validate contract

    For friction:
        1. Apply sigmoid activations to all five raw values
        2. Apply coherence enforcement      (structural web constraints)
        3. Apply topology-specific floors   (minimums from FRICTION_FLOORS)
        4. Derive mitigation strategy       (deterministic from final probabilities)
        5. Construct and validate contract

    For source priority:
        1. Dot-product score computation    (source_matrix @ source_embedding)
        2. Top-k selection                  (phase-aware k from K_BY_PHASE)
        3. Softmax normalization            (over top-k only)
        4. Return (domains, scores) tuple

Bias application semantics
--------------------------
TOPOLOGY_TRAVERSAL_BIAS entries are absolute, not additive.
    "render_mode": "headless"  →  headless, regardless of activation output.
    "depth": 1                 →  depth=1, regardless of sigmoid * 4 + 1.
    "retry_budget": 0          →  zero retries, regardless of sigmoid * 5.
Bias overrides are applied after all activations.
Unknown topology classes receive no bias — raw activation output only.
The bias dict is a module-level constant and is never modified at runtime.

Floor application semantics
---------------------------
FRICTION_FLOORS entries are minimums, not overrides.
    floor applied as: max(model_output, floor_value)
The model can predict higher friction than the floor for known friction classes.
It can never predict lower.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import logging
import math # noqa
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F # noqa

from signal_kernel.contracts import (
    TOPOLOGY_CLASSES, # noqa
    FrictionForecast,
    TopologyTraversalPolicy,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# All out-of-range clamping is logged at WARNING so Witness can detect
# systematic model miscalibration early in training.  Do not suppress these
# warnings — they are the primary signal that activation ranges are drifting.
# ─────────────────────────────────────────────────────────────────────────────

log = logging.getLogger("tag.world_model.wlm_decoders")

# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION MATH CONSTANTS
# All thresholds and range endpoints are defined here as named constants.
# Never inline magic numbers in activation logic — the name carries the reason.
# ─────────────────────────────────────────────────────────────────────────────

# render_mode threshold — sigmoid must exceed this to select "headless".
# 0.60 not 0.50: static rendering is the default; headless requires a strong
# positive signal.  This preference is load-bearing — document and never change
# without understanding its downstream effect on Phantom dispatch frequency.
RENDER_MODE_THRESHOLD: float = 0.60

# tor_required threshold — sigmoid must exceed this to require Tor routing.
# 0.70 not 0.50: Tor is expensive (slower, limited exit nodes, higher latency).
# The model must be strongly confident before the system incurs Tor cost.
TOR_REQUIRED_THRESHOLD: float = 0.70

# depth encoding: sigmoid(raw) * DEPTH_SCALE + DEPTH_OFFSET → [DEPTH_MIN, DEPTH_MAX]
# After rounding to int, the range is [1, 5].
DEPTH_SCALE:  float = 4.0
DEPTH_OFFSET: float = 1.0
DEPTH_MIN:    int   = 1
DEPTH_MAX:    int   = 5

# requests_per_second encoding: softplus(raw) + RPS_OFFSET → [RPS_MIN, ∞) clipped to [RPS_MIN, RPS_MAX]
# softplus chosen over sigmoid — must not impose a hard upper bound from activation.
# clip is applied after activation, not before.
RPS_OFFSET: float = 0.1
RPS_MIN:    float = 0.1   # zero RPS is not a valid policy
RPS_MAX:    float = 100.0

# timeout_ms encoding: softplus(raw) * TIMEOUT_SCALE + TIMEOUT_OFFSET → [TIMEOUT_MIN, ∞)
# clipped to [TIMEOUT_MIN, TIMEOUT_MAX].
# TIMEOUT_MIN = 1000ms: sub-second timeouts cause false friction classifications.
# TIMEOUT_MAX = 30000ms: beyond this Phantom abandons rather than retries.
TIMEOUT_SCALE:  float = 1000.0
TIMEOUT_OFFSET: float = 1000.0
TIMEOUT_MIN:    int   = 1_000
TIMEOUT_MAX:    int   = 30_000

# retry_budget encoding: sigmoid(raw) * RETRY_SCALE → [0.0, RETRY_MAX] → round to int.
# 0 retries is a valid policy — some friction classes should not retry.
RETRY_SCALE: float = 5.0
RETRY_MIN:   int   = 0
RETRY_MAX:   int   = 5

# source priority top-k defaults per phase.
# Phase I explores broadly; Phase III is decisive.
SOURCE_PRIORITY_K_PHASE_I:   int = 10
SOURCE_PRIORITY_K_PHASE_II:  int = 7
SOURCE_PRIORITY_K_PHASE_III: int = 3

# Mapping from phase integer to default k — callers use K_BY_PHASE[phase].
# Phase is 1-indexed per contracts.py PHASE_I / PHASE_II / PHASE_III constants.
K_BY_PHASE: Dict[int, int] = {
    1: SOURCE_PRIORITY_K_PHASE_I,
    2: SOURCE_PRIORITY_K_PHASE_II,
    3: SOURCE_PRIORITY_K_PHASE_III,
}

# Coherence rule thresholds — physical constraints about friction co-occurrence.
# These are not model corrections; they encode structural facts about the web.
CLOUDFLARE_BOT_COHERENCE_THRESHOLD: float = 0.80  # CF implies bot detection
CLOUDFLARE_BOT_MULTIPLIER:          float = 0.90  # bot = max(bot, CF * 0.90)
AUTH_PAYWALL_COHERENCE_THRESHOLD:   float = 0.70  # auth redirect makes paywall irrelevant
AUTH_PAYWALL_CAP:                   float = 0.20  # paywall capped to 0.20 if auth > 0.70
RATE_LIMIT_BOT_COHERENCE_THRESHOLD: float = 0.80  # heavy rate limiting implies bot detection
RATE_LIMIT_BOT_FLOOR:               float = 0.60  # bot detection floor under rate limit

# Mitigation strategy thresholds — ordered by friction priority.
# See _derive_mitigation_strategy() for the full decision tree.
MITIGATION_AUTH_THRESHOLD:       float = 0.70
MITIGATION_PAYWALL_THRESHOLD:    float = 0.70
MITIGATION_CLOUDFLARE_THRESHOLD: float = 0.70
MITIGATION_RATE_LIMIT_THRESHOLD: float = 0.70
MITIGATION_BOT_DETECTION_THRESHOLD: float = 0.60

# Validation tolerance for probability values.
# Values in [-PROBABILITY_TOLERANCE, 1.0 + PROBABILITY_TOLERANCE] are accepted
# after clamping.  Values outside this range indicate a decoder bug.
PROBABILITY_TOLERANCE: float = 1e-6

# Valid render modes — exactly these two strings, nothing else.
VALID_RENDER_MODES: frozenset = frozenset({"static", "headless"})

# Valid mitigation strategies — exactly this set.
VALID_MITIGATION_STRATEGIES: frozenset = frozenset({
    "standard",
    "slow_crawl",
    "headless_retry",
    "tor_extract",
    "tor_headless",
    "skip",
})

# ─────────────────────────────────────────────────────────────────────────────
# TOPOLOGY_TRAVERSAL_BIAS
# Absolute overrides applied after all activations, before contract construction.
#
# Design intent
# -------------
# The model's raw outputs encode learned correlations.  Early in training,
# these correlations may contradict known structural facts about certain
# topology classes.  TOPOLOGY_TRAVERSAL_BIAS ensures structural knowledge
# overrides learned behavior that is known to be wrong for these classes.
#
# Semantics
# ---------
# Every entry is a dict of field-name → absolute value.
# "absolute" means: this value is used instead of the activation output.
# It is NOT added to or multiplied with the activation — it replaces it.
#
# tor_required note
# -----------------
# tor_required is computed by the decoder and used to influence the companion
# FrictionForecast.mitigation_strategy, but TopologyTraversalPolicy does not
# carry a tor_required field (the contract is frozen as defined in contracts.py).
# tor_required entries in this bias dict are consumed by the decoder and logged
# as advisory signals; they do not raise errors when the field is absent from
# the contract.  The corresponding friction decoder enforces Tor routing via
# FRICTION_FLOORS and _derive_mitigation_strategy independently.
#
# Phase III note
# --------------
# All Phase III known topology classes bias toward depth=1.  At Phase III the
# WLM has compiled policy — it navigates directly to signal sources without
# exploratory depth traversal.  This is a correct architectural choice, not a
# heuristic.
# ─────────────────────────────────────────────────────────────────────────────

TOPOLOGY_TRAVERSAL_BIAS: Dict[str, Dict[str, Any]] = {

    # ── Always headless: JavaScript is required for content render.
    # The model cannot override this — these topology classes produce empty
    # static renders on every known instance in the training corpus.
    "SAAS_DOCS_WITH_CODE": {
        "render_mode": "headless",
        # JS-rendered code blocks require a DOM engine to appear.
        # Static fetch yields empty or skeleton HTML.
    },
    "ECOMMERCE_PRODUCT": {
        "render_mode": "headless",
        # Product price, availability, and variant data are JS-injected.
        # Static fetch returns the shell template, not the populated product.
    },
    "ECOMMERCE_PRODUCT_VARIANT": {
        "render_mode": "headless",
        # Variant selection (size, color, config) always requires JS execution.
        # Static fetch cannot resolve the selected variant state.
    },

    # ── Always static: No JavaScript needed; static fetch is faster and cheaper.
    # The model cannot override this — headless rendering for these classes
    # wastes Phantom resources with zero signal quality gain.
    "REST_API_JSON": {
        "render_mode": "static",
        # JSON API responses are server-rendered.  No DOM engine needed.
        # headless would add Chromium overhead with identical output.
    },
    "REST_API_JSON_PAGINATED": {
        "render_mode": "static",
        # Pagination logic lives in query parameters, not JS state.
        # Static is correct; headless is wasteful.
    },
    "JSON_LD_STRUCTURED": {
        "render_mode": "static",
        # JSON-LD is embedded in HTML <script> tags — server-rendered always.
        # Static extraction via grep pipeline is sufficient and optimal.
    },
    "WIKIPEDIA_ARTICLE": {
        "render_mode": "static",
        # Wikipedia serves fully rendered HTML.  No JS dependency for content.
        # Static is faster; headless adds only latency.
    },

    # ── Friction / blocking classes: Minimal depth, no retries.
    # Traversing deeply or retrying against a blocking layer wastes the retry
    # budget and signal-to-noise budget without gaining recoverable content.
    # depth=1: attempt once to identify the block, then classify and route away.
    # retry_budget=0: retrying against a challenge will always fail identically.
    "CLOUDFLARE_CHALLENGE": {
        "depth": 1,
        "retry_budget": 0,
        # Cloudflare challenge pages do not resolve on retry from the same IP.
        # Depth > 1 compounds latency against a wall.
    },
    "RATE_LIMITED": {
        "depth": 1,
        "retry_budget": 0,
        "requests_per_second": RPS_MIN,   # 0.1 RPS — absolute floor
        # Under active rate limiting every additional request risks IP ban.
        # RPS set to minimum possible value; depth and retries suppressed.
    },
    "AUTH_REDIRECT": {
        "depth": 1,
        "retry_budget": 0,
        # Authentication wall terminates traversal — there is no further content
        # to retrieve without credentials.  Phantom must detect and skip.
    },

    # ── News classes: Shallow traversal sufficient.
    "NEWS_ARTICLE": {
        "depth": 2,
        # News articles are single-page content.  depth=2 allows following
        # one canonical redirect or amp URL without over-traversing.
    },
    "NEWS_ARTICLE_PAYWALLED": {
        "depth": 1,
        "tor_required": True,
        # Paywalled articles require anonymized fetch to access full content.
        # tor_required is an advisory signal to the friction decoder.
        # depth=1: the paywall is on the article URL itself — no traversal helps.
        # tor_required flag is consumed by _apply_traversal_bias() and logged;
        # it influences FrictionForecast.mitigation_strategy independently.
    },

    # ── Documentation classes: Deeper traversal needed for full coverage.
    "SAAS_DOCS": {
        "depth": 4,
        # SaaS documentation is hierarchical.  Conceptual pages link to
        # reference pages which link to API pages.  depth=4 covers the
        # typical three-level doc tree with one spare level.
    },
    "SAAS_DOCS_VERSIONED": {
        "depth": 4,
        # Versioned docs add a version-selection layer above the conceptual
        # hierarchy.  depth=4 accommodates: version → section → concept → ref.
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# FRICTION_FLOORS
# Minimum friction probabilities per topology class.
#
# Design intent
# -------------
# Some topology classes have known, near-certain friction regardless of model
# output.  A CLOUDFLARE_CHALLENGE page has Cloudflare friction by definition —
# it cannot have cloudflare_probability < 0.99.  The model may output a lower
# value early in training; the floor corrects this without penalizing the model.
#
# Semantics
# ---------
# Floors are applied as: final_prob = max(model_output, floor_value).
# The model CAN predict higher friction than the floor.
# The model CANNOT predict lower friction than the floor for these classes.
# This is physically correct — the floor encodes certain structural knowledge.
#
# Ordering
# --------
# Floors are applied AFTER coherence enforcement.
# This prevents coherence rules from pulling a floored probability below its
# guaranteed minimum.
# ─────────────────────────────────────────────────────────────────────────────

FRICTION_FLOORS: Dict[str, Dict[str, float]] = {

    "NEWS_ARTICLE_PAYWALLED": {
        "paywall_probability": 0.95,
        # By definition a paywalled article has a paywall.  The floor is 0.95
        # not 1.0 to preserve the model's ability to express uncertainty about
        # whether the specific URL will present the paywall at access time
        # (some paywalls are metered and may not trigger for every visit).
    },

    "CLOUDFLARE_CHALLENGE": {
        "cloudflare_probability": 0.99,
        # A Cloudflare challenge page IS a Cloudflare challenge by classification.
        # 0.99 not 1.0: rounding and floating point noise from sigmoid are real.
        # Any value above 0.99 is valid; below it is a classification inconsistency.
    },

    "RATE_LIMITED": {
        "rate_limit_probability": 0.99,
        # Same argument as CLOUDFLARE_CHALLENGE: the topology class name encodes
        # the friction.  rate_limit_probability < 0.99 for RATE_LIMITED is
        # internally inconsistent.
    },

    "AUTH_REDIRECT": {
        "auth_redirect_probability": 0.99,
        # AUTH_REDIRECT means the server issued a redirect to an authentication
        # endpoint.  auth_redirect_probability=0.99 is structurally certain.
    },

    "ECOMMERCE_PRODUCT": {
        "bot_detection_probability": 0.40,
        # Ecommerce product pages are high-value scraping targets.
        # Bot detection is prevalent — 0.40 floor reflects typical baseline.
        # The model can and should learn higher values for specific hostile domains.
    },

    "ECOMMERCE_PRODUCT_VARIANT": {
        "bot_detection_probability": 0.40,
        # Same rationale as ECOMMERCE_PRODUCT.
        # Variant pages (size/color selectors) are also high-value targets.
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL FRICTION STATE
# Used during the decode pipeline before constructing the frozen FrictionForecast.
#
# FrictionForecast (contracts.py) does not carry bot_detection_probability —
# it is not stored in the final contract.  _FrictionState holds all five raw
# activation outputs through the internal pipeline (coherence enforcement, floor
# application, mitigation derivation) and then projects to FrictionForecast.
#
# Why not store bot_detection in FrictionForecast?
# ------------------------------------------------
# The contract is frozen as defined in contracts.py.  bot_detection is an input
# to mitigation_strategy derivation, not a distinct routing signal.  The callers
# of FrictionForecast (phantom.py, interface.py) consume mitigation_strategy
# directly — they do not branch on bot_detection_probability separately.
# The signal is fully expressed in the chosen mitigation strategy.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _FrictionState:
    """
    Internal mutable representation of all five friction probabilities during
    the decode pipeline.  Never exposed outside this module.

    Pipeline stages that mutate this state:
        1. Activation        — sigmoid applied, raw logits → [0.0, 1.0] each
        2. Coherence         — structural constraints enforced between fields
        3. Floor application — FRICTION_FLOORS maximums applied per topology class
        4. → FrictionForecast construction via .to_forecast()

    Fields match raw tensor layout exactly:
        raw[0] → cloudflare_probability
        raw[1] → paywall_probability
        raw[2] → rate_limit_probability
        raw[3] → auth_redirect_probability
        raw[4] → bot_detection_probability  ← not in final FrictionForecast contract
    """

    topology_class:            str
    cloudflare_probability:    float
    paywall_probability:       float
    rate_limit_probability:    float
    auth_redirect_probability: float
    bot_detection_probability: float

    def to_forecast(self, mitigation_strategy: str) -> FrictionForecast:
        """
        Project internal state to the frozen FrictionForecast contract.

        bot_detection_probability is consumed into mitigation_strategy during
        derivation and is not carried into the contract — the contract only
        stores the four friction probabilities plus the derived strategy.

        Parameters
        ----------
        mitigation_strategy : str
            Deterministic strategy string produced by _derive_mitigation_strategy().
            Must be in VALID_MITIGATION_STRATEGIES.

        Returns
        -------
        FrictionForecast
            Frozen contract with validated probability fields.
        """
        return FrictionForecast(
            topology_class=self.topology_class,
            cloudflare_probability=self.cloudflare_probability,
            paywall_probability=self.paywall_probability,
            rate_limit_probability=self.rate_limit_probability,
            auth_redirect_probability=self.auth_redirect_probability,
            mitigation_strategy=mitigation_strategy,
        )


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL LAYER VIEW
# Typed interface over the structural_layer.pt dictionary.
# Provides named attribute access and shape documentation in one place.
#
# Usage in decode_source_priority():
#     layer = load_structural_layer(path)
#     if isinstance(layer, dict):
#         view = StructuralLayerView.from_dict(layer)
#         # use view.source_matrix, view.domain_index, etc.
#
# StructuralLayerView is a read-only lens over the dict that index_daemon writes.
# It does not copy tensors — it holds references.  Mutation of the underlying
# tensors via the view is the caller's problem, not this module's.
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StructuralLayerView:
    """
    Typed, documented view over the structural_layer.pt dictionary.

    index_daemon writes structural_layer.pt as a raw dict of tensors and Python
    lists.  StructuralLayerView wraps that dict with named attributes, shape
    documentation, and a canonical construction path.

    This is NOT EmptyStructuralLayer.  StructuralLayerView always has at least
    one domain.  load_structural_layer() returns either a StructuralLayerView
    (success) or EmptyStructuralLayer (failure/missing).

    Attributes
    ----------
    source_matrix : torch.Tensor
        Shape (n_domains, 512).
        Each row is the L2-normalized embedding for one domain.
        Row order corresponds to domain_index — row i belongs to domain_index[i].
        Pre-normalized by index_daemon during offline encoding.
        dot(source_matrix[i], query) = cosine_similarity(domain_i, query)
        without division overhead.

    domain_index : list of str
        Length n_domains.
        domain_index[i] is the domain name string for source_matrix row i.
        Example: ["wikipedia.org", "stripe.com/docs", "reactjs.org"]

    intent_clusters : torch.Tensor or None
        Shape (n_clusters, 512) if present, else None.
        Cluster centroids for intent-based routing.
        Built after enough domain events have accumulated for clustering.
        May be None in early training before clustering has run.

    cluster_domains : list of list of str
        Length n_clusters if intent_clusters is present, else [].
        cluster_domains[i] is the list of domain names in cluster i.
        Used by callers that want cluster-level source routing rather than
        individual domain ranking.

    n_domains : int (property)
        Number of domains in the structural layer.  Equivalent to
        len(domain_index) and source_matrix.shape[0].

    n_clusters : int (property)
        Number of intent clusters.  0 if intent_clusters is None.
    """

    source_matrix:   torch.Tensor
    domain_index:    List[str]
    intent_clusters: Optional[torch.Tensor]
    cluster_domains: List[List[str]]

    @classmethod
    def from_dict(cls, layer: Dict) -> "StructuralLayerView":
        """
        Construct a StructuralLayerView from a validated structural layer dict.

        Callers must pass a dict that has already been validated by
        validate_structural_layer().  from_dict() does not re-validate.

        Parameters
        ----------
        layer : dict
            Validated structural layer dict from load_structural_layer().
            Must contain: source_matrix, domain_index, intent_clusters,
            cluster_domains.

        Returns
        -------
        StructuralLayerView
        """
        return cls(
            source_matrix=layer["source_matrix"],
            domain_index=layer["domain_index"],
            intent_clusters=layer.get("intent_clusters"),
            cluster_domains=layer.get("cluster_domains", []),
        )

    @property
    def n_domains(self) -> int:
        """Number of domains in the structural layer."""
        return len(self.domain_index)

    @property
    def n_clusters(self) -> int:
        """Number of intent clusters.  0 if intent_clusters is None."""
        if self.intent_clusters is None:
            return 0
        return self.intent_clusters.shape[0]

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (512 always for WLM structural layer)."""
        return self.source_matrix.shape[1]

    def domain_at(self, index: int) -> str:
        """
        Return the domain name at position index.

        Parameters
        ----------
        index : int
            Row index into source_matrix and domain_index.

        Returns
        -------
        str
            Domain name string.

        Raises
        ------
        IndexError
            If index is out of bounds.
        """
        if index < 0 or index >= self.n_domains:
            raise IndexError(
                f"StructuralLayerView.domain_at({index}): "
                f"index out of bounds for n_domains={self.n_domains}."
            )
        return self.domain_index[index]

    def __bool__(self) -> bool:
        """
        True if this layer has at least one domain.
        Always True for StructuralLayerView (contrast with EmptyStructuralLayer).
        """
        return self.n_domains > 0

    def __repr__(self) -> str:
        return (
            f"StructuralLayerView("
            f"n_domains={self.n_domains}, "
            f"n_clusters={self.n_clusters}, "
            f"embedding_dim={self.embedding_dim})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# EMPTY STRUCTURAL LAYER
# Returned when structural_layer.pt does not exist or fails to load.
#
# Expected state: cold start, before Wikipedia preparse completes.
# Not an error condition.  Phantom falls back to URL-based routing when
# source priority is empty.  WorldLatentModel handles empty source priority
# without try/except on every attribute access.
#
# The class must return empty/None gracefully on every attribute access.
# It must NOT raise AttributeError, KeyError, or any other exception.
# latent_model.py checks structural_layer.domain_index truthiness to branch
# between real structural layer and fallback path — no exception handling
# required at the call site.
# ─────────────────────────────────────────────────────────────────────────────

class EmptyStructuralLayer:
    """
    Sentinel structural layer returned when structural_layer.pt is absent.

    Produces empty source priority on every call to decode_source_priority().
    An empty source priority list is valid — it signals to Phantom that the
    WLM has no source knowledge yet and URL-based routing should be used.

    This is the expected state between AXIOM process start and the completion
    of the first Wikipedia preparse cycle.  It is not a degraded state —
    it is the defined initial state.

    Attribute contract
    ------------------
    All attributes return empty/None without raising.  This allows callers to
    test ``if structural_layer.domain_index:`` without try/except.

    Attributes
    ----------
    source_matrix : None
        No domain embeddings exist yet.  dot-product scoring is not possible.

    domain_index : list (empty)
        No domains have been encoded.  Length check: ``if not domain_index``
        is the canonical check for EmptyStructuralLayer state.

    intent_clusters : None
        No cluster centroids exist.  Intent-based routing is not available.

    cluster_domains : list (empty)
        No cluster membership has been established.
    """

    source_matrix:   None     = None
    domain_index:    List     = []
    intent_clusters: None     = None
    cluster_domains: List     = []

    def __repr__(self) -> str:
        return "EmptyStructuralLayer(state=cold_start, domains=0)"

    def __bool__(self) -> bool:
        """
        False: EmptyStructuralLayer is never truthy.
        Allows: ``if structural_layer:`` as a readiness check.
        """
        return False


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVATION FUNCTIONS
# Each function is standalone, pure, testable in isolation.
# No side effects.  No state.  Input → output only.
#
# Each function documents:
#   - The exact activation applied
#   - The exact output range
#   - The exact rationale for the choice of activation
#   - The exact rationale for the range bounds
#   - Clamping behaviour and logging
# ─────────────────────────────────────────────────────────────────────────────

def bounded_depth(raw: float) -> int:
    """
    Map a raw traversal head depth logit to a validated integer depth in [1, 5].

    Activation
    ----------
    sigmoid(raw) * 4 + 1 → [1.0, 5.0] → round() → int

    Why sigmoid?
        Depth must be bounded.  sigmoid maps (-∞, +∞) to (0, 1), which after
        scaling gives (1.0, 5.0).  The practical extremes are not reached at
        finite raw values, so the effective range before rounding is (1.0, 5.0).
        After rounding, depth is always in {1, 2, 3, 4, 5}.

    Why 4.0 scale + 1.0 offset?
        sigmoid(0) = 0.5 → 0.5 * 4 + 1 = 3.0 → depth 3 at zero initialization.
        This is a sensible default for an untrained model — moderate depth.
        The offset of 1.0 enforces the hard minimum of 1 even at sigmoid → 0.

    Range
    -----
    Output is always in {1, 2, 3, 4, 5}.
    Hard lower bound 1: depth 0 means do not fetch at all — that decision
        belongs to interface.py, not the traversal policy.
    Hard upper bound 5: beyond depth 5, signal-to-noise ratio degrades below
        useful levels for all known topology classes in the training corpus.

    Clamping
    --------
    In normal operation, sigmoid output never produces raw depth outside [1, 5].
    If numerical issues cause out-of-range result, clamp and log at WARNING.
    Systematic WARNING logs for this field indicate depth head miscalibration.

    Parameters
    ----------
    raw : float
        Raw logit from traversal_head output dimension 0.
        Unbounded float.  Typical range after training: (-3.0, 3.0).

    Returns
    -------
    int
        Integer depth in {1, 2, 3, 4, 5}.

    Examples
    --------
    >>> bounded_depth(0.0)    # sigmoid(0.0) = 0.5 → 0.5*4+1 = 3.0 → 3
    3
    >>> bounded_depth(5.0)    # sigmoid(5.0) ≈ 0.993 → 0.993*4+1 ≈ 4.97 → 5
    5
    >>> bounded_depth(-5.0)   # sigmoid(-5.0) ≈ 0.007 → 0.007*4+1 ≈ 1.03 → 1
    1
    >>> bounded_depth(1.386)  # sigmoid(1.386) ≈ 0.80 → 0.80*4+1 = 4.2 → 4
    4
    """
    sig = torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).item()
    depth_float = sig * DEPTH_SCALE + DEPTH_OFFSET
    depth_int = round(depth_float)

    if depth_int < DEPTH_MIN:
        log.warning(
            "bounded_depth: computed depth %d below minimum %d "
            "(raw=%.6f, sigmoid=%.6f, depth_float=%.6f). "
            "Clamping to %d.  Systematic occurrence indicates depth head "
            "initialization or gradient scaling issue.",
            depth_int, DEPTH_MIN, raw, sig, depth_float, DEPTH_MIN,
        )
        depth_int = DEPTH_MIN

    if depth_int > DEPTH_MAX:
        log.warning(
            "bounded_depth: computed depth %d above maximum %d "
            "(raw=%.6f, sigmoid=%.6f, depth_float=%.6f). "
            "Clamping to %d.  Systematic occurrence indicates depth head "
            "saturation — check gradient flow to traversal head.",
            depth_int, DEPTH_MAX, raw, sig, depth_float, DEPTH_MAX,
        )
        depth_int = DEPTH_MAX

    return depth_int


def bounded_rps(raw: float) -> float:
    """
    Map a raw requests_per_second logit to a validated float in [0.1, 100.0].

    Activation
    ----------
    softplus(raw) + 0.1 → [0.1, ∞) → clip to [0.1, 100.0]

    Why softplus instead of sigmoid?
        sigmoid imposes a hard upper bound from activation — sigmoid(raw) < 1.0
        means RPS could never exceed a small multiple of 1.0 after scaling.
        Requests per second has a legitimate range up to 100.0 for aggressive
        crawlers.  softplus = log(1 + exp(raw)) is always positive and grows
        without bound, so the ceiling is enforced by explicit clip, not by the
        activation itself.  This is architecturally correct: the model expresses
        "very high RPS" with a large positive logit, and the clip catches it.

    Why clip AFTER activation?
        Clipping before activation (on the raw logit) would distort the gradient
        signal at training time.  The activation receives the true gradient;
        the clip is a post-hoc range enforcement that does not feed back into
        the model.

    Why 0.1 minimum?
        Zero RPS is not a valid policy — it would mean the WLM is telling
        Phantom to make no requests at all.  If the intent is "do not fetch",
        the routing decision belongs to interface.py.  0.1 RPS is the minimal
        valid crawl rate (one request per 10 seconds).

    Range
    -----
    Output is always in [0.1, 100.0].
    Clamping above 100.0 is logged at WARNING — this value is rarely correct
    and usually indicates the RPS head has overfit to a high-velocity domain.

    Parameters
    ----------
    raw : float
        Raw logit from traversal_head output dimension 2.
        Unbounded float.  softplus(raw) is always > 0.

    Returns
    -------
    float
        Requests per second in [0.1, 100.0].

    Examples
    --------
    >>> bounded_rps(0.0)    # softplus(0.0) = log(2) ≈ 0.693 → 0.693 + 0.1 = 0.793
    0.793...
    >>> bounded_rps(-10.0)  # softplus(-10) ≈ 0.0000454 → +0.1 ≈ 0.100 → clamped to 0.1
    0.1
    >>> bounded_rps(10.0)   # softplus(10) ≈ 10.000 → +0.1 ≈ 10.1
    10.1...
    >>> bounded_rps(100.0)  # softplus(100) >> 100 → +0.1 >> 100.1 → clamped to 100.0
    100.0
    """
    sp = F.softplus(torch.tensor(raw, dtype=torch.float64)).item()
    rps = sp + RPS_OFFSET

    if rps < RPS_MIN:
        # This should not occur given softplus + 0.1 ≥ 0.1 always,
        # but floating point accumulation can produce rps ≈ 0.09999...
        log.warning(
            "bounded_rps: computed RPS %.6f below minimum %.1f "
            "(raw=%.6f, softplus=%.6f). Clamping.  "
            "This indicates floating-point underflow in softplus — investigate.",
            rps, RPS_MIN, raw, sp,
        )
        rps = RPS_MIN

    if rps > RPS_MAX:
        log.warning(
            "bounded_rps: computed RPS %.4f exceeds maximum %.1f "
            "(raw=%.6f, softplus=%.6f). Clamping to %.1f.  "
            "Systematic occurrence indicates RPS head overfit to high-velocity domains.",
            rps, RPS_MAX, raw, sp, RPS_MAX,
        )
        rps = RPS_MAX

    return float(rps)


def bounded_timeout(raw: float) -> int:
    """
    Map a raw timeout logit to a validated integer timeout in [1000, 30000] ms.

    Activation
    ----------
    softplus(raw) * 1000 + 1000 → [1000, ∞) → clip to [1000, 30000] → round to int

    Why softplus?
        Same rationale as bounded_rps: timeout must have a real upper bound
        imposed by clip, not by activation.  A long timeout is a valid model
        output for difficult topology classes; the clip at 30000ms is a hard
        system constraint, not a probability ceiling.

    Why 1000ms minimum?
        Sub-second timeouts cause false friction classifications.  A 500ms
        timeout on a slow network is indistinguishable from a Cloudflare
        challenge page — both produce no response within the window.  The
        classifier then mislabels the URL as CLOUDFLARE_CHALLENGE, polluting
        the training signal.  1000ms is the empirically determined floor below
        which false friction rates spike unacceptably.

    Why 30000ms maximum?
        Beyond 30 seconds, Phantom does not retry — it abandons the request
        and classifies the URL as unreachable.  A timeout policy above 30000ms
        is therefore never honored.  Clamping to 30000ms ensures the traversal
        policy stays within Phantom's operating envelope.

    softplus(0.0) = log(2) ≈ 0.693 → 693ms + 1000ms = 1693ms at zero initialization.
    This is a reasonable default timeout for an untrained model.

    Parameters
    ----------
    raw : float
        Raw logit from traversal_head output dimension 4.
        Unbounded float.

    Returns
    -------
    int
        Timeout in milliseconds, in {1000, 1001, ..., 30000}.

    Examples
    --------
    >>> bounded_timeout(0.0)    # softplus(0)≈0.693 → *1000=693 → +1000=1693ms
    1693
    >>> bounded_timeout(-5.0)   # softplus(-5)≈0.0067 → *1000≈6.7 → +1000≈1007ms
    1007
    >>> bounded_timeout(29.0)   # softplus(29)≈29.0 → *1000=29000 → +1000=30000ms
    30000
    >>> bounded_timeout(100.0)  # >> 30000 → clamped to 30000ms
    30000
    """
    sp = F.softplus(torch.tensor(raw, dtype=torch.float64)).item()
    timeout_ms_float = sp * TIMEOUT_SCALE + TIMEOUT_OFFSET
    timeout_ms = int(round(timeout_ms_float))

    if timeout_ms < TIMEOUT_MIN:
        log.warning(
            "bounded_timeout: computed timeout %dms below minimum %dms "
            "(raw=%.6f, softplus=%.6f, timeout_float=%.2f). "
            "Clamping to %dms.  Sub-minimum timeouts cause false friction classifications.",
            timeout_ms, TIMEOUT_MIN, raw, sp, timeout_ms_float, TIMEOUT_MIN,
        )
        timeout_ms = TIMEOUT_MIN

    if timeout_ms > TIMEOUT_MAX:
        log.warning(
            "bounded_timeout: computed timeout %dms exceeds maximum %dms "
            "(raw=%.6f, softplus=%.6f, timeout_float=%.2f). "
            "Clamping to %dms.  Phantom abandons at 30s — higher values are never honored.",
            timeout_ms, TIMEOUT_MAX, raw, sp, timeout_ms_float, TIMEOUT_MAX,
        )
        timeout_ms = TIMEOUT_MAX

    return timeout_ms


def bounded_retry(raw: float) -> int:
    """
    Map a raw retry_budget logit to a validated integer in [0, 5].

    Activation
    ----------
    sigmoid(raw) * 5 → [0.0, 5.0] → round() → int in {0, 1, 2, 3, 4, 5}

    Why sigmoid?
        Retry budget is bounded.  0 to 5 retries covers the full policy space.
        sigmoid maps the unbounded logit to (0, 1); scaling by 5 gives (0, 5).
        sigmoid(0) = 0.5 → 0.5 * 5 = 2.5 → round → 2 or 3 at initialization.
        This is a sensible default: moderate retry budget for unknown topologies.

    Why 0 minimum is valid?
        Some topology classes (CLOUDFLARE_CHALLENGE, AUTH_REDIRECT, RATE_LIMITED)
        should not retry.  Retrying against these blocking layers is always futile
        and costs Phantom resources.  0 retries is the correct policy for these
        classes, enforced via TOPOLOGY_TRAVERSAL_BIAS.

    Why 5 maximum?
        Empirical upper bound from Phantom's retry scheduler.  More than 5 retries
        against the same URL in a single session signals a routing error, not a
        transient failure.  The model should escalate to a different source or
        strategy before spending 6+ retries on one URL.

    Parameters
    ----------
    raw : float
        Raw logit from traversal_head output dimension 3.
        Unbounded float.

    Returns
    -------
    int
        Retry budget in {0, 1, 2, 3, 4, 5}.

    Examples
    --------
    >>> bounded_retry(0.0)   # sigmoid(0)=0.5 → 0.5*5=2.5 → round → 2 or 3
    2
    >>> bounded_retry(5.0)   # sigmoid(5)≈0.993 → 0.993*5≈4.97 → round → 5
    5
    >>> bounded_retry(-5.0)  # sigmoid(-5)≈0.007 → 0.007*5≈0.033 → round → 0
    0
    >>> bounded_retry(2.2)   # sigmoid(2.2)≈0.90 → 0.90*5=4.5 → round → 4 or 5
    4
    """
    sig = torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).item()
    retry_float = sig * RETRY_SCALE
    retry_int = int(round(retry_float))

    # Paranoid clamp: sigmoid * 5 should always be in [0, 5], but
    # floating-point round on the boundary (e.g. 4.9999999...) needs guarding.
    if retry_int < RETRY_MIN:
        log.warning(
            "bounded_retry: computed retry_budget %d below minimum %d "
            "(raw=%.6f, sigmoid=%.6f, retry_float=%.6f). Clamping.",
            retry_int, RETRY_MIN, raw, sig, retry_float,
        )
        retry_int = RETRY_MIN

    if retry_int > RETRY_MAX:
        log.warning(
            "bounded_retry: computed retry_budget %d above maximum %d "
            "(raw=%.6f, sigmoid=%.6f, retry_float=%.6f). Clamping.",
            retry_int, RETRY_MAX, raw, sig, retry_float,
        )
        retry_int = RETRY_MAX

    return retry_int


def render_mode_from_logit(raw: float) -> str:
    """
    Map a raw render_mode logit to "headless" or "static".

    Activation
    ----------
    sigmoid(raw) > 0.60 → "headless", else → "static"

    Why 0.60 threshold?
        This threshold is load-bearing.  Static rendering is preferred because:
          - Static fetches are ~10x faster than headless (no Chromium launch)
          - Static fetches use ~100x less memory per concurrent request
          - Static fetches impose no JavaScript execution overhead
        The system prefers static unless the model is moderately confident that
        headless is required.  0.50 would flip to headless on any positive logit.
        0.60 requires a clear positive signal — roughly 1.5:1 log-odds in favor
        of headless — before incurring the cost.

        This threshold interacts with TOPOLOGY_TRAVERSAL_BIAS: several classes
        override render_mode absolutely, bypassing this function.  The threshold
        only applies to topology classes NOT in the bias dict.

    Parameters
    ----------
    raw : float
        Raw logit from traversal_head output dimension 1.
        Unbounded float.

    Returns
    -------
    str
        Exactly "headless" or "static".  Never None, never any other value.

    Examples
    --------
    >>> render_mode_from_logit(0.0)   # sigmoid(0.0)=0.5 ≤ 0.60 → "static"
    'static'
    >>> render_mode_from_logit(0.4)   # sigmoid(0.4)≈0.60 → borderline → "static"
    'static'
    >>> render_mode_from_logit(0.5)   # sigmoid(0.5)≈0.622 > 0.60 → "headless"
    'headless'
    >>> render_mode_from_logit(2.0)   # sigmoid(2.0)≈0.88 > 0.60 → "headless"
    'headless'
    >>> render_mode_from_logit(-2.0)  # sigmoid(-2.0)≈0.12 ≤ 0.60 → "static"
    'static'
    """
    sig = torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).item()
    return "headless" if sig > RENDER_MODE_THRESHOLD else "static"


def tor_from_logit(raw: float) -> bool:
    """
    Map a raw tor_required logit to a boolean Tor routing requirement.

    Activation
    ----------
    sigmoid(raw) > 0.70 → True, else → False

    Why 0.70 threshold?
        Tor routing is expensive:
          - Tor circuits introduce 200-500ms additional latency per request
          - Exit node availability is limited and variable
          - Tor exit IPs are often blocked by CDNs and rate-limited by others
          - Tor usage imposes shared-resource costs on the Tor network
        The model must be strongly confident — 0.70 log-odds ≈ 7:3 in favor —
        before the system routes through Tor.  0.50 would route half of all
        requests through Tor based on ambiguous model output.

        This value interacts with TOPOLOGY_TRAVERSAL_BIAS: NEWS_ARTICLE_PAYWALLED
        sets tor_required=True unconditionally, bypassing this threshold.
        For other topology classes, this function enforces the preference for
        non-Tor routing.

    Note on contract propagation
    ----------------------------
    TopologyTraversalPolicy does not carry a tor_required field.  The boolean
    returned by this function is consumed by apply_traversal_bias() for logging
    and advisory purposes.  The Tor routing signal is independently enforced by
    decode_friction() via FRICTION_FLOORS and _derive_mitigation_strategy(),
    which can return "tor_extract" or "tor_headless" strategies.

    Parameters
    ----------
    raw : float
        Raw logit from traversal_head output dimension 5.
        Unbounded float.

    Returns
    -------
    bool
        True if Tor routing is strongly recommended, False otherwise.

    Examples
    --------
    >>> tor_from_logit(0.0)   # sigmoid(0.0)=0.5 ≤ 0.70 → False
    False
    >>> tor_from_logit(0.85)  # sigmoid(0.85)≈0.70 → borderline → False
    False
    >>> tor_from_logit(1.0)   # sigmoid(1.0)≈0.731 > 0.70 → True
    True
    >>> tor_from_logit(3.0)   # sigmoid(3.0)≈0.952 > 0.70 → True
    True
    """
    sig = torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).item()
    return sig > TOR_REQUIRED_THRESHOLD


def friction_probability(raw: float) -> float:
    """
    Map a raw friction logit to a calibrated probability in [0.0, 1.0].

    Activation
    ----------
    sigmoid(raw) → [0.0, 1.0]

    All five friction head dimensions use this function identically.
    No other activation is applied to friction outputs.  The design is
    intentionally uniform — friction probabilities are model confidences,
    not rate signals, and sigmoid is the correct activation for probability
    outputs from an unbounded logit.

    Range
    -----
    Theoretically open (0.0, 1.0) due to sigmoid asymptotes.
    Practically clamped to [0.0, 1.0] after PROBABILITY_TOLERANCE guard.
    Values outside [0.0 - 1e-6, 1.0 + 1e-6] indicate floating-point issues
    and are logged and clamped — they should never occur in practice.

    Parameters
    ----------
    raw : float
        Raw logit from friction_head.  One of the five friction dimensions.
        Unbounded float.

    Returns
    -------
    float
        Probability in [0.0, 1.0].  Can be used directly in probability
        comparisons and FrictionForecast field construction.

    Examples
    --------
    >>> friction_probability(0.0)    # sigmoid(0.0) = 0.5
    0.5
    >>> friction_probability(4.6)    # sigmoid(4.6) ≈ 0.99
    0.990...
    >>> friction_probability(-4.6)   # sigmoid(-4.6) ≈ 0.01
    0.009...
    >>> friction_probability(10.0)   # sigmoid(10.0) ≈ 0.9999546
    0.9999546...
    >>> friction_probability(-10.0)  # sigmoid(-10.0) ≈ 0.0000454
    0.0000454...
    """
    prob = torch.sigmoid(torch.tensor(raw, dtype=torch.float64)).item()

    # Guard against floating-point pathologies — sigmoid should never leave [0,1]
    # but accumulated float64 operations can produce values like -1e-16 or 1+1e-15.
    if prob < 0.0:
        if prob < -PROBABILITY_TOLERANCE:
            log.warning(
                "friction_probability: sigmoid output %.10f is below 0.0 "
                "(raw=%.6f). This indicates floating-point pathology. "
                "Clamping to 0.0.", prob, raw,
            )
        prob = 0.0

    if prob > 1.0:
        if prob > 1.0 + PROBABILITY_TOLERANCE:
            log.warning(
                "friction_probability: sigmoid output %.10f exceeds 1.0 "
                "(raw=%.6f). This indicates floating-point pathology. "
                "Clamping to 1.0.", prob, raw,
            )
        prob = 1.0

    return float(prob)


# ─────────────────────────────────────────────────────────────────────────────
# BIAS APPLICATION — TRAVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def apply_traversal_bias(
    policy:         TopologyTraversalPolicy,
    topology_class: str,
) -> TopologyTraversalPolicy:
    """
    Apply TOPOLOGY_TRAVERSAL_BIAS overrides to a decoded traversal policy.

    Bias overrides are absolute, not additive.  When a bias entry says
    ``"depth": 1``, the policy's depth becomes 1 regardless of what the
    model output was.  This is structurally correct: the bias encodes known
    facts about topology classes that the model may not have learned correctly
    yet, especially early in training.

    Application rules
    -----------------
    1. If topology_class is not in TOPOLOGY_TRAVERSAL_BIAS, return policy unchanged.
    2. If topology_class is in the bias dict, apply each override field absolutely.
    3. tor_required bias entries are logged as advisory signals but do not raise —
       TopologyTraversalPolicy does not carry a tor_required field.
    4. Unrecognized bias field names are logged at WARNING and skipped.
    5. The returned policy is a new frozen dataclass instance — the input is
       unchanged (immutability contract from contracts.py frozen=True).

    Parameters
    ----------
    policy : TopologyTraversalPolicy
        The decoded traversal policy before bias application.
        Produced by applying activations to the raw (7,) tensor.

    topology_class : str
        The topology class for this decode call.
        Used as the key into TOPOLOGY_TRAVERSAL_BIAS.

    Returns
    -------
    TopologyTraversalPolicy
        New policy with all applicable bias overrides applied.
        If no bias exists for topology_class, returns the original policy object
        unchanged (identity return, not a copy).

    Examples
    --------
    SAAS_DOCS_WITH_CODE gets render_mode="headless" regardless of model output::

        policy = TopologyTraversalPolicy(render_mode="static", ...)
        biased = apply_traversal_bias(policy, "SAAS_DOCS_WITH_CODE")
        assert biased.render_mode == "headless"

    CLOUDFLARE_CHALLENGE gets depth=1, retry_budget=0::

        policy = TopologyTraversalPolicy(depth=3, retry_budget=4, ...)
        biased = apply_traversal_bias(policy, "CLOUDFLARE_CHALLENGE")
        assert biased.depth == 1
        assert biased.retry_budget == 0
    """
    bias = TOPOLOGY_TRAVERSAL_BIAS.get(topology_class)

    # Fast path: no bias for this topology class — return unchanged.
    if bias is None:
        return policy

    # Collect field overrides, skipping advisory-only fields.
    new_fields: Dict[str, Any] = {
        "topology_class":      policy.topology_class,
        "depth":               policy.depth,
        "render_mode":         policy.render_mode,
        "requests_per_second": policy.requests_per_second,
        "retry_budget":        policy.retry_budget,
        "timeout_ms":          policy.timeout_ms,
        "confidence":          policy.confidence,
    }

    # Map of recognized TopologyTraversalPolicy field names for validation.
    _POLICY_FIELDS = frozenset(new_fields.keys())

    for bias_field, bias_value in bias.items():

        # tor_required is advisory — log it and continue.
        # TopologyTraversalPolicy has no tor_required field; the signal
        # propagates to FrictionForecast.mitigation_strategy independently.
        if bias_field == "tor_required":
            if bias_value:
                log.debug(
                    "apply_traversal_bias: topology_class=%s has tor_required=True. "
                    "This advisory signal is not stored in TopologyTraversalPolicy. "
                    "FrictionForecast.mitigation_strategy for this class must reflect "
                    "Tor routing requirement (e.g. tor_extract or tor_headless).",
                    topology_class,
                )
            continue

        # topology_class override is not permitted through bias — it's structural identity.
        if bias_field == "topology_class":
            log.warning(
                "apply_traversal_bias: TOPOLOGY_TRAVERSAL_BIAS[%s] contains "
                "'topology_class' override — this field cannot be biased. Skipping.",
                topology_class,
            )
            continue

        # Guard against typos in the bias dict (catches maintenance errors).
        if bias_field not in _POLICY_FIELDS:
            log.warning(
                "apply_traversal_bias: TOPOLOGY_TRAVERSAL_BIAS[%s] contains "
                "unrecognized field '%s' with value %r. Skipping. "
                "Check TOPOLOGY_TRAVERSAL_BIAS for maintenance errors.",
                topology_class, bias_field, bias_value,
            )
            continue

        log.debug(
            "apply_traversal_bias: applying %s[%s] = %r (overrides model output %r).",
            topology_class, bias_field, bias_value, new_fields[bias_field],
        )
        new_fields[bias_field] = bias_value

    # Validate overridden fields before constructing.
    _validate_traversal_fields(new_fields, context=f"apply_traversal_bias({topology_class})")

    return TopologyTraversalPolicy(**new_fields)


def _validate_traversal_fields(fields: Dict[str, Any], context: str) -> None:
    """
    Internal field-level validator for traversal policy dictionaries.

    Called after bias application to catch invalid bias entries before
    contract construction.  Raises ValueError for out-of-contract values.

    Parameters
    ----------
    fields : dict
        Dictionary matching TopologyTraversalPolicy field names.

    context : str
        Caller context string for error messages.
    """
    depth = fields.get("depth")
    if depth is not None and not (DEPTH_MIN <= int(depth) <= DEPTH_MAX):
        raise ValueError(
            f"{context}: depth {depth} is outside valid range "
            f"[{DEPTH_MIN}, {DEPTH_MAX}]. "
            "Check TOPOLOGY_TRAVERSAL_BIAS — bias values must be in-range."
        )

    render_mode = fields.get("render_mode")
    if render_mode is not None and render_mode not in VALID_RENDER_MODES:
        raise ValueError(
            f"{context}: render_mode {render_mode!r} is not in "
            f"VALID_RENDER_MODES {VALID_RENDER_MODES}. "
            "TOPOLOGY_TRAVERSAL_BIAS must use 'static' or 'headless'."
        )

    rps = fields.get("requests_per_second")
    if rps is not None and not (RPS_MIN <= float(rps) <= RPS_MAX):
        raise ValueError(
            f"{context}: requests_per_second {rps} is outside valid range "
            f"[{RPS_MIN}, {RPS_MAX}]."
        )

    retry = fields.get("retry_budget")
    if retry is not None and not (RETRY_MIN <= int(retry) <= RETRY_MAX):
        raise ValueError(
            f"{context}: retry_budget {retry} is outside valid range "
            f"[{RETRY_MIN}, {RETRY_MAX}]."
        )

    timeout = fields.get("timeout_ms")
    if timeout is not None and not (TIMEOUT_MIN <= int(timeout) <= TIMEOUT_MAX):
        raise ValueError(
            f"{context}: timeout_ms {timeout} is outside valid range "
            f"[{TIMEOUT_MIN}, {TIMEOUT_MAX}]."
        )

    confidence = fields.get("confidence")
    if confidence is not None and not (0.0 <= float(confidence) <= 1.0):
        raise ValueError(
            f"{context}: confidence {confidence} is outside [0.0, 1.0]."
        )


# ─────────────────────────────────────────────────────────────────────────────
# COHERENCE ENFORCEMENT — FRICTION
# ─────────────────────────────────────────────────────────────────────────────

def _enforce_friction_coherence_state(state: _FrictionState) -> _FrictionState:
    """
    Apply structural web coherence constraints to the internal five-probability state.

    This is the primary coherence enforcement function.  It operates on
    _FrictionState, which holds all five friction probabilities including
    bot_detection_probability.

    Coherence rules encode physical constraints about friction co-occurrence
    on the real web.  These are not model corrections — the model learns
    correlations; the decoder enforces structural constraints the model cannot
    violate even if it learned incorrect correlations early in training.

    Rules (applied in listed order, each may modify the state for subsequent rules)
    -------------------------------------------------------------------------------

    Rule 1: Cloudflare implies bot detection
        If cloudflare_probability > 0.80:
            bot_detection = max(bot_detection, cloudflare * 0.90)
        Rationale: Cloudflare's challenge infrastructure IS bot detection.
        A site behind Cloudflare challenge has already triggered Cloudflare's
        bot detection system.  Separate bot detection signal is not independent
        — it is subsumed by Cloudflare's infrastructure.

    Rule 2: Auth redirect suppresses paywall
        If auth_redirect_probability > 0.70:
            paywall = min(paywall, 0.20)
        Rationale: An auth redirect means the server redirected the client to a
        login/authentication endpoint before serving any content.  A paywall is a
        content gate — it requires content to be present first.  If the server
        redirects before content, there is no content to gate.  High auth_redirect
        and high paywall simultaneously is a contradiction.

    Rule 3: Rate limiting implies active bot detection
        If rate_limit_probability > 0.80:
            bot_detection = max(bot_detection, 0.60)
        Rationale: Rate limiting at the level detectable by AXIOM (HTTP 429 or
        equivalent) implies the server has identified automated request patterns.
        Bot detection is typically the mechanism that triggers rate limiting.
        0.60 is a floor, not a certainty — the model may legitimately output
        higher values.

    Parameters
    ----------
    state : _FrictionState
        Mutable internal friction state after sigmoid activation, before floor application.

    Returns
    -------
    _FrictionState
        New _FrictionState with coherence constraints applied.
        Always returns a new instance — does not mutate the input.
    """
    cf   = state.cloudflare_probability
    pw   = state.paywall_probability
    rl   = state.rate_limit_probability
    auth = state.auth_redirect_probability
    bot  = state.bot_detection_probability

    # Rule 1: Cloudflare implies bot detection
    if cf > CLOUDFLARE_BOT_COHERENCE_THRESHOLD:
        new_bot = max(bot, cf * CLOUDFLARE_BOT_MULTIPLIER)
        if new_bot != bot:
            log.debug(
                "_enforce_friction_coherence_state [%s]: "
                "Rule 1 (CF→bot): cloudflare=%.4f > %.2f → "
                "bot_detection raised %.4f → %.4f (CF * %.2f).",
                state.topology_class, cf, CLOUDFLARE_BOT_COHERENCE_THRESHOLD,
                bot, new_bot, CLOUDFLARE_BOT_MULTIPLIER,
            )
        bot = new_bot

    # Rule 2: Auth redirect suppresses paywall
    if auth > AUTH_PAYWALL_COHERENCE_THRESHOLD:
        new_pw = min(pw, AUTH_PAYWALL_CAP)
        if new_pw != pw:
            log.debug(
                "_enforce_friction_coherence_state [%s]: "
                "Rule 2 (auth→!paywall): auth_redirect=%.4f > %.2f → "
                "paywall capped %.4f → %.4f.",
                state.topology_class, auth, AUTH_PAYWALL_COHERENCE_THRESHOLD,
                pw, new_pw,
            )
        pw = new_pw

    # Rule 3: Rate limiting implies bot detection
    if rl > RATE_LIMIT_BOT_COHERENCE_THRESHOLD:
        new_bot = max(bot, RATE_LIMIT_BOT_FLOOR)
        if new_bot != bot:
            log.debug(
                "_enforce_friction_coherence_state [%s]: "
                "Rule 3 (RL→bot): rate_limit=%.4f > %.2f → "
                "bot_detection raised %.4f → %.4f.",
                state.topology_class, rl, RATE_LIMIT_BOT_COHERENCE_THRESHOLD,
                bot, new_bot,
            )
        bot = new_bot

    return _FrictionState(
        topology_class=state.topology_class,
        cloudflare_probability=cf,
        paywall_probability=pw,
        rate_limit_probability=rl,
        auth_redirect_probability=auth,
        bot_detection_probability=bot,
    )


def enforce_friction_coherence(forecast: FrictionForecast) -> FrictionForecast:
    """
    Apply structural web coherence constraints to a FrictionForecast.

    Public-facing variant of coherence enforcement operating on the frozen
    FrictionForecast contract.  Applies the subset of coherence rules that
    can be evaluated from the four stored friction probabilities (bot_detection
    is not available in the contract; Rules 1 and 3 are applied to the stored
    fields only where applicable).

    For the full five-probability coherence enforcement, use
    _enforce_friction_coherence_state(), which is called internally during
    decode_friction() before FrictionForecast construction.

    This function exists for post-hoc coherence validation and testing.
    The primary decode pipeline always calls _enforce_friction_coherence_state().

    Parameters
    ----------
    forecast : FrictionForecast
        Frozen friction forecast to enforce coherence on.

    Returns
    -------
    FrictionForecast
        New FrictionForecast with coherence constraints applied.
        Returns a new instance even if no changes were made, to maintain
        clear immutability semantics.
        mitigation_strategy is preserved from the input — callers should
        re-derive strategy if probabilities changed significantly.

    Notes
    -----
    Rule 2 (auth suppresses paywall) is fully applicable here.
    Rules 1 and 3 (cloudflare/rate_limit imply bot detection) cannot be
    fully applied because bot_detection_probability is not in FrictionForecast.
    The auth→paywall rule is the most structurally critical for routing
    decisions and is always applied.
    """
    cf   = forecast.cloudflare_probability
    pw   = forecast.paywall_probability
    rl   = forecast.rate_limit_probability
    auth = forecast.auth_redirect_probability

    # Rule 2: Auth redirect suppresses paywall — fully applicable.
    if auth > AUTH_PAYWALL_COHERENCE_THRESHOLD:
        new_pw = min(pw, AUTH_PAYWALL_CAP)
        if new_pw != pw:
            log.debug(
                "enforce_friction_coherence [%s]: "
                "auth_redirect=%.4f > %.2f → paywall capped %.4f → %.4f.",
                forecast.topology_class, auth,
                AUTH_PAYWALL_COHERENCE_THRESHOLD, pw, new_pw,
            )
            pw = new_pw

    return FrictionForecast(
        topology_class=forecast.topology_class,
        cloudflare_probability=cf,
        paywall_probability=pw,
        rate_limit_probability=rl,
        auth_redirect_probability=auth,
        mitigation_strategy=forecast.mitigation_strategy,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FLOOR APPLICATION — FRICTION
# ─────────────────────────────────────────────────────────────────────────────

def apply_friction_floors(
    forecast:       FrictionForecast,
    topology_class: str,
) -> FrictionForecast:
    """
    Apply FRICTION_FLOORS minimum probabilities for known friction topology classes.

    Floors enforce minimum probability values for topology classes where certain
    friction types are structurally certain.  The model can predict higher
    friction than the floor; it cannot predict lower.

    bot_detection_probability floor handling
    ----------------------------------------
    FrictionForecast does not carry bot_detection_probability.  FRICTION_FLOORS
    entries for bot_detection_probability (ECOMMERCE_PRODUCT,
    ECOMMERCE_PRODUCT_VARIANT) cannot be applied at this stage — they are
    applied in _apply_friction_floors_to_state() during the internal decode
    pipeline, before FrictionForecast construction.

    This function applies floors only to the four fields present in
    FrictionForecast.  bot_detection floors are consumed internally.

    Parameters
    ----------
    forecast : FrictionForecast
        Friction forecast after coherence enforcement.

    topology_class : str
        The topology class for this decode call.
        Used as the key into FRICTION_FLOORS.

    Returns
    -------
    FrictionForecast
        New FrictionForecast with all applicable floor constraints applied.
        mitigation_strategy is NOT re-derived here — this is the caller's
        responsibility if probabilities changed significantly.

    Examples
    --------
    CLOUDFLARE_CHALLENGE ensures cloudflare_probability ≥ 0.99::

        forecast = FrictionForecast(cloudflare_probability=0.50, ...)
        floored = apply_friction_floors(forecast, "CLOUDFLARE_CHALLENGE")
        assert floored.cloudflare_probability == 0.99
    """
    floors = FRICTION_FLOORS.get(topology_class)

    if floors is None:
        return forecast

    cf   = forecast.cloudflare_probability
    pw   = forecast.paywall_probability
    rl   = forecast.rate_limit_probability
    auth = forecast.auth_redirect_probability

    _FORECAST_FLOOR_FIELDS = {
        "cloudflare_probability":    "cf",
        "paywall_probability":       "pw",
        "rate_limit_probability":    "rl",
        "auth_redirect_probability": "auth",
    }

    for field_name, floor_value in floors.items():

        # bot_detection floors cannot be applied to FrictionForecast — skip silently.
        # They are applied in _apply_friction_floors_to_state() internally.
        if field_name == "bot_detection_probability":
            continue

        if field_name not in _FORECAST_FLOOR_FIELDS:
            log.warning(
                "apply_friction_floors: FRICTION_FLOORS[%s] contains "
                "unrecognized field '%s'. Skipping.",
                topology_class, field_name,
            )
            continue

        current = locals()[_FORECAST_FLOOR_FIELDS[field_name]]
        if current < floor_value:
            log.debug(
                "apply_friction_floors [%s]: floor applied %s: %.4f → %.4f.",
                topology_class, field_name, current, floor_value,
            )
            # Assign via dictionary since we can't write to locals() directly.
            var = _FORECAST_FLOOR_FIELDS[field_name]
            if var == "cf":   cf   = floor_value
            elif var == "pw": pw   = floor_value
            elif var == "rl": rl   = floor_value
            elif var == "auth": auth = floor_value

    return FrictionForecast(
        topology_class=forecast.topology_class,
        cloudflare_probability=cf,
        paywall_probability=pw,
        rate_limit_probability=rl,
        auth_redirect_probability=auth,
        mitigation_strategy=forecast.mitigation_strategy,
    )


def _apply_friction_floors_to_state(
    state:          _FrictionState,
    topology_class: str,
) -> _FrictionState:
    """
    Apply FRICTION_FLOORS to the internal _FrictionState.

    This is the primary floor application path, called during decode_friction()
    before FrictionForecast construction.  It operates on all five probabilities
    including bot_detection_probability.

    Parameters
    ----------
    state : _FrictionState
        Friction state after coherence enforcement.

    topology_class : str
        Key into FRICTION_FLOORS.

    Returns
    -------
    _FrictionState
        New state with floors applied to all applicable fields.
    """
    floors = FRICTION_FLOORS.get(topology_class)
    if floors is None:
        return state

    cf   = state.cloudflare_probability
    pw   = state.paywall_probability
    rl   = state.rate_limit_probability
    auth = state.auth_redirect_probability
    bot  = state.bot_detection_probability

    field_map = {
        "cloudflare_probability":    ("cf",   cf),
        "paywall_probability":       ("pw",   pw),
        "rate_limit_probability":    ("rl",   rl),
        "auth_redirect_probability": ("auth", auth),
        "bot_detection_probability": ("bot",  bot),
    }

    for field_name, floor_value in floors.items():
        if field_name not in field_map:
            log.warning(
                "_apply_friction_floors_to_state: FRICTION_FLOORS[%s] contains "
                "unrecognized field '%s'. Skipping.",
                topology_class, field_name,
            )
            continue
        var_name, current = field_map[field_name]
        if current < floor_value:
            log.debug(
                "_apply_friction_floors_to_state [%s]: floor %s: %.4f → %.4f.",
                topology_class, field_name, current, floor_value,
            )
            if var_name == "cf":   cf   = floor_value
            elif var_name == "pw": pw   = floor_value
            elif var_name == "rl": rl   = floor_value
            elif var_name == "auth": auth = floor_value
            elif var_name == "bot": bot  = floor_value

    return _FrictionState(
        topology_class=topology_class,
        cloudflare_probability=cf,
        paywall_probability=pw,
        rate_limit_probability=rl,
        auth_redirect_probability=auth,
        bot_detection_probability=bot,
    )


# ─────────────────────────────────────────────────────────────────────────────
# MITIGATION STRATEGY DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

def derive_mitigation_strategy(
    cloudflare:    float,
    paywall:       float,
    rate_limit:    float,
    auth_redirect: float,
    bot_detection: float,
) -> str:
    """
    Deterministically derive Phantom's mitigation strategy from friction probabilities.

    This function is pure and deterministic.  The same probability vector always
    produces the same strategy string.  No randomness.  No model inference.
    No external state.

    Strategy priority order
    -----------------------
    Priorities are ordered by "friction severity" — the degree to which the
    friction type prevents any signal retrieval.  Higher priority strategies
    take precedence over lower ones when multiple friction types exceed their
    thresholds simultaneously.

        Priority 1: skip (auth_redirect > 0.70)
            Auth redirect means the server never serves the target content.
            No signal is possible from this URL.  Phantom must skip entirely.
            No other strategy can recover signal from an auth redirect.
            skip has priority because attempting any other strategy wastes
            resources against a wall that cannot be bypassed by the WLM.

        Priority 2: tor_extract (paywall > 0.70)
            Paywall content is accessible via anonymized extraction.
            Tor + Chromium combination can access some paywalled content via
            Tor exit nodes that have not been identified as scrapers.
            tor_extract is attempted before headless_retry because paywalls
            typically respond to JS rendering (the paywall itself is JS-gated).

        Priority 3: headless_retry (cloudflare > 0.70)
            Cloudflare challenge pages require JS execution to solve.
            headless_retry uses Chromium to complete the challenge and retry.
            This comes after tor_extract because Cloudflare and paywall are
            distinct friction types that rarely co-occur at threshold levels.

        Priority 4: slow_crawl (rate_limit > 0.70)
            Rate limiting is overcome by reducing request velocity.
            slow_crawl instructs Phantom to reduce requests_per_second further
            and add inter-request jitter.

        Priority 5: tor_headless (bot_detection > 0.60)
            Bot detection without other friction types calls for anonymized
            headless fetch.  tor_headless combines Tor routing with Chromium
            rendering to appear as a legitimate browser from a non-datacenter IP.

        Default: standard
            No friction thresholds exceeded.  Standard Phantom fetch strategy.

    Strategy strings
    ----------------
    "skip"           — No signal possible.  Do not attempt fetch.
    "tor_extract"    — Tor routing + Chromium for paywalled content.
    "headless_retry" — Chromium render + retry for Cloudflare challenges.
    "slow_crawl"     — Reduced velocity + jitter for rate-limited sites.
    "tor_headless"   — Tor routing + Chromium for bot-detected sites.
    "standard"       — No special handling needed.

    Parameters
    ----------
    cloudflare : float
        Cloudflare challenge probability in [0.0, 1.0].
    paywall : float
        Paywall probability in [0.0, 1.0].
    rate_limit : float
        Rate limit probability in [0.0, 1.0].
    auth_redirect : float
        Auth redirect probability in [0.0, 1.0].
    bot_detection : float
        Bot detection probability in [0.0, 1.0].

    Returns
    -------
    str
        One of: "skip", "tor_extract", "headless_retry", "slow_crawl",
        "tor_headless", "standard".
        Always a member of VALID_MITIGATION_STRATEGIES.

    Examples
    --------
    >>> derive_mitigation_strategy(0.0, 0.0, 0.0, 0.0, 0.0)
    'standard'
    >>> derive_mitigation_strategy(0.0, 0.0, 0.0, 0.90, 0.0)
    'skip'
    >>> derive_mitigation_strategy(0.0, 0.95, 0.0, 0.0, 0.0)
    'tor_extract'
    >>> derive_mitigation_strategy(0.85, 0.0, 0.0, 0.0, 0.0)
    'headless_retry'
    >>> derive_mitigation_strategy(0.0, 0.0, 0.90, 0.0, 0.0)
    'slow_crawl'
    >>> derive_mitigation_strategy(0.0, 0.0, 0.0, 0.0, 0.70)
    'tor_headless'
    """
    # Priority 1: auth redirect — no signal possible, must skip.
    if auth_redirect > MITIGATION_AUTH_THRESHOLD:
        return "skip"

    # Priority 2: paywall — Tor + Chromium extraction attempt.
    if paywall > MITIGATION_PAYWALL_THRESHOLD:
        return "tor_extract"

    # Priority 3: Cloudflare challenge — headless JS render + retry.
    if cloudflare > MITIGATION_CLOUDFLARE_THRESHOLD:
        return "headless_retry"

    # Priority 4: rate limiting — reduce velocity.
    if rate_limit > MITIGATION_RATE_LIMIT_THRESHOLD:
        return "slow_crawl"

    # Priority 5: bot detection — anonymized headless fetch.
    if bot_detection > MITIGATION_BOT_DETECTION_THRESHOLD:
        return "tor_headless"

    return "standard"


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT VALIDATION
# Every decoded output passes through these validators before being returned.
# Validation is performed after all decoding, bias, and floor application.
# A failed validation is a decoder bug — raise immediately with full context.
# ─────────────────────────────────────────────────────────────────────────────

def validate_traversal_policy(policy: TopologyTraversalPolicy) -> bool:
    """
    Validate a TopologyTraversalPolicy contract for internal consistency.

    This function checks every field against its documented contract range.
    A policy that passes validate_traversal_policy() is safe to return to
    latent_model.py.

    Validation is performed after bias application so the validator catches
    invalid bias entries as well as invalid activation outputs.

    Parameters
    ----------
    policy : TopologyTraversalPolicy
        The decoded and biased traversal policy.

    Returns
    -------
    bool
        True if all fields pass their contract invariants.
        Raises ValueError if any field violates its contract.

    Raises
    ------
    ValueError
        On any field that violates its contract range.
        The error message includes the field name, observed value, and
        the expected range.

    Notes
    -----
    This function raises rather than returning False on failure.
    The contract is: every decoded policy is valid, or the caller is notified
    immediately.  Returning False silently would allow invalid policies to
    propagate to Phantom, causing silent crawl misconfiguration.
    """
    errors: List[str] = []

    if not policy.topology_class:
        errors.append("topology_class is empty — every policy must identify its class.")

    if not (DEPTH_MIN <= policy.depth <= DEPTH_MAX):
        errors.append(
            f"depth={policy.depth} is outside [{DEPTH_MIN}, {DEPTH_MAX}]."
        )

    if policy.render_mode not in VALID_RENDER_MODES:
        errors.append(
            f"render_mode={policy.render_mode!r} is not in {VALID_RENDER_MODES}."
        )

    if not (RPS_MIN <= policy.requests_per_second <= RPS_MAX):
        errors.append(
            f"requests_per_second={policy.requests_per_second} "
            f"is outside [{RPS_MIN}, {RPS_MAX}]."
        )

    if not (RETRY_MIN <= policy.retry_budget <= RETRY_MAX):
        errors.append(
            f"retry_budget={policy.retry_budget} "
            f"is outside [{RETRY_MIN}, {RETRY_MAX}]."
        )

    if not (TIMEOUT_MIN <= policy.timeout_ms <= TIMEOUT_MAX):
        errors.append(
            f"timeout_ms={policy.timeout_ms} "
            f"is outside [{TIMEOUT_MIN}, {TIMEOUT_MAX}]."
        )

    if not (0.0 <= policy.confidence <= 1.0):
        errors.append(
            f"confidence={policy.confidence} is outside [0.0, 1.0]."
        )

    if errors:
        raise ValueError(
            f"validate_traversal_policy: contract violations in policy for "
            f"topology_class={policy.topology_class!r}:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return True


def validate_friction_forecast(forecast: FrictionForecast) -> bool:
    """
    Validate a FrictionForecast contract for internal consistency.

    Checks every probability field is in [0.0, 1.0], topology_class is
    non-empty, and mitigation_strategy is a recognized value.

    Parameters
    ----------
    forecast : FrictionForecast
        The decoded, coherence-enforced, and floored friction forecast.

    Returns
    -------
    bool
        True if all fields pass their contract invariants.
        Raises ValueError on any violation.

    Raises
    ------
    ValueError
        On any field that violates its contract.
    """
    errors: List[str] = []

    if not forecast.topology_class:
        errors.append("topology_class is empty.")

    for field_name in (
        "cloudflare_probability",
        "paywall_probability",
        "rate_limit_probability",
        "auth_redirect_probability",
    ):
        val = getattr(forecast, field_name)
        if not (0.0 - PROBABILITY_TOLERANCE <= val <= 1.0 + PROBABILITY_TOLERANCE):
            errors.append(
                f"{field_name}={val:.8f} is outside [0.0, 1.0]. "
                "Sigmoid activation must bound all friction probabilities."
            )

    if forecast.mitigation_strategy not in VALID_MITIGATION_STRATEGIES:
        errors.append(
            f"mitigation_strategy={forecast.mitigation_strategy!r} is not in "
            f"VALID_MITIGATION_STRATEGIES {VALID_MITIGATION_STRATEGIES}."
        )

    if errors:
        raise ValueError(
            f"validate_friction_forecast: contract violations in forecast for "
            f"topology_class={forecast.topology_class!r}:\n" +
            "\n".join(f"  - {e}" for e in errors)
        )

    return True


def validate_source_priority(
    domains: List[str],
    scores:  List[float],
) -> bool:
    """
    Validate a (domains, scores) source priority pair.

    Checks:
      - domains and scores have equal length
      - All domain strings are non-empty
      - All scores are in [0.0, 1.0] (softmax-normalized, must sum ≤ 1.0 + ε)
      - Scores are non-negative
      - Score sum is in [1.0 - ε, 1.0 + ε] for non-empty lists (softmax property)

    Empty domains with empty scores is valid — EmptyStructuralLayer state.

    Parameters
    ----------
    domains : list of str
        Domain names from source priority decode.
    scores : list of float
        Corresponding softmax-normalized scores.

    Returns
    -------
    bool
        True if the pair is valid.
        Raises ValueError on any violation.

    Raises
    ------
    ValueError
        On length mismatch, empty domain strings, out-of-range scores,
        or sum-of-scores deviating from 1.0.
    """
    if len(domains) != len(scores):
        raise ValueError(
            f"validate_source_priority: domains length {len(domains)} "
            f"!= scores length {len(scores)}. "
            "decode_source_priority must return parallel lists."
        )

    # Empty is valid — EmptyStructuralLayer path.
    if not domains:
        return True

    for i, domain in enumerate(domains):
        if not domain or not isinstance(domain, str):
            raise ValueError(
                f"validate_source_priority: domains[{i}]={domain!r} "
                "is empty or not a string. "
                "All domain entries must be non-empty strings."
            )

    for i, score in enumerate(scores):
        if not isinstance(score, (int, float)):
            raise ValueError( # noqa
                f"validate_source_priority: scores[{i}]={score!r} "
                "is not numeric."
            )
        if score < 0.0 - PROBABILITY_TOLERANCE:
            raise ValueError(
                f"validate_source_priority: scores[{i}]={score:.8f} "
                "is negative. Softmax outputs must be non-negative."
            )
        if score > 1.0 + PROBABILITY_TOLERANCE:
            raise ValueError(
                f"validate_source_priority: scores[{i}]={score:.8f} "
                "exceeds 1.0. Individual softmax values cannot exceed 1.0."
            )

    score_sum = sum(scores)
    tolerance = max(1e-4, len(scores) * 1e-6)  # tolerance grows with list length
    if not (1.0 - tolerance <= score_sum <= 1.0 + tolerance):
        raise ValueError(
            f"validate_source_priority: score sum {score_sum:.8f} "
            f"deviates from 1.0 by more than tolerance {tolerance:.2e}. "
            "Softmax over top-k scores must sum to 1.0. "
            "Check softmax computation in decode_source_priority."
        )

    return True


# ─────────────────────────────────────────────────────────────────────────────
# STRUCTURAL LAYER — LOADING AND VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def load_structural_layer(
    path: Path,
) -> Union[StructuralLayerView, EmptyStructuralLayer]:
    """
    Load structural_layer.pt from disk with full safety guarantees.

    Returns EmptyStructuralLayer on any failure condition — missing file,
    corrupted file, wrong format, failed validation.  Never raises.

    EmptyStructuralLayer is the expected return when structural_layer.pt
    does not exist (cold start, pre-Wikipedia-preparse).  Callers do not
    need to distinguish between "file missing" and "file corrupt" — both
    result in empty source priority, which Phantom handles gracefully.

    Safety
    ------
    weights_only=True is used on all torch.load() calls.  Loading arbitrary
    Python objects from model files is a security vulnerability.  The
    structural layer is deserialized as a pure tensor/list object.

    Validation
    ----------
    The loaded dict is passed through validate_structural_layer() before
    returning.  An invalid structural layer is treated equivalently to a
    missing one — EmptyStructuralLayer is returned and the failure is logged.

    Parameters
    ----------
    path : Path
        Path to structural_layer.pt on disk.
        Typically: tag/store/structural_layer.pt

    Returns
    -------
    dict or EmptyStructuralLayer
        Validated structural layer dict if load and validation succeed.
        EmptyStructuralLayer if file is missing, corrupt, or invalid.

    Notes
    -----
    This function is the only place in wlm_decoders.py that performs file I/O.
    All other functions are pure.  load_structural_layer() is called once at
    initialize() time and again by the watchdog reload handler.

    Examples
    --------
    Cold start — file not yet written by index_daemon::

        layer = load_structural_layer(Path("tag/store/structural_layer.pt"))
        isinstance(layer, EmptyStructuralLayer)  # True
        not layer.domain_index                   # True — safe to check
    """
    if not path.exists():
        log.info(
            "load_structural_layer: %s does not exist. "
            "This is expected before the first Wikipedia preparse cycle completes. "
            "Returning EmptyStructuralLayer — source priority will be empty until "
            "structural_layer.pt is written by index_daemon.",
            path,
        )
        return EmptyStructuralLayer()

    try:
        # weights_only=True: security requirement.
        # The structural layer is a pure dict of tensors and Python lists.
        # Never load arbitrary Python objects from model store files.
        layer: Dict = torch.load(
            str(path),
            map_location="cpu",  # always load to CPU; GPU placement is caller's job
            weights_only=True,
        )
    except Exception as exc:
        log.error(
            "load_structural_layer: failed to load %s: %s. "
            "Returning EmptyStructuralLayer.  "
            "This may indicate store corruption — store_watchdog should alert.",
            path, exc,
        )
        return EmptyStructuralLayer()

    if not validate_structural_layer(layer):
        log.error(
            "load_structural_layer: %s loaded but failed validation. "
            "Returning EmptyStructuralLayer.  "
            "This indicates index_daemon wrote a malformed structural layer.",
            path,
        )
        return EmptyStructuralLayer()

    n_domains = len(layer["domain_index"])
    log.info(
        "load_structural_layer: loaded %s successfully. "
        "domains=%d, source_matrix=%s, intent_clusters=%s.",
        path,
        n_domains,
        tuple(layer["source_matrix"].shape),
        tuple(layer["intent_clusters"].shape) if layer.get("intent_clusters") is not None else "None",
    )

    return StructuralLayerView.from_dict(layer)


def _clamp_probability(
    value:      float,
    field_name: str,
    context:    str,
) -> float:
    """
    Clamp a probability value to [0.0, 1.0] with a WARNING log if out of range.

    This function is the canonical clamping path for all probability fields.
    Calling it instead of inline ``max(0.0, min(1.0, value))`` ensures that
    every out-of-range value is logged before clamping.  Silent clamping hides
    systematic model miscalibration.

    Parameters
    ----------
    value : float
        The probability value to clamp.

    field_name : str
        Name of the probability field.  Used in the log message.

    context : str
        Caller context (topology_class or function name).  Used in log message.

    Returns
    -------
    float
        Value clamped to [0.0, 1.0].

    Notes
    -----
    Values within [-PROBABILITY_TOLERANCE, 1.0 + PROBABILITY_TOLERANCE] are
    clamped silently (floating-point noise).  Values outside this extended range
    are logged at WARNING.
    """
    if value < 0.0:
        if value < -PROBABILITY_TOLERANCE:
            log.warning(
                "_clamp_probability [%s]: %s=%.10f is below 0.0. "
                "Clamping to 0.0.  Sigmoid activation should never produce "
                "negative values — this indicates floating-point pathology "
                "or a non-sigmoid activation was used.",
                context, field_name, value,
            )
        return 0.0

    if value > 1.0:
        if value > 1.0 + PROBABILITY_TOLERANCE:
            log.warning(
                "_clamp_probability [%s]: %s=%.10f exceeds 1.0. "
                "Clamping to 1.0.  Sigmoid activation should never exceed 1.0 — "
                "this indicates floating-point pathology or a non-sigmoid "
                "activation was used.",
                context, field_name, value,
            )
        return 1.0

    return value


def _assert_tensor_finite(
    tensor:         torch.Tensor,
    name:           str,
    topology_class: str,
) -> None:
    """
    Assert a tensor contains only finite values, raise with diagnostic detail if not.

    Called by all primary decoders as a pre-flight check.  Non-finite values
    indicate gradient explosion, weight corruption, or a programming error in
    the model's forward pass.  They cannot be meaningfully decoded.

    Parameters
    ----------
    tensor : torch.Tensor
        The tensor to check.

    name : str
        Tensor name for error messages (e.g. "raw traversal", "source embedding").

    topology_class : str
        Topology class context for error messages.

    Raises
    ------
    ValueError
        If any value in the tensor is NaN or infinite.
        Error message includes positions of non-finite values and their counts.
    """
    if torch.isfinite(tensor).all():
        return

    nan_mask = torch.isnan(tensor)
    inf_mask = torch.isinf(tensor)
    n_nan    = nan_mask.sum().item()
    n_inf    = inf_mask.sum().item()

    # Get indices of non-finite values for diagnostic output.
    nan_positions = nan_mask.nonzero(as_tuple=False).squeeze(-1).tolist()
    inf_positions = inf_mask.nonzero(as_tuple=False).squeeze(-1).tolist()

    raise ValueError(
        f"{name} tensor contains non-finite values "
        f"for topology_class={topology_class!r}. "
        f"NaN count={n_nan}, positions={nan_positions}. "
        f"Inf count={n_inf}, positions={inf_positions}. "
        f"Tensor shape={tuple(tensor.shape)}, dtype={tensor.dtype}. "
        "Non-finite model output indicates gradient explosion, weight corruption, "
        "or numerical instability in the MambaRouter forward pass. "
        "Check gradient clipping and weight initialization in the training loop."
    )



def validate_structural_layer(layer: Dict) -> bool:
    """
    Validate the structural layer dictionary for required fields and shape consistency.

    Validation rules
    ----------------
    1. Required keys: source_matrix, domain_index, intent_clusters, cluster_domains.
    2. source_matrix must be a 2D float tensor of shape (n_domains, 512).
    3. domain_index must be a list of strings with length matching source_matrix dim 0.
    4. intent_clusters must be a 2D float tensor of shape (n_clusters, 512).
    5. cluster_domains must be a list of lists; outer length matches intent_clusters dim 0.
    6. n_domains must be > 0 for a valid structural layer.
       (An empty structural layer is represented by EmptyStructuralLayer, not an empty dict.)

    Parameters
    ----------
    layer : dict
        Dictionary loaded from structural_layer.pt.

    Returns
    -------
    bool
        True if the layer passes all validation checks.
        False if any check fails (caller logs and returns EmptyStructuralLayer).

    Notes
    -----
    This function does not raise — it returns False and logs.
    load_structural_layer() acts on the False return.
    """
    REQUIRED_KEYS = {"source_matrix", "domain_index", "intent_clusters", "cluster_domains"}
    EMBEDDING_DIM = 512

    # Check required keys exist.
    missing = REQUIRED_KEYS - set(layer.keys())
    if missing:
        log.error(
            "validate_structural_layer: missing required keys: %s. "
            "Expected: %s.  index_daemon must write all four fields.",
            sorted(missing), sorted(REQUIRED_KEYS),
        )
        return False

    source_matrix = layer["source_matrix"]
    domain_index  = layer["domain_index"]
    intent_clusters = layer["intent_clusters"]
    cluster_domains = layer["cluster_domains"]

    # Validate source_matrix shape.
    if not isinstance(source_matrix, torch.Tensor):
        log.error(
            "validate_structural_layer: source_matrix is %s, expected torch.Tensor.",
            type(source_matrix).__name__,
        )
        return False

    if source_matrix.ndim != 2:
        log.error(
            "validate_structural_layer: source_matrix.ndim=%d, expected 2. "
            "Shape: %s.",
            source_matrix.ndim, tuple(source_matrix.shape),
        )
        return False

    n_domains, embedding_dim = source_matrix.shape
    if embedding_dim != EMBEDDING_DIM:
        log.error(
            "validate_structural_layer: source_matrix embedding dim=%d, "
            "expected %d.  Weights may have been built with wrong architecture.",
            embedding_dim, EMBEDDING_DIM,
        )
        return False

    if n_domains == 0:
        log.error(
            "validate_structural_layer: source_matrix has 0 rows (n_domains=0). "
            "A valid structural layer must have at least one domain. "
            "Use EmptyStructuralLayer for the zero-domain state.",
        )
        return False

    # Validate domain_index length.
    if not isinstance(domain_index, list):
        log.error(
            "validate_structural_layer: domain_index is %s, expected list.",
            type(domain_index).__name__,
        )
        return False

    if len(domain_index) != n_domains:
        log.error(
            "validate_structural_layer: domain_index length %d "
            "!= source_matrix rows %d.  These must match exactly — "
            "row i of source_matrix corresponds to domain_index[i].",
            len(domain_index), n_domains,
        )
        return False

    # Spot-check domain_index entries are strings.
    for i, domain in enumerate(domain_index[:10]):
        if not isinstance(domain, str) or not domain:
            log.error(
                "validate_structural_layer: domain_index[%d]=%r is not "
                "a non-empty string.  All domain names must be strings.",
                i, domain,
            )
            return False

    # Validate intent_clusters if present (may be None early in training).
    if intent_clusters is not None:
        if not isinstance(intent_clusters, torch.Tensor):
            log.error(
                "validate_structural_layer: intent_clusters is %s, "
                "expected torch.Tensor or None.",
                type(intent_clusters).__name__,
            )
            return False

        if intent_clusters.ndim != 2:
            log.error(
                "validate_structural_layer: intent_clusters.ndim=%d, expected 2.",
                intent_clusters.ndim,
            )
            return False

        if intent_clusters.shape[1] != EMBEDDING_DIM:
            log.error(
                "validate_structural_layer: intent_clusters embedding dim=%d, "
                "expected %d.",
                intent_clusters.shape[1], EMBEDDING_DIM,
            )
            return False

        n_clusters = intent_clusters.shape[0]

        if not isinstance(cluster_domains, list):
            log.error(
                "validate_structural_layer: cluster_domains is %s, expected list.",
                type(cluster_domains).__name__,
            )
            return False

        if len(cluster_domains) != n_clusters:
            log.error(
                "validate_structural_layer: cluster_domains length %d "
                "!= intent_clusters rows %d.",
                len(cluster_domains), n_clusters,
            )
            return False

    # Validate source_matrix floating point type.
    if not source_matrix.is_floating_point():
        log.error(
            "validate_structural_layer: source_matrix dtype=%s is not floating point. "
            "Dot product scoring requires float32 or float64.",
            source_matrix.dtype,
        )
        return False

    return True


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY DECODER: TRAVERSAL
# ─────────────────────────────────────────────────────────────────────────────

def decode_traversal(
    raw:            torch.Tensor,
    topology_class: str,
) -> TopologyTraversalPolicy:
    """
    Decode the (7,) traversal head output into a TopologyTraversalPolicy contract.

    This is the primary traversal decoder.  It applies all activations, range
    enforcement, topology-specific bias overrides, and output validation.

    Raw tensor layout (must be treated exactly as specified)
    --------------------------------------------------------
    raw[0] → depth               sigmoid * 4 + 1 → [1, 5] int
    raw[1] → render_mode         sigmoid > 0.60 → "headless"|"static"
    raw[2] → requests_per_second softplus + 0.1 → [0.1, 100.0] float
    raw[3] → retry_budget        sigmoid * 5 → [0, 5] int
    raw[4] → timeout_ms          softplus * 1000 + 1000 → [1000, 30000] int
    raw[5] → tor_required        sigmoid > 0.70 → bool (advisory, not in contract)
    raw[6] → confidence          sigmoid → [0.0, 1.0] float

    Processing pipeline
    -------------------
    1. Input validation — shape, dtype, finite values.
    2. Per-dimension activation application.
    3. Range clamping with WARNING logs for out-of-range values.
    4. Topology-specific bias application via apply_traversal_bias().
    5. Contract construction.
    6. Output validation via validate_traversal_policy().

    tor_required handling
    ---------------------
    raw[5] is decoded and logged for observability.  TopologyTraversalPolicy
    does not have a tor_required field — the signal influences
    FrictionForecast.mitigation_strategy independently through FRICTION_FLOORS.
    The TOPOLOGY_TRAVERSAL_BIAS entry for NEWS_ARTICLE_PAYWALLED sets
    tor_required=True; this is logged and does not raise.

    Parameters
    ----------
    raw : torch.Tensor
        Shape (7,).  Raw float output from MambaRouter traversal_head.
        Must be finite (no NaN, no Inf).

    topology_class : str
        The topology class for this decode call.
        Used for topology-specific bias application.
        May be any string — unknown classes receive no bias.

    Returns
    -------
    TopologyTraversalPolicy
        Frozen, validated traversal policy.
        Never None.  Every code path returns a valid contract or raises.

    Raises
    ------
    ValueError
        If raw tensor has wrong shape, wrong dtype, or contains non-finite values.
        If decoded values fail post-bias validation (indicates invalid bias dict entry).

    Examples
    --------
    Zero-initialized model output — moderate defaults::

        raw = torch.zeros(7)
        policy = decode_traversal(raw, "GENERIC_HTML")
        # depth=3, render_mode="static", rps≈0.793, retry=2, timeout≈1693ms

    Bias overrides activation::

        raw = torch.ones(7) * -5.0   # all activations → minimums
        policy = decode_traversal(raw, "SAAS_DOCS")
        assert policy.depth == 4     # TOPOLOGY_TRAVERSAL_BIAS overrides to 4

    CLOUDFLARE_CHALLENGE bias::

        policy = decode_traversal(raw, "CLOUDFLARE_CHALLENGE")
        assert policy.depth == 1
        assert policy.retry_budget == 0
        assert policy.requests_per_second == 0.1
    """
    # ── Input validation ──────────────────────────────────────────────────────

    if not isinstance(raw, torch.Tensor):
        raise ValueError( # noqa
            f"decode_traversal: raw must be torch.Tensor, got {type(raw).__name__}. "
            "MambaRouter traversal_head must return a tensor."
        )

    if raw.shape != (7,):
        raise ValueError(
            f"decode_traversal: raw.shape={tuple(raw.shape)}, expected (7,). "
            "The traversal head must output exactly 7 dimensions as specified."
        )

    if not raw.is_floating_point():
        raise ValueError(
            f"decode_traversal: raw.dtype={raw.dtype} is not floating point. "
            "Traversal head outputs must be float tensors."
        )

    if not torch.isfinite(raw).all():
        nan_mask = torch.isnan(raw)
        inf_mask = torch.isinf(raw)
        raise ValueError(
            f"decode_traversal: raw contains non-finite values for "
            f"topology_class={topology_class!r}. "
            f"NaN positions: {nan_mask.nonzero(as_tuple=False).squeeze().tolist()}. "
            f"Inf positions: {inf_mask.nonzero(as_tuple=False).squeeze().tolist()}. "
            "Non-finite model output indicates gradient explosion or weight corruption. "
            "Check MambaRouter training loop and gradient clipping."
        )
    # Convert to Python scalars for activation functions.
    # float64 for precision during intermediate computation.
    raw_cpu = raw.detach().cpu().to(torch.float64)

    r0 = raw_cpu[0].item()   # depth logit
    r1 = raw_cpu[1].item()   # render_mode logit
    r2 = raw_cpu[2].item()   # requests_per_second logit
    r3 = raw_cpu[3].item()   # retry_budget logit
    r4 = raw_cpu[4].item()   # timeout_ms logit
    r5 = raw_cpu[5].item()   # tor_required logit
    r6 = raw_cpu[6].item()   # confidence logit

    depth               = bounded_depth(r0)
    render_mode         = render_mode_from_logit(r1)
    requests_per_second = bounded_rps(r2)
    retry_budget        = bounded_retry(r3)
    timeout_ms          = bounded_timeout(r4)
    tor_required        = tor_from_logit(r5)
    confidence          = torch.sigmoid(torch.tensor(r6, dtype=torch.float64)).item()

    # Clamp and log confidence separately — it uses sigmoid directly.
    if confidence < 0.0:
        log.warning(
            "decode_traversal [%s]: confidence %.8f < 0.0. Clamping to 0.0.",
            topology_class, confidence,
        )
        confidence = 0.0
    if confidence > 1.0:
        log.warning(
            "decode_traversal [%s]: confidence %.8f > 1.0. Clamping to 1.0.",
            topology_class, confidence,
        )
        confidence = 1.0

    # Log tor_required advisory signal before bias application.
    if tor_required:
        log.debug(
            "decode_traversal [%s]: tor_required=True from model output "
            "(raw[5]=%.4f, sigmoid=%.4f > %.2f). "
            "This signal is advisory — TopologyTraversalPolicy has no tor_required field. "
            "Tor routing is enforced via FrictionForecast.mitigation_strategy.",
            topology_class, r5,
            torch.sigmoid(torch.tensor(r5, dtype=torch.float64)).item(),
            TOR_REQUIRED_THRESHOLD,
        )

    # ── Pre-bias contract construction ────────────────────────────────────────
    pre_bias_policy = TopologyTraversalPolicy(
        topology_class=topology_class,
        depth=depth,
        render_mode=render_mode,
        requests_per_second=requests_per_second,
        retry_budget=retry_budget,
        timeout_ms=timeout_ms,
        confidence=confidence,
    )

    # ── Bias application ──────────────────────────────────────────────────────
    # apply_traversal_bias() returns a new contract with bias overrides applied.
    # If no bias exists for this topology_class, pre_bias_policy is returned unchanged.
    policy = apply_traversal_bias(pre_bias_policy, topology_class)

    # ── Output validation ─────────────────────────────────────────────────────
    # validate_traversal_policy() raises on any contract violation.
    validate_traversal_policy(policy)

    log.debug(
        "decode_traversal [%s]: "
        "depth=%d, render_mode=%s, rps=%.3f, retry=%d, timeout=%dms, "
        "confidence=%.4f, tor_advisory=%s.",
        topology_class,
        policy.depth, policy.render_mode, policy.requests_per_second,
        policy.retry_budget, policy.timeout_ms, policy.confidence,
        tor_required,
    )

    return policy


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY DECODER: FRICTION
# ─────────────────────────────────────────────────────────────────────────────

def decode_friction(
    raw:            torch.Tensor,
    topology_class: str,
) -> FrictionForecast:
    """
    Decode the (5,) friction head output into a FrictionForecast contract.

    This is the primary friction decoder.  It applies sigmoid activations,
    enforces structural web coherence constraints, applies topology-specific
    floor values, derives the mitigation strategy, constructs the contract,
    and validates the result.

    Raw tensor layout (must be treated exactly as specified)
    --------------------------------------------------------
    raw[0] → cloudflare_probability    sigmoid → [0.0, 1.0]
    raw[1] → paywall_probability       sigmoid → [0.0, 1.0]
    raw[2] → rate_limit_probability    sigmoid → [0.0, 1.0]
    raw[3] → auth_redirect_probability sigmoid → [0.0, 1.0]
    raw[4] → bot_detection_probability sigmoid → [0.0, 1.0]

    All five dimensions use sigmoid without exception.  No other activation
    is applied to friction outputs.

    Processing pipeline
    -------------------
    1. Input validation — shape, dtype, finite values.
    2. sigmoid activation to all five dimensions → _FrictionState.
    3. Coherence enforcement (_enforce_friction_coherence_state).
    4. Floor application (_apply_friction_floors_to_state).
    5. Mitigation strategy derivation (derive_mitigation_strategy).
    6. FrictionForecast contract construction (via _FrictionState.to_forecast).
    7. Output validation (validate_friction_forecast).

    bot_detection_probability
    -------------------------
    raw[4] is decoded and used for coherence enforcement (Rules 1 and 3),
    floor application (ECOMMERCE_PRODUCT, ECOMMERCE_PRODUCT_VARIANT floors),
    and mitigation strategy derivation.  It is NOT stored in FrictionForecast —
    the contract does not carry this field.  Its effect is fully expressed
    through mitigation_strategy.

    Parameters
    ----------
    raw : torch.Tensor
        Shape (5,).  Raw float output from MambaRouter friction_head.
        Must be finite (no NaN, no Inf).

    topology_class : str
        The topology class for this decode call.
        Used for floor application and strategy context.

    Returns
    -------
    FrictionForecast
        Frozen, validated friction forecast.
        Never None.  Every code path returns a valid contract or raises.

    Raises
    ------
    ValueError
        If raw tensor has wrong shape, wrong dtype, or contains non-finite values.
        If decoded values fail post-floor validation.

    Examples
    --------
    Zero-initialized model output — moderate friction::

        raw = torch.zeros(5)
        forecast = decode_friction(raw, "GENERIC_HTML")
        # All probabilities ≈ 0.5, strategy="tor_extract" (paywall > 0.70 not met,
        # but paywall=0.5 < 0.70; cloudflare=0.5 < 0.70; all < thresholds)
        # → strategy="standard" since no threshold exceeded at 0.5.

    CLOUDFLARE_CHALLENGE floor enforcement::

        raw = torch.zeros(5)  # model outputs 0.5 for all
        forecast = decode_friction(raw, "CLOUDFLARE_CHALLENGE")
        assert forecast.cloudflare_probability == 0.99
        assert forecast.mitigation_strategy == "headless_retry"

    AUTH_REDIRECT coherence — paywall suppressed::

        raw = torch.tensor([0.0, 3.0, 0.0, 3.0, 0.0])
        # paywall_logit=3.0 → sigmoid≈0.95; auth_redirect_logit=3.0 → ≈0.95
        forecast = decode_friction(raw, "GENERIC_HTML")
        assert forecast.paywall_probability <= 0.20   # auth coherence suppresses paywall
        assert forecast.mitigation_strategy == "skip"  # auth wins priority
    """
    # ── Input validation ──────────────────────────────────────────────────────

    if not isinstance(raw, torch.Tensor):
        raise ValueError( # noqa
            f"decode_friction: raw must be torch.Tensor, got {type(raw).__name__}."
        )

    if raw.shape != (5,):
        raise ValueError(
            f"decode_friction: raw.shape={tuple(raw.shape)}, expected (5,). "
            "The friction head must output exactly 5 dimensions."
        )

    if not raw.is_floating_point():
        raise ValueError(
            f"decode_friction: raw.dtype={raw.dtype} is not floating point."
        )

    if not torch.isfinite(raw).all():
        nan_mask = torch.isnan(raw)
        inf_mask = torch.isinf(raw)
        raise ValueError(
            f"decode_friction: raw contains non-finite values for "
            f"topology_class={topology_class!r}. "
            f"NaN positions: {nan_mask.nonzero(as_tuple=False).squeeze().tolist()}. "
            f"Inf positions: {inf_mask.nonzero(as_tuple=False).squeeze().tolist()}."
        )

    # ── Activation application ─────────────────────────────────────────────
    raw_cpu = raw.detach().cpu().to(torch.float64)

    cf   = friction_probability(raw_cpu[0].item())
    pw   = friction_probability(raw_cpu[1].item())
    rl   = friction_probability(raw_cpu[2].item())
    auth = friction_probability(raw_cpu[3].item())
    bot  = friction_probability(raw_cpu[4].item())

    # ── Build internal friction state ────────────────────────────────────────
    state = _FrictionState(
        topology_class=topology_class,
        cloudflare_probability=cf,
        paywall_probability=pw,
        rate_limit_probability=rl,
        auth_redirect_probability=auth,
        bot_detection_probability=bot,
    )

    # ── Coherence enforcement ─────────────────────────────────────────────────
    # Structural constraints about friction co-occurrence on the real web.
    # Applied before floors so coherence cannot pull values below floor minimums.
    state = _enforce_friction_coherence_state(state)

    # ── Floor application ─────────────────────────────────────────────────────
    # Minimum probabilities for known friction topology classes.
    # Applied after coherence — floors represent physical certainties that
    # coherence rules cannot contradict.
    state = _apply_friction_floors_to_state(state, topology_class)

    # ── Mitigation strategy derivation ────────────────────────────────────────
    # Deterministic from final probability values.
    # Uses all five probabilities including bot_detection.
    strategy = derive_mitigation_strategy(
        cloudflare=state.cloudflare_probability,
        paywall=state.paywall_probability,
        rate_limit=state.rate_limit_probability,
        auth_redirect=state.auth_redirect_probability,
        bot_detection=state.bot_detection_probability,
    )

    # ── Contract construction ─────────────────────────────────────────────────
    forecast = state.to_forecast(strategy)

    # ── Output validation ─────────────────────────────────────────────────────
    validate_friction_forecast(forecast)

    log.debug(
        "decode_friction [%s]: "
        "cf=%.4f, pw=%.4f, rl=%.4f, auth=%.4f, bot=%.4f → strategy=%s.",
        topology_class,
        state.cloudflare_probability,
        state.paywall_probability,
        state.rate_limit_probability,
        state.auth_redirect_probability,
        state.bot_detection_probability,
        strategy,
    )

    return forecast


# ─────────────────────────────────────────────────────────────────────────────
# PRIMARY DECODER: SOURCE PRIORITY
# ─────────────────────────────────────────────────────────────────────────────

def decode_source_priority(
    raw:             torch.Tensor,
    structural_layer: Union[StructuralLayerView, EmptyStructuralLayer],
    topology_class:  str,
    phase:           int,
    k:               int = 10,
) -> Tuple[List[str], List[float]]:
    """
    Decode the (512,) source embedding into a ranked domain list with confidence scores.

    This is the most computationally intensive decoder.  It performs a matrix
    multiply between the source embedding and the structural layer source_matrix,
    selects the top-k scoring domains, normalizes scores via softmax, and returns
    the ranked (domains, scores) pair.

    Computation
    -----------
    1. source_embedding = raw                           # (512,)
    2. scores = source_matrix @ source_embedding        # (n_domains,)  ← dot product only
    3. top_k_indices = topk(scores, k=k)
    4. source_priority = [domain_index[i] for i in top_k_indices]
    5. top_k_scores = softmax(raw_top_k_scores)         # normalized within top-k only

    Dot product, not cosine similarity
    -----------------------------------
    source_matrix rows are L2-normalized during offline encoding by index_daemon.
    For L2-normalized vectors, dot product IS cosine similarity without the
    division.  Using cosine similarity here would repeat the normalization
    unnecessarily and add division overhead.  Do NOT add cosine similarity
    normalization — it is already accounted for in the encoding step.

    Phase-aware k selection
    -----------------------
    k determines the breadth of source exploration:

        Phase I  (learns):    k=10 — explore broadly, world model is building
        Phase II (predicts):  k=7  — narrowing, world model has partial coverage
        Phase III (knows):    k=3  — compiled policy, direct to best 3 sources

    The caller-provided k overrides K_BY_PHASE defaults.  When k is the
    default (10), phase adjustments from K_BY_PHASE are applied automatically.

    Phase III WLM is more decisive, not more exploratory.  The shorter list
    reflects higher confidence in source selection, not less knowledge.

    EmptyStructuralLayer handling
    -----------------------------
    When structural_layer has no domain_index (EmptyStructuralLayer or empty
    loaded layer), return ([], []).  Empty source priority is valid — Phantom
    falls back to URL-based routing.  This is expected before the first
    Wikipedia preparse cycle.

    Score normalization
    -------------------
    Raw dot product scores are not probabilities.  Scores are normalized via
    softmax over the top-k only (not over all n_domains).  This gives relative
    confidence within the shortlist — callers can use scores to rank within
    the returned list and to reason about how decisive the selection is.

    A high score for rank-1 and low scores for rank 2-k indicates strong
    confidence in the top source.  Uniform scores indicate the model is
    uncertain among the candidates.

    Parameters
    ----------
    raw : torch.Tensor
        Shape (512,).  Raw float output from MambaRouter source_head.
        This is the query embedding vector.  Must be finite.

    structural_layer : dict or EmptyStructuralLayer
        The loaded structural layer.  Must contain source_matrix (n_domains, 512)
        and domain_index (List[str] of length n_domains).
        EmptyStructuralLayer returns ([], []) immediately.

    topology_class : str
        The topology class for this decode call.  Used for logging context only —
        source priority computation is topology-agnostic (the embedding encodes
        the query intent, not the topology class).

    phase : int
        Current WLM phase (1, 2, or 3).  Used to select k from K_BY_PHASE
        when k is the default value of 10.

    k : int, optional
        Number of domains to return.  Default 10.
        Override by caller for specific use cases.
        Must be > 0 and ≤ len(domain_index).
        K_BY_PHASE provides the phase-aware defaults.

    Returns
    -------
    Tuple[List[str], List[float]]
        (domains, scores) where:
            domains: List of domain name strings, length k (or 0 if empty layer).
            scores: List of softmax-normalized floats, same length.
                    Scores sum to 1.0.  Higher = more confident match.
        Callers that only need domains may ignore scores.
        Callers building friction forecasts per source should use scores.

    Raises
    ------
    ValueError
        If raw tensor has wrong shape, wrong dtype, or contains non-finite values.
        If k ≤ 0 or k > n_domains (with appropriate clamping and logging).

    Notes
    -----
    Empty source priority ([], []) is not an error.
    Phantom falls back to URL-based routing when source priority is empty.
    Document-level callers should check ``if domains:`` before using the result.

    Examples
    --------
    Empty structural layer — cold start::

        layer = EmptyStructuralLayer()
        domains, scores = decode_source_priority(raw, layer, "SAAS_DOCS", phase=1)
        assert domains == []
        assert scores == []

    Normal decode with phase-aware k::

        domains, scores = decode_source_priority(raw, layer, "SAAS_DOCS", phase=3)
        assert len(domains) == 3   # Phase III: k=3
        assert abs(sum(scores) - 1.0) < 1e-4  # softmax sums to 1.0
    """
    # ── Input validation ──────────────────────────────────────────────────────

    if not isinstance(raw, torch.Tensor):
        raise ValueError( # noqa
            f"decode_source_priority: raw must be torch.Tensor, got {type(raw).__name__}."
        )

    if raw.shape != (512,):
        raise ValueError(
            f"decode_source_priority: raw.shape={tuple(raw.shape)}, expected (512,). "
            "The source head must output exactly 512 dimensions."
        )

    if not raw.is_floating_point():
        raise ValueError(
            f"decode_source_priority: raw.dtype={raw.dtype} is not floating point."
        )

    if not torch.isfinite(raw).all():
        raise ValueError(
            f"decode_source_priority: raw contains non-finite values for "
            f"topology_class={topology_class!r}. "
            "Non-finite source embedding indicates model weight corruption."
        )

    # ── EmptyStructuralLayer fast path ────────────────────────────────────────
    # Check domain_index truthiness — EmptyStructuralLayer returns [].
    # This is the expected state before Wikipedia preparse completes.
    if not structural_layer.domain_index:
        log.debug(
            "decode_source_priority [%s, phase=%d]: "
            "structural_layer has no domains (EmptyStructuralLayer or cold start). "
            "Returning empty source priority — Phantom will use URL-based routing.",
            topology_class, phase,
        )
        return [], []

    # ── Phase-aware k resolution ──────────────────────────────────────────────
    # K_BY_PHASE provides phase defaults.  Caller-provided k overrides
    # only if it differs from the function default (10).
    # If k == 10 (default), use K_BY_PHASE for the current phase.
    if k == 10 and phase in K_BY_PHASE:
        effective_k = K_BY_PHASE[phase]
        log.debug(
            "decode_source_priority [%s, phase=%d]: "
            "k=%d (default) → phase-adjusted k=%d from K_BY_PHASE.",
            topology_class, phase, k, effective_k,
        )
    else:
        effective_k = k

    # ── k bounds enforcement ──────────────────────────────────────────────────
    n_domains = len(structural_layer.domain_index)

    if effective_k <= 0:
        raise ValueError(
            f"decode_source_priority [%s, phase=%d]: "
            "effective_k=%d must be > 0." % (topology_class, phase, effective_k)
        )

    if effective_k > n_domains:
        log.warning(
            "decode_source_priority [%s, phase=%d]: "
            "effective_k=%d exceeds n_domains=%d. "
            "Clamping k to n_domains.  "
            "This may indicate the structural layer was built on fewer domains "
            "than expected, or phase/k parameters are misconfigured.",
            topology_class, phase, effective_k, n_domains,
        )
        effective_k = n_domains

    # ── Source embedding preparation ──────────────────────────────────────────
    # Move to CPU float32 for matrix multiply — source_matrix is float32.
    # float64 query against float32 matrix would require upcasting the matrix;
    # float32 is sufficient for ranking purposes.
    query = raw.detach().cpu().to(torch.float32)

    # ── Source matrix retrieval ───────────────────────────────────────────────
    source_matrix: torch.Tensor = structural_layer.source_matrix
    # source_matrix: (n_domains, 512)

    # Ensure source_matrix is on the same device as query (CPU).
    if source_matrix.device != query.device:
        source_matrix = source_matrix.cpu()

    # Ensure matching dtypes for matrix multiply.
    if source_matrix.dtype != query.dtype:
        source_matrix = source_matrix.to(query.dtype)

    # ── Dot product scoring ───────────────────────────────────────────────────
    # scores = source_matrix @ query  →  (n_domains,)
    # Each score is the dot product of domain embedding row i with the query.
    # source_matrix rows are L2-normalized, so this equals cosine similarity
    # without the division overhead.
    #
    # Do NOT use torch.nn.functional.cosine_similarity — the normalization
    # is already handled in the offline encoding step by index_daemon.
    with torch.no_grad():
        scores = torch.mv(source_matrix, query)  # (n_domains,)

    # ── Top-k selection ────────────────────────────────────────────────────────
    # torch.topk returns (values, indices) sorted descending by default.
    top_k_scores_raw, top_k_indices = torch.topk(
        scores,
        k=effective_k,
        largest=True,
        sorted=True,  # descending order — highest score first
    )

    # ── Score normalization: softmax over top-k only ───────────────────────────
    # Softmax over top-k gives relative confidence within the shortlist.
    # NOT softmax over all n_domains — that would dilute scores by irrelevant domains.
    # The top-k scores represent the candidate shortlist; their relative magnitudes
    # are what matter for caller confidence reasoning.
    top_k_scores_normalized = torch.softmax(
        top_k_scores_raw.to(torch.float64),
        dim=0,
    ).tolist()

    # ── Domain name retrieval ─────────────────────────────────────────────────
    domain_index: List[str] = structural_layer.domain_index
    domains = [domain_index[i.item()] for i in top_k_indices]

    # ── Output validation ─────────────────────────────────────────────────────
    validate_source_priority(domains, top_k_scores_normalized)

    log.debug(
        "decode_source_priority [%s, phase=%d, k=%d]: "
        "top domain=%s (score=%.4f), n_domains=%d.",
        topology_class, phase, effective_k,
        domains[0] if domains else "none",
        top_k_scores_normalized[0] if top_k_scores_normalized else 0.0,
        n_domains,
    )

    return domains, top_k_scores_normalized


# ─────────────────────────────────────────────────────────────────────────────
# BATCH DECODERS
# Used by cold_start_warmup() to pre-populate policy caches for all 18
# topology classes before interface.py accepts its first query.
#
# Batch functions loop over individual decoders — they do not implement
# separate batch logic.  This ensures the validation and bias application
# in each individual decoder is exercised for every entry, not bypassed
# in a vectorized batch path.
# ─────────────────────────────────────────────────────────────────────────────

def decode_traversal_batch(
    raws:             torch.Tensor,
    topology_classes: List[str],
) -> List[TopologyTraversalPolicy]:
    """
    Decode a batch of traversal head outputs into TopologyTraversalPolicy contracts.

    Calls decode_traversal() for each row in raws.  Individual decoder contracts,
    validations, and bias applications are fully honored for each entry.  No
    batch-level shortcuts that bypass individual validation.

    Used by WorldLatentModel.cold_start_warmup() to pre-populate the traversal
    policy cache for all 18 topology classes in a single pass.

    Parameters
    ----------
    raws : torch.Tensor
        Shape (n, 7).  Each row is a raw traversal head output for one
        topology class.  Must be finite throughout.

    topology_classes : list of str
        Length n.  topology_classes[i] corresponds to raws[i].

    Returns
    -------
    list of TopologyTraversalPolicy
        Length n.  policies[i] is the decoded policy for topology_classes[i].
        Every entry is a valid, frozen contract — no Nones in the list.

    Raises
    ------
    ValueError
        If raws.shape[0] != len(topology_classes).
        If raws.ndim != 2 or raws.shape[1] != 7.
        Propagates ValueError from individual decode_traversal() calls.

    Examples
    --------
    Warm all 18 known topology classes::

        raws = torch.zeros(18, 7)
        policies = decode_traversal_batch(raws, TOPOLOGY_CLASSES)
        assert len(policies) == 18
        assert all(isinstance(p, TopologyTraversalPolicy) for p in policies)
    """
    if not isinstance(raws, torch.Tensor):
        raise ValueError( # noqa
            f"decode_traversal_batch: raws must be torch.Tensor, "
            f"got {type(raws).__name__}."
        )

    if raws.ndim != 2:
        raise ValueError(
            f"decode_traversal_batch: raws.ndim={raws.ndim}, expected 2. "
            "Shape must be (n, 7)."
        )

    if raws.shape[1] != 7:
        raise ValueError(
            f"decode_traversal_batch: raws.shape[1]={raws.shape[1]}, expected 7."
        )

    n = raws.shape[0]
    if n != len(topology_classes):
        raise ValueError(
            f"decode_traversal_batch: raws.shape[0]={n} != "
            f"len(topology_classes)={len(topology_classes)}. "
            "Each raw tensor row must have a corresponding topology class."
        )

    if not raws.is_floating_point():
        raise ValueError(
            f"decode_traversal_batch: raws.dtype={raws.dtype} is not floating point."
        )

    policies: List[TopologyTraversalPolicy] = []

    for i in range(n):
        try:
            policy = decode_traversal(raws[i], topology_classes[i])
            policies.append(policy)
        except Exception as exc:
            # Individual decode failure in batch context — log with full context
            # and re-raise.  cold_start_warmup() must not silently skip entries.
            log.error(
                "decode_traversal_batch: failed at index %d "
                "(topology_class=%s): %s",
                i, topology_classes[i], exc,
            )
            raise

    log.info(
        "decode_traversal_batch: decoded %d traversal policies. "
        "topology_classes: %s.",
        len(policies),
        [p.topology_class for p in policies],
    )

    return policies


def decode_friction_batch(
    raws:             torch.Tensor,
    topology_classes: List[str],
) -> List[FrictionForecast]:
    """
    Decode a batch of friction head outputs into FrictionForecast contracts.

    Calls decode_friction() for each row in raws.  Individual decoder contracts,
    coherence enforcement, floor application, and mitigation derivation are
    fully honored for each entry.  No batch shortcuts.

    Used by WorldLatentModel.cold_start_warmup() to pre-populate the friction
    forecast cache for all 18 topology classes.

    Parameters
    ----------
    raws : torch.Tensor
        Shape (n, 5).  Each row is a raw friction head output for one
        topology class.  Must be finite throughout.

    topology_classes : list of str
        Length n.  topology_classes[i] corresponds to raws[i].

    Returns
    -------
    list of FrictionForecast
        Length n.  forecasts[i] is the decoded forecast for topology_classes[i].
        Every entry is a valid, frozen contract.

    Raises
    ------
    ValueError
        If raws.shape[0] != len(topology_classes).
        If raws.ndim != 2 or raws.shape[1] != 5.
        Propagates ValueError from individual decode_friction() calls.

    Examples
    --------
    Warm all 18 known topology classes::

        raws = torch.zeros(18, 5)
        forecasts = decode_friction_batch(raws, TOPOLOGY_CLASSES)
        assert len(forecasts) == 18
        assert all(isinstance(f, FrictionForecast) for f in forecasts)

    Verify floor enforcement across batch::

        cf_idx = TOPOLOGY_CLASSES.index("CLOUDFLARE_CHALLENGE")
        assert forecasts[cf_idx].cloudflare_probability == 0.99
    """
    if not isinstance(raws, torch.Tensor):
        raise ValueError( # noqa
            f"decode_friction_batch: raws must be torch.Tensor, "
            f"got {type(raws).__name__}."
        )

    if raws.ndim != 2:
        raise ValueError(
            f"decode_friction_batch: raws.ndim={raws.ndim}, expected 2. "
            "Shape must be (n, 5)."
        )

    if raws.shape[1] != 5:
        raise ValueError(
            f"decode_friction_batch: raws.shape[1]={raws.shape[1]}, expected 5."
        )

    n = raws.shape[0]
    if n != len(topology_classes):
        raise ValueError(
            f"decode_friction_batch: raws.shape[0]={n} != "
            f"len(topology_classes)={len(topology_classes)}. "
            "Each raw tensor row must have a corresponding topology class."
        )

    if not raws.is_floating_point():
        raise ValueError(
            f"decode_friction_batch: raws.dtype={raws.dtype} is not floating point."
        )

    forecasts: List[FrictionForecast] = []

    for i in range(n):
        try:
            forecast = decode_friction(raws[i], topology_classes[i])
            forecasts.append(forecast)
        except Exception as exc:
            log.error(
                "decode_friction_batch: failed at index %d "
                "(topology_class=%s): %s",
                i, topology_classes[i], exc,
            )
            raise

    log.info(
        "decode_friction_batch: decoded %d friction forecasts. "
        "strategies: %s.",
        len(forecasts),
        {f.topology_class: f.mitigation_strategy for f in forecasts},
    )

    return forecasts


# ─────────────────────────────────────────────────────────────────────────────
# MODULE SELF-TEST
# Lightweight invariant checks run at import time to catch constant definition
# errors before the first decoder call.
#
# These are not substitutes for the full test suite.  They catch:
# - TOPOLOGY_TRAVERSAL_BIAS entries referencing invalid field names (caught by
#   apply_traversal_bias runtime check, but detected earlier here)
# - FRICTION_FLOORS entries with probability values outside [0.0, 1.0]
# - K_BY_PHASE missing expected phase keys
# - VALID_MITIGATION_STRATEGIES missing required strategy values
#
# If any assertion fails, the module raises at import time — a crash at startup
# is infinitely preferable to a silent misconfiguration that produces invalid
# crawl policies.
# ─────────────────────────────────────────────────────────────────────────────

def _self_test_constants() -> None:
    """
    Validate module-level constants for internal consistency.

    Called once at module import.  Raises AssertionError on any violation.
    """
    # ── K_BY_PHASE completeness ────────────────────────────────────────────────
    assert 1 in K_BY_PHASE, "K_BY_PHASE must contain phase 1 (PHASE_I)."
    assert 2 in K_BY_PHASE, "K_BY_PHASE must contain phase 2 (PHASE_II)."
    assert 3 in K_BY_PHASE, "K_BY_PHASE must contain phase 3 (PHASE_III)."
    assert K_BY_PHASE[1] >= K_BY_PHASE[2] >= K_BY_PHASE[3] >= 1, (
        "K_BY_PHASE must be monotonically non-increasing: Phase I >= II >= III >= 1. "
        "Exploration breadth decreases as phase increases."
    )

    # ── FRICTION_FLOORS probability range ─────────────────────────────────────
    for topo, floors in FRICTION_FLOORS.items():
        for field_name, floor_val in floors.items():
            assert 0.0 <= floor_val <= 1.0, (
                f"FRICTION_FLOORS[{topo}][{field_name}]={floor_val} "
                "is outside [0.0, 1.0]. Floors must be probabilities."
            )

    # ── TOPOLOGY_TRAVERSAL_BIAS value types ───────────────────────────────────
    _VALID_BIAS_FIELDS = frozenset({
        "depth", "render_mode", "requests_per_second",
        "retry_budget", "timeout_ms", "tor_required",
    })
    for topo, biases in TOPOLOGY_TRAVERSAL_BIAS.items():
        for field_name, bias_val in biases.items():
            assert field_name in _VALID_BIAS_FIELDS, (
                f"TOPOLOGY_TRAVERSAL_BIAS[{topo}] contains unknown field "
                f"'{field_name}'. Valid fields: {_VALID_BIAS_FIELDS}."
            )
        # Validate render_mode bias values are in VALID_RENDER_MODES.
        if "render_mode" in biases:
            assert biases["render_mode"] in VALID_RENDER_MODES, (
                f"TOPOLOGY_TRAVERSAL_BIAS[{topo}]['render_mode']="
                f"{biases['render_mode']!r} is not in {VALID_RENDER_MODES}."
            )
        # Validate depth bias values are in range.
        if "depth" in biases:
            assert DEPTH_MIN <= biases["depth"] <= DEPTH_MAX, (
                f"TOPOLOGY_TRAVERSAL_BIAS[{topo}]['depth']={biases['depth']} "
                f"is outside [{DEPTH_MIN}, {DEPTH_MAX}]."
            )
        # Validate retry_budget bias values.
        if "retry_budget" in biases:
            assert RETRY_MIN <= biases["retry_budget"] <= RETRY_MAX, (
                f"TOPOLOGY_TRAVERSAL_BIAS[{topo}]['retry_budget']="
                f"{biases['retry_budget']} is outside [{RETRY_MIN}, {RETRY_MAX}]."
            )
        # Validate requests_per_second bias values.
        if "requests_per_second" in biases:
            assert RPS_MIN <= biases["requests_per_second"] <= RPS_MAX, (
                f"TOPOLOGY_TRAVERSAL_BIAS[{topo}]['requests_per_second']="
                f"{biases['requests_per_second']} is outside [{RPS_MIN}, {RPS_MAX}]."
            )

    # ── VALID_MITIGATION_STRATEGIES contains all derived values ───────────────
    required_strategies = {"standard", "slow_crawl", "headless_retry",
                           "tor_extract", "tor_headless", "skip"}
    assert required_strategies == VALID_MITIGATION_STRATEGIES, (
        f"VALID_MITIGATION_STRATEGIES is missing required values. "
        f"Expected: {required_strategies}. "
        f"Found: {VALID_MITIGATION_STRATEGIES}."
    )

    # ── Threshold ordering ─────────────────────────────────────────────────────
    assert RENDER_MODE_THRESHOLD > 0.50, (
        f"RENDER_MODE_THRESHOLD={RENDER_MODE_THRESHOLD} must be > 0.50 "
        "(prefer static over headless)."
    )
    assert TOR_REQUIRED_THRESHOLD > RENDER_MODE_THRESHOLD, (
        f"TOR_REQUIRED_THRESHOLD={TOR_REQUIRED_THRESHOLD} must exceed "
        f"RENDER_MODE_THRESHOLD={RENDER_MODE_THRESHOLD} "
        "(Tor requires stronger signal than headless)."
    )
    assert TIMEOUT_MIN < TIMEOUT_MAX, (
        f"TIMEOUT_MIN={TIMEOUT_MIN} must be less than TIMEOUT_MAX={TIMEOUT_MAX}."
    )
    assert RPS_MIN > 0.0, (
        f"RPS_MIN={RPS_MIN} must be positive (zero RPS is not a valid policy)."
    )
    assert DEPTH_MIN >= 1, "DEPTH_MIN must be at least 1."
    assert DEPTH_MAX <= 10, "DEPTH_MAX must be at most 10 (sanity bound)."

    # ── EmptyStructuralLayer attribute access contract ────────────────────────
    empty = EmptyStructuralLayer()
    assert empty.source_matrix   is None,  "EmptyStructuralLayer.source_matrix must be None."
    assert empty.domain_index    == [],    "EmptyStructuralLayer.domain_index must be []."
    assert empty.intent_clusters is None,  "EmptyStructuralLayer.intent_clusters must be None."
    assert empty.cluster_domains == [],    "EmptyStructuralLayer.cluster_domains must be []."
    assert not empty,                      "EmptyStructuralLayer must be falsy."
    assert not empty.domain_index,         "EmptyStructuralLayer.domain_index must be falsy."


# Run self-test at import time.
_self_test_constants()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE PUBLIC API SURFACE
# Explicit listing of the symbols that latent_model.py imports.
# Anything not in __all__ is internal implementation detail.
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    # Primary decoders
    "decode_traversal",
    "decode_friction",
    "decode_source_priority",

    # Bias and floor application (public for testing)
    "apply_traversal_bias",
    "apply_friction_floors",
    "enforce_friction_coherence",

    # Mitigation strategy
    "derive_mitigation_strategy",

    # Standalone activation functions (public for isolated testing)
    "bounded_depth",
    "bounded_rps",
    "bounded_timeout",
    "bounded_retry",
    "render_mode_from_logit",
    "tor_from_logit",
    "friction_probability",

    # Structural layer
    "EmptyStructuralLayer",
    "StructuralLayerView",
    "load_structural_layer",
    "validate_structural_layer",

    # Output validation
    "validate_traversal_policy",
    "validate_friction_forecast",
    "validate_source_priority",

    # Batch decoders
    "decode_traversal_batch",
    "decode_friction_batch",

    # Constants (importable for test assertions)
    "TOPOLOGY_TRAVERSAL_BIAS",
    "FRICTION_FLOORS",
    "K_BY_PHASE",
    "RENDER_MODE_THRESHOLD",
    "TOR_REQUIRED_THRESHOLD",
    "DEPTH_MIN",
    "DEPTH_MAX",
    "RPS_MIN",
    "RPS_MAX",
    "TIMEOUT_MIN",
    "TIMEOUT_MAX",
    "RETRY_MIN",
    "RETRY_MAX",
    "VALID_RENDER_MODES",
    "VALID_MITIGATION_STRATEGIES",
    "SOURCE_PRIORITY_K_PHASE_I",
    "SOURCE_PRIORITY_K_PHASE_II",
    "SOURCE_PRIORITY_K_PHASE_III",
]