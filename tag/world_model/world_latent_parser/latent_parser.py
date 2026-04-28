"""
tag/world_model/latent_parser.py
================================
AXIOM WLP Model and Orchestration Layer.

The intelligence of the World Latent Parser.

Three components live here and nowhere else:

    1. LatentParser (nn.Module)
       The GraphSAGE model. Three SAGEConv layers with residual connections
       and layer normalization. Four output heads: node classification,
       boundary type, confidence, and zone prototype.
       Trained by preparse_daemon.py on Wikipedia parse trees.
       Called at inference time by WorldLatentParser._l3_fresh_parse().

    2. WorldLatentParser (public class)
       The only public-facing component in tag/world_model/ for WLP.
       Owns the three-tier cache (L1/L2/L3).
       Subscribes to the bus (CleanSignalEvent, SurpriseEvent).
       Registers with WATCHDOG (structural_layer.pt).
       Orchestrates: wlp_graph → LatentParser → wlp_zones → ZoneMap.
       Exposes exactly one public method: query().

    3. WLP = WorldLatentParser()
       Module-level singleton. Initialized at import time, warm before
       interface.py accepts queries.

Dependency direction:
    latent_parser.py → wlp_graph.py (cst_to_pyg_graph)
    latent_parser.py → wlp_zones.py (assemble_zone_map, ZoneMap, etc.)
    latent_parser.py → contracts.py (constants)
    latent_parser.py → exceptions.py (error types)
    latent_parser.py → crawler_bus.py (BUS, event types)
    latent_parser.py → store_watchdog.py (WATCHDOG)

    latent_parser.py NEVER imports from latent_model.py.
    The WLM and WLP are peers. They are not in a hierarchy.

Mathematical foundation:
    GraphSAGE (Hamilton et al. 2017) with mean aggregation.
    InfoNCE contrastive loss (van den Oord et al. 2018) for zone prototypes.
    PCA via truncated SVD for zone coherence validation.
    Focal loss modulation (Lin et al. 2017) for boundary class imbalance.
    Platt-calibrated confidence with isotonic monotonicity enforcement.
    Exponential moving average for cache hit rate statistics.
    Jaccard index on selector sets for structural similarity.
    Information Gain Ratio for discovery zone boundary evaluation.
    Reciprocal Rank Fusion for multi-signal priority aggregation.
    Cosine similarity on the unit hypersphere for prototype clustering.

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import copy # noqa
import dataclasses
import functools # noqa
import hashlib
import logging # noqa
import math
import os
import time
import traceback
import types # noqa
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field # noqa
from pathlib import Path
from typing import ( # noqa
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.nn.functional as F # noqa
from torch import Tensor

try:
    from torch_geometric.nn import SAGEConv
    from torch_geometric.data import Data
except ImportError as _pyg_err:
    raise ImportError(
        "torch_geometric is required by latent_parser.py. "
        "Install with: pip install torch-geometric"
    ) from _pyg_err

import structlog

# ─────────────────────────────────────────────────────────────────────────────
# COMPAT: torch.linalg.LinAlgError was added in PyTorch 1.9.
# Fall back to RuntimeError on older builds so the import resolves cleanly.
# ─────────────────────────────────────────────────────────────────────────────

_LinAlgError: type = getattr(torch.linalg, "LinAlgError", RuntimeError)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL IMPORTS
# Dependency direction: latent_parser.py → {wlp_graph, wlp_zones, contracts,
#                        exceptions, crawler_bus, store_watchdog}
# latent_parser.py NEVER imports from latent_model.py.
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.contracts import ( # noqa
    TOPOLOGY_CLASSES,
    PHASE_I,
    PHASE_II,
    PHASE_III,
    THETA_CONFIDENCE_II,
    THETA_CONFIDENCE_III,
    THETA_WLP_MIN,
    SIGNAL_DENSITY_THRESHOLD,
    PhaseInt,
    ConfidenceFloat,
    TopologyClassStr,
)

from signal_kernel.exceptions import ( # noqa
    WLPQueryFailed,
    EventBusSubscriptionError,
)

from tag.world_model.world_latent_parser.wlp_graph import ( # noqa
    cst_to_pyg_graph,
    CSTNode,
    TOPOLOGY_CLASS_INDEX,
)

from tag.world_model.world_latent_parser.wlp_zones import ( # noqa
    assemble_zone_map,
    classify_nodes,
    group_signal_nodes,
    parse_intent_tags,
    apply_intent_weights,
    generate_css_selector,
    determine_scope,
    infer_content_type,
    compute_density,
    assign_priorities,
    select_extraction_strategy,
    identify_boundaries,
    validate_zone_map,
    ZoneMap,
    ZoneDescriptor,
    make_candidate_zone,
    BoundaryDescriptor,
    IntentTags,
    EmptyZoneMap,
    EmptyZoneKnowledge,
    ExtractionStrategy,
    NODE_SIGNAL,
    NODE_NOISE,
    NODE_BOUNDARY,
    CONFIDENCE_THRESHOLD,
    BOUNDARY_CONFIDENCE_THRESHOLD,
    DISCOVERY_CONFIDENCE_CEILING,
    MIN_ZONE_CONFIDENCE,
    SECTION_SCOPED_CLASSES,
    INTENT_TOKEN_VOCABULARY,
    ZoneMerkleDAG,
    MerkleDiff,
)

from tag.crawler_bus import (
    BUS,
    CleanSignalEvent,
    SurpriseEvent,
    ZoneMapUpdatedEvent,
    ZoneMapInvalidatedEvent,
    TopicEmitter,
)

from tag.store_watchdog import WATCHDOG

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# ─────────────────────────────────────────────────────────────────────────────

log: structlog.BoundLogger = structlog.get_logger("axiom.wlp.parser")

# ─────────────────────────────────────────────────────────────────────────────
# STORE PATHS
# ─────────────────────────────────────────────────────────────────────────────

STRUCTURAL_LAYER_PATH: Path = Path("/store/structural_layer.pt")
WLP_MODEL_CHECKPOINT_PATH: Path = Path("/store/wlp_model_checkpoint.pt")

# ─────────────────────────────────────────────────────────────────────────────
# L1 CACHE ENTRY
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class _CacheEntry:
    """
    The value type stored in WorldLatentParser._l1_cache.

    Wraps a ZoneMap with its Merkle DAG, computed once at _l1_store() time.
    The DAG is never recomputed on cache hits — it is the structural identity
    of the ZoneMap at the moment it was stored, and it does not change unless
    the ZoneMap itself is replaced by a new L3 parse.

    Why the DAG is not recomputed on hit:
        _l1_lookup() is on the hot query path (<0.5ms target).
        ZoneMerkleDAG.from_zone_map() is O(n) in zone count — not O(1).
        Building the DAG on every cache hit would destroy the L1 latency budget.
        The DAG is built once at _l1_store() and amortized across all hits.

    Why confidence decay does not invalidate the DAG:
        Confidence is not a DAG input. ZoneMerkleDAG._zone_leaf() hashes only
        structural geometry: selector, selector_type, scope, content_type,
        average_depth, density, priority. A cache entry whose confidence was
        decayed by a partial SurpriseEvent is still structurally valid — the
        same selectors still address the same zones on the same page. Only a new
        L3 parse producing different selectors or scopes produces a new DAG.

    Why with_intent() does not invalidate the DAG:
        with_intent() calls dataclasses.replace(zone_map, intent_weights=...)
        on the stored zone_map and returns the result to the caller. The _CacheEntry
        itself is never replaced — the intent-conditioned ZoneMap is ephemeral,
        produced per-query and never cached. The dag field therefore always reflects
        the structural geometry of the base zone_map, not any intent variant.

    Surgical eviction contract:
        On a partial SurpriseEvent carrying a surprise_zone_selector:
            dag.contains_selector(selector) → True  → decay zone_map.confidence
            dag.contains_selector(selector) → False → leave entry completely untouched
        On a dissolve SurpriseEvent:
            Entire entry is evicted by _l1_evict_by_topology_class(), DAG irrelevant.
        On watchdog structural_layer.pt reload:
            dag.diff(new_dag).is_empty → True  → cache entry still valid, no eviction
            dag.diff(new_dag).is_empty → False → replace with new entry from reload
    """
    zone_map: ZoneMap
    dag:      ZoneMerkleDAG

# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL CONSTANTS
# Single source of truth for every operational constant used in this file.
# Each constant has documented rationale. Do not duplicate these in code.
# ─────────────────────────────────────────────────────────────────────────────

# ── Cache Configuration ──────────────────────────────────────────────────────

L1_CACHE_MAX_SIZE: int = 10_000
"""
Maximum L1 cache entries (domain, topology_class) → ZoneMap.
At ~2KB per ZoneMap: 10,000 × 2KB = ~20MB memory footprint.
LRU eviction via OrderedDict ensures most-recently-used domains stay warm.
Production hit rates at maturity: >92% observed.
"""

L2_WARMUP_PRIORITY: List[str] = [
    "WIKIPEDIA_ARTICLE",
    "SAAS_DOCS",
    "REST_API_JSON",
    "NEWS_ARTICLE",
    "BLOG_POST",
    "FORUM_THREAD",
    "JSON_LD_STRUCTURED",
    "ECOMMERCE_PRODUCT",
    "SAAS_DOCS_VERSIONED",
    "SAAS_DOCS_WITH_CODE",
    "REST_API_JSON_PAGINATED",
    "ECOMMERCE_PRODUCT_VARIANT",
    "NEWS_ARTICLE_PAYWALLED",
    "LANDING_PAGE",
    "AUTH_REDIRECT",
    "CLOUDFLARE_CHALLENGE",
    "RATE_LIMITED",
    "GENERIC_HTML",
]
"""
L2 cache warmup order. Descending expected confidence.
WIKIPEDIA_ARTICLE first: 6.7M training examples, highest confidence.
GENERIC_HTML last: most heterogeneous, lowest confidence.
Cosmetic ordering — all 18 are processed regardless.
"""

# ── Zone Coherence ───────────────────────────────────────────────────────────

COHERENCE_THRESHOLD: float = 0.15
"""
Maximum prototype variance within a zone before triggering a split.
Empirically calibrated on Wikipedia parse trees.

Mathematical justification:
    Zone prototype vectors are L2-normalized to the unit hypersphere S^63.
    For a coherent zone, all prototypes cluster near a centroid.
    Variance of points on a 64-dim unit sphere:
        Uniform random: E[variance] ≈ 1.0 / d = 0.0156 per dimension
        Total variance for random: ≈ 1.0
    Coherent zone (tight cluster): variance ≈ 0.01-0.05
    Incoherent zone (two clusters): variance ≈ 0.15-0.30
    Threshold 0.15 catches the boundary between tight and bimodal.

Lower threshold → more aggressive splitting (higher precision, lower recall).
Higher threshold → more permissive grouping (lower precision, higher recall).
0.15 balances both for production use.
"""

COHERENCE_MIN_ZONE_SIZE: int = 4
"""
Minimum nodes in a zone to perform coherence validation.
Below 4 nodes, PCA is unreliable: 4 points in 64-dim space are always
linearly separable — the first principal component captures noise, not
genuine bimodal structure. Skip coherence validation for tiny zones.
"""

COHERENCE_MAX_SPLIT_DEPTH: int = 3
"""
Maximum recursive split depth for zone coherence.
Each split produces two sub-zones, each of which may itself be incoherent.
Depth 3 allows: zone → 2 sub-zones → 4 sub-sub-zones → 8 atomic zones.
Deeper splitting indicates the zone grouping in wlp_zones.py is
fundamentally wrong — return to discovery rather than keep splitting.
"""

# ── Similarity and Discovery ─────────────────────────────────────────────────

JACCARD_SIMILARITY_THRESHOLD: float = 0.85
"""
Jaccard index threshold for structural stability.
|A ∩ B| / |A ∪ B| > 0.85 means the zone selector sets are essentially
identical modulo minor noise. A Wikipedia article adding one section
has Jaccard ≈ 0.95 against its previous zone structure.
A complete redesign has Jaccard < 0.30.
"""

DISCOVERY_CONFIRMATION_THRESHOLD: int = 10
"""
Number of confirmed high-quality extractions before emitting a
subclass candidacy event. Prevents premature subclass promotion
from a small number of coincidental matches.
"""

SUBCLASS_CANDIDACY_THRESHOLD: int = 10
"""
Alias for DISCOVERY_CONFIRMATION_THRESHOLD. Used in discover_signal_zones()
to determine when a domain's zone pattern is stable enough for subclass
promotion. 10 confirmations × ~85% Jaccard = statistically stable.
"""

CONFIDENCE_BOOST_PER_CONFIRMATION: float = 0.02
"""
Additive confidence increment per Jaccard-confirmed parse.
Applied when Jaccard > 0.85 and confirmation count >= 10.
Capped at 0.95 — no ZoneMap ever reaches 1.0 confidence.
The geometric interpretation: each confirmation reduces the radius
of the confidence credible interval by approximately sqrt(1/N)
where N is the confirmation count.
"""

CONFIDENCE_MAX: float = 0.95
"""
Hard ceiling for zone map confidence.
Even perfectly confirmed zone maps retain 5% uncertainty budget
to prevent overconfident caching of potentially stale structures.
"""

CONFIDENCE_FLOOR: float = 0.30
"""
Hard floor for confidence after surprise decay.
Below 0.30, topology/parser.py falls back to GENERIC_HTML hardcoded
recipe. The floor prevents useful ZoneMaps from crossing this threshold
due to minor surprise events.
"""

SURPRISE_DECAY_RATE: float = 0.05
"""
Per-unit surprise score confidence decay multiplier.
decay = 1.0 - SURPRISE_DECAY_RATE * surprise_score
For surprise_score=1.0 (maximum): decay = 0.95, losing 5% confidence.
For surprise_score=5.0 (extreme): decay = 0.75, losing 25%.
Linear decay is appropriate because surprise_score is already
calibrated to be approximately proportional to structural divergence.
"""

# ── Contrastive Loss ─────────────────────────────────────────────────────────

CONTRASTIVE_TEMPERATURE: float = 0.07
"""
InfoNCE temperature parameter τ.

The temperature controls the sharpness of the softmax distribution
over similarity scores:
    P(positive | anchor) = exp(sim(z_i, z_j+) / τ) /
                           Σ_k exp(sim(z_i, z_k) / τ)

At τ → 0: distribution collapses to argmax (hard assignment).
At τ → ∞: distribution approaches uniform (no discrimination).

τ = 0.07 is the SimCLR-calibrated value (Chen et al. 2020):
    - Produces well-separated clusters on unit hypersphere S^63.
    - Cosine similarity range [-1, 1] → scaled to [-14.3, 14.3].
    - Softmax operates in a numerically stable regime.
    - Training is stable: gradient magnitudes are bounded.

Do not tune τ casually. Lower τ = sharper distribution = harder negatives
= better separation but unstable training. Higher τ = softer distribution
= easier training but worse cluster separation.
"""

CONTRASTIVE_NEGATIVES: int = 5
"""
Number of negative samples per positive pair in InfoNCE loss.

Full pairwise contrastive: O(n²) where n = nodes per graph.
At n = 10,000: 100M pairs. Infeasible per batch.

5 negatives per positive is the InfoNCE approximation:
    - O(5n) computation per batch.
    - At n = 10,000: 50,000 comparisons. Feasible.
    - Noise-contrastive estimation theory (Gutmann & Hyvärinen 2010)
      shows log(K) negatives approximate the full partition function
      with bias O(1/K). At K=5: bias ≈ 0.2, acceptable.

Sampling strategy: random without replacement within the batch,
seeded per-batch for reproducibility.
"""

# ── Loss Weights ─────────────────────────────────────────────────────────────

DEFAULT_LOSS_WEIGHTS: Tuple[float, float, float, float] = (1.0, 0.3, 0.2, 0.5)
"""
Multi-task loss combination weights:
    λ_1 = 1.0  — L_classification (primary)
    λ_2 = 0.3  — L_boundary (auxiliary, sparse signal)
    λ_3 = 0.2  — L_confidence (auxiliary, regularization)
    λ_4 = 0.5  — L_contrastive (zone prototype learning)

Pareto-optimal weights for multi-task GraphSAGE on DOM structure:
    λ_1 is the anchor — classification is always the primary objective.
    λ_2 is lowered because BOUNDARY nodes are rare (~5-10% of nodes).
        Higher λ_2 causes gradient oscillation on batches with few boundaries.
    λ_3 is the smallest — confidence calibration is a soft regularizer.
        It should not dominate the gradient during early training.
    λ_4 is moderate — contrastive loss has its own temperature scaling.
        Too high: prototype learning dominates, classification degrades.
        Too low: prototypes are uninformative, zone coherence validation fails.
"""

DEFAULT_CLASS_WEIGHTS: Tuple[float, float, float] = (1.0, 1.0, 8.0)
"""
Cross-entropy class weights for [SIGNAL, NOISE, BOUNDARY].
BOUNDARY gets 8× weight to compensate for class imbalance.

Mathematical justification (inverse frequency weighting):
    Training distribution: ~40% SIGNAL, ~50% NOISE, ~10% BOUNDARY.
    Inverse frequency: SIGNAL=2.5, NOISE=2.0, BOUNDARY=10.0
    Normalized to SIGNAL=1.0: NOISE=0.8, BOUNDARY=4.0

    But BOUNDARY precision matters more than recall for recipe compilation:
        A missed BOUNDARY is recoverable (DEPTH_FIRST traversal default).
        A wrong BOUNDARY produces incorrect sed/awk delimiter patterns.
    Empirical calibration on Wikipedia: 8.0× BOUNDARY weight produces
    optimal F1 for boundary detection with acceptable SIGNAL recall.
"""

# ── Focal Loss ───────────────────────────────────────────────────────────────

FOCAL_LOSS_GAMMA: float = 2.0
"""
Focal loss focusing parameter (Lin et al. 2017).

Modulates cross-entropy by (1 - p_t)^γ where p_t is the predicted
probability of the true class:
    FL(p_t) = -(1 - p_t)^γ · log(p_t)

At γ = 0: standard cross-entropy (no focusing).
At γ = 2: well-classified examples (p_t > 0.8) contribute ~4% of their
           standard CE loss. Hard examples (p_t < 0.2) retain ~96%.

Applied on top of class weights for the classification head:
    Total_loss_i = w_c · (1 - p_t)^γ · CE(logits_i, label_i)
    where w_c is the class weight for the true class c.

This double compensation (class weights + focal) is deliberate:
    Class weights handle inter-class frequency imbalance.
    Focal loss handles intra-class difficulty imbalance.
    Together: the model focuses on hard BOUNDARY examples specifically.
"""

FOCAL_LOSS_ENABLED: bool = True
"""
Toggle focal loss modulation. Disabled during warmup (first 100 steps)
to allow initial gradient flow from all examples before focusing.
"""

# ── Numerical Stability ─────────────────────────────────────────────────────

EPS: float = 1e-8
"""
Machine epsilon guard for division, log, and sqrt operations.
Prevents NaN/Inf in:
    - log(p) when p ≈ 0 in cross-entropy computation
    - 1/||v|| when v ≈ 0 in L2 normalization
    - sqrt(variance) when variance ≈ 0 in coherence check
"""

LOG_SAFE_MIN: float = 1e-12
"""
Minimum argument to torch.log() for numerical safety.
torch.log(0) = -inf → NaN in gradients.
torch.log(LOG_SAFE_MIN) ≈ -27.6 → large but finite.
Clamp probabilities to [LOG_SAFE_MIN, 1.0] before log.
"""

INF_PROXY: float = 1e9
"""
Large finite value used in place of float('inf') in torch operations.
Some CUDA kernels produce NaN from inf × 0 interactions.
1e9 is large enough to dominate any realistic logit value
and small enough to avoid overflow in exp().
"""

# ── Training Interface ───────────────────────────────────────────────────────

TRAINING_WARMUP_STEPS: int = 100
"""
Number of training steps before enabling focal loss and confidence loss.
During warmup, use standard cross-entropy only. This allows the model
to learn basic representations before focusing on hard examples.
"""

# ── Health Monitoring ────────────────────────────────────────────────────────

HEALTH_L1_HIT_RATE_WINDOW: int = 1000
"""
Sliding window for L1 cache hit rate computation.
EMA with decay = 2 / (WINDOW + 1) ≈ 0.002.
Smoothed hit rate converges after ~500 queries.
"""

HEALTH_L2_HIT_RATE_WINDOW: int = 500
"""
Sliding window for L2 cache hit rate computation.
Shorter window because L2 misses are rarer and more significant.
"""

# ── Topology Router Version ─────────────────────────────────────────────────

TOPOLOGY_ROUTER_VERSION_KEY: str = "topology_router_version"
"""
Key in structural_layer.pt metadata for the WLM router version.
WLP reads this to determine ZoneMap staleness without importing
from latent_model.py. The dependency boundary is enforced by
reading from the shared store file, not from the WLM code.
"""


# ═════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class WLPConfig:
    """
    Complete hyperparameter configuration for the LatentParser model.
    Stored in structural_layer.pt alongside weights.
    Loaded at inference time — model architecture reconstructed from config.

    Never hardcoded in the model class — always read from config.
    preparse_daemon.py may override defaults during training.

    All fields have documented defaults with mathematical justification.
    Changing any field changes the model architecture and requires retraining.
    """

    # ── Architecture ─────────────────────────────────────────────────────────

    node_feature_dim: int = 128
    """
    Input feature dimension from wlp_graph.py.
    128 = 18 (topology) + 18 (node_type) + 16 (css_class) + 8 (attribute)
          + 8 (structural_position) + 16 (content) + 16 (pattern) + 28 (intent)
    This is a hard contract with wlp_graph.py — changing this requires
    changing all 8 feature group dimensions in parallel.
    """

    hidden_dim: int = 256
    """
    Hidden dimension for all SAGEConv layers and output heads.
    256 provides sufficient capacity for DOM structural representation.

    Capacity analysis:
        3 SAGEConv layers × 256 × 256 ≈ 200K parameters per layer.
        4 output heads ≈ 100K parameters total.
        Total model: ~700K parameters.
        At float32: ~2.8MB model size. Fits in L2 cache on any GPU.

    Lower (128): reduces capacity, struggles with complex page structures.
    Higher (512): overfits to Wikipedia-specific patterns, slower inference.
    256 is the Goldilocks dimension for DOM graph classification.
    """

    prototype_dim: int = 64
    """
    Zone prototype embedding dimension.
    64-dim unit hypersphere S^63 for prototype clustering.

    The prototype space is lower-dimensional than hidden_dim because:
        Prototypes encode zone identity, not node features.
        Zone identity has lower intrinsic dimensionality than node features.
        64 dimensions provide enough capacity for ~100 distinct zone types
        with clear separation on the hypersphere.
        (Johnson-Lindenstrauss lemma: n points in R^d can be projected
        to R^{O(log n / ε²)} with 1±ε distance preservation.
        For 100 zones, ε=0.1: need ~46 dimensions. 64 has margin.)
    """

    num_sage_layers: int = 3
    """
    Number of SAGEConv layers. Three. Not two, not four.
    See spec for full justification based on DOM depth analysis.
    """

    num_node_classes: int = 3
    """SIGNAL=0, NOISE=1, BOUNDARY=2."""

    num_boundary_types: int = 3
    """SECTION_BOUNDARY, CONTENT_BOUNDARY, NOISE_BOUNDARY."""

    dropout: float = 0.1
    """
    Dropout rate applied between SAGEConv layers (training only).
    0.1 is conservative: DOM graph classification has low overfitting risk
    because training graphs are structurally diverse (6.7M articles).
    Higher dropout (0.3-0.5) degrades boundary detection — BOUNDARY nodes
    are sparse, and dropout on their representations is proportionally
    more destructive to the boundary detection signal.
    """

    neighborhood_sample_k: int = 25
    """
    Maximum neighbors sampled per node per SAGEConv layer.
    DOM graphs have variable fan-out:
        <body>: 5-20 children
        <div>:  1-50 children
        <li>:   0-3 children
    25 captures most neighborhoods completely while bounding computation
    for high-fan-out nodes (e.g., <ul> with 200 <li> children).
    """

    intent_alpha_init: float = 0.1
    """
    Initial scale for intent projection modulation.
    Small initial α prevents intent from overwhelming structural
    representations before training stabilizes the intent projection
    weights. α is learned — grows as training progresses.
    """

    # ── Loss Configuration ───────────────────────────────────────────────────

    contrastive_temperature: float = CONTRASTIVE_TEMPERATURE
    contrastive_negatives: int = CONTRASTIVE_NEGATIVES
    loss_weights: Tuple[float, float, float, float] = DEFAULT_LOSS_WEIGHTS
    class_weights: Tuple[float, float, float] = DEFAULT_CLASS_WEIGHTS
    focal_gamma: float = FOCAL_LOSS_GAMMA
    focal_enabled: bool = FOCAL_LOSS_ENABLED

    # ── Zone Coherence ───────────────────────────────────────────────────────

    coherence_threshold: float = COHERENCE_THRESHOLD
    coherence_min_zone_size: int = COHERENCE_MIN_ZONE_SIZE
    coherence_max_split_depth: int = COHERENCE_MAX_SPLIT_DEPTH

    # ── Cache ────────────────────────────────────────────────────────────────

    l1_max_size: int = L1_CACHE_MAX_SIZE

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for checkpoint storage."""
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "WLPConfig":
        """Reconstruct from checkpoint dict. Tolerates missing keys."""
        valid_fields = {f.name for f in dataclasses.fields(cls)}
        filtered = {k: v for k, v in d.items() if k in valid_fields}
        return cls(**filtered)


@dataclass
class ForwardOutput:
    """
    Complete output from LatentParser.forward() during training.

    Contains all four head outputs plus loss computation results.
    preparse_daemon.py uses loss for backward() and loss_components
    for training metrics logging.

    All tensor shapes are documented and contractual:
        logits:          (n_nodes, 3) — raw logits, not softmax
        boundary_logits: (n_nodes, 3) — boundary type logits
        confidences:     (n_nodes, 1) — sigmoid-bounded [0, 1]
        prototypes:      (n_nodes, 64) — L2-normalized on S^63
        loss:            scalar — None if node_labels not provided
        loss_components: per-task loss values for logging
    """

    logits: Tensor
    boundary_logits: Tensor
    confidences: Tensor
    prototypes: Tensor
    hidden_states: Tensor
    loss: Optional[Tensor]
    loss_components: Dict[str, float]


@dataclass
class ReadoutOutput:
    """
    Inference output from LatentParser.readout().

    Subset of ForwardOutput — no loss, no boundary logits (computed
    only for BOUNDARY-classified nodes after readout).

    Tensor shapes:
        logits:      (n_nodes, 3) — for wlp_zones.classify_nodes()
        confidences: (n_nodes, 1) — for wlp_zones.classify_nodes()
        prototypes:  (n_nodes, 64) — for zone coherence validation
    """

    logits: Tensor
    confidences: Tensor
    prototypes: Tensor


@dataclass
class CacheStatistics:
    """
    Exponential Moving Average statistics for cache tier performance.

    EMA update rule:
        ema_new = α · observation + (1 - α) · ema_old
    where α = 2 / (window + 1).

    Properties of EMA:
        - Recent observations have higher weight than old ones.
        - Half-life ≈ window / 2 observations.
        - Converges to true mean for stationary processes.
        - Responds to distribution shifts in O(window) observations.
    """

    l1_hits: int = 0
    l1_misses: int = 0
    l1_evictions: int = 0
    l1_hit_rate_ema: float = 0.0
    l1_ema_alpha: float = 2.0 / (HEALTH_L1_HIT_RATE_WINDOW + 1)

    l2_hits: int = 0
    l2_misses: int = 0
    l2_hit_rate_ema: float = 0.0
    l2_ema_alpha: float = 2.0 / (HEALTH_L2_HIT_RATE_WINDOW + 1)

    l3_parses: int = 0
    l3_failures: int = 0
    l3_avg_latency_ms: float = 0.0
    l3_latency_ema_alpha: float = 0.01

    total_queries: int = 0
    empty_returns: int = 0

    def record_l1_hit(self) -> None:
        """Record an L1 cache hit and update EMA."""
        self.l1_hits += 1
        self.total_queries += 1
        self.l1_hit_rate_ema = (
                self.l1_ema_alpha * 1.0
                + (1.0 - self.l1_ema_alpha) * self.l1_hit_rate_ema
        )

    def record_l1_miss(self) -> None:
        """Record an L1 cache miss and update EMA."""
        self.l1_misses += 1
        self.l1_hit_rate_ema = (
                self.l1_ema_alpha * 0.0
                + (1.0 - self.l1_ema_alpha) * self.l1_hit_rate_ema
        )

    def record_l1_eviction(self) -> None:
        """Record an L1 LRU eviction."""
        self.l1_evictions += 1

    def record_l2_hit(self) -> None:
        """Record an L2 cache hit and update EMA."""
        self.l2_hits += 1
        self.total_queries += 1
        self.l2_hit_rate_ema = (
                self.l2_ema_alpha * 1.0
                + (1.0 - self.l2_ema_alpha) * self.l2_hit_rate_ema
        )

    def record_l2_miss(self) -> None:
        """Record an L2 cache miss and update EMA."""
        self.l2_misses += 1
        self.l2_hit_rate_ema = (
                self.l2_ema_alpha * 0.0
                + (1.0 - self.l2_ema_alpha) * self.l2_hit_rate_ema
        )

    def record_l3_parse(self, latency_ms: float, success: bool) -> None:
        """Record an L3 fresh parse and update latency EMA."""
        self.l3_parses += 1
        self.total_queries += 1
        if not success:
            self.l3_failures += 1
        self.l3_avg_latency_ms = (
                self.l3_latency_ema_alpha * latency_ms
                + (1.0 - self.l3_latency_ema_alpha) * self.l3_avg_latency_ms
        )

    def record_empty_return(self) -> None:
        """Record a query that returned EmptyZoneMap."""
        self.empty_returns += 1

    def snapshot(self) -> Dict[str, Any]:
        """Return a serializable snapshot of all statistics."""
        return {
            "l1_hits": self.l1_hits,
            "l1_misses": self.l1_misses,
            "l1_evictions": self.l1_evictions,
            "l1_hit_rate_ema": round(self.l1_hit_rate_ema, 4),
            "l2_hits": self.l2_hits,
            "l2_misses": self.l2_misses,
            "l2_hit_rate_ema": round(self.l2_hit_rate_ema, 4),
            "l3_parses": self.l3_parses,
            "l3_failures": self.l3_failures,
            "l3_avg_latency_ms": round(self.l3_avg_latency_ms, 2),
            "total_queries": self.total_queries,
            "empty_returns": self.empty_returns,
            "empty_rate": round(
                self.empty_returns / max(self.total_queries, 1), 4
            ),
        }


@dataclass
class ZoneConfirmationTracker:
    """
    Tracks Jaccard-confirmed zone map consistency per (domain, topology_class).

    After DISCOVERY_CONFIRMATION_THRESHOLD confirmations with Jaccard > 0.85,
    the zone map confidence is incrementally boosted and a subclass candidacy
    event may be emitted.

    The tracker implements a simple Bayesian-flavored update:
        posterior_confidence = prior_confidence + Σ(boosts) subject to ceiling

    Each confirmation contributes CONFIDENCE_BOOST_PER_CONFIRMATION to the
    posterior. The ceiling at 0.95 prevents overconfidence.
    """

    confirmations: int = 0
    last_jaccard: float = 0.0
    cumulative_jaccard: float = 0.0
    first_seen: float = 0.0
    last_confirmed: float = 0.0

    def confirm(self, jaccard: float) -> bool:
        """
        Record a Jaccard-confirmed zone map observation.
        Returns True if confirmation threshold is newly reached.
        """
        now = time.monotonic()
        if self.first_seen == 0.0:
            self.first_seen = now

        self.last_jaccard = jaccard
        self.cumulative_jaccard += jaccard
        self.last_confirmed = now
        self.confirmations += 1

        return self.confirmations == DISCOVERY_CONFIRMATION_THRESHOLD

    def mean_jaccard(self) -> float:
        """
        Mean Jaccard index across all confirmations.
        Returns 0.0 if no confirmations yet.
        """
        if self.confirmations == 0:
            return 0.0
        return self.cumulative_jaccard / self.confirmations

    def is_stable(self) -> bool:
        """
        Returns True if the zone pattern is statistically stable.
        Requires: ≥ threshold confirmations AND mean Jaccard > 0.85.
        """
        return (
                self.confirmations >= DISCOVERY_CONFIRMATION_THRESHOLD
                and self.mean_jaccard() > JACCARD_SIMILARITY_THRESHOLD
        )

    def reset(self) -> None:
        """Reset tracker on structural drift detection."""
        self.confirmations = 0
        self.last_jaccard = 0.0
        self.cumulative_jaccard = 0.0
        self.last_confirmed = 0.0


# ═════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL UTILITY FUNCTIONS
#
# Pure functions with no side effects. Used by LatentParser, WorldLatentParser,
# and the zone coherence validation pipeline. Each function documents its
# mathematical foundation, computational complexity, and numerical stability.
# ═════════════════════════════════════════════════════════════════════════════


def _log_sum_exp(tensor: Tensor, dim: int = -1) -> Tensor:
    """
    Numerically stable log-sum-exp computation.

    log(Σ_i exp(x_i)) = max(x) + log(Σ_i exp(x_i - max(x)))

    Standard exp-then-sum overflows when max(x) > ~88 (float32).
    Subtracting max(x) ensures exp arguments are ≤ 0, preventing overflow.
    The max is added back after log to recover the correct value.

    This is the foundation of numerically stable softmax and InfoNCE loss.

    Args:
        tensor: Input tensor.
        dim: Dimension along which to compute.

    Returns:
        Log-sum-exp values, reduced along dim.

    Complexity: O(n) where n = tensor.shape[dim].
    """
    max_val, _ = tensor.max(dim=dim, keepdim=True)
    return max_val.squeeze(dim) + torch.log(
        torch.exp(tensor - max_val).sum(dim=dim).clamp(min=LOG_SAFE_MIN)
    )


def _stable_softmax(logits: Tensor, dim: int = -1, temperature: float = 1.0) -> Tensor:
    """
    Numerically stable temperature-scaled softmax.

    softmax(x_i / τ) = exp(x_i / τ) / Σ_j exp(x_j / τ)

    Implemented as:
        shifted = (x - max(x)) / τ
        softmax = exp(shifted) / sum(exp(shifted))

    The shift by max(x) prevents overflow in exp() without changing
    the softmax output (translation invariance of softmax).

    Args:
        logits: Raw logits tensor.
        dim: Softmax dimension.
        temperature: Scaling factor τ. Lower = sharper distribution.

    Returns:
        Probability distribution tensor (sums to 1 along dim).

    Numerical properties:
        - No overflow: max argument to exp() is 0.
        - No underflow: at least one exp() evaluates to 1.
        - Gradients are well-conditioned for τ ∈ [0.01, 100].
    """
    scaled = logits / max(temperature, EPS)
    shifted = scaled - scaled.max(dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / exp_shifted.sum(dim=dim, keepdim=True).clamp(min=EPS)


def _cosine_similarity_matrix(a: Tensor, b: Tensor) -> Tensor:
    """
    Pairwise cosine similarity matrix between two sets of vectors.

    cos(a_i, b_j) = (a_i · b_j) / (||a_i|| · ||b_j||)

    For L2-normalized inputs (||a_i|| = ||b_j|| = 1):
        cos(a_i, b_j) = a_i · b_j

    This function handles both normalized and unnormalized inputs.

    Args:
        a: (N, D) tensor — first set of vectors.
        b: (M, D) tensor — second set of vectors.

    Returns:
        (N, M) similarity matrix where entry [i, j] = cos(a_i, b_j).

    Complexity: O(N × M × D) for the matrix multiplication.
    Memory: O(N × M) for the output matrix.
    """
    a_norm = F.normalize(a, p=2, dim=1)
    b_norm = F.normalize(b, p=2, dim=1)
    return torch.mm(a_norm, b_norm.t())


def _pca_first_component(data: Tensor) -> Tensor:
    """
    Extract the first principal component via truncated SVD.

    Given centered data matrix X ∈ R^{n×d}:
        X = U Σ V^T  (compact SVD)
    First principal component: v_1 = V[:, 0]  (first right singular vector)
    Projections: scores = X v_1 ∈ R^n

    The first principal component captures the direction of maximum
    variance in the data. For bimodal distributions (two clusters),
    this direction separates the two clusters.

    Used by zone coherence validation to find the optimal split axis
    when prototype variance exceeds COHERENCE_THRESHOLD.

    Args:
        data: (n, d) centered data matrix (mean-subtracted).

    Returns:
        (d,) first principal component vector, unit-length.

    Numerical stability:
        torch.linalg.svd with full_matrices=False computes only the
        min(n, d) singular vectors, which is more numerically stable
        and faster than computing the full eigendecomposition.

    Complexity: O(n × d × min(n, d)) for the SVD.
    For zone validation: n ≈ 10-1000 nodes, d = 64.
    """
    if data.shape[0] < 2:
        return torch.zeros(data.shape[1], device=data.device, dtype=data.dtype)

    try:
        U, S, Vh = torch.linalg.svd(data, full_matrices=False)
        return Vh[0]
    except _LinAlgError:
        log.warning(
            "pca_svd_failed",
            shape=list(data.shape),
            msg="SVD did not converge — returning zero vector",
        )
        return torch.zeros(data.shape[1], device=data.device, dtype=data.dtype)


def _find_maximum_gap(sorted_values: Tensor) -> int:
    """
    Find the index of the maximum consecutive gap in sorted values.

    Given sorted values [v_0, v_1, ..., v_{n-1}]:
        gaps[i] = v_{i+1} - v_i  for i in [0, n-2]
        split_point = argmax(gaps) + 1

    The maximum gap is the natural split point for a bimodal distribution
    projected onto a single dimension (the first principal component).

    Returns the split index: nodes at indices [0, split_point) form group A,
    nodes at indices [split_point, n) form group B.

    Args:
        sorted_values: (n,) tensor of sorted projection values.

    Returns:
        Split index in [1, n-1]. Returns n//2 if all gaps are equal.

    Complexity: O(n).
    """
    if sorted_values.shape[0] < 2:
        return 1

    gaps = sorted_values[1:] - sorted_values[:-1]

    if gaps.max().item() < EPS:
        return sorted_values.shape[0] // 2

    return gaps.argmax().item() + 1


def _jaccard_similarity(set_a: Set[str], set_b: Set[str]) -> float:
    """
    Jaccard similarity index between two sets.

    J(A, B) = |A ∩ B| / |A ∪ B|

    Properties:
        J(A, B) ∈ [0, 1]
        J(A, A) = 1
        J(A, ∅) = 0  (by convention; handled explicitly)
        J(A, B) = J(B, A)  (symmetric)

    The Jaccard index is a metric on the power set:
        d(A, B) = 1 - J(A, B)
    satisfies the triangle inequality.

    Used to compare CSS selector sets of zone maps:
        set_a = {z.selector for z in zone_map_a.signal_zones}
        set_b = {z.selector for z in zone_map_b.signal_zones}
        J > 0.85 → zones are essentially the same structure.

    Args:
        set_a: First set of strings.
        set_b: Second set of strings.

    Returns:
        Jaccard index in [0.0, 1.0].

    Complexity: O(|A| + |B|) average for set operations.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0

    intersection_size = len(set_a & set_b)
    union_size = len(set_a | set_b)

    if union_size == 0:
        return 1.0

    return intersection_size / union_size


def _prototype_variance(prototypes: Tensor) -> float:
    """
    Compute the mean squared distance of prototypes from their centroid.

    For prototype vectors {z_1, ..., z_k} on the unit hypersphere S^{d-1}:
        centroid = mean(z_i)
        variance = mean(||z_i - centroid||²)

    This is the within-cluster sum of squares (WCSS) normalized by cluster size,
    also known as the trace of the within-cluster scatter matrix divided by n:
        Var = (1/n) · Σ_i ||z_i - μ||²  = (1/n) · tr(S_W)
    where S_W = Σ_i (z_i - μ)(z_i - μ)^T.

    For perfectly coherent zones: all z_i ≈ μ → Var ≈ 0.
    For bimodal zones: two clusters → Var ≈ ||μ_1 - μ_2||² / 4.
    For uniformly random on S^63: E[Var] ≈ 1.0.

    COHERENCE_THRESHOLD = 0.15 catches the bimodal case.

    Args:
        prototypes: (k, d) tensor of prototype vectors.

    Returns:
        Scalar variance value.

    Complexity: O(k × d).
    """
    if prototypes.shape[0] < 2:
        return 0.0

    centroid = prototypes.mean(dim=0, keepdim=True)
    deviations = prototypes - centroid
    variance = (deviations ** 2).mean().item()
    return variance


def _exponential_decay(
        confidence: float,
        surprise_score: float,
        decay_rate: float = SURPRISE_DECAY_RATE,
        floor: float = CONFIDENCE_FLOOR,
) -> float:
    """
    Apply exponential surprise-based confidence decay.

    new_confidence = max(confidence × (1 - decay_rate × surprise_score), floor)

    The decay is linear in surprise_score because surprise_score is
    already calibrated to be approximately proportional to structural
    divergence measured in bits (KL divergence of observed vs expected
    extraction distributions).

    For surprise_score = 0: no decay (no structural change detected).
    For surprise_score = 1: 5% decay (minor structural deviation).
    For surprise_score = 10: 50% decay (major structural shift).
    For surprise_score = 20: 100% decay → floor.

    The floor at 0.30 prevents useful ZoneMaps from being invalidated
    by partial surprise events. Full invalidation requires dissolve_triggered
    in the SurpriseEvent, which is handled separately.

    Args:
        confidence: Current zone map confidence in [0, 1].
        surprise_score: Measured surprise in [0, ∞).
        decay_rate: Per-unit decay rate.
        floor: Minimum confidence after decay.

    Returns:
        Decayed confidence in [floor, confidence].
    """
    decay_factor = max(0.0, 1.0 - decay_rate * surprise_score)
    return max(confidence * decay_factor, floor)


def _weighted_confidence_merge(
        heuristic_confidence: float,
        model_confidence: float,
        agreement: bool,
) -> float:
    """
    Merge heuristic and model confidence scores for discover_signal_zones().

    Three-tier confidence computation:
        Both agree:   (heuristic + model) / 2 + agreement_boost
        Heuristic only: heuristic (capped)
        Model only:   model × discount

    Mathematical justification:
        Agreement between independent estimators reduces uncertainty.
        If heuristic and model are conditionally independent given true zone,
        the posterior probability under Bayesian combination is:
            P(signal | heuristic ∧ model) ∝ P(heuristic | signal) × P(model | signal) × P(signal)
        The additive boost (+0.10) is a log-space approximation of this product
        that avoids requiring full probability distributions.

    Args:
        heuristic_confidence: Confidence from structural heuristics.
        model_confidence: Confidence from GraphSAGE model.
        agreement: Whether both estimators agree on SIGNAL classification.

    Returns:
        Merged confidence, capped at DISCOVERY_CONFIDENCE_CEILING.
    """
    if agreement:
        merged = (heuristic_confidence + model_confidence) / 2.0 + 0.10
    elif model_confidence == 0.0:
        merged = heuristic_confidence
    elif heuristic_confidence == 0.0:
        merged = model_confidence * 0.80
    else:
        merged = max(heuristic_confidence, model_confidence * 0.80)

    return min(merged, DISCOVERY_CONFIDENCE_CEILING)


def _compute_selector_set(zone_map: Union[ZoneMap, EmptyZoneMap]) -> Set[str]:
    """
    Extract the set of CSS selectors from a zone map's signal zones.

    Used for Jaccard similarity computation between zone maps.
    EmptyZoneMap returns an empty set.

    Args:
        zone_map: A ZoneMap or EmptyZoneMap.

    Returns:
        Set of selector strings.
    """
    if isinstance(zone_map, EmptyZoneMap):
        return set()
    return {z.selector for z in zone_map.signal_zones}


def _sha256_file(path: Path) -> str:
    """
    Compute SHA-256 digest of a file.

    Reads in 64KB chunks for memory efficiency on large .pt files.
    Used to verify staging file integrity before atomic rename.

    Args:
        path: File path to hash.

    Returns:
        Lowercase hex digest string (64 characters).
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _infer_content_majority(zone_map: Union[ZoneMap, EmptyZoneMap]) -> Optional[str]:
    """
    Determine the majority content type across signal zones.

    Used by _infer_subclass() to suggest topology class promotion.
    Counts the content_type field of each signal zone and returns
    the majority type if it exceeds 50% of zones.

    Args:
        zone_map: Zone map to analyze.

    Returns:
        Majority content type string, or None if no clear majority.
    """
    if isinstance(zone_map, EmptyZoneMap) or not zone_map.signal_zones:
        return None

    counts: Dict[str, int] = defaultdict(int)
    for zone in zone_map.signal_zones:
        counts[zone.content_type] += 1

    total = len(zone_map.signal_zones)
    for content_type, count in counts.items():
        if count > total / 2:
            return content_type
    return None


# ═════════════════════════════════════════════════════════════════════════════
# INTENT PROJECTION MODULE
# ═════════════════════════════════════════════════════════════════════════════


class IntentProjection(nn.Module):
    """
    Projects intent_vector into the node representation space.
    Applied AFTER Layer 1 SAGEConv and BEFORE Layer 2.

    This allows intent to modulate the 2nd and 3rd hop aggregations.
    After Layer 1, each node has a 256-dim representation that captures
    local structure. Intent projection injects a learned bias that shifts
    representations toward intent-relevant structural patterns.

    Mathematical formulation:
        Let h_v^(1) ∈ R^256 be the Layer 1 output for node v.
        Let z = W_intent · intent_vector ∈ R^256 where W_intent ∈ R^{256×256}.
        Let α ∈ R be a learned scalar (initialized to 0.1).

        Modified Layer 1 output:
            h_v^(1)' = h_v^(1) + α · z

        This broadcast-adds the same intent bias to ALL nodes.
        α controls the strength of intent modulation.

    Why between Layer 1 and Layer 2:
        Layer 2 aggregates modified representations via GraphSAGE:
            h_v^(2) = σ(W^(2) · CONCAT(h_v^(1)', MEAN(h_u^(1)' : u ∈ N(v))))

        Intent propagates through the neighborhood because:
            Node v's Layer 2 representation depends on neighbors' Layer 1
            representations, which have been intent-modified.
            Intent flows through the graph structure via message passing.

    Why not before Layer 1:
        Layer 1 operates on raw 128-dim node features.
        Intent_vector is 256-dim. Dimension mismatch.
        More importantly: Layer 1 should learn purely structural features
        before intent modulation shifts the representation space.

    Why not after Layer 3:
        Post-hoc intent weighting (in wlp_zones.apply_intent_weights())
        already handles zone-level intent conditioning after model output.
        In-model conditioning captures neighborhood-level effects.
        Post-hoc captures zone-level semantic matching.
        Both are needed. Placing intent after Layer 3 would make
        in-model conditioning redundant with post-hoc conditioning.

    When intent_vector is None:
        Returns h1 unchanged. No computation. α is not applied.
    """

    def __init__(self, intent_dim: int, hidden_dim: int, alpha_init: float) -> None:
        super().__init__()
        self.projection = nn.Linear(intent_dim, hidden_dim, bias=True)
        self.alpha = nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))

    def forward(
            self,
            h1: Tensor,
            intent_vector: Optional[Tensor],
    ) -> Tensor:
        """
        Apply intent projection to Layer 1 representations.

        Args:
            h1: (n_nodes, hidden_dim) Layer 1 output.
            intent_vector: (intent_dim,) or None.

        Returns:
            (n_nodes, hidden_dim) intent-modified representations.
        """
        if intent_vector is None:
            return h1

        intent_bias = self.projection(intent_vector)
        return h1 + self.alpha * intent_bias.unsqueeze(0)


# ═════════════════════════════════════════════════════════════════════════════
# LATENT PARSER — THE GRAPHSAGE MODEL
# ═════════════════════════════════════════════════════════════════════════════


class LatentParser(nn.Module):
    """
    GraphSAGE node classification model for structural zone detection.

    Architecture:
        3 × SAGEConv layers with residual connections and LayerNorm.
        IntentProjection between Layer 1 and Layer 2.
        4 output heads: classification, boundary_type, confidence, prototype.

    Mathematical specification:

        Layer 1 — Local structure aggregation:
            h^(1) = LayerNorm(ReLU(SAGEConv_1(x, edge_index)))
            Input: (n_nodes, 128)  Output: (n_nodes, 256)
            No residual — dimension change (128 → 256).

        Intent injection (between Layer 1 and Layer 2):
            h^(1)' = h^(1) + α · W_intent · intent_vector

        Layer 2 — Section structure aggregation (with residual):
            h^(2)_raw = SAGEConv_2(h^(1)', edge_index)
            h^(2) = LayerNorm(ReLU(h^(2)_raw) + h^(1)')
            Residual: dimension match (256 → 256).
            Preserves local features through deeper aggregation.

        Layer 3 — Page structure aggregation (with residual):
            h^(3)_raw = SAGEConv_3(h^(2), edge_index)
            h^(3) = LayerNorm(ReLU(h^(3)_raw) + h^(2))
            No dropout after final layer.

        Head 1 — Node Classification (primary):
            Linear(256, 128) → ReLU → Linear(128, 3)
            Two-layer MLP for non-linear decision boundary.

        Head 2 — Boundary Type (auxiliary):
            Linear(256, 64) → ReLU → Linear(64, 3)
            Lighter than Head 1 — boundary type is simpler.

        Head 3 — Per-Node Confidence:
            Linear(256, 32) → ReLU → Linear(32, 1) → Sigmoid
            Bounded [0, 1]. Learned uncertainty signal.

        Head 4 — Zone Prototype (novel):
            Linear(256, 64) — no activation, L2-normalized at use time.
            Projects onto the unit hypersphere S^63.

    Two modes:
        Training mode (forward()): Full output with gradient flow.
        Inference mode (readout()): @torch.no_grad(), eval mode.

    Weight initialization:
        Kaiming (He) for ReLU layers — variance 2/fan_in.
        Xavier (Glorot) for prototype head — symmetric variance.
        PyG defaults for SAGEConv — do not override.

    Parameter count:
        SAGEConv layers: 3 × (256 × 256 + 256 × 256) ≈ 393K
        Intent projection: 256 × 256 + 256 + 1 ≈ 66K
        Head 1: 256 × 128 + 128 + 128 × 3 + 3 ≈ 33K
        Head 2: 256 × 64 + 64 + 64 × 3 + 3 ≈ 17K
        Head 3: 256 × 32 + 32 + 32 × 1 + 1 ≈ 8K
        Head 4: 256 × 64 + 64 ≈ 16K
        Total: ~533K parameters (~2.1MB at float32)
    """

    def __init__(self, config: WLPConfig) -> None:
        super().__init__()
        self.config = config

        # ── SAGEConv layers ──────────────────────────────────────────────────
        # Mean aggregation: h' = W_1 · x + W_2 · MEAN(x[N(v)])
        # Inductive: new nodes classified by aggregating from actual neighbors.
        # No adjacency matrix required at inference.

        self.conv1 = SAGEConv(
            config.node_feature_dim,
            config.hidden_dim,
            aggr="mean",
        )
        self.conv2 = SAGEConv(
            config.hidden_dim,
            config.hidden_dim,
            aggr="mean",
        )
        self.conv3 = SAGEConv(
            config.hidden_dim,
            config.hidden_dim,
            aggr="mean",
        )

        # ── Layer Normalization ──────────────────────────────────────────────
        # LayerNorm normalizes across the feature dimension per-node.
        # Each node normalized independently — node count does not affect
        # normalization statistics. Correct for variable-size graph batches.

        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.norm3 = nn.LayerNorm(config.hidden_dim)

        # ── Dropout ──────────────────────────────────────────────────────────
        self.dropout = nn.Dropout(config.dropout)

        # ── Intent Projection ────────────────────────────────────────────────
        self.intent_proj = IntentProjection(
            intent_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            alpha_init=config.intent_alpha_init,
        )

        # ── Head 1: Node Classification (2-layer MLP) ────────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, config.num_node_classes),
        )

        # ── Head 2: Boundary Type (lighter MLP) ─────────────────────────────
        self.bnd_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, config.num_boundary_types),
        )

        # ── Head 3: Per-Node Confidence ──────────────────────────────────────
        self.conf_head = nn.Sequential(
            nn.Linear(config.hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        # ── Head 4: Zone Prototype Embedding ─────────────────────────────────
        # Linear, no activation. L2-normalized at use time.
        self.proto_head = nn.Linear(config.hidden_dim, config.prototype_dim)

        # ── Weight Initialization ────────────────────────────────────────────
        self._initialize_weights()

        # ── Loss Configuration (cached tensors) ──────────────────────────────
        self._class_weights_tensor: Optional[Tensor] = None
        self._training_step: int = 0

    def _initialize_weights(self) -> None:
        """
        Apply principled weight initialization to all learnable parameters.

        Kaiming (He et al. 2015) initialization for ReLU-activated layers:
            W ~ N(0, sqrt(2 / fan_in))
            Variance: Var(W) = 2 / fan_in
            Prevents vanishing gradients through deep ReLU stacks.
            Derivation: for ReLU activation, E[a²] = 0.5 × Var(input)
            Maintaining variance requires Var(W) = 2 / fan_in.

        Xavier (Glorot & Bengio 2010) for the prototype head:
            W ~ N(0, sqrt(2 / (fan_in + fan_out)))
            Variance: Var(W) = 2 / (fan_in + fan_out)
            For layers without activation (or with linear activation),
            Xavier preserves variance in both forward and backward passes.
            The prototype head has no activation — Xavier is correct.

        SAGEConv layers: PyG's default initialization is used.
            PyG initializes SAGEConv.lin_l and SAGEConv.lin_r with
            their own Glorot-uniform scheme. Do not override — PyG's
            defaults are calibrated for the specific weight matrix
            structure of mean-aggregation SAGEConv.

        LayerNorm: PyTorch default (weight=1, bias=0) is correct.
            LayerNorm's learned affine parameters should start at identity
            to preserve the raw normalization during early training.

        Intent projection alpha: initialized to config.intent_alpha_init (0.1).
            Already set by IntentProjection constructor.
        """
        # Classification head — Kaiming for ReLU
        for module in [self.cls_head, self.bnd_head, self.conf_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

        # Prototype head — Xavier for linear output
        nn.init.xavier_normal_(self.proto_head.weight)
        if self.proto_head.bias is not None:
            nn.init.zeros_(self.proto_head.bias)

        # Intent projection — Kaiming for ReLU in downstream usage
        nn.init.kaiming_normal_(
            self.intent_proj.projection.weight, nonlinearity="relu"
        )
        if self.intent_proj.projection.bias is not None:
            nn.init.zeros_(self.intent_proj.projection.bias)

    def _get_class_weights(self, device: torch.device) -> Tensor:
        """
        Lazily create and cache class weight tensor on the correct device.
        Avoids repeated tensor creation on every forward pass.
        """
        if (
                self._class_weights_tensor is None
                or self._class_weights_tensor.device != device
        ):
            self._class_weights_tensor = torch.tensor(
                self.config.class_weights,
                dtype=torch.float32,
                device=device,
            )
        return self._class_weights_tensor

    def _encode_graph(
            self,
            x: Tensor,
            edge_index: Tensor,
            intent_vector: Optional[Tensor],
    ) -> Tensor:
        """
        Shared encoder: 3-layer SAGEConv with residuals.
        Used by both forward() and readout() to avoid code duplication.

        Returns the final hidden state h^(3).

        Layer 1: SAGEConv_1(x, edge_index)
            Input: (n, 128)  →  Output: (n, 256)
            No residual: dimension change.
            Apply: LayerNorm → ReLU → Dropout (training) → IntentProjection

        Layer 2: SAGEConv_2(h1, edge_index) + h1  (residual)
            Input: (n, 256)  →  Output: (n, 256)
            Residual: dimensions match.
            Apply: ReLU → add residual → LayerNorm → Dropout (training)

        Layer 3: SAGEConv_3(h2, edge_index) + h2  (residual)
            Input: (n, 256)  →  Output: (n, 256)
            Residual: dimensions match.
            Apply: ReLU → add residual → LayerNorm
            No dropout after final layer — preserve full representation.
        """
        # ── Layer 1 ──────────────────────────────────────────────────────────
        h1 = self.conv1(x, edge_index)
        h1 = self.norm1(F.relu(h1))
        h1 = self.dropout(h1)

        # ── Intent Projection (between Layer 1 and Layer 2) ──────────────────
        h1 = self.intent_proj(h1, intent_vector)

        # ── Layer 2 with Residual ────────────────────────────────────────────
        h2_raw = self.conv2(h1, edge_index)
        h2 = self.norm2(F.relu(h2_raw) + h1)
        h2 = self.dropout(h2)

        # ── Layer 3 with Residual ────────────────────────────────────────────
        h3_raw = self.conv3(h2, edge_index)
        h3 = self.norm3(F.relu(h3_raw) + h2)
        # No dropout after final layer

        return h3

    def forward(
            self,
            data: Data,
            intent_vector: Optional[Tensor] = None,
            node_labels: Optional[Tensor] = None,
            zone_labels: Optional[Tensor] = None,
    ) -> ForwardOutput:
        """
        Full forward pass with gradient flow.

        Called ONLY by preparse_daemon.py via WLPTrainingInterface.
        Never called from the query path. Never from bus handlers.

        The forward pass consists of:
            1. Graph encoding (3 SAGEConv layers + intent projection)
            2. Four output heads
            3. Loss computation (if labels provided)

        Args:
            data:          PyG Data object from wlp_graph.cst_to_pyg_graph()
            intent_vector: (256,) tensor or None
            node_labels:   (n_nodes,) int64 ground truth labels
            zone_labels:   (n_nodes,) int64 zone membership IDs

        Returns:
            ForwardOutput with all head outputs and optional loss.
        """
        x, edge_index = data.x, data.edge_index

        # ── Graph Encoding ───────────────────────────────────────────────────
        h3 = self._encode_graph(x, edge_index, intent_vector)

        # ── Output Heads ─────────────────────────────────────────────────────
        logits = self.cls_head(h3)
        bnd_logits = self.bnd_head(h3)
        confidences = self.conf_head(h3)
        protos_raw = self.proto_head(h3)
        prototypes = F.normalize(protos_raw, p=2, dim=1)

        # ── Loss Computation (training only) ─────────────────────────────────
        loss = None
        loss_components: Dict[str, float] = {}
        if node_labels is not None:
            loss, loss_components = self._compute_loss(
                logits=logits,
                bnd_logits=bnd_logits,
                confidences=confidences,
                prototypes=prototypes,
                node_labels=node_labels,
                zone_labels=zone_labels,
            )
            self._training_step += 1

        return ForwardOutput(
            logits=logits,
            boundary_logits=bnd_logits,
            confidences=confidences,
            prototypes=prototypes,
            hidden_states=h3,
            loss=loss,
            loss_components=loss_components,
        )

    @torch.no_grad()
    def readout(
            self,
            data: Data,
            intent_vector: Optional[Tensor] = None,
    ) -> ReadoutOutput:
        """
        Inference forward pass. No gradient flow. Frozen weights.

        Model must be in eval() mode before calling.
        WorldLatentParser ensures this via self._model.eval() in initialize().

        @torch.no_grad() decorator ensures no gradient tape accumulation.
        Combined with model.eval(): dropout disabled, LayerNorm uses
        running statistics instead of batch statistics.

        Returns:
            ReadoutOutput with logits, confidences, and prototypes.
        """
        x, edge_index = data.x, data.edge_index

        # ── Graph Encoding ───────────────────────────────────────────────────
        h3 = self._encode_graph(x, edge_index, intent_vector)

        # ── Output Heads (inference subset) ──────────────────────────────────
        logits = self.cls_head(h3)
        confidences = self.conf_head(h3)
        prototypes = F.normalize(self.proto_head(h3), p=2, dim=1)

        return ReadoutOutput(
            logits=logits,
            confidences=confidences,
            prototypes=prototypes,
        )

    def _compute_loss(
            self,
            logits: Tensor,
            bnd_logits: Tensor,
            confidences: Tensor,
            prototypes: Tensor,
            node_labels: Tensor,
            zone_labels: Optional[Tensor],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        Multi-task loss computation with four components.

        L_total = λ_1 · L_cls + λ_2 · L_bnd + λ_3 · L_conf + λ_4 · L_contrast

        Each component is computed with careful attention to:
            - Numerical stability (log-sum-exp, epsilon guards)
            - Class imbalance (focal loss, class weights)
            - Sparse labels (boundary masking)
            - Contrastive sampling (seeded negative sampling)

        Returns:
            (total_loss, component_dict) for optimizer and logging.
        """
        device = logits.device
        lw = self.config.loss_weights
        n_nodes = logits.shape[0] # noqa

        # ═════════════════════════════════════════════════════════════════════
        # L_classification — Focal Cross-Entropy on All Nodes
        # ═════════════════════════════════════════════════════════════════════
        #
        # Standard cross-entropy:
        #     CE(p, y) = -log(p_y)  where p = softmax(logits)
        #
        # Focal modulation (Lin et al. 2017):
        #     FL(p_t) = -(1 - p_t)^γ · log(p_t)
        #     where p_t = p[y] is the predicted probability for the true class.
        #
        # Combined with class weights:
        #     L_cls = (1/N) · Σ_i w_{y_i} · (1 - p_{t_i})^γ · (-log(p_{t_i}))
        #
        # Implementation via LogSoftmax for numerical stability:
        #     log(p) = LogSoftmax(logits)
        #     p_t = exp(log(p)[y])
        #     focal_weight = (1 - p_t)^γ

        class_weights = self._get_class_weights(device)

        log_probs = F.log_softmax(logits, dim=1)
        nll = F.nll_loss(log_probs, node_labels, weight=class_weights, reduction="none")

        if self.config.focal_enabled and self._training_step >= TRAINING_WARMUP_STEPS:
            probs = torch.exp(log_probs)
            p_t = probs.gather(1, node_labels.unsqueeze(1)).squeeze(1)
            focal_weight = (1.0 - p_t).pow(self.config.focal_gamma)
            loss_cls = (focal_weight * nll).mean()
        else:
            loss_cls = nll.mean()

        # ═════════════════════════════════════════════════════════════════════
        # L_boundary — Masked Cross-Entropy on BOUNDARY Nodes Only
        # ═════════════════════════════════════════════════════════════════════
        #
        # Only computed on ground-truth BOUNDARY nodes.
        # Masked loss: zero out non-boundary nodes before summing.
        #
        # Boundary nodes are identified by:
        #     boundary_mask[i] = 1 if node_labels[i] == NODE_BOUNDARY else 0
        #
        # The loss is:
        #     L_bnd = (1/n_boundary) · Σ_{i ∈ boundary} CE(bnd_logits[i], bnd_label[i])
        #
        # If no boundary nodes exist in this batch: L_bnd = 0.
        # This prevents NaN from division by zero.

        boundary_mask = (node_labels == NODE_BOUNDARY).float()
        n_boundary = boundary_mask.sum().item()

        if n_boundary > 0:
            bnd_log_probs = F.log_softmax(bnd_logits, dim=1)
            bnd_nll = F.nll_loss(bnd_log_probs, node_labels, reduction="none")
            loss_bnd = (boundary_mask * bnd_nll).sum() / max(n_boundary, 1.0)
        else:
            loss_bnd = torch.tensor(0.0, device=device, requires_grad=True)

        # ═════════════════════════════════════════════════════════════════════
        # L_confidence — MSE Between Predicted Confidence and Correctness
        # ═════════════════════════════════════════════════════════════════════
        #
        # The confidence head should output:
        #     high confidence (→ 1.0) when classification is correct
        #     low confidence (→ 0.0) when classification is wrong
        #
        # Target construction:
        #     predicted_labels = argmax(logits, dim=1)
        #     correct[i] = 1.0 if predicted_labels[i] == node_labels[i] else 0.0
        #
        # Loss:
        #     L_conf = (1/N) · Σ_i (confidence[i] - correct[i])²
        #
        # This trains the confidence head as a learned uncertainty signal.
        # It is NOT a calibrated probability — it is a correlation signal.
        # High confidence should CORRELATE with correct classification
        # (Pearson r > 0.6 at convergence).
        #
        # Disabled during warmup: the model's predictions are random,
        # so correctness is random, and the confidence head learns noise.

        if self._training_step >= TRAINING_WARMUP_STEPS:
            with torch.no_grad():
                predicted = logits.argmax(dim=1)
                correct = (predicted == node_labels).float().unsqueeze(1)
            loss_conf = F.mse_loss(confidences, correct)
        else:
            loss_conf = torch.tensor(0.0, device=device, requires_grad=True)

        # ═════════════════════════════════════════════════════════════════════
        # L_contrastive — InfoNCE on Zone Prototype Embeddings
        # ═════════════════════════════════════════════════════════════════════
        #
        # SimCLR-style formulation (Chen et al. 2020):
        #     For anchor node i with zone label z_i:
        #         Positive: node j with z_j == z_i (same zone)
        #         Negatives: K nodes with z_k != z_i (different zones)
        #
        #     L_contrast(i) = -log(
        #         exp(sim(z_i, z_j+) / τ) /
        #         (exp(sim(z_i, z_j+) / τ) + Σ_k exp(sim(z_i, z_k-) / τ))
        #     )
        #
        # Where:
        #     z_i = L2-normalized prototype embedding for node i
        #     sim(a, b) = a · b (cosine similarity on unit sphere)
        #     τ = temperature = 0.07
        #     K = config.contrastive_negatives = 5
        #
        # Sampling strategy:
        #     For each anchor: randomly sample 1 positive and K negatives.
        #     Seeded per-batch for reproducibility: seed = training_step.
        #     If no same-zone positive exists: skip this anchor.
        #     If no different-zone negative exists: skip this anchor.
        #
        # Numerical stability:
        #     The log-softmax form is computed via the log-sum-exp trick:
        #         L(i) = -sim(z_i, z_j+)/τ + log_sum_exp([sim(z_i, z_k)/τ for k])
        #
        # L2-normalization projects all prototypes onto S^{d-1}.
        # Cosine similarity on the unit sphere = dot product.
        # The loss pushes same-zone nodes closer on the sphere
        # and different-zone nodes apart.

        if zone_labels is not None and self._training_step >= TRAINING_WARMUP_STEPS:
            loss_contrast = self._contrastive_loss(
                prototypes, zone_labels, device
            )
        else:
            loss_contrast = torch.tensor(0.0, device=device, requires_grad=True)

        # ═════════════════════════════════════════════════════════════════════
        # Total Loss — Weighted Combination
        # ═════════════════════════════════════════════════════════════════════

        total_loss = (
                lw[0] * loss_cls
                + lw[1] * loss_bnd
                + lw[2] * loss_conf
                + lw[3] * loss_contrast
        )

        components = {
            "loss_classification": loss_cls.item(),
            "loss_boundary": loss_bnd.item(),
            "loss_confidence": loss_conf.item(),
            "loss_contrastive": loss_contrast.item(),
            "loss_total": total_loss.item(),
            "n_boundary_nodes": n_boundary,
            "training_step": self._training_step,
        }

        return total_loss, components

    def _contrastive_loss(
            self,
            prototypes: Tensor,
            zone_labels: Tensor,
            device: torch.device,
    ) -> Tensor:
        """
        InfoNCE contrastive loss on zone prototype embeddings.

        For each anchor node:
            1. Find one same-zone node (positive).
            2. Sample K different-zone nodes (negatives).
            3. Compute InfoNCE loss for this triplet+.

        The loss encourages same-zone nodes to have similar prototype
        embeddings and different-zone nodes to have dissimilar embeddings.

        Sampling is seeded per-batch for reproducibility:
            seed = self._training_step
            torch.randperm with generator ensures deterministic sampling.

        Computational complexity:
            O(N × K × d) where N = sampled anchors, K = negatives, d = proto_dim.
            N is capped at min(n_nodes, 512) for memory efficiency.
            At K=5, d=64, N=512: 163,840 operations. Fast on GPU.

        Numerical stability:
            Cosine similarities are bounded [-1, 1].
            Divided by τ = 0.07: range [-14.3, 14.3].
            Log-sum-exp handles the partition function stably.

        Returns:
            Scalar loss tensor with gradient flow.
        """
        n_nodes = prototypes.shape[0]
        tau = self.config.contrastive_temperature
        K = self.config.contrastive_negatives

        if n_nodes < 4:
            return torch.tensor(0.0, device=device, requires_grad=True)

        unique_zones = torch.unique(zone_labels)
        if unique_zones.shape[0] < 2:
            return torch.tensor(0.0, device=device, requires_grad=True)

        generator = torch.Generator(device="cpu")
        generator.manual_seed(self._training_step)

        zone_to_indices: Dict[int, List[int]] = defaultdict(list)
        for i in range(n_nodes):
            zone_to_indices[zone_labels[i].item()].append(i)

        max_anchors = min(n_nodes, 512)
        perm = torch.randperm(n_nodes, generator=generator)[:max_anchors]

        losses: List[Tensor] = []

        for anchor_idx_t in perm:
            anchor_idx = anchor_idx_t.item()
            anchor_zone = zone_labels[anchor_idx].item()
            same_zone = zone_to_indices[anchor_zone]

            if len(same_zone) < 2:
                continue

            pos_candidates = [j for j in same_zone if j != anchor_idx]
            if not pos_candidates:
                continue

            pos_pick = pos_candidates[
                torch.randint(
                    len(pos_candidates), (1,), generator=generator
                ).item()
            ]

            neg_candidates: List[int] = []
            for z_id, indices in zone_to_indices.items():
                if z_id != anchor_zone:
                    neg_candidates.extend(indices)

            if len(neg_candidates) < K:
                continue

            neg_perm = torch.randperm(
                len(neg_candidates), generator=generator
            )[:K]
            neg_picks = [neg_candidates[j.item()] for j in neg_perm]

            anchor_proto = prototypes[anchor_idx]
            pos_proto = prototypes[pos_pick]
            neg_protos = prototypes[neg_picks]

            pos_sim = torch.dot(anchor_proto, pos_proto) / tau
            neg_sims = torch.mv(neg_protos, anchor_proto) / tau

            all_sims = torch.cat([pos_sim.unsqueeze(0), neg_sims])
            log_partition = _log_sum_exp(all_sims, dim=0)

            loss_i = -pos_sim + log_partition
            losses.append(loss_i)

        if not losses:
            return torch.tensor(0.0, device=device, requires_grad=True)

        return torch.stack(losses).mean()


# ═════════════════════════════════════════════════════════════════════════════
# WLP TRAINING INTERFACE
# ═════════════════════════════════════════════════════════════════════════════


class WLPTrainingInterface:
    """
    Interface between latent_parser.py and preparse_daemon.py.
    The only way preparse_daemon.py accesses the LatentParser model.

    Analogous to WLMTrainingInterface in latent_model.py.
    preparse_daemon.py calls these methods. Nothing else does.

    The model is NOT copied — preparse_daemon.py trains the live model.
    update_model_eval() restores eval mode after every training step.
    The live inference path always sees the model in eval() mode.
    """

    def __init__(self, model: LatentParser) -> None:
        self._model = model
        self._training_step: int = 0
        self._best_validation_loss: float = float("inf")

    def get_model(self) -> LatentParser:
        """
        Returns the LatentParser instance for optimizer attachment.
        preparse_daemon.py creates its optimizer pointing at model.parameters().
        """
        return self._model

    def forward_with_grad(
            self,
            data: Data,
            intent_vector: Optional[Tensor],
            node_labels: Optional[Tensor],
            zone_labels: Optional[Tensor],
    ) -> ForwardOutput:
        """
        Training forward pass with gradient flow.

        Sequence:
            1. model.train() — enable dropout, BatchNorm training mode
            2. model.forward() — full forward with loss computation
            3. Return ForwardOutput (caller calls loss.backward())

        Note: update_model_eval() must be called after optimizer.step()
        to restore eval mode for the inference path.
        """
        self._model.train()
        result = self._model.forward(
            data=data,
            intent_vector=intent_vector,
            node_labels=node_labels,
            zone_labels=zone_labels,
        )
        self._training_step += 1
        return result

    def update_model_eval(self) -> None:
        """
        Restore eval mode after training step.

        Must be called after every optimizer.step().
        Ensures:
            - Dropout disabled on inference path.
            - LayerNorm uses running statistics, not batch statistics.
            - model.training == False.
        """
        self._model.eval()

    def record_validation_loss(self, val_loss: float) -> bool:
        """
        Record validation loss and return True if it's a new best.
        Used by preparse_daemon.py to decide when to save checkpoints.
        """
        if val_loss < self._best_validation_loss:
            self._best_validation_loss = val_loss
            return True
        return False

    def save_checkpoint(self, path: Path) -> None:
        """
        Save model checkpoint for preparse_daemon.py.

        Saves:
            - model.state_dict()
            - config (WLPConfig as dict)
            - training_step count
            - best_validation_loss

        To: path (staging path — caller handles atomic rename).
        This file never writes directly to structural_layer.pt from here.
        The checkpoint goes to wlp_model_checkpoint.pt, a separate file.

        Zone knowledge and model weights are NOT bundled together.
        Two separate files. Two separate write paths.
        """
        checkpoint = {
            "model_state_dict": self._model.state_dict(),
            "config": self._model.config.to_dict(),
            "training_step": self._training_step,
            "best_validation_loss": self._best_validation_loss,
        }
        torch.save(checkpoint, path)
        log.info(
            "wlp_checkpoint_saved",
            path=str(path),
            training_step=self._training_step,
            best_val_loss=round(self._best_validation_loss, 6),
        )

    def load_checkpoint(self, path: Path) -> None:
        """
        Load model checkpoint from preparse_daemon.py managed file.

        Restores model weights, training step count, and best validation loss.
        Config is verified against the current model config — mismatches
        in architecture dimensions are a hard error.
        """
        checkpoint = torch.load(path, weights_only=False)
        saved_config = WLPConfig.from_dict(checkpoint.get("config", {}))

        if saved_config.hidden_dim != self._model.config.hidden_dim:
            raise ValueError(
                f"Checkpoint hidden_dim={saved_config.hidden_dim} != "
                f"current model hidden_dim={self._model.config.hidden_dim}. "
                "Architecture mismatch — cannot load checkpoint."
            )
        if saved_config.node_feature_dim != self._model.config.node_feature_dim:
            raise ValueError(
                f"Checkpoint node_feature_dim={saved_config.node_feature_dim} != "
                f"current model node_feature_dim={self._model.config.node_feature_dim}. "
                "Architecture mismatch — cannot load checkpoint."
            )

        self._model.load_state_dict(checkpoint["model_state_dict"])
        self._training_step = checkpoint.get("training_step", 0)
        self._best_validation_loss = checkpoint.get(
            "best_validation_loss", float("inf")
        )
        log.info(
            "wlp_checkpoint_loaded",
            path=str(path),
            training_step=self._training_step,
            best_val_loss=round(self._best_validation_loss, 6),
        )


# ═════════════════════════════════════════════════════════════════════════════
# WORLD LATENT PARSER — THE ORCHESTRATION CLASS
# ═════════════════════════════════════════════════════════════════════════════


class WorldLatentParser:
    """
    The only public-facing component in tag/world_model/ for WLP.

    Owns the three-tier cache (L1/L2/L3).
    Subscribes to the bus (CleanSignalEvent, SurpriseEvent).
    Registers with WATCHDOG (structural_layer.pt).
    Orchestrates: wlp_graph → LatentParser → wlp_zones → ZoneMap.
    Exposes exactly one public method: query().

    Public contract:
        async def query(
            topology_class, domain, intent_vector, phase
        ) -> Union[ZoneMap, EmptyZoneMap]

    One method. Always returns. Never raises. Never returns None.

    Three-tier cache architecture:
        L1: (domain, topology_class) → ZoneMap
            Exact match. 10,000 entries. LRU eviction.
            Hit: < 0.5ms. Expected hit rate: > 80%.

        L2: topology_class → ZoneMap
            Generalized zone pattern. 18 entries (one per topology class).
            Hit: < 0.5ms. Selectors may not match specific domain.

        L3: Fresh parse via LatentParser.readout().
            Full pipeline: graph construction → model inference → zone assembly.
            < 20ms for 10K-node pages.

    Cache hit rate analysis:
        L1 hit rate > 80% after 1 week of production crawling.
        Top 1000 domains serve millions of URLs → high L1 reuse.
        L2 hit rate > 40% for new domains in known topology classes.
        Total cache hit rate (L1 + L2): > 90%.
        L3 fresh parses: < 10% of all queries.
    """

    def __init__(self) -> None:
        # ── Model ────────────────────────────────────────────────────────────
        self._model: Optional[LatentParser] = None
        self._config: Optional[WLPConfig] = None
        self._device: torch.device = self._select_device()

        # ── Zone Knowledge ───────────────────────────────────────────────────
        self._zone_knowledge: Union[Dict, EmptyZoneKnowledge] = EmptyZoneKnowledge()
        self._zone_confirmations: Dict[Tuple[str, str], ZoneConfirmationTracker] = {}

        # ── Cache Tiers ──────────────────────────────────────────────────────
        self._l1_cache: OrderedDict = OrderedDict()
        self._l2_cache: Dict[str, ZoneMap] = {}

        # ── Locks ────────────────────────────────────────────────────────────
        self._write_lock: asyncio.Lock = asyncio.Lock()

        # ── State ────────────────────────────────────────────────────────────
        self._parser_ready: bool = False
        self._cached_router_version: int = 0
        self._bus_subscriptions: List[str] = []
        self._watchdog_registered: bool = False
        self._untrained_model: bool = False
        self._shutdown_requested: bool = False

        # ── Statistics ───────────────────────────────────────────────────────
        self._stats: CacheStatistics = CacheStatistics()

        # ── Bus Emitters ─────────────────────────────────────────────────────────
        self._zone_map_updated_emitter: Optional[TopicEmitter] = None
        self._zone_map_invalidated_emitter: Optional[TopicEmitter] = None

    # ─────────────────────────────────────────────────────────────────────────
    # DEVICE SELECTION
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _select_device() -> torch.device:
        """
        Select inference device.

        Priority:
            1. CUDA if available (RTX 5080 — always use it).
            2. MPS for Apple Silicon (fallback for development).
            3. CPU as last resort.

        Log which device was selected. For CUDA, include VRAM info.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            try:
                vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                log.info(
                    "wlp_device_selected",
                    device="cuda:0",
                    vram_mb=round(vram_mb, 0),
                    gpu_name=torch.cuda.get_device_name(0),
                )
            except Exception: # noqa
                log.info("wlp_device_selected", device="cuda:0")
            return device

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            log.info("wlp_device_selected", device="mps")
            return torch.device("mps")

        log.warning(
            "wlp_device_selected",
            device="cpu",
            msg="No GPU available — inference will be slower",
        )
        return torch.device("cpu")

    # ─────────────────────────────────────────────────────────────────────────
    # INITIALIZATION AND LIFECYCLE
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Called by cold_start.py before interface.py accepts queries.

        Sequence:
            1. Load LatentParser weights from structural_layer.pt
            2. Set model to eval() mode
            3. Move model to self._device
            4. Load zone_knowledge from structural_layer.pt
            5. Register with WATCHDOG for structural_layer.pt changes
            6. Subscribe to bus: CleanSignalEvent, SurpriseEvent
            7. cold_start_warmup() — populate L2 cache for all 18 topology classes
            8. Set self._parser_ready = True

        If structural_layer.pt does not exist:
            Initialize untrained LatentParser with WLPConfig defaults.
            EmptyZoneKnowledge remains as zone_knowledge.
            Log warning: system is functional but ZoneMaps will be
            low confidence until first preparse cycle completes.
        """
        log.info("wlp_initialize_start")
        t_start = time.monotonic()

        try:
            # ── Step 1: Load Model ───────────────────────────────────────────
            self._config = WLPConfig()
            self._load_model()

            # ── Step 2: Eval Mode ────────────────────────────────────────────
            if self._model is not None:
                self._model.eval()

            # ── Step 3: Move to Device ───────────────────────────────────────
            if self._model is not None:
                self._model = self._model.to(self._device)

            # ── Step 4: Load Zone Knowledge ──────────────────────────────────
            await self._reload_zone_knowledge()

            # ── Step 5: Register with WATCHDOG ───────────────────────────────
            try:
                WATCHDOG.register(
                    path=str(STRUCTURAL_LAYER_PATH),
                    handler=lambda: self._on_watchdog_event(str(STRUCTURAL_LAYER_PATH)),
                )  # handler: WorldLatentParser._on_watchdog_event
                self._watchdog_registered = True
                log.info("wlp_watchdog_registered", path=str(STRUCTURAL_LAYER_PATH))
            except Exception as e:
                log.error(
                    "wlp_watchdog_registration_failed",
                    error=str(e),
                    msg="Proceeding without watchdog — manual reload required",
                )

            # ── Step 5b: Acquire Bus Emitters ────────────────────────────────────────
            self._zone_map_updated_emitter = await BUS.emitter(
                "zone_map_updated", "world_model.wlp", ZoneMapUpdatedEvent
            )
            self._zone_map_invalidated_emitter = await BUS.emitter(
                "zone_map_invalidated", "world_model.wlp", ZoneMapInvalidatedEvent
            )

            # ── Step 6: Subscribe to Bus ─────────────────────────────────────
            try:
                await BUS.subscribe(
                    "clean_signal",
                    group="world_model.wlp",
                    handler=self._on_clean_signal,
                    schema=CleanSignalEvent,
                )  # handler: WorldLatentParser._on_clean_signal
                self._bus_subscriptions.append("clean_signal")
                log.info("wlp_bus_subscribed", topic="clean_signal")
            except Exception as e:
                log.error("wlp_bus_subscribe_failed", topic="clean_signal", error=str(e))

            try:
                await BUS.subscribe(
                    "surprise",
                    group="world_model.wlp",
                    handler=self._on_surprise,
                    schema=SurpriseEvent,
                )  # handler: WorldLatentParser._on_surprise
                self._bus_subscriptions.append("surprise")
                log.info("wlp_bus_subscribed", topic="surprise")
            except Exception as e:
                log.error("wlp_bus_subscribe_failed", topic="surprise", error=str(e))

            # ── Step 7: Cold Start Warmup ────────────────────────────────────
            await self.cold_start_warmup()

            # ── Step 8: Ready ────────────────────────────────────────────────
            self._parser_ready = True
            elapsed_ms = (time.monotonic() - t_start) * 1000
            log.info(
                "wlp_initialize_complete",
                elapsed_ms=round(elapsed_ms, 1),
                device=str(self._device),
                model_loaded=self._model is not None,
                untrained=self._untrained_model,
                zone_knowledge_count=len(self._zone_knowledge)
                if isinstance(self._zone_knowledge, dict) else 0,
                l2_classes_warmed=len(self._l2_cache),
            )

        except Exception as e:
            log.error(
                "wlp_initialize_failed",
                error=str(e),
                traceback=traceback.format_exc(),
            )
            self._parser_ready = False
            raise

    async def shutdown(self) -> None:
        """
        Graceful shutdown. Called by cold_start.py during process termination.

        Sequence:
            1. Set shutdown flag to prevent new cache writes.
            2. Flush any pending zone knowledge writes.
            3. Clear caches to release memory.
            4. Move model to CPU to release GPU memory.
        """
        log.info("wlp_shutdown_start")
        self._shutdown_requested = True
        self._parser_ready = False

        async with self._write_lock:
            pass

        self._l1_cache.clear()
        self._l2_cache.clear()

        if self._model is not None:
            try:
                self._model = self._model.to(torch.device("cpu"))
            except Exception: # noqa
                pass

        log.info("wlp_shutdown_complete")

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY — THE CRITICAL PATH METHOD
    # ─────────────────────────────────────────────────────────────────────────

    async def query(
            self,
            topology_class: str,
            domain: str,
            intent_vector: Optional[List[float]] = None,
            phase: int = PHASE_I,
    ) -> Union[ZoneMap, EmptyZoneMap]:
        """
        THE critical path method. Called for every request via asyncio.gather().
        Must be fast. Must never raise. Must never return None.

        Cache routing:
            L1 check → L2 check → L3 fresh parse.
            Each tier returns immediately if hit.
            Intent conditioning applied on cache hit via zone_map.with_intent().

        Phase conditioning:
            Phase III domains skip L2 and go straight to L3 if L1 misses.
            Phase I/II domains use L2 cache (generalized is acceptable).

        Returns:
            ZoneMap on success.
            EmptyZoneMap on any failure.
            Never None. Never raises.
        """
        if not self._parser_ready:
            self._stats.record_empty_return()
            return EmptyZoneMap()

        try:
            # ── L1 Check ─────────────────────────────────────────────────────
            zone_map = self._l1_lookup(domain, topology_class)
            if zone_map is not None:
                current_version = self._current_router_version()
                if zone_map.is_stale(current_version):
                    self._l1_evict(domain, topology_class)
                else:
                    self._stats.record_l1_hit()
                    if intent_vector is not None:
                        intent_tags = parse_intent_tags(intent_vector)
                        return zone_map.with_intent(intent_vector, intent_tags)
                    return zone_map

            self._stats.record_l1_miss()

            # ── L2 Check (skip for Phase III) ────────────────────────────────
            if phase != PHASE_III:
                zone_map = self._l2_lookup(topology_class)
                if zone_map is not None:
                    self._stats.record_l2_hit()
                    if intent_vector is not None:
                        intent_tags = parse_intent_tags(intent_vector)
                        return zone_map.with_intent(intent_vector, intent_tags)
                    return zone_map

                self._stats.record_l2_miss()

            # ── L3 Fresh Parse ───────────────────────────────────────────────
            t_start = time.monotonic()
            zone_map = await self._l3_fresh_parse(
                topology_class=topology_class,
                domain=domain,
                intent_vector=intent_vector,
            )
            elapsed_ms = (time.monotonic() - t_start) * 1000
            self._stats.record_l3_parse(
                latency_ms=elapsed_ms,
                success=not isinstance(zone_map, EmptyZoneMap),
            )

            if isinstance(zone_map, EmptyZoneMap):
                self._stats.record_empty_return()

            return zone_map

        except Exception as e:
            log.error(
                "wlp_query_failed",
                topology_class=topology_class,
                domain=domain,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            self._stats.record_empty_return()
            return EmptyZoneMap()

    # ─────────────────────────────────────────────────────────────────────────
    # L1 CACHE — (domain, topology_class) → ZoneMap
    # ─────────────────────────────────────────────────────────────────────────

    def _l1_lookup(
            self, domain: str, topology_class: str
    ) -> Optional[Union[ZoneMap, EmptyZoneMap]]:
        """
        L1 cache lookup with LRU maintenance.

        Returns the ZoneMap component of the stored _CacheEntry, not the entry
        itself. Callers (query(), _l3_fresh_parse()) receive a plain ZoneMap —
        they have no reason to interact with the DAG directly. The DAG is an
        internal implementation detail of the cache layer, not part of the
        public contract of this method.

        On hit: entry is moved to end of OrderedDict (most recently used).
        On miss: returns None. Caller falls through to L2 or L3.

        Complexity: O(1) — OrderedDict lookup + move_to_end.
        Latency budget: this method must complete in < 0.1ms.
        It is called on every query before any other work.
        """
        key = (domain, topology_class)
        if key in self._l1_cache:
            self._l1_cache.move_to_end(key)
            return self._l1_cache[key].zone_map  # unwrap: callers get ZoneMap, not _CacheEntry
        return None

    def _l1_store(
            self,
            domain: str,
            topology_class: str,
            zone_map: Union[ZoneMap, EmptyZoneMap],
    ) -> None:
        """
        Store a ZoneMap in L1 cache as a _CacheEntry, building its Merkle DAG
        at storage time.

        This is the only place ZoneMerkleDAG.from_zone_map() is ever called in
        the cache layer. The DAG is built once here and amortized across every
        subsequent cache hit for this (domain, topology_class) key. The O(n)
        cost of DAG construction is paid once at L3-miss time (when the ZoneMap
        was just produced by a full fresh parse anyway — the marginal cost is
        negligible relative to the L3 pipeline itself).

        EmptyZoneMap is never stored. Empty results carry no structural
        information worth caching, and EmptyZoneMap has no signal_zones to
        build a meaningful DAG from.

        LRU eviction: if L1 is at capacity after insertion, the least recently
        used entry (first item in OrderedDict) is evicted. Eviction is logged
        at debug level — high eviction rate indicates L1 is undersized for the
        domain distribution, not a correctness issue.

        Complexity: O(n) where n = total zone + boundary count of zone_map,
        dominated by ZoneMerkleDAG.from_zone_map(). The hash operations are
        SHA3-256 (hardware-accelerated on x86 with SHA-NI extensions).
        Expected wall time for a typical ZoneMap (8–12 zones): < 0.05ms.
        """
        if isinstance(zone_map, EmptyZoneMap):
            return

        key   = (domain, topology_class)
        entry = _CacheEntry(
            zone_map=zone_map,
            dag=ZoneMerkleDAG.from_zone_map(zone_map),  # built once, amortized across all hits
        )
        self._l1_cache[key] = entry
        self._l1_cache.move_to_end(key)

        while len(self._l1_cache) > self._config.l1_max_size:
            evicted_key, _ = self._l1_cache.popitem(last=False)
            self._stats.record_l1_eviction()
            log.debug(
                "l1_cache_eviction",
                evicted_domain=evicted_key[0],
                evicted_class=evicted_key[1],
                cache_size=len(self._l1_cache),
            )

    def _l1_evict(self, domain: str, topology_class: str) -> None:
        """Remove a specific entry from L1 cache."""
        key = (domain, topology_class)
        if key in self._l1_cache:
            del self._l1_cache[key]

    def _l1_evict_by_topology_class(self, topology_class: str) -> int:
        """
        Evict ALL L1 entries matching a topology class.

        Called by _on_surprise() when dissolve_triggered.
        O(n) scan of L1 cache — acceptable because this is infrequent
        (surprise-triggered only) and L1 max size is 10,000.

        Returns the number of evicted entries.
        """
        evict_keys = [
            key for key in self._l1_cache
            if key[1] == topology_class
        ]
        for key in evict_keys:
            del self._l1_cache[key]
        return len(evict_keys)

    # ─────────────────────────────────────────────────────────────────────────
    # L2 CACHE — topology_class → ZoneMap
    # ─────────────────────────────────────────────────────────────────────────

    def _l2_lookup(self, topology_class: str) -> Optional[ZoneMap]:
        """
        L2 cache lookup. Returns the generalized ZoneMap for a topology class.

        L2 has at most 18 entries (one per topology class).
        No LRU — all entries persist until explicitly evicted or replaced.

        Returns None on cache miss.
        Returns ZoneMap on hit (never EmptyZoneMap — not stored).
        """
        return self._l2_cache.get(topology_class)

    def _l2_update(self, topology_class: str, zone_map: Union[ZoneMap, EmptyZoneMap]) -> None:
        """
        Update L2 cache with a new ZoneMap for a topology class.

        Only stores if the new ZoneMap has higher confidence than the
        existing entry (or if no entry exists).

        Never stores EmptyZoneMap.
        """
        if isinstance(zone_map, EmptyZoneMap):
            return

        existing = self._l2_cache.get(topology_class)
        if existing is None or zone_map.confidence > existing.confidence:
            self._l2_cache[topology_class] = zone_map
            log.debug(
                "l2_cache_updated",
                topology_class=topology_class,
                confidence=round(zone_map.confidence, 4),
                previous_confidence=round(existing.confidence, 4) if existing else 0.0,
            )

    def _l2_evict(self, topology_class: str) -> None:
        """Remove L2 cache entry for a topology class."""
        if topology_class in self._l2_cache:
            del self._l2_cache[topology_class]

    # ─────────────────────────────────────────────────────────────────────────
    # MERKLE SURGICAL DECAY
    # ─────────────────────────────────────────────────────────────────────────

    def _merkle_surgical_decay(
            self,
            topology_class: str,
            surprise_selector: str,
            surprise_score: float,
    ) -> int:
        """
        Decay confidence only on L1 cache entries whose Merkle DAG contains
        surprise_selector in their signal zone set. All other entries for the
        same topology class are left completely untouched.

        This is the Merkle-enabled path for partial SurpriseEvents. It replaces
        the topology-wide decay that the fallback path applies to _zone_knowledge
        when no selector is available. The difference in scope:

            Fallback path (no selector):
                Decays ALL ZoneMaps of this topology class in _zone_knowledge.
                Blunt instrument — penalises domains whose structure is unrelated
                to the surprise. A NEWS_ARTICLE surprise on reuters.com should
                not decay the cached ZoneMap for bbc.com unless bbc.com uses the
                same structural selector that caused the surprise.

            Surgical path (selector known):
                Decays only L1 entries where dag.contains_selector(selector).
                A reuters.com surprise on ".article-body > p" only decays entries
                that contain ".article-body > p" in their signal zone set.
                bbc.com's ZoneMap, which uses "[role=main] > article", is untouched.

        Why O(1) per entry:
            dag.contains_selector() is a frozenset __contains__ check — O(1)
            average case, O(n) worst case only on hash collision (negligible for
            CSS selector strings). The outer loop is O(k) where k = number of L1
            entries for this topology_class, not O(L1_CACHE_MAX_SIZE).

        Confidence update mechanics:
            Uses _exponential_decay() — same function as the fallback path.
            The ZoneMap is replaced via dataclasses.replace(confidence=new_conf).
            The DAG is NOT rebuilt — dag.root is unchanged because confidence
            is not a DAG input. The _CacheEntry is replaced with a new instance
            that carries the same dag object and a new zone_map with decayed
            confidence. The structural identity is preserved.

        Why this method does not touch _zone_knowledge:
            _zone_knowledge is the persistent store (structural_layer.pt shadow).
            It is written by _on_watchdog_event() and _l3_fresh_parse().
            This method operates only on the in-memory L1 cache.
            _zone_knowledge decay for surgical events is deferred — the next
            watchdog reload will read updated confidence from structural_layer.pt,
            which surprise_detector.py is responsible for updating.

        Called exclusively from _on_surprise() when dissolve_triggered=False
        and event.surprise_zone_selector is not None.
        Never called from the query hot path.

        Parameters:
            topology_class:     The topology class the SurpriseEvent addresses.
                                Entries for other topology classes are never touched.
            surprise_selector:  The CSS selector of the zone that caused the surprise.
                                Provided by surprise_detector.py in SurpriseEvent.
                                See: surprise_detector.py — wire into SurpriseEvent
                                as surprise_zone_selector when building that module.
            surprise_score:     Float in [0.0, 1.0]. Passed to _exponential_decay().
                                Higher score = steeper confidence decay.

        Returns:
            Number of L1 entries that were decayed. Zero if no entry contains
            surprise_selector. Logged as l1_entries_decayed in the caller.

        Complexity: O(k) where k = L1 entries for topology_class.
        Expected k at production scale: 10–500 (top domains per class).
        """
        decayed = 0
        for key, entry in list(self._l1_cache.items()):
            if key[1] != topology_class:
                continue
            if not entry.dag.contains_selector(surprise_selector):
                continue

            new_conf = _exponential_decay(
                confidence=entry.zone_map.confidence,
                surprise_score=surprise_score,
            )
            new_zone_map = dataclasses.replace(entry.zone_map, confidence=new_conf)

            # Replace ZoneMap in entry. DAG is unchanged — confidence is not a
            # DAG input. dag.root is identical before and after this operation.
            # dataclasses.replace() on a frozen dataclass constructs a new
            # _CacheEntry instance with zone_map overridden. All other fields
            # (dag) are shallow-copied from the original entry.
            self._l1_cache[key] = dataclasses.replace(entry, zone_map=new_zone_map)
            decayed += 1

        return decayed

    # ─────────────────────────────────────────────────────────────────────────
    # L3 — FRESH PARSE (FULL PIPELINE)
    # ─────────────────────────────────────────────────────────────────────────

    async def _l3_fresh_parse(
            self,
            topology_class: str,
            domain: str,
            intent_vector: Optional[List[float]],
            raw_content: Optional[bytes] = None,
            content_type: str = "html",
    ) -> Union[ZoneMap, EmptyZoneMap]:
        """
        Full pipeline: raw bytes → graph → classification → ZoneMap.

        raw_content is None when called from query() cache miss —
        we don't have the content here. Return EmptyZoneMap.
        raw_content is provided when called from _on_clean_signal().

        Pipeline when raw_content is provided:
            1. Intent tensor preparation
            2. Graph construction via wlp_graph.cst_to_pyg_graph()
            3. Model inference via LatentParser.readout()
            4. Zone assembly via wlp_zones.assemble_zone_map()
            5. Zone coherence validation (prototype-based)
            6. Cache store and background write
            7. Return ZoneMap

        Never raises. Returns EmptyZoneMap on any exception.
        """
        if raw_content is None:
            return EmptyZoneMap()

        try:
            # ── Step 1: Intent Tensor ────────────────────────────────────────
            intent_tensor: Optional[Tensor] = None
            if intent_vector is not None:
                intent_tensor = torch.tensor(
                    intent_vector, dtype=torch.float32
                ).to(self._device)

            # ── Step 2: Graph Construction ───────────────────────────────────
            data = await cst_to_pyg_graph(
                content=raw_content,
                topology_class=topology_class,
                content_type=content_type,
                intent_vector=intent_vector,
            )

            if data is None:
                log.warning(
                    "l3_graph_construction_failed",
                    domain=domain,
                    topology_class=topology_class,
                )
                return await self.discover_signal_zones(
                    domain=domain,
                    topology_class=topology_class,
                )

            data = data.to(self._device)

            # ── Step 3: Model Inference ──────────────────────────────────────
            if self._model is None:
                log.warning("l3_model_not_loaded", domain=domain)
                return await self.discover_signal_zones(
                    domain=domain,
                    topology_class=topology_class,
                    data=data,
                )

            self._model.eval()
            result: ReadoutOutput = self._model.readout(data, intent_tensor)

            # ── Step 4: Zone Assembly ────────────────────────────────────────
            cst_nodes = getattr(data, "cst_nodes", None)
            current_version = self._current_router_version()

            zone_map = assemble_zone_map(
                node_classifications=result.logits,
                node_confidences=result.confidences,
                cst_nodes=cst_nodes,
                topology_class=topology_class,
                domain=domain,
                intent_vector=intent_vector,
                topology_router_version=current_version,
            )

            if isinstance(zone_map, EmptyZoneMap):
                log.info(
                    "l3_assembly_produced_empty_zone_map",
                    domain=domain,
                    topology_class=topology_class,
                )
                return zone_map

            # ── Step 5: Zone Coherence Validation ────────────────────────────
            zone_map = self._validate_zone_coherence(
                zone_map=zone_map,
                prototypes=result.prototypes,
                data=data,
            )

            # ── Step 6: Cache and Store ──────────────────────────────────────
            self._l1_store(domain, topology_class, zone_map)
            self._l2_update(topology_class, zone_map)

            if not self._shutdown_requested:
                asyncio.create_task(
                    self._write_zone_knowledge(domain, topology_class, zone_map)
                )

            # ── Step 7: Return ───────────────────────────────────────────────
            log.debug(
                "l3_fresh_parse_complete",
                domain=domain,
                topology_class=topology_class,
                n_signal_zones=len(zone_map.signal_zones),
                confidence=round(zone_map.confidence, 4),
            )
            return zone_map

        except Exception as e:
            log.error(
                "l3_fresh_parse_failed",
                domain=domain,
                topology_class=topology_class,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return EmptyZoneMap()

    # ─────────────────────────────────────────────────────────────────────────
    # ZONE COHERENCE VALIDATION
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_zone_coherence(
            self,
            zone_map: Union[ZoneMap, EmptyZoneMap],
            prototypes: Tensor,
            data: Data,
    ) -> Union[ZoneMap, EmptyZoneMap]:
        """
        Use zone prototype embeddings to validate and refine zone groupings.

        This is the novel feature that Head 4 (prototype head) enables.
        Standard GraphSAGE classification cannot detect mis-grouped zones.
        This can.

        Algorithm for each signal zone:
            1. Collect prototype vectors for nodes in this zone.
            2. Compute prototype variance (WCSS / n).
            3. If variance > COHERENCE_THRESHOLD: split via PCA.
            4. Rebuild ZoneMap with split zones.

        Mathematical foundation:
            The prototype head maps each node to a point on S^63.
            Same-zone nodes cluster together (trained by contrastive loss).
            Different-zone nodes push apart.

            When two structural zones are incorrectly merged by the
            adjacency-based grouping in wlp_zones.py, their prototype
            embeddings form a bimodal distribution on the hypersphere.

            PCA of the centered prototypes finds the axis of maximum
            variance — the direction separating the two clusters.
            The maximum gap in projected values locates the split point.

            This is mathematically principled zone boundary refinement.
            The model learned where boundaries are. We read that from
            the geometry of the prototype space.

        Returns:
            New ZoneMap with split zones if any were incoherent.
            Original ZoneMap unchanged if all zones are coherent.
        """
        if isinstance(zone_map, EmptyZoneMap):
            return zone_map

        if not zone_map.signal_zones:
            return zone_map

        splits_performed = False
        new_signal_zones: List[ZoneDescriptor] = []
        cst_nodes = getattr(data, "cst_nodes", None)

        for zone_idx, zone_desc in enumerate(zone_map.signal_zones):
            zone_node_indices = self._get_zone_node_indices(
                zone_desc, data, zone_map
            )

            if (
                    zone_node_indices is None
                    or len(zone_node_indices) < self._config.coherence_min_zone_size
            ):
                new_signal_zones.append(zone_desc)
                continue

            zone_protos = prototypes[zone_node_indices]
            variance = _prototype_variance(zone_protos)

            if variance > self._config.coherence_threshold:
                sub_zones = self._split_incoherent_zone(
                    zone_desc=zone_desc,
                    node_indices=zone_node_indices,
                    prototypes=prototypes,
                    cst_nodes=cst_nodes,
                    split_depth=0,
                )
                new_signal_zones.extend(sub_zones)
                splits_performed = True
                log.info(
                    "zone_coherence_split",
                    zone_selector=zone_desc.selector,
                    variance=round(variance, 4),
                    threshold=self._config.coherence_threshold,
                    n_sub_zones=len(sub_zones),
                    n_original_nodes=len(zone_node_indices),
                )
            else:
                new_signal_zones.append(zone_desc)

        if not splits_performed:
            return zone_map

        return dataclasses.replace(
            zone_map,
            signal_zones=tuple(new_signal_zones),
            signal_node_count=sum(1 for _ in new_signal_zones),
        )

    def _get_zone_node_indices( # noqa
            self,
            zone_desc: ZoneDescriptor,
            data: Data,
            zone_map: Union[ZoneMap, EmptyZoneMap], # noqa
    ) -> Optional[List[int]]:
        """
        Get node indices belonging to a zone descriptor.

        Uses data.zone_node_indices if available (set by wlp_zones.py).
        Falls back to matching nodes by selector pattern.

        Returns None if indices cannot be determined.
        """
        zone_node_map = getattr(data, "zone_node_indices", None)
        if zone_node_map is not None and isinstance(zone_node_map, dict):
            indices = zone_node_map.get(zone_desc.selector)
            if indices is not None:
                return indices

        cst_nodes = getattr(data, "cst_nodes", None)
        if cst_nodes is None:
            return None

        indices = []
        for i, node in enumerate(cst_nodes):
            node_type = getattr(node, "node_type", "")
            css_classes = getattr(node, "css_classes", [])
            selector_parts = zone_desc.selector.replace(".", " ").split()
            if node_type in selector_parts or any(
                    c in selector_parts for c in css_classes
            ):
                indices.append(i)

        return indices if indices else None

    def _split_incoherent_zone(
            self,
            zone_desc: ZoneDescriptor,
            node_indices: List[int],
            prototypes: Tensor,
            cst_nodes: Optional[List[Any]],
            split_depth: int = 0,
    ) -> List[ZoneDescriptor]:
        """
        Split an incoherent zone using PCA on prototype embeddings.

        Algorithm:
            1. Extract prototype vectors for zone nodes.
            2. Center the prototypes (subtract mean).
            3. Compute first principal component via SVD.
            4. Project nodes onto first PC.
            5. Sort by projection value.
            6. Find maximum gap in sorted projections.
            7. Split at the gap into two sub-zones.
            8. Recursively validate sub-zones (up to max depth).
            9. Generate new ZoneDescriptors for each sub-zone.

        Mathematical details:
            Let Z ∈ R^{k×64} be the prototype matrix for k zone nodes.
            Centered: Z_c = Z - mean(Z)
            SVD: Z_c = U Σ V^T
            First PC: v_1 = V^T[0] ∈ R^64
            Projections: p = Z_c · v_1 ∈ R^k
            Sort: p_sorted = sort(p)
            Gaps: g_i = p_sorted[i+1] - p_sorted[i]
            Split: split_idx = argmax(g) + 1

            Group A: nodes at sorted indices [0, split_idx)
            Group B: nodes at sorted indices [split_idx, k)

        Returns:
            List of ZoneDescriptors for the sub-zones.
            If splitting fails or reaches max depth, returns [zone_desc].
        """
        if split_depth >= self._config.coherence_max_split_depth:
            return [zone_desc]

        if len(node_indices) < 2 * self._config.coherence_min_zone_size:
            return [zone_desc]

        try:
            zone_protos = prototypes[node_indices]

            centroid = zone_protos.mean(dim=0, keepdim=True)
            centered = zone_protos - centroid

            first_pc = _pca_first_component(centered)
            if first_pc.abs().max().item() < EPS:
                return [zone_desc]

            projections = torch.mv(centered, first_pc)
            sorted_indices = torch.argsort(projections)
            sorted_projections = projections[sorted_indices]
            split_point = _find_maximum_gap(sorted_projections)

            if split_point < self._config.coherence_min_zone_size:
                return [zone_desc]
            if (len(node_indices) - split_point) < self._config.coherence_min_zone_size:
                return [zone_desc]

            group_a_sorted = sorted_indices[:split_point].tolist()
            group_b_sorted = sorted_indices[split_point:].tolist()

            group_a_global = [node_indices[i] for i in group_a_sorted]
            group_b_global = [node_indices[i] for i in group_b_sorted]

            sub_zone_a = self._build_sub_zone_descriptor(
                parent_desc=zone_desc,
                node_indices=group_a_global,
                cst_nodes=cst_nodes,
                sub_zone_label="A",
            )

            sub_zone_b = self._build_sub_zone_descriptor(
                parent_desc=zone_desc,
                node_indices=group_b_global,
                cst_nodes=cst_nodes,
                sub_zone_label="B",
            )

            result_zones: List[ZoneDescriptor] = []

            variance_a = _prototype_variance(prototypes[group_a_global])
            if variance_a > self._config.coherence_threshold:
                result_zones.extend(
                    self._split_incoherent_zone(
                        sub_zone_a, group_a_global, prototypes,
                        cst_nodes, split_depth + 1,
                    )
                )
            else:
                result_zones.append(sub_zone_a)

            variance_b = _prototype_variance(prototypes[group_b_global])
            if variance_b > self._config.coherence_threshold:
                result_zones.extend(
                    self._split_incoherent_zone(
                        sub_zone_b, group_b_global, prototypes,
                        cst_nodes, split_depth + 1,
                    )
                )
            else:
                result_zones.append(sub_zone_b)

            return result_zones

        except Exception as e:
            log.warning(
                "zone_split_failed",
                zone_selector=zone_desc.selector,
                error=str(e),
                split_depth=split_depth,
            )
            return [zone_desc]

    def _build_sub_zone_descriptor( # noqa
            self,
            parent_desc: ZoneDescriptor,
            node_indices: List[int],
            cst_nodes: Optional[List[Any]],
            sub_zone_label: str,
    ) -> ZoneDescriptor:
        """
        Build a ZoneDescriptor for a sub-zone produced by coherence splitting.

        Inherits most properties from the parent zone.
        Selector is refined based on the actual nodes in the sub-zone
        if CSTNode objects are available. Otherwise, appends a sub-zone
        suffix to the parent selector.

        Density is recomputed from the sub-zone nodes.
        Priority is inherited (will be reassigned by the caller).
        """
        if cst_nodes is not None and node_indices:
            sub_nodes = [cst_nodes[i] for i in node_indices if i < len(cst_nodes)]
            if sub_nodes:
                try:
                    candidate = make_candidate_zone(
                        nodes=sub_nodes,
                        parent_index=getattr(sub_nodes[0], "parent_index", -1),
                        first_node_index=node_indices[0] if node_indices else 0,
                    )
                    selector = generate_css_selector(candidate)
                    scope = determine_scope(candidate, cst_nodes)
                    content_type = infer_content_type(candidate)
                    density = compute_density(candidate)
                    avg_depth = sum(
                        getattr(n, "depth", 0) for n in sub_nodes
                    ) / max(len(sub_nodes), 1)

                    return ZoneDescriptor(
                        selector=selector,
                        selector_type=parent_desc.selector_type,
                        scope=scope,
                        content_type=content_type,
                        average_depth=avg_depth,
                        density=density,
                        priority=parent_desc.priority,
                    )
                except Exception: # noqa
                    pass

        return dataclasses.replace(
            parent_desc,
            selector=f"{parent_desc.selector}:sub-{sub_zone_label}",
            density=parent_desc.density * 0.9,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # DISCOVER SIGNAL ZONES — THREE-PASS ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    async def discover_signal_zones(
            self,
            domain: str,
            topology_class: str = "GENERIC_HTML",
            data: Optional[Data] = None,
    ) -> Union[ZoneMap, EmptyZoneMap]:
        """
        Auto-discover signal zones for unknown or dissolved topology.

        Called when:
            - topology_class == GENERIC_HTML
            - SurpriseEvent dissolved the existing ZoneMap
            - All cache tiers miss and raw_content unavailable
            - cst_to_pyg_graph() returned None

        Three-pass analysis:

        PASS 1 — Structural Heuristics (no ML, O(n)):
            High text density AND low link density → candidate SIGNAL.
            ARIA roles and semantic elements → strong classification.
            Headings at shallow depth → BOUNDARY.
            Confidence: 0.55 per zone (below threshold for L1 cache).

        PASS 2 — GraphSAGE on GENERIC_HTML:
            If model and data available: run readout().
            Combine with heuristic zones in Pass 3.

        PASS 3 — Confidence-Weighted Merge:
            Agreement boost: +0.10 when both estimators agree.
            Heuristic-only: keep at 0.55.
            Model-only: discount to model_confidence × 0.80.
            Overall ceiling: 0.70 (discovery path maximum).

        Subclass candidacy:
            After 10 confirmed high-quality extractions, emit
            ZoneMapUpdatedEvent with subclass_candidate=True.
        """
        try:
            heuristic_zones: List[ZoneDescriptor] = []
            model_zones: List[ZoneDescriptor] = []

            # ── PASS 1: Structural Heuristics ────────────────────────────────
            cst_nodes = None
            if data is not None:
                cst_nodes = getattr(data, "cst_nodes", None)

            if cst_nodes is not None:
                heuristic_zones = self._heuristic_zone_discovery(cst_nodes)

            # ── PASS 2: GraphSAGE Discovery ──────────────────────────────────
            if self._model is not None and data is not None:
                try:
                    self._model.eval()
                    result = self._model.readout(data)
                    model_zone_map = assemble_zone_map(
                        node_classifications=result.logits,
                        node_confidences=result.confidences,
                        cst_nodes=cst_nodes,
                        topology_class=topology_class,
                        domain=domain,
                        intent_vector=None,
                        topology_router_version=self._current_router_version(),
                    )
                    if not isinstance(model_zone_map, EmptyZoneMap):
                        model_zones = list(model_zone_map.signal_zones)
                except Exception as e:
                    log.warning(
                        "discover_model_pass_failed",
                        domain=domain,
                        error=str(e),
                    )

            # ── PASS 3: Confidence-Weighted Merge ────────────────────────────
            merged_zones = self._merge_discovery_zones(
                heuristic_zones, model_zones
            )

            if not merged_zones:
                log.info(
                    "discover_signal_zones_empty",
                    domain=domain,
                    topology_class=topology_class,
                )
                return EmptyZoneMap()

            overall_confidence = min(
                DISCOVERY_CONFIDENCE_CEILING,
                sum(z.density for z in merged_zones) / max(len(merged_zones), 1),
            )

            zone_map = ZoneMap(
                topology_class=topology_class,
                domain=domain,
                signal_zones=tuple(merged_zones),
                noise_zones=(),
                boundaries=(),
                extraction_strategy=ExtractionStrategy.DEPTH_FIRST,
                intent_weights=tuple(
                    (z.selector, 1.0) for z in merged_zones
                ),
                confidence=overall_confidence,
                node_count=data.x.shape[0] if data is not None else 0,
                signal_node_count=len(merged_zones),
                noise_node_count=0,
                boundary_node_count=0,
                version=0,
                produced_at=time.monotonic(),
                topology_router_version=self._current_router_version(),
            )

            self._l1_store(domain, topology_class, zone_map)
            self._l2_update(topology_class, zone_map)

            tracker = self._zone_confirmations.setdefault(
                (domain, topology_class),
                ZoneConfirmationTracker(),
            )

            if tracker.is_stable():
                subclass = self._infer_subclass(zone_map)
                if subclass is not None:
                    try:
                        await self._zone_map_updated_emitter.emit(ZoneMapUpdatedEvent(
                            topology_class=topology_class,
                            new_zone_map=zone_map,
                        ))
                        log.info(
                            "subclass_candidate_emitted",
                            domain=domain,
                            topology_class=topology_class,
                            suggested_class=subclass,
                        )
                    except Exception as e:
                        log.warning("discover_emit_failed", error=str(e))

            log.info(
                "discover_signal_zones_complete",
                domain=domain,
                topology_class=topology_class,
                n_zones=len(merged_zones),
                confidence=round(overall_confidence, 4),
            )

            return zone_map

        except Exception as e:
            log.error(
                "discover_signal_zones_failed",
                domain=domain,
                topology_class=topology_class,
                error=str(e),
                traceback=traceback.format_exc(),
            )
            return EmptyZoneMap()

    def _heuristic_zone_discovery( # noqa
            self, cst_nodes: List[Any]
    ) -> List[ZoneDescriptor]:
        """
        PASS 1: Structural heuristic zone discovery.

        Rules (O(n) pass over node list):
            High text density (> 0.6) AND low link density (< 0.3)
            AND not matching nav/sidebar/footer patterns → candidate SIGNAL.
            role="main" or <main> → strong SIGNAL.
            role="navigation" or <nav> → NOISE (skip).
            role="complementary" or <aside> → NOISE (skip).
            <footer> → NOISE (skip).
            h1-h3 at depth ≤ 4 → BOUNDARY.

        Returns list of ZoneDescriptors with confidence 0.55.
        """
        zones: List[ZoneDescriptor] = []
        noise_patterns = {"nav", "footer", "aside", "header"}

        for node in cst_nodes:
            node_type = getattr(node, "node_type", "")
            role = getattr(node, "role", "")
            depth = getattr(node, "depth", 999)
            css_classes = getattr(node, "css_classes", [])
            text_density = getattr(node, "text_density_normalized", 0.0)
            link_density = getattr(node, "link_density_normalized", 1.0)

            if node_type in noise_patterns or role in ("navigation", "complementary"):
                continue

            is_main = (role == "main" or node_type == "main")
            is_article = (role == "article" or node_type == "article")
            is_high_text = (text_density > 0.6 and link_density < 0.3)

            if is_main or is_article or is_high_text:
                is_in_noise_context = any(
                    c in css_classes for c in ["sidebar", "nav", "footer", "ad", "ads"]
                )
                if is_in_noise_context:
                    continue

                try:
                    _candidate = make_candidate_zone(
                        nodes=[node],
                        parent_index=getattr(node, "parent_index", -1),
                        first_node_index=getattr(node, "index", 0),
                    )
                    selector = generate_css_selector(_candidate)
                    scope = determine_scope(_candidate, cst_nodes)
                    content_type = infer_content_type(_candidate)

                    zones.append(ZoneDescriptor(
                        selector=selector,
                        selector_type="css",
                        scope=scope,
                        content_type=content_type,
                        average_depth=float(depth),
                        density=0.55,
                        priority=len(zones),
                    ))
                except Exception: # noqa
                    continue

        return zones

    def _merge_discovery_zones( # noqa
            self,
            heuristic_zones: List[ZoneDescriptor],
            model_zones: List[ZoneDescriptor],
    ) -> List[ZoneDescriptor]:
        """
        PASS 3: Merge heuristic and model discovery zones.

        For each zone in the union:
            Both agree → confidence = (0.55 + model_conf) / 2 + 0.10
            Heuristic only → keep at 0.55
            Model only → model_conf × 0.80

        Agreement is determined by selector overlap: if a heuristic zone
        selector is a substring of a model zone selector (or vice versa),
        the zones are considered to agree.
        """
        if not heuristic_zones and not model_zones:
            return []

        if not model_zones:
            return heuristic_zones

        if not heuristic_zones:
            return [
                dataclasses.replace(z, density=z.density * 0.80)
                for z in model_zones
            ]

        heuristic_selectors = {z.selector for z in heuristic_zones} # noqa
        model_selectors = {z.selector for z in model_zones} # noqa

        merged: List[ZoneDescriptor] = []
        used_model: Set[str] = set()

        for h_zone in heuristic_zones:
            match = None
            for m_zone in model_zones:
                if (
                        h_zone.selector in m_zone.selector
                        or m_zone.selector in h_zone.selector
                        or h_zone.selector == m_zone.selector
                ):
                    match = m_zone
                    used_model.add(m_zone.selector)
                    break

            if match is not None:
                merged_confidence = _weighted_confidence_merge(
                    heuristic_confidence=h_zone.density,
                    model_confidence=match.density,
                    agreement=True,
                )
                merged.append(dataclasses.replace(
                    h_zone,
                    density=min(merged_confidence, DISCOVERY_CONFIDENCE_CEILING),
                ))
            else:
                merged.append(h_zone)

        for m_zone in model_zones:
            if m_zone.selector not in used_model:
                merged.append(dataclasses.replace(
                    m_zone,
                    density=min(m_zone.density * 0.80, DISCOVERY_CONFIDENCE_CEILING),
                ))

        return merged

    def _infer_subclass(self, zone_map: Union[ZoneMap, EmptyZoneMap]) -> Optional[str]: # noqa
        """
        Infer a topology subclass from zone map content types.

        Examines signal_zones content_types:
            Majority "code" → SAAS_DOCS_WITH_CODE
            Majority "table" → REST_API_JSON
            Majority "prose" → BLOG_POST
            Majority "list" → FORUM_THREAD
            Default: None (no suggestion)
        """
        majority = _infer_content_majority(zone_map)
        if majority is None:
            return None

        mapping = {
            "code": "SAAS_DOCS_WITH_CODE",
            "table": "REST_API_JSON",
            "prose": "BLOG_POST",
            "list": "FORUM_THREAD",
        }
        return mapping.get(majority)

    # ─────────────────────────────────────────────────────────────────────────
    # BUS HANDLERS
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_clean_signal(self, event: CleanSignalEvent) -> None:
        """
        Receives clean extracted content from signal_kernel/.
        Spawned as background task. Never blocks query().

        Process:
            1. Check if ZoneMap exists for (event.domain, event.topology_class).
            2. Run _l3_fresh_parse with event content to get fresh ZoneMap.
            3. Compare fresh vs stored using Jaccard similarity on selectors.
            4. If Jaccard > 0.85: increment confirmation, boost confidence.
            5. If Jaccard ≤ 0.85: store fresh ZoneMap, emit update event.
        """
        domain = getattr(event, "domain", "")
        topology_class = getattr(event, "topology_class", "GENERIC_HTML")
        content = getattr(event, "content", None)
        content_type = getattr(event, "content_type", "html")

        if not domain or content is None:
            return

        try:
            fresh_map = await self._l3_fresh_parse(
                topology_class=topology_class,
                domain=domain,
                intent_vector=None,
                raw_content=content if isinstance(content, bytes) else content.encode("utf-8"),
                content_type=content_type,
            )

            if isinstance(fresh_map, EmptyZoneMap):
                return

            stored_map = self._l1_lookup(domain, topology_class)
            if stored_map is None:
                stored_map = self._get_stored_zone_map(domain, topology_class)

            if stored_map is None or isinstance(stored_map, EmptyZoneMap):
                self._l1_store(domain, topology_class, fresh_map)
                return

            fresh_selectors = _compute_selector_set(fresh_map)
            stored_selectors = _compute_selector_set(stored_map)
            similarity = _jaccard_similarity(fresh_selectors, stored_selectors)

            tracker = self._zone_confirmations.setdefault(
                (domain, topology_class),
                ZoneConfirmationTracker(),
            )

            if similarity > JACCARD_SIMILARITY_THRESHOLD:
                threshold_reached = tracker.confirm(similarity) # noqa

                if (
                        tracker.confirmations >= DISCOVERY_CONFIRMATION_THRESHOLD
                        and stored_map.confidence < CONFIDENCE_MAX - CONFIDENCE_BOOST_PER_CONFIRMATION
                ):
                    new_confidence = min(
                        stored_map.confidence + CONFIDENCE_BOOST_PER_CONFIRMATION,
                        CONFIDENCE_MAX,
                    )
                    updated = dataclasses.replace(stored_map, confidence=new_confidence)
                    self._l1_store(domain, topology_class, updated)

                    if not self._shutdown_requested:
                        asyncio.create_task(
                            self._write_zone_knowledge(domain, topology_class, updated)
                        )

                    log.debug(
                        "zone_confirmed",
                        domain=domain,
                        topology_class=topology_class,
                        jaccard=round(similarity, 4),
                        confirmations=tracker.confirmations,
                        new_confidence=round(new_confidence, 4),
                    )
            else:
                tracker.reset()
                self._l1_store(domain, topology_class, fresh_map)

                if not self._shutdown_requested:
                    asyncio.create_task(
                        self._write_zone_knowledge(domain, topology_class, fresh_map)
                    )

                try:
                    await self._zone_map_updated_emitter.emit(ZoneMapUpdatedEvent(
                        topology_class=topology_class,
                        new_zone_map=fresh_map,
                    ))
                except Exception as e:
                    log.warning("zone_update_emit_failed", error=str(e))

                log.info(
                    "zone_structural_drift",
                    domain=domain,
                    topology_class=topology_class,
                    jaccard=round(similarity, 4),
                    old_confidence=round(stored_map.confidence, 4),
                    new_confidence=round(fresh_map.confidence, 4),
                )

        except Exception as e:
            log.error(
                "on_clean_signal_failed",
                domain=domain,
                topology_class=topology_class,
                error=str(e),
                traceback=traceback.format_exc(),
            )

    async def _on_surprise(self, event: SurpriseEvent) -> None:
        """
        Receives surprise signal from surprise_detector.py.
        Spawned as background task.

        If dissolve_triggered:
            Invalidate ALL L1 cache entries for this topology class.
            Evict L2 cache entry.
            Set confidence = 0.0 for all stored ZoneMaps of this class.
            Emit ZoneMapInvalidatedEvent.

        If not dissolve_triggered (partial surprise):
            Compute decay factor: 1.0 - 0.05 × surprise_score.
            Apply to all ZoneMaps of this topology class.
            Floor at 0.30. Do NOT evict cache.

        Topology class isolation:
            A surprise on NEWS_ARTICLE never touches SAAS_DOCS ZoneMaps.
        """
        topology_class = getattr(event, "topology_class", "")
        dissolve_triggered = getattr(event, "dissolve_triggered", False)
        surprise_score = getattr(event, "surprise_score", 0.0)

        if not topology_class:
            return

        try:
            if dissolve_triggered:
                evicted = self._l1_evict_by_topology_class(topology_class)
                self._l2_evict(topology_class)

                if isinstance(self._zone_knowledge, dict):
                    keys_to_update = [
                        k for k in self._zone_knowledge
                        if k[1] == topology_class
                    ]
                    for key in keys_to_update:
                        zm = self._zone_knowledge[key]
                        if hasattr(zm, "confidence"):
                            self._zone_knowledge[key] = dataclasses.replace(
                                zm, confidence=0.0
                            )

                for key in list(self._zone_confirmations.keys()):
                    if key[1] == topology_class:
                        self._zone_confirmations[key].reset()

                try:
                    await self._zone_map_invalidated_emitter.emit(ZoneMapInvalidatedEvent(
                        topology_class=topology_class,
                    ))
                except Exception as e:
                    log.warning("invalidation_emit_failed", error=str(e))

                log.info(
                    "surprise_dissolve",
                    topology_class=topology_class,
                    l1_evicted=evicted,
                )

            else:
                if isinstance(self._zone_knowledge, dict):
                    keys_to_decay = [
                        k for k in self._zone_knowledge
                        if k[1] == topology_class
                    ]
                    for key in keys_to_decay:
                        zm = self._zone_knowledge[key]
                        if hasattr(zm, "confidence"):
                            new_conf = _exponential_decay(
                                confidence=zm.confidence,
                                surprise_score=surprise_score,
                            )
                            self._zone_knowledge[key] = dataclasses.replace(
                                zm, confidence=new_conf
                            )

                log.info(
                    "surprise_partial_decay",
                    topology_class=topology_class,
                    surprise_score=round(surprise_score, 4),
                    decay_factor=round(
                        max(0.0, 1.0 - SURPRISE_DECAY_RATE * surprise_score), 4
                    ),
                )

        except Exception as e:
            log.error(
                "on_surprise_failed",
                topology_class=topology_class,
                error=str(e),
                traceback=traceback.format_exc(),
            )

    async def _on_watchdog_event(self, path: str) -> None:
        """
        Watchdog callback when structural_layer.pt changes.
        Reloads zone knowledge and cached router version.
        """
        if Path(path).name != STRUCTURAL_LAYER_PATH.name:
            return

        log.info("watchdog_structural_layer_changed", path=path)

        try:
            await self._reload_zone_knowledge()
            self._cached_router_version = self._read_router_version_from_store()
        except Exception as e:
            log.error(
                "watchdog_reload_failed",
                path=path,
                error=str(e),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # STORE OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    async def _reload_zone_knowledge(self) -> None:
        """
        Load zone knowledge from structural_layer.pt.

        If the file does not exist: zone_knowledge remains EmptyZoneKnowledge.
        If the file exists: load zone_knowledge dict.
        Also reload the cached router version.
        """
        if not STRUCTURAL_LAYER_PATH.exists():
            log.warning(
                "structural_layer_not_found",
                path=str(STRUCTURAL_LAYER_PATH),
            )
            self._zone_knowledge = EmptyZoneKnowledge()
            return

        try:
            data = torch.load(STRUCTURAL_LAYER_PATH, weights_only=False)
            zk = data.get("zone_knowledge", {})
            if isinstance(zk, dict):
                self._zone_knowledge = zk
            else:
                self._zone_knowledge = EmptyZoneKnowledge()

            self._cached_router_version = data.get(
                TOPOLOGY_ROUTER_VERSION_KEY, 0
            )

            count = len(zk) if isinstance(zk, dict) else 0
            log.info(
                "zone_knowledge_loaded",
                count=count,
                router_version=self._cached_router_version,
            )

        except Exception as e:
            log.error(
                "zone_knowledge_load_failed",
                path=str(STRUCTURAL_LAYER_PATH),
                error=str(e),
            )
            self._zone_knowledge = EmptyZoneKnowledge()

    def _load_model(self) -> None:
        """
        Load LatentParser weights from structural_layer.pt or checkpoint.

        If structural_layer.pt exists and contains model weights: load them.
        If checkpoint exists: load from checkpoint.
        If neither: create untrained model with default config.
        """
        config = self._config or WLPConfig()

        if STRUCTURAL_LAYER_PATH.exists():
            try:
                data = torch.load(STRUCTURAL_LAYER_PATH, weights_only=False)
                if "wlp_model_state_dict" in data:
                    saved_config = data.get("wlp_config", {})
                    if saved_config:
                        config = WLPConfig.from_dict(saved_config)
                    self._config = config
                    self._model = LatentParser(config)
                    self._model.load_state_dict(data["wlp_model_state_dict"])
                    self._untrained_model = False
                    log.info("wlp_model_loaded", source="structural_layer.pt")
                    return
            except Exception as e:
                log.warning(
                    "wlp_model_load_from_structural_layer_failed",
                    error=str(e),
                )

        if WLP_MODEL_CHECKPOINT_PATH.exists():
            try:
                checkpoint = torch.load(WLP_MODEL_CHECKPOINT_PATH, weights_only=False)
                saved_config = checkpoint.get("config", {})
                if saved_config:
                    config = WLPConfig.from_dict(saved_config)
                self._config = config
                self._model = LatentParser(config)
                self._model.load_state_dict(checkpoint["model_state_dict"])
                self._untrained_model = False
                log.info("wlp_model_loaded", source="wlp_model_checkpoint.pt")
                return
            except Exception as e:
                log.warning(
                    "wlp_model_load_from_checkpoint_failed",
                    error=str(e),
                )

        self._config = config
        self._model = LatentParser(config)
        self._untrained_model = True
        log.warning(
            "wlp_model_untrained",
            msg="structural_layer.pt not found — using untrained model. "
                "ZoneMaps will be low confidence until first preparse cycle.",
        )

    def _validate_model(self) -> None:
        """
        Validate loaded model architecture against config.

        Checks:
            - Model input dimension matches config.node_feature_dim
            - Model hidden dimension matches config.hidden_dim
            - Model prototype dimension matches config.prototype_dim
            - All four heads are present
        """
        if self._model is None:
            return

        config = self._config or WLPConfig()

        assert self._model.conv1.in_channels == config.node_feature_dim, (
            f"conv1.in_channels={self._model.conv1.in_channels} != "
            f"config.node_feature_dim={config.node_feature_dim}"
        )

        cls_out = self._model.cls_head[-1].out_features
        assert cls_out == config.num_node_classes, (
            f"cls_head output={cls_out} != num_node_classes={config.num_node_classes}"
        )

        proto_out = self._model.proto_head.out_features
        assert proto_out == config.prototype_dim, (
            f"proto_head output={proto_out} != prototype_dim={config.prototype_dim}"
        )

    async def _write_zone_knowledge(
            self,
            domain: str,
            topology_class: str,
            zone_map: Union[ZoneMap, EmptyZoneMap],
    ) -> None:
        """
        Atomic write to structural_layer.pt.
        Never writes EmptyZoneMap.

        Protocol:
            1. Acquire write lock (asyncio.Lock).
            2. Load current structural_layer.pt.
            3. Update zone_knowledge section.
            4. Increment version.
            5. Serialize to staging path.
            6. SHA-256 verify staging.
            7. Atomic rename via os.replace().

        If any step fails: log error, do not raise.
        The in-memory ZoneMap is still valid.
        """
        if isinstance(zone_map, EmptyZoneMap):
            return

        if self._shutdown_requested:
            return

        async with self._write_lock:
            try:
                if STRUCTURAL_LAYER_PATH.exists():
                    data = torch.load(STRUCTURAL_LAYER_PATH, weights_only=False)
                else:
                    data = {}

                if "zone_knowledge" not in data:
                    data["zone_knowledge"] = {}
                if "zone_knowledge_version" not in data:
                    data["zone_knowledge_version"] = 0

                data["zone_knowledge_version"] += 1
                version = data["zone_knowledge_version"]
                zone_map = dataclasses.replace(zone_map, version=version)
                data["zone_knowledge"][(domain, topology_class)] = zone_map

                staging_path = STRUCTURAL_LAYER_PATH.with_suffix(".pt.staging")
                torch.save(data, staging_path)

                digest = _sha256_file(staging_path)
                data["staging_sha256"] = digest
                torch.save(data, staging_path)

                os.replace(staging_path, STRUCTURAL_LAYER_PATH)

                if isinstance(self._zone_knowledge, dict):
                    self._zone_knowledge[(domain, topology_class)] = zone_map

                log.debug(
                    "zone_knowledge_written",
                    domain=domain,
                    topology_class=topology_class,
                    version=version,
                    sha256=digest[:16],
                )

            except Exception as e:
                log.error(
                    "zone_knowledge_write_failed",
                    domain=domain,
                    topology_class=topology_class,
                    error=str(e),
                    traceback=traceback.format_exc(),
                )

    def _get_stored_zone_map(
            self, domain: str, topology_class: str
    ) -> Optional[Union[ZoneMap, EmptyZoneMap]]:
        """
        Retrieve a stored zone map from zone_knowledge.
        Returns None if not found.
        """
        if isinstance(self._zone_knowledge, dict):
            return self._zone_knowledge.get((domain, topology_class))
        return self._zone_knowledge.get(domain, topology_class)

    def _current_router_version(self) -> int:
        """
        Return the cached topology router version.

        The version is read from structural_layer.pt metadata during
        initialization and reload. It is NOT read from latent_model.py.
        The dependency boundary is enforced by reading from the shared
        store file.
        """
        return self._cached_router_version

    def _read_router_version_from_store(self) -> int:
        """
        Read the topology router version directly from structural_layer.pt.
        Used by watchdog callback to update cached version.
        """
        if not STRUCTURAL_LAYER_PATH.exists():
            return 0
        try:
            data = torch.load(STRUCTURAL_LAYER_PATH, weights_only=False)
            return data.get(TOPOLOGY_ROUTER_VERSION_KEY, 0)
        except Exception: # noqa
            return self._cached_router_version

    # ─────────────────────────────────────────────────────────────────────────
    # COLD START WARMUP
    # ─────────────────────────────────────────────────────────────────────────

    async def cold_start_warmup(self) -> None:
        """
        Pre-populate L2 cache for all 18 topology classes.
        Called during initialize(), before parser_ready = True.

        Algorithm:
            For each topology_class in L2_WARMUP_PRIORITY:
                Find all ZoneMaps in zone_knowledge for this class.
                Select the highest-confidence ZoneMap.
                Store in L2 cache.

        Timing target: complete in < 200ms.
        """
        t_start = time.monotonic()
        classes_warmed = 0
        classes_empty = 0

        for topo_class in L2_WARMUP_PRIORITY:
            best_map: Optional[ZoneMap] = None
            best_confidence: float = -1.0

            if isinstance(self._zone_knowledge, dict):
                for key, zm in self._zone_knowledge.items():
                    if key[1] == topo_class and hasattr(zm, "confidence"):
                        if zm.confidence > best_confidence:
                            best_confidence = zm.confidence
                            best_map = zm

            if best_map is not None:
                self._l2_cache[topo_class] = best_map
                classes_warmed += 1
            else:
                classes_empty += 1

        elapsed_ms = (time.monotonic() - t_start) * 1000
        log.info(
            "wlp_l2_warmup_complete",
            classes_warmed=classes_warmed,
            classes_empty=classes_empty,
            elapsed_ms=round(elapsed_ms, 1),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING INTERFACE FACTORY
    # ─────────────────────────────────────────────────────────────────────────

    def get_training_interface(self) -> WLPTrainingInterface:
        """
        Factory method for preparse_daemon.py.
        Returns WLPTrainingInterface wrapping the live model.
        """
        if self._model is None:
            raise RuntimeError(
                "Cannot create training interface: model not loaded. "
                "Call initialize() first."
            )
        return WLPTrainingInterface(self._model)

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Returns health status for cold_start.py validation.

        cold_start.py checks:
            health["parser_ready"] == True
            health["model_loaded"] == True
            (health["untrained_model"] is logged as warning, not failure)
        """
        zone_knowledge_count = 0
        if isinstance(self._zone_knowledge, dict):
            zone_knowledge_count = len(self._zone_knowledge)

        l2_classes_present = list(self._l2_cache.keys()) # noqa
        l2_classes_missing = [
            c for c in TOPOLOGY_CLASSES if c not in self._l2_cache
        ]

        return {
            "parser_ready": self._parser_ready,
            "model_loaded": self._model is not None,
            "model_device": str(self._device),
            "zone_knowledge_count": zone_knowledge_count,
            "l1_cache_size": len(self._l1_cache),
            "l2_cache_classes": len(self._l2_cache),
            "l2_classes_missing": l2_classes_missing,
            "bus_subscriptions": list(self._bus_subscriptions),
            "watchdog_registered": self._watchdog_registered,
            "untrained_model": self._untrained_model,
            "cache_statistics": self._stats.snapshot(),
            "config": self._config.to_dict() if self._config else {},
        }

    def _compute_jaccard_similarity( # noqa
            self,
            zone_map_a: Union[ZoneMap, EmptyZoneMap],
            zone_map_b: Union[ZoneMap, EmptyZoneMap],
    ) -> float:
        """
        Compute Jaccard similarity between two zone maps.
        Compares CSS selector sets of signal zones.
        """
        set_a = _compute_selector_set(zone_map_a)
        set_b = _compute_selector_set(zone_map_b)
        return _jaccard_similarity(set_a, set_b)


# ═════════════════════════════════════════════════════════════════════════════
# SPECTRAL ZONE ANALYZER
# ═════════════════════════════════════════════════════════════════════════════


class SpectralZoneAnalyzer:
    """
    Spectral analysis of zone prototype distributions on the unit hypersphere.

    Uses eigenvalue decomposition of the prototype scatter matrix to detect
    cluster structure. The eigenvalue spectrum reveals intrinsic dimensionality:

        Unimodal (coherent zone):  λ_1 >> λ_2 ≥ ... ≥ λ_d
        Bimodal (merged zones):    λ_1 ≈ λ_2 >> λ_3 ≥ ... ≥ λ_d
        Multimodal:                λ_1 ≈ λ_2 ≈ λ_3 >> λ_4

    The eigenvalue ratio λ_1 / λ_2 is the primary bimodality diagnostic.
    """

    @staticmethod
    def eigenvalue_spectrum(
            prototypes: Tensor,
            max_components: int = 10,
    ) -> Tuple[Tensor, Tensor, float]:
        """
        Compute the eigenvalue spectrum of the prototype scatter matrix.

        Via SVD: Z_c = U Σ V^T, eigenvalues λ_i = σ_i² / n.

        Args:
            prototypes: (n, d) prototype vectors.
            max_components: Maximum eigenvalues to return.

        Returns:
            eigenvalues: (k,) in descending order.
            eigenvectors: (k, d) corresponding.
            explained_ratio: Fraction of total variance in top-k.
        """
        n, d = prototypes.shape
        if n < 2:
            return (
                torch.zeros(1, device=prototypes.device),
                torch.zeros(1, d, device=prototypes.device),
                1.0,
            )

        centroid = prototypes.mean(dim=0, keepdim=True)
        centered = prototypes - centroid
        k = min(max_components, min(n, d))

        try:
            U, S, Vh = torch.linalg.svd(centered, full_matrices=False)
            eigenvalues = (S[:k] ** 2) / n
            eigenvectors = Vh[:k]
            total_variance = (S ** 2).sum().item() / n
            explained = eigenvalues.sum().item()
            explained_ratio = explained / max(total_variance, EPS)
            return eigenvalues, eigenvectors, explained_ratio
        except _LinAlgError:
            return (
                torch.zeros(k, device=prototypes.device),
                torch.zeros(k, d, device=prototypes.device),
                0.0,
            )

    @staticmethod
    def eigenvalue_gap_ratio(eigenvalues: Tensor) -> float:
        """
        Compute λ_1 / λ_2. High ratio → unimodal. Low ratio → bimodal.
        """
        if eigenvalues.shape[0] < 2:
            return float("inf")
        lam_1 = eigenvalues[0].item()
        lam_2 = eigenvalues[1].item()
        if lam_2 < EPS:
            return float("inf")
        return lam_1 / lam_2

    @staticmethod
    def effective_dimensionality(eigenvalues: Tensor) -> float:
        """
        Participation ratio: PR = (Σ λ_i)² / Σ λ_i².
        PR = k for k equal eigenvalues. PR → 1 for one dominant eigenvalue.
        """
        if eigenvalues.shape[0] < 1:
            return 1.0
        positive = eigenvalues[eigenvalues > EPS]
        if positive.shape[0] < 1:
            return 1.0
        sum_lam = positive.sum().item()
        sum_lam2 = (positive ** 2).sum().item()
        if sum_lam2 < EPS:
            return 1.0
        return (sum_lam ** 2) / sum_lam2

    @staticmethod
    def detect_bimodality(
            prototypes: Tensor,
            gap_ratio_threshold: float = 3.0,
            pr_threshold: float = 1.5,
    ) -> Tuple[bool, Dict[str, float]]:
        """
        Combined bimodality detection: eigenvalue gap ratio AND participation ratio.
        Both conditions must hold to declare bimodality, preventing false positives
        from simply elongated distributions.
        """
        eigenvalues, _, explained = SpectralZoneAnalyzer.eigenvalue_spectrum(
            prototypes, max_components=5
        )
        gap_ratio = SpectralZoneAnalyzer.eigenvalue_gap_ratio(eigenvalues)
        pr = SpectralZoneAnalyzer.effective_dimensionality(eigenvalues)
        variance = _prototype_variance(prototypes)
        is_bimodal = (gap_ratio < gap_ratio_threshold) and (pr > pr_threshold)
        diagnostics = {
            "eigenvalue_gap_ratio": round(gap_ratio, 4),
            "participation_ratio": round(pr, 4),
            "prototype_variance": round(variance, 6),
            "explained_ratio": round(explained, 4),
            "top_eigenvalue": round(eigenvalues[0].item(), 6) if eigenvalues.shape[0] > 0 else 0.0,
            "is_bimodal": is_bimodal,
        }
        return is_bimodal, diagnostics


# ═════════════════════════════════════════════════════════════════════════════
# CONFIDENCE CALIBRATOR
# ═════════════════════════════════════════════════════════════════════════════


class ConfidenceCalibrator:
    """
    Post-hoc confidence calibration via Platt scaling (Platt 1999).

    Transforms raw sigmoid output into calibrated probabilities:
        P_calibrated = sigmoid(A · confidence + B)

    A, B fitted by logistic regression on validation data.
    Calibration measured by ECE (Expected Calibration Error):
        ECE = Σ_b (|B_b| / N) · |acc(B_b) - conf(B_b)|

    Target: ECE < 0.05 after calibration.
    """

    def __init__(self, n_bins: int = 15) -> None:
        self.n_bins = n_bins
        self.platt_a: float = 1.0
        self.platt_b: float = 0.0
        self.temperature: float = 1.0
        self.calibrated: bool = False
        self._calibration_ece: float = 1.0

    def fit_platt(
            self,
            confidences: List[float],
            correct: List[bool],
            learning_rate: float = 0.01,
            max_iterations: int = 1000,
    ) -> float:
        """
        Fit Platt scaling parameters A, B via gradient descent on NLL.

        Minimizes: L = -Σ_i [y_i log(σ(Ac_i + B)) + (1-y_i) log(1-σ(Ac_i + B))]

        Returns final NLL loss.
        """
        if len(confidences) < 10:
            return float("inf")

        n = len(confidences)
        A, B = 1.0, 0.0

        total_loss = 0.0

        for _ in range(max_iterations):
            grad_A, grad_B = 0.0, 0.0
            total_loss = 0.0

            for i in range(n):
                z = max(-20.0, min(20.0, A * confidences[i] + B))
                p = 1.0 / (1.0 + math.exp(-z))
                p = max(LOG_SAFE_MIN, min(1.0 - LOG_SAFE_MIN, p))
                y = 1.0 if correct[i] else 0.0
                total_loss -= y * math.log(p) + (1.0 - y) * math.log(1.0 - p)
                error = p - y
                grad_A += error * confidences[i]
                grad_B += error

            grad_A /= n
            grad_B /= n
            A -= learning_rate * grad_A
            B -= learning_rate * grad_B

            if abs(grad_A) < 1e-8 and abs(grad_B) < 1e-8:
                break

        self.platt_a = A
        self.platt_b = B
        self.calibrated = True
        calibrated = [self.calibrate(c) for c in confidences]
        self._calibration_ece = self.expected_calibration_error(calibrated, correct)
        return total_loss / n

    def calibrate(self, confidence: float) -> float:
        """Apply Platt scaling to raw confidence → calibrated probability."""
        z = max(-20.0, min(20.0, self.platt_a * confidence + self.platt_b))
        return 1.0 / (1.0 + math.exp(-z))

    def expected_calibration_error(
            self, confidences: List[float], correct: List[bool],
    ) -> float:
        """
        ECE = Σ_b (|B_b|/N) · |acc(B_b) - conf(B_b)|.
        ECE ∈ [0,1]. ECE = 0 is perfect calibration.
        """
        if not confidences:
            return 1.0
        n = len(confidences)
        ece = 0.0
        for b in range(self.n_bins):
            lo, hi = b / self.n_bins, (b + 1) / self.n_bins
            indices = [
                i for i in range(n)
                if lo <= confidences[i] < hi or (b == self.n_bins - 1 and confidences[i] == hi)
            ]
            if not indices:
                continue
            bs = len(indices)
            acc = sum(1 for i in indices if correct[i]) / bs
            conf = sum(confidences[i] for i in indices) / bs
            ece += (bs / n) * abs(acc - conf)
        return ece

    def maximum_calibration_error(
            self, confidences: List[float], correct: List[bool],
    ) -> float:
        """MCE = max_b |acc(B_b) - conf(B_b)|. Worst-case bin error."""
        if not confidences:
            return 1.0
        n = len(confidences)
        mce = 0.0
        for b in range(self.n_bins):
            lo, hi = b / self.n_bins, (b + 1) / self.n_bins
            indices = [
                i for i in range(n)
                if lo <= confidences[i] < hi or (b == self.n_bins - 1 and confidences[i] == hi)
            ]
            if not indices:
                continue
            bs = len(indices)
            acc = sum(1 for i in indices if correct[i]) / bs
            conf = sum(confidences[i] for i in indices) / bs
            mce = max(mce, abs(acc - conf))
        return mce

    def diagnostics(self) -> Dict[str, Any]:
        """Return calibration state."""
        return {
            "calibrated": self.calibrated,
            "platt_a": round(self.platt_a, 6),
            "platt_b": round(self.platt_b, 6),
            "temperature": round(self.temperature, 6),
            "ece": round(self._calibration_ece, 6),
        }


# ═════════════════════════════════════════════════════════════════════════════
# PROTOTYPE CLUSTER ANALYZER
# ═════════════════════════════════════════════════════════════════════════════


class PrototypeClusterAnalyzer:
    """
    Advanced clustering quality metrics for zone prototype embeddings.
    Silhouette, Davies-Bouldin, Calinski-Harabasz, Hopkins statistic.
    """

    @staticmethod
    def silhouette_coefficient(prototypes: Tensor, zone_labels: Tensor) -> float:
        """
        Mean silhouette coefficient S ∈ [-1,1].
        s(i) = (b(i) - a(i)) / max(a(i), b(i)) where a=intra-cluster, b=nearest-cluster.
        S ≈ 1: well-clustered. S < 0: misassigned nodes.
        Uses cosine distance on the unit sphere.
        """
        n = prototypes.shape[0]
        if n < 4:
            return 0.0
        unique_zones = torch.unique(zone_labels)
        if unique_zones.shape[0] < 2:
            return 0.0

        max_n = min(n, 2000)
        if n > max_n:
            perm = torch.randperm(n)[:max_n]
            prototypes = prototypes[perm]
            zone_labels = zone_labels[perm]
            n = max_n # noqa

        sim = torch.mm(prototypes, prototypes.t())
        dist = 1.0 - sim
        silhouettes: List[float] = []

        for zone_id in unique_zones:
            mask = zone_labels == zone_id
            indices = mask.nonzero(as_tuple=True)[0]
            if indices.shape[0] < 2:
                continue
            for idx in indices:
                i = idx.item()
                same = dist[i][mask]
                ns = same.shape[0]
                if ns <= 1:
                    continue
                a_i = (same.sum().item() - dist[i][i].item()) / (ns - 1)
                b_i = float("inf")
                for oz in unique_zones:
                    if oz == zone_id:
                        continue
                    om = zone_labels == oz
                    od = dist[i][om]
                    if od.shape[0] == 0:
                        continue
                    b_i = min(b_i, od.mean().item())
                if b_i == float("inf"):
                    continue
                denom = max(a_i, b_i)
                silhouettes.append(0.0 if denom < EPS else (b_i - a_i) / denom)

        return sum(silhouettes) / max(len(silhouettes), 1)

    @staticmethod
    def davies_bouldin_index(prototypes: Tensor, zone_labels: Tensor) -> float:
        """
        DB = (1/k) Σ max_{j≠i} (s_i + s_j) / d(c_i, c_j).
        Lower is better. DB = 0 for perfect clustering.
        """
        unique_zones = torch.unique(zone_labels)
        k = unique_zones.shape[0]
        if k < 2:
            return 0.0

        centroids, spreads = {}, {}
        for zid in unique_zones:
            z = zid.item()
            mask = zone_labels == zid
            zp = prototypes[mask]
            c = F.normalize(zp.mean(dim=0, keepdim=True), p=2, dim=1).squeeze(0)
            centroids[z] = c
            spreads[z] = (1.0 - torch.mv(zp, c)).mean().item()

        ids = [z.item() for z in unique_zones]
        db = 0.0
        for zi in ids:
            mr = 0.0
            for zj in ids:
                if zi == zj:
                    continue
                d = max(1.0 - torch.dot(centroids[zi], centroids[zj]).item(), EPS)
                mr = max(mr, (spreads[zi] + spreads[zj]) / d)
            db += mr
        return db / k

    @staticmethod
    def calinski_harabasz_index(prototypes: Tensor, zone_labels: Tensor) -> float:
        """
        CH = [tr(B_k)/(k-1)] / [tr(W_k)/(n-k)].
        Higher is better. Measures ratio of between-cluster to within-cluster variance.
        """
        n = prototypes.shape[0]
        unique_zones = torch.unique(zone_labels)
        k = unique_zones.shape[0]
        if k < 2 or n <= k:
            return 0.0

        gc = prototypes.mean(dim=0)
        tr_B, tr_W = 0.0, 0.0
        for zid in unique_zones:
            mask = zone_labels == zid
            zp = prototypes[mask]
            ni = zp.shape[0]
            zc = zp.mean(dim=0)
            diff = zc - gc
            tr_B += ni * torch.dot(diff, diff).item()
            tr_W += ((zp - zc.unsqueeze(0)) ** 2).sum().item()

        if tr_W < EPS:
            return float("inf")
        return (tr_B / max(k - 1, 1)) / (tr_W / max(n - k, 1))

    @staticmethod
    def full_diagnostics(prototypes: Tensor, zone_labels: Tensor) -> Dict[str, Any]:
        """Run all metrics and return comprehensive report."""
        n = prototypes.shape[0]
        k = torch.unique(zone_labels).shape[0]
        report: Dict[str, Any] = {"n_nodes": n, "n_zones": k}
        if k >= 2 and n >= 4:
            report["silhouette"] = round(
                PrototypeClusterAnalyzer.silhouette_coefficient(prototypes, zone_labels), 4
            )
            report["davies_bouldin"] = round(
                PrototypeClusterAnalyzer.davies_bouldin_index(prototypes, zone_labels), 4
            )
            report["calinski_harabasz"] = round(
                PrototypeClusterAnalyzer.calinski_harabasz_index(prototypes, zone_labels), 4
            )
        per_zone_var: Dict[int, float] = {}
        for zid in torch.unique(zone_labels):
            mask = zone_labels == zid
            per_zone_var[zid.item()] = round(_prototype_variance(prototypes[mask]), 6)
        report["per_zone_variance"] = per_zone_var
        return report


# ═════════════════════════════════════════════════════════════════════════════
# ZONE MAP DIFF ENGINE
# ═════════════════════════════════════════════════════════════════════════════


class ZoneMapDiffEngine:
    """Structural diff engine for comparing zone maps across time."""

    @staticmethod
    def structural_diff(
            old_map: Union[ZoneMap, EmptyZoneMap],
            new_map: Union[ZoneMap, EmptyZoneMap],
    ) -> Dict[str, Any]:
        """Detailed structural comparison including added/removed/modified zones."""
        old_sel = _compute_selector_set(old_map)
        new_sel = _compute_selector_set(new_map)
        added = new_sel - old_sel
        removed = old_sel - new_sel
        shared = old_sel & new_sel
        jaccard = _jaccard_similarity(old_sel, new_sel)

        old_conf = getattr(old_map, "confidence", 0.0)
        new_conf = getattr(new_map, "confidence", 0.0)

        content_changes: List[Dict[str, str]] = []
        if not isinstance(old_map, EmptyZoneMap) and not isinstance(new_map, EmptyZoneMap):
            old_types = {z.selector: z.content_type for z in old_map.signal_zones}
            new_types = {z.selector: z.content_type for z in new_map.signal_zones}
            for sel in shared:
                ot = old_types.get(sel, "unknown")
                nt = new_types.get(sel, "unknown")
                if ot != nt:
                    content_changes.append({"selector": sel, "old": ot, "new": nt})

        return {
            "jaccard": round(jaccard, 4),
            "added_selectors": sorted(added),
            "removed_selectors": sorted(removed),
            "n_added": len(added),
            "n_removed": len(removed),
            "n_shared": len(shared),
            "confidence_delta": round(new_conf - old_conf, 4),
            "content_type_changes": content_changes,
            "is_structural_drift": jaccard <= JACCARD_SIMILARITY_THRESHOLD,
        }

    @staticmethod
    def zone_stability_score(jaccard_history: List[float], decay: float = 0.95) -> float:
        """
        Exponentially weighted stability score from Jaccard history.
        score = Σ decay^(n-1-i) × jaccard_i / Σ decay^(n-1-i).
        Near 1.0: stable. Near 0.0: volatile.
        """
        if not jaccard_history:
            return 0.0
        n = len(jaccard_history)
        weights = [decay ** (n - 1 - i) for i in range(n)]
        ws = sum(weights)
        if ws < EPS:
            return 0.0
        return sum(w * j for w, j in zip(weights, jaccard_history)) / ws


# ═════════════════════════════════════════════════════════════════════════════
# MODEL DIAGNOSTICS
# ═════════════════════════════════════════════════════════════════════════════


class ModelDiagnostics:
    """Diagnostic utilities for LatentParser model health monitoring."""

    @staticmethod
    def parameter_statistics(model: LatentParser) -> Dict[str, Dict[str, float]]:
        """Per-parameter statistics: mean, std, min, max, l2_norm, n_zeros."""
        stats: Dict[str, Dict[str, float]] = {}
        for name, param in model.named_parameters():
            d = param.data
            stats[name] = {
                "mean": round(d.mean().item(), 6),
                "std": round(d.std().item(), 6),
                "min": round(d.min().item(), 6),
                "max": round(d.max().item(), 6),
                "l2_norm": round(d.norm(2).item(), 6),
                "n_params": d.numel(),
            }
        return stats

    @staticmethod
    def gradient_health(model: LatentParser) -> Dict[str, Any]:
        """
        Check for vanishing/exploding/NaN/Inf gradients.
        Must be called after loss.backward() before optimizer.step().
        """
        norms: List[float] = []
        nan_p: List[str] = []
        inf_p: List[str] = []
        dead_p: List[str] = []

        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            g = param.grad
            n = g.norm(2).item()
            norms.append(n)
            if torch.isnan(g).any():
                nan_p.append(name)
            if torch.isinf(g).any():
                inf_p.append(name)
            if n < 1e-10 and param.numel() > 10:
                dead_p.append(name)

        return {
            "has_nan": len(nan_p) > 0,
            "has_inf": len(inf_p) > 0,
            "vanishing": max(norms, default=0.0) < 1e-7 and len(norms) > 0,
            "exploding": max(norms, default=0.0) > 1e3,
            "dead_params": dead_p,
            "max_grad_norm": round(max(norms, default=0.0), 8),
            "mean_grad_norm": round(sum(norms) / max(len(norms), 1), 8),
        }

    @staticmethod
    def model_summary(model: LatentParser) -> Dict[str, Any]:
        """Comprehensive model summary with parameter counts per layer."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        layers: Dict[str, int] = {}
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, SAGEConv)):
                layers[name] = sum(p.numel() for p in module.parameters())
        return {
            "total_parameters": total,
            "trainable_parameters": trainable,
            "model_size_mb": round(total * 4 / (1024 * 1024), 2),
            "layer_counts": layers,
        }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SINGLETON
# ═════════════════════════════════════════════════════════════════════════════

WLP: WorldLatentParser = WorldLatentParser()
"""
Module-level singleton. Initialized at import time.
Warm before interface.py accepts queries via WLP.initialize().
"""

# ═════════════════════════════════════════════════════════════════════════════
# MODULE EXPORTS
# ═════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Public class
    "WorldLatentParser",
    # Module singleton
    "WLP",
    # Configuration
    "WLPConfig",
    # Model
    "LatentParser",
    "IntentProjection",
    # Training interface
    "WLPTrainingInterface",
    # Output types
    "ForwardOutput",
    "ReadoutOutput",
    # Internal types (exposed for testing)
    "CacheStatistics",
    "ZoneConfirmationTracker",
    # Constants
    "L1_CACHE_MAX_SIZE",
    "COHERENCE_THRESHOLD",
    "JACCARD_SIMILARITY_THRESHOLD",
    "DISCOVERY_CONFIRMATION_THRESHOLD",
    "CONTRASTIVE_TEMPERATURE",
    "CONTRASTIVE_NEGATIVES",
    "STRUCTURAL_LAYER_PATH",
    "TOPOLOGY_ROUTER_VERSION_KEY",
    # Advanced analysis
    "SpectralZoneAnalyzer",
    "ConfidenceCalibrator",
    "PrototypeClusterAnalyzer",
    "ZoneMapDiffEngine",
    "ModelDiagnostics",
    "BatchInferenceEngine",
    "GraphStatisticsAnalyzer",
    "ZoneTopologyClassifier",
]


# ═════════════════════════════════════════════════════════════════════════════
# BATCH INFERENCE ENGINE
#
# Handles batched inference for multiple pages simultaneously.
# Used by preparse_daemon.py during training data generation and
# by cold_start_warmup() for parallel L2 cache population.
# ═════════════════════════════════════════════════════════════════════════════


class BatchInferenceEngine:
    """
    Batched inference wrapper for LatentParser.

    Batches multiple PyG Data objects into a single forward pass for
    GPU utilization efficiency. PyG's Batch.from_data_list() merges
    multiple graphs into a single disconnected graph, preserving
    per-graph node indices via batch vectors.

    Performance analysis:
        Single inference on 10K-node graph: ~15ms.
        10× single inference: ~150ms.
        Batched inference on 10 × 10K-node graphs: ~50ms.
        Speedup: 3× from batch amortization of kernel launch overhead.

    Memory analysis:
        Single 10K-node graph at d=256: ~10MB activations.
        Batch of 10: ~100MB. Fits in 5080 VRAM (16GB) with margin.
        Maximum batch size: dynamically computed based on available VRAM.
    """

    def __init__(
            self,
            model: LatentParser,
            device: torch.device,
            max_batch_nodes: int = 100_000,
    ) -> None:
        self._model = model
        self._device = device
        self._max_batch_nodes = max_batch_nodes

    def compute_optimal_batch_size(
            self,
            data_list: List[Data],
    ) -> int:
        """
        Compute the optimal batch size based on total node count.

        Strategy: greedily add graphs to the batch until the total
        node count exceeds max_batch_nodes. This ensures the batch
        fits in GPU memory while maximizing parallelism.

        Args:
            data_list: List of PyG Data objects.

        Returns:
            Optimal batch size (number of graphs).
        """
        if not data_list:
            return 0

        total_nodes = 0
        batch_size = 0

        for data in data_list:
            n_nodes = data.x.shape[0] if data.x is not None else 0
            if total_nodes + n_nodes > self._max_batch_nodes and batch_size > 0:
                break
            total_nodes += n_nodes
            batch_size += 1

        return max(batch_size, 1)

    @torch.no_grad()
    def batch_readout(
            self,
            data_list: List[Data],
            intent_vectors: Optional[List[Optional[Tensor]]] = None,
    ) -> List[ReadoutOutput]:
        """
        Perform batched inference on multiple graphs.

        Each graph is independently classified. The batch dimension
        is handled by PyG's internal batching mechanism via Batch.

        For simplicity and correctness (different intent vectors per graph),
        this method processes graphs individually but with shared model state.

        Args:
            data_list: List of PyG Data objects.
            intent_vectors: Optional per-graph intent tensors.

        Returns:
            List of ReadoutOutput objects, one per input graph.
        """
        self._model.eval()
        results: List[ReadoutOutput] = []

        for i, data in enumerate(data_list):
            intent = None
            if intent_vectors is not None and i < len(intent_vectors):
                intent = intent_vectors[i]

            data = data.to(self._device)
            result = self._model.readout(data, intent)
            results.append(ReadoutOutput(
                logits=result.logits.cpu(),
                confidences=result.confidences.cpu(),
                prototypes=result.prototypes.cpu(),
            ))

        return results

    def batch_classify(
            self,
            data_list: List[Data],
            intent_vectors: Optional[List[Optional[Tensor]]] = None,
    ) -> List[Tuple[List[int], List[float]]]:
        """
        Perform batched classification, returning labels and confidences.

        Convenience method that chains readout → classify_nodes.

        Returns:
            List of (labels, confidences) tuples per graph.
        """
        readout_results = self.batch_readout(data_list, intent_vectors)

        classifications: List[Tuple[List[int], List[float]]] = []
        for result in readout_results:
            labels, confs = classify_nodes(result.logits, result.confidences)
            classifications.append((labels, confs))

        return classifications

    def estimate_memory_mb(self, n_nodes: int) -> float:
        """
        Estimate GPU memory usage for a batch of given total node count.

        Memory model:
            Feature storage: n × d × 4 bytes (float32)
            Hidden states: 3 layers × n × h × 4 bytes
            Output heads: n × (3 + 3 + 1 + 64) × 4 bytes
            Edge index: ~4n × 2 × 8 bytes (int64, avg degree 4)
            Overhead: ~20%

        Args:
            n_nodes: Total number of nodes in the batch.

        Returns:
            Estimated memory in MB.
        """
        d_in = self._model.config.node_feature_dim
        d_h = self._model.config.hidden_dim
        d_p = self._model.config.prototype_dim

        feature_bytes = n_nodes * d_in * 4
        hidden_bytes = 3 * n_nodes * d_h * 4
        output_bytes = n_nodes * (3 + 3 + 1 + d_p) * 4
        edge_bytes = 4 * n_nodes * 2 * 8
        overhead = 0.20

        total_bytes = (feature_bytes + hidden_bytes + output_bytes + edge_bytes) * (1 + overhead)
        return total_bytes / (1024 * 1024)


# ═════════════════════════════════════════════════════════════════════════════
# GRAPH STATISTICS ANALYZER
#
# Provides structural analysis of DOM graphs produced by wlp_graph.py.
# Used for monitoring graph quality and detecting anomalous page structures.
# ═════════════════════════════════════════════════════════════════════════════


class GraphStatisticsAnalyzer:
    """
    Structural statistics for PyG Data objects produced by wlp_graph.py.

    Computes graph-theoretic properties that inform model behavior:
        - Node and edge counts
        - Degree distribution statistics
        - Graph density and connectivity
        - Feature dimension validation
        - Edge type distribution
    """

    @staticmethod
    def basic_statistics(data: Data) -> Dict[str, Any]:
        """
        Compute basic graph statistics.

        Returns:
            n_nodes, n_edges, density, avg_degree, max_degree, etc.
        """
        n_nodes = data.x.shape[0] if data.x is not None else 0
        n_edges = data.edge_index.shape[1] if (
                data.edge_index is not None and data.edge_index.dim() > 1
        ) else 0
        feature_dim = data.x.shape[1] if data.x is not None and data.x.dim() > 1 else 0

        max_possible_edges = n_nodes * (n_nodes - 1)
        density = n_edges / max(max_possible_edges, 1)

        degrees: List[int] = [0] * n_nodes
        if data.edge_index is not None and n_edges > 0:
            edge_idx = data.edge_index
            for j in range(n_edges):
                src = edge_idx[0, j].item()
                if src < n_nodes:
                    degrees[src] += 1

        avg_degree = sum(degrees) / max(n_nodes, 1)
        max_degree = max(degrees) if degrees else 0
        min_degree = min(degrees) if degrees else 0

        isolated_nodes = sum(1 for d in degrees if d == 0)

        degree_variance = 0.0
        if n_nodes > 1:
            degree_variance = sum((d - avg_degree) ** 2 for d in degrees) / (n_nodes - 1)

        return {
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "feature_dim": feature_dim,
            "density": round(density, 6),
            "avg_degree": round(avg_degree, 2),
            "max_degree": max_degree,
            "min_degree": min_degree,
            "degree_std": round(math.sqrt(degree_variance), 2),
            "isolated_nodes": isolated_nodes,
            "isolated_fraction": round(isolated_nodes / max(n_nodes, 1), 4),
        }

    @staticmethod
    def feature_statistics(data: Data) -> Dict[str, Any]:
        """
        Analyze feature vector distributions.

        Checks for:
            - NaN or Inf values (data corruption)
            - Zero vectors (missing features)
            - Feature magnitude distribution
            - Intent bias presence (dimensions 100:128)
        """
        if data.x is None:
            return {"error": "no_features"}

        x = data.x
        n, d = x.shape

        has_nan = bool(torch.isnan(x).any().item())
        has_inf = bool(torch.isinf(x).any().item())

        zero_rows = (x.abs().sum(dim=1) < EPS).sum().item()
        all_zero_fraction = zero_rows / max(n, 1)

        feature_means = x.mean(dim=0)
        feature_stds = x.std(dim=0)

        intent_slice = x[:, 100:128] if d >= 128 else None
        intent_active = False
        if intent_slice is not None:
            intent_active = bool(intent_slice.abs().sum().item() > EPS)

        topology_slice = x[:, 0:18] if d >= 18 else None
        topology_entropy = 0.0
        if topology_slice is not None:
            col_sums = topology_slice.sum(dim=0)
            col_probs = col_sums / max(col_sums.sum().item(), EPS)
            for p in col_probs.tolist():
                if p > 0:
                    topology_entropy -= p * math.log2(max(p, LOG_SAFE_MIN))

        return {
            "n_nodes": n,
            "feature_dim": d,
            "has_nan": has_nan,
            "has_inf": has_inf,
            "zero_vector_fraction": round(all_zero_fraction, 4),
            "feature_mean_min": round(feature_means.min().item(), 6),
            "feature_mean_max": round(feature_means.max().item(), 6),
            "feature_std_min": round(feature_stds.min().item(), 6),
            "feature_std_max": round(feature_stds.max().item(), 6),
            "intent_active": intent_active,
            "topology_entropy": round(topology_entropy, 4),
        }

    @staticmethod
    def edge_type_distribution(data: Data) -> Dict[str, Any]:
        """
        Analyze edge type distribution.

        Edge types from wlp_graph.py:
            PARENT_CHILD  [1, 0, 0]
            SIBLING       [0, 1, 0]
            SKIP_SIBLING  [0, 0, 1]

        Returns counts and fractions per edge type.
        """
        if data.edge_attr is None:
            return {"error": "no_edge_attributes"}

        attr = data.edge_attr
        n_edges = attr.shape[0]

        if n_edges == 0:
            return {"n_edges": 0}

        parent_child = (attr[:, 0] > 0.5).sum().item()
        sibling = (attr[:, 1] > 0.5).sum().item()
        skip_sibling = (attr[:, 2] > 0.5).sum().item()
        untyped = n_edges - parent_child - sibling - skip_sibling

        return {
            "n_edges": n_edges,
            "parent_child": parent_child,
            "sibling": sibling,
            "skip_sibling": skip_sibling,
            "untyped": untyped,
            "parent_child_frac": round(parent_child / max(n_edges, 1), 4),
            "sibling_frac": round(sibling / max(n_edges, 1), 4),
            "skip_sibling_frac": round(skip_sibling / max(n_edges, 1), 4),
        }

    @staticmethod
    def anomaly_detection(data: Data) -> Dict[str, Any]:
        """
        Detect anomalous graph structures that may indicate upstream issues.

        Anomaly indicators:
            - Very high edge density (> 0.1): unusually connected DOM.
            - Very low edge density (< 0.001): disconnected fragments.
            - High isolated node fraction (> 0.2): broken tree structure.
            - NaN/Inf in features: data corruption.
            - Zero feature rows (> 50%): feature extraction failure.
            - Extreme degree variance: inconsistent tree structure.
        """
        basic = GraphStatisticsAnalyzer.basic_statistics(data)
        feat = GraphStatisticsAnalyzer.feature_statistics(data)

        anomalies: List[str] = []

        if basic["density"] > 0.1:
            anomalies.append("high_density")
        if basic["density"] < 0.001 and basic["n_nodes"] > 10:
            anomalies.append("low_density")
        if basic["isolated_fraction"] > 0.2:
            anomalies.append("high_isolation")
        if feat.get("has_nan", False):
            anomalies.append("nan_features")
        if feat.get("has_inf", False):
            anomalies.append("inf_features")
        if feat.get("zero_vector_fraction", 0) > 0.5:
            anomalies.append("excessive_zero_features")
        if basic["max_degree"] > 500:
            anomalies.append("extreme_fan_out")
        if basic["degree_std"] > 50:
            anomalies.append("extreme_degree_variance")

        return {
            "has_anomalies": len(anomalies) > 0,
            "anomalies": anomalies,
            "n_anomalies": len(anomalies),
            "basic_stats": basic,
            "feature_stats": feat,
        }


# ═════════════════════════════════════════════════════════════════════════════
# ZONE TOPOLOGY CLASSIFIER
#
# Lightweight topology class inference from zone structure alone.
# Used by discover_signal_zones() to suggest subclass candidates.
# ═════════════════════════════════════════════════════════════════════════════


class ZoneTopologyClassifier:
    """
    Classify topology from zone structure using rule-based heuristics.

    This is NOT the main topology classifier (which uses ML).
    This is a lightweight fallback that infers topology class from
    the structural properties of a ZoneMap without requiring the
    full topology classification model.

    Classification rules (priority order):

    1. REST_API_JSON: ≥3 zones with content_type="table" AND
       extraction_strategy=SECTION_SCOPED.

    2. SAAS_DOCS_WITH_CODE: ≥2 zones with content_type="code" AND
       ≥1 zone with content_type="prose".

    3. SAAS_DOCS: ≥1 zone with content_type="prose" AND
       any zones with content_type="code".

    4. ECOMMERCE_PRODUCT: ≥2 zones with content_type="table" AND
       extraction_strategy=BREADTH_FIRST.

    5. FORUM_THREAD: ≥3 zones with content_type="list".

    6. BLOG_POST: 1-2 zones with content_type="prose" AND
       no "code" or "table" zones.

    7. NEWS_ARTICLE: 1 zone with content_type="prose" AND
       high density (> 0.7).

    8. GENERIC_HTML: default fallback.
    """

    @staticmethod
    def classify(zone_map: Union[ZoneMap, EmptyZoneMap]) -> str:
        """
        Infer topology class from zone structure.

        Returns a topology class string.
        """
        if isinstance(zone_map, EmptyZoneMap):
            return "GENERIC_HTML"

        if not zone_map.signal_zones:
            return "GENERIC_HTML"

        type_counts: Dict[str, int] = defaultdict(int)
        total_density = 0.0
        n_zones = len(zone_map.signal_zones)

        for z in zone_map.signal_zones:
            type_counts[z.content_type] += 1
            total_density += z.density

        avg_density = total_density / max(n_zones, 1)
        strategy = zone_map.extraction_strategy

        n_code = type_counts.get("code", 0)
        n_prose = type_counts.get("prose", 0)
        n_table = type_counts.get("table", 0)
        n_list = type_counts.get("list", 0)

        if n_table >= 3 and strategy == ExtractionStrategy.SECTION_SCOPED:
            return "REST_API_JSON"

        if n_code >= 2 and n_prose >= 1:
            return "SAAS_DOCS_WITH_CODE"

        if n_prose >= 1 and n_code >= 1:
            return "SAAS_DOCS"

        if n_table >= 2 and strategy == ExtractionStrategy.BREADTH_FIRST:
            return "ECOMMERCE_PRODUCT"

        if n_list >= 3:
            return "FORUM_THREAD"

        if n_prose >= 1 and n_prose <= 2 and n_code == 0 and n_table == 0: # noqa
            if avg_density > 0.7 and n_zones <= 2:
                return "NEWS_ARTICLE"
            return "BLOG_POST"

        return "GENERIC_HTML"

    @staticmethod
    def confidence_for_classification(
            zone_map: Union[ZoneMap, EmptyZoneMap],
            classified_as: str,
    ) -> float:
        """
        Estimate confidence in the topology classification.

        Based on how strongly the zone structure matches the expected
        pattern for the classified topology class.

        Returns confidence in [0.0, 1.0].
        """
        if isinstance(zone_map, EmptyZoneMap):
            return 0.0

        type_counts: Dict[str, int] = defaultdict(int)
        for z in zone_map.signal_zones:
            type_counts[z.content_type] += 1

        n_zones = len(zone_map.signal_zones)
        n_code = type_counts.get("code", 0)
        n_prose = type_counts.get("prose", 0)
        n_table = type_counts.get("table", 0)
        n_list = type_counts.get("list", 0)

        if classified_as == "REST_API_JSON":
            return min(0.9, n_table / max(n_zones, 1) + 0.2)

        if classified_as == "SAAS_DOCS_WITH_CODE":
            code_frac = n_code / max(n_zones, 1)
            prose_frac = n_prose / max(n_zones, 1)
            return min(0.9, (code_frac + prose_frac) / 2 + 0.3)

        if classified_as == "SAAS_DOCS":
            return min(0.85, (n_prose + n_code) / max(n_zones, 1) + 0.2)

        if classified_as == "ECOMMERCE_PRODUCT":
            return min(0.85, n_table / max(n_zones, 1) + 0.2)

        if classified_as == "FORUM_THREAD":
            return min(0.85, n_list / max(n_zones, 1) + 0.1)

        if classified_as in ("BLOG_POST", "NEWS_ARTICLE"):
            return min(0.80, n_prose / max(n_zones, 1) + 0.2)

        return 0.50

    @staticmethod
    def all_candidates(
            zone_map: Union[ZoneMap, EmptyZoneMap],
    ) -> List[Tuple[str, float]]:
        """
        Return all candidate topology classes with confidence scores.

        Useful for discovering zones where multiple topology classes
        are plausible and additional signals are needed.

        Returns: List of (class_name, confidence) sorted by confidence desc.
        """
        candidates: List[Tuple[str, float]] = []

        for topo_class in TOPOLOGY_CLASSES:
            conf = ZoneTopologyClassifier.confidence_for_classification(
                zone_map, topo_class
            )
            if conf > 0.1:
                candidates.append((topo_class, conf))

        candidates.sort(key=lambda x: x[1], reverse=True)
        return candidates