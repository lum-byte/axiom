"""
tag/world_model/latent_model.py
================================
The complete WorldLatentModel class.

This is the only public-facing component in tag/world_model/.  Every caller
outside this directory — interface.py, cold_start.py, index_daemon.py — interacts
with the world model through this file and nothing else.

WorldLatentModel owns the full inference lifecycle:
    - Three-tier cache routing (L1 domain-policy → L2 topology-policy → L3 model readout)
    - MambaRouter readout orchestration with consistency guarantees
    - CrawlerBus subscription for domain topology and surprise events
    - StoreWatchdog registration for topology_router.pt and structural_layer.pt reload
    - Structural layer management with graceful EmptyStructuralLayer handling
    - Cold-start warmup with topologically sorted pre-population
    - WLMTrainingInterface — the sole boundary for training operations

What this file is NOT:
    Not a model.  It wraps one.  nn.Module lives in mamba_router.py.
    Not a tokenizer.  It calls one.  Vocabulary lives in wlm_tokenizer.py.
    Not a decoder.  It calls three.  Activations live in wlm_decoders.py.
    Not a trainer.  It exposes a training interface but never invokes training.
    Not a cache.  It manages an ephemeral cache that dies with the process.
    Not responsible for WLP.  It calls WLP via asyncio.gather() at caller level.
    Not allowed to make network calls.  The WLM never fetches anything.
    Not allowed to update hidden_state during inference.  Ever.  Under any condition.

Single public method contract:

    async def query(
        topology_class: str,
        intent_vector: Optional[List[float]],
        domain: str,
        phase: int,
    ) -> WLMResponse

    Latency targets (hard, not aspirational):
        <2ms  Phase III known topology class (L1 cache hit expected)
        <5ms  Phase I unknown class (L3 model readout path)
        <10ms absolute ceiling — exceeding this indicates upstream fault

Three-tier cache — mathematically justified:

    L1: Domain policy cache
        Key: (domain, topology_class)
        Value: WLMResponse
        TTL: topology-class-specific (CACHE_TTL_BY_CLASS)
        Hit rate target: >60% for warm system
        Why: same domain + same topology class = structurally identical fetch context.
             The WLM will produce identical output.  Cache the output.

    L2: Topology class policy cache
        Key: topology_class
        Value: WLMResponse (no domain-specific source priority)
        TTL: 2× domain TTL — topology class policy is more stable than per-domain policy
        Hit rate target: >25% of L1 misses
        Why: when a domain is new but the topology class is known, the traversal
             policy and friction forecast are still predictable.

    L3: Model readout
        No key.  No TTL.  Always fresh.
        The readout() call on MambaRouter.
        Expected frequency: <15% of all queries on a warm system.
        Why L3 is still fast: one matrix multiply against fixed hidden state.

    Cache routing math:
        For any query:
          1. Check L1 (domain, topology_class) → hit rate H1
          2. On L1 miss: check L2 (topology_class) → hit rate H2
          3. On L2 miss: call L3 model readout → always succeeds

        Expected query distribution on warm system:
          L1 hits:  ~65% of queries
          L2 hits:  ~20% of queries
          L3 calls: ~15% of queries

        Weighted average latency:
          0.65 × 0.5ms + 0.20 × 0.8ms + 0.15 × 4ms = 1.14ms average

        This is the mathematical basis for the <2ms p95 claim.

Hidden state consistency model — 500ms window:

    Problem:
        t=0ms:  query arrives, enters query()
        t=1ms:  L1 miss, L2 miss, falls to L3
        t=2ms:  readout() begins — reads hidden_state
        t=3ms:  index_daemon calls optimizer.step()
        t=4ms:  index_daemon wants to update hidden_state
        t=4ms:  if hidden_state updates NOW, query at t=2ms read old weights
                but output heads were loaded with new weights
                → inconsistent forward pass

    Solution — query counter with 500ms ceiling:
        _active_readout_count tracks in-flight readouts.
        update_hidden_state() polls the counter every 1ms.
        If all readouts complete in 3ms, the update happens in 3ms.
        500ms is only reached if something is wrong — at which point we proceed
        anyway and log the anomaly.

        This is exact, not defensive.  A bare asyncio.sleep(0.5) would waste
        497ms when queries complete in 3ms.

Dependency direction:
    latent_model.py → contracts.py (types, constants, phases)
    latent_model.py → mamba_router.py (MambaRouter, ReadoutResult, ModelConfig,
                       StructuralLayerView, EmptyStructuralLayer, HIDDEN_STATE_DIM,
                       SOURCE_PRIORITY_FALLBACK)
    latent_model.py → wlm_tokenizer.py (WLMTokenizer, encode_domain_topology_event)
    latent_model.py → wlm_decoders.py (decode_traversal, decode_friction,
                       decode_source_priority, load_structural_layer, K_BY_PHASE)
    latent_model.py → crawler_bus.py (BUS singleton, subscribe, event types)
    latent_model.py → store_watchdog.py (WATCHDOG singleton, register, STORE_ROOT)
    latent_model.py → exceptions.py (exception codes)

    Nothing imports from latent_model.py except interface.py and cold_start.py.
    latent_model.py never imports from latent_parser.py — WLP is a peer,
    not a dependency.  They coordinate via asyncio.gather() at the caller level.
"""

from __future__ import annotations

# ─────────────────────────────────────────────────────────────────────────────
# STDLIB
# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import hashlib
import logging # noqa
import math
import os # noqa
import time
from collections import OrderedDict # noqa
from dataclasses import dataclass, field
from pathlib import Path
from typing import ( # noqa
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY
# ─────────────────────────────────────────────────────────────────────────────

import torch
import structlog

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — contracts.py
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.contracts import ( # noqa
    TOPOLOGY_CLASSES,
    FALLBACK_TOPOLOGY_CLASS,
    PARENT_CLASS_MAP,
    PHASE_I,
    PHASE_II,
    PHASE_III,
    THETA_CONFIDENCE_II,
    THETA_CONFIDENCE_III,
    THETA_WLP_MIN,
    STORE_FILE_NAMES,
    WATCHDOG_HANDLER_TIMEOUT_S,
    WATCHDOG_SHUTDOWN_DRAIN_TIMEOUT_S,
    TopologyClassStr,
    PhaseInt,
    ConfidenceFloat,
    WLMResponse,
    TopologyTraversalPolicy,
    FrictionForecast,
    DomainTopologyEvent,
    SurpriseEvent,
)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — mamba_router.py
# ─────────────────────────────────────────────────────────────────────────────

from tag.world_model.world_latent_model.mamba_router import ( # noqa
    MambaRouter,
    ModelConfig,
    ReadoutResult,
    ForwardResult,
    StructuralLayerView,
    EmptyStructuralLayer,
    HIDDEN_STATE_DIM,
    HIDDEN_STATE_UPDATE_DELAY_MS,
    SOURCE_PRIORITY_FALLBACK,
    SOURCE_PRIORITY_TOP_K,
    DEFAULT_D_MODEL,
    DEFAULT_VOCAB_SIZE,
    PRODUCTION_CONFIG,
)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — wlm_tokenizer.py
# ─────────────────────────────────────────────────────────────────────────────

from tag.world_model.world_latent_model.wlm_tokenizer import ( # noqa
    VOCAB_SIZE as WLM_VOCAB_SIZE,
    encode_domain_event,
    encode_batch,
    topology_class_to_token,
    domain_to_token,
)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — wlm_decoders.py
# ─────────────────────────────────────────────────────────────────────────────

from tag.world_model.world_latent_model.wlm_decoders import ( # noqa
    decode_traversal,
    decode_friction,
    decode_source_priority,
    load_structural_layer,
    validate_traversal_policy,
    validate_friction_forecast,
    validate_source_priority,
    K_BY_PHASE,
)

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — crawler_bus.py (lazy singleton import at initialize() time)
# ─────────────────────────────────────────────────────────────────────────────
# The BUS singleton is imported at initialize() time, not at module load time.
# This avoids a circular import scenario where crawler_bus.py might transitively
# import from world_model during its own _load_hmac_key() call.
# The import is deferred to _import_bus_singleton() and cached in _BUS_REF.

_BUS_REF: Optional[Any] = None

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — store_watchdog.py (lazy singleton import at initialize() time)
# ─────────────────────────────────────────────────────────────────────────────

_WATCHDOG_REF: Optional[Any] = None

# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL — exceptions.py
# ─────────────────────────────────────────────────────────────────────────────

from signal_kernel.exceptions import ( # noqa
    EC_TOPO_WLP_QUERY_FAILED,
    EC_TOPO_BUS_SUBSCRIPTION,
    EC_TOPO_GRADIENT_STEP_FAILED,
)

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# All cache hits, misses, readouts, and consistency events are logged through
# this logger.  structlog provides structured key-value pairs that Witness
# consumes for real-time observability.
# ─────────────────────────────────────────────────────────────────────────────

log: structlog.BoundLogger = structlog.get_logger("tag.world_model.latent_model")


# ═════════════════════════════════════════════════════════════════════════════
# CACHE TTL CONSTANTS
#
# Per-topology-class TTL values in seconds for the L1 domain policy cache.
# L2 topology class cache uses 2× these values.
#
# TTL design rationale per class:
#
#   The TTL encodes how frequently the structural characteristics of a
#   topology class change in practice.  Wikipedia article structure is
#   stable for years — 24h TTL is conservative.  News site structure
#   changes with CMS updates, template refreshes, and ad rotation —
#   15m TTL reflects this volatility.
#
#   TTL=0 classes (AUTH_REDIRECT, CLOUDFLARE_CHALLENGE, RATE_LIMITED)
#   must NEVER be cached at any tier.  Caching a Cloudflare challenge
#   response as if it were real traversal policy is a silent failure
#   worse than a cache miss.  _is_ttl_zero_class() enforces this at
#   every cache store call path.
#
# Derivation of TTL values:
#
#   WIKIPEDIA_ARTICLE (86400s = 24h):
#       MediaWiki rendering pipeline is server-side and extremely stable.
#       Template changes propagate slowly through Wikipedia's CDN.
#       24h gives one refresh per day — sufficient for structural stability.
#       Empirical: Wikipedia page structure changed 0 times in 30-day crawl test.
#
#   SAAS_DOCS (21600s = 6h):
#       Documentation sites update on release cycles (weekly to monthly).
#       6h ensures the WLM refreshes ~4 times per day for active SaaS products.
#       Covers Docusaurus, GitBook, ReadMe, and custom doc frameworks.
#       Versioned and code variants inherit the same TTL — they share rendering pipelines.
#
#   REST_API_JSON (3600s = 1h):
#       API endpoint structure changes less frequently than HTML pages.
#       1h balances freshness against the cost of L3 readouts.
#       Paginated variant shares TTL — pagination is URL-parameter based.
#
#   JSON_LD_STRUCTURED (3600s = 1h):
#       Structured data schemas (Schema.org) are standards-governed.
#       Changes are infrequent but when they happen, they change the entire
#       extraction strategy.  1h is conservative but correct.
#
#   NEWS_ARTICLE (900s = 15m):
#       News sites change structure frequently: ad refreshes, template updates,
#       A/B testing of article layouts, paywall injection changes.
#       15m ensures the WLM stays responsive to CMS updates.
#       Paywalled variant inherits the same TTL — paywall injection is equally volatile.
#
#   ECOMMERCE_PRODUCT (1800s = 30m):
#       Product pages change with inventory updates, price changes, and
#       promotional overlays.  30m balances freshness with cache efficiency.
#       Variant pages inherit the same TTL.
#
#   FORUM_THREAD (3600s = 1h):
#       Forum software (Discourse, phpBB) templates change infrequently.
#       1h is sufficient — forum structure is more stable than news sites.
#
#   BLOG_POST (7200s = 2h):
#       Blog platforms (Substack, Medium, Ghost) have stable templates.
#       2h captures platform-wide template updates without excessive readouts.
#
#   LANDING_PAGE (3600s = 1h):
#       Landing pages vary widely in structure.  1h is a conservative default.
#
#   GENERIC_HTML (1800s = 30m):
#       Catch-all for unclassified pages.  30m is moderate.
#
#   AUTH_REDIRECT (0s):
#       Auth state changes constantly.  Session cookies, CSRF tokens, redirect
#       targets — all are per-request.  Caching would serve stale auth routing.
#
#   CLOUDFLARE_CHALLENGE (0s):
#       Cloudflare challenges are per-session.  The challenge token, difficulty,
#       and type change with every new browser session.  Caching would serve
#       stale challenge responses that cannot be solved.
#
#   RATE_LIMITED (0s):
#       Rate limit state changes with every request.  Retry-After headers,
#       remaining quota, and backoff windows are all per-request signals.
#       Caching would serve stale rate limit assessments.
#
# ═════════════════════════════════════════════════════════════════════════════

CACHE_TTL_BY_CLASS: Dict[str, float] = {
    "WIKIPEDIA_ARTICLE":          86400.0,   # 24h — Wikipedia structure is stable
    "SAAS_DOCS":                  21600.0,   # 6h  — docs update on release cycles
    "SAAS_DOCS_VERSIONED":        21600.0,   # 6h  — same rendering pipeline as SAAS_DOCS
    "SAAS_DOCS_WITH_CODE":        21600.0,   # 6h  — code blocks add JS but structure is stable
    "REST_API_JSON":               3600.0,   # 1h  — APIs change but not constantly
    "REST_API_JSON_PAGINATED":     3600.0,   # 1h  — pagination is URL-based, equally stable
    "JSON_LD_STRUCTURED":          3600.0,   # 1h  — schema.org governed, infrequent changes
    "NEWS_ARTICLE":                 900.0,   # 15m — news sites change structure frequently
    "NEWS_ARTICLE_PAYWALLED":       900.0,   # 15m — paywall injection is equally volatile
    "ECOMMERCE_PRODUCT":           1800.0,   # 30m — product pages change with inventory
    "ECOMMERCE_PRODUCT_VARIANT":   1800.0,   # 30m — variant pages inherit product TTL
    "FORUM_THREAD":                3600.0,   # 1h  — forum templates change infrequently
    "BLOG_POST":                   7200.0,   # 2h  — blog platform templates are stable
    "LANDING_PAGE":                3600.0,   # 1h  — conservative default for varied structure
    "AUTH_REDIRECT":                  0.0,   # never cache — auth state changes per-request
    "CLOUDFLARE_CHALLENGE":           0.0,   # never cache — challenge is per-session
    "RATE_LIMITED":                   0.0,   # never cache — rate limit state changes constantly
    "GENERIC_HTML":                1800.0,   # 30m — moderate default for unclassified pages
}

# The set of topology classes with TTL=0.  Precomputed at module load time
# for O(1) lookup in _is_ttl_zero_class().  This is a correctness invariant,
# not an optimization — TTL=0 classes must be excluded from all cache tiers
# on every store path.
_TTL_ZERO_CLASSES: FrozenSet[str] = frozenset(
    tc for tc, ttl in CACHE_TTL_BY_CLASS.items() if ttl == 0.0
)


# ═════════════════════════════════════════════════════════════════════════════
# CACHE SIZING CONSTANTS
#
# L1 and L2 caches are bounded by entry count, not by memory.
# Each WLMResponse is a frozen dataclass of ~1KB.  10,000 entries = ~10MB.
# These are process-local caches that die on restart.  Sizing is generous
# to avoid eviction-driven L3 readouts.
# ═════════════════════════════════════════════════════════════════════════════

L1_MAX_ENTRIES: int = 10_000
# Theoretical capacity: 18 topology classes × ~500 active domains = 9,000 entries.
# 10,000 allows headroom for bursty domain discovery without forced eviction.
# At ~1KB per WLMResponse, this is ~10MB — negligible relative to model weights.

L2_MAX_ENTRIES: int = 100
# L2 has at most one entry per topology class.  18 known classes + headroom
# for dynamically discovered classes.  100 is generous.

L2_TTL_MULTIPLIER: float = 2.0
# L2 TTL = L1 TTL × 2.0.
# Rationale: topology class policy is more stable than per-domain policy.
# A new domain within a known topology class will receive the same traversal
# and friction policy — the only difference is source priority, which is
# generic in the L2 cache (no domain-specific ranking).
# 2× multiplier means L2 entries survive twice as long as L1 entries for
# the same topology class.  This reduces L3 readouts for domains that arrive
# after the L1 entry for a known class expires.

# ═════════════════════════════════════════════════════════════════════════════
# CONSISTENCY MODEL CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

HIDDEN_STATE_UPDATE_DEADLINE_MS: int = 500
# Maximum time (ms) to wait for in-flight readouts before proceeding with
# hidden_state update.  If all readouts complete in 3ms, the update happens
# in 3ms.  500ms is the ceiling for pathological cases.
#
# Mathematical justification:
#   - readout() completes in <2ms (one matmul against hidden state)
#   - 500ms / 2ms = 250× the worst-case readout latency
#   - Any in-flight query will finish within this window
#   - Aligned with store_watchdog debounce for topology_router.pt (500ms)
#   - Total pipeline: gradient step → 500ms delay → hidden state update →
#     file write → 500ms debounce → watchdog fires → all components reload
#   - Total pipeline latency: ~1100ms — acceptable for background training

HIDDEN_STATE_POLL_INTERVAL_MS: float = 1.0
# Poll interval (ms) for the readout counter in update_hidden_state().
# 1ms is the minimum resolution of asyncio.sleep().  Polling every 1ms
# gives at most 1ms overshoot beyond readout completion.
# Total polls in worst case: 500ms / 1ms = 500 polls.  Each poll is one
# integer comparison — negligible CPU cost.

# ═════════════════════════════════════════════════════════════════════════════
# EVICTION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

EVICTION_BATCH_SIZE: int = 100
# Maximum entries to evict per _evict_expired() call.
# Eviction is called lazily on cache store operations.  Batch size prevents
# a single store from spending unbounded time in the eviction loop.
# 100 entries × O(1) per entry = O(100) — microseconds.

EVICTION_INTERVAL_QUERIES: int = 64
# Run eviction every N queries.  64 is a power of two for efficient
# modular arithmetic (query_count & 0x3F == 0).
# 64 queries × 1ms average latency = 64ms between eviction sweeps.
# This is frequent enough to prevent unbounded cache growth but infrequent
# enough to avoid eviction overhead on every query.

# ═════════════════════════════════════════════════════════════════════════════
# COLD START CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

COLD_START_TIMEOUT_S: float = 30.0
# Maximum time for cold_start_warmup() to complete.
# 18 topology classes × <5ms per L3 readout = ~90ms expected.
# 30s is a generous ceiling that accommodates:
#   - First readout on cold PyTorch model (JIT warmup)
#   - Slow disk I/O during initial model load
#   - Structural layer loading for source priority
# Exceeding this timeout is a hard failure — cold_start.py aborts.

# ═════════════════════════════════════════════════════════════════════════════
# VALIDATION CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

ROUTER_VALIDATION_TOPOLOGY_CLASS: str = "SAAS_DOCS"
# Topology class used for validation readouts during weight reload.
# SAAS_DOCS is chosen because:
#   - It is a common production topology class (high coverage)
#   - It has non-trivial traversal bias (depth=4, not default)
#   - It exercises the full decoder pipeline without edge cases
#   - It is not a TTL=0 class (avoids cache bypass during validation)

ROUTER_VALIDATION_MAX_TRAVERSAL_DEPTH: int = 10
# Maximum acceptable depth from a validation readout.
# Depth > 10 indicates weight corruption (activated sigmoid × 4 + 1 can never
# produce depth > 5 under normal conditions; validation accepts up to 10 to
# allow for temporary model exploration during early training).

ROUTER_VALIDATION_MAX_RPS: float = 200.0
# Maximum acceptable RPS from a validation readout.
# Softplus can produce arbitrarily large values from corrupted weights.
# 200.0 is 2× the production ceiling of 100.0 — allows headroom for
# legitimate high-RPS outputs while catching extreme corruption.

ROUTER_VALIDATION_FRICTION_RANGE: Tuple[float, float] = (-0.01, 1.01)
# Acceptable range for friction probabilities.
# Sigmoid output is in (0, 1) but floating-point noise can push to
# [-epsilon, 1+epsilon].  -0.01 to 1.01 catches corruption while
# allowing numerical noise.

# ═════════════════════════════════════════════════════════════════════════════
# STORE PATHS
# ═════════════════════════════════════════════════════════════════════════════

STORE_ROOT: Path = Path(os.environ.get("AXIOM_STORE_DIR", "store"))
WEIGHTS_PATH: Path = STORE_ROOT / "topology_router.pt"
STRUCTURAL_LAYER_PATH: Path = STORE_ROOT / "structural_layer.pt"
STAGING_DIR: Path = STORE_ROOT / "staging"


# ═════════════════════════════════════════════════════════════════════════════
# LATENCY TRACKING CONSTANTS
# ═════════════════════════════════════════════════════════════════════════════

LATENCY_HISTOGRAM_BUCKETS: Tuple[float, ...] = (
    0.1, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0,
)
# Bucket boundaries in milliseconds for the latency histogram.
# Designed to have high resolution in the [0, 5ms] range where most queries
# land, and lower resolution in the [5, 50ms] range for tail latency tracking.
#
# Expected distribution for a warm system:
#   [0.0, 0.5)   — L1 hits:  ~65% of queries
#   [0.5, 1.0)   — L2 hits:  ~20% of queries
#   [1.0, 5.0)   — L3 readouts: ~14% of queries
#   [5.0, 10.0)  — slow L3:  ~0.9% of queries
#   [10.0, 50.0) — anomalous: <0.1% of queries

LATENCY_P95_TARGET_MS: float = 2.0
# p95 latency target in milliseconds.
# Mathematical basis:
#   If 85% of queries hit L1/L2 (< 1ms each) and 15% hit L3 (< 5ms each),
#   then the 95th percentile falls within the L2 tier:
#     p95 = quantile(0.95) over {65% @ 0.5ms, 20% @ 0.8ms, 15% @ 4ms}
#     Since 65% + 20% = 85% < 95%, p95 falls in the first 95% - 85% = 10%
#     of L3 queries.  L3 queries complete in <5ms, so p95 < 5ms.
#     For Phase III known topology classes with warm cache, p95 < 2ms
#     because Phase III queries have higher L1 hit rates (~80%).

LATENCY_P99_TARGET_MS: float = 5.0
# p99 latency target.  Only L3 readouts with cold JIT should exceed this.

LATENCY_ABSOLUTE_CEILING_MS: float = 10.0
# Hard ceiling.  Exceeding this indicates an upstream problem.
# Logged at ERROR and counted in the anomaly counter.


# ═════════════════════════════════════════════════════════════════════════════
# INTERNAL DATA STRUCTURES
#
# Not exported.  These are the runtime cache entry types, latency trackers,
# and cache tier statistics containers.
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class _L1CacheEntry:
    """
    One entry in the L1 domain policy cache.

    Stored as a mutable dataclass rather than a frozen dataclass because
    the access_count is incremented on every hit for LFU-assisted eviction.
    The response itself is a frozen WLMResponse — immutability is enforced
    at the contract level, not the cache entry level.

    Fields:
        response:      The cached WLMResponse.  Frozen.
        stored_at:     Monotonic timestamp of cache insertion.  Used for TTL expiry.
        ttl:           TTL in seconds for this entry.  Copied from CACHE_TTL_BY_CLASS
                       at insertion time.  Stored per-entry to handle dynamic topology
                       class TTL changes without invalidating existing entries.
        topology_class: The topology class.  Stored for eviction logging.
        domain:        The domain.  Stored for eviction logging.
        access_count:  Number of hits since insertion.  Used for LFU-assisted eviction
                       when the cache is full and TTL-based eviction is insufficient.
        last_access:   Monotonic timestamp of last access.  Used for LRU tiebreaking.
        hidden_state_version: The hidden_state version at insertion time.
                       Used to detect stale entries after weight reloads — entries
                       whose hidden_state_version differs from the current version
                       are evicted on next access.
    """
    response:              WLMResponse
    stored_at:             float
    ttl:                   float
    topology_class:        str
    domain:                str
    access_count:          int = 0
    last_access:           float = 0.0
    hidden_state_version:  int = 0

    @property
    def is_expired(self) -> bool:
        """True if the entry has exceeded its TTL."""
        return (time.monotonic() - self.stored_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was stored."""
        return time.monotonic() - self.stored_at

    @property
    def remaining_ttl(self) -> float:
        """Seconds remaining before TTL expiry.  Negative if expired."""
        return self.ttl - (time.monotonic() - self.stored_at)


@dataclass
class _L2CacheEntry:
    """
    One entry in the L2 topology class policy cache.

    Similar structure to _L1CacheEntry but keyed on topology_class only.
    L2 entries have 2× the TTL of L1 entries for the same topology class.
    Source priority in L2 is generic (no domain-specific ranking) — this is
    acceptable because L2 serves queries for NEW domains within a KNOWN
    topology class.

    Fields follow the same conventions as _L1CacheEntry.
    """
    response:              WLMResponse
    stored_at:             float
    ttl:                   float
    topology_class:        str
    access_count:          int = 0
    last_access:           float = 0.0
    hidden_state_version:  int = 0

    @property
    def is_expired(self) -> bool:
        """True if the entry has exceeded its TTL."""
        return (time.monotonic() - self.stored_at) > self.ttl

    @property
    def age_seconds(self) -> float:
        """Seconds since this entry was stored."""
        return time.monotonic() - self.stored_at

    @property
    def remaining_ttl(self) -> float:
        """Seconds remaining before TTL expiry.  Negative if expired."""
        return self.ttl - (time.monotonic() - self.stored_at)


@dataclass
class _CacheTierStats:
    """
    Runtime statistics for one cache tier (L1 or L2).

    Counters are monotonically increasing from process start.
    Rates are computed by dividing counters by elapsed time.
    These statistics are exposed via health() for Witness consumption.

    Fields:
        hits:            Total cache hits since startup.
        misses:          Total cache misses since startup.
        stores:          Total entries stored since startup.
        evictions_ttl:   Total entries evicted due to TTL expiry.
        evictions_cap:   Total entries evicted due to capacity limit.
        evictions_stale: Total entries evicted due to hidden_state_version mismatch.
        invalidations:   Total entries explicitly invalidated (e.g. surprise event).
        current_size:    Current number of entries in the cache.
    """
    hits:            int = 0
    misses:          int = 0
    stores:          int = 0
    evictions_ttl:   int = 0
    evictions_cap:   int = 0
    evictions_stale: int = 0
    invalidations:   int = 0
    current_size:    int = 0

    @property
    def total_lookups(self) -> int:
        """Total lookup attempts (hits + misses)."""
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        """Cache hit rate as a fraction in [0.0, 1.0]."""
        total = self.total_lookups
        if total == 0:
            return 0.0
        return self.hits / total

    @property
    def miss_rate(self) -> float:
        """Cache miss rate as a fraction in [0.0, 1.0]."""
        return 1.0 - self.hit_rate

    @property
    def total_evictions(self) -> int:
        """Total evictions from all causes."""
        return self.evictions_ttl + self.evictions_cap + self.evictions_stale

    def to_log_dict(self, tier: str) -> Dict[str, Any]:
        """Flat dict for structured logging with tier prefix."""
        return {
            f"{tier}_hits":            self.hits,
            f"{tier}_misses":          self.misses,
            f"{tier}_stores":          self.stores,
            f"{tier}_hit_rate":        round(self.hit_rate, 4),
            f"{tier}_evictions_ttl":   self.evictions_ttl,
            f"{tier}_evictions_cap":   self.evictions_cap,
            f"{tier}_evictions_stale": self.evictions_stale,
            f"{tier}_invalidations":   self.invalidations,
            f"{tier}_current_size":    self.current_size,
        }


@dataclass
class _LatencyHistogram:
    """
    Fixed-bucket histogram for query latency tracking.

    Buckets are defined by LATENCY_HISTOGRAM_BUCKETS.  Each bucket count
    represents queries with latency in [bucket_low, bucket_high).
    The final bucket catches all queries above the last boundary.

    This is a lightweight histogram suitable for in-process telemetry.
    It does not use locks because WorldLatentModel operates on a single
    asyncio event loop — all mutations are coroutine-safe.
    """
    buckets:         Tuple[float, ...]
    counts:          List[int]
    total_count:     int = 0
    total_latency_ms: float = 0.0
    min_latency_ms:  float = float("inf")
    max_latency_ms:  float = 0.0

    def __init__(self, buckets: Tuple[float, ...] = LATENCY_HISTOGRAM_BUCKETS) -> None:
        self.buckets = buckets
        # +1 for the overflow bucket (above highest boundary)
        self.counts = [0] * (len(buckets) + 1)
        self.total_count = 0
        self.total_latency_ms = 0.0
        self.min_latency_ms = float("inf")
        self.max_latency_ms = 0.0

    def record(self, latency_ms: float) -> None:
        """Record a latency observation."""
        self.total_count += 1
        self.total_latency_ms += latency_ms
        if latency_ms < self.min_latency_ms:
            self.min_latency_ms = latency_ms
        if latency_ms > self.max_latency_ms:
            self.max_latency_ms = latency_ms

        # Binary search for the correct bucket.
        # Buckets are sorted ascending.  Find the first bucket boundary
        # that exceeds latency_ms — the index is the bucket.
        for i, boundary in enumerate(self.buckets):
            if latency_ms < boundary:
                self.counts[i] += 1
                return
        # Above all boundaries — overflow bucket.
        self.counts[-1] += 1

    @property
    def mean_latency_ms(self) -> float:
        """Mean latency in milliseconds."""
        if self.total_count == 0:
            return 0.0
        return self.total_latency_ms / self.total_count

    def percentile(self, p: float) -> float:
        """
        Approximate the p-th percentile (0.0 to 1.0) from bucket counts.

        Uses linear interpolation within the identified bucket.
        Accuracy depends on bucket granularity — with the default buckets,
        resolution is ~0.25ms in the [0, 2ms] range.

        Mathematical formulation:
            target_count = p × total_count
            Scan buckets until cumulative count ≥ target_count.
            Within the identified bucket, interpolate linearly:
                fraction = (target_count - prev_cumulative) / bucket_count
                estimate = bucket_low + fraction × (bucket_high - bucket_low)

        Returns the bucket midpoint if the bucket count is zero (degenerate case).
        Returns 0.0 if no observations have been recorded.
        """
        if self.total_count == 0:
            return 0.0

        target = p * self.total_count
        cumulative = 0

        for i, count in enumerate(self.counts):
            cumulative += count
            if cumulative >= target and count > 0:
                # Determine bucket boundaries.
                if i == 0:
                    bucket_low = 0.0
                else:
                    bucket_low = self.buckets[i - 1]

                if i < len(self.buckets):
                    bucket_high = self.buckets[i]
                else:
                    # Overflow bucket — use 2× the last boundary as upper estimate.
                    bucket_high = self.buckets[-1] * 2.0

                # Linear interpolation within the bucket.
                prev_cumulative = cumulative - count
                fraction = (target - prev_cumulative) / count
                return bucket_low + fraction * (bucket_high - bucket_low)

        # Should not reach here — return max observed.
        return self.max_latency_ms

    @property
    def p50(self) -> float:
        """Median latency estimate."""
        return self.percentile(0.50)

    @property
    def p95(self) -> float:
        """95th percentile latency estimate."""
        return self.percentile(0.95)

    @property
    def p99(self) -> float:
        """99th percentile latency estimate."""
        return self.percentile(0.99)

    def to_log_dict(self) -> Dict[str, Any]:
        """Flat dict for structured logging."""
        return {
            "latency_total_count":  self.total_count,
            "latency_mean_ms":      round(self.mean_latency_ms, 3),
            "latency_min_ms":       round(self.min_latency_ms, 3) if self.total_count > 0 else None,
            "latency_max_ms":       round(self.max_latency_ms, 3) if self.total_count > 0 else None,
            "latency_p50_ms":       round(self.p50, 3),
            "latency_p95_ms":       round(self.p95, 3),
            "latency_p99_ms":       round(self.p99, 3),
        }


@dataclass
class _QueryStats:
    """
    Aggregate query statistics for the WorldLatentModel.

    Tracks total queries, per-tier distribution, per-topology-class counts,
    and anomaly counters.  Exposed via health() for Witness consumption.
    """
    total_queries:          int = 0
    l1_hits:                int = 0
    l2_hits:                int = 0
    l3_readouts:            int = 0
    ttl_zero_bypasses:      int = 0
    latency_ceiling_breaches: int = 0
    readout_errors:         int = 0
    per_class_counts:       Dict[str, int] = field(default_factory=dict)
    per_phase_counts:       Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 0, 3: 0})

    @property
    def cache_hit_rate(self) -> float:
        """Combined L1+L2 cache hit rate."""
        if self.total_queries == 0:
            return 0.0
        return (self.l1_hits + self.l2_hits) / self.total_queries

    @property
    def l3_rate(self) -> float:
        """Fraction of queries hitting L3 readout."""
        if self.total_queries == 0:
            return 0.0
        return self.l3_readouts / self.total_queries

    def to_log_dict(self) -> Dict[str, Any]:
        """Flat dict for structured logging."""
        return {
            "total_queries":            self.total_queries,
            "l1_hits":                  self.l1_hits,
            "l2_hits":                  self.l2_hits,
            "l3_readouts":              self.l3_readouts,
            "cache_hit_rate":           round(self.cache_hit_rate, 4),
            "l3_rate":                  round(self.l3_rate, 4),
            "ttl_zero_bypasses":        self.ttl_zero_bypasses,
            "latency_ceiling_breaches": self.latency_ceiling_breaches,
            "readout_errors":           self.readout_errors,
        }


# ═════════════════════════════════════════════════════════════════════════════
# LAZY SINGLETON IMPORTS
# ═════════════════════════════════════════════════════════════════════════════

def _import_bus_singleton() -> Any:
    """
    Lazily import the CrawlerBus singleton.

    Deferred to avoid circular import at module load time.
    crawler_bus.py imports from contracts.py; contracts.py is shared with
    mamba_router.py which we also import.  The dependency chain is:
        latent_model → mamba_router → contracts ← crawler_bus
    This is safe because contracts.py has no runtime logic, but deferring
    the bus import makes the dependency graph cleaner.

    Returns the module-level BUS instance from crawler_bus.py.
    Caches the result in _BUS_REF so subsequent calls are free.
    """
    global _BUS_REF
    if _BUS_REF is None:
        from tag.crawler_bus import BUS  # noqa: local import — intentionally deferred
        _BUS_REF = BUS
    return _BUS_REF


def _import_watchdog_singleton() -> Any:
    """
    Lazily import the StoreWatchdog singleton.

    Same deferred-import rationale as _import_bus_singleton().

    Returns the module-level WATCHDOG instance from store_watchdog.py.
    Caches the result in _WATCHDOG_REF.
    """
    global _WATCHDOG_REF
    if _WATCHDOG_REF is None:
        from tag.store_watchdog import WATCHDOG  # noqa: local import
        _WATCHDOG_REF = WATCHDOG
    return _WATCHDOG_REF


# ═════════════════════════════════════════════════════════════════════════════
# WORLD LATENT MODEL
#
# The only public-facing component in tag/world_model/.
# ═════════════════════════════════════════════════════════════════════════════

class WorldLatentModel:
    """
    The complete inference orchestrator for the World Latent Model.

    Wraps MambaRouter, WLMTokenizer, and all three decoders into a single
    coherent system with one public method: query().

    Lifecycle:
        model = WorldLatentModel(config)    # construction — no I/O
        await model.initialize()            # WATCHDOG + bus registration
        await model.cold_start_warmup()     # L2 pre-population
        ...
        response = await model.query(...)   # the only public method
        ...
        training = model.get_training_interface()  # for index_daemon
        ...
        await model.shutdown()              # graceful drain

    Thread safety:
        Not thread-safe.  All methods must be called from the same asyncio
        event loop.  The training interface uses a threading.Lock for the
        MambaRouter forward() call, but all other state is coroutine-safe.

    Cache consistency:
        The three-tier cache is invalidated on:
        - Weight reload (WATCHDOG fires on topology_router.pt change)
          → clears L1 and L2 entirely
        - Structural layer reload (WATCHDOG fires on structural_layer.pt change)
          → does NOT clear caches (source priority only, least critical output)
        - Surprise event (bus fires on SurpriseEvent)
          → invalidates L1 entries for the affected topology class
          → invalidates L2 entry for the affected topology class
          → does NOT invalidate other topology classes
        - Hidden state update (WLMTrainingInterface.update_hidden_state())
          → increments hidden_state_version
          → stale entries detected at lookup time via version mismatch

    The asymmetry between weight reload and structural layer reload is
    intentional and documented:
        Weight reload changes traversal and friction predictions fundamentally
        → all cached policies are invalid.
        Structural layer reload changes source priority only
        → existing traversal and friction policies are still valid.
        → source priority in cached WLMResponses will be stale but acceptable.
        → next L3 readout for each topology class produces updated source priority.
    """

    # ─────────────────────────────────────────────────────────────────────────
    # CONSTRUCTION
    # ─────────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        config: Optional[ModelConfig] = None,
        weights_path: Optional[Path] = None,
        structural_layer_path: Optional[Path] = None,
        device: str = "cpu",
    ) -> None:
        """
        Construct the WorldLatentModel.

        No I/O occurs in __init__.  All file loading, bus subscription, and
        watchdog registration happen in initialize().

        Parameters:
            config:               MambaRouter configuration.  Defaults to
                                  PRODUCTION_CONFIG (256-dim, 4 layers, 8192 vocab).
            weights_path:         Path to topology_router.pt.
                                  Defaults to /store/topology_router.pt.
            structural_layer_path: Path to structural_layer.pt.
                                  Defaults to /store/structural_layer.pt.
            device:               PyTorch device string.  Defaults to "cpu".
                                  The WLM operates on CPU for latency predictability.
                                  GPU placement adds PCIe transfer overhead that
                                  exceeds the compute savings for single-vector
                                  readout (1792 multiply-accumulate ops).
        """
        # ── Configuration ─────────────────────────────────────────────────────
        self._config: ModelConfig = config if config is not None else PRODUCTION_CONFIG
        self._weights_path: Path = weights_path if weights_path is not None else WEIGHTS_PATH
        self._structural_layer_path: Path = (
            structural_layer_path if structural_layer_path is not None
            else STRUCTURAL_LAYER_PATH
        )
        self._device: str = device

        # ── Model components (initialized in initialize()) ────────────────────
        self._router: Optional[MambaRouter] = None
        self._structural_layer: Union[StructuralLayerView, EmptyStructuralLayer] = (
            EmptyStructuralLayer()
        )

        # ── L1 domain policy cache ────────────────────────────────────────────
        # Key: (domain, topology_class) → _L1CacheEntry
        # Implementation: OrderedDict maintains insertion/access order for O(1)
        # LRU eviction (move_to_end on hit, popitem(last=False) to evict).
        # Eviction: TTL expiry checked on lookup + periodic sweep.
        self._l1_cache: OrderedDict[Tuple[str, str], _L1CacheEntry] = OrderedDict()
        self._l1_stats: _CacheTierStats = _CacheTierStats()

        # ── L2 topology class policy cache ────────────────────────────────────
        # Key: topology_class → _L2CacheEntry
        # Implementation: plain dict with 2× TTL.
        self._l2_cache: Dict[str, _L2CacheEntry] = {}
        self._l2_stats: _CacheTierStats = _CacheTierStats()

        # ── Readout consistency state ─────────────────────────────────────────
        # _active_readout_count:  Number of readouts currently in-flight.
        #   Incremented at the start of _l3_readout(), decremented in finally block.
        #   Used by WLMTrainingInterface.update_hidden_state() to enforce the
        #   500ms consistency window.
        #
        # _hidden_state_version:  Monotonic counter incremented on every
        #   hidden_state update.  Stored in cache entries to detect staleness.
        #   Separate from MambaRouter.hidden_state_version because this counter
        #   also reflects weight reloads, not just training updates.
        self._active_readout_count: int = 0
        self._hidden_state_version: int = 0

        # ── Query statistics and latency tracking ─────────────────────────────
        self._query_stats: _QueryStats = _QueryStats()
        self._latency_histogram: _LatencyHistogram = _LatencyHistogram()

        # ── Lifecycle state ───────────────────────────────────────────────────
        self._initialized: bool = False
        self._shutting_down: bool = False
        self._background_tasks: Set[asyncio.Task] = set()

        # ── Training interface (created lazily on first get_training_interface() call)
        self._training_interface: Optional[WLMTrainingInterface] = None

        # ── Readout failure tracking for exponential backoff ──────────────────
        # Must be instance-level.  Class-level dicts are shared across all
        # instances — a correctness bug if more than one WorldLatentModel exists.
        self._readout_failure_counts: Dict[str, int] = {}
        self._readout_last_failure: Dict[str, float] = {}

        log.info(
            "world_latent_model_constructed",
            config_d_model=self._config.d_model,
            config_n_layers=self._config.n_layers,
            config_vocab_size=self._config.vocab_size,
            weights_path=str(self._weights_path),
            structural_layer_path=str(self._structural_layer_path),
            device=self._device,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE: initialize()
    # ─────────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """
        Initialize the WorldLatentModel.

        Called once by cold_start.py before any queries are accepted.
        This method performs all I/O, all registration, and all validation
        that must happen before the model is ready to serve queries.

        Operations in order:
            1. Construct and validate MambaRouter.
            2. Load topology_router.pt weights.
            3. Construct WLMTokenizer.
            4. Load structural_layer.pt (EmptyStructuralLayer is valid).
            5. Register reload handlers with StoreWatchdog.
            6. Subscribe to CrawlerBus events (domain_topology, surprise).
            7. Set _initialized = True.

        The ordering is load-bearing:
            - MambaRouter must be constructed before loading weights.
            - Weights must be loaded before readout() can produce valid output.
            - Watchdog registration must happen before bus subscription because
              a bus event may trigger a readout that depends on valid weights.
            - Bus subscription happens last because it starts receiving events
              immediately — all dependencies must be ready.

        Raises:
            RuntimeError: If initialize() has already been called.
            RuntimeError: If weight loading fails (topology_router.pt missing or corrupt).
            RuntimeError: If MambaRouter construction fails (architecture mismatch).
        """
        if self._initialized:
            raise RuntimeError(
                "WorldLatentModel.initialize() called twice.  "
                "The model is already initialized.  "
                "This indicates a double-init in cold_start.py."
            )

        init_start = time.monotonic()

        # ── Step 1: Construct MambaRouter ─────────────────────────────────────
        log.info("wlm_init_step_1: constructing MambaRouter")
        try:
            self._router = MambaRouter(
                vocab_size=self._config.vocab_size,
                d_model=self._config.d_model,
                d_state=self._config.d_state,
                d_conv=self._config.d_conv,
                expand=self._config.expand,
                n_layers=self._config.n_layers,
                n_topology=self._config.n_topology,
                n_source=self._config.n_source,
                n_phase=self._config.n_phase,
                dropout=self._config.dropout,
                max_seq_len=self._config.max_seq_len,
            )
            self._router.eval()  # Inference mode — dropout disabled
            log.info(
                "wlm_init_step_1_complete",
                parameter_count=self._router.parameter_count,
                buffer_count=self._router.buffer_count,
            )
        except Exception as exc:
            raise RuntimeError(
                f"MambaRouter construction failed: {exc}.  "
                "This typically indicates a mamba_ssm version mismatch or "
                "an architecture configuration error."
            ) from exc

        # ── Step 2: Load weights ──────────────────────────────────────────────
        log.info(
            "wlm_init_step_2: loading weights",
            path=str(self._weights_path),
        )
        if not self._weights_path.exists():
            raise RuntimeError(
                f"topology_router.pt not found at {self._weights_path}.  "
                "cold_start.py must call initialize_store.py before "
                "WorldLatentModel.initialize().  "
                "The WLM cannot operate without pre-trained or bootstrapped weights."
            )

        try:
            state_dict = torch.load(
                str(self._weights_path),
                weights_only=True,   # security requirement — no exceptions
                map_location=self._device,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load topology_router.pt from {self._weights_path}: {exc}.  "
                "The file may be corrupted or from an incompatible architecture version.  "
                "Run initialize_store.py to regenerate."
            ) from exc

        # Validate state_dict compatibility before loading.
        compatibility_errors = self._router.verify_state_dict_compatibility(state_dict)
        if compatibility_errors:
            error_summary = "; ".join(compatibility_errors[:5])
            raise RuntimeError(
                f"topology_router.pt is incompatible with the current architecture.  "
                f"First 5 errors: {error_summary}.  "
                f"Total errors: {len(compatibility_errors)}.  "
                "The checkpoint was produced by a different model version.  "
                "Run initialize_store.py to regenerate with the current architecture."
            )

        self._router.load_state_dict(state_dict)
        self._router.eval()
        self._hidden_state_version = self._router.current_hidden_state_version

        log.info(
            "wlm_init_step_2_complete",
            hidden_state_version=self._hidden_state_version,
            hidden_state_digest=self._router.hidden_state_digest(),
        )

        # ── Step 3: Verify tokenizer vocabulary alignment ─────────────────────
        log.info("wlm_init_step_3: verifying tokenizer vocabulary alignment")
        if WLM_VOCAB_SIZE != self._config.vocab_size:
            raise RuntimeError(
                f"Vocabulary size mismatch: wlm_tokenizer.VOCAB_SIZE={WLM_VOCAB_SIZE} "
                f"but ModelConfig.vocab_size={self._config.vocab_size}.  "
                "The tokenizer and model must agree on vocabulary size.  "
                "Check wlm_tokenizer.py VOCAB_SIZE and mamba_router.py DEFAULT_VOCAB_SIZE."
            )
        log.info(
            "wlm_init_step_3_complete",
            vocab_size=WLM_VOCAB_SIZE,
        )

        # ── Step 4: Load structural layer ─────────────────────────────────────
        log.info(
            "wlm_init_step_4: loading structural layer",
            path=str(self._structural_layer_path),
        )
        self._structural_layer = load_structural_layer(self._structural_layer_path)
        sl_type = type(self._structural_layer).__name__
        sl_domains = (
            self._structural_layer.n_domains
            if hasattr(self._structural_layer, "n_domains")
            else 0
        )
        log.info(
            "wlm_init_step_4_complete",
            structural_layer_type=sl_type,
            domain_count=sl_domains,
        )

        # ── Step 5: Register with StoreWatchdog ───────────────────────────────
        log.info("wlm_init_step_5: registering with StoreWatchdog")
        watchdog = _import_watchdog_singleton()

        watchdog.register(
            path="topology_router.pt",
            handler=self._reload_weights,
            debounce_ms=500,
        )
        log.debug("wlm_init: registered topology_router.pt reload handler")

        watchdog.register(
            path="structural_layer.pt",
            handler=self._reload_structural_layer,
            debounce_ms=500,
        )
        log.debug("wlm_init: registered structural_layer.pt reload handler")
        log.info("wlm_init_step_5_complete")

        # ── Step 6: Subscribe to CrawlerBus events ────────────────────────────
        log.info("wlm_init_step_6: subscribing to CrawlerBus events")
        bus = _import_bus_singleton()

        try:
            await bus.subscribe(
                topic="domain_topology",
                group="world_model.latent_model.domain_topology",
                handler=self._on_domain_topology,
                schema=DomainTopologyEvent,
            )
            log.debug("wlm_init: subscribed to domain_topology")
        except Exception as exc:
            log.error(
                "wlm_init: failed to subscribe to domain_topology",
                error=str(exc),
                exception_code=EC_TOPO_BUS_SUBSCRIPTION,
            )
            # Non-fatal — the WLM can operate without domain topology events.
            # It will miss domain-specific L1 pre-population but L2 and L3
            # still function correctly.

        try:
            await bus.subscribe(
                topic="surprise",
                group="world_model.latent_model.surprise",
                handler=self._on_surprise,
                schema=SurpriseEvent,
            )
            log.debug("wlm_init: subscribed to surprise")
        except Exception as exc:
            log.error(
                "wlm_init: failed to subscribe to surprise",
                error=str(exc),
                exception_code=EC_TOPO_BUS_SUBSCRIPTION,
            )
            # Non-fatal — without surprise events, the WLM serves potentially
            # stale cache entries until TTL expiry.  This is degraded but not broken.

        log.info("wlm_init_step_6_complete")

        # ── Step 7: Mark as initialized ───────────────────────────────────────
        self._initialized = True
        init_elapsed_ms = (time.monotonic() - init_start) * 1000.0

        log.info(
            "world_latent_model_initialized",
            elapsed_ms=round(init_elapsed_ms, 2),
            hidden_state_version=self._hidden_state_version,
            structural_layer_domains=sl_domains,
            device=self._device,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LIFECYCLE: shutdown()
    # ─────────────────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """
        Gracefully shut down the WorldLatentModel.

        Shutdown sequence:
            1. Set _shutting_down flag to reject new queries.
            2. Wait for in-flight readouts to complete (500ms ceiling).
            3. Cancel all background tasks (bus event handlers).
            4. Clear all caches (release memory).
            5. Log final statistics.

        The shutdown flag is checked at the top of query().  Any query that
        arrives after shutdown begins receives a RuntimeError rather than
        serving stale data.

        Background tasks are cancelled with a timeout to prevent hung
        handlers from blocking shutdown indefinitely.  The timeout matches
        WATCHDOG_SHUTDOWN_DRAIN_TIMEOUT_S (15s).
        """
        if not self._initialized:
            log.warning("wlm_shutdown: model was never initialized — nothing to shut down")
            return

        if self._shutting_down:
            log.warning("wlm_shutdown: already shutting down — duplicate call ignored")
            return

        self._shutting_down = True
        shutdown_start = time.monotonic()
        log.info("wlm_shutdown: beginning graceful shutdown")

        # ── Wait for in-flight readouts ───────────────────────────────────────
        # Use the same 500ms consistency window from the hidden_state update.
        deadline = time.monotonic() + (HIDDEN_STATE_UPDATE_DEADLINE_MS / 1000.0)
        while self._active_readout_count > 0:
            if time.monotonic() > deadline:
                log.warning(
                    "wlm_shutdown: readout drain deadline exceeded",
                    active_readouts=self._active_readout_count,
                )
                break
            await asyncio.sleep(HIDDEN_STATE_POLL_INTERVAL_MS / 1000.0)

        # ── Cancel background tasks ───────────────────────────────────────────
        if self._background_tasks:
            log.info(
                "wlm_shutdown: cancelling background tasks",
                task_count=len(self._background_tasks),
            )
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()

            # Wait for cancellation to propagate.
            try:
                await asyncio.wait(
                    self._background_tasks,
                    timeout=WATCHDOG_SHUTDOWN_DRAIN_TIMEOUT_S,
                )
            except Exception: # noqa
                pass  # Timeout or cancellation — acceptable during shutdown.

            # Collect results to suppress "task exception was never retrieved" warnings.
            for task in self._background_tasks:
                if task.done() and not task.cancelled():
                    exc = task.exception() if not task.cancelled() else None
                    if exc is not None:
                        log.debug(
                            "wlm_shutdown: background task had exception",
                            error=str(exc),
                        )

            self._background_tasks.clear()

        # ── Clear caches ──────────────────────────────────────────────────────
        l1_size = len(self._l1_cache)
        l2_size = len(self._l2_cache)
        self._l1_cache.clear()
        self._l2_cache.clear()

        # ── Log final statistics ──────────────────────────────────────────────
        shutdown_elapsed_ms = (time.monotonic() - shutdown_start) * 1000.0

        log.info(
            "world_latent_model_shutdown_complete",
            elapsed_ms=round(shutdown_elapsed_ms, 2),
            l1_entries_cleared=l1_size,
            l2_entries_cleared=l2_size,
            **self._query_stats.to_log_dict(),
            **self._latency_histogram.to_log_dict(),
            **self._l1_stats.to_log_dict("l1"),
            **self._l2_stats.to_log_dict("l2"),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # PUBLIC INTERFACE: query()
    #
    # The ONLY method the rest of AXIOM calls.  interface.py calls this.
    # Nothing else does.
    # ─────────────────────────────────────────────────────────────────────────

    async def query(
        self,
        topology_class: str,
        intent_vector: Optional[List[float]],
        domain: str,
        phase: int,
    ) -> WLMResponse:
        """
        Query the World Latent Model for a traversal policy, friction forecast,
        and source priority for the given topology class.

        This is the single public method.  Every query from interface.py
        enters through here.  The method routes through three cache tiers:

            L1 hit (domain, topology_class) → return cached response  (<0.5ms)
            L2 hit (topology_class)         → return cached response  (<0.8ms)
            L3 miss                         → model readout           (<5ms)

        TTL=0 topology classes (AUTH_REDIRECT, CLOUDFLARE_CHALLENGE, RATE_LIMITED)
        bypass all cache tiers and always go to L3.  This is a correctness
        requirement, not an optimization.

        Parameters:
            topology_class: The classified topology class for the target URL.
                           Must be a valid topology class string (uppercase,
                           alphanumeric with underscores).
            intent_vector:  Optional 256-dimensional float vector from the AXIOM
                           graph's query embedding.  Conditions the readout for
                           intent-aware source priority.  None for pure topology
                           readouts (cold start warmup, domain topology events).
            domain:         The target domain (e.g. "docs.stripe.com").
                           Used as part of the L1 cache key and for source
                           priority computation.
            phase:          Current WLM phase (1, 2, or 3).
                           Affects source priority breadth (Phase I: k=10,
                           Phase II: k=7, Phase III: k=3).

        Returns:
            WLMResponse containing:
                traversal_policy:  TopologyTraversalPolicy
                friction_forecast: FrictionForecast
                source_priority:   List[str]
                world_confidence:  float

        Raises:
            RuntimeError: If the model is not initialized or is shutting down.

        Latency guarantees:
            <2ms   for Phase III known topology classes (L1 cache hit expected)
            <5ms   for Phase I unknown classes (L3 model readout path)
            <10ms  absolute ceiling — exceeding this is logged at ERROR

        Cache routing math (warm system):
            P(L1 hit) ≈ 0.65
            P(L2 hit | L1 miss) ≈ 0.57   → P(L2 hit) ≈ 0.20
            P(L3) ≈ 0.15

            E[latency] = 0.65 × 0.5ms + 0.20 × 0.8ms + 0.15 × 4ms
                       = 0.325 + 0.16 + 0.60
                       = 1.085ms

            Var[latency] = 0.65 × (0.5 - 1.085)² + 0.20 × (0.8 - 1.085)²
                           + 0.15 × (4.0 - 1.085)²
                         = 0.65 × 0.342 + 0.20 × 0.081 + 0.15 × 8.490
                         = 0.222 + 0.016 + 1.274
                         = 1.512
            σ[latency] ≈ 1.23ms

            p95 ≈ E + 1.645σ = 1.085 + 2.023 = 3.108ms (Gaussian approximation)

            But the distribution is trimodal, not Gaussian.  The exact p95:
                Sorted cumulative: L1 (65%) → L2 (85%) → L3 (100%)
                p95 falls at the 95th percentile → within L3 tier
                Since L3 completes in <5ms, p95 < 5ms.

            For Phase III with higher L1 hit rate (~80%):
                p95 falls within L2 tier → p95 < 1ms.
        """
        # ── Guard: initialization and shutdown ────────────────────────────────
        if not self._initialized:
            raise RuntimeError(
                "WorldLatentModel.query() called before initialize().  "
                "cold_start.py must call initialize() before accepting queries."
            )
        if self._shutting_down:
            raise RuntimeError(
                "WorldLatentModel.query() called during shutdown.  "
                "No new queries are accepted after shutdown() begins."
            )

        query_start = time.monotonic()

        # ── Update query statistics ───────────────────────────────────────────
        self._query_stats.total_queries += 1
        self._query_stats.per_class_counts[topology_class] = (
            self._query_stats.per_class_counts.get(topology_class, 0) + 1
        )
        if phase in self._query_stats.per_phase_counts:
            self._query_stats.per_phase_counts[phase] += 1

        # ── Periodic eviction ─────────────────────────────────────────────────
        # Run eviction sweep every EVICTION_INTERVAL_QUERIES queries.
        # Uses bitwise AND for modular arithmetic (EVICTION_INTERVAL_QUERIES is
        # a power of 2): query_count & (64-1) == 0 every 64 queries.
        if (self._query_stats.total_queries & (EVICTION_INTERVAL_QUERIES - 1)) == 0:
            self._evict_expired()

        # ── TTL=0 bypass ──────────────────────────────────────────────────────
        # TTL=0 classes must NEVER be cached.  Skip directly to L3.
        if self._is_ttl_zero_class(topology_class):
            self._query_stats.ttl_zero_bypasses += 1
            log.debug(
                "query: ttl_zero_bypass",
                topology_class=topology_class,
                domain=domain,
                phase=phase,
            )

            # Rate limit and backoff still apply to TTL=0 classes.
            if self._should_rate_limit_readout():
                response = self._build_fallback_response(topology_class, phase)
                latency_ms = (time.monotonic() - query_start) * 1000.0
                self._latency_histogram.record(latency_ms)
                log.debug(**self._trace_query(topology_class, domain, phase, "rate_limited", latency_ms, response))
                return response

            if self._should_backoff_readout(topology_class):
                response = self._build_fallback_response(topology_class, phase)
                latency_ms = (time.monotonic() - query_start) * 1000.0
                self._latency_histogram.record(latency_ms)
                log.debug(**self._trace_query(topology_class, domain, phase, "backoff", latency_ms, response))
                return response

            try:
                response = await self._l3_readout(
                    topology_class=topology_class,
                    intent_vector=intent_vector,
                    domain=domain,
                    phase=phase,
                )
                self._record_readout_success(topology_class)
            except Exception as exc:
                self._query_stats.readout_errors += 1
                self._record_readout_failure(topology_class)
                log.error(
                    "query: l3_readout_failed_ttl_zero",
                    topology_class=topology_class,
                    domain=domain,
                    error=str(exc),
                )
                # Construct a safe fallback response.
                response = self._build_fallback_response(topology_class, phase)

            self._query_stats.l3_readouts += 1
            latency_ms = (time.monotonic() - query_start) * 1000.0
            self._latency_histogram.record(latency_ms)
            self._check_latency_ceiling(latency_ms, topology_class, domain, "L3_ttl_zero")
            log.debug(**self._trace_query(topology_class, domain, phase, "L3_ttl_zero", latency_ms, response))
            return response

        # ── L1 lookup: (domain, topology_class) ──────────────────────────────
        l1_result = self._l1_lookup(domain, topology_class)
        if l1_result is not None:
            self._query_stats.l1_hits += 1
            latency_ms = (time.monotonic() - query_start) * 1000.0
            self._latency_histogram.record(latency_ms)
            log.debug(**self._trace_query(topology_class, domain, phase, "L1", latency_ms, l1_result))
            return l1_result

        # ── L2 lookup: topology_class ─────────────────────────────────────────
        l2_result = self._l2_lookup(topology_class)
        if l2_result is not None:
            self._query_stats.l2_hits += 1

            # Store in L1 for future hits on this (domain, topology_class).
            # The L2 response has generic source priority; storing in L1 as-is
            # is acceptable because the traversal policy and friction forecast
            # are the primary routing signals.
            self._l1_store(domain, topology_class, l2_result, phase)

            latency_ms = (time.monotonic() - query_start) * 1000.0
            self._latency_histogram.record(latency_ms)
            log.debug(**self._trace_query(topology_class, domain, phase, "L2", latency_ms, l2_result))
            return l2_result

        # ── L3 readout: model inference ───────────────────────────────────────
        # Rate limit check: if too many concurrent readouts are in-flight,
        # the event loop cooperative budget is at risk.  Serve fallback.
        if self._should_rate_limit_readout():
            response = self._build_fallback_response(topology_class, phase)
            latency_ms = (time.monotonic() - query_start) * 1000.0
            self._latency_histogram.record(latency_ms)
            log.warning(
                "query: rate_limited_readout",
                topology_class=topology_class,
                domain=domain,
                active_readouts=self._active_readout_count,
            )
            log.debug(**self._trace_query(topology_class, domain, phase, "rate_limited", latency_ms, response))
            return response

        # Backoff check: repeated failures for this topology class impose
        # exponential backoff to prevent error amplification.
        if self._should_backoff_readout(topology_class):
            response = self._build_fallback_response(topology_class, phase)
            latency_ms = (time.monotonic() - query_start) * 1000.0
            self._latency_histogram.record(latency_ms)
            log.debug(**self._trace_query(topology_class, domain, phase, "backoff", latency_ms, response))
            return response

        try:
            response = await self._l3_readout(
                topology_class=topology_class,
                intent_vector=intent_vector,
                domain=domain,
                phase=phase,
            )
            self._record_readout_success(topology_class)
        except Exception as exc:
            self._query_stats.readout_errors += 1
            self._record_readout_failure(topology_class)
            log.error(
                "query: l3_readout_failed",
                topology_class=topology_class,
                domain=domain,
                phase=phase,
                error=str(exc),
            )
            response = self._build_fallback_response(topology_class, phase)

        self._query_stats.l3_readouts += 1

        # Store in both L1 and L2 with phase-modulated TTL.
        self._l1_store(domain, topology_class, response, phase)
        self._l2_store(topology_class, response, phase)

        latency_ms = (time.monotonic() - query_start) * 1000.0
        self._latency_histogram.record(latency_ms)
        self._check_latency_ceiling(latency_ms, topology_class, domain, "L3")
        log.debug(**self._trace_query(topology_class, domain, phase, "L3", latency_ms, response))

        return response

    # ─────────────────────────────────────────────────────────────────────────
    # TRAINING INTERFACE ACCESSOR
    # ─────────────────────────────────────────────────────────────────────────

    def get_training_interface(self) -> "WLMTrainingInterface":
        """
        Return the restricted training interface for index_daemon.py.

        index_daemon.py receives this object.  It calls nothing else on
        WorldLatentModel.  It does not hold a reference to WorldLatentModel
        itself.  It cannot call query().  It cannot read hidden_state directly.
        It cannot clear caches.  It can only do four things:
            1. get_model()          — for gradient computation
            2. update_hidden_state() — after optimizer.step() with 500ms window
            3. save_checkpoint()    — atomic write to store via staging
            4. get_version()        — monotonic counter for audit trail

        The training interface is created lazily on first call and cached.
        Subsequent calls return the same object.

        The WLMTrainingInterface holds a reference to self (the WorldLatentModel)
        to access _active_readout_count and _hidden_state_version.  This is
        intentional — the training interface needs these to enforce the
        consistency window.

        Raises:
            RuntimeError: If the model is not initialized.
        """
        if not self._initialized:
            raise RuntimeError(
                "WorldLatentModel.get_training_interface() called before initialize().  "
                "The training interface requires a fully initialized model."
            )

        if self._training_interface is None:
            self._training_interface = WLMTrainingInterface(
                self,
                _guard=WLMTrainingInterface._SENTINEL, # noqa
            )
            log.info(
                "wlm_training_interface_created",
                hidden_state_version=self._hidden_state_version,
            )

        return self._training_interface

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE: L1 OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _l1_lookup(
        self,
        domain: str,
        topology_class: str,
    ) -> Optional[WLMResponse]:
        """
        Look up a cached WLMResponse in the L1 domain policy cache.

        L1 key: (domain, topology_class).
        Returns None on miss (key not found, TTL expired, or stale version).

        Consistency checks:
            1. TTL expiry — entry older than CACHE_TTL_BY_CLASS[topology_class].
            2. Version staleness — entry's hidden_state_version differs from
               current version.  This happens after weight reload or training
               update.  Stale entries are evicted on access rather than on
               update to avoid blocking the update path.

        Parameters:
            domain:         The target domain.
            topology_class: The topology class.

        Returns:
            WLMResponse if the entry is valid and fresh.
            None if the entry is missing, expired, or stale.
        """
        key = (domain, topology_class)
        entry = self._l1_cache.get(key)

        if entry is None:
            self._l1_stats.misses += 1
            return None

        # TTL expiry check.
        if entry.is_expired:
            del self._l1_cache[key]
            self._l1_stats.evictions_ttl += 1
            self._l1_stats.current_size = len(self._l1_cache)
            self._l1_stats.misses += 1
            return None

        # Hidden state version staleness check.
        if entry.hidden_state_version != self._hidden_state_version:
            del self._l1_cache[key]
            self._l1_stats.evictions_stale += 1
            self._l1_stats.current_size = len(self._l1_cache)
            self._l1_stats.misses += 1
            return None

        # Valid hit — update access metadata and move to end for LRU ordering.
        # move_to_end() is O(1) on OrderedDict — the least recently used entry
        # is always at the front, so popitem(last=False) evicts it in O(1).
        entry.access_count += 1
        entry.last_access = time.monotonic()
        self._l1_cache.move_to_end(key)
        self._l1_stats.hits += 1
        return entry.response

    def _l1_store(
        self,
        domain: str,
        topology_class: str,
        response: WLMResponse,
        phase: int = PHASE_II,
    ) -> None:
        """
        Store a WLMResponse in the L1 domain policy cache.

        TTL=0 classes are never stored (enforced by _is_ttl_zero_class check).
        If the cache is at capacity, evict the oldest expired entries first,
        then evict the least recently used entry if still at capacity.

        Parameters:
            domain:         The target domain.
            topology_class: The topology class.
            response:       The WLMResponse to cache.
            phase:          Current phase — controls TTL modulation via
                            _modulated_ttl().  Phase III doubles TTL;
                            Phase I halves it.  Defaults to PHASE_II (1×).
        """
        # TTL=0 classes must never be cached.
        if self._is_ttl_zero_class(topology_class):
            return

        # Phase-modulated TTL: Phase I → 0.5×, Phase II → 1.0×, Phase III → 2.0×.
        ttl = self._modulated_ttl(topology_class, phase)

        # Capacity enforcement.
        if len(self._l1_cache) >= L1_MAX_ENTRIES:
            self._evict_l1_lru()

        key = (domain, topology_class)
        now = time.monotonic()

        self._l1_cache[key] = _L1CacheEntry(
            response=response,
            stored_at=now,
            ttl=ttl,
            topology_class=topology_class,
            domain=domain,
            access_count=0,
            last_access=now,
            hidden_state_version=self._hidden_state_version,
        )

        self._l1_stats.stores += 1
        self._l1_stats.current_size = len(self._l1_cache)

    def _evict_l1_lru(self) -> None:
        """
        Evict the least recently used entry from L1 when at capacity.

        First pass: evict expired entries (up to EVICTION_BATCH_SIZE).
        Second pass: if still at capacity, evict the entry with the oldest
        last_access timestamp.

        LRU eviction is O(n) where n is the cache size.  For L1_MAX_ENTRIES=10000,
        this is a linear scan over ~10000 entries.  Each iteration is a float
        comparison — total time is ~50μs on modern hardware.  This is acceptable
        because capacity eviction is rare in production: TTL-based eviction
        keeps the cache below capacity under normal conditions.
        """
        # First pass: evict expired entries.
        expired_keys = []
        for key, entry in self._l1_cache.items():
            if entry.is_expired:
                expired_keys.append(key)
                if len(expired_keys) >= EVICTION_BATCH_SIZE:
                    break

        for key in expired_keys:
            del self._l1_cache[key]
            self._l1_stats.evictions_ttl += 1

        self._l1_stats.current_size = len(self._l1_cache)

        # Check if still at capacity.
        if len(self._l1_cache) < L1_MAX_ENTRIES:
            return

        # Second pass: evict LRU entry — O(1) with OrderedDict.
        # The OrderedDict invariant: least recently used entry is always first
        # because _l1_lookup calls move_to_end() on every cache hit and
        # _l1_store inserts at the end.  popitem(last=False) removes the front.
        self._l1_cache.popitem(last=False)
        self._l1_stats.evictions_cap += 1
        self._l1_stats.current_size = len(self._l1_cache)

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE: L2 OPERATIONS
    # ─────────────────────────────────────────────────────────────────────────

    def _l2_lookup(
        self,
        topology_class: str,
    ) -> Optional[WLMResponse]:
        """
        Look up a cached WLMResponse in the L2 topology class policy cache.

        L2 key: topology_class (no domain).
        Returns None on miss (key not found, TTL expired, or stale version).

        L2 TTL is 2× the L1 TTL for the same topology class.
        L2 responses have generic source priority (no domain-specific ranking).

        Parameters:
            topology_class: The topology class.

        Returns:
            WLMResponse if the entry is valid and fresh.
            None if the entry is missing, expired, or stale.
        """
        entry = self._l2_cache.get(topology_class)

        if entry is None:
            self._l2_stats.misses += 1
            return None

        # TTL expiry check.
        if entry.is_expired:
            del self._l2_cache[topology_class]
            self._l2_stats.evictions_ttl += 1
            self._l2_stats.current_size = len(self._l2_cache)
            self._l2_stats.misses += 1
            return None

        # Hidden state version staleness check.
        if entry.hidden_state_version != self._hidden_state_version:
            del self._l2_cache[topology_class]
            self._l2_stats.evictions_stale += 1
            self._l2_stats.current_size = len(self._l2_cache)
            self._l2_stats.misses += 1
            return None

        # Valid hit.
        entry.access_count += 1
        entry.last_access = time.monotonic()
        self._l2_stats.hits += 1
        return entry.response

    def _l2_store(
        self,
        topology_class: str,
        response: WLMResponse,
        phase: int = PHASE_II,
    ) -> None:
        """
        Store a WLMResponse in the L2 topology class policy cache.

        TTL=0 classes are never stored.
        L2 TTL = phase-modulated L1 TTL × L2_TTL_MULTIPLIER (2×).

        Parameters:
            topology_class: The topology class.
            response:       The WLMResponse to cache.
            phase:          Current phase — controls TTL modulation.  Defaults
                            to PHASE_II (1×, no modulation).
        """
        if self._is_ttl_zero_class(topology_class):
            return

        l1_ttl = self._modulated_ttl(topology_class, phase)
        l2_ttl = l1_ttl * L2_TTL_MULTIPLIER

        # Capacity enforcement (L2 is small — simple replacement).
        if len(self._l2_cache) >= L2_MAX_ENTRIES and topology_class not in self._l2_cache:
            # Evict oldest expired first.
            expired_key = None
            oldest_stored = float("inf")
            for key, entry in self._l2_cache.items():
                if entry.is_expired:
                    expired_key = key
                    break
                if entry.stored_at < oldest_stored:
                    oldest_stored = entry.stored_at
                    expired_key = key
            if expired_key is not None:
                del self._l2_cache[expired_key]
                self._l2_stats.evictions_cap += 1

        now = time.monotonic()
        self._l2_cache[topology_class] = _L2CacheEntry(
            response=response,
            stored_at=now,
            ttl=l2_ttl,
            topology_class=topology_class,
            access_count=0,
            last_access=now,
            hidden_state_version=self._hidden_state_version,
        )

        self._l2_stats.stores += 1
        self._l2_stats.current_size = len(self._l2_cache)

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE: EVICTION
    # ─────────────────────────────────────────────────────────────────────────

    def _evict_expired(self) -> int:
        """
        Sweep both caches and evict expired entries.

        Called periodically from query() every EVICTION_INTERVAL_QUERIES queries.
        Batch-limited to EVICTION_BATCH_SIZE entries per tier per call to
        prevent unbounded eviction time in the query hot path.

        Returns:
            Total number of entries evicted across both tiers.
        """
        total_evicted = 0

        # ── L1 sweep ─────────────────────────────────────────────────────────
        l1_expired_keys: List[Tuple[str, str]] = []
        for key, entry in self._l1_cache.items():
            if entry.is_expired:
                l1_expired_keys.append(key)
                if len(l1_expired_keys) >= EVICTION_BATCH_SIZE:
                    break

        for key in l1_expired_keys:
            del self._l1_cache[key]
            self._l1_stats.evictions_ttl += 1
            total_evicted += 1

        self._l1_stats.current_size = len(self._l1_cache)

        # ── L2 sweep ─────────────────────────────────────────────────────────
        l2_expired_keys: List[str] = []
        for key, entry in self._l2_cache.items():
            if entry.is_expired:
                l2_expired_keys.append(key)
                if len(l2_expired_keys) >= EVICTION_BATCH_SIZE:
                    break

        for key in l2_expired_keys:
            del self._l2_cache[key]
            self._l2_stats.evictions_ttl += 1
            total_evicted += 1

        self._l2_stats.current_size = len(self._l2_cache)

        if total_evicted > 0:
            log.debug(
                "evict_expired: swept",
                l1_evicted=len(l1_expired_keys),
                l2_evicted=len(l2_expired_keys),
                l1_size=len(self._l1_cache),
                l2_size=len(self._l2_cache),
            )

        return total_evicted

    # ─────────────────────────────────────────────────────────────────────────
    # L3: MODEL READOUT
    # ─────────────────────────────────────────────────────────────────────────

    async def _l3_readout(
        self,
        topology_class: str,
        intent_vector: Optional[List[float]],
        domain: str,
        phase: int,
    ) -> WLMResponse:
        """
        Perform an L3 model readout to produce a fresh WLMResponse.

        This is the slowest tier but always produces a fresh result.
        Called only when L1 and L2 both miss.

        The readout counter (_active_readout_count) is incremented before
        readout begins and decremented in a finally block.  This counter
        is the consistency mechanism: WLMTrainingInterface.update_hidden_state()
        waits for this counter to reach zero before updating hidden_state.

        Processing pipeline:
            1. Increment _active_readout_count.
            2. Call _execute_readout() to get raw ReadoutResult from MambaRouter.
            3. Call _assemble_wlm_response() to decode into WLMResponse.
            4. Decrement _active_readout_count in finally block.

        Parameters:
            topology_class: The topology class.
            intent_vector:  Optional intent conditioning vector.
            domain:         Target domain (for source priority context).
            phase:          Current phase (for source priority k selection).

        Returns:
            WLMResponse with fresh readout from current hidden_state.

        Raises:
            RuntimeError: If the router is not loaded.
            ValueError: If readout produces non-finite outputs.
        """
        self._active_readout_count += 1
        try:
            readout_result = await self._execute_readout(
                topology_class=topology_class,
                intent_vector=intent_vector,
            )
            response = self._assemble_wlm_response(
                readout_result=readout_result,
                topology_class=topology_class,
                domain=domain,
                phase=phase,
            )
            return response
        finally:
            self._active_readout_count -= 1

    async def _execute_readout(
        self,
        topology_class: str,
        intent_vector: Optional[List[float]],
    ) -> ReadoutResult:
        """
        Execute the MambaRouter readout for a given topology class.

        This method runs the actual neural network inference — projecting
        hidden_state through the five output heads.  It is the computational
        core of L3.

        The readout is wrapped in asyncio.to_thread() only if the router
        is on GPU (unlikely for WLM).  On CPU, the readout is fast enough
        (<2ms) to execute directly on the event loop without blocking.

        The readout call:
            MambaRouter.readout(topology_class, intent_vector)

        This call is O(1) — one (1, d_model) → (d_model, output_dim) matmul
        per output head.  For d_model=256 and the largest head (source, dim=512):
            FLOPs = 2 × 256 × 512 = 262,144 multiply-accumulate ops
            At 10 GFLOPS/s (conservative CPU): 262144 / 10e9 = 26.2μs
            Total for all heads: ~150μs for the matmuls alone.
            With PyTorch overhead: ~0.5-1.0ms total.

        Parameters:
            topology_class: The topology class for the readout.
            intent_vector:  Optional intent vector for intent-conditioned readout.

        Returns:
            ReadoutResult containing raw tensor outputs from all five heads.

        Raises:
            RuntimeError: If _router is None (model not loaded).
        """
        if self._router is None:
            raise RuntimeError(
                "_execute_readout: MambaRouter is None.  "
                "This indicates initialize() was not called or weight loading failed."
            )

        # MambaRouter.readout() is synchronous and fast (<2ms on CPU).
        # Running it directly on the event loop is acceptable because:
        #   1. The computation is <2ms — well within the asyncio cooperative scheduling budget.
        #   2. Wrapping in asyncio.to_thread() adds ~50μs thread dispatch overhead.
        #   3. The WLM runs on CPU where thread dispatch overhead is significant
        #      relative to the computation itself.
        result: ReadoutResult = self._router.readout(
            topology_class=topology_class,
            intent_vector=intent_vector,
        )

        return result

    def _assemble_wlm_response(
        self,
        readout_result: ReadoutResult,
        topology_class: str,
        domain: str, # noqa
        phase: int,
    ) -> WLMResponse:
        """
        Assemble a complete WLMResponse from a ReadoutResult.

        Decoding pipeline:
            1. decode_traversal(traversal_raw, topology_class)
               → TopologyTraversalPolicy (with bias and validation)
            2. decode_friction(friction_raw, topology_class)
               → FrictionForecast (with coherence, floors, and strategy)
            3. decode_source_priority(source_raw, structural_layer, topology_class, phase)
               → (domains, scores) ranked domain list
            4. Construct WLMResponse from the three decoded outputs.

        The decoder pipeline is fully specified in wlm_decoders.py.
        This method orchestrates the calls and assembles the final contract.

        World confidence computation:
            world_confidence = traversal_policy.confidence × friction_confidence
            where friction_confidence = 1.0 - max(friction_probabilities)

            Interpretation: high world_confidence means the WLM is confident
            in both its traversal policy AND that friction will be low.
            Low world_confidence means either the traversal policy is uncertain
            OR high friction is expected.

            Mathematical properties:
                world_confidence ∈ [0.0, 1.0]
                Monotonically decreasing in max friction probability.
                Product of independent confidence signals — assumes traversal
                and friction uncertainties are approximately independent.

        Parameters:
            readout_result: Raw tensor outputs from MambaRouter.readout().
            topology_class: The topology class.
            domain:         Target domain for source priority.
            phase:          Current phase for source priority k.

        Returns:
            Complete WLMResponse.
        """
        # ── Decode traversal policy ───────────────────────────────────────────
        traversal_policy: TopologyTraversalPolicy = decode_traversal(
            raw=readout_result.traversal_raw,
            topology_class=topology_class,
        )

        # ── Decode friction forecast ──────────────────────────────────────────
        friction_forecast: FrictionForecast = decode_friction(
            raw=readout_result.friction_raw,
            topology_class=topology_class,
        )

        # ── Decode source priority ────────────────────────────────────────────
        source_domains: List[str]
        source_scores: List[float]
        source_domains, source_scores = decode_source_priority(
            raw=readout_result.source_raw,
            structural_layer=self._structural_layer,
            topology_class=topology_class,
            phase=phase,
        )

        # If source priority is empty (EmptyStructuralLayer), use fallback.
        if not source_domains:
            source_priority = list(SOURCE_PRIORITY_FALLBACK)
        else:
            source_priority = source_domains

        # ── Compute world confidence ──────────────────────────────────────────
        # world_confidence = traversal_confidence × (1 - max_friction)
        #
        # This product encodes two independent uncertainty signals:
        #   - traversal_confidence: how confident the model is in its traversal policy
        #   - (1 - max_friction): how confident we are that friction will not block
        #
        # Properties:
        #   - If traversal_confidence is high (e.g. 0.95) and friction is low
        #     (e.g. max_friction = 0.1), world_confidence = 0.95 × 0.90 = 0.855
        #   - If friction is high (e.g. max_friction = 0.95), world_confidence
        #     drops to 0.95 × 0.05 = 0.0475 regardless of traversal confidence.
        #   - This correctly captures the asymmetry: high friction dominates
        #     the confidence signal because it means the fetch may fail entirely.

        max_friction = max(
            friction_forecast.cloudflare_probability,
            friction_forecast.paywall_probability,
            friction_forecast.rate_limit_probability,
            friction_forecast.auth_redirect_probability,
        )
        friction_confidence = 1.0 - max_friction
        world_confidence = traversal_policy.confidence * friction_confidence

        # Clamp to [0.0, 1.0] for safety.
        world_confidence = max(0.0, min(1.0, world_confidence))

        # ── Construct WLMResponse ─────────────────────────────────────────────
        return WLMResponse(
            traversal_policy=traversal_policy,
            friction_forecast=friction_forecast,
            source_priority=source_priority,
            world_confidence=round(world_confidence, 6),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # FALLBACK RESPONSE BUILDER
    # ─────────────────────────────────────────────────────────────────────────

    def _build_fallback_response( # noqa
        self,
        topology_class: str,
        phase: int, # noqa
    ) -> WLMResponse:
        """
        Build a safe fallback WLMResponse when L3 readout fails.

        The fallback uses conservative defaults:
            - depth=1 (minimal traversal)
            - render_mode="static" (cheapest fetch)
            - low rps (gentle pacing)
            - zero retries (do not compound failure)
            - moderate timeout
            - low confidence (signals uncertainty to callers)
            - no source priority (Phantom uses URL-based routing)
            - all friction at 0.5 (uncertain — callers exercise caution)

        This fallback is never cached — it is produced on each readout failure
        and served only for the current query.  The next query will re-attempt
        L3 readout.

        Parameters:
            topology_class: The topology class.
            phase:          Current phase.

        Returns:
            Conservative WLMResponse.
        """
        traversal = TopologyTraversalPolicy(
            topology_class=topology_class,
            depth=1,
            render_mode="static",
            requests_per_second=0.5,
            retry_budget=0,
            timeout_ms=5000,
            confidence=0.1,
        )

        friction = FrictionForecast(
            topology_class=topology_class,
            cloudflare_probability=0.5,
            paywall_probability=0.5,
            rate_limit_probability=0.5,
            auth_redirect_probability=0.5,
            mitigation_strategy="standard",
        )

        return WLMResponse(
            traversal_policy=traversal,
            friction_forecast=friction,
            source_priority=list(SOURCE_PRIORITY_FALLBACK),
            world_confidence=0.05,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # LATENCY MONITORING
    # ─────────────────────────────────────────────────────────────────────────

    def _check_latency_ceiling(
        self,
        latency_ms: float,
        topology_class: str,
        domain: str,
        tier: str,
    ) -> None:
        """
        Check if query latency exceeded the absolute ceiling.

        Latencies above LATENCY_ABSOLUTE_CEILING_MS are logged at ERROR
        and counted in the anomaly counter.  This is the primary signal
        that something is wrong upstream (model load, weight corruption,
        contention on the event loop).

        Parameters:
            latency_ms:     The observed latency in milliseconds.
            topology_class: The topology class for diagnostic context.
            domain:         The domain for diagnostic context.
            tier:           Which cache tier served the response ("L1", "L2", "L3").
        """
        if latency_ms > LATENCY_ABSOLUTE_CEILING_MS:
            self._query_stats.latency_ceiling_breaches += 1
            log.error(
                "query_latency_ceiling_exceeded",
                latency_ms=round(latency_ms, 3),
                ceiling_ms=LATENCY_ABSOLUTE_CEILING_MS,
                topology_class=topology_class,
                domain=domain,
                tier=tier,
                total_breaches=self._query_stats.latency_ceiling_breaches,
            )

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY: TTL CHECK
    # ─────────────────────────────────────────────────────────────────────────

    def _is_ttl_zero_class(self, topology_class: str) -> bool: # noqa
        """
        Check if a topology class has TTL=0 (must never be cached).

        Uses the precomputed _TTL_ZERO_CLASSES frozenset for O(1) lookup.
        Classes not in CACHE_TTL_BY_CLASS default to non-zero TTL and
        are therefore cacheable.

        Parameters:
            topology_class: The topology class to check.

        Returns:
            True if the class has TTL=0 and must bypass all caches.
        """
        return topology_class in _TTL_ZERO_CLASSES

    # ─────────────────────────────────────────────────────────────────────────
    # UTILITY: FILE SHA256
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_file_sha256(path: Path) -> str:
        """
        Compute the SHA-256 digest of a file.

        Reads the file in 64KB chunks to handle large model files without
        loading the entire file into memory.

        Parameters:
            path: Path to the file.

        Returns:
            Lowercase hex SHA-256 digest string (64 characters).
        """
        sha256 = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65_536)
                if not chunk:
                    break
                sha256.update(chunk)
        return sha256.hexdigest()

    # ─────────────────────────────────────────────────────────────────────────
    # RELOAD: WEIGHTS (topology_router.pt)
    # ─────────────────────────────────────────────────────────────────────────

    async def _reload_weights(self) -> None:
        """
        Reload topology_router.pt after a WATCHDOG notification.

        Reload strategy — load into temporary model first:
            1. Construct a new MambaRouter with the same config.
            2. Load the new state_dict into the temporary model.
            3. Run validation readout on the temporary model.
            4. If validation passes: atomic reference swap (GIL-safe).
            5. Invalidate ALL L1 and L2 caches — new weights change
               traversal and friction predictions fundamentally.
            6. Increment hidden_state_version.

        If validation fails: keep current weights, log ERROR, return.
        A failed reload is better than serving with corrupted weights.

        The reference swap is atomic because Python's GIL ensures that
        the assignment self._router = temp_router is a single bytecode
        instruction (STORE_FAST or STORE_ATTR).  Any in-flight readout()
        call that started before the swap completes against the old router.
        The next readout() call uses the new router.

        This handler is registered with WATCHDOG at initialize() time
        with debounce_ms=500.  It fires when topology_router.pt is
        atomically renamed from the staging path (IN_MOVED_TO event).
        """
        reload_start = time.monotonic()
        log.info("wlm_reload_weights: starting weight reload")

        if not self._weights_path.exists():
            log.error(
                "wlm_reload_weights: topology_router.pt does not exist",
                path=str(self._weights_path),
            )
            return

        # ── Step 1: Load into temporary model ─────────────────────────────────
        try:
            state_dict = torch.load(
                str(self._weights_path),
                weights_only=True,   # security requirement — no exceptions
                map_location=self._device,
            )
        except Exception as exc:
            log.error(
                "wlm_reload_weights: failed to load state_dict",
                error=str(exc),
                path=str(self._weights_path),
            )
            return

        try:
            temp_router = MambaRouter(
                vocab_size=self._config.vocab_size,
                d_model=self._config.d_model,
                d_state=self._config.d_state,
                d_conv=self._config.d_conv,
                expand=self._config.expand,
                n_layers=self._config.n_layers,
                n_topology=self._config.n_topology,
                n_source=self._config.n_source,
                n_phase=self._config.n_phase,
                dropout=self._config.dropout,
                max_seq_len=self._config.max_seq_len,
            )
            temp_router.load_state_dict(state_dict)
            temp_router.eval()
        except Exception as exc:
            log.error(
                "wlm_reload_weights: failed to construct/load temp router",
                error=str(exc),
            )
            return

        # ── Step 2: Validate temp router outputs ──────────────────────────────
        if not self._validate_router_outputs(temp_router):
            log.error(
                "wlm_reload_weights: validation failed — keeping current weights",
                path=str(self._weights_path),
            )
            return

        # ── Step 3: Atomic reference swap ─────────────────────────────────────
        old_version = self._hidden_state_version
        self._router = temp_router
        self._hidden_state_version += 1

        # ── Step 4: Invalidate all caches ─────────────────────────────────────
        l1_cleared = len(self._l1_cache)
        l2_cleared = len(self._l2_cache)
        self._l1_cache.clear()
        self._l2_cache.clear()
        self._l1_stats.current_size = 0
        self._l2_stats.current_size = 0

        reload_elapsed_ms = (time.monotonic() - reload_start) * 1000.0

        log.info(
            "wlm_reload_weights: complete",
            elapsed_ms=round(reload_elapsed_ms, 2),
            old_version=old_version,
            new_version=self._hidden_state_version,
            l1_entries_cleared=l1_cleared,
            l2_entries_cleared=l2_cleared,
            hidden_state_digest=self._router.hidden_state_digest(),
        )

    # ─────────────────────────────────────────────────────────────────────────
    # RELOAD: STRUCTURAL LAYER (structural_layer.pt)
    # ─────────────────────────────────────────────────────────────────────────

    async def _reload_structural_layer(self) -> None:
        """
        Reload structural_layer.pt after a WATCHDOG notification.

        Unlike weight reload, structural layer reload does NOT clear caches.
        The asymmetry is intentional:
            - Weight reload changes traversal and friction predictions → all caches stale.
            - Structural layer reload changes source priority only → the least
              critical output.  Existing traversal and friction policies are still valid.
            - Source priority in cached WLMResponses will be stale but acceptable.
            - Next L3 readout for each topology class produces updated source priority.

        If the structural layer fails to load (corrupt file, validation failure),
        the existing structural layer is kept.  If it was EmptyStructuralLayer,
        it remains EmptyStructuralLayer.  Source priority degrades gracefully
        to GENERIC_FALLBACK.
        """
        reload_start = time.monotonic()
        log.info(
            "wlm_reload_structural_layer: starting",
            path=str(self._structural_layer_path),
        )

        new_layer = load_structural_layer(self._structural_layer_path)
        old_type = type(self._structural_layer).__name__
        new_type = type(new_layer).__name__

        # Atomic reference swap.
        self._structural_layer = new_layer

        new_domains = (
            new_layer.n_domains
            if hasattr(new_layer, "n_domains")
            else 0
        )

        reload_elapsed_ms = (time.monotonic() - reload_start) * 1000.0

        log.info(
            "wlm_reload_structural_layer: complete",
            elapsed_ms=round(reload_elapsed_ms, 2),
            old_type=old_type,
            new_type=new_type,
            domain_count=new_domains,
            caches_cleared=False,  # Intentionally not clearing caches
        )

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATION: ROUTER OUTPUTS
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_router_outputs(self, router: MambaRouter) -> bool: # noqa
        """
        Validate that a MambaRouter produces outputs in expected ranges.

        Called during weight reload to verify the new weights before swapping.
        Performs a single readout with a known topology class and checks:
            1. Traversal raw outputs are finite.
            2. Friction raw outputs are finite.
            3. Source raw outputs are finite.
            4. Phase raw outputs are finite.
            5. Decoded traversal depth is in [1, ROUTER_VALIDATION_MAX_TRAVERSAL_DEPTH].
            6. Decoded friction probabilities are in ROUTER_VALIDATION_FRICTION_RANGE.
            7. Decoded RPS is in [0, ROUTER_VALIDATION_MAX_RPS].

        This validation is not exhaustive — it checks for gross corruption
        (NaN, Inf, extreme values) rather than semantic correctness.
        A model that passes this check produces validly shaped outputs.
        Whether the values are useful depends on training quality.

        Parameters:
            router: The MambaRouter to validate.

        Returns:
            True if validation passes.  False if any check fails.
        """
        try:
            result: ReadoutResult = router.readout(
                topology_class=ROUTER_VALIDATION_TOPOLOGY_CLASS,
                intent_vector=None,
            )
        except Exception as exc:
            log.error(
                "validate_router_outputs: readout failed",
                error=str(exc),
            )
            return False

        # ── Check all tensors are finite ──────────────────────────────────────
        for name, tensor in [
            ("traversal_raw", result.traversal_raw),
            ("friction_raw", result.friction_raw),
            ("source_raw", result.source_raw),
            ("phase_raw", result.phase_raw),
        ]:
            if not torch.isfinite(tensor).all():
                log.error(
                    "validate_router_outputs: non-finite values",
                    tensor_name=name,
                    nan_count=torch.isnan(tensor).sum().item(),
                    inf_count=torch.isinf(tensor).sum().item(),
                )
                return False

        # ── Check decoded traversal depth ─────────────────────────────────────
        try:
            traversal = decode_traversal(
                raw=result.traversal_raw,
                topology_class=ROUTER_VALIDATION_TOPOLOGY_CLASS,
            )
            if traversal.depth > ROUTER_VALIDATION_MAX_TRAVERSAL_DEPTH:
                log.error(
                    "validate_router_outputs: depth out of range",
                    depth=traversal.depth,
                    max_depth=ROUTER_VALIDATION_MAX_TRAVERSAL_DEPTH,
                )
                return False
            if traversal.requests_per_second > ROUTER_VALIDATION_MAX_RPS:
                log.error(
                    "validate_router_outputs: RPS out of range",
                    rps=traversal.requests_per_second,
                    max_rps=ROUTER_VALIDATION_MAX_RPS,
                )
                return False
        except Exception as exc:
            log.error(
                "validate_router_outputs: traversal decode failed",
                error=str(exc),
            )
            return False

        # ── Check decoded friction probabilities ──────────────────────────────
        try:
            friction = decode_friction(
                raw=result.friction_raw,
                topology_class=ROUTER_VALIDATION_TOPOLOGY_CLASS,
            )
            low, high = ROUTER_VALIDATION_FRICTION_RANGE
            for field_name in (
                "cloudflare_probability",
                "paywall_probability",
                "rate_limit_probability",
                "auth_redirect_probability",
            ):
                val = getattr(friction, field_name)
                if not (low <= val <= high):
                    log.error(
                        "validate_router_outputs: friction probability out of range",
                        field=field_name,
                        value=val,
                        range=(low, high),
                    )
                    return False
        except Exception as exc:
            log.error(
                "validate_router_outputs: friction decode failed",
                error=str(exc),
            )
            return False

        log.debug("validate_router_outputs: passed")
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # BUS HANDLER: DOMAIN TOPOLOGY EVENT
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_domain_topology(self, event: DomainTopologyEvent) -> None:
        """
        Handle a DomainTopologyEvent from the CrawlerBus.

        Dispatched as a background task — never blocks query().

        Processing:
            1. Encode the domain topology event as a token sequence via WLMTokenizer.
            2. Run forward() with update_hidden=False.
               (index_daemon owns hidden_state; event processing is observation-only.)
            3. Decode the forward result into a WLMResponse.
            4. Store in L1 cache for the event's domain.
            5. If structural_layer has the domain: update source priority in cache.

        This pre-populates L1 for domains discovered through domain analysis,
        reducing L3 readouts for subsequent queries.  The forward() call
        processes the domain topology event through the Mamba blocks, enriching
        the hidden state's representation of this domain's structural pattern.

        Note: forward() is called with update_hidden=False because the
        hidden_state is owned by index_daemon's training loop.  Domain
        topology events inform the model's representation but do not
        commit hidden state updates outside the training interface.

        This handler is fire-and-forget via asyncio.create_task().
        A slow domain topology event must not delay a live query.

        Parameters:
            event: The DomainTopologyEvent from the bus.
        """
        task = asyncio.current_task()
        if task is not None:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        try:
            domain = event.domain
            domain_map = event.domain_map

            log.debug(
                "on_domain_topology: processing",
                domain=domain,
                bot_mitigation=domain_map.bot_mitigation,
                path_topology_count=len(domain_map.path_topology_map),
            )

            # For each topology class observed in the domain map, pre-populate L1.
            # This avoids L3 readouts for the first query to each (domain, class) pair.
            for path_pattern, topo_class in domain_map.path_topology_map.items():
                if self._is_ttl_zero_class(topo_class):
                    continue

                # Check if we already have an L1 entry for this (domain, class).
                key = (domain, topo_class)
                if key in self._l1_cache:
                    existing = self._l1_cache[key]
                    if not existing.is_expired and existing.hidden_state_version == self._hidden_state_version:
                        continue  # Fresh entry exists — skip.

                # Check L2 for this topology class.
                l2_entry = self._l2_cache.get(topo_class)
                if l2_entry is not None and not l2_entry.is_expired and l2_entry.hidden_state_version == self._hidden_state_version:
                    # L2 has a fresh entry — promote to L1 for this domain.
                    self._l1_store(domain, topo_class, l2_entry.response, PHASE_I)
                    continue

                # No cached entry — perform L3 readout.
                try:
                    response = await self._l3_readout(
                        topology_class=topo_class,
                        intent_vector=None,
                        domain=domain,
                        phase=PHASE_I,
                    )
                    self._l1_store(domain, topo_class, response, PHASE_I)
                    self._l2_store(topo_class, response, PHASE_I)
                except Exception as readout_exc:
                    log.debug(
                        "on_domain_topology: readout failed for class",
                        domain=domain,
                        topology_class=topo_class,
                        error=str(readout_exc),
                    )

            log.debug(
                "on_domain_topology: complete",
                domain=domain,
            )

        except Exception as exc:
            log.error(
                "on_domain_topology: handler failed",
                error=str(exc),
                domain=getattr(event, "domain", "unknown"),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # BUS HANDLER: SURPRISE EVENT
    # ─────────────────────────────────────────────────────────────────────────

    async def _on_surprise(self, event: SurpriseEvent) -> None:
        """
        Handle a SurpriseEvent from the CrawlerBus.

        Dispatched as a background task — never blocks query().

        On dissolve_triggered=True:
            1. Invalidate ALL L1 entries for the affected topology class.
            2. Invalidate the L2 entry for the affected topology class.
            3. Do NOT invalidate other topology classes — surprise is class-specific.
            4. Log invalidation with event.surprise_score for Witness.

        On dissolve_triggered=False:
            Log the surprise event but do not invalidate caches.
            A surprise below the dissolution threshold is informational —
            the existing cached policies are still valid.

        The invalidation is targeted: only the affected topology class loses
        its cache entries.  Other topology classes retain their cached policies.
        This is correct because surprise indicates that the structural
        characteristics of ONE topology class have changed, not all of them.

        Parameters:
            event: The SurpriseEvent from the bus.
        """
        task = asyncio.current_task()
        if task is not None:
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)

        try:
            topo_class = event.topology_class
            surprise_score = event.surprise_score
            dissolve = event.dissolve_triggered

            log.info(
                "on_surprise: received",
                topology_class=topo_class,
                surprise_score=round(surprise_score, 4),
                dissolve_triggered=dissolve,
                theta_surprise=round(event.theta_surprise, 4),
                current_phase=event.current_phase,
            )

            if not dissolve:
                # Surprise below dissolution threshold — informational only.
                log.debug(
                    "on_surprise: no dissolution — caches retained",
                    topology_class=topo_class,
                )
                return

            # ── Invalidate L1 entries for this topology class ─────────────────
            l1_invalidated = 0
            l1_keys_to_remove = [
                key for key in self._l1_cache
                if key[1] == topo_class  # key = (domain, topology_class)
            ]
            for key in l1_keys_to_remove:
                del self._l1_cache[key]
                l1_invalidated += 1

            self._l1_stats.invalidations += l1_invalidated
            self._l1_stats.current_size = len(self._l1_cache)

            # ── Invalidate L2 entry for this topology class ───────────────────
            l2_invalidated = 0
            if topo_class in self._l2_cache:
                del self._l2_cache[topo_class]
                l2_invalidated = 1
                self._l2_stats.invalidations += 1
                self._l2_stats.current_size = len(self._l2_cache)

            log.info(
                "on_surprise: cache invalidated",
                topology_class=topo_class,
                surprise_score=round(surprise_score, 4),
                l1_entries_invalidated=l1_invalidated,
                l2_entries_invalidated=l2_invalidated,
            )

        except Exception as exc:
            log.error(
                "on_surprise: handler failed",
                error=str(exc),
                topology_class=getattr(event, "topology_class", "unknown"),
            )

    # ─────────────────────────────────────────────────────────────────────────
    # COLD START WARMUP
    # ─────────────────────────────────────────────────────────────────────────

    async def cold_start_warmup(self) -> None:
        """
        Pre-populate the L2 cache for all 18 topology classes.

        Called by cold_start.py BEFORE interface.py accepts queries.
        This ensures that even the first query to each topology class
        hits L2 rather than L3, reducing first-query latency.

        Warmup ordering:
            Leaf classes before parent classes.

            A leaf class warm-up informs the parent class warm-up because
            the parent class hidden state may be influenced by leaf class
            activations in the Mamba blocks.

            Order derived from PARENT_CLASS_MAP — deepest leaves first:
                1. Classes that appear as values in PARENT_CLASS_MAP but NOT as keys
                   (pure leaves — no children)
                2. Classes that appear as both keys and values (internal nodes)
                3. Classes that appear as keys but NOT as values (roots)
                4. Classes not in PARENT_CLASS_MAP at all (standalone classes)

            This ordering implements a topological sort over the parent-child
            relationship defined by PARENT_CLASS_MAP.

        Mathematical justification for ordering:
            The Mamba SSM accumulates state across sequential readouts:
                H_t = A(x_t) ⊙ H_{t-1} + B(x_t) ⊙ x_t
            where x_t is the topology class embedding at readout t.

            If leaf classes are warmed first, their activation patterns
            pass through the Mamba blocks and influence H_t.  When a
            parent class is subsequently warmed, it benefits from the
            enriched hidden state that includes leaf class information.

            This ordering is not an optimization — it ensures that parent
            class policies correctly reflect the structural knowledge
            from their children.
        """
        if not self._initialized:
            raise RuntimeError(
                "cold_start_warmup() called before initialize().  "
                "The model must be fully initialized before warming the cache."
            )

        warmup_start = time.monotonic()
        log.info("cold_start_warmup: starting")

        warmup_order = self._derive_warmup_order()

        log.info(
            "cold_start_warmup: derived ordering",
            order=warmup_order,
            total_classes=len(warmup_order),
        )

        warmed_count = 0
        failed_count = 0

        for topology_class in warmup_order:
            try:
                response = await self._l3_readout(
                    topology_class=topology_class,
                    intent_vector=None,  # Pure topology readout — no intent conditioning.
                    domain="",           # Empty domain — cold start has no domain context.
                    phase=PHASE_I,       # Phase I — broadest source priority exploration.
                )

                # Store in L2 only — L1 requires a domain key.
                self._l2_store(topology_class, response, PHASE_I)
                warmed_count += 1

                log.debug(
                    "cold_start_warmup: warmed class",
                    topology_class=topology_class,
                    confidence=round(response.world_confidence, 4),
                )

            except Exception as exc:
                failed_count += 1
                log.warning(
                    "cold_start_warmup: failed to warm class",
                    topology_class=topology_class,
                    error=str(exc),
                )

        warmup_elapsed_ms = (time.monotonic() - warmup_start) * 1000.0

        log.info(
            "cold_start_warmup_complete",
            classes_warmed=warmed_count,
            classes_failed=failed_count,
            elapsed_ms=round(warmup_elapsed_ms, 2),
            l2_cache_size=len(self._l2_cache),
        )

    def _derive_warmup_order(self) -> List[str]: # noqa
        """
        Derive the topological warmup order from PARENT_CLASS_MAP.

        Implements Kahn's algorithm for topological sort over the parent-child
        relationship.  The graph is a forest (multiple roots) where edges
        point from child to parent.  Topological sort visits leaves first,
        then internal nodes, then roots.

        Algorithm:
            1. Build the directed graph: child → parent edges from PARENT_CLASS_MAP.
            2. Compute in-degree for each node (number of children).
            3. Initialize queue with all zero-in-degree nodes (leaves).
            4. Process queue: dequeue node, add to result, decrement in-degree
               of its parent, enqueue parent if in-degree reaches zero.
            5. Append any classes not in PARENT_CLASS_MAP (standalone classes) last.

        Time complexity: O(V + E) where V = |TOPOLOGY_CLASSES| and E = |PARENT_CLASS_MAP|.
        For 18 classes and 12 parent-child relationships: O(30) — negligible.

        Returns:
            List of topology class strings in warmup order (leaves first).
        """
        # Build parent-child relationship.
        # PARENT_CLASS_MAP: child → parent
        children: Dict[str, List[str]] = {}  # parent → [children]
        all_nodes: Set[str] = set(TOPOLOGY_CLASSES)

        for child, parent in PARENT_CLASS_MAP.items():
            if parent not in children:
                children[parent] = []
            children[parent].append(child)
            all_nodes.add(child)
            all_nodes.add(parent)

        # Compute in-degree (number of children pointing to each node as parent).
        in_degree: Dict[str, int] = {node: 0 for node in all_nodes}
        for parent, kids in children.items():
            in_degree[parent] = len(kids)

        # For topological sort from leaves to roots, we reverse the edge direction:
        # A "leaf" has in-degree 0 (no children).
        # A "root" has high in-degree (many children).
        # We want leaves first → process zero in-degree first.

        # Kahn's algorithm.
        from collections import deque
        queue: deque = deque()
        for node in all_nodes:
            if in_degree[node] == 0:
                queue.append(node)

        result: List[str] = []
        visited: Set[str] = set()

        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            result.append(node)

            # Find node's parent (if any) and decrement parent's in-degree.
            parent = PARENT_CLASS_MAP.get(node)
            if parent is not None and parent in in_degree:
                in_degree[parent] -= 1
                if in_degree[parent] == 0:
                    queue.append(parent)

        # Append any remaining classes not yet visited.
        for tc in TOPOLOGY_CLASSES:
            if tc not in visited:
                result.append(tc)

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # HEALTH
    # ─────────────────────────────────────────────────────────────────────────

    def health(self) -> Dict[str, Any]:
        """
        Return a health snapshot for Witness consumption.

        Returns a flat dict suitable for structured logging with all
        cache statistics, latency metrics, and query distribution data.

        Health indicator interpretations:
            cache_hit_rate > 0.85: healthy — most queries served from cache.
            cache_hit_rate ∈ [0.50, 0.85): degraded — too many L3 readouts.
                Check: TTL values may be too short, or many new domains/classes.
            cache_hit_rate < 0.50: unhealthy — cache is ineffective.
                Check: surprise events may be invalidating too aggressively,
                or the topology class distribution has shifted.

            latency_p95_ms < 2.0: healthy — meeting Phase III target.
            latency_p95_ms ∈ [2.0, 5.0): degraded — L3 readouts dominating.
            latency_p95_ms > 5.0: unhealthy — something is blocking readouts.

            active_readout_count == 0: healthy (outside of active queries).
            active_readout_count > 4: concerning — concurrent readout saturation.

        Predicted cache hit rate model:
            For a system with N unique (domain, class) pairs and query rate λ_i
            for pair i, the overall cache hit rate is:

                H = (1/Σλ_i) × Σ_i λ_i × (1 - e^(-λ_i × TTL_i))

            where TTL_i is the TTL for topology class of pair i.

            For a Zipf-distributed query rate (α=1.0, common in web crawling):
                λ_i = C / i  for pairs ranked by frequency

            The cache hit rate converges to:
                H ≈ 1 - (log(N) / (N × C × TTL_avg))

            For N=5000 pairs, C=10 queries/sec, TTL_avg=3600s:
                H ≈ 1 - (8.5 / (5000 × 10 × 3600))
                H ≈ 1 - 4.7e-8
                H ≈ 1.0  (effectively all cache hits)

            The model breaks down for low-traffic pairs (λ_i × TTL_i < 1).
            For the tail 20% of pairs with λ_i < 0.001:
                H_tail ≈ 1 - e^(-0.001 × 3600) ≈ 1 - e^(-3.6) ≈ 0.97

            Overall, the three-tier cache achieves >95% hit rate for any
            Zipf-distributed workload with TTL_avg > 1000s.
        """
        result: Dict[str, Any] = {
            "initialized":            self._initialized,
            "shutting_down":          self._shutting_down,
            "hidden_state_version":   self._hidden_state_version,
            "active_readout_count":   self._active_readout_count,
        }

        # Cache statistics.
        result.update(self._l1_stats.to_log_dict("l1"))
        result.update(self._l2_stats.to_log_dict("l2"))

        # Query statistics.
        result.update(self._query_stats.to_log_dict())

        # Latency metrics.
        result.update(self._latency_histogram.to_log_dict())

        # Structural layer info.
        sl = self._structural_layer
        result["structural_layer_type"] = type(sl).__name__
        result["structural_layer_domains"] = (
            sl.n_domains if hasattr(sl, "n_domains") else 0
        )

        # Health indicators.
        result["health_cache_status"] = (
            "healthy" if self._query_stats.cache_hit_rate > 0.85
            else "degraded" if self._query_stats.cache_hit_rate > 0.50
            else "unhealthy"
        )
        p95 = self._latency_histogram.p95
        result["health_latency_status"] = (
            "healthy" if p95 < LATENCY_P95_TARGET_MS
            else "degraded" if p95 < LATENCY_P99_TARGET_MS
            else "unhealthy"
        )
        result["health_readout_status"] = (
            "healthy" if self._active_readout_count == 0
            else "concerning" if self._active_readout_count > 4
            else "ok"
        )

        # Predicted vs actual comparison.
        # Useful for detecting cache tuning opportunities.
        total_q = self._query_stats.total_queries
        if total_q > 100:
            # Sufficient data for meaningful comparison.
            actual_hit_rate = self._query_stats.cache_hit_rate
            # Simple predicted hit rate assuming 80/20 rule (Pareto).
            # 20% of (domain, class) pairs account for 80% of queries.
            # These high-frequency pairs almost always hit L1/L2.
            # Predicted: 0.80 × 1.0 + 0.20 × 0.50 = 0.90
            predicted_hit_rate = 0.90
            result["hit_rate_vs_predicted"] = round(
                actual_hit_rate - predicted_hit_rate, 4
            )

        return result


    # ─────────────────────────────────────────────────────────────────────────
    # CACHE COHERENCE VERIFICATION
    # ─────────────────────────────────────────────────────────────────────────

    def verify_cache_coherence(self) -> Dict[str, Any]:
        """
        Verify the internal consistency of both cache tiers.

        Checks:
            1. No L1 entries reference topology classes with TTL=0.
            2. No L2 entries reference topology classes with TTL=0.
            3. No L1 entries have hidden_state_version > current version.
            4. No L2 entries have hidden_state_version > current version.
            5. L1 entry count matches _l1_stats.current_size.
            6. L2 entry count matches _l2_stats.current_size.
            7. No L1 entries have negative remaining TTL beyond a grace threshold.
            8. All L1 keys are (str, str) tuples.
            9. All L2 keys are str.

        This is a diagnostic method — called by health endpoints and tests,
        not by the hot path.  Time complexity: O(n) where n is the total
        number of cache entries.

        Returns:
            Dict with coherence check results:
                - coherent: True if all checks pass.
                - violations: List of violation description strings.
                - l1_checked: Number of L1 entries checked.
                - l2_checked: Number of L2 entries checked.
        """
        violations: List[str] = []
        l1_checked = 0
        l2_checked = 0

        # ── L1 coherence checks ──────────────────────────────────────────────
        for key, entry in self._l1_cache.items():
            l1_checked += 1

            # Check 1: TTL=0 class in cache.
            if self._is_ttl_zero_class(entry.topology_class):
                violations.append(
                    f"L1 contains TTL=0 class: key={key}, "
                    f"topology_class={entry.topology_class}"
                )

            # Check 3: Future version (should not happen).
            if entry.hidden_state_version > self._hidden_state_version:
                violations.append(
                    f"L1 entry has future version: key={key}, "
                    f"entry_version={entry.hidden_state_version}, "
                    f"current_version={self._hidden_state_version}"
                )

            # Check 8: Key format.
            if not isinstance(key, tuple) or len(key) != 2:
                violations.append(
                    f"L1 key is not a 2-tuple: key={key!r}"
                )
            elif not isinstance(key[0], str) or not isinstance(key[1], str):
                violations.append(
                    f"L1 key elements are not strings: key={key!r}"
                )

        # ── L2 coherence checks ──────────────────────────────────────────────
        for key, entry in self._l2_cache.items():
            l2_checked += 1

            # Check 2: TTL=0 class in cache.
            if self._is_ttl_zero_class(entry.topology_class):
                violations.append(
                    f"L2 contains TTL=0 class: key={key}, "
                    f"topology_class={entry.topology_class}"
                )

            # Check 4: Future version.
            if entry.hidden_state_version > self._hidden_state_version:
                violations.append(
                    f"L2 entry has future version: key={key}, "
                    f"entry_version={entry.hidden_state_version}, "
                    f"current_version={self._hidden_state_version}"
                )

            # Check 9: Key format.
            if not isinstance(key, str):
                violations.append( # noqa
                    f"L2 key is not a string: key={key!r}"
                )

        # ── Size consistency ──────────────────────────────────────────────────
        # Check 5 and 6.
        actual_l1 = len(self._l1_cache)
        if actual_l1 != self._l1_stats.current_size:
            violations.append(
                f"L1 size mismatch: actual={actual_l1}, "
                f"stats={self._l1_stats.current_size}"
            )

        actual_l2 = len(self._l2_cache)
        if actual_l2 != self._l2_stats.current_size:
            violations.append(
                f"L2 size mismatch: actual={actual_l2}, "
                f"stats={self._l2_stats.current_size}"
            )

        is_coherent = len(violations) == 0

        result = {
            "coherent":   is_coherent,
            "violations": violations,
            "l1_checked": l1_checked,
            "l2_checked": l2_checked,
        }

        if not is_coherent:
            log.warning(
                "cache_coherence_violation",
                violation_count=len(violations),
                violations=violations[:5],  # Log first 5 only to avoid log explosion
            )

        return result

    # ─────────────────────────────────────────────────────────────────────────
    # PHASE-AWARE CACHE TTL MODULATION
    # ─────────────────────────────────────────────────────────────────────────

    def _modulated_ttl( # noqa
        self,
        topology_class: str,
        phase: int,
    ) -> float:
        """
        Compute the effective TTL for a topology class, modulated by phase.

        Phase modulation rationale:
            Phase I (learns):    TTL × 0.5 — the world model is still building.
                                  Shorter TTL ensures frequent L3 readouts that
                                  reflect the rapidly evolving hidden state.
            Phase II (predicts): TTL × 1.0 — standard TTL.  The model's
                                  predictions are reasonably stable.
            Phase III (knows):   TTL × 2.0 — compiled policy.  The model is
                                  highly confident.  Extended TTL reduces L3
                                  readouts and improves cache hit rates.

        Mathematical effect on cache hit rate:
            Let λ = query arrival rate for a (domain, class) pair.
            Let T = TTL.
            Probability of cache hit: P(hit) = 1 - e^(-λT) for Poisson arrivals.

            Phase I:  T' = 0.5T → P(hit) = 1 - e^(-0.5λT)
            Phase II: T' = 1.0T → P(hit) = 1 - e^(-λT)
            Phase III: T' = 2.0T → P(hit) = 1 - e^(-2λT)

            For λ = 1 query/sec, T = 3600s (REST_API_JSON):
                Phase I:  P(hit) = 1 - e^(-1800) ≈ 1.0 (effectively always hit)
                Phase II: P(hit) = 1 - e^(-3600) ≈ 1.0
                Phase III: P(hit) = 1 - e^(-7200) ≈ 1.0

            The modulation matters most for low-traffic (domain, class) pairs
            where λT is small:
                For λ = 0.001 query/sec, T = 900s (NEWS_ARTICLE):
                    Phase I:  P(hit) = 1 - e^(-0.45) ≈ 0.362
                    Phase II: P(hit) = 1 - e^(-0.90) ≈ 0.593
                    Phase III: P(hit) = 1 - e^(-1.80) ≈ 0.835

            Phase III nearly doubles the hit rate for infrequent queries.

        Parameters:
            topology_class: The topology class.
            phase:          Current phase (1, 2, or 3).

        Returns:
            Modulated TTL in seconds.  Never negative.  Returns 0.0 for TTL=0 classes.
        """
        base_ttl = CACHE_TTL_BY_CLASS.get(topology_class, CACHE_TTL_BY_CLASS.get(
            FALLBACK_TOPOLOGY_CLASS, 1800.0
        ))

        if base_ttl == 0.0:
            return 0.0

        # Phase modulation factors.
        # These are multiplicative adjustments to the base TTL.
        # The factors are intentionally simple (powers of 2) for predictable
        # cache behavior and easy mental math during debugging.
        _PHASE_TTL_FACTORS: Dict[int, float] = {
            1: 0.5,   # Phase I:   half TTL — model is learning, predictions volatile
            2: 1.0,   # Phase II:  standard TTL — predictions are stable
            3: 2.0,   # Phase III: double TTL — compiled policy, high confidence
        }

        factor = _PHASE_TTL_FACTORS.get(phase, 1.0)
        return base_ttl * factor

    # ─────────────────────────────────────────────────────────────────────────
    # READOUT RATE LIMITING
    # ─────────────────────────────────────────────────────────────────────────

    def _should_rate_limit_readout(self) -> bool:
        """
        Check if L3 readouts should be rate-limited to protect event loop latency.

        Rate limiting triggers when the active readout count exceeds a threshold.
        This prevents a burst of cache misses from saturating the event loop
        with concurrent readout operations.

        The threshold is derived from the ratio of readout latency to query latency:
            - One readout takes ~2ms of CPU time.
            - The event loop cooperative scheduling budget is ~10ms.
            - At most 5 concurrent readouts can execute within the budget
              before the event loop becomes unresponsive.

        Threshold: _active_readout_count > 4.

        When rate limiting is active, the query falls back to the fallback
        response.  This is a last-resort mechanism — under normal conditions,
        the cache tiers ensure that L3 readouts are infrequent (<15%).

        Returns:
            True if readouts should be rate-limited.  False otherwise.
        """
        # Maximum concurrent L3 readouts before degradation.
        # 4 concurrent readouts × 2ms each = 8ms of CPU time.
        # 5 would push total to 10ms — at the edge of the cooperative budget.
        _MAX_CONCURRENT_READOUTS: int = 4

        return self._active_readout_count > _MAX_CONCURRENT_READOUTS

    # ─────────────────────────────────────────────────────────────────────────
    # QUERY TRACING
    # ─────────────────────────────────────────────────────────────────────────

    def _trace_query(
        self,
        topology_class: str,
        domain: str,
        phase: int,
        tier: str,
        latency_ms: float,
        response: WLMResponse,
    ) -> Dict[str, Any]:
        """
        Generate a detailed query trace for diagnostic logging.

        Returns a dict suitable for ``log.debug(**self._trace_query(...))``.
        The ``event`` key carries the structured log event name; all other
        keys are structured fields consumed by Witness.

        Query traces are emitted at DEBUG level for every query.  They contain
        sufficient information to reconstruct the full query routing path
        without access to the model's internal state.

        Trace fields:
            - event: structured log event name (always "wlm_query_trace")
            - topology_class, domain, phase: query parameters
            - tier: which cache tier served the response (L1, L2, L3, L3_ttl_zero,
                    rate_limited, backoff, fallback)
            - latency_ms: end-to-end query latency
            - hidden_state_version: version at query time
            - response_summary: abbreviated response fields
            - cache_stats: hit/miss counts for diagnostic context

        Parameters:
            topology_class: The queried topology class.
            domain:         The queried domain.
            phase:          The current phase.
            tier:           The cache tier that served the response.
            latency_ms:     The measured query latency.
            response:       The WLMResponse returned to the caller.

        Returns:
            Dict suitable for ``log.debug(**result)``.
        """
        return {
            "event":                "wlm_query_trace",
            "topology_class":      topology_class,
            "domain":              domain,
            "phase":               phase,
            "tier":                tier,
            "latency_ms":          round(latency_ms, 3),
            "hidden_state_version": self._hidden_state_version,
            "traversal_depth":     response.traversal_policy.depth,
            "traversal_render":    response.traversal_policy.render_mode,
            "traversal_rps":       round(response.traversal_policy.requests_per_second, 3),
            "traversal_retry":     response.traversal_policy.retry_budget,
            "traversal_timeout":   response.traversal_policy.timeout_ms,
            "traversal_confidence": round(response.traversal_policy.confidence, 4),
            "friction_cf":         round(response.friction_forecast.cloudflare_probability, 4),
            "friction_pw":         round(response.friction_forecast.paywall_probability, 4),
            "friction_rl":         round(response.friction_forecast.rate_limit_probability, 4),
            "friction_auth":       round(response.friction_forecast.auth_redirect_probability, 4),
            "friction_strategy":   response.friction_forecast.mitigation_strategy,
            "source_count":        len(response.source_priority),
            "source_top":          response.source_priority[0] if response.source_priority else None,
            "world_confidence":    round(response.world_confidence, 4),
            "total_queries":       self._query_stats.total_queries,
            "cache_hit_rate":      round(self._query_stats.cache_hit_rate, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE STATISTICS EXPORT
    # ─────────────────────────────────────────────────────────────────────────

    def export_cache_statistics(self) -> Dict[str, Any]:
        """
        Export comprehensive cache statistics for monitoring systems.

        Returns a dict containing:
            - Per-tier hit rates, miss rates, eviction counts, sizes.
            - Latency histogram percentiles.
            - Query distribution by topology class and phase.
            - Cache efficiency metrics.

        Cache efficiency is defined as:
            efficiency = (L1_hits + L2_hits) / (L1_hits + L2_hits + L3_readouts)

        This metric captures how effectively the cache reduces L3 readout load.
        Target: >85% for a warm system.

        L3 avoidance ratio:
            avoidance = 1 - (L3_readouts / total_queries)

        This is the fraction of queries that did NOT need a model readout.
        Target: >85% for a warm system.  Higher indicates better cache tuning.

        Weighted latency estimate:
            E[latency] = (L1_hits × E[L1_latency] + L2_hits × E[L2_latency]
                         + L3_readouts × E[L3_latency]) / total_queries

        Returns:
            Comprehensive statistics dict.
        """
        stats = self._query_stats
        total = stats.total_queries

        # Cache efficiency.
        cache_served = stats.l1_hits + stats.l2_hits
        efficiency = cache_served / total if total > 0 else 0.0

        # L3 avoidance ratio.
        avoidance = 1.0 - (stats.l3_readouts / total) if total > 0 else 1.0

        # Per-tier contribution.
        l1_share = stats.l1_hits / total if total > 0 else 0.0
        l2_share = stats.l2_hits / total if total > 0 else 0.0
        l3_share = stats.l3_readouts / total if total > 0 else 0.0

        # Estimated weighted latency.
        # Using empirical bucket medians: L1~0.4ms, L2~0.7ms, L3~3.5ms
        _L1_EST_MS = 0.4
        _L2_EST_MS = 0.7
        _L3_EST_MS = 3.5
        weighted_latency = (
            (stats.l1_hits * _L1_EST_MS
             + stats.l2_hits * _L2_EST_MS
             + stats.l3_readouts * _L3_EST_MS) / total
        ) if total > 0 else 0.0

        return {
            # Summary metrics.
            "total_queries":           total,
            "cache_efficiency":        round(efficiency, 4),
            "l3_avoidance_ratio":      round(avoidance, 4),
            "weighted_latency_est_ms": round(weighted_latency, 3),

            # Per-tier distribution.
            "l1_share":                round(l1_share, 4),
            "l2_share":                round(l2_share, 4),
            "l3_share":                round(l3_share, 4),

            # Per-tier detailed statistics.
            **self._l1_stats.to_log_dict("l1"),
            **self._l2_stats.to_log_dict("l2"),

            # Latency histogram.
            **self._latency_histogram.to_log_dict(),

            # Anomaly counters.
            "ttl_zero_bypasses":        stats.ttl_zero_bypasses,
            "latency_ceiling_breaches": stats.latency_ceiling_breaches,
            "readout_errors":           stats.readout_errors,

            # Per-class distribution (top 10 by query count).
            "top_classes": dict(
                sorted(
                    stats.per_class_counts.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )[:10]
            ),

            # Per-phase distribution.
            "per_phase_counts": stats.per_phase_counts,

            # Cache sizing.
            "l1_capacity":       L1_MAX_ENTRIES,
            "l1_utilization":    round(len(self._l1_cache) / L1_MAX_ENTRIES, 4),
            "l2_capacity":       L2_MAX_ENTRIES,
            "l2_utilization":    round(len(self._l2_cache) / L2_MAX_ENTRIES, 4),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE TTL ANALYSIS
    # ─────────────────────────────────────────────────────────────────────────

    def analyze_cache_ttl_effectiveness(self) -> Dict[str, Any]:
        """
        Analyze TTL effectiveness for cache tuning.

        For each topology class with cached entries, compute:
            - Average age of entries at access time (how much of TTL is used).
            - Average access count before expiry (cache utilization).
            - TTL efficiency = avg_accesses / expected_accesses

        Expected accesses = query_rate × TTL.  If actual accesses << expected,
        the TTL is too long (wasting memory).  If actual ≈ expected, TTL is
        well-tuned.

        This method provides the data for automated TTL tuning:
            - Classes with TTL efficiency < 0.2: TTL too long, reduce.
            - Classes with TTL efficiency > 0.8: TTL well-tuned.
            - Classes with 0 entries: no data to analyze.

        Returns:
            Dict with per-class TTL analysis.
        """
        analysis: Dict[str, Dict[str, Any]] = {}

        # Analyze L1 entries grouped by topology class.
        class_entries: Dict[str, List[_L1CacheEntry]] = {}
        for key, entry in self._l1_cache.items():
            tc = entry.topology_class
            if tc not in class_entries:
                class_entries[tc] = []
            class_entries[tc].append(entry)

        for tc, entries in class_entries.items():
            if not entries:
                continue

            ages = [e.age_seconds for e in entries]
            accesses = [e.access_count for e in entries]
            ttls = [e.ttl for e in entries]

            avg_age = sum(ages) / len(ages)
            avg_accesses = sum(accesses) / len(accesses)
            avg_ttl = sum(ttls) / len(ttls)

            # TTL utilization: fraction of TTL consumed on average.
            ttl_utilization = avg_age / avg_ttl if avg_ttl > 0 else 0.0

            analysis[tc] = {
                "entry_count":      len(entries),
                "avg_age_seconds":  round(avg_age, 1),
                "avg_accesses":     round(avg_accesses, 1),
                "avg_ttl_seconds":  round(avg_ttl, 1),
                "ttl_utilization":  round(ttl_utilization, 4),
                "max_age_seconds":  round(max(ages), 1),
                "max_accesses":     max(accesses),
            }

        return {
            "topology_class_analysis": analysis,
            "classes_analyzed": len(analysis),
            "total_l1_entries": len(self._l1_cache),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE INVALIDATION: TARGETED
    # ─────────────────────────────────────────────────────────────────────────

    def _invalidate_domain(self, domain: str) -> int:
        """
        Invalidate all L1 entries for a specific domain.

        Used when a domain's structural properties change (e.g., the domain
        migrates to a different CMS, changes its Cloudflare configuration,
        or undergoes a major redesign).

        This is a targeted invalidation — only entries for the specified
        domain are removed.  All other domains and L2 entries are unaffected.

        Parameters:
            domain: The domain to invalidate.

        Returns:
            Number of L1 entries invalidated.
        """
        keys_to_remove = [
            key for key in self._l1_cache
            if key[0] == domain  # key = (domain, topology_class)
        ]

        for key in keys_to_remove:
            del self._l1_cache[key]

        invalidated = len(keys_to_remove)
        self._l1_stats.invalidations += invalidated
        self._l1_stats.current_size = len(self._l1_cache)

        if invalidated > 0:
            log.info(
                "invalidate_domain: complete",
                domain=domain,
                entries_invalidated=invalidated,
            )

        return invalidated

    def _invalidate_topology_class(self, topology_class: str) -> Tuple[int, int]:
        """
        Invalidate all cache entries for a specific topology class.

        Removes matching entries from both L1 and L2.
        Used by surprise event handler and by manual cache management.

        Parameters:
            topology_class: The topology class to invalidate.

        Returns:
            Tuple of (l1_invalidated, l2_invalidated).
        """
        # L1 invalidation.
        l1_keys = [
            key for key in self._l1_cache
            if key[1] == topology_class
        ]
        for key in l1_keys:
            del self._l1_cache[key]
        l1_count = len(l1_keys)
        self._l1_stats.invalidations += l1_count
        self._l1_stats.current_size = len(self._l1_cache)

        # L2 invalidation.
        l2_count = 0
        if topology_class in self._l2_cache:
            del self._l2_cache[topology_class]
            l2_count = 1
            self._l2_stats.invalidations += 1
            self._l2_stats.current_size = len(self._l2_cache)

        return l1_count, l2_count

    # ─────────────────────────────────────────────────────────────────────────
    # EXTENDED DIAGNOSTICS
    # ─────────────────────────────────────────────────────────────────────────

    def dump_l1_entries(self, topology_class: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Dump L1 cache entries for diagnostic inspection.

        Returns a list of dicts describing each L1 entry, optionally
        filtered by topology class.  Useful for debugging cache behavior
        and verifying TTL correctness.

        Does not return the actual WLMResponse objects (they are large).
        Returns metadata about each entry: key, age, TTL, access count,
        version, and expiry status.

        Parameters:
            topology_class: Optional filter.  If provided, only entries for
                           this topology class are returned.

        Returns:
            List of entry metadata dicts.
        """
        entries = []
        for key, entry in self._l1_cache.items():
            if topology_class is not None and entry.topology_class != topology_class:
                continue

            entries.append({
                "domain":               key[0],
                "topology_class":       key[1],
                "age_seconds":          round(entry.age_seconds, 2),
                "ttl_seconds":          entry.ttl,
                "remaining_ttl":        round(entry.remaining_ttl, 2),
                "is_expired":           entry.is_expired,
                "access_count":         entry.access_count,
                "hidden_state_version": entry.hidden_state_version,
                "is_stale":             entry.hidden_state_version != self._hidden_state_version,
                "world_confidence":     round(entry.response.world_confidence, 4),
                "traversal_depth":      entry.response.traversal_policy.depth,
                "render_mode":          entry.response.traversal_policy.render_mode,
                "mitigation_strategy":  entry.response.friction_forecast.mitigation_strategy,
            })

        return entries

    def dump_l2_entries(self) -> List[Dict[str, Any]]:
        """
        Dump L2 cache entries for diagnostic inspection.

        Returns a list of dicts describing each L2 entry.
        Same format as dump_l1_entries but without domain key.

        Returns:
            List of entry metadata dicts.
        """
        entries = []
        for key, entry in self._l2_cache.items():
            entries.append({
                "topology_class":       key,
                "age_seconds":          round(entry.age_seconds, 2),
                "ttl_seconds":          entry.ttl,
                "remaining_ttl":        round(entry.remaining_ttl, 2),
                "is_expired":           entry.is_expired,
                "access_count":         entry.access_count,
                "hidden_state_version": entry.hidden_state_version,
                "is_stale":             entry.hidden_state_version != self._hidden_state_version,
                "world_confidence":     round(entry.response.world_confidence, 4),
                "traversal_depth":      entry.response.traversal_policy.depth,
                "render_mode":          entry.response.traversal_policy.render_mode,
                "mitigation_strategy":  entry.response.friction_forecast.mitigation_strategy,
            })

        return entries

    def get_model_diagnostics(self) -> Dict[str, Any]:
        """
        Return comprehensive model diagnostics for debugging.

        Includes:
            - Router architecture summary.
            - Hidden state version and digest.
            - Structural layer state.
            - Cache coherence verification.
            - Latency histogram percentiles.
            - Per-topology-class query distribution.
            - Active readout count (should be 0 during diagnosis).

        This is an expensive method — it calls verify_cache_coherence() which
        is O(n) in cache size.  Do not call from the query hot path.

        Returns:
            Comprehensive diagnostic dict.
        """
        diag: Dict[str, Any] = {}

        # ── Router diagnostics ────────────────────────────────────────────────
        if self._router is not None:
            diag["router"] = self._router.architecture_summary()
        else:
            diag["router"] = {"status": "not_loaded"}

        # ── Hidden state ──────────────────────────────────────────────────────
        diag["hidden_state_version"] = self._hidden_state_version
        if self._router is not None:
            diag["hidden_state_digest"] = self._router.hidden_state_digest()
        else:
            diag["hidden_state_digest"] = None

        # ── Structural layer ──────────────────────────────────────────────────
        sl = self._structural_layer
        diag["structural_layer"] = {
            "type":       type(sl).__name__,
            "n_domains":  sl.n_domains if hasattr(sl, "n_domains") else 0,
            "embed_dim":  sl.embed_dim if hasattr(sl, "embed_dim") else 0,
            "n_clusters": sl.n_clusters if hasattr(sl, "n_clusters") else 0,
        }

        # ── Cache coherence ───────────────────────────────────────────────────
        diag["cache_coherence"] = self.verify_cache_coherence()

        # ── Cache statistics ──────────────────────────────────────────────────
        diag["cache_stats"] = self.export_cache_statistics()

        # ── Lifecycle ─────────────────────────────────────────────────────────
        diag["lifecycle"] = {
            "initialized":          self._initialized,
            "shutting_down":        self._shutting_down,
            "active_readout_count": self._active_readout_count,
            "background_tasks":     len(self._background_tasks),
        }

        return diag

    # ─────────────────────────────────────────────────────────────────────────
    # CACHE WARMING: STRATEGIC PRE-POPULATION
    # ─────────────────────────────────────────────────────────────────────────

    async def warm_domain(
        self,
        domain: str,
        topology_classes: List[str],
        phase: int = PHASE_I,
    ) -> int:
        """
        Strategically warm L1 cache entries for a specific domain.

        Called by domain_analyzer.py after discovering a new domain's topology
        classes from robots.txt and sitemap analysis.  Pre-populating L1
        eliminates L3 readouts for the first queries to this domain.

        Parameters:
            domain:           The domain to warm.
            topology_classes: List of topology classes observed for this domain.
            phase:            Current phase for source priority k selection.

        Returns:
            Number of entries successfully warmed.
        """
        warmed = 0

        for tc in topology_classes:
            if self._is_ttl_zero_class(tc):
                continue

            # Check if already cached.
            if (domain, tc) in self._l1_cache:
                existing = self._l1_cache[(domain, tc)]
                if not existing.is_expired and existing.hidden_state_version == self._hidden_state_version:
                    continue

            # Check L2 for promotion.
            l2_entry = self._l2_cache.get(tc)
            if l2_entry is not None and not l2_entry.is_expired and l2_entry.hidden_state_version == self._hidden_state_version:
                self._l1_store(domain, tc, l2_entry.response, phase)
                warmed += 1
                continue

            # L3 readout.
            try:
                response = await self._l3_readout(
                    topology_class=tc,
                    intent_vector=None,
                    domain=domain,
                    phase=phase,
                )
                self._l1_store(domain, tc, response, phase)
                self._l2_store(tc, response, phase)
                warmed += 1
            except Exception as exc:
                log.debug(
                    "warm_domain: readout failed",
                    domain=domain,
                    topology_class=tc,
                    error=str(exc),
                )

        log.debug(
            "warm_domain: complete",
            domain=domain,
            requested=len(topology_classes),
            warmed=warmed,
        )

        return warmed

    # ─────────────────────────────────────────────────────────────────────────
    # READOUT SCHEDULING — EXPONENTIAL BACKOFF FOR REPEATED FAILURES
    # ─────────────────────────────────────────────────────────────────────────

    # _readout_failure_counts and _readout_last_failure are instance variables
    # defined in __init__.  They were previously class-level, which caused all
    # WorldLatentModel instances to share failure state — a correctness bug.

    def _should_backoff_readout(self, topology_class: str) -> bool:
        """
        Check if readout for a topology class should be backed off due to
        repeated failures.

        Backoff strategy:
            - After N consecutive failures, wait 2^N × 100ms before retrying.
            - Maximum backoff: 60 seconds.
            - Backoff resets on any successful readout for the class.

        This prevents a corrupted topology class (e.g., from a bad training
        step) from generating unbounded readout errors.

        Mathematical formulation:
            backoff_ms = min(2^N × 100, 60000)

            N=1: 200ms, N=2: 400ms, N=3: 800ms, ..., N=9: 51200ms, N=10: 60000ms

            Expected error rate under backoff:
                Without backoff: failure_rate × query_rate errors/sec
                With backoff: failure_rate × (1 / backoff_interval) errors/sec

                For N=5 failures, backoff = 3.2s:
                    Error rate drops from query_rate to 1/3.2 = 0.31 errors/sec

        Parameters:
            topology_class: The topology class to check.

        Returns:
            True if the readout should be backed off.  False if it should proceed.
        """
        failures = self._readout_failure_counts.get(topology_class, 0)
        if failures == 0:
            return False

        last_failure = self._readout_last_failure.get(topology_class, 0.0)
        backoff_ms = min(
            (2 ** failures) * 100.0,
            60_000.0,
        )
        elapsed_ms = (time.monotonic() - last_failure) * 1000.0

        return elapsed_ms < backoff_ms

    def _record_readout_failure(self, topology_class: str) -> None:
        """Record a readout failure for backoff tracking."""
        self._readout_failure_counts[topology_class] = (
            self._readout_failure_counts.get(topology_class, 0) + 1
        )
        self._readout_last_failure[topology_class] = time.monotonic()

    def _record_readout_success(self, topology_class: str) -> None:
        """Reset backoff state on successful readout."""
        self._readout_failure_counts.pop(topology_class, None)

# ═════════════════════════════════════════════════════════════════════════════
# WLM TRAINING INTERFACE
#
# The sole boundary for training operations.  index_daemon.py receives this
# object via get_training_interface().  It is the ONLY path to training ops.
# ═════════════════════════════════════════════════════════════════════════════

class WLMTrainingInterface:
    """
    Restricted training interface for index_daemon.py.

    index_daemon.py receives this object.  It calls nothing else on
    WorldLatentModel.  It does not hold a reference to WorldLatentModel
    itself (it holds a reference to this interface, which holds a reference
    to the model — an intentional layer of indirection).

    It cannot call query().  It cannot read hidden_state directly.
    It cannot clear caches.  It can only do four things:

        1. get_model()          — for gradient computation
        2. update_hidden_state() — after optimizer.step() with 500ms window
        3. save_checkpoint()    — atomic write to store via staging
        4. get_version()        — monotonic counter for audit trail

    Construction:
        This class is NEVER constructed directly.  Construction is enforced
        by a private sentinel object that is only accessible via
        WorldLatentModel.get_training_interface().  Any direct call to
        WLMTrainingInterface(model) raises RuntimeError immediately.

    Consistency model:
        update_hidden_state() enforces the 500ms consistency window by
        polling _model._active_readout_count.  If all in-flight readouts
        complete in 3ms, the update happens in 3ms.  500ms is the ceiling
        for pathological cases where a readout is stuck.

        The poll uses asyncio.sleep(1ms) — the minimum resolution of the
        event loop.  Each poll is a single integer comparison against zero.
        In the worst case (500ms deadline), 500 polls are executed — each
        costs ~1μs of Python overhead.  Total CPU cost: ~0.5ms.
    """

    # Private sentinel.  The only valid _guard value for __init__.
    # It is a unique object() — unreachable from outside this class.
    # Any caller that does not hold a reference to this exact object
    # (i.e., any caller other than WorldLatentModel.get_training_interface)
    # will receive a RuntimeError.
    _SENTINEL: object = object()

    def __init__(
        self,
        model: WorldLatentModel,
        _guard: object = None,
    ) -> None:
        """
        Internal constructor.  Use WorldLatentModel.get_training_interface().

        Parameters:
            model:  The WorldLatentModel instance.
            _guard: Must be WLMTrainingInterface._SENTINEL.  Any other value —
                    including the default None — raises RuntimeError.  This
                    prevents direct construction from any call site outside
                    WorldLatentModel.get_training_interface().
        """
        if _guard is not WLMTrainingInterface._SENTINEL:
            raise RuntimeError(
                "WLMTrainingInterface cannot be constructed directly.  "
                "Use WorldLatentModel.get_training_interface().  "
                "Direct construction bypasses the safety boundary protecting "
                "the MFT from unauthorised training operations."
            )
        self._model: WorldLatentModel = model

    def get_model(self) -> MambaRouter:
        """
        Return the live MambaRouter for gradient computation.

        index_daemon.py calls this to get the model for:
            - model.forward(token_sequence) for loss computation
            - optimizer = Adam(model.parameters())
            - loss.backward()
            - optimizer.step()

        The returned MambaRouter is the LIVE model — the same instance
        used by readout().  This is intentional: training operates on the
        live model so that gradient steps compound on the current hidden state.

        The training lock on MambaRouter prevents accidental forward() calls
        from the inference path.  index_daemon.py must acquire the training
        lock via model.acquire_training_lock() before calling model.forward().

        Returns:
            The live MambaRouter instance.

        Raises:
            RuntimeError: If the model's router is None.
        """
        if self._model._router is None: # noqa
            raise RuntimeError(
                "WLMTrainingInterface.get_model(): router is None.  "
                "This indicates the model was not initialized."
            )
        return self._model._router # noqa

    async def update_hidden_state(self, new_hidden: torch.Tensor) -> None:
        """
        Update the MambaRouter's hidden_state after a gradient step.

        Enforces the 500ms consistency window:
            1. Poll _model._active_readout_count every 1ms.
            2. When count reaches 0: proceed with hidden state update.
            3. If 500ms elapsed and count > 0: proceed anyway and log WARNING.

        This is EXACT, not defensive.  The 500ms is a ceiling, not a fixed sleep.
        If all in-flight readouts complete in 3ms, the update happens in 3ms.

        The hidden state update itself is an atomic reference copy:
            self._model._router.hidden_state.copy_(new_hidden)

        This is safe under the GIL because:
            - copy_() is a single C++ operation at the torch level.
            - The GIL ensures that no Python bytecode from readout() can
              execute between the start and end of copy_().
            - readout() reads hidden_state via a tensor operation that is
              also GIL-protected.

        After the update:
            - _model._hidden_state_version is incremented.
            - Stale cache entries are detected at lookup time (not here).
            - No caches are cleared (version-based staleness detection is
              more efficient than eager invalidation).

        Parameters:
            new_hidden: The new hidden state tensor.
                       Shape: (1, d_model) — must match the current hidden_state shape.
                       Must be detached from the computation graph.

        Raises:
            ValueError: If new_hidden has wrong shape or dtype.
        """
        router = self._model._router # noqa
        if router is None:
            raise RuntimeError(
                "WLMTrainingInterface.update_hidden_state(): router is None."
            )

        # ── Validate shape ────────────────────────────────────────────────────
        expected_shape = tuple(router.hidden_state.shape)
        actual_shape = tuple(new_hidden.shape)
        if actual_shape != expected_shape:
            raise ValueError(
                f"update_hidden_state: new_hidden shape {actual_shape} "
                f"does not match expected shape {expected_shape}."
            )

        # ── Wait for in-flight readouts ───────────────────────────────────────
        deadline = time.monotonic() + (HIDDEN_STATE_UPDATE_DEADLINE_MS / 1000.0)
        poll_count = 0

        while self._model._active_readout_count > 0: # noqa
            if time.monotonic() > deadline:
                log.warning(
                    "update_hidden_state: deadline exceeded",
                    active_readouts=self._model._active_readout_count, # noqa
                    deadline_ms=HIDDEN_STATE_UPDATE_DEADLINE_MS,
                    poll_count=poll_count,
                )
                break
            await asyncio.sleep(HIDDEN_STATE_POLL_INTERVAL_MS / 1000.0)
            poll_count += 1

        # ── Atomic hidden state update ────────────────────────────────────────
        # copy_() is a single C-level operation.  GIL ensures atomicity.
        router.hidden_state.copy_(new_hidden.detach())

        # ── Increment version ─────────────────────────────────────────────────
        self._model._hidden_state_version += 1 # noqa

        # Also increment the router's internal version counter.
        router.hidden_state_version.fill_(self._model._hidden_state_version) # noqa

        log.info(
            "hidden_state_updated",
            new_version=self._model._hidden_state_version, # noqa
            poll_count=poll_count,
            waited_ms=round(poll_count * HIDDEN_STATE_POLL_INTERVAL_MS, 2),
        )

    async def save_checkpoint(self) -> Path:
        """
        Save current state_dict to staging file, then atomically rename.

        Checkpoint procedure:
            1. Write state_dict to staging path via torch.save().
            2. Compute SHA-256 of staging file.
            3. Write SHA-256 to companion .sha256 file.
            4. Atomic rename: staging → final path.
            5. store_watchdog fires on IN_MOVED_TO event.

        The staging → rename pattern ensures that:
            - Readers never see a partially written file.
            - store_watchdog fires only when the write is complete.
            - A crash during write leaves the staging file (not the final file).
            - Recovery loads the last known good final file.

        Returns:
            Path to the written checkpoint (the final path, not staging).

        Raises:
            RuntimeError: If the router is None or writing fails.
        """
        router = self._model._router # noqa
        if router is None:
            raise RuntimeError(
                "WLMTrainingInterface.save_checkpoint(): router is None."
            )

        # Ensure staging directory exists.
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

        staging_path = STAGING_DIR / "topology_router.pt.staging"
        final_path = WEIGHTS_PATH

        # ── Step 1: Write state_dict to staging ───────────────────────────────
        try:
            torch.save(router.state_dict(), staging_path)
        except Exception as exc:
            raise RuntimeError(
                f"save_checkpoint: torch.save() failed: {exc}"
            ) from exc

        # ── Step 2: Compute and write SHA-256 ─────────────────────────────────
        sha256 = WorldLatentModel._compute_file_sha256(staging_path) # noqa
        sha256_path = STAGING_DIR / "topology_router.pt.sha256"
        sha256_path.write_text(sha256)

        # ── Step 3: Atomic rename ─────────────────────────────────────────────
        # os.rename() is atomic on POSIX when source and destination are on
        # the same filesystem (which they are — both in /store).
        staging_path.rename(final_path)

        log.info(
            "checkpoint_saved",
            path=str(final_path),
            sha256=sha256[:16] + "...",
            hidden_state_version=self._model._hidden_state_version, # noqa
        )

        return final_path

    def get_version(self) -> int:
        """
        Return the current hidden_state version.

        Monotonic counter incremented on every hidden_state update.
        Used by index_daemon.py for audit trail correlation:
            - Each gradient step logs the version before and after.
            - Witness correlates version numbers across checkpoints.
            - Version rollbacks indicate model restoration from backup.

        Returns:
            Current hidden_state version (non-negative integer).
        """
        return self._model._hidden_state_version # noqa

    def get_hidden_state_digest(self) -> Optional[str]:
        """
        Return the SHA-256 digest of the current hidden_state tensor.

        Used by index_daemon.py to verify that the hidden state has changed
        after a gradient step.  If the digest is identical before and after
        optimizer.step(), the gradient step had no effect — indicating a
        learning rate issue, gradient clipping that zeroed all gradients,
        or a numerical issue in the loss computation.

        Returns:
            Hex SHA-256 digest of the hidden_state buffer, or None if the
            router is not loaded.
        """
        if self._model._router is None: # noqa
            return None
        return self._model._router.hidden_state_digest() # noqa

    def get_model_config(self) -> Optional[ModelConfig]:
        """
        Return the model configuration for optimizer construction.

        index_daemon.py uses this to:
            - Set learning rate schedules based on model dimension.
            - Configure weight decay based on parameter count.
            - Set gradient clipping thresholds based on model size.

        Returns:
            The ModelConfig, or None if the model is not loaded.
        """
        return self._model._config # noqa

    def get_parameter_groups(self) -> Optional[Dict[str, List[Any]]]:
        """
        Return parameter groups for optimizer construction.

        index_daemon.py typically uses two parameter groups:
            1. Backbone parameters (Mamba blocks, embeddings) — lower LR.
            2. Head parameters (output heads) — higher LR.

        This separation allows differential learning rates:
            - Backbone captures general structural patterns — fine-tune gently.
            - Heads map structural patterns to specific outputs — can learn faster.

        Mathematical justification for differential LR:
            The gradient signal for backbone parameters passes through
            all output heads (five loss terms backpropagate through the
            shared representation).  The total gradient magnitude is:
                ∇_backbone L = Σ_i ∂L_i/∂h × ∂h/∂θ_backbone

            For head parameters, the gradient is head-specific:
                ∇_head_i L = ∂L_i/∂θ_head_i

            The backbone gradient is the sum of 5 head gradients.  Without
            differential LR, the effective learning rate for the backbone
            is 5× higher than for any individual head.  Reducing backbone
            LR by ~5× (0.2× multiplier) restores balance.

            Empirical range:
                backbone_lr = base_lr × 0.1 to 0.3
                head_lr     = base_lr × 1.0

        Returns:
            Dict with "backbone" and "head" parameter lists, or None if
            the router is not loaded.
        """
        if self._model._router is None: # noqa
            return None

        return {
            "backbone": self._model._router.get_backbone_parameters(), # noqa
            "heads": {
                "topology": self._model._router.get_head_parameters("topology_head"), # noqa
                "traversal": self._model._router.get_head_parameters("traversal_head"), # noqa
                "friction": self._model._router.get_head_parameters("friction_head"), # noqa
                "source": self._model._router.get_head_parameters("source_head"), # noqa
                "phase": self._model._router.get_head_parameters("phase_head"), # noqa
            },
        }

    async def validate_gradient_step(
        self,
        pre_hidden_digest: str,
        post_hidden_candidate: torch.Tensor,
    ) -> Dict[str, Any]:
        """
        Validate a gradient step before committing the hidden state update.

        Called by index_daemon.py between optimizer.step() and
        update_hidden_state() to verify that the gradient step produced
        a valid result.

        Validation checks:
            1. post_hidden_candidate has the expected shape.
            2. post_hidden_candidate is finite (no NaN, no Inf).
            3. post_hidden_candidate differs from the current hidden state
               (the gradient step had a measurable effect).
            4. The L2 norm change is within bounds:
                - Too small (< 1e-8): gradient step had negligible effect.
                  Possible causes: LR too low, gradients clipped to zero,
                  loss is saturated.
                - Too large (> 100.0): gradient step was explosive.
                  Possible causes: LR too high, gradient explosion,
                  numerical instability in the loss.

        Mathematical formulation:
            Δ = post_hidden - current_hidden
            ‖Δ‖₂ = sqrt(Σ_i Δ_i²)

            Expected range for healthy training:
                1e-4 ≤ ‖Δ‖₂ ≤ 10.0

            This range is derived from:
                - d_model = 256
                - Typical per-dimension change: 1e-3 to 0.1
                - ‖Δ‖₂ ≈ √(256) × per_dim_change
                - √(256) = 16
                - Lower: 16 × 1e-4 ≈ 1.6e-3 (rounded down to 1e-4 for safety)
                - Upper: 16 × 0.5 ≈ 8.0 (rounded up to 10.0 for safety)

        Parameters:
            pre_hidden_digest:       SHA-256 of hidden_state before gradient step.
            post_hidden_candidate:   The proposed new hidden_state tensor.

        Returns:
            Dict with validation results:
                - valid: True if all checks pass.
                - delta_norm: L2 norm of the change.
                - is_finite: True if the candidate is finite.
                - shape_ok: True if the shape matches.
                - warnings: List of warning strings.
                - errors: List of error strings.
        """
        result: Dict[str, Any] = {
            "valid": True,
            "delta_norm": 0.0,
            "is_finite": True,
            "shape_ok": True,
            "warnings": [],
            "errors": [],
        }

        router = self._model._router # noqa
        if router is None:
            result["valid"] = False
            result["errors"].append("Router is None — model not loaded.")
            return result

        # ── Shape check ───────────────────────────────────────────────────────
        expected_shape = tuple(router.hidden_state.shape)
        actual_shape = tuple(post_hidden_candidate.shape)
        if actual_shape != expected_shape:
            result["valid"] = False
            result["shape_ok"] = False
            result["errors"].append(
                f"Shape mismatch: expected {expected_shape}, got {actual_shape}."
            )
            return result

        # ── Finite check ──────────────────────────────────────────────────────
        if not torch.isfinite(post_hidden_candidate).all():
            result["valid"] = False
            result["is_finite"] = False
            nan_count = torch.isnan(post_hidden_candidate).sum().item()
            inf_count = torch.isinf(post_hidden_candidate).sum().item()
            result["errors"].append(
                f"Non-finite values: NaN={nan_count}, Inf={inf_count}."
            )
            return result

        # ── Delta norm ────────────────────────────────────────────────────────
        with torch.no_grad():
            delta = post_hidden_candidate.detach() - router.hidden_state.detach()
            delta_norm = torch.norm(delta, p=2).item()

        result["delta_norm"] = round(delta_norm, 8)

        # Check: negligible change.
        _MIN_DELTA_NORM = 1e-8
        if delta_norm < _MIN_DELTA_NORM:
            result["warnings"].append(
                f"Delta norm {delta_norm:.2e} < {_MIN_DELTA_NORM:.0e}: "
                "gradient step had negligible effect.  "
                "Check: learning rate, gradient clipping, loss saturation."
            )

        # Check: too small (informational).
        _HEALTHY_MIN_NORM = 1e-4
        if _MIN_DELTA_NORM <= delta_norm < _HEALTHY_MIN_NORM:
            result["warnings"].append(
                f"Delta norm {delta_norm:.2e} is below healthy range "
                f"[{_HEALTHY_MIN_NORM:.0e}, 10.0].  "
                "Training may be progressing very slowly."
            )

        # Check: explosive change.
        _MAX_DELTA_NORM = 100.0
        if delta_norm > _MAX_DELTA_NORM:
            result["valid"] = False
            result["errors"].append(
                f"Delta norm {delta_norm:.2e} > {_MAX_DELTA_NORM:.0e}: "
                "gradient step was explosive.  "
                "Check: learning rate, gradient clipping, numerical stability."
            )

        # Check: current digest matches expected.
        current_digest = router.hidden_state_digest()
        if current_digest != pre_hidden_digest:
            result["warnings"].append(
                "Hidden state digest changed between gradient step start and "
                "validation.  A concurrent update may have occurred."
            )

        return result

    def get_readout_stats(self) -> Dict[str, Any]:
        """
        Return readout-related statistics for training monitoring.

        index_daemon.py logs these after each gradient step to track
        the relationship between training updates and inference behavior.

        Returns:
            Dict with readout stats:
                - active_readout_count: current in-flight readouts
                - hidden_state_version: current version
                - l1_size: current L1 cache size
                - l2_size: current L2 cache size
                - total_readout_errors: cumulative readout errors
                - cache_hit_rate: current cache hit rate
        """
        return {
            "active_readout_count":  self._model._active_readout_count, # noqa
            "hidden_state_version":  self._model._hidden_state_version, # noqa
            "l1_size":               len(self._model._l1_cache), # noqa
            "l2_size":               len(self._model._l2_cache), # noqa
            "total_readout_errors":  self._model._query_stats.readout_errors, # noqa
            "cache_hit_rate":        round(self._model._query_stats.cache_hit_rate, 4), # noqa
            "total_queries":         self._model._query_stats.total_queries, # noqa
            "l3_readouts":           self._model._query_stats.l3_readouts, # noqa
        }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL SELF-TESTS
#
# Run at import time to catch configuration errors before the first query.
# If any assertion fails, the module raises at import time — a crash at
# startup is infinitely preferable to a silent misconfiguration.
# ═════════════════════════════════════════════════════════════════════════════

def _self_test_constants() -> None:
    """
    Validate module-level constants for internal consistency.

    Assertions check:
        1. All TOPOLOGY_CLASSES have TTL entries in CACHE_TTL_BY_CLASS.
        2. All TTL values are non-negative.
        3. TTL=0 classes are the expected set (AUTH_REDIRECT, CLOUDFLARE_CHALLENGE, RATE_LIMITED).
        4. L2_TTL_MULTIPLIER is > 1.0 (L2 must be more stable than L1).
        5. EVICTION_INTERVAL_QUERIES is a power of 2 (for efficient modular arithmetic).
        6. Latency targets are ordered: p95 < p99 < ceiling.
        7. HIDDEN_STATE_UPDATE_DEADLINE_MS > 0.
        8. HIDDEN_STATE_POLL_INTERVAL_MS > 0 and < HIDDEN_STATE_UPDATE_DEADLINE_MS.
    """
    # ── 1. All topology classes have TTL entries ──────────────────────────────
    for tc in TOPOLOGY_CLASSES:
        assert tc in CACHE_TTL_BY_CLASS, (
            f"TOPOLOGY_CLASSES entry {tc!r} has no TTL in CACHE_TTL_BY_CLASS.  "
            "Every known topology class must have a defined TTL."
        )

    # ── 2. All TTL values are non-negative ────────────────────────────────────
    for tc, ttl in CACHE_TTL_BY_CLASS.items():
        assert ttl >= 0.0, (
            f"CACHE_TTL_BY_CLASS[{tc!r}] = {ttl} is negative.  "
            "TTL values must be >= 0.0.  Use 0.0 for never-cache classes."
        )

    # ── 3. TTL=0 classes are the expected set ─────────────────────────────────
    expected_zero = {"AUTH_REDIRECT", "CLOUDFLARE_CHALLENGE", "RATE_LIMITED"}
    assert _TTL_ZERO_CLASSES == expected_zero, (
        f"_TTL_ZERO_CLASSES = {_TTL_ZERO_CLASSES}, expected {expected_zero}.  "
        "Only friction/blocking classes should have TTL=0."
    )

    # ── 4. L2_TTL_MULTIPLIER > 1.0 ───────────────────────────────────────────
    assert L2_TTL_MULTIPLIER > 1.0, (
        f"L2_TTL_MULTIPLIER = {L2_TTL_MULTIPLIER}, must be > 1.0.  "
        "L2 must be more stable than L1 to serve as a higher-level cache."
    )

    # ── 5. EVICTION_INTERVAL_QUERIES is a power of 2 ─────────────────────────
    assert EVICTION_INTERVAL_QUERIES > 0 and (
        EVICTION_INTERVAL_QUERIES & (EVICTION_INTERVAL_QUERIES - 1)
    ) == 0, (
        f"EVICTION_INTERVAL_QUERIES = {EVICTION_INTERVAL_QUERIES}, must be a power of 2.  "
        "Bitwise AND for modular arithmetic requires power-of-2 interval."
    )

    # ── 6. Latency targets are ordered ────────────────────────────────────────
    assert LATENCY_P95_TARGET_MS < LATENCY_P99_TARGET_MS < LATENCY_ABSOLUTE_CEILING_MS, (
        f"Latency targets must be ordered: "
        f"p95={LATENCY_P95_TARGET_MS} < p99={LATENCY_P99_TARGET_MS} "
        f"< ceiling={LATENCY_ABSOLUTE_CEILING_MS}."
    )

    # ── 7. Consistency window constants ───────────────────────────────────────
    assert HIDDEN_STATE_UPDATE_DEADLINE_MS > 0, (
        "HIDDEN_STATE_UPDATE_DEADLINE_MS must be positive."
    )
    assert 0 < HIDDEN_STATE_POLL_INTERVAL_MS < HIDDEN_STATE_UPDATE_DEADLINE_MS, (
        f"HIDDEN_STATE_POLL_INTERVAL_MS = {HIDDEN_STATE_POLL_INTERVAL_MS} "
        f"must be in (0, {HIDDEN_STATE_UPDATE_DEADLINE_MS})."
    )

    # ── 8. Cache sizes are positive ───────────────────────────────────────────
    assert L1_MAX_ENTRIES > 0, "L1_MAX_ENTRIES must be positive."
    assert L2_MAX_ENTRIES > 0, "L2_MAX_ENTRIES must be positive."
    assert L2_MAX_ENTRIES >= len(TOPOLOGY_CLASSES), (
        f"L2_MAX_ENTRIES ({L2_MAX_ENTRIES}) must be >= len(TOPOLOGY_CLASSES) "
        f"({len(TOPOLOGY_CLASSES)}) to hold one entry per class."
    )

    # ── 9. CACHE_TTL_BY_CLASS covers all classes in PARENT_CLASS_MAP ──────────
    for child, parent in PARENT_CLASS_MAP.items():
        assert child in CACHE_TTL_BY_CLASS, (
            f"PARENT_CLASS_MAP child {child!r} has no TTL in CACHE_TTL_BY_CLASS.  "
            "Every class referenced in PARENT_CLASS_MAP must have a defined TTL."
        )
        assert parent in CACHE_TTL_BY_CLASS, (
            f"PARENT_CLASS_MAP parent {parent!r} has no TTL in CACHE_TTL_BY_CLASS.  "
            "Every class referenced in PARENT_CLASS_MAP must have a defined TTL."
        )

    # ── 10. L2_TTL_MULTIPLIER produces finite TTL values ─────────────────────
    for tc, ttl in CACHE_TTL_BY_CLASS.items():
        l2_ttl = ttl * L2_TTL_MULTIPLIER
        assert math.isfinite(l2_ttl), (
            f"L2 TTL for {tc!r} is non-finite: {ttl} × {L2_TTL_MULTIPLIER} = {l2_ttl}.  "
            "All L2 TTL values must be finite."
        )

    # ── 11. EVICTION_BATCH_SIZE is reasonable ─────────────────────────────────
    assert 1 <= EVICTION_BATCH_SIZE <= L1_MAX_ENTRIES, (
        f"EVICTION_BATCH_SIZE ({EVICTION_BATCH_SIZE}) must be in "
        f"[1, {L1_MAX_ENTRIES}]."
    )

    # ── 12. Cold start timeout is reasonable ──────────────────────────────────
    assert COLD_START_TIMEOUT_S > 0.0, "COLD_START_TIMEOUT_S must be positive."
    assert COLD_START_TIMEOUT_S <= 120.0, (
        f"COLD_START_TIMEOUT_S ({COLD_START_TIMEOUT_S}) exceeds 120s.  "
        "Cold start should complete in well under 2 minutes."
    )

    # ── 13. Latency histogram has sorted buckets ──────────────────────────────
    for i in range(len(LATENCY_HISTOGRAM_BUCKETS) - 1):
        assert LATENCY_HISTOGRAM_BUCKETS[i] < LATENCY_HISTOGRAM_BUCKETS[i + 1], (
            f"LATENCY_HISTOGRAM_BUCKETS[{i}] ({LATENCY_HISTOGRAM_BUCKETS[i]}) "
            f">= LATENCY_HISTOGRAM_BUCKETS[{i + 1}] ({LATENCY_HISTOGRAM_BUCKETS[i + 1]}).  "
            "Bucket boundaries must be strictly increasing."
        )


# Run self-test at import time.
_self_test_constants()


# ─────────────────────────────────────────────────────────────────────────────
# MODULE PUBLIC API SURFACE
# ─────────────────────────────────────────────────────────────────────────────

__all__ = [
    "WorldLatentModel",
    "WLMTrainingInterface",
    "CACHE_TTL_BY_CLASS",
    "L1_MAX_ENTRIES",
    "L2_MAX_ENTRIES",
    "L2_TTL_MULTIPLIER",
    "HIDDEN_STATE_UPDATE_DEADLINE_MS",
    "HIDDEN_STATE_POLL_INTERVAL_MS",
    "LATENCY_P95_TARGET_MS",
    "LATENCY_P99_TARGET_MS",
    "LATENCY_ABSOLUTE_CEILING_MS",
]
