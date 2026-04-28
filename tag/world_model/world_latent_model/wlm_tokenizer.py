"""
world_model/wlm_tokenizer.py
============================
The complete vocabulary definition and tokenization engine for the World Latent
Model (WLM).

This file owns the full 8,192-token vocabulary.  It owns every encoding path.
It owns padding, truncation, and sequence assembly.  Nothing outside this file
makes tokenization decisions.

Vocabulary layout (8,192 tokens, indices 0–8,191):
───────────────────────────────────────────────────────────────────────────────
  Index 0          : PAD_TOKEN — reserved for padding, NEVER a vocabulary token
  Indices  1 –  18 : topology class tokens       (18 tokens)
  Indices 19 – 1023: structural primitive tokens (1,005 tokens: 64 named + 1
                       UNKNOWN + 940 reserved for future expansion)
  Indices 1024 – 4095: domain hash tokens        (3,072 collision-tolerant
                       buckets; same domain → same token across restarts)
  Indices 4096 – 8191: intent signal tokens      (4,096 slots; a 256-dim
                       float32 vector quantizes to 43 tokens via 6-dim chunks)
───────────────────────────────────────────────────────────────────────────────

PAD_TOKEN Collision Resolution
───────────────────────────────
The specification originally placed topology class tokens at indices 0–17,
meaning NEWS_ARTICLE would occupy token 0 — the same index as the conventional
padding sentinel.  A padding token that collides with a live vocabulary token
corrupts the attention mask: the SSM cannot distinguish real structure from
silence.

Decision: topology class tokens are shifted by +1 to occupy indices 1–18.
Token 0 is permanently reserved as PAD_TOKEN and is NEVER assigned to any
vocabulary item.  All code that tests ``token == PAD_TOKEN`` depends on this
guarantee.  If the topology class offset is ever changed, the entire MFT is
invalidated — every domain event encoded before the change maps to the wrong
token.

Consequences of this shift:
  TOPOLOGY_TOKEN_OFFSET  = 1   (was 0 in the naive layout)
  STRUCTURAL_TOKEN_OFFSET = 19  (was 18 in the naive layout)
  Structural slots: 19–1023 = 1,005 (not 1,006 as in the naive layout)
  Total active vocabulary: 8,191 tokens occupying indices 1–8,191
  Total VOCAB_SIZE including PAD: 8,192

Intent Quantization Thresholds
────────────────────────────────
INTENT_QUANT_THRESHOLDS defines the single source of truth for how float32
intent dimensions are discretized to 2-bit levels.  Changing these thresholds
invalidates the entire MFT: every intent-encoded sequence stored in any
checkpoint before the change maps to different tokens after the change.
Treat a threshold change as a hard incompatibility requiring a full re-index.

Domain Hash Stability
──────────────────────
Domain tokens are computed with MD5, not Python's built-in hash().  Python's
hash() is randomized per process via PYTHONHASHSEED.  MD5 is deterministic
across all process restarts, machines, and Python versions.  A domain token
that changes between process restarts would silently corrupt the MFT — the
model would learn stripe.com under token X in one run and token Y in the next.
MD5 collision resistance is irrelevant; hash token collisions are a feature.

Dependency constraints:
  Imports: contracts.py (for TOPOLOGY_CLASSES) — no other world_model imports
  Must not import: mamba_router.py, wlm_decoders.py, latent_model.py
  Must not: make network calls, read files, store state between calls
  Must not: use Python's hash() — PYTHONHASHSEED randomises it

AXIOM INTERNAL // DO NOT SURFACE
"""

from __future__ import annotations

import hashlib
import logging
import math
import struct
from typing import (
    Dict,
    FrozenSet,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
)

import torch

# TOPOLOGY_CLASSES is the authoritative ordered list of topology class strings.
# The order is load-bearing: TOPOLOGY_TOKEN_OFFSET + index(class) = token id.
# Never duplicate this list here.  Import it from contracts.py and derive the
# token map at module load time so that any order change in contracts.py
# automatically propagates without touching this file.
from signal_kernel.contracts import TOPOLOGY_CLASSES

# ─────────────────────────────────────────────────────────────────────────────
# MODULE LOGGER
# Structural-signal fallbacks and unknown-signal warnings are logged at DEBUG
# so they appear in development but are suppressed at INFO in production.
# ─────────────────────────────────────────────────────────────────────────────

_log: logging.Logger = logging.getLogger("wlm_tokenizer")


# ═════════════════════════════════════════════════════════════════════════════
# VOCABULARY DIMENSIONS
# Single source of truth for every range boundary.  No arithmetic on these
# values is scattered through the encode/decode functions — all arithmetic
# uses the named constants below.
# ═════════════════════════════════════════════════════════════════════════════

VOCAB_SIZE: int = 8_192
"""Total vocabulary size including the PAD sentinel at index 0."""

MAX_SEQ_LEN: int = 512
"""Default maximum sequence length for encode_domain_event()."""

# ── Padding token ─────────────────────────────────────────────────────────────

PAD_TOKEN: int = 0
"""
Padding sentinel.  Index 0 is permanently reserved.  No vocabulary token may
occupy this index.  Attention masks are built by testing ``token != PAD_TOKEN``.

CRITICAL INVARIANT: PAD_TOKEN is never a valid vocabulary token.  All range
checks in this file verify this property at module load time.
"""

# ── Token range offsets ───────────────────────────────────────────────────────

TOPOLOGY_TOKEN_OFFSET: int = 1
"""
First index in the topology class token range.  Topology tokens occupy
[TOPOLOGY_TOKEN_OFFSET, TOPOLOGY_TOKEN_OFFSET + len(TOPOLOGY_CLASSES) - 1].

Set to 1 (not 0) to ensure PAD_TOKEN=0 never collides with a vocabulary token.
See module docstring for full collision-resolution rationale.
"""

_TOPOLOGY_COUNT: int = len(TOPOLOGY_CLASSES)
"""Number of topology class tokens.  Derived from contracts.TOPOLOGY_CLASSES."""

TOPOLOGY_TOKEN_END: int = TOPOLOGY_TOKEN_OFFSET + _TOPOLOGY_COUNT - 1
"""Last index (inclusive) in the topology class token range (= 18)."""

STRUCTURAL_TOKEN_OFFSET: int = TOPOLOGY_TOKEN_END + 1
"""
First index in the structural primitive token range (= 19).
Structural tokens occupy [STRUCTURAL_TOKEN_OFFSET, DOMAIN_TOKEN_OFFSET - 1].
"""

DOMAIN_TOKEN_OFFSET: int = 1_024
"""
First index in the domain hash token range.  Domain tokens occupy
[1024, 4095] — exactly 3,072 collision-tolerant buckets.
"""

_DOMAIN_HASH_BUCKETS: int = 3_072
"""Number of distinct domain hash buckets = 3,072."""

DOMAIN_TOKEN_END: int = DOMAIN_TOKEN_OFFSET + _DOMAIN_HASH_BUCKETS - 1
"""Last index (inclusive) in the domain hash token range (= 4095)."""

INTENT_TOKEN_OFFSET: int = 4_096
"""
First index in the intent signal token range.  Intent tokens occupy
[4096, 8191] — exactly 4,096 slots.

Each 6-dimensional quantized chunk of an intent vector maps to one token in
this range.  A 256-dimensional vector produces exactly 43 tokens.
"""

INTENT_TOKEN_END: int = VOCAB_SIZE - 1
"""Last index (inclusive) in the intent signal token range (= 8191)."""

# Derived structural range end for range checks.
STRUCTURAL_TOKEN_END: int = DOMAIN_TOKEN_OFFSET - 1
"""Last index (inclusive) in the structural primitive token range (= 1023)."""

_STRUCTURAL_SLOTS: int = STRUCTURAL_TOKEN_END - STRUCTURAL_TOKEN_OFFSET + 1
"""Total structural token slots = 1,005."""


# ═════════════════════════════════════════════════════════════════════════════
# TOPOLOGY CLASS TOKEN CONSTANTS
#
# These constants are derived programmatically from TOPOLOGY_CLASSES so that
# any reordering in contracts.py propagates automatically.  The assertion block
# at module load time verifies the count matches expectations.
#
# Do NOT hardcode the numeric values here.  Always derive: TOPOLOGY_TOKEN_OFFSET
# + TOPOLOGY_CLASSES.index(cls) = token.  The _TOPOLOGY_CLASS_TO_TOKEN dict
# is the runtime lookup path.
# ═════════════════════════════════════════════════════════════════════════════

def _build_topology_maps() -> Tuple[Dict[str, int], Dict[int, str]]:
    """
    Build bidirectional topology class ↔ token maps from TOPOLOGY_CLASSES.

    Called once at module load.  The resulting dicts are immutable in practice
    — neither is mutated after construction.  If contracts.TOPOLOGY_CLASSES
    changes at import time (e.g. a test monkey-patches it), a module reload is
    required to pick up the change.

    Returns:
        (class_to_token, token_to_class) — both covering all 18 classes.
    """
    cls_to_tok: Dict[str, int] = {}
    tok_to_cls: Dict[int, str] = {}
    for idx, cls in enumerate(TOPOLOGY_CLASSES):
        token = TOPOLOGY_TOKEN_OFFSET + idx
        cls_to_tok[cls] = token
        tok_to_cls[token] = cls
    return cls_to_tok, tok_to_cls


_TOPOLOGY_CLASS_TO_TOKEN: Dict[str, int]
_TOKEN_TO_TOPOLOGY_CLASS: Dict[int, str]
_TOPOLOGY_CLASS_TO_TOKEN, _TOKEN_TO_TOPOLOGY_CLASS = _build_topology_maps()

# Convenience constants — derived, not hardcoded.  These are the names that
# other modules might reference for documentation clarity.
TOPOLOGY_TOKEN_NEWS_ARTICLE:            int = _TOPOLOGY_CLASS_TO_TOKEN["NEWS_ARTICLE"]
TOPOLOGY_TOKEN_NEWS_ARTICLE_PAYWALLED:  int = _TOPOLOGY_CLASS_TO_TOKEN["NEWS_ARTICLE_PAYWALLED"]
TOPOLOGY_TOKEN_SAAS_DOCS:               int = _TOPOLOGY_CLASS_TO_TOKEN["SAAS_DOCS"]
TOPOLOGY_TOKEN_SAAS_DOCS_VERSIONED:     int = _TOPOLOGY_CLASS_TO_TOKEN["SAAS_DOCS_VERSIONED"]
TOPOLOGY_TOKEN_SAAS_DOCS_WITH_CODE:     int = _TOPOLOGY_CLASS_TO_TOKEN["SAAS_DOCS_WITH_CODE"]
TOPOLOGY_TOKEN_REST_API_JSON:           int = _TOPOLOGY_CLASS_TO_TOKEN["REST_API_JSON"]
TOPOLOGY_TOKEN_REST_API_JSON_PAGINATED: int = _TOPOLOGY_CLASS_TO_TOKEN["REST_API_JSON_PAGINATED"]
TOPOLOGY_TOKEN_JSON_LD_STRUCTURED:      int = _TOPOLOGY_CLASS_TO_TOKEN["JSON_LD_STRUCTURED"]
TOPOLOGY_TOKEN_ECOMMERCE_PRODUCT:       int = _TOPOLOGY_CLASS_TO_TOKEN["ECOMMERCE_PRODUCT"]
TOPOLOGY_TOKEN_ECOMMERCE_PRODUCT_VARIANT: int = _TOPOLOGY_CLASS_TO_TOKEN["ECOMMERCE_PRODUCT_VARIANT"]
TOPOLOGY_TOKEN_FORUM_THREAD:            int = _TOPOLOGY_CLASS_TO_TOKEN["FORUM_THREAD"]
TOPOLOGY_TOKEN_BLOG_POST:               int = _TOPOLOGY_CLASS_TO_TOKEN["BLOG_POST"]
TOPOLOGY_TOKEN_WIKIPEDIA_ARTICLE:       int = _TOPOLOGY_CLASS_TO_TOKEN["WIKIPEDIA_ARTICLE"]
TOPOLOGY_TOKEN_LANDING_PAGE:            int = _TOPOLOGY_CLASS_TO_TOKEN["LANDING_PAGE"]
TOPOLOGY_TOKEN_AUTH_REDIRECT:           int = _TOPOLOGY_CLASS_TO_TOKEN["AUTH_REDIRECT"]
TOPOLOGY_TOKEN_CLOUDFLARE_CHALLENGE:    int = _TOPOLOGY_CLASS_TO_TOKEN["CLOUDFLARE_CHALLENGE"]
TOPOLOGY_TOKEN_RATE_LIMITED:            int = _TOPOLOGY_CLASS_TO_TOKEN["RATE_LIMITED"]
TOPOLOGY_TOKEN_GENERIC_HTML:            int = _TOPOLOGY_CLASS_TO_TOKEN["GENERIC_HTML"]


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURAL PRIMITIVE TOKEN CONSTANTS
#
# Each constant names one specific structural signal.  Constants are grouped
# by semantic category and assigned contiguous token IDs starting at
# STRUCTURAL_TOKEN_OFFSET (19).
#
# Layout within structural range:
#   19 –  24  CDN fingerprints          ( 6 tokens)
#   25 –  32  CMS fingerprints          ( 8 tokens)
#   33 –  37  Render requirements       ( 5 tokens)
#   38 –  43  Bot mitigation signals    ( 6 tokens)
#   44 –  48  TLS/SSL signals           ( 5 tokens)
#   49 –  51  HTTP version signals      ( 3 tokens)
#   52 –  56  Response time buckets     ( 5 tokens)
#   57 –  61  robots.txt signals        ( 5 tokens)
#   62 –  66  Sitemap signals           ( 5 tokens)
#   67 –  71  Content-Type signals      ( 5 tokens)
#   72 –  76  Header signals            ( 5 tokens)
#   77 –  82  Status code signals       ( 6 tokens)
#   83        UNKNOWN_PRIMITIVE         ( 1 token)
#   84 – 1023 RESERVED (future primitives) (940 tokens)
#
# Total named: 64 + 1 UNKNOWN = 65 tokens.
# NEVER use the raw integer literals in encode functions — always use the named
# constant.  Adding a new primitive requires adding a constant here AND
# updating _SIGNAL_NAME_TO_TOKEN below.
# ═════════════════════════════════════════════════════════════════════════════

# ── CDN fingerprints (19–24) ──────────────────────────────────────────────────

PRIMITIVE_CLOUDFLARE:  int = 19
"""Cloudflare CDN detected (via CF-Ray header or ASN)."""

PRIMITIVE_FASTLY:      int = 20
"""Fastly CDN detected (via X-Served-By or X-Cache header pattern)."""

PRIMITIVE_AKAMAI:      int = 21
"""Akamai CDN detected (via X-Check-Cacheable or AkamaiGHost header)."""

PRIMITIVE_CLOUDFRONT:  int = 22
"""AWS CloudFront CDN detected (via X-Amz-Cf-Id or Via: cloudfront header)."""

PRIMITIVE_VERCEL:      int = 23
"""Vercel edge network detected (via X-Vercel-Id or server: Vercel header)."""

PRIMITIVE_NETLIFY:     int = 24
"""Netlify CDN detected (via X-Nf-Request-Id header or server: Netlify)."""

# ── CMS fingerprints (25–32) ──────────────────────────────────────────────────

PRIMITIVE_WORDPRESS:   int = 25
"""WordPress CMS detected (wp-content path pattern or X-Powered-By: PHP)."""

PRIMITIVE_GHOST:       int = 26
"""Ghost CMS detected (X-Ghost-Cache-Status or ghost.io pattern)."""

PRIMITIVE_DRUPAL:      int = 27
"""Drupal CMS detected (X-Generator: Drupal or /core/modules path)."""

PRIMITIVE_CONFLUENCE:  int = 28
"""Atlassian Confluence detected (X-ASEN header or /wiki/ path pattern)."""

PRIMITIVE_NOTION:      int = 29
"""Notion-backed site detected (notion.so in links or Notion-Version header)."""

PRIMITIVE_GITBOOK:     int = 30
"""GitBook docs detected (gitbook.io pattern or X-GitBook-* headers)."""

PRIMITIVE_DOCUSAURUS:  int = 31
"""Docusaurus static site detected (/docusaurus/ assets or meta generator tag)."""

PRIMITIVE_MKDOCS:      int = 32
"""MkDocs static site detected (mkdocs.yml pattern or MkDocs generator tag)."""

# ── Render requirements (33–37) ───────────────────────────────────────────────

PRIMITIVE_REQUIRES_JS:        int = 33
"""Page requires JavaScript execution for meaningful content (SPA or deferred)."""

PRIMITIVE_STATIC_ONLY:        int = 34
"""Page renders fully without JavaScript — static HTML is complete."""

PRIMITIVE_SPA_DETECTED:       int = 35
"""Single-page application detected — entire app loads in one JS bundle."""

PRIMITIVE_SSR_DETECTED:       int = 36
"""Server-side rendering detected — initial HTML is complete, hydration follows."""

PRIMITIVE_HYDRATION_DETECTED: int = 37
"""React/Vue/Svelte hydration detected — SSR output extended by client JS."""

# ── Bot mitigation signals (38–43) ────────────────────────────────────────────

PRIMITIVE_CLOUDFLARE_CHALLENGE: int = 38
"""Cloudflare challenge page returned (JS challenge or managed challenge)."""

PRIMITIVE_RECAPTCHA:            int = 39
"""Google reCAPTCHA v2/v3 challenge detected on the response page."""

PRIMITIVE_HCAPTCHA:             int = 40
"""hCaptcha challenge widget detected on the response page."""

PRIMITIVE_DATADOME:             int = 41
"""DataDome bot detection service detected (x-datadome-* headers or redirect)."""

PRIMITIVE_PERIMETER_X:         int = 42
"""PerimeterX bot detection service detected (_pxhd cookie or px script)."""

PRIMITIVE_RATE_LIMIT_HEADER:   int = 43
"""Rate limiting header detected (X-RateLimit-* or Retry-After response header)."""

# ── TLS/SSL signals (44–48) ───────────────────────────────────────────────────

PRIMITIVE_TLS_1_2:          int = 44
"""TLS 1.2 negotiated for the connection (older but still widely supported)."""

PRIMITIVE_TLS_1_3:          int = 45
"""TLS 1.3 negotiated for the connection (preferred — lower latency, forward secrecy)."""

PRIMITIVE_CERT_WILDCARD:    int = 46
"""Wildcard TLS certificate (* .domain.com) — typical of CDN-fronted services."""

PRIMITIVE_CERT_ORG:         int = 47
"""Organization Validated (OV) certificate — indicates verified business entity."""

PRIMITIVE_CERT_LETS_ENCRYPT: int = 48
"""Let's Encrypt DV certificate — lightweight, common for static/dev sites."""

# ── HTTP version signals (49–51) ──────────────────────────────────────────────

PRIMITIVE_HTTP_1_1: int = 49
"""HTTP/1.1 protocol used — sequential requests, common on legacy infrastructure."""

PRIMITIVE_HTTP_2:   int = 50
"""HTTP/2 protocol used — multiplexed streams, header compression."""

PRIMITIVE_HTTP_3:   int = 51
"""HTTP/3 (QUIC) protocol used — UDP-based, lowest latency, CDN-preferred."""

# ── Response time buckets (52–56) ─────────────────────────────────────────────

PRIMITIVE_RESPONSE_LT_100MS:   int = 52
"""Response received in under 100 ms — edge-cached or trivially fast origin."""

PRIMITIVE_RESPONSE_100_500MS:  int = 53
"""Response received in 100–500 ms — CDN-cached or fast origin server."""

PRIMITIVE_RESPONSE_500MS_2S:   int = 54
"""Response received in 500 ms – 2 s — typical origin server latency."""

PRIMITIVE_RESPONSE_2S_5S:      int = 55
"""Response received in 2–5 s — slow origin, SSR overhead, or throttled."""

PRIMITIVE_RESPONSE_GT_5S:      int = 56
"""Response received in over 5 s — timeouts likely, high retry cost."""

# ── robots.txt signals (57–61) ────────────────────────────────────────────────

PRIMITIVE_ROBOTS_PRESENT:    int = 57
"""robots.txt exists and was successfully fetched for this domain."""

PRIMITIVE_ROBOTS_ABSENT:     int = 58
"""robots.txt returned 404 or could not be fetched — no crawl policy declared."""

PRIMITIVE_CRAWL_DELAY_SET:   int = 59
"""Crawl-delay directive present in robots.txt — domain requests paced access."""

PRIMITIVE_DISALLOW_HEAVY:    int = 60
"""robots.txt disallows >50% of known signal paths — heavy restriction in place."""

PRIMITIVE_SITEMAP_LINKED:    int = 61
"""Sitemap directive present in robots.txt — structured URL discovery available."""

# ── Sitemap signals (62–66) ───────────────────────────────────────────────────

PRIMITIVE_SITEMAP_PRESENT:  int = 62
"""sitemap.xml exists and was successfully fetched."""

PRIMITIVE_SITEMAP_INDEX:    int = 63
"""Fetched sitemap is a sitemap index (sitemapindex root) referencing sub-sitemaps."""

PRIMITIVE_SITEMAP_URLSET:   int = 64
"""Fetched sitemap is a URL set (urlset root) with direct page listings."""

PRIMITIVE_SITEMAP_NEWS:     int = 65
"""Sitemap contains news namespace entries (news:news elements)."""

PRIMITIVE_SITEMAP_IMAGE:    int = 66
"""Sitemap contains image namespace entries (image:image elements)."""

# ── Content-Type signals (67–71) ──────────────────────────────────────────────

PRIMITIVE_TEXT_HTML:            int = 67
"""Response Content-Type is text/html."""

PRIMITIVE_APPLICATION_JSON:     int = 68
"""Response Content-Type is application/json."""

PRIMITIVE_APPLICATION_LD_JSON:  int = 69
"""Response Content-Type is application/ld+json (JSON-LD structured data)."""

PRIMITIVE_TEXT_PLAIN:           int = 70
"""Response Content-Type is text/plain."""

PRIMITIVE_APPLICATION_XML:      int = 71
"""Response Content-Type is application/xml (sitemap, RSS, Atom)."""

# ── Header signals (72–76) ────────────────────────────────────────────────────

PRIMITIVE_X_POWERED_BY_PRESENT: int = 72
"""X-Powered-By header present — reveals backend technology (PHP, ASP.NET, etc.)."""

PRIMITIVE_SERVER_NGINX:         int = 73
"""Server: nginx header — Nginx is the origin server."""

PRIMITIVE_SERVER_APACHE:        int = 74
"""Server: Apache header — Apache httpd is the origin server."""

PRIMITIVE_SERVER_CADDY:         int = 75
"""Server: Caddy header — Caddy is the origin server."""

PRIMITIVE_VIA_PRESENT:          int = 76
"""Via header present — request passed through one or more proxies."""

# ── Status code signals (77–82) ───────────────────────────────────────────────

PRIMITIVE_STATUS_200: int = 77
"""HTTP 200 OK — page fetched successfully."""

PRIMITIVE_STATUS_301: int = 78
"""HTTP 301 Moved Permanently — permanent redirect; destination may differ in class."""

PRIMITIVE_STATUS_302: int = 79
"""HTTP 302 Found — temporary redirect; frequently auth or CDN-level redirect."""

PRIMITIVE_STATUS_403: int = 80
"""HTTP 403 Forbidden — access denied; paywall, geo-block, or bot-block."""

PRIMITIVE_STATUS_429: int = 81
"""HTTP 429 Too Many Requests — rate limited by the origin or CDN."""

PRIMITIVE_STATUS_503: int = 82
"""HTTP 503 Service Unavailable — origin overloaded or under maintenance."""

# ── Catch-all / fallback (83) ─────────────────────────────────────────────────

PRIMITIVE_UNKNOWN: int = 83
"""
Fallback token for any structural signal string not found in _SIGNAL_NAME_TO_TOKEN.

When the tokenizer encounters an unrecognized signal it maps to this token and
logs a DEBUG message with the unknown signal name.  This ensures encode functions
never silently drop signals — the model always receives *something* for an unknown
signal, and the debug log makes the gap visible during development.

Callers that deliberately pass non-standard signals (for experiments or custom
primitives not yet promoted to named constants) will produce this token.
"""

# ── Reserved structural range (84–1023) ───────────────────────────────────────

FIRST_RESERVED_PRIMITIVE: int = 84
"""
First token index in the reserved-for-future-use structural range.
Tokens 84–1023 are unassigned.  They are reserved for new structural primitive
categories without requiring a TOPOLOGY_TOKEN_OFFSET shift or a VOCAB_SIZE change.

Extending the primitive vocabulary: add a constant in the appropriate category
block above, assign it the next value after the current category's last token,
shift subsequent categories up by the number of new tokens added, and update
_SIGNAL_NAME_TO_TOKEN and SIGNAL_PRIORITY_RANK.  Run verify_vocabulary_integrity()
after any change.
"""

LAST_RESERVED_PRIMITIVE: int = STRUCTURAL_TOKEN_END  # 1023
"""Last token index in the reserved structural range (= 1023)."""

_RESERVED_PRIMITIVE_COUNT: int = LAST_RESERVED_PRIMITIVE - FIRST_RESERVED_PRIMITIVE + 1
"""Reserved structural slots = 940."""


# ═════════════════════════════════════════════════════════════════════════════
# STRUCTURAL SIGNAL NAME → TOKEN MAPPING
#
# This dict is the runtime lookup table for structural_signal_to_token().
# Keys are the canonical string signal names that callers pass in the
# structural_signals list to encode_domain_event().
#
# Design:
#   - Keys use snake_case to match the string constants callers produce from
#     domain_analyzer.py (which emits the same snake_case names).
#   - Multiple aliases are allowed for the same token where the upstream
#     signal producer uses different naming conventions.
#   - Every named PRIMITIVE_* constant above must have at least one key here.
# ═════════════════════════════════════════════════════════════════════════════

_SIGNAL_NAME_TO_TOKEN: Dict[str, int] = {

    # ── CDN fingerprints ──────────────────────────────────────────────────────
    "cloudflare":               PRIMITIVE_CLOUDFLARE,
    "cdn_cloudflare":           PRIMITIVE_CLOUDFLARE,
    "fastly":                   PRIMITIVE_FASTLY,
    "cdn_fastly":               PRIMITIVE_FASTLY,
    "akamai":                   PRIMITIVE_AKAMAI,
    "cdn_akamai":               PRIMITIVE_AKAMAI,
    "cloudfront":               PRIMITIVE_CLOUDFRONT,
    "cdn_cloudfront":           PRIMITIVE_CLOUDFRONT,
    "aws_cloudfront":           PRIMITIVE_CLOUDFRONT,
    "vercel":                   PRIMITIVE_VERCEL,
    "cdn_vercel":               PRIMITIVE_VERCEL,
    "netlify":                  PRIMITIVE_NETLIFY,
    "cdn_netlify":              PRIMITIVE_NETLIFY,

    # ── CMS fingerprints ──────────────────────────────────────────────────────
    "wordpress":                PRIMITIVE_WORDPRESS,
    "cms_wordpress":            PRIMITIVE_WORDPRESS,
    "ghost":                    PRIMITIVE_GHOST,
    "cms_ghost":                PRIMITIVE_GHOST,
    "drupal":                   PRIMITIVE_DRUPAL,
    "cms_drupal":               PRIMITIVE_DRUPAL,
    "confluence":               PRIMITIVE_CONFLUENCE,
    "cms_confluence":           PRIMITIVE_CONFLUENCE,
    "atlassian_confluence":     PRIMITIVE_CONFLUENCE,
    "notion":                   PRIMITIVE_NOTION,
    "cms_notion":               PRIMITIVE_NOTION,
    "gitbook":                  PRIMITIVE_GITBOOK,
    "cms_gitbook":              PRIMITIVE_GITBOOK,
    "docusaurus":               PRIMITIVE_DOCUSAURUS,
    "cms_docusaurus":           PRIMITIVE_DOCUSAURUS,
    "mkdocs":                   PRIMITIVE_MKDOCS,
    "cms_mkdocs":               PRIMITIVE_MKDOCS,

    # ── Render requirements ───────────────────────────────────────────────────
    "requires_js":              PRIMITIVE_REQUIRES_JS,
    "js_required":              PRIMITIVE_REQUIRES_JS,
    "static_only":              PRIMITIVE_STATIC_ONLY,
    "no_js_required":           PRIMITIVE_STATIC_ONLY,
    "spa_detected":             PRIMITIVE_SPA_DETECTED,
    "is_spa":                   PRIMITIVE_SPA_DETECTED,
    "ssr_detected":             PRIMITIVE_SSR_DETECTED,
    "is_ssr":                   PRIMITIVE_SSR_DETECTED,
    "hydration_detected":       PRIMITIVE_HYDRATION_DETECTED,
    "has_hydration":            PRIMITIVE_HYDRATION_DETECTED,

    # ── Bot mitigation signals ────────────────────────────────────────────────
    "cloudflare_challenge":     PRIMITIVE_CLOUDFLARE_CHALLENGE,
    "cf_challenge":             PRIMITIVE_CLOUDFLARE_CHALLENGE,
    "recaptcha":                PRIMITIVE_RECAPTCHA,
    "google_recaptcha":         PRIMITIVE_RECAPTCHA,
    "hcaptcha":                 PRIMITIVE_HCAPTCHA,
    "h_captcha":                PRIMITIVE_HCAPTCHA,
    "datadome":                 PRIMITIVE_DATADOME,
    "data_dome":                PRIMITIVE_DATADOME,
    "perimeter_x":              PRIMITIVE_PERIMETER_X,
    "perimeterx":               PRIMITIVE_PERIMETER_X,
    "px_bot_detect":            PRIMITIVE_PERIMETER_X,
    "rate_limit_header":        PRIMITIVE_RATE_LIMIT_HEADER,
    "ratelimit_header":         PRIMITIVE_RATE_LIMIT_HEADER,
    "retry_after_present":      PRIMITIVE_RATE_LIMIT_HEADER,

    # ── TLS/SSL signals ───────────────────────────────────────────────────────
    "tls_1_2":                  PRIMITIVE_TLS_1_2,
    "tlsv1.2":                  PRIMITIVE_TLS_1_2,
    "tls_1_3":                  PRIMITIVE_TLS_1_3,
    "tlsv1.3":                  PRIMITIVE_TLS_1_3,
    "cert_wildcard":            PRIMITIVE_CERT_WILDCARD,
    "wildcard_cert":            PRIMITIVE_CERT_WILDCARD,
    "cert_org":                 PRIMITIVE_CERT_ORG,
    "ov_cert":                  PRIMITIVE_CERT_ORG,
    "cert_lets_encrypt":        PRIMITIVE_CERT_LETS_ENCRYPT,
    "lets_encrypt":             PRIMITIVE_CERT_LETS_ENCRYPT,
    "letsencrypt":              PRIMITIVE_CERT_LETS_ENCRYPT,

    # ── HTTP version signals ──────────────────────────────────────────────────
    "http_1_1":                 PRIMITIVE_HTTP_1_1,
    "http/1.1":                 PRIMITIVE_HTTP_1_1,
    "http_2":                   PRIMITIVE_HTTP_2,
    "http/2":                   PRIMITIVE_HTTP_2,
    "h2":                       PRIMITIVE_HTTP_2,
    "http_3":                   PRIMITIVE_HTTP_3,
    "http/3":                   PRIMITIVE_HTTP_3,
    "h3":                       PRIMITIVE_HTTP_3,
    "quic":                     PRIMITIVE_HTTP_3,

    # ── Response time buckets ─────────────────────────────────────────────────
    "lt_100ms":                 PRIMITIVE_RESPONSE_LT_100MS,
    "<100ms":                   PRIMITIVE_RESPONSE_LT_100MS,
    "response_lt_100ms":        PRIMITIVE_RESPONSE_LT_100MS,
    "100_500ms":                PRIMITIVE_RESPONSE_100_500MS,
    "100ms_500ms":              PRIMITIVE_RESPONSE_100_500MS,
    "response_100_500ms":       PRIMITIVE_RESPONSE_100_500MS,
    "500ms_2s":                 PRIMITIVE_RESPONSE_500MS_2S,
    "500_2000ms":               PRIMITIVE_RESPONSE_500MS_2S,
    "response_500ms_2s":        PRIMITIVE_RESPONSE_500MS_2S,
    "2s_5s":                    PRIMITIVE_RESPONSE_2S_5S,
    "2000_5000ms":              PRIMITIVE_RESPONSE_2S_5S,
    "response_2s_5s":           PRIMITIVE_RESPONSE_2S_5S,
    "gt_5s":                    PRIMITIVE_RESPONSE_GT_5S,
    ">5s":                      PRIMITIVE_RESPONSE_GT_5S,
    "response_gt_5s":           PRIMITIVE_RESPONSE_GT_5S,

    # ── robots.txt signals ────────────────────────────────────────────────────
    "robots_present":           PRIMITIVE_ROBOTS_PRESENT,
    "has_robots_txt":           PRIMITIVE_ROBOTS_PRESENT,
    "robots_absent":            PRIMITIVE_ROBOTS_ABSENT,
    "no_robots_txt":            PRIMITIVE_ROBOTS_ABSENT,
    "crawl_delay_set":          PRIMITIVE_CRAWL_DELAY_SET,
    "has_crawl_delay":          PRIMITIVE_CRAWL_DELAY_SET,
    "disallow_heavy":           PRIMITIVE_DISALLOW_HEAVY,
    "heavy_disallow":           PRIMITIVE_DISALLOW_HEAVY,
    "sitemap_linked":           PRIMITIVE_SITEMAP_LINKED,
    "robots_has_sitemap":       PRIMITIVE_SITEMAP_LINKED,

    # ── Sitemap signals ───────────────────────────────────────────────────────
    "sitemap_present":          PRIMITIVE_SITEMAP_PRESENT,
    "has_sitemap":              PRIMITIVE_SITEMAP_PRESENT,
    "sitemap_index":            PRIMITIVE_SITEMAP_INDEX,
    "is_sitemap_index":         PRIMITIVE_SITEMAP_INDEX,
    "sitemap_urlset":           PRIMITIVE_SITEMAP_URLSET,
    "is_sitemap_urlset":        PRIMITIVE_SITEMAP_URLSET,
    "sitemap_news":             PRIMITIVE_SITEMAP_NEWS,
    "has_news_sitemap":         PRIMITIVE_SITEMAP_NEWS,
    "sitemap_image":            PRIMITIVE_SITEMAP_IMAGE,
    "has_image_sitemap":        PRIMITIVE_SITEMAP_IMAGE,

    # ── Content-Type signals ──────────────────────────────────────────────────
    "text_html":                PRIMITIVE_TEXT_HTML,
    "content_type_html":        PRIMITIVE_TEXT_HTML,
    "application_json":         PRIMITIVE_APPLICATION_JSON,
    "content_type_json":        PRIMITIVE_APPLICATION_JSON,
    "application_ld_json":      PRIMITIVE_APPLICATION_LD_JSON,
    "content_type_ldjson":      PRIMITIVE_APPLICATION_LD_JSON,
    "json_ld":                  PRIMITIVE_APPLICATION_LD_JSON,
    "text_plain":               PRIMITIVE_TEXT_PLAIN,
    "content_type_plain":       PRIMITIVE_TEXT_PLAIN,
    "application_xml":          PRIMITIVE_APPLICATION_XML,
    "content_type_xml":         PRIMITIVE_APPLICATION_XML,

    # ── Header signals ────────────────────────────────────────────────────────
    "x_powered_by_present":     PRIMITIVE_X_POWERED_BY_PRESENT,
    "has_x_powered_by":         PRIMITIVE_X_POWERED_BY_PRESENT,
    "server_nginx":             PRIMITIVE_SERVER_NGINX,
    "nginx":                    PRIMITIVE_SERVER_NGINX,
    "server_apache":            PRIMITIVE_SERVER_APACHE,
    "apache":                   PRIMITIVE_SERVER_APACHE,
    "server_caddy":             PRIMITIVE_SERVER_CADDY,
    "caddy":                    PRIMITIVE_SERVER_CADDY,
    "via_present":              PRIMITIVE_VIA_PRESENT,
    "has_via_header":           PRIMITIVE_VIA_PRESENT,

    # ── Status code signals ───────────────────────────────────────────────────
    "status_200":               PRIMITIVE_STATUS_200,
    "http_200":                 PRIMITIVE_STATUS_200,
    "status_301":               PRIMITIVE_STATUS_301,
    "http_301":                 PRIMITIVE_STATUS_301,
    "status_302":               PRIMITIVE_STATUS_302,
    "http_302":                 PRIMITIVE_STATUS_302,
    "status_403":               PRIMITIVE_STATUS_403,
    "http_403":                 PRIMITIVE_STATUS_403,
    "status_429":               PRIMITIVE_STATUS_429,
    "http_429":                 PRIMITIVE_STATUS_429,
    "status_503":               PRIMITIVE_STATUS_503,
    "http_503":                 PRIMITIVE_STATUS_503,
}

# Reverse mapping: token → canonical signal name (first name encountered wins).
# Used exclusively by token_to_primitive_name() for debugging and audit.
_TOKEN_TO_SIGNAL_NAME: Dict[int, str] = {}
for _sig_name, _sig_token in _SIGNAL_NAME_TO_TOKEN.items():
    if _sig_token not in _TOKEN_TO_SIGNAL_NAME:
        _TOKEN_TO_SIGNAL_NAME[_sig_token] = _sig_name
# Cleanup loop variables — they should not pollute module namespace.
del _sig_name, _sig_token

# Add the UNKNOWN token to the reverse map explicitly.
_TOKEN_TO_SIGNAL_NAME[PRIMITIVE_UNKNOWN] = "UNKNOWN_PRIMITIVE"


# ═════════════════════════════════════════════════════════════════════════════
# SIGNAL PRIORITY RANK
#
# When encode_domain_event() sorts structural tokens before assembly, it uses
# this rank mapping.  Lower rank number = higher priority = appears earlier in
# the token sequence.  The SSM is order-sensitive: tokens that appear early in
# the sequence contribute to the hidden state before later tokens, so signals
# with the highest routing relevance should come first.
#
# Priority rationale:
#   1. Status codes (rank 0–5):  immediate feasibility signal — a 429 or 503
#      changes everything downstream before any other signal matters.
#   2. Bot mitigation (rank 10–15): second-most decisive; determines whether
#      Phantom should attempt fetch at all or route to alternative strategy.
#   3. CDN fingerprints (rank 20–25): shapes traversal strategy, render mode,
#      and rate limit defaults in a predictable and consistent way.
#   4. Render requirements (rank 30–34): determines static vs. headless path.
#   5. TLS/SSL signals (rank 40–44): security posture, less routing-decisive.
#   6. HTTP version (rank 50–52): latency profile.
#   7. Response time (rank 60–64): reinforces latency expectations.
#   8. CMS fingerprints (rank 70–77): content structure.
#   9. Content-Type (rank 80–84): parsing strategy.
#   10. Header signals (rank 90–94): ancillary infrastructure signals.
#   11. robots.txt (rank 100–104): crawl policy.
#   12. Sitemap (rank 110–114): URL discovery quality.
#   Unknown signals map to rank 200 (lowest priority, still included).
# ═════════════════════════════════════════════════════════════════════════════

SIGNAL_PRIORITY_RANK: Dict[int, int] = {
    # Status codes — rank 0–5
    PRIMITIVE_STATUS_200:        0,
    PRIMITIVE_STATUS_301:        1,
    PRIMITIVE_STATUS_302:        2,
    PRIMITIVE_STATUS_403:        3,
    PRIMITIVE_STATUS_429:        4,
    PRIMITIVE_STATUS_503:        5,
    # Bot mitigation — rank 10–15
    PRIMITIVE_CLOUDFLARE_CHALLENGE: 10,
    PRIMITIVE_RECAPTCHA:            11,
    PRIMITIVE_HCAPTCHA:             12,
    PRIMITIVE_DATADOME:             13,
    PRIMITIVE_PERIMETER_X:         14,
    PRIMITIVE_RATE_LIMIT_HEADER:   15,
    # CDN fingerprints — rank 20–25
    PRIMITIVE_CLOUDFLARE:        20,
    PRIMITIVE_FASTLY:            21,
    PRIMITIVE_AKAMAI:            22,
    PRIMITIVE_CLOUDFRONT:        23,
    PRIMITIVE_VERCEL:            24,
    PRIMITIVE_NETLIFY:           25,
    # Render requirements — rank 30–34
    PRIMITIVE_REQUIRES_JS:       30,
    PRIMITIVE_STATIC_ONLY:       31,
    PRIMITIVE_SPA_DETECTED:      32,
    PRIMITIVE_SSR_DETECTED:      33,
    PRIMITIVE_HYDRATION_DETECTED: 34,
    # TLS/SSL — rank 40–44
    PRIMITIVE_TLS_1_3:           40,
    PRIMITIVE_TLS_1_2:           41,
    PRIMITIVE_CERT_LETS_ENCRYPT: 42,
    PRIMITIVE_CERT_ORG:          43,
    PRIMITIVE_CERT_WILDCARD:     44,
    # HTTP version — rank 50–52
    PRIMITIVE_HTTP_3:            50,
    PRIMITIVE_HTTP_2:            51,
    PRIMITIVE_HTTP_1_1:          52,
    # Response time — rank 60–64
    PRIMITIVE_RESPONSE_LT_100MS:  60,
    PRIMITIVE_RESPONSE_100_500MS: 61,
    PRIMITIVE_RESPONSE_500MS_2S:  62,
    PRIMITIVE_RESPONSE_2S_5S:     63,
    PRIMITIVE_RESPONSE_GT_5S:     64,
    # CMS fingerprints — rank 70–77
    PRIMITIVE_WORDPRESS:         70,
    PRIMITIVE_GHOST:             71,
    PRIMITIVE_DRUPAL:            72,
    PRIMITIVE_CONFLUENCE:        73,
    PRIMITIVE_NOTION:            74,
    PRIMITIVE_GITBOOK:           75,
    PRIMITIVE_DOCUSAURUS:        76,
    PRIMITIVE_MKDOCS:            77,
    # Content-Type — rank 80–84
    PRIMITIVE_TEXT_HTML:          80,
    PRIMITIVE_APPLICATION_JSON:   81,
    PRIMITIVE_APPLICATION_LD_JSON: 82,
    PRIMITIVE_TEXT_PLAIN:         83,
    PRIMITIVE_APPLICATION_XML:    84,
    # Header signals — rank 90–94
    PRIMITIVE_X_POWERED_BY_PRESENT: 90,
    PRIMITIVE_SERVER_NGINX:         91,
    PRIMITIVE_SERVER_APACHE:        92,
    PRIMITIVE_SERVER_CADDY:         93,
    PRIMITIVE_VIA_PRESENT:          94,
    # robots.txt — rank 100–104
    PRIMITIVE_ROBOTS_PRESENT:    100,
    PRIMITIVE_ROBOTS_ABSENT:     101,
    PRIMITIVE_CRAWL_DELAY_SET:   102,
    PRIMITIVE_DISALLOW_HEAVY:    103,
    PRIMITIVE_SITEMAP_LINKED:    104,
    # Sitemap — rank 110–114
    PRIMITIVE_SITEMAP_PRESENT:   110,
    PRIMITIVE_SITEMAP_INDEX:     111,
    PRIMITIVE_SITEMAP_URLSET:    112,
    PRIMITIVE_SITEMAP_NEWS:      113,
    PRIMITIVE_SITEMAP_IMAGE:     114,
    # Catch-all
    PRIMITIVE_UNKNOWN:           200,
}

_DEFAULT_SIGNAL_RANK: int = 200
"""Priority rank assigned to any structural token not listed in SIGNAL_PRIORITY_RANK."""


# ═════════════════════════════════════════════════════════════════════════════
# INTENT VECTOR QUANTIZATION CONSTANTS
#
# A 256-dimensional float32 intent vector is converted to a list of integer
# tokens through two steps:
#
#   Step 1 — Quantize each dimension to one of four levels (0, 1, 2, 3):
#     value <  -0.5  → 0  (strongly negative)
#     value in [-0.5,  0.0) → 1  (mildly negative)
#     value in [ 0.0,  0.5] → 2  (mildly positive)
#     value >   0.5  → 3  (strongly positive)
#
#   Step 2 — Group quantized dimensions into chunks of INTENT_CHUNK_SIZE
#     (6 dims each), pack each chunk as a base-4 number into a 12-bit integer
#     (range 0–4095), then add INTENT_TOKEN_OFFSET (4096).
#
# This produces INTENT_TOKENS_PER_VECTOR (43) tokens per vector, each in the
# range [4096, 8191], fully contained within the intent token range.
#
# The 43rd (last) chunk covers only 4 dimensions (256 mod 6 = 4); the remaining
# 2 positions in the 6-slot chunk are padded with quantized value 0 before
# packing.  The position of the real vs. padded dimensions within the last chunk
# is consistent across all calls (real dims first, zeros appended).
#
# INVARIANT: changing INTENT_QUANT_THRESHOLDS changes the token produced by
# every intent vector.  The entire MFT is invalidated.  This is not a tuning
# parameter — it is a vocabulary definition.  Version-control changes to these
# values the same way you would version a schema migration.
# ═════════════════════════════════════════════════════════════════════════════

INTENT_QUANT_THRESHOLDS: Tuple[float, float, float] = (-0.5, 0.0, 0.5)
"""
Three-boundary quantization thresholds for intent vector dimensions.

Defines four levels via three boundaries:
  (-inf, -0.5)  → quantized level 0
  [-0.5,  0.0)  → quantized level 1
  [ 0.0,  0.5]  → quantized level 2
  ( 0.5, +inf)  → quantized level 3

These thresholds were chosen to divide a roughly zero-centered unit-variance
distribution into four equal-probability buckets.  If the distribution of your
intent vectors is materially different (e.g. bimodal, skewed), recalibrate
these values from a representative sample BEFORE training begins.  Any change
after training invalidates all MFT checkpoints.
"""

INTENT_VECTOR_DIM: int = 256
"""Expected dimensionality of all intent vectors."""

INTENT_CHUNK_SIZE: int = 6
"""
Number of quantized intent dimensions packed into one token.

6 dimensions × 2 bits/dimension = 12 bits → values 0–4095 (= 4^6 - 1).
Adding INTENT_TOKEN_OFFSET maps the value directly into the intent token range
[4096, 8191] with zero additional arithmetic.
"""

INTENT_TOKENS_PER_VECTOR: int = math.ceil(INTENT_VECTOR_DIM / INTENT_CHUNK_SIZE)
"""
Number of intent tokens produced per 256-dimensional vector = 43.

Derivation:
  256 / 6 = 42 full chunks (dims 0–251) + 1 partial chunk (dims 252–255).
  The partial chunk has 4 real dimensions and 2 zero-padded positions.
  Total: 43 tokens.
"""

_INTENT_CHUNK_BASE: int = 4 ** INTENT_CHUNK_SIZE  # = 4096
"""
Maximum exclusive value of a packed 6-dim chunk (= 4^6 = 4096).
Intent token for a chunk = INTENT_TOKEN_OFFSET + chunk_value,
where chunk_value ∈ [0, _INTENT_CHUNK_BASE - 1] = [0, 4095].
Verification: INTENT_TOKEN_OFFSET + 4095 = 8191 = INTENT_TOKEN_END. ✓
"""


# ═════════════════════════════════════════════════════════════════════════════
# DOMAIN NORMALIZATION INFRASTRUCTURE
#
# Domain normalization strips subdomains to the registrable domain (eTLD+1)
# so that docs.stripe.com and api.stripe.com both hash to stripe.com.
# This is load-bearing for MFT routing: the model must learn stripe.com as a
# unit, not as disconnected fragments.
#
# Two mechanisms are used:
#
#   1. Known multi-part TLDs: if the last two domain labels form a known
#      multi-part TLD (e.g. .co.uk, .com.au), the registrable domain is the
#      third-from-last label plus the two-part TLD.
#      Example: api.bbc.co.uk → bbc.co.uk
#
#   2. Single-part TLDs: all other TLDs.  The registrable domain is the last
#      two labels.
#      Example: api.github.com → github.com
#
# Limitations (documented, not silent):
#   - This is a hand-coded approximation of the Public Suffix List.  It covers
#     the vast majority of domains encountered during web crawl but will
#     misclassify edge-case ccTLDs not in _KNOWN_MULTI_PART_TLDS.
#   - No file reads, no network calls, no runtime PSL lookup.
#   - IP addresses and bare hostnames (no TLD) are returned unchanged.
#   - The normalization is case-insensitive and lowercases the result.
# ═════════════════════════════════════════════════════════════════════════════

_KNOWN_MULTI_PART_TLDS: FrozenSet[str] = frozenset({
    # United Kingdom
    "co.uk", "org.uk", "me.uk", "net.uk", "ltd.uk", "plc.uk",
    "sch.uk", "gov.uk", "nhs.uk", "mod.uk", "police.uk",
    # Australia
    "com.au", "net.au", "org.au", "edu.au", "gov.au", "asn.au", "id.au",
    # Brazil
    "com.br", "net.br", "org.br", "gov.br", "edu.br", "mil.br",
    # Japan
    "co.jp", "or.jp", "ne.jp", "ac.jp", "ad.jp", "ed.jp", "go.jp",
    # India
    "co.in", "org.in", "net.in", "gen.in", "gov.in", "mil.in", "nic.in",
    # New Zealand
    "co.nz", "net.nz", "org.nz", "govt.nz", "school.nz",
    # South Africa
    "co.za", "org.za", "net.za", "edu.za", "gov.za",
    # Ireland
    "co.ie", "org.ie", "gov.ie",
    # Hong Kong
    "com.hk", "org.hk", "net.hk", "gov.hk", "edu.hk",
    # Singapore
    "com.sg", "org.sg", "net.sg", "gov.sg", "edu.sg",
    # Germany — note: most use single .de, but some second-level variants exist
    # (kept minimal for correctness)
    # France
    "com.fr",
    # Italy
    "co.it",
    # Spain
    "com.es",
    # China
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    # Taiwan
    "com.tw", "org.tw", "net.tw", "gov.tw", "edu.tw",
    # Mexico
    "com.mx", "org.mx", "net.mx", "gob.mx",
    # Argentina
    "com.ar", "org.ar", "net.ar", "gov.ar",
    # Colombia
    "com.co", "org.co", "net.co", "gov.co",
    # Chile
    "com.cl", "org.cl", "net.cl", "gov.cl",
    # Venezuela
    "com.ve", "org.ve", "net.ve", "gov.ve",
    # Russia — .ru uses single-part, but some regional variants:
    # Pakistan
    "com.pk", "org.pk", "net.pk", "gov.pk",
    # Bangladesh
    "com.bd", "org.bd", "net.bd", "gov.bd",
    # Nigeria
    "com.ng", "org.ng", "net.ng", "gov.ng",
    # Kenya
    "co.ke", "org.ke", "net.ke", "go.ke",
    # Egypt
    "com.eg", "org.eg", "net.eg", "gov.eg",
    # Indonesia
    "co.id", "or.id", "net.id", "go.id", "ac.id",
    # Malaysia
    "com.my", "org.my", "net.my", "gov.my", "edu.my",
    # Philippines
    "com.ph", "org.ph", "net.ph", "gov.ph", "edu.ph",
    # Thailand
    "co.th", "or.th", "net.th", "go.th", "ac.th",
    # Vietnam
    "com.vn", "org.vn", "net.vn", "gov.vn", "edu.vn",
    # Saudi Arabia
    "com.sa", "org.sa", "net.sa", "gov.sa", "edu.sa",
    # United Arab Emirates
    "co.ae", "org.ae", "net.ae", "gov.ae", "ac.ae",
    # Israel
    "co.il", "org.il", "net.il", "gov.il",
    # Turkey
    "com.tr", "org.tr", "net.tr", "gov.tr", "edu.tr",
})
"""
Frozenset of known two-label TLD suffixes that require extracting three domain
labels to reach the registrable domain.  Sourced from the most common public
suffix list entries.  Not exhaustive: edge-case ccTLDs outside this set will
be treated as single-label TLDs (conservative fallback — may over-strip).
"""

_COMMON_SUBDOMAIN_PREFIXES: FrozenSet[str] = frozenset({
    # API / developer subdomains
    "api", "api2", "api3", "rest", "graphql", "grpc",
    # Documentation subdomains
    "docs", "doc", "documentation", "help", "support", "kb",
    "knowledge", "guides", "learn", "tutorial", "reference", "ref",
    # Version-prefixed subdomains (v1, v2, v2-api, etc. — handled by label count)
    # CDN / static asset subdomains
    "cdn", "assets", "static", "media", "images", "img", "files",
    "s3", "storage",
    # App / product subdomains
    "app", "apps", "web", "portal", "console", "dashboard", "admin",
    "manage", "my", "account", "login", "auth", "sso",
    # Environment subdomains
    "staging", "stage", "stg", "dev", "development", "test", "testing",
    "qa", "sandbox", "preview", "beta", "alpha", "demo", "sandbox",
    # Blog / marketing
    "blog", "news", "press", "about", "www2", "m",
    # Regional / geo subdomains
    "us", "eu", "uk", "de", "fr", "jp", "au", "ca",
    "us-east", "us-west", "eu-west", "ap-south",
    # Status / monitoring
    "status", "health", "metrics",
})
"""
Common subdomain prefixes stripped during normalization when they appear as the
leftmost label of a multi-label hostname.  This list extends the eTLD+1
stripping for the specific subdomains most likely to occur in the crawl corpus.

Note: this set is consulted only when the remaining domain after eTLD+1 stripping
still has more than 2 labels.  It is NOT applied to reduce a 2-label domain to 1.
"""


def normalize_domain(domain: str) -> str:
    """
    Strip subdomains from *domain* to produce the registrable domain (eTLD+1).

    Normalisation is case-insensitive and always lowercases the result.  The
    same domain string always produces the same result — this property is
    required for the MD5 hash in domain_to_token() to be stable.

    Algorithm:
      1. Lowercase and strip whitespace.
      2. If the input contains no dot or is an IP address, return it unchanged.
      3. Split on dots to produce labels.
      4. Check if the last two labels form a known multi-part TLD.
         If yes → registrable domain = labels[-3] + "." + labels[-2] + "." + labels[-1]
         If no  → registrable domain = labels[-2] + "." + labels[-1]
      5. Return the registrable domain.

    The result is the minimum string that uniquely identifies the domain owner.
    All subdomains of the same registrable domain map to the same token.

    Examples:
      >>> normalize_domain("docs.stripe.com")
      'stripe.com'
      >>> normalize_domain("api.github.com")
      'github.com'
      >>> normalize_domain("v2.docs.aws.com")
      'aws.com'
      >>> normalize_domain("api.bbc.co.uk")
      'bbc.co.uk'
      >>> normalize_domain("staging.api.acme.co.uk")
      'acme.co.uk'
      >>> normalize_domain("github.com")
      'github.com'
      >>> normalize_domain("192.168.1.1")
      '192.168.1.1'
      >>> normalize_domain("")
      ''

    Args:
        domain: Raw domain name, possibly with subdomains.  May include a port
                suffix (e.g. "example.com:8080") — the port is stripped before
                processing.

    Returns:
        Normalised registrable domain, lowercase.  If the input is already a
        bare registrable domain (two labels, single-part TLD) it is returned
        unchanged (lowercased).  IP addresses and bare hostnames are returned
        as-is (lowercased).
    """
    if not domain:
        return ""

    # Strip port if present.
    domain = domain.split(":")[0]

    # Normalise case.
    domain = domain.strip().lower()

    if not domain:
        return ""

    # Fast-path: no dots → bare hostname or TLD-less identifier.
    if "." not in domain:
        return domain

    # Reject obvious IP addresses (four numeric octets).
    labels: List[str] = domain.split(".")
    if len(labels) == 4:
        try:
            if all(0 <= int(lbl) <= 255 for lbl in labels):
                return domain
        except ValueError:
            pass  # Not an IP — continue with normal logic.

    # IPv6 addresses pass through unchanged (they contain colons not dots,
    # but the port-strip above handles the bracket notation [:1] edge case).

    # With fewer than 2 labels we cannot form a registrable domain.
    if len(labels) < 2:
        return domain

    # Check for a known two-part TLD by examining the last two labels.
    candidate_two_part_tld: str = f"{labels[-2]}.{labels[-1]}"
    if candidate_two_part_tld in _KNOWN_MULTI_PART_TLDS:
        # Registrable domain requires the third-from-last label.
        if len(labels) >= 3:
            return f"{labels[-3]}.{labels[-2]}.{labels[-1]}"
        else:
            # Input was itself only the two-part TLD (e.g. "co.uk") — return as-is.
            return domain
    else:
        # Standard single-part TLD: registrable domain = last two labels.
        return f"{labels[-2]}.{labels[-1]}"


def _hash_domain_to_bucket(normalized_domain: str) -> int:
    """
    Hash a normalised domain string to a token in [DOMAIN_TOKEN_OFFSET, DOMAIN_TOKEN_END].

    Hash function: MD5, take first two bytes as a big-endian uint16, mask to 12
    bits (& 0x0FFF), then modulo _DOMAIN_HASH_BUCKETS (3072) to constrain to the
    domain token range.

    Why MD5 over SHA-256:
      MD5 is faster and produces 128 bits — far more than the 12 bits we use.
      Cryptographic strength is irrelevant; we need speed and consistency.
      MD5 is deterministic across all Python versions and process restarts.

    Why 12-bit mask then modulo 3072 (not direct 12-bit use):
      A direct 12-bit index would produce values in [0, 4095], requiring an
      offset of 1024 to reach the domain range — but 0 + 1024 = 1024 through
      4095 + 1024 = 5119, which exceeds VOCAB_SIZE.  Constraining to 3072
      buckets via modulo keeps the token within [1024, 4095].
      Modulo introduces a small non-uniformity (biases values 0–1023 slightly
      over values 1024–3071) but this is negligible for routing correctness.

    Collision behaviour:
      Hash collisions are expected and acceptable — they cause two different
      domains to share a token.  This is the intended design: the SSM learns
      structural patterns at the token level, and collisions group structurally
      similar domains (which often hash nearby) as a soft prior.

    Args:
        normalized_domain: The lowercased registrable domain (output of
                           normalize_domain()). Must not be empty.

    Returns:
        An integer in [DOMAIN_TOKEN_OFFSET, DOMAIN_TOKEN_END] (i.e. [1024, 4095]).

    Raises:
        ValueError: if normalized_domain is empty.
    """
    if not normalized_domain:
        raise ValueError(
            "normalized_domain must be non-empty.  "
            "Call normalize_domain() before _hash_domain_to_bucket()."
        )
    digest: bytes = hashlib.md5(normalized_domain.encode("utf-8")).digest()
    # Big-endian uint16 from the first two bytes.
    raw_uint16: int = struct.unpack(">H", digest[:2])[0]  # range [0, 65535]
    # Mask to 12 bits → range [0, 4095].
    raw_12bit: int = raw_uint16 & 0x0FFF  # range [0, 4095]
    # Constrain to [0, 3071] then shift to [1024, 4095].
    bucket: int = raw_12bit % _DOMAIN_HASH_BUCKETS  # range [0, 3071]
    return DOMAIN_TOKEN_OFFSET + bucket


# ═════════════════════════════════════════════════════════════════════════════
# CORE TOKEN ENCODING FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def topology_class_to_token(topology_class: str) -> int:
    """
    Map a topology class string to its vocabulary token.

    The mapping is derived from the position of *topology_class* in
    contracts.TOPOLOGY_CLASSES, shifted by TOPOLOGY_TOKEN_OFFSET.  The order
    in TOPOLOGY_CLASSES is canonical; this function must never hardcode the
    numeric value for any class.

    For topology classes not in TOPOLOGY_CLASSES (runtime-registered classes
    or typos), this function maps to the GENERIC_HTML token and logs a WARNING.
    The fallback prevents an unknown class from raising an exception on the
    critical path (where every millisecond matters), while making the
    miss observable in structured logs.

    Args:
        topology_class: Topology class string as defined in contracts.py,
                        e.g. "NEWS_ARTICLE", "SAAS_DOCS", "REST_API_JSON".

    Returns:
        Integer token in [TOPOLOGY_TOKEN_OFFSET, TOPOLOGY_TOKEN_END] (i.e. [1, 18]).

    Examples:
        >>> topology_class_to_token("NEWS_ARTICLE")
        1
        >>> topology_class_to_token("GENERIC_HTML")
        18
        >>> topology_class_to_token("UNKNOWN_CLASS")  # falls back to GENERIC_HTML
        18
    """
    token: Optional[int] = _TOPOLOGY_CLASS_TO_TOKEN.get(topology_class)
    if token is None:
        _log.warning(
            "topology_class_to_token: unrecognised topology class %r — "
            "falling back to GENERIC_HTML token %d.  "
            "If this class is newly registered, update contracts.TOPOLOGY_CLASSES.",
            topology_class,
            TOPOLOGY_TOKEN_GENERIC_HTML,
        )
        return TOPOLOGY_TOKEN_GENERIC_HTML
    return token


def domain_to_token(domain: str) -> int:
    """
    Map a domain name (with or without subdomains) to a domain hash token.

    Calls normalize_domain() first, then _hash_domain_to_bucket().  The full
    pipeline is: raw domain → registrable domain → MD5 hash → 12-bit bucket
    → token in [1024, 4095].

    Args:
        domain: Raw domain name, possibly with subdomains, port, or
                mixed case.  May be the hostname portion of a URL.

    Returns:
        Integer token in [DOMAIN_TOKEN_OFFSET, DOMAIN_TOKEN_END] (i.e. [1024, 4095]).

    Raises:
        ValueError: if *domain* is empty after stripping port and whitespace.

    Examples:
        >>> t = domain_to_token("docs.stripe.com")
        >>> t == domain_to_token("api.stripe.com")   # True — both → stripe.com
        True
        >>> 1024 <= t <= 4095
        True
        >>> domain_to_token("docs.stripe.com") == domain_to_token("stripe.com")
        True
    """
    if not domain:
        raise ValueError(
            "domain must be a non-empty string.  "
            "Pass the hostname portion of the URL, not the full URL."
        )
    normalized: str = normalize_domain(domain)
    if not normalized:
        raise ValueError(
            f"domain {domain!r} normalises to an empty string.  "
            "Verify that the input is a valid hostname."
        )
    return _hash_domain_to_bucket(normalized)


def structural_signal_to_token(signal: str) -> int:
    """
    Map a structural signal name to its vocabulary token.

    Looks up *signal* (case-insensitive) in _SIGNAL_NAME_TO_TOKEN.  If the
    signal is not found, returns PRIMITIVE_UNKNOWN and logs at DEBUG level.

    Unknown signals are never silently dropped.  They are mapped to
    PRIMITIVE_UNKNOWN so the model always receives a token — but the debug log
    makes the gap visible.  If a new signal category is added to the crawler,
    add it to _SIGNAL_NAME_TO_TOKEN before it can produce a named token.

    Args:
        signal: Structural signal name as emitted by domain_analyzer.py,
                e.g. "cloudflare", "wordpress", "tls_1_3", "status_200".

    Returns:
        Integer token in [STRUCTURAL_TOKEN_OFFSET, STRUCTURAL_TOKEN_END]
        (i.e. [19, 1023]).  Unknown signals return PRIMITIVE_UNKNOWN (83).

    Examples:
        >>> structural_signal_to_token("cloudflare")
        19
        >>> structural_signal_to_token("status_200")
        77
        >>> structural_signal_to_token("made_up_signal")
        83
    """
    normalised_signal: str = signal.strip().lower()
    token: Optional[int] = _SIGNAL_NAME_TO_TOKEN.get(normalised_signal)
    if token is None:
        _log.debug(
            "structural_signal_to_token: unrecognised signal %r — "
            "mapping to PRIMITIVE_UNKNOWN (%d).  "
            "Add to _SIGNAL_NAME_TO_TOKEN if this is a new canonical signal.",
            signal,
            PRIMITIVE_UNKNOWN,
        )
        return PRIMITIVE_UNKNOWN
    return token


def quantize_intent_dimension(value: float) -> int:
    """
    Quantize one float32 intent vector dimension to a 2-bit level (0–3).

    Applies INTENT_QUANT_THRESHOLDS to map *value* to one of four discrete
    levels using exclusive upper boundaries:
      value <  -0.5  → 0  (strongly negative)
      value in [-0.5,  0.0) → 1  (mildly negative)
      value in [ 0.0,  0.5] → 2  (mildly positive — includes exactly 0.0)
      value >   0.5  → 3  (strongly positive)

    Boundary ownership:
      The boundary -0.5 belongs to level 1 (half-open interval on the left).
      The boundary 0.0 belongs to level 2.
      The boundary +0.5 belongs to level 2.
      Values exactly at +0.5 are level 2 (upper boundary belongs to level 2).
      Values above +0.5 are level 3.

    The returned value is always in {0, 1, 2, 3}.  Non-finite inputs (inf,
    nan) are clamped:
      -inf → 0 (below all thresholds)
      +inf → 3 (above all thresholds)
      nan  → 1 (treated as zero; this is a data quality fallback, not a contract)

    Args:
        value: A single float32 intent vector component.  Typically in
               approximately [-1.0, 1.0] for unit-normalised intent vectors.

    Returns:
        An integer in {0, 1, 2, 3}.

    Examples:
        >>> quantize_intent_dimension(-0.7)
        0
        >>> quantize_intent_dimension(-0.5)
        1
        >>> quantize_intent_dimension(0.0)
        2
        >>> quantize_intent_dimension(0.5)
        2
        >>> quantize_intent_dimension(0.6)
        3
    """
    # Handle non-finite values defensively.
    if value != value:  # NaN check (NaN != NaN is True)
        _log.debug("quantize_intent_dimension: received NaN — treating as 0.0 (level 1)")
        return 1
    if value == float("-inf"):
        return 0
    if value == float("inf"):
        return 3

    t_low, t_mid, t_high = INTENT_QUANT_THRESHOLDS  # -0.5, 0.0, 0.5
    if value < t_low:
        return 0
    elif value < t_mid:
        return 1
    elif value <= t_high:
        return 2
    else:
        return 3


# ═════════════════════════════════════════════════════════════════════════════
# INTENT VECTOR ENCODING
# ═════════════════════════════════════════════════════════════════════════════

def intent_vector_to_tokens(intent_vector: Sequence[float]) -> List[int]:
    """
    Encode a 256-dimensional float32 intent vector into a list of intent tokens.

    Algorithm (in full):
      1. Validate that len(intent_vector) == INTENT_VECTOR_DIM (256).
      2. Quantize each dimension with quantize_intent_dimension():
           raw float → level in {0, 1, 2, 3}.
      3. Partition the 256 quantized levels into INTENT_TOKENS_PER_VECTOR (43)
         chunks of INTENT_CHUNK_SIZE (6) dimensions each:
           chunk 0 : dims 0–5
           chunk 1 : dims 6–11
           ...
           chunk 41: dims 246–251
           chunk 42: dims 252–255  (4 real dims + 2 zero-padded dims)
      4. For each chunk, compute the base-4 packed value:
           chunk_value = d[0]*4^5 + d[1]*4^4 + d[2]*4^3 + d[3]*4^2 + d[4]*4 + d[5]
         This is equivalent to treating the 6 levels as digits in base 4,
         most-significant digit first.
      5. Produce the token: INTENT_TOKEN_OFFSET + chunk_value.
         chunk_value ∈ [0, 4095] → token ∈ [4096, 8191] ✓

    The encoding is fully deterministic: same vector → same token list across
    all process restarts and Python versions (given fixed INTENT_QUANT_THRESHOLDS).

    Args:
        intent_vector: A sequence of exactly INTENT_VECTOR_DIM (256) floats.
                       Typically produced by the intent encoder before being
                       passed to the tokenizer.

    Returns:
        A list of exactly INTENT_TOKENS_PER_VECTOR (43) integers, each in
        [INTENT_TOKEN_OFFSET, INTENT_TOKEN_END] (i.e. [4096, 8191]).

    Raises:
        ValueError: if len(intent_vector) != INTENT_VECTOR_DIM.

    Examples:
         >>> tokens = intent_vector_to_tokens([0.0] * 256)
         >>> len(tokens)
              43
         >>> min(tokens) >= 4096 and max(tokens) <= 8191
              True
         >>> # All-zeros vector → all dims quantize to level 2 → chunk_value = 2*(4^5+4^4+4^3+4^2+4+1) = 2*1365 = 2730
         >>> tokens[0] == 4096 + 2730
              True
    """
    if len(intent_vector) != INTENT_VECTOR_DIM:
        raise ValueError(
            f"intent_vector must have exactly {INTENT_VECTOR_DIM} dimensions, "
            f"got {len(intent_vector)}.  "
            "The intent encoder must produce a fixed-size 256-dim vector."
        )

    # Step 1: Quantize all 256 dimensions at once.
    quantized: List[int] = [quantize_intent_dimension(float(v)) for v in intent_vector]

    # Step 2: Build one token per chunk.
    tokens: List[int] = []
    for chunk_idx in range(INTENT_TOKENS_PER_VECTOR):
        start: int = chunk_idx * INTENT_CHUNK_SIZE
        end: int = min(start + INTENT_CHUNK_SIZE, INTENT_VECTOR_DIM)
        chunk: List[int] = quantized[start:end]

        # Pad last (partial) chunk with zeros to reach INTENT_CHUNK_SIZE.
        # Zero-padding is applied at the end of the chunk (least-significant
        # positions), so real dimensions always occupy the high-order bits.
        pad_count: int = INTENT_CHUNK_SIZE - len(chunk)
        if pad_count > 0:
            chunk.extend([0] * pad_count)

        # Pack 6 levels (base-4 digits) into a single 12-bit value.
        # Most-significant digit first: chunk[0] contributes 4^5 = 1024.
        chunk_value: int = 0
        for level in chunk:
            chunk_value = chunk_value * 4 + level
        # chunk_value ∈ [0, 4095] guaranteed by the base-4 arithmetic.

        tokens.append(INTENT_TOKEN_OFFSET + chunk_value)

    return tokens


# ═════════════════════════════════════════════════════════════════════════════
# PADDING UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def pad_sequence(
    tokens: List[int],
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """
    Pad or truncate *tokens* to exactly *max_length* positions.

    Padding is applied on the right using PAD_TOKEN (0).  Truncation is applied
    on the right (oldest tokens removed last, newest tokens may be truncated
    before structural tokens if the caller has already applied anchor-aware
    truncation in encode_domain_event()).

    This function is a pure padding utility.  It does NOT re-anchor topology
    class or domain tokens after truncation — that responsibility belongs to
    encode_domain_event() and encode_continuation().  Do not call this function
    on a raw token list and assume the result is well-formed: always go through
    encode_domain_event() or encode_continuation().

    Two outputs are returned simultaneously because the attention mask is
    always computed from the same padding decision, and computing them together
    avoids a second pass through the padded list.

    Args:
        tokens:     List of integer token IDs (may be shorter or longer than
                    max_length).
        max_length: Target sequence length.  Must be ≥ 2 (minimum useful
                    sequence has at least a topology token and a domain token).

    Returns:
        A tuple (padded_tokens, attention_mask) where:
          padded_tokens   — list of max_length integers; positions beyond len(tokens)
                            are PAD_TOKEN (0).
          attention_mask  — list of max_length integers: 1 where the token is real,
                            0 where the token is PAD.

    Raises:
        ValueError: if max_length < 2.

    Examples:
        >>> padded, mask = pad_sequence([1, 24, 77], max_length=6)
        >>> padded
        [1, 24, 77, 0, 0, 0]
        >>> mask
        [1, 1, 1, 0, 0, 0]
        >>> padded, mask = pad_sequence([1, 24, 77, 80, 19, 68, 42], max_length=4)
        >>> padded
        [1, 24, 77, 80]
        >>> mask
        [1, 1, 1, 1]
    """
    if max_length < 2:
        raise ValueError(
            f"max_length must be ≥ 2, got {max_length}.  "
            "A useful sequence needs at least a topology token and a domain token."
        )

    n: int = len(tokens)

    if n >= max_length:
        # Truncate from the right — caller is responsible for anchor placement.
        padded: List[int] = list(tokens[:max_length])
        mask: List[int] = [1] * max_length
    else:
        # Pad on the right.
        pad_len: int = max_length - n
        padded = list(tokens) + [PAD_TOKEN] * pad_len
        mask = [1] * n + [0] * pad_len

    return padded, mask


def make_attention_mask(tokens: List[int]) -> List[int]:
    """
    Build a binary attention mask for an already-padded token list.

    Returns 1 for every position where the token is not PAD_TOKEN, 0 otherwise.
    Intended for cases where the padded list already exists and only the mask
    needs to be (re)derived — e.g. in decode paths and sequence_stats().

    Args:
        tokens: A list of integer token IDs, possibly containing PAD_TOKEN (0).

    Returns:
        A list of the same length as *tokens* with values in {0, 1}.

    Examples:
        >>> make_attention_mask([1, 1024, 77, 0, 0, 0])
        [1, 1, 1, 0, 0, 0]
        >>> make_attention_mask([1, 2048, 0])
        [1, 1, 0]
    """
    return [0 if t == PAD_TOKEN else 1 for t in tokens]


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE ASSEMBLY — INTERNAL HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def _sort_structural_tokens(primitive_tokens: List[int]) -> List[int]:
    """
    Sort a list of structural primitive token IDs by signal priority.

    Lower SIGNAL_PRIORITY_RANK value = higher priority = earlier in the output.
    Tokens not found in SIGNAL_PRIORITY_RANK (reserved-range tokens passed by
    advanced callers) sort to _DEFAULT_SIGNAL_RANK (200), placing them at the
    end of the structural block.

    Stability guarantee: tokens with equal priority rank maintain their
    original relative order (Python's sort is stable).

    Args:
        primitive_tokens: Unsorted list of structural primitive token IDs.
                          May contain duplicates (duplicates are preserved
                          and sorted along with unique tokens).

    Returns:
        A new list with the same elements sorted by ascending priority rank.
    """
    return sorted(
        primitive_tokens,
        key=lambda tok: SIGNAL_PRIORITY_RANK.get(tok, _DEFAULT_SIGNAL_RANK),
    )


def _build_core_sequence(
    topology_token: int,
    domain_token: int,
    sorted_primitive_tokens: List[int],
    intent_tokens: List[int],
) -> List[int]:
    """
    Assemble the ordered token sequence before truncation and padding.

    SSM position semantics:
      Position 0 — topology class token: positions the entire sequence within
                   the topology-class embedding space.  Must ALWAYS be first.
      Position 1 — domain hash token: anchors the sequence to a specific domain
                   cluster.  Must ALWAYS be second.
      Positions 2..N — structural primitives, sorted by priority rank (most
                   routing-decisive signals first).
      Positions N+1..N+43 — intent signal tokens (if provided), one per chunk.
      Remaining — filled with PAD_TOKEN by pad_sequence().

    Args:
        topology_token:          Single topology class token (int in [1, 18]).
        domain_token:            Single domain hash token (int in [1024, 4095]).
        sorted_primitive_tokens: Structural tokens sorted by SIGNAL_PRIORITY_RANK.
        intent_tokens:           List of 0 or INTENT_TOKENS_PER_VECTOR (43) tokens.

    Returns:
        Ordered list: [topology, domain, *primitives, *intent].
        Length varies; callers must apply truncation and padding.
    """
    return [topology_token, domain_token] + sorted_primitive_tokens + intent_tokens


def _truncate_to_max_length(
    sequence: List[int],
    max_length: int,
    topology_token: int,
    domain_token: int,
) -> List[int]:
    """
    Truncate *sequence* to *max_length* while preserving the two mandatory
    anchor tokens at positions 0 and 1.

    Truncation strategy (right-to-left):
      1. Drop intent tokens first (they are the lowest-priority component).
      2. If still too long, drop excess structural tokens from the right.
      3. The topology class token (position 0) and domain token (position 1)
         are NEVER dropped.  If max_length < 2, a ValueError is raised because
         even the mandatory anchors cannot fit.

    Args:
        sequence:        The full assembled token sequence before truncation.
        max_length:      Target length.
        topology_token:  The mandatory topology class token (for re-anchoring).
        domain_token:    The mandatory domain hash token (for re-anchoring).

    Returns:
        A list of length min(len(sequence), max_length) with topology and domain
        tokens guaranteed at positions 0 and 1.

    Raises:
        ValueError: if max_length < 2.
    """
    if max_length < 2:
        raise ValueError(
            f"max_length must be ≥ 2 to accommodate the mandatory topology and "
            f"domain anchor tokens.  Got max_length={max_length}."
        )

    if len(sequence) <= max_length:
        return sequence

    # Truncate from the right.
    # This always preserves position 0 (topology) and position 1 (domain)
    # because they were placed first in _build_core_sequence().
    truncated: List[int] = sequence[:max_length]

    # Sanity check: anchors must be intact (defensive, should never fail).
    if truncated[0] != topology_token or truncated[1] != domain_token:
        # This indicates a bug in _build_core_sequence — re-anchor defensively.
        _log.error(
            "_truncate_to_max_length: anchor tokens misplaced after truncation "
            "(topology expected %d got %d, domain expected %d got %d) — "
            "re-anchoring.  This is a tokenizer bug; please file an issue.",
            topology_token, truncated[0],
            domain_token,   truncated[1],
        )
        truncated[0] = topology_token
        truncated[1] = domain_token

    return truncated


# ═════════════════════════════════════════════════════════════════════════════
# PUBLIC SEQUENCE ASSEMBLY API
# ═════════════════════════════════════════════════════════════════════════════

def encode_domain_event(
    topology_class: str,
    domain: str,
    structural_signals: List[str],
    intent_vector: Optional[List[float]] = None,
    max_length: int = MAX_SEQ_LEN,
) -> torch.Tensor:
    """
    Encode one domain topology event into a fixed-length token tensor.

    This is the primary entry point for the WLM.  It encodes a complete
    domain event — the topology class, the domain name, observable structural
    signals, and an optional intent vector — into a 1-D integer tensor that
    mamba_router.py processes directly.

    Token assembly order (SSM is order-sensitive):
      [0]      topology class token    — mandatory, always position 0
      [1]      domain hash token       — mandatory, always position 1
      [2..N]   structural primitives   — sorted by SIGNAL_PRIORITY_RANK
      [N+1..M] intent signal tokens    — 43 tokens if intent_vector provided
      [M+1..]  PAD_TOKEN (0)           — fill to max_length

    Truncation (if len > max_length):
      Intent tokens are dropped first (right-truncation drops them before
      structural tokens because intent appears last in the sequence).
      If still too long, excess structural tokens are dropped from the right
      (lowest-priority signals are rightmost after sorting).
      The topology class token and domain token are NEVER dropped.
      They are always at positions 0 and 1 regardless of truncation.

    Deduplication:
      Duplicate signal strings in structural_signals map to the same token.
      Duplicate tokens appear only once in the sorted structural block.
      (The deduplication is intentional: signalling "cloudflare" three times
      is the same as signalling it once for routing purposes.)

    Args:
        topology_class:     Topology class string, e.g. "NEWS_ARTICLE".
                            Unknown classes fall back to GENERIC_HTML with a
                            WARNING log.
        domain:             Domain name, possibly with subdomains.
                            Normalised internally; do not pre-normalise.
        structural_signals: List of structural signal name strings as emitted
                            by domain_analyzer.py.  Order does not matter —
                            the tokenizer sorts by priority rank.  Unknown
                            signal names map to PRIMITIVE_UNKNOWN.  May be
                            empty.
        intent_vector:      Optional 256-dimensional float32 intent vector.
                            If None, no intent tokens are included.
                            If provided, must have exactly INTENT_VECTOR_DIM
                            (256) elements.
        max_length:         Maximum sequence length.  Default MAX_SEQ_LEN (512).
                            Must be ≥ 2.

    Returns:
        A 1-D torch.Tensor of dtype torch.long, shape (max_length,).
        - Positions with real tokens: values in [1, VOCAB_SIZE - 1].
        - Padding positions: value = PAD_TOKEN (0).

    Raises:
        ValueError: if domain is empty, max_length < 2, or intent_vector has
                    the wrong dimensionality.

    Examples:
        >>> t = encode_domain_event(
        ...     topology_class="NEWS_ARTICLE",
        ...     domain="api.nytimes.com",
        ...     structural_signals=["cloudflare", "tls_1_3", "status_200"],
        ... )
        >>> t.shape
        torch.Size([512])
        >>> t.dtype
        torch.int64
        >>> t[0].item()   # topology class token for NEWS_ARTICLE
        1
        >>> t[1].item() >= 1024  # domain hash token
        True
    """
    # ── Step 1: Encode the two mandatory anchor tokens ─────────────────────
    topology_token: int = topology_class_to_token(topology_class)
    domain_token:   int = domain_to_token(domain)

    # ── Step 2: Encode and deduplicate structural signals ──────────────────
    seen_primitive_tokens: Dict[int, None] = {}  # ordered set via dict keys
    for sig in structural_signals:
        prim_tok: int = structural_signal_to_token(sig)
        seen_primitive_tokens[prim_tok] = None
    deduplicated_primitives: List[int] = list(seen_primitive_tokens.keys())

    # Sort by signal priority rank.
    sorted_primitives: List[int] = _sort_structural_tokens(deduplicated_primitives)

    # ── Step 3: Encode intent vector (if provided) ─────────────────────────
    encoded_intent: List[int] = []
    if intent_vector is not None:
        encoded_intent = intent_vector_to_tokens(intent_vector)

    # ── Step 4: Assemble the full pre-truncation sequence ──────────────────
    full_sequence: List[int] = _build_core_sequence(
        topology_token, domain_token, sorted_primitives, encoded_intent
    )

    # ── Step 5: Truncate if over max_length ────────────────────────────────
    truncated_sequence: List[int] = _truncate_to_max_length(
        full_sequence, max_length, topology_token, domain_token
    )

    # ── Step 6: Pad to exactly max_length ─────────────────────────────────
    padded, _mask = pad_sequence(truncated_sequence, max_length)

    # ── Step 7: Convert to tensor ──────────────────────────────────────────
    return torch.tensor(padded, dtype=torch.long)


# ═════════════════════════════════════════════════════════════════════════════
# DOMAIN EVENT SPECIFICATION (for encode_batch)
# ═════════════════════════════════════════════════════════════════════════════

class DomainEventSpec(TypedDict, total=False):
    """
    Typed dictionary specifying one domain event for encode_batch().

    Required keys:
        topology_class (str): Topology class string.
        domain (str): Domain name (subdomains allowed; normalised internally).
        structural_signals (List[str]): Structural signal names.

    Optional keys:
        intent_vector (List[float]): 256-dim float32 intent vector.
                                     Omit or set to None to skip intent encoding.

    Example::

        event: DomainEventSpec = {
            "topology_class": "SAAS_DOCS",
            "domain": "docs.stripe.com",
            "structural_signals": ["cloudflare", "tls_1_3", "status_200"],
            "intent_vector": [0.1, -0.3, ...],  # 256 floats
        }
    """
    topology_class:      str
    domain:              str
    structural_signals:  List[str]
    intent_vector:       Optional[List[float]]


def _validate_domain_event_spec(spec: DomainEventSpec, idx: int) -> None:
    """
    Validate that a DomainEventSpec has all required keys with correct types.

    Raises:
        ValueError: with a descriptive message if any required key is missing
                    or has the wrong type.

    Args:
        spec: The spec dict to validate.
        idx:  The index in the batch list (for error messages).
    """
    required_keys = ("topology_class", "domain", "structural_signals")
    for key in required_keys:
        if key not in spec:
            raise ValueError(
                f"encode_batch: event at index {idx} is missing required key "
                f"{key!r}.  DomainEventSpec requires topology_class, domain, "
                f"and structural_signals."
            )
    if not isinstance(spec.get("topology_class"), str):  # type: ignore[arg-type]
        raise ValueError(
            f"encode_batch: event[{idx}].topology_class must be a str, "
            f"got {type(spec.get('topology_class')).__name__}."
        )
    if not isinstance(spec.get("domain"), str):  # type: ignore[arg-type]
        raise ValueError(
            f"encode_batch: event[{idx}].domain must be a str, "
            f"got {type(spec.get('domain')).__name__}."
        )
    if not isinstance(spec.get("structural_signals"), list):  # type: ignore[arg-type]
        raise ValueError(
            f"encode_batch: event[{idx}].structural_signals must be a list, "
            f"got {type(spec.get('structural_signals')).__name__}."
        )
    iv = spec.get("intent_vector")  # type: ignore[assignment]
    if iv is not None and len(iv) != INTENT_VECTOR_DIM:
        raise ValueError(
            f"encode_batch: event[{idx}].intent_vector has {len(iv)} dimensions, "
            f"expected {INTENT_VECTOR_DIM}."
        )


# ═════════════════════════════════════════════════════════════════════════════
# BATCH ENCODING
# ═════════════════════════════════════════════════════════════════════════════

def encode_batch(
    events: List[DomainEventSpec],
    max_length: int = MAX_SEQ_LEN,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Encode a list of domain events into a padded batch of token tensors.

    Each event is first encoded individually (using encode_domain_event),
    then all encoded sequences are padded to the length of the LONGEST
    sequence in the batch — not to max_length.  This minimises wasted compute
    on short sequences in heterogeneous batches.

    Padding is still capped at max_length: no sequence in the output can
    be longer than max_length even if the longest individual encoding is shorter
    than max_length.

    Batch padding (second padding scenario):
      All sequences are padded to: min(max_sequence_length_in_batch, max_length)
      where max_sequence_length_in_batch is the length of the longest real
      token sequence (before padding) in the batch.

      Example: a batch of 8 events where the longest has 47 real tokens and
      max_length=512 produces tensors of shape (8, 47), not (8, 512).  This
      saves approximately 465 × 8 = 3,720 integer comparisons per SSM forward
      pass for that batch.

    Args:
        events:     List of DomainEventSpec dicts.  May be empty; if empty,
                    returns empty tensors of shape (0, 0).
        max_length: Hard ceiling on sequence length.  Each event is first
                    truncated/padded to max_length, then the batch is padded
                    to the longest sequence in the batch (which is ≤ max_length).

    Returns:
        A tuple (token_tensor, attention_mask_tensor) where:
          token_tensor          — torch.Tensor of shape (N, batch_seq_len),
                                  dtype torch.long.  PAD_TOKEN at padding positions.
          attention_mask_tensor — torch.Tensor of shape (N, batch_seq_len),
                                  dtype torch.long.  1 for real tokens, 0 for PAD.

        N = len(events).  batch_seq_len = max real sequence length in the batch
        (capped at max_length).

    Raises:
        ValueError: if any event spec is malformed (missing keys, wrong types,
                    wrong intent_vector dimensionality).

    Examples:
           >>> from typing import cast, List, Tuple
           >>> import torch
           >>> events = cast(List[DomainEventSpec], [
           ...     {"topology_class": "SAAS_DOCS", "domain": "docs.stripe.com",
           ...      "structural_signals": ["cloudflare", "tls_1_3"]},
           ...     {"topology_class": "NEWS_ARTICLE", "domain": "nytimes.com",
           ...      "structural_signals": ["status_200"]},
           ... ])
           >>> result = cast(Tuple[torch.Tensor, torch.Tensor], encode_batch(events))
           >>> result[0].shape[0]
           2
           >>> result[0].shape == result[1].shape
    True
    True
    """
    if not events:
        empty_tokens: torch.Tensor = torch.zeros((0, 0), dtype=torch.long)
        empty_mask:   torch.Tensor = torch.zeros((0, 0), dtype=torch.long)
        return empty_tokens, empty_mask

    # Validate all specs upfront before any encoding work.
    for idx, spec in enumerate(events):
        _validate_domain_event_spec(spec, idx)

    # Encode each event individually to a max_length-length tensor first.
    # This applies per-event truncation and padding.
    encoded_tensors: List[torch.Tensor] = []
    for spec in events:
        tensor = encode_domain_event(
            topology_class=spec["topology_class"],
            domain=spec["domain"],
            structural_signals=spec["structural_signals"],
            intent_vector=spec.get("intent_vector"),  # type: ignore[arg-type]
            max_length=max_length,
        )
        encoded_tensors.append(tensor)

    # Determine the batch sequence length: the length of the shortest non-PAD
    # suffix that would need to be retained across all sequences.
    # We compute the last non-PAD position per sequence, then take the max.
    max_real_length: int = 0
    per_sequence_real_lengths: List[int] = []
    for tensor in encoded_tensors:
        # Work with Python ints for speed — no GPU at tokenization time.
        tok_list: List[int] = tensor.tolist()  # type: ignore[attr-defined]
        # Find last non-PAD index (scan from right).
        real_len: int = 0
        for pos in range(len(tok_list) - 1, -1, -1):
            if tok_list[pos] != PAD_TOKEN:
                real_len = pos + 1
                break
        per_sequence_real_lengths.append(real_len)
        if real_len > max_real_length:
            max_real_length = real_len

    # Edge case: all sequences are all-PAD (degenerate batch).
    if max_real_length == 0:
        max_real_length = 2  # Minimum meaningful sequence: topology + domain.

    batch_seq_len: int = min(max_real_length, max_length)

    # Build batch tensors by truncating (or keeping) each sequence to batch_seq_len
    # and building the attention mask.
    batch_tokens: List[List[int]] = []
    batch_masks:  List[List[int]] = []

    for tensor in encoded_tensors:
        tok_list = tensor.tolist()  # type: ignore[attr-defined]
        # Slice to batch_seq_len (no need to pad — all are already max_length).
        sliced: List[int] = tok_list[:batch_seq_len]
        mask:   List[int] = make_attention_mask(sliced)
        batch_tokens.append(sliced)
        batch_masks.append(mask)

    token_tensor:  torch.Tensor = torch.tensor(batch_tokens, dtype=torch.long)
    mask_tensor:   torch.Tensor = torch.tensor(batch_masks,  dtype=torch.long)

    return token_tensor, mask_tensor


# ═════════════════════════════════════════════════════════════════════════════
# CONTINUATION ENCODING
#
# Continuation encoding handles the third padding scenario: a new signal
# arrives for a domain that already has a token sequence in the MFT.  The new
# tokens are appended to the existing sequence.  If the combined sequence
# would exceed max_length, the oldest tokens are dropped from the LEFT —
# but the topology class token and domain token are always re-inserted at
# positions 0 and 1 after any left-truncation.
#
# The re-anchoring guarantee is load-bearing for MFT correctness:
#   - The SSM always knows which topology class the sequence belongs to
#     (position 0 always contains the topology token).
#   - The SSM always knows which domain the sequence belongs to
#     (position 1 always contains the domain token).
#   Without this guarantee, left-truncation after a long crawl history would
#   cause the SSM to process structurally uncontextualised tokens.
# ═════════════════════════════════════════════════════════════════════════════

def encode_continuation(
    previous_tokens: torch.Tensor,
    topology_class:  str,
    domain:          str,
    structural_signals: List[str],
    intent_vector:   Optional[List[float]] = None,
    max_length:      int = MAX_SEQ_LEN,
) -> torch.Tensor:
    """
    Append a new domain event's tokens to a previous token sequence.

    Used when a new structural or intent signal arrives for a domain that
    already has a partially-encoded token sequence (e.g. a domain already
    visited during this crawl session, now producing new signals).

    The new tokens are encoded from the incoming signals using the same
    path as encode_domain_event(), then appended to the existing sequence
    (excluding the existing sequence's trailing PAD tokens).

    Left-truncation with re-anchoring:
      If the combined sequence exceeds max_length, excess tokens are dropped
      from the LEFT (oldest tokens).  After left-truncation the topology class
      token and domain token are always re-inserted at positions 0 and 1,
      overwriting whatever was there.  This ensures the mandatory context
      anchors are always present.

    Correct usage:
      previous_tokens should be the tensor returned by a previous call to
      encode_domain_event() or encode_continuation() for the SAME topology
      class and domain.  Passing a tensor from a different domain is not
      prevented but is semantically incorrect and will corrupt the continuation
      (the domain anchor will be overwritten with the new domain token).

    Args:
        previous_tokens:    1-D torch.Tensor (shape: (seq_len,), dtype: long)
                            containing the previous encoded sequence.  Trailing
                            PAD tokens are stripped before appending new tokens.
        topology_class:     Topology class for the new event.
        domain:             Domain name for the new event (normalised internally).
        structural_signals: New structural signal names to append.
        intent_vector:      Optional 256-dim intent vector to append.
        max_length:         Hard ceiling on the output sequence length.

    Returns:
        A 1-D torch.Tensor of shape (max_length,), dtype torch.long.
        PAD_TOKEN fills any positions beyond the real token content.

    Raises:
        ValueError: if previous_tokens is not 1-D, max_length < 2,
                    or intent_vector has wrong dimensionality.

    Examples:
        >>> t0 = encode_domain_event("SAAS_DOCS", "stripe.com", ["cloudflare"])
        >>> t1 = encode_continuation(t0, "SAAS_DOCS", "stripe.com", ["tls_1_3"])
        >>> t1.shape
        torch.Size([512])
        >>> t1[0].item()  # topology token always at position 0
        3
        >>> t1[1].item() == t0[1].item()  # domain token preserved
        True
    """
    if previous_tokens.ndim != 1:
        raise ValueError(
            f"previous_tokens must be a 1-D tensor, got shape {tuple(previous_tokens.shape)}.  "
            "Pass the output of encode_domain_event() or encode_continuation()."
        )

    # Resolve the mandatory anchor tokens for the new event.
    new_topology_token: int = topology_class_to_token(topology_class)
    new_domain_token:   int = domain_to_token(domain)

    # ── Step 1: Strip trailing PAD from previous sequence ─────────────────
    prev_list: List[int] = previous_tokens.tolist()  # type: ignore[attr-defined]
    # Find the last non-PAD position.
    last_real: int = -1
    for pos in range(len(prev_list) - 1, -1, -1):
        if prev_list[pos] != PAD_TOKEN:
            last_real = pos
            break
    prev_real: List[int] = prev_list[:last_real + 1] if last_real >= 0 else []

    # ── Step 2: Encode new signals (excluding anchor tokens — they come
    #    from the previous sequence or are re-inserted after truncation) ────
    seen_primitive_tokens: Dict[int, None] = {}
    for sig in structural_signals:
        prim_tok = structural_signal_to_token(sig)
        seen_primitive_tokens[prim_tok] = None
    sorted_new_primitives: List[int] = _sort_structural_tokens(list(seen_primitive_tokens))

    new_intent_tokens: List[int] = []
    if intent_vector is not None:
        new_intent_tokens = intent_vector_to_tokens(intent_vector)

    # ── Step 3: Build the new token block (no anchors — appended to existing)
    new_block: List[int] = sorted_new_primitives + new_intent_tokens

    # ── Step 4: Combine previous real tokens with new block ────────────────
    combined: List[int] = prev_real + new_block

    # ── Step 5: Left-truncate if over max_length, then re-anchor ───────────
    if len(combined) > max_length:
        # Drop oldest tokens from the left.
        # After truncation, positions 0 and 1 may be any token (not the anchors).
        # Re-anchor: force topology and domain tokens at positions 0 and 1.
        overflow: int = len(combined) - max_length
        truncated: List[int] = combined[overflow:]  # left-truncated

        # Re-anchor positions 0 and 1 by inserting the anchors at the front
        # and dropping the last two tokens (right-most, lowest priority) to
        # maintain max_length.
        # This ensures: [topology_token, domain_token, ...rest (max_length-2 tokens)]
        inner: List[int] = truncated[2:]        # everything after position 1
        reanchored: List[int] = (
            [new_topology_token, new_domain_token] + inner
        )
        # reanchored has exactly max_length elements.
        final_sequence: List[int] = reanchored

    elif len(combined) < 2:
        # Degenerate case: previous sequence was empty and no new signals.
        # Construct a minimal sequence with just the anchors.
        final_sequence = [new_topology_token, new_domain_token]
    else:
        # No truncation needed.  Ensure anchors are intact at positions 0 and 1.
        # (They should already be there from the previous encode call, but
        # defensive re-anchoring is cheap and correct.)
        final_sequence = combined
        final_sequence[0] = new_topology_token
        final_sequence[1] = new_domain_token

    # ── Step 6: Pad to max_length ─────────────────────────────────────────
    padded, _mask = pad_sequence(final_sequence, max_length)
    return torch.tensor(padded, dtype=torch.long)


# ═════════════════════════════════════════════════════════════════════════════
# DECODE FUNCTIONS — FOR DEBUGGING AND AUDIT ONLY
#
# These functions are the inverse of the encode path.  They are used to make
# token sequences human-readable in debug logs and audit trails.  They are NOT
# on the inference critical path and must NOT be called from mamba_router.py
# or any code that processes model outputs.
#
# The domain decode function returns a hint string (bucket description) because
# domain hashing is not invertible — multiple domains map to the same token.
# ═════════════════════════════════════════════════════════════════════════════

def token_to_topology_class(token: int) -> Optional[str]:
    """
    Return the topology class string for a given topology token.

    This is the inverse of topology_class_to_token().  It is exact: each
    topology token corresponds to exactly one topology class string.

    Args:
        token: An integer that may or may not be a topology class token.

    Returns:
        The topology class string (e.g. "NEWS_ARTICLE") if *token* is in
        [TOPOLOGY_TOKEN_OFFSET, TOPOLOGY_TOKEN_END], or None if the token is
        outside the topology range (could be a domain, structural, or intent
        token, or PAD).

    Examples:
        >>> token_to_topology_class(1)
        'NEWS_ARTICLE'
        >>> token_to_topology_class(18)
        'GENERIC_HTML'
        >>> token_to_topology_class(0) is None   # PAD_TOKEN
        True
        >>> token_to_topology_class(1024) is None  # domain token
        True
    """
    return _TOKEN_TO_TOPOLOGY_CLASS.get(token)


def token_to_domain_hint(token: int) -> str:
    """
    Return a human-readable description of a domain hash token's bucket.

    Domain hashing is not invertible: many domains map to the same token.
    This function returns a descriptive string indicating the bucket number
    and range, suitable for debug logs and audit trails.

    Args:
        token: An integer that may or may not be a domain hash token.

    Returns:
        If *token* is in [DOMAIN_TOKEN_OFFSET, DOMAIN_TOKEN_END]:
            A string of the form "domain_bucket[N]" where N is the 0-based
            bucket index (token - DOMAIN_TOKEN_OFFSET).
        Otherwise:
            A string indicating the token is not in the domain range, e.g.
            "not_a_domain_token(42)".

    Examples:
        >>> token_to_domain_hint(1024)
        'domain_bucket[0]'
        >>> token_to_domain_hint(1030)
        'domain_bucket[6]'
        >>> token_to_domain_hint(4095)
        'domain_bucket[3071]'
        >>> token_to_domain_hint(1)
        'not_a_domain_token(1)'
    """
    if DOMAIN_TOKEN_OFFSET <= token <= DOMAIN_TOKEN_END:
        bucket_idx: int = token - DOMAIN_TOKEN_OFFSET
        return f"domain_bucket[{bucket_idx}]"
    return f"not_a_domain_token({token})"


def token_to_primitive_name(token: int) -> Optional[str]:
    """
    Return the canonical signal name for a given structural primitive token.

    This is the inverse of structural_signal_to_token(), using the first
    registered alias for each token (see _TOKEN_TO_SIGNAL_NAME construction).

    Args:
        token: An integer that may or may not be a structural primitive token.

    Returns:
        The canonical signal name string (e.g. "cloudflare", "status_200") if
        *token* is in [STRUCTURAL_TOKEN_OFFSET, STRUCTURAL_TOKEN_END] AND
        has a registered name in _TOKEN_TO_SIGNAL_NAME.
        Returns "RESERVED(N)" for tokens in the reserved structural range
        (FIRST_RESERVED_PRIMITIVE to STRUCTURAL_TOKEN_END) that have no entry.
        Returns None if *token* is outside the structural range entirely.

    Examples:
        >>> token_to_primitive_name(19)
        'cloudflare'
        >>> token_to_primitive_name(77)
        'status_200'
        >>> token_to_primitive_name(83)
        'UNKNOWN_PRIMITIVE'
        >>> token_to_primitive_name(84)
        'RESERVED(84)'
        >>> token_to_primitive_name(0) is None   # PAD
        True
        >>> token_to_primitive_name(1024) is None  # domain token
        True
    """
    if not (STRUCTURAL_TOKEN_OFFSET <= token <= STRUCTURAL_TOKEN_END):
        return None
    name: Optional[str] = _TOKEN_TO_SIGNAL_NAME.get(token)
    if name is not None:
        return name
    # Token is in structural range but not named — it's in the reserved block.
    if token >= FIRST_RESERVED_PRIMITIVE:
        return f"RESERVED({token})"
    # Named structural tokens exhausted — this should not happen if constants are
    # consistent with _TOKEN_TO_SIGNAL_NAME.  Return None defensively.
    return None


def token_to_intent_hint(token: int) -> str:
    """
    Return a human-readable description of an intent signal token.

    Intent tokens encode a specific 6-dimensional quantized chunk of an intent
    vector at a specific chunk index.  Because the chunk index is not stored in
    the token, this function can only describe the packed chunk value.

    Args:
        token: An integer that may or may not be an intent signal token.

    Returns:
        If *token* is in [INTENT_TOKEN_OFFSET, INTENT_TOKEN_END]:
            A string of the form "intent_chunk_value[N]" where N is the packed
            base-4 chunk value (token - INTENT_TOKEN_OFFSET) in [0, 4095].
            The value can be decoded into 6 base-4 digits (quantized dim levels)
            for inspection: e.g. value 2730 = digits [2,2,2,2,2,2] (all level 2).
        Otherwise:
            A string "not_an_intent_token(N)".

    Examples:
        >>> token_to_intent_hint(4096)
        'intent_chunk_value[0]'
        >>> token_to_intent_hint(8191)
        'intent_chunk_value[4095]'
        >>> token_to_intent_hint(1024)
        'not_an_intent_token(1024)'
    """
    if INTENT_TOKEN_OFFSET <= token <= INTENT_TOKEN_END:
        chunk_value: int = token - INTENT_TOKEN_OFFSET
        return f"intent_chunk_value[{chunk_value}]"
    return f"not_an_intent_token({token})"


def decode_intent_chunk_value(chunk_value: int) -> List[int]:
    """
    Unpack a 12-bit chunk value into its 6 constituent base-4 digit levels.

    This is the inverse of the packing step in intent_vector_to_tokens().
    Useful for auditing what an intent token encodes.

    Args:
        chunk_value: Integer in [0, 4095] (token - INTENT_TOKEN_OFFSET).

    Returns:
        A list of 6 integers, each in {0, 1, 2, 3}.  The first element is the
        most-significant digit (corresponding to dims[chunk_start+0]).

    Raises:
        ValueError: if chunk_value is outside [0, _INTENT_CHUNK_BASE - 1].

    Examples:
        >>> decode_intent_chunk_value(0)
        [0, 0, 0, 0, 0, 0]
        >>> decode_intent_chunk_value(4095)
        [3, 3, 3, 3, 3, 3]
        >>> decode_intent_chunk_value(2730)
        [2, 2, 2, 2, 2, 2]
    """
    if not (0 <= chunk_value < _INTENT_CHUNK_BASE):
        raise ValueError(
            f"chunk_value must be in [0, {_INTENT_CHUNK_BASE - 1}], got {chunk_value}."
        )
    digits: List[int] = []
    remaining: int = chunk_value
    for _ in range(INTENT_CHUNK_SIZE):
        digits.append(remaining % 4)
        remaining //= 4
    digits.reverse()  # Most-significant digit first.
    return digits


def describe_sequence(
    tokens: torch.Tensor,
    max_tokens_to_show: int = 20,
) -> str:
    """
    Produce a human-readable description of a token sequence for debug logs.

    Shows the topology class, domain hint, first N structural tokens (with
    names), number of intent tokens, and number of padding tokens.

    Args:
        tokens:             1-D token tensor (output of any encode function).
        max_tokens_to_show: Maximum number of structural tokens to describe
                            individually.  Default 20.

    Returns:
        A multi-line string suitable for logging.  Does not include a trailing
        newline.

    Raises:
        ValueError: if tokens is not 1-D.
    """
    if tokens.ndim != 1:
        raise ValueError(f"tokens must be 1-D, got shape {tuple(tokens.shape)}.")

    tok_list: List[int] = tokens.tolist()  # type: ignore[attr-defined]
    lines: List[str] = [
        f"SequenceLength: {len(tok_list)}",
    ]

    if not tok_list:
        lines.append("  (empty sequence)")
        return "\n".join(lines)

    # Position 0: topology class.
    topo_tok: int = tok_list[0]
    topo_cls: Optional[str] = token_to_topology_class(topo_tok)
    lines.append(f"  [0] topology:  {topo_cls or '???'} (token={topo_tok})")

    # Position 1: domain hash.
    if len(tok_list) > 1:
        dom_tok: int = tok_list[1]
        lines.append(f"  [1] domain:    {token_to_domain_hint(dom_tok)}")

    # Classify remaining tokens.
    structural_shown: int = 0
    intent_count: int = 0
    pad_count: int = 0

    for pos in range(2, len(tok_list)):
        tok: int = tok_list[pos]
        if tok == PAD_TOKEN:
            pad_count += 1
        elif STRUCTURAL_TOKEN_OFFSET <= tok <= STRUCTURAL_TOKEN_END:
            if structural_shown < max_tokens_to_show:
                name: Optional[str] = token_to_primitive_name(tok)
                lines.append(f"  [{pos}] primitive: {name or '???'} (token={tok})")
            structural_shown += 1
        elif INTENT_TOKEN_OFFSET <= tok <= INTENT_TOKEN_END:
            intent_count += 1
        else:
            lines.append(f"  [{pos}] unknown:   token={tok}")

    if structural_shown > max_tokens_to_show:
        lines.append(
            f"  ... and {structural_shown - max_tokens_to_show} more structural tokens"
        )
    if intent_count > 0:
        lines.append(f"  ({intent_count} intent tokens, {INTENT_TOKENS_PER_VECTOR} expected)")
    if pad_count > 0:
        lines.append(f"  ({pad_count} PAD tokens)")

    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
# SEQUENCE VALIDATION
# ═════════════════════════════════════════════════════════════════════════════

def validate_sequence(tokens: torch.Tensor) -> bool:
    """
    Verify that all token IDs in *tokens* are in the valid vocabulary range.

    A valid token is any integer in [0, VOCAB_SIZE - 1] (i.e. [0, 8191]).
    Token 0 is PAD_TOKEN and is valid as a vocabulary-level integer — it is
    the padding sentinel.  Tokens outside [0, VOCAB_SIZE - 1] indicate a
    tokenizer bug or memory corruption.

    This function checks vocabulary-range validity only.  It does NOT validate:
      - That position 0 is a topology class token (that is encode_domain_event's
        responsibility).
      - That position 1 is a domain hash token.
      - That the attention mask is consistent with the token values.

    Args:
        tokens: A torch.Tensor of any shape with dtype torch.long.

    Returns:
        True if all values are in [0, VOCAB_SIZE - 1], False otherwise.
        Logs a WARNING with the first out-of-range value if validation fails.

    Examples:
        >>> import torch
        >>> validate_sequence(torch.tensor([1, 1024, 77, 0, 0, 0]))
        True
        >>> validate_sequence(torch.tensor([1, 1024, 9999]))
        False
        >>> validate_sequence(torch.tensor([1, 1024, -1]))
        False
    """
    if tokens.numel() == 0:
        return True

    min_val: int = int(tokens.min().item())
    max_val: int = int(tokens.max().item())

    if min_val < 0:
        _log.warning(
            "validate_sequence: found token value %d < 0 (minimum valid is 0 = PAD).  "
            "This indicates a tokenizer bug or upstream data corruption.",
            min_val,
        )
        return False

    if max_val >= VOCAB_SIZE:
        _log.warning(
            "validate_sequence: found token value %d >= VOCAB_SIZE (%d).  "
            "Valid range is [0, %d].  "
            "This indicates a tokenizer bug or a vocabulary size mismatch.",
            max_val, VOCAB_SIZE, VOCAB_SIZE - 1,
        )
        return False

    return True


def validate_sequence_structure(tokens: torch.Tensor) -> Tuple[bool, str]:
    """
    Perform a deeper structural validation of an encoded sequence.

    Checks:
      1. Vocabulary range (all tokens in [0, VOCAB_SIZE - 1]).
      2. Position 0 is a valid topology class token (in [1, 18]).
      3. Position 1 is a valid domain hash token (in [1024, 4095]).
      4. Padding tokens (0) do not appear before real tokens
         (no PAD in the middle of a sequence; PAD is right-only).

    Args:
        tokens: A 1-D torch.Tensor, dtype torch.long.

    Returns:
        A tuple (is_valid, reason):
          is_valid — True if all checks pass.
          reason   — Empty string if valid; description of first failure if not.

    Raises:
        ValueError: if tokens is not 1-D.

    Examples:
        >>> import torch
        >>> t = encode_domain_event("NEWS_ARTICLE", "example.com", ["cloudflare"])
        >>> valid, reason = validate_sequence_structure(t)
        >>> valid
        True
        >>> reason
        ''
    """
    if tokens.ndim != 1:
        raise ValueError(
            f"validate_sequence_structure requires a 1-D tensor, "
            f"got shape {tuple(tokens.shape)}."
        )

    tok_list: List[int] = tokens.tolist()  # type: ignore[attr-defined]
    n: int = len(tok_list)

    # Check 1: vocabulary range.
    if not validate_sequence(tokens):
        return False, "Token value(s) outside [0, VOCAB_SIZE - 1]."

    if n == 0:
        return False, "Sequence is empty."

    # Check 2: position 0 must be a topology class token.
    tok0: int = tok_list[0]
    if not (TOPOLOGY_TOKEN_OFFSET <= tok0 <= TOPOLOGY_TOKEN_END):
        return (
            False,
            f"Position 0 token {tok0} is not in topology range "
            f"[{TOPOLOGY_TOKEN_OFFSET}, {TOPOLOGY_TOKEN_END}].  "
            f"Expected a topology class token at position 0.",
        )

    # Check 3: position 1 must be a domain hash token.
    if n < 2:
        return False, "Sequence has only 1 token (missing domain hash token at position 1)."

    tok1: int = tok_list[1]
    if not (DOMAIN_TOKEN_OFFSET <= tok1 <= DOMAIN_TOKEN_END):
        return (
            False,
            f"Position 1 token {tok1} is not in domain range "
            f"[{DOMAIN_TOKEN_OFFSET}, {DOMAIN_TOKEN_END}].  "
            f"Expected a domain hash token at position 1.",
        )

    # Check 4: no PAD tokens before the trailing PAD block.
    # Find the first PAD position.  All positions after it must also be PAD.
    first_pad: int = -1
    for pos, tok in enumerate(tok_list):
        if tok == PAD_TOKEN:
            first_pad = pos
            break

    if first_pad != -1:
        # Every token from first_pad to the end must be PAD.
        non_pad_after_first: List[int] = [
            p for p, t in enumerate(tok_list[first_pad:], start=first_pad)
            if t != PAD_TOKEN
        ]
        if non_pad_after_first:
            return (
                False,
                f"PAD token at position {first_pad} is followed by non-PAD tokens "
                f"at positions {non_pad_after_first[:5]}.  "
                f"PAD tokens must form a contiguous right-side block.",
            )

    return True, ""


def sequence_stats(tokens: torch.Tensor) -> Dict[str, int]:
    """
    Count the number of tokens in each vocabulary range within *tokens*.

    Returns a dict with keys:
      "total_length"    — total number of token positions (including PAD)
      "pad_tokens"      — count of PAD_TOKEN (0) positions
      "real_tokens"     — count of non-PAD positions
      "topology_tokens" — count in [1, 18]
      "domain_tokens"   — count in [1024, 4095]
      "structural_tokens" — count in [19, 1023]
      "intent_tokens"   — count in [4096, 8191]
      "unknown_range"   — count of tokens outside any defined range (should be 0)

    This function is used for observability, testing, and audit logging.  It
    does NOT raise on unknown-range tokens — it counts them in "unknown_range"
    and the caller can decide whether to act.

    Args:
        tokens: A torch.Tensor of any shape with dtype torch.long.
                The tensor is flattened before analysis.

    Returns:
        Dict[str, int] as described above.

    Examples:
        >>> import torch
        >>> t = encode_domain_event("SAAS_DOCS", "stripe.com", ["cloudflare", "tls_1_3"])
        >>> stats = sequence_stats(t)
        >>> stats["topology_tokens"]
        1
        >>> stats["domain_tokens"]
        1
        >>> stats["structural_tokens"]
        2
        >>> stats["pad_tokens"] > 0
        True
    """
    flat: List[int] = tokens.flatten().tolist()  # type: ignore[attr-defined]
    total: int = len(flat)

    counts: Dict[str, int] = {
        "total_length":     total,
        "pad_tokens":       0,
        "real_tokens":      0,
        "topology_tokens":  0,
        "domain_tokens":    0,
        "structural_tokens": 0,
        "intent_tokens":    0,
        "unknown_range":    0,
    }

    for tok in flat:
        if tok == PAD_TOKEN:
            counts["pad_tokens"] += 1
        elif TOPOLOGY_TOKEN_OFFSET <= tok <= TOPOLOGY_TOKEN_END:
            counts["topology_tokens"] += 1
            counts["real_tokens"] += 1
        elif STRUCTURAL_TOKEN_OFFSET <= tok <= STRUCTURAL_TOKEN_END:
            counts["structural_tokens"] += 1
            counts["real_tokens"] += 1
        elif DOMAIN_TOKEN_OFFSET <= tok <= DOMAIN_TOKEN_END:
            counts["domain_tokens"] += 1
            counts["real_tokens"] += 1
        elif INTENT_TOKEN_OFFSET <= tok <= INTENT_TOKEN_END:
            counts["intent_tokens"] += 1
            counts["real_tokens"] += 1
        else:
            counts["unknown_range"] += 1
            # Unknown-range tokens are NOT counted as real_tokens because they
            # would confuse consumers that compute real-token fractions.

    return counts


# ═════════════════════════════════════════════════════════════════════════════
# RESPONSE TIME SIGNAL HELPER
#
# domain_analyzer.py emits response time as a float (milliseconds).  This
# helper converts a raw response time to the appropriate bucket signal name,
# which encode_domain_event() can then pass in structural_signals.
# It is not technically part of the tokenizer's responsibility, but it is
# tightly coupled to the response time primitive definitions and lives here
# to keep the bucket boundaries in one place.
# ═════════════════════════════════════════════════════════════════════════════

def response_time_to_signal(latency_ms: float) -> str:
    """
    Map a response latency in milliseconds to its structural signal name.

    Args:
        latency_ms: Response latency in milliseconds.  Must be ≥ 0.

    Returns:
        One of: "lt_100ms", "100_500ms", "500ms_2s", "2s_5s", "gt_5s".

    Raises:
        ValueError: if latency_ms < 0.

    Examples:
        >>> response_time_to_signal(50.0)
        'lt_100ms'
        >>> response_time_to_signal(300.0)
        '100_500ms'
        >>> response_time_to_signal(1200.0)
        '500ms_2s'
        >>> response_time_to_signal(3500.0)
        '2s_5s'
        >>> response_time_to_signal(10000.0)
        'gt_5s'
    """
    if latency_ms < 0:
        raise ValueError(
            f"latency_ms must be ≥ 0, got {latency_ms}.  "
            "Negative latency is a measurement error."
        )
    if latency_ms < 100.0:
        return "lt_100ms"
    elif latency_ms < 500.0:
        return "100_500ms"
    elif latency_ms < 2_000.0:
        return "500ms_2s"
    elif latency_ms < 5_000.0:
        return "2s_5s"
    else:
        return "gt_5s"


def http_status_to_signal(status_code: int) -> Optional[str]:
    """
    Map an HTTP status code to its structural signal name.

    Only the six status codes with named tokens are mapped.  Any other status
    code returns None (caller may handle by not adding a status signal).

    Args:
        status_code: HTTP status code integer.

    Returns:
        One of: "status_200", "status_301", "status_302", "status_403",
        "status_429", "status_503", or None for unmapped codes.

    Examples:
        >>> http_status_to_signal(200)
        'status_200'
        >>> http_status_to_signal(404) is None
        True
        >>> http_status_to_signal(429)
        'status_429'
    """
    _STATUS_MAP: Dict[int, str] = {
        200: "status_200",
        301: "status_301",
        302: "status_302",
        403: "status_403",
        429: "status_429",
        503: "status_503",
    }
    return _STATUS_MAP.get(status_code)


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL VOCABULARY INTEGRITY VERIFICATION
#
# These assertions run at import time (when verify_vocabulary_integrity() is
# called from the module footer, or when running this file directly).  If any
# assertion fails, the tokenizer has an inconsistency in its constant definitions
# that would produce incorrect token sequences fed to the SSM.
#
# verify_vocabulary_integrity() is also the recommended first call in any test
# suite that imports this module.
# ═════════════════════════════════════════════════════════════════════════════

def verify_vocabulary_integrity() -> None:
    """
    Run all structural consistency checks on the vocabulary constants.

    Raises:
        AssertionError: with a descriptive message if any invariant is violated.

    Invariants checked:
      1. VOCAB_SIZE is 8192.
      2. PAD_TOKEN (0) is outside all active vocabulary ranges.
      3. Topology token range is [1, 18] and covers exactly len(TOPOLOGY_CLASSES) tokens.
      4. TOPOLOGY_CLASSES has exactly 18 entries (per the WLM spec).
      5. Every topology class in contracts.TOPOLOGY_CLASSES has a unique token.
      6. Structural token range is [19, 1023] (1005 slots).
      7. Named structural primitive constants (PRIMITIVE_*) are within [19, 83].
      8. PRIMITIVE_UNKNOWN is within the structural range.
      9. FIRST_RESERVED_PRIMITIVE is 84, last is 1023.
      10. Domain token range is [1024, 4095] with 3072 buckets.
      11. Intent token range is [4096, 8191].
      12. _INTENT_CHUNK_BASE equals 4^INTENT_CHUNK_SIZE = 4096.
      13. INTENT_TOKEN_OFFSET + _INTENT_CHUNK_BASE - 1 == INTENT_TOKEN_END.
      14. Every key in _SIGNAL_NAME_TO_TOKEN maps to a value in [STRUCTURAL_TOKEN_OFFSET, STRUCTURAL_TOKEN_END].
      15. Every value in SIGNAL_PRIORITY_RANK is a structural token in [STRUCTURAL_TOKEN_OFFSET, STRUCTURAL_TOKEN_END].
      16. _TOKEN_TO_SIGNAL_NAME reverse dict is consistent with _SIGNAL_NAME_TO_TOKEN.
      17. Range boundaries are contiguous (no gaps between topology→structural→domain→intent).
      18. INTENT_TOKENS_PER_VECTOR == ceil(INTENT_VECTOR_DIM / INTENT_CHUNK_SIZE) == 43.
    """
    # ── Check 1: VOCAB_SIZE ───────────────────────────────────────────────
    assert VOCAB_SIZE == 8_192, (
        f"VOCAB_SIZE must be 8192, got {VOCAB_SIZE}."
    )

    # ── Check 2: PAD_TOKEN is not in any active vocabulary range ──────────
    assert PAD_TOKEN == 0, (
        f"PAD_TOKEN must be 0, got {PAD_TOKEN}."
    )
    assert not (TOPOLOGY_TOKEN_OFFSET <= PAD_TOKEN <= TOPOLOGY_TOKEN_END), (
        f"PAD_TOKEN ({PAD_TOKEN}) collides with topology range "
        f"[{TOPOLOGY_TOKEN_OFFSET}, {TOPOLOGY_TOKEN_END}].  "
        f"This is the collision the offset=1 decision was made to prevent."
    )
    assert not (STRUCTURAL_TOKEN_OFFSET <= PAD_TOKEN <= STRUCTURAL_TOKEN_END), (
        f"PAD_TOKEN ({PAD_TOKEN}) collides with structural range."
    )
    assert not (DOMAIN_TOKEN_OFFSET <= PAD_TOKEN <= DOMAIN_TOKEN_END), (
        f"PAD_TOKEN ({PAD_TOKEN}) collides with domain range."
    )
    assert not (INTENT_TOKEN_OFFSET <= PAD_TOKEN <= INTENT_TOKEN_END), (
        f"PAD_TOKEN ({PAD_TOKEN}) collides with intent range."
    )

    # ── Check 3: Topology token range ────────────────────────────────────
    assert TOPOLOGY_TOKEN_OFFSET == 1, (
        f"TOPOLOGY_TOKEN_OFFSET must be 1 (for PAD collision avoidance), "
        f"got {TOPOLOGY_TOKEN_OFFSET}."
    )
    assert TOPOLOGY_TOKEN_END == TOPOLOGY_TOKEN_OFFSET + len(TOPOLOGY_CLASSES) - 1, (
        f"TOPOLOGY_TOKEN_END ({TOPOLOGY_TOKEN_END}) is inconsistent with "
        f"TOPOLOGY_TOKEN_OFFSET ({TOPOLOGY_TOKEN_OFFSET}) and "
        f"len(TOPOLOGY_CLASSES) ({len(TOPOLOGY_CLASSES)})."
    )

    # ── Check 4: TOPOLOGY_CLASSES count ───────────────────────────────────
    assert len(TOPOLOGY_CLASSES) == 18, (
        f"Expected exactly 18 topology classes (per WLM spec), "
        f"got {len(TOPOLOGY_CLASSES)}.  "
        f"Update the spec or add the new classes to contracts.TOPOLOGY_CLASSES."
    )
    assert TOPOLOGY_TOKEN_END == 18, (
        f"Last topology token must be 18, got {TOPOLOGY_TOKEN_END}."
    )

    # ── Check 5: Unique topology tokens ───────────────────────────────────
    token_values: List[int] = list(_TOPOLOGY_CLASS_TO_TOKEN.values())
    assert len(set(token_values)) == len(token_values), (
        f"Duplicate topology token values detected in _TOPOLOGY_CLASS_TO_TOKEN: "
        f"{[t for t in token_values if token_values.count(t) > 1]}"
    )
    assert len(_TOPOLOGY_CLASS_TO_TOKEN) == len(TOPOLOGY_CLASSES), (
        f"_TOPOLOGY_CLASS_TO_TOKEN has {len(_TOPOLOGY_CLASS_TO_TOKEN)} entries, "
        f"expected {len(TOPOLOGY_CLASSES)}."
    )

    # ── Check 6: Structural token range ───────────────────────────────────
    assert STRUCTURAL_TOKEN_OFFSET == 19, (
        f"STRUCTURAL_TOKEN_OFFSET must be 19 (= TOPOLOGY_TOKEN_END + 1 = 18 + 1), "
        f"got {STRUCTURAL_TOKEN_OFFSET}."
    )
    assert STRUCTURAL_TOKEN_END == DOMAIN_TOKEN_OFFSET - 1, (
        f"STRUCTURAL_TOKEN_END ({STRUCTURAL_TOKEN_END}) must be one less than "
        f"DOMAIN_TOKEN_OFFSET ({DOMAIN_TOKEN_OFFSET})."
    )
    assert _STRUCTURAL_SLOTS == 1_005, (
        f"Structural slot count must be 1005 (19–1023), got {_STRUCTURAL_SLOTS}."
    )

    # ── Check 7: Named structural primitive constants are in range ─────────
    named_primitive_tokens: List[int] = [
        PRIMITIVE_CLOUDFLARE, PRIMITIVE_FASTLY, PRIMITIVE_AKAMAI,
        PRIMITIVE_CLOUDFRONT, PRIMITIVE_VERCEL, PRIMITIVE_NETLIFY,
        PRIMITIVE_WORDPRESS, PRIMITIVE_GHOST, PRIMITIVE_DRUPAL,
        PRIMITIVE_CONFLUENCE, PRIMITIVE_NOTION, PRIMITIVE_GITBOOK,
        PRIMITIVE_DOCUSAURUS, PRIMITIVE_MKDOCS,
        PRIMITIVE_REQUIRES_JS, PRIMITIVE_STATIC_ONLY, PRIMITIVE_SPA_DETECTED,
        PRIMITIVE_SSR_DETECTED, PRIMITIVE_HYDRATION_DETECTED,
        PRIMITIVE_CLOUDFLARE_CHALLENGE, PRIMITIVE_RECAPTCHA, PRIMITIVE_HCAPTCHA,
        PRIMITIVE_DATADOME, PRIMITIVE_PERIMETER_X, PRIMITIVE_RATE_LIMIT_HEADER,
        PRIMITIVE_TLS_1_2, PRIMITIVE_TLS_1_3, PRIMITIVE_CERT_WILDCARD,
        PRIMITIVE_CERT_ORG, PRIMITIVE_CERT_LETS_ENCRYPT,
        PRIMITIVE_HTTP_1_1, PRIMITIVE_HTTP_2, PRIMITIVE_HTTP_3,
        PRIMITIVE_RESPONSE_LT_100MS, PRIMITIVE_RESPONSE_100_500MS,
        PRIMITIVE_RESPONSE_500MS_2S, PRIMITIVE_RESPONSE_2S_5S,
        PRIMITIVE_RESPONSE_GT_5S,
        PRIMITIVE_ROBOTS_PRESENT, PRIMITIVE_ROBOTS_ABSENT,
        PRIMITIVE_CRAWL_DELAY_SET, PRIMITIVE_DISALLOW_HEAVY,
        PRIMITIVE_SITEMAP_LINKED,
        PRIMITIVE_SITEMAP_PRESENT, PRIMITIVE_SITEMAP_INDEX,
        PRIMITIVE_SITEMAP_URLSET, PRIMITIVE_SITEMAP_NEWS, PRIMITIVE_SITEMAP_IMAGE,
        PRIMITIVE_TEXT_HTML, PRIMITIVE_APPLICATION_JSON,
        PRIMITIVE_APPLICATION_LD_JSON, PRIMITIVE_TEXT_PLAIN,
        PRIMITIVE_APPLICATION_XML,
        PRIMITIVE_X_POWERED_BY_PRESENT, PRIMITIVE_SERVER_NGINX,
        PRIMITIVE_SERVER_APACHE, PRIMITIVE_SERVER_CADDY, PRIMITIVE_VIA_PRESENT,
        PRIMITIVE_STATUS_200, PRIMITIVE_STATUS_301, PRIMITIVE_STATUS_302,
        PRIMITIVE_STATUS_403, PRIMITIVE_STATUS_429, PRIMITIVE_STATUS_503,
    ]
    for pt in named_primitive_tokens:
        assert STRUCTURAL_TOKEN_OFFSET <= pt < FIRST_RESERVED_PRIMITIVE, (
            f"Named primitive token {pt} is outside the named range "
            f"[{STRUCTURAL_TOKEN_OFFSET}, {FIRST_RESERVED_PRIMITIVE - 1}]."
        )
    # Named primitives should be unique.
    assert len(set(named_primitive_tokens)) == len(named_primitive_tokens), (
        f"Duplicate named primitive token values detected: "
        f"{[t for t in named_primitive_tokens if named_primitive_tokens.count(t) > 1]}"
    )
    # Expect exactly 64 named primitives.
    assert len(named_primitive_tokens) == 64, (
        f"Expected 64 named primitive tokens (PRIMITIVE_CLOUDFLARE through "
        f"PRIMITIVE_STATUS_503), got {len(named_primitive_tokens)}."
    )

    # ── Check 8: PRIMITIVE_UNKNOWN is in structural range ─────────────────
    assert STRUCTURAL_TOKEN_OFFSET <= PRIMITIVE_UNKNOWN <= STRUCTURAL_TOKEN_END, (
        f"PRIMITIVE_UNKNOWN ({PRIMITIVE_UNKNOWN}) is outside structural range."
    )
    assert PRIMITIVE_UNKNOWN == FIRST_RESERVED_PRIMITIVE - 1, (
        f"PRIMITIVE_UNKNOWN ({PRIMITIVE_UNKNOWN}) should be immediately before "
        f"FIRST_RESERVED_PRIMITIVE ({FIRST_RESERVED_PRIMITIVE})."
    )

    # ── Check 9: Reserved range ────────────────────────────────────────────
    assert FIRST_RESERVED_PRIMITIVE == 84, (
        f"FIRST_RESERVED_PRIMITIVE must be 84, got {FIRST_RESERVED_PRIMITIVE}."
    )
    assert LAST_RESERVED_PRIMITIVE == 1_023, (
        f"LAST_RESERVED_PRIMITIVE must be 1023, got {LAST_RESERVED_PRIMITIVE}."
    )
    assert _RESERVED_PRIMITIVE_COUNT == 940, (
        f"Reserved primitive count must be 940, got {_RESERVED_PRIMITIVE_COUNT}."
    )

    # ── Check 10: Domain token range ──────────────────────────────────────
    assert DOMAIN_TOKEN_OFFSET == 1_024, (
        f"DOMAIN_TOKEN_OFFSET must be 1024, got {DOMAIN_TOKEN_OFFSET}."
    )
    assert DOMAIN_TOKEN_END == 4_095, (
        f"DOMAIN_TOKEN_END must be 4095, got {DOMAIN_TOKEN_END}."
    )
    assert _DOMAIN_HASH_BUCKETS == 3_072, (
        f"_DOMAIN_HASH_BUCKETS must be 3072, got {_DOMAIN_HASH_BUCKETS}."
    )
    assert DOMAIN_TOKEN_END - DOMAIN_TOKEN_OFFSET + 1 == _DOMAIN_HASH_BUCKETS, (
        f"Domain token range width ({DOMAIN_TOKEN_END - DOMAIN_TOKEN_OFFSET + 1}) "
        f"does not equal _DOMAIN_HASH_BUCKETS ({_DOMAIN_HASH_BUCKETS})."
    )

    # ── Check 11: Intent token range ──────────────────────────────────────
    assert INTENT_TOKEN_OFFSET == 4_096, (
        f"INTENT_TOKEN_OFFSET must be 4096, got {INTENT_TOKEN_OFFSET}."
    )
    assert INTENT_TOKEN_END == 8_191, (
        f"INTENT_TOKEN_END must be 8191, got {INTENT_TOKEN_END}."
    )
    assert INTENT_TOKEN_END - INTENT_TOKEN_OFFSET + 1 == 4_096, (
        f"Intent token range must span exactly 4096 slots."
    )

    # ── Check 12: _INTENT_CHUNK_BASE ──────────────────────────────────────
    assert _INTENT_CHUNK_BASE == 4 ** INTENT_CHUNK_SIZE, (
        f"_INTENT_CHUNK_BASE ({_INTENT_CHUNK_BASE}) must equal "
        f"4^INTENT_CHUNK_SIZE ({4 ** INTENT_CHUNK_SIZE})."
    )
    assert _INTENT_CHUNK_BASE == 4_096, (
        f"_INTENT_CHUNK_BASE must be 4096 (= 4^6), got {_INTENT_CHUNK_BASE}."
    )

    # ── Check 13: Intent token arithmetic ────────────────────────────────
    assert INTENT_TOKEN_OFFSET + _INTENT_CHUNK_BASE - 1 == INTENT_TOKEN_END, (
        f"INTENT_TOKEN_OFFSET ({INTENT_TOKEN_OFFSET}) + _INTENT_CHUNK_BASE "
        f"({_INTENT_CHUNK_BASE}) - 1 = "
        f"{INTENT_TOKEN_OFFSET + _INTENT_CHUNK_BASE - 1} ≠ "
        f"INTENT_TOKEN_END ({INTENT_TOKEN_END}).  "
        f"The intent token range exactly accommodates one packed 6-dim chunk."
    )

    # ── Check 14: _SIGNAL_NAME_TO_TOKEN values are in structural range ────
    for sig_name, sig_tok in _SIGNAL_NAME_TO_TOKEN.items():
        assert STRUCTURAL_TOKEN_OFFSET <= sig_tok <= STRUCTURAL_TOKEN_END, (
            f"_SIGNAL_NAME_TO_TOKEN[{sig_name!r}] = {sig_tok} is outside "
            f"structural range [{STRUCTURAL_TOKEN_OFFSET}, {STRUCTURAL_TOKEN_END}]."
        )

    # ── Check 15: SIGNAL_PRIORITY_RANK values are in structural range ─────
    for rank_tok, rank_val in SIGNAL_PRIORITY_RANK.items():
        assert STRUCTURAL_TOKEN_OFFSET <= rank_tok <= STRUCTURAL_TOKEN_END, (
            f"SIGNAL_PRIORITY_RANK key {rank_tok} is outside structural range."
        )
        assert isinstance(rank_val, int) and rank_val >= 0, (
            f"SIGNAL_PRIORITY_RANK[{rank_tok}] = {rank_val} must be a non-negative int."
        )

    # ── Check 16: Reverse dict consistency ────────────────────────────────
    for tok, name in _TOKEN_TO_SIGNAL_NAME.items():
        assert STRUCTURAL_TOKEN_OFFSET <= tok <= STRUCTURAL_TOKEN_END, (
            f"_TOKEN_TO_SIGNAL_NAME key {tok} is outside structural range."
        )
        assert isinstance(name, str) and name, (
            f"_TOKEN_TO_SIGNAL_NAME[{tok}] must be a non-empty string."
        )

    # ── Check 17: Range contiguity ────────────────────────────────────────
    # PAD (0) → topology [1,18] → structural [19,1023] → domain [1024,4095]
    # → intent [4096, 8191]
    assert TOPOLOGY_TOKEN_OFFSET == PAD_TOKEN + 1, (
        f"Topology range must start immediately after PAD_TOKEN."
    )
    assert STRUCTURAL_TOKEN_OFFSET == TOPOLOGY_TOKEN_END + 1, (
        f"Structural range must start immediately after topology range."
    )
    assert DOMAIN_TOKEN_OFFSET == STRUCTURAL_TOKEN_END + 1, (
        f"Domain range must start immediately after structural range."
    )
    assert INTENT_TOKEN_OFFSET == DOMAIN_TOKEN_END + 1, (
        f"Intent range must start immediately after domain range."
    )
    assert INTENT_TOKEN_END == VOCAB_SIZE - 1, (
        f"Intent range must end at VOCAB_SIZE - 1 ({VOCAB_SIZE - 1}), "
        f"got INTENT_TOKEN_END = {INTENT_TOKEN_END}."
    )

    # ── Check 18: INTENT_TOKENS_PER_VECTOR ────────────────────────────────
    assert INTENT_TOKENS_PER_VECTOR == math.ceil(INTENT_VECTOR_DIM / INTENT_CHUNK_SIZE), (
        f"INTENT_TOKENS_PER_VECTOR ({INTENT_TOKENS_PER_VECTOR}) must equal "
        f"ceil({INTENT_VECTOR_DIM}/{INTENT_CHUNK_SIZE}) = "
        f"{math.ceil(INTENT_VECTOR_DIM / INTENT_CHUNK_SIZE)}."
    )
    assert INTENT_TOKENS_PER_VECTOR == 43, (
        f"INTENT_TOKENS_PER_VECTOR must be 43 (ceil(256/6)), "
        f"got {INTENT_TOKENS_PER_VECTOR}."
    )
    # Verify chunk coverage: 42 full chunks × 6 + 1 partial chunk × 4 = 256.
    full_chunks: int = INTENT_VECTOR_DIM // INTENT_CHUNK_SIZE     # 42
    partial_dims: int = INTENT_VECTOR_DIM % INTENT_CHUNK_SIZE     # 4
    assert full_chunks * INTENT_CHUNK_SIZE + partial_dims == INTENT_VECTOR_DIM, (
        f"Chunk coverage arithmetic error."
    )
    assert partial_dims == 4, (
        f"Last chunk must have 4 real dims (padded with 2 zeros), got {partial_dims}."
    )


# ═════════════════════════════════════════════════════════════════════════════
# SELF-CONTAINED UNIT TESTS
#
# These tests run without pytest.  Call run_tests() to execute them.
# They do not import any external test framework and make no file or network
# calls.  They are primarily for smoke-testing after any change to constants,
# thresholds, or hashing logic.
# ═════════════════════════════════════════════════════════════════════════════

class _TokenizerTestResult(NamedTuple):
    test_name: str
    passed:    bool
    message:   str


def _run_test(name: str, fn: "Callable[[], None]") -> _TokenizerTestResult:  # noqa: F821
    """Execute one test function and return a result."""
    try:
        fn()
        return _TokenizerTestResult(name, True, "OK")
    except (AssertionError, ValueError, TypeError, RuntimeError) as exc:
        return _TokenizerTestResult(name, False, str(exc))


def _test_vocabulary_integrity() -> None:
    verify_vocabulary_integrity()


def _test_pad_token_not_in_vocab_ranges() -> None:
    assert PAD_TOKEN == 0
    assert PAD_TOKEN not in _TOPOLOGY_CLASS_TO_TOKEN.values()
    assert PAD_TOKEN not in _SIGNAL_NAME_TO_TOKEN.values()
    # PAD_TOKEN is below DOMAIN_TOKEN_OFFSET and INTENT_TOKEN_OFFSET by construction.
    assert PAD_TOKEN < TOPOLOGY_TOKEN_OFFSET


def _test_topology_token_derivation() -> None:
    # Token for NEWS_ARTICLE must be 1 (first in TOPOLOGY_CLASSES + offset 1).
    assert TOPOLOGY_CLASSES[0] == "NEWS_ARTICLE"
    assert topology_class_to_token("NEWS_ARTICLE") == 1
    # GENERIC_HTML is last.
    assert TOPOLOGY_CLASSES[-1] == "GENERIC_HTML"
    assert topology_class_to_token("GENERIC_HTML") == 18
    # All 18 classes produce distinct tokens in [1, 18].
    tokens = [topology_class_to_token(cls) for cls in TOPOLOGY_CLASSES]
    assert len(set(tokens)) == 18
    assert min(tokens) == 1
    assert max(tokens) == 18


def _test_topology_unknown_fallback() -> None:
    # Unknown topology class falls back to GENERIC_HTML without raising.
    fallback_tok = topology_class_to_token("DEFINITELY_NOT_A_REAL_CLASS")
    assert fallback_tok == TOPOLOGY_TOKEN_GENERIC_HTML


def _test_normalize_domain_basic() -> None:
    cases = [
        ("docs.stripe.com",     "stripe.com"),
        ("api.github.com",      "github.com"),
        ("v2.docs.aws.com",     "aws.com"),
        ("stripe.com",          "stripe.com"),
        ("github.com",          "github.com"),
        ("DOCS.STRIPE.COM",     "stripe.com"),  # case normalisation
        ("",                    ""),
        ("localhost",           "localhost"),
    ]
    for raw, expected in cases:
        result = normalize_domain(raw)
        assert result == expected, (
            f"normalize_domain({raw!r}) = {result!r}, expected {expected!r}"
        )


def _test_normalize_domain_multi_part_tld() -> None:
    cases = [
        ("api.bbc.co.uk",       "bbc.co.uk"),
        ("news.bbc.co.uk",      "bbc.co.uk"),
        ("bbc.co.uk",           "bbc.co.uk"),
        ("www.example.com.au",  "example.com.au"),
        ("example.com.au",      "example.com.au"),
    ]
    for raw, expected in cases:
        result = normalize_domain(raw)
        assert result == expected, (
            f"normalize_domain({raw!r}) = {result!r}, expected {expected!r}"
        )


def _test_normalize_domain_port_stripping() -> None:
    assert normalize_domain("api.stripe.com:443") == "stripe.com"
    assert normalize_domain("example.com:8080") == "example.com"


def _test_domain_to_token_consistency() -> None:
    # Same subdomain of the same domain → same token.
    t1 = domain_to_token("docs.stripe.com")
    t2 = domain_to_token("api.stripe.com")
    t3 = domain_to_token("stripe.com")
    assert t1 == t2 == t3, (
        f"Subdomain normalisation failed: "
        f"docs.stripe.com→{t1}, api.stripe.com→{t2}, stripe.com→{t3}"
    )
    # Token is in domain range.
    assert DOMAIN_TOKEN_OFFSET <= t1 <= DOMAIN_TOKEN_END


def _test_domain_to_token_range() -> None:
    sample_domains = [
        "github.com", "google.com", "stripe.com", "nytimes.com",
        "wikipedia.org", "stackoverflow.com", "reddit.com", "amazon.com",
    ]
    for d in sample_domains:
        tok = domain_to_token(d)
        assert DOMAIN_TOKEN_OFFSET <= tok <= DOMAIN_TOKEN_END, (
            f"domain_to_token({d!r}) = {tok} outside [1024, 4095]"
        )


def _test_domain_to_token_determinism() -> None:
    # Call domain_to_token twice for the same domain and verify the result.
    for domain in ("stripe.com", "docs.stripe.com", "github.com"):
        assert domain_to_token(domain) == domain_to_token(domain), (
            f"Non-deterministic domain_to_token for {domain!r}"
        )


def _test_structural_signal_known() -> None:
    assert structural_signal_to_token("cloudflare") == PRIMITIVE_CLOUDFLARE
    assert structural_signal_to_token("tls_1_3") == PRIMITIVE_TLS_1_3
    assert structural_signal_to_token("status_200") == PRIMITIVE_STATUS_200
    assert structural_signal_to_token("wordpress") == PRIMITIVE_WORDPRESS
    assert structural_signal_to_token("http_2") == PRIMITIVE_HTTP_2


def _test_structural_signal_aliases() -> None:
    # Aliases should map to the same token.
    assert structural_signal_to_token("cdn_cloudflare") == PRIMITIVE_CLOUDFLARE
    assert structural_signal_to_token("cloudflare") == PRIMITIVE_CLOUDFLARE
    assert structural_signal_to_token("lets_encrypt") == PRIMITIVE_CERT_LETS_ENCRYPT
    assert structural_signal_to_token("cert_lets_encrypt") == PRIMITIVE_CERT_LETS_ENCRYPT
    assert structural_signal_to_token("http_429") == PRIMITIVE_STATUS_429
    assert structural_signal_to_token("status_429") == PRIMITIVE_STATUS_429


def _test_structural_signal_case_insensitive() -> None:
    assert structural_signal_to_token("CLOUDFLARE") == PRIMITIVE_CLOUDFLARE
    assert structural_signal_to_token("Cloudflare") == PRIMITIVE_CLOUDFLARE
    assert structural_signal_to_token("TLS_1_3") == PRIMITIVE_TLS_1_3


def _test_structural_signal_unknown() -> None:
    tok = structural_signal_to_token("this_signal_does_not_exist_xyz")
    assert tok == PRIMITIVE_UNKNOWN, (
        f"Unknown signal should map to PRIMITIVE_UNKNOWN ({PRIMITIVE_UNKNOWN}), got {tok}"
    )


def _test_quantize_intent_dimension() -> None:
    # Boundary cases.
    assert quantize_intent_dimension(-1.0) == 0
    assert quantize_intent_dimension(-0.5) == 1   # boundary belongs to level 1
    assert quantize_intent_dimension(-0.4) == 1
    assert quantize_intent_dimension(-0.0) == 2   # 0.0 belongs to level 2
    assert quantize_intent_dimension(0.0)  == 2
    assert quantize_intent_dimension(0.5)  == 2   # upper boundary belongs to level 2
    assert quantize_intent_dimension(0.51) == 3
    assert quantize_intent_dimension(1.0)  == 3
    # Non-finite values.
    assert quantize_intent_dimension(float("inf"))  == 3
    assert quantize_intent_dimension(float("-inf")) == 0
    # NaN: returns 1 (documented fallback).
    assert quantize_intent_dimension(float("nan")) == 1


def _test_intent_vector_to_tokens_shape() -> None:
    v = [0.0] * INTENT_VECTOR_DIM
    tokens = intent_vector_to_tokens(v)
    assert len(tokens) == INTENT_TOKENS_PER_VECTOR  # 43
    assert all(INTENT_TOKEN_OFFSET <= t <= INTENT_TOKEN_END for t in tokens), (
        f"All intent tokens must be in [{INTENT_TOKEN_OFFSET}, {INTENT_TOKEN_END}]"
    )


def _test_intent_vector_to_tokens_all_zeros() -> None:
    v = [0.0] * INTENT_VECTOR_DIM
    tokens = intent_vector_to_tokens(v)
    # All-zero dims → all dims quantize to level 2.
    # One chunk of 6 dims, each level 2: 2*4^5 + 2*4^4 + 2*4^3 + 2*4^2 + 2*4 + 2
    # = 2*(1024 + 256 + 64 + 16 + 4 + 1) = 2*1365 = 2730.
    expected_chunk = 4096 + 2730
    assert all(t == expected_chunk for t in tokens[:42]), (
        "All full chunks from all-zero vector should equal 4096 + 2730"
    )
    # Last chunk: 4 real dims (level 2) + 2 padded zeros (level 0).
    # 2*4^5 + 2*4^4 + 2*4^3 + 2*4^2 + 0*4 + 0 = 2*(1024+256+64+16) = 2*1360 = 2720
    expected_last = 4096 + 2720
    assert tokens[42] == expected_last, (
        f"Last chunk of all-zero vector should be 4096+2720={expected_last}, "
        f"got {tokens[42]}"
    )


def _test_intent_vector_to_tokens_all_ones_max() -> None:
    # All dims above 0.5 → all level 3.
    v = [1.0] * INTENT_VECTOR_DIM
    tokens = intent_vector_to_tokens(v)
    # 6 dims each level 3: 3*(4^5+4^4+4^3+4^2+4+1) = 3*1365 = 4095.
    expected_full = 4096 + 4095  # = 8191 = INTENT_TOKEN_END
    assert all(t == expected_full for t in tokens[:42])
    # Last chunk: 4 dims level 3 + 2 pad zeros.
    # 3*4^5 + 3*4^4 + 3*4^3 + 3*4^2 + 0 + 0 = 3*(1024+256+64+16) = 3*1360 = 4080
    expected_last = 4096 + 4080
    assert tokens[42] == expected_last


def _test_intent_vector_wrong_dim() -> None:
    try:
        intent_vector_to_tokens([0.0] * 255)
        assert False, "Should have raised ValueError for wrong dimensionality"
    except ValueError:
        pass


def _test_intent_vector_determinism() -> None:
    import random
    rng = random.Random(42)
    v = [rng.gauss(0, 1) for _ in range(INTENT_VECTOR_DIM)]
    t1 = intent_vector_to_tokens(v)
    t2 = intent_vector_to_tokens(v)
    assert t1 == t2, "intent_vector_to_tokens must be deterministic"


def _test_pad_sequence_short() -> None:
    padded, mask = pad_sequence([1, 1024, 77], max_length=8)
    assert padded == [1, 1024, 77, 0, 0, 0, 0, 0]
    assert mask   == [1, 1,    1,  0, 0, 0, 0, 0]
    assert len(padded) == 8
    assert len(mask) == 8


def _test_pad_sequence_exact() -> None:
    padded, mask = pad_sequence([1, 1024, 77], max_length=3)
    assert padded == [1, 1024, 77]
    assert mask   == [1, 1,    1 ]


def _test_pad_sequence_truncate() -> None:
    padded, mask = pad_sequence([1, 1024, 77, 78, 79], max_length=3)
    assert padded == [1, 1024, 77]
    assert mask   == [1, 1,    1 ]


def _test_make_attention_mask() -> None:
    mask = make_attention_mask([1, 1024, 77, 0, 0, 0])
    assert mask == [1, 1, 1, 0, 0, 0]
    mask2 = make_attention_mask([0, 0])
    assert mask2 == [0, 0]
    mask3 = make_attention_mask([1, 2])
    assert mask3 == [1, 1]


def _test_encode_domain_event_shape() -> None:
    t = encode_domain_event("NEWS_ARTICLE", "nytimes.com", ["cloudflare", "tls_1_3"])
    assert t.shape == torch.Size([MAX_SEQ_LEN])
    assert t.dtype == torch.long


def _test_encode_domain_event_anchor_positions() -> None:
    t = encode_domain_event("SAAS_DOCS", "docs.stripe.com", ["cloudflare"])
    # Position 0: topology token for SAAS_DOCS.
    assert t[0].item() == topology_class_to_token("SAAS_DOCS")
    # Position 1: domain token for stripe.com.
    assert t[1].item() == domain_to_token("stripe.com")
    # Position 1 from "api.stripe.com" should equal position 1 from "docs.stripe.com".
    t2 = encode_domain_event("SAAS_DOCS", "api.stripe.com", ["cloudflare"])
    assert t[1].item() == t2[1].item()


def _test_encode_domain_event_with_intent() -> None:
    intent = [0.1] * INTENT_VECTOR_DIM
    t = encode_domain_event(
        "REST_API_JSON", "api.stripe.com",
        ["cloudflare", "tls_1_3", "http_2"],
        intent_vector=intent,
    )
    stats = sequence_stats(t)
    assert stats["topology_tokens"] == 1
    assert stats["domain_tokens"] == 1
    assert stats["structural_tokens"] == 3
    assert stats["intent_tokens"] == INTENT_TOKENS_PER_VECTOR


def _test_encode_domain_event_deduplication() -> None:
    # Duplicate signals should produce only one structural token each.
    t_dedup = encode_domain_event(
        "NEWS_ARTICLE", "nytimes.com",
        ["cloudflare", "cloudflare", "cloudflare"],
    )
    t_single = encode_domain_event(
        "NEWS_ARTICLE", "nytimes.com",
        ["cloudflare"],
    )
    assert torch.equal(t_dedup, t_single), (
        "Duplicate signals should be deduplicated to a single token."
    )


def _test_encode_domain_event_priority_ordering() -> None:
    # status_200 (rank 0) should appear before cloudflare (rank 20).
    t = encode_domain_event(
        "NEWS_ARTICLE", "nytimes.com",
        ["cloudflare", "status_200", "tls_1_3"],
    )
    tok_list = t.tolist()
    # Position 0: topology, position 1: domain.
    # Position 2 onwards: structural tokens in priority order.
    status200_pos  = tok_list.index(PRIMITIVE_STATUS_200)
    cloudflare_pos = tok_list.index(PRIMITIVE_CLOUDFLARE)
    tls13_pos      = tok_list.index(PRIMITIVE_TLS_1_3)
    assert status200_pos < cloudflare_pos < tls13_pos, (
        f"Priority ordering violated: status_200 at {status200_pos}, "
        f"cloudflare at {cloudflare_pos}, tls_1_3 at {tls13_pos}"
    )


def _test_encode_domain_event_truncation() -> None:
    # Generate enough signals to exceed max_length=10.
    # 2 anchors + 8 structural = 10 exactly.
    signals = [
        "cloudflare", "tls_1_3", "http_2", "status_200",
        "robots_present", "sitemap_present", "text_html", "server_nginx",
        "wordpress",  # This one would be at position 10 — truncated.
    ]
    t = encode_domain_event("SAAS_DOCS", "stripe.com", signals, max_length=10)
    assert t.shape == torch.Size([10])
    # Anchors preserved.
    assert t[0].item() == topology_class_to_token("SAAS_DOCS")
    assert t[1].item() == domain_to_token("stripe.com")


def _test_encode_domain_event_empty_signals() -> None:
    t = encode_domain_event("GENERIC_HTML", "example.com", [])
    stats = sequence_stats(t)
    assert stats["topology_tokens"] == 1
    assert stats["domain_tokens"] == 1
    assert stats["structural_tokens"] == 0
    assert stats["intent_tokens"] == 0
    assert stats["pad_tokens"] == MAX_SEQ_LEN - 2


def _test_encode_batch_basic() -> None:
    events: list[DomainEventSpec] = [
        {"topology_class": "SAAS_DOCS", "domain": "docs.stripe.com", "structural_signals": ["cloudflare"]},
        {"topology_class": "NEWS_ARTICLE", "domain": "nytimes.com", "structural_signals": ["status_200"]},
    ]
    tokens, mask = encode_batch(events)
    assert tokens.shape[0] == 2
    assert mask.shape == tokens.shape
    assert tokens.dtype == torch.long
    assert mask.dtype == torch.long
    # All mask values are 0 or 1.
    assert mask.min().item() >= 0
    assert mask.max().item() <= 1


def _test_encode_batch_variable_length_padding() -> None:
    # Batch of 2 events: one with many signals, one with few.
    events: list[DomainEventSpec] = [
        {"topology_class": "SAAS_DOCS", "domain": "docs.stripe.com", "structural_signals": ["cloudflare"]},
        {"topology_class": "NEWS_ARTICLE", "domain": "nytimes.com", "structural_signals": ["status_200"]},
    ]
    tokens, mask = encode_batch(events)
    # Batch seq len should be capped at the longest real sequence, not MAX_SEQ_LEN.
    # Longest sequence has 2 anchors + 6 structural = 8 real tokens.
    assert tokens.shape[1] <= MAX_SEQ_LEN
    # The shorter sequence (3 real tokens) should have PAD in the remaining positions.
    short_seq = tokens[1].tolist()
    real_tokens_in_short = sum(1 for t in short_seq if t != PAD_TOKEN)
    assert real_tokens_in_short == 3  # topology + domain + status_200


def _test_encode_batch_empty() -> None:
    tokens, mask = encode_batch([])
    # Empty batch must return tensors with batch dimension 0.
    # Shape is (0, 0) — zero sequences, zero columns.
    assert tokens.numel() == 0, f"Expected empty token tensor, got numel={tokens.numel()}"
    assert mask.numel()   == 0, f"Expected empty mask tensor, got numel={mask.numel()}"
    assert tokens.dtype == torch.long
    assert mask.dtype   == torch.long


def _test_encode_continuation_appends() -> None:
    t0 = encode_domain_event("SAAS_DOCS", "stripe.com", ["cloudflare"])
    t1 = encode_continuation(t0, "SAAS_DOCS", "stripe.com", ["tls_1_3"])
    # Should contain both cloudflare and tls_1_3.
    tok_list = t1.tolist()
    assert PRIMITIVE_CLOUDFLARE in tok_list
    assert PRIMITIVE_TLS_1_3 in tok_list
    # Anchors intact.
    assert t1[0].item() == topology_class_to_token("SAAS_DOCS")
    assert t1[1].item() == domain_to_token("stripe.com")


def _test_encode_continuation_left_truncation_reanchors() -> None:
    # Fill up a sequence to max_length, then continuation must left-truncate.
    many_signals = [
        "cloudflare", "tls_1_3", "http_2", "status_200", "robots_present",
        "sitemap_present", "text_html", "server_nginx",
    ]
    t0 = encode_domain_event("SAAS_DOCS", "stripe.com", many_signals, max_length=10)
    # t0 is now 10 tokens (2 anchor + 8 structural).  Adding more triggers truncation.
    t1 = encode_continuation(t0, "SAAS_DOCS", "stripe.com", ["wordpress"], max_length=10)
    # Anchors must always be at positions 0 and 1.
    assert t1[0].item() == topology_class_to_token("SAAS_DOCS")
    assert t1[1].item() == domain_to_token("stripe.com")
    assert t1.shape == torch.Size([10])


def _test_validate_sequence_valid() -> None:
    t = encode_domain_event("NEWS_ARTICLE", "nytimes.com", ["cloudflare"])
    assert validate_sequence(t) is True


def _test_validate_sequence_invalid() -> None:
    # Token exceeding VOCAB_SIZE.
    bad = torch.tensor([1, 1024, 9999], dtype=torch.long)
    assert validate_sequence(bad) is False
    # Negative token.
    bad2 = torch.tensor([1, 1024, -1], dtype=torch.long)
    assert validate_sequence(bad2) is False


def _test_validate_sequence_structure_valid() -> None:
    t = encode_domain_event("SAAS_DOCS", "stripe.com", ["cloudflare", "tls_1_3"])
    valid, reason = validate_sequence_structure(t)
    assert valid is True, f"Valid sequence failed structure check: {reason}"
    assert reason == ""


def _test_validate_sequence_structure_bad_topology() -> None:
    bad = torch.tensor([0, 1024, 77, 0, 0], dtype=torch.long)  # PAD at pos 0
    valid, reason = validate_sequence_structure(bad)
    assert valid is False
    assert "topology" in reason.lower() or "position 0" in reason.lower()


def _test_sequence_stats_counts() -> None:
    t = encode_domain_event(
        "REST_API_JSON", "api.github.com",
        ["cloudflare", "tls_1_3", "http_2"],
    )
    stats = sequence_stats(t)
    assert stats["topology_tokens"]   == 1
    assert stats["domain_tokens"]     == 1
    assert stats["structural_tokens"] == 3
    assert stats["intent_tokens"]     == 0
    assert stats["pad_tokens"]        == MAX_SEQ_LEN - 5
    assert stats["real_tokens"]       == 5
    assert stats["unknown_range"]     == 0
    assert stats["total_length"]      == MAX_SEQ_LEN


def _test_token_to_topology_class_roundtrip() -> None:
    for cls in TOPOLOGY_CLASSES:
        tok = topology_class_to_token(cls)
        recovered = token_to_topology_class(tok)
        assert recovered == cls, (
            f"Round-trip failed for {cls!r}: encode→{tok}, decode→{recovered!r}"
        )


def _test_token_to_topology_class_out_of_range() -> None:
    assert token_to_topology_class(0)    is None  # PAD
    assert token_to_topology_class(19)   is None  # structural
    assert token_to_topology_class(1024) is None  # domain
    assert token_to_topology_class(4096) is None  # intent


def _test_token_to_domain_hint() -> None:
    assert token_to_domain_hint(1024) == "domain_bucket[0]"
    assert token_to_domain_hint(4095) == "domain_bucket[3071]"
    assert "not_a_domain_token" in token_to_domain_hint(1)


def _test_token_to_primitive_name_roundtrip() -> None:
    known_pairs = [
        (PRIMITIVE_CLOUDFLARE,   "cloudflare"),
        (PRIMITIVE_STATUS_200,   "status_200"),
        (PRIMITIVE_TLS_1_3,      "tls_1_3"),
        (PRIMITIVE_UNKNOWN,      "UNKNOWN_PRIMITIVE"),
    ]
    for tok, expected_name in known_pairs:
        name = token_to_primitive_name(tok)
        assert name == expected_name, (
            f"token_to_primitive_name({tok}) = {name!r}, expected {expected_name!r}"
        )


def _test_token_to_primitive_name_reserved() -> None:
    name = token_to_primitive_name(FIRST_RESERVED_PRIMITIVE)
    assert name == f"RESERVED({FIRST_RESERVED_PRIMITIVE})"


def _test_token_to_primitive_name_out_of_range() -> None:
    assert token_to_primitive_name(0)    is None  # PAD
    assert token_to_primitive_name(1)    is None  # topology
    assert token_to_primitive_name(1024) is None  # domain
    assert token_to_primitive_name(4096) is None  # intent


def _test_response_time_to_signal() -> None:
    assert response_time_to_signal(50.0)    == "lt_100ms"
    assert response_time_to_signal(99.9)    == "lt_100ms"
    assert response_time_to_signal(100.0)   == "100_500ms"
    assert response_time_to_signal(499.9)   == "100_500ms"
    assert response_time_to_signal(500.0)   == "500ms_2s"
    assert response_time_to_signal(1999.9)  == "500ms_2s"
    assert response_time_to_signal(2000.0)  == "2s_5s"
    assert response_time_to_signal(4999.9)  == "2s_5s"
    assert response_time_to_signal(5000.0)  == "gt_5s"
    assert response_time_to_signal(99999.0) == "gt_5s"


def _test_http_status_to_signal() -> None:
    assert http_status_to_signal(200) == "status_200"
    assert http_status_to_signal(301) == "status_301"
    assert http_status_to_signal(404) is None
    assert http_status_to_signal(429) == "status_429"


def _test_decode_intent_chunk_value() -> None:
    # All zeros.
    assert decode_intent_chunk_value(0) == [0, 0, 0, 0, 0, 0]
    # All threes.
    assert decode_intent_chunk_value(4095) == [3, 3, 3, 3, 3, 3]
    # All twos: 2*(4^5+4^4+4^3+4^2+4+1) = 2*1365 = 2730.
    assert decode_intent_chunk_value(2730) == [2, 2, 2, 2, 2, 2]
    # Mixed: [1, 2, 3, 0, 1, 2] = 1*1024 + 2*256 + 3*64 + 0*16 + 1*4 + 2 = 1024+512+192+0+4+2 = 1734
    assert decode_intent_chunk_value(1734) == [1, 2, 3, 0, 1, 2]


def _test_describe_sequence_runs() -> None:
    t = encode_domain_event("SAAS_DOCS", "stripe.com", ["cloudflare", "tls_1_3"])
    desc = describe_sequence(t)
    assert "SAAS_DOCS" in desc
    assert "domain_bucket" in desc
    assert isinstance(desc, str)
    assert len(desc) > 0


def _test_encode_domain_event_all_topology_classes() -> None:
    """All 18 topology classes should encode without error."""
    for cls in TOPOLOGY_CLASSES:
        t = encode_domain_event(cls, "example.com", ["status_200"])
        assert t[0].item() == topology_class_to_token(cls), (
            f"Anchor mismatch for topology class {cls!r}"
        )


def _test_structural_all_known_signals_have_tokens() -> None:
    """Every canonical signal name should map to a token in structural range."""
    canonical_signals = [
        "cloudflare", "fastly", "akamai", "cloudfront", "vercel", "netlify",
        "wordpress", "ghost", "drupal", "confluence", "notion", "gitbook",
        "docusaurus", "mkdocs",
        "requires_js", "static_only", "spa_detected", "ssr_detected",
        "hydration_detected",
        "cloudflare_challenge", "recaptcha", "hcaptcha", "datadome",
        "perimeter_x", "rate_limit_header",
        "tls_1_2", "tls_1_3", "cert_wildcard", "cert_org", "cert_lets_encrypt",
        "http_1_1", "http_2", "http_3",
        "lt_100ms", "100_500ms", "500ms_2s", "2s_5s", "gt_5s",
        "robots_present", "robots_absent", "crawl_delay_set", "disallow_heavy",
        "sitemap_linked",
        "sitemap_present", "sitemap_index", "sitemap_urlset", "sitemap_news",
        "sitemap_image",
        "text_html", "application_json", "application_ld_json", "text_plain",
        "application_xml",
        "x_powered_by_present", "server_nginx", "server_apache", "server_caddy",
        "via_present",
        "status_200", "status_301", "status_302", "status_403", "status_429",
        "status_503",
    ]
    for sig in canonical_signals:
        tok = structural_signal_to_token(sig)
        assert STRUCTURAL_TOKEN_OFFSET <= tok < FIRST_RESERVED_PRIMITIVE, (
            f"Canonical signal {sig!r} mapped to non-named-range token {tok}"
        )
        assert tok != PRIMITIVE_UNKNOWN, (
            f"Canonical signal {sig!r} unexpectedly mapped to PRIMITIVE_UNKNOWN"
        )


def _all_test_functions():
    """Return all test functions in order."""
    return [
        _test_vocabulary_integrity,
        _test_pad_token_not_in_vocab_ranges,
        _test_topology_token_derivation,
        _test_topology_unknown_fallback,
        _test_normalize_domain_basic,
        _test_normalize_domain_multi_part_tld,
        _test_normalize_domain_port_stripping,
        _test_domain_to_token_consistency,
        _test_domain_to_token_range,
        _test_domain_to_token_determinism,
        _test_structural_signal_known,
        _test_structural_signal_aliases,
        _test_structural_signal_case_insensitive,
        _test_structural_signal_unknown,
        _test_quantize_intent_dimension,
        _test_intent_vector_to_tokens_shape,
        _test_intent_vector_to_tokens_all_zeros,
        _test_intent_vector_to_tokens_all_ones_max,
        _test_intent_vector_wrong_dim,
        _test_intent_vector_determinism,
        _test_pad_sequence_short,
        _test_pad_sequence_exact,
        _test_pad_sequence_truncate,
        _test_make_attention_mask,
        _test_encode_domain_event_shape,
        _test_encode_domain_event_anchor_positions,
        _test_encode_domain_event_with_intent,
        _test_encode_domain_event_deduplication,
        _test_encode_domain_event_priority_ordering,
        _test_encode_domain_event_truncation,
        _test_encode_domain_event_empty_signals,
        _test_encode_batch_basic,
        _test_encode_batch_variable_length_padding,
        _test_encode_batch_empty,
        _test_encode_continuation_appends,
        _test_encode_continuation_left_truncation_reanchors,
        _test_validate_sequence_valid,
        _test_validate_sequence_invalid,
        _test_validate_sequence_structure_valid,
        _test_validate_sequence_structure_bad_topology,
        _test_sequence_stats_counts,
        _test_token_to_topology_class_roundtrip,
        _test_token_to_topology_class_out_of_range,
        _test_token_to_domain_hint,
        _test_token_to_primitive_name_roundtrip,
        _test_token_to_primitive_name_reserved,
        _test_token_to_primitive_name_out_of_range,
        _test_response_time_to_signal,
        _test_http_status_to_signal,
        _test_decode_intent_chunk_value,
        _test_describe_sequence_runs,
        _test_encode_domain_event_all_topology_classes,
        _test_structural_all_known_signals_have_tokens,
    ]


def run_tests(verbose: bool = True) -> bool:
    """
    Run all self-contained unit tests and report results.

    Args:
        verbose: If True, print each test result.  If False, only print summary.

    Returns:
        True if all tests passed, False if any failed.
    """
    from typing import Callable # noqa

    tests = _all_test_functions()
    results: List[_TokenizerTestResult] = []

    for fn in tests:
        r = _run_test(fn.__name__, fn)
        results.append(r)
        if verbose:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.test_name}" + (f": {r.message}" if not r.passed else ""))

    passed = sum(1 for r in results if r.passed)
    failed = sum(1 for r in results if not r.passed)

    print(f"\nwlm_tokenizer tests: {passed}/{len(results)} passed", end="")
    if failed:
        print(f", {failed} FAILED")
        for r in results:
            if not r.passed:
                print(f"  FAILED: {r.test_name}\n    {r.message}")
    else:
        print(" ✓")

    return failed == 0


# ═════════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL INTEGRITY CHECK
#
# Run at import time to catch any constant mis-assignment immediately.
# A broken tokenizer that silently produces wrong tokens would corrupt the MFT.
# Fail loudly at import rather than silently at training time.
# ═════════════════════════════════════════════════════════════════════════════

verify_vocabulary_integrity()


# ═════════════════════════════════════════════════════════════════════════════
# MODULE PUBLIC API SURFACE
# Everything not listed here is considered private (prefixed _ or utility).
# ═════════════════════════════════════════════════════════════════════════════

__all__: List[str] = [
    # ── Vocabulary constants ──────────────────────────────────────────────
    "VOCAB_SIZE",
    "PAD_TOKEN",
    "MAX_SEQ_LEN",
    "TOPOLOGY_TOKEN_OFFSET",
    "TOPOLOGY_TOKEN_END",
    "STRUCTURAL_TOKEN_OFFSET",
    "STRUCTURAL_TOKEN_END",
    "DOMAIN_TOKEN_OFFSET",
    "DOMAIN_TOKEN_END",
    "INTENT_TOKEN_OFFSET",
    "INTENT_TOKEN_END",
    # ── Topology token constants ──────────────────────────────────────────
    "TOPOLOGY_TOKEN_NEWS_ARTICLE",
    "TOPOLOGY_TOKEN_NEWS_ARTICLE_PAYWALLED",
    "TOPOLOGY_TOKEN_SAAS_DOCS",
    "TOPOLOGY_TOKEN_SAAS_DOCS_VERSIONED",
    "TOPOLOGY_TOKEN_SAAS_DOCS_WITH_CODE",
    "TOPOLOGY_TOKEN_REST_API_JSON",
    "TOPOLOGY_TOKEN_REST_API_JSON_PAGINATED",
    "TOPOLOGY_TOKEN_JSON_LD_STRUCTURED",
    "TOPOLOGY_TOKEN_ECOMMERCE_PRODUCT",
    "TOPOLOGY_TOKEN_ECOMMERCE_PRODUCT_VARIANT",
    "TOPOLOGY_TOKEN_FORUM_THREAD",
    "TOPOLOGY_TOKEN_BLOG_POST",
    "TOPOLOGY_TOKEN_WIKIPEDIA_ARTICLE",
    "TOPOLOGY_TOKEN_LANDING_PAGE",
    "TOPOLOGY_TOKEN_AUTH_REDIRECT",
    "TOPOLOGY_TOKEN_CLOUDFLARE_CHALLENGE",
    "TOPOLOGY_TOKEN_RATE_LIMITED",
    "TOPOLOGY_TOKEN_GENERIC_HTML",
    # ── Structural primitive constants ────────────────────────────────────
    "PRIMITIVE_CLOUDFLARE",
    "PRIMITIVE_FASTLY",
    "PRIMITIVE_AKAMAI",
    "PRIMITIVE_CLOUDFRONT",
    "PRIMITIVE_VERCEL",
    "PRIMITIVE_NETLIFY",
    "PRIMITIVE_WORDPRESS",
    "PRIMITIVE_GHOST",
    "PRIMITIVE_DRUPAL",
    "PRIMITIVE_CONFLUENCE",
    "PRIMITIVE_NOTION",
    "PRIMITIVE_GITBOOK",
    "PRIMITIVE_DOCUSAURUS",
    "PRIMITIVE_MKDOCS",
    "PRIMITIVE_REQUIRES_JS",
    "PRIMITIVE_STATIC_ONLY",
    "PRIMITIVE_SPA_DETECTED",
    "PRIMITIVE_SSR_DETECTED",
    "PRIMITIVE_HYDRATION_DETECTED",
    "PRIMITIVE_CLOUDFLARE_CHALLENGE",
    "PRIMITIVE_RECAPTCHA",
    "PRIMITIVE_HCAPTCHA",
    "PRIMITIVE_DATADOME",    "PRIMITIVE_PERIMETER_X",
    "PRIMITIVE_RATE_LIMIT_HEADER",
    "PRIMITIVE_TLS_1_2",
    "PRIMITIVE_TLS_1_3",
    "PRIMITIVE_CERT_WILDCARD",
    "PRIMITIVE_CERT_ORG",
    "PRIMITIVE_CERT_LETS_ENCRYPT",
    "PRIMITIVE_HTTP_1_1",
    "PRIMITIVE_HTTP_2",
    "PRIMITIVE_HTTP_3",
    "PRIMITIVE_RESPONSE_LT_100MS",
    "PRIMITIVE_RESPONSE_100_500MS",
    "PRIMITIVE_RESPONSE_500MS_2S",
    "PRIMITIVE_RESPONSE_2S_5S",
    "PRIMITIVE_RESPONSE_GT_5S",
    "PRIMITIVE_ROBOTS_PRESENT",
    "PRIMITIVE_ROBOTS_ABSENT",
    "PRIMITIVE_CRAWL_DELAY_SET",
    "PRIMITIVE_DISALLOW_HEAVY",
    "PRIMITIVE_SITEMAP_LINKED",
    "PRIMITIVE_SITEMAP_PRESENT",
    "PRIMITIVE_SITEMAP_INDEX",
    "PRIMITIVE_SITEMAP_URLSET",
    "PRIMITIVE_SITEMAP_NEWS",
    "PRIMITIVE_SITEMAP_IMAGE",
    "PRIMITIVE_TEXT_HTML",
    "PRIMITIVE_APPLICATION_JSON",
    "PRIMITIVE_APPLICATION_LD_JSON",
    "PRIMITIVE_TEXT_PLAIN",
    "PRIMITIVE_APPLICATION_XML",
    "PRIMITIVE_X_POWERED_BY_PRESENT",
    "PRIMITIVE_SERVER_NGINX",
    "PRIMITIVE_SERVER_APACHE",
    "PRIMITIVE_SERVER_CADDY",
    "PRIMITIVE_VIA_PRESENT",
    "PRIMITIVE_STATUS_200",
    "PRIMITIVE_STATUS_301",
    "PRIMITIVE_STATUS_302",
    "PRIMITIVE_STATUS_403",
    "PRIMITIVE_STATUS_429",
    "PRIMITIVE_STATUS_503",
    "PRIMITIVE_UNKNOWN",
    "FIRST_RESERVED_PRIMITIVE",
    "LAST_RESERVED_PRIMITIVE",
    # ── Intent quantization constants ─────────────────────────────────────
    "INTENT_QUANT_THRESHOLDS",
    "INTENT_VECTOR_DIM",
    "INTENT_CHUNK_SIZE",
    "INTENT_TOKENS_PER_VECTOR",
    # ── Signal priority ───────────────────────────────────────────────────
    "SIGNAL_PRIORITY_RANK",
    # ── Domain normalization ──────────────────────────────────────────────
    "normalize_domain",
    # ── Core encoding functions ───────────────────────────────────────────
    "topology_class_to_token",
    "domain_to_token",
    "structural_signal_to_token",
    "quantize_intent_dimension",
    "intent_vector_to_tokens",
    # ── Padding utilities ─────────────────────────────────────────────────
    "pad_sequence",
    "make_attention_mask",
    # ── Public sequence assembly API ──────────────────────────────────────
    "DomainEventSpec",
    "encode_domain_event",
    "encode_batch",
    "encode_continuation",
    # ── Decode functions (debug / audit only) ─────────────────────────────
    "token_to_topology_class",
    "token_to_domain_hint",
    "token_to_primitive_name",
    "token_to_intent_hint",
    "decode_intent_chunk_value",
    "describe_sequence",
    # ── Validation ────────────────────────────────────────────────────────
    "validate_sequence",
    "validate_sequence_structure",
    "sequence_stats",
    "verify_vocabulary_integrity",
    # ── Helper utilities ──────────────────────────────────────────────────
    "response_time_to_signal",
    "http_status_to_signal",
    # ── Test runner ───────────────────────────────────────────────────────
    "run_tests",
]


if __name__ == "__main__":
    import sys
    success = run_tests(verbose=True)
    sys.exit(0 if success else 1)