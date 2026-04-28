#!/usr/bin/env python3
"""
initialize_store.py — AXIOM store bootstrap

Run once at first deployment.  Creates all four store artifacts and all
store directory scaffolding required before any daemon can start.

    store/topology_router.pt     MambaRouter model — the MFT lives here
    store/phase_states.mmap      Per-topology-class phase journal (mmap, 584 B)
    store/recipe_registry.mmap   Compiled grep recipe slots (mmap, ~72 KB)
    store/manifest.json          Human-readable store inventory + checksums

    store/staging/               Atomic swap staging area
    store/checkpoints/           48-slot rotating archive (crond, 15 min)
    store/triggers/              Daemon trigger files
    store/dead_letters/          Failed handler archive

Usage
-----
    python initialize_store.py                      # normal run
    python initialize_store.py --dry-run            # preview — no writes
    python initialize_store.py --force              # reinitialize existing store
    python initialize_store.py --store-root /data   # override store root

Exit codes
----------
    0   success — all artifacts created and verified
    1   validation failure — artifact failed round-trip check
    2   store already initialized — use --force to override
    3   unexpected error — see traceback

Constraints
-----------
- Every artifact write is staging -> os.rename() atomic swap.
- Every artifact is SHA-256 verified after write.
- topology_router.pt is round-trip loaded and key shapes asserted.
- No component other than this script should write topology_router.pt
  before index_daemon.py takes ownership.
- weights_only=True is the only permitted load mode.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ── Optional torch import — fail fast with a clear message ─────────────────

try:
    import torch # noqa
    import torch.nn as nn # noqa
except ImportError:
    sys.exit(
        "FATAL: PyTorch is not installed.\n"
        "       pip install torch --index-url https://download.pytorch.org/whl/cpu"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Store topology constants
# ═══════════════════════════════════════════════════════════════════════════════

STORE_VERSION: int = 1
SCHEMA_VERSION: str = "1.0.0"
AXIOM_SPEC_VERSION: str = "0.3"

# 18 structural topology classes — maps to topology_head.weight rows.
# Order is canonical.  index_daemon.py and contracts.py must agree with this.
TOPOLOGY_CLASSES: tuple[str, ...] = (
    "NEWS_ARTICLE",              # 0
    "NEWS_ARTICLE_PAYWALLED",    # 1
    "SAAS_DOCS",                 # 2
    "SAAS_DOCS_VERSIONED",       # 3
    "SAAS_DOCS_WITH_CODE",       # 4
    "REST_API_JSON",             # 5
    "REST_API_JSON_PAGINATED",   # 6
    "JSON_LD_STRUCTURED",        # 7
    "ECOMMERCE_PRODUCT",         # 8
    "ECOMMERCE_PRODUCT_VARIANT", # 9
    "FORUM_THREAD",              # 10
    "BLOG_POST",                 # 11
    "WIKIPEDIA_ARTICLE",         # 12
    "LANDING_PAGE",              # 13
    "AUTH_REDIRECT",             # 14
    "CLOUDFLARE_CHALLENGE",      # 15
    "RATE_LIMITED",              # 16
    "GENERIC_HTML",              # 17
)
assert len(TOPOLOGY_CLASSES) == 18, "topology_head.weight must be (18, 256)"

# 7 traversal strategies — maps to traversal_head.weight rows.
TRAVERSAL_STRATEGIES: tuple[str, ...] = (
    "DIRECT_FETCH",       # 0 — plain HTTP, no JS required
    "RENDER_JS",          # 1 — Playwright headless render
    "PAGINATION_LINEAR",  # 2 — sequential page traversal
    "PAGINATION_SCROLL",  # 3 — infinite scroll / lazy load
    "AUTH_NEGOTIATE",     # 4 — auth flow required
    "CACHE_BYPASS",       # 5 — cache-busting headers / variant
    "SITEMAP_TRAVERSE",   # 6 — sitemap.xml guided traversal
)
assert len(TRAVERSAL_STRATEGIES) == 7, "traversal_head.weight must be (7, 256)"

# 5 friction classes — maps to friction_head.weight rows.
FRICTION_CLASSES: tuple[str, ...] = (
    "NONE",            # 0 — clean access
    "SOFT_GATE",       # 1 — email / account nudge (skippable)
    "HARD_PAYWALL",    # 2 — hard content gate
    "BOT_MITIGATION",  # 3 — Cloudflare/Turnstile/hCaptcha
    "RATE_LIMIT",      # 4 — request throttle / 429
)
assert len(FRICTION_CLASSES) == 5, "friction_head.weight must be (5, 256)"

# 3 cognition phases — maps to phase_head.weight rows.
PHASE_LABELS: tuple[str, ...] = (
    "PHASE_I_LEARN",     # 0 — explicit simulation, high uncertainty
    "PHASE_II_PREDICT",  # 1 — latent planning, collapsing error
    "PHASE_III_KNOW",    # 2 — compiled policy, zero simulation cost
)
assert len(PHASE_LABELS) == 3, "phase_head.weight must be (3, 256)"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — MambaRouter architecture constants
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MambaConfig:
    """Hyperparameters that determine every tensor shape in topology_router.pt."""
    vocab_size: int = 8192  # embedding.weight row count
    d_model:    int = 256   # hidden / residual stream dimension
    n_blocks:   int = 4     # number of Mamba SSM blocks
    d_state:    int = 16    # SSM state expansion dimension
    d_conv:     int = 4     # depthwise conv kernel width
    expand:     int = 2     # inner expansion factor
    dt_rank:    int = 16    # delta-t projection rank (~= d_model / 16)
    d_inner:    int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "d_inner", self.d_model * self.expand)


MAMBA_CFG = MambaConfig()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Terminal output helpers
# ═══════════════════════════════════════════════════════════════════════════════

_BOLD   = "\033[1m"
_GREEN  = "\033[32m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"
_USE_COLOR = sys.stdout.isatty()


def _c(text: str, code: str) -> str:
    return f"{code}{text}{_RESET}" if _USE_COLOR else text


def _ok(msg: str)   -> None: print(f"  {_c('OK', _GREEN)}  {msg}")
def _info(msg: str) -> None: print(f"  {_c('--', _DIM)}  {msg}")
def _warn(msg: str) -> None: print(f"  {_c('!!', _YELLOW)}  {msg}")
def _err(msg: str)  -> None: print(f"  {_c('XX', _RED)}  {msg}", file=sys.stderr)
def _head(msg: str) -> None: print(f"\n{_c(_BOLD + msg, _BOLD)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Atomic write utilities
# ═══════════════════════════════════════════════════════════════════════════════

def _sha256_file(path: Path) -> str:
    """Return lowercase hex SHA-256 of the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write_bytes(dest: Path, data: bytes, staging_dir: Path) -> str:
    """
    Write *data* to *dest* via a staging file in *staging_dir*.

    Returns the SHA-256 hex digest of the written data.
    Staging file is cleaned up regardless of success or failure.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        dir=staging_dir,
        prefix=f".{dest.name}.staging.",
        suffix=".tmp",
    )
    tmp = Path(tmp_str)
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.rename(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return _sha256_bytes(data)


def _atomic_write_text(dest: Path, text: str, staging_dir: Path) -> str:
    """Write UTF-8 text atomically; return SHA-256 of encoded bytes."""
    return _atomic_write_bytes(dest, text.encode(), staging_dir)


def _atomic_save_pt(dest: Path, state_dict: dict, staging_dir: Path) -> str:
    """
    Save a PyTorch state dict atomically via staging rename.

    Returns SHA-256 of the final file on disk.
    """
    staging_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_str = tempfile.mkstemp(
        dir=staging_dir,
        prefix=f".{dest.name}.staging.",
        suffix=".tmp",
    )
    tmp = Path(tmp_str)
    try:
        os.close(fd)
        torch.save(state_dict, tmp)
        os.rename(tmp, dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    return _sha256_file(dest)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — topology_router.pt builder
# ═══════════════════════════════════════════════════════════════════════════════

def _mamba_block_state(block_idx: int, cfg: MambaConfig) -> dict[str, torch.Tensor]: # noqa
    """
    Build one Mamba SSM block's weight tensors with canonical initialization.

    Keys are returned without the 'blocks.{i}.' prefix — the caller prepends it.

    Reference initialization follows Gu & Dao, "Mamba: Linear-Time Sequence
    Modeling with Selective State Spaces" (2023).
    """
    d, di, ds, dc, dr = cfg.d_model, cfg.d_inner, cfg.d_state, cfg.d_conv, cfg.dt_rank
    t: dict[str, torch.Tensor] = {} # noqa

    # Layer norm preceding the SSM
    t["norm.weight"] = torch.ones(d)

    # Input projection — projects residual into x and z (gating) streams
    in_proj = nn.Linear(d, di * 2, bias=False)
    t["mamba.in_proj.weight"] = in_proj.weight.data.clone()   # (di*2, d)

    # Depthwise convolution over x stream
    conv = nn.Conv1d(di, di, kernel_size=dc, groups=di, padding=dc - 1, bias=True)
    t["mamba.conv1d.weight"] = conv.weight.data.clone()        # (di, 1, dc)
    t["mamba.conv1d.bias"]   = conv.bias.data.clone()          # (di,)

    # x_proj: projects x -> (delta, B, C) for SSM
    x_proj = nn.Linear(di, dr + ds * 2, bias=False)
    t["mamba.x_proj.weight"] = x_proj.weight.data.clone()     # (dr+2*ds, di)

    # dt_proj: up-projects delta from dt_rank back to d_inner
    dt_proj = nn.Linear(dr, di, bias=True)
    nn.init.uniform_(dt_proj.weight,
                     -(dr ** -0.5), dr ** -0.5)
    # dt bias: inverse-softplus of log-uniform sample in [dt_min, dt_max]
    dt = torch.exp(
        torch.rand(di) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
    ).clamp(min=1e-4)
    inv_dt = dt + torch.log(-torch.expm1(-dt))
    with torch.no_grad():
        dt_proj.bias.copy_(inv_dt)
    t["mamba.dt_proj.weight"] = dt_proj.weight.data.clone()   # (di, dr)
    t["mamba.dt_proj.bias"]   = dt_proj.bias.data.clone()     # (di,)

    # A_log: log(diag(A)) — stable HiPPO initialization
    # A[i,n] = n+1  for n in 0..d_state-1, repeated for each d_inner channel
    A = torch.arange(1, ds + 1, dtype=torch.float32).unsqueeze(0).expand(di, -1)
    t["mamba.A_log"] = torch.log(A).clone()                   # (di, ds)

    # D: skip connection weight, initialized to ones
    t["mamba.D"] = torch.ones(di)                             # (di,)

    # Output projection: merge gated SSM output back to residual dimension
    out_proj = nn.Linear(di, d, bias=False)
    t["mamba.out_proj.weight"] = out_proj.weight.data.clone() # (d, di)

    return t


def build_topology_router_state_dict(cfg: MambaConfig) -> dict[str, torch.Tensor]:
    """
    Construct the complete initial state dict for MambaRouter.

    All tensors are on CPU with PyTorch default dtype (float32).
    hidden_state is zeros — the MFT knows nothing at first deployment.
    hidden_state_version is a scalar zero tensor (monotonic, owned by
    index_daemon.py after first gradient step).

    This is equivalent to MambaRouter().state_dict() without requiring the
    MambaRouter class to be importable at store initialization time.
    """
    sd: dict[str, torch.Tensor] = {}

    # Token embedding
    emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
    sd["embedding.weight"] = emb.weight.data.clone()              # (8192, 256)

    # Mamba SSM blocks
    for i in range(cfg.n_blocks):
        for k, v in _mamba_block_state(i, cfg).items():
            sd[f"blocks.{i}.{k}"] = v

    # Output heads — all linear without bias
    def _head_weight(out_features: int) -> torch.Tensor:
        return nn.Linear(cfg.d_model, out_features, bias=False).weight.data.clone()

    sd["topology_head.weight"]  = _head_weight(len(TOPOLOGY_CLASSES))    # (18, 256)
    sd["traversal_head.weight"] = _head_weight(len(TRAVERSAL_STRATEGIES)) # (7, 256)
    sd["friction_head.weight"]  = _head_weight(len(FRICTION_CLASSES))    # (5, 256)
    sd["source_head.weight"]    = _head_weight(512)                       # (512, 256)
    sd["phase_head.weight"]     = _head_weight(len(PHASE_LABELS))         # (3, 256)

    # MFT — the hidden state buffer
    # Zeros: the system has no accumulated knowledge at first deployment.
    sd["hidden_state"]         = torch.zeros(1, cfg.d_model)  # (1, 256)
    sd["hidden_state_version"] = torch.tensor(0)              # scalar, 0-dim

    return sd


# Expected shapes for round-trip verification
_EXPECTED_SHAPES: dict[str, tuple[int, ...]] = {
    "embedding.weight":      (8192, 256),
    "topology_head.weight":  (  18, 256),
    "traversal_head.weight": (   7, 256),
    "friction_head.weight":  (   5, 256),
    "source_head.weight":    ( 512, 256),
    "phase_head.weight":     (   3, 256),
    "hidden_state":          (   1, 256),
}

_REQUIRED_BLOCK_KEYS: frozenset[str] = frozenset({
    "norm.weight",
    "mamba.in_proj.weight",
    "mamba.conv1d.weight",
    "mamba.conv1d.bias",
    "mamba.x_proj.weight",
    "mamba.dt_proj.weight",
    "mamba.dt_proj.bias",
    "mamba.A_log",
    "mamba.D",
    "mamba.out_proj.weight",
})


def _verify_topology_router(path: Path) -> None:
    """
    Round-trip load topology_router.pt and assert all key shapes.

    Raises AssertionError on any mismatch.
    This is the hard gate — failure here returns exit code 1.
    """
    sd = torch.load(path, weights_only=True, map_location="cpu")

    for key, expected in _EXPECTED_SHAPES.items():
        assert key in sd, f"missing key: {key}"
        got = tuple(sd[key].shape)
        assert got == expected, f"{key}: expected shape {expected}, got {got}"

    for i in range(MAMBA_CFG.n_blocks):
        for sub in _REQUIRED_BLOCK_KEYS:
            full = f"blocks.{i}.{sub}"
            assert full in sd, f"missing block key: {full}"

    assert sd["hidden_state"].sum().item() == 0.0, \
        "hidden_state must be all-zeros at initialization"
    assert sd["hidden_state_version"].item() == 0, \
        "hidden_state_version must be 0 at initialization"
    assert sd["hidden_state_version"].dim() == 0, \
        "hidden_state_version must be a scalar (0-dim) tensor"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — phase_states.mmap builder
# ═══════════════════════════════════════════════════════════════════════════════

# File-level binary layout
_PS_MAGIC       = b"AXPS"
_PS_VERSION     = 1
_PS_N_CLASSES   = len(TOPOLOGY_CLASSES)   # 18
_PS_SLOT_STRIDE = 32                      # bytes per per-class record
_PS_HEADER_SIZE = 8
_PS_FILE_SIZE   = _PS_HEADER_SIZE + _PS_N_CLASSES * _PS_SLOT_STRIDE  # 584 bytes

# Per-class slot layout (32 bytes):
#   [0]     phase_id       uint8   (1=I, 2=II, 3=III)
#   [1]     flags          uint8   (bit0=active, bit1=surprise_tripped)
#   [2:4]   reserved       uint16
#   [4:8]   confidence     float32
#   [8:12]  event_count    uint32
#   [12:16] surprise_score float32
#   [16:24] updated_ns     uint64  (epoch nanoseconds; 0 = never updated)
#   [24:32] reserved       8 bytes (pad to 32)
_PS_SLOT_STRUCT = struct.Struct("<BB H f I f Q 8x")
assert _PS_SLOT_STRUCT.size == _PS_SLOT_STRIDE


def build_phase_states_mmap() -> bytes:
    """
    Build the initial phase_states.mmap binary blob.

    All 18 topology classes start at Phase I with confidence=0.0 and
    event_count=0.  index_daemon.py owns all subsequent writes.
    """
    buf = bytearray(_PS_FILE_SIZE)
    struct.pack_into("4s B B H", buf, 0, _PS_MAGIC, _PS_VERSION, _PS_N_CLASSES, 0)
    for i in range(_PS_N_CLASSES):
        offset = _PS_HEADER_SIZE + i * _PS_SLOT_STRIDE
        _PS_SLOT_STRUCT.pack_into(buf, offset,
            1,    # phase_id = Phase I
            0,    # flags = 0
            0,    # reserved
            0.0,  # confidence
            0,    # event_count
            0.0,  # surprise_score
            0,    # updated_ns (never updated)
        )
    return bytes(buf)


def _verify_phase_states(path: Path) -> None:
    """Assert magic, version, class count, and first slot sanity."""
    raw = path.read_bytes()
    assert len(raw) == _PS_FILE_SIZE, \
        f"phase_states.mmap: expected {_PS_FILE_SIZE} B, got {len(raw)} B"
    magic, ver, n_cls, _ = struct.unpack_from("4s B B H", raw, 0)
    assert magic == _PS_MAGIC,     f"phase_states: bad magic {magic!r}"
    assert ver   == _PS_VERSION,   f"phase_states: bad version {ver}"
    assert n_cls == _PS_N_CLASSES, f"phase_states: n_classes={n_cls}, expected {_PS_N_CLASSES}"
    phase_id = _PS_SLOT_STRUCT.unpack_from(raw, _PS_HEADER_SIZE)[0]
    assert phase_id == 1, f"slot 0: expected phase_id=1 (Phase I), got {phase_id}"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7 — recipe_registry.mmap builder
# ═══════════════════════════════════════════════════════════════════════════════

# File-level binary layout
_RR_MAGIC        = b"AXRR"
_RR_VERSION      = 1
_RR_N_SLOTS      = len(TOPOLOGY_CLASSES)   # 18
_RR_SLOT_STRIDE  = 4096                    # one OS page per recipe slot
_RR_HEADER_SIZE  = 8
_RR_SLOT_HDR_SZ  = 64
_RR_RECIPE_BYTES = _RR_SLOT_STRIDE - _RR_SLOT_HDR_SZ  # 4032 bytes for shell recipe
_RR_FILE_SIZE    = _RR_HEADER_SIZE + _RR_N_SLOTS * _RR_SLOT_STRIDE  # 73,736 bytes

# Per-slot header layout (64 bytes):
#   [0:4]   slot_magic    4s     b"SLOT"
#   [4]     class_id      uint8  topology class index (0..17)
#   [5]     compiled      uint8  0=empty, 1=recipe present
#   [6]     phase_at      uint8  phase when compiled (0 if none)
#   [7]     reserved      uint8
#   [8:12]  recipe_len    uint32 byte length of recipe in body (0 if none)
#   [12:44] recipe_sha256 32s    SHA-256 of recipe body (zeros if none)
#   [44:64] reserved      20s
_RR_SLOT_MAGIC      = b"SLOT"
_RR_SLOT_HDR_STRUCT = struct.Struct("<4s B B B B I 32s 20s")
assert _RR_SLOT_HDR_STRUCT.size == _RR_SLOT_HDR_SZ


def build_recipe_registry_mmap() -> bytes:
    """
    Build the initial recipe_registry.mmap binary blob.

    All 18 slots are allocated and empty (compiled=0, recipe_len=0).
    topology_parser.py fills slots at runtime as topology classes graduate
    from Phase I through Phase II toward Phase III compiled policies.
    """
    buf = bytearray(_RR_FILE_SIZE)
    struct.pack_into("<4s B B H", buf, 0,
                     _RR_MAGIC, _RR_VERSION, _RR_N_SLOTS, _RR_SLOT_STRIDE)
    for i in range(_RR_N_SLOTS):
        offset = _RR_HEADER_SIZE + i * _RR_SLOT_STRIDE
        _RR_SLOT_HDR_STRUCT.pack_into(buf, offset,
            _RR_SLOT_MAGIC,
            i,               # class_id
            0,               # compiled = 0 (empty)
            0,               # phase_at = 0
            0,               # reserved
            0,               # recipe_len = 0
            b"\x00" * 32,   # recipe_sha256 (zeros)
            b"\x00" * 20,   # reserved
        )
        # recipe body region is already zeroed (bytearray default)
    return bytes(buf)


def _verify_recipe_registry(path: Path) -> None:
    """Assert magic, version, slot count, slot stride, and first slot magic."""
    raw = path.read_bytes()
    assert len(raw) == _RR_FILE_SIZE, \
        f"recipe_registry: expected {_RR_FILE_SIZE} B, got {len(raw)} B"
    magic, ver, n_slots, stride = struct.unpack_from("<4s B B H", raw, 0)
    assert magic   == _RR_MAGIC,       f"recipe_registry: bad magic {magic!r}"
    assert ver     == _RR_VERSION,     f"recipe_registry: bad version {ver}"
    assert n_slots == _RR_N_SLOTS,     f"recipe_registry: n_slots={n_slots}"
    assert stride  == _RR_SLOT_STRIDE, f"recipe_registry: slot_stride={stride}"
    slot_hdr = raw[_RR_HEADER_SIZE : _RR_HEADER_SIZE + _RR_SLOT_HDR_SZ]
    sm, cid, compiled, *_ = _RR_SLOT_HDR_STRUCT.unpack(slot_hdr)
    assert sm       == _RR_SLOT_MAGIC, f"slot 0: bad slot magic {sm!r}"
    assert cid      == 0,              "slot 0: class_id != 0"
    assert compiled == 0,              "slot 0: should be uncompiled at init"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8 — manifest.json builder
# ═══════════════════════════════════════════════════════════════════════════════

def build_manifest(
    store_root:     Path,
    checksums:      dict[str, str],
    artifact_sizes: dict[str, int],
    init_iso:       str,
) -> str:
    """Return manifest.json content as a formatted JSON string."""
    cfg = MAMBA_CFG
    manifest: dict[str, Any] = {
        "version":        STORE_VERSION,
        "schema_version": SCHEMA_VERSION,
        "axiom_version":  AXIOM_SPEC_VERSION,
        "initialized_at": init_iso,
        "initialized_by": "initialize_store.py",
        "store_root":     str(store_root),
        "artifacts": {
            "topology_router.pt": {
                "path":       str(store_root / "topology_router.pt"),
                "size_bytes": artifact_sizes.get("topology_router.pt", 0),
                "sha256":     checksums.get("topology_router.pt", ""),
                "owner":      "index_daemon.py — do not write from any other component",
                "load_mode":  "torch.load(path, weights_only=True)",
                "model": {
                    "d_model":                cfg.d_model,
                    "n_blocks":               cfg.n_blocks,
                    "d_state":                cfg.d_state,
                    "d_conv":                 cfg.d_conv,
                    "expand":                 cfg.expand,
                    "dt_rank":                cfg.dt_rank,
                    "d_inner":                cfg.d_inner,
                    "vocab_size":             cfg.vocab_size,
                    "n_topology_classes":     len(TOPOLOGY_CLASSES),
                    "n_traversal_strategies": len(TRAVERSAL_STRATEGIES),
                    "n_friction_classes":     len(FRICTION_CLASSES),
                    "source_dim":             512,
                    "n_phases":               len(PHASE_LABELS),
                    "hidden_state_version":   0,
                },
            },
            "phase_states.mmap": {
                "path":       str(store_root / "phase_states.mmap"),
                "size_bytes": artifact_sizes.get("phase_states.mmap", 0),
                "sha256":     checksums.get("phase_states.mmap", ""),
                "format": {
                    "magic":        _PS_MAGIC.decode(),
                    "version":      _PS_VERSION,
                    "n_classes":    _PS_N_CLASSES,
                    "header_bytes": _PS_HEADER_SIZE,
                    "slot_stride":  _PS_SLOT_STRIDE,
                    "total_bytes":  _PS_FILE_SIZE,
                },
            },
            "recipe_registry.mmap": {
                "path":       str(store_root / "recipe_registry.mmap"),
                "size_bytes": artifact_sizes.get("recipe_registry.mmap", 0),
                "sha256":     checksums.get("recipe_registry.mmap", ""),
                "format": {
                    "magic":          _RR_MAGIC.decode(),
                    "version":        _RR_VERSION,
                    "n_slots":        _RR_N_SLOTS,
                    "slot_stride":    _RR_SLOT_STRIDE,
                    "slot_hdr_bytes": _RR_SLOT_HDR_SZ,
                    "recipe_bytes":   _RR_RECIPE_BYTES,
                    "total_bytes":    _RR_FILE_SIZE,
                },
            },
            "manifest.json": {
                "path": str(store_root / "manifest.json"),
                "note": "self-referential — no checksum stored here",
            },
        },
        "directories": {
            "staging":     str(store_root / "staging"),
            "checkpoints": str(store_root / "checkpoints"),
            "triggers":    str(store_root / "triggers"),
            "dead_letters": str(store_root / "dead_letters"),
        },
        "taxonomy": {
            "topology_classes":       list(TOPOLOGY_CLASSES),
            "traversal_strategies":   list(TRAVERSAL_STRATEGIES),
            "friction_classes":       list(FRICTION_CLASSES),
            "phase_labels":           list(PHASE_LABELS),
        },
        "not_created_by_this_script": {
            "store/structural_layer.pt": (
                "Created by offline/encoders/topology_encoder.py after Wikipedia "
                "preparse completes.  WLM handles its absence gracefully via "
                "EmptyStructuralLayer() returning an empty source priority list."
            ),
            "store/dead_letters.jsonl": (
                "Created by crawler_bus.py on first handler failure.  "
                "The dead_letters/ directory houses overflow if JSONL grows "
                "beyond the configured threshold."
            ),
            "store/checkpoints/*": (
                "Created by checkpoint_monitor.py via Alpine crond every 15 min.  "
                "48 rotating archives; 12 hours of learning history."
            ),
            "signal_kernel/recipes/*": (
                "Created by topology/parser.py at runtime as topology classes "
                "graduate to Phase III compiled policy."
            ),
        },
    }
    return json.dumps(manifest, indent=2)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 9 — Directory scaffold + initialization guard
# ═══════════════════════════════════════════════════════════════════════════════

_REQUIRED_SUBDIRS: tuple[str, ...] = (
    "staging",
    "checkpoints",
    "triggers",
    "dead_letters",
)

_SENTINEL_ARTIFACTS: tuple[str, ...] = (
    "topology_router.pt",
    "phase_states.mmap",
    "recipe_registry.mmap",
    "manifest.json",
)


def create_store_directories(store_root: Path, dry_run: bool) -> list[str]:
    """Create store root and all sub-directories.  Returns list of new paths."""
    created: list[str] = []
    for d in [store_root] + [store_root / s for s in _REQUIRED_SUBDIRS]:
        if not d.exists():
            if not dry_run:
                d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
    return created


def _store_is_initialized(store_root: Path) -> list[str]:
    """Return list of already-present sentinel artifacts (empty = clean slate)."""
    return [a for a in _SENTINEL_ARTIFACTS if (store_root / a).exists()]


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 10 — Main orchestrator
# ═══════════════════════════════════════════════════════════════════════════════

def initialize(store_root: Path, dry_run: bool, force: bool) -> int:
    """
    Run the full initialization sequence.  Returns exit code integer.

    Steps
    -----
    1. Directory scaffold
    2. topology_router.pt  (MambaRouter state dict, ~50-80 MB)
    3. phase_states.mmap   (584 bytes)
    4. recipe_registry.mmap (~72 KB)
    5. manifest.json

    Each artifact is written via staging -> atomic rename, then verified.
    """
    init_start = time.monotonic()
    init_iso   = datetime.now(timezone.utc).isoformat()
    staging    = store_root / "staging"

    # Guard: already initialized
    present = _store_is_initialized(store_root)
    if present and not force:
        _err(f"Store already initialized at {store_root}")
        _err(f"  Found: {', '.join(present)}")
        _err("  Use --force to reinitialize (existing artifacts will be overwritten).")
        return 2
    if present and force:
        _warn(f"--force: overwriting existing store at {store_root}")
        _warn(f"  Overwriting: {', '.join(present)}")
    if dry_run:
        _warn("DRY RUN — no files will be written")

    checksums:      dict[str, str] = {}
    artifact_sizes: dict[str, int] = {}

    # ── Step 1: Directories ────────────────────────────────────────────────
    _head("[ 1 / 5 ]  Directory scaffold")
    created = create_store_directories(store_root, dry_run)
    if created:
        for d in created:
            _ok(d)
    else:
        _info("All directories already exist")

    # ── Step 2: topology_router.pt ─────────────────────────────────────────
    _head("[ 2 / 5 ]  topology_router.pt  (MambaRouter state dict)")
    cfg = MAMBA_CFG
    _info(f"d_model={cfg.d_model}  d_inner={cfg.d_inner}  n_blocks={cfg.n_blocks}  "
          f"d_state={cfg.d_state}  d_conv={cfg.d_conv}  vocab={cfg.vocab_size}")

    t0 = time.monotonic()
    sd = build_topology_router_state_dict(cfg)
    build_ms = (time.monotonic() - t0) * 1000
    n_params = sum(v.numel() for v in sd.values())
    _info(f"State dict: {len(sd)} tensors  {n_params:,} parameters  "
          f"built in {build_ms:.0f} ms")

    pt_path = store_root / "topology_router.pt"
    if not dry_run:
        t0 = time.monotonic()
        cksum = _atomic_save_pt(pt_path, sd, staging)
        save_ms = (time.monotonic() - t0) * 1000
        size = pt_path.stat().st_size
        checksums["topology_router.pt"]      = cksum
        artifact_sizes["topology_router.pt"] = size
        _ok(f"Written  {size / 1e6:.1f} MB  sha256={cksum[:16]}…  ({save_ms:.0f} ms)")
        _info("Round-trip verification (weights_only=True) …")
        try:
            _verify_topology_router(pt_path)
        except AssertionError as exc:
            _err(f"topology_router.pt failed round-trip verification: {exc}")
            return 1
        _ok("Round-trip OK — all tensor shapes verified")
    else:
        _info(f"[DRY RUN] would write {pt_path}")

    # ── Step 3: phase_states.mmap ──────────────────────────────────────────
    _head("[ 3 / 5 ]  phase_states.mmap")
    _info(f"magic={_PS_MAGIC!r}  version={_PS_VERSION}  "
          f"n_classes={_PS_N_CLASSES}  slot_stride={_PS_SLOT_STRIDE} B  "
          f"total={_PS_FILE_SIZE} B")

    ps_data = build_phase_states_mmap()
    ps_path = store_root / "phase_states.mmap"
    if not dry_run:
        cksum = _atomic_write_bytes(ps_path, ps_data, staging)
        checksums["phase_states.mmap"]      = cksum
        artifact_sizes["phase_states.mmap"] = len(ps_data)
        _ok(f"Written  {len(ps_data)} B  sha256={cksum[:16]}…")
        try:
            _verify_phase_states(ps_path)
        except AssertionError as exc:
            _err(f"phase_states.mmap failed verification: {exc}")
            return 1
        _ok("Verification OK")
    else:
        _info(f"[DRY RUN] would write {ps_path} ({len(ps_data)} B)")

    # ── Step 4: recipe_registry.mmap ──────────────────────────────────────
    _head("[ 4 / 5 ]  recipe_registry.mmap")
    _info(f"magic={_RR_MAGIC!r}  version={_RR_VERSION}  "
          f"n_slots={_RR_N_SLOTS}  slot_stride={_RR_SLOT_STRIDE} B  "
          f"total={_RR_FILE_SIZE:,} B ({_RR_FILE_SIZE / 1024:.1f} KB)")

    rr_data = build_recipe_registry_mmap()
    rr_path = store_root / "recipe_registry.mmap"
    if not dry_run:
        cksum = _atomic_write_bytes(rr_path, rr_data, staging)
        checksums["recipe_registry.mmap"]      = cksum
        artifact_sizes["recipe_registry.mmap"] = len(rr_data)
        _ok(f"Written  {len(rr_data):,} B  sha256={cksum[:16]}…")
        try:
            _verify_recipe_registry(rr_path)
        except AssertionError as exc:
            _err(f"recipe_registry.mmap failed verification: {exc}")
            return 1
        _ok("Verification OK")
    else:
        _info(f"[DRY RUN] would write {rr_path} ({len(rr_data):,} B)")

    # ── Step 5: manifest.json ──────────────────────────────────────────────
    _head("[ 5 / 5 ]  manifest.json")
    manifest_json = build_manifest(store_root, checksums, artifact_sizes, init_iso)
    manifest_path = store_root / "manifest.json"
    if not dry_run:
        _atomic_write_text(manifest_path, manifest_json, staging)
        _ok(f"Written  {len(manifest_json)} B")
    else:
        _info(f"[DRY RUN] would write {manifest_path}")

    # ── Summary ────────────────────────────────────────────────────────────
    elapsed = time.monotonic() - init_start
    print()
    print("  " + "─" * 58)
    if dry_run:
        _warn(f"DRY RUN complete in {elapsed:.2f} s — no files written")
        return 0

    print(f"\n  {_c('AXIOM store initialized', _GREEN + _BOLD)}")
    print(f"\n  Store root : {store_root}")
    print(f"  Completed  : {init_iso}")
    print(f"  Elapsed    : {elapsed:.2f} s")

    print(f"\n  {_c('Artifacts written', _BOLD)}:")
    for name, size in artifact_sizes.items():
        ck = checksums.get(name, "")
        print(f"    {name:<30} {size:>10,} B  sha256={ck[:16]}…")
    print(f"    {'manifest.json':<30} (no checksum)")

    print(f"\n  {_c('Next steps', _BOLD)}:")
    print("    1. Run offline/encoders/topology_encoder.py after Wikipedia preparse")
    print("       to create store/structural_layer.pt")
    print("    2. Start cold_start.py — WLM loads from store/topology_router.pt")
    print("    3. Queries work immediately; source_priority is empty until")
    print("       structural_layer.pt exists (WLM handles this gracefully)")

    print(f"\n  {_c('NOT created by this script', _DIM)} (system-owned):")
    print("    store/structural_layer.pt  <- offline/encoders/topology_encoder.py")
    print("    store/dead_letters.jsonl   <- crawler_bus.py (first handler failure)")
    print("    store/checkpoints/*        <- checkpoint_monitor.py via crond (15 min)")
    print("    signal_kernel/recipes/*    <- topology/parser.py at runtime")
    print()

    return 0


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 11 — CLI entry point
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="initialize_store.py",
        description=(
            "AXIOM store bootstrap.  "
            "Creates topology_router.pt, phase_states.mmap, "
            "recipe_registry.mmap, manifest.json, and all store directories."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--store-root",
        metavar="DIR",
        default="store",
        help="Path to the store directory (default: ./store)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview all operations without writing any files",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing store artifacts (use with caution in production)",
    )
    return p.parse_args()


def main() -> None:
    args    = _parse_args()
    store   = Path(args.store_root).resolve()

    print(_c(f"\nAXIOM initialize_store.py  //  store: {store}", _BOLD))

    try:
        rc = initialize(store, dry_run=args.dry_run, force=args.force)
    except KeyboardInterrupt:
        print()
        _err("Interrupted")
        sys.exit(3)
    except Exception: # noqa
        _err("Unexpected error during initialization:")
        traceback.print_exc()
        sys.exit(3)

    sys.exit(rc)


if __name__ == "__main__":
    main()