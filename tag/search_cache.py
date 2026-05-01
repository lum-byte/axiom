"""Mmap-backed search answer cache for TAG.

This cache sits above crawler/DIC/VERITAS work.  It stores the completed search
answer envelope, not raw fetched pages, so repeated expanded queries can return
without rerunning anchor acquisition, context injection, or legitimacy checks.
"""

from __future__ import annotations

import copy
import hashlib
import json
import mmap
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from signal_kernel.contracts import InterfaceResponse
from tag.config import AxiomConfig, load_config


CACHE_VERSION = 10


@dataclass(frozen=True)
class SearchCacheEntry:
    key: str
    payload: Dict[str, Any]
    created_unix: float
    source: str


class SearchResultCache:
    def __init__(self, *, store_dir: Path, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.store_dir = Path(store_dir)
        self.enabled = self.config.bool("search_cache.enabled", True)
        self.persist = self.config.bool("search_cache.persist", True)
        self.ttl_seconds = self.config.float("search_cache.ttl_seconds", 86400.0, low=0.0, high=31536000.0)
        self.max_bytes = self.config.int("search_cache.max_bytes", 32 * 1024 * 1024, low=1024 * 1024, high=1024 * 1024 * 1024)
        self.max_item_bytes = self.config.int("search_cache.max_item_bytes", 2 * 1024 * 1024, low=64 * 1024, high=64 * 1024 * 1024)
        self.path = resolve_store_scoped_path(
            self.config.str("search_cache.path", "store/search_cache.mmap"),
            store_dir=self.store_dir,
            fallback=self.store_dir / "search_cache.mmap",
        )
        self.index_path = resolve_store_scoped_path(
            self.config.str("search_cache.index_path", "store/search_cache_index.json"),
            store_dir=self.store_dir,
            fallback=self.store_dir / "search_cache_index.json",
        )
        self._lock = threading.RLock()
        self._hot: Dict[str, SearchCacheEntry] = {}
        self._index: Dict[str, Dict[str, Any]] = {}
        self._next_offset = 0
        self._mmap: Optional[mmap.mmap] = None
        if self.enabled and self.persist:
            self._open()

    def close(self) -> None:
        with self._lock:
            if self._mmap is not None:
                try:
                    self._mmap.flush()
                except ValueError:
                    pass
                self._mmap.close()
                self._mmap = None

    def get(self, key: str) -> Optional[SearchCacheEntry]:
        if not self.enabled:
            return None
        now = time.time()
        with self._lock:
            hot = self._hot.get(key)
            if hot is not None and self._fresh(hot.created_unix, now):
                return SearchCacheEntry(key=key, payload=copy.deepcopy(hot.payload), created_unix=hot.created_unix, source="hot")
            if hot is not None:
                self._hot.pop(key, None)
            if not self.persist or self._mmap is None:
                return None
            meta = self._index.get(key)
            if not meta:
                return None
            created = float(meta.get("created_unix", 0.0) or 0.0)
            if not self._fresh(created, now):
                self._index.pop(key, None)
                self._save_index()
                return None
            try:
                offset = int(meta["offset"])
                length = int(meta["length"])
                digest = str(meta["sha256"])
            except (KeyError, TypeError, ValueError):
                self._index.pop(key, None)
                self._save_index()
                return None
            if offset < 0 or length <= 0 or length > self.max_item_bytes or offset + length > self.max_bytes:
                self._index.pop(key, None)
                self._save_index()
                return None
            raw = self._mmap[offset : offset + length]
            if hashlib.sha256(raw).hexdigest() != digest:
                self._index.pop(key, None)
                self._save_index()
                return None
            try:
                payload = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._index.pop(key, None)
                self._save_index()
                return None
            entry = SearchCacheEntry(key=key, payload=payload, created_unix=created, source="mmap")
            self._hot[key] = entry
            return SearchCacheEntry(key=key, payload=copy.deepcopy(payload), created_unix=created, source="mmap")

    def put_response(self, key: str, response: InterfaceResponse) -> None:
        if not self.enabled or response.status != "ok":
            return
        payload = {
            "version": CACHE_VERSION,
            "status": response.status,
            "message": response.message,
            "data": copy.deepcopy(response.data),
        }
        self.put_payload(key, payload)

    def put_payload(self, key: str, payload: Dict[str, Any]) -> None:
        if not self.enabled:
            return
        created = time.time()
        with self._lock:
            entry = SearchCacheEntry(key=key, payload=copy.deepcopy(payload), created_unix=created, source="hot")
            self._hot[key] = entry
            if not self.persist:
                return
            if self._mmap is None:
                self._open()
            if self._mmap is None:
                return
            raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
            if len(raw) > self.max_item_bytes:
                return
            if self._next_offset + len(raw) > self.max_bytes:
                self._reset_locked()
            offset = self._next_offset
            self._mmap[offset : offset + len(raw)] = raw
            self._mmap.flush()
            self._index[key] = {
                "offset": offset,
                "length": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "created_unix": created,
            }
            self._next_offset = offset + len(raw)
            self._save_index()

    def response_from_entry(self, entry: SearchCacheEntry, *, run_id: str) -> InterfaceResponse:
        payload = copy.deepcopy(entry.payload)
        data = dict(payload.get("data") or {})
        data.setdefault("cache", {})
        if isinstance(data["cache"], dict):
            data["cache"].update(
                {
                    "hit": True,
                    "source": entry.source,
                    "created_unix": entry.created_unix,
                    "age_ms": round((time.time() - entry.created_unix) * 1000.0, 3),
                }
            )
        return InterfaceResponse(
            run_id=run_id,
            status=str(payload.get("status") or "ok"),
            message=str(payload.get("message") or ""),
            data=data,
        )

    def build_key(self, *, query: str, crawl_plan: Optional[Mapping[str, Any]]) -> str:
        plan = canonical_plan(crawl_plan)
        identity = {
            "version": CACHE_VERSION,
            "query": normalize_query(query),
            "plan": plan,
            "runtime_version": self.config.str("runtime.version", "1.0.5"),
            "dic": {
                "target": self.config.int("dic.target_answer_words", 560, low=80, high=1500),
                "min": self.config.int("dic.min_answer_words", 500, low=80, high=1500),
                "max": self.config.int("dic.max_answer_words", 700, low=80, high=1500),
            },
            "mcp": {
                "enabled": self.config.bool("mcp.enabled", True),
                "anchors": self.config.bool("mcp.anchor_process_enabled", True),
                "anchor_always": self.config.bool("mcp.anchor_always", False),
                "max_anchor_results": self.config.int("mcp.max_anchor_results", 8, low=1, high=50),
                "brave_present": bool(os.environ.get(self.config.str("mcp.brave_api_key_env", "BRAVE_SEARCH_API_KEY"), "").strip()),
            },
        }
        raw = json.dumps(identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def _fresh(self, created_unix: float, now: float) -> bool:
        if created_unix <= 0:
            return False
        if self.ttl_seconds <= 0:
            return True
        return now - created_unix <= self.ttl_seconds

    def _open(self) -> None:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        if not self.path.exists() or self.path.stat().st_size != self.max_bytes:
            with self.path.open("wb") as handle:
                handle.truncate(self.max_bytes)
        handle = self.path.open("r+b")
        self._mmap = mmap.mmap(handle.fileno(), self.max_bytes)
        handle.close()
        self._load_index()

    def _load_index(self) -> None:
        try:
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        if not isinstance(raw, dict) or raw.get("version") != CACHE_VERSION:
            self._index = {}
            self._next_offset = 0
            return
        entries = raw.get("entries")
        self._index = dict(entries) if isinstance(entries, dict) else {}
        try:
            self._next_offset = int(raw.get("next_offset", 0))
        except (TypeError, ValueError):
            self._next_offset = 0
        self._next_offset = max(0, min(self.max_bytes, self._next_offset))

    def _save_index(self) -> None:
        payload = {
            "version": CACHE_VERSION,
            "updated_unix": time.time(),
            "max_bytes": self.max_bytes,
            "next_offset": self._next_offset,
            "entries": self._index,
        }
        tmp_path = self.index_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8")
        tmp_path.replace(self.index_path)

    def _reset_locked(self) -> None:
        self._index = {}
        self._next_offset = 0
        if self._mmap is not None:
            self._mmap.seek(0)
            self._mmap.write(b"\x00" * min(self.max_bytes, 1024 * 1024))
            self._mmap.flush()
        self._save_index()


def canonical_plan(plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    if not plan:
        return {}
    return {
        "watermark": str(plan.get("watermark") or ""),
        "intent": str(plan.get("intent") or ""),
        "worker_count": int(plan.get("worker_count") or plan.get("requested_worker_count") or 0),
        "requested_worker_count": int(plan.get("requested_worker_count") or plan.get("worker_count") or 0),
        "target_documents": int(plan.get("target_documents") or 0),
        "max_waves": int(plan.get("max_waves") or plan.get("depth") or 0),
        "depth": int(plan.get("depth") or plan.get("max_waves") or 0),
        "expansion_count": int(plan.get("expansion_count") or 0),
        "seed_domains": sorted(str(domain) for domain in plan.get("seed_domains", []) if str(domain).strip()),
        "constraints": {
            "one_worker_per_site": bool((plan.get("constraints") or {}).get("one_worker_per_site", True))
            if isinstance(plan.get("constraints"), Mapping)
            else True,
        },
    }


def normalize_query(query: str) -> str:
    return " ".join(str(query or "").strip().lower().split())


def resolve_store_scoped_path(raw: str, *, store_dir: Path, fallback: Path) -> Path:
    value = str(raw or "").strip()
    if not value:
        return fallback
    path = Path(value)
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "store":
        return store_dir.joinpath(*parts[1:]) if len(parts) > 1 else store_dir
    return path
