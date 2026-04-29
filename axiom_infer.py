#!/usr/bin/env python3
"""
Standalone AXIOM native inference entrypoint.

This file intentionally talks to the native AXIOM C ABI (`axi.dll` / `axi.so`)
with ctypes.  It does not import the TAG crawler, does not start a server, and
prints the full JSON envelope so raw runtime behavior is easy to inspect.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


AXIOM_INFER_VERSION = "1.0.5"
ROOT = Path(__file__).resolve().parent
MAX_QUERY_CHARS = 8192

try:
    from tag.runtime_paths import load_runtime_paths

    RUNTIME_PATHS = load_runtime_paths()
    RUNTIME_PATHS.apply_environment()
except Exception:
    RUNTIME_PATHS = None  # type: ignore[assignment]

try:
    from tag.crawler.source_config import crawler_limits, seed_domains_for_query
except Exception:
    crawler_limits = None  # type: ignore[assignment]
    seed_domains_for_query = None  # type: ignore[assignment]


def _runtime_limits() -> Dict[str, Any]:
    if crawler_limits is None:
        return {}
    try:
        return dict(crawler_limits())
    except Exception:
        return {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


_LIMITS = _runtime_limits()
MAX_DEPTH = _env_int("AXIOM_INFER_MAX_DEPTH", int(_LIMITS.get("max_waves", 16) or 16))
MAX_WORKERS = _env_int("AXIOM_INFER_MAX_WORKERS", int(_LIMITS.get("absolute_worker_limit", 100) or 100))


class AxiomNativeError(RuntimeError):
    pass


class AxiomNative:
    def __init__(self, library_path: Path, *, store_dir: Path, socket_path: Optional[str] = None) -> None:
        self.library_path = _validate_library_path(library_path)
        self.store_dir = store_dir
        self.socket_path = socket_path or _default_socket_path()
        self._lib = ctypes.CDLL(str(self.library_path))
        self._configure_abi()
        config = json.dumps(
            {
                "store_dir": str(self.store_dir),
                "socket_path": self.socket_path,
            },
            separators=(",", ":"),
        ).encode("utf-8")
        self._runtime = self._lib.axiom_init(config)
        if not self._runtime:
            raise AxiomNativeError("axiom_init returned null")

    def _configure_abi(self) -> None:
        self._lib.axiom_version.argtypes = []
        self._lib.axiom_version.restype = ctypes.c_char_p
        self._lib.axiom_init.argtypes = [ctypes.c_char_p]
        self._lib.axiom_init.restype = ctypes.c_void_p
        self._lib.axiom_handle_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self._lib.axiom_handle_json.restype = ctypes.c_void_p
        self._lib.axiom_free.argtypes = [ctypes.c_void_p]
        self._lib.axiom_free.restype = None
        self._lib.axiom_shutdown.argtypes = [ctypes.c_void_p]
        self._lib.axiom_shutdown.restype = None

    @property
    def native_version(self) -> str:
        raw = self._lib.axiom_version()
        return raw.decode("utf-8", errors="replace") if raw else "unknown"

    def call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        payload = json.dumps(request, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        ptr = self._lib.axiom_handle_json(self._runtime, payload)
        if not ptr:
            raise AxiomNativeError("axiom_handle_json returned null")
        try:
            raw = ctypes.string_at(ptr).decode("utf-8", errors="replace")
        finally:
            self._lib.axiom_free(ptr)
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AxiomNativeError(f"native runtime returned invalid JSON: {exc}") from exc
        return {"raw": raw, "json": parsed}

    def close(self) -> None:
        runtime = getattr(self, "_runtime", None)
        if runtime:
            self._lib.axiom_shutdown(runtime)
            self._runtime = None

    def __enter__(self) -> "AxiomNative":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        del exc_type, exc, tb
        self.close()


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        query = _resolve_query(args)
        depth = _bounded_int(args.depth, 1, 1, MAX_DEPTH, "depth")
        workers = _bounded_int(args.workers, 10, 1, MAX_WORKERS, "workers")
        library = Path(args.lib).expanduser() if args.lib else find_native_library()
        store_dir = Path(args.store_dir).expanduser().resolve()
        request = build_request(args.command, query, workers=workers, depth=depth)
        output = execute_once(
            library=library,
            store_dir=store_dir,
            request=request,
            query=query,
            workers=workers,
            depth=depth,
            auto_seed=not args.no_auto_seed,
            seed_domains=args.seed,
            cycles=args.cycles,
            summary_only=args.summary_only,
        )
        json.dump(output, sys.stdout, indent=None if args.compact else 2, sort_keys=False)
        sys.stdout.write("\n")
        return 0
    except Exception as exc:
        error = {
            "axiom_infer_version": AXIOM_INFER_VERSION,
            "ok": False,
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }
        json.dump(error, sys.stderr, indent=2)
        sys.stderr.write("\n")
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one AXIOM native inference and print full JSON.")
    parser.add_argument("query_parts", nargs="*", help="Query text when --query is not supplied.")
    parser.add_argument("--query", "-q", help="Query text.")
    parser.add_argument("--search", action="store_const", dest="command", const="search", help="Run search (default).")
    parser.add_argument("--status", action="store_const", dest="command", const="status", help="Run status instead of search.")
    parser.add_argument("--learn", action="store_const", dest="command", const="learn", help="Learn the query as a domain.")
    parser.add_argument("--fetch", action="store_const", dest="command", const="fetch", help="Fetch the query as a URL.")
    parser.add_argument("--workers", "-w", type=int, default=10, help="Requested swarm workers, clamped to 1..100.")
    parser.add_argument("--depth", "-d", type=int, default=1, help="Traversal depth / waves, clamped to 1..8.")
    parser.add_argument("--seed", action="append", default=[], help="Domain to learn before search. Can be repeated.")
    parser.add_argument("--no-auto-seed", action="store_true", help="Do not auto-learn broad source domains before search.")
    parser.add_argument("--store-dir", default=str(_default_native_store_dir()), help="Native store directory.")
    parser.add_argument("--lib", help="Explicit path to axi.dll, axi.so, axirt.dll, or axirt.so.")
    parser.add_argument("--cycles", type=int, default=1, help="Repeat the native request N times in one process.")
    parser.add_argument("--summary-only", action="store_true", help="For cycles > 1, emit aggregate stats instead of every response.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    parser.set_defaults(command="search")
    return parser


def execute_once(
    *,
    library: Path,
    store_dir: Path,
    request: Dict[str, Any],
    query: str,
    workers: int,
    depth: int,
    auto_seed: bool,
    seed_domains: Iterable[str],
    cycles: int,
    summary_only: bool,
) -> Dict[str, Any]:
    cycles = _bounded_int(cycles, 1, 1, 1_000_000, "cycles")
    with AxiomNative(library, store_dir=store_dir) as native:
        bootstrap = bootstrap_sources(native, query, auto_seed=auto_seed, extra_domains=seed_domains)
        responses: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {}
        last: Optional[Dict[str, Any]] = None
        for _ in range(cycles):
            current = native.call({**request, "run_id": str(uuid.uuid4())})
            parsed = current["json"]
            status = str(parsed.get("status", "unknown"))
            status_counts[status] = status_counts.get(status, 0) + 1
            last = current
            if not summary_only:
                responses.append(current)
        return {
            "axiom_infer_version": AXIOM_INFER_VERSION,
            "ok": all(status not in {"error", "unknown"} for status in status_counts),
            "library": str(native.library_path),
            "native_version": native.native_version,
            "runtime_paths": _runtime_path_summary(),
            "request": {
                "command": request["command"],
                "payload": request["payload"],
                "query": query,
                "workers": workers,
                "depth": depth,
                "cycles": cycles,
            },
            "bootstrap": bootstrap,
            "cycles": {
                "requested": cycles,
                "status_counts": status_counts,
            },
            "responses": responses if not summary_only else [],
            "last_response": last,
        }


def bootstrap_sources(
    native: AxiomNative,
    query: str,
    *,
    auto_seed: bool,
    extra_domains: Iterable[str],
) -> Dict[str, Any]:
    domains: List[str] = []
    if auto_seed:
        domains.extend(select_seed_domains(query))
    domains.extend(extra_domains)
    unique: List[str] = []
    seen = set()
    for domain in domains:
        normalized = normalize_domain(domain)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    learned = []
    for domain in unique:
        learned.append(native.call({"command": "learn", "payload": domain, "run_id": str(uuid.uuid4())})["json"])
    return {
        "auto_seed": auto_seed,
        "domains": unique,
        "learn_responses": learned,
    }


def build_request(command: str, query: str, *, workers: int, depth: int) -> Dict[str, Any]:
    if command == "status":
        return {"command": "status", "payload": "", "run_id": str(uuid.uuid4())}
    if command == "learn":
        return {"command": "learn", "payload": query, "run_id": str(uuid.uuid4())}
    if command == "fetch":
        return {"command": "fetch", "payload": query, "run_id": str(uuid.uuid4())}
    payload = f"swarm -{workers} | depth -{depth} | {query}"
    return {"command": "search", "payload": payload, "run_id": str(uuid.uuid4())}


def find_native_library() -> Path:
    candidates = _env_native_library_candidates()
    if RUNTIME_PATHS is not None:
        candidates.extend(RUNTIME_PATHS.native_library_candidates(system=platform.system()))
    else:
        suffixes = [".dll"] if platform.system().lower().startswith("win") else [".so", ".dll"]
        candidates.extend(ROOT / "Releases-x64" / f"axi{suffix}" for suffix in suffixes)
        candidates.extend(ROOT.glob("Releases-x64/compiled/binaries/*/axirt.*"))
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise AxiomNativeError("native AXIOM library not found; run ./axicomp.sh runtime-linux first")


def _env_native_library_candidates() -> List[Path]:
    raw = os.environ.get("AXIOM_NATIVE_LIBRARY_CANDIDATES", "")
    if not raw.strip():
        return []
    return [Path(part) for part in raw.split(os.pathsep) if part.strip()]


def _default_socket_path() -> str:
    if RUNTIME_PATHS is not None:
        return str(RUNTIME_PATHS.interface_socket)
    return str(ROOT / ".axiom_runtime" / "tmp" / "axiom_interface.sock")


def _default_native_store_dir() -> Path:
    if RUNTIME_PATHS is not None:
        return RUNTIME_PATHS.runtime_root / "native-infer-store"
    return ROOT / ".axiom_runtime" / "native-infer-store"


def _runtime_path_summary() -> Dict[str, str]:
    if RUNTIME_PATHS is None:
        return {"root": str(ROOT)}
    return RUNTIME_PATHS.to_dict()


def _validate_library_path(path: Path) -> Path:
    resolved = path.resolve()
    if "\x00" in str(resolved):
        raise AxiomNativeError("library path contains NUL")
    if resolved.suffix.lower() not in {".dll", ".so", ".dylib"}:
        raise AxiomNativeError(f"unsupported native library extension: {resolved.suffix}")
    if not resolved.is_file():
        raise AxiomNativeError(f"native library does not exist: {resolved}")
    return resolved


def _resolve_query(args: argparse.Namespace) -> str:
    query = args.query if args.query is not None else " ".join(args.query_parts)
    query = (query or "").strip()
    if args.command == "status" and not query:
        return ""
    if not query:
        raise ValueError("query is required")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"query too long: {len(query)} > {MAX_QUERY_CHARS}")
    if "\x00" in query:
        raise ValueError("query contains NUL")
    return query


def select_seed_domains(query: str) -> List[str]:
    if seed_domains_for_query is not None:
        try:
            domains = seed_domains_for_query(query)
            if domains:
                return domains
        except Exception:
            pass
    return []


def normalize_domain(raw: str) -> str:
    value = raw.strip().lower()
    if not value:
        return ""
    if "://" in value:
        value = value.split("://", 1)[1]
    value = value.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0].strip(".")
    allowed = set("abcdefghijklmnopqrstuvwxyz0123456789-.")
    if not value or "." not in value or any(ch not in allowed for ch in value):
        return ""
    return value


def _bounded_int(value: int, default: int, low: int, high: int, name: str) -> int:
    if value is None:
        return default
    if value < low or value > high:
        raise ValueError(f"{name} must be between {low} and {high}")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
