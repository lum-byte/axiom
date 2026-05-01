"""
tag/runtime_paths.py
====================
Canonical runtime path resolver for AXIOM.

All runtime filesystem paths should pass through this module before they are
used by Python entrypoints.  The resolver keeps config.toml, environment
overrides, WSL paths, native release artifacts, Tor bundles, sockets, logs, and
temporary capture directories in one audited shape.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from tag.config import AxiomConfig, ROOT, load_config


PATH_ENV_OVERRIDES: Dict[str, str] = {
    "store_dir": "AXIOM_STORE_DIR",
    "runtime_root": "AXIOM_RUNTIME_ROOT",
    "tmp_dir": "AXIOM_TMP_DIR",
    "interface_socket": "AXIOM_INTERFACE_SOCKET",
    "release_root": "AXIOM_RELEASE_ROOT",
    "html_snapshot_dir": "AXIOM_HTML_SNAPSHOT_DIR",
    "fetch_staging_path": "AXIOM_FETCH_STAGING_PATH",
    "tor_work_dir": "AXIOM_TOR_WORK_DIR",
    "search_cache_path": "AXIOM_SEARCH_CACHE_PATH",
    "search_cache_index_path": "AXIOM_SEARCH_CACHE_INDEX_PATH",
    "dead_letter_path": "AXIOM_DEAD_LETTER_PATH",
    "bus_event_log_path": "AXIOM_BUS_EVENT_LOG_PATH",
}


@dataclass(frozen=True)
class RuntimePaths:
    root: Path
    store_dir: Path
    runtime_root: Path
    tmp_dir: Path
    interface_socket: Path
    release_root: Path
    compiled_root: Path
    win_binaries: Path
    linux_binaries: Path
    dead_letter_path: Path
    bus_event_log_path: Path
    search_cache_path: Path
    search_cache_index_path: Path
    html_snapshot_dir: Path
    fetch_staging_path: Path
    tor_work_dir: Path
    tor_data_dir: Path
    torrc_path: Path
    tor_bundle_root: Path

    def ensure_base_dirs(self) -> None:
        for path in (
            self.store_dir,
            self.runtime_root,
            self.tmp_dir,
            self.html_snapshot_dir,
            self.fetch_staging_path,
            self.tor_work_dir,
            self.tor_data_dir,
            self.dead_letter_path.parent,
            self.bus_event_log_path.parent,
            self.search_cache_path.parent,
            self.search_cache_index_path.parent,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def apply_environment(self, *, override: bool = False) -> None:
        values = {
            "AXIOM_ROOT": str(self.root),
            "AXIOM_STORE_DIR": str(self.store_dir),
            "AXIOM_RUNTIME_ROOT": str(self.runtime_root),
            "AXIOM_TMP_DIR": str(self.tmp_dir),
            "AXIOM_INTERFACE_SOCKET": str(self.interface_socket),
            "AXIOM_RELEASE_ROOT": str(self.release_root),
            "AXIOM_DEAD_LETTER_PATH": str(self.dead_letter_path),
            "AXIOM_BUS_EVENT_LOG_PATH": str(self.bus_event_log_path),
            "AXIOM_SEARCH_CACHE_PATH": str(self.search_cache_path),
            "AXIOM_SEARCH_CACHE_INDEX_PATH": str(self.search_cache_index_path),
            "AXIOM_HTML_SNAPSHOT_DIR": str(self.html_snapshot_dir),
            "AXIOM_FETCH_STAGING_PATH": str(self.fetch_staging_path),
            "AXIOM_TOR_WORK_DIR": str(self.tor_work_dir),
        }
        for key, value in values.items():
            if override:
                os.environ[key] = value
            else:
                os.environ.setdefault(key, value)
        native_candidates = os.pathsep.join(str(path) for path in self.native_library_candidates())
        if override:
            os.environ["AXIOM_NATIVE_LIBRARY_CANDIDATES"] = native_candidates
        else:
            os.environ.setdefault("AXIOM_NATIVE_LIBRARY_CANDIDATES", native_candidates)

    def native_library_candidates(self, *, system: Optional[str] = None) -> List[Path]:
        system_name = (system or platform.system()).lower()
        candidates: List[Path] = []
        if system_name.startswith("win"):
            candidates.append(self.release_root / "axi.dll")
        else:
            candidates.extend([self.release_root / "axi.so", self.release_root / "axi.dll"])
        candidates.extend(
            [
                self.win_binaries / "axirt.dll",
                self.linux_binaries / "axirt.so",
            ]
        )
        return _dedupe_paths(candidates)

    def tor_executable_candidates(self, *, os_name: Optional[str] = None) -> List[Path]:
        env_path = os.environ.get("AXIOM_TOR_EXE", "").strip()
        candidates: List[Path] = [Path(env_path)] if env_path else []
        current_os = (os_name or os.name).lower()
        if current_os == "nt":
            candidates.extend(
                [
                    self.root / ".axiom_runtime" / "deps" / "tor" / "tor" / "tor.exe",
                    self.root / "runtime_deps" / "tor" / "tor" / "tor.exe",
                    self.root / "tools" / "tor" / "tor.exe",
                ]
            )
        else:
            candidates.extend(
                [
                    self.root / ".axiom_runtime" / "deps" / "tor-linux" / "tor" / "tor",
                    self.root / ".axiom_runtime" / "deps" / "tor" / "tor" / "tor",
                    self.root / "runtime_deps" / "tor" / "tor" / "tor",
                    self.root / "tools" / "tor" / "tor",
                ]
            )
            system_tor = shutil.which("tor")
            if system_tor:
                candidates.append(Path(system_tor))
        return _dedupe_paths(candidates)

    def tor_data_candidates(self, tor_exe: Optional[Path] = None) -> List[Path]:
        candidates: List[Path] = []
        if tor_exe is not None:
            bundle_root = tor_exe.parent.parent
            candidates.extend([bundle_root / "data", bundle_root])
        candidates.extend(
            [
                self.root / ".axiom_runtime" / "deps" / "tor-linux" / "data",
                self.root / ".axiom_runtime" / "deps" / "tor" / "data",
                self.tor_bundle_root / "data",
                self.tor_data_dir,
            ]
        )
        return _dedupe_paths(candidates)

    def to_dict(self) -> Dict[str, str]:
        return {
            "root": str(self.root),
            "store_dir": str(self.store_dir),
            "runtime_root": str(self.runtime_root),
            "tmp_dir": str(self.tmp_dir),
            "interface_socket": str(self.interface_socket),
            "release_root": str(self.release_root),
            "compiled_root": str(self.compiled_root),
            "win_binaries": str(self.win_binaries),
            "linux_binaries": str(self.linux_binaries),
            "dead_letter_path": str(self.dead_letter_path),
            "bus_event_log_path": str(self.bus_event_log_path),
            "search_cache_path": str(self.search_cache_path),
            "search_cache_index_path": str(self.search_cache_index_path),
            "html_snapshot_dir": str(self.html_snapshot_dir),
            "fetch_staging_path": str(self.fetch_staging_path),
            "tor_work_dir": str(self.tor_work_dir),
            "tor_data_dir": str(self.tor_data_dir),
            "torrc_path": str(self.torrc_path),
            "tor_bundle_root": str(self.tor_bundle_root),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class RuntimePathResolver:
    def __init__(self, *, config: Optional[AxiomConfig] = None, root: Path = ROOT) -> None:
        self.config = config or load_config()
        self.root = root.resolve()

    def resolve(self, *, store_dir_override: Optional[Path] = None) -> RuntimePaths:
        runtime_root = self._configured_path("runtime_root", "paths.runtime_root", self.root / ".axiom_runtime")
        store_dir = (
            self._normalize_path(store_dir_override)
            if store_dir_override is not None
            else self._configured_path("store_dir", "runtime.store_dir", self.root / "store")
        )
        tmp_dir = self._configured_path("tmp_dir", "paths.tmp_dir", runtime_root / "tmp")
        interface_socket = self._configured_path("interface_socket", "paths.interface_socket", tmp_dir / "axiom_interface.sock")
        release_root = self._configured_path("release_root", "paths.release_root", self.root / "Releases-x64")
        compiled_root = release_root / "compiled" / "binaries"
        win_binaries = compiled_root / "Winx64"
        linux_binaries = compiled_root / "Linux64"
        dead_letter_path = self._store_scoped_path(
            "dead_letter_path",
            "paths.dead_letter_path",
            store_dir / "dead_letters.jsonl",
            store_dir_override=store_dir_override,
            default_raw="store/dead_letters.jsonl",
        )
        bus_event_log_path = self._store_scoped_path(
            "bus_event_log_path",
            "paths.bus_event_log_path",
            store_dir / "bus_events.log",
            store_dir_override=store_dir_override,
            default_raw="store/bus_events.log",
        )
        search_cache_path = self._store_scoped_path(
            "search_cache_path",
            "search_cache.path",
            store_dir / "search_cache.mmap",
            store_dir_override=store_dir_override,
            default_raw="store/search_cache.mmap",
        )
        search_cache_index_path = self._store_scoped_path(
            "search_cache_index_path",
            "search_cache.index_path",
            store_dir / "search_cache_index.json",
            store_dir_override=store_dir_override,
            default_raw="store/search_cache_index.json",
        )
        html_snapshot_dir = self._configured_path("html_snapshot_dir", "paths.html_snapshot_dir", tmp_dir / "html")
        fetch_staging_path = self._configured_path("fetch_staging_path", "paths.fetch_staging_path", tmp_dir / "fetch_staging")
        tor_work_dir = self._configured_path("tor_work_dir", "paths.tor_work_dir", runtime_root / "tor")
        tor_data_dir = self._path_from_config_or_default("paths.tor_data_dir", tor_work_dir / "data")
        torrc_path = self._path_from_config_or_default("paths.torrc_path", tor_work_dir / "torrc")
        tor_bundle_root = self._path_from_config_or_default("paths.tor_bundle_root", runtime_root / "deps" / "tor")
        return RuntimePaths(
            root=self.root,
            store_dir=store_dir,
            runtime_root=runtime_root,
            tmp_dir=tmp_dir,
            interface_socket=interface_socket,
            release_root=release_root,
            compiled_root=compiled_root,
            win_binaries=win_binaries,
            linux_binaries=linux_binaries,
            dead_letter_path=dead_letter_path,
            bus_event_log_path=bus_event_log_path,
            search_cache_path=search_cache_path,
            search_cache_index_path=search_cache_index_path,
            html_snapshot_dir=html_snapshot_dir,
            fetch_staging_path=fetch_staging_path,
            tor_work_dir=tor_work_dir,
            tor_data_dir=tor_data_dir,
            torrc_path=torrc_path,
            tor_bundle_root=tor_bundle_root,
        )

    def _configured_path(self, key: str, dotted: str, default: Path) -> Path:
        env_name = PATH_ENV_OVERRIDES.get(key, "")
        env_value = os.environ.get(env_name, "").strip() if env_name else ""
        if env_value:
            return self._normalize_path(Path(env_value))
        return self._path_from_config_or_default(dotted, default)

    def _path_from_config_or_default(self, dotted: str, default: Path) -> Path:
        raw = self.config.str(dotted, str(default)).strip()
        return self._normalize_path(Path(raw or str(default)))

    def _store_scoped_path(
        self,
        key: str,
        dotted: str,
        default: Path,
        *,
        store_dir_override: Optional[Path],
        default_raw: str,
    ) -> Path:
        env_name = PATH_ENV_OVERRIDES.get(key, "")
        env_value = os.environ.get(env_name, "").strip() if env_name else ""
        if env_value:
            env_store = os.environ.get("AXIOM_STORE_DIR", "").strip()
            if store_dir_override is not None and env_store and _path_is_relative_to(Path(env_value), Path(env_store)):
                return self._normalize_path(default)
            return self._normalize_path(Path(env_value))
        raw = self.config.str(dotted, default_raw).strip()
        if store_dir_override is not None and raw == default_raw:
            return self._normalize_path(default)
        return self._normalize_path(Path(raw or str(default)))

    def _normalize_path(self, path: Path) -> Path:
        expanded = Path(os.path.expandvars(str(path))).expanduser()
        if "\x00" in str(expanded):
            raise ValueError(f"runtime path contains NUL: {expanded!r}")
        if not expanded.is_absolute():
            expanded = self.root / expanded
        return expanded.resolve(strict=False)


def load_runtime_paths(*, config: Optional[AxiomConfig] = None, store_dir_override: Optional[Path] = None) -> RuntimePaths:
    return RuntimePathResolver(config=config).resolve(store_dir_override=store_dir_override)


def apply_runtime_path_environment(*, config: Optional[AxiomConfig] = None, store_dir_override: Optional[Path] = None) -> RuntimePaths:
    paths = load_runtime_paths(config=config, store_dir_override=store_dir_override)
    paths.apply_environment()
    return paths


def _dedupe_paths(paths: Sequence[Path]) -> List[Path]:
    seen: set[str] = set()
    out: List[Path] = []
    for path in paths:
        normalized = Path(path).expanduser()
        key = str(normalized)
        if key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _path_is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.expanduser().resolve(strict=False).relative_to(base.expanduser().resolve(strict=False))
        return True
    except ValueError:
        return False
