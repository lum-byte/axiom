"""
tag/config.py
=============
Small TOML-backed runtime configuration loader for AXIOM.

The loader is intentionally dependency-free: Python 3.11+ ships tomllib, and
all values have conservative defaults so tests and dev shells can run even when
config.toml has not been created yet.
"""

from __future__ import annotations

import copy
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "config.toml"
CONFIG_ENV = "AXIOM_CONFIG_TOML"

DEFAULT_CONFIG: Dict[str, Any] = {
    "runtime": {
        "version": "1.0.5",
        "env": "dev",
        "store_dir": "store",
        "autostart": False,
        "status_warmup": True,
        "status_warmup_timeout_seconds": 5.0,
        "status_start_watchdog": False,
    },
    "paths": {
        "runtime_root": ".axiom_runtime",
        "tmp_dir": ".axiom_runtime/tmp",
        "interface_socket": ".axiom_runtime/tmp/axiom_interface.sock",
        "release_root": "Releases-x64",
        "dead_letter_path": "store/dead_letters.jsonl",
        "bus_event_log_path": "store/bus_events.log",
        "html_snapshot_dir": ".axiom_runtime/tmp/html",
        "fetch_staging_path": ".axiom_runtime/tmp/fetch_staging",
        "tor_work_dir": ".axiom_runtime/tor",
        "tor_data_dir": ".axiom_runtime/tor/data",
        "torrc_path": ".axiom_runtime/tor/torrc",
        "tor_bundle_root": ".axiom_runtime/deps/tor",
    },
    "bus": {
        "hmac_min_bytes": 32,
        "auto_dev_hmac": True,
    },
    "cold_start": {
        "initialize_store_formats": True,
        "allow_zero_initialized_dev_store": True,
        "start_daemons": False,
        "self_test": False,
    },
    "index_daemon": {
        "gradient_batch_size": 32,
        "gradient_flush_interval_seconds": 60.0,
        "gradient_item_ttl_seconds": 600.0,
        "phase_scan_interval_seconds": 30.0,
        "health_log_interval_seconds": 120.0,
        "recipe_save_interval_seconds": 30.0,
        "gradient_purge_interval_seconds": 300.0,
        "queue_max_size": 1000,
    },
    "watchdog": {
        "enabled": True,
        "start": True,
        "debounce_ms": {
            "topology_router_pt": 500,
            "structural_layer_pt": 500,
            "recipe_registry_mmap": 100,
            "phase_states_mmap": 100,
        },
    },
    "crawler": {
        "source_config": "config/crawler_sources.json",
        "max_workers": 10,
        "worker_ceiling": 10,
    },
}


@dataclass(frozen=True)
class AxiomConfig:
    data: Mapping[str, Any]
    path: Path
    errors: tuple[str, ...] = ()

    def section(self, name: str) -> Dict[str, Any]:
        value = self.data.get(name, {})
        return dict(value) if isinstance(value, Mapping) else {}

    def get(self, dotted: str, default: Any = None) -> Any:
        current: Any = self.data
        for part in dotted.split("."):
            if not isinstance(current, Mapping) or part not in current:
                return default
            current = current[part]
        return current

    def bool(self, dotted: str, default: bool = False) -> bool:
        value = self.get(dotted, default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        return default

    def int(self, dotted: str, default: int = 0, *, low: Optional[int] = None, high: Optional[int] = None) -> int:
        try:
            value = int(self.get(dotted, default))
        except (TypeError, ValueError):
            value = default
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        return value

    def float(self, dotted: str, default: float = 0.0, *, low: Optional[float] = None, high: Optional[float] = None) -> float:
        try:
            value = float(self.get(dotted, default))
        except (TypeError, ValueError):
            value = default
        if low is not None:
            value = max(low, value)
        if high is not None:
            value = min(high, value)
        return value

    def str(self, dotted: str, default: str = "") -> str:
        value = self.get(dotted, default)
        return str(value) if value is not None else default

    def path_value(self, dotted: str, default: Path) -> Path:
        raw = self.str(dotted, str(default)).strip()
        path = Path(raw or str(default))
        return path if path.is_absolute() else ROOT / path


def load_config(path: Optional[Path] = None) -> AxiomConfig:
    config_path = path or Path(os.environ.get(CONFIG_ENV, "") or DEFAULT_CONFIG_PATH)
    merged: Dict[str, Any] = copy.deepcopy(DEFAULT_CONFIG)
    errors: list[str] = []
    if config_path.exists():
        try:
            with config_path.open("rb") as handle:
                payload = tomllib.load(handle)
            _deep_merge(merged, payload)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return AxiomConfig(data=merged, path=config_path, errors=tuple(errors))


def apply_environment_defaults(config: AxiomConfig) -> None:
    """Seed environment variables expected by older modules without overriding callers."""

    defaults = {
        "AXIOM_ENV": config.str("runtime.env", "dev"),
        "AXIOM_VERSION": config.str("runtime.version", "1.0.5"),
        "AXIOM_STORE_DIR": str(config.path_value("runtime.store_dir", ROOT / "store")),
        "AXIOM_RUNTIME_ROOT": str(config.path_value("paths.runtime_root", ROOT / ".axiom_runtime")),
        "AXIOM_TMP_DIR": str(config.path_value("paths.tmp_dir", ROOT / ".axiom_runtime" / "tmp")),
        "AXIOM_INTERFACE_SOCKET": str(config.path_value("paths.interface_socket", ROOT / ".axiom_runtime" / "tmp" / "axiom_interface.sock")),
        "AXIOM_RELEASE_ROOT": str(config.path_value("paths.release_root", ROOT / "Releases-x64")),
        "AXIOM_CRAWLER_SOURCE_CONFIG": str(config.path_value("crawler.source_config", ROOT / "config" / "crawler_sources.json")),
        "AXIOM_CRAWL_MAX_WORKERS": str(config.int("crawler.worker_ceiling", 10, low=1, high=100)),
        "AXIOM_CRAWL_WORKERS": str(config.int("crawler.max_workers", 10, low=1, high=100)),
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def _deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> None:
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            _deep_merge(base[key], value)  # type: ignore[index]
        else:
            base[key] = copy.deepcopy(value)
