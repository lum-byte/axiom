"""
tag/cold_start.py
=================
Startup orchestrator for AXIOM.

Cold start validates local prerequisites, prepares the durable store, enforces
production HMAC rules, and reports degraded-but-usable development state.  The
store path understands both the compact initialize_store.py binary formats and
the expanded runtime mmap files owned by index_daemon.py.
"""

from __future__ import annotations

import hashlib
import hmac as hmac_lib
import json
import logging
import os
import platform
import shutil
import signal
import struct
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signal_kernel.contracts import SystemStatus, new_run_id
from tag.config import AxiomConfig, load_config
from tag.runtime_paths import RuntimePathResolver


logger = logging.getLogger(__name__)

PHASE_SLOT_BYTES = 32
PHASE_RUNTIME_SLOTS = 4096
PHASE_HEADER = struct.Struct("<4sBBH")
PHASE_MAGIC = b"AXPS"
RECIPE_MAGIC = b"AXRR"
STORE_SUBDIRS = ("staging", "checkpoints", "triggers", "dead_letters", "offline_queue")

STORE_FILES = {
    "topology_router.pt": 1024,
    "recipe_registry.mmap": 16,
    "phase_states.mmap": PHASE_HEADER.size + (18 * PHASE_SLOT_BYTES),
    "structural_layer.pt": 1024,
}

REQUIRED_TOOLS = ("gcc",)
OPTIONAL_TOOLS = ("nvcc", "go", "cargo", "tor", "chromium", "chrome", "node", "npm", "bun")
PRODUCTION_MODES = {"prod", "production", "release"}


@dataclass(frozen=True)
class ColdStartResult:
    ok: bool
    degraded: bool
    store_dir: str
    warnings: List[str]
    errors: List[str]
    platform_name: str = ""
    tool_status: List[Dict[str, object]] = field(default_factory=list)
    store_status: List[Dict[str, object]] = field(default_factory=list)
    started_unix: int = 0
    mode: str = "dev"
    config_path: str = ""
    startup_duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class ToolCheck:
    name: str
    required: bool
    found: bool
    path: Optional[str]

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class StoreFileCheck:
    name: str
    path: str
    required_size: int
    exists: bool
    size: int
    crc32: int
    repaired: bool
    ok: bool
    binary_format_valid: bool = True
    format_detail: str = ""

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class PlatformCheck:
    system: str
    release: str
    production_ready: bool
    degraded_features: List[str]

    def to_dict(self) -> Dict[str, object]:
        return self.__dict__.copy()


class ColdStart:
    def __init__(self, *, store_dir: Path = Path("store"), config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.path_resolver = RuntimePathResolver(config=self.config)
        store_override = None if Path(store_dir) == Path("store") else Path(store_dir)
        self.store_dir = self.path_resolver.resolve(store_dir_override=store_override).store_dir
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.tool_status: List[ToolCheck] = []
        self.store_status: List[StoreFileCheck] = []
        self.platform_status: Optional[PlatformCheck] = None
        self.mode = self._detect_mode()

    def run(self) -> ColdStartResult:
        started = int(time.time())
        t0 = time.perf_counter()
        self.warnings = []
        self.errors = []
        self.tool_status = []
        self.store_status = []
        self.mode = self._detect_mode()

        if self.config.errors:
            self.warnings.extend(f"config load warning: {error}" for error in self.config.errors)

        self._check_hmac()
        self.platform_status = self._check_platform()
        self.tool_status = self._check_tools()
        self.store_status = self._ensure_store()
        duration_ms = (time.perf_counter() - t0) * 1000.0
        return ColdStartResult(
            ok=len(self.errors) == 0,
            degraded=len(self.warnings) > 0,
            store_dir=str(self.store_dir),
            warnings=self.warnings,
            errors=self.errors,
            platform_name=self.platform_status.system if self.platform_status else platform.system(),
            tool_status=[item.to_dict() for item in self.tool_status],
            store_status=[item.to_dict() for item in self.store_status],
            started_unix=started,
            mode=self.mode,
            config_path=str(self.config.path),
            startup_duration_ms=round(duration_ms, 3),
        )

    def status_contract(self) -> SystemStatus:
        result = self.run()
        return SystemStatus(
            run_id=str(new_run_id()),
            bus_started=False,
            bus_mode="unstarted",
            store_ready=result.ok,
            index_daemon_ready=False,
            cold_start_complete=result.ok,
            learned_domains=0,
            queued_work_items=0,
        )

    def _detect_mode(self) -> str:
        env_mode = os.environ.get("AXIOM_ENV", "").strip().lower()
        if env_mode:
            return env_mode
        configured = self.config.str("runtime.env", "dev").strip().lower()
        return configured or "dev"

    def _check_hmac(self) -> None:
        key = os.environ.get("AXIOM_BUS_HMAC_KEY", "")
        key_bytes = key.encode("utf-8")
        min_bytes = self.config.int("bus.hmac_min_bytes", 32, low=16, high=4096)
        all_zero = bool(key_bytes) and hmac_lib.compare_digest(key_bytes, b"0" * len(key_bytes))

        if key and len(key_bytes) >= min_bytes and not all_zero:
            return

        if self.mode in PRODUCTION_MODES:
            if not key:
                self.errors.append("AXIOM_BUS_HMAC_KEY must be set before production bus startup.")
            elif len(key_bytes) < min_bytes:
                self.errors.append(f"AXIOM_BUS_HMAC_KEY must be at least {min_bytes} bytes in production.")
            else:
                self.errors.append("AXIOM_BUS_HMAC_KEY cannot be all-zero or trivially guessable in production.")
            return

        if not self.config.bool("bus.auto_dev_hmac", True):
            self.warnings.append("dev HMAC auto-generation disabled; crawler bus startup may fail without AXIOM_BUS_HMAC_KEY.")
            return

        seed = f"axiom-dev:{self.store_dir.resolve()}".encode("utf-8")
        os.environ["AXIOM_BUS_HMAC_KEY"] = hashlib.sha256(seed).hexdigest()
        self.warnings.append("AXIOM_BUS_HMAC_KEY auto-generated for dev mode.")

    def _check_platform(self) -> PlatformCheck:
        system = platform.system()
        degraded: List[str] = []
        production_ready = system.lower() == "linux"
        if system.lower() == "windows":
            degraded.extend(["gvisor", "iptables", "inotify", "unix_signals", "unix_sockets"])
            self.warnings.append("Windows dev mode: gVisor, iptables, inotify, and Unix-signal checks are degraded.")
        elif system.lower() != "linux":
            degraded.extend(["linux_daemons", "iptables"])
            self.warnings.append(f"{system} dev mode: Linux daemon checks are degraded.")
        return PlatformCheck(system=system, release=platform.release(), production_ready=production_ready, degraded_features=degraded)

    def _check_tools(self) -> List[ToolCheck]:
        checks: List[ToolCheck] = []
        for tool in REQUIRED_TOOLS:
            found = shutil.which(tool)
            checks.append(ToolCheck(name=tool, required=True, found=found is not None, path=found))
            if found is None:
                self.errors.append(f"{tool} not found on PATH.")
        for optional in OPTIONAL_TOOLS:
            found = shutil.which(optional)
            checks.append(ToolCheck(name=optional, required=False, found=found is not None, path=found))
            if found is None:
                self.warnings.append(f"{optional} not found on PATH; related targets will be skipped or degraded.")
        return checks

    def _ensure_store(self) -> List[StoreFileCheck]:
        self.store_dir.mkdir(parents=True, exist_ok=True)
        for dirname in STORE_SUBDIRS:
            (self.store_dir / dirname).mkdir(parents=True, exist_ok=True)

        checks: List[StoreFileCheck] = []
        for name, min_size in STORE_FILES.items():
            path = self.store_dir / name
            repaired = False
            if self._needs_repair(path, min_size):
                payload = self._initial_store_payload(name, min_size)
                self._atomic_write(path, payload)
                repaired = True
            checks.append(self._store_file_check(name, path, min_size, repaired))
        return checks

    def _needs_repair(self, path: Path, min_size: int) -> bool:
        if not path.exists():
            return True
        try:
            return path.stat().st_size < min_size
        except OSError:
            return True

    def _initial_store_payload(self, name: str, min_size: int) -> bytes:
        if self.config.bool("cold_start.initialize_store_formats", True):
            built = self._payload_from_initialize_store(name)
            if built is not None:
                if name == "phase_states.mmap":
                    target = PHASE_HEADER.size + PHASE_RUNTIME_SLOTS * PHASE_SLOT_BYTES
                    return built + b"\x00" * max(0, target - len(built))
                return built
        return b"\x00" * max(min_size, 1024)

    def _payload_from_initialize_store(self, name: str) -> Optional[bytes]:
        try:
            from tag import initialize_store
        except Exception as exc:
            self.warnings.append(f"initialize_store import unavailable for {name}: {type(exc).__name__}: {exc}")
            return None

        builders = {
            "phase_states.mmap": "build_phase_states_mmap",
            "recipe_registry.mmap": "build_recipe_registry_mmap",
        }
        builder_name = builders.get(name)
        if builder_name is None:
            return None
        builder = getattr(initialize_store, builder_name, None)
        if builder is None:
            return None
        try:
            payload = builder()
        except Exception as exc:
            self.warnings.append(f"initialize_store.{builder_name} failed for {name}: {type(exc).__name__}: {exc}")
            return None
        return bytes(payload)

    def _store_file_check(self, name: str, path: Path, required_size: int, repaired: bool) -> StoreFileCheck:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        crc = self._file_header_crc(path) if exists else 0
        binary_ok, detail = self.validate_store_binary_format(
            path,
            name,
            allow_zero_initialized=self._allow_zero_initialized_store(),
        )
        ok = exists and size >= required_size and binary_ok
        if not ok:
            self.errors.append(f"store file {name} failed cold-start validation: {detail or 'missing or undersized'}.")
        return StoreFileCheck(
            name=name,
            path=str(path),
            required_size=required_size,
            exists=exists,
            size=size,
            crc32=crc,
            repaired=repaired,
            ok=ok,
            binary_format_valid=binary_ok,
            format_detail=detail,
        )

    def _allow_zero_initialized_store(self) -> bool:
        return self.mode not in PRODUCTION_MODES and self.config.bool("cold_start.allow_zero_initialized_dev_store", True)

    @staticmethod
    def validate_store_binary_format(path: Path, name: str, *, allow_zero_initialized: bool = False) -> Tuple[bool, str]:
        if name.endswith(".pt"):
            try:
                size = path.stat().st_size
                with path.open("rb") as handle:
                    head = handle.read(4096)
            except OSError as exc:
                return False, f"unreadable: {type(exc).__name__}: {exc}"
            if size == 0:
                return allow_zero_initialized, "empty dev placeholder" if allow_zero_initialized else "empty file"
            if head.startswith(b"PK\x03\x04"):
                return True, "torch zip archive"
            if head.startswith(b"\x80"):
                return True, "pickle/pytorch payload"
            if _all_zero(head):
                return allow_zero_initialized, "dev placeholder .pt" if allow_zero_initialized else "unrecognized .pt payload"
            return allow_zero_initialized, "dev placeholder .pt" if allow_zero_initialized else "unrecognized .pt payload"

        try:
            raw = path.read_bytes()
        except OSError as exc:
            return False, f"unreadable: {type(exc).__name__}: {exc}"
        if not raw:
            return allow_zero_initialized, "empty dev placeholder" if allow_zero_initialized else "empty file"

        if _all_zero(raw):
            return allow_zero_initialized, "zero-initialized dev placeholder" if allow_zero_initialized else "all-zero payload"

        if name == "phase_states.mmap":
            if raw.startswith(PHASE_MAGIC):
                if len(raw) < PHASE_HEADER.size:
                    return False, "AXPS header truncated"
                magic, version, n_classes, _reserved = PHASE_HEADER.unpack(raw[: PHASE_HEADER.size])
                if magic != PHASE_MAGIC or version != 1:
                    return False, "invalid AXPS header"
                if n_classes <= 0:
                    return False, "AXPS n_classes is zero"
                return True, f"AXPS v{version} classes={n_classes}"
            if len(raw) >= PHASE_SLOT_BYTES and len(raw) % PHASE_SLOT_BYTES == 0:
                return True, "legacy flat phase mmap"
            return False, "phase mmap is neither AXPS nor legacy flat slots"

        if name == "recipe_registry.mmap":
            stripped = raw.rstrip(b"\x00").strip()
            if raw.startswith(RECIPE_MAGIC):
                return True, "AXRR recipe registry"
            if stripped.startswith((b"{", b"[")):
                try:
                    json.loads(stripped.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    return False, f"recipe JSON invalid: {exc}"
                return True, "JSON recipe registry"
            return False, "recipe registry is neither AXRR nor JSON"

        return True, "untyped store file"

    @staticmethod
    def _atomic_write(path: Path, payload: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        staging_dir = path.parent / "staging"
        staging_dir.mkdir(parents=True, exist_ok=True)
        tmp = staging_dir / f"{path.name}.{os.getpid()}.{int(time.time() * 1000)}.tmp"
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            dir_fd = os.open(str(path.parent), os.O_DIRECTORY)
        except (OSError, AttributeError):
            return
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)

    @staticmethod
    def _file_header_crc(path: Path, limit: int = 4096) -> int:
        try:
            with path.open("rb") as f:
                return zlib.crc32(f.read(limit)) & 0xFFFFFFFF
        except OSError:
            return 0

    @staticmethod
    def summarize_tools(checks: Iterable[ToolCheck]) -> Dict[str, int]:
        summary = {"required_missing": 0, "optional_missing": 0, "found": 0}
        for check in checks:
            if check.found:
                summary["found"] += 1
            elif check.required:
                summary["required_missing"] += 1
            else:
                summary["optional_missing"] += 1
        return summary


def _all_zero(raw: bytes, *, sample_limit: int = 1024 * 1024) -> bool:
    sample = raw[:sample_limit]
    return hmac_lib.compare_digest(sample, b"\x00" * len(sample)) and (
        len(raw) <= sample_limit or not any(raw[sample_limit:])
    )


def _handle_sigusr1(signum: int, frame: object) -> None:
    del signum, frame
    ColdStart().run()


if hasattr(signal, "SIGUSR1"):
    signal.signal(signal.SIGUSR1, _handle_sigusr1)


def main() -> int:
    result = ColdStart().run()
    print(json.dumps(result.to_dict(), sort_keys=True))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
