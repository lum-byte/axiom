"""
tag/cold_start.py
=================
Startup orchestrator for AXIOM.

This module validates local prerequisites, creates store files, verifies bus
configuration, and prepares the interface socket. Production Linux paths are
strict; Windows development paths use explicit shims and report degraded status.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import signal
import sys
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from signal_kernel.contracts import SystemStatus, new_run_id


STORE_FILES = {
    "topology_router.pt": 1024,
    "recipe_registry.mmap": 1024 * 1024,
    "phase_states.mmap": 4096 * 32,
    "structural_layer.pt": 1024,
}

REQUIRED_TOOLS = ("gcc",)
OPTIONAL_TOOLS = ("nvcc", "go", "cargo", "tor", "chromium", "chrome", "node", "npm", "bun")


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
    def __init__(self, *, store_dir: Path = Path("store")) -> None:
        self.store_dir = store_dir
        self.warnings: List[str] = []
        self.errors: List[str] = []
        self.tool_status: List[ToolCheck] = []
        self.store_status: List[StoreFileCheck] = []
        self.platform_status: Optional[PlatformCheck] = None

    def run(self) -> ColdStartResult:
        started = int(time.time())
        self._check_hmac()
        self.platform_status = self._check_platform()
        self.tool_status = self._check_tools()
        self.store_status = self._ensure_store()
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

    def _check_hmac(self) -> None:
        key = os.environ.get("AXIOM_BUS_HMAC_KEY")
        if not key or len(key.encode("utf-8")) < 32:
            self.errors.append("AXIOM_BUS_HMAC_KEY must be set to at least 32 bytes before bus startup.")

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
        checks: List[StoreFileCheck] = []
        for name, size in STORE_FILES.items():
            path = self.store_dir / name
            repaired = False
            if not path.exists() or path.stat().st_size < size:
                with path.open("wb") as f:
                    f.truncate(size)
                repaired = True
            checks.append(self._store_file_check(name, path, size, repaired))
        return checks

    def _store_file_check(self, name: str, path: Path, required_size: int, repaired: bool) -> StoreFileCheck:
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        crc = self._file_header_crc(path) if exists else 0
        ok = exists and size >= required_size
        if not ok:
            self.errors.append(f"store file {name} is missing or undersized.")
        return StoreFileCheck(name=name, path=str(path), required_size=required_size, exists=exists, size=size, crc32=crc, repaired=repaired, ok=ok)

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
