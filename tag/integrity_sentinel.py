"""
AXIOM integrity sentinel.

This module seals critical runtime files, stores local backups, verifies the
sealed graph periodically, and can restore from the sealed backup when
`integrity_sentinel.repair_mode = "restore"` is enabled in config.toml.
The native AXIOM runtime contributes a compiled proof hash when available.
"""

from __future__ import annotations

import argparse
import ctypes
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tag.config import AxiomConfig, load_config


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SECTION = "integrity_sentinel"
_BLOCK_COUNTS: Dict[str, int] = {}


@dataclass(frozen=True)
class IntegrityPaths:
    root: Path
    manifest_path: Path
    backup_dir: Path
    lock_path: Path
    event_log_path: Path

    def ensure_dirs(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self.event_log_path.parent.mkdir(parents=True, exist_ok=True)


class NativeIntegrityProof:
    def __init__(self, library_path: Optional[Path]) -> None:
        self.library_path = library_path
        self.available = False
        self.version = "unavailable"
        self._lib: Optional[ctypes.CDLL] = None
        if library_path is None or not library_path.exists():
            return
        try:
            lib = ctypes.CDLL(str(library_path))
            lib.axiom_integrity_version.argtypes = []
            lib.axiom_integrity_version.restype = ctypes.c_char_p
            lib.axiom_integrity_hash_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
            lib.axiom_integrity_hash_file.restype = ctypes.c_int
            raw_version = lib.axiom_integrity_version()
            self.version = raw_version.decode("utf-8", errors="replace") if raw_version else "unknown"
            self._lib = lib
            self.available = True
        except Exception:
            self._lib = None
            self.available = False

    def hash_file(self, path: Path) -> Optional[str]:
        if not self.available or self._lib is None:
            return None
        out = ctypes.create_string_buffer(32)
        status = self._lib.axiom_integrity_hash_file(str(path).encode("utf-8"), out, len(out))
        if status != 0:
            return None
        value = out.value.decode("ascii", errors="ignore")
        return f"fnv64:{value}" if value else None


class IntegritySentinel:
    def __init__(self, *, config: Optional[AxiomConfig] = None, root: Path = ROOT) -> None:
        self.config = config or load_config()
        self.root = root.resolve()
        self.paths = self._paths()
        self.repair_mode = str(self.config.get(f"{DEFAULT_SECTION}.repair_mode", "audit")).strip().lower()
        self.interval_seconds = self.config.float(f"{DEFAULT_SECTION}.interval_seconds", 30.0, low=1.0, high=3600.0)
        self.native = NativeIntegrityProof(self._native_library_path())

    def seal(self) -> Dict[str, Any]:
        self.paths.ensure_dirs()
        entries = []
        events = []
        for rel in self._configured_files():
            source = self._safe_source_path(rel)
            if source is None or not source.exists() or not source.is_file():
                events.append({"type": "seal_skip", "path": rel, "reason": "missing"})
                continue
            backup = self._backup_path(rel)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)
            entries.append(self._proof_for(rel, source, backup))
        manifest = {
            "version": 1,
            "sealed_at": _utc_now(),
            "root": str(self.root),
            "login_key": login_key(),
            "repair_mode": self.repair_mode,
            "native": {
                "available": self.native.available,
                "version": self.native.version,
                "library": str(self.native.library_path) if self.native.library_path else None,
            },
            "files": entries,
        }
        _atomic_write_json(self.paths.manifest_path, manifest)
        for event in events:
            self._log(event)
        self._log({"type": "sealed", "files": len(entries), "manifest": str(self.paths.manifest_path)})
        return manifest

    def check_once(self, *, repair: Optional[bool] = None) -> Dict[str, Any]:
        self.paths.ensure_dirs()
        manifest = self._load_or_seal()
        should_repair = self.repair_mode == "restore" if repair is None else repair
        results = []
        repaired = 0
        failed = 0
        for entry in manifest.get("files", []):
            result = self._check_entry(entry, repair=should_repair)
            results.append(result)
            repaired += 1 if result.get("repaired") else 0
            failed += 1 if not result.get("ok") else 0
            if result.get("event"):
                self._log(result["event"])
        summary = {
            "ok": failed == 0,
            "checked": len(results),
            "failed": failed,
            "repaired": repaired,
            "repair_mode": self.repair_mode,
            "native_available": self.native.available,
            "results": results,
        }
        self._log({"type": "check", **{k: v for k, v in summary.items() if k != "results"}})
        return summary

    def daemon(self) -> None:
        self.paths.ensure_dirs()
        self._write_lock()
        self._log({"type": "daemon_started", "login_key": login_key(), "pid": os.getpid()})
        while True:
            self.check_once()
            time.sleep(self.interval_seconds)

    def _check_entry(self, entry: Dict[str, Any], *, repair: bool) -> Dict[str, Any]:
        rel = str(entry.get("path") or "")
        source = self._safe_source_path(rel)
        backup = self.root / str(entry.get("backup") or "")
        if source is None:
            return {"path": rel, "ok": False, "repaired": False, "reason": "unsafe_path"}
        current = self._hash_file(source) if source.exists() else None
        expected = str(entry.get("sha256") or "")
        if current == expected:
            return {"path": rel, "ok": True, "repaired": False}
        event = {
            "type": "drift_detected",
            "path": rel,
            "expected_sha256": expected,
            "current_sha256": current,
            "repair": repair,
        }
        if repair and backup.exists() and backup.is_file():
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(backup, source)
            restored = self._hash_file(source)
            ok = restored == expected
            event["restored"] = ok
            return {"path": rel, "ok": ok, "repaired": ok, "reason": "restored_from_backup", "event": event}
        return {"path": rel, "ok": False, "repaired": False, "reason": "hash_mismatch", "event": event}

    def _proof_for(self, rel: str, source: Path, backup: Path) -> Dict[str, Any]:
        stat = source.stat()
        return {
            "path": rel,
            "sha256": self._hash_file(source),
            "native_hash": self.native.hash_file(source),
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "backup": str(backup.relative_to(self.root)),
            "sealed_at": _utc_now(),
        }

    def _load_or_seal(self) -> Dict[str, Any]:
        if not self.paths.manifest_path.exists():
            return self.seal()
        try:
            return json.loads(self.paths.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return self.seal()

    def _paths(self) -> IntegrityPaths:
        return IntegrityPaths(
            root=self.root,
            manifest_path=self._config_path("manifest_path", ".axiom_runtime/integrity/manifest.json"),
            backup_dir=self._config_path("backup_dir", ".axiom_runtime/integrity/backups"),
            lock_path=self._config_path("lock_path", ".axiom_runtime/integrity/axiom-integrity.lock"),
            event_log_path=self._config_path("event_log_path", ".axiom_runtime/integrity/events.jsonl"),
        )

    def _config_path(self, key: str, default: str) -> Path:
        raw = str(self.config.get(f"{DEFAULT_SECTION}.{key}", default) or default)
        path = Path(raw).expanduser()
        if not path.is_absolute():
            path = self.root / path
        return path.resolve(strict=False)

    def _configured_files(self) -> List[str]:
        raw = self.config.get(f"{DEFAULT_SECTION}.files", [])
        if not isinstance(raw, list):
            return []
        unique = []
        seen = set()
        for item in raw:
            rel = str(item).replace("\\", "/").strip().lstrip("/")
            if rel and rel not in seen:
                seen.add(rel)
                unique.append(rel)
        return unique

    def _safe_source_path(self, rel: str) -> Optional[Path]:
        if not rel or Path(rel).is_absolute() or ".." in Path(rel).parts:
            return None
        path = (self.root / rel).resolve(strict=False)
        try:
            path.relative_to(self.root)
        except ValueError:
            return None
        return path

    def _backup_path(self, rel: str) -> Path:
        return self.paths.backup_dir / rel

    def _hash_file(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _native_library_path(self) -> Optional[Path]:
        configured = str(self.config.get(f"{DEFAULT_SECTION}.native_library", "auto") or "auto").strip()
        if configured and configured.lower() != "auto":
            path = Path(configured).expanduser()
            return (self.root / path).resolve(strict=False) if not path.is_absolute() else path
        try:
            from tag.runtime_paths import RuntimePathResolver

            paths = RuntimePathResolver(config=self.config).resolve()
            for candidate in paths.native_library_candidates():
                if candidate.exists() and candidate.suffix.lower() in {".so", ".dll"}:
                    return candidate
        except Exception:
            return None
        return None

    def _write_lock(self) -> None:
        payload = {"pid": os.getpid(), "login_key": login_key(), "started_at": _utc_now()}
        _atomic_write_json(self.paths.lock_path, payload)

    def _log(self, event: Dict[str, Any]) -> None:
        event.setdefault("time", _utc_now())
        event.setdefault("pid", os.getpid())
        self.paths.event_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.paths.event_log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")


def mark_block(name: str) -> None:
    _BLOCK_COUNTS[name] = _BLOCK_COUNTS.get(name, 0) + 1
    if os.environ.get("AXIOM_INTEGRITY_TRACE", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return
    try:
        root = Path(os.environ.get("AXIOM_ROOT", str(ROOT))).resolve(strict=False)
        path = root / ".axiom_runtime" / "integrity" / "blocks.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"time": _utc_now(), "block": name, "count": _BLOCK_COUNTS[name]}) + "\n")
    except OSError:
        return


def start_once_per_login(config: Optional[AxiomConfig] = None) -> bool:
    config = config or load_config()
    if not config.bool(f"{DEFAULT_SECTION}.enabled", True) or not config.bool(f"{DEFAULT_SECTION}.autostart", False):
        return False
    sentinel = IntegritySentinel(config=config)
    sentinel.paths.ensure_dirs()
    if config.bool(f"{DEFAULT_SECTION}.once_per_login", True) and _lock_is_live(sentinel.paths.lock_path):
        return False
    cmd = [sys.executable, "-m", "tag.integrity_sentinel", "--daemon"]
    kwargs: Dict[str, Any] = {
        "cwd": str(sentinel.root),
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        kwargs["start_new_session"] = True
    subprocess.Popen(cmd, **kwargs)
    return True


def login_key() -> str:
    boot_id = ""
    try:
        boot_id = Path("/proc/sys/kernel/random/boot_id").read_text(encoding="utf-8").strip()
    except OSError:
        boot_id = os.environ.get("SESSIONNAME", "") or os.environ.get("USERDOMAIN", "")
    return f"{getpass.getuser()}:{boot_id or 'session'}"


def _lock_is_live(path: Path) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("login_key") != login_key():
            return False
        pid = int(payload.get("pid", 0))
        if pid <= 0:
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal, check, repair, or run the AXIOM integrity sentinel.")
    parser.add_argument("--seal", action="store_true", help="Create/update the manifest and .bak backups from current files.")
    parser.add_argument("--check", action="store_true", help="Verify the sealed manifest once.")
    parser.add_argument("--repair", action="store_true", help="Verify once and restore drifted files from backups.")
    parser.add_argument("--daemon", action="store_true", help="Run indefinitely, checking on the configured interval.")
    parser.add_argument("--compact", action="store_true", help="Emit compact JSON.")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    sentinel = IntegritySentinel()
    if args.daemon:
        sentinel.daemon()
        return 0
    if args.seal:
        payload = {"ok": True, "operation": "seal", "manifest": sentinel.seal()}
    elif args.repair:
        payload = {"ok": True, "operation": "repair", "result": sentinel.check_once(repair=True)}
    else:
        payload = {"ok": True, "operation": "check", "result": sentinel.check_once(repair=False)}
    json.dump(payload, sys.stdout, indent=None if args.compact else 2, sort_keys=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
