from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tag.config import AxiomConfig
from tag.integrity_sentinel import IntegritySentinel, mark_block


ROOT = Path(__file__).resolve().parents[1]


def _config(root: Path, *, repair_mode: str = "audit") -> AxiomConfig:
    return AxiomConfig(
        data={
            "integrity_sentinel": {
                "enabled": True,
                "autostart": False,
                "once_per_login": True,
                "interval_seconds": 1.0,
                "repair_mode": repair_mode,
                "manifest_path": ".axiom_runtime/integrity/manifest.json",
                "backup_dir": ".axiom_runtime/integrity/backups",
                "lock_path": ".axiom_runtime/integrity/axiom-integrity.lock",
                "event_log_path": ".axiom_runtime/integrity/events.jsonl",
                "native_library": "auto",
                "files": ["tag/interface.py"],
            }
        },
        path=root / "config.toml",
    )


def test_integrity_sentinel_seals_detects_and_repairs(tmp_path: Path) -> None:
    root = tmp_path / "root"
    source = root / "tag" / "interface.py"
    source.parent.mkdir(parents=True)
    source.write_text("good\n", encoding="utf-8")
    sentinel = IntegritySentinel(config=_config(root, repair_mode="restore"), root=root)

    manifest = sentinel.seal()
    assert manifest["files"][0]["path"] == "tag/interface.py"
    assert (root / manifest["files"][0]["backup"]).read_text(encoding="utf-8") == "good\n"

    source.write_text("bad\n", encoding="utf-8")
    result = sentinel.check_once()
    assert result["ok"] is True
    assert result["repaired"] == 1
    assert source.read_text(encoding="utf-8") == "good\n"


def test_integrity_sentinel_audit_mode_does_not_repair(tmp_path: Path) -> None:
    root = tmp_path / "root"
    source = root / "tag" / "interface.py"
    source.parent.mkdir(parents=True)
    source.write_text("good\n", encoding="utf-8")
    sentinel = IntegritySentinel(config=_config(root, repair_mode="audit"), root=root)
    sentinel.seal()

    source.write_text("bad\n", encoding="utf-8")
    result = sentinel.check_once()
    assert result["ok"] is False
    assert result["repaired"] == 0
    assert source.read_text(encoding="utf-8") == "bad\n"


def test_mark_block_is_lightweight_without_trace() -> None:
    mark_block("test.block")
    mark_block("test.block")


def test_native_runtime_exports_integrity_hash(tmp_path: Path) -> None:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not installed")
    out = tmp_path / ("axi.dll" if os.name == "nt" else "axi.so")
    cmd = [gcc, "-std=c11", "-O2", "-Wall", "-Wextra"]
    if os.name != "nt":
        cmd.extend(["-fPIC", "-shared"])
    else:
        cmd.append("-shared")
    cmd.extend([str(ROOT / "axiom_runtime" / "axiom_runtime.c"), "-o", str(out)])
    subprocess.run(cmd, check=True, cwd=ROOT)

    sample = tmp_path / "sample.txt"
    sample.write_text("axiom\n", encoding="utf-8")
    lib = ctypes.CDLL(str(out))
    lib.axiom_integrity_version.restype = ctypes.c_char_p
    lib.axiom_integrity_hash_file.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_size_t]
    lib.axiom_integrity_hash_file.restype = ctypes.c_int
    buf = ctypes.create_string_buffer(32)
    assert lib.axiom_integrity_version().decode("ascii") == "1.0.0"
    assert lib.axiom_integrity_hash_file(str(sample).encode("utf-8"), buf, len(buf)) == 0
    assert buf.value.decode("ascii")
