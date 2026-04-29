from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def compile_native_runtime(tmp_path: Path) -> Path:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not installed")
    out = tmp_path / ("axi.dll" if os.name == "nt" else "axi.so")
    cmd = [
        gcc,
        "-std=c11",
        "-O2",
        "-Wall",
        "-Wextra",
    ]
    if os.name != "nt":
        cmd.extend(["-fPIC", "-shared"])
    else:
        cmd.append("-shared")
    cmd.extend([str(ROOT / "axiom_runtime" / "axiom_runtime.c"), "-o", str(out)])
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def test_axiom_infer_calls_native_runtime_and_prints_full_json(tmp_path: Path) -> None:
    lib = compile_native_runtime(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "axiom_infer.py"),
            "--lib",
            str(lib),
            "--store-dir",
            str(tmp_path / "store"),
            "--workers",
            "10",
            "--depth",
            "2",
            "--query",
            "find me latest AI news",
            "--compact",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["axiom_infer_version"] == "1.0.5"
    assert payload["native_version"] == "1.0.5"
    assert payload["ok"] is True
    assert payload["request"]["payload"] == "swarm -10 | depth -2 | find me latest AI news"
    assert payload["last_response"]["json"]["data"]["query"] == "find me latest AI news"
    assert payload["last_response"]["json"]["data"]["crawl_swarm"]["requested_worker_count"] == 10
    assert payload["last_response"]["json"]["data"]["crawl_swarm"]["depth"] == 2
    source_domains = {source["domain"] for source in payload["last_response"]["json"]["data"]["sources"]}
    assert {"openai.com", "arxiv.org", "reuters.com"} <= source_domains
    assert payload["last_response"]["raw"].startswith("{")


def test_axiom_infer_rejects_invalid_depth() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "axiom_infer.py"),
            "--depth",
            "99",
            "--query",
            "safe query",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 2
    payload = json.loads(proc.stderr)
    assert payload["ok"] is False
    assert payload["error"]["type"] == "ValueError"


def test_axiom_infer_cycle_summary_uses_one_native_process(tmp_path: Path) -> None:
    lib = compile_native_runtime(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "axiom_infer.py"),
            "--lib",
            str(lib),
            "--store-dir",
            str(tmp_path / "store"),
            "--query",
            "AI model release news",
            "--cycles",
            "5",
            "--summary-only",
            "--compact",
        ],
        check=True,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert payload["cycles"]["requested"] == 5
    assert payload["cycles"]["status_counts"].get("ok", 0) == 5
    assert payload["responses"] == []
