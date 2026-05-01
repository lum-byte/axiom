from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_mcp_worker_expands_query() -> None:
    proc = subprocess.run(
        [sys.executable, "-m", "tag.mcp_worker", "expand"],
        input=json.dumps({"query": "what is github", "limit": 4}),
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["expansion"]["effective_limit"] == 4
    assert payload["expansion"]["detected_type"] == "FACTUAL_DIRECT"


@pytest.mark.skipif(shutil.which("go") is None, reason="go is not installed")
def test_go_mcp_server_lists_tools_and_calls_expand() -> None:
    env = os.environ.copy()
    env["AXIOM_MCP_PYTHON"] = sys.executable
    env["AXIOM_CONFIG_TOML"] = str(ROOT / "config.toml")
    proc = subprocess.Popen(
        ["go", "run", "./cmd/tag-mcp"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
        env=env,
    )
    assert proc.stdin is not None
    assert proc.stdout is not None
    try:
        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {},
                        "clientInfo": {"name": "pytest", "version": "1.0"},
                    },
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        init = json.loads(proc.stdout.readline())
        assert init["result"]["capabilities"]["tools"]["listChanged"] is False

        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized", "params": {}}) + "\n")
        proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}) + "\n")
        proc.stdin.flush()
        listed = json.loads(proc.stdout.readline())
        names = {tool["name"] for tool in listed["result"]["tools"]}
        assert {"tag.search", "tag.expand", "anchor.wikipedia", "anchor.news", "anchor.scholar", "anchor.wayback", "anchor.web"} <= names

        proc.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "tag.expand", "arguments": {"query": "what is github", "limit": 3}},
                }
            )
            + "\n"
        )
        proc.stdin.flush()
        called = json.loads(proc.stdout.readline())
        assert called["result"]["isError"] is False
        assert called["result"]["structuredContent"]["expansion"]["effective_limit"] == 3
    finally:
        if proc.stdin:
            proc.stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
