from __future__ import annotations

import os
import json
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

os.environ.setdefault("AXIOM_BUS_HMAC_KEY", "1" * 64)

from signal_kernel.contracts import (  # noqa: E402
    SnapshotCandidateEvent,
    SnapshotCapturedEvent,
    ToolHealthEvent,
    ToolInvocationEvent,
    ToolResultEvent,
)
from tag.tools_bridge import AXIOM_WATERMARK, DEFAULT_TOOLS_ROOT, ToolsBridge, sha256_json  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


def run_id() -> str:
    return str(uuid.uuid4())


def test_tools_bridge_registers_every_top_level_tool_directory() -> None:
    bridge = ToolsBridge(emit_bus=False)
    caps = bridge.list_capabilities()
    names = {cap["name"] for cap in caps}
    dirs = {
        path.name
        for path in DEFAULT_TOOLS_ROOT.iterdir()
        if path.is_dir() and path.name not in {"axiom-sdk", "dist", "node_modules"}
    }
    assert dirs <= names
    assert len(names) >= 40
    assert "AlpineStripTool" in names
    assert "WebFetchTool" in names
    assert "WebSearchTool" in names
    assert "AgentTool" in names
    assert "axiom-sdk" not in names


@pytest.mark.asyncio
async def test_tools_bridge_health_events_cover_all_tools() -> None:
    bridge = ToolsBridge(emit_bus=False)
    events = await bridge.publish_health(run_id=run_id())
    names = {event.tool_name for event in events}
    assert "WebFetchTool" in names
    assert "WorkflowTool" in names
    assert all(isinstance(event, ToolHealthEvent) for event in events)
    assert all(event.status in {"ready", "missing_deps", "disabled", "error"} for event in events)
    workflow = next(event for event in events if event.tool_name == "WorkflowTool")
    assert workflow.status in {"ready", "missing_deps"}
    assert workflow.adapter_kind == "workflow"
    alpine = next(event for event in events if event.tool_name == "AlpineStripTool")
    assert alpine.adapter_kind == "native_strip"
    assert alpine.permission_class == "read_only"


def test_tool_and_snapshot_contracts_validate_core_fields() -> None:
    rid = run_id()
    input_hash = sha256_json({"url": "https://example.com"})
    invocation = ToolInvocationEvent(
        tool_name="WebFetchTool",
        invocation_id="inv-1",
        input_hash=input_hash,
        run_id=rid,
        source_component="tag.tools_bridge",
        mode="snapshot",
        permission_class="network",
    )
    result = ToolResultEvent(
        tool_name="WebFetchTool",
        invocation_id=invocation.invocation_id,
        status="ok",
        output_hash=sha256_json({"ok": True}),
        duration_ms=1.5,
        run_id=rid,
    )
    candidate = SnapshotCandidateEvent(
        url="https://example.com/page",
        reason="unit",
        relevance_score=0.9,
        source_component="test",
        run_id=rid,
    )
    captured = SnapshotCapturedEvent(
        url=candidate.url,
        artifact_path="/tmp/axiom/page.html",
        artifact_kind="raw_html",
        sha256="0" * 64,
        byte_count=12,
        watermark=AXIOM_WATERMARK,
        source_tool="tools_bridge.snapshot",
        run_id=rid,
    )
    assert result.invocation_id == invocation.invocation_id
    assert captured.url == candidate.url


@pytest.mark.asyncio
async def test_snapshot_capture_writes_watermarked_temp_artifacts(tmp_path: Path) -> None:
    bridge = ToolsBridge(snapshot_root=tmp_path, emit_bus=False)
    candidate = bridge.candidate_events(
        ["https://example.com/docs", "https://example.com/docs"],
        query="docs",
        run_id=run_id(),
    )[0]
    events = await bridge.capture_candidate(
        candidate,
        raw_html="<html><body>signal</body></html>",
        markdown="# signal",
    )
    assert len(events) == 3
    for event in events:
        path = Path(event.artifact_path)
        assert path.exists()
        assert tmp_path in path.parents
        assert "store" not in path.parts
        assert event.watermark == AXIOM_WATERMARK
    raw_event = next(event for event in events if event.artifact_kind == "raw_html")
    assert AXIOM_WATERMARK in Path(raw_event.artifact_path).read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_workflow_tool_invocation_returns_real_tool_result_event() -> None:
    bridge = ToolsBridge(emit_bus=False)
    event = await bridge.invoke_tool("WorkflowTool", {"goal": "verify tag/interface.py", "files": ["tag/interface.py"]}, run_id=run_id())
    assert event.tool_name == "WorkflowTool"
    assert event.status == "ok"
    assert event.error_type is None
    assert "WorkflowTool" in event.output_summary


@pytest.mark.asyncio
async def test_alpine_strip_request_and_queue_line_from_snapshot(tmp_path: Path) -> None:
    bridge = ToolsBridge(snapshot_root=tmp_path, emit_bus=False)
    candidate = bridge.candidate_events(["https://example.com/docs"], run_id=run_id())[0]
    captured = await bridge.capture_candidate(candidate, raw_html="<p>Signal</p>")
    raw_event = next(event for event in captured if event.artifact_kind == "raw_html")
    output_path = tmp_path / "signal.txt"

    request = bridge.build_alpine_strip_request(
        raw_event,
        topology_class="SAAS_DOCS",
        output_path=output_path,
        slot_idx=3,
    )
    assert request["tool"] == "AlpineStripTool"
    assert request["artifact_kind"] == "raw_html"
    assert request["topology_class"] == "SAAS_DOCS"
    assert request["slot_idx"] == 3
    assert "tool_strip_accelerator.c" in request["native"]

    line = bridge.build_alpine_strip_queue_line(raw_event, output_path=output_path, slot_idx=3)
    payload = json.loads(line)
    assert payload["url"] == "https://example.com/docs"
    assert payload["slot_idx"] == 3
    assert payload["input_path"] == raw_event.artifact_path


@pytest.mark.asyncio
async def test_alpine_strip_tool_invocation_uses_sidecar_contract(tmp_path: Path) -> None:
    bridge = ToolsBridge(snapshot_root=tmp_path, emit_bus=False)
    candidate = bridge.candidate_events(["https://example.com/docs"], run_id=run_id())[0]
    raw_event = (await bridge.capture_candidate(candidate, raw_html="<p>Signal</p>"))[0]
    event = await bridge.invoke_alpine_strip(raw_event, run_id=run_id(), output_path=tmp_path / "out.signal")
    assert event.tool_name == "AlpineStripTool"
    assert event.status in {"ok", "error"}
    assert event.output_hash


def test_bus_payload_builder_accepts_new_tool_topics() -> None:
    from tag.crawler_bus import event_from_payload

    rid = run_id()
    event = event_from_payload(
        "tool_health",
        {
            "tool_name": "WebFetchTool",
            "status": "ready",
            "dependency_status": {"zod/v4": True},
            "permission_class": "network",
            "adapter_kind": "web",
            "run_id": rid,
            "detail": "unit",
        },
    )
    assert isinstance(event, ToolHealthEvent)
    assert event.tool_name == "WebFetchTool"


def test_axiom_runtime_c_abi_compiles_and_handles_basic_commands(tmp_path: Path) -> None:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not installed")
    out = tmp_path / ("axiom_runtime_test.exe" if os.name == "nt" else "axiom_runtime_test")
    subprocess.run(
        [
            gcc,
            "-std=c11",
            "-O2",
            "-Wall",
            "-Wextra",
            "-DAXIOM_RUNTIME_TEST",
            str(ROOT / "axiom_runtime" / "axiom_runtime.c"),
            "-o",
            str(out),
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run([str(out)], check=True, cwd=ROOT)
