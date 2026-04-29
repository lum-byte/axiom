"""
tag/tools_bridge.py
===================
AXIOM bridge for the adopted tools/ runtime.

The bridge has three jobs:
1. Register every top-level tools/ directory as an AXIOM capability.
2. Convert tool activity into canonical bus events from signal_kernel/contracts.py.
3. Capture temporary, watermarked snapshots for URLs that AXIOM already found
   relevant through TAG routing/frontier traversal.

This module is intentionally Python-facing. TypeScript tools remain behind the
AXIOM SDK sidecar. The bus remains the only inter-component communication seam.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from signal_kernel.contracts import (
    SnapshotCandidateEvent,
    SnapshotCapturedEvent,
    ToolHealthEvent,
    ToolInvocationEvent,
    ToolResultEvent,
    new_run_id,
)


AXIOM_WATERMARK = "AXIOM SNAPSHOT ARTIFACT // TAG ROUTED // DO NOT TRAIN AS CLEAN SIGNAL"
DEFAULT_TOOLS_ROOT = Path(__file__).resolve().parents[1] / "tools"
DEFAULT_SNAPSHOT_ROOT = Path(tempfile.gettempdir()) / "axiom_snapshots"
SIDECAR_ENTRY = DEFAULT_TOOLS_ROOT / "dist" / "axiom-sdk" / "sidecar.js"
SOURCE_SIDECAR_ENTRY = DEFAULT_TOOLS_ROOT / "axiom-sdk" / "sidecar.ts"
ALPINE_STRIP_ACCELERATOR_SOURCE = DEFAULT_TOOLS_ROOT.parent / "alpine_strip" / "tool_strip_accelerator.c"


INFRASTRUCTURE_DIRS = {"axiom-sdk", "dist", "node_modules"}

NETWORK_TOOLS = {"RemoteTriggerTool", "WebFetchTool", "WebSearchTool"}
WRITE_TEMP_TOOLS = {"BriefTool", "FileReadTool"}
WRITE_REPO_TOOLS = {"FileEditTool", "FileWriteTool", "NotebookEditTool"}
ORCHESTRATION_TOOLS = {
    "AgentTool",
    "AskUserQuestionTool",
    "EnterPlanModeTool",
    "EnterWorktreeTool",
    "ExitPlanModeTool",
    "ExitWorktreeTool",
    "ScheduleCronTool",
    "SendMessageTool",
    "TaskCreateTool",
    "TaskGetTool",
    "TaskListTool",
    "TaskOutputTool",
    "TaskStopTool",
    "TaskUpdateTool",
    "TeamCreateTool",
    "TeamDeleteTool",
    "TodoWriteTool",
}

EXTERNAL_DEPENDENCIES: Dict[str, Sequence[str]] = {
    "AlpineStripTool": (),
    "AgentTool": ("zod/v4", "react", "bun:bundle"),
    "AskUserQuestionTool": ("zod/v4", "react", "bun:bundle"),
    "BashTool": ("zod/v4",),
    "BriefTool": ("zod/v4", "axios"),
    "ConfigTool": ("zod/v4", "bun:bundle"),
    "FileEditTool": ("diff", "zod/v4"),
    "FileReadTool": ("zod/v4",),
    "FileWriteTool": ("diff", "zod/v4"),
    "LSPTool": ("zod/v4",),
    "MCPTool": ("zod/v4", "react"),
    "McpAuthTool": ("lodash-es/reject.js", "zod/v4"),
    "PowerShellTool": ("zod/v4",),
    "SkillTool": ("lodash-es", "zod/v4"),
    "SyntheticOutputTool": ("ajv", "zod/v4"),
    "ToolSearchTool": ("lodash-es/memoize.js", "zod/v4"),
    "WebFetchTool": ("axios", "lru-cache", "zod/v4"),
    "WebSearchTool": ("zod/v4",),
}


@dataclass(frozen=True)
class ToolCapability:
    name: str
    source_path: Path
    adapter_kind: str
    permission_class: str
    dependencies: Sequence[str]
    status: str
    line_count: int
    file_count: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "source_path": str(self.source_path),
            "adapter_kind": self.adapter_kind,
            "permission_class": self.permission_class,
            "dependencies": list(self.dependencies),
            "status": self.status,
            "line_count": self.line_count,
            "file_count": self.file_count,
        }


@dataclass(frozen=True)
class ToolInvocationRecord:
    capability: ToolCapability
    invocation_id: str
    input_hash: str
    mode: str
    run_id: str
    started_at: float = field(default_factory=time.perf_counter)


@dataclass(frozen=True)
class SnapshotArtifact:
    url: str
    path: Path
    kind: str
    sha256: str
    byte_count: int
    metadata: Dict[str, Any]
    expires_at: str

    def to_event(self, *, run_id: str, source_tool: str) -> SnapshotCapturedEvent:
        return SnapshotCapturedEvent(
            url=self.url,
            artifact_path=str(self.path),
            artifact_kind=self.kind,
            sha256=self.sha256,
            byte_count=self.byte_count,
            watermark=AXIOM_WATERMARK,
            source_tool=source_tool,
            run_id=run_id,
            metadata=self.metadata,
            expires_at=self.expires_at,
        )


class ToolRegistry:
    """Filesystem-backed registry for every top-level tools/ capability."""

    def __init__(self, tools_root: Path = DEFAULT_TOOLS_ROOT) -> None:
        self.tools_root = tools_root
        self._capabilities: Dict[str, ToolCapability] = {}

    def discover(self) -> Dict[str, ToolCapability]:
        if not self.tools_root.exists():
            self._capabilities = {}
            return {}
        capabilities: Dict[str, ToolCapability] = {}
        for path in sorted(self.tools_root.iterdir(), key=lambda item: item.name.lower()):
            if not path.is_dir() or path.name in INFRASTRUCTURE_DIRS:
                continue
            capability = self._build_capability(path)
            capabilities[capability.name] = capability
        self._capabilities = capabilities
        return dict(capabilities)

    def all(self) -> List[ToolCapability]:
        if not self._capabilities:
            self.discover()
        return [self._capabilities[name] for name in sorted(self._capabilities)]

    def get(self, name: str) -> ToolCapability:
        if not self._capabilities:
            self.discover()
        try:
            return self._capabilities[name]
        except KeyError as exc:
            raise KeyError(f"tool {name!r} is not registered") from exc

    def health_events(self, *, run_id: str) -> List[ToolHealthEvent]:
        events: List[ToolHealthEvent] = []
        for capability in self.all():
            deps = {dep: dependency_available(dep, self.tools_root) for dep in capability.dependencies}
            status = capability.status
            if status == "ready" and deps and not all(deps.values()):
                status = "missing_deps"
            events.append(
                ToolHealthEvent(
                    tool_name=capability.name,
                    status=status,
                    dependency_status=deps,
                    permission_class=capability.permission_class,
                    adapter_kind=capability.adapter_kind,
                    run_id=run_id,
                    detail=f"{capability.file_count} files, {capability.line_count} lines",
                )
            )
        return events

    def _build_capability(self, path: Path) -> ToolCapability:
        files = [item for item in path.rglob("*") if item.is_file() and item.suffix in {".ts", ".tsx", ".js"}]
        line_count = 0
        for file in files:
            try:
                line_count += len(file.read_text(encoding="utf-8", errors="ignore").splitlines())
            except OSError:
                continue
        file_count = len(files)
        status = "ready" if file_count else "disabled"
        return ToolCapability(
            name=path.name,
            source_path=path,
            adapter_kind=adapter_kind_for_tool(path.name),
            permission_class=permission_class_for_tool(path.name),
            dependencies=EXTERNAL_DEPENDENCIES.get(path.name, ("zod/v4",)),
            status=status,
            line_count=line_count,
            file_count=file_count,
        )


class AxiomToolSidecar:
    """
    JSONL client for tools/axiom-sdk/sidecar.ts.

    The sidecar is optional in development. If Node or the compiled sidecar is
    missing, the Python bridge still reports capability health and can capture
    snapshots with its built-in artifact path.
    """

    def __init__(self, *, tools_root: Path = DEFAULT_TOOLS_ROOT, node_bin: Optional[str] = None) -> None:
        self.tools_root = tools_root
        self.node_bin = node_bin or shutil.which("node")
        self.sidecar_entry = tools_root / "dist" / "axiom-sdk" / "sidecar.js"

    @property
    def available(self) -> bool:
        return self.node_bin is not None and self.sidecar_entry.exists()

    def request(self, payload: Dict[str, Any], timeout_s: float = 10.0) -> Dict[str, Any]:
        if not self.available:
            return python_sidecar_fallback(payload, self.tools_root)
        assert self.node_bin is not None
        proc = subprocess.run(
            [self.node_bin, str(self.sidecar_entry)],
            input=json.dumps(payload, sort_keys=True) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(self.tools_root),
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            return {
                "ok": False,
                "error_type": "SidecarProcessFailed",
                "error": proc.stderr.strip() or f"exit {proc.returncode}",
            }
        line = proc.stdout.splitlines()[0] if proc.stdout.splitlines() else "{}"
        try:
            return json.loads(line)
        except json.JSONDecodeError as exc:
            return {"ok": False, "error_type": "SidecarDecodeFailed", "error": str(exc)}


def python_sidecar_fallback(payload: Dict[str, Any], tools_root: Path) -> Dict[str, Any]:
    """Deterministic local runner used when the compiled TypeScript sidecar is absent."""
    op = payload.get("op")
    if op == "call":
        tool = str(payload.get("tool", ""))
        tool_input = payload.get("input") if isinstance(payload.get("input"), dict) else {}
        invocation_id = str(uuid.uuid4())
        started = time.perf_counter()
        try:
            data = python_tool_dispatch(tool, tool_input)  # type: ignore[arg-type]
            return {
                "ok": True,
                "result": {
                    "ok": True,
                    "toolName": tool,
                    "invocationId": invocation_id,
                    "durationMs": (time.perf_counter() - started) * 1000.0,
                    "data": data,
                    "resultBlock": {
                        "type": "tool_result",
                        "tool_use_id": invocation_id,
                        "content": json.dumps(data, sort_keys=True, default=str),
                    },
                },
                "input_hash": sha256_json(tool_input),
                "runner": "python_fallback",
            }
        except Exception as exc:  # pragma: no cover - defensive fallback boundary
            return {
                "ok": False,
                "error_type": exc.__class__.__name__,
                "error": str(exc),
                "runner": "python_fallback",
            }
    if op == "list":
        return {"ok": True, "tools": [cap.to_dict() for cap in ToolRegistry(tools_root).all()], "runner": "python_fallback"}
    if op == "health":
        rid = str(new_run_id())
        return {
            "ok": True,
            "tools": [
                {
                    **cap.to_dict(),
                    "dependencyStatus": {dep: dependency_available(dep, tools_root) for dep in cap.dependencies},
                }
                for cap in ToolRegistry(tools_root).all()
            ],
            "run_id": rid,
            "runner": "python_fallback",
        }
    return {"ok": False, "error_type": "UnknownOperation", "error": f"unknown op {op!r}", "runner": "python_fallback"}


def python_tool_dispatch(tool_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name == "AlpineStripTool":
        return {
            "adapter": "AlpineStripTool",
            "status": "planned",
            "native": "alpine_strip/tool_strip_accelerator.c",
            "request": payload,
            "queue_line": _alpine_queue_line(payload),
        }
    if tool_name == "WorkflowTool":
        return _workflow_result(payload)
    if tool_name == "TungstenTool":
        return _tungsten_result(payload)
    if tool_name == "VerifyPlanExecutionTool":
        return _verify_plan_result(payload)
    if tool_name == "SuggestBackgroundPRTool":
        return _suggest_pr_result(payload)
    if tool_name == "WebSearchTool":
        return {
            "adapter": "WebSearchTool",
            "status": "registered",
            "query": payload.get("query"),
            "note": "Available for explicit diagnostic tool calls; AXIOM TAG search remains primary.",
        }
    if tool_name == "WebFetchTool":
        return {
            "adapter": "WebFetchTool",
            "status": "planned",
            "url": payload.get("url"),
            "note": "Bridge will use the TypeScript WebFetchTool when the sidecar is compiled.",
        }
    return {
        "adapter": tool_name,
        "status": "executed_metadata_adapter",
        "inputHash": sha256_json(payload),
        "adapterKind": adapter_kind_for_tool(tool_name),
        "permissionClass": permission_class_for_tool(tool_name),
    }


def _as_string_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.replace(",", "\n").splitlines() if item.strip()]
    return []


def _workflow_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    goal = str(payload.get("goal") or payload.get("task") or payload.get("prompt") or "AXIOM workflow")
    files = _as_string_list(payload.get("files") or payload.get("changedFiles") or payload.get("changed_files"))
    lower = f"{goal} {' '.join(files)}".lower()
    commands: List[str] = []
    if "tools" in lower:
        commands.append("cd tools && npm run typecheck && npm run build")
    if "preparser" in lower or ".go" in lower:
        commands.append("go test ./preparser/...")
    if "offline" in lower or ".cu" in lower:
        commands.append("sh ./run_cuda_tests.sh")
    if "daemon" in lower or "alpine_strip" in lower or ".c" in lower:
        commands.append("sh ./run_c_tests.sh")
    if "axiom_tui" in lower or ".rs" in lower:
        commands.append("cd axiom_tui && cargo test")
    if "python" in lower or "tag/" in lower or "signal_kernel" in lower:
        commands.append("python -m pytest tests -q")
    if not commands:
        commands.append("python -m pytest tests -q")
    return {
        "adapter": "WorkflowTool",
        "status": "ready",
        "goal": goal,
        "files": files,
        "stages": [
            {"id": "context", "title": "Read local contracts and docs", "commands": []},
            {"id": "implementation", "title": "Apply scoped implementation changes", "commands": []},
            {"id": "verification", "title": "Run focused verification", "commands": sorted(set(commands))},
        ],
        "acceptance": [
            "contract-visible payloads stay canonical",
            "generated artifacts stay outside durable store files unless owned by the component",
            "focused tests pass before integration tests",
        ],
    }


def _tungsten_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    root = Path(str(payload.get("root") or payload.get("cwd") or Path.cwd())).resolve()
    sampled_files = 0
    sampled_bytes = 0
    if root.exists():
        for current, dirs, files in os.walk(root):
            dirs[:] = [name for name in dirs if name not in {".git", "node_modules", "dist", "__pycache__"}]
            for file_name in files:
                sampled_files += 1
                try:
                    sampled_bytes += (Path(current) / file_name).stat().st_size
                except OSError:
                    pass
                if sampled_files >= 5000:
                    break
            if sampled_files >= 5000:
                break
    return {
        "adapter": "TungstenTool",
        "status": "ready",
        "platform": os.name,
        "pid": os.getpid(),
        "cwd": str(root),
        "disk_sample": {"sampled_files": sampled_files, "sampled_bytes": sampled_bytes},
        "checked_at": datetime.now(timezone.utc).isoformat(),
    }


def _verify_plan_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_steps = payload.get("plan") or payload.get("steps") or payload.get("checklist") or []
    if not isinstance(raw_steps, list):
        raw_steps = _as_string_list(raw_steps)
    steps = [_normalize_plan_step(item, idx) for idx, item in enumerate(raw_steps)]
    tests = _as_string_list(payload.get("tests") or payload.get("testOutput") or payload.get("test_output"))
    changed_files = _as_string_list(payload.get("changedFiles") or payload.get("changed_files") or payload.get("files"))
    completed = [step for step in steps if step["status"] == "completed"]
    blocked = [step for step in steps if step["status"] == "blocked"]
    failed_tests = [test for test in tests if any(token in test.lower() for token in ("fail", "error", "nonzero", "timeout"))]
    test_score = 0.0 if not tests else max(0.0, 1.0 - len(failed_tests) / len(tests))
    plan_score = 0.0 if not steps else len(completed) / len(steps)
    file_score = 1.0 if changed_files else 0.0
    score = round(plan_score * 0.55 + test_score * 0.35 + file_score * 0.10, 4)
    missing = []
    if not steps:
        missing.append("plan")
    if len(completed) != len(steps):
        missing.append("completed_steps")
    if not tests:
        missing.append("tests")
    if not changed_files:
        missing.append("changed_files")
    status = "failed" if blocked or failed_tests else "verified" if not missing and score >= 0.95 else "incomplete"
    return {
        "adapter": "VerifyPlanExecutionTool",
        "status": status,
        "score": score,
        "completed_steps": len(completed),
        "total_steps": len(steps),
        "pending_steps": [step for step in steps if step["status"] != "completed"],
        "blocked_steps": blocked,
        "failed_tests": failed_tests,
        "changed_files": changed_files,
        "missing_evidence": missing,
    }


def _suggest_pr_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    changed_files = _as_string_list(payload.get("changedFiles") or payload.get("changed_files") or payload.get("files"))
    goals = _as_string_list(payload.get("goals") or payload.get("goal") or payload.get("summary"))
    risk = _risk_from_files(changed_files)
    tests = _tests_from_files(changed_files)
    title_seed = goals[0] if goals else (f"Update {Path(changed_files[0]).parts[0]}" if changed_files else "AXIOM runtime update")
    return {
        "adapter": "SuggestBackgroundPRTool",
        "status": "ready",
        "branch": str(payload.get("branch") or payload.get("branchName") or "codex/axiom-runtime-hardening"),
        "base": str(payload.get("base") or payload.get("baseBranch") or "main"),
        "title": f"[AXIOM] {title_seed}"[:120],
        "summary": goals or ["Harden AXIOM runtime behavior behind the existing contract seam."],
        "risk": risk,
        "tests": tests,
        "changed_files": changed_files,
    }


def _normalize_plan_step(item: Any, idx: int) -> Dict[str, str]:
    if isinstance(item, dict):
        text = str(item.get("text") or item.get("step") or item.get("title") or "")
        status_value = str(item.get("status") or "").lower()
        step_id = str(item.get("id") or item.get("step_id") or f"step-{idx + 1}")
    else:
        text = str(item)
        status_value = ""
        step_id = f"step-{idx + 1}"
    if status_value in {"done", "complete", "completed", "pass", "passed"}:
        status = "completed"
    elif status_value in {"doing", "active", "in_progress", "running"}:
        status = "in_progress"
    elif status_value in {"blocked", "fail", "failed", "error"}:
        status = "blocked"
    else:
        status = "pending"
    return {"id": step_id, "text": text, "status": status}


def _risk_from_files(files: Sequence[str]) -> Dict[str, str]:
    joined = "\n".join(files).lower()
    if any(token in joined for token in ("contracts", "crawler_bus", ".cu", ".c")):
        return {"level": "high", "reason": "Contract, bus, native, or GPU surfaces changed."}
    if any(token in joined for token in ("interface", "daemon", "offline", ".go", ".rs")):
        return {"level": "medium", "reason": "Runtime implementation changed and needs focused regression."}
    if files:
        return {"level": "low", "reason": "Change set avoids core runtime seams."}
    return {"level": "unknown", "reason": "No changed files were supplied."}


def _tests_from_files(files: Sequence[str]) -> List[str]:
    commands = set()
    for file_name in files:
        lower = file_name.lower()
        if lower.endswith(".py"):
            commands.add("python -m pytest tests -q")
        if lower.endswith((".ts", ".tsx")) or "tools/" in lower:
            commands.add("cd tools && npm run typecheck && npm run build")
        if lower.endswith(".go") or "preparser/" in lower:
            commands.add("go test ./preparser/...")
        if lower.endswith(".c") or "daemons/" in lower or "alpine_strip/" in lower:
            commands.add("sh ./run_c_tests.sh")
        if lower.endswith(".cu") or "offline/" in lower:
            commands.add("sh ./run_cuda_tests.sh")
        if lower.endswith(".rs") or "axiom_tui/" in lower:
            commands.add("cd axiom_tui && cargo test")
    if not commands:
        commands.add("python -m pytest tests -q")
    return sorted(commands)


def _alpine_queue_line(payload: Dict[str, Any]) -> Optional[str]:
    if not payload.get("input_path") or not payload.get("output_path"):
        return None
    return json.dumps(
        {
            "url": payload.get("url", ""),
            "slot_idx": int(payload.get("slot_idx", 0) or 0),
            "input_path": payload["input_path"],
            "output_path": payload["output_path"],
        },
        sort_keys=True,
        separators=(",", ":"),
    )


class SnapshotStore:
    """Temporary artifact store for watermarked AXIOM snapshots."""

    def __init__(self, root: Path = DEFAULT_SNAPSHOT_ROOT) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def capture(
        self,
        *,
        url: str,
        run_id: str,
        artifacts: Dict[str, bytes | str],
        ttl_seconds: int = 3600,
    ) -> List[SnapshotArtifact]:
        if not _valid_http_url(url):
            raise ValueError(f"snapshot url must be http(s), got {url!r}")
        safe_domain = _safe_component(urlparse(url).netloc or "unknown")
        run_dir = self.root / _safe_component(run_id) / safe_domain
        run_dir.mkdir(parents=True, exist_ok=True)
        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=ttl_seconds)).isoformat()
        captured: List[SnapshotArtifact] = []
        for kind, body in artifacts.items():
            payload = self._watermark_payload(url=url, kind=kind, body=body, expires_at=expires_at)
            digest = hashlib.sha256(payload).hexdigest()
            path = run_dir / f"{_safe_component(kind)}.{extension_for_kind(kind)}"
            path.write_bytes(payload)
            captured.append(
                SnapshotArtifact(
                    url=url,
                    path=path,
                    kind=kind,
                    sha256=digest,
                    byte_count=len(payload),
                    metadata={
                        "url": url,
                        "kind": kind,
                        "domain": safe_domain,
                        "watermarked": True,
                        "clean_signal_safe": False,
                    },
                    expires_at=expires_at,
                )
            )
        return captured

    def prune_expired(self, now: Optional[datetime] = None) -> int:
        now = now or datetime.now(timezone.utc)
        removed = 0
        for meta_path in self.root.rglob("metadata.json"):
            try:
                payload = json.loads(meta_path.read_text(encoding="utf-8"))
                expires = payload.get("expires_at")
                if expires and datetime.fromisoformat(expires) <= now:
                    for sibling in meta_path.parent.iterdir():
                        if sibling.is_file():
                            sibling.unlink()
                            removed += 1
            except (OSError, json.JSONDecodeError, ValueError):
                continue
        return removed

    @staticmethod
    def _watermark_payload(*, url: str, kind: str, body: bytes | str, expires_at: str) -> bytes:
        if isinstance(body, str):
            raw = body.encode("utf-8")
        else:
            raw = body
        metadata = {
            "watermark": AXIOM_WATERMARK,
            "url": url,
            "kind": kind,
            "captured_at": datetime.now(timezone.utc).isoformat(),
            "expires_at": expires_at,
        }
        if kind in {"raw_html", "rendered_html", "markdown", "metadata"}:
            prefix = ("<!-- " + json.dumps(metadata, sort_keys=True) + " -->\n").encode("utf-8")
            return prefix + raw
        return raw + b"\n" + json.dumps(metadata, sort_keys=True).encode("utf-8")


class ToolsBridge:
    """Bus-facing orchestrator for tool health, invocation, and snapshots."""

    def __init__(
        self,
        *,
        tools_root: Path = DEFAULT_TOOLS_ROOT,
        snapshot_root: Path = DEFAULT_SNAPSHOT_ROOT,
        emit_bus: bool = True,
    ) -> None:
        self.registry = ToolRegistry(tools_root)
        self.sidecar = AxiomToolSidecar(tools_root=tools_root)
        self.snapshots = SnapshotStore(snapshot_root)
        self.emit_bus = emit_bus
        self._emitters: Dict[str, Any] = {}

    def list_capabilities(self) -> List[Dict[str, Any]]:
        return [capability.to_dict() for capability in self.registry.all()]

    async def publish_health(self, *, run_id: Optional[str] = None) -> List[ToolHealthEvent]:
        rid = run_id or str(new_run_id())
        events = self.registry.health_events(run_id=rid)
        for event in events:
            await self._emit("tool_health", event)
        return events

    def candidate_events(
        self,
        urls: Iterable[str],
        *,
        query: str = "",
        reason: str = "tag_relevance",
        run_id: Optional[str] = None,
        source_component: str = "tag.tools_bridge",
        ttl_seconds: int = 3600,
    ) -> List[SnapshotCandidateEvent]:
        rid = run_id or str(new_run_id())
        clean_urls = []
        for url in urls:
            if _valid_http_url(url) and url not in clean_urls:
                clean_urls.append(url)
        total = max(1, len(clean_urls))
        events = []
        for index, url in enumerate(clean_urls):
            events.append(
                SnapshotCandidateEvent(
                    url=url,
                    reason=reason,
                    relevance_score=max(0.0, 1.0 - (index / total)),
                    source_component=source_component,
                    run_id=rid,
                    query=query,
                    rank=index + 1,
                    ttl_seconds=ttl_seconds,
                )
            )
        return events

    async def capture_candidate(
        self,
        candidate: SnapshotCandidateEvent,
        *,
        raw_html: Optional[str | bytes] = None,
        rendered_html: Optional[str | bytes] = None,
        markdown: Optional[str | bytes] = None,
        screenshot_bytes: Optional[bytes] = None,
        fetch_if_missing: bool = False,
    ) -> List[SnapshotCapturedEvent]:
        artifacts: Dict[str, bytes | str] = {}
        if raw_html is not None:
            artifacts["raw_html"] = raw_html
        if rendered_html is not None:
            artifacts["rendered_html"] = rendered_html
        if markdown is not None:
            artifacts["markdown"] = markdown
        if screenshot_bytes is not None:
            artifacts["screenshot"] = screenshot_bytes
        if not artifacts and fetch_if_missing:
            artifacts["raw_html"] = fetch_url_bytes(candidate.url)
        metadata = {
            "url": candidate.url,
            "query": candidate.query,
            "reason": candidate.reason,
            "rank": candidate.rank,
            "relevance_score": candidate.relevance_score,
            "watermark": AXIOM_WATERMARK,
        }
        artifacts["metadata"] = json.dumps(metadata, sort_keys=True, indent=2)
        captured = self.snapshots.capture(
            url=candidate.url,
            run_id=candidate.run_id,
            artifacts=artifacts,
            ttl_seconds=candidate.ttl_seconds,
        )
        events = [artifact.to_event(run_id=candidate.run_id, source_tool="tools_bridge.snapshot") for artifact in captured]
        for event in events:
            await self._emit("snapshot_captured", event)
        return events

    async def invoke_tool(
        self,
        tool_name: str,
        payload: Dict[str, Any],
        *,
        run_id: Optional[str] = None,
        mode: str = "manual",
    ) -> ToolResultEvent:
        rid = run_id or str(new_run_id())
        capability = self.registry.get(tool_name)
        invocation = ToolInvocationRecord(
            capability=capability,
            invocation_id=str(uuid.uuid4()),
            input_hash=sha256_json(payload),
            mode=mode,
            run_id=rid,
        )
        invocation_event = ToolInvocationEvent(
            tool_name=tool_name,
            invocation_id=invocation.invocation_id,
            input_hash=invocation.input_hash,
            run_id=rid,
            source_component="tag.tools_bridge",
            mode=mode,
            permission_class=capability.permission_class,
        )
        await self._emit("tool_invocation", invocation_event)

        sidecar_result = self.sidecar.request({"op": "call", "tool": tool_name, "input": payload, "run_id": rid, "mode": mode})
        status = "ok" if sidecar_result.get("ok") else "error"
        output = sidecar_result
        error_type = None if status == "ok" else str(sidecar_result.get("error_type", "ToolError"))

        result_event = ToolResultEvent(
            tool_name=tool_name,
            invocation_id=invocation.invocation_id,
            status=status,
            output_hash=sha256_json(output),
            duration_ms=(time.perf_counter() - invocation.started_at) * 1000.0,
            run_id=rid,
            output_summary=json.dumps(output, sort_keys=True)[:500],
            error_type=error_type,
        )
        await self._emit("tool_result", result_event)
        return result_event

    def build_alpine_strip_request(
        self,
        artifact: SnapshotCapturedEvent | SnapshotArtifact,
        *,
        topology_class: str = "GENERIC_HTML",
        inline_payload: Optional[str | bytes] = None,
        output_path: Optional[str | Path] = None,
        slot_idx: int = 0,
        max_output_ratio: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Build the native strip request consumed by tool_strip_accelerator.c."""
        artifact_path = Path(artifact.artifact_path) if hasattr(artifact, "artifact_path") else artifact.path
        artifact_kind = artifact.artifact_kind if hasattr(artifact, "artifact_kind") else artifact.kind
        url = artifact.url
        request: Dict[str, Any] = {
            "tool": "AlpineStripTool",
            "artifact_kind": artifact_kind,
            "url": url,
            "topology_class": topology_class,
            "input_path": str(artifact_path),
            "slot_idx": int(slot_idx),
            "native": str(ALPINE_STRIP_ACCELERATOR_SOURCE),
        }
        if output_path is not None:
            request["output_path"] = str(output_path)
        if inline_payload is not None:
            request["input"] = inline_payload.decode("utf-8", errors="replace") if isinstance(inline_payload, bytes) else inline_payload
        if max_output_ratio is not None:
            request["max_output_ratio"] = float(max_output_ratio)
        return request

    def build_alpine_strip_queue_line(
        self,
        artifact: SnapshotCapturedEvent | SnapshotArtifact,
        *,
        output_path: str | Path,
        slot_idx: int = 0,
    ) -> str:
        """Build a batch_runner JSONL line for a captured tool artifact."""
        artifact_path = Path(artifact.artifact_path) if hasattr(artifact, "artifact_path") else artifact.path
        payload = {
            "url": artifact.url,
            "slot_idx": int(slot_idx),
            "input_path": str(artifact_path),
            "output_path": str(output_path),
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    async def invoke_alpine_strip(
        self,
        artifact: SnapshotCapturedEvent | SnapshotArtifact,
        *,
        run_id: Optional[str] = None,
        topology_class: str = "GENERIC_HTML",
        output_path: Optional[str | Path] = None,
        mode: str = "snapshot",
    ) -> ToolResultEvent:
        request = self.build_alpine_strip_request(
            artifact,
            topology_class=topology_class,
            output_path=output_path,
        )
        return await self.invoke_tool("AlpineStripTool", request, run_id=run_id, mode=mode)

    async def _emit(self, topic: str, event: Any) -> None:
        if not self.emit_bus:
            return
        from tag.crawler_bus import BUS, TOPIC_REGISTRY, is_bus_started

        if not is_bus_started():
            await BUS.start()
        emitter = self._emitters.get(topic)
        if emitter is None:
            emitter = await BUS.emitter(topic=topic, component="tag.tools_bridge", schema=TOPIC_REGISTRY[topic])
            self._emitters[topic] = emitter
        await emitter.emit(event)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def dependency_available(dep: str, tools_root: Path = DEFAULT_TOOLS_ROOT) -> bool:
    if dep == "":
        return True
    if dep == "bun:bundle":
        return shutil.which("bun") is not None or (tools_root / "package.json").exists()
    if dep.startswith("node:"):
        return True
    node_modules = tools_root / "node_modules"
    if dep.endswith(".js"):
        dep = dep.rsplit("/", 1)[0]
    package = dep.split("/", 1)[0] if not dep.startswith("@") else "/".join(dep.split("/", 2)[:2])
    return (node_modules / package).exists()


def permission_class_for_tool(name: str) -> str:
    if name == "AlpineStripTool":
        return "read_only"
    if name in NETWORK_TOOLS:
        return "network"
    if name in WRITE_REPO_TOOLS:
        return "write_repo"
    if name in WRITE_TEMP_TOOLS:
        return "write_temp"
    if name in {"WorkflowTool", "SuggestBackgroundPRTool", "VerifyPlanExecutionTool"}:
        return "orchestration"
    if name in ORCHESTRATION_TOOLS:
        return "orchestration"
    return "read_only"


def adapter_kind_for_tool(name: str) -> str:
    if name == "AlpineStripTool":
        return "native_strip"
    if name == "TungstenTool":
        return "runtime_monitor"
    if name == "VerifyPlanExecutionTool":
        return "verification"
    if name == "WorkflowTool":
        return "workflow"
    if name == "SuggestBackgroundPRTool":
        return "orchestration"
    if "Web" in name:
        return "web"
    if "File" in name or "Notebook" in name or name == "BriefTool":
        return "artifact"
    if "Bash" in name or "PowerShell" in name:
        return "shell_guard"
    if "MCP" in name or "Mcp" in name or "Resource" in name:
        return "connector"
    if "Task" in name or "Team" in name or "Agent" in name or "Todo" in name:
        return "orchestration"
    return "utility"


def fetch_url_bytes(url: str, timeout_s: float = 10.0) -> bytes:
    request = Request(url, headers={"User-Agent": "AXIOM-tools-bridge/0.1"})
    with urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - explicit opt-in snapshot fetch
        return response.read()


def extension_for_kind(kind: str) -> str:
    return {
        "raw_html": "html",
        "rendered_html": "html",
        "markdown": "md",
        "screenshot": "bin",
        "metadata": "json",
        "bundle": "json",
    }.get(kind, "bin")


def _valid_http_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _safe_component(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value)[:160] or "item"


async def _main() -> int:
    bridge = ToolsBridge(emit_bus=False)
    print(json.dumps({"tools": bridge.list_capabilities()}, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
