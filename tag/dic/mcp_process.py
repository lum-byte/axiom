"""External MCP process bridge for TAG anchor acquisition.

This module intentionally speaks a small, strict subset of MCP over stdio:
initialize, initialized notification, and tools/call.  The Go binary owns the
server side.  TAG uses this client only for anchor acquisition so the external
MCP process becomes part of the crawler path without recursively invoking
`tag.search`.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tag.config import AxiomConfig, ROOT, load_config


class MCPProcessError(RuntimeError):
    pass


@dataclass(frozen=True)
class MCPToolResponse:
    tool: str
    structured: Dict[str, Any]
    text: str
    stderr_preview: str = ""


class MCPStdioClient:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.command = resolve_mcp_command(self.config)
        self.timeout_s = self.config.float("mcp.anchor_timeout_seconds", 20.0, low=1.0, high=180.0)
        self.protocol_version = self.config.str("mcp.protocol_version", "2025-11-25")

    async def call_tool(self, name: str, arguments: Dict[str, Any], *, timeout_s: Optional[float] = None) -> MCPToolResponse:
        if not self.command:
            raise MCPProcessError("MCP server command could not be resolved")
        timeout = timeout_s if timeout_s is not None else self.timeout_s
        proc = await asyncio.create_subprocess_exec(
            *self.command,
            cwd=str(ROOT),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=mcp_process_env(self.config),
        )
        assert proc.stdin is not None
        assert proc.stdout is not None
        try:
            await self._send(proc, 1, "initialize", {
                "protocolVersion": self.protocol_version,
                "capabilities": {},
                "clientInfo": {"name": "axiom-tag-python", "version": self.config.str("runtime.version", "1.0.5")},
            })
            await asyncio.wait_for(self._read_response(proc, 1), timeout=timeout)
            await self._notify(proc, "notifications/initialized", {})
            await self._send(proc, 2, "tools/call", {"name": name, "arguments": dict(arguments)})
            response = await asyncio.wait_for(self._read_response(proc, 2), timeout=timeout)
        finally:
            if proc.stdin and not proc.stdin.is_closing():
                proc.stdin.close()
            with contextlib.suppress(Exception):
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            if proc.returncode is None:
                proc.kill()
                with contextlib.suppress(Exception):
                    await proc.wait()
        stderr_preview = ""
        if proc.stderr is not None:
            with contextlib.suppress(Exception):
                stderr = await asyncio.wait_for(proc.stderr.read(4096), timeout=0.1)
                stderr_preview = stderr.decode("utf-8", errors="replace").strip()
        if "error" in response:
            raise MCPProcessError(json.dumps(response["error"], sort_keys=True))
        result = response.get("result") or {}
        structured = result.get("structuredContent") if isinstance(result, dict) else {}
        if not isinstance(structured, dict):
            structured = {}
        content = result.get("content") if isinstance(result, dict) else []
        text = ""
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                text = str(first.get("text") or "")
        return MCPToolResponse(tool=name, structured=structured, text=text, stderr_preview=stderr_preview)

    async def _send(self, proc: asyncio.subprocess.Process, request_id: int, method: str, params: Dict[str, Any]) -> None:
        assert proc.stdin is not None
        payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
        proc.stdin.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
        await proc.stdin.drain()

    async def _notify(self, proc: asyncio.subprocess.Process, method: str, params: Dict[str, Any]) -> None:
        assert proc.stdin is not None
        payload = {"jsonrpc": "2.0", "method": method, "params": params}
        proc.stdin.write(json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n")
        await proc.stdin.drain()

    async def _read_response(self, proc: asyncio.subprocess.Process, request_id: int) -> Dict[str, Any]:
        assert proc.stdout is not None
        while True:
            line = await proc.stdout.readline()
            if not line:
                raise MCPProcessError("MCP server closed stdout before response")
            try:
                payload = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as exc:
                raise MCPProcessError(f"MCP server emitted invalid JSON: {exc}") from exc
            if payload.get("id") == request_id:
                return payload


class ExternalMCPAnchorClient:
    def __init__(self, *, config: Optional[AxiomConfig] = None) -> None:
        self.config = config or load_config()
        self.max_results = self.config.int("mcp.max_anchor_results", 8, low=1, high=50)
        self.timeout_s = self.config.float("mcp.anchor_timeout_seconds", 20.0, low=1.0, high=180.0)

    async def fetch_anchor_blocks(self, query: str, *, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        if not self.config.bool("mcp.enabled", True) or not self.config.bool("mcp.anchor_process_enabled", True):
            return []
        query = str(query or "").strip()
        if not query:
            return []
        per_tool_limit = max(1, min(limit or self.max_results, self.max_results))
        client = MCPStdioClient(config=self.config)
        tool_args = {"query": query, "limit": per_tool_limit, "timeout_seconds": self.timeout_s}
        calls = [
            client.call_tool("anchor.wikipedia", tool_args),
            client.call_tool("anchor.news", tool_args),
            client.call_tool("anchor.scholar", tool_args),
        ]
        brave_key_env = self.config.str("mcp.brave_api_key_env", "BRAVE_SEARCH_API_KEY")
        if os.environ.get(brave_key_env, "").strip():
            calls.append(client.call_tool("anchor.web", tool_args))
        results = await asyncio.gather(*calls, return_exceptions=True)
        blocks: List[Dict[str, Any]] = []
        for result in results:
            if isinstance(result, Exception):
                continue
            blocks.extend(normalize_mcp_blocks(result.structured.get("blocks", []), tool=result.tool))
        return dedupe_blocks(blocks)


def normalize_mcp_blocks(raw_blocks: Any, *, tool: str) -> List[Dict[str, Any]]:
    if not isinstance(raw_blocks, list):
        return []
    blocks: List[Dict[str, Any]] = []
    for raw in raw_blocks:
        if not isinstance(raw, dict):
            continue
        url = str(raw.get("url") or "").strip()
        text = str(raw.get("text") or "").strip()
        if not url or not text:
            continue
        source = str(raw.get("source") or raw.get("domain") or "").strip()
        blocks.append(
            {
                "url": url,
                "domain": normalize_domain(source or url),
                "title": str(raw.get("title") or source or url),
                "text": text,
                "score": float(raw.get("score") or 0.0),
                "topology_class": "MCP_ANCHOR",
                "classification_confidence": 1.0,
                "fetch_mode": "mcp",
                "mcp_tool": tool,
                "trust_tier": raw.get("trust_tier"),
                "metadata": dict(raw.get("metadata") or {}),
            }
        )
    return blocks


def dedupe_blocks(blocks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    unique: List[Dict[str, Any]] = []
    for block in sorted(blocks, key=lambda item: (-float(item.get("score") or 0.0), str(item.get("url") or ""))):
        key = str(block.get("url") or "")
        if key in seen:
            continue
        seen.add(key)
        unique.append(block)
    return unique


def resolve_mcp_command(config: AxiomConfig) -> List[str]:
    configured = config.str("mcp.server_command", "auto").strip()
    if configured and configured.lower() != "auto":
        return shlex.split(configured, posix=os.name != "nt")
    release_root = config.path_value("paths.release_root", ROOT / "Releases-x64")
    candidates = [
        release_root / "compiled" / "binaries" / "Linux64" / "tag-mcp",
        release_root / "tag-mcp",
        release_root / "compiled" / "binaries" / "Winx64" / "tag-mcp.exe",
        release_root / "tag-mcp.exe",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return [str(candidate)]
    go = shutil.which("go")
    if go and (ROOT / "cmd" / "tag-mcp" / "main.go").exists():
        return [go, "run", "./cmd/tag-mcp"]
    return []


def mcp_process_env(config: AxiomConfig) -> Dict[str, str]:
    env = os.environ.copy()
    env.setdefault("AXIOM_ROOT", str(ROOT))
    env.setdefault("AXIOM_CONFIG_TOML", str(config.path))
    python = config.str("mcp.python", "auto").strip()
    if python and python.lower() != "auto":
        env.setdefault("AXIOM_MCP_PYTHON", python)
    return env


def normalize_domain(raw: str) -> str:
    value = str(raw or "").strip().lower()
    if "://" in value:
        value = value.split("://", 1)[1]
    return value.split("/", 1)[0].split("?", 1)[0].strip(".")
