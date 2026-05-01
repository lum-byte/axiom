"""Python-side worker for the external AXIOM TAG MCP process.

The Go server owns the MCP transport and process boundary.  This worker owns
the Python TAG runtime so the MCP process can expose real TAG operations without
duplicating crawler, DIC, or VERITAS internals in Go.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from tag.config import load_config
from tag.dic.assembler import DirectlyInjectContextAssembler
from tag.dic.gbnf_dsl import QueryExpansionEngine
from tag.dic.injector import DirectContextInjector
from tag.interface import AxiomInterface
from tag.veritas.classifier import VeritasEngine


class MCPWorkerError(RuntimeError):
    pass


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="AXIOM TAG MCP Python worker")
    parser.add_argument("tool", choices=["search", "status", "expand", "veritas", "inject_context", "health"])
    args = parser.parse_args(argv)
    try:
        payload = read_json_stdin()
        result = asyncio.run(dispatch(args.tool, payload))
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
        return 0
    except Exception as exc:  # noqa: BLE001 - process boundary must serialize failure
        sys.stdout.write(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "tool": args.tool,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )
        return 1


def read_json_stdin() -> Dict[str, Any]:
    raw = sys.stdin.read()
    if not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MCPWorkerError(f"invalid JSON worker payload: {exc}") from exc
    if not isinstance(payload, dict):
        raise MCPWorkerError("worker payload must be a JSON object")
    return payload


async def dispatch(tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    if tool == "search":
        return await run_search(payload)
    if tool == "status":
        return await run_status(payload)
    if tool == "expand":
        return run_expand(payload)
    if tool == "veritas":
        return await run_veritas(payload)
    if tool == "inject_context":
        return await run_inject_context(payload)
    if tool == "health":
        return run_health(payload)
    raise MCPWorkerError(f"unknown worker tool: {tool}")


async def run_search(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or payload.get("q") or "").strip()
    if not query:
        raise MCPWorkerError("tag.search requires query")
    fanout = bounded_int(payload.get("fanout", payload.get("parallelism", payload.get("swarm", payload.get("workers", 10)))), 1, 500)
    depth = bounded_int(payload.get("depth", 3), 1, 32)
    exp = bounded_int(payload.get("exp", payload.get("expansion", 0)), 0, 100)
    command = str(payload.get("command") or "").strip()
    if not command:
        parts = [f"search | fanout -{fanout}", f"depth -{depth}"]
        if exp > 0:
            parts.append(f"exp -{exp}")
        parts.append(query)
        command = " | ".join(parts)
    started = time.perf_counter()
    response = await handle_interface_line(command)
    elapsed_s = time.perf_counter() - started
    data = dict(response.get("data") or {})
    answer = data.get("answer") if isinstance(data.get("answer"), dict) else {}
    return {
        "status": response.get("status"),
        "message": response.get("message"),
        "query": data.get("query", query),
        "answer": answer,
        "source": answer.get("source") if isinstance(answer, dict) else None,
        "blocks": data.get("blocks", []),
        "citations": (data.get("direct_inject_context") or {}).get("citations", []),
        "query_expansion": data.get("query_expansion", {}),
        "veritas": data.get("veritas", {}),
        "crawl_fanout": data.get("crawl_fanout", data.get("crawl_swarm", {})),
        "crawl_swarm": data.get("crawl_swarm", {}),
        "time_taken_s": round(elapsed_s, 3),
        "run_id": response.get("run_id"),
        "tool": "tag.search",
    }


async def run_status(payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    response = await handle_interface_line("status |")
    return {
        "status": response.get("status"),
        "message": response.get("message"),
        "data": response.get("data", {}),
        "run_id": response.get("run_id"),
        "tool": "tag.status",
    }


def run_expand(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    if not query:
        raise MCPWorkerError("tag.expand requires query")
    limit = bounded_int(payload.get("limit", payload.get("exp", 10)), 0, 100)
    expansion = QueryExpansionEngine().expand(query, requested_limit=limit)
    return {
        "status": "ok",
        "message": f"expanded '{query}' into {len(expansion.directives)} typed directive(s)",
        "query": query,
        "expansion": expansion.to_dict(),
        "tool": "tag.expand",
    }


async def run_veritas(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    blocks = payload.get("blocks", [])
    if not query:
        raise MCPWorkerError("tag.veritas requires query")
    if not isinstance(blocks, list):
        raise MCPWorkerError("tag.veritas blocks must be an array")
    result = await VeritasEngine().classify(query, [dict(item) for item in blocks if isinstance(item, dict)])
    return {
        "status": "ok",
        "message": f"VERITAS classified {result.get('low_confidence', 0)} low-confidence block(s)",
        "query": query,
        "veritas": result,
        "tool": "tag.veritas",
    }


async def run_inject_context(payload: Dict[str, Any]) -> Dict[str, Any]:
    query = str(payload.get("query") or "").strip()
    blocks = payload.get("blocks", [])
    if not query:
        raise MCPWorkerError("tag.inject_context requires query")
    if not isinstance(blocks, list):
        raise MCPWorkerError("tag.inject_context blocks must be an array")
    exp = bounded_int(payload.get("exp", payload.get("expansion", 0)), 0, 100)
    expansion = QueryExpansionEngine().expand(query, requested_limit=exp)
    block_dicts = [dict(item) for item in blocks if isinstance(item, dict)]
    veritas = await VeritasEngine().classify(query, block_dicts)
    context = DirectlyInjectContextAssembler().assemble(
        query=query,
        ranked_blocks=block_dicts,
        expansion=expansion,
        veritas=veritas,
    )
    injected = DirectContextInjector().format(context)
    return {
        "status": "ok",
        "message": context.answer,
        "query": query,
        "answer": {
            "text": context.answer,
            "structured": context.structured_answer,
            "source": context.citations[0] if context.citations else None,
            "sources": list(context.structured_answer.get("citation_spine") or context.citations[:8]),
        },
        "direct_inject_context": injected.json_payload,
        "injection_text": injected.text,
        "veritas": veritas,
        "query_expansion": expansion.to_dict(),
        "tool": "tag.inject_context",
    }


def run_health(payload: Dict[str, Any]) -> Dict[str, Any]:
    del payload
    cfg = load_config()
    return {
        "status": "ok",
        "message": "AXIOM TAG MCP worker is importable",
        "config_path": str(cfg.path),
        "config_errors": list(cfg.errors),
        "cwd": str(Path.cwd()),
        "tool": "health",
    }


async def handle_interface_line(command: str) -> Dict[str, Any]:
    interface = AxiomInterface()
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture), contextlib.redirect_stderr(capture):
        try:
            response = await interface.handle_line(command)
        finally:
            with contextlib.suppress(Exception):
                await interface.runtime.close()
    result = {
        "run_id": response.run_id,
        "status": response.status,
        "message": response.message,
        "data": response.data,
        "completed_at": response.completed_at,
    }
    captured = [line for line in capture.getvalue().splitlines() if line.strip()]
    if captured:
        result["_captured_internal_output"] = {"lines": len(captured), "preview": captured[:8]}
    return result


def bounded_int(value: Any, low: int, high: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = low
    return max(low, min(high, parsed))


if __name__ == "__main__":
    raise SystemExit(main())
