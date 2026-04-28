"""
tag/interface.py
================
The only public AXIOM runtime surface.

Commands:
    search | query
    fetch  | URL
    learn  | domain
    status |
    quit   |
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Protocol
from urllib.parse import urlparse

from signal_kernel.contracts import InterfaceRequest, InterfaceResponse, SystemStatus, new_run_id
from tag.cold_start import ColdStart


COMMAND_RE = re.compile(r"^\s*(search|fetch|learn|status|quit)\s*\|\s*(.*?)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedCommand:
    command: str
    payload: str


@dataclass(frozen=True)
class HistoryItem:
    command: str
    payload: str
    status: str
    run_id: str
    created_unix: int
    latency_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class InterfaceMetrics:
    handled: int = 0
    errors: int = 0
    accepted: int = 0
    empty: int = 0
    total_latency_ms: float = 0.0
    by_command: Dict[str, int] = field(default_factory=dict)

    def record(self, command: str, status: str, latency_ms: float) -> None:
        self.handled += 1
        self.total_latency_ms += latency_ms
        self.by_command[command] = self.by_command.get(command, 0) + 1
        if status == "error":
            self.errors += 1
        elif status == "accepted":
            self.accepted += 1
        elif status == "empty":
            self.empty += 1

    def to_dict(self) -> Dict[str, Any]:
        avg = self.total_latency_ms / self.handled if self.handled else 0.0
        return {
            "handled": self.handled,
            "errors": self.errors,
            "accepted": self.accepted,
            "empty": self.empty,
            "avg_latency_ms": avg,
            "by_command": dict(sorted(self.by_command.items())),
        }


def parse_command(line: str) -> ParsedCommand:
    match = COMMAND_RE.match(line)
    if not match:
        raise ValueError("command must use '<search|fetch|learn|status|quit> | <payload>' syntax")
    return ParsedCommand(command=match.group(1).upper(), payload=match.group(2))


class AxiomInterface:
    def __init__(self, *, store_dir: Path = Path("store"), history_limit: int = 256) -> None:
        self.store_dir = store_dir
        self.learned_domains: set[str] = set()
        self.queued_work: List[Dict[str, Any]] = []
        self.cold_start = ColdStart(store_dir=store_dir)
        self.history: Deque[HistoryItem] = collections.deque(maxlen=history_limit)
        self.metrics = InterfaceMetrics()

    async def handle_line(self, line: str) -> InterfaceResponse:
        started = time.perf_counter()
        parsed = parse_command(line)
        req = InterfaceRequest(query_type=parsed.command, payload=parsed.payload, run_id=str(new_run_id()))
        response = await self.handle_request(req)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(parsed, response, latency_ms)
        return response

    async def handle_request(self, req: InterfaceRequest) -> InterfaceResponse:
        if req.query_type == "STATUS":
            return self._status(req)
        if req.query_type == "QUIT":
            return InterfaceResponse(run_id=req.run_id, status="ok", message="quit accepted", data={"quit": True})
        if req.query_type == "LEARN":
            return self._learn(req)
        if req.query_type == "FETCH":
            return self._fetch(req)
        if req.query_type == "SEARCH":
            return self._search(req)
        return InterfaceResponse(run_id=req.run_id, status="error", message="unknown command", data={})

    async def handle_json(self, payload: Dict[str, Any]) -> InterfaceResponse:
        if "command" in payload:
            return await self.handle_line(f"{payload.get('command')} | {payload.get('payload', '')}")
        query_type = str(payload.get("query_type", "")).upper()
        req = InterfaceRequest(query_type=query_type, payload=str(payload.get("payload", "")), run_id=str(payload.get("run_id") or new_run_id()))
        started = time.perf_counter()
        response = await self.handle_request(req)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self._record(ParsedCommand(query_type, req.payload), response, latency_ms)
        return response

    def _status(self, req: InterfaceRequest) -> InterfaceResponse:
        status = SystemStatus(
            run_id=req.run_id,
            bus_started=False,
            bus_mode="unstarted",
            store_ready=self.store_dir.exists(),
            index_daemon_ready=False,
            cold_start_complete=self.store_dir.exists(),
            learned_domains=len(self.learned_domains),
            queued_work_items=len(self.queued_work),
        )
        return InterfaceResponse(
            run_id=req.run_id,
            status="ok",
            message="status",
            data={
                **status.__dict__,
                "metrics": self.metrics.to_dict(),
                "recent": [item.to_dict() for item in list(self.history)[-10:]],
            },
        )

    def _learn(self, req: InterfaceRequest) -> InterfaceResponse:
        domain = self._normalize_domain(req.payload)
        if not domain:
            return InterfaceResponse(run_id=req.run_id, status="error", message="learn requires a domain", data={"payload": req.payload})
        self.learned_domains.add(domain)
        self._enqueue({"type": "learn", "domain": domain, "run_id": req.run_id})
        return InterfaceResponse(run_id=req.run_id, status="accepted", message="learning queued", data={"domain": domain})

    def _fetch(self, req: InterfaceRequest) -> InterfaceResponse:
        url = req.payload.strip()
        if not self._valid_http_url(url):
            return InterfaceResponse(run_id=req.run_id, status="error", message="fetch requires http(s) URL", data={"url": url})
        self._enqueue({"type": "fetch", "url": url, "run_id": req.run_id})
        return InterfaceResponse(run_id=req.run_id, status="accepted", message="fetch queued", data={"url": url, "fetch_mode": "static"})

    def _search(self, req: InterfaceRequest) -> InterfaceResponse:
        query = req.payload.strip()
        if not query:
            return InterfaceResponse(run_id=req.run_id, status="error", message="query is empty", data={})
        candidates = self._candidate_sources(query)
        if not candidates:
            self._enqueue({"type": "learn_from_query", "query": query, "run_id": req.run_id})
            return InterfaceResponse(run_id=req.run_id, status="empty", message="no learned topology candidates; learning queued", data={"query": query, "sources": []})
        return InterfaceResponse(run_id=req.run_id, status="ok", message=self._synthesize(query, candidates), data={"query": query, "sources": candidates})

    def _candidate_sources(self, query: str) -> List[str]:
        terms = {term for term in re.split(r"\W+", query.lower()) if term}
        ranked: List[tuple[int, str]] = []
        for domain in self.learned_domains:
            score = sum(1 for term in terms if term in domain)
            ranked.append((score, domain))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [f"https://{domain}/" for _, domain in ranked[:5]]

    def _synthesize(self, query: str, candidates: List[str]) -> str:
        return f"AXIOM routed '{query}' through {len(candidates)} learned source(s)."

    def _enqueue(self, item: Dict[str, Any]) -> None:
        item.setdefault("created_unix", int(time.time()))
        self.queued_work.append(item)

    def _record(self, parsed: ParsedCommand, response: InterfaceResponse, latency_ms: float) -> None:
        self.metrics.record(parsed.command, response.status, latency_ms)
        self.history.append(
            HistoryItem(
                command=parsed.command,
                payload=parsed.payload,
                status=response.status,
                run_id=response.run_id,
                created_unix=int(time.time()),
                latency_ms=latency_ms,
            )
        )

    @staticmethod
    def _valid_http_url(raw: str) -> bool:
        parsed = urlparse(raw)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    @staticmethod
    def _normalize_domain(raw: str) -> str:
        raw = raw.strip().lower()
        if not raw:
            return ""
        if "://" in raw:
            parsed = urlparse(raw)
            raw = parsed.netloc
        raw = raw.split("/")[0].strip(".")
        if not raw or "." not in raw or any(ch.isspace() for ch in raw):
            return ""
        return raw


class InterfaceTransport(Protocol):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...


class JsonLineCodec:
    """Strict JSONL codec for TUI and test clients."""

    @staticmethod
    def encode(response: InterfaceResponse) -> bytes:
        return (json.dumps(response.__dict__, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")

    @staticmethod
    def decode_line(line: bytes) -> str:
        raw = line.decode("utf-8").strip()
        if not raw:
            raise ValueError("empty command line")
        if raw.startswith("{"):
            obj = json.loads(raw)
            command = obj.get("command")
            payload = obj.get("payload", "")
            if not isinstance(command, str):
                raise ValueError("JSON command requires string field 'command'")
            return f"{command} | {payload}"
        return raw


class InterfaceSocketServer:
    """
    JSONL socket server for the public AXIOM interface.

    Linux production uses Unix sockets. Windows development uses TCP because
    Unix-socket behavior differs across Python/Windows versions.
    """

    def __init__(
        self,
        *,
        interface: Optional[AxiomInterface] = None,
        unix_socket: Path = Path("/tmp/axiom_interface.sock"),
        host: str = "127.0.0.1",
        port: int = 8766,
    ) -> None:
        self.interface = interface or AxiomInterface()
        self.unix_socket = unix_socket
        self.host = host
        self.port = port
        self.server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        if os.name != "nt":
            if self.unix_socket.exists():
                self.unix_socket.unlink()
            self.server = await asyncio.start_unix_server(self._handle_client, path=str(self.unix_socket))
        else:
            self.server = await asyncio.start_server(self._handle_client, self.host, self.port)

    async def stop(self) -> None:
        if self.server is not None:
            self.server.close()
            await self.server.wait_closed()
            self.server = None
        if os.name != "nt" and self.unix_socket.exists():
            self.unix_socket.unlink()

    async def serve_forever(self) -> None:
        if self.server is None:
            await self.start()
        assert self.server is not None
        async with self.server:
            await self.server.serve_forever()

    async def _handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                try:
                    command_line = JsonLineCodec.decode_line(line)
                    response = await self.interface.handle_line(command_line)
                except Exception as exc:  # noqa: BLE001 - public boundary returns structured error
                    response = InterfaceResponse(
                        run_id=str(new_run_id()),
                        status="error",
                        message=str(exc),
                        data={"error_type": type(exc).__name__},
                    )
                writer.write(JsonLineCodec.encode(response))
                await writer.drain()
                if response.data.get("quit"):
                    break
        finally:
            writer.close()
            await writer.wait_closed()


async def serve_stdio() -> int:
    interface = AxiomInterface()
    loop = asyncio.get_running_loop()
    while True:
        line = await loop.run_in_executor(None, os.sys.stdin.readline)
        if not line:
            return 0
        try:
            response = await interface.handle_line(line)
        except Exception as exc:  # noqa: BLE001 - public interface returns structured error
            response = InterfaceResponse(run_id=str(new_run_id()), status="error", message=str(exc), data={"error_type": type(exc).__name__})
        print(json.dumps(response.__dict__, sort_keys=True), flush=True)
        if response.data.get("quit"):
            return 0


async def serve_socket() -> int:
    server = InterfaceSocketServer()
    await server.serve_forever()
    return 0


def main() -> int:
    if "--socket" in os.sys.argv:
        return asyncio.run(serve_socket())
    return asyncio.run(serve_stdio())


if __name__ == "__main__":
    raise SystemExit(main())
