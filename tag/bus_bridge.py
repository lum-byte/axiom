"""
tag/bus_bridge.py
=================
Native producer bridge for AXIOM's typed event bus.

Go, C, CUDA/C, and Rust components do not import Python objects directly and
they do not publish private wire shapes. They send primitive JSON/msgpack maps
to this bridge. The bridge rebuilds the canonical dataclass from
signal_kernel.contracts, then emits it through tag.crawler_bus.CrawlerBus.

Protocol
--------
Each request is one JSON line:

    {"topic":"signal_extracted","component":"preparser.signal_extractor","payload":{...}}

Each response is one JSON line:

    {"ok":true,"topic":"signal_extracted","schema":"SignalExtractedEvent"}

The bridge can run over a Unix socket on Linux or a localhost TCP socket in
Windows development mode. Kafka mode gives cross-process durability. Degraded
mode remains in-process and is intended for tests/dev only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import msgpack

from tag.crawler_bus import BUS, TOPIC_REGISTRY, event_from_payload, is_bus_started


DEFAULT_UNIX_SOCKET = str(Path(os.environ.get("AXIOM_TMP_DIR", ".axiom_runtime/tmp")) / "axiom_bus_bridge.sock")
DEFAULT_TCP_HOST = "127.0.0.1"
DEFAULT_TCP_PORT = 8765


@dataclass(frozen=True)
class BridgeRequest:
    topic: str
    component: str
    payload: Dict[str, Any]

    @classmethod
    def from_mapping(cls, raw: Dict[str, Any]) -> "BridgeRequest":
        topic = raw.get("topic")
        component = raw.get("component")
        payload = raw.get("payload")
        if not isinstance(topic, str) or not topic:
            raise ValueError("bridge request requires non-empty string field 'topic'.")
        if not isinstance(component, str) or not component:
            raise ValueError("bridge request requires non-empty string field 'component'.")
        if not isinstance(payload, dict):
            raise ValueError("bridge request requires object field 'payload'.")
        return cls(topic=topic, component=component, payload=payload)


class NativeBusBridge:
    """
    Emits primitive native payloads through the canonical CrawlerBus.

    The bridge intentionally performs no schema branching beyond topic lookup.
    The registry and dataclass constructors remain the source of truth.
    """

    def __init__(self, *, start_bus: bool = True) -> None:
        self._start_bus = start_bus
        self._started = False
        self._emitters: Dict[tuple[str, str], Any] = {}

    async def start(self) -> None:
        if self._started:
            return
        if self._start_bus and not is_bus_started():
            await BUS.start()
        self._started = True

    async def stop(self) -> None:
        if self._start_bus and is_bus_started():
            await BUS.stop()
        self._started = False
        self._emitters.clear()

    async def emit_payload(self, *, topic: str, component: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self._started:
            await self.start()
        event = event_from_payload(topic, payload)
        schema = TOPIC_REGISTRY[topic]
        key = (topic, component)
        emitter = self._emitters.get(key)
        if emitter is None:
            emitter = await BUS.emitter(topic=topic, component=component, schema=schema)
            self._emitters[key] = emitter
        await emitter.emit(event)
        return {
            "ok": True,
            "topic": topic,
            "component": component,
            "schema": type(event).__name__,
        }

    async def emit_json_line(self, line: bytes) -> bytes:
        try:
            raw = json.loads(line.decode("utf-8"))
            req = BridgeRequest.from_mapping(raw)
            result = await self.emit_payload(
                topic=req.topic,
                component=req.component,
                payload=req.payload,
            )
            return _json_line(result)
        except Exception as exc:  # noqa: BLE001 - bridge returns structured error
            return _json_line({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })

    async def emit_msgpack(self, body: bytes) -> bytes:
        try:
            raw = msgpack.unpackb(body, raw=False, strict_map_key=False)
            req = BridgeRequest.from_mapping(raw)
            result = await self.emit_payload(
                topic=req.topic,
                component=req.component,
                payload=req.payload,
            )
            return msgpack.packb(result, use_bin_type=True)
        except Exception as exc:  # noqa: BLE001 - bridge returns structured error
            return msgpack.packb({
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc),
            }, use_bin_type=True)


class BridgeServer:
    """JSONL socket server for native producers."""

    def __init__(
        self,
        *,
        bridge: Optional[NativeBusBridge] = None,
        unix_socket: Optional[str] = DEFAULT_UNIX_SOCKET,
        host: str = DEFAULT_TCP_HOST,
        port: int = DEFAULT_TCP_PORT,
    ) -> None:
        self._bridge = bridge or NativeBusBridge()
        self._unix_socket = unix_socket
        self._host = host
        self._port = port
        self._server: Optional[asyncio.AbstractServer] = None

    async def start(self) -> None:
        await self._bridge.start()
        if self._unix_socket and os.name != "nt":
            path = Path(self._unix_socket)
            if path.exists():
                path.unlink()
            self._server = await asyncio.start_unix_server(self._handle_client, path=str(path))
        else:
            self._server = await asyncio.start_server(self._handle_client, self._host, self._port)

    async def serve_forever(self) -> None:
        if self._server is None:
            await self.start()
        assert self._server is not None
        async with self._server:
            await self._server.serve_forever()

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        await self._bridge.stop()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                writer.write(await self._bridge.emit_json_line(line))
                await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()


def _json_line(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


async def _stdin_jsonl() -> int:
    bridge = NativeBusBridge()
    await bridge.start()
    try:
        loop = asyncio.get_running_loop()
        while True:
            line = await loop.run_in_executor(None, os.sys.stdin.buffer.readline)
            if not line:
                return 0
            os.sys.stdout.buffer.write(await bridge.emit_json_line(line))
            os.sys.stdout.buffer.flush()
    finally:
        await bridge.stop()


async def _serve(args: argparse.Namespace) -> int:
    server = BridgeServer(
        unix_socket=args.unix_socket,
        host=args.host,
        port=args.port,
    )
    await server.serve_forever()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="AXIOM native bus bridge")
    parser.add_argument("--stdin-jsonl", action="store_true", help="read bridge requests from stdin")
    parser.add_argument("--unix-socket", default=DEFAULT_UNIX_SOCKET, help="Unix socket path")
    parser.add_argument("--host", default=DEFAULT_TCP_HOST, help="TCP host for Windows/dev mode")
    parser.add_argument("--port", type=int, default=DEFAULT_TCP_PORT, help="TCP port for Windows/dev mode")
    args = parser.parse_args()
    if args.stdin_jsonl:
        return asyncio.run(_stdin_jsonl())
    return asyncio.run(_serve(args))


if __name__ == "__main__":
    raise SystemExit(main())
