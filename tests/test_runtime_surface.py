from __future__ import annotations

import asyncio
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("AXIOM_BUS_HMAC_KEY", "1" * 64)

from tag.cold_start import ColdStart
from signal_kernel.contracts import FetchAnomalyEvent, FetchMode, SignalExtractedEvent, SurpriseEvent, new_run_id
from tag.index_daemon import GRADIENT_PRIORITY_HIGH, IndexDaemon, run_once_for_test
from tag.interface import AxiomInterface, InterfaceSocketServer, JsonLineCodec, parse_command


class RuntimeSurfaceTests(unittest.TestCase):
    def test_parse_command(self) -> None:
        parsed = parse_command("search | hello")
        self.assertEqual(parsed.command, "SEARCH")
        self.assertEqual(parsed.payload, "hello")

    def test_cold_start_creates_store(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            result = ColdStart(store_dir=Path(td)).run()
            self.assertTrue(result.ok)
            self.assertTrue((Path(td) / "phase_states.mmap").exists())
            self.assertEqual(len(result.store_status), 4)
            self.assertTrue(all(item["ok"] for item in result.store_status))

    def test_index_daemon_run_once(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            status = asyncio.run(run_once_for_test(Path(td)))
            self.assertEqual(status["stats"]["signal_events"], 1)

    def test_index_daemon_dispatch_drafts_recipe_and_prioritizes_gradients(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                daemon = IndexDaemon(store_dir=Path(td))
                try:
                    await daemon.dispatch(
                        SignalExtractedEvent(
                            url="https://example.com/docs",
                            topology_class="SAAS_DOCS",
                            signal_type="code",
                            byte_count=1000,
                            token_count=100,
                            signal_density=0.92,
                            zone_count=3,
                            source_component="test",
                            run_id=str(new_run_id()),
                        )
                    )
                    self.assertEqual(daemon.stats.drafted_recipes, 1)
                    self.assertTrue((Path(td) / "recipe_registry.mmap").exists())
                    await daemon.dispatch(
                        SurpriseEvent(
                            topology_class="SAAS_DOCS",
                            surprise_score=0.9,
                            theta_surprise=0.5,
                            dissolve_triggered=True,
                            contributing_signals={"density": 0.9},
                            current_phase=2,
                            run_id=str(new_run_id()),
                            timestamp="2026-04-28T00:00:00Z",
                        )
                    )
                    batch = daemon.pop_gradient_batch()
                    self.assertEqual(batch[0]["priority"], GRADIENT_PRIORITY_HIGH)
                finally:
                    daemon.close()

        asyncio.run(run())

    def test_index_daemon_fetch_anomaly_dispatch(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                daemon = IndexDaemon(store_dir=Path(td))
                try:
                    await daemon.dispatch(
                        FetchAnomalyEvent(
                            url="https://example.com",
                            fetch_mode=FetchMode.STATIC,
                            status_code=429,
                            anomaly_type="rate_limited",
                            run_id=str(new_run_id()),
                            manifest_id="manifest",
                        )
                    )
                    self.assertEqual(daemon.stats.fetch_anomalies, 1)
                    self.assertEqual(daemon.pop_gradient_batch()[0]["priority"], GRADIENT_PRIORITY_HIGH)
                finally:
                    daemon.close()

        asyncio.run(run())

    def test_interface_flow(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                interface = AxiomInterface(store_dir=Path(td))
                learn = await interface.handle_line("learn | example.com")
                self.assertEqual(learn.status, "accepted")
                search = await interface.handle_line("search | docs")
                self.assertEqual(search.status, "ok")
                status = await interface.handle_line("status |")
                self.assertEqual(status.status, "ok")
                self.assertGreaterEqual(status.data["metrics"]["handled"], 2)
                self.assertTrue(status.data["recent"])

        asyncio.run(run())

    def test_interface_json_request_and_validation(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                interface = AxiomInterface(store_dir=Path(td))
                bad = await interface.handle_line("fetch | ftp://example.com/file")
                self.assertEqual(bad.status, "error")
                learn = await interface.handle_json({"query_type": "LEARN", "payload": "https://docs.example.com/path"})
                self.assertEqual(learn.status, "accepted")
                search = await interface.handle_json({"command": "search", "payload": "docs"})
                self.assertEqual(search.status, "ok")

        asyncio.run(run())

    def test_json_line_codec_plain_and_json(self) -> None:
        self.assertEqual(JsonLineCodec.decode_line(b"status |\n"), "status |")
        self.assertEqual(JsonLineCodec.decode_line(b'{"command":"learn","payload":"example.com"}\n'), "learn | example.com")

    def test_interface_tcp_socket_server(self) -> None:
        async def run() -> None:
            with tempfile.TemporaryDirectory() as td:
                server = InterfaceSocketServer(interface=AxiomInterface(store_dir=Path(td)), port=8876)
                await server.start()
                try:
                    reader, writer = await asyncio.open_connection("127.0.0.1", 8876)
                    writer.write(b"status |\n")
                    await writer.drain()
                    line = await reader.readline()
                    self.assertIn(b'"status":"ok"', line)
                    writer.close()
                    await writer.wait_closed()
                finally:
                    await server.stop()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
