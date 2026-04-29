from __future__ import annotations

import json
from pathlib import Path

import pytest

from tag.crawler.fetcher import FetchMode, HtmlCaptureSink, RawFetchEvent


def raw_event(url: str, body: bytes, *, content_type: str = "text/html; charset=utf-8") -> RawFetchEvent:
    return RawFetchEvent(
        url=url,
        raw_bytes=body,
        status_code=200,
        headers={"content-type": content_type},
        fetch_latency=0.01,
        fetch_mode=FetchMode.STATIC,
        is_robots_txt=False,
        is_sitemap=False,
        topology_hint="GENERIC_HTML",
        run_id="run-1",
        manifest_id="manifest-1",
        byte_count=len(body),
    )


@pytest.mark.asyncio
async def test_html_capture_sink_saves_bounded_html_samples(tmp_path: Path) -> None:
    sink = HtmlCaptureSink(capture_dir=tmp_path, limit=2, max_bytes=64)
    await sink.initialize()

    first = await sink.capture(raw_event("https://example.com/a", b"<!doctype html><html><body>alpha</body></html>"))
    second = await sink.capture(raw_event("https://docs.example.com/b", b"<html><body>beta</body></html>"))
    third = await sink.capture(raw_event("https://third.example/c", b"<html><body>gamma</body></html>"))

    assert first is not None and first.exists()
    assert second is not None and second.exists()
    assert third is None
    assert first.suffix == ".html"
    assert "example.com" in first.name
    assert sink.stats["saved"] == 2
    manifest_lines = (tmp_path / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(manifest_lines) == 2
    manifest = [json.loads(line) for line in manifest_lines]
    assert manifest[0]["watermark"] == "axiom.fetch.html_capture.v1"
    assert manifest[0]["byte_count"] <= 64


@pytest.mark.asyncio
async def test_html_capture_sink_skips_non_html_payloads(tmp_path: Path) -> None:
    sink = HtmlCaptureSink(capture_dir=tmp_path, limit=10)
    await sink.initialize()

    captured = await sink.capture(raw_event("https://api.example.com/data", b'{"ok": true}', content_type="application/json"))

    assert captured is None
    assert not list(tmp_path.glob("*.html"))
    assert not (tmp_path / "manifest.jsonl").exists()
