#!/usr/bin/env python3
"""
Run repeated native AXIOM inference batches through axiom_infer.py.

This probe is intentionally outside normal pytest discovery because burn-in
runs should be launched deliberately:

    .venv/bin/python tests/probes/native_burnin.py --cycles 1000
    .venv/bin/python tests/probes/native_burnin.py --duration-seconds 600
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
GBNF_PATH = Path(__file__).with_suffix(".gbnf")


@dataclass(frozen=True)
class BurninJob:
    query: str
    workers: int
    depth: int
    cycles: int


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    offset: int


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=1000, help="Native calls per batch.")
    parser.add_argument("--duration-seconds", type=float, default=0.0, help="Keep running batches until this wall time is reached.")
    parser.add_argument("--query", default="find me latest AI news")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dsl", help="Inline burn-in DSL. Duplicate query entries are rejected before execution.")
    parser.add_argument("--dsl-file", type=Path, help="Path to a burn-in DSL file.")
    args = parser.parse_args()

    if args.duration_seconds < 0:
        raise ValueError("--duration-seconds must be non-negative")
    jobs = resolve_jobs(args)
    assert_unique_queries(jobs)

    started = time.monotonic()
    aggregate: Dict[str, Any] = {
        "ok": True,
        "probe": "axiom_native_burnin",
        "dsl": {
            "enabled": bool(args.dsl or args.dsl_file),
            "grammar": str(GBNF_PATH),
            "unique_query_guard": True,
        },
        "batch_cycles": jobs[0].cycles if len(jobs) == 1 else None,
        "duration_seconds_requested": args.duration_seconds,
        "query": jobs[0].query if len(jobs) == 1 else None,
        "workers": jobs[0].workers if len(jobs) == 1 else None,
        "depth": jobs[0].depth if len(jobs) == 1 else None,
        "jobs_requested": len(jobs),
        "jobs_completed": 0,
        "job_results": [],
        "unique_queries": [job.query for job in jobs],
        "batches": 0,
        "total_cycles": 0,
        "status_counts": {},
        "native_version": None,
        "last_status": None,
        "last_payload": None,
        "duration_exhausted_unique_queries": False,
    }

    for job in jobs:
        payload = run_batch(job)
        status_counts = payload["cycles"]["status_counts"]
        batch_ok = payload["cycles"]["requested"] == job.cycles and status_counts.get("ok", 0) == job.cycles
        aggregate["ok"] = bool(aggregate["ok"] and batch_ok)
        aggregate["batches"] += 1
        aggregate["total_cycles"] += payload["cycles"]["requested"]
        aggregate["native_version"] = payload.get("native_version")
        aggregate["last_status"] = payload.get("last_response", {}).get("json", {}).get("status")
        aggregate["last_payload"] = {
            "request": payload.get("request"),
            "bootstrap_domains": payload.get("bootstrap", {}).get("domains", []),
            "last_response": payload.get("last_response", {}).get("json"),
        }
        aggregate["jobs_completed"] += 1
        aggregate["job_results"].append(
            {
                "query": job.query,
                "workers": job.workers,
                "depth": job.depth,
                "cycles": job.cycles,
                "ok": batch_ok,
                "status_counts": status_counts,
            }
        )
        for status, count in status_counts.items():
            aggregate["status_counts"][status] = aggregate["status_counts"].get(status, 0) + count

        elapsed = time.monotonic() - started
        if args.duration_seconds > 0 and elapsed >= args.duration_seconds:
            break
    else:
        if args.duration_seconds > 0 and (time.monotonic() - started) < args.duration_seconds:
            aggregate["duration_exhausted_unique_queries"] = True

    aggregate["elapsed_seconds"] = round(time.monotonic() - started, 3)

    sys.stdout.write(json.dumps(aggregate, indent=2) + "\n")
    return 0 if aggregate["ok"] else 1


def resolve_jobs(args: argparse.Namespace) -> List[BurninJob]:
    if args.dsl and args.dsl_file:
        raise ValueError("use either --dsl or --dsl-file, not both")
    if args.cycles < 1:
        raise ValueError("--cycles must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.depth < 1:
        raise ValueError("--depth must be at least 1")
    if args.dsl_file:
        dsl_text = args.dsl_file.read_text(encoding="utf-8")
        return parse_burnin_dsl(dsl_text, default_workers=args.workers, default_depth=args.depth, default_cycles=args.cycles)
    if args.dsl:
        return parse_burnin_dsl(args.dsl, default_workers=args.workers, default_depth=args.depth, default_cycles=args.cycles)
    return [BurninJob(query=args.query, workers=args.workers, depth=args.depth, cycles=args.cycles)]


def parse_burnin_dsl(
    text: str,
    *,
    default_workers: int = 10,
    default_depth: int = 2,
    default_cycles: int = 1000,
) -> List[BurninJob]:
    parser = BurninDslParser(
        tokenize_dsl(text),
        default_workers=default_workers,
        default_depth=default_depth,
        default_cycles=default_cycles,
    )
    return parser.parse()


class BurninDslParser:
    def __init__(self, tokens: List[Token], *, default_workers: int, default_depth: int, default_cycles: int) -> None:
        self.tokens = tokens
        self.index = 0
        self.default_workers = default_workers
        self.default_depth = default_depth
        self.default_cycles = default_cycles

    def parse(self) -> List[BurninJob]:
        jobs: List[BurninJob] = []
        wrapped = self.peek_is("ident", "burnin")
        if wrapped:
            self.take("ident", "burnin")
            self.take("symbol", "{")
        while self.peek() is not None and not (wrapped and self.peek_is("symbol", "}")):
            jobs.append(self.parse_job())
        if wrapped:
            self.take("symbol", "}")
        if self.peek() is not None:
            token = self.peek()
            assert token is not None
            raise ValueError(f"unexpected trailing token at byte {token.offset}: {token.value!r}")
        if not jobs:
            raise ValueError("burn-in DSL must contain at least one query job")
        assert_unique_queries(jobs)
        return jobs

    def parse_job(self) -> BurninJob:
        self.take("ident", "query")
        query = self.take("string").value.strip()
        if not query:
            raise ValueError("query string must not be empty")
        fields: Dict[str, int] = {
            "workers": self.default_workers,
            "depth": self.default_depth,
            "cycles": self.default_cycles,
        }
        seen_fields: set[str] = set()
        while True:
            token = self.peek()
            if token is None:
                raise ValueError("unexpected end of DSL inside query job")
            if token.kind == "symbol" and token.value == ";":
                self.take("symbol", ";")
                break
            if token.kind != "ident":
                raise ValueError(f"expected job field or ';' at byte {token.offset}")
            field = self.take("ident").value.lower()
            if field not in fields:
                raise ValueError(f"unknown burn-in field {field!r}")
            if field in seen_fields:
                raise ValueError(f"duplicate burn-in field {field!r}")
            seen_fields.add(field)
            fields[field] = int(self.take("number").value)
        if fields["workers"] < 1 or fields["depth"] < 1 or fields["cycles"] < 1:
            raise ValueError("workers, depth, and cycles must be positive")
        return BurninJob(query=query, workers=fields["workers"], depth=fields["depth"], cycles=fields["cycles"])

    def peek(self) -> Optional[Token]:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def peek_is(self, kind: str, value: str) -> bool:
        token = self.peek()
        return token is not None and token.kind == kind and token.value.lower() == value.lower()

    def take(self, kind: str, value: Optional[str] = None) -> Token:
        token = self.peek()
        if token is None:
            expected = value if value is not None else kind
            raise ValueError(f"unexpected end of DSL; expected {expected}")
        if token.kind != kind or (value is not None and token.value.lower() != value.lower()):
            expected = value if value is not None else kind
            raise ValueError(f"unexpected token at byte {token.offset}: expected {expected}, got {token.value!r}")
        self.index += 1
        return token


def tokenize_dsl(text: str) -> List[Token]:
    tokens: List[Token] = []
    i = 0
    while i < len(text):
        char = text[i]
        if i == 0 and char == "\ufeff":
            i += 1
            continue
        if char.isspace():
            i += 1
            continue
        if char == "#":
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        if char in "{};":
            tokens.append(Token("symbol", char, i))
            i += 1
            continue
        if char == '"':
            start = i
            value, i = read_quoted_string(text, i)
            tokens.append(Token("string", value, start))
            continue
        if char.isdigit():
            start = i
            while i < len(text) and text[i].isdigit():
                i += 1
            tokens.append(Token("number", text[start:i], start))
            continue
        if char.isalpha() or char == "_":
            start = i
            while i < len(text) and (text[i].isalnum() or text[i] in "_-"):
                i += 1
            tokens.append(Token("ident", text[start:i], start))
            continue
        raise ValueError(f"invalid DSL character at byte {i}: {char!r}")
    return tokens


def read_quoted_string(text: str, offset: int) -> tuple[str, int]:
    assert text[offset] == '"'
    i = offset + 1
    parts: List[str] = []
    while i < len(text):
        char = text[i]
        if char == '"':
            return "".join(parts), i + 1
        if char == "\\":
            i += 1
            if i >= len(text):
                raise ValueError(f"unterminated escape sequence at byte {offset}")
            escaped = text[i]
            if escaped == "n":
                parts.append("\n")
            elif escaped == "t":
                parts.append("\t")
            elif escaped in {'"', "\\"}:
                parts.append(escaped)
            else:
                raise ValueError(f"unsupported escape at byte {i}: {escaped!r}")
        else:
            parts.append(char)
        i += 1
    raise ValueError(f"unterminated string at byte {offset}")


def assert_unique_queries(jobs: List[BurninJob]) -> None:
    seen: Dict[str, str] = {}
    for job in jobs:
        key = normalize_query_key(job.query)
        if key in seen:
            raise ValueError(f"duplicate burn-in query rejected: {job.query!r} duplicates {seen[key]!r}")
        seen[key] = job.query


def normalize_query_key(query: str) -> str:
    return " ".join(query.lower().split())


def run_batch(job: BurninJob) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        str(ROOT / "axiom_infer.py"),
        "--query",
        job.query,
        "--workers",
        str(job.workers),
        "--depth",
        str(job.depth),
        "--cycles",
        str(job.cycles),
        "--summary-only",
        "--compact",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    return json.loads(proc.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
