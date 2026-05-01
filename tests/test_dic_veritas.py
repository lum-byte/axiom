from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tag.crawler.swarm_bridge import crawl_config_from_plan, parse_swarm_search_payload
from tag.dic.assembler import DirectlyInjectContextAssembler
from tag.dic.gbnf_dsl import QUESTION_TYPE_SPECS, QueryExpansionEngine, build_gbnf_grammar
from tag.dic.hybrid_search import HybridFusionRanker
from tag.veritas.classifier import VeritasEngine


ROOT = Path(__file__).resolve().parents[1]


def test_query_dsl_has_at_least_100_question_types_and_grammar() -> None:
    assert len(QUESTION_TYPE_SPECS) >= 100
    grammar = build_gbnf_grammar()
    assert "FACTUAL_DIRECT".lower() in grammar
    assert "RUMOR_PROBE".lower() in grammar
    assert grammar.count("::=") >= 100


def test_swarm_bridge_parses_expansion_directive() -> None:
    query, plan = parse_swarm_search_payload("swarm -10 | depth -2 | exp -10 | recheck | what is google")
    assert query == "what is google"
    assert plan is not None
    assert plan["requested_worker_count"] == 10
    assert plan["depth"] == 2
    assert plan["expansion_count"] == 10
    assert plan["recheck"] is True
    config = crawl_config_from_plan(plan)
    assert config.worker_count == 10


def test_query_expansion_generates_typed_directives() -> None:
    result = QueryExpansionEngine().expand("what is github", requested_limit=10)
    assert result.effective_limit == 10
    assert result.detected_type == "FACTUAL_DIRECT"
    assert len(result.directives) == 11
    assert all(item.query for item in result.directives)
    assert result.grammar_hash


def test_hybrid_ranker_prefers_semantic_definition() -> None:
    blocks = [
        {"url": "https://a.example", "domain": "a.example", "title": "noise", "text": "GitHub pricing docs and random unrelated terms", "score": 2.0, "topology_class": "GENERIC_HTML"},
        {"url": "https://b.example", "domain": "wikipedia.org", "title": "GitHub", "text": "GitHub is a developer platform for storing, managing, and sharing code.", "score": 2.0, "topology_class": "GENERIC_HTML"},
    ]
    ranked = HybridFusionRanker().rank("what is github", blocks)
    assert ranked[0]["url"] == "https://b.example"
    assert ranked[0]["fusion"]["semantic"] > 0


def test_veritas_classifies_low_confidence_against_anchor() -> None:
    async def run() -> None:
        blocks = [
            {"url": "https://en.wikipedia.org/wiki/GitHub", "domain": "en.wikipedia.org", "title": "GitHub", "text": "GitHub is a developer platform for code hosting.", "score": 15.0, "rank": 1},
            {"url": "https://rumor.example/x", "domain": "rumor.example", "title": "rumor", "text": "GitHub is a cooking website.", "score": 1.0, "rank": 2},
        ]
        result = await VeritasEngine().classify("what is github", blocks)
        assert result["low_confidence"] >= 1
        assert result["classifications"]
        assert result["classifications"][0]["label"] in {"RUMOR", "CONTESTED", "LEGACY", "CONFIRMED"}

    import asyncio

    asyncio.run(run())


def test_dic_assembler_outputs_500_word_context_when_context_is_available() -> None:
    text = (
        "GitHub is a proprietary developer platform that lets developers create, store, manage, and share code. "
        "It provides distributed version control through Git, collaboration tools, issue tracking, pull requests, automation, and package hosting. "
        "Developers use it to coordinate software projects, review changes, publish documentation, and maintain release histories. "
    ) * 15
    blocks = [
        {"url": "https://en.wikipedia.org/wiki/GitHub", "domain": "en.wikipedia.org", "title": "GitHub", "text": text, "score": 20.0, "rank": 1, "topology_class": "GENERIC_HTML"},
        {"url": "https://github.blog/example", "domain": "github.blog", "title": "GitHub Blog", "text": text, "score": 10.0, "rank": 2, "topology_class": "NEWS_ARTICLE"},
    ]
    expansion = QueryExpansionEngine().expand("what is github", requested_limit=5)
    context = DirectlyInjectContextAssembler().assemble(query="what is github", ranked_blocks=blocks, expansion=expansion, veritas={"counts": {}})
    word_count = len(context.answer.split())
    assert 500 <= word_count <= 700
    assert context.citations
    assert context.structured_answer["summary"].startswith("GitHub")
    assert context.structured_answer["sections"]
    assert context.structured_answer["citation_spine"]
    assert context.query_trace["expansion_count"] == 5


def test_native_runtime_exports_dic_and_veritas_primitives(tmp_path: Path) -> None:
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("gcc is not installed")
    out = tmp_path / ("axi.dll" if os.name == "nt" else "axi.so")
    cmd = [gcc, "-std=c11", "-O2", "-Wall", "-Wextra"]
    if os.name != "nt":
        cmd.extend(["-fPIC", "-shared"])
    else:
        cmd.append("-shared")
    cmd.extend([str(ROOT / "axiom_runtime" / "axiom_runtime.c"), "-o", str(out)])
    subprocess.run(cmd, check=True, cwd=ROOT)
    lib = ctypes.CDLL(str(out))
    lib.axiom_dic_version.restype = ctypes.c_char_p
    lib.axiom_veritas_version.restype = ctypes.c_char_p
    lib.axiom_dic_lexical_overlap.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.axiom_dic_lexical_overlap.restype = ctypes.c_double
    lib.axiom_veritas_label_score.argtypes = [ctypes.c_double, ctypes.c_double, ctypes.c_int]
    lib.axiom_veritas_label_score.restype = ctypes.c_int
    assert lib.axiom_dic_version().decode("ascii") == "1.0.0"
    assert lib.axiom_veritas_version().decode("ascii") == "1.0.0"
    assert lib.axiom_dic_lexical_overlap(b"github code", b"GitHub stores code") >= 0.5
    assert lib.axiom_veritas_label_score(1.0, 0.0, 0) == 0
