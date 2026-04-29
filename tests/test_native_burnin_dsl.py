from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_burnin_probe", ROOT / "tests" / "probes" / "native_burnin.py")
assert SPEC is not None and SPEC.loader is not None
native_burnin = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = native_burnin
SPEC.loader.exec_module(native_burnin)


def test_burnin_dsl_parses_unique_jobs() -> None:
    jobs = native_burnin.parse_burnin_dsl(
        """
        burnin {
          query "find me latest AI news" workers 10 depth 2 cycles 3;
          query "last couple presidents of USA" depth 4 cycles 5;
        }
        """,
        default_workers=7,
        default_depth=2,
        default_cycles=1,
    )

    assert [job.query for job in jobs] == ["find me latest AI news", "last couple presidents of USA"]
    assert jobs[0].workers == 10
    assert jobs[0].depth == 2
    assert jobs[0].cycles == 3
    assert jobs[1].workers == 7
    assert jobs[1].depth == 4
    assert jobs[1].cycles == 5


def test_burnin_dsl_rejects_duplicate_queries() -> None:
    with pytest.raises(ValueError, match="duplicate burn-in query"):
        native_burnin.parse_burnin_dsl(
            """
            query "Find me latest AI news" cycles 1;
            query "find   me latest ai   news" cycles 1;
            """
        )


def test_burnin_dsl_accepts_utf_bom_from_windows_files() -> None:
    jobs = native_burnin.parse_burnin_dsl('\ufeffquery "windows bom query" cycles 1;')
    assert jobs[0].query == "windows bom query"


def test_burnin_gbnf_is_present() -> None:
    grammar = (ROOT / "tests" / "probes" / "native_burnin.gbnf").read_text(encoding="utf-8")
    assert "job ::= \"query\"" in grammar
    assert "positive-int" in grammar
