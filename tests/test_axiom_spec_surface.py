from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_SPEC_FILES = [
    "go.mod",
    "preparser/domain_analyzer.go",
    "preparser/crawl_planner.go",
    "preparser/signal_extractor.go",
    "preparser/recipe_validator.go",
    "preparser/preparser_test.go",
    "alpine_strip/strip_engine.h",
    "alpine_strip/strip_engine.c",
    "alpine_strip/batch_runner.c",
    "alpine_strip/test_strip.c",
    "offline/offline_api.h",
    "offline/gpu_encoder.cu",
    "offline/weight_updater.cu",
    "offline/gradient_accumulator.cu",
    "offline/batch_scheduler.c",
    "offline/test_offline.cu",
    "daemons/daemon_common.h",
    "daemons/phase_daemon.c",
    "daemons/store_sentinel.c",
    "daemons/test_daemons.c",
    "tag/index_daemon.py",
    "tag/cold_start.py",
    "tag/cold_start_c.c",
    "tag/interface.py",
    "axiom_tui/Cargo.toml",
    "axiom_tui/src/main.rs",
    "axiom_tui/src/ui.rs",
    "axiom_tui/src/repl.rs",
    "axiom_tui/src/logo.rs",
    "axiom_tui/src/dispatcher.rs",
    "Makefile",
    "Axiom.sln",
    "AxiomRuntime.vcxproj",
    "axicomp.sh",
    "axicomp.cmd",
    "run_c_tests.sh",
    "run_cuda_tests.sh",
]


def test_every_spec_file_exists() -> None:
    missing = [path for path in REQUIRED_SPEC_FILES if not (ROOT / path).exists()]
    assert missing == []


def test_interface_has_required_commands_without_external_search_engine() -> None:
    interface = (ROOT / "tag/interface.py").read_text(encoding="utf-8").lower()
    for command in ("search", "fetch", "learn", "status", "quit"):
        assert command in interface
    forbidden = ("google", "bing", "serpapi", "external search", "search engine")
    assert not any(term in interface for term in forbidden)


def test_native_layers_reference_bus_bridge_or_contracts() -> None:
    preparser_text = "\n".join(
        (ROOT / path).read_text(encoding="utf-8")
        for path in REQUIRED_SPEC_FILES
        if path.startswith("preparser/") and path.endswith(".go")
    )
    assert "BridgeRequest" in preparser_text
    assert "domain_topology" in preparser_text
    assert "crawl_manifest" in preparser_text
    assert "signal_extracted" in preparser_text
    assert "recipe_health" in preparser_text


def test_loc_audit_is_honest_about_current_spec_depth() -> None:
    production_files = [
        path
        for path in REQUIRED_SPEC_FILES
        if path.endswith((".go", ".c", ".cu", ".rs", ".py"))
        and not path.endswith(("_test.go", "test_strip.c", "test_offline.cu", "test_daemons.c"))
    ]
    audit = {
        path: len((ROOT / path).read_text(encoding="utf-8").splitlines())
        for path in production_files
    }
    assert all(count > 0 for count in audit.values())
    assert sum(audit.values()) >= 2000
