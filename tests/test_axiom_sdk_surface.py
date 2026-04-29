from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def test_axiom_sdk_facade_files_exist() -> None:
    required = [
        TOOLS / "axiom-sdk" / "types.ts",
        TOOLS / "axiom-sdk" / "anthropic.ts",
        TOOLS / "axiom-sdk" / "registry.ts",
        TOOLS / "axiom-sdk" / "runner.ts",
        TOOLS / "axiom-sdk" / "sidecar.ts",
        TOOLS / "AlpineStripTool" / "AlpineStripTool.ts",
        TOOLS / "package.json",
        TOOLS / "tsconfig.json",
    ]
    missing = [str(path) for path in required if not path.exists()]
    assert missing == []


def test_tools_no_longer_import_anthropic_sdk_directly_outside_facade() -> None:
    offenders = []
    for path in TOOLS.rglob("*"):
        if not path.is_file() or path.suffix not in {".ts", ".tsx", ".js"}:
            continue
        if "axiom-sdk" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "@anthropic-ai/sdk" in text:
            offenders.append(str(path.relative_to(ROOT)))
    assert offenders == []


def test_axiom_sdk_exports_provider_compatibility_types() -> None:
    text = (TOOLS / "axiom-sdk" / "anthropic.ts").read_text(encoding="utf-8")
    for token in (
        "AnthropicProviderAdapter",
        "ToolResultBlockParam",
        "ToolUseBlockParam",
        "BetaToolUseBlock",
        "BetaContentBlock",
        "BetaWebSearchTool20250305",
        "APIUserAbortError",
    ):
        assert token in text


def test_axiom_sdk_exports_alpine_strip_tool_surface() -> None:
    index = (TOOLS / "axiom-sdk" / "index.ts").read_text(encoding="utf-8")
    runner = (TOOLS / "axiom-sdk" / "runner.ts").read_text(encoding="utf-8")
    tool = (TOOLS / "AlpineStripTool" / "AlpineStripTool.ts").read_text(encoding="utf-8")
    assert "AlpineStripTool" in index
    assert "tool_strip_accelerator.c" in runner
    assert "normalizeAlpineStripRequest" in tool


def test_tools_bridge_uses_axiom_events_not_private_shapes() -> None:
    text = (ROOT / "tag" / "tools_bridge.py").read_text(encoding="utf-8")
    for token in (
        "SnapshotCandidateEvent",
        "SnapshotCapturedEvent",
        "ToolHealthEvent",
        "ToolInvocationEvent",
        "ToolResultEvent",
    ):
        assert token in text
    assert "AXIOM_WATERMARK" in text


def test_runtime_build_scripts_exist_for_dll_and_so() -> None:
    assert (ROOT / "axicomp.cmd").exists()
    assert (ROOT / "axicomp.sh").exists()
    assert (ROOT / "Axiom.sln").exists()
    assert (ROOT / "AxiomRuntime.vcxproj").exists()
    assert (ROOT / "axiom_runtime" / "axiom_runtime.c").exists()
    assert (ROOT / "axiom_runtime" / "axiom_runtime.h").exists()


def test_swarm_axiom_bridge_surface_exists() -> None:
    bridge = ROOT / "swarm" / "axiom" / "bridge.ts"
    assert bridge.exists()
    text = bridge.read_text(encoding="utf-8")
    assert "axiom.swarm.webwide.v1" in text
    assert "toAxiomCrawlPlan" in text
    assert "one_worker_per_site" in text
    assert (ROOT / "swarm" / "tsconfig.axiom.json").exists()
