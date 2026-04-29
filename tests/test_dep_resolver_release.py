from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dependency_resolver_source_uses_native_windows_flow() -> None:
    source = (ROOT / "tools" / "axi_dep_resolver" / "axi_dep_resolver.c").read_text(encoding="utf-8")
    assert 'ShellExecuteW(NULL, L"runas"' in source
    assert "Write-Progress -Activity 'Resolving AXIOM dependencies'" in source
    assert "vc_redist.x64.exe" in source
    assert "vs_BuildTools.exe" in source
    assert "developer.nvidia.com/cuda-downloads" in source
    assert "dependency-manifest.json" in source
    assert "AXI_DEP_RESOLVER_VERSION" not in source.split("const char *script =", 1)[1]


def test_visual_studio_solution_builds_runtime_and_dep_resolver() -> None:
    solution = (ROOT / "Axiom.sln").read_text(encoding="utf-8")
    resolver_project = (ROOT / "AxiomDepResolver.vcxproj").read_text(encoding="utf-8")
    runtime_project = (ROOT / "AxiomRuntime.vcxproj").read_text(encoding="utf-8")

    assert "AxiomRuntime.vcxproj" in solution
    assert "AxiomDepResolver.vcxproj" in solution
    assert "<ConfigurationType>Application</ConfigurationType>" in resolver_project
    assert "<TargetName>axi-dep-resolver</TargetName>" in resolver_project
    assert "tools\\axi_dep_resolver\\axi_dep_resolver.c" in resolver_project
    assert "Releases-x64\\axi-dep-resolver.exe" in resolver_project
    assert "Releases-x64\\axirt.dll" not in runtime_project


def test_linux_and_windows_release_layout_keep_runtime_aliases_platform_scoped() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    batch = (ROOT / "axicomp.cmd").read_text(encoding="utf-8")
    resolver = (ROOT / "axiom_tui" / "src" / "runtime_resolver.rs").read_text(encoding="utf-8")

    assert "$(RELEASE_ROOT)/axi.so" in makefile
    assert "$(RELEASE_ROOT)/axirt.so" in makefile
    assert "set \"ROOT_DLL=%RELEASE_ROOT%\\axi.dll\"" in batch
    assert "set \"ROOT_DEP_RESOLVER=%RELEASE_ROOT%\\axi-dep-resolver.exe\"" in batch
    assert "del /q \"%RELEASE_ROOT%\\axirt.dll\"" in batch
    assert "set \"BIN_DLL=" not in batch
    assert '"Releases-x64/axi.dll"' in resolver
    assert '"Releases-x64/axi.so"' in resolver
    assert '"Releases-x64/compiled/binaries/Winx64/axirt.dll"' in resolver
    assert '"Releases-x64/compiled/binaries/Linux64/axirt.so"' in resolver
    assert '"Releases-x64/axirt.dll"' not in resolver
    assert '"Releases-x64/axirt.so"' not in resolver
