use std::env;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone)]
pub struct RuntimeCheck {
    pub name: String,
    pub kind: String,
    pub required: bool,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct RuntimeResolution {
    pub root: PathBuf,
    pub python: Option<PathBuf>,
    pub deep: bool,
    pub checks: Vec<RuntimeCheck>,
}

impl RuntimeResolution {
    pub fn ok(&self) -> bool {
        self.checks.iter().all(|check| check.ok || !check.required)
    }

    pub fn to_json(&self) -> String {
        let mut out = String::new();
        out.push('{');
        push_json_field(&mut out, "root", &self.root.display().to_string(), false);
        out.push_str(",\"python\":");
        match &self.python {
            Some(path) => push_json_string(&mut out, &path.display().to_string()),
            None => out.push_str("null"),
        }
        out.push_str(",\"deep\":");
        out.push_str(if self.deep { "true" } else { "false" });
        out.push_str(",\"ok\":");
        out.push_str(if self.ok() { "true" } else { "false" });
        out.push_str(",\"checks\":[");
        for (idx, check) in self.checks.iter().enumerate() {
            if idx > 0 {
                out.push(',');
            }
            out.push('{');
            push_json_field(&mut out, "name", &check.name, false);
            push_json_field(&mut out, "kind", &check.kind, true);
            out.push_str(",\"required\":");
            out.push_str(if check.required { "true" } else { "false" });
            out.push_str(",\"ok\":");
            out.push_str(if check.ok { "true" } else { "false" });
            push_json_field(&mut out, "detail", &check.detail, true);
            out.push('}');
        }
        out.push_str("]}");
        out
    }
}

pub fn resolve_runtime(deep: bool) -> RuntimeResolution {
    let root = find_repo_root().unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let python = resolve_python(&root);
    let mut checks = Vec::new();

    check_repo_files(&root, &mut checks);
    check_commands(&mut checks);
    check_python(&python, &mut checks);
    check_python_imports(&python, &mut checks);
    check_cuda_runtime(&root, &python, &mut checks);
    check_tor_runtime(&root, &mut checks);
    if deep {
        check_deep_mamba(&python, &mut checks);
    }

    RuntimeResolution { root, python, deep, checks }
}

fn find_repo_root() -> Option<PathBuf> {
    if let Ok(root) = env::var("AXIOM_ROOT") {
        let path = PathBuf::from(root);
        if is_repo_root(&path) {
            return Some(path);
        }
    }

    let mut current = env::current_dir().ok()?;
    loop {
        if is_repo_root(&current) {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

fn is_repo_root(path: &Path) -> bool {
    path.join("requirements.txt").is_file()
        && path.join("tag").is_dir()
        && path.join("axiom_runtime").is_dir()
        && path.join("axiom_tui").is_dir()
}

fn resolve_python(root: &Path) -> Option<PathBuf> {
    let candidates = [
        root.join(".venv/bin/python"),
        root.join(".venv/Scripts/python.exe"),
        PathBuf::from("python3"),
        PathBuf::from("python"),
    ];
    candidates.into_iter().find(|path| {
        if path.components().count() > 1 {
            path.is_file()
        } else {
            command_ok(path, &["--version"])
        }
    })
}

fn check_repo_files(root: &Path, checks: &mut Vec<RuntimeCheck>) {
    let required = [
        "requirements.txt",
        "tag/interface.py",
        "tag/topology/classifier.py",
        "tag/world_model/world_latent_model/mamba_router.py",
        "tag/crawler/fetcher.py",
        "axiom_runtime/axiom_runtime.c",
        "axiom_runtime/axiom_runtime.h",
        "Axiom.sln",
        "AxiomRuntime.vcxproj",
        "axicomp.sh",
        "axicomp.cmd",
        "Releases-x64/axi.dll",
        "Releases-x64/axi.so",
        "Releases-x64/axirt.dll",
        "Releases-x64/axirt.so",
        "Releases-x64/compiled/binaries/Winx64/axirt.dll",
        "Releases-x64/compiled/binaries/Linux64/axirt.so",
        "run_cuda_tests.sh",
        "go.mod",
        "tools/package.json",
    ];
    for rel in required {
        let path = root.join(rel);
        checks.push(RuntimeCheck {
            name: rel.to_string(),
            kind: "file".to_string(),
            required: true,
            ok: path.exists(),
            detail: path.display().to_string(),
        });
    }
}

fn check_commands(checks: &mut Vec<RuntimeCheck>) {
    for (name, args, required) in [
        ("gcc", &["--version"][..], true),
        ("g++", &["--version"][..], true),
        ("go", &["version"][..], true),
        ("cargo", &["--version"][..], true),
        ("node", &["--version"][..], true),
        ("npm", &["--version"][..], true),
        ("nvidia-smi", &["--version"][..], true),
    ] {
        let output = run_capture(name, args);
        checks.push(RuntimeCheck {
            name: name.to_string(),
            kind: "command".to_string(),
            required,
            ok: output.0,
            detail: first_line(&output.1),
        });
    }
}

fn check_python(python: &Option<PathBuf>, checks: &mut Vec<RuntimeCheck>) {
    match python {
        Some(path) => {
            let output = run_capture_path(path, &["--version"]);
            checks.push(RuntimeCheck {
                name: "python".to_string(),
                kind: "python".to_string(),
                required: true,
                ok: output.0,
                detail: first_line(&output.1),
            });
        }
        None => checks.push(RuntimeCheck {
            name: "python".to_string(),
            kind: "python".to_string(),
            required: true,
            ok: false,
            detail: "no .venv python or PATH python found".to_string(),
        }),
    }
}

fn check_python_imports(python: &Option<PathBuf>, checks: &mut Vec<RuntimeCheck>) {
    let imports = [
        "aiokafka",
        "aiosqlite",
        "h2",
        "httpx",
        "inotify_simple",
        "mamba_ssm",
        "mmh3",
        "msgpack",
        "numpy",
        "orjson",
        "playwright",
        "pytest",
        "rich",
        "structlog",
        "tenacity",
        "torch",
    ];
    for module in imports {
        let ok = python
            .as_ref()
            .map(|py| run_python(py, &format!("import {module}; print('ok')")).0)
            .unwrap_or(false);
        checks.push(RuntimeCheck {
            name: module.to_string(),
            kind: "python_import".to_string(),
            required: true,
            ok,
            detail: if ok { "import ok".to_string() } else { "import failed".to_string() },
        });
    }

    if let Some(py) = python {
        let output = run_python(
            py,
            "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())",
        );
        checks.push(RuntimeCheck {
            name: "torch_cuda".to_string(),
            kind: "python_cuda".to_string(),
            required: true,
            ok: output.0 && output.1.contains("True"),
            detail: first_line(&output.1),
        });
        let output = run_python(
            py,
            concat!(
                "from playwright.sync_api import sync_playwright; ",
                "p=sync_playwright().start(); ",
                "b=p.chromium.launch(headless=True); ",
                "print(b.version); b.close(); p.stop()",
            ),
        );
        checks.push(RuntimeCheck {
            name: "playwright_chromium".to_string(),
            kind: "browser_runtime".to_string(),
            required: true,
            ok: output.0,
            detail: first_line(&output.1),
        });
    }
}

fn check_cuda_runtime(root: &Path, python: &Option<PathBuf>, checks: &mut Vec<RuntimeCheck>) {
    let cuda_home = find_venv_cuda_home(root);
    let cuda_base = cuda_home
        .as_deref()
        .unwrap_or_else(|| Path::new(".venv/lib/python*/site-packages/nvidia/cu*"));

    let nvcc = cuda_base.join("bin/nvcc");
    let cudart_link = cuda_base.join("lib/libcudart.so");
    let cudart_versioned = cuda_base.join("lib/libcudart.so.13");
    let cuda_target = cuda_base.join("include/nv/target");

    for (name, path, required) in [
        ("venv_nvcc", nvcc, true),
        ("cuda_cudart_link", cudart_link, true),
        ("cuda_cudart_versioned", cudart_versioned, true),
        ("cuda_cccl_nv_target", cuda_target, true),
    ] {
        checks.push(RuntimeCheck {
            name: name.to_string(),
            kind: "cuda_file".to_string(),
            required,
            ok: path.exists(),
            detail: path.display().to_string(),
        });
    }

    if let Some(cuda_home) = cuda_home {
        let output = run_capture_path(&cuda_home.join("bin/nvcc"), &["--version"]);
        checks.push(RuntimeCheck {
            name: "venv_nvcc_version".to_string(),
            kind: "cuda_command".to_string(),
            required: true,
            ok: output.0,
            detail: first_line(&output.1),
        });
    }

    if let Some(py) = python {
        let output = run_python(py, "from mamba_ssm import Mamba; print(Mamba.__name__)");
        checks.push(RuntimeCheck {
            name: "mamba_ssm_mamba_class".to_string(),
            kind: "python_import".to_string(),
            required: true,
            ok: output.0 && output.1.contains("Mamba"),
            detail: first_line(&output.1),
        });
    }
}

fn check_tor_runtime(root: &Path, checks: &mut Vec<RuntimeCheck>) {
    let mut candidates = Vec::new();
    if cfg!(windows) {
        candidates.push(root.join(".axiom_runtime/deps/tor/tor/tor.exe"));
        candidates.push(root.join("runtime_deps/tor/tor/tor.exe"));
        candidates.push(root.join("tools/tor/tor.exe"));
    } else {
        candidates.push(root.join(".axiom_runtime/deps/tor-linux/tor/tor"));
        candidates.push(root.join(".axiom_runtime/deps/tor/tor/tor"));
        candidates.push(root.join("runtime_deps/tor/tor/tor"));
        candidates.push(root.join("tools/tor/tor"));
    }

    let mut checked_path = None;
    let mut captured = None;
    for candidate in candidates {
        if !candidate.is_file() {
            continue;
        }
        if !cfg!(windows) && candidate.extension().and_then(|ext| ext.to_str()) == Some("exe") {
            continue;
        }
        let mut cmd = Command::new(&candidate);
        cmd.arg("--version");
        if !cfg!(windows) {
            cmd.env("LD_LIBRARY_PATH", candidate.parent().unwrap_or_else(|| Path::new(".")));
        }
        checked_path = Some(candidate);
        captured = Some(capture_command(&mut cmd));
        break;
    }

    let (ok, detail) = if let Some(output) = captured {
        (output.0 && output.1.contains("Tor version"), first_line(&output.1))
    } else {
        let system = run_capture("tor", &["--version"]);
        (system.0 && system.1.contains("Tor version"), first_line(&system.1))
    };
    checks.push(RuntimeCheck {
        name: "tor_runtime".to_string(),
        kind: "tor_runtime".to_string(),
        required: true,
        ok,
        detail: checked_path
            .map(|path| format!("{}: {detail}", path.display()))
            .unwrap_or(detail),
    });
}

fn check_deep_mamba(python: &Option<PathBuf>, checks: &mut Vec<RuntimeCheck>) {
    let code = concat!(
        "import torch; from mamba_ssm import Mamba; ",
        "m=Mamba(d_model=16,d_state=16,d_conv=4,expand=2).cuda(); ",
        "x=torch.randn(2,8,16,device='cuda'); y=m(x); ",
        "torch.cuda.synchronize(); print(tuple(y.shape), torch.isfinite(y).all().item())",
    );
    let output = python.as_ref().map(|py| run_python(py, code)).unwrap_or((false, String::new()));
    checks.push(RuntimeCheck {
        name: "mamba_ssm_cuda_forward".to_string(),
        kind: "python_cuda".to_string(),
        required: true,
        ok: output.0 && output.1.contains("True"),
        detail: first_line(&output.1),
    });
}

fn run_python(python: &Path, code: &str) -> (bool, String) {
    let mut cmd = Command::new(python);
    cmd.arg("-c").arg(code);
    inject_runtime_env(&mut cmd);
    capture_command(&mut cmd)
}

fn run_capture(command: &str, args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(command);
    cmd.args(args);
    capture_command(&mut cmd)
}

fn run_capture_path(command: &Path, args: &[&str]) -> (bool, String) {
    let mut cmd = Command::new(command);
    cmd.args(args);
    capture_command(&mut cmd)
}

fn command_ok(command: &Path, args: &[&str]) -> bool {
    run_capture_path(command, args).0
}

fn capture_command(cmd: &mut Command) -> (bool, String) {
    match cmd.stdout(Stdio::piped()).stderr(Stdio::piped()).output() {
        Ok(output) => {
            let mut text = String::new();
            text.push_str(&String::from_utf8_lossy(&output.stdout));
            text.push_str(&String::from_utf8_lossy(&output.stderr));
            (output.status.success(), text.trim().to_string())
        }
        Err(err) => (false, err.to_string()),
    }
}

fn inject_runtime_env(cmd: &mut Command) {
    if let Some(root) = find_repo_root() {
        let mut ld_paths = Vec::new();
        if let Some(cuda_home) = find_venv_cuda_home(&root) {
            ld_paths.push(cuda_home.join("lib"));
        }
        if let Some(site_packages) = find_venv_site_packages(&root) {
            ld_paths.push(site_packages.join("torch/lib"));
        }
        let mut ld = ld_paths
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(":");
        if let Ok(existing) = env::var("LD_LIBRARY_PATH") {
            if !existing.is_empty() {
                if !ld.is_empty() {
                    ld.push(':');
                }
                ld.push_str(&existing);
            }
        }
        if !ld.is_empty() {
            cmd.env("LD_LIBRARY_PATH", ld);
        }
    }
}

fn find_venv_site_packages(root: &Path) -> Option<PathBuf> {
    if let Some(python) = resolve_python(root) {
        if let Some(site_packages) = python_site_packages(&python) {
            return Some(site_packages);
        }
    }

    let unix_lib = root.join(".venv/lib");
    let mut candidates = Vec::new();
    if let Ok(entries) = std::fs::read_dir(&unix_lib) {
        for entry in entries.flatten() {
            let path = entry.path().join("site-packages");
            if path.is_dir() {
                candidates.push(path);
            }
        }
    }

    let windows_site = root.join(".venv/Lib/site-packages");
    if windows_site.is_dir() {
        candidates.push(windows_site);
    }

    candidates.sort();
    candidates.into_iter().next_back()
}

fn find_venv_cuda_home(root: &Path) -> Option<PathBuf> {
    let site_packages = find_venv_site_packages(root)?;
    let nvidia = site_packages.join("nvidia");
    let mut candidates = Vec::new();
    if let Ok(entries) = std::fs::read_dir(nvidia) {
        for entry in entries.flatten() {
            let path = entry.path();
            let name = path.file_name().and_then(|name| name.to_str()).unwrap_or("");
            if path.is_dir() && name.starts_with("cu") {
                candidates.push(path);
            }
        }
    }

    candidates.sort();
    candidates.into_iter().rev().find(|path| {
        path.join("bin/nvcc").is_file()
            || path.join("lib/libcudart.so").exists()
            || path.join("lib/libcudart.so.13").exists()
            || path.join("include/nv/target").is_dir()
    })
}

fn python_site_packages(python: &Path) -> Option<PathBuf> {
    let output = run_capture_path(
        python,
        &[
            "-c",
            "import sysconfig; print(sysconfig.get_paths().get('purelib') or '')",
        ],
    );
    if !output.0 {
        return None;
    }
    let path = PathBuf::from(first_line(&output.1));
    path.is_dir().then_some(path)
}

fn first_line(text: &str) -> String {
    text.lines().next().unwrap_or("").trim().to_string()
}

fn push_json_field(out: &mut String, key: &str, value: &str, comma: bool) {
    if comma {
        out.push(',');
    }
    push_json_string(out, key);
    out.push(':');
    push_json_string(out, value);
}

fn push_json_string(out: &mut String, value: &str) {
    out.push('"');
    for ch in value.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_escapes_strings() {
        let check = RuntimeCheck {
            name: "quote\"".into(),
            kind: "unit".into(),
            required: true,
            ok: true,
            detail: "line\nbreak".into(),
        };
        let resolution = RuntimeResolution {
            root: PathBuf::from("/tmp/axiom"),
            python: None,
            deep: false,
            checks: vec![check],
        };
        let json = resolution.to_json();
        assert!(json.contains("quote\\\""));
        assert!(json.contains("line\\nbreak"));
        assert!(json.contains("\"ok\":true"));
    }
}
