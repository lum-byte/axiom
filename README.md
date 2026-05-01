# AXIOM

AXIOM is a Topology-Addressed Generation runtime. The goal is to replace the usual RAG shape of "embed documents, store vectors, retrieve by similarity" with a system that treats the web as typed topology: classify the source shape, crawl the right parts, strip noise, keep signal, and expose one inference boundary.

The current repo is v1.0.5 of that runtime surface. It includes the Python TAG layer, crawler pieces, signal/kernel tests, native C ABI (`axi.dll` / `axi.so`), Rust terminal runtime checks, TypeScript coordinator bridge, and a standalone Python inference file that calls the native library directly.

## Where The Idea Came From

It started because I was bored and annoyed.

Every search tool I used felt like it was lying to me, not with wrong answers, but with noise dressed up as answers. Paragraphs of hedged, bloated text that made me read four sentences to find one fact. I kept thinking: someone already wrote the clean version of this. It's on a page somewhere. The problem isn't the information, it's everything around it.
So I started pulling at that thread.

The first thing I built was the signal kernel, a grep pipeline inside an Alpine container that strips raw HTML down to just the parts that matter before any language model sees it. Stripe's API docs: 1.1MB in, 8.4KB out, in 7 milliseconds. That worked. That felt like something.
Then I realized the stripping was only useful if you knew what the page was before you fetched it. So I built a classifier, five signal paths that look at a URL, its headers, and the first 4KB of content and decide what topology class the page belongs to before a single full fetch happens. News article. SaaS docs. REST API. Forum thread. Eighteen classes total. The traversal policy, the extraction recipe, the fetch strategy, all of it falls out of that classification automatically.

Then I needed a crawl layer that wouldn't fall over. Bloom filter for deduplication, 44,000 URLs per second, zero false positives at five million URLs. SQLite-backed frontier that survives crashes and resumes from the exact position it died at.
Then a world model. Then a topology parser that compiles extraction recipes. Then an RL loop that treats gradient steps as index updates. Then VERITAS for source confidence. Then DIC for answer fusion. Then a persistent mmap store so a warm query costs 0.001 seconds and a cold one costs 0.002.
None of it was planned. Each piece existed because the previous piece needed it.

The result is a search engine with no vector database, no embedding store, no retrieval step. The weights are the index. Reading is retrieval. It answers in a millisecond and cites its sources.

The practical command shape is:

```text
search | fanout -10 | depth -2 | find me latest AI news
```

Workers are clamped by the runtime. Keep the default ceiling at 10 until you intentionally raise it.

Repeat searches return through the completed-answer cache, stored at `store/search_cache.mmap` with metadata in `store/search_cache_index.json`. Use `recheck` when you want TAG to bypass that cache and rerun crawler, DIC, and VERITAS:

```text
search | fanout -10 | depth -2 | exp -10 | recheck | what is github
```

## Resident Crawl Daemon

AXIOM also ships a Go resident crawler daemon. It boots a fixed worker pool once, keeps HTTP connections warm, sleeps on a JSONL control channel, and wakes when a query sends candidate URLs. The daemon keeps an in-memory link graph and applies a PageRank-style score so URLs repeatedly referenced by useful pages rise in later crawl plans.

Build it:

```bash
make build-go
```

Run it directly:

```bash
Releases-x64/compiled/binaries/Linux64/axiom-crawl-daemon -workers 10
```

Then send JSONL:

```json
{"id":"s1","op":"status"}
{"id":"q1","op":"query","query":"what is github","limit":2,"candidates":[{"url":"https://en.wikipedia.org/wiki/GitHub"},{"url":"https://github.com/about"}]}
```

The response includes `results`, `telemetry`, and `pagerank` stats. This is a resident fanout layer; cached completed answers are still served faster through `store/search_cache.mmap`.

## Dynamic Crawler Config

Crawler source profiles, URL templates, fanout limits, and clearance policies live in:

```text
config/crawler_sources.json
```

Useful knobs:

```bash
export AXIOM_CRAWLER_SOURCE_CONFIG=/path/to/crawler_sources.json
export AXIOM_SOURCE_DOMAINS="reuters.com,arxiv.org,openai.com"
export AXIOM_MAX_SEARCH_SOURCES=128
export AXIOM_LINK_EXPANSION_PER_DOC=12
export AXIOM_NATIVE_MAX_SOURCES=32
```

Clearance policies:

```bash
export AXIOM_CLEARANCE_POLICY=standard  # CL1 only
export AXIOM_CLEARANCE_POLICY=dev       # CL1..CL4, requires AXIOM_ENV=dev
export AXIOM_CLEARANCE_POLICY=deep      # CL1..CL4 where available, no dev gate
export AXIOM_CLEARANCE_POLICY=max       # CL1..CL4 requested even if availability is unknown
export AXIOM_CLEARANCE_LEVELS=1,2,4     # explicit override
```

The fetcher currently exposes four real clearance modes, so deeper policies expand through CL1..CL4 instead of inventing fake modes.

## Release Layout

Build output lives under `Releases-x64/`.

```text
Releases-x64/
  axi.dll
  axi.so
  axi-dep-resolver.exe
  compiled/
    binaries/
      Winx64/
        axirt.dll
        axi-dep-resolver.exe
        tag-mcp.exe
      Linux64/
        axirt.so
        tag-mcp
```

Root `axi.dll` and `axi.so` are the public aliases. Platform runtime files stay inside `compiled/binaries/*` as `axirt.dll` and `axirt.so`, so there is one obvious public library per OS at the release root.

## Build

Linux or WSL:

```bash
./axicomp.sh runtime-linux
```

Windows:

```cmd
axicomp.cmd
```

`axicomp.cmd` prefers Visual Studio/MSBuild through `Axiom.sln`. If Visual Studio is not available, it falls back to `cl.exe` or `gcc.exe` when one is on PATH. From WSL, build the Windows MCP executable with:

```bash
make build-go-windows
```

## External MCP Server

AXIOM TAG also builds as an external MCP stdio process. The Go server owns the JSON-RPC transport and exposes TAG tools plus anchor acquisition tools:

```text
tag.search
tag.status
tag.expand
tag.veritas
tag.inject_context
anchor.wikipedia
anchor.news
anchor.scholar
anchor.wayback
anchor.web
```

Linux or WSL:

```bash
./axicomp.sh linux
Releases-x64/compiled/binaries/Linux64/tag-mcp --mode stdio
```

Example MCP client config is in:

```text
mcp/axiom-tag.example.json
```

The no-key anchor tier uses Wikipedia, GDELT, Crossref, OpenAlex, and Wayback. Set `BRAVE_SEARCH_API_KEY` to enable `anchor.web`, which adds an independent broad-web index through Brave Search.

## Dependency Resolver

Windows users can run:

```cmd
Releases-x64\axi-dep-resolver.exe
```

It is a native Win32 executable, so no JRE is needed. It asks for administrator permission when needed, shows terms, checks official vendor links, downloads safe bootstrapper files into `%LOCALAPPDATA%\Axiom\deps\downloads`, writes `%LOCALAPPDATA%\Axiom\deps\dependency-manifest.json`, opens manual download pages for tools that require license review, and adds already-installed common tool paths to the user PATH.

Direct downloads:

```text
https://aka.ms/vs/17/release/vc_redist.x64.exe
https://aka.ms/vs/17/release/vs_BuildTools.exe
https://win.rustup.rs/x86_64
```

Manual/vendor pages opened by the resolver:

```text
https://www.python.org/downloads/windows/
https://git-scm.com/download/win
https://nodejs.org/en/download
https://go.dev/dl/
https://developer.nvidia.com/cuda-downloads
https://www.torproject.org/download/tor/
```

Restart your terminal or AXIOM app after PATH changes.

The Rust runtime checker is useful for dev machines:

```bash
cargo run --manifest-path axiom_tui/Cargo.toml -- --resolve-runtime-deep
```

It checks source files, release files, Python imports, CUDA runtime visibility, Tor, and a deep `mamba_ssm` CUDA forward pass.

## Standalone Native Inference

`axiom_infer.py` is the small independent inference point. It loads `axi.dll` or `axi.so` with `ctypes`, sends JSON into the native ABI, and prints the full JSON envelope.

```bash
.venv/bin/python axiom_infer.py \
  --query "find me latest AI news" \
  --workers 10 \
  --depth 2
```

Compact output:

```bash
.venv/bin/python axiom_infer.py -q "last couple presidents of USA" -w 10 -d 2 --compact
```

Explicit library:

```bash
.venv/bin/python axiom_infer.py --lib Releases-x64/axi.so -q "AI model release news"
```

Python import surface:

```python
from pathlib import Path

from axiom_infer import AxiomNative, build_request, find_native_library

library = find_native_library()
request = build_request("search", "find me latest AI news", workers=10, depth=2)

with AxiomNative(library, store_dir=Path(".axiom_runtime/native-infer-store")) as axiom:
    result = axiom.call(request)
    print(result["json"])
```

## HTML Capture

The crawler now keeps a tiny bounded sample of the first fetched HTML pages for debugging and extraction work. By default it saves up to 10 successful HTML-ish responses under the OS temp directory:

```text
/tmp/axiom_fetch_html/
  01_example.com_<hash>.html
  manifest.jsonl
```

Controls:

```bash
export AXIOM_HTML_CAPTURE_LIMIT=10
export AXIOM_HTML_CAPTURE_DIR=/tmp/axiom_fetch_html
export AXIOM_HTML_CAPTURE_MAX_BYTES=2097152
```

Set `AXIOM_HTML_CAPTURE_LIMIT=0` to disable it.

## Search Cache

The fast path is intentionally above crawler fetch, MCP anchors, DIC, and VERITAS. It caches the final answer envelope, so repeated expanded queries do not recompute context fusion unless `recheck` is present.

Config lives in `config.toml`:

```toml
[search_cache]
enabled = true
persist = true
ttl_seconds = 86400.0
path = "store/search_cache.mmap"
index_path = "store/search_cache_index.json"
```

## Terminal Interface

For the `axiom>` style loop:

```bash
cd axiom_tui
cargo run
```

Then type:

```text
axiom> search | fanout -10 | depth -2 | find me latest AI news
axiom> fetch | https://example.com
axiom> learn | reuters.com
axiom> status |
axiom> quit |
```

Set `AXIOM_TUI_ECHO=1` for local UI/parser testing without the live Python interface.

## Testing

Focused Python tests:

```bash
.venv/bin/python -m pytest tests/test_axiom_infer.py tests/test_dep_resolver_release.py -q
```

Full Python suite:

```bash
.venv/bin/python -m pytest -q
```

Full project suite:

```bash
make test PYTHON=.venv/bin/python
```

Native inference burn-in:

```bash
.venv/bin/python tests/probes/native_burnin.py --cycles 1000
.venv/bin/python tests/probes/native_burnin.py --dsl 'burnin { query "find me latest AI news" workers 10 depth 2 cycles 1000; query "last couple presidents of USA" depth 2 cycles 1000; }'
```

The burn-in DSL grammar is in `tests/probes/native_burnin.gbnf`. The parser rejects duplicate normalized query strings before it launches native inference, so a burn-in suite does not accidentally hammer the same query twice. `--duration-seconds` is now a ceiling over the unique jobs supplied; if the unique job list finishes early, the probe reports `duration_exhausted_unique_queries: true`.

## Python Dependencies

Start with:

```bash
python -m venv .venv
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m playwright install chromium
```

Key runtime imports include `aiokafka`, `aiosqlite`, `h2`, `httpx`, `inotify_simple`, `mamba_ssm`, `mmh3`, `msgpack`, `numpy`, `orjson`, `playwright`, `rich`, `structlog`, `tenacity`, and `torch`.

## Notes

Do not stub dependency failures. If the runtime checker fails, fix the failing dependency or mark it as explicitly optional in code and docs. AXIOM is meant to be a live crawler and signal runtime, so false green checks are worse than loud red ones.
