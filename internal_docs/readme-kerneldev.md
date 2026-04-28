# signal_kernel — Developer Reference
**AXIOM Core Searching Algorithm — Kernel Layer**
**Classification: AXIOM INTERNAL // DO NOT SURFACE**

---

## What This Is

The signal kernel is the execution layer of TAG (Topology-Addressed Generation). It is not intelligent. It does not learn. It does not make decisions. It executes compiled grep recipes at C speed against raw HTML and JSON, strips off-critical DOM zones before any LLM sees the page, and checkpoints the master index every 15 minutes without human intervention.

It has two completely independent processes sharing one Alpine OS:

- **Process 1 — grep Pipeline**: Invoked per extraction. Stateless. Receives raw HTML or JSON via stdin. Executes the recipe compiled by `topology_parser.py`. Outputs clean signal text to stdout. Exits immediately.
- **Process 2 — Checkpoint Daemon**: Long-running `crond`. Fires every 15 minutes independent of query volume. Reads the four index files from the mounted `/store` volume. Writes compressed archives to `/store/checkpoints/`. Rotates — keeps last 48. 12 hours of history. Zero network. Zero external service.

If Process 2 crashes, Process 1 keeps running. If a Process 1 invocation hangs, Process 2 keeps firing. They share an OS and nothing else. This is intentional and load-bearing.

---

## What Problem This Solves

Every RAG system hands raw HTML to an LLM. The LLM burns tokens on:

```
<div class="sidebar-widget-container">
<nav aria-label="breadcrumb site navigation">
<footer class="site-footer-widgets-area">
```

Before it reads a single word of actual signal. The most expensive component in the system is doing the cheapest possible job.

The signal kernel inverts this. Before any LLM invocation, the kernel strips every off-critical DOM zone and leaves only what the active topology class has learned to call signal. Haiku receives clean text. Not HTML. Not noise. Text.

Token reduction on known topology classes: **60-80%.** This reduction compounds across every URL in every run across the lifetime of the system. The kernel does not get smarter — the recipes it executes get smarter as `topology_parser.py` learns. The kernel just runs whatever it is given at C speed.

`grep` is C. C operates at memory bandwidth. The entire noise stripping operation — strip nav, strip footer, strip aside, extract article zone, remove tags, collapse whitespace — completes in milliseconds and costs zero LLM tokens.

---

## What The Kernel Is Not

The kernel is not:

- A parser. It does not understand HTML. It pattern-matches against raw text.
- Intelligent. All intelligence lives in the recipe. The kernel executes instructions mechanically.
- A service. It has no HTTP interface. It has no port. `network_mode: none`. It communicates exclusively via stdin/stdout subprocess managed by `pipeline.py`.
- Responsible for recipe generation. `topology_parser.py` writes recipes. The kernel executes them read-only.
- A caching layer. It holds no state between invocations. Every Process 1 invocation is fresh.

If the kernel is being asked to make a decision, something has gone architecturally wrong.

---

## File Structure

```
signal_kernel/
    Dockerfile
    entrypoint.sh
    docker-compose.yml
    contracts.py
    exceptions.py
    pipeline.py
    feedback.py
    checkpoint/
        mft_checkpoint.sh
        restore.sh
        checkpoint_monitor.py
    recipes/
        registry.py
        validator.py
        hardcoded/
            news_article.sh
            saas_docs.sh
            rest_api_json.sh
            json_ld.sh
            ecommerce.sh
```

---

## File Responsibilities

### `contracts.py`
Typed dataclasses for every boundary crossing in the kernel. Written first. Every other file imports from here.

```python
@dataclass(frozen=True)
class KernelInput:
    raw_content: str           # raw HTML or JSON from Phantom
    topology_class: str        # classifier output from index_daemon
    intent_vector_hash: str    # for telemetry correlation only
    content_type: Literal["html", "json"]
    source_url: str

@dataclass(frozen=True)
class KernelOutput:
    clean_signal: str          # what exits the grep pipeline
    raw_byte_count: int        # before stripping
    clean_byte_count: int      # after stripping
    token_delta_estimate: int  # estimated token reduction
    recipe_used: str           # recipe path that produced this output
    topology_class: str
    extraction_empty: bool     # True if grep produced no output
    latency_ms: float

@dataclass(frozen=True)
class RecipeMount:
    recipe_path: str           # path to compiled .sh file
    topology_class: str
    recipe_hash: str           # sha256 of recipe content — for audit
    is_hardcoded: bool         # hardcoded vs compiler-generated

@dataclass(frozen=True)
class ExtractionQuality:
    topology_class: str
    token_reduction_pct: float
    signal_density: float      # non-whitespace chars / total chars
    empty_extraction: bool
    structured_field_count: int  # for JSON topologies
    recipe_hash: str
    run_id: str
```

All dataclasses are frozen. No mutation after construction. If you find yourself wanting to mutate a contract, you need a new contract.

---

### `exceptions.py`
Exception taxonomy for every failure mode the kernel can produce. Typed exceptions only — no bare `raise Exception()` anywhere in the codebase.

```python
class KernelException(Exception): ...

class SubprocessTimeout(KernelException): ...
    # grep pipeline exceeded timeout_ms threshold
    # pipeline.py catches this and returns empty KernelOutput

class RecipeMountError(KernelException): ...
    # recipe file missing, unreadable, or failed validator.py check
    # hard stop — do not invoke kernel with invalid recipe

class EmptyExtractionError(KernelException): ...
    # grep pipeline produced zero bytes of output
    # not always an error — some pages have no signal in the expected zone
    # feedback.py uses this to score recipe quality

class ContainerSpawnError(KernelException): ...
    # Alpine container failed to start within spawn_timeout_ms
    # pipeline.py retries once, then raises to caller

class CheckpointWriteError(KernelException): ...
    # crond job failed to write archive
    # checkpoint_monitor.py detects this via integrity check on next cycle

class RecipeInjectionAttempt(KernelException): ...
    # validator.py detected shell injection pattern in compiler-generated recipe
    # hard stop — log full recipe content for forensic review
    # never pass to kernel

class StdinEncodingError(KernelException): ...
    # raw_content could not be encoded for stdin pipe
    # common with malformed HTML from hostile pages

class RestoreFailure(KernelException): ...
    # restore.sh failed or checkpoint archive is corrupt
    # TAG startup sequence catches this — falls back to next checkpoint
```

---

### `pipeline.py` — Opus
The most complex file in the kernel. Manages the full subprocess lifecycle for Process 1.

**Responsibilities:**

**Container lifecycle management.** Decides whether to reuse a warm container or spawn fresh. Warm reuse is valid for sequential invocations of the same topology class with the same recipe. Topology class change requires recipe remount — implementation decision: spawn fresh rather than attempt live remount. Fresh spawn latency on Alpine is milliseconds. The complexity of live remount is not worth it.

**Recipe mounting.** Before subprocess spawn, calls `recipes/registry.py` to resolve the recipe for the active topology class. Calls `recipes/validator.py` to verify the recipe is safe to execute. Mounts the validated recipe at `/recipe/run.sh` via Docker volume bind. Only after validation passes does the container start.

**Stdin/stdout piping.** Raw HTML or JSON is encoded to UTF-8 and written to the container's stdin. This is not trivial. Large pages — 500KB+ of HTML — require chunked writes. Subprocess can block on write if the pipe buffer fills before the grep pipeline consumes. Implement with `asyncio.subprocess` and proper `communicate()` handling, not manual `stdin.write()` + `stdout.read()`.

**Timeout enforcement.** Every invocation has a hard timeout — `timeout_ms` from config, default 5000ms. `asyncio.wait_for()` wraps the `communicate()` call. On timeout: kill the subprocess, return `KernelOutput` with `extraction_empty=True`, raise `SubprocessTimeout` for the caller to handle. Never let a hung grep pipeline block the AXIOM graph.

**Output capture.** stdout is the clean signal. stderr is the error channel. Both are captured. stderr content is logged at WARN level with the recipe hash and topology class — this is diagnostic signal, not noise. If stderr is non-empty and stdout is empty, `EmptyExtractionError` is raised. If both are non-empty, stdout is returned as signal and stderr is logged.

**Telemetry emission.** Every invocation produces a structured log entry: topology class, recipe hash, raw byte count, clean byte count, estimated token delta, latency ms, container spawn time, empty extraction flag. These feed Witness. This is not optional telemetry — it is the data product that validates whether kernel integration is performing as designed.

**Error handling contract.** `pipeline.py` never raises to its caller except `RecipeMountError` and `RecipeInjectionAttempt` — both are hard stops requiring human review. All other exceptions are caught, logged, and returned as `KernelOutput` with `extraction_empty=True`. The AXIOM graph must continue. A failed kernel invocation degrades gracefully to conventional extraction — it does not stop the run.

---

### `feedback.py` — Opus
Computes `ExtractionQuality` from `KernelOutput` and emits it as a training signal back to `topology_parser.py`. This is the file where the learning loop either compounds correctly or silently drifts.

**Responsibilities:**

**Token delta computation.** Estimates token count of `raw_content` vs `clean_signal` using a character-level approximation — `len(text) / 4` is sufficient for feedback purposes. Does not call a tokenizer. Does not call an LLM. Deterministic arithmetic.

**Signal density scoring.** `non_whitespace_chars(clean_signal) / len(clean_signal)`. High density means the recipe extracted real content. Low density means the grep pipeline left mostly whitespace — the recipe may be over-stripping or the page had no signal in the expected zone. Threshold below which recipe is flagged for review: configurable, default `0.35`.

**Empty extraction handling.** `extraction_empty=True` on `KernelOutput` is a quality signal not an error. Some pages in a topology class have no content in the expected structural zone — paywalled content that rendered a login wall, JavaScript-rendered content that the static fetcher received as empty body, A/B test variants with non-standard DOM. `feedback.py` tracks empty extraction rate per topology class. Above threshold, the recipe is flagged for recompilation by `topology_parser.py`.

**Structured field preservation.** For JSON topology classes — `REST_API_JSON`, `JSON_LD_STRUCTURED` — the clean signal should contain specific keys the recipe is designed to extract. `feedback.py` counts expected keys present in `clean_signal`. Missing keys mean the grep recipe is under-extracting. This count feeds recipe quality scoring.

**Per-class quality history.** `feedback.py` maintains a rolling window of quality scores per topology class. Not a database. A Python deque in memory. Size configurable, default last 100 extractions per class. When `topology_parser.py` queries for recipe performance, it reads from this window. When TAG restarts, the window is empty — quality history is not persisted. This is correct: stale quality history from a previous session is worse than no history.

**Feedback emission.** `feedback.py` does not call `topology_parser.py` directly. It emits a structured `FeedbackEvent` that `topology_parser.py` consumes. Decoupled. `topology_parser.py` can be updated, restarted, or replaced without touching `feedback.py`.

---

### `recipes/validator.py` — Opus
Security membrane between `topology_parser.py`'s compiled recipes and the execution environment. Every recipe passes through this before touching the kernel. Every recipe. No exceptions.

**Responsibilities:**

**Shell injection detection.** The recipe is a shell script assembled by `topology_parser.py` from learned patterns. The patterns are learned from web page structures — web pages can be adversarial. A hostile page could in theory produce a DOM structure that causes the recipe compiler to emit a shell injection payload. `validator.py` is the last line of defense.

Patterns that fail validation immediately with `RecipeInjectionAttempt`:

```python
INJECTION_PATTERNS = [
    r'\$\(',        # command substitution $(...)
    r'`[^`]+`',     # backtick execution
    r';\s*rm\s',    # rm after semicolon
    r';\s*curl\s',  # curl after semicolon
    r';\s*wget\s',  # wget after semicolon
    r'>\s*/etc',    # writes to /etc
    r'>\s*/usr',    # writes to /usr
    r'\|\s*bash',   # pipe to bash
    r'\|\s*sh\s',   # pipe to sh
    r'eval\s',      # eval
    r'exec\s',      # exec within recipe
    r'/dev/tcp',    # network via /dev/tcp
    r'/proc/self',  # proc filesystem access
]
```

On detection: log full recipe content with topology class and compiler metadata for forensic review. Raise `RecipeInjectionAttempt`. Do not pass the recipe to the kernel under any circumstances.

**Structural validation.** A recipe that passes injection checks still needs to be structurally valid shell.

- Must use only allowed commands: `grep`, `sed`, `awk`, `cat`, `cut`, `tr`, `head`, `tail`, `sort`, `uniq`. Any other command in the recipe fails validation.
- Must have at least one `grep` or `sed` invocation — a recipe with no transformation is nonsensical.
- Must not exceed max line count — configurable, default 50 lines. Recipes over this limit indicate the compiler has gone wrong.
- Must be valid UTF-8.

**Dry-run validation.** For compiler-generated recipes only — not hardcoded. After structural validation passes, run the recipe against three canonical test pages per topology class stored in `recipes/test_fixtures/`. If the recipe produces empty output on all three, it fails validation. Hardcoded recipes skip dry-run — they were validated by hand before commit.

**Hardcoded recipe protection.** Hardcoded recipes in `recipes/hardcoded/` are immutable at runtime. `validator.py` verifies their sha256 hash against a manifest on every load. If the hash does not match, `RecipeMountError` is raised. The hardcoded recipes cannot be modified by any runtime process — only by a deliberate code change with a corresponding manifest update.

---

### `recipes/registry.py` — Sonnet
Manages the recipe lookup table. Loads hardcoded recipes on startup. Accepts registrations from `topology_parser.py` as new compiler-generated recipes are validated. Provides `get_recipe(topology_class)` with fallback logic.

**Fallback logic is critical.** When the requested topology class has no recipe:

1. Check if any registered recipe covers a parent class — `NEWS_ARTICLE_PAYWALLED` falls back to `NEWS_ARTICLE`.
2. If no parent class match, use `GENERIC_HTML` — a maximally conservative recipe that strips known noise zones without assuming anything about signal location.
3. Never return `None`. Always return a recipe. The caller should never have to handle a missing recipe — the registry handles it.

`GENERIC_HTML` is the safety net. It is not good — it will over-retain noise. It is better than passing raw HTML to Haiku. It exists so the kernel always executes something useful even on completely unknown topology.

---

### `checkpoint/mft_checkpoint.sh`
Runs every 15 minutes via Alpine `crond`. Reads the four index files. Writes a timestamped compressed archive. Rotates.

```sh
#!/bin/sh
CHECKPOINT_DIR=/store/checkpoints
STORE_DIR=/store
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
ARCHIVE=$CHECKPOINT_DIR/mft_$TIMESTAMP.tar.gz

# verify source files exist before attempting archive
for f in topology_router.pt recipe_registry.mmap phase_states.mmap structural_layer.pt; do
    if [ ! -f "$STORE_DIR/$f" ]; then
        echo "CHECKPOINT SKIP: $f not found at $(date)" >> /var/log/checkpoint.log
        exit 1
    fi
done

tar -czf $ARCHIVE \
    -C $STORE_DIR \
    topology_router.pt \
    recipe_registry.mmap \
    phase_states.mmap \
    structural_layer.pt

# verify archive integrity immediately after write
if ! tar -tzf $ARCHIVE > /dev/null 2>&1; then
    echo "CHECKPOINT CORRUPT: $ARCHIVE at $(date)" >> /var/log/checkpoint.log
    rm -f $ARCHIVE
    exit 1
fi

echo "CHECKPOINT OK: $ARCHIVE at $(date)" >> /var/log/checkpoint.log

# rotation — keep last 48 (12 hours at 15min intervals)
ls -t $CHECKPOINT_DIR/mft_*.tar.gz 2>/dev/null | tail -n +49 | xargs -r rm -f
```

The integrity check after write is non-negotiable. A corrupt archive that passes silently is worse than a failed checkpoint that gets logged. If the archive is corrupt, it is deleted immediately and the next crond cycle will attempt a fresh one.

---

### `checkpoint/restore.sh`
Called by TAG's startup sequence via `checkpoint_monitor.py` when index files are missing or corrupted. Finds the most recent valid checkpoint and extracts it.

```sh
#!/bin/sh
CHECKPOINT_DIR=/store/checkpoints
STORE_DIR=/store

echo "RESTORE: scanning checkpoints at $(date)" >> /var/log/checkpoint.log

# find most recent valid archive
for archive in $(ls -t $CHECKPOINT_DIR/mft_*.tar.gz 2>/dev/null); do
    if tar -tzf $archive > /dev/null 2>&1; then
        echo "RESTORE: using $archive" >> /var/log/checkpoint.log
        tar -xzf $archive -C $STORE_DIR
        echo "RESTORE OK: $(date)" >> /var/log/checkpoint.log
        exit 0
    else
        echo "RESTORE SKIP: $archive corrupt" >> /var/log/checkpoint.log
    fi
done

echo "RESTORE FAILED: no valid checkpoint found at $(date)" >> /var/log/checkpoint.log
exit 1
```

Iterates newest to oldest. Skips corrupt archives. Stops at the first valid one. If no valid checkpoint exists, exits with code 1. TAG's startup sequence catches exit code 1 and handles it — either initializing a fresh index or raising `RestoreFailure` to the operator.

---

### `checkpoint/checkpoint_monitor.py` — Sonnet
Python-side health monitoring for the checkpoint system. Runs on TAG startup and periodically thereafter.

**Responsibilities:**

- On startup: verify all four index files exist and are readable. If missing or unreadable, invoke `restore.sh` via subprocess.
- Verify `restore.sh` exit code. If non-zero, raise `RestoreFailure`.
- After restore completes, verify restored files are valid — `.pt` files can be loaded by torch, `.mmap` files are non-empty and correctly sized.
- Periodically verify checkpoint log is being written — if no new log entry in 20 minutes, the crond process has died. Restart it. Emit an alert to Witness.
- Expose checkpoint health via a simple status dict consumed by TAG's telemetry: `last_checkpoint_time`, `checkpoint_count`, `latest_archive_size_bytes`, `restore_invoked_at_startup`.

---

### Hardcoded Recipes
Five shell scripts in `recipes/hardcoded/`. These are written by hand. They are the proof-of-concept validation that the kernel insight is real before `topology_parser.py` is built. They are never overwritten by the recipe compiler at runtime — the registry treats them as read-only with hash verification.

**`news_article.sh`** — signal zone is `<article>`. Strip nav, aside, footer. Remove all tags. Collapse whitespace.

**`saas_docs.sh`** — signal zone is `<main>`. Strip sidebar patterns. Preserve code blocks — `<pre>`, `<code>` zones contain signal in documentation pages, not noise.

**`rest_api_json.sh`** — extract `data`, `results`, `items` envelope keys. Discard `pagination`, `meta`, `links`, `_links`, `cursor` keys.

**`json_ld.sh`** — extract `<script type="application/ld+json">` content from `<head>` only. Discard any JSON-LD found in `<body>` — it is injected noise.

**`ecommerce.sh`** — signal near `data-product`, `data-price`, `data-sku`, `data-name` attributes. Discard `data-analytics`, `data-tracking`, `data-gtm` attributes entirely.

Each recipe is the simplest possible shell pipeline that correctly extracts signal for that topology class. Clarity over cleverness. These recipes will be read by humans diagnosing extraction failures.

---

## Dockerfile

```dockerfile
FROM alpine:3.19

RUN apk add --no-cache \
    grep \
    sed \
    gawk \
    coreutils \
    util-linux \
    dcron \
    tar \
    gzip \
    rsync

COPY checkpoint/mft_checkpoint.sh /etc/periodic/15min/mft_checkpoint
RUN chmod +x /etc/periodic/15min/mft_checkpoint

COPY checkpoint/restore.sh /usr/local/bin/restore
RUN chmod +x /usr/local/bin/restore

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

**entrypoint.sh:**

```sh
#!/bin/sh
crond -f -l 8 &
CROND_PID=$!
exec /bin/sh /recipe/run.sh
kill $CROND_PID
```

`crond -f` runs in foreground, backgrounded with `&`. `exec` replaces the shell with the pipeline process — proper signal propagation. SIGTERM reaches the pipeline directly. crond is killed cleanly on exit. Without `exec` you get the shell sitting between Docker and the pipeline, consuming signals meant for the process that matters.

---

## docker-compose.yml

```yaml
version: "3.9"

services:
  signal_kernel:
    build:
      context: ..
      dockerfile: Dockerfile
    container_name: axiom_signal_kernel
    restart: unless-stopped

    volumes:
      - type: bind
        source: ./recipes/active
        target: /recipe
        read_only: true

      - type: bind
        source: ../store
        target: /store
        read_only: false

      - type: bind
        source: ../store/checkpoints
        target: /store/checkpoints
        read_only: false

    environment:
      - CHECKPOINT_INTERVAL=15min
      - CHECKPOINT_RETAIN=48
      - LOG_LEVEL=warn

    deploy:
      resources:
        limits:
          cpus: "0.5"
          memory: 128M
        reservations:
          cpus: "0.1"
          memory: 32M

    network_mode: none

    stdin_open: true

    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

---

## Non-Negotiable Architectural Rules

**1. `network_mode: none` is permanent.**
The kernel has no business touching a network. It receives HTML via stdin from `pipeline.py`. If something in the kernel ever needs network access, the architecture has gone wrong — investigate rather than accommodate.

**2. Recipe mount is always read-only.**
The kernel executes recipes. `topology_parser.py` writes them. These are different processes with different responsibilities. The container physically cannot modify what it executes. This closes the injection surface that `validator.py` guards against on the Python side.

**3. Every recipe passes `validator.py` before execution. No exceptions.**
Not just compiler-generated recipes. All recipes. Including hardcoded ones on first load — hash verified against manifest. The validator is not a suggestion.

**4. `pipeline.py` never raises to the AXIOM graph except `RecipeMountError` and `RecipeInjectionAttempt`.**
Everything else is caught, logged, and returned as `KernelOutput` with `extraction_empty=True`. The graph continues. Kernel failure degrades gracefully. It never stops a run.

**5. `feedback.py` uses no LLM and calls no external service.**
Quality scoring is deterministic arithmetic. If you find yourself wanting to call a model to evaluate extraction quality, stop — you have introduced circular evaluation where an LLM evaluates LLM input quality. The kernel's quality loop is intentionally model-free.

**6. Process 1 and Process 2 share nothing.**
They share an OS. They share a Docker volume mount. They do not share state, file handles, pipe endpoints, or error channels. If you find yourself writing code that coordinates between them, you are building the wrong thing.

**7. Checkpoint integrity is verified immediately after every write.**
A corrupt archive that passes silently is worse than a logged failure. `mft_checkpoint.sh` verifies its own output. If verification fails, the archive is deleted and the next crond cycle retries.

**8. The store directory is owned by Python processes in TAG.**
The kernel's checkpoint daemon reads from `/store` and writes archives to `/store/checkpoints/`. It does not write to the four index files directly. `topology_router.pt` and the three `.mmap` files are written exclusively by `index_daemon.py` and `topology_parser.py` in the Python layer. The kernel observes the store — it does not modify it.

---

## The Store — Four Files, Everything TAG Knows

```
tag/store/
    topology_router.pt      # tiny MLP weights — IS the MFT index
                            # query embedding → forward pass → routing vector
                            # fine-tuned by RL loop — gradient step = index update
                            # no bulk reindex. no rebuild. just training.

    recipe_registry.mmap    # memory-mapped binary — exact recipe lookup
                            # mmap'd — OS handles paging, access looks like memory reads
                            # survives process death intact on disk
                            # on restart: mmap same file, registry immediately available

    phase_states.mmap       # per topology class phase tracking
                            # which classes are Phase I, II, III
                            # updated by world_model.py on every surprise evaluation

    structural_layer.pt     # hivemind shared weights — invariant primitives
                            # federated via rsync across fleet instances
                            # updated rarely — primitives are structurally stable
                            # rsync is sufficient — no message broker, no Redis

    checkpoints/
        mft_20260305_120000.tar.gz
        mft_20260305_121500.tar.gz
        ...                 # 48 rotating archives — 12 hours of history
```

No Redis. No Qdrant. No external service. No trust required. The index does not live in a database — it is encoded in weights and memory-mapped files that live on disk you own. Back them up however you want. RAID them. Copy to cold storage. Restore from any checkpoint in seconds. Copy the four files to a new machine and TAG has its full compiled knowledge immediately.

---

## Build Sequence Within signal_kernel

Write in this order. Each file depends on the previous.

```
1. contracts.py                 types first — everything imports from here
2. exceptions.py                error taxonomy — everything raises from here
3. recipes/hardcoded/*.sh       five files, written by hand — validate the concept
4. pipeline.py                  subprocess lifecycle — depends on contracts + exceptions
5. recipes/registry.py          recipe lookup — depends on contracts
6. recipes/validator.py         security membrane — depends on contracts + exceptions
7. feedback.py                  quality signal — depends on contracts, imports registry
8. checkpoint/mft_checkpoint.sh shell — no dependencies
9. checkpoint/restore.sh        shell — no dependencies
10. checkpoint/checkpoint_monitor.py  depends on contracts + exceptions
11. Dockerfile + entrypoint.sh  infra — written last, everything else must exist first
12. docker-compose.yml          infra — written last
```

Do not skip the hardcoded recipes. They are not scaffolding to be deleted later — they are permanent fixtures that prove the concept before the compiler exists and serve as regression baselines forever.

---

## Testing Philosophy

**Before `topology_parser.py` exists:**
Test the hardcoded recipes manually. Feed real pages. Read the output. Verify with your eyes that noise is gone and signal is present. This is the validation that the kernel insight is real. Do this before writing any Python.

**Token delta verification:**
On a typical news article page: measure raw HTML byte count, measure clean signal byte count, compute reduction percentage. Target is 60-80%. Below 40% means the recipe is under-stripping. Above 90% means the recipe is likely over-stripping and discarding signal.

**Empty extraction is not always a failure:**
A paywalled page that renders a login wall will produce empty extraction on a `NEWS_ARTICLE` recipe. This is correct behavior — the page has no signal in the expected zone. The recipe is not broken. `feedback.py` tracks empty extraction rate per topology class. Consistent empty extraction above threshold on a class that should have signal is a recipe quality problem.

**Subprocess timeout testing:**
Test with pages up to 1MB of raw HTML. Verify that `pipeline.py` correctly enforces `timeout_ms`. Verify that a timed-out invocation returns `KernelOutput` with `extraction_empty=True` without hanging the caller. This is the most common failure mode in production.

**Injection attempt testing:**
Feed `validator.py` recipes containing every pattern in `INJECTION_PATTERNS`. Verify `RecipeInjectionAttempt` is raised on every one. This test suite is permanent — run it on every code change to `validator.py`.

**Checkpoint integrity:**
Write a checkpoint manually. Corrupt the archive with a random byte flip. Verify `restore.sh` skips the corrupt archive and finds the next valid one. Verify `checkpoint_monitor.py` detects and logs the corruption.

---

## What Gets Surfaced To AXIOM

`pipeline.py` returns `KernelOutput` to `pipeline.py`'s caller in TAG's Python layer. That output passes to `feedback.py` which emits `ExtractionQuality` to `topology_parser.py`.

The AXIOM graph never directly touches the kernel. The graph node that invokes TAG receives a `DaemonResponse` from `interface.py`. Inside that response is `traversal_config`, `source_priority`, `friction_forecast`, and the resolved `recipe` path. The graph passes `recipe` and raw HTML to `pipeline.py` as part of TAG's internal operation. From the graph's perspective: query goes in, clean signal comes out. The kernel does not exist at the graph level.

---

## What Success Looks Like

After the kernel is built and working against real pages:

- Five topology classes producing 60-80% token reduction on canonical test pages
- `validator.py` catching injection attempts on every pattern in the test suite
- `pipeline.py` correctly handling subprocess timeout without hanging the caller
- `feedback.py` producing meaningful quality scores that differ between good and bad extractions
- Checkpoint daemon writing archives every 15 minutes with self-verified integrity
- `restore.sh` correctly recovering from a simulated index corruption
- `GENERIC_HTML` fallback recipe producing non-empty output on pages with no registered topology class

When all of that is true, the kernel is ready. `topology_parser.py` can be built on top of a validated execution layer instead of an assumed one.

---

*AXIOM Core Searching Algorithm — signal_kernel*
*index_daemon.py // topology_parser.py // signal_kernel/ // interface.py*
*AXIOM INTERNAL // DO NOT SURFACE*
