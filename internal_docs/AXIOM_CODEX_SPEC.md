# AXIOM — COMPLETE BUILD SPECIFICATION FOR CODEX
# Total remaining target: ~50,000 LOC of tested, production code
# Single binary. One inference point. Terminal UI. Replace RAG entirely.

---

## WHAT AXIOM IS

AXIOM is a Topology-Addressed Generation (TAG) system. It replaces RAG.

RAG: embed documents → store in vector DB → retrieve by similarity → feed to LLM.
AXIOM: the model weights ARE the index. No vector DB. No embeddings. No retrieval step.
The web is a typed system. Every URL belongs to a topology class. AXIOM knows the class
before it fetches. It dissects pages surgically. Output is signal not noise.

The user interface is a terminal. One binary. One calling point.

```
axiom> search | latest breakthroughs in RNA folding
axiom> fetch | https://arxiv.org/abs/2401.00001
axiom> learn | stripe.com
axiom> status |
axiom> quit |
```

Command left of pipe. Query right of pipe. Single process handles everything.

---

## WHAT IS ALREADY BUILT (DO NOT REBUILD)

```
tag/signal_kernel/          DONE — Alpine container, grep recipes, Stripe 1.1MB→8.4KB
tag/crawler_bus.py          DONE — Kafka + asyncio.Queue dual sink, HMAC, circuit breaker
tag/store_watchdog.py       DONE
tag/world_model/            DONE — mamba_router.py, wlm_tokenizer.py, wlm_decoders.py,
                                   latent_model.py, latent_parser.py
tag/topology/classifier.py  DONE
tag/topology/parser.py      DONE — 6K LOC, IntentConditionedExtractor
tag/topology/sanitizer.py   DONE — 6K LOC, 100-step pipeline, XSS/CSRF/SSRF/CRLF/SQLi
tag/topology/surprise_detector.py  DONE — 10 PhD algorithms, 0.15x steady state latency
crawler/bloom_filter.py     DONE — 44,174 URLs/sec, 0 false positives at 5M URLs
crawler/crawl_cursor.py     DONE
crawler/rate_limiter.py     DONE
crawler/frontier.py         DONE
crawler/fetcher.py          DONE — 8K LOC, 80/80 tests, Observatory, SEP, Tor
```

STORE FILES (4 files own everything, no Redis, no external services):
```
/store/topology_router.pt       — Mamba SSM weights for topology routing
/store/recipe_registry.mmap     — compiled extraction recipes per topology class
/store/phase_states.mmap        — per-domain phase state (COLD/LEARNING/KNOWN)
/store/structural_layer.pt      — structural intelligence weights (updated by offline/)
```

---

## ARCHITECTURE CONSTANTS (NEVER VIOLATE)

1. classifier never fetches
2. parser never executes arbitrary code
3. sanitizer never uses LLM
4. surprise fires on divergence not failure
5. index_daemon async fire-and-forget
6. phases earned not assigned
7. interface.py is the only public surface
8. all store writes use staging + atomic rename
9. hmac.compare_digest never ==
10. os.fsync on dead letter writes
11. WLM and WLP always called via asyncio.gather(), never sequential
12. fetcher runs inside gVisor container, Tor SOCKS5 at 127.0.0.1:9050

---

## LANGUAGE ALLOCATION

Each file is assigned a language based on what it does at runtime.
Codex should not second-guess these allocations.

```
LANGUAGE        REASON                          FILES
────────────────────────────────────────────────────────────────────
Go              Network I/O, concurrency        preparser/domain_analyzer.go
                goroutines outperform asyncio   preparser/crawl_planner.go
                at crawler scale                preparser/signal_extractor.go
                                                preparser/recipe_validator.go

C               System daemons, mmap I/O,       daemons/phase_daemon.c
                zero-overhead hot paths,        daemons/store_sentinel.c
                alpine_strip stripping          alpine_strip/strip_engine.c
                                                alpine_strip/batch_runner.c
                                                cold_start.c (partial)

CUDA + C        GPU batch encoding,             offline/gpu_encoder.cu
                structural_layer.pt updates,    offline/batch_scheduler.c
                offline heavy lifting           offline/weight_updater.cu
                                                offline/gradient_accumulator.cu

Python          RL loop logic (needs numpy/     index_daemon.py
                torch), bus integration,        cold_start.py (orchestrator)
                public interface                interface.py

Rust            Terminal UI — memory safe,      axiom_tui/ (full crate)
                ncurses-level control,          src/main.rs
                single binary compilation,      src/ui.rs
                crossterm for rendering         src/repl.rs
                                                src/logo.rs
                                                src/dispatcher.rs
```

---

## SECTION 1: PREPARSER — Go

### What preparser/ does

The preparser sits between the crawler and the learning loop.
Crawler (fetcher.py) acquires raw bytes. Preparser extracts structural intelligence
before the topology layer ever sees a page. It answers: what IS this domain structurally?

The preparser is the system's geological survey. It maps terrain before anyone walks it.

### File: preparser/domain_analyzer.go

**LOC target:** 2,000–2,500
**Tests:** 40 minimum
**Language:** Go

**Responsibility:**
Analyze a domain's structural fingerprint from accumulated raw fetch results.
Does not fetch. Does not parse content. Reads from the crawl cursor store only.

**What it produces:**
```go
type DomainFingerprint struct {
    Domain              string
    DominantTopology    string          // most common topology class seen
    TopologyDistribution map[string]float64
    URLPatterns         []URLPattern    // discovered URL structure patterns
    RobotsSignals       RobotsAnalysis  // robots.txt as structural signal
    ContentLanguage     string
    AvgResponseSize     int64
    AvgLatencyMs        float64
    FrictionLevel       int             // 0-3, maps to fetcher CL levels
    PhaseRecommendation string          // "COLD" | "LEARNING" | "KNOWN"
    ConfidenceScore     float64
    ObservationCount    int
    LastAnalyzedAt      time.Time
}
```

**Key algorithms:**
- URL pattern extraction via trie compression — same _PatternTrie concept as fetcher SEP
- robots.txt analysis: Disallow density, Crawl-delay as friction signal, sitemap presence
- Response size distribution via Welford online variance (port from surprise_detector)
- Topology distribution via frequency counting with Laplace smoothing

**What it reads:**
- crawl cursor store (domain history from crawl_cursor.py exported format)
- raw fetch metadata (not content — just headers, status, size, latency)

**What it writes:**
- `/store/domain_fingerprints.mmap` — fixed-slot binary format, 256 bytes per domain
  slot layout: `[topology_class_idx:1][friction:1][phase_rec:1][confidence:4][obs_count:4][reserved:245]`

**Interface:**
```go
func AnalyzeDomain(domain string, history []FetchRecord) (*DomainFingerprint, error)
func BatchAnalyze(domains []string, store CursorStore) ([]*DomainFingerprint, error)
func ReadFingerprint(domain string, mmapPath string) (*DomainFingerprint, error)
```

**Tests must cover:**
- Welford variance convergence after 100 observations
- URL pattern trie compression correctness
- robots.txt friction level extraction (all four CL levels)
- Phase recommendation logic at each observation count threshold
- mmap write/read round-trip integrity
- Concurrent BatchAnalyze with 1000 domains (goroutine safety)

---

### File: preparser/crawl_planner.go

**LOC target:** 1,500–2,000
**Tests:** 35 minimum
**Language:** Go

**Responsibility:**
Given a domain fingerprint, produce a crawl plan — which URLs to fetch next,
in what order, at what rate, with what friction level.
This is the system's tactical layer. It reads terrain (fingerprint), produces orders (plan).

**What it produces:**
```go
type CrawlPlan struct {
    Domain          string
    Priority        float64         // 0.0-1.0, computed from phase + signal density
    URLQueue        []PlannedURL
    RateLimit       time.Duration   // computed from friction + robots Crawl-delay
    MaxConcurrency  int
    FrictionLevel   int             // CL1-CL4
    ResumeToken     string          // for crawl_cursor resumability
    EstimatedSignal float64         // expected signal yield from this plan
    PlanGeneratedAt time.Time
}

type PlannedURL struct {
    URL             string
    ExpectedClass   string          // topology class prediction
    Priority        float64
    RetryCount      int
    LastAttemptedAt *time.Time
}
```

**Crawl plan priority formula:**
```
priority = (signal_density × phase_weight × freshness_decay) / friction_cost
where:
  signal_density  = known_signal_zones / total_urls_seen (from fingerprint)
  phase_weight    = {COLD: 0.3, LEARNING: 0.6, KNOWN: 1.0}
  freshness_decay = exp(-λ × days_since_last_crawl), λ = 0.1
  friction_cost   = {CL1: 1.0, CL2: 1.5, CL3: 2.5, CL4: 4.0}
```

**What it reads:** DomainFingerprint from domain_analyzer.go
**What it writes:** plan to an in-memory queue consumed by frontier.py via Unix socket

**Interface:**
```go
func GeneratePlan(fp *DomainFingerprint, opts PlanOptions) (*CrawlPlan, error)
func PrioritizePlan(plans []*CrawlPlan) []*CrawlPlan  // sort by priority desc
func SerializePlan(plan *CrawlPlan) ([]byte, error)   // JSON for frontier socket
```

**Tests must cover:**
- Priority formula numerical correctness at all phase/friction combinations
- URL queue ordering by priority
- Rate limit respects robots.txt Crawl-delay minimum
- ResumeToken generation and reconstruction
- Plan serialization round-trip

---

### File: preparser/signal_extractor.go

**LOC target:** 1,800–2,200
**Tests:** 40 minimum
**Language:** Go

**Responsibility:**
Given a sanitized signal (output of sanitizer.py — already cleaned, already reduced),
extract structured signal zones. What are the actual information-bearing regions?
This is the pre-index step. It runs before index_daemon. It identifies WHAT to index.

**Signal zone types:**
```go
type ZoneType string

const (
    ZoneCode       ZoneType = "code"
    ZoneProse      ZoneType = "prose"
    ZoneTable      ZoneType = "table"
    ZoneList       ZoneType = "list"
    ZoneHeading    ZoneType = "heading"
    ZoneMetadata   ZoneType = "metadata"
    ZoneDefinition ZoneType = "definition"
)

type SignalZone struct {
    ID          string
    Type        ZoneType
    Content     string
    ByteOffset  int
    ByteLen     int
    Density     float64     // information density score
    Language    string      // for code zones: detected language
    Confidence  float64
}

type ExtractedSignal struct {
    URL         string
    Domain      string
    TopologyClass string
    Zones       []SignalZone
    TotalSignalBytes int
    ReductionRatio   float64
    ExtractedAt time.Time
}
```

**Algorithm:**
- Prose vs code split using statistical features (entropy, line length distribution, punctuation density)
- Table detection via whitespace alignment analysis
- Heading detection via length + capitalization patterns
- Information density = (unique_words / total_words) × (1 - compression_ratio)
- Language detection for code blocks via keyword frequency

**What it reads:** sanitized bytes from sanitizer.py output (via crawler_bus)
**What it writes:** ExtractedSignal to bus as `SignalExtractedEvent`

**Tests must cover:**
- Prose/code split accuracy on known fixtures (10 fixture files minimum)
- Table detection correctness
- Density scoring monotonicity (denser content scores higher)
- Code language detection for Python, Go, JavaScript, Rust, C

---

### File: preparser/recipe_validator.go

**LOC target:** 1,200–1,500
**Tests:** 25 minimum
**Language:** Go

**Responsibility:**
Validate that a compiled recipe in recipe_registry.mmap still produces signal
on current page content. Recipes go stale when sites restructure.
This is the system's quality control for the recipe layer.

**What it does:**
- Reads a recipe from recipe_registry.mmap
- Runs it against a sample of recent fetch results for the domain
- Scores the yield: bytes extracted / bytes input
- If yield drops below threshold → emits RecipeStaleEvent to bus
- index_daemon picks this up and triggers recipe recompilation

**Validity thresholds:**
```go
const (
    MinimumYieldRatio    = 0.001  // at least 0.1% signal extraction
    MaximumYieldRatio    = 0.30   // >30% means recipe is too broad (noise)
    StaleWindowSize      = 20     // validate against last 20 fetches
    StaleThresholdFactor = 0.5    // yield dropped to <50% of historical avg
)
```

**Events emitted:**
```go
type RecipeStaleEvent struct {
    Domain          string
    TopologyClass   string
    HistoricalYield float64
    CurrentYield    float64
    DropFactor      float64
    SampleSize      int
    DetectedAt      time.Time
}

type RecipeHealthEvent struct {
    Domain      string
    YieldRatio  float64
    SampleSize  int
    Status      string  // "healthy" | "degraded" | "stale"
}
```

**Tests must cover:**
- Stale detection triggers at correct yield drop threshold
- Healthy recipe does not trigger stale
- Edge case: zero yield (complete recipe failure)
- Window size boundary (exactly 20 samples)

---

## SECTION 2: ALPINE_STRIP — C

### What alpine_strip/ does

Alpine strip is the offline batch stripping layer. The online path uses signal_kernel
(Alpine container, per-URL, real-time). The offline path uses alpine_strip for
bulk reprocessing of already-fetched content when recipes change or improve.

When index_daemon decides a domain's recipes need recompilation, it queues all
cached raw bytes for that domain through alpine_strip for bulk reprocessing.

Both files compile to a shared library `libalpinestrip.so` and a standalone binary.

### File: alpine_strip/strip_engine.c

**LOC target:** 2,500–3,000
**Tests:** 50 minimum (C test runner: Unity or minunit)
**Language:** C (C11, -O3 -march=native)

**Responsibility:**
The core stripping engine. Takes raw HTML bytes, applies a recipe, returns signal bytes.
This is the offline equivalent of signal_kernel's online stripping.

**Key difference from signal_kernel:**
signal_kernel uses grep patterns inside Alpine containers (subprocess isolation).
strip_engine.c does the same work in-process, in C, for 100× throughput on batch jobs.
No subprocess. No Alpine container overhead. Direct regex + pattern matching in memory.

**Data structures:**
```c
typedef struct {
    uint8_t  *pattern;       // compiled regex pattern bytes
    size_t    pattern_len;
    uint8_t   flags;         // STRIP_EXTRACT | STRIP_REMOVE | STRIP_REQUIRE
    float     confidence;    // minimum confidence to apply this step
} StripStep;

typedef struct {
    StripStep *steps;
    size_t     step_count;
    char       topology_class[64];
    uint32_t   checksum;     // CRC32 of recipe — validates against registry
} Recipe;

typedef struct {
    uint8_t  *signal;        // output signal bytes
    size_t    signal_len;
    size_t    input_len;
    float     reduction_ratio;
    uint32_t  steps_fired;
    int       status;        // STRIP_OK | STRIP_EMPTY | STRIP_ERROR
} StripResult;
```

**API:**
```c
// Load a recipe from recipe_registry.mmap at a given slot index
Recipe* strip_load_recipe(int slot_idx, const char *mmap_path);

// Apply recipe to input bytes, write signal to pre-allocated output buffer
StripResult strip_apply(
    const Recipe *recipe,
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t output_capacity
);

// Free recipe resources
void strip_free_recipe(Recipe *recipe);

// Validate recipe checksum against registry
int strip_validate_recipe(const Recipe *recipe, const char *mmap_path);
```

**Implementation requirements:**
- Use PCRE2 for regex (link against libpcre2-8)
- All allocations go through a pool allocator (no malloc in hot path)
- Pool: 64MB pre-allocated at init, bump allocator, reset between batch items
- Zero-copy where possible: output buffer pre-allocated by caller
- Signal bytes written directly to output buffer — no intermediate copy
- Maximum output ratio: 0.30 (output > 30% of input → STRIP_TOO_BROAD, caller retries with tighter recipe)
- Thread-safe: multiple threads can call strip_apply simultaneously with different Recipe pointers
- Each thread must have its own pool (pass pool pointer as parameter)

**Tests must cover:**
- Empty input → STRIP_EMPTY
- Recipe with zero steps → output == input
- Reduction ratio computation
- PCRE2 pattern compilation failure handling
- Pool exhaustion (>64MB single item) → STRIP_ERROR
- CRC32 checksum validation
- Thread safety: 8 threads × 1000 calls each, no corruption

---

### File: alpine_strip/batch_runner.c

**LOC target:** 1,500–2,000
**Tests:** 30 minimum
**Language:** C (C11)

**Responsibility:**
Batch runner for strip_engine. Reads a work queue, processes items in parallel
using pthreads, writes results to an output directory.

**Work queue format:** newline-delimited JSON
```json
{"url": "https://example.com/1", "slot_idx": 3, "input_path": "/tmp/raw/abc123.raw", "output_path": "/tmp/strip/abc123.signal"}
{"url": "https://example.com/2", "slot_idx": 3, "input_path": "/tmp/raw/def456.raw", "output_path": "/tmp/strip/def456.signal"}
```

**Architecture:**
```
main thread: reads work queue → pushes to ring buffer queue
N worker threads (default: nproc): pop from queue → strip_apply → write output
stats thread: every 5 seconds prints throughput to stderr
```

**Ring buffer:**
```c
#define QUEUE_CAPACITY 4096  // must be power of 2

typedef struct {
    WorkItem items[QUEUE_CAPACITY];
    _Atomic uint64_t head;
    _Atomic uint64_t tail;
} RingBuffer;
```

**CLI:**
```bash
./batch_runner --queue /tmp/strip_queue.jsonl \
               --mmap /store/recipe_registry.mmap \
               --threads 16 \
               --pool-mb 64
```

**Exit codes:**
- 0: all items processed successfully
- 1: partial failure (some items failed, see stderr)
- 2: fatal error (queue unreadable, mmap unavailable)

**Tests must cover:**
- Empty queue → exit 0, no crash
- Single item round-trip
- Parallel processing correctness (output files match single-thread output)
- Queue full backpressure (producer blocks, doesn't drop)
- Graceful shutdown on SIGTERM (finish current batch, drain queue)

---

## SECTION 3: OFFLINE — CUDA + C

### What offline/ does

offline/ is the GPU-side learning loop. It takes signal extracted by the pipeline
and updates structural_layer.pt — the weights that make AXIOM smarter over time.

This is the closed loop: online pipeline extracts signal → offline/ trains on signal
→ structural_layer.pt improves → online pipeline extracts better signal.

All CUDA files compile to shared libraries. Python bindings via ctypes (no pybind11).

### File: offline/gpu_encoder.cu

**LOC target:** 2,000–2,500
**Tests:** 30 (CUDA test with cuBLAS verification)
**Language:** CUDA C++ (compute capability 8.9 minimum — RTX 5080/5090)

**Responsibility:**
Batch encode signal zones into embedding vectors for structural_layer.pt update.
This is NOT a general-purpose encoder. It encodes specifically into the shape
that structural_layer.pt expects: fixed (1, 256) hidden state vectors.

**Architecture:**
```cuda
// Input: batch of signal zone strings (tokenized)
// Output: batch of (1, 256) float32 embedding vectors

__global__ void encode_zones_kernel(
    const int    *token_ids,        // [batch_size, max_seq_len]
    const int    *seq_lengths,      // [batch_size]
    float        *embeddings,       // [batch_size, 256] output
    const float  *embedding_table,  // [vocab_size, 256] loaded from structural_layer.pt
    int           batch_size,
    int           max_seq_len,
    int           vocab_size
);

// Mean pooling over sequence dimension
__global__ void mean_pool_kernel(
    const float  *token_embeddings, // [batch_size, max_seq_len, 256]
    const int    *seq_lengths,
    float        *pooled,           // [batch_size, 256]
    int           batch_size,
    int           max_seq_len
);
```

**Exported C API (for ctypes binding):**
```c
// Called from Python via ctypes
int gpu_encoder_init(const char *structural_layer_path);
int gpu_encoder_encode_batch(
    const char **texts,       // array of null-terminated strings
    int          batch_size,
    float       *output,      // pre-allocated [batch_size × 256] float32
    int          max_seq_len
);
void gpu_encoder_shutdown(void);
int gpu_encoder_health(void);  // returns 1 if GPU ready, 0 if not
```

**Performance requirements:**
- Minimum throughput: 10,000 zones/second on RTX 5080
- Maximum batch latency: 50ms for batch_size=512
- Memory: stays within 4GB VRAM regardless of batch size (chunk internally)
- Mixed precision: FP16 compute, FP32 accumulation

**Tests must cover:**
- Single zone encoding shape correctness (1, 256)
- Batch size 1, 32, 512 all produce same per-item result
- Empty string input → zero vector (not crash)
- VRAM limit enforcement (batch > 512 auto-chunked)
- GPU not available → graceful fallback error code

---

### File: offline/weight_updater.cu

**LOC target:** 2,000–2,500
**Tests:** 25
**Language:** CUDA C++

**Responsibility:**
Apply gradient updates to structural_layer.pt using encoded signal zones.
This is online learning — not offline training. Small batches, frequent updates.
Target: update weights within 60 seconds of signal being extracted.

**Learning algorithm:** AdamW with gradient clipping
```cuda
__global__ void adamw_update_kernel(
    float       *weights,       // [vocab_size, 256] — structural_layer.pt subset
    float       *grad,          // [vocab_size, 256] — accumulated gradient
    float       *m,             // first moment
    float       *v,             // second moment
    float        lr,            // learning rate (default: 1e-4)
    float        beta1,         // default: 0.9
    float        beta2,         // default: 0.999
    float        eps,           // default: 1e-8
    float        weight_decay,  // default: 0.01
    int          t,             // timestep
    int          n_elements
);

__global__ void gradient_clip_kernel(
    float       *grad,
    float        max_norm,      // default: 1.0
    int          n_elements,
    float       *grad_norm      // output: pre-clip norm
);
```

**Update cycle:**
1. Accumulate gradients from signal batch (gpu_encoder output)
2. Clip gradient norm
3. AdamW step
4. Write updated weights to staging file `/store/staging/structural_layer.pt.staging`
5. Call atomic rename (from C) to replace `/store/structural_layer.pt`
6. Emit `WeightsUpdatedEvent` to bus

**Exported C API:**
```c
int weight_updater_init(const char *structural_layer_path, float lr);
int weight_updater_accumulate(const float *embeddings, const int *labels, int batch_size);
int weight_updater_step(void);       // AdamW step + atomic rename
int weight_updater_checkpoint(void); // force write without step
void weight_updater_shutdown(void);
```

---

### File: offline/gradient_accumulator.cu

**LOC target:** 1,500–2,000
**Tests:** 20
**Language:** CUDA C++

**Responsibility:**
Accumulate gradients across multiple small batches before applying a weight update.
Gradient accumulation simulates larger effective batch sizes without requiring
the full batch to fit in VRAM at once.

**Why:**
A single signal zone batch may be 32-64 items. One AdamW step on 32 items
is noisy. Accumulate 16 batches (512 effective batch size) before stepping.

```cuda
// Ring buffer of gradient tensors on GPU
// Each call to accumulate() adds to the ring
// When ring is full, flush() is called automatically

typedef struct {
    float   *grad_buffer;    // [accumulate_steps, vocab_size, 256]
    float   *sum_buffer;     // [vocab_size, 256] — running sum
    int      step;
    int      accumulate_steps;  // default: 16
} GradAccumulator;

__global__ void accumulate_kernel(
    float       *sum_buffer,
    const float *new_grad,
    int          n_elements
);

__global__ void normalize_kernel(
    float       *grad,
    int          n_steps,
    int          n_elements
);
```

---

### File: offline/batch_scheduler.c

**LOC target:** 1,500–2,000
**Tests:** 30
**Language:** C

**Responsibility:**
Scheduler that coordinates the offline pipeline. Reads from a work queue
(items produced by index_daemon), dispatches to gpu_encoder → gradient_accumulator
→ weight_updater in sequence. Handles backpressure and failure recovery.

**Schedule loop:**
```
while (running):
    item = dequeue(work_queue, timeout=5s)
    if item == NULL: continue
    
    result = gpu_encoder_encode_batch(item.texts, item.batch_size, embeddings)
    if result != OK: log_error, push to dead_letter, continue
    
    weight_updater_accumulate(embeddings, item.labels, item.batch_size)
    
    if accumulator_full():
        weight_updater_step()
        emit_weights_updated_event()
    
    item_complete(item)
```

**CLI:**
```bash
./batch_scheduler --queue /tmp/offline_queue \
                  --store /store \
                  --accumulate-steps 16 \
                  --lr 1e-4 \
                  --dead-letter /tmp/offline_dead_letter.jsonl
```

---

## SECTION 4: DAEMONS — C

### What daemons/ does

Daemons are long-running background processes that maintain AXIOM's runtime health.
They are NOT part of the inference path. They run in parallel, monitor, and correct.

All daemons fork from cold_start (via C), run as separate processes,
communicate via Unix domain sockets.

### File: daemons/phase_daemon.c

**LOC target:** 2,000–2,500
**Tests:** 30
**Language:** C

**Responsibility:**
Monitor domain phase states and advance phases when promotion criteria are met.
Reads phase_states.mmap. Writes phase transitions (atomic, via staging+rename).
Emits PhaseTransitionEvent to bus when a domain promotes.

**Phase promotion criteria:**
```
COLD → LEARNING:
  - observation_count >= 50
  - dominant_class confidence >= 0.70
  - surprise_rate < 0.20 (fewer than 20% of events fire Condition 1 or 3)

LEARNING → KNOWN:
  - observation_count >= 200
  - dominant_class confidence >= 0.85
  - surprise_rate < 0.05
  - npi >= 0.70 (domain topology is predictable)
  - recipe_yield >= 0.005 (recipe producing signal)
```

**Run loop:**
- Scan all active domains in phase_states.mmap every 60 seconds
- For each domain: read stats, check criteria, promote if met
- Write new phase to staging, atomic rename
- Emit event to bus via Unix socket

**Data structures:**
```c
typedef struct {
    uint8_t  phase;              // 1=COLD, 2=LEARNING, 3=KNOWN
    uint8_t  flags;
    uint16_t padding;
    float    confidence;
    float    surprise_rate;
    uint32_t observation_count;
    float    npi;
    float    recipe_yield;
    uint64_t last_updated_unix;
    uint8_t  reserved[4];
} PhaseSlot;                     // 32 bytes, matches _PHASE_SLOT_BYTES in surprise_detector.py
```

**Tests must cover:**
- Promotion from COLD → LEARNING at exact threshold
- No promotion when any single criterion fails
- Atomic write does not corrupt mmap during concurrent reads
- PhaseTransitionEvent serialization

---

### File: daemons/store_sentinel.c

**LOC target:** 1,500–2,000
**Tests:** 25
**Language:** C

**Responsibility:**
Monitor all four store files for corruption, size anomalies, and access errors.
The sentinel is the canary in the coal mine for store health.

**Checks run every 30 seconds:**
1. File exists and is readable
2. File size within expected bounds (flag if >10% change in 5 minutes)
3. CRC32 checksum of header block (first 4KB) matches last known good
4. mmap can be opened and read without SIGBUS
5. Staging files do not persist for >10 minutes (stuck rename indicates crash)

**On failure:**
- Log structured error to `/var/log/axiom/sentinel.jsonl`
- Emit `StoreHealthEvent` to bus with failure details
- If critical failure (mmap unreadable): send SIGUSR1 to cold_start.py PID
  (cold_start.py handles SIGUSR1 by attempting store reconstruction)

**CLI:**
```bash
./store_sentinel --store /store \
                 --pid-file /var/run/axiom/sentinel.pid \
                 --log /var/log/axiom/sentinel.jsonl \
                 --bus-socket /tmp/axiom_bus.sock
```

---

## SECTION 5: INDEX DAEMON — Python

### File: index_daemon.py

**LOC target:** 5,000–6,000
**Tests:** 60 minimum
**Language:** Python (numpy, torch, asyncio)
**Builder:** Codex — this is the RL loop. The most complex remaining Python file.

**Responsibility:**
index_daemon is the online learning coordinator. It receives events from the bus,
reasons about what they mean for the store, and dispatches updates.

It does NOT perform gradient updates directly. It dispatches to offline/ for GPU work.
It DOES perform immediate lightweight updates to phase_states.mmap and recipe_registry.mmap.

**Event subscriptions:**
```python
SurpriseEvent          → from surprise_detector — classification divergence
FetchAnomalyEvent      → from fetcher — domain behavior anomaly
NewTopologyHintEvent   → from surprise_detector — potential new topology class
RecipeStaleEvent       → from recipe_validator.go — recipe yield degraded
SignalExtractedEvent   → from signal_extractor.go — new signal available
WeightsUpdatedEvent    → from batch_scheduler.c — GPU update complete
PhaseTransitionEvent   → from phase_daemon.c — domain promoted
ZoneMapInvalidatedEvent → internal — recipe needs recompilation
```

**Core RL loop logic:**

For each `SurpriseEvent`:
```python
async def _handle_surprise(self, event: SurpriseEvent):
    domain = event.domain
    cls = event.topology_class
    severity = event.severity
    
    # 1. Immediate: update surprise_rate in phase_states.mmap
    await self._update_surprise_rate(domain, cls)
    
    # 2. Decide gradient step worthiness
    if severity == HIGH and event.dissolve_triggered:
        # Confident wrong classification at KNOWN phase
        # Queue immediate gradient step for this domain's recipe
        await self._queue_gradient_step(domain, cls, event.contributing_signals, priority=HIGH)
    elif severity == MEDIUM:
        # Repeated structural anomaly — queue lower priority step
        await self._queue_gradient_step(domain, cls, event.contributing_signals, priority=MEDIUM)
    elif severity == LOW:
        # Single anomaly — accumulate, don't step yet
        await self._accumulate_signal(domain, cls, event.contributing_signals)
    
    # 3. If enough LOW severity accumulated → promote to MEDIUM
    if self._accumulated_low[domain][cls] >= LOW_ACCUMULATION_THRESHOLD:
        await self._queue_gradient_step(domain, cls, None, priority=MEDIUM)
        self._accumulated_low[domain][cls] = 0
```

For each `NewTopologyHintEvent`:
```python
async def _handle_topology_hint(self, event: NewTopologyHintEvent):
    # Trigger discover_signal_zones() — finds what signal exists in unknown topology
    zones = await self._discover_signal_zones(
        domain=event.domain,
        centroid_vector=event.centroid_vector,
        suggested_parent=event.suggested_parent_class,
        mdl_split=event.mdl_supports_split
    )
    
    if zones:
        # Draft new recipe for this topology pattern
        await self._draft_recipe(event.domain, zones, event.betti0_modes)
        await self._emit_zone_map_invalidated(event.domain)
```

**`discover_signal_zones()` algorithm:**
```python
async def _discover_signal_zones(self, domain, centroid_vector, suggested_parent, mdl_split):
    """
    Find signal zones in a domain that doesn't match any known topology class.
    
    Strategy:
    1. Fetch the last 5 raw bytes from this domain (from crawl cursor store)
    2. Run the parent class recipe as a starting point
    3. Measure yield — what did the parent recipe extract?
    4. Expand/contract recipe steps to improve yield
    5. Return discovered zones
    
    This is the creative step. index_daemon is not just routing signals.
    It is actively learning new topology classes from examples.
    """
```

**RL policy constants:**
```python
LOW_ACCUMULATION_THRESHOLD  = 10   # 10 LOW severity events → treat as MEDIUM
GRADIENT_QUEUE_MAX_SIZE     = 1000 # max pending gradient steps
GRADIENT_PRIORITY_HIGH      = 0    # immediate dispatch
GRADIENT_PRIORITY_MEDIUM    = 1    # dispatch within 60s
GRADIENT_PRIORITY_LOW       = 2    # batch with next scheduled run
RECIPE_DRAFT_CONFIDENCE_MIN = 0.60 # minimum zone confidence to draft recipe
```

**Store write policy (CRITICAL):**
- phase_states.mmap: write directly via struct.pack, atomic using mmap + msync
- recipe_registry.mmap: always staging + atomic rename
- structural_layer.pt: NEVER written by index_daemon — offline/ owns this
- topology_router.pt: NEVER written by index_daemon — offline/ owns this

**Tests must cover:**
- SurpriseEvent HIGH → gradient step queued immediately
- SurpriseEvent LOW × 10 → promotes to MEDIUM gradient step
- NewTopologyHintEvent → discover_signal_zones() called with correct args
- RecipeStaleEvent → ZoneMapInvalidatedEvent emitted
- Gradient queue backpressure (queue full → oldest LOW items dropped)
- Concurrent event handling (asyncio.gather, 50 simultaneous events)
- Store write atomicity (no partial writes under concurrent load)
- Phase transition correctly updates internal domain state
- discover_signal_zones() with mdl_split=True creates two draft recipes

---

## SECTION 6: COLD START — Python + C

### File: cold_start.py

**LOC target:** 2,000–2,500
**Tests:** 30
**Language:** Python (orchestrator) calling cold_start_c.so (C library for gVisor/iptables)

**Responsibility:**
The system's ignition sequence. Called once at startup. Sets up the entire runtime.
After cold_start.py returns successfully, the system is ready to serve queries.

**Startup sequence (strict order):**

```python
async def cold_start():
    # Phase 1: Environment validation (C library)
    validate_gvisor_environment()      # confirms running inside gVisor
    validate_gpu_available()           # CUDA device check
    validate_store_integrity()         # all 4 store files present and valid CRC
    
    # Phase 2: Network isolation (C library, must happen before fetcher starts)
    setup_iptables_fetcher_rules()     # OUTGOING: 9050 ALLOW, 9051 ALLOW, else DROP
                                       # INCOMING: everything DROP
    start_tor_daemon()                 # Tor SOCKS5 at 127.0.0.1:9050
    verify_tor_circuit()               # test circuit, retry 3× before abort
    
    # Phase 3: Store loading
    await load_topology_router()       # topology_router.pt → GPU memory
    await load_structural_layer()      # structural_layer.pt → GPU memory
    await mmap_phase_states()          # phase_states.mmap → mmap handle
    await mmap_recipe_registry()       # recipe_registry.mmap → mmap handle
    
    # Phase 4: Component initialization (parallel where safe)
    await asyncio.gather(
        crawler_bus.initialize(),
        surprise_detector.initialize(),
        classifier.initialize(),
    )
    
    # Phase 5: Start background processes (C daemons via subprocess)
    start_daemon("phase_daemon", args=["--store", STORE_PATH, ...])
    start_daemon("store_sentinel", args=["--store", STORE_PATH, ...])
    start_daemon("batch_scheduler", args=["--queue", OFFLINE_QUEUE, ...])
    
    # Phase 6: Start index_daemon (Python, asyncio)
    asyncio.create_task(index_daemon.run())
    
    # Phase 7: Self-test
    await run_cold_start_self_test()   # 5 known URLs, verify pipeline end-to-end
    
    # Phase 8: Start TUI (signals readiness)
    return SystemReady()
```

**SIGUSR1 handler (called by store_sentinel on critical failure):**
```python
def _handle_store_failure(signum, frame):
    # Attempt store reconstruction
    # Pause all writes, rebuild from scratch if possible
    # If reconstruction fails: controlled shutdown
```

**C library `cold_start_c.so` functions:**
```c
int validate_gvisor_environment(void);     // check /proc/self/cgroup for gVisor
int setup_iptables_fetcher_rules(void);    // iptables rules for fetcher isolation
int teardown_iptables_fetcher_rules(void); // cleanup on shutdown
int start_tor_daemon(const char *config);  // fork Tor, wait for SOCKS5 ready
int verify_tor_circuit(void);             // SOCKS5 connection test
int validate_store_crc(const char *path); // CRC32 header check
```

---

## SECTION 7: INTERFACE — Python

### File: interface.py

**LOC target:** 2,500–3,000
**Tests:** 40
**Language:** Python
**CRITICAL: This is the ONLY public surface. Everything else is internal.**

**Responsibility:**
Receives queries from the TUI, routes them through the full pipeline,
returns structured results. This is the inference point.

**Query types:**
```python
class QueryType(str, Enum):
    SEARCH   = "search"   # full pipeline: fetch → classify → extract → synthesize
    FETCH    = "fetch"    # single URL: fetch → classify → extract → return signal
    LEARN    = "learn"    # crawl a domain: add to frontier, begin phase progression
    STATUS   = "status"   # system health: store, daemons, phase counts
    QUIT     = "quit"     # graceful shutdown

class Query:
    type: QueryType
    text: str             # the query text (right of pipe)
    run_id: str           # UUID4

class Result:
    query: Query
    signal: str           # extracted signal text
    sources: List[str]    # URLs that contributed
    topology_classes: List[str]
    confidence: float
    latency_ms: float
    run_id: str
```

**Search pipeline:**
```python
async def handle_search(self, query: Query) -> Result:
    # 1. WLM: predict which domains/URLs to fetch for this query
    traversal = await self.wlm.predict_traversal(query.text)
    
    # 2. Fetch in parallel (max 5 simultaneous)
    raw_results = await asyncio.gather(*[
        self.fetcher.fetch(url) for url in traversal.urls[:5]
    ])
    
    # 3. Classify each result
    classifications = await asyncio.gather(*[
        self.classifier.classify(raw) for raw in raw_results
    ])
    
    # 4. Parse with intent conditioning
    parsed = await asyncio.gather(*[
        self.parser.parse(raw, cls, query.text)
        for raw, cls in zip(raw_results, classifications)
    ])
    
    # 5. Sanitize
    sanitized = await asyncio.gather(*[
        self.sanitizer.sanitize(p) for p in parsed
    ])
    
    # 6. Synthesize signal into answer
    signal = await self._synthesize(sanitized, query.text)
    
    return Result(signal=signal, sources=[r.url for r in raw_results], ...)
```

**`_synthesize()` — this is where the LLM fits:**
The synthesizer takes extracted signal zones (clean, typed, structured) and
runs a final inference pass to produce a coherent answer. This is the ONLY
LLM call in the entire pipeline. Everything before this point is deterministic.

```python
async def _synthesize(self, sanitized: List[SanitizedSignal], query: str) -> str:
    # Concatenate signal zones, deduplicate, rank by density
    ranked_zones = self._rank_zones(sanitized)
    context = self._build_context(ranked_zones, max_tokens=4096)
    
    # Single LLM call with structured context
    # Model: whatever is configured — Haiku for speed, Sonnet for quality
    response = await self.llm.complete(
        system="You are a precise information synthesizer. Use only the provided signal.",
        user=f"Query: {query}\n\nSignal:\n{context}"
    )
    return response
```

---

## SECTION 8: TERMINAL UI — Rust

### What axiom_tui/ does

The terminal UI is the user-facing shell. It renders the AXIOM logo, accepts queries,
displays results, shows system status. It compiles to a single static binary.

The TUI calls interface.py via a Unix domain socket. It does not import Python.
The binary spawns the Python runtime (cold_start.py) as a subprocess on startup,
then communicates with it via socket.

### Crate: axiom_tui/

**Total LOC target:** 3,000–4,000
**Language:** Rust (edition 2021)
**Dependencies:**
```toml
[dependencies]
crossterm = "0.27"      # terminal control
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
```

### File: axiom_tui/src/logo.rs

**Responsibility:** Render the AXIOM logo in the style of neofetch.
ASCII art logo on left, system info on right.

```
Logo must render:

    ██╗   ██╗██╗ ██████╗ ███╗   ███╗
    ██║   ██║██║██╔═══██╗████╗ ████║
    ███████║██║██║   ██║██╔████╔██║
    ██╔══██║██║██║   ██║██║╚██╔╝██║
    ██║  ██║██║╚██████╔╝██║ ╚═╝ ██║
    ╚═╝  ╚═╝╚═╝ ╚═════╝ ╚═╝     ╚═╝

    topology-addressed generation

Displayed in the following color scheme:
- Logo: deep blue (#0066CC) to cyan (#00CCFF) gradient left-to-right
- Tagline: dim white
- Info panel (right side):
    store:    4 files / healthy
    domains:  12,847 tracked
    phases:   COLD 9,231 | LEARNING 2,891 | KNOWN 725
    uptime:   4h 23m
    gpu:      RTX 5080 / 4.2GB VRAM used
```

```rust
pub struct LogoRenderer {
    pub logo_lines: Vec<String>,
    pub info_lines: Vec<InfoLine>,
}

pub struct InfoLine {
    pub key: String,
    pub value: String,
    pub color: Color,
}

impl LogoRenderer {
    pub fn render(&self, stdout: &mut impl Write) -> io::Result<()>;
    pub fn update_info(&mut self, status: &SystemStatus);
}
```

---

### File: axiom_tui/src/repl.rs

**Responsibility:** The REPL loop. Reads input, parses command | query, sends to dispatcher.

```rust
pub struct Repl {
    history: Vec<String>,          // command history, up arrow to scroll
    current_input: String,
    cursor_pos: usize,
    state: ReplState,
}

pub enum ReplState {
    Idle,
    Processing { run_id: String, started_at: Instant },
    Displaying { result: QueryResult },
    Error { message: String },
}

impl Repl {
    // Main render function — renders prompt + input line
    pub fn render(&self, stdout: &mut impl Write) -> io::Result<()>;
    
    // Parse "command | query text" → (QueryType, String)
    pub fn parse_input(input: &str) -> Result<(QueryType, String), ParseError>;
    
    // Handle keypress events
    pub fn handle_event(&mut self, event: KeyEvent) -> ReplAction;
}

pub enum ReplAction {
    Submit(String),      // user pressed enter
    Continue,            // still typing
    Quit,                // ctrl+c or quit command
    HistoryUp,           // up arrow
    HistoryDown,         // down arrow
}
```

**Prompt rendering:**
```
axiom> _
       ^cursor here, blinking block

While processing:
axiom> search | latest RNA folding research
       ⠋ fetching 3 URLs...              (spinner animation)
       ⠙ classifying...
       ⠹ extracting signal...
       ⠸ synthesizing...

Result display:
───────────────────────────────────────────────────────
RESULT  search | latest RNA folding research  [42ms]
───────────────────────────────────────────────────────
[signal text rendered here with syntax highlighting]

Sources: nature.com  arxiv.org  biorxiv.org
Classes: NEWS_ARTICLE  WIKIPEDIA_ARTICLE
───────────────────────────────────────────────────────
axiom> _
```

---

### File: axiom_tui/src/dispatcher.rs

**Responsibility:** Send queries to interface.py via Unix socket, receive results.

```rust
pub struct Dispatcher {
    socket_path: PathBuf,
    stream: Option<UnixStream>,
}

impl Dispatcher {
    pub async fn connect(socket_path: &Path) -> Result<Self, io::Error>;
    
    pub async fn send_query(
        &mut self,
        query_type: QueryType,
        text: &str,
    ) -> Result<QueryResult, DispatchError>;
    
    pub async fn get_status(&mut self) -> Result<SystemStatus, DispatchError>;
    
    pub async fn graceful_shutdown(&mut self) -> Result<(), DispatchError>;
    
    // Reconnect if socket connection dropped
    async fn ensure_connected(&mut self) -> Result<(), io::Error>;
}
```

**Wire format (newline-delimited JSON):**
```json
// Query (TUI → interface.py):
{"run_id": "uuid4", "type": "search", "text": "latest RNA folding research"}

// Result (interface.py → TUI):
{"run_id": "uuid4", "signal": "...", "sources": [...], "latency_ms": 42.3, "error": null}
```

---

### File: axiom_tui/src/ui.rs

**Responsibility:** Full UI layout. Logo + REPL + status bar in one terminal.

```
Terminal layout (full screen):

┌─────────────────────────────────────────────────────────────────────┐
│  [AXIOM LOGO]                    store:    4 files / healthy         │
│                                  domains:  12,847 tracked            │
│  topology-addressed generation   phases:   COLD 9,231 | KNOWN 725   │
│                                  gpu:      RTX 5080 / 4.2GB used     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  [RESULT DISPLAY AREA — scrollable]                                  │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  axiom> _                                                            │
│  [STATUS BAR: ready | domains learned today: 3 | uptime: 4h 23m]   │
└─────────────────────────────────────────────────────────────────────┘
```

```rust
pub struct AxiomUI {
    logo: LogoRenderer,
    repl: Repl,
    result_scroll: usize,
    terminal_size: (u16, u16),
    last_result: Option<QueryResult>,
    status: SystemStatus,
}

impl AxiomUI {
    pub async fn run(dispatcher: Dispatcher) -> Result<(), io::Error>;
    fn render_full(&self, stdout: &mut impl Write) -> io::Result<()>;
    fn render_logo_panel(&self, stdout: &mut impl Write) -> io::Result<()>;
    fn render_result_panel(&self, stdout: &mut impl Write) -> io::Result<()>;
    fn render_repl_panel(&self, stdout: &mut impl Write) -> io::Result<()>;
    fn render_status_bar(&self, stdout: &mut impl Write) -> io::Result<()>;
    fn handle_resize(&mut self, cols: u16, rows: u16);
    fn scroll_result(&mut self, delta: i32);
}
```

---

### File: axiom_tui/src/main.rs

**Responsibility:** Entry point. Spawn Python runtime. Connect socket. Run TUI.

```rust
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Parse CLI args (--store path, --socket path, --dev mode)
    let args = Args::parse();
    
    // 2. Spawn Python cold_start.py as subprocess
    let mut python = Command::new("python3")
        .arg("cold_start.py")
        .arg("--socket").arg(&args.socket_path)
        .arg("--store").arg(&args.store_path)
        .spawn()?;
    
    // 3. Wait for socket to appear (cold_start.py creates it when ready)
    wait_for_socket(&args.socket_path, timeout_secs=120).await?;
    
    // 4. Connect dispatcher
    let dispatcher = Dispatcher::connect(&args.socket_path).await?;
    
    // 5. Run TUI (blocks until quit)
    AxiomUI::run(dispatcher).await?;
    
    // 6. Graceful shutdown — send quit to interface.py
    python.kill()?;
    
    Ok(())
}
```

**The resulting binary:**
```bash
# Build
cargo build --release
# Binary at target/release/axiom

# Run
./axiom --store /store --socket /tmp/axiom.sock

# First thing user sees:
[logo renders]
[status info populates as cold_start completes]
[prompt appears when system is ready]
axiom> _
```

---

## SECTION 9: BUILD SYSTEM

### Makefile (root)

```makefile
.PHONY: all clean test install

all: rust_tui c_daemons cuda_offline go_preparser python_check

# Rust TUI — produces single binary
rust_tui:
	cd axiom_tui && cargo build --release
	cp axiom_tui/target/release/axiom ./axiom

# C daemons + alpine_strip
c_daemons:
	gcc -O3 -march=native -std=c11 \
	    daemons/phase_daemon.c \
	    -o daemons/phase_daemon \
	    -lpcre2-8
	gcc -O3 -march=native -std=c11 \
	    daemons/store_sentinel.c \
	    -o daemons/store_sentinel
	gcc -O3 -march=native -std=c11 \
	    alpine_strip/strip_engine.c \
	    alpine_strip/batch_runner.c \
	    -o alpine_strip/batch_runner \
	    -lpcre2-8 -lpthread -shared -fPIC \
	    -o alpine_strip/libalpinestrip.so

# CUDA offline
cuda_offline:
	nvcc -O3 -arch=sm_89 \
	    offline/gpu_encoder.cu \
	    offline/weight_updater.cu \
	    offline/gradient_accumulator.cu \
	    -lcublas -lcurand \
	    -shared -fPIC \
	    -o offline/libaxiom_gpu.so
	gcc -O3 -std=c11 \
	    offline/batch_scheduler.c \
	    -o offline/batch_scheduler \
	    -L./offline -laxiom_gpu \
	    -lpthread

# Go preparser — produces shared library + binary
go_preparser:
	cd preparser && go build -o ../bin/domain_analyzer ./...
	cd preparser && go build -buildmode=c-shared -o ../preparser/libpreparser.so ./...

# Cold start C library
cold_start_c:
	gcc -O2 -std=c11 \
	    cold_start_c.c \
	    -shared -fPIC \
	    -o cold_start_c.so \
	    -liptables

# Run all tests
test: test_rust test_c test_cuda test_go test_python
	@echo "ALL TESTS PASSED"

test_rust:
	cd axiom_tui && cargo test

test_c:
	# Unity test runner for C files
	./run_c_tests.sh

test_cuda:
	./run_cuda_tests.sh

test_go:
	cd preparser && go test ./... -v

test_python:
	python -m pytest tag/ index_daemon.py interface.py cold_start.py -v

# Install — copy binary to PATH
install: all
	cp axiom /usr/local/bin/axiom
	@echo "Installed. Run: axiom --store /store"
```

---

## SECTION 10: FINAL FILE TREE (COMPLETE SYSTEM)

```
axiom/
├── axiom                           ← single binary (Rust TUI, built by make)
├── Makefile
│
├── axiom_tui/                      ← Rust crate
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs                 ← entry point, spawns Python, runs TUI
│       ├── ui.rs                   ← full terminal layout
│       ├── repl.rs                 ← REPL loop, input parsing
│       ├── logo.rs                 ← neofetch-style logo renderer
│       └── dispatcher.rs           ← Unix socket comms with interface.py
│
├── tag/
│   ├── signal_kernel/              ← DONE
│   ├── crawler_bus.py              ← DONE
│   ├── store_watchdog.py           ← DONE
│   ├── world_model/                ← DONE
│   └── topology/
│       ├── classifier.py           ← DONE
│       ├── parser.py               ← DONE
│       ├── sanitizer.py            ← DONE
│       └── surprise_detector.py    ← DONE (0.15x steady state latency)
│
├── crawler/                        ← DONE
│   ├── bloom_filter.py
│   ├── crawl_cursor.py
│   ├── rate_limiter.py
│   ├── frontier.py
│   └── fetcher.py                  ← 8K LOC, 80/80 tests
│
├── preparser/                      ← Go
│   ├── domain_analyzer.go          ← domain fingerprinting
│   ├── crawl_planner.go            ← crawl plan generation
│   ├── signal_extractor.go         ← zone extraction from sanitized signal
│   ├── recipe_validator.go         ← recipe yield validation
│   └── *_test.go                   ← Go test files per file above
│
├── alpine_strip/                   ← C
│   ├── strip_engine.c              ← core stripping, PCRE2, pool allocator
│   ├── strip_engine.h
│   ├── batch_runner.c              ← parallel batch processor
│   └── test_strip.c                ← Unity tests
│
├── offline/                        ← CUDA + C
│   ├── gpu_encoder.cu              ← zone → (1,256) embeddings
│   ├── weight_updater.cu           ← AdamW on structural_layer.pt
│   ├── gradient_accumulator.cu     ← gradient accumulation across batches
│   ├── batch_scheduler.c           ← coordinates the three above
│   └── test_offline.cu             ← CUDA tests
│
├── daemons/                        ← C
│   ├── phase_daemon.c              ← phase progression monitor
│   └── store_sentinel.c            ← store health monitor
│
├── index_daemon.py                 ← Python, RL loop, ~5K LOC
├── cold_start.py                   ← Python orchestrator
├── cold_start_c.c                  ← C library for gVisor/iptables/Tor
├── interface.py                    ← ONLY public surface, Unix socket server
│
└── store/
    ├── topology_router.pt          ← Mamba SSM weights
    ├── recipe_registry.mmap        ← compiled recipes
    ├── phase_states.mmap           ← per-domain phase state
    └── structural_layer.pt         ← structural intelligence weights
```

---

## SECTION 11: LOC BREAKDOWN

```
FILE / MODULE                       LANGUAGE    TARGET LOC    TESTS
─────────────────────────────────────────────────────────────────────
preparser/domain_analyzer.go        Go          2,000         40
preparser/crawl_planner.go          Go          1,800         35
preparser/signal_extractor.go       Go          2,000         40
preparser/recipe_validator.go       Go          1,200         25
                                                ──────
                                    Go total:   7,000         140

alpine_strip/strip_engine.c         C           2,800         50
alpine_strip/batch_runner.c         C           1,700         30
                                                ──────
                                    C strip:    4,500         80

offline/gpu_encoder.cu              CUDA        2,200         30
offline/weight_updater.cu           CUDA        2,300         25
offline/gradient_accumulator.cu     CUDA        1,700         20
offline/batch_scheduler.c           C           1,800         30
                                                ──────
                                    offline:    8,000         105

daemons/phase_daemon.c              C           2,200         30
daemons/store_sentinel.c            C           1,700         25
                                                ──────
                                    daemons:    3,900         55

index_daemon.py                     Python      5,500         60
cold_start.py                       Python      2,300         30
cold_start_c.c                      C           1,200         20
interface.py                        Python      2,800         40
                                                ──────
                                    core py:    11,800        150

axiom_tui/src/main.rs               Rust        400           10
axiom_tui/src/ui.rs                 Rust        1,200         20
axiom_tui/src/repl.rs               Rust        900           25
axiom_tui/src/logo.rs               Rust        500           10
axiom_tui/src/dispatcher.rs         Rust        700           20
                                                ──────
                                    Rust TUI:   3,700         85

─────────────────────────────────────────────────────────────────────
ALREADY BUILT (DO NOT REBUILD)
signal_kernel/                      Python      25,000        350
crawler_bus.py                      Python      ~3,000        171
fetcher.py                          Python      8,000         80
classifier.py + parser.py           Python      ~12,000       —
sanitizer.py + surprise_detector.py Python      ~14,000       —
world_model/                        Python      ~8,000        —

─────────────────────────────────────────────────────────────────────
NEW CODE TOTAL:                                 ~38,900       615
EXISTING CODE:                                  ~70,000       601+
GRAND TOTAL:                                    ~109,000      1,216+
```

---

## SECTION 12: WHAT CODEX SHOULD DO WITH THIS DOCUMENT

1. **Build in dependency order.** Do not build index_daemon.py before offline/.
   offline/ must exist for index_daemon's gradient dispatch to have a target.
   The correct order:
   ```
   preparser/ → alpine_strip/ → offline/ → daemons/ → index_daemon.py 
   → cold_start.py + cold_start_c.c → interface.py → axiom_tui/
   ```

2. **Each file is independently testable.** Tests run before the next file starts.
   A file is not done until its tests pass.

3. **Never break the four store file invariant.** All writes staging+rename.
   No file writes directly to /store/. No exceptions.

4. **The bus is the only inter-component communication.** Components do not import
   each other. They subscribe to events and emit events. Bus is crawlerbus.py.

5. **interface.py is built absolutely last.** It imports everything else.
   Building it first would be building the roof before the walls.

6. **The axiom binary is the single calling point.** The user installs nothing else.
   `cargo build --release` produces the binary. `./axiom` starts the system.
   Everything else is spawned by the binary.

7. **Test count target: 615 new tests across all new files.**
   This is non-negotiable. Untested code is not done.

8. **50K new LOC is the target. The system is already 70K LOC built.**
   Combined: ~120K LOC tested production system that replaces RAG entirely.

---

## THE ONE THING THAT MATTERS

Every architectural decision in this document exists to support one invariant:

```
axiom> search | how does AlphaFold 3 handle RNA tertiary structure prediction?
```

That query should return a synthesized, accurate answer drawn from the actual
current state of the scientific literature — not from training data, not from
a vector database, not from cached embeddings — from live pages, surgically
extracted, structurally understood, signal not noise.

Faster than any RAG system. More accurate than any RAG system. Because AXIOM
knows what a page IS before it fetches it. The model weights ARE the index.
Reading IS retrieval. The map IS the territory.

Build this. Test it. Ship it.
