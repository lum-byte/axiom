/*
 * batch_runner_part1.c â€” Ring buffer queue and CLI argument parsing.
 *
 * Lock-free single-producer / multi-consumer ring buffer using C11
 * atomics.  CLI parser handles --queue, --mmap, --threads, --pool-mb,
 * --verbose, and --dry-run flags.
 */

#include "batch_runner.h"
#include "tool_strip_accelerator.h"

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#  include <unistd.h>
#endif

/* ------------------------------------------------------------------ */
/*  Ring buffer implementation                                         */
/* ------------------------------------------------------------------ */

void ring_init(batch_ring_buffer *rb) {
    if (rb == NULL) return;
    memset(rb->items, 0, sizeof(rb->items));
    atomic_store(&rb->head, 0);
    atomic_store(&rb->tail, 0);
    atomic_store(&rb->done, false);
}

bool ring_push(batch_ring_buffer *rb, const batch_work_item *item) {
    if (rb == NULL || item == NULL) return false;

    uint_fast64_t head = atomic_load_explicit(&rb->head, memory_order_relaxed);
    uint_fast64_t tail = atomic_load_explicit(&rb->tail, memory_order_acquire);

    if (head - tail >= BATCH_QUEUE_CAPACITY) {
        return false;
    }
    uint_fast64_t slot = head & (BATCH_QUEUE_CAPACITY - 1u);
    memcpy(&rb->items[slot], item, sizeof(batch_work_item));
    atomic_store_explicit(&rb->head, head + 1, memory_order_release);
    return true;
}

bool ring_pop(batch_ring_buffer *rb, batch_work_item *item) {
    if (rb == NULL || item == NULL) return false;

    uint_fast64_t tail, head;
    for (;;) {
        tail = atomic_load_explicit(&rb->tail, memory_order_relaxed);
        head = atomic_load_explicit(&rb->head, memory_order_acquire);
        if (tail >= head) {
            return false;
        }
        if (atomic_compare_exchange_weak_explicit(
                &rb->tail, &tail, tail + 1,
                memory_order_acq_rel, memory_order_relaxed)) {
            break;
        }
    }
    uint_fast64_t slot = tail & (BATCH_QUEUE_CAPACITY - 1u);
    memcpy(item, &rb->items[slot], sizeof(batch_work_item));
    return true;
}

bool ring_is_empty(const batch_ring_buffer *rb) {
    if (rb == NULL) return true;
    uint_fast64_t head = atomic_load_explicit(
        (atomic_uint_fast64_t *)&rb->head, memory_order_acquire);
    uint_fast64_t tail = atomic_load_explicit(
        (atomic_uint_fast64_t *)&rb->tail, memory_order_acquire);
    return head == tail;
}

size_t ring_size(const batch_ring_buffer *rb) {
    if (rb == NULL) return 0;
    uint_fast64_t head = atomic_load_explicit(
        (atomic_uint_fast64_t *)&rb->head, memory_order_acquire);
    uint_fast64_t tail = atomic_load_explicit(
        (atomic_uint_fast64_t *)&rb->tail, memory_order_acquire);
    return (size_t)(head - tail);
}

/* ------------------------------------------------------------------ */
/*  Ring buffer push with backpressure (spin-wait)                     */
/* ------------------------------------------------------------------ */

bool ring_push_blocking(batch_ring_buffer *rb, const batch_work_item *item) {
    if (rb == NULL || item == NULL) return false;
    int spins = 0;
    while (!ring_push(rb, item)) {
        if (batch_should_shutdown()) return false;
        spins++;
        if (spins > 1000) {
#ifdef _WIN32
            Sleep(1);
#else
            struct timespec ts = {0, 1000000};
            nanosleep(&ts, NULL);
#endif
            spins = 0;
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  Ring buffer pop with wait (spin then sleep)                        */
/* ------------------------------------------------------------------ */

bool ring_pop_blocking(batch_ring_buffer *rb, batch_work_item *item) {
    if (rb == NULL || item == NULL) return false;
    int spins = 0;
    while (!ring_pop(rb, item)) {
        if (atomic_load(&rb->done) && ring_is_empty(rb)) {
            return false;
        }
        if (batch_should_shutdown()) return false;
        spins++;
        if (spins > 1000) {
#ifdef _WIN32
            Sleep(1);
#else
            struct timespec ts = {0, 1000000};
            nanosleep(&ts, NULL);
#endif
            spins = 0;
        }
    }
    return true;
}

/* ------------------------------------------------------------------ */
/*  CLI argument parsing                                               */
/* ------------------------------------------------------------------ */

static void print_usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [OPTIONS]\n"
        "\n"
        "Options:\n"
        "  --queue PATH       Path to JSONL work queue file (required)\n"
        "  --mmap PATH        Path to recipe_registry.mmap (required)\n"
        "  --threads N        Worker thread count (default: nproc)\n"
        "  --pool-mb N        Pool allocator size in MB (default: 64)\n"
        "  --verbose          Enable verbose stderr output\n"
        "  --dry-run          Parse queue but don't process\n"
        "  --help             Show this help message\n"
        "\n"
        "Exit codes:\n"
        "  0  All items processed successfully\n"
        "  1  Partial failure (some items failed)\n"
        "  2  Fatal error (queue unreadable, mmap unavailable)\n",
        prog ? prog : "batch_runner"
    );
}

static int detect_nproc(void) {
#ifdef _WIN32
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    int n = (int)si.dwNumberOfProcessors;
    return n > 0 ? n : 4;
#else
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    return n > 0 ? (int)n : 4;
#endif
}

int batch_parse_args(int argc, char **argv, batch_config *cfg) {
    if (cfg == NULL) return 2;
    memset(cfg, 0, sizeof(*cfg));
    cfg->thread_count = BATCH_DEFAULT_THREADS;
    cfg->pool_mb = BATCH_DEFAULT_POOL_MB;
    cfg->verbose = false;
    cfg->dry_run = false;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return -1;
        }
        if (strcmp(argv[i], "--queue") == 0 && i + 1 < argc) {
            snprintf(cfg->queue_path, BATCH_MAX_PATH, "%s", argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "--mmap") == 0 && i + 1 < argc) {
            snprintf(cfg->mmap_path, BATCH_MAX_PATH, "%s", argv[++i]);
            continue;
        }
        if (strcmp(argv[i], "--threads") == 0 && i + 1 < argc) {
            cfg->thread_count = atoi(argv[++i]);
            if (cfg->thread_count < 1) cfg->thread_count = 1;
            if (cfg->thread_count > 256) cfg->thread_count = 256;
            continue;
        }
        if (strcmp(argv[i], "--pool-mb") == 0 && i + 1 < argc) {
            cfg->pool_mb = atoi(argv[++i]);
            if (cfg->pool_mb < 1) cfg->pool_mb = 1;
            if (cfg->pool_mb > 4096) cfg->pool_mb = 4096;
            continue;
        }
        if (strcmp(argv[i], "--verbose") == 0) {
            cfg->verbose = true;
            continue;
        }
        if (strcmp(argv[i], "--dry-run") == 0) {
            cfg->dry_run = true;
            continue;
        }
        fprintf(stderr, "batch_runner: unknown option: %s\n", argv[i]);
        print_usage(argv[0]);
        return 2;
    }

    if (cfg->queue_path[0] == '\0') {
        fprintf(stderr, "batch_runner: --queue is required\n");
        print_usage(argv[0]);
        return 2;
    }
    if (cfg->mmap_path[0] == '\0') {
        fprintf(stderr, "batch_runner: --mmap is required\n");
        print_usage(argv[0]);
        return 2;
    }
    if (cfg->thread_count == 0) {
        cfg->thread_count = detect_nproc();
    }
    return 0;
}

/*
 * batch_runner_part2.c â€” JSONL work queue parsing.
 *
 * Parses newline-delimited JSON objects from the work queue file.
 * Each line: {"url":"...", "slot_idx":N, "input_path":"...", "output_path":"..."}
 * Uses a minimal hand-rolled JSON parser (no external dependency).
 */


#include <errno.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Minimal JSON string extraction                                     */
/* ------------------------------------------------------------------ */

static const char *skip_ws(const char *p, const char *end) {
    while (p < end && (*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')) {
        p++;
    }
    return p;
}

static const char *find_key(const char *json, size_t len, const char *key,
                            const char **val_start, size_t *val_len) {
    size_t klen = strlen(key);
    const char *end = json + len;
    const char *p = json;
    while (p + klen + 3 < end) {
        if (*p == '"') {
            if ((size_t)(end - p - 1) >= klen &&
                memcmp(p + 1, key, klen) == 0 &&
                p[klen + 1] == '"') {
                p += klen + 2;
                p = skip_ws(p, end);
                if (p < end && *p == ':') {
                    p++;
                    p = skip_ws(p, end);
                    *val_start = p;
                    *val_len = (size_t)(end - p);
                    return p;
                }
            }
        }
        p++;
    }
    *val_start = NULL;
    *val_len = 0;
    return NULL;
}

static int extract_json_string(const char *p, size_t max_len,
                               char *out, size_t out_capacity) {
    if (p == NULL || *p != '"') return -1;
    p++;
    size_t oi = 0;
    size_t remaining = max_len > 0 ? max_len - 1 : 0;
    for (size_t i = 0; i < remaining; ++i) {
        if (p[i] == '"') {
            out[oi] = '\0';
            return (int)(oi);
        }
        if (p[i] == '\\' && i + 1 < remaining) {
            i++;
            switch (p[i]) {
            case '"':  out[oi++] = '"';  break;
            case '\\': out[oi++] = '\\'; break;
            case '/':  out[oi++] = '/';  break;
            case 'n':  out[oi++] = '\n'; break;
            case 'r':  out[oi++] = '\r'; break;
            case 't':  out[oi++] = '\t'; break;
            default:   out[oi++] = p[i]; break;
            }
            if (oi >= out_capacity - 1) break;
            continue;
        }
        if (oi >= out_capacity - 1) break;
        out[oi++] = p[i];
    }
    out[oi] = '\0';
    return -1;
}

static int extract_json_int(const char *p, size_t max_len, int *out) {
    if (p == NULL) return -1;
    const char *end = p + max_len;
    p = skip_ws(p, end);
    if (p >= end) return -1;
    bool negative = false;
    if (*p == '-') {
        negative = true;
        p++;
    }
    if (p >= end || *p < '0' || *p > '9') return -1;
    int val = 0;
    while (p < end && *p >= '0' && *p <= '9') {
        val = val * 10 + (*p - '0');
        p++;
    }
    *out = negative ? -val : val;
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Parse a single JSONL line into a work item                         */
/* ------------------------------------------------------------------ */

int batch_parse_line(const char *line, size_t len, batch_work_item *item) {
    if (line == NULL || item == NULL || len == 0) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(item, 0, sizeof(*item));

    while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) {
        len--;
    }
    if (len == 0) return AXIOM_STRIP_ERR_INVALID_ARG;

    const char *vp;
    size_t vl;

    if (find_key(line, len, "url", &vp, &vl)) {
        extract_json_string(vp, vl, item->url, BATCH_MAX_URL);
    }
    if (find_key(line, len, "slot_idx", &vp, &vl)) {
        extract_json_int(vp, vl, &item->slot_idx);
    }
    if (find_key(line, len, "input_path", &vp, &vl)) {
        extract_json_string(vp, vl, item->input_path, BATCH_MAX_PATH);
    }
    if (find_key(line, len, "output_path", &vp, &vl)) {
        extract_json_string(vp, vl, item->output_path, BATCH_MAX_PATH);
    }

    if (item->input_path[0] == '\0' || item->output_path[0] == '\0') {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Load entire JSONL queue file into ring buffer                      */
/* ------------------------------------------------------------------ */

int batch_load_queue(const char *path, batch_ring_buffer *rb,
                     size_t *items_loaded) {
    if (path == NULL || rb == NULL) {
        return 2;
    }
    FILE *fp = fopen(path, "r");
    if (fp == NULL) {
        fprintf(stderr, "batch_runner: cannot open queue: %s: %s\n",
                path, strerror(errno));
        return 2;
    }

    char *line_buf = (char *)malloc(BATCH_LINE_MAX);
    if (line_buf == NULL) {
        fclose(fp);
        fprintf(stderr, "batch_runner: out of memory for line buffer\n");
        return 2;
    }

    size_t loaded = 0;
    size_t line_num = 0;
    size_t parse_errors = 0;
    size_t queue_full_waits = 0;

    while (fgets(line_buf, (int)BATCH_LINE_MAX, fp) != NULL) {
        line_num++;
        size_t ll = strlen(line_buf);
        while (ll > 0 && (line_buf[ll - 1] == '\n' || line_buf[ll - 1] == '\r')) {
            line_buf[--ll] = '\0';
        }
        if (ll == 0) continue;
        if (line_buf[0] == '#') continue;

        batch_work_item item;
        int rc = batch_parse_line(line_buf, ll, &item);
        if (rc != AXIOM_STRIP_OK) {
            parse_errors++;
            fprintf(stderr, "batch_runner: parse error on line %zu: %s\n",
                    line_num, axiom_strip_strerror(rc));
            continue;
        }

        int push_attempts = 0;
        while (!ring_push(rb, &item)) {
            if (batch_should_shutdown()) {
                free(line_buf);
                fclose(fp);
                if (items_loaded) *items_loaded = loaded;
                return 1;
            }
            push_attempts++;
            queue_full_waits++;
            if (push_attempts > 10000) {
                fprintf(stderr, "batch_runner: queue full, dropping line %zu\n",
                        line_num);
                break;
            }
#ifdef _WIN32
            Sleep(1);
#else
            {
                struct timespec ts = {0, 100000};
                nanosleep(&ts, NULL);
            }
#endif
        }
        loaded++;
    }

    free(line_buf);
    fclose(fp);
    if (items_loaded) *items_loaded = loaded;

    if (parse_errors > 0) {
        fprintf(stderr, "batch_runner: %zu parse errors in queue\n", parse_errors);
    }
    if (queue_full_waits > 0) {
        fprintf(stderr, "batch_runner: %zu queue-full waits\n", queue_full_waits);
    }
    return loaded > 0 ? 0 : (parse_errors > 0 ? 2 : 0);
}

/* ------------------------------------------------------------------ */
/*  Format work item as JSON (for logging / debugging)                 */
/* ------------------------------------------------------------------ */

int batch_work_item_json(const batch_work_item *item,
                         char *buf, size_t buf_capacity) {
    if (item == NULL || buf == NULL || buf_capacity < 64) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int n = snprintf(buf, buf_capacity,
        "{\"url\":\"%s\",\"slot_idx\":%d,"
        "\"input_path\":\"%s\",\"output_path\":\"%s\"}",
        item->url, item->slot_idx,
        item->input_path, item->output_path
    );
    if (n < 0 || (size_t)n >= buf_capacity) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    return AXIOM_STRIP_OK;
}

/*
 * batch_runner_part3.c â€” pthreads worker pool.
 *
 * N worker threads pop items from the ring buffer, load input files,
 * call strip_apply_with_pool (each thread has its own pool), and
 * write output files with atomic rename.
 */


#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#  include <pthread.h>
#  include <unistd.h>
#else
#  include <windows.h>
#  include <process.h>
#endif

/* ------------------------------------------------------------------ */
/*  Thread handles (static, max 256 threads)                           */
/* ------------------------------------------------------------------ */

#define MAX_WORKER_THREADS 256

#ifndef _WIN32
static pthread_t g_threads[MAX_WORKER_THREADS];
#else
static HANDLE    g_threads[MAX_WORKER_THREADS];
#endif
static int       g_thread_count = 0;

/* ------------------------------------------------------------------ */
/*  Worker thread function                                             */
/* ------------------------------------------------------------------ */

static int process_one_item(batch_worker_ctx *ctx, const batch_work_item *item,
                            const axiom_strip_plan *plan, strip_pool *pool) {
    uint8_t *file_data = NULL;
    size_t file_len = 0;
    int rc = batch_read_file(item->input_path, &file_data, &file_len);
    if (rc != 0) {
        fprintf(stderr, "[worker %d] read failed: %s\n",
                ctx->thread_id, item->input_path);
        atomic_fetch_add(&ctx->items_failed, 1);
        return 1;
    }
    atomic_fetch_add(&ctx->bytes_in, file_len);

    size_t out_capacity = file_len + 4096;
    uint8_t *out_buf = (uint8_t *)malloc(out_capacity);
    if (out_buf == NULL) {
        free(file_data);
        fprintf(stderr, "[worker %d] malloc failed for output buffer\n",
                ctx->thread_id);
        atomic_fetch_add(&ctx->items_failed, 1);
        return 1;
    }

    axiom_strip_result result;
    memset(&result, 0, sizeof(result));

    axiom_tool_strip_stats stats;
    rc = axiom_strip_plan_apply(
        plan, file_data, file_len, out_buf, out_capacity, &result, &stats, pool
    );

    free(file_data);

    if (rc != AXIOM_STRIP_OK) {
        fprintf(stderr, "[worker %d] strip failed (%s): %s\n",
                ctx->thread_id, axiom_strip_strerror(rc), item->input_path);
        free(out_buf);
        atomic_fetch_add(&ctx->items_failed, 1);
        return 1;
    }

    rc = batch_write_file_atomic(item->output_path, out_buf, result.bytes_written);
    free(out_buf);
    if (rc != 0) {
        fprintf(stderr, "[worker %d] write failed: %s\n",
                ctx->thread_id, item->output_path);
        atomic_fetch_add(&ctx->items_failed, 1);
        return 1;
    }

    atomic_fetch_add(&ctx->bytes_out, result.bytes_written);
    atomic_fetch_add(&ctx->items_done, 1);
    return 0;
}

int batch_worker_run(batch_worker_ctx *ctx) {
    if (ctx == NULL || ctx->queue == NULL) return 1;

    strip_pool pool;
    size_t pool_bytes = (size_t)ctx->pool_mb * 1024u * 1024u;
    int rc = strip_pool_init(&pool, pool_bytes);
    if (rc != AXIOM_STRIP_OK) {
        fprintf(stderr, "[worker %d] pool init failed\n", ctx->thread_id);
        return 1;
    }

    axiom_tool_snapshot_profile profile;
    axiom_tool_profile_init(&profile);
    snprintf(profile.source_tool, sizeof(profile.source_tool), "%s", "batch_runner");
    axiom_strip_step default_steps[AXIOM_TOOL_PLAN_MAX_STEPS];
    axiom_strip_recipe recipe;
    rc = axiom_tool_profile_build_recipe(
        &profile, default_steps, AXIOM_TOOL_PLAN_MAX_STEPS, &recipe
    );
    if (rc != AXIOM_STRIP_OK) {
        fprintf(stderr, "[worker %d] default recipe build failed\n", ctx->thread_id);
        strip_pool_destroy(&pool);
        return 1;
    }
    axiom_strip_plan *plan = NULL;
    rc = axiom_strip_plan_compile(&recipe, &profile, &plan);
    if (rc != AXIOM_STRIP_OK || plan == NULL) {
        fprintf(stderr, "[worker %d] compiled plan failed: %s\n",
                ctx->thread_id, axiom_strip_strerror(rc));
        strip_pool_destroy(&pool);
        return 1;
    }

    batch_work_item item;
    while (!atomic_load(&ctx->stop)) {
        if (!ring_pop(ctx->queue, &item)) {
            if (atomic_load(&ctx->queue->done) && ring_is_empty(ctx->queue)) {
                break;
            }
            if (batch_should_shutdown()) break;
#ifdef _WIN32
            Sleep(1);
#else
            {
                struct timespec ts = {0, 500000};
                nanosleep(&ts, NULL);
            }
#endif
            continue;
        }
        strip_pool_reset(&pool);
        process_one_item(ctx, &item, plan, &pool);
    }

    axiom_strip_plan_free(plan);
    strip_pool_destroy(&pool);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Thread entry point                                                 */
/* ------------------------------------------------------------------ */

#ifndef _WIN32
static void *worker_thread_entry(void *arg) {
    batch_worker_ctx *ctx = (batch_worker_ctx *)arg;
    batch_worker_run(ctx);
    return NULL;
}
#else
static unsigned __stdcall worker_thread_entry(void *arg) {
    batch_worker_ctx *ctx = (batch_worker_ctx *)arg;
    batch_worker_run(ctx);
    return 0;
}
#endif

/* ------------------------------------------------------------------ */
/*  Start worker threads                                               */
/* ------------------------------------------------------------------ */

int batch_start_workers(batch_worker_ctx *workers, int count,
                        batch_ring_buffer *queue, const batch_config *cfg) {
    if (workers == NULL || queue == NULL || cfg == NULL) return 2;
    if (count < 1) count = 1;
    if (count > MAX_WORKER_THREADS) count = MAX_WORKER_THREADS;

    g_thread_count = count;

    for (int i = 0; i < count; ++i) {
        workers[i].thread_id = i;
        workers[i].queue = queue;
        workers[i].mmap_path = cfg->mmap_path;
        workers[i].pool_mb = cfg->pool_mb;
        atomic_store(&workers[i].items_done, 0);
        atomic_store(&workers[i].items_failed, 0);
        atomic_store(&workers[i].bytes_in, 0);
        atomic_store(&workers[i].bytes_out, 0);
        atomic_store(&workers[i].stop, false);
    }

    for (int i = 0; i < count; ++i) {
#ifndef _WIN32
        int rc = pthread_create(&g_threads[i], NULL,
                                worker_thread_entry, &workers[i]);
        if (rc != 0) {
            fprintf(stderr, "batch_runner: pthread_create failed for worker %d: %s\n",
                    i, strerror(rc));
            g_thread_count = i;
            return 2;
        }
#else
        g_threads[i] = (HANDLE)_beginthreadex(
            NULL, 0, worker_thread_entry, &workers[i], 0, NULL
        );
        if (g_threads[i] == 0) {
            fprintf(stderr, "batch_runner: _beginthreadex failed for worker %d\n", i);
            g_thread_count = i;
            return 2;
        }
#endif
    }
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Stop workers (signal)                                              */
/* ------------------------------------------------------------------ */

void batch_stop_workers(batch_worker_ctx *workers, int count) {
    if (workers == NULL) return;
    for (int i = 0; i < count; ++i) {
        atomic_store(&workers[i].stop, true);
    }
}

/* ------------------------------------------------------------------ */
/*  Join worker threads                                                */
/* ------------------------------------------------------------------ */

void batch_join_workers(int count) {
    if (count > g_thread_count) count = g_thread_count;
    for (int i = 0; i < count; ++i) {
#ifndef _WIN32
        pthread_join(g_threads[i], NULL);
#else
        WaitForSingleObject(g_threads[i], INFINITE);
        CloseHandle(g_threads[i]);
#endif
    }
    g_thread_count = 0;
}

/* ------------------------------------------------------------------ */
/*  Aggregate worker stats                                             */
/* ------------------------------------------------------------------ */

void batch_aggregate_stats(const batch_worker_ctx *workers, int count,
                           uint64_t *total_done, uint64_t *total_failed,
                           uint64_t *total_bytes_in, uint64_t *total_bytes_out) {
    uint64_t done = 0, failed = 0, bin = 0, bout = 0;
    for (int i = 0; i < count; ++i) {
        done += atomic_load(&workers[i].items_done);
        failed += atomic_load(&workers[i].items_failed);
        bin += atomic_load(&workers[i].bytes_in);
        bout += atomic_load(&workers[i].bytes_out);
    }
    if (total_done) *total_done = done;
    if (total_failed) *total_failed = failed;
    if (total_bytes_in) *total_bytes_in = bin;
    if (total_bytes_out) *total_bytes_out = bout;
}

/*
 * batch_runner_part4.c â€” Stats thread and signal handling.
 *
 * Stats thread reports throughput every BATCH_STATS_INTERVAL seconds
 * to stderr.  Signal handler catches SIGTERM/SIGINT for graceful
 * shutdown: finish current items, drain queue, exit cleanly.
 */


#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef _WIN32
#  include <pthread.h>
#  include <unistd.h>
#else
#  include <windows.h>
#  include <process.h>
#endif

/* ------------------------------------------------------------------ */
/*  Signal handling                                                    */
/* ------------------------------------------------------------------ */

static volatile sig_atomic_t g_shutdown_requested = 0;

#ifndef _WIN32
static void signal_handler(int sig) {
    (void)sig;
    g_shutdown_requested = 1;
}
#else
static BOOL WINAPI console_handler(DWORD ctrl_type) {
    if (ctrl_type == CTRL_C_EVENT || ctrl_type == CTRL_BREAK_EVENT ||
        ctrl_type == CTRL_CLOSE_EVENT) {
        g_shutdown_requested = 1;
        return TRUE;
    }
    return FALSE;
}
#endif

void batch_install_signal_handlers(void) {
#ifndef _WIN32
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = signal_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    sigaction(SIGTERM, &sa, NULL);
    sigaction(SIGINT, &sa, NULL);
#else
    SetConsoleCtrlHandler(console_handler, TRUE);
#endif
}

bool batch_should_shutdown(void) {
    return g_shutdown_requested != 0;
}

/* ------------------------------------------------------------------ */
/*  Time helpers                                                       */
/* ------------------------------------------------------------------ */

static double get_time_seconds(void) {
#ifndef _WIN32
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
#else
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
#endif
}

/* ------------------------------------------------------------------ */
/*  Stats thread                                                       */
/* ------------------------------------------------------------------ */

static void print_stats(const batch_worker_ctx *workers, int count,
                        double elapsed) {
    uint64_t done = 0, failed = 0, bytes_in = 0, bytes_out = 0;
    for (int i = 0; i < count; ++i) {
        done += atomic_load(&workers[i].items_done);
        failed += atomic_load(&workers[i].items_failed);
        bytes_in += atomic_load(&workers[i].bytes_in);
        bytes_out += atomic_load(&workers[i].bytes_out);
    }
    double mb_in = (double)bytes_in / (1024.0 * 1024.0);
    double mb_out = (double)bytes_out / (1024.0 * 1024.0);
    double items_per_sec = elapsed > 0.0 ? (double)done / elapsed : 0.0;
    double mb_per_sec = elapsed > 0.0 ? mb_in / elapsed : 0.0;
    double ratio = bytes_in > 0 ? (double)bytes_out / (double)bytes_in : 0.0;

    fprintf(stderr,
        "[stats] %.1fs | items: %llu done, %llu failed | "
        "%.1f MB in, %.1f MB out (ratio %.3f) | "
        "%.1f items/s, %.1f MB/s\n",
        elapsed,
        (unsigned long long)done, (unsigned long long)failed,
        mb_in, mb_out, ratio,
        items_per_sec, mb_per_sec
    );
}

#ifndef _WIN32
static void *stats_thread_entry(void *arg) {
    batch_stats_ctx *ctx = (batch_stats_ctx *)arg;
    double start = get_time_seconds();
    while (!atomic_load(&ctx->stop)) {
        for (int i = 0; i < BATCH_STATS_INTERVAL * 10; ++i) {
            if (atomic_load(&ctx->stop)) goto done;
            struct timespec ts = {0, 100000000};
            nanosleep(&ts, NULL);
        }
        double elapsed = get_time_seconds() - start;
        print_stats(ctx->workers, ctx->worker_count, elapsed);
    }
done:
    {
        double elapsed = get_time_seconds() - start;
        print_stats(ctx->workers, ctx->worker_count, elapsed);
    }
    return NULL;
}

static pthread_t g_stats_thread;
static int       g_stats_thread_valid = 0;
#else
static HANDLE    g_stats_thread = NULL;

static unsigned __stdcall stats_thread_entry(void *arg) {
    batch_stats_ctx *ctx = (batch_stats_ctx *)arg;
    double start = get_time_seconds();
    while (!atomic_load(&ctx->stop)) {
        for (int i = 0; i < BATCH_STATS_INTERVAL * 10; ++i) {
            if (atomic_load(&ctx->stop)) goto done;
            Sleep(100);
        }
        double elapsed = get_time_seconds() - start;
        print_stats(ctx->workers, ctx->worker_count, elapsed);
    }
done:
    {
        double elapsed = get_time_seconds() - start;
        print_stats(ctx->workers, ctx->worker_count, elapsed);
    }
    return 0;
}
#endif

int batch_stats_start(batch_stats_ctx *ctx, batch_worker_ctx *workers,
                      int worker_count) {
    if (ctx == NULL || workers == NULL) return 1;
    ctx->workers = workers;
    ctx->worker_count = worker_count;
    atomic_store(&ctx->stop, false);

#ifndef _WIN32
    int rc = pthread_create(&g_stats_thread, NULL, stats_thread_entry, ctx);
    if (rc != 0) {
        fprintf(stderr, "batch_runner: stats thread create failed: %s\n",
                strerror(rc));
        return 1;
    }
    g_stats_thread_valid = 1;
#else
    g_stats_thread = (HANDLE)_beginthreadex(
        NULL, 0, stats_thread_entry, ctx, 0, NULL
    );
    if (g_stats_thread == 0) {
        fprintf(stderr, "batch_runner: stats thread create failed\n");
        return 1;
    }
#endif
    return 0;
}

void batch_stats_stop(batch_stats_ctx *ctx) {
    if (ctx == NULL) return;
    atomic_store(&ctx->stop, true);
#ifndef _WIN32
    if (g_stats_thread_valid) {
        pthread_join(g_stats_thread, NULL);
        g_stats_thread_valid = 0;
    }
#else
    if (g_stats_thread != NULL) {
        WaitForSingleObject(g_stats_thread, 5000);
        CloseHandle(g_stats_thread);
        g_stats_thread = NULL;
    }
#endif
}

/* ------------------------------------------------------------------ */
/*  Elapsed time tracker for the main function                         */
/* ------------------------------------------------------------------ */

typedef struct {
    double start_time;
} batch_timer;

void batch_timer_start(batch_timer *t) {
    if (t != NULL) {
        t->start_time = get_time_seconds();
    }
}

double batch_timer_elapsed(const batch_timer *t) {
    if (t == NULL) return 0.0;
    return get_time_seconds() - t->start_time;
}

/*
 * batch_runner_part5.c â€” File I/O, atomic rename, and main().
 *
 * Reads input files into malloc'd buffers, writes output files via
 * staging temp file + fsync + atomic rename.  main() wires together
 * CLI parsing, queue loading, worker pool, stats thread, and exit
 * code semantics (0 = all OK, 1 = partial, 2 = fatal).
 */


#include <errno.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <unistd.h>
#else
#  include <windows.h>
#  include <io.h>
#  include <fcntl.h>
#  include <sys/stat.h>
#endif

/* ------------------------------------------------------------------ */
/*  Read an entire file into a malloc'd buffer                         */
/* ------------------------------------------------------------------ */

int batch_read_file(const char *path, uint8_t **data, size_t *len) {
    if (path == NULL || data == NULL || len == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    *data = NULL;
    *len = 0;

    FILE *fp = fopen(path, "rb");
    if (fp == NULL) {
        return AXIOM_STRIP_ERR_MMAP_OPEN;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        fclose(fp);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    long sz = ftell(fp);
    if (sz < 0) {
        fclose(fp);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    if (sz == 0) {
        fclose(fp);
        *data = (uint8_t *)malloc(1);
        if (*data) (*data)[0] = 0;
        *len = 0;
        return AXIOM_STRIP_OK;
    }
    rewind(fp);

    uint8_t *buf = (uint8_t *)malloc((size_t)sz);
    if (buf == NULL) {
        fclose(fp);
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    size_t read_bytes = fread(buf, 1, (size_t)sz, fp);
    fclose(fp);

    if (read_bytes != (size_t)sz) {
        free(buf);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    *data = buf;
    *len = (size_t)sz;
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Write data to a file with atomic rename + fsync                    */
/* ------------------------------------------------------------------ */

static int make_temp_path(const char *path, char *tmp_path, size_t capacity) {
    int n = snprintf(tmp_path, capacity, "%s.tmp.%d",
                     path, (int)
#ifndef _WIN32
                     getpid()
#else
                     GetCurrentProcessId()
#endif
    );
    if (n < 0 || (size_t)n >= capacity) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    return AXIOM_STRIP_OK;
}

static int ensure_parent_dir(const char *path) {
    char dir[BATCH_MAX_PATH];
    snprintf(dir, sizeof(dir), "%s", path);
    char *last_sep = NULL;
    for (char *p = dir; *p; ++p) {
        if (*p == '/' || *p == '\\') last_sep = p;
    }
    if (last_sep == NULL || last_sep == dir) return 0;
    *last_sep = '\0';

#ifndef _WIN32
    struct stat st;
    if (stat(dir, &st) == 0) return 0;
    return mkdir(dir, 0755);
#else
    DWORD attrs = GetFileAttributesA(dir);
    if (attrs != INVALID_FILE_ATTRIBUTES) return 0;
    return CreateDirectoryA(dir, NULL) ? 0 : -1;
#endif
}

int batch_write_file_atomic(const char *path, const uint8_t *data,
                            size_t len) {
    if (path == NULL || (data == NULL && len > 0)) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }

    ensure_parent_dir(path);

    char tmp_path[BATCH_MAX_PATH];
    if (make_temp_path(path, tmp_path, sizeof(tmp_path)) != AXIOM_STRIP_OK) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }

#ifndef _WIN32
    int fd = open(tmp_path, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        return AXIOM_STRIP_ERR_MMAP_OPEN;
    }
    if (len > 0) {
        ssize_t written = write(fd, data, len);
        if (written < 0 || (size_t)written != len) {
            close(fd);
            unlink(tmp_path);
            return AXIOM_STRIP_ERR_MMAP_READ;
        }
    }
    if (fsync(fd) != 0) {
        close(fd);
        unlink(tmp_path);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    close(fd);
    if (rename(tmp_path, path) != 0) {
        unlink(tmp_path);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
#else
    HANDLE fh = CreateFileA(tmp_path, GENERIC_WRITE, 0, NULL,
                            CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        return AXIOM_STRIP_ERR_MMAP_OPEN;
    }
    if (len > 0) {
        DWORD written = 0;
        if (!WriteFile(fh, data, (DWORD)len, &written, NULL) ||
            written != (DWORD)len) {
            CloseHandle(fh);
            DeleteFileA(tmp_path);
            return AXIOM_STRIP_ERR_MMAP_READ;
        }
    }
    FlushFileBuffers(fh);
    CloseHandle(fh);
    if (!MoveFileExA(tmp_path, path, MOVEFILE_REPLACE_EXISTING)) {
        DeleteFileA(tmp_path);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
#endif
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  File existence check                                               */
/* ------------------------------------------------------------------ */

static bool file_exists(const char *path) {
#ifndef _WIN32
    struct stat st;
    return stat(path, &st) == 0;
#else
    DWORD attrs = GetFileAttributesA(path);
    return attrs != INVALID_FILE_ATTRIBUTES;
#endif
}

/* ------------------------------------------------------------------ */
/*  main()                                                             */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv) {
    batch_config cfg;
    int rc = batch_parse_args(argc, argv, &cfg);
    if (rc == -1) return 0;
    if (rc != 0) return rc;

    if (!file_exists(cfg.queue_path)) {
        fprintf(stderr, "batch_runner: queue file not found: %s\n",
                cfg.queue_path);
        return 2;
    }
    if (!file_exists(cfg.mmap_path)) {
        fprintf(stderr, "batch_runner: mmap file not found: %s\n",
                cfg.mmap_path);
        return 2;
    }

    batch_install_signal_handlers();

    fprintf(stderr, "batch_runner: threads=%d pool=%dMB queue=%s mmap=%s\n",
            cfg.thread_count, cfg.pool_mb, cfg.queue_path, cfg.mmap_path);

    static batch_ring_buffer queue;
    ring_init(&queue);

    size_t items_loaded = 0;
    rc = batch_load_queue(cfg.queue_path, &queue, &items_loaded);
    if (rc == 2) {
        fprintf(stderr, "batch_runner: fatal error loading queue\n");
        return 2;
    }
    fprintf(stderr, "batch_runner: loaded %zu items into queue\n", items_loaded);

    if (items_loaded == 0) {
        fprintf(stderr, "batch_runner: empty queue, nothing to do\n");
        return 0;
    }

    atomic_store(&queue.done, true);

    if (cfg.dry_run) {
        fprintf(stderr, "batch_runner: dry run, exiting\n");
        return 0;
    }

    batch_worker_ctx *workers = (batch_worker_ctx *)calloc(
        (size_t)cfg.thread_count, sizeof(batch_worker_ctx)
    );
    if (workers == NULL) {
        fprintf(stderr, "batch_runner: out of memory for worker contexts\n");
        return 2;
    }

    batch_stats_ctx stats_ctx;
    batch_stats_start(&stats_ctx, workers, cfg.thread_count);

    rc = batch_start_workers(workers, cfg.thread_count, &queue, &cfg);
    if (rc != 0) {
        batch_stats_stop(&stats_ctx);
        free(workers);
        return 2;
    }

    batch_join_workers(cfg.thread_count);
    batch_stats_stop(&stats_ctx);

    uint64_t total_done = 0, total_failed = 0;
    uint64_t total_bytes_in = 0, total_bytes_out = 0;
    for (int i = 0; i < cfg.thread_count; ++i) {
        total_done += atomic_load(&workers[i].items_done);
        total_failed += atomic_load(&workers[i].items_failed);
        total_bytes_in += atomic_load(&workers[i].bytes_in);
        total_bytes_out += atomic_load(&workers[i].bytes_out);
    }

    fprintf(stderr,
        "batch_runner: complete â€” %llu done, %llu failed, "
        "%llu bytes in, %llu bytes out\n",
        (unsigned long long)total_done, (unsigned long long)total_failed,
        (unsigned long long)total_bytes_in, (unsigned long long)total_bytes_out
    );

    free(workers);

    if (total_failed > 0 && total_done > 0) return 1;
    if (total_failed > 0 && total_done == 0) return 2;
    return 0;
}
