/*
 * batch_runner.h — Internal header for the batch runner.
 *
 * Ring buffer queue, work item types, CLI config, and shared state
 * between batch_runner_part*.c translation units.
 */

#ifndef AXIOM_BATCH_RUNNER_H
#define AXIOM_BATCH_RUNNER_H

#include "strip_engine.h"

#include <stdatomic.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define BATCH_QUEUE_CAPACITY  4096u   /* must be power of 2 */
#define BATCH_MAX_PATH        4096u
#define BATCH_MAX_URL         8192u
#define BATCH_LINE_MAX        (1024u * 1024u)
#define BATCH_STATS_INTERVAL  5       /* seconds */
#define BATCH_DEFAULT_THREADS 0       /* 0 = nproc */
#define BATCH_DEFAULT_POOL_MB 64

/* ------------------------------------------------------------------ */
/*  Work item                                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    char     url[BATCH_MAX_URL];
    int      slot_idx;
    char     input_path[BATCH_MAX_PATH];
    char     output_path[BATCH_MAX_PATH];
    size_t   input_size;
} batch_work_item;

/* ------------------------------------------------------------------ */
/*  Ring buffer (lock-free SPMC)                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    batch_work_item  items[BATCH_QUEUE_CAPACITY];
    atomic_uint_fast64_t head;
    atomic_uint_fast64_t tail;
    atomic_bool          done;       /* producer signals completion */
} batch_ring_buffer;

void  ring_init(batch_ring_buffer *rb);
bool  ring_push(batch_ring_buffer *rb, const batch_work_item *item);
bool  ring_pop(batch_ring_buffer *rb, batch_work_item *item);
bool  ring_is_empty(const batch_ring_buffer *rb);
size_t ring_size(const batch_ring_buffer *rb);

/* ------------------------------------------------------------------ */
/*  CLI configuration                                                  */
/* ------------------------------------------------------------------ */

typedef struct {
    char     queue_path[BATCH_MAX_PATH];
    char     mmap_path[BATCH_MAX_PATH];
    int      thread_count;
    int      pool_mb;
    bool     verbose;
    bool     dry_run;
} batch_config;

int batch_parse_args(int argc, char **argv, batch_config *cfg);

/* ------------------------------------------------------------------ */
/*  JSONL parsing                                                      */
/* ------------------------------------------------------------------ */

int batch_parse_line(const char *line, size_t len, batch_work_item *item);
int batch_load_queue(const char *path, batch_ring_buffer *rb,
                     size_t *items_loaded);

/* ------------------------------------------------------------------ */
/*  Worker pool                                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    int               thread_id;
    batch_ring_buffer *queue;
    const char        *mmap_path;
    int                pool_mb;
    atomic_uint_fast64_t items_done;
    atomic_uint_fast64_t items_failed;
    atomic_uint_fast64_t bytes_in;
    atomic_uint_fast64_t bytes_out;
    atomic_bool          stop;
} batch_worker_ctx;

int  batch_worker_run(batch_worker_ctx *ctx);
int  batch_start_workers(batch_worker_ctx *workers, int count,
                         batch_ring_buffer *queue, const batch_config *cfg);
void batch_stop_workers(batch_worker_ctx *workers, int count);
void batch_join_workers(int count);

/* ------------------------------------------------------------------ */
/*  Stats thread                                                       */
/* ------------------------------------------------------------------ */

typedef struct {
    batch_worker_ctx *workers;
    int               worker_count;
    atomic_bool       stop;
} batch_stats_ctx;

int  batch_stats_start(batch_stats_ctx *ctx, batch_worker_ctx *workers,
                       int worker_count);
void batch_stats_stop(batch_stats_ctx *ctx);

/* ------------------------------------------------------------------ */
/*  File I/O                                                           */
/* ------------------------------------------------------------------ */

int  batch_read_file(const char *path, uint8_t **data, size_t *len);
int  batch_write_file_atomic(const char *path, const uint8_t *data,
                             size_t len);

/* ------------------------------------------------------------------ */
/*  Signal handling                                                    */
/* ------------------------------------------------------------------ */

void batch_install_signal_handlers(void);
bool batch_should_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* AXIOM_BATCH_RUNNER_H */
