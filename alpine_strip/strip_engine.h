/*
 * strip_engine.h — Public API for the AXIOM alpine_strip engine.
 *
 * Offline batch stripping layer.  Takes raw HTML bytes, applies a
 * compiled recipe and returns signal bytes.  Designed for 100×
 * throughput over the online signal_kernel path (no subprocess, no
 * Alpine container overhead).
 *
 * Compile: C11, -O3 -march=native, link against libpcre2-8.
 * Thread-safety: callers must supply a per-thread strip_pool.
 */

#ifndef AXIOM_STRIP_ENGINE_H
#define AXIOM_STRIP_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define AXIOM_STRIP_MAX_STEPS       128
#define AXIOM_STRIP_DEFAULT_RATIO   0.30
#define AXIOM_POOL_BYTES            (64u * 1024u * 1024u)  /* 64 MB */
#define AXIOM_STRIP_MAX_PATTERN     4096
#define AXIOM_STRIP_TOPOLOGY_LEN    64

/* ------------------------------------------------------------------ */
/*  Status / error codes                                               */
/* ------------------------------------------------------------------ */

#define AXIOM_STRIP_OK                   0
#define AXIOM_STRIP_ERR_INVALID_ARG      1
#define AXIOM_STRIP_ERR_BAD_RECIPE       2
#define AXIOM_STRIP_ERR_POOL_EXHAUSTED   3
#define AXIOM_STRIP_ERR_REDUCTION_RATIO  4
#define AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL 5
#define AXIOM_STRIP_ERR_PCRE2_COMPILE    6
#define AXIOM_STRIP_ERR_PCRE2_MATCH      7
#define AXIOM_STRIP_ERR_MMAP_OPEN        8
#define AXIOM_STRIP_ERR_MMAP_READ        9
#define AXIOM_STRIP_ERR_CRC_MISMATCH     10
#define AXIOM_STRIP_ERR_SLOT_OOB         11
#define AXIOM_STRIP_EMPTY                12
#define AXIOM_STRIP_TOO_BROAD            13

/* ------------------------------------------------------------------ */
/*  Step flags (bit-field)                                             */
/* ------------------------------------------------------------------ */

#define STRIP_EXTRACT   0x01u
#define STRIP_REMOVE    0x02u
#define STRIP_REQUIRE   0x04u

/* ------------------------------------------------------------------ */
/*  Pool allocator (per-thread)                                        */
/* ------------------------------------------------------------------ */

typedef struct strip_pool {
    uint8_t *base;
    size_t   capacity;
    size_t   offset;
} strip_pool;

int   strip_pool_init(strip_pool *pool, size_t capacity_bytes);
void  strip_pool_reset(strip_pool *pool);
void  strip_pool_destroy(strip_pool *pool);
void *strip_pool_alloc(strip_pool *pool, size_t n);
void *strip_pool_alloc_zero(strip_pool *pool, size_t n);
size_t strip_pool_used(const strip_pool *pool);
size_t strip_pool_remaining(const strip_pool *pool);
char *strip_pool_strdup(strip_pool *pool, const char *s);
char *strip_pool_strndup(strip_pool *pool, const char *s, size_t max_len);

/* ------------------------------------------------------------------ */
/*  Recipe types (spec-aligned)                                        */
/* ------------------------------------------------------------------ */

typedef struct {
    const char *kind;         /* step kind string, e.g. "strip_tag"   */
    const char *pattern;      /* pattern / argument                    */
    const char *replacement;  /* replacement / second argument         */
    uint8_t     flags;        /* STRIP_EXTRACT | STRIP_REMOVE | ...   */
    float       confidence;   /* minimum confidence to apply (0–1)    */
} axiom_strip_step;

typedef struct {
    axiom_strip_step *steps;
    size_t            step_count;
    char              topology_class[AXIOM_STRIP_TOPOLOGY_LEN];
    uint32_t          checksum;         /* CRC32 of recipe */
    double            max_output_ratio; /* 0 < r <= 1.0    */
} axiom_strip_recipe;

/* ------------------------------------------------------------------ */
/*  Result / metrics                                                   */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t   bytes_written;
    uint32_t crc32;
    int      code;
    uint32_t steps_fired;
} axiom_strip_result;

typedef struct {
    size_t input_bytes;
    size_t output_bytes;
    size_t ascii_letters;
    size_t ascii_digits;
    size_t whitespace;
    size_t punctuation;
    size_t line_count;
    size_t token_count;
    double output_ratio;
    double signal_density;
} axiom_strip_metrics;

/* ------------------------------------------------------------------ */
/*  PCRE2 regex wrapper                                                */
/* ------------------------------------------------------------------ */

typedef struct axiom_regex axiom_regex;

axiom_regex *axiom_regex_compile(const char *pattern, size_t pattern_len,
                                 uint32_t options, int *errcode);
int          axiom_regex_match(const axiom_regex *re, const char *subject,
                               size_t subject_len, size_t start_offset,
                               size_t *ovector, size_t ovector_count,
                               size_t *match_count);
size_t       axiom_regex_replace(const axiom_regex *re, const char *subject,
                                 size_t subject_len, const char *replacement,
                                 char *output, size_t output_capacity);
size_t       axiom_regex_extract_all(const axiom_regex *re, const char *subject,
                                     size_t subject_len, char *output,
                                     size_t output_capacity);
void         axiom_regex_free(axiom_regex *re);
int          axiom_regex_jit_hint(axiom_regex *re);

/* ------------------------------------------------------------------ */
/*  Recipe mmap loading                                                */
/* ------------------------------------------------------------------ */

#define AXIOM_RECIPE_SLOT_BYTES   4096
#define AXIOM_RECIPE_MAGIC        0x41584D52u  /* "AXMR" */
#define AXIOM_RECIPE_MAX_SLOTS    8192

typedef struct {
    uint32_t magic;
    uint32_t version;
    uint32_t slot_count;
    uint32_t reserved;
} axiom_recipe_registry_header;

typedef struct {
    char     topology_class[AXIOM_STRIP_TOPOLOGY_LEN];
    uint32_t step_count;
    uint32_t checksum;
    uint32_t data_offset;   /* offset within slot to step data */
    uint32_t data_len;
    uint8_t  reserved[16];
} axiom_recipe_slot_header;

axiom_strip_recipe *strip_load_recipe(int slot_idx, const char *mmap_path,
                                      strip_pool *pool);
void                strip_free_recipe(axiom_strip_recipe *recipe);
int                 strip_validate_recipe(const axiom_strip_recipe *recipe,
                                          const char *mmap_path);
int                 axiom_recipe_write_slot(const axiom_strip_recipe *recipe,
                                            uint8_t *slot_buf, size_t slot_size);
int                 axiom_recipe_write_registry(const axiom_strip_recipe *recipes,
                                                size_t count, uint8_t *buf,
                                                size_t buf_size);

/* ------------------------------------------------------------------ */
/*  Core API                                                           */
/* ------------------------------------------------------------------ */

int axiom_strip_apply(
    const uint8_t            *input,
    size_t                    input_len,
    const axiom_strip_recipe *recipe,
    uint8_t                  *output,
    size_t                    output_capacity,
    axiom_strip_result       *result
);

int axiom_strip_apply_with_pool(
    const uint8_t            *input,
    size_t                    input_len,
    const axiom_strip_recipe *recipe,
    uint8_t                  *output,
    size_t                    output_capacity,
    axiom_strip_result       *result,
    strip_pool               *pool
);

int axiom_strip_apply_default_html(
    const uint8_t       *input,
    size_t               input_len,
    uint8_t             *output,
    size_t               output_capacity,
    axiom_strip_result  *result
);

int axiom_strip_validate_recipe(const axiom_strip_recipe *recipe);

int axiom_strip_measure(
    const uint8_t        *input,
    size_t                input_len,
    const uint8_t        *output,
    size_t                output_len,
    axiom_strip_metrics  *metrics
);

/* ------------------------------------------------------------------ */
/*  Utilities                                                          */
/* ------------------------------------------------------------------ */

uint32_t axiom_crc32(const uint8_t *data, size_t len);

const char *axiom_strip_strerror(int code);

#ifdef __cplusplus
}
#endif

#endif /* AXIOM_STRIP_ENGINE_H */
