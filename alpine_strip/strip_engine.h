#ifndef AXIOM_STRIP_ENGINE_H
#define AXIOM_STRIP_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_STRIP_OK 0
#define AXIOM_STRIP_ERR_INVALID_ARG 1
#define AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL 2
#define AXIOM_STRIP_ERR_REDUCTION_RATIO 3
#define AXIOM_STRIP_ERR_POOL_EXHAUSTED 4
#define AXIOM_STRIP_ERR_BAD_RECIPE 5
#define AXIOM_STRIP_ERR_STEP_FAILED 6

#define AXIOM_STRIP_MAX_STEPS 128
#define AXIOM_STRIP_DEFAULT_RATIO 1.0

typedef struct axiom_strip_step {
    const char *kind;
    const char *pattern;
    const char *replacement;
} axiom_strip_step;

typedef struct axiom_strip_recipe {
    const axiom_strip_step *steps;
    size_t step_count;
    double max_output_ratio;
} axiom_strip_recipe;

typedef struct axiom_strip_result {
    int code;
    size_t bytes_written;
    size_t input_bytes;
    uint32_t crc32;
} axiom_strip_result;

typedef struct axiom_strip_metrics {
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

int axiom_strip_apply(
    const uint8_t *input,
    size_t input_len,
    const axiom_strip_recipe *recipe,
    uint8_t *output,
    size_t output_capacity,
    axiom_strip_result *result
);

uint32_t axiom_crc32(const uint8_t *data, size_t len);

int axiom_strip_validate_recipe(const axiom_strip_recipe *recipe);

int axiom_strip_apply_default_html(
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t output_capacity,
    axiom_strip_result *result
);

int axiom_strip_measure(
    const uint8_t *input,
    size_t input_len,
    const uint8_t *output,
    size_t output_len,
    axiom_strip_metrics *metrics
);

#ifdef __cplusplus
}
#endif

#endif
