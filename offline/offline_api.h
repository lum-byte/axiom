#ifndef AXIOM_OFFLINE_API_H
#define AXIOM_OFFLINE_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_OFFLINE_OK 0
#define AXIOM_OFFLINE_INVALID_ARG 1
#define AXIOM_OFFLINE_NO_WORK 2
#define AXIOM_OFFLINE_IO_ERROR 3
#define AXIOM_OFFLINE_NUMERIC_ERROR 4
#define AXIOM_OFFLINE_SHAPE_ERROR 5

#define AXIOM_ENCODER_WIDTH 256u

typedef struct axiom_zone_feature {
    float density;
    float token_count;
    float byte_count;
    float zone_count;
} axiom_zone_feature;

typedef struct axiom_gradient_buffer {
    float *values;
    size_t value_count;
    size_t accumulated_steps;
    size_t target_steps;
} axiom_gradient_buffer;

typedef struct axiom_encoder_config {
    float density_scale;
    float token_scale;
    float byte_scale;
    float zone_scale;
    float bias;
    int normalize_rows;
} axiom_encoder_config;

typedef struct axiom_update_stats {
    float gradient_norm;
    float applied_scale;
    float max_abs_update;
    float mean_abs_update;
} axiom_update_stats;

typedef struct axiom_optimizer_state {
    float *m;
    float *v;
    size_t value_count;
    size_t step;
    float beta1;
    float beta2;
    float epsilon;
} axiom_optimizer_state;

typedef struct axiom_batch_scheduler {
    size_t accepted;
    size_t rejected;
    size_t encoded;
    size_t flushed;
    size_t max_batch;
    float min_density;
} axiom_batch_scheduler;

int axiom_gpu_encode(
    const axiom_zone_feature *features,
    size_t batch_size,
    float *output,
    size_t output_count
);

int axiom_gpu_encode_configured(
    const axiom_zone_feature *features,
    size_t batch_size,
    const axiom_encoder_config *config,
    float *output,
    size_t output_count
);

int axiom_feature_normalize(axiom_zone_feature *features, size_t count);
int axiom_gradient_init(axiom_gradient_buffer *buffer, float *storage, size_t value_count, size_t target_steps);
int axiom_gradient_accumulate(axiom_gradient_buffer *buffer, const float *gradient, size_t value_count);
int axiom_gradient_ready(const axiom_gradient_buffer *buffer);
int axiom_gradient_reset(axiom_gradient_buffer *buffer);
int axiom_gradient_clip(float *gradient, size_t value_count, float clip_norm, float *observed_norm);
int axiom_weight_update(float *weights, const float *gradient, size_t value_count, float lr, float clip_norm);
int axiom_weight_update_with_stats(float *weights, const float *gradient, size_t value_count, float lr, float clip_norm, axiom_update_stats *stats);
int axiom_optimizer_init(axiom_optimizer_state *state, float *m, float *v, size_t value_count);
int axiom_weight_update_adam(float *weights, const float *gradient, size_t value_count, float lr, axiom_optimizer_state *state, axiom_update_stats *stats);
uint32_t axiom_offline_crc32(const uint8_t *data, size_t len);
int axiom_publish_weights(const char *staging_path, const char *final_path, const float *weights, size_t value_count);
int axiom_scheduler_init(axiom_batch_scheduler *stats, size_t max_batch, float min_density);
int axiom_scheduler_run_once(const axiom_zone_feature *features, size_t count, float *encoded, size_t encoded_count, axiom_batch_scheduler *stats);

#ifdef __cplusplus
}
#endif

#endif
