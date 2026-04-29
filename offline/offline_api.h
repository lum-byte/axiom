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
#define AXIOM_OFFLINE_GPU_UNAVAILABLE 6
#define AXIOM_OFFLINE_QUEUE_FULL 7
#define AXIOM_OFFLINE_PARSE_ERROR 8
#define AXIOM_OFFLINE_NOT_READY 9

#define AXIOM_ENCODER_WIDTH 256u
#define AXIOM_ENCODER_DEFAULT_MAX_SEQ_LEN 512
#define AXIOM_ENCODER_MAX_CHUNK 512
#define AXIOM_STRUCTURAL_DEFAULT_ROWS 1024u
#define AXIOM_STRUCTURAL_MAGIC 0x41584f46u
#define AXIOM_OFFLINE_EVENT_JSON_CAP 2048u

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

typedef struct axiom_encoder_stats {
    size_t batches_encoded;
    size_t zones_encoded;
    size_t chunks_encoded;
    size_t bytes_seen;
    size_t empty_inputs;
    size_t rejected_inputs;
    float last_norm_min;
    float last_norm_max;
    float last_norm_mean;
    float last_density_mean;
    int gpu_ready;
    int cpu_fallback;
} axiom_encoder_stats;

typedef struct axiom_text_batch {
    const char **texts;
    const int *labels;
    int batch_size;
    int max_seq_len;
} axiom_text_batch;

typedef struct axiom_update_stats {
    float gradient_norm;
    float applied_scale;
    float max_abs_update;
    float mean_abs_update;
    float weight_decay;
    size_t nonzero_gradient_values;
} axiom_update_stats;

typedef struct axiom_optimizer_state {
    float *m;
    float *v;
    size_t value_count;
    size_t step;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
} axiom_optimizer_state;

typedef struct axiom_accumulator_state {
    float *sum_buffer;
    float *ring_buffer;
    size_t value_count;
    size_t accumulate_steps;
    size_t step;
    size_t cursor;
    size_t flushed_batches;
    float last_observed_norm;
    float max_observed_norm;
} axiom_accumulator_state;

typedef struct axiom_batch_scheduler {
    size_t accepted;
    size_t rejected;
    size_t encoded;
    size_t flushed;
    size_t dead_lettered;
    size_t max_batch;
    float min_density;
} axiom_batch_scheduler;

typedef struct axiom_offline_work_item {
    char run_id[64];
    char url[512];
    char topology_class[96];
    char input_path[512];
    int label;
    float density;
    int priority;
} axiom_offline_work_item;

typedef struct axiom_scheduler_options {
    const char *queue_path;
    const char *store_dir;
    const char *dead_letter_path;
    int accumulate_steps;
    float learning_rate;
    float min_density;
    size_t max_batch;
} axiom_scheduler_options;

typedef struct axiom_scheduler_stats {
    size_t lines_read;
    size_t parse_errors;
    size_t encoded_items;
    size_t skipped_items;
    size_t update_steps;
    size_t dead_lettered;
    float last_latency_ms;
} axiom_scheduler_stats;

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

int gpu_encoder_init(const char *structural_layer_path);
int gpu_encoder_encode_batch(
    const char **texts,
    int batch_size,
    float *output,
    int max_seq_len
);
void gpu_encoder_shutdown(void);
int gpu_encoder_health(void);
int gpu_encoder_get_stats(axiom_encoder_stats *stats);
int gpu_encoder_reset_stats(void);
int axiom_gpu_kernel_symbols_present(void);
int axiom_gpu_estimate_chunks(int batch_size, int max_chunk);

int axiom_gradient_init(axiom_gradient_buffer *buffer, float *storage, size_t value_count, size_t target_steps);
int axiom_gradient_accumulate(axiom_gradient_buffer *buffer, const float *gradient, size_t value_count);
int axiom_gradient_ready(const axiom_gradient_buffer *buffer);
int axiom_gradient_reset(axiom_gradient_buffer *buffer);
int axiom_gradient_clip(float *gradient, size_t value_count, float clip_norm, float *observed_norm);

int axiom_accumulator_init(
    axiom_accumulator_state *state,
    float *sum_storage,
    float *ring_storage,
    size_t value_count,
    size_t accumulate_steps
);
int axiom_accumulator_add(axiom_accumulator_state *state, const float *gradient, size_t value_count);
int axiom_accumulator_ready(const axiom_accumulator_state *state);
int axiom_accumulator_flush(axiom_accumulator_state *state, float *output, size_t value_count);
int axiom_accumulator_reset(axiom_accumulator_state *state);
int axiom_accumulator_kernel_symbols_present(void);

int axiom_weight_update(float *weights, const float *gradient, size_t value_count, float lr, float clip_norm);
int axiom_weight_update_with_stats(
    float *weights,
    const float *gradient,
    size_t value_count,
    float lr,
    float clip_norm,
    axiom_update_stats *stats
);
int axiom_optimizer_init(axiom_optimizer_state *state, float *m, float *v, size_t value_count);
int axiom_weight_update_adam(
    float *weights,
    const float *gradient,
    size_t value_count,
    float lr,
    axiom_optimizer_state *state,
    axiom_update_stats *stats
);
int axiom_weight_update_adamw(
    float *weights,
    const float *gradient,
    size_t value_count,
    float lr,
    float clip_norm,
    axiom_optimizer_state *state,
    axiom_update_stats *stats
);

uint32_t axiom_offline_crc32(const uint8_t *data, size_t len);
int axiom_sha256_hex(const uint8_t *data, size_t len, char out_hex65[65]);
int axiom_file_sha256_hex(const char *path, char out_hex65[65]);
int axiom_publish_weights(const char *staging_path, const char *final_path, const float *weights, size_t value_count);
int axiom_weights_updated_event_json(
    const char *model_name,
    const char *store_path,
    const char *staging_path,
    const char *checksum_sha256,
    int version,
    int batch_count,
    int gradient_steps,
    const char *run_id,
    char *out_json,
    size_t out_capacity
);

int weight_updater_init(const char *structural_layer_path, float lr);
int weight_updater_accumulate(const float *embeddings, const int *labels, int batch_size);
int weight_updater_step(void);
int weight_updater_checkpoint(void);
void weight_updater_shutdown(void);
int axiom_weight_kernel_symbols_present(void);

int axiom_scheduler_init(axiom_batch_scheduler *stats, size_t max_batch, float min_density);
int axiom_scheduler_run_once(
    const axiom_zone_feature *features,
    size_t count,
    float *encoded,
    size_t encoded_count,
    axiom_batch_scheduler *stats
);
int axiom_scheduler_parse_work_line(const char *line, axiom_offline_work_item *item);
int axiom_scheduler_dead_letter(const char *path, const axiom_offline_work_item *item, const char *reason);
int axiom_scheduler_run_queue(const axiom_scheduler_options *options, axiom_scheduler_stats *stats);

#ifdef __cplusplus
}
#endif

#endif
