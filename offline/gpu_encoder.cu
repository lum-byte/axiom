#include "offline_api.h"

#include <ctype.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct EncoderRuntime {
    int initialized;
    int gpu_ready;
    char structural_layer_path[512];
    axiom_encoder_stats stats;
} EncoderRuntime;

typedef struct TokenStats {
    uint64_t hash;
    uint32_t count;
    uint32_t length_sum;
    uint32_t digit_count;
    uint32_t upper_count;
    uint32_t punctuation_count;
} TokenStats;

static EncoderRuntime g_encoder;

#if defined(__CUDACC__)
__global__ void axiom_encode_zones_kernel(
    const int *token_ids,
    const int *seq_lengths,
    float *embeddings,
    const float *embedding_table,
    int batch_size,
    int max_seq_len,
    int vocab_size
) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    if (row >= batch_size || col >= (int)AXIOM_ENCODER_WIDTH) {
        return;
    }
    int length = seq_lengths[row];
    if (length <= 0) {
        embeddings[row * (int)AXIOM_ENCODER_WIDTH + col] = 0.0f;
        return;
    }
    if (length > max_seq_len) {
        length = max_seq_len;
    }
    float sum = 0.0f;
    for (int t = 0; t < length; ++t) {
        int token = token_ids[row * max_seq_len + t];
        if (token < 0) {
            token = -token;
        }
        token = vocab_size > 0 ? token % vocab_size : 0;
        sum += embedding_table[token * (int)AXIOM_ENCODER_WIDTH + col];
    }
    embeddings[row * (int)AXIOM_ENCODER_WIDTH + col] = sum / (float)length;
}

__global__ void axiom_mean_pool_kernel(
    const float *token_embeddings,
    const int *seq_lengths,
    float *pooled,
    int batch_size,
    int max_seq_len
) {
    int row = blockIdx.x;
    int col = threadIdx.x + blockIdx.y * blockDim.x;
    if (row >= batch_size || col >= (int)AXIOM_ENCODER_WIDTH) {
        return;
    }
    int length = seq_lengths[row];
    if (length <= 0) {
        pooled[row * (int)AXIOM_ENCODER_WIDTH + col] = 0.0f;
        return;
    }
    if (length > max_seq_len) {
        length = max_seq_len;
    }
    float sum = 0.0f;
    for (int t = 0; t < length; ++t) {
        size_t idx = ((size_t)row * (size_t)max_seq_len + (size_t)t) * (size_t)AXIOM_ENCODER_WIDTH + (size_t)col;
        sum += token_embeddings[idx];
    }
    pooled[row * (int)AXIOM_ENCODER_WIDTH + col] = sum / (float)length;
}
#endif

static uint64_t fnv1a64_update(uint64_t h, unsigned char b) {
    h ^= (uint64_t)b;
    h *= 1099511628211ull;
    return h;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    x ^= x >> 31;
    return x;
}

static float signed_unit(uint64_t h) {
    uint32_t low = (uint32_t)(h & 0x00FFFFFFu);
    float unit = (float)low / (float)0x00FFFFFFu;
    return unit * 2.0f - 1.0f;
}

static float clamp01(float v) {
    if (v < 0.0f) return 0.0f;
    if (v > 1.0f) return 1.0f;
    return v;
}

static axiom_encoder_config default_encoder_config(void) {
    axiom_encoder_config cfg;
    cfg.density_scale = 1.0f;
    cfg.token_scale = 16.0f;
    cfg.byte_scale = 20.0f;
    cfg.zone_scale = 8.0f;
    cfg.bias = 0.0f;
    cfg.normalize_rows = 1;
    return cfg;
}

static float safe_log_feature(float v, float scale) {
    if (v < 0.0f || !isfinite(v)) {
        v = 0.0f;
    }
    if (scale <= 0.0f || !isfinite(scale)) {
        scale = 1.0f;
    }
    return log1pf(v) / scale;
}

static float row_norm(const float *row, size_t width) {
    float norm = 0.0f;
    for (size_t i = 0; i < width; ++i) {
        norm += row[i] * row[i];
    }
    return sqrtf(norm);
}

static void normalize_row(float *row, size_t width) {
    float norm = row_norm(row, width);
    if (norm <= 0.0f || !isfinite(norm)) {
        return;
    }
    for (size_t i = 0; i < width; ++i) {
        row[i] /= norm;
    }
}

static void zero_row(float *row) {
    memset(row, 0, sizeof(float) * AXIOM_ENCODER_WIDTH);
}

static int is_token_byte(unsigned char b) {
    return isalnum((int)b) || b == '_' || b == '-' || b == '#' || b == '.' || b == '/';
}

static int collect_token_stats(
    const char *text,
    int max_seq_len,
    TokenStats *stats,
    int *out_count,
    int *out_empty
) {
    if (text == 0 || stats == 0 || out_count == 0 || out_empty == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    int max_tokens = max_seq_len <= 0 ? AXIOM_ENCODER_DEFAULT_MAX_SEQ_LEN : max_seq_len;
    if (max_tokens > AXIOM_ENCODER_DEFAULT_MAX_SEQ_LEN) {
        max_tokens = AXIOM_ENCODER_DEFAULT_MAX_SEQ_LEN;
    }
    int token_count = 0;
    size_t len = strlen(text);
    *out_empty = len == 0u ? 1 : 0;
    size_t i = 0;
    while (i < len && token_count < max_tokens) {
        while (i < len && !is_token_byte((unsigned char)text[i])) {
            i++;
        }
        if (i >= len) {
            break;
        }
        size_t start = i;
        uint64_t h = 1469598103934665603ull;
        uint32_t digit_count = 0;
        uint32_t upper_count = 0;
        uint32_t punct_count = 0;
        while (i < len && is_token_byte((unsigned char)text[i])) {
            unsigned char b = (unsigned char)text[i];
            unsigned char lower = (unsigned char)tolower((int)b);
            h = fnv1a64_update(h, lower);
            if (isdigit((int)b)) digit_count++;
            if (isupper((int)b)) upper_count++;
            if (!isalnum((int)b)) punct_count++;
            i++;
        }
        size_t tok_len = i - start;
        if (tok_len > 0u) {
            stats[token_count].hash = mix64(h ^ (uint64_t)tok_len);
            stats[token_count].count = 1u;
            stats[token_count].length_sum = (uint32_t)tok_len;
            stats[token_count].digit_count = digit_count;
            stats[token_count].upper_count = upper_count;
            stats[token_count].punctuation_count = punct_count;
            token_count++;
        }
    }
    *out_count = token_count;
    return AXIOM_OFFLINE_OK;
}

static float lexical_density(const TokenStats *tokens, int token_count, size_t byte_count) {
    if (token_count <= 0 || tokens == 0 || byte_count == 0u) {
        return 0.0f;
    }
    uint64_t unique_probe[64];
    int unique_count = 0;
    int probe_cap = token_count < 64 ? token_count : 64;
    for (int i = 0; i < token_count && unique_count < probe_cap; ++i) {
        int seen = 0;
        for (int j = 0; j < unique_count; ++j) {
            if (unique_probe[j] == tokens[i].hash) {
                seen = 1;
                break;
            }
        }
        if (!seen) {
            unique_probe[unique_count++] = tokens[i].hash;
        }
    }
    float unique_ratio = (float)unique_count / (float)(token_count < 64 ? token_count : 64);
    float token_byte_ratio = (float)token_count / (float)(byte_count + 1u);
    float density = unique_ratio * 0.75f + clamp01(token_byte_ratio * 8.0f) * 0.25f;
    return clamp01(density);
}

static void add_feature_priors(float *row, const TokenStats *tokens, int token_count, size_t byte_count) {
    if (row == 0) {
        return;
    }
    float density = lexical_density(tokens, token_count, byte_count);
    float token_feature = safe_log_feature((float)token_count, 16.0f);
    float byte_feature = safe_log_feature((float)byte_count, 20.0f);
    float punctuation = 0.0f;
    float upper = 0.0f;
    float digits = 0.0f;
    float chars = 0.0f;
    for (int i = 0; i < token_count; ++i) {
        punctuation += (float)tokens[i].punctuation_count;
        upper += (float)tokens[i].upper_count;
        digits += (float)tokens[i].digit_count;
        chars += (float)tokens[i].length_sum;
    }
    if (chars <= 0.0f) {
        chars = 1.0f;
    }
    float structural[8] = {
        density,
        token_feature,
        byte_feature,
        punctuation / chars,
        upper / chars,
        digits / chars,
        density * token_feature,
        density * byte_feature,
    };
    for (size_t d = 0; d < AXIOM_ENCODER_WIDTH; ++d) {
        float basis = structural[d % 8u];
        float phase = (float)((d * 37u + 11u) % 127u) / 127.0f;
        row[d] += basis * (0.02f + phase * 0.01f);
    }
}

static int encode_text_row(const char *text, int max_seq_len, float *row) {
    if (row == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    zero_row(row);
    if (text == 0 || text[0] == '\0') {
        g_encoder.stats.empty_inputs++;
        return AXIOM_OFFLINE_OK;
    }
    TokenStats tokens[AXIOM_ENCODER_DEFAULT_MAX_SEQ_LEN];
    int token_count = 0;
    int empty = 0;
    int code = collect_token_stats(text, max_seq_len, tokens, &token_count, &empty);
    if (code != AXIOM_OFFLINE_OK) {
        return code;
    }
    size_t byte_count = strlen(text);
    g_encoder.stats.bytes_seen += byte_count;
    if (token_count == 0 || empty) {
        g_encoder.stats.empty_inputs++;
        return AXIOM_OFFLINE_OK;
    }
    for (int t = 0; t < token_count; ++t) {
        uint64_t token_hash = tokens[t].hash;
        float len_weight = 1.0f + fminf((float)tokens[t].length_sum, 32.0f) / 64.0f;
        float syntax_weight = 1.0f
            + (float)tokens[t].punctuation_count * 0.015f
            + (float)tokens[t].digit_count * 0.010f
            + (float)tokens[t].upper_count * 0.005f;
        float weight = len_weight * syntax_weight;
        for (size_t d = 0; d < AXIOM_ENCODER_WIDTH; ++d) {
            uint64_t mixed = mix64(token_hash ^ ((uint64_t)d * 0x9e3779b97f4a7c15ull));
            row[d] += signed_unit(mixed) * weight;
        }
    }
    float inv = 1.0f / (float)token_count;
    for (size_t d = 0; d < AXIOM_ENCODER_WIDTH; ++d) {
        row[d] *= inv;
    }
    add_feature_priors(row, tokens, token_count, byte_count);
    normalize_row(row, AXIOM_ENCODER_WIDTH);
    return AXIOM_OFFLINE_OK;
}

static void stats_begin_batch(void) {
    g_encoder.stats.last_norm_min = 0.0f;
    g_encoder.stats.last_norm_max = 0.0f;
    g_encoder.stats.last_norm_mean = 0.0f;
    g_encoder.stats.last_density_mean = 0.0f;
}

static void stats_record_norm(const float *row, int first) {
    float n = row_norm(row, AXIOM_ENCODER_WIDTH);
    if (first) {
        g_encoder.stats.last_norm_min = n;
        g_encoder.stats.last_norm_max = n;
    } else {
        if (n < g_encoder.stats.last_norm_min) g_encoder.stats.last_norm_min = n;
        if (n > g_encoder.stats.last_norm_max) g_encoder.stats.last_norm_max = n;
    }
    g_encoder.stats.last_norm_mean += n;
}

extern "C" int axiom_feature_normalize(axiom_zone_feature *features, size_t count) {
    if (features == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    for (size_t i = 0; i < count; ++i) {
        features[i].density = clamp01(features[i].density);
        if (features[i].token_count < 0.0f || !isfinite(features[i].token_count)) features[i].token_count = 0.0f;
        if (features[i].byte_count < 0.0f || !isfinite(features[i].byte_count)) features[i].byte_count = 0.0f;
        if (features[i].zone_count < 0.0f || !isfinite(features[i].zone_count)) features[i].zone_count = 0.0f;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_gpu_encode(
    const axiom_zone_feature *features,
    size_t batch_size,
    float *output,
    size_t output_count
) {
    axiom_encoder_config cfg = default_encoder_config();
    return axiom_gpu_encode_configured(features, batch_size, &cfg, output, output_count);
}

extern "C" int axiom_gpu_encode_configured(
    const axiom_zone_feature *features,
    size_t batch_size,
    const axiom_encoder_config *config,
    float *output,
    size_t output_count
) {
    if (features == 0 || output == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (output_count < batch_size * AXIOM_ENCODER_WIDTH) {
        return AXIOM_OFFLINE_SHAPE_ERROR;
    }
    axiom_encoder_config cfg = config != 0 ? *config : default_encoder_config();
    if (cfg.density_scale <= 0.0f) cfg.density_scale = 1.0f;
    if (cfg.token_scale <= 0.0f) cfg.token_scale = 16.0f;
    if (cfg.byte_scale <= 0.0f) cfg.byte_scale = 20.0f;
    if (cfg.zone_scale <= 0.0f) cfg.zone_scale = 8.0f;
    for (size_t row = 0; row < batch_size; ++row) {
        const axiom_zone_feature f = features[row];
        float base[4] = {
            clamp01(f.density) * cfg.density_scale,
            safe_log_feature(f.token_count, cfg.token_scale),
            safe_log_feature(f.byte_count, cfg.byte_scale),
            safe_log_feature(f.zone_count, cfg.zone_scale),
        };
        float cross[4] = {
            base[0] * base[1],
            base[0] * base[2],
            base[1] * base[3],
            base[2] * base[3],
        };
        float *row_out = output + row * AXIOM_ENCODER_WIDTH;
        for (size_t col = 0; col < AXIOM_ENCODER_WIDTH; ++col) {
            float harmonic = 1.0f + (float)(col % 17u) / 32.0f;
            float phase = (float)((col * 37u + row * 13u) % 101u) / 101.0f;
            float v = base[col % 4u] * harmonic + cross[(col / 4u) % 4u] * 0.25f + phase * 0.01f + cfg.bias;
            if (!isfinite(v)) {
                return AXIOM_OFFLINE_NUMERIC_ERROR;
            }
            row_out[col] = v;
        }
        if (cfg.normalize_rows) {
            normalize_row(row_out, AXIOM_ENCODER_WIDTH);
        }
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int gpu_encoder_init(const char *structural_layer_path) {
    memset(&g_encoder, 0, sizeof(g_encoder));
    g_encoder.initialized = 1;
    g_encoder.gpu_ready = 0;
    g_encoder.stats.gpu_ready = 0;
    g_encoder.stats.cpu_fallback = 1;
    if (structural_layer_path != 0) {
        snprintf(g_encoder.structural_layer_path, sizeof(g_encoder.structural_layer_path), "%s", structural_layer_path);
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int gpu_encoder_encode_batch(
    const char **texts,
    int batch_size,
    float *output,
    int max_seq_len
) {
    if (!g_encoder.initialized) {
        int code = gpu_encoder_init(0);
        if (code != AXIOM_OFFLINE_OK) {
            return code;
        }
    }
    if (texts == 0 || output == 0 || batch_size < 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (batch_size == 0) {
        return AXIOM_OFFLINE_NO_WORK;
    }
    stats_begin_batch();
    int processed = 0;
    int first_norm = 1;
    while (processed < batch_size) {
        int remaining = batch_size - processed;
        int chunk = remaining > AXIOM_ENCODER_MAX_CHUNK ? AXIOM_ENCODER_MAX_CHUNK : remaining;
        g_encoder.stats.chunks_encoded++;
        for (int i = 0; i < chunk; ++i) {
            float *row = output + ((size_t)(processed + i) * AXIOM_ENCODER_WIDTH);
            int code = encode_text_row(texts[processed + i], max_seq_len, row);
            if (code != AXIOM_OFFLINE_OK) {
                g_encoder.stats.rejected_inputs++;
                return code;
            }
            stats_record_norm(row, first_norm);
            first_norm = 0;
        }
        processed += chunk;
    }
    g_encoder.stats.batches_encoded++;
    g_encoder.stats.zones_encoded += (size_t)batch_size;
    if (batch_size > 0) {
        g_encoder.stats.last_norm_mean /= (float)batch_size;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" void gpu_encoder_shutdown(void) {
    memset(&g_encoder, 0, sizeof(g_encoder));
}

extern "C" int gpu_encoder_health(void) {
    if (!g_encoder.initialized) {
        return 0;
    }
    return 1;
}

extern "C" int gpu_encoder_get_stats(axiom_encoder_stats *stats) {
    if (stats == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    *stats = g_encoder.stats;
    return AXIOM_OFFLINE_OK;
}

extern "C" int gpu_encoder_reset_stats(void) {
    memset(&g_encoder.stats, 0, sizeof(g_encoder.stats));
    g_encoder.stats.gpu_ready = g_encoder.gpu_ready;
    g_encoder.stats.cpu_fallback = 1;
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_gpu_kernel_symbols_present(void) {
#if defined(__CUDACC__)
    return 1;
#else
    return 0;
#endif
}

extern "C" int axiom_gpu_estimate_chunks(int batch_size, int max_chunk) {
    if (batch_size < 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    int chunk = max_chunk <= 0 ? AXIOM_ENCODER_MAX_CHUNK : max_chunk;
    if (chunk <= 0) {
        chunk = AXIOM_ENCODER_MAX_CHUNK;
    }
    if (batch_size == 0) {
        return 0;
    }
    return (batch_size + chunk - 1) / chunk;
}
