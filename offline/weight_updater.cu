#include "offline_api.h"

#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <io.h>
#define AXIOM_FSYNC(fd) _commit(fd)
#else
#include <unistd.h>
#define AXIOM_FSYNC(fd) fsync(fd)
#endif

typedef struct Sha256Ctx {
    uint32_t state[8];
    uint64_t bit_len;
    uint8_t data[64];
    size_t data_len;
} Sha256Ctx;

typedef struct WeightUpdaterRuntime {
    int initialized;
    char structural_layer_path[512];
    char staging_path[576];
    float learning_rate;
    float clip_norm;
    size_t value_count;
    size_t row_count;
    int version;
    int gradient_steps;
    int batch_count;
    float *weights;
    float *m;
    float *v;
    float *gradient;
    axiom_optimizer_state optimizer;
    axiom_update_stats last_stats;
    char last_checksum[65];
} WeightUpdaterRuntime;

static WeightUpdaterRuntime g_updater;

static const uint32_t SHA256_K[64] = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

static uint32_t rotr32(uint32_t x, uint32_t n) {
    return (x >> n) | (x << (32u - n));
}

static void sha256_transform(Sha256Ctx *ctx, const uint8_t data[64]) {
    uint32_t m[64];
    for (int i = 0, j = 0; i < 16; ++i, j += 4) {
        m[i] = ((uint32_t)data[j] << 24)
            | ((uint32_t)data[j + 1] << 16)
            | ((uint32_t)data[j + 2] << 8)
            | ((uint32_t)data[j + 3]);
    }
    for (int i = 16; i < 64; ++i) {
        uint32_t s0 = rotr32(m[i - 15], 7) ^ rotr32(m[i - 15], 18) ^ (m[i - 15] >> 3);
        uint32_t s1 = rotr32(m[i - 2], 17) ^ rotr32(m[i - 2], 19) ^ (m[i - 2] >> 10);
        m[i] = m[i - 16] + s0 + m[i - 7] + s1;
    }
    uint32_t a = ctx->state[0];
    uint32_t b = ctx->state[1];
    uint32_t c = ctx->state[2];
    uint32_t d = ctx->state[3];
    uint32_t e = ctx->state[4];
    uint32_t f = ctx->state[5];
    uint32_t g = ctx->state[6];
    uint32_t h = ctx->state[7];
    for (int i = 0; i < 64; ++i) {
        uint32_t s1 = rotr32(e, 6) ^ rotr32(e, 11) ^ rotr32(e, 25);
        uint32_t ch = (e & f) ^ ((~e) & g);
        uint32_t temp1 = h + s1 + ch + SHA256_K[i] + m[i];
        uint32_t s0 = rotr32(a, 2) ^ rotr32(a, 13) ^ rotr32(a, 22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t temp2 = s0 + maj;
        h = g;
        g = f;
        f = e;
        e = d + temp1;
        d = c;
        c = b;
        b = a;
        a = temp1 + temp2;
    }
    ctx->state[0] += a;
    ctx->state[1] += b;
    ctx->state[2] += c;
    ctx->state[3] += d;
    ctx->state[4] += e;
    ctx->state[5] += f;
    ctx->state[6] += g;
    ctx->state[7] += h;
}

static void sha256_init(Sha256Ctx *ctx) {
    ctx->data_len = 0;
    ctx->bit_len = 0;
    ctx->state[0] = 0x6a09e667u;
    ctx->state[1] = 0xbb67ae85u;
    ctx->state[2] = 0x3c6ef372u;
    ctx->state[3] = 0xa54ff53au;
    ctx->state[4] = 0x510e527fu;
    ctx->state[5] = 0x9b05688cu;
    ctx->state[6] = 0x1f83d9abu;
    ctx->state[7] = 0x5be0cd19u;
}

static void sha256_update(Sha256Ctx *ctx, const uint8_t *data, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        ctx->data[ctx->data_len++] = data[i];
        if (ctx->data_len == 64u) {
            sha256_transform(ctx, ctx->data);
            ctx->bit_len += 512u;
            ctx->data_len = 0;
        }
    }
}

static void sha256_final(Sha256Ctx *ctx, uint8_t hash[32]) {
    size_t i = ctx->data_len;
    if (ctx->data_len < 56u) {
        ctx->data[i++] = 0x80u;
        while (i < 56u) ctx->data[i++] = 0x00u;
    } else {
        ctx->data[i++] = 0x80u;
        while (i < 64u) ctx->data[i++] = 0x00u;
        sha256_transform(ctx, ctx->data);
        memset(ctx->data, 0, 56u);
    }
    ctx->bit_len += ctx->data_len * 8u;
    ctx->data[63] = (uint8_t)(ctx->bit_len);
    ctx->data[62] = (uint8_t)(ctx->bit_len >> 8);
    ctx->data[61] = (uint8_t)(ctx->bit_len >> 16);
    ctx->data[60] = (uint8_t)(ctx->bit_len >> 24);
    ctx->data[59] = (uint8_t)(ctx->bit_len >> 32);
    ctx->data[58] = (uint8_t)(ctx->bit_len >> 40);
    ctx->data[57] = (uint8_t)(ctx->bit_len >> 48);
    ctx->data[56] = (uint8_t)(ctx->bit_len >> 56);
    sha256_transform(ctx, ctx->data);
    for (i = 0; i < 4u; ++i) {
        hash[i] = (uint8_t)((ctx->state[0] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 4] = (uint8_t)((ctx->state[1] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 8] = (uint8_t)((ctx->state[2] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 12] = (uint8_t)((ctx->state[3] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 16] = (uint8_t)((ctx->state[4] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 20] = (uint8_t)((ctx->state[5] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 24] = (uint8_t)((ctx->state[6] >> (24u - i * 8u)) & 0x000000ffu);
        hash[i + 28] = (uint8_t)((ctx->state[7] >> (24u - i * 8u)) & 0x000000ffu);
    }
}

static void bytes_to_hex(const uint8_t *bytes, size_t len, char *out) {
    static const char *hex = "0123456789abcdef";
    for (size_t i = 0; i < len; ++i) {
        out[i * 2u] = hex[(bytes[i] >> 4) & 0x0Fu];
        out[i * 2u + 1u] = hex[bytes[i] & 0x0Fu];
    }
    out[len * 2u] = '\0';
}

static float vector_norm(const float *values, size_t count) {
    float norm = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        norm += values[i] * values[i];
    }
    return sqrtf(norm);
}

static int vector_is_finite(const float *values, size_t count) {
    if (values == 0) {
        return 0;
    }
    for (size_t i = 0; i < count; ++i) {
        if (!isfinite(values[i])) {
            return 0;
        }
    }
    return 1;
}

static int replace_file(const char *staging_path, const char *final_path) {
#if defined(_WIN32)
    remove(final_path);
#endif
    return rename(staging_path, final_path);
}

#if defined(__CUDACC__)
__global__ void axiom_gradient_square_kernel(const float *grad, float *squares, int n_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_elements) {
        squares[idx] = grad[idx] * grad[idx];
    }
}

__global__ void axiom_gradient_clip_kernel(float *grad, float max_norm, int n_elements, const float *grad_norm) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements || grad_norm == 0 || max_norm <= 0.0f) {
        return;
    }
    float norm = grad_norm[0];
    if (norm > max_norm && norm > 0.0f) {
        grad[idx] *= max_norm / norm;
    }
}

__global__ void axiom_adamw_update_kernel(
    float *weights,
    const float *grad,
    float *m,
    float *v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int t,
    int n_elements
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_elements || t <= 0) {
        return;
    }
    float g = grad[idx];
    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;
    float b1_corr = 1.0f - powf(beta1, (float)t);
    float b2_corr = 1.0f - powf(beta2, (float)t);
    if (b1_corr <= 0.0f || b2_corr <= 0.0f) {
        return;
    }
    float mh = m[idx] / b1_corr;
    float vh = v[idx] / b2_corr;
    float update = lr * (mh / (sqrtf(vh) + eps) + weight_decay * weights[idx]);
    weights[idx] -= update;
}
#endif

extern "C" uint32_t axiom_offline_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    if (data == 0 && len > 0u) {
        return 0u;
    }
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int)(crc & 1u);
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

extern "C" int axiom_sha256_hex(const uint8_t *data, size_t len, char out_hex65[65]) {
    if ((data == 0 && len > 0u) || out_hex65 == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    Sha256Ctx ctx;
    uint8_t hash[32];
    sha256_init(&ctx);
    if (len > 0u) {
        sha256_update(&ctx, data, len);
    }
    sha256_final(&ctx, hash);
    bytes_to_hex(hash, sizeof(hash), out_hex65);
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_file_sha256_hex(const char *path, char out_hex65[65]) {
    if (path == 0 || out_hex65 == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    FILE *f = fopen(path, "rb");
    if (f == 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    Sha256Ctx ctx;
    sha256_init(&ctx);
    uint8_t buf[8192];
    size_t n = 0;
    while ((n = fread(buf, 1u, sizeof(buf), f)) > 0u) {
        sha256_update(&ctx, buf, n);
    }
    int err = ferror(f);
    fclose(f);
    if (err) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    uint8_t hash[32];
    sha256_final(&ctx, hash);
    bytes_to_hex(hash, sizeof(hash), out_hex65);
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_weight_update(float *weights, const float *gradient, size_t value_count, float lr, float clip_norm) {
    return axiom_weight_update_with_stats(weights, gradient, value_count, lr, clip_norm, 0);
}

extern "C" int axiom_weight_update_with_stats(
    float *weights,
    const float *gradient,
    size_t value_count,
    float lr,
    float clip_norm,
    axiom_update_stats *stats
) {
    if (weights == 0 || gradient == 0 || value_count == 0 || lr <= 0.0f) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (!vector_is_finite(weights, value_count) || !vector_is_finite(gradient, value_count)) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    float norm = vector_norm(gradient, value_count);
    float scale = 1.0f;
    if (clip_norm > 0.0f && norm > clip_norm) {
        scale = clip_norm / norm;
    }
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    size_t nonzero = 0;
    for (size_t i = 0; i < value_count; ++i) {
        float update = lr * gradient[i] * scale;
        weights[i] -= update;
        float abs_update = fabsf(update);
        if (abs_update > 0.0f) nonzero++;
        if (abs_update > max_abs) max_abs = abs_update;
        mean_abs += abs_update;
    }
    if (stats != 0) {
        stats->gradient_norm = norm;
        stats->applied_scale = scale;
        stats->max_abs_update = max_abs;
        stats->mean_abs_update = mean_abs / (float)value_count;
        stats->weight_decay = 0.0f;
        stats->nonzero_gradient_values = nonzero;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_optimizer_init(axiom_optimizer_state *state, float *m, float *v, size_t value_count) {
    if (state == 0 || m == 0 || v == 0 || value_count == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    state->m = m;
    state->v = v;
    state->value_count = value_count;
    state->step = 0;
    state->beta1 = 0.9f;
    state->beta2 = 0.999f;
    state->epsilon = 1e-8f;
    state->weight_decay = 0.01f;
    memset(m, 0, sizeof(float) * value_count);
    memset(v, 0, sizeof(float) * value_count);
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_weight_update_adam(
    float *weights,
    const float *gradient,
    size_t value_count,
    float lr,
    axiom_optimizer_state *state,
    axiom_update_stats *stats
) {
    if (state != 0) {
        float old_decay = state->weight_decay;
        state->weight_decay = 0.0f;
        int code = axiom_weight_update_adamw(weights, gradient, value_count, lr, 0.0f, state, stats);
        state->weight_decay = old_decay;
        return code;
    }
    return AXIOM_OFFLINE_INVALID_ARG;
}

extern "C" int axiom_weight_update_adamw(
    float *weights,
    const float *gradient,
    size_t value_count,
    float lr,
    float clip_norm,
    axiom_optimizer_state *state,
    axiom_update_stats *stats
) {
    if (weights == 0 || gradient == 0 || state == 0 || state->m == 0 || state->v == 0 || value_count == 0 || value_count != state->value_count || lr <= 0.0f) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (!vector_is_finite(weights, value_count) || !vector_is_finite(gradient, value_count)) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    float norm = vector_norm(gradient, value_count);
    float scale = 1.0f;
    if (clip_norm > 0.0f && norm > clip_norm) {
        scale = clip_norm / norm;
    }
    state->step++;
    float beta1 = state->beta1 > 0.0f ? state->beta1 : 0.9f;
    float beta2 = state->beta2 > 0.0f ? state->beta2 : 0.999f;
    float eps = state->epsilon > 0.0f ? state->epsilon : 1e-8f;
    float decay = state->weight_decay >= 0.0f ? state->weight_decay : 0.01f;
    float b1_corr = 1.0f - powf(beta1, (float)state->step);
    float b2_corr = 1.0f - powf(beta2, (float)state->step);
    if (b1_corr <= 0.0f || b2_corr <= 0.0f) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    size_t nonzero = 0;
    for (size_t i = 0; i < value_count; ++i) {
        float g = gradient[i] * scale;
        state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * g;
        state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * g * g;
        float mh = state->m[i] / b1_corr;
        float vh = state->v[i] / b2_corr;
        float adam = mh / (sqrtf(vh) + eps);
        float update = lr * (adam + decay * weights[i]);
        weights[i] -= update;
        float abs_update = fabsf(update);
        if (fabsf(g) > 0.0f) nonzero++;
        if (abs_update > max_abs) max_abs = abs_update;
        mean_abs += abs_update;
    }
    if (stats != 0) {
        stats->gradient_norm = norm;
        stats->applied_scale = scale;
        stats->max_abs_update = max_abs;
        stats->mean_abs_update = mean_abs / (float)value_count;
        stats->weight_decay = decay;
        stats->nonzero_gradient_values = nonzero;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_publish_weights(const char *staging_path, const char *final_path, const float *weights, size_t value_count) {
    if (staging_path == 0 || final_path == 0 || weights == 0 || value_count == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    FILE *f = fopen(staging_path, "wb");
    if (f == 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    size_t wrote = fwrite(weights, sizeof(float), value_count, f);
    int flush_error = fflush(f);
    int sync_error = AXIOM_FSYNC(fileno(f));
    int close_error = fclose(f);
    if (wrote != value_count || flush_error != 0 || sync_error != 0 || close_error != 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    if (replace_file(staging_path, final_path) != 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    return AXIOM_OFFLINE_OK;
}

static size_t json_escape(const char *src, char *dst, size_t cap) {
    size_t out = 0;
    if (cap == 0u) {
        return 0;
    }
    for (size_t i = 0; src != 0 && src[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)src[i];
        const char *esc = 0;
        char small[7];
        if (c == '\\') esc = "\\\\";
        else if (c == '"') esc = "\\\"";
        else if (c == '\n') esc = "\\n";
        else if (c == '\r') esc = "\\r";
        else if (c == '\t') esc = "\\t";
        else if (c < 0x20u) {
            snprintf(small, sizeof(small), "\\u%04x", (unsigned)c);
            esc = small;
        }
        if (esc != 0) {
            for (size_t j = 0; esc[j] != '\0'; ++j) {
                if (out + 1u >= cap) {
                    dst[out] = '\0';
                    return out;
                }
                dst[out++] = esc[j];
            }
        } else {
            if (out + 1u >= cap) {
                dst[out] = '\0';
                return out;
            }
            dst[out++] = (char)c;
        }
    }
    dst[out] = '\0';
    return out;
}

extern "C" int axiom_weights_updated_event_json(
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
) {
    if (model_name == 0 || store_path == 0 || staging_path == 0 || checksum_sha256 == 0 || run_id == 0 || out_json == 0 || out_capacity == 0u) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    char model[256], store[768], staging[768], run[128];
    json_escape(model_name, model, sizeof(model));
    json_escape(store_path, store, sizeof(store));
    json_escape(staging_path, staging, sizeof(staging));
    json_escape(run_id, run, sizeof(run));
    int n = snprintf(
        out_json,
        out_capacity,
        "{\"topic\":\"weights_updated\",\"component\":\"offline.weight_updater\","
        "\"payload\":{\"model_name\":\"%s\",\"store_path\":\"%s\",\"staging_path\":\"%s\","
        "\"checksum_sha256\":\"%s\",\"version\":%d,\"batch_count\":%d,"
        "\"gradient_steps\":%d,\"run_id\":\"%s\"}}",
        model,
        store,
        staging,
        checksum_sha256,
        version,
        batch_count,
        gradient_steps,
        run
    );
    if (n < 0 || (size_t)n >= out_capacity) {
        return AXIOM_OFFLINE_SHAPE_ERROR;
    }
    return AXIOM_OFFLINE_OK;
}

static int load_existing_weights(FILE *f, float *weights, size_t value_count) {
    if (f == 0 || weights == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    rewind(f);
    size_t got = fread(weights, sizeof(float), value_count, f);
    if (got < value_count && ferror(f)) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    for (size_t i = got; i < value_count; ++i) {
        weights[i] = 0.0f;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int weight_updater_init(const char *structural_layer_path, float lr) {
    if (structural_layer_path == 0 || structural_layer_path[0] == '\0' || lr <= 0.0f || !isfinite(lr)) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    weight_updater_shutdown();
    memset(&g_updater, 0, sizeof(g_updater));
    snprintf(g_updater.structural_layer_path, sizeof(g_updater.structural_layer_path), "%s", structural_layer_path);
    snprintf(g_updater.staging_path, sizeof(g_updater.staging_path), "%s.staging", structural_layer_path);
    g_updater.learning_rate = lr;
    g_updater.clip_norm = 1.0f;
    g_updater.row_count = AXIOM_STRUCTURAL_DEFAULT_ROWS;
    g_updater.value_count = g_updater.row_count * AXIOM_ENCODER_WIDTH;

    FILE *existing = fopen(structural_layer_path, "rb");
    if (existing != 0) {
        if (fseek(existing, 0, SEEK_END) == 0) {
            long bytes = ftell(existing);
            if (bytes > 0 && bytes % (long)sizeof(float) == 0) {
                g_updater.value_count = (size_t)bytes / sizeof(float);
                g_updater.row_count = g_updater.value_count / AXIOM_ENCODER_WIDTH;
                if (g_updater.row_count == 0u) {
                    g_updater.row_count = 1u;
                }
            }
        }
    }

    g_updater.weights = (float *)calloc(g_updater.value_count, sizeof(float));
    g_updater.m = (float *)calloc(g_updater.value_count, sizeof(float));
    g_updater.v = (float *)calloc(g_updater.value_count, sizeof(float));
    g_updater.gradient = (float *)calloc(g_updater.value_count, sizeof(float));
    if (g_updater.weights == 0 || g_updater.m == 0 || g_updater.v == 0 || g_updater.gradient == 0) {
        if (existing != 0) fclose(existing);
        weight_updater_shutdown();
        return AXIOM_OFFLINE_IO_ERROR;
    }
    if (existing != 0) {
        int code = load_existing_weights(existing, g_updater.weights, g_updater.value_count);
        fclose(existing);
        if (code != AXIOM_OFFLINE_OK) {
            weight_updater_shutdown();
            return code;
        }
    }
    int opt = axiom_optimizer_init(&g_updater.optimizer, g_updater.m, g_updater.v, g_updater.value_count);
    if (opt != AXIOM_OFFLINE_OK) {
        weight_updater_shutdown();
        return opt;
    }
    g_updater.initialized = 1;
    return AXIOM_OFFLINE_OK;
}

extern "C" int weight_updater_accumulate(const float *embeddings, const int *labels, int batch_size) {
    if (!g_updater.initialized) {
        return AXIOM_OFFLINE_NOT_READY;
    }
    if (embeddings == 0 || batch_size <= 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    size_t rows = g_updater.row_count == 0u ? 1u : g_updater.row_count;
    float scale = 1.0f / (float)batch_size;
    for (int b = 0; b < batch_size; ++b) {
        int raw_label = labels != 0 ? labels[b] : b;
        if (raw_label < 0) raw_label = -raw_label;
        size_t row = (size_t)raw_label % rows;
        size_t base = row * AXIOM_ENCODER_WIDTH;
        const float *embedding = embeddings + (size_t)b * AXIOM_ENCODER_WIDTH;
        if (!vector_is_finite(embedding, AXIOM_ENCODER_WIDTH)) {
            return AXIOM_OFFLINE_NUMERIC_ERROR;
        }
        for (size_t d = 0; d < AXIOM_ENCODER_WIDTH && base + d < g_updater.value_count; ++d) {
            g_updater.gradient[base + d] += embedding[d] * scale;
        }
    }
    g_updater.batch_count += batch_size;
    return AXIOM_OFFLINE_OK;
}

extern "C" int weight_updater_step(void) {
    if (!g_updater.initialized) {
        return AXIOM_OFFLINE_NOT_READY;
    }
    float observed = vector_norm(g_updater.gradient, g_updater.value_count);
    if (observed <= 0.0f) {
        return AXIOM_OFFLINE_NO_WORK;
    }
    int code = axiom_weight_update_adamw(
        g_updater.weights,
        g_updater.gradient,
        g_updater.value_count,
        g_updater.learning_rate,
        g_updater.clip_norm,
        &g_updater.optimizer,
        &g_updater.last_stats
    );
    if (code != AXIOM_OFFLINE_OK) {
        return code;
    }
    memset(g_updater.gradient, 0, sizeof(float) * g_updater.value_count);
    g_updater.gradient_steps++;
    g_updater.version++;
    code = axiom_publish_weights(g_updater.staging_path, g_updater.structural_layer_path, g_updater.weights, g_updater.value_count);
    if (code != AXIOM_OFFLINE_OK) {
        return code;
    }
    code = axiom_file_sha256_hex(g_updater.structural_layer_path, g_updater.last_checksum);
    return code;
}

extern "C" int weight_updater_checkpoint(void) {
    if (!g_updater.initialized) {
        return AXIOM_OFFLINE_NOT_READY;
    }
    int code = axiom_publish_weights(g_updater.staging_path, g_updater.structural_layer_path, g_updater.weights, g_updater.value_count);
    if (code != AXIOM_OFFLINE_OK) {
        return code;
    }
    return axiom_file_sha256_hex(g_updater.structural_layer_path, g_updater.last_checksum);
}

extern "C" void weight_updater_shutdown(void) {
    free(g_updater.weights);
    free(g_updater.m);
    free(g_updater.v);
    free(g_updater.gradient);
    memset(&g_updater, 0, sizeof(g_updater));
}

extern "C" int axiom_weight_kernel_symbols_present(void) {
#if defined(__CUDACC__)
    return 1;
#else
    return 0;
#endif
}
