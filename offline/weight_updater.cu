#include "offline_api.h"

#include <math.h>
#include <stdio.h>

extern "C" uint32_t axiom_offline_crc32(const uint8_t *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int)(crc & 1u);
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

extern "C" int axiom_weight_update(float *weights, const float *gradient, size_t value_count, float lr, float clip_norm) {
    return axiom_weight_update_with_stats(weights, gradient, value_count, lr, clip_norm, 0);
}

extern "C" int axiom_weight_update_with_stats(float *weights, const float *gradient, size_t value_count, float lr, float clip_norm, axiom_update_stats *stats) {
    if (weights == 0 || gradient == 0 || value_count == 0 || lr <= 0.0f) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    float norm = 0.0f;
    for (size_t i = 0; i < value_count; ++i) {
        if (!isfinite(gradient[i]) || !isfinite(weights[i])) {
            return AXIOM_OFFLINE_NUMERIC_ERROR;
        }
        norm += gradient[i] * gradient[i];
    }
    norm = sqrtf(norm);
    float scale = 1.0f;
    if (clip_norm > 0.0f && norm > clip_norm) {
        scale = clip_norm / norm;
    }
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (size_t i = 0; i < value_count; ++i) {
        float update = lr * gradient[i] * scale;
        weights[i] -= update;
        float abs_update = fabsf(update);
        if (abs_update > max_abs) {
            max_abs = abs_update;
        }
        mean_abs += abs_update;
    }
    if (stats != 0) {
        stats->gradient_norm = norm;
        stats->applied_scale = scale;
        stats->max_abs_update = max_abs;
        stats->mean_abs_update = mean_abs / (float)value_count;
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
    for (size_t i = 0; i < value_count; ++i) {
        m[i] = 0.0f;
        v[i] = 0.0f;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_weight_update_adam(float *weights, const float *gradient, size_t value_count, float lr, axiom_optimizer_state *state, axiom_update_stats *stats) {
    if (weights == 0 || gradient == 0 || state == 0 || state->m == 0 || state->v == 0 || value_count == 0 || value_count != state->value_count || lr <= 0.0f) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    state->step++;
    float beta1 = state->beta1 > 0.0f ? state->beta1 : 0.9f;
    float beta2 = state->beta2 > 0.0f ? state->beta2 : 0.999f;
    float eps = state->epsilon > 0.0f ? state->epsilon : 1e-8f;
    float norm = 0.0f;
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    float b1_corr = 1.0f - powf(beta1, (float)state->step);
    float b2_corr = 1.0f - powf(beta2, (float)state->step);
    if (b1_corr <= 0.0f || b2_corr <= 0.0f) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    for (size_t i = 0; i < value_count; ++i) {
        float g = gradient[i];
        if (!isfinite(g) || !isfinite(weights[i])) {
            return AXIOM_OFFLINE_NUMERIC_ERROR;
        }
        norm += g * g;
        state->m[i] = beta1 * state->m[i] + (1.0f - beta1) * g;
        state->v[i] = beta2 * state->v[i] + (1.0f - beta2) * g * g;
        float mh = state->m[i] / b1_corr;
        float vh = state->v[i] / b2_corr;
        float update = lr * mh / (sqrtf(vh) + eps);
        weights[i] -= update;
        float abs_update = fabsf(update);
        if (abs_update > max_abs) {
            max_abs = abs_update;
        }
        mean_abs += abs_update;
    }
    if (stats != 0) {
        stats->gradient_norm = sqrtf(norm);
        stats->applied_scale = 1.0f;
        stats->max_abs_update = max_abs;
        stats->mean_abs_update = mean_abs / (float)value_count;
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
    if (fclose(f) != 0 || wrote != value_count) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    if (rename(staging_path, final_path) != 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    return AXIOM_OFFLINE_OK;
}
