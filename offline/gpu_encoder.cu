#include "offline_api.h"

#include <math.h>
#include <string.h>

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
    if (v < 0.0f) {
        v = 0.0f;
    }
    if (scale <= 0.0f) {
        scale = 1.0f;
    }
    return log1pf(v) / scale;
}

static void normalize_row(float *row, size_t width) {
    float norm = 0.0f;
    for (size_t i = 0; i < width; ++i) {
        norm += row[i] * row[i];
    }
    norm = sqrtf(norm);
    if (norm <= 0.0f || !isfinite(norm)) {
        return;
    }
    for (size_t i = 0; i < width; ++i) {
        row[i] /= norm;
    }
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
