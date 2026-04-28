#include "offline_api.h"

#include <math.h>
#include <string.h>

extern "C" int axiom_gradient_init(axiom_gradient_buffer *buffer, float *storage, size_t value_count, size_t target_steps) {
    if (buffer == 0 || storage == 0 || value_count == 0 || target_steps == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    buffer->values = storage;
    buffer->value_count = value_count;
    buffer->accumulated_steps = 0;
    buffer->target_steps = target_steps;
    memset(storage, 0, sizeof(float) * value_count);
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_gradient_ready(const axiom_gradient_buffer *buffer) {
    if (buffer == 0 || buffer->values == 0 || buffer->target_steps == 0) {
        return 0;
    }
    return buffer->accumulated_steps >= buffer->target_steps ? 1 : 0;
}

extern "C" int axiom_gradient_reset(axiom_gradient_buffer *buffer) {
    if (buffer == 0 || buffer->values == 0 || buffer->value_count == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    memset(buffer->values, 0, sizeof(float) * buffer->value_count);
    buffer->accumulated_steps = 0;
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_gradient_clip(float *gradient, size_t value_count, float clip_norm, float *observed_norm) {
    if (gradient == 0 || value_count == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    float norm = 0.0f;
    for (size_t i = 0; i < value_count; ++i) {
        if (!isfinite(gradient[i])) {
            return AXIOM_OFFLINE_NUMERIC_ERROR;
        }
        norm += gradient[i] * gradient[i];
    }
    norm = sqrtf(norm);
    if (observed_norm != 0) {
        *observed_norm = norm;
    }
    if (clip_norm <= 0.0f || norm <= clip_norm || norm <= 0.0f) {
        return AXIOM_OFFLINE_OK;
    }
    float scale = clip_norm / norm;
    for (size_t i = 0; i < value_count; ++i) {
        gradient[i] *= scale;
    }
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_gradient_accumulate(axiom_gradient_buffer *buffer, const float *gradient, size_t value_count) {
    if (buffer == 0 || gradient == 0 || buffer->values == 0 || value_count != buffer->value_count) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    for (size_t i = 0; i < value_count; ++i) {
        if (!isfinite(gradient[i])) {
            return AXIOM_OFFLINE_NUMERIC_ERROR;
        }
        buffer->values[i] += gradient[i];
    }
    buffer->accumulated_steps++;
    if (buffer->accumulated_steps >= buffer->target_steps) {
        for (size_t i = 0; i < value_count; ++i) {
            buffer->values[i] /= (float)buffer->accumulated_steps;
        }
        buffer->accumulated_steps = 0;
        return AXIOM_OFFLINE_OK;
    }
    return AXIOM_OFFLINE_NO_WORK;
}
