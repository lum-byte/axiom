#include "offline_api.h"

#include <math.h>
#include <stddef.h>
#include <string.h>

static int finite_vector(const float *values, size_t count) {
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

static float vector_norm(const float *values, size_t count) {
    float norm = 0.0f;
    for (size_t i = 0; i < count; ++i) {
        norm += values[i] * values[i];
    }
    return sqrtf(norm);
}

#if defined(__CUDACC__)
__global__ void axiom_accumulate_kernel(float *sum_buffer, const float *new_grad, int n_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_elements) {
        sum_buffer[idx] += new_grad[idx];
    }
}

__global__ void axiom_normalize_gradient_kernel(float *grad, int n_steps, int n_elements) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n_elements && n_steps > 0) {
        grad[idx] /= (float)n_steps;
    }
}

__global__ void axiom_ring_replace_kernel(
    float *sum_buffer,
    float *ring_buffer,
    const float *new_grad,
    int cursor,
    int value_count,
    int ring_full
) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= value_count) {
        return;
    }
    float *slot = ring_buffer + (size_t)cursor * (size_t)value_count;
    if (ring_full) {
        sum_buffer[idx] -= slot[idx];
    }
    slot[idx] = new_grad[idx];
    sum_buffer[idx] += new_grad[idx];
}
#endif

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
    if (!finite_vector(gradient, value_count)) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    float norm = vector_norm(gradient, value_count);
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
    if (!finite_vector(gradient, value_count)) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    for (size_t i = 0; i < value_count; ++i) {
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

extern "C" int axiom_accumulator_init(
    axiom_accumulator_state *state,
    float *sum_storage,
    float *ring_storage,
    size_t value_count,
    size_t accumulate_steps
) {
    if (state == 0 || sum_storage == 0 || ring_storage == 0 || value_count == 0 || accumulate_steps == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    state->sum_buffer = sum_storage;
    state->ring_buffer = ring_storage;
    state->value_count = value_count;
    state->accumulate_steps = accumulate_steps;
    state->step = 0;
    state->cursor = 0;
    state->flushed_batches = 0;
    state->last_observed_norm = 0.0f;
    state->max_observed_norm = 0.0f;
    memset(sum_storage, 0, sizeof(float) * value_count);
    memset(ring_storage, 0, sizeof(float) * value_count * accumulate_steps);
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_accumulator_add(axiom_accumulator_state *state, const float *gradient, size_t value_count) {
    if (state == 0 || state->sum_buffer == 0 || state->ring_buffer == 0 || gradient == 0 || value_count != state->value_count) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (!finite_vector(gradient, value_count)) {
        return AXIOM_OFFLINE_NUMERIC_ERROR;
    }
    float *slot = state->ring_buffer + state->cursor * state->value_count;
    if (state->step >= state->accumulate_steps) {
        for (size_t i = 0; i < value_count; ++i) {
            state->sum_buffer[i] -= slot[i];
        }
    }
    memcpy(slot, gradient, sizeof(float) * value_count);
    for (size_t i = 0; i < value_count; ++i) {
        state->sum_buffer[i] += gradient[i];
    }
    state->last_observed_norm = vector_norm(gradient, value_count);
    if (state->last_observed_norm > state->max_observed_norm) {
        state->max_observed_norm = state->last_observed_norm;
    }
    state->cursor = (state->cursor + 1u) % state->accumulate_steps;
    state->step++;
    return axiom_accumulator_ready(state) ? AXIOM_OFFLINE_OK : AXIOM_OFFLINE_NO_WORK;
}

extern "C" int axiom_accumulator_ready(const axiom_accumulator_state *state) {
    if (state == 0 || state->sum_buffer == 0 || state->accumulate_steps == 0) {
        return 0;
    }
    return state->step >= state->accumulate_steps ? 1 : 0;
}

extern "C" int axiom_accumulator_flush(axiom_accumulator_state *state, float *output, size_t value_count) {
    if (state == 0 || state->sum_buffer == 0 || output == 0 || value_count != state->value_count) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    size_t divisor = state->step < state->accumulate_steps ? state->step : state->accumulate_steps;
    if (divisor == 0) {
        return AXIOM_OFFLINE_NO_WORK;
    }
    for (size_t i = 0; i < value_count; ++i) {
        output[i] = state->sum_buffer[i] / (float)divisor;
    }
    memset(state->sum_buffer, 0, sizeof(float) * state->value_count);
    memset(state->ring_buffer, 0, sizeof(float) * state->value_count * state->accumulate_steps);
    state->step = 0;
    state->cursor = 0;
    state->flushed_batches++;
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_accumulator_reset(axiom_accumulator_state *state) {
    if (state == 0 || state->sum_buffer == 0 || state->ring_buffer == 0 || state->value_count == 0 || state->accumulate_steps == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    memset(state->sum_buffer, 0, sizeof(float) * state->value_count);
    memset(state->ring_buffer, 0, sizeof(float) * state->value_count * state->accumulate_steps);
    state->step = 0;
    state->cursor = 0;
    state->last_observed_norm = 0.0f;
    return AXIOM_OFFLINE_OK;
}

extern "C" int axiom_accumulator_kernel_symbols_present(void) {
#if defined(__CUDACC__)
    return 1;
#else
    return 0;
#endif
}
