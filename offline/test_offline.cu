#include "offline_api.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_TRUE(x) do { if (!(x)) { printf("assert failed: %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ASSERT_EQ(a,b) do { if ((a)!=(b)) { printf("assert eq failed: %s:%d\n", __FILE__, __LINE__); return 1; } } while (0)

static int test_encoder(void) {
    axiom_zone_feature f[1] = {{0.5f, 10.0f, 100.0f, 2.0f}};
    float out[256];
    ASSERT_EQ(axiom_gpu_encode(f, 1, out, 256), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(out[0] > 0.0f);
    return 0;
}

static int test_configured_encoder_and_normalize(void) {
    axiom_zone_feature f[2] = {{2.0f, -1.0f, 100.0f, 2.0f}, {0.25f, 10.0f, 1000.0f, 3.0f}};
    ASSERT_EQ(axiom_feature_normalize(f, 2), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(f[0].density == 1.0f);
    ASSERT_TRUE(f[0].token_count == 0.0f);
    float out[512];
    axiom_encoder_config cfg = {1.0f, 8.0f, 12.0f, 4.0f, 0.01f, 1};
    ASSERT_EQ(axiom_gpu_encode_configured(f, 2, &cfg, out, 512), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(out[0] > 0.0f);
    return 0;
}

static int test_accumulator(void) {
    float storage[2];
    float grad[2] = {1.0f, 3.0f};
    axiom_gradient_buffer b;
    ASSERT_EQ(axiom_gradient_init(&b, storage, 2, 2), AXIOM_OFFLINE_OK);
    ASSERT_EQ(axiom_gradient_accumulate(&b, grad, 2), AXIOM_OFFLINE_NO_WORK);
    ASSERT_EQ(axiom_gradient_accumulate(&b, grad, 2), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(storage[0] == 1.0f && storage[1] == 3.0f);
    ASSERT_EQ(axiom_gradient_reset(&b), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(storage[0] == 0.0f && storage[1] == 0.0f);
    return 0;
}

static int test_weight_update(void) {
    float weights[2] = {1.0f, 1.0f};
    float grad[2] = {0.5f, 0.5f};
    axiom_update_stats stats;
    ASSERT_EQ(axiom_weight_update_with_stats(weights, grad, 2, 0.1f, 10.0f, &stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(weights[0] < 1.0f);
    ASSERT_TRUE(stats.gradient_norm > 0.0f);
    return 0;
}

static int test_adam_update(void) {
    float weights[2] = {1.0f, 1.0f};
    float grad[2] = {0.25f, -0.25f};
    float m[2];
    float v[2];
    axiom_optimizer_state state;
    axiom_update_stats stats;
    ASSERT_EQ(axiom_optimizer_init(&state, m, v, 2), AXIOM_OFFLINE_OK);
    ASSERT_EQ(axiom_weight_update_adam(weights, grad, 2, 0.01f, &state, &stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(state.step == 1u);
    ASSERT_TRUE(stats.max_abs_update > 0.0f);
    return 0;
}

extern "C" int axiom_scheduler_init(axiom_batch_scheduler *stats, size_t max_batch, float min_density);
extern "C" int axiom_scheduler_run_once(const axiom_zone_feature *features, size_t count, float *encoded, size_t encoded_count, axiom_batch_scheduler *stats);

static int test_scheduler(void) {
    axiom_zone_feature f[2] = {{0.01f, 1.0f, 10.0f, 1.0f}, {0.5f, 10.0f, 100.0f, 2.0f}};
    float out[256];
    axiom_batch_scheduler scheduler;
    ASSERT_EQ(axiom_scheduler_init(&scheduler, 4, 0.1f), AXIOM_OFFLINE_OK);
    ASSERT_EQ(axiom_scheduler_run_once(f, 2, out, 256, &scheduler), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(scheduler.accepted == 1u);
    ASSERT_TRUE(scheduler.rejected == 1u);
    return 0;
}

static int test_crc(void) {
    ASSERT_EQ(axiom_offline_crc32((const uint8_t *)"123456789", 9), 0xcbf43926u);
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_encoder();
    failures += test_configured_encoder_and_normalize();
    failures += test_accumulator();
    failures += test_weight_update();
    failures += test_adam_update();
    failures += test_scheduler();
    failures += test_crc();
    if (failures == 0) {
        printf("offline tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
