#include "offline_api.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#include <sys/stat.h>
#define MKDIR(path) mkdir(path, 0777)
#endif

#define ASSERT_TRUE(x) do { if (!(x)) { printf("assert failed: %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ASSERT_EQ(a,b) do { if ((a)!=(b)) { printf("assert eq failed: %s:%d: %s == %s\n", __FILE__, __LINE__, #a, #b); return 1; } } while (0)
#define ASSERT_NEAR(a,b,eps) do { float _aa=(float)(a); float _bb=(float)(b); if (fabsf(_aa-_bb)>(eps)) { printf("assert near failed: %s:%d: %.9f != %.9f\n", __FILE__, __LINE__, _aa, _bb); return 1; } } while (0)

static int write_text_file(const char *path, const char *body) {
    FILE *f = fopen(path, "wb");
    if (!f) return 0;
    fwrite(body, 1u, strlen(body), f);
    fclose(f);
    return 1;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    fclose(f);
    return 1;
}

static float norm256(const float *row) {
    float n = 0.0f;
    for (size_t i = 0; i < AXIOM_ENCODER_WIDTH; ++i) {
        n += row[i] * row[i];
    }
    return sqrtf(n);
}

static int all_zero(const float *row) {
    for (size_t i = 0; i < AXIOM_ENCODER_WIDTH; ++i) {
        if (row[i] != 0.0f) return 0;
    }
    return 1;
}

static int test_feature_encoder(void) {
    axiom_zone_feature f[1] = {{0.5f, 10.0f, 100.0f, 2.0f}};
    float out[256];
    ASSERT_EQ(axiom_gpu_encode(f, 1, out, 256), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(out[0] > 0.0f);
    ASSERT_NEAR(norm256(out), 1.0f, 0.001f);
    ASSERT_EQ(axiom_gpu_encode(f, 1, out, 8), AXIOM_OFFLINE_SHAPE_ERROR);
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
    ASSERT_NEAR(norm256(out), 1.0f, 0.001f);
    ASSERT_NEAR(norm256(out + 256), 1.0f, 0.001f);
    return 0;
}

static int test_text_encoder_deterministic_and_empty(void) {
    const char *texts[3] = {
        "AlphaFold RNA tertiary structure code block tensor update",
        "AlphaFold RNA tertiary structure code block tensor update",
        "",
    };
    float out[3 * AXIOM_ENCODER_WIDTH];
    gpu_encoder_shutdown();
    ASSERT_EQ(gpu_encoder_init("offline_test_store/structural_layer.pt"), AXIOM_OFFLINE_OK);
    ASSERT_EQ(gpu_encoder_health(), 1);
    ASSERT_EQ(gpu_encoder_encode_batch(texts, 3, out, 512), AXIOM_OFFLINE_OK);
    for (size_t i = 0; i < AXIOM_ENCODER_WIDTH; ++i) {
        ASSERT_NEAR(out[i], out[AXIOM_ENCODER_WIDTH + i], 0.000001f);
    }
    ASSERT_NEAR(norm256(out), 1.0f, 0.001f);
    ASSERT_TRUE(all_zero(out + 2 * AXIOM_ENCODER_WIDTH));
    axiom_encoder_stats stats;
    ASSERT_EQ(gpu_encoder_get_stats(&stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(stats.batches_encoded == 1u);
    ASSERT_TRUE(stats.zones_encoded == 3u);
    ASSERT_TRUE(stats.empty_inputs == 1u);
    gpu_encoder_shutdown();
    return 0;
}

static int test_text_encoder_chunking(void) {
    enum { N = AXIOM_ENCODER_MAX_CHUNK + 3 };
    const char **texts = (const char **)calloc((size_t)N, sizeof(char *));
    float *out = (float *)calloc((size_t)N * AXIOM_ENCODER_WIDTH, sizeof(float));
    ASSERT_TRUE(texts != 0 && out != 0);
    for (int i = 0; i < N; ++i) texts[i] = "chunked signal zone with repeated topology text";
    ASSERT_EQ(gpu_encoder_init("offline_test_store/structural_layer.pt"), AXIOM_OFFLINE_OK);
    ASSERT_EQ(gpu_encoder_reset_stats(), AXIOM_OFFLINE_OK);
    ASSERT_EQ(gpu_encoder_encode_batch(texts, N, out, 512), AXIOM_OFFLINE_OK);
    axiom_encoder_stats stats;
    ASSERT_EQ(gpu_encoder_get_stats(&stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(stats.chunks_encoded >= 2u);
    ASSERT_NEAR(norm256(out), 1.0f, 0.001f);
    free(out);
    free(texts);
    gpu_encoder_shutdown();
    return 0;
}

static int test_gpu_kernel_symbol_reporting_and_chunk_estimate(void) {
    int symbols = axiom_gpu_kernel_symbols_present();
    ASSERT_TRUE(symbols == 0 || symbols == 1);
    ASSERT_TRUE(axiom_accumulator_kernel_symbols_present() == symbols);
    ASSERT_TRUE(axiom_weight_kernel_symbols_present() == symbols);
    ASSERT_EQ(axiom_gpu_estimate_chunks(0, 512), 0);
    ASSERT_EQ(axiom_gpu_estimate_chunks(1, 512), 1);
    ASSERT_EQ(axiom_gpu_estimate_chunks(512, 512), 1);
    ASSERT_EQ(axiom_gpu_estimate_chunks(513, 512), 2);
    ASSERT_EQ(axiom_gpu_estimate_chunks(-1, 512), AXIOM_OFFLINE_INVALID_ARG);
    return 0;
}

static int test_gradient_buffer(void) {
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

static int test_gradient_clip(void) {
    float grad[2] = {3.0f, 4.0f};
    float observed = 0.0f;
    ASSERT_EQ(axiom_gradient_clip(grad, 2, 1.0f, &observed), AXIOM_OFFLINE_OK);
    ASSERT_NEAR(observed, 5.0f, 0.001f);
    ASSERT_NEAR(sqrtf(grad[0] * grad[0] + grad[1] * grad[1]), 1.0f, 0.001f);
    return 0;
}

static int test_accumulator_state(void) {
    float sum[4];
    float ring[12];
    float out[4];
    float a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float b[4] = {2.0f, 3.0f, 4.0f, 5.0f};
    float c[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    axiom_accumulator_state state;
    ASSERT_EQ(axiom_accumulator_init(&state, sum, ring, 4, 3), AXIOM_OFFLINE_OK);
    ASSERT_EQ(axiom_accumulator_add(&state, a, 4), AXIOM_OFFLINE_NO_WORK);
    ASSERT_EQ(axiom_accumulator_add(&state, b, 4), AXIOM_OFFLINE_NO_WORK);
    ASSERT_EQ(axiom_accumulator_add(&state, c, 4), AXIOM_OFFLINE_OK);
    ASSERT_EQ(axiom_accumulator_ready(&state), 1);
    ASSERT_EQ(axiom_accumulator_flush(&state, out, 4), AXIOM_OFFLINE_OK);
    ASSERT_NEAR(out[0], 2.0f, 0.001f);
    ASSERT_NEAR(out[3], 5.0f, 0.001f);
    ASSERT_EQ(axiom_accumulator_ready(&state), 0);
    return 0;
}

static int test_weight_update(void) {
    float weights[2] = {1.0f, 1.0f};
    float grad[2] = {0.5f, 0.5f};
    axiom_update_stats stats;
    ASSERT_EQ(axiom_weight_update_with_stats(weights, grad, 2, 0.1f, 10.0f, &stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(weights[0] < 1.0f);
    ASSERT_TRUE(stats.gradient_norm > 0.0f);
    ASSERT_TRUE(stats.nonzero_gradient_values == 2u);
    return 0;
}

static int test_adamw_update(void) {
    float weights[2] = {1.0f, 1.0f};
    float grad[2] = {0.25f, -0.25f};
    float m[2];
    float v[2];
    axiom_optimizer_state state;
    axiom_update_stats stats;
    ASSERT_EQ(axiom_optimizer_init(&state, m, v, 2), AXIOM_OFFLINE_OK);
    state.weight_decay = 0.01f;
    ASSERT_EQ(axiom_weight_update_adamw(weights, grad, 2, 0.01f, 1.0f, &state, &stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(state.step == 1u);
    ASSERT_TRUE(stats.max_abs_update > 0.0f);
    ASSERT_TRUE(stats.weight_decay > 0.0f);
    return 0;
}

static int test_sha256_and_crc(void) {
    char digest[65];
    ASSERT_EQ(axiom_sha256_hex((const uint8_t *)"abc", 3, digest), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(strcmp(digest, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad") == 0);
    ASSERT_EQ(axiom_offline_crc32((const uint8_t *)"123456789", 9), 0xcbf43926u);
    return 0;
}

static int test_publish_and_file_hash(void) {
    float weights[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    char digest[65];
    remove("offline_publish.bin");
    remove("offline_publish.bin.staging");
    ASSERT_EQ(axiom_publish_weights("offline_publish.bin.staging", "offline_publish.bin", weights, 4), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(file_exists("offline_publish.bin"));
    ASSERT_EQ(axiom_file_sha256_hex("offline_publish.bin", digest), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(strlen(digest) == 64u);
    remove("offline_publish.bin");
    return 0;
}

static int test_weight_updater_runtime(void) {
    MKDIR("offline_test_store");
    remove("offline_test_store/structural_layer.pt");
    remove("offline_test_store/structural_layer.pt.staging");
    const char *texts[2] = {"one signal zone", "second signal zone with code"};
    float embeddings[2 * AXIOM_ENCODER_WIDTH];
    int labels[2] = {3, 4};
    ASSERT_EQ(gpu_encoder_init("offline_test_store/structural_layer.pt"), AXIOM_OFFLINE_OK);
    ASSERT_EQ(gpu_encoder_encode_batch(texts, 2, embeddings, 512), AXIOM_OFFLINE_OK);
    ASSERT_EQ(weight_updater_init("offline_test_store/structural_layer.pt", 0.001f), AXIOM_OFFLINE_OK);
    ASSERT_EQ(weight_updater_accumulate(embeddings, labels, 2), AXIOM_OFFLINE_OK);
    ASSERT_EQ(weight_updater_step(), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(file_exists("offline_test_store/structural_layer.pt"));
    weight_updater_shutdown();
    gpu_encoder_shutdown();
    return 0;
}

static int test_event_json_builder(void) {
    char json[AXIOM_OFFLINE_EVENT_JSON_CAP];
    const char *digest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";
    ASSERT_EQ(
        axiom_weights_updated_event_json(
            "structural_layer",
            "store/structural_layer.pt",
            "store/staging/structural_layer.pt.staging",
            digest,
            2,
            3,
            4,
            "00000000-0000-4000-8000-000000000000",
            json,
            sizeof(json)
        ),
        AXIOM_OFFLINE_OK
    );
    ASSERT_TRUE(strstr(json, "\"topic\":\"weights_updated\"") != 0);
    ASSERT_TRUE(strstr(json, digest) != 0);
    return 0;
}

static int test_scheduler_feature_run_once(void) {
    axiom_zone_feature f[2] = {{0.01f, 1.0f, 10.0f, 1.0f}, {0.5f, 10.0f, 100.0f, 2.0f}};
    float out[256];
    axiom_batch_scheduler scheduler;
    ASSERT_EQ(axiom_scheduler_init(&scheduler, 4, 0.1f), AXIOM_OFFLINE_OK);
    ASSERT_EQ(axiom_scheduler_run_once(f, 2, out, 256, &scheduler), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(scheduler.accepted == 1u);
    ASSERT_TRUE(scheduler.rejected == 1u);
    return 0;
}

static int test_scheduler_parse_and_queue(void) {
    MKDIR("offline_test_store");
    write_text_file("offline_queue_input.txt", "<article>AXIOM offline queue signal</article>");
    write_text_file(
        "offline_queue.jsonl",
        "{\"run_id\":\"00000000-0000-4000-8000-000000000000\",\"url\":\"https://example.com/a\",\"topology_class\":\"DOCS\",\"input_path\":\"offline_queue_input.txt\",\"label\":7,\"density\":0.8}\n"
        "{\"url\":\"https://example.com/b\",\"topology_class\":\"DOCS\",\"input_path\":\"offline_queue_input.txt\",\"label\":8,\"density\":0.9}\n"
    );
    remove("offline_dead.jsonl");
    axiom_offline_work_item item;
    ASSERT_EQ(axiom_scheduler_parse_work_line("{\"input_path\":\"offline_queue_input.txt\",\"density\":0.4}", &item), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(strcmp(item.input_path, "offline_queue_input.txt") == 0);
    axiom_scheduler_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.queue_path = "offline_queue.jsonl";
    opts.store_dir = "offline_test_store";
    opts.dead_letter_path = "offline_dead.jsonl";
    opts.accumulate_steps = 2;
    opts.learning_rate = 0.001f;
    opts.min_density = 0.1f;
    axiom_scheduler_stats stats;
    ASSERT_EQ(axiom_scheduler_run_queue(&opts, &stats), AXIOM_OFFLINE_OK);
    ASSERT_TRUE(stats.lines_read == 2u);
    ASSERT_TRUE(stats.encoded_items == 2u);
    ASSERT_TRUE(stats.update_steps >= 1u);
    ASSERT_TRUE(file_exists("offline_test_store/structural_layer.pt"));
    remove("offline_queue_input.txt");
    remove("offline_queue.jsonl");
    remove("offline_dead.jsonl");
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_feature_encoder();
    failures += test_configured_encoder_and_normalize();
    failures += test_text_encoder_deterministic_and_empty();
    failures += test_text_encoder_chunking();
    failures += test_gpu_kernel_symbol_reporting_and_chunk_estimate();
    failures += test_gradient_buffer();
    failures += test_gradient_clip();
    failures += test_accumulator_state();
    failures += test_weight_update();
    failures += test_adamw_update();
    failures += test_sha256_and_crc();
    failures += test_publish_and_file_hash();
    failures += test_weight_updater_runtime();
    failures += test_event_json_builder();
    failures += test_scheduler_feature_run_once();
    failures += test_scheduler_parse_and_queue();
    if (failures == 0) {
        printf("offline tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
