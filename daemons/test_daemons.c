#define AXIOM_DAEMON_TEST 1
#include "daemon_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <sys/utime.h>
#define AXIOM_UTIME _utime
#define AXIOM_UTIMBUF struct _utimbuf
#else
#include <utime.h>
#define AXIOM_UTIME utime
#define AXIOM_UTIMBUF struct utimbuf
#endif

#define ASSERT_TRUE(x) do { if (!(x)) { printf("assert failed: %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ASSERT_EQ(a,b) do { long long _aa=(long long)(a); long long _bb=(long long)(b); if (_aa!=_bb) { printf("assert eq failed: %s:%d: %s=%lld %s=%lld\n", __FILE__, __LINE__, #a, _aa, #b, _bb); return 1; } } while (0)
#define ASSERT_NEAR(a,b,eps) do { double _aa=(double)(a); double _bb=(double)(b); if (fabs(_aa-_bb)>(eps)) { printf("assert near failed: %s:%d: %.9f != %.9f\n", __FILE__, __LINE__, _aa, _bb); return 1; } } while (0)

static int write_bytes(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "wb");
    if (f == 0) return 0;
    if (len > 0u && fwrite(data, 1u, len, f) != len) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int append_bytes(const char *path, const void *data, size_t len) {
    FILE *f = fopen(path, "ab");
    if (f == 0) return 0;
    if (len > 0u && fwrite(data, 1u, len, f) != len) {
        fclose(f);
        return 0;
    }
    fclose(f);
    return 1;
}

static int file_exists(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == 0) return 0;
    fclose(f);
    return 1;
}

static void remove_test_files(void) {
    remove("daemon_phase_slots.mmap");
    remove("daemon_phase_slots.mmap.staging");
    remove("daemon_store_file.bin");
    remove("daemon_store_file.bin.staging");
    remove("daemon_store_missing.bin");
    remove("daemon_store_log.jsonl");
    remove("daemon_pid.txt");
}

static axiom_phase_slot phase_slot(uint8_t phase, float confidence, uint32_t observations, float surprise, float npi, float yield) {
    axiom_phase_slot slot;
    memset(&slot, 0, sizeof(slot));
    slot.phase = phase;
    slot.flags = 0;
    slot.generation = 1;
    slot.confidence = confidence;
    slot.observation_count = observations;
    slot.surprise_rate = surprise;
    slot.last_updated_unix = 1000u;
    slot.npi = npi;
    slot.recipe_yield = yield;
    return slot;
}

static int test_phase_slot_is_compatible_32_bytes(void) {
    ASSERT_EQ(sizeof(axiom_phase_slot), 32u);
    axiom_phase_slot slot = phase_slot(AXIOM_PHASE_LEARNING, 0.8f, 77u, 0.02f, 0.6f, 0.02f);
    unsigned char *raw = (unsigned char *)&slot;
    ASSERT_EQ(raw[0], AXIOM_PHASE_LEARNING);
    ASSERT_EQ(raw[1], 0);
    uint64_t ts = 0;
    memcpy(&ts, raw + 16, sizeof(ts));
    ASSERT_EQ(ts, 1000u);
    float npi = 0.0f;
    float recipe = 0.0f;
    memcpy(&npi, raw + 24, sizeof(npi));
    memcpy(&recipe, raw + 28, sizeof(recipe));
    ASSERT_NEAR(npi, 0.6f, 0.0001);
    ASSERT_NEAR(recipe, 0.02f, 0.0001);
    return 0;
}

static int test_phase_cold_to_learning_exact_threshold(void) {
    axiom_phase_slot slot = phase_slot(AXIOM_PHASE_COLD, 0.70f, 50u, 0.199f, 0.0f, 0.0f);
    axiom_phase_transition transition;
    ASSERT_EQ(axiom_phase_apply_policy(&slot, 0, "NEWS_ARTICLE", &transition), AXIOM_DAEMON_CHANGED);
    ASSERT_EQ(slot.phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(transition.from_phase, AXIOM_PHASE_COLD);
    ASSERT_EQ(transition.to_phase, AXIOM_PHASE_LEARNING);
    ASSERT_TRUE(strcmp(transition.reason, "confidence_threshold_reached") == 0);
    ASSERT_TRUE(slot.generation == 2u);
    return 0;
}

static int test_phase_cold_does_not_promote_when_each_criterion_fails(void) {
    axiom_phase_slot low_obs = phase_slot(AXIOM_PHASE_COLD, 0.70f, 49u, 0.01f, 0.0f, 0.0f);
    axiom_phase_slot low_conf = phase_slot(AXIOM_PHASE_COLD, 0.699f, 50u, 0.01f, 0.0f, 0.0f);
    axiom_phase_slot high_surprise = phase_slot(AXIOM_PHASE_COLD, 0.70f, 50u, 0.20f, 0.0f, 0.0f);
    ASSERT_EQ(axiom_phase_apply_policy(&low_obs, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_phase_apply_policy(&low_conf, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_phase_apply_policy(&high_surprise, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(low_obs.phase, AXIOM_PHASE_COLD);
    ASSERT_EQ(low_conf.phase, AXIOM_PHASE_COLD);
    ASSERT_EQ(high_surprise.phase, AXIOM_PHASE_COLD);
    return 0;
}

static int test_phase_learning_to_known_exact_threshold(void) {
    axiom_phase_slot slot = phase_slot(AXIOM_PHASE_LEARNING, 0.85f, 200u, 0.049f, 0.70f, 0.005f);
    axiom_phase_transition transition;
    ASSERT_EQ(axiom_phase_apply_policy(&slot, 0, "SCIENTIFIC_PAPER", &transition), AXIOM_DAEMON_CHANGED);
    ASSERT_EQ(slot.phase, AXIOM_PHASE_KNOWN);
    ASSERT_EQ(transition.from_phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(transition.to_phase, AXIOM_PHASE_KNOWN);
    ASSERT_TRUE(strcmp(transition.reason, "confidence_threshold_reached") == 0);
    return 0;
}

static int test_phase_learning_blocks_single_failed_known_criterion(void) {
    axiom_phase_slot low_obs = phase_slot(AXIOM_PHASE_LEARNING, 0.85f, 199u, 0.01f, 0.70f, 0.005f);
    axiom_phase_slot low_conf = phase_slot(AXIOM_PHASE_LEARNING, 0.849f, 200u, 0.01f, 0.70f, 0.005f);
    axiom_phase_slot high_surprise = phase_slot(AXIOM_PHASE_LEARNING, 0.85f, 200u, 0.05f, 0.70f, 0.005f);
    axiom_phase_slot low_npi = phase_slot(AXIOM_PHASE_LEARNING, 0.85f, 200u, 0.01f, 0.699f, 0.005f);
    axiom_phase_slot low_recipe = phase_slot(AXIOM_PHASE_LEARNING, 0.85f, 200u, 0.01f, 0.70f, 0.0049f);
    ASSERT_EQ(axiom_phase_apply_policy(&low_obs, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_phase_apply_policy(&low_conf, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_phase_apply_policy(&high_surprise, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_phase_apply_policy(&low_npi, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_phase_apply_policy(&low_recipe, 0, "A", 0), AXIOM_DAEMON_OK);
    ASSERT_EQ(low_obs.phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(low_conf.phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(high_surprise.phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(low_npi.phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(low_recipe.phase, AXIOM_PHASE_LEARNING);
    return 0;
}

static int test_phase_known_regresses_on_surprise(void) {
    axiom_phase_slot slot = phase_slot(AXIOM_PHASE_KNOWN, 0.95f, 400u, 0.36f, 0.9f, 0.03f);
    axiom_phase_transition transition;
    ASSERT_EQ(axiom_phase_apply_policy(&slot, 0, "SHOP_PRODUCT", &transition), AXIOM_DAEMON_CHANGED);
    ASSERT_EQ(slot.phase, AXIOM_PHASE_LEARNING);
    ASSERT_TRUE(strcmp(transition.reason, "surprise_dissolve_triggered") == 0);
    return 0;
}

static int test_phase_scan_slots_collects_events(void) {
    axiom_phase_slot slots[4];
    memset(slots, 0, sizeof(slots));
    slots[0] = phase_slot(AXIOM_PHASE_COLD, 0.70f, 50u, 0.01f, 0.0f, 0.0f);
    slots[1] = phase_slot(AXIOM_PHASE_LEARNING, 0.85f, 200u, 0.01f, 0.70f, 0.005f);
    slots[2] = phase_slot(AXIOM_PHASE_KNOWN, 0.95f, 400u, 0.40f, 0.90f, 0.02f);
    slots[3].phase = 9u;
    axiom_phase_transition events[4];
    axiom_phase_scan_result result;
    ASSERT_EQ(axiom_phase_scan_slots(slots, 4, 0, events, 4, &result), AXIOM_DAEMON_OK);
    ASSERT_EQ(result.slots_seen, 4u);
    ASSERT_EQ(result.active_slots, 4u);
    ASSERT_EQ(result.changed_slots, 3u);
    ASSERT_EQ(result.invalid_slots, 1u);
    ASSERT_EQ(result.events_written, 3u);
    ASSERT_EQ(slots[0].phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(slots[1].phase, AXIOM_PHASE_KNOWN);
    ASSERT_EQ(slots[2].phase, AXIOM_PHASE_LEARNING);
    return 0;
}

static int test_phase_file_atomic_roundtrip_with_header(void) {
    remove_test_files();
    axiom_phase_slot slots[2];
    slots[0] = phase_slot(AXIOM_PHASE_COLD, 0.70f, 50u, 0.01f, 0.0f, 0.0f);
    slots[1] = phase_slot(AXIOM_PHASE_LEARNING, 0.1f, 5u, 0.5f, 0.0f, 0.0f);
    ASSERT_EQ(axiom_phase_write_file_atomic("daemon_phase_slots.mmap", slots, 2, 1), AXIOM_DAEMON_OK);
    ASSERT_TRUE(file_exists("daemon_phase_slots.mmap"));
    axiom_phase_transition events[2];
    axiom_phase_scan_result result;
    ASSERT_EQ(axiom_phase_scan_file("daemon_phase_slots.mmap", 0, events, 2, &result), AXIOM_DAEMON_OK);
    ASSERT_EQ(result.changed_slots, 2u);
    axiom_phase_slot *read_slots = 0;
    size_t count = 0;
    int has_header = 0;
    ASSERT_EQ(axiom_phase_read_file("daemon_phase_slots.mmap", &read_slots, &count, &has_header), AXIOM_DAEMON_OK);
    ASSERT_EQ(has_header, 1);
    ASSERT_EQ(count, 2u);
    ASSERT_EQ(read_slots[0].phase, AXIOM_PHASE_LEARNING);
    ASSERT_EQ(read_slots[1].phase, AXIOM_PHASE_COLD);
    free(read_slots);
    return 0;
}

static int test_phase_event_json_contract_shape(void) {
    axiom_phase_slot slot = phase_slot(AXIOM_PHASE_COLD, 0.80f, 51u, 0.01f, 0.0f, 0.0f);
    axiom_phase_transition transition;
    char json[AXIOM_EVENT_JSON_MAX];
    ASSERT_EQ(axiom_phase_apply_policy(&slot, 0, "NEWS_ARTICLE", &transition), AXIOM_DAEMON_CHANGED);
    ASSERT_EQ(axiom_phase_transition_event_json(&transition, "00000000-0000-4000-8000-000000000000", json, sizeof(json)), AXIOM_DAEMON_OK);
    ASSERT_TRUE(strstr(json, "\"topic\":\"phase_transition\"") != 0);
    ASSERT_TRUE(strstr(json, "\"topology_class\":\"NEWS_ARTICLE\"") != 0);
    ASSERT_TRUE(strstr(json, "\"from_phase\":1") != 0);
    ASSERT_TRUE(strstr(json, "\"to_phase\":2") != 0);
    ASSERT_TRUE(strstr(json, "confidence_threshold_reached") != 0);
    return 0;
}

static int test_store_missing_critical(void) {
    remove_test_files();
    axiom_store_health health;
    ASSERT_EQ(axiom_store_check_file("daemon_store_missing.bin", &health), AXIOM_DAEMON_OK);
    ASSERT_EQ(health.exists, 0);
    ASSERT_EQ(health.critical, 1);
    ASSERT_TRUE(strcmp(health.status, "missing_critical") == 0);
    return 0;
}

static int test_store_ok_and_manifest_health(void) {
    remove_test_files();
    ASSERT_TRUE(write_bytes("daemon_store_file.bin", "store-bytes", 11u));
    const char *paths[] = {"daemon_store_file.bin", "daemon_store_missing.bin"};
    int critical[] = {AXIOM_STORE_CRITICAL, AXIOM_STORE_OPTIONAL};
    axiom_store_manifest manifest;
    manifest.paths = paths;
    manifest.critical_flags = critical;
    manifest.count = 2u;
    axiom_store_manifest_health health;
    ASSERT_EQ(axiom_store_check_manifest(&manifest, &health), AXIOM_DAEMON_OK);
    ASSERT_EQ(health.checked, 2u);
    ASSERT_EQ(health.missing, 1u);
    ASSERT_EQ(health.critical_failures, 0u);
    ASSERT_TRUE(health.total_bytes >= 11u);
    return 0;
}

static int test_store_baseline_crc_change(void) {
    remove_test_files();
    ASSERT_TRUE(write_bytes("daemon_store_file.bin", "abcdefgh", 8u));
    axiom_store_baseline baseline;
    memset(&baseline, 0, sizeof(baseline));
    axiom_store_health health;
    ASSERT_EQ(axiom_store_check_file_with_baseline("daemon_store_file.bin", AXIOM_STORE_CRITICAL, &baseline, &health), AXIOM_DAEMON_OK);
    ASSERT_TRUE(baseline.initialized);
    ASSERT_TRUE(write_bytes("daemon_store_file.bin", "ABCDEFGH", 8u));
    ASSERT_EQ(axiom_store_check_file_with_baseline("daemon_store_file.bin", AXIOM_STORE_CRITICAL, &baseline, &health), AXIOM_DAEMON_OK);
    ASSERT_EQ(health.header_crc_changed, 1);
    ASSERT_EQ(health.critical, 1);
    ASSERT_TRUE(strstr(health.status, "header_crc_changed") != 0);
    return 0;
}

static int test_store_baseline_size_anomaly_after_window(void) {
    remove_test_files();
    ASSERT_TRUE(write_bytes("daemon_store_file.bin", "1234567890", 10u));
    axiom_store_baseline baseline;
    memset(&baseline, 0, sizeof(baseline));
    axiom_store_health health;
    ASSERT_EQ(axiom_store_check_file_with_baseline("daemon_store_file.bin", AXIOM_STORE_CRITICAL, &baseline, &health), AXIOM_DAEMON_OK);
    baseline.last_seen_unix = baseline.last_seen_unix > 400u ? baseline.last_seen_unix - 400u : 0u;
    ASSERT_TRUE(append_bytes("daemon_store_file.bin", "12345678901234567890", 20u));
    ASSERT_EQ(axiom_store_check_file_with_baseline("daemon_store_file.bin", AXIOM_STORE_CRITICAL, &baseline, &health), AXIOM_DAEMON_OK);
    ASSERT_EQ(health.size_anomaly, 1);
    ASSERT_EQ(health.critical, 1);
    return 0;
}

static int test_store_staging_stale_detection(void) {
    remove_test_files();
    ASSERT_TRUE(write_bytes("daemon_store_file.bin.staging", "pending", 7u));
    AXIOM_UTIMBUF times;
    times.actime = (time_t)(time(NULL) - 1200);
    times.modtime = (time_t)(time(NULL) - 1200);
    AXIOM_UTIME("daemon_store_file.bin.staging", &times);
    axiom_store_health health;
    ASSERT_EQ(axiom_store_staging_health("daemon_store_file.bin", 10u, &health), AXIOM_DAEMON_OK);
    ASSERT_EQ(health.exists, 1);
    ASSERT_EQ(health.staging_stale, 1);
    ASSERT_EQ(health.critical, 1);
    return 0;
}

static int test_store_health_event_json_and_log(void) {
    remove_test_files();
    ASSERT_TRUE(write_bytes("daemon_store_file.bin", "store-bytes", 11u));
    axiom_store_health health;
    char json[AXIOM_EVENT_JSON_MAX];
    ASSERT_EQ(axiom_store_check_file("daemon_store_file.bin", &health), AXIOM_DAEMON_OK);
    ASSERT_EQ(axiom_store_health_event_json(&health, "00000000-0000-4000-8000-000000000000", json, sizeof(json)), AXIOM_DAEMON_OK);
    ASSERT_TRUE(strstr(json, "\"topic\":\"store_health\"") != 0);
    ASSERT_TRUE(strstr(json, "\"checksum_sha256\":null") != 0);
    ASSERT_EQ(axiom_store_append_jsonl("daemon_store_log.jsonl", json), AXIOM_DAEMON_OK);
    ASSERT_TRUE(file_exists("daemon_store_log.jsonl"));
    return 0;
}

static int test_pid_file_validation(void) {
    remove_test_files();
    ASSERT_TRUE(write_bytes("daemon_pid.txt", "0\n", 2u));
    ASSERT_EQ(axiom_store_signal_pid_file("daemon_pid.txt"), AXIOM_DAEMON_SHAPE_ERROR);
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_phase_slot_is_compatible_32_bytes();
    failures += test_phase_cold_to_learning_exact_threshold();
    failures += test_phase_cold_does_not_promote_when_each_criterion_fails();
    failures += test_phase_learning_to_known_exact_threshold();
    failures += test_phase_learning_blocks_single_failed_known_criterion();
    failures += test_phase_known_regresses_on_surprise();
    failures += test_phase_scan_slots_collects_events();
    failures += test_phase_file_atomic_roundtrip_with_header();
    failures += test_phase_event_json_contract_shape();
    failures += test_store_missing_critical();
    failures += test_store_ok_and_manifest_health();
    failures += test_store_baseline_crc_change();
    failures += test_store_baseline_size_anomaly_after_window();
    failures += test_store_staging_stale_detection();
    failures += test_store_health_event_json_and_log();
    failures += test_pid_file_validation();
    remove_test_files();
    if (failures == 0) {
        printf("daemon tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
