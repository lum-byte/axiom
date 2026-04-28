#define AXIOM_DAEMON_TEST 1
#include "daemon_common.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_TRUE(x) do { if (!(x)) { printf("assert failed: %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ASSERT_EQ(a,b) do { if ((a)!=(b)) { printf("assert eq failed: %s:%d\n", __FILE__, __LINE__); return 1; } } while (0)

static int test_phase_promotion(void) {
    axiom_phase_slot slot;
    memset(&slot, 0, sizeof(slot));
    slot.phase = AXIOM_PHASE_COLD;
    slot.confidence = 0.8f;
    slot.observations = 12u;
    slot.crc32 = axiom_phase_crc(&slot);
    int changed = axiom_phase_promote(&slot, 0.7f, 0.9f);
    ASSERT_EQ(changed, 1);
    ASSERT_EQ(slot.phase, AXIOM_PHASE_LEARNING);
    return 0;
}

static int test_phase_crc_rejects_tamper(void) {
    axiom_phase_slot slot;
    memset(&slot, 0, sizeof(slot));
    slot.phase = AXIOM_PHASE_COLD;
    slot.confidence = 0.8f;
    slot.observations = 12u;
    slot.crc32 = axiom_phase_crc(&slot);
    slot.observations = 99u;
    ASSERT_EQ(axiom_phase_promote(&slot, 0.7f, 0.9f), -2);
    return 0;
}

static int test_missing_store_is_critical(void) {
    axiom_store_health h;
    ASSERT_EQ(axiom_store_check_file("this-file-should-not-exist.bin", &h), 0);
    ASSERT_EQ(h.exists, 0);
    ASSERT_EQ(h.critical, 1);
    return 0;
}

static int test_phase_policy_known_and_demote(void) {
    axiom_phase_slot slot;
    memset(&slot, 0, sizeof(slot));
    slot.phase = AXIOM_PHASE_LEARNING;
    slot.confidence = 0.95f;
    slot.observations = 60u;
    slot.surprises = 0u;
    slot.topology_hash = axiom_domain_hash("Example.COM");
    slot.crc32 = axiom_phase_crc(&slot);
    axiom_phase_transition transition;
    ASSERT_EQ(axiom_phase_apply_policy(&slot, NULL, &transition), 1);
    ASSERT_EQ(slot.phase, AXIOM_PHASE_KNOWN);
    ASSERT_TRUE(strcmp(transition.reason, "confidence_reached_known") == 0);
    slot.surprises = 3u;
    slot.crc32 = axiom_phase_crc(&slot);
    ASSERT_EQ(axiom_phase_apply_policy(&slot, NULL, &transition), 1);
    ASSERT_EQ(slot.phase, AXIOM_PHASE_LEARNING);
    ASSERT_TRUE(strcmp(axiom_phase_name(slot.phase), "LEARNING") == 0);
    return 0;
}

static int test_store_manifest_health(void) {
    const char *path = "daemon-test-store.tmp";
    FILE *f = fopen(path, "wb");
    ASSERT_TRUE(f != NULL);
    fputs("store-bytes", f);
    fclose(f);
    const char *paths[] = {path, "daemon-test-missing.tmp"};
    int critical[] = {AXIOM_STORE_CRITICAL, AXIOM_STORE_OPTIONAL};
    axiom_store_manifest manifest = {.paths = paths, .critical_flags = critical, .count = 2};
    axiom_store_manifest_health health;
    ASSERT_EQ(axiom_store_check_manifest(&manifest, &health), 0);
    ASSERT_EQ(health.checked, 2u);
    ASSERT_EQ(health.missing, 1u);
    ASSERT_EQ(health.critical_failures, 0u);
    remove(path);
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_phase_promotion();
    failures += test_phase_crc_rejects_tamper();
    failures += test_missing_store_is_critical();
    failures += test_phase_policy_known_and_demote();
    failures += test_store_manifest_health();
    if (failures == 0) {
        printf("daemon tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
