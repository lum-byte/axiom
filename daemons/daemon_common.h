#ifndef AXIOM_DAEMON_COMMON_H
#define AXIOM_DAEMON_COMMON_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_PHASE_COLD 1u
#define AXIOM_PHASE_LEARNING 2u
#define AXIOM_PHASE_KNOWN 3u

#define AXIOM_DAEMON_OK 0
#define AXIOM_DAEMON_CHANGED 1
#define AXIOM_DAEMON_INVALID_ARG -1
#define AXIOM_DAEMON_CRC_MISMATCH -2
#define AXIOM_DAEMON_IO_ERROR -3
#define AXIOM_DAEMON_SHAPE_ERROR -4
#define AXIOM_DAEMON_NO_CHANGE -5
#define AXIOM_DAEMON_TRUNCATED -6
#define AXIOM_DAEMON_PERMISSION -7

#define AXIOM_STORE_OPTIONAL 0
#define AXIOM_STORE_CRITICAL 1

#define AXIOM_PHASE_SLOT_BYTES 32u
#define AXIOM_PHASE_HEADER_BYTES 8u
#define AXIOM_PHASE_MAGIC "AXPS"
#define AXIOM_PHASE_VERSION 1u
#define AXIOM_PHASE_SCAN_MAX_EVENTS 4096u

#define AXIOM_STORE_PATH_MAX 512u
#define AXIOM_STATUS_TEXT_MAX 64u
#define AXIOM_DETAIL_TEXT_MAX 256u
#define AXIOM_EVENT_JSON_MAX 2048u
#define AXIOM_STORE_HEADER_BYTES 4096u
#define AXIOM_STORE_SIZE_ANOMALY_RATIO 0.10
#define AXIOM_STORE_SIZE_WINDOW_SECONDS 300u
#define AXIOM_STORE_STAGING_STALE_SECONDS 600u

/*
 * 32-byte phase slot shared with Python readers.
 *
 * Existing classifier code unpacks this as:
 *   <BB H f I f Q 8x
 * so the first 24 bytes must not move.  The final 8 bytes were previously
 * ignored padding; daemons now use them for npi and recipe_yield.  This keeps
 * the hot path backward compatible while satisfying the promotion criteria.
 */
typedef struct axiom_phase_slot {
    uint8_t phase;
    uint8_t flags;
    uint16_t generation;
    float confidence;
    uint32_t observation_count;
    float surprise_rate;
    uint64_t last_updated_unix;
    float npi;
    float recipe_yield;
} axiom_phase_slot;

typedef char axiom_phase_slot_must_be_32_bytes[
    sizeof(axiom_phase_slot) == AXIOM_PHASE_SLOT_BYTES ? 1 : -1
];

typedef struct axiom_phase_policy {
    float cold_to_learning_confidence;
    float cold_to_learning_surprise_max;
    uint32_t cold_to_learning_observations;
    float learning_to_known_confidence;
    float learning_to_known_surprise_max;
    uint32_t learning_to_known_observations;
    float learning_to_known_npi;
    float learning_to_known_recipe_yield;
    float known_regress_surprise;
    float known_regress_confidence;
    float learning_regress_confidence;
} axiom_phase_policy;

typedef struct axiom_phase_transition {
    uint8_t from_phase;
    uint8_t to_phase;
    int changed;
    float confidence;
    float surprise_rate;
    uint32_t observation_count;
    float npi;
    float recipe_yield;
    uint64_t updated_unix;
    char topology_class[96];
    char reason[64];
} axiom_phase_transition;

typedef struct axiom_phase_scan_result {
    size_t slots_seen;
    size_t active_slots;
    size_t changed_slots;
    size_t invalid_slots;
    size_t events_written;
    uint32_t input_crc32;
    uint32_t output_crc32;
} axiom_phase_scan_result;

typedef struct axiom_store_health {
    int exists;
    int readable;
    int writable;
    int critical;
    int mmap_readable;
    int header_crc_changed;
    int size_anomaly;
    int staging_stale;
    uint64_t size_bytes;
    uint64_t checked_unix;
    uint64_t modified_unix;
    uint32_t header_crc32;
    char path[AXIOM_STORE_PATH_MAX];
    char status[AXIOM_STATUS_TEXT_MAX];
    char detail[AXIOM_DETAIL_TEXT_MAX];
} axiom_store_health;

typedef struct axiom_store_baseline {
    int initialized;
    uint64_t size_bytes;
    uint64_t first_seen_unix;
    uint64_t last_seen_unix;
    uint32_t header_crc32;
    char path[AXIOM_STORE_PATH_MAX];
} axiom_store_baseline;

typedef struct axiom_store_manifest {
    const char **paths;
    const int *critical_flags;
    size_t count;
} axiom_store_manifest;

typedef struct axiom_store_manifest_health {
    size_t checked;
    size_t missing;
    size_t unreadable;
    size_t critical_failures;
    size_t size_anomalies;
    size_t crc_changes;
    size_t staging_stale;
    uint64_t total_bytes;
    uint32_t combined_crc32;
} axiom_store_manifest_health;

uint32_t axiom_daemon_crc32(const uint8_t *data, size_t len);
uint32_t axiom_domain_hash(const char *domain);
uint64_t axiom_daemon_now_unix(void);
const char *axiom_phase_name(uint32_t phase);
const char *axiom_phase_contract_reason(uint32_t from_phase, uint32_t to_phase);

axiom_phase_policy axiom_phase_default_policy(void);
int axiom_phase_validate_slot(const axiom_phase_slot *slot);
int axiom_phase_apply_policy(
    axiom_phase_slot *slot,
    const axiom_phase_policy *policy,
    const char *topology_class,
    axiom_phase_transition *transition
);
int axiom_phase_promote(axiom_phase_slot *slot, float theta_learning, float theta_known);
int axiom_phase_scan_slots(
    axiom_phase_slot *slots,
    size_t slot_count,
    const axiom_phase_policy *policy,
    axiom_phase_transition *events,
    size_t event_capacity,
    axiom_phase_scan_result *result
);
int axiom_phase_read_file(const char *path, axiom_phase_slot **slots_out, size_t *slot_count_out, int *has_header_out);
int axiom_phase_write_file_atomic(const char *path, const axiom_phase_slot *slots, size_t slot_count, int write_header);
int axiom_phase_scan_file(
    const char *path,
    const axiom_phase_policy *policy,
    axiom_phase_transition *events,
    size_t event_capacity,
    axiom_phase_scan_result *result
);
int axiom_phase_transition_event_json(
    const axiom_phase_transition *transition,
    const char *run_id,
    char *out_json,
    size_t out_capacity
);

int axiom_store_check_file(const char *path, axiom_store_health *health);
int axiom_store_check_file_ex(const char *path, int critical_flag, axiom_store_health *health);
int axiom_store_check_file_with_baseline(
    const char *path,
    int critical_flag,
    axiom_store_baseline *baseline,
    axiom_store_health *health
);
int axiom_store_check_manifest(const axiom_store_manifest *manifest, axiom_store_manifest_health *health);
int axiom_store_check_manifest_with_baselines(
    const axiom_store_manifest *manifest,
    axiom_store_baseline *baselines,
    axiom_store_manifest_health *health
);
int axiom_store_staging_health(
    const char *final_path,
    uint64_t stale_seconds,
    axiom_store_health *health
);
int axiom_store_health_event_json(
    const axiom_store_health *health,
    const char *run_id,
    char *out_json,
    size_t out_capacity
);
int axiom_store_append_jsonl(const char *path, const char *line);
int axiom_store_signal_pid_file(const char *pid_file);

#ifdef __cplusplus
}
#endif

#endif
