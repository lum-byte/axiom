#ifndef AXIOM_DAEMON_COMMON_H
#define AXIOM_DAEMON_COMMON_H

#include <stddef.h>
#include <stdint.h>

#define AXIOM_PHASE_COLD 1
#define AXIOM_PHASE_LEARNING 2
#define AXIOM_PHASE_KNOWN 3

#define AXIOM_DAEMON_OK 0
#define AXIOM_DAEMON_INVALID_ARG -1
#define AXIOM_DAEMON_CRC_MISMATCH -2
#define AXIOM_DAEMON_IO_ERROR -3

#define AXIOM_STORE_OPTIONAL 0
#define AXIOM_STORE_CRITICAL 1

typedef struct axiom_phase_slot {
    uint32_t topology_hash;
    uint32_t phase;
    float confidence;
    uint32_t observations;
    uint32_t surprises;
    uint64_t updated_unix;
    uint32_t crc32;
} axiom_phase_slot;

typedef struct axiom_phase_policy {
    float theta_learning;
    float theta_known;
    uint32_t min_learning_observations;
    uint32_t min_known_observations;
    uint32_t max_known_surprises;
    float demote_confidence;
} axiom_phase_policy;

typedef struct axiom_phase_transition {
    uint32_t old_phase;
    uint32_t new_phase;
    int changed;
    char reason[64];
    uint64_t updated_unix;
} axiom_phase_transition;

typedef struct axiom_store_health {
    int exists;
    int readable;
    int critical;
    int writable;
    uint64_t size_bytes;
    uint32_t header_crc32;
    char path[260];
    char status[32];
} axiom_store_health;

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
    size_t total_bytes;
    uint32_t combined_crc32;
} axiom_store_manifest_health;

uint32_t axiom_daemon_crc32(const uint8_t *data, size_t len);
uint32_t axiom_domain_hash(const char *domain);
uint32_t axiom_phase_crc(const axiom_phase_slot *slot);
int axiom_phase_promote(axiom_phase_slot *slot, float theta_learning, float theta_known);
int axiom_phase_apply_policy(axiom_phase_slot *slot, const axiom_phase_policy *policy, axiom_phase_transition *transition);
axiom_phase_policy axiom_phase_default_policy(void);
const char *axiom_phase_name(uint32_t phase);
int axiom_store_check_file(const char *path, axiom_store_health *health);
int axiom_store_check_file_ex(const char *path, int critical_flag, axiom_store_health *health);
int axiom_store_check_manifest(const axiom_store_manifest *manifest, axiom_store_manifest_health *health);

#endif
