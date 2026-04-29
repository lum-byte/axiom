#include "daemon_common.h"

#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#define AXIOM_FSYNC(fd) _commit(fd)
#define AXIOM_FILENO(f) _fileno(f)
#else
#include <unistd.h>
#define AXIOM_FSYNC(fd) fsync(fd)
#define AXIOM_FILENO(f) fileno(f)
#endif

static int axiom_is_finite_float(float value) {
    return isfinite(value) && value >= 0.0f;
}

static float axiom_clamp01(float value) {
    if (!isfinite(value) || value < 0.0f) {
        return 0.0f;
    }
    if (value > 1.0f) {
        return 1.0f;
    }
    return value;
}

static void axiom_safe_copy(char *dst, size_t cap, const char *src) {
    if (dst == 0 || cap == 0u) {
        return;
    }
    if (src == 0) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, cap, "%s", src);
}

static void axiom_transition_clear(axiom_phase_transition *transition) {
    if (transition != 0) {
        memset(transition, 0, sizeof(*transition));
    }
}

static int axiom_valid_phase(uint32_t phase) {
    return phase == AXIOM_PHASE_COLD || phase == AXIOM_PHASE_LEARNING || phase == AXIOM_PHASE_KNOWN;
}

uint64_t axiom_daemon_now_unix(void) {
    time_t now = time(NULL);
    if (now < 0) {
        return 0u;
    }
    return (uint64_t)now;
}

uint32_t axiom_domain_hash(const char *domain) {
    if (domain == NULL) {
        return 0u;
    }
    uint32_t h = 2166136261u;
    for (const unsigned char *p = (const unsigned char *)domain; *p != 0; ++p) {
        unsigned char c = *p;
        if (c >= 'A' && c <= 'Z') {
            c = (unsigned char)(c - 'A' + 'a');
        }
        h ^= (uint32_t)c;
        h *= 16777619u;
    }
    return h;
}

uint32_t axiom_daemon_crc32(const uint8_t *data, size_t len) {
    if (data == 0 && len > 0u) {
        return 0u;
    }
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

const char *axiom_phase_name(uint32_t phase) {
    switch (phase) {
        case AXIOM_PHASE_COLD:
            return "COLD";
        case AXIOM_PHASE_LEARNING:
            return "LEARNING";
        case AXIOM_PHASE_KNOWN:
            return "KNOWN";
        default:
            return "UNKNOWN";
    }
}

const char *axiom_phase_contract_reason(uint32_t from_phase, uint32_t to_phase) {
    if ((from_phase == AXIOM_PHASE_COLD && to_phase == AXIOM_PHASE_LEARNING) ||
        (from_phase == AXIOM_PHASE_LEARNING && to_phase == AXIOM_PHASE_KNOWN)) {
        return "confidence_threshold_reached";
    }
    if (from_phase == AXIOM_PHASE_KNOWN && to_phase == AXIOM_PHASE_LEARNING) {
        return "surprise_dissolve_triggered";
    }
    if (from_phase == AXIOM_PHASE_LEARNING && to_phase == AXIOM_PHASE_COLD) {
        return "reindex_forced";
    }
    return "reindex_forced";
}

axiom_phase_policy axiom_phase_default_policy(void) {
    axiom_phase_policy policy;
    policy.cold_to_learning_confidence = 0.70f;
    policy.cold_to_learning_surprise_max = 0.20f;
    policy.cold_to_learning_observations = 50u;
    policy.learning_to_known_confidence = 0.85f;
    policy.learning_to_known_surprise_max = 0.05f;
    policy.learning_to_known_observations = 200u;
    policy.learning_to_known_npi = 0.70f;
    policy.learning_to_known_recipe_yield = 0.005f;
    policy.known_regress_surprise = 0.35f;
    policy.known_regress_confidence = 0.55f;
    policy.learning_regress_confidence = 0.20f;
    return policy;
}

static axiom_phase_policy axiom_phase_effective_policy(const axiom_phase_policy *policy) {
    axiom_phase_policy effective = policy != 0 ? *policy : axiom_phase_default_policy();
    if (effective.cold_to_learning_confidence <= 0.0f) effective.cold_to_learning_confidence = 0.70f;
    if (effective.cold_to_learning_observations == 0u) effective.cold_to_learning_observations = 50u;
    if (effective.learning_to_known_confidence <= 0.0f) effective.learning_to_known_confidence = 0.85f;
    if (effective.learning_to_known_observations == 0u) effective.learning_to_known_observations = 200u;
    if (effective.learning_to_known_npi <= 0.0f) effective.learning_to_known_npi = 0.70f;
    if (effective.learning_to_known_recipe_yield <= 0.0f) effective.learning_to_known_recipe_yield = 0.005f;
    effective.cold_to_learning_surprise_max = axiom_clamp01(effective.cold_to_learning_surprise_max);
    effective.learning_to_known_surprise_max = axiom_clamp01(effective.learning_to_known_surprise_max);
    effective.known_regress_surprise = axiom_clamp01(effective.known_regress_surprise);
    effective.known_regress_confidence = axiom_clamp01(effective.known_regress_confidence);
    effective.learning_regress_confidence = axiom_clamp01(effective.learning_regress_confidence);
    return effective;
}

int axiom_phase_validate_slot(const axiom_phase_slot *slot) {
    if (slot == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    if (slot->phase == 0u) {
        return AXIOM_DAEMON_NO_CHANGE;
    }
    if (!axiom_valid_phase(slot->phase)) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    if (!axiom_is_finite_float(slot->confidence) ||
        !axiom_is_finite_float(slot->surprise_rate) ||
        !axiom_is_finite_float(slot->npi) ||
        !axiom_is_finite_float(slot->recipe_yield)) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    if (slot->confidence > 1.0f || slot->surprise_rate > 1.0f || slot->npi > 1.0f) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    return AXIOM_DAEMON_OK;
}

static int axiom_phase_should_promote_cold(const axiom_phase_slot *slot, const axiom_phase_policy *policy) {
    return slot->observation_count >= policy->cold_to_learning_observations &&
           slot->confidence >= policy->cold_to_learning_confidence &&
           slot->surprise_rate < policy->cold_to_learning_surprise_max;
}

static int axiom_phase_should_promote_learning(const axiom_phase_slot *slot, const axiom_phase_policy *policy) {
    return slot->observation_count >= policy->learning_to_known_observations &&
           slot->confidence >= policy->learning_to_known_confidence &&
           slot->surprise_rate < policy->learning_to_known_surprise_max &&
           slot->npi >= policy->learning_to_known_npi &&
           slot->recipe_yield >= policy->learning_to_known_recipe_yield;
}

static int axiom_phase_should_regress_known(const axiom_phase_slot *slot, const axiom_phase_policy *policy) {
    return slot->surprise_rate > policy->known_regress_surprise ||
           slot->confidence < policy->known_regress_confidence;
}

static int axiom_phase_should_regress_learning(const axiom_phase_slot *slot, const axiom_phase_policy *policy) {
    return slot->observation_count > 0u &&
           slot->observation_count < policy->cold_to_learning_observations &&
           slot->confidence < policy->learning_regress_confidence &&
           slot->surprise_rate >= policy->cold_to_learning_surprise_max;
}

static void axiom_phase_set_transition(
    axiom_phase_transition *transition,
    const axiom_phase_slot *slot,
    uint8_t from_phase,
    uint8_t to_phase,
    int changed,
    const char *topology_class
) {
    if (transition == 0 || slot == 0) {
        return;
    }
    memset(transition, 0, sizeof(*transition));
    transition->from_phase = from_phase;
    transition->to_phase = to_phase;
    transition->changed = changed;
    transition->confidence = slot->confidence;
    transition->surprise_rate = slot->surprise_rate;
    transition->observation_count = slot->observation_count;
    transition->npi = slot->npi;
    transition->recipe_yield = slot->recipe_yield;
    transition->updated_unix = slot->last_updated_unix;
    axiom_safe_copy(transition->topology_class, sizeof(transition->topology_class), topology_class != 0 ? topology_class : "UNKNOWN");
    axiom_safe_copy(transition->reason, sizeof(transition->reason), axiom_phase_contract_reason(from_phase, to_phase));
}

int axiom_phase_apply_policy(
    axiom_phase_slot *slot,
    const axiom_phase_policy *policy,
    const char *topology_class,
    axiom_phase_transition *transition
) {
    axiom_transition_clear(transition);
    if (slot == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    int valid = axiom_phase_validate_slot(slot);
    if (valid == AXIOM_DAEMON_NO_CHANGE) {
        return AXIOM_DAEMON_NO_CHANGE;
    }
    if (valid != AXIOM_DAEMON_OK) {
        return valid;
    }

    axiom_phase_policy effective = axiom_phase_effective_policy(policy);
    uint8_t old_phase = slot->phase;
    uint8_t new_phase = old_phase;

    if (slot->phase == AXIOM_PHASE_COLD && axiom_phase_should_promote_cold(slot, &effective)) {
        new_phase = AXIOM_PHASE_LEARNING;
    } else if (slot->phase == AXIOM_PHASE_LEARNING && axiom_phase_should_promote_learning(slot, &effective)) {
        new_phase = AXIOM_PHASE_KNOWN;
    } else if (slot->phase == AXIOM_PHASE_KNOWN && axiom_phase_should_regress_known(slot, &effective)) {
        new_phase = AXIOM_PHASE_LEARNING;
    } else if (slot->phase == AXIOM_PHASE_LEARNING && axiom_phase_should_regress_learning(slot, &effective)) {
        new_phase = AXIOM_PHASE_COLD;
    }

    if (new_phase == old_phase) {
        axiom_phase_set_transition(transition, slot, old_phase, old_phase, 0, topology_class);
        return AXIOM_DAEMON_OK;
    }

    slot->phase = new_phase;
    slot->generation = (uint16_t)(slot->generation + 1u);
    slot->last_updated_unix = axiom_daemon_now_unix();
    axiom_phase_set_transition(transition, slot, old_phase, new_phase, 1, topology_class);
    return AXIOM_DAEMON_CHANGED;
}

int axiom_phase_promote(axiom_phase_slot *slot, float theta_learning, float theta_known) {
    axiom_phase_policy policy = axiom_phase_default_policy();
    policy.cold_to_learning_confidence = theta_learning;
    policy.learning_to_known_confidence = theta_known;
    return axiom_phase_apply_policy(slot, &policy, "UNKNOWN", 0);
}

int axiom_phase_scan_slots(
    axiom_phase_slot *slots,
    size_t slot_count,
    const axiom_phase_policy *policy,
    axiom_phase_transition *events,
    size_t event_capacity,
    axiom_phase_scan_result *result
) {
    if (slots == 0 && slot_count > 0u) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    if (result != 0) {
        memset(result, 0, sizeof(*result));
        result->slots_seen = slot_count;
        result->input_crc32 = axiom_daemon_crc32((const uint8_t *)slots, slot_count * sizeof(axiom_phase_slot));
    }
    for (size_t i = 0; i < slot_count; ++i) {
        axiom_phase_slot *slot = &slots[i];
        if (slot->phase == 0u) {
            continue;
        }
        if (result != 0) {
            result->active_slots++;
        }
        char topology_class[96];
        snprintf(topology_class, sizeof(topology_class), "TOPOLOGY_SLOT_%zu", i);
        axiom_phase_transition transition;
        int code = axiom_phase_apply_policy(slot, policy, topology_class, &transition);
        if (code == AXIOM_DAEMON_CHANGED) {
            if (result != 0) {
                result->changed_slots++;
            }
            if (events != 0 && result != 0 && result->events_written < event_capacity) {
                events[result->events_written++] = transition;
            }
        } else if (code < 0 && result != 0) {
            result->invalid_slots++;
        }
    }
    if (result != 0) {
        result->output_crc32 = axiom_daemon_crc32((const uint8_t *)slots, slot_count * sizeof(axiom_phase_slot));
    }
    return AXIOM_DAEMON_OK;
}

static int axiom_read_whole_file(const char *path, uint8_t **data_out, size_t *len_out) {
    if (path == 0 || data_out == 0 || len_out == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    *data_out = 0;
    *len_out = 0u;
    FILE *f = fopen(path, "rb");
    if (f == 0) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return AXIOM_DAEMON_IO_ERROR;
    }
    long end = ftell(f);
    if (end < 0) {
        fclose(f);
        return AXIOM_DAEMON_IO_ERROR;
    }
    rewind(f);
    size_t len = (size_t)end;
    uint8_t *data = (uint8_t *)malloc(len == 0u ? 1u : len);
    if (data == 0) {
        fclose(f);
        return AXIOM_DAEMON_IO_ERROR;
    }
    size_t got = fread(data, 1u, len, f);
    int failed = ferror(f);
    fclose(f);
    if (failed || got != len) {
        free(data);
        return AXIOM_DAEMON_IO_ERROR;
    }
    *data_out = data;
    *len_out = len;
    return AXIOM_DAEMON_OK;
}

int axiom_phase_read_file(const char *path, axiom_phase_slot **slots_out, size_t *slot_count_out, int *has_header_out) {
    if (path == 0 || slots_out == 0 || slot_count_out == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    *slots_out = 0;
    *slot_count_out = 0u;
    if (has_header_out != 0) {
        *has_header_out = 0;
    }
    uint8_t *data = 0;
    size_t len = 0u;
    int code = axiom_read_whole_file(path, &data, &len);
    if (code != AXIOM_DAEMON_OK) {
        return code;
    }
    size_t offset = 0u;
    if (len >= AXIOM_PHASE_HEADER_BYTES && memcmp(data, AXIOM_PHASE_MAGIC, 4u) == 0) {
        offset = AXIOM_PHASE_HEADER_BYTES;
        if (has_header_out != 0) {
            *has_header_out = 1;
        }
    }
    if (len < offset || ((len - offset) % AXIOM_PHASE_SLOT_BYTES) != 0u) {
        free(data);
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    size_t slot_count = (len - offset) / AXIOM_PHASE_SLOT_BYTES;
    axiom_phase_slot *slots = (axiom_phase_slot *)calloc(slot_count == 0u ? 1u : slot_count, sizeof(axiom_phase_slot));
    if (slots == 0) {
        free(data);
        return AXIOM_DAEMON_IO_ERROR;
    }
    if (slot_count > 0u) {
        memcpy(slots, data + offset, slot_count * sizeof(axiom_phase_slot));
    }
    free(data);
    *slots_out = slots;
    *slot_count_out = slot_count;
    return AXIOM_DAEMON_OK;
}

static int axiom_atomic_replace(const char *staging_path, const char *final_path) {
#if defined(_WIN32)
    remove(final_path);
#endif
    return rename(staging_path, final_path) == 0 ? AXIOM_DAEMON_OK : AXIOM_DAEMON_IO_ERROR;
}

int axiom_phase_write_file_atomic(const char *path, const axiom_phase_slot *slots, size_t slot_count, int write_header) {
    if (path == 0 || (slots == 0 && slot_count > 0u)) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    char staging[AXIOM_STORE_PATH_MAX + 32u];
    int n = snprintf(staging, sizeof(staging), "%s.staging", path);
    if (n < 0 || (size_t)n >= sizeof(staging)) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    FILE *f = fopen(staging, "wb");
    if (f == 0) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    if (write_header) {
        uint8_t header[AXIOM_PHASE_HEADER_BYTES] = {'A', 'X', 'P', 'S', (uint8_t)AXIOM_PHASE_VERSION, 0u, 0u, 0u};
        if (fwrite(header, 1u, sizeof(header), f) != sizeof(header)) {
            fclose(f);
            return AXIOM_DAEMON_IO_ERROR;
        }
    }
    size_t wrote = fwrite(slots, sizeof(axiom_phase_slot), slot_count, f);
    int flush_error = fflush(f);
    int sync_error = AXIOM_FSYNC(AXIOM_FILENO(f));
    int close_error = fclose(f);
    if (wrote != slot_count || flush_error != 0 || sync_error != 0 || close_error != 0) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    return axiom_atomic_replace(staging, path);
}

int axiom_phase_scan_file(
    const char *path,
    const axiom_phase_policy *policy,
    axiom_phase_transition *events,
    size_t event_capacity,
    axiom_phase_scan_result *result
) {
    axiom_phase_slot *slots = 0;
    size_t slot_count = 0u;
    int has_header = 0;
    int code = axiom_phase_read_file(path, &slots, &slot_count, &has_header);
    if (code != AXIOM_DAEMON_OK) {
        return code;
    }
    code = axiom_phase_scan_slots(slots, slot_count, policy, events, event_capacity, result);
    if (code == AXIOM_DAEMON_OK && result != 0 && result->changed_slots > 0u) {
        code = axiom_phase_write_file_atomic(path, slots, slot_count, has_header);
    }
    free(slots);
    return code;
}

static size_t axiom_json_escape(const char *src, char *dst, size_t cap) {
    size_t out = 0u;
    if (dst == 0 || cap == 0u) {
        return 0u;
    }
    for (size_t i = 0; src != 0 && src[i] != '\0'; ++i) {
        unsigned char c = (unsigned char)src[i];
        const char *escape = 0;
        char small[7];
        if (c == '\\') escape = "\\\\";
        else if (c == '"') escape = "\\\"";
        else if (c == '\n') escape = "\\n";
        else if (c == '\r') escape = "\\r";
        else if (c == '\t') escape = "\\t";
        else if (c < 0x20u) {
            snprintf(small, sizeof(small), "\\u%04x", (unsigned)c);
            escape = small;
        }
        if (escape != 0) {
            for (size_t j = 0; escape[j] != '\0'; ++j) {
                if (out + 1u >= cap) {
                    dst[out] = '\0';
                    return out;
                }
                dst[out++] = escape[j];
            }
        } else {
            if (out + 1u >= cap) {
                dst[out] = '\0';
                return out;
            }
            dst[out++] = (char)c;
        }
    }
    dst[out] = '\0';
    return out;
}

int axiom_phase_transition_event_json(
    const axiom_phase_transition *transition,
    const char *run_id,
    char *out_json,
    size_t out_capacity
) {
    if (transition == 0 || run_id == 0 || out_json == 0 || out_capacity == 0u || !transition->changed) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    char topology[192];
    char reason[128];
    char run[128];
    axiom_json_escape(transition->topology_class, topology, sizeof(topology));
    axiom_json_escape(transition->reason, reason, sizeof(reason));
    axiom_json_escape(run_id, run, sizeof(run));
    int n = snprintf(
        out_json,
        out_capacity,
        "{\"topic\":\"phase_transition\",\"component\":\"daemons.phase_daemon\","
        "\"payload\":{\"topology_class\":\"%s\",\"from_phase\":%u,\"to_phase\":%u,"
        "\"confidence\":%.6f,\"reason\":\"%s\",\"run_id\":\"%s\","
        "\"timestamp\":\"%llu\"}}",
        topology,
        (unsigned)transition->from_phase,
        (unsigned)transition->to_phase,
        transition->confidence,
        reason,
        run,
        (unsigned long long)transition->updated_unix
    );
    if (n < 0 || (size_t)n >= out_capacity) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    return AXIOM_DAEMON_OK;
}

#if !defined(AXIOM_DAEMON_TEST) && !defined(AXIOM_PHASE_DAEMON_NO_MAIN)
static const char *axiom_arg_value(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *phase_file = axiom_arg_value(argc, argv, "--phase-file");
    const char *run_id = axiom_arg_value(argc, argv, "--run-id");
    if (phase_file == 0) {
        const char *store = axiom_arg_value(argc, argv, "--store");
        static char derived[AXIOM_STORE_PATH_MAX];
        if (store != 0) {
            size_t len = strlen(store);
            const char *sep = (len > 0u && (store[len - 1u] == '/' || store[len - 1u] == '\\')) ? "" : "/";
            snprintf(derived, sizeof(derived), "%s%sphase_states.mmap", store, sep);
            phase_file = derived;
        }
    }
    if (run_id == 0) {
        run_id = "00000000-0000-4000-8000-000000000000";
    }
    if (phase_file == 0) {
        fprintf(stderr, "usage: phase_daemon --phase-file PATH [--run-id UUID]\n");
        return 2;
    }
    axiom_phase_transition events[AXIOM_PHASE_SCAN_MAX_EVENTS];
    axiom_phase_scan_result result;
    int code = axiom_phase_scan_file(phase_file, 0, events, AXIOM_PHASE_SCAN_MAX_EVENTS, &result);
    if (code != AXIOM_DAEMON_OK) {
        fprintf(stderr, "{\"ok\":false,\"daemon\":\"phase\",\"code\":%d}\n", code);
        return 1;
    }
    for (size_t i = 0; i < result.events_written; ++i) {
        char json[AXIOM_EVENT_JSON_MAX];
        if (axiom_phase_transition_event_json(&events[i], run_id, json, sizeof(json)) == AXIOM_DAEMON_OK) {
            puts(json);
        }
    }
    fprintf(
        stderr,
        "{\"ok\":true,\"daemon\":\"phase\",\"slots_seen\":%zu,\"active_slots\":%zu,\"changed_slots\":%zu}\n",
        result.slots_seen,
        result.active_slots,
        result.changed_slots
    );
    return 0;
}
#endif
