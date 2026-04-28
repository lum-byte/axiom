#include "daemon_common.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

uint32_t axiom_domain_hash(const char *domain) {
    if (domain == NULL) {
        return 0;
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

uint32_t axiom_phase_crc(const axiom_phase_slot *slot) {
    if (slot == NULL) {
        return 0;
    }
    axiom_phase_slot tmp = *slot;
    tmp.crc32 = 0;
    return axiom_daemon_crc32((const uint8_t *)&tmp, sizeof(tmp));
}

static int valid_phase(uint32_t p) {
    return p == AXIOM_PHASE_COLD || p == AXIOM_PHASE_LEARNING || p == AXIOM_PHASE_KNOWN;
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

axiom_phase_policy axiom_phase_default_policy(void) {
    axiom_phase_policy policy;
    policy.theta_learning = 0.70f;
    policy.theta_known = 0.90f;
    policy.min_learning_observations = 10u;
    policy.min_known_observations = 50u;
    policy.max_known_surprises = 0u;
    policy.demote_confidence = 0.45f;
    return policy;
}

static void transition_set(axiom_phase_transition *transition, uint32_t old_phase, uint32_t new_phase, int changed, const char *reason) {
    if (transition == NULL) {
        return;
    }
    memset(transition, 0, sizeof(*transition));
    transition->old_phase = old_phase;
    transition->new_phase = new_phase;
    transition->changed = changed;
    transition->updated_unix = (uint64_t)time(NULL);
    if (reason != NULL) {
        snprintf(transition->reason, sizeof(transition->reason), "%s", reason);
    }
}

int axiom_phase_apply_policy(axiom_phase_slot *slot, const axiom_phase_policy *policy, axiom_phase_transition *transition) {
    if (slot == NULL || !valid_phase(slot->phase)) {
        transition_set(transition, 0u, 0u, 0, "invalid_phase");
        return AXIOM_DAEMON_INVALID_ARG;
    }
    if (slot->crc32 != 0 && slot->crc32 != axiom_phase_crc(slot)) {
        transition_set(transition, slot->phase, slot->phase, 0, "crc_mismatch");
        return AXIOM_DAEMON_CRC_MISMATCH;
    }
    axiom_phase_policy effective = policy != NULL ? *policy : axiom_phase_default_policy();
    if (effective.min_learning_observations == 0u) {
        effective.min_learning_observations = 10u;
    }
    if (effective.min_known_observations == 0u) {
        effective.min_known_observations = 50u;
    }
    uint32_t old = slot->phase;
    const char *reason = "stable";
    if (slot->phase == AXIOM_PHASE_COLD &&
        slot->confidence >= effective.theta_learning &&
        slot->observations >= effective.min_learning_observations) {
        slot->phase = AXIOM_PHASE_LEARNING;
        reason = "confidence_reached_learning";
    } else if (slot->phase == AXIOM_PHASE_LEARNING &&
               slot->confidence >= effective.theta_known &&
               slot->observations >= effective.min_known_observations &&
               slot->surprises <= effective.max_known_surprises) {
        slot->phase = AXIOM_PHASE_KNOWN;
        reason = "confidence_reached_known";
    } else if (slot->phase == AXIOM_PHASE_KNOWN &&
               (slot->surprises > effective.max_known_surprises || slot->confidence < effective.demote_confidence)) {
        slot->phase = AXIOM_PHASE_LEARNING;
        reason = "surprise_or_confidence_demotion";
    } else if (slot->phase == AXIOM_PHASE_LEARNING && slot->confidence < effective.demote_confidence && slot->observations < effective.min_learning_observations) {
        slot->phase = AXIOM_PHASE_COLD;
        reason = "insufficient_learning_evidence";
    }
    if (old != slot->phase) {
        slot->updated_unix = (uint64_t)time(NULL);
        slot->crc32 = axiom_phase_crc(slot);
        transition_set(transition, old, slot->phase, 1, reason);
        return 1;
    }
    if (slot->crc32 == 0) {
        slot->crc32 = axiom_phase_crc(slot);
    }
    transition_set(transition, old, slot->phase, 0, reason);
    return AXIOM_DAEMON_OK;
}

int axiom_phase_promote(axiom_phase_slot *slot, float theta_learning, float theta_known) {
    axiom_phase_policy policy = axiom_phase_default_policy();
    policy.theta_learning = theta_learning;
    policy.theta_known = theta_known;
    return axiom_phase_apply_policy(slot, &policy, NULL);
}

#if !defined(AXIOM_DAEMON_TEST) && !defined(AXIOM_PHASE_DAEMON_NO_MAIN)
int main(void) {
    axiom_phase_slot slot;
    memset(&slot, 0, sizeof(slot));
    slot.phase = AXIOM_PHASE_COLD;
    slot.confidence = 0.8f;
    slot.observations = 16u;
    slot.crc32 = axiom_phase_crc(&slot);
    int changed = axiom_phase_promote(&slot, 0.70f, 0.90f);
    printf("{\"changed\":%d,\"phase\":%u}\n", changed, slot.phase);
    return changed < 0 ? 1 : 0;
}
#endif
