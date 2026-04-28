#include "offline_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int axiom_scheduler_init(axiom_batch_scheduler *stats, size_t max_batch, float min_density) {
    if (stats == NULL) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    memset(stats, 0, sizeof(*stats));
    stats->max_batch = max_batch == 0 ? 1024u : max_batch;
    stats->min_density = min_density < 0.0f ? 0.0f : min_density;
    return AXIOM_OFFLINE_OK;
}

int axiom_scheduler_run_once(const axiom_zone_feature *features, size_t count, float *encoded, size_t encoded_count, axiom_batch_scheduler *stats) {
    if (stats == NULL || features == NULL || encoded == NULL) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (stats->max_batch == 0u) {
        stats->max_batch = 1024u;
    }
    size_t accepted_count = 0;
    axiom_zone_feature *scratch = (axiom_zone_feature *)malloc(sizeof(axiom_zone_feature) * count);
    if (scratch == NULL && count > 0u) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    for (size_t i = 0; i < count && accepted_count < stats->max_batch; ++i) {
        if (features[i].density < stats->min_density) {
            stats->rejected++;
            continue;
        }
        scratch[accepted_count++] = features[i];
    }
    if (accepted_count == 0u) {
        free(scratch);
        return AXIOM_OFFLINE_NO_WORK;
    }
    int code = axiom_gpu_encode(scratch, accepted_count, encoded, encoded_count);
    if (code == AXIOM_OFFLINE_OK) {
        stats->accepted += accepted_count;
        stats->encoded += accepted_count;
        stats->flushed++;
    } else {
        stats->rejected += accepted_count;
    }
    free(scratch);
    return code;
}

#ifndef AXIOM_OFFLINE_TEST
int main(void) {
    axiom_zone_feature f = {.density = 0.5f, .token_count = 20.0f, .byte_count = 200.0f, .zone_count = 2.0f};
    float out[256];
    axiom_batch_scheduler stats = {0};
    axiom_scheduler_init(&stats, 16, 0.1f);
    int code = axiom_scheduler_run_once(&f, 1, out, 256, &stats);
    printf("{\"ok\":%s,\"encoded\":%zu}\n", code == AXIOM_OFFLINE_OK ? "true" : "false", stats.encoded);
    return code == AXIOM_OFFLINE_OK ? 0 : 1;
}
#endif
