#include "offline_api.h"

#include <ctype.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define AXIOM_SCHED_LINE_MAX 8192u
#define AXIOM_SCHED_TEXT_MAX (1024u * 1024u)

static int valid_density(float density) {
    return density >= 0.0f && density <= 1.0f;
}

static void safe_copy(char *dst, size_t cap, const char *src) {
    if (dst == 0 || cap == 0u) {
        return;
    }
    if (src == 0) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, cap, "%s", src);
}

static const char *find_json_key(const char *line, const char *key) {
    if (line == 0 || key == 0) {
        return 0;
    }
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(line, needle);
    if (p == 0) {
        return 0;
    }
    p += strlen(needle);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') {
        return 0;
    }
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    return p;
}

static int json_string_field(const char *line, const char *key, char *out, size_t cap) {
    const char *p = find_json_key(line, key);
    if (p == 0 || *p != '"' || out == 0 || cap == 0u) {
        return 0;
    }
    p++;
    size_t n = 0;
    while (*p && *p != '"') {
        if (*p == '\\' && p[1] != '\0') {
            p++;
            char c = *p;
            if (c == 'n') c = '\n';
            else if (c == 'r') c = '\r';
            else if (c == 't') c = '\t';
            if (n + 1u < cap) out[n++] = c;
            p++;
            continue;
        }
        if (n + 1u < cap) out[n++] = *p;
        p++;
    }
    out[n] = '\0';
    return *p == '"';
}

static int json_int_field(const char *line, const char *key, int *out, int fallback) {
    const char *p = find_json_key(line, key);
    if (out == 0) {
        return 0;
    }
    if (p == 0) {
        *out = fallback;
        return 0;
    }
    char *end = 0;
    long v = strtol(p, &end, 10);
    if (end == p) {
        *out = fallback;
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int json_float_field(const char *line, const char *key, float *out, float fallback) {
    const char *p = find_json_key(line, key);
    if (out == 0) {
        return 0;
    }
    if (p == 0) {
        *out = fallback;
        return 0;
    }
    char *end = 0;
    float v = strtof(p, &end);
    if (end == p) {
        *out = fallback;
        return 0;
    }
    *out = v;
    return 1;
}

static char *read_limited_file(const char *path, size_t max_bytes, size_t *out_len) {
    if (path == 0 || path[0] == '\0') {
        return 0;
    }
    FILE *f = fopen(path, "rb");
    if (f == 0) {
        return 0;
    }
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0;
    }
    long len = ftell(f);
    if (len < 0) {
        fclose(f);
        return 0;
    }
    if ((size_t)len > max_bytes) {
        len = (long)max_bytes;
    }
    rewind(f);
    char *buf = (char *)malloc((size_t)len + 1u);
    if (buf == 0) {
        fclose(f);
        return 0;
    }
    size_t got = fread(buf, 1u, (size_t)len, f);
    fclose(f);
    buf[got] = '\0';
    if (out_len != 0) {
        *out_len = got;
    }
    return buf;
}

static double monotonic_ms(void) {
    return (double)clock() * 1000.0 / (double)CLOCKS_PER_SEC;
}

int axiom_scheduler_init(axiom_batch_scheduler *stats, size_t max_batch, float min_density) {
    if (stats == NULL) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    memset(stats, 0, sizeof(*stats));
    stats->max_batch = max_batch == 0u ? 1024u : max_batch;
    stats->min_density = min_density < 0.0f ? 0.0f : min_density;
    if (stats->min_density > 1.0f) stats->min_density = 1.0f;
    return AXIOM_OFFLINE_OK;
}

int axiom_scheduler_run_once(
    const axiom_zone_feature *features,
    size_t count,
    float *encoded,
    size_t encoded_count,
    axiom_batch_scheduler *stats
) {
    if (stats == NULL || features == NULL || encoded == NULL) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    if (stats->max_batch == 0u) {
        stats->max_batch = 1024u;
    }
    size_t accepted_count = 0;
    axiom_zone_feature *scratch = (axiom_zone_feature *)malloc(sizeof(axiom_zone_feature) * (count == 0u ? 1u : count));
    if (scratch == NULL) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    for (size_t i = 0; i < count && accepted_count < stats->max_batch; ++i) {
        if (!valid_density(features[i].density) || features[i].density < stats->min_density) {
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

int axiom_scheduler_parse_work_line(const char *line, axiom_offline_work_item *item) {
    if (line == 0 || item == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    memset(item, 0, sizeof(*item));
    safe_copy(item->run_id, sizeof(item->run_id), "00000000-0000-4000-8000-000000000000");
    safe_copy(item->topology_class, sizeof(item->topology_class), "UNKNOWN");
    item->label = 0;
    item->density = 0.5f;
    item->priority = 1;
    json_string_field(line, "run_id", item->run_id, sizeof(item->run_id));
    json_string_field(line, "url", item->url, sizeof(item->url));
    json_string_field(line, "topology_class", item->topology_class, sizeof(item->topology_class));
    json_string_field(line, "input_path", item->input_path, sizeof(item->input_path));
    json_int_field(line, "label", &item->label, 0);
    json_int_field(line, "priority", &item->priority, 1);
    json_float_field(line, "density", &item->density, 0.5f);
    if (item->input_path[0] == '\0') {
        return AXIOM_OFFLINE_PARSE_ERROR;
    }
    if (!valid_density(item->density)) {
        return AXIOM_OFFLINE_PARSE_ERROR;
    }
    return AXIOM_OFFLINE_OK;
}

int axiom_scheduler_dead_letter(const char *path, const axiom_offline_work_item *item, const char *reason) {
    if (path == 0 || path[0] == '\0' || item == 0 || reason == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    FILE *f = fopen(path, "ab");
    if (f == 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    fprintf(
        f,
        "{\"run_id\":\"%s\",\"url\":\"%s\",\"topology_class\":\"%s\",\"input_path\":\"%s\",\"reason\":\"%s\"}\n",
        item->run_id,
        item->url,
        item->topology_class,
        item->input_path,
        reason
    );
    int ok = fflush(f);
    int close_error = fclose(f);
    return ok == 0 && close_error == 0 ? AXIOM_OFFLINE_OK : AXIOM_OFFLINE_IO_ERROR;
}

static int build_structural_path(const char *store_dir, char *out, size_t cap) {
    if (store_dir == 0 || out == 0 || cap == 0u) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    size_t len = strlen(store_dir);
    const char *sep = (len > 0u && (store_dir[len - 1u] == '/' || store_dir[len - 1u] == '\\')) ? "" : "/";
    int n = snprintf(out, cap, "%s%sstructural_layer.pt", store_dir, sep);
    return n > 0 && (size_t)n < cap ? AXIOM_OFFLINE_OK : AXIOM_OFFLINE_SHAPE_ERROR;
}

int axiom_scheduler_run_queue(const axiom_scheduler_options *options, axiom_scheduler_stats *stats) {
    if (options == 0 || stats == 0 || options->queue_path == 0 || options->store_dir == 0) {
        return AXIOM_OFFLINE_INVALID_ARG;
    }
    memset(stats, 0, sizeof(*stats));
    double started = monotonic_ms();
    FILE *queue = fopen(options->queue_path, "rb");
    if (queue == 0) {
        return AXIOM_OFFLINE_IO_ERROR;
    }
    char structural_path[768];
    int code = build_structural_path(options->store_dir, structural_path, sizeof(structural_path));
    if (code != AXIOM_OFFLINE_OK) {
        fclose(queue);
        return code;
    }
    code = gpu_encoder_init(structural_path);
    if (code != AXIOM_OFFLINE_OK) {
        fclose(queue);
        return code;
    }
    float lr = options->learning_rate > 0.0f ? options->learning_rate : 1e-4f;
    code = weight_updater_init(structural_path, lr);
    if (code != AXIOM_OFFLINE_OK) {
        fclose(queue);
        return code;
    }
    int accumulate_steps = options->accumulate_steps > 0 ? options->accumulate_steps : 16;
    char line[AXIOM_SCHED_LINE_MAX];
    int pending = 0;
    while (fgets(line, sizeof(line), queue) != 0) {
        stats->lines_read++;
        axiom_offline_work_item item;
        code = axiom_scheduler_parse_work_line(line, &item);
        if (code != AXIOM_OFFLINE_OK) {
            stats->parse_errors++;
            if (options->dead_letter_path != 0) {
                axiom_scheduler_dead_letter(options->dead_letter_path, &item, "parse_error");
                stats->dead_lettered++;
            }
            continue;
        }
        if (item.density < options->min_density) {
            stats->skipped_items++;
            continue;
        }
        size_t text_len = 0;
        char *text = read_limited_file(item.input_path, AXIOM_SCHED_TEXT_MAX, &text_len);
        if (text == 0) {
            stats->dead_lettered++;
            if (options->dead_letter_path != 0) {
                axiom_scheduler_dead_letter(options->dead_letter_path, &item, "input_read_failed");
            }
            continue;
        }
        const char *texts[1] = {text};
        float embedding[AXIOM_ENCODER_WIDTH];
        int label = item.label;
        code = gpu_encoder_encode_batch(texts, 1, embedding, AXIOM_ENCODER_DEFAULT_MAX_SEQ_LEN);
        if (code == AXIOM_OFFLINE_OK) {
            code = weight_updater_accumulate(embedding, &label, 1);
        }
        free(text);
        if (code != AXIOM_OFFLINE_OK) {
            stats->dead_lettered++;
            if (options->dead_letter_path != 0) {
                axiom_scheduler_dead_letter(options->dead_letter_path, &item, "encode_or_accumulate_failed");
            }
            continue;
        }
        stats->encoded_items++;
        pending++;
        if (pending >= accumulate_steps) {
            code = weight_updater_step();
            if (code == AXIOM_OFFLINE_OK) {
                stats->update_steps++;
                pending = 0;
            } else if (code != AXIOM_OFFLINE_NO_WORK) {
                fclose(queue);
                return code;
            }
        }
    }
    if (pending > 0) {
        code = weight_updater_step();
        if (code == AXIOM_OFFLINE_OK) {
            stats->update_steps++;
        } else if (code != AXIOM_OFFLINE_NO_WORK) {
            fclose(queue);
            return code;
        }
    }
    fclose(queue);
    weight_updater_shutdown();
    gpu_encoder_shutdown();
    stats->last_latency_ms = (float)(monotonic_ms() - started);
    return AXIOM_OFFLINE_OK;
}

#ifndef AXIOM_OFFLINE_TEST
static const char *arg_value(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return 0;
}

int main(int argc, char **argv) {
    const char *queue = arg_value(argc, argv, "--queue");
    const char *store = arg_value(argc, argv, "--store");
    const char *dead = arg_value(argc, argv, "--dead-letter");
    const char *steps_s = arg_value(argc, argv, "--accumulate-steps");
    const char *lr_s = arg_value(argc, argv, "--lr");
    if (queue == 0 || store == 0) {
        fprintf(stderr, "usage: batch_scheduler --queue PATH --store DIR [--accumulate-steps N] [--lr LR] [--dead-letter PATH]\n");
        return 2;
    }
    axiom_scheduler_options opts;
    memset(&opts, 0, sizeof(opts));
    opts.queue_path = queue;
    opts.store_dir = store;
    opts.dead_letter_path = dead;
    opts.accumulate_steps = steps_s != 0 ? atoi(steps_s) : 16;
    opts.learning_rate = lr_s != 0 ? (float)atof(lr_s) : 1e-4f;
    opts.min_density = 0.0f;
    opts.max_batch = 1024u;
    axiom_scheduler_stats stats;
    int code = axiom_scheduler_run_queue(&opts, &stats);
    printf(
        "{\"ok\":%s,\"lines_read\":%zu,\"encoded_items\":%zu,\"update_steps\":%zu,\"dead_lettered\":%zu}\n",
        code == AXIOM_OFFLINE_OK ? "true" : "false",
        stats.lines_read,
        stats.encoded_items,
        stats.update_steps,
        stats.dead_lettered
    );
    return code == AXIOM_OFFLINE_OK ? 0 : 1;
}
#endif
