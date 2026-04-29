#include "daemon_common.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <io.h>
#include <process.h>
#include <sys/stat.h>
#define AXIOM_ACCESS _access
#define AXIOM_R_OK 4
#define AXIOM_W_OK 2
#define AXIOM_FSYNC(fd) _commit(fd)
#define AXIOM_FILENO(f) _fileno(f)
#else
#include <signal.h>
#include <sys/stat.h>
#include <unistd.h>
#define AXIOM_ACCESS access
#define AXIOM_R_OK R_OK
#define AXIOM_W_OK W_OK
#define AXIOM_FSYNC(fd) fsync(fd)
#define AXIOM_FILENO(f) fileno(f)
#endif

static void axiom_store_zero_health(axiom_store_health *health, const char *path, int critical_flag) {
    if (health == 0) {
        return;
    }
    memset(health, 0, sizeof(*health));
    health->critical = critical_flag ? 1 : 0;
    health->checked_unix = axiom_daemon_now_unix();
    if (path != 0) {
        size_t n = strlen(path);
        if (n >= sizeof(health->path)) {
            n = sizeof(health->path) - 1u;
        }
        memcpy(health->path, path, n);
        health->path[n] = '\0';
    }
}

static void axiom_store_set_status(axiom_store_health *health, const char *status, const char *detail) {
    if (health == 0) {
        return;
    }
    snprintf(health->status, sizeof(health->status), "%s", status != 0 ? status : "unknown");
    snprintf(health->detail, sizeof(health->detail), "%s", detail != 0 ? detail : "");
}

static uint64_t axiom_stat_mtime_unix(const struct stat *st) {
    if (st == 0 || st->st_mtime < 0) {
        return 0u;
    }
    return (uint64_t)st->st_mtime;
}

static int axiom_header_crc(const char *path, uint32_t *crc_out, int *readable_out) {
    if (path == 0 || crc_out == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    if (readable_out != 0) {
        *readable_out = 0;
    }
    *crc_out = 0u;
    FILE *f = fopen(path, "rb");
    if (f == 0) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    uint8_t header[AXIOM_STORE_HEADER_BYTES];
    size_t got = fread(header, 1u, sizeof(header), f);
    int failed = ferror(f);
    fclose(f);
    if (failed) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    if (readable_out != 0) {
        *readable_out = 1;
    }
    *crc_out = axiom_daemon_crc32(header, got);
    return AXIOM_DAEMON_OK;
}

static int axiom_file_read_probe(const char *path) {
    FILE *f = fopen(path, "rb");
    if (f == 0) {
        return 0;
    }
    uint8_t b = 0u;
    (void)fread(&b, 1u, 1u, f);
    int failed = ferror(f);
    fclose(f);
    return failed ? 0 : 1;
}

static int axiom_file_write_probe(const char *path) {
    if (path == 0) {
        return 0;
    }
    return AXIOM_ACCESS(path, AXIOM_W_OK) == 0 ? 1 : 0;
}

static int axiom_size_anomaly(uint64_t previous, uint64_t current) {
    if (previous == 0u) {
        return 0;
    }
    double prev = (double)previous;
    double cur = (double)current;
    double ratio = cur > prev ? (cur - prev) / prev : (prev - cur) / prev;
    return ratio > AXIOM_STORE_SIZE_ANOMALY_RATIO ? 1 : 0;
}

static void axiom_store_update_baseline(
    axiom_store_baseline *baseline,
    const char *path,
    const axiom_store_health *health
) {
    if (baseline == 0 || health == 0 || !health->exists || !health->readable) {
        return;
    }
    if (!baseline->initialized) {
        memset(baseline, 0, sizeof(*baseline));
        baseline->initialized = 1;
        baseline->first_seen_unix = health->checked_unix;
        snprintf(baseline->path, sizeof(baseline->path), "%s", path != 0 ? path : health->path);
    }
    baseline->size_bytes = health->size_bytes;
    baseline->header_crc32 = health->header_crc32;
    baseline->last_seen_unix = health->checked_unix;
}

static void axiom_store_apply_baseline(
    axiom_store_baseline *baseline,
    axiom_store_health *health
) {
    if (baseline == 0 || health == 0 || !health->exists || !health->readable) {
        return;
    }
    if (baseline->initialized) {
        uint64_t elapsed = health->checked_unix >= baseline->last_seen_unix
            ? health->checked_unix - baseline->last_seen_unix
            : 0u;
        if (elapsed >= AXIOM_STORE_SIZE_WINDOW_SECONDS &&
            axiom_size_anomaly(baseline->size_bytes, health->size_bytes)) {
            health->size_anomaly = 1;
        }
        if (baseline->header_crc32 != 0u && baseline->header_crc32 != health->header_crc32) {
            health->header_crc_changed = 1;
        }
    }
}

int axiom_store_check_file(const char *path, axiom_store_health *health) {
    return axiom_store_check_file_ex(path, AXIOM_STORE_CRITICAL, health);
}

int axiom_store_check_file_ex(const char *path, int critical_flag, axiom_store_health *health) {
    return axiom_store_check_file_with_baseline(path, critical_flag, 0, health);
}

int axiom_store_check_file_with_baseline(
    const char *path,
    int critical_flag,
    axiom_store_baseline *baseline,
    axiom_store_health *health
) {
    if (path == 0 || health == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    axiom_store_zero_health(health, path, critical_flag);
    struct stat st;
    if (stat(path, &st) != 0) {
        axiom_store_set_status(health, critical_flag ? "missing_critical" : "missing_optional", strerror(errno));
        return AXIOM_DAEMON_OK;
    }
    health->exists = 1;
    health->size_bytes = (uint64_t)st.st_size;
    health->modified_unix = axiom_stat_mtime_unix(&st);
    health->writable = axiom_file_write_probe(path);
    int readable_probe = 0;
    int crc_code = axiom_header_crc(path, &health->header_crc32, &readable_probe);
    health->readable = readable_probe && crc_code == AXIOM_DAEMON_OK;
    health->mmap_readable = axiom_file_read_probe(path);

    axiom_store_apply_baseline(baseline, health);

    if (!health->readable) {
        health->critical = critical_flag ? 1 : 0;
        axiom_store_set_status(health, critical_flag ? "unreadable_critical" : "unreadable_optional", "header read failed");
    } else if (!health->mmap_readable) {
        health->critical = critical_flag ? 1 : 0;
        axiom_store_set_status(health, critical_flag ? "mmap_probe_failed_critical" : "mmap_probe_failed_optional", "read probe failed");
    } else if (health->size_bytes == 0u && critical_flag) {
        health->critical = 1;
        axiom_store_set_status(health, "empty_critical", "critical store file is empty");
    } else if (health->size_bytes == 0u) {
        axiom_store_set_status(health, "empty_optional", "optional store file is empty");
    } else if (health->size_anomaly) {
        health->critical = critical_flag ? 1 : 0;
        axiom_store_set_status(health, critical_flag ? "size_anomaly_critical" : "size_anomaly_optional", "size changed by more than 10 percent over baseline window");
    } else if (health->header_crc_changed) {
        health->critical = critical_flag ? 1 : 0;
        axiom_store_set_status(health, critical_flag ? "header_crc_changed_critical" : "header_crc_changed_optional", "first 4KB CRC changed since baseline");
    } else if (!health->writable && critical_flag) {
        health->critical = 1;
        axiom_store_set_status(health, "readonly_critical", "critical store file is not writable by daemon user");
    } else {
        health->critical = 0;
        axiom_store_set_status(health, "ok", "store file passed sentinel checks");
    }

    if (!health->size_anomaly && !health->header_crc_changed && health->readable && health->mmap_readable) {
        axiom_store_update_baseline(baseline, path, health);
    }
    return AXIOM_DAEMON_OK;
}

int axiom_store_staging_health(
    const char *final_path,
    uint64_t stale_seconds,
    axiom_store_health *health
) {
    if (final_path == 0 || health == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    char staging[AXIOM_STORE_PATH_MAX + 32u];
    int n = snprintf(staging, sizeof(staging), "%s.staging", final_path);
    if (n < 0 || (size_t)n >= sizeof(staging)) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    axiom_store_zero_health(health, staging, AXIOM_STORE_OPTIONAL);
    struct stat st;
    if (stat(staging, &st) != 0) {
        axiom_store_set_status(health, "ok", "no staging file present");
        return AXIOM_DAEMON_OK;
    }
    health->exists = 1;
    health->readable = axiom_file_read_probe(staging);
    health->writable = axiom_file_write_probe(staging);
    health->size_bytes = (uint64_t)st.st_size;
    health->modified_unix = axiom_stat_mtime_unix(&st);
    uint64_t now = health->checked_unix;
    uint64_t age = now >= health->modified_unix ? now - health->modified_unix : 0u;
    uint64_t limit = stale_seconds == 0u ? AXIOM_STORE_STAGING_STALE_SECONDS : stale_seconds;
    if (age > limit) {
        health->staging_stale = 1;
        health->critical = 1;
        axiom_store_set_status(health, "staging_stale", "staging file exceeded stale rename window");
    } else {
        axiom_store_set_status(health, "staging_present", "staging file exists but is within allowed window");
    }
    return AXIOM_DAEMON_OK;
}

int axiom_store_check_manifest(const axiom_store_manifest *manifest, axiom_store_manifest_health *health) {
    return axiom_store_check_manifest_with_baselines(manifest, 0, health);
}

int axiom_store_check_manifest_with_baselines(
    const axiom_store_manifest *manifest,
    axiom_store_baseline *baselines,
    axiom_store_manifest_health *health
) {
    if (manifest == 0 || manifest->paths == 0 || health == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    memset(health, 0, sizeof(*health));
    uint32_t combined = 0xFFFFFFFFu;
    for (size_t i = 0; i < manifest->count; ++i) {
        int critical = AXIOM_STORE_CRITICAL;
        if (manifest->critical_flags != 0) {
            critical = manifest->critical_flags[i];
        }
        axiom_store_health item;
        axiom_store_baseline *baseline = baselines != 0 ? &baselines[i] : 0;
        int code = axiom_store_check_file_with_baseline(manifest->paths[i], critical, baseline, &item);
        if (code != AXIOM_DAEMON_OK) {
            return code;
        }
        health->checked++;
        health->total_bytes += item.size_bytes;
        if (!item.exists) health->missing++;
        if (item.exists && !item.readable) health->unreadable++;
        if (item.critical) health->critical_failures++;
        if (item.size_anomaly) health->size_anomalies++;
        if (item.header_crc_changed) health->crc_changes++;
        if (item.staging_stale) health->staging_stale++;
        combined ^= item.header_crc32 + (uint32_t)(i * 16777619u);
        combined = (combined >> 1u) | (combined << 31u);

        axiom_store_health staging;
        code = axiom_store_staging_health(manifest->paths[i], AXIOM_STORE_STAGING_STALE_SECONDS, &staging);
        if (code == AXIOM_DAEMON_OK && staging.staging_stale) {
            health->staging_stale++;
            health->critical_failures++;
        }
    }
    health->combined_crc32 = ~combined;
    return AXIOM_DAEMON_OK;
}

static size_t axiom_json_escape_local(const char *src, char *dst, size_t cap) {
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

int axiom_store_health_event_json(
    const axiom_store_health *health,
    const char *run_id,
    char *out_json,
    size_t out_capacity
) {
    if (health == 0 || run_id == 0 || out_json == 0 || out_capacity == 0u) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    char path[AXIOM_STORE_PATH_MAX * 2u];
    char status[AXIOM_STATUS_TEXT_MAX * 2u];
    char detail[AXIOM_DETAIL_TEXT_MAX * 2u];
    char run[128];
    axiom_json_escape_local(health->path, path, sizeof(path));
    axiom_json_escape_local(health->status, status, sizeof(status));
    axiom_json_escape_local(health->detail, detail, sizeof(detail));
    axiom_json_escape_local(run_id, run, sizeof(run));
    int n = snprintf(
        out_json,
        out_capacity,
        "{\"topic\":\"store_health\",\"component\":\"daemons.store_sentinel\","
        "\"payload\":{\"store_file\":\"%s\",\"status\":\"%s\",\"size_bytes\":%llu,"
        "\"checksum_sha256\":null,\"critical\":%s,\"detail\":\"%s\","
        "\"run_id\":\"%s\",\"checked_at\":\"%llu\"}}",
        path,
        status,
        (unsigned long long)health->size_bytes,
        health->critical ? "true" : "false",
        detail,
        run,
        (unsigned long long)health->checked_unix
    );
    if (n < 0 || (size_t)n >= out_capacity) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
    return AXIOM_DAEMON_OK;
}

int axiom_store_append_jsonl(const char *path, const char *line) {
    if (path == 0 || line == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    FILE *f = fopen(path, "ab");
    if (f == 0) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    size_t len = strlen(line);
    size_t wrote = fwrite(line, 1u, len, f);
    size_t nl = fwrite("\n", 1u, 1u, f);
    int flush_error = fflush(f);
    int sync_error = AXIOM_FSYNC(AXIOM_FILENO(f));
    int close_error = fclose(f);
    return wrote == len && nl == 1u && flush_error == 0 && sync_error == 0 && close_error == 0
        ? AXIOM_DAEMON_OK
        : AXIOM_DAEMON_IO_ERROR;
}

int axiom_store_signal_pid_file(const char *pid_file) {
    if (pid_file == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    FILE *f = fopen(pid_file, "rb");
    if (f == 0) {
        return AXIOM_DAEMON_IO_ERROR;
    }
    long pid = 0;
    int scanned = fscanf(f, "%ld", &pid);
    fclose(f);
    if (scanned != 1 || pid <= 0) {
        return AXIOM_DAEMON_SHAPE_ERROR;
    }
#if defined(_WIN32)
    return AXIOM_DAEMON_OK;
#else
    return kill((pid_t)pid, SIGUSR1) == 0 ? AXIOM_DAEMON_OK : AXIOM_DAEMON_IO_ERROR;
#endif
}

#if !defined(AXIOM_DAEMON_TEST) && !defined(AXIOM_STORE_SENTINEL_NO_MAIN)
static const char *axiom_arg_value_sentinel(int argc, char **argv, const char *name) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (strcmp(argv[i], name) == 0) {
            return argv[i + 1];
        }
    }
    return 0;
}

static int axiom_build_store_paths(const char *store, const char **paths, char owned[4][AXIOM_STORE_PATH_MAX]) {
    static const char *names[4] = {
        "topology_router.pt",
        "recipe_registry.mmap",
        "phase_states.mmap",
        "structural_layer.pt",
    };
    if (store == 0 || paths == 0) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    size_t len = strlen(store);
    const char *sep = (len > 0u && (store[len - 1u] == '/' || store[len - 1u] == '\\')) ? "" : "/";
    for (size_t i = 0; i < 4u; ++i) {
        int n = snprintf(owned[i], AXIOM_STORE_PATH_MAX, "%s%s%s", store, sep, names[i]);
        if (n < 0 || (size_t)n >= AXIOM_STORE_PATH_MAX) {
            return AXIOM_DAEMON_SHAPE_ERROR;
        }
        paths[i] = owned[i];
    }
    return AXIOM_DAEMON_OK;
}

int main(int argc, char **argv) {
    const char *store = axiom_arg_value_sentinel(argc, argv, "--store");
    const char *single = axiom_arg_value_sentinel(argc, argv, "--file");
    const char *log_path = axiom_arg_value_sentinel(argc, argv, "--log");
    const char *run_id = axiom_arg_value_sentinel(argc, argv, "--run-id");
    if (run_id == 0) {
        run_id = "00000000-0000-4000-8000-000000000000";
    }
    if (single != 0) {
        axiom_store_health health;
        int code = axiom_store_check_file(single, &health);
        if (code != AXIOM_DAEMON_OK) {
            return 1;
        }
        char json[AXIOM_EVENT_JSON_MAX];
        if (axiom_store_health_event_json(&health, run_id, json, sizeof(json)) == AXIOM_DAEMON_OK) {
            puts(json);
            if (log_path != 0) {
                axiom_store_append_jsonl(log_path, json);
            }
        }
        return health.critical ? 1 : 0;
    }
    if (store == 0) {
        fprintf(stderr, "usage: store_sentinel --store DIR [--log PATH] [--run-id UUID]\n");
        return 2;
    }
    const char *paths[4];
    char owned[4][AXIOM_STORE_PATH_MAX];
    int code = axiom_build_store_paths(store, paths, owned);
    if (code != AXIOM_DAEMON_OK) {
        return 1;
    }
    int critical[4] = {
        AXIOM_STORE_CRITICAL,
        AXIOM_STORE_CRITICAL,
        AXIOM_STORE_CRITICAL,
        AXIOM_STORE_CRITICAL,
    };
    axiom_store_manifest manifest;
    manifest.paths = paths;
    manifest.critical_flags = critical;
    manifest.count = 4u;
    axiom_store_manifest_health health;
    code = axiom_store_check_manifest(&manifest, &health);
    printf(
        "{\"ok\":%s,\"daemon\":\"store_sentinel\",\"checked\":%zu,\"critical_failures\":%zu,\"total_bytes\":%llu}\n",
        code == AXIOM_DAEMON_OK ? "true" : "false",
        health.checked,
        health.critical_failures,
        (unsigned long long)health.total_bytes
    );
    return code == AXIOM_DAEMON_OK && health.critical_failures == 0u ? 0 : 1;
}
#endif
