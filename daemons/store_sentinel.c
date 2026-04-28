#include "daemon_common.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

int axiom_store_check_file(const char *path, axiom_store_health *health) {
    return axiom_store_check_file_ex(path, AXIOM_STORE_CRITICAL, health);
}

static int file_writable_probe(const char *path) {
    FILE *f = fopen(path, "ab");
    if (f == NULL) {
        return 0;
    }
    fclose(f);
    return 1;
}

static void set_status(axiom_store_health *health, const char *status) {
    if (health == NULL || status == NULL) {
        return;
    }
    snprintf(health->status, sizeof(health->status), "%s", status);
}

int axiom_store_check_file_ex(const char *path, int critical_flag, axiom_store_health *health) {
    if (path == NULL || health == NULL) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    memset(health, 0, sizeof(*health));
    snprintf(health->path, sizeof(health->path), "%s", path);
    health->critical = critical_flag ? 1 : 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        health->exists = 0;
        set_status(health, critical_flag ? "missing_critical" : "missing_optional");
        return AXIOM_DAEMON_OK;
    }
    health->exists = 1;
    health->size_bytes = (uint64_t)st.st_size;
    health->writable = file_writable_probe(path);
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        health->readable = 0;
        set_status(health, critical_flag ? "unreadable_critical" : "unreadable_optional");
        return AXIOM_DAEMON_OK;
    }
    health->readable = 1;
    uint8_t header[4096];
    size_t n = fread(header, 1, sizeof(header), f);
    fclose(f);
    health->header_crc32 = axiom_daemon_crc32(header, n);
    if (health->size_bytes == 0u && critical_flag) {
        health->critical = 1;
        set_status(health, "empty_critical");
    } else if (health->size_bytes == 0u) {
        set_status(health, "empty_optional");
    } else if (!health->writable && critical_flag) {
        health->critical = 1;
        set_status(health, "readonly_critical");
    } else {
        health->critical = 0;
        set_status(health, "ok");
    }
    return AXIOM_DAEMON_OK;
}

int axiom_store_check_manifest(const axiom_store_manifest *manifest, axiom_store_manifest_health *health) {
    if (manifest == NULL || health == NULL || manifest->paths == NULL) {
        return AXIOM_DAEMON_INVALID_ARG;
    }
    memset(health, 0, sizeof(*health));
    uint32_t combined = 0xFFFFFFFFu;
    for (size_t i = 0; i < manifest->count; ++i) {
        int critical = AXIOM_STORE_CRITICAL;
        if (manifest->critical_flags != NULL) {
            critical = manifest->critical_flags[i];
        }
        axiom_store_health item;
        int rc = axiom_store_check_file_ex(manifest->paths[i], critical, &item);
        if (rc != AXIOM_DAEMON_OK) {
            return rc;
        }
        health->checked++;
        health->total_bytes += (size_t)item.size_bytes;
        if (!item.exists) {
            health->missing++;
        }
        if (item.exists && !item.readable) {
            health->unreadable++;
        }
        if (item.critical) {
            health->critical_failures++;
        }
        combined ^= item.header_crc32 + (uint32_t)(i * 16777619u);
        combined = (combined >> 1u) | (combined << 31u);
    }
    health->combined_crc32 = ~combined;
    return AXIOM_DAEMON_OK;
}

#if !defined(AXIOM_DAEMON_TEST) && !defined(AXIOM_STORE_SENTINEL_NO_MAIN)
int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: store_sentinel <file>\n");
        return 2;
    }
    axiom_store_health h;
    int rc = axiom_store_check_file(argv[1], &h);
    if (rc != 0) {
        return 1;
    }
    printf("{\"exists\":%d,\"readable\":%d,\"critical\":%d,\"size_bytes\":%llu,\"crc32\":%u}\n",
           h.exists, h.readable, h.critical, (unsigned long long)h.size_bytes, h.header_crc32);
    return h.critical ? 1 : 0;
}
#endif
