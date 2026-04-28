#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AXIOM_COLD_OK 0
#define AXIOM_COLD_INVALID_ARG 1
#define AXIOM_COLD_IO_ERROR 2
#define AXIOM_COLD_EMPTY_FILE 3
#define AXIOM_COLD_UNDERSIZED 4

typedef struct axiom_cold_file_health {
    int exists;
    int readable;
    unsigned long long size_bytes;
    uint32_t header_crc32;
    int ok;
} axiom_cold_file_health;

uint32_t axiom_cold_crc32(const unsigned char *data, size_t len) {
    uint32_t crc = 0xFFFFFFFFu;
    if (data == NULL && len > 0u) {
        return 0u;
    }
    for (size_t i = 0; i < len; ++i) {
        crc ^= (uint32_t)data[i];
        for (int bit = 0; bit < 8; ++bit) {
            uint32_t mask = (uint32_t)-(int)(crc & 1u);
            crc = (crc >> 1u) ^ (0xEDB88320u & mask);
        }
    }
    return ~crc;
}

int axiom_cold_file_check(const char *path, unsigned long long min_size, axiom_cold_file_health *health) {
    if (path == NULL || strlen(path) == 0 || health == NULL) {
        return AXIOM_COLD_INVALID_ARG;
    }
    memset(health, 0, sizeof(*health));
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        return AXIOM_COLD_IO_ERROR;
    }
    health->exists = 1;
    health->readable = 1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return AXIOM_COLD_IO_ERROR;
    }
    long size = ftell(f);
    if (size < 0) {
        fclose(f);
        return AXIOM_COLD_IO_ERROR;
    }
    health->size_bytes = (unsigned long long)size;
    rewind(f);
    unsigned char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf), f);
    fclose(f);
    health->header_crc32 = axiom_cold_crc32(buf, n);
    if (health->size_bytes == 0u) {
        health->ok = 0;
        return AXIOM_COLD_EMPTY_FILE;
    }
    if (health->size_bytes < min_size) {
        health->ok = 0;
        return AXIOM_COLD_UNDERSIZED;
    }
    health->ok = 1;
    return AXIOM_COLD_OK;
}

int axiom_validate_store_crc(const char *path) {
    axiom_cold_file_health health;
    int code = axiom_cold_file_check(path, 1u, &health);
    return code == AXIOM_COLD_OK ? 0 : code;
}

int axiom_validate_store_file(const char *path, unsigned long long min_size) {
    axiom_cold_file_health health;
    int code = axiom_cold_file_check(path, min_size, &health);
    return code == AXIOM_COLD_OK ? 0 : code;
}

int axiom_validate_tor_port(const char *host, int port) {
    if (host == NULL || strlen(host) == 0 || port <= 0 || port > 65535) {
        return 1;
    }
    return 0;
}

int axiom_validate_gvisor_marker(const char *marker_path) {
    if (marker_path == NULL || strlen(marker_path) == 0) {
        return 1;
    }
    FILE *f = fopen(marker_path, "rb");
    if (f == NULL) {
        return 2;
    }
    fclose(f);
    return 0;
}

#ifdef AXIOM_COLD_START_TEST
int main(void) {
    if (axiom_validate_tor_port("127.0.0.1", 9050) != 0) {
        return 1;
    }
    const char *path = "cold-start-c-test.tmp";
    FILE *f = fopen(path, "wb");
    if (f == NULL) {
        return 2;
    }
    fputs("store", f);
    fclose(f);
    int rc = axiom_validate_store_file(path, 4u);
    remove(path);
    return rc == 0 ? 0 : 3;
}
#endif
