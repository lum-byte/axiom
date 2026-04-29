#define AXIOM_RUNTIME_BUILD 1
#include "axiom_runtime.h"

#include <ctype.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_WIN32)
#include <direct.h>
#include <io.h>
#define AXIOM_MKDIR(path) _mkdir(path)
#define AXIOM_ACCESS(path, mode) _access(path, mode)
#else
#include <sys/stat.h>
#include <unistd.h>
#define AXIOM_MKDIR(path) mkdir(path, 0777)
#define AXIOM_ACCESS(path, mode) access(path, mode)
#endif

#define AXIOM_RUNTIME_VERSION "0.2.0"
#define AXIOM_MAX_FIELD 2048
#define AXIOM_MAX_DOMAIN 256
#define AXIOM_MAX_DOMAINS 512
#define AXIOM_QUEUE_CAPACITY 2048
#define AXIOM_MAX_SOURCES 5
#define AXIOM_RUN_ID_BYTES 64

typedef struct axiom_work_item {
    char kind[32];
    char payload[AXIOM_MAX_FIELD];
    char run_id[AXIOM_RUN_ID_BYTES];
    uint64_t created_unix;
} axiom_work_item;

typedef struct axiom_store_check {
    const char *name;
    uint64_t required_size;
    int exists;
    uint64_t size;
    int ok;
    int repaired;
} axiom_store_check;

struct axiom_runtime {
    char store_dir[AXIOM_MAX_FIELD];
    char socket_path[AXIOM_MAX_FIELD];
    int initialized;
    int store_ready;
    unsigned long handled;
    unsigned long errors;
    unsigned long accepted;
    unsigned long empty;
    unsigned long searches;
    unsigned long fetches;
    unsigned long learns;
    char learned_domains[AXIOM_MAX_DOMAINS][AXIOM_MAX_DOMAIN];
    size_t learned_count;
    axiom_work_item queue[AXIOM_QUEUE_CAPACITY];
    size_t queue_head;
    size_t queue_count;
};

static const axiom_store_check AXIOM_STORE_TEMPLATE[] = {
    {"topology_router.pt", 1024u, 0, 0u, 0, 0},
    {"recipe_registry.mmap", 1024u * 1024u, 0, 0u, 0, 0},
    {"phase_states.mmap", 4096u * 32u, 0, 0u, 0, 0},
    {"structural_layer.pt", 1024u, 0, 0u, 0, 0},
};

static char *json_response(const char *run_id, const char *status, const char *message, const char *data_json);
static void json_escape(const char *input, char *out, size_t out_cap);
static int extract_json_string(const char *json, const char *key, char *out, size_t out_cap);
static void parse_line_command(const char *request, char *command, size_t command_cap, char *payload, size_t payload_cap);
static void upper_ascii(char *text);
static void lower_ascii(char *text);
static void trim_ascii(char *text);
static void make_run_id(char *out, size_t out_cap);
static int is_http_url(const char *value);
static int normalize_domain(const char *raw, char *out, size_t out_cap);
static int ensure_runtime_store(axiom_runtime *runtime);
static int ensure_file_size(const char *path, uint64_t required_size, axiom_store_check *check);
static int path_join(const char *base, const char *name, char *out, size_t out_cap);
static uint64_t file_size_or_zero(const char *path, int *exists);
static int enqueue_work(axiom_runtime *runtime, const char *kind, const char *payload, const char *run_id);
static int add_learned_domain(axiom_runtime *runtime, const char *domain);
static int domain_score(const char *query_lower, const char *domain);
static size_t rank_sources(axiom_runtime *runtime, const char *query, size_t indices[AXIOM_MAX_SOURCES], int scores[AXIOM_MAX_SOURCES]);
static void append_json_text(char *out, size_t out_cap, const char *text);
static void build_sources_json(axiom_runtime *runtime, const size_t *indices, const int *scores, size_t count, char *out, size_t out_cap);
static char *handle_status(axiom_runtime *runtime, const char *run_id);
static char *handle_learn(axiom_runtime *runtime, const char *run_id, const char *payload);
static char *handle_fetch(axiom_runtime *runtime, const char *run_id, const char *payload);
static char *handle_search(axiom_runtime *runtime, const char *run_id, const char *payload);

const char *axiom_version(void) {
    return AXIOM_RUNTIME_VERSION;
}

axiom_runtime *axiom_init(const char *config_json) {
    axiom_runtime *runtime = (axiom_runtime *)calloc(1, sizeof(axiom_runtime));
    if (runtime == NULL) {
        return NULL;
    }
    snprintf(runtime->store_dir, sizeof(runtime->store_dir), "store");
    snprintf(runtime->socket_path, sizeof(runtime->socket_path), "/tmp/axiom_interface.sock");
    if (config_json != NULL) {
        (void)extract_json_string(config_json, "store_dir", runtime->store_dir, sizeof(runtime->store_dir));
        (void)extract_json_string(config_json, "socket_path", runtime->socket_path, sizeof(runtime->socket_path));
    }
    trim_ascii(runtime->store_dir);
    trim_ascii(runtime->socket_path);
    if (runtime->store_dir[0] == '\0') {
        snprintf(runtime->store_dir, sizeof(runtime->store_dir), "store");
    }
    if (runtime->socket_path[0] == '\0') {
        snprintf(runtime->socket_path, sizeof(runtime->socket_path), "/tmp/axiom_interface.sock");
    }
    runtime->initialized = 1;
    runtime->store_ready = ensure_runtime_store(runtime) == 0;
    return runtime;
}

char *axiom_handle_json(axiom_runtime *runtime, const char *request_json) {
    if (runtime == NULL || runtime->initialized == 0) {
        return json_response("00000000-0000-4000-8000-000000000000", "error", "runtime not initialized", "{\"error_type\":\"RuntimeNotInitialized\"}");
    }
    if (request_json == NULL) {
        runtime->errors++;
        return json_response("00000000-0000-4000-8000-000000000000", "error", "request is null", "{\"error_type\":\"InvalidRequest\"}");
    }

    char command[64] = {0};
    char payload[AXIOM_MAX_FIELD] = {0};
    char run_id[AXIOM_RUN_ID_BYTES] = {0};
    if (!extract_json_string(request_json, "run_id", run_id, sizeof(run_id))) {
        make_run_id(run_id, sizeof(run_id));
    }
    if (!extract_json_string(request_json, "command", command, sizeof(command)) &&
        !extract_json_string(request_json, "query_type", command, sizeof(command)) &&
        !extract_json_string(request_json, "type", command, sizeof(command))) {
        parse_line_command(request_json, command, sizeof(command), payload, sizeof(payload));
    } else {
        (void)extract_json_string(request_json, "payload", payload, sizeof(payload));
        if (payload[0] == '\0') {
            (void)extract_json_string(request_json, "text", payload, sizeof(payload));
        }
    }
    trim_ascii(command);
    trim_ascii(payload);
    upper_ascii(command);
    runtime->handled++;

    if (strcmp(command, "STATUS") == 0) {
        return handle_status(runtime, run_id);
    }
    if (strcmp(command, "QUIT") == 0) {
        return json_response(run_id, "ok", "quit accepted", "{\"quit\":true}");
    }
    if (strcmp(command, "LEARN") == 0) {
        return handle_learn(runtime, run_id, payload);
    }
    if (strcmp(command, "FETCH") == 0) {
        return handle_fetch(runtime, run_id, payload);
    }
    if (strcmp(command, "SEARCH") == 0) {
        return handle_search(runtime, run_id, payload);
    }
    runtime->errors++;
    return json_response(run_id, "error", "unknown command", "{\"error_type\":\"UnknownCommand\"}");
}

void axiom_free(char *ptr) {
    free(ptr);
}

void axiom_shutdown(axiom_runtime *runtime) {
    if (runtime != NULL) {
        runtime->initialized = 0;
        free(runtime);
    }
}

static char *handle_status(axiom_runtime *runtime, const char *run_id) {
    runtime->store_ready = ensure_runtime_store(runtime) == 0;
    char esc_store[AXIOM_MAX_FIELD * 2];
    char esc_socket[AXIOM_MAX_FIELD * 2];
    char learned[AXIOM_MAX_FIELD * 2];
    json_escape(runtime->store_dir, esc_store, sizeof(esc_store));
    json_escape(runtime->socket_path, esc_socket, sizeof(esc_socket));
    learned[0] = '\0';
    strncat(learned, "[", sizeof(learned) - strlen(learned) - 1u);
    for (size_t i = 0; i < runtime->learned_count; ++i) {
        char esc[AXIOM_MAX_DOMAIN * 2];
        json_escape(runtime->learned_domains[i], esc, sizeof(esc));
        if (i > 0) strncat(learned, ",", sizeof(learned) - strlen(learned) - 1u);
        strncat(learned, "\"", sizeof(learned) - strlen(learned) - 1u);
        strncat(learned, esc, sizeof(learned) - strlen(learned) - 1u);
        strncat(learned, "\"", sizeof(learned) - strlen(learned) - 1u);
    }
    strncat(learned, "]", sizeof(learned) - strlen(learned) - 1u);
    char data[AXIOM_MAX_FIELD * 5];
    snprintf(
        data,
        sizeof(data),
        "{\"runtime\":\"c_abi\",\"version\":\"%s\",\"store_dir\":\"%s\",\"socket_path\":\"%s\","
        "\"store_ready\":%s,\"bus_started\":false,\"bus_mode\":\"embedded\","
        "\"learned_domains\":%zu,\"learned_domain_names\":%s,\"queued_work_items\":%zu,\"queue_depth\":%zu,"
        "\"handled\":%lu,\"accepted\":%lu,\"empty\":%lu,\"errors\":%lu,"
        "\"searches\":%lu,\"fetches\":%lu,\"learns\":%lu}",
        AXIOM_RUNTIME_VERSION,
        esc_store,
        esc_socket,
        runtime->store_ready ? "true" : "false",
        runtime->learned_count,
        learned,
        runtime->queue_count,
        runtime->queue_count,
        runtime->handled,
        runtime->accepted,
        runtime->empty,
        runtime->errors,
        runtime->searches,
        runtime->fetches,
        runtime->learns
    );
    return json_response(run_id, "ok", "status", data);
}

static char *handle_learn(axiom_runtime *runtime, const char *run_id, const char *payload) {
    char domain[AXIOM_MAX_DOMAIN];
    if (!normalize_domain(payload, domain, sizeof(domain))) {
        runtime->errors++;
        return json_response(run_id, "error", "learn requires a domain", "{\"error_type\":\"InvalidDomain\"}");
    }
    int added = add_learned_domain(runtime, domain);
    enqueue_work(runtime, "learn", domain, run_id);
    runtime->learns++;
    runtime->accepted++;
    char esc_domain[AXIOM_MAX_DOMAIN * 2];
    json_escape(domain, esc_domain, sizeof(esc_domain));
    char data[1024];
    snprintf(
        data,
        sizeof(data),
        "{\"domain\":\"%s\",\"phase\":\"COLD\",\"status\":\"%s\",\"queued\":true,"
        "\"search_engine\":false,\"routing\":\"tag_frontier\"}",
        esc_domain,
        added ? "learned" : "already_known"
    );
    return json_response(run_id, "accepted", "learning queued", data);
}

static char *handle_fetch(axiom_runtime *runtime, const char *run_id, const char *payload) {
    if (!is_http_url(payload)) {
        runtime->errors++;
        return json_response(run_id, "error", "fetch requires http(s) URL", "{\"error_type\":\"InvalidURL\"}");
    }
    char domain[AXIOM_MAX_DOMAIN];
    normalize_domain(payload, domain, sizeof(domain));
    enqueue_work(runtime, "fetch", payload, run_id);
    runtime->fetches++;
    runtime->accepted++;
    char esc_url[AXIOM_MAX_FIELD * 2];
    char esc_domain[AXIOM_MAX_DOMAIN * 2];
    json_escape(payload, esc_url, sizeof(esc_url));
    json_escape(domain, esc_domain, sizeof(esc_domain));
    char data[AXIOM_MAX_FIELD * 3];
    snprintf(
        data,
        sizeof(data),
        "{\"url\":\"%s\",\"domain\":\"%s\",\"status\":\"queued\",\"fetch_mode\":\"static\","
        "\"tor_required\":false,\"chromium_required\":false,\"queued\":true}",
        esc_url,
        esc_domain
    );
    return json_response(run_id, "accepted", "fetch queued", data);
}

static char *handle_search(axiom_runtime *runtime, const char *run_id, const char *payload) {
    if (payload == NULL || payload[0] == '\0') {
        runtime->errors++;
        return json_response(run_id, "error", "query is empty", "{\"error_type\":\"EmptyPayload\"}");
    }
    runtime->searches++;
    size_t indices[AXIOM_MAX_SOURCES];
    int scores[AXIOM_MAX_SOURCES];
    size_t count = rank_sources(runtime, payload, indices, scores);
    if (count == 0u) {
        enqueue_work(runtime, "learn_from_query", payload, run_id);
        runtime->empty++;
        char esc_query[AXIOM_MAX_FIELD * 2];
        json_escape(payload, esc_query, sizeof(esc_query));
        char data[AXIOM_MAX_FIELD * 3];
        snprintf(
            data,
            sizeof(data),
            "{\"query\":\"%s\",\"sources\":[],\"queued_learning\":true,"
            "\"search_engine\":false,\"reason\":\"no learned topology candidates\"}",
            esc_query
        );
        return json_response(run_id, "empty", "no learned topology candidates; learning queued", data);
    }

    char sources[AXIOM_MAX_FIELD * 3];
    build_sources_json(runtime, indices, scores, count, sources, sizeof(sources));
    char esc_query[AXIOM_MAX_FIELD * 2];
    json_escape(payload, esc_query, sizeof(esc_query));
    char signal[AXIOM_MAX_FIELD * 2];
    snprintf(
        signal,
        sizeof(signal),
        "AXIOM routed '%s' through %zu learned topology source(s).",
        payload,
        count
    );
    char esc_signal[AXIOM_MAX_FIELD * 4];
    json_escape(signal, esc_signal, sizeof(esc_signal));
    char data[AXIOM_MAX_FIELD * 7];
    snprintf(
        data,
        sizeof(data),
        "{\"query\":\"%s\",\"signal\":\"%s\",\"sources\":%s,\"topology_classes\":[\"LEARNED_DOMAIN\"],"
        "\"confidence\":%.3f,\"single_inference_point\":\"runtime_synthesizer\","
        "\"search_engine\":false,\"routing\":\"wlm_source_priority_and_frontier\"}",
        esc_query,
        esc_signal,
        sources,
        count > 0u ? 0.72 + ((double)(count > 3u ? 3u : count) * 0.06) : 0.0
    );
    return json_response(run_id, "ok", signal, data);
}

static int enqueue_work(axiom_runtime *runtime, const char *kind, const char *payload, const char *run_id) {
    if (runtime == NULL || kind == NULL || payload == NULL) {
        return 0;
    }
    size_t idx = (runtime->queue_head + runtime->queue_count) % AXIOM_QUEUE_CAPACITY;
    if (runtime->queue_count == AXIOM_QUEUE_CAPACITY) {
        idx = runtime->queue_head;
        runtime->queue_head = (runtime->queue_head + 1u) % AXIOM_QUEUE_CAPACITY;
    } else {
        runtime->queue_count++;
    }
    snprintf(runtime->queue[idx].kind, sizeof(runtime->queue[idx].kind), "%s", kind);
    snprintf(runtime->queue[idx].payload, sizeof(runtime->queue[idx].payload), "%s", payload);
    snprintf(runtime->queue[idx].run_id, sizeof(runtime->queue[idx].run_id), "%s", run_id != NULL ? run_id : "");
    runtime->queue[idx].created_unix = (uint64_t)time(NULL);
    return 1;
}

static int add_learned_domain(axiom_runtime *runtime, const char *domain) {
    if (runtime == NULL || domain == NULL || domain[0] == '\0') {
        return 0;
    }
    for (size_t i = 0; i < runtime->learned_count; ++i) {
        if (strcmp(runtime->learned_domains[i], domain) == 0) {
            return 0;
        }
    }
    if (runtime->learned_count >= AXIOM_MAX_DOMAINS) {
        return 0;
    }
    snprintf(runtime->learned_domains[runtime->learned_count], AXIOM_MAX_DOMAIN, "%s", domain);
    runtime->learned_count++;
    return 1;
}

static size_t rank_sources(axiom_runtime *runtime, const char *query, size_t indices[AXIOM_MAX_SOURCES], int scores[AXIOM_MAX_SOURCES]) {
    if (runtime == NULL || query == NULL || runtime->learned_count == 0u) {
        return 0u;
    }
    char query_lower[AXIOM_MAX_FIELD];
    snprintf(query_lower, sizeof(query_lower), "%s", query);
    lower_ascii(query_lower);
    size_t found = 0u;
    for (size_t i = 0; i < runtime->learned_count; ++i) {
        int score = domain_score(query_lower, runtime->learned_domains[i]);
        size_t pos = found < AXIOM_MAX_SOURCES ? found++ : AXIOM_MAX_SOURCES;
        if (pos == AXIOM_MAX_SOURCES) {
            int worst = 0;
            for (size_t w = 1; w < AXIOM_MAX_SOURCES; ++w) {
                if (scores[w] < scores[worst]) worst = (int)w;
            }
            if (score <= scores[worst]) {
                continue;
            }
            pos = (size_t)worst;
        }
        indices[pos] = i;
        scores[pos] = score;
        for (size_t j = pos; j > 0; --j) {
            if (scores[j] <= scores[j - 1]) {
                break;
            }
            int ts = scores[j - 1];
            scores[j - 1] = scores[j];
            scores[j] = ts;
            size_t ti = indices[j - 1];
            indices[j - 1] = indices[j];
            indices[j] = ti;
        }
    }
    return found > AXIOM_MAX_SOURCES ? AXIOM_MAX_SOURCES : found;
}

static int domain_score(const char *query_lower, const char *domain) {
    int score = 0;
    char domain_lower[AXIOM_MAX_DOMAIN];
    snprintf(domain_lower, sizeof(domain_lower), "%s", domain);
    lower_ascii(domain_lower);
    char token[128];
    size_t ti = 0u;
    for (size_t i = 0;; ++i) {
        unsigned char c = (unsigned char)query_lower[i];
        if (isalnum(c) || c == '-' || c == '_') {
            if (ti + 1u < sizeof(token)) {
                token[ti++] = (char)c;
            }
        } else {
            if (ti > 0u) {
                token[ti] = '\0';
                if (strstr(domain_lower, token) != NULL) {
                    score += 3;
                }
                if (strcmp(token, "paper") == 0 || strcmp(token, "research") == 0 || strcmp(token, "arxiv") == 0) {
                    if (strstr(domain_lower, "arxiv") != NULL) score += 2;
                }
                if (strcmp(token, "wiki") == 0 || strcmp(token, "wikipedia") == 0) {
                    if (strstr(domain_lower, "wikipedia") != NULL) score += 2;
                }
                ti = 0u;
            }
            if (c == '\0') {
                break;
            }
        }
    }
    return score;
}

static void append_json_text(char *out, size_t out_cap, const char *text) {
    if (out == NULL || text == NULL || out_cap == 0u) {
        return;
    }
    size_t used = strlen(out);
    if (used >= out_cap - 1u) {
        return;
    }
    strncat(out, text, out_cap - used - 1u);
}

static void build_sources_json(axiom_runtime *runtime, const size_t *indices, const int *scores, size_t count, char *out, size_t out_cap) {
    if (out == NULL || out_cap == 0u) {
        return;
    }
    out[0] = '\0';
    append_json_text(out, out_cap, "[");
    for (size_t i = 0; i < count; ++i) {
        char esc_domain[AXIOM_MAX_DOMAIN * 2];
        char score_buf[32];
        const char *domain = runtime->learned_domains[indices[i]];
        json_escape(domain, esc_domain, sizeof(esc_domain));
        snprintf(score_buf, sizeof(score_buf), "%d", scores[i]);
        append_json_text(out, out_cap, i == 0 ? "" : ",");
        append_json_text(out, out_cap, "{\"url\":\"https://");
        append_json_text(out, out_cap, esc_domain);
        append_json_text(out, out_cap, "/\",\"domain\":\"");
        append_json_text(out, out_cap, esc_domain);
        append_json_text(out, out_cap, "\",\"score\":");
        append_json_text(out, out_cap, score_buf);
        append_json_text(out, out_cap, "}");
    }
    append_json_text(out, out_cap, "]");
}

static int ensure_runtime_store(axiom_runtime *runtime) {
    if (runtime == NULL) {
        return -1;
    }
    if (AXIOM_MKDIR(runtime->store_dir) != 0 && errno != EEXIST) {
        return -1;
    }
    int ok = 1;
    for (size_t i = 0; i < sizeof(AXIOM_STORE_TEMPLATE) / sizeof(AXIOM_STORE_TEMPLATE[0]); ++i) {
        char path[AXIOM_MAX_FIELD * 2];
        axiom_store_check check = AXIOM_STORE_TEMPLATE[i];
        if (!path_join(runtime->store_dir, check.name, path, sizeof(path))) {
            ok = 0;
            continue;
        }
        if (ensure_file_size(path, check.required_size, &check) != 0 || !check.ok) {
            ok = 0;
        }
    }
    return ok ? 0 : -1;
}

static int ensure_file_size(const char *path, uint64_t required_size, axiom_store_check *check) {
    int exists = 0;
    uint64_t size = file_size_or_zero(path, &exists);
    if (check != NULL) {
        check->exists = exists;
        check->size = size;
        check->repaired = 0;
    }
    if (exists && size >= required_size) {
        if (check != NULL) check->ok = 1;
        return 0;
    }
    FILE *f = fopen(path, "ab");
    if (f == NULL) {
        return -1;
    }
    if (required_size > 0u) {
        if (fseek(f, (long)(required_size - 1u), SEEK_SET) != 0) {
            fclose(f);
            return -1;
        }
        if (fputc('\0', f) == EOF) {
            fclose(f);
            return -1;
        }
    }
    fclose(f);
    if (check != NULL) {
        check->exists = 1;
        check->size = required_size;
        check->ok = 1;
        check->repaired = 1;
    }
    return 0;
}

static uint64_t file_size_or_zero(const char *path, int *exists) {
    FILE *f = fopen(path, "rb");
    if (f == NULL) {
        if (exists != NULL) *exists = 0;
        return 0u;
    }
    if (exists != NULL) *exists = 1;
    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return 0u;
    }
    long end = ftell(f);
    fclose(f);
    return end < 0 ? 0u : (uint64_t)end;
}

static int path_join(const char *base, const char *name, char *out, size_t out_cap) {
    if (base == NULL || name == NULL || out == NULL || out_cap == 0u) {
        return 0;
    }
    size_t len = strlen(base);
    const char *sep = (len > 0u && (base[len - 1u] == '/' || base[len - 1u] == '\\')) ? "" : "/";
    int n = snprintf(out, out_cap, "%s%s%s", base, sep, name);
    return n > 0 && (size_t)n < out_cap;
}

static int normalize_domain(const char *raw, char *out, size_t out_cap) {
    if (raw == NULL || out == NULL || out_cap == 0u) {
        return 0;
    }
    char tmp[AXIOM_MAX_FIELD];
    snprintf(tmp, sizeof(tmp), "%s", raw);
    trim_ascii(tmp);
    lower_ascii(tmp);
    const char *start = tmp;
    const char *scheme = strstr(start, "://");
    if (scheme != NULL) {
        start = scheme + 3;
    }
    while (*start == '/') start++;
    size_t n = 0u;
    int saw_dot = 0;
    for (const char *p = start; *p != '\0'; ++p) {
        unsigned char c = (unsigned char)*p;
        if (c == '/' || c == '?' || c == '#' || c == ':') {
            break;
        }
        if (isspace(c)) {
            return 0;
        }
        if (c == '.') {
            saw_dot = 1;
        }
        if (!(isalnum(c) || c == '-' || c == '.')) {
            return 0;
        }
        if (n + 1u < out_cap) {
            out[n++] = (char)c;
        }
    }
    while (n > 0u && out[n - 1u] == '.') {
        n--;
    }
    out[n] = '\0';
    return n > 0u && saw_dot;
}

static int is_http_url(const char *value) {
    return value != NULL && (strncmp(value, "http://", 7) == 0 || strncmp(value, "https://", 8) == 0);
}

static char *json_response(const char *run_id, const char *status, const char *message, const char *data_json) {
    char esc_run[256];
    char esc_status[64];
    char esc_message[AXIOM_MAX_FIELD * 2];
    json_escape(run_id != NULL ? run_id : "", esc_run, sizeof(esc_run));
    json_escape(status != NULL ? status : "error", esc_status, sizeof(esc_status));
    json_escape(message != NULL ? message : "", esc_message, sizeof(esc_message));
    const char *data = data_json != NULL && data_json[0] != '\0' ? data_json : "{}";
    size_t needed = strlen(esc_run) + strlen(esc_status) + strlen(esc_message) + strlen(data) + 160u;
    char *out = (char *)malloc(needed);
    if (out == NULL) {
        return NULL;
    }
    snprintf(out, needed, "{\"run_id\":\"%s\",\"status\":\"%s\",\"message\":\"%s\",\"data\":%s}", esc_run, esc_status, esc_message, data);
    return out;
}

static void json_escape(const char *input, char *out, size_t out_cap) {
    size_t oi = 0u;
    if (out_cap == 0u) {
        return;
    }
    for (size_t i = 0u; input != NULL && input[i] != '\0' && oi + 2u < out_cap; ++i) {
        unsigned char c = (unsigned char)input[i];
        if (c == '"' || c == '\\') {
            out[oi++] = '\\';
            out[oi++] = (char)c;
        } else if (c == '\n') {
            out[oi++] = '\\';
            out[oi++] = 'n';
        } else if (c == '\r') {
            out[oi++] = '\\';
            out[oi++] = 'r';
        } else if (c == '\t') {
            out[oi++] = '\\';
            out[oi++] = 't';
        } else if (c >= 32u) {
            out[oi++] = (char)c;
        }
    }
    out[oi < out_cap ? oi : out_cap - 1u] = '\0';
}

static int extract_json_string(const char *json, const char *key, char *out, size_t out_cap) {
    if (json == NULL || key == NULL || out == NULL || out_cap == 0u) {
        return 0;
    }
    out[0] = '\0';
    char pattern[128];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *pos = strstr(json, pattern);
    if (pos == NULL) {
        return 0;
    }
    pos += strlen(pattern);
    while (*pos != '\0' && isspace((unsigned char)*pos)) pos++;
    if (*pos != ':') return 0;
    pos++;
    while (*pos != '\0' && isspace((unsigned char)*pos)) pos++;
    if (*pos != '"') return 0;
    pos++;
    size_t oi = 0u;
    int terminated = 0;
    while (*pos != '\0' && oi + 1u < out_cap) {
        if (*pos == '"') {
            terminated = 1;
            break;
        }
        if (*pos == '\\' && pos[1] != '\0') {
            pos++;
            if (*pos == 'n') out[oi++] = '\n';
            else if (*pos == 'r') out[oi++] = '\r';
            else if (*pos == 't') out[oi++] = '\t';
            else out[oi++] = *pos;
            pos++;
            continue;
        }
        out[oi++] = *pos++;
    }
    out[oi] = '\0';
    return terminated;
}

static void parse_line_command(const char *request, char *command, size_t command_cap, char *payload, size_t payload_cap) {
    if (request == NULL || command == NULL || payload == NULL || command_cap == 0u || payload_cap == 0u) {
        return;
    }
    const char *pipe = strchr(request, '|');
    if (pipe == NULL) {
        snprintf(command, command_cap, "%s", request);
        payload[0] = '\0';
        return;
    }
    size_t cmd_len = (size_t)(pipe - request);
    if (cmd_len >= command_cap) cmd_len = command_cap - 1u;
    memcpy(command, request, cmd_len);
    command[cmd_len] = '\0';
    snprintf(payload, payload_cap, "%s", pipe + 1);
}

static void upper_ascii(char *text) {
    for (size_t i = 0u; text != NULL && text[i] != '\0'; ++i) {
        text[i] = (char)toupper((unsigned char)text[i]);
    }
}

static void lower_ascii(char *text) {
    for (size_t i = 0u; text != NULL && text[i] != '\0'; ++i) {
        text[i] = (char)tolower((unsigned char)text[i]);
    }
}

static void trim_ascii(char *text) {
    if (text == NULL) return;
    size_t start = 0u;
    while (isspace((unsigned char)text[start])) start++;
    size_t len = strlen(text + start);
    memmove(text, text + start, len + 1u);
    while (len > 0u && isspace((unsigned char)text[len - 1u])) {
        text[--len] = '\0';
    }
}

static void make_run_id(char *out, size_t out_cap) {
    unsigned int a = (unsigned int)time(NULL);
    unsigned int b = (unsigned int)rand();
    snprintf(out, out_cap, "%08x-%04x-4%03x-8%03x-%012x", a, b & 0xffffu, (b >> 4) & 0xfffu, (b >> 8) & 0xfffu, a ^ b);
}

#ifdef AXIOM_RUNTIME_TEST
int main(void) {
    axiom_runtime *rt = axiom_init("{\"store_dir\":\"axiom_runtime_test_store\"}");
    if (rt == NULL) {
        return 10;
    }
    char *status = axiom_handle_json(rt, "{\"command\":\"status\",\"payload\":\"\"}");
    if (status == NULL || strstr(status, "\"status\":\"ok\"") == NULL || strstr(status, "\"store_ready\":true") == NULL) {
        return 1;
    }
    axiom_free(status);
    char *learn = axiom_handle_json(rt, "{\"command\":\"learn\",\"payload\":\"https://arxiv.org/abs/2401.00001\"}");
    if (learn == NULL || strstr(learn, "\"status\":\"accepted\"") == NULL || strstr(learn, "arxiv.org") == NULL) {
        return 2;
    }
    axiom_free(learn);
    char *search = axiom_handle_json(rt, "{\"command\":\"search\",\"payload\":\"latest RNA folding arxiv papers\"}");
    if (search == NULL || strstr(search, "\"status\":\"ok\"") == NULL || strstr(search, "https://arxiv.org/") == NULL) {
        return 3;
    }
    axiom_free(search);
    char *fetch = axiom_handle_json(rt, "{\"command\":\"fetch\",\"payload\":\"https://example.com\"}");
    if (fetch == NULL || strstr(fetch, "\"status\":\"accepted\"") == NULL) {
        return 4;
    }
    axiom_free(fetch);
    char *quit = axiom_handle_json(rt, "quit |");
    if (quit == NULL || strstr(quit, "\"quit\":true") == NULL) {
        return 5;
    }
    axiom_free(quit);
    axiom_shutdown(rt);
    return 0;
}
#endif
