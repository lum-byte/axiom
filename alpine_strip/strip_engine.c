#include "strip_engine.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define AXIOM_POOL_BYTES (64u * 1024u * 1024u)

typedef struct strip_pool {
    uint8_t *base;
    size_t capacity;
    size_t offset;
} strip_pool;

static strip_pool g_pool = {0};

static const char *strcasestr_local(const char *haystack, size_t hay_len, const char *needle);
static size_t strip_comments(const char *input, size_t len, char *out);
static size_t strip_attributes(const char *input, size_t len, char *out);
static size_t decode_entities(const char *input, size_t len, char *out);
static size_t keep_between(const char *input, size_t len, const char *start, const char *end, char *out);
static size_t strip_between(const char *input, size_t len, const char *start, const char *end, char *out);
static size_t filter_lines_containing(const char *input, size_t len, const char *needle, char *out);
static size_t truncate_bytes(const char *input, size_t len, const char *limit_text, char *out);
static bool step_kind_known(const char *kind);
static int parse_positive_size(const char *text, size_t fallback, size_t *out);

static int pool_init(void) {
    if (g_pool.base != NULL) {
        g_pool.offset = 0;
        return AXIOM_STRIP_OK;
    }
    g_pool.base = (uint8_t *)malloc(AXIOM_POOL_BYTES);
    if (g_pool.base == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    g_pool.capacity = AXIOM_POOL_BYTES;
    g_pool.offset = 0;
    return AXIOM_STRIP_OK;
}

static void *pool_alloc(size_t n) {
    if (n == 0) {
        return g_pool.base + g_pool.offset;
    }
    size_t aligned = (n + 7u) & ~((size_t)7u);
    if (g_pool.offset + aligned > g_pool.capacity) {
        return NULL;
    }
    void *ptr = g_pool.base + g_pool.offset;
    g_pool.offset += aligned;
    return ptr;
}

uint32_t axiom_crc32(const uint8_t *data, size_t len) {
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

static bool starts_with_ci(const char *s, size_t remain, const char *needle) {
    size_t n = strlen(needle);
    if (remain < n) {
        return false;
    }
    for (size_t i = 0; i < n; ++i) {
        if (tolower((unsigned char)s[i]) != tolower((unsigned char)needle[i])) {
            return false;
        }
    }
    return true;
}

static size_t strip_tag_block(const char *input, size_t len, const char *tag, char *out) {
    size_t oi = 0;
    char open[64];
    char close[64];
    snprintf(open, sizeof(open), "<%s", tag);
    snprintf(close, sizeof(close), "</%s>", tag);
    for (size_t i = 0; i < len;) {
        if (starts_with_ci(input + i, len - i, open)) {
            const char *end = strcasestr_local(input + i, len - i, close);
            if (end == NULL) {
                break;
            }
            i = (size_t)(end - input) + strlen(close);
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

static const char *strcasestr_local(const char *haystack, size_t hay_len, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0 || hay_len < n) {
        return NULL;
    }
    for (size_t i = 0; i + n <= hay_len; ++i) {
        if (starts_with_ci(haystack + i, hay_len - i, needle)) {
            return haystack + i;
        }
    }
    return NULL;
}

static size_t collapse_ws(const char *input, size_t len, char *out) {
    bool in_ws = false;
    size_t oi = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)input[i];
        if (isspace(c)) {
            if (!in_ws) {
                out[oi++] = ' ';
                in_ws = true;
            }
            continue;
        }
        out[oi++] = (char)c;
        in_ws = false;
    }
    while (oi > 0 && out[oi - 1] == ' ') {
        oi--;
    }
    return oi;
}

static size_t remove_literal(const char *input, size_t len, const char *needle, char *out) {
    size_t n = strlen(needle);
    if (n == 0) {
        memcpy(out, input, len);
        return len;
    }
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (i + n <= len && memcmp(input + i, needle, n) == 0) {
            i += n;
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

static size_t replace_literal(const char *input, size_t len, const char *needle, const char *repl, char *out) {
    size_t n = strlen(needle);
    size_t r = strlen(repl);
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (n > 0 && i + n <= len && memcmp(input + i, needle, n) == 0) {
            memcpy(out + oi, repl, r);
            oi += r;
            i += n;
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

static size_t strip_html_tags(const char *input, size_t len, char *out) {
    bool in_tag = false;
    size_t oi = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = input[i];
        if (c == '<') {
            in_tag = true;
            out[oi++] = ' ';
            continue;
        }
        if (c == '>') {
            in_tag = false;
            continue;
        }
        if (!in_tag) {
            out[oi++] = c;
        }
    }
    return oi;
}

static size_t strip_comments(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (i + 4u <= len && memcmp(input + i, "<!--", 4u) == 0) {
            const char *end = strcasestr_local(input + i + 4u, len - i - 4u, "-->");
            if (end == NULL) {
                break;
            }
            i = (size_t)(end - input) + 3u;
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

static size_t strip_attributes(const char *input, size_t len, char *out) {
    bool in_tag = false;
    bool in_quote = false;
    char quote = '\0';
    bool wrote_tag_name = false;
    size_t oi = 0;
    for (size_t i = 0; i < len; ++i) {
        char c = input[i];
        if (!in_tag) {
            if (c == '<') {
                in_tag = true;
                wrote_tag_name = false;
                out[oi++] = c;
                continue;
            }
            out[oi++] = c;
            continue;
        }
        if (in_quote) {
            if (c == quote) {
                in_quote = false;
            }
            continue;
        }
        if (c == '"' || c == '\'') {
            in_quote = true;
            quote = c;
            continue;
        }
        if (c == '>') {
            if (oi > 0 && out[oi - 1] == ' ') {
                oi--;
            }
            out[oi++] = c;
            in_tag = false;
            continue;
        }
        if (isspace((unsigned char)c)) {
            if (!wrote_tag_name) {
                out[oi++] = ' ';
                wrote_tag_name = true;
            }
            continue;
        }
        if (!wrote_tag_name || c == '/' || c == '!') {
            out[oi++] = c;
        }
    }
    return oi;
}

static int entity_value(const char *name, size_t len, char *out) {
    if (len == 2u && memcmp(name, "lt", 2u) == 0) { *out = '<'; return 1; }
    if (len == 2u && memcmp(name, "gt", 2u) == 0) { *out = '>'; return 1; }
    if (len == 3u && memcmp(name, "amp", 3u) == 0) { *out = '&'; return 1; }
    if (len == 4u && memcmp(name, "quot", 4u) == 0) { *out = '"'; return 1; }
    if (len == 4u && memcmp(name, "nbsp", 4u) == 0) { *out = ' '; return 1; }
    if (len == 4u && memcmp(name, "apos", 4u) == 0) { *out = '\''; return 1; }
    return 0;
}

static size_t decode_entities(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (input[i] == '&') {
            size_t semi = i + 1u;
            while (semi < len && semi - i <= 16u && input[semi] != ';') {
                semi++;
            }
            if (semi < len && input[semi] == ';') {
                char decoded = '\0';
                if (entity_value(input + i + 1u, semi - i - 1u, &decoded)) {
                    out[oi++] = decoded;
                    i = semi + 1u;
                    continue;
                }
            }
        }
        out[oi++] = input[i++];
    }
    return oi;
}

static size_t keep_between(const char *input, size_t len, const char *start, const char *end, char *out) {
    if (start == NULL || end == NULL || start[0] == '\0' || end[0] == '\0') {
        memcpy(out, input, len);
        return len;
    }
    const char *begin = strcasestr_local(input, len, start);
    if (begin == NULL) {
        return 0;
    }
    begin += strlen(start);
    size_t remain = len - (size_t)(begin - input);
    const char *finish = strcasestr_local(begin, remain, end);
    if (finish == NULL) {
        finish = input + len;
    }
    size_t n = (size_t)(finish - begin);
    memcpy(out, begin, n);
    return n;
}

static size_t strip_between(const char *input, size_t len, const char *start, const char *end, char *out) {
    if (start == NULL || end == NULL || start[0] == '\0' || end[0] == '\0') {
        memcpy(out, input, len);
        return len;
    }
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        const char *begin = strcasestr_local(input + i, len - i, start);
        if (begin == NULL) {
            size_t n = len - i;
            memcpy(out + oi, input + i, n);
            oi += n;
            break;
        }
        size_t prefix = (size_t)(begin - (input + i));
        memcpy(out + oi, input + i, prefix);
        oi += prefix;
        const char *finish = strcasestr_local(begin + strlen(start), len - (size_t)(begin - input) - strlen(start), end);
        if (finish == NULL) {
            break;
        }
        i = (size_t)(finish - input) + strlen(end);
    }
    return oi;
}

static size_t filter_lines_containing(const char *input, size_t len, const char *needle, char *out) {
    if (needle == NULL || needle[0] == '\0') {
        memcpy(out, input, len);
        return len;
    }
    size_t oi = 0;
    size_t line_start = 0;
    while (line_start < len) {
        size_t line_end = line_start;
        while (line_end < len && input[line_end] != '\n') {
            line_end++;
        }
        if (strcasestr_local(input + line_start, line_end - line_start, needle) != NULL) {
            size_t n = line_end - line_start;
            memcpy(out + oi, input + line_start, n);
            oi += n;
            out[oi++] = '\n';
        }
        line_start = line_end < len ? line_end + 1u : line_end;
    }
    return oi;
}

static int parse_positive_size(const char *text, size_t fallback, size_t *out) {
    if (out == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    if (text == NULL || text[0] == '\0') {
        *out = fallback;
        return AXIOM_STRIP_OK;
    }
    char *end = NULL;
    unsigned long long v = strtoull(text, &end, 10);
    if (end == text || v == 0u) {
        *out = fallback;
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    *out = (size_t)v;
    return AXIOM_STRIP_OK;
}

static size_t truncate_bytes(const char *input, size_t len, const char *limit_text, char *out) {
    size_t limit = len;
    if (parse_positive_size(limit_text, len, &limit) != AXIOM_STRIP_OK) {
        limit = len;
    }
    if (limit > len) {
        limit = len;
    }
    memcpy(out, input, limit);
    return limit;
}

static bool step_kind_known(const char *kind) {
    if (kind == NULL) {
        return true;
    }
    return strcmp(kind, "strip_tag") == 0 ||
           strcmp(kind, "collapse_ws") == 0 ||
           strcmp(kind, "remove") == 0 ||
           strcmp(kind, "replace") == 0 ||
           strcmp(kind, "strip_html") == 0 ||
           strcmp(kind, "strip_comments") == 0 ||
           strcmp(kind, "strip_attrs") == 0 ||
           strcmp(kind, "decode_entities") == 0 ||
           strcmp(kind, "keep_between") == 0 ||
           strcmp(kind, "strip_between") == 0 ||
           strcmp(kind, "filter_lines") == 0 ||
           strcmp(kind, "truncate") == 0;
}

static int run_step(const axiom_strip_step *step, const char *input, size_t in_len, char **out, size_t *out_len) {
    char *buf = (char *)pool_alloc((in_len * 2u) + 4096u);
    if (buf == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    if (step == NULL || step->kind == NULL) {
        memcpy(buf, input, in_len);
        *out = buf;
        *out_len = in_len;
        return AXIOM_STRIP_OK;
    }
    if (strcmp(step->kind, "strip_tag") == 0) {
        *out_len = strip_tag_block(input, in_len, step->pattern ? step->pattern : "script", buf);
    } else if (strcmp(step->kind, "collapse_ws") == 0) {
        *out_len = collapse_ws(input, in_len, buf);
    } else if (strcmp(step->kind, "remove") == 0) {
        *out_len = remove_literal(input, in_len, step->pattern ? step->pattern : "", buf);
    } else if (strcmp(step->kind, "replace") == 0) {
        *out_len = replace_literal(input, in_len, step->pattern ? step->pattern : "", step->replacement ? step->replacement : "", buf);
    } else if (strcmp(step->kind, "strip_html") == 0) {
        *out_len = strip_html_tags(input, in_len, buf);
    } else if (strcmp(step->kind, "strip_comments") == 0) {
        *out_len = strip_comments(input, in_len, buf);
    } else if (strcmp(step->kind, "strip_attrs") == 0) {
        *out_len = strip_attributes(input, in_len, buf);
    } else if (strcmp(step->kind, "decode_entities") == 0) {
        *out_len = decode_entities(input, in_len, buf);
    } else if (strcmp(step->kind, "keep_between") == 0) {
        *out_len = keep_between(input, in_len, step->pattern, step->replacement, buf);
    } else if (strcmp(step->kind, "strip_between") == 0) {
        *out_len = strip_between(input, in_len, step->pattern, step->replacement, buf);
    } else if (strcmp(step->kind, "filter_lines") == 0) {
        *out_len = filter_lines_containing(input, in_len, step->pattern ? step->pattern : "", buf);
    } else if (strcmp(step->kind, "truncate") == 0) {
        *out_len = truncate_bytes(input, in_len, step->pattern, buf);
    } else {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    *out = buf;
    return AXIOM_STRIP_OK;
}

int axiom_strip_validate_recipe(const axiom_strip_recipe *recipe) {
    if (recipe == NULL) {
        return AXIOM_STRIP_OK;
    }
    if (recipe->step_count > AXIOM_STRIP_MAX_STEPS) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    if (recipe->step_count > 0 && recipe->steps == NULL) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    if (recipe->max_output_ratio < 0.0) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    for (size_t i = 0; i < recipe->step_count; ++i) {
        const axiom_strip_step *step = &recipe->steps[i];
        if (!step_kind_known(step->kind)) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
        if ((strcmp(step->kind ? step->kind : "", "keep_between") == 0 ||
             strcmp(step->kind ? step->kind : "", "strip_between") == 0) &&
            (step->pattern == NULL || step->replacement == NULL)) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
    }
    return AXIOM_STRIP_OK;
}

int axiom_strip_measure(
    const uint8_t *input,
    size_t input_len,
    const uint8_t *output,
    size_t output_len,
    axiom_strip_metrics *metrics
) {
    if (metrics == NULL || (input == NULL && input_len > 0u) || (output == NULL && output_len > 0u)) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(metrics, 0, sizeof(*metrics));
    metrics->input_bytes = input_len;
    metrics->output_bytes = output_len;
    bool in_token = false;
    for (size_t i = 0; i < output_len; ++i) {
        unsigned char c = output[i];
        if (isalpha(c)) {
            metrics->ascii_letters++;
        } else if (isdigit(c)) {
            metrics->ascii_digits++;
        } else if (isspace(c)) {
            metrics->whitespace++;
        } else if (ispunct(c)) {
            metrics->punctuation++;
        }
        if (c == '\n') {
            metrics->line_count++;
        }
        if (isspace(c)) {
            in_token = false;
        } else if (!in_token) {
            metrics->token_count++;
            in_token = true;
        }
    }
    if (output_len > 0u && (output[output_len - 1u] != '\n')) {
        metrics->line_count++;
    }
    if (input_len > 0u) {
        metrics->output_ratio = (double)output_len / (double)input_len;
    }
    if (output_len > 0u) {
        metrics->signal_density = (double)(metrics->ascii_letters + metrics->ascii_digits) / (double)output_len;
    }
    return AXIOM_STRIP_OK;
}

int axiom_strip_apply_default_html(
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t output_capacity,
    axiom_strip_result *result
) {
    axiom_strip_step steps[] = {
        {.kind = "strip_comments", .pattern = "", .replacement = ""},
        {.kind = "strip_tag", .pattern = "script", .replacement = ""},
        {.kind = "strip_tag", .pattern = "style", .replacement = ""},
        {.kind = "strip_tag", .pattern = "noscript", .replacement = ""},
        {.kind = "strip_attrs", .pattern = "", .replacement = ""},
        {.kind = "decode_entities", .pattern = "", .replacement = ""},
        {.kind = "strip_html", .pattern = "", .replacement = ""},
        {.kind = "collapse_ws", .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = sizeof(steps) / sizeof(steps[0]), .max_output_ratio = AXIOM_STRIP_DEFAULT_RATIO};
    return axiom_strip_apply(input, input_len, &recipe, output, output_capacity, result);
}

int axiom_strip_apply(
    const uint8_t *input,
    size_t input_len,
    const axiom_strip_recipe *recipe,
    uint8_t *output,
    size_t output_capacity,
    axiom_strip_result *result
) {
    if (input == NULL || output == NULL || result == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int recipe_code = axiom_strip_validate_recipe(recipe);
    if (recipe_code != AXIOM_STRIP_OK) {
        result->code = recipe_code;
        return recipe_code;
    }
    int init = pool_init();
    if (init != AXIOM_STRIP_OK) {
        return init;
    }
    memset(result, 0, sizeof(*result));
    result->input_bytes = input_len;
    const char *current = (const char *)input;
    size_t current_len = input_len;
    if (recipe != NULL && recipe->steps != NULL) {
        for (size_t i = 0; i < recipe->step_count; ++i) {
            char *next = NULL;
            size_t next_len = 0;
            int code = run_step(&recipe->steps[i], current, current_len, &next, &next_len);
            if (code != AXIOM_STRIP_OK) {
                result->code = code;
                return code;
            }
            current = next;
            current_len = next_len;
        }
    }
    double ratio = recipe != NULL && recipe->max_output_ratio > 0.0 ? recipe->max_output_ratio : 1.0;
    if (input_len > 0 && (double)current_len > (double)input_len * ratio) {
        result->code = AXIOM_STRIP_ERR_REDUCTION_RATIO;
        return result->code;
    }
    if (current_len > output_capacity) {
        result->code = AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
        return result->code;
    }
    memcpy(output, current, current_len);
    result->bytes_written = current_len;
    result->crc32 = axiom_crc32(output, current_len);
    result->code = AXIOM_STRIP_OK;
    return AXIOM_STRIP_OK;
}
