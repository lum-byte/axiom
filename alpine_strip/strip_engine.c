/*
 * strip_engine_part1.c â€” Pool allocator, CRC32, string utilities.
 *
 * Provides the per-thread bump allocator (64 MB default), a table-driven
 * CRC32 implementation, and case-insensitive string search helpers used
 * throughout the strip engine.
 */

#include "strip_engine_internal.h"

/* ------------------------------------------------------------------ */
/*  CRC32 lookup table (polynomial 0xEDB88320)                         */
/* ------------------------------------------------------------------ */

static uint32_t g_crc32_table[256];
static int      g_crc32_table_ready = 0;

static void crc32_build_table(void) {
    for (uint32_t i = 0; i < 256; ++i) {
        uint32_t crc = i;
        for (int bit = 0; bit < 8; ++bit) {
            if (crc & 1u) {
                crc = (crc >> 1u) ^ 0xEDB88320u;
            } else {
                crc >>= 1u;
            }
        }
        g_crc32_table[i] = crc;
    }
    g_crc32_table_ready = 1;
}

uint32_t axiom_crc32(const uint8_t *data, size_t len) {
    if (!g_crc32_table_ready) {
        crc32_build_table();
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < len; ++i) {
        uint8_t idx = (uint8_t)(crc ^ data[i]);
        crc = (crc >> 8u) ^ g_crc32_table[idx];
    }
    return ~crc;
}

uint32_t axiom_crc32_combine(uint32_t crc_a, const uint8_t *data, size_t len) {
    if (!g_crc32_table_ready) {
        crc32_build_table();
    }
    uint32_t crc = ~crc_a;
    for (size_t i = 0; i < len; ++i) {
        uint8_t idx = (uint8_t)(crc ^ data[i]);
        crc = (crc >> 8u) ^ g_crc32_table[idx];
    }
    return ~crc;
}

/* ------------------------------------------------------------------ */
/*  Pool allocator                                                     */
/* ------------------------------------------------------------------ */

int strip_pool_init(strip_pool *pool, size_t capacity_bytes) {
    if (pool == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    if (capacity_bytes == 0) {
        capacity_bytes = AXIOM_POOL_BYTES;
    }
    pool->base = (uint8_t *)malloc(capacity_bytes);
    if (pool->base == NULL) {
        pool->capacity = 0;
        pool->offset = 0;
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    pool->capacity = capacity_bytes;
    pool->offset = 0;
    return AXIOM_STRIP_OK;
}

void strip_pool_reset(strip_pool *pool) {
    if (pool != NULL) {
        pool->offset = 0;
    }
}

void strip_pool_destroy(strip_pool *pool) {
    if (pool != NULL && pool->base != NULL) {
        free(pool->base);
        pool->base = NULL;
        pool->capacity = 0;
        pool->offset = 0;
    }
}

void *strip_pool_alloc(strip_pool *pool, size_t n) {
    if (pool == NULL || pool->base == NULL) {
        return NULL;
    }
    if (n == 0) {
        return pool->base + pool->offset;
    }
    size_t aligned = (n + 15u) & ~((size_t)15u);
    if (pool->offset + aligned > pool->capacity) {
        return NULL;
    }
    void *ptr = pool->base + pool->offset;
    pool->offset += aligned;
    return ptr;
}

size_t strip_pool_used(const strip_pool *pool) {
    if (pool == NULL) {
        return 0;
    }
    return pool->offset;
}

size_t strip_pool_remaining(const strip_pool *pool) {
    if (pool == NULL || pool->base == NULL) {
        return 0;
    }
    return pool->capacity - pool->offset;
}

void *strip_pool_alloc_zero(strip_pool *pool, size_t n) {
    void *ptr = strip_pool_alloc(pool, n);
    if (ptr != NULL && n > 0) {
        memset(ptr, 0, n);
    }
    return ptr;
}

char *strip_pool_strdup(strip_pool *pool, const char *s) {
    if (s == NULL) {
        return NULL;
    }
    size_t len = strlen(s);
    char *dup = (char *)strip_pool_alloc(pool, len + 1);
    if (dup != NULL) {
        memcpy(dup, s, len + 1);
    }
    return dup;
}

char *strip_pool_strndup(strip_pool *pool, const char *s, size_t max_len) {
    if (s == NULL) {
        return NULL;
    }
    size_t len = 0;
    while (len < max_len && s[len] != '\0') {
        len++;
    }
    char *dup = (char *)strip_pool_alloc(pool, len + 1);
    if (dup != NULL) {
        memcpy(dup, s, len);
        dup[len] = '\0';
    }
    return dup;
}

/* ------------------------------------------------------------------ */
/*  Case-insensitive string helpers                                    */
/* ------------------------------------------------------------------ */

bool si_starts_with_ci(const char *s, size_t remain, const char *needle) {
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

const char *si_strcasestr(const char *haystack, size_t hay_len, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) {
        return haystack;
    }
    if (hay_len < n) {
        return NULL;
    }
    unsigned char first_lower = (unsigned char)tolower((unsigned char)needle[0]);
    size_t limit = hay_len - n;
    for (size_t i = 0; i <= limit; ++i) {
        if ((unsigned char)tolower((unsigned char)haystack[i]) != first_lower) {
            continue;
        }
        if (si_starts_with_ci(haystack + i, hay_len - i, needle)) {
            return haystack + i;
        }
    }
    return NULL;
}

static bool si_starts_with_exact(const char *s, size_t remain, const char *needle) {
    size_t n = strlen(needle);
    if (remain < n) {
        return false;
    }
    return memcmp(s, needle, n) == 0;
}

const char *si_strstr_bounded(const char *haystack, size_t hay_len, const char *needle) {
    size_t n = strlen(needle);
    if (n == 0) {
        return haystack;
    }
    if (hay_len < n) {
        return NULL;
    }
    size_t limit = hay_len - n;
    for (size_t i = 0; i <= limit; ++i) {
        if (si_starts_with_exact(haystack + i, hay_len - i, needle)) {
            return haystack + i;
        }
    }
    return NULL;
}

/* ------------------------------------------------------------------ */
/*  Error code to string                                               */
/* ------------------------------------------------------------------ */

const char *axiom_strip_strerror(int code) {
    switch (code) {
    case AXIOM_STRIP_OK:                   return "OK";
    case AXIOM_STRIP_ERR_INVALID_ARG:      return "invalid argument";
    case AXIOM_STRIP_ERR_BAD_RECIPE:       return "bad recipe";
    case AXIOM_STRIP_ERR_POOL_EXHAUSTED:   return "pool exhausted";
    case AXIOM_STRIP_ERR_REDUCTION_RATIO:  return "reduction ratio exceeded";
    case AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL: return "output buffer too small";
    case AXIOM_STRIP_ERR_PCRE2_COMPILE:    return "PCRE2 compile error";
    case AXIOM_STRIP_ERR_PCRE2_MATCH:      return "PCRE2 match error";
    case AXIOM_STRIP_ERR_MMAP_OPEN:        return "mmap open failed";
    case AXIOM_STRIP_ERR_MMAP_READ:        return "mmap read failed";
    case AXIOM_STRIP_ERR_CRC_MISMATCH:     return "CRC mismatch";
    case AXIOM_STRIP_ERR_SLOT_OOB:         return "slot index out of bounds";
    case AXIOM_STRIP_EMPTY:                return "empty output";
    case AXIOM_STRIP_TOO_BROAD:            return "strip too broad";
    default:                               return "unknown error";
    }
}

/* ------------------------------------------------------------------ */
/*  Bounded memchr / memrchr helpers                                   */
/* ------------------------------------------------------------------ */

const char *si_memchr_bounded(const char *buf, int c, size_t len) {
    const void *p = memchr(buf, c, len);
    return (const char *)p;
}

const char *si_memrchr_bounded(const char *buf, int c, size_t len) {
    for (size_t i = len; i > 0; --i) {
        if ((unsigned char)buf[i - 1] == (unsigned char)c) {
            return buf + i - 1;
        }
    }
    return NULL;
}

size_t si_count_char(const char *buf, size_t len, char c) {
    size_t count = 0;
    for (size_t i = 0; i < len; ++i) {
        if (buf[i] == c) {
            count++;
        }
    }
    return count;
}

size_t si_count_lines(const char *buf, size_t len) {
    if (len == 0) {
        return 0;
    }
    size_t lines = si_count_char(buf, len, '\n');
    if (buf[len - 1] != '\n') {
        lines++;
    }
    return lines;
}

bool si_is_blank_line(const char *line, size_t len) {
    for (size_t i = 0; i < len; ++i) {
        if (!isspace((unsigned char)line[i])) {
            return false;
        }
    }
    return true;
}

size_t si_ltrim(const char *s, size_t len) {
    size_t i = 0;
    while (i < len && isspace((unsigned char)s[i])) {
        i++;
    }
    return i;
}

size_t si_rtrim(const char *s, size_t len) {
    while (len > 0 && isspace((unsigned char)s[len - 1])) {
        len--;
    }
    return len;
}

/*
 * strip_engine_part2.c â€” HTML processing functions.
 *
 * Strip tag blocks, HTML tags, comments, attributes, entities,
 * inline styles, and data-* attributes.  All operate on raw byte
 * buffers with explicit lengths (no NUL requirement).
 */


/* ------------------------------------------------------------------ */
/*  Strip a named tag block:  <tag ...>...</tag>                       */
/* ------------------------------------------------------------------ */

size_t si_strip_tag_block(const char *input, size_t len, const char *tag, char *out) {
    size_t oi = 0;
    char open[64];
    char close[64];
    snprintf(open, sizeof(open), "<%s", tag);
    snprintf(close, sizeof(close), "</%s>", tag);
    for (size_t i = 0; i < len;) {
        if (si_starts_with_ci(input + i, len - i, open)) {
            size_t remain = len - i;
            if (remain > strlen(open)) {
                char after = input[i + strlen(open)];
                if (after != '>' && after != ' ' && after != '\t' &&
                    after != '\n' && after != '\r' && after != '/') {
                    out[oi++] = input[i++];
                    continue;
                }
            }
            const char *end = si_strcasestr(input + i, len - i, close);
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

/* ------------------------------------------------------------------ */
/*  Strip all HTML tags (preserving text content)                      */
/* ------------------------------------------------------------------ */

size_t si_strip_html_tags(const char *input, size_t len, char *out) {
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

/* ------------------------------------------------------------------ */
/*  Strip HTML comments:  <!-- ... -->                                 */
/* ------------------------------------------------------------------ */

size_t si_strip_comments(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (i + 4u <= len && memcmp(input + i, "<!--", 4u) == 0) {
            const char *end = si_strcasestr(input + i + 4u, len - i - 4u, "-->");
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

/* ------------------------------------------------------------------ */
/*  Strip CDATA sections:  <![CDATA[ ... ]]>                           */
/* ------------------------------------------------------------------ */

size_t si_strip_cdata(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (i + 9u <= len && memcmp(input + i, "<![CDATA[", 9u) == 0) {
            const char *end = si_strcasestr(input + i + 9u, len - i - 9u, "]]>");
            if (end == NULL) {
                size_t tail = len - (i + 9u);
                memcpy(out + oi, input + i + 9u, tail);
                oi += tail;
                return oi;
            }
            size_t content_len = (size_t)(end - (input + i + 9u));
            memcpy(out + oi, input + i + 9u, content_len);
            oi += content_len;
            i = (size_t)(end - input) + 3u;
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Strip attributes from tags: <div class="x"> â†’ <div>               */
/* ------------------------------------------------------------------ */

size_t si_strip_attributes(const char *input, size_t len, char *out) {
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

/* ------------------------------------------------------------------ */
/*  HTML entity decoding                                               */
/* ------------------------------------------------------------------ */

static const struct {
    const char *name;
    size_t      name_len;
    char        decoded;
} g_entities[] = {
    {"lt",     2, '<'},
    {"gt",     2, '>'},
    {"amp",    3, '&'},
    {"quot",   4, '"'},
    {"nbsp",   4, ' '},
    {"apos",   4, '\''},
    {"copy",   4, '?'},
    {"reg",    3, '?'},
    {"trade",  5, '?'},
    {"mdash",  5, '-'},
    {"ndash",  5, '-'},
    {"laquo",  5, '<'},
    {"raquo",  5, '>'},
    {"bull",   4, '*'},
    {"hellip", 6, '.'},
    {"prime",  5, '\''},
    {"Prime",  5, '"'},
    {"lsquo",  5, '\''},
    {"rsquo",  5, '\''},
    {"ldquo",  5, '"'},
    {"rdquo",  5, '"'},
    {"minus",  5, '-'},
    {"times",  5, 'x'},
    {"divide", 6, '/'},
    {"cent",   4, 'c'},
    {"pound",  5, '#'},
    {"yen",    3, 'Y'},
    {"euro",   4, 'E'},
    {"sect",   4, 'S'},
    {"para",   4, 'P'},
    {"deg",    3, 'o'},
    {"plusmn", 6, '+'},
    {"frac12", 6, ' '},
    {"frac14", 6, ' '},
    {"frac34", 6, ' '},
    {NULL,     0, '\0'}
};

int si_entity_value(const char *name, size_t len, char *out) {
    for (int i = 0; g_entities[i].name != NULL; ++i) {
        if (len == g_entities[i].name_len &&
            memcmp(name, g_entities[i].name, len) == 0) {
            *out = g_entities[i].decoded;
            return 1;
        }
    }
    if (len >= 2 && name[0] == '#') {
        unsigned long code = 0;
        if (name[1] == 'x' || name[1] == 'X') {
            for (size_t i = 2; i < len; ++i) {
                char c = name[i];
                if (c >= '0' && c <= '9') code = code * 16 + (unsigned long)(c - '0');
                else if (c >= 'a' && c <= 'f') code = code * 16 + 10 + (unsigned long)(c - 'a');
                else if (c >= 'A' && c <= 'F') code = code * 16 + 10 + (unsigned long)(c - 'A');
                else return 0;
            }
        } else {
            for (size_t i = 1; i < len; ++i) {
                char c = name[i];
                if (c < '0' || c > '9') return 0;
                code = code * 10 + (unsigned long)(c - '0');
            }
        }
        if (code > 0 && code < 128) {
            *out = (char)code;
            return 1;
        }
        if (code >= 128 && code <= 0xFFFF) {
            *out = '?';
            return 1;
        }
        return 0;
    }
    return 0;
}

size_t si_decode_entities(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (input[i] == '&') {
            size_t semi = i + 1u;
            while (semi < len && semi - i <= 16u && input[semi] != ';') {
                semi++;
            }
            if (semi < len && input[semi] == ';') {
                char decoded = '\0';
                if (si_entity_value(input + i + 1u, semi - i - 1u, &decoded)) {
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

/* ------------------------------------------------------------------ */
/*  Strip inline style attributes                                      */
/* ------------------------------------------------------------------ */

size_t si_strip_inline_styles(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        if (i + 6u <= len && si_starts_with_ci(input + i, len - i, "style=")) {
            size_t j = i + 6u;
            if (j < len && (input[j] == '"' || input[j] == '\'')) {
                char q = input[j];
                j++;
                while (j < len && input[j] != q) {
                    j++;
                }
                if (j < len) j++;
                i = j;
                continue;
            }
            while (j < len && !isspace((unsigned char)input[j]) && input[j] != '>') {
                j++;
            }
            i = j;
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Strip data-* attributes from within tags                           */
/* ------------------------------------------------------------------ */

size_t si_strip_data_attrs(const char *input, size_t len, char *out) {
    size_t oi = 0;
    bool in_tag = false;
    for (size_t i = 0; i < len;) {
        if (!in_tag) {
            if (input[i] == '<') {
                in_tag = true;
            }
            out[oi++] = input[i++];
            continue;
        }
        if (input[i] == '>') {
            in_tag = false;
            out[oi++] = input[i++];
            continue;
        }
        if (i + 5u <= len && si_starts_with_ci(input + i, len - i, "data-")) {
            while (i < len && input[i] != '>' && !isspace((unsigned char)input[i])) {
                i++;
            }
            if (i < len && input[i] == '=') {
                i++;
                if (i < len && (input[i] == '"' || input[i] == '\'')) {
                    char q = input[i++];
                    while (i < len && input[i] != q) i++;
                    if (i < len) i++;
                } else {
                    while (i < len && !isspace((unsigned char)input[i]) && input[i] != '>') {
                        i++;
                    }
                }
            }
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Strip <noscript> blocks                                            */
/* ------------------------------------------------------------------ */

size_t si_strip_noscript(const char *input, size_t len, char *out) {
    return si_strip_tag_block(input, len, "noscript", out);
}

/* ------------------------------------------------------------------ */
/*  Strip <iframe> blocks                                              */
/* ------------------------------------------------------------------ */

size_t si_strip_iframe(const char *input, size_t len, char *out) {
    return si_strip_tag_block(input, len, "iframe", out);
}

/* ------------------------------------------------------------------ */
/*  Strip <svg> blocks                                                 */
/* ------------------------------------------------------------------ */

size_t si_strip_svg(const char *input, size_t len, char *out) {
    return si_strip_tag_block(input, len, "svg", out);
}

/* ------------------------------------------------------------------ */
/*  Strip <template> blocks                                            */
/* ------------------------------------------------------------------ */

size_t si_strip_template(const char *input, size_t len, char *out) {
    return si_strip_tag_block(input, len, "template", out);
}

/*
 * strip_engine_part3.c â€” Text manipulation functions.
 *
 * Whitespace collapsing, literal removal/replacement, region extraction,
 * line filtering, truncation, deduplication, trimming, and text-run
 * extraction.  All operate on explicit-length buffers.
 */


/* ------------------------------------------------------------------ */
/*  Whitespace collapse                                                */
/* ------------------------------------------------------------------ */

size_t si_collapse_ws(const char *input, size_t len, char *out) {
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

/* ------------------------------------------------------------------ */
/*  Remove all occurrences of a literal string                         */
/* ------------------------------------------------------------------ */

size_t si_remove_literal(const char *input, size_t len, const char *needle, char *out) {
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

/* ------------------------------------------------------------------ */
/*  Replace all occurrences of a literal string                        */
/* ------------------------------------------------------------------ */

size_t si_replace_literal(const char *input, size_t len,
                          const char *needle, const char *repl, char *out) {
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

/* ------------------------------------------------------------------ */
/*  Keep content between first occurrence of start..end markers        */
/* ------------------------------------------------------------------ */

size_t si_keep_between(const char *input, size_t len,
                       const char *start, const char *end, char *out) {
    if (start == NULL || end == NULL || start[0] == '\0' || end[0] == '\0') {
        memcpy(out, input, len);
        return len;
    }
    const char *begin = si_strcasestr(input, len, start);
    if (begin == NULL) {
        return 0;
    }
    begin += strlen(start);
    size_t remain = len - (size_t)(begin - input);
    const char *finish = si_strcasestr(begin, remain, end);
    if (finish == NULL) {
        finish = input + len;
    }
    size_t n = (size_t)(finish - begin);
    memcpy(out, begin, n);
    return n;
}

/* ------------------------------------------------------------------ */
/*  Keep all regions between repeated start..end markers               */
/* ------------------------------------------------------------------ */

size_t si_keep_between_all(const char *input, size_t len,
                           const char *start, const char *end, char *out) {
    if (start == NULL || end == NULL || start[0] == '\0' || end[0] == '\0') {
        memcpy(out, input, len);
        return len;
    }
    size_t oi = 0;
    size_t slen = strlen(start);
    size_t elen = strlen(end);
    size_t pos = 0;
    while (pos < len) {
        const char *begin = si_strcasestr(input + pos, len - pos, start);
        if (begin == NULL) break;
        begin += slen;
        size_t remain = len - (size_t)(begin - input);
        const char *finish = si_strcasestr(begin, remain, end);
        if (finish == NULL) {
            size_t n = len - (size_t)(begin - input);
            memcpy(out + oi, begin, n);
            oi += n;
            break;
        }
        size_t n = (size_t)(finish - begin);
        memcpy(out + oi, begin, n);
        oi += n;
        if (oi > 0 && out[oi - 1] != '\n') {
            out[oi++] = '\n';
        }
        pos = (size_t)(finish - input) + elen;
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Remove content between start..end markers (all occurrences)        */
/* ------------------------------------------------------------------ */

size_t si_strip_between(const char *input, size_t len,
                        const char *start, const char *end, char *out) {
    if (start == NULL || end == NULL || start[0] == '\0' || end[0] == '\0') {
        memcpy(out, input, len);
        return len;
    }
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        const char *begin = si_strcasestr(input + i, len - i, start);
        if (begin == NULL) {
            size_t n = len - i;
            memcpy(out + oi, input + i, n);
            oi += n;
            break;
        }
        size_t prefix = (size_t)(begin - (input + i));
        memcpy(out + oi, input + i, prefix);
        oi += prefix;
        const char *finish = si_strcasestr(
            begin + strlen(start),
            len - (size_t)(begin - input) - strlen(start),
            end
        );
        if (finish == NULL) {
            break;
        }
        i = (size_t)(finish - input) + strlen(end);
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Filter: keep lines containing needle                               */
/* ------------------------------------------------------------------ */

size_t si_filter_lines_containing(const char *input, size_t len,
                                  const char *needle, char *out) {
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
        if (si_strcasestr(input + line_start, line_end - line_start, needle) != NULL) {
            size_t n = line_end - line_start;
            memcpy(out + oi, input + line_start, n);
            oi += n;
            out[oi++] = '\n';
        }
        line_start = line_end < len ? line_end + 1u : line_end;
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Filter: keep lines NOT containing needle                           */
/* ------------------------------------------------------------------ */

size_t si_filter_lines_not_containing(const char *input, size_t len,
                                      const char *needle, char *out) {
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
        if (si_strcasestr(input + line_start, line_end - line_start, needle) == NULL) {
            size_t n = line_end - line_start;
            memcpy(out + oi, input + line_start, n);
            oi += n;
            out[oi++] = '\n';
        }
        line_start = line_end < len ? line_end + 1u : line_end;
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Truncate to a byte limit                                           */
/* ------------------------------------------------------------------ */

int si_parse_positive_size(const char *text, size_t fallback, size_t *out) {
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

size_t si_truncate_bytes(const char *input, size_t len,
                         const char *limit_text, char *out) {
    size_t limit = len;
    if (si_parse_positive_size(limit_text, len, &limit) != AXIOM_STRIP_OK) {
        limit = len;
    }
    if (limit > len) {
        limit = len;
    }
    memcpy(out, input, limit);
    return limit;
}

/* ------------------------------------------------------------------ */
/*  Normalize unicode whitespace to ASCII spaces                       */
/* ------------------------------------------------------------------ */

size_t si_normalize_unicode_ws(const char *input, size_t len, char *out) {
    size_t oi = 0;
    for (size_t i = 0; i < len;) {
        unsigned char c = (unsigned char)input[i];
        if (c == 0xC2 && i + 1 < len && (unsigned char)input[i + 1] == 0xA0) {
            out[oi++] = ' ';
            i += 2;
            continue;
        }
        if (c == 0xE2 && i + 2 < len) {
            unsigned char b1 = (unsigned char)input[i + 1];
            unsigned char b2 = (unsigned char)input[i + 2];
            if (b1 == 0x80 && (b2 >= 0x80 && b2 <= 0x8A)) {
                out[oi++] = ' ';
                i += 3;
                continue;
            }
            if (b1 == 0x80 && b2 == 0xAF) {
                out[oi++] = ' ';
                i += 3;
                continue;
            }
            if (b1 == 0x81 && b2 == 0x9F) {
                out[oi++] = ' ';
                i += 3;
                continue;
            }
        }
        if (c == 0xE3 && i + 2 < len &&
            (unsigned char)input[i + 1] == 0x80 &&
            (unsigned char)input[i + 2] == 0x80) {
            out[oi++] = ' ';
            i += 3;
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Deduplicate consecutive identical lines                            */
/* ------------------------------------------------------------------ */

size_t si_deduplicate_lines(const char *input, size_t len, char *out) {
    size_t oi = 0;
    const char *prev_line = NULL;
    size_t prev_len = 0;
    size_t line_start = 0;
    while (line_start < len) {
        size_t line_end = line_start;
        while (line_end < len && input[line_end] != '\n') {
            line_end++;
        }
        size_t line_len = line_end - line_start;
        bool is_dup = false;
        if (prev_line != NULL && line_len == prev_len) {
            is_dup = (memcmp(input + line_start, prev_line, line_len) == 0);
        }
        if (!is_dup) {
            memcpy(out + oi, input + line_start, line_len);
            oi += line_len;
            out[oi++] = '\n';
            prev_line = input + line_start;
            prev_len = line_len;
        }
        line_start = line_end < len ? line_end + 1 : line_end;
    }
    if (oi > 0 && out[oi - 1] == '\n' && len > 0 && input[len - 1] != '\n') {
        oi--;
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Trim leading/trailing whitespace from each line                    */
/* ------------------------------------------------------------------ */

size_t si_trim_lines(const char *input, size_t len, char *out) {
    size_t oi = 0;
    size_t line_start = 0;
    while (line_start < len) {
        size_t line_end = line_start;
        while (line_end < len && input[line_end] != '\n') {
            line_end++;
        }
        size_t ls = line_start;
        while (ls < line_end && isspace((unsigned char)input[ls])) ls++;
        size_t le = line_end;
        while (le > ls && isspace((unsigned char)input[le - 1])) le--;
        if (le > ls) {
            memcpy(out + oi, input + ls, le - ls);
            oi += le - ls;
        }
        out[oi++] = '\n';
        line_start = line_end < len ? line_end + 1 : line_end;
    }
    if (oi > 0 && out[oi - 1] == '\n' && len > 0 && input[len - 1] != '\n') {
        oi--;
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Extract text runs of at least min_run printable characters         */
/* ------------------------------------------------------------------ */

size_t si_extract_text_runs(const char *input, size_t len,
                            size_t min_run, char *out) {
    if (min_run == 0) min_run = 4;
    size_t oi = 0;
    size_t run_start = 0;
    bool in_run = false;
    for (size_t i = 0; i <= len; ++i) {
        bool printable = (i < len) &&
                         ((unsigned char)input[i] >= 32) &&
                         ((unsigned char)input[i] < 127);
        if (printable && !in_run) {
            run_start = i;
            in_run = true;
        } else if (!printable && in_run) {
            size_t run_len = i - run_start;
            if (run_len >= min_run) {
                memcpy(out + oi, input + run_start, run_len);
                oi += run_len;
                out[oi++] = '\n';
            }
            in_run = false;
        }
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Remove blank lines                                                 */
/* ------------------------------------------------------------------ */

size_t si_remove_blank_lines(const char *input, size_t len, char *out) {
    size_t oi = 0;
    size_t line_start = 0;
    while (line_start < len) {
        size_t line_end = line_start;
        while (line_end < len && input[line_end] != '\n') {
            line_end++;
        }
        if (!si_is_blank_line(input + line_start, line_end - line_start)) {
            memcpy(out + oi, input + line_start, line_end - line_start);
            oi += line_end - line_start;
            out[oi++] = '\n';
        }
        line_start = line_end < len ? line_end + 1 : line_end;
    }
    if (oi > 0 && out[oi - 1] == '\n' && len > 0 && input[len - 1] != '\n') {
        oi--;
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Squeeze consecutive blank lines to at most one                     */
/* ------------------------------------------------------------------ */

size_t si_squeeze_blank_lines(const char *input, size_t len, char *out) {
    size_t oi = 0;
    bool prev_blank = false;
    size_t line_start = 0;
    while (line_start < len) {
        size_t line_end = line_start;
        while (line_end < len && input[line_end] != '\n') {
            line_end++;
        }
        bool blank = si_is_blank_line(input + line_start, line_end - line_start);
        if (blank && prev_blank) {
            line_start = line_end < len ? line_end + 1 : line_end;
            continue;
        }
        memcpy(out + oi, input + line_start, line_end - line_start);
        oi += line_end - line_start;
        out[oi++] = '\n';
        prev_blank = blank;
        line_start = line_end < len ? line_end + 1 : line_end;
    }
    if (oi > 0 && out[oi - 1] == '\n' && len > 0 && input[len - 1] != '\n') {
        oi--;
    }
    return oi;
}

/*
 * strip_engine_part4.c â€” PCRE2 regex engine integration.
 *
 * Wraps libpcre2-8 to provide compile, match, replace, and extract
 * operations.  Supports JIT compilation hints for hot patterns.
 *
 * When PCRE2 is not available (AXIOM_NO_PCRE2 defined), falls back
 * to literal substring matching via the si_strcasestr helpers.
 */


#ifndef AXIOM_NO_PCRE2

#define PCRE2_CODE_UNIT_WIDTH 8
#include <pcre2.h>

/* ------------------------------------------------------------------ */
/*  Opaque regex handle                                                */
/* ------------------------------------------------------------------ */

struct axiom_regex {
    pcre2_code       *code;
    pcre2_match_data *match_data;
    uint32_t          capture_count;
    int               jit_ready;
};

/* ------------------------------------------------------------------ */
/*  Compile                                                            */
/* ------------------------------------------------------------------ */

axiom_regex *axiom_regex_compile(const char *pattern, size_t pattern_len,
                                 uint32_t options, int *errcode) {
    if (pattern == NULL || pattern_len == 0) {
        if (errcode) *errcode = AXIOM_STRIP_ERR_INVALID_ARG;
        return NULL;
    }
    int ec = 0;
    PCRE2_SIZE eoff = 0;
    uint32_t pcre2_opts = PCRE2_UTF;
    if (options & 0x01u) pcre2_opts |= PCRE2_CASELESS;
    if (options & 0x02u) pcre2_opts |= PCRE2_MULTILINE;
    if (options & 0x04u) pcre2_opts |= PCRE2_DOTALL;
    if (options & 0x08u) pcre2_opts |= PCRE2_EXTENDED;

    pcre2_code *code = pcre2_compile(
        (PCRE2_SPTR)pattern, (PCRE2_SIZE)pattern_len,
        pcre2_opts, &ec, &eoff, NULL
    );
    if (code == NULL) {
        if (errcode) *errcode = AXIOM_STRIP_ERR_PCRE2_COMPILE;
        return NULL;
    }
    uint32_t cap = 0;
    pcre2_pattern_info(code, PCRE2_INFO_CAPTURECOUNT, &cap);

    pcre2_match_data *md = pcre2_match_data_create_from_pattern(code, NULL);
    if (md == NULL) {
        pcre2_code_free(code);
        if (errcode) *errcode = AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        return NULL;
    }
    axiom_regex *re = (axiom_regex *)malloc(sizeof(axiom_regex));
    if (re == NULL) {
        pcre2_match_data_free(md);
        pcre2_code_free(code);
        if (errcode) *errcode = AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        return NULL;
    }
    re->code = code;
    re->match_data = md;
    re->capture_count = cap;
    re->jit_ready = 0;
    if (errcode) *errcode = AXIOM_STRIP_OK;
    return re;
}

/* ------------------------------------------------------------------ */
/*  JIT hint                                                           */
/* ------------------------------------------------------------------ */

int axiom_regex_jit_hint(axiom_regex *re) {
    if (re == NULL || re->code == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int rc = pcre2_jit_compile(re->code, PCRE2_JIT_COMPLETE);
    if (rc == 0) {
        re->jit_ready = 1;
    }
    return rc == 0 ? AXIOM_STRIP_OK : AXIOM_STRIP_ERR_PCRE2_COMPILE;
}

/* ------------------------------------------------------------------ */
/*  Match                                                              */
/* ------------------------------------------------------------------ */

int axiom_regex_match(const axiom_regex *re, const char *subject,
                      size_t subject_len, size_t start_offset,
                      size_t *ovector, size_t ovector_count,
                      size_t *match_count) {
    if (re == NULL || subject == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int rc;
    if (re->jit_ready) {
        rc = pcre2_jit_match(
            re->code, (PCRE2_SPTR)subject, (PCRE2_SIZE)subject_len,
            (PCRE2_SIZE)start_offset, 0, re->match_data, NULL
        );
    } else {
        rc = pcre2_match(
            re->code, (PCRE2_SPTR)subject, (PCRE2_SIZE)subject_len,
            (PCRE2_SIZE)start_offset, 0, re->match_data, NULL
        );
    }
    if (rc < 0) {
        if (rc == PCRE2_ERROR_NOMATCH) {
            if (match_count) *match_count = 0;
            return AXIOM_STRIP_OK;
        }
        return AXIOM_STRIP_ERR_PCRE2_MATCH;
    }
    PCRE2_SIZE *ov = pcre2_get_ovector_pointer(re->match_data);
    size_t pairs = (size_t)rc;
    if (match_count) *match_count = pairs;
    if (ovector != NULL && ovector_count > 0) {
        size_t copy = pairs * 2;
        if (copy > ovector_count) copy = ovector_count;
        for (size_t i = 0; i < copy; ++i) {
            ovector[i] = (size_t)ov[i];
        }
    }
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Replace all matches                                                */
/* ------------------------------------------------------------------ */

size_t axiom_regex_replace(const axiom_regex *re, const char *subject,
                           size_t subject_len, const char *replacement,
                           char *output, size_t output_capacity) {
    if (re == NULL || subject == NULL || output == NULL) {
        return 0;
    }
    size_t repl_len = replacement ? strlen(replacement) : 0;
    size_t oi = 0;
    size_t pos = 0;
    while (pos < subject_len) {
        int rc;
        if (re->jit_ready) {
            rc = pcre2_jit_match(
                re->code, (PCRE2_SPTR)subject, (PCRE2_SIZE)subject_len,
                (PCRE2_SIZE)pos, 0, re->match_data, NULL
            );
        } else {
            rc = pcre2_match(
                re->code, (PCRE2_SPTR)subject, (PCRE2_SIZE)subject_len,
                (PCRE2_SIZE)pos, 0, re->match_data, NULL
            );
        }
        if (rc < 1) break;
        PCRE2_SIZE *ov = pcre2_get_ovector_pointer(re->match_data);
        size_t match_start = (size_t)ov[0];
        size_t match_end = (size_t)ov[1];
        size_t prefix_len = match_start - pos;
        if (oi + prefix_len > output_capacity) break;
        memcpy(output + oi, subject + pos, prefix_len);
        oi += prefix_len;
        if (replacement != NULL && repl_len > 0) {
            if (oi + repl_len > output_capacity) break;
            memcpy(output + oi, replacement, repl_len);
            oi += repl_len;
        }
        if (match_end == match_start) {
            if (oi + 1 > output_capacity) break;
            if (pos < subject_len) {
                output[oi++] = subject[pos];
            }
            pos = match_end + 1;
        } else {
            pos = match_end;
        }
    }
    if (pos < subject_len) {
        size_t tail = subject_len - pos;
        if (oi + tail <= output_capacity) {
            memcpy(output + oi, subject + pos, tail);
            oi += tail;
        }
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Extract all matching groups (group 0 = full match)                 */
/* ------------------------------------------------------------------ */

size_t axiom_regex_extract_all(const axiom_regex *re, const char *subject,
                               size_t subject_len, char *output,
                               size_t output_capacity) {
    if (re == NULL || subject == NULL || output == NULL) {
        return 0;
    }
    size_t oi = 0;
    size_t pos = 0;
    while (pos < subject_len) {
        int rc;
        if (re->jit_ready) {
            rc = pcre2_jit_match(
                re->code, (PCRE2_SPTR)subject, (PCRE2_SIZE)subject_len,
                (PCRE2_SIZE)pos, 0, re->match_data, NULL
            );
        } else {
            rc = pcre2_match(
                re->code, (PCRE2_SPTR)subject, (PCRE2_SIZE)subject_len,
                (PCRE2_SIZE)pos, 0, re->match_data, NULL
            );
        }
        if (rc < 1) break;
        PCRE2_SIZE *ov = pcre2_get_ovector_pointer(re->match_data);
        size_t match_start = (size_t)ov[0];
        size_t match_end = (size_t)ov[1];
        size_t mlen = match_end - match_start;
        if (mlen > 0) {
            if (oi + mlen + 1 > output_capacity) break;
            memcpy(output + oi, subject + match_start, mlen);
            oi += mlen;
            output[oi++] = '\n';
        }
        if (match_end == match_start) {
            pos = match_end + 1;
        } else {
            pos = match_end;
        }
    }
    return oi;
}

/* ------------------------------------------------------------------ */
/*  Extract named capture group from last match                        */
/* ------------------------------------------------------------------ */

size_t axiom_regex_get_named(const axiom_regex *re, const char *subject,
                             const char *name, char *output,
                             size_t output_capacity) {
    if (re == NULL || subject == NULL || name == NULL || output == NULL) {
        return 0;
    }
    int idx = pcre2_substring_number_from_name(re->code, (PCRE2_SPTR)name);
    if (idx < 0) return 0;
    PCRE2_SIZE *ov = pcre2_get_ovector_pointer(re->match_data);
    size_t start = (size_t)ov[2 * idx];
    size_t end = (size_t)ov[2 * idx + 1];
    if (start == (size_t)PCRE2_UNSET || end == (size_t)PCRE2_UNSET) return 0;
    size_t mlen = end - start;
    if (mlen > output_capacity) mlen = output_capacity;
    memcpy(output, subject + start, mlen);
    return mlen;
}

/* ------------------------------------------------------------------ */
/*  Free                                                               */
/* ------------------------------------------------------------------ */

void axiom_regex_free(axiom_regex *re) {
    if (re == NULL) return;
    if (re->match_data) pcre2_match_data_free(re->match_data);
    if (re->code) pcre2_code_free(re->code);
    free(re);
}

/* ------------------------------------------------------------------ */
/*  Batch compile + match helper for step execution                    */
/* ------------------------------------------------------------------ */

size_t axiom_regex_strip_matches(const char *pattern, size_t pattern_len,
                                 const char *subject, size_t subject_len,
                                 char *output, size_t output_capacity) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) {
        if (subject_len <= output_capacity) {
            memcpy(output, subject, subject_len);
        }
        return subject_len <= output_capacity ? subject_len : 0;
    }
    size_t result = axiom_regex_replace(re, subject, subject_len, "", output, output_capacity);
    axiom_regex_free(re);
    return result;
}

size_t axiom_regex_extract_matches(const char *pattern, size_t pattern_len,
                                   const char *subject, size_t subject_len,
                                   char *output, size_t output_capacity) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) {
        return 0;
    }
    size_t result = axiom_regex_extract_all(re, subject, subject_len, output, output_capacity);
    axiom_regex_free(re);
    return result;
}

size_t axiom_regex_replace_matches(const char *pattern, size_t pattern_len,
                                   const char *subject, size_t subject_len,
                                   const char *replacement,
                                   char *output, size_t output_capacity) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) {
        if (subject_len <= output_capacity) {
            memcpy(output, subject, subject_len);
        }
        return subject_len <= output_capacity ? subject_len : 0;
    }
    size_t result = axiom_regex_replace(re, subject, subject_len, replacement,
                                        output, output_capacity);
    axiom_regex_free(re);
    return result;
}

int axiom_regex_test(const char *pattern, size_t pattern_len,
                     const char *subject, size_t subject_len) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) return -1;
    size_t mc = 0;
    int rc = axiom_regex_match(re, subject, subject_len, 0, NULL, 0, &mc);
    axiom_regex_free(re);
    if (rc != AXIOM_STRIP_OK) return -1;
    return mc > 0 ? 1 : 0;
}

#else /* AXIOM_NO_PCRE2 â€” built-in regex engine */

#define AXIOM_REGEX_OPT_CASELESS 0x01u
#define AXIOM_REGEX_INF         ((size_t)-1)

typedef enum {
    AXIOM_RX_LITERAL = 0,
    AXIOM_RX_DOT,
    AXIOM_RX_CLASS
} axiom_rx_atom_kind;

typedef struct {
    axiom_rx_atom_kind kind;
    unsigned char      literal;
    unsigned char      bitmap[256];
    bool               negate;
    size_t             min_count;
    size_t             max_count;
} axiom_regex_token;

struct axiom_regex {
    char              *pattern;
    size_t             pattern_len;
    uint32_t           options;
    axiom_regex_token *tokens;
    size_t             token_count;
    bool               anchor_start;
    bool               anchor_end;
};

static void rx_class_clear(axiom_regex_token *tok) {
    memset(tok->bitmap, 0, sizeof(tok->bitmap));
}

static void rx_class_add_char(axiom_regex_token *tok, unsigned char c) {
    tok->bitmap[c] = 1u;
}

static void rx_class_add_range(axiom_regex_token *tok,
                               unsigned char first,
                               unsigned char last) {
    if (first > last) {
        unsigned char tmp = first;
        first = last;
        last = tmp;
    }
    for (unsigned int c = first; c <= last; ++c) {
        tok->bitmap[c] = 1u;
    }
}

static void rx_class_add_escape(axiom_regex_token *tok, char esc, bool *handled) {
    *handled = true;
    switch (esc) {
    case 'd':
        rx_class_add_range(tok, '0', '9');
        break;
    case 'D':
        for (unsigned int c = 0; c < 256; ++c) tok->bitmap[c] = 1u;
        for (unsigned int c = '0'; c <= '9'; ++c) tok->bitmap[c] = 0u;
        break;
    case 'w':
        rx_class_add_range(tok, 'A', 'Z');
        rx_class_add_range(tok, 'a', 'z');
        rx_class_add_range(tok, '0', '9');
        rx_class_add_char(tok, '_');
        break;
    case 'W':
        for (unsigned int c = 0; c < 256; ++c) tok->bitmap[c] = 1u;
        for (unsigned int c = 'A'; c <= 'Z'; ++c) tok->bitmap[c] = 0u;
        for (unsigned int c = 'a'; c <= 'z'; ++c) tok->bitmap[c] = 0u;
        for (unsigned int c = '0'; c <= '9'; ++c) tok->bitmap[c] = 0u;
        tok->bitmap[(unsigned char)'_'] = 0u;
        break;
    case 's':
        rx_class_add_char(tok, ' ');
        rx_class_add_char(tok, '\t');
        rx_class_add_char(tok, '\n');
        rx_class_add_char(tok, '\r');
        rx_class_add_char(tok, '\f');
        rx_class_add_char(tok, '\v');
        break;
    case 'S':
        for (unsigned int c = 0; c < 256; ++c) tok->bitmap[c] = 1u;
        tok->bitmap[(unsigned char)' '] = 0u;
        tok->bitmap[(unsigned char)'\t'] = 0u;
        tok->bitmap[(unsigned char)'\n'] = 0u;
        tok->bitmap[(unsigned char)'\r'] = 0u;
        tok->bitmap[(unsigned char)'\f'] = 0u;
        tok->bitmap[(unsigned char)'\v'] = 0u;
        break;
    default:
        *handled = false;
        break;
    }
}

static bool rx_is_escaped(const char *pattern, size_t pos) {
    size_t slash_count = 0;
    while (pos > 0 && pattern[pos - 1u] == '\\') {
        slash_count++;
        pos--;
    }
    return (slash_count & 1u) != 0u;
}

static int rx_parse_class_char(const char *pattern, size_t limit, size_t *idx,
                               unsigned char *out) {
    if (*idx >= limit) {
        return 0;
    }
    unsigned char c = (unsigned char)pattern[*idx];
    (*idx)++;
    if (c == '\\' && *idx < limit) {
        c = (unsigned char)pattern[*idx];
        (*idx)++;
        switch (c) {
        case 'n': c = '\n'; break;
        case 'r': c = '\r'; break;
        case 't': c = '\t'; break;
        default: break;
        }
    }
    *out = c;
    return 1;
}

static int rx_parse_class(const char *pattern, size_t limit, size_t *idx,
                          axiom_regex_token *tok) {
    bool saw_content = false;
    tok->kind = AXIOM_RX_CLASS;
    tok->negate = false;
    rx_class_clear(tok);

    if (*idx < limit && pattern[*idx] == '^') {
        tok->negate = true;
        (*idx)++;
    }

    while (*idx < limit) {
        if (pattern[*idx] == ']' && saw_content) {
            (*idx)++;
            return 1;
        }

        if (pattern[*idx] == '\\' && *idx + 1u < limit) {
            char esc = pattern[*idx + 1u];
            bool handled = false;
            rx_class_add_escape(tok, esc, &handled);
            if (handled) {
                *idx += 2u;
                saw_content = true;
                continue;
            }
        }

        unsigned char first = 0;
        if (!rx_parse_class_char(pattern, limit, idx, &first)) {
            return 0;
        }

        if (*idx < limit && pattern[*idx] == '-' &&
            *idx + 1u < limit && pattern[*idx + 1u] != ']') {
            (*idx)++;
            unsigned char last = 0;
            if (!rx_parse_class_char(pattern, limit, idx, &last)) {
                return 0;
            }
            rx_class_add_range(tok, first, last);
        } else {
            rx_class_add_char(tok, first);
        }
        saw_content = true;
    }
    return 0;
}

static int rx_parse_uint(const char *pattern, size_t limit, size_t *idx,
                         size_t *out) {
    if (*idx >= limit || !isdigit((unsigned char)pattern[*idx])) {
        return 0;
    }
    size_t value = 0;
    while (*idx < limit && isdigit((unsigned char)pattern[*idx])) {
        value = (value * 10u) + (size_t)(pattern[*idx] - '0');
        (*idx)++;
    }
    *out = value;
    return 1;
}

static int rx_parse_quantifier(const char *pattern, size_t limit, size_t *idx,
                               axiom_regex_token *tok) {
    tok->min_count = 1u;
    tok->max_count = 1u;
    if (*idx >= limit) {
        return 1;
    }

    if (pattern[*idx] == '*') {
        tok->min_count = 0u;
        tok->max_count = AXIOM_REGEX_INF;
        (*idx)++;
        return 1;
    }
    if (pattern[*idx] == '+') {
        tok->min_count = 1u;
        tok->max_count = AXIOM_REGEX_INF;
        (*idx)++;
        return 1;
    }
    if (pattern[*idx] == '?') {
        tok->min_count = 0u;
        tok->max_count = 1u;
        (*idx)++;
        return 1;
    }
    if (pattern[*idx] != '{') {
        return 1;
    }

    size_t pos = *idx + 1u;
    size_t min_value = 0;
    size_t max_value = 0;
    if (!rx_parse_uint(pattern, limit, &pos, &min_value)) {
        return 0;
    }
    if (pos < limit && pattern[pos] == ',') {
        pos++;
        if (pos < limit && isdigit((unsigned char)pattern[pos])) {
            if (!rx_parse_uint(pattern, limit, &pos, &max_value)) {
                return 0;
            }
        } else {
            max_value = AXIOM_REGEX_INF;
        }
    } else {
        max_value = min_value;
    }
    if (pos >= limit || pattern[pos] != '}') {
        return 0;
    }
    if (max_value != AXIOM_REGEX_INF && max_value < min_value) {
        return 0;
    }
    tok->min_count = min_value;
    tok->max_count = max_value;
    *idx = pos + 1u;
    return 1;
}

static int rx_init_escape_token(axiom_regex_token *tok, char esc) {
    bool handled = false;
    tok->kind = AXIOM_RX_CLASS;
    tok->negate = false;
    rx_class_clear(tok);
    rx_class_add_escape(tok, esc, &handled);
    if (handled) {
        return 1;
    }

    tok->kind = AXIOM_RX_LITERAL;
    tok->literal = (unsigned char)esc;
    switch (esc) {
    case 'n': tok->literal = '\n'; break;
    case 'r': tok->literal = '\r'; break;
    case 't': tok->literal = '\t'; break;
    default: break;
    }
    return 1;
}

static bool rx_char_equal(unsigned char a, unsigned char b, uint32_t options) {
    if ((options & AXIOM_REGEX_OPT_CASELESS) == 0u) {
        return a == b;
    }
    return (unsigned char)tolower(a) == (unsigned char)tolower(b);
}

static bool rx_token_matches(const axiom_regex_token *tok, unsigned char c,
                             uint32_t options) {
    switch (tok->kind) {
    case AXIOM_RX_LITERAL:
        return rx_char_equal(c, tok->literal, options);
    case AXIOM_RX_DOT:
        return c != '\n';
    case AXIOM_RX_CLASS: {
        bool present = tok->bitmap[c] != 0u;
        if ((options & AXIOM_REGEX_OPT_CASELESS) != 0u && isalpha(c)) {
            present = present ||
                      tok->bitmap[(unsigned char)tolower(c)] != 0u ||
                      tok->bitmap[(unsigned char)toupper(c)] != 0u;
        }
        return tok->negate ? !present : present;
    }
    default:
        return false;
    }
}

static bool rx_match_here(const axiom_regex *re, size_t token_idx,
                          const char *subject, size_t subject_len,
                          size_t pos, size_t *match_end) {
    if (token_idx == re->token_count) {
        if (re->anchor_end && pos != subject_len) {
            return false;
        }
        *match_end = pos;
        return true;
    }

    const axiom_regex_token *tok = &re->tokens[token_idx];
    size_t max_count = 0;
    size_t scan = pos;
    while (scan < subject_len &&
           (tok->max_count == AXIOM_REGEX_INF || max_count < tok->max_count) &&
           rx_token_matches(tok, (unsigned char)subject[scan], re->options)) {
        scan++;
        max_count++;
    }
    if (max_count < tok->min_count) {
        return false;
    }

    size_t count = max_count;
    for (;;) {
        size_t next_pos = pos + count;
        if (rx_match_here(re, token_idx + 1u, subject, subject_len,
                          next_pos, match_end)) {
            return true;
        }
        if (count == tok->min_count) {
            break;
        }
        count--;
    }
    return false;
}

static bool rx_find(const axiom_regex *re, const char *subject,
                    size_t subject_len, size_t start_offset,
                    size_t *match_start, size_t *match_end) {
    if (start_offset > subject_len) {
        return false;
    }
    if (re->anchor_start) {
        if (start_offset != 0u) {
            return false;
        }
        if (rx_match_here(re, 0, subject, subject_len, 0, match_end)) {
            *match_start = 0;
            return true;
        }
        return false;
    }

    for (size_t pos = start_offset; pos <= subject_len; ++pos) {
        if (rx_match_here(re, 0, subject, subject_len, pos, match_end)) {
            *match_start = pos;
            return true;
        }
        if (pos == subject_len) {
            break;
        }
    }
    return false;
}

axiom_regex *axiom_regex_compile(const char *pattern, size_t pattern_len,
                                 uint32_t options, int *errcode) {
    if (pattern == NULL || pattern_len == 0) {
        if (errcode) *errcode = AXIOM_STRIP_ERR_INVALID_ARG;
        return NULL;
    }

    axiom_regex *re = (axiom_regex *)calloc(1, sizeof(axiom_regex));
    if (re == NULL) {
        if (errcode) *errcode = AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        return NULL;
    }

    re->pattern = (char *)malloc(pattern_len + 1u);
    re->tokens = (axiom_regex_token *)calloc(pattern_len + 1u, sizeof(axiom_regex_token));
    if (re->pattern == NULL || re->tokens == NULL) {
        axiom_regex_free(re);
        if (errcode) *errcode = AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        return NULL;
    }
    memcpy(re->pattern, pattern, pattern_len);
    re->pattern[pattern_len] = '\0';
    re->pattern_len = pattern_len;
    re->options = options;

    size_t i = 0;
    size_t limit = pattern_len;
    if (pattern[0] == '^') {
        re->anchor_start = true;
        i = 1u;
    }
    if (limit > i && pattern[limit - 1u] == '$' &&
        !rx_is_escaped(pattern, limit - 1u)) {
        re->anchor_end = true;
        limit--;
    }
    if (i == limit) {
        axiom_regex_free(re);
        if (errcode) *errcode = AXIOM_STRIP_ERR_PCRE2_COMPILE;
        return NULL;
    }

    while (i < limit) {
        axiom_regex_token tok;
        memset(&tok, 0, sizeof(tok));

        char c = pattern[i++];
        if (c == '[') {
            if (!rx_parse_class(pattern, limit, &i, &tok)) {
                axiom_regex_free(re);
                if (errcode) *errcode = AXIOM_STRIP_ERR_PCRE2_COMPILE;
                return NULL;
            }
        } else if (c == '.') {
            tok.kind = AXIOM_RX_DOT;
        } else if (c == '\\') {
            if (i >= limit || !rx_init_escape_token(&tok, pattern[i++])) {
                axiom_regex_free(re);
                if (errcode) *errcode = AXIOM_STRIP_ERR_PCRE2_COMPILE;
                return NULL;
            }
        } else {
            tok.kind = AXIOM_RX_LITERAL;
            tok.literal = (unsigned char)c;
        }

        if (!rx_parse_quantifier(pattern, limit, &i, &tok)) {
            axiom_regex_free(re);
            if (errcode) *errcode = AXIOM_STRIP_ERR_PCRE2_COMPILE;
            return NULL;
        }
        re->tokens[re->token_count++] = tok;
    }

    if (errcode) *errcode = AXIOM_STRIP_OK;
    return re;
}

int axiom_regex_jit_hint(axiom_regex *re) {
    return re == NULL ? AXIOM_STRIP_ERR_INVALID_ARG : AXIOM_STRIP_OK;
}

int axiom_regex_match(const axiom_regex *re, const char *subject,
                      size_t subject_len, size_t start_offset,
                      size_t *ovector, size_t ovector_count,
                      size_t *match_count) {
    if (re == NULL || subject == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }

    size_t match_start = 0;
    size_t match_end = 0;
    if (!rx_find(re, subject, subject_len, start_offset, &match_start, &match_end)) {
        if (match_count) *match_count = 0;
        return AXIOM_STRIP_OK;
    }

    if (match_count) *match_count = 1;
    if (ovector != NULL && ovector_count >= 2u) {
        ovector[0] = match_start;
        ovector[1] = match_end;
    }
    return AXIOM_STRIP_OK;
}

size_t axiom_regex_replace(const axiom_regex *re, const char *subject,
                           size_t subject_len, const char *replacement,
                           char *output, size_t output_capacity) {
    if (re == NULL || subject == NULL || output == NULL) return 0;

    size_t repl_len = replacement ? strlen(replacement) : 0u;
    size_t oi = 0;
    size_t pos = 0;
    while (pos <= subject_len) {
        size_t match_start = 0;
        size_t match_end = 0;
        if (!rx_find(re, subject, subject_len, pos, &match_start, &match_end)) {
            size_t tail = subject_len - pos;
            if (oi + tail > output_capacity) return 0;
            memcpy(output + oi, subject + pos, tail);
            oi += tail;
            break;
        }

        size_t prefix_len = match_start - pos;
        if (oi + prefix_len + repl_len > output_capacity) return 0;
        memcpy(output + oi, subject + pos, prefix_len);
        oi += prefix_len;
        if (repl_len > 0u) {
            memcpy(output + oi, replacement, repl_len);
            oi += repl_len;
        }

        if (match_end == match_start) {
            if (match_end >= subject_len) {
                pos = subject_len + 1u;
            } else {
                if (oi + 1u > output_capacity) return 0;
                output[oi++] = subject[match_end];
                pos = match_end + 1u;
            }
        } else {
            pos = match_end;
        }
    }

    if (oi < output_capacity) {
        output[oi] = '\0';
    }
    return oi;
}

size_t axiom_regex_extract_all(const axiom_regex *re, const char *subject,
                               size_t subject_len, char *output,
                               size_t output_capacity) {
    if (re == NULL || subject == NULL || output == NULL) return 0;

    size_t oi = 0;
    size_t pos = 0;
    while (pos <= subject_len) {
        size_t match_start = 0;
        size_t match_end = 0;
        if (!rx_find(re, subject, subject_len, pos, &match_start, &match_end)) {
            break;
        }

        size_t mlen = match_end - match_start;
        if (mlen > 0u) {
            if (oi + mlen + 1u > output_capacity) return 0;
            memcpy(output + oi, subject + match_start, mlen);
            oi += mlen;
            output[oi++] = '\n';
        }

        if (match_end == match_start) {
            if (match_end >= subject_len) break;
            pos = match_end + 1u;
        } else {
            pos = match_end;
        }
    }

    if (oi < output_capacity) {
        output[oi] = '\0';
    }
    return oi;
}

void axiom_regex_free(axiom_regex *re) {
    if (re == NULL) return;
    free(re->pattern);
    free(re->tokens);
    free(re);
}

size_t axiom_regex_strip_matches(const char *pattern, size_t pattern_len,
                                 const char *subject, size_t subject_len,
                                 char *output, size_t output_capacity) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) {
        return 0;
    }
    size_t result = axiom_regex_replace(re, subject, subject_len, "",
                                        output, output_capacity);
    axiom_regex_free(re);
    return result;
}

size_t axiom_regex_extract_matches(const char *pattern, size_t pattern_len,
                                   const char *subject, size_t subject_len,
                                   char *output, size_t output_capacity) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) return 0;
    size_t r = axiom_regex_extract_all(re, subject, subject_len, output, output_capacity);
    axiom_regex_free(re);
    return r;
}

size_t axiom_regex_replace_matches(const char *pattern, size_t pattern_len,
                                   const char *subject, size_t subject_len,
                                   const char *replacement,
                                   char *output, size_t output_capacity) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) {
        return 0;
    }
    size_t r = axiom_regex_replace(re, subject, subject_len,
                                   replacement ? replacement : "",
                                   output, output_capacity);
    axiom_regex_free(re);
    return r;
}

int axiom_regex_test(const char *pattern, size_t pattern_len,
                     const char *subject, size_t subject_len) {
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(pattern, pattern_len, 0, &ec);
    if (re == NULL) return -1;
    size_t mc = 0;
    int rc = axiom_regex_match(re, subject, subject_len, 0, NULL, 0, &mc);
    axiom_regex_free(re);
    if (rc != AXIOM_STRIP_OK) return -1;
    return mc > 0 ? 1 : 0;
}

#endif /* AXIOM_NO_PCRE2 */

/*
 * strip_engine_part5.c â€” Recipe loading from mmap and CRC validation.
 *
 * Loads compiled recipes from recipe_registry.mmap at a given slot
 * index.  Validates magic, version, slot bounds, and CRC32 checksums.
 * Uses platform-specific mmap (Unix) or MapViewOfFile (Windows).
 */


#ifdef _WIN32
#  include <windows.h>
#  include <io.h>
#else
#  include <fcntl.h>
#  include <sys/mman.h>
#  include <sys/stat.h>
#  include <unistd.h>
#endif

/* ------------------------------------------------------------------ */
/*  Platform-abstracted mmap read                                      */
/* ------------------------------------------------------------------ */

typedef struct {
    const uint8_t *data;
    size_t         len;
#ifdef _WIN32
    HANDLE         file_handle;
    HANDLE         map_handle;
#else
    int            fd;
#endif
} mmap_view;

static int mmap_open(const char *path, mmap_view *view) {
    if (path == NULL || view == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(view, 0, sizeof(*view));

#ifdef _WIN32
    HANDLE fh = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ, NULL,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (fh == INVALID_HANDLE_VALUE) {
        return AXIOM_STRIP_ERR_MMAP_OPEN;
    }
    LARGE_INTEGER sz;
    if (!GetFileSizeEx(fh, &sz) || sz.QuadPart == 0) {
        CloseHandle(fh);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    HANDLE mh = CreateFileMappingA(fh, NULL, PAGE_READONLY, 0, 0, NULL);
    if (mh == NULL) {
        CloseHandle(fh);
        return AXIOM_STRIP_ERR_MMAP_OPEN;
    }
    void *ptr = MapViewOfFile(mh, FILE_MAP_READ, 0, 0, 0);
    if (ptr == NULL) {
        CloseHandle(mh);
        CloseHandle(fh);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    view->data = (const uint8_t *)ptr;
    view->len = (size_t)sz.QuadPart;
    view->file_handle = fh;
    view->map_handle = mh;
#else
    int fd = open(path, O_RDONLY);
    if (fd < 0) {
        return AXIOM_STRIP_ERR_MMAP_OPEN;
    }
    struct stat st;
    if (fstat(fd, &st) != 0 || st.st_size == 0) {
        close(fd);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    void *ptr = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (ptr == MAP_FAILED) {
        close(fd);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    view->data = (const uint8_t *)ptr;
    view->len = (size_t)st.st_size;
    view->fd = fd;
#endif
    return AXIOM_STRIP_OK;
}

static void mmap_close(mmap_view *view) {
    if (view == NULL || view->data == NULL) return;
#ifdef _WIN32
    UnmapViewOfFile((void *)view->data);
    if (view->map_handle) CloseHandle(view->map_handle);
    if (view->file_handle) CloseHandle(view->file_handle);
#else
    munmap((void *)view->data, view->len);
    if (view->fd >= 0) close(view->fd);
#endif
    memset(view, 0, sizeof(*view));
}

/* ------------------------------------------------------------------ */
/*  Parse step data from a slot's binary blob                          */
/* ------------------------------------------------------------------ */

/*
 * Step binary format within a slot's data region:
 *   [1 byte: kind_len] [kind_len bytes: kind string]
 *   [2 bytes LE: pattern_len] [pattern_len bytes: pattern]
 *   [2 bytes LE: replacement_len] [replacement_len bytes: replacement]
 *   [1 byte: flags]
 *   [4 bytes LE: confidence as float bits]
 *
 * Repeated step_count times.
 */

static uint16_t read_u16_le(const uint8_t *p) {
    return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1] << 8));
}

static uint32_t read_u32_le(const uint8_t *p) {
    return (uint32_t)p[0] | ((uint32_t)p[1] << 8) |
           ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}

static float read_f32_le(const uint8_t *p) {
    uint32_t bits = read_u32_le(p);
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static int parse_slot_steps(const uint8_t *data, size_t data_len,
                            uint32_t step_count, strip_pool *pool,
                            axiom_strip_step **out_steps) {
    if (step_count == 0) {
        *out_steps = NULL;
        return AXIOM_STRIP_OK;
    }
    if (step_count > AXIOM_STRIP_MAX_STEPS) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    axiom_strip_step *steps = (axiom_strip_step *)strip_pool_alloc_zero(
        pool, step_count * sizeof(axiom_strip_step)
    );
    if (steps == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    size_t pos = 0;
    for (uint32_t i = 0; i < step_count; ++i) {
        if (pos >= data_len) return AXIOM_STRIP_ERR_MMAP_READ;

        uint8_t kind_len = data[pos++];
        if (pos + kind_len > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        steps[i].kind = strip_pool_strndup(pool, (const char *)data + pos, kind_len);
        if (steps[i].kind == NULL) return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        pos += kind_len;

        if (pos + 2 > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        uint16_t pat_len = read_u16_le(data + pos);
        pos += 2;
        if (pos + pat_len > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        if (pat_len > 0) {
            steps[i].pattern = strip_pool_strndup(pool, (const char *)data + pos, pat_len);
            if (steps[i].pattern == NULL) return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        }
        pos += pat_len;

        if (pos + 2 > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        uint16_t repl_len = read_u16_le(data + pos);
        pos += 2;
        if (pos + repl_len > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        if (repl_len > 0) {
            steps[i].replacement = strip_pool_strndup(pool, (const char *)data + pos, repl_len);
            if (steps[i].replacement == NULL) return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        }
        pos += repl_len;

        if (pos + 1 > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        steps[i].flags = data[pos++];

        if (pos + 4 > data_len) return AXIOM_STRIP_ERR_MMAP_READ;
        steps[i].confidence = read_f32_le(data + pos);
        pos += 4;
    }
    *out_steps = steps;
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Load recipe from mmap                                              */
/* ------------------------------------------------------------------ */

axiom_strip_recipe *strip_load_recipe(int slot_idx, const char *mmap_path,
                                      strip_pool *pool) {
    if (mmap_path == NULL || pool == NULL || slot_idx < 0) {
        return NULL;
    }
    mmap_view view;
    if (mmap_open(mmap_path, &view) != AXIOM_STRIP_OK) {
        return NULL;
    }
    if (view.len < sizeof(axiom_recipe_registry_header)) {
        mmap_close(&view);
        return NULL;
    }
    const axiom_recipe_registry_header *hdr =
        (const axiom_recipe_registry_header *)view.data;
    if (hdr->magic != AXIOM_RECIPE_MAGIC) {
        mmap_close(&view);
        return NULL;
    }
    if ((uint32_t)slot_idx >= hdr->slot_count) {
        mmap_close(&view);
        return NULL;
    }
    size_t slot_offset = sizeof(axiom_recipe_registry_header) +
                         (size_t)slot_idx * AXIOM_RECIPE_SLOT_BYTES;
    if (slot_offset + AXIOM_RECIPE_SLOT_BYTES > view.len) {
        mmap_close(&view);
        return NULL;
    }
    const uint8_t *slot_base = view.data + slot_offset;
    const axiom_recipe_slot_header *sh =
        (const axiom_recipe_slot_header *)slot_base;

    if (sh->step_count > AXIOM_STRIP_MAX_STEPS) {
        mmap_close(&view);
        return NULL;
    }
    size_t data_end = (size_t)sh->data_offset + (size_t)sh->data_len;
    if (data_end > AXIOM_RECIPE_SLOT_BYTES) {
        mmap_close(&view);
        return NULL;
    }
    axiom_strip_recipe *recipe = (axiom_strip_recipe *)strip_pool_alloc_zero(
        pool, sizeof(axiom_strip_recipe)
    );
    if (recipe == NULL) {
        mmap_close(&view);
        return NULL;
    }
    memcpy(recipe->topology_class, sh->topology_class,
           AXIOM_STRIP_TOPOLOGY_LEN);
    recipe->topology_class[AXIOM_STRIP_TOPOLOGY_LEN - 1] = '\0';
    recipe->step_count = sh->step_count;
    recipe->checksum = sh->checksum;
    recipe->max_output_ratio = AXIOM_STRIP_DEFAULT_RATIO;

    const uint8_t *step_data = slot_base + sh->data_offset;
    int rc = parse_slot_steps(step_data, sh->data_len, sh->step_count,
                              pool, &recipe->steps);
    mmap_close(&view);
    if (rc != AXIOM_STRIP_OK) {
        return NULL;
    }
    return recipe;
}

/* ------------------------------------------------------------------ */
/*  Free recipe (pool-backed: no-op unless heap-allocated steps)       */
/* ------------------------------------------------------------------ */

void strip_free_recipe(axiom_strip_recipe *recipe) {
    (void)recipe;
}

/* ------------------------------------------------------------------ */
/*  Validate recipe CRC against the registry on disk                   */
/* ------------------------------------------------------------------ */

int strip_validate_recipe(const axiom_strip_recipe *recipe,
                          const char *mmap_path) {
    if (recipe == NULL || mmap_path == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    mmap_view view;
    int rc = mmap_open(mmap_path, &view);
    if (rc != AXIOM_STRIP_OK) {
        return rc;
    }
    if (view.len < sizeof(axiom_recipe_registry_header)) {
        mmap_close(&view);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    const axiom_recipe_registry_header *hdr =
        (const axiom_recipe_registry_header *)view.data;
    if (hdr->magic != AXIOM_RECIPE_MAGIC) {
        mmap_close(&view);
        return AXIOM_STRIP_ERR_MMAP_READ;
    }
    bool found = false;
    for (uint32_t i = 0; i < hdr->slot_count && !found; ++i) {
        size_t off = sizeof(axiom_recipe_registry_header) +
                     (size_t)i * AXIOM_RECIPE_SLOT_BYTES;
        if (off + sizeof(axiom_recipe_slot_header) > view.len) break;
        const axiom_recipe_slot_header *sh =
            (const axiom_recipe_slot_header *)(view.data + off);
        if (memcmp(sh->topology_class, recipe->topology_class,
                   AXIOM_STRIP_TOPOLOGY_LEN) == 0) {
            found = true;
            if (sh->checksum != recipe->checksum) {
                mmap_close(&view);
                return AXIOM_STRIP_ERR_CRC_MISMATCH;
            }
            const uint8_t *slot_data = view.data + off +
                                       sh->data_offset;
            uint32_t computed = axiom_crc32(slot_data, sh->data_len);
            if (computed != recipe->checksum) {
                mmap_close(&view);
                return AXIOM_STRIP_ERR_CRC_MISMATCH;
            }
        }
    }
    mmap_close(&view);
    if (!found) {
        return AXIOM_STRIP_ERR_SLOT_OOB;
    }
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Compute recipe CRC from its steps (for recipe builders)            */
/* ------------------------------------------------------------------ */

uint32_t axiom_recipe_compute_crc(const axiom_strip_recipe *recipe) {
    if (recipe == NULL || recipe->steps == NULL) {
        return 0;
    }
    uint32_t crc = 0xFFFFFFFFu;
    for (size_t i = 0; i < recipe->step_count; ++i) {
        const axiom_strip_step *s = &recipe->steps[i];
        if (s->kind) {
            crc = axiom_crc32_combine(crc, (const uint8_t *)s->kind, strlen(s->kind));
        }
        if (s->pattern) {
            crc = axiom_crc32_combine(crc, (const uint8_t *)s->pattern, strlen(s->pattern));
        }
        if (s->replacement) {
            crc = axiom_crc32_combine(crc, (const uint8_t *)s->replacement, strlen(s->replacement));
        }
        crc = axiom_crc32_combine(crc, &s->flags, 1);
    }
    return crc;
}

/* ------------------------------------------------------------------ */
/*  Write a recipe to a binary slot buffer (for test fixtures)         */
/* ------------------------------------------------------------------ */

static void write_u16_le(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
}

static void write_u32_le(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xFF);
    p[1] = (uint8_t)((v >> 8) & 0xFF);
    p[2] = (uint8_t)((v >> 16) & 0xFF);
    p[3] = (uint8_t)((v >> 24) & 0xFF);
}

int axiom_recipe_write_slot(const axiom_strip_recipe *recipe,
                            uint8_t *slot_buf, size_t slot_capacity) {
    if (recipe == NULL || slot_buf == NULL ||
        slot_capacity < AXIOM_RECIPE_SLOT_BYTES) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(slot_buf, 0, slot_capacity);
    axiom_recipe_slot_header *sh = (axiom_recipe_slot_header *)slot_buf;
    memcpy(sh->topology_class, recipe->topology_class, AXIOM_STRIP_TOPOLOGY_LEN);
    sh->step_count = (uint32_t)recipe->step_count;
    sh->data_offset = (uint32_t)sizeof(axiom_recipe_slot_header);

    uint8_t *data = slot_buf + sh->data_offset;
    size_t data_capacity = slot_capacity - sh->data_offset;
    size_t pos = 0;

    for (size_t i = 0; i < recipe->step_count; ++i) {
        const axiom_strip_step *s = &recipe->steps[i];
        size_t kind_len = s->kind ? strlen(s->kind) : 0;
        size_t pat_len = s->pattern ? strlen(s->pattern) : 0;
        size_t repl_len = s->replacement ? strlen(s->replacement) : 0;
        size_t step_bytes = 1 + kind_len + 2 + pat_len + 2 + repl_len + 1 + 4;
        if (pos + step_bytes > data_capacity) {
            return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
        }
        data[pos++] = (uint8_t)kind_len;
        if (kind_len > 0) {
            memcpy(data + pos, s->kind, kind_len);
            pos += kind_len;
        }
        write_u16_le(data + pos, (uint16_t)pat_len);
        pos += 2;
        if (pat_len > 0) {
            memcpy(data + pos, s->pattern, pat_len);
            pos += pat_len;
        }
        write_u16_le(data + pos, (uint16_t)repl_len);
        pos += 2;
        if (repl_len > 0) {
            memcpy(data + pos, s->replacement, repl_len);
            pos += repl_len;
        }
        data[pos++] = s->flags;
        float conf = s->confidence;
        uint32_t conf_bits;
        memcpy(&conf_bits, &conf, sizeof(conf_bits));
        write_u32_le(data + pos, conf_bits);
        pos += 4;
    }
    sh->data_len = (uint32_t)pos;
    sh->checksum = axiom_crc32(data, pos);
    return AXIOM_STRIP_OK;
}

int axiom_recipe_write_registry(const axiom_strip_recipe *recipes,
                                size_t recipe_count,
                                uint8_t *buf, size_t buf_capacity) {
    size_t needed = sizeof(axiom_recipe_registry_header) +
                    recipe_count * AXIOM_RECIPE_SLOT_BYTES;
    if (buf == NULL || buf_capacity < needed) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    memset(buf, 0, needed);
    axiom_recipe_registry_header *hdr = (axiom_recipe_registry_header *)buf;
    hdr->magic = AXIOM_RECIPE_MAGIC;
    hdr->version = 1;
    hdr->slot_count = (uint32_t)recipe_count;

    for (size_t i = 0; i < recipe_count; ++i) {
        uint8_t *slot = buf + sizeof(axiom_recipe_registry_header) +
                        i * AXIOM_RECIPE_SLOT_BYTES;
        int rc = axiom_recipe_write_slot(&recipes[i], slot, AXIOM_RECIPE_SLOT_BYTES);
        if (rc != AXIOM_STRIP_OK) return rc;
    }
    return AXIOM_STRIP_OK;
}

/*
 * strip_engine_part6.c â€” Step execution engine.
 *
 * Dispatches each recipe step to the appropriate handler function,
 * evaluates confidence gating and step flags (STRIP_EXTRACT,
 * STRIP_REMOVE, STRIP_REQUIRE), and tracks which steps fired.
 */


/* ------------------------------------------------------------------ */
/*  Known step kinds                                                   */
/* ------------------------------------------------------------------ */

static const char *g_known_kinds[] = {
    "strip_tag",
    "collapse_ws",
    "remove",
    "replace",
    "strip_html",
    "strip_comments",
    "strip_attrs",
    "decode_entities",
    "keep_between",
    "keep_between_all",
    "strip_between",
    "filter_lines",
    "filter_lines_not",
    "truncate",
    "normalize_ws",
    "deduplicate_lines",
    "trim_lines",
    "extract_runs",
    "remove_blank_lines",
    "squeeze_blank_lines",
    "strip_cdata",
    "strip_inline_styles",
    "strip_data_attrs",
    "strip_noscript",
    "strip_iframe",
    "strip_svg",
    "strip_template",
    "regex_remove",
    "regex_extract",
    "regex_replace",
    NULL
};

bool si_step_kind_known(const char *kind) {
    if (kind == NULL) {
        return true;
    }
    for (int i = 0; g_known_kinds[i] != NULL; ++i) {
        if (strcmp(kind, g_known_kinds[i]) == 0) {
            return true;
        }
    }
    return false;
}

/* ------------------------------------------------------------------ */
/*  Confidence gating                                                  */
/* ------------------------------------------------------------------ */

bool si_should_fire_step(const axiom_strip_step *step, float confidence) {
    if (step == NULL) {
        return false;
    }
    if (step->confidence <= 0.0f) {
        return true;
    }
    return confidence >= step->confidence;
}

/* ------------------------------------------------------------------ */
/*  Flag evaluation helpers                                            */
/* ------------------------------------------------------------------ */

static bool flag_extract(uint8_t flags) {
    return (flags & STRIP_EXTRACT) != 0;
}

static bool flag_require(uint8_t flags) {
    return (flags & STRIP_REQUIRE) != 0;
}

/* ------------------------------------------------------------------ */
/*  Pool buffer allocation for step intermediates                      */
/* ------------------------------------------------------------------ */

static char *alloc_step_buf(strip_pool *pool, size_t in_len) {
    size_t need = (in_len * 2u) + 4096u;
    return (char *)strip_pool_alloc(pool, need);
}

/* ------------------------------------------------------------------ */
/*  Step dispatch                                                      */
/* ------------------------------------------------------------------ */

int si_run_step(const axiom_strip_step *step, const char *input,
                size_t in_len, char **out, size_t *out_len,
                strip_pool *pool, float current_confidence) {
    if (pool == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    char *buf = alloc_step_buf(pool, in_len);
    if (buf == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }

    if (step == NULL || step->kind == NULL) {
        memcpy(buf, input, in_len);
        *out = buf;
        *out_len = in_len;
        return AXIOM_STRIP_OK;
    }

    if (!si_should_fire_step(step, current_confidence)) {
        memcpy(buf, input, in_len);
        *out = buf;
        *out_len = in_len;
        return AXIOM_STRIP_OK;
    }

    const char *kind = step->kind;
    const char *pattern = step->pattern ? step->pattern : "";
    const char *replacement = step->replacement ? step->replacement : "";

    if (strcmp(kind, "strip_tag") == 0) {
        *out_len = si_strip_tag_block(input, in_len,
                                      pattern[0] ? pattern : "script", buf);
    } else if (strcmp(kind, "collapse_ws") == 0) {
        *out_len = si_collapse_ws(input, in_len, buf);
    } else if (strcmp(kind, "remove") == 0) {
        *out_len = si_remove_literal(input, in_len, pattern, buf);
    } else if (strcmp(kind, "replace") == 0) {
        *out_len = si_replace_literal(input, in_len, pattern, replacement, buf);
    } else if (strcmp(kind, "strip_html") == 0) {
        *out_len = si_strip_html_tags(input, in_len, buf);
    } else if (strcmp(kind, "strip_comments") == 0) {
        *out_len = si_strip_comments(input, in_len, buf);
    } else if (strcmp(kind, "strip_attrs") == 0) {
        *out_len = si_strip_attributes(input, in_len, buf);
    } else if (strcmp(kind, "decode_entities") == 0) {
        *out_len = si_decode_entities(input, in_len, buf);
    } else if (strcmp(kind, "keep_between") == 0) {
        *out_len = si_keep_between(input, in_len, step->pattern,
                                   step->replacement, buf);
    } else if (strcmp(kind, "keep_between_all") == 0) {
        *out_len = si_keep_between_all(input, in_len, step->pattern,
                                       step->replacement, buf);
    } else if (strcmp(kind, "strip_between") == 0) {
        *out_len = si_strip_between(input, in_len, step->pattern,
                                    step->replacement, buf);
    } else if (strcmp(kind, "filter_lines") == 0) {
        *out_len = si_filter_lines_containing(input, in_len, pattern, buf);
    } else if (strcmp(kind, "filter_lines_not") == 0) {
        *out_len = si_filter_lines_not_containing(input, in_len, pattern, buf);
    } else if (strcmp(kind, "truncate") == 0) {
        *out_len = si_truncate_bytes(input, in_len, pattern, buf);
    } else if (strcmp(kind, "normalize_ws") == 0) {
        *out_len = si_normalize_unicode_ws(input, in_len, buf);
    } else if (strcmp(kind, "deduplicate_lines") == 0) {
        *out_len = si_deduplicate_lines(input, in_len, buf);
    } else if (strcmp(kind, "trim_lines") == 0) {
        *out_len = si_trim_lines(input, in_len, buf);
    } else if (strcmp(kind, "extract_runs") == 0) {
        size_t min_run = 4;
        si_parse_positive_size(pattern, 4, &min_run);
        *out_len = si_extract_text_runs(input, in_len, min_run, buf);
    } else if (strcmp(kind, "remove_blank_lines") == 0) {
        *out_len = si_remove_blank_lines(input, in_len, buf);
    } else if (strcmp(kind, "squeeze_blank_lines") == 0) {
        *out_len = si_squeeze_blank_lines(input, in_len, buf);
    } else if (strcmp(kind, "strip_cdata") == 0) {
        *out_len = si_strip_cdata(input, in_len, buf);
    } else if (strcmp(kind, "strip_inline_styles") == 0) {
        *out_len = si_strip_inline_styles(input, in_len, buf);
    } else if (strcmp(kind, "strip_data_attrs") == 0) {
        *out_len = si_strip_data_attrs(input, in_len, buf);
    } else if (strcmp(kind, "strip_noscript") == 0) {
        *out_len = si_strip_noscript(input, in_len, buf);
    } else if (strcmp(kind, "strip_iframe") == 0) {
        *out_len = si_strip_iframe(input, in_len, buf);
    } else if (strcmp(kind, "strip_svg") == 0) {
        *out_len = si_strip_svg(input, in_len, buf);
    } else if (strcmp(kind, "strip_template") == 0) {
        *out_len = si_strip_template(input, in_len, buf);
    } else if (strcmp(kind, "regex_remove") == 0) {
        size_t plen = strlen(pattern);
        *out_len = axiom_regex_strip_matches(pattern, plen, input, in_len,
                                             buf, (in_len * 2u) + 4096u);
    } else if (strcmp(kind, "regex_extract") == 0) {
        size_t plen = strlen(pattern);
        *out_len = axiom_regex_extract_matches(pattern, plen, input, in_len,
                                               buf, (in_len * 2u) + 4096u);
    } else if (strcmp(kind, "regex_replace") == 0) {
        size_t plen = strlen(pattern);
        *out_len = axiom_regex_replace_matches(pattern, plen, input, in_len,
                                               replacement,
                                               buf, (in_len * 2u) + 4096u);
    } else {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }

    if (flag_require(step->flags) && *out_len == 0) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }

    *out = buf;
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Run full step pipeline                                             */
/* ------------------------------------------------------------------ */

int si_run_pipeline(const axiom_strip_recipe *recipe, const char *input,
                    size_t input_len, char **out, size_t *out_len,
                    strip_pool *pool, uint32_t *steps_fired) {
    if (recipe == NULL || input == NULL || out == NULL || out_len == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    const char *current = input;
    size_t current_len = input_len;
    uint32_t fired = 0;
    float confidence = 1.0f;

    if (recipe->steps == NULL || recipe->step_count == 0) {
        char *buf = (char *)strip_pool_alloc(pool, input_len + 1);
        if (buf == NULL && input_len > 0) {
            return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
        }
        if (input_len > 0) {
            memcpy(buf, input, input_len);
        }
        *out = buf;
        *out_len = input_len;
        if (steps_fired) *steps_fired = 0;
        return AXIOM_STRIP_OK;
    }

    for (size_t i = 0; i < recipe->step_count; ++i) {
        const axiom_strip_step *step = &recipe->steps[i];
        char *next = NULL;
        size_t next_len = 0;

        if (!si_should_fire_step(step, confidence)) {
            char *passthrough = (char *)strip_pool_alloc(pool, current_len + 1);
            if (passthrough == NULL && current_len > 0) {
                return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
            }
            if (current_len > 0) {
                memcpy(passthrough, current, current_len);
            }
            current = passthrough;
            continue;
        }

        int code = si_run_step(step, current, current_len, &next, &next_len,
                               pool, confidence);
        if (code != AXIOM_STRIP_OK) {
            return code;
        }
        if (next_len != current_len || memcmp(next, current, current_len) != 0) {
            fired |= (1u << (i < 31 ? i : 31));
        }

        if (flag_extract(step->flags)) {
            if (next_len > 0 && current_len > 0) {
                float ratio = (float)next_len / (float)current_len;
                confidence *= ratio;
                if (confidence < 0.0f) confidence = 0.0f;
                if (confidence > 1.0f) confidence = 1.0f;
            }
        }
        current = next;
        current_len = next_len;
    }

    *out = (char *)current;
    *out_len = current_len;
    if (steps_fired) *steps_fired = fired;
    return AXIOM_STRIP_OK;
}

/*
 * strip_engine_part7.c â€” Metrics, measurement, and recipe validation.
 *
 * Computes output statistics (token count, signal density, character
 * class distribution), validates recipes structurally, and provides
 * diagnostic helpers for batch processing.
 */


/* ------------------------------------------------------------------ */
/*  Recipe validation (structural)                                     */
/* ------------------------------------------------------------------ */

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
        if (!si_step_kind_known(step->kind)) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
        if (step->kind != NULL &&
            (strcmp(step->kind, "keep_between") == 0 ||
             strcmp(step->kind, "keep_between_all") == 0 ||
             strcmp(step->kind, "strip_between") == 0)) {
            if (step->pattern == NULL || step->replacement == NULL) {
                return AXIOM_STRIP_ERR_BAD_RECIPE;
            }
        }
        if (step->kind != NULL &&
            (strcmp(step->kind, "regex_remove") == 0 ||
             strcmp(step->kind, "regex_extract") == 0 ||
             strcmp(step->kind, "regex_replace") == 0)) {
            if (step->pattern == NULL || step->pattern[0] == '\0') {
                return AXIOM_STRIP_ERR_BAD_RECIPE;
            }
        }
        if (step->confidence < 0.0f || step->confidence > 1.0f) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
    }
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Output measurement                                                 */
/* ------------------------------------------------------------------ */

int axiom_strip_measure(
    const uint8_t *input,
    size_t input_len,
    const uint8_t *output,
    size_t output_len,
    axiom_strip_metrics *metrics
) {
    if (metrics == NULL ||
        (input == NULL && input_len > 0u) ||
        (output == NULL && output_len > 0u)) {
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
    if (output_len > 0u && output[output_len - 1u] != '\n') {
        metrics->line_count++;
    }
    if (input_len > 0u) {
        metrics->output_ratio = (double)output_len / (double)input_len;
    }
    if (output_len > 0u) {
        metrics->signal_density =
            (double)(metrics->ascii_letters + metrics->ascii_digits) /
            (double)output_len;
    }
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Batch metrics aggregation                                          */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t  total_input_bytes;
    size_t  total_output_bytes;
    size_t  total_tokens;
    size_t  items_processed;
    size_t  items_failed;
    double  min_ratio;
    double  max_ratio;
    double  sum_ratio;
    double  sum_density;
    double  min_density;
    double  max_density;
} axiom_batch_metrics;

void axiom_batch_metrics_init(axiom_batch_metrics *bm) {
    if (bm == NULL) return;
    memset(bm, 0, sizeof(*bm));
    bm->min_ratio = 1.0;
    bm->max_ratio = 0.0;
    bm->min_density = 1.0;
    bm->max_density = 0.0;
}

void axiom_batch_metrics_add(axiom_batch_metrics *bm,
                             const axiom_strip_metrics *m,
                             bool success) {
    if (bm == NULL) return;
    if (!success) {
        bm->items_failed++;
        return;
    }
    bm->items_processed++;
    bm->total_input_bytes += m->input_bytes;
    bm->total_output_bytes += m->output_bytes;
    bm->total_tokens += m->token_count;
    bm->sum_ratio += m->output_ratio;
    bm->sum_density += m->signal_density;
    if (m->output_ratio < bm->min_ratio) bm->min_ratio = m->output_ratio;
    if (m->output_ratio > bm->max_ratio) bm->max_ratio = m->output_ratio;
    if (m->signal_density < bm->min_density) bm->min_density = m->signal_density;
    if (m->signal_density > bm->max_density) bm->max_density = m->signal_density;
}

double axiom_batch_metrics_avg_ratio(const axiom_batch_metrics *bm) {
    if (bm == NULL || bm->items_processed == 0) return 0.0;
    return bm->sum_ratio / (double)bm->items_processed;
}

double axiom_batch_metrics_avg_density(const axiom_batch_metrics *bm) {
    if (bm == NULL || bm->items_processed == 0) return 0.0;
    return bm->sum_density / (double)bm->items_processed;
}

/* ------------------------------------------------------------------ */
/*  Format metrics as a single-line JSON string                        */
/* ------------------------------------------------------------------ */

int axiom_strip_metrics_json(const axiom_strip_metrics *m,
                             char *buf, size_t buf_capacity) {
    if (m == NULL || buf == NULL || buf_capacity < 128) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int n = snprintf(buf, buf_capacity,
        "{\"input_bytes\":%zu,\"output_bytes\":%zu,"
        "\"tokens\":%zu,\"lines\":%zu,"
        "\"ratio\":%.6f,\"density\":%.6f,"
        "\"letters\":%zu,\"digits\":%zu,"
        "\"ws\":%zu,\"punct\":%zu}",
        m->input_bytes, m->output_bytes,
        m->token_count, m->line_count,
        m->output_ratio, m->signal_density,
        m->ascii_letters, m->ascii_digits,
        m->whitespace, m->punctuation
    );
    if (n < 0 || (size_t)n >= buf_capacity) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Compare two metrics for regression detection                       */
/* ------------------------------------------------------------------ */

typedef struct {
    double ratio_delta;
    double density_delta;
    int    token_delta;
    bool   regression;
} axiom_metrics_diff;

void axiom_strip_metrics_diff(const axiom_strip_metrics *before,
                              const axiom_strip_metrics *after,
                              axiom_metrics_diff *diff) {
    if (before == NULL || after == NULL || diff == NULL) return;
    memset(diff, 0, sizeof(*diff));
    diff->ratio_delta = after->output_ratio - before->output_ratio;
    diff->density_delta = after->signal_density - before->signal_density;
    diff->token_delta = (int)after->token_count - (int)before->token_count;
    diff->regression = (diff->density_delta < -0.05) ||
                       (diff->ratio_delta > 0.10);
}

/* ------------------------------------------------------------------ */
/*  Signal quality classifier                                          */
/* ------------------------------------------------------------------ */

typedef enum {
    SIGNAL_QUALITY_EXCELLENT,
    SIGNAL_QUALITY_GOOD,
    SIGNAL_QUALITY_FAIR,
    SIGNAL_QUALITY_POOR,
    SIGNAL_QUALITY_EMPTY
} axiom_signal_quality;

axiom_signal_quality axiom_strip_classify_quality(
    const axiom_strip_metrics *m
) {
    if (m == NULL || m->output_bytes == 0) {
        return SIGNAL_QUALITY_EMPTY;
    }
    if (m->signal_density >= 0.85 && m->output_ratio <= 0.10) {
        return SIGNAL_QUALITY_EXCELLENT;
    }
    if (m->signal_density >= 0.70 && m->output_ratio <= 0.20) {
        return SIGNAL_QUALITY_GOOD;
    }
    if (m->signal_density >= 0.50 && m->output_ratio <= 0.30) {
        return SIGNAL_QUALITY_FAIR;
    }
    return SIGNAL_QUALITY_POOR;
}

const char *axiom_signal_quality_str(axiom_signal_quality q) {
    switch (q) {
    case SIGNAL_QUALITY_EXCELLENT: return "excellent";
    case SIGNAL_QUALITY_GOOD:     return "good";
    case SIGNAL_QUALITY_FAIR:     return "fair";
    case SIGNAL_QUALITY_POOR:     return "poor";
    case SIGNAL_QUALITY_EMPTY:    return "empty";
    default:                      return "unknown";
    }
}

/* ------------------------------------------------------------------ */
/*  Throughput tracker (for batch_runner stats thread)                  */
/* ------------------------------------------------------------------ */

typedef struct {
    size_t   bytes_processed;
    size_t   items_processed;
    double   elapsed_seconds;
} axiom_throughput;

double axiom_throughput_mbps(const axiom_throughput *t) {
    if (t == NULL || t->elapsed_seconds <= 0.0) return 0.0;
    return (double)t->bytes_processed / (1024.0 * 1024.0) / t->elapsed_seconds;
}

double axiom_throughput_items_per_sec(const axiom_throughput *t) {
    if (t == NULL || t->elapsed_seconds <= 0.0) return 0.0;
    return (double)t->items_processed / t->elapsed_seconds;
}

int axiom_throughput_json(const axiom_throughput *t,
                          char *buf, size_t buf_capacity) {
    if (t == NULL || buf == NULL || buf_capacity < 128) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int n = snprintf(buf, buf_capacity,
        "{\"bytes\":%zu,\"items\":%zu,"
        "\"elapsed_s\":%.3f,\"mb_per_s\":%.3f,"
        "\"items_per_s\":%.1f}",
        t->bytes_processed, t->items_processed,
        t->elapsed_seconds,
        axiom_throughput_mbps(t),
        axiom_throughput_items_per_sec(t)
    );
    if (n < 0 || (size_t)n >= buf_capacity) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    return AXIOM_STRIP_OK;
}

/*
 * strip_engine_part8.c â€” Main public API.
 *
 * axiom_strip_apply(), axiom_strip_apply_with_pool(), and
 * axiom_strip_apply_default_html().  These are the entry points
 * callers use.  The global-pool variant maintains backward compat
 * with single-threaded callers; the with_pool variant is thread-safe.
 */


/* ------------------------------------------------------------------ */
/*  Global fallback pool (single-threaded legacy path)                 */
/* ------------------------------------------------------------------ */

static strip_pool g_pool = {0};

static int ensure_global_pool(void) {
    if (g_pool.base != NULL) {
        strip_pool_reset(&g_pool);
        return AXIOM_STRIP_OK;
    }
    return strip_pool_init(&g_pool, AXIOM_POOL_BYTES);
}

#define AXIOM_RATIO_SMALL_SIGNAL_FLOOR 32u

static bool si_allow_small_signal_ratio(const axiom_strip_recipe *recipe,
                                        size_t input_len,
                                        size_t output_len,
                                        uint32_t fired) {
    if (recipe == NULL || recipe->step_count == 0 || fired == 0u) {
        return false;
    }
    if (output_len >= input_len) {
        return false;
    }
    return output_len <= AXIOM_RATIO_SMALL_SIGNAL_FLOOR;
}

/* ------------------------------------------------------------------ */
/*  Core apply â€” thread-safe with explicit pool                        */
/* ------------------------------------------------------------------ */

int axiom_strip_apply_with_pool(
    const uint8_t            *input,
    size_t                    input_len,
    const axiom_strip_recipe *recipe,
    uint8_t                  *output,
    size_t                    output_capacity,
    axiom_strip_result       *result,
    strip_pool               *pool
) {
    if (input == NULL || output == NULL || result == NULL || pool == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(result, 0, sizeof(*result));
    (void)input_len; /* input_len used below; silence unused-on-early-return */

    if (input_len == 0) {
        result->code = AXIOM_STRIP_OK;
        result->bytes_written = 0;
        result->crc32 = 0;
        return AXIOM_STRIP_OK;
    }

    int val = axiom_strip_validate_recipe(recipe);
    if (val != AXIOM_STRIP_OK) {
        result->code = val;
        return val;
    }

    strip_pool_reset(pool);

    char *pipeline_out = NULL;
    size_t pipeline_len = 0;
    uint32_t fired = 0;

    if (recipe == NULL || recipe->steps == NULL || recipe->step_count == 0) {
        pipeline_out = (char *)input;
        pipeline_len = input_len;
    } else {
        int rc = si_run_pipeline(recipe, (const char *)input, input_len,
                                 &pipeline_out, &pipeline_len, pool, &fired);
        if (rc != AXIOM_STRIP_OK) {
            result->code = rc;
            return rc;
        }
    }

    double ratio = (recipe != NULL && recipe->max_output_ratio > 0.0)
                   ? recipe->max_output_ratio : 1.0;
    if (input_len > 0 && (double)pipeline_len > (double)input_len * ratio &&
        !si_allow_small_signal_ratio(recipe, input_len, pipeline_len, fired)) {
        result->code = AXIOM_STRIP_ERR_REDUCTION_RATIO;
        return result->code;
    }

    if (pipeline_len > output_capacity) {
        result->code = AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
        return result->code;
    }

    memcpy(output, pipeline_out, pipeline_len);
    result->bytes_written = pipeline_len;
    result->crc32 = axiom_crc32(output, pipeline_len);
    result->steps_fired = fired;
    result->code = AXIOM_STRIP_OK;
    return AXIOM_STRIP_OK;
}

/* ------------------------------------------------------------------ */
/*  Legacy apply â€” uses global pool (not thread-safe)                  */
/* ------------------------------------------------------------------ */

int axiom_strip_apply(
    const uint8_t            *input,
    size_t                    input_len,
    const axiom_strip_recipe *recipe,
    uint8_t                  *output,
    size_t                    output_capacity,
    axiom_strip_result       *result
) {
    if (input == NULL || output == NULL || result == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    int init = ensure_global_pool();
    if (init != AXIOM_STRIP_OK) {
        return init;
    }
    return axiom_strip_apply_with_pool(
        input, input_len, recipe, output, output_capacity, result, &g_pool
    );
}

/* ------------------------------------------------------------------ */
/*  Default HTML pipeline                                              */
/* ------------------------------------------------------------------ */

int axiom_strip_apply_default_html(
    const uint8_t       *input,
    size_t               input_len,
    uint8_t             *output,
    size_t               output_capacity,
    axiom_strip_result  *result
) {
    axiom_strip_step steps[] = {
        {.kind = "strip_comments",      .pattern = "", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "script", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "style", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "noscript", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "svg", .replacement = ""},
        {.kind = "strip_attrs",         .pattern = "", .replacement = ""},
        {.kind = "decode_entities",     .pattern = "", .replacement = ""},
        {.kind = "strip_html",          .pattern = "", .replacement = ""},
        {.kind = "collapse_ws",         .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {
        .steps = steps,
        .step_count = sizeof(steps) / sizeof(steps[0]),
        .max_output_ratio = AXIOM_STRIP_DEFAULT_RATIO,
    };
    return axiom_strip_apply(input, input_len, &recipe, output,
                             output_capacity, result);
}

/* ------------------------------------------------------------------ */
/*  Aggressive strip pipeline (more steps)                             */
/* ------------------------------------------------------------------ */

int axiom_strip_apply_aggressive(
    const uint8_t       *input,
    size_t               input_len,
    uint8_t             *output,
    size_t               output_capacity,
    axiom_strip_result  *result
) {
    axiom_strip_step steps[] = {
        {.kind = "strip_comments",      .pattern = "", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "script", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "style", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "noscript", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "svg", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "iframe", .replacement = ""},
        {.kind = "strip_tag",           .pattern = "template", .replacement = ""},
        {.kind = "strip_inline_styles", .pattern = "", .replacement = ""},
        {.kind = "strip_data_attrs",    .pattern = "", .replacement = ""},
        {.kind = "strip_attrs",         .pattern = "", .replacement = ""},
        {.kind = "decode_entities",     .pattern = "", .replacement = ""},
        {.kind = "strip_html",          .pattern = "", .replacement = ""},
        {.kind = "normalize_ws",        .pattern = "", .replacement = ""},
        {.kind = "collapse_ws",         .pattern = "", .replacement = ""},
        {.kind = "remove_blank_lines",  .pattern = "", .replacement = ""},
        {.kind = "trim_lines",          .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {
        .steps = steps,
        .step_count = sizeof(steps) / sizeof(steps[0]),
        .max_output_ratio = 0.15,
    };
    return axiom_strip_apply(input, input_len, &recipe, output,
                             output_capacity, result);
}

/* ------------------------------------------------------------------ */
/*  Strip and measure in one call                                      */
/* ------------------------------------------------------------------ */

int axiom_strip_and_measure(
    const uint8_t            *input,
    size_t                    input_len,
    const axiom_strip_recipe *recipe,
    uint8_t                  *output,
    size_t                    output_capacity,
    axiom_strip_result       *result,
    axiom_strip_metrics      *metrics
) {
    int rc = axiom_strip_apply(input, input_len, recipe, output,
                               output_capacity, result);
    if (rc != AXIOM_STRIP_OK) {
        return rc;
    }
    return axiom_strip_measure(input, input_len, output,
                               result->bytes_written, metrics);
}

/* ------------------------------------------------------------------ */
/*  Thread-safe strip and measure                                      */
/* ------------------------------------------------------------------ */

int axiom_strip_and_measure_with_pool(
    const uint8_t            *input,
    size_t                    input_len,
    const axiom_strip_recipe *recipe,
    uint8_t                  *output,
    size_t                    output_capacity,
    axiom_strip_result       *result,
    axiom_strip_metrics      *metrics,
    strip_pool               *pool
) {
    int rc = axiom_strip_apply_with_pool(input, input_len, recipe, output,
                                         output_capacity, result, pool);
    if (rc != AXIOM_STRIP_OK) {
        return rc;
    }
    return axiom_strip_measure(input, input_len, output,
                               result->bytes_written, metrics);
}
