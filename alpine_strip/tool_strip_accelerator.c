/*
 * tool_strip_accelerator.c - native AXIOM tools bridge acceleration.
 *
 * The regular strip_engine API is intentionally small: callers provide a
 * recipe and one buffer, and the engine runs that recipe immediately. That is
 * correct for low-volume callers, but the tools bridge creates a different
 * workload:
 *
 *   - captured WebFetchTool / browser snapshots arrive as temporary artifacts;
 *   - the same topology recipe is applied to many artifacts in a batch;
 *   - regex-heavy cleanup must not compile patterns for every input item;
 *   - watermarked tool artifacts must never be mistaken for clean signal.
 *
 * This file adds a compiled-plan layer and a JSON request surface. It does not
 * replace strip_engine.c. It reuses its pool allocator, HTML/text helpers, and
 * PCRE2-backed axiom_regex wrapper. When PCRE2 is available to strip_engine.c,
 * compiled plans keep PCRE2 code handles hot across items. When PCRE2 is not
 * available, the same plan API runs against the built-in regex fallback so the
 * development and test environment stays deterministic.
 */

#include "tool_strip_accelerator.h"
#include "strip_engine_internal.h"

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TOOL_JSON_SCRATCH           8192u
#define TOOL_MAX_RECIPE_STEPS       AXIOM_TOOL_PLAN_MAX_STEPS
#define TOOL_STEP_KIND_LEN          48u
#define TOOL_STEP_REPLACEMENT_LEN   2048u
#define TOOL_MIN_SMALL_SIGNAL_BYTES 32u

typedef struct {
    char         kind[TOOL_STEP_KIND_LEN];
    char         pattern[AXIOM_STRIP_MAX_PATTERN + 1u];
    char         replacement[TOOL_STEP_REPLACEMENT_LEN];
    uint8_t      flags;
    float        confidence;
    bool         is_regex;
    bool         is_extract;
    bool         is_remove;
    axiom_regex *compiled;
} axiom_plan_step;

struct axiom_strip_plan {
    axiom_plan_step             steps[TOOL_MAX_RECIPE_STEPS];
    size_t                      step_count;
    char                        topology_class[AXIOM_STRIP_TOPOLOGY_LEN];
    uint32_t                    checksum;
    double                      max_output_ratio;
    axiom_tool_snapshot_profile profile;
    size_t                      regex_step_count;
};

typedef struct {
    const char *start;
    size_t      len;
} json_slice;

static void profile_set_kind_flags(axiom_tool_snapshot_profile *profile);
static int json_get_string(const char *json, const char *key, char *out, size_t out_cap);
static int json_get_number(const char *json, const char *key, double *out);
static int json_find_value(const char *json, const char *key, json_slice *slice);
static int json_copy_string(const char *src, size_t len, char *out, size_t out_cap);
static void json_escape(const char *input, char *out, size_t out_cap);
static void safe_copy(char *dst, size_t dst_cap, const char *src);
static bool str_eq_ci(const char *a, const char *b);
static bool has_token_ci(const char *haystack, const char *needle);
static bool is_regex_kind(const char *kind);
static int copy_step_to_plan(axiom_plan_step *dst, const axiom_strip_step *src);
static int plan_step_to_recipe_step(const axiom_plan_step *step, axiom_strip_step *out);
static char *pool_alloc_stage(strip_pool *pool, size_t in_len);
static size_t strip_tool_watermark(const char *input, size_t len, char *out, size_t *removed);
static int apply_plan_step(const axiom_plan_step *step, const char *input, size_t in_len,
                           char **out, size_t *out_len, strip_pool *pool,
                           uint32_t *step_bit, axiom_tool_strip_stats *stats);
static bool output_ratio_allowed(const axiom_strip_plan *plan, size_t input_len,
                                 size_t output_len, uint32_t fired);
static void stats_from_result(axiom_tool_strip_stats *stats, const uint8_t *input,
                              size_t input_len, const uint8_t *output,
                              size_t output_len, const axiom_strip_result *result);
static int parse_request_input(const char *json, char *out, size_t out_cap);
static int format_response_json(const axiom_tool_snapshot_profile *profile,
                                const axiom_tool_strip_stats *stats,
                                const axiom_strip_result *result,
                                char *out, size_t out_cap);
static uint32_t fnv1a_update(uint32_t hash, const char *data);
static void recipe_defaults_for_html(axiom_strip_step *steps, size_t *count);
static void recipe_defaults_for_markdown(axiom_strip_step *steps, size_t *count);
static void recipe_defaults_for_metadata(axiom_strip_step *steps, size_t *count);
static void recipe_defaults_for_plain(axiom_strip_step *steps, size_t *count);

const char *axiom_tool_strip_version(void) {
    return AXIOM_TOOL_STRIP_VERSION;
}

void axiom_tool_profile_init(axiom_tool_snapshot_profile *profile) {
    if (profile == NULL) {
        return;
    }
    memset(profile, 0, sizeof(*profile));
    safe_copy(profile->source_tool, sizeof(profile->source_tool), "AlpineStripTool");
    safe_copy(profile->artifact_kind, sizeof(profile->artifact_kind), "raw_html");
    safe_copy(profile->topology_class, sizeof(profile->topology_class), "GENERIC_HTML");
    profile->flags = AXIOM_TOOL_PROFILE_FLAG_WATERMARK |
                     AXIOM_TOOL_PROFILE_FLAG_HTML |
                     AXIOM_TOOL_PROFILE_FLAG_SNAPSHOT;
    profile->max_output_ratio = AXIOM_STRIP_DEFAULT_RATIO;
}

int axiom_tool_profile_from_json(
    const char *json,
    axiom_tool_snapshot_profile *profile
) {
    if (json == NULL || profile == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    axiom_tool_profile_init(profile);

    (void)json_get_string(json, "tool", profile->source_tool, sizeof(profile->source_tool));
    (void)json_get_string(json, "source_tool", profile->source_tool, sizeof(profile->source_tool));
    (void)json_get_string(json, "artifact_kind", profile->artifact_kind, sizeof(profile->artifact_kind));
    (void)json_get_string(json, "kind", profile->artifact_kind, sizeof(profile->artifact_kind));
    (void)json_get_string(json, "url", profile->url, sizeof(profile->url));
    (void)json_get_string(json, "query", profile->query, sizeof(profile->query));
    (void)json_get_string(json, "topology_class", profile->topology_class, sizeof(profile->topology_class));
    if (profile->topology_class[0] == '\0') {
        safe_copy(profile->topology_class, sizeof(profile->topology_class), "GENERIC_HTML");
    }

    double ratio = 0.0;
    if (json_get_number(json, "max_output_ratio", &ratio) && ratio > 0.0 && ratio <= 1.0) {
        profile->max_output_ratio = ratio;
    }
    profile_set_kind_flags(profile);
    return AXIOM_STRIP_OK;
}

int axiom_tool_profile_build_recipe(
    const axiom_tool_snapshot_profile *profile,
    axiom_strip_step *steps,
    size_t step_capacity,
    axiom_strip_recipe *recipe
) {
    if (profile == NULL || steps == NULL || recipe == NULL || step_capacity == 0) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }

    memset(steps, 0, sizeof(axiom_strip_step) * step_capacity);
    memset(recipe, 0, sizeof(*recipe));

    size_t count = 0;
    size_t required = 3u;
    if ((profile->flags & AXIOM_TOOL_PROFILE_FLAG_HTML) != 0u) {
        required = 17u;
        if (step_capacity < required) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
        recipe_defaults_for_html(steps, &count);
    } else if ((profile->flags & AXIOM_TOOL_PROFILE_FLAG_MARKDOWN) != 0u) {
        required = 6u;
        if (step_capacity < required) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
        recipe_defaults_for_markdown(steps, &count);
    } else if ((profile->flags & AXIOM_TOOL_PROFILE_FLAG_METADATA) != 0u) {
        required = 4u;
        if (step_capacity < required) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
        recipe_defaults_for_metadata(steps, &count);
    } else {
        required = 3u;
        if (step_capacity < required) {
            return AXIOM_STRIP_ERR_BAD_RECIPE;
        }
        recipe_defaults_for_plain(steps, &count);
    }

    if (count > step_capacity) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    recipe->steps = steps;
    recipe->step_count = count;
    recipe->max_output_ratio = profile->max_output_ratio > 0.0
        ? profile->max_output_ratio
        : AXIOM_STRIP_DEFAULT_RATIO;
    safe_copy(recipe->topology_class, sizeof(recipe->topology_class), profile->topology_class);

    uint32_t checksum = 2166136261u;
    checksum = fnv1a_update(checksum, recipe->topology_class);
    for (size_t i = 0; i < count; ++i) {
        checksum = fnv1a_update(checksum, steps[i].kind ? steps[i].kind : "");
        checksum = fnv1a_update(checksum, steps[i].pattern ? steps[i].pattern : "");
        checksum = fnv1a_update(checksum, steps[i].replacement ? steps[i].replacement : "");
    }
    recipe->checksum = checksum;
    return AXIOM_STRIP_OK;
}

int axiom_strip_plan_compile(
    const axiom_strip_recipe *recipe,
    const axiom_tool_snapshot_profile *profile,
    axiom_strip_plan **out_plan
) {
    if (out_plan == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    *out_plan = NULL;
    int valid = axiom_strip_validate_recipe(recipe);
    if (valid != AXIOM_STRIP_OK) {
        return valid;
    }

    axiom_strip_plan *plan = (axiom_strip_plan *)calloc(1, sizeof(axiom_strip_plan));
    if (plan == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }

    if (profile != NULL) {
        memcpy(&plan->profile, profile, sizeof(plan->profile));
    } else {
        axiom_tool_profile_init(&plan->profile);
    }

    if (recipe == NULL || recipe->steps == NULL || recipe->step_count == 0) {
        plan->step_count = 0;
        plan->max_output_ratio = 1.0;
        safe_copy(plan->topology_class, sizeof(plan->topology_class), plan->profile.topology_class);
        *out_plan = plan;
        return AXIOM_STRIP_OK;
    }

    if (recipe->step_count > TOOL_MAX_RECIPE_STEPS) {
        free(plan);
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    plan->step_count = recipe->step_count;
    plan->max_output_ratio = recipe->max_output_ratio > 0.0
        ? recipe->max_output_ratio
        : AXIOM_STRIP_DEFAULT_RATIO;
    plan->checksum = recipe->checksum;
    safe_copy(plan->topology_class, sizeof(plan->topology_class),
              recipe->topology_class[0] ? recipe->topology_class : plan->profile.topology_class);

    for (size_t i = 0; i < recipe->step_count; ++i) {
        int rc = copy_step_to_plan(&plan->steps[i], &recipe->steps[i]);
        if (rc != AXIOM_STRIP_OK) {
            axiom_strip_plan_free(plan);
            return rc;
        }
        if (plan->steps[i].is_regex) {
            int ec = AXIOM_STRIP_OK;
            plan->steps[i].compiled = axiom_regex_compile(
                plan->steps[i].pattern,
                strlen(plan->steps[i].pattern),
                0,
                &ec
            );
            if (plan->steps[i].compiled == NULL) {
                axiom_strip_plan_free(plan);
                return ec == AXIOM_STRIP_OK ? AXIOM_STRIP_ERR_PCRE2_COMPILE : ec;
            }
            (void)axiom_regex_jit_hint(plan->steps[i].compiled);
            plan->regex_step_count++;
        }
    }

    *out_plan = plan;
    return AXIOM_STRIP_OK;
}

void axiom_strip_plan_free(axiom_strip_plan *plan) {
    if (plan == NULL) {
        return;
    }
    for (size_t i = 0; i < plan->step_count; ++i) {
        if (plan->steps[i].compiled != NULL) {
            axiom_regex_free(plan->steps[i].compiled);
            plan->steps[i].compiled = NULL;
        }
    }
    free(plan);
}

int axiom_strip_plan_apply(
    const axiom_strip_plan *plan,
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t output_capacity,
    axiom_strip_result *result,
    axiom_tool_strip_stats *stats,
    strip_pool *pool
) {
    if (plan == NULL || input == NULL || output == NULL || result == NULL || pool == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(result, 0, sizeof(*result));
    if (stats != NULL) {
        memset(stats, 0, sizeof(*stats));
        stats->input_bytes = input_len;
        stats->regex_steps_compiled = plan->regex_step_count;
    }
    if (input_len == 0) {
        result->code = AXIOM_STRIP_OK;
        return AXIOM_STRIP_OK;
    }

    strip_pool_reset(pool);
    const char *current = (const char *)input;
    size_t current_len = input_len;
    uint32_t fired = 0;

    if ((plan->profile.flags & AXIOM_TOOL_PROFILE_FLAG_WATERMARK) != 0u) {
        char *clean = pool_alloc_stage(pool, current_len);
        if (clean == NULL) {
            result->code = AXIOM_STRIP_ERR_POOL_EXHAUSTED;
            return result->code;
        }
        size_t removed = 0;
        size_t clean_len = strip_tool_watermark(current, current_len, clean, &removed);
        if (removed > 0) {
            current = clean;
            current_len = clean_len;
            if (stats != NULL) {
                stats->watermark_bytes_removed += removed;
            }
        }
    }

    if (plan->step_count == 0) {
        if (current_len > output_capacity) {
            result->code = AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
            return result->code;
        }
        memcpy(output, current, current_len);
        result->bytes_written = current_len;
        result->crc32 = axiom_crc32(output, current_len);
        result->code = AXIOM_STRIP_OK;
        stats_from_result(stats, input, input_len, output, current_len, result);
        return AXIOM_STRIP_OK;
    }

    for (size_t i = 0; i < plan->step_count; ++i) {
        char *next = NULL;
        size_t next_len = 0;
        uint32_t bit = (1u << (i < 31u ? i : 31u));
        int rc = apply_plan_step(&plan->steps[i], current, current_len,
                                 &next, &next_len, pool, &bit, stats);
        if (rc != AXIOM_STRIP_OK) {
            result->code = rc;
            return rc;
        }
        if (next_len != current_len || memcmp(next, current, current_len) != 0) {
            fired |= bit;
        }
        current = next;
        current_len = next_len;
    }

    if (input_len > 0 &&
        (double)current_len > (double)input_len * plan->max_output_ratio &&
        !output_ratio_allowed(plan, input_len, current_len, fired)) {
        result->code = AXIOM_STRIP_ERR_REDUCTION_RATIO;
        return result->code;
    }
    if (current_len > output_capacity) {
        result->code = AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
        return result->code;
    }

    memcpy(output, current, current_len);
    if (current_len < output_capacity) {
        output[current_len] = '\0';
    }
    result->bytes_written = current_len;
    result->crc32 = axiom_crc32(output, current_len);
    result->steps_fired = fired;
    result->code = AXIOM_STRIP_OK;
    stats_from_result(stats, input, input_len, output, current_len, result);
    return AXIOM_STRIP_OK;
}

int axiom_tool_strip_request_json(
    const char *request_json,
    char *signal_output,
    size_t signal_capacity,
    char *response_json,
    size_t response_capacity,
    strip_pool *pool
) {
    if (request_json == NULL || signal_output == NULL || response_json == NULL || pool == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }

    axiom_tool_snapshot_profile profile;
    int rc = axiom_tool_profile_from_json(request_json, &profile);
    if (rc != AXIOM_STRIP_OK) {
        return rc;
    }

    char *input = (char *)malloc(TOOL_JSON_SCRATCH);
    if (input == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    rc = parse_request_input(request_json, input, TOOL_JSON_SCRATCH);
    if (rc != AXIOM_STRIP_OK) {
        free(input);
        return rc;
    }
    size_t input_len = strlen(input);

    axiom_strip_step steps[TOOL_MAX_RECIPE_STEPS];
    axiom_strip_recipe recipe;
    rc = axiom_tool_profile_build_recipe(&profile, steps, TOOL_MAX_RECIPE_STEPS, &recipe);
    if (rc != AXIOM_STRIP_OK) {
        free(input);
        return rc;
    }

    axiom_strip_plan *plan = NULL;
    rc = axiom_strip_plan_compile(&recipe, &profile, &plan);
    if (rc != AXIOM_STRIP_OK) {
        free(input);
        return rc;
    }

    axiom_strip_result result;
    axiom_tool_strip_stats stats;
    rc = axiom_strip_plan_apply(plan, (const uint8_t *)input, input_len,
                                (uint8_t *)signal_output, signal_capacity,
                                &result, &stats, pool);
    axiom_strip_plan_free(plan);
    free(input);
    if (rc != AXIOM_STRIP_OK) {
        stats.code = rc;
        (void)format_response_json(&profile, &stats, &result, response_json, response_capacity);
        return rc;
    }
    return format_response_json(&profile, &stats, &result, response_json, response_capacity);
}

int axiom_tool_strip_make_queue_line(
    const char *url,
    int slot_idx,
    const char *input_path,
    const char *output_path,
    char *out_jsonl,
    size_t out_capacity
) {
    if (url == NULL || input_path == NULL || output_path == NULL ||
        out_jsonl == NULL || out_capacity == 0) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }

    char esc_url[AXIOM_TOOL_URL_LEN * 2u];
    char esc_in[AXIOM_TOOL_URL_LEN * 2u];
    char esc_out[AXIOM_TOOL_URL_LEN * 2u];
    json_escape(url, esc_url, sizeof(esc_url));
    json_escape(input_path, esc_in, sizeof(esc_in));
    json_escape(output_path, esc_out, sizeof(esc_out));
    int n = snprintf(out_jsonl, out_capacity,
                     "{\"url\":\"%s\",\"slot_idx\":%d,\"input_path\":\"%s\",\"output_path\":\"%s\"}",
                     esc_url, slot_idx, esc_in, esc_out);
    if (n < 0 || (size_t)n >= out_capacity) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    return AXIOM_STRIP_OK;
}

bool axiom_tool_strip_is_snapshot_profile(
    const axiom_tool_snapshot_profile *profile
) {
    return profile != NULL && (profile->flags & AXIOM_TOOL_PROFILE_FLAG_SNAPSHOT) != 0u;
}

static void recipe_defaults_for_html(axiom_strip_step *steps, size_t *count) {
    size_t i = 0;
    steps[i++] = (axiom_strip_step){.kind = "regex_remove", .pattern = "AXIOM SNAPSHOT ARTIFACT[^\\n]*", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_comments", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "script", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "style", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "noscript", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "svg", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "iframe", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "template", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "nav", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "footer", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_tag", .pattern = "aside", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_attrs", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "decode_entities", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "strip_html", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "regex_replace", .pattern = "[\\t\\r\\n ]{2,}", .replacement = " "};
    steps[i++] = (axiom_strip_step){.kind = "collapse_ws", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "trim_lines", .pattern = "", .replacement = ""};
    *count = i;
}

static void recipe_defaults_for_markdown(axiom_strip_step *steps, size_t *count) {
    size_t i = 0;
    steps[i++] = (axiom_strip_step){.kind = "regex_remove", .pattern = "<!-- AXIOM SNAPSHOT ARTIFACT[^>]*-->", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "regex_remove", .pattern = "!\\[[^\\]]*\\]\\([^\\)]*\\)", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "regex_replace", .pattern = "\\[[^\\]]+\\]\\([^\\)]*\\)", .replacement = " link "};
    steps[i++] = (axiom_strip_step){.kind = "regex_remove", .pattern = "[#*_`>{}\\[\\]]+", .replacement = " "};
    steps[i++] = (axiom_strip_step){.kind = "collapse_ws", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "trim_lines", .pattern = "", .replacement = ""};
    *count = i;
}

static void recipe_defaults_for_metadata(axiom_strip_step *steps, size_t *count) {
    size_t i = 0;
    steps[i++] = (axiom_strip_step){.kind = "regex_remove", .pattern = "AXIOM SNAPSHOT ARTIFACT[^\\n]*", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "regex_extract", .pattern = "https?://[^\" ,}\\]]+", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "deduplicate_lines", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "trim_lines", .pattern = "", .replacement = ""};
    *count = i;
}

static void recipe_defaults_for_plain(axiom_strip_step *steps, size_t *count) {
    size_t i = 0;
    steps[i++] = (axiom_strip_step){.kind = "regex_remove", .pattern = "AXIOM SNAPSHOT ARTIFACT[^\\n]*", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "collapse_ws", .pattern = "", .replacement = ""};
    steps[i++] = (axiom_strip_step){.kind = "trim_lines", .pattern = "", .replacement = ""};
    *count = i;
}

static void profile_set_kind_flags(axiom_tool_snapshot_profile *profile) {
    profile->flags &= ~(AXIOM_TOOL_PROFILE_FLAG_HTML |
                        AXIOM_TOOL_PROFILE_FLAG_MARKDOWN |
                        AXIOM_TOOL_PROFILE_FLAG_METADATA);
    profile->flags |= AXIOM_TOOL_PROFILE_FLAG_WATERMARK;
    if (profile->source_tool[0] == '\0') {
        safe_copy(profile->source_tool, sizeof(profile->source_tool), "AlpineStripTool");
    }
    if (profile->artifact_kind[0] == '\0') {
        safe_copy(profile->artifact_kind, sizeof(profile->artifact_kind), "raw_html");
    }
    if (has_token_ci(profile->source_tool, "WebFetch") ||
        has_token_ci(profile->source_tool, "Browser") ||
        str_eq_ci(profile->artifact_kind, "raw_html") ||
        str_eq_ci(profile->artifact_kind, "rendered_html") ||
        str_eq_ci(profile->artifact_kind, "html")) {
        profile->flags |= AXIOM_TOOL_PROFILE_FLAG_HTML;
        if (profile->max_output_ratio <= 0.0) {
            profile->max_output_ratio = AXIOM_STRIP_DEFAULT_RATIO;
        }
    } else if (str_eq_ci(profile->artifact_kind, "markdown") ||
               str_eq_ci(profile->artifact_kind, "md")) {
        profile->flags |= AXIOM_TOOL_PROFILE_FLAG_MARKDOWN;
        if (profile->max_output_ratio == AXIOM_STRIP_DEFAULT_RATIO) {
            profile->max_output_ratio = 0.75;
        }
    } else if (str_eq_ci(profile->artifact_kind, "metadata") ||
               has_token_ci(profile->source_tool, "WebSearch")) {
        profile->flags |= AXIOM_TOOL_PROFILE_FLAG_METADATA;
        if (profile->max_output_ratio == AXIOM_STRIP_DEFAULT_RATIO) {
            profile->max_output_ratio = 1.0;
        }
    }
    if (has_token_ci(profile->source_tool, "Tool") ||
        profile->url[0] != '\0') {
        profile->flags |= AXIOM_TOOL_PROFILE_FLAG_SNAPSHOT;
    }
}

static int copy_step_to_plan(axiom_plan_step *dst, const axiom_strip_step *src) {
    if (dst == NULL || src == NULL || src->kind == NULL) {
        return AXIOM_STRIP_ERR_BAD_RECIPE;
    }
    memset(dst, 0, sizeof(*dst));
    safe_copy(dst->kind, sizeof(dst->kind), src->kind);
    safe_copy(dst->pattern, sizeof(dst->pattern), src->pattern ? src->pattern : "");
    safe_copy(dst->replacement, sizeof(dst->replacement), src->replacement ? src->replacement : "");
    dst->flags = src->flags;
    dst->confidence = src->confidence;
    dst->is_regex = is_regex_kind(dst->kind);
    dst->is_extract = strcmp(dst->kind, "regex_extract") == 0;
    dst->is_remove = strcmp(dst->kind, "regex_remove") == 0;
    return AXIOM_STRIP_OK;
}

static int plan_step_to_recipe_step(const axiom_plan_step *step, axiom_strip_step *out) {
    if (step == NULL || out == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    memset(out, 0, sizeof(*out));
    out->kind = step->kind;
    out->pattern = step->pattern;
    out->replacement = step->replacement;
    out->flags = step->flags;
    out->confidence = step->confidence;
    return AXIOM_STRIP_OK;
}

static int apply_plan_step(const axiom_plan_step *step, const char *input, size_t in_len,
                           char **out, size_t *out_len, strip_pool *pool,
                           uint32_t *step_bit, axiom_tool_strip_stats *stats) {
    if (step == NULL || input == NULL || out == NULL || out_len == NULL || pool == NULL) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }

    char *buf = pool_alloc_stage(pool, in_len);
    if (buf == NULL) {
        return AXIOM_STRIP_ERR_POOL_EXHAUSTED;
    }
    size_t cap = (in_len * 2u) + 4096u;

    if (step->is_regex && step->compiled != NULL) {
        if (step->is_extract) {
            *out_len = axiom_regex_extract_all(step->compiled, input, in_len, buf, cap);
        } else {
            const char *replacement = step->is_remove ? "" : step->replacement;
            *out_len = axiom_regex_replace(step->compiled, input, in_len, replacement, buf, cap);
        }
        *out = buf;
        if (stats != NULL && (*out_len != in_len || memcmp(*out, input, in_len) != 0)) {
            stats->regex_steps_fired++;
        }
        return AXIOM_STRIP_OK;
    }

    axiom_strip_step raw_step;
    int rc = plan_step_to_recipe_step(step, &raw_step);
    if (rc != AXIOM_STRIP_OK) {
        return rc;
    }
    rc = si_run_step(&raw_step, input, in_len, out, out_len, pool, 1.0f);
    if (rc == AXIOM_STRIP_OK && stats != NULL &&
        (*out_len != in_len || memcmp(*out, input, in_len) != 0)) {
        stats->literal_steps_fired++;
    }
    (void)step_bit;
    return rc;
}

static char *pool_alloc_stage(strip_pool *pool, size_t in_len) {
    size_t need = (in_len * 2u) + 4096u;
    return (char *)strip_pool_alloc(pool, need);
}

static size_t strip_tool_watermark(const char *input, size_t len, char *out, size_t *removed) {
    if (removed != NULL) {
        *removed = 0;
    }
    if (input == NULL || out == NULL || len == 0) {
        return 0;
    }

    size_t start = 0;
    if (len >= 4u && memcmp(input, "<!--", 4u) == 0) {
        const char *end = si_strcasestr(input, len, "-->");
        if (end != NULL) {
            size_t comment_len = (size_t)(end - input) + 3u;
            if (comment_len <= len &&
                si_strcasestr(input, comment_len, "AXIOM SNAPSHOT ARTIFACT") != NULL) {
                start = comment_len;
                while (start < len && isspace((unsigned char)input[start])) {
                    start++;
                }
            }
        }
    }

    size_t oi = 0;
    if (start > 0) {
        if (removed != NULL) {
            *removed += start;
        }
    }
    for (size_t i = start; i < len;) {
        if (i + 24u < len &&
            si_strcasestr(input + i, len - i, "AXIOM SNAPSHOT ARTIFACT") == input + i) {
            while (i < len && input[i] != '\n' && input[i] != '\r') {
                i++;
                if (removed != NULL) (*removed)++;
            }
            continue;
        }
        out[oi++] = input[i++];
    }
    return oi;
}

static bool output_ratio_allowed(const axiom_strip_plan *plan, size_t input_len,
                                 size_t output_len, uint32_t fired) {
    if (plan == NULL || fired == 0u || output_len >= input_len) {
        return false;
    }
    if (output_len <= TOOL_MIN_SMALL_SIGNAL_BYTES) {
        return true;
    }
    return (plan->profile.flags & AXIOM_TOOL_PROFILE_FLAG_METADATA) != 0u &&
           output_len <= 256u;
}

static void stats_from_result(axiom_tool_strip_stats *stats, const uint8_t *input,
                              size_t input_len, const uint8_t *output,
                              size_t output_len, const axiom_strip_result *result) {
    if (stats == NULL) {
        return;
    }
    axiom_strip_metrics metrics;
    memset(&metrics, 0, sizeof(metrics));
    (void)axiom_strip_measure(input, input_len, output, output_len, &metrics);
    stats->input_bytes = input_len;
    stats->output_bytes = output_len;
    stats->output_ratio = metrics.output_ratio;
    stats->signal_density = metrics.signal_density;
    if (result != NULL) {
        stats->steps_fired = result->steps_fired;
        stats->crc32 = result->crc32;
        stats->code = result->code;
    }
}

static int parse_request_input(const char *json, char *out, size_t out_cap) {
    if (json_get_string(json, "input", out, out_cap)) {
        return AXIOM_STRIP_OK;
    }
    if (json_get_string(json, "html", out, out_cap)) {
        return AXIOM_STRIP_OK;
    }
    if (json_get_string(json, "body", out, out_cap)) {
        return AXIOM_STRIP_OK;
    }
    if (json_get_string(json, "raw_html", out, out_cap)) {
        return AXIOM_STRIP_OK;
    }
    return AXIOM_STRIP_ERR_INVALID_ARG;
}

static int format_response_json(const axiom_tool_snapshot_profile *profile,
                                const axiom_tool_strip_stats *stats,
                                const axiom_strip_result *result,
                                char *out, size_t out_cap) {
    if (profile == NULL || stats == NULL || result == NULL || out == NULL || out_cap == 0) {
        return AXIOM_STRIP_ERR_INVALID_ARG;
    }
    char esc_tool[AXIOM_TOOL_NAME_LEN * 2u];
    char esc_kind[AXIOM_TOOL_ARTIFACT_KIND_LEN * 2u];
    char esc_url[AXIOM_TOOL_URL_LEN * 2u];
    json_escape(profile->source_tool, esc_tool, sizeof(esc_tool));
    json_escape(profile->artifact_kind, esc_kind, sizeof(esc_kind));
    json_escape(profile->url, esc_url, sizeof(esc_url));

    int n = snprintf(out, out_cap,
        "{\"ok\":%s,\"status\":\"%s\",\"tool\":\"%s\",\"artifact_kind\":\"%s\","
        "\"url\":\"%s\",\"bytes_written\":%zu,\"crc32\":%u,"
        "\"steps_fired\":%u,\"regex_steps_compiled\":%zu,"
        "\"regex_steps_fired\":%zu,\"literal_steps_fired\":%zu,"
        "\"watermark_bytes_removed\":%zu,\"output_ratio\":%.6f,"
        "\"signal_density\":%.6f,\"code\":%d}",
        result->code == AXIOM_STRIP_OK ? "true" : "false",
        result->code == AXIOM_STRIP_OK ? "ok" : "error",
        esc_tool,
        esc_kind,
        esc_url,
        result->bytes_written,
        result->crc32,
        result->steps_fired,
        stats->regex_steps_compiled,
        stats->regex_steps_fired,
        stats->literal_steps_fired,
        stats->watermark_bytes_removed,
        stats->output_ratio,
        stats->signal_density,
        result->code
    );
    if (n < 0 || (size_t)n >= out_cap) {
        return AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL;
    }
    return AXIOM_STRIP_OK;
}

static int json_find_value(const char *json, const char *key, json_slice *slice) {
    if (json == NULL || key == NULL || slice == NULL) {
        return 0;
    }
    char needle[128];
    snprintf(needle, sizeof(needle), "\"%s\"", key);
    const char *p = strstr(json, needle);
    if (p == NULL) {
        return 0;
    }
    p += strlen(needle);
    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != ':') return 0;
    p++;
    while (*p && isspace((unsigned char)*p)) p++;
    slice->start = p;
    if (*p == '"') {
        p++;
        while (*p) {
            if (*p == '\\' && p[1] != '\0') {
                p += 2;
                continue;
            }
            if (*p == '"') {
                slice->len = (size_t)(p - slice->start + 1);
                return 1;
            }
            p++;
        }
        return 0;
    }
    const char *start = p;
    while (*p && *p != ',' && *p != '}' && *p != '\n' && *p != '\r') p++;
    slice->start = start;
    slice->len = (size_t)(p - start);
    return slice->len > 0;
}

static int json_get_string(const char *json, const char *key, char *out, size_t out_cap) {
    json_slice slice;
    if (!json_find_value(json, key, &slice) || slice.len < 2u || slice.start[0] != '"') {
        return 0;
    }
    return json_copy_string(slice.start + 1u, slice.len - 2u, out, out_cap);
}

static int json_get_number(const char *json, const char *key, double *out) {
    json_slice slice;
    if (!json_find_value(json, key, &slice) || out == NULL || slice.len == 0) {
        return 0;
    }
    char tmp[64];
    size_t len = slice.len < sizeof(tmp) - 1u ? slice.len : sizeof(tmp) - 1u;
    memcpy(tmp, slice.start, len);
    tmp[len] = '\0';
    *out = strtod(tmp, NULL);
    return 1;
}

static int json_copy_string(const char *src, size_t len, char *out, size_t out_cap) {
    if (src == NULL || out == NULL || out_cap == 0) {
        return 0;
    }
    size_t oi = 0;
    for (size_t i = 0; i < len && oi + 1u < out_cap; ++i) {
        char c = src[i];
        if (c == '\\' && i + 1u < len) {
            c = src[++i];
            switch (c) {
            case 'n': out[oi++] = '\n'; break;
            case 'r': out[oi++] = '\r'; break;
            case 't': out[oi++] = '\t'; break;
            case '"': out[oi++] = '"'; break;
            case '\\': out[oi++] = '\\'; break;
            default: out[oi++] = c; break;
            }
        } else {
            out[oi++] = c;
        }
    }
    out[oi] = '\0';
    return 1;
}

static void json_escape(const char *input, char *out, size_t out_cap) {
    if (out == NULL || out_cap == 0) {
        return;
    }
    size_t oi = 0;
    for (size_t i = 0; input != NULL && input[i] != '\0' && oi + 2u < out_cap; ++i) {
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

static void safe_copy(char *dst, size_t dst_cap, const char *src) {
    if (dst == NULL || dst_cap == 0) {
        return;
    }
    if (src == NULL) {
        dst[0] = '\0';
        return;
    }
    snprintf(dst, dst_cap, "%s", src);
}

static bool str_eq_ci(const char *a, const char *b) {
    if (a == NULL || b == NULL) {
        return false;
    }
    while (*a && *b) {
        if (tolower((unsigned char)*a) != tolower((unsigned char)*b)) {
            return false;
        }
        a++;
        b++;
    }
    return *a == '\0' && *b == '\0';
}

static bool has_token_ci(const char *haystack, const char *needle) {
    if (haystack == NULL || needle == NULL) {
        return false;
    }
    return si_strcasestr(haystack, strlen(haystack), needle) != NULL;
}

static bool is_regex_kind(const char *kind) {
    return kind != NULL &&
           (strcmp(kind, "regex_remove") == 0 ||
            strcmp(kind, "regex_replace") == 0 ||
            strcmp(kind, "regex_extract") == 0);
}

static uint32_t fnv1a_update(uint32_t hash, const char *data) {
    for (size_t i = 0; data != NULL && data[i] != '\0'; ++i) {
        hash ^= (uint8_t)data[i];
        hash *= 16777619u;
    }
    return hash;
}

#ifdef AXIOM_TOOL_STRIP_ACCELERATOR_TEST
static int test_profile_recipe(void) {
    axiom_tool_snapshot_profile profile;
    axiom_tool_profile_from_json(
        "{\"tool\":\"WebFetchTool\",\"artifact_kind\":\"raw_html\",\"url\":\"https://example.com\"}",
        &profile
    );
    axiom_strip_step steps[TOOL_MAX_RECIPE_STEPS];
    axiom_strip_recipe recipe;
    if (axiom_tool_profile_build_recipe(&profile, steps, TOOL_MAX_RECIPE_STEPS, &recipe) != AXIOM_STRIP_OK) {
        return 1;
    }
    return recipe.step_count >= 8 ? 0 : 2;
}

int main(void) {
    return test_profile_recipe();
}
#endif
