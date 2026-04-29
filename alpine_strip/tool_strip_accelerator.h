/*
 * tool_strip_accelerator.h - AXIOM tools bridge adapter for alpine_strip.
 *
 * This layer connects captured artifacts from tag/tools_bridge.py and the
 * tools/axiom-sdk sidecar to the native strip engine. It adds compiled recipe
 * plans for repeated offline stripping and a JSON request surface that can be
 * called by lightweight runtimes without binding directly to internal structs.
 */

#ifndef AXIOM_TOOL_STRIP_ACCELERATOR_H
#define AXIOM_TOOL_STRIP_ACCELERATOR_H

#include "strip_engine.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define AXIOM_TOOL_STRIP_VERSION          "0.2.0"
#define AXIOM_TOOL_NAME_LEN               96u
#define AXIOM_TOOL_URL_LEN                2048u
#define AXIOM_TOOL_QUERY_LEN              1024u
#define AXIOM_TOOL_ARTIFACT_KIND_LEN      64u
#define AXIOM_TOOL_PROFILE_FLAG_WATERMARK 0x00000001u
#define AXIOM_TOOL_PROFILE_FLAG_HTML      0x00000002u
#define AXIOM_TOOL_PROFILE_FLAG_MARKDOWN  0x00000004u
#define AXIOM_TOOL_PROFILE_FLAG_METADATA  0x00000008u
#define AXIOM_TOOL_PROFILE_FLAG_SNAPSHOT  0x00000010u
#define AXIOM_TOOL_PLAN_MAX_STEPS         AXIOM_STRIP_MAX_STEPS

typedef struct {
    char     source_tool[AXIOM_TOOL_NAME_LEN];
    char     artifact_kind[AXIOM_TOOL_ARTIFACT_KIND_LEN];
    char     url[AXIOM_TOOL_URL_LEN];
    char     query[AXIOM_TOOL_QUERY_LEN];
    char     topology_class[AXIOM_STRIP_TOPOLOGY_LEN];
    uint32_t flags;
    double   max_output_ratio;
} axiom_tool_snapshot_profile;

typedef struct {
    size_t   input_bytes;
    size_t   output_bytes;
    size_t   watermark_bytes_removed;
    size_t   regex_steps_compiled;
    size_t   regex_steps_fired;
    size_t   literal_steps_fired;
    uint32_t steps_fired;
    uint32_t crc32;
    double   output_ratio;
    double   signal_density;
    int      code;
} axiom_tool_strip_stats;

typedef struct axiom_strip_plan axiom_strip_plan;

const char *axiom_tool_strip_version(void);

void axiom_tool_profile_init(axiom_tool_snapshot_profile *profile);

int axiom_tool_profile_from_json(
    const char *json,
    axiom_tool_snapshot_profile *profile
);

int axiom_tool_profile_build_recipe(
    const axiom_tool_snapshot_profile *profile,
    axiom_strip_step *steps,
    size_t step_capacity,
    axiom_strip_recipe *recipe
);

int axiom_strip_plan_compile(
    const axiom_strip_recipe *recipe,
    const axiom_tool_snapshot_profile *profile,
    axiom_strip_plan **out_plan
);

void axiom_strip_plan_free(axiom_strip_plan *plan);

int axiom_strip_plan_apply(
    const axiom_strip_plan *plan,
    const uint8_t *input,
    size_t input_len,
    uint8_t *output,
    size_t output_capacity,
    axiom_strip_result *result,
    axiom_tool_strip_stats *stats,
    strip_pool *pool
);

int axiom_tool_strip_request_json(
    const char *request_json,
    char *signal_output,
    size_t signal_capacity,
    char *response_json,
    size_t response_capacity,
    strip_pool *pool
);

int axiom_tool_strip_make_queue_line(
    const char *url,
    int slot_idx,
    const char *input_path,
    const char *output_path,
    char *out_jsonl,
    size_t out_capacity
);

bool axiom_tool_strip_is_snapshot_profile(
    const axiom_tool_snapshot_profile *profile
);

#ifdef __cplusplus
}
#endif

#endif /* AXIOM_TOOL_STRIP_ACCELERATOR_H */
