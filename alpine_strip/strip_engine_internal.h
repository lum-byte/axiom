/*
 * strip_engine_internal.h — Private declarations shared across
 * strip_engine_part*.c translation units.  Not part of the public API.
 */

#ifndef AXIOM_STRIP_ENGINE_INTERNAL_H
#define AXIOM_STRIP_ENGINE_INTERNAL_H

#include "strip_engine.h"

#include <ctype.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ */
/*  Internal string helpers  (part1)                                   */
/* ------------------------------------------------------------------ */

bool        si_starts_with_ci(const char *s, size_t remain, const char *needle);
const char *si_strcasestr(const char *haystack, size_t hay_len, const char *needle);

/* ------------------------------------------------------------------ */
/*  Internal HTML helpers  (part2)                                     */
/* ------------------------------------------------------------------ */

size_t si_strip_tag_block(const char *input, size_t len, const char *tag, char *out);
size_t si_strip_html_tags(const char *input, size_t len, char *out);
size_t si_strip_comments(const char *input, size_t len, char *out);
size_t si_strip_attributes(const char *input, size_t len, char *out);
int    si_entity_value(const char *name, size_t len, char *out);
size_t si_decode_entities(const char *input, size_t len, char *out);
size_t si_strip_inline_styles(const char *input, size_t len, char *out);
size_t si_strip_data_attrs(const char *input, size_t len, char *out);

/* ------------------------------------------------------------------ */
/*  Internal text-manipulation helpers  (part3)                        */
/* ------------------------------------------------------------------ */

size_t si_collapse_ws(const char *input, size_t len, char *out);
size_t si_remove_literal(const char *input, size_t len, const char *needle, char *out);
size_t si_replace_literal(const char *input, size_t len,
                          const char *needle, const char *repl, char *out);
size_t si_keep_between(const char *input, size_t len,
                       const char *start, const char *end, char *out);
size_t si_strip_between(const char *input, size_t len,
                        const char *start, const char *end, char *out);
size_t si_filter_lines_containing(const char *input, size_t len,
                                  const char *needle, char *out);
size_t si_filter_lines_not_containing(const char *input, size_t len,
                                      const char *needle, char *out);
size_t si_truncate_bytes(const char *input, size_t len,
                         const char *limit_text, char *out);
int    si_parse_positive_size(const char *text, size_t fallback, size_t *out);
size_t si_normalize_unicode_ws(const char *input, size_t len, char *out);
size_t si_deduplicate_lines(const char *input, size_t len, char *out);
size_t si_trim_lines(const char *input, size_t len, char *out);
size_t si_extract_text_runs(const char *input, size_t len,
                            size_t min_run, char *out);

/* ------------------------------------------------------------------ */
/*  Step execution  (part6)                                            */
/* ------------------------------------------------------------------ */

bool si_step_kind_known(const char *kind);
int  si_run_step(const axiom_strip_step *step, const char *input,
                 size_t in_len, char **out, size_t *out_len,
                 strip_pool *pool, float current_confidence);

/* ------------------------------------------------------------------ */
/*  Helpers for step flag evaluation  (part6)                          */
/* ------------------------------------------------------------------ */

bool si_should_fire_step(const axiom_strip_step *step, float confidence);

#ifdef __cplusplus
}
#endif

#endif /* AXIOM_STRIP_ENGINE_INTERNAL_H */
