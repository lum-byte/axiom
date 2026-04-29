/*
 * test_strip_part1.c â€” Test framework and basic strip engine tests.
 *
 * Defines ASSERT macros and a test runner.  Tests cover: empty input,
 * null args, CRC32, pool allocator init/alloc/reset/destroy, error
 * code strings, and basic strip_apply with no recipe.
 */

#include "strip_engine.h"
#include "strip_engine_internal.h"
#include "tool_strip_accelerator.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ */
/*  Test framework                                                     */
/* ------------------------------------------------------------------ */

static int g_test_count = 0;
static int g_test_pass  = 0;
static int g_test_fail  = 0;

#define TEST_BEGIN(name) \
    static int name(void) { \
        const char *_test_name = #name; \
        g_test_count++; \
        do {

#define TEST_END \
        } while (0); \
        g_test_pass++; \
        return 0; \
    }

#define ASSERT_TRUE(x) do { \
    if (!(x)) { \
        printf("FAIL %s:%d: %s: %s\n", __FILE__, __LINE__, _test_name, #x); \
        g_test_fail++; \
        return 1; \
    } \
} while (0)

#define ASSERT_FALSE(x) ASSERT_TRUE(!(x))

#define ASSERT_EQ(a, b) do { \
    if ((a) != (b)) { \
        printf("FAIL %s:%d: %s: %s != %s\n", __FILE__, __LINE__, _test_name, #a, #b); \
        g_test_fail++; \
        return 1; \
    } \
} while (0)

#define ASSERT_NE(a, b) do { \
    if ((a) == (b)) { \
        printf("FAIL %s:%d: %s: %s == %s\n", __FILE__, __LINE__, _test_name, #a, #b); \
        g_test_fail++; \
        return 1; \
    } \
} while (0)

#define ASSERT_STR_CONTAINS(haystack, needle) do { \
    if (strstr((haystack), (needle)) == NULL) { \
        printf("FAIL %s:%d: %s: \"%s\" not in output\n", __FILE__, __LINE__, _test_name, (needle)); \
        g_test_fail++; \
        return 1; \
    } \
} while (0)

#define ASSERT_STR_NOT_CONTAINS(haystack, needle) do { \
    if (strstr((haystack), (needle)) != NULL) { \
        printf("FAIL %s:%d: %s: \"%s\" found in output\n", __FILE__, __LINE__, _test_name, (needle)); \
        g_test_fail++; \
        return 1; \
    } \
} while (0)

#define RUN_TEST(fn) do { fn(); } while (0)

/* ------------------------------------------------------------------ */
/*  CRC32 tests                                                        */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_crc32_known_value)
    uint32_t crc = axiom_crc32((const uint8_t *)"123456789", 9);
    ASSERT_EQ(crc, 0xcbf43926u);
TEST_END

TEST_BEGIN(test_crc32_empty)
    uint32_t crc = axiom_crc32((const uint8_t *)"", 0);
    ASSERT_EQ(crc, 0x00000000u);
TEST_END

TEST_BEGIN(test_crc32_single_byte)
    uint32_t crc = axiom_crc32((const uint8_t *)"A", 1);
    ASSERT_NE(crc, 0u);
TEST_END

TEST_BEGIN(test_crc32_different_inputs)
    uint32_t a = axiom_crc32((const uint8_t *)"hello", 5);
    uint32_t b = axiom_crc32((const uint8_t *)"world", 5);
    ASSERT_NE(a, b);
TEST_END

/* ------------------------------------------------------------------ */
/*  Pool allocator tests                                               */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_pool_init_destroy)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 1024), AXIOM_STRIP_OK);
    ASSERT_TRUE(pool.base != NULL);
    ASSERT_EQ(pool.capacity, 1024u);
    ASSERT_EQ(pool.offset, 0u);
    strip_pool_destroy(&pool);
    ASSERT_TRUE(pool.base == NULL);
TEST_END

TEST_BEGIN(test_pool_alloc_basic)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 4096), AXIOM_STRIP_OK);
    void *p1 = strip_pool_alloc(&pool, 64);
    ASSERT_TRUE(p1 != NULL);
    void *p2 = strip_pool_alloc(&pool, 128);
    ASSERT_TRUE(p2 != NULL);
    ASSERT_TRUE(p2 != p1);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_alloc_zero_size)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 1024), AXIOM_STRIP_OK);
    void *p = strip_pool_alloc(&pool, 0);
    ASSERT_TRUE(p != NULL);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_exhaustion)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 256), AXIOM_STRIP_OK);
    void *p = strip_pool_alloc(&pool, 512);
    ASSERT_TRUE(p == NULL);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_reset)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 1024), AXIOM_STRIP_OK);
    strip_pool_alloc(&pool, 512);
    ASSERT_TRUE(pool.offset > 0);
    strip_pool_reset(&pool);
    ASSERT_EQ(pool.offset, 0u);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_strdup)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 4096), AXIOM_STRIP_OK);
    char *s = strip_pool_strdup(&pool, "hello world");
    ASSERT_TRUE(s != NULL);
    ASSERT_EQ(strcmp(s, "hello world"), 0);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_strndup)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 4096), AXIOM_STRIP_OK);
    char *s = strip_pool_strndup(&pool, "hello world", 5);
    ASSERT_TRUE(s != NULL);
    ASSERT_EQ(strcmp(s, "hello"), 0);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_null_args)
    ASSERT_EQ(strip_pool_init(NULL, 1024), AXIOM_STRIP_ERR_INVALID_ARG);
    ASSERT_TRUE(strip_pool_alloc(NULL, 64) == NULL);
    strip_pool_reset(NULL);
    strip_pool_destroy(NULL);
TEST_END

TEST_BEGIN(test_pool_default_capacity)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 0), AXIOM_STRIP_OK);
    ASSERT_EQ(pool.capacity, AXIOM_POOL_BYTES);
    strip_pool_destroy(&pool);
TEST_END

/* ------------------------------------------------------------------ */
/*  Error string tests                                                 */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_strerror_ok)
    const char *s = axiom_strip_strerror(AXIOM_STRIP_OK);
    ASSERT_TRUE(s != NULL);
    ASSERT_STR_CONTAINS(s, "OK");
TEST_END

TEST_BEGIN(test_strerror_all_codes)
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_INVALID_ARG) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_BAD_RECIPE) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_POOL_EXHAUSTED) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_REDUCTION_RATIO) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_PCRE2_COMPILE) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_MMAP_OPEN) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_ERR_CRC_MISMATCH) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(AXIOM_STRIP_EMPTY) != NULL);
    ASSERT_TRUE(axiom_strip_strerror(999) != NULL);
TEST_END

/* ------------------------------------------------------------------ */
/*  Basic strip_apply tests                                            */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_apply_empty_input)
    uint8_t out[16];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)"", 0, NULL, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, 0u);
TEST_END

TEST_BEGIN(test_apply_null_recipe)
    const char *input = "hello world";
    uint8_t out[64];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), NULL, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, strlen(input));
    ASSERT_EQ(memcmp(out, input, strlen(input)), 0);
TEST_END

TEST_BEGIN(test_apply_zero_steps)
    const char *input = "passthrough";
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 0, .max_output_ratio = 1.0};
    uint8_t out[64];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, strlen(input));
TEST_END

TEST_BEGIN(test_apply_null_args)
    uint8_t out[16];
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply(NULL, 5, NULL, out, sizeof(out), &result), AXIOM_STRIP_ERR_INVALID_ARG);
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)"x", 1, NULL, NULL, 16, &result), AXIOM_STRIP_ERR_INVALID_ARG);
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)"x", 1, NULL, out, sizeof(out), NULL), AXIOM_STRIP_ERR_INVALID_ARG);
TEST_END

TEST_BEGIN(test_apply_output_too_small)
    const char *input = "this is a long string";
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 0, .max_output_ratio = 1.0};
    uint8_t out[4];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_ERR_OUTPUT_TOO_SMALL);
TEST_END

TEST_BEGIN(test_apply_crc_populated)
    const char *input = "test data";
    uint8_t out[64];
    axiom_strip_result result;
    axiom_strip_apply((const uint8_t *)input, strlen(input), NULL, out, sizeof(out), &result);
    ASSERT_NE(result.crc32, 0u);
    uint32_t expected = axiom_crc32(out, result.bytes_written);
    ASSERT_EQ(result.crc32, expected);
TEST_END

/* ------------------------------------------------------------------ */
/*  Ratio guard test                                                   */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_ratio_guard_triggers)
    const char *input = "abcdef";
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 0, .max_output_ratio = 0.5};
    uint8_t out[64];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_ERR_REDUCTION_RATIO);
TEST_END

TEST_BEGIN(test_ratio_guard_passes)
    const char *input = "abcdef";
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 0, .max_output_ratio = 1.0};
    uint8_t out[64];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
TEST_END

/* ------------------------------------------------------------------ */
/*  Recipe validation tests                                            */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_validate_null_recipe)
    ASSERT_EQ(axiom_strip_validate_recipe(NULL), AXIOM_STRIP_OK);
TEST_END

TEST_BEGIN(test_validate_too_many_steps)
    axiom_strip_recipe recipe = {.steps = (axiom_strip_step *)1, .step_count = AXIOM_STRIP_MAX_STEPS + 1};
    ASSERT_EQ(axiom_strip_validate_recipe(&recipe), AXIOM_STRIP_ERR_BAD_RECIPE);
TEST_END

TEST_BEGIN(test_validate_null_steps_nonzero_count)
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 5};
    ASSERT_EQ(axiom_strip_validate_recipe(&recipe), AXIOM_STRIP_ERR_BAD_RECIPE);
TEST_END

TEST_BEGIN(test_validate_bad_ratio)
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 0, .max_output_ratio = -1.0};
    ASSERT_EQ(axiom_strip_validate_recipe(&recipe), AXIOM_STRIP_ERR_BAD_RECIPE);
TEST_END

TEST_BEGIN(test_validate_unknown_step)
    axiom_strip_step steps[] = {{.kind = "nonexistent_step"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    ASSERT_EQ(axiom_strip_validate_recipe(&recipe), AXIOM_STRIP_ERR_BAD_RECIPE);
TEST_END

TEST_BEGIN(test_validate_keep_between_missing_args)
    axiom_strip_step steps[] = {{.kind = "keep_between", .pattern = NULL, .replacement = NULL}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    ASSERT_EQ(axiom_strip_validate_recipe(&recipe), AXIOM_STRIP_ERR_BAD_RECIPE);
TEST_END

TEST_BEGIN(test_validate_valid_recipe)
    axiom_strip_step steps[] = {
        {.kind = "strip_tag", .pattern = "script"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 2, .max_output_ratio = 1.0};
    ASSERT_EQ(axiom_strip_validate_recipe(&recipe), AXIOM_STRIP_OK);
TEST_END

/* ------------------------------------------------------------------ */
/*  String helper tests                                                */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_starts_with_ci)
    ASSERT_TRUE(si_starts_with_ci("Hello World", 11, "hello"));
    ASSERT_TRUE(si_starts_with_ci("<SCRIPT>", 8, "<script"));
    ASSERT_FALSE(si_starts_with_ci("Hi", 2, "Hello"));
    ASSERT_FALSE(si_starts_with_ci("ab", 2, "abc"));
TEST_END

TEST_BEGIN(test_strcasestr)
    const char *hay = "The Quick Brown Fox";
    const char *found = si_strcasestr(hay, strlen(hay), "quick");
    ASSERT_TRUE(found != NULL);
    ASSERT_EQ(found - hay, 4);
    ASSERT_TRUE(si_strcasestr(hay, strlen(hay), "zebra") == NULL);
TEST_END

/* ------------------------------------------------------------------ */
/*  Runner (called from test_strip_part5.c main)                       */
/* ------------------------------------------------------------------ */

int run_part1_tests(void) {
    int before = g_test_fail;
    RUN_TEST(test_crc32_known_value);
    RUN_TEST(test_crc32_empty);
    RUN_TEST(test_crc32_single_byte);
    RUN_TEST(test_crc32_different_inputs);
    RUN_TEST(test_pool_init_destroy);
    RUN_TEST(test_pool_alloc_basic);
    RUN_TEST(test_pool_alloc_zero_size);
    RUN_TEST(test_pool_exhaustion);
    RUN_TEST(test_pool_reset);
    RUN_TEST(test_pool_strdup);
    RUN_TEST(test_pool_strndup);
    RUN_TEST(test_pool_null_args);
    RUN_TEST(test_pool_default_capacity);
    RUN_TEST(test_strerror_ok);
    RUN_TEST(test_strerror_all_codes);
    RUN_TEST(test_apply_empty_input);
    RUN_TEST(test_apply_null_recipe);
    RUN_TEST(test_apply_zero_steps);
    RUN_TEST(test_apply_null_args);
    RUN_TEST(test_apply_output_too_small);
    RUN_TEST(test_apply_crc_populated);
    RUN_TEST(test_ratio_guard_triggers);
    RUN_TEST(test_ratio_guard_passes);
    RUN_TEST(test_validate_null_recipe);
    RUN_TEST(test_validate_too_many_steps);
    RUN_TEST(test_validate_null_steps_nonzero_count);
    RUN_TEST(test_validate_bad_ratio);
    RUN_TEST(test_validate_unknown_step);
    RUN_TEST(test_validate_keep_between_missing_args);
    RUN_TEST(test_validate_valid_recipe);
    RUN_TEST(test_starts_with_ci);
    RUN_TEST(test_strcasestr);
    return g_test_fail - before;
}

/*
 * test_strip_part2.c â€” HTML processing and text manipulation tests.
 */



/* ------------------------------------------------------------------ */
/*  HTML stripping tests                                               */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_strip_script_tag)
    const char *in = "<html><script>bad()</script><body>Hello</body></html>";
    axiom_strip_step steps[] = {{.kind = "strip_tag", .pattern = "script"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "bad");
    ASSERT_STR_CONTAINS((char *)out, "Hello");
TEST_END

TEST_BEGIN(test_strip_style_tag)
    const char *in = "<style>.x{color:red}</style><p>Text</p>";
    axiom_strip_step steps[] = {{.kind = "strip_tag", .pattern = "style"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "color");
    ASSERT_STR_CONTAINS((char *)out, "Text");
TEST_END

TEST_BEGIN(test_strip_html_tags)
    const char *in = "<div><h1>Title</h1><p>Body</p></div>";
    axiom_strip_step steps[] = {{.kind = "strip_html"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "Title");
    ASSERT_STR_CONTAINS((char *)out, "Body");
    ASSERT_STR_NOT_CONTAINS((char *)out, "<div>");
TEST_END

TEST_BEGIN(test_strip_comments)
    const char *in = "before<!-- hidden -->after";
    axiom_strip_step steps[] = {{.kind = "strip_comments"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "hidden");
    ASSERT_STR_CONTAINS((char *)out, "before");
    ASSERT_STR_CONTAINS((char *)out, "after");
TEST_END

TEST_BEGIN(test_strip_attributes)
    const char *in = "<div class=\"x\" id=\"y\">text</div>";
    axiom_strip_step steps[] = {{.kind = "strip_attrs"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "class");
    ASSERT_STR_NOT_CONTAINS((char *)out, "id");
    ASSERT_STR_CONTAINS((char *)out, "text");
TEST_END

TEST_BEGIN(test_decode_entities)
    const char *in = "A&amp;B&lt;C&gt;D&quot;E&nbsp;F";
    axiom_strip_step steps[] = {{.kind = "decode_entities"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "A&B");
    ASSERT_STR_CONTAINS((char *)out, "<C>");
TEST_END

TEST_BEGIN(test_decode_numeric_entities)
    const char *in = "&#65;&#x42;";
    axiom_strip_step steps[] = {{.kind = "decode_entities"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "AB");
TEST_END

TEST_BEGIN(test_default_html_pipeline)
    const char *in = "<html><!-- c --><body><p class=\"x\">A&amp;B</p><style>.x{}</style><script>x()</script></body></html>";
    uint8_t out[256] = {0};
    axiom_strip_result result;
    int code = axiom_strip_apply_default_html((const uint8_t *)in, strlen(in), out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "A&B");
    ASSERT_STR_NOT_CONTAINS((char *)out, "class");
    ASSERT_STR_NOT_CONTAINS((char *)out, "x()");
    ASSERT_STR_NOT_CONTAINS((char *)out, "<!-- c -->");
TEST_END

/* ------------------------------------------------------------------ */
/*  Text manipulation tests                                            */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_collapse_ws)
    const char *in = "  hello   world  \n\t  foo  ";
    axiom_strip_step steps[] = {{.kind = "collapse_ws"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "hello world");
TEST_END

TEST_BEGIN(test_remove_literal)
    const char *in = "hello JUNK world JUNK end";
    axiom_strip_step steps[] = {{.kind = "remove", .pattern = "JUNK "}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "JUNK");
TEST_END

TEST_BEGIN(test_replace_literal)
    const char *in = "foo bar foo baz";
    axiom_strip_step steps[] = {{.kind = "replace", .pattern = "foo", .replacement = "qux"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.5};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "foo");
    ASSERT_STR_CONTAINS((char *)out, "qux");
TEST_END

TEST_BEGIN(test_keep_between)
    const char *in = "before <main>content</main> after";
    axiom_strip_step steps[] = {{.kind = "keep_between", .pattern = "<main>", .replacement = "</main>"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "content");
    ASSERT_STR_NOT_CONTAINS((char *)out, "before");
TEST_END

TEST_BEGIN(test_strip_between)
    const char *in = "keep <nav>remove</nav> keep";
    axiom_strip_step steps[] = {{.kind = "strip_between", .pattern = "<nav>", .replacement = "</nav>"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "remove");
TEST_END

TEST_BEGIN(test_filter_lines)
    const char *in = "keep this\nskip that\nkeep also\n";
    axiom_strip_step steps[] = {{.kind = "filter_lines", .pattern = "keep"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "keep this");
    ASSERT_STR_CONTAINS((char *)out, "keep also");
    ASSERT_STR_NOT_CONTAINS((char *)out, "skip");
TEST_END

TEST_BEGIN(test_filter_lines_not)
    const char *in = "line1\nbad line\nline3\n";
    axiom_strip_step steps[] = {{.kind = "filter_lines_not", .pattern = "bad"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "bad");
    ASSERT_STR_CONTAINS((char *)out, "line1");
TEST_END

TEST_BEGIN(test_truncate)
    const char *in = "abcdefghij";
    axiom_strip_step steps[] = {{.kind = "truncate", .pattern = "5"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, 5u);
TEST_END

TEST_BEGIN(test_multi_step_pipeline)
    const char *in = "<html><script>x()</script><body><h1>Hello</h1> world</body></html>";
    axiom_strip_step steps[] = {
        {.kind = "strip_tag", .pattern = "script"},
        {.kind = "strip_html"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 3, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "Hello");
    ASSERT_STR_NOT_CONTAINS((char *)out, "x()");
TEST_END

TEST_BEGIN(test_keep_between_and_filter)
    const char *in = "before <main>\nkeep this\nskip that\nkeep also\n</main> after";
    axiom_strip_step steps[] = {
        {.kind = "keep_between", .pattern = "<main>", .replacement = "</main>"},
        {.kind = "filter_lines", .pattern = "keep"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 3, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "keep this");
    ASSERT_STR_NOT_CONTAINS((char *)out, "skip");
TEST_END

/* ------------------------------------------------------------------ */
/*  Measurement tests                                                  */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_measure_basic)
    const char *in = "<p>Hello 123</p>";
    uint8_t out[128] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply_default_html((const uint8_t *)in, strlen(in), out, sizeof(out), &result), AXIOM_STRIP_OK);
    axiom_strip_metrics m;
    ASSERT_EQ(axiom_strip_measure((const uint8_t *)in, strlen(in), out, result.bytes_written, &m), AXIOM_STRIP_OK);
    ASSERT_TRUE(m.token_count >= 2u);
    ASSERT_TRUE(m.signal_density > 0.0);
    ASSERT_TRUE(m.output_ratio > 0.0);
    ASSERT_TRUE(m.output_ratio <= 1.0);
TEST_END

TEST_BEGIN(test_measure_null_args)
    ASSERT_EQ(axiom_strip_measure(NULL, 0, NULL, 0, NULL), AXIOM_STRIP_ERR_INVALID_ARG);
TEST_END

/* ------------------------------------------------------------------ */
/*  Runner                                                             */
/* ------------------------------------------------------------------ */

int run_part2_tests(void) {
    int before = g_test_fail;
    RUN_TEST(test_strip_script_tag);
    RUN_TEST(test_strip_style_tag);
    RUN_TEST(test_strip_html_tags);
    RUN_TEST(test_strip_comments);
    RUN_TEST(test_strip_attributes);
    RUN_TEST(test_decode_entities);
    RUN_TEST(test_decode_numeric_entities);
    RUN_TEST(test_default_html_pipeline);
    RUN_TEST(test_collapse_ws);
    RUN_TEST(test_remove_literal);
    RUN_TEST(test_replace_literal);
    RUN_TEST(test_keep_between);
    RUN_TEST(test_strip_between);
    RUN_TEST(test_filter_lines);
    RUN_TEST(test_filter_lines_not);
    RUN_TEST(test_truncate);
    RUN_TEST(test_multi_step_pipeline);
    RUN_TEST(test_keep_between_and_filter);
    RUN_TEST(test_measure_basic);
    RUN_TEST(test_measure_null_args);
    return g_test_fail - before;
}

/*
 * test_strip_part3.c â€” PCRE2 regex, recipe mmap, and confidence tests.
 */



/* ------------------------------------------------------------------ */
/*  Regex compile/free tests                                           */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_regex_compile_null)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile(NULL, 0, 0, &ec);
    ASSERT_TRUE(re == NULL);
    ASSERT_NE(ec, AXIOM_STRIP_OK);
TEST_END

TEST_BEGIN(test_regex_compile_empty)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("", 0, 0, &ec);
    ASSERT_TRUE(re == NULL);
TEST_END

TEST_BEGIN(test_regex_compile_valid)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("hello", 5, 0, &ec);
    ASSERT_TRUE(re != NULL);
    ASSERT_EQ(ec, AXIOM_STRIP_OK);
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_match_found)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("world", 5, 0, &ec);
    ASSERT_TRUE(re != NULL);
    const char *subj = "hello world";
    size_t mc = 0;
    ASSERT_EQ(axiom_regex_match(re, subj, strlen(subj), 0, NULL, 0, &mc), AXIOM_STRIP_OK);
    ASSERT_TRUE(mc > 0);
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_match_not_found)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("zebra", 5, 0, &ec);
    ASSERT_TRUE(re != NULL);
    size_t mc = 0;
    ASSERT_EQ(axiom_regex_match(re, "hello", 5, 0, NULL, 0, &mc), AXIOM_STRIP_OK);
    ASSERT_EQ(mc, 0u);
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_match_ovector)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("world", 5, 0, &ec);
    ASSERT_TRUE(re != NULL);
    const char *subj = "hello world";
    size_t ov[4] = {0};
    size_t mc = 0;
    ASSERT_EQ(axiom_regex_match(re, subj, strlen(subj), 0, ov, 4, &mc), AXIOM_STRIP_OK);
    ASSERT_TRUE(mc > 0);
    ASSERT_EQ(ov[0], 6u);
    ASSERT_EQ(ov[1], 11u);
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_replace_all)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("cat", 3, 0, &ec);
    ASSERT_TRUE(re != NULL);
    const char *subj = "the cat sat on the cat mat";
    char out[256] = {0};
    size_t len = axiom_regex_replace(re, subj, strlen(subj), "dog", out, sizeof(out));
    ASSERT_TRUE(len > 0);
    ASSERT_STR_CONTAINS(out, "dog");
    ASSERT_STR_NOT_CONTAINS(out, "cat");
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_extract_all)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("cat", 3, 0, &ec);
    ASSERT_TRUE(re != NULL);
    const char *subj = "cat dog cat bird cat";
    char out[256] = {0};
    size_t len = axiom_regex_extract_all(re, subj, strlen(subj), out, sizeof(out));
    ASSERT_TRUE(len > 0);
    ASSERT_STR_CONTAINS(out, "cat");
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_jit_hint)
    int ec = 0;
    axiom_regex *re = axiom_regex_compile("hello", 5, 0, &ec);
    ASSERT_TRUE(re != NULL);
    int rc = axiom_regex_jit_hint(re);
    ASSERT_EQ(rc, AXIOM_STRIP_OK);
    axiom_regex_free(re);
TEST_END

TEST_BEGIN(test_regex_free_null)
    axiom_regex_free(NULL);
TEST_END

/* ------------------------------------------------------------------ */
/*  Regex step integration tests                                       */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_regex_remove_step)
    const char *in = "hello 123 world 456";
    axiom_strip_step steps[] = {{.kind = "regex_remove", .pattern = "[0-9]+"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_NOT_CONTAINS((char *)out, "123");
    ASSERT_STR_NOT_CONTAINS((char *)out, "456");
    ASSERT_STR_CONTAINS((char *)out, "hello");
TEST_END

TEST_BEGIN(test_regex_replace_step)
    const char *in = "foo123bar456baz";
    axiom_strip_step steps[] = {{.kind = "regex_replace", .pattern = "[0-9]+", .replacement = "NUM"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.5};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "fooNUM");
    ASSERT_STR_NOT_CONTAINS((char *)out, "123");
TEST_END

TEST_BEGIN(test_regex_extract_step)
    const char *in = "price: $100 and $200";
    axiom_strip_step steps[] = {{.kind = "regex_extract", .pattern = "\\$[0-9]+"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "$100");
    ASSERT_STR_CONTAINS((char *)out, "$200");
TEST_END

/* ------------------------------------------------------------------ */
/*  Confidence gating tests                                            */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_confidence_gate_blocks)
    const char *in = "should not change";
    axiom_strip_step steps[] = {{.kind = "collapse_ws", .confidence = 0.99f}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, strlen(in));
TEST_END

TEST_BEGIN(test_confidence_zero_always_fires)
    const char *in = "  spacy  text  ";
    axiom_strip_step steps[] = {{.kind = "collapse_ws", .confidence = 0.0f}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_TRUE(result.bytes_written < strlen(in));
TEST_END

/* ------------------------------------------------------------------ */
/*  Step flag tests                                                    */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_require_flag_on_empty_output)
    const char *in = "no match here";
    axiom_strip_step steps[] = {{.kind = "keep_between", .pattern = "<x>", .replacement = "</x>", .flags = STRIP_REQUIRE}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    int rc = axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(rc, AXIOM_STRIP_ERR_BAD_RECIPE);
TEST_END

/* ------------------------------------------------------------------ */
/*  Recipe write/load round-trip test (in-memory)                      */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_recipe_write_slot)
    axiom_strip_step steps[] = {
        {.kind = "strip_tag", .pattern = "script", .replacement = "", .flags = STRIP_REMOVE, .confidence = 0.5f},
        {.kind = "collapse_ws", .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {
        .steps = steps,
        .step_count = 2,
        .max_output_ratio = 0.3,
    };
    memcpy(recipe.topology_class, "test_class", 10);

    uint8_t slot[AXIOM_RECIPE_SLOT_BYTES];
    int rc = axiom_recipe_write_slot(&recipe, slot, sizeof(slot));
    ASSERT_EQ(rc, AXIOM_STRIP_OK);

    axiom_recipe_slot_header *sh = (axiom_recipe_slot_header *)slot;
    ASSERT_EQ(sh->step_count, 2u);
    ASSERT_TRUE(sh->checksum != 0);
    ASSERT_TRUE(sh->data_len > 0);
TEST_END

TEST_BEGIN(test_recipe_write_registry)
    axiom_strip_step steps1[] = {{.kind = "strip_html"}};
    axiom_strip_step steps2[] = {{.kind = "collapse_ws"}};
    axiom_strip_recipe recipes[] = {
        {.steps = steps1, .step_count = 1, .max_output_ratio = 0.3},
        {.steps = steps2, .step_count = 1, .max_output_ratio = 0.3},
    };
    memcpy(recipes[0].topology_class, "class_a", 7);
    memcpy(recipes[1].topology_class, "class_b", 7);

    size_t buf_size = sizeof(axiom_recipe_registry_header) + 2 * AXIOM_RECIPE_SLOT_BYTES;
    uint8_t *buf = (uint8_t *)malloc(buf_size);
    ASSERT_TRUE(buf != NULL);

    ASSERT_EQ(axiom_recipe_write_registry(recipes, 2, buf, buf_size), AXIOM_STRIP_OK);

    axiom_recipe_registry_header *hdr = (axiom_recipe_registry_header *)buf;
    ASSERT_EQ(hdr->magic, AXIOM_RECIPE_MAGIC);
    ASSERT_EQ(hdr->slot_count, 2u);

    free(buf);
TEST_END

/* ------------------------------------------------------------------ */
/*  New step kinds tests                                               */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_deduplicate_lines)
    const char *in = "line1\nline1\nline2\nline2\nline3\n";
    axiom_strip_step steps[] = {{.kind = "deduplicate_lines"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_TRUE(result.bytes_written < strlen(in));
TEST_END

TEST_BEGIN(test_trim_lines)
    const char *in = "  hello  \n  world  \n";
    axiom_strip_step steps[] = {{.kind = "trim_lines"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "hello");
    ASSERT_STR_CONTAINS((char *)out, "world");
TEST_END

TEST_BEGIN(test_remove_blank_lines)
    const char *in = "line1\n\n\nline2\n\n";
    axiom_strip_step steps[] = {{.kind = "remove_blank_lines"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_TRUE(result.bytes_written < strlen(in));
    ASSERT_STR_CONTAINS((char *)out, "line1");
    ASSERT_STR_CONTAINS((char *)out, "line2");
TEST_END

/* ------------------------------------------------------------------ */
/*  Tool strip accelerator tests                                       */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_tool_profile_from_json)
    axiom_tool_snapshot_profile profile;
    ASSERT_EQ(
        axiom_tool_profile_from_json(
            "{\"tool\":\"WebFetchTool\",\"artifact_kind\":\"raw_html\",\"url\":\"https://example.com/docs\",\"query\":\"docs\"}",
            &profile
        ),
        AXIOM_STRIP_OK
    );
    ASSERT_EQ(strcmp(profile.source_tool, "WebFetchTool"), 0);
    ASSERT_EQ(strcmp(profile.artifact_kind, "raw_html"), 0);
    ASSERT_TRUE((profile.flags & AXIOM_TOOL_PROFILE_FLAG_HTML) != 0u);
    ASSERT_TRUE(axiom_tool_strip_is_snapshot_profile(&profile));
TEST_END

TEST_BEGIN(test_tool_profile_builds_html_recipe)
    axiom_tool_snapshot_profile profile;
    axiom_tool_profile_init(&profile);
    axiom_strip_step steps[AXIOM_TOOL_PLAN_MAX_STEPS];
    axiom_strip_recipe recipe;
    ASSERT_EQ(axiom_tool_profile_build_recipe(&profile, steps, AXIOM_TOOL_PLAN_MAX_STEPS, &recipe), AXIOM_STRIP_OK);
    ASSERT_TRUE(recipe.step_count >= 10u);
    ASSERT_EQ(strcmp(recipe.topology_class, "GENERIC_HTML"), 0);
    ASSERT_TRUE(recipe.checksum != 0u);
TEST_END

TEST_BEGIN(test_strip_plan_compiles_and_reuses_regex)
    axiom_strip_step steps[] = {
        {.kind = "regex_remove", .pattern = "[0-9]+"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 2, .max_output_ratio = 1.0};
    axiom_tool_snapshot_profile profile;
    axiom_tool_profile_init(&profile);
    axiom_strip_plan *plan = NULL;
    ASSERT_EQ(axiom_strip_plan_compile(&recipe, &profile, &plan), AXIOM_STRIP_OK);
    ASSERT_TRUE(plan != NULL);

    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    uint8_t out[256] = {0};
    axiom_strip_result result;
    axiom_tool_strip_stats stats;
    const char *in = "alpha 123 beta 456";
    ASSERT_EQ(axiom_strip_plan_apply(plan, (const uint8_t *)in, strlen(in), out, sizeof(out), &result, &stats, &pool), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "alpha");
    ASSERT_STR_CONTAINS((char *)out, "beta");
    ASSERT_STR_NOT_CONTAINS((char *)out, "123");
    ASSERT_STR_NOT_CONTAINS((char *)out, "456");
    ASSERT_EQ(stats.regex_steps_compiled, 1u);
    ASSERT_TRUE(stats.regex_steps_fired >= 1u);
    strip_pool_destroy(&pool);
    axiom_strip_plan_free(plan);
TEST_END

TEST_BEGIN(test_tool_html_snapshot_strips_watermark_and_noise)
    axiom_tool_snapshot_profile profile;
    ASSERT_EQ(
        axiom_tool_profile_from_json(
            "{\"tool\":\"WebFetchTool\",\"artifact_kind\":\"raw_html\",\"url\":\"https://example.com/docs\"}",
            &profile
        ),
        AXIOM_STRIP_OK
    );
    axiom_strip_step steps[AXIOM_TOOL_PLAN_MAX_STEPS];
    axiom_strip_recipe recipe;
    ASSERT_EQ(axiom_tool_profile_build_recipe(&profile, steps, AXIOM_TOOL_PLAN_MAX_STEPS, &recipe), AXIOM_STRIP_OK);
    axiom_strip_plan *plan = NULL;
    ASSERT_EQ(axiom_strip_plan_compile(&recipe, &profile, &plan), AXIOM_STRIP_OK);

    const char *in =
        "<!-- {\"watermark\":\"AXIOM SNAPSHOT ARTIFACT // TAG ROUTED // DO NOT TRAIN AS CLEAN SIGNAL\"} -->\n"
        "<html><body><nav>menu</nav><script>x()</script><main><h1>Title</h1><p>Signal text 123</p></main></body></html>";
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    uint8_t out[512] = {0};
    axiom_strip_result result;
    axiom_tool_strip_stats stats;
    ASSERT_EQ(axiom_strip_plan_apply(plan, (const uint8_t *)in, strlen(in), out, sizeof(out), &result, &stats, &pool), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "Title");
    ASSERT_STR_CONTAINS((char *)out, "Signal text");
    ASSERT_STR_NOT_CONTAINS((char *)out, "AXIOM SNAPSHOT");
    ASSERT_STR_NOT_CONTAINS((char *)out, "menu");
    ASSERT_STR_NOT_CONTAINS((char *)out, "x()");
    ASSERT_TRUE(stats.watermark_bytes_removed > 0u);
    strip_pool_destroy(&pool);
    axiom_strip_plan_free(plan);
TEST_END

TEST_BEGIN(test_tool_markdown_snapshot_recipe)
    axiom_tool_snapshot_profile profile;
    ASSERT_EQ(axiom_tool_profile_from_json("{\"tool\":\"AlpineStripTool\",\"artifact_kind\":\"markdown\"}", &profile), AXIOM_STRIP_OK);
    axiom_strip_step steps[AXIOM_TOOL_PLAN_MAX_STEPS];
    axiom_strip_recipe recipe;
    ASSERT_EQ(axiom_tool_profile_build_recipe(&profile, steps, AXIOM_TOOL_PLAN_MAX_STEPS, &recipe), AXIOM_STRIP_OK);
    axiom_strip_plan *plan = NULL;
    ASSERT_EQ(axiom_strip_plan_compile(&recipe, &profile, &plan), AXIOM_STRIP_OK);
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    const char *in = "# Title\n![hero](x.png)\n[docs](https://example.com) **Signal**";
    uint8_t out[256] = {0};
    axiom_strip_result result;
    axiom_tool_strip_stats stats;
    ASSERT_EQ(axiom_strip_plan_apply(plan, (const uint8_t *)in, strlen(in), out, sizeof(out), &result, &stats, &pool), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "Title");
    ASSERT_STR_CONTAINS((char *)out, "Signal");
    ASSERT_STR_NOT_CONTAINS((char *)out, "hero");
    ASSERT_STR_NOT_CONTAINS((char *)out, "https://example.com");
    strip_pool_destroy(&pool);
    axiom_strip_plan_free(plan);
TEST_END

TEST_BEGIN(test_tool_metadata_extracts_urls)
    axiom_tool_snapshot_profile profile;
    ASSERT_EQ(axiom_tool_profile_from_json("{\"tool\":\"WebSearchTool\",\"artifact_kind\":\"metadata\"}", &profile), AXIOM_STRIP_OK);
    axiom_strip_step steps[AXIOM_TOOL_PLAN_MAX_STEPS];
    axiom_strip_recipe recipe;
    ASSERT_EQ(axiom_tool_profile_build_recipe(&profile, steps, AXIOM_TOOL_PLAN_MAX_STEPS, &recipe), AXIOM_STRIP_OK);
    axiom_strip_plan *plan = NULL;
    ASSERT_EQ(axiom_strip_plan_compile(&recipe, &profile, &plan), AXIOM_STRIP_OK);
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    const char *in = "{\"hits\":[{\"url\":\"https://example.com/a\"},{\"url\":\"https://example.com/b\"}]}";
    uint8_t out[256] = {0};
    axiom_strip_result result;
    axiom_tool_strip_stats stats;
    ASSERT_EQ(axiom_strip_plan_apply(plan, (const uint8_t *)in, strlen(in), out, sizeof(out), &result, &stats, &pool), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "https://example.com/a");
    ASSERT_STR_CONTAINS((char *)out, "https://example.com/b");
    strip_pool_destroy(&pool);
    axiom_strip_plan_free(plan);
TEST_END

TEST_BEGIN(test_tool_request_json_strips_inline_payload)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    char signal[512] = {0};
    char response[1024] = {0};
    const char *req =
        "{\"tool\":\"WebFetchTool\",\"artifact_kind\":\"raw_html\",\"url\":\"https://example.com\","
        "\"input\":\"<!-- AXIOM SNAPSHOT ARTIFACT -->\\n<p>Hello 123</p><script>x()</script>\"}";
    ASSERT_EQ(axiom_tool_strip_request_json(req, signal, sizeof(signal), response, sizeof(response), &pool), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS(signal, "Hello 123");
    ASSERT_STR_NOT_CONTAINS(signal, "AXIOM SNAPSHOT");
    ASSERT_STR_NOT_CONTAINS(signal, "x()");
    ASSERT_STR_CONTAINS(response, "\"ok\":true");
    ASSERT_STR_CONTAINS(response, "\"regex_steps_compiled\"");
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_tool_queue_line_builder)
    char line[1024] = {0};
    ASSERT_EQ(
        axiom_tool_strip_make_queue_line(
            "https://example.com/docs",
            7,
            "C:\\tmp\\raw.html",
            "C:\\tmp\\signal.txt",
            line,
            sizeof(line)
        ),
        AXIOM_STRIP_OK
    );
    ASSERT_STR_CONTAINS(line, "\"slot_idx\":7");
    ASSERT_STR_CONTAINS(line, "raw.html");
    ASSERT_STR_CONTAINS(line, "signal.txt");
TEST_END

/* ------------------------------------------------------------------ */
/*  Runner                                                             */
/* ------------------------------------------------------------------ */

int run_part3_tests(void) {
    int before = g_test_fail;
    RUN_TEST(test_regex_compile_null);
    RUN_TEST(test_regex_compile_empty);
    RUN_TEST(test_regex_compile_valid);
    RUN_TEST(test_regex_match_found);
    RUN_TEST(test_regex_match_not_found);
    RUN_TEST(test_regex_match_ovector);
    RUN_TEST(test_regex_replace_all);
    RUN_TEST(test_regex_extract_all);
    RUN_TEST(test_regex_jit_hint);
    RUN_TEST(test_regex_free_null);
    RUN_TEST(test_regex_remove_step);
    RUN_TEST(test_regex_replace_step);
    RUN_TEST(test_regex_extract_step);
    RUN_TEST(test_confidence_gate_blocks);
    RUN_TEST(test_confidence_zero_always_fires);
    RUN_TEST(test_require_flag_on_empty_output);
    RUN_TEST(test_recipe_write_slot);
    RUN_TEST(test_recipe_write_registry);
    RUN_TEST(test_deduplicate_lines);
    RUN_TEST(test_trim_lines);
    RUN_TEST(test_remove_blank_lines);
    RUN_TEST(test_tool_profile_from_json);
    RUN_TEST(test_tool_profile_builds_html_recipe);
    RUN_TEST(test_strip_plan_compiles_and_reuses_regex);
    RUN_TEST(test_tool_html_snapshot_strips_watermark_and_noise);
    RUN_TEST(test_tool_markdown_snapshot_recipe);
    RUN_TEST(test_tool_metadata_extracts_urls);
    RUN_TEST(test_tool_request_json_strips_inline_payload);
    RUN_TEST(test_tool_queue_line_builder);
    return g_test_fail - before;
}

/*
 * test_strip_part4.c â€” Threading, pool edge cases, with_pool API,
 * aggressive pipeline, and misc edge case tests.
 */



#ifndef _WIN32
#  include <pthread.h>
#else
#  include <windows.h>
#  include <process.h>
#endif

/* ------------------------------------------------------------------ */
/*  with_pool API tests                                                */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_apply_with_pool_basic)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    const char *in = "<p>Hello</p>";
    uint8_t out[256] = {0};
    axiom_strip_result result;
    axiom_strip_step steps[] = {
        {.kind = "strip_html"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 2, .max_output_ratio = 1.0};
    ASSERT_EQ(axiom_strip_apply_with_pool((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result, &pool), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "Hello");
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_apply_with_pool_null_pool)
    const char *in = "test";
    uint8_t out[64];
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply_with_pool((const uint8_t *)in, strlen(in), NULL, out, sizeof(out), &result, NULL), AXIOM_STRIP_ERR_INVALID_ARG);
TEST_END

TEST_BEGIN(test_apply_with_pool_resets)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, AXIOM_POOL_BYTES), AXIOM_STRIP_OK);
    const char *in = "hello world";
    uint8_t out[256];
    axiom_strip_result result;
    for (int i = 0; i < 100; ++i) {
        ASSERT_EQ(axiom_strip_apply_with_pool((const uint8_t *)in, strlen(in), NULL, out, sizeof(out), &result, &pool), AXIOM_STRIP_OK);
        ASSERT_EQ(result.bytes_written, strlen(in));
    }
    strip_pool_destroy(&pool);
TEST_END

/* ------------------------------------------------------------------ */
/*  Pool edge cases                                                    */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_pool_alloc_alignment)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 4096), AXIOM_STRIP_OK);
    void *p1 = strip_pool_alloc(&pool, 1);
    ASSERT_TRUE(p1 != NULL);
    ASSERT_EQ((uintptr_t)p1 % 16, 0u);
    void *p2 = strip_pool_alloc(&pool, 3);
    ASSERT_TRUE(p2 != NULL);
    ASSERT_EQ((uintptr_t)p2 % 16, 0u);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_many_small_allocs)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 1024 * 1024), AXIOM_STRIP_OK);
    int count = 0;
    while (strip_pool_alloc(&pool, 16) != NULL) {
        count++;
        if (count > 100000) break;
    }
    ASSERT_TRUE(count > 1000);
    strip_pool_destroy(&pool);
TEST_END

TEST_BEGIN(test_pool_alloc_zero_returns_valid)
    strip_pool pool;
    ASSERT_EQ(strip_pool_init(&pool, 1024), AXIOM_STRIP_OK);
    void *p = strip_pool_alloc_zero(&pool, 64);
    ASSERT_TRUE(p != NULL);
    uint8_t *bytes = (uint8_t *)p;
    for (int i = 0; i < 64; ++i) {
        ASSERT_EQ(bytes[i], 0u);
    }
    strip_pool_destroy(&pool);
TEST_END

/* ------------------------------------------------------------------ */
/*  Thread safety test                                                 */
/* ------------------------------------------------------------------ */

#define THREAD_COUNT 8
#define ITERS_PER_THREAD 200

typedef struct {
    int thread_id;
    int failures;
} thread_test_ctx;

#ifndef _WIN32
static void *thread_strip_worker(void *arg) {
#else
static unsigned __stdcall thread_strip_worker(void *arg) {
#endif
    thread_test_ctx *ctx = (thread_test_ctx *)arg;
    ctx->failures = 0;

    strip_pool pool;
    if (strip_pool_init(&pool, AXIOM_POOL_BYTES) != AXIOM_STRIP_OK) {
        ctx->failures = ITERS_PER_THREAD;
#ifndef _WIN32
        return NULL;
#else
        return 1;
#endif
    }

    axiom_strip_step steps[] = {
        {.kind = "strip_comments"},
        {.kind = "strip_tag", .pattern = "script"},
        {.kind = "strip_html"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 4, .max_output_ratio = 1.0};

    for (int i = 0; i < ITERS_PER_THREAD; ++i) {
        const char *input = "<html><!-- comment --><script>x()</script><body><h1>Hello</h1> World</body></html>";
        uint8_t out[512] = {0};
        axiom_strip_result result;
        strip_pool_reset(&pool);
        int rc = axiom_strip_apply_with_pool(
            (const uint8_t *)input, strlen(input),
            &recipe, out, sizeof(out), &result, &pool
        );
        if (rc != AXIOM_STRIP_OK) {
            ctx->failures++;
            continue;
        }
        if (strstr((char *)out, "Hello") == NULL) {
            ctx->failures++;
        }
        if (strstr((char *)out, "x()") != NULL) {
            ctx->failures++;
        }
    }

    strip_pool_destroy(&pool);
#ifndef _WIN32
    return NULL;
#else
    return 0;
#endif
}

TEST_BEGIN(test_thread_safety)
    thread_test_ctx contexts[THREAD_COUNT];
#ifndef _WIN32
    pthread_t threads[THREAD_COUNT];
#else
    HANDLE threads[THREAD_COUNT];
#endif

    for (int i = 0; i < THREAD_COUNT; ++i) {
        contexts[i].thread_id = i;
        contexts[i].failures = 0;
#ifndef _WIN32
        pthread_create(&threads[i], NULL, thread_strip_worker, &contexts[i]);
#else
        threads[i] = (HANDLE)_beginthreadex(NULL, 0, thread_strip_worker, &contexts[i], 0, NULL);
#endif
    }
    for (int i = 0; i < THREAD_COUNT; ++i) {
#ifndef _WIN32
        pthread_join(threads[i], NULL);
#else
        WaitForSingleObject(threads[i], INFINITE);
        CloseHandle(threads[i]);
#endif
    }
    int total_failures = 0;
    for (int i = 0; i < THREAD_COUNT; ++i) {
        total_failures += contexts[i].failures;
    }
    ASSERT_EQ(total_failures, 0);
TEST_END

/* ------------------------------------------------------------------ */
/*  Large input test                                                   */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_large_input)
    size_t sz = 256 * 1024;
    char *big = (char *)malloc(sz + 1);
    ASSERT_TRUE(big != NULL);
    for (size_t i = 0; i < sz; ++i) {
        big[i] = 'A' + (char)(i % 26);
    }
    big[sz] = '\0';
    uint8_t *out = (uint8_t *)malloc(sz + 4096);
    ASSERT_TRUE(out != NULL);
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)big, sz, NULL, out, sz + 4096, &result), AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, sz);
    free(big);
    free(out);
TEST_END

/* ------------------------------------------------------------------ */
/*  Aggressive pipeline test                                           */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_aggressive_pipeline)
    const char *in =
        "<html><head><style>.x{color:red}</style></head>"
        "<body><div class='main' data-id='123' style='color:blue'>"
        "<!-- nav --><nav>menu</nav>"
        "<h1>Title</h1><p>Body text here</p>"
        "<script>track()</script><noscript>no js</noscript>"
        "<svg><path d='M0 0'/></svg><iframe src='x'></iframe>"
        "</div></body></html>";
    uint8_t out[1024] = {0};
    axiom_strip_result result;
    axiom_strip_step steps[] = {
        {.kind = "strip_comments"},
        {.kind = "strip_tag", .pattern = "script"},
        {.kind = "strip_tag", .pattern = "style"},
        {.kind = "strip_tag", .pattern = "noscript"},
        {.kind = "strip_tag", .pattern = "svg"},
        {.kind = "strip_tag", .pattern = "iframe"},
        {.kind = "strip_attrs"},
        {.kind = "decode_entities"},
        {.kind = "strip_html"},
        {.kind = "collapse_ws"},
        {.kind = "trim_lines"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 11, .max_output_ratio = 1.0};
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "Title");
    ASSERT_STR_CONTAINS((char *)out, "Body text here");
    ASSERT_STR_NOT_CONTAINS((char *)out, "track()");
    ASSERT_STR_NOT_CONTAINS((char *)out, "color:red");
    ASSERT_STR_NOT_CONTAINS((char *)out, "no js");
    ASSERT_STR_NOT_CONTAINS((char *)out, "<svg>");
TEST_END

/* ------------------------------------------------------------------ */
/*  Edge cases                                                         */
/* ------------------------------------------------------------------ */

TEST_BEGIN(test_unclosed_comment)
    const char *in = "before <!-- unclosed";
    axiom_strip_step steps[] = {{.kind = "strip_comments"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "before");
TEST_END

TEST_BEGIN(test_unclosed_tag)
    const char *in = "<script>this never closes";
    axiom_strip_step steps[] = {{.kind = "strip_tag", .pattern = "script"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
TEST_END

TEST_BEGIN(test_nested_tags)
    const char *in = "<div><div>inner</div></div>after";
    axiom_strip_step steps[] = {{.kind = "strip_html"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "inner");
    ASSERT_STR_CONTAINS((char *)out, "after");
TEST_END

TEST_BEGIN(test_empty_tags)
    const char *in = "<br/><img src='x'/><hr>";
    axiom_strip_step steps[] = {{.kind = "strip_html"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
TEST_END

TEST_BEGIN(test_binary_input_safety)
    uint8_t bin[] = {0x00, 0xFF, 0x80, 0x7F, '<', 'p', '>', 'A', '<', '/', 'p', '>', 0x00};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    axiom_strip_step steps[] = {{.kind = "strip_html"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    int rc = axiom_strip_apply(bin, sizeof(bin), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(rc, AXIOM_STRIP_OK);
TEST_END

TEST_BEGIN(test_only_whitespace_input)
    const char *in = "   \t\n\r  \n  ";
    axiom_strip_step steps[] = {{.kind = "collapse_ws"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[64] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, 0u);
TEST_END

TEST_BEGIN(test_keep_between_no_end_marker)
    const char *in = "before <main>content goes on";
    axiom_strip_step steps[] = {{.kind = "keep_between", .pattern = "<main>", .replacement = "</main>"}};
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)in, strlen(in), &recipe, out, sizeof(out), &result), AXIOM_STRIP_OK);
    ASSERT_STR_CONTAINS((char *)out, "content goes on");
TEST_END

/* ------------------------------------------------------------------ */
/*  Runner                                                             */
/* ------------------------------------------------------------------ */

int run_part4_tests(void) {
    int before = g_test_fail;
    RUN_TEST(test_apply_with_pool_basic);
    RUN_TEST(test_apply_with_pool_null_pool);
    RUN_TEST(test_apply_with_pool_resets);
    RUN_TEST(test_pool_alloc_alignment);
    RUN_TEST(test_pool_many_small_allocs);
    RUN_TEST(test_pool_alloc_zero_returns_valid);
    RUN_TEST(test_thread_safety);
    RUN_TEST(test_large_input);
    RUN_TEST(test_aggressive_pipeline);
    RUN_TEST(test_unclosed_comment);
    RUN_TEST(test_unclosed_tag);
    RUN_TEST(test_nested_tags);
    RUN_TEST(test_empty_tags);
    RUN_TEST(test_binary_input_safety);
    RUN_TEST(test_only_whitespace_input);
    RUN_TEST(test_keep_between_no_end_marker);
    return g_test_fail - before;
}

/*
 * test_strip_part5.c â€” Main test runner.
 *
 * Calls all part runners and prints summary.
 */


extern int g_test_count, g_test_pass, g_test_fail;

int run_part1_tests(void);
int run_part2_tests(void);
int run_part3_tests(void);
int run_part4_tests(void);

int main(void) {
    printf("=== alpine_strip test suite ===\n\n");

    printf("--- Part 1: CRC32, pool, validation, basics ---\n");
    run_part1_tests();
    printf("  Part 1 done: %d passed, %d failed\n\n", g_test_pass, g_test_fail);

    int p1_fail = g_test_fail;

    printf("--- Part 2: HTML processing, text ops, measurement ---\n");
    run_part2_tests();
    printf("  Part 2 done: %d passed, %d failed (cumulative)\n\n", g_test_pass, g_test_fail);

    int p2_fail = g_test_fail;

    printf("--- Part 3: Regex, recipe mmap, confidence, flags ---\n");
    run_part3_tests();
    printf("  Part 3 done: %d passed, %d failed (cumulative)\n\n", g_test_pass, g_test_fail);

    int p3_fail = g_test_fail;

    printf("--- Part 4: Threading, pool edges, aggressive, edge cases ---\n");
    run_part4_tests();
    printf("  Part 4 done: %d passed, %d failed (cumulative)\n\n", g_test_pass, g_test_fail);

    printf("=== RESULTS: %d tests, %d passed, %d failed ===\n",
           g_test_count, g_test_pass, g_test_fail);

    if (g_test_fail == 0) {
        printf("\nALL TESTS PASSED\n");
    } else {
        printf("\n%d TESTS FAILED\n", g_test_fail);
    }

    return g_test_fail == 0 ? 0 : 1;
}
