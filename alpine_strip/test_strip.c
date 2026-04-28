#include "strip_engine.h"

#include <stdio.h>
#include <string.h>

#define ASSERT_TRUE(x) do { if (!(x)) { printf("assert failed: %s:%d: %s\n", __FILE__, __LINE__, #x); return 1; } } while (0)
#define ASSERT_EQ(a,b) do { if ((a)!=(b)) { printf("assert eq failed: %s:%d\n", __FILE__, __LINE__); return 1; } } while (0)

static int test_empty_input(void) {
    uint8_t out[16];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)"", 0, NULL, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_EQ(result.bytes_written, 0u);
    return 0;
}

static int test_strip_html_and_script(void) {
    const char *input = "<html><script>bad()</script><body><h1>Hello</h1> world</body></html>";
    axiom_strip_step steps[] = {
        {.kind = "strip_tag", .pattern = "script"},
        {.kind = "strip_html"},
        {.kind = "collapse_ws"},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 3, .max_output_ratio = 1.0};
    uint8_t out[256];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_TRUE(strstr((const char *)out, "Hello") != NULL);
    ASSERT_TRUE(strstr((const char *)out, "bad") == NULL);
    return 0;
}

static int test_ratio_guard(void) {
    const char *input = "abcdef";
    axiom_strip_recipe recipe = {.steps = NULL, .step_count = 0, .max_output_ratio = 0.5};
    uint8_t out[64];
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_ERR_REDUCTION_RATIO);
    return 0;
}

static int test_crc32(void) {
    uint32_t crc = axiom_crc32((const uint8_t *)"123456789", 9);
    ASSERT_EQ(crc, 0xcbf43926u);
    return 0;
}

static int test_default_html_entities_and_comments(void) {
    const char *input = "<html><!-- hidden --><body><p class=\"x\">A&amp;B&nbsp;C</p><style>.x{}</style></body></html>";
    uint8_t out[256] = {0};
    axiom_strip_result result;
    int code = axiom_strip_apply_default_html((const uint8_t *)input, strlen(input), out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_TRUE(strstr((const char *)out, "A&B C") != NULL);
    ASSERT_TRUE(strstr((const char *)out, "hidden") == NULL);
    ASSERT_TRUE(strstr((const char *)out, "class") == NULL);
    return 0;
}

static int test_keep_between_and_filter_lines(void) {
    const char *input = "before <main>\nkeep this\nskip that\nkeep also\n</main> after";
    axiom_strip_step steps[] = {
        {.kind = "keep_between", .pattern = "<main>", .replacement = "</main>"},
        {.kind = "filter_lines", .pattern = "keep", .replacement = ""},
        {.kind = "collapse_ws", .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 3, .max_output_ratio = 1.0};
    uint8_t out[256] = {0};
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result);
    ASSERT_EQ(code, AXIOM_STRIP_OK);
    ASSERT_TRUE(strstr((const char *)out, "keep this") != NULL);
    ASSERT_TRUE(strstr((const char *)out, "keep also") != NULL);
    ASSERT_TRUE(strstr((const char *)out, "skip") == NULL);
    return 0;
}

static int test_strip_measure(void) {
    const char *input = "<p>Hello 123</p>";
    uint8_t out[128] = {0};
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply_default_html((const uint8_t *)input, strlen(input), out, sizeof(out), &result), AXIOM_STRIP_OK);
    axiom_strip_metrics metrics;
    ASSERT_EQ(axiom_strip_measure((const uint8_t *)input, strlen(input), out, result.bytes_written, &metrics), AXIOM_STRIP_OK);
    ASSERT_TRUE(metrics.token_count >= 2u);
    ASSERT_TRUE(metrics.signal_density > 0.0);
    return 0;
}

static int test_bad_recipe_rejected(void) {
    const char *input = "hello";
    axiom_strip_step steps[] = {
        {.kind = "unknown_step", .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 1, .max_output_ratio = 1.0};
    uint8_t out[64];
    axiom_strip_result result;
    ASSERT_EQ(axiom_strip_apply((const uint8_t *)input, strlen(input), &recipe, out, sizeof(out), &result), AXIOM_STRIP_ERR_BAD_RECIPE);
    return 0;
}

int main(void) {
    int failures = 0;
    failures += test_empty_input();
    failures += test_strip_html_and_script();
    failures += test_ratio_guard();
    failures += test_crc32();
    failures += test_default_html_entities_and_comments();
    failures += test_keep_between_and_filter_lines();
    failures += test_strip_measure();
    failures += test_bad_recipe_rejected();
    if (failures == 0) {
        printf("alpine_strip tests passed\n");
    }
    return failures == 0 ? 0 : 1;
}
