#include "strip_engine.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int run_one_line(const char *line) {
    axiom_strip_step steps[] = {
        {.kind = "strip_tag", .pattern = "script", .replacement = ""},
        {.kind = "strip_tag", .pattern = "style", .replacement = ""},
        {.kind = "strip_html", .pattern = "", .replacement = ""},
        {.kind = "collapse_ws", .pattern = "", .replacement = ""},
    };
    axiom_strip_recipe recipe = {.steps = steps, .step_count = 4, .max_output_ratio = 1.0};
    size_t n = strlen(line);
    uint8_t *out = (uint8_t *)malloc(n + 4096u);
    if (out == NULL) {
        return 2;
    }
    axiom_strip_result result;
    int code = axiom_strip_apply((const uint8_t *)line, n, &recipe, out, n + 4096u, &result);
    if (code != AXIOM_STRIP_OK) {
        fprintf(stdout, "{\"ok\":false,\"code\":%d}\n", code);
        free(out);
        return 1;
    }
    fprintf(stdout, "{\"ok\":true,\"bytes\":%zu,\"crc32\":%u,\"signal\":\"", result.bytes_written, result.crc32);
    for (size_t i = 0; i < result.bytes_written; ++i) {
        unsigned char c = out[i];
        if (c == '"' || c == '\\') {
            fputc('\\', stdout);
            fputc(c, stdout);
        } else if (c >= 32 && c < 127) {
            fputc(c, stdout);
        }
    }
    fprintf(stdout, "\"}\n");
    free(out);
    return 0;
}

int main(void) {
    char line[1024 * 1024];
    int failed = 0;
    while (fgets(line, sizeof(line), stdin) != NULL) {
        if (run_one_line(line) != 0) {
            failed++;
        }
    }
    return failed == 0 ? 0 : 1;
}
