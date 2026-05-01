#ifndef AXIOM_RUNTIME_H
#define AXIOM_RUNTIME_H

#include <stddef.h>

#ifdef _WIN32
#  ifdef AXIOM_RUNTIME_BUILD
#    define AXIOM_API __declspec(dllexport)
#  else
#    define AXIOM_API __declspec(dllimport)
#  endif
#else
#  define AXIOM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct axiom_runtime axiom_runtime;

AXIOM_API const char *axiom_version(void);
AXIOM_API const char *axiom_integrity_version(void);
AXIOM_API const char *axiom_dic_version(void);
AXIOM_API const char *axiom_veritas_version(void);
AXIOM_API unsigned long long axiom_integrity_hash_bytes(const void *data, size_t size);
AXIOM_API int axiom_integrity_hash_file(const char *path, char *out_hex, size_t out_cap);
AXIOM_API double axiom_dic_lexical_overlap(const char *query, const char *text);
AXIOM_API int axiom_veritas_label_score(double confirm_score, double deny_score, int anchor_newer);
AXIOM_API axiom_runtime *axiom_init(const char *config_json);
AXIOM_API char *axiom_handle_json(axiom_runtime *runtime, const char *request_json);
AXIOM_API void axiom_free(char *ptr);
AXIOM_API void axiom_shutdown(axiom_runtime *runtime);

#ifdef __cplusplus
}
#endif

#endif
