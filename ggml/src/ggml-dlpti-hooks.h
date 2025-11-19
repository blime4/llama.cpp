#ifndef GGML_DLPTI_HOOKS_H
#define GGML_DLPTI_HOOKS_H

#include "../include/ggml.h"
#include <stdio.h>
#include <string.h>

#ifdef DLPTI_ENABLED
#ifdef __cplusplus
#include <dlpti/dl/hook.hpp>
#define DLPTI_CB_DOMAIN_RUNTIME_API ((uint32_t)2)
#define DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord ((uint32_t)135)
#else
#include "ggml-dlpti-compat.h"
#endif
#else
#include "ggml-dlpti-compat.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline const char * ggml_dlpti_get_op_name(const struct ggml_tensor * node) {
#ifdef DLPTI_ENABLED
    static char name_buf[256];
    const char * name = ggml_op_desc(node);
    if (name == NULL || name[0] == '\0') {
        name = ggml_op_name(node->op);
    }
    if (name == NULL || name[0] == '\0') {
        name = "ggml_op";
    }
    if (strncmp(name, "ggml_op_", 8) == 0) {
        return name;
    }
    snprintf(name_buf, sizeof(name_buf), "ggml_op_%s", name);
    return name_buf;
#else
    (void) node;
    return "";
#endif
}

#ifdef __cplusplus
}
#endif

#if defined(__cplusplus) && defined(DLPTI_ENABLED) && \
        defined(DLPTI_CPP_FUNCTION_TRACE_BEGIN_FUNC_NAME) && \
        defined(DLPTI_CPP_FUNCTION_TRACE_END)
#define GGML_DLPTI_TRACE_FUNCTION(func_literal)                                 \
    do {                                                                        \
        DLPTI_CPP_FUNCTION_TRACE_BEGIN_FUNC_NAME(                               \
            DLPTI_CB_DOMAIN_RUNTIME_API,                                        \
            DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord,                           \
            func_literal,                                                       \
            void);                                                              \
        DLPTI_CPP_FUNCTION_TRACE_END();                                         \
    } while (0)
#elif defined(DLPTI_ENABLED)
#define GGML_DLPTI_TRACE_FUNCTION(func_literal)                                                                 \
    do {                                                                                                        \
        DLPTI_SharedData __dlpti_shared_data = {0, 0};                                                          \
        if (dlptiFunctionTraceEnabled(DLPTI_CB_DOMAIN_RUNTIME_API, DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord)) { \
            dlptiFunctionEnter(DLPTI_CB_DOMAIN_RUNTIME_API,                                                     \
                                DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord,                                       \
                                func_literal,                                                                   \
                                NULL,                                                                           \
                                &__dlpti_shared_data);                                                          \
        }                                                                                                       \
        dlptiFunctionExit(DLPTI_CB_DOMAIN_RUNTIME_API,                                                          \
                            DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord,                                           \
                            func_literal,                                                                       \
                            NULL,                                                                               \
                            NULL,                                                                               \
                            &__dlpti_shared_data);                                                              \
    } while (0)
#else
#define GGML_DLPTI_TRACE_FUNCTION(func_literal)                                 \
    do {                                                                        \
        (void) (func_literal);                                                  \
    } while (0)
#endif

#if defined(__cplusplus) && defined(DLPTI_ENABLED) && \
        defined(DLPTI_CPP_FUNCTION_TRACE_BEGIN_FUNC_NAME) && \
        defined(DLPTI_CPP_FUNCTION_TRACE_END)
#define GGML_DLPTI_TRACE_OPERATOR(node_, BODY)                                  \
    do {                                                                        \
        const char * __dlpti_func_name = ggml_dlpti_get_op_name((node_));       \
        DLPTI_CPP_FUNCTION_TRACE_BEGIN_FUNC_NAME(                               \
            DLPTI_CB_DOMAIN_RUNTIME_API,                                        \
            DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord,                           \
            __dlpti_func_name,                                                  \
            void);                                                              \
        DLPTI_CPP_FUNCTION_TRACE_END();                                         \
        do {                                                                    \
            BODY                                                                \
        } while (0);                                                            \
    } while (0)
#elif defined(DLPTI_ENABLED)
#define GGML_DLPTI_TRACE_OPERATOR(node_, BODY)                                                                  \
    do {                                                                                                        \
        const char * __dlpti_func_name = ggml_dlpti_get_op_name((node_));                                       \
        DLPTI_SharedData __dlpti_shared_data = {0, 0};                                                          \
        if (dlptiFunctionTraceEnabled(DLPTI_CB_DOMAIN_RUNTIME_API, DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord)) { \
            dlptiFunctionEnter(DLPTI_CB_DOMAIN_RUNTIME_API,                                                     \
                              DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord,                                         \
                              __dlpti_func_name,                                                                \
                              NULL,                                                                             \
                              &__dlpti_shared_data);                                                            \
        }                                                                                                       \
        do {                                                                                                    \
            BODY                                                                                                \
        } while (0);                                                                                            \
        if (dlptiFunctionTraceEnabled(DLPTI_CB_DOMAIN_RUNTIME_API, DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord)) { \
            dlptiFunctionExit(DLPTI_CB_DOMAIN_RUNTIME_API,                                                      \
                             DLPTI_RUNTIME_TRACE_CBID_cudaEventRecord,                                          \
                             __dlpti_func_name,                                                                 \
                             NULL,                                                                              \
                             NULL,                                                                              \
                             &__dlpti_shared_data);                                                             \
        }                                                                                                       \
    } while (0)
#else
#define GGML_DLPTI_TRACE_OPERATOR(node_, BODY)                                  \
    do {                                                                        \
        (void) (node_);                                                         \
        BODY                                                                    \
    } while (0)
#endif

#endif // GGML_DLPTI_HOOKS_H
