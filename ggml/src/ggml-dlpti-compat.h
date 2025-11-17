#ifndef GGML_DLPTI_COMPAT_H
#define GGML_DLPTI_COMPAT_H

#include <stdint.h>

// Minimal subset of DLPTI definitions required by ggml C sources without pulling
// in the C++-only headers from the SDK.

#define DLPTI_CB_DOMAIN_HC_API ((uint32_t)65)
#define DLPTI_HC_CBID_PLATFORM_INIT ((uint32_t)1)

#ifndef GGML_DLPTI_SHARED_DATA_DEFINED
#define GGML_DLPTI_SHARED_DATA_DEFINED
typedef struct DLPTI_SharedData {
    uint64_t internal_data;
    uint64_t user_data;
} DLPTI_SharedData;
#endif

#ifdef __cplusplus
extern "C" {
#endif

int dlptiFunctionTraceEnabled(uint32_t domain, uint32_t cbid);
int dlptiFunctionEnter(uint32_t domain,
                       uint32_t cbid,
                       const char *func_name,
                       const void *func_params,
                       DLPTI_SharedData *shared_data);
void dlptiFunctionExit(uint32_t domain,
                       uint32_t cbid,
                       const char *func_name,
                       const void *func_params,
                       const void *retval,
                       DLPTI_SharedData *shared_data);

#ifdef __cplusplus
}
#endif

#endif // GGML_DLPTI_COMPAT_H

