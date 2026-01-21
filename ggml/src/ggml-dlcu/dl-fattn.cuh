#pragma once

/**
 * @file dl-fattn.cuh
 * @brief DengLin (DL) Flash Attention CUDA extensions
 *
 * This file contains DLDNN-specific Flash Attention implementations using:
 * - cuDNN MHA Forward (for ALiBi support)
 * - cuDNN ScaledDotProductAttention (for mask support)
 *
 * The implementation is enabled via GGML_USE_DLFA compile flag.
 */

#ifdef GGML_USE_DLFA

#include "../ggml-cuda/common.cuh"
#include <cudnn.h>

// Macro to control flash attention info printing (debug mode only)
#ifdef NDEBUG
#define GGML_DL_FATTN_DEBUG_PRINT(...) ((void)0)
#else
#define GGML_DL_FATTN_DEBUG_PRINT(...) do { \
    const char* env_dl_fattn_debug = getenv("GGML_DL_FATTN_DEBUG"); \
    if (env_dl_fattn_debug != nullptr && strcmp(env_dl_fattn_debug, "1") == 0) { \
        printf("[DL-FATTN] " __VA_ARGS__); \
        fflush(stdout); \
    } \
} while(0)
#endif

// ============================================================================
// Flash Attention - DLDNN Implementation
// ============================================================================

namespace ggml_dl {

/**
 * @brief Check if DLDNN Flash Attention is available for this operation
 * @param dst Output tensor
 * @return true if DLDNN can handle this flash attention operation
 */
bool flash_attn_dldnn_available(const ggml_tensor* dst);

/**
 * @brief Execute Flash Attention using DLDNN (cuDNN)
 * @param ctx CUDA context
 * @param dst Output tensor (KQV)
 *
 * This function automatically selects the appropriate cuDNN interface:
 * - cudnnMHAForward for ALiBi support
 * - cudnnScaledDotProductAttention for mask support
 */
void flash_attn_ext_dldnn(
    ggml_backend_cuda_context& ctx,
    ggml_tensor* dst);

#if 0 // will be removed later
/**
 * @brief Precise FAIL-case skip check for FLASH_ATTN_EXT
 * @param src   Q,K,V,mask tensor array (op->src)
 * @param op_params  pointer to op->op_params (int32_t array with floats)
 * @return true if this case should be skipped (known failing), false otherwise
 */
bool flash_attn_ext_should_skip(
    const ggml_tensor * const * src,
    const int32_t * op_params);
#endif

/**
 * @brief Convert tensor data between GGML types on GPU
 * @param src_data Source data pointer (device)
 * @param dst_data Destination data pointer (device)
 * @param src_tensor Source tensor metadata
 * @param dst_type Target GGML type
 * @param stream CUDA stream
 */
void convert_tensor_data(
    const void* src_data,
    void* dst_data,
    const ggml_tensor* src_tensor,
    enum ggml_type dst_type,
    cudaStream_t stream);

} // namespace ggml_dl

#endif // GGML_USE_DLFA
