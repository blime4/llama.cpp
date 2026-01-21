#pragma once
#ifdef GGML_USE_DLCU

#include "../ggml-cuda/common.cuh"
#include <unordered_map>
#include <mutex>

// Macro to control dl-mulmat debug printing (debug mode only)
#ifdef NDEBUG
#define GGML_DL_MULMAT_DEBUG_PRINT(...) ((void)0)
#else
#define GGML_DL_MULMAT_DEBUG_PRINT(...) do { \
    const char* env_dl_mulmat_debug = getenv("GGML_DL_MULMAT_DEBUG"); \
    if (env_dl_mulmat_debug != nullptr && strcmp(env_dl_mulmat_debug, "1") == 0) { \
        printf("[DL-MULMAT] " __VA_ARGS__); \
        fflush(stdout); \
    } \
} while(0)
#endif

namespace ggml_dl {

struct gptq_weight_data {
    void* qweight;
    void* qzeros;
    void* scales;

    size_t qweight_size;
    size_t qzeros_size;
    size_t scales_size;

    int M;
    int K;
    int group_size;
    int bits;

    cudaDataType_t qweight_type;
    cudaDataType_t qzeros_type;
    cudaDataType_t scales_type;

    gptq_weight_data() : qweight(nullptr), qzeros(nullptr), scales(nullptr),
                         qweight_size(0), qzeros_size(0), scales_size(0),
                         M(0), K(0), group_size(0), bits(0) {}
};


/**
 * @brief Quantize tensor to GPTQ format from CPU
 * @param device_id CUDA device ID
 * @param tensor Source tensor (must have data on CPU)
 *
 * This function:
 * 1. Copies tensor data to GPU
 * 2. Performs GPTQ quantization
 * 3. Stores quantized data in global registry
 *
 * Called from llama_model::load_tensors() for all MUL_MAT tensors.
 */
void quantize_and_store_from_cpu(int device_id, const ggml_tensor* tensor);

/**
 * @brief Calculate required memory size for GPTQ quantization
 * @param K Number of columns (input features)
 * @param M Number of rows (output features)
 * @param bits Quantization bits (4 or 8)
 * @param group_size Group size for quantization
 * @return Required memory size in bytes (aligned to 256 bytes)
 *
 * This function can be used to pre-allocate tensor memory with sufficient size
 * to allow in-place quantization (overwriting original weights).
 */
size_t calculate_gptq_required_size(int K, int M, int bits, int group_size);

/**
 * @brief Calculate required memory size for MoE GPTQ quantization
 * @param K Number of columns (input features)
 * @param M Number of rows (output features)
 * @param E Number of experts
 * @param bits Quantization bits (currently only 4 is supported)
 * @param group_size Group size for quantization
 * @return Required memory size in bytes (aligned to 256 bytes)
 *
 * This function can be used to pre-allocate tensor memory with sufficient size
 * to allow in-place quantization (overwriting original weights) for MoE tensors.
 */
size_t calculate_moe_gptq_required_size(int K, int M, int E, int bits, int group_size);


void mul_mat_dlblas(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst);


bool is_dlblas_available(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* dst,
    bool split);

void mul_mat_id_dlblas(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* ids,
    ggml_tensor* dst);


} // namespace ggml_dl

/**
 * @brief Quantize and store tensor from CPU
 * @note This function is called from llama-model.cpp during model loading
 */
void ggml_backend_cuda_gptq_quantize_and_store_from_cpu(
    int device_id,
    const ggml_tensor* tensor);

/**
 * @brief Quantize and store tensor for MoE
 * @note This function is called from llama-model.cpp during model loading
 */
void ggml_backend_cuda_moe_gptq_quantize_and_store(
    int device_id,
    const ggml_tensor* tensor);

#endif // GGML_USE_DLCU
