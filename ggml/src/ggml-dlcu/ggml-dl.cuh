#pragma once

/**
 * @file ggml-dl.cuh
 * @brief DengLin (DL) CUDA extensions plugin interface
 *
 * This plugin provides DL-specific CUDA optimizations including:
 * - GPTQ weight quantization and storage
 * - dlblas integration for optimized GEMM operations
 * - Debug utilities
 *
 * The plugin is enabled via GGML_USE_DLCU compile flag.
 */

#ifdef GGML_USE_DLCU

#include "../ggml-cuda/common.cuh"
#include <unordered_map>
#include <mutex>

// ============================================================================
// Configuration and Environment
// ============================================================================

namespace ggml_dl {

/**
 * @brief Get DEVIT debug flag from environment
 */
bool is_devit_enabled();

/**
 * @brief Get GPTQ group size from environment (default: 32)
 */
int get_gptq_group_size();


// ============================================================================
// GPTQ Weight Quantization
// ============================================================================

/**
 * @brief GPTQ quantized weight data structure
 */
struct gptq_weight_data {
    void* qweight;              // Quantized weight on device
    void* qzeros;               // Zero points on device
    void* scales;               // Scales on device

    size_t qweight_size;        // Size in bytes
    size_t qzeros_size;
    size_t scales_size;

    int M;                      // Matrix dimensions
    int K;
    int group_size;             // GPTQ group size
    int bits;                   // Quantization bits (4 or 8)

    cudaDataType_t qweight_type;
    cudaDataType_t qzeros_type;
    cudaDataType_t scales_type;

    gptq_weight_data() : qweight(nullptr), qzeros(nullptr), scales(nullptr),
                         qweight_size(0), qzeros_size(0), scales_size(0),
                         M(0), K(0), group_size(0), bits(0) {}
};

/**
 * @brief Get GPTQ quantized weight for a tensor
 * @param tensor Input tensor (can be view, will trace to base)
 * @return Pointer to GPTQ data, or nullptr if not quantized
 */
gptq_weight_data* get_gptq_weight(const ggml_tensor* tensor);

/**
 * @brief Store GPTQ quantized weight for a tensor
 * @param tensor Target tensor (view will be traced to base)
 * @param gptq_data GPTQ quantized data (ownership transferred)
 *
 * If tensor already has GPTQ data, old data will be freed.
 */
void store_gptq_weight(const ggml_tensor* tensor, gptq_weight_data* gptq_data);

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
 * @brief Quantize tensor to GPTQ format (internal)
 * @param ctx CUDA context
 * @param src0 Source tensor on device
 * @param bits Quantization bits (4 or 8)
 */
void quantize_tensor_gptq(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    int bits);

// ============================================================================
// dlblas GEMM Operations
// ============================================================================

/**
 * @brief Perform matrix multiplication using dlblas with GPTQ weights
 * @param ctx CUDA context
 * @param src0 Weight tensor (must have GPTQ data)
 * @param src1 Input tensor
 * @param dst Output tensor
 *
 * This is the main entry point for DL-optimized matrix multiplication.
 * Falls back to standard CUDA operations if GPTQ data not found.
 */
void mul_mat_dlblas(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst);

/**
 * @brief Check if dlblas should be used for this operation
 * @param src0 Weight tensor
 * @param src1 Input tensor
 * @param dst Output tensor
 * @return true if dlblas can handle this operation
 */
bool should_use_dlblas(
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* dst);

// ============================================================================
// Debug Utilities
// ============================================================================

/**
 * @brief Print tensor values for debugging (if DEVIT enabled)
 * @param name Tensor name for logging
 * @param data Device pointer
 * @param count Number of elements to print
 * @param type Data type
 * @param stream CUDA stream
 */
void debug_print_tensor(
    const char* name,
    const void* data,
    int count,
    ggml_type type,
    cudaStream_t stream);

/**
 * @brief Verify GPTQ dequantization for debugging
 * @param gptq_data GPTQ quantized data
 * @param src0 Original tensor for comparison
 */
void debug_verify_dequant(
    const gptq_weight_data& gptq_data,
    const ggml_tensor* src0);


/**
 * @brief Cleanup all DL resources
 *
 * Should be called during shutdown to free GPTQ weights.
 */
void cleanup();

} // namespace ggml_dl

// ============================================================================
// Backend Integration (exposed to llama.cpp)
// ============================================================================

/**
 * @brief Quantize and store tensor from CPU
 * @note This function is called from llama-model.cpp during model loading
 */
void ggml_backend_cuda_gptq_quantize_and_store_from_cpu(
    int device_id,
    const ggml_tensor* tensor);

#endif // GGML_USE_DLCU
