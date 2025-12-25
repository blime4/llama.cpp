#include <csignal>
#ifdef GGML_USE_DLCU
#include "dl-mulmat.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "../ggml-cuda/common.cuh"
#include "../ggml-cuda/convert.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

// DL-specific headers
#include "dlblas_ext.h"
#include "dldnn_ext.h"

#include "vector_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <vector>
#include <string>
#include <array>

// Import CUDA functions we need from ggml-cuda namespace
using ggml_backend_cuda_context = struct ggml_backend_cuda_context;
extern int ggml_backend_cuda_get_device_count();
extern void ggml_cuda_set_device(int device);

// ============================================================================
// Global Configuration Variables
// ============================================================================

static int GGML_CUDA_GPTQ_GROUP_SIZE = []() {
    const char* env = getenv("GGML_CUDA_GPTQ_GROUP_SIZE");
    if (env) {
        int val = atoi(env);
        if (val > 0) return val;
    }
    return 128;
}();

static int GGML_QUANT_BITS = []() {
    const char * env = getenv("GGML_QUANT_BITS");
    if (env) {
        int val = atoi(env);
        if (val == 4 || val == 8) return val;
    }
    return 4;
}();


// ============================================================================
// CUDA Kernels for GPTQ Quantization
// ============================================================================
__global__ void ggml_cuda_moe_init_data(
    int32_t* __restrict__ sorted_ids,
    int32_t* __restrict__ export_ids,
    int total_topk_ids,
    int max_num_tokens_padded,
    int max_num_m_blocks
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= max_num_tokens_padded + max_num_m_blocks) {
        return;
    }

    if (tid < max_num_tokens_padded) {
        sorted_ids[tid] = total_topk_ids;
    } else {
        export_ids[tid - max_num_tokens_padded] = 0;
    }
}

__global__ void ggml_cuda_moe_align_kernel(
    const int32_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ expert_ids,
    int32_t* __restrict__ total_tokens_post_pad,
    int32_t num_experts,
    int32_t padded_num_experts,
    int32_t experts_per_warp,
    int32_t block_size,
    size_t numel,
    int32_t* __restrict__ cumsum
) {
    extern __shared__ int32_t shared_counts[];

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int my_expert_start = warp_id * experts_per_warp;

    for (int i = 0; i < experts_per_warp; ++i) {
        if (my_expert_start + i < padded_num_experts) {
            shared_counts[warp_id * experts_per_warp + i] = 0;
        }
    }

    __syncthreads();

    const size_t tid = threadIdx.x;
    const size_t stride = blockDim.x;

    for (size_t i = tid; i < numel; i += stride) {
        int expert_id = topk_ids[i];
        int warp_idx = expert_id / experts_per_warp;
        int expert_offset = expert_id % experts_per_warp;
        atomicAdd(&shared_counts[warp_idx * experts_per_warp + expert_offset], 1);
    }

    __syncthreads();

    if (threadIdx.x == 0) {
        cumsum[0] = 0;
        for (int i = 1; i <= num_experts; ++i) {
        int expert_count = 0;
        int warp_idx = (i - 1) / experts_per_warp;
        int expert_offset = (i - 1) % experts_per_warp;
        expert_count = shared_counts[warp_idx * experts_per_warp + expert_offset];

        cumsum[i] =
            cumsum[i - 1] + (expert_count + block_size - 1) / block_size * block_size;
        }
        *total_tokens_post_pad = cumsum[num_experts];
    }

    __syncthreads();

    if (threadIdx.x < num_experts) {
        for (int i = cumsum[threadIdx.x]; i < cumsum[threadIdx.x + 1];
            i += block_size) {
        expert_ids[i / block_size] = threadIdx.x;
        }
    }
}

template <typename scalar_t>
__global__ void ggml_cuda_moe_count_and_sort_expert_tokens_kernel(
    const scalar_t* __restrict__ topk_ids,
    int32_t* __restrict__ sorted_token_ids,
    int32_t* __restrict__ cumsum_buffer,
    size_t numel
) {
    const size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = blockDim.x * gridDim.x;

    for (size_t i = tid; i < numel; i += stride) {
        int32_t expert_id = topk_ids[i];
        int32_t rank_post_pad = atomicAdd(&cumsum_buffer[expert_id], 1);
        sorted_token_ids[rank_post_pad] = i;
    }
}


template <typename T>
__global__ void ggml_cuda_moe_gptq_quantize_4_bit_fp16(
    const float* __restrict__ w,    // [M, K]
    uint8_t* __restrict__ qweight,
    uint8_t* __restrict__ qzeros,
    T* __restrict__ scales,
    int K, int group_size
) {
    int row = blockIdx.x;
    int gid = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    float value[2];
    float max_val[2];
    float min_val[2];

    value[0] = static_cast<float>(w[2 * row * K + (gid * group_size) + tid]);
    value[1] = static_cast<float>(w[(2 * row  + 1) * K + (gid * group_size) + tid]);
    max_val[0] = value[0];
    max_val[1] = value[1];
    min_val[0] = value[0];
    min_val[1] = value[1];

    // reduce max
    for (int st = 1; st < 32; st <<= 1) {
        max_val[0] = ::max(max_val[0], __shfl_xor_sync(uint32_t(-1), max_val[0], st));
        max_val[1] = ::max(max_val[1], __shfl_xor_sync(uint32_t(-1), max_val[1], st));
        min_val[0] = ::min(min_val[0], __shfl_xor_sync(uint32_t(-1), min_val[0], st));
        min_val[1] = ::min(min_val[1], __shfl_xor_sync(uint32_t(-1), min_val[1], st));
    }

    __shared__ float s_max_val[2][32];
    __shared__ float s_min_val[2][32];
    if (lane == 0) {
        s_max_val[0][warp] = max_val[0];
        s_max_val[1][warp] = max_val[1];
        s_min_val[0][warp] = min_val[0];
        s_min_val[1][warp] = min_val[1];
    }
    __syncthreads();

    max_val[0] = lane < (group_size / 32) ? s_max_val[0][lane] : -FLT_MAX;
    max_val[1] = lane < (group_size / 32) ? s_max_val[1][lane] : -FLT_MAX;
    min_val[0] = lane < (group_size / 32) ? s_min_val[0][lane] : FLT_MAX;
    min_val[1] = lane < (group_size / 32) ? s_min_val[1][lane] : FLT_MAX;

    for (int st = 1; st < 32; st <<= 1) {
        max_val[0] = ::max(max_val[0], __shfl_xor_sync(uint32_t(-1), max_val[0], st));
        max_val[1] = ::max(max_val[1], __shfl_xor_sync(uint32_t(-1), max_val[1], st));
        min_val[0] = ::min(min_val[0], __shfl_xor_sync(uint32_t(-1), min_val[0], st));
        min_val[1] = ::min(min_val[1], __shfl_xor_sync(uint32_t(-1), min_val[1], st));
    }

    float max_uint4 = 15.0f;
    float scale_0 = ::max(max_val[0] - min_val[0], 1e-5f) / max_uint4;
    float scale_1 = ::max(max_val[1] - min_val[1], 1e-5f) / max_uint4;
    float zp_0 = ::round(::abs(min_val[0] / scale_0));
    zp_0 = ::min(::max(zp_0, 0.0f), 15.0f);
    float zp_1 = ::round(::abs(min_val[1] / scale_1));
    zp_1 = ::min(::max(zp_1, 0.0f), 15.0f);

    float w_f_0 = ::round(value[0] / scale_0) + zp_0;
    w_f_0 = ::min(::max(w_f_0, 0.0f), 15.0f);
    float w_f_1 = ::round(value[1] / scale_1) + zp_1;
    w_f_1 = ::min(::max(w_f_1, 0.0f), 15.0f);

    float w_f_other_0 = __shfl_xor_sync(uint32_t(-1), w_f_0, 1);
    float w_f_other_1 = __shfl_xor_sync(uint32_t(-1), w_f_1, 1);

    uint8_t w_u8_0 = ((uint8_t)(w_f_other_0) << 4) + (uint8_t)(w_f_0);
    uint8_t w_u8_1 = ((uint8_t)(w_f_other_1) << 4) + (uint8_t)(w_f_1);

    if (tid % 2 == 0) {
        qweight[(2 * row * K + gid * group_size + tid) / 2] = w_u8_0;
        qweight[((2 * row + 1) * K + gid * group_size + tid) / 2] = w_u8_1;
    }

    uint8_t zp_0_u8 = (uint8_t)(zp_0);
    uint8_t zp_1_u8 = (uint8_t)(zp_1);

    if (tid == 0) {
        scales[2 * row * K / group_size + gid] = static_cast<T>(scale_0);
        scales[(2 * row + 1)* K / group_size + gid] = static_cast<T>(scale_1);
        qzeros[row * K / group_size + gid] = (zp_1_u8 << 4) + zp_0_u8;
    }
}

__global__ void ggml_cuda_gptq_quantize_8_bit_fp16(
    const half* __restrict__ w,     // [M, K], half type
    uint8_t* __restrict__ qweight,  // [K, M], uint8
    half* __restrict__ qzeros,      // [num_groups, M]
    half* __restrict__ scales,      // [num_groups, M]
    int K, int M, int group_size
) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int group = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || group * group_size >= K) return;

    float max_val = -FLT_MAX, min_val = FLT_MAX;
    for (int k = 0; k < group_size && group * group_size + k < K; ++k) {
        float v = __half2float(w[row * K + (group * group_size + k)]);
        if (v > max_val) max_val = v;
        if (v < min_val) min_val = v;
    }

    float scale, zero_point;
    const float qmin = 0.0f;
    const float qmax = 255.0f;

    if (max_val == min_val) {
        if (min_val == 0.0f) {
            scale = 1.0f;
            zero_point = 0.0f;
        } else {
            scale = std::abs(min_val) / 255.0f;
            if (min_val > 0.0f) {
                zero_point = qmin;
            } else {
                zero_point = qmax;
            }
        }
    } else {
        scale = (max_val - min_val) / (qmax - qmin);
        zero_point = qmin - min_val / scale;

        if (zero_point < qmin) zero_point = qmin;
        if (zero_point > qmax) zero_point = qmax;

        if (scale < 1e-8f) scale = 1e-8f;
    }

    // const int num_groups_row = (K + group_size - 1) / group_size;
    // int scale_idx = row * num_groups_row + group;
    int scale_idx = group * M + row;
    scales[scale_idx] = __float2half(scale);
    qzeros[scale_idx] = __float2half(zero_point);

    for (int k = 0; k < group_size && group * group_size + k < K; ++k) {
        int col = group * group_size + k;
        float v = __half2float(w[row * K + col]);

        // q = round(v/scale + zero_point)
        int q;
        if (max_val == min_val) {
            if (min_val == 0.0f) {
                q = 0;
            } else {
                q = (int)roundf(v / scale + zero_point);
            }
        } else {
            q = (int)roundf(v / scale + zero_point);
        }

        if (q < 0) q = 0;
        if (q > 255) q = 255;

        // qweight[row * K + col] = (uint8_t)q;
        qweight[col * M + row] = (uint8_t)q;
    }
}

__global__ void ggml_cuda_gptq_quantize_4_bit_fp16(
    const half* __restrict__ w,     // [M, K], half type
    uint8_t* __restrict__ qweight,  // [K/2, M], uint8 view as [K, M/2]
    half* __restrict__ qzeros,      // [num_groups, M]
    half* __restrict__ scales,      // [num_groups, M]
    int K, int M, int group_size
) {
    int row = blockIdx.x;
    int gid = blockIdx.y;
    int tid = threadIdx.x;
    int warp = tid / 32;
    int lane = tid % 32;

    float value[2];
    float max_val[2];
    float min_val[2];

    value[0] = static_cast<float>(w[2 * row * K + (gid * group_size) + tid]);
    value[1] = static_cast<float>(w[(2 * row  + 1) * K + (gid * group_size) + tid]);
    max_val[0] = value[0];
    max_val[1] = value[1];
    min_val[0] = value[0];
    min_val[1] = value[1];

    // reduce max
    for (int st = 1; st < 32; st <<= 1) {
        max_val[0] = ::max(max_val[0], __shfl_xor_sync(uint32_t(-1), max_val[0], st));
        max_val[1] = ::max(max_val[1], __shfl_xor_sync(uint32_t(-1), max_val[1], st));
        min_val[0] = ::min(min_val[0], __shfl_xor_sync(uint32_t(-1), min_val[0], st));
        min_val[1] = ::min(min_val[1], __shfl_xor_sync(uint32_t(-1), min_val[1], st));
    }

    __shared__ float s_max_val[2][32];
    __shared__ float s_min_val[2][32];
    if (lane == 0) {
        s_max_val[0][warp] = max_val[0];
        s_max_val[1][warp] = max_val[1];
        s_min_val[0][warp] = min_val[0];
        s_min_val[1][warp] = min_val[1];
    }
    __syncthreads();

    max_val[0] = lane < (group_size / 32) ? s_max_val[0][lane] : -FLT_MAX;
    max_val[1] = lane < (group_size / 32) ? s_max_val[1][lane] : -FLT_MAX;
    min_val[0] = lane < (group_size / 32) ? s_min_val[0][lane] : FLT_MAX;
    min_val[1] = lane < (group_size / 32) ? s_min_val[1][lane] : FLT_MAX;

    for (int st = 1; st < 32; st <<= 1) {
        max_val[0] = ::max(max_val[0], __shfl_xor_sync(uint32_t(-1), max_val[0], st));
        max_val[1] = ::max(max_val[1], __shfl_xor_sync(uint32_t(-1), max_val[1], st));
        min_val[0] = ::min(min_val[0], __shfl_xor_sync(uint32_t(-1), min_val[0], st));
        min_val[1] = ::min(min_val[1], __shfl_xor_sync(uint32_t(-1), min_val[1], st));
    }

    float max_uint4 = 15.0f;
    float scale_0 = ::max(max_val[0] - min_val[0], 1e-5f) / max_uint4;
    float scale_1 = ::max(max_val[1] - min_val[1], 1e-5f) / max_uint4;
    // float zp_0 = ::round(::abs(min_val[0] / scale_0));
    float zp_0 = ::abs(min_val[0] / scale_0);
    zp_0 = ::min(::max(zp_0, 0.0f), 15.0f);
    // float zp_1 = ::round(::abs(min_val[1] / scale_1));
    float zp_1 = ::abs(min_val[1] / scale_1);
    zp_1 = ::min(::max(zp_1, 0.0f), 15.0f);

    float w_f_0 = ::round(value[0] / scale_0 + zp_0);
    w_f_0 = ::min(::max(w_f_0, 0.0f), 15.0f);
    float w_f_1 = ::round(value[1] / scale_1 + zp_1);
    w_f_1 = ::min(::max(w_f_1, 0.0f), 15.0f);

    uint8_t w_u8 = ((uint8_t)(w_f_1) << 4) + (uint8_t)(w_f_0);
    qweight[(gid * group_size + tid) * M / 2 + row] = w_u8;

    if (tid == 0) {
        scales[gid * M + 2 * row] = static_cast<half>(scale_0);
        scales[gid * M + 2 * row + 1] = static_cast<half>(scale_1);
        qzeros[gid * M + 2 * row] = static_cast<half>(zp_0);
        qzeros[gid * M + 2 * row + 1] = static_cast<half>(zp_1);
    }
}

// ============================================================================
// GPTQ Data Structure
// ============================================================================

struct ggml_gptq_data {
    void * qweight = nullptr;
    void * qzeros = nullptr;
    void * scales = nullptr;
    int bits = 0;
    int group_size = 0;
    cudaDataType_t scales_type;
    cudaDataType_t qzeros_type;
    int num_groups = 0;
    int E = 0;
    int M = 0;
    int K = 0;
    int qweight_size = 0;
    int qzeros_size = 0;
    int scales_size = 0;
    bool is_moe = false;
};

// ============================================================================
// Helper Functions
// ============================================================================

static cudaDataType_t ggml_ptr_elem_size_to_cuda_dtype(size_t elem_size) {
    switch (elem_size) {
        case 2: return CUDA_R_16F;   // half
        case 4: return CUDA_R_32F;   // float
        case 1: return CUDA_R_8U;    // uint8
        default: return CUDA_R_32F;  // fallback
    }
}

// Calculate the total memory size required for GPTQ quantization
// Returns the size in bytes, aligned to 256 bytes boundary for better memory access
static size_t ggml_cuda_gptq_calculate_required_size(int K, int M, int bits, int group_size) {
    int num_groups = K / group_size;

    size_t qweight_size = (bits == 4) ? (size_t)M * (size_t)(K / 2) : (size_t)M * (size_t)K;
    size_t qzeros_size = (size_t)M * (size_t)num_groups * sizeof(half);
    size_t scales_size = (size_t)M * (size_t)num_groups * sizeof(half);

    // Total size with 256-byte alignment for better GPU memory access
    size_t total_size = qweight_size + qzeros_size + scales_size;
    size_t aligned_size = (total_size + 255) & ~255;  // Align to 256 bytes

    return aligned_size;
}

// Calculate the total memory size required for MoE GPTQ quantization
// Returns the size in bytes, aligned to 256 bytes boundary for better memory access
static size_t ggml_cuda_moe_gptq_calculate_required_size(int K, int M, int E, int bits, int group_size) {
    GGML_ASSERT(bits == 4);  // MoE only supports 4-bit for now
    int num_groups = K / group_size;

    size_t qweight_size = static_cast<size_t>(E) * M * K / 2;  // uint8_t
    size_t qzeros_size = static_cast<size_t>(E) * M * num_groups / 2;  // uint8_t
    size_t scales_size = static_cast<size_t>(E) * M * num_groups * sizeof(half);  // half

    // Total size with 256-byte alignment for better GPU memory access
    size_t total_size = qweight_size + qzeros_size + scales_size;
    size_t aligned_size = (total_size + 255) & ~255;  // Align to 256 bytes

    return aligned_size;
}

// ============================================================================
// Memory Management Functions
// ============================================================================

// Enhanced GPTQ Cache Key Implementation
struct ggml_gptq_cache_key {
    const ggml_tensor* base;
    int device;
    ggml_type type;
    size_t data_size;
    std::array<int64_t, GGML_MAX_DIMS> ne;

    bool operator==(const ggml_gptq_cache_key & other) const noexcept {
        return base == other.base &&
               device == other.device &&
               type == other.type &&
               data_size == other.data_size &&
               ne == other.ne;
    }
};

static inline std::array<int64_t, GGML_MAX_DIMS> ggml_gptq_tensor_shape(const ggml_tensor * tensor) {
    std::array<int64_t, GGML_MAX_DIMS> shape{};
    for (int i = 0; i < GGML_MAX_DIMS; ++i) {
        shape[i] = tensor->ne[i];
    }
    return shape;
}

struct ggml_gptq_cache_key_hash {
    size_t operator()(const ggml_gptq_cache_key & key) const noexcept {
        size_t h1 = std::hash<const void*>{}(key.base);
        size_t h2 = std::hash<int>{}(key.device);
        size_t h3 = std::hash<int>{}(static_cast<int>(key.type));
        size_t h4 = std::hash<size_t>{}(key.data_size);

        size_t result = h1;
        result = result * 31 + h2;
        result = result * 31 + h3;
        result = result * 31 + h4;

        for (const auto dim : key.ne) {
            result = result * 31 + std::hash<int64_t>{}(dim);
        }

        return result;
    }
};

static void ggml_cuda_gptq_free_weight_data(ggml_gptq_data* data, int device_id) {
    if (data == nullptr) {
        return;
    }

    ggml_cuda_set_device(device_id);

    if (data->qweight) {
        CUDA_CHECK(cudaFree(data->qweight));
    }
    if (data->qzeros) {
        CUDA_CHECK(cudaFree(data->qzeros));
    }
    if (data->scales) {
        CUDA_CHECK(cudaFree(data->scales));
    }

    delete data;
}

// DL: used to store the gptq-format weight tensor per (tensor, device).
static std::unordered_map<ggml_gptq_cache_key, ggml_gptq_data*, ggml_gptq_cache_key_hash> g_gptq_weights_map;
static std::mutex g_gptq_weights_mutex;

static void ggml_cuda_gptq_clean_cache() {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);
    for (auto & entry : g_gptq_weights_map) {
        ggml_cuda_gptq_free_weight_data(entry.second, entry.first.device);
    }
    g_gptq_weights_map.clear();
}



// helper: trace to the base (non-view) tensor
static inline const ggml_tensor * ggml_cuda_get_base_tensor(const ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

// Helper function to create enhanced cache key
static inline ggml_gptq_cache_key make_gptq_cache_key(const ggml_tensor* tensor, int device_id) {
    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);
    return {
        base,
        device_id,
        tensor->type,
        ggml_nbytes(tensor),
        ggml_gptq_tensor_shape(tensor)
    };
}

// DL: used to store the gptq-format weight tensor.
static void ggml_cuda_gptq_store_weight(const ggml_tensor* tensor, ggml_gptq_data* gptq_data, int device_id) {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);

    const ggml_gptq_cache_key key = make_gptq_cache_key(tensor, device_id);

    // DL: if the weight tensor already exists, release the old one.
    auto it = g_gptq_weights_map.find(key);
    if (it != g_gptq_weights_map.end()) {
        ggml_gptq_data* old_data = it->second;
        // Only free memory if it's not pointing to the original tensor data
        // Check if pointers are within the original tensor's memory range
        const void* tensor_base = tensor->data;
        size_t tensor_size = ggml_nbytes(tensor);
        const char* tensor_end = (const char*)tensor_base + tensor_size;

        bool old_qweight_in_tensor = (old_data->qweight >= tensor_base &&
                                      (const char*)old_data->qweight < tensor_end);
        bool old_qzeros_in_tensor = (old_data->qzeros >= tensor_base &&
                                     (const char*)old_data->qzeros < tensor_end);
        bool old_scales_in_tensor = (old_data->scales >= tensor_base &&
                                     (const char*)old_data->scales < tensor_end);

        if (old_data->qweight && !old_qweight_in_tensor) {
            cudaFree(old_data->qweight);
        }
        if (old_data->qzeros && !old_qzeros_in_tensor) {
            cudaFree(old_data->qzeros);
        }
        if (old_data->scales && !old_scales_in_tensor) {
            cudaFree(old_data->scales);
        }
        delete old_data;
    }

    g_gptq_weights_map[key] = gptq_data;
}

static ggml_gptq_data* ggml_cuda_gptq_get_weight(const ggml_tensor* tensor, int device_id) {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);

    const ggml_gptq_cache_key key = make_gptq_cache_key(tensor, device_id);

    auto it = g_gptq_weights_map.find(key);
    return (it != g_gptq_weights_map.end()) ? it->second : nullptr;
}

static bool ggml_cuda_gptq_has_weight(const ggml_tensor* tensor, int device_id) {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);

    const ggml_gptq_cache_key key = make_gptq_cache_key(tensor, device_id);

    return g_gptq_weights_map.find(key) != g_gptq_weights_map.end();
}

static ggml_gptq_data* ggml_cuda_gptq_get_weight_any(const ggml_tensor* tensor, int * device_id_out) {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);
    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);
    const auto shape = ggml_gptq_tensor_shape(tensor);
    const ggml_type tensor_type = tensor->type;
    const size_t tensor_size = ggml_nbytes(tensor);
    for (const auto & entry : g_gptq_weights_map) {
        if (entry.first.base == base &&
            entry.first.ne == shape &&
            entry.first.type == tensor_type &&
            entry.first.data_size == tensor_size) {
            if (device_id_out) {
                *device_id_out = entry.first.device;
            }
            return entry.second;
        }
    }
    if (device_id_out) {
        *device_id_out = -1;
    }
    return nullptr;
}

static bool ggml_cuda_gptq_has_any_weight(const ggml_tensor* tensor) {
    int dummy_device = -1;
    return ggml_cuda_gptq_get_weight_any(tensor, &dummy_device) != nullptr;
}

static void ggml_cuda_copy_device_buffer(void * dst, int dst_device, const void * src, int src_device, size_t size) {
    if (size == 0) {
        return;
    }

    if (src_device == dst_device) {
        ggml_cuda_set_device(dst_device);
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
        return;
    }

    cudaError_t err = cudaMemcpyPeer(dst, dst_device, src, src_device, size);
    if (err == cudaErrorInvalidDevice || err == cudaErrorPeerAccessUnsupported || err == cudaErrorInvalidValue || err == cudaErrorInvalidDevicePointer) {
        // Peer access unavailable, fall back to host staging
        (void)cudaGetLastError(); // clear the error
        std::vector<char> host_buffer(size);

        ggml_cuda_set_device(src_device);
        CUDA_CHECK(cudaMemcpy(host_buffer.data(), src, size, cudaMemcpyDeviceToHost));

        ggml_cuda_set_device(dst_device);
        CUDA_CHECK(cudaMemcpy(dst, host_buffer.data(), size, cudaMemcpyHostToDevice));
    } else {
        CUDA_CHECK(err);
    }
}

static ggml_gptq_data* ggml_cuda_gptq_clone_weight(const ggml_tensor* tensor, const ggml_gptq_data* src_data, int src_device, int dst_device) {
    GGML_ASSERT(src_data != nullptr);
    GGML_DL_MULMAT_DEBUG_PRINT("cloning GPTQ weight '%s' from device %d to device %d\n",
                               tensor->name, src_device, dst_device);

    ggml_gptq_data* cloned = new ggml_gptq_data();
    cloned->bits         = src_data->bits;
    cloned->group_size   = src_data->group_size;
    cloned->scales_type  = src_data->scales_type;
    cloned->qzeros_type  = src_data->qzeros_type;
    cloned->num_groups   = src_data->num_groups;
    cloned->E            = src_data->E;
    cloned->M            = src_data->M;
    cloned->K            = src_data->K;
    cloned->qweight_size = src_data->qweight_size;
    cloned->qzeros_size  = src_data->qzeros_size;
    cloned->scales_size  = src_data->scales_size;
    cloned->is_moe       = src_data->is_moe;

    ggml_cuda_set_device(dst_device);

    if (cloned->qweight_size > 0) {
        CUDA_CHECK(cudaMalloc(&cloned->qweight, cloned->qweight_size));
        ggml_cuda_copy_device_buffer(cloned->qweight, dst_device, src_data->qweight, src_device, cloned->qweight_size);
    } else {
        cloned->qweight = nullptr;
    }

    if (cloned->qzeros_size > 0) {
        CUDA_CHECK(cudaMalloc(&cloned->qzeros, cloned->qzeros_size));
        ggml_cuda_copy_device_buffer(cloned->qzeros, dst_device, src_data->qzeros, src_device, cloned->qzeros_size);
    } else {
        cloned->qzeros = nullptr;
    }

    if (cloned->scales_size > 0) {
        CUDA_CHECK(cudaMalloc(&cloned->scales, cloned->scales_size));
        ggml_cuda_copy_device_buffer(cloned->scales, dst_device, src_data->scales, src_device, cloned->scales_size);
    } else {
        cloned->scales = nullptr;
    }

    ggml_cuda_gptq_store_weight(tensor, cloned, dst_device);

    return cloned;
}

static void ggml_cuda_gptq_ensure_on_device(const ggml_tensor* tensor, ggml_gptq_data* data, int device_id) {
    // Check if pointers are within the original tensor's memory range (reused memory)
    const void* tensor_base = tensor->data;
    size_t tensor_size = ggml_nbytes(tensor);
    const char* tensor_end = (const char*)tensor_base + tensor_size;

    bool qweight_in_tensor = (data->qweight >= tensor_base &&
                               (const char*)data->qweight < tensor_end);
    bool qzeros_in_tensor = (data->qzeros >= tensor_base &&
                              (const char*)data->qzeros < tensor_end);
    bool scales_in_tensor = (data->scales >= tensor_base &&
                              (const char*)data->scales < tensor_end);

    // If memory is reused from tensor, we should not migrate it separately
    // The tensor buffer management will handle device placement
    if (qweight_in_tensor && qzeros_in_tensor && scales_in_tensor) {
        GGML_DL_MULMAT_DEBUG_PRINT("GPTQ data for tensor '%s' uses reused memory, skipping device migration\n",
                                   tensor->name);
        return;
    }

    auto ensure_ptr = [&](void** ptr, size_t size, const char* label) {
        if (*ptr == nullptr || size == 0) {
            return;
        }

        // Skip if pointer is within tensor memory (reused)
        if ((*ptr >= tensor_base && (const char*)*ptr < tensor_end)) {
            return;
        }

        cudaPointerAttributes attr;
        cudaError_t err = cudaPointerGetAttributes(&attr, *ptr);
        if (err != cudaSuccess || attr.type != cudaMemoryTypeDevice || attr.device != device_id) {
            const int src_device = (err == cudaSuccess && attr.type == cudaMemoryTypeDevice) ? attr.device : device_id;
            void* new_ptr = nullptr;
            ggml_cuda_set_device(device_id);
            CUDA_CHECK(cudaMalloc(&new_ptr, size));
            ggml_cuda_copy_device_buffer(new_ptr, device_id, *ptr, src_device, size);
            if (attr.type == cudaMemoryTypeDevice && attr.device >= 0) {
                ggml_cuda_set_device(attr.device);
                CUDA_CHECK(cudaFree(*ptr));
            } else {
                ggml_cuda_set_device(device_id);
                CUDA_CHECK(cudaFree(*ptr));
            }
            *ptr = new_ptr;
            GGML_DL_MULMAT_DEBUG_PRINT("migrated GPTQ buffer %s for tensor '%s' to device %d\n",
                                       label, tensor->name, device_id);
        }
    };

    ensure_ptr(&data->qweight, data->qweight_size, "qweight");
    ensure_ptr(&data->qzeros,  data->qzeros_size,  "qzeros");
    ensure_ptr(&data->scales,  data->scales_size,  "scales");
}

// ============================================================================
// GPTQ Quantization Implementation
// ============================================================================

static void ggml_cuda_gptq_quantize_and_store(ggml_backend_cuda_context & ctx, const void * src0_ptr, const ggml_tensor * src0, int bits = 8, bool reuse_original_memory = false) {

    // 2. quantize the weight tensor to gptq-format
    int K = src0->ne[0];
    int M = src0->ne[1];
    int group_size = std::min(GGML_CUDA_GPTQ_GROUP_SIZE, K); // Ensure group_size <= K to avoid issues with small K

    if(!GGML_IS_TEST) {
        // Support both 4-bit and 8-bit quantization
        GGML_ASSERT((bits == 4 || bits == 8) && "Only 4-bit and 8-bit quantization supported");
        GGML_ASSERT(K % group_size == 0);
        GGML_ASSERT(M % 2 == 0);
        GGML_ASSERT(group_size % 32 == 0);
    }{
        // Handle special cases for small K values
        if (bits == 4 && (K < 2 || K % 2 != 0)) {
            GGML_DL_MULMAT_DEBUG_PRINT("WARNING: K=%d is not suitable for 4-bit quantization (requires K >= 2 and even), falling back to 8-bit\n", K);
            bits = 8;
        }

        // Ensure minimum reasonable group_size for 4-bit quantization
        if (bits == 4) {
            if (group_size < 2) {
                group_size = std::min(2, K);
                GGML_DL_MULMAT_DEBUG_PRINT("WARNING: Adjusted group_size from %d to %d for 4-bit quantization\n",
                                        std::min(GGML_CUDA_GPTQ_GROUP_SIZE, K), group_size);
            }
            // Ensure group_size is even for 4-bit packing
            if (group_size % 2 != 0) {
                group_size = (group_size + 1) & ~1; // Round up to next even number
                group_size = std::min(group_size, K);
                GGML_DL_MULMAT_DEBUG_PRINT("INFO: Adjusted group_size to %d (must be even for 4-bit packing)\n", group_size);
            }
        }
    }

    int num_groups = K / group_size;

    const int id = ctx.device;
    ggml_cuda_set_device(id);

    // 3. create gptq_data
    ggml_gptq_data* gptq_data = new ggml_gptq_data;

    // 4. Calculate required memory sizes
    // For 4-bit: each byte stores 2 values, so we need M * (K/2) bytes for qweight
    // w [M, K] fp16
    // qweight [K / pack_factor, M] uint8
    // qzeros [num_groups, M] fp16
    // scales [num_groups, M] fp16
    size_t qweight_size = (bits == 4) ? (size_t)M * (size_t)(K / 2) : (size_t)M * (size_t)K;
    size_t qzeros_size = (size_t)M * (size_t)num_groups * sizeof(half);
    size_t scales_size = (size_t)M * (size_t)num_groups * sizeof(half);
    size_t total_gptq_size = ggml_cuda_gptq_calculate_required_size(K, M, bits, group_size);

    // Check if we can reuse original tensor memory
    size_t original_tensor_size = ggml_nbytes(src0);
    bool can_reuse = reuse_original_memory && (original_tensor_size >= total_gptq_size);

    if (can_reuse) {
        // Reuse original tensor memory: layout GPTQ data in-place
        // Memory layout: [qweight][qzeros][scales]
        void* base_ptr = const_cast<void*>(src0_ptr);
        gptq_data->qweight = base_ptr;
        gptq_data->qzeros = (char*)base_ptr + qweight_size;
        gptq_data->scales = (char*)base_ptr + qweight_size + qzeros_size;

        GGML_DL_MULMAT_DEBUG_PRINT("Reusing original memory for GPTQ quantization of tensor '%s': original_size=%zu, gptq_size=%zu\n",
                                   src0->name, original_tensor_size, total_gptq_size);
    } else {
        // Allocate new GPU memory (original behavior)
        if (reuse_original_memory) {
            GGML_DL_MULMAT_DEBUG_PRINT("WARNING: Cannot reuse memory for tensor '%s': original_size=%zu < required=%zu, allocating new memory\n",
                                       src0->name, original_tensor_size, total_gptq_size);
        }
        CUDA_CHECK(cudaMalloc(&gptq_data->qweight, qweight_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&gptq_data->qzeros, qzeros_size));
        CUDA_CHECK(cudaMalloc(&gptq_data->scales, scales_size));
    }

    // nowaday, only support float.
    cudaStream_t stream = ctx.stream();

    // 3.1 If src0 is quantized (or not FP16), dequantize/convert to FP16 first on device
    ggml_cuda_pool_alloc<half> src0_as_f16(ctx.pool(id));
    if (src0->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src0->type);
        GGML_ASSERT(to_fp16_cuda != nullptr);

        const size_t ne = (size_t) M * (size_t) K;
        src0_as_f16.alloc(ne);

        // Check if tensor is contiguously allocated, if not use non-contiguous conversion
        if (ggml_is_contiguously_allocated(src0)) {
            to_fp16_cuda(src0_ptr, src0_as_f16.get(), ne, stream);
        } else {
            // Use non-contiguous conversion for view tensors
            const to_fp16_nc_cuda_t to_fp16_nc = ggml_get_to_fp16_nc_cuda(src0->type);
            GGML_ASSERT(to_fp16_nc != nullptr);
            const int64_t ts = ggml_type_size(src0->type);
            const int64_t s01 = src0->nb[1] / ts;
            const int64_t s02 = src0->nb[2] / ts;
            const int64_t s03 = src0->nb[3] / ts;
            to_fp16_nc(src0_ptr, src0_as_f16.get(), K, M, src0->ne[2], src0->ne[3], s01, s02, s03, stream);
        }
    }
    const half * src0_ptr_as_fp16 = src0->type == GGML_TYPE_F16 ? (const half *) src0_ptr : src0_as_f16.get();

    if (bits == 8) {
        dim3 blockDim(32, 8);
        dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (num_groups + blockDim.y - 1) / blockDim.y);
        ggml_cuda_gptq_quantize_8_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
            src0_ptr_as_fp16, (uint8_t*)gptq_data->qweight, (half*)gptq_data->qzeros, (half*)gptq_data->scales,
            K, M, group_size);
    } else if (bits == 4) {
        dim3 blockDim(group_size);
        dim3 gridDim(M / 2, num_groups);
        ggml_cuda_gptq_quantize_4_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
            src0_ptr_as_fp16, (uint8_t*)gptq_data->qweight,
            (half*)gptq_data->qzeros, (half*)gptq_data->scales,
            K, M, group_size);
    }

    CUDA_CHECK(cudaGetLastError());

    gptq_data->group_size = group_size;
    gptq_data->bits = bits;
    gptq_data->scales_type = ggml_ptr_elem_size_to_cuda_dtype(sizeof(half));
    gptq_data->qzeros_type  = ggml_ptr_elem_size_to_cuda_dtype(sizeof(half));
    gptq_data->num_groups = num_groups;
    gptq_data->M = M;
    gptq_data->K = K;
    gptq_data->qweight_size = (int)qweight_size * sizeof(uint8_t);
    gptq_data->qzeros_size = (int)qzeros_size;
    gptq_data->scales_size = (int)scales_size;
    // DL: store it to the mapping table
    ggml_cuda_gptq_store_weight(src0, gptq_data, id);
}

static void ggml_cuda_moe_gptq_quantize_and_store(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, int bits = 4, bool reuse_original_memory = false) {
    // 1. get the weight tensor
    // 2. quantize the weight tensor to gptq-format

    // 1. get the weight tensor
    // GGML_LOG_DEBUG("DL: [%s] quantizing weight tensor %s to %d-bit\n", __FUNCTION__, src0->name, bits);
    const void* src0_ptr = src0->data;
    void* data = src0->data;

    // 2. quantize the weight tensor to gptq-format
    int K = src0->ne[0];
    int M = src0->ne[1];
    int E = src0->ne[2];
    int group_size = 128;

    GGML_ASSERT(bits == 4);
    GGML_ASSERT(K % group_size == 0);
    GGML_ASSERT(M % 2 == 0);
    GGML_ASSERT(group_size % 32 == 0);

    int num_groups = K / group_size;

    const int id = ctx.device;
    ggml_cuda_set_device(id);

    // 3. create gptq_data
    ggml_gptq_data* gptq_data = new ggml_gptq_data;

    // 4. Calculate required memory sizes
    using ScalarType = half;
    size_t qweight_size = static_cast<size_t>(E) * M * K / 2;
    size_t qzeros_size = static_cast<size_t>(E) * M * num_groups / 2;
    size_t scales_size = static_cast<size_t>(E) * M * num_groups;
    size_t total_gptq_size = ggml_cuda_moe_gptq_calculate_required_size(K, M, E, bits, group_size);

    // Check if we can reuse original tensor memory
    size_t original_tensor_size = ggml_nbytes(src0);
    bool can_reuse = reuse_original_memory && (original_tensor_size >= total_gptq_size);

    if (can_reuse) {
        // Reuse original tensor memory: layout GPTQ data in-place
        // Memory layout: [qweight][qzeros][scales]
        void* base_ptr = const_cast<void*>(src0_ptr);
        gptq_data->qweight = base_ptr;
        gptq_data->qzeros = (char*)base_ptr + qweight_size;
        gptq_data->scales = (char*)base_ptr + qweight_size + qzeros_size;

        GGML_DL_MULMAT_DEBUG_PRINT("Reusing original memory for MoE GPTQ quantization of tensor '%s': original_size=%zu, gptq_size=%zu\n",
                                   src0->name, original_tensor_size, total_gptq_size);
    } else {
        // Allocate new GPU memory (original behavior)
        if (reuse_original_memory) {
            GGML_DL_MULMAT_DEBUG_PRINT("WARNING: Cannot reuse memory for MoE tensor '%s': original_size=%zu < required=%zu, allocating new memory\n",
                                       src0->name, original_tensor_size, total_gptq_size);
        }
        CUDA_CHECK(cudaMalloc(&gptq_data->qweight, qweight_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&gptq_data->qzeros, qzeros_size * sizeof(uint8_t)));
        CUDA_CHECK(cudaMalloc(&gptq_data->scales, scales_size * sizeof(ScalarType)));
    }

    cudaStream_t stream = ctx.stream();

    // 3.1 If src0 is quantized (or not FP16), dequantize/convert to FP16 first on device
    ggml_cuda_pool_alloc<float> src0_as_fp32(ctx.pool(id));
    if (src0->type != GGML_TYPE_F32) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(src0->type);
        GGML_ASSERT(to_fp32_cuda != nullptr);

        const size_t ne = static_cast<size_t>(E) * M * K;
        src0_as_fp32.alloc(ne);

        // Check if tensor is contiguously allocated, if not use non-contiguous conversion
        if (ggml_is_contiguously_allocated(src0)) {
            to_fp32_cuda(src0_ptr, src0_as_fp32.get(), ne, stream);
        } else {
            // Use non-contiguous conversion for view tensors
            const to_fp32_nc_cuda_t to_fp32_nc = ggml_get_to_fp32_nc_cuda(src0->type);
            GGML_ASSERT(to_fp32_nc != nullptr);
            const int64_t ts = ggml_type_size(src0->type);
            const int64_t s01 = src0->nb[1] / ts;
            const int64_t s02 = src0->nb[2] / ts;
            const int64_t s03 = src0->nb[3] / ts;
            to_fp32_nc(src0_ptr, src0_as_fp32.get(), src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3], s01, s02, s03, stream);
        }
    }
    const float * src0_ptr_as_fp32 = src0->type == GGML_TYPE_F32 ? (const float *) src0_ptr : src0_as_fp32.get();

    dim3 blockDim(group_size);
    dim3 gridDim(E * M / 2, num_groups);
    ggml_cuda_moe_gptq_quantize_4_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
        src0_ptr_as_fp32, (uint8_t*)gptq_data->qweight, (uint8_t*)gptq_data->qzeros,
        (ScalarType*)gptq_data->scales, K, group_size);

    CUDA_CHECK(cudaGetLastError());

    gptq_data->group_size = group_size;
    gptq_data->bits = bits;
    gptq_data->scales_type = ggml_ptr_elem_size_to_cuda_dtype(sizeof(ScalarType));
    gptq_data->qzeros_type  = ggml_ptr_elem_size_to_cuda_dtype(sizeof(uint8_t));
    gptq_data->num_groups = num_groups;
    gptq_data->E = E;
    gptq_data->M = M;
    gptq_data->K = K;
    gptq_data->qweight_size = (int)(qweight_size * sizeof(uint8_t));
    gptq_data->qzeros_size = (int)(qzeros_size * sizeof(uint8_t));
    gptq_data->scales_size = (int)(scales_size * sizeof(ScalarType));
    gptq_data->is_moe = true;
    // DL: store it to the mapping table
    ggml_cuda_gptq_store_weight(src0, gptq_data, id);
}

static void ggml_cuda_debug_verify_dequant(const ggml_gptq_data & gptq_data, const ggml_tensor * src0) {
    const int M = gptq_data.M;
    const int K = gptq_data.K;
    const int group_size = gptq_data.group_size;
    const int num_groups = gptq_data.num_groups;

    // For 4-bit quantization, each byte stores 2 values, so we need M * (K/2) bytes
    size_t qweight_size = (gptq_data.bits == 4) ? (size_t)M * (size_t)(K / 2) : (size_t)M * (size_t)K;
    std::vector<uint8_t> qweight_h(qweight_size);
    CUDA_CHECK(cudaMemcpy(qweight_h.data(), gptq_data.qweight, qweight_h.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));

    std::vector<float> qzeros_f((size_t)M * (size_t)num_groups);
    std::vector<float> scales_f((size_t)M * (size_t)num_groups);
    if (gptq_data.qzeros_type == CUDA_R_16F) {
        std::vector<half> tmp((size_t)M * (size_t)num_groups);
        CUDA_CHECK(cudaMemcpy(tmp.data(), gptq_data.qzeros, tmp.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < tmp.size(); ++i) qzeros_f[i] = __half2float(tmp[i]);
    } else if (gptq_data.qzeros_type == CUDA_R_32F) {
        CUDA_CHECK(cudaMemcpy(qzeros_f.data(), gptq_data.qzeros, qzeros_f.size() * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        GGML_LOG("[dequant-verify] unsupported qzeros_type=%d\n", (int)gptq_data.qzeros_type);
        return;
    }

    if (gptq_data.scales_type == CUDA_R_16F) {
        std::vector<half> tmp((size_t)M * (size_t)num_groups);
        CUDA_CHECK(cudaMemcpy(tmp.data(), gptq_data.scales, tmp.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < tmp.size(); ++i) scales_f[i] = __half2float(tmp[i]);
    } else if (gptq_data.scales_type == CUDA_R_32F) {
        CUDA_CHECK(cudaMemcpy(scales_f.data(), gptq_data.scales, scales_f.size() * sizeof(float), cudaMemcpyDeviceToHost));
    } else {
        printf("[dequant-verify] unsupported scales_type=%d\n", (int)gptq_data.scales_type);
        return;
    }

    std::vector<float> weight_ref((size_t)M * (size_t)K);
    if (src0->type == GGML_TYPE_F16) {
        std::vector<half> w_h((size_t)M * (size_t)K);
        CUDA_CHECK(cudaMemcpy(w_h.data(), src0->data, w_h.size() * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < w_h.size(); ++i) weight_ref[i] = __half2float(w_h[i]);
    } else if (ggml_is_quantized(src0->type)) {
        const ggml_type qtype = src0->type;
        const size_t row_size_q = ggml_row_size(qtype, K);
        std::vector<uint8_t> qbuf((size_t)M * row_size_q);
        CUDA_CHECK(cudaMemcpy(qbuf.data(), src0->data, qbuf.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost));

        const ggml_type_traits *traits = ggml_get_type_traits(qtype);
        GGML_ASSERT(traits && traits->to_float && "quantized type missing to_float dequantizer");
        for (int row = 0; row < M; ++row) {
            const void *row_q = (const void *)(qbuf.data() + (size_t)row * row_size_q);
            float *row_f = weight_ref.data() + (size_t)row * (size_t)K;
            traits->to_float(row_q, row_f, K);
        }
    } else {
        CUDA_CHECK(cudaMemcpy(weight_ref.data(), src0->data, weight_ref.size() * sizeof(float), cudaMemcpyDeviceToHost));
    }

    printf("[dequant-verify] scales (first 5): ");
    for (int i = 0; i < 5 && i < (int)scales_f.size(); ++i) printf("%f ", scales_f[i]);
    printf("\n");
    printf("[dequant-verify] qzeros (first 5): ");
    for (int i = 0; i < 5 && i < (int)qzeros_f.size(); ++i) printf("%f ", qzeros_f[i]);
    printf("\n");
    printf("[dequant-verify] bits=%d, qweight (first 5): ", gptq_data.bits);
    if (gptq_data.bits == 4) {
        // For 4-bit, show unpacked values
        for (int i = 0; i < 5 && i < (int)qweight_size; ++i) {
            uint8_t byte = qweight_h[i];
            int val1 = byte & 0x0F;        // Low 4 bits
            int val2 = (byte >> 4) & 0x0F; // High 4 bits
            printf("[%d,%d] ", val1, val2);
        }
    } else {
        // For 8-bit, show direct values
        for (int i = 0; i < 5 && i < (int)qweight_size; ++i) printf("%d ", (int)qweight_h[i]);
    }
    printf("\n");

    const size_t total = (size_t)M * (size_t)K;
    const size_t max_check = std::min(total, (size_t)10000);
    double total_err = 0.0;
    double max_err = 0.0;

    // Function to get quantized value for a given logical index
    auto get_qweight_val = [&](size_t logical_idx) -> int {
        if (gptq_data.bits == 4) {
            size_t byte_idx = logical_idx / 2;
            bool is_high_4bit = (logical_idx % 2) == 1;
            uint8_t byte = qweight_h[byte_idx];
            if (is_high_4bit) {
                return (byte >> 4) & 0x0F;
            } else {
                return byte & 0x0F;
            }
        } else {
            return qweight_h[logical_idx];
        }
    };

    for (int i = 0; i < 5 && i < (int)total; ++i) {
        int col = i % K;
        int row = i / K;
        int scale_idx = row * num_groups + (col / group_size);
        int q_val = get_qweight_val(i);
        float dq = (float(q_val) - qzeros_f[scale_idx]) * scales_f[scale_idx];
        float abs_err = fabsf(weight_ref[i] - dq);
        printf("[dequant-verify] w[%d] ref=%f, q=%d, dq=%f, abs_err=%f\n", i, weight_ref[i], q_val, dq, abs_err);
    }

    for (size_t i = 0; i < max_check; ++i) {
        int col = (int)(i % K);
        int row = (int)(i / K);
        int scale_idx = row * num_groups + (col / group_size);
        int q_val = get_qweight_val(i);
        float dq = (float(q_val) - qzeros_f[scale_idx]) * scales_f[scale_idx];
        float abs_err = fabsf(weight_ref[i] - dq);
        total_err += abs_err;
        if (abs_err > max_err) max_err = abs_err;
    }
    double avg_err = (max_check > 0) ? (total_err / (double)max_check) : 0.0;
    printf("[dequant-verify] samples=%zu, avg_abs_err=%e, max_abs_err=%e\n", max_check, avg_err, max_err);
}

static ggml_gptq_data* ggml_cuda_gptq_get_or_create_weight(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor* tensor,
    int device_id) {

    ggml_gptq_data* gptq_weight = nullptr;
    if(GGML_IS_TEST){
        ggml_cuda_gptq_clean_cache();
    }

    gptq_weight = ggml_cuda_gptq_get_weight(tensor, device_id);
    if (gptq_weight) {
        return gptq_weight;
    }

    int existing_device = -1;
    ggml_gptq_data* existing_weight = ggml_cuda_gptq_get_weight_any(tensor, &existing_device);
    if (existing_weight) {
        gptq_weight = ggml_cuda_gptq_clone_weight(tensor, existing_weight, existing_device, device_id);
    } else {
        const int bits = GGML_QUANT_BITS;
        GGML_DL_MULMAT_DEBUG_PRINT("[performance warning] device %d missing GPTQ weight for '%s', re-quantizing (%d-bit)\n",
            device_id, tensor->name, bits);

        if (tensor->ne[2] > 1) {
            ggml_cuda_moe_gptq_quantize_and_store(ctx, tensor, bits);
        } else {
            ggml_cuda_gptq_quantize_and_store(ctx, tensor->data, tensor, bits);
        }
        gptq_weight = ggml_cuda_gptq_get_weight(tensor, device_id);
    }

    if (gptq_weight) {
        ggml_cuda_gptq_ensure_on_device(tensor, gptq_weight, device_id);
    }

    return gptq_weight;
}

// DL: used to quantize the weight tensor from CPU, and then store it to the mapping table.
void ggml_backend_cuda_gptq_quantize_and_store_from_cpu(int device_id, const ggml_tensor* tensor) {
    CUDA_CHECK(cudaSetDevice(device_id));
    ggml_backend_cuda_context ctx(device_id);

    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);

    int bits = GGML_QUANT_BITS;

    // ensure base mapping exists for reuse scenario
    if (!ggml_cuda_gptq_has_weight(base, device_id)) {
        // Try to reuse original tensor memory to save VRAM
        // This will overwrite the original weight data with GPTQ quantized data
        ggml_cuda_gptq_quantize_and_store(ctx, base->data, base, bits, true);
    }

    GGML_ASSERT(ggml_cuda_gptq_has_weight(base, device_id) && "ggml_cuda_gptq_quantize_and_store unified the base and view quantization, so we can just return here.");
}

void ggml_backend_cuda_moe_gptq_quantize_and_store(int device_id, const ggml_tensor* tensor) {
    CUDA_CHECK(cudaSetDevice(device_id));
    ggml_backend_cuda_context ctx(device_id);

    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);
    int bits = 4;   // moe only support gptq int4 now

    if (!ggml_cuda_gptq_has_weight(base, device_id)) {
        // Try to reuse original tensor memory to save VRAM
        // This will overwrite the original weight data with GPTQ quantized data
        ggml_cuda_moe_gptq_quantize_and_store(ctx, base, bits, true);
    }

    GGML_ASSERT(ggml_cuda_gptq_has_weight(base, device_id));
}

// ============================================================================
// dlblas GEMM Operations
// ============================================================================

static cudaDataType_t ggml_type_to_cuda_dtype(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return CUDA_R_16F;
        case GGML_TYPE_F32: return CUDA_R_32F;
        case GGML_TYPE_BF16: return CUDA_R_16BF;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return CUDA_R_8U;
        default:
            return CUDA_R_32F;
    }
}

static cudnnDataType_t ggml_type_to_cudnn_dtype(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16: return CUDNN_DATA_HALF;
        case GGML_TYPE_F32: return CUDNN_DATA_FLOAT;
        case GGML_TYPE_BF16: return CUDNN_DATA_BFLOAT16;
        case GGML_TYPE_I32: return CUDNN_DATA_INT32;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
        case GGML_TYPE_Q2_K:
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:
        case GGML_TYPE_Q8_K:
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ1_S:
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ4_XS:
        case GGML_TYPE_I8:
        case GGML_TYPE_I16:
        // case GGML_TYPE_I32:
        case GGML_TYPE_I64:
        case GGML_TYPE_IQ1_M:
        case GGML_TYPE_TQ1_0:
        case GGML_TYPE_TQ2_0:
            return CUDNN_DATA_UINT8;
        default:
            return CUDNN_DATA_FLOAT;
    }
}

struct InvokeFusedMoeArgs {

    struct TensorInfo {
        void* ptr = nullptr;
        cudnnTensorDescriptor_t desc;

        TensorInfo() {
            int dim[3] = {1, 1, 1};
            int stride[3] = {1, 1, 1};
            CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc, CUDNN_DATA_FLOAT, 3, dim, stride));
        }

        void set(const ggml_type type_, const int dim_[3], const int stride_[3], void* ptr_) {
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(desc,
                    ggml_type_to_cudnn_dtype(type_), 3, dim_, stride_));
            ptr = ptr_;
        }

        ~TensorInfo() {
            CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
        }
    };

    TensorInfo A;
    TensorInfo B;
    TensorInfo C;
    TensorInfo B_scale;
    TensorInfo B_zp;
    TensorInfo topk_weights;
    TensorInfo topk_ids;
    TensorInfo sorted_token_ids;
    TensorInfo expert_ids;
    TensorInfo num_tokens_post_padded;

    InvokeFusedMoeArgs() = default;

    void set(TensorInfo& ti, const ggml_tensor * t) {
        size_t elem_size = ggml_type_size(t->type);
        int dims[3] = {static_cast<int>(t->ne[2]),
                       static_cast<int>(t->ne[1]),
                       static_cast<int>(t->ne[0])};
        int strides[3] = {static_cast<int>(t->nb[2] / elem_size),
                          static_cast<int>(t->nb[1] / elem_size),
                          static_cast<int>(t->nb[0] / elem_size)};
        ti.set(t->type, dims, strides, t->data);
    }

    void set_a(const ggml_tensor * t, half* a_data) {
        size_t elem_size = ggml_type_size(t->type);
        int dims[3] = {1,
                       static_cast<int>(t->ne[1] * t->ne[2]),
                       static_cast<int>(t->ne[0])};
        int strides[3] = {static_cast<int>(t->nb[3] / elem_size),
                          static_cast<int>(t->nb[1] / elem_size),
                          static_cast<int>(t->nb[0] / elem_size)};
        A.set(GGML_TYPE_F16, dims, strides, a_data);
    }

    void set_c(const ggml_tensor * t, half* c_data, int topk) {
        GGML_ASSERT(ggml_is_contiguous(t));
        int dims[3] = {static_cast<int>(t->ne[1] * t->ne[2] / topk),
                       topk,
                       static_cast<int>(t->ne[0])};
        int strides[3] = {dims[1] * dims[2], dims[2], 1};
        C.set(GGML_TYPE_F16, dims, strides, c_data);
    }

    void set_b(const ggml_gptq_data * t) {
        GGML_ASSERT(t->bits == 4);
        GGML_ASSERT(t->scales_type == CUDA_R_16F);
        GGML_ASSERT(t->qzeros_type == CUDA_R_8U);
        // set B
        {
            int dims[3] = {t->E, t->M, t->K / 2};
            int strides[3] = {t->M * t->K / 2, t->K / 2, 1};
            B.set(GGML_TYPE_I8, dims, strides, t->qweight);
        }
        // set B_scale
        {
            int new_k = t->num_groups;
            int dims[3] = {t->E, t->M, new_k};
            int strides[3] = {(t->M) * new_k, new_k, 1};
            B_scale.set(GGML_TYPE_F16, dims, strides, t->scales);
        }
        // set B_zp
        {
            int new_m = t->M / 2;
            int new_k = t->num_groups;
            int dims[3] = {t->E, new_m, new_k};
            int strides[3] = {new_m * new_k, new_k, 1};
            B_zp.set(GGML_TYPE_I8, dims, strides, t->qzeros);
        }
    }

    void set_topk_ids(const ggml_tensor * t, int* topk_ids_data, int topk) {
        int dims[3] = {1,
                       static_cast<int>(t->ne[0] * t->ne[1] / topk),
                       topk};
        int strides[3] = {dims[1] * dims[2], dims[2], 1};
        topk_ids.set(t->type, dims, strides, topk_ids_data);
        topk_weights.set(GGML_TYPE_F32, dims, strides, topk_ids_data);
    }

    void set_align_info(int* sorted_ids_data, int sorted_id_size,
                        int* expert_ids_data, int expert_ids_size,
                        int* num_tokens_post_padded_data, int num_tokens_post_padded_size) {
        // sorted_ids
        {
            int dims[3] = {1, 1, sorted_id_size};
            int strides[3] = {dims[1] * dims[2], dims[2], 1};
            sorted_token_ids.set(GGML_TYPE_I32, dims, strides, sorted_ids_data);
        }
        // expert_ids
        {
            int dims[3] = {1, 1, expert_ids_size};
            int strides[3] = {dims[1] * dims[2], dims[2], 1};
            expert_ids.set(GGML_TYPE_I32, dims, strides, expert_ids_data);
        }
        // num_tokens_post_padded
        {
            int dims[3] = {1, 1, num_tokens_post_padded_size};
            int strides[3] = {dims[1] * dims[2], dims[2], 1};
            num_tokens_post_padded.set(GGML_TYPE_I32, dims, strides,
                                       num_tokens_post_padded_data);
        }
    }
};

static void ggml_cuda_dlblas_gemmex(
    ggml_backend_cuda_context & ctx,
    ggml_gptq_data & gptq_data,
    const void * src1_ptr,
    const ggml_type src1_type,
    const ggml_tensor * src0,
    const ggml_tensor * src1,
    void * dst_ptr,
    const ggml_type dst_type

){

    int m = (int)src0->ne[1];
    int k = (int)src0->ne[0];
    int n = (int)src1->ne[1];

    cublasOperation_t transA = CUBLAS_OP_N;
    cublasOperation_t transB = CUBLAS_OP_N;
    int lda = transA == CUBLAS_OP_T ? k : m;
    int ldb = (int)src1->ne[0];
    int ldc = m; // dst->ne[0]

    cudaDataType_t Atype = (gptq_data.bits == 4) ? CUDA_R_4U : CUDA_R_8U;
    cudaDataType_t Btype = ggml_type_to_cuda_dtype(src1_type);
    cudaDataType_t Ctype = ggml_type_to_cuda_dtype(dst_type);

    dlblasExtQuantParametersV2_t extParameters = {};
    extParameters.a_group_size_k = gptq_data.group_size;
    extParameters.a_group_size_m = 1;
    extParameters.a_zeropoints = gptq_data.qzeros;
    extParameters.a_zeropoints_type = gptq_data.qzeros_type;
    extParameters.a_scales = gptq_data.scales;
    extParameters.a_scales_type = gptq_data.scales_type;

    // For 4-bit quantization, dlblas with CUDA_R_4U should handle unpacking internally

    const float alpha_f = 1.0f;
    const float beta_f  = 0.0f;

    cublasHandle_t handle = ctx.cublas_handle();
    // !!!! make sure the stream is set.
    CUBLAS_CHECK(cublasSetStream(handle, ctx.stream()));
    CUBLAS_CHECK(dlblasGemmExV2(
        handle,
        transA, transB,
        m, n, k,
        &alpha_f,
        gptq_data.qweight, Atype, lda,
        src1_ptr,          Btype, ldb,
        &beta_f,
        dst_ptr,           Ctype, ldc,
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP,
        &extParameters
    ));
}

static void ggml_cuda_mul_mat_dlblas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int id = ggml_cuda_get_device();
    ggml_cuda_set_device(id);

    // DL: ensure weight tensor is available on this device
    ggml_gptq_data* gptq_weight = ggml_cuda_gptq_get_or_create_weight(ctx, src0, id);
    if (!gptq_weight) {
        printf("src0->name: %s, src0->type: %s\n", src0->name, ggml_type_name(src0->type));
        GGML_ABORT("DL: [%s] failed to prepare gptq_data", __FUNCTION__);
    }

    cudaStream_t stream = ctx.stream();

    // Prepare src1 pointer/type conversion (fp16 path preferred)
    ggml_cuda_pool_alloc<half> src1_fp16_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<half> dst_fp16_mem(ctx.pool(id));

    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        GGML_ASSERT(ggml_is_contiguous(src1));

        const size_t src1_size = ggml_nelements(src1);
        src1_fp16_mem.alloc(src1_size);

        to_fp16_cuda(src1->data, src1_fp16_mem.get(), src1_size, stream);
    }

    if (dst->type != GGML_TYPE_F16) {
        dst_fp16_mem.alloc(ggml_nelements(dst));
    }

    half* src1_fp16 = (src1->type != GGML_TYPE_F16) ?
            src1_fp16_mem.get() : (half *) src1->data;
    half* dst_fp16 = (dst->type != GGML_TYPE_F16) ?
            dst_fp16_mem.get() : (half *) dst->data;

    ggml_cuda_dlblas_gemmex(ctx, *gptq_weight, src1_fp16, GGML_TYPE_F16,
            src0, src1, dst_fp16, GGML_TYPE_F16);

    if (dst->type != GGML_TYPE_F16) {
        GGML_ASSERT(dst->type == GGML_TYPE_F32);
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        GGML_ASSERT(to_fp32_cuda != nullptr);
        GGML_ASSERT(ggml_is_contiguous(dst));

        const size_t dst_size = ggml_nelements(dst);
        to_fp32_cuda(dst_fp16_mem.get(), (float*)dst->data, dst_size, stream);
    }
}

[[noreturn]] static void ggml_cuda_op_mul_mat_dlblas(
    ggml_backend_cuda_context & ctx,        // CUDA context, stream, device, etc.
    const ggml_tensor * src0,               // weight tensor, original format
    const ggml_tensor * src1,               // input tensor
    ggml_tensor * dst,                      // output tensor
    const char * src0_dd_i,                 // src0 device data pointer, original format, quantized or unquantized weight.
    const float * src1_ddf_i,               // src1 device data pointer, fp32 input.
    const char * src1_ddq_i,                // src1 device data pointer, quantized input.
    float * dst_dd_i,                       // dst device data pointer, fp32 output.
    const int64_t row_low,                  // the start row of the input tensor.
    const int64_t row_high,                 // the end row of the input tensor.
    const int64_t src1_ncols,               // the number of columns of the input tensor. usually is batch_size.
    const int64_t src1_padded_row_size,     // the padded row size of the input tensor.
    cudaStream_t stream                     // CUDA stream
) {
    // TODO: implement this.
    GGML_UNUSED(ctx);
    GGML_UNUSED(src0);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);
    GGML_UNUSED(src0_dd_i);
    GGML_UNUSED(src1_ddf_i);
    GGML_UNUSED(src1_ddq_i);
    GGML_UNUSED(dst_dd_i);
    GGML_UNUSED(row_low);
    GGML_UNUSED(row_high);
    GGML_UNUSED(src1_ncols);
    GGML_UNUSED(src1_padded_row_size);
    GGML_UNUSED(stream);
    GGML_ABORT("DL: do not support mul_mat_dlblas for now.");
}

// ============================================================================
// Namespace Wrapper for Plugin API
// ============================================================================

namespace ggml_dl {

void quantize_and_store_from_cpu(int device_id, const ggml_tensor* tensor) {
    ggml_backend_cuda_gptq_quantize_and_store_from_cpu(device_id, tensor);
}

// Calculate the required memory size for GPTQ quantization
// This can be used to pre-allocate tensor memory with sufficient size
size_t calculate_gptq_required_size(int K, int M, int bits, int group_size) {
    return ggml_cuda_gptq_calculate_required_size(K, M, bits, group_size);
}

// Calculate the required memory size for MoE GPTQ quantization
// This can be used to pre-allocate tensor memory with sufficient size
size_t calculate_moe_gptq_required_size(int K, int M, int E, int bits, int group_size) {
    return ggml_cuda_moe_gptq_calculate_required_size(K, M, E, bits, group_size);
}

// dlblas Operations
void mul_mat_dlblas(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst) {

    ggml_cuda_mul_mat_dlblas(ctx, src0, src1, dst);
}

static cudaError_t ggml_cuda_cpy_tensor_2d(
    void * dst, const struct ggml_tensor * src, int64_t i3, int64_t i2, int64_t i1_low, int64_t i1_high, cudaStream_t stream) {

    const char * src_ptr = (const char *) src->data;
    char       * dst_ptr = (char       *) dst;

    const int64_t ne0 = src->ne[0];
    const int64_t nb0 = src->nb[0];
    const int64_t nb1 = src->nb[1];
    const int64_t nb2 = src->nb[2];
    const int64_t nb3 = src->nb[3];
    const enum ggml_type type = src->type;
    const int64_t ts = ggml_type_size(type);
    const int64_t bs = ggml_blck_size(type);
    const int64_t i1_diff = i1_high - i1_low;

    const char * x = src_ptr + i1_low*nb1 + i2*nb2 + i3*nb3;
    if (nb0 == ts && nb1 == ts*ne0/bs) {
        return cudaMemcpyAsync(dst_ptr, x, i1_diff*nb1, cudaMemcpyDeviceToDevice, stream);
    } else if (nb0 == ts) {
        return cudaMemcpy2DAsync(dst_ptr, ts*ne0/bs, x, nb1, ts*ne0/bs, i1_diff, cudaMemcpyDeviceToDevice, stream);
    } else {
        for (int64_t i1 = 0; i1 < i1_diff; i1++) {
            const void * rx = (const void *) ((const char *) x + i1*nb1);
            void * rd = (void *) (dst_ptr + i1*ts*ne0/bs);
            // pretend the row is a matrix with cols=1
            cudaError_t r = cudaMemcpy2DAsync(rd, ts/bs, rx, nb0, ts/bs, ne0, cudaMemcpyDeviceToDevice, stream);
            if (r != cudaSuccess) {
                return r;
            }
        }
        return cudaSuccess;
    }
}

void mul_mat_id_dlblas(
        ggml_backend_cuda_context& ctx,
        const ggml_tensor* src0,
        const ggml_tensor* src1,
        const ggml_tensor* ids,
        ggml_tensor* dst) {

    const int id = ggml_cuda_get_device();
    ggml_cuda_set_device(id);
    ggml_cuda_pool_alloc<int> topk_ids_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> workspace_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<half> src1_fp16_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<half> dst_fp16_mem(ctx.pool(id));

    ggml_cuda_pool_alloc<int> sorted_ids_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<int> expert_ids_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<int> num_tokend_post_pad_mem(ctx.pool(id));

    cudaStream_t stream = ctx.stream();

    ggml_gptq_data* gptq_weight = ggml_cuda_gptq_get_or_create_weight(ctx, src0, id);
    if (!gptq_weight) {
        printf("src0->name: %s, src0->type: %s\n", src0->name, ggml_type_name(src0->type));
        GGML_ABORT("DL: [%s] failed to prepare gptq_data", __FUNCTION__);
    }

    bool src1_on_host = ggml_backend_buffer_is_host(src1->buffer);
    bool dst_on_host = ggml_backend_buffer_is_host(dst->buffer);
    GGML_ASSERT(src1_on_host == false);
    GGML_ASSERT(dst_on_host == false);

    int topk = ids->ne[0] / src1->ne[1];
    cudnnHandle_t handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    // TODO: move to build_graph
    // topk_ids must be continuous in memory.
    int* topk_ids = topk_ids_mem.alloc(ids->ne[2] * ids->ne[1] * ids->ne[0]);
    CUDA_CHECK(ggml_cuda_cpy_tensor_2d(topk_ids, ids, 0, 0, 0, ids->ne[1], stream));

    // gate: input [tokens, 1, hidden_size]
    //   up: input [tokens, 1, hidden_size]
    // down: input [tokens, topk, hidden_size]

    if (src1->type != GGML_TYPE_F16) {
        const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
        GGML_ASSERT(to_fp16_cuda != nullptr);
        GGML_ASSERT(ggml_is_contiguous(src1));

        const size_t src1_size = ggml_nelements(src1);
        src1_fp16_mem.alloc(src1_size);

        to_fp16_cuda(src1->data, src1_fp16_mem.get(), src1_size, stream);
    }

    if (dst->type != GGML_TYPE_F16) {
        dst_fp16_mem.alloc(ggml_nelements(dst));
    }
    half* src1_fp16 = (src1->type != GGML_TYPE_F16) ?
            src1_fp16_mem.get() : (half*) src1->data;
    half* dst_fp16 = (dst->type != GGML_TYPE_F16) ?
            dst_fp16_mem.get() : (half*) dst->data;

    InvokeFusedMoeArgs args;
    args.set_a(src1, src1_fp16);
    args.set_b(gptq_weight);
    args.set_c(dst, dst_fp16, topk);
    args.set_topk_ids(ids, topk_ids, topk);

    // moe_align_block_size, port from vllm

    int total_topk_ids = ggml_nelements(ids);
    int num_experts = gptq_weight->E;
    int avg_tokens_per_expert = total_topk_ids / num_experts;

    int tokens = ids->ne[1];
    int block_size_m = 64;
    if (avg_tokens_per_expert < 6) {
        block_size_m = ::min(16, tokens);
    } else if (tokens <= 20) {
        block_size_m = 16;
    } else if (tokens <= 40) {
        block_size_m = 32;
    }
    int block_size_n = 0;
    int block_size_k = gptq_weight->group_size;

    if (avg_tokens_per_expert > 10) {
        auto max_num_m_blocks = ::min(total_topk_ids,
                                     (total_topk_ids - num_experts) / block_size_m + num_experts);
        auto max_num_tokens_padded = max_num_m_blocks * block_size_m;

        int* sorted_ids = sorted_ids_mem.alloc(max_num_tokens_padded);
        int* expert_ids = expert_ids_mem.alloc(max_num_m_blocks + num_experts + 1);
        int* cumsum_buffer = expert_ids + max_num_m_blocks;
        int* num_tokens_post_pad = num_tokend_post_pad_mem.alloc(1);
        // sorted_ids.fill_(total_topk_ids)
        // expert_ids.fill_(0)
        ggml_cuda_moe_init_data<<<
                (max_num_tokens_padded + max_num_m_blocks + num_experts + 1 + 1024 - 1) / 1024,
                1024, 0, stream>>>(
            sorted_ids,
            expert_ids,
            total_topk_ids,
            max_num_tokens_padded,
            max_num_m_blocks + num_experts + 1);

        int num_warps = (num_experts + 32 - 1) / 32;
        int padded_num_experts = num_warps * 32;
        int experts_per_warp = 32;
        int threads = 1024;
        int shared_mem_size = num_warps * experts_per_warp * sizeof(int);

        ggml_cuda_moe_align_kernel<<<1, threads, shared_mem_size, stream>>>(
            topk_ids,
            sorted_ids,
            expert_ids,
            num_tokens_post_pad,
            num_experts, padded_num_experts, experts_per_warp,
            block_size_m,
            total_topk_ids,
            cumsum_buffer);

        ggml_cuda_moe_count_and_sort_expert_tokens_kernel<<<
                (total_topk_ids + 256 - 1) / 256,
                256, 0, stream>>>(
            topk_ids,
            sorted_ids,
            cumsum_buffer,
            total_topk_ids);

        args.set_align_info(sorted_ids, max_num_tokens_padded,
                            expert_ids, max_num_m_blocks,
                            num_tokens_post_pad, 1);
    }

    size_t mem_size = 0;
    size_t block_shape[2] = {1, static_cast<size_t>(gptq_weight->group_size)};
    CUDNN_CHECK(cudnnGetInvokeFusedMoeKernelWorkspaceSize(
        handle,
        args.A.desc, args.A.ptr,
        args.B.desc, args.B.ptr,
        args.C.desc, args.C.ptr,
        args.B_scale.desc, args.B_scale.ptr,
        args.B_zp.desc, args.B_zp.ptr,
        args.topk_weights.desc, args.topk_weights.ptr,
        args.topk_ids.desc, args.topk_ids.ptr,
        args.sorted_token_ids.desc, args.sorted_token_ids.ptr,
        args.expert_ids.desc, args.expert_ids.ptr,
        args.num_tokens_post_padded.desc, args.num_tokens_post_padded.ptr,
        false,  // mul_routed_weight
        topk,
        block_size_m,     // block_size_m, not used
        block_size_n,     // block_size_n, not used
        block_size_k,     // block_size_k, not useds
        true,   // has_zp
        true,   // use_int4_w4a16, int4 gptq
        false,  // use_int8_w8a16, int8 blockwise
        false,  // use_fp8_w8a8, fp8 blockwise
        block_shape,   // block_shape
        &mem_size));

    void* workspace = nullptr;
    if (mem_size > 0) {
        workspace = workspace_mem.alloc(mem_size);
    }

    CUDNN_CHECK(cudnnInvokeFusedMoeKernel(
        handle,
        args.A.desc, args.A.ptr,
        args.B.desc, args.B.ptr,
        args.C.desc, args.C.ptr,
        args.B_scale.desc, args.B_scale.ptr,
        args.B_zp.desc, args.B_zp.ptr,
        args.topk_weights.desc, args.topk_weights.ptr,
        args.topk_ids.desc, args.topk_ids.ptr,
        args.sorted_token_ids.desc, args.sorted_token_ids.ptr,
        args.expert_ids.desc, args.expert_ids.ptr,
        args.num_tokens_post_padded.desc, args.num_tokens_post_padded.ptr,
        false,  // mul_routed_weight
        topk,
        block_size_m,     // block_size_m, not used
        block_size_n,     // block_size_n, not used
        block_size_k,     // block_size_k, not useds
        true,   // has_zp
        true,   // use_int4_w4a16, int4 gptq
        false,  // use_int8_w8a16, int8 blockwise
        false,  // use_fp8_w8a8, fp8 blockwise
        block_shape,   // block_shape
        workspace,
        mem_size));

    if (dst->type != GGML_TYPE_F16) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(GGML_TYPE_F16);
        GGML_ASSERT(to_fp32_cuda != nullptr);
        GGML_ASSERT(ggml_is_contiguous(dst));

        const size_t dst_size = ggml_nelements(dst);
        to_fp32_cuda(dst_fp16_mem.get(), (float*)dst->data, dst_size, stream);
    }
}

bool is_dlblas_available(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* dst,
    bool split) {

    GGML_UNUSED(dst);
    // do not support split now.
    if (split) return false;

    // Check if we're in force test mode first
    const char* env_force_dlblas_test = getenv("GGML_FORCE_DLBLAS_TEST");
    bool force_test_mode = (env_force_dlblas_test && env_force_dlblas_test[0] == '1');

    // Only check GGML_FORCE_NO_DLBLAS if we're not in force test mode
    if (!force_test_mode) {
        const char* env_force_no_dlblas = getenv("GGML_FORCE_NO_DLBLAS");
        if (env_force_no_dlblas && env_force_no_dlblas[0] == '1') {
            return false;
        }
    }

    // check single batch (bugid: 16276)
    bool single_batch = src0->ne[2] * src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] == 1;
    if (!single_batch) {
        GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_PATH_OVERRIDE] Not single batch, use original path\n");
        return false;
    }

    if (force_test_mode) {
        GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_TEST_MODE] Force enabling DLBLAS for testing\n");
        return true;
    }

    const int device_id = ggml_cuda_get_device();
    bool has_weight = ggml_cuda_gptq_has_weight(src0, device_id);
    bool has_any_weight =  ggml_cuda_gptq_has_any_weight(src0);
    if (GGML_IS_TEST) { // if is test, quantize and store the weight anyway.
        ggml_cuda_gptq_quantize_and_store(ctx, src0->data, src0, GGML_QUANT_BITS);
        return true;
    }
    // GGML_DL_MULMAT_DEBUG_PRINT("has_weight: %d\n", has_weight);
    // GGML_DL_MULMAT_DEBUG_PRINT("has_any_weight: %d\n", has_any_weight);
    return has_weight || has_any_weight;
}

} // namespace ggml_dl

#endif // GGML_USE_DLCU
