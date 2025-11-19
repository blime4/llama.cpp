#include <csignal>
#ifdef GGML_USE_DLCU
#include "dl-mulmat.cuh"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"
#include "../ggml-cuda/common.cuh"
#include "../ggml-cuda/convert.cuh"
#include "dlblas_ext.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>

#include "vector_types.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <vector>

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

__global__ void ggml_cuda_gptq_quantize_8_bit_fp16(
    const half* __restrict__ w,    // [M, K], half type
    uint8_t* __restrict__ qweight,
    half* __restrict__ qzeros,
    half* __restrict__ scales,
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

    const int num_groups_row = (K + group_size - 1) / group_size;
    int scale_idx = row * num_groups_row + group;
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

        qweight[row * K + col] = (uint8_t)q;
    }
}

__global__ void ggml_cuda_gptq_quantize_4_bit_fp16(
    const half* __restrict__ w,    // [M, K], half type
    uint8_t* __restrict__ qweight,
    half* __restrict__ qzeros,
    half* __restrict__ scales,
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
    const float qmax = 15.0f;  // 4-bit quantization range: 0-15

    if (max_val == min_val) {
        if (min_val == 0.0f) {
            scale = 1.0f;
            zero_point = 0.0f;
        } else {
            scale = std::abs(min_val) / 15.0f;
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

    const int num_groups_row = (K + group_size - 1) / group_size;
    int scale_idx = row * num_groups_row + group;
    scales[scale_idx] = __float2half(scale);
    qzeros[scale_idx] = __float2half(zero_point);

    // Process elements in pairs to avoid race conditions in packing
    for (int k = 0; k < group_size && group * group_size + k < K; k += 2) {
        // Process two consecutive elements at once
        for (int i = 0; i < 2 && group * group_size + k + i < K; ++i) {
            int col = group * group_size + k + i;
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
            if (q > 15) q = 15;  // Clamp to 4-bit range

            // Pack two 4-bit values into one byte
            // Similar to gguf_linear_quantize_weights: (w_q[:,1::2] << 4) | w_q[:, ::2]
            int packed_col = col / 2;

            if (i == 0) {
                // First element: store in low 4 bits
                qweight[row * (K / 2) + packed_col] = (uint8_t)q;
            } else {
                // Second element: store in high 4 bits
                qweight[row * (K / 2) + packed_col] |= (uint8_t)(q << 4);
            }
        }
    }
}

// ============================================================================
// GPTQ Data Structure
// ============================================================================

struct ggml_gptq_data {
    void * qweight;
    void * qzeros;
    void * scales;
    int bits;
    int group_size;
    cudaDataType_t scales_type;
    cudaDataType_t qzeros_type;
    int num_groups;
    int M;
    int K;
    int qweight_size;
    int qzeros_size;
    int scales_size;
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

// ============================================================================
// Memory Management Functions
// ============================================================================

// Enhanced GPTQ Cache Key Implementation
struct ggml_gptq_cache_key {
    const ggml_tensor* base;
    int device;
    int K;
    int M;
    ggml_type type;
    size_t data_size;

    bool operator==(const ggml_gptq_cache_key & other) const noexcept {
        return base == other.base &&
               device == other.device &&
               K == other.K &&
               M == other.M &&
               type == other.type &&
               data_size == other.data_size;
    }
};

struct ggml_gptq_cache_key_hash {
    size_t operator()(const ggml_gptq_cache_key & key) const noexcept {
        size_t h1 = std::hash<const void*>{}(key.base);
        size_t h2 = std::hash<int>{}(key.device);
        size_t h3 = std::hash<int>{}(key.K);
        size_t h4 = std::hash<int>{}(key.M);
        size_t h5 = std::hash<int>{}(static_cast<int>(key.type));
        size_t h6 = std::hash<size_t>{}(key.data_size);

        size_t result = h1;
        result = result * 31 + h2;
        result = result * 31 + h3;
        result = result * 31 + h4;
        result = result * 31 + h5;
        result = result * 31 + h6;
        return result;
    }
};

// DL: used to store the gptq-format weight tensor per (tensor, device).
static std::unordered_map<ggml_gptq_cache_key, ggml_gptq_data*, ggml_gptq_cache_key_hash> g_gptq_weights_map;
static std::mutex g_gptq_weights_mutex;

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
        static_cast<int>(base->ne[0]),
        static_cast<int>(base->ne[1]),
        base->type,
        ggml_nbytes(base)
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
        if (old_data->qweight) cudaFree(old_data->qweight);
        if (old_data->qzeros) cudaFree(old_data->qzeros);
        if (old_data->scales) cudaFree(old_data->scales);
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
    for (const auto & entry : g_gptq_weights_map) {
        if (entry.first.base == base) {
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
                               tensor->name ? tensor->name : "(null)", src_device, dst_device);

    ggml_gptq_data* cloned = new ggml_gptq_data();
    cloned->bits         = src_data->bits;
    cloned->group_size   = src_data->group_size;
    cloned->scales_type  = src_data->scales_type;
    cloned->qzeros_type  = src_data->qzeros_type;
    cloned->num_groups   = src_data->num_groups;
    cloned->M            = src_data->M;
    cloned->K            = src_data->K;
    cloned->qweight_size = src_data->qweight_size;
    cloned->qzeros_size  = src_data->qzeros_size;
    cloned->scales_size  = src_data->scales_size;

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
    auto ensure_ptr = [&](void** ptr, size_t size, const char* label) {
        if (*ptr == nullptr || size == 0) {
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
                                       label, tensor->name ? tensor->name : "(null)", device_id);
        }
    };

    ensure_ptr(&data->qweight, data->qweight_size, "qweight");
    ensure_ptr(&data->qzeros,  data->qzeros_size,  "qzeros");
    ensure_ptr(&data->scales,  data->scales_size,  "scales");
}

static void ggml_cuda_gptq_clear_weights() {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);
    for (auto& pair : g_gptq_weights_map) {
        ggml_gptq_data* data = pair.second;
        if (data->qweight) cudaFree(data->qweight);
        if (data->qzeros) cudaFree(data->qzeros);
        if (data->scales) cudaFree(data->scales);
        delete data;
    }
    g_gptq_weights_map.clear();
}


// ============================================================================
// GPTQ Quantization Implementation
// ============================================================================

static void ggml_cuda_gptq_quantize_and_store(ggml_backend_cuda_context & ctx, const void * src0_ptr, const ggml_tensor * src0, int bits = 8) {

    int K = src0->ne[0];
    int M = src0->ne[1];
    int group_size = std::min(GGML_CUDA_GPTQ_GROUP_SIZE, K); // Ensure group_size <= K to avoid issues with small K

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

    GGML_ASSERT((bits == 4 || bits == 8) && "Only 4-bit and 8-bit quantization supported");

    int num_groups = (K + group_size - 1) / group_size;

    const int id = ctx.device;
    ggml_cuda_set_device(id);
    GGML_DL_MULMAT_DEBUG_PRINT("quantizing tensor '%s' on device %d with %d-bit GPTQ\n",
                               src0->name ? src0->name : "(null)", id, bits);

    ggml_gptq_data* gptq_data = new ggml_gptq_data;

    // For 4-bit: each byte stores 2 values, so we need M * (K/2) bytes for qweight
    size_t qweight_size = (bits == 4) ? (size_t)M * (size_t)(K / 2) : (size_t)M * (size_t)K;
    CUDA_CHECK(cudaMalloc(&gptq_data->qweight, qweight_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&gptq_data->qzeros, M * num_groups * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&gptq_data->scales, M * num_groups * sizeof(half)));

    // nowaday, only support float.
    dim3 blockDim(32, 8);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (num_groups + blockDim.y - 1) / blockDim.y);
    cudaStream_t stream = ctx.stream();

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

    // kernel launch directly to persistent memory (eliminating one cudaMemcpyDeviceToDevice)
    // TODO : support bf16 later.
    if (bits == 8) {
        ggml_cuda_gptq_quantize_8_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
            src0_ptr_as_fp16, (uint8_t*)gptq_data->qweight, (half*)gptq_data->qzeros, (half*)gptq_data->scales,
            K, M, group_size);
    } else if (bits == 4) {
        ggml_cuda_gptq_quantize_4_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
            src0_ptr_as_fp16, (uint8_t*)gptq_data->qweight, (half*)gptq_data->qzeros, (half*)gptq_data->scales,
            K, M, group_size);
    }

    // check the CUDA error after kernel launch
    cudaError_t r = cudaGetLastError();
    if (r != cudaSuccess) {
        GGML_LOG("gptq_quantize_%d_bit error: %s\n", bits, cudaGetErrorString(r));
    }

    gptq_data->group_size = group_size;
    gptq_data->bits = bits;
    gptq_data->scales_type = ggml_ptr_elem_size_to_cuda_dtype(sizeof(half));
    gptq_data->qzeros_type  = ggml_ptr_elem_size_to_cuda_dtype(sizeof(half));
    gptq_data->num_groups = num_groups;
    gptq_data->M = M;
    gptq_data->K = K;
    gptq_data->qweight_size = (int)qweight_size * sizeof(uint8_t);
    gptq_data->qzeros_size = M * num_groups * sizeof(half);
    gptq_data->scales_size = M * num_groups * sizeof(half);
    // DL: store it to the mapping table
    ggml_cuda_gptq_store_weight(src0, gptq_data, id);
}

static ggml_gptq_data* ggml_cuda_gptq_get_or_create_weight(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor* tensor,
    int device_id) {

    ggml_gptq_data* gptq_weight = ggml_cuda_gptq_get_weight(tensor, device_id);
    if (gptq_weight) {
        return gptq_weight;
    }

    int existing_device = -1;
    ggml_gptq_data* existing_weight = ggml_cuda_gptq_get_weight_any(tensor, &existing_device);
    if (existing_weight) {
        gptq_weight = ggml_cuda_gptq_clone_weight(tensor, existing_weight, existing_device, device_id);
    } else {
        const int bits = GGML_QUANT_BITS;
        GGML_DL_MULMAT_DEBUG_PRINT("device %d missing GPTQ weight for '%s', re-quantizing (%d-bit)\n",
                                   device_id, tensor->name ? tensor->name : "(null)", bits);
        ggml_cuda_gptq_quantize_and_store(ctx, tensor->data, tensor, bits);
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
        ggml_cuda_gptq_quantize_and_store(ctx, base->data, base, bits);
    }

    GGML_ASSERT(ggml_cuda_gptq_has_weight(base, device_id) && "ggml_cuda_gptq_quantize_and_store unified the base and view quantization, so we can just return here.");
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

    cublasOperation_t transA = CUBLAS_OP_T;
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
    // Helpful debug signal so we can tell when the DLBLAS path is taken.
    GGML_DL_MULMAT_DEBUG_PRINT("ggml_cuda_mul_mat_dlblas: using DLBLAS backend (dst=%s, src0=%s, src1=%s)\n",
                               dst->name, src0->name, src1->name);

    const int id = ggml_cuda_get_device();
    ggml_cuda_set_device(id);
    {
        // In unit tests. TODO: Refactor : Others that do not go through llama_model::load_tensors are theoretically needed
        static const bool is_test_mode = getenv("GGML_TEST_MODE") != nullptr;
        if (is_test_mode) {
            GGML_DL_MULMAT_DEBUG_PRINT("GGML_TEST_MODE=1, clearing weights and re-quantizing\n");
            ggml_cuda_gptq_clear_weights();
            const ggml_tensor * base = ggml_cuda_get_base_tensor(src0);

            int bits = GGML_QUANT_BITS;

            // quantize the base tensor anyway.
            ggml_cuda_gptq_quantize_and_store(ctx, base->data, base, bits);
            GGML_ASSERT(ggml_cuda_gptq_has_weight(base, id) && "DL: [GGML_TEST_MODE] ggml_cuda_gptq_quantize_and_store failed");
        }
    }

    // DL: ensure weight tensor is available on this device
    ggml_gptq_data* gptq_weight = ggml_cuda_gptq_get_or_create_weight(ctx, src0, id);
    if (!gptq_weight) {
        GGML_DL_MULMAT_DEBUG_PRINT("src0->name: %s, src0->type: %s\n", src0->name, ggml_type_name(src0->type));
        GGML_ABORT("DL: [%s] failed to prepare gptq_data", __FUNCTION__);
    }

    cudaStream_t stream = ctx.stream();
    const int64_t ne10 = src1->ne[0];
    const int64_t ne11 = src1->ne[1];
    const size_t ne_dst = (size_t) ggml_nelements(dst);

    // Stage src1 on device if needed
    const bool src1_on_host = ggml_backend_buffer_is_host(src1->buffer);
    ggml_cuda_pool_alloc<char> src1_device_storage(ctx.pool(id));
    const void * src1_device_ptr = nullptr;
    if (src1_on_host) {
        size_t bytes = ggml_nbytes(src1);
        char * staging = src1_device_storage.alloc(bytes);
        CUDA_CHECK(cudaMemcpyAsync(staging, src1->data, bytes, cudaMemcpyHostToDevice, stream));
        src1_device_ptr = staging;
    } else {
        src1_device_ptr = src1->data;
    }

    // Prepare src1 pointer/type conversion (fp16 path preferred)
    const void * src1_ptr = nullptr;
    ggml_type src1_type = src1->type;
    ggml_cuda_pool_alloc<half> src1_as_f16(ctx.pool(id));
    ggml_cuda_pool_alloc<nv_bfloat16> src1_as_bf16(ctx.pool(id));
    if (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16) {
        if (src1->type != GGML_TYPE_F16) {
            const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(src1->type);
            GGML_ASSERT(to_fp16_cuda != nullptr);
            size_t ne = ne10*ne11;
            src1_as_f16.alloc(ne);
            to_fp16_cuda(src1_device_ptr, src1_as_f16.get(), ne, stream);
            src1_ptr = src1_as_f16.get();
        } else {
            src1_ptr = src1_device_ptr;
        }
        src1_type = GGML_TYPE_F16;
    } else {
        GGML_ABORT("DL: do not support other types for now. src1->type: %s", ggml_type_name(src1->type));
        const to_bf16_cuda_t to_bf16_cuda = ggml_get_to_bf16_cuda(src1->type);
        GGML_ASSERT(to_bf16_cuda != nullptr);
        size_t ne = ne10*ne11;
        src1_as_bf16.alloc(ne);
        to_bf16_cuda(src1_device_ptr, src1_as_bf16.get(), ne, stream);
        src1_ptr = src1_as_bf16.get();
        src1_type = GGML_TYPE_BF16;
    }

    CUDA_CHECK(cudaGetLastError());

    // Allocate destination buffer on device (compute in FP32)
    const bool dst_on_host = ggml_backend_buffer_is_host(dst->buffer);
    ggml_cuda_pool_alloc<float> dst_fp32_storage(ctx.pool(id));
    float * dst_device_fp32 = nullptr;

    if (!dst_on_host && dst->type == GGML_TYPE_F32) {
        dst_device_fp32 = (float *) dst->data;
    } else {
        dst_device_fp32 = dst_fp32_storage.alloc(ne_dst);
    }

    ggml_cuda_dlblas_gemmex(ctx, *gptq_weight, src1_ptr, src1_type, src0, src1, dst_device_fp32, GGML_TYPE_F32);
    CUDA_CHECK(cudaGetLastError());

    if (dst->type == GGML_TYPE_F32) {
        size_t bytes = ne_dst * sizeof(float);
        if (dst_on_host) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, dst_device_fp32, bytes, cudaMemcpyDeviceToHost, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, dst_device_fp32, bytes, cudaMemcpyDeviceToDevice, stream));
        }
    } else if (dst->type == GGML_TYPE_F16) {
        ggml_cuda_pool_alloc<half> dst_fp16(ctx.pool(id));
        half * dst_device_fp16 = dst_fp16.alloc(ne_dst);
        const to_fp16_cuda_t to_fp16 = ggml_get_to_fp16_cuda(GGML_TYPE_F32);
        GGML_ASSERT(to_fp16 != nullptr);
        to_fp16(dst_device_fp32, dst_device_fp16, ne_dst, stream);
        size_t bytes = ne_dst * sizeof(half);
        if (dst_on_host) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, dst_device_fp16, bytes, cudaMemcpyDeviceToHost, stream));
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, dst_device_fp16, bytes, cudaMemcpyDeviceToDevice, stream));
        }
    } else {
        GGML_ABORT("DL: [%s] unsupported dst type %s", __FUNCTION__, ggml_type_name(dst->type));
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

// dlblas Operations
void mul_mat_dlblas(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    ggml_tensor* dst) {

    ggml_cuda_mul_mat_dlblas(ctx, src0, src1, dst);
}

bool is_dlblas_available_simple(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* dst,
    bool split) {

    GGML_UNUSED(ctx);
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

    // just for debug what some go dlblas is so slow. remove this later.
    const char* only_one_layer_go_to_dlblas = getenv("GGML_ONLY_ONE_LAYER_GO_TO_DLBLAS");
    if (only_one_layer_go_to_dlblas && only_one_layer_go_to_dlblas[0] == '1') {
        if (strstr(src0->name, "blk.0.attn_k.weight") != nullptr) {
            printf("for debug DL: [%s] %s, go dlblas path\n", __FUNCTION__, src0->name);
            return true;
        } else {
            return false;
        }
    }

    // check single batch (bugid: 16276)
    bool single_batch = src0->ne[2] * src0->ne[3] == 1 && src1->ne[2] * src1->ne[3] == 1;
    if (!single_batch) {
        GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_PATH_OVERRIDE] Not single batch, use original path\n");
        // GGML_DL_MULMAT_DEBUG_PRINT("src0: %8d %8d %8d %8d\n", src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3]);
        // GGML_DL_MULMAT_DEBUG_PRINT("      %8d %8d %8d %8d\n", src0->nb[0], src0->nb[1], src0->nb[2], src0->nb[3]);
        // GGML_DL_MULMAT_DEBUG_PRINT("src1: %8d %8d %8d %8d\n", src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3]);
        // GGML_DL_MULMAT_DEBUG_PRINT("      %8d %8d %8d %8d\n", src1->nb[0], src1->nb[1], src1->nb[2], src1->nb[3]);
        // GGML_DL_MULMAT_DEBUG_PRINT("src0 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src0), ggml_is_transposed(src0), ggml_type_name(src0->type), src0->name);
        // GGML_DL_MULMAT_DEBUG_PRINT("src1 is contiguous %d, transposed %d, type = %s, name = %s\n", ggml_is_contiguous(src1), ggml_is_transposed(src1), ggml_type_name(src1->type), src1->name);
        return false;
    }

    if (force_test_mode) {
        GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_TEST_MODE] Force enabling DLBLAS for testing\n");
        return true;
    }

    const int device_id = ggml_cuda_get_device();
    bool has_weight = ggml_cuda_gptq_has_weight(src0, device_id);
    bool has_any_weight =  ggml_cuda_gptq_has_any_weight(src0);
    // GGML_DL_MULMAT_DEBUG_PRINT("has_weight: %d\n", has_weight);
    // GGML_DL_MULMAT_DEBUG_PRINT("has_any_weight: %d\n", has_any_weight);
    return has_weight || has_any_weight;
}

bool should_use_dlblas_path(
    bool use_mul_mat_vec,
    bool use_mul_mat_vec_q) {

    const char* env_force_dlblas_test = getenv("GGML_FORCE_DLBLAS_TEST");
    if (env_force_dlblas_test && env_force_dlblas_test[0] == '1') {
        GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_PATH_OVERRIDE] Force using DLBLAS in test mode\n");
        return true;
    }

    const char* env_consistent = getenv("GGML_DLBLAS_CONSISTENT");
    bool use_consistent = (env_consistent && env_consistent[0] == '1');

    if (use_consistent) {
        if (use_mul_mat_vec || use_mul_mat_vec_q) {
            GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_PATH_OVERRIDE] Use dlblas for consistency\n");
        }
        return true;
    }

    // bugid: 15564 - if not satisfy vec condition, use dlblas
    if (!(use_mul_mat_vec || use_mul_mat_vec_q)) {
        GGML_DL_MULMAT_DEBUG_PRINT("[DLBLAS_PATH_OVERRIDE] Not satisfy vec condition, use dlblas path\n");
        return true;
    }

    return false;
}

void cleanup() {
    ggml_cuda_gptq_clear_weights();
}

} // namespace ggml_dl

#endif // GGML_USE_DLCU
