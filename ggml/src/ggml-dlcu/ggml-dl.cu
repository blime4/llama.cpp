/**
* @file ggml-dl.cu
* @brief DengLin (DL) CUDA extensions implementation
 */

#ifdef GGML_USE_DLCU

#include "ggml-dl.cuh"
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
// Forward declarations
// ============================================================================

static const ggml_tensor* ggml_dl_get_base_tensor(const ggml_tensor* t);


// ============================================================================
// Global Configuration Variables
// ============================================================================

static bool DEVIT = getenv("DEVIT") != nullptr;
static int GGML_CUDA_GPTQ_GROUP_SIZE = []() {
    const char* env = getenv("GGML_CUDA_GPTQ_GROUP_SIZE");
    if (env) {
        int val = atoi(env);
        if (val > 0) return val;
    }
    return 32;
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

struct ggml_gptq_cache_key {
    const ggml_tensor* base;
    int device;

    bool operator==(const ggml_gptq_cache_key & other) const noexcept {
        return base == other.base && device == other.device;
    }
};

struct ggml_gptq_cache_key_hash {
    size_t operator()(const ggml_gptq_cache_key & key) const noexcept {
        const size_t h1 = std::hash<const void*>{}(key.base);
        const size_t h2 = std::hash<int>{}(key.device);
        return h1 ^ (h2 + 0x9e3779b97f4a7c15ull + (h1 << 6) + (h1 >> 2));
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

// DL: used to store the gptq-format weight tensor.
static void ggml_cuda_gptq_store_weight(const ggml_tensor* tensor, ggml_gptq_data* gptq_data, int device_id) {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);

    // always use base tensor as the key
    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);

    const ggml_gptq_cache_key key { base, device_id };

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
    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);

    const ggml_gptq_cache_key key { base, device_id };

    auto it = g_gptq_weights_map.find(key);
    return (it != g_gptq_weights_map.end()) ? it->second : nullptr;
}

static bool ggml_cuda_gptq_has_weight(const ggml_tensor* tensor, int device_id) {
    std::lock_guard<std::mutex> lock(g_gptq_weights_mutex);
    const ggml_tensor * base = ggml_cuda_get_base_tensor(tensor);

    const ggml_gptq_cache_key key { base, device_id };

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

static int ggml_cuda_gptq_default_bits() {
    const char * env = getenv("GGML_QUANT_BITS");
    if (env != nullptr) {
        int bits = atoi(env);
        if (bits == 4 || bits == 8) {
            return bits;
        }
    }
    return 4;
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
    fprintf(stderr, "[ggml-dlcu] cloning GPTQ weight '%s' from device %d to device %d\n",
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
            fprintf(stderr, "[ggml-dlcu] migrated GPTQ buffer %s for tensor '%s' to device %d\n",
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
    // 1. get the weight tensor
    // 2. quantize the weight tensor to gptq-format

    // 1. get the weight tensor
    // GGML_LOG_DEBUG("DL: [%s] quantizing weight tensor %s to %d-bit\n", __FUNCTION__, src0->name, bits);
    void* data = src0->data;
    int64_t ne00 = src0->ne[0]; // ne [row, col, batch, shard]
    int64_t ne01 = src0->ne[1];
    int64_t ne02 = src0->ne[2];
    int64_t ne03 = src0->ne[3];
    int64_t nb00 = src0->nb[0];
    int64_t nb01 = src0->nb[1];
    int64_t nb02 = src0->nb[2];
    int64_t nb03 = src0->nb[3];


    // 2. quantize the weight tensor to gptq-format
    int K = src0->ne[0];
    int M = src0->ne[1];
    int group_size = std::min(GGML_CUDA_GPTQ_GROUP_SIZE, K); // Ensure group_size <= K to avoid issues with small K

    // Support both 4-bit and 8-bit quantization
    GGML_ASSERT((bits == 4 || bits == 8) && "Only 4-bit and 8-bit quantization supported");

    int num_groups = (K + group_size - 1) / group_size;

    const int id = ctx.device;
    ggml_cuda_set_device(id);
    fprintf(stderr, "[ggml-dlcu] quantizing tensor '%s' on device %d with %d-bit GPTQ\n",
            src0->name ? src0->name : "(null)", id, bits);

    // 3. create gptq_data and allocate persistent GPU memory directly
    ggml_gptq_data* gptq_data = new ggml_gptq_data;

    // 4. allocate persistent GPU memory directly (avoiding temporary allocations)
    // For 4-bit: each byte stores 2 values, so we need M * (K/2) bytes for qweight
    size_t qweight_size = (bits == 4) ? (size_t)M * (size_t)(K / 2) : (size_t)M * (size_t)K;
    CUDA_CHECK(cudaMalloc(&gptq_data->qweight, qweight_size * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&gptq_data->qzeros, M * num_groups * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&gptq_data->scales, M * num_groups * sizeof(half)));

    // nowaday, only support float.
    dim3 blockDim(32, 8);
    dim3 gridDim((M + blockDim.x - 1) / blockDim.x, (num_groups + blockDim.y - 1) / blockDim.y);
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

    if(DEVIT){
        CUDA_CHECK(cudaDeviceSynchronize());
        std::vector<half> src0_ptr_as_fp16_cpu(10);
        cudaMemcpy(src0_ptr_as_fp16_cpu.data(), src0_ptr_as_fp16, 10*sizeof(half), cudaMemcpyDeviceToHost);
        CUDA_CHECK(cudaDeviceSynchronize());
        for(int i = 0; i < 10; i++){
            printf("[%s] src0_ptr_as_fp16_cpu[%d]: %f\n", __FUNCTION__, i, __half2float(src0_ptr_as_fp16_cpu[i]));
        }
    }

    // 5. kernel launch directly to persistent memory (eliminating one cudaMemcpyDeviceToDevice)
    // TODO : support bf16 later.
    CUDA_CHECK(cudaDeviceSynchronize());

    if (bits == 8) {
        ggml_cuda_gptq_quantize_8_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
            src0_ptr_as_fp16, (uint8_t*)gptq_data->qweight, (half*)gptq_data->qzeros, (half*)gptq_data->scales,
            K, M, group_size);
    } else if (bits == 4) {
        ggml_cuda_gptq_quantize_4_bit_fp16<<<gridDim, blockDim, 0, stream>>>(
            src0_ptr_as_fp16, (uint8_t*)gptq_data->qweight, (half*)gptq_data->qzeros, (half*)gptq_data->scales,
            K, M, group_size);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    // 6. check the CUDA error after kernel launch
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

    ggml_gptq_data* gptq_weight = ggml_cuda_gptq_get_weight(tensor, device_id);
    if (gptq_weight) {
        return gptq_weight;
    }

    int existing_device = -1;
    ggml_gptq_data* existing_weight = ggml_cuda_gptq_get_weight_any(tensor, &existing_device);
    if (existing_weight) {
        gptq_weight = ggml_cuda_gptq_clone_weight(tensor, existing_weight, existing_device, device_id);
    } else {
        const int bits = ggml_cuda_gptq_default_bits();
        fprintf(stderr, "[ggml-dlcu] device %d missing GPTQ weight for '%s', re-quantizing (%d-bit)\n",
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

    const char * GGML_QUANT_BITS = getenv("GGML_QUANT_BITS");
    int bits = GGML_QUANT_BITS ? atoi(GGML_QUANT_BITS) : 4; // default to 4-bit

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
    // In unit tests. TODO: Refactor : Others that do not go through llama_model::load_tensors are theoretically needed
    static const bool is_test_mode = getenv("GGML_TEST_MODE") != nullptr;
    const int id = ggml_cuda_get_device();
    ggml_cuda_set_device(id);
    if (is_test_mode) {
        // GGML_LOG_DEBUG("DL: [%s] GGML_TEST_MODE=1\n", __FUNCTION__);
        const ggml_tensor * base = ggml_cuda_get_base_tensor(src0);

        const char * GGML_QUANT_BITS = getenv("GGML_QUANT_BITS");
        int bits = GGML_QUANT_BITS ? atoi(GGML_QUANT_BITS) : 4; // default to 4-bit

        // quantize the base tensor anyway.
        ggml_cuda_gptq_quantize_and_store(ctx, base->data, base, bits);
        GGML_ASSERT(ggml_cuda_gptq_has_weight(base, id) && "DL: [GGML_TEST_MODE] ggml_cuda_gptq_quantize_and_store failed");
    }

    // DL: ensure weight tensor is available on this device
    ggml_gptq_data* gptq_weight = ggml_cuda_gptq_get_or_create_weight(ctx, src0, id);
    if (!gptq_weight) {
        printf("src0->name: %s, src0->type: %s\n", src0->name, ggml_type_name(src0->type));
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

    if (DEVIT) {
        ggml_cuda_debug_verify_dequant(*gptq_weight, src0);
    }
    CUDA_CHECK(cudaGetLastError());

    // Allocate destination buffer on device (compute in FP32)
    const bool dst_on_host = ggml_backend_buffer_is_host(dst->buffer);
    ggml_cuda_pool_alloc<float> dst_fp32_storage(ctx.pool(id));
    float * dst_device_fp32 = nullptr;
    bool needs_write_back = true;

    if (!dst_on_host && dst->type == GGML_TYPE_F32) {
        dst_device_fp32 = (float *) dst->data;
        needs_write_back = false;
    } else {
        dst_device_fp32 = dst_fp32_storage.alloc(ne_dst);
    }

    ggml_cuda_dlblas_gemmex(ctx, *gptq_weight, src1_ptr, src1_type, src0, src1, dst_device_fp32, GGML_TYPE_F32);
    CUDA_CHECK(cudaGetLastError());

    // Write back results to destination tensor
    if (!needs_write_back) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return;
    }

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
        CUDA_CHECK(cudaStreamSynchronize(stream));
        GGML_ABORT("DL: [%s] unsupported dst type %s", __FUNCTION__, ggml_type_name(dst->type));
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return;

    // should not reach here
    GGML_ABORT("DL: [%s] no valid GPTQ mapping found", __FUNCTION__);
}

static void ggml_cuda_op_mul_mat_dlblas(
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

// Configuration
bool is_devit_enabled() {
    return DEVIT;
}

int get_gptq_group_size() {
    return GGML_CUDA_GPTQ_GROUP_SIZE;
}

// Helper to match plugin interface
static const ggml_tensor* ggml_dl_get_base_tensor(const ggml_tensor* t) {
    return ggml_cuda_get_base_tensor(t);
}

// GPTQ Weight Management (exposing static functions)
gptq_weight_data* get_gptq_weight(const ggml_tensor* tensor) {
    const int device_id = ggml_cuda_get_device();
    ggml_gptq_data* gptq = ggml_cuda_gptq_get_weight(tensor, device_id);
    if (!gptq) return nullptr;

    // Convert to plugin interface type
    gptq_weight_data* data = new gptq_weight_data();
    data->qweight = gptq->qweight;
    data->qzeros = gptq->qzeros;
    data->scales = gptq->scales;
    data->qweight_size = gptq->qweight_size;
    data->qzeros_size = gptq->qzeros_size;
    data->scales_size = gptq->scales_size;
    data->M = gptq->M;
    data->K = gptq->K;
    data->group_size = gptq->group_size;
    data->bits = gptq->bits;
    data->qweight_type = CUDA_R_8U;
    data->qzeros_type = gptq->qzeros_type;
    data->scales_type = gptq->scales_type;
    return data;
}

void store_gptq_weight(const ggml_tensor* tensor, gptq_weight_data* data) {
    // Convert plugin type to internal type
    ggml_gptq_data* gptq = new ggml_gptq_data();
    gptq->qweight = data->qweight;
    gptq->qzeros = data->qzeros;
    gptq->scales = data->scales;
    gptq->qweight_size = data->qweight_size;
    gptq->qzeros_size = data->qzeros_size;
    gptq->scales_size = data->scales_size;
    gptq->M = data->M;
    gptq->K = data->K;
    gptq->group_size = data->group_size;
    gptq->bits = data->bits;
    gptq->qzeros_type = data->qzeros_type;
    gptq->scales_type = data->scales_type;
    gptq->num_groups = (data->K + data->group_size - 1) / data->group_size;

    const int device_id = ggml_cuda_get_device();
    ggml_cuda_gptq_store_weight(tensor, gptq, device_id);
}

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

bool should_use_dlblas(
    ggml_backend_cuda_context& ctx,
    const ggml_tensor* src0,
    const ggml_tensor* src1,
    const ggml_tensor* dst) {
    GGML_UNUSED(ctx);
    GGML_UNUSED(src1);
    GGML_UNUSED(dst);

    const int device_id = ggml_cuda_get_device();
    return ggml_cuda_gptq_has_weight(src0, device_id) || ggml_cuda_gptq_has_any_weight(src0);
}

// Debug utilities
void debug_print_tensor(
    const char* name,
    const void* data,
    int count,
    ggml_type type,
    cudaStream_t stream) {

    if (!is_devit_enabled()) return;

    // TODO: Implement if needed
}

void debug_verify_dequant(
    const gptq_weight_data& gptq_data,
    const ggml_tensor* src0) {

    if (!is_devit_enabled()) return;

    // Convert to internal type
    ggml_gptq_data internal_data;
    internal_data.qweight = gptq_data.qweight;
    internal_data.qzeros = gptq_data.qzeros;
    internal_data.scales = gptq_data.scales;
    internal_data.M = gptq_data.M;
    internal_data.K = gptq_data.K;
    internal_data.group_size = gptq_data.group_size;
    internal_data.bits = gptq_data.bits;
    internal_data.qzeros_type = gptq_data.qzeros_type;
    internal_data.scales_type = gptq_data.scales_type;
    internal_data.num_groups = (gptq_data.K + gptq_data.group_size - 1) / gptq_data.group_size;

    ggml_cuda_debug_verify_dequant(internal_data, src0);
}

void cleanup() {
    ggml_cuda_gptq_clear_weights();
}

} // namespace ggml_dl

#endif // GGML_USE_DLCU
