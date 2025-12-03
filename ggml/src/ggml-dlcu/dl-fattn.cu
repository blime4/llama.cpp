#include <unistd.h>
#include "ggml.h"
#ifdef GGML_USE_DLFA

#include "dl-fattn.cuh"
#include "../../../include/ggml-dlfa.h"
#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "../ggml-cuda/common.cuh"
#include "../ggml-cuda/convert.cuh"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cudnn.h>
#include <dldnn_ext.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>
#include <mutex>
#include <algorithm>

template<typename T>
static inline float to_float(const T& val) {
    return static_cast<float>(val);
}

template<>
inline float to_float<ggml_fp16_t>(const ggml_fp16_t& val) {
    return ggml_fp16_to_fp32(val);
}

template<>
inline float to_float<ggml_bf16_t>(const ggml_bf16_t& val) {
    return ggml_bf16_to_fp32(val);
}

// Debug helper: print first N elements from a device pointer if total elements > N
static void debug_print_first_n_device(const void * dev_ptr,
                                       int64_t total_elements,
                                       int print_n,
                                       enum ggml_type type,
                                       const char * name) {
    if (dev_ptr == nullptr || total_elements <= print_n) {
        return;
    }
    // Validate pointer is device memory accessible from current device
    cudaPointerAttributes attrs{};
    cudaError_t attr_err = cudaPointerGetAttributes(&attrs, dev_ptr);
    if (attr_err != cudaSuccess) {
        printf("[DEBUG] %s: skip print - cudaPointerGetAttributes failed: %s\n",
               name ? name : "tensor", cudaGetErrorString(attr_err));
        return;
    }
#if CUDART_VERSION >= 10000
    const bool is_dev_mem = (attrs.type == cudaMemoryTypeDevice);
    const int dev_attr = attrs.device;
#else
    const bool is_dev_mem = (attrs.memoryType == cudaMemoryTypeDevice);
    const int dev_attr = attrs.device;
#endif
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    if (!is_dev_mem || cur_dev != dev_attr) {
        printf("[DEBUG] %s: skip print - pointer not device memory on current device (cur=%d, ptr_dev=%d)\n",
               name ? name : "tensor", cur_dev, dev_attr);
        return;
    }
    const size_t elem_size = ggml_type_size(type);
    const size_t copy_bytes = (size_t) print_n * elem_size;
    std::vector<uint8_t> host_buf(copy_bytes);
    cudaError_t cerr = cudaMemcpy(host_buf.data(), dev_ptr, copy_bytes, cudaMemcpyDeviceToHost);
    if (cerr != cudaSuccess) {
        printf("[DEBUG] %s: cudaMemcpy failed: %s\n", name ? name : "tensor", cudaGetErrorString(cerr));
        return;
    }
    printf("[%s] first %d elements:", name ? name : "tensor", print_n);
    switch (type) {
        case GGML_TYPE_F32: {
            const float * data = reinterpret_cast<const float *>(host_buf.data());
            for (int i = 0; i < print_n; ++i) {
                printf(" %.6f", data[i]);
            }
        } break;
        case GGML_TYPE_F16: {
            const ggml_fp16_t * data = reinterpret_cast<const ggml_fp16_t *>(host_buf.data());
            for (int i = 0; i < print_n; ++i) {
                printf(" %.6f", ggml_fp16_to_fp32(data[i]));
            }
        } break;
        case GGML_TYPE_BF16: {
            const ggml_bf16_t * data = reinterpret_cast<const ggml_bf16_t *>(host_buf.data());
            for (int i = 0; i < print_n; ++i) {
                printf(" %.6f", ggml_bf16_to_fp32(data[i]));
            }
        } break;
        default:
            printf(" (printing not implemented for type %d)", (int) type);
            break;
    }
    printf("\n");
}

// ============================================================================
// cuDNN Helper Functions
// ============================================================================

static cudnnDataType_t ggml_type_to_cudnn_type(enum ggml_type type) {
    switch (type) {
        case GGML_TYPE_F16:  return CUDNN_DATA_HALF;
        case GGML_TYPE_F32:  return CUDNN_DATA_FLOAT;
        case GGML_TYPE_BF16: return CUDNN_DATA_BFLOAT16;
        default:
            GGML_ASSERT(false && "Unsupported data type for cuDNN");
            return CUDNN_DATA_FLOAT;
    }
}

// ============================================================================
// GGMLTensorDescriptor: Bridge between GGML and cuDNN tensor layouts
// ============================================================================

struct GGMLTensorDescriptor {
    cudnnTensorDescriptor_t desc;

    GGMLTensorDescriptor() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }

    ~GGMLTensorDescriptor() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    }

#if 0 // sdpa have problem.[1. output do support jumping stride. 2. mask do not support GQA.] do not use this.
    void sdpa_set_from_ggml_qkv(const struct ggml_tensor* tensor, enum ggml_type target_type) const {
        // 1. SDPA : Q/K/V Tensors (zero-copy mapping):
        // --------------------------------------
        // GGML format: ne[0]=D, ne[1]=S, ne[2]=H, ne[3]=B (row-major, D is innermost)
        //   Physical offset (bytes): [d,s,h,b] = d*nb[0] + s*nb[1] + h*nb[2] + b*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN format: [B, H, S, D] (row-major, D is innermost)
        //   Physical offset (bytes): [b,h,s,d] = b*nb[3] + h*nb[2] + s*nb[1] + d*nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[2]/nb[0], nb[1]/nb[0], 1]
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);

        const int64_t D = tensor->ne[0];  // head_dim
        const int64_t S = tensor->ne[1];  // seq_len
        const int64_t H = tensor->ne[2];  // num_heads
        const int64_t B = tensor->ne[3];  // batch_size

        // cuDNN expects shape: [B, H, S, D]
        int dims[4] = {
            static_cast<int>(B),
            static_cast<int>(H),
            static_cast<int>(S),
            static_cast<int>(D)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),
            1
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: Q/K/V GGML [D=%ld,S=%ld,H=%ld,B=%ld] -> cuDNN dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            D, S, H, B, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    void sdpa_set_from_ggml_output(const struct ggml_tensor* tensor, enum ggml_type target_type) const {
        // 2. SDPA : Output O Tensor (zero-copy with special strides):
        // -----------------------------------------------------
        // GGML format: ne[0]=D, ne[1]=H, ne[2]=S, ne[3]=B (row-major, D is innermost)
        //   Physical offset (bytes): [d,h,s,b] = d*nb[0] + h*nb[1] + s*nb[2] + b*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN format: [B, H, S, D] (same as Q/K/V output format)
        //   Physical offset (bytes): [b,h,s,d] = b*nb[3] + h*nb[1] + s*nb[2] + d*nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[1]/nb[0], nb[2]/nb[0], 1]
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);
        // IMPORTANT note: output is permute(0, 2, 1, 3)
        const int64_t D = tensor->ne[0];  // head_dim
        const int64_t H = tensor->ne[1];  // num_heads
        const int64_t S = tensor->ne[2];  // seq_len
        const int64_t B = tensor->ne[3];  // batch_size

        // cuDNN expects shape: [B, H, S, D]
        int dims[4] = {
            static_cast<int>(B),
            static_cast<int>(H),
            static_cast<int>(S),
            static_cast<int>(D)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),
            1
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: OUTPUT GGML [D=%ld,H=%ld,S=%ld,B=%ld] -> cuDNN dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            D, H, S, B, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    // DL : NOT FULLY VALIDATED YET. TODO: FULLY VALIDATE THIS.
    void sdpa_set_from_ggml_mask(const struct ggml_tensor* tensor, enum ggml_type target_type, int64_t actual_sq) const {
        // 3. SDPA : Output Mask Tensor (zero-copy with special strides):
        // -------------------------------------------------------
        // GGML format: ne[0]=Sk, ne[1]=Sq_pad, ne[2]=ne32, ne[3]=ne33 (row-major, Sk is innermost)
        //   Physical offset (bytes): [sk,sq_pad,ne32,ne33] = sk*nb[0] + sq_pad*nb[1] + ne32*nb[2] + ne33*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN format: [B, H, Sq, Sk] (row-major, Sk is innermost)
        //   Physical offset (bytes): [b,h,sq,sk] = b*nb[3] + h*nb[2] + sq*nb[1] + sk*nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[2]/nb[0], nb[1]/nb[0], 1]
        //
        // GGML mask format: [Sk, Sq_pad, ne32, ne33] (Sk innermost) | nowadays only support [Sk, Sq_pad, 1, 1] format.
        // cuDNN expects: [B, H, Sq, Sk] (Sk innermost)              | [Sk, Sq_pad, 1, 1] --> [1, 1, Sq, Sk]
        // q:    [n_embd_k, n_batch,     n_head,    ne3 ]            | [D, Sq, H, B]      --> [B, H, Sq, D]
        // k:    [n_embd_k, n_kv,        n_head_kv, ne3 ]            | [D, Sk, H, B]      --> [B, H, Sk, D]
        // v:    [n_embd_v, n_kv,        n_head_kv, ne3 ] !! not transposed !!
        // mask: [n_kv,     n_batch_pad, ne32,      ne33] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
        // res:  [n_embd_v, n_head,      n_batch,   ne3 ] !! permuted !!
        //
        // broadcast:
        //   n_head % n_head_kv == 0
        //   n_head % ne32      == 0
        //   ne3    % ne33      == 0
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);

        const int64_t Sk = tensor->ne[0];  // key_len
        const int64_t Sq_pad = tensor->ne[1];  // seq_len_pad
        const int64_t ne32 = tensor->ne[2];
        const int64_t ne33 = tensor->ne[3];

        // cuDNN expects shape: [B, H, S, D]
        int dims[4] = {
            static_cast<int>(ne33),
            static_cast<int>(ne32),
            static_cast<int>(actual_sq),
            static_cast<int>(Sk)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),
            1
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: MASK GGML [Sk=%ld,Sq_pad=%ld,ne32=%ld,ne33=%ld] -> cuDNN dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            Sk, Sq_pad, ne32, ne33, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }
#endif

    void mha_set_from_ggml_qkv(const struct ggml_tensor* tensor, enum ggml_type target_type) const {
        // MHA : Q/K/V Tensors (zero-copy mapping for cudnnMHAForward):
        // --------------------------------------
        // GGML format: ne[0]=D, ne[1]=S, ne[2]=H, ne[3]=B (row-major, D is innermost)
        //   Physical offset (bytes): [d,s,h,b] = d*nb[0] + s*nb[1] + h*nb[2] + b*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN MHA format: [B, S, H, D] (BSHD, row-major, D is innermost)
        //   Physical offset (bytes): [b,s,h,d] = b*stride[0] + s*stride[1] + h*stride[2] + d*stride[3]
        //   Mapping: b→nb[3], s→nb[1], h→nb[2], d→nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[1]/nb[0], nb[2]/nb[0], 1]
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);

        const int64_t D = tensor->ne[0];  // head_dim
        const int64_t S = tensor->ne[1];  // seq_len
        const int64_t H = tensor->ne[2];  // num_heads
        const int64_t B = tensor->ne[3];  // batch_size

        // cuDNN MHA expects shape: [B, S, H, D] (BSHD format)
        int dims[4] = {
            static_cast<int>(B),
            static_cast<int>(S),
            static_cast<int>(H),
            static_cast<int>(D)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),  // B stride
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),  // S stride
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),  // H stride
            1                                               // D stride
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: MHA Q/K/V GGML [D=%ld,S=%ld,H=%ld,B=%ld] -> cuDNN BSHD dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            D, S, H, B, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    void mha_set_from_ggml_output(const struct ggml_tensor* tensor, enum ggml_type target_type) const {
        // MHA : Output O Tensor (zero-copy mapping for cudnnMHAForward):
        // -----------------------------------------------------
        // GGML format: ne[0]=D, ne[1]=H, ne[2]=S, ne[3]=B (row-major, D is innermost)
        //   Physical offset (bytes): [d,h,s,b] = d*nb[0] + h*nb[1] + s*nb[2] + b*nb[3]
        //   Strides: [1, nb[1]/nb[0], nb[2]/nb[0], nb[3]/nb[0]]
        //
        // cuDNN MHA format: [B, S, H, D] (BSHD, same as Q/K/V format)
        //   Physical offset (bytes): [b,s,h,d] = b*stride[0] + s*stride[1] + h*stride[2] + d*stride[3]
        //   Mapping: b→nb[3], s→nb[2], h→nb[1], d→nb[0]
        //   Expected strides: [nb[3]/nb[0], nb[2]/nb[0], nb[1]/nb[0], 1]
        //
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(target_type);

        const int64_t D = tensor->ne[0];  // head_dim
        const int64_t H = tensor->ne[1];  // num_heads
        const int64_t S = tensor->ne[2];  // seq_len
        const int64_t B = tensor->ne[3];  // batch_size

        // cuDNN MHA expects shape: [B, S, H, D] (BSHD format)
        int dims[4] = {
            static_cast<int>(B),
            static_cast<int>(S),
            static_cast<int>(H),
            static_cast<int>(D)
        };

        int strides[4] = {
            static_cast<int>(tensor->nb[3]/tensor->nb[0]),  // B stride
            static_cast<int>(tensor->nb[2]/tensor->nb[0]),  // S stride (from GGML dim 2)
            static_cast<int>(tensor->nb[1]/tensor->nb[0]),  // H stride (from GGML dim 1)
            1                                               // D stride
        };

        GGML_DL_FATTN_DEBUG_PRINT("GGMLTensorDescriptor: MHA OUTPUT GGML [D=%ld,H=%ld,S=%ld,B=%ld] -> cuDNN BSHD dims=[%d,%d,%d,%d] strides=[%d,%d,%d,%d]\n",
            D, H, S, B, dims[0], dims[1], dims[2], dims[3], strides[0], strides[1], strides[2], strides[3]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    // DL : REMOVE LATER
    // For ALiBi slopes: [batch_size, 1, num_heads]
    void set_from_dims_3d(int dim_0, int dim_1, int dim_2, enum ggml_type type) const {
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(type);
        int dims[3] = {dim_0, dim_1, dim_2};
        int strides[3] = {
            dim_1 * dim_2,  // batch stride
            dim_2,          // seq_len stride (should be 1 for ALiBi)
            1               // num_heads stride
        };

        GGML_DL_FATTN_DEBUG_PRINT("Setting 3D descriptor: dims=[%d, %d, %d], strides=[%d, %d, %d]\n",
                      dims[0], dims[1], dims[2], strides[0], strides[1], strides[2]);

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            3,
            dims,
            strides
        ));
    }

    cudnnTensorDescriptor_t get() const { return desc; }
};

struct dldnn_mha_qkvo_pack {
    ggml_tensor * KQV;
    const ggml_tensor * Q;
    const ggml_tensor * K;
    const ggml_tensor * V;
    const void * q_data;
    const void * k_data;
    const void * v_data;
    GGMLTensorDescriptor q_desc;
    GGMLTensorDescriptor k_desc;
    GGMLTensorDescriptor v_desc;
    GGMLTensorDescriptor out_desc;

    explicit dldnn_mha_qkvo_pack(ggml_tensor * dst)
        : KQV(dst),
          Q(dst->src[0]),
          K(dst->src[1]),
          V(dst->src[2]),
          q_data(Q->data),
          k_data(K->data),
          v_data(V->data) {
        q_desc.mha_set_from_ggml_qkv(Q, Q->type);
        k_desc.mha_set_from_ggml_qkv(K, K->type);
        v_desc.mha_set_from_ggml_qkv(V, V->type);
        out_desc.mha_set_from_ggml_output(KQV, KQV->type);
    }

    dldnn_mha_qkvo_pack(const dldnn_mha_qkvo_pack &) = delete;
    dldnn_mha_qkvo_pack & operator=(const dldnn_mha_qkvo_pack &) = delete;
};

// ============================================================================
// ALiBi Slopes Helper
// ============================================================================

// Helper function to expand ALiBi slopes to 3D format for cuDNN
// Similar to expandTo3D in flash-attention
static void* expand_alibi_slopes_to_3d(
    ggml_cuda_pool_alloc<float>& mem_pool,
    const struct ggml_tensor* mask,
    float max_bias,
    const uint32_t n_head,
    const uint32_t n_head_log2,
    const float m0,
    const float m1,
    cudaStream_t stream
) {
    if (mask == nullptr || max_bias <= 0.0f) {
        return nullptr;
    }

    // Generate ALiBi slopes and expand to 3D format for cuDNN
    printf("Generating ALiBi slopes: max_bias=%f, n_head=%d, n_head_log2=%d, m0=%f, m1=%f\n",
        max_bias, n_head, n_head_log2, m0, m1);

    // Calculate ALiBi slopes for each head
    const size_t slopes_size = n_head * sizeof(float);
    float* slopes_cpu = (float*)malloc(slopes_size);
    if (slopes_cpu == nullptr) {
        GGML_LOG_ERROR("Failed to allocate memory for ALiBi slopes\n");
        return nullptr;
    }

    // Generate slopes for each head
    for (uint32_t h = 0; h < n_head; h++) {
        const float base = h < n_head_log2 ? m0 : m1;
        const int exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;
        slopes_cpu[h] = powf(base, exph);
    }

    // Allocate GPU memory for slopes: just need n_head floats
    // The 3D format is handled by the descriptor, not the actual memory layout
    void* slopes_gpu = nullptr;
    slopes_gpu = mem_pool.alloc(n_head);

    // Copy slopes to GPU
    CUDA_CHECK(cudaMemcpyAsync(slopes_gpu, slopes_cpu, slopes_size, cudaMemcpyHostToDevice, stream));

    // Free CPU memory
    free(slopes_cpu);

    return slopes_gpu;
}

// ============================================================================
// Flash Attention DLDNN Implementation - Internal Functions
// ============================================================================

// MHA Forward implementation for ALiBi support
static void flash_attn_ext_dldnn_mha_forward_for_alibi(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_DL_FATTN_DEBUG_PRINT("\n========== ENTERING %s ==========\n", __FUNCTION__);

    const int id = ggml_cuda_get_device();
    ggml_cuda_pool_alloc<float> alibi_slopes_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> workspace_mem(ctx.pool(id));

    const struct ggml_tensor * KQV  = dst;
    const struct ggml_tensor * Q    = dst->src[0];
    const struct ggml_tensor * K    = dst->src[1];
    const struct ggml_tensor * V    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Getting cuDNN handle...\n");

    // Check CUDA device status before anything
    int device_id = -1;
    cudaError_t cuda_err = cudaGetDevice(&device_id);
    if (cuda_err != cudaSuccess) {
        GGML_LOG_ERROR("Failed to get CUDA device: %s\n", cudaGetErrorString(cuda_err));
        return;
    }
    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Current CUDA device: %d\n", device_id);

    // Check GPU memory status
    size_t free_mem=0;
    size_t total_mem=0;
    cuda_err = cudaMemGetInfo(&free_mem, &total_mem);
    if (cuda_err == cudaSuccess) {
        GGML_DL_FATTN_DEBUG_PRINT("DEBUG: GPU memory - Free: %.2f MB / Total: %.2f MB (%.1f%% free)\n",
               free_mem / (1024.0 * 1024.0),
               total_mem / (1024.0 * 1024.0),
               100.0 * free_mem / total_mem);
    }

    cudnnHandle_t cudnn_handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, ctx.stream()));

    // Get cuDNN version
    size_t cudnn_version = cudnnGetVersion();
    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: cuDNN version: %zu\n", cudnn_version);

    // Pointers to the actual data, either original or converted
    const void* q_data = Q->data;
    const void* k_data = K->data;
    const void* v_data = V->data;

    GGMLTensorDescriptor q_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor k_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor v_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor out_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor alibi_slopes_desc = GGMLTensorDescriptor();

    // Use zero-copy descriptors: directly map GGML format to cuDNN format via strides
    // No physical data reordering needed
    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Setting up zero-copy descriptors for MHA...\n");

    // For Q/K/V: GGML [D,S,H,B] -> cuDNN [B,S,H,D] via stride mapping
    // If type conversion occurred, converted data is contiguous, so use original tensor's stride
    // (convert_tensor_data preserves layout, so stride should match)
    q_desc.mha_set_from_ggml_qkv(Q, Q->type);
    k_desc.mha_set_from_ggml_qkv(K, K->type);
    v_desc.mha_set_from_ggml_qkv(V, V->type);

    // For output: GGML [D,H,S,B] -> cuDNN [B,S,H,D] via stride mapping
    // Directly write to final output tensor, no temporary buffer needed
    out_desc.mha_set_from_ggml_output(KQV, KQV->type);

    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Zero-copy descriptors created successfully.\n");

    // TODO : add more check
    // head_num % head_num_k == 0
    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);

    size_t workspace_size = 0;

    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        // logit_softcap is not used in dldnn_mha_fwd, keep scale as is
        GGML_LOG_ERROR("logit_softcap is not supported for DLDNN yet.\n");
        // TODO : support logit_softcap, need cudnnMHAForward support.
        // scale /= logit_softcap;
    }
    const uint32_t n_head      = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    void* alibi_slopes_ptr = expand_alibi_slopes_to_3d(
        alibi_slopes_mem,
        mask, max_bias, n_head, n_head_log2, m0, m1, ctx.stream()
    );


    // Set alibi_slopes_desc for 3D format: [1, 1, num_heads]
    // cuDNN expects: [1, 1, num_heads] where first dimension must be 1
    if (alibi_slopes_ptr != nullptr) {
        alibi_slopes_desc.set_from_dims_3d(1, 1, n_head, GGML_TYPE_F32);
    }

    // - alibi_slopes_desc : Now properly set for 3D ALiBi slopes
    // - softmax_lse_desc : return_softmax is false, pass nullptr.
    // - p_desc : return_softmax is false, pass nullptr.

    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Calling cudnnGetMHAForwardWorkspaceSize...\n");

    CUDNN_CHECK(cudnnGetMHAForwardWorkspaceSize(
        cudnn_handle, q_desc.get(), k_desc.get(), v_desc.get(),
        alibi_slopes_ptr != nullptr ? alibi_slopes_desc.get() : nullptr, // Pass nullptr if no ALiBi
        out_desc.get(), nullptr, nullptr,
        0.0f, scale,
        false, -1, -1,
        false,
        &workspace_size
    ));

    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: cudnnGetMHAForwardWorkspaceSize completed, workspace_size=%zu\n", workspace_size);

    void* workspace = nullptr;
    if (workspace_size > 0) {
        GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Allocating workspace memory: %zu bytes (%.2f MB)...\n",
               workspace_size, workspace_size / (1024.0 * 1024.0));

        workspace = workspace_mem.alloc(workspace_size);

        GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Workspace allocated successfully at %p\n", workspace);
    } else {
        GGML_DL_FATTN_DEBUG_PRINT("DEBUG: No workspace memory needed.\n");
    }

    unsigned long long philox_seed = 0;
    unsigned long long philox_offset = 0;

    // Debug: Print detailed parameters before cudnnMHAForward call
    GGML_DL_FATTN_DEBUG_PRINT("\n=== DLDNN MHA Forward Debug Info ===\n");
    GGML_DL_FATTN_DEBUG_PRINT("Tensor shapes (DSHB format):\n");
    GGML_DL_FATTN_DEBUG_PRINT("  Q=[D:%ld, S:%ld, H:%ld, B:%ld]\n", Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3]);
    GGML_DL_FATTN_DEBUG_PRINT("  K=[D:%ld, S:%ld, H:%ld, B:%ld]\n", K->ne[0], K->ne[1], K->ne[2], K->ne[3]);
    GGML_DL_FATTN_DEBUG_PRINT("  V=[D:%ld, S:%ld, H:%ld, B:%ld]\n", V->ne[0], V->ne[1], V->ne[2], V->ne[3]);
    GGML_DL_FATTN_DEBUG_PRINT("Derived parameters:\n");
    GGML_DL_FATTN_DEBUG_PRINT("  Head size (D): %ld\n", Q->ne[0]);
    GGML_DL_FATTN_DEBUG_PRINT("  Sequence length (S): %ld\n", Q->ne[1]);
    GGML_DL_FATTN_DEBUG_PRINT("  Num heads Q: %ld, K: %ld, V: %ld\n", Q->ne[2], K->ne[2], V->ne[2]);
    GGML_DL_FATTN_DEBUG_PRINT("  Batch size (B): %ld\n", Q->ne[3]);
    GGML_DL_FATTN_DEBUG_PRINT("  GQA ratio: %ld\n", Q->ne[2] / K->ne[2]);
    GGML_DL_FATTN_DEBUG_PRINT("Attention parameters:\n");
    GGML_DL_FATTN_DEBUG_PRINT("  scale=%.6f, max_bias=%.6f, logit_softcap=%.6f\n", scale, max_bias, logit_softcap);
    GGML_DL_FATTN_DEBUG_PRINT("  Has mask: %s\n", mask ? "yes" : "no");
    GGML_DL_FATTN_DEBUG_PRINT("  Has ALiBi: %s\n", alibi_slopes_ptr ? "yes" : "no");
    GGML_DL_FATTN_DEBUG_PRINT("Memory info:\n");
    GGML_DL_FATTN_DEBUG_PRINT("  Workspace size: %zu bytes (%.2f MB)\n", workspace_size, workspace_size / (1024.0 * 1024.0));
    GGML_DL_FATTN_DEBUG_PRINT("  Data type: %s\n", ggml_type_name(KQV->type));
    GGML_DL_FATTN_DEBUG_PRINT("GPU pointers (zero-copy):\n");
    GGML_DL_FATTN_DEBUG_PRINT("  q_data=%p (direct GGML [D,S,H,B] via strides)\n", q_data);
    GGML_DL_FATTN_DEBUG_PRINT("  k_data=%p (direct GGML [D,S,H,B] via strides)\n", k_data);
    GGML_DL_FATTN_DEBUG_PRINT("  v_data=%p (direct GGML [D,S,H,B] via strides)\n", v_data);
    GGML_DL_FATTN_DEBUG_PRINT("  output=%p (direct GGML [D,H,S,B] via strides)\n", KQV->data);
    GGML_DL_FATTN_DEBUG_PRINT("  workspace=%p\n", workspace);
    GGML_DL_FATTN_DEBUG_PRINT("  alibi_slopes=%p\n", alibi_slopes_ptr);
    GGML_DL_FATTN_DEBUG_PRINT("\n>>> Calling cudnnMHAForward (zero-copy mode)...\n");

    // Call cuDNN MHA Forward with zero-copy pointers
    // Q/K/V: GGML [D,S,H,B] read as cuDNN [B,S,H,D] (BSHD) via strides
    // Output: cuDNN [B,S,H,D] (BSHD) written as GGML [D,H,S,B] via strides
    CUDNN_CHECK(cudnnMHAForward(
        cudnn_handle, q_desc.get(), q_data,
        k_desc.get(), k_data, v_desc.get(), v_data,
        alibi_slopes_ptr != nullptr ? alibi_slopes_desc.get() : nullptr,
        alibi_slopes_ptr, // This will be nullptr if no ALiBi
        out_desc.get(), KQV->data,  // Direct output to final tensor
        nullptr, nullptr,
        nullptr, nullptr,
        0.0f, scale,
        false, -1, -1,
        false,
        &philox_seed, &philox_offset,
        workspace, workspace_size
    ));

    GGML_DL_FATTN_DEBUG_PRINT("cudnnMHAForward completed successfully (zero-copy mode).\n");

    // Check kernel execution
    GGML_DL_FATTN_DEBUG_PRINT("DEBUG: Checking CUDA errors...\n");
    CUDA_CHECK(cudaGetLastError());
}

// MHA Forward implementation when mask semantic can use window/causal parameters
static void flash_attn_ext_dldnn_mha_forward_for_mask(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_DL_FATTN_DEBUG_PRINT("\n========== ENTERING %s ==========\n", __FUNCTION__);

    const int id = ggml_cuda_get_device();
    ggml_cuda_pool_alloc<float> softmax_lse_mem(ctx.pool(id));
    ggml_cuda_pool_alloc<uint8_t> workspace_mem(ctx.pool(id));

    const ggml_tensor * mask = dst->src[3];
    ggml_flash_attn_mask_params mask_info = {
        /*.present          =*/ mask != nullptr,
        /*.is_causal        =*/ false,
        /*.window_left      =*/ -1,
        /*.window_right     =*/ -1,
        /*.per_token_window =*/ false,
        /*.multi_sequence   =*/ false,
        /*.has_alibi_bias   =*/ false,
    };
    ggml_flash_attn_ext_get_mask_params(dst, &mask_info);

    const char* env_dl_fattn_debug = getenv("GGML_DL_FATTN_DEBUG");
    if (env_dl_fattn_debug != nullptr && strcmp(env_dl_fattn_debug, "1") == 0)
    { // debug code.
        const ggml_tensor * Q    = dst->src[0];
        const ggml_tensor * K    = dst->src[1];
        const ggml_tensor * V    = dst->src[2];

        printf("Mask metadata summary:\n");
        printf("  Q=[D:%ld,S:%ld,H:%ld,B:%ld] %s\n",
            Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3], ggml_type_name(Q->type));
        printf("  K=[D:%ld,S:%ld,H:%ld,B:%ld] %s\n",
            K->ne[0], K->ne[1], K->ne[2], K->ne[3], ggml_type_name(K->type));
        printf("  V=[D:%ld,S:%ld,H:%ld,B:%ld] %s\n",
            V->ne[0], V->ne[1], V->ne[2], V->ne[3], ggml_type_name(V->type));
        printf("  Output=[D:%ld,H:%ld,S:%ld,B:%ld] %s\n",
            dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], ggml_type_name(dst->type));
        if (mask) {
            printf("  mask shape=[%ld,%ld,%ld,%ld] type=%s\n",
                mask->ne[0], mask->ne[1], mask->ne[2], mask->ne[3], ggml_type_name(mask->type));
        } else {
            printf("  mask=nullptr\n");
        }
        printf("  meta: present=%d, causal=%d, window_left=%d, window_right=%d, per_token=%d, multi_seq=%d, has_alibi=%d\n",
            mask_info.present ? 1 : 0,
            mask_info.is_causal ? 1 : 0,
            mask_info.window_left,
            mask_info.window_right,
            mask_info.per_token_window ? 1 : 0,
            mask_info.multi_sequence ? 1 : 0,
            mask_info.has_alibi_bias ? 1 : 0);

        if (mask) {
            const int64_t mask_sk = mask->ne[0];
            const int64_t mask_sq_pad = mask->ne[1];
            const int64_t max_rows = std::min<int64_t>(mask_sq_pad, (int64_t) 4);
            const int64_t max_cols = std::min<int64_t>(mask_sk, (int64_t) 16);
            const size_t elem_size = ggml_type_size(mask->type);
            const size_t row_bytes = mask_sk * elem_size;
            std::vector<uint8_t> row_buf(row_bytes);

            printf("  mask preview (sq=%lld, sk=%lld):\n", (long long) mask_sq_pad, (long long) mask_sk);
            for (int64_t r = 0; r < max_rows; ++r) {
                printf("    row %lld:", (long long) r);
                ggml_backend_tensor_get(mask, row_buf.data(), r * mask->nb[1], row_bytes);

                for (int64_t c = 0; c < max_cols; ++c) {
                    float val = 0.0f;
                    switch (mask->type) {
                        case GGML_TYPE_F32:
                            val = reinterpret_cast<float *>(row_buf.data())[c];
                            break;
                        case GGML_TYPE_F16:
                            val = ggml_fp16_to_fp32(reinterpret_cast<ggml_fp16_t *>(row_buf.data())[c]);
                            break;
                        case GGML_TYPE_BF16:
                            val = ggml_bf16_to_fp32(reinterpret_cast<ggml_bf16_t *>(row_buf.data())[c]);
                            break;
                        default:
                            val = to_float(reinterpret_cast<float *>(row_buf.data())[c]);
                            break;
                    }

                    if (std::isinf(val) && val < 0) {
                        printf("  -INF");
                    } else {
                        printf(" %6.2f", val);
                    }
                }
                if (mask_sk > max_cols) {
                    printf(" ...");
                }
                printf("\n");
            }
            if (mask_sq_pad > max_rows) {
                printf("    ...\n");
            }
        }
    }
    const bool mask_supports_cudnn = mask_info.present && !mask_info.per_token_window && !mask_info.multi_sequence;
    if (!mask_supports_cudnn || mask_info.has_alibi_bias) {
        if (!mask_info.present) {
            GGML_DL_FATTN_DEBUG_PRINT("INFO: mask metadata not present\n");
        } else if (mask_info.per_token_window || mask_info.multi_sequence) {
            GGML_DL_FATTN_DEBUG_PRINT(
                "INFO: mask_metadata has unsupported configuration (per_token_window=%d, multi_sequence=%d)\n",
                mask_info.per_token_window, mask_info.multi_sequence);
        } else if (mask_info.has_alibi_bias) {
            GGML_DL_FATTN_DEBUG_PRINT("INFO: mask_metadata->has_alibi_bias is true\n");
        }
    }

    cudnnHandle_t cudnn_handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, ctx.stream()));

    dldnn_mha_qkvo_pack pack(dst);

    const int64_t seq_q = pack.Q->ne[1];
    const int64_t seq_k = pack.K->ne[1];

    // Track the real (unpadded) key length during decode using the shared
    // thread-local state so tests can reset it between runs.
    int64_t & seq_k_real = ggml_dl::flash_attn_ext_dldnn_decode_state().seq_k_real;

    GGMLTensorDescriptor k_desc_trunc;
    GGMLTensorDescriptor v_desc_trunc;
    const GGMLTensorDescriptor * k_desc_ptr = &pack.k_desc;
    const GGMLTensorDescriptor * v_desc_ptr = &pack.v_desc;

    // cauase seq_k(256) will padding [DSHB] --> BHSD
    if (mask != nullptr) {
        auto set_trunc_desc = [&](const ggml_tensor * tensor, GGMLTensorDescriptor & desc, int64_t trunc_seq_len) {
            const int64_t clamped_seq = std::min<int64_t>(trunc_seq_len, tensor->ne[1]);
            GGML_ASSERT(clamped_seq > 0);
            int dims[4] = {
                (int) tensor->ne[3],
                (int) clamped_seq,
                (int) tensor->ne[2],
                (int) tensor->ne[0],
            };
            int strides[4] = {
                (int) (tensor->nb[3]/tensor->nb[0]),
                (int) (tensor->nb[1]/tensor->nb[0]),
                (int) (tensor->nb[2]/tensor->nb[0]),
                1,
            };
            CUDNN_CHECK(cudnnSetTensorNdDescriptor(
                desc.get(),
                ggml_type_to_cudnn_type(tensor->type),
                4,
                dims,
                strides));
        };

        if (seq_q != 1) {
            // Prefill: real key length matches the prompt length.
            seq_k_real = std::min<int64_t>(seq_q, seq_k);
            set_trunc_desc(pack.K, k_desc_trunc, seq_k_real);
            set_trunc_desc(pack.V, v_desc_trunc, seq_k_real);
            k_desc_ptr = &k_desc_trunc;
            v_desc_ptr = &v_desc_trunc;
        } else {
            // Decode: grow the real key length one token at a time.
            if (pack.K->ne[3] != 1) {
                GGML_ABORT(
                    "decode path currently supports batch=1 (got %lld); falling back to padded descriptors\n",
                    (long long) pack.K->ne[3]);
            } else {
                if (seq_k_real <= 0 || seq_k_real > seq_k) {
                    // First decode step or stale state (e.g. after reset). Start from current prompt len.
                    seq_k_real = std::min<int64_t>(seq_q, seq_k);
                }
                const int64_t prev_seq_k_real = seq_k_real;
                const int64_t new_seq_k_real = std::min<int64_t>(prev_seq_k_real + seq_q, seq_k);

                set_trunc_desc(pack.K, k_desc_trunc, new_seq_k_real);
                set_trunc_desc(pack.V, v_desc_trunc, new_seq_k_real);
                k_desc_ptr = &k_desc_trunc;
                v_desc_ptr = &v_desc_trunc;

                seq_k_real = new_seq_k_real;
                GGML_DL_FATTN_DEBUG_PRINT(
                    "decode seq_k_real updated: prev=%lld new=%lld (max=%lld)\n",
                    (long long) prev_seq_k_real,
                    (long long) seq_k_real,
                    (long long) seq_k);
            }
        }
    }

    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));
    GGML_UNUSED(max_bias);
    GGML_UNUSED(logit_softcap);

    GGML_DL_FATTN_DEBUG_PRINT("calling cudnnGetMHAForwardWorkspaceSize...\n");
    GGML_DL_FATTN_DEBUG_PRINT("INFO: params scale=%.6f max_bias=%.6f logit_softcap=%.6f\n", scale, max_bias, logit_softcap);
    GGML_DL_FATTN_DEBUG_PRINT("INFO: is_causal=%d, window_left=%d, window_right=%d\n", mask_info.is_causal, mask_info.window_left, mask_info.window_right);

    // cuDNN requires a tensor descriptor and storage for the softmax log-sum-exp output.
    const int64_t batch = pack.Q->ne[3];
    const int64_t n_heads = pack.Q->ne[2];
    GGMLTensorDescriptor softmax_lse_desc;
    {
        int dims[3] = {
            (int) batch,
            (int) n_heads,
            (int) seq_q,
        };
        int strides[3] = {
            (int) (n_heads * seq_q),
            (int) seq_q,
            1,
        };
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            softmax_lse_desc.get(),
            CUDNN_DATA_FLOAT,
            3,
            dims,
            strides));
    }

    float * softmax_lse = nullptr;
    softmax_lse = softmax_lse_mem.alloc(batch * n_heads * seq_q);

    size_t workspace_size = 0;
    CUDNN_CHECK(cudnnGetMHAForwardWorkspaceSize(
        cudnn_handle,
        pack.q_desc.get(), k_desc_ptr->get(), v_desc_ptr->get(),
        nullptr,
        pack.out_desc.get(), softmax_lse_desc.get(), nullptr,
        0.0f, scale,
        mask_info.is_causal,
        mask_info.window_left,
        mask_info.window_right,
        false,
        &workspace_size));

    void * workspace = nullptr;
    if (workspace_size > 0) {
        workspace = workspace_mem.alloc(workspace_size);
    }

    unsigned long long philox_seed = 0;
    unsigned long long philox_offset = 0;

    CUDNN_CHECK(cudnnMHAForward(
        cudnn_handle,
        pack.q_desc.get(), pack.q_data,
        k_desc_ptr->get(), pack.k_data,
        v_desc_ptr->get(), pack.v_data,
        nullptr, nullptr,
        pack.out_desc.get(), pack.KQV->data,
        softmax_lse_desc.get(), softmax_lse,
        nullptr, nullptr,
        0.0f,
        scale,
        mask_info.is_causal,
        mask_info.window_left,
        mask_info.window_right,
        false,
        &philox_seed, &philox_offset,
        workspace, workspace_size));

    CUDA_CHECK(cudaGetLastError());
}

#if 0 // sdpa have problem.[1. output do support jumping stride. 2. mask do not support GQA.] do not use this.
static void flash_attn_ext_dldnn_scaled_dot_product(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);
    GGML_DL_FATTN_DEBUG_PRINT("\n========== ENTERING %s ==========\n", __FUNCTION__);
    bool ok = true;
    const struct ggml_tensor * KQV  = dst;
    const struct ggml_tensor * Q    = dst->src[0];
    const struct ggml_tensor * K    = dst->src[1];
    const struct ggml_tensor * V    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    cudnnHandle_t cudnn_handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, ctx.stream()));

    // Extract parameters
    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

    // Tensor dimensions (GGML format: [D, S, H, B])
    const int64_t D_q = Q->ne[0];   // head_dim
    const int64_t S_q = Q->ne[1];   // query seq_len
    const int64_t H_q = Q->ne[2];   // num query heads
    const int64_t B   = Q->ne[3];   // batch_size

    const int64_t D_k = K->ne[0];
    const int64_t S_k = K->ne[1];   // key seq_len
    const int64_t H_k = K->ne[2];   // num key heads (may differ for GQA)

    const int64_t D_v = V->ne[0];
    const int64_t H_v = V->ne[2];   // num value heads (may differ for GQA)

    GGML_DL_FATTN_DEBUG_PRINT("Q: [D=%ld, Sq=%ld, Hq=%ld, B=%ld], K: [D=%ld, Sk=%ld, Hk=%ld, B=%ld], V: [D=%ld, _, Hv=%ld, B=%ld]\n",
        D_q, S_q, H_q, B, D_k, S_k, H_k, B, D_v, H_v, B);

    // Type conversion if needed (only time we might copy data)
    const void* q_data = Q->data;
    const void* k_data = K->data;
    const void* v_data = V->data;
    const void* mask_data = (mask != nullptr) ? mask->data : nullptr;

    // DL : REMOVE LATER
    GGML_DL_FATTN_DEBUG_PRINT("type : KQV=%d, Q=%d, K=%d, V=%d\n", KQV->type, Q->type, K->type, V->type);

    // Check GQA (Grouped Query Attention) - K and V may have fewer heads than Q
    if (H_q != H_k || H_q != H_v) {
        GGML_DL_FATTN_DEBUG_PRINT("GQA detected: Q_heads=%ld, K_heads=%ld, V_heads=%ld\n", H_q, H_k, H_v);
    }

    // nowadays QKVO must have the same data type.
    // but in llama.cpp. O can be different from QKV.
    // out_desc is the temporary descriptor for the cudnn expected output.
    // we will convert the output to the expected type in the end.
    GGMLTensorDescriptor q_desc;
    GGMLTensorDescriptor k_desc;
    GGMLTensorDescriptor v_desc;
    GGMLTensorDescriptor out_desc;
    GGMLTensorDescriptor mask_desc;

    // cudnnScaledDotProductAttention only support QKVO are the same type.
    GGML_ASSERT(Q->type == K->type && Q->type == V->type && Q->type == KQV->type);
    q_desc.sdpa_set_from_ggml_qkv(Q, Q->type);         // [D,S,H,B] -> [B,H,S,D]
    k_desc.sdpa_set_from_ggml_qkv(K, K->type);         // [D,S,H,B] -> [B,H,S,D]
    v_desc.sdpa_set_from_ggml_qkv(V, V->type);         // [D,S,H,B] -> [B,H,S,D]
    out_desc.sdpa_set_from_ggml_output(KQV, Q->type);  // [D,H,S,B] -> [B,H,S,D]
    // bugid : 16411, need to support output Unconventional stride.
    // out_desc.sdpa_set_from_ggml_qkv(KQV, KQV->type);  // [D,H,S,B] -> [B,H,S,D]

    GGML_DL_FATTN_DEBUG_PRINT("Zero-copy descriptors created: Q/K/V[D=%ld,S=%ld,H=%ld,B=%ld], O[D=%ld,H=%ld,S=%ld,B=%ld]\n",
        D_q, S_q, H_q, B, D_q, H_q, S_q, B);

    bool is_causal = false;

    if (mask != nullptr) {
        // GGML mask format: [Sk, Sq_pad, ne32, ne33] (Sk innermost) | nowadays only support [Sk, Sq_pad, 1, 1] format.
        // cuDNN expects: [B, H, Sq, Sk] (Sk innermost)              | [Sk, Sq_pad, 1, 1] --> [1, 1, Sq, Sk]
        // q:    [n_embd_k, n_batch,     n_head,    ne3 ]            | [D, Sq, H, B]      --> [B, H, Sq, D]
        // k:    [n_embd_k, n_kv,        n_head_kv, ne3 ]            | [D, Sk, H, B]      --> [B, H, Sk, D]
        // v:    [n_embd_v, n_kv,        n_head_kv, ne3 ] !! not transposed !!
        // mask: [n_kv,     n_batch_pad, ne32,      ne33] !! n_batch_pad = GGML_PAD(n_batch, GGML_KQ_MASK_PAD) !!
        // res:  [n_embd_v, n_head,      n_batch,   ne3 ] !! permuted !!
        //
        // broadcast:
        //   n_head % n_head_kv == 0
        //   n_head % ne32      == 0
        //   ne3    % ne33      == 0
        //

        const int64_t mask_sk = mask->ne[0];      // key sequence length
        const int64_t mask_sq_pad = mask->ne[1];  // padded query sequence length
        const int64_t mask_dim2 = mask->ne[2];
        const int64_t mask_dim3 = mask->ne[3];
        const int64_t actual_sq = S_q;            // actual query sequence length
        const int64_t actual_sk = S_k;            // actual key sequence length

        GGML_DL_FATTN_DEBUG_PRINT("Mask dimensions: GGML [Sk=%ld, Sq_pad=%ld, %ld, %ld], need cuDNN [1, 1, Sq=%ld, Sk=%ld]\n",
               mask_sk, mask_sq_pad, mask_dim2, mask_dim3, actual_sq, actual_sk);

        // Validate mask dimensions
        if (mask_dim2 != 1 || mask_dim3 != 1) {
            GGML_LOG_WARN("[bugid : 16426. cudnnScaledDotProductAttention mask need to support GQA. remove when support.] "
                "DLDNN: Mask with dimensions [%ld, %ld, %ld, %ld] not supported. Only [Sk, Sq, 1, 1] format supported. Falling back.\n",
                         mask_sk, mask_sq_pad, mask_dim2, mask_dim3);
            ok = false;
        }

        if (ok) {
            GGML_DL_FATTN_DEBUG_PRINT("mask dtype : %d.\n", mask->type);
            mask_desc.sdpa_set_from_ggml_mask(mask, mask->type, actual_sq);
        }
    } else {
        // No explicit mask, use causal attention
        is_causal = true;
        GGML_DL_FATTN_DEBUG_PRINT("No explicit mask provided, using causal attention\n");
    }

    // Get cuDNN workspace size
    if (ok) {
        size_t workspace_size = 0;
        CUDNN_CHECK(cudnnGetScaledDotProductAttentionWorkspaceSize(
            cudnn_handle,
            q_desc.get(),
            k_desc.get(),
            v_desc.get(),
            mask ? mask_desc.get() : nullptr,
            out_desc.get(),
            0.0f,
            is_causal,
            scale,
            &workspace_size
        ));

        GGML_DL_FATTN_DEBUG_PRINT("cuDNN workspace size: %zu bytes\n", workspace_size);

        // Allocate workspace
        void* workspace = nullptr;
        if (workspace_size > 0) {
            CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
        }

        CUDA_CHECK(cudaGetLastError());

        GGML_DL_FATTN_DEBUG_PRINT("Calling cudnnScaledDotProductAttention:\n");
        GGML_DL_FATTN_DEBUG_PRINT("  Q ptr: %p (direct GGML [D,S,H,B] via strides)\n", q_data);
        GGML_DL_FATTN_DEBUG_PRINT("  K ptr: %p (direct GGML [D,S,H,B] via strides)\n", k_data);
        GGML_DL_FATTN_DEBUG_PRINT("  V ptr: %p (direct GGML [D,S,H,B] via strides)\n", v_data);
        GGML_DL_FATTN_DEBUG_PRINT("  mask ptr: %p %s\n", mask_data, mask ? "(direct GGML [Sk, Sq_pad, ne32, ne33])" : "(causal)");
        GGML_DL_FATTN_DEBUG_PRINT("  output ptr: %p (direct GGML [D,H,S,B] via special strides)\n", KQV->data);
        GGML_DL_FATTN_DEBUG_PRINT("  dropout: %.6f, is_causal: %s, scale: %.6f\n",
            0.0f, is_causal ? "true" : "false", scale);

        // Print first 10 elements for Q/K/V/Mask/Output if element count > 10
        {
            const int print_n = 10;
            const int64_t q_elems = D_q * S_q * H_q * B;
            const int64_t k_elems = D_k * S_k * H_k * B;
            const int64_t v_elems = D_v * S_k * H_v * B; // V shares S_k, H_v
            const int64_t o_elems = D_q * H_q * S_q * B;
            debug_print_first_n_device(q_data, q_elems, print_n, Q->type, "Q");
            debug_print_first_n_device(k_data, k_elems, print_n, K->type, "K");
            debug_print_first_n_device(v_data, v_elems, print_n, V->type, "V");
            if (mask) {
                const int64_t mask_elems = mask->ne[0] * mask->ne[1] * mask->ne[2] * mask->ne[3];
                debug_print_first_n_device(mask_data, mask_elems, print_n, mask->type, "Mask");
            }
            debug_print_first_n_device(KQV->data, o_elems, print_n, KQV->type, "O(pre)");
        }

        CUDA_CHECK(cudaGetLastError());

        // Call cuDNN Flash Attention with zero-copy pointers
        cudnnStatus_t status = cudnnScaledDotProductAttention(
            cudnn_handle,
            q_desc.get(), q_data,    // Q: GGML [D,S,H,B] read as cuDNN [B,H,S,D] via strides
            k_desc.get(), k_data,    // K: GGML [D,S,H,B] read as cuDNN [B,H,S,D] via strides
            v_desc.get(), v_data,    // V: GGML [D,S,H,B] read as cuDNN [B,H,S,D] via strides
            mask_data ? mask_desc.get() : nullptr, mask_data,  // mask: physically transposed
            0.0f,                    // dropout probability
            is_causal,               // causal attention flag
            scale,                   // attention scale factor
            workspace, workspace_size,
            out_desc.get(), KQV->data  // O: cuDNN [B,H,S,D] written as GGML [D,H,S,B] via strides!
        );

        CUDA_CHECK(cudaGetLastError());

        // Cleanup workspace
        if (workspace) {
            CUDA_CHECK(cudaFree(workspace));
        }

        CUDA_CHECK(cudaGetLastError());


        if (status != CUDNN_STATUS_SUCCESS) {
            GGML_LOG_ERROR("cudnnScaledDotProductAttention failed: %s\n", cudnnGetErrorString(status));
            ok = false;
        } else {
            GGML_DL_FATTN_DEBUG_PRINT("cudnnScaledDotProductAttention succeeded, status : %s\n", cudnnGetErrorString(status));
            // Print first 10 elements of output after compute
            const int print_n = 10;
            const int64_t o_elems = D_q * H_q * S_q * B;
            debug_print_first_n_device(KQV->data, o_elems, print_n, KQV->type, "O(post)");
        }
    }
}
#endif

// ============================================================================
// Flash Attention DLDNN Implementation - Public Interface
// ============================================================================

namespace ggml_dl {

bool flash_attn_dldnn_available(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    const char *env_force_no_dlfa = getenv("GGML_FORCE_NO_DLFA");
    if (env_force_no_dlfa != nullptr && strcmp(env_force_no_dlfa, "1") == 0) {
        return false;
    }

    const struct ggml_tensor * Q = dst->src[0];
    const struct ggml_tensor * K = dst->src[1];
    const struct ggml_tensor * V = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    // Extract parameters
    const int64_t hsk = Q->ne[0];       // head size for K/Q
    // const int64_t hsv = V->ne[0];       // head size for V
    // const int64_t kv  = K->ne[1];       // sequence length
    // const int64_t nb  = Q->ne[3];       // batch size

    const int64_t n_head_q = Q->ne[2];  // Number of query heads
    const int64_t n_head_k = K->ne[2];  // Number of key heads
    const int64_t n_head_v = V->ne[2];  // Number of value heads

    const int64_t gqa_ratio = n_head_k > 0 ? n_head_q / n_head_k : 1;
    // const bool has_gqa = (gqa_ratio > 1);
    const bool has_mask = (mask != nullptr);

    // Extract max_bias and logit_softcap from op_params
    float max_bias;
    float logit_softcap;
    memcpy(&max_bias, ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

    const bool has_alibi = (max_bias > 0.0f);
    // const bool has_softcap = (logit_softcap != 0.0f);
    ggml_flash_attn_mask_params mask_info = {
        /*.present          =*/ mask != nullptr,
        /*.is_causal        =*/ false,
        /*.window_left      =*/ -1,
        /*.window_right     =*/ -1,
        /*.per_token_window =*/ false,
        /*.multi_sequence   =*/ false,
        /*.has_alibi_bias   =*/ false,
    };
    const bool have_mask_params = ggml_flash_attn_ext_get_mask_params(dst, &mask_info);

    // Check for GQA (Grouped Query Attention) support
    if (n_head_q != n_head_k || n_head_k != n_head_v) {
        // This is GQA configuration - now supported by SDK
        GGML_DL_FATTN_DEBUG_PRINT("DLDNN Flash Attention: GQA configuration detected (Q heads: %ld, K heads: %ld, V heads: %ld, ratio: %ld)\n",
                      n_head_q, n_head_k, n_head_v, gqa_ratio);
        // Verify that head counts are compatible
        if (n_head_q % n_head_k != 0 || n_head_k != n_head_v) {
            GGML_LOG_WARN("DLDNN Flash Attention: Invalid GQA configuration - Q heads must be divisible by K heads, and K/V heads must match\n");
            return false;
        }
    }

    if (!has_alibi) {
        if (!has_mask) {
            // Nothing special to encode, allowed.
        } else if (!have_mask_params || !mask_info.present) {
            GGML_DL_FATTN_DEBUG_PRINT("mask metadata missing or not present\n");
            return false;
        } else if (mask_info.per_token_window || mask_info.multi_sequence) {
            GGML_DL_FATTN_DEBUG_PRINT(
                "mask metadata unsupported (per_token_window=%d, multi_sequence=%d).\n",
                mask_info.per_token_window, mask_info.multi_sequence);
            return false;
        }
    }

    // Check basic requirements
    if (hsk > 288) { // adapt from flash-attn
        GGML_LOG_WARN("DLDNN is not available for ne[0] %ld\n", hsk);
        return false;
    }

    return true;
}

void flash_attn_ext_dldnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {

    ggml_cuda_set_device(ctx.device);

    cudnnHandle_t cudnn_handle = ctx.cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn_handle, ctx.stream()));

    float max_bias;
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));

    bool has_alibi = (max_bias > 0.0f);
    if (has_alibi) {
        flash_attn_ext_dldnn_mha_forward_for_alibi(ctx, dst);
    } else {
        flash_attn_ext_dldnn_mha_forward_for_mask(ctx, dst);
    }
}

} // namespace ggml_dl

#endif // GGML_USE_DLFA
