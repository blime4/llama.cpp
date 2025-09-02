#include <cmath>
#include <cstdio>
#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-mma-f16.cuh"
#include "fattn-tile-f16.cuh"
#include "fattn-tile-f32.cuh"
#include "fattn-vec-f16.cuh"
#include "fattn-vec-f32.cuh"
#include "fattn-wmma-f16.cuh"
#include "fattn.cuh"
#include "ggml-impl.h"
#include "ggml.h"

#if defined(GGML_USE_DLFA)
#include <dldnn_ext.h>
#include <unordered_map>
#include <memory>
#include <mutex>
#include <cudnn.h>

template<typename T>
static __global__ void permute_3120_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3
) {
    /* Permutation pattern: 3120
     * Input:  [dim_0][dim_1][dim_2][dim_3]
     * Output: [dim_3][dim_1][dim_2][dim_0]
     */
    const size_t total_elements = static_cast<size_t>(dim_0) * dim_1 * dim_2 * dim_3;
    size_t idx = blockIdx.y * blockDim.y + threadIdx.y;
    idx = idx * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= total_elements) return;

    // calculate original DSHB coordinates
    const int i0 = idx % dim_0;
    const int i1 = (idx / dim_0) % dim_1;
    const int i2 = (idx / (dim_0 * dim_1)) % dim_2;
    const int i3 = idx / (dim_0 * dim_1 * dim_2);

    // calculate BSHD index (prevent overflow)
    const size_t out_idx =
        static_cast<size_t>(i3) * (dim_1 * dim_2 * dim_0) +
        static_cast<size_t>(i1) * (dim_2 * dim_0) +
        static_cast<size_t>(i2) * dim_0 +
        i0;

    // Copy element by element (no vectorization)
    output[out_idx] = input[idx];
}

template<typename T>
static __host__ void call_permute_3120_kernel(
    const void* input,
    void* output,
    int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3,
    cudaStream_t stream = 0
) {
    if (dim_0 <= 0 || dim_1 <= 0 || dim_2 <= 0 || dim_3 <= 0) return;

    // 2D thread block: Adapts to seq_len and batch_size
    struct dim3 block_dim(16, 16);  // 256 threads
    struct dim3 grid_dim(
        (dim_1 + block_dim.x - 1) / block_dim.x,
        (dim_3 + block_dim.y - 1) / block_dim.y
    );

    permute_3120_kernel<T><<<grid_dim, block_dim, 0, stream>>>(
        static_cast<const T*>(input),
        static_cast<T*>(output),
        dim_0, dim_1, dim_2, dim_3
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}

template<typename T>
static __global__ void permute_3210_kernel(
    const T* __restrict__ input,
    T* __restrict__ output,
    const int64_t dim_0,
    const int64_t dim_1,
    const int64_t dim_2,
    const int64_t dim_3
) {
    // 1D thread block: Adapts to dim_3
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total_elements = dim_0 * dim_1 * dim_2 * dim_3;

    if (idx >= total_elements) return;

    // Decompose input index [dim_0][dim_1][dim_2][dim_3]
    const int64_t i3 = idx % dim_3;
    const int64_t i2 = (idx / dim_3) % dim_2;
    const int64_t i1 = (idx / (dim_3 * dim_2)) % dim_1;
    const int64_t i0 = idx / (dim_3 * dim_2 * dim_1);

    const int64_t out_idx =
        i3 * (dim_1 * dim_2 * dim_0) +
        i1 * (dim_2 * dim_0) +
        i2 * dim_0 +
        i0;

    // Copy element by element (no vectorization)
    output[out_idx] = input[idx];
}

template<typename T>
static __host__ void call_permute_3210_kernel(
    const void* input,
    void* output,
    int64_t dim_0,
    int64_t dim_1,
    int64_t dim_2,
    int64_t dim_3,
    cudaStream_t stream = 0
) {
    if (dim_0 <= 0 || dim_1 <= 0 || dim_2 <= 0 || dim_3 <= 0) return;

    // 1D thread block: Adapts to dim_3
    const int block_size = 256;  // fixed thread block size
    const int64_t total_elements = dim_0 * dim_1 * dim_2 * dim_3;
    const int grid_size = (total_elements + block_size - 1) / block_size;

    permute_3210_kernel<T><<<grid_size, block_size, 0, stream>>>(
        static_cast<const T*>(input),
        static_cast<T*>(output),
        dim_0, dim_1, dim_2, dim_3
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    }
}
#endif

template <int DKQ, int DV, int ncols2>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const ggml_tensor * Q = dst->src[0];

    if constexpr (ncols2 <= 8) {
        if (Q->ne[1] <= 8/ncols2) {
            ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 8/ncols2, ncols2>(ctx, dst);
            return;
        }
    }

    if (Q->ne[1] <= 16/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 16/ncols2, ncols2>(ctx, dst);
        return;
    }

    if (ggml_cuda_highest_compiled_arch(cc) == GGML_CUDA_CC_TURING || Q->ne[1] <= 32/ncols2) {
        ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 32/ncols2, ncols2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_case<DKQ, DV, 64/ncols2, ncols2>(ctx, dst);
}

template <int DKQ, int DV>
static void ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * mask = dst->src[3];

    float max_bias = 0.0f;
    memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

    const bool use_gqa_opt = mask && max_bias == 0.0f;

    GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
    const int gqa_ratio = Q->ne[2] / K->ne[2];

    if (use_gqa_opt && gqa_ratio % 8 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 4 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 4>(ctx, dst);
        return;
    }

    if (use_gqa_opt && gqa_ratio % 2 == 0) {
        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);
        return;
    }

    ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 1>(ctx, dst);
}

static void ggml_cuda_flash_attn_ext_mma_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    switch (Q->ne[0]) {
        case 64:
            GGML_ASSERT(V->ne[0] == 64);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 64,  64>(ctx, dst);
            break;
        case 80:
            GGML_ASSERT(V->ne[0] == 80);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 80,  80>(ctx, dst);
            break;
        case 96:
            GGML_ASSERT(V->ne[0] == 96);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2< 96,  96>(ctx, dst);
            break;
        case 112:
            GGML_ASSERT(V->ne[0] == 112);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<112, 112>(ctx, dst);
            break;
        case 128:
            GGML_ASSERT(V->ne[0] == 128);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<128, 128>(ctx, dst);
            break;
        case 256:
            GGML_ASSERT(V->ne[0] == 256);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols2<256, 256>(ctx, dst);
            break;
        case 576: {
            // For Deepseek, go straight to the ncols1 switch to avoid compiling unnecessary kernels.
            GGML_ASSERT(V->ne[0] == 512);
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) KQV->op_params + 1, sizeof(float));

            const bool use_gqa_opt = mask && max_bias == 0.0f;
            GGML_ASSERT(use_gqa_opt);

            GGML_ASSERT(Q->ne[2] % K->ne[2] == 0);
            const int gqa_ratio = Q->ne[2] / K->ne[2];
            GGML_ASSERT(gqa_ratio % 16 == 0);
            ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<576, 512, 16>(ctx, dst);
        } break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

#define FATTN_VEC_F16_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f16_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f16(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16 )

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F16_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F16_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F16_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(Q->ne[0]);
}

#define FATTN_VEC_F32_CASE(D, type_K, type_V)                               \
    if (Q->ne[0] == (D) && K->type == (type_K) && V->type == (type_V)) {    \
        ggml_cuda_flash_attn_ext_vec_f32_case<D, type_K, type_V>(ctx, dst); \
        return;                                                             \
    }                                                                       \

static void ggml_cuda_flash_attn_ext_vec_f32(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * Q = dst->src[0];
    ggml_tensor * K = dst->src[1];
    ggml_tensor * V = dst->src[2];

#ifdef GGML_CUDA_FA_ALL_QUANTS
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q4_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q4_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q5_1)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q5_1)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q5_1, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16,  GGML_TYPE_F16)

    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#else
    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q4_0, GGML_TYPE_Q4_0)

    FATTN_VEC_F32_CASE(128, GGML_TYPE_Q8_0, GGML_TYPE_Q8_0)

    FATTN_VEC_F32_CASE( 64, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(128, GGML_TYPE_F16, GGML_TYPE_F16)
    FATTN_VEC_F32_CASE(256, GGML_TYPE_F16, GGML_TYPE_F16)
#endif // GGML_CUDA_FA_ALL_QUANTS

    on_no_fattn_vec_case(Q->ne[0]);
}

#if defined(GGML_USE_DLFA)
// cuDNN handle management
namespace {
    // Thread-safe cache for cuDNN handles per device
    std::unordered_map<int, cudnnHandle_t> handle_cache;
    std::mutex handle_cache_mutex;

    cudnnHandle_t get_cached_cudnn_handle(int device_id) {
        std::lock_guard<std::mutex> lock(handle_cache_mutex);
        auto it = handle_cache.find(device_id);
        if (it != handle_cache.end()) {
            return it->second;
        }

        cudnnHandle_t handle;
        cudnnStatus_t status = cudnnCreate(&handle);
        if (status != CUDNN_STATUS_SUCCESS) {
            GGML_LOG_ERROR("Failed to create cuDNN handle for device %d: %s\n",
                          device_id, cudnnGetErrorString(status));
            return nullptr;
        }

        handle_cache[device_id] = handle;
        return handle;
    }

    void destroy_cudnn_handles() {
        std::lock_guard<std::mutex> lock(handle_cache_mutex);
        GGML_LOG_DEBUG("Destroying %zu cuDNN handles\n", handle_cache.size());
        for (auto& pair : handle_cache) {
            if (pair.second != nullptr) {
                GGML_LOG_DEBUG("Destroying cuDNN handle for device %d\n", pair.first);

                // Additional safety check: verify handle address is reasonable
                if ((uintptr_t)pair.second < 0x10000 || (uintptr_t)pair.second > 0xFFFFFFFFFFFFFFFFULL) {
                    GGML_LOG_WARN("cuDNN handle for device %d has suspicious address: %p, skipping\n",
                                 pair.first, pair.second);
                    continue;
                }

                // Try to destroy the handle
                cudnnStatus_t status = cudnnDestroy(pair.second);
                if (status != CUDNN_STATUS_SUCCESS) {
                    GGML_LOG_WARN("Failed to destroy cuDNN handle for device %d: %s\n",
                                 pair.first, cudnnGetErrorString(status));
                } else {
                    GGML_LOG_DEBUG("Successfully destroyed cuDNN handle for device %d\n", pair.first);
                }
            } else {
                GGML_LOG_WARN("cuDNN handle for device %d is null\n", pair.first);
            }
        }
        handle_cache.clear();
        GGML_LOG_DEBUG("cuDNN handle cleanup completed\n");
    }
}

// Cleanup function to be called at program exit
static void cleanup_cudnn_handles() {
    // Skip cuDNN cleanup entirely at program exit to prevent crashes
    // The operating system will clean up resources automatically
    GGML_LOG_DEBUG("Skipping cuDNN handle cleanup at program exit to prevent crashes\n");

    // Just clear the cache without destroying handles
    std::lock_guard<std::mutex> lock(handle_cache_mutex);
    handle_cache.clear();
}

// Register cleanup function
static int cleanup_registered = 0;
static void register_cleanup() {
    if (!cleanup_registered) {
        std::atexit(cleanup_cudnn_handles);
        cleanup_registered = 1;
    }
}

#define CUDNN_CHECK(x) do { \
    cudnnStatus_t status = (x); \
    if (status != CUDNN_STATUS_SUCCESS) { \
        GGML_LOG_ERROR("cuDNN error: %s\n", cudnnGetErrorString(status)); \
        GGML_ASSERT(false); \
    } \
} while(0)

// Get cuDNN handle for current device
static cudnnHandle_t getCudnnHandle() {
    // Register cleanup function on first call
    register_cleanup();

    int device_id;
    cudaError_t err = cudaGetDevice(&device_id);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("Failed to get current CUDA device: %s\n", cudaGetErrorString(err));
        return nullptr;
    }

    cudnnHandle_t handle = get_cached_cudnn_handle(device_id);
    if (handle == nullptr) {
        return nullptr;
    }

    // Set the current CUDA stream
    cudaStream_t stream = 0;
    CUDNN_CHECK(cudnnSetStream(handle, stream));

    return handle;
}

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

struct GGMLTensorDescriptor {
    cudnnTensorDescriptor_t desc;

    GGMLTensorDescriptor() {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&desc));
    }

    ~GGMLTensorDescriptor() {
        CUDNN_CHECK(cudnnDestroyTensorDescriptor(desc));
    }

    void set_from_ggml_tensor(const struct ggml_tensor* tensor) const {
        // adapt from SW/dl_gpgpu_alg/dlLibTest.git use cudnnSetTensorNdDescriptor to set Descriptor.
        // BSHD format.
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(tensor->type);
        int dims[4] = {(int)tensor->ne[3], (int)tensor->ne[1], (int)tensor->ne[2], (int)tensor->ne[0]};
        // [batch, seq_len, heads, head_dim]
        int strides[4] = {
            (int)(tensor->nb[3] / sizeof(data_type)),  // batch stride
            (int)(tensor->nb[1] / sizeof(data_type)),  // seq stride
            (int)(tensor->nb[2] / sizeof(data_type)),  // head stride
            (int)(tensor->nb[0] / sizeof(data_type))   // head_dim stride
        };

        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    void set_from_dims(int dim_0, int dim_1, int dim_2, int dim_3, enum ggml_type type) const {
        // adapt from SW/dl_gpgpu_alg/dlLibTest.git use cudnnSetTensorNdDescriptor to set Descriptor.
        // BSHD format.
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(type);
        int dims[4] = {dim_0, dim_1, dim_2, dim_3};
        int strides[4] = {
            dim_1 * dim_2 * dim_3,
            dim_2 * dim_3,
            dim_3,
            1
        };
        CUDNN_CHECK(cudnnSetTensorNdDescriptor(
            desc,
            data_type,
            4,
            dims,
            strides
        ));
    }

    void set_from_dims_3d(int dim_0, int dim_1, int dim_2, enum ggml_type type) const {
        // Set 3D tensor descriptor for ALiBi slopes: [batch_size, 1, num_heads]
        cudnnDataType_t data_type = ggml_type_to_cudnn_type(type);
        int dims[3] = {dim_0, dim_1, dim_2};
        int strides[3] = {
            dim_1 * dim_2,  // batch stride
            dim_2,         // seq_len stride (should be 1 for ALiBi)
            1             // num_heads stride
        };

        // For debugging: print the dimensions and strides
        GGML_LOG_INFO("Setting 3D descriptor: dims=[%d, %d, %d], strides=[%d, %d, %d]\n",
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


// --- CUDA Type Conversion (ggml_cuda_convert_tensor_data) ---

// Forward declaration for type conversion kernels
template <typename S, typename D> __global__ void convert_tensor_kernel(const S* src, D* dst, size_t n);
template <typename S, typename D> __global__ void convert_tensor_nc_kernel(const S* src, D* dst, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int64_t s01, int64_t s02, int64_t s03);

// Helper to get conversion kernel function pointer
// TODO: Extend this to support all desired source and destination types
template <typename DST_TYPE>
static void (*get_to_fp_cuda(enum ggml_type src_type))(const void* src, DST_TYPE* dst, size_t n, cudaStream_t stream) {
    switch (src_type) {
        case GGML_TYPE_F32: return (void (*)(const void*, DST_TYPE*, size_t, cudaStream_t)) convert_tensor_kernel<float, DST_TYPE>;
        case GGML_TYPE_F16: return (void (*)(const void*, DST_TYPE*, size_t, cudaStream_t)) convert_tensor_kernel<ggml_fp16_t, DST_TYPE>;
        case GGML_TYPE_BF16: return (void (*)(const void*, DST_TYPE*, size_t, cudaStream_t)) convert_tensor_kernel<ggml_bf16_t, DST_TYPE>;
        // Add other quantized types as needed (e.g., Q4_0, Q4_1, Q5_0, Q5_1, Q8_0)
        default: return nullptr;
    }
}

template <typename DST_TYPE>
static void (*get_to_fp_nc_cuda(enum ggml_type src_type))(const void* src, DST_TYPE* dst, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int64_t s01, int64_t s02, int64_t s03, cudaStream_t stream) {
    switch (src_type) {
        case GGML_TYPE_F32: return (void (*)(const void*, DST_TYPE*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t)) convert_tensor_nc_kernel<float, DST_TYPE>;
        case GGML_TYPE_F16: return (void (*)(const void*, DST_TYPE*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t)) convert_tensor_nc_kernel<ggml_fp16_t, DST_TYPE>;
        case GGML_TYPE_BF16: return (void (*)(const void*, DST_TYPE*, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t)) convert_tensor_nc_kernel<ggml_bf16_t, DST_TYPE>;
        // TODO: Add other quantized types as needed
        default: return nullptr;
    }
}

// Generic type conversion kernel for contiguous tensors
template <typename S, typename D>
__global__ void convert_tensor_kernel(const S* src, D* dst, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = (D)src[i];
    }
}

// Generic type conversion kernel for non-contiguous tensors
template <typename S, typename D>
__global__ void convert_tensor_nc_kernel(const S* src, D* dst, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3, int64_t s01, int64_t s02, int64_t s03) {
    int64_t i0 = threadIdx.x + blockIdx.x * blockDim.x;
    int64_t i1 = threadIdx.y + blockIdx.y * blockDim.y;
    int64_t i2 = threadIdx.z + blockIdx.z * blockDim.z;

    if (i0 < ne0 && i1 < ne1 && i2 < ne2) {
        const S* src_row = src + i0 + i1 * s01 + i2 * s02;
        D* dst_row = dst + i0 + i1 * ne0 + i2 * ne0 * ne1;
        // TODO: Handle ne3 (batch size) if needed.
        // This is a simplified 3D conversion, might need extension for 4D or custom strides.
        *dst_row = (D)*src_row;
    }
}

// ggml_cuda_convert_tensor_data implementation
void ggml_cuda_convert_tensor_data(
    const void* src_data,
    void* dst_data,
    const ggml_tensor* src_tensor,
    enum ggml_type dst_type,
    cudaStream_t stream
) {
    // Dimensions of the source tensor
    const int64_t ne0 = src_tensor->ne[0];
    const int64_t ne1 = src_tensor->ne[1];
    const int64_t ne2 = src_tensor->ne[2];
    const int64_t ne3 = src_tensor->ne[3];
    const enum ggml_type src_type = src_tensor->type;

    const size_t n_elements = ne0 * ne1 * ne2 * ne3;

    // Check if tensor is contiguously allocated
    const bool is_contiguous = ggml_is_contiguously_allocated(src_tensor);

    // Get conversion function based on destination type
    if (dst_type == GGML_TYPE_F16) {
        if (is_contiguous) {
            to_fp16_cuda_t convert_func = ggml_get_to_fp16_cuda(src_type);
            GGML_ASSERT(convert_func != nullptr && "No F16 conversion kernel for source type");
            convert_func(src_data, (__half*)dst_data, n_elements, stream);
        } else {
            to_fp16_nc_cuda_t convert_func = ggml_get_to_fp16_nc_cuda(src_type);
            GGML_ASSERT(convert_func != nullptr && "No F16 non-contiguous conversion kernel for source type");
            const int64_t ts = ggml_type_size(src_type);
            const int64_t s01 = src_tensor->nb[1] / ts;
            const int64_t s02 = src_tensor->nb[2] / ts;
            const int64_t s03 = src_tensor->nb[3] / ts;
            convert_func(src_data, (__half*)dst_data, ne0, ne1, ne2, ne3, s01, s02, s03, stream);
        }
    } else if (dst_type == GGML_TYPE_F32) {
        if (is_contiguous) {
            to_fp32_cuda_t convert_func = ggml_get_to_fp32_cuda(src_type);
            GGML_ASSERT(convert_func != nullptr && "No F32 conversion kernel for source type");
            convert_func(src_data, (float*)dst_data, n_elements, stream);
        } else {
            to_fp32_nc_cuda_t convert_func = ggml_get_to_fp32_nc_cuda(src_type);
            GGML_ASSERT(convert_func != nullptr && "No F32 non-contiguous conversion kernel for source type");
            const int64_t ts = ggml_type_size(src_type);
            const int64_t s01 = src_tensor->nb[1] / ts;
            const int64_t s02 = src_tensor->nb[2] / ts;
            const int64_t s03 = src_tensor->nb[3] / ts;
            convert_func(src_data, (float*)dst_data, ne0, ne1, ne2, ne3, s01, s02, s03, stream);
        }
    } else if (dst_type == GGML_TYPE_BF16) {
        if (is_contiguous) {
            to_bf16_cuda_t convert_func = ggml_get_to_bf16_cuda(src_type);
            GGML_ASSERT(convert_func != nullptr && "No BF16 conversion kernel for source type");
            convert_func(src_data, (__nv_bfloat16*)dst_data, n_elements, stream);
        } else {
            to_bf16_nc_cuda_t convert_func = ggml_get_to_bf16_nc_cuda(src_type);
            GGML_ASSERT(convert_func != nullptr && "No BF16 non-contiguous conversion kernel for source type");
            const int64_t ts = ggml_type_size(src_type);
            const int64_t s01 = src_tensor->nb[1] / ts;
            const int64_t s02 = src_tensor->nb[2] / ts;
            const int64_t s03 = src_tensor->nb[3] / ts;
            convert_func(src_data, (__nv_bfloat16*)dst_data, ne0, ne1, ne2, ne3, s01, s02, s03, stream);
        }
    } else {
        GGML_ABORT("Unsupported destination type for ggml_cuda_convert_tensor_data");
    }
}


// Helper function to expand ALiBi slopes to 3D format for cuDNN
// Similar to expandTo3D in flash-attention
static void* expand_alibi_slopes_to_3d(
    const struct ggml_tensor* mask,
    float max_bias,
    const uint32_t n_head,
    const uint32_t n_head_log2,
    const float m0,
    const float m1,
    enum ggml_type target_type,
    cudaStream_t stream
) {
    if (mask == nullptr || max_bias <= 0.0f) {
        return nullptr;
    }

    // Generate ALiBi slopes and expand to 3D format for cuDNN
    GGML_LOG_INFO("Generating ALiBi slopes: max_bias=%f, n_head=%d, n_head_log2=%d, m0=%f, m1=%f\n",
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
    CUDA_CHECK(cudaMalloc(&slopes_gpu, slopes_size));

    // Copy slopes to GPU
    CUDA_CHECK(cudaMemcpyAsync(slopes_gpu, slopes_cpu, slopes_size, cudaMemcpyHostToDevice, stream));

    // Free CPU memory
    free(slopes_cpu);

    return slopes_gpu;
}

static bool ggml_cuda_flash_attn_ext_dldnn_available(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    GGML_UNUSED(ctx);

    const struct ggml_tensor * Q = dst->src[0];
    const struct ggml_tensor * K = dst->src[1];
    const struct ggml_tensor * V = dst->src[2];

    // Check for GQA (Grouped Query Attention) support
    const int64_t n_head_q = Q->ne[2];  // Number of query heads
    const int64_t n_head_k = K->ne[2];  // Number of key heads
    const int64_t n_head_v = V->ne[2];  // Number of value heads

    // GQA ratio (nr2 parameter in tests)
    const int64_t gqa_ratio = n_head_k > 0 ? n_head_q / n_head_k : 1;

    // TODO: support GQA, need cudnnMHAForward support.
    // Check if this is GQA (not standard MHA)
    if (n_head_q != n_head_k || n_head_k != n_head_v) {
        // This is GQA configuration
        GGML_LOG_WARN("DLDNN Flash Attention: GQA unsupported yet. GQA configuration detected (Q heads: %ld, K heads: %ld, V heads: %ld)\n",
                      n_head_q, n_head_k, n_head_v);
        return false;
    }

    // DRAFT: The type conversion is now handled inside ggml_cuda_flash_attn_ext_dldnn
    // so we only need to check if the types are supported for cuDNN (F16, F32, BF16)
    // OR if they can be converted to these types.
    // For now, we allow any type here and let the dldnn function handle the conversion.
    // TODO: move this check and conversion to model loading stage
    //      (FYI: ggml_backend_cuda_gptq_quantize_and_store_from_cpu).

    // TODO: support logit_softcap, need cudnnMHAForward support.
    // Check for logit_softcap support
    float logit_softcap;
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));
    if (logit_softcap != 0.0f) {
        GGML_LOG_WARN("DLDNN Flash Attention: logit_softcap (%.3f) is not supported. Falling back to standard implementation.\n", logit_softcap);
        return false;
    }

    // Check basic requirements
    if (Q->ne[0] > 288) { // adapt from flash-attn
        GGML_LOG_WARN("DLDNN is not available for ne[0] %d\n", (int)Q->ne[0]);
        return false;
    }
    return true;
}


// MHA Forward implementation for ALiBi support
static void ggml_cuda_flash_attn_ext_dldnn_mha_forward(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    bool ok = true;
    const struct ggml_tensor * KQV  = dst;
    const struct ggml_tensor * Q    = dst->src[0];
    const struct ggml_tensor * K    = dst->src[1];
    const struct ggml_tensor * V    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    cudnnHandle_t cudnn_handle = getCudnnHandle();
    if (cudnn_handle == nullptr) {
        GGML_LOG_ERROR("Failed to get cuDNN handle for MHA Forward\n");
        return;
    }

    // Determine target_type based on Q, K, V
    enum ggml_type target_type = GGML_TYPE_F16; // Default to F16
    if (Q->type == GGML_TYPE_F32 || K->type == GGML_TYPE_F32 || V->type == GGML_TYPE_F32) {
        target_type = GGML_TYPE_F32;
    } else if (Q->type == GGML_TYPE_BF16 || K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16) {
        target_type = GGML_TYPE_BF16;
    }

    // Pointers to the actual data, either original or converted
    const void* q_data_source = Q->data;
    const void* k_data_source = K->data;
    const void* v_data_source = V->data;

    // Temporary buffers for converted Q, K, V data if needed
    void* q_converted_gpu = nullptr;
    void* k_converted_gpu = nullptr;
    void* v_converted_gpu = nullptr;

    // Convert Q to target_type if its type differs
    if (Q->type != target_type) {
        GGML_LOG_INFO("Converting Q from %s to %s for DLDNN attention.\n", ggml_type_name(Q->type),
        ggml_type_name(target_type));
        size_t q_converted_size = Q->ne[0] * Q->ne[1] * Q->ne[2] * Q->ne[3] * ggml_type_size
        (target_type);
        CUDA_CHECK(cudaMalloc(&q_converted_gpu, q_converted_size));
        ggml_cuda_convert_tensor_data(Q->data, q_converted_gpu, Q, target_type, ctx.stream());
        q_data_source = q_converted_gpu;
    }

    // Convert K to target_type if its type differs
    if (K->type != target_type) {
        GGML_LOG_INFO("Converting K from %s to %s for DLDNN attention.\n", ggml_type_name(K->type),
        ggml_type_name(target_type));
        size_t k_converted_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * ggml_type_size
        (target_type);
        CUDA_CHECK(cudaMalloc(&k_converted_gpu, k_converted_size));
        ggml_cuda_convert_tensor_data(K->data, k_converted_gpu, K, target_type, ctx.stream());
        k_data_source = k_converted_gpu;
    }

    // Convert V to target_type if its type differs
    if (V->type != target_type) {
        GGML_LOG_INFO("Converting V from %s to %s for DLDNN attention.\n", ggml_type_name(V->type),
        ggml_type_name(target_type));
        size_t v_converted_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * ggml_type_size
        (target_type);
        CUDA_CHECK(cudaMalloc(&v_converted_gpu, v_converted_size));
        ggml_cuda_convert_tensor_data(V->data, v_converted_gpu, V, target_type, ctx.stream());
        v_data_source = v_converted_gpu;
    }

    GGMLTensorDescriptor q_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor k_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor v_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor temp_out_desc = GGMLTensorDescriptor();
    GGMLTensorDescriptor alibi_slopes_desc = GGMLTensorDescriptor();

    // Convert Q, K, V data from DHSB to BSHD format for cuDNN
    void* q_bshd = nullptr;
    void* k_bshd = nullptr;
    void* v_bshd = nullptr;

    // Allocate temporary buffers for BSHD format data, using target_type for size
    size_t q_bshd_size = Q->ne[0] * Q->ne[1] * Q->ne[2] * Q->ne[3] * ggml_type_size(target_type);
    size_t k_bshd_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * ggml_type_size(target_type);
    size_t v_bshd_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * ggml_type_size(target_type);

    CUDA_CHECK(cudaMalloc(&q_bshd, q_bshd_size));
    CUDA_CHECK(cudaMalloc(&k_bshd, k_bshd_size));
    CUDA_CHECK(cudaMalloc(&v_bshd, v_bshd_size));

    // Convert data from DSHB -> BSHD format using the correct source pointers and target_type
    auto convert_to_bshd = [](const void* input, void* output, int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3, enum ggml_type type) -> bool {
        switch (type) {
            case GGML_TYPE_F16:
                call_permute_3120_kernel<ggml_fp16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            case GGML_TYPE_F32:
                call_permute_3120_kernel<float>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            case GGML_TYPE_BF16:
                call_permute_3120_kernel<ggml_bf16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            default:
                // This case should ideally not be reached if type conversion is handled prior to this.
                // If it is reached, it means an unsupported type made it through for permutation.
                GGML_LOG_ERROR("Unsupported data type %s for permute_3120_kernel.\n", ggml_type_name(type));
                return false;
        }
    };

    if (!convert_to_bshd(q_data_source, q_bshd, Q->ne[0], Q->ne[1], Q->ne[2], Q->ne[3], target_type)) {
        GGML_LOG_ERROR("Failed to permute Q data to BSHD format with type %s.\n", ggml_type_name(target_type));
        ok = false;
    }
    if (ok && !convert_to_bshd(k_data_source, k_bshd, K->ne[0], K->ne[1], K->ne[2], K->ne[3], target_type)) {
        GGML_LOG_ERROR("Failed to permute K data to BSHD format with type %s.\n", ggml_type_name(target_type));
        ok = false;
    }
    if (ok && !convert_to_bshd(v_data_source, v_bshd, V->ne[0], V->ne[1], V->ne[2], V->ne[3], target_type)) {
        GGML_LOG_ERROR("Failed to permute V data to BSHD format with type %s.\n", ggml_type_name(target_type));
        ok = false;
    }

    if (ok) {
    // TODO : DLDNN impl this. return the DHSB format output directly.
    // create temp_output tensor
    // DLDNN expected format: BSHD
    int64_t temp_ne[4] = {
        Q->ne[3], // batch_size
        Q->ne[1], // seq_len
        Q->ne[2], // num_heads
        V->ne[0], // head_dim
    };

    // allocate temp_output tensor on GPU using the determined target_type
    void* temp_output = nullptr;
    size_t temp_output_size = temp_ne[0] * temp_ne[1] * temp_ne[2] * temp_ne[3] * ggml_type_size(target_type);
    CUDA_CHECK(cudaMalloc(&temp_output, temp_output_size));

    // Use the determined target_type for descriptors
    enum ggml_type data_type = target_type; // Use the determined target_type

    // Set all descriptors to BSHD format for consistency
    // BSHD: (batch_size, seq_len, num_heads, head_dim)
    q_desc.set_from_dims(Q->ne[3], Q->ne[1], Q->ne[2], Q->ne[0], data_type);
    k_desc.set_from_dims(K->ne[3], K->ne[1], K->ne[2], K->ne[0], data_type);
    v_desc.set_from_dims(V->ne[3], V->ne[1], V->ne[2], V->ne[0], data_type);
    temp_out_desc.set_from_dims(temp_ne[0], temp_ne[1], temp_ne[2], temp_ne[3], data_type);

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
        mask, max_bias, n_head, n_head_log2, m0, m1, target_type, ctx.stream()
    );


    // Set alibi_slopes_desc for 3D format: [1, 1, num_heads]
    // cuDNN expects: [1, 1, num_heads] where first dimension must be 1
    if (alibi_slopes_ptr != nullptr) {
        alibi_slopes_desc.set_from_dims_3d(1, 1, n_head, GGML_TYPE_F32);
    }

    // - alibi_slopes_desc : Now properly set for 3D ALiBi slopes
    // - softmax_lse_desc : return_softmax is false, pass nullptr.
    // - p_desc : return_softmax is false, pass nullptr.

    CUDNN_CHECK(cudnnGetMHAForwardWorkspaceSize(
        cudnn_handle, q_desc.get(), k_desc.get(), v_desc.get(),
        alibi_slopes_ptr != nullptr ? alibi_slopes_desc.get() : nullptr, // Pass nullptr if no ALiBi
        temp_out_desc.get(), nullptr, nullptr,
        0.0f, scale,
        false, -1, -1,
        false,
        &workspace_size
    ));

    void* workspace = nullptr;
    if (workspace_size > 0) {
        CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
    }

    unsigned long long philox_seed = 0;
    unsigned long long philox_offset = 0;

    CUDNN_CHECK(cudnnMHAForward(
        cudnn_handle, q_desc.get(), q_bshd,
        k_desc.get(), k_bshd, v_desc.get(), v_bshd,
        alibi_slopes_ptr != nullptr ? alibi_slopes_desc.get() : nullptr,
        alibi_slopes_ptr, // This will be nullptr if no ALiBi
        temp_out_desc.get(), temp_output,
        nullptr, nullptr,
        nullptr, nullptr,
        0.0f, scale,
        false, -1, -1,
        false,
        &philox_seed, &philox_offset,
        workspace, workspace_size
    ));

    if (workspace != nullptr) {
        CUDA_CHECK(cudaFree(workspace));
    }

    // permute temp_output to final KQV format
    // from BSHD - DLDNN output format
    // to DHSB   - llama.cpp expected format

    // call permute kernel according to data_type - use 3210 permute for DHSB format
    auto convert_bhsd_to_dhsb = [](const void* input, void* output, int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3, enum ggml_type type) -> bool {
        switch (type) {
            case GGML_TYPE_F16:
                call_permute_3210_kernel<ggml_fp16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            case GGML_TYPE_F32:
                call_permute_3210_kernel<float>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            case GGML_TYPE_BF16:
                call_permute_3210_kernel<ggml_bf16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            default:
                return false;
        }
    };

    if (!convert_bhsd_to_dhsb(temp_output, KQV->data, temp_ne[0], temp_ne[1], temp_ne[2], temp_ne[3], data_type)) {
        GGML_LOG_ERROR("Unsupported data type for permute: %d\n", data_type);
        // free temp_output
        if (temp_output != nullptr) {
            CUDA_CHECK(cudaFree(temp_output));
        }
        ok = false;
    }

    if (ok) {
        // check kernel execution
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // free temp_output
    if (temp_output != nullptr) {
        CUDA_CHECK(cudaFree(temp_output));
    }

    // Clean up temporary BSHD format data
    if (q_bshd != nullptr) {
        CUDA_CHECK(cudaFree(q_bshd));
    }
    if (k_bshd != nullptr) {
        CUDA_CHECK(cudaFree(k_bshd));
    }
    if (v_bshd != nullptr) {
        CUDA_CHECK(cudaFree(v_bshd));
    }

    // Clean up ALiBi slopes memory
    if (alibi_slopes_ptr != nullptr) {
        CUDA_CHECK(cudaFree(alibi_slopes_ptr));
    }

    if (q_converted_gpu != nullptr) {
        CUDA_CHECK(cudaFree(q_converted_gpu));
    }
    if (k_converted_gpu != nullptr) {
        CUDA_CHECK(cudaFree(k_converted_gpu));
    }
    if (v_converted_gpu != nullptr) {
        CUDA_CHECK(cudaFree(v_converted_gpu));
    }
    }
}

// ScaledDotProductAttention implementation for mask support
static void ggml_cuda_flash_attn_ext_dldnn_scaled_dot_product(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    bool ok = true;
    const struct ggml_tensor * KQV  = dst;
    const struct ggml_tensor * Q    = dst->src[0];
    const struct ggml_tensor * K    = dst->src[1];
    const struct ggml_tensor * V    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    cudnnHandle_t cudnn_handle = getCudnnHandle();
    if (cudnn_handle == nullptr) {
        GGML_LOG_ERROR("Failed to get cuDNN handle for ScaledDotProductAttention\n");
        return;
    }

    // Extract parameters
    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

    // Determine target_type based on Q, K, V
    enum ggml_type target_type = GGML_TYPE_F16; // Default to F16
    if (Q->type == GGML_TYPE_F32 || K->type == GGML_TYPE_F32 || V->type == GGML_TYPE_F32) {
        target_type = GGML_TYPE_F32;
    } else if (Q->type == GGML_TYPE_BF16 || K->type == GGML_TYPE_BF16 || V->type == GGML_TYPE_BF16) {
        target_type = GGML_TYPE_BF16;
    }

    // Convert Q, K, V to target type if needed
    const void* q_data_source = Q->data;
    const void* k_data_source = K->data;
    const void* v_data_source = V->data;
    void* q_converted_gpu = nullptr;
    void* k_converted_gpu = nullptr;
    void* v_converted_gpu = nullptr;

    if (Q->type != target_type) {
        size_t q_converted_size = Q->ne[0] * Q->ne[1] * Q->ne[2] * Q->ne[3] * ggml_type_size(target_type);
        CUDA_CHECK(cudaMalloc(&q_converted_gpu, q_converted_size));
        ggml_cuda_convert_tensor_data(Q->data, q_converted_gpu, Q, target_type, ctx.stream());
        q_data_source = q_converted_gpu;
    }
    if (K->type != target_type) {
        size_t k_converted_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * ggml_type_size(target_type);
        CUDA_CHECK(cudaMalloc(&k_converted_gpu, k_converted_size));
        ggml_cuda_convert_tensor_data(K->data, k_converted_gpu, K, target_type, ctx.stream());
        k_data_source = k_converted_gpu;
    }
    if (V->type != target_type) {
        size_t v_converted_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * ggml_type_size(target_type);
        CUDA_CHECK(cudaMalloc(&v_converted_gpu, v_converted_size));
        ggml_cuda_convert_tensor_data(V->data, v_converted_gpu, V, target_type, ctx.stream());
        v_data_source = v_converted_gpu;
    }

    // Convert data from DHSB to BHSD format for cudnnScaledDotProductAttention
    void* q_bhsd = nullptr;
    void* k_bhsd = nullptr;
    void* v_bhsd = nullptr;

    size_t q_bhsd_size = Q->ne[0] * Q->ne[1] * Q->ne[2] * Q->ne[3] * ggml_type_size(target_type);
    size_t k_bhsd_size = K->ne[0] * K->ne[1] * K->ne[2] * K->ne[3] * ggml_type_size(target_type);
    size_t v_bhsd_size = V->ne[0] * V->ne[1] * V->ne[2] * V->ne[3] * ggml_type_size(target_type);

    CUDA_CHECK(cudaMalloc(&q_bhsd, q_bhsd_size));
    CUDA_CHECK(cudaMalloc(&k_bhsd, k_bhsd_size));
    CUDA_CHECK(cudaMalloc(&v_bhsd, v_bhsd_size));

    auto permute_3210 = [](const void* input, void* output, int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3, enum ggml_type type) -> bool {
        switch (type) {
            case GGML_TYPE_F16:
                call_permute_3210_kernel<ggml_fp16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            case GGML_TYPE_F32:
                call_permute_3210_kernel<float>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            case GGML_TYPE_BF16:
                call_permute_3210_kernel<ggml_bf16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                return true;
            default:
                return false;
        }
    };

    // Convert to BHSD format: DSHB -> BHSD
    // Q: [head_dim, seq_len, num_heads, batch_size] -> [batch_size, num_heads, seq_len, head_dim]
    if (!permute_3210(q_data_source, q_bhsd, Q->ne[3], Q->ne[1], Q->ne[2], Q->ne[0], target_type)) {
        GGML_LOG_ERROR("Failed to permute Q data to BHSD format\n");
        ok = false;
    }
    if (ok && !permute_3210(k_data_source, k_bhsd, K->ne[3], K->ne[1], K->ne[2], K->ne[0], target_type)) {
        GGML_LOG_ERROR("Failed to permute K data to BHSD format\n");
        ok = false;
    }
    if (ok && !permute_3210(v_data_source, v_bhsd, V->ne[3], V->ne[1], V->ne[2], V->ne[0], target_type)) {
        GGML_LOG_ERROR("Failed to permute V data to BHSD format\n");
        ok = false;
    }

    if (ok) {
        // Handle GQA (Grouped Query Attention) - K and V may have fewer heads than Q
        const int64_t q_heads = Q->ne[2];
        const int64_t k_heads = K->ne[2];
        const int64_t v_heads = V->ne[2];

        // Check if this is GQA
        if (q_heads != k_heads || q_heads != v_heads) {
            GGML_LOG_INFO("GQA detected: Q heads=%ld, K heads=%ld, V heads=%ld\n", q_heads, k_heads, v_heads);

            // For cudnnScaledDotProductAttention, we need to handle GQA by expanding K and V
            // This is a limitation - cuDNN expects all tensors to have the same number of heads
            GGML_LOG_ERROR("cudnnScaledDotProductAttention doesn't support GQA directly. Need to expand K/V heads or use fallback.\n");
            ok = false;
        }

        if (ok) {
            // Create tensor descriptors
            GGMLTensorDescriptor q_desc, k_desc, v_desc, out_desc, mask_desc;

            // BHSD format: (batch_size, num_heads, seq_len, head_dim)
            q_desc.set_from_dims(Q->ne[3], Q->ne[2], Q->ne[1], Q->ne[0], target_type);
            k_desc.set_from_dims(K->ne[3], K->ne[2], K->ne[1], K->ne[0], target_type);
            v_desc.set_from_dims(V->ne[3], V->ne[2], V->ne[1], V->ne[0], target_type);

            // Allocate output buffer in BHSD format
            int64_t temp_ne[4] = { Q->ne[3], Q->ne[2], Q->ne[1], V->ne[0] }; // [batch_size, num_heads, seq_len, head_dim]
            size_t temp_output_size = temp_ne[0] * temp_ne[1] * temp_ne[2] * temp_ne[3] * ggml_type_size(target_type);
            void* temp_output = nullptr;
            CUDA_CHECK(cudaMalloc(&temp_output, temp_output_size));

            out_desc.set_from_dims(temp_ne[0], temp_ne[1], temp_ne[2], temp_ne[3], target_type);

            // Handle mask
            void* mask_dldnn = nullptr;
            bool is_causal = false;

            if (mask != nullptr) {
                // GGML mask format: [n_kv, n_batch_pad, 1, 1] = [Sk, Sq, 1, 1]
                // cuDNN expects: [B, H, Sq, Sk] , [1, 1, Sq, Sk]
                GGML_LOG_INFO("Converting mask from GGML [Sk, Sq, 1, 1] to cuDNN [1, 1, Sq, Sk] format\n");

                const int64_t mask_sk = mask->ne[0];      // n_kv (key sequence length)
                const int64_t mask_sq_pad = mask->ne[1];  // n_batch_pad (padded query sequence length)
                const int64_t actual_sq = Q->ne[1];       // actual query sequence length
                const int64_t actual_sk = K->ne[1];       // actual key sequence length

                // Verify mask dimensions match attention dimensions
                if (mask_sk != actual_sk) {
                    GGML_LOG_ERROR("Mask key dimension mismatch: mask_sk=%ld, actual_sk=%ld\n", mask_sk, actual_sk);
                    ok = false;
                } else {
                    // Allocate memory for converted mask in [1, 1, Sq, Sk] format
                    size_t mask_size = 1 * 1 * actual_sq * actual_sk * ggml_type_size(target_type);
                    CUDA_CHECK(cudaMalloc(&mask_dldnn, mask_size));

                    // Convert mask: [Sk, Sq_pad, 1, 1] -> [1, 1, Sq, Sk] (removing padding)
                    // We need to extract the valid [Sk, Sq] portion and transpose it to [Sq, Sk]
                    if (mask_sq_pad == actual_sq) {
                        // No padding, direct transpose
                        permute_3210(mask->data, mask_dldnn, 1, 1, mask_sq_pad, mask_sk, target_type);
                    } else {
                        // Has padding, need to extract valid portion first
                        // Create intermediate buffer for valid mask [Sk, Sq, 1, 1] (no padding)
                        GGML_LOG_WARN(
                            "Mask padding detected: mask_sq_pad=%ld, actual_sq=%ld.
                            cuDNN may not handle padding yet.
                            remove the padding in the mask.\n",
                            mask_sq_pad, actual_sq);

                        void* mask_no_pad = nullptr;
                        size_t no_pad_size = mask_sk * actual_sq * ggml_type_size(target_type);
                        CUDA_CHECK(cudaMalloc(&mask_no_pad, no_pad_size));

                        // Copy valid portion: extract [Sk, Sq] from [Sk, Sq_pad]
                        // This is a 2D copy operation for each Sk row
                        const size_t element_size = ggml_type_size(target_type);
                        for (int64_t sk_idx = 0; sk_idx < mask_sk; sk_idx++) {
                            const void* src_row = (const char*)mask->data + sk_idx * mask_sq_pad * element_size;
                            void* dst_row = (char*)mask_no_pad + sk_idx * actual_sq * element_size;
                            CUDA_CHECK(cudaMemcpyAsync(dst_row, src_row, actual_sq * element_size, cudaMemcpyDeviceToDevice));
                        }

                        // Now transpose the no-pad mask [Sk, Sq, 1, 1] -> [1, 1, Sq, Sk]
                        permute_3210(mask_no_pad, mask_dldnn, 1, 1, actual_sq, mask_sk, target_type);

                        // Clean up intermediate buffer
                        CUDA_CHECK(cudaFree(mask_no_pad));
                    }

                    // Set mask descriptor for [1, 1, Sq, Sk] format (no padding)
                    mask_desc.set_from_dims(1, 1, actual_sq, actual_sk, target_type);
                }
            } else {
                // No explicit mask, use causal attention
                is_causal = true;
                GGML_LOG_INFO("No explicit mask provided, using causal attention\n");
            }

            // Get workspace size first
            size_t workspace_size = 0;
            CUDNN_CHECK(cudnnGetScaledDotProductAttentionWorkspaceSize(
                cudnn_handle,
                q_desc.get(),
                k_desc.get(),
                v_desc.get(),
                mask_dldnn ? mask_desc.get() : nullptr,
                out_desc.get(),
                0.0f,
                is_causal,
                scale,
                &workspace_size
            ));

            // Allocate workspace
            void* workspace = nullptr;
            if (workspace_size > 0) {
                CUDA_CHECK(cudaMalloc(&workspace, workspace_size));
            }

            // Call cudnnScaledDotProductAttention
            cudnnStatus_t status = cudnnScaledDotProductAttention(
                cudnn_handle,
                q_desc.get(), q_bhsd,
                k_desc.get(), k_bhsd,
                v_desc.get(), v_bhsd,
                mask_dldnn ? mask_desc.get() : nullptr, mask_dldnn,
                0.0f,
                is_causal,
                scale,
                workspace, workspace_size,
                out_desc.get(), temp_output
            );

            if (status != CUDNN_STATUS_SUCCESS) {
                GGML_LOG_ERROR("cudnnScaledDotProductAttention failed: %s\n", cudnnGetErrorString(status));
                ok = false;
            }

            if (ok) {
                // Convert output from BHSD back to DHSB format
                auto convert_bhsd_to_dhsb = [](const void* input, void* output, int64_t dim_0, int64_t dim_1, int64_t dim_2, int64_t dim_3, enum ggml_type type) -> bool {
                    switch (type) {
                        case GGML_TYPE_F16:
                            call_permute_3120_kernel<ggml_fp16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                            return true;
                        case GGML_TYPE_F32:
                            call_permute_3120_kernel<float>(input, output, dim_0, dim_1, dim_2, dim_3);
                            return true;
                        case GGML_TYPE_BF16:
                            call_permute_3120_kernel<ggml_bf16_t>(input, output, dim_0, dim_1, dim_2, dim_3);
                            return true;
                        default:
                            return false;
                    }
                };

                // Convert BHSD -> DHSB: [batch_size, num_heads, seq_len, head_dim] -> [head_dim, seq_len, num_heads, batch_size]
                if (!convert_bhsd_to_dhsb(temp_output, KQV->data, temp_ne[3], temp_ne[2], temp_ne[1], temp_ne[0], target_type)) {
                    GGML_LOG_ERROR("Failed to convert output to DHSB format\n");
                    ok = false;
                }
            }

            // Cleanup
            if (temp_output) CUDA_CHECK(cudaFree(temp_output));
            if (mask_dldnn) CUDA_CHECK(cudaFree(mask_dldnn));
            if (workspace) CUDA_CHECK(cudaFree(workspace));
        }
    }

    // Cleanup converted tensors and BHSD buffers
    if (q_converted_gpu) CUDA_CHECK(cudaFree(q_converted_gpu));
    if (k_converted_gpu) CUDA_CHECK(cudaFree(k_converted_gpu));
    if (v_converted_gpu) CUDA_CHECK(cudaFree(v_converted_gpu));
    if (q_bhsd) CUDA_CHECK(cudaFree(q_bhsd));
    if (k_bhsd) CUDA_CHECK(cudaFree(k_bhsd));
    if (v_bhsd) CUDA_CHECK(cudaFree(v_bhsd));

    if (ok) {
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}

static void ggml_cuda_flash_attn_ext_dldnn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    bool ok = true;
    const struct ggml_tensor * KQV  = dst;
    const struct ggml_tensor * Q    = dst->src[0];
    const struct ggml_tensor * K    = dst->src[1];
    const struct ggml_tensor * V    = dst->src[2];
    const struct ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);

    cudnnHandle_t cudnn_handle = getCudnnHandle();
    if (cudnn_handle == nullptr) {
        GGML_LOG_ERROR("Failed to get cuDNN handle\n");
        return;
    }

    // Extract parameters
    float scale;
    float max_bias;
    float logit_softcap;
    memcpy(&scale,         ((const int32_t *) dst->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) dst->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) dst->op_params) + 2, sizeof(logit_softcap));

    // Determine which cuDNN interface to use
    bool has_alibi = (max_bias > 0.0f);
    bool has_mask = (mask != nullptr);
    bool use_mha_forward = has_alibi && !has_mask;  // Use MHA for ALiBi only
    bool use_scaled_dot_product = has_mask && !has_alibi;  // Use ScaledDotProduct for mask only

    if (has_alibi && has_mask) {
        GGML_LOG_WARN("DLDNN: Both ALiBi and mask present. Currently not supported simultaneously. Falling back to standard implementation.\n");
        return;
    }

    if (!has_alibi && !has_mask) {
        // GGML_LOG_INFO("DLDNN: No ALiBi or mask, using cudnnScaledDotProductAttention with is_causal=true\n");
        // use_scaled_dot_product = true;
        GGML_LOG_INFO("DLDNN: No ALiBi or mask, using cudnnMHAForward\n");
        use_mha_forward = true;
    }

    GGML_LOG_INFO("DLDNN interface selection: has_alibi=%s, has_mask=%s, using %s\n",
                  has_alibi ? "true" : "false",
                  has_mask ? "true" : "false",
                  use_mha_forward ? "cudnnMHAForward" : "cudnnScaledDotProductAttention");

    if (use_mha_forward) {
        ggml_cuda_flash_attn_ext_dldnn_mha_forward(ctx, dst);
    } else if (use_scaled_dot_product) {
        ggml_cuda_flash_attn_ext_dldnn_scaled_dot_product(ctx, dst);
    } else {
        GGML_LOG_ERROR("DLDNN: Unable to determine appropriate cuDNN interface\n");
        return;
    }
}

#endif // GGML_USE_DLFA

void ggml_cuda_flash_attn_ext(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * KQV  = dst;
    const ggml_tensor * Q    = dst->src[0];
    const ggml_tensor * K    = dst->src[1];
    const ggml_tensor * V    = dst->src[2];
    const ggml_tensor * mask = dst->src[3];

    ggml_cuda_set_device(ctx.device);

#if defined(GGML_USE_DLFA)
    // use DLDNN if available
    if (ggml_cuda_flash_attn_ext_dldnn_available(ctx, dst)) {
        ggml_cuda_flash_attn_ext_dldnn(ctx, dst);
        return;
    }
    GGML_LOG_WARN("DLDNN is not available, use fallback.\n");
    // return;
#endif

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const enum ggml_prec prec = ggml_flash_attn_ext_get_prec(KQV);

    if (GGML_CUDA_CC_IS_AMD(cc)) {
#if defined(GGML_HIP_ROCWMMA_FATTN)
        if (fp16_mma_available(cc)) {
            ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
            return;
        }
#endif // defined(GGML_HIP_ROCWMMA_FATTN)

        // On AMD the tile kernels perform poorly, use the vec kernel instead:
        if (prec == GGML_PREC_DEFAULT && fast_fp16_available(cc)) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    if (!fast_fp16_available(cc)) {
        if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
        }
        return;
    }

    if (!fp16_mma_available(cc)) {
        if (prec == GGML_PREC_DEFAULT) {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f16(ctx, dst);
            }
        } else {
            if (Q->ne[1] <= 8 || Q->ne[0] == 256) {
                ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
            } else {
                ggml_cuda_flash_attn_ext_tile_f32(ctx, dst);
            }
        }
        return;
    }

    const bool gqa_opt_applies = ((Q->ne[2] / K->ne[2]) % 2 == 0) && mask; // The mma-based kernels have GQA-specific optimizations
    const bool mma_needs_data_conversion = K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16;
    const bool mma_faster_for_bs1 = new_mma_available(cc) && gqa_opt_applies && cc < GGML_CUDA_CC_ADA_LOVELACE && !mma_needs_data_conversion;
    const bool can_use_vector_kernel = Q->ne[0] <= 256 && Q->ne[0] % (2*warp_size) == 0;
    if (Q->ne[1] == 1 && can_use_vector_kernel && !mma_faster_for_bs1) {
        if (prec == GGML_PREC_DEFAULT) {
            ggml_cuda_flash_attn_ext_vec_f16(ctx, dst);
        } else {
            ggml_cuda_flash_attn_ext_vec_f32(ctx, dst);
        }
        return;
    }

    // The MMA implementation needs Turing or newer, use the old WMMA code for Volta:
    if (fp16_mma_available(cc) && !new_mma_available(cc)) {
        ggml_cuda_flash_attn_ext_wmma_f16(ctx, dst);
    }

#ifdef GGML_USE_DLCU
    // GGML_ABORT("ggml_cuda_flash_attn_ext_mma_f16 is not supported in DLIN yet");
    GGML_LOG_ERROR("ggml_cuda_flash_attn_ext_mma_f16 is not supported in DLIN yet");
#else
    ggml_cuda_flash_attn_ext_mma_f16(ctx, dst);
#endif
}
