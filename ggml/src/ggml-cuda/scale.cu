#include "scale.cuh"
#include "common.cuh"
#include <cuda_fp16.h>

#define MAX_GRIDDIM_X 0x7FFFFFFF

template <typename T>
static __global__ void scale_kernel(const T * x, T * dst, const float scale, const float bias, const int64_t nelements) {
    int64_t tid = (int64_t)blockIdx.x * (int64_t)blockDim.x + (int64_t)threadIdx.x;
    int64_t stride = (int64_t)blockDim.x * (int64_t)gridDim.x;

    for (int64_t i = tid; i < nelements; i += stride) {
        dst[i] = from_float<T>(scale * to_float(x[i]) + bias);
    }
}

template <typename T>
static void scale_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_kernel<T><<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, nelements);
}

template void scale_cuda<float>(const float *, float *, const float, const float, const int64_t, cudaStream_t);
template void scale_cuda<half>(const half *, half *, const float, const float, const int64_t, cudaStream_t);

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        scale_cuda<half>((const half*)src0_d, (half*)dst_d, scale, bias, ggml_nelements(src0), stream);
    } else {
        scale_cuda<float>((const float*)src0_d, (float*)dst_d, scale, bias, ggml_nelements(src0), stream);
    }
}
