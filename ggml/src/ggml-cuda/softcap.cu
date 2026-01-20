#include "softcap.cuh"
#include "common.cuh"
#include <cuda_fp16.h>

template <typename T>
static __global__ void softcap_kernel(const T * x, T * dst, const float scale, const float softcap, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = from_float<T>(tanhf(scale * to_float(x[i])) * softcap);
}

// Explicit instantiations for the kernel
template __global__ void softcap_kernel<float>(const float *, float *, const float, const float, const int);
template __global__ void softcap_kernel<half>(const half *, half *, const float, const float, const int);

template <typename T>
static void softcap_cuda(const T * x, T * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    softcap_kernel<T><<<num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream>>>(x, dst, scale, softcap, k);
}

// Explicit instantiations for the wrapper
template void softcap_cuda<float>(const float *, float *, const float, const float, const int, cudaStream_t);
template void softcap_cuda<half>(const half *, half *, const float, const float, const int, cudaStream_t);

// fused GGML_OP_SCALE + GGML_UNARY_OP_TANH + GGML_OP_SCALE
void ggml_cuda_op_softcap(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * src) {
    const ggml_tensor * src0 = src->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float scale;
    float softcap;
    memcpy(&scale,   (float *) src->op_params + 0, sizeof(float));
    memcpy(&softcap, (float *) dst->op_params + 0, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        softcap_cuda<half>((const half*)src0_d, (half*)dst_d, scale, softcap, ggml_nelements(src0), stream);
    } else {
        softcap_cuda<float>((const float*)src0_d, (float*)dst_d, scale, softcap, ggml_nelements(src0), stream);
    }
}
