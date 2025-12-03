#include "scale.cuh"

template <typename T>
static __global__ void scale_kernel(const T * x, T * dst, const float scale, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = static_cast<T>(scale * static_cast<float>(x[i]));
}

template <typename T>
static void scale_cuda(const T * x, T * dst, const float scale, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_kernel<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, k);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);

    float scale;
    memcpy(&scale, dst->op_params, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        scale_cuda((const half*)src0_d, (half*)dst_d, scale, ggml_nelements(src0), stream);
    } else if (src0->type == GGML_TYPE_F32) {
        scale_cuda(src0_d, dst_d, scale, ggml_nelements(src0), stream);
    } else {
        GGML_ASSERT(false);
    }
}
