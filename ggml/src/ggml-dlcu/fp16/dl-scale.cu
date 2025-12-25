#include "dl-fp16.cuh"
#include "../../ggml-cuda/scale.cuh"

static __global__ void scale_f16(const half * x, half * dst, const float scale, const float bias, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = static_cast<half>(scale * static_cast<float>(x[i]) + bias);
}

void scale_f16_cuda(const half * x, half * dst, const float scale, const float bias, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f16<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, k);
}

