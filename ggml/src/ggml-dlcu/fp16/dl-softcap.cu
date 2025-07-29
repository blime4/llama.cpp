#include "dl-fp16.cuh"
#include "../../ggml-cuda/softcap.cuh"

static __global__ void softcap_f16(const half * x, half * dst, const float scale, const float softcap, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = static_cast<half>(tanhf(scale * static_cast<float>(x[i])) * softcap);
}

void softcap_f16_cuda(const half * x, half * dst, const float scale, const float softcap, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SOFTCAP_BLOCK_SIZE - 1) / CUDA_SOFTCAP_BLOCK_SIZE;
    softcap_f16<<<num_blocks, CUDA_SOFTCAP_BLOCK_SIZE, 0, stream>>>(x, dst, scale, softcap, k);
}