#include "scale.cuh"
#include "common.cuh"
#include <cstdio>
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

// Explicit instantiations for the kernel
template __global__ void scale_kernel<float>(const float *, float *, const float, const float, const int64_t);
template __global__ void scale_kernel<half>(const half *, half *, const float, const float, const int64_t);

template <typename T>
static void scale_cuda(const T * x, T * dst, const float scale, const float bias, const int64_t nelements, cudaStream_t stream) {
    const int64_t num_blocks = (nelements + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_kernel<T><<<MIN(MAX_GRIDDIM_X, num_blocks), CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, nelements);
}

// Explicit instantiations for the wrapper
template void scale_cuda<float>(const float *, float *, const float, const float, const int64_t, cudaStream_t);
template void scale_cuda<half>(const half *, half *, const float, const float, const int64_t, cudaStream_t);

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_F16);
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

#ifndef NDEBUG
    const int print_n = (ggml_nelements(src0) < 5) ? (int)ggml_nelements(src0) : 5;
    if (print_n > 0) {
        // ensure the CUDA work on `stream` has finished before copying
        // cudaStreamSynchronize(stream);
        if (src0->type == GGML_TYPE_F16) {
            const half * src_h0 = (const half*) src0->data;
            half h_src[5];
            half h_dst[5];
            cudaMemcpy(h_src, src_h0, print_n * sizeof(half), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dst, dst_d,  print_n * sizeof(half), cudaMemcpyDeviceToHost);
            printf("scale (f16) src:");
            for (int i = 0; i < print_n; ++i) {
                printf(" %f", __half2float(h_src[i]));
            }
            printf("\n");
            printf("scale (f16) dst:");
            for (int i = 0; i < print_n; ++i) {
                printf(" %f", __half2float(h_dst[i]));
            }
            printf("\n");
        } else {
            float h_src[5];
            float h_dst[5];
            cudaMemcpy(h_src, src0_d, print_n * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(h_dst, dst_d,  print_n * sizeof(float), cudaMemcpyDeviceToHost);
            printf("scale (f32) src:");
            for (int i = 0; i < print_n; ++i) {
                printf(" %f", h_src[i]);
            }
            printf("\n");
            printf("scale (f32) dst:");
            for (int i = 0; i < print_n; ++i) {
                printf(" %f", h_dst[i]);
            }
            printf("\n");
        }
    }
#endif

}
