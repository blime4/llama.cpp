#include "scale.cuh"
#include <cstdio>
#include <cuda_fp16.h>

static __global__ void scale_f32(const float * x, float * dst, const float scale, const float bias, const int k) {
    const int i = blockDim.x*blockIdx.x + threadIdx.x;

    if (i >= k) {
        return;
    }

    dst[i] = scale * x[i] + bias;
}

static void scale_f32_cuda(const float * x, float * dst, const float scale, const float bias, const int k, cudaStream_t stream) {
    const int num_blocks = (k + CUDA_SCALE_BLOCK_SIZE - 1) / CUDA_SCALE_BLOCK_SIZE;
    scale_f32<<<num_blocks, CUDA_SCALE_BLOCK_SIZE, 0, stream>>>(x, dst, scale, bias, k);
}

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

#ifdef GGML_USE_DLCU // DL-FP16
    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT( dst->type == GGML_TYPE_F32 ||  dst->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
#else
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);
#endif

    float scale;
    float bias;
    memcpy(&scale, (float *) dst->op_params + 0, sizeof(float));
    memcpy(&bias,  (float *) dst->op_params + 1, sizeof(float));

    if (src0->type == GGML_TYPE_F16) {
        scale_f16_cuda((const half*)src0_d, (half*)dst_d, scale, bias, ggml_nelements(src0), stream);
    } else if (src0->type == GGML_TYPE_F32) {
        scale_f32_cuda(src0_d, dst_d, scale, bias, ggml_nelements(src0), stream);
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
