#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../ggml-cuda/common.cuh"

void rms_norm_f16_cuda(const half * x, half * dst, const int ncols, const int nrows, const int nchannels, const int nsamples, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream);

void rms_norm_mul_f16_cuda(
    const half * x, const half * mul, half * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
    const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
    const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
    const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
    const float eps, cudaStream_t stream);

void scale_f16_cuda(const half * x, half * dst, const float scale, const float bias, const int k, cudaStream_t stream);

