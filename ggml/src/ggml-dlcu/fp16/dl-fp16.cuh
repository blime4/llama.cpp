#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "../../ggml-cuda/common.cuh"

void rms_norm_f16_cuda(const half * x, half * dst, const int ncols, const int nrows, const int nchannels, const int nsamples, const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream);

void rms_norm_mul_f16_cuda(const half * x,
    const float * mul,
    const float * add,
    half * dst,
    const int     ncols,
    const int     nrows,
    const int     nchannels,
    const int     nsamples,
    const int64_t stride_row,
    const int64_t stride_channel,
    const int64_t stride_sample,
    const int64_t mul_stride_row,
    const int64_t mul_stride_channel,
    const int64_t mul_stride_sample,
    const int     mul_ncols,
    const int     mul_nrows,
    const int     mul_nchannels,
    const int     mul_nsamples,
    const int64_t add_stride_row,
    const int64_t add_stride_channel,
    const int64_t add_stride_sample,
    const int     add_ncols,
    const int     add_nrows,
    const int     add_nchannels,
    const int     add_nsamples,
    const float   eps,
    cudaStream_t  stream);

void scale_f16_cuda(const half * x, half * dst, const float scale, const float bias, const int k, cudaStream_t stream);

void softcap_f16_cuda(const half * x, half * dst, const float scale, const float softcap, const int k, cudaStream_t stream);

template <typename T>
static __device__ __forceinline__ float t2f32(T val) {
    return (float) val;
}

template <>
__device__ float __forceinline__ t2f32<half>(half val) {
    return __half2float(val);
}

struct soft_max_params {

    int64_t nheads;
    uint32_t n_head_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    float scale;
    float max_bias;
    float m0;
    float m1;
};

template<typename T>
void soft_max_f16_cuda(const half * x, const T * mask, const float * sinks, half * dst, const soft_max_params & params, cudaStream_t stream);

template <bool norm>
__global__ void reduce_rows_f16(const half * __restrict__ x, half * __restrict__ dst, const int ncols);

void sum_rows_f16_cuda(const half * x, half * dst, const int ncols, const int nrows, cudaStream_t stream);

