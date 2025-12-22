#include "norm.cuh"
#include <cstdint>

template <int block_size, bool do_multiply = false>
static __global__ void rms_norm_f16(
        const half * x, half * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * mul = nullptr, const int64_t mul_stride_row = 0,
        const int64_t mul_stride_channel = 0, const int64_t mul_stride_sample = 0, const int mul_ncols = 0,
        const int mul_nrows = 0, const int mul_nchannels = 0, const int mul_nsamples = 0) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const int mul_row = row % mul_nrows;
        const int mul_channel = channel % mul_nchannels;
        const int mul_sample = sample % mul_nsamples;
        mul += mul_sample*mul_stride_sample + mul_channel*mul_stride_channel + mul_row*mul_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = static_cast<float>(x[col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply) {
            const int mul_col = col % mul_ncols;
            dst[col] = static_cast<half>(scale * static_cast<float>(x[col]) * static_cast<float>(mul[mul_col]));
        } else {
            dst[col] = static_cast<half>(scale * static_cast<float>(x[col]));
        }
    }
}

template <int block_size, bool do_multiply = false>
static __global__ void rms_norm_f16(
        const half * x, half * dst, const int ncols, const int64_t stride_row, const int64_t stride_channel,
        const int64_t stride_sample, const float eps, const float * mul = nullptr, const int64_t mul_stride_row = 0,
        const int64_t mul_stride_channel = 0, const int64_t mul_stride_sample = 0, const int mul_ncols = 0,
        const int mul_nrows = 0, const int mul_nchannels = 0, const int mul_nsamples = 0) {
    const int nrows     = gridDim.x;
    const int nchannels = gridDim.y;

    const int row       = blockIdx.x;
    const int channel   = blockIdx.y;
    const int sample    = blockIdx.z;
    const int tid       = threadIdx.x;

    x   += sample*stride_sample + channel*stride_channel + row*stride_row;
    dst += ((sample*nchannels + channel)*nrows + row)*ncols;

    if constexpr (do_multiply) {
        const int mul_row = row % mul_nrows;
        const int mul_channel = channel % mul_nchannels;
        const int mul_sample = sample % mul_nsamples;
        mul += mul_sample*mul_stride_sample + mul_channel*mul_stride_channel + mul_row*mul_stride_row;
    }

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = static_cast<float>(x[col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if constexpr (block_size > WARP_SIZE) {
        static_assert(block_size == 1024, "unexpected block_size");
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += block_size) {
        if constexpr (do_multiply) {
            const int mul_col = col % mul_ncols;
            dst[col] = static_cast<half>(scale * static_cast<float>(x[col]) * static_cast<float>(mul[mul_col]));
        } else {
            dst[col] = static_cast<half>(scale * static_cast<float>(x[col]));
        }
    }
}

// non-static wrapper so other translation units can call it (declared in norm.cuh)
void rms_norm_f16_cuda(
        const half * x, half * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample, const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f16<WARP_SIZE, false><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f16<1024, false><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps);
    }
}

static void rms_norm_mul_f16_cuda(
        const half * x, const half * mul, half * dst, const int ncols, const int nrows, const int nchannels, const int nsamples,
        const int64_t stride_row, const int64_t stride_channel, const int64_t stride_sample,
        const int64_t mul_stride_row, const int64_t mul_stride_channel, const int64_t mul_stride_sample,
        const int mul_ncols, const int mul_nrows, const int mul_nchannels, const int mul_nsamples,
        const float eps, cudaStream_t stream) {
    const dim3 blocks_num(nrows, nchannels, nsamples);
    if (mul == nullptr) {
        rms_norm_f16_cuda(x, dst, ncols, nrows, nchannels, nsamples, stride_row, stride_channel, stride_sample, eps, stream);
        return;
    }
    if (ncols < 1024) {
        const dim3 block_dims(WARP_SIZE, 1, 1);
        rms_norm_f16<WARP_SIZE, true><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples);
    } else {
        const dim3 block_dims(1024, 1, 1);
        rms_norm_f16<1024, true><<<blocks_num, block_dims, 0, stream>>>(x, dst, ncols, stride_row, stride_channel, stride_sample, eps, mul, mul_stride_row, mul_stride_channel, mul_stride_sample, mul_ncols, mul_nrows, mul_nchannels, mul_nsamples);
    }
}