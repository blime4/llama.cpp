#include "dl-fp16.cuh"

// Row reduction kernel template - compute sum (norm=false) or mean (norm=true)
template <bool norm>
__global__ void reduce_rows_f16(const half * __restrict__ x, half * __restrict__ dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float     sum        = 0.0f;
    const int num_unroll = 8;
    float     temp[num_unroll];
    float     sum_temp[num_unroll] = { 0.0f };
    for (int i = col; i < ncols;) {
        for (int j = 0; j < num_unroll; ++j) {
            if (i < ncols) {
                temp[j] = static_cast<float>(x[row * ncols + i]);
            } else {
                temp[j] = 0;
            }
            i += blockDim.x;
        }
        for (int j = 0; j < num_unroll; ++j) {
            sum_temp[j] += temp[j];
        }
    }
    for (int j = 0; j < num_unroll; ++j) {
        sum += sum_temp[j];
    }

    // sum up partial sums
    sum = warp_reduce_sum(sum);
    if (blockDim.x > WARP_SIZE) {
        assert((blockDim.x <= 1024) && (blockDim.x % WARP_SIZE) == 0);
        __shared__ float s_sum[32];
        const int        warp_id = threadIdx.x / WARP_SIZE;
        const int        lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = sum;
        }
        __syncthreads();
        sum = 0.0f;
        if (lane_id < (blockDim.x / WARP_SIZE)) {
            sum = s_sum[lane_id];
        }
        sum = warp_reduce_sum(sum);
    }

    if (col != 0) {
        return;
    }

    dst[row] = norm ? static_cast<half>(sum / ncols) : static_cast<half>(sum);
}

void sum_rows_f16_cuda(const half * x, half * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows_f16</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows_f16</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    }
}

// Explicit template instantiation
template __global__ void reduce_rows_f16<true>(const half * __restrict__ x, half * __restrict__ dst, const int ncols);