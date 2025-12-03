#include "sumrows.cuh"

template<bool norm, typename T>
static __global__ void reduce_rows(const T * x, T * dst, const int ncols) {
    const int row = blockIdx.x;
    const int col = threadIdx.x;

    float sum = 0.0f;
    for (int i = col; i < ncols; i += blockDim.x) {
        sum += static_cast<float>(x[row * ncols + i]);
    }

    sum = warp_reduce_sum(sum);

    if (col != 0) {
        return;
    }

    dst[row] = norm ? static_cast<T>(sum / ncols) : static_cast<T>(sum);
}

void sum_rows_f32_cuda(const float * x, float * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);
    reduce_rows_f32</*norm*/false><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
}

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const float * src0_d = (const float *)src0->data;
    float * dst_d = (float *)dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    // GGML_ASSERT( dst->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    const dim3 block_dims(WARP_SIZE, 1, 1);
    const dim3 block_nums(nrows, 1, 1);

    if (src0->type == GGML_TYPE_F32) {
        reduce_rows</*norm=*/false><<<block_nums, block_dims, 0, stream>>>(src0_d, dst_d, ncols);
    } else if (src0->type == GGML_TYPE_F16) {
        reduce_rows</*norm=*/false><<<block_nums, block_dims, 0, stream>>>((const half*)src0_d, (half*)dst_d, ncols);
    } else {
        GGML_ASSERT(false);
    }
}
