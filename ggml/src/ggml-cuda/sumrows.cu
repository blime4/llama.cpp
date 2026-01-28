#include "cuda_fp16.h"
#include "reduce_rows.cuh"
#include "sumrows.cuh"

template <typename T>
void sum_rows_cuda(const T * x, T * dst, const int ncols, const int nrows, cudaStream_t stream) {
    const int  id  = ggml_cuda_get_device();
    const int  nsm = ggml_cuda_info().devices[id].nsm;
    const dim3 block_nums(nrows, 1, 1);
    if ((nrows / nsm) < 2) {
        const dim3 block_dims(512, 1, 1);
        reduce_rows<false, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    } else {
        const dim3 block_dims(ncols < 1024 ? 32 : 128, 1, 1);
        reduce_rows<false, T><<<block_nums, block_dims, 0, stream>>>(x, dst, ncols);
    }
}

// Explicit instantiations for the wrapper
template void sum_rows_cuda<float>(const float *, float *, const int, const int, cudaStream_t);
template void sum_rows_cuda<half>(const half *, half *, const int, const int, cudaStream_t);

void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const void * src0_d = src0->data;
    void * dst_d = dst->data;
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src0->type == dst->type);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t ncols = src0->ne[0];
    const int64_t nrows = ggml_nrows(src0);

    if (src0->type == GGML_TYPE_F32) {
        sum_rows_cuda<float>((const float*)src0_d, (float*)dst_d, ncols, nrows, stream);
    } else {
        sum_rows_cuda<half>((const half*)src0_d, (half*)dst_d, ncols, nrows, stream);
    }
}
