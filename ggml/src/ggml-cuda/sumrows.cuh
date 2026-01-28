#include "common.cuh"

template <typename T>
void sum_rows_cuda(const T * x, T * dst, const int ncols, const int nrows, cudaStream_t stream);
void ggml_cuda_op_sum_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
