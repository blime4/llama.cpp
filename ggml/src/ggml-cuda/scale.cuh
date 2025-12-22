#include "common.cuh"

#define CUDA_SCALE_BLOCK_SIZE 256

// Forward declare DL-FP16 wrapper (implemented in ggml-dlcu/fp16/dl-scale.cu)
#ifdef GGML_USE_DLCU
void scale_f16_cuda(const half * x, half * dst, const float scale, const float bias, const int k, cudaStream_t stream);
#endif

void ggml_cuda_op_scale(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
