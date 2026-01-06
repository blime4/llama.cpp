#pragma once

#ifdef GGML_USE_DLFA

#include "ggml.h"
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <memory>

// Forward declaration
struct ggml_backend_cuda_context;

namespace ggml_dl {

struct flash_attn_ext_decode_state {
    // Track the real KV length per layer (keyed by the K buffer address) so
    // decode steps don't accidentally share state across layers.
    std::unordered_map<uintptr_t, int64_t> seq_k_real_by_k_ptr;
};

// Host-side precomputed metadata to avoid recomputing lengths during forward.
// Filled in set_input and consumed by the CUDA forward path.
// Only keeps essential varlen metadata: cu_seqlens_q, seqused_k, block_table
struct flash_attn_dlfa_runtime {
    int batch = 0;
    int seqlen_q = 0;
    int seqlen_k_real = 0;
    int block_table_stride = 0;
    std::vector<int32_t> cu_seqlens_q; // size batch+1
    std::vector<int32_t> seqused_k;    // size batch
    std::vector<int32_t> block_table;  // size batch * block_table_stride

    // Per-device cached buffers for multi-GPU support
    // Key: device_id, Value: device-specific cache
    std::unordered_map<int, void*> device_caches;
};

inline void store_mask_metadata(ggml_tensor * attn, const ggml_flash_attn_mask_params & params) {
    GGML_ASSERT(attn != nullptr);
    // Store mask params directly in attn tensor's op_params following ggml.c pattern
    // Reference: @ggml/src/ggml.c:4137-4143 for similar parameter storage approach

    // Directly access op_params array to store mask parameters
    // The magic value indicates that mask params are present
    int32_t * op_params = (int32_t *)(attn->op_params);
    op_params[11] = 0x46414d31; // GGML_FLASH_ATTN_PARAM_MASK_MAGIC_VALUE
    op_params[4] = params.present ? 1 : 0;
    op_params[5] = params.is_causal ? 1 : 0;
    op_params[6] = params.window_left;
    op_params[7] = params.window_right;
    op_params[8] = params.per_token_window ? 1 : 0;
    op_params[9] = params.multi_sequence ? 1 : 0;
    op_params[10] = params.has_alibi_bias ? 1 : 0;
}

// Reset per-thread decode bookkeeping so unit tests (or new inference sessions)
// start from a clean state.
void flash_attn_ext_dldnn_reset_decode_state();

// Accessor for the thread-local decode state used by the DLFA kernels.
flash_attn_ext_decode_state & flash_attn_ext_dldnn_decode_state();

// Prepare device buffers for varlen forward (called early for async overlap)
// This should be called after set_inputs and before graph_compute
// Note: ggml_backend_cuda_context is defined in ggml-cuda/common.cuh
void flash_attn_ext_dldnn_prepare_varlen_buffers(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

} // namespace ggml_dl

#endif // GGML_USE_DLFA
