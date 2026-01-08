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
struct flash_attn_dlfa_runtime {
    int batch = 0;
    int seqlen_q = 0;
    int seqlen_k_real = 0;
    int block_table_stride = 0;
    std::vector<int32_t> cu_seqlens_q; // size batch+1
    std::vector<int32_t> seqused_k;    // size batch
    std::vector<int32_t> block_table;  // size batch * block_table_stride
    ggml_flash_attn_mask_params mask_params{};
    bool has_mask_params = false;
    // Device-side cached buffers for runtime layout (allocated lazily in CUDA path).
    // Raw pointer on purpose: lifetime tied to process; avoid CUDA teardown ordering issues.
    void * device_cache = nullptr;
};

inline void store_mask_metadata(ggml_tensor * attn, const ggml_flash_attn_mask_params & params) {
    GGML_ASSERT(attn != nullptr);
    GGML_ASSERT(attn->src[3] != nullptr);
    ggml_tensor * mask = attn->src[3];

    auto * runtime = static_cast<flash_attn_dlfa_runtime *>(mask->extra);
    if (runtime == nullptr) {
        runtime = new flash_attn_dlfa_runtime();
        mask->extra = runtime;
    }

    runtime->mask_params = params;
    runtime->has_mask_params = true;
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
