#pragma once

#ifdef GGML_USE_DLFA

#include "ggml.h"
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <memory>

// Forward declaration
struct ggml_backend_cuda_context;

// Forward declaration for device cache structure (defined in dl-fattn.cu)
struct flash_attn_device_layout_cache;

namespace ggml_dl {

struct flash_attn_ext_decode_state {
    // Track the real KV length per layer (keyed by the K buffer address) so
    // decode steps don't accidentally share state across layers.
    std::unordered_map<uintptr_t, int64_t> seq_k_real_by_k_ptr;
};

// Reset per-thread decode bookkeeping so unit tests (or new inference sessions)
// start from a clean state.
void flash_attn_ext_dldnn_reset_decode_state();

// Accessor for the thread-local decode state used by the DLFA kernels.
flash_attn_ext_decode_state & flash_attn_ext_dldnn_decode_state();

// Prepare device buffers for varlen forward (called early for async overlap)
// This should be called after set_inputs and before graph_compute
// Note: ggml_backend_cuda_context is defined in ggml-cuda/common.cuh
// NEW: Updated to accept seq_id parameter for proper slot isolation
void flash_attn_ext_dldnn_prepare_varlen_buffers(ggml_backend_cuda_context & ctx, ggml_tensor * dst, int32_t seq_id);

// Cleanup function to free all varlen_data and device memory
// NOTE: This function is available for manual cleanup but is not automatically
// called due to build system linking constraints. CUDA memory will be freed
// when the process exits. To manually call this function from C++ code, use:
//   extern "C" void ggml_dl_flash_attn_ext_dldnn_cleanup_varlen_data();
//   ggml_dl_flash_attn_ext_dldnn_cleanup_varlen_data();
void flash_attn_ext_dldnn_cleanup_varlen_data();

} // namespace ggml_dl

#endif // GGML_USE_DLFA
