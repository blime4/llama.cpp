#pragma once

#ifdef GGML_USE_DLFA

#include "ggml.h"

namespace ggml_dl {

struct flash_attn_mask_info {
    bool    present          = false;
    bool    is_causal        = false;
    int32_t window_left      = -1;
    int32_t window_right     = -1;
    bool    per_token_window = false;
    bool    multi_sequence   = false;
    bool    has_alibi_bias   = false;
};

struct flash_attn_ext_decode_state {
    int64_t seq_k_real = 0;
};

inline ggml_flash_attn_mask_params to_params(const flash_attn_mask_info & info) {
    ggml_flash_attn_mask_params params{};
    params.present          = info.present;
    params.is_causal        = info.is_causal;
    params.window_left      = info.window_left;
    params.window_right     = info.window_right;
    params.per_token_window = info.per_token_window;
    params.multi_sequence   = info.multi_sequence;
    params.has_alibi_bias   = info.has_alibi_bias;
    return params;
}

inline flash_attn_mask_info from_params(const ggml_flash_attn_mask_params & params) {
    flash_attn_mask_info info{};
    info.present          = params.present;
    info.is_causal        = params.is_causal;
    info.window_left      = params.window_left;
    info.window_right     = params.window_right;
    info.per_token_window = params.per_token_window;
    info.multi_sequence   = params.multi_sequence;
    info.has_alibi_bias   = params.has_alibi_bias;
    return info;
}

inline void store_mask_metadata(ggml_tensor * mask, const flash_attn_mask_info & info) {
    GGML_ASSERT(mask != nullptr);
    auto params = to_params(info);
    ggml_flash_attn_ext_set_mask_params(mask, &params);
}

inline bool load_mask_metadata(const ggml_tensor * mask, flash_attn_mask_info * out_info) {
    GGML_ASSERT(mask != nullptr);
    ggml_flash_attn_mask_params params{};
    if (!ggml_flash_attn_ext_get_mask_params(mask, &params)) {
        return false;
    }
    if (out_info != nullptr) {
        *out_info = from_params(params);
    }
    return true;
}

// Reset per-thread decode bookkeeping so unit tests (or new inference sessions)
// start from a clean state.
void flash_attn_ext_dldnn_reset_decode_state();

// Accessor for the thread-local decode state used by the DLFA kernels.
flash_attn_ext_decode_state & flash_attn_ext_dldnn_decode_state();


} // namespace ggml_dl

#endif // GGML_USE_DLFA
