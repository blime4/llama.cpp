#include "../../include/ggml-dlfa.h"

#ifdef GGML_USE_DLFA

namespace {
    thread_local ggml_dl::flash_attn_ext_decode_state g_flash_attn_decode_state;
}

namespace ggml_dl {

flash_attn_ext_decode_state & flash_attn_ext_dldnn_decode_state() {
    return g_flash_attn_decode_state;
}

void flash_attn_ext_dldnn_reset_decode_state() {
    g_flash_attn_decode_state.seq_k_real_by_k_ptr.clear();
    // Note: reset_accumulation_state() should also be called when starting a new inference session
    // It is defined in dl-fattn.cu and can be called via ggml_dl::flash_attn_ext_dldnn_reset_accumulation_state()
    // or the C linkage wrapper ggml_dl_flash_attn_ext_dldnn_reset_accumulation_state()
}

} // namespace ggml_dl

#endif // GGML_USE_DLFA
