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
    g_flash_attn_decode_state = {};
}

} // namespace ggml_dl

#endif // GGML_USE_DLFA
