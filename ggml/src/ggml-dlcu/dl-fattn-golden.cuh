/**
 * @file dl-fattn-golden.cuh
 * @brief Golden reference implementation header for DLDNN Flash Attention verification
 *
 * This file will be removed once the DLDNN implementation is fully validated.
 */

#pragma once

#ifdef GGML_USE_DLFA

#include "ggml.h"
#include "ggml-backend.h"

/**
 * CPU Reference Implementation of Scaled Dot-Product Attention
 *
 * @param q_data Query tensor in DSHB format [D, Sq, H, B]
 * @param k_data Key tensor in DSHB format [D, Sk, H, B]
 * @param v_data Value tensor in DSHB format [D, Sk, H, B]
 * @param mask_data Mask tensor in [1, 1, Sq, Sk] format or nullptr
 * @param gpu_output GPU output tensor in DHSB format [D, H, Sq, B]
 * @param B Batch size
 * @param H Number of heads
 * @param Sq Query sequence length
 * @param Sk Key sequence length
 * @param D Head dimension
 * @param scale Attention scale factor (typically 1/sqrt(D))
 * @param is_causal Whether to apply causal masking
 * @param tolerance NMSE tolerance for passing verification
 * @return true if verification passes, false otherwise
 */
template<typename T_in, typename T_out>
bool verify_attention_golden(
    const T_in* q_data,
    const T_in* k_data,
    const T_in* v_data,
    const T_in* mask_data,
    const T_out* gpu_output,
    int B, int H, int Sq, int Sk, int D,
    float scale,
    bool is_causal,
    float tolerance = 0.03f
);

#endif // GGML_USE_DLFA

