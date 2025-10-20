/**
 * @file dl-fattn-golden.cu
 * @brief Golden reference implementation for DLDNN Flash Attention verification
 *
 * This file contains CPU-based reference implementations for attention mechanisms.
 * It will be removed once the DLDNN implementation is fully validated.
 */

#ifdef GGML_USE_DLFA

#include "dl-fattn-golden.cuh"
#include "ggml-impl.h"

#include <cmath>
#include <cstdio>
#include <algorithm>
#include <vector>
#include <typeinfo>

// ============================================================================
// Type Conversion Helpers (internal)
// ============================================================================

namespace {

template<typename T>
inline float to_float_impl(const T& val) {
    return static_cast<float>(val);
}

template<>
inline float to_float_impl<ggml_fp16_t>(const ggml_fp16_t& val) {
    return ggml_fp16_to_fp32(val);
}

template<>
inline float to_float_impl<ggml_bf16_t>(const ggml_bf16_t& val) {
    return ggml_bf16_to_fp32(val);
}

template<typename T>
inline T from_float_impl(float val) {
    return static_cast<T>(val);
}

template<>
inline ggml_fp16_t from_float_impl<ggml_fp16_t>(float val) {
    return ggml_fp32_to_fp16(val);
}

template<>
inline ggml_bf16_t from_float_impl<ggml_bf16_t>(float val) {
    return ggml_fp32_to_bf16(val);
}

// ============================================================================
// Error Metrics (internal)
// ============================================================================

template<typename T>
float compute_nmse(const T* tensor1, const T* tensor2, size_t num_elements) {
    double sum_diff_sq = 0.0;
    double sum_ref_sq = 0.0;

    for (size_t i = 0; i < num_elements; i++) {
        const float val1 = to_float_impl(tensor1[i]);
        const float val2 = to_float_impl(tensor2[i]);
        const float diff = val1 - val2;

        sum_diff_sq += diff * diff;
        sum_ref_sq += val2 * val2;
    }

    return std::sqrt(sum_diff_sq / (sum_ref_sq + 1e-12));
}

template<typename T>
float compute_max_abs_diff(const T* tensor1, const T* tensor2, size_t num_elements) {
    float max_diff = 0.0f;
    for (size_t i = 0; i < num_elements; i++) {
        float diff = std::abs(to_float_impl(tensor1[i]) - to_float_impl(tensor2[i]));
        max_diff = std::max(max_diff, diff);
    }
    return max_diff;
}

template<typename T>
float compute_mean_abs_diff(const T* tensor1, const T* tensor2, size_t num_elements) {
    double sum_abs_diff = 0.0;
    for (size_t i = 0; i < num_elements; i++) {
        sum_abs_diff += std::abs(to_float_impl(tensor1[i]) - to_float_impl(tensor2[i]));
    }
    return sum_abs_diff / num_elements;
}

} // namespace (internal helpers)

// ============================================================================
// Golden Reference: Scaled Dot-Product Attention
// ============================================================================

/**
 * CPU Reference Implementation of Scaled Dot-Product Attention
 *
 * Computes: Attention(Q, K, V) = softmax(Q @ K^T / scale + mask) @ V
 *
 * Input formats (GGML/llama.cpp native):
 * - Q, K, V: [D, S, H, B] (DSHB format)
 * - mask: [1, 1, Sq, Sk] or nullptr
 *
 * Output format (GGML/llama.cpp native):
 * - output: [D, H, S, B] (DHSB format)
 *
 * Note: This uses a numerically stable 3-pass softmax implementation.
 */
template<typename T_in, typename T_out>
bool verify_attention_golden(
    const T_in* q_data,         // Query: [D, Sq, H, B]
    const T_in* k_data,         // Key: [D, Sk, H, B]
    const T_in* v_data,         // Value: [D, Sk, H, B]
    const T_in* mask_data,      // Mask: [1, 1, Sq, Sk] or nullptr
    const T_out* gpu_output,    // GPU output: [D, H, Sq, B]
    int B, int H, int Sq, int Sk, int D,
    float scale,
    bool is_causal,
    float tolerance
) {
    printf("=== GOLDEN VERIFICATION START ===\n");
    printf("Config: B=%d, H=%d, Sq=%d, Sk=%d, D=%d\n", B, H, Sq, Sk, D);
    printf("Params: scale=%.6f, is_causal=%s, has_mask=%s\n",
           scale, is_causal ? "true" : "false", mask_data ? "true" : "false");
    printf("Types: input=%s, output=%s\n", typeid(T_in).name(), typeid(T_out).name());

    // Allocate CPU reference output in DHSB format
    std::vector<T_out> cpu_output(B * H * Sq * D);

    // Compute attention for each batch and head
    for (int b = 0; b < B; b++) {
        for (int h = 0; h < H; h++) {
            // ====================================================================
            // STAGE 1: Compute attention scores (Q @ K^T)
            // ====================================================================
            std::vector<float> attn_scores(Sq * Sk);
            for (int i = 0; i < Sq; i++) {
                for (int j = 0; j < Sk; j++) {
                    float score = 0.0f;
                    for (int d = 0; d < D; d++) {
                        // Q[D, Sq, H, B]: index = d*Sq*H*B + i*H*B + h*B + b
                        // K[D, Sk, H, B]: index = d*Sk*H*B + j*H*B + h*B + b
                        const size_t q_idx = d*Sq*H*B + i*H*B + h*B + b;
                        const size_t k_idx = d*Sk*H*B + j*H*B + h*B + b;

                        float q_val = to_float_impl(q_data[q_idx]);
                        float k_val = to_float_impl(k_data[k_idx]);
                        score += q_val * k_val;
                    }
                    attn_scores[i*Sk + j] = score * scale;
                }
            }

            // ====================================================================
            // STAGE 2: Apply masks and softmax
            // ====================================================================

            // Apply causal mask (upper triangular mask)
            if (is_causal) {
                for (int i = 0; i < Sq; i++) {
                    for (int j = 0; j < Sk; j++) {
                        // Causal: query position i can only attend to key positions [0, i]
                        if (j > i) {
                            attn_scores[i*Sk + j] = -INFINITY;
                        }
                    }
                }
            }

            // Apply explicit mask (currently disabled - enable if needed)
            if (mask_data != nullptr) {
                for (int i = 0; i < Sq; i++) {
                    for (int j = 0; j < Sk; j++) {
                        // Mask format: [1, 1, Sq, Sk]
                        const size_t mask_idx = i*Sk + j;
                        const float mask_val = to_float_impl(mask_data[mask_idx]);

                        // If mask value is -inf or very negative, mask out this position
                        if (!std::isfinite(mask_val) || mask_val < -1e10f) {
                            attn_scores[i*Sk + j] = -INFINITY;
                        } else {
                            // Otherwise add mask value (for attention bias)
                            attn_scores[i*Sk + j] += mask_val;
                        }
                    }
                }
            }

            // Numerically stable softmax (3-pass algorithm)
            // Reference: https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf
            std::vector<float> attn_weights(Sq * Sk);
            for (int i = 0; i < Sq; i++) {
                // Pass 1: Find row-wise max
                float max_val = -INFINITY;
                for (int j = 0; j < Sk; j++) {
                    max_val = std::max(max_val, attn_scores[i*Sk + j]);
                }

                // Handle degenerate case: all -INF (no valid attention positions)
                if (!std::isfinite(max_val)) {
                    for (int j = 0; j < Sk; j++) {
                        attn_weights[i*Sk + j] = 0.0f;
                    }
                    continue;
                }

                // Pass 2: Compute exp and sum
                std::vector<float> exp_vals(Sk);
                double sum_exp = 0.0;
                for (int j = 0; j < Sk; j++) {
                    float val = std::exp(attn_scores[i*Sk + j] - max_val);
                    exp_vals[j] = val;
                    sum_exp += (double)val;
                }

                // Pass 3: Normalize to get probabilities
                double inv_sum = 1.0 / (sum_exp + 1e-20);
                for (int j = 0; j < Sk; j++) {
                    attn_weights[i*Sk + j] = (float)(exp_vals[j] * inv_sum);
                }
            }

            // ====================================================================
            // STAGE 3: Compute output (attention_weights @ V)
            // ====================================================================
            for (int i = 0; i < Sq; i++) {
                for (int d = 0; d < D; d++) {
                    float result = 0.0f;
                    for (int j = 0; j < Sk; j++) {
                        // V[D, Sk, H, B]: index = d*Sk*H*B + j*H*B + h*B + b
                        const size_t v_idx = d*Sk*H*B + j*H*B + h*B + b;
                        result += attn_weights[i*Sk + j] * to_float_impl(v_data[v_idx]);
                    }

                    // Output[D, H, Sq, B]: index = d*H*Sq*B + h*Sq*B + i*B + b
                    const size_t out_idx = d*H*Sq*B + h*Sq*B + i*B + b;
                    cpu_output[out_idx] = from_float_impl<T_out>(result);
                }
            }
        }
    }

    // ====================================================================
    // Compute error metrics
    // ====================================================================
    const size_t total_elements = B * H * Sq * D;

    float nmse = compute_nmse(gpu_output, cpu_output.data(), total_elements);
    float max_abs_diff = compute_max_abs_diff(gpu_output, cpu_output.data(), total_elements);
    float mean_abs_diff = compute_mean_abs_diff(gpu_output, cpu_output.data(), total_elements);

    printf("\n--- Error Metrics ---\n");
    printf("NMSE (Normalized MSE):  %.9f %s %.9f\n",
           nmse, nmse <= tolerance ? "<=" : ">", tolerance);
    printf("Max Absolute Diff:      %.9f\n", max_abs_diff);
    printf("Mean Absolute Diff:     %.9f\n", mean_abs_diff);

    bool passed = (nmse <= tolerance);

    if (!passed) {
        GGML_LOG_WARN("GOLDEN VERIFICATION FAILED!\n");

        // Print sample values for debugging
        const int max_samples = std::min(10, (int)total_elements);
        printf("\n--- Sample Comparison (first %d values) ---\n", max_samples);
        for (int i = 0; i < max_samples; i++) {
            printf("[%d]: GPU=%.6f, CPU_ref=%.6f, diff=%.6f\n",
                   i,
                   to_float_impl(gpu_output[i]),
                   to_float_impl(cpu_output[i]),
                   to_float_impl(gpu_output[i]) - to_float_impl(cpu_output[i]));
        }
    } else {
        printf("\n=== GOLDEN VERIFICATION PASSED ===\n");
    }

    return passed;
}

// ============================================================================
// Explicit Template Instantiations
// ============================================================================

// Export instantiations for common types
template bool verify_attention_golden<ggml_fp16_t, ggml_fp16_t>(
    const ggml_fp16_t*, const ggml_fp16_t*, const ggml_fp16_t*, const ggml_fp16_t*,
    const ggml_fp16_t*, int, int, int, int, int, float, bool, float);

template bool verify_attention_golden<float, float>(
    const float*, const float*, const float*, const float*,
    const float*, int, int, int, int, int, float, bool, float);

template bool verify_attention_golden<ggml_bf16_t, ggml_bf16_t>(
    const ggml_bf16_t*, const ggml_bf16_t*, const ggml_bf16_t*, const ggml_bf16_t*,
    const ggml_bf16_t*, int, int, int, int, int, float, bool, float);

#endif // GGML_USE_DLFA

