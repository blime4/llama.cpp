// SNAC vocoder ggml-based implementation for GPU acceleration
// Phase 1: Single sequence processing with ggml graph

#pragma once

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <vector>
#include <string>
#include <cstring>

// SNAC model constants (snac_24khz from hubertsiuzdak/snac_24khz)
static const int SNAC_GGML_FRAME_SIZE = 7;
static const int SNAC_GGML_SAMPLE_RATE = 24000;
static const int SNAC_GGML_UPSAMPLE_FACTOR = 512;
static const int SNAC_GGML_CODEBOOK_SIZE = 4096;
static const int SNAC_GGML_DECODER_DIM = 1024;     // Decoder dimension (from config.json decoder_dim)
static const int SNAC_GGML_QUANTIZER_DIM = 768;     // Quantizer output dimension (latent_dim = encoder_dim * 16 = 48 * 16)
static const int SNAC_GGML_CODEBOOK_DIM = 8;
static const int SNAC_GGML_N_QUANTIZERS = 3;
static const int SNAC_GGML_N_DECODER_LAYERS = 4;

// SNAC ggml weights structure - holds all model weights as ggml tensors
struct snac_ggml_weights {
    // Input convolutions
    struct ggml_tensor * in_conv_kernel;   // [1024, 7] depthwise conv (QUANTIZER_DIM, kernel_size)
    struct ggml_tensor * in_conv_bias;     // [1024]
    struct ggml_tensor * up_conv_kernel;   // [1536, 1024] 1x1 conv (first_decoder_ch, QUANTIZER_DIM)
    struct ggml_tensor * up_conv_bias;     // [1536]

    // Attention layer (optional)
    struct ggml_tensor * attn_norm_weight; // [1024]
    struct ggml_tensor * attn_norm_bias;   // [1024]
    struct ggml_tensor * attn_to_qkv;      // [3072, 1024]
    struct ggml_tensor * attn_inv_freq;    // [32]
    struct ggml_tensor * attn_to_out;      // [1024, 1024]

    // Output layer
    struct ggml_tensor * out_conv_kernel;  // [1, 64, 7]
    struct ggml_tensor * out_conv_bias;    // [1] or nullptr
    struct ggml_tensor * snake_alpha_out;  // [64]

    // Decoder layers (4 layers)
    struct ggml_tensor * layer_alpha[4];           // [in_channels]
    struct ggml_tensor * layer_kernel[4];          // [in, out, k]
    struct ggml_tensor * layer_bias[4];            // [out]
    struct ggml_tensor * layer_noise_kernel[4];    // [out, out] or nullptr

    // Residual units per layer (3 units per layer)
    struct ggml_tensor * residual_in_alpha[4][3];   // [channels]
    struct ggml_tensor * residual_in_kernel[4][3];  // [ch, 1, 7] depthwise
    struct ggml_tensor * residual_in_bias[4][3];    // [ch]
    struct ggml_tensor * residual_out_alpha[4][3];  // [channels]
    struct ggml_tensor * residual_out_kernel[4][3]; // [ch, ch]
    struct ggml_tensor * residual_out_bias[4][3];   // [ch]

    // Quantizers (3 quantizers for TTS)
    struct ggml_tensor * quant_codebook[3];       // [4096, 8]
    struct ggml_tensor * quant_out_proj[3];       // [8, 1024] - projects from codebook_dim to quantizer_dim
    struct ggml_tensor * quant_out_bias[3];       // [1024]

    snac_ggml_weights() {
        memset(this, 0, sizeof(*this));
    }
};

// SNAC ggml context - manages ggml resources
struct snac_ggml_context {
    struct ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_backend_buffer_t buffer = nullptr;
    struct snac_ggml_weights weights;

    int n_quantizers = SNAC_GGML_N_QUANTIZERS;
    int decoder_rates[4] = {8, 8, 4, 2};  // From GGUF metadata: decoder_rate_0..3
    int vq_strides[4] = {4, 2, 1, 1};     // From GGUF metadata: vq_stride_0..3 (snac_24khz has [4, 2, 1])

    bool loaded = false;
};

// Initialize SNAC ggml context
// Returns true on success
bool snac_ggml_init(
    struct snac_ggml_context & ctx,
    const char * model_path,
    ggml_backend_t backend = nullptr);

// Free SNAC ggml context
void snac_ggml_free(struct snac_ggml_context & ctx);

// Decode audio tokens to waveform
// pyramid_tokens: vector of 3 codebook token vectors (head0, head1, head2)
// Returns: vector of PCM samples in range [-1, 1]
std::vector<float> snac_ggml_decode(
    struct snac_ggml_context & ctx,
    const std::vector<std::vector<int>> & pyramid_tokens);

// ============================================================================
// Phase 2: Batched Processing Structures
// ============================================================================

// Batched input structure for multiple sequences
struct snac_batch_input {
    int32_t * tokens;         // Packed token data: [n_seqs][max_tokens][3] or separate heads
                              // Layout: tokens[seq * max_tokens * 3 + t * 3 + head]
    int32_t * seq_lengths;    // [n_seqs] actual token lengths per sequence
    int32_t n_seqs;           // Number of sequences in batch
    int32_t max_tokens;       // Maximum token count across sequences (padded)

    // Alternative: Separate token arrays for each quantizer head
    int32_t * tokens_head0;   // [n_seqs * max_tokens] - quantizer 0 tokens
    int32_t * tokens_head1;   // [n_seqs * max_tokens/2] - quantizer 1 tokens
    int32_t * tokens_head2;   // [n_seqs * max_tokens/4] - quantizer 2 tokens
    int32_t head0_len;        // tokens per sequence for head 0
    int32_t head1_len;        // tokens per sequence for head 1
    int32_t head2_len;        // tokens per sequence for head 2
};

// Batched output structure
struct snac_batch_output {
    float * pcm;              // [n_seqs][max_pcm_samples] interleaved or packed
    int32_t * pcm_lengths;    // [n_seqs] actual PCM samples per sequence
    int32_t n_seqs;           // Number of sequences
    int32_t max_pcm_samples;  // Maximum PCM samples per sequence (padded)
};

// Batched decode context - manages memory pool for repeated batch processing
struct snac_batch_context {
    struct ggml_context * ctx = nullptr;
    ggml_backend_t backend = nullptr;
    ggml_gallocr_t allocr = nullptr;

    // Pre-allocated buffers for common batch sizes
    int max_batch_size = 0;
    int max_tokens = 0;

    // Reusable graph structures
    struct ggml_cgraph * gf = nullptr;

    bool initialized = false;
};

// Initialize batch context with memory pool
// Returns true on success
bool snac_batch_init(
    struct snac_ggml_context & model_ctx,
    struct snac_batch_context & batch_ctx,
    int max_batch_size = 16,
    int max_tokens = 512);

// Free batch context resources
void snac_batch_free(struct snac_batch_context & batch_ctx);

// Batched decode - process multiple sequences in one graph execution
// Returns: vector of PCM outputs, one per input sequence
std::vector<std::vector<float>> snac_batch_decode(
    struct snac_ggml_context & model_ctx,
    struct snac_batch_context & batch_ctx,
    const std::vector<snac_batch_input> & batches);

// Convenience function: Batch decode with pyramid token format
// pyramid_tokens_batch: vector of pyramid_tokens, one per sequence
// Returns: vector of PCM outputs, one per sequence
std::vector<std::vector<float>> snac_batch_decode_pyramid(
    struct snac_ggml_context & model_ctx,
    struct snac_batch_context & batch_ctx,
    const std::vector<std::vector<std::vector<int>>> & pyramid_tokens_batch);
