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
static const int SNAC_GGML_SAMPLES_PER_FRAME = 2048;  // Each streaming frame produces 2048 samples (4 head2 tokens * 512)
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

// ============================================================================
// Phase 4: Streaming Processing Structures
// ============================================================================

// Streaming token buffer - accumulates tokens until frame boundary
// The SNAC vocoder uses a pyramid token structure where each frame consists
// of 7 tokens distributed across 3 quantizer heads:
//   Position: |  0  |  1  |  2  |  3  |  4  |  5  |  6  |
//   Head:     |  0  |  1  |  2  |  2  |  1  |  2  |  2  |
// This means per frame: 1 token for head0, 2 for head1, 4 for head2
struct snac_streaming_buffer {
    // Pyramid token buffers for each head
    std::vector<int32_t> head0_tokens;  // Quantizer 0 tokens (1 per frame)
    std::vector<int32_t> head1_tokens;  // Quantizer 1 tokens (2 per frame)
    std::vector<int32_t> head2_tokens;  // Quantizer 2 tokens (4 per frame)

    // Frame accumulation state
    int frame_position = 0;  // Current position in frame (0-6)

    // Pyramid mapping: position -> which head this token belongs to
    // [0, 1, 2, 2, 1, 2, 2] means: pos0->head0, pos1->head1, pos2->head2, etc.
    static constexpr int PYRAMID_MAP[7] = {0, 1, 2, 2, 1, 2, 2};

    // Token value constants
    static constexpr int CODEBOOK_SIZE = SNAC_GGML_CODEBOOK_SIZE;

    // Statistics
    int total_frames_completed = 0;

    // Reset buffer state
    void reset();

    // Add a single audio token, returns true if frame completed
    // raw_token: token value in range [0, CODEBOOK_SIZE-1]
    bool add_token(int32_t raw_token);

    // Check if we have at least N complete frames
    bool has_frames(int n) const;

    // Get completed frames (does not consume)
    // Returns vector of 3 token vectors (head0, head1, head2)
    // max_frames: -1 for all, otherwise limit to N frames
    std::vector<std::vector<int32_t>> get_completed_frames(int max_frames = -1) const;

    // Consume frames from buffer (after successful decode)
    void consume_frames(int n_frames);

    // Get total frame count
    int get_frame_count() const { return total_frames_completed; }
};

// ============================================================================
// Phase 4.2: Streaming Decode Configuration and Context
// ============================================================================

// Streaming decode configuration
struct snac_streaming_config {
    int min_chunk_frames = 8;      // Minimum frames before decode
    int overlap_frames = 4;         // Overlap context for clean boundaries (default 4 frames)
    bool crossfade_chunks = false;  // Apply crossfade at chunk boundaries
    int crossfade_samples = 256;    // Crossfade duration in samples
};

// Audio output callback type
// Parameters: pcm_data, num_samples, user_data
// Returns: true to continue, false to abort
typedef bool (*snac_audio_callback)(
    const float * pcm_data,
    int num_samples,
    void * user_data
);

// Streaming decode context - manages state for incremental decoding
struct snac_streaming_context {
    snac_ggml_context * model_ctx = nullptr;
    snac_streaming_buffer buffer;
    snac_streaming_config config;

    // Full-context streaming with audio caching:
    // We keep the FULL decoded audio from the last decode
    // When new tokens arrive, we decode again and only output the NEW samples
    // This ensures: (1) vocoder sees full context, (2) no discontinuity from re-decode
    std::vector<float> cached_full_audio;  // All audio decoded so far
    int samples_already_output = 0;         // How many samples sent to callback
    int last_output_frame_count = 0;        // Frame count at last output

    // Statistics
    int total_pcm_samples = 0;
    int chunks_decoded = 0;

    // Initialize streaming context
    bool init(snac_ggml_context * ctx, const snac_streaming_config & cfg);

    // Add token and decode if chunk ready
    // Returns true if audio was output via callback
    bool add_token_and_decode(
        int raw_token,
        snac_audio_callback callback,
        void * user_data
    );

    // Flush remaining tokens (call at end of stream)
    void flush(snac_audio_callback callback, void * user_data);

    // Reset for new stream
    void reset();

    // Get statistics
    int get_total_samples() const { return total_pcm_samples; }
    int get_chunks_decoded() const { return chunks_decoded; }
};
