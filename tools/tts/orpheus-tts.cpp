// Orpheus-TTS inference tool for llama.cpp
//
// Orpheus-TTS is an emotion-expressive TTS model that uses:
// - A LLaMA-based LLM to generate discrete audio tokens
// - A SNAC neural vocoder to convert tokens to waveform
//
// Usage: orpheus-tts -m model.gguf -p "Hello world" -o output.wav

#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "log.h"
#include "llama.h"
#include "gguf.h"
#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include "snac-ggml.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <regex>
#include <string>
#include <vector>

// Orpheus-TTS specific token IDs
static const llama_token TOKEN_BOS = 128000;
static const llama_token TOKEN_EOS = 128001;
static const llama_token TOKEN_START_SPEECH = 128259;
static const llama_token TOKEN_END_TURN = 128009;
static const llama_token TOKEN_START_RESPONSE = 128260;
static const llama_token TOKEN_END_RESPONSE = 128261;
static const llama_token TOKEN_AUDIO_START = 128257;
static const llama_token TOKEN_STOP = 128258;

// Audio token range (custom tokens for discrete audio codes)
static const llama_token AUDIO_TOKEN_START = 128266;
static const llama_token AUDIO_TOKEN_END = 156937;

// SNAC vocoder parameters (snac_24khz model)
static const int SNAC_FRAME_SIZE = 7;  // 7 tokens per frame (pyramid structure for TTS output)
static const int SNAC_SAMPLE_RATE = 24000;
static const int SNAC_UPSAMPLE_FACTOR = 512;  // 8*8*4*2 = 512 for snac_24khz (from config.json decoder_rates)
static const int SNAC_CODEBOOK_SIZE = 4096;
static const int SNAC_DECODER_DIM = 1024;  // snac_24khz decoder dimension (after up_conv, from config.json decoder_dim)
static const int SNAC_QUANTIZER_DIM = 768; // snac_24khz quantizer output dimension (before up_conv, from tensor shapes)
static const int SNAC_CODEBOOK_DIM = 8;    // snac_24khz codebook dimension
static const int SNAC_N_QUANTIZERS = 3;    // snac_24khz has 3 quantizers (from config.json vq_strides length)
static const int SNAC_N_DECODER_LAYERS = 4;
static const int SNAC_DECODER_RATES[] = {8, 8, 4, 2};  // snac_24khz decoder rates (from config.json)
static const int SNAC_VQ_STRIDES[] = {8, 4, 2, 1};     // snac_24khz vq strides

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1; // PCM
    uint16_t num_channels = 1; // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

static bool save_wav16(const std::string & fname, const std::vector<float> & data, int sample_rate) {
    std::ofstream file(fname, std::ios::binary);
    if (!file) {
        LOG_ERR("%s: Failed to open file '%s' for writing.\n", __func__, fname.c_str());
        return false;
    }

    wav_header header;
    header.sample_rate = sample_rate;
    header.byte_rate = header.sample_rate * header.num_channels * (header.bits_per_sample / 8);
    header.block_align = header.num_channels * (header.bits_per_sample / 8);
    header.data_size = data.size() * (header.bits_per_sample / 8);
    header.chunk_size = 36 + header.data_size;

    file.write(reinterpret_cast<const char*>(&header), sizeof(header));

    for (const auto & sample : data) {
        int16_t pcm_sample = static_cast<int16_t>(std::clamp(sample * 32767.0, -32768.0, 32767.0));
        file.write(reinterpret_cast<const char*>(&pcm_sample), sizeof(pcm_sample));
    }

    return file.good();
}

static void print_usage(int, char ** argv) {
    LOG("\nOrpheus-TTS: Emotion-expressive Text-to-Speech\n");
    LOG("\nusage: %s [options]\n", argv[0]);
    LOG("\noptions:\n");
    LOG("  -h, --help            show this help message and exit\n");
    LOG("  -m, --model PATH      path to LLM GGUF model (required)\n");
    LOG("  --model-vocoder PATH  path to SNAC vocoder GGUF model (required)\n");
    LOG("  -p, --prompt TEXT     text to synthesize (required)\n");
    LOG("  -o, --output PATH     output WAV file (default: output.wav)\n");
    LOG("  -v, --voice NAME      voice/speaker name (optional)\n");
    LOG("  -t, --threads N       number of threads (default: auto)\n");
    LOG("  -n, --n-predict N     max tokens to generate (default: 2048)\n");
    LOG("  --temp N              temperature (default: 0.1)\n");
    LOG("  --top-k N             top-k sampling (default: 40)\n");
    LOG("  --top-p N             top-p sampling (default: 0.9)\n");
    LOG("  -ngl, --n-gpu-layers N  number of layers to offload to GPU (default: -1 = all)\n");
    LOG("  --use-snac-ggml       use ggml-based SNAC vocoder (enables GPU acceleration)\n");
    LOG("  --batch-size N        batch size for SNAC processing (default: 1)\n");
    LOG("  --gpu                 use GPU backend for SNAC vocoder\n");
    LOG("  --streaming           enable streaming TTS (incremental audio output)\n");
    LOG("  --streaming-chunk-frames N  frames per chunk (default: 8)\n");
    LOG("  --streaming-overlap N overlap frames for clean boundaries (default: 4)\n");
    LOG("\nexample:\n");
    LOG("  %s -m orpheus-3b-f16.gguf --model-vocoder snac-24khz-f16.gguf \\\n", argv[0]);
    LOG("      -p \"Hello, how are you?\" -o greeting.wav\n");
    LOG("\n");
}

// ============================================================================
// Streaming TTS Support (Phase 3)
// ============================================================================

// Streaming callback context for collecting audio chunks
struct streaming_callback_ctx {
    std::vector<float> * all_samples;  // Accumulate all PCM samples
    int chunks_received;
    bool verbose;

    streaming_callback_ctx() : all_samples(nullptr), chunks_received(0), verbose(true) {}
};

// Audio callback for streaming TTS - receives decoded audio chunks
static bool streaming_audio_callback(
    const float * pcm_data,
    int num_samples,
    void * user_data
) {
    auto * ctx = static_cast<streaming_callback_ctx *>(user_data);
    if (!ctx) return false;

    // Accumulate samples for final output
    if (ctx->all_samples) {
        ctx->all_samples->insert(ctx->all_samples->end(),
                                  pcm_data, pcm_data + num_samples);
    }

    ctx->chunks_received++;

    if (ctx->verbose) {
        float duration = (float)num_samples / SNAC_GGML_SAMPLE_RATE;
        LOG_INF("Streaming chunk %d: %d samples (%.2fs)\n",
                ctx->chunks_received, num_samples, duration);
    }

    return true;  // Continue streaming
}

// Build the prompt for Orpheus-TTS
static std::vector<llama_token> build_orpheus_prompt(
    const llama_vocab * vocab,
    const std::string & text,
    const std::string & voice
) {
    std::vector<llama_token> tokens;

    // Start token
    tokens.push_back(TOKEN_START_SPEECH);
    tokens.push_back(TOKEN_BOS);

    // Add voice prefix if specified
    std::string prompt_text = voice.empty() ? text : voice + ": " + text;

    // Tokenize the text using common_tokenize
    auto text_tokens = common_tokenize(vocab, prompt_text, false, true);
    tokens.insert(tokens.end(), text_tokens.begin(), text_tokens.end());

    // End turn and start audio generation
    tokens.push_back(TOKEN_END_TURN);
    tokens.push_back(TOKEN_START_RESPONSE);
    tokens.push_back(TOKEN_END_RESPONSE);
    tokens.push_back(TOKEN_AUDIO_START);

    // Debug: print prompt tokens
    LOG_INF("Prompt tokens (%zu): ", tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        printf("%d ", tokens[i]);
    }
    printf("\n");
    fflush(stdout);

    return tokens;
}

// Collect audio tokens from the generated sequence
// Returns tokens organized by codebook (3 heads)
// Uses pyramid pre-order binary tree traversal matching SNAC's expected ordering.
// For 3 quantizers, the pyramid traversal is: [0, 1, 2, 2, 1, 2, 2]
// Each position within a frame has a position-based offset: pos * codebook_size
// that must be subtracted to get the actual codebook index.
static std::vector<std::vector<int>> collect_audio_tokens_pyramid(const std::vector<llama_token> & tokens) {
    std::vector<int> all_tokens;

    for (const auto & token : tokens) {
        if (token >= AUDIO_TOKEN_START && token <= AUDIO_TOKEN_END) {
            all_tokens.push_back(token - AUDIO_TOKEN_START);
        }
    }

    // Pyramid structure via pre-order binary tree traversal:
    // Level 0 (root), Level 1 (left), Level 2 (left-left), Level 2 (left-right),
    // Level 1 (right), Level 2 (right-left), Level 2 (right-right)
    // = [0, 1, 2, 2, 1, 2, 2]
    static const int pyramid_map[SNAC_FRAME_SIZE] = {0, 1, 2, 2, 1, 2, 2};

    std::vector<std::vector<int>> result(3);
    size_t n_frames = all_tokens.size() / SNAC_FRAME_SIZE;

    for (size_t f = 0; f < n_frames; f++) {
        for (int pos = 0; pos < SNAC_FRAME_SIZE; pos++) {
            int codebook = pyramid_map[pos];
            // Subtract position-based offset to get actual codebook index
            int id = all_tokens[f * SNAC_FRAME_SIZE + pos] - pos * SNAC_CODEBOOK_SIZE;
            if (id < 0 || id >= SNAC_CODEBOOK_SIZE) {
                // Invalid token, log and skip
                if (f < 3) {
                    LOG_WRN("DEBUG: Invalid token at frame %zu pos %d: raw=%d, id=%d (expected 0-%d)\n",
                            f, pos, all_tokens[f * SNAC_FRAME_SIZE + pos], id, SNAC_CODEBOOK_SIZE-1);
                }
                continue;
            }
            result[codebook].push_back(id);
        }
    }

    return result;
}

// ============================================================================
// SNAC Vocoder Implementation
// ============================================================================

// Snake1D activation function
// snake(x, alpha) = x + (sin(alpha * x)^2) / alpha
static void snake_1d_inplace(float * data, const float * alpha, int64_t n, int64_t channels) {
    for (int64_t i = 0; i < n; i++) {
        for (int64_t c = 0; c < channels; c++) {
            float x = data[i * channels + c];
            float a = alpha[c];
            float sin_val = std::sin(a * x);
            data[i * channels + c] = x + (sin_val * sin_val) / (a + 1e-9f);
        }
    }
}

// 1D Convolution (depthwise with groups)
// PyTorch Conv1d weight layout: [out_channels, in_channels/groups, kernel_size]
// Data is stored in row-major order (PyTorch/numpy convention)
static void conv1d_dw(
    const float * input, const float * kernel, const float * bias,
    float * output,
    int64_t input_len, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups
) {
    int64_t output_len = (input_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
    int64_t in_ch_per_group = in_channels / groups;
    int64_t out_ch_per_group = out_channels / groups;

    for (int64_t g = 0; g < groups; g++) {
        int64_t in_ch_start = g * in_ch_per_group;
        int64_t out_ch_start = g * out_ch_per_group;

        for (int64_t i = 0; i < output_len; i++) {
            for (int64_t oc = 0; oc < out_ch_per_group; oc++) {
                float sum = bias ? bias[out_ch_start + oc] : 0.0f;

                for (int64_t k = 0; k < kernel_size; k++) {
                    int64_t input_pos = i * stride + k * dilation - padding;

                    if (input_pos >= 0 && input_pos < input_len) {
                        for (int64_t ic = 0; ic < in_ch_per_group; ic++) {
                            // PyTorch weight[out_ch, in_ch_local, k]
                            // flat_index = (out_ch_start+oc) * in_ch_per_group * kernel_size + ic * kernel_size + k
                            float w = kernel[(out_ch_start + oc) * in_ch_per_group * kernel_size + ic * kernel_size + k];
                            sum += input[input_pos * in_channels + in_ch_start + ic] * w;
                        }
                    }
                }

                output[i * out_channels + out_ch_start + oc] = sum;
            }
        }
    }
}

// 1D Transposed Convolution
// Implements the same algorithm as chatllm.cpp:
// 1. Insert (stride-1) zeros between input elements
// 2. Flip kernel along kernel dimension
// 3. Apply regular 1D convolution
// Weight layout: [in_channels, out_channels, kernel_size]
static void conv_transpose1d(
    const float * input, const float * kernel, const float * bias,
    float * output,
    int64_t input_len, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding
) {
    int64_t output_len = (input_len - 1) * stride + kernel_size - 2 * padding + output_padding;

    // Initialize output with bias
    for (int64_t i = 0; i < output_len * out_channels; i++) {
        output[i] = bias ? bias[i % out_channels] : 0.0f;
    }

    // Method: Insert zeros between input elements, then convolve with flipped kernel
    // This matches chatllm.cpp's approach:
    // 1. Insert (stride-1) zeros between input elements
    // 2. Flip kernel along kernel dimension
    // 3. Apply regular convolution with p = kernel_size - 1 - padding

    // Effective padding for the equivalent conv1d
    int64_t p = kernel_size - 1 - padding;

    // Create input with zeros inserted: [expanded_len, in_channels]
    int64_t expanded_len = (input_len - 1) * stride + 1;
    std::vector<float> expanded(expanded_len * in_channels, 0.0f);

    // Copy input with stride spacing
    for (int64_t i = 0; i < input_len; i++) {
        for (int64_t ic = 0; ic < in_channels; ic++) {
            expanded[i * stride * in_channels + ic] = input[i * in_channels + ic];
        }
    }

    // Apply conv1d with flipped kernel
    // This is equivalent to ConvTranspose1D
    for (int64_t o = 0; o < output_len; o++) {
        for (int64_t k = 0; k < kernel_size; k++) {
            int64_t input_pos = o + k - p;  // Use effective padding p
            if (input_pos >= 0 && input_pos < expanded_len) {
                int64_t k_flipped = kernel_size - 1 - k;  // FLIP the kernel!
                for (int64_t oc = 0; oc < out_channels; oc++) {
                    float sum = 0.0f;
                    for (int64_t ic = 0; ic < in_channels; ic++) {
                        float w = kernel[ic * out_channels * kernel_size + oc * kernel_size + k_flipped];
                        sum += expanded[input_pos * in_channels + ic] * w;
                    }
                    output[o * out_channels + oc] += sum;
                }
            }
        }
    }
}

// 1D Convolution
// PyTorch Conv1d weight layout: [out_channels, in_channels, kernel_size]
// Data is stored in row-major order (PyTorch/numpy convention)
static void conv1d(
    const float * input, const float * kernel, const float * bias,
    float * output,
    int64_t input_len, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation
) {
    int64_t output_len = (input_len + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

    for (int64_t i = 0; i < output_len; i++) {
        for (int64_t oc = 0; oc < out_channels; oc++) {
            float sum = bias ? bias[oc] : 0.0f;

            for (int64_t k = 0; k < kernel_size; k++) {
                int64_t input_pos = i * stride + k * dilation - padding;

                if (input_pos >= 0 && input_pos < input_len) {
                    for (int64_t ic = 0; ic < in_channels; ic++) {
                        // PyTorch weight[oc, ic, k]
                        // flat_index = oc * in_channels * kernel_size + ic * kernel_size + k
                        float w = kernel[oc * in_channels * kernel_size + ic * kernel_size + k];
                        sum += input[input_pos * in_channels + ic] * w;
                    }
                }
            }

            output[i * out_channels + oc] = sum;
        }
    }
}

// LocalMHA: Local Multi-Head Attention with Rotary Position Embeddings
// Implements windowed self-attention matching Python SNAC's LocalMHA
// Data layout: [T, C] (time-major, single batch)
static void local_mha_forward(
    float * data,               // [T, C] input/output (in-place with residual)
    int64_t T, int64_t C,
    const float * norm_weight,  // [C] LayerNorm weight
    const float * norm_bias,    // [C] LayerNorm bias
    const float * qkv_weight,   // [C*3, C] linear (no bias)
    const float * out_weight,   // [C, C] linear (no bias)
    const float * inv_freq,     // [dim_head/2] for RoPE
    int window_size,
    int dim_head
) {
    int heads = C / dim_head;
    int half_head = dim_head / 2;
    int n_windows = T / window_size;

    // If T is not divisible by window_size, skip attention (safety)
    if (n_windows <= 0 || T % window_size != 0) {
        LOG_WRN("local_mha_forward: T=%lld not divisible by window_size=%d, skipping attention\n",
                (long long)T, window_size);
        return;
    }

    // Save residual (data is [T, C])
    std::vector<float> residual(data, data + T * C);

    // Step 1: LayerNorm over C dimension for each time step
    // Input: [T, C], normalize over C
    for (int64_t t = 0; t < T; t++) {
        float * row = &data[t * C];
        // Compute mean
        float mean = 0.0f;
        for (int c = 0; c < C; c++) mean += row[c];
        mean /= C;
        // Compute variance
        float var = 0.0f;
        for (int c = 0; c < C; c++) {
            float d = row[c] - mean;
            var += d * d;
        }
        var /= C;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        // Normalize and apply affine
        for (int c = 0; c < C; c++) {
            row[c] = (row[c] - mean) * inv_std * norm_weight[c] + norm_bias[c];
        }
    }

    // Step 2: QKV projection: [T, C] @ [C, 3*C] -> [T, 3*C]
    // qkv_weight is [3*C, C] in PyTorch (row-major), so weight[out, in]
    // output[t, o] = sum_i input[t, i] * weight[o, i]
    std::vector<float> qkv(T * 3 * C);
    for (int64_t t = 0; t < T; t++) {
        for (int o = 0; o < 3 * C; o++) {
            float sum = 0.0f;
            for (int i = 0; i < C; i++) {
                sum += data[t * C + i] * qkv_weight[o * C + i];
            }
            qkv[t * 3 * C + o] = sum;
        }
    }

    // Step 3: Split into Q, K, V and reshape to [heads, n_windows, window_size, dim_head]
    // From [T, 3*C] -> chunk into 3x [T, C]
    // Then rearrange: [T, C] = [(w*n), (h*d)] -> [h, w, n, d]
    // where w=n_windows, n=window_size, h=heads, d=dim_head

    // Allocate Q, K, V as [heads, n_windows, window_size, dim_head]
    int64_t attn_size = (int64_t)heads * n_windows * window_size * dim_head;
    std::vector<float> Q(attn_size), K(attn_size), V(attn_size);

    for (int64_t t = 0; t < T; t++) {
        int w = t / window_size;
        int n = t % window_size;
        for (int h = 0; h < heads; h++) {
            for (int d = 0; d < dim_head; d++) {
                int64_t src_c = h * dim_head + d;
                int64_t dst_idx = ((int64_t)h * n_windows + w) * window_size * dim_head + n * dim_head + d;
                Q[dst_idx] = qkv[t * 3 * C + src_c];
                K[dst_idx] = qkv[t * 3 * C + C + src_c];
                V[dst_idx] = qkv[t * 3 * C + 2 * C + src_c];
            }
        }
    }

    // Step 4: Apply Rotary Position Embeddings (RoPE) with scale
    // SinusoidalEmbeddings: inv_freq has dim_head/2 values
    // freqs[n, i] = n * inv_freq[i], then cat(freqs, freqs) -> [window_size, dim_head]
    // scale[n, i] = base_scale[i] ^ power[n], where power[n] = (n - window_size/2) / scale_base
    // base_scale[i] = (arange(0, dim_head, 2)[i] + 0.4*dim_head) / (1.4*dim_head)
    // scale_base = window_size / 2
    if (inv_freq != nullptr) {
        float scale_base = (float)(window_size / 2);

        // Precompute base_scale for xpos-like scaling
        std::vector<float> base_scale(half_head);
        for (int i = 0; i < half_head; i++) {
            base_scale[i] = ((float)(2 * i) + 0.4f * dim_head) / (1.4f * dim_head);
        }

        // Precompute cos/sin of freqs and scale for each position in window
        std::vector<float> cos_freqs(window_size * dim_head);
        std::vector<float> sin_freqs(window_size * dim_head);
        std::vector<float> pos_scale(window_size * dim_head);
        std::vector<float> pos_inv_scale(window_size * dim_head);

        for (int n = 0; n < window_size; n++) {
            float power = ((float)n - (float)(window_size / 2)) / scale_base;
            for (int i = 0; i < half_head; i++) {
                float freq = (float)n * inv_freq[i];
                float c = std::cos(freq);
                float s = std::sin(freq);
                float sc = std::pow(base_scale[i], power);
                // freqs is cat(freqs, freqs) so both halves are the same
                cos_freqs[n * dim_head + i] = c;
                cos_freqs[n * dim_head + half_head + i] = c;
                sin_freqs[n * dim_head + i] = s;
                sin_freqs[n * dim_head + half_head + i] = s;
                pos_scale[n * dim_head + i] = sc;
                pos_scale[n * dim_head + half_head + i] = sc;
                pos_inv_scale[n * dim_head + i] = 1.0f / sc;
                pos_inv_scale[n * dim_head + half_head + i] = 1.0f / sc;
            }
        }

        // Apply RoPE to Q and K for each window
        // rotate_half(x): for x = [x0..x_{d/2-1}, x_{d/2}..x_{d-1}]
        //   returns [-x_{d/2}...-x_{d-1}, x_0...x_{d/2-1}]
        // q = (q * cos * scale) + (rotate_half(q) * sin * scale)
        // k = (k * cos * inv_scale) + (rotate_half(k) * sin * inv_scale)
        for (int h = 0; h < heads; h++) {
            for (int w = 0; w < n_windows; w++) {
                for (int n = 0; n < window_size; n++) {
                    int64_t base = ((int64_t)h * n_windows + w) * window_size * dim_head + n * dim_head;
                    float * q = &Q[base];
                    float * k = &K[base];

                    // Temporary for rotate_half
                    std::vector<float> q_rot(dim_head), k_rot(dim_head);
                    // rotate_half: [-x_{d/2}...-x_{d-1}, x_0...x_{d/2-1}]
                    for (int i = 0; i < half_head; i++) {
                        q_rot[i] = -q[half_head + i];
                        q_rot[half_head + i] = q[i];
                        k_rot[i] = -k[half_head + i];
                        k_rot[half_head + i] = k[i];
                    }

                    for (int d = 0; d < dim_head; d++) {
                        float c = cos_freqs[n * dim_head + d];
                        float s = sin_freqs[n * dim_head + d];
                        float sc = pos_scale[n * dim_head + d];
                        float isc = pos_inv_scale[n * dim_head + d];
                        q[d] = (q[d] * c * sc) + (q_rot[d] * s * sc);
                        k[d] = (k[d] * c * isc) + (k_rot[d] * s * isc);
                    }
                }
            }
        }
    }

    // Step 5: Scaled dot-product attention per window
    // Q, K, V are [heads, n_windows, window_size, dim_head]
    // For each (h, w): attn = softmax(Q @ K^T / sqrt(dim_head)) @ V
    float scale = 1.0f / std::sqrt((float)dim_head);
    std::vector<float> attn_out(attn_size);

    for (int h = 0; h < heads; h++) {
        for (int w = 0; w < n_windows; w++) {
            int64_t block_base = ((int64_t)h * n_windows + w) * window_size * dim_head;
            const float * q_block = &Q[block_base];
            const float * k_block = &K[block_base];
            const float * v_block = &V[block_base];
            float * o_block = &attn_out[block_base];

            // Compute attention scores [window_size, window_size]
            std::vector<float> scores(window_size * window_size);
            for (int qi = 0; qi < window_size; qi++) {
                float max_score = -1e30f;
                for (int ki = 0; ki < window_size; ki++) {
                    float dot = 0.0f;
                    for (int d = 0; d < dim_head; d++) {
                        dot += q_block[qi * dim_head + d] * k_block[ki * dim_head + d];
                    }
                    dot *= scale;
                    scores[qi * window_size + ki] = dot;
                    max_score = std::max(max_score, dot);
                }
                // Softmax
                float sum_exp = 0.0f;
                for (int ki = 0; ki < window_size; ki++) {
                    scores[qi * window_size + ki] = std::exp(scores[qi * window_size + ki] - max_score);
                    sum_exp += scores[qi * window_size + ki];
                }
                for (int ki = 0; ki < window_size; ki++) {
                    scores[qi * window_size + ki] /= sum_exp;
                }
                // Weighted sum of V
                for (int d = 0; d < dim_head; d++) {
                    float val = 0.0f;
                    for (int ki = 0; ki < window_size; ki++) {
                        val += scores[qi * window_size + ki] * v_block[ki * dim_head + d];
                    }
                    o_block[qi * dim_head + d] = val;
                }
            }
        }
    }

    // Step 6: Rearrange back: [h, w, n, d] -> [(w*n), (h*d)] = [T, C]
    std::vector<float> attn_flat(T * C);
    for (int h = 0; h < heads; h++) {
        for (int w = 0; w < n_windows; w++) {
            for (int n = 0; n < window_size; n++) {
                int64_t t = (int64_t)w * window_size + n;
                int64_t src_idx = ((int64_t)h * n_windows + w) * window_size * dim_head + n * dim_head;
                for (int d = 0; d < dim_head; d++) {
                    attn_flat[t * C + h * dim_head + d] = attn_out[src_idx + d];
                }
            }
        }
    }

    // Step 7: Output projection: [T, C] @ [C, C] -> [T, C]
    // out_weight is [C, C] in PyTorch (row-major), weight[out, in]
    for (int64_t t = 0; t < T; t++) {
        for (int o = 0; o < C; o++) {
            float sum = 0.0f;
            for (int i = 0; i < C; i++) {
                sum += attn_flat[t * C + i] * out_weight[o * C + i];
            }
            // Add residual and write back
            data[t * C + o] = sum + residual[t * C + o];
        }
    }
}

// SNAC Residual Unit
struct snac_residual_unit {
    std::vector<float> in_alpha;
    std::vector<float> in_kernel;
    std::vector<float> in_bias;
    std::vector<float> out_alpha;
    std::vector<float> out_kernel;
    std::vector<float> out_bias;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    int channels = 0;
    int kernel_size = 0;

    std::vector<float> forward(const float * input, int64_t input_len) {
        // Snake + Conv + Snake + Conv + Residual
        std::vector<float> cur(input, input + input_len * channels);

        // Snake1D in
        snake_1d_inplace(cur.data(), in_alpha.data(), input_len, channels);

        // Conv1D in
        std::vector<float> temp(input_len * channels);
        if (groups > 1) {
            conv1d_dw(cur.data(), in_kernel.data(), in_bias.data(),
                      temp.data(), input_len, channels, channels,
                      kernel_size, 1, padding, dilation, groups);
        } else {
            conv1d(cur.data(), in_kernel.data(), in_bias.data(),
                   temp.data(), input_len, channels, channels,
                   kernel_size, 1, padding, dilation);
        }
        cur = std::move(temp);

        // Snake1D out
        snake_1d_inplace(cur.data(), out_alpha.data(), input_len, channels);

        // Conv1D out (kernel_size=1)
        std::vector<float> output(input_len * channels);
        conv1d(cur.data(), out_kernel.data(), out_bias.data(),
               output.data(), input_len, channels, channels,
               1, 1, 0, 1);

        // Residual connection
        for (size_t i = 0; i < output.size(); i++) {
            output[i] += input[i];
        }

        return output;
    }
};

// SNAC Decoder Layer
struct snac_decoder_layer {
    std::vector<float> in_alpha;
    std::vector<float> in_kernel;
    std::vector<float> in_bias;
    std::vector<float> noise_kernel;
    std::vector<snac_residual_unit> residual_units;

    int stride = 2;
    int padding = 0;
    int groups = 1;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    bool use_noise = false;

    int layer_idx = -1;  // For debug output

    std::vector<float> forward(const float * input, int64_t input_len) {
        int64_t out_padding = stride % 2;  // output_padding per PyTorch ConvTranspose1d
        int64_t output_len = (input_len - 1) * stride + kernel_size - 2 * padding + out_padding;

        if (layer_idx == 0) {
            LOG_WRN("DEBUG forward layer 0: input_len=%lld, stride=%d, kernel=%d, padding=%d, out_padding=%lld, output_len=%lld\n",
                    (long long)input_len, stride, kernel_size, padding, (long long)out_padding, (long long)output_len);
        }

        std::vector<float> cur;

        auto debug_mean = [](const std::vector<float>& v, const char* name, int layer) {
            if (layer != 3) return;  // Only debug layer 3
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (auto& val : v) { mn = std::min(mn, val); mx = std::max(mx, val); sum += val; }
            LOG_WRN("  Layer %d %s: min=%.6f, max=%.6f, mean=%.6f\n", layer, name, mn, mx, sum / v.size());
        };

        // Snake1D
        cur.resize(input_len * in_channels);
        std::copy(input, input + input_len * in_channels, cur.begin());

        if (layer_idx == 2) {
            float diff_sum = 0;
            for (int i = 1; i < 100 && i < input_len; i++) {
                diff_sum += std::abs(cur[i * in_channels] - cur[(i-1) * in_channels]);
            }
            LOG_WRN("  Layer 2 BEFORE snake: diff_avg=%.6f\n", diff_sum / 99);
        }

        snake_1d_inplace(cur.data(), in_alpha.data(), input_len, in_channels);
        debug_mean(cur, "after snake", layer_idx);

        if (layer_idx == 2) {
            float diff_sum = 0;
            for (int i = 1; i < 100 && i < input_len; i++) {
                diff_sum += std::abs(cur[i * in_channels] - cur[(i-1) * in_channels]);
            }
            LOG_WRN("  Layer 2 AFTER snake: diff_avg=%.6f\n", diff_sum / 99);
        }

        // ConvTranspose1D
        std::vector<float> temp(output_len * out_channels);

        if (layer_idx == 0) {
            LOG_WRN("  Layer 0 ConvTranspose1D: input_len=%lld, in_ch=%d, out_ch=%d, k=%d, s=%d, p=%d, op=%d\n",
                    (long long)input_len, in_channels, out_channels, kernel_size, stride, padding, out_padding);
            LOG_WRN("    input sample: cur[0]=%.6f, cur[100]=%.6f, cur[1000]=%.6f\n",
                    cur[0], cur[100], cur[1000]);
            LOG_WRN("    kernel sample: in_kernel[0]=%.6f, in_kernel[100]=%.6f, in_kernel[1000]=%.6f\n",
                    in_kernel[0], in_kernel[100], in_kernel[1000]);
            LOG_WRN("    bias sample: in_bias[0]=%.6f, in_bias[100]=%.6f, in_bias[200]=%.6f\n",
                    in_bias[0], in_bias[100], in_bias[200]);
        }

        conv_transpose1d(cur.data(), in_kernel.data(), in_bias.data(),
                        temp.data(), input_len, in_channels, out_channels,
                        kernel_size, stride, padding, out_padding);

        if (layer_idx == 2) {
            float diff_sum = 0;
            for (int i = 1; i < 100 && i < output_len; i++) {
                diff_sum += std::abs(temp[i * out_channels] - temp[(i-1) * out_channels]);
            }
            LOG_WRN("  Layer 2 AFTER conv_t: diff_avg=%.6f (output_len=%lld, stride=%d)\n", diff_sum / 99, (long long)output_len, stride);
        }

        if (layer_idx == 0) {
            LOG_WRN("    output sample: temp[0]=%.6f, temp[100]=%.6f, temp[1000]=%.6f\n",
                    temp[0], temp[100], temp[1000]);
            // Check variation in first few output samples for first channel
            LOG_WRN("    Output variation (channel 0):");
            for (int i = 0; i < 10 && i < output_len; i++) {
                LOG_WRN("      temp[%d * 512 + 0] = %.6f\n", i, temp[i * out_channels]);
            }
        }

        cur = std::move(temp);
        debug_mean(cur, "after conv_t", layer_idx);

        // Noise injection: output = x + linear(x) * noise
        // noise is random [1, T] broadcast over channels
        if (use_noise && !noise_kernel.empty()) {
            std::vector<float> noise_out(output_len * out_channels, 0.0f);
            conv1d(cur.data(), noise_kernel.data(), nullptr,
                   noise_out.data(), output_len, out_channels, out_channels,
                   1, 1, 0, 1);
            // Generate random noise [1, output_len] and broadcast over channels
            static std::mt19937 rng(42);
            std::normal_distribution<float> dist(0.0f, 1.0f);
            for (int64_t t = 0; t < output_len; t++) {
                float noise_val = dist(rng);
                for (int64_t c = 0; c < out_channels; c++) {
                    cur[t * out_channels + c] += noise_out[t * out_channels + c] * noise_val;
                }
            }
        }
        debug_mean(cur, "after noise", layer_idx);

        // Residual units
        int unit_idx = 0;
        for (auto & unit : residual_units) {
            cur = unit.forward(cur.data(), output_len);
            if (layer_idx == 3) {
                char buf[64];
                snprintf(buf, sizeof(buf), "after residual_unit %d", unit_idx);
                debug_mean(cur, buf, layer_idx);
            }
            unit_idx++;
        }

        return cur;
    }
};

// SNAC Quantizer Layer (snac_24khz format)
struct snac_quantizer_layer {
    std::vector<float> codebook;       // [codebook_size, codebook_dim]
    std::vector<float> in_proj_bias;   // [codebook_dim]
    std::vector<float> in_proj_weight; // [codebook_dim, quantizer_dim, 1]
    std::vector<float> out_proj_bias;  // [quantizer_dim]
    std::vector<float> out_proj_weight;// [quantizer_dim, codebook_dim, 1]

    int codebook_size = SNAC_CODEBOOK_SIZE;
    int codebook_dim = SNAC_CODEBOOK_DIM;
    int quantizer_dim = SNAC_QUANTIZER_DIM;  // Output dimension (1024)

    // Lookup token in codebook and apply projections
    // TTS.cpp: cur = ggml_get_rows(codebook, tokens) -> [codebook_dim, seq_len]
    //          cur = ggml_transpose(cur) -> [seq_len, codebook_dim]
    //          cur = ggml_conv_1d(out_proj_kernel, cur) -> [seq_len, quantizer_dim]
    std::vector<float> forward(const int * tokens, int64_t seq_len) {
        std::vector<float> output(seq_len * quantizer_dim, 0.0f);

        // Codebook is stored as [codebook_dim, codebook_size] = [8, 4096] in row-major
        // After loading from GGUF (which stores column-major), we have:
        // codebook[d * codebook_size + t] = embedding value for dimension d, token t

        for (int64_t i = 0; i < seq_len; i++) {
            int token = tokens[i];
            if (token < 0 || token >= codebook_size) {
                token = 0;  // Clamp to valid range
            }

            // Codebook lookup: embedding for token is [codebook_dim]
            // Data is stored as [codebook_size, codebook_dim] = [4096, 8] in row-major
            // Token t's embedding is at: codebook[t * codebook_dim + d]
            float emb[8];  // codebook_dim = 8
            for (int d = 0; d < codebook_dim; d++) {
                emb[d] = codebook[token * codebook_dim + d];  // FIXED: was d * codebook_size + token
            }

            // Output projection (out_proj): 1x1 conv
            // TTS.cpp: ggml_conv_1d expects weight [out_ch, in_ch, 1] = [quantizer_dim, codebook_dim, 1]
            // GGUF shape [8, 768] = [codebook_dim, quantizer_dim] needs transpose to [768, 8]
            for (int d = 0; d < quantizer_dim; d++) {
                float sum = out_proj_bias[d];
                for (int c = 0; c < codebook_dim; c++) {
                    sum += emb[c] * out_proj_weight[d * codebook_dim + c];
                }
                output[i * quantizer_dim + d] = sum;
            }
        }

        return output;
    }
};

// SNAC Model (snac_24khz format)
struct snac_model {
    // Input convolutions
    std::vector<float> in_conv_kernel;
    std::vector<float> in_conv_bias;
    std::vector<float> up_conv_kernel;
    std::vector<float> up_conv_bias;

    // Attention layer (for long sequences)
    std::vector<float> attn_norm_weight;
    std::vector<float> attn_norm_bias;
    std::vector<float> attn_to_qkv_weight;
    std::vector<float> attn_rel_pos_inv_freq;
    std::vector<float> attn_to_out_weight;

    // Output layer
    std::vector<float> out_conv_kernel;
    std::vector<float> out_conv_bias;
    std::vector<float> snake_alpha_out;

    // Decoder layers
    std::vector<snac_decoder_layer> layers;

    // Quantizers (4 heads for snac_24khz)
    std::vector<snac_quantizer_layer> quantizers;

    int quantizer_dim = SNAC_QUANTIZER_DIM;  // 1024 - output dimension of quantizers
    int decoder_dim = SNAC_DECODER_DIM;       // 1536 - dimension after up_conv
    int codebook_dim = SNAC_CODEBOOK_DIM;
    int n_quantizers = SNAC_N_QUANTIZERS;
    int n_decoder_layers = SNAC_N_DECODER_LAYERS;
    int upsample_factor = SNAC_UPSAMPLE_FACTOR;
    int decoder_rates[4] = {8, 8, 4, 2};  // snac_24khz: [8, 8, 4, 2]
    int vq_strides[4] = {4, 2, 1, 1};     // TTS pyramid: head0*4, head1*2, head2*1

    bool loaded = false;

    // Decode audio tokens to PCM samples
    // Note: Orpheus TTS generates 3 codebook outputs, but snac_24khz has 4 quantizers
    // We use the first 3 quantizers for the TTS output (vq_strides 8,4,2)
    std::vector<float> decode(const std::vector<std::vector<int>> & pyramid_tokens) {
        if (!loaded) {
            LOG_WRN("%s: SNAC model not loaded, returning silence\n", __func__);
            return {};
        }

        // Get token counts for each head
        if (pyramid_tokens[2].empty()) {
            return {};
        }

        // Step 1: Quantizer forward pass for each head (use first 3 for TTS)
        // Quantizers output quantizer_dim (1024) channels
        // Head0 has N tokens, Head1 has 2N tokens, Head2 has 4N tokens
        std::vector<std::vector<float>> head_embeddings(3);
        for (int h = 0; h < 3 && h < (int)quantizers.size(); h++) {
            int64_t head_len = pyramid_tokens[h].size();
            if (head_len == 0) {
                continue;
            }

            head_embeddings[h] = quantizers[h].forward(pyramid_tokens[h].data(), head_len);
        }

        // Debug: check quantizer outputs
        for (int h = 0; h < 3; h++) {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < head_embeddings[h].size(); i++) {
                float v = head_embeddings[h][i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            LOG_WRN("DEBUG: head%d embedding: size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                    h, head_embeddings[h].size(), mn, mx, sum / head_embeddings[h].size());
        }

        // Step 2: Combine quantizer outputs SEQUENTIALLY with upsampling
        // This matches chatllm.cpp's ResidualVectorQuantize::dequantize implementation:
        //   output = z_q_0
        //   output = z_q_1 + repeat_interleave(output, vq_strides[0]/vq_strides[1])
        //   output = z_q_2 + repeat_interleave(output, vq_strides[1]/vq_strides[2])
        // For vq_strides [4, 2, 1]:
        //   head0: N tokens -> output size N
        //   head1: 2N tokens -> upsample previous by 2, add -> size 2N
        //   head2: 4N tokens -> upsample previous by 2, add -> size 4N
        std::vector<float> quantizer_output;
        int64_t quantizer_output_len = 0;

        for (int h = 0; h < 3; h++) {
            int64_t head_len = pyramid_tokens[h].size();
            if (head_len == 0) {
                continue;
            }

            const auto& z_q_i = head_embeddings[h];

            if (quantizer_output.empty()) {
                // First head: just use its output
                quantizer_output = z_q_i;
                quantizer_output_len = head_len;
            } else {
                // Subsequent heads: upsample previous output, then add current
                int upsample_factor = vq_strides[h-1] / vq_strides[h];  // e.g., 4/2=2, 2/1=2
                int64_t new_len = head_len;

                // Upsample previous output by repeat_interleave
                std::vector<float> upsampled(new_len * quantizer_dim, 0.0f);
                for (int64_t i = 0; i < quantizer_output_len; i++) {
                    for (int u = 0; u < upsample_factor; u++) {
                        int64_t dst_idx = i * upsample_factor + u;
                        if (dst_idx >= new_len) break;
                        for (int d = 0; d < quantizer_dim; d++) {
                            upsampled[dst_idx * quantizer_dim + d] = quantizer_output[i * quantizer_dim + d];
                        }
                    }
                }

                // Add current head's output to upsampled previous
                quantizer_output.resize(new_len * quantizer_dim);
                for (int64_t i = 0; i < new_len; i++) {
                    for (int d = 0; d < quantizer_dim; d++) {
                        quantizer_output[i * quantizer_dim + d] = z_q_i[i * quantizer_dim + d] + upsampled[i * quantizer_dim + d];
                    }
                }
                quantizer_output_len = new_len;
            }
        }

        int64_t seq_len = quantizer_output_len;
        std::vector<float> combined = std::move(quantizer_output);

        // Debug: check combined
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < combined.size(); i++) {
                float v = combined[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            LOG_WRN("DEBUG: combined: size=%zu, seq_len=%lld, min=%.6f, max=%.6f, mean=%.6f\n",
                    combined.size(), (long long)seq_len, mn, mx, sum / combined.size());
        }

        // Step 3: Input convolution (depthwise)
        std::vector<float> cur = combined;
        int64_t cur_len = seq_len;
        int cur_channels = quantizer_dim;

        // in_conv: depthwise conv_1d with kernel_size=7, padding=3, groups=quantizer_dim
        std::vector<float> temp(cur_len * cur_channels);
        conv1d_dw(cur.data(), in_conv_kernel.data(), in_conv_bias.data(),
                  temp.data(), cur_len, cur_channels, cur_channels,
                  7, 1, 3, 1, cur_channels);
        cur = std::move(temp);

        // Debug: after in_conv
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < cur.size(); i++) {
                float v = cur[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            LOG_WRN("DEBUG: after in_conv: size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                    cur.size(), mn, mx, sum / cur.size());
        }

        // up_conv: 1x1 conv to project (quantizer_dim=1024 -> decoder_dim=1536)
        std::vector<float> temp2(cur_len * decoder_dim);
        conv1d(cur.data(), up_conv_kernel.data(), up_conv_bias.data(),
               temp2.data(), cur_len, cur_channels, decoder_dim,
               1, 1, 0, 1);
        cur = std::move(temp2);
        cur_channels = decoder_dim;

        // Debug: after up_conv
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < cur.size(); i++) {
                float v = cur[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            // Check consecutive diff for first 100 samples (channel 0)
            float diff_sum = 0, diff_max = 0;
            for (int i = 1; i < 100 && i < cur_len; i++) {
                float diff = std::abs(cur[i * cur_channels] - cur[(i-1) * cur_channels]);
                diff_sum += diff;
                diff_max = std::max(diff_max, diff);
            }
            LOG_WRN("DEBUG: after up_conv: size=%zu, min=%.6f, max=%.6f, mean=%.6f, diff_avg=%.6f, diff_max=%.6f\n",
                    cur.size(), mn, mx, sum / cur.size(), diff_sum / 99, diff_max);
        }

        // Step 3.5: LocalMHA attention (between up_conv and decoder layers)
        // Python: layers += [LocalMHA(dim=channels, window_size=32)]
        if (!attn_norm_weight.empty() && !attn_to_qkv_weight.empty() && !attn_to_out_weight.empty()) {
            local_mha_forward(
                cur.data(), cur_len, cur_channels,
                attn_norm_weight.data(),
                attn_norm_bias.data(),
                attn_to_qkv_weight.data(),
                attn_to_out_weight.data(),
                attn_rel_pos_inv_freq.empty() ? nullptr : attn_rel_pos_inv_freq.data(),
                32,  // window_size
                64   // dim_head
            );

            // Debug: after attention
            {
                float mn = 1e30f, mx = -1e30f, sum = 0;
                for (size_t i = 0; i < cur.size(); i++) {
                    float v = cur[i];
                    mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
                }
                LOG_WRN("DEBUG: after attention: size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                        cur.size(), mn, mx, sum / cur.size());
            }
        }

        // Step 4: Decoder layers
        for (size_t l = 0; l < layers.size(); l++) {
            auto & layer = layers[l];
            cur = layer.forward(cur.data(), cur_len);
            int64_t out_padding = layer.stride % 2;
            cur_len = (cur_len - 1) * layer.stride + layer.kernel_size - 2 * layer.padding + out_padding;
            cur_channels = layer.out_channels;

            // Debug: after each decoder layer
            {
                float mn = 1e30f, mx = -1e30f, sum = 0;
                for (size_t i = 0; i < cur.size(); i++) {
                    float v = cur[i];
                    mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
                }
                // Check consecutive diff for first 100 samples (channel 0)
                float diff_sum = 0, diff_max = 0;
                for (int i = 1; i < 100 && i < cur_len; i++) {
                    float diff = std::abs(cur[i * cur_channels] - cur[(i-1) * cur_channels]);
                    diff_sum += diff;
                    diff_max = std::max(diff_max, diff);
                }
                LOG_WRN("DEBUG: after decoder layer %zu: size=%zu, cur_len=%lld, channels=%d, min=%.6f, max=%.6f, mean=%.6f, diff_avg=%.6f, diff_max=%.6f\n",
                        l, cur.size(), (long long)cur_len, cur_channels, mn, mx, sum / cur.size(), diff_sum / 99, diff_max);
            }
        }

        // Step 5: Output layer
        // Debug: before snake
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < cur.size(); i++) {
                float v = cur[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            LOG_WRN("DEBUG: BEFORE snake_alpha_out: size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                    cur.size(), mn, mx, sum / cur.size());
        }

        snake_1d_inplace(cur.data(), snake_alpha_out.data(), cur_len, cur_channels);

        // Debug: after snake, before out_conv
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < cur.size(); i++) {
                float v = cur[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            LOG_WRN("DEBUG: AFTER snake_alpha_out (before out_conv): size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                    cur.size(), mn, mx, sum / cur.size());
        }

        // Debug: check snake_alpha_out and final weight
        {
            float alpha_mn = 1e30f, alpha_mx = -1e30f;
            for (auto v : snake_alpha_out) { alpha_mn = std::min(alpha_mn, v); alpha_mx = std::max(alpha_mx, v); }
            float w_mn = 1e30f, w_mx = -1e30f, w_sum = 0;
            int w_pos = 0, w_neg = 0;
            for (auto v : out_conv_kernel) { w_mn = std::min(w_mn, v); w_mx = std::max(w_mx, v); w_sum += v; if (v > 0) w_pos++; else w_neg++; }
            LOG_WRN("DEBUG: snake_alpha_out: size=%zu, min=%.6f, max=%.6f\n", snake_alpha_out.size(), alpha_mn, alpha_mx);
            LOG_WRN("DEBUG: out_conv_kernel: size=%zu, min=%.6f, max=%.6f, mean=%.6f, pos=%d, neg=%d\n",
                    out_conv_kernel.size(), w_mn, w_mx, w_sum / out_conv_kernel.size(), w_pos, w_neg);
        }

        std::vector<float> output(cur_len * 1);
        const float * bias_ptr = out_conv_bias.empty() ? nullptr : out_conv_bias.data();

        // Debug: manually compute first few output values to verify
        LOG_WRN("DEBUG: Manual out_conv check:\n");
        {
            // Check input statistics at t=0 with sign info
            float input_sum = 0, input_abs_sum = 0;
            int input_pos_count = 0, input_neg_count = 0;
            for (int ic = 0; ic < cur_channels; ic++) {
                float v = cur[0 * cur_channels + ic];
                input_sum += v;
                input_abs_sum += std::abs(v);
                if (v > 0) input_pos_count++;
                else if (v < 0) input_neg_count++;
            }
            LOG_WRN("  Input at t=0: sum=%.6f, abs_sum=%.6f, mean=%.6f, pos=%d, neg=%d\n",
                    input_sum, input_abs_sum, input_sum / cur_channels, input_pos_count, input_neg_count);

            // Print some actual input values
            LOG_WRN("  Input values at t=0: ");
            for (int ic = 0; ic < 10 && ic < cur_channels; ic++) {
                printf("%.4f ", cur[ic]);
            }
            printf("...\n");

            // Check kernel statistics per k position
            // Kernel is now stored in PyTorch format: kernel[ic*7 + k]
            for (int k = 0; k < 7; k++) {
                float k_sum = 0, k_abs_sum = 0;
                for (int ic = 0; ic < cur_channels; ic++) {
                    float w = out_conv_kernel[ic * 7 + k];  // PyTorch format [ic, k]
                    k_sum += w;
                    k_abs_sum += std::abs(w);
                }
                LOG_WRN("  Kernel k=%d: sum=%.6f, abs_sum=%.6f, mean=%.6f\n",
                        k, k_sum, k_abs_sum, k_sum / cur_channels);
            }

            // Check kernel statistics
            float kernel_sum = 0, kernel_abs_sum = 0;
            for (int i = 0; i < (int)out_conv_kernel.size(); i++) {
                kernel_sum += out_conv_kernel[i];
                kernel_abs_sum += std::abs(out_conv_kernel[i]);
            }
            LOG_WRN("  Kernel total: sum=%.6f, abs_sum=%.6f, mean=%.6f, size=%zu\n",
                    kernel_sum, kernel_abs_sum, kernel_sum / out_conv_kernel.size(), out_conv_kernel.size());
        }
        // Manual verification using same kernel layout as conv1d
        for (int t = 0; t < 3 && t < (int)cur_len; t++) {
            float sum = bias_ptr ? bias_ptr[0] : 0.0f;
            float pos_contrib = 0, neg_contrib = 0;
            for (int k = 0; k < 7; k++) {
                int input_pos = t + k - 3;  // padding=3
                if (input_pos >= 0 && input_pos < cur_len) {
                    float k_pos = 0, k_neg = 0;
                    for (int ic = 0; ic < cur_channels; ic++) {
                        float w = out_conv_kernel[ic * 7 + k];  // PyTorch format [ic, k]
                        float inp = cur[input_pos * cur_channels + ic];
                        float contrib = inp * w;
                        sum += contrib;
                        if (contrib > 0) { pos_contrib += contrib; k_pos += contrib; }
                        else { neg_contrib += contrib; k_neg += contrib; }
                    }
                    LOG_WRN("    t=%d k=%d pos=%.6f neg=%.6f net=%.6f\n", t, k, k_pos, k_neg, k_pos+k_neg);
                }
            }
            LOG_WRN("  Manual output[%d] = %.6f (pos_contrib=%.6f, neg_contrib=%.6f)\n", t, sum, pos_contrib, neg_contrib);
        }

        conv1d(cur.data(), out_conv_kernel.data(), bias_ptr,
               output.data(), cur_len, cur_channels, 1,
               7, 1, 3, 1);

        // Debug: print first few actual outputs
        LOG_WRN("DEBUG: Actual out_conv outputs:\n");
        for (int t = 0; t < 3 && t < (int)cur_len; t++) {
            LOG_WRN("  output[%d] = %.6f\n", t, output[t]);
        }

        // Debug: pre-tanh output
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            int neg_count = 0;
            for (size_t i = 0; i < output.size(); i++) {
                float v = output[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
                if (v < 0) neg_count++;
            }
            LOG_WRN("DEBUG: pre-tanh output: size=%zu, min=%.6f, max=%.6f, mean=%.6f, neg_ratio=%.2f%%\n",
                    output.size(), mn, mx, sum / output.size(), 100.0 * neg_count / output.size());
        }

        // Tanh
        for (auto & v : output) {
            v = std::tanh(v);
        }

        // Debug: final output
        {
            float mn = 1e30f, mx = -1e30f, sum = 0;
            for (size_t i = 0; i < output.size(); i++) {
                float v = output[i];
                mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
            }
            LOG_WRN("DEBUG: final output: size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                    output.size(), mn, mx, sum / output.size());
        }

        return output;
    }
};

// Helper function to dequantize and copy tensor data to std::vector<float>
static bool copy_tensor_to_vector(struct ggml_tensor * tensor, std::vector<float> & dst) {
    if (!tensor || !tensor->data) {
        return false;
    }

    size_t n_elements = ggml_nelements(tensor);
    dst.resize(n_elements);

    // If tensor is already F32, just copy
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(dst.data(), tensor->data, n_elements * sizeof(float));
        return true;
    }

    // Otherwise, dequantize
    // For simplicity, we'll use ggml backend to dequantize
    // Create a temporary context for dequantization
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * 2 + n_elements * sizeof(float) + 1024,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        return false;
    }

    // Create F32 destination tensor
    struct ggml_tensor * dst_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_elements);
    if (!dst_tensor) {
        ggml_free(ctx);
        return false;
    }

    // Copy source tensor structure and data pointer
    struct ggml_tensor src_copy = *tensor;

    // Dequantize using ggml_cast
    struct ggml_tensor * result = ggml_cast(ctx, &src_copy, GGML_TYPE_F32);
    if (!result) {
        ggml_free(ctx);
        return false;
    }

    // Build and compute graph
    struct ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // Create compute plan and execute
    // Note: For CPU-only, we need to manually dequantize
    // This is a simplified approach - in production, use ggml backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        ggml_free(ctx);
        return false;
    }

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors_from_buft(ctx, ggml_backend_get_default_buffer_type(backend));
    if (!buffer) {
        ggml_backend_free(backend);
        ggml_free(ctx);
        return false;
    }

    // Copy data to backend buffer
    ggml_backend_tensor_set(&src_copy, tensor->data, 0, ggml_nbytes(tensor));

    ggml_backend_graph_compute(backend, gf);

    // Get result
    ggml_backend_tensor_get(result, dst.data(), 0, n_elements * sizeof(float));

    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(ctx);

    return true;
}

// Helper to parse layer index from tensor name
static std::pair<int, std::string> parse_layer_name(const std::string & name) {
    // Parse names like "layers.0.alpha" -> (0, ".alpha")
    std::regex pattern(R"((?:layers|quantizers)\.(\d+)(.*))");
    std::smatch match;
    if (std::regex_match(name, match, pattern)) {
        return {std::stoi(match[1].str()), match[2].str()};
    }
    return {-1, name};
}

// Helper to parse residual unit index from tensor name
static std::pair<int, std::string> parse_residual_unit_name(const std::string & name) {
    // Parse names like ".0.in_alpha" -> (0, ".in_alpha")
    std::regex pattern(R"(\.(\d+)(.*))");
    std::smatch match;
    if (std::regex_match(name, match, pattern)) {
        return {std::stoi(match[1].str()), match[2].str()};
    }
    return {-1, name};
}

// Load SNAC model from a separate PyTorch file (for development/testing)
// Returns false if file not found or loading failed
static bool load_snac_model_from_pytorch(snac_model & model, const char * snac_path) {
    // Try to load from PyTorch file using Python
    // For now, just check if file exists and log
    std::ifstream f(snac_path);
    if (!f.good()) {
        return false;
    }
    f.close();

    LOG_INF("%s: Found SNAC model at %s\n", __func__, snac_path);
    LOG_WRN("%s: Direct PyTorch loading not implemented, using Python to convert\n", __func__);
    return false;
}

// Helper to detect GGUF format
static bool is_ttscpp_format(struct gguf_context * gguf_ctx, size_t n_tensors) {
    for (size_t i = 0; i < n_tensors; i++) {
        const char * tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        if (tensor_name && strncmp(tensor_name, "snac.", 5) == 0) {
            return true;
        }
    }
    return false;
}

// Load TTS.cpp format SNAC tensors
static int load_ttscpp_tensors(snac_model & model, struct gguf_context * gguf_ctx, struct ggml_context * ggml_ctx, size_t n_tensors) {
    int tensors_loaded = 0;

    // TTS.cpp dimensions: 768 (in_conv) -> 1024 (up_conv) -> 512 -> 256 -> 128 -> 64
    model.quantizer_dim = 768;
    model.decoder_dim = 1024;

    // NOTE: Do NOT read decoder_rates from GGUF metadata - it has wrong values!
    // The GGUF has [8, 5, 4, 2] but the correct values are [8, 8, 4, 2]
    // We use the hardcoded values set in load_snac_model() instead
    // The kernel sizes confirm correct rates: [16, 16, 8, 4] = [2*8, 2*8, 2*4, 2*2]

    // VQ strides for SNAC pyramid structure - these are NOT the same as decoder_rates!
    // Pyramid structure: head0 (1 token/frame) -> repeat 4x
    //                    head1 (2 tokens/frame) -> repeat 2x
    //                    head2 (4 tokens/frame) -> no repeat (stride 1)
    // This matches TTS.cpp's repeats array: [4, 2, 1]
    model.vq_strides[0] = 4;  // head0: 1 token per frame, repeat 4x
    model.vq_strides[1] = 2;  // head1: 2 tokens per frame, repeat 2x
    model.vq_strides[2] = 1;  // head2: 4 tokens per frame, no repeat
    model.vq_strides[3] = 1;  // unused for TTS

    LOG_INF("%s: TTS.cpp format - decoder_rates: %d, %d, %d, %d\n", __func__,
            model.decoder_rates[0], model.decoder_rates[1], model.decoder_rates[2], model.decoder_rates[3]);

    int layer_dims[4] = {1024, 512, 256, 128};
    int out_dims[4] = {512, 256, 128, 64};

    for (size_t i = 0; i < n_tensors; i++) {
        const char * tensor_name = gguf_get_tensor_name(gguf_ctx, i);
        struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, tensor_name);
        if (!tensor) continue;

        std::string name(tensor_name);

        // Input convolution (depthwise conv1d)
        // GGUF stores as [7, 768] column-major, which reads as (768, 7) row-major
        // For depthwise conv: need [768, 1, 7] but stored as [768, 7]
        if (name == "snac.in.bias") {
            if (copy_tensor_to_vector(tensor, model.in_conv_bias)) tensors_loaded++;
        }
        else if (name == "snac.in.weight") {
            // GGUF shape [7, 768] = [kernel_size, channels] column-major
            // Element (k, c) is at: k + 7*c
            // For depthwise conv1d_dw: expected index = c * kernel_size + k = c*7 + k
            // These are the SAME (k + 7*c = c*7 + k), so NO transpose needed!
            if (copy_tensor_to_vector(tensor, model.in_conv_kernel)) {
                tensors_loaded++;
            }
        }
        // Up convolution (1x1 conv)
        // GGUF stores as [768, 1024] column-major, reads as (1024, 768)
        // For 1x1 conv: need [1024, 768, 1] which is just [1024, 768]
        else if (name == "snac.up.bias") {
            if (copy_tensor_to_vector(tensor, model.up_conv_bias)) tensors_loaded++;
        }
        else if (name == "snac.up.weight") {
            // GGUF shape [768, 1024] column-major (ne0=768, ne1=1024)
            // Element (in, out) is at: in + 768*out
            // We need row-major [1024, 768] for 1x1 conv: element (out, in) at out*768 + in
            // Transpose needed!
            std::vector<float> raw;
            if (copy_tensor_to_vector(tensor, raw)) {
                int64_t ne0 = tensor->ne[0];  // 768 = in_channels
                int64_t ne1 = tensor->ne[1];  // 1024 = out_channels
                model.up_conv_kernel.resize(ne0 * ne1);
                for (int64_t out = 0; out < ne1; out++) {
                    for (int64_t in = 0; in < ne0; in++) {
                        // Source: column-major (in, out) at in + 768*out
                        // Dest: row-major (out, in) at out*768 + in
                        model.up_conv_kernel[out * ne0 + in] = raw[in + ne0 * out];
                    }
                }
                tensors_loaded++;
            }
        }
        // Output layer
        else if (name == "snac.alpha_out") {
            if (copy_tensor_to_vector(tensor, model.snake_alpha_out)) tensors_loaded++;
        }
        else if (name == "snac.final.bias") {
            // Handle empty bias tensor (shape=[])
            if (ggml_nelements(tensor) > 0) {
                if (copy_tensor_to_vector(tensor, model.out_conv_bias)) tensors_loaded++;
            } else {
                // Set empty bias (will be treated as zero)
                model.out_conv_bias.clear();
                tensors_loaded++;
                LOG_WRN("DEBUG: final.bias is empty (no bias)\n");
            }
        }
        else if (name == "snac.final.weight") {
            // GGUF shape=[64, 7] column-major (ne0=64, ne1=7)
            // For column-major: element (ic, k) is at: ic + 64*k
            // Need PyTorch Conv1d weight: [out=1, in=64, k=7]
            // For out=1: flat index = ic * 7 + k
            std::vector<float> raw;
            if (copy_tensor_to_vector(tensor, raw)) {
                int64_t ne0 = tensor->ne[0];  // 64 = in_channels
                int64_t ne1 = tensor->ne[1];  // 7 = kernel_size
                LOG_WRN("DEBUG: final.weight GGUF shape [%lld, %lld]\n", (long long)ne0, (long long)ne1);
                int ks = ne1;  // 7
                int ch = ne0;  // 64
                model.out_conv_kernel.resize(ch * ks);
                // raw is column-major [64, 7]: element (ic, k) at ic + 64*k
                // need row-major [64, 7]: element (ic, k) at ic*7 + k
                for (int k = 0; k < ks; k++) {
                    for (int ic = 0; ic < ch; ic++) {
                        // src: column-major (ic, k) = ic + 64*k
                        // dst: row-major (ic, k) = ic*7 + k
                        model.out_conv_kernel[ic * ks + k] = raw[ic + ne0 * k];
                    }
                }
                tensors_loaded++;
                // Debug: print weight stats
                float mn = 1e30f, mx = -1e30f, sum = 0;
                for (auto v : model.out_conv_kernel) {
                    mn = std::min(mn, v); mx = std::max(mx, v); sum += v;
                }
                LOG_WRN("DEBUG: final.weight loaded (transposed): size=%zu, min=%.6f, max=%.6f, mean=%.6f\n",
                        model.out_conv_kernel.size(), mn, mx, sum / model.out_conv_kernel.size());
            }
        }
        // Decoder layers (snac.layers.X.*)
        else if (name.find("snac.layers.") == 0) {
            std::string rest = name.substr(12);
            auto dot_pos = rest.find('.');
            if (dot_pos != std::string::npos) {
                int layer_idx = std::stoi(rest.substr(0, dot_pos));
                std::string tensor_rest = rest.substr(dot_pos);

                if (layer_idx >= 0 && layer_idx < (int)model.layers.size()) {
                    auto & layer = model.layers[layer_idx];
                    // Derive stride and kernel from tensor shape, not metadata
                    // Metadata decoder_rates might be wrong
                    layer.in_channels = layer_dims[layer_idx];
                    layer.out_channels = out_dims[layer_idx];
                    // ConvTranspose1D uses groups=1 (only ResidualUnit uses groups)
                    layer.groups = 1;
                    layer.use_noise = true;

                    if (tensor_rest == ".alpha") {
                        if (copy_tensor_to_vector(tensor, layer.in_alpha)) tensors_loaded++;
                    }
                    else if (tensor_rest == ".bias") {
                        if (copy_tensor_to_vector(tensor, layer.in_bias)) tensors_loaded++;
                    }
                    else if (tensor_rest == ".weight") {
                        // Get kernel size from tensor shape
                        // GGUF shape [out, k, in] → we need [in, out, k]
                        int64_t ne0 = tensor->ne[0];  // out_channels (fastest dim)
                        int64_t ne1 = tensor->ne[1];  // kernel_size
                        int64_t ne2 = tensor->ne[2];  // in_channels (slowest dim)

                        layer.kernel_size = ne1;
                        // Use metadata stride directly, not derived from kernel
                        // For stride=5, kernel=16 (not 10), so kernel/2 formula fails
                        layer.stride = model.decoder_rates[layer_idx];
                        layer.padding = (layer.stride + 1) / 2;
                        layer.in_channels = ne2;
                        layer.out_channels = ne0;

                        LOG_WRN("DEBUG: layer %d weight: GGUF ne=[%lld, %lld, %lld], kernel=%d, stride=%d, in=%d, out=%d\n",
                                layer_idx, (long long)ne0, (long long)ne1, (long long)ne2,
                                layer.kernel_size, layer.stride, layer.in_channels, layer.out_channels);

                        // ConvTranspose1d weight permutation
                        // GGUF reports shape [out, k, in] but copy_tensor_to_vector returns
                        // data already converted to row-major [in, k, out] format
                        // We need [in, out, k] for our ConvTranspose1D implementation
                        std::vector<float> raw;
                        if (copy_tensor_to_vector(tensor, raw)) {
                            int in_c = layer.in_channels;
                            int out_c = layer.out_channels;
                            int ks = layer.kernel_size;
                            layer.in_kernel.resize(in_c * out_c * ks);
                            // Source: row-major [in, k, out] from copy_tensor_to_vector
                            // Element (ic, k, oc) at: ic * ks * out_c + k * out_c + oc
                            // Dest: row-major [in, out, k] for ConvTranspose1D
                            // Element (ic, oc, k) at: ic * out_c * ks + oc * ks + k
                            for (int ic = 0; ic < in_c; ic++) {
                                for (int oc = 0; oc < out_c; oc++) {
                                    for (int k = 0; k < ks; k++) {
                                        // Source: row-major [in, k, out]
                                        int src_idx = ic * ks * out_c + k * out_c + oc;
                                        // Dest: row-major [in, out, k]
                                        int dst_idx = ic * out_c * ks + oc * ks + k;
                                        layer.in_kernel[dst_idx] = raw[src_idx];
                                    }
                                }
                            }
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".noise_weight" || tensor_rest == ".noise_weight_v") {
                        // 1x1 conv weight - may need transpose
                        std::vector<float> raw;
                        if (copy_tensor_to_vector(tensor, raw)) {
                            int ch = layer.out_channels;
                            layer.noise_kernel.resize(ch * ch);
                            // GGUF column-major [ch, ch] → need transpose for row-major
                            if (raw.size() == (size_t)ch * ch) {
                                for (int r = 0; r < ch; r++) {
                                    for (int c = 0; c < ch; c++) {
                                        // src: column-major (r, c) at r + ch*c
                                        // dst: row-major (r, c) at r*ch + c
                                        layer.noise_kernel[r * ch + c] = raw[r + ch * c];
                                    }
                                }
                            }
                            tensors_loaded++;
                        }
                    }
                    // Residual units (snac.layers.X.Y.*)
                    else if (tensor_rest.size() > 1 && tensor_rest[0] == '.') {
                        std::string unit_rest = tensor_rest.substr(1);
                        auto unit_dot_pos = unit_rest.find('.');
                        if (unit_dot_pos != std::string::npos) {
                            int unit_idx = std::stoi(unit_rest.substr(0, unit_dot_pos));
                            std::string unit_tensor = unit_rest.substr(unit_dot_pos);
                            if (unit_idx >= 0) {
                                if ((int)layer.residual_units.size() <= unit_idx) {
                                    layer.residual_units.resize(unit_idx + 1);
                                }
                                auto & unit = layer.residual_units[unit_idx];
                                unit.channels = layer.out_channels;
                                unit.kernel_size = 7;
                                unit.dilation = (int)std::pow(3, unit_idx);
                                unit.padding = ((unit.kernel_size - 1) * unit.dilation) / 2;
                                unit.groups = unit.channels;

                                if (unit_tensor == ".in_alpha") {
                                    if (copy_tensor_to_vector(tensor, unit.in_alpha)) tensors_loaded++;
                                }
                                else if (unit_tensor == ".in_bias") {
                                    // TTS.cpp transposes biases - they're stored as [1, ch] in GGUF
                                    std::vector<float> raw;
                                    if (copy_tensor_to_vector(tensor, raw)) {
                                        unit.in_bias = std::move(raw);
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".in_weight" || unit_tensor == ".in_weight_v") {
                                    // GGUF shape [7, ch] in column-major
                                    // For depthwise conv, we need [ch, kernel] layout
                                    // In column-major [7, ch], element (k, c) is at: k + 7*c
                                    // For row-major [ch, 7], element (c, k) is at: c*7 + k
                                    // These give the SAME flat index! So no transpose needed.
                                    if (copy_tensor_to_vector(tensor, unit.in_kernel)) tensors_loaded++;
                                }
                                else if (unit_tensor == ".out_alpha") {
                                    if (copy_tensor_to_vector(tensor, unit.out_alpha)) tensors_loaded++;
                                }
                                else if (unit_tensor == ".out_bias") {
                                    // TTS.cpp transposes biases
                                    std::vector<float> raw;
                                    if (copy_tensor_to_vector(tensor, raw)) {
                                        unit.out_bias = std::move(raw);
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".out_weight" || unit_tensor == ".out_weight_v") {
                                    // 1x1 conv weight: GGUF numpy shape [ch, ch] in row-major
                                    // No transpose needed
                                    if (copy_tensor_to_vector(tensor, unit.out_kernel)) {
                                        tensors_loaded++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Quantizers (snac.quantizers.X.*)
        else if (name.find("snac.quantizers.") == 0) {
            std::string rest = name.substr(16);
            auto dot_pos = rest.find('.');
            if (dot_pos != std::string::npos) {
                int quant_idx = std::stoi(rest.substr(0, dot_pos));
                std::string tensor_rest = rest.substr(dot_pos);

                if (quant_idx >= 0 && quant_idx < (int)model.quantizers.size()) {
                    auto & quant = model.quantizers[quant_idx];
                    quant.codebook_size = SNAC_CODEBOOK_SIZE;
                    quant.codebook_dim = model.codebook_dim;
                    quant.quantizer_dim = model.quantizer_dim;

                    if (tensor_rest == ".codebook.weight") {
                        // Original model codebook shape: [codebook_size, codebook_dim] = [4096, 8]
                        // Each row is an 8-dimensional embedding for a codebook entry
                        // For token t, embedding is codebook[t, :] = raw[t * 8 + d]
                        //
                        // IMPORTANT: GGUF reports reversed dimensions but data is in original layout!
                        // GGUF reports ne[0]=8, ne[1]=4096, but actual data is (4096, 8) = [codebook_size, codebook_dim]
                        // So the data is already in the correct layout [4096, 8] - NO transpose needed!
                        std::vector<float> raw;
                        if (copy_tensor_to_vector(tensor, raw)) {
                            // The raw data is already in [codebook_size, codebook_dim] = [4096, 8] layout
                            // For token t, embedding[t, d] = raw[t * 8 + d]
                            quant.codebook = raw;
                            quant.codebook_dim = 8;  // Always 8 for snac_24khz
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".in_proj.bias") {
                        if (copy_tensor_to_vector(tensor, quant.in_proj_bias)) tensors_loaded++;
                    }
                    else if (tensor_rest == ".in_proj.weight") {
                        if (copy_tensor_to_vector(tensor, quant.in_proj_weight)) tensors_loaded++;
                    }
                    else if (tensor_rest == ".out_proj.bias") {
                        if (copy_tensor_to_vector(tensor, quant.out_proj_bias)) tensors_loaded++;
                    }
                    else if (tensor_rest == ".out_proj.weight") {
                        // GGUF shape [8, 768] = [codebook_dim, quantizer_dim] in column-major
                        // We need [quantizer_dim, codebook_dim] = [768, 8] in row-major
                        // Column-major [8, 768]: element (d, q) at index d + 8*q
                        // Row-major [768, 8]: element (q, d) at index q*8 + d
                        // These are THE SAME! No transpose needed.
                        if (copy_tensor_to_vector(tensor, quant.out_proj_weight)) {
                            tensors_loaded++;
                        }
                    }
                }
            }
        }
    }
    return tensors_loaded;
}

// Load SNAC model from GGUF file (supports both snac_24khz and TTS.cpp formats)
static bool load_snac_model(snac_model & model, const char * model_path) {
    LOG_INF("%s: Loading SNAC model from %s\n", __func__, model_path);

    // Initialize GGUF context
    struct gguf_init_params gguf_params = {
        /*.no_alloc = */ false,
        /*.ctx      = */ NULL,
    };

    struct gguf_context * gguf_ctx = gguf_init_from_file(model_path, gguf_params);
    if (!gguf_ctx) {
        LOG_ERR("%s: Failed to load GGUF file: %s\n", __func__, model_path);
        return false;
    }

    // Create ggml context for loading tensors
    size_t n_tensors = gguf_get_n_tensors(gguf_ctx);

    // Re-init with context to load tensor data
    gguf_free(gguf_ctx);

    struct ggml_context * ggml_ctx = NULL;
    gguf_params.ctx = &ggml_ctx;
    gguf_ctx = gguf_init_from_file(model_path, gguf_params);

    if (!gguf_ctx || !ggml_ctx) {
        LOG_ERR("%s: Failed to initialize GGUF/GGML context\n", __func__);
        return false;
    }

    // Read SNAC metadata (snac_24khz format)
    int key_id = gguf_find_key(gguf_ctx, "snac.n_quantizers");
    if (key_id != -1) {
        model.n_quantizers = gguf_get_val_u32(gguf_ctx, key_id);
    }

    key_id = gguf_find_key(gguf_ctx, "snac.decoder_dim");
    if (key_id != -1) {
        model.decoder_dim = gguf_get_val_u32(gguf_ctx, key_id);
    }

    key_id = gguf_find_key(gguf_ctx, "snac.codebook_dim");
    if (key_id != -1) {
        model.codebook_dim = gguf_get_val_u32(gguf_ctx, key_id);
    }

    // Read decoder rates
    for (int i = 0; i < 4; i++) {
        std::string rate_key = "snac.decoder_rate_" + std::to_string(i);
        key_id = gguf_find_key(gguf_ctx, rate_key.c_str());
        if (key_id != -1) {
            model.decoder_rates[i] = gguf_get_val_u32(gguf_ctx, key_id);
        }
    }

    // Read VQ strides
    for (int i = 0; i < 4; i++) {
        std::string stride_key = "snac.vq_stride_" + std::to_string(i);
        key_id = gguf_find_key(gguf_ctx, stride_key.c_str());
        if (key_id != -1) {
            model.vq_strides[i] = gguf_get_val_u32(gguf_ctx, key_id);
        }
    }

    LOG_INF("%s: SNAC config: n_quantizers=%d, decoder_dim=%d, codebook_dim=%d\n",
            __func__, model.n_quantizers, model.decoder_dim, model.codebook_dim);

    // Always use correct decoder_rates for snac_24khz
    // The GGUF metadata may have wrong values
    // Correct rates: [8, 8, 4, 2] for snac_24khz from hubertsiuzdak/snac_24khz
    model.decoder_rates[0] = 8;
    model.decoder_rates[1] = 8;
    model.decoder_rates[2] = 4;
    model.decoder_rates[3] = 2;
    LOG_INF("%s: decoder_rates set to: %d, %d, %d, %d (for snac_24khz)\n", __func__,
            model.decoder_rates[0], model.decoder_rates[1], model.decoder_rates[2], model.decoder_rates[3]);

    // Always use correct vq_strides for TTS pyramid structure
    // The GGUF metadata may be corrupted or have wrong values
    // Pyramid structure: head0 (1 token/frame) -> repeat 4x
    //                    head1 (2 tokens/frame) -> repeat 2x
    //                    head2 (4 tokens/frame) -> no repeat (stride 1)
    model.vq_strides[0] = 4;  // head0: 1 token per frame, repeat 4x
    model.vq_strides[1] = 2;  // head1: 2 tokens per frame, repeat 2x
    model.vq_strides[2] = 1;  // head2: 4 tokens per frame, no repeat
    model.vq_strides[3] = 1;  // unused
    LOG_INF("%s: vq_strides set to: %d, %d, %d, %d (for TTS pyramid)\n", __func__,
            model.vq_strides[0], model.vq_strides[1], model.vq_strides[2], model.vq_strides[3]);

    // Initialize model structures
    model.quantizers.resize(model.n_quantizers);
    model.layers.resize(model.n_decoder_layers);

    // Track which tensors we've loaded
    int tensors_loaded = 0;

    // Detect format and load tensors
    bool use_ttscpp = is_ttscpp_format(gguf_ctx, n_tensors);
    if (use_ttscpp) {
        LOG_INF("%s: Detected TTS.cpp GGUF format (snac. prefix)\n", __func__);
        tensors_loaded = load_ttscpp_tensors(model, gguf_ctx, ggml_ctx, n_tensors);
    } else {
        LOG_INF("%s: Detected snac_24khz GGUF format (decoder. prefix)\n", __func__);
        // Iterate through all tensors and load SNAC ones
        for (size_t i = 0; i < n_tensors; i++) {
            const char * tensor_name = gguf_get_tensor_name(gguf_ctx, i);
            struct ggml_tensor * tensor = ggml_get_tensor(ggml_ctx, tensor_name);

            if (!tensor) {
                continue;
            }

            std::string name(tensor_name);

            // Parse tensor name and assign to model structure
            // New naming convention from convert_hf_to_gguf_snac.py

            // Input convolution
            if (name == "decoder.in_conv.bias") {
                if (copy_tensor_to_vector(tensor, model.in_conv_bias)) {
                    tensors_loaded++;
                }
            }
            else if (name == "decoder.in_conv.weight") {
                // GGUF shape [7, 768] = [kernel_size, channels] in column-major
                // For depthwise conv, code expects kernel[ch * kernel_size + k]
                // In column-major [7, 768], element (k, ch) is at k + ch * 7
                // This is the same as kernel[ch * 7 + k], so no transpose needed
                if (copy_tensor_to_vector(tensor, model.in_conv_kernel)) {
                    tensors_loaded++;
                }
            }
            // Up convolution
            else if (name == "decoder.up_conv.bias") {
                if (copy_tensor_to_vector(tensor, model.up_conv_bias)) {
                    tensors_loaded++;
                }
            }
            else if (name == "decoder.up_conv.weight") {
                // GGUF shape can be 2D [768, 1024] or 3D [768, 1, 1024]
                // PyTorch expects [out_ch, in_ch] = [1024, 768] in row-major (1x1 conv)
                std::vector<float> raw;
                if (copy_tensor_to_vector(tensor, raw)) {
                    int64_t ne0 = tensor->ne[0];  // 768 = in_channels
                    int n_dims = ggml_n_dims(tensor);

                    int in_ch, out_ch;
                    if (n_dims == 2) {
                        // 2D tensor [in_ch, out_ch]
                        int64_t ne1 = tensor->ne[1];  // 1024 = out_channels
                        in_ch = ne0;
                        out_ch = ne1;
                    } else {
                        // 3D tensor [in_ch, kernel, out_ch]
                        int64_t ne2 = tensor->ne[2];  // 1024 = out_channels
                        in_ch = ne0;
                        out_ch = ne2;
                    }

                    model.up_conv_kernel.resize(out_ch * in_ch);

                    // GGUF column-major [768, 1024]: element (ic, oc) at ic + ne0*oc
                    // PyTorch row-major [1024, 768]: element (oc, ic) at oc*in_ch + ic
                    for (int oc = 0; oc < out_ch; oc++) {
                        for (int ic = 0; ic < in_ch; ic++) {
                            // Source: column-major (ic, oc) at ic + ne0*oc
                            int src_idx = ic + ne0 * oc;
                            // Dest: row-major (oc, ic) at oc*in_ch + ic
                            int dst_idx = oc * in_ch + ic;
                            model.up_conv_kernel[dst_idx] = raw[src_idx];
                        }
                    }

                    LOG_INF("%s: up_conv.weight loaded: in_ch=%d, out_ch=%d, n_dims=%d\n",
                            __func__, in_ch, out_ch, n_dims);
                    tensors_loaded++;
                }
            }
            // Attention layer
        else if (name == "decoder.attn.norm.weight") {
            if (copy_tensor_to_vector(tensor, model.attn_norm_weight)) {
                tensors_loaded++;
            }
        }
        else if (name == "decoder.attn.norm.bias") {
            if (copy_tensor_to_vector(tensor, model.attn_norm_bias)) {
                tensors_loaded++;
            }
        }
        else if (name == "decoder.attn.to_qkv.weight") {
            if (copy_tensor_to_vector(tensor, model.attn_to_qkv_weight)) {
                tensors_loaded++;
            }
        }
        else if (name == "decoder.attn.rel_pos.inv_freq") {
            if (copy_tensor_to_vector(tensor, model.attn_rel_pos_inv_freq)) {
                tensors_loaded++;
            }
        }
        else if (name == "decoder.attn.to_out.weight") {
            if (copy_tensor_to_vector(tensor, model.attn_to_out_weight)) {
                tensors_loaded++;
            }
        }
        // Output layer
        else if (name == "decoder.alpha_out") {
            if (copy_tensor_to_vector(tensor, model.snake_alpha_out)) {
                tensors_loaded++;
            }
        }
        else if (name == "decoder.out_conv.bias") {
            if (copy_tensor_to_vector(tensor, model.out_conv_bias)) {
                tensors_loaded++;
            }
        }
        else if (name == "decoder.out_conv.weight") {
            // Converter saves as numpy [K, IC, OC] = [7, 64, 1]
            // GGUF stores dimensions in REVERSE order: ne[0]=OC, ne[1]=IC, ne[2]=K
            // So: ne[0]=1, ne[1]=64, ne[2]=7
            //
            // Our conv1d expects PyTorch format: kernel[oc, ic, k] = kernel[ic*kernel_size + k] for oc=0
            // Raw data layout: raw[k*IC*OC + ic*OC + oc] = raw[k*64 + ic] for OC=1
            std::vector<float> raw;
            if (copy_tensor_to_vector(tensor, raw)) {
                int64_t out_ch = tensor->ne[0];       // 1 (OC - reversed from numpy)
                int64_t in_ch = tensor->ne[1];        // 64 (IC - same)
                int64_t kernel_size = tensor->ne[2];  // 7 (K - reversed from numpy)

                LOG_INF("%s: out_conv.weight shape [kernel=%lld, in=%lld, out=%lld] (GGML format)\n",
                        __func__, (long long)kernel_size, (long long)in_ch, (long long)out_ch);

                // Load in GGML format: raw[k*64 + ic] -> kernel[ic*7 + k]
                // This is the same layout our conv1d expects
                model.out_conv_kernel.resize(out_ch * in_ch * kernel_size);
                for (int ic = 0; ic < in_ch; ic++) {
                    for (int k = 0; k < kernel_size; k++) {
                        // GGUF stores: raw[k*in_ch*oc + ic*oc + oc] = raw[k*64 + ic] for oc=0
                        // conv1d expects: kernel[ic*kernel_size + k] = kernel[ic*7 + k]
                        model.out_conv_kernel[ic * kernel_size + k] = raw[k * in_ch + ic];
                    }
                }

                // Debug: show first few values
                LOG_INF("%s: out_conv_kernel loaded: [0]=%e, [7]=%e, [1]=%e\n",
                        __func__, model.out_conv_kernel[0], model.out_conv_kernel[7], model.out_conv_kernel[1]);
                tensors_loaded++;
            }
        }
        // Decoder layers
        else if (name.find("decoder.layers.") == 0) {
            std::string rest = name.substr(15);  // Remove "decoder.layers."
            auto dot_pos = rest.find('.');
            if (dot_pos != std::string::npos) {
                int layer_idx = std::stoi(rest.substr(0, dot_pos));
                std::string tensor_rest = rest.substr(dot_pos);

                if (layer_idx >= 0 && layer_idx < (int)model.layers.size()) {
                    auto & layer = model.layers[layer_idx];
                    int stride = model.decoder_rates[layer_idx];
                    layer.layer_idx = layer_idx;  // For debug output
                    layer.stride = stride;
                    layer.in_channels = (layer_idx == 0) ? model.decoder_dim : model.layers[layer_idx - 1].out_channels;
                    // Each decoder layer halves the channels (1024 -> 512 -> 256 -> 128)
                    // The final out_conv projects from 64 to 1
                    layer.out_channels = layer.in_channels / 2;
                    layer.kernel_size = 2 * stride;
                    layer.padding = (stride + 1) / 2;
                    // IMPORTANT: ConvTranspose1D uses groups=1 (not out_channels)
                    // The groups parameter is only for ResidualUnit
                    layer.groups = 1;
                    layer.use_noise = true;

                    LOG_INF("DEBUG: decoder layer %d: stride=%d, kernel=%d, padding=%d, in_ch=%d, out_ch=%d, groups=%d\n",
                            layer_idx, stride, layer.kernel_size, layer.padding, layer.in_channels, layer.out_channels, layer.groups);

                    if (tensor_rest == ".alpha") {
                        if (copy_tensor_to_vector(tensor, layer.in_alpha)) {
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".conv_t.bias") {
                        if (copy_tensor_to_vector(tensor, layer.in_bias)) {
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".conv_t.weight") {
                        // ConvTranspose1d weight loading for SNAC model
                        // SNAC stores weights in format: [in_channels, kernel_size, out_channels]
                        // Example: [1024, 16, 512] for in=1024, k=16, out=512
                        // GGUF stores dimensions in REVERSE order from numpy
                        // So numpy [1024, 16, 512] -> GGUF ne[]=[512, 16, 1024]
                        // Our conv_transpose1d expects: [in_channels, out_channels, kernel_size]
                        std::vector<float> raw;
                        if (copy_tensor_to_vector(tensor, raw)) {
                            int64_t ne0 = tensor->ne[0];  // in_channels (first dim)
                            int64_t ne1 = tensor->ne[1];  // out_channels (second dim)
                            int64_t ne2 = tensor->ne[2];  // kernel_size (third dim)

                            int in_c = ne0;   // in_channels = ne0
                            int out_c = ne1;  // out_channels = ne1
                            int ks = ne2;     // kernel_size = ne2

                            layer.kernel_size = ks;  // Use actual kernel size from tensor
                            layer.padding = (ks - stride) / 2;  // Recalculate padding

                            // GGUF stores data in row-major order [in_c, out_c, ks]
                            // This is the same format our conv_transpose1d expects!
                            // No transpose needed, just copy directly
                            layer.in_kernel.resize(in_c * out_c * ks);
                            for (int i = 0; i < in_c * out_c * ks; i++) {
                                layer.in_kernel[i] = raw[i];
                            }

                            LOG_WRN("DEBUG: decoder.layers.%d.conv_t.weight: GGUF dims [%lld,%lld,%lld] -> [%d,%d,%d] (in_c, out_c, ks), SNAC format transposed\n",
                                    layer_idx, (long long)ne0, (long long)ne1, (long long)ne2, in_c, out_c, ks);
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".noise_proj.weight") {
                        // 1x1 conv weight: GGUF [ch, 1, ch] in column-major
                        // For 1x1 conv, effective shape is [ch, ch] matrix
                        // Need transpose from column-major to row-major
                        std::vector<float> raw;
                        if (copy_tensor_to_vector(tensor, raw)) {
                            int64_t ne0 = tensor->ne[0];  // out_channels
                            int64_t ne2 = tensor->ne[2];  // in_channels
                            int ch = ne0;  // Should be same as ne2 for noise_proj
                            layer.noise_kernel.resize(ch * ch);
                            // GGUF column-major [ch, ch]: element (oc, ic) at oc + ch*ic
                            // PyTorch row-major [ch, ch]: element (oc, ic) at oc*ch + ic
                            for (int oc = 0; oc < ch; oc++) {
                                for (int ic = 0; ic < ch; ic++) {
                                    layer.noise_kernel[oc * ch + ic] = raw[oc + ch * ic];
                                }
                            }
                            tensors_loaded++;
                        }
                    }
                    // Residual units
                    else if (tensor_rest.find(".residual_units.") == 0) {
                        std::string unit_rest = tensor_rest.substr(16);  // Remove ".residual_units."
                        auto unit_dot_pos = unit_rest.find('.');
                        if (unit_dot_pos != std::string::npos) {
                            int unit_idx = std::stoi(unit_rest.substr(0, unit_dot_pos));
                            std::string unit_tensor = unit_rest.substr(unit_dot_pos);

                            if (unit_idx >= 0) {
                                if ((int)layer.residual_units.size() <= unit_idx) {
                                    layer.residual_units.resize(unit_idx + 1);
                                }
                                auto & unit = layer.residual_units[unit_idx];
                                unit.channels = layer.out_channels;
                                unit.kernel_size = 7;
                                unit.dilation = (int)std::pow(3, unit_idx);
                                unit.padding = ((unit.kernel_size - 1) * unit.dilation) / 2;
                                unit.groups = unit.channels;

                                if (unit_tensor == ".in_alpha") {
                                    if (copy_tensor_to_vector(tensor, unit.in_alpha)) {
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".in_conv.bias") {
                                    if (copy_tensor_to_vector(tensor, unit.in_bias)) {
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".in_conv.weight") {
                                    // Depthwise conv weight: GGUF [7, ch] in column-major
                                    // For depthwise conv, we access kernel[ch * kernel_size + k]
                                    // Column-major [7, ch]: element (k, c) at k + 7*c = c*7 + k (same!)
                                    // No transpose needed
                                    if (copy_tensor_to_vector(tensor, unit.in_kernel)) {
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".out_alpha") {
                                    if (copy_tensor_to_vector(tensor, unit.out_alpha)) {
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".out_conv.bias") {
                                    if (copy_tensor_to_vector(tensor, unit.out_bias)) {
                                        tensors_loaded++;
                                    }
                                }
                                else if (unit_tensor == ".out_conv.weight") {
                                    // 1x1 conv weight: GGUF numpy shape [ch, ch] in row-major
                                    // No transpose needed - data is already in correct order
                                    if (copy_tensor_to_vector(tensor, unit.out_kernel)) {
                                        tensors_loaded++;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Quantizers
        else if (name.find("quantizers.") == 0) {
            std::string rest = name.substr(11);  // Remove "quantizers."
            auto dot_pos = rest.find('.');
            if (dot_pos != std::string::npos) {
                int quant_idx = std::stoi(rest.substr(0, dot_pos));
                std::string tensor_rest = rest.substr(dot_pos);

                if (quant_idx >= 0 && quant_idx < (int)model.quantizers.size()) {
                    auto & quant = model.quantizers[quant_idx];
                    quant.codebook_size = SNAC_CODEBOOK_SIZE;
                    quant.codebook_dim = model.codebook_dim;
                    quant.quantizer_dim = model.quantizer_dim;

                    if (tensor_rest == ".codebook.weight") {
                        // Original model codebook shape: [codebook_size, codebook_dim] = [4096, 8]
                        // GGUF reports reversed dimensions but data is in original layout!
                        // Data is already in [4096, 8] = [codebook_size, codebook_dim] - NO transpose needed!
                        std::vector<float> raw;
                        if (copy_tensor_to_vector(tensor, raw)) {
                            quant.codebook = raw;
                            quant.codebook_dim = 8;  // Always 8 for snac_24khz
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".in_proj.bias") {
                        if (copy_tensor_to_vector(tensor, quant.in_proj_bias)) {
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".in_proj.weight") {
                        if (copy_tensor_to_vector(tensor, quant.in_proj_weight)) {
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".out_proj.bias") {
                        if (copy_tensor_to_vector(tensor, quant.out_proj_bias)) {
                            tensors_loaded++;
                        }
                    }
                    else if (tensor_rest == ".out_proj.weight") {
                        if (copy_tensor_to_vector(tensor, quant.out_proj_weight)) {
                            tensors_loaded++;
                        }
                    }
                }
            }
        }
    }
    }  // end else (snac_24khz format)

    ggml_free(ggml_ctx);
    gguf_free(gguf_ctx);

    if (tensors_loaded == 0) {
        LOG_WRN("%s: No SNAC tensors found in GGUF file, using placeholder weights\n", __func__);
        // Fall back to random initialization
        std::mt19937 rng(42);
        std::normal_distribution<float> dist(0.0f, 0.02f);

        auto rand_init = [&](std::vector<float> & v, size_t size) {
            v.resize(size);
            for (auto & x : v) x = dist(rng);
        };

        // Initialize with random weights
        for (int h = 0; h < model.n_quantizers; h++) {
            rand_init(model.quantizers[h].codebook, SNAC_CODEBOOK_SIZE * model.codebook_dim);
            rand_init(model.quantizers[h].in_proj_bias, model.codebook_dim);
            rand_init(model.quantizers[h].in_proj_weight, model.codebook_dim * model.quantizer_dim);
            rand_init(model.quantizers[h].out_proj_bias, model.quantizer_dim);
            rand_init(model.quantizers[h].out_proj_weight, model.quantizer_dim * model.codebook_dim);
        }

        rand_init(model.in_conv_kernel, 7 * 1 * model.quantizer_dim);
        rand_init(model.in_conv_bias, model.quantizer_dim);
        rand_init(model.up_conv_kernel, 1 * model.quantizer_dim * model.decoder_dim);
        rand_init(model.up_conv_bias, model.decoder_dim);
        rand_init(model.out_conv_kernel, 7 * model.decoder_dim * 1);
        rand_init(model.out_conv_bias, 1);
        rand_init(model.snake_alpha_out, model.decoder_dim);

        int channels = model.decoder_dim;
        for (int l = 0; l < model.n_decoder_layers; l++) {
            auto & layer = model.layers[l];
            int out_channels = channels / 2;
            if (l == model.n_decoder_layers - 1) out_channels = 1;

            layer.layer_idx = l;  // For debug output
            layer.stride = model.decoder_rates[l];
            layer.in_channels = channels;
            layer.out_channels = out_channels;
            layer.kernel_size = 2 * layer.stride;
            layer.padding = (layer.stride + 1) / 2;
            layer.groups = 1;  // ConvTranspose1D uses groups=1
            layer.use_noise = false;  // Disable noise for debugging

            rand_init(layer.in_alpha, channels);
            rand_init(layer.in_kernel, layer.kernel_size * 1 * out_channels);
            rand_init(layer.in_bias, out_channels);
            rand_init(layer.noise_kernel, 1 * out_channels * out_channels);

            layer.residual_units.resize(3);
            for (int u = 0; u < 3; u++) {
                auto & unit = layer.residual_units[u];
                unit.channels = out_channels;
                unit.kernel_size = 7;
                unit.dilation = (int)std::pow(3, u);
                unit.padding = ((unit.kernel_size - 1) * unit.dilation) / 2;
                unit.groups = out_channels;

                rand_init(unit.in_alpha, out_channels);
                rand_init(unit.in_kernel, unit.kernel_size * 1 * out_channels);
                rand_init(unit.in_bias, out_channels);
                rand_init(unit.out_alpha, out_channels);
                rand_init(unit.out_kernel, 1 * out_channels * out_channels);
                rand_init(unit.out_bias, out_channels);
            }
            channels = out_channels;
        }

        LOG_WRN("%s: SNAC model initialized with random weights - output will be noise\n", __func__);
    } else {
        LOG_INF("%s: Loaded %d SNAC tensors from GGUF\n", __func__, tensors_loaded);
    }

    model.loaded = true;
    return true;
}

// Normalize audio samples to proper amplitude
// Uses peak normalization to ensure maximum utilization of int16 range
static void normalize_audio(std::vector<float> & samples, float target_peak = 0.95f) {
    if (samples.empty()) {
        return;
    }

    // Remove DC offset first (snake activation adds positive bias)
    float mean = 0.0f;
    for (const auto & sample : samples) {
        mean += sample;
    }
    mean /= samples.size();

    if (std::abs(mean) > 1e-9f) {  // Always remove DC offset (snake activation adds positive bias)
        for (auto & sample : samples) {
            sample -= mean;
        }
        LOG_INF("%s: Removed DC offset (mean was %.6f)\n", __func__, mean);
    }

    // Find peak absolute value
    float peak = 0.0f;
    for (const auto & sample : samples) {
        peak = std::max(peak, std::abs(sample));
    }

    // If peak is too low, normalize; if already near target, leave as is
    const float min_threshold = 0.01f;  // Only normalize if peak < 1% of target
    if (peak < min_threshold) {
        float scale = target_peak / peak;
        for (auto & sample : samples) {
            sample *= scale;
        }
        LOG_INF("%s: Normalized audio (peak was %.6f, scaled by %.2f)\n",
                __func__, peak, scale);
    } else if (peak < target_peak * 0.5f) {
        // Peak is moderate but could be better
        float scale = target_peak / peak;
        for (auto & sample : samples) {
            sample *= scale;
        }
        LOG_INF("%s: Boosted audio (peak was %.6f, scaled by %.2f)\n",
                __func__, peak, scale);
    } else if (peak > 1.0f) {
        // Peak exceeds 1.0, need to attenuate to prevent clipping
        float scale = target_peak / peak;
        for (auto & sample : samples) {
            sample *= scale;
        }
        LOG_INF("%s: Attenuated audio (peak was %.6f, scaled by %.2f)\n",
                __func__, peak, scale);
    } else {
        LOG_INF("%s: Audio already properly normalized (peak=%.6f)\n",
                __func__, peak);
    }
}

// Decode audio tokens to PCM using SNAC
static void decode_snac_tokens(
    snac_model & snac,
    const std::vector<std::vector<int>> & pyramid_tokens,
    std::vector<float> & pcm_samples
) {
    printf("DEBUG: decode_snac_tokens called, pyramid_tokens.size()=%zu\n", pyramid_tokens.size());
    fflush(stdout);
    if (!pyramid_tokens.empty()) {
        printf("DEBUG: pyramid_tokens[0].size()=%zu, [1].size()=%zu, [2].size()=%zu\n",
               pyramid_tokens[0].size(), pyramid_tokens[1].size(), pyramid_tokens[2].size());
        fflush(stdout);
    }

    pcm_samples = snac.decode(pyramid_tokens);

    printf("DEBUG: snac.decode returned, pcm_samples.size()=%zu\n", pcm_samples.size());
    fflush(stdout);

    // Normalize audio samples to proper amplitude
    normalize_audio(pcm_samples);

    LOG_INF("%s: Decoded %zu samples (%.2f seconds)\n",
            __func__, pcm_samples.size(),
            (float)pcm_samples.size() / SNAC_SAMPLE_RATE);
}

int main(int argc, char ** argv) {
    // Parameters
    std::string model_path;
    std::string vocoder_path;
    std::string prompt_text;
    std::string output_path = "output.wav";
    std::string voice_name;
    int n_threads = -1;
    int n_predict = 2048;
    float temp = 0.1f;
    int top_k = 40;
    float top_p = 0.9f;
    bool test_vocoder = false;
    bool test_snac_ggml = false;
    int test_frames = 10;  // Number of frames for vocoder test
    bool use_snac_ggml = false;  // Use ggml-based SNAC implementation
    int batch_size = 1;  // Batch size for SNAC processing
    bool use_gpu = false;  // Use GPU backend for SNAC
    int n_gpu_layers = -1;  // Number of layers to offload to GPU (-1 = all)

    // Streaming TTS parameters
    bool streaming_mode = false;
    int streaming_chunk_frames = 32;  // Increased from 8 to reduce re-decode frequency (O(n²) -> O(n))
    int streaming_overlap = 4;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argc, argv);
            return 0;
        } else if ((arg == "-m" || arg == "--model") && i + 1 < argc) {
            model_path = argv[++i];
        } else if (arg == "--model-vocoder" && i + 1 < argc) {
            vocoder_path = argv[++i];
        } else if ((arg == "-p" || arg == "--prompt") && i + 1 < argc) {
            prompt_text = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            output_path = argv[++i];
        } else if ((arg == "-v" || arg == "--voice") && i + 1 < argc) {
            voice_name = argv[++i];
        } else if ((arg == "-t" || arg == "--threads") && i + 1 < argc) {
            n_threads = std::stoi(argv[++i]);
        } else if ((arg == "-n" || arg == "--n-predict") && i + 1 < argc) {
            n_predict = std::stoi(argv[++i]);
        } else if (arg == "--temp" && i + 1 < argc) {
            temp = std::stof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            top_k = std::stoi(argv[++i]);
        } else if (arg == "--top-p" && i + 1 < argc) {
            top_p = std::stof(argv[++i]);
        } else if ((arg == "-ngl" || arg == "--n-gpu-layers") && i + 1 < argc) {
            n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--test-vocoder") {
            test_vocoder = true;
        } else if (arg == "--test-snac-ggml") {
            test_snac_ggml = true;
        } else if (arg == "--test-frames" && i + 1 < argc) {
            test_frames = std::stoi(argv[++i]);
        } else if (arg == "--use-snac-ggml") {
            use_snac_ggml = true;
        } else if (arg == "--batch-size" && i + 1 < argc) {
            batch_size = std::stoi(argv[++i]);
            if (batch_size < 1) {
                LOG_ERR("Error: batch-size must be >= 1\n");
                return 1;
            }
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--streaming") {
            streaming_mode = true;
        } else if (arg == "--streaming-chunk-frames" && i + 1 < argc) {
            streaming_chunk_frames = std::stoi(argv[++i]);
            if (streaming_chunk_frames < 4) {
                LOG_ERR("Error: streaming-chunk-frames must be >= 4\n");
                return 1;
            }
        } else if (arg == "--streaming-overlap" && i + 1 < argc) {
            streaming_overlap = std::stoi(argv[++i]);
            if (streaming_overlap < 2 || streaming_overlap > 16) {
                LOG_ERR("Error: streaming-overlap must be in range [2, 16]\n");
                return 1;
            }
        } else {
            LOG_ERR("Unknown argument: %s\n", arg.c_str());
            print_usage(argc, argv);
            return 1;
        }
    }

    // Validate required arguments
    // For test-vocoder mode, only vocoder path is required
    if (test_vocoder) {
        if (vocoder_path.empty()) {
            LOG_ERR("Error: Vocoder path is required for test mode (--model-vocoder)\n");
            print_usage(argc, argv);
            return 1;
        }

        LOG_INF("Testing SNAC vocoder only (no LLM)...\n");

        // Load SNAC vocoder
        snac_model snac;
        if (!load_snac_model(snac, vocoder_path.c_str())) {
            LOG_ERR("Failed to load SNAC vocoder from %s\n", vocoder_path.c_str());
            return 1;
        }

        // Generate random tokens for testing
        // SNAC pyramid structure: head0 has 4x tokens, head1 has 2x, head2 has 1x per frame
        std::vector<std::vector<int>> pyramid_tokens(3);
        for (int f = 0; f < test_frames; f++) {
            // head0: 4 tokens per frame
            for (int j = 0; j < 4; j++) {
                pyramid_tokens[0].push_back(rand() % 4096);
            }
            // head1: 2 tokens per frame
            for (int j = 0; j < 2; j++) {
                pyramid_tokens[1].push_back(rand() % 4096);
            }
            // head2: 1 token per frame
            pyramid_tokens[2].push_back(rand() % 4096);
        }

        LOG_INF("Generated %d test frames with random tokens\n", test_frames);
        LOG_INF("head0: %zu tokens, head1: %zu tokens, head2: %zu tokens\n",
                pyramid_tokens[0].size(), pyramid_tokens[1].size(), pyramid_tokens[2].size());

        // Decode using SNAC
        std::vector<float> pcm_samples;
        decode_snac_tokens(snac, pyramid_tokens, pcm_samples);

        if (pcm_samples.empty()) {
            LOG_ERR("SNAC decode returned no samples\n");
            return 1;
        }

        // Analyze output
        float mn = 1e30f, mx = -1e30f, sum = 0.0f;
        for (float s : pcm_samples) {
            mn = std::min(mn, s);
            mx = std::max(mx, s);
            sum += s;
        }
        float mean = sum / pcm_samples.size();
        LOG_INF("PCM output: %zu samples, min=%.6f, max=%.6f, mean=%.6f\n",
                pcm_samples.size(), mn, mx, mean);

        // Write WAV file
        save_wav16(output_path, pcm_samples, SNAC_SAMPLE_RATE);
        LOG_INF("Wrote test audio to %s\n", output_path.c_str());

        return 0;
    }

    // Test SNAC GGML implementation
    if (test_snac_ggml) {
        if (vocoder_path.empty()) {
            LOG_ERR("Error: Vocoder path is required for test mode (--model-vocoder)\n");
            print_usage(argc, argv);
            return 1;
        }

        LOG_INF("Testing SNAC GGML implementation...\n");

        // Initialize SNAC GGML context
        snac_ggml_context snac_ctx;
        if (!snac_ggml_init(snac_ctx, vocoder_path.c_str(), nullptr)) {
            LOG_ERR("Failed to initialize SNAC GGML context from %s\n", vocoder_path.c_str());
            return 1;
        }

        LOG_INF("SNAC GGML context initialized successfully\n");

        // Known-good Orpheus tokens for testing (should produce intelligible speech)
        // head0 has 19 tokens, head1 has 20 tokens, head2 has 21 tokens
        // vq_strides = [4, 2, 1] means head0*4 should match head2 length
        // So 19*4=76 != 21, but this is the actual Orpheus output format
        const std::vector<int> orpheus_head0 = {3919, 717, 2663, 2775, 1802, 140, 4010, 144, 1838, 412, 342, 3417, 401, 3876, 2690, 234, 2866, 3919, 2068};
        const std::vector<int> orpheus_head1 = {1265, 1685, 3418, 3418, 2512, 1293, 492, 3773, 3323, 3323, 2173, 2186, 2267, 2749, 182, 1983, 2531, 2531, 2609, 1736};
        const std::vector<int> orpheus_head2 = {916, 1043, 1514, 161, 1514, 3909, 3909, 4018, 1775, 3781, 3781, 1514, 916, 1135, 1135, 1135, 1278, 2651, 1559, 2045, 1855};

        // Use Orpheus tokens instead of random tokens for meaningful test
        std::vector<std::vector<int>> pyramid_tokens(3);
        pyramid_tokens[0] = orpheus_head0;
        pyramid_tokens[1] = orpheus_head1;
        pyramid_tokens[2] = orpheus_head2;

        LOG_INF("Using known Orpheus tokens for testing:\n");
        LOG_INF("head0: %zu tokens, head1: %zu tokens, head2: %zu tokens\n",
                pyramid_tokens[0].size(), pyramid_tokens[1].size(), pyramid_tokens[2].size());

        // Decode using SNAC GGML
        std::vector<float> pcm_samples = snac_ggml_decode(snac_ctx, pyramid_tokens);

        if (pcm_samples.empty()) {
            LOG_WRN("SNAC GGML decode returned no samples (Phase 1 - graph execution not yet implemented)\n");
            snac_ggml_free(snac_ctx);
            return 0;  // Not an error - Phase 1 is still in progress
        }

        // Analyze output statistics
        float mn = 1e30f, mx = -1e30f, sum = 0.0f;
        int neg_count = 0;
        for (float s : pcm_samples) {
            mn = std::min(mn, s);
            mx = std::max(mx, s);
            sum += s;
            if (s < 0) neg_count++;
        }
        float mean = sum / pcm_samples.size();
        float neg_ratio = (float)neg_count / pcm_samples.size();

        LOG_INF("PCM output statistics:\n");
        LOG_INF("  Samples: %zu\n", pcm_samples.size());
        LOG_INF("  Min: %.6f\n", mn);
        LOG_INF("  Max: %.6f\n", mx);
        LOG_INF("  Mean: %.6f\n", mean);
        LOG_INF("  Negative ratio: %.2f%% (%d/%zu)\n", neg_ratio * 100, neg_count, pcm_samples.size());

        // Validate statistics against expected values (from MEMORY.md)
        bool pass = true;

        // Expected: neg_ratio should be ~47-50% (not 0%)
        if (neg_ratio < 0.40f || neg_ratio > 0.60f) {
            LOG_ERR("FAIL: Negative ratio %.2f%% is outside expected range [40%%, 60%%]\n", neg_ratio * 100);
            LOG_ERR("      (Expected ~47-50%% for properly functioning vocoder)\n");
            pass = false;
        } else {
            LOG_INF("PASS: Negative ratio is within expected range\n");
        }

        // Expected: mean should be near 0 (not 0.3+)
        if (std::abs(mean) > 0.1f) {
            LOG_ERR("FAIL: Mean %.6f is too far from 0 (expected |mean| < 0.1)\n", mean);
            LOG_ERR("      (DC bias indicates potential issues in vocoder)\n");
            pass = false;
        } else {
            LOG_INF("PASS: Mean is near zero (no significant DC bias)\n");
        }

        // Write WAV file for inspection
        save_wav16(output_path, pcm_samples, SNAC_SAMPLE_RATE);
        LOG_INF("Wrote test audio to %s\n", output_path.c_str());

        // Cleanup
        snac_ggml_free(snac_ctx);

        if (pass) {
            LOG_INF("\n=== SNAC GGML TEST PASSED ===\n");
            return 0;
        } else {
            LOG_ERR("\n=== SNAC GGML TEST FAILED ===\n");
            return 1;
        }
    }

    // Normal mode - require model and prompt
    if (model_path.empty()) {
        LOG_ERR("Error: Model path is required (-m, --model)\n");
        print_usage(argc, argv);
        return 1;
    }

    if (vocoder_path.empty()) {
        LOG_ERR("Error: Vocoder path is required (--model-vocoder)\n");
        print_usage(argc, argv);
        return 1;
    }

    if (prompt_text.empty()) {
        LOG_ERR("Error: Prompt text is required (-p, --prompt)\n");
        print_usage(argc, argv);
        return 1;
    }

    // Streaming mode requires SNAC GGML
    if (streaming_mode && !use_snac_ggml) {
        LOG_ERR("Error: Streaming mode requires --use-snac-ggml\n");
        print_usage(argc, argv);
        return 1;
    }

    // Initialize llama.cpp
    llama_backend_init();
    llama_numa_init(GGML_NUMA_STRATEGY_DISABLED);

    // Load LLM model
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    LOG_INF("Loading model from %s (n_gpu_layers=%d)...\n", model_path.c_str(), n_gpu_layers);
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) {
        LOG_ERR("Failed to load model from %s\n", model_path.c_str());
        return 1;
    }

    // Get vocab
    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Create context
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_batch = 512;
    if (n_threads > 0) {
        ctx_params.n_threads = n_threads;
        ctx_params.n_threads_batch = n_threads;
    }

    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) {
        LOG_ERR("Failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    // Load SNAC vocoder from separate file
    LOG_INF("Loading SNAC vocoder from %s...\n", vocoder_path.c_str());

    // Initialize backend for SNAC GGML
    ggml_backend_t snac_backend = nullptr;
    if (use_gpu) {
        // Try to initialize CUDA backend
        // First try device 0 (primary GPU)
        snac_backend = ggml_backend_cuda_init(0);
        if (snac_backend) {
            LOG_INF("SNAC vocoder using CUDA GPU backend (device 0)\n");
        } else {
            LOG_WRN("CUDA backend initialization failed, falling back to CPU\n");
            snac_backend = ggml_backend_cpu_init();
            if (snac_backend && n_threads > 0) {
                ggml_backend_cpu_set_n_threads(snac_backend, n_threads);
                LOG_INF("SNAC CPU backend using %d threads\n", n_threads);
            }
        }
    } else {
        snac_backend = ggml_backend_cpu_init();
        if (snac_backend && n_threads > 0) {
            ggml_backend_cpu_set_n_threads(snac_backend, n_threads);
            LOG_INF("SNAC CPU backend using %d threads\n", n_threads);
        }
    }

    snac_ggml_context snac_ggml_ctx;
    snac_model snac;

    if (use_snac_ggml) {
        // Use ggml-based SNAC implementation
        if (!snac_ggml_init(snac_ggml_ctx, vocoder_path.c_str(), snac_backend)) {
            LOG_ERR("Failed to load SNAC GGML vocoder from %s\n", vocoder_path.c_str());
            llama_free(ctx);
            llama_model_free(model);
            if (snac_backend) ggml_backend_free(snac_backend);
            return 1;
        }
        LOG_INF("SNAC GGML vocoder loaded successfully\n");

        // Initialize batch context if batch_size > 1
        if (batch_size > 1) {
            snac_batch_context batch_ctx;
            // Note: batch context initialization would be done when we have actual batch processing
            LOG_INF("Batch mode enabled with batch_size=%d\n", batch_size);
        }
    } else {
        // Use legacy SNAC implementation
        if (!load_snac_model(snac, vocoder_path.c_str())) {
            LOG_ERR("Failed to load SNAC vocoder from %s\n", vocoder_path.c_str());
            llama_free(ctx);
            llama_model_free(model);
            if (snac_backend) ggml_backend_free(snac_backend);
            return 1;
        }
        LOG_INF("Legacy SNAC vocoder loaded successfully\n");
    }

    // Initialize streaming context if streaming mode is enabled
    snac_streaming_context streaming_ctx;
    streaming_callback_ctx streaming_cb_ctx;
    std::vector<float> streaming_pcm_samples;  // To collect all streaming output

    if (streaming_mode) {
        snac_streaming_config streaming_cfg;
        streaming_cfg.min_chunk_frames = streaming_chunk_frames;
        streaming_cfg.overlap_frames = streaming_overlap;
        streaming_cfg.crossfade_chunks = true;
        streaming_cfg.crossfade_samples = 256;

        if (!streaming_ctx.init(&snac_ggml_ctx, streaming_cfg)) {
            LOG_ERR("Failed to initialize streaming context\n");
            llama_free(ctx);
            llama_model_free(model);
            if (use_snac_ggml) snac_ggml_free(snac_ggml_ctx);
            if (snac_backend) ggml_backend_free(snac_backend);
            return 1;
        }

        streaming_cb_ctx.all_samples = &streaming_pcm_samples;
        streaming_cb_ctx.verbose = true;
        LOG_INF("Streaming mode enabled (chunk_frames=%d, overlap=%d)\n",
                streaming_chunk_frames, streaming_overlap);
    }

    // Build prompt
    LOG_INF("Building prompt for: \"%s\"\n", prompt_text.c_str());
    std::vector<llama_token> prompt_tokens = build_orpheus_prompt(vocab, prompt_text, voice_name);
    LOG_INF("Prompt tokens: %d\n", (int)prompt_tokens.size());

    // Initialize sampling parameters
    common_params_sampling sampling_params;
    sampling_params.temp = temp;
    sampling_params.top_k = top_k;
    sampling_params.top_p = top_p;
    sampling_params.penalty_repeat = 1.2f;  // Prevent token repetition loops
    // IMPORTANT: COMMON_SAMPLER_TYPE_PENALTIES must be included for penalty_repeat to work!
    sampling_params.samplers = {COMMON_SAMPLER_TYPE_PENALTIES, COMMON_SAMPLER_TYPE_TOP_K, COMMON_SAMPLER_TYPE_TOP_P, COMMON_SAMPLER_TYPE_TEMPERATURE};

    // Initialize sampler
    common_sampler * sampler = common_sampler_init(model, sampling_params);

    // Create batch
    llama_batch batch = llama_batch_init(512, 0, 1);

    // Encode prompt
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        common_batch_add(batch, prompt_tokens[i], i, {0}, false);
    }
    // Set logits for last token
    if (batch.n_tokens > 0) {
        batch.logits[batch.n_tokens - 1] = true;
    }

    if (llama_decode(ctx, batch) != 0) {
        LOG_ERR("Failed to decode prompt\n");
        llama_batch_free(batch);
        common_sampler_free(sampler);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    // Generate tokens
    LOG_INF("Generating audio tokens...\n");
    std::vector<llama_token> generated_tokens;
    int n_pos = prompt_tokens.size();

    // LLM inference timing (starts after prompt encoding)
    auto llm_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < n_predict; i++) {
        // Sample next token
        llama_token token = common_sampler_sample(sampler, ctx, batch.n_tokens - 1);
        common_sampler_accept(sampler, token, true);

        // Check for stop token
        if (token == TOKEN_STOP || token == TOKEN_EOS) {
            LOG_INF("Stop token generated\n");
            break;
        }

        generated_tokens.push_back(token);

        // Print progress and debug token values
        if (token >= AUDIO_TOKEN_START && token <= AUDIO_TOKEN_END) {
            printf(".");
            fflush(stdout);
            // Dump raw audio token values for analysis
            static int audio_token_count = 0;
            if (audio_token_count < 50 || (audio_token_count >= 140 && audio_token_count < 190)) {
                printf("\n[AUDIO_TOK_%d: raw=%d, norm=%d, pos=%d]",
                       audio_token_count, token, token - AUDIO_TOKEN_START, audio_token_count % 7);
                fflush(stdout);
            }
            audio_token_count++;

            // Streaming mode: feed token to streaming decoder
            if (streaming_mode) {
                int normalized_token = token - AUDIO_TOKEN_START;
                streaming_ctx.add_token_and_decode(
                    normalized_token,
                    streaming_audio_callback,
                    &streaming_cb_ctx
                );
            }
        } else if (i < 20) {
            // Debug: print first 20 non-audio tokens
            printf("[tok:%d]", token);
            fflush(stdout);
        } else {
            // Print any non-audio tokens after initial sequence
            printf("\n[NON_AUDIO_TOK_%d: %d]", i, token);
            fflush(stdout);
        }

        // Prepare next batch
        common_batch_clear(batch);
        common_batch_add(batch, token, n_pos++, {0}, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("Failed to decode token\n");
            break;
        }
    }
    printf("\n");

    // LLM inference timing (ends after generation)
    auto llm_end = std::chrono::high_resolution_clock::now();
    auto llm_duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(llm_end - llm_start);

    LOG_INF("Generated %d tokens\n", (int)generated_tokens.size());
    LOG_INF("LLM inference time: %ld ms\n", (long)llm_duration_ms.count());

    std::vector<float> pcm_samples;
    auto decode_start = std::chrono::high_resolution_clock::now();

    // For streaming mode, track SNAC time before flush (inside LLM loop)
    int64_t snac_time_before_flush = 0;

    if (streaming_mode) {
        // Streaming mode: flush remaining tokens and use collected PCM
        // Note: flush SNAC time is tracked in streaming context but happens after LLM loop
        // We need to exclude flush time from SNAC timing for accurate RTF calculation
        snac_time_before_flush = streaming_ctx.get_decode_time_ms();
        LOG_INF("  SNAC time inside LLM loop: %ld ms\n", (long)snac_time_before_flush);

        LOG_INF("Flushing streaming decoder...\n");
        streaming_ctx.flush(streaming_audio_callback, &streaming_cb_ctx);

        int64_t snac_flush_time = streaming_ctx.get_decode_time_ms() - snac_time_before_flush;
        LOG_INF("  SNAC flush time: %ld ms\n", (long)snac_flush_time);

        pcm_samples = std::move(streaming_pcm_samples);

        LOG_INF("Streaming decode complete:\n");
        LOG_INF("  Total chunks: %d\n", streaming_cb_ctx.chunks_received);
        LOG_INF("  Total samples: %zu (%.2fs)\n",
                pcm_samples.size(), (float)pcm_samples.size() / SNAC_GGML_SAMPLE_RATE);

    } else {
        // Non-streaming mode: collect tokens and decode all at once
        printf("DEBUG: Collecting audio tokens...\n");
        fflush(stdout);
        auto pyramid_tokens = collect_audio_tokens_pyramid(generated_tokens);
        printf("DEBUG: pyramid_tokens collected, heads=%zu\n", pyramid_tokens.size());
        fflush(stdout);
        LOG_INF("Audio tokens per head: head0=%zu, head1=%zu, head2=%zu\n",
                pyramid_tokens[0].size(), pyramid_tokens[1].size(), pyramid_tokens[2].size());

        // Dump first and last few tokens from each head for analysis
        printf("\n=== TOKEN DUMP FOR ANALYSIS ===\n");
        for (int h = 0; h < 3; h++) {
            printf("HEAD%d (%zu tokens): ", h, pyramid_tokens[h].size());
            size_t show = std::min((size_t)10, pyramid_tokens[h].size());
            printf("FIRST[%zu]: ", show);
            for (size_t i = 0; i < show; i++) {
                printf("%d ", pyramid_tokens[h][i]);
            }
            if (pyramid_tokens[h].size() > 10) {
                printf("... LAST[%zu]: ", show);
                for (size_t i = pyramid_tokens[h].size() - show; i < pyramid_tokens[h].size(); i++) {
                    printf("%d ", pyramid_tokens[h][i]);
                }
            }
            printf("\n");

            // Check for invalid tokens
            int invalid = 0;
            for (size_t i = 0; i < pyramid_tokens[h].size(); i++) {
                if (pyramid_tokens[h][i] < 0 || pyramid_tokens[h][i] >= 4096) {
                    invalid++;
                }
            }
            if (invalid > 0) {
                printf("  WARNING: %d INVALID TOKENS (outside 0-4095 range)\n", invalid);
            }
        }
        printf("=== END TOKEN DUMP ===\n\n");

        // Decode audio tokens to PCM using SNAC
        printf("DEBUG: Decoding audio tokens...\n");
        fflush(stdout);

        if (use_snac_ggml) {
            // Use ggml-based SNAC implementation
            LOG_INF("Using SNAC GGML decoder (batch_size=%d)\n", batch_size);

            if (batch_size > 1) {
                LOG_WRN("Batch mode requested but currently processing single sequence\n");
            }

            pcm_samples = snac_ggml_decode(snac_ggml_ctx, pyramid_tokens);

        } else {
            // Use legacy SNAC implementation
            decode_snac_tokens(snac, pyramid_tokens, pcm_samples);
        }
    }

    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_duration = std::chrono::duration_cast<std::chrono::milliseconds>(decode_end - decode_start);

    printf("DEBUG: Decode returned, samples=%zu\n", pcm_samples.size());
    fflush(stdout);

    // Report performance metrics with RTF breakdown
    if (!pcm_samples.empty()) {
        float audio_duration = (float)pcm_samples.size() / SNAC_SAMPLE_RATE;

        // Calculate individual RTFs
        // For streaming mode, the timing is complex because SNAC decode happens interleaved with LLM
        // The total time (llm_duration_ms for streaming) is the wall-clock time
        // SNAC time is tracked separately but happens inside the LLM loop
        int64_t pure_llm_ms, snac_ms, total_time_ms;
        if (streaming_mode) {
            snac_ms = snac_time_before_flush;  // SNAC time from inside LLM loop
            // For streaming, total time is just the LLM loop time (SNAC is inside it)
            total_time_ms = llm_duration_ms.count();
            // Pure LLM time cannot be negative - if SNAC > total, just show 0
            pure_llm_ms = std::max((int64_t)0, total_time_ms - snac_ms);
        } else {
            snac_ms = decode_duration.count();
            pure_llm_ms = llm_duration_ms.count();
            total_time_ms = pure_llm_ms + snac_ms;
        }

        float llm_rtf = (pure_llm_ms / 1000.0f) / audio_duration;
        float snac_rtf = (snac_ms / 1000.0f) / audio_duration;
        float total_rtf = (total_time_ms / 1000.0f) / audio_duration;

        LOG_INF("\n");
        LOG_INF("========================================\n");
        LOG_INF("PERFORMANCE REPORT (RTF Breakdown)\n");
        LOG_INF("========================================\n");
        LOG_INF("  Audio duration:    %.2f seconds\n", audio_duration);
        LOG_INF("  LLM inference:     %ld ms  (RTF: %.2fx)\n", (long)pure_llm_ms, llm_rtf);
        LOG_INF("  SNAC vocoder:      %ld ms  (RTF: %.2fx)\n", (long)snac_ms, snac_rtf);
        LOG_INF("  ----------------------------------------\n");
        LOG_INF("  Total processing:  %ld ms  (RTF: %.2fx)\n", (long)total_time_ms, total_rtf);
        LOG_INF("========================================\n");

        if (total_rtf < 1.0f) {
            LOG_INF("✓ RTF < 1.0: REAL-TIME CAPABLE\n");
        } else {
            LOG_INF("✗ RTF >= 1.0: NOT REAL-TIME (need optimization)\n");
        }
        LOG_INF("\n");

        if (streaming_mode) {
            LOG_INF("  Mode: Streaming (chunk_frames=%d, overlap=%d)\n",
                    streaming_chunk_frames, streaming_overlap);
        } else if (use_snac_ggml) {
            LOG_INF("  SNAC Backend: GGML (%s)\n", use_gpu ? "GPU" : "CPU");
        } else {
            LOG_INF("  SNAC Backend: Legacy CPU\n");
        }
    }

    // Save to WAV
    if (!pcm_samples.empty()) {
        LOG_INF("Saving to %s...\n", output_path.c_str());
        if (save_wav16(output_path, pcm_samples, SNAC_SAMPLE_RATE)) {
            LOG_INF("Audio saved to %s (%.2f seconds)\n",
                    output_path.c_str(),
                    (float)pcm_samples.size() / SNAC_SAMPLE_RATE);
        } else {
            LOG_ERR("Failed to save audio to %s\n", output_path.c_str());
        }
    } else {
        LOG_WRN("No audio samples generated\n");
    }

    // Cleanup
    llama_batch_free(batch);
    common_sampler_free(sampler);

    // Free SNAC resources
    if (use_snac_ggml) {
        snac_ggml_free(snac_ggml_ctx);
        if (snac_backend) {
            ggml_backend_free(snac_backend);
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();

    LOG_INF("Done\n");
    return 0;
}
