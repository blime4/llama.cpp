# Orpheus-TTS Implementation for llama.cpp

## Overview

Orpheus-TTS is an emotion-expressive Text-to-Speech model that uses a two-stage architecture:
1. **LLM-based Token Generation**: A LLaMA-based language model generates discrete audio tokens
2. **SNAC Neural Vocoder**: Converts audio tokens to PCM waveform

This document provides comprehensive documentation of the implementation in `tools/tts/orpheus-tts.cpp`.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Special Tokens](#special-tokens)
3. [SNAC Vocoder Constants](#snac-vocoder-constants)
4. [Prompt Building](#prompt-building)
5. [Token Collection (Pyramid Structure)](#token-collection-pyramid-structure)
6. [SNAC Vocoder Implementation](#snac-vocoder-implementation)
7. [Weight Loading and GGUF Handling](#weight-loading-and-gguf-handling)
8. [Audio Output](#audio-output)
9. [Main Function Flow](#main-function-flow)
10. [Usage](#usage)
11. [Debugging](#debugging)

---

## Architecture Overview

```
+-----------------------------------------------------------------+
|                         Orpheus-TTS Pipeline                     |
+-----------------------------------------------------------------+
|                                                                 |
|   Text Input --> LLM (LLaMA) --> Audio Tokens --> SNAC --> WAV  |
|                                                                 |
|   +-----------+    +----------+    +---------+    +-----+       |
|   |  Prompt   |---|  Token   |---| Pyramid |---|PCM  |       |
|   |  Builder  |    | Generator|    | Structure|   |Output|     |
|   +-----------+    +----------+    +---------+    +-----+       |
|                                                                 |
|   +---------------------------------------------------------+   |
|   |                    SNAC Vocoder                         |   |
|   |  Quantizers --> in_conv --> up_conv --> Decoder --> Out |   |
|   |   (3 heads)    (DW Conv)  (1x1 Conv)   (4 layers)       |   |
|   +---------------------------------------------------------+   |
|                                                                 |
+-----------------------------------------------------------------+
```

---

## Special Tokens

### Constants Definition

```cpp
static const llama_token TOKEN_BOS = 128000;
static const llama_token TOKEN_EOS = 128001;
static const llama_token TOKEN_START_SPEECH = 128259;
static const llama_token TOKEN_END_TURN = 128009;
static const llama_token TOKEN_START_RESPONSE = 128260;
static const llama_token TOKEN_END_RESPONSE = 128261;
static const llama_token TOKEN_AUDIO_START = 128257;
static const llama_token TOKEN_STOP = 128258;

static const llama_token AUDIO_TOKEN_START = 128266;
static const llama_token AUDIO_TOKEN_END = 156937;
```

### Token Descriptions

| Token | Value | Description |
|-------|-------|-------------|
| `TOKEN_BOS` | 128000 | Beginning of sequence |
| `TOKEN_EOS` | 128001 | End of sequence |
| `TOKEN_START_SPEECH` | 128259 | Marks start of speech input |
| `TOKEN_END_TURN` | 128009 | Marks end of user turn |
| `TOKEN_START_RESPONSE` | 128260 | Marks start of model response |
| `TOKEN_END_RESPONSE` | 128261 | Marks end of model response |
| `TOKEN_AUDIO_START` | 128257 | Marks start of audio token sequence |
| `TOKEN_STOP` | 128258 | Stop token for generation |
| `AUDIO_TOKEN_START` | 128266 | First audio codebook token |
| `AUDIO_TOKEN_END` | 156937 | Last audio codebook token |

### Audio Token Range

- **Range**: 128266 to 156937 (28,672 tokens)
- **Purpose**: Represent discrete audio codes from SNAC codebooks
- **Per-frame**: 7 tokens per frame (pyramid structure)

---

## SNAC Vocoder Constants

### Model Configuration

```cpp
static const int SNAC_FRAME_SIZE = 7;                  // 7 tokens per frame
static const int SNAC_SAMPLE_RATE = 24000;             // 24kHz output
static const int SNAC_UPSAMPLE_FACTOR = 512;           // 8*8*4*2 = 512
static const int SNAC_CODEBOOK_SIZE = 4096;            // 4096 entries per codebook
static const int SNAC_DECODER_DIM = 1024;              // Decoder dimension
static const int SNAC_QUANTIZER_DIM = 768;             // Quantizer output dimension
static const int SNAC_CODEBOOK_DIM = 8;                // Codebook embedding dimension
static const int SNAC_N_QUANTIZERS = 3;                // 3 quantizer heads
static const int SNAC_N_DECODER_LAYERS = 4;            // 4 decoder layers
static const int SNAC_DECODER_RATES[] = {8, 8, 4, 2};  // Upsampling rates
static const int SNAC_VQ_STRIDES[] = {8, 4, 2, 1};     // VQ strides for pyramid
```

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Sample Rate | 24000 Hz | Output audio sample rate |
| Frame Size | 7 | Tokens per frame (pyramid) |
| Codebook Size | 4096 | Entries in each codebook |
| Codebook Dim | 8 | Embedding dimension |
| Quantizer Dim | 768 | Output dimension of quantizers |
| Decoder Dim | 1024 | Decoder hidden dimension |
| Upsample Factor | 512 | Total upsampling (8x8x4x2) |

---

## Prompt Building

### `build_orpheus_prompt()`

**Purpose**: Construct the prompt token sequence for Orpheus TTS.

```cpp
static std::vector<llama_token> build_orpheus_prompt(
    const llama_vocab * vocab,
    const std::string & text,
    const std::string & voice
);
```

**Parameters**:
- `vocab`: LLaMA vocabulary for tokenization
- `text`: Text to synthesize
- `voice`: Optional voice/speaker name (e.g., "tara", "leo")

**Process**:
1. Add `TOKEN_START_SPEECH` and `TOKEN_BOS`
2. Add voice prefix if specified: `voice + ": " + text`
3. Tokenize text using `common_tokenize()`
4. Add turn markers: `TOKEN_END_TURN`, `TOKEN_START_RESPONSE`, `TOKEN_END_RESPONSE`
5. Add `TOKEN_AUDIO_START` to signal audio generation

**Prompt Structure**:
```
[START_SPEECH][BOS][voice: text tokens...][END_TURN][START_RESPONSE][END_RESPONSE][AUDIO_START]
```

**Example**:
```cpp
// Input: text = "Hello world", voice = "tara"
// Output tokens:
// [128259][128000][tara: Hello world...][128009][128260][128261][128257]
```

---

## Token Collection (Pyramid Structure)

### SNAC Pyramid Structure

SNAC uses a pyramid structure for audio tokens across 3 quantizers:

```
Frame: [T0, T1, T2, T3, T4, T5, T6]
        |   |   |   |   |   |   |
        v   v   v   v   v   v   v
Head:   0   1   2   2   1   2   2

Token Distribution:
- Head 0: 1 token per frame (T0) --> stride 4 --> 4N output
- Head 1: 2 tokens per frame (T1, T4) --> stride 2 --> 4N output
- Head 2: 4 tokens per frame (T2, T3, T5, T6) --> stride 1 --> 4N output
```

### `collect_audio_tokens_pyramid()`

**Purpose**: Extract and organize audio tokens into pyramid structure.

```cpp
static std::vector<std::vector<int>> collect_audio_tokens_pyramid(
    const std::vector<llama_token> & tokens
);
```

**Process**:
1. Filter tokens in range `[AUDIO_TOKEN_START, AUDIO_TOKEN_END]`
2. Normalize: `token - AUDIO_TOKEN_START`
3. Apply pyramid mapping: `{0, 1, 2, 2, 1, 2, 2}`
4. Calculate position-based offset: `id = token - pos * 4096`
5. Distribute to 3 codebook vectors

**Key Code**:
```cpp
static const int pyramid_map[SNAC_FRAME_SIZE] = {0, 1, 2, 2, 1, 2, 2};

for (size_t f = 0; f < n_frames; f++) {
    for (int pos = 0; pos < SNAC_FRAME_SIZE; pos++) {
        int codebook = pyramid_map[pos];
        // Position-based offset: each position in frame uses different codebook range
        int id = all_tokens[f * SNAC_FRAME_SIZE + pos] - pos * SNAC_CODEBOOK_SIZE;
        result[codebook].push_back(id);
    }
}
```

**Position-Based Offset Explanation**:
- Position 0: range [0, 4095]
- Position 1: range [4096, 8191]
- Position 2: range [8192, 12287]
- etc.

---

## SNAC Vocoder Implementation

### Activation Functions

#### `snake_1d_inplace()`

**Purpose**: Apply Snake1D activation function in-place.

```cpp
static void snake_1d_inplace(
    float * data,
    const float * alpha,
    int64_t n,
    int64_t channels
);
```

**Formula**:
```
snake(x, alpha) = x + sin^2(alpha * x) / alpha
```

**Implementation**:
```cpp
for (int64_t i = 0; i < n; i++) {
    for (int64_t c = 0; c < channels; c++) {
        float x = data[i * channels + c];
        float a = alpha[c];
        float sin_val = std::sin(a * x);
        data[i * channels + c] = x + (sin_val * sin_val) / (a + 1e-9f);
    }
}
```

**Note**: Snake activation inherently adds positive values (sin^2 is always >= 0), which can cause DC bias accumulation through the network. The model compensates during training, but residual DC may remain.

### Convolution Functions

#### `conv1d_dw()` - Depthwise 1D Convolution

**Purpose**: Perform depthwise 1D convolution with grouped channels.

```cpp
static void conv1d_dw(
    const float * input, const float * kernel, const float * bias,
    float * output,
    int64_t input_len, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation,
    int64_t groups
);
```

**Weight Layout**: `[out_channels, in_channels/groups, kernel_size]` (row-major)

**Key Feature**: Each input channel is convolved with its own filter (groups = channels).

#### `conv1d()` - Standard 1D Convolution

**Purpose**: Perform standard 1D convolution.

```cpp
static void conv1d(
    const float * input, const float * kernel, const float * bias,
    float * output,
    int64_t input_len, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation
);
```

**Weight Layout**: `[out_channels, in_channels, kernel_size]` (row-major)

#### `conv_transpose1d()` - Transposed 1D Convolution

**Purpose**: Perform transposed 1D convolution (upsampling).

```cpp
static void conv_transpose1d(
    const float * input, const float * kernel, const float * bias,
    float * output,
    int64_t input_len, int64_t in_channels, int64_t out_channels,
    int64_t kernel_size, int64_t stride, int64_t padding, int64_t output_padding
);
```

**Weight Layout**: `[in_channels, out_channels/groups, kernel_size]` (row-major)

**Output Length Formula**:
```
output_len = (input_len - 1) * stride + kernel_size - 2 * padding + output_padding
```

**Implementation Algorithm**:
1. Insert (stride-1) zeros between input elements: `expanded[i*stride] = input[i]`
2. Flip kernel along kernel dimension: `k_flipped = kernel_size - 1 - k`
3. Apply regular conv1d with effective padding: `p = kernel_size - 1 - padding`

**Example**: For stride=8, kernel=16, padding=4:
- Expanded input length: (input_len - 1) * 8 + 1
- Effective conv1d padding: p = 16 - 1 - 4 = 11
- Output length: (input_len - 1) * 8 + 16 - 2*4 = 8*input_len

### Attention Functions

#### `local_mha_forward()` - Local Multi-Head Attention

**Purpose**: Implement windowed self-attention with Rotary Position Embeddings (RoPE).

```cpp
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
);
```

**Steps**:
1. Save residual connection
2. Apply LayerNorm over channel dimension
3. QKV projection: `[T, C] @ [C, 3C] --> [T, 3C]`
4. Reshape to `[heads, n_windows, window_size, dim_head]`
5. Apply RoPE with xpos-like scaling
6. Scaled dot-product attention per window
7. Reshape back to `[T, C]`
8. Output projection with residual

### SNAC Structures

#### `snac_residual_unit`

**Purpose**: Residual block with Snake activation and dilated convolution.

```cpp
struct snac_residual_unit {
    std::vector<float> in_alpha;      // Snake alpha for input
    std::vector<float> in_kernel;     // Depthwise conv kernel
    std::vector<float> in_bias;
    std::vector<float> out_alpha;     // Snake alpha for output
    std::vector<float> out_kernel;    // 1x1 conv kernel
    std::vector<float> out_bias;
    int padding = 0;
    int dilation = 1;
    int groups = 1;
    int channels = 0;
    int kernel_size = 0;

    std::vector<float> forward(const float * input, int64_t input_len);
};
```

**Forward Pass**:
```
output = Conv1x1(Snake(Conv(Snake(x)))) + x  (residual)
```

#### `snac_decoder_layer`

**Purpose**: Decoder layer with transposed convolution and residual units.

```cpp
struct snac_decoder_layer {
    std::vector<float> in_alpha;      // Snake alpha
    std::vector<float> in_kernel;     // ConvTranspose1d kernel
    std::vector<float> in_bias;
    std::vector<float> noise_kernel;  // Noise injection kernel
    std::vector<snac_residual_unit> residual_units;

    int stride = 2;
    int padding = 0;
    int groups = 1;
    int in_channels = 0;
    int out_channels = 0;
    int kernel_size = 0;
    bool use_noise = false;

    std::vector<float> forward(const float * input, int64_t input_len);
};
```

**Forward Pass**:
```
x = Snake(x)
x = ConvTranspose1D(x)
x = x + Linear(x) * noise  (noise injection)
for unit in residual_units:
    x = unit.forward(x)
return x
```

#### `snac_quantizer_layer`

**Purpose**: Quantizer that converts discrete tokens to embeddings.

```cpp
struct snac_quantizer_layer {
    std::vector<float> codebook;       // [codebook_dim, codebook_size]
    std::vector<float> in_proj_bias;   // [codebook_dim]
    std::vector<float> in_proj_weight; // [codebook_dim, quantizer_dim, 1]
    std::vector<float> out_proj_bias;  // [quantizer_dim]
    std::vector<float> out_proj_weight;// [quantizer_dim, codebook_dim, 1]

    std::vector<float> forward(const int * tokens, int64_t seq_len);
};
```

**Forward Pass**:
1. Codebook lookup: `embedding = codebook[:, token]`
2. Output projection: `output = Conv1D(embedding)`

#### `snac_model`

**Purpose**: Complete SNAC vocoder model.

```cpp
struct snac_model {
    // Input convolutions
    std::vector<float> in_conv_kernel;   // Depthwise conv
    std::vector<float> in_conv_bias;
    std::vector<float> up_conv_kernel;   // 1x1 conv (768 --> 1024)
    std::vector<float> up_conv_bias;

    // Attention layer
    std::vector<float> attn_norm_weight;
    std::vector<float> attn_norm_bias;
    std::vector<float> attn_to_qkv_weight;
    std::vector<float> attn_rel_pos_inv_freq;
    std::vector<float> attn_to_out_weight;

    // Output layer
    std::vector<float> out_conv_kernel;  // Final conv (64 --> 1)
    std::vector<float> out_conv_bias;
    std::vector<float> snake_alpha_out;

    // Decoder layers
    std::vector<snac_decoder_layer> layers;

    // Quantizers (3 heads for TTS)
    std::vector<snac_quantizer_layer> quantizers;

    std::vector<float> decode(const std::vector<std::vector<int>> & pyramid_tokens);
};
```

**Decode Pipeline**:
```
1. Quantizer forward pass for each head (codebook lookup + projection)
2. Combine quantizer outputs using repeat_interleave with vq_strides
3. in_conv: depthwise convolution (768 channels, kernel=7)
4. up_conv: 1x1 convolution (768 --> 1024 channels)
5. LocalMHA attention (window_size=32, dim_head=64)
6. Decoder layers (4 layers with progressive upsampling)
7. Snake activation
8. out_conv: convolution (1024 --> 1 channel, kernel=7)
9. Tanh activation
```

---

## Weight Loading and GGUF Handling

### GGUF Format Considerations

**CRITICAL: GGUF Storage Format**

GGUF stores tensors in **ROW-MAJOR order** (same as numpy), NOT column-major!

However, GGUF stores dimensions in **REVERSE order** from numpy:
- Numpy shape `[a, b, c]` --> GGUF `ne[0]=c, ne[1]=b, ne[2]=a`

**The raw data bytes are in row-major order**, so for numpy array with shape `[a, b, c]`:
- Element `[i, j, k]` is at flat index `i*b*c + j*c + k`
- This is the same in both numpy and GGUF raw bytes!

### Weight Loading Rules

| Tensor | Numpy Shape | GGUF ne[] | Transpose Needed? |
|--------|------------|-----------|-------------------|
| `in.weight` | [768, 7] | ne[0]=7, ne[1]=768 | NO (copy directly) |
| `up.weight` | [1024, 768] | ne[0]=768, ne[1]=1024 | NO (copy directly) |
| `decoder.out_conv.weight` | [7, 64, 1] | ne[0]=1, ne[1]=64, ne[2]=7 | NO (copy directly) |
| `decoder.layers.X.conv_t.weight` | [in_c, out_c, ks] | ne[0]=ks, ne[1]=out_c, ne[2]=in_c | NO (copy directly) |
| `residual_units.Y.in_conv.weight` | [ch, 7] | ne[0]=7, ne[1]=ch | NO (copy directly) |
| `residual_units.Y.out_conv.weight` | [ch, ch] | ne[0]=ch, ne[1]=ch | NO (copy directly) |

**Key insight**: All weights can be copied directly from GGUF raw bytes without transpose!

### `copy_tensor_to_vector()`

**Purpose**: Dequantize and copy tensor data to float vector.

```cpp
static bool copy_tensor_to_vector(
    struct ggml_tensor * tensor,
    std::vector<float> & dst
);
```

**Process**:
1. If F32, direct memcpy
2. Otherwise, use ggml backend to dequantize
3. Handle quantized formats (Q4, Q8, etc.)

### `load_snac_model()`

**Purpose**: Load SNAC model from GGUF file.

```cpp
static bool load_snac_model(
    snac_model & model,
    const char * model_path
);
```

**Supported Formats**:
1. **TTS.cpp format**: Tensors prefixed with `snac.*`
2. **snac_24khz format**: Tensors prefixed with `decoder.*` and `quantizers.*`

**Process**:
1. Initialize GGUF context
2. Read metadata (n_quantizers, decoder_dim, etc.)
3. Detect format using `is_ttscpp_format()`
4. Load tensors with appropriate handling
5. Fall back to random weights if no tensors found

### Weight Normalization Handling

SNAC uses PyTorch's weight normalization which decomposes weights into:
- `parametrizations.weight.original0` (magnitude g)
- `parametrizations.weight.original1` (direction v)

Combination formula: `w = g * (v / ||v||)`

Some models also use `_g` and `_v` suffixes directly.

**Important**: Do NOT scale weights - small weights (mean~0.000003) are correct due to weight normalization.

---

## Audio Output

### `normalize_audio()`

**Purpose**: Normalize audio samples to proper amplitude.

```cpp
static void normalize_audio(
    std::vector<float> & samples,
    float target_peak = 0.95f
);
```

**Logic**:
- If peak < 1% of target: normalize to target
- If peak < 50% of target: boost to target
- If peak > 100%: attenuate to target
- Otherwise: leave as-is

**DC Offset Removal**: After tanh activation, DC offset is removed to compensate for snake activation's inherent positive bias.

### `save_wav16()`

**Purpose**: Save audio samples as 16-bit WAV file.

```cpp
static bool save_wav16(
    const std::string & fname,
    const std::vector<float> & data,
    int sample_rate
);
```

**WAV Header Structure**:
```cpp
struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1;     // PCM
    uint16_t num_channels = 1;     // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};
```

### `decode_snac_tokens()`

**Purpose**: Decode audio tokens to PCM samples.

```cpp
static void decode_snac_tokens(
    snac_model & snac,
    const std::vector<std::vector<int>> & pyramid_tokens,
    std::vector<float> & pcm_samples
);
```

**Process**:
1. Call `snac.decode()` with pyramid tokens
2. Normalize audio with `normalize_audio()`
3. Return PCM samples

---

## Main Function Flow

### `main()`

**Flow**:

```
1. Parse arguments
   |-- Model path (-m)
   |-- Vocoder path (--model-vocoder)
   |-- Prompt text (-p)
   |-- Output path (-o)
   |-- Voice name (-v)
   |-- Sampling params (--temp, --top-k, --top-p)
   |-- Test mode (--test-vocoder)

2. Handle test vocoder mode
   |-- Load SNAC, generate random tokens, decode, save

3. Normal TTS mode
   |-- Initialize llama backend
   |-- Load LLM model
   |-- Load SNAC vocoder
   |-- Build prompt
   |-- Configure sampling (temp=0.1, top_k=40, top_p=0.9, penalty_repeat=1.1)
   |-- Generate tokens

4. Token generation loop
   |-- Sample next token
   |-- Check for stop tokens
   |-- Track audio tokens
   |-- Decode batch

5. Audio synthesis
   |-- Collect audio tokens in pyramid structure
   |-- Decode with SNAC
   |-- Normalize audio
   |-- Save WAV file

6. Cleanup
```

### Critical Sampling Parameter

```cpp
sampling_params.penalty_repeat = 1.1f;  // Prevent token repetition loops
```

This parameter is **essential** to prevent the LLM from getting stuck in token repetition loops, which causes audio degradation after initial speech.

---

## Usage

### Build

```bash
# From llama.cpp root directory
cmake -B build -DLLAMA_CURL=OFF
cmake --build build --target llama-orpheus-tts -j$(nproc)

# Executable location: build/bin/llama-orpheus-tts
```

### Model Conversion

**Orpheus LLM**:
```bash
python tools/tts/convert_hf_to_gguf_orpheus.py /path/to/orpheus-3b \
    --outfile orpheus-3b-f16.gguf --outtype f16
```

**SNAC Vocoder**:
```bash
python tools/tts/convert_hf_to_gguf_snac.py /path/to/snac_24khz \
    --outfile snac-24khz-f16.gguf --outtype f16
```

### Basic Usage

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello, how are you today?" \
    -o output.wav \
    -t 8
```

### With Voice Selection

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello, how are you today?" \
    -v tara \
    -o output.wav
```

### Test Vocoder Only

```bash
./build/bin/llama-orpheus-tts \
    --model-vocoder models/snac-24khz-f16.gguf \
    --test-vocoder \
    -o test.wav
```

### Complete Example with All Options

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "This is a test of the Orpheus text to speech system." \
    -v tara \
    -o test_output.wav \
    -t 8 \
    -n 4096 \
    --temp 0.1 \
    --top-k 40 \
    --top-p 0.9
```

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-m, --model` | Path to LLM GGUF model | Required |
| `--model-vocoder` | Path to SNAC vocoder GGUF | Required |
| `-p, --prompt` | Text to synthesize | Required |
| `-o, --output` | Output WAV file | output.wav |
| `-v, --voice` | Voice/speaker name | None |
| `-t, --threads` | Number of threads | Auto |
| `-n, --n-predict` | Max tokens to generate | 2048 |
| `--temp` | Temperature | 0.1 |
| `--top-k` | Top-k sampling | 40 |
| `--top-p` | Top-p sampling | 0.9 |
| `--test-vocoder` | Test vocoder only | False |
| `--test-frames` | Frames for vocoder test | 10 |

---

## Debugging

### Expected Output Statistics

For a working vocoder:
- `neg_ratio`: ~47-50% (not 0%)
- `mean`: Near 0 (not 0.3+)
- Values: Symmetric around 0

**Good output indicators**:
- `neg_ratio` should be 35-65% (values near 50% indicate symmetric waveform)
- `0-200 Hz` energy should be <40% (higher values indicate DC bias or noise)
- `mean` should be near 0 (values > 0.3 indicate DC bias)

### DC Bias Symptoms

If output is all positive (DC bias):
1. Check weight loading - all GGUF weights should be copied directly without transpose
2. Check codebook loading
3. Check decoder layer ConvTranspose1D weights
4. **Snake activation inherently adds positive values**: snake(x,alpha) = x + sin^2(alpha*x)/alpha adds positive for both positive and negative x
5. **Solution**: Remove DC offset in normalize_audio() function

### Snake Activation Behavior

- Formula: `snake(x, alpha) = x + sin^2(alpha*x) / alpha`
- Always adds positive values (sin^2 is always >= 0)
- For negative x: makes it less negative (closer to 0)
- For positive x: makes it more positive
- This causes DC bias to accumulate through the network
- Model was trained to compensate, but some residual DC remains
- **Fix**: Remove DC offset after tanh and before normalization

### Common Bug: Incorrect Column-Major Transpose

A common mistake is to assume GGUF stores data in column-major order and apply transpose.
This will corrupt the weights and produce:
- DC-biased output (neg_ratio ~0% instead of ~50%)
- 95%+ energy in 0-200 Hz band (should be <20%)
- Quiet, muffled audio

**Fix**: Remove all transpose operations - copy GGUF raw data directly.

### Token Repetition

If audio degrades after initial speech:
1. Check `penalty_repeat = 1.1f` is set
2. Analyze generated tokens for repetition patterns

### Frequency Distribution Note

Random tokens produce low-frequency noise (95%+ energy in 0-200 Hz band). This is expected - meaningful speech requires actual LLM-generated tokens. The chatllm.cpp reference with real text shows proper distribution: ~11% in 0-200 Hz, ~51% in 200-500 Hz, ~31% in 500-1000 Hz.

---

## Related Files

- `tools/tts/orpheus-tts.cpp` - Main implementation
- `tools/tts/convert_hf_to_gguf_snac.py` - SNAC model converter
- `tools/tts/convert_hf_to_gguf_orpheus.py` - Orpheus LLM converter

---

## Recent Fixes

1. **DC Bias Fix**: Corrected weight loading logic - GGUF stores data in row-major order, copy directly without transpose
2. **Amplitude Fix**: Added `normalize_audio()` for proper output amplitude
3. **Token Loop Fix**: Added `penalty_repeat = 1.1f` to prevent LLM token repetition
4. **ConvTranspose1D Fix**: Fixed ConvTranspose1D implementation:
   - Insert (stride-1) zeros between input elements
   - Flip kernel along kernel dimension
   - Apply regular 1D convolution with effective padding `p = kernel_size - 1 - padding`
   - This fix resolves the buzzing/noise artifacts in the audio output

---

## Commits

- `0a7a22480`: feat(tts): add audio amplitude normalization to SNAC output
- `dd6fb4899`: fix(tts): fix weight loading transpose issues in SNAC vocoder
- `76fea46a3`: fix(tts): correct SNAC model structure and weight loading
- `39eea3cd7`: fix(tts): fix ConvTranspose1D implementation for Orpheus TTS
- `e470a4148`: fix(tts): add repetition penalty to prevent token loops

---

## SNAC Vocoder Quality Investigation (2026-03-10) - RESOLVED

### Current Audio Quality Status - ALL GOOD

| Audio Type | Duration | ZCR | Status |
|------------|----------|-----|--------|
| Short audio | <2s | 0.045-0.057 | ✓ GOOD (speech-like) |
| Long audio | >10s | 0.057 | ✓ GOOD (FIXED!) |
| Target | Any | 0.02-0.06 | ✓ ACHIEVED |

### All Issues Fixed

#### 1. Residual Depthwise Conv Transpose Bug
**Commit:** `3c104c9ac fix(tts): remove unnecessary transpose in residual depthwise conv`

**Root Cause:** The residual unit depthwise convolution in `snac-ggml.cpp` had an unnecessary transpose operation that corrupted audio data.

**Impact:**
- Before fix: ZCR = 0.49 (noise-like output)
- After fix: ZCR = 0.045 (speech-like output) for short audio

**Files Modified:**
- `tools/tts/snac-ggml.cpp` - Removed incorrect transpose
- `tools/tts/snac-ggml.h` - Header updates

#### 2. Output Convolution Kernel Format Fix (NEW)
**Root Cause:** The output convolution using `ggml_im2col` required kernel format `[K, IC, 1, OC]` but was receiving incorrectly formatted data.

**Impact:**
- Before fix: Long audio ZCR = 0.155 (distorted)
- After fix: Long audio ZCR = 0.057 (good quality)

**Fix Applied:**
- Fixed kernel format to `[K, IC, 1, OC]` for ggml_im2col compatibility
- Added F32 conversion for input tensor (CUDA im2col requires F32)
- Fixed matrix multiplication order to `mul_mat(im2col_2d, kernel_2d)`

#### 3. ConvTranspose1D Kernel Type Fix (NEW)
**Root Cause:** CUDA `ggml_conv_transpose_1d` requires F32 kernel, but F16 was being passed.

**Fix Applied:**
- Changed kernel type conversion from F16 to F32
- Added kernel permutation from `[OC, K, IC]` to `[K, OC, IC]`

#### 4. Decoder Layer Depthwise Conv Fix (NEW)
**Root Cause:** Residual depthwise convolution kernels needed F16 conversion for consistency.

**Fix Applied:**
- Added F16 conversion for depthwise conv kernels in decoder layers

### Quality Metrics Guide

**Zero Crossing Rate (ZCR) Reference:**
| ZCR Range | Quality |
|-----------|---------|
| 0.02-0.06 | Excellent (clean speech) |
| 0.06-0.10 | Good |
| 0.10-0.15 | Acceptable |
| 0.15-0.30 | Distorted |
| > 0.30 | Noise/Unusable |

**Current Achievement:** ZCR 0.045-0.057 ✓ EXCELLENT

**Other Quality Indicators:**
- `neg_ratio`: Should be 35-65% (near 50% = symmetric waveform)
- `mean`: Should be near 0 (DC offset removed)
- `0-200 Hz energy`: Should be <40%

### Testing Commands

```bash
# Test vocoder with short audio
./build/bin/llama-orpheus-tts \
    --model-vocoder models/snac-24khz-f16.gguf \
    --test-vocoder --test-frames 10 \
    -o test_short.wav

# Test vocoder with longer audio
./build/bin/llama-orpheus-tts \
    --model-vocoder models/snac-24khz-f16.gguf \
    --test-vocoder --test-frames 50 \
    -o test_long.wav

# Test with full TTS pipeline
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello, this is a test of the Orpheus text to speech system with longer output." \
    -o test_tts.wav
```

### Investigation Files

For detailed investigation results, see:
- `test_output/DISTORTION_INVESTIGATION_PLAN.md` - Investigation summary (RESOLVED)
