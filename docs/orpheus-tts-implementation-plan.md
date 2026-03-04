# Orpheus-TTS Implementation Plan for llama.cpp

## Overview

Orpheus-TTS is a text-to-speech model developed by Canopy Labs. It consists of two components:
1. **LLM** - A LLaMA-based language model that generates discrete audio tokens
2. **SNAC Vocoder** - A neural audio codec that converts discrete tokens to waveform

**Issue**: https://github.com/ggml-org/llama.cpp/issues/12476

---

## Reference Implementation Analysis (Updated 2026-02-27)

### TTS.cpp vs chatllm.cpp 对比

| 维度 | TTS.cpp | chatllm.cpp | 评估 |
|------|---------|-------------|------|
| **代码量** | 865行 C++ + 11K Python | 717行 (h + cpp + converter) | chatllm.cpp 更精简 |
| **架构基础** | 独立实现 | 基于 llama.cpp v3.2 | chatllm.cpp 兼容性更好 |
| **SNAC 实现** | 完整 (snac_model.cpp + general_neural_audio_codec.cpp) | 完整 (orpheus.cpp 内) | 两��相当 |
| **LLM 实现** | 独立 model.cpp | 继承 llama::v3_2::ConditionalGeneration | chatllm.cpp 复用度高 |
| **模型格式** | GGUF (单一文件) | 自定义 .bin (单一文件) | TTS.cpp 更标准 |
| **预构建模型** | ✅ `mmwillet2/Orpheus_GGUF` | ❌ 需自行转换 | TTS.cpp 有优势 |
| **迁移难度** | 中等 (需适配接口) | 低 (架构已兼容) | chatllm.cpp 更易迁移 |
| **独立验证** | ✅ 完全独立实现 | ❌ 基于 llama.cpp | TTS.cpp 可作 golden |

### 推荐方案

| 用途 | 推荐参考 | 理由 |
|------|----------|------|
| **代码实现参考** | **chatllm.cpp** | 架构完全基于 llama.cpp v3.2，继承体系清晰，迁移成本最低 |
| **正确性验证 (Golden)** | **TTS.cpp** | 独立实现，有预构建模型，可作为交叉验证的基准 |

### 具体分析

#### chatllm.cpp 优势 (作为代码参考)
1. **架构兼容**: 完全基于 llama.cpp v3.2，继承 `llama::v3_2::ConditionalGeneration`
2. **���码精简**: 仅 717 行实现完整功能
3. **复用度高**: Tokenizer、推理引擎、内存管理全部复用
4. **清晰继承**: SNAC 和 TTS 分别在 `chatllm::orpheus::snac` 和 `chatllm::orpheus::tts` 命名空间

#### TTS.cpp 优势 (作为 Golden Reference)
1. **独立实现**: 不依赖 llama.cpp，可交叉验证正确性
2. **预构建模型**: `mmwillet2/Orpheus_GGUF` 可直接使用
3. **已验证可用**: 已成功生成音频 (`test_orpheus_ttscpp_final.wav`)
4. **GGUF 标准**: 使用标准 GGUF 格式

---

## Correctness Verification Plan (正确性验证计划)

### 验证策略

使用 **TTS.cpp** 作为 Golden Reference，分阶段验证 llama.cpp 实现的正确性。

### Phase V1: LLM Token Generation 验证

**目标**: 验证 LLM 生成的 audio tokens 与 TTS.cpp 一致

**步骤**:
1. 使用相同的输入文本和 voice
2. 使用相同的随机种子 (temperature=0, top_k=1)
3. 对比生成的 token 序列

**验证脚本**:
```bash
# TTS.cpp (Golden)
./tts-cli --model-path models/Orpheus_prebuilt.gguf \
    --prompt "Hello world" --voice leah \
    --temperature 0 --save-path golden.wav \
    --log-tokens golden_tokens.txt

# llama.cpp (待验证)
./orpheus-tts -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello world" --voice leah \
    --temp 0 -o test.wav \
    --log-tokens test_tokens.txt

# 对比
diff golden_tokens.txt test_tokens.txt
```

**预期**: 相同输入下 token 序列完全一致

### Phase V2: SNAC Decoder 验证

**目标**: 验证 SNAC 解码器输出与 TTS.cpp 一致

**步骤**:
1. 使用相同的 audio token 序列
2. 分别用 TTS.cpp 和 llama.cpp 的 SNAC 解码
3. 对比输出波形

**验证脚本**:
```bash
# 使用预录制的 token 序列
python3 tools/tts/verify_snac_decoder.py \
    --tokens test_tokens.txt \
    --tts-cpp-model models/Orpheus_prebuilt.gguf \
    --llama-snac models/snac-24khz-f16.gguf \
    --output-comparison comparison.png
```

**验证指标**:
- 波形相似度 > 99.9%
- 峰值信噪比 (PSNR) > 100 dB
- 最大绝对误差 < 1e-6

### Phase V3: End-to-End 验证

**目标**: 验证完整 TTS 流程输出的音频质量

**测试集**:
| ID | 文本 | Voice | 预期时长 |
|----|------|-------|----------|
| 1 | "Hello world" | leah | ~1s |
| 2 | "This is a longer sentence to test continuous generation." | zoe | ~4s |
| 3 | "你好，这是中文测试。" | leo | ~2s |

**验证方法**:
```bash
for test in test_cases; do
    # 生成音频
    ./orpheus-tts ... -p "$text" --voice $voice -o test_$id.wav
    ./tts-cli ... -p "$text" --voice $voice --save-path golden_$id.wav

    # 计算相似度
    python3 tools/tts/compare_audio.py \
        golden_$id.wav test_$id.wav \
        --output metrics_$id.json
done
```

**通过标准**:
- 所有测试用例音频可播放
- 波形相似度 > 99%
- 无明显杂音或失真

### Phase V4: 性能基准

**目标**: 确保性能不低于 TTS.cpp

**基准测试**:
```bash
# 测试不同长度文本的生成时间
python3 tools/tts/benchmark.py \
    --llama-cpp ./orpheus-tts \
    --tts-cpp ./tts-cli \
    --model-llama models/orpheus-3b-f16.gguf \
    --model-tts models/Orpheus_prebuilt.gguf \
    --output benchmark_results.json
```

**通过标准**:
- 首个 token 延迟 <= TTS.cpp
- 总生成时间 <= TTS.cpp * 1.1
- 内存占用 <= TTS.cpp * 1.2

---

## Architecture Design (Updated 2026-02-25)

### Separate Model Files

The LLM and SNAC vocoder are stored in separate GGUF files for modularity:

```
models/orpheus-tts/orpheus-3b-f16.gguf   # LLM model
models/snac/snac-24khz-f16.gguf          # SNAC vocoder (from hubertsiuzdak/snac_24khz)
```

### Usage

```bash
./orpheus-tts \
  -m models/orpheus-tts/orpheus-3b-f16.gguf \
  --model-vocoder models/snac/snac-24khz-f16.gguf \
  -p "Hello, how are you?" \
  -o output.wav
```

### Rationale for Separate Models

1. **Modularity**: LLM and vocoder can be updated independently
2. **Memory**: Load vocoder only when needed
3. **Reusability**: Same vocoder can be used with different TTS models
4. **Clarity**: Cleaner separation of concerns

---

## Architecture

### 1. LLM (Language Model)

Based on LLaMA-3 architecture with custom head for audio token prediction.

**Key Parameters** (from TTS.cpp):
```cpp
struct orpheus_model {
    uint32_t vocab_size = 156940;
    uint32_t n_attn_heads = 24;
    uint32_t n_kv_attn_heads = 8;
    uint32_t head_size = 128;
    uint32_t max_context_length = 1024;
    uint32_t max_generation_size = 2100;  // 25.6 seconds
    uint32_t hidden_size = 3072;
    uint32_t kv_hidden_size = 1024;
    uint32_t n_layers = 28;
    uint32_t stopping_token_id = 128258;
    uint32_t eos_token_id = 128001;
    uint32_t bos_token_id = 128000;
};
```

**Structure**:
- Embedding layer
- 28 Transformer layers (GQA: 24 heads, 8 KV heads)
- Output norm
- LM head (predicts audio tokens)

### 2. SNAC Vocoder (snac_24khz)

Source: https://huggingface.co/hubertsiuzdak/snac_24khz

**Key Parameters** (from config.json):
```json
{
    "sampling_rate": 24000,
    "decoder_dim": 1536,
    "decoder_rates": [8, 8, 3, 2],
    "codebook_size": 4096,
    "codebook_dim": 8,
    "vq_strides": [8, 4, 2, 1],
    "noise": true,
    "depthwise": true
}
```

**Structure**:
- Input convolution (depthwise, kernel=7)
- Up convolution (1x1 projection to decoder_dim=1536)
- Attention layer (optional, for long sequences)
- 4 Decoder blocks (ConvTranspose1d + SnakeBeta + ResidualUnits)
- Output convolution (SnakeBeta + Conv1d)
- 4 Quantizers with codebook

**Tensor Naming (PyTorch)**:
```
decoder.model.0.*                 # Input conv (depthwise)
decoder.model.1.*                 # Up conv (1x1)
decoder.model.2.*                 # Attention layer
decoder.model.{3,4,5,6}.block.*   # Decoder layers
decoder.model.7.alpha             # Snake alpha
decoder.model.8.*                 # Output conv
quantizer.quantizers.{0,1,2,3}.*  # 4 Quantizers
```

**Weight Normalization**:
- Uses `parametrizations.weight.original0` and `original1`
- weight = original0 * original1 (broadcast multiply)

---

## Special Tokens

### Voices
```cpp
static constexpr std::array<const char *, 7> orpheus_voices{
    "zoe", "zac", "jess", "leo", "mia", "julia", "leah"
};
```

### Control Tokens
```cpp
// Prepended to input
static constexpr std::array<uint32_t, 2> orpheus_prepended_tokens = { 128259, 128000 };

// Appended to input
static constexpr std::array<uint32_t, 4> orpheus_appended_tokens = { 128009, 128260, 128261, 128257 };
```

---

## Reference Implementations

### TTS.cpp (`TTS.cpp/src/models/orpheus/`)
- `model.cpp` - LLM implementation
- `loader.cpp` - Model loading

### chatllm.cpp (`chatllm.cpp/models/orpheus.cpp`)
- `orpheus.h` - Header with class definitions
- `orpheus.cpp` - Full implementation including SNAC

---

## Implementation Steps

### Phase 1: GGUF Conversion Scripts

**Files**:
- `tools/tts/convert_hf_to_gguf_orpheus.py` - LLM conversion only
- `tools/tts/convert_hf_to_gguf_snac.py` - SNAC vocoder conversion (NEW)

#### 1.1 LLM Conversion
```python
# Tensor mapping
model.embed_tokens.weight     → token_embd.weight
model.norm.weight             → output_norm.weight
model.lm_head.weight          → output.weight
model.layers.{i}.*           → blk.{i}.*

# GGUF metadata
orpheus.vocab_size = 156940
orpheus.attn_heads = 24
orpheus.kv_attn_heads = 8
orpheus.head_dim = 128
orpheus.hidden_size = 3072
orpheus.kv_hidden_size = 1024
orpheus.layers = 28
orpheus.stopping_token_id = 128258
tokenizer.ggml.bos_token_id = 128000
tokenizer.ggml.eos_token_id = 128001
```

#### 1.2 SNAC Vocoder Conversion (NEW - separate file)
```python
# File: tools/tts/convert_hf_to_gguf_snac.py
# Source: https://huggingface.co/hubertsiuzdak/snac_24khz

# Tensor mapping (PyTorch -> GGUF)
decoder.model.0.*                 → decoder.in_conv.*
decoder.model.1.*                 → decoder.up_conv.*
decoder.model.2.*                 → decoder.attn.*
decoder.model.{3,4,5,6}.block.*   → decoder.layers.{0,1,2,3}.*
decoder.model.7.alpha             → decoder.alpha_out
decoder.model.8.*                 → decoder.out_conv.*
quantizer.quantizers.{0-3}.*      → quantizers.{0-3}.*

# Weight normalization: original0 * original1 -> combined weight

# SNAC metadata
snac.sampling_rate = 24000
snac.decoder_dim = 1536
snac.decoder_rates = [8, 8, 3, 2]
snac.codebook_size = 4096
snac.codebook_dim = 8
snac.vq_strides = [8, 4, 2, 1]
snac.n_quantizers = 4
snac.noise = true
snac.depthwise = true
```

---

### Phase 2: Model Architecture Registration

**Files**: `include/llama.h`, `src/llama-arch.cpp`

#### 2.1 Add Model Type
```c
// In llama.h
LLAMA_MODEL_ORPHEUS,
```

#### 2.2 Register Architecture
```c
// In llama-arch.cpp
{ LLM_ARCH_ORPHEUS, "orpheus" }

// LLM tensor names: token_embd.*, blk.{i}.*, output_norm.*, output.*
// SNAC loaded separately via --model-vocoder
```

---

### Phase 3: Model Loading

**File**: `src/llama-model.cpp`

#### 3.1 LLM Tensors (reuse LLaMA-3 patterns)
```c
if (arch == LLM_ARCH_ORPHEUS) {
    model.tok_embd = create_tensor("token_embd.weight", {n_embd, n_vocab});
    model.output_norm = create_tensor("output_norm.weight", {n_embd});
    model.output = create_tensor("output.weight", {n_vocab, n_embd});

    for (int i = 0; i < n_layers; ++i) {
        // GQA attention: 24 heads, 8 KV heads, head_dim=128
        // Standard attention + MLP tensors
    }
}
```

#### 3.2 SNAC Tensors (separate file, loaded in orpheus-tts.cpp)
```c
// Loaded via --model-vocoder option in orpheus-tts.cpp
// Not part of llama.cpp core model loading
```

---

### Phase 4: Compute Graph Implementation

**File**: `src/llama-graph.cpp`

#### 4.1 LLM Forward Pass (reuse LLaMA-3 patterns)
```
Input: text_tokens + voice_tokens
       position_ids

1. embd = embedding(tokens)
2. For each layer (0..27):
   - RMSNorm
   - GQA Self-Attention (RoPE, KV cache)
   - SwiGLU FFN
3. output_norm
4. lm_head → logits

Output: audio_token_ids
```

#### 4.2 SNAC Decoder Forward Pass
```
Input: audio_token_ids [seq_len]

1. token_emb = embedding(token_ids)  [seq_len, codebook_dim]
2. Repeat interleaved based on codebook structure
3. in_conv: [seq_len, codebook_dim] → [seq_len, embd]
4. For each decoder block (0..3):
   - SnakeBeta activation
   - ConvTranspose1d (upsample by factor)
   - Residual units with dilated convs
5. out_conv → [samples, 1]
6. Tanh activation

Output: waveform [samples]
```

#### 4.3 SnakeBeta Implementation
```c
// From chatllm.cpp - uses existing ggml ops!
ggml_tensor *snake(ggml_context *ctx, ggml_tensor *input, 
                   ggml_tensor *alpha) {
    // snake(x) = x + sin²(αx) * (1/α)
    auto ax = ggml_mul(ctx, input, alpha);
    auto sin_ax = ggml_sin(ctx, ax);
    auto sin_sq = ggml_sqr(ctx, sin_ax);
    // Compute 1/α for reciprocal
    auto reciprocal = ...;  // Precomputed in model loading
    auto term = ggml_mul(ctx, sin_sq, reciprocal);
    return ggml_add(ctx, input, term);
}
```

---

### Phase 5: Inference Pipeline

**File**: `tools/tts/orpheus-tts.cpp`

#### 5.1 End-to-End Pipeline
```cpp
// 1. Load LLM model
llama_model *model = llama_load_model_from_file("orpheus-3b-f16.gguf");

// 2. Load SNAC vocoder (separate file)
snac_model snac;
load_snac_model_from_gguf(snac, "snac-24khz-f16.gguf");

// 3. Prepare input
std::vector<llama_token> tokens = prepare_input(text, voice);

// 4. Autoregressive generation
std::vector<uint32_t> audio_tokens;
while (!eos) {
    auto logits = llm_forward(tokens);
    auto token = sample(logits);
    audio_tokens.push_back(token);
    tokens.push_back(token);
}

// 5. Decode with SNAC
std::vector<float> waveform = snac_decode(snac, audio_tokens);

// 6. Save WAV
save_wav("output.wav", waveform, 24000);
```

#### 5.2 Command Line Interface
```cpp
// Usage:
// ./orpheus-tts -m orpheus-3b-f16.gguf --model-vocoder snac-24khz-f16.gguf -p "text" -o output.wav

// CLI arguments:
//   -m, --model PATH          LLM model path (required)
//   --model-vocoder PATH      SNAC vocoder path (required)
//   -p, --prompt TEXT         Text to synthesize (required)
//   -o, --output PATH         Output WAV file (default: output.wav)
//   -v, --voice NAME          Voice name (zoe, zac, jess, leo, mia, julia, leah)
//   -t, --threads N           Number of threads
//   -n, --n-predict N         Max tokens to generate
//   --temp N                  Temperature (default: 0.1)
//   --top-k N                 Top-k sampling (default: 40)
//   --top-p N                 Top-p sampling (default: 0.9)
```

#### 5.3 Voice Selection
```cpp
// Voices: zoe, zac, jess, leo, mia, julia, leah
// Each voice has predefined token embeddings
```

---

## File Changes Summary

| File | Action | Description |
|------|--------|-------------|
| `tools/tts/convert_hf_to_gguf_orpheus.py` | MODIFY | LLM conversion only (remove SNAC) |
| `tools/tts/convert_hf_to_gguf_snac.py` | NEW | SNAC vocoder conversion (snac_24khz) |
| `include/llama.h` | MODIFY | Add ORPHEUS model type |
| `src/llama-arch.cpp` | MODIFY | Register architecture |
| `src/llama-hparams.h` | MODIFY | Add Orpheus hyperparameters |
| `src/llama-model.h` | MODIFY | Add tensor fields |
| `src/llama-model.cpp` | MODIFY | Tensor loading |
| `src/llama-graph.cpp` | MODIFY | Compute graph |
| `tools/tts/orpheus-tts.cpp` | MODIFY | Add --model-vocoder option |

---

## Implementation Order

1. **GGUF Converter** (Day 1-2)
   - Write conversion script
   - Convert LLM + SNAC models
   - Verify tensor shapes

2. **Architecture Registration** (Day 2-3)
   - Add enums
   - Register tensor names

3. **LLM Implementation** (Day 3-5)
   - Load tensors (reuse LLaMA-3 patterns)
   - Build compute graph
   - Test token generation

4. **SNAC Decoder** (Day 5-7)
   - Load SNAC tensors
   - Implement SnakeBeta
   - Build decoder graph

5. **Integration** (Day 7-10)
   - Connect LLM → SNAC
   - WAV output
   - Voice selection
   - Testing

---

## Key Technical Details

### GQA Configuration
- Attention heads: 24
- KV heads: 8
- Head dim: 128
- This is different from standard LLaMA-3 (32 heads, 8 KV heads)

### SNAC Pyramid Structure
- 4 layers with different upsampling rates
- Uses repeat_interleave for token expansion
- SnakeBeta activation in each block

### ggml Operations Needed
All existing in llama.cpp:
- `ggml_conv_transpose_1d` ✅
- `ggml_conv_1d` ✅
- `ggml_rms_norm` ✅
- `ggml_rope` ✅
- `ggml_silu` ✅
- `ggml_sin` ✅
- `ggml_sqr` ✅
- `ggml_mul` ✅
- `ggml_add` ✅

---

## References

- GitHub Issue: https://github.com/ggml-org/llama.cpp/issues/12476
- TTS.cpp: `TTS.cpp/src/models/orpheus/`
- chatllm.cpp: `chatllm.cpp/models/orpheus.h/cpp`
- Orpheus Model: https://huggingface.co/canopylabs/orpheus-3b-0.1
