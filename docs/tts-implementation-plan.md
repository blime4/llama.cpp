# llama.cpp TTS Implementation Plan

This document outlines the plan for adding Text-to-Speech (TTS) model support to llama.cpp.

---

## Table of Contents

1. [Current Status](#current-status)
2. [Planned TTS Models](#planned-tts-models)
3. [Implementation Guides](#implementation-guides)
4. [Adding New Model Architecture Guide](#adding-new-model-architecture-guide)
5. [Common Components](#common-components)
6. [References](#references)

---

## Current Status

### Supported Models

| Model | Status | Reference |
|-------|--------|-----------|
| **OuteTTS** | ✅ Supported | `tools/tts/tts-outetts.py` |

---

## Planned TTS Models

### Priority List

| Priority | Model | Difficulty | Status | Implementation Guide |
|----------|-------|------------|--------|---------------------|
| 1 | **Orpheus-TTS** | Medium | Planning | [docs/orpheus-tts-implementation-plan.md](./orpheus-tts-implementation-plan.md) |
| 2 | **Qwen3-TTS** | High | Planning | [docs/qwen3-tts-implementation-plan.md](./qwen3-tts-implementation-plan.md) |
| 3 | **Parler-TTS** | Medium | Future | TTS.cpp reference |
| 4 | **Kokoro** | Medium | Future | TTS.cpp reference |

---

## Implementation Guides

### 1. Orpheus-TTS (Priority 1)

**Issue**: https://github.com/ggml-org/llama.cpp/issues/12476

**Detailed Plan**: [docs/orpheus-tts-implementation-plan.md](./orpheus-tts-implementation-plan.md)

**Quick Overview**:
- 2-model system: LLM + SNAC Vocoder
- LLM: LLaMA-based (28 layers, GQA)
- Vocoder: SNAC neural codec
- Voices: 7 built-in voices (zoe, zac, jess, leo, mia, julia, leah)

#### 参考实现选择 (Updated 2026-02-27)

| 用途 | 推荐参考 | 理由 |
|------|----------|------|
| **代码实现** | chatllm.cpp | 基于 llama.cpp v3.2，架构兼容，迁移成本低 |
| **正确性验证** | TTS.cpp | 独立实现，有预构建模型，可交叉验证 |

**chatllm.cpp 优势** (作为代码参考):
- 架构完全基于 llama.cpp v3.2
- 仅 717 行实现完整功能
- 继承体系清晰 (`llama::v3_2::ConditionalGeneration`)

**TTS.cpp 优势** (作为 Golden Reference):
- 完全独立实现，可交叉验证
- 预构建模型: `mmwillet2/Orpheus_GGUF`
- 已验证可用: 成功生成音频

> **完整实现指南**: [chatllm.cpp/docs/orpheus-tts-guide.md](../chatllm.cpp/docs/orpheus-tts-guide.md)
>
> **TTS.cpp 指南**: [TTS.cpp/docs/orpheus-tts-guide.md](../TTS.cpp/docs/orpheus-tts-guide.md)

**Key Reference Files**:
- chatllm.cpp: `chatllm.cpp/models/orpheus.h/cpp` (推荐代码参考)
- TTS.cpp: `TTS.cpp/src/models/orpheus/` (Golden Reference)
- TTS.cpp SNAC: `TTS.cpp/src/decoder/snac_model.cpp`

---

### 2. Qwen3-TTS (Priority 2)

**Detailed Plan**: [docs/qwen3-tts-implementation-plan.md](./qwen3-tts-implementation-plan.md)

**Quick Overview**:
- 3-model system: Talker + Vocoder + Speaker Encoder
- Talker: Qwen3-based (32 layers)
- Vocoder: Custom neural codec (16 codebooks)
- Features: Voice cloning, voice design

**Key Reference Files**:
- chatllm.cpp: `chatllm.cpp/models/qwen_tts.h/cpp`

---

### 3. Parler-TTS (Future)

**Reference**: TTS.cpp (`TTS.cpp/src/models/parler/`)

**Architecture**: T5-based encoder-decoder

---

### 4. Kokoro (Future)

**Reference**: TTS.cpp (`TTS.cpp/src/models/kokoro/`)

**Architecture**: Flow-matching based TTS

---

## Adding New Model Architecture Guide

*Based on [llama.cpp Discussion #16770](https://github.com/ggml-org/llama.cpp/discussions/16770) and [Official HOWTO](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md)*

**Full Guide (English)**: [new-model-architecture-guide.md](./new-model-architecture-guide.md)

**完整中文指南**: [new-model-architecture-guide-cn.md](./new-model-architecture-guide-cn.md) (✨ **已重构** - 基于原文和官方文档完全重写)

This guide covers:
- Step-by-step workflow for adding new models
- GGUF conversion process
- Model architecture definition
- GGML graph implementation
- GGML vs PyTorch differences
- Debugging techniques

---

## Common Components

### SnakeBeta Activation

Used by: Qwen3-TTS, Orpheus (SNAC), Parler-TTS

```c
// Using existing ggml operations:
ggml_tensor *snake(ggml_context *ctx, ggml_tensor *input,
                   ggml_tensor *alpha) {
    auto sin_ax = ggml_sin(ctx, ggml_mul(ctx, input, alpha));
    auto sin_sq = ggml_sqr(ctx, sin_ax);
    auto reciprocal = ...;  // Precomputed
    auto term = ggml_mul(ctx, sin_sq, reciprocal);
    return ggml_add(ctx, input, term);
}
```

### Available ggml Operations

All operations needed for TTS models already exist in llama.cpp:

| Operation | Status |
|-----------|--------|
| `ggml_conv_transpose_1d` | ✅ |
| `ggml_conv_1d` | ✅ |
| `ggml_rms_norm` | ✅ |
| `ggml_rope` | ✅ |
| `ggml_silu` | ✅ |
| `ggml_sin` | ✅ |
| `ggml_sqr` | ✅ |
| `ggml_mul` | ✅ |
| `ggml_add` | ✅ |

**Conclusion**: No ggml extensions needed for any planned TTS models!

---

## Verification Strategy (正确性验证策略)

### Golden Reference 选择原则

| TTS 模型 | 代码参考 | Golden Reference | 理由 |
|----------|----------|------------------|------|
| Orpheus-TTS | chatllm.cpp | TTS.cpp | TTS.cpp 独立实现，有预构建模型 |
| Qwen3-TTS | chatllm.cpp | 官方 Python | 官方实现最权威 |
| Parler-TTS | TTS.cpp | 官方 Python | 官方实现最权威 |
| Kokoro | TTS.cpp | 官方 Python | 官方实现最权威 |

### 验证流程

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Phase V1       │ ──► │  Phase V2       │ ──► │  Phase V3       │
│  LLM Tokens     │     │  SNAC Decoder   │     │  End-to-End     │
│  验证 token 序列 │     │  验证波形输出    │     │  验证音频质量    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
         │                      │                      │
         ▼                      ▼                      ▼
   token diff = 0         波形相似度 > 99%        主观测试通过
```

### 详细计划

各模型的详细验证计划参见：
- Orpheus-TTS: [orpheus-tts-implementation-plan.md](./orpheus-tts-implementation-plan.md) 中的 "Correctness Verification Plan" 章节
- Qwen3-TTS: [qwen3-tts-implementation-plan.md](./qwen3-tts-implementation-plan.md) (待添加)

---

## Architecture Overview

### TTS Directory Structure
```
tools/tts/
├── tts.cpp                    # Main CLI
├── CMakeLists.txt             # Build config
├── README.md                  # Documentation
├── convert_pt_to_hf.py        # Model conversion utilities
├── tts-outetts.py            # OuteTTS converter
├── orpheus-tts.cpp           # (NEW) Orpheus-TTS CLI
└── convert_hf_to_gguf_*.py   # (NEW) Model converters
```

### Model Integration Pattern

```cpp
// In llama.h
enum llama_model_type {
    // ... existing ...
    LLAMA_MODEL_ORPHEUS,     // Priority 1
    LLAMA_MODEL_QWEN3_TTS,   // Priority 2
};

// In llama-model.cpp
switch (model_type) {
    case LLAMA_MODEL_ORPHEUS:
        return load_orpheus_model(ctx, ...);
    case LLAMA_MODEL_QWEN3_TTS:
        return load_qwen3_tts_model(ctx, ...);
}
```

---

## References

### External Resources
- [Adding New Model Architectures Guide #16770](https://github.com/ggml-org/llama.cpp/discussions/16770)
- [Orpheus-TTS Issue #12476](https://github.com/ggml-org/llama.cpp/issues/12476)
- [Qwen3-TTS HuggingFace](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base)
- [Qwen3-TTS Technical Report](https://arxiv.org/abs/2601.15621)
- [Orpheus Model](https://huggingface.co/canopylabs/orpheus-3b-0.1)

### Local Reference Files
| Model | Files |
|-------|-------|
| Orpheus-TTS | [orpheus-tts-implementation-plan.md](./orpheus-tts-implementation-plan.md) |
| Qwen3-TTS | [qwen3-tts-implementation-plan.md](./qwen3-tts-implementation-plan.md) |
| New Model Guide (EN) | [new-model-architecture-guide.md](./new-model-architecture-guide.md) |
| 新模型指南 (中文) | [new-model-architecture-guide-cn.md](./new-model-architecture-guide-cn.md) |
| chatllm.cpp | `chatllm.cpp/models/qwen_tts.h/cpp`, `chatllm.cpp/models/orpheus.h/cpp` |
| TTS.cpp | `TTS.cpp/src/models/orpheus/`, `TTS.cpp/src/models/kokoro/`, `TTS.cpp/src/models/parler/` |

### Documentation Guides
| Project | Guide | Description |
|---------|-------|-------------|
| TTS.cpp | [TTS.cpp/docs/orpheus-tts-guide.md](../TTS.cpp/docs/orpheus-tts-guide.md) | TTS.cpp 实现指南 (英文) |
| TTS.cpp | [TTS.cpp/docs/orpheus-tts-guide-zh.md](../TTS.cpp/docs/orpheus-tts-guide-zh.md) | TTS.cpp 实现指南 (中文) |
| chatllm.cpp | [chatllm.cpp/docs/orpheus-tts-guide.md](../chatllm.cpp/docs/orpheus-tts-guide.md) | chatllm.cpp 实现指南 (英文) |

### llama.cpp Key Files for Model Implementation
| File | Purpose |
|------|---------|
| `gguf-py/gguf/constants.py` | Model architecture constants |
| `gguf-py/gguf/tensor_mapping.py` | Tensor name mappings |
| `convert_hf_to_gguf.py` | Model conversion script |
| `include/llama.h` | Model type enums |
| `src/llama-arch.cpp` | Architecture registration |
| `src/llama-model.cpp` | Model loading |
| `src/models/` | Graph builders |
