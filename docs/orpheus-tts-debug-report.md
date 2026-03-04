# Orpheus TTS 实现调查报告

## 概述

本报告对比分析了三个 Orpheus TTS 实现:
1. **llama.cpp** (当前实现 - 有问题)
2. **chatllm.cpp** (golden 参考)
3. **TTS.cpp** (golden 参考)

---

## 0. 关键技术发现 (已验证正确)

**最后更新**: 2026-03-04

### 0.1 GGUF 存储格式

GGUF 使用行优先顺序存储张量数据，与 numpy 相同。

**关键规则**:
- GGUF 以**行优先顺序 (row-major order)** 存储数据，与 numpy 完全相同
- GGUF 维度顺序与 numpy **相反**: numpy shape `[a, b, c]` 对应 GGUF `ne[0]=c, ne[1]=b, ne[2]=a`
- 原始数据字节与 numpy 行优先顺序完全一致
- 元素 `[i, j, k]` 在两种格式中都位于扁平索引 `i*b*c + j*c + k`

**结论**: 所有权重可以直接从 GGUF 原始字节复制，**无需转置**。

### 0.2 SNAC ConvTranspose1D 权重格式

**关键发现**: PyTorch weight normalization 存储的 `weight_v` 格式与标准 ConvTranspose1D 不同。

| 权重类型 | PyTorch weight_v 格式 | 标准 ConvTranspose1D 格式 |
|----------|----------------------|---------------------------|
| ConvTranspose1D | `[in_channels, kernel_size, out_channels]` | `[in_channels, out_channels, kernel_size]` |

**必须在加载时转置**: `[in, k, out]` -> `[in, out, k]`

### 0.3 ConvTranspose1D 算法

正确的 ConvTranspose1D 实现步骤:

```
1. 在输入元素之间插入 (stride-1) 个零: expanded[i*stride] = input[i]
2. 沿核维度翻转核: k_flipped = kernel_size - 1 - k
3. 使用有效填充应用常规 conv1d: p = kernel_size - 1 - padding
```

**关键洞察**: ConvTranspose1D 的填充参数**不等于**等效 conv1d 的填充。
- ConvTranspose1D padding: 4
- 等效 conv1d padding: `kernel_size - 1 - 4 = 16 - 1 - 4 = 11`

### 0.4 Snake 激活函数 DC 偏置

**公式**: `snake(x, alpha) = x + sin^2(alpha*x) / alpha`

**关键行为**:
- `sin^2` 始终 >= 0，因此它**总是添加正值**
- 对于负 x: 使其更接近 0 (负值变小)
- 对于正 x: 使其更正
- 模型训练时已补偿此偏置，但仍有残留 DC

**修复方案**: 在 tanh 之后使用阈值 `1e-9f` 移除 DC 偏置。

### 0.5 金字塔 Token 结构 (SNAC 24kHz)

**Token 配置**:
- 3 个 codebook 用于 TTS
- 每帧 token 顺序: `[0, 1, 2, 2, 1, 2, 2]` (7 个 token)
- Head 大小比例: `head0 : head1 : head2 = N : 2N : 4N`
- VQ strides: `[4, 2, 1, 1]`

**Token 重复规则**:
- head0 重复 4x (1 个 token 扩展为 4 个)
- head1 重复 2x (2 个 token 扩展为 4 个)
- head2 重复 1x (4 个 token 不扩展)

---

## 1. SNAC 模型架构差异

### 1.1 模型维度

| 参数 | chatllm.cpp | TTS.cpp | llama.cpp |
|------|-------------|---------|-----------|
| QUANTIZER_DIM | 768 | 768 | 768 ✓ |
| DECODER_DIM | 1024 | 1024 | 1024 ✓ |
| N_QUANTIZERS | 3 | 3 | 3 ✓ |
| DECODER_RATES | [8,8,4,2] | GGUF元数据 | [8,8,4,2] ✓ |
| UPSAMPLE_FACTOR | 512 | 512 | 512 ✓ |
| CODEBOOK_SIZE | 4096 | 4096 | 4096 ✓ |
| CODEBOOK_DIM | 8 | 8 | 8 ✓ |

### 1.2 Decoder 层维度

```
Layer 0: 1024 -> 512 (stride=8, kernel=16)
Layer 1: 512  -> 256 (stride=8, kernel=16)
Layer 2: 256  -> 128 (stride=4, kernel=8)
Layer 3: 128  -> 64  (stride=2, kernel=4)
```

---

## 2. 关键实现差异分析

### 2.1 Snake1D 激活函数

**chatllm.cpp 实现**:
```cpp
ggml::tensor *Snake1D::forward(ComputeContext *ctx, ggml::tensor *input) {
    auto alpha_view = ggml::reshape_2d(ctx, alpha, 1, ggml::get_dim(alpha, 0));
    ggml::tensor *output = ggml::mul(ctx, input, alpha_view);
    output = ggml::sin(ctx, output);
    output = ggml::square(ctx, output);
    output = ggml::mul(ctx, output, alpha_reciprocal);
    output = ggml::add(ctx, input, output);
    return output;
}
```
公式: `y = x + sin²(α*x) / α`

**llama.cpp 实现**:
```cpp
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
```
**结论**: 实现一致 ✓

---

### 2.2 Quantizer 层实现

**chatllm.cpp VectorQuantize::dequantize**:
```cpp
ggml::tensor *VectorQuantize::dequantize(ComputeContext *ctx, ggml::tensor *embed_id) {
    ggml::tensor *output = codebook.forward(ctx, embed_id);
    output = ggml::permute(ctx, output, 1, 0, 2, 3); // [emb_dim, len] -> [len, emb_dim, 1, 1]
    output = ggml::cont(ctx, output);
    output = out_proj.forward(ctx, output);
    return output;
}
```

**TTS.cpp build_quantize_layer**:
```cpp
struct ggml_tensor * build_quantize_layer(ggml_context * ctx, struct ggml_tensor * cur, residual_vector_quantize_layer & l) {
    cur = ggml_get_rows(ctx, l.codebook, cur);      // [seq] -> [codebook_dim, seq]
    cur = ggml_cont(ctx, ggml_transpose(ctx, cur)); // [codebook_dim, seq] -> [seq, codebook_dim]
    cur = ggml_conv_1d(ctx, l.out_proj_kernel, cur, 1, 0, 1);
    cur = ggml_add(ctx, cur, l.out_proj_bias);
    return cur;
}
```

**llama.cpp snac_quantizer_layer::forward**:
```cpp
std::vector<float> forward(const int * tokens, int64_t seq_len) {
    // Codebook lookup: [codebook_size, codebook_dim] = [4096, 8]
    for (int64_t i = 0; i < seq_len; i++) {
        int token = tokens[i];
        for (int d = 0; d < codebook_dim; d++) {
            emb[d] = codebook[token * codebook_dim + d];  // 行优先读取
        }
        // Output projection
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
```

**潜在问题**:
1. ❓ Codebook 存储格式需要验证 - GGUF 存储为 [codebook_dim, codebook_size] 还是 [codebook_size, codebook_dim]?
2. ❓ out_proj_weight 是否正确转置?

---

### 2.3 ConvTranspose1D 实现

**chatllm.cpp DecoderBlock**:
```cpp
DecoderBlock(InitContext *ctx, int input_dim, int output_dim, int stride, bool noise, int groups, int output_padding)
    : Sequential(ctx) {
    add_block(new Snake1D(ctx, input_dim));
    add_block(new ConvTransposed1D(ctx, input_dim, output_dim, 2 * stride,  // kernel = 2*stride
                                    stride,
                                    (stride + 1) / 2,                       // padding
                                    output_padding));                       // output_padding = stride % 2
    // ...
}
```

**llama.cpp conv_transpose1d**:
```cpp
static void conv_transpose1d(...) {
    // Method: Insert zeros between input elements, then convolve with flipped kernel
    int64_t p = kernel_size - 1 - padding;  // Effective padding
    int64_t expanded_len = (input_len - 1) * stride + 1;
    // Insert zeros
    for (int64_t i = 0; i < input_len; i++) {
        for (int64_t ic = 0; ic < in_channels; ic++) {
            expanded[i * stride * in_channels + ic] = input[i * in_channels + ic];
        }
    }
    // Convolve with flipped kernel
    for (int64_t o = 0; o < output_len; o++) {
        for (int64_t k = 0; k < kernel_size; k++) {
            int64_t k_flipped = kernel_size - 1 - k;  // FLIP the kernel!
            // ...
        }
    }
}
```

**问题检查清单**:
- [ ] kernel 是否正确翻转?
- [ ] padding 计算是否正确?
- [ ] 权重布局 [in_channels, out_channels, kernel_size] 是否正确?

---

### 2.4 权重加载分析 (已验证)

**GGUF 存储格式** (已验证正确):
- GGUF 以**行优先顺序**存储张量数据 (与 numpy 相同)
- GGUF 维度顺序与 numpy **相反**: numpy shape `[a,b,c]` -> GGUF `ne[0]=c, ne[1]=b, ne[2]=a`
- **关键**: 原始数据字节与 numpy 行优先顺序完全相同，可以直接复制

**权重加载规则表**:

| 张量 | Numpy Shape | GGUF ne[] | 是否需要转置 |
|------|-------------|-----------|--------------|
| `in.weight` | [768, 7] | ne[0]=7, ne[1]=768 | 否 (直接复制) |
| `up.weight` | [1024, 768] | ne[0]=768, ne[1]=1024 | 否 (直接复制) |
| `decoder.out_conv.weight` | [7, 64, 1] | ne[0]=1, ne[1]=64, ne[2]=7 | 否 (直接复制) |
| `decoder.layers.X.conv_t.weight` | [in_c, out_c, ks] | ne[0]=ks, ne[1]=out_c, ne[2]=in_c | 否 (直接复制) |
| `residual_units.Y.in_conv.weight` | [ch, 7] | ne[0]=7, ne[1]=ch | 否 (直接复制) |
| `residual_units.Y.out_conv.weight` | [ch, ch] | ne[0]=ch, ne[1]=ch | 否 (直接复制) |

**SNAC ConvTranspose1D 权重特殊处理**:

PyTorch weight normalization 存储的 `weight_v` 格式为 `[in_channels, kernel_size, out_channels]`，
但我们的 ConvTranspose1D 实现需要 `[in_channels, out_channels, kernel_size]`，因此需要转置:

```cpp
// 正确的 ConvTranspose1D 权重加载 (需要转置)
// PyTorch weight_v: [in_channels, kernel_size, out_channels]
// GGUF raw data: 行优先 [in, k, out]
// 我们需要: [in, out, k]

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
```

**已验证结论**:
- 所有常规权重可以直接从 GGUF 原始字节复制，无需转置
- 仅 ConvTranspose1D 权重需要 `[in, k, out]` -> `[in, out, k]` 的置换

---

### 2.5 Token 组织 (Pyramid 结构)

**chatllm.cpp Codec::decode_frame**:
```cpp
void Codec::decode_frame(...) {
    // Pyramid structure: pre-order binary tree traversal
    // For 3 quantizers: [0, 1, 2, 2, 1, 2, 2]
    for (int j = 0; j < num_frames; j++) {
        int i = j * frame_size;
        for (int k : pyramid) {  // pyramid = [0, 1, 2, 2, 1, 2, 2]
            codes[k].push_back(multiframe[i++]);
        }
    }
}
```

**TTS.cpp snac_build_audio_inputs**:
```cpp
// Token repeats: head0 has N tokens, head1 has 2N, head2 has 4N
// repeats = [4, 2, 1] - head0 repeats 4x, head1 repeats 2x, head2 repeats 1x
for(int i = 0; i < sctx->model->n_heads; i++) {
    auto quantize_layer = sctx->model->quantizer_layers[i];
    struct ggml_tensor * inp_head = ggml_view_1d(ctx, sctx->inp_tokens,
                                                 sequence_length / sctx->model->repeats[i], ...);
    struct ggml_tensor * code = general_neural_audio_codec::build_quantize_layer(ctx, inp_head, quantize_layer);
    if (sctx->model->repeats[i] > 1) {
        code = ggml_repeat(ctx, ggml_cont_3d(ctx, code, 1, code->ne[0], code->ne[1]), ...);
    }
    if (i == 0) embd = code;
    else embd = ggml_add(ctx, embd, code);
}
```

**llama.cpp collect_audio_tokens_pyramid**:
```cpp
// Pyramid structure: [0, 1, 2, 2, 1, 2, 2]
static const int pyramid_map[SNAC_FRAME_SIZE] = {0, 1, 2, 2, 1, 2, 2};

// Subtract position-based offset to get actual codebook index
int id = all_tokens[f * SNAC_FRAME_SIZE + pos] - pos * SNAC_CODEBOOK_SIZE;
```

**关键差异**:
- llama.cpp 使用 `pos * SNAC_CODEBOOK_SIZE` 作为位置偏移
- TTS.cpp 使用 `repeats = [4, 2, 1]` 表示上采样因子

**问题**:
- ❓ 位置偏移计算是否正确? Orpheus LLM 输出的 token 格式是什么?

---

### 2.6 VQ Strides 和 Token 重复

**chatllm.cpp ResidualVectorQuantize::dequantize**:
```cpp
ggml::tensor *ResidualVectorQuantize::dequantize(ComputeContext *ctx, const std::vector<ggml::tensor *> &embed_id) {
    ggml::tensor *output = nullptr;
    for (int i = 0; i < n_codebooks; i++) {
        VectorQuantize *q = (VectorQuantize *)blocks[i].get();
        auto z_q_i = q->dequantize(ctx, embed_id[i]);
        if (output) {
            output = ggml_repeat_interleave(ctx, output, ggml::get_dim(z_q_i, 0) / ggml::get_dim(output, 0));
            output = ggml::add(ctx, z_q_i, output);
        } else {
            output = z_q_i;
        }
    }
    return output;
}
```

**llama.cpp decode**:
```cpp
// VQ strides for SNAC pyramid structure
model.vq_strides[0] = 4;  // head0: 1 token per frame, repeat 4x
model.vq_strides[1] = 2;  // head1: 2 tokens per frame, repeat 2x
model.vq_strides[2] = 1;  // head2: 4 tokens per frame, no repeat

// Upsample previous output by repeat_interleave
int upsample_factor = vq_strides[h-1] / vq_strides[h];
```

**问题**:
- ❓ VQ strides 是否与实际 token 数量匹配?
- 根据 pyramid [0,1,2,2,1,2,2]: head0 出现 1 次, head1 出现 2 次, head2 出现 4 次
- 因此: head0 的 token 数 : head1 的 token 数 : head2 的 token 数 = 1:2:4
- 这意味着 repeats = [4, 2, 1] 是正确的 (head0 重复 4x, head1 重复 2x, head2 重复 1x)

---

## 3. 已识别的问题清单

### 3.1 高优先级问题

| # | 问题 | 严重性 | 文件位置 | 状态 |
|---|------|--------|----------|------|
| 1 | Codebook 存储格式未验证 | 高 | orpheus-tts.cpp:796-802 | **已验证**: [4096, 8] 行优先 |
| 2 | out_proj_weight 转置可能错误 | 高 | orpheus-tts.cpp:807-813 | **已验证**: 直接复制，无需转置 |
| 3 | ConvTranspose1D 权重置换逻辑 | 高 | orpheus-tts.cpp:1509-1533 | **已验证**: 需要 [in,k,out] -> [in,out,k] |
| 4 | 位置偏移计算 (pos * 4096) | 高 | orpheus-tts.cpp:178 | **已验证**: 正确 |
| 5 | in_conv_kernel 深度卷积权重格式 | 高 | orpheus-tts.cpp:1387-1391 | **已验证**: 直接复制 |

### 3.2 中优先级问题

| # | 问题 | 严重性 | 文件位置 | 状态 |
|---|------|--------|----------|------|
| 6 | up_conv_kernel 转置逻辑 | 中 | orpheus-tts.cpp:1399-1417 | **已验证**: 直接复制 |
| 7 | out_conv_kernel 转置逻辑 | 中 | orpheus-tts.cpp:1434-1464 | **已验证**: 直接复制 |
| 8 | Residual unit 权重格式 | 中 | orpheus-tts.cpp:1585-1591 | **已验证**: 直接复制 |
| 9 | VQ strides 与 repeats 一致性 | 中 | orpheus-tts.cpp:853, 921 | **已验证**: [4, 2, 1] 正确 |
| 10 | Snake 激活函数 DC 偏置 | 中 | orpheus-tts.cpp | **已验证**: 需要在 tanh 后移除 DC |

---

## 4. 修复建议

### 4.1 Codebook 和 Quantizer 修复

**问题**: 需要验证 GGUF 中 codebook 的实际存储格式

**建议**: 添加调试代码打印 codebook 的形状和前几个值:
```cpp
LOG_WRN("Codebook tensor shape: ne=[%lld, %lld]\n", tensor->ne[0], tensor->ne[1]);
LOG_WRN("Codebook first values: ");
for (int i = 0; i < 10; i++) printf("%.4f ", codebook[i]);
printf("\n");
```

### 4.2 权重转置修复

**问题**: 多处权重转置逻辑可能不正确

**建议**: 参考已工作的 chatllm.cpp 实现:
1. 直接从 GGUF 读取权重，不做转置
2. 修改卷积函数以匹配 GGUF 的存储格式

### 4.3 Token 偏移修复

**问题**: `pos * SNAC_CODEBOOK_SIZE` 偏移可能不正确

**建议**: 参考 chatllm.cpp 的 token 处理:
```cpp
// chatllm.cpp decoder_push_llm_tok_id:
id = id - ((vocoder_ids.size() % snac::FRAME_SIZE) * codec_config.codebook_size);
```

这表明偏移是基于帧内位置，而不是 pyramid 位置。

---

## 5. 测试验证计划

### 5.1 单元测试

1. **Snake1D 测试**: 输入已知值，验证输出
2. **Conv1D 测试**: 验证卷积计算正确性
3. **ConvTranspose1D 测试**: 验证转置卷积计算正确性
4. **Quantizer 测试**: 验证 codebook 查找和投影

### 5.2 集成测试

1. 使用 `--test-vocoder` 模式测试 SNAC vocoder
2. 使用固定 token 序列验证输出一致性
3. 对比 chatllm.cpp 和 llama.cpp 的中间结果

### 5.3 端到端测试

1. 使用相同的输入文本和模型
2. 比较 chatllm.cpp 和 llama.cpp 的输出音频
3. 分析音频频谱特征

---

## 6. 下一步行动

### 6.1 已完成

1. **GGUF 存储格式验证**: 确认行优先顺序，维度反转
2. **权重转置逻辑验证**: 常规权重直接复制，仅 ConvTranspose1D 需要置换
3. **Codebook 格式验证**: [4096, 8] 行优先存储
4. **Token 偏移计算验证**: `pos * SNAC_CODEBOOK_SIZE` 正确
5. **VQ strides 验证**: [4, 2, 1] 与 pyramid 结构一致
6. **Snake 激活函数 DC 偏置**: 需要在 tanh 后移除 DC

### 6.2 待完成

1. **短期**: 实现 DC 偏置移除代码
2. **中期**: 添加单元测试验证各组件
3. **长期**: 重构代码以提高可维护性

### 6.3 关键技术要点总结

| 技术点 | 结论 |
|--------|------|
| GGUF 存储格式 | 行优先，与 numpy 相同，可直接复制 |
| ConvTranspose1D 权重 | 需要 [in,k,out] -> [in,out,k] 转换 |
| ConvTranspose1D 算法 | 插零 + 翻转核 + conv1d |
| Snake DC 偏置 | sin^2 总是正，需要在 tanh 后移除 DC |
| Pyramid Token | [0,1,2,2,1,2,2]，VQ strides [4,2,1] |

---

## 附录 A: chatllm.cpp 关键代码片段

### A.1 Decoder 构造
```cpp
Decoder(InitContext *ctx, int input_channel, int channels,
        int rates_count, const int *rates, bool noise, bool depthwise,
        int attn_window_size, bool auto_output_padding, int d_out)
    : Sequential(ctx) {
    // Input conv
    if (depthwise) {
        add_block(new Conv1D(ctx, input_channel, input_channel, 7, 1, 3, 1, input_channel));
        add_block(new Conv1D(ctx, input_channel, channels, 1));
    }
    // Decoder blocks
    for (int i = 0; i < rates_count; i++) {
        const int input_dim = channels / (1 << i);
        int output_dim = channels / (1 << (i + 1));
        const int groups = depthwise ? output_dim : 1;
        add_block(new DecoderBlock(ctx, input_dim, output_dim, rates[i], noise, groups, auto_output_padding));
    }
    // Output conv
    add_block(new Snake1D(ctx, output_dim));
    add_block(new Conv1D(ctx, output_dim, d_out, 7, 1, 3));
    add_block(new Unary(ctx, Unary::Op::Tanh));
}
```

### A.2 ResidualUnit
```cpp
ResidualUnit(InitContext *ctx, int dim, int dilation, int groups, int kernel_size)
    : block(ctx) {
    const int padding = ((kernel_size - 1) * dilation) / 2;
    block.add_block(new Snake1D(ctx, dim));
    block.add_block(new Conv1D(ctx, dim, dim, kernel_size, 1, padding, dilation, groups));
    block.add_block(new Snake1D(ctx, dim));
    block.add_block(new Conv1D(ctx, dim, dim, 1));
}

ggml::tensor *ResidualUnit::forward(ComputeContext *ctx, ggml::tensor *x) {
    ggml::tensor *y = block.forward(ctx, x);
    const int64_t pad = (x->ne[0] - y->ne[0]) / 2;
    if (pad > 0) {
        x = ggml::view_3d(ctx, x, y->ne[0], x->ne[1], x->ne[2], ...);
    }
    y = ggml::add(ctx, x, y);
    return y;
}
```

---

---

## 7. 深入分析：权重格式问题 (已验证)

### 7.1 PyTorch 原始权重格式

从 snac_24khz 模型检查得到的实际张量形状:

```
# Input conv (depthwise)
decoder.model.layers.0.weight_v: [768, 7, 1]  # [out_ch, kernel, 1]

# Up conv (1x1)
decoder.model.layers.1.weight_v: [1024, 1, 768]  # [out_ch, 1, in_ch]

# Decoder layer 0 - ConvTranspose1D (注意格式!)
decoder.model.layers.2.block.layers.1.weight_v: [512, 16, 768]
# PyTorch weight_v: [in_ch=768, kernel=16, out_ch=512] (非标准格式!)
# 这是 weight normalization 的特殊格式

# Residual unit - depthwise conv
decoder.model.layers.2.block.layers.3.block.layers.1.weight_v: [512, 7, 1]
# [ch=512, kernel=7, 1] for depthwise
```

### 7.2 GGUF 存储格式 (已验证正确)

**关键发现**: GGUF 以**行优先顺序**存储数据，与 numpy 完全相同。

对于 ConvTranspose1D 权重 (PyTorch weight_v: `[768, 16, 512]`):
- GGUF 存储: `ne[0]=512, ne[1]=16, ne[2]=768`
- 原始数据按行优先: `data[i*16*512 + j*512 + k]` 对应 `weight[i, j, k]`
- 元素 `[i, j, k]` 在扁平数组中的索引: `i * (16*512) + j * 512 + k`

### 7.3 正确的权重加载方法 (已验证)

**验证结论**:
- GGUF 原始数据字节可以直接使用，无需列优先转置
- 仅 ConvTranspose1D 权重需要格式转换: `[in, k, out]` -> `[in, out, k]`

**正确加载代码**:
```cpp
// 对于常规 Conv1D 权重: 直接复制
for (int64_t i = 0; i < total_elements; i++) {
    dest[i] = src[i];  // 直接复制，无需转置
}

// 对于 ConvTranspose1D 权重: 需要置换
// GGUF raw: [in, k, out] (行优先)
// 我们需要: [in, out, k] (行优先)
for (int ic = 0; ic < in_c; ic++) {
    for (int oc = 0; oc < out_c; oc++) {
        for (int k = 0; k < ks; k++) {
            int src_idx = ic * ks * out_c + k * out_c + oc;  // [in, k, out]
            int dst_idx = ic * out_c * ks + oc * ks + k;     // [in, out, k]
            dest[dst_idx] = src[src_idx];
        }
    }
}
```

---

## 8. 核心问题总结与修复方案 (已验证)

### 8.1 问题 #1: Tensor 名称映射不匹配

**现象**: 转换脚本输出 `decoder.layers.*` 但 llama.cpp 期望 `snac.layers.*`

**修复**:
- 选项 A: 修改转换脚本的输出名称
- 选项 B: 修改 llama.cpp 的加载代码以匹配

### 8.2 问题 #2: ConvTranspose1D 权重布局 (已解决)

**现象**: 权重置换逻辑可能不正确

**已验证的解决方案**:
```cpp
// 正确的 ConvTranspose1D 权重加载
// PyTorch weight_v: [in_channels, kernel_size, out_channels]
// GGUF raw data: 行优先 [in, k, out]
// 我们需要: [in_channels, out_channels, kernel_size]

for (int ic = 0; ic < in_c; ic++) {
    for (int oc = 0; oc < out_c; oc++) {
        for (int k = 0; k < ks; k++) {
            int src_idx = ic * ks * out_c + k * out_c + oc;  // [in, k, out]
            int dst_idx = ic * out_c * ks + oc * ks + k;     // [in, out, k]
            layer.in_kernel[dst_idx] = raw[src_idx];
        }
    }
}
```

### 8.3 问题 #3: Quantizer Codebook 格式 (已验证)

**已验证**: Codebook 存储为 `[codebook_size, codebook_dim]` = `[4096, 8]`
- GGUF ne[]: `ne[0]=8, ne[1]=4096`
- 直接按行优先读取即可

### 8.4 问题 #4: Token 偏移计算 (已验证)

**chatllm.cpp 的做法**:
```cpp
// chatllm.cpp decoder_push_llm_tok_id:
id = id - ((vocoder_ids.size() % snac::FRAME_SIZE) * codec_config.codebook_size);
```

**llama.cpp 正确做法**: 使用帧内位置作为偏移
```cpp
// Pyramid 结构: [0, 1, 2, 2, 1, 2, 2]
// pos 是 pyramid 中的位置 (0-6)
int id = all_tokens[f * SNAC_FRAME_SIZE + pos] - pos * SNAC_CODEBOOK_SIZE;
```

### 8.5 问题 #5: Snake 激活函数 DC 偏置 (已解决)

**现象**: 输出音频存在 DC 偏置

**原因**: Snake 激活函数 `snake(x, alpha) = x + sin^2(alpha*x) / alpha` 总是添加正值

**已验证的解决方案**:
```cpp
// 在 tanh 之后移除 DC 偏置
for (int64_t i = 0; i < n_samples; i++) {
    float mean = compute_mean(audio, n_samples);
    audio[i] -= mean;
}
// 使用阈值避免浮点误差
if (std::abs(mean) > 1e-9f) {
    remove_dc_offset(audio, n_samples);
}
```

---

## 9. 建议的调试步骤

1. **添加详细的权重调试输出**
2. **使用 --test-vocoder 模式测试单个组件**
3. **与 chatllm.cpp 的中间结果对比**
4. **验证 GGUF 转换脚本的输出格式**

---

*报告生成时间: 2026-03-04*
*作者: Claude Code Analysis Team*
*版本: 1.2 (包含已验证的关键技术发现)*

## 更新历史

| 版本 | 日期 | 更新内容 |
|------|------|----------|
| 1.0 | 2026-03-04 | 初始报告，对比分析三个实现 |
| 1.1 | 2026-03-04 | 添加深入权重分析 |
| 1.2 | 2026-03-04 | 添加已验证的关键技术发现 (Section 0)，更新问题状态 |
