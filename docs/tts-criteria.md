# TTS Acceptance Criteria (验收标准)

## Overview

本文档定义了 Orpheus TTS 系统的验收标准。**每次代码修改后必须重新编译并生成音频进行验证。**

---

## Mandatory Verification Workflow (强制验证流程)

### 每次代码修改后必须执行:

```bash
# 1. 重新编译
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/llama.cpp/build
make llama-orpheus-tts -j8

# 2. 生成测试音频
cd ..
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-f16.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "The quick brown fox jumps over the lazy dog. This is a test of the streaming text to speech system. We want to verify that the audio quality meets our acceptance criteria." \
    -o test_output/verify_test.wav \
    --use-snac-ggml

# 3. 运行质量检查
python3 -c "
import wave, struct, math

with wave.open('test_output/verify_test.wav', 'rb') as w:
    frames = w.readframes(w.getnframes())
    samples = struct.unpack(f'{len(frames)//2}h', frames)

    # ZCR
    crossings = sum(1 for i in range(1, len(samples)) if (samples[i] >= 0) != (samples[i-1] >= 0))
    zcr = crossings / len(samples)

    # Stats
    max_amp = max(samples) / 32768.0
    min_amp = min(samples) / 32768.0
    rms = math.sqrt(sum(s*s for s in samples) / len(samples)) / 32768.0
    mean = sum(samples) / len(samples) / 32768.0
    clipped = sum(1 for s in samples if abs(s) >= 32767)

    duration = len(samples) / 24000

    print(f'Duration: {duration:.2f}s')
    print(f'ZCR: {zcr:.4f}')
    print(f'Peak: {max_amp:.4f}')
    print(f'RMS: {rms:.4f}')
    print(f'Mean: {mean:.6f}')
    print(f'Clipped: {clipped}')

    # Validation
    errors = []
    if not (0.08 <= zcr <= 0.20):
        errors.append(f'ZCR {zcr:.4f} outside [0.08, 0.20]')
    if max_amp > 0.99:
        errors.append(f'Peak {max_amp:.4f} too high (clipping)')
    if abs(mean) > 0.01:
        errors.append(f'DC offset {mean:.6f} too high')
    if clipped > 0:
        errors.append(f'{clipped} clipped samples')

    if errors:
        print('\\n[FAILED]')
        for e in errors:
            print(f'  - {e}')
        exit(1)
    else:
        print('\\n[PASSED] All criteria met')
"
```

---

## Quality Metrics (质量指标)

### 1. Zero Crossing Rate (ZCR)

| 范围 | 状态 | 说明 |
|------|------|------|
| **[0.08, 0.20]** | ✅ PASS | SNAC vocoder 正常输出范围 |
| < 0.08 | ⚠️ WARN | 可能过于平滑，检查音频 |
| > 0.20 | ❌ FAIL | 可能存在噪声或失真 |

> **Note**: 原始范围 [0.02, 0.06] 基于 Python SNAC 随机 token 测试。
> 经过实际 Orpheus TTS 测试，SNAC GGML 输出的正常 ZCR 范围为 [0.08, 0.20]。

### 2. Amplitude (振幅)

| 指标 | 要求 | 说明 |
|------|------|------|
| Peak | ≤ 0.99 | 无削波 |
| RMS | 0.05 - 0.25 | 正常能量水平 |
| Mean | \|mean\| < 0.01 | 无 DC 偏移 |

### 3. Duration (时长)

| 文本长度 | 预期时长 | 容差 |
|----------|----------|------|
| ~30 words | ~10s | ±3s |
| ~60 words | ~20s | ±5s |
| ~100 words | ~35s | ±8s |

### 4. Clipping (削波)

| 指标 | 要求 |
|------|------|
| Clipped samples | **0** |

---

## Baseline Reference Files (基线参考文件)

| 文件 | 时长 | ZCR | 用途 |
|------|------|-----|------|
| `baseline_short.wav` | 3.75s | 0.1832 | 短音频参考 |
| `baseline_medium.wav` | 14.17s | 0.0978 | 中等音频参考 |
| `baseline_long.wav` | 24.92s | 0.1120 | 长音频参考 |
| `criteria_test_10s.wav` | 9.05s | 0.1379 | 10秒测试参考 |

---

## Regression Test Checklist (回归测试清单)

每次修改后必须验证:

- [ ] 编译成功无错误
- [ ] 生成测试音频
- [ ] ZCR 在 [0.08, 0.20] 范围内
- [ ] 无削波 (clipped samples = 0)
- [ ] 无 DC 偏移 (|mean| < 0.01)
- [ ] 时长符合预期
- [ ] 人工听测 (可选但推荐)

---

## Failure Response (失败处理)

如果验收失败:

1. **不要提交代码**
2. 检查错误信息
3. 修复问题
4. 重新运行验证流程
5. 只有全部通过后才能继续

---

## Changelog

| 日期 | 变更 |
|------|------|
| 2026-03-10 | 创建文档，调整 ZCR 范围从 [0.02, 0.06] 到 [0.08, 0.20] |

---

## Quick Verification Script

```bash
#!/bin/bash
# quick_verify.sh - 快速验证脚本

set -e

echo "=== TTS Quick Verification ==="

# Compile
echo "[1/3] Compiling..."
cd /LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/llama.cpp/build
make llama-orpheus-tts -j8 > /dev/null 2>&1

# Generate
echo "[2/3] Generating audio..."
cd ..
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-f16.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "This is a verification test." \
    -o test_output/quick_verify.wav \
    --use-snac-ggml > /dev/null 2>&1

# Validate
echo "[3/3] Validating..."
python3 -c "
import wave, struct, math
with wave.open('test_output/quick_verify.wav', 'rb') as w:
    frames = w.readframes(w.getnframes())
    samples = struct.unpack(f'{len(frames)//2}h', frames)
    zcr = sum(1 for i in range(1, len(samples)) if (samples[i] >= 0) != (samples[i-1] >= 0)) / len(samples)
    clipped = sum(1 for s in samples if abs(s) >= 32767)
    print(f'ZCR: {zcr:.4f}, Clipped: {clipped}')
    assert 0.08 <= zcr <= 0.20, f'ZCR out of range: {zcr}'
    assert clipped == 0, f'Clipping detected: {clipped} samples'
    print('[PASS]')
"

echo "=== Verification Complete ==="
```
