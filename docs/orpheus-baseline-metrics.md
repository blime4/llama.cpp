# Orpheus TTS Baseline Performance Metrics

**Date**: 2026-03-05
**Branch**: feat/orpheus-tts
**Commit**: dd6fb4899 (fix(tts): fix weight loading transpose issues in SNAC vocoder)

## System Configuration

| Component | Value |
|-----------|-------|
| Model | orpheus-3b-f16.gguf |
| Vocoder | snac-24khz-f16.gguf |
| Threads | 4 |
| CPU | Linux 5.15.0-107-generic |

## Baseline Metrics Summary

| Metric | Baseline | Target | Status |
|--------|----------|--------|--------|
| LLM ms/tok | ~208-301 ms | <15 ms | Below target |
| Vocoder RTF | 0.02 x realtime | >1.0 | Below target |
| First audio latency | ~660 ms | <500 ms | Close to target |
| Prompt encode (14 tok) | 661 ms | - | - |

## Detailed Test Results

### Test 1: Very Short Prompt ("Hello world.")

| Run | LLM (ms) | Tokens | ms/tok | Vocoder (ms) | RTF | Total (ms) | Audio (s) |
|-----|----------|--------|--------|--------------|-----|------------|-----------|
| 1 | 20456 | 98 | 208.73 | 66990 | 0.02 | 87666 | 1.19 |
| 2 | 24956 | 119 | 209.71 | 81503 | 0.02 | 106757 | 1.45 |

### Test 2: Short Prompt ("Hello world, this is a test.")

| Run | LLM (ms) | Tokens | ms/tok | Vocoder (ms) | RTF | Total (ms) | Audio (s) |
|-----|----------|--------|--------|--------------|-----|------------|-----------|
| 1 | 59098 | 196 | 301.52 | 141760 | 0.02 | 201521 | 2.39 |

## Analysis

### Current Performance Characteristics

1. **LLM Generation**: ~200-300 ms/token is significantly above the 15 ms/token target
   - This is expected for f16 model on CPU without optimization
   - Quantized models (Q4_0, Q5_0) should improve this
   - GPU inference would dramatically improve this

2. **Vocoder Performance**: 0.02x realtime indicates the vocoder is ~50x slower than realtime
   - Current SNAC implementation is not optimized for CPU
   - Significant optimization opportunities exist:
     - ConvTranspose1D can be optimized using GGML operations
     - Matrix operations can use GGML BLAS
     - Snake activation can be simplified

3. **Overall Latency**: ~120-200 seconds for 1-2 seconds of audio
   - Total generation time is ~80-100x slower than realtime
   - For interactive use, target should be <1x (realtime) or better

## Optimization Opportunities

### High Priority
1. **SNAC Vocoder Optimization** (biggest impact)
   - Optimize ConvTranspose1D implementation (currently naive loops)
   - Use GGML matrix operations instead of manual loops
   - Consider fixed-point arithmetic for audio output

2. **Model Quantization**
   - Use Q4_0 or Q5_0 quantized LLM model
   - Quantize SNAC vocoder weights

### Medium Priority
3. **Batch Processing**
   - Process multiple audio frames in parallel
   - Vectorize operations where possible

4. **Memory Layout**
   - Ensure cache-friendly data access patterns
   - Reduce memory allocations in hot paths

## Next Steps

1. Implement SNAC vocoder optimizations using GGML operations
2. Benchmark with quantized models
3. Add GPU support (CUDA/Metal)
4. Implement streaming output for lower perceived latency
