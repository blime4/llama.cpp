# Orpheus TTS Test Suite - Baseline Tests Summary

**Generated:** 2026-03-04

## Overview

This document summarizes the baseline tests implemented for the Orpheus TTS system in llama.cpp.

## Test Categories

### 1. Unit Tests (No Model Required)

| Test | File | Status |
|------|------|--------|
| Audio Metrics | `tests/test-audio-metrics.cpp` | ✅ PASS |
| Snake Activation | `tests/snac/test-snake.cpp` | 📝 Placeholder |

### 2. Vocoder Tests (SNAC Model Only)

| Test | Input | Output | Status |
|------|-------|--------|--------|
| Random Tokens (100 frames) | Random tokens | `audio/vocoder_random_100frames.wav` | ✅ Generated |
| Random Tokens (500 frames) | Random tokens | (partial) | ⏸️ Timeout |

### 3. End-to-End Tests (Full Pipeline)

| Test | Voice | Text | Duration | Status |
|------|-------|------|----------|--------|
| hello_leah | leah | "Hello, how are you today?" | 2.65s | ✅ PASS |
| long_tara | tara | "This is a longer test..." | 3.93s | ✅ PASS |

## Golden Reference Files

```
tests/golden/orpheus/
├── README.md                           # Documentation
├── audio/
│   ├── vocoder_random_100frames.wav   # 102KB, 2.13s
│   ├── e2e_hello_leah.wav             # 127KB, 2.65s
│   └── e2e_long_tara.wav              # 188KB, 3.93s
└── metrics/
    ├── vocoder_random_100frames.json  # Expected metrics
    ├── e2e_hello_leah.json            # E2E test metrics
    └── e2e_long_tara.json             # E2E test metrics
```

## Audio Quality Metrics

### Target Ranges

| Metric | Target | Range |
|--------|--------|-------|
| neg_ratio | 50% | 35-65% |
| dc_bias | 0 | \|mean\| < 0.01 |
| peak_amplitude | 0.85 | 0.7-0.95 |
| energy_0_200Hz | <20% | <40% |

### Measured Results

| Test | neg_ratio | DC Bias | Peak | Duration |
|------|-----------|---------|------|----------|
| vocoder_random_100frames | ~50% | ~0 | - | 2.13s |
| e2e_hello_leah | 41.77% | 0.003644 | 0.95 | 2.65s |
| e2e_long_tara | - | - | - | 3.93s |

## Running Tests

### Unit Tests
```bash
cmake --build build --target test-audio-metrics
./build/bin/test-audio-metrics
```

### Vocoder Test
```bash
./build/bin/llama-orpheus-tts \
    --model-vocoder models/snac-24khz-f16.gguf \
    --test-vocoder \
    --test-frames 100 \
    -o test_vocoder.wav
```

### E2E Test
```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-3b-f16.gguf \
    --model-vocoder models/snac-24khz-f16.gguf \
    -p "Hello, how are you today?" \
    -v leah \
    -o test_e2e.wav
```

## Test Infrastructure Files

| File | Purpose |
|------|---------|
| `tests/orpheus-test-utils.h` | Audio metrics, WAV I/O, token generation |
| `tests/test-audio-metrics.cpp` | Audio metrics unit tests |
| `tests/test-orpheus-perf.cpp` | Performance benchmarks |
| `src/tts/snac-snake.h/cpp` | Snake activation module |
| `tests/snac/test-snake.cpp` | Snake activation tests |

## Next Steps

1. **Add more voice tests** - Test all available voices (tara, leo, zoe, jad, bria, leah, dan, mimi, jess, carly)
2. **Generate comparison golden references** - Use TTS.cpp or chatllm.cpp for cross-validation
3. **Implement remaining SNAC component tests**
4. **Add performance regression testing**
5. **Create CI/CD workflow for automated testing**

## Notes

- Golden references should be regenerated when model weights change
- Audio quality should be verified by listening tests
- Performance metrics should be tracked over time
