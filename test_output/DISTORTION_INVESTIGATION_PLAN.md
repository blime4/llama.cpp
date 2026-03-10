# GGML SNAC Vocoder - Investigation Complete

**Status:** RESOLVED (2026-03-10)

## Final Status
- **Short audio (<2s):** ZCR 0.045-0.057 ✓ GOOD
- **Long audio (>10s):** ZCR 0.057 ✓ GOOD
- **Target ZCR:** 0.02-0.06 for clean speech ✓ ACHIEVED

## Root Cause Analysis

### Primary Issue: Output Convolution Kernel Format

The distortion was caused by incorrect kernel format in the output convolution layer.

**Problem:** The output convolution using `ggml_im2col` required a specific kernel format `[K, IC, 1, OC]` but was receiving incorrectly formatted data.

**Fix Applied:**
1. Fixed kernel format to `[K, IC, 1, OC]` for ggml_im2col compatibility
2. Added F32 conversion for input tensor (CUDA im2col requires F32)
3. Fixed matrix multiplication order to `mul_mat(im2col_2d, kernel_2d)`

### Secondary Issue: ConvTranspose1D Kernel Type

**Problem:** CUDA `ggml_conv_transpose_1d` requires F32 kernel, but F16 was being passed.

**Fix Applied:**
1. Changed kernel type conversion from F16 to F32
2. Added kernel permutation from `[OC, K, IC]` to `[K, OC, IC]`

### Tertiary Issue: Decoder Layer Depthwise Conv Kernel

**Problem:** Residual depthwise convolution kernels needed F16 conversion for consistency.

**Fix Applied:**
1. Added F16 conversion for depthwise conv kernels in decoder layers

## Results

| Metric | Before Fix | After Fix | Improvement |
|--------|------------|-----------|-------------|
| Long audio ZCR | 0.155 (DISTORTED) | 0.057 (GOOD) | 63% reduction |
| Short audio ZCR | 0.045 (GOOD) | 0.045-0.057 (GOOD) | Stable |
| Audio quality | Distorted/buzzing | Clean speech | Fully resolved |

## Key Files Modified

- `/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/llama.cpp/tools/tts/snac-ggml.cpp`
- `/LocalRun/shaobo.xie/2_Pytorch/docker/test/debug/llama.cpp/tools/tts/snac-ggml.h`

## Technical Details

### Output Convolution Fix (snac-ggml.cpp)

```cpp
// Kernel format fix for ggml_im2col
// GGUF stores [K, IC, OC], im2col expects [K, IC, 1, OC]
struct ggml_tensor * kernel_4d = ggml_reshape_4d(ctx, w.out_conv_kernel, K, IC, 1, OC);

// F32 conversion for CUDA im2col input
if (cur_4d->type != GGML_TYPE_F32) {
    cur_4d_f32 = ggml_cpy(ctx, cur_4d, cur_f32);
}

// Correct matrix multiplication order
cur = ggml_mul_mat(ctx, im2col_2d, kernel_2d);
```

### ConvTranspose1D Fix (snac-ggml.cpp)

```cpp
// CUDA requires F32 kernel
if (kernel->type != GGML_TYPE_F32) {
    kernel_f32 = ggml_cpy(ctx, kernel, kernel_f32_tensor);
}

// Permute kernel from [OC, K, IC] to [K, OC, IC]
kernel = ggml_permute(ctx, kernel, 1, 0, 2, 3);
```

## Lessons Learned

1. **GGML im2col format requirements:** The im2col operation has strict kernel format requirements that must be followed exactly
2. **CUDA type requirements:** CUDA operations often require F32 types even when CPU works with F16
3. **Matrix multiplication order:** The order of operands in `mul_mat` matters significantly for correctness

## Verification

To verify the fix works correctly:

```bash
# Test vocoder with random tokens
./build/bin/llama-orpheus-tts \
    --model-vocoder models/snac-24khz-f16.gguf \
    --test-vocoder --test-frames 50 \
    -o test_output.wav

# Check ZCR in output (should be 0.02-0.06)
python3 -c "
import wave
import struct
with wave.open('test_output.wav', 'rb') as w:
    frames = w.readframes(w.getnframes())
    samples = struct.unpack(f'{len(frames)//2}h', frames)
    zcr = sum(1 for i in range(1, len(samples)) if (samples[i] > 0) != (samples[i-1] > 0)) / len(samples)
    print(f'ZCR: {zcr:.3f}')
"
```

## Historical Investigation (Preserved for Reference)

### Phase 1: Data Collection (Parallel Agents) - COMPLETED

| Agent | Focus Area | Result |
|-------|------------|--------|
| Kernel Analysis | All 7 kernel layouts | VERIFIED CORRECT after fixes |
| Snake Activation | Epsilon difference | NEGLIGIBLE IMPACT |
| Tensor Comparison | Layer-by-layer | IDENTIFIED output conv issue |
| Length Analysis | Duration vs ZCR | NOT length-dependent |

### Original Test Files
- Short (good): `test_output/snac_fix_test.wav`
- Long (distorted): `test_output/snac_longer_test.wav` (now fixed)
