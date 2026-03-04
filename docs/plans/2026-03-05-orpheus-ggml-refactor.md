# Orpheus TTS GGML Refactor Plan

**Date**: 2026-03-05
**Branch**: feat/orpheus-tts
**Reference PRs**: #12636 (mimi decoder), #12648 (CSM + Mimi)

## Goal

Refactor the SNAC vocoder in orpheus-tts to use GGML computation graphs instead of manual loops, achieving significant performance improvements while maintaining test compatibility.

## Current State Analysis

### Performance Baseline
- **LLM**: ~200-300 ms/token (expected for f16 on CPU)
- **Vocoder**: 0.02x RTF (50x slower than realtime) - **PRIMARY BOTTLENECK**
- **Total latency**: ~120-200 seconds for 1-2 seconds of audio

### Code Structure
- `orpheus-tts.cpp`: 2719 lines, monolithic
- Manual implementations:
  - `snake_1d_inplace()` - Snake activation function
  - `snac_conv_transpose1d()` - ConvTranspose1D with manual loops
  - `conv1d()` - Standard convolution with manual loops
  - `snac_decoder_layer::forward()` - Decoder layer processing
  - `snac_quantizer_layer::forward()` - Quantizer lookup

### Test Infrastructure
- `tests/orpheus-test-utils.h` - Audio quality metrics, golden references
- `tests/test-orpheus-acceptance.cpp` - Acceptance tests
- `tests/test-orpheus-perf.cpp` - Performance tests
- `scripts/benchmark-orpheus.sh` - Benchmark script

## Target Architecture

Following the CSM pattern from PR #12636:

```
tools/tts/
├── orpheus-tts.cpp          # Main entry (LLM generation, token decoding)
├── snac-model.h             # SNAC vocoder API
├── snac-model.cpp           # SNAC vocoder GGML implementation
└── CMakeLists.txt           # Build configuration
```

### Key Components

#### 1. `snac_ggml_ctx` (like `mimi_ggml_ctx`)
```cpp
struct snac_ggml_ctx {
    gguf_context * ctx_gguf = nullptr;
    ggml_context * ctx_data = nullptr;
    ggml_context * ctx_gf   = nullptr;

    ggml_backend_t backend     = nullptr;
    ggml_backend_buffer_t buf  = nullptr;
    ggml_backend_sched_ptr sched;

    ggml_cgraph * gf = nullptr;
    std::vector<uint8_t> buf_compute_meta;

    std::unordered_map<std::string, ggml_tensor *> tensors;

    void load_gguf(const char * fname);
    void build_graph(std::function<void(ggml_context *, ggml_cgraph *)> builder_fn);
    ggml_status compute();
    ggml_tensor * get_weight(const char *fmt, ...);
};
```

#### 2. `snac_decoder_layer`
```cpp
// Convert manual loops to GGML ops:
// - snake_1d_inplace() -> ggml custom op or combination of existing ops
// - snac_conv_transpose1d() -> ggml_conv_transpose_1d()
// - conv1d() -> ggml_conv_1d()
```

#### 3. `snac_quantizer_layer`
```cpp
// Use ggml_mul_mat for codebook lookup instead of manual loops
```

#### 4. `snac_model` API
```cpp
struct snac_model {
    std::unique_ptr<snac_ggml_ctx> ctx;
    std::unique_ptr<snac_decoder> decoder;
    std::unique_ptr<snac_quantizer> quantizer;

    snac_model(const char * fname, bool verbose = false);
    ~snac_model();

    int get_sample_rate() const;

    // Decode tokens to waveform
    std::vector<float> decode(const std::vector<std::vector<int>> & tokens);

private:
    std::vector<float> decode_frame(const std::vector<int> & tokens);
};
```

## Implementation Phases

### Phase 1: Foundation (snac-model.h/cpp skeleton)
**Estimated effort**: 2-3 hours

1. Create `snac-model.h` with API definitions
2. Create `snac-model.cpp` with skeleton implementation
3. Implement `snac_ggml_ctx` infrastructure (copy from mimi-model.cpp)
4. Update CMakeLists.txt to build new files
5. **Validation**: Code compiles, basic GGUF loading works

### Phase 2: Quantizer Migration
**Estimated effort**: 3-4 hours

1. Port `snac_quantizer_layer` to GGML
2. Replace manual codebook lookup with `ggml_mul_mat`
3. Implement token embedding using GGML tensors
4. **Validation**: Quantizer produces same output as manual implementation

### Phase 3: Decoder Layer Migration
**Estimated effort**: 4-6 hours

1. Port `snac_decoder_layer` to GGML
2. Replace `conv1d()` with `ggml_conv_1d()`
3. Replace `snac_conv_transpose1d()` with `ggml_conv_transpose_1d()`
4. Implement snake activation:
   - Option A: Custom GGML op (more efficient)
   - Option B: Compose from existing ops (easier, less efficient)
   - **Decision**: Start with Option B, optimize to Option A if needed
5. Port residual units
6. **Validation**: Decoder produces same output as manual implementation

### Phase 4: Integration
**Estimated effort**: 2-3 hours

1. Update `orpheus-tts.cpp` to use `snac_model` API
2. Remove old manual implementation code
3. Keep DC bias removal (already working)
4. Update timing infrastructure to work with new implementation
5. **Validation**: End-to-end TTS works

### Phase 5: Testing & Benchmarking
**Estimated effort**: 2-3 hours

1. Run `tests/test-orpheus-acceptance.cpp`
2. Run `tests/test-orpheus-perf.cpp`
3. Run `scripts/benchmark-orpheus.sh`
4. Compare audio quality metrics with baseline
5. Document performance improvements
6. **Validation**: All tests pass, performance improved

### Phase 6: Optimization (Optional)
**Estimated effort**: 2-4 hours

1. Profile with GGML timing
2. Identify remaining bottlenecks
3. Consider:
   - Custom snake activation op
   - Batch processing multiple frames
   - Quantized weights (Q4_0, Q5_0)
   - GPU backend support
4. **Validation**: Performance targets met

## Success Criteria

### Correctness
- [ ] All acceptance tests pass (`test-orpheus-acceptance`)
- [ ] Audio quality metrics match baseline:
  - neg_ratio: 35-65%
  - dc_bias: |mean| < 0.01
  - energy_0_200Hz: <50%
  - peak_amplitude: 0.7-0.95
- [ ] Generated audio is perceptually similar to baseline

### Performance
- [ ] Vocoder RTF improves from 0.02x to **at least 0.1x** (5x speedup)
- [ ] Stretch goal: **0.5x RTF** (25x speedup, approaching realtime)
- [ ] Total generation time reduced proportionally

### Code Quality
- [ ] Code follows CSM PR style (modular, GGML-based)
- [ ] No manual loops in hot paths
- [ ] Proper error handling
- [ ] Memory management via GGML allocators

## Technical Decisions

### 1. Snake Activation Implementation
**Decision**: Compose from existing GGML ops initially
```cpp
// snake(x, alpha) = x + sin²(alpha*x) / alpha
ggml_tensor * snake = ggml_add(ctx0,
    x,
    ggml_div(ctx0,
        ggml_sqr(ctx0, ggml_sin(ctx0, ggml_mul(ctx0, alpha, x))),
        alpha
    )
);
```
**Rationale**: Faster to implement, good enough for initial version

### 2. Backend Selection
**Decision**: CPU-only initially, like mimi-model.cpp
```cpp
backend = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
```
**Rationale**: Simpler, focuses on correctness first. GPU support can be added later.

### 3. Graph Building Strategy
**Decision**: Build graph per-frame (like mimi) vs build once and reuse
- Mimi builds graph for each decode call
- For Orpheus, we could build once since frame size is constant
- **Start with per-frame for simplicity**, optimize later if needed

### 4. Weight Loading
**Decision**: Direct copy from GGUF (no transpose)
```cpp
// From MEMORY.md: All GGUF weights can be copied directly - no transpose needed!
```
**Rationale**: Proven to work correctly in current implementation

## Risk Mitigation

### Risk: Audio Quality Regression
**Mitigation**:
- Run comprehensive acceptance tests
- Compare metrics with baseline before/after
- Keep DC bias removal logic (already working)

### Risk: Performance Not Improved
**Mitigation**:
- Profile to identify bottlenecks
- Ensure backend is using optimized kernels
- Consider custom ops for snake activation
- Add GPU support if needed

### Risk: Integration Issues
**Mitigation**:
- Incremental migration (phase by phase)
- Keep old code until new code validated
- Test each phase thoroughly

## Dependencies

- GGML library (already in llama.cpp)
- Existing SNAC GGUF model file
- Test infrastructure (already in place)

## Timeline

- **Phase 1-2**: Day 1 (5-7 hours)
- **Phase 3**: Day 2 (4-6 hours)
- **Phase 4-5**: Day 3 (4-6 hours)
- **Phase 6**: Optional, ongoing optimization

**Total**: 13-20 hours for complete implementation and validation

## References

- CSM PR #12636: https://github.com/ggml-org/llama.cpp/pull/12636
- CSM PR #12648: https://github.com/ggml-org/llama.cpp/pull/12648
- MEMORY.md: Weight loading rules and debugging tips
- Baseline metrics: `docs/orpheus-baseline-metrics.md`
