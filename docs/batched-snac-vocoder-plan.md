# Batched SNAC Vocoder Implementation Plan

## Project Overview

### Goal
Convert the current SNAC vocoder implementation from manual tensor operations to ggml graph-based computation, enabling batched processing for improved performance in the orpheus-tts system.

### Current State
- Location: `tools/tts/orpheus-tts.cpp`
- Implementation: Manual tensor operations with custom allocators
- Processing: Single-token-at-a-time decoding
- Performance: Limited by sequential processing overhead

### Target State
- Implementation: ggml compute graph with cplan scheduling
- Processing: Batched decoding (configurable batch size)
- Performance: Optimized parallel processing with ggml scheduling
- Maintainability: Clean separation between graph building and execution

---

## Architecture Design

### High-Level Architecture

```
+-------------------+     +------------------+     +------------------+
|   Orpheus TTS     |     |  Batched SNAC    |     |   Audio Output   |
|   (LLM Tokens)    |---->|    Vocoder       |---->|   (WAV/PCM)      |
+-------------------+     +------------------+     +------------------+
                                  |
                                  v
                          +---------------+
                          |  ggml Graph   |
                          |  (cgraph)     |
                          +---------------+
                                  |
                    +-------------+-------------+
                    |             |             |
                    v             v             v
              +----------+ +----------+ +----------+
              |Quantizer | |Quantizer | |Quantizer |
              | Layer 0  | | Layer 1  | | Layer 2  |
              +----------+ +----------+ +----------+
                    |             |             |
                    +-------------+-------------+
                                  |
                                  v
                          +---------------+
                          |   Decoder     |
                          |   Network     |
                          +---------------+
                                  |
                                  v
                          +---------------+
                          |   Output      |
                          |   (Audio)     |
                          +---------------+
```

### SNAC Model Architecture

```
Input: Codebook Indices [batch, 3, T]
       (3 quantizers, hierarchical temporal structure)

       +------------------+
       | Embedding Lookup |
       | [3 codebooks]    |
       +------------------+
               |
               v (sum embeddings)
       +------------------+
       | Decoder Input    |
       | [batch, T, 768]  |
       +------------------+
               |
               v
       +------------------+
       | Decoder Layer 0  |
       | ConvT1D(768,1024)|
       | stride=8, k=16   |
       +------------------+
               |
               v
       +------------------+
       | Decoder Layer 1  |
       | ConvT1D(1024,512)|
       | stride=8, k=16   |
       +------------------+
               |
               v
       +------------------+
       | Decoder Layer 2  |
       | ConvT1D(512,256) |
       | stride=4, k=8    |
       +------------------+
               |
               v
       +------------------+
       | Decoder Layer 3  |
       | ConvT1D(256,128) |
       | stride=2, k=4    |
       +------------------+
               |
               v
       +------------------+
       | Output Conv      |
       | Conv1D(128, 1)   |
       | kernel=7         |
       +------------------+
               |
               v
Output: Audio [batch, T*512]
```

### Quantizer Temporal Structure

```
Quantizer 0: Full resolution (T frames)
Quantizer 1: Half resolution (T/2 frames) - applied every other frame
Quantizer 2: Quarter resolution (T/4 frames) - applied every 4th frame

Example for 8 output frames:
Frame:   | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
Q0:      | A | B | C | D | E | F | G | H |  (8 codes)
Q1:      | - | a | - | b | - | c | - | d |  (4 codes at odd positions)
Q2:      | - | - | - | x | - | - | - | y |  (2 codes at positions 3,7)

Embedding combination:
Frame 0: E(Q0[A])
Frame 1: E(Q0[B]) + E(Q1[a])
Frame 2: E(Q0[C])
Frame 3: E(Q0[D]) + E(Q1[b]) + E(Q2[x])
...
```

### Batched Processing Flow

```
+------------------+
| Token Accumulator|
| (collect N tokens|
|  per quantizer)  |
+------------------+
         |
         v (when batch full or EOS)
+------------------+
| Build ggml Graph |
| - Create tensors |
| - Add operations |
| - Schedule ops   |
+------------------+
         |
         v
+------------------+
| Execute Graph    |
| (ggml_graph_compute)        |
+------------------+
         |
         v
+------------------+
| Extract Audio    |
| - De-interleave  |
| - Post-process   |
+------------------+
         |
         v
+------------------+
| Output Buffer    |
+------------------+
```

---

## Implementation Phases

### Phase 1: ggml Graph Conversion

**Objective:** Convert current manual tensor operations to ggml compute graph

**Tasks:**

| Task ID | Description | Priority | Dependencies |
|---------|-------------|----------|--------------|
| 1.1 | Create ggml context and tensor allocator for SNAC | HIGH | None |
| 1.2 | Implement codebook embedding lookup as ggml operations | HIGH | 1.1 |
| 1.3 | Convert ConvTranspose1D to ggml_conv_transpose_1d | HIGH | 1.1 |
| 1.4 | Implement Snake1 activation as custom ggml op or combination | MEDIUM | 1.1 |
| 1.5 | Build decoder layer graph structure | HIGH | 1.3, 1.4 |
| 1.6 | Implement full decoder graph building | HIGH | 1.5 |
| 1.7 | Create graph execution function with cplan | HIGH | 1.6 |
| 1.8 | Add weight loading from GGUF to graph tensors | HIGH | 1.1 |
| 1.9 | Implement unit tests for graph-based vocoder | HIGH | 1.7 |
| 1.10 | Verify output matches reference implementation | HIGH | 1.9 |

**Estimated Time:** 3-4 days

### Phase 2: Batched Processing

**Objective:** Enable processing multiple tokens in a single graph execution

**Tasks:**

| Task ID | Description | Priority | Dependencies |
|---------|-------------|----------|--------------|
| 2.1 | Design batch tensor layout for codebook indices | HIGH | Phase 1 |
| 2.2 | Implement batched embedding lookup | HIGH | 2.1 |
| 2.3 | Update ConvTranspose1D for batch dimension | HIGH | 2.2 |
| 2.4 | Handle variable-length sequences in batch | MEDIUM | 2.3 |
| 2.5 | Implement token accumulation buffer | HIGH | 2.1 |
| 2.6 | Create batch scheduling logic | HIGH | 2.5 |
| 2.7 | Add de-interleaving for batch output | MEDIUM | 2.6 |
| 2.8 | Implement dynamic batch size configuration | MEDIUM | 2.7 |
| 2.9 | Performance benchmarking (single vs batched) | HIGH | 2.8 |
| 2.10 | Memory optimization for large batches | MEDIUM | 2.9 |

**Estimated Time:** 2-3 days

### Phase 3: Integration & Testing

**Objective:** Integrate batched vocoder with orpheus-tts and validate

**Tasks:**

| Task ID | Description | Priority | Dependencies |
|---------|-------------|----------|--------------|
| 3.1 | Replace existing vocoder calls with batched version | HIGH | Phase 2 |
| 3.2 | Update streaming callback interface | HIGH | 3.1 |
| 3.3 | Add configuration options for batch size | MEDIUM | 3.1 |
| 3.4 | Implement graceful degradation (fallback to single) | MEDIUM | 3.1 |
| 3.5 | Integration tests with full TTS pipeline | HIGH | 3.2 |
| 3.6 | Latency measurement and optimization | HIGH | 3.5 |
| 3.7 | Memory usage profiling | MEDIUM | 3.5 |
| 3.8 | Edge case testing (empty, single token, max batch) | HIGH | 3.5 |
| 3.9 | Documentation update | MEDIUM | 3.5 |
| 3.10 | Performance comparison report | HIGH | 3.6, 3.7 |

**Estimated Time:** 2-3 days

---

## Technical Specifications

### Model Constants

```cpp
// SNAC 24kHz model constants
constexpr int QUANTIZER_DIM    = 768;   // Embedding dimension
constexpr int DECODER_DIM      = 1024;  // Decoder hidden dimension
constexpr int N_QUANTIZERS     = 3;     // Number of codebook levels
constexpr int UPSAMPLE_FACTOR  = 512;   // Total upsampling (8*8*4*2)
constexpr int CODEBOOK_SIZE    = 4096;  // Codes per codebook

// Decoder layer configuration
constexpr int DECODER_RATES[]  = {8, 8, 4, 2};  // Strides per layer
constexpr int DECODER_KERNELS[] = {16, 16, 8, 4}; // Kernel sizes
constexpr int DECODER_CHANNELS[] = {1024, 512, 256, 128}; // Channel dims
```

### Tensor Layouts

#### Input Tensors
```
codebook_indices: [batch, n_quantizers, seq_len]
  - Type: GGML_TYPE_I32
  - Values: 0-4095 (codebook indices)

codebook_embeddings: [n_quantizers, codebook_size, quantizer_dim]
  - Type: GGML_TYPE_F32
  - Stored in GGUF as: snac.codebooks.X.weight
```

#### Decoder Weights
```
decoder.layers[i].conv_t.weight: [in_channels, kernel_size, out_channels]
  - Type: GGML_TYPE_F32
  - Note: ConvTranspose1D weight format

decoder.layers[i].conv_t.bias: [out_channels]
  - Type: GGML_TYPE_F32

decoder.out_conv.weight: [1, kernel_size, out_channels=1]
  - Type: GGML_TYPE_F32

decoder.out_conv.bias: [1]
  - Type: GGML_TYPE_F32
```

#### Intermediate Tensors
```
embedded: [batch, seq_len, quantizer_dim]
  - Sum of codebook embeddings

after_layer[i]: [batch, seq_len * rate[i], channels[i]]
  - After ConvTranspose1D + Snake1 activation
```

#### Output Tensor
```
audio: [batch, seq_len * UPSAMPLE_FACTOR]
  - Type: GGML_TYPE_F32
  - Range: [-1.0, 1.0] after tanh
```

### Weight Loading Format

GGUF tensor naming conventions:
```
snac.codebooks.0.weight    -> [4096, 768]
snac.codebooks.1.weight    -> [4096, 768]
snac.codebooks.2.weight    -> [4096, 768]
snac.decoder.model.0.conv_t.weight   -> [768, 16, 1024]
snac.decoder.model.0.conv_t.bias     -> [1024]
snac.decoder.model.1.conv_t.weight   -> [1024, 16, 512]
snac.decoder.model.1.conv_t.bias     -> [512]
snac.decoder.model.2.conv_t.weight   -> [512, 8, 256]
snac.decoder.model.2.conv_t.bias     -> [256]
snac.decoder.model.3.conv_t.weight   -> [256, 4, 128]
snac.decoder.model.3.conv_t.bias     -> [128]
snac.decoder.out_conv.weight         -> [128, 7, 1]
snac.decoder.out_conv.bias           -> [1]
```

### Snake1 Activation

```cpp
// Snake activation function
// snake(x, alpha) = x + (1/alpha) * sin^2(alpha * x)
// Can be decomposed as:
//   temp = sin(alpha * x)
//   temp = temp * temp  // sin^2
//   temp = temp / alpha
//   output = x + temp

ggml_tensor* snake1(ggml_context* ctx, ggml_tensor* x, ggml_tensor* alpha) {
    // temp = alpha * x
    auto temp = ggml_mul(ctx, alpha, x);

    // temp = sin(temp)
    temp = ggml_sin(ctx, temp);

    // temp = temp * temp (sin^2)
    temp = ggml_mul(ctx, temp, temp);

    // temp = temp / alpha
    temp = ggml_div(ctx, temp, alpha);

    // output = x + temp
    return ggml_add(ctx, x, temp);
}
```

---

## Risk Assessment

### High Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| ggml_conv_transpose_1d not available or incompatible | Project blocked | Research existing implementations; implement custom op if needed |
| Memory fragmentation with large batches | Crashes, instability | Pre-allocate memory pools; implement memory limits |
| Output mismatch with reference | Incorrect audio output | Comprehensive unit tests; numerical comparison at each stage |

### Medium Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Performance regression | Slower than single-token | Profile early; optimize hot paths |
| ggml scheduling overhead | Diminishing returns | Benchmark different batch sizes; tune scheduling |
| Quantizer temporal alignment bugs | Audio artifacts | Careful testing of edge cases; visual inspection of spectrograms |

### Low Risk

| Risk | Impact | Mitigation |
|------|--------|------------|
| Integration issues with existing code | Delays | Incremental integration; maintain backward compatibility |
| Configuration complexity | User confusion | Sensible defaults; clear documentation |

---

## Success Criteria

### Functional Requirements

- [ ] Batched vocoder produces identical audio output to reference implementation
- [ ] Supports batch sizes from 1 to configurable maximum (default 16)
- [ ] Handles variable-length sequences within a batch
- [ ] Integrates seamlessly with existing orpheus-tts pipeline
- [ ] No audio artifacts or glitches in output

### Performance Requirements

- [ ] Batch size 8: At least 2x throughput vs single-token processing
- [ ] Batch size 16: At least 3x throughput vs single-token processing
- [ ] Memory usage scales linearly with batch size
- [ ] No memory leaks over extended operation

### Quality Requirements

- [ ] Code follows project style guidelines
- [ ] Unit test coverage >= 80% for new code
- [ ] Integration tests pass for all supported configurations
- [ ] Documentation updated for new features

---

## Task Tracking

### Progress Overview

| Phase | Status | Completion | Notes |
|-------|--------|------------|-------|
| Phase 1: ggml Graph Conversion | NOT STARTED | 0% | |
| Phase 2: Batched Processing | NOT STARTED | 0% | Blocked by Phase 1 |
| Phase 3: Integration & Testing | NOT STARTED | 0% | Blocked by Phase 2 |

### Detailed Task Status

#### Phase 1 Tasks

| Task ID | Status | Assignee | Started | Completed | Notes |
|---------|--------|----------|---------|-----------|-------|
| 1.1 | TODO | - | - | - | |
| 1.2 | TODO | - | - | - | |
| 1.3 | TODO | - | - | - | |
| 1.4 | TODO | - | - | - | |
| 1.5 | TODO | - | - | - | |
| 1.6 | TODO | - | - | - | |
| 1.7 | TODO | - | - | - | |
| 1.8 | TODO | - | - | - | |
| 1.9 | TODO | - | - | - | |
| 1.10 | TODO | - | - | - | |

#### Phase 2 Tasks

| Task ID | Status | Assignee | Started | Completed | Notes |
|---------|--------|----------|---------|-----------|-------|
| 2.1 | TODO | - | - | - | |
| 2.2 | TODO | - | - | - | |
| 2.3 | TODO | - | - | - | |
| 2.4 | TODO | - | - | - | |
| 2.5 | TODO | - | - | - | |
| 2.6 | TODO | - | - | - | |
| 2.7 | TODO | - | - | - | |
| 2.8 | TODO | - | - | - | |
| 2.9 | TODO | - | - | - | |
| 2.10 | TODO | - | - | - | |

#### Phase 3 Tasks

| Task ID | Status | Assignee | Started | Completed | Notes |
|---------|--------|----------|---------|-----------|-------|
| 3.1 | TODO | - | - | - | |
| 3.2 | TODO | - | - | - | |
| 3.3 | TODO | - | - | - | |
| 3.4 | TODO | - | - | - | |
| 3.5 | TODO | - | - | - | |
| 3.6 | TODO | - | - | - | |
| 3.7 | TODO | - | - | - | |
| 3.8 | TODO | - | - | - | |
| 3.9 | TODO | - | - | - | |
| 3.10 | TODO | - | - | - | |

### Status Legend
- **TODO**: Not yet started
- **IN_PROGRESS**: Currently being worked on
- **BLOCKED**: Cannot proceed due to dependency
- **REVIEW**: Implementation complete, needs review
- **DONE**: Completed and verified

---

## References

### Code Locations
- Current SNAC implementation: `tools/tts/orpheus-tts.cpp`
- SNAC model weights: `snac_24khz.gguf`
- GGUF format documentation: `docs/gguf.md`

### External Resources
- SNAC paper: https://arxiv.org/abs/2301.02409
- ggml documentation: https://github.com/ggerganov/ggml
- chatllm.cpp SNAC reference: https://github.com/fllmer/chatllm.cpp

---

## Changelog

| Date | Author | Description |
|------|--------|-------------|
| 2026-03-06 | Team | Initial plan creation |
