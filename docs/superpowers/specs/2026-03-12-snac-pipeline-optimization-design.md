# SNAC Pipeline Optimization Design

## Overview

Optimize Orpheus TTS streaming performance by overlapping LLM token generation with SNAC audio decoding to achieve **RTF < 0.5x** with **200-500ms first-audio latency**.

## Current State

- **Non-streaming RTF**: 1.31x (LLM 0.57x + SNAC 0.74x)
- **Streaming RTF**: 0.58x - 1.02x (with sliding window optimization)
- **Execution model**: Sequential - LLM generates tokens, then SNAC decodes

## Target

- **Total RTF**: < 0.5x (effective)
- **First-audio latency**: 200-500ms
- **Use case**: Real-time streaming for conversational AI

## Design

### Architecture

```
Current Flow (Sequential):
LLM generates token → wait for LLM → SNAC decodes → wait for SNAC → repeat

Proposed Flow (Pipelined):
Time ──────────────────────────────────────────────────────►
LLM:    [Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4] ...
               ↓         ↓         ↓         ↓
SNAC:         [Chunk 1] [Chunk 2] [Chunk 3] [Chunk 4] ...
```

### Components

1. **Double Buffer** (`snac_double_buffer`)
   - Two SNAC contexts for ping-pong operation
   - Thread-safe state management

2. **GPU Stream Manager**
   - Separate CUDA streams for LLM (stream 0) and SNAC (stream 1)
   - Automatic hardware scheduling

3. **Audio Output Queue**
   - Thread-safe queue for decoded audio chunks
   - Non-blocking for LLM thread

### Data Structures

```cpp
// snac-ggml.h
struct snac_double_buffer {
    snac_ggml_context ctx[2];
    int active_idx = 0;
    int ready_idx = -1;

    std::atomic<bool> decoding{false};
    std::vector<float> output_pcm;

    std::mutex mtx;
    std::condition_variable cv;
};

// orpheus-tts.cpp
struct gpu_stream_manager {
    ggml_backend_t llm_backend;
    ggml_backend_t snac_backend;
    cudaStream_t llm_stream;
    cudaStream_t snac_stream;
};
```

### Execution Flow

```
主线程 (LLM)                    SNAC 线程
    │                               │
    ▼                               ▼
[生成 token]                    [等待 chunk]
    │                               │
    ▼                               │
[积累到 chunk?]                  │
    │ Yes                           │
    ▼                               │
[提交到 SNAC queue] ──────────► [接收 chunk]
    │                               │
    ▼                               ▼
[继续生成 token]                [SNAC 解码]
    │                               │
    │                               ▼
    │                          [输出音频]
    │                               │
    ▼                               ▼
[下一个 chunk...]              [等待下一个 chunk]
```

### Key Implementation Points

1. **Files to modify**:
   - `tools/tts/snac-ggml.h` - Add `snac_double_buffer` structure
   - `tools/tts/snac-ggml.cpp` - Implement async decode interface
   - `tools/tts/orpheus-tts.cpp` - Add SNAC thread, modify main loop

2. **Thread synchronization**:
   - Use `std::mutex` + `std::condition_variable` for chunk handoff
   - `std::atomic<bool>` for decoding state

3. **GPU resource sharing**:
   - Both LLM and SNAC share same GPU memory
   - CUDA streams handle automatic scheduling

## Error Handling

| Scenario | Handling |
|----------|----------|
| SNAC decode fails | Mark chunk invalid, continue to next |
| GPU OOM | Fallback to single-buffer mode |
| LLM finishes before SNAC | Wait for SNAC completion |
| Audio queue full | Block or drop old data (configurable) |

## Edge Cases

```cpp
// Short text (< 1 chunk): sync processing
if (estimated_frames < min_chunk_frames) {
    return sync_decode(tokens);
}

// Final chunk: ensure flush completes
void finalize() {
    snac_thread.join();
    output_remaining_audio();
}
```

## Expected Performance

### Latency Improvement

| Stage | Current | Optimized |
|-------|---------|-----------|
| LLM first chunk | ~200ms | ~200ms |
| SNAC first chunk | ~150ms | ~0ms (parallel) |
| **First audio output** | **~350ms** | **~200ms** |

### RTF Prediction

- **Single GPU**: Effective RTF ~0.3-0.4x (1.3-1.5x speedup from parallelism)
- **With F16 optimization**: Potentially < 0.3x

## Implementation Phases

1. **Phase 1**: Add double buffer structure
2. **Phase 2**: Implement async SNAC decode
3. **Phase 3**: Modify main loop for pipeline
4. **Phase 4**: Testing and benchmarking

## Success Criteria

- [ ] RTF < 0.5x for streaming mode
- [ ] First-audio latency < 500ms
- [ ] Audio quality maintained (ZCR 0.08-0.20)
- [ ] No memory leaks in long-running tests
