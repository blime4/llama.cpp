# SNAC Pipeline Optimization Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement LLM+SNAC pipelined execution to achieve RTF < 0.5x with 200-500ms first-audio latency.

**Architecture:** Double-buffer with async SNAC thread. While LLM generates tokens for chunk N+1, SNAC decodes chunk N in parallel. Thread-safe queue bridges producer (LLM) and consumer (SNAC).

**Tech Stack:** C++17, std::thread, std::mutex, std::condition_variable, ggml backend

---

## File Structure

```
tools/tts/
├── snac-ggml.h           # Add snac_async_decoder struct
├── snac-ggml.cpp         # Implement async decode logic
└── orpheus-tts.cpp       # Modify main loop for pipeline
```

---

## Chunk 1: Async Decoder Infrastructure

### Task 1: Add Async Decoder Struct to snac-ggml.h

**Files:**
- Modify: `tools/tts/snac-ggml.h` (add after line 288)

- [ ] **Step 1: Add thread synchronization includes**

Add at the top of `snac-ggml.h` (after existing includes):
```cpp
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <queue>
#include <future>
```

- [ ] **Step 2: Add snac_chunk_task struct**

Add after `snac_streaming_context` (after line 288):
```cpp
// ============================================================================
// Async Pipeline: LLM + SNAC Parallel Execution
// ============================================================================

// A chunk of tokens ready for SNAC decoding
struct snac_chunk_task {
    std::vector<int> tokens_head0;
    std::vector<int> tokens_head1;
    std::vector<int> tokens_head2;
    int chunk_id = 0;
    bool is_final = false;  // True for flush operation
};

// Result of SNAC decoding
struct snac_chunk_result {
    std::vector<float> pcm_samples;
    int chunk_id = 0;
    bool success = false;
    std::string error_msg;
};

// Async decoder with double-buffer and background thread
struct snac_async_decoder {
    // Model context (shared, read-only)
    snac_ggml_context * model_ctx = nullptr;
    ggml_backend_t backend = nullptr;

    // Thread management
    std::thread worker_thread;
    std::atomic<bool> running{false};
    std::atomic<bool> stop_requested{false};

    // Task queue (producer: LLM thread, consumer: SNAC thread)
    std::queue<snac_chunk_task> task_queue;
    std::mutex queue_mtx;
    std::condition_variable queue_cv;

    // Result queue (producer: SNAC thread, consumer: LLM thread)
    std::queue<snac_chunk_result> result_queue;
    std::mutex result_mtx;
    std::condition_variable result_cv;

    // Statistics
    std::atomic<int> chunks_submitted{0};
    std::atomic<int> chunks_completed{0};

    // Initialize async decoder
    bool init(snac_ggml_context * ctx, ggml_backend_t backend);

    // Start background thread
    void start();

    // Stop background thread (waits for completion)
    void stop();

    // Submit chunk for async decoding (non-blocking)
    void submit_chunk(const snac_chunk_task & task);

    // Submit final flush task
    void submit_flush();

    // Try to get completed result (non-blocking)
    // Returns true if result was available
    bool try_get_result(snac_chunk_result & result);

    // Wait for all pending chunks to complete
    void wait_all();

    // Check if worker is idle
    bool is_idle() const;

    // Get pending task count
    int pending_count() const;

private:
    // Worker thread function
    void worker_loop();

    // Decode a single chunk (called from worker thread)
    snac_chunk_result decode_chunk(const snac_chunk_task & task);
};
```

- [ ] **Step 3: Commit header changes**

```bash
git add tools/tts/snac-ggml.h
git commit -m "feat(tts): add async decoder struct for pipeline optimization"
```

---

### Task 2: Implement Async Decoder in snac-ggml.cpp

**Files:**
- Modify: `tools/tts/snac-ggml.cpp` (add at end of file)

- [ ] **Step 1: Implement init() method**

Add at end of `snac-ggml.cpp`:
```cpp
// ============================================================================
// snac_async_decoder Implementation
// ============================================================================

bool snac_async_decoder::init(snac_ggml_context * ctx, ggml_backend_t backend) {
    if (!ctx || !ctx->loaded) {
        LOG_ERR("%s: Invalid model context\n", __func__);
        return false;
    }

    model_ctx = ctx;
    this->backend = backend ? backend : ggml_backend_cpu_init();

    return true;
}
```

- [ ] **Step 2: Implement start() method**

```cpp
void snac_async_decoder::start() {
    if (running.load()) {
        return;  // Already running
    }

    stop_requested.store(false);
    running.store(true);
    worker_thread = std::thread(&snac_async_decoder::worker_loop, this);

    LOG_INF("%s: Async SNAC decoder started\n", __func__);
}
```

- [ ] **Step 3: Implement stop() method**

```cpp
void snac_async_decoder::stop() {
    if (!running.load()) {
        return;
    }

    stop_requested.store(true);
    queue_cv.notify_all();  // Wake up worker

    if (worker_thread.joinable()) {
        worker_thread.join();
    }

    running.store(false);

    // Clear queues
    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        while (!task_queue.empty()) {
            task_queue.pop();
        }
    }
    {
        std::lock_guard<std::mutex> lock(result_mtx);
        while (!result_queue.empty()) {
            result_queue.pop();
        }
    }

    LOG_INF("%s: Async SNAC decoder stopped (completed %d chunks)\n",
            __func__, chunks_completed.load());
}
```

- [ ] **Step 4: Implement submit methods**

```cpp
void snac_async_decoder::submit_chunk(const snac_chunk_task & task) {
    {
        std::lock_guard<std::mutex> lock(queue_mtx);
        task_queue.push(task);
        chunks_submitted++;
    }
    queue_cv.notify_one();
}

void snac_async_decoder::submit_flush() {
    snac_chunk_task flush_task;
    flush_task.is_final = true;
    flush_task.chunk_id = -1;
    submit_chunk(flush_task);
}
```

- [ ] **Step 5: Implement result retrieval**

```cpp
bool snac_async_decoder::try_get_result(snac_chunk_result & result) {
    std::lock_guard<std::mutex> lock(result_mtx);
    if (result_queue.empty()) {
        return false;
    }
    result = result_queue.front();
    result_queue.pop();
    return true;
}

void snac_async_decoder::wait_all() {
    std::unique_lock<std::mutex> lock(queue_mtx);
    queue_cv.wait(lock, [this]() {
        return task_queue.empty() || stop_requested.load();
    });
}

bool snac_async_decoder::is_idle() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mtx));
    return task_queue.empty();
}

int snac_async_decoder::pending_count() const {
    std::lock_guard<std::mutex> lock(const_cast<std::mutex&>(queue_mtx));
    return (int)task_queue.size();
}
```

- [ ] **Step 6: Implement worker loop**

```cpp
void snac_async_decoder::worker_loop() {
    while (!stop_requested.load()) {
        snac_chunk_task task;

        // Wait for task
        {
            std::unique_lock<std::mutex> lock(queue_mtx);
            queue_cv.wait(lock, [this]() {
                return !task_queue.empty() || stop_requested.load();
            });

            if (stop_requested.load()) {
                break;
            }

            if (task_queue.empty()) {
                continue;
            }

            task = task_queue.front();
            task_queue.pop();
        }

        // Decode the chunk
        snac_chunk_result result = decode_chunk(task);

        // Store result
        {
            std::lock_guard<std::mutex> lock(result_mtx);
            result_queue.push(result);
            chunks_completed++;
        }
        result_cv.notify_one();
    }
}
```

- [ ] **Step 7: Implement decode_chunk()**

```cpp
snac_chunk_result snac_async_decoder::decode_chunk(const snac_chunk_task & task) {
    snac_chunk_result result;
    result.chunk_id = task.chunk_id;

    if (task.is_final) {
        // Flush task - return empty success
        result.success = true;
        return result;
    }

    if (task.tokens_head0.empty()) {
        result.success = true;
        return result;
    }

    // Calculate output length
    int num_frames = (int)task.tokens_head0.size();
    int output_samples = num_frames * SNAC_GGML_SAMPLES_PER_FRAME;

    // Decode using existing snac_ggml_decode function
    std::vector<std::vector<int>> pyramid_tokens(3);
    pyramid_tokens[0] = task.tokens_head0;
    pyramid_tokens[1] = task.tokens_head1;
    pyramid_tokens[2] = task.tokens_head2;

    auto decode_start = std::chrono::high_resolution_clock::now();

    result.pcm_samples = snac_ggml_decode(*model_ctx, pyramid_tokens);

    auto decode_end = std::chrono::high_resolution_clock::now();
    auto decode_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        decode_end - decode_start).count();

    result.success = !result.pcm_samples.empty();

    if (!result.success) {
        result.error_msg = "SNAC decode returned empty audio";
        LOG_WRN("%s: Chunk %d decode failed: %s\n",
                __func__, task.chunk_id, result.error_msg.c_str());
    } else {
        LOG_DBG("%s: Chunk %d decoded %zu samples in %ld ms\n",
                __func__, task.chunk_id, result.pcm_samples.size(), decode_ms);
    }

    return result;
}
```

- [ ] **Step 8: Commit implementation**

```bash
git add tools/tts/snac-ggml.cpp
git commit -m "feat(tts): implement async SNAC decoder for pipeline optimization"
```

---

## Chunk 2: Main Loop Integration

### Task 3: Add Pipeline Mode to orpheus-tts.cpp

**Files:**
- Modify: `tools/tts/orpheus-tts.cpp`

- [ ] **Step 1: Add command-line option for pipeline mode**

Find the command-line argument parsing section and add:
```cpp
// Add near other streaming options (around line 2400)
bool pipeline_mode = false;
int pipeline_chunk_frames = 16;  // Frames per chunk for pipeline

// In argument parsing section, add:
} else if (arg == "--pipeline") {
    pipeline_mode = true;
} else if (arg == "--pipeline-chunk-frames" && i + 1 < argc) {
    pipeline_chunk_frames = std::stoi(argv[++i]);
```

- [ ] **Step 2: Add pipeline mode to help text**

In `print_usage()` function, add:
```cpp
LOG("  --pipeline              enable LLM+SNAC pipeline mode (parallel execution)\n");
LOG("  --pipeline-chunk-frames N  frames per pipeline chunk (default: 16)\n");
```

- [ ] **Step 3: Add async decoder instance**

Find where `streaming_ctx` is declared (around line 2850) and add:
```cpp
// Async pipeline decoder
snac_async_decoder async_decoder;
```

- [ ] **Step 4: Initialize async decoder in streaming mode**

Find where SNAC backend is initialized (around line 2665) and add:
```cpp
// Initialize async decoder for pipeline mode
if (pipeline_mode && streaming_mode && use_snac_ggml) {
    if (!async_decoder.init(&snac_ggml_ctx, snac_backend)) {
        LOG_ERR("Failed to initialize async SNAC decoder\n");
        return 1;
    }
    async_decoder.start();
    LOG_INF("Pipeline mode enabled: async SNAC decoder started\n");
}
```

- [ ] **Step 5: Commit main loop setup changes**

```bash
git add tools/tts/orpheus-tts.cpp
git commit -m "feat(tts): add pipeline mode command-line options and initialization"
```

---

### Task 4: Implement Pipeline Token Handling

**Files:**
- Modify: `tools/tts/orpheus-tts.cpp`

- [ ] **Step 1: Add pipeline state tracking**

Add near the async_decoder declaration:
```cpp
// Pipeline state
int pipeline_chunk_id = 0;
int pipeline_tokens_accumulated = 0;
std::vector<int> pipeline_head0, pipeline_head1, pipeline_head2;
```

- [ ] **Step 2: Implement pipeline token accumulation and submission**

Find the streaming mode token handling (around line 2816) and modify:
```cpp
// Streaming mode: feed token to streaming decoder
if (streaming_mode) {
    int normalized_token = token - AUDIO_TOKEN_START;

    if (pipeline_mode) {
        // Pipeline mode: accumulate tokens and submit chunks
        // Use existing streaming_buffer for token parsing
        streaming_ctx.buffer.add_token(normalized_token);
        pipeline_tokens_accumulated++;

        // Check if we have enough frames for a chunk
        if (streaming_ctx.buffer.has_frames(pipeline_chunk_frames)) {
            // Extract tokens for this chunk
            auto frames = streaming_ctx.buffer.get_completed_frames(pipeline_chunk_frames);

            snac_chunk_task task;
            task.tokens_head0 = frames[0];
            task.tokens_head1 = frames[1];
            task.tokens_head2 = frames[2];
            task.chunk_id = pipeline_chunk_id++;

            async_decoder.submit_chunk(task);
            streaming_ctx.buffer.consume_frames(pipeline_chunk_frames);
            pipeline_tokens_accumulated = 0;

            LOG_DBG("Submitted pipeline chunk %d (%zu frames)\n",
                    task.chunk_id, task.tokens_head0.size());
        }

        // Collect completed results (non-blocking)
        snac_chunk_result result;
        while (async_decoder.try_get_result(result)) {
            if (result.success && !result.pcm_samples.empty()) {
                streaming_pcm_samples.insert(
                    streaming_pcm_samples.end(),
                    result.pcm_samples.begin(),
                    result.pcm_samples.end()
                );
                streaming_cb_ctx.chunks_received++;
                LOG_DBG("Pipeline chunk %d completed: %zu samples\n",
                        result.chunk_id, result.pcm_samples.size());
            }
        }
    } else {
        // Original synchronous streaming mode
        streaming_ctx.add_token_and_decode(
            normalized_token,
            streaming_audio_callback,
            &streaming_cb_ctx
        );
    }
}
```

- [ ] **Step 3: Implement pipeline flush handling**

Find the streaming flush section (around line 2855) and modify:
```cpp
if (streaming_mode) {
    if (pipeline_mode) {
        // Submit remaining tokens as final chunk
        if (streaming_ctx.buffer.has_frames(1)) {
            auto frames = streaming_ctx.buffer.get_completed_frames(-1);  // Get all
            snac_chunk_task task;
            task.tokens_head0 = frames[0];
            task.tokens_head1 = frames[1];
            task.tokens_head2 = frames[2];
            task.chunk_id = pipeline_chunk_id++;
            async_decoder.submit_chunk(task);
        }

        // Submit flush task
        async_decoder.submit_flush();

        // Wait for all chunks to complete
        LOG_INF("Waiting for %d pending chunks...\n", async_decoder.pending_count());
        while (async_decoder.chunks_completed.load() < async_decoder.chunks_submitted.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));

            // Collect results
            snac_chunk_result result;
            while (async_decoder.try_get_result(result)) {
                if (result.success && !result.pcm_samples.empty()) {
                    streaming_pcm_samples.insert(
                        streaming_pcm_samples.end(),
                        result.pcm_samples.begin(),
                        result.pcm_samples.end()
                    );
                }
            }
        }

        // Stop async decoder
        async_decoder.stop();

        pcm_samples = std::move(streaming_pcm_samples);
    } else {
        // Original synchronous flush
        streaming_ctx.flush(streaming_audio_callback, &streaming_cb_ctx);
        pcm_samples = std::move(streaming_pcm_samples);
    }
}
```

- [ ] **Step 4: Commit pipeline token handling**

```bash
git add tools/tts/orpheus-tts.cpp
git commit -m "feat(tts): implement pipeline token accumulation and async submission"
```

---

## Chunk 3: Testing and Validation

### Task 5: Build and Test

**Files:**
- None (testing only)

- [ ] **Step 1: Build the project**

```bash
cmake --build build --target llama-orpheus-tts -j8
```

Expected: Build succeeds with no errors.

- [ ] **Step 2: Test non-streaming mode (regression test)**

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-q4_k_m.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "Hello world" \
    -o test_output/pipeline_test_baseline.wav \
    --use-snac-ggml \
    -ngl 99 -t 8
```

Expected: Audio generated successfully, RTF ~1.3x.

- [ ] **Step 3: Test streaming mode without pipeline (regression test)**

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-q4_k_m.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "Hello world, this is a test of streaming text to speech." \
    -o test_output/pipeline_test_streaming.wav \
    --use-snac-ggml --streaming \
    -ngl 99 -t 8
```

Expected: Audio generated successfully, RTF ~0.6-1.0x.

- [ ] **Step 4: Test pipeline mode**

```bash
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-q4_k_m.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "Hello world, this is a test of the pipeline optimization for text to speech." \
    -o test_output/pipeline_test_pipeline.wav \
    --use-snac-ggml --streaming --pipeline \
    -ngl 99 -t 8
```

Expected: Audio generated successfully, RTF should improve.

- [ ] **Step 5: Verify audio quality**

```bash
python3 -c "
import wave, struct, math

for name in ['pipeline_test_baseline', 'pipeline_test_streaming', 'pipeline_test_pipeline']:
    path = f'test_output/{name}.wav'
    try:
        with wave.open(path, 'rb') as w:
            frames = w.readframes(w.getnframes())
            samples = struct.unpack(f'{len(frames)//2}h', frames)
            zcr = sum(1 for i in range(1, len(samples)) if (samples[i] >= 0) != (samples[i-1] >= 0)) / len(samples)
            max_amp = max(abs(s) for s in samples) / 32768.0
            print(f'{name}: ZCR={zcr:.4f}, Peak={max_amp:.4f}')
            if 0.08 <= zcr <= 0.20:
                print(f'  ✅ Quality OK')
            else:
                print(f'  ❌ Quality issue')
    except Exception as e:
        print(f'{name}: Error - {e}')
"
```

Expected: All files show ZCR in range [0.08, 0.20].

---

### Task 6: Performance Benchmarking

**Files:**
- None (benchmarking only)

- [ ] **Step 1: Run RTF comparison test**

```bash
echo "=== Performance Comparison ===" > test_output/pipeline_benchmark.txt

echo -e "\n--- Non-streaming ---" >> test_output/pipeline_benchmark.txt
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-q4_k_m.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "The quick brown fox jumps over the lazy dog. This is a test of the streaming text to speech system." \
    -o test_output/bench_nonstream.wav \
    --use-snac-ggml -ngl 99 -t 8 2>&1 | grep -A 10 "PERFORMANCE" >> test_output/pipeline_benchmark.txt

echo -e "\n--- Streaming (sync) ---" >> test_output/pipeline_benchmark.txt
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-q4_k_m.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "The quick brown fox jumps over the lazy dog. This is a test of the streaming text to speech system." \
    -o test_output/bench_stream_sync.wav \
    --use-snac-ggml --streaming -ngl 99 -t 8 2>&1 | grep -A 10 "PERFORMANCE" >> test_output/pipeline_benchmark.txt

echo -e "\n--- Streaming (pipeline) ---" >> test_output/pipeline_benchmark.txt
./build/bin/llama-orpheus-tts \
    -m models/orpheus-tts/orpheus-3b-q4_k_m.gguf \
    --model-vocoder models/snac/snac-24khz-f16.gguf \
    -p "The quick brown fox jumps over the lazy dog. This is a test of the streaming text to speech system." \
    -o test_output/bench_stream_pipeline.wav \
    --use-snac-ggml --streaming --pipeline -ngl 99 -t 8 2>&1 | grep -A 10 "PERFORMANCE" >> test_output/pipeline_benchmark.txt

cat test_output/pipeline_benchmark.txt
```

Expected: Pipeline mode shows improved RTF compared to sync streaming.

- [ ] **Step 2: Verify success criteria**

Check if all success criteria are met:
- [ ] RTF < 0.5x for streaming mode
- [ ] First-audio latency < 500ms
- [ ] Audio quality maintained (ZCR 0.08-0.20)
- [ ] No memory leaks in long-running tests

---

### Task 7: Final Commit

- [ ] **Step 1: Update documentation**

Update `docs/streaming-tts-implementation-plan.md` with pipeline mode status.

- [ ] **Step 2: Final commit**

```bash
git add -A
git commit -m "feat(tts): implement LLM+SNAC pipeline optimization for RTF < 0.5x

- Add snac_async_decoder with background thread
- Implement double-buffer for parallel LLM+SNAC execution
- Add --pipeline command-line option
- Maintain audio quality (ZCR 0.08-0.20)

Performance improvement:
- Streaming RTF: 0.58-1.02x → target < 0.5x
- First-audio latency: ~350ms → target < 500ms"
```

---

## Success Criteria

- [ ] RTF < 0.5x for streaming mode
- [ ] First-audio latency < 500ms
- [ ] Audio quality maintained (ZCR 0.08-0.20)
- [ ] No memory leaks in long-running tests
- [ ] Build succeeds on all platforms
- [ ] Regression tests pass (non-streaming, sync streaming)
