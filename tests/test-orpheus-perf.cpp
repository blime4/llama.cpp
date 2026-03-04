// Orpheus-TTS Performance Benchmarks
//
// This test measures performance of basic audio operations
// to establish baseline metrics for optimization tracking.

#include "orpheus-test-utils.h"

#include <cstdio>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>

using namespace orpheus_test;

// ============================================================================
// Performance Timing Utilities
// ============================================================================

struct PerfResult {
    std::string name;
    int64_t total_us;
    int64_t min_us;
    int64_t max_us;
    int iterations;
    float avg_us;
};

// Run a benchmark function multiple times and return statistics
template<typename Func>
PerfResult benchmark(const std::string& name, Func func, int iterations) {
    PerfResult result;
    result.name = name;
    result.iterations = iterations;
    result.total_us = 0;
    result.min_us = INT64_MAX;
    result.max_us = 0;

    for (int i = 0; i < iterations; i++) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        int64_t elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

        result.total_us += elapsed;
        result.min_us = std::min(result.min_us, elapsed);
        result.max_us = std::max(result.max_us, elapsed);
    }

    result.avg_us = (float)result.total_us / iterations;
    return result;
}

// Print benchmark result
void print_result(const PerfResult& r) {
    printf("  %s: avg=%.2f us, min=%lld us, max=%lld us (%d iterations)\n",
           r.name.c_str(), r.avg_us, (long long)r.min_us, (long long)r.max_us, r.iterations);
}

// ============================================================================
// Performance Tests
// ============================================================================

static void bench_vector_ops() {
    printf("Benchmarking vector operations...\n");

    const int size = 1000000;

    std::vector<float> a(size, 0.5f);
    std::vector<float> b(size, 0.3f);
    std::vector<float> c(size);

    // Vector add
    {
        auto result = benchmark("vector_add_1M", [&]() {
            for (int i = 0; i < size; i++) {
                c[i] = a[i] + b[i];
            }
        }, 10);
        print_result(result);
    }

    // Vector multiply
    {
        auto result = benchmark("vector_mul_1M", [&]() {
            for (int i = 0; i < size; i++) {
                c[i] = a[i] * b[i];
            }
        }, 10);
        print_result(result);
    }
}

static void bench_trig_functions() {
    printf("Benchmarking trig functions...\n");

    const int size = 100000;

    std::vector<float> input(size);
    std::vector<float> output(size);

    for (int i = 0; i < size; i++) {
        input[i] = (float)i / size * 2.0f - 1.0f;  // -1 to 1
    }

    // sin
    {
        auto result = benchmark("sin_100k", [&]() {
            for (int i = 0; i < size; i++) {
                output[i] = std::sin(input[i]);
            }
        }, 10);
        print_result(result);
    }

    // cos
    {
        auto result = benchmark("cos_100k", [&]() {
            for (int i = 0; i < size; i++) {
                output[i] = std::cos(input[i]);
            }
        }, 10);
        print_result(result);
    }

    // tanh
    {
        auto result = benchmark("tanh_100k", [&]() {
            for (int i = 0; i < size; i++) {
                output[i] = std::tanh(input[i]);
            }
        }, 10);
        print_result(result);
    }
}


static void bench_token_generation() {
    printf("Benchmarking token generation...\n");

    const int n_frames = 100;

    {
        auto result = benchmark("random_tokens_100frames", [&]() {
        auto tokens = generate_deterministic_tokens(n_frames, 42);
        }, 1000);
        print_result(result);
    }

    {
        auto result = benchmark("zeros_tokens_100frames", [&]() {
        auto tokens = generate_zeros_tokens(n_frames);
        }, 1000);
        print_result(result);
    }

    {
        auto result = benchmark("sequential_tokens_100frames", [&]() {
        auto tokens = generate_sequential_tokens(n_frames);
        }, 1000);
        print_result(result);
    }
}

static void bench_audio_metrics() {
    printf("Benchmarking audio metrics computation...\n");

    const int n_samples = 24000;  // 1 second of audio

    // Generate test signal (440 Hz sine wave)
    std::vector<float> samples(n_samples);
    for (int i = 0; i < n_samples; i++) {
        samples[i] = 0.5f * std::sin(2.0f * (float)M_PI * 440.0f * i / 24000);
    }

    {
        auto result = benchmark("compute_metrics_sine_1s", [&]() {
        auto m = compute_audio_metrics(samples, 24000);
        (void)m;
    }, 100);
        print_result(result);
    }

    // White noise
    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& s : samples) {
        s = dist(rng);
    }

    {
        auto result = benchmark("compute_metrics_noise_1s", [&]() {
        auto m = compute_audio_metrics(samples, 24000);
        (void)m;
    }, 100);
        print_result(result);
    }
}

// ============================================================================
// Memory Usage Estimation
// ============================================================================

static void estimate_memory_usage() {
    printf("Estimating memory usage for SNAC components...\n");

    // Token storage (3 heads)
    const int n_frames = 1000;
    auto tokens = generate_deterministic_tokens(n_frames, 42);

    size_t token_memory = 0;
    for (const auto& head : tokens) {
        token_memory += head.size() * sizeof(int);
    }
    printf("  Token storage (1000 frames): %.2f KB\n", (float)token_memory / 1024.0);

    // Audio buffer (1 second)
    const int audio_samples = 24000;
    size_t audio_memory = audio_samples * sizeof(float);
    printf("  Audio buffer (1 second): %.2f KB\n", (float)audio_memory / 1024.0);

    // Decoder weights estimate (simplified)
    // Typical SNAC decoder has ~10-20M parameters
    size_t decoder_params = 15 * 1024 * 1024;  // ~15M params
    printf("  Decoder weights estimate: %.2f MB\n", (float)decoder_params / (1024 * 1024));
}

// ============================================================================
// Main
// ============================================================================

int main() {
    printf("=== Orpheus-TTS Performance Benchmarks ===\n\n    printf("\n--- Vector Operations ---\n");
    bench_vector_ops();

    printf("\n--- Trig Functions ---\n");
    bench_trig_functions();

    printf("\n--- Token Generation ---\n");
    bench_token_generation();

    printf("\n--- Audio Metrics ---\n");
    bench_audio_metrics();

    printf("\n--- Memory Usage ---\n");
    estimate_memory_usage();

    printf("\n=== All Performance Benchmarks Complete ===\n");
    return 0;
}
