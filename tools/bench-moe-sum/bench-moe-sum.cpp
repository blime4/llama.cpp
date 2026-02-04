// Simple benchmark for GGML_OP_MOE_SUM
//
// This program compares the performance of:
// 1. moe_sum operator - specialized MoE expert aggregation
// 2. Traditional ADD loop - equivalent functionality using sequential adds
//
// Usage:
//   ./bench-moe-sum [options]
//   Options:
//     -h, --hidden <n>     Hidden dimension (default: 4096)
//     -e, --experts <n>    Number of experts (default: 4)
//     -t, --tokens <n>     Number of tokens (default: 256)
//     -i, --iterations <n> Number of iterations (default: 100)
//     -v, --verbose        Enable verbose output

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>
#include <ggml-cpp.h>
#include <ggml-cpu.h>

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <string>

static void print_usage(const char * prog_name) {
    printf("Usage: %s [options]\n", prog_name);
    printf("\nOptions:\n");
    printf("  -h, --hidden <n>     Hidden dimension (default: 4096)\n");
    printf("  -e, --experts <n>    Number of experts (default: 4)\n");
    printf("  -t, --tokens <n>     Number of tokens (default: 256)\n");
    printf("  -i, --iterations <n> Number of iterations (default: 100)\n");
    printf("  -v, --verbose        Enable verbose output\n");
    printf("  --help               Show this help message\n");
}

static double get_time_ms() {
#ifdef _WIN32
    LARGE_INTEGER frequency, counter;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return 1000.0 * counter.QuadPart / frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * ts.tv_sec + 1e-6 * ts.tv_nsec;
#endif
}

// Benchmark 1: Using moe_sum operator
static double benchmark_moe_sum(
    ggml_backend_t backend,
    int64_t hidden_dim,
    int64_t n_expert_used,
    int64_t n_tokens,
    int iterations,
    bool verbose) {


    // Build compute graph for moe_sum
    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,  // 16 MB
        .no_alloc = true,
    };
    ggml_context * ctx = ggml_init(params);

    // Input: [hidden_dim, n_expert_used, n_tokens]
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_dim, n_expert_used, n_tokens);
    ggml_set_name(input, "input");

    ggml_tensor * output = ggml_moe_sum(ctx, input, n_expert_used);
    ggml_set_name(output, "output");

    ggml_cgraph * gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, output);

    // Allocate tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate tensors\n");
        ggml_free(ctx);
        return -1.0;
    }

    // Initialize input data
    std::vector<float> input_data(hidden_dim * n_expert_used * n_tokens);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = (float)(i % 100) / 100.0f;  // Simple pattern
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Warmup
    ggml_backend_graph_compute(backend, gf);

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    double end = get_time_ms();
    double elapsed = end - start;

    // Verify output (simple check)
    std::vector<float> output_data(hidden_dim * n_tokens);
    ggml_backend_tensor_get(output, output_data.data(), 0, output_data.size() * sizeof(float));

    // For moe_sum, output[t][d] = sum over experts k: input[d][k][t]
    bool correct = true;
    for (int64_t t = 0; t < n_tokens && correct; t++) {
        for (int64_t d = 0; d < hidden_dim; d++) {
            float expected = 0.0f;
            for (int64_t k = 0; k < n_expert_used; k++) {
                expected += input_data[d + k * hidden_dim + t * hidden_dim * n_expert_used];
            }
            float actual = output_data[d + t * hidden_dim];
            if (std::abs(expected - actual) > 1e-3f) {
                if (verbose) {
                    printf("Mismatch at t=%ld d=%ld: expected=%f actual=%f\n",
                           t, d, expected, actual);
                }
                correct = false;
                break;
            }
        }
    }

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (correct) {
        if (verbose) {
            printf("moe_sum verification: PASSED\n");
        }
    } else {
        printf("moe_sum verification: FAILED\n");
    }

    return elapsed;
}

// Benchmark 2: Using traditional ADD loop (equivalent to CPU implementation)
static double benchmark_add_loop(
    ggml_backend_t backend,
    int64_t hidden_dim,
    int64_t n_expert_used,
    int64_t n_tokens,
    int iterations,
    bool verbose) {

    struct ggml_init_params params = {
        .mem_size = 16 * 1024 * 1024,  // 16 MB
        .no_alloc = true,
    };
    ggml_context * ctx = ggml_init(params);

    // Input: [hidden_dim, n_expert_used, n_tokens]
    ggml_tensor * input = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden_dim, n_expert_used, n_tokens);
    ggml_set_name(input, "input");

    // Build graph: simulate moe_sum by creating views and adding them
    // This mimics the CPU implementation
    ggml_tensor * result = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
    ggml_set_name(result, "result");

    ggml_cgraph * gf = ggml_new_graph(ctx);

    // Initialize result to zero (via a mul with 0)
    ggml_tensor * zero = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, hidden_dim, n_tokens);
    ggml_set_name(zero, "zero");
    ggml_tensor * cur = ggml_mul(ctx, result, zero);

    // Add each expert's contribution
    for (int64_t k = 0; k < n_expert_used; k++) {
        // Create view of expert k's output: [hidden_dim, n_tokens]
        ggml_tensor * expert_view = ggml_view_3d(ctx, input,
            hidden_dim, n_tokens, 1,
            input->nb[0], input->nb[2], k * input->nb[1]);
        ggml_format_name(expert_view, "expert_%ld", k);
        cur = ggml_add(ctx, cur, expert_view);
    }

    ggml_build_forward_expand(gf, cur);

    // Allocate tensors
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        fprintf(stderr, "Failed to allocate tensors\n");
        ggml_free(ctx);
        return -1.0;
    }

    // Initialize input data
    std::vector<float> input_data(hidden_dim * n_expert_used * n_tokens);
    for (size_t i = 0; i < input_data.size(); i++) {
        input_data[i] = (float)(i % 100) / 100.0f;
    }
    ggml_backend_tensor_set(input, input_data.data(), 0, input_data.size() * sizeof(float));

    // Warmup
    ggml_backend_graph_compute(backend, gf);

    // Benchmark
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        ggml_backend_graph_compute(backend, gf);
    }
    double end = get_time_ms();
    double elapsed = end - start;

    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);

    if (verbose) {
        printf("add_loop: Verification skipped (same backend as moe_sum)\n");
    }

    return elapsed;
}

int main(int argc, char ** argv) {
    int64_t hidden_dim = 4096;
    int64_t n_expert_used = 4;
    int64_t n_tokens = 256;
    int iterations = 100;
    bool verbose = false;

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--hidden") == 0) {
            if (i + 1 < argc) {
                hidden_dim = atoll(argv[++i]);
            } else {
                fprintf(stderr, "Error: --hidden requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--experts") == 0) {
            if (i + 1 < argc) {
                n_expert_used = atoll(argv[++i]);
            } else {
                fprintf(stderr, "Error: --experts requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--tokens") == 0) {
            if (i + 1 < argc) {
                n_tokens = atoll(argv[++i]);
            } else {
                fprintf(stderr, "Error: --tokens requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--iterations") == 0) {
            if (i + 1 < argc) {
                iterations = atoi(argv[++i]);
            } else {
                fprintf(stderr, "Error: --iterations requires an argument\n");
                return 1;
            }
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            verbose = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            return 0;
        } else {
            fprintf(stderr, "Error: Unknown option '%s'\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    printf("=================================================\n");
    printf("GGML_OP_MOE_SUM Performance Benchmark\n");
    printf("=================================================\n");
    printf("Configuration:\n");
    printf("  Hidden dimension: %ld\n", hidden_dim);
    printf("  Number of experts: %ld\n", n_expert_used);
    printf("  Number of tokens: %ld\n", n_tokens);
    printf("  Iterations: %d\n", iterations);
    printf("=================================================\n\n");

    // Initialize backend
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        fprintf(stderr, "Failed to initialize backend\n");
        return 1;
    }

    printf("Using CPU backend\n");

    // Run benchmarks
    double time_moe_sum = benchmark_moe_sum(backend, hidden_dim, n_expert_used, n_tokens, iterations, verbose);
    double time_add_loop = benchmark_add_loop(backend, hidden_dim, n_expert_used, n_tokens, iterations, verbose);

    // Print results
    printf("\n=================================================\n");
    printf("Results (averaged over %d iterations):\n", iterations);
    printf("=================================================\n");

    if (time_moe_sum >= 0) {
        printf("  moe_sum:      %8.2f ms  (%8.2f us/iter)\n", time_moe_sum, time_moe_sum * 1000.0 / iterations);
    } else {
        printf("  moe_sum:      NOT SUPPORTED\n");
    }

    if (time_add_loop >= 0) {
        printf("  add_loop:     %8.2f ms  (%8.2f us/iter)\n", time_add_loop, time_add_loop * 1000.0 / iterations);
    }

    if (time_moe_sum >= 0 && time_add_loop >= 0) {
        double speedup = time_add_loop / time_moe_sum;
        printf("\n  Speedup:      %.2fx\n", speedup);

        // Calculate effective bandwidth
        size_t bytes_read = hidden_dim * n_expert_used * n_tokens * sizeof(float);
        size_t bytes_written = hidden_dim * n_tokens * sizeof(float);
        size_t total_bytes = (bytes_read + bytes_written) * iterations;
        double gb_per_sec = (total_bytes / 1e9) / (time_moe_sum / 1000.0);
        printf("  moe_sum bandwidth: %.2f GB/s\n", gb_per_sec);
    }

    printf("=================================================\n");

    ggml_backend_free(backend);

    return 0;
}
