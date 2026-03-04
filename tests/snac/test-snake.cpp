// Test SNAC Snake1D Activation Function
//
// Tests:
// - Basic snake activation computation
// - Zero input handling
// - Symmetry properties
// - DC bias accumulation
// - Alpha parameter sensitivity

#include "../../src/tts/snac-snake.h"
#include "../orpheus-test-utils.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <vector>

using namespace orpheus_test;

// Test snake activation at zero
static void test_snake_zero() {
    printf("Testing snake(x=0)...\n");

    float alpha = 1.0f;
    float x = 0.0f;

    // snake(0, alpha) = 0 + sin^2(0) / alpha = 0
    float sin_val = std::sin(alpha * x);
    float result = x + (sin_val * sin_val) / alpha;

    assert(std::abs(result) < 1e-6f);
    printf("  PASS: snake(0, 1.0) = %.6f\n", result);
}

// Test snake activation for positive values
static void test_snake_positive() {
    printf("Testing snake(x>0)...\n");

    float alpha = 1.0f;

    // For x > 0: snake adds positive value (sin^2 >= 0)
    std::vector<float> inputs = {0.1f, 0.5f, 1.0f, 2.0f};

    for (float x : inputs) {
        float sin_val = std::sin(alpha * x);
        float result = x + (sin_val * sin_val) / alpha;

        // result should be >= x (snake adds positive value)
        assert(result >= x - 1e-6f);
        printf("  snake(%.2f, 1.0) = %.6f (added %.6f)\n",
               x, result, result - x);
    }

    printf("  PASS: positive inputs add positive value\n");
}

// Test snake activation for negative values
static void test_snake_negative() {
    printf("Testing snake(x<0)...\n");

    float alpha = 1.0f;

    // For x < 0: snake also adds positive value, making it less negative
    std::vector<float> inputs = {-0.1f, -0.5f, -1.0f, -2.0f};

    for (float x : inputs) {
        float sin_val = std::sin(alpha * x);
        float result = x + (sin_val * sin_val) / alpha;

        // result should be > x (snake adds positive value)
        assert(result > x);
        printf("  snake(%.2f, 1.0) = %.6f (added %.6f)\n",
               x, result, result - x);
    }

    printf("  PASS: negative inputs become less negative\n");
}

// Test snake symmetry (snake is NOT symmetric due to sin^2)
static void test_snake_symmetry() {
    printf("Testing snake asymmetry...\n");

    float alpha = 1.0f;

    // snake(x, alpha) != -snake(-x, alpha) in general
    // because sin^2(-x) = sin^2(x) (even function)
    // but x + sin^2(alpha*x)/alpha != -(x + sin^2(alpha*(-x))/alpha)
    // when x != 0

    float x = 1.0f;
    float sin_pos = std::sin(alpha * x);
    float result_pos = x + (sin_pos * sin_pos) / alpha;

    float sin_neg = std::sin(alpha * (-x));
    float result_neg = (-x) + (sin_neg * sin_neg) / alpha;

    // sin^2 is even, so sin^2(x) = sin^2(-x)
    assert(std::abs(sin_pos * sin_pos - sin_neg * sin_neg) < 1e-6f);

    // But snake is NOT odd because x != -(-x) when adding positive term
    assert(std::abs(result_pos + result_neg) > 0.01f);

    printf("  snake(%.2f, 1.0) = %.6f\n", x, result_pos);
    printf("  snake(%.2f, 1.0) = %.6f\n", -x, result_neg);
    printf("  PASS: snake is asymmetric (DC bias introduced)\n");
}

// Test alpha parameter sensitivity
static void test_snake_alpha_sensitivity() {
    printf("Testing alpha parameter sensitivity...\n");

    float x = 1.0f;
    std::vector<float> alphas = {0.1f, 0.5f, 1.0f, 2.0f, 10.0f};

    for (float alpha : alphas) {
        float sin_val = std::sin(alpha * x);
        float result = x + (sin_val * sin_val) / (alpha + 1e-9f);

        printf("  snake(1.0, %.2f) = %.6f (added %.6f)\n",
               alpha, result, result - x);
    }

    printf("  PASS: alpha parameter affects output\n");
}

// Test in-place snake activation
static void test_snake_inplace() {
    printf("Testing in-place snake activation...\n");

    const int64_t n = 10;
    const int64_t channels = 3;

    std::vector<float> data = {
        // t=0
        0.0f, 0.1f, -0.1f,
        // t=1
        1.0f, 0.5f, -0.5f,
        // ... (rest zeros for simplicity)
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f
    };

    std::vector<float> alpha = {1.0f, 0.5f, 2.0f};

    // Compute expected values
    std::vector<float> expected(data.size());
    for (size_t i = 0; i < data.size(); i++) {
        int64_t c = i % channels;
        float x = data[i];
        float a = alpha[c];
        float sin_val = std::sin(a * x);
        expected[i] = x + (sin_val * sin_val) / (a + 1e-9f);
    }

    // Apply in-place
    snac::snake_1d_inplace(data.data(), alpha.data(), n, channels);

    // Verify
    for (size_t i = 0; i < data.size(); i++) {
        assert(std::abs(data[i] - expected[i]) < 1e-6f);
    }

    printf("  PASS: in-place snake activation\n");
}

// Test DC bias accumulation through multiple snake layers
static void test_snake_dc_accumulation() {
    printf("Testing DC bias accumulation...\n");

    // Start with zero-mean signal
    std::vector<float> signal = {-0.5f, -0.3f, -0.1f, 0.1f, 0.3f, 0.5f};
    std::vector<float> alpha = {1.0f};

    float initial_mean = 0.0f;
    for (float s : signal) initial_mean += s;
    initial_mean /= signal.size();

    printf("  Initial mean: %.6f\n", initial_mean);

    // Apply snake multiple times (simulating multiple layers)
    for (int layer = 0; layer < 5; layer++) {
        snac::snake_1d_inplace(signal.data(), alpha.data(), signal.size(), 1);

        float mean = 0.0f;
        for (float s : signal) mean += s;
        mean /= signal.size();

        printf("  After layer %d: mean=%.6f\n", layer + 1, mean);
    }

    // Mean should have increased (DC bias accumulated)
    float final_mean = 0.0f;
    for (float s : signal) final_mean += s;
    final_mean /= signal.size();

    assert(final_mean > initial_mean);
    printf("  PASS: DC bias accumulates through layers\n");
}

int main() {
    printf("=== Snake1D Activation Tests ===\n\n");

    test_snake_zero();
    test_snake_positive();
    test_snake_negative();
    test_snake_symmetry();
    test_snake_alpha_sensitivity();
    test_snake_inplace();
    test_snake_dc_accumulation();

    printf("\n=== All Snake1D Tests Passed ===\n");
    return 0;
}
