// SNAC Snake1D Activation Function
//
// Snake activation is used in SNAC vocoder to learn periodic patterns:
//   snake(x, alpha) = x + sin^2(alpha * x) / alpha
//
// Key properties:
// - Always adds positive values (sin^2 >= 0)
// - For negative x: makes it less negative (closer to 0)
// - For positive x: makes it more positive
// - DC bias accumulates through multiple snake layers

#include "snac-snake.h"
#include <cmath>
#include <vector>

namespace snac {

void snake_1d_inplace(float* data, const float* alpha, int64_t n, int64_t channels) {
    for (int64_t i = 0; i < n; i++) {
        for (int64_t c = 0; c < channels; c++) {
            float x = data[i * channels + c];
            float a = alpha[c];
            float sin_val = std::sin(a * x);
            data[i * channels + c] = x + (sin_val * sin_val) / (a + 1e-9f);
        }
    }
}

std::vector<float> snake_1d(const float* input, const float* alpha, int64_t n, int64_t channels) {
    std::vector<float> output(n * channels);

    for (int64_t i = 0; i < n; i++) {
        for (int64_t c = 0; c < channels; c++) {
            float x = input[i * channels + c];
            float a = alpha[c];
            float sin_val = std::sin(a * x);
            output[i * channels + c] = x + (sin_val * sin_val) / (a + 1e-9f);
        }
    }

    return output;
}

}  // namespace snac
