// SNAC Snake1D Activation Function Header
//
// Snake activation: snake(x, alpha) = x + sin^2(alpha * x) / alpha
// Used in SNAC vocoder to learn periodic patterns (fundamental frequency, harmonics)

#pragma once

#include <vector>
#include <cstdint>

namespace snac {

// In-place Snake1D activation
// data: [n, channels] input/output
// alpha: [channels] learnable parameters
// n: number of time steps
// channels: number of channels
void snake_1d_inplace(float* data, const float* alpha, int64_t n, int64_t channels);

// Out-of-place Snake1D activation (returns new vector)
std::vector<float> snake_1d(const float* input, const float* alpha, int64_t n, int64_t channels);

}  // namespace snac
