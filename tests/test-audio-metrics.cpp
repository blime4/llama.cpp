// Test Audio Quality Metrics for Orpheus-TTS
//
// This test validates the audio quality measurement functions
// used to verify correct SNAC vocoder output.

#include "orpheus-test-utils.h"

#include <cstdio>
#include <cmath>
#include <cassert>

using namespace orpheus_test;

// Test metrics for silence (all zeros)
static void test_metrics_silence() {
    printf("Testing silence metrics...\n");

    std::vector<float> samples(1000, 0.0f);
    AudioMetrics m = compute_audio_metrics(samples, 24000);

    assert(std::abs(m.mean) < 1e-6f);
    assert(std::abs(m.dc_bias) < 1e-6f);
    assert(std::abs(m.min_val) < 1e-6f);
    assert(std::abs(m.max_val) < 1e-6f);
    assert(std::abs(m.peak_amplitude) < 1e-6f);
    assert(std::abs(m.neg_ratio) < 1e-6f);  // No negative samples

    printf("  PASS: silence metrics\n");
}

// Test metrics for DC bias signal
static void test_metrics_dc_bias() {
    printf("Testing DC bias detection...\n");

    // Positive DC bias
    {
        std::vector<float> samples_positive(1000, 0.5f);
        AudioMetrics m_pos = compute_audio_metrics(samples_positive, 24000);

        assert(std::abs(m_pos.mean - 0.5f) < 0.001f);
        assert(std::abs(m_pos.dc_bias - 0.5f) < 0.001f);
        assert(std::abs(m_pos.neg_ratio) < 0.001f);  // All positive
    }

    // Negative DC bias
    {
        std::vector<float> samples_negative(1000, -0.3f);
        AudioMetrics m_neg = compute_audio_metrics(samples_negative, 24000);

        assert(std::abs(m_neg.mean - (-0.3f)) < 0.001f);
        assert(std::abs(m_neg.dc_bias - (-0.3f)) < 0.001f);
        assert(std::abs(m_neg.neg_ratio - 100.0f) < 0.001f);  // All negative
    }

    printf("  PASS: DC bias detection\n");
}

// Test metrics for sine wave
static void test_metrics_sine_wave() {
    printf("Testing sine wave metrics...\n");

    // Generate 440 Hz sine wave
    const int sample_rate = 24000;
    const float freq = 440.0f;
    const int n_samples = sample_rate;  // 1 second

    std::vector<float> samples(n_samples);
    for (int i = 0; i < n_samples; i++) {
        samples[i] = std::sin(2.0f * (float)M_PI * freq * i / sample_rate);
    }

    AudioMetrics m = compute_audio_metrics(samples, sample_rate);

    // Sine wave should have ~50% negative samples
    assert(m.neg_ratio > 45.0f && m.neg_ratio < 55.0f);

    // Mean should be near 0
    assert(std::abs(m.mean) < 0.01f);

    // Peak should be near 1.0
    assert(m.peak_amplitude > 0.99f && m.peak_amplitude < 1.01f);

    printf("  PASS: sine wave metrics (neg_ratio=%.1f%%, mean=%.6f)\n",
           m.neg_ratio, m.mean);
}

// Test quality validation
static void test_quality_validation() {
    printf("Testing quality validation...\n");

    // Good signal
    {
        std::vector<float> samples(1000);
        for (size_t i = 0; i < samples.size(); i++) {
            samples[i] = 0.8f * std::sin(2.0f * (float)M_PI * i / 100);
        }
        AudioMetrics m = compute_audio_metrics(samples, 24000);
        assert(validate_audio_quality(m, true));
    }
}

int main() {
    printf("=== Audio Metrics Tests ===\n\n");

    test_metrics_silence();
    test_metrics_dc_bias();
    test_metrics_sine_wave();
    test_quality_validation();

    printf("\n=== All Audio Metrics Tests Passed ===\n");
    return 0;
}
