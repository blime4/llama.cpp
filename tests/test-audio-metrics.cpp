// Test Audio Quality Metrics for Orpheus-TTS
//
// This test validates the audio quality measurement functions
// used to verify correct SNAC vocoder output.
//
// Acceptance Tests:
// - test_vocoder_output_quality: Validates vocoder output against quality thresholds
// - test_frequency_distribution: Validates proper frequency distribution in output
// - test_symmetric_waveform: Validates symmetric positive/negative samples
// - test_no_dc_offset: Validates absence of DC bias in output

#include "orpheus-test-utils.h"

#include <cstdio>
#include <cmath>
#include <cassert>
#include <map>
#include <sstream>

using namespace orpheus_test;

// ============================================================================
// Acceptance Criteria Thresholds (from MEMORY.md and issue investigation)
// ============================================================================
namespace acceptance_criteria {

// Time domain criteria
constexpr float NEG_RATIO_MIN = 35.0f;        // Symmetric waveform: 35-65%
constexpr float NEG_RATIO_MAX = 65.0f;
constexpr float DC_BIAS_MAX = 0.01f;          // No DC offset: |mean| < 0.01
constexpr float PEAK_AMPLITUDE_MIN = 0.7f;    // Proper amplitude: 0.7-0.95
constexpr float PEAK_AMPLITUDE_MAX = 0.95f;
constexpr float STD_DEV_MIN = 0.1f;           // Reasonable variance

// Frequency domain criteria
constexpr float ENERGY_0_200HZ_MAX = 50.0f;   // Speech can have ~46% in this band
constexpr float ENERGY_200_500HZ_MIN = 10.0f; // Speech frequency range present
constexpr float ENERGY_1000HZ_PLUS_MIN = 5.0f; // High frequency content present

// Test tolerance for floating point comparisons
constexpr float TOLERANCE = 0.001f;

}  // namespace acceptance_criteria

// ============================================================================
// Test Result Reporting
// ============================================================================

struct AcceptanceResult {
    bool passed = true;
    std::map<std::string, std::string> failures;

    void fail(const std::string& metric, const std::string& reason) {
        passed = false;
        failures[metric] = reason;
    }

    void report(const std::string& test_name) const {
        if (passed) {
            printf("  [PASS] %s\n", test_name.c_str());
        } else {
            printf("  [FAIL] %s\n", test_name.c_str());
            for (const auto& entry : failures) {
                const std::string& metric = entry.first;
                const std::string& reason = entry.second;
                printf("    - %s: %s\n", metric.c_str(), reason.c_str());
            }
        }
    }
};

// ============================================================================
// Basic Unit Tests (existing)
// ============================================================================

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

// ============================================================================
// Acceptance Tests (new)
// ============================================================================

// Acceptance Test 1: Validate vocoder output quality metrics
// This test simulates the expected output from a correctly functioning vocoder
// and validates that the audio metrics fall within acceptable ranges.
static AcceptanceResult test_vocoder_output_quality(const std::vector<float>& samples) {
    AcceptanceResult result;
    namespace ac = acceptance_criteria;

    AudioMetrics m = compute_audio_metrics(samples, 24000);

    // Check sample count (must be sufficient for analysis)
    if (m.sample_count < 256) {
        result.fail("sample_count", "Insufficient samples for analysis (need >= 256)");
        return result;
    }

    // Check neg_ratio (symmetric waveform)
    if (m.neg_ratio < ac::NEG_RATIO_MIN) {
        std::ostringstream oss;
        oss << "neg_ratio=" << m.neg_ratio << "% is too low (min=" << ac::NEG_RATIO_MIN << "%) - DC bias positive";
        result.fail("neg_ratio", oss.str());
    } else if (m.neg_ratio > ac::NEG_RATIO_MAX) {
        std::ostringstream oss;
        oss << "neg_ratio=" << m.neg_ratio << "% is too high (max=" << ac::NEG_RATIO_MAX << "%) - DC bias negative";
        result.fail("neg_ratio", oss.str());
    }

    // Check DC bias
    if (std::abs(m.dc_bias) > ac::DC_BIAS_MAX) {
        std::ostringstream oss;
        oss << "dc_bias=" << m.dc_bias << " exceeds threshold " << ac::DC_BIAS_MAX;
        result.fail("dc_bias", oss.str());
    }

    // Check peak amplitude
    if (m.peak_amplitude < ac::PEAK_AMPLITUDE_MIN) {
        std::ostringstream oss;
        oss << "peak_amplitude=" << m.peak_amplitude << " is too low (min=" << ac::PEAK_AMPLITUDE_MIN << ")";
        result.fail("peak_amplitude", oss.str());
    } else if (m.peak_amplitude > ac::PEAK_AMPLITUDE_MAX) {
        std::ostringstream oss;
        oss << "peak_amplitude=" << m.peak_amplitude << " is too high (max=" << ac::PEAK_AMPLITUDE_MAX << ") - possible clipping";
        result.fail("peak_amplitude", oss.str());
    }

    // Check standard deviation (signal should have reasonable variance)
    if (m.std_dev < ac::STD_DEV_MIN) {
        std::ostringstream oss;
        oss << "std_dev=" << m.std_dev << " is too low (min=" << ac::STD_DEV_MIN << ") - signal too quiet";
        result.fail("std_dev", oss.str());
    }

    // Print metrics for debugging
    printf("    Metrics: neg_ratio=%.1f%%, dc_bias=%.6f, peak=%.3f, std_dev=%.3f\n",
           m.neg_ratio, m.dc_bias, m.peak_amplitude, m.std_dev);

    return result;
}

// Acceptance Test 2: Validate frequency distribution
// Ensures the vocoder output has a proper frequency distribution,
// not concentrated entirely in low frequencies (which indicates DC bias or noise).
static AcceptanceResult test_frequency_distribution(const std::vector<float>& samples) {
    AcceptanceResult result;
    namespace ac = acceptance_criteria;

    AudioMetrics m = compute_audio_metrics(samples, 24000);

    // Check low frequency energy (should not dominate)
    if (m.energy_0_200Hz > ac::ENERGY_0_200HZ_MAX) {
        std::ostringstream oss;
        oss << "energy_0_200Hz=" << m.energy_0_200Hz << "% is too high (max=" << ac::ENERGY_0_200HZ_MAX << "%) - low frequency noise dominant";
        result.fail("energy_0_200Hz", oss.str());
    }

    // Check mid-low frequency energy (should be present)
    if (m.energy_200_500Hz < ac::ENERGY_200_500HZ_MIN) {
        std::ostringstream oss;
        oss << "energy_200_500Hz=" << m.energy_200_500Hz << "% is too low (min=" << ac::ENERGY_200_500HZ_MIN << "%) - speech frequencies missing";
        result.fail("energy_200_500Hz", oss.str());
    }

    // Check high frequency energy (should be present)
    if (m.energy_1000Hz_plus < ac::ENERGY_1000HZ_PLUS_MIN) {
        std::ostringstream oss;
        oss << "energy_1000Hz_plus=" << m.energy_1000Hz_plus << "% is too low (min=" << ac::ENERGY_1000HZ_PLUS_MIN << "%) - high frequencies missing";
        result.fail("energy_1000Hz_plus", oss.str());
    }

    // Print frequency distribution for debugging
    printf("    Frequency: 0-200Hz=%.1f%%, 200-500Hz=%.1f%%, 500-1000Hz=%.1f%%, 1000+Hz=%.1f%%\n",
           m.energy_0_200Hz, m.energy_200_500Hz, m.energy_500_1000Hz, m.energy_1000Hz_plus);

    return result;
}

// Acceptance Test 3: Validate symmetric waveform (no DC bias)
// A healthy audio signal should have roughly equal positive and negative samples.
static AcceptanceResult test_symmetric_waveform(const std::vector<float>& samples) {
    AcceptanceResult result;
    namespace ac = acceptance_criteria;

    // Count positive and negative samples
    int positive_count = 0;
    int negative_count = 0;
    int zero_count = 0;

    for (const auto& s : samples) {
        if (s > ac::TOLERANCE) {
            positive_count++;
        } else if (s < -ac::TOLERANCE) {
            negative_count++;
        } else {
            zero_count++;
        }
    }

    int total = positive_count + negative_count;
    if (total == 0) {
        result.fail("symmetry", "All samples are zero");
        return result;
    }

    float positive_ratio = 100.0f * positive_count / samples.size();
    float negative_ratio = 100.0f * negative_count / samples.size();

    // Check that positive and negative ratios are roughly equal
    float ratio_diff = std::abs(positive_ratio - negative_ratio);
    if (ratio_diff > 30.0f) {  // Allow up to 30% difference
        std::ostringstream oss;
        oss << "Waveform asymmetry: positive=" << positive_ratio << "%, negative=" << negative_ratio << "% (diff=" << ratio_diff << "%)";
        result.fail("symmetry", oss.str());
    }

    printf("    Symmetry: positive=%.1f%%, negative=%.1f%%, zero=%.1f%%\n",
           positive_ratio, negative_ratio, 100.0f * zero_count / samples.size());

    return result;
}

// Acceptance Test 4: Validate no DC offset using running average
// Computes running average to detect DC drift over time.
static AcceptanceResult test_no_dc_offset(const std::vector<float>& samples) {
    AcceptanceResult result;
    namespace ac = acceptance_criteria;

    if (samples.size() < 1000) {
        result.fail("dc_offset", "Need at least 1000 samples for DC offset test");
        return result;
    }

    // Compute running averages over segments
    const int segment_size = 1000;
    int n_segments = samples.size() / segment_size;

    float max_dc = 0.0f;
    int worst_segment = 0;

    for (int seg = 0; seg < n_segments; seg++) {
        float sum = 0.0f;
        for (int i = 0; i < segment_size; i++) {
            sum += samples[seg * segment_size + i];
        }
        float avg = sum / segment_size;

        if (std::abs(avg) > std::abs(max_dc)) {
            max_dc = avg;
            worst_segment = seg;
        }
    }

    if (std::abs(max_dc) > ac::DC_BIAS_MAX) {
        std::ostringstream oss;
        oss << "DC offset detected in segment " << worst_segment << ": " << max_dc << " (max allowed=" << ac::DC_BIAS_MAX << ")";
        result.fail("dc_offset", oss.str());
    }

    printf("    DC Offset: max=%.6f in segment %d\n", max_dc, worst_segment);

    return result;
}

// Acceptance Test 5: Validate amplitude distribution
// Checks that the amplitude distribution is reasonable (not all same value, not clipping).
static AcceptanceResult test_amplitude_distribution(const std::vector<float>& samples) {
    AcceptanceResult result;

    // Check for clipping (many samples at exactly +/- 1.0)
    int clip_count = 0;
    for (const auto& s : samples) {
        if (std::abs(std::abs(s) - 1.0f) < 0.001f) {
            clip_count++;
        }
    }

    float clip_ratio = 100.0f * clip_count / samples.size();
    if (clip_ratio > 5.0f) {  // More than 5% clipped is problematic
        std::ostringstream oss;
        oss << "Clipping detected: " << clip_ratio << "% of samples at full scale";
        result.fail("clipping", oss.str());
    }

    // Check for quantization (many samples at the same values)
    std::map<int, int> histogram;  // Quantized to 1000 levels
    for (const auto& s : samples) {
        int bin = static_cast<int>(s * 500 + 500);  // Map [-1,1] to [0,1000]
        histogram[bin]++;
    }

    // Find the most common value
    int max_count = 0;
    for (const auto& entry : histogram) {
        int count = entry.second;
        if (count > max_count) {
            max_count = count;
        }
    }

    float concentration = 100.0f * max_count / samples.size();
    if (concentration > 10.0f) {  // More than 10% at same value is suspicious
        std::ostringstream oss;
        oss << "Amplitude concentration: " << concentration << "% at same value";
        result.fail("amplitude_distribution", oss.str());
    }

    printf("    Amplitude: clipping=%.1f%%, max_concentration=%.1f%%\n",
           clip_ratio, concentration);

    return result;
}

// ============================================================================
// Golden Test Data Generation
// ============================================================================

// Generate a synthetic "good" vocoder output for testing
// This simulates what a correctly functioning vocoder should produce.
static std::vector<float> generate_synthetic_good_vocoder_output(int n_samples, uint64_t seed = 42) {
    std::vector<float> samples(n_samples);
    std::mt19937 rng(seed);

    // Generate a complex waveform that simulates speech characteristics
    // Combination of multiple frequencies with varying amplitudes
    const int sample_rate = 24000;

    for (int i = 0; i < n_samples; i++) {
        float t = static_cast<float>(i) / sample_rate;

        // Fundamental frequency (~150 Hz for male speech)
        float fundamental = std::sin(2.0f * (float)M_PI * 150.0f * t);

        // Harmonics
        float h2 = 0.5f * std::sin(2.0f * (float)M_PI * 300.0f * t);
        float h3 = 0.3f * std::sin(2.0f * (float)M_PI * 450.0f * t);
        float h4 = 0.2f * std::sin(2.0f * (float)M_PI * 600.0f * t);

        // Formant-like resonance
        float formant = 0.4f * std::sin(2.0f * (float)M_PI * 800.0f * t);

        // Add some noise for naturalness
        std::uniform_real_distribution<float> noise_dist(-0.05f, 0.05f);
        float noise = noise_dist(rng);

        // Amplitude modulation (like syllables)
        float envelope = 0.7f + 0.3f * std::sin(2.0f * (float)M_PI * 3.0f * t);

        // Combine
        samples[i] = envelope * (fundamental + h2 + h3 + h4 + formant + noise);

        // Scale to typical vocoder output range
        samples[i] *= 0.8f;
    }

    return samples;
}

// Generate a "bad" vocoder output with DC bias (for negative testing)
static std::vector<float> generate_bad_vocoder_output_dc_bias(int n_samples) {
    std::vector<float> samples = generate_synthetic_good_vocoder_output(n_samples);

    // Add DC bias
    for (auto& s : samples) {
        s += 0.3f;  // Significant DC offset
    }

    return samples;
}

// Generate a "bad" vocoder output with all low frequencies (for negative testing)
static std::vector<float> generate_bad_vocoder_output_low_freq(int n_samples) {
    std::vector<float> samples(n_samples);
    const int sample_rate = 24000;

    // Only low frequency content
    for (int i = 0; i < n_samples; i++) {
        float t = static_cast<float>(i) / sample_rate;
        samples[i] = 0.5f * std::sin(2.0f * (float)M_PI * 50.0f * t);  // 50 Hz only
    }

    return samples;
}

// ============================================================================
// Main Test Runner
// ============================================================================

// Run all acceptance tests on a given audio sample
static void run_acceptance_tests(const std::vector<float>& samples, const std::string& test_case_name) {
    printf("\n--- Acceptance Tests: %s ---\n", test_case_name.c_str());

    AcceptanceResult r1 = test_vocoder_output_quality(samples);
    r1.report("vocoder_output_quality");

    AcceptanceResult r2 = test_frequency_distribution(samples);
    r2.report("frequency_distribution");

    AcceptanceResult r3 = test_symmetric_waveform(samples);
    r3.report("symmetric_waveform");

    AcceptanceResult r4 = test_no_dc_offset(samples);
    r4.report("no_dc_offset");

    AcceptanceResult r5 = test_amplitude_distribution(samples);
    r5.report("amplitude_distribution");

    // Overall result
    bool all_passed = r1.passed && r2.passed && r3.passed && r4.passed && r5.passed;
    printf("  Overall: %s\n\n", all_passed ? "ALL PASSED" : "SOME FAILED");
}

int main() {
    printf("=== Audio Metrics Tests ===\n\n");

    // Run basic unit tests
    test_metrics_silence();
    test_metrics_dc_bias();
    test_metrics_sine_wave();
    test_quality_validation();

    printf("\n=== Acceptance Tests ===\n");

    // Test 1: Synthetic good vocoder output (should pass all tests)
    {
        printf("\n[Acceptance Test 1] Synthetic good vocoder output\n");
        std::vector<float> samples = generate_synthetic_good_vocoder_output(24000);  // 1 second
        run_acceptance_tests(samples, "Good Vocoder Output (synthetic)");
    }

    // Test 2: Bad output with DC bias (should fail DC bias tests)
    {
        printf("\n[Acceptance Test 2] Bad vocoder output with DC bias\n");
        std::vector<float> samples = generate_bad_vocoder_output_dc_bias(24000);
        run_acceptance_tests(samples, "Bad Output - DC Bias");
    }

    // Test 3: Bad output with all low frequencies (should fail frequency tests)
    {
        printf("\n[Acceptance Test 3] Bad vocoder output with low frequencies only\n");
        std::vector<float> samples = generate_bad_vocoder_output_low_freq(24000);
        run_acceptance_tests(samples, "Bad Output - Low Frequency Only");
    }

    // Test 4: 440 Hz sine wave (should pass basic tests)
    {
        printf("\n[Acceptance Test 4] 440 Hz sine wave (known good signal)\n");
        const int sample_rate = 24000;
        std::vector<float> samples(sample_rate);
        for (int i = 0; i < sample_rate; i++) {
            samples[i] = 0.8f * std::sin(2.0f * (float)M_PI * 440.0f * i / sample_rate);
        }
        run_acceptance_tests(samples, "440 Hz Sine Wave");
    }

    // Test 5: Load and test external WAV file if provided
    // (This allows testing actual vocoder output files)
    {
        printf("\n[Acceptance Test 5] External WAV file test (if available)\n");
        std::vector<float> samples;
        int sample_rate;
        // Try common test output file names
        const char* test_files[] = {
            "test_current.wav",
            "test_vocoder_output.wav",
            "test_speech.wav"
        };

        bool loaded = false;
        for (const char* fname : test_files) {
            if (load_wav_file(fname, samples, sample_rate)) {
                printf("  Loaded: %s (%zu samples @ %d Hz)\n", fname, samples.size(), sample_rate);
                run_acceptance_tests(samples, std::string("External File: ") + fname);
                loaded = true;
                break;
            }
        }

        if (!loaded) {
            printf("  No external WAV file found (skipped)\n");
            printf("  To test vocoder output, place a file named test_current.wav in the working directory\n");
        }
    }

    printf("\n=== All Audio Metrics Tests Completed ===\n");
    return 0;
}
