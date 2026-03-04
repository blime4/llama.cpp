// Orpheus-TTS Acceptance Tests
//
// This file contains acceptance tests for validating the Orpheus TTS vocoder output.
// These tests are designed to detect common issues like DC bias, frequency distribution
// problems, and amplitude anomalies that indicate incorrect vocoder behavior.
//
// Acceptance Criteria (based on MEMORY.md analysis):
// - neg_ratio: 35-65% (symmetric waveform around zero)
// - dc_bias: |mean| < 0.01 (no DC offset)
// - energy_0_200Hz: < 40% (not all low frequency)
// - peak_amplitude: 0.7-0.95 (proper amplitude, not clipping)
//
// Usage:
//   ./test-orpheus-acceptance                    # Run all tests
//   ./test-orpheus-acceptance test_current.wav   # Test specific WAV file

#include "orpheus-test-utils.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>
#include <sstream>

using namespace orpheus_test;

// ============================================================================
// Acceptance Criteria (from MEMORY.md and investigation)
// ============================================================================

struct AcceptanceConfig {
    // Time domain thresholds
    float neg_ratio_min = 35.0f;
    float neg_ratio_max = 65.0f;
    float dc_bias_max = 0.01f;
    float peak_amplitude_min = 0.7f;
    float peak_amplitude_max = 0.95f;
    float std_dev_min = 0.1f;

    // Frequency domain thresholds
    float energy_0_200Hz_max = 50.0f;  // Speech can have ~46% in this band
    float energy_200_500Hz_min = 10.0f;
    float energy_1000Hz_plus_min = 5.0f;

    // Test configuration
    int min_samples = 256;
    int segment_size = 1000;
};

// ============================================================================
// Test Result Structure
// ============================================================================

struct AcceptanceTestResult {
    std::string test_name;
    bool passed = true;
    std::vector<std::string> failures;
    std::vector<std::string> warnings;
    AudioMetrics metrics;

    void add_failure(const std::string& metric, const std::string& message) {
        passed = false;
        std::ostringstream oss;
        oss << metric << ": " << message;
        failures.push_back(oss.str());
    }

    void add_warning(const std::string& metric, const std::string& message) {
        std::ostringstream oss;
        oss << metric << ": " << message;
        warnings.push_back(oss.str());
    }

    void print_report() const {
        printf("\n=== Acceptance Test: %s ===\n", test_name.c_str());

        // Print metrics
        printf("\nMetrics:\n");
        printf("  Samples: %zu (%.3f seconds @ %d Hz)\n",
               metrics.sample_count, metrics.duration_s, metrics.sample_rate);
        printf("  neg_ratio: %.1f%% (target: 35-65%%)\n", metrics.neg_ratio);
        printf("  dc_bias: %.6f (target: |mean| < 0.01)\n", metrics.dc_bias);
        printf("  peak_amplitude: %.3f (target: 0.7-0.95)\n", metrics.peak_amplitude);
        printf("  std_dev: %.3f (target: > 0.1)\n", metrics.std_dev);
        printf("  Frequency: 0-200Hz=%.1f%%, 200-500Hz=%.1f%%, 500-1000Hz=%.1f%%, 1000+Hz=%.1f%%\n",
               metrics.energy_0_200Hz, metrics.energy_200_500Hz,
               metrics.energy_500_1000Hz, metrics.energy_1000Hz_plus);

        // Print warnings
        if (!warnings.empty()) {
            printf("\nWarnings:\n");
            for (const auto& w : warnings) {
                printf("  [WARN] %s\n", w.c_str());
            }
        }

        // Print failures
        if (!failures.empty()) {
            printf("\nFailures:\n");
            for (const auto& f : failures) {
                printf("  [FAIL] %s\n", f.c_str());
            }
        }

        // Print result
        printf("\nResult: %s\n", passed ? "PASS" : "FAIL");
        printf("============================\n");
    }
};

// ============================================================================
// Acceptance Test Functions
// ============================================================================

AcceptanceTestResult run_acceptance_tests(const std::vector<float>& samples,
                                          int sample_rate,
                                          const std::string& test_name,
                                          const AcceptanceConfig& config = AcceptanceConfig()) {
    AcceptanceTestResult result;
    result.test_name = test_name;
    result.metrics = compute_audio_metrics(samples, sample_rate);

    // Check minimum samples
    if (result.metrics.sample_count < (size_t)config.min_samples) {
        result.add_failure("sample_count", "Insufficient samples for analysis");
        return result;
    }

    // Test 1: Symmetric waveform (neg_ratio)
    if (result.metrics.neg_ratio < config.neg_ratio_min) {
        std::ostringstream oss;
        oss << result.metrics.neg_ratio << "% is below minimum " << config.neg_ratio_min << "%";
        result.add_failure("neg_ratio", oss.str());
    } else if (result.metrics.neg_ratio > config.neg_ratio_max) {
        std::ostringstream oss;
        oss << result.metrics.neg_ratio << "% is above maximum " << config.neg_ratio_max << "%";
        result.add_failure("neg_ratio", oss.str());
    }

    // Test 2: No DC bias
    if (std::abs(result.metrics.dc_bias) > config.dc_bias_max) {
        std::ostringstream oss;
        oss << "|" << result.metrics.dc_bias << "| > " << config.dc_bias_max;
        result.add_failure("dc_bias", oss.str());
    }

    // Test 3: Peak amplitude in range
    if (result.metrics.peak_amplitude < config.peak_amplitude_min) {
        std::ostringstream oss;
        oss << result.metrics.peak_amplitude << " < " << config.peak_amplitude_min << " (signal too quiet)";
        result.add_failure("peak_amplitude", oss.str());
    } else if (result.metrics.peak_amplitude > config.peak_amplitude_max) {
        std::ostringstream oss;
        oss << result.metrics.peak_amplitude << " > " << config.peak_amplitude_max << " (possible clipping)";
        result.add_warning("peak_amplitude", oss.str());
    }

    // Test 4: Standard deviation (signal variance)
    if (result.metrics.std_dev < config.std_dev_min) {
        std::ostringstream oss;
        oss << result.metrics.std_dev << " < " << config.std_dev_min << " (low variance)";
        result.add_warning("std_dev", oss.str());
    }

    // Test 5: Frequency distribution - low frequency not dominant
    if (result.metrics.energy_0_200Hz > config.energy_0_200Hz_max) {
        std::ostringstream oss;
        oss << result.metrics.energy_0_200Hz << "% > " << config.energy_0_200Hz_max << "%";
        result.add_failure("energy_0_200Hz", oss.str());
    }

    // Test 6: Frequency distribution - speech frequencies present
    if (result.metrics.energy_200_500Hz < config.energy_200_500Hz_min) {
        std::ostringstream oss;
        oss << result.metrics.energy_200_500Hz << "% < " << config.energy_200_500Hz_min << "%";
        result.add_warning("energy_200_500Hz", oss.str());
    }

    // Test 7: High frequency content present
    if (result.metrics.energy_1000Hz_plus < config.energy_1000Hz_plus_min) {
        std::ostringstream oss;
        oss << result.metrics.energy_1000Hz_plus << "% < " << config.energy_1000Hz_plus_min << "%";
        result.add_warning("energy_1000Hz_plus", oss.str());
    }

    // Test 8: Clipping detection
    int clip_count = 0;
    for (const auto& s : samples) {
        if (std::abs(std::abs(s) - 1.0f) < 0.001f) {
            clip_count++;
        }
    }
    float clip_ratio = 100.0f * clip_count / samples.size();
    if (clip_ratio > 5.0f) {
        std::ostringstream oss;
        oss << clip_ratio << "% of samples at full scale";
        result.add_warning("clipping", oss.str());
    }

    return result;
}

// Test with synthetic good vocoder output
bool test_synthetic_good_output() {
    printf("\n--- Test: Synthetic Good Vocoder Output ---\n");

    std::vector<float> samples = generate_synthetic_vocoder_output(24000, 24000, 42);
    AcceptanceTestResult result = run_acceptance_tests(samples, 24000, "Synthetic Good Output");
    result.print_report();

    return result.passed;
}

// Test with DC bias signal (should fail)
bool test_dc_bias_detection() {
    printf("\n--- Test: DC Bias Detection (should FAIL) ---\n");

    std::vector<float> samples = generate_synthetic_vocoder_output(24000, 24000, 42);

    // Add DC bias
    for (auto& s : samples) {
        s += 0.3f;
    }

    AcceptanceTestResult result = run_acceptance_tests(samples, 24000, "DC Bias Signal");
    result.print_report();

    // This test should FAIL (detect the DC bias)
    bool detected_dc_bias = !result.passed &&
        std::any_of(result.failures.begin(), result.failures.end(),
            [](const std::string& f) { return f.find("dc_bias") != std::string::npos; });

    printf("\nDC Bias Detection: %s\n", detected_dc_bias ? "CORRECTLY DETECTED" : "FAILED TO DETECT");
    return detected_dc_bias;
}

// Test with low-frequency-only signal (should fail)
bool test_low_frequency_detection() {
    printf("\n--- Test: Low Frequency Detection (should FAIL) ---\n");

    const int sample_rate = 24000;
    std::vector<float> samples(sample_rate);

    // Generate only 50 Hz (low frequency)
    for (int i = 0; i < sample_rate; i++) {
        samples[i] = 0.5f * std::sin(2.0f * (float)M_PI * 50.0f * i / sample_rate);
    }

    AcceptanceTestResult result = run_acceptance_tests(samples, sample_rate, "Low Frequency Only");
    result.print_report();

    // This test should FAIL (detect low frequency dominance)
    bool detected_low_freq = !result.passed &&
        std::any_of(result.failures.begin(), result.failures.end(),
            [](const std::string& f) { return f.find("energy_0_200Hz") != std::string::npos; });

    printf("\nLow Frequency Detection: %s\n", detected_low_freq ? "CORRECTLY DETECTED" : "FAILED TO DETECT");
    return detected_low_freq;
}

// Test WAV file if provided
bool test_wav_file(const std::string& filepath) {
    printf("\n--- Test: WAV File: %s ---\n", filepath.c_str());

    std::vector<float> samples;
    int sample_rate;

    if (!load_wav_file(filepath, samples, sample_rate)) {
        printf("ERROR: Failed to load WAV file: %s\n", filepath.c_str());
        return false;
    }

    printf("Loaded: %zu samples @ %d Hz (%.3f seconds)\n",
           samples.size(), sample_rate, (float)samples.size() / sample_rate);

    AcceptanceTestResult result = run_acceptance_tests(samples, sample_rate, filepath);
    result.print_report();

    return result.passed;
}

// Test golden reference files
bool test_golden_references() {
    printf("\n--- Test: Golden Reference Files ---\n");

    bool all_passed = true;

    // Golden reference files from tests/golden/orpheus/audio/
    struct GoldenRef {
        const char* path;
        const char* name;
        bool required;  // If false, skip if file doesn't exist
    };

    GoldenRef golden_files[] = {
        {"tests/golden/orpheus/audio/vocoder_random_100frames.wav", "Vocoder Random 100 Frames", false},
        {"tests/golden/orpheus/audio/e2e_hello_leah.wav", "E2E Hello Leah", false},
        {"tests/golden/orpheus/audio/e2e_long_tara.wav", "E2E Long Tara", false},
        {"tests/golden/orpheus/audio/e2e_fox_zoe.wav", "E2E Fox Zoe", false},
    };

    int tested = 0;
    int passed = 0;

    for (const auto& ref : golden_files) {
        std::vector<float> samples;
        int sample_rate;

        if (load_wav_file(ref.path, samples, sample_rate)) {
            tested++;
            printf("\nTesting golden reference: %s\n", ref.name);

            AcceptanceTestResult result = run_acceptance_tests(samples, sample_rate, ref.name);
            result.print_report();

            if (result.passed) {
                passed++;
            } else {
                all_passed = false;
            }
        } else if (ref.required) {
            printf("ERROR: Required golden file not found: %s\n", ref.path);
            all_passed = false;
        }
    }

    if (tested == 0) {
        printf("No golden reference files found. Skipping golden tests.\n");
        printf("Golden files should be in tests/golden/orpheus/audio/\n");
    } else {
        printf("\nGolden reference tests: %d/%d passed\n", passed, tested);
    }

    return all_passed;
}

// Test golden token sequences
void test_golden_sequences() {
    printf("\n--- Test: Golden Token Sequences ---\n");

    auto sequences = get_golden_token_sequences(100);  // 100 frames
    print_golden_sequences_info(sequences);

    printf("\nNote: Golden token sequences require actual vocoder to decode.\n");
    printf("These sequences are for integration testing with the full pipeline.\n");
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* program) {
    printf("Usage: %s [options] [wav_file]\n", program);
    printf("\n");
    printf("Orpheus-TTS Acceptance Tests\n");
    printf("\n");
    printf("Options:\n");
    printf("  --help          Show this help message\n");
    printf("  --all           Run all built-in tests\n");
    printf("  --synthetic     Test synthetic good vocoder output\n");
    printf("  --dc-bias       Test DC bias detection (should fail)\n");
    printf("  --low-freq      Test low frequency detection (should fail)\n");
    printf("  --golden-refs   Test golden reference WAV files\n");
    printf("  --golden-tokens Show golden token sequences info\n");
    printf("  wav_file        Test a specific WAV file\n");
    printf("\n");
    printf("Acceptance Criteria:\n");
    printf("  neg_ratio: 35-65%% (symmetric waveform)\n");
    printf("  dc_bias: |mean| < 0.01 (no DC offset)\n");
    printf("  energy_0_200Hz: < 40%% (not all low frequency)\n");
    printf("  peak_amplitude: 0.7-0.95 (proper amplitude)\n");
    printf("\n");
    printf("Golden Reference Files:\n");
    printf("  tests/golden/orpheus/audio/vocoder_random_100frames.wav\n");
    printf("  tests/golden/orpheus/audio/e2e_hello_leah.wav\n");
    printf("  tests/golden/orpheus/audio/e2e_long_tara.wav\n");
    printf("  tests/golden/orpheus/audio/e2e_fox_zoe.wav\n");
    printf("\n");
    printf("Exit codes:\n");
    printf("  0 - All tests passed\n");
    printf("  1 - One or more tests failed\n");
}

int main(int argc, char* argv[]) {
    printf("=== Orpheus-TTS Acceptance Tests ===\n");

    bool run_all = false;
    bool run_synthetic = false;
    bool run_dc_bias = false;
    bool run_low_freq = false;
    bool run_golden_refs = false;
    bool run_golden_tokens = false;
    std::string wav_file;

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "--all") {
            run_all = true;
        } else if (arg == "--synthetic") {
            run_synthetic = true;
        } else if (arg == "--dc-bias") {
            run_dc_bias = true;
        } else if (arg == "--low-freq") {
            run_low_freq = true;
        } else if (arg == "--golden-refs") {
            run_golden_refs = true;
        } else if (arg == "--golden-tokens") {
            run_golden_tokens = true;
        } else if (arg == "--golden") {
            // Backward compatibility: run both golden tests
            run_golden_refs = true;
            run_golden_tokens = true;
        } else if (arg[0] != '-') {
            wav_file = arg;
        } else {
            printf("Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }

    // Default: run all tests if no specific tests selected
    if (!run_synthetic && !run_dc_bias && !run_low_freq && !run_golden_refs &&
        !run_golden_tokens && wav_file.empty()) {
        run_all = true;
    }

    bool all_passed = true;

    // Run selected tests
    if (run_all || run_synthetic) {
        if (!test_synthetic_good_output()) {
            all_passed = false;
        }
    }

    if (run_all || run_dc_bias) {
        if (!test_dc_bias_detection()) {
            all_passed = false;
        }
    }

    if (run_all || run_low_freq) {
        if (!test_low_frequency_detection()) {
            all_passed = false;
        }
    }

    if (run_all || run_golden_refs) {
        if (!test_golden_references()) {
            all_passed = false;
        }
    }

    if (run_all || run_golden_tokens) {
        test_golden_sequences();
    }

    // Test WAV file if provided
    if (!wav_file.empty()) {
        if (!test_wav_file(wav_file)) {
            all_passed = false;
        }
    }

    // Summary
    printf("\n=== Summary ===\n");
    printf("Result: %s\n", all_passed ? "ALL TESTS PASSED" : "SOME TESTS FAILED");

    return all_passed ? 0 : 1;
}
