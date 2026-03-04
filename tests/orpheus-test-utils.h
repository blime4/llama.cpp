// Orpheus-TTS Test Utilities
// Shared utilities and golden data for Orpheus TTS tests
//
// This header provides:
// - Audio quality metrics computation
// - Golden token sequence generation
// - WAV file I/O utilities
// - Test comparison helpers

#pragma once

#include <vector>
#include <string>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <algorithm>  // for std::clamp

namespace orpheus_test {

// ============================================================================
// Audio Quality Metrics
// ============================================================================

struct AudioMetrics {
    // Time domain metrics
    float mean = 0.0f;
    float std_dev = 0.0f;
    float min_val = 0.0f;
    float max_val = 0.0f;
    float neg_ratio = 0.0f;      // Percentage of negative samples (target: 35-65%)
    float dc_bias = 0.0f;        // Mean value (target: |mean| < 0.01)
    float peak_amplitude = 0.0f; // Peak absolute value (target: 0.7-0.95)

    // Frequency domain metrics (energy distribution)
    float energy_0_200Hz = 0.0f;     // Low frequency energy (target: <40%)
    float energy_200_500Hz = 0.0f;
    float energy_500_1000Hz = 0.0f;
    float energy_1000Hz_plus = 0.0f;

    // Metadata
    size_t sample_count = 0;
    int sample_rate = 24000;
    float duration_s = 0.0f;
};

// Compute audio quality metrics from samples
inline AudioMetrics compute_audio_metrics(const std::vector<float>& samples, int sample_rate = 24000) {
    AudioMetrics m;
    m.sample_count = samples.size();
    m.sample_rate = sample_rate;
    m.duration_s = (float)samples.size() / sample_rate;

    if (samples.empty()) {
        return m;
    }

    // Compute time domain metrics
    float sum = 0.0f;
    float min_v = samples[0];
    float max_v = samples[0];
    int neg_count = 0;

    for (const auto& s : samples) {
        sum += s;
        min_v = std::min(min_v, s);
        max_v = std::max(max_v, s);
        if (s < 0) neg_count++;
    }

    m.mean = sum / samples.size();
    m.min_val = min_v;
    m.max_val = max_v;
    m.dc_bias = m.mean;
    m.neg_ratio = 100.0f * neg_count / samples.size();

    // Compute standard deviation
    float var_sum = 0.0f;
    for (const auto& s : samples) {
        float d = s - m.mean;
        var_sum += d * d;
    }
    m.std_dev = std::sqrt(var_sum / samples.size());

    // Peak amplitude
    m.peak_amplitude = std::max(std::abs(min_v), std::abs(max_v));

    // Compute frequency distribution using simple DFT (for test purposes)
    // This is a simplified approach - use FFT for production
    if (samples.size() >= 256) {
        const int n_bins = 256;
        std::vector<float> magnitude(n_bins / 2, 0.0f);

        // Simple DFT for first N/2 frequency bins
        for (int k = 0; k < n_bins / 2; k++) {
            float freq = (float)k * sample_rate / n_bins;
            float cos_sum = 0.0f, sin_sum = 0.0f;

            for (size_t n = 0; n < std::min(samples.size(), (size_t)n_bins); n++) {
                float angle = 2.0f * M_PI * k * n / n_bins;
                cos_sum += samples[n] * std::cos(angle);
                sin_sum += samples[n] * std::sin(angle);
            }
            magnitude[k] = std::sqrt(cos_sum * cos_sum + sin_sum * sin_sum);
        }

        // Compute energy in frequency bands
        float total_energy = 0.0f;
        float e_0_200 = 0.0f, e_200_500 = 0.0f, e_500_1000 = 0.0f, e_1000_plus = 0.0f;

        for (int k = 1; k < n_bins / 2; k++) {  // Skip DC (k=0)
            float freq = (float)k * sample_rate / n_bins;
            float energy = magnitude[k] * magnitude[k];
            total_energy += energy;

            if (freq < 200) e_0_200 += energy;
            else if (freq < 500) e_200_500 += energy;
            else if (freq < 1000) e_500_1000 += energy;
            else e_1000_plus += energy;
        }

        if (total_energy > 0) {
            m.energy_0_200Hz = 100.0f * e_0_200 / total_energy;
            m.energy_200_500Hz = 100.0f * e_200_500 / total_energy;
            m.energy_500_1000Hz = 100.0f * e_500_1000 / total_energy;
            m.energy_1000Hz_plus = 100.0f * e_1000_plus / total_energy;
        }
    }

    return m;
}

// Validate audio quality against expected ranges
inline bool validate_audio_quality(const AudioMetrics& m, bool strict = false) {
    bool pass = true;

    // neg_ratio should be 35-65% (symmetric waveform)
    if (m.neg_ratio < 35.0f || m.neg_ratio > 65.0f) {
        if (strict) pass = false;
    }

    // DC bias should be small
    if (std::abs(m.dc_bias) > 0.01f) {
        if (strict) pass = false;
    }

    // Low frequency energy should not dominate
    if (m.energy_0_200Hz > 40.0f) {
        if (strict) pass = false;
    }

    // Peak amplitude should be reasonable (not too quiet, not clipping)
    if (m.peak_amplitude < 0.1f || m.peak_amplitude > 1.0f) {
        pass = false;
    }

    return pass;
}

// Print audio quality report
inline void print_audio_report(const AudioMetrics& m, const std::string& label = "Audio") {
    printf("=== %s Quality Report ===\n", label.c_str());
    printf("Samples: %zu (%.3f seconds @ %d Hz)\n", m.sample_count, m.duration_s, m.sample_rate);
    printf("Time Domain:\n");
    printf("  mean=%.6f (dc_bias=%.6f)\n", m.mean, m.dc_bias);
    printf("  std_dev=%.6f\n", m.std_dev);
    printf("  range=[%.6f, %.6f]\n", m.min_val, m.max_val);
    printf("  peak=%.6f\n", m.peak_amplitude);
    printf("  neg_ratio=%.1f%% (target: 35-65%%)\n", m.neg_ratio);
    printf("Frequency Distribution:\n");
    printf("  0-200 Hz:   %.1f%% (target: <40%%)\n", m.energy_0_200Hz);
    printf("  200-500 Hz: %.1f%%\n", m.energy_200_500Hz);
    printf("  500-1000 Hz:%.1f%%\n", m.energy_500_1000Hz);
    printf("  1000+ Hz:   %.1f%%\n", m.energy_1000Hz_plus);
    printf("Quality: %s\n", validate_audio_quality(m, true) ? "PASS" : "FAIL");
    printf("============================\n");
}

// ============================================================================
// Golden Token Sequences
// ============================================================================

// Generate deterministic token sequences for testing
// These provide reproducible test inputs for the vocoder
inline std::vector<std::vector<int>> generate_deterministic_tokens(int n_frames, uint64_t seed = 42) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(0, 4095);

    // SNAC pyramid structure for TTS:
    // head0: 4 tokens per frame (stride 4)
    // head1: 2 tokens per frame (stride 2)
    // head2: 1 token per frame  (stride 1)
    std::vector<std::vector<int>> tokens(3);
    tokens[0].reserve(n_frames * 4);
    tokens[1].reserve(n_frames * 2);
    tokens[2].reserve(n_frames);

    for (int f = 0; f < n_frames; f++) {
        // head0: 4 tokens
        for (int j = 0; j < 4; j++) {
            tokens[0].push_back(dist(rng));
        }
        // head1: 2 tokens
        for (int j = 0; j < 2; j++) {
            tokens[1].push_back(dist(rng));
        }
        // head2: 1 token
        tokens[2].push_back(dist(rng));
    }

    return tokens;
}

// Generate zeros token sequence
inline std::vector<std::vector<int>> generate_zeros_tokens(int n_frames) {
    std::vector<std::vector<int>> tokens(3);
    tokens[0].resize(n_frames * 4, 0);
    tokens[1].resize(n_frames * 2, 0);
    tokens[2].resize(n_frames, 0);
    return tokens;
}

// Generate sequential token sequence (0, 1, 2, ..., 4095, 0, 1, ...)
inline std::vector<std::vector<int>> generate_sequential_tokens(int n_frames) {
    std::vector<std::vector<int>> tokens(3);
    int idx = 0;

    for (int f = 0; f < n_frames; f++) {
        for (int j = 0; j < 4; j++) {
            tokens[0].push_back((idx++) % 4096);
        }
        for (int j = 0; j < 2; j++) {
            tokens[1].push_back((idx++) % 4096);
        }
        tokens[2].push_back((idx++) % 4096);
    }

    return tokens;
}

// ============================================================================
// WAV File I/O
// ============================================================================

struct wav_header {
    char riff[4] = {'R', 'I', 'F', 'F'};
    uint32_t chunk_size;
    char wave[4] = {'W', 'A', 'V', 'E'};
    char fmt[4] = {'f', 'm', 't', ' '};
    uint32_t fmt_chunk_size = 16;
    uint16_t audio_format = 1;  // PCM
    uint16_t num_channels = 1;  // Mono
    uint32_t sample_rate;
    uint32_t byte_rate;
    uint16_t block_align;
    uint16_t bits_per_sample = 16;
    char data[4] = {'d', 'a', 't', 'a'};
    uint32_t data_size;
};

inline bool load_wav_file(const std::string& path, std::vector<float>& samples, int& sample_rate) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;

    wav_header header;
    if (fread(&header, sizeof(header), 1, f) != 1) {
        fclose(f);
        return false;
    }

    // Validate header
    if (header.audio_format != 1 || header.bits_per_sample != 16) {
        fclose(f);
        return false;
    }

    sample_rate = header.sample_rate;
    samples.resize(header.data_size / 2);  // 16-bit samples

    for (auto& s : samples) {
        int16_t pcm;
        if (fread(&pcm, sizeof(pcm), 1, f) != 1) break;
        s = pcm / 32768.0f;
    }

    fclose(f);
    return true;
}

inline bool save_wav_file(const std::string& path, const std::vector<float>& samples, int sample_rate) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) return false;

    wav_header header;
    header.sample_rate = sample_rate;
    header.byte_rate = sample_rate * 1 * 2;  // 1 channel, 16-bit
    header.block_align = 1 * 2;
    header.data_size = samples.size() * 2;
    header.chunk_size = 36 + header.data_size;

    fwrite(&header, sizeof(header), 1, f);

    for (const auto& s : samples) {
        int16_t pcm = static_cast<int16_t>(std::max(-32768.0, std::min(32767.0, s * 32767.0)));
        fwrite(&pcm, sizeof(pcm), 1, f);
    }

    fclose(f);
    return true;
}

// ============================================================================
// Comparison Helpers
// ============================================================================

// Compare audio samples with tolerance
inline bool compare_audio(const std::vector<float>& a, const std::vector<float>& b, float tolerance = 0.01f) {
    if (a.size() != b.size()) return false;

    for (size_t i = 0; i < a.size(); i++) {
        if (std::abs(a[i] - b[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

// Compute max absolute difference between two audio signals
inline float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return 1e30f;

    float max_diff = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        max_diff = std::max(max_diff, std::abs(a[i] - b[i]));
    }
    return max_diff;
}

// Compute correlation between two audio signals
inline float correlation(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return 0.0f;

    float mean_a = 0.0f, mean_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= a.size();
    mean_b /= b.size();

    float num = 0.0f, den_a = 0.0f, den_b = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        float da = a[i] - mean_a;
        float db = b[i] - mean_b;
        num += da * db;
        den_a += da * da;
        den_b += db * db;
    }

    if (den_a == 0 || den_b == 0) return 0.0f;
    return num / std::sqrt(den_a * den_b);
}

// ============================================================================
// Acceptance Test Helpers
// ============================================================================

// Acceptance criteria thresholds for vocoder output validation
// These values are based on analysis of working SNAC vocoder output
struct AcceptanceThresholds {
    // Time domain criteria
    float neg_ratio_min = 35.0f;        // Symmetric waveform: 35-65%
    float neg_ratio_max = 65.0f;
    float dc_bias_max = 0.01f;          // No DC offset: |mean| < 0.01
    float peak_amplitude_min = 0.7f;    // Proper amplitude: 0.7-0.95
    float peak_amplitude_max = 0.95f;
    float std_dev_min = 0.1f;           // Reasonable variance

    // Frequency domain criteria
    float energy_0_200Hz_max = 50.0f;   // Speech can have ~46% in this band
    float energy_200_500Hz_min = 10.0f; // Speech frequency range present
    float energy_1000Hz_plus_min = 5.0f; // High frequency content present
};

// Default acceptance thresholds
inline AcceptanceThresholds default_acceptance_thresholds() {
    return AcceptanceThresholds();
}

// Strict acceptance thresholds (for production quality)
inline AcceptanceThresholds strict_acceptance_thresholds() {
    AcceptanceThresholds t;
    t.neg_ratio_min = 40.0f;
    t.neg_ratio_max = 60.0f;
    t.dc_bias_max = 0.005f;
    t.peak_amplitude_min = 0.75f;
    t.peak_amplitude_max = 0.9f;
    t.std_dev_min = 0.15f;
    t.energy_0_200Hz_max = 35.0f;
    t.energy_200_500Hz_min = 15.0f;
    t.energy_1000Hz_plus_min = 10.0f;
    return t;
}

// Validate audio metrics against custom thresholds
inline bool validate_against_thresholds(const AudioMetrics& m, const AcceptanceThresholds& t) {
    // Check neg_ratio
    if (m.neg_ratio < t.neg_ratio_min || m.neg_ratio > t.neg_ratio_max) {
        return false;
    }

    // Check DC bias
    if (std::abs(m.dc_bias) > t.dc_bias_max) {
        return false;
    }

    // Check peak amplitude
    if (m.peak_amplitude < t.peak_amplitude_min || m.peak_amplitude > t.peak_amplitude_max) {
        return false;
    }

    // Check standard deviation
    if (m.std_dev < t.std_dev_min) {
        return false;
    }

    // Check frequency distribution
    if (m.energy_0_200Hz > t.energy_0_200Hz_max) {
        return false;
    }

    if (m.energy_200_500Hz < t.energy_200_500Hz_min) {
        return false;
    }

    if (m.energy_1000Hz_plus < t.energy_1000Hz_plus_min) {
        return false;
    }

    return true;
}

// Generate synthetic vocoder output for testing
// This creates a complex waveform that simulates speech characteristics
inline std::vector<float> generate_synthetic_vocoder_output(int n_samples, int sample_rate, uint64_t seed = 42) {
    std::vector<float> samples(n_samples);
    std::mt19937 rng(seed);

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

        // Combine and scale to typical vocoder output range
        samples[i] = 0.8f * envelope * (fundamental + h2 + h3 + h4 + formant + noise);
    }

    return samples;
}

// Generate golden test token sequences for SNAC vocoder
// These are deterministic sequences that can be used to test the vocoder pipeline
struct GoldenTokenSequence {
    std::string name;
    std::vector<std::vector<int>> tokens;
    std::string description;
};

inline std::vector<GoldenTokenSequence> get_golden_token_sequences(int n_frames) {
    std::vector<GoldenTokenSequence> sequences;

    // Sequence 1: All zeros (silence)
    {
        GoldenTokenSequence seq;
        seq.name = "silence";
        seq.description = "All zero tokens - should produce silence or near-silence";
        seq.tokens = generate_zeros_tokens(n_frames);
        sequences.push_back(seq);
    }

    // Sequence 2: Deterministic random
    {
        GoldenTokenSequence seq;
        seq.name = "deterministic_random";
        seq.description = "Deterministic random tokens - should produce audio with expected characteristics";
        seq.tokens = generate_deterministic_tokens(n_frames, 42);
        sequences.push_back(seq);
    }

    // Sequence 3: Sequential
    {
        GoldenTokenSequence seq;
        seq.name = "sequential";
        seq.description = "Sequential tokens (0,1,2,...) - tests codebook coverage";
        seq.tokens = generate_sequential_tokens(n_frames);
        sequences.push_back(seq);
    }

    // Sequence 4: Mid-range values (typical for speech)
    {
        GoldenTokenSequence seq;
        seq.name = "mid_range";
        seq.description = "Mid-range token values (1000-3000) - typical for speech tokens";
        seq.tokens.resize(3);
        std::mt19937 rng(12345);
        std::uniform_int_distribution<int> dist(1000, 3000);
        for (int f = 0; f < n_frames; f++) {
            for (int j = 0; j < 4; j++) seq.tokens[0].push_back(dist(rng));
            for (int j = 0; j < 2; j++) seq.tokens[1].push_back(dist(rng));
            seq.tokens[2].push_back(dist(rng));
        }
        sequences.push_back(seq);
    }

    return sequences;
}

// Print a summary of golden token sequences
inline void print_golden_sequences_info(const std::vector<GoldenTokenSequence>& sequences) {
    printf("=== Golden Token Sequences ===\n");
    for (const auto& seq : sequences) {
        printf("  %s: %s\n", seq.name.c_str(), seq.description.c_str());
        printf("    head0: %zu tokens, head1: %zu tokens, head2: %zu tokens\n",
               seq.tokens[0].size(), seq.tokens[1].size(), seq.tokens[2].size());
    }
}

}  // namespace orpheus_test
