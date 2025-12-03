#pragma once

#ifdef GGML_USE_DLCU

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>

static inline bool ggml_dl_timing_enabled() {
    static const bool enabled = []() {
        const char* env = getenv("GGML_DL_TIMING");
        return env != nullptr && env[0] == '1';
    }();
    return enabled;
}

static inline double ggml_dl_timing_env_threshold_ms() {
    static const double threshold = []() {
        const char* env = getenv("GGML_DL_TIMING_MIN_MS");
        if (env == nullptr) {
            return 0.0;
        }
        return atof(env);
    }();
    return threshold;
}

class ggml_dl_scope_timer {
public:
    ggml_dl_scope_timer(std::string label,
                        double threshold_ms = -1.0,
                        cudaStream_t stream_to_sync = nullptr)
        : label_(std::move(label)),
          threshold_ms_(threshold_ms),
          stream_(stream_to_sync),
          start_(std::chrono::steady_clock::now()) {
    }

    ~ggml_dl_scope_timer() {
        if (!ggml_dl_timing_enabled()) {
            return;
        }

        if (stream_ != nullptr) {
            cudaStreamSynchronize(stream_);
        }

        const auto end = std::chrono::steady_clock::now();
        const double elapsed_ms =
            std::chrono::duration<double, std::milli>(end - start_).count();
        const double threshold =
            threshold_ms_ >= 0.0 ? threshold_ms_ : ggml_dl_timing_env_threshold_ms();

        if (threshold <= 0.0 || elapsed_ms >= threshold) {
            fprintf(stderr, "[DL-TIMER] %s took %.3f ms\n", label_.c_str(), elapsed_ms);
            fflush(stderr);
        }
    }

private:
    std::string label_;
    double threshold_ms_;
    cudaStream_t stream_;
    std::chrono::time_point<std::chrono::steady_clock> start_;
};

#define GGML_DL_SCOPE_TIMER_CAT(a, b) a##b
#define GGML_DL_SCOPE_TIMER_MAKE_NAME(a, b) GGML_DL_SCOPE_TIMER_CAT(a, b)
#define GGML_DL_SCOPE_TIMER_UNIQUE_NAME() \
    GGML_DL_SCOPE_TIMER_MAKE_NAME(_ggml_dl_scope_timer_, __COUNTER__)

#define GGML_DL_SCOPE_TIMER(label) \
    ggml_dl_scope_timer GGML_DL_SCOPE_TIMER_UNIQUE_NAME()(label)

#define GGML_DL_SCOPE_TIMER_MIN(label, threshold_ms) \
    ggml_dl_scope_timer GGML_DL_SCOPE_TIMER_UNIQUE_NAME()(label, threshold_ms)

#define GGML_DL_SCOPE_TIMER_WITH_STREAM(label, stream) \
    ggml_dl_scope_timer GGML_DL_SCOPE_TIMER_UNIQUE_NAME()(label, -1.0, stream)

#endif // GGML_USE_DLCU

