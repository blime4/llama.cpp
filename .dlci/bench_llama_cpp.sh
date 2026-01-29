#!/bin/bash
set -e

# Load shared utility helpers
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Parse command line arguments
CI_TEST_MODE=false
BINARY_PATH=""
VERBOSE=false
MODEL_PATH=""
GPU_LAYERS=999

while [[ $# -gt 0 ]]; do
    case $1 in
        --ci-test)
            CI_TEST_MODE=true
            shift
            ;;
        --binary-path)
            BINARY_PATH="$2"
            shift 2
            ;;
        --model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --gpu-layers)
            GPU_LAYERS="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        *)
            echo "[ERROR] Unknown parameter: $1"
            echo "Usage: $0 [--ci-test] [--binary-path <path>] [--model-path <path>] [--gpu-layers <num>] [--verbose]"
            echo "  --ci-test: Enable CI test mode using external binaries"
            echo "  --binary-path: Path to the release directory containing binaries"
            echo "  --model-path: Path to the model file (default: from LOCAL_MODEL_PATH)"
            echo "  --gpu-layers: Number of GPU layers (default: 999)"
            echo "  --verbose: Enable verbose output"
            exit 1
            ;;
    esac
done

# Validate CI test mode parameters
if [ "$CI_TEST_MODE" = true ]; then
    if [ -z "$BINARY_PATH" ]; then
        echo "[ERROR] --binary-path is required when using --ci-test mode"
        exit 1
    fi

    if [ ! -d "$BINARY_PATH" ]; then
        echo "[ERROR] Binary path does not exist: $BINARY_PATH"
        exit 1
    fi

    echo "[INFO] CI Test Mode enabled"
    echo "[INFO] Using binaries from: $BINARY_PATH"

    # Verify llama-bench binary exists
    if [ ! -x "$BINARY_PATH/llama-bench" ]; then
        echo "[ERROR] llama-bench binary not found or not executable: $BINARY_PATH/llama-bench"
        exit 1
    fi
    echo "[INFO] llama-bench binary validated successfully"
fi

# Record start time
bench_start_time=$(date +%s)

# Create logs directory
logs_dir="/LocalRun/$(whoami)/logs/llama_cpp_bench"
mkdir -p "$logs_dir"

# Log files
bench_log="${logs_dir}/bench_$(date +%Y%m%d_%H%M%S).log"
summary_log="${logs_dir}/bench_summary_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Benchmark suite started at: $(date)" | tee -a "$bench_log"
echo "[INFO] Detailed logs will be saved to: $bench_log" | tee -a "$summary_log"
echo "[INFO] Summary will be saved to: $summary_log" | tee -a "$summary_log"
if [ "$VERBOSE" = "true" ]; then
    echo "[INFO] VERBOSE MODE ENABLED - All outputs will be displayed in real-time" | tee -a "$summary_log"
fi

# Setup environment
echo "[INFO] Setting up environment..." | tee -a "$summary_log"
env >> "$bench_log" 2>&1

# env.sh invokes rm before it resets LD_LIBRARY_PATH, so dlPTI's injector must be disabled temporarily.
if [ -n "$LD_PRELOAD" ]; then
    saved_ld_preload="$LD_PRELOAD"
    unset LD_PRELOAD
    source "${SDK_DIR}/env.sh" >> "$bench_log" 2>&1
    export LD_PRELOAD="$saved_ld_preload"
    unset saved_ld_preload
else
    source "${SDK_DIR}/env.sh" >> "$bench_log" 2>&1
fi
env >> "$bench_log" 2>&1

# Configure ccache
echo "[INFO] Configuring ccache..." | tee -a "$summary_log"
ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
{
    ccache --set-config cache_dir="${ccache_dir}"
    ccache --set-config max_size=20G
    ccache --zero-stats
} >> "$bench_log" 2>&1

# Enter repository directory
echo "[INFO] REPO_PATH: ${REPO_PATH}" | tee -a "$summary_log"
echo "[INFO] LOCAL_MODEL_PATH: ${LOCAL_MODEL_PATH}" | tee -a "$summary_log"
cd "${REPO_PATH}"

ARCH=${DOCKER_PLATFORM}
echo "[INFO] Platform: $ARCH" | tee -a "$summary_log"

# Set build directory based on test mode
if [ "$CI_TEST_MODE" = true ]; then
    # In CI test mode, use the provided binary path
    build_dir_bin="$BINARY_PATH"
    echo "[INFO] CI Mode: Using binaries from: $build_dir_bin" | tee -a "$summary_log"

    # Set LD_LIBRARY_PATH to include the binary directory for shared libraries
    if [ -d "$build_dir_bin" ]; then
        export LD_LIBRARY_PATH="$build_dir_bin:${LD_LIBRARY_PATH:-}"
        echo "[INFO] CI Mode: Set LD_LIBRARY_PATH to include: $build_dir_bin" | tee -a "$summary_log"
    fi
else
    # In normal mode, use the standard build directory
    build_dir=$(pwd)/build_${ARCH}
    build_dir_bin="${build_dir}/bin"

    # Check if build directory exists
    if [ ! -d "${build_dir}" ]; then
        echo "[ERROR] ${build_dir} does not exist ..." | tee -a "$summary_log"
        exit 1
    fi
    echo "[INFO] Build directory: $build_dir" | tee -a "$summary_log"
fi

# Check if llama-bench exists
if [ ! -x "${build_dir_bin}/llama-bench" ]; then
    echo "[ERROR] llama-bench not found: ${build_dir_bin}/llama-bench" | tee -a "$summary_log"
    exit 1
fi

# Determine model path
if [ -z "$MODEL_PATH" ]; then
    if [ -z "$LOCAL_MODEL_PATH" ]; then
        echo "[ERROR] Model path not specified. Use --model-path or set LOCAL_MODEL_PATH" | tee -a "$summary_log"
        exit 1
    fi
    # Default model: Qwen3-30B-A3B-Q4_K_M.gguf
    MODEL_PATH="${LOCAL_MODEL_PATH}/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
fi

# Check if model file exists
if [ ! -f "$MODEL_PATH" ]; then
    echo "[ERROR] Model file not found: $MODEL_PATH" | tee -a "$summary_log"
    exit 1
fi

echo "[INFO] ============================================================" | tee -a "$summary_log"
echo "[INFO] llama-bench Performance Test" | tee -a "$summary_log"
echo "[INFO] ============================================================" | tee -a "$summary_log"
echo "[INFO] Model: $MODEL_PATH" | tee -a "$summary_log"
echo "[INFO] GPU Layers: $GPU_LAYERS" | tee -a "$summary_log"
echo "[INFO] ============================================================" | tee -a "$summary_log"
echo "" | tee -a "$summary_log"

# Run benchmark
echo "[INFO] Starting llama-bench..." | tee -a "$summary_log"
echo "[INFO] Command: CUDA_VISIBLE_DEVICES=0,1 QWEN_USE_FP16=1 ${build_dir_bin}/llama-bench -m \"$MODEL_PATH\" -ngl $GPU_LAYERS --progress -fa 1 -pg 512,128 -p 0 -n 0 -r 1" | tee -a "$summary_log"
echo "" | tee -a "$summary_log"
echo "-----------------------------" | tee -a "$bench_log"

start_time=$(date +%s)
set +e

CUDA_VISIBLE_DEVICES=0,1 QWEN_USE_FP16=1 "${build_dir_bin}/llama-bench" -m "$MODEL_PATH" -ngl "$GPU_LAYERS" --progress -fa 1 -pg 512,128 -p 0 -n 0 -r 1 2>&1 | tee -a "$bench_log"
ret=$?

set -e
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "-----------------------------" | tee -a "$bench_log"
echo "" | tee -a "$summary_log"

# Check results
if [ $ret -eq 0 ]; then
    echo "[SUCCESS] Benchmark completed successfully!" | tee -a "$summary_log"
    echo "[INFO] Duration: ${duration}s" | tee -a "$summary_log"
else
    echo "[ERROR] Benchmark failed with exit code: $ret" | tee -a "$summary_log"
    echo "[ERROR] Duration: ${duration}s" | tee -a "$summary_log"

    if [ -f "$bench_log" ]; then
        echo "[ERROR] Last 50 lines of benchmark output:" | tee -a "$summary_log"
        tail -n 50 "$bench_log" | tee -a "$summary_log"
    fi

    exit $ret
fi

# Calculate total time
bench_end_time=$(date +%s)
total_duration=$((bench_end_time - bench_start_time))

echo "" | tee -a "$summary_log"