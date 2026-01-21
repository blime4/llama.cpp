#!/bin/bash
set -e

# Load shared utility helpers (includes safe_rm)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

set_global_defaults() {
    CI_TEST_MODE=false
    BINARY_PATH=""
    SIMPLE_TEST=false
    SIMPLE_MODEL=false
    SIMPLE_PERF=false
    SIMPLE_TP=false
    BIG_MODEL=false
    VERBOSE=false
    NO_FA=false
    SIMPLE_MODEL_GPU_LAYERS=999

    test_start_time=0
    logs_dir=""
    test_log=""
    summary_log=""
    build_dir_bin=""
}

print_usage() {
    cat <<'EOF'
Usage: test_llama_cpp.sh [OPTIONS]
  --ci-test        Enable CI test mode using external binaries
  --binary-path    Path to the release directory containing binaries
  --simple-test    Run only backend-ops tests (MUL_MAT and DLFA)
  --simple-model   Run only Qwen2.5 model tests (GGML_DLFA_READY=1)
  --simple-perf    Run model performance stress tests (Pool vs Legacy)
  --simple-tp      Run Tensor Parallel tests with split-mode variations
  --big            Use large model for TP tests (Qwen3-30B)
  --verbose        Enable verbose output (real-time logs)
  --no-fa          Disable Flash Attention (standard inference mode)
EOF
}

parse_arguments() {
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
            --simple-test)
                SIMPLE_TEST=true
                shift
                ;;
            --simple-model)
                SIMPLE_MODEL=true
                shift
                ;;
            --simple-perf)
                SIMPLE_PERF=true
                shift
                ;;
            --simple-tp)
                SIMPLE_TP=true
                shift
                ;;
            --big)
                BIG_MODEL=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --no-fa)
                NO_FA=true
                shift
                ;;
            *)
                echo "[ERROR] Unknown parameter: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

validate_ci_mode() {
    if [ "$CI_TEST_MODE" != true ]; then
        return
    fi

    if [ -z "$BINARY_PATH" ]; then
        echo "[ERROR] --binary-path is required when using --ci-test mode"
        print_usage
        exit 1
    fi

    if [ ! -d "$BINARY_PATH" ]; then
        echo "[ERROR] Binary path does not exist: $BINARY_PATH"
        exit 1
    fi

    echo "[INFO] CI Test Mode enabled"
    echo "[INFO] Using binaries from: $BINARY_PATH"

    local essential_binaries=("llama-cli" "test-c")
    for binary in "${essential_binaries[@]}"; do
        if [ ! -x "$BINARY_PATH/$binary" ]; then
            echo "[ERROR] Essential binary not found or not executable: $BINARY_PATH/$binary"
            exit 1
        fi
    done
    echo "[INFO] Essential binaries validated successfully"
}

init_logging() {
    test_start_time=$(date +%s)
    logs_dir="/LocalRun/$(whoami)/logs/llama_cpp_test"
    mkdir -p "$logs_dir"

    local timestamp
    timestamp=$(date +%Y%m%d_%H%M%S)
    test_log="${logs_dir}/test_${timestamp}.log"
    summary_log="${logs_dir}/test_summary_${timestamp}.log"

    echo "[INFO] Test suite started at: $(date)" | tee -a "$test_log"
    echo "[INFO] Detailed logs will be saved to: $test_log" | tee -a "$summary_log"
    echo "[INFO] Summary will be saved to: $summary_log" | tee -a "$summary_log"
    if [ "$VERBOSE" = "true" ]; then
        echo "[INFO] VERBOSE MODE ENABLED - All test outputs will be displayed in real-time" | tee -a "$summary_log"
    else
        echo "[INFO] Standard mode - Only limited output shown (use --verbose for full output)" | tee -a "$summary_log"
    fi

    export TEST_LOG_FILE="$test_log"
}

setup_environment_context() {
    echo "[INFO] Setting up environment..." | tee -a "$summary_log"
    env >> "$test_log" 2>&1
    if [ -n "${LD_PRELOAD:-}" ]; then
        local saved_ld_preload="$LD_PRELOAD"
        unset LD_PRELOAD
        source "${SDK_DIR}/env.sh" >> "$test_log" 2>&1
        export LD_PRELOAD="$saved_ld_preload"
        unset saved_ld_preload
    else
        source "${SDK_DIR}/env.sh" >> "$test_log" 2>&1
    fi
    env >> "$test_log" 2>&1
}

configure_ccache() {
    echo "[INFO] Configuring ccache..." | tee -a "$summary_log"
    local ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
    {
        ccache --set-config cache_dir="${ccache_dir}"
        ccache --set-config max_size=20G
        ccache --zero-stats
    } >> "$test_log" 2>&1
}

prepare_build_dir() {
    echo "[INFO] REPO_PATH: ${REPO_PATH}" | tee -a "$summary_log"
    echo "[INFO] LOCAL_MODEL_PATH: ${LOCAL_MODEL_PATH}" | tee -a "$summary_log"
    cd "${REPO_PATH}"

    ARCH="${DOCKER_PLATFORM:-$(uname -m)}"
    echo "[INFO] Platform: $ARCH" | tee -a "$summary_log"

    if [ "$CI_TEST_MODE" = true ]; then
        build_dir_bin="$BINARY_PATH"
        echo "[INFO] CI Mode: Using binaries from: $build_dir_bin" | tee -a "$summary_log"
        if [ -d "$build_dir_bin" ]; then
            export LD_LIBRARY_PATH="$build_dir_bin:${LD_LIBRARY_PATH:-}"
            echo "[INFO] CI Mode: Set LD_LIBRARY_PATH to include: $build_dir_bin" | tee -a "$summary_log"
            echo "[INFO] CI Mode: Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | tee -a "$summary_log"
        fi
    else
        local build_dir
        build_dir=$(pwd)/build_"${ARCH}"
        build_dir_bin="${build_dir}/bin"
        if [ ! -d "${build_dir}" ]; then
            echo "[ERROR] ${build_dir} does not exist ..." | tee -a "$summary_log"
            exit 1
        fi
        echo "[INFO] Build directory: $build_dir" | tee -a "$summary_log"
    fi
}

set_common_runtime_env() {
    export GGML_DEBUG=1
    export CUDA_VISIBLE_DEVICES=0
    echo "[INFO] CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES" | tee -a "$summary_log"
}

declare -a SIMPLE_TEST_CASES=()
declare -a simple_test_failures=()
declare -a simple_test_skips=()

simple_test_define_cases() {
    SIMPLE_TEST_CASES=(
        # "MUL_MAT|${build_dir_bin}/test-backend-ops -o MUL_MAT|GGML_CUDA_DISABLE_GRAPHS=1 GGML_CUDA_GPTQ_GROUP_SIZE=128 GGML_DL_MULMAT_DEBUG=1"
        # "FLASH_ATTN_EXT|${build_dir_bin}/test-backend-ops -o FLASH_ATTN_EXT|GGML_CUDA_DISABLE_GRAPHS=1"
        "FLASH_ATTN_EXT|${build_dir_bin}/test-backend-ops -o FLASH_ATTN_EXT --verbose|"
        # "GET_ROWS|${build_dir_bin}/test-backend-ops -o GET_ROWS --verbose|"
    )
}

simple_test_print_header() {
    echo "[INFO] =============================" | tee -a "$summary_log"
    echo "[INFO] Simple test mode enabled" | tee -a "$summary_log"
    echo "[INFO] =============================" | tee -a "$summary_log"
}

simple_test_print_catalog() {
    echo "[INFO] Test list (${#SIMPLE_TEST_CASES[@]} tests):" | tee -a "$summary_log"
    for i in "${!SIMPLE_TEST_CASES[@]}"; do
        IFS='|' read -r test_name _ <<< "${SIMPLE_TEST_CASES[$i]}"
        echo "[INFO]   $((i+1)). $test_name" | tee -a "$summary_log"
    done
    echo "" | tee -a "$summary_log"
}

simple_test_reset_counters() {
    simple_test_failures=()
    simple_test_skips=()
    simple_test_fail_count=0
    simple_test_pass_count=0
    simple_test_skip_count=0
    simple_test_total=${#SIMPLE_TEST_CASES[@]}
}

simple_test_parse_sub_results() {
    local temp_output="$1"
    local -n _pass_ref=$2
    local -n _fail_ref=$3
    local -n _skip_ref=$4
    local total_fail_lines=0
    local backend_fail_lines=0
    local fail_lines_text=""
    local backend_lines_text=""

    _pass_ref=0
    _fail_ref=0
    _skip_ref=0

    if [ ! -f "$temp_output" ]; then
        return
    fi

    _pass_ref=$(grep -c "test passed \[" "$temp_output" 2>/dev/null || echo "0")
    _pass_ref=$(echo "$_pass_ref" | tr -d '\n\r' | xargs)
    if [ "$_pass_ref" -eq 0 ]; then
        _pass_ref=$(grep -c "32mOK" "$temp_output" 2>/dev/null || echo "0")
        _pass_ref=$(echo "$_pass_ref" | tr -d '\n\r' | xargs)
    fi

    _fail_ref=$(grep -c "test failed \[" "$temp_output" 2>/dev/null || true)
    _fail_ref=$(echo "${_fail_ref:-0}" | tr -d '\n\r' | xargs)
    if [ "$_fail_ref" -eq 0 ]; then
        fail_lines_text=$(grep -c "31mFAIL" "$temp_output" 2>/dev/null || true)
        total_fail_lines=$(echo "${fail_lines_text:-0}" | tr -d '\n\r' | xargs)
        backend_lines_text=$(grep "31mFAIL" "$temp_output" 2>/dev/null | grep -c "Backend" 2>/dev/null || true)
        backend_fail_lines=$(echo "${backend_lines_text:-0}" | tr -d '\n\r' | xargs)
        if [ "$total_fail_lines" -gt 0 ]; then
            _fail_ref=$((total_fail_lines - backend_fail_lines - 1))
            if [ "$_fail_ref" -lt 0 ]; then
                _fail_ref=0
            fi
        fi
        _fail_ref=$(echo "$_fail_ref" | tr -d '\n\r' | xargs)
    fi

    _skip_ref=$(grep -E -c "not supported|skipping \(" "$temp_output" 2>/dev/null || echo "0")
    _skip_ref=$(echo "$_skip_ref" | tr -d '\n\r' | xargs)
}

simple_test_execute_case() {
    local test_index="$1"
    local config="$2"
    local total="$3"
    local actual_total=0
    local ret=0

    IFS='|' read -r test_name test_cmd test_env <<< "$config"

    echo "" | tee -a "$summary_log"
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "[INFO] Test $test_index/$total: $test_name" | tee -a "$summary_log"
    if [ "$test_env" != "NONE" ]; then
        echo "[INFO] Running: $test_env $test_cmd" | tee -a "$summary_log"
    else
        echo "[INFO] Running: $test_cmd" | tee -a "$summary_log"
    fi
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "-----------------------------" | tee -a "$test_log"

    local start_time end_time duration
    start_time=$(date +%s)
    set +e
    local temp_output="/tmp/test_output_${test_index}_$$.log"

    if [ "$test_env" != "NONE" ]; then
        bash -c "$test_env $test_cmd" 2>&1 | tee "$temp_output" | tee -a "$test_log"
        ret=${PIPESTATUS[0]}
    else
        eval "$test_cmd" 2>&1 | tee "$temp_output" | tee -a "$test_log"
        ret=${PIPESTATUS[0]}
    fi
    set -e
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "-----------------------------" | tee -a "$test_log"

    local sub_pass=0
    local sub_fail=0
    local sub_skip=0
    simple_test_parse_sub_results "$temp_output" sub_pass sub_fail sub_skip

    if [ "$sub_fail" -gt 0 ] || [ "$sub_pass" -gt 0 ] || [ "$sub_skip" -gt 0 ]; then
        actual_total=$((sub_pass + sub_fail + sub_skip))
        echo "[INFO] Sub-test results: Pass=$sub_pass, Fail=$sub_fail, Skip=$sub_skip, Total=$actual_total" | tee -a "$summary_log"
    fi

    safe_rm -f "$temp_output"

    if [ "$ret" -ne 0 ]; then
        echo "[FAIL] $test_name (exit code: $ret, ${duration}s)" | tee -a "$summary_log"
        simple_test_fail_count=$((simple_test_fail_count + 1))
        simple_test_failures+=("$test_name (exit code $ret)")
    elif [ "$sub_fail" -gt 0 ]; then
        echo "[FAIL] $test_name (${duration}s, $sub_fail sub-test failures)" | tee -a "$summary_log"
        simple_test_fail_count=$((simple_test_fail_count + 1))
        simple_test_failures+=("$test_name ($sub_fail sub-test failures)")
    elif [ "$sub_pass" -eq 0 ] && [ "$sub_skip" -gt 0 ]; then
        echo "[SKIP] $test_name (${duration}s, $sub_skip sub-tests skipped)" | tee -a "$summary_log"
        simple_test_skip_count=$((simple_test_skip_count + 1))
        simple_test_skips+=("$test_name (all $sub_skip sub-tests not supported)")
    else
        echo "[PASS] $test_name (${duration}s)" | tee -a "$summary_log"
        simple_test_pass_count=$((simple_test_pass_count + 1))
    fi
}

simple_test_print_summary() {
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Simple Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO]   Total tests:  $simple_test_total" | tee -a "$summary_log"
    echo "[INFO]   Passed:       $simple_test_pass_count" | tee -a "$summary_log"
    echo "[INFO]   Failed:       $simple_test_fail_count" | tee -a "$summary_log"
    echo "[INFO]   Skipped:      $simple_test_skip_count" | tee -a "$summary_log"
    echo "[INFO]   Total time:   ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    if [ "$simple_test_skip_count" -gt 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[INFO] Skipped tests (not supported):" | tee -a "$summary_log"
        for skip_item in "${simple_test_skips[@]}"; do
            echo "  - $skip_item" | tee -a "$summary_log"
        done
    fi

    if [ "$simple_test_fail_count" -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] Simple tests failed:" | tee -a "$summary_log"
        for fail_item in "${simple_test_failures[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    fi

    echo "" | tee -a "$summary_log"
    if [ "$simple_test_skip_count" -gt 0 ]; then
        echo "[LLAMA_CPP_PASS] All enabled tests passed! ($simple_test_skip_count tests skipped)" | tee -a "$summary_log"
    else
        echo "[LLAMA_CPP_PASS] All simple tests passed!" | tee -a "$summary_log"
    fi
    exit 0
}

run_simple_test_mode() {
    simple_test_print_header
    simple_test_define_cases
    simple_test_print_catalog
    simple_test_reset_counters

    for i in "${!SIMPLE_TEST_CASES[@]}"; do
        simple_test_execute_case "$((i+1))" "${SIMPLE_TEST_CASES[$i]}" "$simple_test_total"
    done

    simple_test_print_summary
}

if [ "$SIMPLE_TEST" = true ]; then
    run_simple_test_mode
fi

extract_model_output() {
    local full_output="$1"

    echo "$full_output" | sed -n '/^$/,/^llama_perf_sampler_print:/p' | sed '$d' | tail -n +2
}

validate_output() {
    local output_content="$1"
    local expected_content="$2"

    if echo "$output_content" | grep -qi "$expected_content"; then
        return 0
    else
        return 1
    fi
}

declare -a SIMPLE_MODEL_CASES=(
    "geography|请用一个词回答：北京是中国的什么？|首都|地理知识测试"
)
declare -a simple_model_fail_list=()

simple_model_intro() {
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Qwen2.5 model correctness test" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Target model: Qwen2.5-1.5B (q4_k_m)" | tee -a "$summary_log"
    if [ "$NO_FA" = "true" ]; then
        echo "[INFO] Test mode: standard inference (no Flash Attention)" | tee -a "$summary_log"
        echo "[INFO] Purpose: verify standard inference output matches expected" | tee -a "$summary_log"
    else
        echo "[INFO] Test mode: Flash Attention inference (GGML_DLFA_READY=1)" | tee -a "$summary_log"
        echo "[INFO] Purpose: verify Flash Attention output matches expected" | tee -a "$summary_log"
    fi
    echo "[INFO] ============================================================" | tee -a "$summary_log"
}

simple_model_prepare_model() {
    simple_model_base="${LOCAL_MODEL_PATH}"
    simple_model_relative="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    simple_model_path="${simple_model_base%/}/${simple_model_relative}"
    simple_model_name="qwen2.5-1.5b-instruct-q4_k_m"

    if [ ! -f "$simple_model_path" ]; then
        echo "[ERROR] Model file not found: $simple_model_path" | tee -a "$summary_log"
        exit 1
    fi
    echo "[INFO] Model file: $simple_model_path" | tee -a "$summary_log"
}

simple_model_reset_state() {
    simple_model_fail_list=()
    simple_model_fail_count=0
    simple_model_pass_count=0
    simple_model_total=${#SIMPLE_MODEL_CASES[@]}
}

simple_model_execute_case() {
    local index="$1"
    local config="$2"
    local total="$3"

    IFS='|' read -r test_name prompt expected_content description <<< "$config"

    echo "" | tee -a "$summary_log"
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "[INFO] test $index/$total: $simple_model_name - $test_name" | tee -a "$summary_log"
    echo "[INFO] description: $description" | tee -a "$summary_log"
    echo "[INFO] prompt: $prompt" | tee -a "$summary_log"
    echo "[INFO] expected output: $expected_content" | tee -a "$summary_log"
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "-----------------------------" | tee -a "$test_log"

    local start_time end_time duration ret temp_output full_output model_output
    start_time=$(date +%s)
    set +e
    temp_output=$(mktemp)
    if [ "$NO_FA" = "true" ]; then
        echo "[DEBUG] (without Flash Attention), -ngl ${SIMPLE_MODEL_GPU_LAYERS}" | tee -a "$test_log"
        echo "[DEBUG] Prompt content: '$prompt'" | tee -a "$test_log"
        echo "[DEBUG] Command: CUDA_VISIBLE_DEVICES=0 \"${build_dir_bin}/llama-cli\" -m \"$simple_model_path\" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -ngl ${SIMPLE_MODEL_GPU_LAYERS} -p \"$prompt\" --no-warmup > \"$temp_output\" 2>&1" | tee -a "$test_log"
        CUDA_VISIBLE_DEVICES=0 "${build_dir_bin}/llama-cli" -m "$simple_model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -ngl ${SIMPLE_MODEL_GPU_LAYERS} -p "$prompt" --no-warmup > "$temp_output" 2>&1
        ret=$?
    else
        echo "[DEBUG] (with Flash Attention), -ngl ${SIMPLE_MODEL_GPU_LAYERS}" | tee -a "$test_log"
        echo "[DEBUG] Prompt content: '$prompt'" | tee -a "$test_log"
        echo "[DEBUG] Command: CUDA_VISIBLE_DEVICES=0 GGML_DLFA_READY=1 \"${build_dir_bin}/llama-cli\" -m \"$simple_model_path\" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -fa -ngl ${SIMPLE_MODEL_GPU_LAYERS} -p \"$prompt\" --no-warmup > \"$temp_output\" 2>&1" | tee -a "$test_log"
        CUDA_VISIBLE_DEVICES=0 GGML_DLFA_READY=1 "${build_dir_bin}/llama-cli" -m "$simple_model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -fa -ngl ${SIMPLE_MODEL_GPU_LAYERS} -p "$prompt" --no-warmup > "$temp_output" 2>&1
        ret=$?
    fi
    set -e
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    cat "$temp_output" | tee -a "$test_log"
    echo "-----------------------------" | tee -a "$test_log"

    if [ $ret -ne 0 ]; then
        echo "[FAIL] $simple_model_name - $test_name (exit code: $ret, ${duration}s)" | tee -a "$summary_log"
        simple_model_fail_count=$((simple_model_fail_count + 1))
        simple_model_fail_list+=("$simple_model_name - $test_name (exit code $ret)")
        safe_rm -f "$temp_output"
        return
    fi

    full_output=$(cat "$temp_output")
    model_output=$(extract_model_output "$full_output")
    echo "[DEBUG] Model generated text: $model_output" | tee -a "$test_log"

    if validate_output "$model_output" "$expected_content"; then
        echo "[PASS] $simple_model_name - $test_name (${duration}s)" | tee -a "$summary_log"
        echo "[INFO] ✓ Output contains expected: $expected_content" | tee -a "$summary_log"
        simple_model_pass_count=$((simple_model_pass_count + 1))
    else
        echo "[FAIL] $simple_model_name - $test_name (${duration}s)" | tee -a "$summary_log"
        echo "[ERROR] Output does NOT contain expected: $expected_content" | tee -a "$summary_log"
        echo "[ERROR] Model generated: $model_output" | tee -a "$summary_log"
        simple_model_fail_count=$((simple_model_fail_count + 1))
        simple_model_fail_list+=("$simple_model_name - $test_name (validation failed)")
    fi

    safe_rm -f "$temp_output"
}

simple_model_print_summary() {
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Qwen2.5 Model Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO]   Model:            $simple_model_name" | tee -a "$summary_log"
    if [ "$NO_FA" = "true" ]; then
        echo "[INFO]   Mode:             without Flash Attention" | tee -a "$summary_log"
    else
        echo "[INFO]   Mode:             Flash Attention (GGML_DLFA_READY=1)" | tee -a "$summary_log"
    fi
    echo "[INFO]   Total tests:      $simple_model_total" | tee -a "$summary_log"
    echo "[INFO]   Passed:           $simple_model_pass_count" | tee -a "$summary_log"
    echo "[INFO]   Failed:           $simple_model_fail_count" | tee -a "$summary_log"
    echo "[INFO]   Total time:       ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    if [ $simple_model_fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] Tests failed:" | tee -a "$summary_log"
        for fail_item in "${simple_model_fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    fi

    echo "" | tee -a "$summary_log"
    if [ "$NO_FA" = "true" ]; then
        echo "[LLAMA_CPP_PASS] Standard inference tests passed!" | tee -a "$summary_log"
    else
        echo "[LLAMA_CPP_PASS] Flash Attention tests passed!" | tee -a "$summary_log"
    fi
    exit 0
}

run_simple_model_mode() {
    simple_model_intro
    simple_model_prepare_model
    simple_model_reset_state
    echo "[INFO] Total tests to run: $simple_model_total" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    for i in "${!SIMPLE_MODEL_CASES[@]}"; do
        local test_idx=$((i + 1))
        simple_model_execute_case "$test_idx" "${SIMPLE_MODEL_CASES[$i]}" "$simple_model_total"
    done
    simple_model_print_summary
}

if [ "$SIMPLE_MODEL" = true ]; then
    run_simple_model_mode
fi

declare -a SIMPLE_PERF_MODES=("POOL" "LEGACY")
declare -a simple_perf_fail_list=()
declare -A simple_perf_mode_times=()
declare -A simple_perf_mode_results=()

simple_perf_intro() {
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Performance Test Mode - Pool vs Legacy Mode Stress Testing" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Target Models: Qwen2.5-1.5B (fp16)" | tee -a "$summary_log"
    echo "[INFO] Test Type: Stress testing with multiple iterations" | tee -a "$summary_log"
    echo "[INFO] Comparison: 🔄 Pool Mode vs 🔧 Legacy Mode performance" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"
}

simple_perf_prepare_model() {
    simple_perf_base="${LOCAL_MODEL_PATH}"
    simple_perf_model_relative="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
    simple_perf_model_path="${simple_perf_base%/}/${simple_perf_model_relative}"
    if [ ! -f "$simple_perf_model_path" ]; then
        echo "[ERROR] Performance test model not found: $simple_perf_model_path" | tee -a "$summary_log"
        echo "[INFO] Please ensure the Qwen2.5-1.5B model is available for performance testing" | tee -a "$summary_log"
        exit 1
    fi
    echo "[INFO] Performance test model: $simple_perf_model_relative" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
}

simple_perf_reset_state() {
    simple_perf_fail_list=()
    simple_perf_fail_count=0
    simple_perf_iterations=5
    simple_perf_tokens=100
    simple_perf_prompt="Write a detailed explanation of artificial intelligence and machine learning, including their applications in modern technology and their potential impact on society."
    simple_perf_mode_times=()
    simple_perf_mode_results=()
}

simple_perf_mode_metadata() {
    local mode="$1"
    if [ "$mode" = "POOL" ]; then
        simple_perf_mode_env="GGML_GPTQ_USE_POOL=1"
        simple_perf_mode_icon="🔄"
        simple_perf_mode_desc="Pool Mode"
    else
        simple_perf_mode_env="GGML_GPTQ_USE_POOL=0"
        simple_perf_mode_icon="🔧"
        simple_perf_mode_desc="Legacy Mode"
    fi
}

simple_perf_run_iteration() {
    local iteration="$1"
    local temp_output
    local start_time
    local end_time
    local duration
    temp_output=$(mktemp)
    start_time=$(date +%s.%N)
    set +e
    bash -c "$simple_perf_mode_env ${build_dir_bin}/llama-cli -m '$simple_perf_model_path' -no-cnv -n $simple_perf_tokens --temp 0.7 --top-k 40 --top-p 0.9 -s $iteration -p '$simple_perf_prompt'" > "$temp_output" 2>&1
    local ret=$?
    end_time=$(date +%s.%N)
    duration=$(echo "$end_time - $start_time" | bc -l)
    if [ "$VERBOSE" = "true" ]; then
        echo "[INFO] Performance test output (VERBOSE):" | tee -a "$test_log"
        cat "$temp_output" | tee -a "$test_log"
        echo "-----------------------------" | tee -a "$test_log"
    fi
    set -e
    safe_rm -f "$temp_output"
    echo "$ret|$duration"
}

simple_perf_execute_mode() {
    local mode="$1"
    simple_perf_mode_metadata "$mode"
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "[INFO] Performance Testing: $simple_perf_mode_icon $simple_perf_mode_desc" | tee -a "$summary_log"
    echo "[INFO] Iterations: $simple_perf_iterations, Tokens: $simple_perf_tokens" | tee -a "$summary_log"
    echo "[INFO] ========================================" | tee -a "$summary_log"

    local mode_total_time=0
    local mode_success_count=0

    for iteration in $(seq 1 $simple_perf_iterations); do
        echo "[INFO] $simple_perf_mode_icon $simple_perf_mode_desc - Iteration $iteration/$simple_perf_iterations" | tee -a "$summary_log"
        IFS='|' read -r ret duration <<< "$(simple_perf_run_iteration "$iteration")"
        if [ "$ret" -eq 0 ]; then
            mode_success_count=$((mode_success_count + 1))
            mode_total_time=$(echo "$mode_total_time + $duration" | bc -l)
            echo "[PASS] $simple_perf_mode_icon Iteration $iteration: ${duration}s" | tee -a "$summary_log"
        else
            simple_perf_fail_count=$((simple_perf_fail_count + 1))
            echo "[FAIL] $simple_perf_mode_icon Iteration $iteration: Exit code $ret" | tee -a "$summary_log"
            simple_perf_fail_list+=("$simple_perf_mode_desc - Iteration $iteration (exit code $ret)")
        fi
    done

    if [ $mode_success_count -gt 0 ]; then
        local avg_time
        avg_time=$(echo "scale=3; $mode_total_time / $mode_success_count" | bc -l)
        simple_perf_mode_times["$mode"]=$avg_time
        simple_perf_mode_results["$mode"]="$mode_success_count/$simple_perf_iterations successful"
        echo "[INFO] $simple_perf_mode_icon $simple_perf_mode_desc Results: $mode_success_count/$simple_perf_iterations successful, Avg time: ${avg_time}s" | tee -a "$summary_log"
    else
        simple_perf_mode_times["$mode"]="N/A"
        simple_perf_mode_results["$mode"]="0/$simple_perf_iterations successful"
        echo "[FAIL] $simple_perf_mode_icon $simple_perf_mode_desc Results: All iterations failed" | tee -a "$summary_log"
    fi

    echo "" | tee -a "$summary_log"
}

simple_perf_print_summary() {
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Performance Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Model: $simple_perf_model_relative" | tee -a "$summary_log"
    echo "[INFO] Test Configuration: $simple_perf_iterations iterations, $simple_perf_tokens tokens each" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "[INFO] 🔄 Pool Mode Results: ${simple_perf_mode_results[POOL]}" | tee -a "$summary_log"
    echo "[INFO] 🔄 Pool Mode Avg Time: ${simple_perf_mode_times[POOL]}s" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "[INFO] 🔧 Legacy Mode Results: ${simple_perf_mode_results[LEGACY]}" | tee -a "$summary_log"
    echo "[INFO] 🔧 Legacy Mode Avg Time: ${simple_perf_mode_times[LEGACY]}s" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"

    if [ "${simple_perf_mode_times[POOL]}" != "N/A" ] && [ "${simple_perf_mode_times[LEGACY]}" != "N/A" ]; then
        local pool_time=${simple_perf_mode_times[POOL]}
        local legacy_time=${simple_perf_mode_times[LEGACY]}
        if (( $(echo "$pool_time < $legacy_time" | bc -l) )); then
            local improvement
            improvement=$(echo "scale=2; ($legacy_time - $pool_time) / $legacy_time * 100" | bc -l)
            echo "[INFO] 🚀 Performance: Pool Mode is ${improvement}% faster than Legacy Mode" | tee -a "$summary_log"
        elif (( $(echo "$pool_time > $legacy_time" | bc -l) )); then
            local degradation
            degradation=$(echo "scale=2; ($pool_time - $legacy_time) / $legacy_time * 100" | bc -l)
            echo "[INFO] ⚠️  Performance: Pool Mode is ${degradation}% slower than Legacy Mode" | tee -a "$summary_log"
        else
            echo "[INFO] ⚖️  Performance: Pool Mode and Legacy Mode have similar performance" | tee -a "$summary_log"
        fi
    fi

    echo "[INFO] Total test time: ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    if [ $simple_perf_fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] Performance tests failed:" | tee -a "$summary_log"
        for fail_item in "${simple_perf_fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    fi

    echo "" | tee -a "$summary_log"
    echo "[LLAMA_CPP_PASS] All performance tests completed successfully!" | tee -a "$summary_log"
    echo "[INFO] Performance comparison between Pool and Legacy modes completed" | tee -a "$summary_log"
    exit 0
}

run_simple_perf_mode() {
    simple_perf_intro
    simple_perf_prepare_model
    simple_perf_reset_state
    for mode in "${SIMPLE_PERF_MODES[@]}"; do
        simple_perf_execute_mode "$mode"
    done
    simple_perf_print_summary
}

if [ "$SIMPLE_PERF" = true ]; then
    run_simple_perf_mode
fi

declare -a SIMPLE_TP_SPLIT_MODES=("row" "layer" "none")
declare -a SIMPLE_TP_GPTQ_MODES=("POOL" "LEGACY")
declare -a simple_tp_fail_list=()

simple_tp_intro() {
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Simple TP Test Mode - Tensor Parallel Split Mode Testing" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"
}

simple_tp_prepare_model() {
    simple_tp_base="${LOCAL_MODEL_PATH}"
    if [ "$BIG_MODEL" = true ]; then
        simple_tp_model_relative="Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
        simple_tp_model_name="Qwen3-30B (Q4_K_M)"
        echo "[INFO] Target Model: $simple_tp_model_name (BIG MODEL)" | tee -a "$summary_log"
    else
        simple_tp_model_relative="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
        simple_tp_model_name="Qwen2.5-1.5B (fp16)"
        echo "[INFO] Target Model: $simple_tp_model_name (SMALL MODEL)" | tee -a "$summary_log"
    fi

    simple_tp_model_path="${simple_tp_base%/}/${simple_tp_model_relative}"
    if [ ! -f "$simple_tp_model_path" ]; then
        echo "[ERROR] TP test model not found: $simple_tp_model_path" | tee -a "$summary_log"
        if [ "$BIG_MODEL" = true ]; then
            echo "[INFO] Expected path: ${simple_tp_base%/}/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf" | tee -a "$summary_log"
        else
            echo "[INFO] Expected path: ${simple_tp_base%/}/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf" | tee -a "$summary_log"
        fi
        exit 1
    fi

    echo "[INFO] TP test model: $simple_tp_model_relative" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"

    if [ "$BIG_MODEL" = true ]; then
        simple_tp_tokens=10
        echo "[INFO] Using reduced token count ($simple_tp_tokens) for large model" | tee -a "$summary_log"
    else
        simple_tp_tokens=30
    fi
    simple_tp_prompt="What is 2+3? Answer with only the number."
    simple_tp_expected="5"
}

simple_tp_reset_state() {
    simple_tp_fail_list=()
    simple_tp_fail_count=0
    simple_tp_pass_count=0
    simple_tp_total=$((${#SIMPLE_TP_SPLIT_MODES[@]} * ${#SIMPLE_TP_GPTQ_MODES[@]}))
    simple_tp_current=0
}

simple_tp_mode_metadata() {
    local mode="$1"
    if [ "$mode" = "POOL" ]; then
        simple_tp_mode_env="GGML_GPTQ_USE_POOL=1"
        simple_tp_mode_icon="🔄"
        simple_tp_mode_desc="Pool Mode"
    else
        simple_tp_mode_env="GGML_GPTQ_USE_POOL=0"
        simple_tp_mode_icon="🔧"
        simple_tp_mode_desc="Legacy Mode"
    fi
}

simple_tp_execute_case() {
    local split_mode="$1"
    local gptq_mode="$2"
    simple_tp_current=$((simple_tp_current + 1))
    simple_tp_mode_metadata "$gptq_mode"
    local test_name="TP-${split_mode^^}-${gptq_mode}"

    echo "" | tee -a "$summary_log"
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "[INFO] Test $simple_tp_current/$simple_tp_total: $test_name ($simple_tp_mode_icon $simple_tp_mode_desc)" | tee -a "$summary_log"
    echo "[INFO] Split Mode: $split_mode" | tee -a "$summary_log"
    echo "[INFO] Model: $simple_tp_model_path" | tee -a "$summary_log"
    echo "[INFO] Environment: $simple_tp_mode_env" | tee -a "$summary_log"
    echo "[INFO] ========================================" | tee -a "$summary_log"
    echo "-----------------------------" | tee -a "$test_log"

    local start_time end_time duration ret temp_output output_content
    start_time=$(date +%s)
    set +e
    temp_output=$(mktemp)
    echo "[DEBUG] Running TP test with split-mode=$split_mode and $simple_tp_mode_env" | tee -a "$test_log"
    bash -c "$simple_tp_mode_env ${build_dir_bin}/llama-cli -m '$simple_tp_model_path' --split-mode '$split_mode' -no-cnv -n $simple_tp_tokens --temp 0.7 --top-k 40 --top-p 0.9 -s 42 -p '$simple_tp_prompt'" > "$temp_output" 2>&1
    ret=$?

    if [ "$VERBOSE" = "true" ]; then
        echo "[INFO] Test output (VERBOSE - showing all output):" | tee -a "$test_log"
        cat "$temp_output" | tee -a "$test_log"
    else
        echo "[INFO] Test output (first 10 lines):" | tee -a "$test_log"
        head -n 10 "$temp_output" | tee -a "$test_log"
    fi

    set -e
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo "-----------------------------" | tee -a "$test_log"

    if [ $ret -eq 0 ]; then
        output_content=$(cat "$temp_output")
        if echo "$output_content" | grep -qi "$simple_tp_expected"; then
            echo "[PASS] $test_name ($simple_tp_mode_icon $simple_tp_mode_desc) (${duration}s) ✓ Correctness verified" | tee -a "$summary_log"
            echo "[INFO] Split mode '$split_mode' with $simple_tp_mode_desc: Functionality ✓ Correctness ✓" | tee -a "$summary_log"
            echo "[INFO] Expected content '$simple_tp_expected' found in output" | tee -a "$summary_log"
            simple_tp_pass_count=$((simple_tp_pass_count + 1))
        else
            echo "[FAIL] $test_name ($simple_tp_mode_icon $simple_tp_mode_desc) (${duration}s) ❌ Correctness failed" | tee -a "$summary_log"
            echo "[WARN] Split mode '$split_mode' with $simple_tp_mode_desc: Functionality ✓ Correctness ❌" | tee -a "$summary_log"
            echo "[DEBUG] Expected content '$simple_tp_expected' not found in output" | tee -a "$summary_log"
            echo "[DEBUG] Actual output excerpt:" | tee -a "$test_log"
            echo "$output_content" | tail -n 10 | tee -a "$test_log"
            simple_tp_fail_count=$((simple_tp_fail_count + 1))
            simple_tp_fail_list+=("$test_name ($simple_tp_mode_desc, correctness validation failed)")
        fi
    else
        echo "[FAIL] $test_name ($simple_tp_mode_icon $simple_tp_mode_desc) (exit code: $ret, ${duration}s) ❌ Functionality failed" | tee -a "$summary_log"
        echo "[WARN] Split mode '$split_mode' with $simple_tp_mode_desc: Functionality ❌" | tee -a "$summary_log"
        simple_tp_fail_count=$((simple_tp_fail_count + 1))
        simple_tp_fail_list+=("$test_name ($simple_tp_mode_desc, exit code $ret)")
        echo "[DEBUG] Error output:" | tee -a "$test_log"
        tail -n 20 "$temp_output" | tee -a "$test_log"
    fi

    safe_rm -f "$temp_output"
}

simple_tp_split_mode_analysis() {
    echo "" | tee -a "$summary_log"
    echo "[INFO] Split Mode Performance Analysis:" | tee -a "$summary_log"
    for split_mode in "${SIMPLE_TP_SPLIT_MODES[@]}"; do
        local pool_test="TP-${split_mode^^}-POOL"
        local legacy_test="TP-${split_mode^^}-LEGACY"
        local pool_passed=true
        local legacy_passed=true

        for fail_item in "${simple_tp_fail_list[@]}"; do
            if [[ "$fail_item" == *"$pool_test"* ]]; then
                pool_passed=false
            fi
            if [[ "$fail_item" == *"$legacy_test"* ]]; then
                legacy_passed=false
            fi
        done

        if [ "$pool_passed" = true ] && [ "$legacy_passed" = true ]; then
            echo "[INFO] ✅ Split mode '$split_mode': Both Pool and Legacy modes working" | tee -a "$summary_log"
        elif [ "$pool_passed" = true ]; then
            echo "[INFO] 🔄 Split mode '$split_mode': Only Pool mode working" | tee -a "$summary_log"
        elif [ "$legacy_passed" = true ]; then
            echo "[INFO] 🔧 Split mode '$split_mode': Only Legacy mode working" | tee -a "$summary_log"
        else
            echo "[WARN] ❌ Split mode '$split_mode': Both modes failed" | tee -a "$summary_log"
        fi
    done
}

simple_tp_print_summary() {
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Simple TP Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Model: $simple_tp_model_relative" | tee -a "$summary_log"
    echo "[INFO] Split modes tested: ${SIMPLE_TP_SPLIT_MODES[*]}" | tee -a "$summary_log"
    echo "[INFO] GPTQ modes tested: ${SIMPLE_TP_GPTQ_MODES[*]}" | tee -a "$summary_log"
    echo "[INFO] Total tests: $simple_tp_total" | tee -a "$summary_log"
    echo "[INFO] Passed: $simple_tp_pass_count" | tee -a "$summary_log"
    echo "[INFO] Failed: $simple_tp_fail_count" | tee -a "$summary_log"
    echo "[INFO] Total time: ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    simple_tp_split_mode_analysis

    if [ $simple_tp_fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] TP tests failed:" | tee -a "$summary_log"
        for fail_item in "${simple_tp_fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    fi

    echo "" | tee -a "$summary_log"
    echo "[LLAMA_CPP_PASS] All TP tests passed successfully!" | tee -a "$summary_log"
    echo "[INFO] Tensor Parallel implementation verified across all split modes" | tee -a "$summary_log"
    exit 0
}

run_simple_tp_mode() {
    simple_tp_intro
    simple_tp_prepare_model
    simple_tp_reset_state
    for split_mode in "${SIMPLE_TP_SPLIT_MODES[@]}"; do
        for gptq_mode in "${SIMPLE_TP_GPTQ_MODES[@]}"; do
            simple_tp_execute_case "$split_mode" "$gptq_mode"
        done
    done
    simple_tp_print_summary
}

if [ "$SIMPLE_TP" = true ]; then
    run_simple_tp_mode
fi

declare -a test_cases_part1=()
declare -a test_cases_part2=()
declare -a test_cases=()
declare -a qwen_model_tests=()
declare -a multi_turn_tests=()
declare -a full_suite_fail_list=()
declare -a test_results_names=()
declare -a test_results_status=()
full_suite_fail_count=0

detect_gpu_resources() {
    local gpu_count=0
    local total_memory_gb=0
    local gpu_memory_list=""

    # Check if nvidia-smi is available
    if command -v nvidia-smi >/dev/null 2>&1; then
        # Get number of GPUs
        gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | head -1)
        if [ -z "$gpu_count" ] || [ "$gpu_count" = "count" ]; then
            gpu_count=0
        fi

        # Get memory for each GPU (in MB, then convert to GB)
        if [ "$gpu_count" -gt 0 ]; then
            local memory_values
            memory_values=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null)
            local gpu_memory_array=()
            local total_memory_mb=0
            local index=0
            while IFS= read -r mem_mb; do
                if [[ "$mem_mb" =~ ^[0-9]+$ ]]; then
                    local mem_gb=$((mem_mb / 1024))
                    gpu_memory_array[index]=$mem_gb
                    total_memory_mb=$((total_memory_mb + mem_mb))
                    index=$((index + 1))
                fi
            done <<< "$memory_values"
            total_memory_gb=$((total_memory_mb / 1024))

            # Create comma-separated list of GPU memories
            gpu_memory_list=$(IFS=,; echo "${gpu_memory_array[*]}")
        fi
    fi

    echo "$gpu_count|$total_memory_gb|$gpu_memory_list"
}

check_model_file() {
    local model_path="$1"
    local model_name="$2"
    if [ ! -f "$model_path" ]; then
        echo "[WARN] Model file not found: $model_path ($model_name)" | tee -a "$summary_log"
        return 1
    fi
    if [ ! -r "$model_path" ]; then
        echo "[WARN] Model file not readable: $model_path ($model_name)" | tee -a "$summary_log"
        return 1
    fi
    echo "[INFO] Model file validated: $model_name" | tee -a "$summary_log"
    return 0
}

parse_yaml_tests() {
    local model_name="$1"
    local yaml_file="$2"
    awk -v model="$model_name" '
    BEGIN { in_model = 0; in_tests = 0; }
    /^  [a-zA-Z0-9_.-]+:/ {
        current_model = $1
        gsub(/:/, "", current_model)
        in_model = (current_model == model)
        in_tests = 0
    }
    in_model && /^    tests:/ { in_tests = 1; next }
    in_model && in_tests && /^      [a-zA-Z0-9_-]+:/ {
        test_name = $1
        gsub(/:/, "", test_name)
        current_test = test_name
    }
    in_model && in_tests && /^        prompt:/ {
        gsub(/^        prompt: "/, "")
        gsub(/"$/, "")
        prompt = $0
    }
    in_model && in_tests && /^        expected_content:/ {
        gsub(/^        expected_content: "/, "")
        gsub(/"$/, "")
        expected = $0
        print prompt "|" expected "|" current_test
    }
    ' "$yaml_file"
}

validate_content_enhanced() {
    local model_name="$1"
    local test_name="$2"
    local output_content="$3"
    local expected_content="$4"

    echo "[DEBUG] Enhanced validation started" | tee -a "$test_log"
    echo "[DEBUG] Expected: '$expected_content'" | tee -a "$test_log"

    if echo "$output_content" | grep -q "$expected_content"; then
        echo "[DEBUG] Found '$expected_content' in output" | tee -a "$test_log"
        echo "Found expected '$expected_content' in output"
        return 0
    else
        echo "[DEBUG] '$expected_content' not found in output" | tee -a "$test_log"
        echo "Expected '$expected_content' not found in output"
        return 1
    fi
}

validate_content_only() {
    local model_name="$1"
    local test_name="$2"
    local output_content="$3"
    local expected_content="$4"

    echo "[INFO] Validating content for $model_name - $test_name"
    echo "[DEBUG] Calling validate_content_enhanced..." | tee -a "$test_log"
    validation_result=$(validate_content_enhanced "$model_name" "$test_name" "$output_content" "$expected_content")
    local enhanced_result=$?
    echo "[DEBUG] Enhanced validation returned: $enhanced_result" | tee -a "$test_log"

    if [ $enhanced_result -eq 0 ]; then
        echo "[PASS] Content validation: $validation_result"
        return 0
    else
        echo "[FAIL] Content validation: $validation_result"
        return 1
    fi
}

add_qwen_model_tests() {
    echo "[INFO] Adding Qwen model tests" | tee -a "$summary_log"

    # Add the Qwen model test script
    local test_script="${REPO_PATH}/tests/test_qwen_models.sh"

    if [ ! -f "$test_script" ]; then
        echo "[WARN] Qwen model test script not found: $test_script" | tee -a "$summary_log"
        qwen_model_tests=()
        return
    fi

    if [ ! -x "$test_script" ]; then
        echo "[INFO] Making Qwen model test script executable" | tee -a "$summary_log"
        chmod +x "$test_script"
    fi

    # Add the test to the array
    qwen_model_tests=("$test_script")
    echo "[INFO] Added Qwen model test script" | tee -a "$summary_log"
}

add_multi_turn_tests() {
    echo "[INFO] Adding multi-turn conversation test" | tee -a "$summary_log"

    # Add the multi-turn conversation test script
    local test_script="${REPO_PATH}/tests/test_multi_turn_chat.sh"

    if [ ! -f "$test_script" ]; then
        echo "[WARN] Multi-turn test script not found: $test_script" | tee -a "$summary_log"
        return
    fi

    if [ ! -x "$test_script" ]; then
        echo "[INFO] Making multi-turn test script executable" | tee -a "$summary_log"
        chmod +x "$test_script"
    fi

    # Add the test to the array
    multi_turn_tests+=("$test_script")
    echo "[INFO] Added multi-turn conversation test" | tee -a "$summary_log"
}

validate_qwen_output() {
    local model_name="$1"
    local test_name="$2"
    local prompt="$3"
    local expected_content="$4"
    local model_path="$5"
    local test_mode="$6"

    echo "[INFO] Running correctness test: $model_name - $test_name ($test_mode)"
    local output_file
    output_file=$(mktemp)

    # Handle CUDA_VISIBLE_DEVICES for different test modes
    local saved_cuda_visible_devices=""
    local cuda_visible_devices_modified=false

    if [[ "$test_mode" == "dual_gpu_16GB" ]]; then
        # For dual GPU 16GB mode, find and set two 16GB GPUs
        local gpu_info
        gpu_info=$(detect_gpu_resources)
        IFS='|' read -r gpu_count total_memory_gb gpu_memory_list <<< "$gpu_info"
        IFS=',' read -r -a gpu_memories <<< "$gpu_memory_list"

        local sixteen_gb_gpus=()
        for i in "${!gpu_memories[@]}"; do
            if [ "${gpu_memories[i]}" -eq 16 ]; then
                sixteen_gb_gpus+=("$i")
                if [ ${#sixteen_gb_gpus[@]} -eq 2 ]; then
                    break
                fi
            fi
        done

        if [ ${#sixteen_gb_gpus[@]} -eq 2 ]; then
            if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
                saved_cuda_visible_devices="$CUDA_VISIBLE_DEVICES"
                cuda_visible_devices_modified=true
            fi
            export CUDA_VISIBLE_DEVICES="${sixteen_gb_gpus[0]},${sixteen_gb_gpus[1]}"
            echo "[INFO] Using dual GPU mode for Qwen3: GPUs ${sixteen_gb_gpus[0]},${sixteen_gb_gpus[1]} (16GB each)"
        else
            echo "[WARN] Could not find two 16GB GPUs for dual GPU mode, falling back to single GPU"
        fi
    elif [[ "$test_mode" == single_gpu_* ]]; then
        echo "[INFO] Using single GPU mode for Qwen3: $test_mode"
    fi

    export DLEOL_DISABLE_CU_MATMUL=1     # bugid : 16579. | when use CUDA GRAPH

    # Print command before executing for easier debugging
    echo "[INFO] Executing command: ${build_dir_bin}/llama-cli -m \"$model_path\" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -ngl 999 -p \"$prompt\""

    "${build_dir_bin}/llama-cli" -m "$model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -ngl 999 -p "$prompt" 2>&1 | tee "$output_file"
    local cmd_result=$?

    # Restore CUDA_VISIBLE_DEVICES if it was modified
    if [ "$cuda_visible_devices_modified" = true ]; then
        if [ -n "$saved_cuda_visible_devices" ]; then
            export CUDA_VISIBLE_DEVICES="$saved_cuda_visible_devices"
            echo "[INFO] Restored CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"
        else
            unset CUDA_VISIBLE_DEVICES
            echo "[INFO] Unset CUDA_VISIBLE_DEVICES (was not set before)"
        fi
    fi
    if [ $cmd_result -ne 0 ]; then
        echo "[FAIL] llama-cli command failed with exit code $cmd_result"
        echo "[DEBUG] Command output:" >> "$test_log"
        cat "$output_file" >> "$test_log"

        # Try to capture crash information from system logs when core dump occurs
        if command -v dmesg >/dev/null 2>&1; then
            echo "[DEBUG] Recent kernel messages (possible crash info):" >> "$test_log"
            dmesg | tail -n 100 >> "$test_log" 2>&1 || true
        else
            echo "[DEBUG] 'dmesg' command not available; cannot capture kernel crash messages." >> "$test_log"
        fi

        safe_rm -f "$output_file"
        return 1
    fi

    local output_content
    output_content=$(cat "$output_file")
    echo "[DEBUG] Model response extract:" | tee -a "$test_log"
    local response_part
    response_part=$(echo "$output_content" | sed -n '/^$/,/llama_perf_sampler_print:/p' | head -n -1 | tail -n +2)
    echo "$response_part" | tee -a "$test_log"

    echo "[DEBUG] Starting content validation..." | tee -a "$test_log"
    validate_content_only "$model_name" "$test_name" "$output_content" "$expected_content"
    local validation_result=$?
    echo "[DEBUG] Validation completed with result: $validation_result" | tee -a "$test_log"

    if [ $validation_result -eq 0 ]; then
        echo "[PASS] Correctness test passed: $model_name - $test_name"
        safe_rm -f "$output_file"
        return 0
    else
        echo "[FAIL] Correctness test failed: $model_name - $test_name"
        echo "[DEBUG] Failed test output:" >> "$test_log"
        cat "$output_file" >> "$test_log"
        safe_rm -f "$output_file"
        return 1
    fi
}

add_tokenizer_vocab_tests() {
    echo "[INFO] Adding vocab tests for all platforms"
    for vocab in models/*.gguf; do
        local inp="${vocab}.inp"
        local out="${vocab}.out"
        if [[ -f "$inp" && -f "$out" ]]; then
            test_cases_part2+=("${build_dir_bin}/test-tokenizer-0 $vocab")
        fi
    done
}

full_suite_prepare_part1_cases() {
    test_cases_part1=(
        "${build_dir_bin}/test-arg-parser"
        "${build_dir_bin}/test-autorelease"
        "${build_dir_bin}/test-backend-ops"
        "${build_dir_bin}/test-c"
        "${build_dir_bin}/test-chat"
        "${build_dir_bin}/test-chat-parser"
        "${build_dir_bin}/test-chat-template"
        "${build_dir_bin}/test-gbnf-validator grammars/json.gbnf -c '{\"name\": \"Alice\", \"age\": 25}'"
        "${build_dir_bin}/test-gbnf-validator grammars/json.gbnf -c '{\"name\": \"Bob\", \"age\": thirty}'"
        "${build_dir_bin}/test-gbnf-validator grammars/arithmetic.gbnf -c 'x = 5'"
        "${build_dir_bin}/test-gbnf-validator grammars/arithmetic.gbnf -c 'x = 5 +'"
        "${build_dir_bin}/test-gbnf-validator grammars/list.gbnf -c '- First item\n- Second item\n'"
        "${build_dir_bin}/test-gguf"
        "${build_dir_bin}/test-grammar-integration"
        "${build_dir_bin}/test-grammar-parser"
        "${build_dir_bin}/test-json-partial"
        "${build_dir_bin}/test-llama-grammar"
        "${build_dir_bin}/test-log"
        "${build_dir_bin}/test-model-load-cancel"
        "${build_dir_bin}/test-mtmd-c-api"
        "${build_dir_bin}/test-regex-partial"
        "${build_dir_bin}/test-sampling"
        "${build_dir_bin}/test-thread-safety --prompt 'hello, llama.cpp' --model ${LOCAL_MODEL_PATH}/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
    )
    if [ "${ARCH}" != "loongarch64" ]; then
        echo "[INFO] Adding test-json-schema-to-grammar (not LoongArch64 platform)" | tee -a "$summary_log"
        test_cases_part1+=("${build_dir_bin}/test-json-schema-to-grammar")
    else
        echo "[INFO] Skipping test-json-schema-to-grammar on LoongArch64 platform (because the ggml-ci node lacks Python 3.8)" | tee -a "$summary_log"
    fi
}

full_suite_prepare_part2_cases() {
    test_cases_part2=(
        "${build_dir_bin}/test-tokenizer-1-bpe models/ggml-vocab-llama-bpe.gguf"
        "${build_dir_bin}/test-tokenizer-1-spm models/ggml-vocab-llama-spm.gguf"
    )
    add_tokenizer_vocab_tests
}

full_suite_record_backend_subtests() {
    local temp_output="$1"
    local test_name_for_summary="$2"
    local temp_clean
    temp_clean=$(mktemp)
    sed $'s/\033\[[0-9;]*m//g' "$temp_output" > "$temp_clean"

    # Track the last seen test to associate status-only lines with it
    local last_seen_test=""
    local last_seen_params=""

    local regex_pattern='^[[:space:]]*([A-Z_]+)\(([^)]+)\):[[:space:]]*(.+)$'
    local status_pattern='^[[:space:]]*(not supported|skipping|passed|failed|OK|\.OK)([[:space:]].*)?$'

    while IFS= read -r line; do
        if [[ "$line" =~ $regex_pattern ]]; then
            local sub_test_name="${BASH_REMATCH[1]}"
            local sub_test_params="${BASH_REMATCH[2]}"
            local sub_test_result="${BASH_REMATCH[3]}"

            local sub_case_name="${test_name_for_summary}-${sub_test_name}(${sub_test_params})"

            # Remember this test for potential future status-only lines
            last_seen_test="$sub_case_name"
            last_seen_params="$sub_test_params"

            # Determine status for this sub-test line
            local sub_status
            if [[ "$sub_test_result" =~ ^OK ]] || [[ "$sub_test_result" =~ \.OK$ ]]; then
                sub_status="PASSED"
            elif [[ "$sub_test_result" =~ skipping ]] || [[ "$sub_test_result" =~ "not supported" ]]; then
                sub_status="skipped"
            else
                sub_status="FAILED"
            fi

            local already_recorded=false
            for i in "${!test_results_names[@]}"; do
                if [ "${test_results_names[$i]}" = "$sub_case_name" ]; then
                    already_recorded=true
                    break
                fi
            done

            if [ "$already_recorded" = false ]; then
                test_results_names+=("$sub_case_name")
                test_results_status+=("$sub_status")
            fi
        elif [[ -n "$last_seen_test" && "$line" =~ $status_pattern ]]; then
            # This is a status-only line (no test name), associate with last seen test
            local status_text="${BASH_REMATCH[1]}"
            local sub_status
            if [[ "$status_text" =~ ^OK ]] || [[ "$status_text" =~ \.OK$ ]] || [[ "$status_text" =~ passed ]]; then
                sub_status="PASSED"
            elif [[ "$status_text" =~ skipping ]] || [[ "$status_text" =~ "not supported" ]]; then
                sub_status="skipped"
            else
                sub_status="FAILED"
            fi

            # Update the status for the last seen test
            for i in "${!test_results_names[@]}"; do
                if [ "${test_results_names[$i]}" = "$last_seen_test" ]; then
                    test_results_status[$i]="$sub_status"
                    break
                fi
            done
        fi
    done < "$temp_clean"
    safe_rm -f "$temp_clean"
}

full_suite_print_case_summary() {
    local test_name_for_summary="$1"
    local ret="$2"
    local had_sub_tests="$3"
    echo "" | tee -a "$summary_log"
    echo "================================================================================" | tee -a "$summary_log"
    echo "FORMATTED TEST RESULTS SUMMARY:" | tee -a "$summary_log"
    echo "================================================================================" | tee -a "$summary_log"
    if [ "$had_sub_tests" = true ]; then
        for i in "${!test_results_names[@]}"; do
            if [[ "${test_results_names[$i]}" =~ ^${test_name_for_summary}- ]]; then
                local case_name="${test_results_names[$i]}"
                local case_status="${test_results_status[$i]}"
                local case_result_lower
                case_result_lower=$(echo "$case_status" | tr '[:upper:]' '[:lower:]')
                echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result_lower}" | tee -a "$summary_log"
            fi
        done
    else
        if [ "$ret" -eq 0 ]; then
            echo "CASE_NAME: ${test_name_for_summary}, CASE_RESULT: passed" | tee -a "$summary_log"
        else
            echo "CASE_NAME: ${test_name_for_summary}, CASE_RESULT: failed" | tee -a "$summary_log"
        fi
    fi
    echo "================================================================================" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
}

full_suite_run_qwen_case() {
    local test_script="$1"
    local test_name_for_summary="qwen_model_tests"

    echo "Running Qwen model tests from script: $test_script" | tee -a "$summary_log"
    echo "Running Qwen model tests from script: $test_script" >> "$test_log"
    echo "-----------------------------" | tee -a "$test_log"

    local start_time end_time duration
    start_time=$(date +%s)

    set +e
    # Output to both stdout and log file using tee
    bash "$test_script" 2>&1 | tee -a "$test_log"
    local ret=${PIPESTATUS[0]}
    set -e

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ $re_rene 0 ]; then
        full_suite_fail_list+=("$test_name_for_summary (exit code $ret)")
        full_suite_fail_count=$((full_suite_fail_count + 1))
        test_results_names+=("$test_name_for_summary")
        test_results_status+=("FAILED")
    else
        test_results_names+=("$test_name_for_summary")
        test_results_status+=("PASSED")
    fi

    echo "-----------------------------" | tee -a "$test_log"
    full_suite_print_case_summary "$test_name_for_summary" "$ret" false
}

full_suite_run_regular_case() {
    local test_case="$1"
    local test_bin
    test_bin=$(echo "$test_case" | awk '{print $1}')
    local test_name_for_summary
    test_name_for_summary=$(basename "$test_bin" 2>/dev/null || echo "$test_case")

    local word_count
    word_count=$(echo "$test_case" | wc -w)
    if [ "$word_count" -gt 1 ]; then
        local test_args
        test_args=$(echo "$test_case" | awk '{$1=""; print $0}' | sed 's/^ //')
        local test_args_short
        test_args_short=$(echo "$test_args" | sed 's|/LocalRun/[^/]*/[^/]*/LLM/model/|...|g' | sed 's|models/|...|g' | cut -c1-80)
        test_name_for_summary="${test_name_for_summary} ${test_args_short}"
    fi

    if [ ! -x "$test_bin" ]; then
        echo "[LLAMA_CPP_FAIL] $test_bin does not exist or is not executable, skipping" | tee -a "$summary_log"
        full_suite_fail_count=$((full_suite_fail_count + 1))
        full_suite_fail_list+=("$test_case (not found or not executable)")
        test_results_names+=("$test_name_for_summary")
        test_results_status+=("FAILED (not executable)")
        full_suite_print_case_summary "$test_name_for_summary" 1 false
        return
    fi

    echo "Running: $test_case" | tee -a "$summary_log"
    echo "Running: $test_case" >> "$test_log"
    echo "-----------------------------" | tee -a "$test_log"
    local start_time
    start_time=$(date +%s)
    local temp_output
    temp_output=$(mktemp)

    local test_case_with_env
    if [[ "$test_case" == *"test-backend-ops"* ]]; then
        test_case_with_env="GGML_CUDA_DISABLE_GRAPHS=1 $test_case"
    else
        test_case_with_env="$test_case"
    fi

    set +e
    eval "$test_case_with_env" > "$temp_output" 2>&1
    local ret=$?
    set -e

    cat "$temp_output" | tee -a "$test_log"
    echo "-----------------------------" | tee -a "$test_log"

    local had_sub_tests=false
    if [[ "$test_name_for_summary" == "test-backend-ops" ]] || [[ "$test_name_for_summary" =~ test-backend-ops ]]; then
        had_sub_tests=true
        full_suite_record_backend_subtests "$temp_output" "$test_name_for_summary"
    fi

    safe_rm -f "$temp_output"
    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    if [ $ret -ne 0 ]; then
        full_suite_fail_count=$((full_suite_fail_count + 1))
        full_suite_fail_list+=("$test_case (exit code $ret)")
        if [ "$had_sub_tests" = false ]; then
            test_results_names+=("$test_name_for_summary")
            test_results_status+=("FAILED")
        fi
    elif [ "$had_sub_tests" = false ]; then
        test_results_names+=("$test_name_for_summary")
        test_results_status+=("PASSED")
    fi

    full_suite_print_case_summary "$test_name_for_summary" "$ret" "$had_sub_tests"
}

full_suite_run_multi_turn_case() {
    local test_script="$1"
    local test_name_for_summary="multi_turn_tests"

    echo "Running multi-turn conversation tests from script: $test_script" | tee -a "$summary_log"
    echo "Running multi-turn conversation tests from script: $test_script" >> "$test_log"
    echo "-----------------------------" | tee -a "$test_log"

    local start_time end_time duration
    start_time=$(date +%s)

    set +e
    # Output to both stdout and log file using tee
    bash "$test_script" 2>&1 | tee -a "$test_log"
    local ret=${PIPESTATUS[0]}
    set -e

    end_time=$(date +%s)
    duration=$((end_time - start_time))

    if [ $ret -ne 0 ]; then
        full_suite_fail_list+=("$test_name_for_summary (exit code $ret)")
        full_suite_fail_count=$((full_suite_fail_count + 1))
        test_results_names+=("$test_name_for_summary")
        test_results_status+=("FAILED")
    else
        test_results_names+=("$test_name_for_summary")
        test_results_status+=("PASSED")
    fi

    echo "-----------------------------" | tee -a "$test_log"
    full_suite_print_case_summary "$test_name_for_summary" "$ret" false
}

full_suite_run_all_cases() {
    full_suite_fail_count=0
    full_suite_fail_list=()
    test_results_names=()
    test_results_status=()
    for test_case in "${test_cases[@]}"; do
        # Check if it's a test script (ends with .sh)
        if [[ "$test_case" == *.sh ]]; then
            # Determine which type of test script it is
            if [[ "$test_case" == *"test_qwen_models.sh"* ]]; then
                full_suite_run_qwen_case "$test_case"
            elif [[ "$test_case" == *"test_multi_turn_chat.sh"* ]]; then
                full_suite_run_multi_turn_case "$test_case"
            else
                # Generic script execution
                full_suite_run_regular_case "$test_case"
            fi
        elif [[ "$test_case" == *"|"* ]]; then
            # Legacy format - should not be used anymore
            echo "[WARN] Legacy test format detected, skipping: $test_case" | tee -a "$summary_log"
        else
            full_suite_run_regular_case "$test_case"
        fi
    done
}

full_suite_print_summary() {
    test_end_time=$(date +%s)
    local test_duration=$((test_end_time - test_start_time))
    local test_hours=$((test_duration / 3600))
    local test_minutes=$(((test_duration % 3600) / 60))
    local test_seconds=$((test_duration % 60))

    echo "[INFO] Test suite completed at: $(date)" | tee -a "$summary_log"
    if [ $test_hours -gt 0 ]; then
        echo "[INFO] Total test time: ${test_hours}h ${test_minutes}m ${test_seconds}s (${test_duration} seconds)" | tee -a "$summary_log"
    elif [ $test_minutes -gt 0 ]; then
        echo "[INFO] Total test time: ${test_minutes}m ${test_seconds}s (${test_duration} seconds)" | tee -a "$summary_log"
    else
        echo "[INFO] Total test time: ${test_seconds}s" | tee -a "$summary_log"
    fi

    local total_tests=${#test_cases[@]}
    local passed_tests=$((total_tests - full_suite_fail_count))
    local qwen_tests_count=${#qwen_model_tests[@]}

    echo "" | tee -a "$summary_log"
    echo "[INFO] Test Statistics:" | tee -a "$summary_log"
    echo "[INFO]   Total tests: ${total_tests}" | tee -a "$summary_log"
    echo "[INFO]   Passed: ${passed_tests}" | tee -a "$summary_log"
    echo "[INFO]   Failed: ${full_suite_fail_count}" | tee -a "$summary_log"
    echo "[INFO]   Qwen model tests: ${qwen_tests_count}" | tee -a "$summary_log"

    if [ "${qwen_tests_count}" -gt 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[INFO] Qwen Model Test Summary:" | tee -a "$summary_log"
        echo "[INFO] ==============================" | tee -a "$summary_log"
        local qwen_failed=0
        for fail_item in "${full_suite_fail_list[@]}"; do
            if [[ "$fail_item" == *"qwen"* ]] || [[ "$fail_item" == *"Qwen"* ]]; then
                qwen_failed=$((qwen_failed + 1))
            fi
        done
        local qwen_passed=$((qwen_tests_count - qwen_failed))
        echo "[INFO]   Qwen tests passed: ${qwen_passed}/${qwen_tests_count}" | tee -a "$summary_log"
        echo "[INFO]   Qwen tests failed: ${qwen_failed}/${qwen_tests_count}" | tee -a "$summary_log"
        if [ ${qwen_failed} -eq 0 ]; then
            echo "[INFO]   ✅ All Qwen model tests passed successfully!" | tee -a "$summary_log"
            echo "[INFO]   ✅ Qwen2 MoE, Qwen2.5, and Qwen3 models show correct functionality" | tee -a "$summary_log"
        else
            echo "[WARN]   ⚠️  Some Qwen model tests failed. Please check the failed test details above." | tee -a "$summary_log"
            echo "[INFO]   Failed Qwen tests:" | tee -a "$summary_log"
            for fail_item in "${full_suite_fail_list[@]}"; do
                if [[ "$fail_item" == *"qwen"* ]] || [[ "$fail_item" == *"Qwen"* ]]; then
                    echo "[INFO]     - $fail_item" | tee -a "$summary_log"
                fi
            done
        fi
        echo "[INFO] ==============================" | tee -a "$summary_log"
        echo "[INFO] Qwen Model Test Coverage:" | tee -a "$summary_log"
        echo "[INFO]   - Qwen2 MoE (1.5B): Mixture of Experts architecture correctness" | tee -a "$summary_log"
        echo "[INFO]   - Qwen2.5 (1.5B): Standard instruction-following accuracy" | tee -a "$summary_log"
        echo "[INFO]   - Qwen3 (30B): Large-scale model reasoning (selected tests)" | tee -a "$summary_log"
        echo "[INFO]   - Content validation: Output compared against expected answers" | tee -a "$summary_log"
        echo "[INFO]   - Deterministic testing: Fixed temperature (0.0) and seed (42)" | tee -a "$summary_log"
        echo "[INFO]   - Test categories: Math calculation, language understanding, reasoning" | tee -a "$summary_log"
        echo "[INFO]   - No performance benchmarking - focus on correctness only" | tee -a "$summary_log"
    else
        echo "" | tee -a "$summary_log"
        echo "[WARN] No Qwen model tests were executed" | tee -a "$summary_log"
        echo "[WARN] Please ensure Qwen2, Qwen2.5, and Qwen3 model files are available in ${LOCAL_MODEL_PATH}" | tee -a "$summary_log"
    fi

    local test_log_size
    test_log_size=$(du -h "$test_log" | cut -f1)
    local summary_log_size
    summary_log_size=$(du -h "$summary_log" | cut -f1)
    echo "[INFO] Log files created:" | tee -a "$summary_log"
    echo "[INFO]   Detailed log: $test_log ($test_log_size)" | tee -a "$summary_log"
    echo "[INFO]   Summary log: $summary_log ($summary_log_size)" | tee -a "$summary_log"

    if [ $full_suite_fail_count -ne 0 ]; then
        echo "=============================" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] $full_suite_fail_count test(s) failed:" | tee -a "$summary_log"
        for fail_item in "${full_suite_fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        echo "[INFO] Detailed failure logs can be found in: $test_log" | tee -a "$summary_log"
        echo "" | tee -a "$summary_log"
        echo "OVERALL RESULT: FAILED (${passed_tests}/${total_tests} tests passed)" | tee -a "$summary_log"
        exit 1
    fi

    echo "=============================" | tee -a "$summary_log"
    echo "[LLAMA_CPP_PASS] All tests passed!" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "OVERALL RESULT: PASSED (${total_tests}/${total_tests} tests passed)" | tee -a "$summary_log"
}
run_full_test_suite() {
    echo "[INFO] Running full test suite for all platforms" | tee -a "$summary_log"
    echo "[INFO] Note: All platforms now use the complete test set for comprehensive coverage" | tee -a "$summary_log"

    full_suite_prepare_part1_cases
    echo "[INFO] Preparing Qwen model tests for correctness validation" | tee -a "$summary_log"
    add_qwen_model_tests
    add_multi_turn_tests
    full_suite_prepare_part2_cases

    test_cases=(
        "${test_cases_part1[@]}"
        "${test_cases_part2[@]}"
        "${qwen_model_tests[@]}"
        "${multi_turn_tests[@]}"
    )

    full_suite_run_all_cases
    full_suite_print_summary
}

dispatch_test_modes() {
    if [ "$SIMPLE_TEST" = true ]; then
        run_simple_test_mode
    fi

    if [ "$SIMPLE_MODEL" = true ]; then
        run_simple_model_mode
    fi

    if [ "$SIMPLE_PERF" = true ]; then
        run_simple_perf_mode
    fi

    if [ "$SIMPLE_TP" = true ]; then
        run_simple_tp_mode
    fi

    run_full_test_suite
}

main() {
    set_global_defaults
    parse_arguments "$@"
    validate_ci_mode
    init_logging
    setup_environment_context
    configure_ccache
    prepare_build_dir
    set_common_runtime_env
    dispatch_test_modes
}

main "$@"

