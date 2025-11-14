#!/bin/bash
set -e

# Load shared utility helpers (includes safe_rm)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Parse command line arguments
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
            echo "Usage: $0 [--ci-test] [--binary-path <path>] [--simple-test] [--simple-model] [--simple-perf] [--simple-tp] [--big] [--verbose] [--no-fa]"
            echo "  --ci-test: Enable CI test mode using external binaries"
            echo "  --binary-path: Path to the release directory containing binaries"
            echo "  --simple-test: Run only backend-ops tests (MUL_MAT and DLFA)"
            echo "  --simple-model: Run only Qwen2.5 model tests with GGML_DLFA_READY=1"
            echo "  --simple-perf: Run model performance stress tests comparing Pool vs Legacy modes"
            echo "  --simple-tp: Run Tensor Parallel tests with split-mode (row/layer/none)"
            echo "  --big: Use large model for TP tests (Qwen3-30B instead of Qwen2.5-1.5B)"
            echo "  --verbose: Enable verbose output (show detailed logs in real-time)"
            echo "  --no-fa: Disable Flash Attention (test with standard inference)"
            exit 1
            ;;
    esac
done

# Validate CI test mode parameters
if [ "$CI_TEST_MODE" = true ]; then
    if [ -z "$BINARY_PATH" ]; then
        echo "[ERROR] --binary-path is required when using --ci-test mode"
        echo "Usage: $0 --ci-test --binary-path <path>"
        exit 1
    fi

    if [ ! -d "$BINARY_PATH" ]; then
        echo "[ERROR] Binary path does not exist: $BINARY_PATH"
        exit 1
    fi

    echo "[INFO] CI Test Mode enabled"
    echo "[INFO] Using binaries from: $BINARY_PATH"

    # Verify essential binaries exist in the provided path
    essential_binaries=("llama-cli" "test-c")
    for binary in "${essential_binaries[@]}"; do
        if [ ! -x "$BINARY_PATH/$binary" ]; then
            echo "[ERROR] Essential binary not found or not executable: $BINARY_PATH/$binary"
            exit 1
        fi
    done
    echo "[INFO] Essential binaries validated successfully"
fi

# Record start time for overall test timing
test_start_time=$(date +%s)

# Create logs directory
logs_dir="/LocalRun/$(whoami)/logs/llama_cpp_test"
mkdir -p "$logs_dir"

# Log files
test_log="${logs_dir}/test_$(date +%Y%m%d_%H%M%S).log"
summary_log="${logs_dir}/test_summary_$(date +%Y%m%d_%H%M%S).log"

echo "[INFO] Test suite started at: $(date)" | tee -a "$test_log"
echo "[INFO] Detailed logs will be saved to: $test_log" | tee -a "$summary_log"
echo "[INFO] Summary will be saved to: $summary_log" | tee -a "$summary_log"
if [ "$VERBOSE" = "true" ]; then
    echo "[INFO] VERBOSE MODE ENABLED - All test outputs will be displayed in real-time" | tee -a "$summary_log"
else
    echo "[INFO] Standard mode - Only limited output shown (use --verbose for full output)" | tee -a "$summary_log"
fi

# ---------- ci/cd ----------
echo "[INFO] Setting up environment..." | tee -a "$summary_log"
env >> "$test_log" 2>&1
# env.sh invokes rm before it resets LD_LIBRARY_PATH, so dlPTI's injector must be disabled temporarily.
if [ -n "$LD_PRELOAD" ]; then
    saved_ld_preload="$LD_PRELOAD"
    unset LD_PRELOAD
    source "${SDK_DIR}/env.sh" >> "$test_log" 2>&1
    export LD_PRELOAD="$saved_ld_preload"
    unset saved_ld_preload
else
    source "${SDK_DIR}/env.sh" >> "$test_log" 2>&1
fi
env >> "$test_log" 2>&1

# Configure ccache
echo "[INFO] Configuring ccache..." | tee -a "$summary_log"
ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
{
    ccache --set-config cache_dir="${ccache_dir}"
    ccache --set-config max_size=20G
    ccache --zero-stats
} >> "$test_log" 2>&1
# ---------- ci/cd ----------

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
        echo "[INFO] CI Mode: Current LD_LIBRARY_PATH: $LD_LIBRARY_PATH" | tee -a "$summary_log"
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

export GGML_TEST_MODE=1

# Set environment variables
export GGML_DEBUG=1

# Limit CUDA devices to maximum 1 to avoid long test times
export CUDA_VISIBLE_DEVICES=0
echo "[INFO] CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES" | tee -a "$summary_log"

# Simple test mode
if [ "$SIMPLE_TEST" = true ]; then
    echo "[INFO] =============================" | tee -a "$summary_log"
    echo "[INFO] Simple test mode enabled" | tee -a "$summary_log"
    echo "[INFO] =============================" | tee -a "$summary_log"

    # ============================================================
    # Test List Configuration
    # Format: "TEST_NAME|COMMAND|ENV_VARS"
    # - TEST_NAME: Short name for the test
    # - COMMAND: The actual command to run (use ${build_dir_bin} for binary path)
    # - ENV_VARS: Environment variables (optional, use "NONE" if not needed)
    #
    # To add a new test, simply add a new line here
    # To remove a test, comment out or delete the line
    # ============================================================
    declare -a SIMPLE_TEST_CASES=(
        "MUL_MAT|${build_dir_bin}/test-backend-ops -o MUL_MAT|GGML_DLBLAS_CONSISTENT=1 GGML_DEBUG_PATH_SELECTION=1 GGML_FORCE_DLBLAS_TEST=1 GGML_CUDA_GPTQ_GROUP_SIZE=128"
        # "FLASH_ATTN_EXT|${build_dir_bin}/test-backend-ops -o FLASH_ATTN_EXT|GGML_DLFA_READY=1"
        # "MUL_MAT_ID|${build_dir_bin}/test-backend-ops -o MUL_MAT -p \"(type_a=q4_1,type_b=f32,n_mats=4,n_used=1,b=1,m=512,n=1,k=256)\"|NONE"
        # "ADD|${build_dir_bin}/test-backend-ops -o ADD|NONE"  # Example: Add more tests here
        # "MUL|${build_dir_bin}/test-backend-ops -o MUL|NONE"  # Example
    )

    # Print test list
    echo "[INFO] Test list (${#SIMPLE_TEST_CASES[@]} tests):" | tee -a "$summary_log"
    for i in "${!SIMPLE_TEST_CASES[@]}"; do
        IFS='|' read -r test_name test_cmd test_env <<< "${SIMPLE_TEST_CASES[$i]}"
        echo "[INFO]   $((i+1)). $test_name" | tee -a "$summary_log"
    done
    echo "" | tee -a "$summary_log"

    # Initialize counters
    fail_count=0
    pass_count=0
    skip_count=0
    fail_list=()
    skip_list=()
    total_tests=${#SIMPLE_TEST_CASES[@]}

    # Execute tests from list
    for i in "${!SIMPLE_TEST_CASES[@]}"; do
        # Parse test configuration
        IFS='|' read -r test_name test_cmd test_env <<< "${SIMPLE_TEST_CASES[$i]}"
        test_num=$((i+1))

        echo "" | tee -a "$summary_log"
        echo "[INFO] ========================================" | tee -a "$summary_log"
        echo "[INFO] Test $test_num/$total_tests: $test_name" | tee -a "$summary_log"

        # Display command with environment
        if [ "$test_env" != "NONE" ]; then
            echo "[INFO] Running: $test_env $test_cmd" | tee -a "$summary_log"
        else
            echo "[INFO] Running: $test_cmd" | tee -a "$summary_log"
        fi
        echo "[INFO] ========================================" | tee -a "$summary_log"
        echo "-----------------------------" | tee -a "$test_log"

        start_time=$(date +%s)
        set +e

        # Create temporary file to capture this specific test's output
        temp_output="/tmp/test_output_${test_num}_$$.log"

        # Execute with or without environment variables
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

        # Check results
        # Count sub-test failures, passes, and skips in the output (for tests like FLASH_ATTN_EXT that have many sub-tests)
        sub_test_fail_count=0
        sub_test_pass_count=0
        sub_test_skip_count=0
        sub_test_total_count=0

        # Check if test output contains sub-test results
        if [ -f "$temp_output" ]; then
            # Count passes: lines with "test passed [test]" or "test passed [grad]" (summary lines only)
            # Note: Using summary lines instead of real-time "OK" output to avoid double counting
            sub_test_pass_count=$(grep -c "test passed \[" "$temp_output" 2>/dev/null || echo "0")
            sub_test_pass_count=$(echo "$sub_test_pass_count" | tr -d '\n\r' | xargs)
            # Fallback: if no "test passed" lines, count ANSI color-coded OK (32mOK from console output)
            if [ "$sub_test_pass_count" -eq 0 ]; then
                sub_test_pass_count=$(grep -c "32mOK" "$temp_output" 2>/dev/null || echo "0")
                sub_test_pass_count=$(echo "$sub_test_pass_count" | tr -d '\n\r' | xargs)
            fi

            # Count failures: only use "test failed" summary lines (not real-time FAIL output)
            # This prevents double counting since FAIL appears in both real-time and summary output
            sub_test_fail_count=$(grep -c "test failed \[" "$temp_output" 2>/dev/null || echo "0")
            sub_test_fail_count=$(echo "$sub_test_fail_count" | tr -d '\n\r' | xargs)
            # Fallback: if no "test failed" lines, count ANSI color-coded FAIL (31mFAIL from console output)
            # Only count FAIL lines that are not summary/status lines
            if [ "$sub_test_fail_count" -eq 0 ]; then
                # Count 31mFAIL but exclude lines containing "Backend" or lines that are at the end (summary)
                total_fail_lines=$(grep -c "31mFAIL" "$temp_output" 2>/dev/null || echo "0")
                backend_fail_lines=$(grep "31mFAIL" "$temp_output" 2>/dev/null | grep -c "Backend" || echo "0")
                # Exclude the last FAIL line if it's a summary (typically the final overall status)
                if [ "$total_fail_lines" -gt 0 ]; then
                    sub_test_fail_count=$((total_fail_lines - backend_fail_lines - 1))
                    if [ "$sub_test_fail_count" -lt 0 ]; then
                        sub_test_fail_count=0
                    fi
                else
                    sub_test_fail_count=0
                fi
                sub_test_fail_count=$(echo "$sub_test_fail_count" | tr -d '\n\r' | xargs)
            fi

            # Count "not supported" lines as skipped tests
            sub_test_skip_count=$(grep -c "not supported" "$temp_output" 2>/dev/null || echo "0")
            sub_test_skip_count=$(echo "$sub_test_skip_count" | tr -d '\n\r' | xargs)

            # Calculate total
            sub_test_total_count=$((sub_test_pass_count + sub_test_fail_count + sub_test_skip_count))

            # Display summary if there are failures, passes, or skips
            if [ "$sub_test_fail_count" -gt 0 ] || [ "$sub_test_pass_count" -gt 0 ] || [ "$sub_test_skip_count" -gt 0 ]; then
                if [ "$sub_test_total_count" -gt 0 ]; then
                    # Calculate actual total including skipped tests
                    actual_total=$((sub_test_pass_count + sub_test_fail_count + sub_test_skip_count))
                    echo "[INFO] Sub-test results: Pass=$sub_test_pass_count, Fail=$sub_test_fail_count, Skip=$sub_test_skip_count, Total=$actual_total" | tee -a "$summary_log"
                else
                    echo "[INFO] Sub-test results: Pass=?, Fail=$sub_test_fail_count, Skip=$sub_test_skip_count" | tee -a "$summary_log"
                fi
            fi

            # Clean up temp file
            safe_rm -f "$temp_output"
        fi

        # Determine test result status:
        # - FAIL: if exit code is non-zero OR there are sub-test failures
        # - SKIP: if exit code is 0, no failures, but all sub-tests were skipped
        # - PASS: if exit code is 0, no failures, and there are some passed tests
        if [ "$ret" -ne 0 ]; then
            # Exit code non-zero
            echo "[FAIL] $test_name (exit code: $ret, ${duration}s)" | tee -a "$summary_log"
            fail_count=$((fail_count+1))
            fail_list+=("$test_name (exit code $ret)")
        elif [ "$sub_test_fail_count" -gt 0 ]; then
            # Exit code is 0 but sub-tests failed
            echo "[FAIL] $test_name (${duration}s, $sub_test_fail_count sub-test failures)" | tee -a "$summary_log"
            fail_count=$((fail_count+1))
            fail_list+=("$test_name ($sub_test_fail_count sub-test failures)")
        elif [ "$sub_test_pass_count" -eq 0 ] && [ "$sub_test_skip_count" -gt 0 ]; then
            # All sub-tests were skipped (not supported)
            echo "[SKIP] $test_name (${duration}s, $sub_test_skip_count sub-tests skipped)" | tee -a "$summary_log"
            skip_count=$((skip_count+1))
            skip_list+=("$test_name (all $sub_test_skip_count sub-tests not supported)")
        else
            # Passed
            echo "[PASS] $test_name (${duration}s)" | tee -a "$summary_log"
            pass_count=$((pass_count+1))
        fi
    done

    # Summary
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Simple Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO]   Total tests:  $total_tests" | tee -a "$summary_log"
    echo "[INFO]   Passed:       $pass_count" | tee -a "$summary_log"
    echo "[INFO]   Failed:       $fail_count" | tee -a "$summary_log"
    echo "[INFO]   Skipped:      $skip_count" | tee -a "$summary_log"
    echo "[INFO]   Total time:   ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    # Display skip details if any
    if [ $skip_count -gt 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[INFO] Skipped tests (not supported):" | tee -a "$summary_log"
        for skip_item in "${skip_list[@]}"; do
            echo "  - $skip_item" | tee -a "$summary_log"
        done
    fi

    # Display fail details and exit if any failures
    if [ $fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] Simple tests failed:" | tee -a "$summary_log"
        for fail_item in "${fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    else
        echo "" | tee -a "$summary_log"
        if [ $skip_count -gt 0 ]; then
            echo "[LLAMA_CPP_PASS] All enabled tests passed! ($skip_count tests skipped)" | tee -a "$summary_log"
        else
            echo "[LLAMA_CPP_PASS] All simple tests passed!" | tee -a "$summary_log"
        fi
    fi

    exit 0
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

if [ "$SIMPLE_MODEL" = true ]; then
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

    # Initialize counters
    fail_count=0
    pass_count=0
    skip_count=0
    fail_list=()
    skip_list=()

    model_base_path="${LOCAL_MODEL_PATH}"
    model_path="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    full_model_path="${model_base_path%/}/${model_path}"
    model_name="qwen2.5-1.5b-instruct-q4_k_m"

    if [ ! -f "$full_model_path" ]; then
        echo "[ERROR] Model file not found: $full_model_path" | tee -a "$summary_log"
        exit 1
    fi
    echo "[INFO] Model file: $full_model_path" | tee -a "$summary_log"

    declare -a SIMPLE_MODEL_CASES=(
        # "math_calculation|请计算 2+3*4 的结果|14|数学计算测试"
        "geography|请用一个词回答：北京是中国的什么？|首都|地理知识测试"
    )

    total_tests=${#SIMPLE_MODEL_CASES[@]}
    echo "[INFO] Total tests to run: $total_tests" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"

    for i in "${!SIMPLE_MODEL_CASES[@]}"; do
        IFS='|' read -r test_name prompt expected_content description <<< "${SIMPLE_MODEL_CASES[$i]}"
        test_num=$((i+1))

        echo "" | tee -a "$summary_log"
        echo "[INFO] ========================================" | tee -a "$summary_log"
        echo "[INFO] test $test_num/$total_tests: $model_name - $test_name" | tee -a "$summary_log"
        echo "[INFO] description: $description" | tee -a "$summary_log"
        echo "[INFO] prompt: $prompt" | tee -a "$summary_log"
        echo "[INFO] expected output: $expected_content" | tee -a "$summary_log"
        echo "[INFO] ========================================" | tee -a "$summary_log"
        echo "-----------------------------" | tee -a "$test_log"

        start_time=$(date +%s)
        set +e

        temp_output=$(mktemp)
        if [ "$NO_FA" = "true" ]; then
            echo "[DEBUG] (without Flash Attention), -ngl ${SIMPLE_MODEL_GPU_LAYERS}" | tee -a "$test_log"
            echo "[DEBUG] Prompt content: '$prompt'" | tee -a "$test_log"
            CUDA_VISIBLE_DEVICES=0 "${build_dir_bin}/llama-cli" -m "$full_model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -ngl ${SIMPLE_MODEL_GPU_LAYERS} -p "$prompt" > "$temp_output" 2>&1
            ret=$?
        else
            echo "[DEBUG] (with Flash Attention), -ngl ${SIMPLE_MODEL_GPU_LAYERS}" | tee -a "$test_log"
            echo "[DEBUG] Prompt content: '$prompt'" | tee -a "$test_log"
            CUDA_VISIBLE_DEVICES=0 GGML_DLFA_READY=1 "${build_dir_bin}/llama-cli" -m "$full_model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -fa -ngl ${SIMPLE_MODEL_GPU_LAYERS} -p "$prompt" > "$temp_output" 2>&1
            ret=$?
        fi

        cat "$temp_output" | tee -a "$test_log"

        set -e
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        echo "-----------------------------" | tee -a "$test_log"

        if [ $ret -ne 0 ]; then
            echo "[FAIL] $model_name - $test_name (exit code: $ret, ${duration}s)" | tee -a "$summary_log"
            fail_count=$((fail_count+1))
            fail_list+=("$model_name - $test_name (exit code $ret)")
        else
            full_output=$(cat "$temp_output")
            model_output=$(extract_model_output "$full_output")

            echo "[DEBUG] Model generated text: $model_output" | tee -a "$test_log"
            if validate_output "$model_output" "$expected_content"; then
                echo "[PASS] $model_name - $test_name (${duration}s)" | tee -a "$summary_log"
                echo "[INFO] ✓ Output contains expected: $expected_content" | tee -a "$summary_log"
                pass_count=$((pass_count+1))
            else
                echo "[FAIL] $model_name - $test_name (${duration}s)" | tee -a "$summary_log"
                echo "[ERROR] Output does NOT contain expected: $expected_content" | tee -a "$summary_log"
                echo "[ERROR] Model generated: $model_output" | tee -a "$summary_log"
                fail_count=$((fail_count+1))
                fail_list+=("$model_name - $test_name (validation failed)")
            fi
        fi
        safe_rm -f "$temp_output"
    done

    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Qwen2.5 Model Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO]   Model:            $model_name" | tee -a "$summary_log"
    if [ "$NO_FA" = "true" ]; then
        echo "[INFO]   Mode:             without Flash Attention" | tee -a "$summary_log"
    else
        echo "[INFO]   Mode:             Flash Attention (GGML_DLFA_READY=1)" | tee -a "$summary_log"
    fi
    echo "[INFO]   Total tests:      $total_tests" | tee -a "$summary_log"
    echo "[INFO]   Passed:           $pass_count" | tee -a "$summary_log"
    echo "[INFO]   Failed:           $fail_count" | tee -a "$summary_log"
    echo "[INFO]   Total time:       ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    if [ $fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] Tests failed:" | tee -a "$summary_log"
        for fail_item in "${fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    else
        echo "" | tee -a "$summary_log"
        if [ "$NO_FA" = "true" ]; then
            echo "[LLAMA_CPP_PASS] Standard inference tests passed!" | tee -a "$summary_log"
        else
            echo "[LLAMA_CPP_PASS] Flash Attention tests passed!" | tee -a "$summary_log"
        fi
    fi

    exit 0
fi

# Performance test mode - Model stress testing comparing Pool vs Legacy modes
if [ "$SIMPLE_PERF" = true ]; then
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Performance Test Mode - Pool vs Legacy Mode Stress Testing" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Target Models: Qwen2.5-1.5B (fp16)" | tee -a "$summary_log"
    echo "[INFO] Test Type: Stress testing with multiple iterations" | tee -a "$summary_log"
    echo "[INFO] Comparison: 🔄 Pool Mode vs 🔧 Legacy Mode performance" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"

    # Initialize counters
    fail_count=0
    pass_count=0
    fail_list=()

    # Define test model
    model_base_path="${LOCAL_MODEL_PATH}"
    performance_model="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
    model_path="${model_base_path%/}/${performance_model}"

    # Check if model exists
    if [ ! -f "$model_path" ]; then
        echo "[ERROR] Performance test model not found: $model_path" | tee -a "$summary_log"
        echo "[INFO] Please ensure the Qwen2.5-1.5B model is available for performance testing" | tee -a "$summary_log"
        exit 1
    fi

    echo "[INFO] Performance test model: $performance_model" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"

    # Performance test configuration
    PERF_ITERATIONS=5
    PERF_TOKENS=100
    PERF_PROMPT="Write a detailed explanation of artificial intelligence and machine learning, including their applications in modern technology and their potential impact on society."

    # Test both modes
    declare -A mode_times
    declare -A mode_results

    for mode in "POOL" "LEGACY"; do
        if [ "$mode" = "POOL" ]; then
            mode_env="GGML_GPTQ_USE_POOL=1"
            mode_icon="🔄"
            mode_desc="Pool Mode"
        else
            mode_env="GGML_GPTQ_USE_POOL=0"
            mode_icon="🔧"
            mode_desc="Legacy Mode"
        fi

        echo "[INFO] ========================================" | tee -a "$summary_log"
        echo "[INFO] Performance Testing: $mode_icon $mode_desc" | tee -a "$summary_log"
        echo "[INFO] Iterations: $PERF_ITERATIONS, Tokens: $PERF_TOKENS" | tee -a "$summary_log"
        echo "[INFO] ========================================" | tee -a "$summary_log"

        mode_total_time=0
        mode_success_count=0
        mode_fail_count=0

        for i in $(seq 1 $PERF_ITERATIONS); do
            echo "[INFO] $mode_icon $mode_desc - Iteration $i/$PERF_ITERATIONS" | tee -a "$summary_log"

            start_time=$(date +%s.%N)
            set +e

            # Run performance test
            temp_output=$(mktemp)
            bash -c "$mode_env ${build_dir_bin}/llama-cli -m '$model_path' -no-cnv -n $PERF_TOKENS --temp 0.7 --top-k 40 --top-p 0.9 -s $i -p '$PERF_PROMPT'" > "$temp_output" 2>&1
            ret=$?

            end_time=$(date +%s.%N)
            duration=$(echo "$end_time - $start_time" | bc -l)

            # Show output in verbose mode
            if [ "$VERBOSE" = "true" ]; then
                echo "[INFO] Performance test output (VERBOSE):" | tee -a "$test_log"
                cat "$temp_output" | tee -a "$test_log"
                echo "-----------------------------" | tee -a "$test_log"
            fi

            set -e

            if [ $ret -eq 0 ]; then
                mode_success_count=$((mode_success_count + 1))
                mode_total_time=$(echo "$mode_total_time + $duration" | bc -l)
                echo "[PASS] $mode_icon Iteration $i: ${duration}s" | tee -a "$summary_log"
            else
                mode_fail_count=$((mode_fail_count + 1))
                echo "[FAIL] $mode_icon Iteration $i: Exit code $ret" | tee -a "$summary_log"
                fail_count=$((fail_count + 1))
                fail_list+=("$mode_desc - Iteration $i (exit code $ret)")
            fi

            safe_rm -f "$temp_output"
        done

        # Calculate statistics for this mode
        if [ $mode_success_count -gt 0 ]; then
            mode_avg_time=$(echo "scale=3; $mode_total_time / $mode_success_count" | bc -l)
            mode_times[$mode]=$mode_avg_time
            mode_results[$mode]="$mode_success_count/$PERF_ITERATIONS successful"
            echo "[INFO] $mode_icon $mode_desc Results: $mode_success_count/$PERF_ITERATIONS successful, Avg time: ${mode_avg_time}s" | tee -a "$summary_log"
        else
            mode_times[$mode]="N/A"
            mode_results[$mode]="0/$PERF_ITERATIONS successful"
            echo "[FAIL] $mode_icon $mode_desc Results: All iterations failed" | tee -a "$summary_log"
        fi

        echo "" | tee -a "$summary_log"
    done

    # Performance comparison summary
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Performance Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Model: $performance_model" | tee -a "$summary_log"
    echo "[INFO] Test Configuration: $PERF_ITERATIONS iterations, $PERF_TOKENS tokens each" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "[INFO] 🔄 Pool Mode Results: ${mode_results[POOL]}" | tee -a "$summary_log"
    echo "[INFO] 🔄 Pool Mode Avg Time: ${mode_times[POOL]}s" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "[INFO] 🔧 Legacy Mode Results: ${mode_results[LEGACY]}" | tee -a "$summary_log"
    echo "[INFO] 🔧 Legacy Mode Avg Time: ${mode_times[LEGACY]}s" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"

    # Performance comparison
    if [ "${mode_times[POOL]}" != "N/A" ] && [ "${mode_times[LEGACY]}" != "N/A" ]; then
        pool_time=${mode_times[POOL]}
        legacy_time=${mode_times[LEGACY]}

        # Calculate performance difference (using bc for floating point)
        if (( $(echo "$pool_time < $legacy_time" | bc -l) )); then
            improvement=$(echo "scale=2; ($legacy_time - $pool_time) / $legacy_time * 100" | bc -l)
            echo "[INFO] 🚀 Performance: Pool Mode is ${improvement}% faster than Legacy Mode" | tee -a "$summary_log"
        elif (( $(echo "$pool_time > $legacy_time" | bc -l) )); then
            degradation=$(echo "scale=2; ($pool_time - $legacy_time) / $legacy_time * 100" | bc -l)
            echo "[INFO] ⚠️  Performance: Pool Mode is ${degradation}% slower than Legacy Mode" | tee -a "$summary_log"
        else
            echo "[INFO] ⚖️  Performance: Pool Mode and Legacy Mode have similar performance" | tee -a "$summary_log"
        fi
    fi

    echo "[INFO] Total test time: ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    # Final result
    if [ $fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] Performance tests failed:" | tee -a "$summary_log"
        for fail_item in "${fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    else
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_PASS] All performance tests completed successfully!" | tee -a "$summary_log"
        echo "[INFO] Performance comparison between Pool and Legacy modes completed" | tee -a "$summary_log"
    fi

    exit 0
fi

# Simple TP test mode - Tensor Parallel testing with split-mode variations
if [ "$SIMPLE_TP" = true ]; then
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    echo "[INFO] Simple TP Test Mode - Tensor Parallel Split Mode Testing" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"
    # Define test model based on --big option
    model_base_path="${LOCAL_MODEL_PATH}"
    if [ "$BIG_MODEL" = true ]; then
        tp_model="Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
        model_name="Qwen3-30B (Q4_K_M)"
        echo "[INFO] Target Model: $model_name (BIG MODEL)" | tee -a "$summary_log"
    else
        tp_model="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
        model_name="Qwen2.5-1.5B (fp16)"
        echo "[INFO] Target Model: $model_name (SMALL MODEL)" | tee -a "$summary_log"
    fi

    echo "[INFO] Split Modes: row, layer, none" | tee -a "$summary_log"
    echo "[INFO] GPTQ Modes: 🔄 Pool Mode vs 🔧 Legacy Mode" | tee -a "$summary_log"
    echo "[INFO] ============================================================" | tee -a "$summary_log"

    # Initialize counters
    fail_count=0
    pass_count=0
    fail_list=()

    model_path="${model_base_path%/}/${tp_model}"

    # Check if model exists
    if [ ! -f "$model_path" ]; then
        echo "[ERROR] TP test model not found: $model_path" | tee -a "$summary_log"
        if [ "$BIG_MODEL" = true ]; then
            echo "[INFO] Please ensure the Qwen3-30B model is available for TP testing" | tee -a "$summary_log"
            echo "[INFO] Expected path: ${model_base_path%/}/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf" | tee -a "$summary_log"
        else
            echo "[INFO] Please ensure the Qwen2.5-1.5B model is available for TP testing" | tee -a "$summary_log"
            echo "[INFO] Expected path: ${model_base_path%/}/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf" | tee -a "$summary_log"
        fi
        exit 1
    fi

    echo "[INFO] TP test model: $tp_model" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"

    # TP test configuration - adjust parameters based on model size
    if [ "$BIG_MODEL" = true ]; then
        TP_TOKENS=10  # Use fewer tokens for large model to avoid memory issues
        echo "[INFO] Using reduced token count ($TP_TOKENS) for large model" | tee -a "$summary_log"
    else
        TP_TOKENS=30  # Standard token count for small model
    fi
    TP_PROMPT="What is 2+3? Answer with only the number."
    TP_EXPECTED_CONTENT="5"  # Expected answer for correctness validation

    # Define split modes and GPTQ modes
    split_modes=("row" "layer" "none")
    gptq_modes=("POOL" "LEGACY")

    total_tp_tests=$((${#split_modes[@]} * ${#gptq_modes[@]}))
    current_test=0

    # Test each combination
    for split_mode in "${split_modes[@]}"; do
        for gptq_mode in "${gptq_modes[@]}"; do
            current_test=$((current_test + 1))

            if [ "$gptq_mode" = "POOL" ]; then
                mode_env="GGML_GPTQ_USE_POOL=1"
                mode_icon="🔄"
                mode_desc="Pool Mode"
            else
                mode_env="GGML_GPTQ_USE_POOL=0"
                mode_icon="🔧"
                mode_desc="Legacy Mode"
            fi

            test_name="TP-${split_mode^^}-${gptq_mode}"

            echo "" | tee -a "$summary_log"
            echo "[INFO] ========================================" | tee -a "$summary_log"
            echo "[INFO] Test $current_test/$total_tp_tests: $test_name ($mode_icon $mode_desc)" | tee -a "$summary_log"
            echo "[INFO] Split Mode: $split_mode" | tee -a "$summary_log"
            echo "[INFO] Model: $model_path" | tee -a "$summary_log"
            echo "[INFO] Environment: $mode_env" | tee -a "$summary_log"
            echo "[INFO] ========================================" | tee -a "$summary_log"
            echo "-----------------------------" | tee -a "$test_log"

            start_time=$(date +%s)
            set +e

            # Run TP test
            temp_output=$(mktemp)
            echo "[DEBUG] Running TP test with split-mode=$split_mode and $mode_env" | tee -a "$test_log"
            bash -c "$mode_env ${build_dir_bin}/llama-cli -m '$model_path' --split-mode '$split_mode' -no-cnv -n $TP_TOKENS --temp 0.7 --top-k 40 --top-p 0.9 -s 42 -p '$TP_PROMPT'" > "$temp_output" 2>&1
            ret=$?

            # Display output (verbose mode shows all, otherwise first 10 lines)
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

            # Validate result - both functionality and correctness
            if [ $ret -eq 0 ]; then
                # Check if expected content is in the output for correctness validation
                output_content=$(cat "$temp_output")
                if echo "$output_content" | grep -qi "$TP_EXPECTED_CONTENT"; then
                    echo "[PASS] $test_name ($mode_icon $mode_desc) (${duration}s) ✓ Correctness verified" | tee -a "$summary_log"
                    echo "[INFO] Split mode '$split_mode' with $mode_desc: Functionality ✓ Correctness ✓" | tee -a "$summary_log"
                    echo "[INFO] Expected content '$TP_EXPECTED_CONTENT' found in output" | tee -a "$summary_log"
                    pass_count=$((pass_count+1))
                else
                    echo "[FAIL] $test_name ($mode_icon $mode_desc) (${duration}s) ❌ Correctness failed" | tee -a "$summary_log"
                    echo "[WARN] Split mode '$split_mode' with $mode_desc: Functionality ✓ Correctness ❌" | tee -a "$summary_log"
                    echo "[DEBUG] Expected content '$TP_EXPECTED_CONTENT' not found in output" | tee -a "$summary_log"
                    echo "[DEBUG] Actual output excerpt:" | tee -a "$test_log"
                    echo "$output_content" | tail -n 10 | tee -a "$test_log"
                    fail_count=$((fail_count+1))
                    fail_list+=("$test_name ($mode_desc, correctness validation failed)")
                fi
            else
                echo "[FAIL] $test_name ($mode_icon $mode_desc) (exit code: $ret, ${duration}s) ❌ Functionality failed" | tee -a "$summary_log"
                echo "[WARN] Split mode '$split_mode' with $mode_desc: Functionality ❌" | tee -a "$summary_log"
                fail_count=$((fail_count+1))
                fail_list+=("$test_name ($mode_desc, exit code $ret)")

                # Show error details
                echo "[DEBUG] Error output:" | tee -a "$test_log"
                tail -n 20 "$temp_output" | tee -a "$test_log"
            fi

            # Clean up temp file
            safe_rm -f "$temp_output"
        done
    done

    # Summary
    test_end_time=$(date +%s)
    test_duration=$((test_end_time - test_start_time))

    echo "" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Simple TP Test Summary" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"
    echo "[INFO] Model: $tp_model" | tee -a "$summary_log"
    echo "[INFO] Split modes tested: ${split_modes[*]}" | tee -a "$summary_log"
    echo "[INFO] GPTQ modes tested: ${gptq_modes[*]}" | tee -a "$summary_log"
    echo "[INFO] Total tests: $total_tp_tests" | tee -a "$summary_log"
    echo "[INFO] Passed: $pass_count" | tee -a "$summary_log"
    echo "[INFO] Failed: $fail_count" | tee -a "$summary_log"
    echo "[INFO] Total time: ${test_duration}s" | tee -a "$summary_log"
    echo "[INFO] =============================================" | tee -a "$summary_log"

    # Performance comparison by split mode
    echo "" | tee -a "$summary_log"
    echo "[INFO] Split Mode Performance Analysis:" | tee -a "$summary_log"
    for split_mode in "${split_modes[@]}"; do
        pool_test="TP-${split_mode^^}-POOL"
        legacy_test="TP-${split_mode^^}-LEGACY"

        # Check if tests passed (default to true, set false if found in fail_list)
        pool_passed=true
        legacy_passed=true

        # Check if pool test failed
        for i in "${!fail_list[@]}"; do
            if [[ "${fail_list[$i]}" == *"$pool_test"* ]]; then
                pool_passed=false
                break
            fi
        done

        # Check if legacy test failed
        for i in "${!fail_list[@]}"; do
            if [[ "${fail_list[$i]}" == *"$legacy_test"* ]]; then
                legacy_passed=false
                break
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

    # Final result
    if [ $fail_count -ne 0 ]; then
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_FAIL] TP tests failed:" | tee -a "$summary_log"
        for fail_item in "${fail_list[@]}"; do
            echo "  - $fail_item" | tee -a "$summary_log"
        done
        exit 1
    else
        echo "" | tee -a "$summary_log"
        echo "[LLAMA_CPP_PASS] All TP tests passed successfully!" | tee -a "$summary_log"
        echo "[INFO] Tensor Parallel implementation verified across all split modes" | tee -a "$summary_log"
    fi

    exit 0
fi

# Full test set for all platforms
echo "[INFO] Running full test suite for all platforms" | tee -a "$summary_log"
echo "[INFO] Note: All platforms now use the complete test set for comprehensive coverage" | tee -a "$summary_log"

test_cases_part1=(
    "${build_dir_bin}/test-arg-parser"
    "${build_dir_bin}/test-autorelease"
    "${build_dir_bin}/test-backend-ops"
    "${build_dir_bin}/test-c"
    "${build_dir_bin}/test-chat"
    "${build_dir_bin}/test-chat-parser"
    "${build_dir_bin}/test-chat-template"
    "${build_dir_bin}/test-gbnf-validator grammars/json.gbnf -c '{\"name\": \"Alice\", \"age\": 25}'"        # Test valid JSON
    "${build_dir_bin}/test-gbnf-validator grammars/json.gbnf -c '{\"name\": \"Bob\", \"age\": thirty}'"      # Test invalid JSON (should fail)
    "${build_dir_bin}/test-gbnf-validator grammars/arithmetic.gbnf -c 'x = 5'"                               # Test valid arithmetic
    "${build_dir_bin}/test-gbnf-validator grammars/arithmetic.gbnf -c 'x = 5 +'"                             # Test invalid arithmetic (should fail)
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

# Add platform-specific tests
if [ "${ARCH}" != "loongarch64" ]; then
    echo "[INFO] Adding test-json-schema-to-grammar (not LoongArch64 platform)" | tee -a "$summary_log"
    test_cases_part1+=("${build_dir_bin}/test-json-schema-to-grammar")
else
    echo "[INFO] Skipping test-json-schema-to-grammar on LoongArch64 platform (because the ggml-ci node lacks Python 3.8)" | tee -a "$summary_log"
fi

# Function to check if model file exists and is readable
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

# Qwen2 and Qwen3 model correctness tests with output validation
echo "[INFO] Preparing Qwen2 and Qwen3 model tests for correctness validation" | tee -a "$summary_log"

# Define model test cases for Qwen2 and Qwen3
qwen_model_tests=()

# YAML configuration file for test references
yaml_config="${REPO_PATH}/.dlci/qwen_references.yml"

# Note: No longer using reference files - validation is done by keyword matching only

# Function to parse YAML and extract test cases for a model
parse_yaml_tests() {
    local model_name="$1"
    local yaml_file="$2"

    # Simple YAML parsing for our specific structure
    # Extract tests for the given model
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

# Function to validate content (no longer generates reference files)
validate_content_only() {
    local model_name="$1"
    local test_name="$2"
    local output_content="$3"
    local expected_content="$4"

    echo "[INFO] Validating content for $model_name - $test_name"

    # Use enhanced validation
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

# Enhanced content validation function with better pattern matching
validate_content_enhanced() {
    local model_name="$1"
    local test_name="$2"
    local output_content="$3"
    local expected_content="$4"

    echo "[DEBUG] Enhanced validation started" | tee -a "$test_log"
    echo "[DEBUG] Expected: '$expected_content'" | tee -a "$test_log"

    # Simple approach: just check if the expected content appears in the output
    # Use a basic grep without complex regex to avoid hanging
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

# Removed: validate_output function - now using validate_content_only

# Function to add Qwen model tests
add_qwen_model_tests() {
    local model_base_path="${LOCAL_MODEL_PATH}"

    # Qwen2 models (based on your available models)
    local qwen2_models=(
        "Qwen2-1.5B-Moe-GGUF/Qwen2-1.5Moe.Q4_K_M.gguf"
    )

    # Qwen2.5 models (based on your available models)
    local qwen25_models=(
        "Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
        "Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )

    # Qwen3 models (based on your available models)
    local qwen3_models=(
        "Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
    )

    # Add correctness tests for available Qwen2 models
    for model_path in "${qwen2_models[@]}"; do
        local full_model_path="${model_base_path%/}/${model_path}"
        local model_name=$(basename "$model_path" .gguf)

        if check_model_file "$full_model_path" "Qwen2-$model_name"; then
            echo "[INFO] Adding Qwen2 MoE correctness tests for: $model_name"

            # Add correctness validation tests from YAML (no reference files needed)
            if [ -f "$yaml_config" ]; then
                echo "[INFO] Loading test cases from YAML for $model_name" | tee -a "$summary_log"
                # Use readarray to properly handle the output
                readarray -t test_cases < <(parse_yaml_tests "$model_name" "$yaml_config")
                for test_case in "${test_cases[@]}"; do
                    IFS='|' read -r prompt expected_content test_name <<< "$test_case"
                    # Store test parameters for later execution
                    qwen_model_tests+=("$model_name|$test_name|$prompt|$expected_content|$full_model_path")
                done
            else
                echo "[WARN] YAML config not found, skipping tests for $model_name" | tee -a "$summary_log"
            fi
        fi
    done

    # Add correctness tests for available Qwen2.5 models
    for model_path in "${qwen25_models[@]}"; do
        local full_model_path="${model_base_path%/}/${model_path}"
        local model_name=$(basename "$model_path" .gguf)

        if check_model_file "$full_model_path" "Qwen2.5-$model_name"; then
            echo "[INFO] Adding Qwen2.5 correctness tests for: $model_name"

            # Add correctness validation tests from YAML (no reference files needed)
            if [ -f "$yaml_config" ]; then
                echo "[INFO] Loading test cases from YAML for $model_name" | tee -a "$summary_log"
                # Use readarray to properly handle the output
                readarray -t test_cases < <(parse_yaml_tests "$model_name" "$yaml_config")
                for test_case in "${test_cases[@]}"; do
                    IFS='|' read -r prompt expected_content test_name <<< "$test_case"
                    # Store test parameters for later execution
                    qwen_model_tests+=("$model_name|$test_name|$prompt|$expected_content|$full_model_path")
                done
            else
                echo "[WARN] YAML config not found, skipping tests for $model_name" | tee -a "$summary_log"
            fi

            # Only test the first available Qwen2.5 model to avoid redundancy
            break
        fi
    done

    # Add correctness tests for available Qwen3 models
    for model_path in "${qwen3_models[@]}"; do
        local full_model_path="${model_base_path%/}/${model_path}"
        local model_name=$(basename "$model_path" .gguf)

        if check_model_file "$full_model_path" "Qwen3-$model_name"; then
            echo "[INFO] Adding Qwen3-30B correctness tests for: $model_name (Large model - reduced test scope)"

            # Add correctness validation tests from YAML (no reference files needed)
            if [ -f "$yaml_config" ]; then
                echo "[INFO] Loading test cases from YAML for $model_name" | tee -a "$summary_log"
                # Use readarray to properly handle the output
                readarray -t test_cases < <(parse_yaml_tests "$model_name" "$yaml_config")
                for test_case in "${test_cases[@]}"; do
                    IFS='|' read -r prompt expected_content test_name <<< "$test_case"
                    # Store test parameters for later execution
                    qwen_model_tests+=("$model_name|$test_name|$prompt|$expected_content|$full_model_path")
                done
            else
                echo "[WARN] YAML config not found, skipping tests for $model_name" | tee -a "$summary_log"
            fi
        fi
    done

    # No bench tests - focusing on correctness validation only

    if [ ${#qwen_model_tests[@]} -eq 0 ]; then
        echo "[WARN] No Qwen2, Qwen2.5 or Qwen3 models found for testing"
        echo "[INFO] Expected model paths under ${model_base_path}:"
        echo "[INFO] Qwen2 models:"
        for model_path in "${qwen2_models[@]}"; do
            echo "  - ${model_base_path%/}/${model_path}"
        done
        echo "[INFO] Qwen2.5 models:"
        for model_path in "${qwen25_models[@]}"; do
            echo "  - ${model_base_path%/}/${model_path}"
        done
        echo "[INFO] Qwen3 models:"
        for model_path in "${qwen3_models[@]}"; do
            echo "  - ${model_base_path%/}/${model_path}"
        done
    else
        echo "[INFO] Added ${#qwen_model_tests[@]} Qwen model test cases"
        echo "[INFO] Testing: Qwen2 MoE (1.5B), Qwen2.5 (1.5B), Qwen3 (30B)"
    fi
}

# Function to run validation test and return exit code
validate_qwen_output() {
    local model_name="$1"
    local test_name="$2"
    local prompt="$3"
    local expected_content="$4"
    local model_path="$5"

    echo "[INFO] Running correctness test: $model_name - $test_name"

    # Use temporary file for output
    local output_file=$(mktemp)

    # Set ngl parameter based on model type
    local ngl_value=999
    if [[ "$model_name" == *"Qwen3"* ]]; then
        ngl_value=20
        echo "[INFO] Using ngl=20 for Qwen3 model: $model_name"
    fi

    # Run inference with fixed parameters for deterministic output
    "${build_dir_bin}/llama-cli" -m "$model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -ngl $ngl_value -p "$prompt" > "$output_file" 2>&1
    local cmd_result=$?

    if [ $cmd_result -ne 0 ]; then
        echo "[FAIL] llama-cli command failed with exit code $cmd_result"
        echo "[DEBUG] Command output:" >> "$test_log"
        cat "$output_file" >> "$test_log"
        safe_rm -f "$output_file"
        return 1
    fi

    # Validate output content using enhanced validation
    local output_content
    output_content=$(cat "$output_file")
    echo "[DEBUG] Model response extract:" | tee -a "$test_log"
    # Extract the actual response (skip system info and performance stats)
    local response_part
    response_part=$(echo "$output_content" | sed -n '/^$/,/llama_perf_sampler_print:/p' | head -n -1 | tail -n +2)
    echo "$response_part" | tee -a "$test_log"

    echo "[DEBUG] Starting content validation..." | tee -a "$test_log"
    local validation_result
    validate_content_only "$model_name" "$test_name" "$output_content" "$expected_content"
    validation_result=$?
    echo "[DEBUG] Validation completed with result: $validation_result" | tee -a "$test_log"

    if [ $validation_result -eq 0 ]; then
        echo "[PASS] Correctness test passed: $model_name - $test_name"
        # Clean up temporary output file
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

# Execute function to add Qwen model tests
add_qwen_model_tests

# Full tokenizer tests for all platforms
test_cases_part2=(
    "${build_dir_bin}/test-tokenizer-1-bpe models/ggml-vocab-llama-bpe.gguf"
    "${build_dir_bin}/test-tokenizer-1-spm models/ggml-vocab-llama-spm.gguf"
)

# Function to auto-discover vocab test cases
add_tokenizer_vocab_tests() {
    echo "[INFO] Adding vocab tests for all platforms"

    for vocab in models/*.gguf; do
        inp="${vocab}.inp"
        out="${vocab}.out"
        if [[ -f "$inp" && -f "$out" ]]; then
            test_cases_part2+=("${build_dir_bin}/test-tokenizer-0 $vocab")
        fi
    done
}

add_tokenizer_vocab_tests

# Combine all test cases including Qwen model tests
test_cases=(
    "${test_cases_part1[@]}"
    "${test_cases_part2[@]}"
    "${qwen_model_tests[@]}"
)

fail_count=0
fail_list=()
# Arrays to store detailed test results for final summary
test_results_names=()
test_results_status=()

for test_case in "${test_cases[@]}"; do
    # Initialize test name variable
    test_name_for_summary=""

    # Check if this is a Qwen test (contains pipe separators)
    if [[ "$test_case" == *"|"* ]]; then
        # Parse Qwen test parameters
        IFS='|' read -r model_name test_name prompt expected_content model_path <<< "$test_case"
        test_name_for_summary="qwen_test.${model_name}.${test_name}"
        echo "Running Qwen test: $model_name - $test_name" | tee -a "$summary_log"
        echo "Running Qwen test: $model_name - $test_name" >> "$test_log"
        start_time=$(date +%s)

        # Run Qwen validation test
        validate_qwen_output "$model_name" "$test_name" "$prompt" "$expected_content" "$model_path"
        ret=$?
        end_time=$(date +%s)
        duration=$((end_time - start_time))
    else
        # Regular test case
        # Get the executable name (first word)
        test_bin=$(echo "$test_case" | awk '{print $1}')
        # Extract test name from binary path
        test_name_for_summary=$(basename "$test_bin" 2>/dev/null || echo "$test_case")

        # Check if there are arguments (more than one word in test_case)
        word_count=$(echo "$test_case" | wc -w)
        if [ "$word_count" -gt 1 ]; then
            # Extract arguments (everything after the first word)
            test_args=$(echo "$test_case" | awk '{$1=""; print $0}' | sed 's/^ //')
            # Simplify arguments for summary (remove long paths)
            test_args_short=$(echo "$test_args" | sed 's|/LocalRun/[^/]*/[^/]*/LLM/model/|...|g' | sed 's|models/|...|g' | cut -c1-80)
            test_name_for_summary="${test_name_for_summary} ${test_args_short}"
        fi

        if [ ! -x "$test_bin" ]; then
            echo "[LLAMA_CPP_FAIL] $test_bin does not exist or is not executable, skipping" | tee -a "$summary_log"
            fail_count=$((fail_count+1))
            fail_list+=("$test_case (not found or not executable)")
            # Record result for summary
            test_results_names+=("$test_name_for_summary")
            test_results_status+=("FAILED (not executable)")
            continue
        fi
        echo "Running: $test_case" | tee -a "$summary_log"
        echo "Running: $test_case" >> "$test_log"
        echo "-----------------------------" | tee -a "$test_log"
        start_time=$(date +%s)

        # Create temporary file to capture output
        temp_output=$(mktemp)

        # Check if this is test-backend-ops and add GGML_CUDA_DISABLE_GRAPHS=1
        if [[ "$test_case" == *"test-backend-ops"* ]]; then
            test_case_with_env="GGML_CUDA_DISABLE_GRAPHS=1 $test_case"
        else
            test_case_with_env="$test_case"
        fi

        # Normal execution - output to both screen and temp file
        set +e  # Disable exit on error temporarily
        eval "$test_case_with_env" > "$temp_output" 2>&1
        ret=$?
        set -e  # Re-enable exit on error

        # Display output to screen and log
        cat "$temp_output" | tee -a "$test_log"
        echo "-----------------------------" | tee -a "$test_log"

        # Parse sub-test results if this is test-backend-ops or similar tests
        if [[ "$test_name_for_summary" == "test-backend-ops" ]] || [[ "$test_name_for_summary" =~ test-backend-ops ]]; then
            # Remove ANSI codes once for the entire file (much faster than per-line)
            temp_clean=$(mktemp)
            sed $'s/\033\[[0-9;]*m//g' "$temp_output" > "$temp_clean"

            # Define regex pattern as variable (required for bash regex with special chars)
            regex_pattern='^[[:space:]]*([A-Z_]+)\(([^)]+)\):[[:space:]]*(.+)$'

            # Parse using bash built-in regex (no external process calls per line)
            while IFS= read -r line; do
                # Match patterns like: "  CPY(...): OK" or "  CPY(...): not supported [CUDA1]"
                if [[ "$line" =~ $regex_pattern ]]; then
                    sub_test_name="${BASH_REMATCH[1]}"
                    sub_test_params="${BASH_REMATCH[2]}"
                    sub_test_result="${BASH_REMATCH[3]}"

                    # Determine result status using bash pattern matching
                    if [[ "$sub_test_result" =~ ^OK ]]; then
                        sub_status="PASSED"
                    elif [[ "$sub_test_result" == *"not supported"* ]]; then
                        sub_status="skipped"
                    else
                        sub_status="FAILED"
                    fi

                    # Record sub-test result
                    sub_case_name="${test_name_for_summary}-${sub_test_name}(${sub_test_params})"
                    test_results_names+=("$sub_case_name")
                    test_results_status+=("$sub_status")
                fi
            done < "$temp_clean"
            safe_rm -f "$temp_clean"
        fi

        # Clean up temp file
        safe_rm -f "$temp_output"

        end_time=$(date +%s)
        duration=$((end_time - start_time))
    fi

    # Output formatted test results summary for this test
    echo "" | tee -a "$summary_log"
    echo "================================================================================" | tee -a "$summary_log"
    echo "FORMATTED TEST RESULTS SUMMARY:" | tee -a "$summary_log"
    echo "================================================================================" | tee -a "$summary_log"

    # Check if this test had sub-tests parsed
    had_sub_tests=false
    if [[ "$test_name_for_summary" == "test-backend-ops" ]] || [[ "$test_name_for_summary" =~ test-backend-ops ]]; then
        # Check if we have sub-test results in the arrays (look for entries added in this iteration)
        for i in "${!test_results_names[@]}"; do
            if [[ "${test_results_names[$i]}" =~ ^${test_name_for_summary}- ]]; then
                had_sub_tests=true
                break
            fi
        done
    fi

    if [ "$had_sub_tests" = true ]; then
        # Output all sub-test results
        for i in "${!test_results_names[@]}"; do
            if [[ "${test_results_names[$i]}" =~ ^${test_name_for_summary}- ]]; then
                case_name="${test_results_names[$i]}"
                case_status="${test_results_status[$i]}"
                case_result_lower=$(echo "$case_status" | tr '[:upper:]' '[:lower:]')
                echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result_lower}" | tee -a "$summary_log"
            fi
        done

        # Update fail count if test failed
        if [ $ret -ne 0 ]; then
            fail_count=$((fail_count+1))
            fail_list+=("$test_case (exit code $ret)")
        fi
    else
        # No sub-tests, output single result
        if [ $ret -eq 0 ]; then
            echo "CASE_NAME: ${test_name_for_summary}, CASE_RESULT: passed" | tee -a "$summary_log"
            test_results_names+=("$test_name_for_summary")
            test_results_status+=("PASSED")
        else
            echo "CASE_NAME: ${test_name_for_summary}, CASE_RESULT: failed" | tee -a "$summary_log"
            fail_count=$((fail_count+1))
            fail_list+=("$test_case (exit code $ret)")
            test_results_names+=("$test_name_for_summary")
            test_results_status+=("FAILED")
        fi
    fi

    echo "================================================================================" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
done

# Record end time and calculate total test duration
test_end_time=$(date +%s)
test_duration=$((test_end_time - test_start_time))
test_hours=$((test_duration / 3600))
test_minutes=$(((test_duration % 3600) / 60))
test_seconds=$((test_duration % 60))

echo "[INFO] Test suite completed at: $(date)" | tee -a "$summary_log"
if [ $test_hours -gt 0 ]; then
    echo "[INFO] Total test time: ${test_hours}h ${test_minutes}m ${test_seconds}s (${test_duration} seconds)" | tee -a "$summary_log"
elif [ $test_minutes -gt 0 ]; then
    echo "[INFO] Total test time: ${test_minutes}m ${test_seconds}s (${test_duration} seconds)" | tee -a "$summary_log"
else
    echo "[INFO] Total test time: ${test_seconds}s" | tee -a "$summary_log"
fi

# Calculate test statistics
total_tests=${#test_cases[@]}
passed_tests=$((total_tests - fail_count))
qwen_tests_count=${#qwen_model_tests[@]}

echo "" | tee -a "$summary_log"
echo "[INFO] Test Statistics:" | tee -a "$summary_log"
echo "[INFO]   Total tests: ${total_tests}" | tee -a "$summary_log"
echo "[INFO]   Passed: ${passed_tests}" | tee -a "$summary_log"
echo "[INFO]   Failed: ${fail_count}" | tee -a "$summary_log"
echo "[INFO]   Qwen model tests: ${qwen_tests_count}" | tee -a "$summary_log"

# Generate Qwen model test summary
if [ "${qwen_tests_count}" -gt 0 ]; then
    echo "" | tee -a "$summary_log"
    echo "[INFO] Qwen Model Test Summary:" | tee -a "$summary_log"
    echo "[INFO] ==============================" | tee -a "$summary_log"

    qwen_passed=0
    qwen_failed=0

    # Count Qwen-specific test results
    for fail_item in "${fail_list[@]}"; do
        if [[ "$fail_item" == *"qwen"* ]] || [[ "$fail_item" == *"Qwen"* ]]; then
            qwen_failed=$((qwen_failed + 1))
        fi
    done

    qwen_passed=$((qwen_tests_count - qwen_failed))

    echo "[INFO]   Qwen tests passed: ${qwen_passed}/${qwen_tests_count}" | tee -a "$summary_log"
    echo "[INFO]   Qwen tests failed: ${qwen_failed}/${qwen_tests_count}" | tee -a "$summary_log"

    if [ ${qwen_failed} -eq 0 ]; then
        echo "[INFO]   ✅ All Qwen model tests passed successfully!" | tee -a "$summary_log"
        echo "[INFO]   ✅ Qwen2 MoE, Qwen2.5, and Qwen3 models show correct functionality" | tee -a "$summary_log"
    else
        echo "[WARN]   ⚠️  Some Qwen model tests failed. Please check the failed test details above." | tee -a "$summary_log"
        echo "[INFO]   Failed Qwen tests:" | tee -a "$summary_log"
        for fail_item in "${fail_list[@]}"; do
            if [[ "$fail_item" == *"qwen"* ]] || [[ "$fail_item" == *"Qwen"* ]]; then
                echo "[INFO]     - $fail_item" | tee -a "$summary_log"
            fi
        done
    fi

    echo "[INFO] ==============================" | tee -a "$summary_log"

    # Correctness validation insights
    echo "[INFO] Qwen Model Test Coverage:" | tee -a "$summary_log"
    echo "[INFO]   - Qwen2 MoE (1.5B): Mixture of Experts architecture correctness" | tee -a "$summary_log"
    echo "[INFO]   - Qwen2.5 (1.5B): Standard instruction-following accuracy" | tee -a "$summary_log"
    echo "[INFO]   - Qwen3 (30B): Large-scale model reasoning (selected tests)" | tee -a "$summary_log"
    echo "[INFO]   - Content validation: Output compared against expected answers" | tee -a "$summary_log"
    echo "[INFO]   - Deterministic testing: Fixed temperature (0.0) and seed (42)" | tee -a "$summary_log"
    echo "[INFO]   - Reference comparison: Outputs validated against baseline" | tee -a "$summary_log"
    echo "[INFO]   - Test categories: Math calculation, language understanding, reasoning" | tee -a "$summary_log"
    echo "[INFO]   - No performance benchmarking - focus on correctness only" | tee -a "$summary_log"
else
    echo "" | tee -a "$summary_log"
    echo "[WARN] No Qwen model tests were executed" | tee -a "$summary_log"
    echo "[WARN] Please ensure Qwen2, Qwen2.5, and Qwen3 model files are available in ${LOCAL_MODEL_PATH}" | tee -a "$summary_log"
    echo "[WARN] Expected paths:" | tee -a "$summary_log"
    echo "[WARN]   - Qwen2-1.5B-Moe-GGUF/Qwen2-1.5Moe.Q4_K_M.gguf" | tee -a "$summary_log"
    echo "[WARN]   - Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-*.gguf" | tee -a "$summary_log"
    echo "[WARN]   - Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf" | tee -a "$summary_log"
fi

# Final test results and log information
test_log_size=$(du -h "$test_log" | cut -f1)
summary_log_size=$(du -h "$summary_log" | cut -f1)
echo "[INFO] Log files created:" | tee -a "$summary_log"
echo "[INFO]   Detailed log: $test_log ($test_log_size)" | tee -a "$summary_log"
echo "[INFO]   Summary log: $summary_log ($summary_log_size)" | tee -a "$summary_log"

if [ $fail_count -ne 0 ]; then
    echo "=============================" | tee -a "$summary_log"
    echo "[LLAMA_CPP_FAIL] $fail_count test(s) failed:" | tee -a "$summary_log"
    for fail_item in "${fail_list[@]}"; do
        echo "  - $fail_item" | tee -a "$summary_log"
    done
    echo "[INFO] Detailed failure logs can be found in: $test_log" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "OVERALL RESULT: FAILED (${passed_tests}/${total_tests} tests passed)" | tee -a "$summary_log"
    exit 1
else
    echo "=============================" | tee -a "$summary_log"
    echo "[LLAMA_CPP_PASS] All tests passed!" | tee -a "$summary_log"
    echo "" | tee -a "$summary_log"
    echo "OVERALL RESULT: PASSED (${total_tests}/${total_tests} tests passed)" | tee -a "$summary_log"
fi
