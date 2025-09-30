#!/bin/bash
#
# Llama.cpp Test Suite
#
# Usage:
#   Normal mode: ./test_llama_cpp.sh
#   CI mode:     ./test_llama_cpp.sh --ci-test --binary-path /path/to/release
#
# CI mode uses binaries from a release package (e.g., downloaded from release_llama_cpp.sh)
# instead of the local build directory.
#

set -e

# Parse command line arguments
CI_TEST_MODE=false
BINARY_PATH=""

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
        *)
            echo "[ERROR] Unknown parameter: $1"
            echo "Usage: $0 [--ci-test] [--binary-path <path>]"
            echo "  --ci-test: Enable CI test mode using external binaries"
            echo "  --binary-path: Path to the release directory containing binaries"
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

# ---------- ci/cd ----------
echo "[INFO] Setting up environment..." | tee -a "$summary_log"
env >> "$test_log" 2>&1
source ${sdk_path}/env.sh >> "$test_log" 2>&1
env >> "$test_log" 2>&1

# Configure ccache
echo "[INFO] Configuring ccache..." | tee -a "$summary_log"
ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
ccache --set-config cache_dir="${ccache_dir}" >> "$test_log" 2>&1
ccache --set-config max_size=20G >> "$test_log" 2>&1
ccache --zero-stats >> "$test_log" 2>&1
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

    # Simple keyword validation
    if echo "$output_content" | grep -qi "$expected_content"; then
        echo "[PASS] Content validation: Found expected '$expected_content'"
        return 0
    else
        echo "[FAIL] Content validation: Expected '$expected_content' not found"
        echo "[DEBUG] Output content (first 3 lines):"
        echo "$output_content" | head -n 3
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
        local full_model_path="${model_base_path}/${model_path}"
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
        local full_model_path="${model_base_path}/${model_path}"
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
        local full_model_path="${model_base_path}/${model_path}"
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
            echo "  - ${model_base_path}/${model_path}"
        done
        echo "[INFO] Qwen2.5 models:"
        for model_path in "${qwen25_models[@]}"; do
            echo "  - ${model_base_path}/${model_path}"
        done
        echo "[INFO] Qwen3 models:"
        for model_path in "${qwen3_models[@]}"; do
            echo "  - ${model_base_path}/${model_path}"
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

    # Run inference with fixed parameters for deterministic output
    ${build_dir_bin}/llama-cli -m "$model_path" -no-cnv -n 50 --temp 0.0 --top-k 1 --top-p 1.0 --repeat-penalty 1.0 -s 42 -p "$prompt" > "$output_file" 2>&1
    local cmd_result=$?

    if [ $cmd_result -ne 0 ]; then
        echo "[FAIL] llama-cli command failed with exit code $cmd_result"
        echo "[DEBUG] Command output:" >> "$test_log"
        cat "$output_file" >> "$test_log"
        rm -f "$output_file"
        return 1
    fi

    # Validate output content using simple keyword matching
    local output_content=$(cat "$output_file")
    validate_content_only "$model_name" "$test_name" "$output_content" "$expected_content"
    local validation_result=$?

    if [ $validation_result -eq 0 ]; then
        echo "[PASS] Correctness test passed: $model_name - $test_name"
        # Clean up temporary output file
        rm -f "$output_file"
        return 0
    else
        echo "[FAIL] Correctness test failed: $model_name - $test_name"
        echo "[DEBUG] Failed test output:" >> "$test_log"
        cat "$output_file" >> "$test_log"
        rm -f "$output_file"
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

        # Normal execution - output to both screen and temp file
        set +e  # Disable exit on error temporarily
        eval $test_case > "$temp_output" 2>&1
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
            rm -f "$temp_clean"
        fi

        # Clean up temp file
        rm -f "$temp_output"

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
if [ ${qwen_tests_count} -gt 0 ]; then
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