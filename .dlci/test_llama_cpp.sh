#!/bin/bash
set -e
# ---------- ci/cd ----------
env
source "${SDK_WORKSPACE}/sdk/env.sh"
env

# Configure ccache
ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
ccache --set-config cache_dir="${ccache_dir}"
ccache --set-config max_size=20G
ccache --zero-stats
# ---------- ci/cd ----------

# Enter repository directory
echo "[INFO] REPO_PATH: ${REPO_PATH}"
echo "[INFO] LOCAL_MODEL_PATH: ${LOCAL_MODEL_PATH}"
cd "${REPO_PATH}"
build_dir="$(pwd)/../build"

# Check if build directory exists
if [ ! -d "${build_dir}" ]; then
    echo "[ERROR] ${build_dir} does not exist ..."
    exit 1
fi

# Set environment variables
export GGML_DEBUG=1

# basic tests
test_cases_part1=(
    "${build_dir}/bin/test-arg-parser"
    "${build_dir}/bin/test-autorelease"
    "${build_dir}/bin/test-backend-ops"
    "${build_dir}/bin/test-c"
    "${build_dir}/bin/test-chat"
    "${build_dir}/bin/test-chat-parser"
    "${build_dir}/bin/test-chat-template"
    "${build_dir}/bin/test-gbnf-validator grammars/json.gbnf -c '{\"name\": \"Alice\", \"age\": 25}'"        # Test valid JSON
    "${build_dir}/bin/test-gbnf-validator grammars/json.gbnf -c '{\"name\": \"Bob\", \"age\": thirty}'"      # Test invalid JSON (should fail)
    "${build_dir}/bin/test-gbnf-validator grammars/arithmetic.gbnf -c 'x = 5'"                               # Test valid arithmetic
    "${build_dir}/bin/test-gbnf-validator grammars/arithmetic.gbnf -c 'x = 5 +'"                             # Test invalid arithmetic (should fail)
    "${build_dir}/bin/test-gbnf-validator grammars/list.gbnf -c '- First item\n- Second item\n'"
    "${build_dir}/bin/test-gguf"
    "${build_dir}/bin/test-grammar-integration"
    "${build_dir}/bin/test-grammar-parser"
    "${build_dir}/bin/test-json-partial"
    "${build_dir}/bin/test-json-schema-to-grammar"
    "${build_dir}/bin/test-llama-grammar"
    "${build_dir}/bin/test-log"
    "${build_dir}/bin/test-model-load-cancel"
    "${build_dir}/bin/test-mtmd-c-api"
    "${build_dir}/bin/test-regex-partial"
    "${build_dir}/bin/test-sampling"
    "${build_dir}/bin/test-thread-safety --prompt 'hello, llama.cpp' --model ${LOCAL_MODEL_PATH}/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
)

# tokenizer tests
test_cases_part2=(
    "${build_dir}/bin/test-tokenizer-1-bpe models/ggml-vocab-llama-bpe.gguf"
    "${build_dir}/bin/test-tokenizer-1-spm models/ggml-vocab-llama-spm.gguf"
)

# Function to auto-discover vocab test cases
add_tokenizer_vocab_tests() {
    for vocab in models/*.gguf; do
        inp="${vocab}.inp"
        out="${vocab}.out"
        if [[ -f "$inp" && -f "$out" ]]; then
            test_cases_part2+=("${build_dir}/bin/test-tokenizer-0 $vocab")
        fi
    done
}

add_tokenizer_vocab_tests

test_cases=(
    "${test_cases_part1[@]}"
    "${test_cases_part2[@]}"
)

fail_count=0
fail_list=()

for test_case in "${test_cases[@]}"; do
    # Get the executable name (first word)
    test_bin=$(echo "$test_case" | awk '{print $1}')
    if [ ! -x "$test_bin" ]; then
        echo "[LLAMA_CPP_FAIL] $test_bin does not exist or is not executable, skipping"
        fail_count=$((fail_count+1))
        fail_list+=("$test_case (not found or not executable)")
        continue
    fi
    echo "Running: $test_case"
    start_time=$(date +%s)
    eval $test_case
    ret=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    if [ $ret -eq 0 ]; then
        echo "[LLAMA_CPP_PASS] Test $test_case succeeded, duration: ${duration} seconds"
    else
        echo "[LLAMA_CPP_FAIL] Test $test_case failed, duration: ${duration} seconds, exit code $ret"
        fail_count=$((fail_count+1))
        fail_list+=("$test_case (exit code $ret)")
    fi
    echo "-----------------------------"
done

if [ $fail_count -ne 0 ]; then
    echo "============================="
    echo "[LLAMA_CPP_FAIL] $fail_count test(s) failed:"
    for fail_item in "${fail_list[@]}"; do
        echo "  - $fail_item"
    done
    exit 1
else
    echo "============================="
    echo "[LLAMA_CPP_PASS] All tests passed!"
fi

echo "<>test-backend-ops end ...<>"
