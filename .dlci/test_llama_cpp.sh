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
build_dir="$(pwd)/build"

# Check if build directory exists
if [ ! -d "${build_dir}" ]; then
    echo "[ERROR] ${build_dir} does not exist ..."
    exit 1
fi

# Set environment variables
export GGML_DEBUG=1
export PATH="${build_dir}/bin:$PATH"
export LD_LIBRARY_PATH="${build_dir}/bin:$LD_LIBRARY_PATH"

# Enter build directory and run all test cases
cd "${build_dir}/bin"

# basic tests
test_cases_part1=(
    "test-arg-parser"
    "test-autorelease"
    "test-backend-ops"
    "test-c"
    "test-chat"
    "test-chat-parser"
    "test-chat-template"
    "test-gbnf-validator ../../grammars/json.gbnf -c '{\"name\": \"Alice\", \"age\": 25}'"        # Test valid JSON
    "test-gbnf-validator ../../grammars/json.gbnf -c '{\"name\": \"Bob\", \"age\": thirty}'"      # Test invalid JSON (should fail)
    "test-gbnf-validator ../../grammars/arithmetic.gbnf -c 'x = 5'"                               # Test valid arithmetic
    "test-gbnf-validator ../../grammars/arithmetic.gbnf -c 'x = 5 +'"                             # Test invalid arithmetic (should fail)
    "test-gbnf-validator ../../grammars/list.gbnf -c '- First item\n- Second item\n'"
    "test-gguf"
    "test-grammar-integration"
    "test-grammar-parser"
    "test-json-partial"
    # "test-json-schema-to-grammar" # TODO: fix it
    "test-llama-grammar"
    "test-log"
    "test-model-load-cancel"
    "test-mtmd-c-api"
    "test-regex-partial"
    "test-sampling"
    "test-thread-safety --prompt 'hello, llama.cpp' --model ${LOCAL_MODEL_PATH}/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
)

# tokenizer tests
test_cases_part2=(
    # "test-tokenizer-1-bpe ../../models/ggml-vocab-llama-bpe.gguf" # TODO: fix it
    # "test-tokenizer-1-spm ../../models/ggml-vocab-llama-spm.gguf" # TODO: fix it
)

# Function to auto-discover vocab test cases
add_tokenizer_vocab_tests() {
    for vocab in ../../models/*.gguf; do
        inp="${vocab}.inp"
        out="${vocab}.out"
        if [[ -f "$inp" && -f "$out" ]]; then
            test_cases_part2+=("test-tokenizer-0 $vocab")
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