#!/bin/bash
set -e
env
source "${SDK_WORKSPACE}/sdk/env.sh"
env

# Configure ccache
ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
ccache --set-config cache_dir="${ccache_dir}"
ccache --set-config max_size=20G
ccache --zero-stats

# Enter repository directory
echo "[INFO] REPO_PATH: ${REPO_PATH}"
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
cd "${build_dir}"

test_bins=(
    ./bin/test-arg-parser
    ./bin/test-c
    ./bin/test-gguf
    ./bin/test-grammar-parser
    ./bin/test-sampling
    ./bin/test-llama-grammar
    ./bin/test-log
    ./bin/test-chat-parser
    ./bin/test-chat-template
    ./bin/test-grammar-integration
    ./bin/test-json-partial
    ./bin/test-mtmd-c-api
    ./bin/test-regex-partial
    ./bin/test-backend-ops
    ./bin/test-arg-parser
    ./bin/test-autorelease
)

fail_count=0
fail_list=()

for test_bin in "${test_bins[@]}"; do
    if [ ! -x "$test_bin" ]; then
        echo "[LLAMA_CPP_FAIL] $test_bin does not exist or is not executable, skipping"
        fail_count=$((fail_count+1))
        fail_list+=("$test_bin (not found or not executable)")
        continue
    fi
    echo "Running: $test_bin"
    start_time=$(date +%s)
    "$test_bin"
    ret=$?
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    if [ $ret -eq 0 ]; then
        echo "[LLAMA_CPP_PASS] Test $test_bin succeeded, duration: ${duration} seconds"
    else
        echo "[LLAMA_CPP_FAIL] Test $test_bin failed, duration: ${duration} seconds, exit code $ret"
        fail_count=$((fail_count+1))
        fail_list+=("$test_bin (exit code $ret)")
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
