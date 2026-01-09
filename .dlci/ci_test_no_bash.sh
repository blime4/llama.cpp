#!/system/bin/sh
#
# CI Test Script for Android (No Bash)
#
# This script runs in Android adb shell environment (basic sh only, no bash).
# It directly executes test binaries without complex bash features.
#
# Required environment variables (must be set by CI):
#   BINARY_PATH - Path to binaries directory (e.g., /data/local/tmp/bin)
#   REPO_PATH   - Path to repository root (e.g., /data/local/tmp/llama.cpp)
#
# Optional environment variables:
#   MODEL_PATH  - Path to model files (default: /data/local/tmp/models)
#

set -e

echo "======================================"
echo "CI Test for Android (No Bash)"
echo "======================================"
echo "Started at: $(date)"
echo ""

# Validate BINARY_PATH
if [ -z "$BINARY_PATH" ]; then
    echo "[ERROR] BINARY_PATH is not set"
    exit 1
fi

if [ ! -d "$BINARY_PATH" ]; then
    echo "[ERROR] BINARY_PATH does not exist: $BINARY_PATH"
    exit 1
fi

echo "[INFO] BINARY_PATH: $BINARY_PATH"

# Validate REPO_PATH
if [ -z "$REPO_PATH" ]; then
    echo "[ERROR] REPO_PATH is not set"
    exit 1
fi

if [ ! -d "$REPO_PATH" ]; then
    echo "[ERROR] REPO_PATH does not exist: $REPO_PATH"
    exit 1
fi

echo "[INFO] REPO_PATH: $REPO_PATH"

# Set MODEL_PATH default
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH="/data/local/tmp/models"
fi

echo "[INFO] MODEL_PATH: $MODEL_PATH"
echo ""

# Change to repo directory
cd "$REPO_PATH"

# Test counters
TOTAL=0
PASSED=0
FAILED=0
FAILED_TESTS=""

# Helper function to run a test
run_test() {
    TEST_CMD="$1"
    TEST_NAME="$2"

    TOTAL=$((TOTAL + 1))
    echo "----------------------------------------"
    echo "[TEST $TOTAL] $TEST_NAME"
    echo "Command: $TEST_CMD"
    echo "----------------------------------------"

    if eval "$TEST_CMD"; then
        echo "[PASS] $TEST_NAME"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] $TEST_NAME"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS
  - $TEST_NAME"
    fi
    echo ""
}

# ============================================
# Part 1: Basic Unit Tests
# ============================================
echo "========================================"
echo "Part 1: Basic Unit Tests"
echo "========================================"
echo ""

run_test "$BINARY_PATH/test-arg-parser" "test-arg-parser"
run_test "$BINARY_PATH/test-autorelease" "test-autorelease"
run_test "$BINARY_PATH/test-backend-ops" "test-backend-ops"
run_test "$BINARY_PATH/test-c" "test-c"
run_test "$BINARY_PATH/test-chat" "test-chat"
run_test "$BINARY_PATH/test-chat-parser" "test-chat-parser"
run_test "$BINARY_PATH/test-chat-template" "test-chat-template"
run_test "$BINARY_PATH/test-gguf" "test-gguf"
run_test "$BINARY_PATH/test-grammar-integration" "test-grammar-integration"
run_test "$BINARY_PATH/test-grammar-parser" "test-grammar-parser"
run_test "$BINARY_PATH/test-json-partial" "test-json-partial"
run_test "$BINARY_PATH/test-llama-grammar" "test-llama-grammar"
run_test "$BINARY_PATH/test-log" "test-log"
run_test "$BINARY_PATH/test-model-load-cancel" "test-model-load-cancel"
run_test "$BINARY_PATH/test-mtmd-c-api" "test-mtmd-c-api"
run_test "$BINARY_PATH/test-regex-partial" "test-regex-partial"
run_test "$BINARY_PATH/test-sampling" "test-sampling"
run_test "$BINARY_PATH/test-json-schema-to-grammar" "test-json-schema-to-grammar"

# ============================================
# Part 2: Tokenizer Tests
# ============================================
echo "========================================"
echo "Part 2: Tokenizer Tests"
echo "========================================"
echo ""

run_test "$BINARY_PATH/test-tokenizer-1-bpe models/ggml-vocab-llama-bpe.gguf" "test-tokenizer-1-bpe"
run_test "$BINARY_PATH/test-tokenizer-1-spm models/ggml-vocab-llama-spm.gguf" "test-tokenizer-1-spm"

# ============================================
# Part 3: Qwen Model Tests (if available)
# ============================================
# Part 3: Qwen Model Tests
# ============================================
echo "========================================"
echo "Part 3: Qwen Model Tests"
echo "========================================"
echo ""

# Test 1: Qwen2.5-1.5B Q4_K_M
if [ -f "/models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf" ]; then
    TOTAL=$((TOTAL + 1))
    echo "----------------------------------------"
    echo "[TEST $TOTAL] qwen_2.5_1.5b_q4_k_m"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=0 QWEN_USE_FP16=1 DLEOL_DISABLE_CU_MATMUL=1 \
    $BINARY_PATH/llama-cli \
        -m /models/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf \
        -no-cnv \
        --temp 0.0 \
        --top-k 1 \
        --top-p 1.0 \
        --repeat-penalty 1.0 \
        -s 42 \
        -p "北京是中国的什么？" \
        -n 10 \
        --no-warmup \
        -fa \
        -ngl 999

    if [ $? -eq 0 ]; then
        echo "[PASS] qwen_2.5_1.5b_q4_k_m"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] qwen_2.5_1.5b_q4_k_m"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS
  - qwen_2.5_1.5b_q4_k_m"
    fi
    echo ""
else
    echo "[SKIP] Qwen2.5-1.5B model not found"
    echo ""
fi

# Test 2: Qwen3-30B Q4_K_M
if [ -f "/models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf" ]; then
    TOTAL=$((TOTAL + 1))
    echo "----------------------------------------"
    echo "[TEST $TOTAL] qwen_3_30b_q4_k_m"
    echo "----------------------------------------"

    CUDA_VISIBLE_DEVICES=0 QWEN_USE_FP16=1 DLEOL_DISABLE_CU_MATMUL=1 \
    $BINARY_PATH/llama-cli \
        -m /models/Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf \
        -no-cnv \
        --temp 0.0 \
        --top-k 1 \
        --top-p 1.0 \
        --repeat-penalty 1.0 \
        -s 42 \
        -p "北京是中国的什么？" \
        -n 10 \
        --no-warmup \
        -fa \
        -ngl 999

    if [ $? -eq 0 ]; then
        echo "[PASS] qwen_3_30b_q4_k_m"
        PASSED=$((PASSED + 1))
    else
        echo "[FAIL] qwen_3_30b_q4_k_m"
        FAILED=$((FAILED + 1))
        FAILED_TESTS="$FAILED_TESTS
  - qwen_3_30b_q4_k_m"
    fi
    echo ""
else
    echo "[SKIP] Qwen3-30B model not found"
    echo ""
fi


# ============================================
# Summary
# ============================================
echo ""
echo "========================================"
echo "Test Summary"
echo "========================================"
echo "Total:  $TOTAL"
echo "Passed: $PASSED"
echo "Failed: $FAILED"
echo "========================================"

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "Failed tests:$FAILED_TESTS"
    echo ""
    echo "[RESULT] FAILED"
    exit 1
else
    echo ""
    echo "[RESULT] ALL PASSED"
    exit 0
fi
