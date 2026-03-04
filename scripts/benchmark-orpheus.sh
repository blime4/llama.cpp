#!/bin/bash
# benchmark-orpheus.sh - Performance benchmark for Orpheus TTS

set -e

# Configuration
MODEL="${MODEL:-/LocalRun/shaobo.xie/models/llama.cpp/orpheus-3b-f16.gguf}"
VOCODER="${VOCODER:-/LocalRun/shaobo.xie/models/llama.cpp/feature-wip/snac-24khz-f16.gguf}"
BIN="${BIN:-./build/bin/llama-orpheus-tts}"
THREADS="${THREADS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-benchmark_results_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

# Test prompts
declare -a PROMPTS=(
    "Hello world."
    "This is a test of the Orpheus text to speech system."
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
)

echo "# Orpheus TTS Benchmark Results"
echo ""
echo "**Date**: $(date)"
echo "**Model**: $MODEL"
echo "**Vocoder**: $VOCODER"
echo "**Threads**: $THREADS"
echo ""
echo "---"
echo ""

run_test() {
    local idx=$1
    local prompt="$2"
    local run=$3

    echo "Running test $idx, run $run: \"$prompt\"" >&2

    output=$($BIN -m "$MODEL" --model-vocoder "$VOCODER" \
        -p "$prompt" -t "$THREADS" \
        -o "$OUTPUT_DIR/test_${idx}_run${run}.wav" 2>&1)

    # Extract metrics
    llm_ms=$(echo "$output" | grep "LLM generation:" | grep -oE '[0-9]+ ms' | head -1 | grep -oE '[0-9]+')
    llm_toks=$(echo "$output" | grep "LLM generation:" | grep -oE '/ [0-9]+ tokens' | grep -oE '[0-9]+')
    vocoder_ms=$(echo "$output" | grep "Vocoder:" | grep -oE '[0-9]+ ms' | head -1 | grep -oE '[0-9]+')
    vocoder_rtf=$(echo "$output" | grep "Vocoder:" | grep -oE '\([0-9.]+ x' | grep -oE '[0-9.]+')
    total_ms=$(echo "$output" | grep "Total:" | grep -oE '[0-9]+ ms' | grep -oE '[0-9]+')
    audio_secs=$(echo "$output" | grep "Audio:" | grep -oE '[0-9.]+ seconds' | grep -oE '[0-9.]+')

    # Calculate ms/tok
    local ms_per_tok=""
    if [ -n "$llm_ms" ] && [ -n "$llm_toks" ] && [ "$llm_toks" -gt 0 ]; then
        ms_per_tok=$(echo "scale=2; $llm_ms / $llm_toks" | bc)
    fi

    echo "| Run $run | ${llm_ms:-N/A} | ${llm_toks:-N/A} | ${ms_per_tok:-N/A} | ${vocoder_ms:-N/A} | ${vocoder_rtf:-N/A} | ${total_ms:-N/A} | ${audio_secs:-N/A} |"
}

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    idx=$((i+1))

    echo "## Test $idx: \"${prompt:0:50}$([ ${#prompt} -gt 50 ] && echo '...')\""
    echo ""
    echo "| Run | LLM (ms) | Tokens | ms/tok | Vocoder (ms) | RTF | Total (ms) | Audio (s) |"
    echo "|-----|----------|--------|--------|--------------|-----|------------|-----------|"

    for run in 1 2 3; do
        run_test "$idx" "$prompt" "$run"
    done

    echo ""
done

echo "---"
echo ""
echo "## System Info"
echo ""
echo "| Component | Value |"
echo "|-----------|-------|"
echo "| CPU | $(lscpu | grep 'Model name' | cut -d: -f2 | xargs) |"
echo "| Cores | $(nproc) |"
echo "| Memory | $(free -h | grep Mem | awk '{print $2}') |"
echo ""
