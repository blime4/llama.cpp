#!/bin/bash

# Qwen模型单元测试脚本
# 使用 llama-completion 进行确定性输出测试

# 不使用 set -e，因为需要手动处理错误
set -o pipefail

# ============ 配置区 ============
# 支持从环境变量或参数传入配置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# 默认配置，可通过参数覆盖
DEFAULT_MODEL_BASE_PATH="${LOCAL_MODEL_PATH:-/mars/aebox/LLM/model}"
DEFAULT_LLAMA_COMPLETION="./build_x86_64/bin/llama-completion"

# 解析命令行参数
UPDATE_MODE=false
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --update)
            UPDATE_MODE=true
            shift
            ;;
        --help)
            echo "Usage: $0 [--update] [MODEL_BASE_PATH] [LLAMA_COMPLETION]"
            echo ""
            echo "Options:"
            echo "  --update           Update expected answers from current model responses"
            echo "  --help             Show this help message"
            echo ""
            echo "Arguments:"
            echo "  MODEL_BASE_PATH    Base path for model files (default: $DEFAULT_MODEL_BASE_PATH)"
            echo "  LLAMA_COMPLETION          Path to llama-completion binary (default: $DEFAULT_LLAMA_COMPLETION)"
            exit 0
            ;;
        *)
            # 收集位置参数
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

# 从位置参数或环境变量获取配置
MODEL_BASE_PATH="${POSITIONAL_ARGS[0]:-${MODEL_BASE_PATH:-$DEFAULT_MODEL_BASE_PATH}}"
LLAMA_COMPLETION="${POSITIONAL_ARGS[1]:-${LLAMA_COMPLETION:-$DEFAULT_LLAMA_COMPLETION}}"
LOG_FILE="qwen_model_test_$(date +%Y%m%d_%H%M%S).log"

# 测试参数 - 设置为确定性输出
TEMPERATURE=0.0
TOP_K=1
TOP_P=1.0
SEED=42
MAX_TOKENS=50
GPU_LAYERS=999

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# YAML文件路径
TEST_CASES_FILE="tests/test_cases.yml"

# ============ 函数定义 ============

# 从test_cases.yml文件中读取指定测试组和测试用例的期望答案
get_expected_answer() {
    local test_group="$1"
    local test_name="$2"

    if [ ! -f "$TEST_CASES_FILE" ]; then
        echo ""
        return 1
    fi

    # 使用简单的文本解析来获取答案
    local in_group=false
    local in_test=false

    while IFS= read -r line; do
        # 检查是否进入指定的测试组部分
        if [[ "$line" =~ ^${test_group}: ]]; then
            in_group=true
            continue
        fi

        # 如果在测试组中，查找对应的测试用例
        if [ "$in_group" = true ]; then
            # 检查是否到达下一个测试组
            if [[ "$line" =~ ^[a-zA-Z0-9_\.\-]+:[[:space:]]*$ ]] && [[ ! "$line" =~ ^${test_group}: ]]; then
                break
            fi

            # 查找指定的测试用例
            if [[ "$line" =~ ^[[:space:]]+${test_name}:[[:space:]]*$ ]]; then
                in_test=true
                continue
            fi

            # 如果在测试用例中，查找expected字段
            if [ "$in_test" = true ]; then
                if [[ "$line" =~ ^[[:space:]]+expected:[[:space:]]*\"(.*)\"[[:space:]]*$ ]]; then
                    echo "${BASH_REMATCH[1]}"
                    return 0
                fi
                # 检查是否到达下一个测试用例
                if [[ "$line" =~ ^[[:space:]]+[a-zA-Z0-9_\.\-]+:[[:space:]]*$ ]] && [[ ! "$line" =~ ^[[:space:]]+expected: ]]; then
                    in_test=false
                fi
            fi
        fi
    done < "$TEST_CASES_FILE"

    echo ""
    return 1
}

# 更新test_cases.yml文件中的期望答案
update_expected_answer() {
    local test_group="$1"
    local test_name="$2"
    local new_answer="$3"

    new_answer=$(echo "$new_answer" | sed 's/[[:space:]]\+/ /g' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//')
    
    awk -v group="$test_group" -v test="$test_name" -v val="$new_answer" '
        $0 == group ":" { in_group=1; print; next }
        in_group && /^[a-z0-9_]+_test_cases:$/ && $0 != group ":" { in_group=0; in_test=0 }
        in_group && $0 == "  " test ":" { in_test=1; print; next }
        in_test && /^[a-z_]+:$/ { in_test=0 }
        in_test && /^    expected:/ { 
            print "    expected: \"" val "\""
            next 
        }
        { print }
    ' "$TEST_CASES_FILE" > "${TEST_CASES_FILE}.tmp" && mv "${TEST_CASES_FILE}.tmp" "$TEST_CASES_FILE"
}

# 从YAML文件读取测试用例组
load_test_cases() {
    local test_group="$1"

    if [ ! -f "$TEST_CASES_FILE" ]; then
        echo "[ERROR] 测试用例文件不存在: $TEST_CASES_FILE" | tee -a "$LOG_FILE"
        return 1
    fi

    local in_group=false
    local test_name=""
    local in_test_case=false
    local in_prompt=false
    local in_expected=false
    local prompt=""
    local expected=""
    local output_final=true

    while IFS= read -r line; do
        # 检查是否进入指定的测试组
        if [[ "$line" =~ ^${test_group}: ]]; then
            in_group=true
            continue
        fi

        # 如果在测试组中，查找测试用例
        if [ "$in_group" = true ]; then
            # 检查是否到达下一个组
            if [[ "$line" =~ ^[a-zA-Z0-9_\.\-]+:[[:space:]]*$ ]] && [[ ! "$line" =~ ^${test_group}: ]]; then
                if [ -n "$test_name" ] && [ -n "$prompt" ]; then
                    echo "${test_name}|${prompt}|${expected}"
                fi
                output_final=false
                break
            fi

            # 查找测试用例名称
            if [[ "$line" =~ ^[[:space:]]+([a-zA-Z0-9_\.\-]+):[[:space:]]*$ ]]; then
                if [ -n "$test_name" ] && [ -n "$prompt" ]; then
                    echo "${test_name}|${prompt}|${expected}"
                fi
                test_name="${BASH_REMATCH[1]}"
                in_test_case=true
                prompt=""
                expected=""
                continue
            fi

            # 如果在测试用例中，查找 prompt 和 expected
            if [ "$in_test_case" = true ]; then
                if [[ "$line" =~ ^[[:space:]]+prompt:[[:space:]]*\"(.*)\"[[:space:]]*$ ]]; then
                    prompt="${BASH_REMATCH[1]}"
                elif [[ "$line" =~ ^[[:space:]]+expected:[[:space:]]*\"(.*)\"[[:space:]]*$ ]]; then
                    expected="${BASH_REMATCH[1]}"
                fi
            fi
        fi
    done < "$TEST_CASES_FILE"

    # 输出最后一个测试用例（仅当到达文件末尾时）
    if [ "$output_final" = true ] && [ -n "$test_name" ] && [ -n "$prompt" ]; then
        echo "${test_name}|${prompt}|${expected}"
    fi
}

# 检查依赖
check_dependencies() {
    # 检查llama-completion二进制文件
    if [ ! -x "$LLAMA_COMPLETION" ]; then
        echo "[ERROR] llama-completion not found or not executable: $LLAMA_COMPLETION" | tee -a "$LOG_FILE"
        return 1
    fi

    # 检查模型基础路径
    if [ ! -d "$MODEL_BASE_PATH" ]; then
        echo "[ERROR] Model base path not found: $MODEL_BASE_PATH" | tee -a "$LOG_FILE"
        return 1
    fi

    return 0
}

# 提取模型输出内容（只提取 prompt + response，去除所有日志）
extract_model_output() {
    local full_output="$1"

    local result=""
    local found_generate=false
    
    while IFS= read -r line; do
        [[ "$line" == \[* ]] && continue
        [[ "$line" == load_tensors:* ]] && continue
        [[ "$line" == print_info:* ]] && continue
        [[ "$line" == load:* ]] && continue
        [[ "$line" == system_info:* ]] && continue
        [[ "$line" == sampler* ]] && continue
        [[ "$line" == DEPRECATED:* ]] && continue
        [[ "$line" == build:* ]] && continue
        [[ "$line" == main:* ]] && continue
        [[ "$line" == llama_* ]] && continue
        [[ "$line" == common_* ]] && continue
        [[ "$line" == Reset* ]] && continue
        [[ "$line" == Capturing* ]] && continue
        [[ "$line" == n_ctx* ]] && continue
        [[ "$line" == n_batch* ]] && continue
        [[ "$line" == n_predict* ]] && continue
        [[ "$line" == n_keep* ]] && continue
        [[ "$line" == *repeat_penalty* ]] && continue
        [[ "$line" == *frequency_penalty* ]] && continue
        [[ "$line" == *presence_penalty* ]] && continue
        [[ "$line" == *top_k* ]] && continue
        [[ "$line" == *top_p* ]] && continue
        [[ "$line" == *dry_* ]] && continue
        [[ "$line" == mirostat* ]] && continue
        [[ "$line" == typical* ]] && continue
        [[ "$line" == *temp* ]] && continue
        [[ "$line" =~ ^\.\.*$ ]] && continue
        
        if [[ "$line" == generate:* ]]; then
            found_generate=true
            continue
        fi
        
        if [[ "$line" == common_perf_print:* ]] || [[ "$line" == "-----------------------------" ]]; then
            break
        fi
        
        if [[ "$line" == 计算过程如下* ]] || [[ "$line" == 首先* ]] || [[ "$line" == 好的，用户* ]]; then
            break
        fi
        
        if [[ "$found_generate" == true ]]; then
            [[ -z "${line// }" ]] && continue
            
            if [[ -z "$result" ]]; then
                result="$line"
            else
                result="$result $line"
            fi
        fi
    done <<< "$full_output"
    
    if [[ -n "$result" ]]; then
        # 去除 \n \r 等转义字符，压缩成一行
        echo "$result" | sed 's/\\n/ /g' | sed 's/\\r/ /g' | sed 's/[[:space:]]\+/ /g' | sed 's/^[[:space:]]*//' | sed 's/[[:space:]]*$//'
    fi
}

# 验证测试结果或更新期望答案
validate_test_result() {
    local test_name="$1"
    local expected_content="$2"
    local actual_response="$3"
    local model_name="${4:-}"
    local test_case="${5:-}"
    local test_group="${6:-}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    if [ "$UPDATE_MODE" = "true" ]; then
        # 更新模式：收集实际答案并更新test_cases.yml文件
        if [ -n "$actual_response" ] && [ "$actual_response" != "[错误: 无法解析响应]" ]; then
            if [ -n "$test_group" ] && [ -n "$test_case" ]; then
                update_expected_answer "$test_group" "$test_case" "$actual_response"
                echo "[UPDATE] $test_name: 已更新期望答案为 '$actual_response'" | tee -a "$LOG_FILE"
            else
                echo "[UPDATE] $test_name: 获得有效回复 '$actual_response'（未指定test_group/test_case，跳过更新）" | tee -a "$LOG_FILE"
            fi
            PASSED_TESTS=$((PASSED_TESTS + 1))
            return 0
        else
            echo "[UPDATE_FAIL] $test_name: 未获得有效回复" | tee -a "$LOG_FILE"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            return 1
        fi
    else
        # 正常验证模式：从test_cases.yml文件读取期望答案进行精确匹配
        if [ -n "$test_group" ] && [ -n "$test_case" ]; then
            expected_content=$(get_expected_answer "$test_group" "$test_case")
        fi

        if [ -z "$expected_content" ]; then
            # 如果没有指定期望内容，只要有回复就算通过
            if [ -n "$actual_response" ] && [ "$actual_response" != "[错误: 无法解析响应]" ]; then
                echo "[PASS] $test_name: 获得有效回复" | tee -a "$LOG_FILE"
                PASSED_TESTS=$((PASSED_TESTS + 1))
                return 0
            else
                echo "[FAIL] $test_name: 未获得有效回复" | tee -a "$LOG_FILE"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                return 1
            fi
        else
            # 精确匹配模式
            if [ "$actual_response" = "$expected_content" ]; then
                echo "[PASS] $test_name: 回复完全匹配期望内容" | tee -a "$LOG_FILE"
                PASSED_TESTS=$((PASSED_TESTS + 1))
                return 0
            else
                echo "[FAIL] $test_name: 回复不完全匹配期望内容" | tee -a "$LOG_FILE"
                echo "[FAIL] 期望: '$expected_content'" | tee -a "$LOG_FILE"
                echo "[FAIL] 实际: '$actual_response'" | tee -a "$LOG_FILE"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                return 1
            fi
        fi
    fi
}

# 运行单个模型测试
run_model_test() {
    local model_name="$1"
    local test_name="$2"
    local prompt="$3"
    local expected_content="$4"
    local model_path="$5"
    local use_fa="${6:-false}"
    local test_group="${7:-}"

    echo "" | tee -a "$LOG_FILE"
    echo "[INFO] ========================================" | tee -a "$LOG_FILE"
    echo "[INFO] 测试: $model_name - $test_name" | tee -a "$LOG_FILE"
    echo "[INFO] 提示: $prompt" | tee -a "$LOG_FILE"
    if [ "$UPDATE_MODE" != "true" ]; then
        echo "[INFO] 期望: $expected_content" | tee -a "$LOG_FILE"
    fi
    echo "[INFO] ========================================" | tee -a "$LOG_FILE"
    echo "-----------------------------" | tee -a "$LOG_FILE"

    local start_time end_time duration ret temp_output full_output model_output
    start_time=$(date +%s)
    set +e
    temp_output=$(mktemp)

    # 输出执行命令
    local cmd
    if [ "$use_fa" = "true" ]; then
        cmd="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} QWEN_USE_FP16=1 DLEOL_DISABLE_CU_MATMUL=1 \"$LLAMA_COMPLETION\" -m \"$model_path\" -no-cnv -n 50 --temp $TEMPERATURE --top-k $TOP_K --top_p $TOP_P --repeat-penalty 1.0 -s $SEED -fa on -ngl $GPU_LAYERS -p \"$prompt\" -no-cnv"
        echo "[CMD] $cmd" | tee -a "$LOG_FILE"
        echo "[DEBUG] 使用 Flash Attention (-fa)" | tee -a "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} QWEN_USE_FP16=1 DLEOL_DISABLE_CU_MATMUL=1 "$LLAMA_COMPLETION" \
            -m "$model_path" \
            -no-cnv \
            -n 50 \
            --temp $TEMPERATURE \
            --top-k $TOP_K \
            --top_p $TOP_P \
            --repeat-penalty 1.0 \
            -s $SEED \
            -fa on \
            -ngl $GPU_LAYERS \
            -p "$prompt" \
            -no-cnv > "$temp_output" 2>&1
        ret=$?
    else
        cmd="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} QWEN_USE_FP16=1 DLEOL_DISABLE_CU_MATMUL=1 \"$LLAMA_COMPLETION\" -m \"$model_path\" -no-cnv -n 50 --temp $TEMPERATURE --top-k $TOP_K --top_p $TOP_P --repeat-penalty 1.0 -s $SEED -ngl $GPU_LAYERS -p \"$prompt\" -no-cnv"
        echo "[CMD] $cmd" | tee -a "$LOG_FILE"
        echo "[DEBUG] 标准推理模式" >> "$LOG_FILE"
        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} QWEN_USE_FP16=1 DLEOL_DISABLE_CU_MATMUL=1 "$LLAMA_COMPLETION" \
            -m "$model_path" \
            -no-cnv \
            -n 50 \
            --temp $TEMPERATURE \
            --top-k $TOP_K \
            --top_p $TOP_P \
            --repeat-penalty 1.0 \
            -s $SEED \
            -ngl $GPU_LAYERS \
            -p "$prompt" \
            -no-cnv > "$temp_output" 2>&1
        ret=$?
    fi
    # 不恢复 set -e，因为脚本最初就没有设置它
    end_time=$(date +%s)
    duration=$((end_time - start_time))

    cat "$temp_output" >> "$LOG_FILE"
    cat "$temp_output"
    echo "-----------------------------" | tee -a "$LOG_FILE"

    if [ $ret -ne 0 ]; then
        echo "[FAIL] $model_name - $test_name (exit code: $ret, ${duration}s)" | tee -a "$LOG_FILE"
        validate_test_result "$model_name - $test_name" "" "[错误: 无法解析响应]" "$model_name" "$test_name"
        rm -f "$temp_output"
        return 1
    fi

    full_output=$(cat "$temp_output")
    model_output=$(extract_model_output "$full_output")
    echo "[DEBUG] 模型生成文本: $model_output" | tee -a "$LOG_FILE"

    validate_test_result "$model_name - $test_name" "$expected_content" "$model_output" "$model_name" "$test_name" "$test_group"
    local result=$?

    rm -f "$temp_output"
    return $result
}

# 检测GPU资源
detect_gpu_resources() {
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "0|0|"
        return
    fi

    local gpu_count
    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    local total_memory_gb=0
    local gpu_memory_list=""

    for ((i=0; i<gpu_count; i++)); do
        local mem_mb
        mem_mb=$(nvidia-smi -i $i --query-gpu=memory.total --format=csv,noheader,nounits)
        local mem_gb=$((mem_mb / 1024))
        total_memory_gb=$((total_memory_gb + mem_gb))
        if [ -z "$gpu_memory_list" ]; then
            gpu_memory_list="$mem_gb"
        else
            gpu_memory_list="${gpu_memory_list},${mem_gb}"
        fi
    done

    echo "${gpu_count}|${total_memory_gb}|${gpu_memory_list}"
}

# 打印测试总结
print_test_summary() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Qwen模型测试总结" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "总测试数: $TOTAL_TESTS" | tee -a "$LOG_FILE"
    echo "通过: $PASSED_TESTS" | tee -a "$LOG_FILE"
    echo "失败: $FAILED_TESTS" | tee -a "$LOG_FILE"
    if [ $TOTAL_TESTS -gt 0 ]; then
        echo "成功率: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%" | tee -a "$LOG_FILE"
    fi
    echo "========================================" | tee -a "$LOG_FILE"

    if [ $FAILED_TESTS -eq 0 ]; then
        echo "[QWEN_MODEL_PASS] 所有Qwen模型测试通过!" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[QWEN_MODEL_FAIL] $FAILED_TESTS 个测试失败" | tee -a "$LOG_FILE"
        return 1
    fi
}

# ============ 主程序 ============

main() {
    echo "Qwen模型单元测试" | tee "$LOG_FILE"
    if [ "$UPDATE_MODE" = "true" ]; then
        echo "模式: 更新期望答案" | tee -a "$LOG_FILE"
    else
        echo "模式: 验证测试（精确匹配）" | tee -a "$LOG_FILE"
    fi
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    echo "模型基础路径: $MODEL_BASE_PATH" | tee -a "$LOG_FILE"
    echo "llama-completion: $LLAMA_COMPLETION" | tee -a "$LOG_FILE"
    echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"

    # 检查依赖
    if ! check_dependencies; then
        echo "依赖检查失败，退出测试" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检查测试用例文件是否存在
    if [ ! -f "$TEST_CASES_FILE" ]; then
        echo "错误: 测试用例文件不存在: $TEST_CASES_FILE" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 检测GPU资源
    local gpu_info
    gpu_info=$(detect_gpu_resources)
    IFS='|' read -r gpu_count total_memory_gb gpu_memory_list <<< "$gpu_info"
    echo "[INFO] GPU资源: ${gpu_count} GPUs, 总内存 ${total_memory_gb}GB" | tee -a "$LOG_FILE"

    # 定义测试模型和用例
    declare -A qwen2_models=(
        # ["Qwen2-1.5Moe.Q4_K_M"]="Qwen2-1.5B-Moe-GGUF/Qwen2-1.5Moe.Q4_K_M.gguf"
    )

    declare -A qwen25_models=(
        # ["qwen2.5-1.5b-instruct-fp16"]="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-fp16.gguf"
        ["qwen2.5-1.5b-instruct-q4_k_m"]="Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
    )

    declare -A qwen3_models=(
        ["Qwen3-30B-A3B-Q4_K_M"]="Qwen3-30B-A3B-GGUF/Qwen3-30B-A3B-Q4_K_M.gguf"
    )

    # 运行Qwen2测试
    for model_name in "${!qwen2_models[@]}"; do
        local model_path="${MODEL_BASE_PATH}/${qwen2_models[$model_name]}"
        if [ -f "$model_path" ]; then
            echo "[INFO] 测试模型: $model_name" | tee -a "$LOG_FILE"
            while IFS='|' read -r test_name prompt expected; do
                if [ -n "$test_name" ] && [ -n "$prompt" ]; then
                    run_model_test "$model_name" "$test_name" "$prompt" "$expected" "$model_path" "false" "test_cases"
                fi
            done < <(load_test_cases "test_cases")
        else
            echo "[WARN] 模型文件不存在: $model_path" | tee -a "$LOG_FILE"
        fi
    done

    # 运行Qwen2.5测试
    for model_name in "${!qwen25_models[@]}"; do
        local model_path="${MODEL_BASE_PATH}/${qwen25_models[$model_name]}"
        if [ -f "$model_path" ]; then
            echo "[INFO] 测试模型: $model_name" | tee -a "$LOG_FILE"
            # qwen2.5-1.5b-instruct-q4_k_m 使用Flash Attention
            local use_fa="false"
            if [[ "$model_name" == *"q4_k_m"* ]]; then
                use_fa="true"
            fi
            while IFS='|' read -r test_name prompt expected; do
                if [ -n "$test_name" ] && [ -n "$prompt" ]; then
                    run_model_test "$model_name" "$test_name" "$prompt" "$expected" "$model_path" "$use_fa" "qwen25_test_cases"
                fi
            done < <(load_test_cases "qwen25_test_cases")
        else
            echo "[WARN] 模型文件不存在: $model_path" | tee -a "$LOG_FILE"
        fi
    done

    # 运行Qwen3测试（使用qwen3_test_cases）
    for model_name in "${!qwen3_models[@]}"; do
        local model_path="${MODEL_BASE_PATH}/${qwen3_models[$model_name]}"
        if [ -f "$model_path" ]; then
            echo "[INFO] 测试模型: $model_name" | tee -a "$LOG_FILE"
            while IFS='|' read -r test_name prompt expected; do
                if [ -n "$test_name" ] && [ -n "$prompt" ]; then
                    run_model_test "$model_name" "$test_name" "$prompt" "$expected" "$model_path" "false" "qwen3_test_cases"
                fi
            done < <(load_test_cases "qwen3_test_cases")
        else
            echo "[WARN] 模型文件不存在: $model_path" | tee -a "$LOG_FILE"
        fi
    done

    # 打印总结并返回结果
    if print_test_summary; then
        echo "完整日志已保存到: $LOG_FILE"
        exit 0
    else
        echo "完整日志已保存到: $LOG_FILE"
        exit 1
    fi
}

# 如果直接运行此脚本，执行主程序
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
