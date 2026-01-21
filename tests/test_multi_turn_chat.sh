#!/bin/bash

# 多轮对话单元测试脚本
# 使用 llama-server 实现真正的上下文保持
#
# 支持通过 TEST_LOG_FILE 环境变量从父脚本接收日志文件路径
# 以实现测试日志集中管理（详见 OpenSpec 变更：consolidate-test-logs）

# 不使用 set -e，因为 jq 可能返回非零退出码
set -o pipefail

# ============ 配置区 ============
# 支持从环境变量或参数传入配置
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export QWEN_USE_FP16=${QWEN_USE_FP16:-1}
export DLEOL_DISABLE_CU_MATMUL=${DLEOL_DISABLE_CU_MATMUL:-1}

# 默认配置，可通过参数覆盖
DEFAULT_MODEL_PATH="${LOCAL_MODEL_PATH:-/mars/aebox/LLM/model}/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_LLAMA_SERVER="./build_x86_64/bin/llama-server"
DEFAULT_SERVER_PORT=8080

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
            echo "Usage: $0 [--update] [MODEL_PATH] [LLAMA_SERVER] [SERVER_PORT]"
            echo ""
            echo "Options:"
            echo "  --update    Update expected answers from current model responses"
            echo "  --help      Show this help message"
            echo ""
            echo "Arguments:"
            echo "  MODEL_PATH    Path to the model file (default: $DEFAULT_MODEL_PATH)"
            echo "  LLAMA_SERVER  Path to llama-server binary (default: $DEFAULT_LLAMA_SERVER)"
            echo "  SERVER_PORT   Port for the server (default: $DEFAULT_SERVER_PORT)"
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
MODEL_PATH="${POSITIONAL_ARGS[0]:-${MODEL_PATH:-$DEFAULT_MODEL_PATH}}"
LLAMA_SERVER="${POSITIONAL_ARGS[1]:-${LLAMA_SERVER:-$DEFAULT_LLAMA_SERVER}}"
SERVER_PORT="${POSITIONAL_ARGS[2]:-${SERVER_PORT:-$DEFAULT_SERVER_PORT}}"
SERVER_URL="http://localhost:${SERVER_PORT}"
if [ -n "${TEST_LOG_FILE:-}" ]; then
    LOG_FILE="$TEST_LOG_FILE"
else
    LOG_FILE="multi_turn_test_$(date +%Y%m%d_%H%M%S).log"
fi
SCRIPT_NAME="test_multi_turn_chat"

# 测试参数 - 设置为确定性输出
TEMPERATURE=0.0
TOP_K=1
TOP_P=1.0
SEED=42
MAX_TOKENS=300

# 测试结果统计
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

# YAML文件路径
EXPECTED_ANSWERS_FILE="tests/expected_answers.yml"

# ============ 函数定义 ============

# 转义字符串中的特殊字符用于 sed 替换
# 处理: \ / & 和换行符
escape_for_sed() {
    local str="$1"
    # 转义反斜杠和分隔符
    str="${str//\\/\\\\}"
    str="${str//\//\\/}"
    str="${str//&/\\&}"
    # 将换行符转义为 \n
    str="${str//$'\n'/\\n}"
    printf '%s' "$str"
}

# 验证测试用例名称和轮次标识的安全性
validate_test_case_name() {
    local name="$1"
    # 只允许字母、数字、下划线和连字符
    if [[ ! "$name" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*$ ]]; then
        echo "[错误] 测试用例名称包含非法字符: $name" | tee -a "$LOG_FILE"
        return 1
    fi
    return 0
}

# 读取YAML文件中指定测试用例和轮次的期望答案
get_expected_answer() {
    local test_case="$1"
    local round="$2"

    if [ ! -f "$EXPECTED_ANSWERS_FILE" ]; then
        echo ""
        return 1
    fi

    # 使用简单的文本解析来获取答案
    local section_found=false
    local in_test_case=false

    while IFS= read -r line; do
        # 检查是否进入指定的测试用例部分
        if [[ "$line" =~ ^${test_case}: ]]; then
            in_test_case=true
            continue
        fi

        # 如果在测试用例中，查找对应的轮次
        if [ "$in_test_case" = true ]; then
            # 检查是否到达下一个测试用例（以字母开头，后跟冒号）
            if [[ "$line" =~ ^[a-zA-Z_]+: ]] && [[ ! "$line" =~ ^${test_case}: ]]; then
                break
            fi

            # 查找指定的轮次
            if [[ "$line" =~ ^[[:space:]]*${round}:[[:space:]]*\"(.*)\" ]]; then
                echo "${BASH_REMATCH[1]}"
                return 0
            fi
        fi
    done < "$EXPECTED_ANSWERS_FILE"

    echo ""
    return 1
}

# 更新YAML文件中的期望答案
update_expected_answer() {
    local test_case="$1"
    local round="$2"
    local new_answer="$3"

    # 验证名称安全性
    if ! validate_test_case_name "$test_case"; then
        return 1
    fi
    if ! validate_test_case_name "$round"; then
        return 1
    fi

    # 如果YAML文件不存在，创建它
    if [ ! -f "$EXPECTED_ANSWERS_FILE" ]; then
        mkdir -p "$(dirname "$EXPECTED_ANSWERS_FILE")"
        cat > "$EXPECTED_ANSWERS_FILE" << 'EOF'
# 多轮对话测试期望答案
# 此文件由 --update 选项自动更新

EOF
    fi

    local temp_file="${EXPECTED_ANSWERS_FILE}.tmp"
    local updated=false
    local in_test_case=false

    # 使用更安全的方法：逐行处理
    while IFS= read -r line || [ -n "$line" ]; do
        # 检查是否进入指定的测试用例部分
        if [[ "$line" =~ ^${test_case}: ]]; then
            in_test_case=true
            echo "$line" >> "$temp_file"
            continue
        fi

        if [ "$in_test_case" = true ]; then
            # 检查是否到达下一个测试用例
            if [[ "$line" =~ ^[a-zA-Z_][a-zA-Z0-9_-]*: ]]; then
                in_test_case=false
                # 在新测试用例之前添加本轮次（如果未找到）
                if [ "$updated" = false ]; then
                    echo "  ${round}: \"$(escape_for_sed "$new_answer")\"" >> "$temp_file"
                    updated=true
                fi
                echo "$line" >> "$temp_file"
                continue
            fi

            # 查找指定的轮次并更新
            if [[ "$line" =~ ^[[:space:]]*${round}: ]]; then
                echo "  ${round}: \"$(escape_for_sed "$new_answer")\"" >> "$temp_file"
                updated=true
                continue
            fi

            echo "$line" >> "$temp_file"
        else
            echo "$line" >> "$temp_file"
        fi
    done < "$EXPECTED_ANSWERS_FILE"

    # 如果测试用例不存在或轮次未找到，追加到文件末尾
    if [ "$updated" = false ]; then
        if ! grep -q "^${test_case}:" "$EXPECTED_ANSWERS_FILE"; then
            echo "" >> "$temp_file"
            echo "${test_case}:" >> "$temp_file"
        fi
        echo "  ${round}: \"$(escape_for_sed "$new_answer")\"" >> "$temp_file"
    fi

    # 原子替换文件
    mv "$temp_file" "$EXPECTED_ANSWERS_FILE"
}

# 清理可能残留的服务器进程
cleanup_stale_processes() {
    # 查找可能在同一端口上运行的旧服务器进程
    local old_pids=$(lsof -ti:$SERVER_PORT 2>/dev/null || true)
    if [ -n "$old_pids" ]; then
        echo "检测到端口 $SERVER_PORT 上有残留进程，正在清理..." | tee -a "$LOG_FILE"
        for pid in $old_pids; do
            # 检查进程是否是 llama-server
            if ps -p "$pid" -o command= 2>/dev/null | grep -q "llama-server"; then
                echo "  杀死残留进程 PID: $pid" | tee -a "$LOG_FILE"
                kill "$pid" 2>/dev/null || true
            fi
        done
        # 等待进程完全退出
        sleep 1
    fi
}

# 检查依赖
check_dependencies() {
    # 检查模型文件
    if [ ! -f "$MODEL_PATH" ]; then
        echo "[ERROR] Model file not found: $MODEL_PATH" | tee -a "$LOG_FILE"
        return 1
    fi

    # 检查服务器二进制文件
    if [ ! -x "$LLAMA_SERVER" ]; then
        echo "[ERROR] llama-server not found or not executable: $LLAMA_SERVER" | tee -a "$LOG_FILE"
        return 1
    fi

    # 检查必要工具
    if ! command -v curl >/dev/null 2>&1; then
        echo "[ERROR] curl is required but not installed" | tee -a "$LOG_FILE"
        return 1
    fi

    if ! command -v jq >/dev/null 2>&1; then
        echo "[WARN] jq not found, will use basic JSON parsing" | tee -a "$LOG_FILE"
    fi

    return 0
}

# 启动服务器
start_server() {
    # 先清理可能残留的进程
    cleanup_stale_processes

    echo "启动 llama-server..." | tee -a "$LOG_FILE"
    # Simplified configuration to avoid crashes:
    # - No warmup/CUDA graphs (--no-warmup)
    # - Multiple parallel slots (-np 3) for test isolation
    # - Larger context to handle multi-turn conversations (-c 4096)
    # - Disabled slot prompt similarity (-sps 0) to avoid cross-test pollution
    $LLAMA_SERVER \
        -m "$MODEL_PATH" \
        --port $SERVER_PORT \
        -ngl 999 \
        -fa on \
        --no-warmup \
        -np 3 \
        -c 4096 \
        -sps 0 \
        > "${LOG_FILE}.server" 2>&1 &

    SERVER_PID=$!
    echo "服务器 PID: $SERVER_PID" | tee -a "$LOG_FILE"

    # 等待服务器启动
    echo "等待服务器就绪..."
    for i in {1..120}; do
        # -f: HTTP错误时失败(如503), -s: 静默模式, -o: 输出到文件
        if curl -s -f -o /dev/null "${SERVER_URL}/health" 2>&1; then
            echo "服务器已就绪，等待模型加载完成..." | tee -a "$LOG_FILE"
            # 没有warmup，等待时间可以短一些
            sleep 5
            echo "模型加载完成" | tee -a "$LOG_FILE"
            return 0
        fi
        sleep 1
    done

    echo "错误: 服务器启动超时" | tee -a "$LOG_FILE"
    return 1
}

# 停止服务器
stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        echo "停止服务器 (PID: $SERVER_PID)..." | tee -a "$LOG_FILE"
        kill $SERVER_PID 2>/dev/null || true
        wait $SERVER_PID 2>/dev/null || true
    fi
}

# 发送单轮对话请求
# 参数: $1 = messages_json, $2 = slot_id (optional, default -1 for auto-selection)
send_message() {
    local messages_json="$1"
    local slot_id="${2:--1}"  # 默认-1表示自动选择slot
    local response
    local max_retries=5  # 增加重试次数
    local retry=0

    # 构建完整的请求JSON
    local request_json=$(cat <<EOF
{
    "messages": $messages_json,
    "max_tokens": $MAX_TOKENS,
    "temperature": $TEMPERATURE,
    "top_k": $TOP_K,
    "top_p": $TOP_P,
    "seed": $SEED,
    "id_slot": $slot_id
}
EOF
)

    # Debug: 记录请求JSON
    echo "请求JSON: $request_json" >> "${LOG_FILE}.debug"

    while [ $retry -lt $max_retries ]; do
        response=$(curl -s -w "\nHTTP_CODE:%{http_code}" \
            --max-time 300 \
            --connect-timeout 30 \
            "${SERVER_URL}/v1/chat/completions" \
            -H "Content-Type: application/json" \
            -d "$request_json")

        # 提取HTTP状态码
        local http_code=$(echo "$response" | grep "HTTP_CODE:" | sed 's/HTTP_CODE://')
        response=$(echo "$response" | sed '/HTTP_CODE:/d')

        echo "HTTP状态码: $http_code, 响应长度: ${#response}" >> "${LOG_FILE}.debug"

        # 检查是否是503错误（模型加载中）或空响应
        if echo "$response" | grep -q '"code":503'; then
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "模型加载中，等待重试 ($retry/$max_retries)..." | tee -a "$LOG_FILE"
                sleep 5  # 增加等待时间
                continue
            fi
        elif [ -z "$response" ]; then
            retry=$((retry + 1))
            if [ $retry -lt $max_retries ]; then
                echo "收到空响应，等待重试 ($retry/$max_retries)..." | tee -a "$LOG_FILE"
                sleep 3
                continue
            fi
        else
            # 成功获得响应
            echo "$response"
            return 0
        fi
    done

    echo "$response"
    return 1
}

# 提取回复内容
extract_content() {
    local response="$1"
    local content=""

    # 记录原始响应用于调试
    if [ -z "$response" ]; then
        echo "[错误: 收到空响应]"
        echo "原始响应: (empty)" >> "${LOG_FILE}.debug"
        return 1
    fi

    # 优先使用 jq 解析
    if command -v jq >/dev/null 2>&1; then
        content=$(echo "$response" | jq -r '.choices[0].message.content // ""' 2>/dev/null)
        if [ $? -eq 0 ] && [ -n "$content" ] && [ "$content" != "null" ]; then
            echo "$content"
            return 0
        fi
    fi

    # 备用：简单的文本解析
    content=$(echo "$response" | grep -o '"content":"[^"]*"' | sed 's/"content":"//' | sed 's/"$//' | head -1)
    if [ -n "$content" ]; then
        echo "$content"
        return 0
    fi

    echo "[错误: 无法解析响应]"
    echo "原始响应: $response" >> "${LOG_FILE}.debug"
    return 1
}

# 构建消息历史 JSON
build_messages() {
    local result="["
    local first=true

    for ((i=0; i<${#CONVERSATION[@]}; i++)); do
        if [ "$first" = true ]; then
            first=false
        else
            result+=","
        fi

        local role="${CONVERSATION[$i]%%:*}"
        local content="${CONVERSATION[$i]#*:}"
        # 使用 jq 正确转义 JSON 字符串（如果可用）
        if command -v jq >/dev/null 2>&1; then
            # 使用 -R 读取原始输入，-r 输出原始 JSON 字符串
            local escaped_content=$(printf '%s' "$content" | jq -R .)
        else
            # 简单转义：转义双引号和反斜杠，控制字符转为 Unicode
            local escaped_content="\"$(printf '%s' "$content" | sed 's/\\/\\\\/g; s/"/\\"/g; s/$$/\\n/g' | tr -d '\n' | sed 's/\\n$//')\""
        fi
        result+="{\"role\":\"$role\",\"content\":$escaped_content}"
    done

    result+="]"
    echo "$result"
}

# 验证测试结果或更新期望答案
validate_test_result() {
    local test_name="$1"
    local expected_content="$2"
    local actual_response="$3"
    local exact_match="${4:-false}"
    local test_case="${5:-}"
    local round="${6:-}"

    TOTAL_TESTS=$((TOTAL_TESTS + 1))

    local case_result="pass"
    local case_name="${SCRIPT_NAME}_${test_case:-${test_name}}"

    if [ "$UPDATE_MODE" = "true" ]; then
        if [ -n "$actual_response" ] && [ "$actual_response" != "[错误: 无法解析响应]" ]; then
            if [ -n "$test_case" ] && [ -n "$round" ]; then
                update_expected_answer "$test_case" "$round" "$actual_response"
                echo "[UPDATE] $test_name: 已更新期望答案为 '$actual_response'" | tee -a "$LOG_FILE"
            else
                echo "[UPDATE] $test_name: 获得有效回复 '$actual_response'（未指定test_case/round，跳过更新）" | tee -a "$LOG_FILE"
            fi
            PASSED_TESTS=$((PASSED_TESTS + 1))
            echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
            return 0
        else
            echo "[UPDATE_FAIL] $test_name: 未获得有效回复" | tee -a "$LOG_FILE"
            FAILED_TESTS=$((FAILED_TESTS + 1))
            case_result="fail"
            echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
            return 1
        fi
    else
        if [ -n "$test_case" ] && [ -n "$round" ]; then
            expected_content=$(get_expected_answer "$test_case" "$round")
        fi

        if [ -z "$expected_content" ]; then
            if [ -n "$actual_response" ] && [ "$actual_response" != "[错误: 无法解析响应]" ]; then
                echo "[PASS] $test_name: 获得有效回复" | tee -a "$LOG_FILE"
                PASSED_TESTS=$((PASSED_TESTS + 1))
                echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
                return 0
            else
                echo "[FAIL] $test_name: 未获得有效回复" | tee -a "$LOG_FILE"
                FAILED_TESTS=$((FAILED_TESTS + 1))
                case_result="fail"
                echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
                return 1
            fi
        else
            if [ "$exact_match" = "true" ]; then
                if [ "$actual_response" = "$expected_content" ]; then
                    echo "[PASS] $test_name: 回复完全匹配期望内容" | tee -a "$LOG_FILE"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                    echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
                    return 0
                else
                    echo "[FAIL] $test_name: 回复不完全匹配期望内容" | tee -a "$LOG_FILE"
                    echo "[FAIL] 期望: '$expected_content'" | tee -a "$LOG_FILE"
                    echo "[FAIL] 实际: '$actual_response'" | tee -a "$LOG_FILE"
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                    case_result="fail"
                    echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
                    return 1
                fi
            else
                local found=false
                IFS='|' read -ra KEYWORDS <<< "$expected_content"
                for keyword in "${KEYWORDS[@]}"; do
                    if echo "$actual_response" | grep -qi "$keyword"; then
                        found=true
                        break
                    fi
                done

                if [ "$found" = true ]; then
                    echo "[PASS] $test_name: 回复包含期望内容" | tee -a "$LOG_FILE"
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                    echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
                    return 0
                else
                    echo "[FAIL] $test_name: 回复不包含期望内容 (期望: $expected_content)" | tee -a "$LOG_FILE"
                    echo "[FAIL] 实际回复: $actual_response" | tee -a "$LOG_FILE"
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                    case_result="fail"
                    echo "CASE_NAME: ${case_name}, CASE_RESULT: ${case_result}" | tee -a "$LOG_FILE"
                    return 1
                fi
            fi
        fi
    fi
}

# ============ 测试用例 ============

# 测试用例1: 基础多轮对话 (使用 slot 0)
test_case_1() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "测试用例 1: 基础多轮对话（全匹配模式）" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    local SLOT_ID=0  # 使用slot 0
    CONVERSATION=()

    # 第1轮
    echo -e "\n[轮次 1] 用户: 北京是中国的什么？" | tee -a "$LOG_FILE"
    CONVERSATION+=("user:北京是中国的什么？")
    messages=$(build_messages)
    response=$(send_message "$messages" $SLOT_ID)
    assistant_reply=$(extract_content "$response")
    if [ $? -eq 0 ]; then
        echo "[轮次 1] 助手: $assistant_reply" | tee -a "$LOG_FILE"
        CONVERSATION+=("assistant:$assistant_reply")
        # 精确匹配期望回复
        validate_test_result "基础对话-第1轮" "" "$assistant_reply" "true" "test_case_1" "round_1"
    else
        echo "[轮次 1] 错误: 获取回复失败" | tee -a "$LOG_FILE"
        validate_test_result "基础对话-第1轮" "" "" "true" "test_case_1" "round_1"
        return 1
    fi

    # 第2轮 - 测试上下文理解
    echo -e "\n[轮次 2] 用户: 它有多少人口？" | tee -a "$LOG_FILE"
    CONVERSATION+=("user:它有多少人口？")
    messages=$(build_messages)
    response=$(send_message "$messages" $SLOT_ID)
    assistant_reply=$(extract_content "$response")
    if [ $? -eq 0 ]; then
        echo "[轮次 2] 助手: $assistant_reply" | tee -a "$LOG_FILE"
        CONVERSATION+=("assistant:$assistant_reply")
        # 精确匹配期望回复，检查是否理解"它"指的是北京
        validate_test_result "基础对话-第2轮(上下文)" "" "$assistant_reply" "true" "test_case_1" "round_2"
    else
        echo "[轮次 2] 错误: 获取回复失败" | tee -a "$LOG_FILE"
        validate_test_result "基础对话-第2轮(上下文)" "" "" "true" "test_case_1" "round_2"
        return 1
    fi

    echo -e "\n测试用例 1 完成" | tee -a "$LOG_FILE"
}

# 测试用例2: 数学计算连续性 (使用 slot 1)
test_case_2() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "测试用例 2: 数学计算连续性（全匹配模式）" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    local SLOT_ID=1  # 使用slot 1（与测试用例1隔离）
    CONVERSATION=()

    # 第1轮
    echo -e "\n[轮次 1] 用户: 1+1等于几？" | tee -a "$LOG_FILE"
    CONVERSATION+=("user:1+1等于几？")
    messages=$(build_messages)
    response=$(send_message "$messages" $SLOT_ID)
    assistant_reply=$(extract_content "$response")
    echo "[轮次 1] 助手: $assistant_reply" | tee -a "$LOG_FILE"
    CONVERSATION+=("assistant:$assistant_reply")
    # 精确匹配期望回复
    validate_test_result "数学计算-第1轮" "" "$assistant_reply" "true" "test_case_2" "round_1"

    # 第2轮 - 测试上下文中的数学理解
    echo -e "\n[轮次 2] 用户: 那结果乘以3呢？" | tee -a "$LOG_FILE"
    CONVERSATION+=("user:那结果乘以3呢？")
    messages=$(build_messages)
    response=$(send_message "$messages" $SLOT_ID)
    assistant_reply=$(extract_content "$response")
    echo "[轮次 2] 助手: $assistant_reply" | tee -a "$LOG_FILE"
    # 精确匹配期望回复，测试上下文理解
    validate_test_result "数学计算-第2轮(上下文)" "" "$assistant_reply" "true" "test_case_2" "round_2"

    echo -e "\n测试用例 2 完成" | tee -a "$LOG_FILE"
}

# 测试用例3: 长对话稳定性 (使用 slot 2)
test_case_3() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "测试用例 3: 长对话稳定性 (5轮，全匹配模式)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    local SLOT_ID=2  # 使用slot 2（与其他测试用例隔离）
    CONVERSATION=()

    local questions=(
        "你好，我是小明"
        "我刚才说我叫什么？"
        "很好，现在告诉我2+2等于几？"
        "那4+4呢？"
        "最后，你还记得我的名字吗？"
    )

    # 期望回复现在从YAML文件读取

    for i in "${!questions[@]}"; do
        local round=$((i+1))
        echo -e "\n[轮次 $round] 用户: ${questions[$i]}" | tee -a "$LOG_FILE"
        CONVERSATION+=("user:${questions[$i]}")
        messages=$(build_messages)
        response=$(send_message "$messages" $SLOT_ID)
        assistant_reply=$(extract_content "$response")
        echo "[轮次 $round] 助手: $assistant_reply" | tee -a "$LOG_FILE"
        CONVERSATION+=("assistant:$assistant_reply")
        # 使用精确匹配
        validate_test_result "长对话-第${round}轮" "" "$assistant_reply" "true" "test_case_3" "round_$round"
    done

    echo -e "\n测试用例 3 完成" | tee -a "$LOG_FILE"
}

# 打印测试总结
print_test_summary() {
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "多轮对话测试总结" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "总测试数: $TOTAL_TESTS" | tee -a "$LOG_FILE"
    echo "通过: $PASSED_TESTS" | tee -a "$LOG_FILE"
    echo "失败: $FAILED_TESTS" | tee -a "$LOG_FILE"
    if [ $TOTAL_TESTS -gt 0 ]; then
        echo "成功率: $(( PASSED_TESTS * 100 / TOTAL_TESTS ))%" | tee -a "$LOG_FILE"
    else
        echo "成功率: N/A (无测试执行)" | tee -a "$LOG_FILE"
    fi
    echo "========================================" | tee -a "$LOG_FILE"

    if [ $FAILED_TESTS -eq 0 ]; then
        echo "[MULTI_TURN_PASS] 所有多轮对话测试通过!" | tee -a "$LOG_FILE"
        return 0
    else
        echo "[MULTI_TURN_FAIL] $FAILED_TESTS 个测试失败" | tee -a "$LOG_FILE"
        return 1
    fi
}

# ============ 主程序 ============

main() {
    echo "多轮对话单元测试" | tee "$LOG_FILE"
    if [ "$UPDATE_MODE" = "true" ]; then
        echo "模式: 更新期望答案" | tee -a "$LOG_FILE"
    else
        echo "模式: 验证测试" | tee -a "$LOG_FILE"
    fi
    echo "开始时间: $(date)" | tee -a "$LOG_FILE"
    echo "模型路径: $MODEL_PATH" | tee -a "$LOG_FILE"
    echo "服务器: $LLAMA_SERVER" | tee -a "$LOG_FILE"
    echo "端口: $SERVER_PORT" | tee -a "$LOG_FILE"
    echo "日志文件: $LOG_FILE" | tee -a "$LOG_FILE"

    # 检查依赖
    if ! check_dependencies; then
        echo "依赖检查失败，退出测试" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 在验证模式下检查YAML文件是否存在
    if [ "$UPDATE_MODE" != "true" ] && [ ! -f "$EXPECTED_ANSWERS_FILE" ]; then
        echo "错误: 在验证模式下需要期望答案文件: $EXPECTED_ANSWERS_FILE" | tee -a "$LOG_FILE"
        echo "请先运行 --update 选项来生成期望答案文件" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 设置清理函数 - 捕获多种退出信号
    trap stop_server EXIT INT TERM HUP ERR

    # 启动服务器
    if ! start_server; then
        echo "服务器启动失败，退出测试" | tee -a "$LOG_FILE"
        exit 1
    fi

    # 运行测试用例
    test_case_1
    test_case_2
    test_case_3

    # 打印总结并返回结果
    if print_test_summary; then
        echo "完整日志已保存到: $LOG_FILE"
        echo "服务器日志: ${LOG_FILE}.server"
        exit 0
    else
        echo "完整日志已保存到: $LOG_FILE"
        echo "服务器日志: ${LOG_FILE}.server"
        exit 1
    fi
}

# 如果直接运行此脚本，执行主程序
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
    main "$@"
fi
