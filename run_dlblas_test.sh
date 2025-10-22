#!/bin/bash

# DlblasTest 测试脚本
# 功能：下载、解压、编译和测试 DlblasTest
# 作者：自动生成
# 日期：$(date +"%Y-%m-%d %H:%M:%S")

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${BLUE}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

# 检查命令是否存在
check_command() {
    if ! command -v $1 &> /dev/null; then
        log_error "命令 '$1' 未找到，请先安装"
        exit 1
    fi
}

# 清理函数
cleanup() {
    if [ $? -ne 0 ]; then
        log_error "脚本执行失败，正在清理..."
    fi
}

trap cleanup EXIT

# 主函数
main() {
    log_info "开始 DlblasTest 测试流程"

    # 检查必要的命令
    log_info "检查必要的命令..."
    check_command "jf"
    check_command "tar"
    check_command "bash"

    # 设置工作目录
    WORK_DIR=$(pwd)
    DLBLAS_DIR="DlblasTest"
    ARCHIVE_NAME="DlblasTest_20250805.tar.bz2"

    log_info "工作目录: $WORK_DIR"

    # 步骤1: 下载
    log_info "步骤1: 下载 $ARCHIVE_NAME"
    if [ -f "$ARCHIVE_NAME" ]; then
        log_warning "文件 $ARCHIVE_NAME 已存在，跳过下载"
    else
        log_info "正在从 JFrog 下载文件..."
        if jf rt dl "test_dlblas/$ARCHIVE_NAME"; then
            log_success "下载完成"
        else
            log_error "下载失败"
            exit 1
        fi
    fi

    # 步骤2: 解压
    log_info "步骤2: 解压 $ARCHIVE_NAME"
    if [ -d "$DLBLAS_DIR" ]; then
        log_warning "目录 $DLBLAS_DIR 已存在，正在删除旧版本..."
        rm -rf "$DLBLAS_DIR"
    fi

    log_info "正在解压文件..."
    if tar -xvf "$ARCHIVE_NAME"; then
        log_success "解压完成"
    else
        log_error "解压失败"
        exit 1
    fi

    # 检查解压后的目录结构
    if [ ! -d "$DLBLAS_DIR" ]; then
        log_error "解压后未找到 $DLBLAS_DIR 目录"
        exit 1
    fi

    # 步骤3: 编译
    log_info "步骤3: 编译 DlblasTest"
    cd "$DLBLAS_DIR"

    if [ ! -f "compile_dlblas_test.sh" ]; then
        log_error "未找到编译脚本 compile_dlblas_test.sh"
        exit 1
    fi

    log_info "正在执行编译脚本..."
    if bash compile_dlblas_test.sh; then
        log_success "编译完成"
    else
        log_error "编译失败"
        exit 1
    fi

    # 步骤4: 测试
    log_info "步骤4: 运行测试"
    TEST_BINARY="./bin/apply_test/debug/dlblasGemmExV2_debug"

    if [ ! -f "$TEST_BINARY" ]; then
        log_error "未找到测试可执行文件: $TEST_BINARY"
        exit 1
    fi

    if [ ! -x "$TEST_BINARY" ]; then
        log_warning "测试文件不可执行，正在添加执行权限..."
        chmod +x "$TEST_BINARY"
    fi

    log_info "正在运行测试: $TEST_BINARY"
    echo "=========================================="
    echo "测试输出:"
    echo "=========================================="

    if $TEST_BINARY; then
        echo "=========================================="
        log_success "测试执行完成"
    else
        echo "=========================================="
        log_error "测试执行失败"
        exit 1
    fi

    # 返回原始工作目录
    cd "$WORK_DIR"

    log_success "DlblasTest 测试流程全部完成！"
}

# 显示帮助信息
show_help() {
    echo "DlblasTest 测试脚本"
    echo ""
    echo "用法: $0 [选项]"
    echo ""
    echo "选项:"
    echo "  -h, --help     显示此帮助信息"
    echo "  -c, --clean    清理下载和解压的文件"
    echo "  -s, --skip-download  跳过下载步骤（假设文件已存在）"
    echo ""
    echo "功能:"
    echo "  1. 下载: jf rt dl test_dlblas/DlblasTest_20250805.tar.bz2"
    echo "  2. 解压: tar -xvf DlblasTest_20250805.tar.bz2"
    echo "  3. 编译: bash compile_dlblas_test.sh"
    echo "  4. 测试: ./bin/apply_test/debug/dlblasGemmExV2_debug"
}

# 清理函数
clean_files() {
    log_info "正在清理文件..."

    if [ -f "DlblasTest_20250805.tar.bz2" ]; then
        rm -f "DlblasTest_20250805.tar.bz2"
        log_success "已删除 DlblasTest_20250805.tar.bz2"
    fi

    if [ -d "DlblasTest" ]; then
        rm -rf "DlblasTest"
        log_success "已删除 DlblasTest 目录"
    fi

    log_success "清理完成"
}

# 跳过下载的主函数
main_skip_download() {
    log_info "开始 DlblasTest 测试流程（跳过下载）"

    # 检查必要的命令
    log_info "检查必要的命令..."
    check_command "tar"
    check_command "bash"

    # 设置工作目录
    WORK_DIR=$(pwd)
    DLBLAS_DIR="DlblasTest"
    ARCHIVE_NAME="DlblasTest_20250805.tar.bz2"

    log_info "工作目录: $WORK_DIR"

    # 检查文件是否存在
    if [ ! -f "$ARCHIVE_NAME" ]; then
        log_error "文件 $ARCHIVE_NAME 不存在，无法跳过下载"
        exit 1
    fi

    # 步骤2: 解压
    log_info "步骤2: 解压 $ARCHIVE_NAME"
    if [ -d "$DLBLAS_DIR" ]; then
        log_warning "目录 $DLBLAS_DIR 已存在，正在删除旧版本..."
        rm -rf "$DLBLAS_DIR"
    fi

    log_info "正在解压文件..."
    if tar -xvf "$ARCHIVE_NAME"; then
        log_success "解压完成"
    else
        log_error "解压失败"
        exit 1
    fi

    # 检查解压后的目录结构
    if [ ! -d "$DLBLAS_DIR" ]; then
        log_error "解压后未找到 $DLBLAS_DIR 目录"
        exit 1
    fi

    # 步骤3: 编译
    log_info "步骤3: 编译 DlblasTest"
    cd "$DLBLAS_DIR"

    if [ ! -f "compile_dlblas_test.sh" ]; then
        log_error "未找到编译脚本 compile_dlblas_test.sh"
        exit 1
    fi

    log_info "正在执行编译脚本..."
    if bash compile_dlblas_test.sh; then
        log_success "编译完成"
    else
        log_error "编译失败"
        exit 1
    fi

    # 步骤4: 测试
    log_info "步骤4: 运行测试"
    TEST_BINARY="./bin/apply_test/debug/dlblasGemmExV2_debug"

    if [ ! -f "$TEST_BINARY" ]; then
        log_error "未找到测试可执行文件: $TEST_BINARY"
        exit 1
    fi

    if [ ! -x "$TEST_BINARY" ]; then
        log_warning "测试文件不可执行，正在添加执行权限..."
        chmod +x "$TEST_BINARY"
    fi

    log_info "正在运行测试: $TEST_BINARY"
    echo "=========================================="
    echo "测试输出:"
    echo "=========================================="

    if $TEST_BINARY; then
        echo "=========================================="
        log_success "测试执行完成"
    else
        echo "=========================================="
        log_error "测试执行失败"
        exit 1
    fi

    # 返回原始工作目录
    cd "$WORK_DIR"

    log_success "DlblasTest 测试流程全部完成！"
}

# 解析命令行参数
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    -c|--clean)
        clean_files
        exit 0
        ;;
    -s|--skip-download)
        main_skip_download
        exit 0
        ;;
    "")
        main
        exit 0
        ;;
    *)
        log_error "未知选项: $1"
        show_help
        exit 1
        ;;
esac
