#!/bin/bash

# llama.cpp integrated compilation and testing script
# Supports x86_64, aarch64, loongarch64 platforms
# Optimized for LoongArch64 platform, automatically handles libgomp dependency issues

set -e

# Script version
VERSION="1.0.0"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Display banner
show_banner() {
    echo -e "${BLUE}"
    cat << 'EOF'
╔══════════════════════════════════════════════════════════════╗
║                    llama.cpp Build Tool                      ║
║               Cross-platform Compilation and Testing         ║
╚══════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
    echo -e "${CYAN}Version: $VERSION${NC}"
    echo -e "${CYAN}Supported platforms: x86_64, aarch64, loongarch64${NC}"
    echo ""
}

# Display usage instructions
show_usage() {
    cat << EOF
llama.cpp integrated compilation and testing script

Usage: $0 [options]

Options:
  --platform PLATFORM     Target platform (x86_64, aarch64, loongarch64)
  --action ACTION         Action to execute (compile, test, all)
  --sdk-path PATH         SDK path (or use environment variable sdk_path)
  --repo-path PATH        Repository path (or use environment variable REPO_PATH)
  --model-path PATH       Model path (or use environment variable LOCAL_MODEL_PATH)
  --no-auto-install       Disable automatic installation of missing dependencies
  --interactive           Interactive setup
  --debug-mode            Enable debug mode with verbose output
  --simple-test           Run simplified CPU-only tests for troubleshooting
  --skip-device-check     Skip device card status check (use with caution)
                          Device check uses standalone script: .dlci/check_device_status.sh
  --help, -h              Show help information

Action descriptions:
  compile                 Compile llama.cpp only
  test                    Run tests only
  all                     Compile + Test (default)

Environment variables:
  sdk_path                SDK path
  REPO_PATH              Repository path (default: current directory)
  LOCAL_MODEL_PATH       Model path (default: /mars/aebox/LLM/model)

Examples:
  # Auto-detect platform, full process
  $0

  # Specify LoongArch64 platform compilation
  $0 --platform loongarch64 --action compile --sdk-path /opt/sdk

  # Interactive setup
  $0 --interactive

Quick start:
  export sdk_path="/path/to/your/sdk"
  $0

EOF
}

# Auto-detect platform
detect_platform() {
    case "$(uname -m)" in
        x86_64|amd64) echo "x86_64" ;;
        aarch64|arm64) echo "aarch64" ;;
        loongarch64) echo "loongarch64" ;;
        *)
            log_error "Unsupported architecture: $(uname -m)"
            exit 1
            ;;
    esac
}

# Platform-specific configuration
configure_platform() {
    local platform="$1"

    case "$platform" in
        x86_64)
            log_info "Configuring x86_64 platform parameters"
            CMAKE_ARGS="-DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF"
            TEST_TIMEOUT=600
            ENABLE_FULL_TESTS=true
            ;;
        aarch64)
            log_info "Configuring aarch64 platform parameters"
            CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=OFF -DGGML_RVV=OFF"
            TEST_TIMEOUT=600
            ENABLE_FULL_TESTS=true
            export HC_CE_DISPATCH_MODE=1
            ;;
        loongarch64)
            log_info "Configuring loongarch64 platform parameters"
            CMAKE_ARGS="-DGGML_CPU_ALL_VARIANTS=OFF -DGGML_RVV=OFF"
            TEST_TIMEOUT=600
            ENABLE_FULL_TESTS=true
            ;;
        *)
            log_error "Unknown platform: $platform"
            exit 1
            ;;
    esac
}

# Interactive environment setup
interactive_setup() {
    echo -e "${CYAN}=== Interactive Environment Setup ===${NC}"

    # SDK path setup
    if [ -z "$sdk_path" ]; then
        echo -e "${YELLOW}Please set SDK path (required):${NC}"
        printf "SDK path: "
        if read sdk_input && [ -n "$sdk_input" ] && [ -d "$sdk_input" ]; then
            export sdk_path="$sdk_input"
            log_success "SDK path set: $sdk_path"
        else
            log_error "Invalid SDK path"
            return 1
        fi
    else
        log_success "SDK path already set: $sdk_path"
    fi

    # Repository path setup
    if [ -z "$REPO_PATH" ]; then
        local current_dir=$(pwd)
        echo -e "${YELLOW}Repository path (current: $current_dir):${NC}"
        printf "Repository path [press Enter to use current directory]: "
        if read repo_input; then
            if [ -n "$repo_input" ]; then
                export REPO_PATH="$repo_input"
            else
                export REPO_PATH="$current_dir"
            fi
            log_success "Repository path: $REPO_PATH"
        fi
    else
        log_success "Repository path already set: $REPO_PATH"
    fi

    # Platform selection
    echo -e "${CYAN}Select target platform:${NC}"
    echo "1) Auto-detect current platform"
    echo "2) x86_64 (Intel/AMD 64-bit)"
    echo "3) aarch64 (ARM 64-bit)"
    echo "4) loongarch64 (LoongArch 64-bit)"

    printf "Please select (1-4): "
    if read choice; then
        case "$choice" in
            1) INTERACTIVE_PLATFORM=$(detect_platform) ;;
            2) INTERACTIVE_PLATFORM="x86_64" ;;
            3) INTERACTIVE_PLATFORM="aarch64" ;;
            4) INTERACTIVE_PLATFORM="loongarch64" ;;
            *) log_error "Invalid selection"; return 1 ;;
        esac
        log_info "Selected platform: $INTERACTIVE_PLATFORM"
    fi

    # Action selection
    echo -e "${CYAN}Select action to execute:${NC}"
    echo "1) Full process (compile + test)"
    echo "2) Compile only"
    echo "3) Test only"

    printf "Please select (1-3): "
    if read action_choice; then
        case "$action_choice" in
            1) INTERACTIVE_ACTION="all" ;;
            2) INTERACTIVE_ACTION="compile" ;;
            3) INTERACTIVE_ACTION="test" ;;
            *) log_error "Invalid selection"; return 1 ;;
        esac
        log_info "Selected action: $INTERACTIVE_ACTION"
    fi

    return 0
}

# Check necessary tools and libraries
check_tools() {
    local auto_install_enabled="${1:-true}"
    local missing_tools=()
    local missing_packages=()
    local tools=("cmake" "ninja" "git" "gcc" "g++")

    # Check basic tools
    for tool in "${tools[@]}"; do
        if ! command -v "$tool" >/dev/null 2>&1; then
            missing_tools+=("$tool")
        fi
    done

    if [ ${#missing_tools[@]} -ne 0 ]; then
        log_error "Missing required tools:"
        for tool in "${missing_tools[@]}"; do
            echo "  - $tool"
        done
        return 1
    fi

    # Check platform-specific library dependencies
    local platform=$(uname -m)
    case "$platform" in
        loongarch64)
            log_info "Checking LoongArch64 platform-specific dependencies..."

            # Check libgomp.so (OpenMP library)
            local libgomp_paths=(
                "/usr/lib/gcc/loongarch64-openEuler-linux/12/libgomp.so"
                "/usr/lib/gcc/loongarch64-OpenCloudOS-linux/12/libgomp.so"
                "/usr/lib/gcc/loongarch64-*/libgomp.so"
                "/usr/lib/loongarch64-*/libgomp.so"
                "/usr/lib/libgomp.so"
                "/lib/libgomp.so"
            )

            local libgomp_found=false
            local actual_path=""
            for path in "${libgomp_paths[@]}"; do
                if ls $path >/dev/null 2>&1; then
                    log_info "Found libgomp: $path"
                    libgomp_found=true
                    actual_path="$path"
                    break
                fi
            done

            if [ "$libgomp_found" = false ]; then
                log_error "Missing OpenMP library (libgomp), which is required for LoongArch64 platform compilation"
                echo "Please install one of the following packages:"
                echo "  - OpenEuler/CentOS: dnf install libgomp-devel gcc-gfortran"
                echo "  - Ubuntu/Debian: apt install libgomp1-dev"
                echo "  - LoongArch64: yum install libomp-devel.loongarch64"

                # Try automatic installation
                if [ "$auto_install_enabled" = "true" ]; then
                    if command -v dnf >/dev/null 2>&1 && ([ "$EUID" -eq 0 ] || sudo -n true 2>/dev/null); then
                        log_info "Attempting to automatically install libgomp-devel..."
                        if sudo dnf install -y libgomp-devel gcc-gfortran; then
                            log_success "libgomp installation successful"
                        else
                            missing_packages+=("libgomp")
                        fi
                    else
                        missing_packages+=("libgomp")
                    fi
                else
                    missing_packages+=("libgomp")
                fi
            else
                # Check path compatibility
                local expected_path="/usr/lib/gcc/loongarch64-OpenCloudOS-linux/12/libgomp.so"
                if [ "$actual_path" != "$expected_path" ] && [ ! -e "$expected_path" ]; then
                    log_info "Creating libgomp compatibility symbolic link..."
                    local expected_dir=$(dirname "$expected_path")
                    sudo mkdir -p "$expected_dir" 2>/dev/null || true
                    sudo ln -sf "$actual_path" "$expected_path" 2>/dev/null || {
                        log_warn "Unable to create symbolic link, execute manually: sudo ln -sf $actual_path $expected_path"
                    }
                fi
            fi
            ;;
        aarch64)
            if ! ldconfig -p | grep -q "libgomp"; then
                log_warn "libgomp not found, may need to install: apt install libgomp1 or dnf install libgomp"
            fi
            ;;
    esac

    if [ ${#missing_packages[@]} -ne 0 ]; then
        log_error "Missing required library files, please install them before running compilation"
        return 1
    fi

    log_success "Required tools and libraries check passed"
    return 0
}

# Setup SDK environment
setup_sdk() {
    local sdk_path_arg="$1"

    if [ -z "$sdk_path_arg" ]; then
        log_error "SDK path not set"
        return 1
    fi

    if [ ! -d "$sdk_path_arg" ]; then
        log_error "SDK path does not exist: $sdk_path_arg"
        return 1
    fi

    if [ ! -f "$sdk_path_arg/env.sh" ]; then
        log_error "SDK environment script does not exist: $sdk_path_arg/env.sh"
        return 1
    fi

    log_info "Configuring SDK environment: $sdk_path_arg"
    source "$sdk_path_arg/env.sh"

    log_success "SDK environment configuration completed"
    return 0
}

# Configure ccache
setup_ccache() {
    if ! command -v ccache >/dev/null 2>&1; then
        log_warn "ccache not installed, skipping ccache configuration"
        return 0
    fi

    log_info "Configuring ccache..."

    local ccache_dir="/LocalRun/$(whoami)/cache/llama_cpp_ccache"
    ccache --set-config cache_dir="$ccache_dir"
    ccache --set-config max_size=20G
    ccache --zero-stats

    export CCACHE_LOGFILE="/LocalRun/$(whoami)/cache/ccache.log"

    # Create custom bin directory for ccache wrappers
    local custom_bin_dir="$sdk_path/custom_bin"
    mkdir -p "$custom_bin_dir"

    # Create ccache symbolic links
    ln -sf /usr/bin/ccache "$custom_bin_dir/dlcc"
    export PATH="$custom_bin_dir:$PATH"
    export CUDA_NVCC_EXECUTABLE="$custom_bin_dir/dlcc"

    log_success "ccache configuration completed"
}

# Get version information
get_version_info() {
    local tag commit sdk_num base_version denglin_version

    tag=$(git describe --tags --exact 2>/dev/null || true)
    commit=$(git rev-parse --short=8 HEAD)
    sdk_num=$(echo "${SDK_TAG:-}" | grep -oE '[0-9]{12}' || echo "000000000000")

    if [[ -n "$tag" ]]; then
        base_version=$(echo "$tag" | sed -E 's/^v([0-9.]+\.dev[0-9]+).*/\1/')
        denglin_version=$(echo "$tag" | grep -oE '\+.*' | sed 's/^+//' | sed 's/[-_]//g')
        export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
    else
        base_version=$(cat version.txt 2>/dev/null | sed 's/[^0-9.].*$//' || echo "0.0.0")
        denglin_version="git${commit}"
        export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
    fi

    log_info "Build version: $LLAMA_CPP_BUILD_VERSION"
}

# Check device card status and handle failures
# This function now delegates to the standalone device check script for consistency
check_device_status() {
    local device_check_script="${REPO_PATH}/.dlci/check_device_status.sh"

    # Check if the standalone device check script exists
    if [ -f "$device_check_script" ] && [ -x "$device_check_script" ]; then
        log_info "Running device status check using standalone script..."

        # Set DOCKER_PLATFORM for the script
        export DOCKER_PLATFORM="${platform:-$(uname -m)}"

        # Run the standalone device check script
        if bash "$device_check_script"; then
            log_success "Device status check completed successfully"
            return 0
        else
            log_error "Device status check failed"
            return 1
        fi
    fi
}

# LoongArch64 specific test preparation
prepare_loongarch64_test() {
    log_info "Preparing LoongArch64 test environment..."

    # Set conservative memory limits for LoongArch64
    export GGML_CUDA_MALLOC_SOFT_LIMIT=1073741824  # 1GB
    export GGML_CUDA_HOST_MALLOC_LIMIT=2147483648  # 2GB

    log_info "LoongArch64 test environment ready"
}

# Compile llama.cpp
compile_llama_cpp() {
    local platform="$1"
    local build_dir="$2"

    log_info "Starting llama.cpp compilation (platform: $platform)"
    local compile_start_time=$(date +%s)

    # Ensure we're in the correct directory
    if [ -z "$REPO_PATH" ]; then
        export REPO_PATH=$(pwd)
    fi
    cd "$REPO_PATH"

    # Create build directory
    mkdir -p "$build_dir"

    # Build CMake arguments
    local cmake_common_args=(
        -G Ninja
        -B "$build_dir"
        -DGGML_DLCU=ON
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -DCMAKE_BUILD_TYPE=Release
        -DGGML_BACKEND_DL=ON
        -DGGML_CUDA_GRAPHS=OFF
        -DLLAMA_CURL=OFF
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DGGML_CUDA_FA=ON
        -DGGML_CUDA_FA_ALL_QUANTS=ON
        -DSDK_DIR="$sdk_path"
    )

    # Add platform-specific arguments
    IFS=' ' read -ra platform_args <<< "$CMAKE_ARGS"
    cmake_common_args+=("${platform_args[@]}")

    # LoongArch64 special handling: set correct OpenMP library path
    if [ "$platform" = "loongarch64" ]; then
        local actual_libgomp=""
        local libgomp_search_paths=(
            "/usr/lib/gcc/loongarch64-openEuler-linux/12/libgomp.so"
            "/usr/lib/gcc/loongarch64-OpenCloudOS-linux/12/libgomp.so"
            "/usr/lib/gcc/loongarch64-*/libgomp.so"
            "/usr/lib/loongarch64-*/libgomp.so"
            "/usr/lib/libgomp.so"
            "/lib/libgomp.so"
        )

        # Find actual libgomp path
        for path in "${libgomp_search_paths[@]}"; do
            if ls $path >/dev/null 2>&1; then
                actual_libgomp="$path"
                log_info "Using libgomp for compilation: $actual_libgomp"
                break
            fi
        done

        if [ -n "$actual_libgomp" ]; then
            # Set OpenMP-related CMake variables
            cmake_common_args+=(
                "-DOpenMP_gomp_LIBRARY=$actual_libgomp"
                "-DOpenMP_CXX_LIB_NAMES=gomp"
                "-DOpenMP_C_LIB_NAMES=gomp"
            )
            export OpenMP_gomp_LIBRARY="$actual_libgomp"
        fi
    fi

    log_info "CMake arguments: ${cmake_common_args[*]}"

    # Run CMake configuration
    cmake "${cmake_common_args[@]}"

    # Compile
    cd "$build_dir"
    ninja -j $(nproc)

    # Calculate compilation time
    local compile_end_time=$(date +%s)
    local compile_duration=$((compile_end_time - compile_start_time))

    log_success "Compilation completed, time taken: ${compile_duration} seconds"
}

# Run tests
run_tests() {
    local platform="$1"
    local build_dir="$2"
    local debug_mode="${3:-false}"
    local simple_test="${4:-false}"

    log_info "Starting test execution (platform: $platform)"
    local test_start_time=$(date +%s)

    # Set test environment variables
    export GGML_TEST_MODE=1
    export GGML_DEBUG=1

    # Platform-specific environment variables
    if [ "$platform" = "loongarch64" ]; then
        # LoongArch64 specific settings to avoid timeout issues
        export GGML_CUDA_DEVICE_TIMEOUT=30
        export GGML_CUDA_FORCE_MMQ=1
        export GGML_CUDA_PEER_MAX_BATCH_SIZE=128
        export DENGLIN_TIMEOUT=60
        log_info "Applied LoongArch64 specific CUDA settings"
        prepare_loongarch64_test
    fi

    # Apply debug mode settings
    if [ "$debug_mode" = "true" ]; then
        export GGML_DEBUG=2
        export GGML_VERBOSE=1
        export CUDA_LAUNCH_BLOCKING=1
        log_info "Enabled debug mode"
    fi

    export CUDA_VISIBLE_DEVICES=0

    # Set up environment variables for the comprehensive test script
    export DOCKER_PLATFORM="$platform"
    export REPO_PATH="${REPO_PATH}"

    # Check if the comprehensive test script exists
    local test_script="${REPO_PATH}/.dlci/test_llama_cpp.sh"

    if [ ! -f "$test_script" ]; then
        log_error "Comprehensive test script not found: $test_script"
        log_error "Please ensure the test script exists in .dlci/test_llama_cpp.sh"
        return 1
    fi

    if [ ! -x "$test_script" ]; then
        log_info "Making test script executable..."
        chmod +x "$test_script"
    fi

    log_info "Running comprehensive test suite using: $test_script"
    log_info "This includes Qwen model correctness validation and full llama.cpp tests"

    # Determine test mode based on simple_test flag
    if [ "$simple_test" = "true" ]; then
        log_info "Simple test mode requested - will run basic tests only"
        # We could add a flag to test_llama_cpp.sh for simple mode, but for now run full suite
    fi

    # Run the comprehensive test script
    local ret=0
    local start_time=$(date +%s)

    if [ "$platform" = "loongarch64" ]; then
        # LoongArch64 specific handling with timeout monitoring
        log_info "Running comprehensive test suite with LoongArch64 timeout monitoring..."

        # Run the test script in background
        bash "$test_script" &
        local test_pid=$!

        # Monitor for timeout with periodic status
        local elapsed=0
        local check_interval=60  # Check every minute for comprehensive tests
        local comprehensive_timeout=$((TEST_TIMEOUT * 3))  # Longer timeout for comprehensive tests

        while [ $elapsed -lt $comprehensive_timeout ]; do
            if ! kill -0 $test_pid 2>/dev/null; then
                # Process finished
                wait $test_pid
                ret=$?
                break
            fi

            sleep $check_interval
            elapsed=$((elapsed + check_interval))

            if [ $((elapsed % 300)) -eq 0 ]; then  # Report every 5 minutes
                log_info "Comprehensive test still running... elapsed: ${elapsed}s / ${comprehensive_timeout}s"
            fi
        done

        # If we reach here and process is still running, it's a timeout
        if kill -0 $test_pid 2>/dev/null; then
            log_warn "Comprehensive test timeout reached (${comprehensive_timeout}s), terminating process..."
            kill -TERM $test_pid 2>/dev/null
            sleep 10
            kill -KILL $test_pid 2>/dev/null
            ret=124  # timeout exit code
        fi
    else
        # Normal execution for other platforms
        log_info "Executing comprehensive test suite..."
        bash "$test_script"
        ret=$?
    fi

    local end_time=$(date +%s)
    local test_duration=$((end_time - start_time))

    # Format duration for better readability
    local duration_str=""
    if [ $test_duration -ge 3600 ]; then
        local hours=$((test_duration / 3600))
        local minutes=$(((test_duration % 3600) / 60))
        local seconds=$((test_duration % 60))
        duration_str="${hours}h ${minutes}m ${seconds}s"
    elif [ $test_duration -ge 60 ]; then
        local minutes=$((test_duration / 60))
        local seconds=$((test_duration % 60))
        duration_str="${minutes}m ${seconds}s"
    else
        duration_str="${test_duration}s"
    fi

    # Report results
    if [ $ret -eq 0 ]; then
        log_success "Comprehensive test suite completed successfully!"
        log_success "Total execution time: $duration_str"
        log_info "This includes:"
        log_info "  - Full llama.cpp backend and functionality tests"
        log_info "  - Qwen2, Qwen2.5, and Qwen3 model correctness validation"
        log_info "  - Content-based output verification with reference comparison"
        return 0
    elif [ $ret -eq 124 ]; then
        log_error "Comprehensive test suite timed out after $duration_str"
        log_error "Consider running with --simple-test for faster execution"
        return 1
    else
        log_error "Comprehensive test suite failed (exit code: $ret)"
        log_error "Execution time: $duration_str"
        log_error "Check the detailed logs in the test script output above"
        return 1
    fi
}

# Main function
main() {
    local platform=""
    local action="all"
    local sdk_path_arg=""
    local repo_path_arg=""
    local model_path_arg=""
    local auto_install=true
    local interactive=false
    local debug_mode=false
    local simple_test=false
    local skip_device_check=false

    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --platform)
                platform="$2"
                shift 2
                ;;
            --action)
                action="$2"
                shift 2
                ;;
            --sdk-path)
                sdk_path_arg="$2"
                shift 2
                ;;
            --repo-path)
                repo_path_arg="$2"
                shift 2
                ;;
            --model-path)
                model_path_arg="$2"
                shift 2
                ;;
            --no-auto-install)
                auto_install=false
                shift
                ;;
            --interactive)
                interactive=true
                shift
                ;;
            --debug-mode)
                debug_mode=true
                shift
                ;;
            --simple-test)
                simple_test=true
                shift
                ;;
            --skip-device-check)
                skip_device_check=true
                shift
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown parameter: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Display banner
    show_banner

    # Interactive setup
    if [ "$interactive" = "true" ]; then
        if ! interactive_setup; then
            exit 1
        fi
        # Use interactive setup values
        if [ -n "$INTERACTIVE_PLATFORM" ]; then
            platform="$INTERACTIVE_PLATFORM"
        fi
        if [ -n "$INTERACTIVE_ACTION" ]; then
            action="$INTERACTIVE_ACTION"
        fi
    fi

    # Set default platform
    if [ -z "$platform" ]; then
        platform=$(detect_platform)
        log_info "Auto-detected platform: $platform"
    fi

    # Set environment variables
    if [ -n "$sdk_path_arg" ]; then
        export sdk_path="$sdk_path_arg"
    fi

    if [ -n "$repo_path_arg" ]; then
        export REPO_PATH="$repo_path_arg"
    elif [ -z "$REPO_PATH" ]; then
        export REPO_PATH="$(pwd)"
    fi

    if [ -n "$model_path_arg" ]; then
        export LOCAL_MODEL_PATH="$model_path_arg"
    elif [ -z "$LOCAL_MODEL_PATH" ]; then
        export LOCAL_MODEL_PATH="/mars/aebox/LLM/model"
        log_info "Using default LOCAL_MODEL_PATH: $LOCAL_MODEL_PATH"
    fi

    # Validate required parameters
    if [ -z "$sdk_path" ]; then
        log_error "SDK path not set, please use --sdk-path or set environment variable sdk_path"
        echo "Quick setup: export sdk_path=\"/path/to/your/sdk\""
        exit 1
    fi

    # Display configuration information
    log_info "=== Execution Configuration ==="
    log_info "Platform: $platform"
    log_info "Action: $action"
    log_info "SDK path: $sdk_path"
    log_info "Repository path: $REPO_PATH"
    log_info "Model path: $LOCAL_MODEL_PATH"
    if [ "$debug_mode" = "true" ]; then
        log_info "Debug mode: enabled"
    fi
    if [ "$simple_test" = "true" ]; then
        log_info "Simple test mode: enabled"
    fi
    if [ "$skip_device_check" = "true" ]; then
        log_warn "Device check: disabled (--skip-device-check)"
    fi
    echo ""

    # Configure platform parameters
    configure_platform "$platform"

    # Set build directory
    local build_dir="${REPO_PATH}/build_${platform}"

    # Execute corresponding action
    case "$action" in
        compile|all)
            log_info "=== Compilation Phase ==="

            # Check device status before compilation (critical for CUDA builds)
            if [ "$skip_device_check" = "false" ]; then
                check_device_status
            else
                log_warn "Device status check skipped by user request"
            fi

            check_tools "$auto_install"
            setup_sdk "$sdk_path"
            setup_ccache "$platform"
            get_version_info
            compile_llama_cpp "$platform" "$build_dir"

            if [ "$action" = "compile" ]; then
                log_success "Compilation completed"
                echo "Build artifacts: $build_dir/bin/"
                exit 0
            fi
            ;&
        test)
            if [ "$action" = "test" ] || [ "$action" = "all" ]; then
                log_info "=== Testing Phase ==="

                # Check device status before testing (critical for CUDA tests)
                if [ "$skip_device_check" = "false" ]; then
                    check_device_status
                else
                    log_warn "Device status check skipped by user request"
                fi

                run_tests "$platform" "$build_dir" "$debug_mode" "$simple_test"

                if [ "$action" = "test" ]; then
                    log_success "Testing completed"
                    exit 0
                fi
            fi
            ;;
        *)
            log_error "Unknown action: $action"
            exit 1
            ;;
    esac

    log_success "All steps completed! Platform: $platform"
    echo "Build artifacts: $build_dir/bin/"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
