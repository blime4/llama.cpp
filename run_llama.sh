#!/bin/bash

# llama.cpp integrated compilation and testing script
# Supports x86_64, aarch64, loongarch64 platforms
# Optimized for LoongArch64 platform, automatically handles libgomp dependency issues

set -e

# Source unified utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/.dlci/utils.sh"

# Script version
VERSION="1.0.0"

# Color definitions
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Global log file
MAIN_LOG_FILE=""

# Initialize logging
init_logging() {
    MAIN_LOG_FILE="${REPO_PATH}/run_llama_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "=== llama.cpp Build and Test Log ==="
        echo "Start Time: $(date)"
        echo "Platform: ${platform:-auto-detect}"
        echo "SDK Path: ${SDK_DIR:-not set}"
        echo "========================================"
    } > "$MAIN_LOG_FILE"
}

# Unified logging functions
log_info() {
    local msg="[INFO] $1"
    echo -e "${BLUE}${msg}${NC}"
    [ -n "$MAIN_LOG_FILE" ] && echo "$msg" >> "$MAIN_LOG_FILE"
}

log_error() {
    local msg="[ERROR] $1"
    echo -e "${RED}${msg}${NC}"
    [ -n "$MAIN_LOG_FILE" ] && echo "$msg" >> "$MAIN_LOG_FILE"
}

log_success() {
    local msg="[SUCCESS] $1"
    echo -e "${GREEN}${msg}${NC}"
    [ -n "$MAIN_LOG_FILE" ] && echo "$msg" >> "$MAIN_LOG_FILE"
}

log_warn() {
    local msg="[WARN] $1"
    echo -e "${YELLOW}${msg}${NC}"
    [ -n "$MAIN_LOG_FILE" ] && echo "$msg" >> "$MAIN_LOG_FILE"
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
    echo -e "${CYAN}Supported platforms: x86_64, aarch64, loongarch64, android${NC}"
    echo ""
}

# Display usage instructions
show_usage() {
    cat << EOF
llama.cpp integrated compilation and testing script

Usage: $0 [options]

Options:
  --platform PLATFORM     Target platform (x86_64, aarch64, loongarch64, android)
  --action ACTION         Action to execute (compile, test, all)
  -c                      Shortcut for "--action compile"
  -t                      Shortcut for "--action test"
  -cc                     Clean build_* directories, then run "--action compile"
  -ccc                    Clean build_* directories and ccache, then run "--action compile"
  --sdk-path PATH         SDK path (or use environment variable SDK_DIR)
  --repo-path PATH        Repository path (or use environment variable REPO_PATH)
  --model-path PATH       Model path (or use environment variable LOCAL_MODEL_PATH)
  --no-auto-install       Disable automatic installation of missing dependencies
  --interactive           Interactive setup
  --debug                 Enable full debug mode (Debug build + verbose runtime output)
  --release               Disable debug mode (Release build) [DEFAULT]
                          Or use --debug=0 to disable debug mode (Release build)
  --simple-test, -st      Run only backend-ops tests (MUL_MAT and DLFA tests)
  --simple-model, -sm     Run only Qwen2.5 model tests with GGML_DLFA_READY=1
  --simple-tp             Run tensor parallel tests with split-mode variations
  --simple-perf           Run performance comparison tests (Pool vs Legacy modes)
  --simple-bench, -sb     Run llama-bench performance test with Qwen3-30B model
  --big                   Use big model for testing (e.g., Qwen3-30B instead of Qwen2.5-1.5B)
  --repeat-test N         Repeat test execution N times and collect statistics
  --skip-device-check     Skip device card status check (use with caution)
  --no-fa                 Disable Flash Attention (do not use -fa flag)
  --dlpti "OPTIONS"       Enable dlPTI profiling with specified options
                          Example: --dlpti "--activity-mask cmd,cu,curt --data-file profile.db"
                          See dlpti_tools capture --help for available options
  --help, -h              Show help information

Action descriptions:
  compile                 Compile llama.cpp only
  test                    Run tests only
  all                     Compile + Test (default)

Environment variables:
  SDK_DIR                SDK path
  REPO_PATH              Repository path (default: current directory)
  LOCAL_MODEL_PATH       Model path (default: from config.yml)

Examples:
  # Auto-detect platform, full process (release mode by default)
  $0

  # Specify LoongArch64 platform compilation in release mode
  $0 --platform loongarch64 --action compile --sdk-path /opt/sdk --release

  # Android cross-compilation
  $0 --platform android --action compile --sdk-path /opt/sdk

  # Interactive setup
  $0 --interactive

  # Enable dlPTI profiling with simple model test
  $0 --simple-model --dlpti "--activity-mask cmd,cu,curt --data-file llama_profile.db"

  # Enable dlPTI profiling with big model and TP test
  $0 --simple-tp --big --dlpti "--activity-mask cmd,cu,curt,nne --data-file tp_profile_{datetime}.db"

Quick start:
  export SDK_DIR="/path/to/your/sdk"
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
            log_error "Note: For Android cross-compilation, use --platform android"
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
            ;;
        aarch64)
            log_info "Configuring aarch64 platform parameters"
            # CMAKE_ARGS="-DGGML_CPU_ARM_ARCH=armv8-a -DGGML_NATIVE=OFF -DGGML_RVV=OFF"
            CMAKE_ARGS="-DGGML_CPU_ALL_VARIANTS=ON -DGGML_NATIVE=OFF"
            TEST_TIMEOUT=600
            export HC_CE_DISPATCH_MODE=1
            ;;
        loongarch64)
            log_info "Configuring loongarch64 platform parameters"
            log_info "GGML_CPU_ALL_VARIANTS is disabled for loongarch64 platform"
            CMAKE_ARGS="-DGGML_CPU_ALL_VARIANTS=OFF -DGGML_RVV=OFF -DGGML_NATIVE=OFF"
            TEST_TIMEOUT=600
            ;;
        android)
            log_info "Configuring Android platform parameters (cross-compilation)"
            CMAKE_ARGS="-DGGML_CPU_ALL_VARIANTS=ON -DGGML_RVV=OFF -DGGML_NATIVE=OFF"
            TEST_TIMEOUT=300  # Shorter timeout for cross-compiled binaries
            log_warn "Android platform detected - tests will be limited (cross-compilation target)"
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
    if [ -z "$SDK_DIR" ]; then
        echo -e "${YELLOW}Please set SDK path (required):${NC}"
        printf "SDK path: "
        if read -r sdk_input && [ -n "$sdk_input" ] && [ -d "$sdk_input" ]; then
            export SDK_DIR="$sdk_input"
            log_success "SDK path set: $SDK_DIR"
        else
            log_error "Invalid SDK path"
            return 1
        fi
    else
        log_success "SDK path already set: $SDK_DIR"
    fi

    # Repository path setup
    if [ -z "$REPO_PATH" ]; then
        local current_dir
        current_dir=$(pwd)
        echo -e "${YELLOW}Repository path (current: $current_dir):${NC}"
        printf "Repository path [press Enter to use current directory]: "
        if read -r repo_input; then
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
    echo "5) android (Android ARM64 cross-compilation)"

    printf "Please select (1-5): "
    if read -r choice; then
        case "$choice" in
            1) INTERACTIVE_PLATFORM=$(detect_platform) ;;
            2) INTERACTIVE_PLATFORM="x86_64" ;;
            3) INTERACTIVE_PLATFORM="aarch64" ;;
            4) INTERACTIVE_PLATFORM="loongarch64" ;;
            5) INTERACTIVE_PLATFORM="android" ;;
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
    if read -r action_choice; then
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

# Compare version numbers (returns 0 if version1 >= version2, 1 otherwise)
version_ge() {
    local version1="$1"
    local version2="$2"

    # If versions are equal, return 0
    if [ "$version1" = "$version2" ]; then
        return 0
    fi

    # Use sort -V for version comparison
    # Sort both versions and check which one comes first
    local sorted_versions
    sorted_versions=$(printf '%s\n%s\n' "$version1" "$version2" | sort -V)
    local first_version
    first_version=$(echo "$sorted_versions" | head -n1)

    # If version2 is first (or equal), then version1 >= version2
    # If version1 is first, then version1 < version2
    if [ "$first_version" = "$version2" ]; then
        return 0  # version1 >= version2
    else
        return 1  # version1 < version2
    fi
}

# Check GCC version
check_gcc_version() {
    local min_version="9.4.0"

    if ! command -v gcc >/dev/null 2>&1; then
        log_error "gcc not found, cannot check version"
        return 1
    fi

    # Get gcc version (format: gcc (Ubuntu 9.4.0-1ubuntu1~20.04.2) 9.4.0)
    local gcc_version_output
    gcc_version_output=$(gcc --version 2>/dev/null | head -n1)

    # Extract version number (x.y.z format)
    local gcc_version
    gcc_version=$(echo "$gcc_version_output" | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -n1)

    if [ -z "$gcc_version" ]; then
        log_error "Failed to parse gcc version from: $gcc_version_output"
        return 1
    fi

    log_info "Detected gcc version: $gcc_version"

    # Compare versions (must be >= 9.4.0)
    if ! version_ge "$gcc_version" "$min_version"; then
        log_error "gcc version $gcc_version is too old"
        log_error "Required gcc version: >= $min_version"
        log_error "Please upgrade gcc to version $min_version or higher"
        return 1
    fi

    log_success "gcc version check passed: $gcc_version >= $min_version"
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

    # Check GCC version (must be > 9.4.0)
    log_info "Checking gcc version requirement..."
    if ! check_gcc_version; then
        log_error "gcc version check failed, compilation cannot proceed"
        return 1
    fi

    # Check platform-specific library dependencies
    local platform
    platform=$(uname -m)
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
                if ls "$path" >/dev/null 2>&1; then
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
                    local expected_dir
                    expected_dir=$(dirname "$expected_path")
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
    local SDK_DIR_arg="$1"

    if [ -z "$SDK_DIR_arg" ]; then
        log_error "SDK path not set"
        return 1
    fi

    if [ ! -d "$SDK_DIR_arg" ]; then
        log_error "SDK path does not exist: $SDK_DIR_arg"
        return 1
    fi

    if [ ! -f "$SDK_DIR_arg/env.sh" ]; then
        log_error "SDK environment script does not exist: $SDK_DIR_arg/env.sh"
        return 1
    fi

    log_info "Configuring SDK environment: $SDK_DIR_arg"
    source "$SDK_DIR_arg/env.sh"

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

    local ccache_dir
    ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"
    # Create ccache directory if it doesn't exist
    mkdir -p "$ccache_dir"
    ccache --set-config cache_dir="$ccache_dir"
    local ccache_max_size="${CCACHE_MAXSIZE:-20G}"
    ccache --set-config max_size="$ccache_max_size"
    if [ -n "${CCACHE_BASEDIR:-}" ]; then
        # Create base_dir directory if it doesn't exist
        mkdir -p "$CCACHE_BASEDIR"
        ccache --set-config base_dir="$CCACHE_BASEDIR"
    fi
    ccache --zero-stats

    if [ -z "${CCACHE_LOGFILE:-}" ]; then
        CCACHE_LOGFILE="/LocalRun/$(whoami)/cache/ccache.log"
        # Create log file directory if it doesn't exist
        mkdir -p "$(dirname "$CCACHE_LOGFILE")"
        export CCACHE_LOGFILE
    fi

    # Create custom bin directory for ccache wrappers
    local custom_bin_dir="$SDK_DIR/custom_bin"
    mkdir -p "$custom_bin_dir"

    # Create ccache symbolic links
    ln -sf /usr/bin/ccache "$custom_bin_dir/dlcc"
    export PATH="$custom_bin_dir:$PATH"
    export CUDA_NVCC_EXECUTABLE="$custom_bin_dir/dlcc"

    log_success "ccache configuration completed"
}

# Clean build directories (build_*)
clean_build_directories() {
    local repo_path="$1"
    if [ -z "$repo_path" ] || [ ! -d "$repo_path" ]; then
        log_warn "Cannot clean build directories: invalid repository path ($repo_path)"
        return
    fi

    mapfile -t build_dirs < <(find "$repo_path" -maxdepth 1 -type d -name 'build_*' -print 2>/dev/null)

    if [ ${#build_dirs[@]} -eq 0 ]; then
        log_info "No build_* directories found to clean under $repo_path"
        return
    fi

    for dir in "${build_dirs[@]}"; do
        log_info "Removing build directory: $dir"
        rm -rf "$dir"
    done

    log_success "Build directories cleaned"
}

# Clean ccache directory
clean_ccache_cache() {
    local ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"

    if [ -d "$ccache_dir" ]; then
        log_info "Removing ccache directory: $ccache_dir"
        rm -rf "$ccache_dir"
        log_success "ccache cache cleaned"
    else
        log_info "No ccache directory to clean (path: $ccache_dir)"
    fi
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

# Check dlPTI tools availability
check_dlpti_tools() {
    log_info "Checking dlPTI tools availability..."

    # Ensure SDK bin directory is in PATH and lib directory is in LD_LIBRARY_PATH
    if [ -n "$SDK_DIR" ]; then
        if [ -d "$SDK_DIR/bin" ]; then
            export PATH="$SDK_DIR/bin:$PATH"
            log_info "Added SDK bin directory to PATH: $SDK_DIR/bin"
        fi

        if [ -d "$SDK_DIR/lib" ]; then
            export LD_LIBRARY_PATH="$SDK_DIR/lib:${LD_LIBRARY_PATH:-}"
            log_info "Ensured SDK lib directory in LD_LIBRARY_PATH: $SDK_DIR/lib"
        fi
    fi

    if ! command -v dlpti_tools >/dev/null 2>&1; then
        log_error "dlpti_tools not found in PATH"
        log_error "Expected location: $SDK_DIR/bin/dlpti_tools"
        log_error "Please ensure dlPTI SDK is properly installed and sourced"
        log_error "Typical setup: source /path/to/dlpti/sdk/env.sh"
        return 1
    fi

    # Test dlpti_tools basic functionality
    # Note: dlpti_tools --help returns exit code 255, so we check if it produces expected output
    local help_output
    help_output=$(dlpti_tools --help 2>&1)
    if [[ ! "$help_output" =~ "dlpti_tools - A comprehensive tool" ]]; then
        log_error "dlpti_tools command failed or produced unexpected output"
        log_error "Output: $help_output"
        return 1
    fi

    log_success "dlPTI tools check passed"
    log_info "dlpti_tools location: $(which dlpti_tools)"
    return 0
}

# Setup dlPTI environment and validate options
setup_dlpti_environment() {
    local dlpti_options="$1"

    log_info "Setting up dlPTI environment..."

    # Validate dlPTI options format
    if [[ "$dlpti_options" =~ --data-file ]]; then
        log_info "Custom data file specified in dlPTI options"
    else
        log_info "No custom data file specified, dlpti_tools will use default naming"
    fi

    # Set dlPTI environment variables for better performance and compatibility
    export DLPTI_AUTO_LOAD=1

    # Ensure library paths are properly set for dlPTI
    if [ -n "$SDK_DIR" ] && [ -d "$SDK_DIR/lib" ]; then
        export LD_LIBRARY_PATH="$SDK_DIR/lib:${LD_LIBRARY_PATH:-}"
        log_info "Reinforced LD_LIBRARY_PATH with SDK lib: $SDK_DIR/lib"
    fi

    # Check for LD_PRELOAD conflicts and provide warnings
    if [ -n "$LD_PRELOAD" ]; then
        log_info "LD_PRELOAD detected: $LD_PRELOAD"
        if [[ "$LD_PRELOAD" =~ libdlpti_injection\.so ]]; then
            log_warn "dlPTI injection library detected in LD_PRELOAD"
            log_warn "This may cause issues with system commands (rm, ls, etc.)"
            log_info "Using safe command wrappers to avoid conflicts"
        fi
    fi

    # Check device permissions and setup
    log_info "Checking dlPTI device access..."

    # Ensure we have access to DL devices
    if [ -d "/dev/dl" ]; then
        local dl_devices
        dl_devices=$(find /dev -maxdepth 1 -name 'dl*' 2>/dev/null | wc -l)
        log_info "Found $dl_devices DL device(s)"

        # Check device permissions
        if ! ls -la /dev/dl* >/dev/null 2>&1; then
            log_warn "Cannot access DL devices, dlPTI may fail"
            log_warn "Consider running with appropriate permissions or check device status"
        fi
    else
        log_warn "No DL devices found in /dev/, dlPTI capture may not work properly"
    fi

    # Test basic dlpti_tools functionality with a simple command
    log_info "Testing dlPTI capture functionality..."

    # First, test if dlpti_tools can run without library issues
    local basic_test_output
    basic_test_output=$(dlpti_tools --help 2>&1)
    if [[ ! "$basic_test_output" =~ "dlpti_tools - A comprehensive tool" ]]; then
        log_warn "dlpti_tools basic test failed"
        log_warn "Output: $basic_test_output"

        # Run troubleshooting if basic test failed
        echo ""
        dlpti_troubleshoot
        echo ""

        return 1  # Indicate setup failure
    fi

    # Test capture functionality with a very simple command
    local test_output
    test_output=$(timeout 10 dlpti_tools capture --activity-mask cmd --data-file /tmp/dlpti_test_$$.db -- /bin/true 2>&1)
    local test_ret=$?

    if [ $test_ret -eq 0 ]; then
        log_success "dlPTI capture test successful"
        # Use safe_rm to avoid dlpti injection issues
        safe_rm -f "/tmp/dlpti_test_$$.db" 2>/dev/null
    else
        log_warn "dlPTI capture test failed (exit code: $test_ret)"
        log_warn "Test output: $test_output"

        # Check if it's a library loading issue
        if [[ "$test_output" != "libdlpti.so" ]] || [[ "$test_output" =~ "cannot open shared object file" ]]; then
            log_error "dlPTI library loading issue detected"

            # Run troubleshooting for library issues
            echo ""
            dlpti_troubleshoot
            echo ""

            return 1  # Indicate setup failure
        else
            log_warn "dlPTI profiling may not work correctly, but continuing..."
            log_warn "This might be due to device access issues rather than library problems"
        fi
    fi

    log_success "dlPTI environment setup completed"
    return 0
}

# Safe command wrapper for commands that might conflict with dlPTI injection
safe_exec() {
    # Execute command without LD_PRELOAD to avoid dlpti injection issues
    # Use bash builtin to avoid LD_PRELOAD affecting env command itself
    local old_preload="$LD_PRELOAD"
    unset LD_PRELOAD
    "$@"
    local ret=$?
    if [ -n "$old_preload" ]; then
        export LD_PRELOAD="$old_preload"
    fi
    return $ret
}

# dlPTI troubleshooting and diagnostics
dlpti_troubleshoot() {
    log_info "=== dlPTI Troubleshooting ==="

    # Check basic environment
    log_info "1. Environment Check:"
    log_info "   SDK path: ${SDK_DIR:-NOT SET}"
    log_info "   dlpti_tools: $(which dlpti_tools 2>/dev/null || echo "NOT FOUND")"
    log_info "   DLPTI_AUTO_LOAD: ${DLPTI_AUTO_LOAD:-NOT SET}"
    log_info "   LD_LIBRARY_PATH: ${LD_LIBRARY_PATH:-NOT SET}"
    log_info "   LD_PRELOAD: ${LD_PRELOAD:-NOT SET}"

    # Check library files
    log_info "2. Library Check:"
    if [ -n "$SDK_DIR" ]; then
        local libdlpti="$SDK_DIR/lib/libdlpti.so"
        local libdlpti_injection="$SDK_DIR/lib/libdlpti_injection.so"

        log_info "   libdlpti.so: $([ -f "$libdlpti" ] && echo "EXISTS" || echo "MISSING")"
        log_info "   libdlpti_injection.so: $([ -f "$libdlpti_injection" ] && echo "EXISTS" || echo "MISSING")"

        if [ -f "$libdlpti" ]; then
            log_info "   libdlpti.so size: $(stat -c%s "$libdlpti" 2>/dev/null || echo "UNKNOWN") bytes"
        fi
    fi

    # Check device access
    log_info "3. Device Access Check:"
    if [ -d "/dev" ]; then
        local dl_devices
        dl_devices=$(ls /dev/dl* 2>/dev/null || echo "")
        if [ -n "$dl_devices" ]; then
            log_info "   DL devices found:"
            find /dev -maxdepth 1 -name 'dl*' -ls 2>/dev/null | while read -r line; do
                log_info "     $line"
            done
        else
            log_warn "   No DL devices found in /dev/"
        fi
    fi

    # Check permissions
    log_info "4. Permission Check:"
    log_info "   Current user: $(whoami)"
    log_info "   User groups: $(groups)"

    # Suggest solutions
    log_info "5. Common Solutions:"
    log_info "   - Ensure DL devices are available and accessible"
    log_info "   - Check if running with appropriate permissions"
    log_info "   - Verify SDK environment is properly sourced"
    log_info "   - For permission issues, consider: sudo usermod -a -G dl \$(whoami)"

    # Check for LD_PRELOAD conflicts
    if [ -n "$LD_PRELOAD" ] && [[ "$LD_PRELOAD" =~ libdlpti_injection\.so ]]; then
        log_info "6. LD_PRELOAD Conflict Solutions:"
        log_info "   - LD_PRELOAD injection may cause system command failures"
        log_info "   - Use 'env -u LD_PRELOAD command' for problematic commands"
        log_info "   - Consider temporarily disabling LD_PRELOAD if issues persist"
        log_info "   - Example: env -u LD_PRELOAD rm /tmp/file"
    fi
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
    local debug="${3:-false}"

    log_info "Starting llama.cpp compilation (platform: $platform)"
    local compile_start_time
    compile_start_time=$(date +%s)

    # Ensure we're in the correct directory
    if [ -z "$REPO_PATH" ]; then
        REPO_PATH=$(pwd)
        export REPO_PATH
    fi
    cd "$REPO_PATH"

    # Create build directory
    mkdir -p "$build_dir"

    # Determine build type
    local build_type="Release"
    if [ "$debug" = "true" ]; then
        build_type="Debug"
        log_info "Compiling in Debug mode (with debug symbols, no optimization)"
    else
        log_info "Compiling in Release mode (optimized)"
    fi

    # Propagate Kineto toggle from environment (default OFF if unset)
    local cmake_kineto_option="-DLLAMA_KINETO=${LLAMA_KINETO:-OFF}"

    # Build CMake arguments
    local cmake_common_args=(
        -G Ninja
        -B "$build_dir"
        -DGGML_DLCU=ON
        -DCMAKE_VERBOSE_MAKEFILE=ON
        -DCMAKE_BUILD_TYPE="$build_type"
        -DGGML_BACKEND_DL=ON
        -DGGML_CUDA_GRAPHS=ON
        -DLLAMA_CURL=OFF
        -DLLAMA_OPENSSL=ON
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
        -DGGML_CUDA_FA=ON
        -DGGML_CUDA_FA_ALL_QUANTS=ON
        -DSDK_DIR="$SDK_DIR"
        ${cmake_kineto_option}
    )

    # Add platform-specific arguments
    IFS=' ' read -ra platform_args <<< "$CMAKE_ARGS"
    cmake_common_args+=("${platform_args[@]}")

    # Android special handling: set Android NDK toolchain
    if [ "$platform" = "android" ]; then
        log_info "Setting up Android NDK toolchain for cross-compilation..."

        local android_ndk_root="${ANDROID_NDK_ROOT:-${NDK_ROOT:-/opt/android-sdk-linux/ndk/25.2.9519653}}"
        local android_toolchain_file="${ANDROID_TOOLCHAIN_FILE:-${tool_chain_cmake:-${android_ndk_root}/build/cmake/android.toolchain.cmake}}"

        # Support: ANDROID_API_LEVEL, ANDROID_PLATFORM, android_api_level, API
        local android_api_candidate="${ANDROID_API_LEVEL:-${ANDROID_PLATFORM:-${android_api_level:-${API:-25}}}}"
        if [[ "$android_api_candidate" =~ ^android- ]]; then
            android_api_candidate="${android_api_candidate#android-}"
        fi
        local android_api_level="$android_api_candidate"
        local android_platform="android-${android_api_level}"

        # Allow overriding ABI/target triple
        local android_abi="${ANDROID_ABI:-arm64-v8a}"
        local android_target="${ANDROID_CLANG_TRIPLE:-${ANDROID_TARGET:-${TARGET:-aarch64-linux-android}}}"

        # Validate Android NDK exists
        if [ ! -f "$android_toolchain_file" ]; then
            log_error "Android NDK toolchain not found: $android_toolchain_file"
            log_error "Set NDK path via ANDROID_NDK_ROOT or tool_chain_cmake"
            return 1
        fi

        log_info "Using Android NDK: $android_ndk_root"
        log_info "Using Android toolchain: $android_toolchain_file"
        log_info "Android ABI: $android_abi"
        log_info "Android API level: $android_api_level"
        log_info "Android target triple: $android_target"
        log_info "CUDA compiler (dlcc): $CUDA_NVCC_EXECUTABLE"

        # Set explicit Android compilers
        local android_c_compiler="${ANDROID_CC:-${android_ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/bin/${android_target}${android_api_level}-clang}"
        local android_cxx_compiler="${ANDROID_CXX:-${android_ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/bin/${android_target}${android_api_level}-clang++}"
        log_info "Android C compiler: $android_c_compiler"
        log_info "Android C++ compiler: $android_cxx_compiler"

        # Ensure CUDA compilation uses dlcc from SDK, not Android NDK
        if [ ! -f "$CUDA_NVCC_EXECUTABLE" ]; then
            log_error "CUDA compiler (dlcc) not found: $CUDA_NVCC_EXECUTABLE"
            log_error "Please ensure SDK environment is properly set up"
            return 1
        fi

        # Android-specific library paths and flags
        local android_sysroot="${ANDROID_SYSROOT:-${android_ndk_root}/toolchains/llvm/prebuilt/linux-x86_64/sysroot}"
        local android_lib_path="${android_sysroot}/usr/lib/${android_target}/${android_api_level}"
        local default_linker_flags="-L${android_lib_path} -latomic"
        local android_linker_flags="${ANDROID_LINKER_FLAGS:-$default_linker_flags}"
        local android_shared_linker_flags="${ANDROID_SHARED_LINKER_FLAGS:-$android_linker_flags}"
        local android_c_flags="${ANDROID_C_FLAGS:--march=armv8-a}"
        local android_cxx_flags="${ANDROID_CXX_FLAGS:--march=armv8-a}"

        log_info "Android sysroot: $android_sysroot"
        log_info "Android lib path: $android_lib_path"

        cmake_common_args+=(
            "-DCMAKE_TOOLCHAIN_FILE=$android_toolchain_file"
            "-DANDROID_ABI=$android_abi"
            "-DANDROID_PLATFORM=$android_platform"
            "-DANDROID_NDK=$android_ndk_root"
            "-DCMAKE_C_COMPILER=$android_c_compiler"
            "-DCMAKE_CXX_COMPILER=$android_cxx_compiler"
            "-DCMAKE_C_FLAGS=$android_c_flags"
            "-DCMAKE_CXX_FLAGS=$android_cxx_flags"
            "-DCMAKE_EXE_LINKER_FLAGS=$android_linker_flags"
            "-DCMAKE_SHARED_LINKER_FLAGS=$android_shared_linker_flags"
            "-DGGML_OPENMP=OFF"
            "-DGGML_LLAMAFILE=OFF"
            "-DGGML_INTERNAL_MATMUL_INT8=OFF"
            "-DCUDA_NVCC_EXECUTABLE=$CUDA_NVCC_EXECUTABLE"
            "-DCMAKE_CUDA_COMPILER=$CUDA_NVCC_EXECUTABLE"
            "-DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH"
            "-DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH"
        )
    # LoongArch64 special handling: set correct OpenMP library path
    elif [ "$platform" = "loongarch64" ]; then
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
            if ls "$path" >/dev/null 2>&1; then
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
    ninja -j "$(cpu_num=$(lscpu | grep '^CPU(s):' | awk '{print $2}'); if [ $((cpu_num-2)) -gt 0 ]; then echo $((cpu_num-2)); else echo 1; fi)"

    # Calculate compilation time
    local compile_end_time
    compile_end_time=$(date +%s)
    local compile_duration=$((compile_end_time - compile_start_time))

    log_success "Compilation completed, time taken: ${compile_duration} seconds"

    # Update compile_commands.json symlink for IDE support
    local compile_commands_src="$build_dir/compile_commands.json"
    local compile_commands_dst="${REPO_PATH}/compile_commands.json"
    if [ -f "$compile_commands_src" ]; then
        rm -f "$compile_commands_dst"
        ln -sf "$compile_commands_src" "$compile_commands_dst"
        log_info "Updated compile_commands.json symlink"
    else
        log_warn "compile_commands.json not found in $build_dir"
    fi
}

# Run tests with repeat support
run_tests_with_repeat() {
    local platform="$1"
    local build_dir="$2"
    local debug="$3"
    local simple_test="$4"
    local simple_model="$5"
    local simple_bench="$6"
    local repeat_count="${7:-1}"
    local enable_dlpti="${8:-false}"
    local dlpti_options="${9:-}"
    local no_fa="${10:-false}"

    if [ "$repeat_count" -eq 1 ]; then
        # Single run
        run_tests "$platform" "$build_dir" "$debug" "$simple_test" "$simple_model" "$simple_bench" "$enable_dlpti" "$dlpti_options" "$no_fa"
        return $?
    fi

    # Multiple runs with statistics
    log_info "Running tests $repeat_count times with statistics collection..."
    echo ""

    local total_runs=0
    local failed_runs=0
    local passed_runs=0

    # Progress bar display
    local start_time=$(date +%s)

    for i in $(seq 1 $repeat_count); do
        total_runs=$((total_runs + 1))

        # Print progress on same line
        printf "\r${BLUE}[INFO]${NC} Progress: [%d/%d] " "$i" "$repeat_count"

        if run_tests "$platform" "$build_dir" "$debug" "$simple_test" "$simple_model" "$simple_bench" "$enable_dlpti" "$dlpti_options" "$no_fa" >/dev/null 2>&1; then
            passed_runs=$((passed_runs + 1))
            printf "${GREEN}✓${NC} Pass: %d  ${RED}✗${NC} Fail: %d" "$passed_runs" "$failed_runs"
        else
            failed_runs=$((failed_runs + 1))
            printf "${GREEN}✓${NC} Pass: %d  ${RED}✗${NC} Fail: %d ${RED}[FAILED at iteration %d]${NC}" "$passed_runs" "$failed_runs" "$i"

            # Log failure to main log
            log_error "Test iteration $i failed"
        fi

        # Brief pause between runs
        if [ "$i" -lt "$repeat_count" ]; then
            sleep 1
        fi
    done

    # New line after progress bar
    echo ""
    echo ""

    # Calculate elapsed time
    local end_time
    end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_str=""
    if [ $elapsed -ge 60 ]; then
        local minutes=$((elapsed / 60))
        local seconds=$((elapsed % 60))
        elapsed_str="${minutes}m ${seconds}s"
    else
        elapsed_str="${elapsed}s"
    fi

    # Print summary
    log_info ""
    log_info "=========================================="
    log_info "  Test Repeat Statistics Summary"
    log_info "=========================================="
    log_info "Total Runs:   $total_runs"
    log_success "Passed Runs:  $passed_runs"
    if [ $failed_runs -gt 0 ]; then
        log_error "Failed Runs:  $failed_runs"
    else
        log_info "Failed Runs:  $failed_runs"
    fi

    local pass_rate=$(awk "BEGIN {printf \"%.2f\", ($passed_runs/$total_runs)*100}")
    log_info "Pass Rate:    ${pass_rate}%"
    log_info "Total Time:   $elapsed_str"
    log_info "=========================================="

    if [ $failed_runs -gt 0 ]; then
        log_warn "Some test iterations failed - check main log for details"
    else
        log_success "All test iterations passed!"
    fi

    # Return failure if any run failed
    if [ $failed_runs -gt 0 ]; then
        return 1
    else
        return 0
    fi
}

# Run tests
run_tests() {
    local platform="$1"
    local build_dir="$2"
    local debug="${3:-false}"
    local simple_test="${4:-false}"
    local simple_model="${5:-false}"
    local simple_bench="${6:-false}"
    local enable_dlpti="${7:-false}"
    local dlpti_options="${8:-}"
    local no_fa="${9:-false}"

    log_info "Starting test execution (platform: $platform)"

    # Set test environment variables
    export GGML_DEBUG=1

    # Platform-specific environment variables
    if [ "$platform" = "android" ]; then
        # Android cross-compilation - skip most tests as they can't run on host
        log_warn "Android platform detected - most tests will be skipped (cross-compilation target)"
        log_warn "Only basic compilation verification will be performed"

        # Check if binaries were created
        local android_build_dir="${REPO_PATH}/build_android"
        if [ ! -d "$android_build_dir/bin" ]; then
            log_error "Android build directory not found: $android_build_dir/bin"
            return 1
        fi

        local android_binaries=("llama-cli" "llama-server" "llama-bench")
        local missing_binaries=()

        for binary in "${android_binaries[@]}"; do
            if [ ! -f "$android_build_dir/bin/$binary" ]; then
                missing_binaries+=("$binary")
            fi
        done

        if [ ${#missing_binaries[@]} -gt 0 ]; then
            log_error "Missing Android binaries:"
            for binary in "${missing_binaries[@]}"; do
                log_error "  - $binary"
            done
            return 1
        fi

        log_success "Android cross-compilation verification completed"
        log_info "Android binaries created in: $android_build_dir/bin/"

        # List created binaries
        log_info "Created Android binaries:"
        ls -la "$android_build_dir/bin/"

        return 0
    elif [ "$platform" = "loongarch64" ]; then
        # LoongArch64 specific settings to avoid timeout issues
        export GGML_CUDA_DEVICE_TIMEOUT=30
        export GGML_CUDA_FORCE_MMQ=1
        export GGML_CUDA_PEER_MAX_BATCH_SIZE=128
        export DENGLIN_TIMEOUT=60
        log_info "Applied LoongArch64 specific CUDA settings"
        prepare_loongarch64_test
    fi

    # Apply debug mode settings
    if [ "$debug" = "true" ]; then
        export GGML_DEBUG=2
        export GGML_VERBOSE=1
        export CUDA_LAUNCH_BLOCKING=1
        log_info "Enabled debug mode (verbose output + synchronous CUDA)"
    fi

    export CUDA_VISIBLE_DEVICES=0

    # Set up environment variables for the comprehensive test script
    export DOCKER_PLATFORM="$platform"
    export REPO_PATH="${REPO_PATH}"

    # Handle simple benchmark mode separately - call bench_llama_cpp.sh directly
    if [ "$simple_bench" = "true" ]; then
        unset CUDA_VISIBLE_DEVICES
        log_info "Simple benchmark test mode enabled - calling bench_llama_cpp.sh directly"

        # Check if benchmark script exists
        local bench_script="${REPO_PATH}/.dlci/bench_llama_cpp.sh"

        if [ ! -f "$bench_script" ]; then
            log_error "Benchmark script not found: $bench_script"
            return 1
        fi

        if [ ! -x "$bench_script" ]; then
            log_info "Making benchmark script executable..."
            chmod +x "$bench_script"
        fi

        log_info "Running benchmark using: $bench_script"

        # Prepare arguments for bench script
        local bench_args=()
        if [ "$VERBOSE" = "true" ]; then
            bench_args+=("--verbose")
        fi

        # Run benchmark script
        local ret=0
        local start_time
        start_time=$(date +%s)

        log_info "Output will be saved to $MAIN_LOG_FILE"
        bash "$bench_script" "${bench_args[@]}" >> "$MAIN_LOG_FILE" 2>&1
        ret=$?

        local end_time
        end_time=$(date +%s)
        local bench_duration=$((end_time - start_time))

        # Format duration
        local duration_str=""
        if [ $bench_duration -ge 3600 ]; then
            local hours=$((bench_duration / 3600))
            local minutes=$(((bench_duration % 3600) / 60))
            local seconds=$((bench_duration % 60))
            duration_str="${hours}h ${minutes}m ${seconds}s"
        elif [ $bench_duration -ge 60 ]; then
            local minutes=$((bench_duration / 60))
            local seconds=$((bench_duration % 60))
            duration_str="${minutes}m ${seconds}s"
        else
            duration_str="${bench_duration}s"
        fi

        if [ "$ret" -eq 0 ]; then
            # bench_llama_cpp.sh already outputs success message, just show log location
            log_info "Benchmark log saved to: $MAIN_LOG_FILE"
            return 0
        else
            log_error "Benchmark failed (exit code: $ret)"
            log_error "Execution time: $duration_str"
            log_error "Benchmark log file: $MAIN_LOG_FILE"

            # Show log content on error
            if [ -f "$MAIN_LOG_FILE" ]; then
                log_error "=== Last 50 lines of benchmark output ==="
                tail -n 50 "$MAIN_LOG_FILE"
                log_error "=== End of benchmark output ==="
            fi

            return 1
        fi
    fi

    # Build test script arguments
    local test_script_args=""
    if [ "$simple_test" = "true" ]; then
        test_script_args="--simple-test"
        log_info "Simple test mode enabled"
    fi
    if [ "$simple_model" = "true" ]; then
        test_script_args="$test_script_args --simple-model"
        log_info "Simple model test mode enabled"
    fi
    if [ "$simple_tp" = "true" ]; then
        test_script_args="$test_script_args --simple-tp"
        log_info "Simple TP test mode enabled"
    fi
    if [ "$simple_perf" = "true" ]; then
        test_script_args="$test_script_args --simple-perf"
        log_info "Simple performance test mode enabled"
    fi
    if [ "$big_model" = "true" ]; then
        test_script_args="$test_script_args --big"
        log_info "Big model mode enabled"
    fi
    if [ "$no_fa" = "true" ]; then
        test_script_args="$test_script_args --no-fa"
        log_info "Flash Attention disabled (--no-fa)"
    fi

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

    log_info "Running test suite using: $test_script"

    # dlPTI integration
    local dlpti_prefix=""
    local dlpti_data_file=""

    if [ "$enable_dlpti" = "true" ]; then
        log_info "dlPTI profiling enabled, setting up capture environment..."

        # Check dlPTI tools availability
        if ! check_dlpti_tools; then
            log_error "dlPTI tools check failed, disabling profiling"
            enable_dlpti=false
        else
            if setup_dlpti_environment "$dlpti_options"; then
                # Build dlpti_tools capture command prefix
                dlpti_prefix="dlpti_tools capture $dlpti_options --"

                # Extract data file name for reporting (if specified)
                if [[ "$dlpti_options" =~ --data-file[[:space:]]+([^[:space:]]+) ]]; then
                    dlpti_data_file="${BASH_REMATCH[1]}"
                else
                    dlpti_data_file="capture-$(date +%Y%m%d%H%M%S).db"
                fi

                log_info "dlPTI capture command: $dlpti_prefix"
                log_info "dlPTI data file: $dlpti_data_file"
            else
                log_warn "dlPTI environment setup failed, but continuing with profiling attempt..."
                # Still try to set up the command, but with warnings
                dlpti_prefix="dlpti_tools capture $dlpti_options --"
                if [[ "$dlpti_options" =~ --data-file[[:space:]]+([^[:space:]]+) ]]; then
                    dlpti_data_file="${BASH_REMATCH[1]}"
                else
                    dlpti_data_file="capture-$(date +%Y%m%d%H%M%S).db"
                fi

                log_warn "dlPTI may not work correctly due to setup issues"
            fi
        fi
    fi

    # Run the comprehensive test script
    local ret=0
    local start_time
    start_time=$(date +%s)

    # Use main log file for all output

    if [ "$platform" = "loongarch64" ]; then
        # LoongArch64 specific handling with timeout monitoring
        log_info "Running comprehensive test suite with LoongArch64 timeout monitoring..."
        log_info "Output will be saved to $MAIN_LOG_FILE"
        if [ "$enable_dlpti" = "true" ]; then
            env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PATH="$PATH" DLPTI_AUTO_LOAD="$DLPTI_AUTO_LOAD" \
            $dlpti_prefix bash "$test_script" $test_script_args >> "$MAIN_LOG_FILE" 2>&1 &
        else
            bash "$test_script" $test_script_args >> "$MAIN_LOG_FILE" 2>&1 &
        fi
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

        log_info "Output will be saved to $MAIN_LOG_FILE"
        if [ "$enable_dlpti" = "true" ]; then
            env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" PATH="$PATH" DLPTI_AUTO_LOAD="$DLPTI_AUTO_LOAD" \
            $dlpti_prefix bash "$test_script" $test_script_args >> "$MAIN_LOG_FILE" 2>&1
            ret=$?
        else
            bash "$test_script" $test_script_args >> "$MAIN_LOG_FILE" 2>&1
            ret=$?
        fi
    fi

    local end_time
    end_time=$(date +%s)
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

    # Report results (unified for all test modes)
    local test_mode_name="Test suite"
    if [ "$simple_test" = "true" ]; then
        test_mode_name="Simple tests"
    else
        test_mode_name="Comprehensive test suite"
    fi

    if [ "$ret" -eq 0 ]; then
        log_success "$test_mode_name completed successfully!"
        log_success "Total execution time: $duration_str"
        log_info "Full test log saved to: $MAIN_LOG_FILE"

        # Report dlPTI data file if profiling was enabled
        if [ "$enable_dlpti" = "true" ] && [ -n "$dlpti_data_file" ]; then
            echo ""
            log_info "=== dlPTI Profiling Results ==="

            if [ -f "$dlpti_data_file" ]; then
                local file_size
                file_size=$(stat -c%s "$dlpti_data_file" 2>/dev/null || echo "0")
                if [ "$file_size" -gt 1024 ]; then
                    local dlpti_abs_path
                    dlpti_abs_path=$(realpath "$dlpti_data_file")
                    log_success "dlPTI profile data saved to: $dlpti_abs_path"
                    log_success "Profile data size: $(du -h "$dlpti_data_file" | cut -f1)"
                    echo ""
                    log_info "Analysis options:"
                    log_info "  1. GUI Analysis: dlsys-ui $dlpti_abs_path"
                    log_info "  2. Export to Perfetto: dlpti_tools export --format perfetto-json $dlpti_abs_path"
                    log_info "  3. Range-based export: dlpti_tools export --export-range 10%:90% --format perfetto-json $dlpti_abs_path"
                else
                    local dlpti_abs_path
                    dlpti_abs_path=$(realpath "$dlpti_data_file")
                    log_warn "dlPTI data file exists but is very small ($file_size bytes): $dlpti_abs_path"
                    log_warn "This may indicate a capture failure or very short execution time"
                fi
            else
                local dlpti_abs_path
                dlpti_abs_path=$(realpath "$dlpti_data_file" 2>/dev/null || echo "$dlpti_data_file")
                log_warn "dlPTI data file not found: $dlpti_abs_path"
                log_info "Checking for alternative dlPTI output files..."

                # Look for any capture files in current directory
                local capture_files
                capture_files=$(find . -maxdepth 1 -name 'capture-*.db' 2>/dev/null | head -5)
                if [ -n "$capture_files" ]; then
                    log_info "Found alternative capture files:"
                    for file in $capture_files; do
                        local size
                        size=$(du -h "$file" | cut -f1)
                        local file_abs_path
                        file_abs_path=$(realpath "$file")
                        log_info "  - $file_abs_path ($size)"
                    done
                else
                    log_warn "No dlPTI capture files found in current directory"
                    log_info "This may indicate:"
                    log_info "  - dlPTI capture failed due to device access issues"
                    log_info "  - Insufficient permissions to access DL devices"
                    log_info "  - Target application executed too quickly"
                fi
            fi

            # Check for dlPTI error messages in main log
            if [ -f "$MAIN_LOG_FILE" ]; then
                local dlpti_errors
                dlpti_errors=$(grep -i "dlpti.*error\|capture failed" "$MAIN_LOG_FILE" 2>/dev/null || true)
                if [ -n "$dlpti_errors" ]; then
                    log_warn "dlPTI errors detected in log:"
                    echo "$dlpti_errors" | while read -r line; do
                        log_warn "  $line"
                    done
                fi
            fi
        fi
        return 0
    elif [ "$ret" -eq 124 ]; then
        log_error "$test_mode_name timed out after $duration_str"
        log_error "Test log file: $MAIN_LOG_FILE"

        # Show log content on timeout (always show for timeout errors)
        if [ -f "$MAIN_LOG_FILE" ]; then
            log_error "=== Last 50 lines of test output ==="
            tail -n 50 "$MAIN_LOG_FILE"
            log_error "=== End of test output ==="
        fi

        if [ "$simple_test" != "true" ]; then
            log_error "Consider running with --simple-test for faster execution"
        fi
        return 1
    else
        log_error "$test_mode_name failed (exit code: $ret)"
        log_error "Execution time: $duration_str"
        log_error "Test log file: $MAIN_LOG_FILE"

        # Show log content on error (always show for failed tests)
        if [ -f "$MAIN_LOG_FILE" ] && [ "$verbose" != "true" ]; then
            log_error "=== Last 50 lines of test output ==="
            tail -n 50 "$MAIN_LOG_FILE"
            log_error "=== End of test output ==="
        elif [ "$verbose" = "true" ]; then
            log_error "Error details already shown above in verbose mode"
        fi

        return 1
    fi
}

# Main function
main() {
    local platform=""
    local action="all"
    local SDK_DIR_arg=""
    local REPO_PATH_arg=""
    local model_path_arg=""
    local auto_install=true
    local interactive=false
    local debug=false  # Default to release mode; use --debug to enable debug mode
    local simple_test=false
    local simple_model=false
    local simple_tp=false
    local simple_perf=false
    local simple_bench=false
    local big_model=false
    local skip_device_check=true
    local repeat_test=1
    local enable_dlpti=false
    local dlpti_options=""
    local no_fa=false
    local clean_build_dirs_flag=false
    local clean_ccache_flag=false

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
            -c)
                action="compile"
                shift
                ;;
            -t)
                action="test"
                shift
                ;;
            -cc)
                clean_build_dirs_flag=true
                action="compile"
                shift
                ;;
            -ccc)
                clean_build_dirs_flag=true
                clean_ccache_flag=true
                action="compile"
                shift
                ;;
            --sdk-path)
                SDK_DIR_arg="$2"
                shift 2
                ;;
            --repo-path)
                REPO_PATH_arg="$2"
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
            --debug)
                debug=true
                shift
                ;;
            --debug=*)
                debug_value="${1#*=}"
                if [ "$debug_value" = "0" ]; then
                    debug=false
                else
                    debug=true
                fi
                shift
                ;;
            --release)
                debug=false
                shift
                ;;
            --simple-test)
                simple_test=true
                shift
                ;;
            -st)
                simple_test=true
                shift
                ;;
            --simple-model)
                simple_model=true
                shift
                ;;
            -sm)
                simple_model=true
                shift
                ;;
            --simple-tp)
                simple_tp=true
                shift
                ;;
            --simple-perf)
                simple_perf=true
                shift
                ;;
            --simple-bench)
                simple_bench=true
                shift
                ;;
            -sb)
                simple_bench=true
                shift
                ;;
            --big)
                big_model=true
                shift
                ;;
            --repeat-test)
                repeat_test="$2"
                shift 2
                ;;
            --skip-device-check)
                skip_device_check=true
                shift
                ;;
            --dlpti)
                enable_dlpti=true
                dlpti_options="$2"
                shift 2
                ;;
            --no-fa)
                no_fa=true
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

    # Set default repository path for logging
    if [ -z "$REPO_PATH" ]; then
        export REPO_PATH="$(pwd)"
    fi

    # Ensure log directory exists
    mkdir -p "$REPO_PATH"

    # Initialize logging
    init_logging
    log_info "Script started with arguments: $*"

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
    if [ -n "$SDK_DIR_arg" ]; then
        export SDK_DIR="$SDK_DIR_arg"
    fi

    if [ -n "$REPO_PATH_arg" ]; then
        export REPO_PATH="$REPO_PATH_arg"
    elif [ -z "$REPO_PATH" ]; then
        export REPO_PATH="$(pwd)"
    fi

    if [ -n "$model_path_arg" ]; then
        export LOCAL_MODEL_PATH="$model_path_arg"
    elif [ -z "$LOCAL_MODEL_PATH" ]; then
        export LOCAL_MODEL_PATH="$(get_model_path)"
        log_info "Using default LOCAL_MODEL_PATH from config: $LOCAL_MODEL_PATH"
    fi

    # Map DLGPU_X86_SDK_DIR and llvm_devel_path to SDK_DIR if not already set
    if [ -z "$SDK_DIR" ]; then
        if [ -n "$DLGPU_X86_SDK_DIR" ]; then
            export SDK_DIR="$DLGPU_X86_SDK_DIR"
            log_info "Using DLGPU_X86_SDK_DIR as SDK path: $SDK_DIR"
        elif [ -n "$llvm_devel_path" ]; then
            export SDK_DIR="$llvm_devel_path"
            log_info "Using llvm_devel_path as SDK path: $SDK_DIR"
        fi
    fi

    # Validate required parameters
    if [ -z "$SDK_DIR" ]; then
        log_error "SDK path not set, please use --sdk-path or set environment variable SDK_DIR"
        log_error "Alternatively, set DLGPU_X86_SDK_DIR or llvm_devel_path"
        echo "Quick setup: export SDK_DIR=\"/path/to/your/sdk\""
        echo "Or export llvm_devel_path=\"/path/to/your/sdk\""
        exit 1
    fi

    # Display configuration information
    log_info "=== Execution Configuration ==="
    log_info "Platform: $platform"
    log_info "Action: $action"
    log_info "SDK path: $SDK_DIR"
    log_info "Repository path: $REPO_PATH"
    log_info "Model path: $LOCAL_MODEL_PATH"
    if [ "$debug" = "true" ]; then
        log_info "Build mode: DEBUG (Debug build + verbose output)"
    else
        log_info "Build mode: RELEASE (optimized, default; can be forced with --release or --debug=0)"
    fi
    if [ "$simple_test" = "true" ]; then
        log_info "Simple test mode: enabled"
    fi
    if [ "$simple_model" = "true" ]; then
        log_info "Simple model test mode: enabled (Qwen2.5 with GGML_DLFA_READY=1)"
    fi
    if [ "$simple_bench" = "true" ]; then
        log_info "Simple benchmark test mode: enabled (llama-bench with Qwen3-30B)"
    fi
    if [ "$repeat_test" -gt 1 ]; then
        log_info "Repeat test: $repeat_test times"
    fi
    if [ "$skip_device_check" = "true" ]; then
        log_warn "Device check: disabled (--skip-device-check)"
    fi
    if [ "$enable_dlpti" = "true" ]; then
        log_info "dlPTI profiling: ENABLED"
        log_info "dlPTI options: $dlpti_options"
    fi
    if [ "$no_fa" = "true" ]; then
        log_info "Flash Attention: DISABLED (--no-fa)"
    fi
    echo ""

    if [ "$clean_build_dirs_flag" = "true" ]; then
        log_info "Cleaning build directories before compilation (-cc/-ccc)"
        clean_build_directories "$REPO_PATH"
    fi

    if [ "$clean_ccache_flag" = "true" ]; then
        log_info "Cleaning ccache cache before compilation (-ccc)"
        clean_ccache_cache
    fi

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
            setup_sdk "$SDK_DIR"
            setup_ccache "$platform"
            get_version_info
            compile_llama_cpp "$platform" "$build_dir" "$debug"

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

                run_tests_with_repeat "$platform" "$build_dir" "$debug" "$simple_test" "$simple_model" "$simple_bench" "$repeat_test" "$enable_dlpti" "$dlpti_options" "$no_fa"

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

    # Log completion
    log_info "Script completed successfully"
    log_info "Complete log saved to: $MAIN_LOG_FILE"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
