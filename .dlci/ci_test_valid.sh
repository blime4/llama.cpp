#!/bin/bash
#
# CI Test Validation Script for Llama.cpp
#
# This script automatically:
# 1. Detects the platform architecture
# 2. Sets default SDK path
# 3. Downloads the latest release package
# 4. Extracts and sets up the binary path
# 5. Runs CI tests using ci_test.sh
#
# No manual environment variable setup required!
#

set -e

# Source configuration utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Configuration - loaded from unified config
DEFAULT_SDK_TAG=$(get_sdk_tag)
DEFAULT_LOCAL_MODEL_PATH=$(get_model_path)
WORK_DIR="/tmp/llama_cpp_ci_test_$$"  # Use PID for uniqueness
RELEASE_ARTIFACTORY_BASE="http://ext-artifactory.denglin.com:8082/artifactory/llama.cpp-release"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Function to print banner
print_banner() {
    echo "=========================================="
    echo "   CI Test Validation for Llama.cpp"
    echo "=========================================="
    echo ""
}

# Function to detect platform architecture
detect_platform() {
    local arch=$(uname -m)
    case $arch in
        x86_64)
            echo "x86_64"
            ;;
        aarch64|arm64)
            echo "aarch64"
            ;;
        riscv64)
            echo "riscv64"
            ;;
        loongarch64)
            echo "loongarch64"
            ;;
        *)
            log_error "Unsupported architecture: $arch"
            log_error "Supported: x86_64, aarch64, riscv64, loongarch64"
            exit 1
            ;;
    esac
}

# Function to set up SDK path
setup_sdk_path() {
    local sdk_tag="${1:-$DEFAULT_SDK_TAG}"
    local platform="$2"

    # Try to find existing SDK installation
    local potential_paths=(
        "/LocalRun/$(whoami)/local_builds/sdk_llama_cpp/${sdk_tag}/sdk"
        "${CI_BUILDS_DIR}/sdk_llama_cpp/${sdk_tag}/sdk"
        "/tmp/sdk_llama_cpp/${sdk_tag}/sdk"
    )

    # For RISCV64, SDK has different naming
    if [ "$platform" = "riscv64" ]; then
        potential_paths+=(
            "/LocalRun/$(whoami)/local_builds/sdk_llama_cpp/${sdk_tag}/sdk_riscv64"
            "${CI_BUILDS_DIR}/sdk_llama_cpp/${sdk_tag}/sdk_riscv64"
            "/tmp/sdk_llama_cpp/${sdk_tag}/sdk_riscv64"
        )
    fi

    for path in "${potential_paths[@]}"; do
        if [ -d "$path" ]; then
            echo "$path"
            return 0
        fi
    done

    # If no existing SDK found, use default location
    local default_sdk_path="/LocalRun/$(whoami)/local_builds/sdk_llama_cpp/${sdk_tag}/sdk"
    if [ "$platform" = "riscv64" ]; then
        default_sdk_path="/LocalRun/$(whoami)/local_builds/sdk_llama_cpp/${sdk_tag}/sdk_riscv64"
    fi

    log_warn "No existing SDK found, will use: $default_sdk_path"
    echo "$default_sdk_path"
}

# Function to get platform suffix based on architecture
get_platform_suffix() {
    local platform="$1"

    # Use unified configuration
    local suffix=$(get_config_value ".platform_suffixes.${platform}")
    if [[ -n "$suffix" ]]; then
        echo "$suffix"
    else
        log_warning "Unknown platform: ${platform}, using generic naming"
        echo "linux-${platform}"
    fi
}

# Function to get latest release tag from artifactory
get_latest_release_tag() {
    local platform="$1"
    local sdk_tag="$2"
    local platform_suffix=$(get_platform_suffix "$platform")

    # Transform SDK_TAG from V2_SOFTWARE_master_202510082141 to sdk202509180241
    local sdk_tag_transformed=""
    if [[ "$sdk_tag" =~ V2_SOFTWARE_master_([0-9]+) ]]; then
        sdk_tag_transformed="sdk${BASH_REMATCH[1]}"
    else
        # Fallback: if pattern doesn't match, use original SDK_TAG
        sdk_tag_transformed="$sdk_tag"
    fi

    # Try to get the latest release tag using jf CLI if available
    if command -v jf >/dev/null 2>&1; then
        local latest_release=$(jf rt s "llama.cpp-release/llama-*-${sdk_tag_transformed}-bin-${platform_suffix}.zip" --sort-by=modified --sort-order=desc --limit=1 2>/dev/null | jq -r '.[] | .path' | head -1 2>/dev/null || echo "")

        if [ -n "$latest_release" ]; then
            local version=$(echo "$latest_release" | sed -n "s/.*llama-\([^-]*\)-${sdk_tag_transformed//\//\\\/}-bin-${platform_suffix//\//\\\/}\.zip.*/\1/p")
            if [ -n "$version" ]; then
                echo "$version"
                return 0
            else
                # Try a simpler extraction method
                version=$(basename "$latest_release" .zip | sed 's/llama-//' | sed "s/-${sdk_tag_transformed//\//\\\/}-bin-${platform_suffix//\//\\\/}.*//")
                if [ -n "$version" ]; then
                    echo "$version"
                    return 0
                fi
            fi
        fi
    fi

    # Fallback: try common version patterns
    log_warn "Could not determine latest version automatically, using fallback method..."
    local fallback_versions=("v1.0.0" "v0.9.0" "v0.8.0" "latest")

    for version in "${fallback_versions[@]}"; do
        local test_url="${RELEASE_ARTIFACTORY_BASE}/llama-${version}-${sdk_tag_transformed}-bin-${platform_suffix}.zip"
        log_info "Testing availability of version: $version"

        if curl --head --silent --fail "$test_url" >/dev/null 2>&1; then
            log_success "Found available version: $version"
            echo "$version"
            return 0
        fi
    done

    log_error "Could not find any available release version"
    return 1
}

# Function to download and extract release package
download_and_extract_release() {
    local platform="$1"
    local version="$2"
    local work_dir="$3"
    local sdk_tag="$4"

    local platform_suffix=$(get_platform_suffix "$platform")
    local release_filename="llama-${version}-${sdk_tag}-bin-${platform_suffix}.zip"
    local download_path="${work_dir}/${release_filename}"
    local extract_path="${work_dir}/extracted_release"

    log_info "Downloading release package..."
    log_info "Release filename: $release_filename"
    log_info "Work directory: $work_dir"

    # Create work directory
    mkdir -p "$work_dir"

    # Change to work directory for JFrog CLI
    log_info "Changing to work directory..."
    cd "$work_dir"
    log_info "Current directory: $(pwd)"

    # Download using JFrog CLI (we know this works from testing)
    log_info "Executing JFrog CLI download..."
    log_info "Command: jf rt dl \"llama.cpp-release/${release_filename}\" --flat=true"

    # Execute with explicit error handling
    if jf rt dl "llama.cpp-release/${release_filename}" --flat=true; then
        log_success "JFrog CLI download completed"
    else
        log_error "JFrog CLI download failed"
        return 1
    fi

    # Find the downloaded file
    local actual_download_path=""
    if [ -f "$download_path" ]; then
        actual_download_path="$download_path"
    else
        actual_download_path=$(find "$work_dir" -name "*.zip" -type f | head -1)
    fi

    # Verify download
    if [ -z "$actual_download_path" ] || [ ! -f "$actual_download_path" ]; then
        log_error "Downloaded file not found in: $work_dir"
        log_info "Files in work directory:"
        ls -la "$work_dir"
        return 1
    fi

    local file_size=$(du -h "$actual_download_path" | cut -f1)
    log_success "Downloaded successfully: $(basename "$actual_download_path") ($file_size)"

    # Extract the package
    log_info "Extracting release package..."
    mkdir -p "$extract_path"

    unzip -q "$actual_download_path" -d "$extract_path"
    log_success "Extraction completed successfully"

    # Find the release directory
    local release_dir=$(find "$extract_path" -type d -name "release" | head -1)
    if [ -z "$release_dir" ]; then
        log_warn "No 'release' directory found, using extracted directory"
        release_dir="$extract_path"
    fi

    log_success "Using binary directory: $release_dir"
    log_info "Contents of binary directory:"
    ls -la "$release_dir"

    echo "$release_dir"
}

# Function to setup Docker repo path
setup_docker_repo_path() {
    local sdk_path="$1"

    # Try to find docker directory relative to SDK path
    local potential_paths=(
        "${sdk_path}/../docker"
        "${sdk_path}/../../docker"
        "/LocalRun/$(whoami)/local_builds/sdk_llama_cpp/docker"
        "${CI_BUILDS_DIR}/sdk_llama_cpp/docker"
    )

    for path in "${potential_paths[@]}"; do
        local real_path=$(realpath "$path" 2>/dev/null || echo "$path")
        if [ -d "$real_path" ] && [ -f "$real_path/bash.sh" ]; then
            echo "$real_path"
            return 0
        fi
    done

    log_error "Could not find Docker utilities directory"
    log_error "Expected to find bash.sh in one of these locations:"
    for path in "${potential_paths[@]}"; do
        echo "  - $path"
    done
    return 1
}

# Function to run CI test
run_ci_test() {
    local binary_path="$1"
    local sdk_path="$2"
    local docker_platform="$3"
    local repo_path="$4"
    local local_model_path="$5"
    local docker_repo_path="$6"

    log_info "Setting up environment variables for CI test..."

    export BINARY_PATH="$binary_path"
    export SDK_PATH="$sdk_path"
    export DOCKER_PLATFORM="$docker_platform"
    export REPO_PATH="$repo_path"
    export LOCAL_MODEL_PATH="$local_model_path"
    export DOCKER_REPO_PATH="$docker_repo_path"

    log_info "Environment variables set:"
    log_info "  BINARY_PATH: $BINARY_PATH"
    log_info "  SDK_PATH: $SDK_PATH"
    log_info "  DOCKER_PLATFORM: $DOCKER_PLATFORM"
    log_info "  REPO_PATH: $REPO_PATH"
    log_info "  LOCAL_MODEL_PATH: $LOCAL_MODEL_PATH"
    log_info "  DOCKER_REPO_PATH: $DOCKER_REPO_PATH"
    echo ""

    log_info "Starting CI test execution..."

    # Run the CI test script
    local ci_test_script="${repo_path}/.dlci/ci_test.sh"
    if [ ! -f "$ci_test_script" ]; then
        log_error "CI test script not found: $ci_test_script"
        return 1
    fi

    "$ci_test_script"
}

# Cleanup function
cleanup() {
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log_info "Cleaning up temporary directory: $WORK_DIR"
        safe_rm -rf "$WORK_DIR"
    fi
}

# Cleanup function for errors (more verbose)
cleanup_on_error() {
    log_error "Script encountered an error, cleaning up..."
    if [ -n "$WORK_DIR" ] && [ -d "$WORK_DIR" ]; then
        log_info "Work directory contents before cleanup:"
        ls -la "$WORK_DIR" 2>/dev/null || true
        log_info "Cleaning up temporary directory: $WORK_DIR"
        safe_rm -rf "$WORK_DIR"
    fi
}

# Main function
main() {
    print_banner

    # Set up cleanup trap for errors
    trap cleanup_on_error ERR

    log_info "Starting automated CI test validation..."
    echo ""

    # Step 1: Detect platform
    log_info "Step 1: Detecting platform architecture..."
    DOCKER_PLATFORM=$(detect_platform)
    log_success "Detected platform: $DOCKER_PLATFORM"
    echo ""

    # Step 2: Set up repository path
    log_info "Step 2: Setting up repository path..."
    REPO_PATH=$(realpath "$(dirname "$0")/..")
    log_success "Repository path: $REPO_PATH"
    echo ""

    # Step 3: Set up SDK path
    log_info "Step 3: Setting up SDK path..."
    SDK_PATH=$(setup_sdk_path "$DEFAULT_SDK_TAG" "$DOCKER_PLATFORM")
    if [ -d "$SDK_PATH" ]; then
        log_info "Found existing SDK at: $SDK_PATH"
        log_success "SDK path: $SDK_PATH"
    else
        log_warn "SDK path does not exist: $SDK_PATH"
        log_success "SDK path (will be used): $SDK_PATH"
    fi
    echo ""

    # Step 4: Set up Docker repo path
    log_info "Step 4: Setting up Docker utilities path..."
    DOCKER_REPO_PATH=$(setup_docker_repo_path "$SDK_PATH")
    if [ -d "$DOCKER_REPO_PATH" ] && [ -f "$DOCKER_REPO_PATH/bash.sh" ]; then
        log_info "Found Docker utilities at: $DOCKER_REPO_PATH"
        log_success "Docker repo path: $DOCKER_REPO_PATH"
    else
        log_error "Docker utilities not found or invalid: $DOCKER_REPO_PATH"
        exit 1
    fi
    echo ""

    # Step 5: Set up model path
    log_info "Step 5: Setting up model path..."
    LOCAL_MODEL_PATH="$DEFAULT_LOCAL_MODEL_PATH"
    log_success "Model path: $LOCAL_MODEL_PATH"
    echo ""

    # Step 6: Get latest release version
    log_info "Step 6: Finding latest release version..."
    log_info "Fetching latest release information for platform: $DOCKER_PLATFORM"
    log_info "Using JFrog CLI to find latest release..."

    RELEASE_VERSION=$(get_latest_release_tag "$DOCKER_PLATFORM" "$DEFAULT_SDK_TAG")
    local version_exit_code=$?

    if [ $version_exit_code -ne 0 ] || [ -z "$RELEASE_VERSION" ]; then
        log_error "Failed to determine release version"
        log_error "Exit code: $version_exit_code"
        log_error "Version: '$RELEASE_VERSION'"
        exit 1
    fi

    log_success "Found latest release version: $RELEASE_VERSION"
    echo ""

    # Step 7: Download and extract release (inline implementation)
    log_info "Step 7: Downloading and extracting release package..."
    echo ""

    # Temporarily disable ERR trap
    trap - ERR

    # Inline download and extract logic
    local platform_suffix=$(get_platform_suffix "$DOCKER_PLATFORM")

    # Transform SDK_TAG from V2_SOFTWARE_master_202510082141 to sdk202509180241
    local sdk_tag_transformed=""
    if [[ "$DEFAULT_SDK_TAG" =~ V2_SOFTWARE_master_([0-9]+) ]]; then
        sdk_tag_transformed="sdk${BASH_REMATCH[1]}"
    else
        # Fallback: if pattern doesn't match, use original SDK_TAG
        sdk_tag_transformed="$DEFAULT_SDK_TAG"
    fi

    local release_filename="llama-${RELEASE_VERSION}-${sdk_tag_transformed}-bin-${platform_suffix}.zip"
    local download_path="${WORK_DIR}/${release_filename}"
    local extract_path="${WORK_DIR}/extracted_release"

    log_info "Release filename: $release_filename"
    log_info "Work directory: $WORK_DIR"

    # Create work directory
    mkdir -p "$WORK_DIR"

    # Change to work directory and download
    log_info "Changing to work directory..."
    cd "$WORK_DIR"
    log_info "Current directory: $(pwd)"

    # Download using JFrog CLI
    log_info "Executing JFrog CLI download..."
    log_info "Command: jf rt dl \"llama.cpp-release/${release_filename}\" --flat=true"

    if jf rt dl "llama.cpp-release/${release_filename}" --flat=true; then
        log_success "JFrog CLI download completed"
    else
        log_error "JFrog CLI download failed"
        cleanup_on_error
        exit 1
    fi

    # Find the downloaded file
    local actual_download_path=""
    if [ -f "$download_path" ]; then
        actual_download_path="$download_path"
    else
        actual_download_path=$(find "$WORK_DIR" -name "*.zip" -type f | head -1)
    fi

    # Verify download
    if [ -z "$actual_download_path" ] || [ ! -f "$actual_download_path" ]; then
        log_error "Downloaded file not found in: $WORK_DIR"
        log_info "Files in work directory:"
        ls -la "$WORK_DIR"
        cleanup_on_error
        exit 1
    fi

    local file_size=$(du -h "$actual_download_path" | cut -f1)
    log_success "Downloaded successfully: $(basename "$actual_download_path") ($file_size)"

    # Extract the package
    log_info "Extracting release package..."
    mkdir -p "$extract_path"

    if unzip -q "$actual_download_path" -d "$extract_path"; then
        log_success "Extraction completed successfully"
    else
        log_error "Failed to extract the zip file"
        cleanup_on_error
        exit 1
    fi

    # Find the release directory
    local release_dir=$(find "$extract_path" -type d -name "release" | head -1)
    if [ -z "$release_dir" ]; then
        log_warn "No 'release' directory found, using extracted directory"
        release_dir="$extract_path"
    fi

    BINARY_PATH="$release_dir"
    log_success "Binary path: $BINARY_PATH"
    log_info "Contents of binary directory:"
    ls -la "$BINARY_PATH"

    # Re-enable ERR trap
    trap cleanup_on_error ERR

    echo ""

    # Step 8: Run CI test
    log_info "Step 8: Running CI test..."
    echo "=========================================="
    run_ci_test "$BINARY_PATH" "$SDK_PATH" "$DOCKER_PLATFORM" "$REPO_PATH" "$LOCAL_MODEL_PATH" "$DOCKER_REPO_PATH"

    local test_exit_code=$?

    if [ $test_exit_code -eq 0 ]; then
        echo ""
        echo "=========================================="
        log_success "CI Test Validation completed successfully!"
        log_success "All tests passed for platform: $DOCKER_PLATFORM"
        log_success "Release version tested: $RELEASE_VERSION"
        echo "=========================================="

        # Clean up on success
        cleanup
    else
        echo ""
        echo "=========================================="
        log_error "CI Test Validation failed!"
        log_error "Test exit code: $test_exit_code"
        echo "=========================================="

        # Clean up on failure (with more verbose output)
        cleanup_on_error
        exit 1
    fi
}

# Script entry point
log_info "CI Test Validation Script started at: $(date)"
main "$@"
log_info "CI Test Validation Script completed at: $(date)"
