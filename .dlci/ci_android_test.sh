#!/bin/bash
#
# CI Android Test Script for Llama.cpp
#
# This script runs Android tests using redroid-dl container with NDK25 driver support.
# It launches a redroid-dl Docker container and executes tests inside the Android environment.
#
# Required Environment Variables:
#   SDK_DIR        - Path to the SDK directory (e.g., /path/to/sdk)
#   DOCKER_PLATFORM - Must be set to "android"
#   BINARY_PATH     - Path to the extracted release directory containing Android binaries
#
# Optional Environment Variables:
#   LOCAL_MODEL_PATH - Path to model files (default: /mars/aebox/LLM/model)
#   REPO_PATH       - Path to the repository (default: current directory)
#   DOCKER_REPO_PATH - Path to Docker utilities (default: auto-detected)
#   ANDROID_CONTAINER_NAME - Name for the redroid container (default: redroid-dl)
#   ANDROID_CONTAINER_PORT - Port mapping for redroid (default: 5555:5555)
#

set -e

# Load utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

# Default values
DEFAULT_LOCAL_MODEL_PATH="/mars/aebox/LLM/model"
DEFAULT_REPO_PATH="$(pwd)"
DEFAULT_ANDROID_CONTAINER_NAME="redroid-dl"
DEFAULT_ANDROID_CONTAINER_PORT="5555:5555"

# Function to print usage
print_usage() {
    echo "Usage: $0"
    echo ""
    echo "Required Environment Variables:"
    echo "  BINARY_PATH      - Path to extracted release directory with Android binaries"
    echo "  SDK_DIR         - Path to SDK directory"
    echo "  DOCKER_PLATFORM  - Must be 'android'"
    echo "  REPO_PATH        - Repository path (default: current directory)"
    echo "  LOCAL_MODEL_PATH - Path to model files (default: ${DEFAULT_LOCAL_MODEL_PATH})"
    echo ""
    echo "Optional Environment Variables:"
    echo "  DOCKER_REPO_PATH - Docker utilities path (auto-detected if not set)"
    echo "  ANDROID_CONTAINER_NAME - Redroid container name (default: ${DEFAULT_ANDROID_CONTAINER_NAME})"
    echo "  ANDROID_CONTAINER_PORT - Port mapping (default: ${DEFAULT_ANDROID_CONTAINER_PORT})"
    echo ""
    echo "Example:"
    echo "  export BINARY_PATH=/path/to/extracted/android/release"
    echo "  export SDK_DIR=/path/to/sdk"
    echo "  export DOCKER_PLATFORM=android"
    echo "  export REPO_PATH=/path/to/repository"
    echo "  export LOCAL_MODEL_PATH=/path/to/model"
    echo "  ./ci_android_test.sh"
}

# Validate required environment variables
validate_env_vars() {
    local missing_vars=()

    if [ -z "$BINARY_PATH" ]; then
        missing_vars+=("BINARY_PATH")
    fi

    if [ -z "$SDK_DIR" ]; then
        missing_vars+=("SDK_DIR")
    fi

    if [ -z "$DOCKER_PLATFORM" ]; then
        missing_vars+=("DOCKER_PLATFORM")
    fi

    if [ "$DOCKER_PLATFORM" != "android" ]; then
        echo "[ERROR] DOCKER_PLATFORM must be 'android' for this script"
        echo "[ERROR] Current value: $DOCKER_PLATFORM"
        exit 1
    fi

    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "[ERROR] Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        print_usage
        exit 1
    fi
}

# Set default values for optional variables
set_defaults() {
    if [ -z "$LOCAL_MODEL_PATH" ]; then
        LOCAL_MODEL_PATH="$DEFAULT_LOCAL_MODEL_PATH"
        echo "[INFO] Using default LOCAL_MODEL_PATH: $LOCAL_MODEL_PATH"
    fi

    if [ -z "$REPO_PATH" ]; then
        REPO_PATH="$DEFAULT_REPO_PATH"
        echo "[INFO] Using default REPO_PATH: $REPO_PATH"
    fi

    if [ -z "$ANDROID_CONTAINER_NAME" ]; then
        ANDROID_CONTAINER_NAME="$DEFAULT_ANDROID_CONTAINER_NAME"
        echo "[INFO] Using default ANDROID_CONTAINER_NAME: $ANDROID_CONTAINER_NAME"
    fi

    if [ -z "$ANDROID_CONTAINER_PORT" ]; then
        ANDROID_CONTAINER_PORT="$DEFAULT_ANDROID_CONTAINER_PORT"
        echo "[INFO] Using default ANDROID_CONTAINER_PORT: $ANDROID_CONTAINER_PORT"
    fi

    # Auto-detect and setup DOCKER_REPO_PATH if not set
    if [ -z "$DOCKER_REPO_PATH" ]; then
        if ! auto_setup_docker_repo "$SDK_DIR"; then
            echo "[ERROR] Failed to setup Docker repository"
            exit 1
        fi
    else
        # DOCKER_REPO_PATH is manually set, just ensure it's setup correctly
        if ! setup_docker_repo "$DOCKER_REPO_PATH"; then
            echo "[ERROR] Failed to setup Docker repository: $DOCKER_REPO_PATH"
            exit 1
        fi
    fi
}

# Validate paths exist
validate_paths() {
    local error_count=0

    if [ ! -d "$BINARY_PATH" ]; then
        echo "[ERROR] BINARY_PATH does not exist: $BINARY_PATH"
        error_count=$((error_count + 1))
    fi

    if [ ! -d "$SDK_DIR" ]; then
        echo "[ERROR] SDK_DIR does not exist: $SDK_DIR"
        error_count=$((error_count + 1))
    fi

    if [ ! -d "$REPO_PATH" ]; then
        echo "[ERROR] REPO_PATH does not exist: $REPO_PATH"
        error_count=$((error_count + 1))
    fi

    if [ ! -f "$REPO_PATH/.dlci/test_llama_cpp.sh" ]; then
        echo "[ERROR] test_llama_cpp.sh not found: $REPO_PATH/.dlci/test_llama_cpp.sh"
        error_count=$((error_count + 1))
    fi

    if [ $error_count -gt 0 ]; then
        echo "[ERROR] $error_count validation error(s) found. Aborting."
        exit 1
    fi
}

# Find NDK25 driver path
find_ndk25_driver() {
    echo "[INFO] Searching for NDK25 driver..."
    local ndk25_driver_path
    ndk25_driver_path=$(find "$REPO_PATH" -name "denglin-driver*ndk25.run" 2>/dev/null | head -1)

    if [ -z "$ndk25_driver_path" ]; then
        echo "[ERROR] NDK25 driver not found in workspace"
        echo "[ERROR] Expected pattern: denglin-driver*ndk25.run"
        exit 1
    fi

    echo "[INFO] Found NDK25 driver: $ndk25_driver_path"
    echo "$ndk25_driver_path"
}

# Extract NDK25 driver run file path from driver path
get_ndk25_driver_run() {
    local driver_path="$1"
    echo "$(echo "$driver_path" | awk -F '/' '{print $NF}')"
}

# Stop and remove existing redroid container
cleanup_redroid_container() {
    echo "[INFO] Checking for existing redroid container..."
    if docker ps -a --format '{{.Names}}' | grep -q "^${ANDROID_CONTAINER_NAME}$"; then
        echo "[INFO] Removing existing container: $ANDROID_CONTAINER_NAME"
        docker rm -f "$ANDROID_CONTAINER_NAME" || true
    fi
}

# Create and start redroid-dl container
start_redroid_container() {
    local ndk25_driver_path="$1"
    local ndk25_driver_run
    ndk25_driver_run=$(get_ndk25_driver_run "$ndk25_driver_path")

    echo "[INFO] ========================================"
    echo "[INFO] Starting redroid-dl container"
    echo "[INFO] ========================================"
    echo "[INFO] Container name: $ANDROID_CONTAINER_NAME"
    echo "[INFO] Port mapping: $ANDROID_CONTAINER_PORT"
    echo "[INFO] NDK25 driver: $ndk25_driver_run"
    echo "[INFO] SDK mount: $SDK_DIR:/sdk"
    echo "[INFO] Driver mount: $ndk25_driver_path:/$ndk25_driver_run"
    echo ""

    # Create redroid container
    docker run -itd \
        --name "$ANDROID_CONTAINER_NAME" \
        -p "$ANDROID_CONTAINER_PORT" \
        --privileged \
        -v "$ndk25_driver_path:/$ndk25_driver_run" \
        -v "$SDK_DIR:/sdk" \
        quay-containers.denglin.com/localtest/redroid:12.0.0_64only-denglin \
        > /dev/null 2>&1

    local ret=$?
    if [ $ret -eq 0 ]; then
        echo "[INFO] The container: $ANDROID_CONTAINER_NAME created successfully"
    else
        echo "[ERROR] The container created failed!"
        return 1
    fi

    # Verify container is running
    echo "[INFO] Verifying container status..."
    docker ps

    # Wait for container to be ready
    echo "[INFO] Waiting for container to initialize (60 seconds)..."
    sleep 60

    return 0
}

# Install NDK25 driver in redroid container
install_ndk25_driver() {
    local ndk25_driver_run="$1"
    local module="$2"

    echo "[INFO] ========================================"
    echo "[INFO] Installing NDK25 driver in container"
    echo "[INFO] ========================================"
    echo "[INFO] Driver file: $ndk25_driver_run"
    echo "[INFO] Module: $module"
    echo ""

    # Check if driver is already installed
    if docker exec -i "$ANDROID_CONTAINER_NAME" sh -c "/$ndk25_driver_run --no-knd && dlsmi" 2>&1 | grep -q "base_driver_otest"; then
        echo "[INFO] NDK25 driver already installed and working"
        return 0
    fi

    # Install the driver
    echo "[INFO] Installing driver..."
    docker exec -i "$ANDROID_CONTAINER_NAME" sh -c "/$ndk25_driver_run --no-knd && dlsmi"

    local ret=$?
    if [ $ret -eq 0 ]; then
        echo "[INFO] NDK25 driver installed successfully"
    else
        echo "[ERROR] Failed to install NDK25 driver"
        return 1
    fi

    return 0
}

# Run tests in redroid container
run_tests_in_container() {
    local module="$1"

    echo "[INFO] ========================================"
    echo "[INFO] Running tests in redroid container"
    echo "[INFO] ========================================"
    echo "[INFO] Module: $module"
    echo "[INFO] Binary path: $BINARY_PATH"
    echo "[INFO] SDK path: /sdk"
    echo ""

    # Set up environment and run tests
    docker exec -i "$ANDROID_CONTAINER_NAME" sh -c "
        export PATH=/sdk/bin:/sdk/base_driver_kit/bin:\$PATH && \
        export LD_LIBRARY_PATH=/sdk/lib:/sdk/base_driver_kit/lib:\$LD_LIBRARY_PATH && \
        cd /sdk/base_driver_kit/test/hal2 && \
        ./hal2_unit_test
    "

    local ret=$?

    if [ $ret -eq 0 ]; then
        echo ""
        echo "[INFO] ========================================"
        echo "[SUCCESS] Tests completed successfully!"
        echo "[INFO] ========================================"
    else
        echo ""
        echo "[ERROR] ========================================"
        echo "[ERROR] Tests failed with exit code: $ret"
        echo "[ERROR] ========================================"
        return 1
    fi

    return 0
}

# Main execution function
main() {
    echo "======================================"
    echo "CI Android Test Script for Llama.cpp"
    echo "======================================"
    echo "[INFO] Starting Android CI test execution..."
    echo "[INFO] Platform: $DOCKER_PLATFORM"
    echo "[INFO] Binary Path: $BINARY_PATH"
    echo "[INFO] SDK Path: $SDK_DIR"
    echo "[INFO] Repository Path: $REPO_PATH"
    echo "[INFO] Model Path: $LOCAL_MODEL_PATH"
    echo "[INFO] Container Name: $ANDROID_CONTAINER_NAME"
    echo ""

    # Find NDK25 driver
    local ndk25_driver_path
    ndk25_driver_path=$(find_ndk25_driver)
    local ndk25_driver_run
    ndk25_driver_run=$(get_ndk25_driver_run "$ndk25_driver_path")

    # Determine module name from driver
    local module="base_driver_otest"
    if [[ "$ndk25_driver_run" == *"base_driver_otest"* ]]; then
        module="base_driver_otest"
    fi

    # Clean up any existing container
    cleanup_redroid_container

    # Start redroid container
    if ! start_redroid_container "$ndk25_driver_path"; then
        echo "[ERROR] Failed to start redroid container"
        exit 1
    fi

    # Install NDK25 driver
    if ! install_ndk25_driver "$ndk25_driver_run" "$module"; then
        echo "[ERROR] Failed to install NDK25 driver"
        cleanup_redroid_container
        exit 1
    fi

    # Run tests
    if ! run_tests_in_container "$module"; then
        echo "[ERROR] Tests failed"
        cleanup_redroid_container
        exit 1
    fi

    # Clean up
    echo ""
    echo "[INFO] Cleaning up container..."
    cleanup_redroid_container

    echo ""
    echo "======================================"
    echo "[SUCCESS] Android CI test completed successfully!"
    echo "======================================"
}

# Script entry point
echo "[INFO] Android CI Test Script started at: $(date)"

# Validate environment
validate_env_vars
set_defaults
validate_paths

# Execute main function
main

echo "[INFO] Android CI Test Script completed at: $(date)"
