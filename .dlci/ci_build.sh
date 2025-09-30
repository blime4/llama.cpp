#!/bin/bash
#
# CI Build Script for Llama.cpp
#
# This script runs CI builds by launching Docker containers and executing
# compile_llama_cpp.sh to compile the llama.cpp project.
#
# Required Environment Variables:
#   SDK_PATH        - Path to the SDK directory (e.g., /path/to/sdk)
#   DOCKER_PLATFORM - Target platform (x86_64, aarch64, riscv64, loongarch64)
#
# Optional Environment Variables:
#   REPO_PATH       - Path to the repository (default: current directory)
#   DOCKER_REPO_PATH - Path to Docker utilities (default: auto-detected)
#   SDK_TAG         - SDK tag for version information (default: auto-detected)
#

set -e

# Load utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

# Default values
DEFAULT_REPO_PATH="$(pwd)"

# Function to print usage
print_usage() {
    echo "Usage: $0"
    echo ""
    echo "Required Environment Variables:"
    echo "  SDK_PATH         - Path to SDK directory"
    echo "  DOCKER_PLATFORM  - Target platform (x86_64, aarch64, riscv64, loongarch64)"
    echo ""
    echo "Optional Environment Variables:"
    echo "  REPO_PATH        - Repository path (default: current directory)"
    echo "  DOCKER_REPO_PATH - Docker utilities path (auto-detected if not set)"
    echo "  SDK_TAG          - SDK tag for version information (auto-detected if not set)"
    echo ""
    echo "Example:"
    echo "  export SDK_PATH=/path/to/sdk"
    echo "  export DOCKER_PLATFORM=x86_64"
    echo "  export REPO_PATH=/path/to/repository"
    echo "  ./ci_build.sh"
}

# Validate required environment variables
validate_env_vars() {
    local missing_vars=()

    if [ -z "$SDK_PATH" ]; then
        missing_vars+=("SDK_PATH")
    fi

    if [ -z "$DOCKER_PLATFORM" ]; then
        missing_vars+=("DOCKER_PLATFORM")
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
    if [ -z "$REPO_PATH" ]; then
        REPO_PATH="$DEFAULT_REPO_PATH"
        echo "[INFO] Using default REPO_PATH: $REPO_PATH"
    fi

    # Auto-detect and setup DOCKER_REPO_PATH if not set
    if [ -z "$DOCKER_REPO_PATH" ]; then
        if ! auto_setup_docker_repo "$SDK_PATH"; then
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

    # Auto-detect SDK_TAG if not set
    if [ -z "$SDK_TAG" ]; then
        # Try to extract SDK tag from SDK_PATH
        local sdk_basename=$(basename "$SDK_PATH")
        if [[ "$sdk_basename" =~ ^sdk-[0-9]{12}$ ]]; then
            SDK_TAG="$sdk_basename"
            echo "[INFO] Auto-detected SDK_TAG: $SDK_TAG"
        else
            echo "[WARNING] Cannot auto-detect SDK_TAG from SDK_PATH. Using default pattern."
            # Set a default pattern that compile_llama_cpp.sh can handle
            SDK_TAG="sdk-$(date +%Y%m%d%H%M)"
            echo "[INFO] Using generated SDK_TAG: $SDK_TAG"
        fi
    fi
}

# Validate paths exist
validate_paths() {
    local error_count=0

    if [ ! -d "$SDK_PATH" ]; then
        echo "[ERROR] SDK_PATH does not exist: $SDK_PATH"
        error_count=$((error_count + 1))
    fi

    if [ ! -d "$REPO_PATH" ]; then
        echo "[ERROR] REPO_PATH does not exist: $REPO_PATH"
        error_count=$((error_count + 1))
    fi

    if [ ! -f "$DOCKER_REPO_PATH/bash.sh" ]; then
        echo "[ERROR] Docker bash.sh not found: $DOCKER_REPO_PATH/bash.sh"
        error_count=$((error_count + 1))
    fi

    if [ ! -f "$REPO_PATH/.dlci/compile_llama_cpp.sh" ]; then
        echo "[ERROR] compile_llama_cpp.sh not found: $REPO_PATH/.dlci/compile_llama_cpp.sh"
        error_count=$((error_count + 1))
    fi

    # Check if SDK has required env.sh
    if [ ! -f "$SDK_PATH/env.sh" ]; then
        echo "[ERROR] SDK env.sh not found: $SDK_PATH/env.sh"
        error_count=$((error_count + 1))
    fi

    if [ $error_count -gt 0 ]; then
        echo "[ERROR] $error_count validation error(s) found. Aborting."
        exit 1
    fi
}

# Prepare Docker environment variables
prepare_docker_envs() {
    echo "[INFO] Preparing Docker environment variables..."

    # Export all required variables for Docker
    export SDK_PATH
    export DOCKER_PLATFORM
    export REPO_PATH
    export SDK_TAG
    export sdk_path="$SDK_PATH"

    # Collect environment variables matching the pattern
    DOCKER_ENVS=()
    while IFS='=' read -r key value; do
        DOCKER_ENVS+=(--env "${key}=${value}")
    done < <(env | grep -E '^(SDK_|CI_|sdk_|REPO_PATH|DOCKER_PLATFORM)')

    echo "[INFO] Docker environment variables prepared: ${#DOCKER_ENVS[@]} variables"
}

# Main execution function
main() {
    echo "======================================"
    echo "CI Build Script for Llama.cpp"
    echo "======================================"
    echo "[INFO] Starting CI build execution..."
    echo "[INFO] Platform: $DOCKER_PLATFORM"
    echo "[INFO] SDK Path: $SDK_PATH"
    echo "[INFO] SDK Tag: $SDK_TAG"
    echo "[INFO] Repository Path: $REPO_PATH"
    echo ""

    # Get Docker image for the platform
    DOCKER_IMAGE=$(get_docker_image)
    echo "[INFO] Using Docker image: $DOCKER_IMAGE"
    echo ""

    # Prepare environment variables
    prepare_docker_envs

    # Construct Docker command
    local docker_cmd=(
        "${DOCKER_REPO_PATH}/bash.sh"
        --env "DOCKER_PLATFORM=${DOCKER_PLATFORM}"
        "${DOCKER_ENVS[@]}"
        "$DOCKER_IMAGE"
        "./.dlci/compile_llama_cpp.sh"
    )

    # Handle platform-specific Docker options
    if [ "$DOCKER_PLATFORM" = "riscv64" ]; then
        # Insert platform flag after bash.sh
        docker_cmd=(
            "${DOCKER_REPO_PATH}/bash.sh"
            --platform "linux/riscv64"
            --env "DOCKER_PLATFORM=${DOCKER_PLATFORM}"
            "${DOCKER_ENVS[@]}"
            "$DOCKER_IMAGE"
            "./.dlci/compile_llama_cpp.sh"
        )
    fi

    echo "[INFO] Executing Docker command:"
    echo "  ${docker_cmd[*]}"
    echo ""

    # Change to repository directory
    cd "$REPO_PATH"

    # Execute the Docker command
    echo "[INFO] Starting build execution in Docker container..."
    "${docker_cmd[@]}"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "[SUCCESS] CI build completed successfully!"
        echo "======================================"

        # Show build artifacts information
        local build_dir="${REPO_PATH}/build_${DOCKER_PLATFORM}"
        if [ -d "$build_dir" ]; then
            echo "[INFO] Build artifacts created in: $build_dir"
            if [ -f "$build_dir/bin/llama-cli" ]; then
                echo "[INFO] Main executable: $build_dir/bin/llama-cli"
            fi
        fi

        # Show log information
        echo "[INFO] Build logs are available in: /LocalRun/$(whoami)/logs/llama_cpp_compile/"
    else
        echo ""
        echo "======================================"
        echo "[FAILED] CI build failed with exit code: $exit_code"
        echo "======================================"
        echo "[INFO] Check build logs in: /LocalRun/$(whoami)/logs/llama_cpp_compile/"
        exit $exit_code
    fi
}

# Script entry point
echo "[INFO] CI Build Script started at: $(date)"

# Validate environment
validate_env_vars
set_defaults
validate_paths

# Execute main function
main

echo "[INFO] CI Build Script completed at: $(date)"
