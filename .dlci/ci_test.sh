#!/bin/bash
#
# CI Test Script for Llama.cpp
#
# This script runs CI tests using pre-built binaries from release packages.
# It launches Docker containers and executes test_llama_cpp.sh in CI mode.
#
# Required Environment Variables:
#   BINARY_PATH     - Path to the extracted release directory containing binaries
#   SDK_PATH        - Path to the SDK directory (e.g., /path/to/sdk)
#   DOCKER_PLATFORM - Target platform (x86_64, aarch64, riscv64, loongarch64)
#
# Optional Environment Variables:
#   LOCAL_MODEL_PATH - Path to model files (default: /mars/aebox/LLM/model)
#   REPO_PATH       - Path to the repository (default: current directory)
#   DOCKER_REPO_PATH - Path to Docker utilities (default: auto-detected)
#

set -e

# Default values
DEFAULT_LOCAL_MODEL_PATH="/mars/aebox/LLM/model"
DEFAULT_REPO_PATH="$(pwd)"

# Function to print usage
print_usage() {
    echo "Usage: $0"
    echo ""
    echo "Required Environment Variables:"
    echo "  BINARY_PATH      - Path to extracted release directory with binaries"
    echo "  SDK_PATH         - Path to SDK directory"
    echo "  DOCKER_PLATFORM  - Target platform (x86_64, aarch64, riscv64, loongarch64)"
    echo "  REPO_PATH        - Repository path (default: current directory)"
    echo "  LOCAL_MODEL_PATH - Path to model files (default: ${DEFAULT_LOCAL_MODEL_PATH})"
    echo ""
    echo "Optional Environment Variables:"
    echo "  DOCKER_REPO_PATH - Docker utilities path (auto-detected if not set)"
    echo ""
    echo "Example:"
    echo "  export BINARY_PATH=/path/to/extracted/release"
    echo "  export SDK_PATH=/path/to/sdk"
    echo "  export DOCKER_PLATFORM=x86_64"
    echo "  export REPO_PATH=/path/to/repository"
    echo "  export LOCAL_MODEL_PATH=/path/to/model"
    echo "  ./ci_test.sh"
}

# Validate required environment variables
validate_env_vars() {
    local missing_vars=()

    if [ -z "$BINARY_PATH" ]; then
        missing_vars+=("BINARY_PATH")
    fi

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
    if [ -z "$LOCAL_MODEL_PATH" ]; then
        LOCAL_MODEL_PATH="$DEFAULT_LOCAL_MODEL_PATH"
        echo "[INFO] Using default LOCAL_MODEL_PATH: $LOCAL_MODEL_PATH"
    fi

    if [ -z "$REPO_PATH" ]; then
        REPO_PATH="$DEFAULT_REPO_PATH"
        echo "[INFO] Using default REPO_PATH: $REPO_PATH"
    fi

    # Auto-detect DOCKER_REPO_PATH if not set
    if [ -z "$DOCKER_REPO_PATH" ]; then
        # Try to find docker directory relative to SDK_PATH
        local potential_docker_path="${SDK_PATH}/../docker"
        if [ -d "$potential_docker_path" ] && [ -f "$potential_docker_path/bash.sh" ]; then
            DOCKER_REPO_PATH="$(realpath "$potential_docker_path")"
            echo "[INFO] Auto-detected DOCKER_REPO_PATH: $DOCKER_REPO_PATH"
        else
            echo "[ERROR] Cannot auto-detect DOCKER_REPO_PATH. Please set it manually."
            echo "[INFO] Expected to find bash.sh in: $potential_docker_path"
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

    if [ ! -f "$REPO_PATH/.dlci/test_llama_cpp.sh" ]; then
        echo "[ERROR] test_llama_cpp.sh not found: $REPO_PATH/.dlci/test_llama_cpp.sh"
        error_count=$((error_count + 1))
    fi

    if [ $error_count -gt 0 ]; then
        echo "[ERROR] $error_count validation error(s) found. Aborting."
        exit 1
    fi
}

# Get Docker image based on platform
get_docker_image() {
    case "$DOCKER_PLATFORM" in
        x86_64)
            echo "ext-artifactory.denglin.com:8082/ci-docker-images/c-29:manylinux_2_28-gcc12-amd64-20250703"
            ;;
        aarch64)
            echo "ext-artifactory.denglin.com:8082/ci-docker-images/c-25:manylinux_2_28-gcc12-aarch64-20250702"
            ;;
        riscv64)
            echo "ext-artifactory.denglin.com:8082/ci-docker-images/c-37:ubuntu22.04-riscv-20250822"
            ;;
        loongarch64)
            echo "ext-artifactory.denglin.com:8082/ci-docker-images/c-41:manylinux_2_38-loongarch64-20250910"
            ;;
        *)
            echo "[ERROR] Unsupported DOCKER_PLATFORM: $DOCKER_PLATFORM"
            echo "[INFO] Supported platforms: x86_64, aarch64, riscv64, loongarch64"
            exit 1
            ;;
    esac
}

# Prepare Docker environment variables
prepare_docker_envs() {
    echo "[INFO] Preparing Docker environment variables..."

    # Export all required variables for Docker
    export BINARY_PATH
    export SDK_PATH
    export DOCKER_PLATFORM
    export LOCAL_MODEL_PATH
    export REPO_PATH
    export sdk_path="$SDK_PATH"

    # Collect environment variables matching the pattern
    DOCKER_ENVS=()
    while IFS='=' read -r key value; do
        DOCKER_ENVS+=(--env "${key}=${value}")
    done < <(env | grep -E '^(SDK_|CI_|sdk_|REPO_PATH|LOCAL_MODEL_PATH|BINARY_PATH|DOCKER_PLATFORM)')

    echo "[INFO] Docker environment variables prepared: ${#DOCKER_ENVS[@]} variables"
}

# Main execution function
main() {
    echo "======================================"
    echo "CI Test Script for Llama.cpp"
    echo "======================================"
    echo "[INFO] Starting CI test execution..."
    echo "[INFO] Platform: $DOCKER_PLATFORM"
    echo "[INFO] Binary Path: $BINARY_PATH"
    echo "[INFO] SDK Path: $SDK_PATH"
    echo "[INFO] Repository Path: $REPO_PATH"
    echo "[INFO] Model Path: $LOCAL_MODEL_PATH"
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
        "./.dlci/test_llama_cpp.sh"
        "--ci-test"
        "--binary-path" "$BINARY_PATH"
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
            "./.dlci/test_llama_cpp.sh"
            "--ci-test"
            "--binary-path" "$BINARY_PATH"
        )
    fi

    echo "[INFO] Executing Docker command:"
    echo "  ${docker_cmd[*]}"
    echo ""

    # Change to repository directory
    cd "$REPO_PATH"

    # Execute the Docker command
    echo "[INFO] Starting test execution in Docker container..."
    "${docker_cmd[@]}"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "[SUCCESS] CI test completed successfully!"
        echo "======================================"
    else
        echo ""
        echo "======================================"
        echo "[FAILED] CI test failed with exit code: $exit_code"
        echo "======================================"
        exit $exit_code
    fi
}

# Script entry point
echo "[INFO] CI Test Script started at: $(date)"

# Validate environment
validate_env_vars
set_defaults
validate_paths

# Execute main function
main

echo "[INFO] CI Test Script completed at: $(date)"
