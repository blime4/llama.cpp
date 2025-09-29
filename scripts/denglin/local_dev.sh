#!/bin/bash
set -e

# Default SDK_TAG, can be passed as parameter
DEFAULT_SDK_TAG="V2_SOFTWARE_master_202509180241"

# Show help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --sdk-tag TAG    Set SDK_TAG (default: $DEFAULT_SDK_TAG)"
    echo "  -c, --compile        Compile llama.cpp before entering docker"
    echo "  -cc, --re-compile    Re-compile (clean and build) llama.cpp before entering docker"
    echo "  -h, --help           Show this help message"
    echo "  -p, --platform       Set platform (supported: x86_64, aarch64, loongarch64, riscv64, default: auto-detect)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Enter docker with default SDK_TAG"
    echo "  $0 -t V2_SOFTWARE_master_202508201444 # Specify SDK_TAG"
    echo "  $0 -c                                 # Compile first, then enter docker"
    echo "  $0 -t V2_SOFTWARE_master_202508201444 -c # Specify SDK_TAG and compile"
}

# Parse command line arguments
SDK_TAG="$DEFAULT_SDK_TAG"
COMPILE_FIRST=false
RE_COMPILE=false
# Auto-detect platform based on system architecture
DOCKER_PLATFORM=$(uname -m)
case "$DOCKER_PLATFORM" in
    x86_64)
        DOCKER_PLATFORM="x86_64"
        ;;
    aarch64|arm64)
        DOCKER_PLATFORM="aarch64"
        ;;
    loongarch64)
        DOCKER_PLATFORM="loongarch64"
        ;;
    riscv64)
        DOCKER_PLATFORM="riscv64"
        ;;
    *)
        echo "[WARNING] Unknown architecture: $DOCKER_PLATFORM, defaulting to x86_64"
        DOCKER_PLATFORM="x86_64"
        ;;
esac

while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--sdk-tag)
            SDK_TAG="$2"
            shift 2
            ;;
        -c|--compile)
            COMPILE_FIRST=true
            shift
            ;;
        -cc|--re-compile)
            RE_COMPILE=true
            COMPILE_FIRST=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        -p|--platform)
            if [[ "$2" =~ ^(x86_64|aarch64|loongarch64|riscv64)$ ]]; then
                DOCKER_PLATFORM="$2"
            else
                echo "[ERROR] Unsupported platform: $2"
                echo "Supported platforms: x86_64, aarch64, loongarch64, riscv64"
                exit 1
            fi
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

echo "=== Local Development Environment ==="
echo "SDK_TAG: $SDK_TAG"
echo "Compile first: $COMPILE_FIRST"
echo "Re-compile: $RE_COMPILE"
echo "DOCKER_PLATFORM: $DOCKER_PLATFORM"

# Set environment variables
export SDK_TAG="$SDK_TAG"
export LOCAL_BUILDS_DIR="/LocalRun/$(whoami)/local_builds"
export REPO_PATH="$(cd "$(dirname "$0")/../.." && pwd)"
export LOCAL_MODEL_PATH="/models"
export SDK_WORKSPACE="${LOCAL_BUILDS_DIR}/sdk_llama_cpp/${SDK_TAG}"
export DOCKER_REPO_PATH="${LOCAL_BUILDS_DIR}/sdk_llama_cpp/docker"
export DOCKER_PLATFORM="$DOCKER_PLATFORM"
build_dir=$(readlink -m ${REPO_PATH}/build_${DOCKER_PLATFORM})
echo "build_dir: $build_dir"

DOCKER_IMAGE_X86="ext-artifactory.denglin.com:8082/ci-docker-images/c-29:manylinux_2_28-gcc12-amd64-20250703-dev"
DOCKER_IMAGE_AARCH64="ext-artifactory.denglin.com:8082/ci-docker-images/c-25:manylinux_2_28-gcc12-aarch64-20250702-dev"
DOCKER_IMAGE_RISCV64="ext-artifactory.denglin.com:8082/ci-docker-images/c-37:ubuntu22.04-riscv-20250822-dev"
DOCKER_IMAGE_LOONGAARCH64="ext-artifactory.denglin.com:8082/ci-docker-images/c-41:manylinux_2_38-loongarch64-20250910"

case "$DOCKER_PLATFORM" in
    x86_64)
        DOCKER_IMAGE_COMPILE="$DOCKER_IMAGE_X86"
        ;;
    aarch64)
        DOCKER_IMAGE_COMPILE="$DOCKER_IMAGE_AARCH64"
        ;;
    riscv64)
        DOCKER_IMAGE_COMPILE="$DOCKER_IMAGE_RISCV64"
        ;;
    loongarch64)
        DOCKER_IMAGE_COMPILE="$DOCKER_IMAGE_LOONGAARCH64"
        ;;
    *)
        echo "[ERROR] unsupported DOCKER_PLATFORM: $DOCKER_PLATFORM"
        exit 1
        ;;
esac

echo "SDK_WORKSPACE: $SDK_WORKSPACE"
echo "DOCKER_IMAGE_COMPILE: $DOCKER_IMAGE_COMPILE"

# Check required script files
if [ ! -f "../../.dlci/download_and_unpack_sdk.sh" ]; then
    echo "[ERROR] Cannot find ../../.dlci/download_and_unpack_sdk.sh script"
    exit 1
fi

# Step 1: Sync submodules
echo "[Step 1] Syncing git submodules..."
git submodule sync && git submodule update --init --recursive

# Step 2: Download and unpack SDK
echo "[Step 2] Downloading and unpacking SDK..."
bash ../../.dlci/download_and_unpack_sdk.sh "${SDK_TAG}" "${SDK_WORKSPACE}"

# Step 3: Set SDK path
export sdk_path="${SDK_WORKSPACE}/sdk"
echo "[Step 3] SDK path set to: $sdk_path"

# Check if SDK exists
if [ ! -d "$sdk_path" ]; then
    echo "[ERROR] SDK path does not exist: $sdk_path"
    exit 1
fi

# Step 4: Prepare docker environment variables
echo "[Step 4] Preparing docker environment variables..."
export git_email=`git config user.email || echo "example@example.com"`
export git_name=`git config user.name || echo "example"`
DOCKER_ENVS=()
while IFS='=' read -r key value; do
    DOCKER_ENVS+=(--env "${key}=${value}")
done < <(env | grep -E '^(SDK_|CI_|sdk_|REPO_PATH|LOCAL_MODEL_PATH|git_|DOCKER_PLATFORM)')

echo "Number of Docker environment variables: ${#DOCKER_ENVS[@]}"

# Step 5: Compile if requested
if [ "$COMPILE_FIRST" = true ]; then
    echo "[Step 5] Compiling llama.cpp first..."
    if [ "$RE_COMPILE" = true ]; then
        echo "[Step 5] Cleaning local build cache..."
        rm -rf "${build_dir}"
        rm -rf /LocalRun/$(whoami)/cache/llama_cpp_ccache
    fi

    if [ ! -f "../../.dlci/compile_llama_cpp.sh" ]; then
        echo "[ERROR] Cannot find ../../.dlci/compile_llama_cpp.sh script"
        exit 1
    fi

    # Change to project root directory and compile
    cd ../..
    if [ ! -f "${DOCKER_REPO_PATH}/bash.sh" ]; then
        echo "[WARNING] Cannot find docker script: ${DOCKER_REPO_PATH}/bash.sh"
        echo "Trying to run compile script directly..."
        bash .dlci/compile_llama_cpp.sh
    else
        echo "Running compilation in docker..."
        ${DOCKER_REPO_PATH}/bash.sh "${DOCKER_ENVS[@]}" "${DOCKER_IMAGE_COMPILE}" ./.dlci/compile_llama_cpp.sh
    fi
    cd scripts/denglin
fi

# Step 6: Enter docker for development
echo "[Step 6] Entering docker for development..."
echo "You can now interactively develop with llama.cpp in the docker container."
echo "You need to source $sdk_path/env.sh to use the SDK."
echo "One-click command: test-backend-ops -o FLASH_ATTN_EXT"
echo "One-click command: test-backend-ops -o FLASH_ATTN_EXT -p \"(hsk=64.*hsv=64|hsk=128.*hsv=128|hsk=256.*hsv=256).*nb=1\""
echo "One-click command: test-backend-ops -o GATED_LINEAR_ATTN"

echo "Available commands:"
echo "  - llama-cli --help"
echo "  - llama-server --help"
echo "  - llama-bench --help"
echo "  - exit (to leave docker)"
echo ""

# Change to project root directory before entering docker
cd ../..

# Always use direct docker run for interactive development
echo "Entering docker container for interactive development..."
echo "Using direct docker run to ensure interactive session..."

# Use exec to replace current shell with docker bash
exec docker run --rm -it                                   \
    --runtime=dlrt -e DENGLIN_DEVICES=all                  \
    -v "$(pwd):/workspace"                                 \
    -v "${sdk_path}:/sdk_path"                             \
    -v /mars/aebox/LLM/model:/models                       \
    -w /workspace                                          \
    -v "${build_dir}:/workspace/build"                     \
    --network host                                         \
    "${DOCKER_ENVS[@]}"                                    \
    "${DOCKER_IMAGE_COMPILE}"                              \
    /bin/bash --rcfile scripts/denglin/docker_env.sh

# This line will only be reached if docker command fails
echo "=== Development session ended ==="
