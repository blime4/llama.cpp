#!/bin/bash
set -e

# Source unified utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../../.dlci/utils.sh"

# Default SDK_TAG loaded from unified config, can be overridden by parameter
DEFAULT_SDK_TAG=$(get_sdk_tag)

# Show help information
show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -t, --sdk-tag TAG    Set SDK_TAG (default: $DEFAULT_SDK_TAG)"
    echo "  -c, --compile        Compile llama.cpp before entering docker"
    echo "  -cc, --re-compile    Re-compile (clean and build) llama.cpp before entering docker"
    echo "  --debug              Enable debug mode (Debug build + verbose runtime output)"
    echo "  -h, --help           Show this help message"
    echo "  -p, --platform       Set platform (supported: x86_64, aarch64, loongarch64, riscv64, android, default: auto-detect)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Enter docker with default SDK_TAG"
    echo "  $0 -t V2_SOFTWARE_master_202508201444 # Specify SDK_TAG"
    echo "  $0 -c                                 # Compile first, then enter docker"
    echo "  $0 -c --debug                         # Compile in debug mode, then enter docker"
    echo "  $0 -cc --debug                        # Clean and compile in debug mode, then enter docker"
    echo "  $0 -p android                         # Android cross-compilation environment"
    echo "  $0 -p android -c                      # Android cross-compilation with compile"
    echo "  $0 -t V2_SOFTWARE_master_202508201444 -c # Specify SDK_TAG and compile"
}

# Parse command line arguments
SDK_TAG="$DEFAULT_SDK_TAG"
COMPILE_FIRST=false
RE_COMPILE=false
DEBUG_MODE=false
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
        echo "[INFO] For Android cross-compilation, use: $0 -p android"
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
            if [[ "$2" =~ ^(x86_64|aarch64|loongarch64|riscv64|android)$ ]]; then
                DOCKER_PLATFORM="$2"
            else
                echo "[ERROR] Unsupported platform: $2"
                echo "Supported platforms: x86_64, aarch64, loongarch64, riscv64, android"
                exit 1
            fi
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
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
echo "Debug mode: $DEBUG_MODE"
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

# Get Docker images from unified configuration with network detection
DOCKER_IMAGE_X86=$(get_docker_image_with_network "x86_64" "dev_image")
DOCKER_IMAGE_AARCH64=$(get_docker_image_with_network "aarch64" "dev_image")
DOCKER_IMAGE_RISCV64=$(get_docker_image_with_network "riscv64" "dev_image")
DOCKER_IMAGE_LOONGAARCH64=$(get_docker_image_with_network "loongarch64" "dev_image")
DOCKER_IMAGE_ANDROID=$(get_docker_image_with_network "android" "dev_image")

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
    android)
        DOCKER_IMAGE_COMPILE="$DOCKER_IMAGE_ANDROID"
        echo "[INFO] Android cross-compilation mode enabled"
        echo "[INFO] Android binaries will be generated for ARM64 architecture"
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
export SDK_DIR="${SDK_WORKSPACE%/}/sdk"
echo "[Step 3] SDK path set to: $SDK_DIR"

# Check if SDK exists
if [ ! -d "$SDK_DIR" ]; then
    echo "[ERROR] SDK path does not exist: $SDK_DIR"
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

append_env_var() {
    local var_name="$1"
    if [ -n "${!var_name:-}" ]; then
        DOCKER_ENVS+=(--env "${var_name}=${!var_name}")
    fi
}

# Pass through optional ccache / Android overrides used by run_llama.sh
append_env_var "CCACHE_DIR"
append_env_var "CCACHE_BASEDIR"
append_env_var "CCACHE_MAXSIZE"
append_env_var "CCACHE_LOGFILE"

# Android NDK configuration variables
append_env_var "ANDROID_NDK_ROOT"
append_env_var "NDK_ROOT"
append_env_var "ANDROID_TOOLCHAIN_FILE"
append_env_var "tool_chain_cmake"

# Android API and platform configuration
append_env_var "ANDROID_API_LEVEL"
append_env_var "android_api_level"
append_env_var "ANDROID_PLATFORM"
append_env_var "API"
append_env_var "ANDROID_ABI"
append_env_var "TARGET"
append_env_var "ANDROID_CLANG_TRIPLE"

# Android compiler and linker flags
append_env_var "ANDROID_LINKER_FLAGS"
append_env_var "ANDROID_SHARED_LINKER_FLAGS"
append_env_var "ANDROID_C_FLAGS"
append_env_var "ANDROID_CXX_FLAGS"
append_env_var "LLVM"

# Support android-doc.md variable names for compatibility
append_env_var "DLGPU_X86_SDK_DIR"
append_env_var "llvm_devel_path"

# CUDA configuration
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    DOCKER_ENVS+=(--env "CUDA_VISIBLE_DEVICES=0")
else
    DOCKER_ENVS+=(--env "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}")
fi

echo "Number of Docker environment variables: ${#DOCKER_ENVS[@]}"

# Step 5: Compile if requested
if [ "$COMPILE_FIRST" = true ]; then
    echo "[Step 5] Compiling llama.cpp first..."
    if [ "$RE_COMPILE" = true ]; then
        echo "[Step 5] Cleaning local build cache..."
        rm -rf "${build_dir}"
        ccache_dir_cleanup="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"
        rm -rf "${ccache_dir_cleanup}"
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
        COMPILE_FLAGS=""
        if [ "$DEBUG_MODE" = true ]; then
            COMPILE_FLAGS="--debug"
        fi
        DISABLE_CCACHE_ANDROID="${DISABLE_CCACHE_ANDROID:-}" bash .dlci/compile_llama_cpp.sh $COMPILE_FLAGS
    else
        echo "Running compilation in docker..."
        # Build compile flags based on debug mode
        COMPILE_FLAGS=""
        if [ "$DEBUG_MODE" = true ]; then
            COMPILE_FLAGS="--debug"
        fi
        # Pass DISABLE_CCACHE_ANDROID and debug flags to docker environment
        if [ -n "${DISABLE_CCACHE_ANDROID}" ]; then
            ${DOCKER_REPO_PATH}/bash.sh --env DISABLE_CCACHE_ANDROID="${DISABLE_CCACHE_ANDROID}" "${DOCKER_ENVS[@]}" "${DOCKER_IMAGE_COMPILE}" ./.dlci/compile_llama_cpp.sh $COMPILE_FLAGS
        else
            ${DOCKER_REPO_PATH}/bash.sh "${DOCKER_ENVS[@]}" "${DOCKER_IMAGE_COMPILE}" ./.dlci/compile_llama_cpp.sh $COMPILE_FLAGS
        fi
    fi
    cd scripts/denglin
fi

# Step 6: Enter docker for development
echo "[Step 6] Entering docker for development..."

if [ "$DOCKER_PLATFORM" = "android" ]; then
    echo "=== Android Cross-Compilation Development Environment ==="
    echo "You are entering an Android cross-compilation environment."
    echo "Generated binaries are for Android ARM64 and cannot run in this container."
    echo ""
    echo "Setup commands:"
    echo "  - source $SDK_DIR/env.sh  # Load SDK environment"
    echo ""
    echo "Build commands (supports both variable styles):"
    echo "  # New style:"
    echo "  export SDK_DIR=\"$SDK_DIR\""
    echo "  ./run_llama.sh --platform android"
    echo ""
    echo "  # android-doc.md compatible style:"
    echo "  export llvm_devel_path=\"$SDK_DIR\""
    echo "  export android_api_level=25"
    echo "  export tool_chain_cmake=\"/opt/android-sdk-linux/ndk/25.2.9519653/build/cmake/android.toolchain.cmake\""
    echo "  ./run_llama.sh --platform android"
    echo ""
    echo "Development commands:"
    echo "  - Check binaries: file build_android/bin/* (verify ARM64 architecture)"
    echo "  - List binaries: ls -la build_android/bin/"
    echo "  - Static analysis: readelf -d build_android/bin/llama-cli"
    echo ""
    echo "Deploy to Android device:"
    echo "  - adb push build_android/bin/* /data/local/tmp/llama/"
    echo "  - adb push $SDK_DIR/lib/*.so /data/local/tmp/llama/"
    echo "  - adb shell 'cd /data/local/tmp/llama && LD_LIBRARY_PATH=. ./llama-cli --help'"
    echo ""
    echo "Note: test-backend-ops and direct binary execution will NOT work (cross-compilation)"
else
    echo "You can now interactively develop with llama.cpp in the docker container."
    echo "You need to source $SDK_DIR/env.sh to use the SDK."
    echo "One-click command: test-backend-ops -o FLASH_ATTN_EXT"
    echo "One-click command: test-backend-ops -o FLASH_ATTN_EXT -p \"(hsk=64.*hsv=64|hsk=128.*hsv=128|hsk=256.*hsv=256).*nb=1\""
    echo "One-click command: test-backend-ops -o GATED_LINEAR_ATTN"

    echo "Available commands:"
    echo "  - llama-cli --help"
    echo "  - llama-server --help"
    echo "  - llama-bench --help"
fi

echo "  - exit (to leave docker)"
echo ""

# Change to project root directory before entering docker
cd ../..

# Always use direct docker run for interactive development
echo "Entering docker container for interactive development..."
echo "Using direct docker run to ensure interactive session..."

# Platform-specific Docker configuration
if [ "$DOCKER_PLATFORM" = "android" ]; then
    echo "Configuring Docker for Android cross-compilation environment..."
    # Android cross-compilation doesn't need GPU runtime, but needs access to Android NDK
    # Add user mapping to fix permission issues and ensure HOME directory is writable
    exec docker run --rm -it                                   \
        --user $(id -u):$(id -g)                               \
        -v "$(pwd):/workspace"                                 \
        -v "${SDK_DIR}:/SDK_DIR"                             \
        -v "$(get_model_path):/models"                         \
        -w /workspace                                          \
        -v "${build_dir}:/workspace/build"                     \
        -e HOME=/workspace                                     \
        --network host                                         \
        "${DOCKER_ENVS[@]}"                                    \
        "${DOCKER_IMAGE_COMPILE}"                              \
        /bin/bash --rcfile scripts/denglin/docker_env.sh
else
    # Standard development environment with GPU runtime
    exec docker run --rm -it                                   \
        --runtime=dlrt -e DENGLIN_DEVICES=all                  \
        -v "$(pwd):/workspace"                                 \
        -v "${SDK_DIR}:/SDK_DIR"                               \
        -v "$(get_model_path):/models"                         \
        -w /workspace                                          \
        -v "${build_dir}:/workspace/build"                     \
        --network host                                         \
        "${DOCKER_ENVS[@]}"                                    \
        "${DOCKER_IMAGE_COMPILE}"                              \
        /bin/bash --rcfile scripts/denglin/docker_env.sh
fi

# This line will only be reached if docker command fails
echo "=== Development session ended ==="
