#!/bin/bash

set -e

# Source Android compilation utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/android_compile_utils.sh"

# Function to clear ccache if compilation fails
clear_ccache_on_failure() {
    local arch="$1"
    local compile_log="$2"

    if [ "$arch" = "android" ]; then
        echo "[WARNING] Android compilation failed, clearing ccache cache..." | tee -a "$compile_log"
        ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"

        if [ -d "$ccache_dir" ]; then
            echo "[INFO] Removing ccache directory: $ccache_dir" | tee -a "$compile_log"
            rm -rf "$ccache_dir"
            mkdir -p "$ccache_dir"
            echo "[INFO] ccache cache cleared successfully" | tee -a "$compile_log"
        else
            echo "[INFO] ccache directory does not exist: $ccache_dir" | tee -a "$compile_log"
        fi

        # Also try clearing via ccache command if available
        if command -v ccache &> /dev/null; then
            echo "[INFO] Clearing ccache via ccache --clear..." | tee -a "$compile_log"
            ccache --clear >> "$compile_log" 2>&1 || true
            echo "[INFO] ccache command executed" | tee -a "$compile_log"
        fi
    fi
}

# Record start time for overall compilation timing
compile_start_time=$(date +%s)

# Create logs directory
if [ -n "$CI_PROJECT_DIR" ]; then
    # GitLab CI environment - save logs in project directory
    logs_dir="${CI_PROJECT_DIR}/logs/compile"
    mkdir -p "$logs_dir"
    echo "[INFO] CI environment detected, logs will be saved in: $logs_dir"
else
    # Local environment - use user-specific path
    logs_dir="/LocalRun/$(whoami)/logs/llama_cpp_compile"
    mkdir -p "$logs_dir"
fi

# Single log file
timestamp=$(date +%Y%m%d_%H%M%S)
compile_log="${logs_dir}/compile_${timestamp}.log"

# Function to output log content on error
output_log_on_error() {
    echo "==================== ERROR LOG CONTENT ====================" >&2
    echo "[DEBUG] Log file path: $compile_log" >&2
    if [ -f "$compile_log" ]; then
        echo "[DEBUG] Log file exists, size: $(du -h "$compile_log" | cut -f1)" >&2
        echo "[DEBUG] Outputting log content:" >&2
        cat "$compile_log"
    else
        echo "[ERROR] Log file not found: $compile_log" >&2
        echo "[DEBUG] Current directory: $(pwd)" >&2
        echo "[DEBUG] Available files in logs directory:" >&2
        ls -la "$(dirname "$compile_log")" 2>/dev/null || echo "Directory not found"
    fi
    echo "=============================================================" >&2
}

# Function to execute command with logging
exec_with_log() {
    local cmd="$1"
    eval "$cmd" >> "$compile_log" 2>&1
    return $?
}

echo "[INFO] Compilation started at: $(date)" | tee -a "$compile_log"
echo "[INFO] Logs will be saved to: $compile_log"

# Ensure SDK_DIR is set and valid
if [ -z "$SDK_DIR" ] || [ ! -d "$SDK_DIR" ]; then
  echo "[ERROR] SDK_DIR is not set or is not a valid directory. Current value: '$SDK_DIR'" | tee -a "$compile_log"
  output_log_on_error
  exit 1
fi

# Normalize SDK path to avoid double slashes and path issues
SDK_DIR=$(readlink -f "$SDK_DIR")
echo "[INFO] Normalized SDK path: $SDK_DIR" | tee -a "$compile_log"

echo "[INFO] Environment setup..." | tee -a "$compile_log"
env >> "$compile_log" 2>&1

# Ensure critical environment variables are set for Android cross-compilation
if [ "$ARCH" = "android" ]; then
    echo "[INFO] Setting up Android cross-compilation environment..." | tee -a "$compile_log"
    export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/opt/android-sdk-linux/ndk/25.2.9519653}"
    export ANDROID_API_LEVEL="${ANDROID_API_LEVEL:-25}"
    export ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
    # Set additional environment variables to match local environment
    export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk-linux}"
    export NDK_ROOT="${NDK_ROOT:-$ANDROID_NDK_ROOT}"
    echo "[INFO] ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT" | tee -a "$compile_log"
    echo "[INFO] ANDROID_API_LEVEL: $ANDROID_API_LEVEL" | tee -a "$compile_log"
    echo "[INFO] ANDROID_ABI: $ANDROID_ABI" | tee -a "$compile_log"
    echo "[INFO] ANDROID_SDK_ROOT: $ANDROID_SDK_ROOT" | tee -a "$compile_log"
fi

source ${SDK_DIR}/env.sh >> "$compile_log" 2>&1
# Set LIBRARY_PATH for dlcc compiler
export LIBRARY_PATH="${SDK_DIR}/lib:${LIBRARY_PATH}"
# Also set LD_LIBRARY_PATH to match local environment
export LD_LIBRARY_PATH="${SDK_DIR}/lib:${LD_LIBRARY_PATH}"
echo "LIBRARY_PATH set to: $LIBRARY_PATH" | tee -a "$compile_log"
echo "LD_LIBRARY_PATH set to: $LD_LIBRARY_PATH" | tee -a "$compile_log"
env >> "$compile_log" 2>&1

# Get the latest tag (if any), otherwise empty
# --exact returns tag only if HEAD is exactly at a tag, otherwise non-zero exit
# 2>/dev/null suppresses error output, || true ensures tag is empty if not found

echo "[INFO] Getting version information..." | tee -a "$compile_log"
tag=$(git describe --tags --exact 2>/dev/null || true)
commit=$(git rev-parse --short=8 HEAD)
# SDK_TAG may be unset in some CI jobs, so fall back to a placeholder rather than exiting due to set -e
sdk_num=$(echo "${SDK_TAG:-}" | grep -oE '[0-9]{12}' || true)
sdk_num=${sdk_num:-000000000000}
echo "[INFO] commit: $commit" | tee -a "$compile_log"

if [[ -n "$tag" ]]; then
    # Example: v2.7.0.dev20250312+dl-main-1
    # Extract base version (remove 'v' and everything after '+')
    base_version=$(echo "$tag" | sed -E 's/^v([0-9.]+\.dev[0-9]+).*/\1/')
    # Extract denglin_version (the part after '+', remove '-' and '_')
    denglin_version=$(echo "$tag" | grep -oE '\+.*' | sed 's/^+//' | sed 's/[-_]//g')
    echo "[INFO] base_version: $base_version" | tee -a "$compile_log"
    echo "[INFO] denglin_version: $denglin_version" | tee -a "$compile_log"
    export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
else
    # No tag, use main version from version.txt
    base_version=$(cat version.txt 2>/dev/null | sed 's/[^0-9.].*$//' || true)
    base_version=${base_version:-0.0.0}
    echo "[INFO] base_version: $base_version" | tee -a "$compile_log"
    denglin_version="git${commit}"
    echo "[INFO] denglin_version: $denglin_version" | tee -a "$compile_log"
    export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
fi

echo "[INFO] LLAMA_CPP_BUILD_VERSION: $LLAMA_CPP_BUILD_VERSION" | tee -a "$compile_log"

# Define a custom bin directory within SDK_DIR for our tools like the ccache wrapper.
# This avoids polluting the main PATH with the entire SDK_DIR.
CUSTOM_BIN_DIR="$SDK_DIR/custom_bin"

echo "[INFO] Setting up custom CUDA compiler wrapper..." | tee -a "$compile_log"

# Create the custom bin directory if it doesn't exist.
mkdir -p "$CUSTOM_BIN_DIR" || {
    echo "[ERROR] Failed to create custom bin directory at $CUSTOM_BIN_DIR." | tee -a "$compile_log"
    output_log_on_error
    exit 1
}

# Setup compiler wrapper based on DISABLE_CCACHE flag
if [ "$DISABLE_CCACHE" != "true" ]; then
    # Create a symlink named 'dlcc' in the custom bin directory, pointing to ccache.
    # This allows us to use ccache for CUDA compilation by setting CUDA_NVCC_EXECUTABLE to this symlink.
    ln -sf /usr/bin/ccache "$CUSTOM_BIN_DIR/dlcc" || {
      echo "[ERROR] Failed to create symlink for ccache at $CUSTOM_BIN_DIR/dlcc." | tee -a "$compile_log"
      exit 1
    }
    echo "[INFO] Using ccache wrapper for CUDA compilation" | tee -a "$compile_log"
else
    # When ccache is disabled, create a direct symlink to the SDK compiler
    # Find the actual dlcc path from the SDK
    ACTUAL_DLCC="$SDK_DIR/bin/dlcc"
    if [ ! -f "$ACTUAL_DLCC" ]; then
        # Fall back to clang++
        ACTUAL_DLCC="$SDK_DIR/bin/clang++"
    fi
    ln -sf "$ACTUAL_DLCC" "$CUSTOM_BIN_DIR/dlcc" || {
      echo "[ERROR] Failed to create symlink for direct compiler at $CUSTOM_BIN_DIR/dlcc." | tee -a "$compile_log"
      exit 1
    }
    echo "[INFO] Using direct compiler wrapper (ccache disabled) at $ACTUAL_DLCC" | tee -a "$compile_log"
fi

# Add the custom bin directory to the PATH.
# This makes 'dlcc' (and any other tools placed here) accessible if needed directly from the shell,
# though CUDA_NVCC_EXECUTABLE uses an absolute path.
export PATH="$CUSTOM_BIN_DIR:$PATH"

# correct ccache path in PATH (only if ccache is enabled).
if [ "$DISABLE_CCACHE" != "true" ]; then
    export PATH=$(echo "$PATH" | tr ':' '\n' | awk '/ccache/{ccache=$0; next} {print} END{if(ccache) print ccache}' | paste -sd:)
    echo "[INFO] which ccache: $(which ccache)" | tee -a "$compile_log"
fi
echo "[INFO] PATH: $PATH" | tee -a "$compile_log"

# Set CUDA_NVCC_EXECUTABLE to use the custom wrapper (absolute path).
# This tells nvcc (NVIDIA CUDA Compiler) to use our wrapper (ccache or direct compiler).
export CUDA_NVCC_EXECUTABLE="$CUSTOM_BIN_DIR/dlcc"

echo "[INFO] CUDA_NVCC_EXECUTABLE is set to: $CUDA_NVCC_EXECUTABLE" | tee -a "$compile_log"
echo "[INFO] DISABLE_CCACHE: $DISABLE_CCACHE" | tee -a "$compile_log"
echo "[INFO] Custom bin directory '$CUSTOM_BIN_DIR' added to PATH." >> "$compile_log"
# --- End of improved ccache setup ---

# --- Compile llama.cpp ---
sdk=$SDK_DIR
echo "[INFO] REPO_PATH: ${REPO_PATH}" | tee -a "$compile_log"
cd ${REPO_PATH}

ARCH=${DOCKER_PLATFORM}
build_dir=${REPO_PATH}/build_${ARCH}

echo "[INFO] ARCH value: '$ARCH'" | tee -a "$compile_log"
echo "[INFO] Build directory: $build_dir" | tee -a "$compile_log"

# Check if ccache should be disabled for Android
# This must be done AFTER ARCH is set
DISABLE_CCACHE=false
if [ "$ARCH" = "android" ] && [ "$DISABLE_CCACHE_ANDROID" = "true" ]; then
    echo "[INFO] DISABLE_CCACHE_ANDROID=true detected, disabling ccache" | tee -a "$compile_log"
    DISABLE_CCACHE=true
fi

# Setup ccache if not disabled
if [ "$DISABLE_CCACHE" != "true" ]; then
    echo "[INFO] Setting up ccache..." | tee -a "$compile_log"
    ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"
    ccache_max_size="${CCACHE_MAXSIZE:-20G}"
    ccache --set-config cache_dir="$ccache_dir" >> "$compile_log" 2>&1
    ccache --set-config max_size="$ccache_max_size" >> "$compile_log" 2>&1
    if [ -n "${CCACHE_BASEDIR:-}" ]; then
        ccache --set-config base_dir="$CCACHE_BASEDIR" >> "$compile_log" 2>&1
    fi
    ccache --zero-stats >> "$compile_log" 2>&1

    if [ -z "${CCACHE_LOGFILE:-}" ]; then
        export CCACHE_LOGFILE="/LocalRun/$(whoami)/cache/ccache.log"
    fi
else
    echo "[INFO] ccache is disabled" | tee -a "$compile_log"
fi

# Export the flag for child processes
export DISABLE_CCACHE

# Check if running on ARM platform
echo "[INFO] Configuring cmake for $ARCH platform..." | tee -a "$compile_log"

# Prepare ccache option for CMake
cmake_ccache_option="-DGGML_CCACHE=ON"
if [ "$DISABLE_CCACHE" = "true" ]; then
    cmake_ccache_option="-DGGML_CCACHE=OFF"
    echo "[INFO] CMake: Disabling GGML_CCACHE" | tee -a "$compile_log"
fi

# Propagate Kineto toggle from environment (default OFF if unset)
cmake_kineto_option="-DLLAMA_KINETO=${LLAMA_KINETO:-OFF}"
echo "[INFO] CMake: Using ${cmake_kineto_option}" | tee -a "$compile_log"

if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    # Execute CMake with logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=OFF \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_NATIVE=OFF \
        -DSDK_DIR=${sdk} \
        ${cmake_ccache_option} \
        ${cmake_kineto_option}"

        # -DGGML_CPU_ARM_ARCH=armv8-a \

    exec_with_log "$cmake_cmd"
    if [ $? -ne 0 ]; then
        echo "[ERROR] CMake configuration failed for ARM platform" | tee -a "$compile_log"
        echo "[ERROR] Check cmake logs for details: $compile_log" | tee -a "$compile_log"
        output_log_on_error
        exit 1
    fi
elif [ "$ARCH" = "loongarch64" ]; then
    echo "[INFO] Detected LoongArch64 platform" | tee -a "$compile_log"
    echo "[INFO] GGML_CPU_ALL_VARIANTS is disabled for LoongArch64 platform" | tee -a "$compile_log"

    # Execute CMake with logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=OFF \
        -DGGML_NATIVE=OFF \
        -DSDK_DIR=${sdk} \
        ${cmake_ccache_option} \
        ${cmake_kineto_option}"

    exec_with_log "$cmake_cmd"
elif [ "$ARCH" = "android" ]; then
    echo "[INFO] Using proven local_dev.sh Android compilation method" | tee -a "$compile_log"

    # Use the shared Android compilation function (extracted from successful local_dev.sh)
    if run_android_compilation "$sdk" "$REPO_PATH" "$ARCH" "$compile_log"; then
        echo "[INFO] Android compilation completed successfully using local_dev.sh method" | tee -a "$compile_log"
        # Skip the normal ninja build since it's already done in the function
        ninja_exit_code=0
    else
        echo "[ERROR] Android compilation failed using local_dev.sh method" | tee -a "$compile_log"
        # Clear ccache cache on failure
        clear_ccache_on_failure "$ARCH" "$compile_log"
        exit 1
    fi
elif [ "$ARCH" = "riscv64" ]; then
    echo "[INFO] Detected RISC-V 64-bit platform" | tee -a "$compile_log"

    # Execute CMake with logging
    cmake_cmd="/usr/local/bin/cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=ON \
        -DGGML_NATIVE=OFF \
        -DSDK_DIR=${sdk} \
        ${cmake_ccache_option} \
        ${cmake_kineto_option}"

    exec_with_log "$cmake_cmd"

    if [ $? -ne 0 ]; then
        echo "[ERROR] CMake configuration failed for RISC-V 64-bit platform" | tee -a "$compile_log"
        echo "[ERROR] Check cmake logs for details: $compile_log" | tee -a "$compile_log"
        output_log_on_error
        exit 1
    fi
    echo "[INFO] CMake configuration completed successfully for RISC-V 64-bit" | tee -a "$compile_log"
else
    echo "[INFO] Detected x86_64 platform" | tee -a "$compile_log"

    # Execute CMake with logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_VERBOSE_MAKEFILE=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CUDA_GRAPHS=ON \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=OFF \
        -DGGML_NATIVE=OFF \
        -DSDK_DIR=${sdk} \
        ${cmake_ccache_option} \
        ${cmake_kineto_option}"

    exec_with_log "$cmake_cmd"
    if [ $? -ne 0 ]; then
        echo "[ERROR] CMake configuration failed for x86_64 platform" | tee -a "$compile_log"
        echo "[ERROR] Check cmake logs for details: $compile_log" | tee -a "$compile_log"
        output_log_on_error
        exit 1
    fi
fi

echo "[INFO] CMake configuration completed successfully" | tee -a "$compile_log"

echo "[INFO] Checking ccache stats before build..." | tee -a "$compile_log"
ccache --show-stats >> "$compile_log" 2>&1

# Skip ninja build for Android (already done in shared function)
if [ "$ARCH" != "android" ]; then
    #cmake --build $build_dir --config Release -j 12
    cd $build_dir

    echo "[INFO] Starting ninja build with 12 parallel jobs..." | tee -a "$compile_log"
    echo "[INFO] Compile log: $compile_log"
    exec_with_log "ninja -v -j 12"

    ninja_exit_code=$?
    if [ $ninja_exit_code -ne 0 ]; then
        echo "[ERROR] Outputting full compile log($compile_log) below:"
        cat "$compile_log"
        exit 1
    fi
    echo "[INFO] Ninja build completed successfully" | tee -a "$compile_log"
else
    echo "[INFO] Android build already completed in shared function" | tee -a "$compile_log"
fi

# Record end time and calculate total compilation duration
compile_end_time=$(date +%s)
compile_duration=$((compile_end_time - compile_start_time))
compile_hours=$((compile_duration / 3600))
compile_minutes=$(((compile_duration % 3600) / 60))
compile_seconds=$((compile_duration % 60))

echo "[INFO] Compilation completed at: $(date)" | tee -a "$compile_log"
if [ $compile_hours -gt 0 ]; then
    echo "[INFO] Total compilation time: ${compile_hours}h ${compile_minutes}m ${compile_seconds}s (${compile_duration} seconds)" | tee -a "$compile_log"
elif [ $compile_minutes -gt 0 ]; then
    echo "[INFO] Total compilation time: ${compile_minutes}m ${compile_seconds}s (${compile_duration} seconds)" | tee -a "$compile_log"
else
    echo "[INFO] Total compilation time: ${compile_seconds}s" | tee -a "$compile_log"
fi

# Final ccache stats
echo "[INFO] Final ccache statistics:" | tee -a "$compile_log"
ccache --show-stats >> "$compile_log" 2>&1

# Log file sizes for reference
compile_log_size=$(du -h "$compile_log" | cut -f1)
echo "[INFO] Log file created:" | tee -a "$compile_log"
echo "[INFO]   Compile log: $compile_log ($compile_log_size)" | tee -a "$compile_log"

# Clear global error trap on successful completion
trap - ERR EXIT

echo "<> compile succeed ... <>" | tee -a "$compile_log"
