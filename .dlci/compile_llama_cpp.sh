#!/bin/bash

set -e

# Record start time for overall compilation timing
compile_start_time=$(date +%s)

# Create logs directory
# Use persistent path that survives CI job completion
if [ -n "$CI_PROJECT_DIR" ]; then
    # GitLab CI environment - save logs in persistent GitLab Runner directory
    # Use CI_JOB_ID to make logs unique per job
    logs_dir="/LocalRun/gitlab-runner/logs/llama_cpp_compile"
    mkdir -p "$logs_dir"
    echo "[INFO] CI environment detected, logs will be saved in persistent directory: $logs_dir"

    # Also create a copy in project directory for artifacts (if it works)
    project_logs_dir="${CI_PROJECT_DIR}/logs/compile"
    mkdir -p "$project_logs_dir"
else
    # Local environment - use user-specific path
    logs_dir="/LocalRun/$(whoami)/logs/llama_cpp_compile"
    mkdir -p "$logs_dir"
fi

# Log files
timestamp=$(date +%Y%m%d_%H%M%S)
compile_log="${logs_dir}/compile_${timestamp}.log"
summary_log="${logs_dir}/compile_summary_${timestamp}.log"

# In CI environment, also create project log files for artifacts
if [ -n "$CI_PROJECT_DIR" ] && [ -n "$project_logs_dir" ]; then
    project_compile_log="${project_logs_dir}/compile_${timestamp}.log"
    project_summary_log="${project_logs_dir}/compile_summary_${timestamp}.log"

    # Create symlinks or use tee to duplicate logs
    echo "[INFO] Will also save logs to project directory for artifacts"
fi

# Function to execute command with dual logging
exec_with_dual_log() {
    local cmd="$1"

    if [ -n "$project_compile_log" ]; then
        # Dual logging: write to both persistent and project logs
        # Use PIPESTATUS to capture the original command's exit code
        eval "$cmd" 2>&1 | tee -a "$compile_log" "$project_compile_log"
        return ${PIPESTATUS[0]}
    else
        # Single logging: write to persistent log only
        eval "$cmd" >> "$compile_log" 2>&1
        return $?
    fi
}

echo "[INFO] Compilation started at: $(date)" | tee -a "$compile_log"
echo "[INFO] Detailed logs will be saved to: $compile_log" | tee -a "$summary_log"
echo "[INFO] Summary will be saved to: $summary_log" | tee -a "$summary_log"

# In CI, also log project paths
if [ -n "$project_compile_log" ]; then
    echo "[INFO] Project artifacts logs: $project_compile_log" | tee -a "$summary_log"
    echo "[INFO] Project artifacts summary: $project_summary_log" | tee -a "$summary_log"

    # Initialize project log files
    echo "[INFO] Compilation started at: $(date)" >> "$project_compile_log"
    echo "[INFO] Detailed logs will be saved to: $project_compile_log" >> "$project_summary_log"
    echo "[INFO] Summary will be saved to: $project_summary_log" >> "$project_summary_log"
fi

# Ensure sdk_path is set and valid
if [ -z "$sdk_path" ] || [ ! -d "$sdk_path" ]; then
  echo "[ERROR] sdk_path is not set or is not a valid directory. Current value: '$sdk_path'" | tee -a "$compile_log"
  exit 1
fi

echo "[INFO] Environment setup..." | tee -a "$summary_log"
env >> "$compile_log" 2>&1
source ${sdk_path}/env.sh >> "$compile_log" 2>&1
env >> "$compile_log" 2>&1

echo "[INFO] Setting up ccache..." | tee -a "$summary_log"
ccache --set-config cache_dir=/LocalRun/$(whoami)/cache/llama_cpp_ccache >> "$compile_log" 2>&1
ccache --set-config max_size=20G >> "$compile_log" 2>&1
ccache --zero-stats >> "$compile_log" 2>&1

export CCACHE_LOGFILE=/LocalRun/$(whoami)/cache/ccache.log

# Get the latest tag (if any), otherwise empty
# --exact returns tag only if HEAD is exactly at a tag, otherwise non-zero exit
# 2>/dev/null suppresses error output, || true ensures tag is empty if not found

echo "[INFO] Getting version information..." | tee -a "$summary_log"
tag=$(git describe --tags --exact 2>/dev/null || true)
commit=$(git rev-parse --short=8 HEAD)
sdk_num=$(echo $SDK_TAG | grep -oE '[0-9]{12}')
echo "[INFO] commit: $commit" | tee -a "$summary_log"

if [[ -n "$tag" ]]; then
    # Example: v2.7.0.dev20250312+dl-main-1
    # Extract base version (remove 'v' and everything after '+')
    base_version=$(echo "$tag" | sed -E 's/^v([0-9.]+\.dev[0-9]+).*/\1/')
    # Extract denglin_version (the part after '+', remove '-' and '_')
    denglin_version=$(echo "$tag" | grep -oE '\+.*' | sed 's/^+//' | sed 's/[-_]//g')
    echo "[INFO] base_version: $base_version" | tee -a "$summary_log"
    echo "[INFO] denglin_version: $denglin_version" | tee -a "$summary_log"
    export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
else
    # No tag, use main version from version.txt
    base_version=$(cat version.txt | sed 's/[^0-9.].*$//')
    echo "[INFO] base_version: $base_version" | tee -a "$summary_log"
    denglin_version="git${commit}"
    echo "[INFO] denglin_version: $denglin_version" | tee -a "$summary_log"
    export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
fi

echo "[INFO] LLAMA_CPP_BUILD_VERSION: $LLAMA_CPP_BUILD_VERSION" | tee -a "$summary_log"

# Define a custom bin directory within sdk_path for our tools like the ccache wrapper.
# This avoids polluting the main PATH with the entire sdk_path.
CUSTOM_BIN_DIR="$sdk_path/custom_bin"

echo "[INFO] Setting up custom CUDA compiler wrapper..." | tee -a "$summary_log"

# Create the custom bin directory if it doesn't exist.
mkdir -p "$CUSTOM_BIN_DIR" || {
    echo "[ERROR] Failed to create custom bin directory at $CUSTOM_BIN_DIR." | tee -a "$compile_log"
    exit 1
}

# Create a symlink named 'dlcc' in the custom bin directory, pointing to ccache.
# This allows us to use ccache for CUDA compilation by setting CUDA_NVCC_EXECUTABLE to this symlink.
ln -sf /usr/bin/ccache "$CUSTOM_BIN_DIR/dlcc" || {
  echo "[ERROR] Failed to create symlink for ccache at $CUSTOM_BIN_DIR/dlcc." | tee -a "$compile_log"
  exit 1
}

# Add the custom bin directory to the PATH.
# This makes 'dlcc' (and any other tools placed here) accessible if needed directly from the shell,
# though CUDA_NVCC_EXECUTABLE uses an absolute path.
export PATH="$CUSTOM_BIN_DIR:$PATH"

# correct ccache path in PATH.
export PATH=$(echo "$PATH" | tr ':' '\n' | awk '/ccache/{ccache=$0; next} {print} END{if(ccache) print ccache}' | paste -sd:)
echo "[INFO] which ccache: $(which ccache)" | tee -a "$summary_log"
echo "[INFO] PATH: $PATH" | tee -a "$summary_log"

# Set CUDA_NVCC_EXECUTABLE to use the ccache symlink (absolute path).
# This tells nvcc (NVIDIA CUDA Compiler) to use our ccache-enabled wrapper.
export CUDA_NVCC_EXECUTABLE="$CUSTOM_BIN_DIR/dlcc"

echo "[INFO] CUDA_NVCC_EXECUTABLE is set to: $CUDA_NVCC_EXECUTABLE" | tee -a "$summary_log"
echo "[INFO] Custom bin directory '$CUSTOM_BIN_DIR' added to PATH." >> "$compile_log"
# --- End of improved ccache setup ---

# --- Compile llama.cpp ---
sdk=$sdk_path
echo "[INFO] REPO_PATH: ${REPO_PATH}" | tee -a "$summary_log"
cd ${REPO_PATH}

ARCH=${DOCKER_PLATFORM}
build_dir=${REPO_PATH}/build_${ARCH}

echo "[INFO] ARCH value: '$ARCH'" | tee -a "$summary_log"
echo "[INFO] Build directory: $build_dir" | tee -a "$summary_log"

# Check if running on ARM platform
echo "[INFO] Configuring cmake for $ARCH platform..." | tee -a "$summary_log"
if [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    echo "[INFO] Detected ARM platform, setting GGML_CPU_ARM_ARCH=armv8-a" | tee -a "$summary_log"

    # Execute CMake with dual logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CUDA_GRAPHS=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=OFF \
        -DGGML_CPU_ARM_ARCH=armv8-a \
        -DGGML_NATIVE=OFF \
        -DSDK_DIR=${sdk}"

    exec_with_dual_log "$cmake_cmd"
elif [ "$ARCH" = "loongarch64" ]; then
    echo "[INFO] Detected LoongArch64 platform" | tee -a "$summary_log"

    # Execute CMake with dual logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_CUDA_GRAPHS=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=OFF \
        -DSDK_DIR=${sdk}"

    exec_with_dual_log "$cmake_cmd"
elif [ "$ARCH" = "riscv64" ]; then
    echo "[INFO] Detected RISC-V 64-bit platform" | tee -a "$summary_log"

    # Execute CMake with dual logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=OFF \
        -DGGML_CUDA_GRAPHS=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=ON \
        -DGGML_NATIVE=OFF \
        -DSDK_DIR=${sdk}"

    exec_with_dual_log "$cmake_cmd"

    if [ $? -ne 0 ]; then
        echo "[ERROR] CMake configuration failed for RISC-V 64-bit platform" | tee -a "$summary_log"
        echo "[ERROR] Check cmake logs for details: $compile_log" | tee -a "$summary_log"
        exit 1
    fi
    echo "[INFO] CMake configuration completed successfully for RISC-V 64-bit" | tee -a "$summary_log"
else
    echo "[INFO] Detected x86_64 platform" | tee -a "$summary_log"

    # Execute CMake with dual logging
    cmake_cmd="cmake -G Ninja -B ${build_dir} \
        -DGGML_DLCU=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DGGML_CUDA_GRAPHS=OFF \
        -DLLAMA_CURL=OFF \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DGGML_CUDA_FA=ON \
        -DGGML_CUDA_FA_ALL_QUANTS=ON \
        -DGGML_RVV=OFF \
        -DSDK_DIR=${sdk}"

    exec_with_dual_log "$cmake_cmd"
fi

cmake_exit_code=$?
if [ $cmake_exit_code -ne 0 ]; then
    echo "[ERROR] CMake configuration failed with exit code: $cmake_exit_code" | tee -a "$summary_log"
    echo "[ERROR] Check detailed logs in: $compile_log" | tee -a "$summary_log"
    exit 1
fi
echo "[INFO] CMake configuration completed successfully" | tee -a "$summary_log"

echo "[INFO] Checking ccache stats before build..." | tee -a "$summary_log"
ccache --show-stats >> "$compile_log" 2>&1

#cmake --build $build_dir --config Release -j 12
cd $build_dir

echo "[INFO] Starting ninja build with 12 parallel jobs..." | tee -a "$summary_log"
exec_with_dual_log "ninja -j 12"

ninja_exit_code=$?
if [ $ninja_exit_code -ne 0 ]; then
    echo "[ERROR] Ninja build failed with exit code: $ninja_exit_code" | tee -a "$summary_log"
    echo "[ERROR] Check detailed logs in: $compile_log" | tee -a "$summary_log"
    exit 1
fi
echo "[INFO] Ninja build completed successfully" | tee -a "$summary_log"

# Record end time and calculate total compilation duration
compile_end_time=$(date +%s)
compile_duration=$((compile_end_time - compile_start_time))
compile_hours=$((compile_duration / 3600))
compile_minutes=$(((compile_duration % 3600) / 60))
compile_seconds=$((compile_duration % 60))

echo "[INFO] Compilation completed at: $(date)" | tee -a "$summary_log"
if [ $compile_hours -gt 0 ]; then
    echo "[INFO] Total compilation time: ${compile_hours}h ${compile_minutes}m ${compile_seconds}s (${compile_duration} seconds)" | tee -a "$summary_log"
elif [ $compile_minutes -gt 0 ]; then
    echo "[INFO] Total compilation time: ${compile_minutes}m ${compile_seconds}s (${compile_duration} seconds)" | tee -a "$summary_log"
else
    echo "[INFO] Total compilation time: ${compile_seconds}s" | tee -a "$summary_log"
fi

# Final ccache stats
echo "[INFO] Final ccache statistics:" | tee -a "$summary_log"
ccache --show-stats >> "$summary_log" 2>&1

# Log file sizes for reference
compile_log_size=$(du -h "$compile_log" | cut -f1)
summary_log_size=$(du -h "$summary_log" | cut -f1)
echo "[INFO] Log files created:" | tee -a "$summary_log"
echo "[INFO]   Detailed log: $compile_log ($compile_log_size)" | tee -a "$summary_log"
echo "[INFO]   Summary log: $summary_log ($summary_log_size)" | tee -a "$summary_log"

echo "<> compile succeed ... <>" | tee -a "$summary_log"
