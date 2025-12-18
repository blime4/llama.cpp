#!/bin/bash

# Android compilation utilities - shared between local_dev.sh and CI/CD
# This extracts the successful compilation logic from local_dev.sh

# Source utility functions (needed for get_sdk_tag, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

# Function to set up Android compilation environment
setup_android_compile_env() {
    local SDK_DIR="$1"
    local arch="$2"

    echo "[INFO] Setting up Android compilation environment..."

    # Set up Android environment variables (matching local_dev.sh)
    export ANDROID_NDK_ROOT="${ANDROID_NDK_ROOT:-/opt/android-sdk-linux/ndk/25.2.9519653}"
    export ANDROID_API_LEVEL="${ANDROID_API_LEVEL:-25}"
    export ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
    export ANDROID_SDK_ROOT="${ANDROID_SDK_ROOT:-/opt/android-sdk-linux}"
    export NDK_ROOT="${NDK_ROOT:-$ANDROID_NDK_ROOT}"

    # Set SDK paths
    export SDK_TAG=$(get_sdk_tag)
    export REPO_PATH="${REPO_PATH:-$(pwd)}"
    export DOCKER_PLATFORM="$arch"

    # Set library paths (matching local environment)
    export LIBRARY_PATH="${SDK_DIR}/lib:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${SDK_DIR}/lib:${LD_LIBRARY_PATH}"

    echo "[INFO] Android environment configured:"
    echo "[INFO]   ANDROID_NDK_ROOT: $ANDROID_NDK_ROOT"
    echo "[INFO]   ANDROID_API_LEVEL: $ANDROID_API_LEVEL"
    echo "[INFO]   ANDROID_ABI: $ANDROID_ABI"
    echo "[INFO]   SDK_DIR: $SDK_DIR"
    echo "[INFO]   DOCKER_PLATFORM: $DOCKER_PLATFORM"
}

# Function to configure Android CMake flags (based on successful local compilation)
get_android_cmake_flags() {
    local SDK_DIR="$1"
    local build_dir="$2"

    # Android configuration (matching local_dev.sh success pattern)
    local android_target="aarch64-linux-android"
    local android_toolchain_file="/opt/android-sdk-linux/ndk/25.2.9519653/build/cmake/android.toolchain.cmake"
    local android_sysroot="/opt/android-sdk-linux/ndk/25.2.9519653/toolchains/llvm/prebuilt/linux-x86_64/sysroot"

    # SDK paths for dual SDK configuration
    local sdk_aarch64="${SDK_DIR}_aarch64"
    if [ ! -d "$sdk_aarch64" ]; then
        sdk_aarch64="${SDK_DIR%/*}/sdk_aarch64"
    fi

    # Android-specific compiler flags (minimal, proven to work)
    local android_c_flags="-march=armv8-a -Wno-macro-redefined"
    local android_cxx_flags="-march=armv8-a -Wno-macro-redefined -Wno-unused-command-line-argument"

    # Android-specific linker flags
    local android_lib_path="${android_sysroot}/usr/lib/${android_target}/${ANDROID_API_LEVEL}"
    local android_linker_flags="-L${android_lib_path} -latomic -Wl,--allow-shlib-undefined -Wl,--unresolved-symbols=ignore-all"

    # CUDA compiler setup
    local cuda_nvcc_executable="${SDK_DIR}/custom_bin/dlcc"

    # Check DISABLE_CCACHE from global scope and set CMake option accordingly
    local cmake_ccache_option="-DGGML_CCACHE=ON"
    if [ "$DISABLE_CCACHE" = "true" ]; then
        cmake_ccache_option="-DGGML_CCACHE=OFF"
    fi

    local cmake_kineto_option="-DLLAMA_KINETO=${LLAMA_KINETO:-OFF}"

    # Generate CMake command (matching successful local pattern)
    cat << EOF
cmake -G Ninja -B ${build_dir} \\
    -DCMAKE_TOOLCHAIN_FILE=${android_toolchain_file} \\
    -DANDROID_ABI=${ANDROID_ABI} \\
    -DANDROID_PLATFORM=android-${ANDROID_API_LEVEL} \\
    -DANDROID_NDK=${ANDROID_NDK_ROOT} \\
    -DCMAKE_C_FLAGS="${android_c_flags}" \\
    -DCMAKE_CXX_FLAGS="${android_cxx_flags}" \\
    -DCMAKE_EXE_LINKER_FLAGS="${android_linker_flags}" \\
    -DCMAKE_SHARED_LINKER_FLAGS="${android_linker_flags}" \\
    -DGGML_DLCU=ON \\
    -DCMAKE_BUILD_TYPE=Release \\
    -DGGML_BACKEND_DL=ON \\
    -DGGML_CPU_ALL_VARIANTS=ON \\
    -DGGML_OPENMP=OFF \\
    -DGGML_LLAMAFILE=OFF \\
    -DGGML_INTERNAL_MATMUL_INT8=OFF \\
    -DGGML_CUDA_GRAPHS=ON \\
    -DLLAMA_CURL=OFF \\
    -DCURL_FOUND=FALSE \\
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \\
    -DGGML_CUDA_FA=ON \\
    -DGGML_CUDA_FA_ALL_QUANTS=ON \\
    -DGGML_RVV=OFF \\
    -DGGML_NATIVE=OFF \\
    -DSDK_DIR=${SDK_DIR} \\
    -DSDK_AARCH64_DIR=${sdk_aarch64} \\
    -DCMAKE_IGNORE_PATH="${SDK_DIR}/include/crt;${SDK_DIR}/lib/clang" \\
    -DCUDA_NVCC_EXECUTABLE=${cuda_nvcc_executable} \\
    -DCMAKE_CUDA_COMPILER=${cuda_nvcc_executable} \\
    -DCMAKE_FIND_ROOT_PATH_MODE_LIBRARY=BOTH \\
    -DCMAKE_FIND_ROOT_PATH_MODE_INCLUDE=BOTH \\
    ${cmake_ccache_option} \\
    ${cmake_kineto_option}
EOF
}

# Function to run Android compilation (extracted from successful local_dev.sh pattern)
run_android_compilation() {
    local SDK_DIR="$1"
    local REPO_PATH="$2"
    local arch="$3"
    local compile_log="$4"

    echo "[INFO] Starting Android compilation with proven local_dev.sh method..." | tee -a "$compile_log"

    # Setup environment
    setup_android_compile_env "$SDK_DIR" "$arch"

    # Source SDK environment
    source "${SDK_DIR}/env.sh" >> "$compile_log" 2>&1

    # Setup ccache (matching local_dev.sh)
    # Check if ccache is disabled
    if [ "$DISABLE_CCACHE" != "true" ]; then
        local ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"
        ccache --set-config cache_dir="$ccache_dir" >> "$compile_log" 2>&1
        ccache --set-config max_size="20G" >> "$compile_log" 2>&1
        ccache --zero-stats >> "$compile_log" 2>&1

        # Setup custom CUDA compiler wrapper (matching local_dev.sh)
        local custom_bin_dir="$SDK_DIR/custom_bin"
        mkdir -p "$custom_bin_dir"
        ln -sf /usr/bin/ccache "$custom_bin_dir/dlcc"
        export PATH="$custom_bin_dir:$PATH"
        export CUDA_NVCC_EXECUTABLE="$custom_bin_dir/dlcc"
    else
        echo "[INFO] ccache is disabled, using direct compiler wrapper" | tee -a "$compile_log"
    fi

    # Build directory
    local build_dir="${REPO_PATH}/build_${arch}"

    # Generate and run CMake configuration
    local cmake_cmd=$(get_android_cmake_flags "$SDK_DIR" "$build_dir")
    echo "[INFO] CMake configuration:" | tee -a "$compile_log"
    echo "$cmake_cmd" | tee -a "$compile_log"

    # Execute CMake
    cd "$REPO_PATH"
    eval "$cmake_cmd" >> "$compile_log" 2>&1

    if [ $? -ne 0 ]; then
        echo "[ERROR] CMake configuration failed" | tee -a "$compile_log"
        # Clear ccache on CMake failure
        if [ "$DISABLE_CCACHE" != "true" ]; then
            echo "[WARNING] Clearing ccache due to CMake configuration failure..." | tee -a "$compile_log"
            local ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"
            if [ -d "$ccache_dir" ]; then
                rm -rf "$ccache_dir" && mkdir -p "$ccache_dir"
                echo "[INFO] ccache cleared successfully" | tee -a "$compile_log"
            fi
        fi
        return 1
    fi

    # Build with ninja
    cd "$build_dir"
    echo "[INFO] Starting ninja build..." | tee -a "$compile_log"
    echo "[INFO] compile log path: $compile_log"
    ninja -v -j 12 >> "$compile_log" 2>&1

    if [ $? -ne 0 ]; then
        echo "[ERROR] Outputting full compile log($compile_log) below:"
        cat "$compile_log"
        # Clear ccache on ninja build failure
        if [ "$DISABLE_CCACHE" != "true" ]; then
            echo "[WARNING] Clearing ccache due to ninja build failure..." | tee -a "$compile_log"
            local ccache_dir="${CCACHE_DIR:-/LocalRun/$(whoami)/cache/llama_cpp_ccache}"
            if [ -d "$ccache_dir" ]; then
                rm -rf "$ccache_dir" && mkdir -p "$ccache_dir"
                echo "[INFO] ccache cleared successfully" | tee -a "$compile_log"
            fi
        fi
        return 1
    fi

    echo "[INFO] Android compilation completed successfully" | tee -a "$compile_log"
    return 0
}

# Function to check if we should use the proven local_dev.sh method
should_use_local_dev_method() {
    local arch="$1"

    # Use local_dev.sh method for Android compilation
    if [ "$arch" = "android" ]; then
        return 0
    fi

    return 1
}
