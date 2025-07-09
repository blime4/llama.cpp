set -e
env
source ${SDK_WORKSPACE}/sdk/env.sh
env

ccache --set-config cache_dir=/LocalRun/$(whoami)/cache/llama_cpp_ccache
ccache --set-config max_size=20G
ccache --zero-stats

export CCACHE_LOGFILE=/LocalRun/$(whoami)/cache/ccache.log

# Get the latest tag (if any), otherwise empty
# --exact returns tag only if HEAD is exactly at a tag, otherwise non-zero exit
# 2>/dev/null suppresses error output, || true ensures tag is empty if not found

tag=$(git describe --tags --exact 2>/dev/null || true)
commit=$(git rev-parse --short=8 HEAD)
sdk_num=$(echo $SDK_TAG | grep -oE '[0-9]{12}')
echo "[INFO] commit: $commit"

if [[ -n "$tag" ]]; then
    # Example: v2.7.0.dev20250312+dl-main-1
    # Extract base version (remove 'v' and everything after '+')
    base_version=$(echo "$tag" | sed -E 's/^v([0-9.]+\.dev[0-9]+).*/\1/')
    # Extract denglin_version (the part after '+', remove '-' and '_')
    denglin_version=$(echo "$tag" | grep -oE '\+.*' | sed 's/^+//' | sed 's/[-_]//g')
    echo "[INFO] base_version: $base_version"
    echo "[INFO] denglin_version: $denglin_version"
    export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
else
    # No tag, use main version from version.txt
    base_version=$(cat version.txt | sed 's/[^0-9.].*$//')
    echo "[INFO] base_version: $base_version"
    denglin_version="git${commit}"
    echo "[INFO] denglin_version: $denglin_version"
    export LLAMA_CPP_BUILD_VERSION="${base_version}+${denglin_version}.sdk${sdk_num}"
fi

# --- Improved ccache setup for CUDA compilation ---

# Ensure sdk_path is set and valid
if [ -z "$sdk_path" ] || [ ! -d "$sdk_path" ]; then
  echo "[ERROR] sdk_path is not set or is not a valid directory. Current value: '$sdk_path'"
  exit 1
fi

# Define a custom bin directory within sdk_path for our tools like the ccache wrapper.
# This avoids polluting the main PATH with the entire sdk_path.
CUSTOM_BIN_DIR="$sdk_path/custom_bin"

# Create the custom bin directory if it doesn't exist.
mkdir -p "$CUSTOM_BIN_DIR" || {
    echo "[ERROR] Failed to create custom bin directory at $CUSTOM_BIN_DIR."
    exit 1
}

# Create a symlink named 'dlcc' in the custom bin directory, pointing to ccache.
# This allows us to use ccache for CUDA compilation by setting CUDA_NVCC_EXECUTABLE to this symlink.
ln -sf /usr/bin/ccache "$CUSTOM_BIN_DIR/dlcc" || {
  echo "[ERROR] Failed to create symlink for ccache at $CUSTOM_BIN_DIR/dlcc."
  exit 1
}

# Add the custom bin directory to the PATH.
# This makes 'dlcc' (and any other tools placed here) accessible if needed directly from the shell,
# though CUDA_NVCC_EXECUTABLE uses an absolute path.
export PATH="$CUSTOM_BIN_DIR:$PATH"

# Set CUDA_NVCC_EXECUTABLE to use the ccache symlink (absolute path).
# This tells nvcc (NVIDIA CUDA Compiler) to use our ccache-enabled wrapper.
export CUDA_NVCC_EXECUTABLE="$CUSTOM_BIN_DIR/dlcc"

echo "[INFO] CUDA_NVCC_EXECUTABLE is set to: $CUDA_NVCC_EXECUTABLE"
echo "[INFO] Custom bin directory '$CUSTOM_BIN_DIR' added to PATH."
# --- End of improved ccache setup ---

# --- Compile llama.cpp ---
sdk=${SDK_WORKSPACE}/sdk

cmake -G Ninja -B ../build \
    -DGGML_DLCU=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CUDA_GRAPHS=OFF \
    -DLLAMA_CURL=OFF \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=OFF \
    -DSDK_DIR=${sdk}

ccache --show-stats

#cmake --build ../build --config Release -j 12
cd ../build

ninja -j 12

cd -

echo "<> compile succed ... <>"
