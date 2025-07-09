set -e
env
source ${SDK_WORKSPACE}/sdk/env.sh
env

ccache --set-config cache_dir=/LocalRun/$(whoami)/cache/llama_cpp_ccache
ccache --set-config max_size=20G
ccache --zero-stats

# --- test-backend-ops ---
build_dir=$(pwd)/../build

if [ ! -d "${build_dir}" ]; then
    echo "[ERROR] $build_dir is not exist ..."
    exit 1
fi

echo "Pass 1111111111"

case_file=${build_dir}/bin/test-backend-ops

# Ensure case_file is valid
if [ ! -f "$case_file" ]; then
  echo "[ERROR] $case_file is not exist ..."
  exit 1
fi
echo "Pass 222222222"

export PATH=${build_dir}/bin:$PATH
export LD_LIBRARY_PATH=${build_dir}/bin:$LD_LIBRARY_PATH

chmod +x case_file

echo "Pass 333333333"

ldd $case_file | grep -q "not found"

cd ${build_dir}/bin

echo "Pass 444444444444"

./test-backend-ops

cd -
