set -e
env
source ${SDK_WORKSPACE}/sdk/env.sh
env

ccache --set-config cache_dir=/LocalRun/$(whoami)/cache/llama_cpp_ccache
ccache --set-config max_size=20G
ccache --zero-stats

# --- test-backend-ops ---
echo "[INFO] REPO_PATH: ${REPO_PATH}"
cd ${REPO_PATH}
build_dir=$(pwd)/build

if [ ! -d "${build_dir}" ]; then
    echo "[ERROR] $build_dir is not exist ..."
    exit 1
fi

case_file=${build_dir}/bin/test-backend-ops

# Ensure case_file is valid
if [ ! -f "$case_file" ]; then
  echo "[ERROR] $case_file is not exist ..."
  exit 1
fi

export GGML_DEBUG=1
export PATH=${build_dir}/bin:$PATH
export LD_LIBRARY_PATH=${build_dir}/bin:$LD_LIBRARY_PATH

chmod +x $case_file

#ldd $case_file | grep -q "not found"
#
ldd $case_file

cd ${build_dir}

./bin/test-arg-parser
./bin/test-c
./bin/test-gguf
./bin/test-grammar-parser
./bin/test-sampling
./bin/test-llama-grammar
./bin/test-log
./bin/test-chat-parser
./bin/test-chat-template
./bin/test-grammar-integration
./bin/test-json-partial
./bin/test-mtmd-c-api
./bin/test-regex-partial
./bin/test-backend-ops

cd -

echo "<>test-backend-ops end ...<>"
