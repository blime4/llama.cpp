#!/bin/bash
set -e
if [ -z "$SDK_DIR" ]; then
    echo "Error: SDK_DIR environment variable is not set"
    echo "Please run: source scripts/denglin/docker_env.sh"
    exit 1
fi

echo "SDK_DIR: $SDK_DIR"

# Build
echo "Building..."
dlcc -o test-cuda-gptq-dlblas \
    test-cuda-gptq-dlblas.cu \
    -I${SDK_DIR}/include \
    -use_fast_math \
    -mllvm -dlgpu-lower-xtpvn=true \
    --save-temps=. \
    -fPIC \
    --offload-arch=dlgput64 \
    -x cuda \
    -lcurt \
    -lcublas \
    -ldlblas \
    -std=c++17

if [ $? -eq 0 ]; then
    echo "Build succeeded"
    echo ""
    echo "=== Running test ==="
    ./test-cuda-gptq-dlblas
else
    echo "Build failed"
    exit 1
fi