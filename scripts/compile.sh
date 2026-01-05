#!/usr/bin/env bash
sdk=/LocalRun/wenjian.ma/llama.cpp/sdk/sdk

cd /LocalRun/wenjian.ma/llama.cpp.wsc/llama.cpp.github

cmake -B build -DGGML_DLCU=ON \
    -DCMAKE_VERBOSE_MAKEFILE=ON \
    -DCMAKE_BUILD_TYPE=Debug -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
    -DGGML_CUDA_GRAPHS=OFF -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=ON \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DGGML_CUDA_FA=ON \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
    -DSDK_DIR=${sdk}

#cmake --build build --config Release -j 12 --verbose
cmake --build build --config Release -j 12

#ninja -j 8

cd build/bin

./test-backend-ops
