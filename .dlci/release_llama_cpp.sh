#!/bin/bash

set -e

echo "[INFO] Starting release_llama_cpp_x86_64 script"
build_dir="${REPO_PATH}/../build"

if [ -z "$REPO_PATH" ]; then
    echo "[ERROR] REPO_PATH environment variable is not set"
    exit 1
fi
if [ -z "$CI_COMMIT_TAG" ]; then
    echo "[ERROR] CI_COMMIT_TAG environment variable is not set"
    exit 1
fi

mkdir -p release

if [ -d "${build_dir}/bin" ]; then
    echo "[INFO] Copying release files from build/bin"
    find "${build_dir}/bin" -type f ! -name "test*" -exec cp {} release/ \;

    echo "[INFO] Files to be released:"
    ls -la release/

    ARCH=${DOCKER_PLATFORM}

    echo "Release Version: ${CI_COMMIT_TAG}" > release/version.txt
    echo "Build Date: $(date)" >> release/version.txt
    echo "Commit SHA: ${CI_COMMIT_SHA}" >> release/version.txt
    echo "Build Platform: ${ARCH}" >> release/version.txt
    echo "Architecture: $(uname -m)" >> release/version.txt

    RELEASE_NAME="llama-${CI_COMMIT_TAG}-bin-manylinux_2_28-${ARCH}.zip"
    zip -r "${RELEASE_NAME}" release/*

    echo "[INFO] Release package created: ${RELEASE_NAME}"
    ls -la "${RELEASE_NAME}"

    jf rt u "${RELEASE_NAME}" dl-pypi/llamacpp/  -flat=true
    package_name=$(basename $(ls "${RELEASE_NAME}" | head -n1))
    echo "【Artifact Download Link】http://ext-artifactory.denglin.com:8082/artifactory/dl-pypi/llamacpp/$package_name"
else
    echo "[ERROR] build/bin directory not found at ${build_dir}/bin"
    exit 1
fi
