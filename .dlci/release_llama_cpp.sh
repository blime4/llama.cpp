#!/bin/bash

set -e

# Source configuration utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

echo "[INFO] Starting release_llama_cpp_x86_64 script"
ARCH=${DOCKER_PLATFORM}
build_dir=${REPO_PATH}/build_${ARCH}

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
    cp "${build_dir}/bin"/* release/

    echo "[INFO] Files to be released:"
    ls -la release/

    echo "Release Version: ${CI_COMMIT_TAG}" > release/version.txt
    echo "Build Date: $(date)" >> release/version.txt
    echo "Commit SHA: ${CI_COMMIT_SHA}" >> release/version.txt
    echo "Build Platform: ${ARCH}" >> release/version.txt
    echo "Architecture: $(uname -m)" >> release/version.txt

    # Determine the correct platform suffix based on architecture using unified config
    PLATFORM_SUFFIX=$(get_platform_suffix "${ARCH}")
    if [[ -z "$PLATFORM_SUFFIX" ]]; then
        echo "[WARNING] Unknown architecture: ${ARCH}, using generic naming"
        PLATFORM_SUFFIX="linux-${ARCH}"
    fi

    # Transform SDK_TAG from V2_SOFTWARE_master_202510082141 to sdk202509180241
    SDK_TAG_TRANSFORMED=""
    if [[ "$SDK_TAG" =~ V2_SOFTWARE_master_([0-9]+) ]]; then
        SDK_TAG_TRANSFORMED="sdk${BASH_REMATCH[1]}"
    else
        # Fallback: if pattern doesn't match, use original SDK_TAG
        SDK_TAG_TRANSFORMED="$SDK_TAG"
    fi

    RELEASE_NAME="llama-${CI_COMMIT_TAG}-${SDK_TAG_TRANSFORMED}-bin-${PLATFORM_SUFFIX}.zip"
    echo "[INFO] Creating release package: ${RELEASE_NAME}"
    zip -r "${RELEASE_NAME}" release/*

    echo "[INFO] Release package created: ${RELEASE_NAME}"
    ls -la "${RELEASE_NAME}"

    jf rt u "${RELEASE_NAME}" llama.cpp-release/  -flat=true
    package_name=$(basename $(ls "${RELEASE_NAME}" | head -n1))
    echo "【Artifact Download Link】http://ext-artifactory.denglin.com:8082/artifactory/llama.cpp-release/$package_name"
else
    echo "[ERROR] build/bin directory not found at ${build_dir}/bin"
    exit 1
fi
