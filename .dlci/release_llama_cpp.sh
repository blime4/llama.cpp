#!/bin/bash

set -e

# Source configuration utilities
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/utils.sh"

echo "[INFO] Starting release_llama_cpp script"
echo "[INFO] Platform: ${DOCKER_PLATFORM}"

# Validate required environment variables
if [ -z "$REPO_PATH" ]; then
    echo "[ERROR] REPO_PATH environment variable is not set"
    exit 1
fi
if [ -z "$CI_COMMIT_TAG" ]; then
    echo "[ERROR] CI_COMMIT_TAG environment variable is not set"
    exit 1
fi
if [ -z "$DOCKER_PLATFORM" ]; then
    echo "[ERROR] DOCKER_PLATFORM environment variable is not set"
    exit 1
fi

# Create release package using unified function
if ! create_release_package; then
    echo "[ERROR] Failed to create release package"
    exit 1
fi

# Upload to artifactory
echo ""
echo "======================================"
echo "Uploading Release Package"
echo "======================================"

# Determine platform suffix and SDK tag for finding the package
ARCH="${DOCKER_PLATFORM}"
PLATFORM_SUFFIX=$(get_platform_suffix "${ARCH}")
if [[ -z "$PLATFORM_SUFFIX" ]]; then
    PLATFORM_SUFFIX="linux-${ARCH}"
fi

SDK_TAG_TRANSFORMED=""
if [[ "$SDK_TAG" =~ V2_SOFTWARE_master_([0-9]+) ]]; then
    SDK_TAG_TRANSFORMED="sdk${BASH_REMATCH[1]}"
else
    SDK_TAG_TRANSFORMED="${SDK_TAG:-unknown}"
fi

RELEASE_NAME="llama-${CI_COMMIT_TAG:-dev}-${SDK_TAG_TRANSFORMED}-bin-${PLATFORM_SUFFIX}.zip"

# Upload the package
if [ -f "${REPO_PATH}/${RELEASE_NAME}" ]; then
    echo "[INFO] Uploading ${RELEASE_NAME} to artifactory..."
    cd "${REPO_PATH}"
    jf rt u "${RELEASE_NAME}" llama.cpp-release/ --flat=true

    package_name=$(basename "${RELEASE_NAME}")
    echo ""
    echo "[SUCCESS] Package uploaded successfully!"
    echo "【Artifact Download Link】http://ext-artifactory.denglin.com:8082/artifactory/llama.cpp-release/$package_name"
else
    echo "[ERROR] Release package not found: ${REPO_PATH}/${RELEASE_NAME}"
    exit 1
fi
