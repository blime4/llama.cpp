#!/bin/bash
set -e

# Load utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

SDK_TAG="$1"
SDK_WORKSPACE="$2"

if [[ -z "$SDK_TAG" || -z "$SDK_WORKSPACE" ]]; then
  echo "[ERROR] Usage: $0 <SDK_TAG> <SDK_WORKSPACE>"
  exit 1
fi

echo "[INFO] Creating SDK workspace: $SDK_WORKSPACE"
mkdir -p "$SDK_WORKSPACE"
cd "$SDK_WORKSPACE"

# Detect architecture
ARCH=${DOCKER_PLATFORM}

# Step 1: Download SDK archive
if [[ "$ARCH" == "x86_64" ]]; then
  if ! ls sdk.tar.bz2 1> /dev/null 2>&1; then
    echo "[INFO] Downloading SDK archive for x86_64: $SDK_TAG"
    jf rt dl ai-sw-dailybuild-v2/${SDK_TAG}/sdk.tar.bz2 -flat=true
  else
    echo "[INFO] SDK archive already exists, skipping download"
  fi
elif [[ "$ARCH" == "aarch64" ]]; then
  if ! ls *-aarch64.tar.bz2 1> /dev/null 2>&1; then
    echo "[INFO] Downloading SDK archive for aarch64: $SDK_TAG"
    jf rt dl ai-sw-sdk-v2-test/${SDK_TAG}/*-aarch64.tar.bz2 -flat=true
  else
    echo "[INFO] SDK archive already exists, skipping download"
  fi
elif [[ "$ARCH" == "riscv64" ]]; then
  if ! ls *riscv64.tar.bz2 1> /dev/null 2>&1; then
    echo "[INFO] Downloading SDK archive for risv64: $SDK_TAG"
    jf rt dl ai-sw-sdk-v2-test/${SDK_TAG}/*-riscv64.tar.bz2 -flat=true
  else
    echo "[INFO] SDK archive already exists, skipping download"
  fi
elif [[ "$ARCH" == "loongarch64" ]]; then
  if ! ls *loongarch.tar.bz2 1> /dev/null 2>&1; then
    echo "[INFO] Downloading SDK archive for loongarch64: $SDK_TAG"
    jf rt dl ai-sw-sdk-v2-test/${SDK_TAG}/*loongarch.tar.bz2 -flat=true
  else
    echo "[INFO] SDK archive already exists, skipping download"
  fi
else
  echo "[ERROR] Unsupported architecture: $ARCH"
  exit 1
fi

# Step 2: Extract SDK
if [[ "$ARCH" == "x86_64" && ! -d sdk ]]; then
  echo "[INFO] Extracting SDK archive for x86_64..."
  tar xf sdk.tar.bz2
elif [[ "$ARCH" == "aarch64" && ! -d sdk ]]; then
  echo "[INFO] Extracting SDK archive for aarch64..."
  tar xf *-aarch64.tar.bz2
  sudo chown -R $(id -u):$(id -g) "${SDK_WORKSPACE}"
elif [[ "$ARCH" == "loongarch64" && ! -d sdk ]]; then
  echo "[INFO] Extracting SDK archive for loongarch64..."
  tar xf *-loongarch.tar.bz2
  sudo chown -R $(id -u):$(id -g) "${SDK_WORKSPACE}"
elif [[ "$ARCH" == "riscv64" && ! -d sdk_${ARCH} ]]; then
  echo "[INFO] Extracting SDK archive for riscv64..."
  tar xf *-riscv64.tar.bz2
  sudo chown -R $(id -u):$(id -g) "${SDK_WORKSPACE}"
  mv sdk sdk_${ARCH}
else
  echo "[INFO] SDK already extracted, skipping"
fi

# Step 3: Download PyTorch .whl packages
# cd sdk
# echo "[INFO] Downloading PyTorch .whl packages to $(pwd)"
# jf rt dl daily-pytorch-v2-pt2.5/${SDK_TAG}/cp312-cp312_manylinux/ -flat=true

# Step 4: Setup Docker repository
if [ -n "$DOCKER_REPO_PATH" ]; then
  echo "[INFO] Setting up Docker repository..."
  if ! setup_docker_repo "$DOCKER_REPO_PATH"; then
    echo "[ERROR] Failed to setup Docker repository: $DOCKER_REPO_PATH"
    exit 1
  fi
else
  echo "[INFO] DOCKER_REPO_PATH not set, skipping Docker repository setup"
fi
