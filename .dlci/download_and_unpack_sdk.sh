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

# Detect architecture
ARCH=${DOCKER_PLATFORM}

# Validate platform is supported
if ! is_platform_supported "$ARCH"; then
    supported_platforms=$(get_supported_platforms)
    echo "[ERROR] Unsupported DOCKER_PLATFORM: $ARCH" >&2
    echo "[INFO] Supported platforms: $supported_platforms" >&2
    exit 1
fi

echo "[INFO] Creating SDK workspace: $SDK_WORKSPACE"
mkdir -p "$SDK_WORKSPACE"
cd "$SDK_WORKSPACE"

# Get SDK download configuration from config.yml
get_sdk_download_config() {
    local platform="$1"
    local config_key="$2"

    local value=$(get_config_value ".build.sdk.download.${platform}.${config_key}")
    if [[ -z "$value" ]]; then
        echo "[ERROR] Missing SDK download configuration for platform $platform, key: $config_key" >&2
        return 1
    fi
    echo "$value"
}

# Get SDK download configuration for current platform
REPOSITORY=$(get_sdk_download_config "$ARCH" "repository")
FILENAME=$(get_sdk_download_config "$ARCH" "filename" 2>/dev/null || echo "")
FILENAME_PATTERN=$(get_sdk_download_config "$ARCH" "filename_pattern" 2>/dev/null || echo "")
EXTRACT_DIR=$(get_sdk_download_config "$ARCH" "extract_dir")

# Determine the actual filename to use
if [[ -n "$FILENAME" ]]; then
    # Use exact filename
    ACTUAL_FILENAME="$FILENAME"
    DOWNLOAD_PATTERN="$FILENAME"
elif [[ -n "$FILENAME_PATTERN" ]]; then
    # Use filename pattern
    ACTUAL_FILENAME="$FILENAME_PATTERN"
    DOWNLOAD_PATTERN="$FILENAME_PATTERN"
else
    echo "[ERROR] Neither filename nor filename_pattern is configured for platform: $ARCH" >&2
    exit 1
fi

echo "[INFO] Platform: $ARCH"
echo "[INFO] Repository: $REPOSITORY"
echo "[INFO] Download pattern: $DOWNLOAD_PATTERN"
echo "[INFO] Extract directory: $EXTRACT_DIR"

# Step 1: Download SDK archive
download_sdk_archive() {
    local platform="$1"
    local repository="$2"
    local download_pattern="$3"
    local sdk_tag="$4"

    echo "[INFO] Checking if SDK archive already exists..."

    # Check if file already exists
    if [[ "$download_pattern" == *"*"* ]]; then
        # Pattern-based filename
        if ls $download_pattern 1> /dev/null 2>&1; then
            echo "[INFO] SDK archive already exists, skipping download"
            return 0
        fi
    else
        # Exact filename
        if [[ -f "$download_pattern" ]]; then
            echo "[INFO] SDK archive already exists, skipping download"
            return 0
        fi
    fi

    echo "[INFO] Downloading SDK archive for $platform: $sdk_tag"
    echo "[INFO] Download path: $repository/$sdk_tag/$download_pattern"

    if ! jf rt dl "$repository/$sdk_tag/$download_pattern" -flat=true; then
        echo "[ERROR] Failed to download SDK archive" >&2
        return 1
    fi

    echo "[INFO] SDK archive downloaded successfully"
    return 0
}

# Download the SDK archive
if ! download_sdk_archive "$ARCH" "$REPOSITORY" "$DOWNLOAD_PATTERN" "$SDK_TAG"; then
    echo "[ERROR] Failed to download SDK archive"
    exit 1
fi

# Step 2: Extract SDK archive
extract_sdk_archive() {
    local platform="$1"
    local extract_dir="$2"
    local download_pattern="$3"

    # Check if already extracted
    if [[ -d "$extract_dir" ]]; then
        echo "[INFO] SDK already extracted to $extract_dir, skipping"
        return 0
    fi

    # Find the actual file to extract
    local file_to_extract=""
    if [[ "$download_pattern" == *"*"* ]]; then
        # Pattern-based filename - find the actual file
        file_to_extract=$(ls $download_pattern 2>/dev/null | head -1)
        if [[ -z "$file_to_extract" ]]; then
            echo "[ERROR] No file found matching pattern: $download_pattern" >&2
            return 1
        fi
    else
        # Exact filename
        file_to_extract="$download_pattern"
        if [[ ! -f "$file_to_extract" ]]; then
            echo "[ERROR] File not found: $file_to_extract" >&2
            return 1
        fi
    fi

    echo "[INFO] Extracting SDK archive for $platform: $file_to_extract"

    if ! tar xf "$file_to_extract"; then
        echo "[ERROR] Failed to extract SDK archive: $file_to_extract" >&2
        return 1
    fi

    # Handle special cases for different platforms
    case "$platform" in
        "riscv64")
            # RISC-V needs to rename the extracted directory
            if [[ -d "sdk" && "$extract_dir" != "sdk" ]]; then
                echo "[INFO] Renaming sdk directory to $extract_dir for $platform"
                mv sdk "$extract_dir"
            fi
            ;;
    esac

    # Fix ownership for non-x86_64 platforms
    if [[ "$platform" != "x86_64" ]]; then
        echo "[INFO] Fixing ownership for $platform"
        sudo chown -R $(id -u):$(id -g) "${SDK_WORKSPACE}"
    fi

    echo "[INFO] SDK extracted successfully to: $extract_dir"
    return 0
}

# Extract the SDK archive
if ! extract_sdk_archive "$ARCH" "$EXTRACT_DIR" "$DOWNLOAD_PATTERN"; then
    echo "[ERROR] Failed to extract SDK archive"
    exit 1
fi

# Step 3: Setup Docker repository
if [ -n "$DOCKER_REPO_PATH" ]; then
  echo "[INFO] Setting up Docker repository..."
  if ! setup_docker_repo "$DOCKER_REPO_PATH"; then
    echo "[ERROR] Failed to setup Docker repository: $DOCKER_REPO_PATH"
    exit 1
  fi
else
  echo "[INFO] DOCKER_REPO_PATH not set, skipping Docker repository setup"
fi

echo "[INFO] SDK download and extraction completed successfully"
echo "[INFO] SDK workspace: $SDK_WORKSPACE"
echo "[INFO] Extracted directory: $EXTRACT_DIR"