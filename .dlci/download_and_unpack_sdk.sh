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

# Check if cached SDK has a different TAG
SDK_TAG_FILE=".sdk_tag"
CURRENT_CACHED_TAG=""
if [[ -f "$SDK_TAG_FILE" ]]; then
    CURRENT_CACHED_TAG=$(cat "$SDK_TAG_FILE" 2>/dev/null || echo "")
fi

if [[ -n "$CURRENT_CACHED_TAG" && "$CURRENT_CACHED_TAG" != "$SDK_TAG" ]]; then
    echo "[INFO] SDK TAG has changed from '$CURRENT_CACHED_TAG' to '$SDK_TAG'"
    echo "[INFO] Cleaning up old SDK files..."
    # Remove old SDK archives and extracted directories
    rm -rf *.tar.xz sdk sdk_aarch64 2>/dev/null || true
    echo "[INFO] Old SDK files cleaned up"
elif [[ -z "$CURRENT_CACHED_TAG" ]]; then
    echo "[INFO] No cached SDK TAG found, will use: $SDK_TAG"
else
    echo "[INFO] Using cached SDK with matching TAG: $SDK_TAG"
fi

# Save current SDK TAG
echo "$SDK_TAG" > "$SDK_TAG_FILE"

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

# Initialize variables for dual SDK configuration (Android)
X86_REPOSITORY=""
X86_FILENAME_PATTERN=""
X86_EXTRACT_DIR=""
AARCH64_REPOSITORY=""
AARCH64_FILENAME_PATTERN=""
AARCH64_EXTRACT_DIR=""

# Special handling for Android dual SDK configuration
if [[ "$ARCH" == "android" ]]; then
    echo "[INFO] Android platform detected - using dual SDK configuration"

    # Get x86 SDK configuration for compilation
    echo "[INFO] Configuring x86 SDK for compilation..."
    X86_REPOSITORY=$(get_config_value ".build.sdk.download.${ARCH}.x86_sdk.repository")
    X86_FILENAME_PATTERN=$(get_config_value ".build.sdk.download.${ARCH}.x86_sdk.filename_pattern")
    X86_EXTRACT_DIR=$(get_config_value ".build.sdk.download.${ARCH}.x86_sdk.extract_dir")

    # Get aarch64 SDK configuration for linking
    echo "[INFO] Configuring aarch64 SDK for linking..."
    AARCH64_REPOSITORY=$(get_config_value ".build.sdk.download.${ARCH}.aarch64_sdk.repository")
    AARCH64_FILENAME_PATTERN=$(get_config_value ".build.sdk.download.${ARCH}.aarch64_sdk.filename_pattern")
    AARCH64_EXTRACT_DIR=$(get_config_value ".build.sdk.download.${ARCH}.aarch64_sdk.extract_dir")

    # Set primary SDK (x86) for backward compatibility
    REPOSITORY="$X86_REPOSITORY"
    FILENAME=""
    FILENAME_PATTERN="$X86_FILENAME_PATTERN"
    EXTRACT_DIR="$X86_EXTRACT_DIR"

    echo "[INFO] x86 SDK: $X86_REPOSITORY -> $X86_EXTRACT_DIR"
    echo "[INFO] aarch64 SDK: $AARCH64_REPOSITORY -> $AARCH64_EXTRACT_DIR"

    # Validate Android dual SDK configuration
    if [[ -z "$X86_REPOSITORY" || -z "$X86_FILENAME_PATTERN" || -z "$X86_EXTRACT_DIR" ]]; then
        echo "[ERROR] Incomplete x86 SDK configuration for Android" >&2
        echo "[ERROR] x86_repository: '$X86_REPOSITORY', x86_pattern: '$X86_FILENAME_PATTERN', x86_extract: '$X86_EXTRACT_DIR'" >&2
        exit 1
    fi

    if [[ -z "$AARCH64_REPOSITORY" || -z "$AARCH64_FILENAME_PATTERN" || -z "$AARCH64_EXTRACT_DIR" ]]; then
        echo "[ERROR] Incomplete aarch64 SDK configuration for Android" >&2
        echo "[ERROR] aarch64_repository: '$AARCH64_REPOSITORY', aarch64_pattern: '$AARCH64_FILENAME_PATTERN', aarch64_extract: '$AARCH64_EXTRACT_DIR'" >&2
        exit 1
    fi
else
    # Standard single SDK configuration for other platforms
    REPOSITORY=$(get_sdk_download_config "$ARCH" "repository")
    FILENAME=$(get_sdk_download_config "$ARCH" "filename" 2>/dev/null || echo "")
    FILENAME_PATTERN=$(get_sdk_download_config "$ARCH" "filename_pattern" 2>/dev/null || echo "")
    EXTRACT_DIR=$(get_sdk_download_config "$ARCH" "extract_dir")
fi

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

    # Verify that files were actually downloaded
    local downloaded_files=""
    if [[ "$download_pattern" == *"*"* ]]; then
        # Pattern-based filename - check if any files match
        downloaded_files=$(ls $download_pattern 2>/dev/null)
        if [[ -z "$downloaded_files" ]]; then
            echo "[ERROR] Download command succeeded but no files were downloaded" >&2
            echo "[ERROR] No file found matching pattern: $download_pattern" >&2
            echo "[INFO] This may indicate:" >&2
            echo "[INFO]   1. The SDK archive does not exist for tag: $sdk_tag" >&2
            echo "[INFO]   2. The filename pattern does not match any files in the repository" >&2
            echo "[INFO]   3. Check if the SDK tag exists: jf rt search $repository/$sdk_tag/" >&2
            return 1
        fi
    else
        # Exact filename
        if [[ ! -f "$download_pattern" ]]; then
            echo "[ERROR] Download command succeeded but file was not found: $download_pattern" >&2
            echo "[INFO] This may indicate the SDK archive does not exist for tag: $sdk_tag" >&2
            return 1
        fi
        downloaded_files="$download_pattern"
    fi

    echo "[INFO] SDK archive downloaded successfully: $downloaded_files"
    return 0
}

# Download the SDK archive(s)
if ! download_sdk_archive "$ARCH" "$REPOSITORY" "$DOWNLOAD_PATTERN" "$SDK_TAG"; then
    echo "[ERROR] Failed to download SDK archive for platform: $ARCH"
    exit 1
fi

# For Android, also download aarch64 SDK
if [[ "$ARCH" == "android" ]]; then
    echo "[INFO] Downloading aarch64 SDK for Android linking..."

    # Determine aarch64 download pattern
    if [[ -n "$AARCH64_FILENAME_PATTERN" ]]; then
        AARCH64_DOWNLOAD_PATTERN="$AARCH64_FILENAME_PATTERN"
    else
        echo "[ERROR] aarch64 filename_pattern not configured for Android" >&2
        exit 1
    fi

    if ! download_sdk_archive "android-aarch64" "$AARCH64_REPOSITORY" "$AARCH64_DOWNLOAD_PATTERN" "$SDK_TAG"; then
        echo "[ERROR] Failed to download aarch64 SDK archive"
        exit 1
    fi
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

    # Handle special extraction for Android aarch64 SDK
    if [[ "$platform" == "android-aarch64" ]]; then
        echo "[INFO] Extracting aarch64 SDK with special handling..."
        mkdir -p "$extract_dir"
        if ! tar xf "$file_to_extract" -C "$extract_dir" --strip-components=1; then
            echo "[ERROR] Failed to extract aarch64 SDK archive: $file_to_extract" >&2
            return 1
        fi
    else
        # Standard extraction
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
    fi

    echo "[INFO] SDK extracted successfully to: $extract_dir"
    return 0
}

# Extract the SDK archive(s)
if ! extract_sdk_archive "$ARCH" "$EXTRACT_DIR" "$DOWNLOAD_PATTERN"; then
    echo "[ERROR] Failed to extract SDK archive for platform: $ARCH"
    exit 1
fi

# For Android, also extract aarch64 SDK
if [[ "$ARCH" == "android" ]]; then
    echo "[INFO] Extracting aarch64 SDK for Android linking..."

    if ! extract_sdk_archive "android-aarch64" "$AARCH64_EXTRACT_DIR" "$AARCH64_DOWNLOAD_PATTERN"; then
        echo "[ERROR] Failed to extract aarch64 SDK archive"
        exit 1
    fi
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

if [[ "$ARCH" == "android" ]]; then
    echo "[INFO] Android dual SDK configuration:"
    echo "[INFO]   x86 SDK (compile): $EXTRACT_DIR"
    echo "[INFO]   aarch64 SDK (link): $AARCH64_EXTRACT_DIR"
else
    echo "[INFO] Extracted directory: $EXTRACT_DIR"
fi
