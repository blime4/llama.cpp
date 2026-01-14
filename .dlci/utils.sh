#!/bin/bash
#
# Unified Utility functions for CI scripts
#
# This file contains all utility functions including configuration management,
# network detection, Docker operations, and CI build functionality.
# Previously split across multiple files, now unified for better maintainability.
#

# Safe command execution without LD_PRELOAD injection
# This prevents issues when LD_PRELOAD contains dlpti_injection.so
safe_rm() {
    # Execute rm without LD_PRELOAD to avoid dlpti injection issues
    # Use bash builtin to avoid LD_PRELOAD affecting env command itself
    local old_preload="$LD_PRELOAD"
    unset LD_PRELOAD
    rm "$@"
    local ret=$?
    if [ -n "$old_preload" ]; then
        export LD_PRELOAD="$old_preload"
    fi
    return $ret
}

get_config_file() {
    local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    echo "${script_dir}/config.yml"
}

# No need to check for yq or python3, we use bash-yaml parser directly
# Global flag to track if yaml parser is loaded
_YAML_PARSER_LOADED=0

# Load yaml parser function once
load_yaml_parser() {
    if [[ $_YAML_PARSER_LOADED -eq 0 ]]; then
        local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        if [[ ! -f "$script_dir/yaml_parser.sh" ]]; then
            echo "[ERROR] yaml_parser.sh not found in $script_dir" >&2
            return 1
        fi
        source "$script_dir/yaml_parser.sh"
        _YAML_PARSER_LOADED=1
    fi
    return 0
}

# Bash-based YAML parser using open-source bash-yaml
# Source: https://github.com/jasperes/bash-yaml
parse_yaml_bash() {
    local key="$1"
    local file="$2"

    # Load the yaml parser (only once)
    if ! load_yaml_parser; then
        return 1
    fi

    # Parse YAML file and create variables with prefix "cfg_"
    local yaml_vars=$(parse_yaml "$file" "cfg_" 2>/dev/null)

    # Evaluate the variables in a subshell to avoid polluting current environment
    (
        eval "$yaml_vars" 2>/dev/null

        # Convert dot notation key to underscore notation variable name
        # e.g., ".build.sdk.download.aarch64.repository" -> "cfg_build_sdk_download_aarch64_repository"
        local var_name=$(echo "$key" | sed 's/^\.//; s/\./_/g; s/-/_/g')
        var_name="cfg_${var_name}"

        # Check if it's an array variable
        if declare -p "$var_name" 2>/dev/null | grep -q '^declare -a'; then
            # It's an array, print each element without quotes
            eval "for item in \"\${${var_name}[@]}\"; do echo \"\$item\" | sed 's/^\"//' | sed 's/\"$//'; done"
        else
            # It's a scalar value
            local value="${!var_name}"
            if [[ -n "$value" ]]; then
                # Remove quotes from value
                echo "$value" | sed 's/^"//; s/"$//'
            else
                return 1
            fi
        fi
    )
}

get_config_value() {
    local key="$1"
    local config_file="$(get_config_file)"

    if [[ ! -f "$config_file" ]]; then
        echo "[ERROR] Configuration file not found: $config_file" >&2
        return 1
    fi

    # Use bash-yaml parser directly
    parse_yaml_bash "$key" "$config_file"
    return $?
}

get_docker_image() {
    local platform="$1"
    local image_type="${2:-image}"

    local base_url=$(get_config_value ".docker_registry.base_url")
    local namespace=$(get_config_value ".docker_registry.namespace")
    local image=$(get_config_value ".docker_images.${platform}.${image_type}")

    if [[ -z "$base_url" || -z "$namespace" || -z "$image" ]]; then
        echo "[ERROR] Failed to get docker image configuration for platform: $platform" >&2
        return 1
    fi

    echo "${base_url}/${namespace}/${image}"
}

# Get Docker image with network environment detection
# This function handles both internal and external network environments
get_docker_image_with_network() {
    local platform="$1"
    local image_type="${2:-image}"

    # Get base image from config
    local base_image=$(get_docker_image "$platform" "$image_type")
    if [[ $? -ne 0 ]]; then
        return 1
    fi

    # Detect network environment
    local network_env="internal"
    if curl --max-time 3 http://ext-gitlab.denglin.com > /dev/null 2>&1; then
        network_env="external"
    fi

    # Set appropriate registry prefix based on network environment
    local registry_prefix=""
    if [ "$network_env" = "external" ]; then
        registry_prefix="ext-artifactory.denglin.com:8082"
    else
        registry_prefix="artifactory.denglin.com:8082"
    fi

    # Replace the registry prefix with the appropriate one for current network
    local config_base_url=$(get_config_value ".docker_registry.base_url")
    echo "${base_image/$config_base_url/$registry_prefix}"
}

get_platform_suffix() {
    local platform="$1"

    local suffix=$(get_config_value ".platform_suffixes.${platform}")

    if [[ -n "$suffix" ]]; then
        echo "$suffix"
    else
        local default_template=$(get_config_value ".utils.default_platform_suffix_template")
        if [[ -n "$default_template" ]]; then
            echo "${default_template//\{platform\}/$platform}"
        else
            echo "linux-${platform}"
        fi
    fi
}

get_supported_platforms() {
    get_config_value ".ci.supported_platforms" | tr '\n' ' '
}

get_build_tag() {
    local platform="$1"
    get_config_value ".ci.build_tags.${platform}"
}

get_sdk_tag() {
    get_config_value ".build.sdk.tag"
}

get_model_path() {
    get_config_value ".build.model_path"
}

get_build_dir() {
    local platform="$1"
    get_config_value ".build.build_dirs.${platform}"
}

get_release_branch_pattern() {
    get_config_value ".ci.release_branch_pattern"
}

is_platform_supported() {
    local platform="$1"
    local supported_platforms=($(get_supported_platforms))

    for supported in "${supported_platforms[@]}"; do
        if [[ "$supported" == "$platform" ]]; then
            return 0
        fi
    done
    return 1
}

show_config_info() {
    local platform="$1"

    if [[ -n "$platform" ]]; then
        echo "=== Configuration for platform: $platform ==="
        echo "Docker Image: $(get_docker_image "$platform")"
        echo "Dev Docker Image: $(get_docker_image "$platform" "dev_image")"
        echo "Platform Suffix: $(get_platform_suffix "$platform")"
        echo "Build Tag: $(get_build_tag "$platform")"
        echo "Build Directory: $(get_build_dir "$platform")"
    else
        echo "=== General Configuration ==="
        echo "SDK Tag: $(get_sdk_tag)"
        echo "Model Path: $(get_model_path)"
        echo "Supported Platforms: $(get_supported_platforms)"
        echo "Release Branch Pattern: $(get_release_branch_pattern)"
    fi
}

# =============================================================================
# Network and Docker Utilities
# =============================================================================

# Detect network environment (internal vs external)
# Returns: "external" if external network is accessible, "internal" otherwise
detect_network_env() {
    # Increase timeout to 3 seconds for more reliable detection
    curl --max-time 3 http://ext-gitlab.denglin.com > /dev/null 2>&1
    curl_result=$?
    declare -g NETWORK_ENV=''
    if [ $curl_result -eq 0 ]; then
        NETWORK_ENV="external"
        echo "external"  # Output to stdout for command substitution
    else
        NETWORK_ENV="internal"
        echo "internal"  # Output to stdout for command substitution
    fi
}

# Get git repository URL based on network environment
# Returns: Appropriate git repository URL for current network environment
get_git_repo_url() {
    local network_env=$(detect_network_env)
    if [ "$network_env" = "external" ]; then
        echo "ssh://git@ext-gitlab.denglin.com:23/software/ci/docker.git"
    else
        echo "ssh://git@gitlab.denglin.com:23/ai/docker.git"
    fi
}

# Get Docker image based on platform and network environment (for utils.sh compatibility)
# Usage: get_docker_image_for_platform
# Requires: DOCKER_PLATFORM environment variable
get_docker_image_for_platform() {
    # Validate platform is supported
    if ! is_platform_supported "$DOCKER_PLATFORM"; then
        local supported_platforms=$(get_supported_platforms)
        echo "[ERROR] Unsupported DOCKER_PLATFORM: $DOCKER_PLATFORM" >&2
        echo "[INFO] Supported platforms: $supported_platforms" >&2
        return 1
    fi

    # Use the network-aware function
    get_docker_image_with_network "$DOCKER_PLATFORM"
}

# Setup Docker repository (clone if needed and ensure main branch)
# Usage: setup_docker_repo <docker_REPO_PATH>
# Returns: 0 on success, 1 on failure
setup_docker_repo() {
    local docker_REPO_PATH="$1"

    if [ -z "$docker_REPO_PATH" ]; then
        echo "[ERROR] Docker repository path is required" >&2
        return 1
    fi

    # Clone docker repo and check branch if it doesn't exist
    if [ ! -d "$docker_REPO_PATH" ]; then
        local git_repo_url=$(get_git_repo_url)
        local network_env=$(detect_network_env)
        echo "[INFO] Docker directory does not exist, cloning from $network_env network..."
        echo "[INFO] Using repository: $git_repo_url"
        git clone "$git_repo_url" --depth 1 "$docker_REPO_PATH"
    else
        echo "[INFO] Docker directory already exists, checking branch..."
        cd "$docker_REPO_PATH"
        local current_branch=$(git rev-parse --abbrev-ref HEAD)
        if [ "$current_branch" != "main" ]; then
            echo "[INFO] Current branch is $current_branch, switching to main..."
            git fetch origin main
            git checkout main
        else
            echo "[INFO] Already on main branch."
        fi
        cd - > /dev/null
    fi

    # Verify bash.sh exists after cloning/updating
    if [ ! -f "$docker_REPO_PATH/bash.sh" ]; then
        echo "[ERROR] bash.sh not found after cloning/updating: $docker_REPO_PATH/bash.sh" >&2
        return 1
    fi

    echo "[INFO] Docker repository setup completed: $docker_REPO_PATH"
    return 0
}

# Auto-detect and setup Docker repository path
# Usage: auto_setup_docker_repo <SDK_DIR>
# Sets: DOCKER_REPO_PATH global variable
# Returns: 0 on success, 1 on failure
auto_setup_docker_repo() {
    local SDK_DIR="$1"

    if [ -z "$SDK_DIR" ]; then
        echo "[ERROR] SDK path is required for auto-detection" >&2
        return 1
    fi

    if [ -z "$DOCKER_REPO_PATH" ]; then
        # Try to find docker directory relative to SDK_DIR
        local potential_docker_path="${SDK_DIR}/../docker"
        DOCKER_REPO_PATH="$(realpath "$potential_docker_path")"
        echo "[INFO] Auto-detected DOCKER_REPO_PATH: $DOCKER_REPO_PATH"
    fi

    # Setup the docker repository
    setup_docker_repo "$DOCKER_REPO_PATH"
    return $?
}

# Print network environment information
# Usage: print_network_info
print_network_info() {
    local network_env=$(detect_network_env)
    local git_repo_url=$(get_git_repo_url)
    echo "[INFO] Network environment: $network_env"
    echo "[INFO] Git repository URL: $git_repo_url"
}

# =============================================================================
# CI Build Functions (merged from ci_build.sh)
# =============================================================================

# Function to print CI build usage
print_ci_build_usage() {
    echo "Usage: ci_build"
    echo ""
    echo "Required Environment Variables:"
    echo "  SDK_DIR         - Path to SDK directory"
    echo "  DOCKER_PLATFORM - Target platform (x86_64, aarch64, riscv64, loongarch64, android)"
    echo ""
    echo "Optional Environment Variables (Android):"
    echo "  ANDROID_SDK_DIR - Path to Android SDK for linking (required for android platform)"
    echo ""
    echo "Optional Environment Variables:"
    echo "  REPO_PATH        - Repository path (default: current directory)"
    echo "  DOCKER_REPO_PATH - Docker utilities path (auto-detected if not set)"
    echo "  SDK_TAG          - SDK tag for version information (auto-detected if not set)"
    echo ""
    echo "Example:"
    echo "  export SDK_DIR=/path/to/sdk"
    echo "  export DOCKER_PLATFORM=android"
    echo "  export ANDROID_SDK_DIR=/path/to/android_sdk"
    echo "  export REPO_PATH=/path/to/repository"
    echo "  ci_build"
}

# Validate required environment variables for CI build
validate_ci_build_env_vars() {
    local missing_vars=()

    if [ -z "$SDK_DIR" ]; then
        missing_vars+=("SDK_DIR")
    fi

    if [ -z "$DOCKER_PLATFORM" ]; then
        missing_vars+=("DOCKER_PLATFORM")
    fi

    # Android platform requires ANDROID_SDK_DIR for linking
    if [ "$DOCKER_PLATFORM" = "android" ] && [ -z "$ANDROID_SDK_DIR" ]; then
        missing_vars+=("ANDROID_SDK_DIR")
    fi

    if [ ${#missing_vars[@]} -gt 0 ]; then
        echo "[ERROR] Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        echo ""
        print_ci_build_usage
        exit 1
    fi
}

# Set default values for CI build optional variables
set_ci_build_defaults() {
    local default_REPO_PATH="$(pwd)"

    if [ -z "$REPO_PATH" ]; then
        REPO_PATH="$default_REPO_PATH"
        echo "[INFO] Using default REPO_PATH: $REPO_PATH"
    fi

    # Auto-detect and setup DOCKER_REPO_PATH if not set
    if [ -z "$DOCKER_REPO_PATH" ]; then
        if ! auto_setup_docker_repo "$SDK_DIR"; then
            echo "[ERROR] Failed to setup Docker repository"
            exit 1
        fi
    else
        # DOCKER_REPO_PATH is manually set, just ensure it's setup correctly
        if ! setup_docker_repo "$DOCKER_REPO_PATH"; then
            echo "[ERROR] Failed to setup Docker repository: $DOCKER_REPO_PATH"
            exit 1
        fi
    fi

    # Auto-detect SDK_TAG if not set
    if [ -z "$SDK_TAG" ]; then
        # Try to extract SDK tag from SDK_DIR
        local sdk_basename=$(basename "$SDK_DIR")
        if [[ "$sdk_basename" =~ ^sdk-[0-9]{12}$ ]]; then
            SDK_TAG="$sdk_basename"
            echo "[INFO] Auto-detected SDK_TAG: $SDK_TAG"
        else
            echo "[WARNING] Cannot auto-detect SDK_TAG from SDK_DIR. Using default pattern."
            # Set a default pattern that compile_llama_cpp.sh can handle
            SDK_TAG="sdk-$(date +%Y%m%d%H%M)"
            echo "[INFO] Using generated SDK_TAG: $SDK_TAG"
        fi
    fi

    # Print Android SDK information if building for android
    if [ "$DOCKER_PLATFORM" = "android" ]; then
        if [ -n "$ANDROID_SDK_DIR" ]; then
            echo "[INFO] Android SDK Path: $ANDROID_SDK_DIR"
        fi
    fi
}

# Validate paths for CI build
validate_ci_build_paths() {
    local error_count=0

    if [ ! -d "$SDK_DIR" ]; then
        echo "[ERROR] SDK_DIR does not exist: $SDK_DIR"
        error_count=$((error_count + 1))
    fi

    if [ ! -d "$REPO_PATH" ]; then
        echo "[ERROR] REPO_PATH does not exist: $REPO_PATH"
        error_count=$((error_count + 1))
    fi

    if [ ! -f "$DOCKER_REPO_PATH/bash.sh" ]; then
        echo "[ERROR] Docker bash.sh not found: $DOCKER_REPO_PATH/bash.sh"
        error_count=$((error_count + 1))
    fi

    if [ ! -f "$REPO_PATH/.dlci/compile_llama_cpp.sh" ]; then
        echo "[ERROR] compile_llama_cpp.sh not found: $REPO_PATH/.dlci/compile_llama_cpp.sh"
        error_count=$((error_count + 1))
    fi

    # Check if SDK has required env.sh
    if [ ! -f "$SDK_DIR/env.sh" ]; then
        echo "[ERROR] SDK env.sh not found: $SDK_DIR/env.sh"
        error_count=$((error_count + 1))
    fi

    # For Android platform, validate ANDROID_SDK_DIR
    if [ "$DOCKER_PLATFORM" = "android" ]; then
        if [ ! -d "$ANDROID_SDK_DIR" ]; then
            echo "[ERROR] ANDROID_SDK_DIR does not exist: $ANDROID_SDK_DIR"
            error_count=$((error_count + 1))
        fi
    fi

    if [ $error_count -gt 0 ]; then
        echo "[ERROR] $error_count validation error(s) found. Aborting."
        exit 1
    fi
}

# Prepare Docker environment variables for CI build
prepare_ci_build_docker_envs() {
    echo "[INFO] Preparing Docker environment variables..."

    # Export all required variables for Docker
    export SDK_DIR
    export DOCKER_PLATFORM
    export REPO_PATH
    export SDK_TAG
    export SDK_DIR="$SDK_DIR"

    # Export ANDROID_SDK_DIR for Android builds
    if [ "$DOCKER_PLATFORM" = "android" ]; then
        export ANDROID_SDK_DIR
    fi

    # Collect environment variables matching the pattern
    DOCKER_ENVS=()
    while IFS='=' read -r key value; do
        DOCKER_ENVS+=(--env "${key}=${value}")
    done < <(env | grep -E '^(SDK_|CI_|sdk_|REPO_PATH|DOCKER_PLATFORM|ANDROID_)')

    echo "[INFO] Docker environment variables prepared: ${#DOCKER_ENVS[@]} variables"
}

# Main CI build execution function
ci_build() {
    echo "======================================"
    echo "CI Build Script for Llama.cpp"
    echo "======================================"
    echo "[INFO] Starting CI build execution..."
    echo "[INFO] Platform: $DOCKER_PLATFORM"
    echo "[INFO] SDK Path: $SDK_DIR"
    echo "[INFO] SDK Tag: $SDK_TAG"
    echo "[INFO] Repository Path: $REPO_PATH"
    echo ""

    # Get Docker image for the platform
    DOCKER_IMAGE=$(get_docker_image_for_platform)
    echo "[INFO] Using Docker image: $DOCKER_IMAGE"
    echo ""

    # Prepare environment variables
    prepare_ci_build_docker_envs

    # Construct Docker command
    local docker_cmd=(
        "${DOCKER_REPO_PATH}/bash.sh"
        --env "DOCKER_PLATFORM=${DOCKER_PLATFORM}"
        "${DOCKER_ENVS[@]}"
        "$DOCKER_IMAGE"
        "./.dlci/compile_llama_cpp.sh"
    )

    # Handle platform-specific Docker options
    if [ "$DOCKER_PLATFORM" = "riscv64" ]; then
        # Insert platform flag after bash.sh
        docker_cmd=(
            "${DOCKER_REPO_PATH}/bash.sh"
            --platform "linux/riscv64"
            --env "DOCKER_PLATFORM=${DOCKER_PLATFORM}"
            "${DOCKER_ENVS[@]}"
            "$DOCKER_IMAGE"
            "./.dlci/compile_llama_cpp.sh"
        )
    elif [ "$DOCKER_PLATFORM" = "android" ]; then
        # Android uses x86_64 host for cross-compilation
        docker_cmd=(
            "${DOCKER_REPO_PATH}/bash.sh"
            --env "DOCKER_PLATFORM=${DOCKER_PLATFORM}"
            "${DOCKER_ENVS[@]}"
            "$DOCKER_IMAGE"
            "./.dlci/compile_llama_cpp.sh"
        )
    fi

    echo "[INFO] Executing Docker command:"
    echo "  ${docker_cmd[*]}"
    echo ""

    # Change to repository directory
    cd "$REPO_PATH"

    # Execute the Docker command
    echo "[INFO] Starting build execution in Docker container..."
    "${docker_cmd[@]}"

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        echo "======================================"
        echo "[SUCCESS] CI build completed successfully!"
        echo "======================================"

        # Show build artifacts information
        local build_dir="${REPO_PATH}/build_${DOCKER_PLATFORM}"
        if [ -d "$build_dir" ]; then
            echo "[INFO] Build artifacts created in: $build_dir"
            if [ -f "$build_dir/bin/llama-cli" ]; then
                echo "[INFO] Main executable: $build_dir/bin/llama-cli"
            fi
        fi

        # Show log information
        echo "[INFO] Build logs are available in: /LocalRun/$(whoami)/logs/llama_cpp_compile/"

        # Create release package after successful build
        echo ""
        echo "[INFO] Creating release package..."
        if create_release_package; then
            echo "[INFO] Release package created successfully"
        else
            echo "[WARNING] Failed to create release package, but build was successful"
            # Don't fail the entire build if packaging fails
        fi
    else
        echo ""
        echo "======================================"
        echo "[FAILED] CI build failed with exit code: $exit_code"
        echo "======================================"
        echo "[INFO] Check build logs in: /LocalRun/$(whoami)/logs/llama_cpp_compile/"
        exit $exit_code
    fi
}

# =============================================================================
# Release Package Functions
# =============================================================================

# Create release package (without uploading)
# Usage: create_release_package
# Requires: REPO_PATH, DOCKER_PLATFORM, CI_COMMIT_TAG, CI_COMMIT_SHA, SDK_TAG
# Returns: 0 on success, 1 on failure
create_release_package() {
    local arch="${DOCKER_PLATFORM}"
    local build_dir="${REPO_PATH}/build_${arch}"
    local release_dir="${REPO_PATH}/release"

    echo ""
    echo "======================================"
    echo "Creating Release Package"
    echo "======================================"
    echo "[INFO] Platform: ${arch}"
    echo "[INFO] Build directory: ${build_dir}"
    echo "[INFO] Release directory: ${release_dir}"
    echo ""

    # Validate build directory exists
    if [ ! -d "${build_dir}/bin" ]; then
        echo "[ERROR] Build directory not found: ${build_dir}/bin"
        echo "[ERROR] Cannot create release package without successful build"
        return 1
    fi

    # Create clean release directory
    safe_rm -rf "${release_dir}"
    mkdir -p "${release_dir}"

    # Copy release files from build/bin
    echo "[INFO] Copying release files from ${build_dir}/bin"
    cp "${build_dir}/bin"/* "${release_dir}/"

    echo "[INFO] Files to be released:"
    ls -la "${release_dir}/"
    echo ""

    # Create version.txt with build information
    echo "[INFO] Creating version.txt"
    cat > "${release_dir}/version.txt" << EOF
Release Version: ${CI_COMMIT_TAG:-unknown}
Build Date: $(date)
Commit SHA: ${CI_COMMIT_SHA:-unknown}
Build Platform: ${arch}
Architecture: $(uname -m)
SDK Tag: ${SDK_TAG:-unknown}
EOF

    # Determine platform suffix using unified config
    local platform_suffix=$(get_platform_suffix "${arch}")
    if [[ -z "$platform_suffix" ]]; then
        echo "[WARNING] Unknown architecture: ${arch}, using generic naming"
        platform_suffix="linux-${arch}"
    fi

    local sdk_tag_transformed=""
    if [[ "$SDK_TAG" =~ V2_SOFTWARE_master_([0-9]+) ]]; then
        sdk_tag_transformed="sdk${BASH_REMATCH[1]}"
    else
        # Fallback: if pattern doesn't match, use original SDK_TAG
        sdk_tag_transformed="${SDK_TAG:-unknown}"
    fi

    # Create release package name
    local release_name="llama-${CI_COMMIT_TAG:-dev}-${sdk_tag_transformed}-bin-${platform_suffix}.zip"

    echo "[INFO] Creating release package: ${release_name}"
    cd "${REPO_PATH}"
    zip -r "${release_name}" release

    if [ $? -eq 0 ]; then
        echo ""
        echo "[SUCCESS] Release package created: ${release_name}"
        echo "[INFO] Package location: ${REPO_PATH}/${release_name}"
        ls -lh "${REPO_PATH}/${release_name}"
        echo ""
        return 0
    else
        echo "[ERROR] Failed to create release package"
        return 1
    fi
}

# CI Build entry point function
run_ci_build() {
    echo "[INFO] CI Build Script started at: $(date)"

    # Validate environment
    validate_ci_build_env_vars
    set_ci_build_defaults
    validate_ci_build_paths

    # Execute main function
    ci_build

    echo "[INFO] CI Build Script completed at: $(date)"
}