#!/bin/bash
#
# Utility functions for CI scripts
#
# This file contains common utility functions that can be shared across
# different CI scripts to avoid code duplication.
#

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

# Get Docker image based on platform and network environment
# Usage: get_docker_image
# Requires: DOCKER_PLATFORM environment variable
get_docker_image() {
    local network_env=$(detect_network_env)
    local registry_prefix=""

    # Set registry prefix based on network environment
    if [ "$network_env" = "external" ]; then
        registry_prefix="ext-artifactory.denglin.com:8082"
    else
        registry_prefix="artifactory.denglin.com:8082"
    fi

    case "$DOCKER_PLATFORM" in
        x86_64)
            echo "${registry_prefix}/ci-docker-images/c-29:manylinux_2_28-gcc12-amd64-20250703"
            ;;
        aarch64)
            echo "${registry_prefix}/ci-docker-images/c-25:manylinux_2_28-gcc12-aarch64-20250702"
            ;;
        riscv64)
            echo "${registry_prefix}/ci-docker-images/c-37:ubuntu22.04-riscv-20250822"
            ;;
        loongarch64)
            echo "${registry_prefix}/ci-docker-images/c-41:manylinux_2_38-loongarch64-20250910"
            ;;
        *)
            echo "[ERROR] Unsupported DOCKER_PLATFORM: $DOCKER_PLATFORM" >&2
            echo "[INFO] Supported platforms: x86_64, aarch64, riscv64, loongarch64" >&2
            return 1
            ;;
    esac
}

# Setup Docker repository (clone if needed and ensure main branch)
# Usage: setup_docker_repo <docker_repo_path>
# Returns: 0 on success, 1 on failure
setup_docker_repo() {
    local docker_repo_path="$1"

    if [ -z "$docker_repo_path" ]; then
        echo "[ERROR] Docker repository path is required" >&2
        return 1
    fi

    # Clone docker repo and check branch if it doesn't exist
    if [ ! -d "$docker_repo_path" ]; then
        local git_repo_url=$(get_git_repo_url)
        local network_env=$(detect_network_env)
        echo "[INFO] Docker directory does not exist, cloning from $network_env network..."
        echo "[INFO] Using repository: $git_repo_url"
        git clone "$git_repo_url" "$docker_repo_path"
    else
        echo "[INFO] Docker directory already exists, checking branch..."
        cd "$docker_repo_path"
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
    if [ ! -f "$docker_repo_path/bash.sh" ]; then
        echo "[ERROR] bash.sh not found after cloning/updating: $docker_repo_path/bash.sh" >&2
        return 1
    fi

    echo "[INFO] Docker repository setup completed: $docker_repo_path"
    return 0
}

# Auto-detect and setup Docker repository path
# Usage: auto_setup_docker_repo <sdk_path>
# Sets: DOCKER_REPO_PATH global variable
# Returns: 0 on success, 1 on failure
auto_setup_docker_repo() {
    local sdk_path="$1"

    if [ -z "$sdk_path" ]; then
        echo "[ERROR] SDK path is required for auto-detection" >&2
        return 1
    fi

    if [ -z "$DOCKER_REPO_PATH" ]; then
        # Try to find docker directory relative to SDK_PATH
        local potential_docker_path="${sdk_path}/../docker"
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
