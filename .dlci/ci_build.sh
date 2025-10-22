#!/bin/bash
#
# CI Build Script for Llama.cpp
#
# This script runs CI builds by launching Docker containers and executing
# compile_llama_cpp.sh to compile the llama.cpp project.
#
# Required Environment Variables:
#   SDK_PATH        - Path to the SDK directory (e.g., /path/to/sdk)
#   DOCKER_PLATFORM - Target platform (x86_64, aarch64, riscv64, loongarch64, android)
#
# Optional Environment Variables:
#   REPO_PATH       - Path to the repository (default: current directory)
#   DOCKER_REPO_PATH - Path to Docker utilities (default: auto-detected)
#   SDK_TAG         - SDK tag for version information (default: auto-detected)
#

set -e

# Load utility functions
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/utils.sh"

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    # Script is being executed directly
    run_ci_build
else
    # Script is being sourced, just make functions available
    echo "[INFO] CI build functions loaded from utils.sh"
fi