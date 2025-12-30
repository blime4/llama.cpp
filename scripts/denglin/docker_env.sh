#!/bin/bash

# Docker environment setup script for llama.cpp development
# This script sets up the SDK environment and PATH for development

echo "Setting up Docker development environment..."

# Set SDK environment variables
export SDK_DIR="/SDK_DIR"
echo "SDK_DIR: $SDK_DIR"

# Check if SDK exists
if [ ! -d "$SDK_DIR" ]; then
    echo "[ERROR] SDK path does not exist: $SDK_DIR"
    echo "Please make sure the SDK is properly mounted in docker container"
    return 1
fi

# Source SDK environment
if [ -f "$SDK_DIR/env.sh" ]; then
    echo "Sourcing SDK environment from $SDK_DIR/env.sh..."
    source "$SDK_DIR/env.sh"
    echo "SDK environment loaded successfully"
else
    echo "[WARNING] SDK env.sh not found at $SDK_DIR/env.sh"
fi

# Set up PATH for build/bin commands
BUILD_BIN_PATH="/workspace/build/bin"
if [ -d "$BUILD_BIN_PATH" ]; then
    echo "Adding $BUILD_BIN_PATH to PATH..."
    export PATH="$BUILD_BIN_PATH:$PATH"
    echo "Build binaries are now available in PATH"
else
    echo "[WARNING] Build directory not found at $BUILD_BIN_PATH"
    echo "Please compile llama.cpp first: bash .dlci/compile_llama_cpp.sh"
fi

# Set up LD_LIBRARY_PATH for build libraries
if [ -d "$BUILD_BIN_PATH" ]; then
    export LD_LIBRARY_PATH="$BUILD_BIN_PATH:$LD_LIBRARY_PATH"
    echo "Build libraries added to LD_LIBRARY_PATH"
fi

# Display available commands
echo ""
echo "=== Available Commands ==="
if [ -f "$BUILD_BIN_PATH/llama-cli" ]; then
    echo "✓ llama-cli"
fi
if [ -f "$BUILD_BIN_PATH/llama-server" ]; then
    echo "✓ llama-server"
fi
if [ -f "$BUILD_BIN_PATH/llama-bench" ]; then
    echo "✓ llama-bench"
fi
if [ -f "$BUILD_BIN_PATH/test-backend-ops" ]; then
    echo "✓ test-backend-ops"
fi

echo ""
echo "=== Environment Summary ==="
echo "SDK_DIR: $SDK_DIR"
echo "BUILD_BIN: $BUILD_BIN_PATH"
echo "PATH includes build/bin: $(echo $PATH | grep -q build/bin && echo "Yes" || echo "No")"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-Not set}"

# alias
ALIAS_FILE="$(dirname "${BASH_SOURCE[0]}")/aliases.sh"
if [ -f "$ALIAS_FILE" ]; then
    echo "Sourcing developer aliases from $ALIAS_FILE"
    source "$ALIAS_FILE"
else
    echo "[WARNING] Alias file $ALIAS_FILE not found."
fi

export REPO_PATH="/workspace"

export PATH=$(echo "$PATH" | tr ':' '\n' | awk '/ccache/{ccache=$0; next} {print} END{if(ccache) print ccache}' | paste -sd:)

echo ""
echo "Environment setup complete! You can now use llama.cpp commands."
