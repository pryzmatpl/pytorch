#!/bin/bash

# PyTorch Build Script for AMD Radeon RX 7900 XTX (gfx1110)
# This script builds PyTorch with ROCm support

set -e  # Exit on any error

echo "=== PyTorch ROCm Build Script ==="
echo "Target GPU: AMD Radeon RX 7900 XTX (gfx1100)"
echo "ROCm Version: $(hipconfig --version)"
echo "=================================="

# Source the build configuration
source ./build_config.sh

# Check prerequisites
echo "=== Checking Prerequisites ==="

# Check if we're in the right directory
if [ ! -f "setup.py" ]; then
    echo "Error: setup.py not found. Please run this script from the PyTorch root directory."
    exit 1
fi

# Check if submodules are initialized
if [ ! -d "third_party/pybind11" ]; then
    echo "Initializing submodules..."
    git submodule update --init --recursive
fi

# Check Python dependencies
echo "Checking Python dependencies..."
python3 -c "import setuptools, wheel, ninja" 2>/dev/null || {
    echo "Installing build dependencies..."
    pip3 install setuptools wheel ninja
}

# Clean previous builds
echo "=== Cleaning Previous Builds ==="
if [ -d "build" ]; then
    echo "Removing build directory..."
    rm -rf build
fi

if [ -d "torch/lib/build" ]; then
    echo "Removing torch/lib/build directory..."
    rm -rf torch/lib/build
fi

# Set additional environment variables for the build
export CMAKE_FRESH=1  # Force fresh CMake configuration
export FORCE_CUDA=0   # Ensure CUDA is disabled
export USE_CUDA=0     # Double-check CUDA is disabled

# Network resilience settings
export CURL_CONNECT_TIMEOUT=60
export CURL_TIMEOUT=300
export CMAKE_DOWNLOAD_TIMEOUT=300

# Build PyTorch
echo "=== Building PyTorch with ROCm Support ==="
echo "This may take 30-60 minutes depending on your system..."

# Use pip to build and install
python3 -m pip install --no-build-isolation -v -e . 2>&1 | tee build.log

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "=== Build Successful! ==="
    echo "PyTorch has been built and installed with ROCm support."
    echo "You can now test the installation with:"
    echo "  python3 -c \"import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm available: {torch.backends.cuda.is_built()}'); print(f'HIP version: {torch.version.hip}')\""
else
    echo "=== Build Failed! ==="
    echo "Check the build.log file for detailed error information."
    exit 1
fi

echo "=== Build Complete ===" 