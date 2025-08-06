#!/bin/bash

# PyTorch Build Script for AMD Radeon RX 7900 XTX (gfx1100)
# This script builds PyTorch with ROCm support using CMake

set -e  # Exit on any error

echo "=== PyTorch ROCm Build Script ==="
echo "Target GPU: AMD Radeon RX 7900 XTX (gfx1100)"
echo "ROCm Version: $(hipconfig --version)"
echo "CMake Version: $(cmake --version | head -n 1)"
echo "=================================="

# Source the build configuration (if it exists)
if [ -f "./build_config.sh" ]; then
    echo "Sourcing build configuration..."
    source ./build_config.sh
fi

# Check prerequisites
echo "=== Checking Prerequisites ==="

# Check if we're in the PyTorch root directory
if [ ! -f "setup.py" ] || [ ! -f "CMakeLists.txt" ]; then
    echo "Error: setup.py or CMakeLists.txt not found. Please run this script from the PyTorch root directory."
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

# Check ROCm installation
echo "Checking ROCm installation..."
if ! command -v rocminfo &> /dev/null; then
    echo "Error: rocminfo not found. Ensure ROCm is installed."
    exit 1
fi
rocminfo | grep -q "gfx1100" || {
    echo "Warning: gfx1100 (Radeon RX 7900 XTX) not detected. Ensure GPU is properly configured."
}

# Clean previous builds
echo "=== Cleaning Previous Builds ==="
if [ -d "build" ]; then
    echo "Removing build directory..."
    rm -rf build
fi

# Set environment variables for ROCm and CMake
echo "=== Setting Environment Variables ==="
export ROCM_PATH=/opt/rocm
export CMAKE_PREFIX_PATH=$ROCM_PATH:$CMAKE_PREFIX_PATH
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH

# Explicitly set RCCL paths to avoid NCCL-related errors
export NCCL_INCLUDE_DIRS=$ROCM_PATH/include
export NCCL_LIBRARIES=$ROCM_PATH/lib/librccl.so

# Disable CUDA and enable ROCm
export USE_CUDA=0
export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx1100  # Target architecture for RX 7900 XTX
export CMAKE_FRESH=1  # Force fresh CMake configuration

# Network resilience settings
export CURL_CONNECT_TIMEOUT=60
export CURL_TIMEOUT=300
export CMAKE_DOWNLOAD_TIMEOUT=300

# Create and enter build directory
echo "=== Configuring Build ==="
mkdir -p build
cd build

# Run CMake with explicit RCCL and ROCm settings
echo "Running CMake configuration..."
cmake .. \
    -DUSE_ROCM=ON \
    -DUSE_CUDA=OFF \
    -DROCM_PATH=$ROCM_PATH \
    -DNCCL_INCLUDE_DIRS=$ROCM_PATH/include \
    -DNCCL_LIBRARIES=$ROCM_PATH/lib/librccl.so \
    -DPYTORCH_ROCM_ARCH=gfx1100 \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TEST=OFF \
    -DUSE_NCCL=ON \
    -DUSE_SYSTEM_NCCL=ON \
    -DUSE_PYTORCH_GLOO_WITH_ROCM=ON \
    -DCMAKE_VERBOSE_MAKEFILE=ON 2>&1 | tee cmake_config.log

# Check if CMake configuration was successful
if [ $? -ne 0 ]; then
    echo "=== CMake Configuration Failed! ==="
    echo "Check cmake_config.log for details."
    exit 1
fi

# Build PyTorch
echo "=== Building PyTorch with ROCm Support ==="
echo "This may take 30-60 minutes depending on your system..."
make -j$(nproc) 2>&1 | tee build.log

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "=== Build Successful ==="
else
    echo "=== Build Failed! ==="
    echo "Check build.log for detailed error information."
    exit 1
fi

# Install PyTorch
echo "=== Installing PyTorch ==="
make install

# Install Python wheel
echo "=== Building and Installing Python Wheel ==="
cd ..
python3 setup.py bdist_wheel 2>&1 | tee wheel_build.log

# Install the wheel
pip3 install dist/*.whl

# Verify installation
echo "=== Verifying Installation ==="
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'ROCm available: {torch.cuda.is_available()}'); print(f'HIP version: {torch.version.hip}')" || {
    echo "Verification failed. Check the installation and library paths."
    exit 1
}

echo "=== Build and Installation Complete ==="
echo "PyTorch has been built and installed with ROCm support for gfx1100."
