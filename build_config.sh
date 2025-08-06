#!/bin/bash

# PyTorch Build Configuration for AMD Radeon RX 7900 XTX (gfx1110)
# This script sets up all necessary environment variables for building PyTorch with ROCm support

export USE_ROCM=1
export PYTORCH_ROCM_ARCH=gfx1100
export ROCM_PATH=/opt/rocm

# Disable CUDA since we're using ROCm
export USE_CUDA=0
export USE_CUDNN=0
export USE_CUSPARSELT=0
export USE_CUDSS=0
export USE_CUFILE=0

# Enable ROCm-specific optimizations
export USE_ROCM_KERNEL_ASSERT=1

# Build AOTriton from source to avoid network download issues
export AOTRITON_INSTALL_FROM_SOURCE=1

# Build optimizations
export MAX_JOBS=$(nproc)
export CMAKE_BUILD_TYPE=Release

# Enable distributed training support
export USE_DISTRIBUTED=1
export USE_GLOO=1
export USE_MPI=0  # Set to 1 if you have MPI installed

# Enable performance libraries
export USE_FBGEMM=1
export USE_KINETO=1
export USE_NUMPY=1
export USE_MKLDNN=1

# Disable unnecessary components for ROCm
export USE_XPU=0
export USE_VULKAN=0
export USE_OPENCL=0

# Set compiler flags for optimal performance
export CFLAGS="-O3 -march=native -mtune=native"
export CXXFLAGS="-O3 -march=native -mtune=native"
export TORCH_USE_HIP_DSA=1
# Enable debug info for better profiling
export REL_WITH_DEB_INFO=1

# Print configuration
echo "=== PyTorch ROCm Build Configuration ==="
echo "USE_ROCM: $USE_ROCM"
echo "PYTORCH_ROCM_ARCH: $PYTORCH_ROCM_ARCH"
echo "ROCM_PATH: $ROCM_PATH"
echo "MAX_JOBS: $MAX_JOBS"
echo "CMAKE_BUILD_TYPE: $CMAKE_BUILD_TYPE"
echo "USE_DISTRIBUTED: $USE_DISTRIBUTED"
echo "USE_GLOO: $USE_GLOO"
echo "USE_ROCM_KERNEL_ASSERT: $USE_ROCM_KERNEL_ASSERT"
echo "========================================"

# Verify ROCm installation
echo "=== ROCm Verification ==="
echo "ROCm version: $(hipconfig --version)"
echo "GPU architecture: $(rocm_agent_enumerator)"
echo "ROCm path exists: $(test -d $ROCM_PATH && echo "YES" || echo "NO")"
echo "=========================" 
