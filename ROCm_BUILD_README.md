# PyTorch ROCm Build Guide for AMD Radeon RX 7900 XTX

This guide provides instructions for building PyTorch with ROCm support specifically optimized for the AMD Radeon RX 7900 XTX (gfx1100) GPU.

## System Requirements

- **GPU**: AMD Radeon RX 7900 XTX (gfx1100 architecture)
- **ROCm**: 6.4 or later (tested with 6.4.43483)
- **OS**: Linux (Arch Linux tested)
- **Python**: 3.8 or later
- **RAM**: 16GB+ recommended for compilation
- **Storage**: 20GB+ free space

## Prerequisites

### 1. ROCm Installation
Ensure ROCm is properly installed and configured:
```bash
# Check ROCm installation
hipconfig --version
rocm_agent_enumerator  # Should return gfx1100
echo $ROCM_PATH  # Should be /opt/rocm
```

### 2. Python Dependencies
Install required Python packages:
```bash
pip3 install setuptools wheel ninja
```

### 3. System Dependencies
Install system dependencies (Arch Linux):
```bash
sudo pacman -S cmake ninja gcc clang openmp
```

## Build Configuration

The build is configured through environment variables set in `build_config.sh`:

### Key Configuration Variables:
- `USE_ROCM=1`: Enable ROCm support
- `PYTORCH_ROCM_ARCH=gfx1100`: Target GPU architecture
- `ROCM_PATH=/opt/rocm`: ROCm installation path
- `USE_CUDA=0`: Disable CUDA (we're using ROCm)
- `USE_ROCM_KERNEL_ASSERT=1`: Enable kernel assertions for debugging
- `MAX_JOBS=$(nproc)`: Use all CPU cores for compilation
- `REL_WITH_DEB_INFO=1`: Include debug information

### Performance Optimizations:
- `USE_FBGEMM=1`: Enable FBGEMM for quantized operations
- `USE_KINETO=1`: Enable profiling support
- `USE_MKLDNN=1`: Enable MKL-DNN for CPU operations
- `USE_DISTRIBUTED=1`: Enable distributed training support
- `USE_GLOO=1`: Enable Gloo backend for distributed training

## Build Process

### Quick Build (Recommended)
```bash
# 1. Source the configuration
source ./build_config.sh

# 2. Run the automated build script
./build_pytorch.sh
```

### Manual Build
```bash
# 1. Set environment variables
source ./build_config.sh

# 2. Initialize submodules
git submodule update --init --recursive

# 3. Build and install
python3 -m pip install --no-build-isolation -v -e .
```

## Build Time and Resources

- **Expected Build Time**: 30-60 minutes (depending on system)
- **CPU Usage**: Will utilize all available cores
- **Memory Usage**: 8-16GB during compilation
- **Disk Space**: ~20GB for build artifacts

## Verification

After successful build, test the installation:

```bash
# Run the test script
python3 test_installation.py
```

Expected output:
```
=== PyTorch ROCm Installation Test ===
PyTorch version: 2.x.x
CUDA available: True
ROCm available: True
HIP version: 6.4.x
GPU device count: 1
GPU 0: AMD Radeon RX 7900 XTX
✓ Matrix multiplication on GPU successful
✓ PyTorch installation successful
✓ ROCm support enabled
✓ GPU operations working
✓ Ready for deep learning workloads!
```

## Troubleshooting

### Common Issues:

1. **GPU Architecture Detection**
   ```bash
   # Verify GPU detection
   rocm_agent_enumerator
   # Should return: gfx1100
   ```

2. **ROCm Path Issues**
   ```bash
   # Check ROCm installation
   ls -la /opt/rocm/
   # Ensure ROCM_PATH is set correctly
   echo $ROCM_PATH
   ```

3. **Build Failures**
   ```bash
   # Check build log
   tail -f build.log
   
   # Clean and retry
   rm -rf build torch/lib/build
   source ./build_config.sh
   python3 -m pip install --no-build-isolation -v -e .
   ```

4. **Memory Issues**
   ```bash
   # Reduce parallel jobs if out of memory
   export MAX_JOBS=8  # Instead of $(nproc)
   ```

### Performance Tuning:

1. **Optimize for Your GPU**:
   ```bash
   # The build is already optimized for gfx1100
   # No additional tuning needed
   ```

2. **Memory Management**:
   ```bash
   # PyTorch will automatically manage GPU memory
   # For large models, consider gradient checkpointing
   ```

## Usage Examples

### Basic GPU Operations:
```python
import torch

# Check GPU availability
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")

# Create tensors on GPU
device = torch.device('cuda:0')
x = torch.randn(1000, 1000, device=device)
y = torch.randn(1000, 1000, device=device)

# Perform operations
z = torch.mm(x, y)
print(f"Result shape: {z.shape}")
```

### Training Example:
```python
import torch
import torch.nn as nn

# Define model
model = nn.Sequential(
    nn.Linear(784, 512),
    nn.ReLU(),
    nn.Linear(512, 10)
).cuda()

# Training loop
optimizer = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

# Your training code here...
```

## Support

For issues specific to this build configuration:
1. Check the build log (`build.log`)
2. Verify ROCm installation
3. Ensure all environment variables are set correctly
4. Check system resources (memory, disk space)

For general PyTorch issues:
- [PyTorch Documentation](https://pytorch.org/docs/)
- [PyTorch GitHub Issues](https://github.com/pytorch/pytorch/issues)
- [ROCm Documentation](https://rocmdocs.amd.com/)

## Performance Notes

The RX 7900 XTX with ROCm 6.4 provides excellent performance for:
- Large language model training
- Computer vision tasks
- Scientific computing
- Mixed precision training (FP16)

The build is optimized for:
- Maximum GPU utilization
- Efficient memory usage
- Fast compilation times
- Debugging capabilities 