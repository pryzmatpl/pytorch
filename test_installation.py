#!/usr/bin/env python3
"""
Test script to verify PyTorch installation with ROCm support
for AMD Radeon RX 7900 XTX (gfx1100)
"""

import torch
import sys

def test_pytorch_installation():
    """Test PyTorch installation and ROCm support"""
    
    print("=== PyTorch ROCm Installation Test ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    
    # Check if CUDA/ROCm is available
    print(f"\n--- GPU Support ---")
    print(f"CUDA available: {torch.backends.cuda.is_built()}")
    print(f"ROCm available: {torch.backends.cuda.is_built() and torch.version.hip is not None}")
    
    if torch.version.hip:
        print(f"HIP version: {torch.version.hip}")
    
    # Check GPU device
    if torch.cuda.is_available():
        print(f"GPU device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"GPU {i} capability: {torch.cuda.get_device_capability(i)}")
        
        # Test basic tensor operations on GPU
        print(f"\n--- GPU Tensor Operations Test ---")
        try:
            device = torch.device('cuda:0')
            x = torch.randn(1000, 1000, device=device)
            y = torch.randn(1000, 1000, device=device)
            z = torch.mm(x, y)
            print("✓ Matrix multiplication on GPU successful")
            
            # Test memory allocation
            print(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            print(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
            
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
            return False
    else:
        print("✗ No GPU devices available")
        return False
    
    # Test ROCm-specific features
    print(f"\n--- ROCm Features Test ---")
    try:
        # Test if we can create tensors with specific dtypes
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            
            # Test different data types
            for dtype in [torch.float32, torch.float16, torch.int32]:
                tensor = torch.zeros(100, 100, dtype=dtype, device=device)
                print(f"✓ Created {dtype} tensor on GPU")
            
            # Test basic operations
            a = torch.randn(100, 100, device=device)
            b = torch.randn(100, 100, device=device)
            c = a + b
            d = torch.relu(c)
            print("✓ Basic operations (add, relu) successful")
            
    except Exception as e:
        print(f"✗ ROCm features test failed: {e}")
        return False
    
    print(f"\n=== Test Results ===")
    print("✓ PyTorch installation successful")
    print("✓ ROCm support enabled")
    print("✓ GPU operations working")
    print("✓ Ready for deep learning workloads!")
    
    return True

if __name__ == "__main__":
    success = test_pytorch_installation()
    sys.exit(0 if success else 1) 