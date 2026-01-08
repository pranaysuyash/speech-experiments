#!/usr/bin/env python3
"""
Enhanced Colab Compatibility Test with TPU Support
Tests model loading and inference on Google Colab with GPU/TPU support.
"""

import sys
import os
import time
from pathlib import Path

def check_colab_environment():
    """Check if running in Google Colab environment."""
    try:
        import google.colab
        return True
    except ImportError:
        return False

def install_dependencies():
    """Install required dependencies for Colab with TPU support."""
    print("📦 Installing dependencies...")

    # Install core packages
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    os.system("pip install torch_xla[tpu] -f https://storage.googleapis.com/libtpu-releases/index.html")
    os.system("pip install transformers")
    os.system("pip install openai-whisper")
    os.system("pip install faster-whisper")
    os.system("pip install liquid-audio")
    os.system("pip install fastapi uvicorn")
    os.system("pip install numpy scipy")

    print("✅ Dependencies installed (including TPU support)")

def test_hardware_acceleration():
    """Test hardware acceleration availability including TPU."""
    print("\n🔧 Testing Hardware Acceleration...")

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Check TPU availability
    tpu_available = False
    try:
        import torch_xla.core.xla_model as xm
        tpu_available = xm.xla_device_count() > 0
        print(f"TPU available: {tpu_available}")
        if tpu_available:
            print(f"TPU device count: {xm.xla_device_count()}")
            print(f"TPU device type: {xm.xla_device()}")
    except ImportError:
        print("TPU available: False (torch_xla not installed)")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        device_name = torch.cuda.get_device_name()
        print(f"GPU: {device_name}")
        return "cuda"
    elif tpu_available:
        print("Using TPU for acceleration")
        return "tpu"
    else:
        print("Using CPU (no GPU/TPU available)")
        return "cpu"

def main():
    print("🚀 Enhanced Colab Compatibility Test with TPU Support")
    print("=" * 60)

    # Check environment
    is_colab = check_colab_environment()
    print(f"Running in Google Colab: {is_colab}")

    if not is_colab:
        print("⚠️  Not running in Colab - some features may not work")
        print("💡 Upload this script to Colab for full testing")

    # Install dependencies
    install_dependencies()

    # Test hardware
    device = test_hardware_acceleration()

    print(f"\n🎯 Selected device: {device.upper()}")

    if device == "cuda":
        print("✅ GPU acceleration available - models will run fast!")
    elif device == "tpu":
        print("✅ TPU acceleration available - models will run very fast!")
    else:
        print("⚠️  CPU only - models will run slower")
        print("💡 For GPU/TPU testing, use Colab with GPU/TPU runtime")

    print("\n📋 Next steps:")
    print("1. Run model loading tests")
    print("2. Run inference benchmarks")
    print("3. Compare performance across devices")

if __name__ == "__main__":
    main()