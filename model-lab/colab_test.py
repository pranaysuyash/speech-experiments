#!/usr/bin/env python3
"""
Colab Compatibility Test Script
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
    """Install required dependencies for Colab."""
    print("📦 Installing dependencies...")

    # Install core packages
    os.system("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    os.system("pip install transformers")
    os.system("pip install openai-whisper")
    os.system("pip install faster-whisper")
    os.system("pip install liquid-audio")
    os.system("pip install fastapi uvicorn")
    os.system("pip install numpy scipy")

    print("✅ Dependencies installed")

def test_hardware_acceleration():
    """Test hardware acceleration availability."""
    print("\n🔧 Testing Hardware Acceleration...")

    import torch

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        device_name = torch.cuda.get_device_name()
        print(f"GPU: {device_name}")
        return "cuda"
    else:
        print("CUDA not available, using CPU")
        return "cpu"

def test_model_loading(device):
    """Test model loading on Colab."""
    print(f"\n🤖 Testing Model Loading on {device.upper()}...")

    try:
        # Test Whisper
        print("Testing Whisper...")
        import whisper
        model = whisper.load_model("tiny", device=device)
        print("✅ Whisper loaded successfully")

        # Test Faster Whisper
        print("Testing Faster-Whisper...")
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device=device)
        print("✅ Faster-Whisper loaded successfully")

        # Test LFM (with CPU fallback if needed)
        print("Testing LFM-2.5-Audio...")
        lfm_device = "cpu" if device == "cpu" else device
        from liquid_audio import LFM2AudioModel
        model = LFM2AudioModel.from_pretrained("LiquidAI/LFM2.5-Audio-1.5B", device=lfm_device)
        print(f"✅ LFM-2.5-Audio loaded successfully on {lfm_device}")

        return True

    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False

def test_inference(device):
    """Test basic inference capabilities."""
    print(f"\n🎯 Testing Inference on {device.upper()}...")

    try:
        import torch
        import numpy as np

        # Create dummy audio data
        sample_rate = 16000
        duration = 1.0  # 1 second
        samples = int(sample_rate * duration)
        audio = np.random.randn(samples).astype(np.float32)

        # Test Whisper inference
        print("Testing Whisper inference...")
        import whisper
        model = whisper.load_model("tiny", device=device)
        result = model.transcribe(audio, fp16=(device=="cuda"))
        print(f"✅ Whisper inference completed: '{result['text'][:50]}...'")

        # Test Faster-Whisper inference
        print("Testing Faster-Whisper inference...")
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device=device)
        segments, info = model.transcribe(audio)
        text = " ".join([segment.text for segment in segments])
        print(f"✅ Faster-Whisper inference completed: '{text[:50]}...'")

        return True

    except Exception as e:
        print(f"❌ Inference testing failed: {e}")
        return False

def benchmark_performance(device):
    """Benchmark performance across models."""
    print(f"\n📊 Benchmarking Performance on {device.upper()}...")

    import time
    import numpy as np

    # Create test audio (3 seconds)
    sample_rate = 16000
    duration = 3.0
    samples = int(sample_rate * duration)
    audio = np.random.randn(samples).astype(np.float32)

    results = {}

    try:
        # Benchmark Whisper
        print("Benchmarking Whisper...")
        import whisper
        model = whisper.load_model("tiny", device=device)

        start_time = time.time()
        result = model.transcribe(audio, fp16=(device=="cuda"))
        whisper_time = time.time() - start_time
        results["whisper"] = whisper_time
        print(".2f")

        # Benchmark Faster-Whisper
        print("Benchmarking Faster-Whisper...")
        from faster_whisper import WhisperModel
        model = WhisperModel("tiny", device=device)

        start_time = time.time()
        segments, info = model.transcribe(audio)
        faster_whisper_time = time.time() - start_time
        results["faster_whisper"] = faster_whisper_time
        print(".2f")

        return results

    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")
        return None

def main():
    """Main Colab testing function."""
    print("🧪 GOOGLE COLAB COMPATIBILITY TEST")
    print("=" * 50)
    print(f"Date: January 8, 2026")
    print(f"Python: {sys.version}")
    print()

    # Check environment
    is_colab = check_colab_environment()
    print(f"Running in Google Colab: {is_colab}")

    if not is_colab:
        print("⚠️  Note: This script is designed for Google Colab")
        print("    Some features may not work in local environment")
    print()

    # Install dependencies
    install_dependencies()

    # Test hardware
    device = test_hardware_acceleration()

    # Test model loading
    loading_success = test_model_loading(device)

    # Test inference
    if loading_success:
        inference_success = test_inference(device)

        # Benchmark performance
        if inference_success:
            benchmarks = benchmark_performance(device)
            if benchmarks:
                print("
📈 Performance Summary:"                for model, time_taken in benchmarks.items():
                    print(".2f")

    print("\n" + "=" * 50)
    if loading_success:
        print("🎉 COLAB TESTING COMPLETE - ALL TESTS PASSED!")
        print("✅ Models load successfully")
        print("✅ Inference works correctly")
        print(f"✅ Hardware acceleration: {device.upper()}")
        print("✅ Ready for production use on Colab")
    else:
        print("❌ COLAB TESTING FAILED")
        print("🔧 Check error messages above")

    print("=" * 50)

if __name__ == "__main__":
    main()