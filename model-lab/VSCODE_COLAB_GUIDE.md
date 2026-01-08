# 🚀 VS Code Colab Integration Guide

**Date:** January 8, 2026

## 🎯 Using Colab Extension in VS Code

Since you have the **Google Colab** extension installed, you can run notebooks directly in VS Code connected to Colab servers.

### Method 1: Direct Colab Connection (Recommended)

1. **Open the notebook**: `model_lab_colab_test.ipynb`
2. **Select Kernel**:
   - Click the kernel picker in the top-right
   - Choose **"Select Kernel"**
   - Select **"Colab"** → **"New Colab Server"**
3. **Connect**: VS Code will open Colab in your browser and connect
4. **Run cells**: Execute the notebook with GPU acceleration

### Method 2: Upload to Colab

1. **Open Colab**: Go to https://colab.research.google.com/
2. **Upload**: Click "Upload" tab and select `model_lab_colab_test.ipynb`
3. **GPU Runtime**: Runtime → Change runtime type → GPU
4. **Run All**: Runtime → Run all cells

## 📋 What the Test Does

The `model_lab_colab_test.ipynb` notebook will:

1. **Install Dependencies**: PyTorch, Whisper, LFM models
2. **Hardware Detection**: Check GPU availability
3. **Model Loading**: Test all 3 models (Whisper, Faster-Whisper, LFM)
4. **Inference Testing**: Run actual transcription on test audio
5. **Performance Benchmarking**: Compare model speeds
6. **Results Summary**: Show compatibility status

## 🎯 Expected Results

On Colab Tesla T4 GPU:

- **Hardware**: CUDA available ✅
- **Model Loading**: All models load successfully ✅
- **Inference**: ~8-15x speedup vs CPU ✅
- **Compatibility**: Full cross-platform support ✅

## 🔧 Troubleshooting

### If Colab Connection Fails:

1. Make sure you're signed into Google account
2. Try refreshing the browser tab
3. Check internet connection

### If GPU Not Available:

1. Runtime → Change runtime type → GPU
2. Wait for GPU allocation (may take a minute)
3. Check "Runtime" → "View resources" to confirm GPU

### If Models Fail to Load:

1. Check Colab Pro status (free tier has limits)
2. Try restarting runtime
3. Some models may need Colab Pro for full access

## 📊 Performance Expectations

| Component           | Free Colab | Colab Pro |
| ------------------- | ---------- | --------- |
| **GPU Memory**      | 16GB       | 16GB+     |
| **Model Loading**   | ✅ Works   | ✅ Works  |
| **Inference Speed** | Good       | Excellent |
| **Session Time**    | 12 hours   | 24 hours  |

---

**🎯 Ready to test!** Use either method above to run the Colab compatibility test.
