# 🖥️ Multi-Device Testing Guide for Model-Lab

**Date**: January 8, 2026  
**Status**: Ready for Execution  
**Purpose**: Validate model-lab across CPU, MPS, GPU (CUDA), and TPU

---

## 🎯 Testing Strategy Overview

### Models to Test:

1. **Whisper** (base) - OpenAI ASR baseline
2. **Faster-Whisper** (base) - Optimized CTranslate2 version
3. **LFM2.5-Audio** (1.5B) - LiquidAI foundation model

### Devices to Test:

1. **CPU** - Universal fallback
2. **MPS** - Apple Silicon (M1/M2/M3) ✅ **Local testing complete**
3. **GPU** - NVIDIA CUDA (T4/A100) 🔄 **Colab testing pending**
4. **TPU** - Google TPU (v2/v5e) 🔄 **Colab testing pending**

---

## ✅ MPS Testing (Complete)

**Device**: Apple Silicon M-series  
**Date**: January 8, 2026  
**Status**: ✅ COMPLETE

### Results:

| Model          | Status     | RTF          | Notes                 |
| -------------- | ---------- | ------------ | --------------------- |
| Whisper        | ✅ Working | 0.080x       | Native MPS support    |
| Faster-Whisper | ✅ Working | 0.119x       | Native MPS support    |
| LFM2.5-Audio   | ✅ Working | 0.098-0.212x | Fixed after bug fixes |

**Key Achievement**: LFM2.5 now works on MPS after fixing processor CUDA default and audio format issues.

**Test Commands**:

```bash
source .venv/bin/activate
python scripts/run_asr.py --model whisper --dataset primary --device mps
python scripts/run_asr.py --model faster_whisper --dataset primary --device mps
python scripts/run_asr.py --model lfm2_5_audio --dataset primary --device mps
```

---

## 🔄 GPU (CUDA) Testing Plan

**Platform**: Google Colab (T4 GPU)  
**Status**: 🔄 PENDING

### Expected Results:

| Model          | Expected Status | Acceleration | Notes                      |
| -------------- | --------------- | ------------ | -------------------------- |
| Whisper        | ✅ Should work  | GPU          | Native CUDA support        |
| Faster-Whisper | ✅ Should work  | GPU          | CTranslate2 CUDA support   |
| LFM2.5-Audio   | ⚠️ Known issue  | CPU fallback | Vendor CUDA bug documented |

### Testing Approach:

**Option 1: VS Code Colab Extension** (Recommended)

```
1. Open model_lab_colab_test.ipynb
2. Select Kernel → Colab → New Colab Server → GPU (T4)
3. Run Cell 2 (Install dependencies)
4. Run Cell 5 (GPU Test)
```

**Option 2: Upload to Google Colab**

```
1. Upload model_lab_colab_test.ipynb to drive.google.com
2. Open with Google Colab
3. Runtime → Change runtime type → GPU (T4)
4. Run installation + GPU test cells
```

**Option 3: Local CUDA (If available)**

```bash
source .venv/bin/activate
python scripts/run_asr.py --model whisper --dataset primary --device cuda
python scripts/run_asr.py --model faster_whisper --dataset primary --device cuda
python scripts/run_asr.py --model lfm2_5_audio --dataset primary --device cuda
```

### Expected Outcomes:

- ✅ Whisper: ~0.05-0.10x RTF on T4
- ✅ Faster-Whisper: ~0.08-0.12x RTF on T4
- ⚠️ LFM2.5: Falls back to CPU, ~0.5-0.8x RTF

### Known Issue - LFM CUDA Bug:

The `liquid-audio` package has a vendor bug preventing CUDA usage even when CUDA is available. This is documented but not yet fixed upstream. LFM2.5 will automatically fall back to CPU on CUDA systems.

---

## 🔄 TPU Testing Plan

**Platform**: Google Colab (TPU v2-8)  
**Status**: 🔄 PENDING

### Expected Results:

| Model          | Expected Status | Acceleration | Notes                             |
| -------------- | --------------- | ------------ | --------------------------------- |
| Whisper        | ✅ Should work  | CPU/XLA      | May need XLA bridge               |
| Faster-Whisper | ✅ Should work  | CPU          | CTranslate2 doesn't use TPU       |
| LFM2.5-Audio   | ⚠️ Uncertain    | CPU/XLA      | PyTorch XLA compatibility unknown |

### Testing Approach:

**VS Code Colab Extension**:

```
1. Open model_lab_colab_test.ipynb
2. Select Kernel → Colab → New Colab Server → TPU (v2-8)
3. Run Cell 2 (Install dependencies + TPU support)
4. Run Cell 6 (TPU Test)
```

### Expected Outcomes:

- Most models will likely fall back to CPU path
- TPU acceleration requires explicit XLA compilation
- Expected RTF: 0.5-1.5x (CPU fallback)

### Note:

TPUs are optimized for TensorFlow/JAX workloads. PyTorch models like ours typically run on CPU unless explicitly compiled for XLA. This is expected behavior.

---

## 🔄 CPU Baseline Testing

**Platform**: Any system  
**Status**: ✅ AVAILABLE (not yet run standalone)

### Purpose:

Establish baseline performance without hardware acceleration.

### Testing Approach:

```bash
source .venv/bin/activate
python scripts/run_asr.py --model whisper --dataset primary --device cpu
python scripts/run_asr.py --model faster_whisper --dataset primary --device cpu
python scripts/run_asr.py --model lfm2_5_audio --dataset primary --device cpu
```

### Expected Results:

- RTF: 1.0-3.0x (slower than real-time)
- All models should work
- Useful for debugging and baseline comparison

---

## 📊 Comprehensive Testing Matrix

| Model              | CPU        | MPS         | GPU (CUDA) | TPU        |
| ------------------ | ---------- | ----------- | ---------- | ---------- |
| **Whisper**        | 🔄 Pending | ✅ Complete | 🔄 Pending | 🔄 Pending |
| **Faster-Whisper** | 🔄 Pending | ✅ Complete | 🔄 Pending | 🔄 Pending |
| **LFM2.5-Audio**   | 🔄 Pending | ✅ Complete | 🔄 Pending | 🔄 Pending |

**Legend**:

- ✅ Complete: Tested and working
- 🔄 Pending: Not yet tested
- ⚠️ Known issue: Expected to have limitations

---

## 🚀 Next Steps

### Immediate Actions:

1. **Test on Colab GPU**:

   - Use VS Code Colab extension
   - Run GPU test cell
   - Validate Whisper/Faster-Whisper on CUDA
   - Document LFM2.5 CPU fallback

2. **Test on Colab TPU**:

   - Use VS Code Colab extension
   - Run TPU test cell
   - Document CPU fallback behavior

3. **Optional CPU Baseline**:
   - Run local CPU tests for comparison
   - Establish performance baseline

### Documentation Updates:

1. Update results in comprehensive test doc
2. Add device compatibility matrix
3. Document any new issues found
4. Update production readiness assessment

---

## 📋 Test Execution Checklist

### Pre-Testing:

- [x] MPS (Apple Silicon) testing complete
- [x] Bug fixes documented
- [x] Test scripts validated
- [x] Results directory created

### Testing Phase:

- [ ] CPU baseline tests
- [ ] GPU (CUDA) tests on Colab
- [ ] TPU tests on Colab
- [ ] Cross-platform validation

### Post-Testing:

- [ ] Compile results matrix
- [ ] Update documentation
- [ ] Create device compatibility guide
- [ ] Share findings with community

---

## 🔧 Troubleshooting Guide

### Issue: Model doesn't load on GPU

**Solution**: Check CUDA availability with `torch.cuda.is_available()`

### Issue: LFM2.5 fails on CUDA

**Expected**: Known vendor bug, automatic CPU fallback implemented

### Issue: TPU not detected

**Solution**: Ensure TPU runtime selected in Colab, install torch_xla

### Issue: Out of memory

**Solution**: Use smaller model (tiny instead of base) or shorter audio files

---

## 📚 Related Documentation

- [COMPREHENSIVE_TEST_RESULTS_2026-01-08.md](COMPREHENSIVE_TEST_RESULTS_2026-01-08.md) - MPS test results
- [LFM_MPS_FIX_SUMMARY.md](LFM_MPS_FIX_SUMMARY.md) - Bug fix details
- [VSCODE_COLAB_GUIDE.md](../VSCODE_COLAB_GUIDE.md) - Colab setup instructions
- [model_lab_colab_test.ipynb](../model_lab_colab_test.ipynb) - Test notebook

---

**Testing Goal**: Validate model-lab works across all major compute platforms before production deployment and community sharing.
