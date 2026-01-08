# 🚀 Model Lab - Quick Start Guide

## ✅ Your Environment is Ready!

**Status**: 🟢 **GO FOR TESTING** - All systems operational

## 🎯 Immediate Next Steps (15 minutes)

### 1. Launch Jupyter Lab
```bash
cd /Users/pranay/Projects/speech_experiments/model-lab
source .venv/bin/activate
jupyter lab
```

### 2. Run Environment Validation (2 minutes)
- **File**: `test_environment.ipynb`
- **Purpose**: Confirm all dependencies work
- **Expected**: ✅ All cells should run without errors
- **What to check**:
  - All imports should work (torch, liquid-audio, etc.)
  - MPS device should be detected
  - Audio files should load successfully

### 3. Test LFM Model (5 minutes)
- **File**: `lfm_local_working.ipynb`
- **Purpose**: Test actual LFM model functionality
- **Expected**: Model should load and respond to basic commands
- **What to check**:
  - Model loads successfully (1.45B parameters)
  - Audio preprocessing works
  - Performance metrics are recorded

## 🔍 Understanding Your Test Environment

### What You Have Available
```python
# Hardware
Device: MPS (Apple Silicon GPU)
Memory: Available via psutil
Performance: Hardware acceleration active

# Software
Python: 3.12.10
PyTorch: 2.9.1
Liquid-Audio: 1.1.0
Model: LiquidAI/LFM2.5-Audio-1.5B (1.45B parameters)

# Data
Test Audio: 16 WAV files
Ground Truth: 2 TXT files
Test Types: Single-speaker, multi-speaker, synthetic
```

### Key Model Capabilities
```python
# What LFM-2.5-Audio Can Do
✅ Speech-to-Text (Transcription)
✅ Text-to-Speech (Generation)
✅ Multi-turn Conversations
✅ Mixed Modality (Text + Audio)
✅ Speaker Diarization (Multi-speaker)
```

## 📋 Testing Checklist

### Phase 1: Basic Validation ✅
- [x] Environment setup complete
- [x] Dependencies installed
- [x] Jupyter kernel configured
- [x] Model loading tested
- [x] Audio loading tested

### Phase 2: API Exploration (Current Focus)
- [ ] Test basic transcription
- [ ] Document audio input format
- [ ] Validate output format
- [ ] Measure transcription accuracy

### Phase 3: Systematic Testing (Next Steps)
- [ ] Run stability tests (100 iterations)
- [ ] Calculate WER/CER metrics
- [ ] Performance benchmarking
- [ ] Quality evaluation

### Phase 4: Model Comparison (Future)
- [ ] Add Whisper model testing
- [ ] Implement comparison framework
- [ ] Cross-model evaluation
- [ ] Production recommendation

## 🎯 Today's Focus: API Exploration

### Goal: Get Real Transcription Working

#### Step 1: Understand Audio Input Format
```python
# In lfm_local_working.ipynb, Cell 7
# Experiment with different audio input methods

# Try these approaches:
chat.add_audio(waveform.numpy(), sample_rate=24000)
# OR
chat.add_audio_with_sr(waveform, sample_rate=24000)
# OR
model.generate(audio_inputs=waveform)
```

#### Step 2: Test Text Generation
```python
# Start simple with text-only generation
chat.new_turn('user')
chat.add_text('Hello, can you hear me?')
chat.end_turn()

chat.new_turn('assistant')
# Generate response
```

#### Step 3: Validate Transcription Quality
```python
# Load ground truth
with open('data/text/clean_speech_10s.txt', 'r') as f:
    ground_truth = f.read()

# Calculate WER
from harness.evals import calculate_wer
wer = calculate_wer(transcription, ground_truth)
print(f'Word Error Rate: {wer:.3f}')
```

## 🛠️ Troubleshooting Common Issues

### Issue: "ModuleNotFoundError: No module named 'torch'"
**Status**: ✅ **FIXED** - Jupyter kernel configuration updated
**If persists**: Restart Jupyter Lab after kernel update

### Issue: "CUDA not available"
**Status**: ✅ **Expected** - Using MPS (Apple Silicon) instead
**Action**: Not an error, MPS is correct for Mac

### Issue: "Audio preprocessing failed"
**Status**: ⚠️ **API Exploration Needed**
**Action**: This is expected - liquid-audio API format needs discovery

### Issue: "Model loading takes forever"
**Status**: ✅ **Normal** - First download of 1.45B model
**Action**: Be patient, subsequent loads will be faster

## 📊 Expected Performance Targets

### Baseline Targets (Based on Model Size)
- **Latency**: <500ms for 10s audio segments
- **Memory**: <2GB for model + inference
- **Accuracy**: WER <10% on clean speech
- **Stability**: 100 consecutive runs without crashes

### Excellent Targets
- **Latency**: <250ms (real-time capable)
- **Memory**: <1GB (efficient deployment)
- **Accuracy**: WER <5% (human-level)
- **Stability**: 1000+ runs without issues

## 🎓 Learning Resources

### Liquid Audio Documentation
- **GitHub**: https://github.com/Liquid4All/liquid-audio
- **HuggingFace**: https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B
- **Examples**: Check repository for usage examples

### Your Test Infrastructure
- **Harness**: `harness/` directory contains testing utilities
- **Metrics**: `harness/evals.py` has WER/CER functions
- **Audio**: `harness/audio_io.py` handles audio loading

## 🚀 Ready to Start!

### Open These Files in Jupyter:
1. **`test_environment.ipynb`** - Validate everything works (2 min)
2. **`lfm_local_working.ipynb`** - Start LFM testing (5 min)
3. **`CURRENT_STATUS_REPORT.md`** - Comprehensive project status

### Expected Timeline:
- **Today**: API exploration and basic transcription
- **This Week**: Systematic testing and metrics gathering
- **Next Week**: Model comparison and production evaluation

## 💡 Pro Tips

1. **Start Simple**: Test text generation before audio transcription
2. **Monitor Performance**: Watch memory usage and timing
3. **Document Everything**: Save results for comparison
4. **Be Systematic**: Follow the same test sequence each time
5. **Stay Patient**: Model loading and API exploration take time

---

**You're ready to begin systematic model testing! 🎉**

The foundation is solid, the environment is working, and you have clear objectives. Focus on API exploration today and you'll have real transcription results by the end of the day.

**Next review**: Check `CURRENT_STATUS_REPORT.md` after API exploration to update implementation status.