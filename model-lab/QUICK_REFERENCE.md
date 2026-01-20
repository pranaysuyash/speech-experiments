# 🚀 Quick Reference - Model Lab Testing

**Updated**: January 8, 2026  
**Status**: Production Ready

---

## ⚡ Quick Commands

### Activate Environment:

```bash
cd /Users/pranay/Projects/speech_experiments/model-lab
uv sync --all-extras --dev
source .venv/bin/activate
```

### Run Tests (MPS):

```bash
# All models on PRIMARY dataset (163s audio)
uv run python scripts/run_asr.py --model whisper --dataset primary --device mps
uv run python scripts/run_asr.py --model faster_whisper --dataset primary --device mps
uv run python scripts/run_asr.py --model lfm2_5_audio --dataset primary --device mps

# All models on CONVERSATION dataset (944s audio)
uv run python scripts/run_asr.py --model whisper --dataset conversation --device mps
uv run python scripts/run_asr.py --model faster_whisper --dataset conversation --device mps
uv run python scripts/run_asr.py --model lfm2_5_audio --dataset conversation --device mps
```

### View Results:

```bash
# List recent test runs
ls -lt runs/*/asr/*.json | head -10

# View specific result
cat runs/faster_whisper/asr/2026-01-08_14-27-50.json | python -m json.tool
```

---

## 📊 Latest Test Results (Jan 8, 2026)

### Production Ready Models (MPS):

| Model              | WER    | RTF    | Status           |
| ------------------ | ------ | ------ | ---------------- |
| **Faster-Whisper** | 24.1%  | 0.119x | ✅ Grade A+      |
| **Whisper**        | 28.5%  | 0.080x | ✅ Grade A       |
| **LFM2.5-Audio**   | 137.8% | 0.212x | ⚠️ Research Only |

**Recommendation**: Use Faster-Whisper for production ASR

---

## 📚 Documentation Links

### Test Results:

- [Session Summary](docs/SESSION_SUMMARY_2026-01-08.md)
- [Comprehensive Results](docs/COMPREHENSIVE_TEST_RESULTS_2026-01-08.md)
- [Model Comparison](docs/MODEL_COMPARISON_SCORECARD_2026-01-08.md)

### Technical Guides:

- [LFM MPS Fix](docs/LFM25_CUDA_MPS_RESOLUTION.md)
- [Multi-Device Testing](docs/MULTI_DEVICE_TESTING_PLAN.md)
- [Twitter Reply Draft](docs/TWITTER_REPLY_MAXIME_LFM25.md)

---

## 🔧 Dependencies

**Environment**: Python 3.12.10 with UV  
**Location**: `.venv/` (existing venv)

### Key Packages:

- torch 2.9.1
- openai-whisper 20250625
- faster-whisper 1.2.1
- liquid-audio 1.1.0

---

## 🎯 Next Actions

### Immediate:

1. ✅ All local MPS testing complete
2. 📝 Documentation ready for community
3. 🐦 Twitter reply draft prepared

### Optional:

1. Test on Colab GPU (T4)
2. Test on Colab TPU
3. Post to Twitter/X
4. Deploy production endpoint

---

## 🚀 Production Deployment

**Primary Model**: Faster-Whisper (base)  
**Device**: MPS (Apple Silicon)  
**Expected RTF**: 0.119x (sub-realtime)  
**Expected WER**: ~24%

**Deployment Command**:

```bash
python scripts/run_asr.py \
  --model faster_whisper \
  --dataset primary \
  --device mps
```

---

## 🐛 Known Issues

### LFM2.5-Audio:

- ⚠️ High WER (137.8%) - not optimized for pure ASR
- ⚠️ Underpredicts on long audio (14% of expected output)
- ✅ Multi-modal capabilities remain valuable

### Resolved:

- ✅ MPS compatibility fixed (processor CUDA default)
- ✅ Audio format conversion working (numpy→tensor)

---

## 💡 Tips

1. **Use existing venv**: Already configured with UV
2. **MPS tested**: All models working on Apple Silicon
3. **Results in runs/**: JSON manifests with full metadata
4. **Documentation complete**: Ready for sharing

---

**All systems ready for production and community contribution! 🎉**
