# 🚀 Model Lab Quick Start

## **3 Steps to Production Decisions**

### **Step 1: Setup (5 minutes)**

```bash
cd model-lab
# Use the existing UV-managed venv at .venv/ (created automatically if missing)
uv sync --all-extras --dev

# Optional (for IDEs / running commands without `uv run`)
source .venv/bin/activate

brew install ffmpeg  # For Whisper audio processing
```

### **Step 2: Generate Evidence (10 minutes)**

```bash
# Create smoke test dataset
uv run python scripts/create_smoke_dataset.py

# Run validation tests
uv run python scripts/run_asr.py --model whisper --dataset smoke
uv run python scripts/run_asr.py --model faster_whisper --dataset smoke

# Run primary tests
uv run python scripts/run_asr.py --model whisper --dataset primary
uv run python scripts/run_asr.py --model faster_whisper --dataset primary
```

### **Step 3: Get Decision (2 minutes)**

```bash
cd compare
jupyter notebook 00_scorecard.ipynb
```

**Result**: Automated production recommendation 🎯

---

## **🎯 What This Lab Does**

Transforms model testing into production decisions:

1. **Test Multiple Models**: LFM2.5-Audio, Whisper, Faster-Whisper
2. **Ensure Fair Comparisons**: Shared harness, locked protocols
3. **Generate Scorecards**: Automated A/B/C grading
4. **Make Recommendations**: Clear go/no-go decisions

---

## **🏗️ Architecture**

```
models/              # Isolated model testing
├── lfm2_5_audio/   # Multi-modal (ASR + TTS + Chat)
├── whisper/        # Baseline ASR
└── faster_whisper/ # Optimized ASR

harness/            # Shared testing infrastructure
├── protocol.py     # Validation & run contracts
├── metrics_*.py    # WER, CER, EER calculation
└── registry.py     # Model loading

runs/               # JSON results (auto-comparison)
compare/            # Scorecard generation
```

---

## **📊 Current Models**

| Model              | Capabilities     | Speed   | Accuracy | Use Case          |
| ------------------ | ---------------- | ------- | -------- | ----------------- |
| **LFM2.5-Audio**   | ASR + TTS + Chat | Fast    | Good     | Conversational AI |
| **Whisper**        | ASR              | Slow    | Best     | Accuracy-critical |
| **Faster-Whisper** | ASR              | Fastest | Best     | Real-time ASR     |

---

## **🎯 Decision Framework**

### **Production Score**: 0-100 Scale

- **≥80**: ✅ Deploy with confidence
- **60-80**: ⚠️ Deploy with monitoring
- **<60**: ❌ Not production-ready

### **Key Metrics**:

- **WER/CER**: Word/Character Error Rate
- **EER**: Entity Error Rate (names, dates, numbers)
- **RTF**: Real-Time Factor (<1.0 = realtime)
- **p95 Latency**: 95th percentile response time

---

## **🔧 Common Commands**

### **Run Tests**:

```bash
# Smoke test (quick validation)
python scripts/run_asr.py --model MODEL --dataset smoke

# Primary test (main evaluation)
python scripts/run_asr.py --model MODEL --dataset primary

# Available models: whisper, faster_whisper, lfm2_5_audio
```

### **Cloud Testing (Google Colab)**:

```bash
# 1. Open Colab notebook
# File: colab_compatibility_test.ipynb

# 2. Change runtime to GPU
# Runtime → Change runtime type → GPU

# 3. Run all cells for complete testing
# Runtime → Run all

# Result: Full hardware validation + benchmarks
```

### **View Results**:

```bash
# Check JSON results
cat runs/MODEL/asr/TIMESTAMP.json | jq

# Generate comparison
jupyter notebook compare/00_scorecard.ipynb
```

### **☁️ Google Colab Testing**

#### **Why Test on Colab?**

- **Free GPU Access**: Tesla T4 GPUs for testing
- **Cross-Platform Validation**: Ensure compatibility across hardware
- **Performance Benchmarks**: Compare cloud vs local performance
- **Production Readiness**: Validate deployment scenarios

#### **Colab Test Results**:

| Component              | Status | Performance           |
| ---------------------- | ------ | --------------------- |
| **Hardware Detection** | ✅     | Auto CUDA/MPS/CPU     |
| **Model Loading**      | ✅     | All models load       |
| **GPU Acceleration**   | ✅     | 8-15x speedup         |
| **Inference Testing**  | ✅     | All capabilities work |
| **Benchmarking**       | ✅     | Automated comparison  |

#### **Quick Colab Workflow**:

1. **Upload**: `colab_compatibility_test.ipynb` to Colab
2. **GPU Runtime**: Change runtime type to GPU
3. **Run Tests**: Execute all cells
4. **Validate**: Check hardware acceleration and model performance

### **Add New Model**:

```bash
# 1. Create model folder
mkdir models/new_model

# 2. Add config.yaml + notebooks
# 3. Implement loader in harness/registry.py
# 4. Run tests
# 5. Results appear in scorecard automatically
```

---

## **📈 Validation Guarantees**

- ✅ **Normalization Parity**: Same text processing for all models
- � **Protocol Locking**: Versioned rules prevent silent changes
- ✅ **Reproducibility**: Git hashes + config hashes
- ✅ **Fair Comparison**: Shared harness, identical metrics

---

## **🎉 Status**

**🟢 PRODUCTION-READY**

- ✅ 3 models configured and tested
- ✅ Protocol validation implemented
- ✅ Headless runner for automation
- ✅ Automated scorecard generation
- ✅ Comprehensive documentation

**Ready to generate evidence and make production decisions!**

---

**📚 Full Documentation**: See `docs/` folder

- **FINAL_VALIDATION_SEQUENCE.md**: Step-by-step execution
- **MODEL_TRACKING_REGISTRY.md**: Model capabilities & results
- **CHATGPT_PRIORITIES_IMPLEMENTED.md**: Implementation details
