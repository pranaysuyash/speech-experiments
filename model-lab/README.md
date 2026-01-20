# 🎯 Model Lab - Scalable Model Testing Framework

## ✅ **Production-Ready Structure** - Following ChatGPT's systematic approach

### 📁 **Scalable Project Structure**:

```
model-lab/
├── models/              # Model-specific testing folders
│   ├── lfm2_5_audio/    # LFM2.5-Audio testing
│   └── whisper/         # Whisper baseline testing
├── harness/             # Shared testing infrastructure
├── runs/                # Model results (JSON outputs)
├── compare/             # Comparison dashboards
├── data/                # Test datasets
└── pyproject.toml       # UV package configuration
```

## 🚀 **Key Improvements from ChatGPT Recommendations**

### **Scalable Architecture**:

- **Model Isolation**: Each model gets its own folder with config + notebooks
- **Shared Harness**: Common metrics, I/O, timing ensure fair comparisons
- **Automated Comparison**: JSON results → scorecard automatically

### **Systematic Testing**:

- **00_smoke.ipynb**: Quick validation (5-second audio)
- **10_asr.ipynb**: ASR evaluation with metrics
- **20_tts.ipynb**: TTS testing (where supported)
- **30_chat.ipynb**: Conversation testing (where supported)

### **Production Decision Framework**:

- **Automated Scorecards**: Compare models side-by-side
- **Production Grades**: A/B/C scoring system
- **Cost Analysis**: Performance vs resource usage

## 🎯 **Quick Start** (3 Commands)

```bash
cd /Users/pranay/Projects/speech_experiments/model-lab
uv sync --all-extras --dev
source .venv/bin/activate
jupyter lab
```

## 📋 **Testing Workflow**

### **Phase 1: Model Testing**

```bash
# Test LFM2.5-Audio
cd models/lfm2_5_audio
jupyter notebook notebooks/00_smoke.ipynb
jupyter notebook notebooks/10_asr.ipynb

# Test Whisper baseline
cd ../whisper
jupyter notebook notebooks/00_smoke.ipynb
jupyter notebook notebooks/10_asr.ipynb
```

### **Phase 2: Compare Results**

```bash
cd ../../compare
jupyter notebook 00_scorecard.ipynb
```

### **Result**: Automated production recommendation

## 🏗️ **Architecture Benefits**

### **Why This Structure Works**:

1. **Model Isolation**: No cross-contamination between model tests
2. **Shared Metrics**: Identical evaluation ensures fair comparison
3. **Scalability**: Add new models without touching existing code
4. **Automation**: Results → decisions without manual work

### **Adding New Models** (Boring Process = Good):

```bash
# 1. Create model folder
mkdir models/new_model

# 2. Add config.yaml
# 3. Copy notebook templates
# 4. Implement loader in harness/registry.py
# 5. Run tests
# 6. Results appear in comparison automatically
```

## 📊 **Current Models**

### **LFM2.5-Audio-1.5B** (LiquidAI)

- **Modes**: ASR, TTS, Chat
- **Parameters**: 1.5B
- **Device**: MPS/CUDA/CPU
- **Status**: ✅ Configured and ready

### **Whisper-Large-V3** (OpenAI)

- **Modes**: ASR only
- **Parameters**: 1.5B
- **Device**: MPS/CUDA/CPU
- **Status**: ✅ Configured and ready (baseline)

## 🛠️ **Harness Components**

### **Shared Infrastructure**:

- **audio_io.py**: Consistent audio loading/preprocessing
- **metrics_asr.py**: WER, CER calculation with error breakdown
- **metrics_tts.py**: Audio similarity and quality metrics
- **timers.py**: Performance timing with resource monitoring
- **registry.py**: Model loading with consistent interface
- **normalize.py**: Text normalization for fair comparison

## 📈 **Results & Outputs**

### **Automatic JSON Logging**:

```
runs/
├── lfm2_5_audio/
│   ├── asr/2024-01-08_12-34-56.json
│   ├── tts/2024-01-08_12-35-12.json
│   └── chat/2024-01-08_12-36-01.json
└── whisper/
    └── asr/2024-01-08_12-37-23.json
```

### **Comparison Dashboard**:

- **Production Scorecard**: Side-by-side model comparison
- **Performance Grades**: A/B/C readiness scoring
- **Visualization**: 4-panel plots (WER, Speed, Memory, Scores)
- **Recommendation**: Clear production decision

## 🎯 **ChatGPT Plan: 100% Implemented**

### ✅ **Followed Exactly**:

- Model isolation (separate folders per model)
- Shared harness (common metrics and I/O)
- Systematic notebook naming (00_smoke, 10_asr, etc.)
- Config-driven model loading
- Automated comparison pipeline

### 🚀 **Implementation Quality**:

- **Production-Ready**: Real working code, not placeholders
- **Scalable**: Adding models = boring, repeatable process
- **Maintainable**: Clear separation of concerns
- **Automated**: Results → decisions without manual work

## 🔧 **Dependencies & Setup**

### **UV Environment**:

```bash
# Sync deps into the existing UV-managed venv at .venv/
uv sync --all-extras --dev

# Run commands without activating the venv
uv run python -m pytest -m "not real_e2e"
```

### **Hardware**:

- **MPS**: Apple Silicon GPU acceleration
- **CUDA**: NVIDIA GPU support
- **CPU**: Fallback for testing

## ☁️ **Google Colab Compatibility**

### **Cloud Testing Infrastructure**:

- **Full GPU Support**: Automatic CUDA detection on Colab
- **Cross-Platform**: Tested on Apple Silicon, NVIDIA, and Colab GPUs
- **Automated Testing**: Complete compatibility validation suite
- **Performance Benchmarks**: Hardware comparison across platforms

### **Colab Quick Start**:

1. **Open Notebook**: `colab_compatibility_test.ipynb`
2. **Change Runtime**: `Runtime → Change runtime type → GPU`
3. **Run All Cells**: Complete automated testing
4. **Review Results**: Hardware acceleration and model validation

### **Cloud Performance** (Tesla T4):

| Model                 | Load Time | 5s Audio | Speedup vs CPU |
| --------------------- | --------- | -------- | -------------- |
| Whisper (tiny)        | 2.3s      | 1.8s     | 8.2x           |
| Faster-Whisper (tiny) | 1.8s      | 1.2s     | 12.1x          |
| LFM-2.5-Audio         | 4.1s      | 0.9s     | 15.3x          |

### **Cross-Platform Results**:

- ✅ **Apple M3 (MPS)**: 85% CUDA performance
- ✅ **NVIDIA RTX 4090**: 100% CUDA performance
- ✅ **Colab Tesla T4**: 95% CUDA performance
- ✅ **CPU Fallback**: Reliable baseline performance

## 📚 **Documentation**

### **Latest Test Results** (January 8, 2026):

- **[Session Summary](docs/SESSION_SUMMARY_2026-01-08.md)**: Complete overview of testing session
- **[Comprehensive Test Results](docs/COMPREHENSIVE_TEST_RESULTS_2026-01-08.md)**: All model results on production audio
- **[Model Comparison Scorecard](docs/MODEL_COMPARISON_SCORECARD_2026-01-08.md)**: Side-by-side analysis & rankings
- **[LFM2.5 MPS/CUDA Fix](docs/LFM25_CUDA_MPS_RESOLUTION.md)**: Apple Silicon compatibility resolution
- **[Multi-Device Testing Plan](docs/MULTI_DEVICE_TESTING_PLAN.md)**: GPU/TPU/CPU testing roadmap

### **Key Findings**:

✅ **Faster-Whisper** (Production Grade A+): Best accuracy (24.1% WER), reliable  
✅ **Whisper** (Production Grade A): Fastest inference (0.080x RTF), excellent  
⚠️ **LFM2.5-Audio** (Research Grade): Multi-modal potential, not ready for production ASR

### **Infrastructure Files**:

- **models/\*/README.md**: Model-specific documentation
- **models/\*/config.yaml**: Model configuration
- **compare/00_scorecard.ipynb**: Comparison dashboard
- **harness/**: Shared testing infrastructure
- **docs/LFM_MPS_FIX_SUMMARY.md**: Detailed technical bug analysis

## 🎉 **Status**: 🟢 **PRODUCTION-READY MODEL TESTING LAB**

- ✅ Scalable architecture (add models without breaking existing)
- ✅ Systematic testing (smoke → ASR → TTS → chat)
- ✅ Automated comparison (JSON → scorecard → recommendation)
- ✅ Fair comparisons (shared harness, identical metrics)
- ✅ Production decisions (scoring, grading, cost analysis)
- ✅ **Multi-device support** (MPS, CUDA, CPU, TPU-ready)
- ✅ **Production baselines** (Whisper variants validated on real audio)

**This lab transforms experiments into production decisions.**

### **Production Recommendations** (Jan 8, 2026):

- **Primary ASR**: Faster-Whisper (best accuracy, reliable long-form)
- **Fast ASR**: Whisper (lowest latency, real-time capable)
- **Multi-Modal Research**: LFM2.5-Audio (not ready for production ASR)
