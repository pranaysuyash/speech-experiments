# 🎯 Model Registry & Test Results Tracking

## **Provider Registry** (Live Document)

### **Format**:
- Model Name
- Inference Type (Local/API/Browser)
- Size on Disk
- Capabilities (STT/TTS/Other)
- Test Results (by Dataset)
- Notes

---

## **Current Providers**

### **1. LFM2.5-Audio-1.5B** (LiquidAI)

| Field | Value |
|-------|-------|
| **Model Name** | LFM2.5-Audio-1.5B |
| **Provider** | LiquidAI |
| **Inference Type** | Local |
| **Size on Disk** | ~2.8GB (model files) |
| **Parameters** | 1.5B |
| **STT** | ✅ Yes |
| **TTS** | ✅ Yes |
| **Conversation** | ✅ Yes |
| **Other Capabilities** | Multi-modal interleaved processing |
| **Device Support** | MPS, CUDA, CPU |
| **Precision** | bfloat16, float16, float32 |
| **Status** | 🟢 Configured & Ready |
| **Config Path** | `models/lfm2_5_audio/config.yaml` |

**Test Results**:
- **SMOKE Dataset**: 🔄 Pending
- **PRIMARY Dataset**: 🔄 Pending
- **CONVERSATION Dataset**: 🔄 Pending

**Notes**:
- Only model with TTS + Conversation capabilities
- Supports multi-modal audio/text interleaving
- Best for conversational applications
- Uses official liquid-audio API

---

### **2. Whisper-Large-V3** (OpenAI)

| Field | Value |
|-------|-------|
| **Model Name** | openai/whisper-large-v3 |
| **Provider** | OpenAI |
| **Inference Type** | Local |
| **Size on Disk** | ~3.0GB (model files) |
| **Parameters** | 1.5B |
| **STT** | ✅ Yes |
| **TTS** | ❌ No |
| **Conversation** | ❌ No |
| **Other Capabilities** | Language detection, multilingual (99 languages) |
| **Device Support** | MPS, CUDA, CPU |
| **Precision** | float16, float32 |
| **Status** | 🟢 Configured & Ready |
| **Config Path** | `models/whisper/config.yaml` |

**Test Results**:
- **SMOKE Dataset**: 🔄 Pending
- **PRIMARY Dataset**: 🔄 Pending
- **CONVERSATION Dataset**: 🔄 Pending

**Notes**:
- Original Whisper implementation
- State-of-the-art ASR accuracy
- Slower but most accurate baseline
- Mature, well-tested model
- Requires `openai-whisper` package

---

### **3. Faster-Whisper-Large-V3** (guillaumekln)

| Field | Value |
|-------|-------|
| **Model Name** | guillaumekln/faster-whisper-large-v3 |
| **Provider** | guillaumekln |
| **Inference Type** | Local |
| **Size on Disk** | ~1.5GB (optimized model) |
| **Parameters** | 1.5B (same architecture) |
| **STT** | ✅ Yes |
| **TTS** | ❌ No |
| **Conversation** | ❌ No |
| **Other Capabilities** | Same as Whisper, optimized inference |
| **Device Support** | CPU, CUDA (no MPS direct) |
| **Precision** | float16, int8_float16 |
| **Status** | 🟢 Configured & Ready |
| **Config Path** | `models/faster_whisper/config.yaml` |

**Test Results**:
- **SMOKE Dataset**: 🔄 Pending
- **PRIMARY Dataset**: 🔄 Pending
- **CONVERSATION Dataset**: 🔄 Pending

**Notes**:
- 4x+ faster than original Whisper
- Lower memory footprint
- Same accuracy as Whisper
- Uses CTranslate2 engine
- Better for real-time applications
- Requires `faster-whisper` package

---

## **Dataset Registry**

### **Test Datasets**:

#### **1. SMOKE Dataset** (Quick Validation)
- **Audio**: `llm_recording_pranay_10s.wav`
- **Text**: `llm_10s.txt` (first 200 chars)
- **Duration**: ~10 seconds
- **Purpose**: Quick smoke test for all models
- **Status**: ✅ Created
- **Hash**: TBD

#### **2. PRIMARY Dataset** (Main Test)
- **Audio**: `llm_recording_pranay.m4a`
- **Text**: `llm.txt` (full Wikipedia text)
- **Duration**: ~2 minutes
- **Purpose**: Main ASR evaluation
- **Status**: ✅ Available
- **Hash**: TBD

#### **3. CONVERSATION Dataset** (Multi-Speaker)
- **Audio**: `UX_Psychology_From_Miller_s_Law_to_AI.m4a`
- **Text**: None (NotebookLM podcast)
- **Duration**: ~15 minutes
- **Purpose**: Multi-speaker conversation analysis
- **Status**: ✅ Available
- **Hash**: TBD

---

## **Test Results Registry**

### **Result Format**:
```json
{
  "provider_id": "model_name",
  "dataset": "smoke|primary|conversation",
  "wer": 0.05,
  "cer": 0.03,
  "latency_ms_p50": 180,
  "rtf": 0.42,
  "timestamp": "2026-01-08T12:34:56",
  "protocol_version": "1.0"
}
```

### **Current Results**: (Awaiting first test runs)

---

## **Protocol Versioning**

### **Current Protocol Versions**:
- **Normalization**: v1.0
- **Entity Extraction**: v1.0
- **WER/CER**: v1.0
- **JSON Schema**: v1.0

### **Locked Rules**:
- **Normalization**: Lowercase, strip punctuation, normalize whitespace, expand contractions
- **Entity Numbers**: `\b\d+(?:\.\d+)?\b` (decimals included)
- **Entity Dates**: MM/DD/YYYY, YYYY-MM-DD, Month DD, YYYY
- **Entity Currency**: `$10.50`, `$1,000.00` formats

---

## **Run Manifest** (Reproducibility)

Each run includes:
```json
{
  "provider_id": "model_name",
  "git_hash": "abc123...",
  "timestamp": "2026-01-08T12:34:56",
  "provider_versions": {
    "openai-whisper": "20231117",
    "faster-whisper": "1.0.3",
    "liquid-audio": "0.1.0"
  },
  "config_hash": "def456...",
  "dataset_hash": "ghi789...",
  "audio_file": "filename.wav",
  "text_file": "filename.txt"
}
```

---

## **Status Legend**
- 🟢 **Configured & Ready**: All setup complete, awaiting tests
- 🟡 **Testing**: Tests in progress
- 🟢 **Complete**: All tests passed
- 🔴 **Failed**: Tests failed, needs attention
- 🔄 **Pending**: Awaiting configuration or setup

---

## **Next Steps**

### **Immediate**:
1. Create SMOKE dataset: `python scripts/create_smoke_dataset.py`
2. Run smoke tests: `python scripts/run_asr.py --model MODEL --dataset smoke`
3. Review first scorecard: `jupyter notebook compare/00_scorecard.ipynb`

### **Data Collection**:
- Track WER, CER, latency, RTF for all models
- Monitor p95 latency spikes
- Check entity error rates (numbers, dates)
- Measure run-to-run variance

### **Decision Criteria**:
- **Production Winner**: Lowest EER + stable p95 latency
- **Real-Time Winner**: Lowest RTF (<1.0x)
- **Accuracy Winner**: Lowest WER/CER

---

**📊 This document tracks all model capabilities, test results, and protocol versions for reproducible comparisons.**

**Last Updated**: 2026-01-08
**Protocol Version**: 1.0
**Next Review**: After first smoke tests complete