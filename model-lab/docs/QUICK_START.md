# Quick Start Guide

## ✅ What's Ready

- **Environment**: uv-based Python 3.11 with all dependencies
- **Harness**: Clean instrumentation modules (timers, audio_io, prompts, evals)
- **Test Data**: 6 synthetic audio files + framework for canonical recording
- **Notebook**: Proper LFM-2.5-Audio testing structure
- **Test Plan**: Complete testing methodology defined

## 🎯 Next Steps (In Order)

### 1. Get API Access
```bash
export LFM_AUDIO_API_KEY="your_liquid_ai_api_key_here"
```

### 2. Record Canonical Test Audio
Follow `data/RECORDING_INSTRUCTIONS.md`:
- Record yourself reading 10-15 seconds of Wikipedia text
- Save as `data/audio/clean_speech_10s.wav`
- Save exact text as `data/audio/clean_speech_10s.txt`

### 3. Validate Setup
```bash
python setup_environment.py
```
Should show all checks passing.

### 4. Launch Experiment
```bash
jupyter lab
# Open notebooks/audio/lfm2_5_audio.ipynb
```

## 📋 Test Structure (Non-Negotiable)

Every notebook follows this exact order:

1. **Metadata** (model, date, hardware, precision)
2. **Canonical Audio** (same file across all models)
3. **Model Initialization** (isolated from inference)
4. **Single Responsibility Functions** (one function per task)
5. **Run + Log** (latency, memory, basic validation)
6. **Instrumentation** (system monitoring before quality)

## 🧪 Test Axes

**Input**: Audio (speech, music, noise, silence)  
**Output**: Text (transcription, analysis, translation)  
**Constraints**: <500ms latency, <2GB memory  
**Failure Modes**: Silence, hallucination, truncation, drift  
**Metrics**: WER, CER, latency distribution, confidence calibration

## 📊 Output Schema

All results follow this canonical format:
```json
{
  "model": "LFM-2.5-Audio-1.5B",
  "input": {"type": "audio", "file": "clean_speech_10s.wav"},
  "output": {"type": "text", "text": "...", "confidence": 0.95},
  "metrics": {"latency_ms": 312, "wer": 0.023, "memory_mb": 487}
}
```

## 🚫 Common Traps Avoided

- ✅ Same audio file across all models
- ✅ Same sample rate (16kHz) for all tests
- ✅ No SDK defaults deciding behavior
- ✅ Stability testing before quality evaluation
- ✅ System monitoring before model judgments
- ✅ Versioned results for comparison

## 🔬 After LFM-2.5-Audio

1. **Clone notebook** for Whisper, SeamlessM4T, GPT-4o-mini-audio
2. **Run identical tests** on same audio files
3. **Plot latency vs quality** for decision making
4. **Scale to production** with unified abstraction

## 📁 File Structure

```
model-lab/
├── data/audio/              # Test audio files
├── harness/                 # Reusable instrumentation
├── notebooks/audio/         # Model-specific experiment logs
├── TEST_PLAN.md            # Complete testing methodology
├── RECORDING_INSTRUCTIONS.md # Canonical audio requirements
└── setup_environment.py    # Validation script
```

## 🎯 Success Criteria

- **Stability**: 100 runs without crashes
- **Consistency**: <5% variance across identical runs  
- **Latency**: P95 <500ms for 10s audio
- **Memory**: No leaks over extended testing
- **Accuracy**: WER <10% on clean speech

The lab bench is ready. Record your canonical audio and start systematic testing.