# ✅ Canonical Test Setup Complete

## 🎉 What You Have

### **Canonical Test Audio**
- **File**: `data/audio/clean_speech_10s.wav`
- **Duration**: 10.0 seconds
- **Sample Rate**: 16kHz
- **Source**: Your recording of Wikipedia text about LLMs
- **Format**: Mono WAV (standardized for testing)

### **Ground Truth Text**
- **File**: `data/text/clean_speech_10s.txt`
- **Content**: "A large language model (LLM) is a language model trained with self-supervised machine learning on a vast amount of text, designed for natural language processing"
- **Length**: 161 characters, 25 words
- **Matches**: First 10 seconds of your audio recording

### **Full Recording Preserved**
- **File**: `data/audio/clean_speech_full.wav`
- **Duration**: 163.1 seconds (full original)
- **Use**: For longer tests or additional experiments

## 🧪 Test Suite Ready

### **Synthetic Test Files** (6 files)
- `silence_5s.wav` - Complete silence
- `white_noise_10s.wav` - White noise at 20% amplitude  
- `pink_noise_10s.wav` - Pink noise (1/f) at 20% amplitude
- `sine_sweep_20_2000hz_10s.wav` - Logarithmic sweep from 20Hz to 2kHz
- `multitone_10s.wav` - Multi-tone signal (A3, A4, A5, A6)
- `clicks_10s_10hz.wav` - Click train at 10 Hz

### **Harness Modules** (All Tested)
- **timers.py**: Lab-grade timing with zero overhead
- **audio_io.py**: Clean audio loading with validation
- **prompts.py**: Prompt management with versioning
- **evals.py**: Evaluation metrics (WER, CER, latency, memory)

## 🚀 Ready to Test

### **Next Steps**
1. **Get API Key**: `export LFM_AUDIO_API_KEY=your_liquid_ai_key`
2. **Validate Setup**: `python setup_environment.py` (should pass all checks)
3. **Launch Experiments**: `jupyter lab` → open LFM notebook

### **What You'll Test**
- **Stability**: 100 runs on your clean speech for consistency
- **Robustness**: Silence, noise, and edge case handling
- **Performance**: Latency, memory usage, accuracy metrics
- **Comparison**: Same tests on Whisper, SeamlessM4T, etc.

### **Output Schema**
All results follow the canonical format:
```json
{
  "model": "LFM-2.5-Audio-1.5B",
  "input": {"type": "audio", "file": "clean_speech_10s.wav"},
  "output": {"type": "text", "text": "...", "confidence": 0.95},
  "metrics": {"latency_ms": 312, "wer": 0.023, "memory_mb": 487}
}
```

## 🎯 Success Criteria

- **Stability**: 100 consecutive runs without crashes
- **Consistency**: <5% variance across identical runs  
- **Latency**: P95 <500ms for 10s audio segments
- **Accuracy**: WER <10% on your clean speech recording

The lab bench is complete and ready for systematic model comparison. Your canonical audio provides the ground truth for fair evaluation across all models.