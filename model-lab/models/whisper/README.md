# Whisper Model Testing

## Overview

This folder contains systematic testing for **OpenAI Whisper-Large-V3** as a baseline for ASR comparison.

## Model Specifications

- **Provider**: OpenAI
- **Parameters**: 1.5B (large-v3)
- **Modes**: ASR only
- **Device**: MPS (Apple Silicon), CUDA, or CPU
- **Precision**: float16/float32

## Notebook Structure

### Core Testing

1. **`00_smoke.ipynb`** - Smoke test (5-second audio)
2. **`10_asr.ipynb`** - ASR evaluation (speech-to-text)

## Usage

### Quick Start

```bash
# From model-lab root
cd models/whisper
jupyter notebook notebooks/00_smoke.ipynb
```

### Installation

```bash
pip install openai-whisper
```

## Configuration

Edit `config.yaml` to adjust:
- Device selection (mps/cuda/cpu)
- Language detection settings
- Audio processing parameters

## Results

Results are automatically saved to `runs/whisper/` with timestamps:
```
runs/whisper/
└── asr/
    └── 2024-01-08_12-34-56.json
```

## Comparison with LFM2.5-Audio

### Advantages of Whisper
- **Mature**: Well-tested, widely deployed
- **Accurate**: State-of-the-art ASR performance
- **Multilingual**: Supports 99 languages
- **Robust**: Handles noisy audio well

### Disadvantages
- **ASR-only**: No TTS or conversation capabilities
- **Slower**: Higher latency for real-time applications
- **Larger**: Larger model size for same parameter count

### LFM2.5-Audio Advantages
- **Multi-modal**: ASR + TTS + conversation
- **Faster**: Lower latency for real-time use
- **Smaller**: More efficient architecture

## Performance Targets

- **Latency**: <1000ms for 10s audio clip
- **Memory**: <2GB for inference
- **WER**: <5% on clean speech (state-of-the-art)
- **CER**: <3% on clean speech

## Notes

- Whisper serves as the baseline for ASR performance
- Use this to validate the test harness works correctly
- Compare LFM2.5-Audio ASR performance against Whisper

## Troubleshooting

### Common Issues

1. **Installation issues**: Ensure ffmpeg is installed
   ```bash
   brew install ffmpeg  # macOS
   ```

2. **Slow inference**: Use fp16 precision on GPU/MPS
3. **Language detection errors**: Set `fallback_language` in config

### Dependencies

```bash
pip install openai-whisper
pip install torch torchaudio
```