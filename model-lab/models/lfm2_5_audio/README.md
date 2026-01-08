# LFM2.5-Audio Model Testing

## Overview

This folder contains systematic testing for the **LiquidAI LFM-2.5-Audio-1.5B** model.

## Model Specifications

- **Provider**: Liquid AI
- **Parameters**: 1.5B
- **Modes**: ASR, TTS, Chat
- **Device**: MPS (Apple Silicon), CUDA, or CPU
- **Precision**: bfloat16/float16/float32

## Notebook Structure

### Core Testing

1. **`00_smoke.ipynb`** - Smoke test (5-second audio)
2. **`10_asr.ipynb`** - ASR evaluation (speech-to-text)
3. **`20_tts.ipynb`** - TTS evaluation (text-to-speech)
4. **`30_chat.ipynb`** - Multi-turn conversation

### Legacy Notebooks (being migrated)

- `lfm_complete_working.ipynb` - Original implementation
- `test_environment.ipynb` - Environment validation
- `asr_evaluation.ipynb` - ASR evaluation with user recordings
- `tts_evaluation.ipynb` - TTS evaluation
- `conversation_analysis.ipynb` - Conversation analysis

## Usage

### Quick Start

```bash
# From model-lab root
cd models/lfm2_5_audio
jupyter notebook notebooks/00_smoke.ipynb
```

### Configuration

Edit `config.yaml` to adjust:
- Device selection (mps/cuda/cpu)
- Data type precision
- Audio processing parameters
- Testing constraints

## Results

Results are automatically saved to `runs/lfm2_5_audio/` with timestamps:
```
runs/lfm2_5_audio/
├── asr/
│   └── 2024-01-08_12-34-56.json
├── tts/
│   └── 2024-01-08_12-35-12.json
└── chat/
    └── 2024-01-08_12-36-01.json
```

## Comparison

Compare results with other models using:
```bash
jupyter notebook ../../compare/00_scorecard.ipynb
```

## Troubleshooting

### Common Issues

1. **MPS not available**: Falls back to CPU automatically
2. **Out of memory**: Reduce `max_audio_length` in config
3. **Slow inference**: Check device configuration, ensure MPS is used

### Dependencies

Install required packages:
```bash
pip install liquid-audio torch torchaudio
```

## Performance Targets

- **Latency**: <500ms for 10s audio clip
- **Memory**: <2GB for inference
- **WER**: <10% on clean speech
- **CER**: <5% on clean speech

## Notes

- This model supports both ASR and TTS, unlike Whisper which is ASR-only
- Multi-turn conversation capabilities are unique to this model
- Best results with bfloat16 precision on MPS