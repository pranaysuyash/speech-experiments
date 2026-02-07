# Faster-Whisper Large V3

LCS-14: Systran's optimized Whisper large-v3 with CTranslate2 runtime.

## Overview

- **Architecture**: Whisper large-v3 with CTranslate2 backend
- **Parameters**: ~1.5B
- **Languages**: 99+ with auto-detection
- **Sample Rate**: 16kHz
- **License**: MIT

## Installation

```bash
make model-install MODEL=faster_whisper_large_v3
```

**Note**: First run downloads ~3GB model weights.

## Usage

```bash
# Quick test
make asr-audio MODEL=faster_whisper_large_v3 AUDIO=inputs/sample_16k.wav

# Pipeline test
make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| asr | transcribe | ✓ |

## Output Format

```python
{
    "text": "Hello world",
    "segments": [
        {"start": 0.0, "end": 1.5, "text": "Hello world", "words": [...]}
    ],
    "language": "en",
}
```

## References

- [HuggingFace](https://huggingface.co/Systran/faster-whisper-large-v3)
- [faster-whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
