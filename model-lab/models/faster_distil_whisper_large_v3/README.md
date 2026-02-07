# Faster-Distil-Whisper Large V3

LCS-15: Distilled Whisper with CTranslate2 for 2-3x faster inference.

## Overview

- **Architecture**: Distilled Whisper large-v3 + CTranslate2
- **Speed**: 2-3x faster than full Whisper
- **Languages**: 99+ with auto-detection
- **Sample Rate**: 16kHz
- **License**: MIT

## Installation

```bash
make model-install MODEL=faster_distil_whisper_large_v3
```

## Usage

```bash
# Quick test
make asr-audio MODEL=faster_distil_whisper_large_v3 AUDIO=inputs/sample_16k.wav

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
    "segments": [{"start": 0.0, "end": 1.5, "text": "Hello world"}],
    "language": "en",
}
```

## References

- [HuggingFace](https://huggingface.co/Systran/faster-distil-whisper-large-v3)
- [Distil-Whisper](https://github.com/huggingface/distil-whisper)
