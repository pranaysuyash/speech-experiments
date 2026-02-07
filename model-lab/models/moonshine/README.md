# Moonshine Tiny

LCS-04: Fast CPU-first ASR model from Useful Sensors.

## Overview

- **Params**: 27M (tiny), 61M (base)
- **Speed**: 5-15x faster than Whisper on short segments
- **Target**: Real-time on-device transcription
- **License**: MIT

## Installation

```bash
make model-install MODEL=moonshine
```

## Usage

```bash
make model-info MODEL=moonshine
make asr MODEL=moonshine AUDIO=path/to/audio.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| asr | transcribe | ✓ |

## Claims

See `claims.yaml` for full claim manifest.

## References

- [GitHub](https://github.com/usefulsensors/moonshine)
- [HuggingFace](https://huggingface.co/UsefulSensors/moonshine-tiny)
