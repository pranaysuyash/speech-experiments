# GLM-ASR-Nano-2512

LCS-16: Lightweight non-Whisper ASR from THUDM.

## Overview

- **Architecture**: GLM-4 decoder (non-Whisper)
- **Runtime**: PyTorch
- **Sample Rate**: 16kHz
- **License**: Apache 2.0

## Installation

```bash
make model-install MODEL=glm_asr_nano_2512
```

## Usage

```bash
# Quick test
make asr-audio MODEL=glm_asr_nano_2512 AUDIO=inputs/sample_16k.wav

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
}
```

## Notes

First non-Whisper batch ASR in Batch 2. Different decoder behavior
provides robustness testing across architectures.

## References

- [HuggingFace](https://huggingface.co/THUDM/glm-4-voice-decoder)
- [GLM-4](https://github.com/THUDM/GLM-4)
