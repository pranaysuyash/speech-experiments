# NB-Whisper-Small-ONNX

LCS-17: Norwegian Whisper with ONNX runtime.

## Overview

- **Architecture**: Whisper small + ONNX
- **Runtime**: ONNXRuntime
- **Language**: Norwegian (optimized)
- **Sample Rate**: 16kHz
- **License**: MIT

## Installation

```bash
make model-install MODEL=nb_whisper_small_onnx
```

## Usage

```bash
# Quick test
make asr-audio MODEL=nb_whisper_small_onnx AUDIO=inputs/sample_16k.wav

# Pipeline test
make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| asr | transcribe | ✓ |

## Notes

ONNX runtime provides cross-platform inference without PyTorch.
Good for edge deployment on Mac where other runtimes may have packaging issues.

## References

- [HuggingFace](https://huggingface.co/NbAiLab/nb-whisper-small-ONNX)
- [NbAiLab](https://github.com/NbAiLab)
