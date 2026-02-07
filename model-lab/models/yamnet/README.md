# YAMNet

LCS-05: Audio event classification from TensorFlow Hub.

## Overview

- **Classes**: 521 AudioSet event classes
- **Architecture**: MobileNetV1 depthwise-separable convolutions
- **Input**: 16kHz mono audio, 0.96s frames
- **License**: Apache 2.0

## Installation

```bash
make model-install MODEL=yamnet
```

**Note**: TensorFlow is heavyweight. Use isolated venv.

## Usage

```bash
make model-info MODEL=yamnet
make classify MODEL=yamnet AUDIO=path/to/audio.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| classify | predict | ✓ |

## Output Format

```python
{
    "labels": ["Speech", "Music", "Silence", ...],
    "scores": [0.85, 0.10, 0.03, ...],
    "top_k": 5,
    "embeddings": np.array(...)  # optional
}
```

## References

- [TensorFlow Hub](https://tfhub.dev/google/yamnet/1)
- [Paper](https://arxiv.org/abs/1609.04243)
