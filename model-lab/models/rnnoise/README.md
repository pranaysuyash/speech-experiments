# RNNoise

LCS-06: Real-time audio noise suppression from Mozilla/Xiph.org.

## Overview

- **Architecture**: GRU-based RNN
- **Latency**: <10ms per frame
- **Sample Rate**: 48kHz native (resampling handled automatically)
- **License**: BSD-3-Clause

## Installation

```bash
make model-install MODEL=rnnoise
```

## Usage

```bash
make model-info MODEL=rnnoise
make enhance MODEL=rnnoise AUDIO=path/to/noisy.wav OUTPUT=clean.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| enhance | process | ✓ |

## Output Format

```python
{
    "audio": np.array([...]),  # Enhanced audio
    "sample_rate": 48000,
    "vad_probs": [...],  # Optional VAD probabilities per frame
}
```

## References

- [GitHub](https://github.com/xiph/rnnoise)
- [Paper](https://arxiv.org/abs/1709.08243)
