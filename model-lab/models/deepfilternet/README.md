# DeepFilterNet

LCS-07: Production-grade speech enhancement using ERB-scale deep filtering.

## Overview

- **Architecture**: Deep filtering on ERB-scale spectral features
- **Variants**: DF2 (balanced), DF3 (highest quality)
- **Sample Rate**: 48kHz native (auto-resampling in wrapper)
- **License**: MIT

## Installation

```bash
make model-install MODEL=deepfilternet
```

**Note**: Requires PyTorch. Use isolated venv.

## Usage

```bash
make model-info MODEL=deepfilternet
make enhance MODEL=deepfilternet AUDIO=noisy.wav OUTPUT=clean.wav
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
    "meta": {"variant": "df2", "device": "cpu"}
}
```

## References

- [GitHub](https://github.com/Rikorose/DeepFilterNet)
- [Paper](https://arxiv.org/abs/2110.05588)
