# Demucs

LCS-11: Hybrid transformer audio source separation from Meta.

## Overview

- **Architecture**: Hybrid Transformer Demucs (U-Net + Transformer)
- **Stems**: vocals, drums, bass, other (4-stem default)
- **Sample Rate**: 44.1kHz native
- **License**: MIT

## Installation

```bash
make model-install MODEL=demucs
```

**Note**: Requires PyTorch. Use isolated venv.

## Usage

```bash
make model-info MODEL=demucs
make separate MODEL=demucs AUDIO=inputs/song.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| separate | separate | ✓ |

## Output Format

```python
{
    "stems": {
        "vocals": np.array([...]),   # float32, same length as input
        "drums": np.array([...]),
        "bass": np.array([...]),
        "other": np.array([...]),
    },
    "sr": 44100,
}
```

**Length Alignment**: All stems have same length as input audio.

## References

- [GitHub](https://github.com/facebookresearch/demucs)
- [Paper](https://arxiv.org/abs/2111.03600)
