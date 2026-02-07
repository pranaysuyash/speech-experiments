# CLAP

LCS-08: Contrastive Language-Audio Pretraining for embeddings and zero-shot classification.

## Overview

- **Architecture**: Contrastive audio-text encoder
- **Embedding Dim**: 512
- **Sample Rate**: 48kHz native
- **License**: Apache 2.0

## Installation

```bash
make model-install MODEL=clap
```

**Note**: Requires PyTorch. Use isolated venv.

## Usage

```bash
make model-info MODEL=clap
make embed MODEL=clap AUDIO=audio.wav
make classify MODEL=clap AUDIO=audio.wav LABELS="speech,music,silence"
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| embed | encode | ✓ |
| classify | predict | ✓ |

## Output Format

**Embed**:
```python
{"embedding": np.array([...]), "dim": 512}
```

**Classify**:
```python
{"labels": ["music", "speech", ...], "scores": [0.8, 0.15, ...]}
```

## References

- [GitHub](https://github.com/LAION-AI/CLAP)
- [Paper](https://arxiv.org/abs/2211.06687)
