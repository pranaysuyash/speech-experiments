# GLM-TTS

LCS-21: Text-to-speech synthesis from THUDM GLM-4 Voice.

## Overview

- **Architecture**: GLM-4 Voice decoder
- **Runtime**: PyTorch
- **Sample Rate**: 24kHz output
- **License**: Apache 2.0

## Installation

```bash
make model-install MODEL=glm_tts
```

## Usage


```bash
# Requires HuggingFace token for gated model THUDM/glm-4-voice
export HF_TOKEN=your_token_here
make tts MODEL=glm_tts TEXT="Hello world"
```

OR

```bash
huggingface-cli login
make tts MODEL=glm_tts TEXT="Hello world"
```

```python
from harness.registry import ModelRegistry

bundle = ModelRegistry.load_model("glm_tts", {}, device="cpu")
audio, sr = bundle["tts"]["synthesize"]("Hello world")
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| tts | synthesize | ✓ |

## Output Format

```python
# Returns tuple: (audio_array, sample_rate)
audio, sr = synthesize("Hello world")
# audio: np.ndarray (float32)
# sr: 24000
```

## References

- [HuggingFace](https://huggingface.co/THUDM/glm-4-voice)
- [GLM-4](https://github.com/THUDM/GLM-4)
