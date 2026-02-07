# Parakeet Multitalker ASR

LCS-20: NVIDIA Parakeet multitalker ASR with speaker diarization support.

## Overview

- **Architecture**: Parakeet RNN-T (1.1B params)
- **Runtime**: NeMo (heavy)
- **Sample Rate**: 16kHz
- **Max Speakers**: 4
- **License**: Apache 2.0

> [!IMPORTANT]
> This model requires NVIDIA NeMo toolkit which has heavy dependencies.
> Use a dedicated virtual environment to avoid conflicts.

> [!NOTE]
> If the model needs pre-diarization, pass `speaker_segments` arg.
> Check README for explicit input requirements.

## Installation

```bash
# Create dedicated venv
python -m venv .venv.nemo_parakeet
source .venv.nemo_parakeet/bin/activate

# Install dependencies
pip install -r models/parakeet_multitalker/requirements.txt
```

## Usage

```python
from harness.registry import ModelRegistry

bundle = ModelRegistry.load_model("parakeet_multitalker", {}, device="cuda")
result = bundle["asr"]["transcribe"](audio, sr=16000)

# With speaker segments (if pre-diarization required)
result = bundle["asr"]["transcribe"](audio, sr=16000, speaker_segments=[...])
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| asr | transcribe | ✓ |

## Output Format

```python
{
    "text": "Speaker 1: Hello. Speaker 2: Hi there.",
    "speakers": [
        {"id": 0, "start": 0.0, "end": 1.5, "text": "Hello"},
        {"id": 1, "start": 1.5, "end": 3.0, "text": "Hi there"},
    ]
}
```

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Parakeet](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
