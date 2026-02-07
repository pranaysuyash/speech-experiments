# Nemotron Streaming ASR

LCS-18: NVIDIA Nemotron streaming ASR with NeMo runtime.

## Overview

- **Architecture**: Nemotron streaming encoder
- **Runtime**: NeMo (heavy)
- **Sample Rate**: 16kHz
- **Streaming**: Yes
- **License**: Apache 2.0

> [!IMPORTANT]
> This model requires NVIDIA NeMo toolkit which has heavy dependencies.
> Use a dedicated virtual environment to avoid conflicts.

## Installation

```bash
# Create dedicated venv
python -m venv .venv.nemo_nemotron
source .venv.nemo_nemotron/bin/activate

# Install dependencies
pip install -r models/nemotron_streaming/requirements.txt
```

## Usage

```python
from harness.registry import ModelRegistry

bundle = ModelRegistry.load_model("nemotron_streaming", {}, device="cuda")
stream = bundle["asr_stream"]

handle = stream["start"](sr=16000)
stream["push_audio"](handle, audio_chunk)
result = stream["get_transcript"](handle)
final = stream["finalize"](handle)
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| asr_stream | start, push_audio, get_transcript, finalize | ✓ |

## Streaming Contract

- **seq_monotonic**: Sequence numbers never decrease
- **segment_id_stable**: Segment IDs stable across partial updates
- **finalize_idempotent**: Multiple finalize() calls safe
- **push_after_finalize_raises**: RuntimeError

## References

- [NVIDIA NeMo](https://github.com/NVIDIA/NeMo)
- [Nemotron](https://huggingface.co/nvidia)
