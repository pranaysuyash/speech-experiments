# Kyutai Streaming ASR

LCS-19: Lightweight PyTorch streaming ASR. First streaming model in Batch 3.

## Overview

- **Architecture**: Kyutai streaming encoder
- **Runtime**: PyTorch
- **Sample Rate**: 16kHz
- **Streaming**: Yes (chunk_ms configurable)
- **License**: MIT

## Installation

```bash
make model-install MODEL=kyutai_streaming
```

## Usage

```python
from harness.registry import ModelRegistry

bundle = ModelRegistry.load_model("kyutai_streaming", {}, device="cpu")
stream = bundle["asr_stream"]

# Streaming lifecycle
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

All streaming models must enforce:
- **seq_monotonic**: Sequence numbers never decrease
- **segment_id_stable**: Segment IDs stable across partial updates
- **finalize_idempotent**: Multiple finalize() calls safe
- **push_after_finalize_raises**: RuntimeError if push_audio after finalize

## References

- [Kyutai](https://huggingface.co/kyutai)
