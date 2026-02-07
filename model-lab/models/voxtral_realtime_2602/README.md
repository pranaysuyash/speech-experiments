# Voxtral Realtime 2602

LCS-22: Real-time streaming ASR with configurable transcription delay.

## Overview

- **Architecture**: Voxtral Mini 4B (open-weights)
- **Runtime**: PyTorch
- **Sample Rate**: 16kHz
- **Streaming**: Yes
- **License**: Apache 2.0

## Configuration

| Parameter | Default | Model Range | Recommended |
|-----------|---------|-------------|-------------|
| `transcription_delay_ms` | 200 | 80-2400 | 100-500 |
| `chunk_ms` | 100 | 50-500 | 100 |

## Installation

```bash
make model-install MODEL=voxtral_realtime_2602
export MISTRAL_API_KEY=your_key_here  # If using API
```

## Usage

```python
from harness.registry import ModelRegistry

bundle = ModelRegistry.load_model("voxtral_realtime_2602", {
    "transcription_delay_ms": 150  # Tune latency
}, device="cpu")

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

All streaming models enforce:
- **seq_monotonic**: Sequence numbers never decrease
- **segment_id_stable**: Segment IDs stable across partial updates
- **finalize_idempotent**: Multiple finalize() calls safe
- **push_after_finalize_raises**: RuntimeError if push_audio after finalize

## Comparison with Other Streaming Models

Use the streaming config matrix harness to compare:
- Voxtral Realtime (this) - configurable delay
- Kyutai Streaming - lightweight
- Nemotron Streaming - NeMo (heavy)

## References

- [Mistral AI](https://mistral.ai)
- [Voxtral](https://huggingface.co/mistralai)
