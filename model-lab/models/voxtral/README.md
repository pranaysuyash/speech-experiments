# Voxtral

LCS-10: Streaming ASR from Mistral AI with low-latency real-time transcription.

## Overview

- **Provider**: Mistral AI
- **Variants**: Mini (3B), Small (24B), Realtime (streaming)
- **Sample Rate**: 16kHz
- **Streaming**: Yes (websocket-based)
- **License**: Apache 2.0

## Installation

```bash
make model-install MODEL=voxtral
export MISTRAL_API_KEY=your_key_here
```

## Usage

```bash
make model-info MODEL=voxtral
make asr-stream MODEL=voxtral AUDIO=inputs/sample_16k.wav
```

## Surfaces

| Surface | Method | Status |
|---------|--------|--------|
| asr_stream | start_stream, push_audio, flush, finalize, close | ✓ |
| asr | transcribe | ✓ (batch fallback) |

## Streaming Output

```python
# Partial events
{"type": "partial", "text": "Hello", "seq": 0, "segment_id": "seg_0"}
{"type": "partial", "text": "Hello world", "seq": 1, "segment_id": "seg_0"}

# Final event
{"type": "final", "text": "Hello world", "seq": 2, "segment_id": "seg_0", "is_endpoint": true}
```

## References

- [Mistral AI Docs](https://docs.mistral.ai)
- [Voxtral on HuggingFace](https://huggingface.co/mistralai)
