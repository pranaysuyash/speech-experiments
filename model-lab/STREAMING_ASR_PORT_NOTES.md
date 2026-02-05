# Streaming ASR (Ported from EchoPanel)

**Date**: 2026-02-05  
**Goal**: Keep EchoPanel’s proven real-time ASR chunking/provider architecture available inside Model-Lab so we can benchmark streaming behavior (latency, boundary artifacts, VAD tradeoffs) alongside batch evals.

## What was ported

- `model-lab/harness/streaming_asr/providers.py`: provider interface + registry
- `model-lab/harness/streaming_asr/stream.py`: `stream_asr(...)` pipeline that yields `asr_final/asr_partial`-style events
- `model-lab/harness/streaming_asr/provider_faster_whisper.py`: local `faster-whisper` provider (thread-safe inference lock, chunked PCM16 processing)

## Environment variables

These mirror EchoPanel but use a `MODEL_LAB_` prefix:

- `MODEL_LAB_ASR_PROVIDER` (default: `faster_whisper`)
- `MODEL_LAB_WHISPER_MODEL` (default: `base`)
- `MODEL_LAB_WHISPER_DEVICE` (default: `auto`; `mps` coerces to `cpu` for faster-whisper)
- `MODEL_LAB_WHISPER_COMPUTE` (default: `int8`; `float16` coerces to `int8` on CPU)
- `MODEL_LAB_WHISPER_LANGUAGE` (default: unset = auto-detect)
- `MODEL_LAB_ASR_CHUNK_SECONDS` (default: `4`)
- `MODEL_LAB_ASR_VAD` (default: `0`; set `1` to enable faster-whisper’s VAD filter)

## Minimal usage sketch (async)

Use this for research harnesses or quick scripts; it’s not wired into the FastAPI server yet.

```python
from harness.streaming_asr import stream_asr

async def pcm_iter():
    yield pcm_bytes_1
    yield pcm_bytes_2

async for event in stream_asr(pcm_iter(), sample_rate=16000, source="mic"):
    print(event)
```

## Notes / caveats

- The faster-whisper provider **serializes** `model.transcribe()` calls; concurrent transcription on a single provider instance is unsafe.
- faster-whisper/CTranslate2 does **not** support MPS; the provider coerces `auto/mps → cpu` on macOS.

