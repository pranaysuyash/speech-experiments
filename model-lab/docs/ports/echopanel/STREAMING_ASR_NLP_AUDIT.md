# EchoPanel Streaming ASR & NLP Pipeline Audit

**Provenance**: Ported from `EchoPanel/docs/audit/STREAMING_ASR_NLP_AUDIT.md` on 2026-02-05 for Model-Lab reference and experiment planning.

**Date:** 2026-02-04  
**Scope:** Client capture → WebSocket transport → ASR → NLP → diarization  
**Goal:** Identify root causes of unreliable streaming ASR/NLP and provide patch-level fixes

---

## Executive Summary

The streaming pipeline has **several P0/P1 correctness issues** that can cause:
- Missing or duplicated transcripts
- Memory growth under backpressure
- Race conditions on WebSocket sends
- Incomplete final summaries due to premature NLP execution

Most issues are in concurrency control, queue management, and stop-flush sequencing.

---

## 1. Architecture Map

### 1.1 Module Inventory

| Layer | Component | File | Purpose |
|-------|-----------|------|---------|
| **Client** | System Audio | `AudioCaptureManager.swift` | ScreenCaptureKit → 16kHz mono PCM16 |
| **Client** | Mic Audio | `MicrophoneCaptureManager.swift` | AVAudioEngine → 16kHz mono PCM16 |
| **Client** | Transport | `WebSocketStreamer.swift` | JSON framing, base64 audio, reconnect |
| **Client** | Lifecycle | `BackendManager.swift` | Server subprocess, health checks |
| **Server** | Entry | `main.py` | FastAPI app, lifespan, /health |
| **Server** | WebSocket | `ws_live_listener.py` | Session state, queues, task management |
| **Server** | ASR Pipeline | `asr_stream.py` | Provider abstraction, config |
| **Server** | ASR Provider | `provider_faster_whisper.py` | faster-whisper inference |
| **Server** | Provider Registry | `asr_providers.py` | Singleton caching, config keying |
| **Server** | NLP | `analysis_stream.py` | Cards, entities, rolling summary |
| **Server** | Diarization | `diarization.py` | pyannote batch diarization |

### 1.2 Dataflow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              macOS Client                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ScreenCaptureKit (48kHz stereo float32)                                  │
│         │                                                                   │
│         ▼                                                                   │
│   AVAudioConverter → 16kHz mono float32                                    │
│         │                                                                   │
│         ▼                                                                   │
│   emitPCMFrames() → 320-sample frames (640 bytes PCM16)                    │
│         │                                        ┌───────────────────────┐  │
│         └────────────────────────────────────────►  WebSocketStreamer   │  │
│                                                  │  sendPCMFrame(data,  │  │
│   AVAudioEngine (mic) → same conversion          │    source="system")  │  │
│         │                                        └───────────┬───────────┘  │
│         └────────────────────────────────────────────────────┘              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ WebSocket (JSON + base64 audio)
                                   ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FastAPI Server                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ws_live_listener.py                                                       │
│         │                                                                   │
│         ├── SessionState                                                    │
│         │      ├── queues: Dict[source, asyncio.Queue]  (maxsize=48)       │
│         │      ├── transcript: List[dict]                                   │
│         │      ├── asr_tasks: List[Task]                                   │
│         │      └── send_lock: asyncio.Lock                                 │
│         │                                                                   │
│         ├── put_audio() → enqueue with drop-oldest on full                 │
│         │                                                                   │
│         └── _asr_loop() per source                                         │
│                  │                                                          │
│                  ▼                                                          │
│   asr_stream.py::stream_asr()                                              │
│         │                                                                   │
│         ▼                                                                   │
│   provider_faster_whisper.py::transcribe_stream()                          │
│         │                                                                   │
│         ├── buffer accumulation (4s chunks by default)                     │
│         ├── asyncio.to_thread(_transcribe) with inference lock             │
│         └── yield ASRSegment (is_final=True only)                          │
│                                                                             │
│   Analysis (periodic):                                                      │
│         ├── _analysis_loop() every 12s → entities                          │
│         └── _analysis_loop() every 40s → cards                             │
│                                                                             │
│   On stop:                                                                  │
│         ├── await ASR flush (8s timeout)                                   │
│         ├── cancel analysis tasks                                          │
│         ├── generate_rolling_summary() via to_thread                       │
│         ├── extract_cards() via to_thread                                  │
│         ├── extract_entities() via to_thread                               │
│         └── send final_summary                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Threading/Async Boundaries

| Boundary | Mechanism | Notes |
|----------|-----------|-------|
| Client audio callback → main | DispatchQueue (Swift) | Audio callbacks on global queue |
| WebSocket send | URLSession (Swift) | Async, not serialized by client |
| Server WebSocket receive | asyncio event loop | Single-threaded |
| ASR inference | `asyncio.to_thread` | Offloaded to thread pool |
| NLP extraction | `asyncio.to_thread` | Offloaded to thread pool |
| faster-whisper model | `threading.Lock` | Serializes inference |

---

## 2. Protocol Audit

### 2.1 Current Message Schema

**Client → Server:**

```json
// start
{"type": "start", "session_id": "uuid", "sample_rate": 16000, "format": "pcm_s16le", "channels": 1}

// audio (v0.2)
{"type": "audio", "source": "system"|"mic", "data": "<base64 PCM16>"}

// stop
{"type": "stop", "session_id": "uuid"}

// Legacy: raw binary PCM16 (treated as source="system")
```

**Server → Client:**

```json
// ASR events
{"type": "asr_partial"|"asr_final", "t0": float, "t1": float, "text": "...", "confidence": float, "source": "system"|"mic", "stable": bool}

// Analysis
{"type": "entities_update", "people": [...], "orgs": [...], ...}
{"type": "cards_update", "actions": [...], "decisions": [...], "risks": [...], "window": {...}}

// Status
{"type": "status", "state": "streaming"|"reconnecting"|"error", "message": "..."}

// Final
{"type": "final_summary", "markdown": "...", "json": {...}}
```

### 2.2 Protocol Issues

| Issue | Severity | Location | Problem |
|-------|----------|----------|---------|
| No protocol version | P2 | start message | No way to negotiate features |
| No format validation on binary | P1 | `ws_live_listener.py:280-298` | Binary assumed to be PCM16 16kHz |
| `sample_rate` only validated at start | P1 | `ws_live_listener.py:167` | Per-frame rate changes ignored |
| Missing `encoding` field in audio | P2 | `WS_CONTRACT.md:69` | Spec says `data`, impl uses `data` |
| No backpressure signal | P1 | N/A | Client has no drop notification |

### 2.3 Audio Format Validation

**Current validation (start only):**
```python
# ws_live_listener.py:167-173
if sample_rate != 16000 or encoding != "pcm_s16le" or channels != 1:
    await ws_send(state, websocket, {"type": "error", ...})
    await websocket.close()
```

**Observation:** macOS ScreenCaptureKit defaults to 48kHz stereo float32. The client performs resampling via AVAudioConverter. **This is correct**, but:

1. The client doesn't explicitly set `SCStreamConfiguration.sampleRate` to 16000
2. Resampling quality depends on AVAudioConverter (Apple Accelerate framework - good quality)

**Recommendation:** Add explicit sample rate configuration in ScreenCaptureKit:

```swift
// AudioCaptureManager.swift - add to startCapture()
configuration.sampleRate = 16000  // Request 16kHz directly (supported per Apple docs)
```

### 2.4 Recommended Protocol v0.3

```json
// start (add version)
{"type": "start", "version": "0.3", "session_id": "uuid", "sample_rate": 16000, "encoding": "pcm_s16le", "channels": 1}

// audio (add sequence for ordering)
{"type": "audio", "source": "system"|"mic", "seq": int, "data": "<base64>"}

// backpressure notification (new)
{"type": "backpressure", "source": "system"|"mic", "dropped_frames": int, "queue_depth": int}
```

---

## 3. Streaming Correctness Audit

### 3.1 Backpressure Analysis

**Current Implementation:**

```python
# ws_live_listener.py:64-78
async def put_audio(q: asyncio.Queue, chunk: bytes, state: Optional["SessionState"] = None, source: str = "") -> None:
    try:
        q.put_nowait(chunk)
    except asyncio.QueueFull:
        _ = q.get_nowait()  # Drop oldest
        q.put_nowait(chunk)
        if state is not None:
            state.dropped_frames = getattr(state, 'dropped_frames', 0) + 1
```

**Issues:**

| Issue | ID | Severity | Problem |
|-------|-----|----------|---------|
| Silent drops | BP-1 | P1 | Client never learns about dropped frames |
| Queue size arbitrary | BP-2 | P2 | QUEUE_MAX=48 ≈ 48*640B = 30KB ≈ ~1s of audio at 16kHz mono |
| No drop logging in production | BP-3 | P2 | Only logged when DEBUG=1 |
| PCM buffer unbounded | BP-4 | P0 | `state.pcm_buffer` grows unbounded if diarization disabled but audio arrives |

**P0 Fix for BP-4:**

The `pcm_buffer` truncation logic is inside the `if state.diarization_enabled` block, but buffer still extends in the diarization block. Currently safe because diarization is commented out, but the architecture is fragile.

**Recommended Fix:**
```python
# ws_live_listener.py - move buffer management outside conditional
if state.diarization_enabled:
    state.pcm_buffer.extend(chunk)
    # Truncate ALWAYS when buffer exceeds limit
    if len(state.pcm_buffer) > state.diarization_max_bytes:
        del state.pcm_buffer[:len(state.pcm_buffer) - state.diarization_max_bytes]
```

### 3.2 Concurrency: WebSocket Send Serialization

**Current Implementation:**

```python
# ws_live_listener.py:45-54
async def ws_send(state: SessionState, websocket: WebSocket, event: dict) -> None:
    if state.closed:
        return
    async with state.send_lock:
        try:
            await websocket.send_text(json.dumps(event))
        except RuntimeError:
            state.closed = True
```

**Analysis:** ✅ **Correctly serialized** via `send_lock`. Multiple ASR tasks and the analysis task all use `ws_send()`.

**Web Research Confirmation:**
- Starlette/FastAPI WebSocket `send_text` is **not thread-safe** for concurrent calls
- The `send_lock` pattern is correct
- The gist at GitHub shows the same pattern for thread-safe WebSocket streaming

**Potential Issue:** The lock is an `asyncio.Lock`, which only prevents concurrent async sends. If a sync thread tried to send directly (not via `ws_send`), it would bypass the lock. **Current code is safe** - all sends go through `ws_send`.

### 3.3 Stop Semantics and Flush Ordering

**Current Implementation:**

```python
# ws_live_listener.py:207-271 (stop handler)
elif msg_type == "stop":
    # 1. Signal EOF to all queues
    for q in state.queues.values():
        await q.put(None)
    
    # 2. Wait for ASR to flush
    try:
        await asyncio.wait_for(
            asyncio.gather(*state.asr_tasks, return_exceptions=True),
            timeout=float(os.getenv("ECHOPANEL_ASR_FLUSH_TIMEOUT", "8")),
        )
    except asyncio.TimeoutError:
        pass
    
    # 3. Cancel analysis tasks
    for t in state.analysis_tasks:
        t.cancel()
    
    # 4. Generate final summary/cards/entities
    transcript_snapshot = list(state.transcript)
    summary_md = await asyncio.to_thread(generate_rolling_summary, transcript_snapshot)
    # ...
```

**Analysis:** ✅ **Correct flush ordering** - ASR completes before NLP runs on transcript.

**Issue SF-1 (P1):** If ASR flush times out (8s default), the transcript may be incomplete when NLP runs. No error is surfaced to the client.

**Fix:**
```python
except asyncio.TimeoutError:
    logger.warning("ASR flush timed out, transcript may be incomplete")
    await ws_send(state, websocket, {
        "type": "status",
        "state": "warning",
        "message": "ASR flush timed out, some speech may be missing"
    })
```

### 3.4 Chunking Correctness

**Current Implementation:**

```python
# provider_faster_whisper.py:118-121
while len(buffer) >= chunk_bytes:
    audio_bytes = bytes(buffer[:chunk_bytes])
    del buffer[:chunk_bytes]
    # ... process
```

**Analysis:** ✅ **Correct bounded chunking** - processes exactly `chunk_bytes` at a time, leaves remainder in buffer.

**Previous Bug (now fixed):** The comment at line 7-8 notes "Fixed chunk loop to process exactly chunk_bytes at a time (prevents runaway growth)". This was a past P0 fix.

### 3.5 Transcript Ordering

**Current State:**

- Each source (system/mic) has its own queue and ASR task
- Transcripts are appended to `state.transcript` as they complete
- No explicit ordering/sorting

**Issue TO-1 (P1):** Multi-source transcripts may interleave non-deterministically based on ASR processing time, not actual audio timing.

**Evidence:** Two audio sources with overlapping speech will have segments appended in completion order, not timestamp order.

**Fix Option A (simple):** Sort transcript by `t0` before NLP:
```python
# ws_live_listener.py - before running final NLP
transcript_snapshot = sorted(state.transcript, key=lambda s: s.get("t0", 0.0))
```

**Fix Option B (streaming):** Merge streams using a priority queue based on timestamps. More complex, better for real-time display.

### 3.6 Cancellation and Resource Cleanup

**Current Implementation:**

```python
# ws_live_listener.py:311-317 (finally block)
finally:
    for q in state.queues.values():
        await q.put(None)
    all_tasks = state.tasks + state.asr_tasks + state.analysis_tasks
    for task in all_tasks:
        task.cancel()
    await asyncio.gather(*all_tasks, return_exceptions=True)
```

**Analysis:** ✅ **Mostly correct** - EOF signals sent, tasks cancelled, gathered with exceptions.

**Issue CL-1 (P2):** `state.tasks` is never populated (empty list). Harmless but confusing.

**Issue CL-2 (P1):** If WebSocket disconnects during ASR inference (which is in `to_thread`), the thread continues running until completion. The CancelledError is only raised when the async wrapper resumes.

**Mitigation:** faster-whisper inference is typically fast (< chunk_seconds). Low risk in practice.

---

## 4. ASR Provider Audit

### 4.1 Registry Caching

**Current Implementation:**

```python
# asr_providers.py:107-125
@classmethod
def _cfg_key(cls, name: str, cfg: ASRConfig) -> str:
    return f"{name}|{cfg.model_name}|{cfg.device}|{cfg.compute_type}|{cfg.language}|{int(cfg.vad_enabled)}|{cfg.chunk_seconds}"

@classmethod
def get_provider(cls, name: Optional[str] = None, config: Optional[ASRConfig] = None) -> Optional[ASRProvider]:
    # ...
    key = cls._cfg_key(name, cfg)
    if key not in cls._instances:
        cls._instances[key] = cls._providers[name](cfg)
    return cls._instances[key]
```

**Issue RC-1 (P1):** No thread safety for `_instances` dict access. Multiple concurrent `get_provider` calls with a new config could create duplicate instances.

**Fix:**
```python
import threading

class ASRProviderRegistry:
    _providers: dict[str, type[ASRProvider]] = {}
    _instances: dict[str, ASRProvider] = {}
    _lock = threading.Lock()  # Add lock
    
    @classmethod
    def get_provider(cls, name: Optional[str] = None, config: Optional[ASRConfig] = None) -> Optional[ASRProvider]:
        # ... name/cfg setup ...
        key = cls._cfg_key(name, cfg)
        
        with cls._lock:  # Protect instance creation
            if key not in cls._instances:
                cls._instances[key] = cls._providers[name](cfg)
            return cls._instances[key]
```

### 4.2 faster-whisper Thread Safety

**Current Implementation:**

```python
# provider_faster_whisper.py:41, 133-140
def __init__(self, config: ASRConfig):
    # ...
    self._infer_lock = threading.Lock()

def _transcribe():
    with self._infer_lock:
        segments, info = model.transcribe(audio, ...)
    return list(segments), info
```

**Web Research Confirmation:**
- CTranslate2/faster-whisper `model.transcribe()` is **NOT thread-safe**
- Multiple concurrent calls can cause crashes or corrupted output
- The `_infer_lock` pattern is the correct solution

**Analysis:** ✅ **Correctly protected** - inference lock serializes all transcribe calls.

**Issue FW-1 (P2):** The lock is per-provider-instance, but the registry caches by config. If two different config keys somehow point to the same underlying model (unlikely), they'd have separate locks.

### 4.3 VAD Settings

**Current Implementation:**

```python
# asr_stream.py:29
vad_enabled=os.getenv("ECHOPANEL_ASR_VAD", "0") == "1",
```

**Default:** VAD disabled.

**Issue VAD-1 (P2):** With VAD enabled, faster-whisper's Silero VAD removes silence > 2s by default. This is appropriate for pre-recorded audio but may cause issues with real-time streams where pauses are meaningful.

**Recommendation:** Keep VAD disabled for real-time streaming (current default is correct). For batch/offline processing, VAD can improve accuracy.

### 4.4 Timestamp Math

**Current Implementation:**

```python
# provider_faster_whisper.py:124-127
t0 = processed_samples / sample_rate
chunk_samples = len(audio_bytes) // bytes_per_sample
t1 = (processed_samples + chunk_samples) / sample_rate
processed_samples += chunk_samples
```

**Analysis:** ✅ **Correct sample-based timestamps** - uses processed sample count, not wall clock.

**Issue TS-1 (P2):** Timestamps are relative to stream start, not session start. If there's latency before first audio arrives, timestamps won't match wall clock.

**Recommendation:** Accept this limitation. Sample-based timestamps are more reliable than wall clock for audio processing.

### 4.5 Repeated/Fragmented Text

**Symptoms:** Users may see repeated phrases or sentence fragments.

**Root Causes:**

1. **Chunk boundary issues:** Whisper transcribes each chunk independently. Words split across chunks may be repeated or cut off.

2. **No prompt conditioning:** faster-whisper supports `initial_prompt` to provide context. Not currently used.

**Fix (P2):**
```python
# provider_faster_whisper.py - add prompt conditioning
last_text = ""

for chunk in chunks:
    # ...
    segments, info = model.transcribe(
        audio,
        initial_prompt=last_text[-200:] if last_text else None,  # Last ~200 chars as context
        # ...
    )
    for segment in segments:
        last_text += " " + segment.text
```

**Caveat:** Prompt conditioning adds latency. Benchmark impact before enabling.

---

## 5. NLP and Diarization Audit

### 5.1 NLP Off Event Loop

**Current Implementation:**

```python
# ws_live_listener.py:109, 114, 244-253
entities = await asyncio.to_thread(extract_entities, snapshot)
cards = await asyncio.to_thread(extract_cards, snapshot)
summary_md = await asyncio.to_thread(generate_rolling_summary, transcript_snapshot)
```

**Analysis:** ✅ **Correct** - all NLP work is offloaded via `asyncio.to_thread`.

### 5.2 Diarization Input Coherence

**Current Implementation:**

```python
# ws_live_listener.py:200-205 (disabled)
if state.diarization_enabled:
    state.pcm_buffer.extend(chunk)
```

**Issue DI-1 (P0):** The `pcm_buffer` mixes audio from both system and mic sources into a single buffer. Diarization expects a coherent single-stream waveform.

**Why This Breaks:**
- System audio (Zoom/Meet) already has multiple speakers
- Mic audio has local user
- Interleaving creates an incoherent waveform where diarization can't distinguish speakers

**Current Status:** Diarization is commented out (lines 232-238), correctly avoiding this issue.

**Proper Fix (if diarization is re-enabled):**

Option A: Separate buffers per source, diarize independently:
```python
state.pcm_buffers: Dict[str, bytearray] = {}  # per-source buffers
```

Option B: Only diarize system audio (typical meeting capture):
```python
if state.diarization_enabled and source == "system":
    state.pcm_buffer.extend(chunk)
```

Option C: Mix sources with known timing for coherent stream (complex).

### 5.3 merge_transcript_with_speakers

**Current Implementation:**

```python
# diarization.py:206-222
def merge_transcript_with_speakers(transcript: List[dict], speaker_segments: List[dict]) -> List[dict]:
    for seg in transcript:
        t0 = seg.get("t0", 0.0)
        t1 = seg.get("t1", 0.0)
        mid = (t0 + t1) / 2.0
        
        for spk_seg in speaker_segments:
            if spk_seg["t0"] <= mid <= spk_seg["t1"]:
                speaker = spk_seg["speaker"]
                break
```

**Issue MT-1 (P1):** Uses midpoint matching. If a transcript segment spans multiple speakers, only the speaker at the midpoint is assigned.

**Issue MT-2 (P2):** Linear search through speaker_segments for each transcript segment. O(n*m) complexity.

**Improved Implementation:**
```python
def merge_transcript_with_speakers(transcript: List[dict], speaker_segments: List[dict]) -> List[dict]:
    if not speaker_segments:
        return transcript
    
    # Pre-sort both by t0
    sorted_segments = sorted(speaker_segments, key=lambda s: s["t0"])
    
    result = []
    for seg in sorted(transcript, key=lambda s: s.get("t0", 0.0)):
        mid = (seg.get("t0", 0.0) + seg.get("t1", 0.0)) / 2.0
        
        # Binary search for speaker segment containing midpoint
        speaker = None
        for spk_seg in sorted_segments:
            if spk_seg["t0"] <= mid <= spk_seg["t1"]:
                speaker = spk_seg["speaker"]
                break
            if spk_seg["t0"] > mid:
                break
        
        merged = dict(seg)
        if speaker:
            merged["speaker"] = speaker
        result.append(merged)
    
    return result
```

---

## 6. Prioritized Issue List

### P0 - Critical (Fix Immediately)

| ID | Issue | Symptom | Root Cause | Fix Location |
|----|-------|---------|------------|--------------|
| BP-4 | PCM buffer grows unbounded | Memory leak over long sessions | Buffer extends outside truncation logic | `ws_live_listener.py:200-205` |
| RC-1 | Registry race condition | Duplicate provider instances, potential crashes | No locking on `_instances` dict | `asr_providers.py:111-125` |
| DI-1 | Mixed-source diarization buffer | Incoherent diarization | Both sources write to single buffer | `ws_live_listener.py:200-205` |

### P1 - High (Fix Before Production)

| ID | Issue | Symptom | Root Cause | Fix Location |
|----|-------|---------|------------|--------------|
| BP-1 | Silent frame drops | Gaps in transcript | No backpressure notification to client | `ws_live_listener.py:64-78` |
| SF-1 | Silent ASR flush timeout | Incomplete final transcript | Timeout not surfaced to user | `ws_live_listener.py:219-225` |
| TO-1 | Non-deterministic transcript order | Out-of-order segments in final summary | Append order vs timestamp order | `ws_live_listener.py:241` |
| MT-1 | Poor speaker assignment | Wrong speaker labels | Midpoint matching too simplistic | `diarization.py:206-222` |

### P2 - Medium (Improve Quality)

| ID | Issue | Symptom | Root Cause | Fix Location |
|----|-------|---------|------------|--------------|
| BP-2 | Arbitrary queue size | May be too small/large | QUEUE_MAX=48 not tuned | `ws_live_listener.py:19` |
| BP-3 | Missing drop logging | No visibility into drops | Only logs when DEBUG=1 | `ws_live_listener.py:76-77` |
| CL-1 | Unused state.tasks | Code confusion | Never populated | `ws_live_listener.py:26` |
| VAD-1 | VAD may drop speech | Missing words | Silence detection too aggressive | `asr_stream.py:29` |
| TS-1 | Timestamps drift from wall clock | Timing mismatch | Sample-based vs real-time | N/A (accept) |
| FW-2 | Repeated text | Hallucinations/duplicates | No prompt conditioning | `provider_faster_whisper.py` |
| MT-2 | O(n*m) speaker merge | Slow finalization | Linear search | `diarization.py:206-222` |

---

## 7. Observability Recommendations

### 7.1 Structured Logging

Add counters/gauges to `ws_live_listener.py`:

```python
import logging
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class SessionMetrics:
    frames_received: int = 0
    frames_dropped: int = 0
    bytes_received: int = 0
    asr_segments_emitted: int = 0
    asr_latency_sum: float = 0.0
    asr_latency_count: int = 0
    queue_high_water: Dict[str, int] = field(default_factory=dict)

# In put_audio():
state.metrics.frames_received += 1
state.metrics.bytes_received += len(chunk)
if state.queues[source].qsize() > state.metrics.queue_high_water.get(source, 0):
    state.metrics.queue_high_water[source] = state.queues[source].qsize()

# On stop, log summary:
logger.info(
    "Session complete",
    extra={
        "session_id": state.session_id,
        "frames_received": state.metrics.frames_received,
        "frames_dropped": state.metrics.frames_dropped,
        "asr_segments": state.metrics.asr_segments_emitted,
        "avg_asr_latency_ms": (state.metrics.asr_latency_sum / max(1, state.metrics.asr_latency_count)) * 1000,
        "queue_high_water": state.metrics.queue_high_water,
    }
)
```

### 7.2 Status Events for UI

Add new status events:

```python
# Model loading
await ws_send(state, websocket, {"type": "status", "state": "loading_model", "message": "Loading ASR model..."})

# Backpressure
await ws_send(state, websocket, {"type": "status", "state": "backpressure", "dropped_frames": state.dropped_frames})

# Format mismatch
await ws_send(state, websocket, {"type": "status", "state": "format_error", "message": "..."})
```

### 7.3 Soak Test Harness

```python
# scripts/soak_test.py
"""
Simulate 30-minute real-time audio stream, assert bounded latency.

Usage: python scripts/soak_test.py --duration 1800 --check-interval 60
"""

import asyncio
import json
import struct
import time
import websockets

async def soak_test(duration_seconds: int = 1800, check_interval: int = 60):
    uri = "ws://127.0.0.1:8000/ws/live-listener"
    latencies = []
    
    async with websockets.connect(uri) as ws:
        await ws.send(json.dumps({
            "type": "start", "session_id": "soak", "sample_rate": 16000, "format": "pcm_s16le", "channels": 1
        }))
        
        start = time.time()
        frame_count = 0
        
        async def send_audio():
            nonlocal frame_count
            while time.time() - start < duration_seconds:
                # Send 20ms of silence (320 samples = 640 bytes)
                frame = struct.pack('<320h', *([0] * 320))
                await ws.send(json.dumps({"type": "audio", "source": "system", "data": base64.b64encode(frame).decode()}))
                frame_count += 1
                await asyncio.sleep(0.02)  # Real-time 20ms
        
        async def receive_events():
            while time.time() - start < duration_seconds:
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    event = json.loads(msg)
                    if event.get("type") == "asr_final":
                        latencies.append(time.time() - start - event.get("t1", 0))
                except asyncio.TimeoutError:
                    pass
        
        async def check_stats():
            while time.time() - start < duration_seconds:
                await asyncio.sleep(check_interval)
                avg_latency = sum(latencies[-100:]) / max(1, len(latencies[-100:]))
                print(f"[{int(time.time() - start)}s] Frames: {frame_count}, Avg latency (last 100): {avg_latency:.2f}s")
                assert avg_latency < 5.0, f"Latency too high: {avg_latency}s"
        
        await asyncio.gather(send_audio(), receive_events(), check_stats())
        
        await ws.send(json.dumps({"type": "stop", "session_id": "soak"}))
        
    print(f"Soak test complete. {len(latencies)} ASR events, max latency: {max(latencies):.2f}s")

if __name__ == "__main__":
    asyncio.run(soak_test())
```

---

## 8. Tests and Safeguards

### 8.1 Unit Tests Needed

```python
# tests/test_streaming_correctness.py

import pytest
from server.api.ws_live_listener import put_audio, SessionState
import asyncio

@pytest.mark.asyncio
async def test_put_audio_drops_oldest_on_full():
    """Verify queue drops oldest frame when full."""
    state = SessionState()
    q = asyncio.Queue(maxsize=2)
    
    await put_audio(q, b"frame1", state, "system")
    await put_audio(q, b"frame2", state, "system")
    await put_audio(q, b"frame3", state, "system")  # Should drop frame1
    
    assert q.qsize() == 2
    assert state.dropped_frames == 1
    assert await q.get() == b"frame2"
    assert await q.get() == b"frame3"

@pytest.mark.asyncio
async def test_asr_flush_before_nlp():
    """Verify ASR completes before NLP runs on transcript."""
    # ... mock ASR provider that delays, verify transcript populated before NLP
    pass

def test_transcript_sorted_by_timestamp():
    """Verify transcript is sorted before NLP."""
    transcript = [
        {"t0": 5.0, "t1": 6.0, "text": "second"},
        {"t0": 1.0, "t1": 2.0, "text": "first"},
    ]
    sorted_t = sorted(transcript, key=lambda s: s.get("t0", 0.0))
    assert sorted_t[0]["text"] == "first"
```

### 8.2 Integration Test

```python
# tests/test_e2e_streaming.py

@pytest.mark.asyncio
async def test_multi_source_interleaved_audio():
    """Verify both system and mic sources produce separate transcripts."""
    # ... send alternating system/mic frames, verify both sources in final transcript
    pass
```

### 8.3 Type Checking

```bash
# pyproject.toml additions
[tool.mypy]
python_version = "3.11"
strict = true
plugins = ["pydantic.mypy"]

[[tool.mypy.overrides]]
module = "faster_whisper.*"
ignore_missing_imports = true
```

Run: `mypy server/`

---

## 9. Stabilization Plan

### Phase 1: Correctness and Determinism (1-2 days)

**Goal:** Eliminate crashes, data loss, and non-deterministic behavior.

| Task | Files | Effort |
|------|-------|--------|
| Add threading.Lock to ASRProviderRegistry | `asr_providers.py` | 15 min |
| Sort transcript by t0 before final NLP | `ws_live_listener.py:241` | 10 min |
| Surface ASR flush timeout as warning | `ws_live_listener.py:223-225` | 15 min |
| Fix pcm_buffer truncation logic | `ws_live_listener.py:200-205` | 20 min |
| Add backpressure status event | `ws_live_listener.py:74-77` | 30 min |
| Add unit tests for queue drop behavior | `tests/test_streaming_correctness.py` | 1 hour |
| Remove unused `state.tasks` field | `ws_live_listener.py:26` | 5 min |

**Validation:**
```bash
pytest tests/test_streaming_correctness.py -v
python scripts/test_websocket_server.py --live
```

### Phase 2: Performance and Latency (2-3 days)

**Goal:** Reduce latency, tune queue sizes, add observability.

| Task | Files | Effort |
|------|-------|--------|
| Add SessionMetrics dataclass | `ws_live_listener.py` | 30 min |
| Add structured logging on session end | `ws_live_listener.py` | 30 min |
| Tune QUEUE_MAX based on soak test | `ws_live_listener.py` | 2 hours |
| Add soak test harness | `scripts/soak_test.py` | 2 hours |
| Profile ASR inference time per chunk | `provider_faster_whisper.py` | 1 hour |
| Consider reducing chunk_seconds from 4 to 2 | `asr_stream.py` | 1 hour (testing) |

**Validation:**
```bash
python scripts/soak_test.py --duration 1800
# Assert: avg latency < 3s, no OOM, no crashes
```

### Phase 3: Quality Improvements (1 week)

**Goal:** Improve transcript accuracy and diarization.

| Task | Files | Effort |
|------|-------|--------|
| Add prompt conditioning to reduce repeats | `provider_faster_whisper.py` | 2 hours |
| Implement per-source diarization buffers | `ws_live_listener.py`, `diarization.py` | 4 hours |
| Improve speaker merge algorithm | `diarization.py` | 2 hours |
| Add overlap detection/deduplication | `analysis_stream.py` | 3 hours |
| Add protocol version negotiation | `ws_live_listener.py` | 2 hours |
| Request 16kHz directly from ScreenCaptureKit | `AudioCaptureManager.swift` | 30 min |

**Validation:**
- Manual testing with real meetings
- Compare WER before/after prompt conditioning
- Verify diarization segments are coherent

---

## 10. Appendix: Web Research Summary

### FastAPI/Starlette WebSocket Concurrency

**Finding:** Starlette's `WebSocket.send_text()` is NOT thread-safe for concurrent calls. Multiple async tasks sending simultaneously can cause race conditions.

**Evidence:** GitHub gist showing thread-safe queue pattern for WebSocket streaming.

**Implication:** The current `send_lock` pattern in `ws_send()` is correct and necessary.

### faster-whisper/CTranslate2 Thread Safety

**Finding:** `model.transcribe()` is NOT thread-safe. Concurrent calls can crash or produce corrupted output.

**Evidence:** faster-whisper GitHub issues discuss this limitation.

**Implication:** The current `_infer_lock` pattern is correct and necessary.

### macOS ScreenCaptureKit Audio Format

**Finding:** 
- Default audio is 48kHz stereo float32
- `SCStreamConfiguration.sampleRate` supports 8000, 16000, 24000, 48000
- If not specified, defaults to 48kHz

**Implication:** Current client-side resampling via AVAudioConverter is correct. Optionally request 16kHz directly to reduce CPU.

---

*End of Audit Document*
