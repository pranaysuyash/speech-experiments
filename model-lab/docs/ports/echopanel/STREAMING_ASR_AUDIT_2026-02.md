# EchoPanel Streaming ASR/NLP Audit Report

**Provenance**: Ported from `EchoPanel/docs/audit/STREAMING_ASR_AUDIT_2026-02.md` on 2026-02-05 for Model-Lab reference and experiment planning.

**Date**: February 2026  
**Scope**: Client-side capture → WebSocket transport → ASR → NLP → diarization  
**Status**: P0 issues fixed, P1/P2 issues documented

---

## Executive Summary

This audit identified **3 P0 (critical)** issues that would prevent the streaming ASR system from functioning, **5 P1 (high)** reliability issues, and **4 P2 (medium)** quality issues.

### Fixed in this PR

| Issue | Description | Status |
|-------|-------------|--------|
| P0-1 | `_pcm_stream()` function was undefined | ✅ Fixed |
| P0-2 | ASR task spawn logic was broken | Already fixed (started_sources) |
| P0-3 | CTranslate2/float16 on CPU crashes | ✅ Fixed (int8 fallback) |
| P1-2 | Stop semantics: ASR flush after analysis cancel | ✅ Fixed (order swapped) |
| P1-3 | Queue drop policy silent | ✅ Fixed (logging added) |
| Race | WebSocket send after close | ✅ Fixed (closed flag) |

---

## 1. Architecture Map

### Dataflow

```
┌────────────────────────────────────────────────────────────────────┐
│                         macOS Client                                │
├────────────────────────────────────────────────────────────────────┤
│ ScreenCaptureKit (float32, 48kHz) → AVAudioConverter → 16kHz mono  │
│ AVAudioEngine (mic) → AVAudioConverter → 16kHz mono                │
│        ↓                                                            │
│ WebSocketStreamer.sendPCMFrame(data, source="system"|"mic")        │
│        ↓ JSON: {"type":"audio","source":"...","data":"base64"}     │
└────────────────────────────────────────────────────────────────────┘
                              │ WebSocket
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                               │
├────────────────────────────────────────────────────────────────────┤
│ ws_live_listener.py                                                 │
│   ├─ SessionState (queues per source, transcript, send_lock)       │
│   ├─ on "audio" → base64 decode → get_queue → put_audio            │
│   │               → spawn _asr_loop if new source                  │
│   └─ on "stop"  → EOF queues → wait ASR flush → final_summary      │
│                                                                     │
│ _pcm_stream(queue) → yields bytes until EOF                        │
│                                                                     │
│ stream_asr(pcm_stream) → provider.transcribe_stream()              │
│                                                                     │
│ FasterWhisperProvider:                                              │
│   - Buffer accumulates until chunk_bytes (4s default)              │
│   - _transcribe() runs in thread with _infer_lock                  │
│   - Yields ASRSegment (is_final=True)                              │
└────────────────────────────────────────────────────────────────────┘
```

### Key Files

| File | Role |
|------|------|
| `server/api/ws_live_listener.py` | WebSocket handler, session state, task orchestration |
| `server/services/asr_stream.py` | Streaming ASR interface |
| `server/services/provider_faster_whisper.py` | faster-whisper ASR provider |
| `server/services/analysis_stream.py` | Entity/card extraction, rolling summary |
| `macapp/.../AudioCaptureManager.swift` | System audio capture |
| `macapp/.../MicrophoneCaptureManager.swift` | Microphone capture |
| `macapp/.../WebSocketStreamer.swift` | WebSocket client |

---

## 2. Issues Fixed

### P0-1: `_pcm_stream()` Undefined

**Location**: `ws_live_listener.py:67`

**Fix**: Added async generator:

```python
async def _pcm_stream(queue: asyncio.Queue) -> AsyncIterator[bytes]:
    """Drain audio queue until EOF (None sentinel)."""
    while True:
        chunk = await queue.get()
        if chunk is None:
            return
        yield chunk
```

### P0-3: CTranslate2 MPS/float16 Unsupported

**Location**: `provider_faster_whisper.py:59-71`

**Fix**: Applied pattern from `model-lab/harness/registry.py`:

```python
# MPS not supported by faster-whisper
if device == "mps":
    device = "cpu"

# float16 not supported on CPU
if device == "cpu" and compute_type == "float16":
    compute_type = "int8"
```

### P1-2: Stop Semantics Order

**Before**:
1. Cancel analysis tasks
2. Wait for ASR flush
3. Generate final summary (missing late transcripts)

**After**:
1. Wait for ASR flush (all transcripts captured)
2. Cancel analysis tasks
3. Generate final summary (complete transcript)

### Race Condition: Send After Close

**Fix**: Added `closed` flag to SessionState, check before send:

```python
async def ws_send(state, websocket, event):
    if state.closed:
        return
    async with state.send_lock:
        try:
            await websocket.send_text(json.dumps(event))
        except RuntimeError:
            state.closed = True
```

---

## 3. Remaining Issues (P1/P2)

### P1-1: Model Load Latency

**Symptom**: First ASR result delayed by 5-30s (model load time).

**Recommendation**: Pre-load model in FastAPI lifespan:

```python
# In main.py lifespan()
provider = ASRProviderRegistry.get_provider()
if hasattr(provider, '_get_model'):
    await asyncio.to_thread(provider._get_model)
```

### P1-5: Timestamp Drift

**Issue**: Server uses sample-based time, not client capture time.

**Recommendation**: Client should send monotonic timestamp with audio frames.

### P2-1: VAD Disabled by Default

**Issue**: Silence chunks are transcribed, wasting CPU.

**Recommendation**: Enable VAD for mic source:

```python
if source == AudioSource.MICROPHONE:
    vad_filter = True
```

### P2-2: Diarization Disabled

**Issue**: Multi-source buffer concatenation produces incoherent audio.

**Status**: Correctly disabled. Future: per-source buffers.

---

## 4. Web Research Findings

### FastAPI/Starlette WebSocket Concurrency

- WebSocket is NOT thread-safe for concurrent sends
- **Mitigation**: `asyncio.Lock` (already implemented) ✅

### faster-whisper/CTranslate2

- MPS not supported (CPU fallback required)
- float16 not supported on CPU (int8 fallback)
- Thread-safe for inference (can remove lock for performance)

### macOS Audio Capture

- ScreenCaptureKit: float32, 48kHz (varies by hardware)
- AVAudioEngine: varies by device
- Both managers use AVAudioConverter correctly ✅

---

## 5. Test Results

```
tests/test_services.py::test_extract_cards_empty PASSED
tests/test_services.py::test_extract_entities_empty PASSED
tests/test_ws_integration.py::test_source_tagged_audio_flow PASSED
tests/test_ws_live_listener.py::test_ws_live_listener_start_stop PASSED
======================== 4 passed ========================
```

---

## 6. Reliability and Failure-Mode Audit

### Network Failure Modes

**WebSocket Disconnect:**
- **Detection**: `WebSocketDisconnect` exception in receive loop
- **Recovery**: Client auto-reconnects with exponential backoff
- **Data Loss**: Audio frames in flight are lost
- **Session Continuity**: New connection starts fresh session

**Network Jitter:**
- **Impact**: Variable latency between audio frames
- **Current Handling**: Queues absorb jitter, drop oldest on overflow
- **Issue**: Timestamp accuracy degraded

**DNS/Network Outage:**
- **Detection**: Connection failures in `WebSocketStreamer.reconnect()`
- **Recovery**: Exponential backoff (1s → 10s max)
- **User Impact**: Streaming pauses until reconnection

### Server Failure Modes

**ASR Provider Unavailable:**
- **Detection**: `is_available = False` check
- **Fallback**: Yields status event, continues processing
- **Impact**: No transcription, but audio still flows

**Model Load Failure:**
- **Detection**: Exception in `_get_model()`
- **Impact**: ASR tasks fail silently
- **Logging**: Errors logged but not surfaced to client

**OOM from Large Sessions:**
- **Risk**: `pcm_buffer` grows to 1800s × 16kHz × 2 bytes = ~115MB
- **Current Limit**: 1800s hardcoded max
- **Issue**: No graceful degradation for very long sessions

### Client Failure Modes

**Audio Capture Failure:**
- **Detection**: `SCStream` errors in `AudioCaptureManager`
- **Recovery**: Logs error, continues (no crash)
- **Impact**: Audio streaming stops, but connection maintained

**Permission Denied:**
- **Detection**: `CGRequestScreenCaptureAccess()` returns false
- **User Impact**: Feature doesn't work, no clear error message

**AVAudioConverter Failure:**
- **Detection**: Converter creation returns nil
- **Impact**: Audio processing stops for that session

### Resource Limits

**Server Resources:**
- **CPU**: ASR inference blocks thread pool
- **Memory**: Bounded queues (48 chunks), time-limited PCM buffer
- **Connections**: No explicit limit on concurrent WebSocket connections

**Client Resources:**
- **CPU**: Audio conversion + base64 encoding
- **Memory**: Frame accumulation buffer
- **Network**: WebSocket + base64 overhead (~33% bandwidth increase)

### Timeout Analysis

**Current Timeouts:**
- **ASR Flush**: 8 seconds (configurable via `ECHOPANEL_ASR_FLUSH_TIMEOUT`)
- **Client Reconnect**: None (infinite retry)
- **WebSocket Ping**: 10 seconds (client-side only)

**Missing Timeouts:**
- **Session Duration**: No maximum session length
- **Client Inactivity**: No timeout for idle connections
- **ASR Inference**: No timeout on model transcribe calls

### Graceful Shutdown

**Server Shutdown:**
- **Current**: No signal handlers, abrupt termination
- **Issue**: In-flight sessions terminated uncleanly
- **Fix Needed**: SIGTERM handler to flush active sessions

**Client Shutdown:**
- **Current**: `stopCapture()` waits for SCStream cleanup
- **WebSocket**: Sends close frame, cancels tasks

### Chaos Testing Scenarios

**P1-7: Network partition mid-session**
- **Test**: Disconnect network during streaming
- **Expected**: Client reconnects, server handles disconnect gracefully
- **Current Issue**: Session state lost on reconnect

**P1-8: High latency network**
- **Test**: 500ms+ latency via network simulator
- **Expected**: Queues handle backpressure, no crashes
- **Current Issue**: Timestamp accuracy suffers

**P1-9: Rapid reconnect**
- **Test**: Kill server, restart quickly
- **Expected**: Client reconnects successfully
- **Current Issue**: May hit connection refused during restart

**P1-10: Large audio backlog**
- **Test**: Send 100+ audio frames before server processes
- **Expected**: Queue drops oldest, continues processing
- **Current**: May cause memory pressure

### Failure Mode Table

| Symptom | Detection | Mitigation | Priority |
|---------|-----------|------------|----------|
| WS disconnect | Exception in receive | Auto-reconnect | P1 |
| ASR unavailable | Provider check | Status event | P2 |
| Queue overflow | drop_oldest policy | Monitor drop rate | P2 |
| Model load fail | Exception handling | Log + fallback | P1 |
| OOM on long session | Memory limits | Time-based bounds | P1 |
| Audio capture fail | SCStream errors | Log + continue | P2 |
| Invalid audio format | Start validation | Error response | P2 |

---

## 7. Security and Privacy Audit (Practical, Not Theoretical)

### Authentication & Authorization

**Current State: NONE**
- **WebSocket**: No auth required, accepts any connection
- **Session IDs**: Client-generated, no validation
- **Origin Check**: No CORS or origin validation

**P0-4: Missing authentication**
- **Risk**: Anyone can connect and stream audio
- **Impact**: Unauthorized access to ASR resources
- **Mitigation**: Add token-based auth or API key validation

### Rate Limiting & Abuse Controls

**Current State: NONE**
- **Connections**: Unlimited concurrent WebSocket connections
- **Audio Volume**: No limits on audio data rate or volume
- **Session Duration**: No time limits on sessions
- **Requests**: No rate limiting on WebSocket messages

**P1-11: No rate limiting**
- **Risk**: DoS via unlimited connections or audio flood
- **Impact**: Server resource exhaustion
- **Mitigation**: Add connection limits, audio rate validation

### PII Handling & Privacy

**Data Stored:**
- **Transcripts**: In-memory only, cleared on session end
- **Audio**: PCM buffers in memory, up to 1800s
- **Analysis Results**: Entities, actions, decisions (in-memory)

**Data Transmission:**
- **WebSocket**: Unencrypted (ws://), audio as base64
- **Client Storage**: No persistent storage of transcripts
- **Server Logs**: May contain transcript snippets in debug logs

**P1-12: Unencrypted WebSocket**
- **Risk**: Audio/transcripts intercepted in transit
- **Impact**: Privacy violation for sensitive conversations
- **Mitigation**: WSS (WebSocket Secure) required

**P2-7: Potential PII in logs**
- **Risk**: Transcripts logged in debug mode
- **Impact**: Sensitive information in log files
- **Mitigation**: Redact PII from logs or disable transcript logging

### Dependency Security

**Python Dependencies:**
- **fastapi**: Generally secure, active maintenance
- **uvicorn**: Secure, active maintenance  
- **faster-whisper**: Depends on CTranslate2, check for CVEs
- **pyannote.audio**: Requires HF token, check model supply chain

**macOS Dependencies:**
- **ScreenCaptureKit**: System framework, secure by design
- **AVFoundation**: System framework, secure
- **WebSocket**: URLSession, secure implementation

**P2-8: Dependency scanning needed**
- **Action**: Run `pip audit` or `safety check` on Python deps
- **Action**: Check macOS app for vulnerable frameworks

### File Upload & Path Safety

**Current State: NONE**
- No file uploads implemented
- Audio data sent as base64 in JSON messages
- No file system access on server

**Future Risk:** If file uploads added, need:
- Path traversal prevention
- File type validation
- Size limits
- Virus scanning

### Threat Model (Lightweight)

**Assets:**
- Audio streams (privacy-sensitive)
- ASR transcripts (contain PII)
- Server compute resources
- Client system audio access

**Attackers:**
- **Network eavesdropper**: Can intercept unencrypted WS traffic
- **Malicious client**: Can flood server with audio/data
- **Unauthorized user**: Can connect without auth
- **Insider**: Access to logs containing transcripts

**Entry Points:**
- WebSocket connection (no auth)
- Audio data processing pipeline
- Model inference (potential model poisoning if models compromised)
- Log files (may contain PII)

**Mitigations Needed:**
- WSS encryption
- Connection authentication
- Rate limiting
- Input validation
- Log redaction

### Privacy Posture Assessment

**Data Collection:** Audio + transcripts (ephemeral, in-memory only)
**Data Retention:** Session duration only, cleared on disconnect
**Data Sharing:** None (no third parties)
**User Control:** None (no opt-out, no data deletion)
**Compliance:** Not GDPR/CCPA compliant (no user rights, no consent)

**P1-13: No privacy controls**
- **Issue**: Users cannot control data collection or request deletion
- **Impact**: Legal compliance risk
- **Mitigation**: Add privacy policy, data handling controls

---

## 8. Observability and Operability Audit

### Current Logging

**Server Logs:**
- **WebSocket events**: Connection, messages, disconnects (DEBUG level)
- **ASR processing**: Chunk processing, model loading, inference (DEBUG level)
- **Errors**: Exceptions in ASR/analysis loops (ERROR level)
- **Audio stats**: Bytes received, dropped frames (DEBUG level)

**Client Logs:**
- **WebSocket**: Connection, reconnection, errors (console)
- **Audio capture**: Sample counts, format details (console)
- **Performance**: Frame processing timing (DEBUG)

**Issues:**
- **P2-9: Inconsistent log levels**
  - Some debug info at INFO level, errors not consistently logged
- **P2-10: No structured logging**
  - Plain text logs, hard to parse/aggregate
- **P2-11: No log redaction**
  - Transcripts may appear in debug logs (privacy issue)

### Missing Observability

**Metrics (None currently):**
- **Connection count**: Active WebSocket connections
- **Audio throughput**: Bytes/second processed
- **ASR latency**: Time from audio chunk to transcript segment
- **Queue depth**: Current queue lengths
- **Error rates**: Failed connections, ASR errors, dropped frames
- **Session duration**: Average/completed session lengths

**Tracing (None currently):**
- **Request tracing**: Session ID through entire pipeline
- **Performance tracing**: Time spent in each pipeline stage
- **Error correlation**: Link errors to specific sessions

**Health Checks:**
- **Basic**: `/health` endpoint checks ASR provider availability
- **Missing**: Queue health, memory usage, connection pool status

### Debug Capabilities

**Current Debug Features:**
- **Environment variable**: `ECHOPANEL_DEBUG=1` enables verbose logging
- **Session logging**: Per-session debug info with session IDs
- **Audio inspection**: Buffer sizes, format validation

**Missing Debug Features:**
- **Audio dump**: Save last N seconds of audio for repro (opt-in)
- **Session state dump**: Export full session state on error
- **Performance profiling**: CPU/memory profiling hooks
- **Test hooks**: Inject synthetic audio/errors for testing

### Operational Pain Points

**Troubleshooting:**
- **Hard to correlate**: No session IDs in all log lines
- **No metrics dashboard**: Can't see system health at a glance
- **Debug overhead**: Verbose logging impacts performance
- **Post-mortem**: Limited data for incident analysis

**Monitoring:**
- **No alerts**: No alerting on high error rates or resource usage
- **No dashboards**: No visualization of system metrics
- **No SLOs**: No service level objectives defined

### Recommended Observability Stack

**Minimal Ops Dashboard:**
```
Active Connections: 5
ASR Latency P95: 2.3s
Queue Drop Rate: 0.1%
Error Rate (1h): 0.05%
Memory Usage: 450MB
```

**Log Schema:**
```json
{
  "timestamp": "2026-02-04T10:30:45.123Z",
  "level": "INFO",
  "session_id": "sess_12345",
  "component": "asr_stream",
  "event": "chunk_processed",
  "duration_ms": 2340,
  "chunk_bytes": 128000,
  "segments": 3
}
```

**Key Metrics to Track:**
- `websocket_connections_active`: Gauge
- `audio_bytes_processed_total`: Counter  
- `asr_inference_duration_seconds`: Histogram
- `audio_frames_dropped_total`: Counter
- `session_duration_seconds`: Histogram
- `asr_errors_total`: Counter

### Debug Hook Implementation

**Audio Dump (Opt-in):**
```python
if os.getenv("ECHOPANEL_DEBUG_AUDIO_DUMP"):
    with open(f"debug_audio_{session_id}_{timestamp}.pcm", "wb") as f:
        f.write(pcm_buffer[-debug_seconds * 16000 * 2:])
```

**Session State Export:**
```python
@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    # Export session state for debugging
    return get_session_state(session_id)
```

### Operational Readiness Score

**Current: 3/10**
- ✅ Basic health check
- ✅ Error logging
- ❌ No metrics
- ❌ No structured logging
- ❌ No tracing
- ❌ No alerting
- ❌ Limited debug capabilities
- ❌ No runbooks
- ❌ No incident response plan

---

## 9. Testing Strategy Audit (This is Where Products Become Real)

### Current Test Coverage

**Unit Tests:**
- ✅ `test_extract_cards_empty`: Tests card extraction with empty input
- ✅ `test_extract_entities_empty`: Tests entity extraction with empty input
- ❌ No ASR pipeline unit tests
- ❌ No WebSocket protocol tests
- ❌ No audio format validation tests

**Integration Tests:**
- ✅ `test_source_tagged_audio_flow`: Tests WS with source-tagged audio
- ✅ `test_ws_live_listener_start_stop`: Tests session lifecycle
- ❌ No end-to-end audio processing tests
- ❌ No failure mode tests

**Test Harnesses Missing:**
- ❌ Simulated client for load testing
- ❌ Audio injection for ASR testing
- ❌ Network fault injection
- ❌ Chaos testing framework

### Required Test Harnesses

**1. Unit Test Expansion**

**ASR Pipeline Tests:**
```python
def test_asr_provider_fallback():
    # Test behavior when faster-whisper unavailable
    
def test_asr_segment_formatting():
    # Test ASRSegment → dict conversion
    
def test_provider_config_validation():
    # Test invalid config handling
```

**WebSocket Protocol Tests:**
```python
def test_invalid_start_message():
    # Test error response for invalid format
    
def test_stop_before_start():
    # Test idempotency
    
def test_concurrent_audio_sources():
    # Test multiple source handling
```

**2. Integration Test Expansion**

**End-to-End Audio Flow:**
```python
def test_full_audio_pipeline():
    # Inject PCM → get transcript segments
    
def test_session_reconnect():
    # Test session state across reconnect
    
def test_long_session_memory():
    # Test memory usage over time
```

**3. Load and Stress Tests**

**Audio Throughput Test:**
```python
def test_high_audio_volume():
    # Send audio faster than real-time
    # Verify queue handling, no crashes
```

**Concurrent Sessions Test:**
```python
def test_multiple_concurrent_sessions():
    # 10+ simultaneous sessions
    # Verify resource isolation
```

**4. Chaos and Failure Tests**

**Network Partition Test:**
```python
def test_websocket_disconnect_recovery():
    # Simulate network failure mid-session
    # Verify reconnect and state consistency
```

**Resource Exhaustion Test:**
```python
def test_memory_limits():
    # Send unlimited audio
    # Verify graceful degradation
```

### Test Data and Fixtures

**Audio Test Vectors:**

**Silence (4 seconds):**
- **File**: `tests/fixtures/silence_16k_mono.pcm`
- **Expected**: No ASR segments yielded

**Speech Sample:**
- **File**: `tests/fixtures/speech_sample_16k_mono.pcm` 
- **Expected**: Valid transcript segments with text

**Multi-speaker:**
- **File**: `tests/fixtures/two_speakers_16k_mono.pcm`
- **Expected**: Speaker labels assigned correctly

**Format Edge Cases:**
- **Invalid sample rate**: Error response, connection close
- **Wrong channels**: Error response, connection close
- **Corrupted base64**: Graceful handling

### Test Execution Strategy

**CI Pipeline:**
```
1. Unit tests (fast, < 30s)
2. Integration tests (medium, < 5min)  
3. Load tests (slow, < 15min, nightly)
4. Chaos tests (manual, on-demand)
```

**Test Environments:**
- **Unit**: Mocked dependencies, fast
- **Integration**: Real server, synthetic audio
- **Load**: Multiple server instances, real load
- **Chaos**: Network simulation, fault injection

### Test Quality Gates

**Definition of Done for Features:**
- ✅ Unit test coverage > 80%
- ✅ Integration tests pass
- ✅ Load test handles 2x expected load
- ✅ Chaos tests pass for known failure modes
- ✅ Manual testing confirms UX quality

**Regression Prevention:**
- ✅ All tests run on every PR
- ✅ Performance regression detection
- ✅ Memory leak detection in long-running tests

### Golden Fixtures and Baselines

**Transcript Accuracy Baselines:**
- **WER target**: < 10% on clean speech
- **Latency target**: < 3 seconds P95
- **Memory target**: < 500MB per session

**Performance Baselines:**
- **Throughput**: 100 concurrent sessions
- **Startup time**: < 5 seconds
- **Recovery time**: < 10 seconds after failure

### Testing Debt Assessment

**Current State: Poor (2/10)**
- ✅ Basic unit tests exist
- ✅ Integration tests exist  
- ❌ No load testing
- ❌ No chaos testing
- ❌ No test data fixtures
- ❌ No CI quality gates
- ❌ No regression detection

**Priority Improvements:**
1. **P1**: Add ASR pipeline unit tests
2. **P1**: Create audio test fixtures
3. **P2**: Add load testing harness
4. **P2**: Implement chaos testing
5. **P2**: Add performance monitoring

---

## 10. Output: Issue Log + Execution Backlog

### Session State Machine

**States:**
- `IDLE`: WebSocket connected, no session started
- `STREAMING`: Session started, ASR tasks running, accepting audio
- `STOPPING`: Stop received, flushing ASR tasks, running final analysis
- `FINALIZING`: ASR flushed, running final NLP analysis
- `DONE`: Final summary sent, connection closed

**Transitions:**
```
IDLE → STREAMING: "start" message received
STREAMING → STOPPING: "stop" message received  
STOPPING → FINALIZING: ASR tasks complete
FINALIZING → DONE: Final analysis complete, summary sent
```

### Session Invariants

1. **Audio Queue Bounds**: Each source queue maxsize=48 (bounded memory)
2. **Task Lifecycle**: ASR tasks only created once per source via `started_sources` set
3. **Transcript Ordering**: Segments appended in timestamp order (ASR yields chronologically)
4. **Stop Atomicity**: Final transcript snapshot taken after ASR flush completes
5. **Connection Safety**: No sends after `state.closed = True`
6. **Source Isolation**: Audio from different sources processed independently

### Race Condition Analysis

#### Identified Risks & Mitigations

| Risk | Location | Mitigation | Status |
|------|----------|------------|--------|
| Concurrent WS sends | `ws_send()` | `asyncio.Lock` | ✅ Fixed |
| Send after close | `ws_send()` | `closed` flag check | ✅ Fixed |
| Late ASR events | Stop handler | Wait for ASR flush first | ✅ Fixed |
| Queue overflow | `put_audio()` | `drop_oldest` policy | ✅ Fixed |
| Multiple starts | `started_sources` | Set prevents duplicate tasks | ✅ Fixed |
| Transcript mutation | Final summary | Snapshot after flush | ✅ Fixed |

#### Remaining Race Risks

**P2-3: Analysis task snapshot race**
- **Risk**: Analysis tasks snapshot `state.transcript` concurrently with ASR appends
- **Impact**: Inconsistent entity/card counts between real-time and final
- **Mitigation**: Use `asyncio.Lock` for transcript access or accept minor inconsistency

### Backpressure Analysis

**Client → Server:**
- WebSocket has natural backpressure (TCP flow control)
- Server queues drop oldest on full (prevents unbounded growth)
- No explicit backpressure signaling to client

**Server Internal:**
- ASR processing is async, no blocking queues
- Analysis runs off-event-loop via `asyncio.to_thread`
- Diarization is synchronous but run off-event-loop

### Idempotency Testing

**Start called twice:**
- Second start ignored (no state reset)
- **Issue**: Should reset session or reject?

**Stop called twice:** 
- Second stop sends duplicate final_summary
- **Issue**: Should be idempotent (no-op on second call)

**Reconnect mid-session:**
- New WebSocket connection, new SessionState
- **Issue**: No session resumption, starts fresh

### Buffer Bounds Analysis

| Buffer | Type | Limit | Growth Risk |
|--------|------|-------|-------------|
| `pcm_buffer` | Diarization | 1800s max | Bounded by time limit |
| `state.transcript` | List[dict] | None | Grows with session length |
| Audio queues | asyncio.Queue | maxsize=48 | Bounded, drops oldest |
| ASR buffer | bytearray | chunk_bytes | Bounded by chunk size |
| Analysis snapshots | List copy | None | Temporary, GC'd after use |

---

## 8. Audio Pipeline Audit (Format + Timing)

### Audio Contract Specification

**Client Output Format:**
- **Sample Rate**: 16000 Hz (fixed)
- **Channels**: 1 (mono)
- **Sample Format**: PCM16 signed little-endian (-32768 to +32767)
- **Frame Size**: 320 samples (20ms at 16kHz)
- **Transport**: Base64-encoded in JSON WebSocket messages

**Server Input Expectations:**
- **Sample Rate**: 16000 Hz only (validated on start)
- **Channels**: 1 only (validated on start)  
- **Sample Format**: PCM16 signed little-endian only
- **Frame Size**: Variable (accepts any size chunks)
- **Transport**: WebSocket binary or JSON with base64

### Format Validation

**✅ Correctly Implemented:**
- Client validates start message format requirements
- Server rejects unsupported formats with error message
- AVAudioConverter handles sample rate conversion
- PCM16 conversion with proper clamping: `max(-1.0, min(1.0, samples[i]))`

**❌ Issues Found:**

**P2-4: Frame size inconsistency**
- **Client**: Fixed 320-sample frames (20ms)
- **Server**: Accepts variable chunk sizes
- **Impact**: ASR chunking may not align with client framing
- **Risk**: Partial frames at chunk boundaries

### Timing Analysis

**Client Capture Timing:**
- ScreenCaptureKit: ~20ms native frame rate
- Frame emission: Synchronous with capture
- No timestamps sent with audio frames

**Server Receive Timing:**
- WebSocket buffering: Variable latency
- Processing: Async, non-blocking
- ASR chunking: 4-second windows, timestamped from processed samples

**P1-5: Timestamp drift (already documented)**
- **Issue**: Server timestamps based on processed samples, not capture time
- **Impact**: Transcript timestamps don't match real-world timing
- **Mitigation**: Client should send capture timestamps with frames

### Silence Handling

**Current Behavior:**
- No VAD (Voice Activity Detection) on system audio
- All audio chunks processed, including silence
- ASR yields segments for silence (empty text filtered out)

**P2-1: VAD disabled (already documented)**
- **Impact**: Wasted CPU on silence transcription
- **Recommendation**: Enable VAD for microphone source

### Sample Rate Conversion Quality

**Client Path:**
```
ScreenCaptureKit (float32, 48kHz+) 
    ↓ AVAudioConverter 
16kHz mono float32 
    ↓ Manual conversion
16kHz mono PCM16
```

**Quality Assessment:**
- ✅ AVAudioConverter uses high-quality resampling
- ✅ Proper dithering and anti-aliasing
- ✅ Manual float32→PCM16 with clamping prevents clipping

### Clock Synchronization

**Current Issues:**
- No synchronization between client capture time and server processing time
- WebSocket latency not accounted for in timestamps
- Network jitter affects timing accuracy

**Recommended Fix:**
```json
{
  "type": "audio",
  "source": "system", 
  "data": "base64...",
  "timestamp": 1640995200.123456  // CACurrentMediaTime() or monotonic
}
```

---

## 9. Diarization and Merge Audit (Semantic Correctness)

### Current Diarization Status

**Status: DISABLED** (correctly, due to multi-source issues)

**Reason for Disable:**
- Diarization requires concatenated PCM buffer from all sources
- Multi-source audio (system + mic) creates incoherent mixed audio
- Speaker labels become meaningless

### Diarization Pipeline (When Enabled)

**Input Processing:**
```python
# diarize_pcm() in diarization.py
audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
waveform = torch.from_numpy(audio).unsqueeze(0)
diarization = pipeline({"waveform": waveform, "sample_rate": sample_rate})
```

**Segment Generation:**
- pyannote.audio yields raw segments: `(start, end, speaker_id)`
- Speaker IDs: "SPEAKER_00", "SPEAKER_01", etc.
- No confidence scores or overlap handling

**Post-Processing:**
1. **Merge Adjacent**: Same speaker segments within 0.5s gap
2. **Name Assignment**: "SPEAKER_00" → "Speaker 1", "SPEAKER_01" → "Speaker 2"
3. **Transcript Merge**: Assign speakers to transcript segments by midpoint overlap

### Merge Algorithm Analysis

**Current Implementation:**
```python
def merge_transcript_with_speakers(transcript, speaker_segments):
    for seg in transcript:
        mid = (seg["t0"] + seg["t1"]) / 2.0
        # Find speaker segment containing midpoint
        for spk_seg in speaker_segments:
            if spk_seg["t0"] <= mid <= spk_seg["t1"]:
                seg["speaker"] = spk_seg["speaker"]
                break
```

**✅ Correctly Handles:**
- Segments within speaker boundaries
- Multiple speakers in same transcript
- Speaker boundaries not aligning with transcript boundaries

**❌ Issues Found:**

**P2-5: No fallback for uncovered segments**
- **Issue**: Transcript segments not covered by any speaker segment get no speaker label
- **Impact**: Some transcript text appears without speaker attribution
- **Mitigation**: Assign "Unknown Speaker" or interpolate from adjacent segments

**P2-6: Midpoint overlap is simplistic**
- **Issue**: Uses segment midpoint only, ignores segment duration
- **Impact**: Long segments might be misassigned if midpoint is in wrong speaker region
- **Better**: Use majority overlap or weighted center of mass

### Determinism Analysis

**Current Behavior:**
- pyannote.audio produces deterministic output for same input
- Speaker ID assignment is deterministic (first-seen ordering)
- Merge algorithm is deterministic

**Edge Cases:**
- **Overlapping speech**: pyannote may assign same speaker to overlapping segments
- **Quick turn-taking**: May create very short segments
- **Background noise**: May create spurious speaker segments

### Confidence and Quality Handling

**Missing Features:**
- No confidence scores from diarization
- No filtering of low-confidence speaker assignments
- No handling of "unknown" or "overlapping" speakers

### Future Diarization Architecture

**Per-Source Diarization (Recommended):**
```
System Audio Buffer ──→ Diarization ──→ Speaker Segments (System)
Mic Audio Buffer    ──→ Diarization ──→ Speaker Segments (Mic)
                                        ↓
                             Merge by Time + Source
```

**Cross-Source Speaker Matching:**
- Challenge: Same person in system vs mic audio
- Solution: Voice fingerprinting or user-provided speaker names

---

## 10. Reliability and Failure-Mode Audit

### Network Failure Modes

**WebSocket Disconnect:**
- **Detection**: `WebSocketDisconnect` exception in receive loop
- **Recovery**: Client auto-reconnects with exponential backoff
- **Data Loss**: Audio frames in flight are lost
- **Session Continuity**: New connection starts fresh session

**Network Jitter:**
- **Impact**: Variable latency between audio frames
- **Current Handling**: Queues absorb jitter, drop oldest on overflow
- **Issue**: Timestamp accuracy degraded

**DNS/Network Outage:**
- **Detection**: Connection failures in `WebSocketStreamer.reconnect()`
- **Recovery**: Exponential backoff (1s → 10s max)
- **User Impact**: Streaming pauses until reconnection

### Server Failure Modes

**ASR Provider Unavailable:**
- **Detection**: `is_available = False` check
- **Fallback**: Yields status event, continues processing
- **Impact**: No transcription, but audio still flows

**Model Load Failure:**
- **Detection**: Exception in `_get_model()`
- **Impact**: ASR tasks fail silently
- **Logging**: Errors logged but not surfaced to client

**OOM from Large Sessions:**
- **Risk**: `pcm_buffer` grows to 1800s × 16kHz × 2 bytes = ~115MB
- **Current Limit**: 1800s hardcoded max
- **Issue**: No graceful degradation for very long sessions

### Client Failure Modes

**Audio Capture Failure:**
- **Detection**: `SCStream` errors in `AudioCaptureManager`
- **Recovery**: Logs error, continues (no crash)
- **Impact**: Audio streaming stops, but connection maintained

**Permission Denied:**
- **Detection**: `CGRequestScreenCaptureAccess()` returns false
- **User Impact**: Feature doesn't work, no clear error message

**AVAudioConverter Failure:**
- **Detection**: Converter creation returns nil
- **Impact**: Audio processing stops for that session

### Resource Limits

**Server Resources:**
- **CPU**: ASR inference blocks thread pool
- **Memory**: Bounded queues (48 chunks), time-limited PCM buffer
- **Connections**: No explicit limit on concurrent WebSocket connections

**Client Resources:**
- **CPU**: Audio conversion + base64 encoding
- **Memory**: Frame accumulation buffer
- **Network**: WebSocket + base64 overhead (~33% bandwidth increase)

### Timeout Analysis

**Current Timeouts:**
- **ASR Flush**: 8 seconds (configurable via `ECHOPANEL_ASR_FLUSH_TIMEOUT`)
- **Client Reconnect**: None (infinite retry)
- **WebSocket Ping**: 10 seconds (client-side only)

**Missing Timeouts:**
- **Session Duration**: No maximum session length
- **Client Inactivity**: No timeout for idle connections
- **ASR Inference**: No timeout on model transcribe calls

### Graceful Shutdown

**Server Shutdown:**
- **Current**: No signal handlers, abrupt termination
- **Issue**: In-flight sessions terminated uncleanly
- **Fix Needed**: SIGTERM handler to flush active sessions

**Client Shutdown:**
- **Current**: `stopCapture()` waits for SCStream cleanup
- **WebSocket**: Sends close frame, cancels tasks

### Chaos Testing Scenarios

**P1-7: Network partition mid-session**
- **Test**: Disconnect network during streaming
- **Expected**: Client reconnects, server handles disconnect gracefully
- **Current Issue**: Session state lost on reconnect

**P1-8: High latency network**
- **Test**: 500ms+ latency via network simulator
- **Expected**: Queues handle backpressure, no crashes
- **Current Issue**: Timestamp accuracy suffers

**P1-9: Rapid reconnect**
- **Test**: Kill server, restart quickly
- **Expected**: Client reconnects successfully
- **Current Issue**: May hit connection refused during restart

**P1-10: Large audio backlog**
- **Test**: Send 100+ audio frames before server processes
- **Expected**: Queue drops oldest, continues processing
- **Current**: May cause memory pressure

---

## 11. Security and Privacy Audit (Practical, Not Theoretical)

### Authentication & Authorization

**Current State: NONE**
- **WebSocket**: No auth required, accepts any connection
- **Session IDs**: Client-generated, no validation
- **Origin Check**: No CORS or origin validation

**P0-4: Missing authentication**
- **Risk**: Anyone can connect and stream audio
- **Impact**: Unauthorized access to ASR resources
- **Mitigation**: Add token-based auth or API key validation

### Rate Limiting & Abuse Controls

**Current State: NONE**
- **Connections**: Unlimited concurrent WebSocket connections
- **Audio Volume**: No limits on audio data rate or volume
- **Session Duration**: No time limits on sessions
- **Requests**: No rate limiting on WebSocket messages

**P1-11: No rate limiting**
- **Risk**: DoS via unlimited connections or audio flood
- **Impact**: Server resource exhaustion
- **Mitigation**: Connection limits, audio rate validation

### PII Handling & Privacy

**Data Stored:**
- **Transcripts**: In-memory only, cleared on session end
- **Audio**: PCM buffers in memory, up to 1800s
- **Analysis Results**: Entities, actions, decisions (in-memory)

**Data Transmission:**
- **WebSocket**: Unencrypted (ws://), audio as base64
- **Client Storage**: No persistent storage of transcripts
- **Server Logs**: May contain transcript snippets in debug logs

**P1-12: Unencrypted WebSocket**
- **Risk**: Audio/transcripts intercepted in transit
- **Impact**: Privacy violation for sensitive conversations
- **Mitigation**: WSS (WebSocket Secure) required

**P2-7: Potential PII in logs**
- **Risk**: Transcripts logged in debug mode
- **Impact**: Sensitive information in log files
- **Mitigation**: Redact PII from logs or disable transcript logging

### Dependency Security

**Python Dependencies:**
- **fastapi**: Generally secure, active maintenance
- **uvicorn**: Secure, active maintenance  
- **faster-whisper**: Depends on CTranslate2, check for CVEs
- **pyannote.audio**: Requires HF token, check model supply chain

**macOS Dependencies:**
- **ScreenCaptureKit**: System framework, secure by design
- **AVFoundation**: System framework, secure
- **WebSocket**: URLSession, secure implementation

**P2-8: Dependency scanning needed**
- **Action**: Run `pip audit` or `safety check` on Python deps
- **Action**: Check macOS app for vulnerable frameworks

### File Upload & Path Safety

**Current State: NONE**
- No file uploads implemented
- Audio data sent as base64 in JSON messages
- No file system access on server

**Future Risk:** If file uploads added, need:
- Path traversal prevention
- File type validation
- Size limits
- Virus scanning

### Threat Model (Lightweight)

**Assets:**
- Audio streams (privacy-sensitive)
- ASR transcripts (contain PII)
- Server compute resources
- Client system audio access

**Attackers:**
- **Network eavesdropper**: Can intercept unencrypted WS traffic
- **Malicious client**: Can flood server with audio/data
- **Unauthorized user**: Can connect without auth
- **Insider**: Access to logs containing transcripts

**Entry Points:**
- WebSocket connection (no auth)
- Audio data processing pipeline
- Model inference (potential model poisoning if models compromised)
- Log files (may contain PII)

**Mitigations Needed:**
- WSS encryption
- Connection authentication
- Rate limiting
- Input validation
- Log redaction

---

## 12. Observability and Operability Audit

### Current Logging

**Server Logs:**
- **WebSocket events**: Connection, messages, disconnects (DEBUG level)
- **ASR processing**: Chunk processing, model loading, inference (DEBUG level)
- **Errors**: Exceptions in ASR/analysis loops (ERROR level)
- **Audio stats**: Bytes received, dropped frames (DEBUG level)

**Client Logs:**
- **WebSocket**: Connection, reconnection, errors (console)
- **Audio capture**: Sample counts, format details (console)
- **Performance**: Frame processing timing (DEBUG)

**Issues:**
- **P2-9: Inconsistent log levels**
  - Some debug info at INFO level, errors not consistently logged
- **P2-10: No structured logging**
  - Plain text logs, hard to parse/aggregate
- **P2-11: No log redaction**
  - Transcripts may appear in debug logs (privacy issue)

### Missing Observability

**Metrics (None currently):**
- **Connection count**: Active WebSocket connections
- **Audio throughput**: Bytes/second processed
- **ASR latency**: Time from audio chunk to transcript segment
- **Queue depth**: Current queue lengths
- **Error rates**: Failed connections, ASR errors, dropped frames
- **Session duration**: Average/completed session lengths

**Tracing (None currently):**
- **Request tracing**: Session ID through entire pipeline
- **Performance tracing**: Time spent in each pipeline stage
- **Error correlation**: Link errors to specific sessions

**Health Checks:**
- **Basic**: `/health` endpoint checks ASR provider availability
- **Missing**: Queue health, memory usage, connection pool status

### Debug Capabilities

**Current Debug Features:**
- **Environment variable**: `ECHOPANEL_DEBUG=1` enables verbose logging
- **Session logging**: Per-session debug info with session IDs
- **Audio inspection**: Buffer sizes, format validation

**Missing Debug Features:**
- **Audio dump**: Save last N seconds of audio for repro (opt-in)
- **Session state dump**: Export full session state on error
- **Performance profiling**: CPU/memory profiling hooks
- **Test hooks**: Inject synthetic audio/errors for testing

### Operational Pain Points

**Troubleshooting:**
- **Hard to correlate**: No session IDs in all log lines
- **No metrics dashboard**: Can't see system health at a glance
- **Debug overhead**: Verbose logging impacts performance
- **Post-mortem**: Limited data for incident analysis

**Monitoring:**
- **No alerts**: No alerting on high error rates or resource usage
- **No dashboards**: No visualization of system metrics
- **No SLOs**: No service level objectives defined

### Recommended Observability Stack

**Minimal Ops Dashboard:**
```
Active Connections: 5
ASR Latency P95: 2.3s
Queue Drop Rate: 0.1%
Error Rate (1h): 0.05%
Memory Usage: 450MB
```

**Log Schema:**
```json
{
  "timestamp": "2026-02-04T10:30:45.123Z",
  "level": "INFO",
  "session_id": "sess_12345",
  "component": "asr_stream",
  "event": "chunk_processed",
  "duration_ms": 2340,
  "chunk_bytes": 128000,
  "segments": 3
}
```

**Key Metrics to Track:**
- `websocket_connections_active`: Gauge
- `audio_bytes_processed_total`: Counter  
- `asr_inference_duration_seconds`: Histogram
- `audio_frames_dropped_total`: Counter
- `session_duration_seconds`: Histogram
- `asr_errors_total`: Counter

### Debug Hook Implementation

**Audio Dump (Opt-in):**
```python
if os.getenv("ECHOPANEL_DEBUG_AUDIO_DUMP"):
    with open(f"debug_audio_{session_id}_{timestamp}.pcm", "wb") as f:
        f.write(pcm_buffer[-debug_seconds * 16000 * 2:])
```

**Session State Export:**
```python
@app.get("/debug/session/{session_id}")
def debug_session(session_id: str):
    # Export session state for debugging
    return get_session_state(session_id)
```

---

## 13. Testing Strategy Audit (This is Where Products Become Real)

### Current Test Coverage

**Unit Tests:**
- ✅ `test_extract_cards_empty`: Tests card extraction with empty input
- ✅ `test_extract_entities_empty`: Tests entity extraction with empty input
- ❌ No ASR pipeline unit tests
- ❌ No WebSocket protocol tests
- ❌ No audio format validation tests

**Integration Tests:**
- ✅ `test_source_tagged_audio_flow`: Tests WS with source-tagged audio
- ✅ `test_ws_live_listener_start_stop`: Tests session lifecycle
- ❌ No end-to-end audio processing tests
- ❌ No failure mode tests

**Test Harnesses Missing:**
- ❌ Simulated client for load testing
- ❌ Audio injection for ASR testing
- ❌ Network fault injection
- ❌ Chaos testing framework

### Required Test Harnesses

**1. Unit Test Expansion**

**ASR Pipeline Tests:**
```python
def test_asr_provider_fallback():
    # Test behavior when faster-whisper unavailable
    
def test_asr_segment_formatting():
    # Test ASRSegment → dict conversion
    
def test_provider_config_validation():
    # Test invalid config handling
```

**WebSocket Protocol Tests:**
```python
def test_invalid_start_message():
    # Test error response for invalid format
    
def test_stop_before_start():
    # Test idempotency
    
def test_concurrent_audio_sources():
    # Test multiple source handling
```

**2. Integration Test Expansion**

**End-to-End Audio Flow:**
```python
def test_full_audio_pipeline():
    # Inject PCM → get transcript segments
    
def test_session_reconnect():
    # Test session state across reconnect
    
def test_long_session_memory():
    # Test memory usage over time
```

**3. Load and Stress Tests**

**Audio Throughput Test:**
```python
def test_high_audio_volume():
    # Send audio faster than real-time
    # Verify queue handling, no crashes
```

**Concurrent Sessions Test:**
```python
def test_multiple_concurrent_sessions():
    # 10+ simultaneous sessions
    # Verify resource isolation
```

**4. Chaos and Failure Tests**

**Network Partition Test:**
```python
def test_websocket_disconnect_recovery():
    # Simulate network failure mid-session
    # Verify reconnect and state consistency
```

**Resource Exhaustion Test:**
```python
def test_memory_limits():
    # Send unlimited audio
    # Verify graceful degradation
```

### Test Data and Fixtures

**Audio Test Vectors:**

**Silence (4 seconds):**
- **File**: `tests/fixtures/silence_16k_mono.pcm`
- **Expected**: No ASR segments yielded

**Speech Sample:**
- **File**: `tests/fixtures/speech_sample_16k_mono.pcm` 
- **Expected**: Valid transcript segments with text

**Multi-speaker:**
- **File**: `tests/fixtures/two_speakers_16k_mono.pcm`
- **Expected**: Speaker labels assigned correctly

**Format Edge Cases:**
- **Invalid sample rate**: Error response, connection close
- **Wrong channels**: Error response, connection close
- **Corrupted base64**: Graceful handling

### Test Execution Strategy

**CI Pipeline:**
```
1. Unit tests (fast, < 30s)
2. Integration tests (medium, < 5min)  
3. Load tests (slow, < 15min, nightly)
4. Chaos tests (manual, on-demand)
```

**Test Environments:**
- **Unit**: Mocked dependencies, fast
- **Integration**: Real server, synthetic audio
- **Load**: Multiple server instances, real load
- **Chaos**: Network simulation, fault injection

### Test Quality Gates

**Definition of Done for Features:**
- ✅ Unit test coverage > 80%
- ✅ Integration tests pass
- ✅ Load test handles 2x expected load
- ✅ Chaos tests pass for known failure modes
- ✅ Manual testing confirms UX quality

**Regression Prevention:**
- ✅ All tests run on every PR
- ✅ Performance regression detection
- ✅ Memory leak detection in long-running tests

### Golden Fixtures and Baselines

**Transcript Accuracy Baselines:**
- **WER target**: < 10% on clean speech
- **Latency target**: < 3 seconds P95
- **Memory target**: < 500MB per session

**Performance Baselines:**
- **Throughput**: 100 concurrent sessions
- **Startup time**: < 5 seconds
- **Recovery time**: < 10 seconds after failure

### Testing Debt Assessment

**Current State: Poor (2/10)**
- ✅ Basic unit tests exist
- ✅ Integration tests exist  
- ❌ No load testing
- ❌ No chaos testing
- ❌ No performance baselines
- ❌ No test data fixtures
- ❌ No CI quality gates
- ❌ No regression detection

**Priority Improvements:**
1. **P1**: Add ASR pipeline unit tests
2. **P1**: Create audio test fixtures
3. **P2**: Add load testing harness
4. **P2**: Implement chaos testing
5. **P2**: Add performance monitoring

---

## 10. Output: Issue Log + Execution Backlog

### Issue Classification

**Severity Rubric:**
- **P0**: Data loss, security breach, crashes, "cannot ship"
- **P1**: Reliability failures, performance issues, major UX breaks
- **P2**: Quality issues, edge cases, nice-to-have improvements

**Confidence Levels:**
- **High**: Proven with code analysis or testing
- **Medium**: Likely based on architecture review
- **Low**: Speculative, needs investigation

---

### P0 Issues (Critical - Block Release)

| ID | Issue | Evidence | Impact | Fix | Confidence |
|----|-------|----------|--------|-----|------------|
| P0-4 | Missing authentication | WS accepts any connection | Unauthorized access to ASR | Add token-based auth | High |
| P0-5 | Unencrypted WebSocket | ws:// protocol used | Audio/transcripts intercepted | Require WSS | High |

---

### P1 Issues (High Priority - Next Sprint)

| ID | Issue | Evidence | Impact | Fix | Confidence |
|----|-------|----------|--------|-----|------------|
| P1-5 | Timestamp drift | Server uses processed samples, not capture time | Transcript timing inaccurate | Send client timestamps | High |
| P1-7 | Network partition handling | Session state lost on reconnect | Streaming interrupted | Implement session resumption | Medium |
| P1-8 | High latency degradation | Queues absorb but timestamps suffer | Poor UX on slow networks | Client-side buffering | Medium |
| P1-9 | Rapid reconnect failures | May hit connection refused | User confusion | Better retry logic | Medium |
| P1-10 | Large backlog handling | Memory pressure from queued audio | Performance degradation | Monitor queue depths | Medium |
| P1-11 | No rate limiting | Unlimited connections/audio | DoS vulnerability | Add connection limits | High |
| P1-12 | Unencrypted transport | Privacy violation risk | Legal/compliance issues | WSS required | High |
| P1-13 | No privacy controls | No user data rights | Compliance risk | Add privacy policy | High |

---

### P2 Issues (Medium Priority - Future Sprints)

| ID | Issue | Evidence | Impact | Fix | Confidence |
|----|-------|----------|--------|-----|------------|
| P2-1 | VAD disabled | Silence processed wastefully | CPU inefficiency | Enable VAD for mic | High |
| P2-2 | Diarization disabled | No speaker labels | Feature incomplete | Per-source diarization | High |
| P2-3 | Analysis snapshot race | Concurrent transcript access | Inconsistent results | Add locking | Medium |
| P2-4 | Frame size inconsistency | Client 20ms, server variable | ASR alignment issues | Standardize framing | Medium |
| P2-5 | No speaker fallback | Uncovered segments unlabeled | Incomplete transcripts | "Unknown Speaker" assignment | High |
| P2-6 | Simple midpoint overlap | Long segments misassigned | Wrong speaker attribution | Weighted overlap | Medium |
| P2-7 | PII in logs | Transcripts in debug logs | Privacy violation | Log redaction | High |
| P2-8 | No dependency scanning | Vulnerable dependencies | Security risks | Regular CVE scanning | High |
| P2-9 | Inconsistent log levels | Mixed debug/info usage | Hard to troubleshoot | Standardize logging | High |
| P2-10 | No structured logging | Plain text logs | Hard to monitor | JSON structured logs | High |
| P2-11 | No metrics | Can't monitor health | Operational blindness | Add key metrics | High |
| P2-12 | No tracing | Hard to debug issues | Poor debuggability | Request tracing | Medium |
| P2-13 | Limited debug hooks | Hard to reproduce issues | Slow incident response | Audio dump capability | Medium |

---

### Execution Backlog (Prioritized)

#### Phase 1: Security & Reliability (Week 1-2)
1. **P0-4**: Implement WebSocket authentication
2. **P0-5**: Configure WSS (WebSocket Secure)
3. **P1-11**: Add connection rate limiting
4. **P1-12**: Ensure WSS deployment
5. **P1-13**: Add basic privacy controls

#### Phase 2: Core UX Improvements (Week 3-4)
1. **P1-5**: Fix timestamp accuracy with client timestamps
2. **P2-1**: Enable VAD for microphone audio
3. **P2-4**: Standardize audio framing
4. **P1-7**: Improve reconnect handling

#### Phase 3: Observability (Week 5-6)
1. **P2-10**: Implement structured logging
2. **P2-11**: Add core metrics (latency, throughput, errors)
3. **P2-13**: Add debug hooks for audio dumping
4. **P2-9**: Standardize log levels

#### Phase 4: Advanced Features (Week 7-8)
1. **P2-2**: Re-enable diarization with per-source approach
2. **P2-5**: Improve speaker assignment algorithm
3. **P2-12**: Add request tracing
4. **P1-8**: Optimize for high-latency networks

#### Phase 5: Testing & Quality (Ongoing)
1. **Testing**: Expand unit test coverage to 80%+
2. **Testing**: Add integration test fixtures
3. **Testing**: Implement load testing harness
4. **Testing**: Add chaos testing capabilities

---

### Risk Assessment

**High-Risk Items:**
- P0-4/P0-5: Security issues could lead to data breaches
- P1-11: DoS vulnerability could take down service
- P1-12: Privacy violations could lead to legal action

**Medium-Risk Items:**
- P1-5: Poor UX from inaccurate timestamps
- P1-7: User frustration from connection issues
- P2-2: Feature incomplete affects product value

**Low-Risk Items:**
- P2-9 through P2-13: Operational improvements, not functional blockers

---

### Success Metrics

**Security & Privacy:**
- ✅ All P0 issues resolved
- ✅ WSS encryption enabled
- ✅ Authentication implemented
- ✅ Privacy controls added

**Reliability:**
- ✅ < 1% error rate in production
- ✅ < 5% reconnect failures
- ✅ < 10 second recovery time

**Performance:**
- ✅ < 3s ASR latency P95
- ✅ < 500MB memory per session
- ✅ 50+ concurrent sessions supported

**Quality:**
- ✅ 80%+ test coverage
- ✅ All chaos tests passing
- ✅ Structured logging implemented

---

*Audit completed: 4 February 2026*
*Next review: 4 March 2026*
