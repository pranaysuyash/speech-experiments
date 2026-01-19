# Backend Logging Cleanup

## Problem

Backend server logs were cluttered with too many INFO-level messages:
- HTTP access logs for every request
- FFmpeg command-line echoing
- Audio conversion details (mono, resample, save)
- Cache hits
- Timer measurements
- Index refresh counts

This made it difficult to spot errors and important step transitions.

## Solution

Configured logging levels to reduce noise while preserving critical information:

### 1. Server Configuration

**File:** [server/main.py](file:///Users/pranay/Projects/speech_experiments/model-lab/server/main.py)

- Set uvicorn `access_log=False` to disable HTTP request logging
- Configured `uvicorn.access` logger to `WARNING` level
- Configured `multipart` logger to `WARNING` level
- Added consistent log format: `%(levelname)s [%(name)s] %(message)s`

### 2. Demoted Verbose Operations to DEBUG

#### Index Operations
**File:** [server/services/runs_index.py](file:///Users/pranay/Projects/speech_experiments/model-lab/server/services/runs_index.py)
- `INFO → DEBUG`: "Reloading transcript for {run_id}"
- `INFO → DEBUG`: "Indexed {n} runs."

#### Media Processing
**File:** [harness/media_ingest.py](file:///Users/pranay/Projects/speech_experiments/model-lab/harness/media_ingest.py)
- `INFO → DEBUG`: FFmpeg command invocations

#### Audio I/O
**File:** [harness/audio_io.py](file:///Users/pranay/Projects/speech_experiments/model-lab/harness/audio_io.py)
- `INFO → DEBUG`: "Converted to mono"
- `INFO → DEBUG`: "Resampled to {sr}Hz"
- `INFO → DEBUG`: "Loaded {file}: {duration}s"
- `INFO → DEBUG`: "Saved audio to {path}"

#### LLM Operations
**File:** [harness/llm_provider.py](file:///Users/pranay/Projects/speech_experiments/model-lab/harness/llm_provider.py)
- `INFO → DEBUG`: "Cache hit for {key}"

#### Performance Measurements
**File:** [harness/timers.py](file:///Users/pranay/Projects/speech_experiments/model-lab/harness/timers.py)
- `INFO → DEBUG`: "⏱️ {operation}: {ms}ms"

### 3. What Stays at INFO Level

**Preserved for visibility:**
- Step transitions: "Running Step: {name}"
- Errors and warnings
- Run start/complete events
- Model loading notifications
- Pipeline completion messages
- Gate decisions and checks

## Result

**Before:**
```
INFO [uvicorn.access] "GET /api/runs/abc123/status HTTP/1.1" 200
INFO [server.services.runs_index] Indexed 47 runs.
INFO [harness.audio_io] Loaded meeting.wav: 125.3s @ 16000Hz
INFO [harness.audio_io] Converted to mono: meeting.wav
INFO [harness.timers] ⏱️  load_audio: 234.5ms
INFO [harness.media_ingest] Running ffmpeg: ffmpeg -i input.mp4 -ar 16000...
INFO [harness.llm_provider] Cache hit for a1b2c3d4e5f6
...
```

**After:**
```
INFO [session] Running Step: asr
INFO [harness.asr] Loading faster_whisper on cpu
ERROR [session] Step failed: asr - ModelNotFoundError: Model not found
```

Clean, actionable logs that highlight what matters.

## To Enable Debug Logs

If you need verbose logging for debugging:

```bash
export LOG_LEVEL=DEBUG
python server/main.py
```

Or in code:
```python
logging.basicConfig(level=logging.DEBUG)
```
