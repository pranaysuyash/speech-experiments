# Model Lab Enhancement Roadmap

**Date**: January 25, 2026  
**Status**: Planning Complete  
**Scope**: 6 Enhancement Areas

---

## Executive Summary

| # | Enhancement | Effort | Priority | Dependencies |
|---|-------------|--------|----------|--------------|
| 2 | Optional/Configurable Steps | M | 🔴 HIGH | None |
| 6 | Model Configuration in UI | M | 🔴 HIGH | None |
| 1 | More Preprocessing Steps | M | 🟡 MEDIUM | None |
| 3 | More Use Cases | S-L | 🟡 MEDIUM | #2, #6 |
| 5 | Error Handling & Retry | M | 🟡 MEDIUM | #2 |
| 4 | Streaming/Real-time Mode | L | 🟢 LOW | #2, #5 |

**Recommended Build Order**: 2 → 6 → 1 → 3 → 5 → 4

---

## Enhancement 1: More Preprocessing Steps

**Effort**: M (1-3 hours)  
**Files to Modify**:
- `harness/preprocess_ops.py` (add operators)
- `harness/media_ingest.py` (channel handling)

**Files to Create**:
- `harness/audio_utils.py` (optional - shared helpers)

### New Operators

| Operator | Description | Optional Dependency |
|----------|-------------|---------------------|
| `normalize_volume` | Peak/RMS normalization | None |
| `extract_channel` | Select channel from stereo | None |
| `speed` | Time-stretch adjustment | `librosa` |
| `denoise` | Background noise reduction | `noisereduce` |
| `resample` | Sample rate conversion | Already exists |

### Implementation Steps

1. **Update operator contract** to support multichannel audio (shape `(n,)` or `(n, c)`)

2. **Add `normalize_volume` operator**:
```python
class NormalizeVolume(Operator):
    name = "normalize_volume"

    def process(self, audio: np.ndarray, sr: int, target_dbfs: float = -1.0, **kwargs):
        peak = float(np.max(np.abs(audio))) + 1e-12
        target_linear = 10 ** (target_dbfs / 20)
        gain = target_linear / peak
        out = (audio * gain).astype(np.float32)
        metrics = {
            "peak_before": 20 * np.log10(peak),
            "peak_after": 20 * np.log10(float(np.max(np.abs(out))) + 1e-12),
            "gain_db": 20 * np.log10(gain),
        }
        return out, sr, metrics
```

3. **Add `extract_channel` operator**:
```python
class ExtractChannel(Operator):
    name = "extract_channel"

    def process(self, audio: np.ndarray, sr: int, channel: int = 0, **kwargs):
        if audio.ndim == 1:
            return audio, sr, {"no_op": True, "reason": "already_mono"}
        c = audio.shape[1]
        if channel < 0 or channel >= c:
            raise ValueError(f"channel out of range: {channel}, available=0..{c-1}")
        out = audio[:, channel].astype(np.float32)
        return out, sr, {"channels_in": c, "channel_selected": channel}
```

4. **Add `denoise` operator** (optional dependency):
```python
class Denoise(Operator):
    name = "denoise"

    def process(self, audio: np.ndarray, sr: int, prop_decrease: float = 1.0, **kwargs):
        try:
            import noisereduce as nr
        except ImportError:
            return audio, sr, {"skipped": True, "reason": "noisereduce_not_installed"}
        out = nr.reduce_noise(y=audio, sr=sr, prop_decrease=prop_decrease)
        return out.astype(np.float32), sr, {"skipped": False}
```

5. **Add `speed` operator**:
```python
class SpeedAdjust(Operator):
    name = "speed"

    def process(self, audio: np.ndarray, sr: int, factor: float = 1.0, **kwargs):
        if abs(factor - 1.0) < 0.01:
            return audio, sr, {"no_op": True}
        try:
            import librosa
            out = librosa.effects.time_stretch(audio, rate=factor)
            return out.astype(np.float32), sr, {"factor": factor}
        except ImportError:
            return audio, sr, {"skipped": True, "reason": "librosa_not_installed"}
```

6. **Register operators** in `OPERATORS` dict

7. **Update `IngestConfig`** to include `operators: list[str]`

---

## Enhancement 2: Optional/Configurable Pipeline Steps

**Effort**: M (1-3 hours)  
**Priority**: 🔴 HIGH (unblocks other features)

**Files to Modify**:
- `harness/session.py` (step selection, dependency expansion)
- `server/api/workbench.py` (expand `PRESETS`)
- `server/api/candidates.py` (reference new presets)
- `client/src/pages/WorkbenchPage.tsx` (preset selection UI)
- `client/src/lib/api.ts` (send `steps_requested`)

### New Presets

Add to `server/api/workbench.py`:

```python
PRESETS = {
    "ingest": {
        "label": "Ingest Only",
        "description": "Audio normalization without processing",
        "steps": ["ingest"]
    },
    "fast_asr_only": {
        "label": "Fast ASR",
        "description": "Quick transcription without diarization or summarization",
        "steps": ["ingest", "asr"]
    },
    "asr_with_diarization": {
        "label": "ASR + Diarization",
        "description": "Transcription with speaker identification",
        "steps": ["ingest", "asr", "diarization", "alignment"]
    },
    "full": {
        "label": "Full Pipeline",
        "description": "Complete pipeline with summarization and action items",
        "steps": None  # All steps
    },
    "diarization_focus": {
        "label": "Diarization Focus",
        "description": "Speaker analysis without LLM steps",
        "steps": ["ingest", "asr", "diarization", "alignment"]
    },
}
```

### Implementation Steps

1. **Add dependency expansion in `SessionRunner`**:
```python
def _expand_steps_with_deps(self, requested: list[str]) -> list[str]:
    """Expand requested steps to include dependencies."""
    unknown = [s for s in requested if s not in self.steps]
    if unknown:
        raise RuntimeError(f"E_UNKNOWN_STEP: {unknown}. Available={list(self.steps.keys())}")

    seen = set()
    order = []

    def visit(name: str):
        if name in seen:
            return
        seen.add(name)
        for dep in self.steps[name].deps:
            visit(dep)
        order.append(name)

    for s in requested:
        visit(s)

    return order
```

2. **Update run execution** to use expanded steps

3. **Add `/api/workbench/steps` endpoint**:
```python
@router.get("/steps")
def get_available_steps() -> list[dict]:
    """Return available pipeline steps with dependencies."""
    return [
        {"name": "ingest", "deps": [], "description": "Audio normalization"},
        {"name": "asr", "deps": ["ingest"], "description": "Transcription"},
        {"name": "diarization", "deps": ["ingest"], "description": "Speaker identification"},
        {"name": "alignment", "deps": ["asr", "diarization"], "description": "Merge ASR + speakers"},
        {"name": "chapters", "deps": ["alignment"], "description": "Topic segmentation"},
        {"name": "summarize_by_speaker", "deps": ["alignment"], "description": "Per-speaker summary"},
        {"name": "action_items_assignee", "deps": ["alignment"], "description": "Extract action items"},
        {"name": "bundle", "deps": ["*"], "description": "Package as Meeting Pack"},
    ]
```

4. **Update WorkbenchPage.tsx** to show preset dropdown:
```tsx
const [preset, setPreset] = useState('full');
const [presets, setPresets] = useState<Preset[]>([]);

useEffect(() => {
  api.getPresets().then(setPresets);
}, []);

// In form:
<label>
  <span>Pipeline</span>
  <select value={preset} onChange={(e) => setPreset(e.target.value)}>
    {presets.map(p => (
      <option key={p.steps_preset} value={p.steps_preset}>
        {p.label}
      </option>
    ))}
  </select>
</label>
```

5. **Pass `steps_preset` in API call**:
```ts
const form = new FormData();
form.append('steps_preset', preset);
// ... rest of form
```

---

## Enhancement 3: More Use Cases in Workbench

**Effort**: S-L (varies by use case)  
**Dependencies**: #2 (presets), #6 (model config)

### New Use Cases

#### 3a. ASR Model Comparison (Effort: S)

Add to `server/api/candidates.py`:

```python
USE_CASES["asr_model_comparison"] = UseCase(
    use_case_id="asr_model_comparison",
    title="ASR Model Comparison",
    description="Compare Whisper vs Faster-Whisper on the same input",
    supported_steps_presets=["fast_asr_only"],
)

CANDIDATES["asr_whisper_base"] = Candidate(
    candidate_id="asr_whisper_base",
    label="Whisper (base)",
    use_case_id="asr_model_comparison",
    steps_preset="fast_asr_only",
    params={"asr": {"engine": "whisper", "model_size": "base"}},
    expected_artifacts=["bundle/transcript.json"],
)

CANDIDATES["asr_faster_whisper_base"] = Candidate(
    candidate_id="asr_faster_whisper_base",
    label="Faster-Whisper (base)",
    use_case_id="asr_model_comparison",
    steps_preset="fast_asr_only",
    params={"asr": {"engine": "faster_whisper", "model_size": "base"}},
    expected_artifacts=["bundle/transcript.json"],
)
```

#### 3b. TTS Quality Test (Effort: L)

Requires new pipeline step + UI changes:

1. **Add `tts` step in `harness/session.py`**:
```python
def tts_func(ctx: SessionContext) -> Dict[str, Any]:
    from harness.tts import run_tts
    prompts = self.extra_config.get("tts", {}).get("prompts", [])
    return run_tts(prompts, ctx.artifacts_dir)

self.steps["tts"] = StepDef(
    name="tts",
    deps=["ingest"],  # or no deps if standalone
    func=tts_func,
    artifact_paths=lambda res: [Path(p["path"]) for p in res.get("artifacts", [])]
)
```

2. **Create `harness/tts.py`** with `run_tts(prompts, output_dir)` using registry

3. **Add use case**:
```python
USE_CASES["tts_quality"] = UseCase(
    use_case_id="tts_quality",
    title="TTS Quality Test",
    description="Generate audio from text prompts and evaluate quality",
    supported_steps_presets=["tts_only"],
)
```

4. **Update UI** to show prompt input when use case is TTS

#### 3c. Real-time Latency Test (Effort: M)

1. **Add latency measurement to ASR step**:
```python
# In asr_func, track timing
metrics = {
    "wall_time_s": elapsed,
    "audio_duration_s": audio_duration,
    "rtf": elapsed / audio_duration,
    "latency_ms": elapsed * 1000,
}
```

2. **Add use case**:
```python
USE_CASES["latency_test"] = UseCase(
    use_case_id="latency_test",
    title="Real-time Latency Test",
    description="Measure RTF and latency for voice assistant use cases",
    supported_steps_presets=["fast_asr_only"],
)
```

---

## Enhancement 4: Streaming/Real-time Mode

**Effort**: L (1-2 days)  
**Dependencies**: #2 (step selection), #5 (error handling)

**Files to Modify**:
- `server/main.py` (add WebSocket router)
- `harness/session.py` (emit progress events)
- `client/src/pages/RunDetailPage.tsx` (subscribe to WS)

**Files to Create**:
- `server/api/ws_runs.py` (WebSocket endpoint)
- `harness/events.py` (event writer)
- `client/src/lib/ws.ts` (WS helper)

### Event Schema

```json
{"ts": "2026-01-25T12:00:01Z", "type": "run_started", "run_id": "xxx"}
{"ts": "2026-01-25T12:00:02Z", "type": "step_started", "step": "asr", "progress": 0.0}
{"ts": "2026-01-25T12:00:15Z", "type": "step_progress", "step": "asr", "progress": 0.5, "message": "50% complete"}
{"ts": "2026-01-25T12:00:30Z", "type": "step_completed", "step": "asr", "progress": 1.0}
{"ts": "2026-01-25T12:01:00Z", "type": "run_completed", "status": "COMPLETED"}
```

### Implementation Steps

1. **Create event writer**:
```python
# harness/events.py
import json
from pathlib import Path
from datetime import datetime

class EventWriter:
    def __init__(self, run_dir: Path):
        self.events_file = run_dir / "events.jsonl"
    
    def emit(self, event_type: str, **payload):
        event = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "type": event_type,
            **payload
        }
        with self.events_file.open("a") as f:
            f.write(json.dumps(event) + "\n")
```

2. **Hook into SessionRunner**:
```python
self.events = EventWriter(self.session_dir)
ctx.on_progress = lambda step, pct, msg: self.events.emit(
    "step_progress", step=step, progress=pct, message=msg
)
```

3. **Add WebSocket endpoint**:
```python
# server/api/ws_runs.py
from fastapi import WebSocket
import asyncio

@router.websocket("/api/runs/{run_id}/ws")
async def run_websocket(websocket: WebSocket, run_id: str):
    await websocket.accept()
    events_file = get_run_dir(run_id) / "events.jsonl"
    
    # Stream existing events
    if events_file.exists():
        for line in events_file.open():
            await websocket.send_text(line)
    
    # Tail for new events
    last_pos = events_file.stat().st_size if events_file.exists() else 0
    while True:
        await asyncio.sleep(0.3)
        if events_file.exists():
            with events_file.open() as f:
                f.seek(last_pos)
                new_lines = f.read()
                if new_lines:
                    for line in new_lines.strip().split("\n"):
                        await websocket.send_text(line)
                    last_pos = f.tell()
```

4. **Update frontend** to connect and update UI live

---

## Enhancement 5: Better Error Handling & Retry

**Effort**: M (1-3 hours)  
**Dependencies**: #2 (step selection)

**Files to Modify**:
- `harness/session.py` (wrap step execution, retry logic)
- `server/api/runs.py` (expose errors in status)
- `client/src/components/RunDetail.tsx` (display errors)

**Files to Create**:
- `harness/errors.py` (structured error types)

### Error Schema

```python
# harness/errors.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class StepError:
    code: str
    message: str
    step: str
    traceback_path: Optional[str] = None
    recoverable: bool = False

# Error codes
E_MODEL_NOT_FOUND = "E_MODEL_NOT_FOUND"
E_MODEL_OOM = "E_MODEL_OOM"
E_AUDIO_CORRUPT = "E_AUDIO_CORRUPT"
E_STEP_TIMEOUT = "E_STEP_TIMEOUT"
E_DEPENDENCY_FAILED = "E_DEPENDENCY_FAILED"
E_UNKNOWN = "E_UNKNOWN"
```

### Manifest Schema Update

```json
{
  "steps": {
    "asr": {
      "status": "FAILED",
      "attempts": 2,
      "error_code": "E_MODEL_OOM",
      "error_message": "CUDA out of memory while loading model",
      "traceback_path": "logs/asr_trace.txt",
      "started_at": "2026-01-25T12:00:00Z",
      "ended_at": "2026-01-25T12:00:30Z"
    }
  },
  "status": "FAILED",
  "failure_step": "asr",
  "error_code": "E_MODEL_OOM",
  "error_message": "CUDA out of memory while loading model"
}
```

### Implementation Steps

1. **Add retry wrapper**:
```python
def run_step_with_retry(self, step_name: str, max_attempts: int = 1):
    step_def = self.steps[step_name]
    attempts = 0
    last_error = None
    
    while attempts < max_attempts:
        attempts += 1
        try:
            result = step_def.func(self.ctx)
            return result
        except Exception as e:
            last_error = e
            self._record_step_error(step_name, e, attempts)
            if not self._is_retryable(e):
                break
    
    raise last_error
```

2. **Add timeout support**:
```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Step exceeded time limit")

def run_step_with_timeout(self, step_name: str, timeout_s: int = 300):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_s)
    try:
        return self.run_step_with_retry(step_name)
    finally:
        signal.alarm(0)
```

3. **Update UI** to show per-step errors and retry count

---

## Enhancement 6: Model Configuration in UI

**Effort**: M (1-3 hours)  
**Priority**: 🔴 HIGH

**Files to Modify**:
- `client/src/pages/WorkbenchPage.tsx` (add form controls)
- `client/src/lib/api.ts` (send config JSON)
- `harness/asr.py` (honor engine/model_size/language)
- `harness/registry.py` (pass config to loaders)

### Config Schema

```json
{
  "asr": {
    "engine": "whisper",       // whisper | faster_whisper | lfm2_5_audio
    "model_size": "base",      // tiny | base | small | medium | large
    "language": "en"           // ISO code or "auto"
  },
  "diarization": {
    "model": "pyannote",       // pyannote | heuristic
    "max_speakers": 10
  },
  "device_preference": ["mps", "cpu"]  // Ordered preference
}
```

### Implementation Steps

1. **Add form controls in WorkbenchPage.tsx**:
```tsx
const [config, setConfig] = useState({
  asr: { engine: 'whisper', model_size: 'base', language: 'en' },
  device_preference: ['mps', 'cpu']
});

// In form:
<label>
  <span>ASR Model Size</span>
  <select 
    value={config.asr.model_size}
    onChange={(e) => setConfig({
      ...config,
      asr: { ...config.asr, model_size: e.target.value }
    })}
  >
    <option value="tiny">Tiny (fastest)</option>
    <option value="base">Base</option>
    <option value="small">Small</option>
    <option value="medium">Medium</option>
    <option value="large">Large (most accurate)</option>
  </select>
</label>

<label>
  <span>Language</span>
  <select 
    value={config.asr.language}
    onChange={(e) => setConfig({
      ...config,
      asr: { ...config.asr, language: e.target.value }
    })}
  >
    <option value="en">English</option>
    <option value="auto">Auto-detect</option>
    <option value="es">Spanish</option>
    <option value="fr">French</option>
    <option value="de">German</option>
    <option value="zh">Chinese</option>
    <option value="ja">Japanese</option>
  </select>
</label>

<label>
  <span>Device</span>
  <select 
    value={config.device_preference[0]}
    onChange={(e) => setConfig({
      ...config,
      device_preference: [e.target.value, 'cpu']
    })}
  >
    <option value="mps">Apple Silicon (MPS)</option>
    <option value="cuda">NVIDIA GPU (CUDA)</option>
    <option value="cpu">CPU</option>
  </select>
</label>
```

2. **Send config in API call**:
```ts
const form = new FormData();
form.append('config', JSON.stringify(config));
```

3. **Update `harness/asr.py`** to read config:
```python
def run_asr(audio_path, output_dir, config=None, **kwargs):
    config = config or {}
    engine = config.get("engine", "whisper")
    model_size = config.get("model_size", "base")
    language = config.get("language", "en")
    
    # Select model based on engine
    if engine == "faster_whisper":
        model_name = f"guillaumekln/faster-whisper-{model_size}"
    else:
        model_name = f"openai/whisper-{model_size}"
    
    # Load and run...
```

4. **Merge config with candidate params** in `server/api/experiments.py`:
```python
def deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result

final_config = deep_merge(candidate.params, ui_config)
```

---

## Implementation Timeline

### Phase 1: Foundation (Week 1)
- [ ] **#2**: Optional/Configurable Steps + Presets
- [ ] **#6**: Model Configuration in UI

### Phase 2: Operators & Use Cases (Week 2)
- [ ] **#1**: More Preprocessing Steps
- [ ] **#3a**: ASR Model Comparison use case

### Phase 3: Stability (Week 3)
- [ ] **#5**: Error Handling & Retry
- [ ] **#3b**: TTS Quality Test use case (if needed)

### Phase 4: Real-time (Week 4+)
- [ ] **#4**: WebSocket Streaming
- [ ] **#3c**: Latency Test use case

---

## Quick Win Checklist

If limited time, do these first:

- [ ] Add `fast_asr_only` preset (30 min)
- [ ] Add model size dropdown to WorkbenchPage (1 hr)
- [ ] Add ASR Model Comparison use case (1 hr)
- [ ] Add `normalize_volume` operator (30 min)

**Total for quick wins: ~3 hours**
