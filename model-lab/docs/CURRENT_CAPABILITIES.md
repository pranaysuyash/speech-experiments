# Model Lab - Current Capabilities

**Date**: January 25, 2026  
**Status**: Internal Testing Ready  
**Version**: 0.1.0

---

## Overview

Model Lab is a model testing and evaluation framework with:
- **CLI runners** for batch evaluation (ASR, TTS, Diarization, VAD)
- **Web UI** for interactive testing (Workbench, Experiments, Results)
- **Harness** infrastructure for metrics, provenance, and reproducibility

---

## 1. Registered Models

### Production Status

| Model | Type | Capabilities | Hardware | Status |
|-------|------|--------------|----------|--------|
| **whisper** | OpenAI Whisper | ASR | CPU, MPS, CUDA | ✅ PRODUCTION |
| **faster_whisper** | CTranslate2 | ASR | CPU, CUDA | ✅ PRODUCTION |
| **pyannote_diarization** | Pyannote 3.1 | Diarization | CPU, CUDA | ✅ PRODUCTION |
| **silero_vad** | Silero VAD 4.0 | VAD | CPU, MPS, CUDA | ✅ PRODUCTION |

### Candidate Status

| Model | Type | Capabilities | Hardware | Status | Notes |
|-------|------|--------------|----------|--------|-------|
| **lfm2_5_audio** | LiquidAI | ASR, TTS, Chat | CPU, MPS | ⚠️ CANDIDATE | MPS workaround applied |

### Experimental Status

| Model | Type | Capabilities | Hardware | Status |
|-------|------|--------------|----------|--------|
| **seamlessm4t** | Meta | ASR, MT | CPU, MPS, CUDA | 🔬 EXPERIMENTAL |
| **distil_whisper** | HuggingFace | ASR | CPU, MPS, CUDA | 🔬 EXPERIMENTAL |
| **heuristic_diarization** | Silero+Heuristic | Diarization | CPU, MPS, CUDA | 🔬 EXPERIMENTAL |
| **whisper_cpp** | C++ Backend | ASR | CPU | 🔬 EXPERIMENTAL |

---

## 2. CLI Runners

### ASR Runner (`scripts/run_asr.py`)

**Modes:**
- **Dataset mode**: Evaluate against golden datasets with ground truth
- **Adhoc mode**: Process single audio/video file

```bash
# Dataset mode (with WER/CER metrics)
uv run python -m scripts.run_asr --model faster_whisper --dataset primary

# Adhoc mode (single file)
uv run python -m scripts.run_asr --model whisper --audio path/to/file.wav

# Video input (auto-extracts audio)
uv run python -m scripts.run_asr --model whisper --input meeting.mp4

# With preprocessing
uv run python -m scripts.run_asr --model whisper --audio file.wav --pre trim_silence,normalize_loudness
```

**Outputs:**
- `runs/{model}/asr/{timestamp}.json` - Full artifact with provenance
- `runs/{model}/asr/summary.json` - Latest run summary

**Metrics Computed:**
| Metric | Description | Requires Ground Truth |
|--------|-------------|----------------------|
| WER | Word Error Rate | ✅ Yes |
| CER | Character Error Rate | ✅ Yes |
| RTF | Real-Time Factor (latency/duration) | No |
| Latency (ms) | Inference time | No |
| Word Count | Output words | No |
| Segment Count | Number of segments | No |

**Sanity Gates:**
- Length ratio check (hyp vs ref)
- Words-per-second bounds
- Missing text detection

---

### TTS Runner (`scripts/run_tts.py`)

**Supported Models:** `lfm2_5_audio` only

```bash
uv run python -m scripts.run_tts --model lfm2_5_audio --dataset tts_smoke_v1
```

**Outputs:**
- `runs/{model}/tts/{timestamp}.json` - Run artifact
- `runs/{model}/tts/audio/*.wav` - Generated audio files
- `runs/{model}/tts/summary.json` - Latest summary

**Audio Health Gates:**
| Gate | Description |
|------|-------------|
| `is_clipped` | Audio exceeds amplitude bounds |
| `is_silent` | No audio energy detected |
| `is_too_short` | Duration below threshold |
| `is_too_long` | Duration above threshold |
| `has_failure` | Any gate failed |

---

### Diarization Runner (`scripts/run_diarization.py`)

```bash
# Dataset mode
uv run python -m scripts.run_diarization --model pyannote_diarization --dataset diar_smoke_v1

# Adhoc mode
uv run python -m scripts.run_diarization --model heuristic_diarization --audio meeting.wav

# With preprocessing
uv run python -m scripts.run_diarization --model pyannote_diarization --input call.mp4 --pre trim_silence
```

**Metrics:**
| Metric | Description |
|--------|-------------|
| `num_speakers_pred` | Detected speaker count |
| `num_speakers_auth` | Ground truth speaker count |
| `speaker_count_error` | |pred - auth| |
| `der_proxy` | DER approximation |
| RTF | Real-time factor |

---

### VAD Runner (`scripts/run_vad.py`)

```bash
uv run python -m scripts.run_vad --model silero_vad --dataset vad_smoke_v1
```

---

### Other Runners

| Runner | Script | Status |
|--------|--------|--------|
| Alignment | `run_alignment.py` | ⚠️ Partial |
| Chapters | `run_chapters.py` | ⚠️ Partial |
| V2V | `run_v2v.py` | 🔬 Experimental |
| Meeting Session | `run_meeting.py` | ✅ Working |
| NER | `run_ner.py` | ⚠️ Partial |
| Summarize | `run_summarize.py` | ✅ Working (needs LLM) |

---

## 3. Web UI

### Pages

| Page | Route | Purpose | Status |
|------|-------|---------|--------|
| **Runs** | `/lab/runs` | Browse all run artifacts | ✅ Working |
| **Run Detail** | `/runs/:runId` | View single run with status, artifacts, transcript | ✅ Working |
| **Workbench** | `/lab/workbench` | Upload file → select use case → run | ✅ Working |
| **Experiments** | `/lab/experiments` | Group runs for comparison | ⚠️ Partial |
| **Candidates** | `/lab/candidates` | Model promotion workflow | ⚠️ Partial |
| **Results** | `/lab/results` | Aggregated results view | ⚠️ Partial |
| **Findings** | `/lab/findings` | Insights & recommendations | ⚠️ Partial |

### Workbench Flow

```
1. Select Use Case (Meeting Smoke, ASR Smoke, Diarization Smoke)
2. Select Mode (Single Run / Compare)
3. Upload File (audio or video)
4. Select Candidate(s) (Fast Ingest, Full Pipeline, etc.)
5. Click "Start Run"
6. → Redirects to Run Detail page with live status
```

### Use Cases & Candidates

| Use Case | Candidates | Steps |
|----------|------------|-------|
| **Meeting Smoke** | Fast Ingest | `ingest` only |
| | Full Pipeline | All steps |
| **ASR Smoke** | Ingest Only | `ingest` only |
| | Full ASR | All steps |
| **Diarization Smoke** | Full Diarization | All steps |

---

## 4. Session Pipeline Steps

The full pipeline (`steps=None`) runs these in order:

| Step | Dependencies | What It Does | Output |
|------|--------------|--------------|--------|
| **ingest** | - | Normalize audio (ffmpeg) | `artifacts/audio_normalized.wav` |
| **asr** | ingest | Transcription | `artifacts/asr.json` |
| **diarization** | ingest | Speaker identification | `artifacts/diarization.json` |
| **alignment** | asr, diarization | Merge ASR + speakers | `artifacts/alignment.json` |
| **chapters** | alignment | Topic segmentation | `artifacts/chapters.json` |
| **summarize_by_speaker** | alignment | Per-speaker summary | `artifacts/summary_by_speaker.json` |
| **action_items_assignee** | alignment | Extract action items | `artifacts/action_items.csv` |
| **bundle** | all | Package as Meeting Pack | `bundle/*` |

### Meeting Pack Bundle Contents

```
bundle/
├── bundle_manifest.json   # Index of all artifacts
├── transcript.json        # Canonical transcript
├── transcript.txt         # Plain text version
├── summary.md             # Meeting summary
├── action_items.csv       # Action items with assignees
├── decisions.md           # Key decisions
└── audio_normalized.wav   # Processed audio
```

---

## 5. Datasets

### Golden Datasets (With Ground Truth)

| Dataset ID | Task | Audio | Duration | Ground Truth |
|------------|------|-------|----------|--------------|
| `asr_golden_v1` | ASR | 3 cases | ~4 min total | ✅ Yes |
| `primary` / `llm_primary` | ASR | llm_recording_pranay.wav | 163s | ✅ llm.txt |
| `ux_primary` | ASR | ux_psychology_30s.wav | 30s | ✅ Yes |

### Smoke Datasets (Health Gates Only)

| Dataset ID | Task | Cases |
|------------|------|-------|
| `asr_smoke_v1` | ASR | 1 |
| `tts_smoke_v1` | TTS | 3 prompts |
| `diar_smoke_v1` | Diarization | 2 |
| `diar_smoke_v2` | Diarization | TBD |
| `vad_smoke_v1` | VAD | 2 |
| `vad_smoke_v2` | VAD | TBD |

### Dataset Locations

```
data/
├── golden/              # Dataset definitions (YAML)
│   ├── asr_golden_v1.yaml
│   ├── tts_smoke_v1.yaml
│   └── ...
├── audio/
│   ├── PRIMARY/         # Golden audio files
│   └── SMOKE/           # Smoke test audio
├── text/
│   └── PRIMARY/         # Ground truth transcripts
└── truth/               # Additional ground truth
```

---

## 6. Preprocessing Operators

Available via `--pre` flag:

| Operator | Description |
|----------|-------------|
| `trim_silence` | Remove leading/trailing silence |
| `normalize_loudness` | Apply loudness normalization |

```bash
uv run python -m scripts.run_asr --model whisper --audio file.wav --pre trim_silence,normalize_loudness
```

---

## 7. API Endpoints

### Runs API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/runs` | GET | List all runs |
| `/api/runs/{runId}` | GET | Get run details |
| `/api/runs/{runId}/status` | GET | Get run status with steps |
| `/api/runs/{runId}/transcript` | GET | Get transcript |
| `/api/runs/{runId}/audio` | GET | Stream audio |
| `/api/runs/{runId}/bundle.zip` | GET | Download meeting pack |
| `/api/runs/{runId}/kill` | POST | Kill running job |
| `/api/runs/{runId}/retry` | POST | Retry failed run |

### Workbench API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/workbench/runs` | POST | Create run from upload |
| `/api/workbench/presets` | GET | List step presets |

### Experiments API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/experiments` | POST | Create experiment |
| `/api/experiments/{id}` | GET | Get experiment |
| `/api/experiments/{id}/runs/start` | POST | Start next run |
| `/api/experiments/{id}/runs/start-all` | POST | Start all runs |
| `/api/experiments/{id}/compare-results` | GET | Get comparison |

### Candidates API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/use-cases` | GET | List use cases |
| `/api/use-cases/{id}/candidates` | GET | List candidates for use case |
| `/api/candidates/{id}` | GET | Get candidate details |

---

## 8. Known Issues & Workarounds

### LFM2.5-Audio MPS CUDA Error

**Problem:** `liquid-audio` library hardcodes `.cuda()` calls  
**Fix:** Dynamic detokenizer injection in `harness/registry.py`  
**Doc:** [TTS_MPS_WORKAROUND.md](TTS_MPS_WORKAROUND.md)

### Faster-Whisper MPS Not Supported

**Problem:** CTranslate2 doesn't support MPS  
**Workaround:** Falls back to CPU on Apple Silicon  

### Pyannote Requires HF Token

**Problem:** `pyannote/speaker-diarization-3.1` requires authentication  
**Fix:** Set `HF_TOKEN` environment variable  

---

## 9. Development Commands

```bash
# Start dev servers (backend + frontend)
./dev.sh

# Or manually:
uv run python -m server.main              # Backend on :8000
cd client && npm run dev                  # Frontend on :5173

# Run tests
uv run pytest -m "not real_e2e"

# Type check
cd client && npm run build

# Quick smoke test
uv run python -m scripts.run_asr --model whisper --dataset asr_smoke_v1
```

---

## 10. File Structure

```
model-lab/
├── harness/                 # Core testing infrastructure
│   ├── registry.py          # Model loaders (Bundle Contract v1)
│   ├── session.py           # SessionRunner (pipeline orchestration)
│   ├── audio_io.py          # Audio loading utilities
│   ├── asr.py               # ASR step implementation
│   ├── diarization.py       # Diarization step
│   ├── alignment.py         # Alignment step
│   ├── nlp.py               # Summarization, action items (LLM)
│   ├── meeting_pack.py      # Bundle generation
│   ├── metrics_*.py         # Metrics (ASR, TTS, diarization, VAD)
│   ├── contracts.py         # Bundle validation
│   └── media_ingest.py      # Audio/video ingestion
├── scripts/                 # CLI runners
│   ├── run_asr.py
│   ├── run_tts.py
│   ├── run_diarization.py
│   └── run_vad.py
├── server/                  # FastAPI backend
│   ├── main.py
│   └── api/
│       ├── runs.py
│       ├── workbench.py
│       ├── experiments.py
│       └── candidates.py
├── client/                  # React frontend
│   └── src/
│       ├── pages/
│       │   ├── WorkbenchPage.tsx
│       │   ├── ExperimentPage.tsx
│       │   └── ...
│       └── lib/api.ts
├── data/
│   ├── golden/              # Dataset definitions
│   ├── audio/               # Audio files
│   └── text/                # Ground truth
├── runs/                    # Run artifacts (JSON)
├── models/                  # Per-model configs
│   ├── whisper/config.yaml
│   ├── faster_whisper/config.yaml
│   └── lfm2_5_audio/config.yaml
└── docs/                    # Documentation
```
