# Model Lab Infrastructure Status

**Date**: January 25, 2026  
**Purpose**: Audit of current capabilities for public app readiness

---

## Executive Summary

| Category | Status | Ready for Public? |
|----------|--------|-------------------|
| **ASR Pipeline** | ✅ Production | Yes (batch) |
| **TTS Pipeline** | ✅ Working | Yes (limited to LFM2.5) |
| **Diarization Pipeline** | ✅ Working | Yes |
| **VAD Pipeline** | ✅ Working | Yes |
| **Frontend UI** | ⚠️ Internal | Needs polish |
| **Backend API** | ⚠️ Single-user | Needs multi-tenant |
| **Auth/Multi-user** | ❌ Missing | Required for public |
| **Job Queue** | ❌ Missing | Required for public |

---

## 1. Models Ready to Test

### Production-Ready (Status: PRODUCTION)

| Model | Capabilities | Hardware | CLI Script |
|-------|-------------|----------|------------|
| **whisper** | ASR | CPU, MPS, CUDA | `run_asr.py --model whisper` |
| **faster_whisper** | ASR | CPU, CUDA | `run_asr.py --model faster_whisper` |
| **pyannote_diarization** | Diarization | CPU, CUDA | `run_diarization.py --model pyannote_diarization` |
| **silero_vad** | VAD | CPU, MPS, CUDA | `run_vad.py --model silero_vad` |

### Candidate (Status: CANDIDATE)

| Model | Capabilities | Hardware | Notes |
|-------|-------------|----------|-------|
| **lfm2_5_audio** | ASR, TTS, Chat | CPU, MPS | MPS workaround applied (see TTS_MPS_WORKAROUND.md) |

### Experimental (Status: EXPERIMENTAL)

| Model | Capabilities | Hardware | Notes |
|-------|-------------|----------|-------|
| **seamlessm4t** | ASR, MT | CPU, MPS, CUDA | Multi-lingual translation |
| **distil_whisper** | ASR | CPU, MPS, CUDA | 6x faster Whisper |
| **heuristic_diarization** | Diarization | CPU, MPS, CUDA | Silero VAD + 1-speaker assumption |
| **whisper_cpp** | ASR | CPU | Edge-friendly C++ backend |

---

## 2. End-to-End Pipelines

### ✅ ASR Pipeline (Fully Working)

```bash
# Dataset mode (with ground truth → WER/CER computed)
uv run python -m scripts.run_asr --model faster_whisper --dataset primary

# Adhoc mode (single file, video supported)
uv run python -m scripts.run_asr --model whisper --audio path/to/file.mp4

# With preprocessing
uv run python -m scripts.run_asr --model whisper --audio file.wav --pre trim_silence,normalize_loudness
```

**Outputs**:
- `runs/{model}/asr/{timestamp}.json` - Full artifact with provenance
- `runs/{model}/asr/summary.json` - Latest summary

**Metrics Computed**:
- WER, CER (when ground truth available)
- RTF (real-time factor)
- Latency, word count, segment count
- Sanity gates (length ratio, WPS)

---

### ✅ TTS Pipeline (Working)

```bash
uv run python -m scripts.run_tts --model lfm2_5_audio --dataset tts_smoke_v1
```

**Supported Models**: Only `lfm2_5_audio` currently has TTS capability

**Outputs**:
- `runs/{model}/tts/{timestamp}.json` - Full artifact
- `runs/{model}/tts/audio/*.wav` - Generated audio files

**Metrics/Gates**:
- RTF, latency
- Audio health gates (clipping, silence, noise)

---

### ✅ Diarization Pipeline (Working)

```bash
# Dataset mode
uv run python -m scripts.run_diarization --model pyannote_diarization --dataset diar_smoke_v1

# Adhoc mode (video supported)
uv run python -m scripts.run_diarization --model pyannote_diarization --audio meeting.mp4
```

**Metrics Computed**:
- Speaker count (pred vs. auth)
- DER proxy (when reference available)
- RTF, segment count

---

### ✅ VAD Pipeline (Working)

```bash
uv run python -m scripts.run_vad --model silero_vad --dataset vad_smoke_v1
```

---

### ⚠️ Alignment Pipeline (Partial)

```bash
uv run python -m scripts.run_alignment --model whisper --audio file.wav
```

Forces word-level timestamps. Works but less polished.

---

### ⚠️ V2V Pipeline (Experimental)

Voice-to-voice conversion. Dataset exists (`v2v_smoke_v1.yaml`) but runner is experimental.

---

## 3. Available Datasets

### Golden Datasets (With Ground Truth)

| Dataset ID | Task | Audio Files | Ground Truth |
|------------|------|-------------|--------------|
| `asr_golden_v1` | ASR | 3 cases (llm, numbers, noisy) | ✅ Yes |
| `primary` / `llm_primary` | ASR | llm_recording_pranay.wav (163s) | ✅ llm.txt |
| `ux_primary` | ASR | ux_psychology_30s.wav | ✅ Yes |

### Smoke Datasets (Health Gates Only)

| Dataset ID | Task | Cases |
|------------|------|-------|
| `asr_smoke_v1` | ASR | 1 case |
| `tts_smoke_v1` | TTS | 3 prompts |
| `diar_smoke_v1` | Diarization | 2 cases |
| `diar_smoke_v2` | Diarization | TBD |
| `vad_smoke_v1` | VAD | 2 cases |
| `vad_smoke_v2` | VAD | TBD |
| `v2v_smoke_v1` | V2V | Experimental |

---

## 4. Frontend Pages

| Page | Route | Status | Purpose |
|------|-------|--------|---------|
| Runs | `/lab/runs` | ✅ Working | Browse all run artifacts |
| Run Detail | `/runs/:runId` | ✅ Working | View single run details |
| Workbench | `/lab/workbench` | ⚠️ Partial | Manual run execution |
| Experiments | `/lab/experiments` | ⚠️ Partial | Group runs into experiments |
| Candidates | `/lab/candidates` | ⚠️ Partial | Model promotion workflow |
| Results | `/lab/results` | ⚠️ Partial | Aggregated results view |
| Findings | `/lab/findings` | ⚠️ Partial | Insights & recommendations |

---

## 5. Backend API Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/runs` | GET | List all runs |
| `/api/runs/:id` | GET | Get run details |
| `/api/results` | GET | Query results |
| `/api/workbench/*` | * | Manual run execution |
| `/api/experiments/*` | * | Experiment CRUD |
| `/api/candidates/*` | * | Candidate promotion |
| `/api/lifecycle/*` | * | Run lifecycle management |
| `/health` | GET | Health check |

---

## 6. Use Cases Testable TODAY

### Tier 1: Fully Testable (E2E working)

| Use Case | Models | Command |
|----------|--------|---------|
| **Batch ASR Evaluation** | whisper, faster_whisper, lfm2_5_audio, distil_whisper | `run_asr.py --model X --dataset primary` |
| **Adhoc Transcription** | All ASR models | `run_asr.py --model X --audio file.wav` |
| **Video Transcription** | All ASR models | `run_asr.py --model X --input video.mp4` |
| **TTS Generation** | lfm2_5_audio | `run_tts.py --model lfm2_5_audio` |
| **Speaker Diarization** | pyannote_diarization, heuristic_diarization | `run_diarization.py --model X` |
| **Voice Activity Detection** | silero_vad | `run_vad.py --model silero_vad` |

### Tier 2: Partially Testable

| Use Case | Gap |
|----------|-----|
| **Multi-lingual ASR** | seamlessm4t works but no golden datasets for non-English |
| **Speech Translation** | seamlessm4t has MT capability, needs runner script |
| **Real-time Streaming** | No streaming runner (batch only) |

### Tier 3: Not Yet Testable

| Use Case | Missing |
|----------|---------|
| **Browser-Only Demo** | No WebGPU/WASM models |
| **Mobile Voice Notes** | whisper_cpp registered but no mobile runner |
| **Voice Cloning** | No model registered |

---

## 7. Gaps for Public App

### Critical (Blockers)

| Gap | Effort | Notes |
|-----|--------|-------|
| **Authentication** | L | OAuth/magic link + API tokens |
| **Multi-tenancy** | L | Org/project/user isolation |
| **Async Job Queue** | L | Redis + RQ/Celery for long-running runs |
| **Object Storage** | M | S3/GCS for audio + artifacts |
| **Rate Limiting** | M | Per-user quotas, upload limits |

### Important (Launch Quality)

| Gap | Effort | Notes |
|-----|--------|-------|
| **Evidence-backed Recommendations** | M | Fix "0 score" in use_cases.md |
| **Shareable Run Permalinks** | S | Currently internal paths |
| **Preloaded Demo Datasets** | S | Quick "try it" experience |
| **Polished Landing → Run → Compare Flow** | M | UX streamlining |

### Nice-to-Have

| Gap | Effort | Notes |
|-----|--------|-------|
| **Streaming ASR Runner** | L | WebSocket-based partial results |
| **Human Eval Loop (MOS)** | M | TTS quality ratings |
| **Regression Tracking** | M | Scheduled runs, alerts |

---

## 8. Quick Verification Commands

```bash
# 1. Verify backend health
curl http://localhost:8000/health

# 2. Run smoke ASR test
uv run python -m scripts.run_asr --model whisper --dataset asr_smoke_v1

# 3. Run production ASR test (with metrics)
uv run python -m scripts.run_asr --model faster_whisper --dataset primary

# 4. Run TTS smoke
uv run python -m scripts.run_tts --model lfm2_5_audio --dataset tts_smoke_v1

# 5. Run diarization
uv run python -m scripts.run_diarization --model pyannote_diarization --dataset diar_smoke_v1

# 6. Start dev servers
./dev.sh  # or: make dev
```

---

## 9. Recommended Next Steps (Prioritized)

1. **Fix evidence backing** in `docs/use_cases.md` generation (1-2 hrs)
2. **Add simple auth** (OAuth skeleton) for API (1 day)
3. **Add Redis job queue** for async runs (1-2 days)
4. **Create "Quick Demo" landing page** with preloaded samples (1 day)
5. **Add shareable permalinks** for run results (0.5 day)

---

## Appendix: File Structure Reference

```
model-lab/
├── harness/           # Core testing infrastructure
│   ├── registry.py    # Model loaders (Bundle Contract v1)
│   ├── audio_io.py    # Audio loading
│   ├── metrics_*.py   # ASR, TTS, diarization metrics
│   └── contracts.py   # Bundle validation
├── scripts/           # CLI runners
│   ├── run_asr.py     # ASR runner
│   ├── run_tts.py     # TTS runner
│   └── run_diarization.py
├── server/            # FastAPI backend
│   ├── main.py        # Entry point
│   └── api/           # Route handlers
├── client/            # React frontend
│   └── src/pages/     # UI pages
├── data/
│   ├── golden/        # Test datasets (YAML)
│   ├── audio/         # Audio files
│   └── text/          # Ground truth
├── runs/              # Run artifacts (JSON)
└── models/            # Per-model configs
```
