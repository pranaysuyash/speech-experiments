# Model Lab Diagnostic Report — 2026-02-13

## TL;DR

The architecture is solid. The app boots, the pipeline runs, the UI is wired up.
But **63% of all runs fail** (86/136), and the HF Sprint ran today with **55 failures out of 142 tasks**.
The root causes are fixable: missing dependencies, missing HF auth, and two code bugs in the `alignment` step.

**HF Pro expires March 1st.** We have ~16 days to actually use it.

---

## 1. Current State

### What Works ✅
- Server boots clean (FastAPI on :8000, Vite on :5173)
- All 26 model loaders are **registered** (every sprint model has a loader)
- Core models load and transcribe: `whisper`, `faster_whisper`, `distil_whisper`, `silero_vad`, `glm_asr_nano_2512`, `yamnet`, `nb_whisper_small_onnx`, `heuristic_diarization`
- Workbench upload → run pipeline works for `ingest` and `fast_asr_only` presets
- 38 completed runs exist with full artifacts
- UI has Runs, Workbench, Experiments, Candidates, Results, Findings pages

### What Fails ❌

| Failure Category | Count | Root Cause |
|---|---:|---|
| `alignment` step: ASR artifact not found | 29 | Step looks for file at wrong path or ASR step was skipped |
| `alignment` step: `PosixPath.get()` | 4 | Code bug — passing Path where dict expected |
| `alignment` step: diarization missing | 5 | Running `asr_with_diarization` preset without pyannote installed |
| `chapters` step: unexpected kwarg | 2 | API signature mismatch |
| Simulated failures (env var trigger) | 6 | Test harness artifacts — not real |
| Missing deps (moonshine, pyannote, demucs, deepfilternet, clap) | ~15 | Packages not installed |
| whisper_cpp: binary not found | 6 | WHISPER_CPP_BIN not set |
| ffmpeg errors | 5 | Bad input audio files |
| HF auth / rate limits | scattered | No HF_TOKEN configured |

### HF Sprint Status (ran today)

| Agent | OK | Failed | Skipped |
|---|---:|---:|---:|
| general_baselines | 16 | 7 | 3 |
| edge_small | 9 | 18 | 0 |
| domain_specialist | 4 | 8 | 2 |
| realtime_streaming | 3 | 22 | 0 |

**Sprint models that succeeded**: distil_whisper, faster_distil_whisper_large_v3, nb_whisper_small_onnx, silero_vad, heuristic_diarization, demucs, yamnet, whisper, faster_whisper, faster_whisper_large_v3, glm_asr_nano_2512

**Sprint models that failed**: moonshine (no pkg), whisper_cpp (no binary), pyannote_diarization (no pkg), clap (load error), deepfilternet (no pkg), all realtime_streaming models (API deps/mock-only)

---

## 2. Blocking Issues (Priority Order)

### P0: HF Token Not Configured
- No `HF_TOKEN` in `.env` or `~/.cache/huggingface/token`
- HF Pro subscription is **active until March 1st** but we can't use gated models
- **Action**: User must run `hf auth login` or add `HF_TOKEN=hf_xxx` to `.env`

### P1: Missing Python Dependencies (5 models broken)
Models with loaders but missing packages:

| Model | Missing Package | Install |
|---|---|---|
| `moonshine` | `moonshine` | `pip install -r models/moonshine/requirements.txt` |
| `pyannote_diarization` | `pyannote.audio` | `uv add pyannote.audio` (needs HF token for model weights) |
| `deepfilternet` | `deepfilternet` | `uv add deepfilternet` |
| `demucs` | Already works ✅ | — |
| `clap` | Loads but crashes on inference | Debug needed |

### P2: Alignment Step Bugs (29 failures)
The `alignment` step is the #1 source of failures:
- Looks for ASR artifact at a file path but gets `None` or wrong path
- `PosixPath.get()` — code passes a Path object where a dict is expected
- **Fix**: Debug `harness/alignment.py` artifact resolution logic

### P3: Realtime Streaming Models (all mock-only)
`voxtral`, `voxtral_realtime_2602`, `kyutai_streaming`, `nemotron_streaming`, `parakeet_multitalker` — all 5 streaming models are mock/stub implementations that don't actually call real APIs or load real weights.

---

## 3. What We Can Do With HF Pro (16 days left)

### High-Value Models to Test (HF Pro unlocks)
1. **pyannote/speaker-diarization-3.1** — gated, needs HF token + Pro
2. **Whisper large-v3-turbo** — via transformers, HF Pro gives faster downloads
3. **SeamlessM4T** — already registered, needs HF bandwidth
4. **LFM2.5-Audio** — already registered, may need HF for some weights
5. **Any new models on HF** — faster inference API, higher rate limits

### Immediate Sprint Fix Plan
1. Get HF_TOKEN configured ← **user action needed**
2. Install missing deps: `pyannote.audio`, `moonshine`, `deepfilternet`
3. Fix `alignment.py` artifact path bug
4. Re-run HF sprint with fixes
5. Extract WER/latency numbers from successful runs (currently 0 WER rows in report!)

---

## 4. Successful Models — What We Know

From sprint + session runs, these models work end-to-end on this Mac (M-series, MPS):

| Model | Caps | Load Time | Notes |
|---|---|---|---|
| whisper (tiny/base/large-v3) | ASR | ~3s (tiny) | Production-grade, FP32 on CPU |
| faster_whisper | ASR | fast | CTranslate2, production status |
| faster_whisper_large_v3 | ASR | ~60s run | Best accuracy candidate |
| distil_whisper | ASR | ~80s run | Good accuracy, slower than expected |
| nb_whisper_small_onnx | ASR | ~12-20s run | ONNX optimized, edge candidate |
| glm_asr_nano_2512 | ASR | works | GLM-4 based, experimental |
| silero_vad | VAD | instant | Production status, torch hub |
| heuristic_diarization | Diarization | instant | Energy-based, no ML |
| yamnet | Classify | ~13s run | Audio event classification |
| demucs | Separate | ~18s run | Source separation works |

---

## 5. Fixes Applied

### Alignment Step Bug — FIXED ✅
- **Root cause**: Fallback path used hardcoded `asr.json` but actual files are named `asr_faster_whisper_default.json` etc.
- **Fix**: `harness/session.py` alignment_func now globs `asr_*.json` and `diarization/diarization_*.json` as fallback.
- **Verified**: Full pipeline (ingest→asr→diarization→alignment→chapters→summarize→action_items→bundle) now completes end-to-end.
- **Tests**: `test_backend_invariants.py` (7/7 pass), `test_artifact_download_security.py` (1/1 pass).

### Full Pipeline Smoke Test — PASS ✅
```
ingest:                 COMPLETED (880ms)
asr:                    COMPLETED (8562ms)
diarization:            COMPLETED (2893ms)
alignment:              COMPLETED (8ms)
chapters:               COMPLETED (7964ms)
summarize_by_speaker:   COMPLETED (1ms)
action_items_assignee:  COMPLETED (4ms)
bundle:                 COMPLETED (4ms)
```

---

## 6. Next Steps

- [ ] **User**: Run `.venv/bin/python -m huggingface_hub.cli.hf_cli login` with Pro account token
- [ ] **User**: Add `HF_TOKEN=hf_xxx` to `.env`
- [x] ~~Fix alignment step bug~~ (DONE)
- [ ] Install pyannote.audio (`pip install pyannote.audio`) — needs HF token for gated model weights
- [ ] Install moonshine (`pip install -r models/moonshine/requirements.txt`)
- [ ] Install deepfilternet (`pip install deepfilternet`)
- [ ] Re-run HF sprint → collect actual WER numbers
- [ ] Test pyannote diarization (HF Pro gated model — high value)
- [x] ~~Run full pipeline end-to-end~~ (DONE — all 8 steps complete)
- [ ] Document which models are viable for which use cases
