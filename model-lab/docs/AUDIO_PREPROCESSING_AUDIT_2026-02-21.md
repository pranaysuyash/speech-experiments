# Audio Preprocessing Audit (2026-02-21)

## Scope
This audit covers preprocessing for all audio-facing tasks in `model-lab`:
- ASR (batch + adhoc + streaming)
- diarization
- VAD
- enhancement/separation pipelines
- TTS post-processing and audio quality checks
- media ingest/extraction and canonicalization

It answers:
- what exists now
- what is possible/standard in 2026
- what should be done by task
- what is missing
- what to implement first

---

## Executive Summary

1. Strong foundation exists for deterministic ingest and traceability.
- `harness/media_ingest.py` canonicalizes to mono PCM WAV and records hashes + ffmpeg command/version.

2. Preprocessing is split across two systems with drift risk.
- System A: FFmpeg ingest filters via `IngestConfig`.
- System B: in-memory operator chain via `harness/preprocess_ops.py`.
- Registries overlap but are not fully unified in behavior or defaults.

3. Current default preprocessing depth is basic.
- Good: trim silence, loudness normalize, resample, simple denoise, speed, channel ops.
- Missing for production robustness: high-quality VAD-based endpointing, modern denoise options, dereverb, clipping repair, AGC policy, dataset-specific policy profiles.

4. One high-severity code consistency issue exists.
- `scripts/run_vad.py` and `scripts/run_diarization.py` appear to call `ingest_media(...)` with an outdated signature/object model compared with current `harness/media_ingest.py`.

5. Standards posture is partial.
- Loudness and canonical sample-rate patterns exist, but no explicit standards profile matrix by task (broadcast/podcast/ASR-stream/realtime-agent).

---

## What You Have Today (Repo Ground Truth)

### A) Ingest-time preprocessing (FFmpeg filtergraph)
Source: `harness/media_ingest.py`

Implemented knobs (`IngestConfig`):
- `trim_silence` (`silenceremove`)
- `normalize` (single-pass `loudnorm`)
- `denoise` (highpass+lowpass approximation)
- `speed` (`atempo`)
- `peak_normalize` (`dynaudnorm`)
- `compress_dynamics` (`acompressor`)
- `gate_noise` (`agate`)
- `mono_mix` (`pan=mono`)
- canonical output: mono, `sample_rate` (default 16k), PCM WAV

Strengths:
- deterministic artifacting with content hash + preprocess hash + audio fingerprint.
- explicit ffmpeg argv + version captured.

Limitations:
- loudnorm is single-pass only.
- denoise uses simple filtering, not learned denoiser or spectral NR.
- no dereverb stage.
- no clipping restoration/declip.

### B) In-memory operator chain
Source: `harness/preprocess_ops.py`

Implemented operators:
- `trim_silence`
- `normalize_loudness` (pyloudnorm)
- `normalize_volume`
- `resample`
- `extract_channel`
- `denoise` (noisereduce, optional)
- `speed`

Strengths:
- per-step hashes + metrics + durations.
- artifact-friendly chain logging.

Limitations:
- not equivalent to ingest filters for all ops.
- no built-in VAD-based trim operator yet.
- no dereverb/declip/AEC/beamforming operators.

### C) Pipeline preprocessing registry
Source: `harness/pipeline_config.py`

Registry exposes more names:
- `compress_dynamics`, `convert_samplerate`, `gate_noise`, `mono_mix`, `normalize_peak`, etc.

Important nuance:
- these registry items map into `IngestConfig` (FFmpeg path), not necessarily `run_preprocessing_chain` operators.
- this creates user-facing semantic drift depending on run path.

### D) Task templates using preprocess
Sources:
- `config/pipelines/enhance_asr.yaml`
- `config/pipelines/separate_vocals_asr.yaml`
- `config/pipelines/separate_vocals_transcribe.yaml`

You already support preprocess-before-ASR via either ingest or explicit chain and have enhance/separate pipeline templates.

### E) Metrics support relevant to preprocessing quality
- `harness/metrics_asr.py`: WER/CER
- `harness/metrics_enhance.py`: SI-SNR (always), optional STOI/PESQ
- `harness/metrics_vad.py`: speech ratio/segment count gates
- `harness/metrics_diarization.py`: speaker count + DER proxy

Gaps:
- diarization still proxy-level for DER.
- no dedicated objective preproc QA suite (LUFS delta pass bands, clipping %, SNR lift, VAD boundary F1, etc.) unified across tasks.

---

## Key Findings (Severity-ordered)

### [P0] Preprocessing API drift between scripts and ingest module
- `harness/media_ingest.py` current signature is `ingest_media(input_path, artifacts_dir, cfg)` and returns a dict.
- `scripts/run_vad.py` and `scripts/run_diarization.py` appear written for an older object-returning API (`ingest.cleanup()`, `ingest.audio`, etc.).
- Risk: preprocessing behavior differs or fails depending on entrypoint.

### [P1] Dual-path preprocessing without single contract
- FFmpeg ingest and in-memory chain both exist with overlapping but not identical semantics.
- Risk: same operator name can mean different implementation details depending on where invoked.

### [P1] No task-specific standards profiles
- No explicit profile set like `asr_realtime`, `asr_batch_hq`, `diarization_meeting`, `tts_delivery` with fixed thresholds and defaults.
- Risk: inconsistent quality and regressions when new models/runtimes are added.

### [P1] Enhancement stack defaults are basic
- Current denoise is filter-based by default; no primary learned denoiser path in preprocessing contract.
- Risk: poor robustness in real-world noisy recordings.

### [P2] Missing preproc observability dashboard metrics
- No central report over preprocessing KPIs (trim ratio, LUFS in/out, clipping ratio, DC offset, VAD speech coverage, SNR lift).

---

## Standards and Best-Practice Baseline (2026)

### 1) Canonical ingest and reproducibility
Use deterministic decode + canonical format for downstream tasks:
- mono, fixed SR (usually 16 kHz for ASR/VAD/diarization, task-dependent exceptions)
- keep source hash + processed hash + config hash
- keep exact tool/version and argv

Status: mostly implemented in `harness/media_ingest.py`.

### 2) Loudness normalization policy
- Use ITU/EBU-aligned loudness normalization when target is human playback consistency.
- For ML inference, avoid over-aggressive loudness changes unless validated for model family.

Status: `loudnorm` + `pyloudnorm` available.
Gap: no explicit profile table by task and no two-pass option.

### 3) VAD/endpointing
- For streaming/realtime ASR, endpointing behavior is critical (latency/accuracy tradeoff).
- Prefer tuned VAD + chunk/stride policy instead of only energy trim.

Status: partial streaming VAD flag exists; trim is mostly energy/ffmpeg-silence based.
Gap: no unified VAD-based trim operator and no benchmarked endpointing profiles.

### 4) Speech enhancement
- Use objective metrics (SI-SNR/STOI/PESQ where legal) plus downstream WER/DER impact.
- Learned denoisers often outperform simple EQ filtering in non-stationary noise.

Status: SI-SNR + optional STOI/PESQ implemented; enhancement templates exist.
Gap: denoise preprocessing default still basic.

### 5) Separation-first paths
- Separation can help ASR in overlapped speech/music-heavy input; must be benchmarked for WER and latency impact per dataset.

Status: demucs pipeline templates exist.
Gap: no standardized gating policy that decides when to apply separation.

---

## What Is Possible / Should Be Added

## A) Core preprocessing capabilities to support

1. Decode/canonicalization
- keep current ffmpeg canonicalization and hashing contract
- add explicit dithering/bit-depth strategy when needed

2. Silence/endpoint control
- retain edge trim
- add VAD-based boundary trim operator (batch) and shared endpoint policy (streaming)

3. Loudness/leveling
- profile-based targets:
  - `asr_inference_default`: minimal normalization, preserve speech envelope
  - `delivery_podcast`: integrated loudness target + peak safety

4. Noise handling
- keep lightweight filter denoise for fast path
- add learned denoise option (model-backed) for HQ path

5. Dynamics + gating
- compression/noise gate should be profile-gated and disabled by default in ASR unless validated

6. Channel policy
- explicit downmix strategy and channel-selection policy for meetings/calls

7. Optional advanced path
- dereverb, declip, and (for live comms) AEC/NS/AGC via dedicated DSP stack

---

## B) Recommended model/package stack (pragmatic)

### Ingest/DSP backbone
- FFmpeg filters for deterministic transforms (`silenceremove`, `loudnorm`, `acompressor`, `agate`, `arnndn`, `afftdn`).
- Python front-end remains `IngestConfig` + `PipelineConfig`.

### Speech enhancement / denoise
- Keep `noisereduce` as fallback.
- Add learned enhancement lane (SpeechBrain/DeepFilterNet class models) with explicit opt-in profile.

### VAD/endpointing
- Keep current VAD surface.
- Add production VAD profile (frame size, hysteresis, min speech/silence, endpoint timeout) usable by both batch trim and streaming.

### Diarization prep
- Keep pyannote path.
- Add overlap-aware preproc profile (less aggressive gating, preserve low-energy speaker tails).

### ASR-specific
- Preserve per-model tested defaults (as already done for `trim_silence` policy).
- Require preprocess contract checks for every newly added ASR/streaming model.

---

## Current vs Needed Matrix

| Area | Current | Needed |
|---|---|---|
| Canonical ingest | Strong | Keep; add profile versioning |
| Operator chain traceability | Strong | Unify semantics with ingest path |
| VAD endpoint policy | Partial | Shared batch+streaming policy + tests |
| Loudness standards profiles | Partial | Task profiles + pass/fail thresholds |
| Learned denoise | Limited | Add model-backed denoise lane |
| Dereverb/declip/AEC | Missing | Add optional advanced stack |
| Separation gating | Partial | Decision policy based on overlap/music detection |
| Preproc QA dashboard | Missing | Aggregate KPIs and regression alerts |
| Script/API consistency | Risky | Fix ingest API drift in VAD/diar scripts |

---

## Implementation Backlog (Ordered)

### Phase 0 (must do first)
1. Fix script/API drift for ingest usage in:
- `scripts/run_vad.py`
- `scripts/run_diarization.py`

2. Add a single preprocessing contract object used by both:
- ingest (ffmpeg path)
- in-memory chain

3. Add CI checks ensuring operator name parity and behavior parity tests.

### Phase 1 (production hardening)
1. Define profile presets:
- `asr_fast`
- `asr_hq`
- `streaming_realtime`
- `diarization_meeting`
- `tts_delivery`

2. Add KPI thresholds per profile:
- trim ratio bounds
- LUFS delta bounds
- clipping ratio max
- WER/DER non-regression constraints

3. Add learned denoise profile lane and benchmark.

### Phase 2 (advanced quality)
1. Add dereverb/declip options.
2. Add conditional separation policy.
3. Build preprocessing leaderboard report by dataset/model/profile.

---

## Concrete Standards Profile Proposal

### `asr_fast` (default)
- trim: conservative edge trim
- no compression/gate by default
- optional light loudness normalize only if level is extreme
- objective gate: no WER degradation vs no-preproc baseline > X%

### `asr_hq`
- trim + loudness + optional learned denoise
- optional separation if overlap/music score above threshold
- gate: WER improvement required on at least one paired dataset without catastrophic latency increase

### `streaming_realtime`
- fixed frame/chunk policy
- VAD hysteresis and endpoint latency SLO
- gate: partial/final stability + finalize latency p95

### `diarization_meeting`
- avoid aggressive gating that removes weak speaker turns
- preserve channel cues where useful
- gate: speaker-count and DER proxy non-regression

### `tts_delivery`
- loudness/peak compliance profile for output media
- gate: clipping/DC offset/silence ratio

---

## External References Used (standards/docs/model cards)
- FFmpeg audio filters (`loudnorm`, `silenceremove`, `acompressor`, `agate`, `arnndn`, `afftdn`): https://ffmpeg.org/ffmpeg-filters.html
- py-webrtcvad package (WebRTC VAD Python): https://github.com/wiseman/py-webrtcvad
- Hugging Face ASR task docs (chunking/stride pipeline patterns): https://huggingface.co/docs/transformers/tasks/asr
- Whisper model docs (feature extractor sampling rate conventions): https://huggingface.co/docs/transformers/model_doc/whisper
- pyannote speaker diarization model card: https://huggingface.co/pyannote/speaker-diarization-3.1
- SpeechBrain streaming ASR models:
  - https://huggingface.co/speechbrain/asr-streaming-conformer-librispeech
  - https://huggingface.co/speechbrain/asr-streaming-conformer-gigaspeech
- Voxtral model cards:
  - https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
  - https://huggingface.co/mistralai/Voxtral-Small-24B-2507
- Librosa docs: https://librosa.org/doc/main/index.html
- Torchaudio docs: https://pytorch.org/audio/stable/index.html
- ONNX Runtime docs: https://onnxruntime.ai/docs/

