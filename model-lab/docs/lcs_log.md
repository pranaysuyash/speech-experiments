# LCS Log (Procedures & Rules)

## Procedure-only enforcement

This document contains procedures and illustrative schemas only. Do not paste outputs, results, timings, or pass/fail claims here. Paste receipts into `docs/run_receipts.md`.

To scan this file for disallowed icons/keywords used by your policy, run a repo-wide search based on your current banlist.
If the command prints any lines, edit this file to remove those tokens.

## ⚠️ Critical Findings & Workarounds

### MPS Threading Fix (2026-02-07)

**Problem**: MPS benchmarks crash with mutex lock error (`Invalid argument`)

**Root Cause**: macOS default `fork` multiprocessing copies mutexes in invalid state

**Fix**: Set spawn start method before importing torch:
```python
import multiprocessing
multiprocessing.set_start_method("spawn", force=True)
```

**Implementation note**: Ensure this is set at module import time (e.g., in `bench/runner.py`) before importing torch

---

### Demucs API Mismatch

**Problem**: Registry uses `demucs.api` but it doesn't exist in demucs 4.0.1

**Root Cause**: `.api` module added in later versions not yet on PyPI

**Known limitation**: Some `demucs` releases may not expose `demucs.api`. If imports fail, adjust the loader or pin to a compatible commit/version.

---

### pyrnnoise API Change

**Problem**: `RNNoise()` fails with `missing required positional argument: 'sample_rate'`

**Root Cause**: pyrnnoise 0.4.3 requires sample_rate in constructor

**Potential incompatibility**: `pyrnnoise` 0.4.3 may require `sample_rate` in the constructor. If you see a missing-argument error, update the loader to pass `sample_rate=48000`.

---

### Model Dependency Isolation

**Best Practice**: Heavy models should use isolated venvs to avoid conflicts:
- NeMo models: `.venv.nemo_*`
- TensorFlow models: may conflict with torch
- Demucs/CLAP: need torchvision which conflicts

---

## LCS-21: GLM-TTS (procedure-only)

### Local procedure

Run the following sequence and paste the terminal output into `docs/run_receipts.md`.

```bash
rm -rf models/glm_tts/venv models/glm_tts/repo
cd models/glm_tts && ./install.sh
huggingface-cli download zai-org/GLM-TTS --local-dir ckpt
cd ../..
./scripts/verify_glm_tts.sh
```

Gate:
- Run the repo's claim-leak gate against the GLM-TTS bundle files and paste the terminal output into docs/run_receipts.md.

## LCS-B2: Batch ASR Benchmark

**Multi-model sweep with comparison table**

```bash
# Sweep all CI-safe ASR models on one audio file
make bench-asr-all AUDIO=data/audio/clean_speech_10s.wav REF="expected text" DEVICE=cpu

# View sorted report
make bench-report-asr
```

**Enhanced metrics (wall_ms, text_len, num_segments)**

**New Makefile targets:**
- `bench-asr-all` - Run all ASR models via selector
- `bench-report-asr` - Generate sorted table from results/
- `bench-report-asr-stream` - Same for streaming

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-B1: Streaming ASR Benchmark Framework

**Latency + WER/CER benchmarks for asr_stream models**

```bash
# Run benchmark with optional reference for WER
make bench-asr-stream MODEL=kyutai_streaming AUDIO=data/audio/clean_speech_10s.wav REF="expected text"
```

**Example output schema (illustrative, not measured):**
```json
{
  "run_id": "<run_id>",
  "model_id": "kyutai_streaming",
  "surface": "asr_stream",
  "input": {"path": "...", "duration_s": "<number>", "sr": 16000},
  "metrics": {
    "first_token_latency_ms": "<number>",
    "partial_update_rate_hz": "<number>",
    "finalize_latency_ms": "<number>",
    "rtf": "<number>",
    "wer": "<number>",
    "cer": "<number>"
  },
  "timing": {"wall_s": "<number>", "rtf": "<number>"},
  "env": {"device": "cpu", "model_type": "kyutai_streaming"}
}
```

**Also added:** `make bench-asr` for batch ASR

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-Z: Model Selector

**Filter models by device/runtime/surface/ci**

```python
from harness.selector import list_models_by_filter, get_streaming_models

# Show all streaming models I can run on MPS
mps_streaming = list_models_by_filter(device="mps", surface="asr_stream")

# Shortcuts
streaming = get_streaming_models(device="mps")
ci_safe = get_ci_safe_models(surface="asr")
pytorch = get_models_by_runtime("pytorch")
```

**Filters:**
| Filter | Values |
|--------|--------|
| device | cpu, cuda, mps |
| surface | asr, asr_stream, tts, classify, enhance, embed, separate, music_transcription |
| runtime | pytorch, nemo, onnx, ctranslate2 |
| ci | true/false |

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-Y: Streaming Latency Measurement

**Dependency-free metrics for asr_stream models**

```bash
make asr-stream-audio MODEL=kyutai_streaming AUDIO=inputs/sample_16k.wav CHUNK_MS=160
```

**Example output schema (illustrative, not measured):**
```json
{
  "first_token_latency_ms": "<number>",
  "partial_update_rate_hz": "<number>",
  "finalize_latency_ms": "<number>",
  "real_time_factor": "<number>",
  "num_events": "<number>",
  "num_partials": "<number>",
  "num_finals": "<number>",
  "audio_duration_s": "<number>"
}
```

**Metrics:**
| Metric | Description |
|--------|-------------|
| first_token_latency_ms | Time until first non-empty text |
| partial_update_rate_hz | Partials per second |
| finalize_latency_ms | Time in finalize() call |
| real_time_factor | Processing time / audio duration |

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-18–22: Batch 3 Streaming + TTS

**5 models: 3 streaming ASR, 1 batch ASR with diarization, 1 TTS**

| LCS | Model | Runtime | Surface |
|-----|-------|---------|---------|
| LCS-19 | kyutai_streaming | PyTorch | asr_stream |
| LCS-18 | nemotron_streaming | NeMo | asr_stream |
| LCS-20 | parakeet_multitalker | NeMo | asr (batch) |
| LCS-21 | glm_tts | PyTorch | tts |
| LCS-22 | voxtral_realtime_2602 | PyTorch | asr_stream |

**Streaming Contract**:
- seq_monotonic, segment_id_stable, finalize_idempotent
- push_after_finalize raises RuntimeError

**Voxtral Realtime**: Configurable `transcription_delay_ms` (100-500ms range)

**NeMo models**: Use dedicated venvs (`.venv.nemo_*`)

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-14–17: Batch 2 ASR Models

**4 ASR models across 3 runtimes**:

| LCS | Model | Runtime | Notes |
|-----|-------|---------|-------|
| LCS-14 | faster_whisper_large_v3 | CTranslate2 | |
| LCS-15 | faster_distil_whisper_large_v3 | CTranslate2 | |
| LCS-16 | glm_asr_nano_2512 | PyTorch | Non-Whisper architecture |
| LCS-17 | nb-whisper-small-ONNX | ONNX | |

**Files per model**: config.yaml, claims.yaml, requirements.txt, README.md, smoke tests

**CLI**:
```bash
make asr-audio MODEL=faster_whisper_large_v3 AUDIO=inputs/sample_16k.wav
make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav
```

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-13: Pipeline Integration

**Feature**: Linear pipeline runner for chaining models

**Files**:
- `harness/pipeline.py` - PipelineRunner, PipelineConfig, surface handlers
- `config/pipelines/enhance_asr.yaml` - preprocess → transcribe
- `config/pipelines/separate_vocals_asr.yaml` - extract vocals → transcribe
- `config/pipelines/separate_vocals_transcribe.yaml` - vocals → MIDI
- `tests/unit/test_pipeline.py` (fake models)
- `tests/integration/test_pipeline_integration.py`
- `Makefile` - run-pipeline, model-install targets

**Pipeline Configs**:
1. `enhance → asr` (DeepFilterNet/RNNoise → Moonshine)
2. `separate(vocals) → asr` (Demucs → ASR)
3. `separate(vocals) → music_transcription` (Demucs → Basic Pitch)

**Interface**:
```python
from harness.pipeline import run_pipeline

result = run_pipeline("config/pipelines/enhance_asr.yaml", audio, sr)
print(result.final)        # Last step output
print(result.artifacts)    # All step outputs
```

**CLI**:
```bash
make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav
```

**Error Handling**: Errors include step_id, model_id, surface for debugging.

**Test surface**: Unit and integration tests exist for this area. Run the test suite locally/CI to obtain receipts (paste into `docs/run_receipts.md`).

---

## LCS-12: Basic Pitch Music Transcription

**Surfaces**: music_transcription

**Runtime**: tensorflow

**Devices**: cpu

**Files**:
- `models/basic_pitch/config.yaml`
- `models/basic_pitch/claims.yaml` (music_transcription surface)
- `models/basic_pitch/requirements.txt`
- `models/basic_pitch/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_basic_pitch_smoke.py`

**Output Contract**:
- `velocity`: Note intensity (0.0-1.0)

**Commands**:
```bash
make model-install MODEL=basic_pitch
python -m pytest tests/integration/test_model_basic_pitch_smoke.py -v
```

**Notes**: 22.05kHz native, auto-resampling. Instrument-agnostic polyphonic detection.

---

## LCS-11: Demucs Source Separation

**Surfaces**: separate

**Runtime**: pytorch

**Devices**: cpu, mps, cuda

**Files**:
- `models/demucs/config.yaml` (variants: htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra)
- `models/demucs/claims.yaml` (separate surface, stem structure)
- `models/demucs/requirements.txt`
- `models/demucs/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_demucs_smoke.py`

**Output Contract**:
```python
{
    "stems": {
        "vocals": np.array([...]),
        "drums": np.array([...]),
        "bass": np.array([...]),
        "other": np.array([...]),
    },
    "sr": 44100
}
```

**Length Alignment**: All stems same length as input (enforced via trim/pad).

**Commands**:
```bash
make model-install MODEL=demucs
python -m pytest tests/integration/test_model_demucs_smoke.py -v
```

**Notes**: 44.1kHz native, auto-resampling. Non-speech audio category.

---

## LCS-10: Voxtral Streaming ASR

**Surfaces**: asr_stream, asr (batch via streaming)

**Runtime**: api (Mistral cloud)

**Devices**: cpu (API-based)

**Files**:
- `models/voxtral/config.yaml` (variants: mini, small, realtime)
- `models/voxtral/claims.yaml` (streaming=true, asr_stream)
- `models/voxtral/requirements.txt` (mistralai[realtime])
- `models/voxtral/README.md`
- `harness/registry.py` - VoxtralStreamingAdapter using StreamingAdapter base
- `tests/integration/test_model_voxtral_smoke.py`

**Streaming Lifecycle**:
```python
asr_stream["start_stream"]()
for chunk in audio_chunks:
    events = list(asr_stream["push_audio"](chunk, sr))
flush_events = list(asr_stream["flush"]())
result = asr_stream["finalize"]()
asr_stream["close"]()
```

**Commands**:
```bash
make model-install MODEL=voxtral
export MISTRAL_API_KEY=your_key
make asr-stream MODEL=voxtral AUDIO=inputs/sample_16k.wav
```

**Notes**: Uses harness/streaming.py infrastructure. Mock mode when API key not set. seq monotonic, segment_id stable.

---

## LCS-09: Streaming Utilities

**Surfaces**: asr_stream infrastructure (no models, pure utilities)

**Files**:
- `harness/streaming.py` - Core streaming infrastructure
- `tests/unit/test_streaming_utils.py`

**Components**:

1. **Chunking Helpers** (`ChunkConfig`, `AudioChunker`)
   - `frame_ms`: 20ms default
   - `chunk_ms`: 160ms default
   - Sample-rate normalization (resample once, not per chunk)
   - Accepts `bytes` (pcm_s16le) or `np.ndarray` (float32/int16)

2. **StreamingAdapter Base Class**
   - Enforces `seq` monotonicity (always increasing)
   - Enforces `segment_id` stability (same ID across updates)
   - Lifecycle: `start_stream()` → `push_audio()` → `flush()` → `finalize()` → `close()`
   - `push_audio` before `start_stream` raises `StreamingAdapterError`
   - `finalize` is idempotent (return same result)

3. **SilenceEndpointer** (optional utility)
   - Energy-based silence detection
   - Default OFF, only when model doesn't provide endpoints

4. **FakeStreamingAdapter** for testing

**Commands**:
```bash
python -m pytest tests/unit/test_streaming_utils.py -v
```

**Notes**: No model runtime imports. Lightweight. Ready for Voxtral adapter.

---

## LCS-08: CLAP Embed + Classify

**Surfaces**: embed, classify

**Runtime**: pytorch

**Devices**: cpu, mps, cuda

**Files**:
- `models/clap/config.yaml`
- `models/clap/claims.yaml` (embed + classify claims)
- `models/clap/requirements.txt` (laion-clap)
- `models/clap/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_clap_smoke.py`

**Commands**:
```bash
make model-install MODEL=clap
python -m pytest tests/integration/test_model_clap_smoke.py -v
```

**Notes**: 512-d embeddings. Zero-shot classification via text prompts. First multi-surface model.

---

## LCS-07: DeepFilterNet Audio Enhancement

**Surfaces**: enhance

**Runtime**: pytorch

**Devices**: cpu, mps (fallback to cpu)

**Files**:
- `models/deepfilternet/config.yaml` (variants df2/df3)
- `models/deepfilternet/claims.yaml` (ci=false, PyTorch)
- `models/deepfilternet/requirements.txt`
- `models/deepfilternet/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_deepfilternet_smoke.py`

**Commands**:
```bash
make model-install MODEL=deepfilternet
python -m pytest tests/integration/test_model_deepfilternet_smoke.py -v
```

**Notes**: 48kHz native, auto-resampling, length preservation enforced. Pre-ASR pipeline candidate.

---

## LCS-06: RNNoise Audio Enhancement

**Surfaces**: enhance

**Runtime**: native (C library)

**Devices**: cpu

**Files**:
- `models/rnnoise/config.yaml`
- `models/rnnoise/claims.yaml` (streaming=true)
- `models/rnnoise/requirements.txt` (pyrnnoise)
- `models/rnnoise/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_rnnoise_smoke.py`

**Commands**:
```bash
make model-install MODEL=rnnoise
python -m pytest tests/integration/test_model_rnnoise_smoke.py -v
```

**Notes**: 48kHz native, auto-resampling, length preservation enforced. VAD probs returned.

---

## LCS-05: YAMNet Audio Classification

**Surfaces**: classify

**Runtime**: tensorflow

**Devices**: cpu

**Files**:
- `models/yamnet/config.yaml`
- `models/yamnet/claims.yaml` (ci=false, TF heavyweight)
- `models/yamnet/requirements.txt`
- `models/yamnet/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_yamnet_smoke.py`

**Commands**:
```bash
make model-install MODEL=yamnet
python -m pytest tests/integration/test_model_yamnet_smoke.py -v
```

**Notes**: 521 AudioSet classes. TF Hub model. Isolated venv recommended.

---

## LCS-04: Moonshine Tiny ASR

**Surfaces**: asr

**Runtime**: pytorch

**Devices**: cpu, mps (fallback to cpu)

**Files**:
- `models/moonshine/config.yaml` - model config with variants
- `models/moonshine/claims.yaml` - claims manifest
- `models/moonshine/requirements.txt` - isolated dependencies
- `models/moonshine/README.md` - usage docs
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_moonshine_smoke.py`

**Commands**:
```bash
make model-install MODEL=moonshine
python -m pytest tests/integration/test_model_moonshine_smoke.py -v
```

**Notes**: 27M params. English-only.

---

## LCS-03: Enhance + Separate Metrics + Streaming Misuse

**Surfaces**: enhance, separate (metrics support)

**Files**:
- `harness/metrics_enhance.py` - si_snr, stoi, pesq (optional deps)
- `harness/metrics_separate.py` - bss_eval, sdr, multi_source_sdr (mir_eval optional)
- `tests/unit/test_metrics_enhance.py`
- `tests/unit/test_metrics_separate.py`
- `tests/unit/test_asr_stream_contract_misuse.py`

**Commands**:
```bash
python -m pytest tests/unit/test_metrics_enhance.py tests/unit/test_metrics_separate.py -v
python -m pytest tests/unit/test_asr_stream_contract_misuse.py -v
```

**Notes**:
- SI-SNR always available, STOI/PESQ optional
- SDR/SIR/SAR via mir_eval (optional)
- Streaming lifecycle: finalize is idempotent, close always safe

---

## LCS-02: Classification Metrics

**Surfaces**: classify (metrics support)

**Files**:
- `harness/metrics_classify.py` - accuracy_top1, precision_recall_f1, confusion_matrix, per_class_metrics
- `tests/unit/test_metrics_classify.py`

**Commands**:
```bash
python -m pytest tests/unit/test_metrics_classify.py -v
```

**Notes**: CI-safe, dataset-free. Macro vs micro vs weighted averaging supported.

---

## LCS-01: 6 New Capability Surfaces

**Surfaces**: classify, embed, enhance, separate, music_transcription, asr_stream

**Files**:
- `harness/contracts.py` - 6 result types, 6 namespaces, Bundle v2
- `tests/unit/test_contract_enforcement.py` - 6 new surface tests

**Commands**:
```bash
python -c "from harness.contracts import CONTRACT_VERSION; print(CONTRACT_VERSION)"
python -m pytest tests/unit/test_contract_enforcement.py::TestNewSurfaces -v
```

**Notes**: Streaming lifecycle: start_stream → push_audio → flush → finalize → close. validate_bundle now data-driven.
