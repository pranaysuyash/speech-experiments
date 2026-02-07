# LCS Log

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

**Tests**: All structural tests pass, model tests skip without deps.

---

## LCS-14–17: Batch 2 ASR Models

**4 ASR models across 3 runtimes**:

| LCS | Model | Runtime | Notes |
|-----|-------|---------|-------|
| LCS-14 | faster_whisper_large_v3 | CTranslate2 | Systran optimized Whisper |
| LCS-15 | faster_distil_whisper_large_v3 | CTranslate2 | 2-3x faster distilled |
| LCS-16 | glm_asr_nano_2512 | PyTorch | Non-Whisper architecture |
| LCS-17 | nb_whisper_small_onnx | ONNX | Cross-platform runtime |

**Files per model**: config.yaml, claims.yaml, requirements.txt, README.md, smoke tests

**CLI**:
```bash
make asr-audio MODEL=faster_whisper_large_v3 AUDIO=inputs/sample_16k.wav
make run-pipeline PIPELINE=config/pipelines/enhance_asr.yaml AUDIO=inputs/sample.wav
```

**Tests**: All structural tests pass, model tests skip without deps.

---

## LCS-13: Pipeline Integration

**Feature**: Linear pipeline runner for chaining models

**Files**:
- `harness/pipeline.py` - PipelineRunner, PipelineConfig, surface handlers
- `config/pipelines/enhance_asr.yaml` - preprocess → transcribe
- `config/pipelines/separate_vocals_asr.yaml` - extract vocals → transcribe
- `config/pipelines/separate_vocals_transcribe.yaml` - vocals → MIDI
- `tests/unit/test_pipeline.py` - 22 unit tests (fake models)
- `tests/integration/test_pipeline_integration.py` - 5 tests
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

**Testing**: All 27 tests CI-safe (use fake models), RNNoise integration test skipped if not installed.

---

## LCS-12: Basic Pitch Music Transcription

**Surfaces**: music_transcription (first!)

**Runtime**: tensorflow

**Devices**: cpu

**Files**:
- `models/basic_pitch/config.yaml`
- `models/basic_pitch/claims.yaml` (music_transcription surface)
- `models/basic_pitch/requirements.txt`
- `models/basic_pitch/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_basic_pitch_smoke.py` - 10 tests

**Output Contract**:
```python
{
    "notes": [
        {"onset": 0.5, "offset": 1.2, "pitch": 60, "velocity": 0.8},
        ...
    ],
    "midi": <PrettyMIDI>,
}
```

**Note Fields**:
- `onset`: Note start time (seconds)
- `offset`: Note end time (seconds)
- `pitch`: MIDI pitch (0-127)
- `velocity`: Note intensity (0.0-1.0)

**Commands**:
```bash
make model-install MODEL=basic_pitch
python -m pytest tests/integration/test_model_basic_pitch_smoke.py -v
```

**Notes**: 22.05kHz native, auto-resampling. Instrument-agnostic polyphonic detection.

---

## LCS-11: Demucs Source Separation

**Surfaces**: separate (first!)

**Runtime**: pytorch

**Devices**: cpu, mps, cuda

**Files**:
- `models/demucs/config.yaml` (variants: htdemucs, htdemucs_ft, htdemucs_6s, mdx_extra)
- `models/demucs/claims.yaml` (separate surface, stem structure)
- `models/demucs/requirements.txt`
- `models/demucs/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_demucs_smoke.py` - 13 tests

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

**Notes**: 44.1kHz native, auto-resampling. Non-speech audio category. Unlocks separate surface.

---

## LCS-10: Voxtral Streaming ASR

**Surfaces**: asr_stream (first!), asr (batch via streaming)

**Runtime**: api (Mistral cloud)

**Devices**: cpu (API-based)

**Files**:
- `models/voxtral/config.yaml` (variants: mini, small, realtime)
- `models/voxtral/claims.yaml` (streaming=true, asr_stream)
- `models/voxtral/requirements.txt` (mistralai[realtime])
- `models/voxtral/README.md`
- `harness/registry.py` - VoxtralStreamingAdapter using StreamingAdapter base
- `tests/integration/test_model_voxtral_smoke.py` - 13 tests

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
- `tests/unit/test_streaming_utils.py` - 33 tests

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
python -m pytest tests/unit/test_streaming_utils.py -v  # 33 passed
```

**Notes**: No model runtime imports. Lightweight. Ready for Voxtral adapter.

---

## LCS-08: CLAP Embed + Classify

**Surfaces**: embed (first!), classify (multi-surface model!)

**Runtime**: pytorch

**Devices**: cpu, mps, cuda

**Files**:
- `models/clap/config.yaml`
- `models/clap/claims.yaml` (embed + classify claims)
- `models/clap/requirements.txt` (laion-clap)
- `models/clap/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_clap_smoke.py` - 10 tests

**Commands**:
```bash
make model-install MODEL=clap
python -m pytest tests/integration/test_model_clap_smoke.py -v
```

**Notes**: 512-d embeddings. Zero-shot classification via text prompts. First multi-surface model.

---

## LCS-07: DeepFilterNet Audio Enhancement

**Surfaces**: enhance (second implementation, production-grade)

**Runtime**: pytorch

**Devices**: cpu, mps (fallback to cpu)

**Files**:
- `models/deepfilternet/config.yaml` (variants df2/df3)
- `models/deepfilternet/claims.yaml` (ci=false, PyTorch)
- `models/deepfilternet/requirements.txt`
- `models/deepfilternet/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_deepfilternet_smoke.py` - 10 tests

**Commands**:
```bash
make model-install MODEL=deepfilternet
python -m pytest tests/integration/test_model_deepfilternet_smoke.py -v
```

**Notes**: 48kHz native, auto-resampling, length preservation enforced. Pre-ASR pipeline candidate.

---

## LCS-06: RNNoise Audio Enhancement

**Surfaces**: enhance (first implementation!)

**Runtime**: native (C library)

**Devices**: cpu

**Files**:
- `models/rnnoise/config.yaml`
- `models/rnnoise/claims.yaml` (streaming=true)
- `models/rnnoise/requirements.txt` (pyrnnoise)
- `models/rnnoise/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_rnnoise_smoke.py` - 8 tests

**Commands**:
```bash
make model-install MODEL=rnnoise
python -m pytest tests/integration/test_model_rnnoise_smoke.py -v
```

**Notes**: Real-time <10ms latency. 48kHz native, auto-resampling. VAD probs returned.

---

## LCS-05: YAMNet Audio Classification

**Surfaces**: classify (first implementation!)

**Runtime**: tensorflow

**Devices**: cpu

**Files**:
- `models/yamnet/config.yaml`
- `models/yamnet/claims.yaml` (ci=false, TF heavyweight)
- `models/yamnet/requirements.txt`
- `models/yamnet/README.md`
- `harness/registry.py` - loader + registration
- `tests/integration/test_model_yamnet_smoke.py` - 8 tests (4 structural, 4 model)

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
- `tests/integration/test_model_moonshine_smoke.py` - 7 tests (3 structural, 4 model)

**Commands**:
```bash
make model-install MODEL=moonshine
python -m pytest tests/integration/test_model_moonshine_smoke.py -v
```

**Notes**: 27M params, 5-15x faster than Whisper on short segments. English-only.

---

## LCS-03: Enhance + Separate Metrics + Streaming Misuse

**Surfaces**: enhance, separate (metrics support)

**Files**:
- `harness/metrics_enhance.py` - si_snr, stoi, pesq (optional deps)
- `harness/metrics_separate.py` - bss_eval, sdr, multi_source_sdr (mir_eval optional)
- `tests/unit/test_metrics_enhance.py` - 12 tests
- `tests/unit/test_metrics_separate.py` - 8 tests
- `tests/unit/test_asr_stream_contract_misuse.py` - 8 tests

**Commands**:
```bash
python -m pytest tests/unit/test_metrics_enhance.py tests/unit/test_metrics_separate.py -v  # 14 pass, 6 skip
python -m pytest tests/unit/test_asr_stream_contract_misuse.py -v  # 8 pass
```

**Notes**: 
- SI-SNR always available, STOI/PESQ optional (never fail CI)
- SDR/SIR/SAR via mir_eval (optional)
- Streaming lifecycle: finalize is idempotent, close always safe

---

## LCS-02: Classification Metrics

**Surfaces**: classify (metrics support)

**Files**:
- `harness/metrics_classify.py` - accuracy_top1, precision_recall_f1, confusion_matrix, per_class_metrics
- `tests/unit/test_metrics_classify.py` - 14 tests

**Commands**:
```bash
python -m pytest tests/unit/test_metrics_classify.py -v  # 14 passed
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
python -c "from harness.contracts import CONTRACT_VERSION; print(CONTRACT_VERSION)"  # 2.0.0
python -m pytest tests/unit/test_contract_enforcement.py::TestNewSurfaces -v  # 6 passed
```

**Notes**: CONTRACT_VERSION=2.0.0. Streaming lifecycle: start_stream → push_audio → flush → finalize → close. validate_bundle now data-driven.
