# Model Arsenal

> Auto-generated from registry + configs + runs. Do not edit manually.
> Generated from commit: 861fa73 (tree: c965ec75304f)
> ⚠️ Working directory has uncommitted changes

## Summary

| Model | Status | Capabilities | Runtimes | Targets | WER | RTF |
|-------|--------|--------------|----------|---------|-----|-----|
| **faster_whisper** | ✅ production | asr | local | desktop | - | - |
| **pyannote_diarization** | ✅ production | diarization | local, api | server | - | - |
| **silero_vad** | ✅ production | vad | local, mobile, browser | server, edge | - | - |
| **whisper** | ✅ production | asr | local, api | desktop, server | - | - |
| **distil_whisper** | 🔬 experimental | asr | local | desktop, server | - | - |
| **heuristic_diarization** | 🔬 experimental | diarization | local | server, edge | - | - |
| **lfm2_5_audio** | 🟡 candidate | asr, tts, chat | local, api | desktop, server | - | - |
| **seamlessm4t** | 🔬 experimental | asr, mt | local | desktop | - | - |
| **whisper_cpp** | 🔬 experimental | asr | local, cli | desktop, edge, mobile | - | - |

---

## Model Details

### distil_whisper

**distil-whisper/distil-large-v3** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, mps, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=distil_whisper DATASET=primary
```

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | mps | wer:83.3%<br>rtf:0.02x<br>latency_ms:3244ms | ❌ wer_valid | 2026-01-09 |

---

### faster_whisper

**base** by guillaumekln

> Faster-Whisper is an optimized reimplementation of OpenAI Whisper.

| Attribute | Value |
|-----------|-------|
| Status | production |
| Capabilities | asr |
| Accelerators | cpu, cuda |
| Offline | ✅ |
| License | MIT |

**Run:**
```bash
make asr MODEL=faster_whisper DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | unknown | Optimized Whisper implementation for faster inference with minimal accuracy loss. |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | mps | wer:24.1%<br>rtf:0.12x<br>latency_ms:19802ms | ❌ is_truncated | 2026-01-09 |

---

### heuristic_diarization

**heuristic_diarization** by Silero-Based (Local)

> Heuristic diarizer extending Silero VAD with single-speaker assumption.

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | diarization |
| Accelerators | cpu, cuda, mps |
| Offline | ✅ |
| License | MIT |

**Run:**
```bash
make asr MODEL=heuristic_diarization DATASET=primary
```

**Sources:**
- [repo](https://github.com/snakers4/silero-vad) - Underlying VAD engine

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.DIARIZATION | TaskRole.PRIMARY | unknown | Baseline for pipeline verification. Assumes 1 speaker max. |

**Strengths:** fast, zero_dependency

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| diarization | EvidenceGrade.SMOKE | cpu | num_speakers:0.50<br>rtf:0.01x | ✅ Pass | - |

**Hardware Notes:**
- Lightweight baseline

---

### lfm2_5_audio

**lfm2_5_audio** by Liquid AI

> LFM-2.5-Audio is a multi-modal model for audio understanding and generation.

| Attribute | Value |
|-----------|-------|
| Status | candidate |
| Capabilities | asr, tts, chat |
| Accelerators | cpu, mps |
| Offline | ✅ |
| License | Unknown |

**Best for:** voice_assistant, interactive_kiosk

**Avoid if:** verbatim_transcription, subtitle_generation

**Run:**
```bash
make asr MODEL=lfm2_5_audio DATASET=primary
```

**Sources:**
- [hf](https://huggingface.co/LiquidAI/LFM2.5-Audio-1.5B) - Primary model card
- [paper](https://www.liquid.ai/) - Project page / docs

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.V2V | TaskRole.PRIMARY | unknown | Hero use-case: real-time speech-to-speech interaction (Conversational). |
| TaskType.ASR | TaskRole.SECONDARY | unknown | Designed for dialogue understanding, not verbatim transcription. |
| TaskType.TTS | TaskRole.SECONDARY | unknown | Functional text-to-speech, but optimized for stability over expressivity. |

**Strengths:** real_time_speech_to_speech, conversational_audio, interleaved_generation

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.ADHOC | mps | wer:137.8%<br>rtf:0.18x<br>latency_ms:29259ms | ❌ is_truncated | 2026-01-09 |
| tts | EvidenceGrade.SMOKE | mps | latency_ms:7053ms<br>rtf:2.13x | ✅ Pass | 2026-01-09 |
| v2v | EvidenceGrade.SMOKE | cpu | latency_ms:5287ms | ✅ Pass | - |

**Known Issues:**
- Requires patched harness for MPS support (detokenizer fix)

**Hardware Notes:**
- MPS works well on Apple Silicon (via workaround)
- CPU inference is possible but slow

---

### pyannote_diarization

**pyannote_diarization** by Pyannote

> State-of-the-art speaker diarization pipeline.

| Attribute | Value |
|-----------|-------|
| Status | production |
| Capabilities | diarization |
| Accelerators | cpu, cuda |
| Offline | ❌ |
| License | MIT |

**Run:**
```bash
make asr MODEL=pyannote_diarization DATASET=primary
```

**Sources:**
- [hf](https://huggingface.co/pyannote/speaker-diarization-3.1) - Official model card

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.DIARIZATION | TaskRole.PRIMARY | unknown | Reference implementation for speaker diarization. |

**Strengths:** accurate_clustering, overlap_detection

**Known Issues:**
- Requires pyannote.audio and HF_TOKEN

**Hardware Notes:**
- MPS support experimental via PyTorch

---

### seamlessm4t

**facebook/seamless-m4t-v2-large** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr, mt |
| Accelerators | cpu, mps, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=seamlessm4t DATASET=primary
```

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.ADHOC | mps | wer:96.3%<br>rtf:0.11x<br>latency_ms:17563ms | ✅ Pass | 2026-01-09 |

---

### silero_vad

**silero_vad** by Silero

> Pre-trained enterprise-grade Voice Activity Detection (VAD) model.

| Attribute | Value |
|-----------|-------|
| Status | production |
| Capabilities | vad |
| Accelerators | cpu, mps, cuda |
| Offline | ✅ |
| License | MIT |

**Run:**
```bash
make asr MODEL=silero_vad DATASET=primary
```

**Sources:**
- [repo](https://github.com/snakers4/silero-vad) - Official repository

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.VAD | TaskRole.PRIMARY | unknown | Production standard for open-source VAD. |

**Strengths:** lightweight, robust, fast

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| vad | EvidenceGrade.ADHOC | cpu | speech_ratio:0.44<br>num_segments:1.00<br>rtf:0.01x | ✅ Pass | 2026-01-09 |

---

### whisper

**large-v3** by OpenAI

> Whisper is a general-purpose speech recognition model.

| Attribute | Value |
|-----------|-------|
| Status | production |
| Capabilities | asr |
| Accelerators | cpu, mps, cuda |
| Offline | ✅ |
| License | MIT |

**Best for:** batch_transcription, desktop_productivity, call_analytics

**Avoid if:** real_time_voice_assistant, mobile_voice_notes

**Run:**
```bash
make asr MODEL=whisper DATASET=primary
```

**Sources:**
- [repo](https://github.com/openai/whisper) - Official repository
- [paper](https://arxiv.org/abs/2212.04356) - Whisper paper

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | unknown | Production-grade general purpose ASR. |
| TaskType.ALIGNMENT | TaskRole.PRIMARY | unknown | Implicit alignment via high-quality timestamp generation. |

**Strengths:** robustness, multilingual

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| alignment | EvidenceGrade.SMOKE | cpu | violations_mean:0.00<br>coverage_mean:1.00 | ✅ Pass | - |
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:29.1%<br>rtf:1.89x<br>latency_ms:308317ms | ❌ is_truncated | 2026-01-09 |

**Known Issues:**
- Requires ffmpeg for some audio formats

**Hardware Notes:**
- MPS works well on Apple Silicon
- CPU is slow but functional

---

### whisper_cpp

**whisper.cpp** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=whisper_cpp DATASET=primary
```

**Hardware Notes:**
- Uses AVX/NEON SIMD for fast CPU inference
- No GPU required

---
