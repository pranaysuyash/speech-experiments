# Model Arsenal

> Auto-generated from registry + configs + runs. Do not edit manually.
> Generated from commit: 12aadb1 (tree: 9336a89819b9)
> ⚠️ Working directory has uncommitted changes

## Summary

| Model | Status | Capabilities | Runtimes | Targets | WER | RTF |
|-------|--------|--------------|----------|---------|-----|-----|
| **faster_whisper** | ✅ production | asr | local | desktop | - | - |
| **pyannote_diarization** | ✅ production | diarization | local, api | server | - | - |
| **silero_vad** | ✅ production | vad | local, mobile, browser | server, edge | - | - |
| **whisper** | ✅ production | asr | local, api | desktop, server | - | - |
| **basic_pitch** | 🔬 experimental | music_transcription | local | desktop | - | - |
| **clap** | 🔬 experimental | embed, classify | local | desktop | - | - |
| **deepfilternet** | 🔬 experimental | enhance | local | desktop | - | - |
| **demucs** | 🔬 experimental | separate | local | desktop | - | - |
| **distil_whisper** | 🔬 experimental | asr | local | desktop, server | - | - |
| **faster_distil_whisper_large_v3** | 🔬 experimental | asr | local | desktop | - | - |
| **faster_whisper_large_v3** | 🔬 experimental | asr | local | desktop | - | - |
| **glm_asr_nano_2512** | 🔬 experimental | asr | local | desktop | - | - |
| **glm_tts** | 🔬 experimental | tts | local | desktop | - | - |
| **heuristic_diarization** | 🔬 experimental | diarization | local | server, edge | - | - |
| **kyutai_streaming** | 🔬 experimental | asr_stream | local | desktop | - | - |
| **lfm2_5_audio** | 🟡 candidate | asr, tts, chat | local, api | desktop, server | - | - |
| **moonshine** | 🔬 experimental | asr | local | desktop | - | - |
| **nb_whisper_small_onnx** | 🔬 experimental | asr | local | desktop | - | - |
| **nemotron_streaming** | 🔬 experimental | asr_stream | local | desktop | - | - |
| **parakeet_multitalker** | 🔬 experimental | asr | local | desktop | - | - |
| **rnnoise** | 🔬 experimental | enhance | local | desktop | - | - |
| **seamlessm4t** | 🔬 experimental | asr, mt | local | desktop | - | - |
| **voxtral** | 🔬 experimental | asr, asr_stream | local | desktop | - | - |
| **voxtral_realtime_2602** | 🔬 experimental | asr_stream | local | desktop | - | - |
| **whisper_cpp** | 🔬 experimental | asr | local, cli | desktop, edge, mobile | - | - |
| **yamnet** | 🔬 experimental | classify | local | desktop | - | - |

---

## Model Details

### basic_pitch

**basic_pitch** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | music_transcription |
| Accelerators | cpu |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=basic_pitch DATASET=primary
```

---

### clap

**clap** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | embed, classify |
| Accelerators | cpu, mps, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=clap DATASET=primary
```

---

### deepfilternet

**deepfilternet** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | enhance |
| Accelerators | cpu, mps |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=deepfilternet DATASET=primary
```

---

### demucs

**demucs** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | separate |
| Accelerators | cpu, mps, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=demucs DATASET=primary
```

---

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

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:136.2%<br>rtf:0.32x<br>latency_ms:9567ms | ❌ is_truncated | 2026-02-13 |

---

### faster_distil_whisper_large_v3

**faster_distil_whisper_large_v3** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=faster_distil_whisper_large_v3 DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:136.2%<br>rtf:0.50x<br>latency_ms:14982ms | ❌ is_truncated | 2026-02-13 |

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
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:134.5%<br>rtf:0.10x<br>latency_ms:2883ms | ❌ is_truncated | 2026-02-13 |

---

### faster_whisper_large_v3

**faster_whisper_large_v3** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=faster_whisper_large_v3 DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:136.2%<br>rtf:1.28x<br>latency_ms:38534ms | ❌ is_truncated | 2026-02-13 |

---

### glm_asr_nano_2512

**glm_asr_nano_2512** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, cuda, mps |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=glm_asr_nano_2512 DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:100.0%<br>rtf:0.00x<br>latency_ms:1ms | ❌ wer_valid | 2026-02-13 |

---

### glm_tts

**glm_tts** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | tts |
| Accelerators | cpu, cuda, mps |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=glm_tts DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.TTS | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

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

### kyutai_streaming

**kyutai_streaming** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr_stream |
| Accelerators | cpu, cuda, mps |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=kyutai_streaming DATASET=primary
```

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
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:134.5%<br>rtf:0.46x<br>latency_ms:13748ms | ❌ is_truncated | 2026-02-13 |
| tts | EvidenceGrade.SMOKE | cpu | latency_ms:4447ms<br>rtf:1.49x | ✅ Pass | 2026-02-13 |

**Known Issues:**
- Requires patched harness for MPS support (detokenizer fix)

**Hardware Notes:**
- MPS works well on Apple Silicon (via workaround)
- CPU inference is possible but slow

---

### moonshine

**moonshine** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, mps |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=moonshine DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

---

### nb_whisper_small_onnx

**nb_whisper_small_onnx** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=nb_whisper_small_onnx DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:100.0%<br>rtf:0.00x<br>latency_ms:1ms | ❌ wer_valid | 2026-02-13 |

---

### nemotron_streaming

**nemotron_streaming** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr_stream |
| Accelerators | cpu, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=nemotron_streaming DATASET=primary
```

---

### parakeet_multitalker

**parakeet_multitalker** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr |
| Accelerators | cpu, cuda |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=parakeet_multitalker DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

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

### rnnoise

**rnnoise** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | enhance |
| Accelerators | cpu |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=rnnoise DATASET=primary
```

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

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |
| TaskType.MT | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

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
| vad | EvidenceGrade.SMOKE | cpu | speech_ratio:0.44<br>num_segments:1.00<br>rtf:0.01x | ✅ Pass | 2026-02-13 |

---

### voxtral

**voxtral** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr, asr_stream |
| Accelerators | cpu |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=voxtral DATASET=primary
```

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

#### Observed Evidence
| Task | Grade | Device | Metrics | Gates | Ver. |
|------|-------|--------|---------|-------|------|
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:219.8%<br>rtf:0.00x<br>latency_ms:16ms | ❌ is_truncated | 2026-02-13 |

---

### voxtral_realtime_2602

**voxtral_realtime_2602** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | asr_stream |
| Accelerators | cpu, cuda, mps |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=voxtral_realtime_2602 DATASET=primary
```

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
| asr | EvidenceGrade.GOLDEN_BATCH | cpu | wer:136.2%<br>rtf:0.50x<br>latency_ms:14948ms | ❌ is_truncated | 2026-02-13 |

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

#### Declared Capabilities (Intent)
| Task | Role | Confidence | Notes |
|------|------|------------|-------|
| TaskType.ASR | TaskRole.PRIMARY | inferred | Inferred from registry capabilities |

**Hardware Notes:**
- Uses AVX/NEON SIMD for fast CPU inference
- No GPU required

---

### yamnet

**yamnet** by Unknown

| Attribute | Value |
|-----------|-------|
| Status | experimental |
| Capabilities | classify |
| Accelerators | cpu |
| Offline | ✅ |
| License | Unknown |

**Run:**
```bash
make asr MODEL=yamnet DATASET=primary
```

---
