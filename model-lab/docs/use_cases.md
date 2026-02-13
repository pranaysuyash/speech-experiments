# Use Case Recommendations

> Auto-generated from `data/use_cases.yaml` + model evidence.
> Do not edit manually.

## Desktop Offline Transcription

_Privacy-focused batch transcription on desktop with no network_

### ❌ Not Recommended

- **lfm2_5_audio**: no verified evidence
- **whisper**: no verified evidence
- **faster_whisper**: no verified evidence
- **seamlessm4t**: no verified evidence
- **distil_whisper**: no verified evidence
- **pyannote_diarization**: missing capability: asr, wrong targets: ['server'], no verified evidence
- **heuristic_diarization**: missing capability: asr, wrong targets: ['server', 'edge'], no verified evidence
- **whisper_cpp**: no verified evidence
- **silero_vad**: missing capability: asr, wrong targets: ['server', 'edge'], no verified evidence
- **moonshine**: no verified evidence
- **yamnet**: missing capability: asr, no verified evidence
- **rnnoise**: missing capability: asr, no verified evidence
- **deepfilternet**: missing capability: asr, no verified evidence
- **clap**: missing capability: asr, no verified evidence
- **voxtral**: no verified evidence
- **demucs**: missing capability: asr, no verified evidence
- **basic_pitch**: missing capability: asr, no verified evidence
- **faster_whisper_large_v3**: no verified evidence
- **faster_distil_whisper_large_v3**: no verified evidence
- **glm_asr_nano_2512**: no verified evidence
- **nb_whisper_small_onnx**: no verified evidence
- **kyutai_streaming**: missing capability: asr, no verified evidence
- **glm_tts**: missing capability: asr, no verified evidence
- **voxtral_realtime_2602**: missing capability: asr, no verified evidence
- **nemotron_streaming**: missing capability: asr, no verified evidence
- **parakeet_multitalker**: no verified evidence

---

## Server-Side Batch Transcription

_High-volume transcription on server infrastructure_

### ❌ Not Recommended

- **lfm2_5_audio**: no verified evidence
- **whisper**: no verified evidence
- **faster_whisper**: no verified evidence
- **seamlessm4t**: no verified evidence
- **distil_whisper**: no verified evidence
- **pyannote_diarization**: missing capability: asr, no verified evidence
- **heuristic_diarization**: missing capability: asr, no verified evidence
- **whisper_cpp**: no verified evidence
- **silero_vad**: missing capability: asr, no verified evidence
- **moonshine**: no verified evidence
- **yamnet**: missing capability: asr, no verified evidence
- **rnnoise**: missing capability: asr, no verified evidence
- **deepfilternet**: missing capability: asr, no verified evidence
- **clap**: missing capability: asr, no verified evidence
- **voxtral**: no verified evidence
- **demucs**: missing capability: asr, no verified evidence
- **basic_pitch**: missing capability: asr, no verified evidence
- **faster_whisper_large_v3**: no verified evidence
- **faster_distil_whisper_large_v3**: no verified evidence
- **glm_asr_nano_2512**: no verified evidence
- **nb_whisper_small_onnx**: no verified evidence
- **kyutai_streaming**: missing capability: asr, no verified evidence
- **glm_tts**: missing capability: asr, no verified evidence
- **voxtral_realtime_2602**: missing capability: asr, no verified evidence
- **nemotron_streaming**: missing capability: asr, no verified evidence
- **parakeet_multitalker**: no verified evidence

---

## Real-Time Voice Assistant

_Low-latency ASR for conversational agents_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **lfm2_5_audio** | 0 | - |
| **whisper** | 0 | - |
| **faster_whisper** | 0 | - |
| **seamlessm4t** | 0 | - |
| **distil_whisper** | 0 | - |

### ❌ Not Recommended

- **pyannote_diarization**: missing capability: asr
- **heuristic_diarization**: missing capability: asr
- **silero_vad**: missing capability: asr
- **yamnet**: missing capability: asr
- **rnnoise**: missing capability: asr
- **deepfilternet**: missing capability: asr
- **clap**: missing capability: asr
- **demucs**: missing capability: asr
- **basic_pitch**: missing capability: asr
- **kyutai_streaming**: missing capability: asr
- **glm_tts**: missing capability: asr
- **voxtral_realtime_2602**: missing capability: asr
- **nemotron_streaming**: missing capability: asr

---

## Mobile Voice Notes

_On-device ASR for mobile apps with battery/size constraints_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **whisper_cpp** | 0 | - |

### ❌ Not Recommended

- **lfm2_5_audio**: wrong targets: ['desktop', 'server']
- **whisper**: wrong targets: ['desktop', 'server']
- **faster_whisper**: wrong targets: ['desktop']
- **seamlessm4t**: wrong targets: ['desktop']
- **distil_whisper**: wrong targets: ['desktop', 'server']
- **pyannote_diarization**: missing capability: asr, wrong targets: ['server']
- **heuristic_diarization**: missing capability: asr
- **silero_vad**: missing capability: asr
- **moonshine**: wrong targets: ['desktop']
- **yamnet**: missing capability: asr, wrong targets: ['desktop']
- **rnnoise**: missing capability: asr, wrong targets: ['desktop']
- **deepfilternet**: missing capability: asr, wrong targets: ['desktop']
- **clap**: missing capability: asr, wrong targets: ['desktop']
- **voxtral**: wrong targets: ['desktop']
- **demucs**: missing capability: asr, wrong targets: ['desktop']
- **basic_pitch**: missing capability: asr, wrong targets: ['desktop']
- **faster_whisper_large_v3**: wrong targets: ['desktop']
- **faster_distil_whisper_large_v3**: wrong targets: ['desktop']
- **glm_asr_nano_2512**: wrong targets: ['desktop']
- **nb_whisper_small_onnx**: wrong targets: ['desktop']
- **kyutai_streaming**: missing capability: asr, wrong targets: ['desktop']
- **glm_tts**: missing capability: asr, wrong targets: ['desktop']
- **voxtral_realtime_2602**: missing capability: asr, wrong targets: ['desktop']
- **nemotron_streaming**: missing capability: asr, wrong targets: ['desktop']
- **parakeet_multitalker**: wrong targets: ['desktop']

> **Note:** Requires small model size and CPU-only operation

---

## Browser-Only Demo

_WebGPU/WebAssembly models for browser without backend_

### ❌ Not Recommended

- **lfm2_5_audio**: wrong targets: ['desktop', 'server']
- **whisper**: wrong targets: ['desktop', 'server']
- **faster_whisper**: wrong targets: ['desktop']
- **seamlessm4t**: wrong targets: ['desktop']
- **distil_whisper**: wrong targets: ['desktop', 'server']
- **pyannote_diarization**: missing capability: asr, wrong targets: ['server']
- **heuristic_diarization**: missing capability: asr, wrong targets: ['server', 'edge']
- **whisper_cpp**: wrong targets: ['desktop', 'edge', 'mobile']
- **silero_vad**: missing capability: asr, wrong targets: ['server', 'edge']
- **moonshine**: wrong targets: ['desktop']
- **yamnet**: missing capability: asr, wrong targets: ['desktop']
- **rnnoise**: missing capability: asr, wrong targets: ['desktop']
- **deepfilternet**: missing capability: asr, wrong targets: ['desktop']
- **clap**: missing capability: asr, wrong targets: ['desktop']
- **voxtral**: wrong targets: ['desktop']
- **demucs**: missing capability: asr, wrong targets: ['desktop']
- **basic_pitch**: missing capability: asr, wrong targets: ['desktop']
- **faster_whisper_large_v3**: wrong targets: ['desktop']
- **faster_distil_whisper_large_v3**: wrong targets: ['desktop']
- **glm_asr_nano_2512**: wrong targets: ['desktop']
- **nb_whisper_small_onnx**: wrong targets: ['desktop']
- **kyutai_streaming**: missing capability: asr, wrong targets: ['desktop']
- **glm_tts**: missing capability: asr, wrong targets: ['desktop']
- **voxtral_realtime_2602**: missing capability: asr, wrong targets: ['desktop']
- **nemotron_streaming**: missing capability: asr, wrong targets: ['desktop']
- **parakeet_multitalker**: wrong targets: ['desktop']

> **Note:** Currently no browser-compatible ASR models registered

---

## Multi-Lingual Speech Translation

_Translate spoken audio across languages_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **lfm2_5_audio** | 0 | - |
| **whisper** | 0 | - |
| **faster_whisper** | 0 | - |
| **seamlessm4t** | 0 | - |
| **distil_whisper** | 0 | - |

### ❌ Not Recommended

- **pyannote_diarization**: missing capabilities: ['mt', 'asr']
- **heuristic_diarization**: missing capabilities: ['mt', 'asr']
- **silero_vad**: missing capabilities: ['mt', 'asr']
- **yamnet**: missing capabilities: ['mt', 'asr']
- **rnnoise**: missing capabilities: ['mt', 'asr']
- **deepfilternet**: missing capabilities: ['mt', 'asr']
- **clap**: missing capabilities: ['mt', 'asr']
- **demucs**: missing capabilities: ['mt', 'asr']
- **basic_pitch**: missing capabilities: ['mt', 'asr']
- **kyutai_streaming**: missing capabilities: ['mt', 'asr']
- **glm_tts**: missing capabilities: ['mt', 'asr']
- **voxtral_realtime_2602**: missing capabilities: ['mt', 'asr']
- **nemotron_streaming**: missing capabilities: ['mt', 'asr']

---

## TTS for Narration

_High-quality TTS for audiobook/podcast narration_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **lfm2_5_audio** | 0 | - |
| **glm_tts** | 0 | - |

### ❌ Not Recommended

- **whisper**: missing capability: tts
- **faster_whisper**: missing capability: tts
- **seamlessm4t**: missing capability: tts
- **distil_whisper**: missing capability: tts
- **pyannote_diarization**: missing capability: tts
- **heuristic_diarization**: missing capability: tts
- **whisper_cpp**: missing capability: tts
- **silero_vad**: missing capability: tts
- **moonshine**: missing capability: tts
- **yamnet**: missing capability: tts
- **rnnoise**: missing capability: tts
- **deepfilternet**: missing capability: tts
- **clap**: missing capability: tts
- **voxtral**: missing capability: tts
- **demucs**: missing capability: tts
- **basic_pitch**: missing capability: tts
- **faster_whisper_large_v3**: missing capability: tts
- **faster_distil_whisper_large_v3**: missing capability: tts
- **glm_asr_nano_2512**: missing capability: tts
- **nb_whisper_small_onnx**: missing capability: tts
- **kyutai_streaming**: missing capability: tts
- **voxtral_realtime_2602**: missing capability: tts
- **nemotron_streaming**: missing capability: tts
- **parakeet_multitalker**: missing capability: tts

> **Note:** TTS quality currently assessed via health gates only (clipping, silence)

---

## TTS for Conversational Agent

_Low-latency TTS for voice assistants_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **lfm2_5_audio** | 0 | - |
| **glm_tts** | 0 | - |

### ❌ Not Recommended

- **whisper**: missing capability: tts
- **faster_whisper**: missing capability: tts
- **seamlessm4t**: missing capability: tts
- **distil_whisper**: missing capability: tts
- **pyannote_diarization**: missing capability: tts
- **heuristic_diarization**: missing capability: tts
- **whisper_cpp**: missing capability: tts
- **silero_vad**: missing capability: tts
- **moonshine**: missing capability: tts
- **yamnet**: missing capability: tts
- **rnnoise**: missing capability: tts
- **deepfilternet**: missing capability: tts
- **clap**: missing capability: tts
- **voxtral**: missing capability: tts
- **demucs**: missing capability: tts
- **basic_pitch**: missing capability: tts
- **faster_whisper_large_v3**: missing capability: tts
- **faster_distil_whisper_large_v3**: missing capability: tts
- **glm_asr_nano_2512**: missing capability: tts
- **nb_whisper_small_onnx**: missing capability: tts
- **kyutai_streaming**: missing capability: tts
- **voxtral_realtime_2602**: missing capability: tts
- **nemotron_streaming**: missing capability: tts
- **parakeet_multitalker**: missing capability: tts

---

## Edge Device Deployment

_Raspberry Pi class edge devices_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **heuristic_diarization** | 0 | - |
| **whisper_cpp** | 0 | - |
| **silero_vad** | 0 | - |

### ❌ Not Recommended

- **lfm2_5_audio**: wrong targets: ['desktop', 'server']
- **whisper**: wrong targets: ['desktop', 'server']
- **faster_whisper**: wrong targets: ['desktop']
- **seamlessm4t**: wrong targets: ['desktop']
- **distil_whisper**: wrong targets: ['desktop', 'server']
- **pyannote_diarization**: wrong targets: ['server']
- **moonshine**: wrong targets: ['desktop']
- **yamnet**: wrong targets: ['desktop']
- **rnnoise**: wrong targets: ['desktop']
- **deepfilternet**: wrong targets: ['desktop']
- **clap**: wrong targets: ['desktop']
- **voxtral**: wrong targets: ['desktop']
- **demucs**: wrong targets: ['desktop']
- **basic_pitch**: wrong targets: ['desktop']
- **faster_whisper_large_v3**: wrong targets: ['desktop']
- **faster_distil_whisper_large_v3**: wrong targets: ['desktop']
- **glm_asr_nano_2512**: wrong targets: ['desktop']
- **nb_whisper_small_onnx**: wrong targets: ['desktop']
- **kyutai_streaming**: wrong targets: ['desktop']
- **glm_tts**: wrong targets: ['desktop']
- **voxtral_realtime_2602**: wrong targets: ['desktop']
- **nemotron_streaming**: wrong targets: ['desktop']
- **parakeet_multitalker**: wrong targets: ['desktop']

---
