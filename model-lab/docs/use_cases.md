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

---

## TTS for Narration

_High-quality TTS for audiobook/podcast narration_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **lfm2_5_audio** | 0 | - |

### ❌ Not Recommended

- **whisper**: missing capability: tts
- **faster_whisper**: missing capability: tts
- **seamlessm4t**: missing capability: tts
- **distil_whisper**: missing capability: tts
- **pyannote_diarization**: missing capability: tts
- **heuristic_diarization**: missing capability: tts
- **whisper_cpp**: missing capability: tts
- **silero_vad**: missing capability: tts

> **Note:** TTS quality currently assessed via health gates only (clipping, silence)

---

## TTS for Conversational Agent

_Low-latency TTS for voice assistants_

### ✅ Recommended Models

| Model | Score | Reasons |
|-------|-------|---------|
| **lfm2_5_audio** | 0 | - |

### ❌ Not Recommended

- **whisper**: missing capability: tts
- **faster_whisper**: missing capability: tts
- **seamlessm4t**: missing capability: tts
- **distil_whisper**: missing capability: tts
- **pyannote_diarization**: missing capability: tts
- **heuristic_diarization**: missing capability: tts
- **whisper_cpp**: missing capability: tts
- **silero_vad**: missing capability: tts

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

---
