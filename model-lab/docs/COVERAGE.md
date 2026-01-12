# Model Coverage Report

*Generated: 2026-01-12T16:13:59.252096*

This report shows what's actually tested vs what's claimed.

---

## Summary

| Model | Type | Capabilities | Evidence Tasks | Best Grade |
|-------|------|--------------|----------------|------------|
| seamlessm4t | seamlessm4t | asr, mt | asr | smoke |
| silero_vad | silero_vad | vad | vad | smoke |
| whisper_cpp | whisper_cpp | asr | none | none |
| lfm2_5_audio | lfm2_5_audio | asr, tts, chat | asr, v2v, tts | smoke |
| faster_whisper | faster_whisper | asr | asr | golden_batch |
| heuristic_diarization | heuristic_diarization | diarization | diarization | smoke |
| pyannote_diarization | pyannote_diarization | diarization | none | none |
| distil_whisper | distil_whisper | asr | asr | smoke |
| whisper | whisper | asr | asr | golden_batch |

## Details

### seamlessm4t

- **Version:** 2.0.0
- **Type:** seamlessm4t
- **Status:** experimental
- **Devices:** cpu, mps, cuda

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | ✅ | ✅ | smoke | 1 | latency_ms_p50=20379.07, rtf=0.12 |
| vad | — | — | — | — | — |
| diarization | — | — | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ⚠️ not selected
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['diarization']

---

### silero_vad

- **Version:** 4.0.0
- **Type:** silero_vad
- **Status:** production
- **Devices:** cpu, mps, cuda

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | — | — | — | — | — |
| vad | ✅ | ✅ | smoke | 2 | rtf=0.01, speech_ratio=0.44, num_segments=1.00 |
| diarization | — | — | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ❌ missing: ['asr']
- **real_time_assistant:** ❌ missing: ['v2v']
- **meeting_analysis:** ❌ missing: ['asr', 'diarization']

---

### whisper_cpp

- **Version:** 1.0.0
- **Type:** whisper_cpp
- **Status:** experimental
- **Devices:** cpu

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | ✅ | ❌ | — | — | — |
| vad | — | — | — | — | — |
| diarization | — | — | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ⚠️ not selected
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['diarization']

---

### lfm2_5_audio

- **Version:** 2.5.0
- **Type:** lfm2_5_audio
- **Status:** candidate
- **Devices:** cpu, mps

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | ✅ | ✅ | smoke | 1 | latency_ms_p50=31658.17, rtf=0.19 |
| vad | — | — | — | — | — |
| diarization | — | — | — | — | — |
| v2v | — | ✅ | smoke | 2 | latency_ms=4446.70, input_duration_s=4.00, response_duration_s=0.00 |
| tts | ✅ | ✅ | smoke | 1 | latency_ms_total=18198.33, latency_ms_avg=6066.11, rtf=2.05 |

#### Use Case Eligibility

- **offline_transcription:** ⚠️ not selected
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['diarization']

---

### faster_whisper

- **Version:** 1.0.0
- **Type:** faster_whisper
- **Status:** production
- **Devices:** cpu, cuda

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | ✅ | ✅ | golden_batch | 2 | latency_ms_p50=3237.73, rtf=0.11, wer=1.34 |
| vad | — | — | — | — | — |
| diarization | — | — | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ⚠️ not selected
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['diarization']

---

### heuristic_diarization

- **Version:** 1.0.0
- **Type:** heuristic_diarization
- **Status:** experimental
- **Devices:** cpu, cuda, mps

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | — | — | — | — | — |
| vad | — | — | — | — | — |
| diarization | ✅ | ✅ | smoke | 2 | num_speakers_pred=1.00, speaker_count_error=0.00, der_proxy=0.00 |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ❌ missing: ['asr']
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['asr']

---

### pyannote_diarization

- **Version:** 3.1.0
- **Type:** pyannote_diarization
- **Status:** production
- **Devices:** cpu, cuda

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | — | — | — | — | — |
| vad | — | — | — | — | — |
| diarization | ✅ | ❌ | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ❌ missing: ['asr']
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['asr']

---

### distil_whisper

- **Version:** 3.0.0
- **Type:** distil_whisper
- **Status:** experimental
- **Devices:** cpu, mps, cuda

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | ✅ | ✅ | smoke | 1 | latency_ms_p50=1464.85, rtf=0.01 |
| vad | — | — | — | — | — |
| diarization | — | — | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ⚠️ not selected
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['diarization']

---

### whisper

- **Version:** 3.0.0
- **Type:** whisper
- **Status:** production
- **Devices:** cpu, mps, cuda

#### Task Evidence

| Task | Declared | Has Evidence | Grade | Runs | Key Metrics |
|------|----------|--------------|-------|------|-------------|
| asr | ✅ | ✅ | golden_batch | 3 | latency_ms_p50=146444.91, rtf=0.90, wer=0.33 |
| vad | — | — | — | — | — |
| diarization | — | — | — | — | — |
| v2v | — | — | — | — | — |
| tts | — | — | — | — | — |

#### Use Case Eligibility

- **offline_transcription:** ⚠️ not selected
- **real_time_assistant:** ❌ missing: ['vad', 'v2v']
- **meeting_analysis:** ❌ missing: ['diarization']

---

## Coverage Gaps

Tasks declared but not tested:

- `whisper_cpp` declares `asr` but has no evidence
- `lfm2_5_audio` declares `chat` but has no evidence
- `pyannote_diarization` declares `diarization` but has no evidence
