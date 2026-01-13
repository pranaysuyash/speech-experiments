# Arsenal Decision Matrix

> **Core Principle**: Decisions = Observed Evidence + Declared Intent.
> **Decision Semantics**: v2.0 (Graduated Outcomes + Pipeline Support)

## Summary

### Offline Transcription
*Evaluation mode: single_model*

**✅ Recommended:** `faster_whisper`

| Model | Outcome | Score | Details |
|-------|---------|-------|---------|
| **faster_whisper** | ✅ RECOMMENDED | 1.5 | 6 strengths, 0 warnings |
| **whisper** | ⚠️ ACCEPTABLE | 1.5 | 6 strengths, 1 warnings |
| **seamlessm4t** | ❌ REJECTED | 0.0 | asr: evidence grade smoke < required golden_batch |
| **silero_vad** | ❌ REJECTED | 0.0 | Missing primary capability: asr |
| **whisper_cpp** | ❌ REJECTED | 0.0 | No valid evidence for: asr |
| **lfm2_5_audio** | ❌ REJECTED | 0.0 | asr: evidence grade smoke < required golden_batch |
| **heuristic_diarization** | ❌ REJECTED | 0.0 | Missing primary capability: asr |
| **pyannote_diarization** | ❌ REJECTED | 0.0 | Missing primary capability: asr<br>Requires offline capability |
| **distil_whisper** | ❌ REJECTED | 0.0 | No valid evidence for: asr<br>WER outside acceptable bounds |

---

### Real-Time Voice Assistant
*Evaluation mode: pipeline*

**✅ RECOMMENDED**

**Recommended Pipeline:**
```
  vad: silero_vad
  v2v: lfm2_5_audio
  asr: faster_whisper
```

| Task | Model | Grade | Metric |
|------|-------|-------|--------|
| vad | silero_vad | smoke | speech_ratio |
| v2v | lfm2_5_audio | smoke | rtf_like=0.56 |
| asr | faster_whisper | golden_batch | 1-WER |

---

### Meeting Intelligence
*Evaluation mode: pipeline*

**✅ RECOMMENDED**

**Recommended Pipeline:**
```
  asr: faster_whisper
  diarization: heuristic_diarization
  vad: silero_vad
```

| Task | Model | Grade | Metric |
|------|-------|-------|--------|
| asr | faster_whisper | golden_batch | 1-WER |
| diarization | heuristic_diarization | smoke | speaker_accuracy |
| vad | silero_vad | smoke | speech_ratio |

---
