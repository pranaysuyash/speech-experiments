# HF Pro Sprint Plan — Maximize Before March 1

**Window**: 16 days (Feb 13 → Mar 1, 2026)
**Goal**: Make model-lab the definitive audio AI testing platform — every modality, not just ASR.
**Ticket**: TCK-20260213-002

---

## Current Coverage (11 working models)

| Capability | Models Working | Gap |
|---|---|---|
| ASR/STT | whisper, faster_whisper, distil_whisper, glm_asr, nb_whisper, faster_whisper_large_v3 | ✅ Covered |
| VAD | silero_vad | ✅ Covered |
| Diarization | heuristic_diarization | ⚠️ Need pyannote (gated) |
| Source Separation | demucs | ✅ Covered |
| Audio Classification | yamnet | ⚠️ CLAP broken |
| TTS | glm_tts (registered) | ❌ No working TTS |
| Voice Cloning | — | ❌ Missing entirely |
| Music Generation | basic_pitch (transcription only) | ❌ No generation |
| Emotion Recognition | — | ❌ Missing entirely |
| NER (from transcripts) | — | ❌ Missing entirely |
| Language ID | — | ❌ Missing entirely |
| Speaker Embedding | — | ❌ Missing entirely |
| Audio Captioning | — | ❌ Missing entirely |
| Enhancement | deepfilternet, rnnoise (registered) | ⚠️ Not installed |
| Keyword Spotting | — | ❌ Missing entirely |

---

## Phase 1: Quick Wins (Day 1-2) — Fix broken + install ready

These are already registered or nearly working:

| Model | Capability | Action | HF Dependency |
|---|---|---|---|
| pyannote 3.1 | diarization | Accept license at hf.co/pyannote/speaker-diarization-3.1 | ✅ Gated |
| CLAP | classify + embed | Debug inference crash | ✅ laion/larger_clap_music |
| deepfilternet | enhancement | `uv pip install deepfilternet` | No |
| glm_tts | TTS | Test existing loader | ✅ THUDM |
| Fix CLAP loader | classify | Debug the load error | — |

## Phase 2: High-Value New Models (Day 2-5) — HF Pro bandwidth

Models that need HF downloads (maximize Pro):

### TTS & Voice Synthesis
| Model | HF Repo | Size | Why |
|---|---|---|---|
| **Kokoro-82M** | `hexgrad/Kokoro-82M` | 82M params | Tiny, runs on CPU, state-of-art quality for size |
| **Parler-TTS Mini** | `parler-tts/parler-tts-mini-v1.1` | ~600M | Describe voice in text, HF native |
| **Bark** | `suno/bark` | ~1.5B | Multi-speaker, effects, music |
| **F5-TTS** | `SWivid/F5-TTS` | ~350M | Zero-shot voice cloning, fast |

### Music & Audio Generation
| Model | HF Repo | Size | Why |
|---|---|---|---|
| **MusicGen Small** | `facebook/musicgen-small` | 300M | Text→music, runs on MPS |
| **MusicGen Medium** | `facebook/musicgen-medium` | 1.5B | Higher quality |
| **AudioLDM 2** | `cvssp/audioldm2` | ~1.5B | Text→any sound |

### Emotion & Paralinguistics
| Model | HF Repo | Size | Why |
|---|---|---|---|
| **emotion2vec+** | `emotion2vec/emotion2vec_plus_large` | ~300M | SOTA speech emotion |
| **SpeechBrain emotion** | `speechbrain/emotion-recognition-wav2vec2-IEMOCAP` | ~95M | Classic, well-tested |

### Speaker Analysis
| Model | HF Repo | Size | Why |
|---|---|---|---|
| **WeSpeaker** | `pyannote/wespeaker-voxceleb-resnet34-LM` | ~34M | Speaker verification/embedding |
| **SpeechBrain spk-verify** | `speechbrain/spkrec-ecapa-voxceleb` | ~15M | Speaker verification |

### Language ID
| Model | HF Repo | Size | Why |
|---|---|---|---|
| **SpeechBrain LangID** | `speechbrain/lang-id-voxlingua107-ecapa` | ~15M | 107 languages |
| *(whisper already does this)* | — | — | Built-in to existing ASR |

### Audio Understanding & Captioning
| Model | HF Repo | Size | Why |
|---|---|---|---|
| **Qwen2-Audio** | `Qwen/Qwen2-Audio-7B-Instruct` | 7B | Audio→text understanding (if fits in RAM) |
| **SALMONN** (if feasible) | — | 13B | May be too large |

## Phase 3: Non-Model Pipelines (Day 5-8) — No HF needed

These use existing libraries, no model downloads:

| Pipeline | Library | Capability |
|---|---|---|
| **NER from transcript** | `spacy` or `transformers` NER | Extract entities (names, places, orgs) from ASR output |
| **Keyword Spotting** | Energy + template matching | Detect trigger words in audio |
| **Audio Fingerprinting** | `chromaprint` / `acoustid` | Audio identification |
| **Pitch Detection** | `crepe` or `pyin` (librosa) | F0 extraction for voice analysis |
| **Formant Analysis** | `parselmouth` (Praat) | Voice quality metrics |
| **Rhythm/Tempo** | `librosa.beat` | BPM detection |
| **Silence/Activity Detection** | Energy-based | Non-ML VAD baseline |

## Phase 4: Integration & Evaluation (Day 8-14)

- Wire all new models into `harness/registry.py` with Bundle Contract
- Create evaluation datasets per capability (not just ASR)
- Build comparison scorecards per modality
- Run full sprint with all models
- Collect metrics: latency, memory, quality scores

## Phase 5: Documentation & Findings (Day 14-16)

- Model comparison matrix across all capabilities
- Production readiness grades per model
- Cost/quality tradeoff analysis
- Recommendations per use case

---

## Install Plan (ordered by priority)

```bash
# Phase 1 - Fix existing
# User: visit https://hf.co/pyannote/speaker-diarization-3.1 → Accept
uv pip install deepfilternet

# Phase 2 - TTS
uv pip install kokoro>=0.9  # Kokoro-82M
uv pip install parler-tts    # Parler TTS
uv pip install bark           # Suno Bark

# Phase 2 - Music
# MusicGen via transformers (already installed)

# Phase 2 - Emotion
uv pip install funasr          # emotion2vec
uv pip install speechbrain     # emotion, langid, speaker verify

# Phase 3 - NLP/Non-model
uv pip install spacy
python -m spacy download en_core_web_sm
uv pip install crepe           # Pitch detection
uv pip install parselmouth     # Formant analysis
```

---

## Registry Pattern for New Models

Every new model gets:
1. `models/<model_name>/config.yaml` — model metadata
2. Loader function in `harness/registry.py` → returns Bundle Contract
3. Entry in `config/hf_sprint_2026q1.yaml` under appropriate agent
4. Smoke test with test audio

Bundle capabilities to add: `tts`, `voice_clone`, `music_gen`, `emotion`, `langid`, `speaker_embed`, `speaker_verify`, `ner`, `caption`, `pitch`, `enhance`

---

## Success Metrics (by March 1)

- [ ] 30+ models registered and working (currently 11)
- [ ] 10+ distinct capability types (currently 6)
- [ ] WER numbers for all ASR models
- [ ] MOS/quality scores for TTS models
- [ ] Full pipeline: audio → ASR → diarization → NER → summary working
- [ ] At least 1 working model per capability category
- [ ] All findings documented with evidence
