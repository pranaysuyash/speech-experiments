# Accurate Status Update for ChatGPT (Jan 9, 2026)

**Current Objective**: Harden Multi-Capability Arsenal (ASR + TTS + Generic)

## 🚀 Accomplishments

### 1. Multi-Capability Evidence Pipelining
We successfully transformed the Arsenal system from a single-task (ASR) leaderboard into a multi-capability evidence gathering engine.

- **Pipeline**: `scan` → `EvidenceEntry` → `ModelCard` → `Arsenal Docs`
- **Supported Tasks**: ASR (legacy & new), TTS (new).
- **Extensible**: Generic schema supports future tasks (MT, VAD).

### 2. TTS Level 0 Pipeline (Active)
implemented a "Level 0" smoke test capability for TTS models.
- **Runner**: `scripts/run_tts.py` runs standard prompt set (`data/golden/tts_smoke_v1.yaml`).
- **Gates**: Captures `silence_ratio`, `clipping_ratio`, `dc_offset`, and latency.
- **Evidence**: Normalized into `EvidenceEntry` with health checks (valid/invalid).

### 3. Critical Fix: LFM2.5 on Apple Silicon (MPS)
We resolved blocking issues preventing LiquidAI's `lfm2_5_audio` from running on Mac.
- **ASR Fix**: Processor loaded on CPU, then moved to MPS (fixes `cuda` default).
- **TTS Fix**: Manually initialized `audio_detokenizer` property in `registry.py` to bypass hardcoded `.cuda()` call in `liquid-audio` library.
- **Status**: **Fully Working** for both ASR and TTS on MPS.

### 4. Arsenal Generator Refactor
The documentation generator (`scripts/generate_arsenal.py`) was heavily refactored:
- **Scans All Tasks**: Looks in `runs/<model_id>/*/*.json`.
- **Normalization**: Converts raw run JSONs into `EvidenceEntry` objects.
- **Schema**: Updated `ModelCard` to store `List[EvidenceEntry]` instead of just ASR summaries.
- **Keys**: Standardized on `device`, `verified_at`, `gates`, `metrics`.

## 🛠️ Technical Details

### EvidenceEntry Schema
New unified dataclass for all evidence:
```python
@dataclass
class EvidenceEntry:
    task: str                  # "asr", "tts"
    metrics: Dict          # e.g., {wer: 0.1} or {rtf: 2.1}
    gates: Dict            # e.g., {has_failure: False}
    valid: bool            # True if all gates pass
    device: str            # "mps", "cuda", "cpu"
    verified_at: str       # ISO date
    # ... hashes & metadata
```

### Generator Logic
- **ASR**: Auto-computes WER/CER if missing. Checks `wer_valid` flag.
- **TTS**: Extracts audio health (`healthy_count` vs `failed_count`). Valid if `healthy_count > 0`.

## 📉 Known Issues / Next Steps
- **Promotion Logic**: `validate_for_promotion` in `model_card.py` needs to be updated to check `evidence` list instead of deprecated `observed` fields.
- **UseCase Scoring**: `USE_CASES.md` generation logic needs to actively filter candidates based on the new `EvidenceEntry` validation flags.
- **Model Support**: Only `lfm2_5_audio` has TTS evidence. Need to run `scripts/run_tts.py` for other models (e.g., `seamlessm4t` if supported).

## 📄 File Manifest
- `scripts/run_tts.py`: TTS runner.
- `scripts/generate_arsenal.py`: Doc generator (refactored).
- `harness/model_card.py`: Schema (EvidenceEntry + ModelCard updates).
- `harness/registry.py`: Model loader (Patched for MPS).
- `docs/TTS_MPS_WORKAROUND.md`: Tech deep dive on the fix.