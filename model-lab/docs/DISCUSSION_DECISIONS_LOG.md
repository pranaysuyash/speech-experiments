# Discussion Decisions Log

Purpose: capture major discussion outcomes and resulting implementation actions in chronological order.

---

## 2026-02-22

### Topic
Preprocessing audit, pipeline reliability, optional workflows, and broader modality expansion.

### Decisions Recorded
1. Fix current blockers before expansion:
- `mlx_whisper` runtime compatibility.
- `nb_whisper_small_onnx` real inference path (no placeholders).
- UI/backend support for reference transcript and WER/CER in same run flow.

2. Preprocessing must be optional and composable:
- Keep per-run optional step selection.
- Support suggested workflows based on task intent.
- Keep explicit full-pipeline option.

3. Treat current state as foundation, not end-state:
- Expand from audio-only to modality-agnostic model lab and finder.

### Implemented Actions
1. Runner reliability and contracts:
- Patched `scripts/run_vad.py` and `scripts/run_diarization.py` to current ingest contract.
- Added regressions:
  - `tests/unit/test_adhoc_ingest_contract.py`
  - `tests/unit/test_preprocessing_contract_parity.py`

2. Workflow suggestion system:
- Added `GET /api/pipelines/suggestions`.
- Added UI consumption and one-click apply in Workbench.
- Added suggestion controls: goal/quality/realtime.

3. Preprocessing mapping improvement:
- `normalize_volume(method=peak)` now maps to ingest `peak_normalize` instead of warning-only fallback.

4. Documentation:
- Added `docs/AUDIO_PREPROCESSING_AUDIT_2026-02-21.md`.
- Added `docs/MODALITY_LAB_VISION.md`.

### Validation
1. Backend integration/unit suites for pipelines + preprocess contracts passed.
2. Frontend API tests and production build passed.
3. Real adhoc runs for VAD/diarization against provided `.m4a` completed successfully.

### Next Active Track
1. Profile presets and enforceable KPI gates:
- `asr_fast`
- `asr_hq`
- `streaming_realtime`
- `diarization_meeting`
- `tts_delivery`
2. Continue modality-generalized architecture (`contracts_v2` + finder ranking service).
