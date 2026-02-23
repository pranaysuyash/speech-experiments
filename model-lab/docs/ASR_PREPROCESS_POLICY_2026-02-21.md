# ASR Preprocess Policy (2026-02-21)

Canonical pair:
- Audio: `data/audio/PRIMARY/llm_recording_pranay.m4a`
- Text: `data/text/PRIMARY/llm.txt`

## Artifacts

- `runs/hf_sprint_2026q1/preprocess_matrix/matrix_20260221_170134.md`
- `runs/hf_sprint_2026q1/preprocess_matrix/matrix_20260221_170418.md`

## Results Summary

### faster_whisper

- `none`: WER 0.241, CER 0.061, RTF 0.207
- `trim_silence`: WER 0.223, CER 0.047, RTF 0.094
- `normalize_loudness`: WER 0.241, CER 0.061, RTF 0.098
- `trim_silence,normalize_loudness`: WER 0.223, CER 0.047, RTF 0.092

### faster_distil_whisper_large_v3

- `none`: WER 0.164, CER 0.046, RTF 0.413
- `trim_silence,normalize_loudness`: WER 0.139, CER 0.028, RTF 0.409

## Policy Decision

- Default preprocessing for ASR benchmarks: `trim_silence`
- Optional high-quality profile: `trim_silence,normalize_loudness`
- Do not use `normalize_loudness` alone as a default.

## Pending Validation

- Verify policy on second paired dataset (`ux_primary`) before global default rollout.
- Re-run after MLX/ONNX runtime fixes to confirm no runtime-specific regressions.
