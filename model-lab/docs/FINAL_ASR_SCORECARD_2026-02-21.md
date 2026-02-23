# Final ASR Scorecard (2026-02-21)

Scope:
- Models: `faster_whisper`, `faster_distil_whisper_large_v3`
- Inputs:
  - `llm_primary` (`llm_recording_pranay.m4a` + `llm.txt`)
  - `ux_primary` (`ux_psychology_30s.wav` + `ux_psychology_30s.txt`)
- Preprocessing policy:
  - `faster_whisper`: `trim_silence`
  - `faster_distil_whisper_large_v3`: `trim_silence,normalize_loudness`

## Source Artifacts

- `runs/hf_sprint_2026q1/preprocess_matrix/matrix_20260221_170134.json`
- `runs/hf_sprint_2026q1/preprocess_matrix/matrix_20260221_170418.json`
- `runs/hf_sprint_2026q1/preprocess_matrix/matrix_20260221_194435.json`

## Results

| model | dataset | preprocess | WER | CER | RTF |
|---|---|---|---:|---:|---:|
| `faster_whisper` | `llm_primary` | `trim_silence` | 0.223 | 0.047 | 0.094 |
| `faster_whisper` | `ux_primary` | `trim_silence` | 1.345 | 0.849 | 0.472 |
| `faster_distil_whisper_large_v3` | `llm_primary` | `trim_silence,normalize_loudness` | 0.139 | 0.028 | 0.409 |
| `faster_distil_whisper_large_v3` | `ux_primary` | `trim_silence,normalize_loudness` | 1.362 | 0.853 | 1.315 |

## Analysis

- On `llm_primary`, `faster_distil_whisper_large_v3` is more accurate, `faster_whisper` is much faster.
- On `ux_primary`, both models are very poor (WER > 1.34), indicating transcript/pair mismatch or domain mismatch in that dataset.
- Because `ux_primary` quality is invalid for model ranking, do not freeze a global winner from this dataset pair yet.

## Operational Decision (Now)

- Keep production default: `faster_whisper` with `trim_silence`.
- Keep `faster_distil_whisper_large_v3` as quality reference model.
- Quarantine `ux_primary` for transcript alignment audit before using it in promotion decisions.
