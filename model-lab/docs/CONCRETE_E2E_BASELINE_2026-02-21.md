# Concrete E2E Baseline (2026-02-21)

Goal: verify one reproducible ASR pipeline end-to-end on the real user sample, then rank candidate models by quality and speed.

## Canonical Test Input
- Audio: `data/audio/PRIMARY/llm_recording_pranay.m4a`
- Transcript: `data/text/PRIMARY/llm.txt`
- Duration: `163.18s` (2m43s)
- Dataset ID: `llm_primary`

## Pipeline Readiness Verdict
- `YES` for one production-usable E2E pipeline today: `scripts/run_asr.py --model faster_whisper --audio data/audio/PRIMARY/llm_recording_pranay.m4a --device mps`
- `PARTIAL` for "all runtimes are ready": MLX runs and performs strongly; ONNX runs but quality remains unacceptable for this English benchmark.

## Fresh Results (same input, same date)

| model | status | WER | CER | RTF | latency_ms | notes |
|---|---|---:|---:|---:|---:|---|
| `mlx_whisper` | pass | 0.127 | 0.028 | 0.052 | 8526.1 | best quality and speed on this sample |
| `faster_distil_whisper_large_v3` | pass | 0.164 | 0.046 | 0.545 | 88945.7 | strong quality, slower |
| `faster_whisper` | pass | 0.242 | 0.061 | 0.305 | 49850.4 | reliable baseline |
| `nb_whisper_small_onnx` | pass_with_risk | 0.981 | 0.639 | 0.866 | 141327.0 | ONNX runtime path works, but wrong-language/poor output on this English sample |

## Analysis
- Winner on this sample is `mlx_whisper` (best WER/CER and best latency).
- `faster_distil_whisper_large_v3` remains a good accuracy reference with higher latency.
- `faster_whisper` remains reliable and balanced.
- `nb_whisper_small_onnx` now runs real ONNX via Optimum, but quality is not production-acceptable for this English scenario.

## Recommendation (freeze now)
- Keep `faster_whisper` as stable default until MLX is validated across both paired datasets.
- Promote `mlx_whisper` to top candidate for immediate expanded benchmarking.
- Keep `faster_distil_whisper_large_v3` as quality reference model.
- Keep `nb_whisper_small_onnx` experimental for Norwegian lanes; do not use for English production lanes.

## Contract Coverage Added
- Added registry coverage test so sprint-config models must be registered.
- Added MLX loader registration to remove silent model onboarding drift.

## Evidence Artifacts
- `runs/faster_whisper/asr/adhoc_1771683694.json`
- `runs/faster_distil_whisper_large_v3/asr/adhoc_1771683735.json`
- `runs/mlx_whisper/asr/adhoc_1771683752.json`
- `runs/nb_whisper_small_onnx/asr/adhoc_1771684318.json`
