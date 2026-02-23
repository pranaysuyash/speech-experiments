# HF Research Tracker

Last refreshed: 2026-02-21

## Purpose

Maintain a dated, reproducible view of new speech/realtime model activity relevant to:
- Transformers runtime
- MLX runtime (Apple Silicon)
- GGUF / llama.cpp runtime
- Realtime ASR benchmarks (Voxtral and adjacent)

## Source of Truth

- Frontier scan artifact:
  - `runs/hf_sprint_2026q1/frontier/frontier_scan.json`
  - `runs/hf_sprint_2026q1/frontier/frontier_scan.md`

## Current Watchlist (2026-02-21)

Prioritized from `runs/hf_sprint_2026q1/frontier/frontier_scan.json` using:
- speech pipeline relevance (`automatic-speech-recognition`/streaming tags)
- runtime lane coverage (transformers, mlx, gguf/llama.cpp)
- usage signals (downloads, likes)

### Add Now (high signal)

- `mistralai/Voxtral-Mini-4B-Realtime-2602` (official realtime baseline)
- `mistralai/Voxtral-Small-24B-2507` (larger official Voxtral tier)
- `speechbrain/asr-streaming-conformer-gigaspeech` (streaming conformer lane)
- `speechbrain/asr-streaming-conformer-librispeech` (streaming conformer lane)

### Track for Runtime Comparison

- `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`
- `mlx-community/Voxtral-Mini-4B-Realtime-6bit`
- `mlx-community/Voxtral-Mini-4B-Realtime-2602-fp16`
- `mlx-community/whisper-small.en-asr-fp16`

### GGUF / llama.cpp Evaluation Lane

- `xkeyC/whisper-large-v3-turbo-gguf`
- `oxide-lab/whisper-large-v3-GGUF`
- `andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF`

## Update Workflow

1. Refresh frontier scan:
```bash
uv run python scripts/hf_frontier_scan.py --limit 60
```
2. Run official-vs-community runtime matrix benchmark:
```bash
uv run python scripts/runtime_matrix_benchmark.py --dataset asr_smoke_v1 --stream-dataset asr_smoke_v1
```
2. Refresh cache (small assets):
```bash
uv run python scripts/hf_prefetch.py --strategy small
```
3. Run sprint pass:
```bash
uv run python scripts/hf_sprint_run_all.py --preflight --task-timeout-sec 1800
```
4. Run preprocessing matrix on the canonical user sample:
```bash
make asr-preprocess-matrix AUDIO=data/audio/PRIMARY/llm_recording_pranay.m4a TEXT=data/text/PRIMARY/llm.txt PRE="trim_silence trim_silence,normalize_loudness"
```
5. Review:
  - `runs/hf_sprint_2026q1/reports/summary.md`
  - `runs/hf_sprint_2026q1/reports/task_results.csv`
  - `runs/hf_sprint_2026q1/frontier/runtime_matrix.md`
  - `runs/hf_sprint_2026q1/preprocess_matrix/`

## External References

- Transformers Voxtral docs: https://huggingface.co/docs/transformers/main/en/model_doc/voxtral
- Transformers Voxtral Realtime docs: https://huggingface.co/docs/transformers/main/en/model_doc/voxtral_realtime
- Voxtral Realtime model card: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
