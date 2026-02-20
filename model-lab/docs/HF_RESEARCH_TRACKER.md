# HF Research Tracker

Last refreshed: 2026-02-20

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

## Current Watchlist (2026-02-20)

- `mistralai/Voxtral-Mini-4B-Realtime-2602`
- `mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit`
- `mlx-community/Voxtral-Mini-4B-Realtime-fp16`
- `andrijdavid/Voxtral-Mini-4B-Realtime-2602-GGUF`
- `xkeyC/whisper-large-v3-turbo-gguf`
- `mlx-community/whisper-small.en-asr-fp16`

## Update Workflow

1. Refresh frontier scan:
```bash
uv run python scripts/hf_frontier_scan.py --limit 60
```
2. Refresh cache (small assets):
```bash
uv run python scripts/hf_prefetch.py --strategy small
```
3. Run sprint pass:
```bash
uv run python scripts/hf_sprint_run_all.py --preflight --task-timeout-sec 1800
```
4. Review:
  - `runs/hf_sprint_2026q1/reports/summary.md`
  - `runs/hf_sprint_2026q1/reports/task_results.csv`

## External References

- Transformers Voxtral docs: https://huggingface.co/docs/transformers/main/en/model_doc/voxtral
- Transformers Voxtral Realtime docs: https://huggingface.co/docs/transformers/main/en/model_doc/voxtral_realtime
- Voxtral Realtime model card: https://huggingface.co/mistralai/Voxtral-Mini-4B-Realtime-2602
