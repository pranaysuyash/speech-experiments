# HF Pro Sprint Runbook (2026 Q1)

Sprint window in config: **2026-02-12 to 2026-03-01** (`config/hf_sprint_2026q1.yaml`).

## 0) Security (Do This First)

If an HF token was ever pasted into chat/issue/email, treat it as compromised:
1. Revoke it in Hugging Face settings.
2. Create a new token.
3. Update your local `.env`.

This repo ignores `.env` via `.gitignore`. Use `.env.example` as a template.

## 1) One-Time Setup

```bash
cd /Users/pranay/Projects/speech_experiments/model-lab
uv sync
```

Optional: install extra deps to cover more sprint models (opt-in).

```bash
# Lighter installs (still non-trivial): demucs/laion-clap/onnxruntime
uv sync --extra hf_sprint_light

# Heavy installs: NeMo + TensorFlow (may be slow / platform-sensitive)
uv sync --extra hf_sprint_heavy
```

Optional but recommended for sprint reproducibility:

```bash
mkdir -p .huggingface
```

## 2) Preflight Checks

Preflight catches missing tools, missing dataset files, missing HF auth, and missing python deps for known heavy models:

```bash
uv run python scripts/hf_sprint_preflight.py --config config/hf_sprint_2026q1.yaml
```

Notes:
- Some models in the sprint are intentionally "heavy" (TensorFlow, NeMo, ONNX runtime). If you don't install those deps, tasks will be skipped as `skipped_prereq` by the worker.
- `pyannote_diarization` requires HF auth (`HF_TOKEN` or `HUGGINGFACE_HUB_TOKEN`) and often requires accepting the model terms on the HF model page.
- `deepfilternet` currently conflicts with this repo's `numpy>=2.x` constraints; treat it as "isolated venv only" or leave it skipped.

## 3) Generate Plan

```bash
uv run python scripts/hf_sprint_plan.py \
  --config config/hf_sprint_2026q1.yaml \
  --output-dir runs/hf_sprint_2026q1
```

Outputs:
- `runs/hf_sprint_2026q1/agent_queues/*.json`
- `runs/hf_sprint_2026q1/dispatch_commands.sh`
- `runs/hf_sprint_2026q1/assignment_matrix.csv`

## 4) Run (Parallel)

Single-command orchestrator:

```bash
uv run python scripts/hf_sprint_run_all.py --preflight
```

Recommended for stability (per-task timeout to avoid queue stalls):

```bash
uv run python scripts/hf_sprint_run_all.py --preflight --task-timeout-sec 1800
```

Dry run (no execution, just ledger entries):

```bash
uv run python scripts/hf_sprint_run_all.py --preflight --dry-run
```

## 5) Report

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

Key artifacts:
- `runs/hf_sprint_2026q1/reports/task_results.csv`
- `runs/hf_sprint_2026q1/reports/summary.md`

## 6) Optional: Prefetch HF Hub Assets (Reduce March 1 Risk)

This warms the HF cache for sprint models without pulling multi-GB weights by default:

```bash
uv run python scripts/hf_prefetch.py --strategy small
```

Report output:
- `runs/hf_sprint_2026q1/prefetch/prefetch.json`

## 7) Frontier Scan (Transformers / MLX / GGUF / llama.cpp)

Scan Hub for newly active speech/realtime repos and runtime compatibility tags:

```bash
uv run python scripts/hf_frontier_scan.py --limit 60
```

Outputs:
- `runs/hf_sprint_2026q1/frontier/frontier_scan.json`
- `runs/hf_sprint_2026q1/frontier/frontier_scan.md`

Use this before each sprint pass to pick candidate additions (especially realtime and Voxtral-adjacent repos).

## 8) Daily Cadence Through March 1, 2026

Run this sequence once daily (or twice on high-activity days):

```bash
# 1) Refresh model landscape and warm cache
uv run python scripts/hf_frontier_scan.py --limit 60
uv run python scripts/hf_prefetch.py --strategy small

# 2) Plan and execute with preflight + task timeout guard
uv run python scripts/hf_sprint_run_all.py --preflight --task-timeout-sec 1800

# 3) Review report
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```
