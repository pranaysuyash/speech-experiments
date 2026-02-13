# HF Pro Sprint: Multi-Agent + Multi-LLM Workflow

**Sprint window:** February 12, 2026 to March 1, 2026  
**Goal:** Use the remaining Hugging Face Pro time to run the broadest high-signal benchmark sweep possible, with parallel execution and strict artifact discipline.

## What Is Implemented

The repo now includes an operational sprint stack:

1. `config/hf_sprint_2026q1.yaml`
2. `scripts/hf_sprint_plan.py`
3. `scripts/hf_sprint_worker.py`
4. `scripts/hf_sprint_report.py`
5. `scripts/run_asr_stream.py`

This enables:

1. Automatic model-to-agent assignment from the live `harness.registry` catalog.
2. Per-agent task queue generation (JSON + Markdown).
3. Reproducible sequential execution with ledger + logs.
4. Consolidated cross-agent report generation.
5. Streaming-ASR models evaluated with the same process as batch models.

## Agent Topology

Configured agents in `config/hf_sprint_2026q1.yaml`:

1. `edge_small`
2. `domain_specialist`
3. `realtime_streaming`
4. `general_baselines`

Recommended extra coordination agents (human or LLM):

1. `qa_referee` (artifact validation, outlier triage)
2. `synthesis` (daily leaderboard and final recommendation pack)

## Date-Based Plan

## February 12, 2026

1. Generate plan + queues.
2. Validate all worker commands with small task caps.
3. Fix immediate environment blockers.

## February 13-15, 2026

1. Run smoke sweep across all queues.
2. Track fail classes: load failures, runtime crashes, schema gaps.
3. Patch runner issues quickly and rerun failed smoke tasks.

## February 16-22, 2026

1. Run primary dataset passes for ASR and ASR-streaming.
2. Run specialist capability tests (VAD/diarization/TTS/enhancement/etc.).
3. Produce daily summary and top-candidate shortlist.

## February 23-27, 2026

1. Deep-dive on top models only.
2. Re-test with secondary dataset (`ux_primary`) and robustness inputs.
3. Lock recommendation candidates and fallback options.

## February 28, 2026

1. Freeze final evidence.
2. Generate final sprint report + leaderboard.
3. Package artifacts for handoff and post-Pro operation.

## March 1, 2026

1. Final review and model selection.
2. Archive all run outputs and report bundle.

## Execution Commands

Generate plan and queue files:

```bash
uv run python scripts/hf_sprint_plan.py \
  --config config/hf_sprint_2026q1.yaml \
  --output-dir runs/hf_sprint_2026q1
```

Run one agent queue:

```bash
uv run python scripts/hf_sprint_worker.py \
  --queue runs/hf_sprint_2026q1/agent_queues/edge_small.json \
  --execution-root runs/hf_sprint_2026q1/execution \
  --continue-on-error
```

Generate consolidated report:

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

## Handoff Contract (All Agents)

Each task must produce:

1. Queue task ID.
2. Exit status and runtime.
3. `ARTIFACT_PATH` when available.
4. Stdout/stderr logs captured by worker.

Do not mark a task complete without an artifact path or explicit failure reason.

## Multi-LLM Prompt Templates

Use these prompts with different LLM agents in parallel.

### Prompt: Edge/Small Agent

```
You own the `edge_small` queue.
Run tasks from: runs/hf_sprint_2026q1/agent_queues/edge_small.json
Use: uv run python scripts/hf_sprint_worker.py --queue <queue> --execution-root runs/hf_sprint_2026q1/execution --continue-on-error
Focus on low-footprint models and report latency/RTF plus failure patterns.
Do not edit scoring logic. Only fix runner breakages that block execution.
```

### Prompt: Domain Specialist Agent

```
You own the `domain_specialist` queue.
Run tasks from: runs/hf_sprint_2026q1/agent_queues/domain_specialist.json
Use the sprint worker and keep artifact/log discipline.
Prioritize diarization/VAD/enhancement correctness and identify unsupported capability gaps.
```

### Prompt: Realtime Agent

```
You own the `realtime_streaming` queue.
Run tasks from: runs/hf_sprint_2026q1/agent_queues/realtime_streaming.json
Pay special attention to first-token latency and real-time factor.
If stream models fail, capture root cause in worker logs and move to next task.
```

### Prompt: Baseline Agent

```
You own the `general_baselines` queue.
Run tasks from: runs/hf_sprint_2026q1/agent_queues/general_baselines.json
Prioritize quality-first baselines and ensure primary/secondary ASR datasets complete.
```

### Prompt: QA Referee Agent

```
Audit ledgers in runs/hf_sprint_2026q1/execution/*/ledger.jsonl.
Find failed tasks, missing artifacts, and suspicious metrics.
Propose reruns only for high-impact failures.
```

### Prompt: Synthesis Agent

```
Generate sprint rollup using scripts/hf_sprint_report.py.
Produce top-model recommendations per capability with fallback choices.
Call out models with good quality but poor latency and vice versa.
```

## Parallel Operating Rules

1. One queue file per execution agent.
2. One ledger per agent (`execution/<agent>/ledger.jsonl`).
3. Never overwrite another agent's ledger.
4. Keep reruns explicit (`--force-rerun`) to preserve audit trail.
5. Regenerate report after any major rerun batch.

## Known Blocker (Observed)

1. `kyutai_streaming` currently fails in `run_asr_stream.py` because the adapter cannot instantiate:
   `KyutaiStreamingAdapter` is missing implementations for abstract methods.
2. Treat this as a model integration bug, not a sprint framework failure.
3. Capture and track this in the realtime queue ledger, then continue with other tasks.
