# HF Pro Sprint - Realtime Streaming Executor Prompt v1.0

## Purpose

Execute realtime/streaming ASR tasks and capture latency-first evidence.

## Use When

Use this prompt when assigned `realtime_streaming`.

## Non-Negotiable Rules

1. Prioritize `asr_stream` tasks first.
2. Record first-token latency and RTF where available.
3. Continue past model integration failures.
4. Do not hide blockers; log them explicitly.

## Inputs

1. Queue: `runs/hf_sprint_2026q1/agent_queues/realtime_streaming.json`
2. Execution root: `runs/hf_sprint_2026q1/execution`
3. Known issue doc: `docs/HF_PRO_SPRINT_MULTI_AGENT_WORKFLOW_2026-02-12_to_2026-03-01.md`

## Preconditions

```bash
uv run python scripts/hf_sprint_plan.py --config config/hf_sprint_2026q1.yaml --output-dir runs/hf_sprint_2026q1
ls -la runs/hf_sprint_2026q1/agent_queues/realtime_streaming.json
```

## Steps

1. Execute queue:

```bash
uv run python scripts/hf_sprint_worker.py \
  --queue runs/hf_sprint_2026q1/agent_queues/realtime_streaming.json \
  --execution-root runs/hf_sprint_2026q1/execution \
  --continue-on-error
```

2. Regenerate report:

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

3. Explicitly flag failures involving adapter/runtime initialization.

## Required Output Format

1. `Execution Summary`
2. `Latency Findings` (first-token, RTF-like notes)
3. `Blockers` (task_id -> error)
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop when all queue tasks are attempted and blockers are documented.
