# HF Pro Sprint - Edge Small Executor Prompt v1.0

## Purpose

Execute the `edge_small` queue for compact/edge model candidates during the HF Pro sprint.

## Use When

Use this prompt when you are assigned the edge and small-model lane.

## Non-Negotiable Rules

1. Work on the current branch only. Do not create new branches.
2. Do not delete or revert unrelated files.
3. Run queue tasks through `scripts/hf_sprint_worker.py` only.
4. Keep append-only evidence via ledger/log files.
5. Continue past single-task failures and capture root cause.

## Inputs

1. Queue: `runs/hf_sprint_2026q1/agent_queues/edge_small.json`
2. Execution root: `runs/hf_sprint_2026q1/execution`
3. Sprint report: `runs/hf_sprint_2026q1/reports/summary.md`

## Preconditions

Run:

```bash
uv run python scripts/hf_sprint_plan.py --config config/hf_sprint_2026q1.yaml --output-dir runs/hf_sprint_2026q1
```

Verify queue exists:

```bash
ls -la runs/hf_sprint_2026q1/agent_queues/edge_small.json
```

## Steps

1. Execute queue:

```bash
uv run python scripts/hf_sprint_worker.py \
  --queue runs/hf_sprint_2026q1/agent_queues/edge_small.json \
  --execution-root runs/hf_sprint_2026q1/execution \
  --continue-on-error
```

2. Generate updated rollup:

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

3. Summarize top failures for handoff from:
   `runs/hf_sprint_2026q1/execution/edge_small/ledger.jsonl`

## Required Output Format

Provide:

1. `Execution Summary` (executed, failed, skipped).
2. `Top Failures` (task_id + short cause).
3. `Artifacts Produced` (paths).
4. `Evidence Snippet` with:
   - `Observed`
   - `Inferred`
   - `Unknown`

## Stop Condition

Stop when queue is fully attempted and report regenerated.
