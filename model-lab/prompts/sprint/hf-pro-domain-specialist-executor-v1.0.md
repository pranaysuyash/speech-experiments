# HF Pro Sprint - Domain Specialist Executor Prompt v1.0

## Purpose

Execute specialist-capability models (VAD, diarization, enhancement, separation, classify/embed).

## Use When

Use this prompt when assigned the `domain_specialist` lane.

## Non-Negotiable Rules

1. Use the generated queue as source of truth.
2. Do not change scoring rules while executing tasks.
3. Capture all failures in ledger/log files.
4. If a task is marked manual, do not fabricate results.

## Inputs

1. Queue: `runs/hf_sprint_2026q1/agent_queues/domain_specialist.json`
2. Execution root: `runs/hf_sprint_2026q1/execution`
3. Report dir: `runs/hf_sprint_2026q1/reports`

## Preconditions

```bash
uv run python scripts/hf_sprint_plan.py --config config/hf_sprint_2026q1.yaml --output-dir runs/hf_sprint_2026q1
ls -la runs/hf_sprint_2026q1/agent_queues/domain_specialist.json
```

## Steps

1. Run queue:

```bash
uv run python scripts/hf_sprint_worker.py \
  --queue runs/hf_sprint_2026q1/agent_queues/domain_specialist.json \
  --execution-root runs/hf_sprint_2026q1/execution \
  --continue-on-error
```

2. Refresh report:

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

3. Extract specialist blockers from:
   `runs/hf_sprint_2026q1/execution/domain_specialist/ledger.jsonl`

## Required Output Format

1. `Execution Summary`
2. `Capability Notes` (vad/diarization/enhance/separate/classify/embed)
3. `Manual Tasks Pending` (if any)
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop once all queue tasks are attempted and report is updated.
