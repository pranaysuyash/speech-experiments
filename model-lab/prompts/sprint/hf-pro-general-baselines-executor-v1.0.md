# HF Pro Sprint - General Baselines Executor Prompt v1.0

## Purpose

Execute high-signal baseline models and complete primary/secondary ASR evidence.

## Use When

Use this prompt when assigned `general_baselines`.

## Non-Negotiable Rules

1. Complete baseline ASR sweeps before optional tasks.
2. Keep strict artifact discipline via worker ledger.
3. Do not edit queue ordering unless instructed.
4. Preserve reproducibility of commands and outputs.

## Inputs

1. Queue: `runs/hf_sprint_2026q1/agent_queues/general_baselines.json`
2. Execution root: `runs/hf_sprint_2026q1/execution`
3. Reports: `runs/hf_sprint_2026q1/reports`

## Preconditions

```bash
uv run python scripts/hf_sprint_plan.py --config config/hf_sprint_2026q1.yaml --output-dir runs/hf_sprint_2026q1
ls -la runs/hf_sprint_2026q1/agent_queues/general_baselines.json
```

## Steps

1. Execute queue:

```bash
uv run python scripts/hf_sprint_worker.py \
  --queue runs/hf_sprint_2026q1/agent_queues/general_baselines.json \
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

3. Produce shortlist candidates for follow-up deep dive.

## Required Output Format

1. `Execution Summary`
2. `Baseline Candidate Table` (model, capability, key metrics available)
3. `Failures and Rerun Recommendations`
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop when queue is fully attempted and shortlist is prepared.
