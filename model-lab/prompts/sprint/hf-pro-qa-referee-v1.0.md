# HF Pro Sprint - QA Referee Prompt v1.0

## Purpose

Audit sprint execution quality across all agents and produce rerun priorities.

## Use When

Use this prompt after one or more executor agents have produced ledgers.

## Non-Negotiable Rules

1. Do not run model tasks unless explicitly requested.
2. Evaluate only from ledger/log/artifact evidence.
3. Keep findings severity-ranked.
4. Separate `Observed` facts from inferred causes.

## Inputs

1. Ledgers: `runs/hf_sprint_2026q1/execution/*/ledger.jsonl`
2. Logs: `runs/hf_sprint_2026q1/execution/*/logs/*.log`
3. Aggregate report: `runs/hf_sprint_2026q1/reports/summary.md`

## Preconditions

```bash
ls -la runs/hf_sprint_2026q1/execution/*/ledger.jsonl
uv run python scripts/hf_sprint_report.py --execution-root runs/hf_sprint_2026q1/execution --out-dir runs/hf_sprint_2026q1/reports --sprint-id hf_pro_2026q1
```

## Steps

1. Count failures by agent and capability.
2. Group failures into:
   - environment/setup
   - model integration
   - data/input
   - transient/runtime
3. Produce rerun list with priority:
   - P0: blocks baseline recommendation
   - P1: narrows candidate comparison
   - P2: optional or low-impact
4. Recommend exact rerun commands using `hf_sprint_worker.py --force-rerun`.

## Required Output Format

1. `Findings` (severity ordered, with paths)
2. `Rerun Queue` (P0/P1/P2)
3. `Risk to March 1 Deadline`
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop when rerun priorities are explicit and command-ready.
