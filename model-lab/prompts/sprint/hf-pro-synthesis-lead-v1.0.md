# HF Pro Sprint - Synthesis Lead Prompt v1.0

## Purpose

Consolidate cross-agent outputs into ranked recommendations and decision-ready artifacts.

## Use When

Use this prompt after executor queues have been run and QA has triaged blockers.

## Non-Negotiable Rules

1. Use produced artifacts only; no unsupported claims.
2. Flag missing evidence explicitly.
3. Separate quality vs latency tradeoffs.
4. Produce recommendation + fallback per capability.

## Inputs

1. `runs/hf_sprint_2026q1/reports/task_results.csv`
2. `runs/hf_sprint_2026q1/reports/summary.md`
3. `runs/hf_sprint_2026q1/execution/*/ledger.jsonl`

## Preconditions

```bash
uv run python scripts/hf_sprint_report.py \
  --execution-root runs/hf_sprint_2026q1/execution \
  --out-dir runs/hf_sprint_2026q1/reports \
  --sprint-id hf_pro_2026q1
```

## Steps

1. Build capability-level ranking tables (asr, asr_stream, vad, diarization, tts, others where available).
2. Identify:
   - best quality model
   - best latency model
   - balanced pick
3. For each capability, provide:
   - primary recommendation
   - fallback recommendation
   - blocker notes
4. Write final synthesis to:
   `runs/hf_sprint_2026q1/reports/final_recommendations.md`

## Required Output Format

1. `Capability Recommendations`
2. `Tradeoff Notes`
3. `Open Gaps Before March 1, 2026`
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop when final recommendations are written and path is shared.
