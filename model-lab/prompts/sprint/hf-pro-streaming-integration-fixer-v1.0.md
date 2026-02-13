# HF Pro Sprint - Streaming Integration Fixer Prompt v1.0

## Purpose

Fix streaming-model integration blockers that prevent queue completion.

## Use When

Use this prompt when realtime queue failures show adapter/runtime initialization issues.

## Non-Negotiable Rules

1. Reproduce failure first before patching.
2. Scope changes to streaming integration paths only.
3. Add or update targeted tests for the fix.
4. Do not rewrite unrelated model loaders.

## Inputs

1. Failure evidence from:
   `runs/hf_sprint_2026q1/execution/realtime_streaming/ledger.jsonl`
2. Streaming runner:
   `scripts/run_asr_stream.py`
3. Likely integration code:
   `harness/streaming_asr/`
   `harness/registry.py`

## Preconditions

Reproduce one failing task:

```bash
uv run python scripts/run_asr_stream.py --model kyutai_streaming --dataset asr_smoke_v1 --device cpu --chunk-ms 160
```

## Steps

1. Capture exact exception and locate source.
2. Implement minimal fix.
3. Add/adjust unit test for regression coverage.
4. Re-run:
   - targeted tests
   - failing command above
5. If fixed, request rerun with:

```bash
uv run python scripts/hf_sprint_worker.py \
  --queue runs/hf_sprint_2026q1/agent_queues/realtime_streaming.json \
  --execution-root runs/hf_sprint_2026q1/execution \
  --continue-on-error \
  --force-rerun
```

## Required Output Format

1. `Root Cause`
2. `Patch Summary` (files + why)
3. `Verification` (commands + outcomes)
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop when failing streaming task reproduces as pass or is blocked by an external dependency.
