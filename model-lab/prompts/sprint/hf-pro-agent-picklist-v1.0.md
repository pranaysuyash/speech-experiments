# HF Pro Sprint - Agent Picklist Prompt v1.0

## Purpose

Let an agent pick one clear role and start immediately with the correct prompt.

## Use When

Use this as the first prompt when onboarding multiple agents in parallel.

## Role Options

Pick exactly one:

1. `edge_small_executor` -> `prompts/sprint/hf-pro-edge-small-executor-v1.0.md`
2. `domain_specialist_executor` -> `prompts/sprint/hf-pro-domain-specialist-executor-v1.0.md`
3. `realtime_streaming_executor` -> `prompts/sprint/hf-pro-realtime-streaming-executor-v1.0.md`
4. `general_baselines_executor` -> `prompts/sprint/hf-pro-general-baselines-executor-v1.0.md`
5. `qa_referee` -> `prompts/sprint/hf-pro-qa-referee-v1.0.md`
6. `synthesis_lead` -> `prompts/sprint/hf-pro-synthesis-lead-v1.0.md`
7. `streaming_integration_fixer` -> `prompts/sprint/hf-pro-streaming-integration-fixer-v1.0.md`

## Startup Command

Always refresh plan first:

```bash
uv run python scripts/hf_sprint_plan.py --config config/hf_sprint_2026q1.yaml --output-dir runs/hf_sprint_2026q1
```

Then run the selected role prompt exactly.

## Required Output

1. `Selected Role`
2. `Prompt Path`
3. `First Command You Will Run`
4. `Evidence Snippet` (`Observed`, `Inferred`, `Unknown`)

## Stop Condition

Stop after role selection + first command confirmation, then switch to the selected role prompt.
