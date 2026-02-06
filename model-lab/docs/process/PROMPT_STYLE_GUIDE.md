# Prompt Style Guide (Repo-Native)

This guide helps keep prompts consistent, testable, and aligned with AGENTS.md.

## Core Rules

- **Evidence-first**: require Observed/Inferred/Unknown labels for non-trivial claims.
- **Scope discipline**: one work unit per run; new findings → new ticket.
- **Preservation-first**: refactor in place; avoid `*_v2` parallel implementations.
- **Append-only tracking**: `docs/WORKLOG_TICKETS.md` is canonical.
- **Model claims require evidence**: benchmarks must include hardware, dataset, and reproducible command.

## Recommended Prompt Skeleton

1) Purpose
2) Use When
3) Non-negotiable rules
4) Inputs (explicit)
5) Preconditions (if any)
6) Steps with required commands
7) Output format (required sections)
8) Stop condition (hard)

## Prompting Techniques We Use (Practical)

- **Gates**: "don't code before ticket + plan"
- **Checklists**: required commands; required artifacts; required outputs
- **Rubrics**: PASS/FAIL criteria for completeness
- **Structured output**: headings + bullet lists; consistent templates
- **Test scenarios**: tabletop cases to validate the prompt design
- **Model-specific context**: always specify which models/hardware are in scope

## Anti-Patterns (Avoid)

- "Do whatever you think is best" without scope
- Prompts that allow silent scope creep
- Prompts that don't demand verification evidence
- Copy/pasting external prompts verbatim (use patterns, not phrasing)
- Model performance claims without reproducible evidence

## Evidence Snippet Template

- Every prompt-driven artifact must include an evidence snippet (e.g., in the plan, worklog, reality check) that labels the most consequential claims as **Observed**, **Inferred**, or **Unknown**.
- Template:

```
**Evidence snippet**:
- **Observed**: [fact, e.g., "Whisper large-v3 achieved 2.5% WER on test-clean"]
- **Inferred**: [logical deduction, e.g., "MPS backend should work on M2 chips"]
- **Unknown**: [gaps, e.g., "CUDA performance on RTX 4090 untested"]
```

- Place this snippet near the conclusion or findings section so reviewers know what evidence supports each assertion.

## Model-Specific Guidelines

**This is an open exploration lab.** Any AI model capability is in scope - don't artificially limit to common tasks.

When writing prompts that involve model evaluation:

1. Always specify target models by exact name (e.g., `whisper-large-v3`, `musicgen-large`, `demucs`)
2. Specify domain broadly (audio, vision, video, multimodal, generative, scientific, etc.)
3. Be specific about the task (not "audio model" but "music generation" or "audio separation")
4. Require hardware specification (MPS, CUDA, CPU)
5. Require input/dataset specification
6. Require reproducible command that can be re-run
7. Log results to `docs/CLAIMS.md` for cross-agent consistency
8. Use appropriate metrics for the task (see `docs/CLAIMS.md` for extensive reference)
9. Define custom metrics if the task is novel - document how they're computed
10. **Explore freely** - if a model does something interesting, evaluate it

## Related Prompts

- Pre-flight: `prompts/workflow/pre-flight-check-v1.0.md`
- Prompt QA: `prompts/workflow/prompt-quality-gate-v1.0.md`
- Comprehensive audit: `prompts/audit/comprehensive-audit-v1.0.md`
