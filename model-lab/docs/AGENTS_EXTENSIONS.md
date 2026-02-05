# Agent Coordination — Extensions for Project Management

This file augments `docs/AGENTS.md` with additional coordination rules, workflows, and required artifacts inspired by a mature project-management setup.

## Core Principles

### Evidence-First Development

- Back claims with evidence: `Observed`, `Inferred`, or `Unknown`.
- Preserve evidence in audit artifacts and worklog tickets.

### Single Source of Truth

- **Worklog**: `docs/WORKLOG_TICKETS.md` — canonical, append-only ticket registry.
- **Audits**: `docs/audit/` — one audit file per audit.
- **Prompts**: `prompts/` — repository prompt library and `prompts/README.md` index.

### Preservation First

- Investigate before deleting. Follow `docs/process/CODE_PRESERVATION_GUIDELINES.md`.
- Prefer activation or archiving over deletion; document decisions in an audit and create a ticket before removal.

## Agent Workflow (overview)

1. Intake
   - Determine work type and define scope contract (in-scope, out-of-scope, behavior change allowed).
   - Pick the correct prompt from `prompts/README.md` and record it.
   - Create or update a worklog ticket in `docs/WORKLOG_TICKETS.md` BEFORE implementing.

2. Execution
   - Follow the prompt; run tests and verification steps; preserve unrelated changes.

3. Documentation
   - Produce: worklog entry, audit artifact (if applicable), evidence log, and prompt traceability.

## Audit-to-Ticket Workflow

- Create a ticket for every actionable audit finding before remediation.
- Use `prompts/workflow/worklog-v1.0.md` ticket template and link evidence.

## Required Artifacts for Work Units

- Worklog ticket in `docs/WORKLOG_TICKETS.md` (append-only)
- Audit file `docs/audit/<file>.md` for audits
- Evidence log (command outputs, test runs)
- Prompt(s) and persona/roles used for traceability

---

This file is intentionally additive to `docs/AGENTS.md` to keep the primary file small and focused. Consider merging contents into `docs/AGENTS.md` after review.
