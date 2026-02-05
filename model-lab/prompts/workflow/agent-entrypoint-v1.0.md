# Agent Entrypoint Prompt (agent-entrypoint-v1.0)

**Purpose:** Standardize how agents start work on this repo.

Steps:

1. Determine work type: audit, remediation, hardening, PR review, verification, merge conflict, post-merge, triage.
2. Define scope contract: target files/scope, behavior change allowed (YES/NO), in-scope/out-of-scope, acceptance criteria.
3. Select prompt: Use `prompts/README.md` to find the appropriate prompt for the work type.
4. Ticket action: Create/update `docs/WORKLOG_TICKETS.md` (append-only) BEFORE implementing.
5. Execute: Follow the selected prompt, preserve unrelated changes, run verification.
6. Document: Update worklog, produce audit artifact if applicable, log evidence.

Critical Rules:

- NEVER create new git branches unless explicitly asked by the user.
- NEVER delete or revert files with unrecognized changes (preserve parallel work).
- Stage comprehensively with `git add -A` unless user requests narrow staging.
- Follow preservation-first: investigate before deleting (see `docs/process/CODE_PRESERVATION_GUIDELINES.md`).
