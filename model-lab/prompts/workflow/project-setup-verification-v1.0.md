# Project Setup Verification (project-setup-verification-v1.0)

**Purpose:** Verify project setup before starting work.

Checklist:

- Git hooks enabled: `git config core.hooksPath .githooks` (run `./scripts/setup-githooks.sh` if needed).
- Worklog exists: `docs/WORKLOG_TICKETS.md` is present and append-only.
- Prompts indexed: `prompts/README.md` lists available prompts.
- Audit skeleton: `docs/audit/` exists with at least `.gitkeep`.
- Scripts executable: `scripts/audit_review.sh` and `scripts/setup-githooks.sh` are executable.
- PR template: `.github/PULL_REQUEST_TEMPLATE.md` references worklog tickets.

If any fail, create a ticket in `docs/WORKLOG_TICKETS.md` to fix.
