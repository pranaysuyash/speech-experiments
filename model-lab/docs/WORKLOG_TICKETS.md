# Worklog Tickets (append-only)

This file is the canonical, append-only registry for work items discovered by audits, reviews, or agents.

Guidelines:
- Always append new tickets to the top.
- Do not rewrite previous entries; add a new status update entry instead.
- Include evidence, owner, and acceptance criteria.
- Link back to the source audit or report.

---

### TCK-20260204-003 :: Add comprehensive audit prompt

Type: [IMPROVEMENT]
Owner: GitHub Copilot (agent)
Created: 2026-02-04
Status: **OPEN**
Priority: P2

Scope contract:
- In-scope: Add `prompts/audit/comprehensive-audit-v1.0.md` and update `prompts/README.md`
- Out-of-scope: No code changes or other prompts
- Behavior change allowed: NO

Acceptance Criteria:
- [ ] Comprehensive audit prompt added with full content
- [ ] Prompts README updated to include the new audit prompt

Source:
- Request: User provided prompt text for comprehensive product audit
- Evidence: File created at `prompts/audit/comprehensive-audit-v1.0.md`

Execution log:
- 2026-02-04 11:00 Added prompt file and updated index

Status updates:
- 2026-02-04 11:00 **OPEN** — Ticket created

---

### TCK-20260204-002 :: Expand project-management scaffold to comprehensive setup

Type: [IMPROVEMENT]
Owner: GitHub Copilot (agent)
Created: 2026-02-04
Status: **OPEN**
Priority: P1

Scope contract:
- In-scope: Add full workflow prompts, process docs, scripts, hooks, and merge coordination rules into AGENTS.md
- Out-of-scope: Code behavior changes or new features
- Behavior change allowed: NO

Acceptance Criteria:
- [ ] Add agent-entrypoint, project-setup-verification, repo-hygiene-sweep prompts
- [ ] Add PROMPT_STYLE_GUIDE.md, PROCESS_REMINDER.md, ISSUES_WORKFLOW.md
- [ ] Add worklog_checker.sh script and .githooks/pre-commit
- [ ] Merge full coordination rules into docs/AGENTS.md
- [ ] Update prompts/README.md with all new entries

Source:
- Inspiration: Full `learning_for_kids` project-management setup
- Evidence: Files created and updated as listed

Execution log:
- 2026-02-04 10:00 Expanded scaffold with comprehensive artifacts

Status updates:
- 2026-02-04 10:00 **OPEN** — Ticket created for expansion

---

### TCK-20260204-001 :: Scaffold project-management artifacts

Type: [IMPROVEMENT]
Owner: GitHub Copilot (agent)
Created: 2026-02-04
Status: **OPEN**
Priority: P2

Scope contract:
- In-scope: `prompts/`, `docs/WORKLOG_TICKETS.md`, `docs/audit/`, `docs/process/*`, `scripts/*`, `.github/PULL_REQUEST_TEMPLATE.md`
- Out-of-scope: code behavior changes outside scaffolding
- Behavior change allowed: NO

Acceptance Criteria:
- [ ] Add `prompts/README.md` and `prompts/workflow/*` minimal prompts
- [ ] Add `docs/WORKLOG_TICKETS.md` (this file) and initial ticket
- [ ] Add `docs/audit/` skeleton and `docs/process/CODE_PRESERVATION_GUIDELINES.md`
- [ ] Add `scripts/audit_review.sh` and `scripts/setup-githooks.sh`
- [ ] Add PR template referencing worklog tickets

Source:
- Action: repository scaffolding inspired by `learning_for_kids` project
- Evidence: creation of files listed in scope

Execution log:
- 2026-02-04 09:00 Created scaffold files locally

Status updates:
- 2026-02-04 09:00 **OPEN** — Ticket created

---

### TCK-2026MMDD-001 :: Example ticket (template)

Type: [AUDIT_FINDING | BUG | FEATURE | IMPROVEMENT]
Owner: Pranay
Created: 2026-02-04
Status: **OPEN**
Priority: P2

Scope contract:
- In-scope: docs/AGENTS.md
- Out-of-scope: unrelated docs or code changes
- Behavior change allowed: NO

Acceptance Criteria:
- [ ] Add worklog template files
- [ ] Add audit review script

Source:
- Audit file: docs/REALIGNMENT_SNAPSHOT.md
- Finding ID: Example
- Evidence: See audit

Execution log:
- 2026-02-04 09:00 Ticket created by agent

Status updates:
- 2026-02-04 09:00 **OPEN** — Created
