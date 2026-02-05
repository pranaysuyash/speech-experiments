# Worklog Update Prompt (worklog-v1.0)

**Purpose:** Ensure every actionable finding is recorded in `docs/WORKLOG_TICKETS.md` before implementation.

Steps:

1. Read `docs/audit/*.md` and `docs/*` for any findings or items that require tracking.
2. For each actionable item, create a ticket entry in `docs/WORKLOG_TICKETS.md` using the ticket template.
3. Use an append-only discipline. Do not modify existing ticket history; only add status updates.
4. Reference evidence (file, line numbers, quote) and the prompt(s) used to discover the issue.
5. If you need to implement a fix, create the ticket first and include the ticket ID in code comments and PR description.

Ticket Template Example:

```
### TCK-YYYYMMDD-001 :: Short descriptive title

Type: [AUDIT_FINDING | BUG | FEATURE | IMPROVEMENT]
Owner: [human or agent name]
Created: [YYYY-MM-DD]
Status: **OPEN**
Priority: [P0 | P1 | P2 | P3]

Scope contract:
- In-scope: [files, behaviors]
- Out-of-scope: [explicit non-goals]
- Behavior change allowed: [YES/NO]

Acceptance Criteria:
- [ ] criterion 1
- [ ] criterion 2

Source:
- Audit file: `docs/audit/<file>.md`
- Finding ID: [X]
- Evidence: [quote or line numbers]

Execution log:
- [timestamp] [action] | Evidence: [command output or link]

Status updates:
- [timestamp] **OPEN** — Ticket created, awaiting implementation
```
