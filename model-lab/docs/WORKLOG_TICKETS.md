# Worklog Tickets (append-only)

This file is the canonical, append-only registry for work items discovered by audits, reviews, or agents.

Guidelines:

- Always append new tickets to the top.
- Do not rewrite previous entries; add a new status update entry instead.
- Include evidence, owner, and acceptance criteria.
- Link back to the source audit or report.

---

### TCK-20260205-001 :: Refactor backend to layered architecture (ARCH-001 remediation)

Type: [IMPROVEMENT]
Owner: GitHub Copilot (agent)
Created: 2026-02-05
Status: **COMPLETED**
Priority: P0

Scope contract:

- In-scope: Refactor server/ to layered architecture (API → Services → Repos). Extract business logic from api/ files into services/.
- Out-of-scope: Frontend changes, new features, database addition
- Behavior change allowed: NO (preserve API contracts)

Acceptance Criteria:

- [ ] server/api/ files contain only routing and request/response handling (<200 lines each)
- [ ] server/services/ contains business logic (e.g., runs_service.py, results_service.py)
- [ ] server/repos/ (new) for data access (if needed, e.g., runs_repo.py for file ops)
- [ ] All existing endpoints work unchanged
- [ ] Unit tests pass for refactored modules

Source:

- Audit: docs/audit/comprehensive-audit-20260204.md (Finding ARCH-001)
- Evidence: server/api/runs.py has 877 lines; mixed concerns

Execution log:

- 2026-02-05 09:00 Ticket created, planning refactor
- 2026-02-05 10:00 Created server/services/runs_service.py with business logic extraction
- 2026-02-05 10:15 Created server/repos/runs_repo.py (placeholder for data access)
- 2026-02-05 10:30 Updated server/api/runs.py to use RunsService for all endpoints
- 2026-02-05 10:45 Fixed syntax error in rerun_pipeline method
- 2026-02-05 11:00 Verified import and method availability

Status updates:

- 2026-02-05 09:00 **OPEN** — Ticket created
- 2026-02-05 11:00 **COMPLETED** — Successfully refactored runs API to layered architecture. All endpoints now use RunsService, preserving API contracts. Ready for next API refactor (results, workbench, etc.)

---

### TCK-20260204-004 :: Conduct comprehensive product audit

Type: [AUDIT]
Owner: GitHub Copilot (agent)
Created: 2026-02-04
Status: **COMPLETED**
Priority: P1

Scope contract:

- In-scope: Full audit per prompts/audit/comprehensive-audit-v1.0.md (repo inventory, product reconstruction, findings, roadmap, research)
- Out-of-scope: Implementation of fixes
- Behavior change allowed: NO

Acceptance Criteria:

- [x] Repo inventory and coverage plan documented
- [x] Product behavior reconstruction (flows, entities)
- [x] Evidence-backed findings with IDs, severity, fixes
- [x] Prioritized roadmap with owners, sequencing, acceptance criteria
- [x] Research appendix with citations
- [x] Audit artifact saved to docs/audit/comprehensive-audit-20260204.md

Source:

- Prompt: prompts/audit/comprehensive-audit-v1.0.md
- Evidence: Audit completed with 10 findings, roadmap phases, research notes

Execution log:

- 2026-02-04 12:00 Started Pass 1 (comprehension)
- 2026-02-04 12:30 Completed Pass 2 (audit + roadmap)
- 2026-02-04 13:00 Audit artifact created

Status updates:

- 2026-02-04 12:00 **IN_PROGRESS** — Audit in progress
- 2026-02-04 13:00 **COMPLETED** — Audit completed and documented

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
- 2026-02-04 11:15 **COMPLETED** — Prompt added and committed

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
- 2026-02-04 11:15 **COMPLETED** — Scaffold expanded and committed

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
- 2026-02-04 11:15 **COMPLETED** — Initial scaffold completed and expanded

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

---

# Ticket: PORT-AUDIO-BENCHMARKS-20260205

Type: IMPROVEMENT
Owner: Pranay
Created: 2026-02-05
Status: **DONE**
Priority: P2

Scope contract:

- In-scope: Port chat-shared audio model research; create machine-readable benchmark tables; keep provenance markers.
- Out-of-scope: Changing scoring logic, model registry wiring, or making benchmark claims "Observed".
- Behavior change allowed: NO (docs + data only)

Acceptance Criteria:

- [x] Store raw research note as a dated doc under `model-lab/docs/`.
- [x] Store extracted explicit benchmark numbers as CSV/JSON under `model-lab/data/`.
- [x] Add lightweight index pointers in `model-lab/README.md` and/or `model-lab/PERFORMANCE_RESULTS.md`.

Source:

- User-provided research note pasted in chat (2026-02-05).

Execution log:

- 2026-02-05 Ported research note + extracted benchmark tables (Reported, unverified).

Status updates:

- 2026-02-05 **DONE** — Docs/data added; no behavior change.

---

# Ticket: PORT-AUDIO-MODEL-AUDIT-FULLTEXT-20260205

Type: IMPROVEMENT  
Owner: Pranay  
Created: 2026-02-05  
Status: **DONE**  
Priority: P2

Scope contract:

- In-scope: Store the full chat-provided “Audio Model Audit (Comprehensive Survey, 2026)” text (and the alternate long-form audit excerpt) under `model-lab/docs/from_chat/`; keep provenance + evidence markers; add an index and pointers.
- Out-of-scope: Verifying any claims, updating model performance “Observed” status, or adding new benchmark runs.
- Behavior change allowed: NO (docs + data + links only)

Acceptance Criteria:

- [x] Store the audit fulltext(s) under `model-lab/docs/from_chat/` with explicit provenance and “Reported” status.
- [x] Add/refresh a `model-lab/docs/from_chat/README.md` index of all chat-captured artifacts.
- [x] Add minimal pointers from `model-lab/PERFORMANCE_RESULTS.md` (and/or `model-lab/README.md`) to the new artifacts.

Source:

- User-provided audit text and citation alternatives pasted in chat (2026-02-05) around “Audio Model Audit (Comprehensive Survey, 2026)”.

Execution log:

- 2026-02-05 Added fulltext chat captures under `model-lab/docs/from_chat/` and updated `model-lab/PERFORMANCE_RESULTS.md` pointers.

Status updates:

- 2026-02-05 **DONE** — Captures + index + pointers added (Reported, unverified)

---

# Ticket: PORT-AUDIO-AI-REVOLUTION-NOTE-20260205

Type: IMPROVEMENT
Owner: Pranay
Created: 2026-02-05
Status: **DONE**
Priority: P2

Scope contract:

- In-scope: Store the final chat-shared “Audio AI Revolution” report; extract explicit numeric claims into CSV/JSON; add index pointers.
- Out-of-scope: Verifying claims, adding web citations, or changing lab scoring/selection behavior.
- Behavior change allowed: NO (docs + data only)

Acceptance Criteria:

- [x] Store raw report note under `model-lab/docs/from_chat/`.
- [x] Extract explicit numeric claims into machine-readable CSV/JSON under `model-lab/data/from_chat/`.
- [x] Add pointers from `model-lab/README.md` and `model-lab/PERFORMANCE_RESULTS.md`.

Source:

- User-provided research note pasted in chat (2026-02-05), labeled “here's the last one”.

Execution log:

- 2026-02-05 Added raw note + extracted numeric claims as **Reported** (unverified).

Status updates:

- 2026-02-05 **DONE** — Docs/data added; no behavior change.

---

# Ticket: AUDIT-ACE-STEP-KUGELAUDIO-20260205

Type: AUDIT  
Owner: Pranay  
Created: 2026-02-05  
Status: **DONE**  
Priority: P2

Scope contract:

- In-scope: Add an evidence-first audit note for **ACE-Step 1.5** and **KugelAudio-0** (aka KugelAudio-0-Open); add/refresh catalog rows in `model-lab/data/model_catalog.csv`; link from `model-lab/PERFORMANCE_RESULTS.md`.
- Out-of-scope: Running benchmarks, validating quality claims, or integrating either model into harness registries.
- Behavior change allowed: NO (docs + data + links only)

Acceptance Criteria:

- [x] Create an audit note under `model-lab/docs/audit/` with sources and “Reported” labels.
- [x] Add model rows to `model-lab/data/model_catalog.csv`.
- [x] Add a pointer from `model-lab/PERFORMANCE_RESULTS.md`.

Source:

- User request in chat (2026-02-05): “check these out as well: Ace Step 1.5 & Kugel Audio 0”.

Execution log:

- 2026-02-05 Added audit note + catalog rows + PERFORMANCE_RESULTS pointer (Reported, unverified).

Status updates:

- 2026-02-05 **DONE** — Audit note + catalog rows + pointers added (Reported, unverified)

---

# Ticket: UPGRADE-MODEL-CATALOG-EXPANDED-PLUS-20260205

Type: IMPROVEMENT  
Owner: Pranay  
Created: 2026-02-05  
Status: **DONE**  
Priority: P1

Scope contract:

- In-scope:
  - Capture the full chat-provided “Audio Model Audit (Comprehensive Survey, 2026)” text end-to-end under `model-lab/docs/from_chat/`.
  - Upgrade `model-lab/data/model_catalog.csv` to the **expanded_plus** schema (with `evidence_links`) and preserve any locally-added rows.
  - Populate `evidence_links` for applicable rows using the chat-provided citation list(s).
- Out-of-scope: Benchmarking, claim verification, harness registry changes.
- Behavior change allowed: NO (docs + data + pointers only)

Acceptance Criteria:

- [x] Add `model-lab/docs/from_chat/AUDIO_MODEL_AUDIT_FULLTEXT_COMPLETE_2026-02-05.md` marked **Reported**.
- [x] `model-lab/data/model_catalog.csv` uses expanded_plus columns and still includes custom additions (e.g., `ace-step-1-5`, `kugelaudio-0-open`).
- [x] Catalog rows include `evidence_links` populated from docs URLs + applicable items in the provided bibliography.
- [x] `model-lab/docs/from_chat/README.md` and `model-lab/PERFORMANCE_RESULTS.md` point to the new artifacts.

Source:

- User approval in chat (2026-02-05): “yes” to capturing full audit text + wiring citations into catalog / upgrading schema.

Execution log:

- 2026-02-05 Upgraded `model_catalog.csv` to expanded_plus schema; preserved local additions; filled `evidence_links`; added end-to-end audit capture + pointers.

Status updates:

- 2026-02-05 **DONE** — Completed per acceptance criteria (Reported, unverified)
