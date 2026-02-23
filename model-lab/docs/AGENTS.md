# Agent Instructions

<!-- PROJECTS_MEMORY_AGENT_ALIGNMENT_BEGIN -->

## Projects-Level Agent Alignment (Workspace Memory)

**Purpose:** ensure any agent/LLM (Codex, Copilot, Claude Code, Qwen, GLM, etc.) starts aligned with the same workspace memory + project context.

### Step 0 (first time in this folder)
Generate the per-project context pack:
```bash
/Users/pranay/Projects/agent-start
```

### Step 1 (per shell)
Load the shared defaults for this project session:
```bash
source .agent/STEP1_ENV.sh
# Or (no file read) print exports and eval:
/Users/pranay/Projects/agent-start --print-step1 --skip-index
```

### Step 2 (generate aligned context pack)
```bash
/Users/pranay/Projects/agent-start
```

Outputs:
- `.agent/SESSION_CONTEXT.md`
- `.agent/AGENT_KICKOFF_PROMPT.txt`
- `.agent/STEP1_ENV.sh`

### Automation (already configured)
- Terminal auto-loads `.agent/STEP1_ENV.sh` when you `cd` into a project under `/Users/pranay/Projects` (zsh hook).
- VS Code/Antigravity can run `agent-start --skip-index` on folder open via `.vscode/tasks.json`.

### How agents should use this
- Provide `.agent/AGENT_KICKOFF_PROMPT.txt` and `.agent/SESSION_CONTEXT.md` as the first context for the agent.
- If sources conflict, the agent must cite concrete file paths and ask before proceeding.
- If `.agent` files are missing or stale, run `/Users/pranay/Projects/agent-start --skip-index` before planning changes.
- Do not start implementation until `.agent/AGENT_KICKOFF_PROMPT.txt` and `.agent/SESSION_CONTEXT.md` are loaded.

### Optional commit safety net
Install repo-local git pre-commit hooks that refresh and stage `.agent/*` before commit:
```bash
python3 /Users/pranay/Projects/workspace_memory/scripts/install_git_precommit_agent_hook.py
```

### Shared Idea Pad Protocol (Required)
- Canonical file: `/Users/pranay/Projects/idea_pad/IDEA_PAD.md`
- Raw capture file: `/Users/pranay/Projects/idea_pad/IDEA_DUMP.md`
- Do not create per-model primary copies of the idea pad.
- Do not overwrite the whole file; use append/update workflow with validation.
- Capture rough ideas in `IDEA_DUMP.md`, then promote high-signal items into `IDEA_PAD.md`.
- Before edits:
```bash
python3 /Users/pranay/Projects/idea_pad/scripts/idea_pad_tool.py validate
```
- Add new ideas safely:
```bash
python3 /Users/pranay/Projects/idea_pad/scripts/idea_pad_tool.py add --title "<title>" --owner "<agent>" --type build
```
- After updates, refresh shared memory index:
```bash
cd /Users/pranay/Projects
./projects-memory index
```

<!-- PROJECTS_MEMORY_AGENT_ALIGNMENT_END -->

## Worktrees

- All work happens in the primary worktree on master branch.
- No separate worktrees needed.

## Branch

- Use branch: master
- Keep diffs comprehensive but tested. Include regression checks.

## Non-negotiable verification before pushing to master

In the workspace:

- PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
- PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py
- cd client && npm run build
- git status --porcelain must be empty (after commit)
- For commits with >10% LOC changes in existing files, review diff for improvements

## Regression Check Process

- git add -A to stage all changes
- Check git diff --cached --stat for LOC changes
- For files with significant changes (>10% of total lines), review diff to ensure better/comprehensive/updated
- If not satisfactory, update code before committing
- Commit with descriptive message
- Push if approved

## Do not change

- docs/CHANGELOG.md must not be modified or removed.

---

## Core Principles

### Evidence-First Development

- Every claim must be backed by evidence: `Observed` (directly verified), `Inferred` (logical implication), `Unknown` (cannot determine).
- Never upgrade `Inferred` to `Observed` without verification.
- Preserve evidence in audit artifacts and worklog tickets.

### Single Source of Truth

- **Worklog**: `docs/WORKLOG_TICKETS.md` — canonical, append-only ticket registry.
- **Audits**: `docs/audit/` — one audit file per audit.
- **Prompts**: `prompts/` — repository prompt library and `prompts/README.md` index.
- **Code**: Repository source under version control.

### Preservation First + Implementation Over Deletion

**Principle:** Don't just delete unused code. Understand why it exists, see if it can make the app better, and implement functionality rather than delete.

- Never discard contributor code unless clearly inferior.
- Keep meaningful comments/tests/docs unless incorrect.
- Prefer merging both sides when resolving conflicts.
- **Investigate before deleting**: When you find unused code, investigate its history and purpose.
- **Prefer activation**: If code is 70%+ complete and adds value, complete it rather than delete.
- **See**: `docs/process/CODE_PRESERVATION_GUIDELINES.md` for detailed workflow.

**No deletions without explicit approval**:

- Never delete files (code, docs, audits, tickets, assets) unless the user explicitly asks for deletion **or** there is explicit, recorded approval in the active ticket.
- If cleanup is needed, move to an `archive/` folder and leave a pointer note (preserve history).
- **Exception**: Deletion is acceptable after completing the investigation workflow in CODE_PRESERVATION_GUIDELINES.md and documenting why deletion was chosen over implementation.

### Staging Is Always Comprehensive

- Always stage changes with: `git add -A`.
- Do not "selectively stage" unless the user explicitly asks.
- Do not use staging as a mechanism to "drop" other agents' work.

### Branch and Parallel Work Preservation (CRITICAL)

**🚫 NEVER create new git branches unless explicitly asked by the user.**

- Always work on the current branch (main/master).
- If a feature branch already exists, the user created it — work there.
- Do not create `feature/`, `fix/`, `hotfix/`, or any other branches.

**🚫 NEVER delete or revert files with unrecognized changes.**

- Unrecognized changes may be from parallel agents working simultaneously.
- If you see changes you do not recognize, PRESERVE them.
- Only modify/delete files you are explicitly tasked to work on.
- When in doubt, ask the user before removing anything.

---

## Agent Workflow

### Phase 1: Intake

Before starting ANY work, determine:

1. What type of work? (New file audit, Remediation PR from audit, Hardening PR one scope area, PR Review/Verification, Merge conflict resolution, Post-merge validation)
2. Define scope contract: Target file OR hardening scope, Behavior change allowed: YES/NO, Explicit non-goals, Acceptance criteria, Base branch: main/master.
3. Select the correct repo prompt (MANDATORY): Use `prompts/README.md` to find the appropriate prompt for the work type. Open and follow that prompt's required steps + required artifacts. If the user provides an external prompt, curate it into `prompts/` (repo-native) and add it to `prompts/README.md` so future agents use the same source of truth.
4. Ticket Action (MANDATORY): Create or update `docs/WORKLOG_TICKETS.md`. Append-only discipline.

### Phase 2: Work Execution

Based on work type, follow the appropriate prompt. (Note: This repo currently has workflow prompts; expand as needed for audits, reviews, etc.)

### Phase 3: Documentation

Every work unit MUST produce:

1. **Worklog Entry** in `docs/WORKLOG_TICKETS.md`
2. **Audit Artifact** (for audits) in `docs/audit/<file>.md`
3. **Verifier Pack** (for PRs) in PR description
4. **Evidence Log** with raw command outputs
5. **Docs updates** when you change behavior or workflow: If you add or change prompts, update `prompts/README.md`. If you add tooling/scripts/hooks, update relevant docs and scripts.
6. **Prompt & persona traceability**: In every artifact (worklog entry, plan doc, reality check, audit, etc.) note which prompts(s) were used — single, combined, or sequential — along with the audit axis, personas, or lenses that guided the analysis so future agents can reproduce the reasoning.

---

## Audit-to-Ticket Workflow

### Overview

A critical gap identified: Audit reports contain comprehensive findings but ~90% are not systematically converted to worklog tickets. This causes important issues to be forgotten and context to be lost.

### Process

When reading audit documents and finding actionable issues:

1. **Immediate Action:** Create ticket IMMEDIATELY (even OPEN) before starting implementation. Document the audit source and finding ID.
2. **Ticket Creation Template:** Use the template in `prompts/workflow/worklog-v1.0.md`.
3. **Audit Discovery Phase Best Practices:** Create tickets FIRST before code changes. Always link to specific audit file and line numbers. Use evidence (quotes, screenshots, line numbers) from audit. Don't batch unrelated fixes in one ticket. One issue = one ticket (unless explicitly scoped).
4. **Regular Audit Review:** Use `./scripts/audit_review.sh` weekly to review audit docs for untracked findings.
5. **Ticket Creation Discipline:** ALWAYS create a worklog ticket (even OPEN) before starting implementation. NEVER just implement the fix without a tracking ticket. ALWAYS reference the specific audit file and finding ID. ALWAYS include the evidence from the audit.

### Root Cause

The audit-to-ticket gap exists because: No systematic workflow for converting audit findings → tickets. Silent backlog building — Audit docs contain "roadmaps" and "improvement plans" but aren't tracked. Discovery disconnect — Finding issues (Phase 1) isn't tracked, only remediation (Phase 2) is.

---

## Quality Gates

### Audit Gate

Pass if:

- [ ] Discovery appendix complete
- [ ] Evidence labels correct (Observed/Inferred/Unknown)
- [ ] Patch plan is scoped + testable
- [ ] Artifact exists in docs/audit/

### Implementation Gate

Pass if:

- [ ] Diff limited to scoped files + tests
- [ ] Each change maps to finding ID
- [ ] Invariants preserved (or Behavior change: YES)
- [ ] Tests/verification artifacts for HIGH/MED findings
- [ ] Docs/claims match diff

### PR Review Gate

Pass if:

- [ ] Diff-only scope
- [ ] Findings-driven review
- [ ] Tests pass
- [ ] CI status noted

---

## Naming Conventions

### Ticket IDs

```
TCK-YYYYMMDD-###

Examples:
TCK-20260205-001
TCK-20260205-002
```

### Claim IDs

```
CLM-YYYYMMDD-###

Examples:
CLM-20260205-001
```

### Audit Files

```
docs/audit/<descriptive-name>.md

Examples:
docs/audit/comprehensive-audit-20260204.md
docs/audit/security-review-api-20260205.md
```

---

## Communication Protocol

### When Starting Work

```markdown
**Agent**: [Agent Name]
**Action**: Starting [work type] on [target]
**Ticket**: TCK-YYYYMMDD-###
**Scope**: [brief description]
**Base**: master@[commit-sha]
```

### When Completing Work

```markdown
**Agent**: [Agent Name]
**Action**: Completed [work type] on [target]
**Ticket**: TCK-YYYYMMDD-###
**Status**: [OPEN/IN_PROGRESS/BLOCKED/DONE/DROPPED]
**Evidence**: [link to evidence/outputs]
**Next**: [next action or agent]
```

### When Blocked

```markdown
**Agent**: [Agent Name]
**Blocked On**: [specific issue]
**Evidence**: [what was attempted]
**Help Needed**: [specific question]
**Ticket**: TCK-YYYYMMDD-###
```

---

## Resources

### Prompts Directory

See `prompts/README.md` for full index. Key prompts:

```
prompts/
├── audit/comprehensive-audit-v1.0.md
├── remediation/implementation-v1.0.md
├── review/pr-review-v1.0.md
├── qa/test-plan-v1.0.md
├── model/model-evaluation-v1.0.md
├── security/security-review-v1.0.md
└── workflow/agent-entrypoint-v1.0.md
```

### Process Documentation

- `docs/process/COMMANDS.md` - Common commands
- `docs/process/PROMPT_STYLE_GUIDE.md` - Prompt conventions
- `docs/process/CODE_PRESERVATION_GUIDELINES.md` - Deletion vs implementation
- `docs/process/OWNERSHIP_POLICY.md` - Code ownership

### Tracking

- `docs/WORKLOG_TICKETS.md` - Work tracking
- `docs/CLAIMS.md` - Model performance claims (append-only)
- `docs/audit/*.md` - Audit artifacts

### Templates

- `docs/templates/CHANGE_CLASSIFICATION.md` - Classify changes

---

## Quick Reference

### Common Commands

```bash
# Check git status
git status --porcelain

# Stage changes
git add -A

# Run core tests
PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py

# Build frontend
cd client && npm run build

# Type check
PYTHONPATH=. mypy server/ harness/ --ignore-missing-imports

# Model smoke test
PYTHONPATH=. python -m harness.run --model whisper-tiny --quick

# Find references
rg -n "symbol_name" server/
```

### Hooks Setup

```bash
# Enable repo hooks
git config core.hooksPath .githooks
```

---

## Prohibited Actions

1. **Never** commit secrets to git
2. **Never** upgrade Inferred to Observed without verification
3. **Never** mix unrelated fixes in one PR
4. **Never** delete contributor code without investigation
5. **Never** skip worklog updates
6. **Never** claim "ready" without evidence
7. **Never** expand scope without explicit approval
8. **Never** modify docs/CHANGELOG.md
9. **Never** create branches without user request
10. **Never** delete other agents' work without approval

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0 | 2026-02-05 | Added quality gates, naming conventions, communication protocol, resources section, quick reference |
| 1.0 | 2026-02-04 | Initial version |

---

**Remember**: Evidence first. Scope discipline. Preservation over perfection.
