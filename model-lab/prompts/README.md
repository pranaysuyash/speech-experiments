# Prompts Index

This folder contains reusable prompts that guide agents and contributors when performing work in this repository.

## Start Here (Any Agent)

1. Use: `prompts/workflow/agent-entrypoint-v1.0.md`
2. Update tracking: `docs/WORKLOG_TICKETS.md` (append-only)
3. Ensure local enforcement: `git config core.hooksPath .githooks`

## 🚫 Critical Rules

**NEVER create new git branches unless explicitly asked by the user.** Work on `master` (or existing feature branch if user created it).

**NEVER delete or revert files with unrecognized changes.** Unrecognized changes may be from parallel agents - always preserve them.

---

## Prompt Map (By Role / Job)

### Workflow / Tracking

| Prompt | Purpose |
|--------|---------|
| `prompts/workflow/agent-entrypoint-v1.0.md` | Starting point for any agent work |
| `prompts/workflow/worklog-v1.0.md` | Create/update worklog tickets |
| `prompts/workflow/pre-flight-check-v1.0.md` | Pre-work verification |
| `prompts/workflow/project-setup-verification-v1.0.md` | Verify project setup |
| `prompts/workflow/repo-hygiene-sweep-v1.0.md` | Clean up repo issues |
| `prompts/workflow/prompt-quality-gate-v1.0.md` | Validate prompt quality |
| `prompts/workflow/handoff-v1.0.md` | Document handoff between sessions |
| `prompts/workflow/completion-report-v1.0.md` | Document work completion |

### Audit / Review

| Prompt | Purpose |
|--------|---------|
| `prompts/audit/comprehensive-audit-v1.0.md` | Full product/codebase audit |
| `prompts/audit/codebase-audit-improvement-planner-v1.0.md` | Deep audit with Top 10 improvements |
| `prompts/audit/reality-first-auditor-v1.0.md` | Verify docs vs code, find missing standards |
| `prompts/review/pr-review-v1.0.md` | Review pull requests |
| `prompts/review/code-review-checklist-v1.0.md` | Quick review checklist |

### Implementation / Remediation

| Prompt | Purpose |
|--------|---------|
| `prompts/remediation/implementation-v1.0.md` | Implement audit findings |
| `prompts/hardening/hardening-v1.0.md` | Production hardening |

### QA / Testing

| Prompt | Purpose |
|--------|---------|
| `prompts/qa/test-plan-v1.0.md` | Create test plans |
| `prompts/qa/regression-hunt-v1.0.md` | Find regressions |

### Model-Specific (ASR/TTS)

| Prompt | Purpose |
|--------|---------|
| `prompts/model/model-evaluation-v1.0.md` | Evaluate model performance |
| `prompts/model/model-addition-v1.0.md` | Add new model to registry |

### Security

| Prompt | Purpose |
|--------|---------|
| `prompts/security/security-review-v1.0.md` | Security vulnerability review |

### Triage

| Prompt | Purpose |
|--------|---------|
| `prompts/triage/issue-triage-v1.0.md` | Triage issues to tickets |

### Exploration / Research

| Prompt | Purpose |
|--------|---------|
| `prompts/exploration/research-deep-dive-v1.0.md` | Thorough research with citations |
| `prompts/exploration/technology-evaluation-v1.0.md` | Evaluate tech for adoption |

### Agent Coordination

| Prompt | Purpose |
|--------|---------|
| `prompts/coordination/agent-delegation-v1.0.md` | Delegate work to sub-agents |
| `prompts/coordination/agent-handoff-protocol-v1.0.md` | Standardize agent handoffs |

### Sprint Operations

| Prompt | Purpose |
|--------|---------|
| `prompts/sprint/hf-pro-agent-picklist-v1.0.md` | Pick a sprint role and start immediately |
| `prompts/sprint/hf-pro-edge-small-executor-v1.0.md` | Execute edge/small model queue |
| `prompts/sprint/hf-pro-domain-specialist-executor-v1.0.md` | Execute specialist capability queue |
| `prompts/sprint/hf-pro-realtime-streaming-executor-v1.0.md` | Execute realtime/streaming queue |
| `prompts/sprint/hf-pro-general-baselines-executor-v1.0.md` | Execute baseline/high-coverage queue |
| `prompts/sprint/hf-pro-qa-referee-v1.0.md` | Audit ledgers and prioritize reruns |
| `prompts/sprint/hf-pro-synthesis-lead-v1.0.md` | Build final recommendation synthesis |
| `prompts/sprint/hf-pro-streaming-integration-fixer-v1.0.md` | Fix streaming adapter/integration blockers |

### Documentation

| Prompt | Purpose |
|--------|---------|
| `prompts/documentation/documentation-writer-v1.0.md` | Write clear documentation |
| `prompts/documentation/changelog-writer-v1.0.md` | Write user-focused changelogs |

### Mentoring / Learning

| Prompt | Purpose |
|--------|---------|
| `prompts/mentoring/technical-tutor-v1.0.md` | Teach concepts with active learning |
| `prompts/mentoring/concept-explainer-v1.0.md` | Build intuition for any concept |

### Guidance

| Prompt | Purpose |
|--------|---------|
| `prompts/guidance/debugging-guide-v1.0.md` | Systematic debugging approach |
| `prompts/guidance/architecture-decision-record-v1.0.md` | Document architecture decisions |

---

## Quick Reference

### Common Commands

```bash
# Stage all changes
git add -A

# Run core tests
PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py

# Build frontend
cd client && npm run build

# Type check
PYTHONPATH=. mypy server/ harness/ --ignore-missing-imports

# Model inference test
PYTHONPATH=. python -m harness.run --model whisper-tiny --quick
```

### Ticket Template (Quick)

```markdown
### TCK-YYYYMMDD-### :: [Title]

Type: [BUG|FEATURE|MODEL|IMPROVEMENT|AUDIT]
Owner: [name]
Created: YYYY-MM-DD
Status: **OPEN**
Priority: P0|P1|P2|P3

Scope contract:
- In-scope: [what to do]
- Out-of-scope: [what not to do]
- Behavior change allowed: YES|NO

Acceptance Criteria:
- [ ] [criterion]

Execution log:
- [timestamp] [action] | Evidence: [output]
```

---

## Project Context (Read Before Making Changes)

- **Agent instructions**: `docs/AGENTS.md`
- **Project rules**: `docs/PROJECT_RULES.md`
- **Architecture**: `docs/` directory
- **Process docs**: `docs/process/`
  - `COMMANDS.md` - Common commands
  - `PROMPT_STYLE_GUIDE.md` - Prompt conventions
  - `CODE_PRESERVATION_GUIDELINES.md` - When to delete vs implement
  - `OWNERSHIP_POLICY.md` - Code ownership
- **Claims registry**: `docs/CLAIMS.md` - Evidence for model performance claims
- **Issues workflow**: `docs/ISSUES_WORKFLOW.md`

---

## Directory Structure

```
prompts/
├── audit/
│   ├── comprehensive-audit-v1.0.md
│   ├── codebase-audit-improvement-planner-v1.0.md
│   └── reality-first-auditor-v1.0.md
├── coordination/
│   ├── agent-delegation-v1.0.md
│   └── agent-handoff-protocol-v1.0.md
├── documentation/
│   ├── documentation-writer-v1.0.md
│   └── changelog-writer-v1.0.md
├── exploration/
│   ├── research-deep-dive-v1.0.md
│   └── technology-evaluation-v1.0.md
├── guidance/
│   ├── debugging-guide-v1.0.md
│   └── architecture-decision-record-v1.0.md
├── hardening/
│   └── hardening-v1.0.md
├── mentoring/
│   ├── technical-tutor-v1.0.md
│   └── concept-explainer-v1.0.md
├── model/
│   ├── model-addition-v1.0.md
│   └── model-evaluation-v1.0.md
├── qa/
│   ├── regression-hunt-v1.0.md
│   └── test-plan-v1.0.md
├── remediation/
│   └── implementation-v1.0.md
├── review/
│   ├── code-review-checklist-v1.0.md
│   └── pr-review-v1.0.md
├── security/
│   └── security-review-v1.0.md
├── sprint/
│   ├── hf-pro-agent-picklist-v1.0.md
│   ├── hf-pro-edge-small-executor-v1.0.md
│   ├── hf-pro-domain-specialist-executor-v1.0.md
│   ├── hf-pro-realtime-streaming-executor-v1.0.md
│   ├── hf-pro-general-baselines-executor-v1.0.md
│   ├── hf-pro-qa-referee-v1.0.md
│   ├── hf-pro-synthesis-lead-v1.0.md
│   └── hf-pro-streaming-integration-fixer-v1.0.md
├── triage/
│   └── issue-triage-v1.0.md
├── workflow/
│   ├── agent-entrypoint-v1.0.md
│   ├── completion-report-v1.0.md
│   ├── handoff-v1.0.md
│   ├── pre-flight-check-v1.0.md
│   ├── project-setup-verification-v1.0.md
│   ├── prompt-quality-gate-v1.0.md
│   ├── repo-hygiene-sweep-v1.0.md
│   └── worklog-v1.0.md
└── README.md (this file)
```

---

## Adding New Prompts

When adding a new prompt:

1. Create file in appropriate subdirectory
2. Follow `docs/process/PROMPT_STYLE_GUIDE.md`
3. Update this README with the new prompt
4. Version using `-v1.0.md` suffix

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 4.0 | 2026-02-12 | Added HF Pro sprint multi-agent prompt pack |
| 3.0 | 2026-02-05 | Added exploration, coordination, documentation, mentoring, guidance prompts |
| 2.0 | 2026-02-05 | Added model, QA, security, triage, hardening prompts |
| 1.0 | 2026-02-04 | Initial workflow and audit prompts |
