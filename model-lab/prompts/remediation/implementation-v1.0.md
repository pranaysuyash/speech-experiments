# Implementation Prompt (Remediation)

**Version**: 1.0  
**Purpose**: Implement fixes for audit findings with evidence-backed verification.

## Use When

- Fixing issues identified in an audit report
- Implementing changes from a worklog ticket
- Making scoped changes with clear acceptance criteria

## Non-Negotiable Rules

1. **Ticket first**: Ensure a worklog ticket exists before coding
2. **Scope discipline**: Only change files in scope contract
3. **Evidence required**: Every change must have verification evidence
4. **No scope creep**: New issues → new tickets, not scope expansion
5. **Preserve behavior**: Unless behavior change is explicitly allowed

## Inputs

- Worklog ticket ID (e.g., TCK-20260205-001)
- Audit finding ID (e.g., ARCH-001)
- Scope contract (files, behavior change allowed, acceptance criteria)

## Steps

### 1. Verify Preconditions

```bash
# Confirm ticket exists
rg "TCK-YYYYMMDD-###" docs/WORKLOG_TICKETS.md

# Check current branch
git branch --show-current

# Check for dirty state
git status --porcelain
```

### 2. Review Scope Contract

From the ticket, confirm:
- [ ] Files in scope
- [ ] Behavior change allowed: YES/NO
- [ ] Acceptance criteria documented
- [ ] Out-of-scope items noted

### 3. Implement Changes

For each change:
1. Make the code change
2. Run relevant tests
3. Document evidence in ticket

### 4. Verification

```bash
# Run core tests
PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py
PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py

# Build frontend (if applicable)
cd client && npm run build

# Type check
PYTHONPATH=. mypy server/ harness/ --ignore-missing-imports
```

### 5. Update Ticket

Add execution log entry:
```markdown
- [timestamp] Implemented [change] | Evidence: [test output/command]
```

Update status:
```markdown
- [timestamp] **COMPLETED** — [summary of changes]
```

## Output Format

### Required Sections in Ticket

1. **Execution log** with timestamped entries
2. **Evidence** for each acceptance criterion
3. **Status update** with completion timestamp
4. **Files changed** summary

### Commit Message Format

```
fix(<scope>): <description>

Refs: TCK-YYYYMMDD-###
Fixes: <finding-id>

- Change 1
- Change 2

Evidence: [brief verification summary]
```

## Stop Condition

Stop when:
- All acceptance criteria met with evidence
- Tests pass
- Ticket status updated to COMPLETED
- No scope creep occurred

## Anti-Patterns

- ❌ Fixing "one more thing" outside scope
- ❌ Skipping verification
- ❌ Not updating ticket with evidence
- ❌ Changing behavior when not allowed
