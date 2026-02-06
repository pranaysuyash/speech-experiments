# Completion Report Prompt

**Version**: 1.0  
**Purpose**: Document successful completion of a work unit with evidence.

## Use When

- Completing a worklog ticket
- Finishing a feature implementation
- Closing out an audit remediation

## Non-Negotiable Rules

1. **Evidence for every criterion**: Each acceptance criterion needs proof
2. **Tests documented**: Show test results
3. **No loose ends**: All scope items addressed
4. **Worklog updated**: Ticket marked COMPLETED

## Completion Report Template

```markdown
# Completion Report

**Ticket**: TCK-YYYYMMDD-###
**Title**: [ticket title]
**Date Completed**: YYYY-MM-DD HH:MM
**Agent**: [name]

## Summary

[One paragraph summary of what was accomplished]

## Acceptance Criteria Verification

| Criterion | Status | Evidence |
|-----------|--------|----------|
| [criterion 1] | ✅ PASS | [link/output] |
| [criterion 2] | ✅ PASS | [link/output] |
| [criterion 3] | ⚠️ PARTIAL | [explanation] |

## Files Changed

| File | Change Type | Lines |
|------|-------------|-------|
| path/file.py | Modified | +50/-20 |
| path/new.py | Added | +100 |

## Test Results

```bash
PYTHONPATH=. pytest -q tests/
# Output:
# X passed, Y skipped in Z seconds
```

## Verification Commands

```bash
# How to verify the changes work
[command 1]
[command 2]
```

## Evidence Snippet

- **Observed**: [verified facts from this work]
- **Inferred**: [logical conclusions]
- **Unknown**: [remaining gaps, if any]

## Scope Compliance

- [x] All in-scope items addressed
- [x] No out-of-scope changes made
- [x] Behavior change: [as allowed per ticket]

## Documentation Updated

- [ ] README (if applicable)
- [ ] API docs (if applicable)
- [ ] Worklog ticket (required)
- [ ] CLAIMS.md (if performance claims)

## Commit Information

```bash
git log -1 --format="%H %s"
# Output: [commit hash and message]
```

## Remaining Items (if any)

[List any follow-up tickets or known limitations]

## Lessons Learned

[Optional: What would you do differently?]
```

## Completion Checklist

Before marking complete:

- [ ] All acceptance criteria met (or documented as out-of-scope)
- [ ] Tests pass
- [ ] Build succeeds
- [ ] Documentation updated
- [ ] Worklog ticket updated with COMPLETED status
- [ ] Commit message references ticket
- [ ] No uncommitted changes remain

## Stop Condition

Complete when:
- All criteria verified with evidence
- Ticket status is COMPLETED
- Report documented
- Changes committed
