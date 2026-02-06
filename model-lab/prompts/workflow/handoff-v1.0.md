# Handoff Prompt

**Version**: 1.0  
**Purpose**: Document context for handoff between agents or sessions.

## Use When

- Ending a work session with incomplete work
- Passing work to another agent
- Complex task requires continuation

## Non-Negotiable Rules

1. **Complete state capture**: Everything needed to continue
2. **No assumptions**: Don't assume next agent knows context
3. **Links to evidence**: Reference all relevant files/tickets
4. **Clear next steps**: Explicit, actionable items

## Handoff Template

```markdown
# Agent Handoff Note

**Date**: YYYY-MM-DD HH:MM
**From**: [agent/session name]
**Ticket**: TCK-YYYYMMDD-###

## Work Summary

### Completed
- [x] [what was done]
- [x] [what was done]

### In Progress
- [ ] [what's partially done and current state]

### Not Started
- [ ] [what remains from original scope]

## Current State

### Files Modified
| File | Status | Notes |
|------|--------|-------|
| path/file.py | Modified | [brief description] |

### Git State
```bash
# Current branch
git branch --show-current
# Output: master

# Uncommitted changes
git status --porcelain
# Output: [paste]

# Last commit
git log -1 --oneline
# Output: [paste]
```

### Tests Status
```bash
PYTHONPATH=. pytest -q tests/
# Output: [pass/fail summary]
```

## Context Needed

### Why This Approach
[Explain key decisions made]

### Gotchas
[Things that tripped you up or might trip up next agent]

### Dependencies
[External factors, blocked on something, etc.]

## Next Steps (Priority Order)

1. **[First priority]**
   - Specific action
   - Verification: [how to verify done]

2. **[Second priority]**
   - Specific action
   - Verification: [how to verify done]

3. **[Third priority]**
   - Specific action
   - Verification: [how to verify done]

## Relevant Files

- `docs/WORKLOG_TICKETS.md` - Ticket details
- `docs/audit/[file].md` - Related audit (if applicable)
- `[other relevant files]`

## Questions / Decisions Needed

- [ ] [Question that needs human decision]
- [ ] [Unclear requirement]

## Evidence Snapshot

- **Observed**: [verified facts]
- **Inferred**: [logical conclusions]
- **Unknown**: [gaps]
```

## Handoff Checklist

Before handoff:

- [ ] Worklog ticket updated with current status
- [ ] All modified files listed
- [ ] Git state documented
- [ ] Next steps are actionable
- [ ] No uncommitted changes that would be lost
- [ ] Context sufficient for cold start

## Stop Condition

Handoff complete when:
- Template filled completely
- Ticket updated
- Next agent can start without asking questions
