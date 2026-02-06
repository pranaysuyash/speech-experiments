# Agent Handoff Protocol

**Version**: 1.0  
**Purpose**: Standardize handoffs between agents to preserve context.

## When This Applies

- Switching between agents mid-task
- Resuming work after context loss
- Transferring ownership of a task
- Multi-agent collaboration

## Handoff Package Structure

Every handoff MUST include:

### 1. Context Summary

```markdown
## Handoff: [Task Title]

**From**: [source agent/session]
**To**: [target agent/session]
**Date**: YYYY-MM-DD HH:MM
**Ticket**: TCK-YYYYMMDD-###

### What we're trying to accomplish
[1-2 paragraph summary of the goal]

### Why this matters
[business/technical context]
```

### 2. Current State

```markdown
### Progress

**Completed**:
- [x] [task 1] — Evidence: [link/output]
- [x] [task 2] — Evidence: [link/output]

**In progress**:
- [ ] [task 3] — Status: [where it stands]

**Not started**:
- [ ] [task 4]
- [ ] [task 5]

### Files touched
| File | Status | Notes |
|------|--------|-------|
| [path] | Modified/Created | [brief note] |

### Git state
- Branch: [branch name]
- Last commit: [hash] [message]
- Uncommitted changes: [yes/no, what]
```

### 3. Key Decisions Made

```markdown
### Decisions

| Decision | Rationale | Alternative considered |
|----------|-----------|----------------------|
| [decision] | [why] | [what else we could have done] |

### Assumptions
- [assumption 1]
- [assumption 2]
```

### 4. Blockers and Gotchas

```markdown
### Blockers
- [blocker 1]: [what's needed to unblock]

### Gotchas (things that tripped me up)
- [gotcha 1]: [how to avoid]
- [gotcha 2]: [how to avoid]

### Known issues
- [issue]: [workaround if any]
```

### 5. Next Steps (Priority Ordered)

```markdown
### Immediate next steps

1. **[First thing to do]**
   - Details: [specifics]
   - Verification: [how to know it's done]

2. **[Second thing to do]**
   - Details: [specifics]
   - Verification: [how to know it's done]

### Questions needing answers
- [ ] [question 1]
- [ ] [question 2]
```

### 6. Resources

```markdown
### Key files to read
- `[path]` — [why it's important]
- `[path]` — [why it's important]

### Relevant docs
- `[doc path]` — [what it covers]

### External links
- [link] — [what it is]
```

### 7. Evidence Snapshot

```markdown
### Evidence state

- **Observed**: [verified facts]
- **Inferred**: [logical conclusions]
- **Unknown**: [gaps in knowledge]
```

## Receiving a Handoff

When you receive a handoff:

1. **Read the full handoff** before taking any action
2. **Verify current state** matches what's described
3. **Check git state** with `git status` and `git log -3`
4. **Run tests** to confirm nothing is broken
5. **Ask clarifying questions** if anything is unclear
6. **Acknowledge receipt** with brief confirmation

```markdown
## Handoff Received

**Acknowledged by**: [agent]
**Date**: YYYY-MM-DD HH:MM

**State verification**:
- [ ] Git state matches
- [ ] Files match description
- [ ] Tests pass
- [ ] No questions

**Proceeding with**: [first action]
```

## Anti-Patterns

- ❌ "It's all in the code" (no, write it down)
- ❌ Assuming receiving agent has context
- ❌ Skipping the evidence snapshot
- ❌ Not listing gotchas (they WILL hit them)
- ❌ Vague next steps ("finish it up")
