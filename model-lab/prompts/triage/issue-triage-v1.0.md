# Issue Triage Prompt

**Version**: 1.0  
**Purpose**: Triage incoming issues/bugs and convert to worklog tickets.

## Use When

- New GitHub issue filed
- Bug report received
- User feedback needs processing
- Audit finding needs ticketing

## Non-Negotiable Rules

1. **One issue = one ticket**: Don't batch unrelated issues
2. **Evidence required**: Document how issue was reproduced or verified
3. **Priority assigned**: Every ticket gets a priority
4. **Linked to source**: Reference original issue/report

## Steps

### 1. Understand the Issue

Gather:
- What is the problem?
- How to reproduce?
- What is expected behavior?
- What is actual behavior?
- Environment details

### 2. Verify/Reproduce

```bash
# Try to reproduce the issue
# Document exact steps and output
```

### 3. Assess Priority

**P0 (Critical)**
- System is down
- Data loss/corruption
- Security vulnerability
- Blocks all users

**P1 (High)**
- Major functionality broken
- Affects many users
- No workaround

**P2 (Medium)**
- Minor functionality broken
- Workaround exists
- Affects some users

**P3 (Low)**
- Cosmetic issues
- Minor inconvenience
- Edge cases

### 4. Create Ticket

```markdown
### TCK-YYYYMMDD-### :: [Short descriptive title]

Type: [BUG | FEATURE | IMPROVEMENT | MODEL]
Owner: Pranay
Created: YYYY-MM-DD
Status: **OPEN**
Priority: P0 | P1 | P2 | P3

Source:
- GitHub Issue: #[number] (if applicable)
- Report: [link or description]

Description:
[Clear description of the issue]

Steps to Reproduce:
1. [step]
2. [step]
3. [step]

Expected Behavior:
[what should happen]

Actual Behavior:
[what actually happens]

Environment:
- OS: [os]
- Python: [version]
- Hardware: [MPS/CUDA/CPU]

Evidence:
- **Observed**: [what was directly verified]
- **Inferred**: [logical conclusions]
- **Unknown**: [gaps in understanding]

Acceptance Criteria:
- [ ] [specific fix criteria]
- [ ] [verification criteria]

Scope contract:
- In-scope: [what to fix]
- Out-of-scope: [what not to change]
- Behavior change allowed: [YES/NO]
```

### 5. Respond to Reporter

If external issue:
- Acknowledge receipt
- Provide ticket ID
- Set expectations on timeline

## Triage Decision Tree

```
Is it reproducible?
├── YES → Create ticket, assign priority
├── NO → Request more info
│   └── After 7 days with no response → Close as "needs info"
└── UNKNOWN → Investigate further, then decide

Is it a duplicate?
├── YES → Link to existing ticket, close
└── NO → Proceed with new ticket

Is it in scope?
├── YES → Create ticket
├── NO → Document why, close with explanation
└── PARTIAL → Create ticket for in-scope portion
```

## Output Format

```markdown
## Triage Summary

**Issue**: [title/link]
**Date**: YYYY-MM-DD
**Triager**: [name/agent]

### Decision

- **Action**: Created Ticket / Duplicate / Needs Info / Out of Scope
- **Ticket**: TCK-YYYYMMDD-### (if created)
- **Priority**: P0/P1/P2/P3
- **Rationale**: [why this decision]

### Reproduction

- **Reproduced**: YES / NO / PARTIAL
- **Steps**: [what was tried]
- **Evidence**: [output]

### Response Sent

[message sent to reporter, if applicable]
```

## Stop Condition

Stop when:
- Issue understood
- Ticket created (or decision documented)
- Priority assigned
- Reporter notified (if applicable)
