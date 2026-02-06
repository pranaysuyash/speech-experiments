# Architecture Decision Record (ADR) Prompt

**Version**: 1.0  
**Purpose**: Document significant architectural decisions.

## Role

You document architectural decisions so future developers understand why things are the way they are.

## When to Write an ADR

- Choosing between competing technologies
- Designing a new system or major feature
- Changing existing architecture
- Making a decision that's hard to reverse
- Making a decision others might question

## ADR Template

```markdown
# ADR-XXX: [Title]

**Status**: Proposed / Accepted / Deprecated / Superseded by ADR-YYY
**Date**: YYYY-MM-DD
**Author**: [name]
**Reviewers**: [names]

## Context

### Problem Statement

[What problem are we trying to solve? What's the current situation?]

### Constraints

- [constraint 1]
- [constraint 2]

### Requirements

**Must have**:
- [requirement]

**Nice to have**:
- [requirement]

## Options Considered

### Option 1: [Name]

**Description**: [What is this option?]

**Pros**:
- [pro 1]
- [pro 2]

**Cons**:
- [con 1]
- [con 2]

**Effort**: S/M/L
**Risk**: Low/Med/High

### Option 2: [Name]

[Same structure...]

### Option 3: [Name]

[Same structure...]

## Decision

**Chosen option**: [Option N]

**Rationale**:
[Why this option? What were the deciding factors?]

**Trade-offs accepted**:
- [trade-off 1]
- [trade-off 2]

## Consequences

### Positive

- [consequence 1]
- [consequence 2]

### Negative

- [consequence 1]
- [consequence 2]

### Neutral

- [consequence 1]

## Implementation

### Phases

1. [Phase 1]: [description]
2. [Phase 2]: [description]

### Migration (if applicable)

[How to migrate from current state to new state]

### Rollback Plan

[How to undo if this doesn't work out]

## Related

- ADR-XXX: [Related decision]
- [External reference]

## Notes

[Any additional context or caveats]
```

## Writing Guidelines

### Be Honest About Trade-offs

```markdown
# Bad
Option A is clearly the best choice.

# Good
We chose Option A because [reason], accepting that this means we cannot [trade-off]. Option B would have given us [benefit] but required [cost].
```

### Include Rejected Options

Future developers need to know what was considered and why it was rejected. Otherwise they'll propose the same thing.

### Link to Evidence

```markdown
# Bad
Option A is faster.

# Good
Option A is faster (see benchmarks in docs/benchmarks/adr-015-comparison.md).
```

### Date Everything

Decisions that made sense in 2024 might not make sense in 2026. Dating helps future readers understand the context.

## ADR Lifecycle

1. **Proposed**: Written, awaiting review
2. **Accepted**: Approved, will be implemented
3. **Deprecated**: No longer applies (explain why)
4. **Superseded**: Replaced by another ADR (link to it)

## File Naming

```
docs/architecture/decisions/
├── 001-use-fastapi-for-backend.md
├── 002-mps-first-inference.md
├── 003-modality-abstraction.md
└── README.md (index of all ADRs)
```

## Quality Checklist

- [ ] Problem is clearly stated
- [ ] Multiple options considered
- [ ] Pros/cons for each option
- [ ] Decision rationale is clear
- [ ] Trade-offs acknowledged
- [ ] Consequences documented
- [ ] Rollback plan exists
- [ ] Related decisions linked
