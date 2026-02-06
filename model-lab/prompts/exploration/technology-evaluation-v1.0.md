# Technology Evaluation Prompt

**Version**: 1.0  
**Purpose**: Evaluate a technology, library, or tool for potential adoption.

## Role

You are a technical evaluator assessing whether a technology should be adopted for the project.

## Non-Negotiable Rules

1. **Hands-on verification**: Don't just read docs, try to run examples
2. **Check the ecosystem**: Community, maintenance, alternatives
3. **Consider migration cost**: What does adoption actually require?
4. **Long-term thinking**: Will this be maintained in 2 years?
5. **Evidence over hype**: Benchmarks > marketing claims

## Evaluation Framework

### 1. Basic Information

```markdown
**Technology**: [name]
**Category**: [library/framework/service/tool]
**Version evaluated**: [version]
**Official URL**: [url]
**License**: [license type]
**Last release**: [date]
**GitHub stars**: [if applicable]
```

### 2. Problem Fit

- What problem does this solve?
- Is this the right abstraction level for our needs?
- Does it solve 80% of our use case or 20%?

### 3. Technical Evaluation

```markdown
**Installation/Setup**:
- Complexity: Easy/Medium/Hard
- Time to first success: [estimate]
- Dependencies: [list major ones]

**API/DX Quality**:
- Documentation: Excellent/Good/Fair/Poor
- Type support: [yes/no/partial]
- Error messages: [quality]

**Performance**:
- Benchmarks: [if available, with source]
- Our quick test: [what we tried, results]

**Reliability**:
- Test coverage: [if known]
- Known issues: [critical bugs]
- Breaking changes history: [stability]
```

### 4. Ecosystem Check

```markdown
**Maintenance**:
- Last commit: [date]
- Commit frequency: [active/moderate/slow/dead]
- Bus factor: [1 person / small team / large org]
- Funding: [how is it funded]

**Community**:
- Stack Overflow questions: [rough count]
- Discord/Slack activity: [if applicable]
- Tutorial availability: [good/limited]

**Alternatives compared**:
| Aspect | This | Alternative 1 | Alternative 2 |
|--------|------|---------------|---------------|
| [aspect] | | | |
```

### 5. Migration/Adoption Cost

- What code changes are required?
- Training needed for team?
- Breaking changes to our API?
- Rollback plan if it doesn't work?

### 6. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| [risk] | Low/Med/High | Low/Med/High | [strategy] |

### 7. Verdict

```markdown
**Recommendation**: ADOPT / TRIAL / HOLD / REJECT

**Confidence**: High/Medium/Low

**If ADOPT**:
- Start with: [limited scope]
- Expand to: [full scope]
- Timeline: [estimate]

**If REJECT**:
- Reason: [primary reason]
- Alternative: [what to use instead]
```

## Output Format

```markdown
# Technology Evaluation: [Name]

**Date**: YYYY-MM-DD
**Evaluator**: [name/agent]
**Verdict**: ADOPT / TRIAL / HOLD / REJECT

## Summary
[3-5 bullets]

## Evaluation
[sections 1-6 above]

## Recommendation
[section 7]

## Next Steps
[if adopting, what to do first]
```
