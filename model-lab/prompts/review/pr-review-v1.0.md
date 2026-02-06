# PR Review Prompt

**Version**: 1.0  
**Purpose**: Review pull requests with evidence-based findings.

## Use When

- Reviewing a pull request before merge
- Verifying implementation matches ticket scope
- Checking for regressions or quality issues

## Non-Negotiable Rules

1. **Diff-only scope**: Only review what's in the diff
2. **Evidence required**: Every finding must cite specific code
3. **No drive-by fixes**: New issues → new tickets
4. **Ticket alignment**: Changes must match ticket scope

## Inputs

- PR URL or diff
- Related worklog ticket ID
- Original audit finding (if remediation PR)

## Steps

### 1. Scope Verification

```bash
# Get diff stats
git diff --stat origin/master...HEAD

# Check LOC changes
git diff --stat origin/master...HEAD | tail -1
```

Verify:
- [ ] Changes align with ticket scope
- [ ] No files outside scope modified
- [ ] LOC changes are reasonable

### 2. Code Review Checklist

For each file in diff:

**Correctness**
- [ ] Logic is correct
- [ ] Edge cases handled
- [ ] Error handling present

**Quality**
- [ ] Code style consistent
- [ ] No dead code introduced
- [ ] No TODOs without tickets

**Security**
- [ ] No secrets exposed
- [ ] Input validation present (if applicable)
- [ ] No injection vulnerabilities

**Testing**
- [ ] Tests added/updated for changes
- [ ] Tests cover edge cases
- [ ] No test skips without justification

### 3. Verification Evidence

Run and document:
```bash
# Tests
PYTHONPATH=. pytest -q tests/

# Build
cd client && npm run build

# Type check
PYTHONPATH=. mypy server/ harness/ --ignore-missing-imports
```

### 4. Document Findings

Format each finding:
```markdown
### Finding: [Title]

**Severity**: [High/Medium/Low/Info]
**File**: [path:line]
**Issue**: [description]
**Recommendation**: [what to do]
**Evidence**: [code snippet or command output]
```

## Output Format

### Review Summary

```markdown
## PR Review: [Title]

**Ticket**: TCK-YYYYMMDD-###
**Reviewer**: [name/agent]
**Verdict**: APPROVE / REQUEST_CHANGES / COMMENT

### Scope Alignment
- [x] Changes match ticket scope
- [x] No out-of-scope modifications

### Verification
- [x] Tests pass
- [x] Build succeeds
- [x] Type check passes

### Findings

[List findings or "No issues found"]

### Evidence Snippet
- **Observed**: [facts from review]
- **Inferred**: [logical conclusions]
- **Unknown**: [gaps]
```

## Stop Condition

Stop when:
- All diff files reviewed
- Findings documented with evidence
- Verdict rendered (APPROVE/REQUEST_CHANGES)
- Scope alignment verified

## Anti-Patterns

- ❌ Reviewing files not in diff
- ❌ Requesting changes outside PR scope
- ❌ Approving without running tests
- ❌ Findings without specific code references
