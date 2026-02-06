## Description
<!-- Describe your changes in detail -->

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update
- [ ] Model addition/update
- [ ] Performance improvement
- [ ] Refactor (no behavior change)

## Related Ticket
<!-- Link to worklog ticket: Refs TCK-YYYYMMDD-### -->

## Changes Made
<!-- List the main changes -->
- 
- 

## Testing
<!-- Describe the tests you ran -->
- [ ] `PYTHONPATH=. pytest -q tests/integration/test_backend_invariants.py`
- [ ] `PYTHONPATH=. pytest -q tests/api/test_artifact_download_security.py`
- [ ] `cd client && npm run build`
- [ ] Manual testing completed
- [ ] Model inference verified (if applicable)

## Screenshots (if applicable)
<!-- Add screenshots to help explain your changes -->

## LOC Review (Required for >10% changes)
<!-- Run: git diff --stat origin/master...HEAD -->
```
paste diff stats here
```

- [ ] Reviewed files with >10% LOC changes
- [ ] Changes are improvements (not just churn)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Evidence documented in worklog ticket
- [ ] No new warnings generated
- [ ] Tests added/updated
- [ ] All tests pass
- [ ] `docs/CLAIMS.md` updated (if making performance claims)
- [ ] `prompts/README.md` updated (if adding prompts)

## Evidence Snippet
<!-- Required: Label key claims -->
- **Observed**: 
- **Inferred**: 
- **Unknown**: 

## Reviewer Notes
<!-- Any specific areas you want reviewers to focus on -->
