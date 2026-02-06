# Change Classification

> Use this template to document significant changes. Required for REFACTOR and BREAKING changes.
> Delete sections that don't apply.

## Change Type

- [ ] BUG_FIX - Fixes broken behavior (no interface changes)
- [ ] FEATURE - Adds new functionality (additive only)
- [ ] REFACTOR - Changes implementation without changing behavior
- [ ] PERFORMANCE - Improves performance without changing behavior
- [ ] BREAKING - Changes external interface or removes functionality
- [ ] MODEL - Adds/updates model in registry, benchmark results, or inference logic

## Ticket Reference

Refs: TCK-XXXXXXXX-XXX

## Summary

Brief one-line description of what changed and why.

---

## Files Modified

| File | Change |
|------|--------|
| `path/to/file.py` | Description of change |

## Behavior Changes

### Before

- Description of old behavior

### After

- Description of new behavior

## Test Coverage

- [ ] Existing tests still pass
- [ ] New tests added for changed behavior
- [ ] Manual verification completed
- [ ] Model inference verified (if applicable)

## Regression Risk Assessment

| Risk Level | Justification |
|------------|---------------|
| Low / Medium / High | Why this risk level |

## Prevention Applied

- [ ] Added/updated tests for this functionality
- [ ] Updated documentation
- [ ] Added logging for debugging
- [ ] Updated model registry (if model change)
- [ ] N/A

## Rollback Plan

```bash
git revert <commit-sha>
```
