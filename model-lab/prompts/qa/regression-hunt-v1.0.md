# Regression Hunt Prompt

**Version**: 1.0  
**Purpose**: Systematically find and document regressions after changes.

## Use When

- After a significant refactor
- When tests start failing unexpectedly
- Before a release
- After merging multiple PRs

## Non-Negotiable Rules

1. **Baseline first**: Know what "working" looks like
2. **Bisect if needed**: Use git bisect for hard-to-find regressions
3. **Document everything**: Every finding gets a ticket
4. **No silent failures**: Flaky tests are regressions too

## Inputs

- Baseline commit (known good state)
- Current commit (potentially broken)
- Test suite to run

## Steps

### 1. Establish Baseline

```bash
# Checkout known good commit
git checkout <baseline-commit>

# Run full test suite
PYTHONPATH=. pytest -q tests/ > baseline_results.txt 2>&1

# Run benchmarks
PYTHONPATH=. python -m harness.run --benchmark > baseline_benchmark.txt 2>&1

# Return to current
git checkout -
```

### 2. Run Current Tests

```bash
# Full test suite
PYTHONPATH=. pytest -q tests/ > current_results.txt 2>&1

# Benchmarks
PYTHONPATH=. python -m harness.run --benchmark > current_benchmark.txt 2>&1
```

### 3. Compare Results

```bash
# Diff test results
diff baseline_results.txt current_results.txt

# Diff benchmarks
diff baseline_benchmark.txt current_benchmark.txt
```

### 4. Categorize Findings

**True Regressions**
- Test passed before, fails now
- Performance degraded significantly (>10%)
- Feature no longer works

**Flaky Tests**
- Passes sometimes, fails others
- Timing-dependent
- External dependency issues

**Expected Changes**
- Behavior intentionally changed
- Test needs update, not code

**False Positives**
- Environment differences
- Data differences
- Test infrastructure issues

### 5. Document Each Regression

```markdown
### Regression: [Title]

**Type**: True Regression / Flaky / Expected
**Severity**: P0 / P1 / P2
**Introduced**: [commit hash or range]

**Evidence**:
- Baseline: [passed/metric]
- Current: [failed/metric]

**Root Cause**: [if known]

**Fix**: [proposed fix or ticket reference]
```

### 6. Git Bisect (if needed)

```bash
git bisect start
git bisect bad HEAD
git bisect good <baseline-commit>

# Run test that fails
git bisect run pytest tests/path/to/failing_test.py

# When done
git bisect reset
```

## Output Format

```markdown
# Regression Hunt Report

**Date**: YYYY-MM-DD
**Baseline**: [commit hash]
**Current**: [commit hash]
**Agent**: [name]

## Summary

- **Total tests**: [number]
- **Passed**: [number]
- **Failed**: [number]
- **New failures**: [number]
- **Fixed**: [number]

## Regressions Found

### P0 (Critical)
[list]

### P1 (High)
[list]

### P2 (Medium)
[list]

## Flaky Tests Identified
[list]

## Tickets Created
- TCK-YYYYMMDD-### - [title]
- TCK-YYYYMMDD-### - [title]

## Recommendations

[next steps]
```

## Stop Condition

Stop when:
- All test differences categorized
- Tickets created for true regressions
- Report documented
- Baseline re-established or regression fixed
