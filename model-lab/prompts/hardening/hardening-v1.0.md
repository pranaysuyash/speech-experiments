# Hardening Prompt

**Version**: 1.0  
**Purpose**: Production hardening for a specific scope area.

## Use When

- Preparing code for production deployment
- Improving reliability of a specific component
- Adding error handling, logging, monitoring

## Non-Negotiable Rules

1. **One scope**: Harden one area per run (e.g., "API error handling")
2. **No behavior change**: Hardening should not change functionality
3. **Backward compatible**: Existing integrations must work
4. **Tested**: Every hardening change must have tests

## Inputs

- Scope area (e.g., "server/api error handling")
- Current issues or gaps
- Target reliability level

## Steps

### 1. Define Scope Contract

```markdown
**Hardening Scope**: [area]
**Files in scope**: [list]
**Behavior change allowed**: NO
**Goal**: [specific hardening goal]
```

### 2. Audit Current State

```bash
# Find error handling patterns
rg -n "try:|except|raise|Error" [scope-path]

# Find logging
rg -n "logger\.|logging\." [scope-path]

# Find TODO/FIXME related to reliability
rg -n "TODO|FIXME|HACK" [scope-path]
```

### 3. Hardening Checklist

**Error Handling**
- [ ] All exceptions caught appropriately
- [ ] Errors logged with context
- [ ] User-facing errors are clear
- [ ] No silent failures

**Logging**
- [ ] Key operations logged
- [ ] Log levels appropriate
- [ ] Sensitive data not logged
- [ ] Request IDs for tracing

**Input Validation**
- [ ] All inputs validated
- [ ] Type checking present
- [ ] Bounds checking where needed
- [ ] Sanitization for security

**Resource Management**
- [ ] Connections properly closed
- [ ] Memory leaks addressed
- [ ] Timeouts configured
- [ ] Retry logic where needed

**Configuration**
- [ ] Secrets from environment
- [ ] Defaults are safe
- [ ] Configuration validated at startup

### 4. Implement Hardening

For each hardening change:
1. Make minimal change
2. Add/update tests
3. Verify no behavior change

### 5. Verification

```bash
# Run tests
PYTHONPATH=. pytest -q tests/

# Check no behavior change
# Run integration tests
PYTHONPATH=. pytest -q tests/integration/
```

## Output Format

```markdown
# Hardening Report: [Scope]

**Ticket**: TCK-YYYYMMDD-###
**Date**: YYYY-MM-DD

## Changes Made

| File | Change | Category |
|------|--------|----------|
| path/file.py | Added error handling | Error Handling |

## Before/After

### Before
[issue description]

### After
[improvement description]

## Tests Added

- [ ] test_error_handling.py
- [ ] test_validation.py

## Verification

- [ ] All tests pass
- [ ] No behavior change
- [ ] Integration tests pass
```

## Stop Condition

Stop when:
- All checklist items addressed (or documented as out-of-scope)
- Tests pass
- No behavior changes introduced
- Hardening documented
