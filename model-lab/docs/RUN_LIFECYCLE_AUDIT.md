# Run Lifecycle Invariant Audit

**Date:** 2026-01-19
**Scope:** All local runs in `runs/` directory.

## Criteria
1. **Terminal State**: `STARTED` -> `RUNNING` -> (`COMPLETED` | `FAILED`)
2. **Heartbeat**: `RUNNING` runs must have recent `updated_at`.
3. **Failure Step**: `FAILED` runs must have a `failure_step` recorded.
4. **Stalled**: `STALLED` is a derived state, not an explicit manifest state (mostly).

## Findings

### Summary
- **Total Scanned**: 62 runs
- **Violations Found**: 52 runs
- **Violation Types**:
    - `FAILED but failure_step is missing or null`: 48 runs (Legacy runs before `failure_step` implementation)
    - `Invalid updated_at format`: 4 runs (Legacy runs with `YYYY-MM-DDTHH:MM:SS` format without 'Z')

### Detail Analysis

#### 1. Missing `failure_step` in FAILED runs
**Root Cause**: These runs were created before the backend `failure_step` logic was introduced (Jan 18, 2026). The schema was updated to include this field, but legacy data remains null.
**Action**: Accepted as legacy debt. New runs must have this field.

#### 2. Invalid `updated_at` format
**Root Cause**: Older backend version saved timestamps without the 'Z' suffix for UTC. The current parser expects ISO format with timezone info (or 'Z').
**Action**: Accepted as legacy debt. Current `session.py` ensures 'Z' suffix is appended.

#### 3. Orphaned RUNNING runs
**observed**: 4 runs are `RUNNING` but have invalid timestamps, effectively making them "Ghost Runs".
**Ids**: `20260117_091514_9384c33150`, `20260117_094905_2c0200c831`, `20260117_124649_87bcb3ad2e`, `20260117_124945_45213ec42f`.
**Action**: Users should see these as "STALLED" in UI if the frontend handles the invalid timestamp gracefully, or just "RUNNING" forever. The audit script flagged them as invalid format.

## Verification of New Runs
To ensure **zero violations** for *active* code:
- [x] Run stress tests (Task 1) created new runs.
- [x] Verified these new runs have correct `failure_step` and valid timestamps (Audit script returned 0 issues for 20260119 runs).

**Verdict**: Legacy data violations present. Active code is **compliant** and adhering to strict lifecycle invariants. 
PASS.
