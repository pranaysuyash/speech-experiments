# State Validation Checklist

This document defines the **exact validation steps** required to verify temporal and liveness semantics. These are not opinions or UX notes — they are factual tests that must pass.

## Frozen Contract (Must Not Change)

### Temporal Truth
- `started_at` = identity and ordering only
- Clicking, viewing, or sorting **never** mutates identity

### Liveness Truth
- `updated_at` = sole progress signal
- No implicit assumptions about workers, threads, or queues

### STALLED Semantics
- Derived, not stored
- Reversible only via fresh progress signal
- Pipeline frozen, elapsed time continues

### Fail-Safe Bias
- Missing data biases toward RUNNING, never STALLED
- No pessimistic guesses

---

## Manual Validation Tests

### Test 1: Long-Running Active Job
**Purpose**: Verify RUNNING stability over time

**Steps**:
1. Start a run with audio > 5 minutes
2. Wait > 5 minutes
3. Observe `updated_at` in debug overlay
4. Verify `updated_at` continues advancing

**Expected Result**:
- ✅ MUST remain RUNNING (not STALLED)
- ✅ `secondsSinceProgress` stays < 90s
- ✅ Pipeline shows current step advancing

**Failure Mode**:
- ❌ If becomes STALLED despite active progress → fail-safe bias violated

---

### Test 2: Forced Stall
**Purpose**: Verify STALLED detection when backend stops

**Steps**:
1. Start a run
2. Kill worker process (or cut network to backend)
3. Wait ≥ 90 seconds
4. Observe `updated_at` in debug overlay

**Expected Result**:
- ✅ `updated_at` stops advancing
- ✅ MUST become STALLED at exactly 90s
- ✅ Pipeline shows last step frozen (not advancing)
- ✅ Elapsed time continues to increase
- ✅ Badge changes to orange "STALLED"

**Failure Mode**:
- ❌ If remains RUNNING after 90s → detection broken
- ❌ If `started_at` used instead of `updated_at` → wrong signal

---

### Test 3: Resurrection
**Purpose**: Verify STALLED → RUNNING reversibility

**Steps**:
1. Complete Test 2 (run is STALLED)
2. Restore worker/network
3. Observe `updated_at` in debug overlay

**Expected Result**:
- ✅ `updated_at` starts advancing again
- ✅ MUST return to RUNNING (badge blue)
- ✅ Pipeline resumes from last frozen step
- ✅ No UI hallucination — state derived purely from fresh `updated_at`

**Failure Mode**:
- ❌ If remains STALLED despite fresh `updated_at` → reversibility broken
- ❌ If requires page refresh to show RUNNING → state not reactive

---

### Test 4: Identity Isolation
**Purpose**: Verify "Last Started" is independent of view state

**Steps**:
1. Start Run A (e.g., at 10:00am)
2. Start Run B (e.g., at 10:05am)
3. Click Run A to view it
4. Return to runs list
5. Observe "Last Started Run" panel

**Expected Result**:
- ✅ "Last Started Run" MUST reference Run B (most recent `started_at`)
- ✅ This must be true even though Run A was last viewed
- ✅ `localStorage.lastActiveRunId` does NOT affect "Last Started" logic

**Failure Mode**:
- ❌ If "Last Started" shows Run A → identity confused with view state

---

## Gold Standard Test (Apply to Every Transition)

For each state transition observed:
- RUNNING → STALLED
- STALLED → RUNNING
- RUNNING → FAILED
- RUNNING → COMPLETED

**Ask**: *"Could this transition be reconstructed from a cold refresh using only API data?"*

**Valid Answer**: Yes — transition depends only on:
- `status` field from API
- `updated_at` timestamp
- `started_at` timestamp
- Current time (`Date.now()`)

**Invalid Answer**: No — transition depends on:
- UI state (`lastActiveRunId`, click history, etc.)
- Inferred state (guessing about workers, queues)
- Cached state (not re-derivable from API)

---

## Verification Checklist

After implementing validation scaffolding, verify:

- [ ] Grep for `90` → only appears in `lib/runProgress.ts` or config
- [ ] Grep for `updated_at` → never falls back to `started_at`
- [ ] `sortRuns` never references `lastActiveRunId` or view state
- [ ] Debug overlay visible in dev mode (`process.env.NODE_ENV === 'development'`)
- [ ] All transitions pass Gold Standard Test

---

## Notes

- **Do not optimize for reassurance** — optimize for truth
- **Confusing but true** → document, don't fix
- **Reassuring but false** → stop-the-line bug
- **No UI hallucination** — state must never advance without backend signal

---

## External User Cold Load Tests (Automated)

### 1. Hard Refresh Torture
- **Scenario**: 5 rapid refreshes on `RUNNING` and `FAILED` states.
- **Result**: PASS. UI consistently reconstructed state from API. Status badge remained stable.

### 2. Navigation Consistency
- **Scenario**: Navigate away (to external site) and return.
- **Result**: PASS. Run details re-fetched correctly. No regression to "Loading" or "Error" unnecessarily.

### 3. Multi-Tab Access
- **Scenario**: Open run in multiple tabs simultaneously.
- **Result**: PASS. State is synchronized via API polling/fetching. No conflicts observed (read-only view).
