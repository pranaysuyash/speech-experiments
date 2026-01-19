# Error Surface Invariants

**Goal**: Ensure the system never lies about failure. If it fails, it must say FAILED. If it doesn't know, it must say UNKNOWN.

## 1. Explicit Failure UI
- **Invariant**: A run in `FAILED` status must never show "Completed" or success iconography.
- **Audit**: `RunDetail.tsx` (Lines 381-476) checks `isFailed` explicitly.
    - Header: "Run Failed" (Red)
    - Icon: ❌ (Red)
    - Message: Shows `error_code` and `error_message` from API or "Run failed without detailed error message."
- **Verdict**: PASS.

## 2. No "Ghost" Success
- **Invariant**: Partial results (e.g. Ingest done, ASR failed) must be marked `PARTIAL`.
- **Audit**: `RunDetail.tsx` (Line 490) checks `result?.quality_flags.is_partial`.
    - Badge: "PARTIAL" (Orange)
- **Verdict**: PASS.

## 3. Truthful Error Codes
- **Invariant**: Frontend displays raw error code if available, not a generic "Something went wrong".
- **Audit**: `RunDetail.tsx` displays `{status.error_code}` and `{status.error_message}` directly.
- **Verdict**: PASS.

## 4. Derived Stalled State
- **Invariant**: If backend dies, frontend must eventually show STALLED, not infinite spinner.
- **Audit**: `deriveProgressSignal` and `isStalled` logic handles this. `RunDetail.tsx` (Line 213) shows "Run Stalled" if `isStalled` is true.
- **Verdict**: PASS.
