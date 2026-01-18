# Results & Compare UX Verification Report

## Phase 1: RunDetailPage Integrity

**Goal**: Ensure `RunDetailPage` is a self-contained, correct unit governed by strict backend state.

### Audit Findings
*   **Entry Contract Violation**: The component consumed `api.getTranscript()` directly, bypassing the `Run` object status.
*   **Lie by Omission**: `QUEUED` and `FAILED` runs appeared as "Completed with 0 words" because `transcript.segments` was empty.
*   **Lifecycle Drift**: No UI states for `QUEUED` or `FAILED`.

### Fixes Applied (`RunDetail.tsx`)
1.  **Status-First Loading**: Component now polls `api.getRunStatus(runId)` before fetching any data.
2.  **Explicit State Handling**:
    *   `QUEUED`/`RUNNING`: Renders a Loading spinner and 'Current Step' info.
    *   `FAILED`: Renders a Red error state with error code/message.
    *   `COMPLETED`: Only then renders the Transcript/Media view.
3.  **Artifact Gating**: `loadDetail` and `loadMeetingPack` are strictly gated behind `status === 'COMPLETED'`.

## Phase 2: Results Surface (ExperimentPage)

**Goal**: Ensure Results views are pure projections of experiment data without side effects.

### Audit Findings
*   **Read-Only Violation**: `ExperimentPage` contained logic to "auto-start" runs (`loadAndStart`) and "queue-manage" (`poll` checking for empty slots).
*   **Drift Risk**: This logic contradicted the "Workbench initiates" contract and hid backend failures by silently retrying.

### Fixes Applied (`ExperimentPage.tsx`)
1.  **Stripped Side Effects**: Removed `startExperimentAll` and `startExperimentNext` calls entirely.
2.  **Pure Polling**: The page now only polls `api.getExperiment()` to reflect the current state of truth.
3.  **Flow Verification**: Confirmed that `WorkbenchPage` handles the `startExperimentAll` call and error handling before navigation.

## Phase 3: Compare UX

**Goal**: Ensure comparison is only available when valid.

### Audit Findings
*   **Premature Activation**: Compare fetching triggered immediately upon ID selection, potentially receiving partial/invalid data for running experiments.

### Fixes Applied
1.  **Terminal Guard**: Added strict check in `useEffect`:
    ```typescript
    if (!lRun || !rRun || !isTerminal(lRun.status) || !isTerminal(rRun.status)) return;
    ```
    This ensures `api.getExperimentComparison` is only called when both runs are `COMPLETED` or `FAILED`.

## Final State
*   **Workbench**: Enforces 1 vs 2 candidates. Starts runs. Handles start errors.
*   **Experiment Page**: Read-only dashboard. Gates Compare.
*   **Run Detail**: Status-aware. Honest about failures.
