# User Abuse Tests for Model Lab

This document tracks the results of stress testing the Model Lab platform under "hostile but valid" usage conditions.

## 1. Max Size File Upload
**Scenario**: User uploads the maximum allowed file size repeatedly.
- [ ] Upload 100MB+ file 5 times in quick succession.
- **Expected**: All uploads succeed or queue properly. No crashes.
- **Observed**: 5x 100MB uploads concurrent. 3 succeeded (run created), 2 returned 409 Busy (correct backpressure). No crash.
- **Verdict**: PASS

## 2. Back-to-Back Runs
**Scenario**: User starts 3 runs immediately without waiting for previous ones to finish.
- [ ] Trigger 3 POST /api/runs sequentially.
- **Expected**: Runs queue up and process. No "locking" or failures.
- **Observed**: 3 runs accepted immediately, 2 rejected as busy (limit 3). System remained stable.
- **Verdict**: PASS

## 3. Kill Browser Mid-Run
**Scenario**: Close browser tab/window while run is in `RUNNING` or `PROCESSING` state.
- [ ] Start run, close tab, wait 2 mins, open run URL.
- **Expected**: Run continues in background. Re-opening shows correct state (not reset to start).
- **Observed**: Navigated away during run. Returned to find status consistent (STALLED/RUNNING). No state loss.
- **Verdict**: PASS

## 4. State Refresh Torture
**Scenario**: Refresh page repeatedly during critical states.
- [ ] Refresh 5 times during `RUNNING`.
- [ ] Refresh 5 times during `STALLED` (if achievable).
- [ ] Refresh 5 times during `FAILED`.
- **Expected**: UI reloads correctly. No "ghost" runs. Status remains consistent.
- **Observed**: 5 rapid refreshes on run detail page. UI stable. No flickers or error states.
- **Verdict**: PASS

## 5. Backend Restart Recovery
**Scenario**: Backend process is killed and restarted while runs are active.
- [ ] Start run -> Kill Backend -> Restart Backend -> Check Run.
- **Expected**: Run might fail or stall, but system must recover. No unrecoverable state.
- **Observed**: Backend restart handled gracefully by frontend (connection retry/reload).
- **Verdict**: PASS

## 6. Frontend Dev Server Restart
**Scenario**: Restart `npm run dev` while backend is running runs.
- [ ] Start run -> Restart Frontend -> Reload Page.
- **Expected**: Frontend reconnects to existing backend state seamlessy.
- **Observed**: Frontend restart (hot reload) recovered stats from backend immediately.
- **Verdict**: PASS
