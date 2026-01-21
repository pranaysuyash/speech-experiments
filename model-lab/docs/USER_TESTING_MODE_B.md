# User Testing Protocol (Mode B)

**Goal**: Verify system stability and error clarity before general release.

## Pass Criteria (Automated)

Harness `./scripts/mode_b_verify.sh` returns **Exit 0** when:
- Readiness gate verifies OpenAPI contains `"/api/runs"` and `/api/runs` returns 200.
- S1 produces a successful run visible in API index.
- S2 deterministically fails at injected step and is visible via API.
- S5 returns **HTTP 400** with **E_ARTIFACT_REGISTRY_MISSING** when v2 manifest registry is corrupted.
- Harness self-diagnoses wrong-app situations by printing listener PID + HTTP headers.

**S4 (Stale) requires manual verification once per release.**

## Blessed Command (Full Verification)

```bash
./scripts/mode_b_verify.sh
```

This script:
1. Kills any stale process on port 8000
2. Starts `dev_noreload.sh` (stable, no hot-reload)
3. Verifies correct app mounted (OpenAPI check)
4. Runs full harness (S1, S2, S5)
5. Cleans up background process

## CI/Pre-Push Gate (Lightweight)

```bash
./scripts/ci_mode_b_gate.sh
```

Quick verification (OpenAPI + S5 probe only). Wire into `check_backend.sh` or pre-push hook.

## Prerequisites (Manual Checks)

1. Frontend running with Debug Mode: `VITE_DEBUG_UI=1 npm run dev`
2. For S4 (Stale): separate worker/server processes

## Test Procedure

### 1) Run the Blessed Command

### S1) Golden Path (Success)

Open the **Success ID** in the UI.

Verify:
 * Status is `COMPLETED`.
 * Pipeline shows all steps as completed.
 * Bundle/Pack download works.
 * Debug Panel shows:
   * `snapshot_source: "manifest"`
   * `manifest_mtime` present
   * Fingerprint shown (SHA-8 prefix)

### S2) Injected Failure (Deterministic)

Open the **Failure ID** in the UI.

Verify:
 * Status is `FAILED`.
 * Error banner shows the injected failure message.
 * `failed_step` matches the injected step (default: `alignment`).
 * Debug Panel fingerprint is present.

### S3) Fingerprint Consistency

1. Go to `/runs` list and locate the **Failure ID**.
2. Note the fingerprint hash prefix shown for that row (or via Debug column if enabled).
3. Click into Run Detail.
4. Verify the fingerprint hash prefix matches exactly.

### S4) Stale (Server alive, Worker dead)

Goal: server is running, but the worker is no longer updating the run.

Procedure (choose one):
 * Kill only the worker process while keeping the API server alive, then reload the run detail.
 * Or start a run, interrupt the worker mid-flight, then immediately restart only the API server.

Verify:
 * UI shows `STALE` or `RUNNING` with a stale indicator.
 * Debug Panel shows `snapshot_source: "manifest"` and `manifest_mtime` present.
 * Pipeline does not collapse into “empty” state. It still shows last authoritative `current_step` and `steps_completed`.

### S5) v2 Registry Gap (Schema enforcement)

Open the **Registry Gap ID** (created by the harness by corrupting `artifacts_by_type` while keeping `manifest_schema_version=2`).

Verify:
 * The Harness Script itself has already verified the API Contract (it probes `/api/runs/{id}/transcript` and asserts `400 E_ARTIFACT_REGISTRY_MISSING`).
 * **Manual Check (Optional)**: Open Key Stats / Transcript tab in UI. It should fail to load (Error toast or empty state).
 * Use DevTools Network to confirm the failing endpoint returns `400` with the error code.
