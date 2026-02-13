# Detailed Flow Specifications

**Generated:** 2026-02-13 14:16:24

---

# F001 Workbench Run

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Process audio file through configurable pipeline
- **Primary components:** WorkbenchPage.tsx, workbench.py, SessionRunner
- **Boundaries crossed:** Client → API → Worker → Filesystem

## Entry Points (ALL)
- **UI actions:** WorkbenchPage.tsx file upload → POST /api/workbench/runs
- **Auto triggers:** None
- **External triggers:** None

**Evidence:**
- `client/src/pages/WorkbenchPage.tsx` - Upload component
- `server/api/workbench.py:266` - POST /workbench/runs endpoint

## Preconditions / Dependencies
- **Permissions:** None (no auth)
- **Settings/flags:** Steps preset, config overrides, preprocessing
- **Auth/session state:** None required
- **Required resources:** Audio file, disk space, model files

**Evidence:**
- `server/api/workbench.py:296` - max_upload size limit from env
- `harness/session.py:168` - Input path validation

## State Model
- **states:** QUEUED → RUNNING → (COMPLETED | FAILED)
- **transitions:**
  - Upload → QUEUED (via API response)
  - Worker spawn → RUNNING
  - Step completion → RUNNING (with progress updates)
  - Final step → COMPLETED or FAILED

**Evidence:**
- `server/services/lifecycle.py:80-110` - Initial manifest creation
- `harness/session.py` - SessionRunner state management

## Sequence (Happy Path)
1. User selects audio file in UI
2. UI calls POST /api/workbench/runs with FormData
3. Server validates file, computes SHA256
4. Server creates SessionRunner with configured steps
5. Server writes run_request.json and initial manifest
6. Server spawns background worker subprocess
7. Worker processes each step sequentially
8. Progress events written to events.jsonl
9. WebSocket clients receive progress updates
10. Final status written to manifest.json

**Evidence:**
- `server/api/workbench.py:304` - File save and SHA256
- `server/services/lifecycle.py:133-142` - Worker spawn
- `harness/session.py:350-500` - Step execution

## Alternate Paths + Micro-Flows

### Empty State / First Success
- No special empty state - any audio file accepted
- First successful run creates session directory structure

### Permission Denied / Recovery
- N/A - No permissions in this app

### Offline/Degraded
- Network required for model downloads (first run)
- Offline not explicitly handled - fails with network error

### Retry/Fallback
- Retry from failed step: POST /api/runs/{id}/retry
- Retry from specific step: POST /api/runs/{id}/retry with from_step

**Evidence:**
- `server/api/lifecycle.py:28` - Retry endpoint
- `server/services/lifecycle.py:282-420` - Retry logic

### Paywall/Upgrade/Restore
- Not applicable - no monetization

### Reset/Recovery/Safe Mode
- Kill running job: POST /api/runs/{id}/kill
- Re-run with config changes: POST /api/runs/{id}/rerun

**Evidence:**
- `server/api/lifecycle.py:13` - Kill endpoint
- `server/api/runs.py:64` - Rerun endpoint

## UI Copy + Messaging
- Run created: "run_id: xxx"
- Progress: Step name + percentage
- Completion: "COMPLETED" or "FAILED"
- Error: Error code + message from step

**Evidence:**
- `harness/session.py` - Progress events
- `server/api/runs.py:77-88` - Status endpoint

## Monetization/Entitlements
**Proof of absence:** No billing, quota, or entitlement code found in codebase.

## Data Lifecycle
- **Data created:** manifest.json, run_request.json, events.jsonl, step artifacts
- **Storage locations:** runs/sessions/{input_hash}/{run_id}/
- **Retention/delete controls:** No automatic deletion
- **Migration impacts:** None - flat file storage

**Evidence:**
- `harness/session.py:193` - session_dir structure

## Observability
- **Logs:** worker.log in run directory
- **Events:** events.jsonl with step progress
- **Metrics:** Duration, step timing in manifest
- **Traces:** None (no OTEL configured by default)

**Evidence:**
- `server/services/lifecycle.py:113` - Log file creation
- `server/api/ws_runs.py` - WebSocket event streaming

## Failure Modes (10+)

| Detection Point | Handling | User-Visible Outcome | Evidence |
|----------------|----------|---------------------|----------|
| File too large | HTTP 413 | "Upload too large" | workbench.py:259 |
| Invalid preset | HTTP 400 | "Invalid steps_preset" | workbench.py:168 |
| Runner busy | HTTP 409 | "RUNNER_BUSY" | workbench.py:293 |
| Input file missing | FileNotFoundError | "Input file not found" | session.py:190 |
| Model OOM | E_MODEL_OOM | Step fails with OOM | errors.py:100 |
| Audio corrupt | E_AUDIO_CORRUPT | Step fails | errors.py:109 |
| Audio too short | E_AUDIO_TOO_SHORT | Step fails | errors.py:112 |
| Network error | E_NETWORK_ERROR | Step fails, retry possible | errors.py:128 |
| Disk full | E_DISK_FULL | Step fails | errors.py:120 |
| Step timeout | E_STEP_TIMEOUT | Step fails, retry possible | errors.py:116 |
| Worker crash | Process exit | Status: FAILED | lifecycle.py:164 |
| Manifest corrupt | JSONDecodeError | Status: FAILED | runs.py:104 |

## Findings
- No auth layer - any user can process files
- No rate limiting
- No automatic cleanup of old runs

---

# F002 Experiment Creation

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Compare multiple models/pipelines on same input
- **Primary components:** ExperimentPage.tsx, experiments.py, SessionRunner
- **Boundaries crossed:** Client → API → Worker (x2 for A/B)

## Entry Points (ALL)
- **UI actions:** ComparePage.tsx → POST /api/experiments
- **Auto triggers:** None
- **External triggers:** None

**Evidence:**
- `client/src/pages/ComparePage.tsx`
- `server/api/experiments.py:128` - POST /experiments

## Preconditions / Dependencies
- **Permissions:** None
- **Settings/flags:** Use case, candidate IDs, pipeline config
- **Auth/session state:** None required
- **Required resources:** Audio file, 2+ candidates configured

**Evidence:**
- `server/api/experiments.py:144-207` - Candidate resolution

## State Model
- **states:** CREATED → (QUEUED_A → RUNNING_A → COMPLETED_A) + (QUEUED_B → RUNNING_B → COMPLETED_B)
- **transitions:** Parallel or sequential execution of slots

**Evidence:**
- `docs/experiment_lifecycle.md` - State machine diagram

## Sequence (Happy Path)
1. User uploads audio file in Compare UI
2. User selects use case and candidates
3. UI calls POST /api/experiments with FormData
4. Server creates experiment directory
5. Server saves input file with SHA256
6. Server creates experiment_request.json
7. Server initializes experiment_state.json with QUEUED slots
8. Server returns experiment_id and candidates
9. User triggers start via start-all or start-next

**Evidence:**
- `server/api/experiments.py:209-344` - Creation flow

## Alternate Paths + Micro-Flows

### Invalid Candidate
- Returns 400 with INVALID_CANDIDATE error code

### Candidate Use Case Mismatch
- Returns 400 with CANDIDATE_USE_CASE_MISMATCH

### Insufficient Candidates
- Falls back to presets if <2 candidates

**Evidence:**
- `server/api/experiments.py:150-207` - Validation and fallback

## UI Copy + Messaging
- Experiment created: experiment_id returned
- Candidate status: QUEUED/RUNNING/COMPLETED/FAILED
- Provenance status: VERIFIED/CORRUPTED/UNVERIFIED

**Evidence:**
- `server/api/experiments.py:80-98` - Provenance verification

## Findings
- Same as F001 for failure modes
- Additional: Provenance tracking for reproducibility

---

# F003 Run Status Monitoring (WebSocket)

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Real-time view of run progress
- **Primary components:** ws.ts, ws_runs.py, events.jsonl
- **Boundaries crossed:** Client ↔ Server (bidirectional)

## Entry Points (ALL)
- **UI actions:** Client connects to WebSocket /api/runs/{run_id}/ws
- **Auto triggers:** On run start, UI auto-connects
- **External triggers:** None

**Evidence:**
- `client/src/lib/ws.ts` - WebSocket client
- `server/api/ws_runs.py:52` - WebSocket endpoint

## Preconditions / Dependencies
- **Permissions:** None
- **Settings/flags:** None
- **Auth/session state:** None required
- **Required resources:** Run must exist, events.jsonl accessible

## State Model
- **Connection states:** CONNECTING → CONNECTED → DISCONNECTED
- **Event types:** step_started, step_progress, step_completed, step_failed, run_completed, run_failed, heartbeat

**Evidence:**
- `server/api/ws_runs.py:79-88` - Event streaming

## Sequence (Happy Path)
1. Client connects to /api/runs/{run_id}/ws
2. Server sends all existing events immediately
3. Server tails events.jsonl for new events
4. Server sends heartbeat every 5 seconds
5. Server closes when run reaches terminal state

**Evidence:**
- `server/api/ws_runs.py:79-148` - Event handling

## Failure Modes

| Detection Point | Handling | User-Visible Outcome |
|----------------|----------|---------------------|
| Run not found | HTTP 4004 | Error message |
| Connection timeout | Close 4008 | Timeout message |
| Client disconnect | Log only | None |

---

# F004 View Run Results

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** View semantic results of processed audio
- **Primary components:** ResultsPage.tsx, results_v1.py
- **Boundaries crossed:** API → Filesystem → Client

## Entry Points (ALL)
- **UI actions:** ResultsPage → GET /api/runs/{id}/results
- **Auto triggers:** None
- **External triggers:** None

## Sequence
1. User navigates to Results page
2. UI calls GET /api/runs/{id}/results
3. Server loads eval.json if present
4. Server projects to ResultSummary schema
5. Returns to client

**Evidence:**
- `server/api/runs.py:20-34` - Results endpoint
- `server/services/results_v1.py` - Result projection

---

# F005 Download Artifacts

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Download processed outputs (transcript, summary, audio, bundle)
- **Primary components:** RunDetail.tsx, runs.py
- **Boundaries crossed:** API → Filesystem → Client

## Entry Points (ALL)
- **UI actions:** Download buttons → GET /api/runs/{id}/bundle.zip
- **Auto triggers:** None
- **External triggers:** None

## Sequence
1. User clicks download
2. Server validates artifact exists in bundle_manifest.json
3. Server serves file via FileResponse
4. Client initiates download

**Evidence:**
- `server/api/runs.py:388-439` - Bundle zip download

## Security
- Path traversal prevention via safe_file_path()
- Artifact allowlisting via bundle_manifest.json
- Downloadable flag check

**Evidence:**
- `server/api/runs.py:280-310` - Path validation

---

# F006 Compare Runs

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Compare two runs side-by-side
- **Primary components:** runs_service.py
- **Boundaries crossed:** API → Multiple runs → Client

## Entry Points
- **UI:** Compare tab → GET /runs/compare?run_a=X&run_b=Y
- **Evidence:** `server/api/runs.py:55-61`

---

# F007 Retry Failed Run

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Retry failed run from beginning or specific step
- **Primary components:** lifecycle.py

## Entry Points
- **UI:** Run detail → POST /runs/{id}/retry
- **Evidence:** `server/api/lifecycle.py:28`

---

# F008 Kill Running Run

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Cancel running job
- **Primary components:** lifecycle.py

## Entry Points
- **UI:** Run detail → POST /runs/{id}/kill
- **Evidence:** `server/api/lifecycle.py:13`

---

# F009 Search Transcript

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** Search within transcript
- **Primary components:** runs_index.py

## Entry Points
- **UI:** Search bar → GET /runs/{id}/search?q=query
- **Evidence:** `server/api/runs.py:238-247`

---

# F010 Browse Presets

## Summary
- **Category:** User-Facing
- **Status:** Implemented
- **User goal:** See available pipeline presets
- **Primary components:** workbench.py

## Entry Points
- **UI:** Dropdown → GET /workbench/presets
- **Evidence:** `server/api/workbench.py:94`

---

# F011-F014: Experiment Flows

See F002 for experiment creation. Additional flows:
- F011: Start single experiment run
- F012: Start all experiment runs
- F013: Compare experiment results (semantic)
- F014: Compare experiment artifacts (text diff)

**Evidence:** `server/api/experiments.py:357-580`

---

# F015-F022: Pipeline Configuration Flows

## F015 View Available Steps
- **Endpoint:** GET /pipelines/steps
- **Evidence:** `server/api/pipelines.py:114`

## F016 View Pipeline Templates
- **Endpoint:** GET /pipelines/templates
- **Evidence:** `server/api/pipelines.py:144`

## F017 Validate Pipeline Config
- **Endpoint:** POST /pipelines/validate
- **Evidence:** `server/api/pipelines.py:174`

## F018 Resolve Dependencies
- **Endpoint:** POST /pipelines/resolve
- **Evidence:** `server/api/pipelines.py:220`

## F019-F021: User Templates CRUD
- **Evidence:** `server/api/pipelines.py:347-454`

---

# F023-F030: Data/Results Flows

All implemented with evidence in respective API files.

---

# F031-F034: Candidate/Use Case Flows

All implemented in `server/api/candidates.py:235-266`

---

# F035 Stream Processed Audio

## Summary
- **Category:** Audio Playback
- **Status:** Implemented
- **User goal:** Playback processed audio in browser

## Entry Points
- **UI:** Audio player → GET /runs/{id}/audio
- **Evidence:** `server/api/runs.py:250-267`

---

## Missing Expected Flows (None)

After negative space search, no material flows were identified as missing that should exist in this audio processing platform.
