# Copy Surface Map

**Generated:** 2026-02-13 14:16:24

---

## User-Facing Strings

### Error Messages (API Responses)

| Surface | String | Trigger | Location |
|---------|--------|---------|-----------|
| HTTP 400 | Invalid steps_preset. Available: [...] | Invalid preset in workbench | workbench.py:168 |
| HTTP 400 | Invalid config JSON | Malformed JSON in config | workbench.py:317 |
| HTTP 400 | Upload too large | File > max_upload | workbench.py:259 |
| HTTP 400 | Invalid artifact name | Path traversal attempt | runs.py:287-291 |
| HTTP 400 | Preview not supported for this artifact | Non-text preview | runs.py:370 |
| HTTP 403 | Artifact not downloadable | download=false | runs.py:499 |
| HTTP 403 | Invalid artifact path | Path traversal | runs.py:513 |
| HTTP 403 | Invalid artifact path | Bundle path mismatch | runs.py:341 |
| HTTP 404 | Run not found | Invalid run_id | runs.py:98,245 |
| HTTP 404 | Run not found in index | Run not indexed | runs.py:98 |
| HTTP 404 | Manifest not found | manifest.json missing | runs.py:473 |
| HTTP 404 | Artifact not found | Not in manifest | runs.py:332,495 |
| HTTP 404 | Audio not found | No processed audio | runs.py:267 |
| HTTP 404 | Artifact file not found on disk | File deleted | runs.py:516 |
| HTTP 404 | Experiment not found | Invalid exp_id | experiments.py:352 |
| HTTP 404 | Experiment input file missing | Input deleted | experiments.py:393 |
| HTTP 404 | No bundle artifacts found | Empty bundle | runs.py:423 |
| HTTP 404 | Use case not found | Invalid use_case_id | candidates.py:246 |
| HTTP 404 | Candidate not found | Invalid candidate_id | candidates.py:265 |
| HTTP 404 | User template not found | Invalid template name | pipelines.py:415,449 |
| HTTP 409 | RUNNER_BUSY | Max concurrent runs | workbench.py:293, experiments.py:421 |
| HTTP 409 | System busy | Max concurrent runs | lifecycle.py:37 |
| HTTP 413 | Preview too large | max_bytes exceeded | runs.py:372-378 |
| HTTP 500 | Failed to compute results | Result computation fails | runs.py:34 |
| HTTP 500 | Failed to get run status | Status fetch fails | runs.py:88 |
| HTTP 500 | Failed to read bundle manifest | Bundle corrupt | runs.py:277 |
| HTTP 500 | Failed to read manifest | Manifest corrupt | runs.py:479 |
| HTTP 500 | Artifact path missing | Internal error | runs.py:504 |

### Error Codes (JSON)

| Code | Meaning | Location |
|------|---------|----------|
| RUNNER_BUSY | Max concurrent runs reached | workbench.py:293 |
| INVALID_CANDIDATE | Candidate not found | experiments.py:154 |
| CANDIDATE_USE_CASE_MISMATCH | Candidate wrong use case | experiments.py:162 |
| INVALID_CANDIDATE_COUNT | <2 candidates | experiments.py:177 |
| INVALID_PIPELINE | Pipeline config invalid | experiments.py:258 |
| PREVIEW_TOO_LARGE | File too large | runs.py:375 |

### Progress Messages

| Surface | String | Trigger |
|---------|--------|---------|
| WebSocket | {"type": "step_started", "step": "..."} | Step begins |
| WebSocket | {"type": "step_progress", "step": "...", "progress_pct": N} | Progress update |
| WebSocket | {"type": "step_completed", "step": "..."} | Step completes |
| WebSocket | {"type": "step_failed", "step": "...", "error": {...}} | Step fails |
| WebSocket | {"type": "run_completed", "status": "COMPLETED"} | Run completes |
| WebSocket | {"type": "run_failed", "status": "FAILED"} | Run fails |
| WebSocket | {"type": "heartbeat", "run_id": "..."} | Keep-alive |

### Run Status Values

| Status | Meaning |
|--------|---------|
| QUEUED | Waiting to start |
| RUNNING | Currently processing |
| STALE | Running but no progress >90s |
| COMPLETED | Successfully finished |
| FAILED | Error occurred |
| CANCELLED | User killed |

### Step Status Values

| Status | Meaning |
|--------|---------|
| PENDING | Not yet executed |
| RUNNING | Currently executing |
| COMPLETED | Successfully finished |
| FAILED | Error occurred |
| SKIPPED | Skipped due to dependency |

### UI Labels (TypeScript)

| String | Component | File |
|--------|-----------|------|
| "Run" | Button | WorkbenchPage.tsx |
| "Compare" | Tab | ComparePage.tsx |
| "Results" | Tab | ResultsPage.tsx |
| "Experiments" | Nav | App.tsx |
| "Workbench" | Nav | App.tsx |
| "Run ID" | Label | RunDetail.tsx |
| "Status" | Label | RunDetail.tsx |
| "Steps" | Label | RunDetail.tsx |
| "Artifacts" | Label | RunDetail.tsx |
| "Transcript" | Tab | RunDetail.tsx |
| "Summary" | Tab | RunDetail.tsx |
| "Action Items" | Tab | RunDetail.tsx |

### Preset Names

| ID | Label | Description |
|----|-------|-------------|
| ingest | Ingest Only | Fast preprocessing: normalize audio |
| fast_asr_only | Fast ASR | Quick transcription |
| asr_with_diarization | ASR + Diarization | Transcription with speaker ID |
| diarization_focus | Diarization Focus | Speaker analysis |
| full | Full Pipeline | Complete pipeline |

### Use Case IDs

| ID | Title |
|----|-------|
| meeting_smoke | Meeting Smoke Test |
| asr_smoke | ASR Smoke Test |
| diarization_smoke | Diarization Smoke Test |
| asr_model_comparison | ASR Model Comparison |
| tts_quality | TTS Quality Test |
| latency_test | Real-time Latency Test |

### Artifact Names

| Name | Type | Description |
|------|------|-------------|
| manifest | Bundle.json | JSON manifest |
| audio_normalized.wav | Audio | Processed audio |
| transcript.json | JSON | Full transcript |
| summary.md | Markdown | Meeting summary |
| action_items.csv | CSV | Extracted action items |
| diarization.json | JSON | Speaker segments |
| chapters.json | JSON | Topic chapters |

---

## Strings Grouped by Flow

### Workbench Run (F001)
- "Invalid steps_preset. Available: [...]"
- "Invalid config JSON"
- "Upload too large"
- "RUNNER_BUSY"

### Experiment Creation (F002)
- "INVALID_CANDIDATE"
- "CANDIDATE_USE_CASE_MISMATCH"
- "INVALID_CANDIDATE_COUNT"
- "INVALID_PIPELINE"
- "Experiment not found"
- "Experiment input file missing"

### Run Status (F003)
- Step progress events
- Run completion events
- Heartbeat

### Results (F004)
- "Evaluation not available for this run"

### Artifacts (F005)
- "Artifact not found"
- "Artifact not downloadable"
- "Invalid artifact path"
- "Preview not supported"
- "Preview too large"

### Comparison (F006, F013, F014)
- "Run IDs must belong to this experiment"
- "Invalid artifact. Allowed: [...]"

### Kill/Retry (F007, F008)
- "Run not found"
- "System busy"
- "Failed to kill run"

### Pipeline (F015-F022)
- "Step '...' not found"
- "Template '...' not found"
- "User template '...' not found"

### Candidates (F031-F034)
- "Use case not found"
- "Candidate not found"

---

## Proof of Absence

### No Localization Keys
- No i18n/l10n implementation found
- No translation files
- All strings are hardcoded English

### No Monetization Strings
- No billing, quota, credit strings
- No upgrade prompts
- No paywall messages
- No tier labels (free/paid/etc)

### No Auth Strings
- No login/logout strings
- No permission denied messages
- No session expired messages

### No Admin Strings
- No user management strings
- No role labels (admin/user)
- No settings strings
