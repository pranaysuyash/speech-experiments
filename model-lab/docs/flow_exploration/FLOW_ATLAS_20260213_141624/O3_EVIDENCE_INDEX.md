# Evidence Index

**Generated:** 2026-02-13 14:16:24

---

## Evidence by Flow

### F001 Workbench Run

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/workbench.py | line 266: POST /workbench/runs |
| UI Component | client/src/pages/WorkbenchPage.tsx | Upload file handling |
| Runner | harness/session.py | SessionRunner class |
| Worker Launch | server/services/lifecycle.py | line 45: launch_run_worker |
| Step Execution | harness/session.py | line 350-500: step execution |
| Progress Events | harness/session.py | StepProgress dataclass |
| Status Endpoint | server/api/runs.py | line 77: GET /runs/{id}/status |

### F002 Experiment Creation

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/experiments.py | line 128: POST /experiments |
| UI Component | client/src/pages/ComparePage.tsx | Experiment creation |
| Candidate Resolution | server/api/experiments.py | line 144-207 |
| State Management | server/api/experiments.py | experiment_state.json |
| Provenance | server/api/experiments.py | line 79-98 |

### F003 Run Status Monitoring (WebSocket)

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| WebSocket Server | server/api/ws_runs.py | line 52: /api/runs/{run_id}/ws |
| WebSocket Client | client/src/lib/ws.ts | WebSocket connection |
| Event Format | harness/session.py | StepProgress.to_dict() |
| Event Persistence | server/api/ws_runs.py | line 73: events.jsonl |

### F004 View Run Results

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/runs.py | line 20: GET /runs/{id}/results |
| Result Projection | server/services/results_v1.py | compute_result_v1() |
| Schema | client/src/lib/api.ts | ResultSummary interface |

### F005 Download Artifacts

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| Bundle Download | server/api/runs.py | line 388: GET /runs/{id}/bundle.zip |
| Artifact Endpoint | server/api/runs.py | line 449: GET /runs/{id}/artifacts/{id} |
| Security | server/services/safe_files.py | safe_file_path() |
| Allowlist | server/api/runs.py | line 313-348 |

### F006 Compare Runs

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/runs.py | line 55: GET /runs/compare |
| Service | server/services/runs_service.py | compare_runs() |

### F007 Retry Failed Run

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/lifecycle.py | line 28: POST /runs/{id}/retry |
| Logic | server/services/lifecycle.py | line 282-420 |
| Step Invalidation | server/services/lifecycle.py | line 290-345 |

### F008 Kill Running Run

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/lifecycle.py | line 13: POST /runs/{id}/kill |
| Logic | server/services/lifecycle.py | line 214-279 |

### F009 Search Transcript

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/runs.py | line 238: GET /runs/{id}/search |
| Implementation | server/services/runs_index.py | search_run() |

### F010 Browse Presets

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/workbench.py | line 94: GET /workbench/presets |
| Presets Definition | server/api/workbench.py | line 65-91: PRESETS dict |

### F011-F014 Experiment Flows

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| Start Single | server/api/experiments.py | line 357: POST /experiments/{id}/runs/start |
| Start All | server/api/experiments.py | line 444: POST /experiments/{id}/runs/start-all |
| Compare Results | server/api/experiments.py | line 542: GET /experiments/{id}/compare-results |
| Compare Artifacts | server/api/experiments.py | line 460: GET /experiments/{id}/compare |

### F015-F022 Pipeline Configuration

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| Steps List | server/api/pipelines.py | line 114: GET /pipelines/steps |
| Templates | server/api/pipelines.py | line 144: GET /pipelines/templates |
| Validation | server/api/pipelines.py | line 174: POST /pipelines/validate |
| Resolution | server/api/pipelines.py | line 220: POST /pipelines/resolve |
| User Templates | server/api/pipelines.py | line 347-454 |
| Preprocessing | server/api/pipelines.py | line 130: GET /pipelines/preprocessing |

### F023-F030 Data/Results

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| List Runs | server/api/runs.py | line 37: GET /runs |
| Run Details | server/api/runs.py | line 208: GET /runs/{id} |
| Transcript | server/api/runs.py | line 219: GET /runs/{id}/transcript |
| Eval | server/api/results.py | line 18: GET /runs/{id}/eval |
| Findings | server/api/results.py | line 101: GET /findings |

### F031-F034 Candidates/Use Cases

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| Use Cases | server/api/candidates.py | line 235: GET /use-cases |
| Candidates | server/api/candidates.py | line 250: GET /use-cases/{id}/candidates |

### F035 Stream Audio

| Evidence Type | File Path | Symbol/Location |
|--------------|-----------|-----------------|
| API Entry | server/api/runs.py | line 250: GET /runs/{id}/audio |
| Processing | harness/media_ingest.py | ingest_media() |

---

## Evidence by File

### server/main.py
- Line 1-102: FastAPI app setup, router registration

### server/api/workbench.py
- Line 1-403: Workbench API endpoints
- Line 65-91: PRESETS definition
- Line 94-108: GET /workbench/presets
- Line 266-402: POST /workbench/runs

### server/api/experiments.py
- Line 1-580: Experiment API endpoints
- Line 128-344: POST /experiments
- Line 357-441: POST /experiments/{id}/runs/start

### server/api/runs.py
- Line 1-523: Runs API endpoints
- Line 20-34: GET /runs/{id}/results
- Line 55-61: GET /runs/compare
- Line 250-267: GET /runs/{id}/audio
- Line 388-439: GET /runs/{id}/bundle.zip

### server/api/pipelines.py
- Line 1-455: Pipeline API
- Line 114-127: GET /pipelines/steps
- Line 347-454: User templates CRUD

### server/api/lifecycle.py
- Line 1-44: Lifecycle endpoints
- Line 13: POST /runs/{id}/kill
- Line 28: POST /runs/{id}/retry

### server/api/ws_runs.py
- Line 1-191: WebSocket endpoint
- Line 52-163: WebSocket /api/runs/{run_id}/ws

### server/api/results.py
- Line 1-176: Results API
- Line 18-35: GET /runs/{id}/eval

### server/api/candidates.py
- Line 1-267: Candidates API
- Line 235-266: Use cases and candidates endpoints

### server/services/lifecycle.py
- Line 1-421: Worker lifecycle management
- Line 29-35: try_acquire_worker()
- Line 45-180: launch_run_worker()
- Line 214-279: kill_run()
- Line 282-420: retry_run()

### harness/session.py
- Line 1-1200: SessionRunner class
- Line 155-200: __init__
- Line 350-500: Step execution
- StepProgress dataclass for progress

### harness/errors.py
- Line 1-171: Error definitions
- Line 18-40: Error codes
- Line 81-135: classify_error()

### harness/pipeline_config.py
- Line 1-450: Pipeline configuration
- STEP_REGISTRY, PREPROCESSING_REGISTRY, PIPELINE_TEMPLATES

### client/src/lib/api.ts
- Line 1-421: API client
- All API method definitions

### client/src/pages/WorkbenchPage.tsx
- File upload UI and handling

### client/src/pages/ComparePage.tsx
- Experiment creation UI

---

## Error Codes

| Code | Meaning | File |
|------|---------|------|
| RUNNER_BUSY | Max concurrent runs reached | workbench.py:293 |
| INVALID_CANDIDATE | Candidate not found | experiments.py:154 |
| INVALID_PIPELINE | Pipeline config invalid | experiments.py:258 |
| PREVIEW_TOO_Large | File too large for preview | runs.py:375 |
| E_MODEL_NOT_FOUND | Model file missing | errors.py |
| E_MODEL_OOM | Out of memory | errors.py:100 |
| E_AUDIO_CORRUPT | Audio file corrupted | errors.py:109 |
| E_NETWORK_ERROR | Network failure | errors.py:128 |

---

## Environment Variables

| Variable | Purpose | File |
|----------|---------|------|
| MODEL_LAB_RUNS_ROOT | Runs directory | lifecycle.py:26 |
| MODEL_LAB_INPUTS_ROOT | Inputs directory | workbench.py:30 |
| MODEL_LAB_WORKBENCH_MAX_UPLOAD_BYTES | Upload limit | workbench.py:296 |
| MODEL_LAB_DATA_ROOT | Data directory | pipelines.py:45 |
| MODEL_LAB_ASR_PROVIDER | ASR provider | streaming_asr/providers.py |
| MODEL_LAB_WHISPER_MODEL | Whisper model | streaming_asr/provider_faster_whisper.py |
