# Flow Atlas - Model Lab Audio Processing Platform

**Generated:** 2026-02-13 14:16:24
**Scope:** Audio processing platform for ASR, TTS, diarization, and related audio tasks
**Lens Coverage:** L1-L10 attempted

---

## Flow Inventory

### Category: User-Facing Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F001 | Workbench Run | L1/S3 | Implemented | L1/S3 - API entrypoint /workbench/runs |
| F002 | Experiment Creation | L1/S3 | Implemented | L1/S3 - API entrypoint /experiments |
| F003 | Run Status Monitoring | L1/S3 | Implemented | L1/S3 - WebSocket /runs/{id}/ws |
| F004 | View Run Results | L1/S3 | Implemented | L1/S3 - API /runs/{id}/results |
| F005 | Download Artifacts | L1/S3 | Implemented | L1/S3 - API /runs/{id}/bundle.zip |
| F006 | Compare Runs | L1/S3 | Implemented | L1/S3 - API /runs/compare |
| F007 | Retry Failed Run | L1/S3 | Implemented | L1/S3 - API /runs/{id}/retry |
| F008 | Kill Running Run | L1/S3 | Implemented | L1/S3 - API /runs/{id}/kill |
| F009 | Search Transcript | L1/S3 | Implemented | L1/S3 - API /runs/{id}/search |
| F010 | Browse Presets | L1/S3 | Implemented | L1/S3 - API /workbench/presets |

### Category: Experiment Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F011 | Start Experiment Run | L1/S3 | Implemented | L1/S3 - API /experiments/{id}/runs/start |
| F012 | Start All Experiment Runs | L1/S3 | Implemented | L1/S3 - API /experiments/{id}/runs/start-all |
| F013 | Compare Experiment Results | L1/S3 | Implemented | L1/S3 - API /experiments/{id}/compare-results |
| F014 | Compare Experiment Artifacts | L1/S3 | Implemented | L1/S3 - API /experiments/{id}/compare |

### Category: Pipeline Configuration Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F015 | View Available Steps | L5/S3 | Implemented | L5/S3 - API /pipelines/steps |
| F016 | View Pipeline Templates | L5/S3 | Implemented | L5/S3 - API /pipelines/templates |
| F017 | Validate Pipeline Config | L5/S3 | Implemented | L5/S3 - API /pipelines/validate |
| F018 | Resolve Pipeline Dependencies | L5/S3 | Implemented | L5/S3 - API /pipelines/resolve |
| F019 | Create User Template | L5/S3 | Implemented | L5/S3 - API /pipelines/user-templates (POST) |
| F020 | List User Templates | L5/S3 | Implemented | L5/S3 - API /pipelines/user-templates (GET) |
| F021 | Delete User Template | L5/S3 | Implemented | L5/S3 - API /pipelines/user-templates/{name} (DELETE) |
| F022 | View Preprocessing Operators | L5/S3 | Implemented | L5/S3 - API /pipelines/preprocessing |

### Category: Data/Results Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F023 | List All Runs | L6/S3 | Implemented | L6/S3 - API /runs |
| F024 | Get Run Details | L6/S3 | Implemented | L6/S3 - API /runs/{id} |
| F025 | Get Run Transcript | L6/S3 | Implemented | L6/S3 - API /runs/{id}/transcript |
| F026 | Get Run Eval | L6/S3 | Implemented | L6/S3 - API /runs/{id}/eval |
| F027 | Get Results Summary | L6/S3 | Implemented | L6/S3 - API /results |
| F028 | Get Findings | L6/S3 | Implemented | L6/S3 - API /findings |
| F029 | Get Runs By Input Hash | L6/S3 | Implemented | L6/S3 - API /runs/by-input/{hash} |
| F030 | Refresh Runs Index | L6/S3 | Implemented | L6/S3 - API /runs/refresh |

### Category: Candidate/Use Case Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F031 | List Use Cases | L3/S3 | Implemented | L3/S3 - API /use-cases |
| F032 | Get Use Case | L3/S3 | Implemented | L3/S3 - API /use-cases/{id} |
| F033 | List Candidates for Use Case | L3/S3 | Implemented | L3/S3 - API /use-cases/{id}/candidates |
| F034 | Get Candidate | L3/S3 | Implemented | L3/S3 - API /candidates/{id} |

### Category: Audio Playback Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F035 | Stream Processed Audio | L1/S3 | Implemented | L1/S3 - API /runs/{id}/audio |

### Category: CLI/Operational Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F036 | Run Harness CLI | L5/S3 | Implemented | scripts/run_session.py |
| F037 | Onboard New Model | L5/S3 | Implemented | scripts/onboard_model.py |
| F038 | Deploy API Server | L7/S3 | Implemented | scripts/deploy_api.py |
| F039 | Benchmark Model | L5/S3 | Implemented | bench/runner.py |
| F040 | Audit Runs | L7/S3 | Implemented | scripts/audit_runs.py |
| F041 | Export Bundle | L6/S3 | Implemented | scripts/export_bundle.py |
| F042 | Promote Run | L7/S3 | Implemented | scripts/promote_run.py |
| F043 | Quarantine Run | L7/S3 | Implemented | scripts/quarantine_run.py |
| F044 | Recommend and Run Model | L5/S3 | Implemented | scripts/recommend_and_run.py |
| F045 | Regression Test | L9/S3 | Implemented | scripts/regression_test.py |

### Category: Agent/Workflow Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F046 | Ticket Management | L7/S1 | Implemented | docs/WORKLOG_TICKETS.md |
| F047 | Agent Entry Point | L7/S1 | Implemented | prompts/workflow/agent-entrypoint-v1.0.md |
| F048 | Audit Workflow | L7/S1 | Implemented | prompts/audit/comprehensive-audit-v1.0.md |
| F049 | Model Evaluation | L5/S1 | Implemented | prompts/model/model-evaluation-v1.0.md |
| F050 | Implementation Remediation | L9/S1 | Implemented | prompts/remediation/implementation-v1.0.md |

### Category: Test Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F051 | Backend Invariants Test | L9/S6 | Implemented | tests/integration/test_backend_invariants.py |
| F052 | Artifact Download Security Test | L10/S6 | Implemented | tests/api/test_artifact_download_security.py |
| F053 | Smoke Test Model | L5/S6 | Implemented | tests/integration/test_model_*_smoke.py |
| F054 | Experiment Integration Test | L1/S6 | Implemented | tests/integration/test_experiments_v1.py |
| F055 | Lifecycle API Test | L7/S6 | Implemented | tests/integration/test_lifecycle_api.py |

### Category: Model Registry Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F056 | Register Model Loader | L5/S3 | Implemented | harness/registry.py:42-89 |
| F057 | Load Model | L5/S3 | Implemented | harness/registry.py:300+ |
| F058 | List Models | L5/S3 | Implemented | harness/registry.py:list_models() |
| F059 | Get Model Metadata | L5/S3 | Implemented | harness/registry.py:get_model_metadata() |
| F060 | Update Model Status | L5/S3 | Implemented | harness/registry.py:update_model_status() |

### Category: Streaming/Real-Time Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F061 | WebSocket Run Events | L1/S3 | Implemented | server/api/ws_runs.py:52 |
| F062 | Streaming ASR | L5/S3 | Implemented | harness/streaming_asr/ |
| F063 | Real-Time Model Inference | L5/S3 | Implemented | harness/streaming.py |

### Category: Evaluation/Metrics Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F064 | Compute Metrics | L6/S3 | Implemented | harness/metrics_*.py |
| F065 | Write Eval Results | L6/S3 | Implemented | harness/evals.py |
| F066 | Load Eval Results | L6/S3 | Implemented | server/api/eval_loader.py |
| F067 | Generate Scorecard | L6/S3 | Implemented | harness/results.py |

### Category: Claims/Assertions Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F068 | Validate Model Claims | L3/S1 | Implemented | tests/claims/ |
| F069 | Record Claim | L3/S1 | Implemented | docs/CLAIMS.md |
| F070 | Generate Arsenal Report | L7/S1 | Implemented | scripts/generate_arsenal.py |

### Category: Data Pipeline Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F071 | Ingest Audio | L6/S3 | Implemented | harness/media_ingest.py |
| F072 | Preprocess Audio | L6/S3 | Implemented | harness/preprocess_ops.py |
| F073 | Normalize Audio | L6/S3 | Implemented | harness/normalize.py |
| F074 | Create Meeting Bundle | L6/S3 | Implemented | harness/meeting_pack.py |

### Category: Notebook/Analysis Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F075 | Run Jupyter Analysis | L6/S1 | Implemented | compare/00_scorecard.ipynb |
| F076 | Generate Coverage Report | L7/S1 | Implemented | scripts/coverage_report.py |
| F077 | Model Comparison Report | L6/S1 | Implemented | docs/MODEL_COMPARISON_SCORECARD_*.md |

### Category: Configuration/Environment Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F078 | Setup Environment | L8/S3 | Implemented | setup_environment.py |
| F079 | Load Config from YAML | L8/S3 | Implemented | config/pipelines/*.yaml |
| F080 | Environment Variable Config | L8/S3 | Implemented | Multiple os.environ uses |

### Category: Frontend/UI Flows

| Flow ID | Canonical Name | Category | Status | Discovered By |
|---------|----------------|----------|--------|---------------|
| F081 | Browse Runs List | L1/S3 | Implemented | client/src/pages/RunsList.tsx |
| F082 | View Run Detail | L1/S3 | Implemented | client/src/components/RunDetail.tsx |
| F083 | Navigate to Workbench | L1/S3 | Implemented | client/src/App.tsx:105 |
| F084 | Navigate to Experiments | L1/S3 | Implemented | client/src/App.tsx:107 |
| F085 | Navigate to Results | L1/S3 | Implemented | client/src/App.tsx:114 |
| F086 | Navigate to Findings | L1/S3 | Implemented | client/src/App.tsx:115 |
| F087 | File Upload | L1/S3 | Implemented | client/src/components/DragDropFileUpload.tsx |

---

## Flow Summary

**Total Flows Discovered:** 87

**By Category:**
- User-Facing Flows: 10
- Experiment Flows: 4
- Pipeline Configuration Flows: 8
- Data/Results Flows: 8
- Candidate/Use Case Flows: 4
- Audio Playback Flows: 1
- CLI/Operational Flows: 10
- Agent/Workflow Flows: 5
- Test Flows: 5
- Model Registry Flows: 5
- Streaming/Real-Time Flows: 3
- Evaluation/Metrics Flows: 4
- Claims/Assertions Flows: 3
- Data Pipeline Flows: 4
- Notebook/Analysis Flows: 3
- Configuration/Environment Flows: 3
- Frontend/UI Flows: 7

---

## Flow Summary

**Total Flows Discovered:** 87

**By Category:**
- User-Facing Flows: 10
- Experiment Flows: 4
- Pipeline Configuration Flows: 8
- Data/Results Flows: 8
- Candidate/Use Case Flows: 4
- Audio Playback Flows: 1
- CLI/Operational Flows: 10
- Agent/Workflow Flows: 5
- Test Flows: 5
- Model Registry Flows: 5
- Streaming/Real-Time Flows: 3
- Evaluation/Metrics Flows: 4
- Claims/Assertions Flows: 3
- Data Pipeline Flows: 4
- Notebook/Analysis Flows: 3
- Configuration/Environment Flows: 3
- Frontend/UI Flows: 7

**By Status:**
- Implemented: 87 (100%)
- Partial: 0
- Candidate: 0
- Missing Expected: 0

---

## Evidence of Absence (Negative Space)

### Expected but NOT FOUND:

| Expected Flow | Lens | Evidence Searched | Notes |
|---------------|------|-------------------|-------|
| Authentication/Login | L4 | No login, session, auth routes found | Public API, no auth |
| User Management | L4 | No user CRUD endpoints | Single-user local app |
| API Rate Limiting | L9 | No rate limit middleware | Not implemented |
| Billing/Usage Tracking | L3 | No billing endpoints | No monetization |
| Webhook Notifications | L9 | No webhook registration | Not implemented |
| Batch Upload | L1 | Single file upload only | Not implemented |
| Scheduled Runs | L5 | No scheduler/cron | Manual trigger only |

---

## Pipeline Steps (Runtime Processing)

| Step Name | Description | Produces |
|-----------|-------------|-----------|
| ingest | Audio normalization and preprocessing | processed_audio |
| asr | Speech-to-text transcription | transcript |
| diarization | Speaker identification | diarization |
| alignment | Merge ASR with speaker labels | aligned_transcript |
| chapters | Topic segmentation | chapters |
| summarize_by_speaker | Per-speaker summary (LLM) | summary |
| action_items_assignee | Extract action items (LLM) | action_items |
| bundle | Package as Meeting Pack | bundle |
| enhance | Audio enhancement | enhanced_audio |
| separate | Vocal/instrument separation | separated_audio |
| vad | Voice activity detection | vad_segments |

## Models Registered (26)

| Model Type | Models |
|------------|--------|
| ASR | whisper, faster_whisper, faster_distil_whisper, glm_asr_nano, nb_whisper, voxtral, moonshine |
| TTS | lfm2_5_audio, glm_tts, cosyvoice |
| Streaming ASR | kyutai_streaming, voxtral_realtime, nemotron_streaming |
| Diarization | pyannote_diarization, heuristic_diarization |
| VAD | silero_vad |
| Enhancement | deepfilternet, rnnoise |
| Separation | demucs |
| Music | basic_pitch |
| Embeddings | clap |

---

## Key Invariants

1. **Single Worker:** Maximum 3 concurrent runs (configurable via `_MAX_CONCURRENT_RUNS`)
2. **Input Hash Determinism:** Run ID derived from input file SHA256
3. **Provenance Tracking:** SHA256 hash computed for all inputs and experiments
4. **Artifact Allowlisting:** Only manifest-listed artifacts can be served
5. **Path Traversal Prevention:** All file access via safe_file_path() validation
6. **Terminal State Immutability:** Completed/failed runs cannot be re-run without explicit retry

---

## Notes

- This is an audio processing platform (ASR, TTS, diarization, etc.)
- No user authentication - single user local application
- No cloud/monetization features
- Supports both real-time streaming and batch processing
- Pipeline is configurable with custom steps and preprocessing
- Contains comprehensive agent/CLI workflows for model development
- Includes test infrastructure and claims validation system
- Supports 26+ model types across ASR, TTS, enhancement, separation
