# Product-Critical Backlog for Model Lab

## Executive Summary

Model Lab is an audio processing platform for ASR, TTS, diarization, and related tasks. It has a working backend API, a React frontend, and model testing infrastructure. However, there are significant gaps that prevent it from being a reliable, repeatable tool for production use.

**This is a demo unless we ship these:**

---

## A) "This is a demo unless we ship these"

### 1. Add Structured Logging Across All API Endpoints

**Why it blocks product reality:** Without structured logging, debugging production issues is guesswork. When a run fails at 2am, you can't trace what happened.

**Evidence:**
- `server/api/workbench.py`: 0 logger.info() calls
- `server/api/experiments.py`: 0 structured logs  
- `server/services/lifecycle.py`: Only 4 logger calls, mostly warnings
- 30+ bare `except Exception:` blocks that silently swallow errors

**Acceptance Criteria:**
- Every API endpoint logs: request start, key decisions, response (success/error)
- Every background job logs: start, progress, completion, failure
- Logs include correlation IDs (run_id, request_id)
- Logs use structured format (JSON) for machine parsing

**Minimal PR Plan:**
1. Add `logger = logging.getLogger(__name__)` to each API module
2. Add structured logging to: workbench.py, experiments.py, lifecycle.py
3. Include correlation IDs in log context
4. Verify with: `rg "logger\.(info|warning|error)" server/api/ | wc -l` shows >50 entries

**Verification Commands:**
```bash
# Check current logging coverage
rg "logger\.(info|warning|error)" server/api/ | wc -l
# Expected: >50, Current: ~10

# Check bare excepts
rg "except Exception:" server/api/ | wc -l  
# Expected: <5, Current: ~30
```

---

### 2. Implement Request ID / Correlation ID Propagation

**Why it blocks product reality:** Without correlation IDs, you can't trace a user complaint back to specific logs, making debugging impossible.

**Evidence:**
- No HTTP request ID middleware
- No correlation between API logs and run events
- WebSocket events in `ws_runs.py` don't carry request context

**Acceptance Criteria:**
- Every API request gets a unique request_id
- Request ID propagates to background jobs
- Logs include request_id for filtering
- Client receives request_id in responses

**Minimal PR Plan:**
1. Add FastAPI middleware for request ID generation
2. Pass request_id to background workers via manifest
3. Include request_id in all log statements
4. Return request_id in API response headers

---

### 3. Add Cost/Resource Tracking Per Run

**Why it blocks product reality:** Users have no visibility into how much a run cost (GPU minutes, API credits), making it impossible to manage budgets.

**Evidence:**
- `harness/timers.py` tracks wall time but not GPU time
- No API for cost estimation
- `bench/runner.py` has metrics but not exposed via API

**Acceptance Criteria:**
- Each run tracks: GPU time, CPU time, peak memory, API calls
- Results endpoint exposes cost metrics
- Optional: cost caps per run or per user

**Minimal PR Plan:**
1. Add resource tracking to SessionRunner
2. Expose via `/api/runs/{id}/results` 
3. Add optional cost cap in workbench.py

---

### 4. Implement Graceful Shutdown / Run Reconciliation

**Why it blocks product reality:** If the server crashes mid-run, there's no way to know which runs were interrupted, causing stale state.

**Evidence:**
- `lifecycle.py`: Worker PID tracked but no health check
- No mechanism to detect orphaned runs on startup
- No reconciliation between filesystem state and manifest

**Acceptance Criteria:**
- On startup, scan for runs in "RUNNING" state
- Verify worker PID is alive; if not, mark as STALE/FAILED
- Add health check that reports orphaned runs

**Minimal PR Plan:**
1. Add startup reconciliation in `RunsIndex._refresh_index()`
2. Check worker PIDs for RUNNING runs
3. Add `/api/admin/health` endpoint
4. Verify with: Start run, kill worker, restart server, check status

---

### 5. Add API Versioning / Stability Contract

**Why it blocks product reality:** Without API versioning, frontend upgrades break silently. Users can't depend on stable contracts.

**Evidence:**
- No API version headers
- No deprecation path
- `server/api/` has no version prefixes (e.g., /api/v1/)

**Acceptance Criteria:**
- All endpoints under `/api/v1/`
- Version header in responses (Accept-Version)
- Deprecation warnings before breaking changes

**Minimal PR Plan:**
1. Add version prefix to all routes
2. Add response headers for version
3. Document breaking change process

---

### 6. Implement Run Timeout / Max Duration

**Why it blocks product reality:** Runs can hang indefinitely, consuming resources with no progress. No way to auto-cancel.

**Evidence:**
- No timeout configuration
- `lifecycle.py` has no max runtime
- Step timeout defined in errors.py but not enforced

**Acceptance Criteria:**
- Configurable max run duration (env var)
- Automatic cancellation after timeout
- Clear error message: "Run timed out after X minutes"

**Minimal PR Plan:**
1. Add `MODEL_LAB_MAX_RUN_DURATION_SECONDS` env var
2. Add timeout check in worker loop
3. Update manifest status to TIMEOUT

---

### 7. Add Input Validation with Schema Contracts

**Why it blocks product reality:** Invalid inputs cause confusing errors deep in the stack. Users get 500 instead of clear "your file is too big."

**Evidence:**
- `workbench.py`: No input validation before processing
- `experiments.py`: Validation happens mid-function
- No Pydantic models for request validation

**Acceptance Criteria:**
- All endpoints use Pydantic request models
- Clear 422 responses for validation errors
- File type/size validation before upload

**Minimal PR Plan:**
1. Add Pydantic models to workbench.py, experiments.py
2. Add file type validation
3. Verify: Upload invalid file → 422 response

---

## B) "Quick wins that unlock compounding"

### 8. Add Health Check with Dependencies

**Why it blocks:** Can't verify system is ready.

**Evidence:**
- `/health` returns static `{"status": "ok"}`
- Doesn't check: runs directory writable, disk space, model cache

**Fix:** Enhance `/api/admin/health` to check dependencies

---

### 9. Document Required Environment Variables

**Why it blocks:** New users don't know what's required.

**Evidence:**
- 10+ env vars used but no single source of truth
- `onboard_model.py` has ENV_REQUIREMENTS dict (good pattern)

**Fix:** Create `.env.example` with all variables

---

### 10. Add Run Queue / Priority

**Why it blocks:** All runs have same priority; can't bump urgent jobs.

**Fix:** Add priority field to run request

---

### 11. Expose Prometheus Metrics

**Why it blocks:** No observability for production monitoring.

**Evidence:**
- `scripts/deploy_api.py:702` has `/metrics` endpoint but not wired in main app

**Fix:** Wire up metrics endpoint in main.py

---

### 12. Add CLI for Common Operations

**Why it blocks:** Users must use API for everything.

**Fix:** Add `python -m model_lab admin cleanup`, `python -m model_lab run status`

---

## C) "Stop doing"

### 13. Stop Using Bare `except Exception:` 

**Current:** 30+ bare excepts silently swallow errors

**Replace with:** Specific exception handling + logging

---

### 14. Stop Hardcoding Paths in Error Messages

**Current:** Error messages leak paths like `/Users/pranay/...`

**Replace with:** Relative paths or generic messages

---

### 15. Stop Mixing Responsibilities in `lifecycle.py`

**Current:** 500+ lines handling: worker spawning, PID tracking, retry, kill

**Replace with:** Separate into: worker_manager.py, retry_handler.py

---

## Evidence Commands

```bash
# Check logging coverage (goal: >50)
rg "logger\.(info|warning|error)" server/api/ | wc -l

# Check bare excepts (goal: <5)
rg "except Exception:" server/api/ | wc -l

# Check API versioning
rg "@router\.(get|post)" server/api/ | head -20

# Check timeout handling
rg "timeout|TIMEOUT" server/api/ harness/ | head -10

# Check health endpoint depth
rg "def health" server/
```
