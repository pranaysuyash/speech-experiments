# API Error Taxonomy & Contract

This document defines the standard error response format for the Model Lab API.

## 1. Standard Error Schema

All API errors (4xx, 5xx) and Logical Failures (e.g. STALE status) MUST adhere to this shape where possible.

```typescript
interface ErrorResponse {
  error_code: string;    // SCREAMING_SNAKE_CASE, stable for programmatic handling
  error_message: string; // Human-readable, actionable message for the USER
  trace_id?: string;     // Optional, for debugging
  details?: any;         // Optional structured context
}
```

## 2. Fast API & Compatibility

*   **Custom JSONResponse**: MUST use keys `error_code` and `error_message`.
*   **HTTPException**: The `detail` field is legacy. Frontend MUST normalize `detail` -> `error_message` if the structured fields are missing.

## 3. Standard Error Codes

### Experiment Creation
| Code | HTTP | Description |
| :--- | :--- | :--- |
| `INVALID_CANDIDATE` | 400 | Candidate ID does not exist. |
| `CANDIDATE_USE_CASE_MISMATCH` | 400 | Candidate belongs to a different Use Case. |
| `INVALID_CANDIDATE_COUNT` | 400 | Provided ID count violates mode (must be 1 or 2). |
| `FILE_TOO_LARGE` | 413 | Upload exceeds limit. |

### Execution
| Code | HTTP | Description |
| :--- | :--- | :--- |
| `RUN_FAILED` | 200/500 | Run process exited with non-zero code. |
| `STALE_RUN` | 200 | Run is technically "RUNNING" but heartbeats stopped. |
| `START_FAILED` | 500 | Failed to spawn the runner process. |

### Artifacts (RunDetail)
| Code | HTTP | Description |
| :--- | :--- | :--- |
| `ARTIFACT_NOT_FOUND` | 404 | Specific artifact missing from bundle. |
| `PREVIEW_TOO_LARGE` | 413 | Artifact exceeds preview byte cap. |

## 4. UX Guidelines

1.  **Workbench**: Display `error_message` prominently in Red. If `error_code` is available, show it in monospace/small font for support.
2.  **Run Detail**:
    *   `FAILED` / `STALE`: Show the `error_message`.
    *   Legacy `detail`: Normalize and show.
3.  **No Retries in Results**: Results/Run views are Read-Only. Errors are terminal. Fixes happen in Workbench (New Experiment). 
