# Architecture Freeze: Model Lab Core Invariants

> **STATUS**: FROZEN
> **DATE**: 2026-01-18
> **SCOPE**: Execution Pipeline, Experiment Lifecycle, UX Contracts

This document defines the **immutable architectural pillars** of the Model Lab system. These invariants guard correctness and trust. **Do not modify them** without a formal Request for Comments (RFC) and migration plan.

---

## 1. Temporal Integrity (The Snapshot Law)

**Invariant**: An experiment record is a frozen snapshot of the world at creation time. It is **never** a set of references to live configuration.

*   **Identities**: `candidate_id`, `label`, `steps_preset`, and `params` are copied by value into `experiment_request.json`.
*   **Slot Semantics**: Slots (A/B) are assigned positionally at creation and never re-mapped.
*   **Drift Policy**: If valid candidates are deleted or modified in `candidates.py` or the database, historical experiments **MUST** continue to display the original labels and params from the snapshot.

**Forbidden**:
*   Using `api.getCandidate(id)` to render Experiment or Run headers.
*   Re-running an "old" experiment with "new" code using the old ID. (Must create new experiment).

---

## 2. Execution Boundaries (The Workbench Law)

**Invariant**: System state mutation is restricted to the **Workbench** surface.

*   **Initiator**: Only `WorkbenchPage` (and its API equivalents) can spawn processes.
*   **Read-Only**: `ExperimentPage`, `RunDetailPage`, and `Results` views are strictly **Read-Only**.
*   **No Auto-Magic**: Read views never trigger "auto-retry", "auto-start", or "queue repair".

**Forbidden**:
*   "Retry" buttons on the Results page (Must redirect to Workbench with pre-filled state).
*   RunPage strictly fetching artifacts; never triggering generation.

---

## 3. UX Honesty (The Status Law)

**Invariant**: The UI reflects the Backend State Machine 1:1.

*   **Status-First**: Call `getRunStatus` before rendering.
*   **No Optimism**: `QUEUED` runs show as Queued. `FAILED` runs show as Failed.
*   **Artifact Gating**: Transcripts and Downloads are only accessible when `status === 'COMPLETED'`.

**Forbidden**:
*   Rendering an empty transcript as "Assessment Complete".
*   Hiding backend errors behind generic "Something went wrong" toasts (Use `error_message`).

---

## 4. Error Semantics (The Taxonomy Law)

**Invariant**: Errors are structured, stable, and humane.

*   **Schema**: `{ "error_code": "SCREAMING_SNAKE", "error_message": "Human readable" }`.
*   **Truth**: `error_message` is the UX source of truth.
*   **Codes**: `error_code` is for programmatic handling only.

**Forbidden**:
*   Returning raw Python tracebacks in `detail`.
*   Frontend inventing error copy because the backend sent `500 Internal Server Error`.

---

## 5. File System Artifacts

*   `model_id` / `run_id` are the root keys.
*   `manifest.json` is the ground truth for a Run.
*   `experiment_request.json` is the ground truth for an Experiment.

---

**Violating any of the above constitutes a regression in system correctness.**
