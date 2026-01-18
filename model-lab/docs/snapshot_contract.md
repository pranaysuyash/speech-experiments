# Snapshot Correctness Contract

This document defines the guarantees for **Temporal Integrity** in Model Lab. It ensures that experiments remaining interpretable and comparable indefinitely, regardless of how the underlying `candidates` or `pipelines` evolve.

## 1. Experiment Immutability

**Guarantee**: An Experiment Record is a self-contained snapshot, not a set of references.

### Creation Time Snapshot
When `create_experiment` is called, the system MUST freeze the following metadata into `experiment_request.json`:

*   **Candidate Identity**: `candidate_ref` (the original ID) and `label`.
*   **Execution Strategy**: `steps_preset` and `params`.
*   **Slot Mapping**: The assignment of Slot A / Slot B.

**Violation**: Storing only `candidate_id` and resolving `label` or `preset` via `get_candidate()` at read time.

### Implementation Status
*   ✅ `server/api/experiments.py`: `create_experiment` constructs a `candidate_snapshot` dictionary and writes it to disk.
*   ✅ `server/api/experiments.py`: `_load_experiment` serves this snapshot directly.

## 2. Run Isolation

**Guarantee**: A Run View (`RunDetailPage`) must render exclusively from the Run's artifacts (Manifest, Transcript). It MUST NOT attempt to fetch "current" candidate metadata.

*   **Source of Truth**: `runs/<id>/manifest.json`
*   **Allowed Live Data**: `status` (for polling completion).

**Violation**: `RunDetail.tsx` calling `api.getCandidate(candidateId)` to display a header.

### Implementation Status
*   ✅ `RunDetail.tsx`: Only fetches `getRunStatus`, `getTranscript`, `getMeetingPack`. No candidate resolution.

## 3. Compare Reproducibility

**Guarantee**: Comparisons are derived solely from the Experiment Snapshot and the two immutable Run Artifacts.

*   **Labels**: Must come from the Experiment Snapshot (Slot A/B labels).
*   **Mapping**: Must use the Slot assignment frozen at creation.

**Violation**: If we rename Candidate X to "Old Model", historical experiments showing Candidate X should still show "Original Name" (if that was the label at the time) or "Candidate X" as frozen, unless we explicitly migrate snapshots. *Current implementation preserves the label at time of creation.*

### Implementation Status
*   ✅ `ExperimentPage.tsx`: Resolves headers/labels from `experiment.candidates` (the snapshot), not a live API.

## 4. Updates & Migrations

*   **Immutable**: `experiment_request.json` should generally never change.
*   **Mutable**: `experiment_state.json` (runs status) and `manifest.json` (run artifacts) evolve until terminal.
*   **Drift Policy**: If a `steps_preset` definition changes in code, old runs using that preset name are **not** re-run or invalidated. They accurately represent what *was* run. Ideally, one should allow param overrides in the snapshot if exact reproducibility of the *steps* logic is required (currently `params` are snapshotted).
