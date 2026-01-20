# Changelog

## 2026-01-20: Stability Invariants Release

**Commit:** `bf3dc5e` (merged to master)

### What Changed

- **Disk-first merge in `_save_manifest`**: When the manifest on disk has a terminal status (CANCELLED, FAILED, STALE), the runner adopts the disk state and preserves all disk fields before writing. This prevents external kills from being overwritten by runner saves.

- **Step cancellation respect**: Step completion logic now checks disk status and downgrades step status to CANCELLED if the run was killed mid-step.

- **Kill endpoint idempotence**: `POST /api/runs/{id}/kill` returns specific outcomes (`killed`, `already_dead`, `forced_cancel`, `already_terminal`) and always returns 200 for stopped runs.

- **Config transparency**: `resolved_config` (with `reason`) and `requested_config` are persisted to manifest and promoted to step entries.

- **Client build fix**: Removed unused state, added `runId` guard, fixed `URLSearchParams` construction in `RunDetail.tsx`.

### Invariants That Now Hold

| Invariant | Guardrail |
|-----------|-----------|
| Terminal status monotonic | `session.py` (disk-first merge logic) |
| Arbitrary kill metadata preserved | `test_status_regression_prevention_strict` |
| Step failure locks run to FAILED | `test_failure_propagation` |
| Kill is idempotent | `test_kill_run_idempotence` |

### What Remains Untested

- **Retry semantics**: No integration test yet (needs to be added).
- **Path traversal**: No explicit security test for artifact downloads.
- **Symlink containment**: No hardening for symlink escapes.
