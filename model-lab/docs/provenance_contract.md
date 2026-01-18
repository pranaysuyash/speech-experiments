# Provenance Contract: Experiment Integrity

**Status**: PROPOSED
**Scope**: Server-side Experiment Storage

This document defines the mechanism for cryptographic verification of Experiment Snapshots.

## 1. Hashing Algorithm

*   **Hash Function**: `SHA-256`
*   **Input**: The byte sequence of `experiment_request.json` serialized with **Canonical Formatting**:
    *   Unsorted keys? NO. **Keys must be sorted.**
    *   Indentation? NO. **Compact (separators=(',', ':'))** to avoid whitespace drift.
    *   Encoding: `utf-8`

`hash = sha256(json.dumps(request_data, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()`

## 2. Storage

The hash is stored alongside the experiment, typically in `experiment_state.json` (which wraps the immutable request).

```json
{
  "experiment_id": "...",
  "provenance": {
    "hash": "a1b2c3...",
    "algorithm": "sha256",
    "timestamp": "2026-01-01T12:00:00Z"
  },
  "runs": [...]
}
```

## 3. Verification (Read-Time)

When `_load_experiment(id)` is called:

1.  Read `experiment_request.json` from disk.
2.  Compute canonical hash.
3.  Compare with `provenance.hash` in `experiment_state.json`.

## 4. Integrity States

The API response for `GET /experiments/:id` will include a `provenance_status` field:

| Status | Meaning | Action |
| :--- | :--- | :--- |
| `VERIFIED` | Hash matches snapshot. | Normal operation. |
| `UNVERIFIED` | Legacy experiment or missing hash. | Warn in logs (optional). |
| `CORRUPTED` | Hash mismatch. | **Log ERROR.** Return 200 OK (Do not crash). UI may show warning. |

**Rationale**: We never block read access to data, even if corrupted, to allow debugging. We flag it instead.
