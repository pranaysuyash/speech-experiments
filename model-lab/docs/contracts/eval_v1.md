# Eval V1 Contract

## Canonical Location

- **Primary**: `<run_dir>/eval.json`
- **Fallback (deprecated)**: `<run_dir>/bundle/eval.json` - read-only legacy support

## Schema

```json
{
  "schema_version": "1",
  "run_id": "string",
  "use_case_id": "string | null",
  "model_id": "string | null",
  "params": {},
  "metrics": {},
  "checks": [],
  "findings": [],
  "generated_at": "ISO8601 timestamp"
}
```

## Fixed Check Names (10)

| # | Name | What It Checks | Severity |
|---|------|----------------|----------|
| 1 | `bundle_manifest_present` | bundle_manifest.json exists | info |
| 2 | `bundle_manifest_parseable` | Valid JSON | warn (if exists but invalid) |
| 3 | `transcript_present` | bundle/transcript.txt exists | info |
| 4 | `summary_present` | bundle/summary.md exists | info |
| 5 | `action_items_present` | bundle/action_items.csv exists | info |
| 6 | `decisions_present` | bundle/decisions.md exists | info |
| 7 | `asr_output_present` | artifacts/asr.json exists | info |
| 8 | `diarization_present` | artifacts/diarization.json exists | info |
| 9 | `alignment_present` | artifacts/alignment.json exists | info |
| 10 | `run_terminal_status_ok` | status == COMPLETED | fail (if FAILED/STALE) |

## Check Structure

```json
{
  "name": "string (from fixed list)",
  "passed": true | false,
  "severity": "info" | "warn" | "fail",
  "message": "string",
  "evidence_paths": ["relative/path/to/file"]
}
```

**Evidence paths**:
- Always relative to run_dir
- Only included when file exists
- Empty array if file doesn't exist

## Finding Structure

```json
{
  "finding_id": "category:identifier",
  "severity": "low" | "medium" | "high",
  "category": "system" | "asr" | "diarization" | "alignment",
  "title": "Human-readable title",
  "details": "Extended description",
  "evidence_paths": ["relative/path"]
}
```

Findings generated when:
- `run_terminal_status_ok` fails (FAILED/STALE)
- `bundle_manifest_parseable` fails

## Feature Flag

**Env var**: `MODEL_LAB_EVAL_MODE`

| Value | Behavior |
|-------|----------|
| `enriched` (default) | 10 checks + findings |
| `identity` | Empty checks/findings, schema header only |

## API Endpoints

| Endpoint | Behavior |
|----------|----------|
| `GET /api/runs/{id}/eval` | Returns eval.json, 404 if missing |
| `GET /api/results` | Includes `eval_available`, `checks_total`, `checks_passed` |
| `GET /api/findings` | Aggregates findings across all runs |

## Loader Preference

```python
# load_eval prefers canonical over legacy
for p in [run_root / "eval.json", run_root / "bundle" / "eval.json"]:
    # try each, return first valid
```

## Metrics

Currently `{}` - no computed metrics in V1.
