# Builder Error Taxonomy

## 1. Terminal Errors (Run Failed)

These errors result in a `FAILED` run status.

| Error Code | Meaning | Retry Strategy | Source |
|------------|---------|----------------|--------|
| `RuntimeError` | Subprocess or library processing failure (e.g. ffmpeg ingest). | No (Fix Input) | Backend Worker |
| `FileNotFoundError` | Missing input file or model file. | No (Fix Path) | Backend Worker |
| `ProcessError` | Generic processing exception. | Maybe (Transient?) | Backend Logic |
| `Unknown` | Uncaught exception. | Check Logs | Global Catch-All |

## 2. API Errors (Request Rejected)

These occur before a run starts.

| status | Message | Meaning | Retry Strategy |
|--------|---------|---------|----------------|
| `400` | Bad Request | Invalid metadata, missing file, or zero-byte input. | No (Fix Request) |
| `409` | Runner Busy | Concurrency limit (3) reached. | Yes (Backoff) |
| `413` | Payload Too Large | File exceeds limit (200MB). | No (Reduce Size) |
| `429` | Too Many Requests | Rate limit (client/browser). | Yes (Backoff) |
| `500` | Internal Server Error | Logic crash. | No (Report Bug) |

## 3. System States

| Status | Meaning | Action |
|--------|---------|--------|
| `STALLED` | Backend not reporting progress (>90s). | Check Logs / Restart Server |
| `STALE` | Run state unknown/sync issue. | Refresh / Check Persistence |

## 4. Diagnosis Flow
1. Check **API Status Code** (4xx/5xx).
2. If 200/Accepted, check **Run Status** in UI/Manifest.
3. If FAILED, check `error_code` + `error_message`.
4. If STALLED, assume Backend Worker crash/hang.
