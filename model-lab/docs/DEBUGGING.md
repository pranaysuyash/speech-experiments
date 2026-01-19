# Debugging Guide

## Frontend Debug Overlay
In `development` mode (`npm run dev`), the *Run Detail* page includes a collapsible "Debug (dev only)" section at the bottom.

### Fields
| Field | Purpose |
|-------|---------|
| `run_id` | Unique identifier for the run. |
| `status` | Raw status from API (`RUNNING`, `FAILED`, etc.). |
| `updated_at` | Last heartbeat from backend. Critical for determining stalls. |
| `steps_completed` | List of raw step keys completed. |
| `derived.secondsSinceProgress` | Time elapsed since `updated_at`. If > 90s, run is likely STALLED. |
| `derived.isStalledState` | Boolean flag driving the "STALLED" UI. |

### Usage
1. **Diagnosing Stuck Runs**: Check `derived.secondsSinceProgress`. If it's increasing but `status` is still `RUNNING`, the backend worker might be hung.
2. **Verifying Step Logic**: specific steps (like `ingest`) might be completed but not showing in the main pipeline UI if mapped incorrectly. Check `steps_completed` raw array.

## Server Logs
Server logs are standard stdout/stderr from `python server/main.py`.
- **Search Key**: `run_id` (e.g., `grep "20260119_..." server.log`)
- Worker logs are typically independent per process.

## Common Issues
- **409 Conflict**: "Runner is busy" - System limits active runs (default: 3).
- **429 Too Many Requests**: Frontend polling too aggressively or browser test spam.
