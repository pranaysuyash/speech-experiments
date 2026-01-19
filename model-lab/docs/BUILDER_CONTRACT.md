# Builder Contract

## 1. Audience
**In Scope:**
- Internal developers debugging model pipelines.
- Alpha testers verifying pipeline stability on local hardware.
- Engineers integrating the API into larger workflows.

**Out of Scope:**
- End-users expecting a SaaS product (No payments, no SLAs).
- Public internet traffic (No specialized DDoS protection, no multi-tenant isolation).
- Production-critical dependence (API may break, storage is volatile).

## 2. Guarantees (Provided)
- **State Integrity**: A run will settle in a terminal state (`COMPLETED` or `FAILED`) or be marked `STALLED` if the worker dies.
- **Fail-Safe**: Invalid inputs will be rejected or failed safely; they will not crash the server.
- **Input Isolation**: Concurrent runs will not overwrite each other's inputs.
- **Observability**: Every run has a unique ID, status, and persisted logs.

## 3. Guarantees (NOT Provided)
- **Performance**: No latency or throughput guarantees.
- **Retention**: Data may be deleted at any time (local filesystem storage).
- **Security**: No authentication/authorization currently implemented (Single User Mode).
- **Uptime**: Server is self-hosted; no HA/failover.

## 4. Behavior Model
- **Correct Behavior**: System accepts valid audio, processes via configured pipeline, and produces JSON metadata + artifacts.
- **Known Limitations**:
    - Large files (>500MB) may timeout or exhaust memory depending on host specs.
    - Long filenames (FS limit) are sanitized.
    - No automatic cleanup of successful runs (Disk fills up).
