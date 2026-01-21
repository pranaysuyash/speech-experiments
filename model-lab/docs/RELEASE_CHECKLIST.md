# Release Checklist

## Before Every Push
Pre-push hook runs `ci_mode_b_gate.sh` automatically.

## Before Tag/Release
```bash
./scripts/mode_b_verify.sh
```

## S4 Manual Ritual (Once Per Release)

**Procedure:**
1. Start a run, then kill only the worker process mid-flight
2. Reload the run detail in UI while API server is still alive
3. Record evidence below

### Evidence Template
```
Release: vX.Y.Z
Date: YYYY-MM-DD
Tester: [name]

Run ID: [paste]
Worker PID killed: [paste]
Timestamp of kill: [paste]

UI Observations:
- Status badge: [ ] STALE / [ ] RUNNING with stale indicator
- snapshot_source value: [paste from Debug Panel]
- Pipeline state preserved: [ ] Yes / [ ] No

Notes: [any anomalies]
```

---

## Evidence Log

### v0.1.0 (Template)
```
Release: v0.1.0
Date: 2026-01-21
Tester: [pending]

Run ID: 
Worker PID killed: 
Timestamp of kill: 

UI Observations:
- Status badge: [ ] STALE / [ ] RUNNING with stale indicator
- snapshot_source value: 
- Pipeline state preserved: [ ] Yes / [ ] No

Notes: 
```
