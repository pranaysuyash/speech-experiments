import json
from datetime import UTC, datetime
from pathlib import Path

RUNS_ROOT = Path("runs")


def audit_run_manifest(manifest_path):
    issues = []
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        run_id = data.get("run_id", "unknown")
        status = data.get("status")
        created_at = data.get("created_at") or data.get("started_at")  # Fallback
        updated_at = data.get("updated_at")
        steps = data.get("steps", {})

        # 1. Terminal State Check
        # If pipeline is done, status must be COMPLETED or FAILED
        # We need a way to know if pipeline is intended to be done.
        # Simple heuristic: If last step is 'completed', status should be COMPLETED?
        # Better: Check for invalid states like 'RUNNING' forever.

        # 2. RUNNING vs Heartbeat
        if status == "RUNNING":
            if not updated_at:
                issues.append("RUNNING but no updated_at")
            else:
                try:
                    # Parse timestamp, assuming ISO format
                    # Most formatted as "YYYY-MM-DDTHH:MM:SSZ" or similar
                    last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                    now = datetime.now(UTC)
                    elapsed = (now - last_update).total_seconds()

                    # 5 minutes without update is suspicious for RUNNING
                    if elapsed > 300:
                        issues.append(
                            f"RUNNING but no heartbeat for {int(elapsed)}s (Stalled candidate)"
                        )
                except Exception:
                    issues.append(f"Invalid updated_at format: {updated_at}")

        # 3. STALLED is non-terminal
        # STALLED is a derived state in API, but if written to manifest, it shouldn't stuck there?
        # Actually API derives STALLED, manifest stays RUNNING.
        # If manifest says STALLED explicitly, that might be legacy or wrong?
        if status == "STALLED":
            issues.append("Manifest has explicit STALLED status (should be derived from RUNNING)")

        # 4. Failure Step Consistency
        failure_step = data.get("failure_step")
        if status == "FAILED" and not failure_step:
            # Not strict violation provided failure step logic is new, but verifying.
            # Actually user asked: "Capture failure_step deterministically"
            issues.append("FAILED but failure_step is missing or null")

        if status != "FAILED" and failure_step:
            issues.append(f"Status {status} but has failure_step: {failure_step}")

        return run_id, status, issues

    except Exception as e:
        return "error", "ERROR", [f"Failed to parse {manifest_path}: {e}"]


def main():
    print("Auditing runs...")
    count = 0
    violations = 0

    # Walk through all manifest.json files
    for manifest_path in RUNS_ROOT.rglob("manifest.json"):
        count += 1
        run_id, status, issues = audit_run_manifest(manifest_path)

        if issues:
            violations += 1
            print(f"\n[Run: {run_id}] Status: {status}")
            for issue in issues:
                print(f"  - {issue}")

    print(f"\nAudit Complete. Scanned {count} runs. Found {violations} with issues.")


if __name__ == "__main__":
    main()
