import json
import os
import time
from pathlib import Path

import requests

BASE_URL = "http://localhost:8000/api"
RUNS_DIR = Path("runs/sessions/ghost_test_session/ghost_run_01")
MANIFEST_PATH = RUNS_DIR / "manifest.json"


def log(msg):
    print(f"[TEST] {msg}")


def setup_ghost_run():
    log(f"Creating dummy run at {RUNS_DIR}")
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "run_id": "ghost_run_01",
        "status": "COMPLETED",
        "started_at": "2026-01-01T00:00:00Z",
        "input_path": "inputs/test.wav",
        "steps": {"ingest": {}},
    }
    MANIFEST_PATH.write_text(json.dumps(manifest))


def verify_healing():
    try:
        # 1. Setup
        setup_ghost_run()

        # 2. Force Index Refresh via API
        log("Refreshing index via API...")
        resp = requests.get(f"{BASE_URL}/runs?refresh=true")
        resp.raise_for_status()
        runs = resp.json()

        found = any(r["run_id"] == "ghost_run_01" for r in runs)
        if not found:
            log("FAIL: Created run not found in index after refresh.")
            return False
        log("Run 'ghost_run_01' successfully indexed.")

        # 3. DELETE FROM DISK
        log("Simulating deletion (rm -rf)...")
        os.remove(MANIFEST_PATH)
        os.rmdir(RUNS_DIR)
        os.rmdir(RUNS_DIR.parent)  # Clean session dir too if empty

        # 4. Request Status (Should trigger healing)
        log("Requesting status for deleted run...")
        start_time = time.time()
        resp = requests.get(f"{BASE_URL}/runs/ghost_run_01/status")
        duration = time.time() - start_time

        log(f"Status Request returned: {resp.status_code} in {duration:.2f}s")

        if resp.status_code == 404:
            log("SUCCESS: Server returned 404 Not Found (Correctly handled).")
        elif resp.status_code == 500:
            log("FAIL: Server returned 500 Internal Error (Crash).")
            return False
        else:
            log(f"FAIL: Unexpected status {resp.status_code}")
            return False

        # 5. Verify Index is cleaned
        log("Verifying index is clean...")
        resp = requests.get(f"{BASE_URL}/runs")
        runs = resp.json()
        found = any(r["run_id"] == "ghost_run_01" for r in runs)
        if found:
            log("FAIL: Run still in index cache after healing trigger!")
            return False

        log("SUCCESS: Index was refreshed and run is gone.")
        return True

    except Exception as e:
        log(f"EXCEPTION: {e}")
        return False
    finally:
        # Cleanup if left over
        if RUNS_DIR.exists():
            import shutil

            shutil.rmtree(RUNS_DIR.parent)


if __name__ == "__main__":
    if verify_healing():
        print("\n\n*** TEST PASSED: Index Healing Works ***")
        exit(0)
    else:
        print("\n\n*** TEST FAILED ***")
        exit(1)
