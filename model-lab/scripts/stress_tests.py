import concurrent.futures
import os
import time

import requests

BASE_URL = "http://localhost:8000"
FILE_PATH = "inputs/abuse_large_file.wav"


def start_workbench_run(i, filename):
    """Starts a workbench run which handles upload + run."""
    print(f"[Run {i}] Starting run for {filename}...")
    start_time = time.time()
    try:
        with open(FILE_PATH, "rb") as f:
            files = {"file": (f"{filename}_{i}.wav", f, "audio/wav")}
            data = {
                "use_case_id": "stress_test",
                "steps_preset": "ingest",  # Use ingest for faster/lighter test
            }
            response = requests.post(f"{BASE_URL}/api/workbench/runs", files=files, data=data)
            elapsed = time.time() - start_time
            if response.status_code == 200:
                print(f"[Run {i}] Success in {elapsed:.2f}s: {response.json().get('run_id')}")
                return (True, i, response.json())
            elif response.status_code == 409:
                print(
                    f"[Run {i}] Busy (Expected if full) in {elapsed:.2f}s: {response.status_code}"
                )
                return (True, i, "BUSY")
            else:
                print(
                    f"[Run {i}] Failed in {elapsed:.2f}s: {response.status_code} - {response.text}"
                )
                return (False, i, response.text)
    except Exception as e:
        print(f"[Run {i}] Exception: {e}")
        return (False, i, str(e))


def test_mass_uploads_and_runs():
    print("\n--- TEST: Mass Upload & Run (5x Parallel) ---")
    # This covers both "Max Size File Upload" (implicit in run creation) and "Back-to-Back Runs"
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(start_workbench_run, i, "abuse_test") for i in range(1, 6)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
    return results


if __name__ == "__main__":
    if not os.path.exists(FILE_PATH):
        print(f"Error: {FILE_PATH} does not exist. Run generate_large_wav.py first.")
        exit(1)

    print(f"Targeting {BASE_URL}")

    results = test_mass_uploads_and_runs()

    print("\n--- Summary ---")
    print(f"Runs/Uploads: {sum(1 for r in results if r[0])}/{len(results)} passed/busy")
