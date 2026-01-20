
import os
import time
import requests
import json
import sys
from pathlib import Path

# Config
BASE_URL = "http://localhost:8000"
import wave
import struct

# Config
BASE_URL = "http://localhost:8000"
TEST_FILENAME = "lifecycle_test.wav"

def generate_silent_wav(filename, duration_sec=10):
    sample_rate = 44100
    n_frames = sample_rate * duration_sec
    with wave.open(filename, 'wb') as obj:
        obj.setnchannels(1) # mono
        obj.setsampwidth(2) # 2 bytes
        obj.setframerate(sample_rate)
        # Write silence
        data = struct.pack('<h', 0) * n_frames
        obj.writeframes(data)
    with open(filename, 'rb') as f:
        return f.read()

TEST_FILE_CONTENT = generate_silent_wav(TEST_FILENAME, duration_sec=30) # 30s to be safe

def info(msg):
    print(f"[INFO] {msg}")

def check(condition, msg):
    if not condition:
        print(f"[FAIL] {msg}")
        sys.exit(1)
    print(f"[PASS] {msg}")

def wait_for_status(run_id, target_statuses, timeout=30):
    start = time.time()
    while time.time() - start < timeout:
        r = requests.get(f"{BASE_URL}/api/runs/{run_id}/status")
        s = r.json()
        if s["status"] in target_statuses:
            return s
        time.sleep(0.5)
    return None

def main():
    # 1. Create Run
    info("Creating Run...")
    files = {'file': (TEST_FILENAME, TEST_FILE_CONTENT, 'audio/wav')}
    data = {'use_case_id': 'default', 'steps_preset': 'ingest'} # Use ingest for speed, or full?
    # Actually ingest is fast. We need a slow run to kill it?
    # If ingest finishes too fast, we can't kill it.
    # The 'full' preset will fail eventually on fake audio, but stay running long enough to kill?
    # Or 'ingest' might be too fast.
    # We can use 'full'. ASR will start and hang or run on mock if mock is active (it's not).
    # Real ASR on fake audio might fail fast or slow.
    
    r = requests.post(f"{BASE_URL}/api/workbench/runs", files=files, data={'use_case_id': 'default', 'steps_preset': 'full'})
    if r.status_code != 200:
        print(f"Failed to create run: {r.text}")
        sys.exit(1)
    
    run_id = r.json()["run_id"]
    info(f"Run Created: {run_id}")
    
    # 2. Wait for RUNNING
    status = wait_for_status(run_id, ["RUNNING", "QUEUED"])
    check(status is not None, "Run reached valid state")
    
    # Allow worker to start and write PID
    time.sleep(2)
    
    # 3. Kill Run
    info("Killing Run...")
    r = requests.post(f"{BASE_URL}/api/runs/{run_id}/kill")
    check(r.status_code == 200, f"Kill request successful: {r.text}")
    
    # 4. Verify CANCELLED
    status = wait_for_status(run_id, ["CANCELLED", "FAILED"])
    check(status is not None, "Run reached terminal state")
    check(status["status"] == "CANCELLED", f"Run status is CANCELLED (got {status['status']})")
    
    # 5. Retry Run
    info("Retrying Run...")
    r = requests.post(f"{BASE_URL}/api/runs/{run_id}/retry")
    check(r.status_code == 200, f"Retry request successful: {r.text}")
    
    # 6. Verify RUNNING again
    time.sleep(1) # Wait for status update
    status = wait_for_status(run_id, ["RUNNING"])
    check(status is not None and status["status"] == "RUNNING", f"Run status is RUNNING (got {status['status']})")
    
    info("Lifecycle Verification Complete!")

if __name__ == "__main__":
    main()
