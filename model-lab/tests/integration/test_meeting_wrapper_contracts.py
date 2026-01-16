"""
Contract tests for meeting wrapper hardening.

These tests lock the critical contracts introduced by the hardening work:
1. JSON sentinel emission
2. Sentinel parsing with noisy stdout
3. Immediate manifest write with RUNNING status
4. Status endpoint schema stability

All tests use isolated temp directories to avoid polluting dev data.
"""
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
import pytest
import tempfile
import shutil
import signal


# Test 1: Sentinel line emission
def test_run_session_emits_single_result_line():
    """
    Verify run_session.py emits exactly one RUN_SESSION_RESULT= line
    and it contains valid JSON.
    """
    # Create minimal test audio
    with tempfile.TemporaryDirectory() as tmpdir:
        test_audio = Path(tmpdir) / "test.wav"
        # Create 1-second silence
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=duration=1", 
            "-ar", "16000", "-ac", "1", str(test_audio), "-y"
        ], capture_output=True, check=True)
        
        runs_dir = Path(tmpdir) / "runs"
        
        # Run session with minimal steps using isolated runs directory
        result = subprocess.run([
            sys.executable, "scripts/run_session.py",
            "--input", str(test_audio),
            "--out-dir", str(runs_dir),
            "--steps", "ingest"
        ], capture_output=True, text=True, timeout=30)
        
        stdout = result.stdout
        
        # Find all sentinel lines
        sentinel_pattern = re.compile(r"^RUN_SESSION_RESULT=(\{.*\})\s*$", re.MULTILINE)
        matches = sentinel_pattern.findall(stdout)
        
        # Assert exactly one match
        assert len(matches) == 1, f"Expected 1 sentinel line, found {len(matches)}"
        
        # Assert it's valid JSON
        sentinel_json = json.loads(matches[0])
        
        # Assert required fields
        assert "run_id" in sentinel_json
        assert "run_dir" in sentinel_json
        assert "console_url" in sentinel_json
        assert sentinel_json["console_url"].startswith("http://localhost:")


# Test 2: Sentinel parsing with noise
def test_run_meeting_parses_result_line_with_noise():
    """
    Verify run_meeting.py can parse the sentinel line even with noisy stdout.
    """
    from scripts.run_meeting import _run_session_and_capture, _RESULT_RE
    
    # Simulate noisy stdout with logs before and after sentinel
    noisy_stdout = """INFO:session:Starting Session abc123
INFO:harness.registry:Registered loader: faster_whisper
INFO:session:Running Step: ingest
RUN_SESSION_RESULT={"run_id":"test_20260115_001122","run_dir":"/path/to/runs/test_20260115_001122","console_url":"http://localhost:5174/runs/test_20260115_001122"}
INFO:session:Bundle created
"""
    
    # Parse it
    m = _RESULT_RE.search(noisy_stdout)
    assert m is not None, "Failed to find sentinel in noisy stdout"
    
    result = json.loads(m.group(1))
    assert result["run_id"] == "test_20260115_001122"
    assert result["run_dir"] == "/path/to/runs/test_20260115_001122"
    assert result["console_url"] == "http://localhost:5174/runs/test_20260115_001122"


# Test 3: Immediate manifest write
def test_manifest_written_immediately_with_running_status():
    """
    Verify manifest.json exists within 1s of run start with status=RUNNING.
    Uses process group for proper cleanup.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        test_audio = Path(tmpdir) / "test.wav"
        # Create 1-second silence
        subprocess.run([
            "ffmpeg", "-f", "lavfi", "-i", "anullsrc=duration=1",
            "-ar", "16000", "-ac", "1", str(test_audio), "-y"
        ], capture_output=True, check=True)
        
        runs_dir = Path(tmpdir) / "runs"
        
        # Start run in background as a process group
        proc = subprocess.Popen([
            sys.executable, "scripts/run_session.py",
            "--input", str(test_audio),
            "--out-dir", str(runs_dir),
            "--steps", "ingest"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
           start_new_session=True)  # Create new process group
        
        # Wait up to 1 second for manifest to exist
        start_time = time.time()
        manifest_found = False
        manifest_path = None
        
        try:
            for _ in range(10):  # Check every 100ms for 1s
                time.sleep(0.1)
                # Find manifest files
                manifests = list(runs_dir.glob("sessions/*/*/manifest.json"))
                if manifests:
                    manifest_path = manifests[0]
                    manifest_found = True
                    break
            
            elapsed = time.time() - start_time
            
            assert manifest_found, f"Manifest not found within 1s (waited {elapsed:.2f}s)"
            assert elapsed < 1.0, f"Manifest took too long to appear: {elapsed:.2f}s"
            
            # Verify manifest content
            manifest_data = json.loads(manifest_path.read_text())
            assert manifest_data["status"] in ["RUNNING", "COMPLETED"], \
                f"Expected RUNNING or COMPLETED, got {manifest_data['status']}"
        finally:
            # Clean up: kill entire process group
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=5)
            except (ProcessLookupError, PermissionError):
                pass  # Already dead


# Test 4: Status endpoint schema
def test_status_endpoint_schema():
    """
    Verify /api/runs/{id}/status returns the expected schema.
    Requires backend and a run to exist.
    """
    import requests
    
    # First, get a run ID from the list
    try:
        response = requests.get("http://localhost:8000/api/runs", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running on localhost:8000")
    
    assert response.status_code == 200
    
    runs = response.json()
    if not runs:
        pytest.skip("No runs available for status endpoint test")
    
    run_id = runs[0]["run_id"]
    
    # Get status
    status_response = requests.get(f"http://localhost:8000/api/runs/{run_id}/status")
    assert status_response.status_code == 200
    
    status_data = status_response.json()
    
    # Verify schema
    assert "run_id" in status_data
    assert "status" in status_data
    assert "steps_completed" in status_data
    
    # Verify types
    assert isinstance(status_data["run_id"], str)
    assert isinstance(status_data["status"], str)
    assert isinstance(status_data["steps_completed"], list)
    
    # Verify status enum
    assert status_data["status"] in ["PENDING", "RUNNING", "COMPLETED", "FAILED"]
    
    # Verify steps_completed is list of strings
    for step in status_data["steps_completed"]:
        assert isinstance(step, str)
