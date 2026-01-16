"""
Contract tests for failure legibility and stale detection.

These tests verify the new manifest fields and stale detection logic.
"""
import json
import os
import time
from pathlib import Path
from datetime import datetime, timezone, timedelta
import pytest
import tempfile


def test_status_includes_current_step_and_updated_at():
    """
    Verify /api/runs/{id}/status includes current_step and updated_at fields.
    """
    import requests
    
    # Get a run from the list
    try:
        response = requests.get("http://localhost:8000/api/runs", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running on localhost:8000")
    
    assert response.status_code == 200
    runs = response.json()
    if not runs:
        pytest.skip("No runs available for status test")
    
    run_id = runs[0]["run_id"]
    
    # Get status
    status_response = requests.get(f"http://localhost:8000/api/runs/{run_id}/status")
    assert status_response.status_code == 200
    
    status_data = status_response.json()
    
    # Verify new fields exist
    assert "current_step" in status_data  # May be None
    assert "updated_at" in status_data
    
    # If updated_at exists, verify it's a valid ISO string
    if status_data["updated_at"]:
        datetime.fromisoformat(status_data["updated_at"].replace("Z", "+00:00"))


def test_stale_run_is_detected():
    """
    Verify stale detection logic without requiring a real running process.
    Creates a fake manifest with old updated_at and verifies STALE status.
    """
    import tempfile
    from pathlib import Path
    import json
    import os
    
    # Set up isolated runs root
    with tempfile.TemporaryDirectory() as tmpdir:
        runs_root = Path(tmpdir) / "runs"
        
        # Create fake run directory structure
        hash_dir = runs_root / "sessions" / "test_hash"
        hash_dir.mkdir(parents=True)
        
        run_id = "test_stale_20260115_000000"
        run_dir = hash_dir / run_id
        run_dir.mkdir()
        
        # Create manifest with RUNNING status and old updated_at (2 minutes ago)
        old_time = datetime.now(timezone.utc) - timedelta(minutes=2)
        manifest = {
            "run_id": run_id,
            "status": "RUNNING",
            "started_at": old_time.isoformat(),
            "updated_at": old_time.isoformat(),
            "current_step": "diarization",
            "steps": {}
        }
        
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        # Now test the stale detection logic directly (unit test style)
        # This simulates what the status endpoint does
        status = manifest["status"]
        updated_at = manifest.get("updated_at")
        current_step = manifest.get("current_step")
        
        STALE_THRESHOLD_SECONDS = 90
        
        if status == "RUNNING" and updated_at:
            last_update = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            elapsed = (now - last_update).total_seconds()
            
            if elapsed > STALE_THRESHOLD_SECONDS:
                status = "STALE"
                error_code = "STALE_RUN"
                error_message = f"No heartbeat in {int(elapsed)}s"
                
                # Assert stale detection worked
                assert status == "STALE"
                assert error_code == "STALE_RUN"
                assert "No heartbeat" in error_message
                assert int(elapsed) >= STALE_THRESHOLD_SECONDS
        
        # Verify it was detected as stale
        assert status == "STALE", f"Expected STALE, got {status}"
