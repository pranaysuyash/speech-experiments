"""
Hermetic contract tests for workbench endpoint.

Uses FastAPI TestClient - no uvicorn required.
"""
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def test_workbench_runs_returns_409_when_busy():
    """Verify 409 RUNNER_BUSY when worker is active."""
    from fastapi.testclient import TestClient
    from server.main import app
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MODEL_LAB_RUNS_ROOT"] = tmpdir
        os.environ["MODEL_LAB_INPUTS_ROOT"] = tmpdir
        
        # Simulate busy state by patching try_acquire_worker in the workbench module
        # (where it's imported and used)
        with patch("server.api.workbench.try_acquire_worker", return_value=False):
            client = TestClient(app)
            
            # Create a minimal wav file
            wav_data = b"RIFF" + b"\x00" * 40  # Minimal fake wav
            
            response = client.post(
                "/api/workbench/runs",
                files={"file": ("test.wav", wav_data, "audio/wav")},
                data={"use_case_id": "test", "steps_preset": "ingest"}
            )
            
            assert response.status_code == 409
            assert response.json()["error_code"] == "RUNNER_BUSY"


def test_workbench_runs_success_returns_run_id():
    """Verify success path returns run_id and starts run."""
    from fastapi.testclient import TestClient
    from server.main import app
    from harness.session import SessionRunner
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MODEL_LAB_RUNS_ROOT"] = tmpdir
        os.environ["MODEL_LAB_INPUTS_ROOT"] = tmpdir
        
        session_dir = Path(tmpdir) / "sessions" / "test" / "test_run_123"
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Mock SessionRunner to avoid actually running
        mock_runner = MagicMock(spec=SessionRunner)
        mock_runner.run_id = "test_run_123"
        mock_runner.session_dir = session_dir
        mock_runner.manifest_path = session_dir / "manifest.json"
        mock_runner.input_path = Path(tmpdir) / "inputs" / "test.wav"
        
        # Create manifest as RUNNING
        mock_runner.manifest_path.write_text(json.dumps({
            "run_id": "test_run_123",
            "status": "RUNNING"
        }))
        
        def mock_run():
            pass  # No-op
        
        mock_runner.run = mock_run
        
        # Mock launch_run_worker to return success without actually launching
        mock_launch_result = {"worker_pid": 12345}
        
        with patch("server.api.workbench.SessionRunner", return_value=mock_runner), \
             patch("server.api.workbench.launch_run_worker", return_value=mock_launch_result):
            client = TestClient(app)
            
            wav_data = b"RIFF" + b"\x00" * 40
            
            response = client.post(
                "/api/workbench/runs",
                files={"file": ("test.wav", wav_data, "audio/wav")},
                data={"use_case_id": "test", "steps_preset": "ingest"}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert "run_id" in data
            assert "console_url" in data


def test_workbench_runs_invalid_preset_returns_400():
    """Verify invalid steps_preset returns 400."""
    from fastapi.testclient import TestClient
    from server.main import app
    
    with tempfile.TemporaryDirectory() as tmpdir:
        os.environ["MODEL_LAB_RUNS_ROOT"] = tmpdir
        os.environ["MODEL_LAB_INPUTS_ROOT"] = tmpdir
        
        # Make sure worker is free - mock try_acquire_worker to return True
        # and release_worker to be a no-op (patch at import location)
        with patch("server.api.workbench.try_acquire_worker", return_value=True), \
             patch("server.api.workbench.release_worker"):
            client = TestClient(app)
            
            wav_data = b"RIFF" + b"\x00" * 40
            
            response = client.post(
                "/api/workbench/runs",
                files={"file": ("test.wav", wav_data, "audio/wav")},
                data={"use_case_id": "test", "steps_preset": "invalid_preset"}
            )
            
            assert response.status_code == 400
