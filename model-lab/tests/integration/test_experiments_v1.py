"""
Hermetic tests for Experiments V1 API.

All tests use temp directories, no uvicorn, no real pipeline runs.
"""
import json
import hashlib
import tempfile
import os
import wave
import struct
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest


def _create_tiny_wav(path: Path, duration_s: float = 0.1, sample_rate: int = 16000) -> bytes:
    """Create a minimal WAV file and return its bytes."""
    nframes = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        silence = struct.pack("<h", 0) * nframes
        wf.writeframes(silence)
    return path.read_bytes()


def test_presets_endpoint_returns_list():
    """Verify GET /api/workbench/presets returns list of presets."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from fastapi.testclient import TestClient
    import server.main
    
    client = TestClient(server.main.app)
    response = client.get("/api/workbench/presets")
    
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) >= 2
    assert all("steps_preset" in p for p in data)
    assert all("label" in p for p in data)


def test_create_experiment_400_if_less_than_two_presets():
    """Verify experiment creation fails if fewer than 2 presets."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from fastapi.testclient import TestClient
    import server.main
    import server.api.workbench as workbench
    import server.api.experiments as experiments
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)
        
        # Monkeypatch presets to only have one
        original_presets = workbench.PRESETS.copy()
        try:
            workbench.PRESETS.clear()
            workbench.PRESETS["ingest"] = {"label": "Ingest Only", "steps": ["ingest"]}
            
            # Reload experiments module to pick up patched PRESETS
            from importlib import reload
            reload(experiments)
            
            client = TestClient(server.main.app)
            response = client.post(
                "/api/experiments",
                files={"file": ("test.wav", wav_bytes, "audio/wav")},
                data={"use_case_id": "test"}
            )
            
            assert response.status_code == 400
            assert response.json()["error_code"] == "NEED_TWO_CANDIDATES"
        finally:
            workbench.PRESETS.clear()
            workbench.PRESETS.update(original_presets)
            reload(experiments)


def test_create_experiment_writes_request_and_state():
    """Verify experiment creation writes both JSON files with correct schema."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from fastapi.testclient import TestClient
    import server.main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)
        expected_sha = hashlib.sha256(wav_bytes).hexdigest()
        
        client = TestClient(server.main.app)
        response = client.post(
            "/api/experiments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"use_case_id": "test_case"}
        )
        
        assert response.status_code == 201
        data = response.json()
        
        exp_id = data["experiment_id"]
        exp_dir = tmpdir_path / "runs" / "experiments" / exp_id
        
        # Verify request file exists and has correct schema
        request_path = exp_dir / "experiment_request.json"
        assert request_path.exists()
        request = json.loads(request_path.read_text())
        
        assert request["schema_version"] == "2"
        assert request["experiment_id"] == exp_id
        assert request["use_case_id"] == "test_case"
        assert len(request["candidates"]) == 2
        assert request["source"]["sha256"] == expected_sha
        assert request["source"]["bytes"] == len(wav_bytes)
        
        # Verify state file exists
        state_path = exp_dir / "experiment_state.json"
        assert state_path.exists()
        state = json.loads(state_path.read_text())
        
        assert state["schema_version"] == "1"
        assert len(state["runs"]) == 2
        assert all(r["status"] == "QUEUED" for r in state["runs"])


def test_start_next_409_when_runner_busy():
    """Verify start returns 409 when runner is busy."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from fastapi.testclient import TestClient
    import server.main
    import server.api.experiments as experiments
    import server.api.workbench as workbench
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)
        
        client = TestClient(server.main.app)
        
        # Create experiment
        create_resp = client.post(
            "/api/experiments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"use_case_id": "busy_test"}
        )
        exp_id = create_resp.json()["experiment_id"]
        
        # Patch start_run_from_path to raise RunnerBusyError
        with patch.object(experiments, "start_run_from_path", side_effect=workbench.RunnerBusyError):
            start_resp = client.post(f"/api/experiments/{exp_id}/runs/start")
            
            assert start_resp.status_code == 409
            assert start_resp.json()["error_code"] == "RUNNER_BUSY"
        
        # Verify slot remains QUEUED
        state_path = tmpdir_path / "runs" / "experiments" / exp_id / "experiment_state.json"
        state = json.loads(state_path.read_text())
        assert state["runs"][0]["status"] == "QUEUED"


def test_start_next_sets_run_id_when_runner_free():
    """Verify start updates state with run_id when runner is free."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from fastapi.testclient import TestClient
    import server.main
    import server.api.experiments as experiments
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)
        
        client = TestClient(server.main.app)
        
        create_resp = client.post(
            "/api/experiments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"use_case_id": "free_test"}
        )
        exp_id = create_resp.json()["experiment_id"]
        
        mock_result = {
            "run_id": "mock_run_12345",
            "run_dir": "/tmp/mock_run",
            "console_url": "/runs/mock_run_12345"
        }
        
        with patch.object(experiments, "start_run_from_path", return_value=mock_result):
            start_resp = client.post(f"/api/experiments/{exp_id}/runs/start")
            
            assert start_resp.status_code == 200
            data = start_resp.json()
            assert data["started"] is True
            assert data["run_id"] == "mock_run_12345"
            assert data["candidate_id"] == "A"
        
        # Verify state updated
        state_path = tmpdir_path / "runs" / "experiments" / exp_id / "experiment_state.json"
        state = json.loads(state_path.read_text())
        assert state["runs"][0]["status"] == "RUNNING"
        assert state["runs"][0]["run_id"] == "mock_run_12345"
        assert state["runs"][0]["started_at"] is not None


def test_get_experiment_summary_shape():
    """Verify GET returns merged request + state."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from fastapi.testclient import TestClient
    import server.main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        wav_path = tmpdir_path / "test.wav"
        wav_bytes = _create_tiny_wav(wav_path)
        
        client = TestClient(server.main.app)
        
        create_resp = client.post(
            "/api/experiments",
            files={"file": ("test.wav", wav_bytes, "audio/wav")},
            data={"use_case_id": "summary_test"}
        )
        exp_id = create_resp.json()["experiment_id"]
        
        get_resp = client.get(f"/api/experiments/{exp_id}")
        
        assert get_resp.status_code == 200
        data = get_resp.json()
        
        # From request
        assert "experiment_id" in data
        assert "use_case_id" in data
        assert "source" in data
        assert "candidates" in data
        
        # From state
        assert "runs" in data
        assert "last_updated_at" in data
        assert len(data["runs"]) == 2
