import json
import sys
import threading
import time
from pathlib import Path


def test_create_workbench_run_returns_quickly_and_writes_running_manifest(monkeypatch, tmp_path):
    # Ensure the API writes inputs/runs under temp dirs
    monkeypatch.setenv("MODEL_LAB_INPUTS_ROOT", str(tmp_path / "inputs"))
    monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(tmp_path / "runs"))

    # Fix imports for importlib mode
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    # Patch SessionRunner.run to avoid real processing while preserving the contract:
    # create run dir + RUNNING manifest, then block briefly so worker remains busy.
    import harness.session
    import server.services.lifecycle as lifecycle
    import server.api.workbench as workbench

    # Reset worker count to ensure test starts fresh
    lifecycle._ACTIVE_RUNS_COUNT = 0
    
    # Mock launch_run_worker to avoid subprocess spawning
    def mock_launch(runner, run_request_data, background=True):
        runner.session_dir.mkdir(parents=True, exist_ok=True)
        runner.manifest_path.write_text(json.dumps({
            "run_id": runner.run_id,
            "status": "RUNNING",
            "started_at": "now",
            "updated_at": "now",
        }))
        # Write run_request.json like the real function does
        request_path = runner.session_dir / "run_request.json"
        run_request_data["run_id"] = runner.run_id
        run_request_data["input_path"] = str(runner.input_path)
        request_path.write_text(json.dumps(run_request_data, indent=2))
        # Release worker since we're not actually launching a subprocess
        lifecycle.release_worker()
        return {"worker_pid": 12345}
    
    monkeypatch.setattr(workbench, "launch_run_worker", mock_launch)

    from fastapi.testclient import TestClient
    import server.main

    client = TestClient(server.main.app)

    t0 = time.monotonic()
    resp = client.post(
        "/api/workbench/runs",
        data={"use_case_id": "uc_smoke", "steps_preset": "ingest"},
        files={"file": ("input.wav", b"RIFFxxxxWAVEfmt ", "audio/wav")},
    )
    dt = time.monotonic() - t0

    assert resp.status_code == 200
    assert dt < 1.0  # contract is <=500ms; allow slack for CI

    payload = resp.json()
    assert "run_id" in payload
    assert "run_dir" in payload
    assert payload["console_url"].endswith(f"/runs/{payload['run_id']}")

    run_dir = Path(payload["run_dir"])
    manifest_path = run_dir / "manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "RUNNING"


def test_create_workbench_run_busy_returns_409(monkeypatch, tmp_path):
    """Test that a second request returns 409 when worker is busy.
    
    Uses direct mocking of try_acquire_worker to avoid race conditions.
    """
    monkeypatch.setenv("MODEL_LAB_INPUTS_ROOT", str(tmp_path / "inputs"))
    monkeypatch.setenv("MODEL_LAB_RUNS_ROOT", str(tmp_path / "runs"))

    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    from fastapi.testclient import TestClient
    import server.main
    import server.api.workbench as workbench

    client = TestClient(server.main.app)

    # Mock try_acquire_worker to return False (simulating busy state)
    # Patch at the import location in workbench module
    monkeypatch.setattr(workbench, "try_acquire_worker", lambda: False)

    # Request while worker is "busy" -> 409
    r2 = client.post(
        "/api/workbench/runs",
        data={"use_case_id": "uc_smoke", "steps_preset": "ingest"},
        files={"file": ("input.wav", b"RIFFxxxxWAVEfmt ", "audio/wav")},
    )
    assert r2.status_code == 409
    assert r2.json().get("error_code") == "RUNNER_BUSY"
