
"""
Hermetic tests for Experiment Compare V1 API.
"""
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

def test_compare_valid_transcript():
    """Verify comparing valid transcripts between two runs."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import server.main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        runs_root = tmpdir_path / "runs"
        exp_id = "exp_test_compare"
        run_a = "run_a"
        run_b = "run_b"
        
        # Setup experiment state
        exp_dir = runs_root / "experiments" / exp_id
        exp_dir.mkdir(parents=True)
        
        (exp_dir / "experiment_request.json").write_text(json.dumps({
            "experiment_id": exp_id,
        }))
        
        (exp_dir / "experiment_state.json").write_text(json.dumps({
            "experiment_id": exp_id,
            "runs": [
                {"run_id": run_a},
                {"run_id": run_b}
            ],
            "last_updated_at": "2024-01-01T00:00:00Z"
        }))
        
        # Setup runs with bundle artifacts
        (runs_root / run_a / "bundle").mkdir(parents=True)
        (runs_root / run_b / "bundle").mkdir(parents=True)
        
        (runs_root / run_a / "bundle" / "transcript.txt").write_text("Hello World A")
        (runs_root / run_b / "bundle" / "transcript.txt").write_text("Hello World B")
        
        client = TestClient(server.main.app)
        response = client.get(
            f"/api/experiments/{exp_id}/compare",
            params={"left": run_a, "right": run_b, "artifact": "transcript"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["left"]["available"] is True
        assert data["left"]["text"] == "Hello World A"
        assert data["right"]["available"] is True
        assert data["right"]["text"] == "Hello World B"


def test_compare_missing_artifact():
    """Verify behavior when one artifact is missing."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import server.main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        runs_root = tmpdir_path / "runs"
        exp_id = "exp_test_missing"
        run_a = "run_a"
        run_b = "run_b"
        
        # Setup experiment
        exp_dir = runs_root / "experiments" / exp_id
        exp_dir.mkdir(parents=True)
        (exp_dir / "experiment_request.json").write_text(json.dumps({
            "experiment_id": exp_id,
        }))
        (exp_dir / "experiment_state.json").write_text(json.dumps({
            "experiment_id": exp_id,
            "runs": [{"run_id": run_a}, {"run_id": run_b}],
            "last_updated_at": "2024-01-01T00:00:00Z"
        }))
        
        # A has file, B does not
        (runs_root / run_a / "bundle").mkdir(parents=True)
        (runs_root / run_a / "bundle" / "summary.md").write_text("# Summary A")
        
        client = TestClient(server.main.app)
        response = client.get(
            f"/api/experiments/{exp_id}/compare",
            params={"left": run_a, "right": run_b, "artifact": "summary"}
        )
        
        assert response.status_code == 200
        data = response.json()
        
        assert data["left"]["available"] is True
        assert data["right"]["available"] is False
        assert data["right"]["text"] is None


def test_compare_too_large():
    """Verify 413 when file exceeds max_bytes."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import server.main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        runs_root = tmpdir_path / "runs"
        exp_id = "exp_test_large"
        run_a = "run_a"
        run_b = "run_b"
        
        exp_dir = runs_root / "experiments" / exp_id
        exp_dir.mkdir(parents=True)
        (exp_dir / "experiment_request.json").write_text(json.dumps({
            "experiment_id": exp_id,
        }))
        (exp_dir / "experiment_state.json").write_text(json.dumps({
            "experiment_id": exp_id,
            "runs": [{"run_id": run_a}, {"run_id": run_b}],
            "last_updated_at": "2024-01-01T00:00:00Z"
        }))
        
        (runs_root / run_a / "bundle").mkdir(parents=True)
        (runs_root / run_b / "bundle").mkdir(parents=True)
        
        # Create large file (1000 bytes)
        large_content = "x" * 1000
        (runs_root / run_a / "bundle" / "transcript.txt").write_text(large_content)
        (runs_root / run_b / "bundle" / "transcript.txt").write_text("Small")
        
        client = TestClient(server.main.app)
        response = client.get(
            f"/api/experiments/{exp_id}/compare",
            params={
                "left": run_a, 
                "right": run_b, 
                "artifact": "transcript",
                "max_bytes": 500  # limit below file size
            }
        )
        
        assert response.status_code == 413
        data = response.json()
        
        assert data["error_code"] == "PREVIEW_TOO_LARGE"
        assert data["left"]["error"] == "PREVIEW_TOO_LARGE"
        assert data["left"]["truncated"] is True
        # Right should still be readable in the response object, but overall status is 413
        assert data["right"]["text"] == "Small"


def test_compare_runs_not_in_experiment():
    """Verify 400 when run_id does not belong to experiment."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    import server.main
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        os.environ["MODEL_LAB_RUNS_ROOT"] = str(tmpdir_path / "runs")
        
        runs_root = tmpdir_path / "runs"
        exp_id = "exp_test_bad_run"
        run_a = "run_a"
        
        exp_dir = runs_root / "experiments" / exp_id
        exp_dir.mkdir(parents=True)
        (exp_dir / "experiment_request.json").write_text(json.dumps({
            "experiment_id": exp_id,
        }))
        (exp_dir / "experiment_state.json").write_text(json.dumps({
            "experiment_id": exp_id,
            "runs": [{"run_id": run_a}],
            "last_updated_at": "2024-01-01T00:00:00Z"
        }))
        
        client = TestClient(server.main.app)
        response = client.get(
            f"/api/experiments/{exp_id}/compare",
            params={"left": run_a, "right": "run_rogue", "artifact": "transcript"}
        )
        
        assert response.status_code == 400
