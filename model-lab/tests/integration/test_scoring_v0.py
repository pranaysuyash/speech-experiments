
import pytest
from pathlib import Path
import json
import shutil
import sys
import os

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from server.main import app
from harness.session import SessionRunner

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    # Mock environment
    import os
    os.environ["MODEL_LAB_RUNS_ROOT"] = str(root)
    yield root

def test_scoring_logic_direct(tmp_path):
    # Test _compute_proxy_scores directly via SessionRunner subclass or mocking
    
    # Create dummy session dir
    session_dir = tmp_path / "test_run"
    session_dir.mkdir()
    (session_dir / "bundle").mkdir()
    
    # Create artifacts
    (session_dir / "bundle" / "transcript.txt").write_text("Long enough transcript " * 10, encoding="utf-8")
    (session_dir / "bundle" / "summary.md").write_text("Summary content " * 5, encoding="utf-8")
    (session_dir / "bundle" / "action_items.csv").write_text("header,col\nval,val", encoding="utf-8")
    
    # Create dummy input file required by SessionRunner
    input_path = tmp_path / "input.wav"
    input_path.write_text("dummy audio")

    # We can mock SessionRunner
    runner = SessionRunner(input_path, tmp_path/"output", steps=[])
    runner.session_dir = session_dir
    runner.run_id = "test_run"
    
    # Test compute
    scores = runner._compute_proxy_scores()
    
    # Verify correctness
    assert len(scores) == 4
    
    completeness = next(s for s in scores if s["name"] == "artifact_completeness")
    assert completeness["score"] == 100
    
    transcript = next(s for s in scores if s["name"] == "transcript_length_ok")
    assert transcript["score"] == 100
    
    summary = next(s for s in scores if s["name"] == "summary_nonempty")
    assert summary["score"] == 100
    
    action = next(s for s in scores if s["name"] == "action_items_parseable")
    assert action["score"] == 100

def test_api_enrichment(client, run_dir):
    # Setup Experiment with 2 runs
    exp_id = "exp_scoring"
    exp_dir = run_dir / "experiments" / exp_id
    exp_dir.mkdir(parents=True)
    
    (exp_dir / "experiment_request.json").write_text(json.dumps({
        "schema_version": "2",
        "experiment_id": exp_id,
        "created_at": "2024-01-01T00:00:00Z",
        "use_case_id": "meeting_smoke",
        "source": {"filename_original": "test.wav", "bytes": 100, "sha256": "abc"},
        "candidates": [{"candidate_id": "A"}, {"candidate_id": "B"}]
    }))
    
    # State with runs
    run_id = "run_scoring_1"
    (exp_dir / "experiment_state.json").write_text(json.dumps({
        "schema_version": "1",
        "experiment_id": exp_id,
        "last_updated_at": "2024-01-01T00:00:00Z",
        "runs": [
            {"candidate_id": "A", "run_id": run_id, "status": "COMPLETED"},
            {"candidate_id": "B", "run_id": None, "status": "QUEUED"}
        ]
    }))
    
    # Create Run Dir with Eval
    run_root = run_dir / run_id
    run_root.mkdir()
    (run_root / "manifest.json").write_text(json.dumps({"run_id": run_id, "status": "COMPLETED"}))
    
    (run_root / "eval.json").write_text(json.dumps({
        "schema_version": "1",
        "run_id": run_id,
        "use_case_id": "meeting_smoke",
        "metrics": {},
        "checks": [],
        "findings": [],
        "generated_at": "2024-01-01T00:00:00Z",
        "score_cards": [
            {"name": "test_score", "label": "Test Score", "type": "proxy", "score": 85, "evidence_paths": []}
        ]
    }))
    
    # Call GET API
    resp = client.get(f"/api/experiments/{exp_id}")
    assert resp.status_code == 200
    data = resp.json()
    
    runs = data["runs"]
    run_a = next(r for r in runs if r["candidate_id"] == "A")
    assert "score_cards" in run_a
    assert len(run_a["score_cards"]) == 1
    assert run_a["score_cards"][0]["score"] == 85
