
import pytest
from pathlib import Path
import json
import sys
import os

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient
from server.main import app
from harness.session import SessionRunner
from server.utils.artifacts import resolve_artifact_relpaths

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def run_dir(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    os.environ["MODEL_LAB_RUNS_ROOT"] = str(root)
    yield root

def test_resolve_artifact_relpaths():
    # Test that we get the right priority order
    paths = resolve_artifact_relpaths("transcript")
    assert "bundle/transcript.json" == paths[0]
    assert "bundle/transcript.txt" == paths[1]
    
    paths_summary = resolve_artifact_relpaths("summary")
    assert "bundle/summary.md" == paths_summary[0]
    
def test_scoring_with_transcript_txt(tmp_path):
    """Test scoring works with legacy .txt format"""
    session_dir = tmp_path / "test_run"
    session_dir.mkdir()
    (session_dir / "bundle").mkdir()
    
    # Create .txt artifacts
    long_text = "Long enough transcript " * 10
    (session_dir / "bundle" / "transcript.txt").write_text(long_text, encoding="utf-8")
    (session_dir / "bundle" / "summary.md").write_text("Summary content " * 5, encoding="utf-8")
    (session_dir / "bundle" / "action_items.csv").write_text("header,col\nval,val", encoding="utf-8")
    
    input_path = tmp_path / "input.wav" 
    input_path.write_text("dummy audio")

    runner = SessionRunner(input_path, tmp_path/"output", steps=[])
    runner.session_dir = session_dir
    runner.run_id = "test_run"
    
    scores = runner._compute_proxy_scores()
    
    assert len(scores) == 4
    
    completeness = next(s for s in scores if s["name"] == "artifact_completeness")
    assert completeness["score"] == 100
    
    transcript = next(s for s in scores if s["name"] == "transcript_length_ok")
    assert transcript["score"] == 100
    assert "bundle/transcript.txt" in transcript["evidence_paths"][0]

def test_scoring_with_transcript_json(tmp_path):
    """Test scoring works with canonical .json format (Meeting Pack)"""
    session_dir = tmp_path / "test_run"
    session_dir.mkdir()
    (session_dir / "bundle").mkdir()
    
    # Create .json artifacts (Meeting Pack canonical format)
    transcript_data = {
        "segments": [
            {"text": "This is a long enough transcript with substantial content. ", "start_s": 0.0, "end_s": 1.0},
            {"text": "It has multiple segments that will be joined together. ", "start_s": 1.0, "end_s": 2.0},
            {"text": "Each segment contains text to ensure we exceed the threshold. ", "start_s": 2.0, "end_s": 3.0},
        ]
    }
    (session_dir / "bundle" / "transcript.json").write_text(json.dumps(transcript_data), encoding="utf-8")
    (session_dir / "bundle" / "summary.md").write_text("Summary content " * 5, encoding="utf-8")
    (session_dir / "bundle" / "action_items.csv").write_text("header,col\nval,val", encoding="utf-8")
    
    input_path = tmp_path / "input.wav"
    input_path.write_text("dummy audio")

    runner = SessionRunner(input_path, tmp_path/"output", steps=[])
    runner.session_dir = session_dir
    runner.run_id = "test_run"
    
    scores = runner._compute_proxy_scores()
    
    assert len(scores) == 4
    
    completeness = next(s for s in scores if s["name"] == "artifact_completeness")
    assert completeness["score"] == 100
    
    transcript = next(s for s in scores if s["name"] == "transcript_length_ok")
    assert transcript["score"] == 100
    assert "bundle/transcript.json" in transcript["evidence_paths"][0]

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
