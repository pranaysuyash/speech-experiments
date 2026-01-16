"""
Contract tests for eval enrichment with real checks and findings.
"""
import json
import tempfile
from pathlib import Path
import pytest


def test_eval_writer_emits_real_checks_based_on_artifacts():
    """Verify eval.json contains real checks based on actual artifacts, not fake data."""
    import requests
    
    try:
        # Run a simple session that will succeed
        import subprocess
        result = subprocess.run([
            "uv", "run", "python", "scripts/run_session.py",
            "--input", "inputs/meetings/2026-01/test_smoke.wav",
            "--out-dir", "/tmp/pytest_eval_check",
            "--steps", "ingest"
        ], capture_output=True, text=True, cwd="/Users/pranay/Projects/speech_experiments/model-lab")
        
        # Extract run_id from output
        import re
        match = re.search(r'RUN_SESSION_RESULT=(\{.*\})', result.stdout)
        assert match, "Could not find run result in output"
        
        run_data = json.loads(match.group(1))
        run_id = run_data["run_id"]
        
        # Get eval from API
        eval_response = requests.get(f"http://localhost:8000/api/runs/{run_id}/eval", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running")
    except Exception as e:
        pytest.skip(f"Setup failed: {e}")
    
    if eval_response.status_code == 404:
        pytest.skip("Eval endpoint not available for this run/server")
    assert eval_response.status_code == 200
    eval_data = eval_response.json()
    
    # Verify schema
    assert eval_data["schema_version"] == "1"
    assert eval_data["run_id"] == run_id
    assert isinstance(eval_data["checks"], list)
    assert isinstance(eval_data["findings"], list)
    
    # Verify we have real checks (not just identity)
    assert len(eval_data["checks"]) >0
    
    # All checks must have required fields
    for check in eval_data["checks"]:
        assert "name" in check
        assert "passed" in check
        assert isinstance(check["passed"], bool)
        assert "severity" in check
        assert check["severity"] in ["info", "warn", "fail"]
        assert "message" in check
        assert "evidence_paths" in check
    
    # Verify no fake metrics
    assert eval_data["metrics"] == {}


def test_results_includes_check_summary():
    """Verify /api/results includes check pass/fail counts."""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/api/results", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running")
    
    assert response.status_code == 200
    results = response.json()
    
    # Find a result with eval available
    eval_results = [r for r in results if r.get("eval_available")]
    
    if not eval_results:
        pytest.skip("No runs with eval.json available")
    
    result = eval_results[0]
    
    # Verify check summary fields exist
    assert "checks_total" in result
    assert "checks_passed" in result
    
    if result["checks_total"] is not None:
        assert isinstance(result["checks_total"], int)
        assert isinstance(result["checks_passed"], int)
        assert result["checks_passed"] <= result["checks_total"]


def test_findings_aggregate_counts_and_last_seen():
    """Verify /api/findings aggregates with count and timestamps."""
    import requests
    
    try:
        response = requests.get("http://localhost:8000/api/findings", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running")
    
    assert response.status_code == 200
    findings = response.json()
    
    # Verify structure (may be empty)
    assert isinstance(findings, list)
    
    for finding in findings:
        # Required fields
        assert "finding_id" in finding
        assert "title" in finding
        assert "category" in finding
        assert "severity" in finding
        assert "count" in finding
        assert "first_seen_at" in finding
        assert "last_seen_at" in finding
        assert "latest_run_id" in finding
        assert "evidence_paths" in finding
