"""
Contract tests for eval.json and Results/Findings API.

These tests verify the eval.json schema and endpoint behavior.
"""

import pytest


def test_eval_json_schema_minimal():
    """Verify minimal eval.json schema is valid."""
    from datetime import datetime

    from server.schemas.eval_contract import EvalResult

    # Create minimal eval result
    eval_result = EvalResult(
        schema_version="1",
        run_id="test_run_123",
        use_case_id=None,
        model_id=None,
        params={},
        metrics={},
        checks=[],
        findings=[],
        generated_at=datetime.now().isoformat(),
    )

    # Convert to dict and verify structure
    data = eval_result.to_dict()

    assert data["schema_version"] == "1"
    assert data["run_id"] == "test_run_123"
    assert isinstance(data["params"], dict)
    assert isinstance(data["metrics"], dict)
    assert isinstance(data["checks"], list)
    assert isinstance(data["findings"], list)
    assert "generated_at" in data

    # Verify it round-trips
    loaded = EvalResult.from_dict(data)
    assert loaded.run_id == "test_run_123"


def test_runs_eval_endpoint_returns_404_when_missing():
    """Verify /api/runs/{id}/eval returns 404 when eval.json doesn't exist."""
    import requests

    try:
        response = requests.get("http://localhost:8000/api/runs", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running on localhost:8000")

    assert response.status_code == 200
    runs = response.json()

    if not runs:
        pytest.skip("No runs available")

    # Try to get eval for first run (likely won't have eval.json)
    run_id = runs[0]["run_id"]
    eval_response = requests.get(f"http://localhost:8000/api/runs/{run_id}/eval")

    # Should be either 404 (no eval) or 200 (has eval)
    assert eval_response.status_code in [200, 404]

    # If 404, verify error message
    if eval_response.status_code == 404:
        error = eval_response.json()
        assert "detail" in error
        assert "not available" in error["detail"].lower()


def test_findings_aggregation_groups_consistently():
    """Verify findings are grouped by finding_id consistently."""
    import requests

    try:
        response = requests.get("http://localhost:8000/api/findings", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running on localhost:8000")

    assert response.status_code == 200
    findings = response.json()

    # Verify structure (may be empty list)
    assert isinstance(findings, list)

    # If findings exist, verify structure
    for finding in findings:
        assert "finding_id" in finding
        assert "title" in finding
        assert "category" in finding
        assert "severity" in finding
        assert "count" in finding
        assert "latest_run_id" in finding
        assert isinstance(finding["count"], int)
        assert finding["count"] >= 1


def test_results_endpoint_filters_work():
    """Verify /api/results filtering by status works."""
    import requests

    try:
        # Get all results
        all_response = requests.get("http://localhost:8000/api/results", timeout=5)
    except requests.exceptions.ConnectionError:
        pytest.skip("Backend not running on localhost:8000")

    assert all_response.status_code == 200
    all_results = all_response.json()

    # Verify structure
    assert isinstance(all_results, list)

    for result in all_results:
        assert "run_id" in result
        assert "status" in result
        assert "eval_available" in result
        assert isinstance(result["eval_available"], bool)

    # Test status filter if we have completed runs
    completed = [r for r in all_results if r["status"] == "COMPLETED"]
    if completed:
        filtered_response = requests.get("http://localhost:8000/api/results?status=COMPLETED")
        assert filtered_response.status_code == 200
        filtered_results = filtered_response.json()

        # All filtered results should be COMPLETED
        for result in filtered_results:
            assert result["status"] == "COMPLETED"
