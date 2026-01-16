"""
Contract tests for eval enrichment with real checks and findings.

Discipline:
- Marked as real E2E and runs only when RUN_REAL_E2E=1
- Reads base URL from MODEL_LAB_BASE_URL (default http://localhost:8000)
- Generates its own tiny WAV input (no hardcoded fixture paths)
"""

import os
import json
import tempfile
from pathlib import Path

import pytest

RUN_REAL_E2E = os.environ.get("RUN_REAL_E2E") == "1"
MODEL_LAB_BASE_URL = os.environ.get("MODEL_LAB_BASE_URL", "http://localhost:8000").rstrip("/")

pytestmark = pytest.mark.real_e2e


def _repo_root() -> Path:
    # tests/integration/test_eval_enrichment.py -> model-lab/
    return Path(__file__).resolve().parents[2]


def _write_tiny_wav(path: Path, *, sample_rate: int = 16000, duration_s: float = 1.0) -> None:
    import struct
    import wave

    nframes = int(sample_rate * duration_s)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(sample_rate)
        silence = struct.pack("<h", 0)
        wf.writeframes(silence * nframes)


def test_eval_writer_emits_real_checks_based_on_artifacts():
    """Verify eval.json contains real checks based on actual artifacts, not fake data."""
    if not RUN_REAL_E2E:
        pytest.skip("RUN_REAL_E2E not set")

    import requests
    import re
    import subprocess
    import sys

    repo_root = _repo_root()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        wav_path = tmpdir_path / "smoke.wav"
        _write_tiny_wav(wav_path)

        runs_root = os.environ.get("MODEL_LAB_RUNS_ROOT") or str(tmpdir_path / "runs")

        result = subprocess.run(
            [
                sys.executable,
                "scripts/run_session.py",
                "--input",
                str(wav_path),
                "--out-dir",
                str(runs_root),
                "--steps",
                "ingest",
            ],
            capture_output=True,
            text=True,
            cwd=str(repo_root),
            check=False,
        )

        match = re.search(r"^RUN_SESSION_RESULT=(\\{.*\\})\\s*$", result.stdout, flags=re.MULTILINE)
        assert match, f"Could not find run result in stdout.\nSTDERR:\n{result.stderr}"

        run_data = json.loads(match.group(1))
        run_id = run_data["run_id"]

        eval_response = requests.get(f"{MODEL_LAB_BASE_URL}/api/runs/{run_id}/eval", timeout=10)

    assert eval_response.status_code == 200
    eval_data = eval_response.json()

    assert eval_data["schema_version"] == "1"
    assert eval_data["run_id"] == run_id
    assert isinstance(eval_data["checks"], list)
    assert isinstance(eval_data["findings"], list)
    assert len(eval_data["checks"]) > 0

    for check in eval_data["checks"]:
        assert "name" in check
        assert "passed" in check
        assert isinstance(check["passed"], bool)
        assert "severity" in check
        assert check["severity"] in ["info", "warn", "fail"]
        assert "message" in check
        assert "evidence_paths" in check

    assert eval_data["metrics"] == {}


def test_results_includes_check_summary():
    """Verify /api/results includes check pass/fail counts."""
    if not RUN_REAL_E2E:
        pytest.skip("RUN_REAL_E2E not set")

    import requests

    response = requests.get(f"{MODEL_LAB_BASE_URL}/api/results", timeout=10)
    assert response.status_code == 200
    results = response.json()

    eval_results = [r for r in results if r.get("eval_available")]
    if not eval_results:
        pytest.skip("No runs with eval.json available")

    result = eval_results[0]
    assert "checks_total" in result
    assert "checks_passed" in result

    if result["checks_total"] is not None:
        assert isinstance(result["checks_total"], int)
        assert isinstance(result["checks_passed"], int)
        assert result["checks_passed"] <= result["checks_total"]


def test_findings_aggregate_counts_and_last_seen():
    """Verify /api/findings aggregates with count and timestamps."""
    if not RUN_REAL_E2E:
        pytest.skip("RUN_REAL_E2E not set")

    import requests

    response = requests.get(f"{MODEL_LAB_BASE_URL}/api/findings", timeout=10)
    assert response.status_code == 200
    findings = response.json()

    assert isinstance(findings, list)

    for finding in findings:
        assert "finding_id" in finding
        assert "title" in finding
        assert "category" in finding
        assert "severity" in finding
        assert "count" in finding
        assert "first_seen_at" in finding
        assert "last_seen_at" in finding
        assert "latest_run_id" in finding
        assert "evidence_paths" in finding
