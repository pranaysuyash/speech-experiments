"""
Hermetic contract tests for eval.json writer.

Tests the writer directly without running the pipeline.
"""
import json
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock
import pytest


def _create_mock_runner(session_dir: Path, manifest: dict):
    """Create a mock SessionRunner with just enough for _write_eval_json."""
    from harness.session import SessionRunner
    
    # Create a minimal runner mock
    runner = MagicMock(spec=SessionRunner)
    runner.session_dir = session_dir
    runner._write_eval_json = SessionRunner._write_eval_json.__get__(runner, SessionRunner)
    runner.EVAL_CHECK_NAMES = SessionRunner.EVAL_CHECK_NAMES
    return runner


def test_eval_written_to_run_root():
    """Verify eval.json is written to run_root, not bundle/."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        bundle_dir = run_root / "bundle"
        bundle_dir.mkdir()
        
        manifest = {"run_id": "test_123", "status": "COMPLETED", "config": {}}
        runner = _create_mock_runner(run_root, manifest)
        
        runner._write_eval_json(manifest)
        
        assert (run_root / "eval.json").exists(), "eval.json should be at run root"
        assert not (bundle_dir / "eval.json").exists(), "eval.json should NOT be in bundle/"


def test_eval_valid_json_schema():
    """Verify eval.json has all required schema fields."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        (run_root / "bundle").mkdir()
        
        manifest = {"run_id": "test_456", "status": "COMPLETED", "config": {"key": "val"}}
        runner = _create_mock_runner(run_root, manifest)
        
        runner._write_eval_json(manifest)
        
        eval_data = json.loads((run_root / "eval.json").read_text())
        
        # Required fields
        assert eval_data["schema_version"] == "1"
        assert eval_data["run_id"] == "test_456"
        assert "checks" in eval_data
        assert "findings" in eval_data
        assert eval_data["metrics"] == {}
        assert "generated_at" in eval_data


def test_eval_enriched_has_10_checks():
    """Verify enriched mode produces exactly 10 checks with fixed names."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        bundle_dir = run_root / "bundle"
        bundle_dir.mkdir()
        (run_root / "artifacts").mkdir()
        
        # Create bundle manifest
        (bundle_dir / "bundle_manifest.json").write_text('{"files": []}')
        
        manifest = {"run_id": "test_789", "status": "COMPLETED", "config": {}}
        runner = _create_mock_runner(run_root, manifest)
        
        runner._write_eval_json(manifest)
        
        eval_data = json.loads((run_root / "eval.json").read_text())
        checks = eval_data["checks"]
        
        assert len(checks) == 10, f"Expected 10 checks, got {len(checks)}"
        
        # Verify all check names are from the fixed set
        check_names = {c["name"] for c in checks}
        expected_names = set(runner.EVAL_CHECK_NAMES)
        assert check_names == expected_names, f"Check names mismatch: {check_names ^ expected_names}"


def test_eval_evidence_paths_relative_and_valid():
    """Verify evidence_paths are relative and reference existing files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        bundle_dir = run_root / "bundle"
        bundle_dir.mkdir()
        (run_root / "artifacts").mkdir()
        
        # Create some artifacts
        (bundle_dir / "bundle_manifest.json").write_text('{"files": []}')
        (bundle_dir / "transcript.txt").write_text("hello world")
        (run_root / "artifacts" / "asr.json").write_text('{}')
        (run_root / "manifest.json").write_text('{}')  # Create manifest for evidence path
        
        manifest = {"run_id": "test_evidence", "status": "COMPLETED", "config": {}}
        runner = _create_mock_runner(run_root, manifest)
        
        runner._write_eval_json(manifest)
        
        eval_data = json.loads((run_root / "eval.json").read_text())
        
        for check in eval_data["checks"]:
            for path in check.get("evidence_paths", []):
                # Must be relative
                assert not path.startswith("/"), f"Evidence path must be relative: {path}"
                # Must exist (only if passed or evidence explicitly included)
                if check["passed"] and path:
                    full_path = run_root / path
                    assert full_path.exists(), f"Evidence path does not exist: {path}"


def test_eval_failed_run_generates_finding():
    """Verify FAILED status generates a finding."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        (run_root / "bundle").mkdir()
        
        manifest = {
            "run_id": "test_failed",
            "status": "FAILED",
            "error_step": "asr",
            "error_code": "ValueError",
            "error_message": "Model loading failed",
            "config": {}
        }
        runner = _create_mock_runner(run_root, manifest)
        
        runner._write_eval_json(manifest)
        
        eval_data = json.loads((run_root / "eval.json").read_text())
        
        # Check run_terminal_status_ok failed
        status_check = next(c for c in eval_data["checks"] if c["name"] == "run_terminal_status_ok")
        assert not status_check["passed"]
        assert status_check["severity"] == "fail"
        
        # Verify finding generated
        assert len(eval_data["findings"]) >= 1
        finding = eval_data["findings"][0]
        assert "asr" in finding["title"]
        assert finding["severity"] == "high"


def test_eval_identity_mode_empty_checks():
    """Verify identity mode produces empty checks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        (run_root / "bundle").mkdir()
        
        manifest = {"run_id": "test_identity", "status": "COMPLETED", "config": {}}
        runner = _create_mock_runner(run_root, manifest)
        
        # Set identity mode
        os.environ["MODEL_LAB_EVAL_MODE"] = "identity"
        try:
            runner._write_eval_json(manifest)
        finally:
            os.environ.pop("MODEL_LAB_EVAL_MODE", None)
        
        eval_data = json.loads((run_root / "eval.json").read_text())
        
        assert eval_data["checks"] == []
        assert eval_data["findings"] == []
        assert eval_data["metrics"] == {}
