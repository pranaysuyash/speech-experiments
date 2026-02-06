"""
Hermetic contract tests for eval.json loading and writing.

These are NOT real e2e tests - they use mocked run directories.
"""

import json
import tempfile
from pathlib import Path


def test_load_eval_prefers_run_root():
    """Verify load_eval prefers run_root/eval.json over bundle/eval.json."""
    from server.api.eval_loader import load_eval

    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        bundle_dir = run_root / "bundle"
        bundle_dir.mkdir()

        # Create both locations with different data
        root_eval = {"location": "root", "checks": []}
        bundle_eval = {"location": "bundle", "checks": []}

        (run_root / "eval.json").write_text(json.dumps(root_eval))
        (bundle_dir / "eval.json").write_text(json.dumps(bundle_eval))

        result = load_eval(run_root)
        assert result is not None
        assert result["location"] == "root", "Should prefer run root over bundle"


def test_load_eval_falls_back_to_bundle():
    """Verify load_eval falls back to bundle when root is malformed."""
    from server.api.eval_loader import load_eval

    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        bundle_dir = run_root / "bundle"
        bundle_dir.mkdir()

        # Root has malformed JSON
        (run_root / "eval.json").write_text("{invalid json")

        # Bundle has valid JSON
        bundle_eval = {"location": "bundle", "checks": []}
        (bundle_dir / "eval.json").write_text(json.dumps(bundle_eval))

        result = load_eval(run_root)
        assert result is not None
        assert result["location"] == "bundle", "Should fall back to bundle on malformed root"


def test_load_eval_returns_none_when_missing():
    """Verify load_eval returns None when no eval.json exists."""
    from server.api.eval_loader import load_eval

    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        result = load_eval(run_root)
        assert result is None, "Should return None when no eval.json exists"


def test_load_eval_returns_none_when_all_malformed():
    """Verify load_eval returns None when all locations have malformed JSON."""
    from server.api.eval_loader import load_eval

    with tempfile.TemporaryDirectory() as tmpdir:
        run_root = Path(tmpdir)
        bundle_dir = run_root / "bundle"
        bundle_dir.mkdir()

        (run_root / "eval.json").write_text("{bad json")
        (bundle_dir / "eval.json").write_text("{also bad")

        result = load_eval(run_root)
        assert result is None, "Should return None when all evals are malformed"
